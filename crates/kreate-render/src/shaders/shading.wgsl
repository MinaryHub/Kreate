// A point, its material, and every light in the scene.
//
// The tail of the composed shader: `brdf.wgsl` and `lights.wgsl` are
// concatenated ahead of this, so `brdf_evaluate`, `illuminance_at`, the `dfg`
// table and the `lights` array are all already in scope. See
// `kreate_render::shading::shader_source`.
//
// A mirror of `kreate_render::shading`, held to it by a device test. The loop
// is small and its decisions do not show in a picture, which is precisely why
// they are worth a test: lights add rather than compete, a light can skip a
// lobe, and nothing is clamped because the display transform owns the range.

/// How many lights are live, and what to do with them.
///
/// A count rather than the buffer's length, because the buffer is grown and
/// never shrunk — the slots past the count hold an earlier frame's lights, and
/// a pass looping to the array's length would light the scene from lamps the
/// user deleted.
struct Shading {
    /// How many tiles across the target is. See `kreate_render::tiles`.
    tiles_across: u32,
    /// How many light indices one tile's list can hold.
    tile_capacity: u32,
    _pad0: u32,
    _pad1: u32,
    /// rgb: the world's radiance, linear. w: nonzero when an environment cube
    /// is bound and this colour is not what lights the scene.
    ///
    /// One colour in every direction, which is what §2.2's flat background is.
    /// It is not a special case of the image path but the *same* one: a uniform
    /// environment irradiates every direction with `πL` and prefilters to `L`
    /// at every roughness, so `ambient_from` takes both as arguments and this
    /// supplies them directly.
    world: vec4<f32>,
    /// x: `cos` of the world's rotation about up, y: `sin` of it.
    /// z: the roughest mip of the environment cube. w: how much light it gives.
    ///
    /// The rotation is applied to the *sampling direction* rather than baked
    /// into the cube, so that dragging the slider costs a rotation per fragment
    /// instead of rebuilding six faces and their mips.
    environment: vec4<f32>,
    /// Irradiance coefficients, with the cosine-lobe factors already folded in.
    /// See `kreate_render::environment::irradiance_sh9`.
    sh9: array<vec4<f32>, 9>,
    /// Turns a clip position back into a world direction, for the background.
    ///
    /// Here rather than in a buffer of the background's own, because it changes
    /// with the camera and so does everything else in this uniform — two
    /// buffers to keep in step with one camera is one too many.
    inv_view_proj: mat4x4<f32>,
}

@group(2) @binding(2) var<uniform> shading: Shading;

/// One entry per light, parallel to the light array.
///
/// Six frames, because a point light sees in every direction and one frustum
/// cannot. Everything else uses face zero — which for it is not a direction but
/// the slot the sampler looks in.
///
/// `layers` says which layer of the depth array each face landed in, or -1 for
/// a face with nothing in it, and `texels` how much world one texel of that
/// face covers. `params` is z: the depth bias, w: nonzero if this light casts.
/// See `kreate_render::shadow::GpuShadow`.
struct ShadowInfo {
    view_proj: array<mat4x4<f32>, 6>,
    layers: array<i32, 8>,
    /// One texel in metres per unit clip `w`, per face.
    texels: array<f32, 8>,
    /// Texels of penumbra per unit of stored-depth difference, per face.
    /// Zero for a source with no size. See `kreate_render::shadow::GpuShadow`.
    penumbra: array<f32, 8>,
    params: vec4<f32>,
}

/// Which face of a cube a direction points through.
///
/// The largest component decides, which is what a cube map means. The order is
/// `+X -X +Y -Y +Z -Z`, matching `kreate_render::shadow::point_view` — the one
/// thing this file and that one have to agree about, and the reason the frames
/// themselves are stored rather than rebuilt here.
fn cube_face(d: vec3<f32>) -> i32 {
    let a = abs(d);
    if a.x >= a.y && a.x >= a.z {
        return select(1, 0, d.x > 0.0);
    }
    if a.y >= a.z {
        return select(3, 2, d.y > 0.0);
    }
    return select(5, 4, d.z > 0.0);
}

/// How far off the surface to move a lookup, in shadow texels.
///
/// Enough to clear the texel the surface itself wrote, with room for the 3×3
/// filter's neighbours. Larger detaches a shadow from thin geometry — a leaf
/// or a sheet thinner than this stops shadowing itself, which is the price and
/// is the right way round: an unshadowed crease is a picture, and acne is a
/// fault.
const NORMAL_OFFSET_TEXELS: f32 = 2.0;

/// How far out to look for something casting, in texels.
///
/// The blocker search has to cover the widest penumbra it could produce, or a
/// shadow softer than this is found only near its own edge and hardens in the
/// middle. It also cannot be unbounded: every texel of it is a tap.
///
/// Six is about a metre of penumbra on a sun's near cascade, which is a sharp
/// enough limit that nothing in a viewport reaches it and a cheap enough one
/// that the search is sixteen taps rather than a hundred.
const BLOCKER_SEARCH_TEXELS: f32 = 6.0;

/// The widest the filter will spread, in texels.
///
/// The taps are a fixed grid, so spreading them further does not soften the
/// shadow — it separates them, and the shadow bands. This is where that starts,
/// and a penumbra that wants to be wider is drawn at this width instead: a
/// shadow slightly too sharp reads as a shadow, and a banded one reads as a
/// fault.
const MAX_PENUMBRA_TEXELS: f32 = 8.0;

/// How much of the map is worth searching before giving up on softness.
///
/// A hard source needs none of this: the search finds nothing to average and
/// the filter is the three-by-three it always was, at the cost it always had.
/// Only a light with a size pays for the rest.
fn average_blocker(
    layer: i32,
    centre: vec2<i32>,
    size: vec2<f32>,
    depth: f32,
) -> f32 {
    var total = 0.0;
    var found = 0.0;
    let step = max(i32(BLOCKER_SEARCH_TEXELS / 1.5), 1);
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let at = clamp(
                centre + vec2<i32>(dx * step, dy * step),
                vec2<i32>(0, 0),
                vec2<i32>(size) - vec2<i32>(1, 1),
            );
            let stored = textureLoad(shadow_maps, at, layer, 0);
            // Reversed-Z: a blocker is *nearer the light*, so its depth is the
            // greater one.
            if stored > depth {
                total = total + stored;
                found = found + 1.0;
            }
        }
    }
    return select(-1.0, total / found, found > 0.0);
}

/// Per tile: a count, then that many light indices.
///
/// The shading loop walks this rather than every light, which is the whole of
/// what the cull pass buys. See `kreate_render::tiles`.
@group(2) @binding(5) var<storage, read> tile_lights: array<u32>;

/// The prefiltered environment. Mip `m` is the roughness `m / roughest`.
///
/// Always bound, even when there is no environment — a bind group cannot have
/// a hole in it, so "no environment" is a one-texel black cube and the flag in
/// `shading.world.w` is what says not to read it.
@group(2) @binding(6) var environment_cube: texture_cube<f32>;
@group(2) @binding(7) var environment_sampler: sampler;

@group(2) @binding(3) var<storage, read> shadows: array<ShadowInfo>;
@group(2) @binding(4) var shadow_maps: texture_depth_2d_array;

/// How much of a light reaches `position`, from zero to one.
///
/// # No comparison sampler
///
/// `textureLoad` and a hand-written filter, not `textureSampleCompare`. Two
/// reasons, and the second is the one that decided it: a comparison sampler
/// puts the filtering in fixed-function hardware whose rounding a parity test
/// cannot see, and the same code then runs unchanged in a compute pass — which
/// is what §2.5's path tracer will want, and `textureSampleCompare` is not
/// available there.
///
/// # The acne fix is two halves, and the larger one is lateral
///
/// A surface lit at a glancing angle spans more depth across one shadow texel
/// than one lit head-on, so the depth bias is scaled by the tangent of the
/// angle. That alone is not enough and assuming it was is a bug this file
/// shipped with: a depth bias is in clip units, and how many metres one unit
/// of depth covers depends on how large a scene the frustum was fitted to.
///
/// So the sample point is first moved off the surface along its normal, by a
/// multiple of one shadow texel's width *in metres*. That is a lateral move
/// rather than a move in depth, which is why it costs almost nothing at a
/// contact: the offset slides the lookup to a neighbouring texel instead of
/// pushing the comparison toward the light, so a shadow stays attached to what
/// casts it. It is measured in texels, so it scales with the scene by
/// construction — which is exactly what the depth bias cannot do.
fn shadow_factor(index: u32, position: vec3<f32>, normal: vec3<f32>, to_light: vec3<f32>) -> f32 {
    let info = shadows[index];
    // A light with no frame this frame. Redundant with the empty faces such a
    // light carries — every layer is -1 and the test below catches it — and
    // kept because the intent should be stated rather than implied. See
    // `GpuShadow::NONE`.
    if info.params.w == 0.0 {
        return 1.0;
    }

    // Which of the light's frames covers this fragment.
    //
    // Three kinds of answer, and the slot is the same slot: a point light's
    // frames are the faces of a cube and the direction picks one; a parallel
    // light's are slices of the view and the *first that contains the point* is
    // the tightest, because the slices are nested; and everything else has one
    // frame in slot zero.
    //
    // Searching rather than comparing against stored split distances. A split
    // is a number this file and the one that fitted the frames would both have
    // to agree about, and asking the frame itself cannot disagree with it.
    let light = lights[index];
    var face = 0;
    if is_cube(light) {
        face = cube_face(-to_light);
        if info.layers[face] < 0 {
            // A face this light was not given — either nothing is in it, or
            // the frame budget ran out before it. Lit, for the same reason as
            // outside a frustum: the map says nothing about it.
            return 1.0;
        }
    } else if light.falloff == FALLOFF_PARALLEL {
        var found = -1;
        for (var i = 0; i < 6; i = i + 1) {
            if info.layers[i] < 0 {
                continue;
            }
            let c = info.view_proj[i] * vec4<f32>(position, 1.0);
            let n = c.xyz / c.w;
            if c.w > 0.0 && abs(n.x) <= 1.0 && abs(n.y) <= 1.0 && n.z > 0.0 {
                found = i;
                break;
            }
        }
        if found < 0 {
            // Past the last cascade, which is where a sun stops shadowing.
            // Lit, not black: the far distance is where a shadow is a few
            // pixels wide and a hard edge across the picture would be the
            // loudest thing in it.
            //
            // No test reaches this line, and that is recorded rather than
            // worked around. A viewport camera looks *down* at what it is
            // framing, so the ground it shows runs out well before the three
            // hundred metres the cascades cover — even over a floor a mile
            // wide. Turning this into `0.0` changes no pixel of any fixture
            // here. It becomes reachable the first time somebody puts the
            // camera near the ground and looks along it, which is a framing
            // worth a fixture of its own rather than a contortion of these.
            return 1.0;
        }
        face = found;
    } else if info.layers[0] < 0 {
        return 1.0;
    }
    let layer = info.layers[face];
    let frame = info.view_proj[face];
    // One texel in metres, here: for an orthographic frustum `w` is one and
    // the size is constant; for a perspective one `w` is the distance along the
    // light's axis and a texel grows with it.
    let first = frame * vec4<f32>(position, 1.0);
    let texel_world = info.texels[face] * max(first.w, 1.0);
    // More offset at a glancing angle, where one texel spans more surface.
    let incidence = clamp(dot(normal, to_light), 1e-3, 1.0);
    let lateral = sqrt(1.0 - incidence * incidence);
    let offset = texel_world * (1.0 + 2.0 * lateral) * NORMAL_OFFSET_TEXELS;
    let clip = frame * vec4<f32>(position + normal * offset, 1.0);
    if clip.w <= 0.0 {
        // Behind the light's near plane: the map says nothing about it, and
        // claiming shadow there would put a dark band behind every spot.
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z <= 0.0 {
        // Outside what the light was framed for. Lit rather than shadowed,
        // because the alternative is a hard black edge at the frustum's border
        // — which reads as a bug and is one.
        return 1.0;
    }

    // Clip to texture coordinates: y flips, because clip space is up and a
    // texture is down.
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let size = vec2<f32>(textureDimensions(shadow_maps));
    let texel = vec2<i32>(uv * size);

    // Slope-scaled, and clamped: at ninety degrees the tangent is infinite and
    // the bias would push the comparison past the far plane, unshadowing the
    // whole surface.
    let slope = min(lateral / incidence, 8.0);
    let bias = info.params.z * (1.0 + slope);

    // How wide to filter. A source with size casts a penumbra that grows with
    // how far the thing casting it is from the thing it falls on — which is why
    // a box sitting on a floor has a sharp shadow at its base and a soft one
    // further out, with no separate rule for either.
    //
    // One texel is the floor and not zero: below that the filter is narrower
    // than the map can express, and a shadow cannot be sharper than its own
    // resolution however small the source is.
    var radius = 1.0;
    let scale = info.penumbra[face];
    if scale > 0.0 {
        let blocker = average_blocker(layer, texel, size, ndc.z + bias);
        if blocker < 0.0 {
            // Nothing between this point and the light, anywhere the search
            // looked. Fully lit, and the filter below is skipped entirely —
            // which is most of the picture and most of the saving.
            return 1.0;
        }
        radius = clamp(scale * (blocker - ndc.z), 1.0, MAX_PENUMBRA_TEXELS);
    }

    // A fixed grid spread over that radius. Five by five rather than three,
    // because the taps are what the softness is made of: nine of them across
    // eight texels is a shadow with steps in it.
    var lit = 0.0;
    let spacing = radius * 0.5;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let at = clamp(
                texel + vec2<i32>(vec2<f32>(f32(dx), f32(dy)) * spacing),
                vec2<i32>(0, 0),
                vec2<i32>(size) - vec2<i32>(1, 1),
            );
            let stored = textureLoad(shadow_maps, at, layer, 0);
            // Reversed-Z: nearer the light is *greater*. This fragment is lit
            // if nothing was closer.
            lit += select(0.0, 1.0, ndc.z + bias >= stored);
        }
    }
    return lit / 25.0;
}

/// Which way the light is, from `at`.
///
/// A parallel light has only a direction, and it is the direction it *points*,
/// so the way to it is the other way.
fn direction_to(light: Light, at: vec3<f32>) -> vec3<f32> {
    let axis = light.direction.xyz;
    if light.falloff == FALLOFF_PARALLEL {
        return -axis;
    }
    let offset = light.position.xyz - at;
    // A point exactly on the light has no direction to be lit from; its
    // illuminance is zero anyway, so any unit vector does.
    if dot(offset, offset) <= 0.0 {
        return -axis;
    }
    return normalize(offset);
}

/// What a uniform environment of `radiance` contributes.
///
/// Exact rather than approximate, which is worth saying because ambient terms
/// usually are not. Under an environment that is the same in every direction,
/// the prefiltered radiance is that colour whatever the roughness, so the
/// split-sum specular is `radiance × (F0·a + b)` — the DFG table straight out,
/// with no cube map. And the diffuse half integrates to `albedo × radiance`
/// against the cosine, coupled by the same share the direct path uses.
///
/// Which makes the whole thing conserve by construction: a white surface under
/// an environment of `L` returns `L × (share + 1 - share)`, and a furnace test
/// says so.
fn ambient_from(
    s: Surface,
    nov: f32,
    irradiance: vec3<f32>,
    prefiltered: vec3<f32>,
) -> vec3<f32> {
    let roughness = clamp(s.roughness, 0.0, 1.0);
    let share = specular_share(surface_f0(s), nov, roughness);
    let diffuse = surface_diffuse_albedo(s) * (vec3<f32>(1.0) - share);
    // Irradiance over `π` is the radiance a Lambertian surface reflects, which
    // is the one place the factor belongs: the coefficients carry it in for the
    // lookup and it comes back out here.
    return prefiltered * share + diffuse * irradiance / PI_SHADING;
}

const PI_SHADING: f32 = 3.14159265359;

/// The flat case, spelled as what it is.
fn ambient(s: Surface, normal: vec3<f32>, view: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    let nov = clamp(dot(normal, view), 1e-4, 1.0);
    return ambient_from(s, nov, radiance * PI_SHADING, radiance);
}

/// The world's rotation, applied to a direction on its way to the cube.
///
/// About up, which is what a world rotation is: turning a sky sideways is
/// framing and turning it end over end is nothing anybody wants.
fn unrotate(dir: vec3<f32>) -> vec3<f32> {
    let c = shading.environment.x;
    let s = shading.environment.y;
    return vec3<f32>(dir.x * c + dir.y * s, -dir.x * s + dir.y * c, dir.z);
}

/// The irradiance the environment sends to a surface facing `normal`.
///
/// Nine numbers and a dot product. Clamped at zero because a truncated harmonic
/// series undershoots around a small bright source, and a negative irradiance
/// darkens a surface the environment is *lighting*.
fn environment_irradiance(normal: vec3<f32>) -> vec3<f32> {
    let d = unrotate(normal);
    var total = shading.sh9[0].rgb * 0.2820948;
    total += shading.sh9[1].rgb * (0.4886025 * d.y);
    total += shading.sh9[2].rgb * (0.4886025 * d.z);
    total += shading.sh9[3].rgb * (0.4886025 * d.x);
    total += shading.sh9[4].rgb * (1.0925484 * d.x * d.y);
    total += shading.sh9[5].rgb * (1.0925484 * d.y * d.z);
    total += shading.sh9[6].rgb * (0.3153916 * (3.0 * d.z * d.z - 1.0));
    total += shading.sh9[7].rgb * (1.0925484 * d.x * d.z);
    total += shading.sh9[8].rgb * (0.5462742 * (d.x * d.x - d.y * d.y));
    return max(total, vec3<f32>(0.0));
}

/// What the environment contributes at a point.
///
/// One branch, and it is on whether an environment is bound rather than on
/// which formula to use: both arms hand the same function an irradiance and a
/// prefiltered radiance. A uniform environment makes the two arms agree
/// exactly, which is what the device test asserts.
fn environment_ambient(s: Surface, normal: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let nov = clamp(dot(normal, view), 1e-4, 1.0);
    if shading.world.w <= 0.0 {
        return ambient_from(s, nov, shading.world.rgb * PI_SHADING, shading.world.rgb);
    }
    let roughness = clamp(s.roughness, 0.0, 1.0);
    // The mirror direction, which is what a reflection is about. The prefilter
    // assumed the normal and the view agreed, so this is where that assumption
    // is paid for: head on it is exact, and at a grazing angle the highlight
    // does not stretch.
    let reflection = reflect(-view, normal);
    let level = roughness * shading.environment.z;
    // The strength multiplies both halves, which is what makes it a dimmer on
    // the environment rather than on one lobe of it.
    let strength = shading.environment.w;
    let prefiltered = textureSampleLevel(
        environment_cube,
        environment_sampler,
        unrotate(reflection),
        level,
    ).rgb * strength;
    return ambient_from(s, nov, environment_irradiance(normal) * strength, prefiltered);
}

/// The light leaving `position` toward `view`, in the working space.
///
/// Scene-linear and unbounded. Nothing is clamped: a surface under a sun is in
/// the thousands, and clamping before the display transform is the mistake
/// §2.1 records twice.
/// Which tile a pixel is in.
///
/// Taken from the fragment's own position rather than passed down, because
/// every caller would otherwise compute it the same way and one of them would
/// eventually compute it differently.
/// A sweep that flips the x or changes the tile size is caught; one that
/// returns zero for every pixel is not, because in the fixture that reaches
/// this the corner tile happens to hold both lamps. Recorded rather than
/// contorted around — the arithmetic here is what the other two exercise, and a
/// scene built so that the corner excludes a lamp is a fixture of its own.
fn tile_of(pixel: vec2<u32>) -> u32 {
    let across = max(shading.tiles_across, 1u);
    let x = min(pixel.x / TILE_SIZE, across - 1u);
    let y = pixel.y / TILE_SIZE;
    return y * across + x;
}

/// The side of a tile in pixels. See `kreate_render::tiles::TILE_SIZE`.
const TILE_SIZE: u32 = 16u;

fn shade_point(
    s: Surface,
    position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    pixel: vec2<u32>,
) -> vec3<f32> {
    // The world first, because it is the light that is always there.
    var total = environment_ambient(s, normal, view);

    // This tile's short list rather than every light in the scene. A tile holds
    // a count and then that many indices; a light not in the list does not
    // reach this square of the screen, which is a claim about its range and the
    // reason the range had to exist before this pass could.
    let base = tile_of(pixel) * (shading.tile_capacity + 1u);
    let here = min(tile_lights[base], shading.tile_capacity);
    for (var slot = 0u; slot < here; slot = slot + 1u) {
        let i = tile_lights[base + 1u + slot];
        let light = lights[i];
        let to_light = direction_to(light, position);
        var illuminance = illuminance_at(light, position);
        if casts_shadow(light) {
            illuminance *= shadow_factor(i, position, normal, to_light);
        }
        total += brdf_shade(
            s,
            normal,
            view,
            to_light,
            illuminance,
            lights_diffuse(light),
            lights_specular(light),
        );
    }
    return total;
}

// --- the probe pass --------------------------------------------------------
//
// For the device test. It shades a list of points against the whole light
// buffer, which is the loop a fragment shader runs — so what is compared is
// the sum and the per-light switches, not one evaluation.

struct Sample {
    /// rgb: base colour. w: metallic.
    material: vec4<f32>,
    /// x: roughness.
    params: vec4<f32>,
    position: vec4<f32>,
    normal: vec4<f32>,
    view: vec4<f32>,
}

struct Probe {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> samples: array<Sample>;
@group(0) @binding(1) var<storage, read_write> answers: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> probe: Probe;

@compute @workgroup_size(64)
fn probe_shading(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= probe.count {
        return;
    }
    let sample = samples[i];
    var s: Surface;
    s.base_color = sample.material.rgb;
    s.metallic = sample.material.w;
    s.roughness = sample.params.x;

    // Tile zero, and the fixture puts every light in it. The probe is about
    // the shading arithmetic and not about which lights reach which square of a
    // screen it does not have — but it goes through the same `shade_point`,
    // because a probe that took a different path would stop being evidence
    // about the one the picture uses.
    let lit = shade_point(
        s,
        sample.position.xyz,
        sample.normal.xyz,
        sample.view.xyz,
        vec2<u32>(0u, 0u),
    );
    answers[i] = vec4<f32>(lit, 0.0);
}
