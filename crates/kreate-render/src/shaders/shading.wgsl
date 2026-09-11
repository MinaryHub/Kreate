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
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    /// rgb: the world's radiance, linear. w: unused.
    ///
    /// One colour in every direction, which is what §2.2's flat background is.
    /// An image background is the IBL step and needs a prefiltered cube map;
    /// this is the half that costs a lookup.
    world: vec4<f32>,
}

@group(2) @binding(2) var<uniform> shading: Shading;

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
fn ambient(s: Surface, normal: vec3<f32>, view: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    let nov = clamp(dot(normal, view), 1e-4, 1.0);
    let roughness = clamp(s.roughness, 0.0, 1.0);
    let share = specular_share(surface_f0(s), nov, roughness);
    let diffuse = surface_diffuse_albedo(s) * (vec3<f32>(1.0) - share);
    return radiance * (share + diffuse);
}

/// The light leaving `position` toward `view`, in the working space.
///
/// Scene-linear and unbounded. Nothing is clamped: a surface under a sun is in
/// the thousands, and clamping before the display transform is the mistake
/// §2.1 records twice.
fn shade_point(
    s: Surface,
    position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
) -> vec3<f32> {
    // The world first, because it is the light that is always there.
    var total = ambient(s, normal, view, shading.world.rgb);
    for (var i = 0u; i < shading.light_count; i = i + 1u) {
        let light = lights[i];
        let to_light = direction_to(light, position);
        let illuminance = illuminance_at(light, position);
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

    let lit = shade_point(
        s,
        sample.position.xyz,
        sample.normal.xyz,
        sample.view.xyz,
    );
    answers[i] = vec4<f32>(lit, 0.0);
}
