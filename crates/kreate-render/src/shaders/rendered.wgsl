// The rendered viewport: scene lights, one BRDF, linear output.
//
// The tail of a composition. `brdf.wgsl`, `lights.wgsl` and `shading.wgsl`'s
// library half are concatenated ahead of this, so `Surface`, `shade_point` and
// the group-2 buffers are already in scope. See
// `kreate_render::shading::shader_source`.
//
// # What this adds and what it does not
//
// It adds geometry: a vertex stage and an interpolated position and normal. It
// does not add any shading — every light and every lobe is `shade_point`, which
// the CPU has its own copy of and a device test holds the two together.
//
// And it does not decide what a surface is made of. `material_sample` is
// appended after this, generated from the object's shading graph by
// `kreate-shader`, so a material is compiled once and this pass calls it. §3's
// fifth decision: one graph, one compiler, and no hand-written shading mode.
//
// # The output is linear and unbounded
//
// This writes into `Rgba16Float` and the display transform is a later pass. A
// surface under a sun is in the thousands here, and that is correct: clamping
// in a fragment shader is the mistake §2.1 records twice, one pass earlier
// than the target that would have clipped it.
//
// # Studio shading is not replaced
//
// `solid.wgsl` keeps its three camera-relative lights. While modelling, light
// that shows the *shape* is the right light, and a scene lit for a render is
// often a scene you cannot model in. This is the other mode, not the new one.

struct Globals {
    view_proj: mat4x4<f32>,
    /// xyz: unused here. w: one when the view is orthographic.
    params: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    /// The camera's back, which is the way to the eye from every point of an
    /// orthographic view.
    camera_back: vec4<f32>,
}

struct RenderedObject {
    /// Eye-relative model matrix. See `camera::model_matrix_relative`.
    model: mat4x4<f32>,
    /// The inverse transpose of the linear part, by column, padded.
    normal_col0: vec4<f32>,
    normal_col1: vec4<f32>,
    normal_col2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: RenderedObject;
/// The solids as they were shaded, before anything that passes light was drawn
/// over them: what is seen through glass. See `Viewport::encode_lit_scene`.
@group(2) @binding(9) var behind: texture_2d<f32>;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    /// Eye-relative, in metres: the same space the lights are packed into.
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    /// The surface coordinate, for the material.
    @location(2) uv: vec2<f32>,
    /// The point and its normal in the object's own space, for a procedural
    /// texture: one that moves with the object rather than the camera.
    @location(3) local: vec3<f32>,
    @location(4) local_normal: vec3<f32>,
    /// The vertex colour, which multiplies the base colour as glTF's
    /// `COLOR_0` does: white where the mesh has none.
    @location(5) tint: vec4<f32>,
}

@vertex
fn vs_rendered(
    @location(0) position: vec3<f32>,
    @location(1) packed_normal: u32,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
) -> VertexOut {
    let world = object.model * vec4<f32>(position, 1.0);
    var out: VertexOut;
    out.clip = globals.view_proj * world;
    out.world = world.xyz;

    out.uv = uv;
    out.local = position;
    out.tint = color;

    let n = oct_decode(packed_normal);
    out.local_normal = n;
    // The inverse transpose, because a non-uniform scale tilts a normal
    // transformed by the model matrix — which reads as broken shading rather
    // than as a transform bug.
    out.normal = vec3<f32>(
        dot(vec3<f32>(object.normal_col0.x, object.normal_col1.x, object.normal_col2.x), n),
        dot(vec3<f32>(object.normal_col0.y, object.normal_col1.y, object.normal_col2.y), n),
        dot(vec3<f32>(object.normal_col0.z, object.normal_col1.z, object.normal_col2.z), n),
    );
    return out;
}

// Copied from `solid.wgsl`, character for character, and a test in
// `mesh_gpu` says so. The two shaders are not in a composition with each
// other, so the duplication cannot be removed — but it can be stopped from
// drifting, which is the part that would show as one pass shading
// differently from another.
fn unpack_snorm16(v: u32) -> f32 {
    // Shift up then arithmetic-shift down to sign-extend the low 16 bits.
    let s = i32(v << 16u) >> 16u;
    return max(f32(s) / 32767.0, -1.0);
}

// Mirrors `mesh_gpu::oct_decode` on the CPU. The two must stay in step; the CPU
// side has the round-trip test that pins the encoding down.
fn oct_decode(packed: u32) -> vec3<f32> {
    let ex = unpack_snorm16(packed & 0xFFFFu);
    let ey = unpack_snorm16(packed >> 16u);
    var n = vec3<f32>(ex, ey, 1.0 - abs(ex) - abs(ey));
    if (n.z < 0.0) {
        let nx = n.x;
        let ny = n.y;
        n.x = (1.0 - abs(ny)) * select(-1.0, 1.0, nx >= 0.0);
        n.y = (1.0 - abs(nx)) * select(-1.0, 1.0, ny >= 0.0);
    }
    return normalize(n);
}

// The light, and beside it what screen-space reflections need: the normal and
// roughness, the specular share, and the environment's own reflection.
struct RenderedOut {
    @location(0) light: vec4<f32>,
    @location(1) surface: vec4<f32>,
    @location(2) share: vec4<f32>,
    @location(3) environment: vec4<f32>,
};

@fragment
fn fs_rendered(in: VertexOut) -> RenderedOut {
    // Toward the camera, which eye-relative space puts at the origin.
    // Orthographic: the camera's back everywhere. The eye point there is only
    // where positions are measured from, and zooming in can put it inside
    // the model, where "toward it" turns round halfway across a face.
    let to_eye = -in.world;
    var view = globals.camera_back.xyz;
    if globals.params.w < 0.5 && dot(to_eye, to_eye) > 1e-12 {
        view = normalize(to_eye);
    }

    // Flipped toward the viewer rather than culled. A modeller routinely has
    // open and inconsistently wound geometry mid-edit, and a black backface is
    // worse than a lit one — the same judgement `solid.wgsl` makes, and the
    // reason neither pass culls.
    var geometric = normalize(in.normal);
    let flipped = dot(geometric, view) < 0.0;
    if flipped {
        geometric = -geometric;
    }

    // Derivatives first and unconditionally: they are only defined in uniform
    // control flow, and the material needs the UV's to choose a mip level.
    let uv_dx = dpdx(in.uv);
    let uv_dy = dpdy(in.uv);
    var geometry: MaterialGeometry;
    geometry.position = in.local;
    geometry.normal = normalize(in.local_normal);
    geometry.position_dx = dpdx(in.local);
    geometry.position_dy = dpdy(in.local);
    // The cosine between the eye and the smooth normal, for a Fresnel: the
    // side seen, so a back face reads as one seen face on, not from behind.
    geometry.facing = abs(dot(normalize(in.normal), view));
    let sample = material_sample(in.uv, uv_dx, uv_dy, geometry);
    var surface = sample.surface;
    surface.base_color *= in.tint.rgb;
    var normal = perturb_normal(geometric, in.world, uv_dx, uv_dy, sample.normal);
    // A bump's tilt, as a surface gradient in the object's space: carried
    // into this one the way a normal is, by the inverse transpose, and
    // measured against the normal's own length there — the gradient was
    // taken against a unit normal, and the interpolated one is not.
    // On a face seen from behind the normal was turned over, and the tilt
    // turns with it.
    if any(sample.gradient != vec3<f32>(0.0)) {
        let to_eye = mat3x3<f32>(object.normal_col0.xyz, object.normal_col1.xyz, object.normal_col2.xyz);
        var g = to_eye * sample.gradient / max(length(in.normal), 1e-20);
        if flipped {
            g = -g;
        }
        let tilted = normal - g;
        if dot(tilted, tilted) > 1e-20 {
            normal = normalize(tilted);
        }
    }

    // `in.clip` is the fragment's position in pixels here — the same builtin
    // the vertex stage wrote, reinterpreted by the rasteriser. Which tile that
    // is decides which lights this fragment sees.
    // What the surface emits, added to what it reflects: a glow is light the
    // surface makes, so no light, shadow or occlusion has a say in it.
    var lit = shade_point(surface, in.world, normal, view, vec2<u32>(in.clip.xy)) + shown_glow(sample.emission);
    // An occlusion map shades only the light from the surroundings — what
    // the environment pours in from every side and a crease cannot see —
    // not a lamp's, which shadows already decide.
    if (sample.occlusion < 1.0) {
        lit -= (1.0 - sample.occlusion) * environment_ambient(surface, normal, view);
    }
    if (sample.transmission > 0.0) {
        lit = mix(lit, through_glass(surface, sample, in.world, normal, view, in.clip.xy), sample.transmission);
    }
    // Linear and unbounded. The display transform is a later pass.
    var out: RenderedOut;
    out.light = vec4<f32>(lit, 1.0);
    let roughness = clamp(surface.roughness, 0.0, 1.0);
    out.surface = vec4<f32>(normal, roughness);
    let nov = clamp(dot(normal, view), 1e-4, 1.0);
    out.share = vec4<f32>(specular_share(surface_f0(surface), nov, roughness), 1.0);
    out.environment = vec4<f32>(environment_specular(surface, normal, view) * sample.occlusion, 1.0);
    // A cut-out: nothing here. Last, after every derivative this stage
    // takes — a derivative after a discard reads a neighbour that is gone.
    if (sample.clip > 0.0 && sample.alpha < sample.clip) {
        discard;
    }
    return out;
}

// What a transmissive surface shows: its reflection, which is a clear
// dielectric's — no diffuse at all — and what is behind it, bent.
//
// Screen-space refraction, the way EEVEE does it: the ray is bent at the
// surface by the index of refraction, followed a short way on, and wherever
// that lands on the screen is what the solids' picture had there. It is not
// what a ray through a real block of glass would meet — that is the path
// tracer's — but it bends by the right amount in the right direction, which is
// what reads as glass.
fn through_glass(
    surface: Surface,
    sample: MaterialSample,
    world: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    pixel: vec2<f32>,
) -> vec3<f32> {
    var clear = surface;
    clear.base_color = vec3<f32>(0.0);
    clear.metallic = 0.0;
    let reflected = shade_point(clear, world, normal, view, vec2<u32>(pixel));
    let region = vec2<f32>(shading.region);
    var at = pixel;
    let bent = refract(-view, normal, 1.0 / sample.ior);
    if (dot(bent, bent) > 0.0 && region.x >= 1.0) {
        let ahead = world + bent * (length(world) * 0.05);
        let clip = shading.view_proj * vec4<f32>(ahead, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xy / clip.w;
            at = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * region;
        }
    }
    let last = max(region - vec2<f32>(1.0), vec2<f32>(0.0));
    let seen = textureLoad(behind, vec2<i32>(clamp(at, vec2<f32>(0.0), last)), 0).rgb;
    // What is not reflected goes in, tinted by the colour: Schlick's Fresnel
    // for the dielectric the clear surface is.
    let nov = clamp(dot(normal, view), 1e-4, 1.0);
    let f = DIELECTRIC_F0 + (1.0 - DIELECTRIC_F0) * pow(1.0 - nov, 5.0);
    return reflected + (1.0 - f) * surface.base_color * seen + shown_glow(sample.emission);
}

// Tilts `n` by a tangent-space normal, with the tangent frame taken from the
// screen derivatives of position and UV.
//
// No per-vertex tangents: the frame is solved per pixel from how the surface
// and its coordinates move across the screen (Schüler's cotangent frame). It
// needs nothing uploaded and nothing kept in step with the mesh, and it is
// exactly the frame the UVs imply — which is what a map painted on those UVs
// was painted in. Where it differs from a baked MikkTSpace frame is across
// smooth-shaded faces, by the per-face variation of the UV mapping; a
// glTF normal map baked against MikkTSpace can show faint facets there.
//
// +X along U and +Y along V, with V up. That is the OpenGL and glTF convention
// once glTF's top-left UV origin is flipped to ours, which the importer does.
fn perturb_normal(
    n: vec3<f32>,
    p: vec3<f32>,
    uv_dx: vec2<f32>,
    uv_dy: vec2<f32>,
    tangent_normal: vec3<f32>,
) -> vec3<f32> {
    let dp_dx = dpdx(p);
    let dp_dy = dpdy(p);
    // The crosses are the other way round from the usual GLSL listing, and
    // must be: WGSL's `dpdy` runs *down* the screen where GLSL's `dFdy` runs
    // up, which negates both the position and the UV term and so the whole
    // frame. Copied as written, every normal map is lit inside out — a test
    // that tilts a map toward the light caught exactly that.
    let dp_dy_perp = cross(n, dp_dy);
    let dp_dx_perp = cross(dp_dx, n);
    let t = dp_dy_perp * uv_dx.x + dp_dx_perp * uv_dy.x;
    let b = dp_dy_perp * uv_dx.y + dp_dx_perp * uv_dy.y;
    let scale = max(dot(t, t), dot(b, b));
    // A surface whose UVs do not change across the pixel — no UVs at all, or
    // a degenerate projection — has no frame to tilt in. Unchanged rather
    // than NaN, which would be a black pixel.
    if scale < 1e-30 {
        return n;
    }
    let inv = inverseSqrt(scale);
    let tilted = t * (inv * tangent_normal.x) + b * (inv * tangent_normal.y) + n * tangent_normal.z;
    let len2 = dot(tilted, tilted);
    if len2 < 1e-20 {
        return n;
    }
    return tilted * inverseSqrt(len2);
}
