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
// And it does not decide what a surface is made of. `material_surface` is
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
    /// xyz: unused here. w: unused.
    params: vec4<f32>,
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

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    /// Eye-relative, in metres: the same space the lights are packed into.
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn vs_rendered(
    @location(0) position: vec3<f32>,
    @location(1) packed_normal: u32,
) -> VertexOut {
    let world = object.model * vec4<f32>(position, 1.0);
    var out: VertexOut;
    out.clip = globals.view_proj * world;
    out.world = world.xyz;

    let n = oct_decode(packed_normal);
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

@fragment
fn fs_rendered(in: VertexOut) -> @location(0) vec4<f32> {
    // Toward the camera, which eye-relative space puts at the origin.
    let to_eye = -in.world;
    var view = vec3<f32>(0.0, 0.0, 1.0);
    if dot(to_eye, to_eye) > 1e-12 {
        view = normalize(to_eye);
    }

    // Flipped toward the viewer rather than culled. A modeller routinely has
    // open and inconsistently wound geometry mid-edit, and a black backface is
    // worse than a lit one — the same judgement `solid.wgsl` makes, and the
    // reason neither pass culls.
    var normal = normalize(in.normal);
    if dot(normal, view) < 0.0 {
        normal = -normal;
    }

    // The compiled material. A surface coordinate of zero for now: the render
    // mesh has no UVs yet, so a texture would sample one texel. Every material
    // is currently constants, which do not read it — and the day UVs arrive
    // this line becomes `in.uv` and nothing else changes.
    let surface = material_surface(vec2<f32>(0.0));

    // `in.clip` is the fragment's position in pixels here — the same builtin
    // the vertex stage wrote, reinterpreted by the rasteriser. Which tile that
    // is decides which lights this fragment sees.
    let lit = shade_point(surface, in.world, normal, view, vec2<u32>(in.clip.xy));
    // Linear and unbounded. The display transform is a later pass.
    return vec4<f32>(lit, 1.0);
}
