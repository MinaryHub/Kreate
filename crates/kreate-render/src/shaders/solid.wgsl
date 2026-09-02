// Solid shading for the viewport.
//
// Positions arrive camera-relative: the camera sits at the origin and object
// transforms have already been rebased in f64 on the CPU. Nothing here ever sees
// a large world coordinate, which is what keeps a scene 10 km from the origin
// from jittering.
//
// Normals arrive octahedron-encoded in a single u32 — 4 bytes instead of 12 on a
// buffer that is re-uploaded every frame during a drag.

struct Globals {
    view_proj: mat4x4<f32>,
    // x: shading mode (0 studio, 1 flat, 2 matcap-ish)
    // y: unused, z: unused, w: unused
    params: vec4<f32>,
};

struct ObjectData {
    model: mat4x4<f32>,
    // The normal matrix, one column per vec4 because a mat3x3 in a uniform block
    // has 16-byte column alignment anyway and the padding is better made explicit
    // than discovered as a garbled normal.
    normal_col0: vec4<f32>,
    normal_col1: vec4<f32>,
    normal_col2: vec4<f32>,
    base_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: ObjectData;

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

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) view_pos: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal_packed: u32,
) -> VsOut {
    var out: VsOut;
    let world = object.model * vec4<f32>(position, 1.0);
    out.clip = globals.view_proj * world;
    let nm = mat3x3<f32>(
        object.normal_col0.xyz,
        object.normal_col1.xyz,
        object.normal_col2.xyz,
    );
    out.normal = nm * oct_decode(normal_packed);
    out.view_pos = world.xyz;
    return out;
}

// Three camera-relative lights: a key above and to the left, a dimmer fill from
// the opposite side, and a rim from behind. Camera-relative rather than
// world-fixed so a surface never goes fully black as the model is orbited —
// unlit geometry reads as a modelling mistake rather than as lighting.
const KEY_DIR: vec3<f32> = vec3<f32>(-0.45, 0.35, 0.82);
const FILL_DIR: vec3<f32> = vec3<f32>(0.6, -0.2, 0.5);
const RIM_DIR: vec3<f32> = vec3<f32>(0.1, 0.9, -0.4);

fn studio(n: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let key = max(dot(n, normalize(KEY_DIR)), 0.0);
    let fill = max(dot(n, normalize(FILL_DIR)), 0.0);
    let rim = pow(max(dot(n, normalize(RIM_DIR)), 0.0), 2.0);
    let ambient = 0.22;
    return base * (ambient + key * 0.75 + fill * 0.28) + vec3<f32>(rim * 0.12);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    // Flip the normal on backfaces, so a mesh with inconsistent winding is lit
    // rather than showing black patches that read as a shading bug instead of a
    // topology one.
    //
    // The test is the rasteriser's own facing flag, which is constant across a
    // triangle. Comparing the normal against the view vector instead — the
    // obvious-looking `dot(n, to_eye) < 0.0` — is wrong: the view vector varies
    // per fragment under perspective, so on a face seen at a grazing angle the
    // sign flips partway across it and the shading breaks along a curve through
    // the middle of a flat surface.
    var n = normalize(in.normal);
    if (!front_facing) {
        n = -n;
    }
    let to_eye = normalize(-in.view_pos);

    let base = object.base_color.rgb;
    var color: vec3<f32>;
    if (globals.params.x < 0.5) {
        color = studio(n, base);
    } else {
        // Flat: a single camera-facing term. Useful for reading silhouette and
        // topology without lighting getting in the way.
        color = base * (0.35 + 0.65 * max(dot(n, to_eye), 0.0));
    }

    // The surface is written to an sRGB target, so the hardware does the
    // encoding; shading stays in linear space.
    return vec4<f32>(color, object.base_color.a);
}
