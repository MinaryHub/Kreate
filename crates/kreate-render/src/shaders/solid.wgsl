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
    // y: one to colour by face orientation, z: unused, w: unused
    params: vec4<f32>,
    // The camera's right, up and back in the world: the studio lights are
    // placed in these, so they turn with the view.
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_back: vec4<f32>,
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
    // x: one to colour by the weight in the first UV coordinate.
    paint: vec4<f32>,
    // The colour of each material slot's faces: the run being drawn is the
    // instance index. See `SOLID_SLOTS`.
    slot_colors: array<vec4<f32>, 8>,
    // The image layer each slot shows in the solid modes, four to a vec4;
    // below zero for its colour.
    slot_layers: array<vec4<f32>, 2>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: ObjectData;
// The images the solid modes show: see `solid_textures.rs`.
@group(3) @binding(0) var solid_tex: texture_2d_array<f32>;
@group(3) @binding(1) var solid_sampler: sampler;

// ---------------------------------------------------------------------------
// The cluster path.
//
// A second vertex entry point in *this* file rather than a shader of its own,
// and that is the whole point: it shares `fs_main` with the direct path
// literally rather than by resemblance. Two copies of the lighting would drift,
// and the drift would show up as one path shading differently from the other —
// which is exactly what the pixel-identity test would then be unable to
// distinguish from a real bug.
//
// Bindings a given pipeline's entry point does not touch are simply absent from
// its layout; validation is per entry point, so the direct pipeline is
// unaffected by everything below.

struct Cluster {
    sphere: vec4<f32>,
    cone: vec4<f32>,
    index_offset: u32,
    triangle_count: u32,
    lod: u32,
    _pad: u32,
};

struct Instance {
    model: mat4x4<f32>,
    normal_col0: vec4<f32>,
    normal_col1: vec4<f32>,
    normal_col2: vec4<f32>,
    // x: cluster count. y: base index into the visible list.
    counts: vec4<u32>,
    scale: vec4<f32>,
    base_color: vec4<f32>,
};

@group(0) @binding(1) var<storage, read> visible: array<u32>;
// Binding 1, not 0: `ObjectData` owns (1, 0) for the direct path, and one
// module cannot declare two resources at the same binding. The cluster
// pipeline's group 1 layout simply omits binding 0.
@group(1) @binding(1) var<uniform> instance: Instance;
// Declared flat and indexed by hand: a `vec3<f32>` in a storage array has a
// 16-byte stride in WGSL, which would waste a third of the largest buffer in
// the system. The same convention as the overlay and id shaders.
@group(2) @binding(0) var<storage, read> cluster_positions: array<f32>;
@group(2) @binding(1) var<storage, read> cluster_normals: array<u32>;
@group(2) @binding(2) var<storage, read> cluster_records: array<Cluster>;
@group(2) @binding(3) var<storage, read> cluster_indices: array<u32>;

fn cluster_position(i: u32) -> vec3<f32> {
    let b = 3u * i;
    return vec3<f32>(
        cluster_positions[b],
        cluster_positions[b + 1u],
        cluster_positions[b + 2u],
    );
}

// One instance per surviving cluster, `VERTICES_PER_CLUSTER` vertices each.
//
// The vertex count is fixed because a single draw cannot vary it per instance,
// so a cluster with fewer than the maximum triangles has its surplus vertices
// collapsed onto one point — a zero-area triangle the rasteriser discards. That
// is much cheaper than the alternative of a draw call per cluster, which is the
// thing this whole design exists to avoid.
//
// Worth being precise about what the early-out buys, because a falsification
// pass found that deleting it changes not one pixel. Clusters average about 95
// of the 128 triangles, so without the check the surplus indices read on into
// the *next* cluster's range and re-draw its triangles — exactly coincident with
// the real ones, and rejected by the depth test, hence invisible. The last
// cluster reads past the end of the buffer, and WebGPU clamps out-of-bounds
// storage reads, so its three corners come out identical and the triangle is
// degenerate anyway.
//
// So this is a performance measure, not a correctness one: it stops roughly a
// quarter of the issued vertices from being shaded to no purpose. Stated here
// rather than left implied, because a future reader who assumes it is load
// bearing will not understand why removing it looks free.
@vertex
fn vs_cluster(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VsOut {
    var out: VsOut;
    let cluster = cluster_records[visible[instance.counts.y + ii]];

    if (vi / 3u >= cluster.triangle_count) {
        // Degenerate: every surplus vertex goes to the same place, so the
        // triangle has no area. Returned rather than discarded because a vertex
        // shader cannot discard.
        out.clip = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.normal = vec3<f32>(0.0, 0.0, 1.0);
        out.view_pos = vec3<f32>(0.0);
        out.base_color = instance.base_color;
        out.weight = vec2<f32>(0.0);
        out.ramp = 0.0;
        out.layer = -1.0;
        return out;
    }

    let index = cluster_indices[cluster.index_offset + vi];
    let world = instance.model * vec4<f32>(cluster_position(index), 1.0);
    out.clip = globals.view_proj * world;
    let nm = mat3x3<f32>(
        instance.normal_col0.xyz,
        instance.normal_col1.xyz,
        instance.normal_col2.xyz,
    );
    out.normal = nm * oct_decode(cluster_normals[index]);
    out.view_pos = world.xyz;
    out.base_color = instance.base_color;
    out.weight = vec2<f32>(0.0);
    out.ramp = 0.0;
    out.layer = -1.0;
    return out;
}

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
    // Carried through the vertex stage rather than read from a uniform in the
    // fragment stage, so that `fs_main` depends on no per-draw binding at all.
    // That is what lets the direct path and the cluster path share it: they
    // supply the colour from different uniforms in different vertex shaders and
    // the fragment stage cannot tell which.
    @location(2) @interpolate(flat) base_color: vec4<f32>,
    // The weight being painted, and whether to show it — see `ramp`. In
    // sculpt mode, the mask (x) and the face set (y).
    @location(3) weight: vec2<f32>,
    @location(4) @interpolate(flat) ramp: f32,
    // The image layer the face shows, below zero for none.
    @location(5) @interpolate(flat) layer: f32,
};

// A face set's colour: a hue from its number, soft, so the form still reads.
fn face_set_color(id: f32) -> vec3<f32> {
    let h = fract(id * 0.618034) * 6.0;
    let x = 1.0 - abs(h % 2.0 - 1.0);
    var rgb: vec3<f32>;
    if (h < 1.0) { rgb = vec3<f32>(1.0, x, 0.0); }
    else if (h < 2.0) { rgb = vec3<f32>(x, 1.0, 0.0); }
    else if (h < 3.0) { rgb = vec3<f32>(0.0, 1.0, x); }
    else if (h < 4.0) { rgb = vec3<f32>(0.0, x, 1.0); }
    else if (h < 5.0) { rgb = vec3<f32>(x, 0.0, 1.0); }
    else { rgb = vec3<f32>(1.0, 0.0, x); }
    return mix(vec3<f32>(0.9), rgb, 0.5);
}

// Blender's weight ramp: blue at nothing, through green, to red at all.
fn weight_color(w: f32) -> vec3<f32> {
    let t = clamp(w, 0.0, 1.0);
    let blue = vec3<f32>(0.0, 0.0, 1.0);
    let cyan = vec3<f32>(0.0, 1.0, 1.0);
    let green = vec3<f32>(0.0, 1.0, 0.0);
    let yellow = vec3<f32>(1.0, 1.0, 0.0);
    let red = vec3<f32>(1.0, 0.0, 0.0);
    if (t < 0.25) {
        return mix(blue, cyan, t * 4.0);
    } else if (t < 0.5) {
        return mix(cyan, green, (t - 0.25) * 4.0);
    } else if (t < 0.75) {
        return mix(green, yellow, (t - 0.5) * 4.0);
    }
    return mix(yellow, red, (t - 0.75) * 4.0);
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal_packed: u32,
    @location(2) uv: vec2<f32>,
    @builtin(instance_index) slot: u32,
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
    // The run's material colour; the object's alpha, which x-ray reads.
    out.base_color = vec4<f32>(object.slot_colors[min(slot, 7u)].rgb, object.base_color.a);
    out.weight = uv;
    out.ramp = object.paint.x;
    let s = min(slot, 7u);
    out.layer = object.slot_layers[s / 4u][s % 4u];
    return out;
}

// Three lights in the camera's frame — x right, y up, z towards the viewer: a
// key above and to the left, a dimmer fill from the opposite side, and a rim
// from behind. Camera-relative rather than world-fixed so a surface never goes
// fully black as the model is orbited — unlit geometry reads as a modelling
// mistake rather than as lighting.
const KEY_DIR: vec3<f32> = vec3<f32>(-0.45, 0.55, 0.70);
const FILL_DIR: vec3<f32> = vec3<f32>(0.65, -0.15, 0.55);
const RIM_DIR: vec3<f32> = vec3<f32>(0.2, 0.6, -0.75);

fn camera_dir(d: vec3<f32>) -> vec3<f32> {
    let v = globals.camera_right.xyz * d.x + globals.camera_up.xyz * d.y + globals.camera_back.xyz * d.z;
    return normalize(v);
}

fn studio(n: vec3<f32>, v: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let key_dir = camera_dir(KEY_DIR);
    // Wrapped diffuse on the key, so the terminator is soft as under a big
    // studio softbox rather than the hard edge of a point light.
    let key = max((dot(n, key_dir) + 0.25) / 1.25, 0.0);
    let fill = max(dot(n, camera_dir(FILL_DIR)), 0.0);
    // Sky above, floor below, in the camera's frame: the upward faces of a
    // model read lighter than the downward ones, which is half of how a shape
    // is seen.
    let hemi = dot(n, globals.camera_up.xyz) * 0.5 + 0.5;
    let ambient = mix(0.10, 0.30, hemi);
    let diffuse = base * (ambient + key * 0.72 + fill * 0.22);
    // A broad highlight from the key, and a rim where the surface turns away
    // — the two cues that say "round" and "edge" at a glance.
    let h = normalize(key_dir + v);
    let spec = pow(max(dot(n, h), 0.0), 40.0) * 0.22;
    let rim = pow(max(dot(n, camera_dir(RIM_DIR)), 0.0), 2.0) * 0.10;
    let fresnel = pow(1.0 - max(dot(n, v), 0.0), 4.0) * 0.12;
    return diffuse + vec3<f32>(spec + rim + fresnel);
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

    // The ramp is light's reflectance like any colour: shaded, so the form
    // still reads under the weights, as Blender's weight paint does.
    // Sampled before any branch: a sample needs its neighbours' coordinates,
    // which a branch some pixels skip does not give it.
    let image = textureSample(solid_tex, solid_sampler, in.weight, max(i32(in.layer), 0)).rgb;
    var base = in.base_color.rgb;
    if (in.ramp < 0.5 && in.layer > -0.5) {
        base = image;
    }
    if (in.ramp > 0.5 && in.ramp < 1.5) {
        base = weight_color(in.weight.x);
    } else if (in.ramp > 1.5) {
        // Sculpt: a face set tints its faces, and the mask darkens what it
        // holds, as Blender shows both.
        if (in.weight.y > 0.5) {
            base = base * face_set_color(in.weight.y);
        }
        base = base * (1.0 - 0.7 * clamp(in.weight.x, 0.0, 1.0));
    }
    // Blender's Face Orientation overlay: blue where the front of a face is
    // seen, red where its back is — a flipped normal shows at a glance.
    if (globals.params.y > 0.5) {
        base = select(vec3<f32>(0.8, 0.02, 0.02), vec3<f32>(0.03, 0.12, 0.8), front_facing);
    }
    var color: vec3<f32>;
    if (globals.params.x < 0.5) {
        color = studio(n, to_eye, base);
    } else {
        // Flat: no light at all, Blender's — the plain colour. It reads the
        // silhouette, and the material colours as they were picked.
        color = base;
    }

    // The surface is written to an sRGB target, so the hardware does the
    // encoding; shading stays in linear space.
    // X-ray: see-through, by the amount the globals say.
    let alpha = select(in.base_color.a, globals.params.z, globals.params.z > 0.0);
    return vec4<f32>(color, alpha);
}
