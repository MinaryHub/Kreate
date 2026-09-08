// The interactive transform, on the GPU.
//
// While a drag is running the CPU uploads one 64-byte uniform per frame and
// nothing else. The positions and the normals the rasteriser reads are computed
// here from an immutable copy of the mesh as it was when the drag began.
//
// The plan calls this the single highest-leverage optimisation in the renderer,
// and the measurement agrees: getting the mesh ready for the GPU on the CPU
// costs 83 ms a frame at two hundred thousand vertices, against a budget of
// one. Most of that is work a move does not change — the triangulation, the
// vertex split, the edge list — and the rest is arithmetic a GPU does in
// microseconds.
//
// # Always from the rest state, never incrementally
//
// §2.2's rule, and the reason it is a rule: a long drag applying deltas
// accumulates float error, and a cancel has nothing exact to go back to. Every
// frame recomputes from `rest_*`, which nothing writes to for the life of the
// drag.
//
// # Why the gesture is decomposed rather than a matrix
//
// A weighted transform is not a weighted matrix. Proportional editing blends
// the *transform* towards the identity, not the *result* towards the rest
// position, and for a rotation those differ: blending positions drags a
// partly-weighted vertex along the chord and shrinks the mesh towards the
// pivot, while blending the rotation turns it through a smaller angle — which
// is what the tool means. So the rotation arrives as a quaternion to be
// slerped, and this file mirrors `ops::mesh`'s arithmetic term for term.
//
// A gesture is one of move, rotate or scale, so exactly one of the three is
// away from the identity. Composing all three costs a few instructions and
// needs no branch, and it keeps this shader from having a mode.

struct Gesture {
    // xyz used, w ignored.
    translation: vec4<f32>,
    // A quaternion, xyzw.
    rotation: vec4<f32>,
    scale: vec4<f32>,
    // What the rotation and the scale happen about.
    pivot: vec4<f32>,
}

// Positions are `array<f32>` indexed by `3 * v`, not `array<vec3<f32>>`: a
// `vec3` in a storage buffer has a sixteen-byte stride, which would waste a
// third of the largest buffer in the system. The same bytes are bound as a
// vertex buffer with a twelve-byte stride for the draw.
@group(0) @binding(0) var<storage, read> rest_positions: array<f32>;
@group(0) @binding(1) var<storage, read> rest_normals: array<u32>;
@group(0) @binding(2) var<storage, read_write> live_positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> live_normals: array<u32>;
// One per render vertex: how much of the gesture it takes. One for the
// selection, falling off for what proportional editing reaches, zero elsewhere.
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<uniform> gesture: Gesture;

// Which render vertices make up each face, as CSR over the mesh's corners.
// Face `f` owns `corner_render[face_offsets[f] .. face_offsets[f + 1]]`.
@group(1) @binding(0) var<storage, read> face_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> corner_render: array<u32>;
@group(1) @binding(2) var<storage, read_write> face_normals: array<f32>;
// Which faces each render vertex averages: its *smooth group*, not every face
// at its vertex. A render vertex is a (vertex, smooth group) pair by
// construction — that is what the vertex split is for — so this is exactly the
// set `compute_corner_normals` sums on the CPU.
@group(1) @binding(3) var<storage, read> group_offsets: array<u32>;
@group(1) @binding(4) var<storage, read> group_faces: array<u32>;

// Three loaders rather than one taking a pointer: WGSL does not allow a
// pointer into the storage address space as a function parameter without the
// `unrestricted_pointer_parameters` language extension, which is not something
// to require of a driver for the sake of saving six lines. A device test caught
// this — the module did not compile at all.
fn load_rest(v: u32) -> vec3<f32> {
    let i = v * 3u;
    return vec3<f32>(rest_positions[i], rest_positions[i + 1u], rest_positions[i + 2u]);
}

fn load_live(v: u32) -> vec3<f32> {
    let i = v * 3u;
    return vec3<f32>(live_positions[i], live_positions[i + 1u], live_positions[i + 2u]);
}

fn load_face_normal(f: u32) -> vec3<f32> {
    let i = f * 3u;
    return vec3<f32>(face_normals[i], face_normals[i + 1u], face_normals[i + 2u]);
}

fn store_live_position(v: u32, p: vec3<f32>) {
    let i = v * 3u;
    live_positions[i] = p.x;
    live_positions[i + 1u] = p.y;
    live_positions[i + 2u] = p.z;
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    // The usual identity: v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v).
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

/// The identity turned `w` of the way towards `q`.
///
/// Spherical, to match `Quat::slerp` on the CPU. Normalised linear
/// interpolation would agree at the ends and differ in between, and "in
/// between" is exactly where proportional editing lives.
fn slerp_from_identity(q: vec4<f32>, w: f32) -> vec4<f32> {
    let identity = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    // The shorter arc. A quaternion and its negation are the same rotation, and
    // taking the long way round would spin a partly-weighted vertex most of the
    // way about the pivot.
    var to = q;
    var cos_theta = dot(identity, q);
    if (cos_theta < 0.0) {
        to = -q;
        cos_theta = -cos_theta;
    }
    // Almost the identity: the sines underflow and linear is exact to float
    // precision anyway.
    if (cos_theta > 0.9995) {
        return normalize(mix(identity, to, w));
    }
    let theta = acos(clamp(cos_theta, -1.0, 1.0));
    let sin_theta = sin(theta);
    let a = sin((1.0 - w) * theta) / sin_theta;
    let b = sin(w * theta) / sin_theta;
    return identity * a + to * b;
}

fn unpack_snorm16(v: u32) -> f32 {
    let s = i32(v << 16u) >> 16u;
    return max(f32(s) / 32767.0, -1.0);
}

fn oct_decode(packed: u32) -> vec3<f32> {
    let ex = unpack_snorm16(packed & 0xFFFFu);
    let ey = unpack_snorm16(packed >> 16u);
    var n = vec3<f32>(ex, ey, 1.0 - abs(ex) - abs(ey));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

/// Mirrors `mesh_gpu::oct_encode` term for term. A test asserts they agree.
fn oct_encode(normal: vec3<f32>) -> u32 {
    let n = normalize(normal);
    let denom = abs(n.x) + abs(n.y) + abs(n.z);
    if (denom == 0.0) {
        return 0u;
    }
    let p = n / denom;
    var x = p.x;
    var y = p.y;
    if (p.z <= 0.0) {
        // Fold the lower hemisphere out onto the square.
        let px = x;
        let py = y;
        x = (1.0 - abs(py)) * select(-1.0, 1.0, px >= 0.0);
        y = (1.0 - abs(px)) * select(-1.0, 1.0, py >= 0.0);
    }
    let qx = u32(i32(round(clamp(x, -1.0, 1.0) * 32767.0)) & 0xFFFF);
    let qy = u32(i32(round(clamp(y, -1.0, 1.0) * 32767.0)) & 0xFFFF);
    return qx | (qy << 16u);
}

@compute @workgroup_size(64)
fn deform_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = id.x;
    if (v >= arrayLength(&weights)) {
        return;
    }
    let w = weights[v];
    let rest = load_rest(v);
    if (w == 0.0) {
        // Untouched. Written anyway rather than skipped, because the live
        // buffer is what the rasteriser reads and the previous frame's answer
        // is in it.
        store_live_position(v, rest);
        return;
    }
    // Scale then rotate about the pivot, then translate. Only one of the three
    // is ever away from the identity, so the order is not a decision anyone has
    // to agree with.
    let s = mix(vec3<f32>(1.0, 1.0, 1.0), gesture.scale.xyz, w);
    let q = slerp_from_identity(gesture.rotation, w);
    let about = gesture.pivot.xyz;
    let local = (rest - about) * s;
    store_live_position(v, about + quat_rotate(q, local) + gesture.translation.xyz * w);
}

/// Newell's method over the face's ring, from the deformed positions.
///
/// Newell rather than one cross product because a face may not be planar, and
/// any single cross product would then depend on which corner happened to be
/// sampled. The same choice `tessellate::newell_normal` makes, and it has to be
/// the same choice: the CPU recomputes these on mouse-up and a different
/// formula would make the shading jump.
///
/// Unnormalised, so that the vertex pass below averages by area.
@compute @workgroup_size(64)
fn compute_face_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = id.x;
    if (f + 1u >= arrayLength(&face_offsets)) {
        return;
    }
    let start = face_offsets[f];
    let end = face_offsets[f + 1u];
    var n = vec3<f32>(0.0, 0.0, 0.0);
    if (end > start) {
        var previous = load_live(corner_render[end - 1u]);
        for (var c = start; c < end; c = c + 1u) {
            let current = load_live(corner_render[c]);
            n += vec3<f32>(
                (previous.y - current.y) * (previous.z + current.z),
                (previous.z - current.z) * (previous.x + current.x),
                (previous.x - current.x) * (previous.y + current.y),
            );
            previous = current;
        }
    }
    let i = f * 3u;
    face_normals[i] = n.x;
    face_normals[i + 1u] = n.y;
    face_normals[i + 2u] = n.z;
}

@compute @workgroup_size(64)
fn compute_vertex_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = id.x;
    if (v >= arrayLength(&weights)) {
        return;
    }
    let start = group_offsets[v];
    let end = group_offsets[v + 1u];
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var g = start; g < end; g = g + 1u) {
        sum += load_face_normal(group_faces[g]);
    }
    // A degenerate neighbourhood would otherwise give a zero normal and a black
    // surface. Keeping what was there is better than that and better than a
    // guess: it is the answer the mesh had before the drag.
    if (dot(sum, sum) == 0.0) {
        live_normals[v] = rest_normals[v];
        return;
    }
    live_normals[v] = oct_encode(sum);
}
