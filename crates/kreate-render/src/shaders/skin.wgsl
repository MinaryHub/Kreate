// Skinning, on the GPU.
//
// While a pose changes the CPU uploads the bones' matrices and nothing else.
// Each render vertex is recomputed from an immutable copy of the rest mesh:
// the weighted mean of where each bone that weighs on it would put it.
//
// This is `kreate_scene::skin::skin` term for term, and it has to be — the
// CPU's answer is the one picking, snapping and the path tracer see, and a
// drawn mesh that disagrees with the one the pointer hits is a bug however
// small the difference. So: every bone with a positive weight, not the four
// largest; divided by the weights' sum, not assumed normalised; and a vertex
// no bone weighs on stays where it is.

@group(0) @binding(0) var<storage, read> rest_positions: array<f32>;
@group(0) @binding(1) var<storage, read_write> live_positions: array<f32>;
// The mesh vertex each render vertex was split from.
@group(0) @binding(2) var<storage, read> render_to_vert: array<u32>;
// Influences, as CSR over the mesh's vertices: vertex `v` is weighed on by
// `influence_bones[offsets[v] .. offsets[v + 1]]`.
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> influence_bones: array<u32>;
@group(0) @binding(5) var<storage, read> influence_weights: array<f32>;
// Three rows of an affine matrix per bone, the fourth column the translation.
@group(0) @binding(6) var<storage, read> bones: array<vec4<f32>>;

@compute @workgroup_size(64)
fn skin_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if (r >= arrayLength(&render_to_vert)) {
        return;
    }
    let i = r * 3u;
    let rest = vec4<f32>(rest_positions[i], rest_positions[i + 1u], rest_positions[i + 2u], 1.0);
    let v = render_to_vert[r];
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var total = 0.0;
    if (v + 1u < arrayLength(&offsets)) {
        for (var k = offsets[v]; k < offsets[v + 1u]; k = k + 1u) {
            let w = influence_weights[k];
            let b = influence_bones[k] * 3u;
            // A bone the matrices do not cover is skipped, as the CPU skips a
            // weight layer whose bone has no matrix.
            if (w > 0.0 && b + 2u < arrayLength(&bones)) {
                sum += vec3<f32>(dot(bones[b], rest), dot(bones[b + 1u], rest), dot(bones[b + 2u], rest)) * w;
                total += w;
            }
        }
    }
    var out = rest.xyz;
    if (total > 0.0) {
        out = sum / total;
    }
    live_positions[i] = out.x;
    live_positions[i + 1u] = out.y;
    live_positions[i + 2u] = out.z;
}
