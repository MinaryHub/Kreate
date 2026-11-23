// The path tracer's rays walked through the scene's own trees, on any
// device: a tree over the placings, and at each placing the ray carried into
// its mesh's space and walked through that mesh's tree. A placing moves the
// ray by an affine map and leaves its direction unnormalised, so a distance
// along it is the same number in both spaces.

struct PtNode {
    min: vec3<f32>,
    // Leaf: the first index into `order`. Interior: the left child; the
    // right is the next.
    first: u32,
    max: vec3<f32>,
    // Primitives in a leaf — placings in the scene's tree, triangles in a
    // mesh's; zero for an interior node.
    count: u32,
};

@group(0) @binding(4) var<storage, read> pt_nodes: array<PtNode>;
@group(0) @binding(5) var<storage, read> pt_order: array<u32>;

fn pt_box_near(origin: vec3<f32>, inv_dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, limit: f32) -> f32 {
    let a = (lo - origin) * inv_dir;
    let b = (hi - origin) * inv_dir;
    let near = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
    let far = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    return select(1e30, max(near, 0.0), far >= max(near, 0.0) && near < limit);
}

fn pt_inverse_dir(dir: vec3<f32>) -> vec3<f32> {
    return 1.0 / select(dir, vec3<f32>(1e-12), abs(dir) < vec3<f32>(1e-12));
}

fn pt_trace(origin: vec3<f32>, dir: vec3<f32>, limit: f32, any: bool, skip: vec2<u32>) -> PtHit {
    var best = PtHit(-1.0, 0u, 0.0, 0.0, 0u);
    var nearest = limit;
    let near = select(1e-4, -1e-5, skip.x != PT_NONE);
    let inv_dir = pt_inverse_dir(dir);
    var stack: array<u32, 48>;
    var top = 1;
    stack[0] = 0u;
    loop {
        if (top == 0) {
            break;
        }
        top -= 1;
        let node = pt_nodes[stack[top]];
        if (pt_box_near(origin, inv_dir, node.min, node.max, nearest) >= 1e29) {
            continue;
        }
        if (node.count == 0u) {
            // A child is never the root: a link to node 0 is the empty
            // scene's placeholder, and following it would not end.
            if (node.first != 0u && top < 46) {
                let c0 = node.first;
                let c1 = node.first + 1u;
                let n0 = pt_nodes[c0];
                let n1 = pt_nodes[c1];
                let t0 = pt_box_near(origin, inv_dir, n0.min, n0.max, nearest);
                let t1 = pt_box_near(origin, inv_dir, n1.min, n1.max, nearest);
                if (t0 < 1e29 && t1 < 1e29) {
                    // Push farther first so nearer is popped first.
                    stack[top] = select(c0, c1, t0 <= t1);
                    stack[top + 1] = select(c1, c0, t0 <= t1);
                    top += 2;
                } else if (t0 < 1e29) {
                    stack[top] = c0;
                    top += 1;
                } else if (t1 < 1e29) {
                    stack[top] = c1;
                    top += 1;
                }
            }
            continue;
        }
        for (var k = 0u; k < node.count; k++) {
            let inst = pt_order[node.first + k];
            let placing = pt_instances[inst];
            let o = pt_point_through(placing.local, origin);
            let d = pt_vector_through(placing.local, dir);
            let inv_d = pt_inverse_dir(d);
            var inner: array<u32, 48>;
            var depth = 1;
            inner[0] = placing.root;
            loop {
                if (depth == 0) {
                    break;
                }
                depth -= 1;
                let mesh_node = pt_nodes[inner[depth]];
                if (pt_box_near(o, inv_d, mesh_node.min, mesh_node.max, nearest) >= 1e29) {
                    continue;
                }
                if (mesh_node.count == 0u) {
                    if (mesh_node.first != 0u && depth < 46) {
                        let mc0 = mesh_node.first;
                        let mc1 = mesh_node.first + 1u;
                        let mn0 = pt_nodes[mc0];
                        let mn1 = pt_nodes[mc1];
                        let mt0 = pt_box_near(o, inv_d, mn0.min, mn0.max, nearest);
                        let mt1 = pt_box_near(o, inv_d, mn1.min, mn1.max, nearest);
                        if (mt0 < 1e29 && mt1 < 1e29) {
                            inner[depth] = select(mc0, mc1, mt0 <= mt1);
                            inner[depth + 1] = select(mc1, mc0, mt0 <= mt1);
                            depth += 2;
                        } else if (mt0 < 1e29) {
                            inner[depth] = mc0;
                            depth += 1;
                        } else if (mt1 < 1e29) {
                            inner[depth] = mc1;
                            depth += 1;
                        }
                    }
                    continue;
                }
                for (var j = 0u; j < mesh_node.count; j++) {
                    let index = pt_order[mesh_node.first + j];
                    if (inst == skip.x && index == skip.y) {
                        continue;
                    }
                    let tri = pt_triangles[index];
                    let a = pt_positions[tri.x].xyz;
                    let e1 = pt_positions[tri.y].xyz - a;
                    let e2 = pt_positions[tri.z].xyz - a;
                    let h = cross(d, e2);
                    let det = dot(e1, h);
                    if (abs(det) < 1e-12) {
                        continue;
                    }
                    let f = 1.0 / det;
                    let s = o - a;
                    let u = f * dot(s, h);
                    if (u < 0.0 || u > 1.0) {
                        continue;
                    }
                    let q = cross(s, e1);
                    let v = f * dot(d, q);
                    if (v < 0.0 || u + v > 1.0) {
                        continue;
                    }
                    let t = f * dot(e2, q);
                    if (t <= near || t >= nearest || !pt_admit(index, u, v, any)) {
                        continue;
                    }
                    nearest = t;
                    best = PtHit(max(t, 0.0), index, u, v, inst);
                    if (any) {
                        return best;
                    }
                }
            }
        }
    }
    return best;
}
