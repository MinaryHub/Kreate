// The path tracer's rays asked of the device's own ray tracing: the same
// placings, built by the driver into its acceleration structure, one bottom
// level per mesh. What the trees' walk decides in the shader — the triangle
// a ray leaves from, glass a shadow passes, a cut-out's clear part — is
// decided here on each candidate, so every triangle is built non-opaque and
// comes back to be asked.

@group(0) @binding(20) var pt_scene: acceleration_structure;

fn pt_trace(origin: vec3<f32>, dir: vec3<f32>, limit: f32, any: bool, skip: vec2<u32>) -> PtHit {
    // The trees' walk lets a ray that leaves a triangle meet another a hair
    // behind its start; here the least distance is zero.
    let near = select(1e-4, 0.0, skip.x != PT_NONE);
    if (limit <= near) {
        return PtHit(-1.0, 0u, 0.0, 0.0, 0u);
    }
    let flags = select(RAY_FLAG_NONE, RAY_FLAG_TERMINATE_ON_FIRST_HIT, any);
    var query: ray_query;
    rayQueryInitialize(&query, pt_scene, RayDesc(flags, 0xffu, near, limit, origin, dir));
    while (rayQueryProceed(&query)) {
        let c = rayQueryGetCandidateIntersection(&query);
        if (c.kind != RAY_QUERY_INTERSECTION_TRIANGLE) {
            continue;
        }
        let inst = c.instance_custom_data;
        let index = pt_instances[inst].first + c.primitive_index;
        if (inst == skip.x && index == skip.y) {
            continue;
        }
        if (pt_admit(index, c.barycentrics.x, c.barycentrics.y, any)) {
            rayQueryConfirmIntersection(&query);
        }
    }
    let hit = rayQueryGetCommittedIntersection(&query);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return PtHit(-1.0, 0u, 0.0, 0.0, 0u);
    }
    let inst = hit.instance_custom_data;
    return PtHit(hit.t, pt_instances[inst].first + hit.primitive_index, hit.barycentrics.x, hit.barycentrics.y, inst);
}
