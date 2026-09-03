// Cluster culling: one thread per cluster, survivors appended to a list.
//
// The output is deliberately not a per-cluster visibility flag but a *compacted
// list* plus a count, because the count is written straight into the instance
// field of a `draw_indirect` argument buffer. That is what turns "which clusters
// are visible" into a draw with no readback, no CPU involvement and no stall:
// the GPU decides and the GPU draws.
//
// Three rejections, cheapest first. Each is exact and safe — none of them can
// discard a cluster that has a visible triangle, which matters far more here
// than in a game: a hole in the model reads as corrupt geometry, and the user
// will believe their mesh is broken rather than the renderer.

struct Cluster {
    // Object-space bounding sphere: centre xyz, radius w.
    sphere: vec4<f32>,
    // Normal cone: axis xyz, cosine cutoff w.
    cone: vec4<f32>,
    index_offset: u32,
    triangle_count: u32,
    lod: u32,
    _pad: u32,
};

struct Globals {
    // Camera-relative: projection times the view *rotation*, with no
    // translation, because the sphere centres arriving here are already relative
    // to the eye. Mixing an absolute matrix with relative positions would put
    // every cluster in the wrong place and cull almost all of them.
    view_proj: mat4x4<f32>,
    // Inward-facing frustum planes, xyz = normal, w = distance. Five are in use:
    // the renderer's far plane is at infinity, so the sixth is degenerate and
    // the CPU fills the slot with a plane that accepts everything rather than
    // leaving it zeroed — a zero plane rejects the entire scene.
    planes: array<vec4<f32>, 6>,
    // x: viewport height in device pixels. y: diameter in pixels below which a
    // cluster is too small to see. z: tangent of half the vertical field of
    // view. w: 1 for perspective, 0 for orthographic.
    params: vec4<f32>,
    // x: orthographic view height, unused under perspective.
    ortho: vec4<f32>,
    // x, y: the pyramid's level-0 size in texels. z: how many levels it has.
    // w: 1 to test occlusion, 0 to skip it.
    hiz: vec4<f32>,
};

struct DrawArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

struct Instance {
    // Camera-relative model matrix. Camera-relative because everything handed
    // to the GPU is: a scene ten kilometres from the origin has visibly less
    // float precision than the geometry deserves, and the cull has to agree
    // with the draw about where things are or it culls the wrong ones.
    model: mat4x4<f32>,
    // Inverse transpose of the linear part, for the cone axis. A direction does
    // not transform like a position under non-uniform scale, and getting this
    // wrong makes half a squashed model disappear.
    normal: mat3x3<f32>,
    // x: how many clusters this instance owns. y: where its slice of the
    // visible list starts. z: the largest scale factor in `model`, so the
    // bounding sphere's radius can be scaled without decomposing the matrix
    // here. w: unused.
    counts: vec4<u32>,
    scale: vec4<f32>,
};

// Group 0 belongs to the viewport: the shared frustum, the shared visible list,
// and the per-draw instance and argument slots reached by dynamic offset. Group
// 1 belongs to the mesh, because clusters are baked from geometry and geometry
// is shared between linked duplicates — one cluster buffer, many instances.
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<uniform> instance: Instance;
@group(0) @binding(2) var<storage, read_write> visible: array<u32>;
@group(0) @binding(3) var<storage, read_write> args: DrawArgs;

@group(0) @binding(4) var pyramid: texture_2d<f32>;

@group(1) @binding(0) var<storage, read> clusters: array<Cluster>;

// Whether any part of a sphere is inside the frustum.
//
// Conservative in the safe direction: a sphere straddling a corner can pass all
// six plane tests while lying outside. That costs a wasted cluster, where the
// other kind of error costs a hole.
fn inside_frustum(centre: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = g.planes[i];
        if (dot(plane.xyz, centre) + plane.w < -radius) {
            return false;
        }
    }
    return true;
}

// Whether every triangle in the cluster faces away from the eye.
//
// meshoptimizer's formula, in the form that uses the bounding sphere rather
// than the cone apex:
//
//     dot(centre - eye, axis) >= cutoff * length(centre - eye) + radius
//
// Chosen over the apex form because the sphere is already here for frustum
// culling, and because this form has no singularity when the eye is at the
// centre — a real case, since the camera can be inside a mesh.
//
// Camera-relative, so the eye is the origin and the vector to the centre is the
// centre itself.
fn is_backfacing(cluster: Cluster, centre: vec3<f32>, radius: f32) -> bool {
    let axis = instance.normal * cluster.cone.xyz;
    let length_squared = dot(axis, axis);
    if (length_squared < 1e-12) {
        // A degenerate cone claims no direction, so it can never be proven
        // backfacing. Resolving towards drawing.
        return false;
    }
    let unit = axis / sqrt(length_squared);
    return dot(centre, unit) >= cluster.cone.w * length(centre) + radius;
}

// The cluster's diameter in device pixels.
//
// The same arithmetic as the object-level screen-size cull on the CPU, and for
// the same reason: something under a couple of pixels across cannot be seen and
// cannot be clicked, so skipping it loses the user nothing.
fn diameter_px(centre: vec3<f32>, radius: f32) -> f32 {
    let height = g.params.x;
    if (g.params.w < 0.5) {
        let view_height = g.ortho.x;
        if (view_height <= 0.0) {
            return height;
        }
        return 2.0 * radius / view_height * height;
    }
    let distance = length(centre);
    // The eye is inside the sphere, so it covers the screen. Also the guard
    // against dividing by zero.
    if (distance <= radius) {
        return height;
    }
    let half_extent = distance * g.params.z;
    if (half_extent <= 0.0) {
        return height;
    }
    return radius / half_extent * height;
}

// Whether a cluster is entirely behind what has already been drawn.
//
// The deepest of the three rejections and the one that matters most in a real
// scene: frustum and cone rejection only remove what is off screen and what
// faces away, while in a scene with any depth complexity the largest share of
// clusters is simply *behind something*.
//
// The test, in one line: a cluster is hidden when the farthest thing already
// drawn anywhere in the screen region it covers is still nearer than the
// cluster's own nearest point.
//
// Every approximation below errs towards *keeping* the cluster. That is not
// symmetry — an editor that hides geometry the user can see has told a lie about
// what exists, and a hole in a model reads as corrupt data rather than as a
// renderer bug. A cluster wrongly kept costs a fraction of a millisecond.
fn is_occluded(centre: vec3<f32>, radius: f32) -> bool {
    if (g.hiz.w < 0.5) {
        return false;
    }

    // The screen extent and nearest depth of the box enclosing the sphere.
    //
    // The box rather than the sphere's true silhouette: it is strictly larger,
    // so its screen region is larger and the farthest depth found in it is
    // smaller, which culls less. And because a perspective projection of a
    // convex body wholly in front of the eye is convex, the extremes over the
    // box are attained at its corners — so eight points bound it exactly.
    var lo = vec2<f32>(1e30, 1e30);
    var hi = vec2<f32>(-1e30, -1e30);
    var nearest = -1e30;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let offset = vec3<f32>(
            select(-radius, radius, (i & 1u) != 0u),
            select(-radius, radius, (i & 2u) != 0u),
            select(-radius, radius, (i & 4u) != 0u),
        );
        let clip = g.view_proj * vec4<f32>(centre + offset, 1.0);
        // The box crosses the plane through the eye, so the projection folds and
        // a screen extent is meaningless. Kept rather than guessed.
        if (clip.w <= 1e-6) {
            return false;
        }
        let ndc = clip.xyz / clip.w;
        lo = min(lo, ndc.xy);
        hi = max(hi, ndc.xy);
        // Reversed-Z: larger is nearer, so the cluster's nearest point is the
        // largest depth over the box. Taking the box's nearest rather than the
        // sphere's makes `nearest` larger than the truth, which makes the
        // comparison below harder to satisfy — again the safe direction.
        nearest = max(nearest, ndc.z);
    }

    // NDC to texture coordinates. Y flips: NDC has +Y up, a texture has +V down.
    let uv_lo = clamp(
        vec2<f32>(lo.x, -hi.y) * 0.5 + 0.5,
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );
    let uv_hi = clamp(
        vec2<f32>(hi.x, -lo.y) * 0.5 + 0.5,
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );

    // Pick the level where the cluster spans about one texel, so four samples
    // cover it whatever its size on screen. That is the entire reason for the
    // pyramid: the cost of the test does not grow with the cluster.
    //
    // Rounding the level *up* picks a coarser level, whose texels each summarise
    // more of the screen and so report a farther depth. Conservative.
    let span = (uv_hi - uv_lo) * g.hiz.xy;
    let level = clamp(
        ceil(log2(max(max(span.x, span.y), 1.0))),
        0.0,
        g.hiz.z - 1.0,
    );
    let li = i32(level);
    let level_size = max(floor(g.hiz.xy / exp2(level)), vec2<f32>(1.0, 1.0));
    let last = level_size - vec2<f32>(1.0, 1.0);
    let t_lo = clamp(floor(uv_lo * level_size), vec2<f32>(0.0, 0.0), last);
    let t_hi = clamp(floor(uv_hi * level_size), vec2<f32>(0.0, 0.0), last);

    var farthest = textureLoad(pyramid, vec2<i32>(i32(t_lo.x), i32(t_lo.y)), li).r;
    farthest = min(
        farthest,
        textureLoad(pyramid, vec2<i32>(i32(t_hi.x), i32(t_lo.y)), li).r,
    );
    farthest = min(
        farthest,
        textureLoad(pyramid, vec2<i32>(i32(t_lo.x), i32(t_hi.y)), li).r,
    );
    farthest = min(
        farthest,
        textureLoad(pyramid, vec2<i32>(i32(t_hi.x), i32(t_hi.y)), li).r,
    );

    // Strictly greater, so a cluster exactly coplanar with what is in front of
    // it survives. Ties happen constantly — a cluster tested against the depth
    // it wrote itself last frame — and rejecting them would make surfaces
    // flicker.
    return farthest > nearest;
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= instance.counts.x) {
        return;
    }
    let cluster = clusters[index];

    let centre = (instance.model * vec4<f32>(cluster.sphere.xyz, 1.0)).xyz;
    let radius = cluster.sphere.w * instance.scale.x;

    if (!inside_frustum(centre, radius)) {
        return;
    }
    if (is_backfacing(cluster, centre, radius)) {
        return;
    }
    if (diameter_px(centre, radius) < g.params.y) {
        return;
    }
    // Last, because it is the most expensive: eight matrix-vector products and
    // four texture loads, against three dot products for the tests above.
    if (is_occluded(centre, radius)) {
        return;
    }

    // The count *is* the indirect draw's instance field, incremented in place.
    // A separate pass to copy a counter into the arguments would be one more
    // dispatch and one more chance for the two to disagree.
    let slot = atomicAdd(&args.instance_count, 1u);
    visible[instance.counts.y + slot] = index;
}
