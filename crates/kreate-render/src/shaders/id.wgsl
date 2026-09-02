// The id pass: what the user would click, written as ids instead of colour.
//
// Rendered into `Rg32Uint` — `.r` is the object id plus one so that zero means
// "nothing", `.g` is the element id. Sixty-four bits, no packing tricks, and no
// id-space exhaustion at a hundred million elements.
//
// Three things make id-buffer picking feel better than ray casting, and all
// three are here rather than in the code that reads the result:
//
// - **Occlusion is free and exact.** The pass tests against the depth buffer the
//   visible image was drawn with, so what looks clickable is what is clickable.
// - **The pick tolerance is in screen space.** Vertices and edges are drawn
//   *expanded* to a pick radius — larger than they appear — so a near miss is a
//   hit, uniformly, at every zoom level and with no per-caller tolerance.
// - **The layering matches the overlays.** The same relative view-space depth
//   offset is applied here as in the visible overlay, so the z order in the id
//   buffer is the z order on screen.
//
// The three element domains are separate entry points sharing one fragment
// shader. Only one is drawn per pass: the domain being selected in. Drawing all
// three would need a channel to say which domain a pixel belongs to, and a
// modeller in vertex mode is not picking faces anyway.

struct IdGlobals {
    view_proj: mat4x4<f32>,
    // xy: viewport size in pixels. z: pick radius in pixels. w: relative depth
    // offset toward the viewer.
    params: vec4<f32>,
};

struct IdObject {
    model: mat4x4<f32>,
    // x: object id plus one. Zero is reserved for "nothing here".
    ids: vec4<u32>,
};

@group(0) @binding(0) var<uniform> g: IdGlobals;

// Group 1 is the *mesh*: geometry, shared by every object referencing the
// datablock. Positions are a flat `array<f32>` because a `vec3<f32>` in a
// storage array has a 16-byte stride — a third of the memory on the largest
// buffer in the system.
@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<storage, read> indices: array<u32>;
@group(1) @binding(2) var<storage, read> tri_to_face: array<u32>;
@group(1) @binding(3) var<storage, read> edge_verts: array<u32>;
@group(1) @binding(4) var<storage, read> vert_to_render: array<u32>;

// Group 2 is the *draw*, bound by dynamic offset so two linked duplicates of one
// mesh each get their own transform and their own object id.
@group(2) @binding(0) var<uniform> object: IdObject;

fn position_of(i: u32) -> vec3<f32> {
    let b = 3u * i;
    return vec3<f32>(positions[b], positions[b + 1u], positions[b + 2u]);
}

/// Pulls a clip position toward the viewer by a fraction of its own depth.
///
/// Relative, not a constant bias: a modelling scene spans millimetres to
/// kilometres and a constant offset is invisible at one end and gross at the
/// other. Reversed-Z, so a larger z is nearer.
fn offset_toward_viewer(clip: vec4<f32>) -> vec4<f32> {
    var out = clip;
    out.z *= 1.0 + g.params.w;
    return out;
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) element: u32,
    // Offset from the centre of the expanded quad, in pixels.
    //
    // The *vector* rather than its length: a quad interpolates its corner values
    // bilinearly, and the length of an interpolated vector is not the
    // interpolation of the lengths. Interpolating a distance would put
    // sqrt(2) * radius at the centre of a vertex quad — every dot would be a
    // ring with its middle discarded.
    //
    // `linear` rather than the default perspective-correct interpolation: this
    // is a screen-space quantity, and perspective would skew it along an edge
    // whose ends are at very different depths.
    @location(1) @interpolate(linear) offset_px: vec2<f32>,
};

// --- faces ------------------------------------------------------------------

// Drawn non-indexed with the index buffer read as storage, so that
// `vertex_index` counts triangle corners and the face id is `tri_to_face[vi /
// 3]`. With an indexed draw `vertex_index` is the *vertex* fetched from the
// index buffer, which says nothing about which triangle is being drawn, and
// there is no portable primitive index in WebGPU.
//
// The alternative — a de-indexed vertex stream carrying a face id per corner —
// costs 12 bytes per triangle corner on every mesh in the scene. This costs one
// storage fetch and no memory at all.
@vertex
fn vs_face(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    out.element = tri_to_face[vi / 3u];
    out.offset_px = vec2<f32>(0.0, 0.0);
    let world = object.model * vec4<f32>(position_of(indices[vi]), 1.0);
    out.clip = offset_toward_viewer(g.view_proj * world);
    return out;
}

// --- edges ------------------------------------------------------------------

// One instanced quad per real mesh edge, expanded to the pick radius. The same
// construction as the visible wireframe, but wider: the tolerance being a little
// larger than the drawn line is what turns a near miss into a hit.
@vertex
fn vs_edge(@builtin(vertex_index) vi: u32, @builtin(instance_index) ei: u32) -> VsOut {
    var out: VsOut;
    out.element = ei;

    let a_world = object.model * vec4<f32>(position_of(edge_verts[2u * ei]), 1.0);
    let b_world = object.model * vec4<f32>(position_of(edge_verts[2u * ei + 1u]), 1.0);
    var a_clip = offset_toward_viewer(g.view_proj * a_world);
    var b_clip = offset_toward_viewer(g.view_proj * b_world);

    let viewport = g.params.xy;
    let radius = g.params.z;

    // A point behind the camera divides by a negative w and flips the quad.
    // Clamping is not clipping, but it keeps the geometry sane; an edge
    // straddling the near plane is properly a clipping problem.
    let a_w = max(a_clip.w, 1e-6);
    let b_w = max(b_clip.w, 1e-6);
    let a_screen = a_clip.xy / a_w * 0.5 * viewport;
    let b_screen = b_clip.xy / b_w * 0.5 * viewport;

    var along = b_screen - a_screen;
    let length_px = length(along);
    along = select(vec2<f32>(1.0, 0.0), along / length_px, length_px > 1e-5);
    let across = vec2<f32>(-along.y, along.x);

    // A triangle strip: (a, -1), (a, +1), (b, -1), (b, +1).
    let at_end = vi >= 2u;
    let side = select(-1.0, 1.0, (vi & 1u) == 1u);
    var clip = select(a_clip, b_clip, at_end);
    let w = select(a_w, b_w, at_end);

    let offset = across * side * radius;
    clip.x += offset.x / (0.5 * viewport.x) * w;
    clip.y += offset.y / (0.5 * viewport.y) * w;

    out.clip = clip;
    out.offset_px = vec2<f32>(side * radius, 0.0);
    return out;
}

// --- vertices ---------------------------------------------------------------

// One instanced quad per *mesh* vertex, a constant size on screen. Indexed
// through `vert_to_render` so that a cube corner is one id rather than the three
// render vertices the normal split produced.
@vertex
fn vs_vert(@builtin(vertex_index) vi: u32, @builtin(instance_index) pi: u32) -> VsOut {
    var out: VsOut;
    out.element = pi;

    let world = object.model * vec4<f32>(position_of(vert_to_render[pi]), 1.0);
    var clip = offset_toward_viewer(g.view_proj * world);
    let w = max(clip.w, 1e-6);

    let viewport = g.params.xy;
    let radius = g.params.z;
    // A quad from a triangle strip: (-1,-1), (+1,-1), (-1,+1), (+1,+1).
    let corner = vec2<f32>(
        select(-1.0, 1.0, (vi & 1u) == 1u),
        select(-1.0, 1.0, vi >= 2u),
    );
    clip.x += corner.x * radius / (0.5 * viewport.x) * w;
    clip.y += corner.y * radius / (0.5 * viewport.y) * w;

    out.clip = clip;
    out.offset_px = corner * radius;
    return out;
}

// --- shared fragment --------------------------------------------------------

@fragment
fn fs_id(in: VsOut) -> @location(0) vec2<u32> {
    // Round vertex quads off into discs. Without it the corners of the square
    // extend the tolerance by 40% diagonally, which reads as the pick region
    // being lopsided — the sort of thing a user feels and cannot describe.
    if (length(in.offset_px) > g.params.z) {
        discard;
    }
    return vec2<u32>(object.ids.x, in.element);
}
