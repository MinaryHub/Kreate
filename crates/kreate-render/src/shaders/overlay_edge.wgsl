// The wireframe overlay.
//
// One instanced quad per *real mesh edge*, expanded in screen space. Not
// `PolygonMode::Line` (native-only, absent on Metal, fixed at one pixel, and it
// draws triangulation edges so every quad shows a diagonal), not `LineList`
// (line width is fixed at 1.0 in WebGPU with no anti-aliasing), and not a
// barycentric derivative trick (WGSL has no barycentric builtin, it still draws
// triangulation edges without a per-corner "is this a real edge" mask, and it
// cannot express per-edge state).
//
// Per-edge state is the deciding argument. A modeller has to show selected,
// seam, sharp, boundary and non-manifold edges differently, and here that is one
// `u32` fetch per instance — changing a selection writes a flags buffer and never
// touches geometry.
//
// The whole wireframe of a mesh is a single draw call: `draw(0..4, 0..edges)`.

struct EdgeGlobals {
    view_proj: mat4x4<f32>,
    // xy: viewport size in pixels. z: line width in pixels. w: relative depth
    // offset toward the viewer.
    params: vec4<f32>,
    // The three edge diagnostics, linear RGBA, from the shared palette.
    non_manifold_color: vec4<f32>,
    seam_color: vec4<f32>,
    sharp_color: vec4<f32>,
};

struct EdgeObject {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    selected_color: vec4<f32>,
    // x: boundary highlight, y: non-manifold highlight, z: object selected,
    // w: the rubber-band preview applies to this draw.
    flags_params: vec4<f32>,
    // x: draw only the selected edges — the wireframe overlay is off and the
    // mesh is being edited.
    show: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: EdgeGlobals;

// Group 1 is the *mesh*: geometry, shared by every object that references the
// datablock. A `vec3<f32>` inside a storage array has a 16-byte stride, so
// positions are declared flat and indexed by hand. That is worth the arithmetic:
// it is a third of the memory on the largest buffer in the system.
@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<storage, read> edge_verts: array<u32>;
@group(1) @binding(2) var<storage, read> edge_flags: array<u32>;

// Group 2 is the *draw*: the transform, bound by dynamic offset so two linked
// duplicates of one mesh each get their own.
@group(2) @binding(0) var<uniform> object: EdgeObject;

// Group 3 is the *drag*: what a rubber band in progress would select, read
// straight out of the bitset the compute reduction just wrote. No readback and
// no upload, which is why the highlight can follow the pointer — and why what
// the user sees during the drag is exactly what the commit will apply.

struct PreviewParams {
    // x: preview active. y: domain — 0 vert, 1 edge, 2 face. z: the drag removes
    // rather than adds. w: how many elements the bitset covers.
    mode: vec4<u32>,
};

@group(3) @binding(0) var<uniform> preview: PreviewParams;
@group(3) @binding(1) var<storage, read> preview_bits: array<u32>;

/// Whether the drag covers this element.
///
/// `applies` is per *draw*: a drag resolves elements of the one object being
/// edited, and the bitset is indexed by that mesh's numbering, so applying it to
/// a neighbour would light up whatever elements happened to share those indices.
fn previewed(index: u32, domain: u32, applies: f32) -> bool {
    if (preview.mode.x == 0u || applies < 0.5 || preview.mode.y != domain) {
        return false;
    }
    // A bit past the end would name an element the mesh no longer has: the
    // bitset outlives an edit by a frame or two.
    if (index >= preview.mode.w) {
        return false;
    }
    return (preview_bits[index >> 5u] & (1u << (index & 31u))) != 0u;
}

/// Whether an element should be drawn as selected, with the drag folded in.
///
/// A drag that adds shows its elements already selected; one that removes shows
/// them already gone. Previewing only the additions would make Ctrl+drag look
/// like it was doing nothing until the moment it took effect.
fn selected_with_preview(selected: bool, index: u32, domain: u32, applies: f32) -> bool {
    if (!previewed(index, domain, applies)) {
        return selected;
    }
    return preview.mode.z == 0u;
}


const FLAG_SELECTED: u32 = 1u;
const FLAG_SHARP: u32 = 8u;
const FLAG_SEAM: u32 = 16u;
const FLAG_BOUNDARY: u32 = 32u;
const FLAG_NON_MANIFOLD: u32 = 64u;
const FLAG_HIDDEN: u32 = 4u;

fn position_of(i: u32) -> vec3<f32> {
    let b = 3u * i;
    return vec3<f32>(positions[b], positions[b + 1u], positions[b + 2u]);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // Signed distance from the line's centre, in pixels.
    //
    // `linear` rather than the default perspective-correct interpolation: this is
    // a screen-space quantity, and interpolating it with perspective skews the
    // coverage along an edge whose two ends are at very different depths.
    @location(0) @interpolate(linear) across_px: f32,
    @location(1) @interpolate(flat) flags: u32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ei: u32,
) -> VsOut {
    var out: VsOut;
    out.flags = edge_flags[ei];
    // Hidden in edit mode: every corner at one point, so nothing rasterises.
    if ((out.flags & FLAG_HIDDEN) != 0u) {
        out.clip = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.across_px = 0.0;
        return out;
    }
    // The preview is folded into the flag rather than carried alongside it, so
    // everything downstream — the colour rules, the diagnostic states — treats a
    // previewed edge exactly as it treats a selected one.
    if (selected_with_preview(
        (out.flags & FLAG_SELECTED) != 0u,
        ei,
        1u,
        object.flags_params.w,
    )) {
        out.flags = out.flags | FLAG_SELECTED;
    } else {
        out.flags = out.flags & ~FLAG_SELECTED;
    }
    // Only the selection: every other edge collapses, as a hidden one does.
    if (object.show.x > 0.5 && (out.flags & FLAG_SELECTED) == 0u) {
        out.clip = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.across_px = 0.0;
        return out;
    }

    let a_world = object.model * vec4<f32>(position_of(edge_verts[2u * ei]), 1.0);
    let b_world = object.model * vec4<f32>(position_of(edge_verts[2u * ei + 1u]), 1.0);
    var a_clip = g.view_proj * a_world;
    var b_clip = g.view_proj * b_world;

    // Pull the line toward the viewer by a fraction of its own depth. A relative
    // offset rather than a constant bias, because a modeller spans millimetres to
    // kilometres in one scene and a constant would be invisible at one end and
    // gross at the other. Reversed-Z, so a larger z is nearer.
    a_clip.z *= 1.0 + g.params.w;
    b_clip.z *= 1.0 + g.params.w;

    let viewport = g.params.xy;
    let half_width = g.params.z * 0.5;
    // The quad is a pixel wider than the line on each side, leaving room for the
    // coverage to fall off. Sizing it to the line exactly — the obvious thing —
    // means almost no pixel centre lands near the middle of a thin quad, so the
    // line never reaches full opacity and reads as a faint smudge.
    let expand = half_width + 1.0;

    // Behind-the-camera endpoints would divide by a negative w and flip the
    // quad. Clamping is not a clip, but it keeps the geometry sane; a segment
    // straddling the near plane is properly a clipping problem and is left for
    // when it is worth solving.
    let a_w = max(a_clip.w, 1e-6);
    let b_w = max(b_clip.w, 1e-6);
    let a_screen = a_clip.xy / a_w * 0.5 * viewport;
    let b_screen = b_clip.xy / b_w * 0.5 * viewport;

    var along = b_screen - a_screen;
    let length_px = length(along);
    // An edge shorter than a pixel has no direction to expand along; picking one
    // keeps it visible as a dot rather than collapsing it.
    along = select(vec2<f32>(1.0, 0.0), along / length_px, length_px > 1e-5);
    let across = vec2<f32>(-along.y, along.x);

    // A triangle strip: (a, -1), (a, +1), (b, -1), (b, +1).
    let at_end = vi >= 2u;
    let side = select(-1.0, 1.0, (vi & 1u) == 1u);
    var clip = select(a_clip, b_clip, at_end);
    let w = select(a_w, b_w, at_end);

    // Screen-space offset back into clip space.
    let offset = across * side * expand;
    clip.x += offset.x / (0.5 * viewport.x) * w;
    clip.y += offset.y / (0.5 * viewport.y) * w;

    out.clip = clip;
    out.across_px = side * expand;
    return out;
}

fn edge_color(flags: u32) -> vec4<f32> {
    // Object-mode selection recolours the *ordinary* wireframe and leaves the
    // diagnostic states alone: a seam or a non-manifold edge has to stay
    // legible on a selected object, which is exactly when it is being worked on.
    //
    // Per draw rather than per mesh, because two linked duplicates share one
    // datablock and only one of them is selected.
    let base = mix(object.base_color, object.selected_color, object.flags_params.z);

    if ((flags & FLAG_SELECTED) != 0u) {
        return object.selected_color;
    }
    if ((flags & FLAG_NON_MANIFOLD) != 0u) {
        // Loud on purpose: a non-manifold edge is nearly always a mistake, and
        // one the user cannot otherwise see until an operation fails on it much
        // later.
        return g.non_manifold_color * object.flags_params.y
            + base * (1.0 - object.flags_params.y);
    }
    if ((flags & FLAG_SEAM) != 0u) {
        return g.seam_color;
    }
    if ((flags & FLAG_SHARP) != 0u) {
        return g.sharp_color;
    }
    if ((flags & FLAG_BOUNDARY) != 0u) {
        return base * (1.0 - object.flags_params.x)
            + vec4<f32>(1.0, 1.0, 1.0, 1.0) * object.flags_params.x;
    }
    return base;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Analytic anti-aliasing: a solid core with a one-pixel falloff, computed
    // from the distance in pixels. This is why the wireframe needs no MSAA, and
    // why a line thinner than a pixel fades rather than aliasing into a dotted
    // mess — which is what makes a dense mesh readable.
    let half_width = g.params.z * 0.5;
    let coverage = clamp(half_width + 0.5 - abs(in.across_px), 0.0, 1.0);
    if (coverage <= 0.002) {
        discard;
    }
    let color = edge_color(in.flags);
    return vec4<f32>(color.rgb, color.a * coverage);
}
