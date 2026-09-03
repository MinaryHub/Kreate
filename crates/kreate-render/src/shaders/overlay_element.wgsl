// Selected faces and vertex points.
//
// The other two thirds of element selection. Edges have their own shader because
// a screen-space line quad is a different construction; these two are here
// together because they share every uniform and differ only in how they build a
// primitive.
//
// Both read per-element flags, which is the whole design: changing a selection
// writes a flags buffer and never touches geometry. A renderer that rebuilds a
// mesh when the selection changes turns one click on a dense mesh into a
// two-hundred-millisecond stall.

struct ElementGlobals {
    view_proj: mat4x4<f32>,
    // xy: viewport size in pixels. z: point size in pixels. w: unused.
    params: vec4<f32>,
    // x: face layer depth offset. y: vertex layer depth offset. z: face fill
    // opacity. w: unused.
    layers: vec4<f32>,
};

struct ElementObject {
    model: mat4x4<f32>,
    selected_color: vec4<f32>,
    unselected_color: vec4<f32>,
    // x: whether the rubber-band preview applies to this draw.
    // y: overlay density fade for the vertex dots. The face fill ignores it —
    //    a fill does not become less legible as a mesh gets denser.
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: ElementGlobals;

// Group 1 is the *mesh*, shared with the id pass: the same geometry read the
// same way, so what is highlighted and what is pickable cannot drift apart.
@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<storage, read> indices: array<u32>;
@group(1) @binding(2) var<storage, read> tri_to_face: array<u32>;
@group(1) @binding(3) var<storage, read> edge_verts: array<u32>;
@group(1) @binding(4) var<storage, read> vert_to_render: array<u32>;
@group(1) @binding(5) var<storage, read> vert_flags: array<u32>;
@group(1) @binding(6) var<storage, read> face_flags: array<u32>;

@group(2) @binding(0) var<uniform> object: ElementObject;

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

fn position_of(i: u32) -> vec3<f32> {
    let b = 3u * i;
    return vec3<f32>(positions[b], positions[b + 1u], positions[b + 2u]);
}

fn offset_toward_viewer(clip: vec4<f32>, amount: f32) -> vec4<f32> {
    var out = clip;
    out.z *= 1.0 + amount;
    return out;
}

// --- selected faces ---------------------------------------------------------

struct FaceOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) selected: u32,
};

// Drawn non-indexed with the index buffer read as storage, for the same reason
// the id pass does it: `vertex_index` then counts triangle corners, so the face
// a triangle belongs to is a lookup rather than something the rasteriser would
// have to tell us — and WebGPU has no portable primitive index.
@vertex
fn vs_face(@builtin(vertex_index) vi: u32) -> FaceOut {
    var out: FaceOut;
    let face = tri_to_face[vi / 3u];
    let selected = selected_with_preview(
        (face_flags[face] & FLAG_SELECTED) != 0u,
        face,
        2u,
        object.flags.x,
    );
    out.selected = u32(selected);

    let world = object.model * vec4<f32>(position_of(indices[vi]), 1.0);
    // Overlay layer 1: nearer than the surface, further than the wireframe, so a
    // selected face's own edges still read on top of its fill.
    out.clip = offset_toward_viewer(g.view_proj * world, g.layers.x);
    return out;
}

@fragment
fn fs_face(in: FaceOut) -> @location(0) vec4<f32> {
    // Unselected faces are discarded rather than drawn transparent: a full-screen
    // pass of blended zero-alpha triangles costs the same bandwidth as one that
    // does something.
    if (in.selected == 0u) {
        discard;
    }
    // Translucent, so the shading underneath still reads. An opaque fill hides
    // the form of what is being worked on, which is the thing the user is
    // actually looking at.
    return vec4<f32>(object.selected_color.rgb, g.layers.z);
}

// --- vertex points ----------------------------------------------------------

struct PointOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) selected: u32,
    // Offset from the centre of the quad, in pixels.
    //
    // The vector, not its length: a quad interpolates corner values bilinearly,
    // and the length of an interpolated vector is not the interpolation of the
    // lengths. Interpolating a distance puts sqrt(2) * radius at the centre, and
    // every dot comes out as a ring.
    @location(1) @interpolate(linear) offset_px: vec2<f32>,
};

// One instanced quad per *mesh* vertex, a constant size on screen — a point
// sprite, not geometry. A dot that grows as the camera approaches stops reading
// as a handle and starts reading as a blob.
//
// Indexed through `vert_to_render` so a cube corner is one dot rather than the
// three render vertices the normal split produced.
@vertex
fn vs_point(@builtin(vertex_index) vi: u32, @builtin(instance_index) pi: u32) -> PointOut {
    var out: PointOut;
    let selected = selected_with_preview(
        (vert_flags[pi] & FLAG_SELECTED) != 0u,
        pi,
        0u,
        object.flags.x,
    );
    out.selected = u32(selected);

    let world = object.model * vec4<f32>(position_of(vert_to_render[pi]), 1.0);
    // Overlay layer 3, the nearest: a vertex sits on top of both the edges and
    // the faces meeting at it, or the handle disappears exactly where the user
    // needs to grab it.
    var clip = offset_toward_viewer(g.view_proj * world, g.layers.y);
    let w = max(clip.w, 1e-6);

    let viewport = g.params.xy;
    let radius = g.params.z * 0.5;
    // A quad from a triangle strip: (-1,-1), (+1,-1), (-1,+1), (+1,+1). Expanded
    // by a pixel so the coverage has room to fall off; sized to the dot exactly,
    // almost no pixel centre lands near the middle of a small quad and the dot
    // never reaches full opacity.
    let expand = radius + 1.0;
    let corner = vec2<f32>(
        select(-1.0, 1.0, (vi & 1u) == 1u),
        select(-1.0, 1.0, vi >= 2u),
    );
    clip.x += corner.x * expand / (0.5 * viewport.x) * w;
    clip.y += corner.y * expand / (0.5 * viewport.y) * w;

    out.clip = clip;
    out.offset_px = corner * expand;
    return out;
}

@fragment
fn fs_point(in: PointOut) -> @location(0) vec4<f32> {
    // Analytic anti-aliasing on a disc: a solid core with a one-pixel falloff.
    // This is why the overlays need no MSAA — which matters more than it sounds,
    // because WebGPU cannot resolve a multisampled depth texture and every
    // depth-based effect would need a second pass at 1x.
    let radius = g.params.z * 0.5;
    let coverage = clamp(radius + 0.5 - length(in.offset_px), 0.0, 1.0);
    if (coverage <= 0.002) {
        discard;
    }
    let color = select(object.unselected_color, object.selected_color, in.selected != 0u);
    // `flags.y` is the density fade: below about a pixel of screen per vertex a
    // field of dots is an unreadable grey smear that hides the shading it is
    // supposed to annotate, so it fades out. The draw is skipped entirely once
    // this reaches zero — see the draw loop — so what arrives here is always a
    // partial fade.
    return vec4<f32>(color.rgb, color.a * coverage * object.flags.y);
}
