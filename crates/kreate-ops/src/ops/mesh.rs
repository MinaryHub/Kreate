//! Edit-mode mesh operators.
//!
//! Each one is a thin wrapper: find the mesh, cross into [`BMesh`], call the
//! topology routine in `kreate-mesh`, cross back. All the modelling logic lives
//! in the kernel, where it can be tested without a document, and all the
//! bookkeeping lives here, where it can be tested without geometry.
//!
//! # The crossing is temporary
//!
//! Converting the whole mesh on each operation is O(V + E + F) per edit, which
//! is right for correctness and wrong for a 10 M-vertex mesh under an
//! interactive tool. The eventual shape is an edit *session* that holds one
//! `BMesh` for as long as the user stays in edit mode, with these operators
//! acting on it directly. The operator signatures do not change when that lands
//! — only what sits behind [`with_bmesh`].

use kreate_mesh::arena::VertId;
use kreate_mesh::bmesh::{BMesh, SelectMode};
use kreate_mesh::edit;
use kreate_mesh::selection;
use kreate_mesh::ArenaIndex;
use kreate_scene::document::{MeshId, Object, ObjectData};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

/// The mesh an edit-mode operator acts on.
///
/// The active object's, falling back to the first selected one — which is what
/// the user means when they have selected several and reached for a tool.
pub fn target_mesh(doc: &Document) -> Option<MeshId> {
    let active = doc
        .selection
        .active
        .or_else(|| doc.selection.objects.first().copied())?;
    doc.object(active)?.mesh()
}

/// The object whose space an edit-mode operator works in.
///
/// The same choice `target_mesh` makes, and it has to be the same one: the
/// transform this returns is the one that maps that mesh's vertices into the
/// world.
pub fn target_object(doc: &Document) -> Option<kreate_scene::document::ObjectId> {
    doc.selection
        .active
        .or_else(|| doc.selection.objects.first().copied())
}

/// One mesh in edit mode, and the object whose space it is in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EditTarget {
    pub object: kreate_scene::document::ObjectId,
    pub mesh: MeshId,
    /// Object to world. Held rather than looked up again, because working it
    /// out walks the parent chain and every vertex of the mesh needs it.
    pub to_world: glam::DAffine3,
}

/// Every mesh edit mode is open on.
///
/// # Why edit mode is a set
///
/// A modeller selects a body and its two arms and moves a shared seam; doing
/// that one object at a time means three drags that have to agree, and they
/// will not. So the transform operators walk this list, and the pivot is the
/// median of every selected vertex in *all* of them — which is the difference
/// between one transform and three: three separate medians is three separate
/// rotations, and the pieces come apart.
///
/// # Each datablock once
///
/// Deduplicated by [`MeshId`], and that is a correctness requirement rather
/// than an efficiency one. Two linked duplicates share one `MeshData`: visiting
/// it twice applies the transform twice, so a one-unit drag moves the vertices
/// two units and a rotation turns them through double the angle. The second
/// object's own transform is then unused, and the mesh moves in the first
/// one's space — which is the same answer Blender gives, and the only
/// self-consistent one available when one set of vertices is asked to live in
/// two spaces at once.
///
/// # Order
///
/// The active object first, then the selection in its own order. So
/// [`target_mesh`] is this list's first entry by construction, and an operator
/// that still works on one mesh works on the one the user would name.
///
/// Linear in the selection squared, because the seen-set is a `Vec` scan. At
/// the scale anyone puts into edit mode together — tens of objects — that is
/// nothing, and it is paid once per operator rather than per vertex.
pub fn edit_targets(doc: &Document) -> Vec<EditTarget> {
    let mut out: Vec<EditTarget> = Vec::new();
    let order = doc
        .selection
        .active
        .into_iter()
        .chain(doc.selection.objects.iter().copied());
    for id in order {
        let Some(mesh) = doc.object(id).and_then(|o| o.mesh()) else {
            continue;
        };
        if out.iter().any(|t| t.mesh == mesh) {
            continue;
        }
        out.push(EditTarget {
            object: id,
            mesh,
            to_world: doc.world_transform(id),
        });
    }
    out
}

/// Runs `f` over one named mesh in edit form and writes the result back.
///
/// Takes the coarse undo path: every one of these changes topology, and a
/// declared write set over positions cannot describe a mesh that has grown
/// faces.
pub(crate) fn with_bmesh_id<T>(
    doc: &mut DocMut<'_>,
    mesh_id: MeshId,
    f: impl FnOnce(&mut BMesh) -> Result<T, OpError>,
) -> Result<T, OpError> {
    // The edit-mode structure, when edit mode is open and still agrees with the
    // document. Held across operators, so the decode happens once on entering
    // edit mode rather than once per extrude — and now for every mesh in edit
    // mode rather than only the active one.
    if doc.edit_bmesh(mesh_id).is_some() {
        let out = {
            let bm = doc
                .edit_bmesh(mesh_id)
                .expect("checked on the line above, and nothing has borrowed since");
            f(bm)?
        };
        if !doc.bake_edit(mesh_id) {
            return Err(OpError::Failed("mesh vanished mid-edit".into()));
        }
        return Ok(out);
    }

    // No session: convert for this call. The path a script, a test, or an
    // operator run outside edit mode takes.
    let mut bm = doc
        .document()
        .mesh(mesh_id)
        .ok_or_else(|| OpError::Failed("no mesh to edit".into()))?
        .to_bmesh();

    let out = f(&mut bm)?;

    let baked = bm.to_mesh_data();
    let d = doc.document_mut();
    let slot = d
        .mesh_mut(mesh_id)
        .ok_or_else(|| OpError::Failed("mesh vanished mid-edit".into()))?;
    *slot = baked;
    Ok(out)
}

/// Runs `f` over the *target* mesh in edit form and writes the result back.
///
/// For the operators that name an element by index — a loop cut names the edge
/// the pointer was over — and can therefore only mean one mesh: an index is in
/// one mesh's numbering, and handing it to another names a different edge.
pub(crate) fn with_bmesh<T>(
    doc: &mut DocMut<'_>,
    f: impl FnOnce(&mut BMesh) -> Result<T, OpError>,
) -> Result<T, OpError> {
    let mesh_id =
        target_mesh(doc.document()).ok_or_else(|| OpError::Failed("no mesh to edit".into()))?;
    with_bmesh_id(doc, mesh_id, f)
}

/// Runs `f` over every mesh in edit mode in edit form, writing each back.
///
/// What a selection-driven topology operator wants: extruding with a body and
/// its two arms in edit mode extrudes the selected faces of all three, and a
/// mesh with nothing selected is a call in which `f` finds nothing to do.
///
/// Returns what `f` said for each mesh, in [`edit_targets`] order, so the
/// caller folds the answers its own way — an extrude asks whether any mesh
/// grew, a delete adds up what was removed.
///
/// # Each mesh baked as it goes
///
/// Not all decoded, then all edited, then all baked. A bake writes the whole
/// mesh back and takes the coarse undo snapshot, and holding several structures
/// mid-edit would mean holding several `BMesh`es — two hundred bytes a vertex —
/// at once. One at a time costs nothing extra: the undo snapshot is taken on
/// the first bake and the rest are free.
pub(crate) fn each_bmesh<T>(
    doc: &mut DocMut<'_>,
    mut f: impl FnMut(&mut BMesh) -> Result<T, OpError>,
) -> Result<Vec<T>, OpError> {
    let targets = edit_targets(doc.document());
    if targets.is_empty() {
        return Err(OpError::Failed("no mesh to edit".into()));
    }
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        out.push(with_bmesh_id(doc, target.mesh, &mut f)?);
    }
    Ok(out)
}

/// Runs a change against the document mesh, without building a [`BMesh`].
///
/// The path selection takes. A `BMesh` round trip rebuilds every array, which
/// costs a decode and an encode of the whole mesh — and, worse, leaves the
/// renderer nothing to compare: it decides what to re-upload by pointer identity
/// per array, so a rebuilt mesh means a click on a million-vertex model
/// re-uploads all of it. Writing the selection layers in place leaves the
/// geometry arrays untouched and shared, which is what makes a click three small
/// buffer writes.
///
/// The closure reports whether anything changed, so an operator that decides
/// there was nothing to do returns `Cancelled` without a history step.
fn with_mesh<T>(
    doc: &mut DocMut<'_>,
    f: impl FnOnce(&mut kreate_mesh::MeshData) -> T,
) -> Result<T, OpError> {
    let mesh_id =
        target_mesh(doc.document()).ok_or_else(|| OpError::Failed("no mesh to edit".into()))?;
    let mesh = doc
        .document_mut()
        .mesh_mut(mesh_id)
        .ok_or_else(|| OpError::Failed("no mesh to edit".into()))?;
    Ok(f(mesh))
}

/// Runs `f` over every mesh in edit mode, and says whether any said yes.
///
/// The selection counterpart of [`run_transform`]'s loop. A selection change is
/// a layer write — no decode, no encode, and the geometry arrays stay shared —
/// so this is as cheap per mesh as the single-mesh version was, and "select
/// all" with three objects in edit mode means all of all three.
///
/// `f` is told which target it is looking at, because a caller may hold
/// something that belongs to one of them: the edit session's adjacency map is
/// the *target* mesh's, and handing it to another would be indexing one mesh's
/// edges with another's numbering.
fn for_each_edit_mesh(
    doc: &mut DocMut<'_>,
    mut f: impl FnMut(&EditTarget, &mut kreate_mesh::MeshData) -> bool,
) -> Result<bool, OpError> {
    let targets = edit_targets(doc.document());
    if targets.is_empty() {
        return Err(OpError::Failed("no mesh to edit".into()));
    }
    let mut any = false;
    for target in targets {
        let Some(mesh) = doc.document_mut().mesh_mut(target.mesh) else {
            continue;
        };
        any |= f(&target, mesh);
    }
    Ok(any)
}

/// Runs `f` over one named mesh, whichever of the edit-mode set it is.
fn with_named_mesh<T>(
    doc: &mut DocMut<'_>,
    mesh: MeshId,
    f: impl FnOnce(&mut kreate_mesh::MeshData) -> T,
) -> Result<T, OpError> {
    let data = doc
        .document_mut()
        .mesh_mut(mesh)
        .ok_or_else(|| OpError::Failed("no mesh to edit".into()))?;
    Ok(f(data))
}

/// The corner-to-edge map for the mesh being edited, from the session.
///
/// `None` outside edit mode, or in a script, or in a test: those recompute it
/// and pay a pass over the mesh, which is correct and rare. Inside edit mode it
/// is what keeps a click from rebuilding a sorted table of every edge — four
/// hundred milliseconds on a million faces, for a flush that takes two.
fn edit_adjacency(doc: &mut DocMut<'_>) -> Option<std::sync::Arc<Vec<u32>>> {
    let mesh = target_mesh(doc.document())?;
    doc.edit_corner_edges(mesh)
}

/// Flushes a selection, using the session's map when there is one.
fn flush(
    mesh: &mut kreate_mesh::MeshData,
    mode: SelectMode,
    adjacency: &Option<std::sync::Arc<Vec<u32>>>,
) {
    match adjacency {
        Some(map) => selection::flush_with(mesh, mode, map),
        None => selection::flush(mesh, mode),
    }
}

/// Whether nothing at all is selected, in any mesh in edit mode.
///
/// Several operators fall back to the whole mesh when the selection is empty —
/// recalculating normals or shading a model flat is something a user does to
/// something they have just imported, and making them select everything first
/// is friction with no purpose.
///
/// With edit mode holding several meshes that question has to be asked of all
/// of them and not of each. Asked per mesh, selecting a few faces in one object
/// would silently rewrite the normals of every *other* object in edit mode —
/// which is the opposite of what selecting something means.
fn nothing_selected_anywhere(doc: &Document) -> bool {
    edit_targets(doc)
        .iter()
        .all(|t| doc.mesh(t.mesh).is_none_or(|m| !selection::any_selected(m)))
}

/// True when there is a mesh to edit at all.
pub(crate) fn can_edit(ctx: &Context, doc: &Document) -> bool {
    ctx.mode.edits_elements() && target_mesh(doc).is_some()
}

// --- transforming a selection -----------------------------------------------

/// Every vertex a selection touches, whichever element type it was made in.
///
/// The union of the three domains: selected vertices, both ends of every
/// selected edge, and every corner of every selected face. One rule for all
/// three, rather than a branch on the current select mode, and that is
/// deliberate — an edge is its two vertices and a face is its ring, so moving a
/// face *is* moving its vertices. Writing it once means a transform cannot
/// behave differently in one mode by accident, and it keeps working when a
/// selection spans modes.
///
/// De-duplicated, which matters more than it looks: two adjacent selected faces
/// share vertices, and applying a translation twice to a shared vertex would
/// pull it twice as far as the rest of the face.
pub fn selected_vertices(bm: &BMesh) -> Vec<VertId> {
    let mut verts: Vec<VertId> = bm.selected_verts();
    for e in bm.selected_edges() {
        if let Some([a, b]) = bm.edge_verts(e) {
            verts.push(a);
            verts.push(b);
        }
    }
    for f in bm.selected_faces() {
        verts.extend(bm.face_verts(f));
    }
    verts.sort_unstable_by_key(|v| v.index());
    verts.dedup_by_key(|v| v.index());
    verts
}

/// Every vertex a `MeshData` selection touches, as plain indices.
///
/// The same union rule as [`selected_vertices`], against the stored mesh rather
/// than its edit form. Exposed because the application needs the selection's
/// median to place a transform's pivot, and converting to a `BMesh` to ask would
/// be an O(V + E + F) rebuild per pointer event.
pub fn selection_vertices(mesh: &kreate_mesh::MeshData) -> Vec<u32> {
    use kreate_mesh::attribute::names;
    let attrs = mesh.attributes();
    let mut verts: Vec<u32> = Vec::new();
    if let Some(flags) = attrs.get_bool(names::SELECT_VERT) {
        verts.extend(
            flags
                .iter()
                .enumerate()
                .filter(|(_, on)| **on)
                .map(|(i, _)| i as u32),
        );
    }
    if let Some(flags) = attrs.get_bool(names::SELECT_EDGE) {
        for (edge, on) in mesh.edges().iter().zip(flags) {
            if *on {
                verts.push(edge[0]);
                verts.push(edge[1]);
            }
        }
    }
    if let Some(flags) = attrs.get_bool(names::SELECT_FACE) {
        for (face, on) in (0..mesh.num_faces()).zip(flags) {
            if *on {
                verts.extend_from_slice(mesh.face_verts(face));
            }
        }
    }
    verts.sort_unstable();
    verts.dedup();
    verts
}

// --- transform --------------------------------------------------------------
//
// # Why a transform does not cross into `BMesh`
//
// Every other operator in this file changes topology, and the crossing is what
// pays for the adjacency it needs. A transform needs none: it moves positions
// and leaves the connectivity exactly as it found it.
//
// Crossing anyway costs an encode of the whole mesh per pointer event, and it
// costs it twice over. Once directly — **95 ms at two hundred thousand
// vertices, measured, against a budget of one**. And once indirectly, because a
// re-encode hands back arrays the renderer has never seen, so geometry that did
// not change is re-uploaded. Writing through [`DocMut::edit_positions`] leaves
// every other array shared, which is the property §4.5 asks for by name: a
// change to positions must not invalidate the triangulation, the edge list or
// the index buffers.
//
// It also fixes the undo record. The `BMesh` path took the coarse snapshot, so
// moving four vertices of a million recorded a million; the declared write set
// records the four.

/// How influence falls off with distance, for proportional editing.
///
/// Smooth by default. A modeller reaches for proportional editing to shape a
/// surface, and a linear falloff leaves a visible cone with a crease at its rim
/// — the very artefact the tool exists to avoid.
fn falloff(distance: f64, radius: f64, curve: u32) -> f32 {
    if radius <= 0.0 {
        return 0.0;
    }
    let t = (distance / radius).clamp(0.0, 1.0);
    // 1 at the centre, 0 at the rim.
    let s = 1.0 - t;
    let w = match curve {
        // Sphere: a dome, round at the top and steep at the rim.
        1 => (s * (2.0 - s)).sqrt(),
        // Root: a quick rise near the rim, a broad top.
        2 => s.sqrt(),
        // Sharp: a point at the centre.
        3 => s * s,
        // Linear: a cone.
        4 => s,
        // Constant: everything in reach moves fully.
        5 if t < 1.0 => 1.0,
        5 => 0.0,
        // Smooth, the default — smoothstep, flat at both ends, so the moved
        // region joins the rest of the mesh without a crease.
        _ => s * s * (3.0 - 2.0 * s),
    };
    w as f32
}

/// Blender's proportional falloff curves, in its order; the index is what
/// `proportional_falloff` carries.
pub const FALLOFF_CURVES: [&str; 6] = ["smooth", "sphere", "root", "sharp", "linear", "constant"];

/// The one place a set of world positions occupies, or `None` for more.
///
/// Rotate and scale refuse a selection that occupies a single place: every
/// affected vertex is then the pivot, so the operation is exactly the identity,
/// and reporting success for a no-op puts an entry in the history that undoes
/// to the same mesh.
///
/// Returns the place rather than a count, because the answer has to compose
/// across the meshes in edit mode: one vertex selected in each of two objects
/// occupies two places, and no per-mesh count can see that. Comparing the two
/// meshes' places can.
///
/// Stops at the first disagreement, so a large selection costs two positions
/// rather than a sort.
fn one_place(world: impl Iterator<Item = glam::DVec3>) -> Option<[i64; 3]> {
    // Quantised at a micrometre, so two vertices a float's breadth apart count
    // as one place. Without this a "single vertex" that happens to be two
    // welded ones would pass for something scalable.
    let key = |p: glam::DVec3| {
        let q = |x: f64| (x * 1e6).round() as i64;
        [q(p.x), q(p.y), q(p.z)]
    };
    let mut first: Option<[i64; 3]> = None;
    for p in world {
        let k = key(p);
        match first {
            None => first = Some(k),
            Some(seen) if seen == k => {}
            Some(_) => return None,
        }
    }
    first
}

/// A weight for every vertex the transform reaches: 1 for the selection,
/// falling off for the rest.
///
/// The distance is to the *nearest selected vertex*, not to the selection’s
/// median. That matters as soon as the selection is more than a point: dragging
/// a whole edge loop should pull the mesh evenly along its length, and a
/// distance to the median would pull hardest in the middle and leave the ends
/// behind.
///
/// # Getting here took three structures, and the two that failed are the
/// interesting part
///
/// It began O(V x S): every vertex measured against every selected one. Two
/// hundred thousand vertices with two thousand selected is four hundred million
/// distances — **450 ms a frame**, against a budget of one.
///
/// A **sorted cell table**, copying `weld_by_distance`, made it 187 ms and for
/// a *small* radius made it 118 — worse than it had any business being, because
/// twenty-seven `partition_point` calls a vertex is three hundred comparisons
/// spent before an anchor is touched at all. The lookup was the cost.
///
/// A **hashed cell grid**, O(1) per lookup with the anchors’ bounding box
/// rejecting the rest of the mesh first, made the small radius 3.5 ms and left
/// the large one at 43. That is the shape of the real problem: a modeller’s
/// selection is a compact patch, so every anchor lands in the same handful of
/// cells, and a cell lookup narrows two thousand candidates to two thousand
/// candidates. A grid answers "anything within `r`" well and "which is nearest"
/// not at all.
///
/// A [`PointTree`] answers the question that was actually being asked. **1.0 to
/// 4.3 ms across every radius**, and the cost now scales with the logarithm of
/// the selection rather than with its size.
///
/// The tree is still rebuilt every frame, which §4.5 says it should not be: the
/// weights depend only on the rest positions, the selection and the radius, and
/// none of those change during a drag. Computing them once when the gesture
/// starts is the GPU preview path, and it is where the rest of this goes.
///
/// # Measured in world space, over the whole selection
///
/// `positions` arrives already in world coordinates, because the radius is a
/// world length: it is the circle the viewport draws and the number the wheel
/// adjusts. On an object scaled by two, a falloff measured in local space would
/// reach twice as far on screen as the circle says — and differently in each
/// axis under a non-uniform scale.
///
/// And `tree` is built over *every* selected vertex in edit mode, not this
/// mesh's own. A tree per mesh cannot pull on a neighbour, so dragging a seam
/// between two touching objects would leave the far side of it behind — which
/// is the whole reason anyone puts two objects in edit mode together.
fn weigh_neighbourhood(
    tree: &kreate_math::PointTree,
    positions: &[glam::Vec3],
    selected: &[u32],
    radius: f64,
    curve: u32,
) -> (Vec<u32>, Vec<f32>) {
    let limit = radius as f32;

    let mut indices: Vec<u32> = Vec::with_capacity(selected.len());
    let mut weights: Vec<f32> = Vec::with_capacity(selected.len());
    let mut next_selected = 0usize;

    for (v, p) in positions.iter().enumerate() {
        let v = v as u32;
        if selected.get(next_selected) == Some(&v) {
            // Full weight, and not measured: a selected vertex is at distance
            // zero from itself, but saying so explicitly means a rounding error
            // can never give the selection less than 1. `selected` is
            // ascending, so a cursor finds them without a search.
            next_selected += 1;
            indices.push(v);
            weights.push(1.0);
            continue;
        }
        let Some(nearest) = tree.nearest(*p, limit) else {
            continue;
        };
        let weight = falloff(nearest.distance as f64, radius, curve);
        if weight > 0.0 {
            indices.push(v);
            weights.push(weight);
        }
    }
    (indices, weights)
}

/// What a transform is about to move, read off the document mesh.
///
/// Public because the GPU transform preview needs *these* weights and not its
/// own. The preview and the operator have to agree exactly — the preview is
/// what the user shapes the mesh against, and the operator is what runs when
/// the pointer comes up — so a second implementation of the falloff would
/// diverge, and the divergence would show as the mesh moving at the end of
/// every drag. Same argument as §2.3's one-graph-two-consumers rule, applied to
/// arithmetic rather than to shading.
pub struct Weighting {
    /// Which vertices move, ascending. The order the weights are in.
    pub indices: Vec<u32>,
    /// Each one’s share of the transform. One for the selection itself.
    pub weights: Vec<f32>,
    /// Which mesh, and the object whose space it is in.
    pub target: EditTarget,
}

/// Everything a transform across edit mode is about to move.
///
/// One of these rather than a `Weighting` per mesh, because the two things a
/// transform needs besides the weights — the pivot and the falloff — are
/// properties of the *whole* selection and cannot be worked out one mesh at a
/// time. The pivot obviously so. The falloff less obviously: it measures the
/// distance to the nearest selected vertex, and the nearest one may be in
/// another object.
pub struct EditWeighting {
    /// One per mesh with something to move, in [`edit_targets`] order.
    ///
    /// A mesh with nothing selected and nothing within the radius is absent
    /// rather than present and empty.
    pub parts: Vec<Weighting>,
    /// What a rotation or a scale happens about: the median of every selected
    /// vertex in every mesh.
    ///
    /// The mean rather than the bounding-box centre. For a symmetric selection
    /// they agree; for a lopsided one the mean follows where the geometry
    /// actually is, and the bounding-box centre can sit in empty space away
    /// from every vertex — which makes a rotation swing rather than turn.
    ///
    /// **In world space**, like everything a transform operator is told. The
    /// pointer speaks world coordinates and so does a typed number, so the
    /// pivot has to as well or a rotation on a moved object turns about the
    /// wrong point. And `f64`, like the rest of the world: an object ten units
    /// out already costs a rotation about this point 3e-7 of accuracy in `f32`.
    pub pivot: glam::DVec3,
    /// Where the selection sits, if it sits in one place.
    ///
    /// A world position quantised at a micrometre, or `None` when the selection
    /// spans more than one. Rotate and scale refuse a selection occupying a
    /// single place — every affected vertex is then the pivot, so the operation
    /// is exactly the identity.
    ///
    /// Asked of the whole selection, which is the only way it composes: one
    /// vertex selected in each of two objects is two places with a point
    /// between them, and turning them about it is a real operation.
    pub one_place: Option<[i64; 3]>,
}

/// Weighs the selection across every mesh in edit mode.
///
/// `None` when nothing at all is selected anywhere. Everything it returns is
/// measured in world space — see [`EditWeighting::pivot`] and
/// [`weigh_neighbourhood`].
///
/// Public because the GPU transform preview needs *these* weights and not its
/// own. The preview and the operator have to agree exactly — the preview is
/// what the user shapes the mesh against, and the operator is what runs when
/// the pointer comes up — so a second implementation of the falloff would
/// diverge, and the divergence would show as the mesh moving at the end of
/// every drag. Same argument as §2.3's one-graph-two-consumers rule, applied to
/// arithmetic rather than to shading.
pub fn weigh_edit_selection(doc: &Document, radius: f64) -> Option<EditWeighting> {
    weigh_edit_selection_with(doc, radius, 0)
}

/// [`weigh_edit_selection`], with one of [`FALLOFF_CURVES`].
pub fn weigh_edit_selection_with(doc: &Document, radius: f64, curve: u32) -> Option<EditWeighting> {
    let targets = edit_targets(doc);
    if targets.is_empty() {
        return None;
    }

    // Every mesh's own selection, and — when there is a falloff to measure —
    // its vertices in world coordinates. Gathered first, because the anchors
    // the falloff measures against are the union of all of them.
    struct Gathered {
        target: EditTarget,
        selected: Vec<u32>,
        /// This mesh's vertices in world space. Empty when the radius is zero,
        /// where nothing outside the selection moves and transforming the whole
        /// mesh would be a pass over it to conclude exactly that.
        world: Vec<glam::Vec3>,
    }

    let mut gathered: Vec<Gathered> = Vec::with_capacity(targets.len());
    let mut anchors: Vec<glam::Vec3> = Vec::new();
    let mut sum = glam::DVec3::ZERO;
    let mut count = 0usize;
    for target in targets {
        let Some(mesh) = doc.mesh(target.mesh) else {
            continue;
        };
        let selected = selection_vertices(mesh);
        // A mesh with nothing selected is still gathered when there is a
        // falloff, because the falloff may reach *into* it from a neighbour:
        // that is what dragging a seam between two touching objects means, and
        // skipping it here is the bug
        // `the_proportional_falloff_does_not_reach_across_objects` was written
        // to catch. With no falloff there is nothing to reach with, and it is
        // dropped.
        if selected.is_empty() && radius <= 0.0 {
            continue;
        }
        let local = mesh.positions();
        let into_world = |p: glam::Vec3| target.to_world.transform_point3(p.as_dvec3());
        for v in &selected {
            let p = into_world(local[*v as usize]);
            sum += p;
            anchors.push(p.as_vec3());
        }
        count += selected.len();
        let world: Vec<glam::Vec3> = if radius > 0.0 {
            local
                .iter()
                .copied()
                .map(|p| into_world(p).as_vec3())
                .collect()
        } else {
            Vec::new()
        };
        gathered.push(Gathered {
            target,
            selected,
            world,
        });
    }
    if count == 0 {
        return None;
    }

    let pivot = sum / count as f64;
    // Over the anchors, which are already in world space and already the whole
    // selection. The `f32` they were narrowed to for the tree is fine here: the
    // quantisation is a micrometre and the question is only whether two
    // vertices are in the same place.
    let one = one_place(anchors.iter().map(|p| p.as_dvec3()));

    // One tree for the whole selection. Building one per mesh is what let the
    // falloff stop at an object boundary, and it is also more work: the tree's
    // cost is logarithmic in the anchors, so N trees over S/N anchors each is
    // slower than one over S as well as wrong.
    let tree = (radius > 0.0).then(|| kreate_math::PointTree::build(&anchors));

    let mut parts = Vec::with_capacity(gathered.len());
    for g in gathered {
        let (indices, weights) = match &tree {
            Some(tree) => weigh_neighbourhood(tree, &g.world, &g.selected, radius, curve),
            // Proportional editing off, which is the default and the common
            // case: nothing outside the selection moves.
            None => {
                let weights = vec![1.0; g.selected.len()];
                (g.selected, weights)
            }
        };
        if indices.is_empty() {
            continue;
        }
        parts.push(Weighting {
            indices,
            weights,
            target: g.target,
        });
    }
    if parts.is_empty() {
        return None;
    }
    Some(EditWeighting {
        parts,
        pivot,
        one_place: one,
    })
}

/// Applies a positional transform to whatever the selection reaches.
///
/// `place` is given a vertex’s position, its weight and the selection’s
/// median, and returns where it goes. Whether a partial weight blends the
/// *transform* or the *result* is the caller’s business, because the two
/// differ for a rotation — see [`RotateSelection`].
///
/// `needs_extent` refuses a selection occupying a single place, which rotate
/// and scale must and translate must not.
///
/// # World in, world out
///
/// `place` sees world coordinates, and its answer is brought back into the
/// mesh's own space before it is written. Which is the only arrangement that
/// works: the parameters come from a pointer travelling across the view, or
/// from a typed number, and both of those mean world units. Applying a world
/// vector to local positions — which is what this did — moves a quarter-turned
/// object's vertices at a right angle to the drag.
///
/// Done per vertex as `world⁻¹(op(world(p)))` rather than by folding the
/// object's transform into the operation, and deliberately: conjugating a world
/// rotation into the space of a non-uniformly scaled object is a shear, which
/// no axis-and-angle can express. Two matrix-vector products a vertex costs
/// nothing here — the operator runs once per drag — and it is exact for any
/// object transform, including a sheared one.
///
/// # Every mesh in edit mode, about one pivot
///
/// The pivot is the median of every selected vertex in *all* of them, and that
/// is what makes this one transform rather than several. Rotating a body and
/// its two arms about three separate medians turns each about its own centre
/// and pulls the model apart at the seams — which is not a subtler version of
/// what the user asked for, it is a different operation.
///
/// A mesh with nothing selected contributes nothing and is skipped, but does
/// not cancel: reaching for a tool with three objects in edit mode and elements
/// selected in one of them is an ordinary thing to do.
fn run_transform(
    doc: &mut DocMut<'_>,
    radius: f64,
    curve: u32,
    needs_extent: bool,
    pivot: Option<glam::DVec3>,
    place: impl Fn(glam::DVec3, f32, glam::DVec3) -> glam::DVec3,
) -> Result<OpResult, OpError> {
    if edit_targets(doc.document()).is_empty() {
        return Err(OpError::Failed("no mesh to edit".into()));
    }
    // The whole selection at once, because the pivot and the falloff are both
    // properties of all of it and nothing can be written until they are known.
    let Some(weighing) = weigh_edit_selection_with(doc.document(), radius, curve) else {
        return Ok(OpResult::Cancelled);
    };
    // Rotate and scale refuse a selection occupying a single place: every
    // affected vertex is then the pivot, so the operation is the identity —
    // unless it turns about somewhere else, the 3D cursor.
    if needs_extent && pivot.is_none() && weighing.one_place.is_some() {
        return Ok(OpResult::Cancelled);
    }

    let mut moved_any = false;
    for part in &weighing.parts {
        // A degenerate object transform — a zero scale in some axis — has no
        // inverse, and every vertex of the mesh is already at the same world
        // point. Skipped rather than written as NaN, which would be a mesh
        // nobody can recover; the others still move.
        let to_world = part.target.to_world;
        let to_local = to_world.inverse();
        if !to_local.is_finite() {
            continue;
        }

        // `edit_positions` visits the indices in the order given, which is the
        // order the weights are in. Asserted rather than assumed: a change of
        // order there would pair each vertex with someone else’s weight, and a
        // mesh transformed by scrambled weights still looks like a mesh.
        let mut next = 0usize;
        doc.edit_positions(part.target.mesh, &part.indices, |index, p| {
            debug_assert_eq!(
                index, part.indices[next],
                "a weight was paired with the wrong vertex"
            );
            let world = to_world.transform_point3(p.as_dvec3());
            let placed = place(world, part.weights[next], pivot.unwrap_or(weighing.pivot));
            *p = to_local.transform_point3(placed).as_vec3();
            next += 1;
        })
        .map_err(|e| OpError::Failed(e.to_string()))?;
        moved_any = true;
    }
    if !moved_any {
        return Ok(OpResult::Cancelled);
    }
    Ok(OpResult::Finished)
}

/// How close, in a mesh's own space, two vertices must be for Auto Merge to
/// take them as one place: a hundredth of a millimetre, far under anything
/// drawn on purpose and far over a snap's rounding.
pub const AUTO_MERGE_DISTANCE: f32 = 1e-5;

/// After a transform: with Auto Merge on, every selected vertex that landed
/// on another is merged with it, in the same step.
fn merge_after(
    doc: &mut DocMut<'_>,
    done: OpResult,
    auto_merge: bool,
) -> Result<OpResult, OpError> {
    if !auto_merge || !matches!(done, OpResult::Finished) {
        return Ok(done);
    }
    each_bmesh(doc, |bm| {
        let moved = bm.selected_verts();
        edit::weld_onto(bm, &moved, AUTO_MERGE_DISTANCE).map_err(|e| OpError::Failed(e.to_string()))
    })?;
    Ok(done)
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MeshTranslateParams {
    /// How far to move the selection, in object space.
    #[param(unit = "length")]
    pub translation: [f64; 3],
    /// Proportional editing radius. Zero is off.
    #[param(soft_min = 0.0, soft_max = 10.0, unit = "length")]
    pub proportional_size: f64,
    /// The shape the influence falls off in. See [`FALLOFF_CURVES`].
    #[param(variants("smooth", "sphere", "root", "sharp", "linear", "constant"))]
    pub proportional_falloff: u32,
    /// Merge each moved vertex with any other now in its place: Auto
    /// Merge, filled in by the application from its header toggle.
    #[param(hidden)]
    pub auto_merge: bool,
}

/// Moves the selected vertices, edges or faces.
pub struct TranslateSelection;

impl Operator for TranslateSelection {
    type Params = MeshTranslateParams;

    fn id() -> &'static str {
        "mesh.translate"
    }
    fn label() -> &'static str {
        "Move Selection"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let offset = glam::DVec3::from_array(params.translation);
        let done = run_transform(
            doc,
            params.proportional_size,
            params.proportional_falloff,
            false,
            None,
            |p, weight, _| p + offset * weight as f64,
        )?;
        merge_after(doc, done, params.auto_merge)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DissolveLimitedParams {
    /// How far apart two faces' directions may be and still merge.
    #[param(unit = "angle", min = 0.0, max = 3.0)]
    pub angle: f64,
}

impl Default for DissolveLimitedParams {
    fn default() -> Self {
        Self {
            angle: 5f64.to_radians(),
        }
    }
}

/// Merges faces that lie in one plane, across every mesh being edited:
/// Blender's Limited Dissolve, and what tidies a boolean's or an import's
/// triangles back into the faces they were.
pub struct DissolveLimited;

impl Operator for DissolveLimited {
    type Params = DissolveLimitedParams;

    fn id() -> &'static str {
        "mesh.dissolve_limited"
    }
    fn label() -> &'static str {
        "Limited Dissolve"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let angle = params.angle.clamp(0.0, std::f64::consts::PI) as f32;
        let gone: usize = each_bmesh(doc, |bm| {
            edit::limited_dissolve(bm, angle).map_err(|e| OpError::Failed(e.to_string()))
        })?
        .into_iter()
        .sum();
        Ok(if gone > 0 {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

/// What `mesh.check_select` can find, in its numbering.
pub const CHECK_ISSUES: [&str; 7] = [
    "open",
    "non_manifold",
    "flipped",
    "degenerate",
    "loose",
    "intersecting",
    "thin",
];

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct CheckSelectParams {
    /// Which fault to select. See [`CHECK_ISSUES`].
    #[param(variants(
        "open",
        "non_manifold",
        "flipped",
        "degenerate",
        "loose",
        "intersecting",
        "thin"
    ))]
    pub issue: u32,
    /// The thinnest wall allowed, in the world, for `thin`.
    #[param(min = 0.0, unit = "length")]
    pub thickness: f64,
}

impl Default for CheckSelectParams {
    fn default() -> Self {
        Self {
            issue: 0,
            thickness: 0.001,
        }
    }
}

/// Selects what a mesh check found, in every mesh being edited: the rims of
/// holes, faces passing through each other, walls too thin to print. The
/// rest is deselected, so what is left selected is exactly the fault.
pub struct CheckSelect;

impl Operator for CheckSelect {
    type Params = CheckSelectParams;

    fn id() -> &'static str {
        "mesh.check_select"
    }
    fn label() -> &'static str {
        "Select Mesh Problems"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::attribute::Domain;
        let issue = params.issue.min(CHECK_ISSUES.len() as u32 - 1);
        let thickness = if issue == 6 {
            params.thickness.max(0.0)
        } else {
            0.0
        };
        let mut count = 0usize;
        let found = for_each_edit_mesh(doc, |target, mesh| {
            let report = kreate_mesh::check::check(
                mesh,
                |p| target.to_world.transform_point3(p),
                thickness,
                issue == 5,
            );
            let (domain, mode, list) = match issue {
                0 => (Domain::Edge, SelectMode::Edge, report.open_edges),
                1 => (Domain::Edge, SelectMode::Edge, report.non_manifold_edges),
                2 => (Domain::Edge, SelectMode::Edge, report.flipped_edges),
                3 => (Domain::Face, SelectMode::Face, report.degenerate_faces),
                4 => (Domain::Point, SelectMode::Vert, report.loose_verts),
                5 => (Domain::Face, SelectMode::Face, report.intersecting_faces),
                _ => (Domain::Face, SelectMode::Face, report.thin_faces),
            };
            selection::select_none(mesh);
            for i in &list {
                selection::set_selected(mesh, domain, *i, true);
            }
            selection::flush(mesh, mode);
            count += list.len();
            !list.is_empty()
        })?;
        // How many, said: "Cancelled" for none and "finished" for some read
        // as how the operator went, not as what the check found.
        let what = [
            "open edges",
            "non-manifold edges",
            "flipped edges",
            "degenerate faces",
            "loose vertices",
            "intersecting faces",
            "thin faces",
        ][issue as usize];
        crate::report::say(if count == 0 {
            format!("found 0 {what}: none")
        } else {
            format!("found {count} {what}, selected")
        });
        Ok(if found {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MedianSetParams {
    /// Where the selection's middle goes, in the world.
    #[param(unit = "length")]
    pub location: [f64; 3],
}

/// Puts the selection's middle at a typed place: Blender's Median field in
/// the sidebar, or Vertex for one. The selection moves as one, by the
/// difference — what a coordinate on a drawing means.
pub struct MedianSet;

impl Operator for MedianSet {
    type Params = MedianSetParams;

    fn id() -> &'static str {
        "mesh.median_set"
    }
    fn label() -> &'static str {
        "Set Selection Position"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let to = glam::DVec3::from_array(params.location);
        let Some(weighing) = weigh_edit_selection_with(doc.document(), 0.0, 0) else {
            return Ok(OpResult::Cancelled);
        };
        let offset = to - weighing.pivot;
        if !offset.is_finite() || offset.length() < 1e-12 {
            return Ok(OpResult::Cancelled);
        }
        run_transform(doc, 0.0, 0, false, None, |p, weight, _| {
            p + offset * weight as f64
        })
    }
}

/// The selection's middle in the world, and how many vertices it is — what
/// the sidebar shows as Median, or Vertex for one.
pub fn selection_median(doc: &Document) -> Option<(glam::DVec3, usize)> {
    let count: usize = edit_targets(doc)
        .iter()
        .filter_map(|t| doc.mesh(t.mesh))
        .map(|m| selection_vertices(m).len())
        .sum();
    if count == 0 {
        return None;
    }
    weigh_edit_selection_with(doc, 0.0, 0).map(|w| (w.pivot, count))
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MeshRotateParams {
    #[param(unit = "angle")]
    pub angle: f64,
    /// Which axis to turn about, through the selection's median.
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
    /// Proportional editing radius. Zero is off.
    #[param(soft_min = 0.0, soft_max = 10.0, unit = "length")]
    pub proportional_size: f64,
    /// The shape the influence falls off in. See [`FALLOFF_CURVES`].
    #[param(variants("smooth", "sphere", "root", "sharp", "linear", "constant"))]
    pub proportional_falloff: u32,
    /// What it turns about: the selection's median, or the 3D cursor.
    #[param(variants("median", "cursor"))]
    pub pivot: u32,
    /// Merge each moved vertex with any other now in its place: Auto
    /// Merge, filled in by the application from its header toggle.
    #[param(hidden)]
    pub auto_merge: bool,
}

/// Rotates the selected vertices, edges or faces about their median.
pub struct RotateSelection;

impl Operator for RotateSelection {
    type Params = MeshRotateParams;

    fn id() -> &'static str {
        "mesh.rotate"
    }
    fn label() -> &'static str {
        "Rotate Selection"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let axis = match params.axis {
            0 => glam::DVec3::X,
            1 => glam::DVec3::Y,
            _ => glam::DVec3::Z,
        };
        let rotation = glam::DQuat::from_axis_angle(axis, params.angle);
        let pivot = (params.pivot == 1).then_some(doc.document().cursor);
        let done = run_transform(
            doc,
            params.proportional_size,
            params.proportional_falloff,
            true,
            pivot,
            |p, weight, about| {
                // The rotation is interpolated towards the identity rather than the
                // *result* being interpolated towards the original. The two differ:
                // blending positions drags a partly-weighted vertex along the chord
                // and shrinks the mesh towards the pivot, while blending the
                // rotation turns it through a smaller angle, which is what the tool
                // means.
                let partial = glam::DQuat::IDENTITY.slerp(rotation, weight as f64);
                about + partial * (p - about)
            },
        )?;
        merge_after(doc, done, params.auto_merge)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct MeshScaleParams {
    /// Scale factor per axis, about the selection's median.
    pub scale: [f64; 3],
    /// Proportional editing radius. Zero is off.
    #[param(soft_min = 0.0, soft_max = 10.0, unit = "length")]
    pub proportional_size: f64,
    /// The shape the influence falls off in. See [`FALLOFF_CURVES`].
    #[param(variants("smooth", "sphere", "root", "sharp", "linear", "constant"))]
    pub proportional_falloff: u32,
    /// What it scales about: the selection's median, or the 3D cursor.
    #[param(variants("median", "cursor"))]
    pub pivot: u32,
    /// Merge each moved vertex with any other now in its place: Auto
    /// Merge, filled in by the application from its header toggle.
    #[param(hidden)]
    pub auto_merge: bool,
}

impl Default for MeshScaleParams {
    fn default() -> Self {
        Self {
            scale: [1.0; 3],
            proportional_size: 0.0,
            proportional_falloff: 0,
            pivot: 0,
            auto_merge: false,
        }
    }
}

/// Scales the selected edges or faces about their median.
///
/// # Why a single vertex is refused
///
/// Scaling happens about the selection's median, and a lone vertex *is* its own
/// median — so the operation is exactly the identity. Refusing it is not a
/// restriction on vertex mode: a selected loop or patch of vertices scales
/// normally, and tapering a ring of vertices that way is a core modelling move.
/// What is refused is the case that could not do anything, which otherwise puts
/// an entry in the history that undoes to the same mesh.
pub struct ScaleSelection;

impl Operator for ScaleSelection {
    type Params = MeshScaleParams;

    fn id() -> &'static str {
        "mesh.scale"
    }
    fn label() -> &'static str {
        "Scale Selection"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let factor = glam::DVec3::from_array(params.scale);
        let pivot = (params.pivot == 1).then_some(doc.document().cursor);
        let done = run_transform(
            doc,
            params.proportional_size,
            params.proportional_falloff,
            true,
            pivot,
            |p, weight, about| {
                // The factor is interpolated towards one.
                //
                // Unlike the rotation, this is *not* a correction: scaling is
                // linear in the position, so blending the factor and blending the
                // result are algebraically the same thing, and a test cannot tell
                // them apart. Written this way for symmetry with the rotation above
                // and because it says what is meant — a vertex at half weight is
                // scaled half as much, rather than dragged half way to where a full
                // scale would put it.
                let partial = glam::DVec3::ONE.lerp(factor, weight as f64);
                about + (p - about) * partial
            },
        )?;
        merge_after(doc, done, params.auto_merge)
    }
}

// --- extrude ----------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ExtrudeParams {
    /// How far to move the new faces.
    #[param(soft_min = -10.0, soft_max = 10.0, unit = "length")]
    pub offset: f64,
    /// Which way, in the world; zero for along the region's normal. What an
    /// axis key pressed during the extrude picks — the world's or the
    /// object's own.
    pub direction: [f64; 3],
    /// Each face out along its own normal, the region holding together —
    /// Blender's Extrude Faces Along Normals. `direction` is then unused.
    pub along_normals: bool,
}

impl Default for ExtrudeParams {
    fn default() -> Self {
        Self {
            offset: 1.0,
            direction: [0.0; 3],
            along_normals: false,
        }
    }
}

/// Extrudes the selected faces along their shared normal.
pub struct ExtrudeRegion;

impl Operator for ExtrudeRegion {
    type Params = ExtrudeParams;

    fn id() -> &'static str {
        "mesh.extrude_region"
    }
    fn label() -> &'static str {
        "Extrude Region"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        // Every mesh in edit mode. A mesh with no faces selected returns
        // early and is left alone, which is what makes this safe to run over
        // the whole set.
        let direction = glam::DVec3::from_array(params.direction);
        let along_axis = !params.along_normals && direction.length_squared() > 1e-12;
        let grew = each_bmesh(doc, |bm| {
            let faces = bm.selected_faces();
            if faces.is_empty() {
                return Ok(false);
            }
            // Along an axis: made where they are, then moved in the world,
            // where the axis is.
            let distance = if along_axis {
                0.0
            } else {
                params.offset as f32
            };
            if params.along_normals {
                edit::extrude_selected_faces_along_normals(bm, distance)
            } else {
                edit::extrude_selected_faces(bm, distance)
            }
            .map_err(|e| OpError::Failed(e.to_string()))?;
            Ok(true)
        })?
        .into_iter()
        .any(|g| g);
        if grew && along_axis {
            let offset = direction.normalize() * params.offset;
            run_transform(doc, 0.0, 0, false, None, |p, weight, _| {
                p + offset * weight as f64
            })?;
        }

        // Cancelled rather than Finished, so nothing lands in the history and
        // the user's next undo still reaches the edit they actually made.
        Ok(if grew {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

// --- delete and dissolve ----------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DeleteParams {
    /// What the delete takes with it.
    #[param(variants("verts", "edges", "faces", "only_faces"))]
    pub mode: u32,
}

/// Removes the selected geometry, leaving a hole.
pub struct DeleteElements;

impl Operator for DeleteElements {
    type Params = DeleteParams;

    fn id() -> &'static str {
        "mesh.delete"
    }
    fn label() -> &'static str {
        "Delete"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mode = match params.mode {
            0 => edit::DeleteMode::Verts,
            1 => edit::DeleteMode::Edges,
            3 => edit::DeleteMode::FacesOnly,
            _ => edit::DeleteMode::Faces,
        };
        let removed = each_bmesh(doc, |bm| Ok(edit::delete(bm, mode)))?
            .into_iter()
            .fold(
                (0, 0, 0),
                |a: (usize, usize, usize), b: (usize, usize, usize)| {
                    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
                },
            );
        Ok(if removed == (0, 0, 0) {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DissolveParams {
    /// Which element type to dissolve.
    #[param(variants("verts", "edges", "faces"))]
    pub mode: u32,
}

/// Removes the selected geometry and heals what is left into one face.
pub struct DissolveElements;

impl Operator for DissolveElements {
    type Params = DissolveParams;

    fn id() -> &'static str {
        "mesh.dissolve"
    }
    fn label() -> &'static str {
        "Dissolve"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let healed = each_bmesh(doc, |bm| {
            let result = match params.mode {
                0 => edit::dissolve_verts(bm, &bm.selected_verts()),
                2 => edit::dissolve_faces(bm, &bm.selected_faces(), &[]),
                _ => edit::dissolve_edges(bm, &bm.selected_edges()),
            };
            result.map_err(|e| OpError::Failed(e.to_string()))
        })?
        .into_iter()
        .any(|out| !out.faces.is_empty());

        // A dissolve that healed nothing but refused some groups is not a
        // failure the user should have to undo — it is a report. Telling them
        // apart matters: `Cancelled` leaves the history clean.
        Ok(if !healed {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- subdivide --------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SubdivideParams {
    /// How many times to halve every edge. Each pass turns one face into one
    /// quad per corner.
    #[param(min = 1, max = 6, unit = "count")]
    pub iterations: u32,
}

impl Default for SubdivideParams {
    fn default() -> Self {
        Self { iterations: 1 }
    }
}

/// Splits every edge of the selected faces at its midpoint.
pub struct Subdivide;

impl Operator for Subdivide {
    type Params = SubdivideParams;

    fn id() -> &'static str {
        "mesh.subdivide"
    }
    fn label() -> &'static str {
        "Subdivide"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let grew = each_bmesh(doc, |bm| {
            let mut any = false;
            // Clamped rather than trusted: the parameter reaches here from a
            // script or a saved keymap as well as from the spinner, and six
            // iterations already multiplies a face count by four thousand.
            for _ in 0..params.iterations.clamp(1, 6) {
                let out =
                    edit::subdivide_selected(bm).map_err(|e| OpError::Failed(e.to_string()))?;
                any |= !out.faces.is_empty();
            }
            Ok(any)
        })?
        .into_iter()
        .any(|g| g);
        Ok(if grew {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

// --- bevel ------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct BevelParams {
    /// How wide the chamfer is, measured along the neighbouring edges.
    #[param(min = 0.0, soft_max = 1.0, unit = "length")]
    pub width: f64,
    /// Strips across each edge: one is a flat chamfer, more round it off.
    #[param(min = 1, max = 64, unit = "count")]
    pub segments: u32,
}

impl Default for BevelParams {
    fn default() -> Self {
        Self {
            width: 0.1,
            segments: 1,
        }
    }
}

/// Chamfers the selected edges.
pub struct BevelEdges;

impl Operator for BevelEdges {
    type Params = BevelParams;

    fn id() -> &'static str {
        "mesh.bevel"
    }
    fn label() -> &'static str {
        "Bevel"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let built: usize = each_bmesh(doc, |bm| {
            edit::bevel_selected_segments(bm, params.width as f32, params.segments)
                .map(|out| out.faces.len())
                .map_err(|e| OpError::Failed(e.to_string()))
        })?
        .into_iter()
        .sum();
        // Every selected edge was a boundary or a non-manifold fan, so there was
        // nowhere to put a chamfer. A no-op rather than a failure: the selection
        // was legitimate, the operation just does not apply to it.
        Ok(if built == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct FilletParams {
    /// The round's radius: where it leaves each face, for an edge at right
    /// angles; or how far a chamfer cuts back along each face.
    #[param(min = 0.0, soft_max = 0.05, unit = "length")]
    pub radius: f64,
    /// A flat cut instead of a round.
    pub chamfer: bool,
}

impl Default for FilletParams {
    fn default() -> Self {
        Self {
            radius: 0.001,
            chamfer: false,
        }
    }
}

/// The selected edges rounded to a radius — a fillet — or cut back a
/// distance — a chamfer: a bevel in a machinist's terms.
pub struct Fillet;

impl Operator for Fillet {
    type Params = FilletParams;
    fn id() -> &'static str {
        "mesh.fillet"
    }
    fn label() -> &'static str {
        "Fillet or Chamfer"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let segments = if params.chamfer {
            1
        } else {
            kreate_mesh::modifiers::bevel::FILLET_SEGMENTS
        };
        let built: usize = each_bmesh(doc, |bm| {
            edit::bevel_selected_segments(bm, params.radius as f32, segments)
                .map(|out| out.faces.len())
                .map_err(|e| OpError::Failed(e.to_string()))
        })?
        .into_iter()
        .sum();
        Ok(if built == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- split and rip ----------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SplitParams {
    /// Detach the selected faces, or open a seam along the selected edges.
    #[param(variants("faces", "edges"))]
    pub mode: u32,
}

/// Takes geometry apart: detaches faces, or opens a seam along edges.
pub struct SplitGeometry;

impl Operator for SplitGeometry {
    type Params = SplitParams;

    fn id() -> &'static str {
        "mesh.split"
    }
    fn label() -> &'static str {
        "Split"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let opened: usize = each_bmesh(doc, |bm| {
            let result = if params.mode == 1 {
                edit::rip_selected_edges(bm)
            } else {
                edit::split_selected_faces(bm)
            };
            result
                .map(|out| out.seams)
                .map_err(|e| OpError::Failed(e.to_string()))
        })?
        .into_iter()
        .sum();
        // Nothing shared means nothing to detach, which is a no-op rather than a
        // failure — the geometry was already separate.
        Ok(if opened == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- fill -------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct FillParams {
    /// Build a quad patch instead of one n-gon.
    pub grid: bool,
    /// How many edges along one side of the patch. Zero picks the squarest
    /// division the border allows.
    #[param(min = 0, max = 256, unit = "count")]
    pub span: u32,
}

/// Closes an open border.
pub struct FillHole;

impl Operator for FillHole {
    type Params = FillParams;

    fn id() -> &'static str {
        "mesh.fill"
    }
    fn label() -> &'static str {
        "Fill"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let built: usize = each_bmesh(doc, |bm| {
            let span = (params.span > 0).then_some(params.span as usize);
            match edit::fill_selected(bm, params.grid, span) {
                Ok(out) => Ok(out.faces.len() + out.edges.len()),
                // A refusal is information, not a crash: "two borders" and "an
                // odd border" both tell the user exactly what to change.
                Err(e) => Err(OpError::Failed(e.to_string())),
            }
        })?
        .into_iter()
        .sum();
        Ok(if built == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- spin -------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SpinParams {
    /// Total angle to sweep.
    #[param(soft_min = -6.29, soft_max = 6.29, unit = "angle")]
    pub angle: f64,
    /// How many bands to divide the sweep into.
    #[param(min = 1, max = 512, unit = "count")]
    pub steps: u32,
    /// Axis to revolve about.
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
    /// Point on the axis.
    #[param(unit = "length")]
    pub centre: [f64; 3],
    /// Weld the ends when a full revolution brings them together.
    pub close: bool,
}

impl Default for SpinParams {
    fn default() -> Self {
        Self {
            angle: std::f64::consts::TAU,
            steps: 12,
            axis: 2,
            centre: [0.0; 3],
            close: true,
        }
    }
}

/// Revolves the selected faces about an axis.
pub struct SpinFaces;

impl Operator for SpinFaces {
    type Params = SpinParams;

    fn id() -> &'static str {
        "mesh.spin"
    }
    fn label() -> &'static str {
        "Spin"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let axis = match params.axis {
            0 => kreate_math::Vec3::X,
            1 => kreate_math::Vec3::Y,
            _ => kreate_math::Vec3::Z,
        };
        let centre = kreate_math::Vec3::new(
            params.centre[0] as f32,
            params.centre[1] as f32,
            params.centre[2] as f32,
        );
        let grew: usize = each_bmesh(doc, |bm| {
            let out = edit::spin_selected(bm, centre, axis, params.angle as f32, params.steps)
                .map_err(|e| OpError::Failed(e.to_string()))?;
            if params.close {
                edit::close_spin(bm, &out).map_err(|e| OpError::Failed(e.to_string()))?;
            }
            Ok(out.verts.len())
        })?
        .into_iter()
        .sum();
        Ok(if grew == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- normals ----------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct NormalsParams {
    /// What to do to the normals.
    #[param(variants("recalculate_outside", "recalculate_inside", "flip"))]
    pub action: u32,
}

/// Fixes or reverses face winding.
pub struct RecalculateNormals;

impl Operator for RecalculateNormals {
    type Params = NormalsParams;

    fn id() -> &'static str {
        "mesh.normals"
    }
    fn label() -> &'static str {
        "Normals"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        // Asked of the whole of edit mode, not of each mesh: see
        // `nothing_selected_anywhere`.
        let empty = nothing_selected_anywhere(doc.document());
        let changed: usize = each_bmesh(doc, |bm| {
            // The whole mesh when nothing is selected anywhere: recalculating
            // normals is something a user does to a model they have just
            // imported, and making them select everything first is friction
            // with no purpose.
            let mut faces = bm.selected_faces();
            if faces.is_empty() && empty {
                faces = bm.face_ids().collect();
            }
            Ok(match params.action {
                2 => edit::flip(bm, &faces),
                inward => {
                    let out = edit::recalculate(bm, &faces, inward == 1);
                    // `ambiguous` is deliberately not reported here. On an open
                    // surface there is no outside, so the winding was made
                    // *consistent* and the direction is a guess — but the user
                    // asked for consistency too, and having flipped nothing is
                    // already the honest answer for "there was nothing to fix".
                    // Surfacing the distinction is the interface's job, and it
                    // needs a channel an operator does not have yet.
                    out.flipped
                }
            })
        })?
        .into_iter()
        .sum();
        Ok(if changed == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ShadingParams {
    /// Smooth, or flat.
    pub smooth: bool,
}

impl Default for ShadingParams {
    fn default() -> Self {
        Self { smooth: true }
    }
}

/// Sets whether the selected faces shade smooth or flat.
pub struct ShadeFaces;

impl Operator for ShadeFaces {
    type Params = ShadingParams;

    fn id() -> &'static str {
        "mesh.shade"
    }
    fn label() -> &'static str {
        "Shade Smooth"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let empty = nothing_selected_anywhere(doc.document());
        let changed: usize = each_bmesh(doc, |bm| {
            let mut faces = bm.selected_faces();
            if faces.is_empty() && empty {
                faces = bm.face_ids().collect();
            }
            Ok(edit::set_shade_smooth(bm, &faces, params.smooth))
        })?
        .into_iter()
        .sum();
        Ok(if changed == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SharpParams {
    /// Mark, or clear.
    pub sharp: bool,
    /// Mark every edge creased more than this instead of the selection. Zero
    /// uses the selection.
    #[param(min = 0.0, max = 3.15, unit = "angle")]
    pub angle: f64,
}

/// Marks edges sharp, either the selection or by crease angle.
pub struct MarkSharp;

impl Operator for MarkSharp {
    type Params = SharpParams;

    fn id() -> &'static str {
        "mesh.mark_sharp"
    }
    fn label() -> &'static str {
        "Mark Sharp"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let changed: usize = each_bmesh(doc, |bm| {
            Ok(if params.angle > 0.0 {
                edit::auto_smooth(bm, params.angle as f32)
            } else {
                edit::set_sharp_selected(bm, params.sharp)
            })
        })?
        .into_iter()
        .sum();
        Ok(if changed == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- loop cut ---------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct LoopCutParams {
    /// How many parallel loops to add.
    #[param(min = 1, max = 64, unit = "count")]
    pub cuts: u32,
    /// Where the group of cuts sits between the ring's edges, from 0 to 1.
    ///
    /// This is the slide: adjusting it after the cut re-runs the operator from
    /// the original mesh, so dragging it is exact rather than cumulative.
    #[param(min = 0.0, max = 1.0, step = 0.01)]
    pub position: f64,
    /// The edge of the active mesh the cut runs across, or −1 for the first
    /// selected one. What `Ctrl+R` names when the pointer chose the edge — so
    /// the cut replays without depending on a selection it never made.
    #[param(hidden)]
    pub edge: i64,
}

impl Default for LoopCutParams {
    fn default() -> Self {
        Self {
            cuts: 1,
            position: 0.5,
            edge: -1,
        }
    }
}

/// Runs new edge loops across the ring through the selected edge.
pub struct LoopCut;

impl Operator for LoopCut {
    type Params = LoopCutParams;

    fn id() -> &'static str {
        "mesh.loop_cut"
    }
    fn label() -> &'static str {
        "Loop Cut"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        // A named edge becomes the selection first, the way a click on it
        // would have made it: the cut then reads it as it reads any other.
        if params.edge >= 0 {
            let mesh_id = target_mesh(doc.document())
                .ok_or_else(|| OpError::Failed("no mesh to edit".into()))?;
            let named = with_named_mesh(doc, mesh_id, |mesh| {
                if params.edge as u64 >= mesh.num_edges() as u64 {
                    return false;
                }
                selection::select_none(mesh);
                selection::set_selected(mesh, kreate_mesh::Domain::Edge, params.edge as u32, true);
                selection::flush(mesh, SelectMode::Edge);
                true
            })?;
            if !named {
                return Ok(OpResult::Cancelled);
            }
        }
        let cut = with_bmesh(doc, |bm| {
            edit::loop_cut_selected(bm, params.cuts, params.position as f32)
                .map_err(|e| OpError::Failed(e.to_string()))
        })?;
        Ok(if cut.faces.is_empty() {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- extrude edges and vertices ---------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ExtrudeMoveParams {
    /// Where the new copies go, as a move in the world.
    #[param(unit = "length")]
    pub translation: [f64; 3],
}

/// Blender's `E` on edges and vertices: the selection pulled out into new
/// faces behind the edges and new edges behind lone vertices, and the copies
/// moved.
pub struct ExtrudeMove;

impl Operator for ExtrudeMove {
    type Params = ExtrudeMoveParams;
    fn id() -> &'static str {
        "mesh.extrude_move"
    }
    fn label() -> &'static str {
        "Extrude"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let made = with_bmesh(doc, |bm| {
            edit::extrude_edges_and_verts(bm).map_err(|e| OpError::Failed(e.to_string()))
        })?;
        if made.new_verts.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let offset = glam::DVec3::from_array(params.translation);
        if offset != glam::DVec3::ZERO {
            run_transform(doc, 0.0, 0, false, None, |p, weight, _| {
                p + offset * weight as f64
            })?;
        }
        Ok(OpResult::Finished)
    }
}

// --- seams ------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MarkSeamParams {
    /// Take the seam off instead.
    pub clear: bool,
}

/// Marks the selected edges as seams — where Unwrap cuts the surface open.
pub struct MarkSeam;

impl Operator for MarkSeam {
    type Params = MarkSeamParams;
    fn id() -> &'static str {
        "mesh.mark_seam"
    }
    fn label() -> &'static str {
        "Mark Seam"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let changed = for_each_edit_mesh(doc, |_, mesh| {
            let selected = selection::selected(mesh, kreate_mesh::Domain::Edge);
            if selected.is_empty() {
                return false;
            }
            let n = mesh.num_edges() as usize;
            let mut seams = mesh
                .attributes()
                .get_bool(kreate_mesh::attribute::names::SEAM)
                .filter(|s| s.len() == n)
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![false; n]);
            let mut any = false;
            for e in selected {
                if let Some(s) = seams.get_mut(e as usize) {
                    any |= *s == params.clear;
                    *s = !params.clear;
                }
            }
            if any {
                mesh.attributes_mut().insert(
                    kreate_mesh::attribute::names::SEAM,
                    kreate_mesh::Domain::Edge,
                    kreate_mesh::AttrData::Bool(std::sync::Arc::new(seams)),
                );
            }
            any
        })?;
        Ok(if changed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

// --- knife ------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct KnifeParams {
    /// The path the knife was drawn through: points on the active mesh's
    /// surface, in its own space, three numbers each.
    pub points: Vec<f64>,
    /// The way the view looked while it was drawn, in the mesh's space: each
    /// stroke cuts what is under it as seen from there.
    pub view: [f64; 3],
}

/// Cuts new edges along a path drawn over the surface — Blender's `K`.
pub struct KnifeCut;

impl Operator for KnifeCut {
    type Params = KnifeParams;

    fn id() -> &'static str {
        "mesh.knife"
    }
    fn label() -> &'static str {
        "Knife"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let path: Vec<glam::Vec3> = params
            .points
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| glam::Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect();
        let view = glam::DVec3::from_array(params.view).as_vec3();
        if path.len() < 2 || !path.iter().all(|p| p.is_finite()) || !view.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let cut = with_bmesh(doc, |bm| {
            edit::knife(bm, &path, view).map_err(|e| OpError::Failed(e.to_string()))
        })?;
        Ok(if cut.faces.is_empty() {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- inset ------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct InsetParams {
    /// How far to shrink the face inside its own outline.
    #[param(min = 0.0, soft_max = 2.0, unit = "length")]
    pub thickness: f64,
    /// How far to move the shrunken face along its normal.
    #[param(soft_min = -2.0, soft_max = 2.0, unit = "length")]
    pub depth: f64,
    /// Inset each face on its own instead of the selection as one region.
    pub individual: bool,
}

impl Default for InsetParams {
    fn default() -> Self {
        Self {
            thickness: 0.1,
            depth: 0.0,
            individual: false,
        }
    }
}

/// Shrinks the selected faces and frames them with a band of quads.
pub struct InsetFaces;

impl Operator for InsetFaces {
    type Params = InsetParams;

    fn id() -> &'static str {
        "mesh.inset"
    }
    fn label() -> &'static str {
        "Inset Faces"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let out = each_bmesh(doc, |bm| {
            let faces = bm.selected_faces();
            if faces.is_empty() {
                return Ok(None);
            }
            let t = params.thickness as f32;
            let d = params.depth as f32;
            let result = if params.individual {
                edit::inset_individual(bm, &faces, t, d)
            } else {
                edit::inset_region(bm, &faces, t, d)
            };
            result.map(Some).map_err(|e| OpError::Failed(e.to_string()))
        })?
        .into_iter()
        .flatten()
        // The largest of them, when several meshes inset something. What the
        // caller reads off this is whether anything happened and whether the
        // thickness had to be clamped; adding several meshes' face lists
        // together would answer neither question about any of them.
        .reduce(|a, b| {
            if a.inner_faces.len() >= b.inner_faces.len() {
                a
            } else {
                b
            }
        });
        Ok(match out {
            Some(inset) if !inset.inner_faces.is_empty() => OpResult::Finished,
            _ => OpResult::Cancelled,
        })
    }
}

// --- bridge -----------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct BridgeParams {}

/// Joins two open borders with a band of quads.
pub struct BridgeEdgeLoops;

impl Operator for BridgeEdgeLoops {
    type Params = BridgeParams;

    fn id() -> &'static str {
        "mesh.bridge"
    }
    fn label() -> &'static str {
        "Bridge Edge Loops"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let empty = nothing_selected_anywhere(doc.document());
        let built: usize = each_bmesh(doc, |bm| {
            // Fall back to every open border when nothing is selected anywhere:
            // two open ends is the common case, and making the user select both
            // first is friction with no purpose.
            if bm.selected_edges().is_empty() {
                if !empty {
                    return Ok(0);
                }
                let Some(borders) = edit::open_borders(bm) else {
                    return Ok(0);
                };
                let [a, b] = &borders[..] else { return Ok(0) };
                let (a, b) = (a.clone(), b.clone());
                return edit::bridge_loops(bm, &a, &b)
                    .map(|out| out.faces.len())
                    .map_err(|e| OpError::Failed(e.to_string()));
            }
            match edit::bridge_selected(bm) {
                Ok(out) => Ok(out.faces.len()),
                Err(e) => Err(OpError::Failed(e.to_string())),
            }
        })?
        .into_iter()
        .sum();
        Ok(if built == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- merge and weld ---------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct MergeParams {
    /// Where the merged vertices end up.
    #[param(variants("centre", "first", "last"))]
    pub target: u32,
    /// Merge everything closer together than this instead of merging the
    /// selection into one point. Zero merges the selection.
    #[param(min = 0.0, soft_max = 1.0, step = 0.0001, unit = "length")]
    pub distance: f64,
}

impl Default for MergeParams {
    fn default() -> Self {
        Self {
            target: 0,
            distance: 0.0,
        }
    }
}

/// Collapses vertices together.
pub struct MergeVerts;

impl Operator for MergeVerts {
    type Params = MergeParams;

    fn id() -> &'static str {
        "mesh.merge"
    }
    fn label() -> &'static str {
        "Merge"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let target = match params.target {
            1 => edit::MergeTarget::First,
            2 => edit::MergeTarget::Last,
            _ => edit::MergeTarget::Centre,
        };
        let empty = nothing_selected_anywhere(doc.document());
        let out = each_bmesh(doc, |bm| {
            let result = if params.distance > 0.0 {
                // By distance: the selection if there is one, the whole mesh
                // otherwise. Welding a whole imported mesh is the common case,
                // and making the user select everything first is friction with
                // no purpose.
                let mut verts = bm.selected_verts();
                if verts.is_empty() && empty {
                    verts = bm.vert_ids().collect();
                }
                // A distance longer than the mesh's own detail does not weld
                // seams, it crushes features — a thread's flanks into each
                // other — and opens the solid. A closed mesh that would come
                // out open is left as it was, and the reason said.
                let closed = kreate_mesh::measure::is_closed(&bm.to_mesh_data());
                let backup = closed.then(|| bm.clone());
                let merged = edit::weld_by_distance(bm, &verts, params.distance as f32);
                if let (Some(backup), Ok(_)) = (backup, &merged) {
                    if !kreate_mesh::measure::is_closed(&bm.to_mesh_data()) {
                        *bm = backup;
                        return Err(OpError::Failed(format!(
                            "merging within {} m would open this closed mesh: it has detail finer than that. Use a smaller distance",
                            params.distance
                        )));
                    }
                }
                merged
            } else {
                edit::merge_selected(bm, target)
            };
            result.map_err(|e| OpError::Failed(e.to_string()))
        })?;
        let removed: usize = out.iter().map(|m| m.removed).sum();
        Ok(if removed == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- selection --------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelectAction {
    #[default]
    All,
    None,
    Invert,
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectParams {
    /// What to do to the selection.
    #[param(variants(
        "all",
        "none",
        "invert",
        "loop",
        "ring",
        "face_loop",
        "more",
        "less",
        "linked"
    ))]
    pub action: u32,
    /// Which element type an inversion, a grow or a shrink works in.
    #[param(variants("vert", "edge", "face"))]
    pub mode: u32,
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SelectSimilarParams {
    /// What to compare. See `kreate_mesh::selection::SIMILAR`.
    #[param(variants(
        "face_normal",
        "face_area",
        "face_coplanar",
        "face_material",
        "face_sides",
        "edge_length",
        "edge_direction",
        "vert_normal",
        "vert_faces"
    ))]
    pub kind: u32,
    /// How close counts: radians for a direction, a fraction for a size, a
    /// length for a plane's offset.
    #[param(min = 0.0, soft_max = 1.0)]
    pub threshold: f64,
}

impl Default for SelectSimilarParams {
    fn default() -> Self {
        Self {
            kind: 0,
            threshold: 0.01,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectShortestPathParams {
    /// The element clicked, in the domain `mode` names.
    #[param(min = 0, unit = "count")]
    pub element: u32,
    #[param(variants("vert", "edge", "face"))]
    pub mode: u32,
}

/// Blender's Ctrl+click in edit mode: the shortest way from the selection to
/// the clicked element joins the selection.
pub struct SelectShortestPath;

impl Operator for SelectShortestPath {
    type Params = SelectShortestPathParams;

    fn id() -> &'static str {
        "mesh.select_shortest_path"
    }
    fn label() -> &'static str {
        "Select Shortest Path"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mode = match params.mode {
            1 => SelectMode::Edge,
            2 => SelectMode::Face,
            _ => SelectMode::Vert,
        };
        let found = with_mesh(doc, |mesh| {
            selection::select_shortest_path(mesh, mode, params.element)
        })?;
        Ok(if found {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

/// Blender's Shift+G: everything like what is selected joins it.
pub struct SelectSimilar;

impl Operator for SelectSimilar {
    type Params = SelectSimilarParams;

    fn id() -> &'static str {
        "mesh.select_similar"
    }
    fn label() -> &'static str {
        "Select Similar"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let grew = for_each_edit_mesh(doc, |_, mesh| {
            selection::select_similar(mesh, params.kind as usize, params.threshold)
        })?;
        Ok(if grew {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

/// Changes the mesh element selection.
pub struct SelectElements;

impl Operator for SelectElements {
    type Params = SelectParams;

    fn id() -> &'static str {
        "mesh.select"
    }
    fn label() -> &'static str {
        "Select"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mode = match params.mode {
            1 => SelectMode::Edge,
            2 => SelectMode::Face,
            _ => SelectMode::Vert,
        };
        // All, none and invert are layer writes and go straight to the mesh —
        // no decode, no encode, and the geometry arrays stay shared so the
        // renderer rewrites three flag buffers rather than the model.
        //
        // Loop, ring and face-loop walk radial and disk cycles, which is what
        // the edit structure is for, and pay for it.
        if params.action <= 2 {
            // Every mesh in edit mode: "select all" with three objects open
            // means all of all three, and a user who wanted one of them would
            // not have put three in.
            //
            // The session's adjacency map belongs to the *target* mesh, so it
            // goes only to that one; the others recompute, which is a pass over
            // their edges and is exactly what having no session costs. Only
            // inversion reads it at all.
            let target = target_mesh(doc.document());
            let adjacency = edit_adjacency(doc);
            let grew = for_each_edit_mesh(doc, |t, mesh| {
                match params.action {
                    1 => selection::select_none(mesh),
                    2 => match &adjacency {
                        Some(map) if Some(t.mesh) == target => {
                            selection::select_invert_with(mesh, mode, map)
                        }
                        _ => selection::select_invert(mesh, mode),
                    },
                    _ => selection::select_all(mesh),
                }
                true
            })?;
            return Ok(if grew {
                OpResult::Finished
            } else {
                OpResult::Cancelled
            });
        }

        // More, less and linked: layer writes too, over the mesh's own
        // adjacency, and every mesh in edit mode like select all.
        if params.action >= 6 {
            return Ok(
                if for_each_edit_mesh(doc, |_, mesh| {
                    let corner_edges = mesh.corner_edges();
                    match params.action {
                        6 => selection::select_more(mesh, mode, &corner_edges),
                        7 => selection::select_less(mesh, mode, &corner_edges),
                        _ => selection::select_linked(mesh, &corner_edges),
                    }
                })? {
                    OpResult::Finished
                } else {
                    OpResult::Cancelled
                },
            );
        }

        let grew = each_bmesh(doc, |bm| {
            // Loop, ring and face-loop grow from what is already selected.
            // Without a click to start from they have no seed, so an empty
            // selection means there is nothing to do rather than an error.
            let seeds = bm.selected_edges();
            if seeds.is_empty() {
                return Ok(false);
            }
            for e in seeds {
                match params.action {
                    3 => bm.select_edge_loop(e),
                    4 => bm.select_edge_ring(e),
                    _ => bm.select_face_loop(e),
                };
            }
            Ok(true)
        })?
        .into_iter()
        .any(|g| g);
        // A walk goes wherever the mesh goes, hidden or not; what is hidden
        // does not stay selected.
        if grew {
            for_each_edit_mesh(doc, |_, mesh| {
                selection::deselect_hidden(mesh);
                true
            })?;
        }
        Ok(if grew {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct PickPieceParams {
    /// The element under the pointer, in the domain `mode` names.
    #[param(min = 0, unit = "count")]
    pub element: u32,
    #[param(variants("vert", "edge", "face"))]
    pub mode: u32,
    /// Take the piece out of the selection — `Shift+L`.
    pub deselect: bool,
}

/// Blender's `L`: the whole piece under the pointer, added to the selection.
pub struct SelectLinkedPick;

impl Operator for SelectLinkedPick {
    type Params = PickPieceParams;
    fn id() -> &'static str {
        "mesh.select_linked_pick"
    }
    fn label() -> &'static str {
        "Select Linked Pick"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mesh_id =
            target_mesh(doc.document()).ok_or_else(|| OpError::Failed("no mesh to edit".into()))?;
        let changed = with_named_mesh(doc, mesh_id, |mesh| {
            let seed: Vec<u32> = match params.mode {
                2 if params.element < mesh.num_faces() => mesh.face_verts(params.element).to_vec(),
                1 => mesh
                    .edges()
                    .get(params.element as usize)
                    .map(|e| e.to_vec())
                    .unwrap_or_default(),
                0 if params.element < mesh.num_verts() => vec![params.element],
                _ => Vec::new(),
            };
            if seed.is_empty() {
                return false;
            }
            selection::select_linked_from(mesh, &seed, params.deselect)
        })?;
        Ok(if changed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct LoopPickParams {
    /// The edge clicked.
    #[param(min = 0, unit = "count")]
    pub edge: u32,
    /// A ring across the edge rather than the loop along it — `Ctrl+Alt`.
    pub ring: bool,
    /// Add to the selection — `Shift+Alt`.
    pub extend: bool,
    /// Faces rather than edges: in face mode a loop is a loop of faces.
    pub faces: bool,
}

/// Blender's Alt+click: the edge loop through the edge clicked.
pub struct SelectLoopPick;

impl Operator for SelectLoopPick {
    type Params = LoopPickParams;
    fn id() -> &'static str {
        "mesh.select_loop_pick"
    }
    fn label() -> &'static str {
        "Select Loop"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mesh_id =
            target_mesh(doc.document()).ok_or_else(|| OpError::Failed("no mesh to edit".into()))?;
        // The edge by where its ends are: the stored mesh numbers its edges,
        // and the edit structure — after a delete — may number them otherwise.
        let ends = with_named_mesh(doc, mesh_id, |mesh| {
            let [a, b] = *mesh.edges().get(params.edge as usize)?;
            if !params.extend {
                selection::select_none(mesh);
            }
            Some([mesh.positions()[a as usize], mesh.positions()[b as usize]])
        })?;
        let Some([pa, pb]) = ends else {
            return Ok(OpResult::Cancelled);
        };
        let grew = with_bmesh(doc, |bm| {
            let found = bm.edge_ids().find(|e| {
                bm.edge_verts(*e).is_some_and(|[x, y]| {
                    let (px, py) = (bm.position(x), bm.position(y));
                    (px == Some(pa) && py == Some(pb)) || (px == Some(pb) && py == Some(pa))
                })
            });
            let Some(e) = found else { return Ok(0) };
            Ok(if params.faces {
                bm.select_face_loop(e)
            } else if params.ring {
                bm.select_edge_ring(e)
            } else {
                bm.select_edge_loop(e)
            })
        })?;
        Ok(if grew > 0 {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct HideParams {
    /// Hide everything that is *not* selected — `Shift+H`.
    pub unselected: bool,
    /// Which element type the selection is read in.
    #[param(variants("vert", "edge", "face"))]
    pub mode: u32,
}

/// Puts the selection out of sight in edit mode — Blender's `H`.
///
/// Hidden elements are not drawn, cannot be picked or selected, and so no
/// operator reaches them; object mode shows the mesh whole. The state is
/// layers on the mesh, so it is undone, saved and replayed like a selection.
pub struct HideElements;

impl Operator for HideElements {
    type Params = HideParams;

    fn id() -> &'static str {
        "mesh.hide"
    }
    fn label() -> &'static str {
        "Hide"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mode = match params.mode {
            1 => SelectMode::Edge,
            2 => SelectMode::Face,
            _ => SelectMode::Vert,
        };
        let any = for_each_edit_mesh(doc, |_, mesh| {
            let corner_edges = mesh.corner_edges();
            selection::hide(mesh, mode, params.unselected, &corner_edges)
        })?;
        Ok(if any {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RevealParams {}

/// Brings back everything hidden, selected — Blender's `Alt+H`.
pub struct RevealElements;

impl Operator for RevealElements {
    type Params = RevealParams;

    fn id() -> &'static str {
        "mesh.reveal"
    }
    fn label() -> &'static str {
        "Reveal Hidden"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let any = for_each_edit_mesh(doc, |_, mesh| selection::reveal(mesh))?;
        Ok(if any {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectPickParams {
    /// Index of the element to select, in the domain named by `mode`.
    #[param(min = 0, unit = "count")]
    pub element: u32,
    /// Which element type the index refers to.
    #[param(variants("vert", "edge", "face"))]
    pub mode: u32,
    /// Add to the selection instead of replacing it, toggling the element.
    ///
    /// What Shift+Click does everywhere: a second Shift+Click on the same
    /// element takes it back out, so a mis-click is undone by repeating it
    /// rather than by starting the selection again.
    pub extend: bool,
    /// Which object's mesh the element belongs to, as its arena slot **plus
    /// one**.
    ///
    /// Zero means the target mesh — what a keystroke or a script means when it
    /// does not say, and what every caller meant before edit mode could hold
    /// more than one. Plus-one for the reason the id buffer uses it too: slot
    /// zero is a real object, so zero has to mean "not stated".
    ///
    /// An object that is not in edit mode is refused rather than selected in.
    /// The id buffer covers the whole scene, so a click in edit mode lands on
    /// objects whose cage is not even drawn, and selecting elements the user
    /// cannot see is worse than a click that does nothing.
    #[param(min = 0, unit = "count")]
    pub object: u32,
}

/// Selects one element by index — what a click in the viewport resolves to.
///
/// The index comes from the GPU id buffer and is an index into the *document*
/// mesh, which is why edit mode draws the cage rather than the evaluated
/// result: an index into a mirrored copy names a different face.
///
/// An operator like every other click-driven change, so a selection is undoable,
/// scriptable and recordable without any of that being thought about here.
pub struct SelectPick;

impl Operator for SelectPick {
    type Params = SelectPickParams;

    fn id() -> &'static str {
        "mesh.select_pick"
    }
    fn label() -> &'static str {
        "Select Element"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mode = match params.mode {
            1 => SelectMode::Edge,
            2 => SelectMode::Face,
            _ => SelectMode::Vert,
        };
        let domain = mode.domain();
        // Which mesh the click landed in. Named by the caller when it knows —
        // the viewport does, because the id buffer stores the object alongside
        // the element — and the target otherwise.
        let target = target_mesh(doc.document());
        let mesh_id = if params.object == 0 {
            target
        } else {
            let slot = params.object - 1;
            let targets = edit_targets(doc.document());
            doc.document()
                .object_at_slot(slot)
                .and_then(|id| targets.iter().find(|t| t.object == id))
                .map(|t| t.mesh)
        };
        // A click on something not in edit mode. Not an error: the id buffer
        // covers the whole scene, so this is an ordinary miss.
        let Some(mesh_id) = mesh_id else {
            return Ok(OpResult::Cancelled);
        };
        // The session's map is the target's, so it is only useful when the
        // click landed there.
        let adjacency = edit_adjacency(doc).filter(|_| Some(mesh_id) == target);
        // A plain click replaces the selection, and "the selection" is every
        // mesh's — clicking a vertex of one object has to clear what was
        // selected in the others, or the next transform moves geometry the user
        // believes they deselected.
        if !params.extend {
            for_each_edit_mesh(doc, |_, mesh| {
                selection::select_none(mesh);
                true
            })?;
        }
        let changed = with_named_mesh(doc, mesh_id, |mesh| {
            // An index the mesh does not have is not an error worth failing on:
            // the id buffer describes a frame the document may have moved on
            // from, which is normal a frame or two after an edit.
            if params.element as usize >= mesh.domain_len(domain) {
                return false;
            }
            let already = selection::is_selected(mesh, domain, params.element);
            // A plain click replaces the selection — already cleared above,
            // across every mesh; Shift+Click toggles this one element and
            // leaves the rest.
            let on = if params.extend { !already } else { true };
            selection::set_selected(mesh, domain, params.element, on);
            // Flushed so the other domains agree: selecting a face has to mark
            // its edges and vertices, or the next operator acts on a selection
            // the user cannot see.
            flush(mesh, mode, &adjacency);
            true
        })?;
        Ok(if changed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

/// Clears the element selection — what clicking empty space resolves to.
///
/// Separate from `mesh.select` with `action = none` only in that it is the thing
/// a click dispatches; it shares the implementation.
pub struct SelectNothing;

impl Operator for SelectNothing {
    type Params = crate::ops::mesh::EmptyParams;

    fn id() -> &'static str {
        "mesh.select_nothing"
    }
    fn label() -> &'static str {
        "Deselect All"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        // Every mesh, because clicking empty space means "nothing is
        // selected" and not "nothing is selected in the active object".
        let had = for_each_edit_mesh(doc, |_, mesh| {
            let had = selection::any_selected(mesh);
            selection::select_none(mesh);
            had
        })?;
        // Nothing was selected, so nothing changed: clicking empty space twice
        // should not fill the history with steps that undo to the same state.
        Ok(if had {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct EmptyParams {}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectSetParams {
    /// The elements to select, by index in the domain named by `mode`.
    pub elements: Vec<u32>,
    /// Which element type the indices refer to.
    #[param(variants("vert", "edge", "face"))]
    pub mode: u32,
    /// Add to the selection instead of replacing it.
    pub extend: bool,
    /// Remove the listed elements from the selection instead of adding them.
    ///
    /// What Ctrl+drag does: a box that deselects. Only meaningful with `extend`,
    /// because without it the selection is being replaced anyway.
    pub deselect: bool,
    /// Which objects' meshes the elements belong to, as arena slots plus one
    /// (as `mesh.select_pick` names one), with `counts` saying how many of
    /// `elements` are each one's, in order. Empty: every element is the
    /// target mesh's — what a box in single-object edit mode means.
    pub objects: Vec<u32>,
    pub counts: Vec<u32>,
}

/// Selects a resolved set of elements — what a box, lasso or circle drag commits.
///
/// The parameter is the *set*, not the screen region that produced it. That is
/// deliberate: a region depends on a camera, cannot be replayed headlessly, and
/// re-running it after the view has moved silently selects something else. A set
/// replays exactly, in a script or from a file, and `Copy as Python` produces a
/// line that pastes back and works.
///
/// The set comes from a compute pass over the id buffer rather than from
/// projecting vertices on the CPU, so the cost is independent of how many
/// elements the mesh has — see `kreate-render`'s region selection.
pub struct SelectSet;

impl Operator for SelectSet {
    type Params = SelectSetParams;

    fn id() -> &'static str {
        "mesh.select_set"
    }
    fn label() -> &'static str {
        "Select Elements"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mode = match params.mode {
            1 => SelectMode::Edge,
            2 => SelectMode::Face,
            _ => SelectMode::Vert,
        };
        // An empty box that replaces the selection is a deselect-all, which is
        // what dragging over nothing should do. An empty box that extends is
        // nothing at all.
        //
        // A fast path rather than a correctness guard: the general path below
        // reaches the same answer, and the dispatcher restores the document on a
        // cancel either way. What this avoids is the `BMesh` round trip — a
        // whole-mesh conversion — on a gesture the user makes by accident
        // whenever a shift-drag misses.
        if params.elements.is_empty() && params.extend {
            return Ok(OpResult::Cancelled);
        }
        if !params.objects.is_empty() {
            return select_set_many(params, mode, doc);
        }

        let domain = mode.domain();
        let adjacency = edit_adjacency(doc);
        let changed = with_mesh(doc, |mesh| {
            if !params.extend {
                selection::select_none(mesh);
            }
            let on = !params.deselect;
            let count = mesh.domain_len(domain);
            let mut touched = false;
            for index in &params.elements {
                // Indices the mesh does not have are skipped rather than failing
                // the whole operation: the id buffer is a frame or two old, so a
                // drag that ends just after an edit can name elements that have
                // gone.
                if (*index as usize) < count {
                    selection::set_selected(mesh, domain, *index, on);
                    touched = true;
                }
            }
            // Flushed once at the end rather than per element: selecting a
            // thousand faces has to mark their edges and vertices, and doing
            // that a thousand times over is the difference between a box select
            // that lands instantly and one that hitches.
            flush(mesh, mode, &adjacency);
            touched || !params.extend
        })?;
        Ok(if changed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

/// [`SelectSet`] across several meshes in edit mode: each named object's
/// share of the elements into its own mesh, and — replacing — every edited
/// mesh cleared first, so a box over one object deselects the others.
fn select_set_many(
    params: &SelectSetParams,
    mode: SelectMode,
    doc: &mut DocMut<'_>,
) -> Result<OpResult, OpError> {
    let domain = mode.domain();
    let targets = edit_targets(doc.document());
    let target = target_mesh(doc.document());
    let mut groups = Vec::new();
    let mut at = 0usize;
    for (slot, count) in params.objects.iter().zip(&params.counts) {
        let count = *count as usize;
        let elements = params.elements.get(at..at + count).unwrap_or(&[]).to_vec();
        at += count;
        let mesh = slot.checked_sub(1).and_then(|s| {
            doc.document()
                .object_at_slot(s)
                .and_then(|id| targets.iter().find(|t| t.object == id))
                .map(|t| t.mesh)
        });
        if let Some(mesh) = mesh {
            groups.push((mesh, elements));
        }
    }
    let mut changed = false;
    if !params.extend {
        changed |= for_each_edit_mesh(doc, |_, mesh| {
            let had = selection::any_selected(mesh);
            selection::select_none(mesh);
            had
        })?;
    }
    let on = !params.deselect;
    for (mesh_id, elements) in groups {
        let adjacency = edit_adjacency(doc).filter(|_| Some(mesh_id) == target);
        changed |= with_named_mesh(doc, mesh_id, |mesh| {
            let count = mesh.domain_len(domain);
            let mut touched = false;
            for index in &elements {
                if (*index as usize) < count {
                    selection::set_selected(mesh, domain, *index, on);
                    touched = true;
                }
            }
            flush(mesh, mode, &adjacency);
            touched
        })?;
    }
    Ok(if changed {
        OpResult::Finished
    } else {
        OpResult::Cancelled
    })
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SeparateParams {}

/// Moves the selected faces into a new object.
///
/// The new object takes the source's transform, so the geometry does not move:
/// a separate is a change of *ownership*, not a change of shape.
pub struct Separate;

impl Operator for Separate {
    type Params = SeparateParams;

    fn id() -> &'static str {
        "mesh.separate"
    }
    fn label() -> &'static str {
        "Separate Selection"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let source = doc
            .document()
            .selection
            .active
            .or_else(|| doc.document().selection.objects.first().copied());
        let Some(source) = source else {
            return Err(OpError::Failed("no object to separate from".into()));
        };
        let (name, transform, parent, parent_inverse, parent_bone) = {
            let Some(object) = doc.document().object(source) else {
                return Err(OpError::Failed("the object is missing".into()));
            };
            (
                format!("{}.part", object.name),
                object.transform,
                object.parent,
                object.parent_inverse,
                object.parent_bone.clone(),
            )
        };

        let taken = with_bmesh(doc, |bm| Ok(edit::separate_selected(bm)))?;
        let Some(taken) = taken else {
            // Nothing selected, or everything: separating a whole mesh leaves
            // an empty object behind, which is worse than declining.
            return Ok(OpResult::Cancelled);
        };

        let d = doc.document_mut();
        let mesh = d.add_mesh(taken);
        let mut object = Object::new(name, ObjectData::Mesh(mesh));
        // The source's placement, so the geometry stays where it was: the
        // vertices were copied in the source's local space, and reading them
        // in any other space would move them.
        object.transform = transform;
        object.parent = parent;
        object.parent_inverse = parent_inverse;
        object.parent_bone = parent_bone;
        let new_id = d.add_object(object);

        // Selected and active, because the next thing a user does is move it.
        d.selection.objects = vec![new_id];
        d.selection.active = Some(new_id);
        Ok(OpResult::Finished)
    }
}

// --- crease -------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct CreaseParams {
    /// How sharp the selected edges stay under a subdivide modifier: 0 is
    /// smooth, 1 is sharp at every level, and between them the edge is sharp
    /// for a few levels and then relaxes — a softly rounded edge.
    #[param(min = 0.0, max = 1.0)]
    pub value: f64,
}

impl Default for CreaseParams {
    fn default() -> Self {
        Self { value: 1.0 }
    }
}

/// Sets the subdivision crease of the selected edges — Blender's Shift+E.
pub struct Crease;

impl Operator for Crease {
    type Params = CreaseParams;

    fn id() -> &'static str {
        "mesh.crease"
    }
    fn label() -> &'static str {
        "Edge Crease"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let changed: usize = each_bmesh(doc, |bm| {
            let edges = bm.selected_edges();
            Ok(edit::set_crease(bm, &edges, params.value as f32))
        })?
        .into_iter()
        .sum();
        Ok(if changed == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- select by condition ------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SelectWhereParams {
    /// Which elements to test: a vertex by its position, an edge by its
    /// midpoint, a face by its centre.
    #[param(variants("vert", "edge", "face"))]
    pub mode: u32,
    /// The low corner of a world-space box the element must be in, metres.
    #[param(unit = "length")]
    pub min: [f64; 3],
    /// The high corner of that box.
    #[param(unit = "length")]
    pub max: [f64; 3],
    /// The world direction the element must face, e.g. (0, 0, -1) for
    /// undersides: a face's normal, a vertex's (the average of its faces'),
    /// an edge's (its two vertices'). Zero: any direction.
    pub normal: [f64; 3],
    /// How far from `normal` a face may point, in radians.
    #[param(min = 0.0, max = 3.15, unit = "angle")]
    pub angle: f64,
    /// Faces only: the material slot the face must use. −1: any.
    #[param(min = -1.0)]
    pub material: i64,
    /// Add to the selection instead of replacing it.
    pub extend: bool,
    /// Remove the matches from the selection instead of adding them.
    pub deselect: bool,
    /// How far outside the box, in metres, still counts as in it: a box
    /// read off numbers rounded to a tenth of a millimetre misses the
    /// corner it was meant to catch. Negative: a ten-thousandth of the
    /// meshes' size.
    #[param(unit = "length")]
    pub tolerance: f64,
}

impl Default for SelectWhereParams {
    fn default() -> Self {
        Self {
            mode: 0,
            min: [-1.0e6; 3],
            max: [1.0e6; 3],
            normal: [0.0; 3],
            angle: 0.35,
            material: -1,
            extend: false,
            deselect: false,
            tolerance: -1.0,
        }
    }
}

/// Selects the elements that meet a condition rather than a list of indices:
/// the vertices below a height, the faces that point down, the faces of one
/// material — what a script can say without knowing a mesh's numbering.
///
/// Every mesh in edit mode, each in world space, so a box read off
/// `get_scene`'s bounds means the same thing for all of them.
pub struct SelectWhere;

impl Operator for SelectWhere {
    type Params = SelectWhereParams;

    fn id() -> &'static str {
        "mesh.select_where"
    }
    fn label() -> &'static str {
        "Select Where"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let mode = match params.mode {
            1 => SelectMode::Edge,
            2 => SelectMode::Face,
            _ => SelectMode::Vert,
        };
        let margin = if params.tolerance >= 0.0 {
            params.tolerance
        } else {
            // The size of what is being edited, in the world.
            let d = doc.document();
            let size = edit_targets(d)
                .iter()
                .filter_map(|t| {
                    let m = d.mesh(t.mesh)?;
                    let b = m.bounds();
                    let (lo, hi) = (
                        t.to_world.transform_point3(b.min.as_dvec3()),
                        t.to_world.transform_point3(b.max.as_dvec3()),
                    );
                    Some((hi - lo).length())
                })
                .fold(0.0, f64::max);
            size * 1e-4
        };
        // Either corner may be given first: a box is its two corners,
        // whichever way round, and one typed back to front caught nothing.
        let (a, b) = (
            glam::DVec3::from_array(params.min),
            glam::DVec3::from_array(params.max),
        );
        let (lo, hi) = (
            a.min(b) - glam::DVec3::splat(margin),
            a.max(b) + glam::DVec3::splat(margin),
        );
        // The nearest tested point outside the box, for saying where things
        // are when nothing was caught.
        let nearest: std::cell::Cell<Option<(f64, glam::DVec3)>> = std::cell::Cell::new(None);
        let inside = |p: glam::DVec3| {
            let hit = p.cmpge(lo).all() && p.cmple(hi).all();
            if !hit {
                let gap = (p.clamp(lo, hi) - p).length();
                if nearest.get().is_none_or(|(d, _)| gap < d) {
                    nearest.set(Some((gap, p)));
                }
            }
            hit
        };
        let wanted = glam::DVec3::from_array(params.normal).normalize_or_zero();
        let cos_limit = params.angle.cos();
        let mut matched = 0usize;
        let changed = for_each_edit_mesh(doc, |target, mesh| {
            let world = target.to_world;
            let to_world_normal = world.matrix3.inverse().transpose();
            let at = |p: glam::Vec3| world.transform_point3(p.as_dvec3());
            let positions = mesh.positions().to_vec();
            let facing = |n: glam::Vec3| {
                wanted == glam::DVec3::ZERO
                    || (to_world_normal * n.as_dvec3())
                        .normalize_or_zero()
                        .dot(wanted)
                        >= cos_limit
            };
            let vertex_normals = if wanted != glam::DVec3::ZERO && mode != SelectMode::Face {
                mesh.compute_vertex_normals()
            } else {
                Vec::new()
            };
            let hits: Vec<u32> = match mode {
                SelectMode::Vert => (0..mesh.num_verts())
                    .filter(|v| {
                        inside(at(positions[*v as usize]))
                            && (vertex_normals.is_empty() || facing(vertex_normals[*v as usize]))
                    })
                    .collect(),
                SelectMode::Edge => mesh
                    .edges()
                    .iter()
                    .enumerate()
                    .filter(|(_, [a, b])| {
                        inside(at((positions[*a as usize] + positions[*b as usize]) * 0.5))
                            && (vertex_normals.is_empty()
                                || facing(
                                    vertex_normals[*a as usize] + vertex_normals[*b as usize],
                                ))
                    })
                    .map(|(e, _)| e as u32)
                    .collect(),
                SelectMode::Face => {
                    let normals = mesh.compute_face_normals();
                    let slots = mesh
                        .attributes()
                        .get_i32(kreate_mesh::attribute::names::MATERIAL_INDEX)
                        .map(<[i32]>::to_vec);
                    (0..mesh.num_faces())
                        .filter(|f| {
                            let ring = mesh.face_verts(*f);
                            let centre = ring
                                .iter()
                                .map(|v| positions[*v as usize])
                                .sum::<glam::Vec3>()
                                / ring.len() as f32;
                            if !inside(at(centre)) {
                                return false;
                            }
                            if !facing(normals[*f as usize]) {
                                return false;
                            }
                            params.material < 0
                                || slots
                                    .as_ref()
                                    .and_then(|s| s.get(*f as usize))
                                    .copied()
                                    .unwrap_or(0) as i64
                                    == params.material
                        })
                        .collect()
                }
            };
            if !params.extend {
                selection::select_none(mesh);
            }
            let domain = mode.domain();
            for index in &hits {
                selection::set_selected(mesh, domain, *index, !params.deselect);
            }
            selection::flush(mesh, mode);
            matched += hits.len();
            !hits.is_empty() || !params.extend
        })?;
        // Nothing caught is a miss to say, not a quiet success: whatever the
        // next line does to "the selection" would act on nothing.
        if matched == 0 && !params.deselect {
            let what = ["vertex", "edge midpoint", "face center"][params.mode.min(2) as usize];
            let near = match nearest.get() {
                Some((gap, p)) => format!(
                    "; the nearest {what} outside the box is at ({:.4}, {:.4}, {:.4}), {gap:.6} m away — tolerance={:.6} would take it",
                    p.x, p.y, p.z, gap + margin + 1e-9
                ),
                None => String::new(),
            };
            let facing = if wanted == glam::DVec3::ZERO {
                String::new()
            } else {
                format!(
                    " facing ({:.2}, {:.2}, {:.2})",
                    wanted.x, wanted.y, wanted.z
                )
            };
            return Err(OpError::Failed(format!(
                "nothing matched: no {what}{facing} in the box from ({:.4}, {:.4}, {:.4}) to ({:.4}, {:.4}, {:.4}){near}. \
                 get_mesh(faces=true) lists every face's index and center",
                lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
            )));
        }
        Ok(if changed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<SelectWhere>();
    registry.register::<Crease>();
    registry.register::<DeleteElements>();
    registry.register::<SelectPick>();
    registry.register::<SelectSet>();
    registry.register::<SelectNothing>();
    registry.register::<DissolveElements>();
    registry.register::<BevelEdges>();
    registry.register::<Fillet>();
    registry.register::<BridgeEdgeLoops>();
    registry.register::<ExtrudeRegion>();
    registry.register::<MedianSet>();
    registry.register::<CheckSelect>();
    registry.register::<DissolveLimited>();
    registry.register::<FillHole>();
    registry.register::<InsetFaces>();
    registry.register::<LoopCut>();
    registry.register::<EdgeSlide>();
    registry.register::<Symmetrize>();
    registry.register::<MarkSharp>();
    registry.register::<MergeVerts>();
    registry.register::<SelectElements>();
    registry.register::<SelectSimilar>();
    registry.register::<SelectShortestPath>();
    registry.register::<HideElements>();
    registry.register::<KnifeCut>();
    registry.register::<MarkSeam>();
    registry.register::<ExtrudeMove>();
    registry.register::<SelectLinkedPick>();
    registry.register::<SelectLoopPick>();
    registry.register::<RevealElements>();
    registry.register::<RecalculateNormals>();
    registry.register::<ShadeFaces>();
    registry.register::<SpinFaces>();
    registry.register::<SplitGeometry>();
    registry.register::<Subdivide>();
    registry.register::<TranslateSelection>();
    registry.register::<RotateSelection>();
    registry.register::<ScaleSelection>();
    registry.register::<Separate>();
}

/// The selected faces' shared normal in the world, across every mesh in edit
/// mode — the direction `E` drags an extrusion along. Each face's Newell normal,
/// area-weighted as Newell's is, through its object's normal matrix; `None`
/// when no face is selected or they cancel out.
/// Whether any mesh in edit mode has a face selected.
pub fn any_face_selected(doc: &Document) -> bool {
    edit_targets(doc).iter().any(|target| {
        doc.mesh(target.mesh)
            .and_then(|m| {
                m.attributes()
                    .get_bool(kreate_mesh::attribute::names::SELECT_FACE)
            })
            .is_some_and(|s| s.iter().any(|on| *on))
    })
}

pub fn selected_faces_normal(doc: &Document) -> Option<glam::DVec3> {
    let mut sum = glam::DVec3::ZERO;
    for target in edit_targets(doc) {
        let Some(mesh) = doc.mesh(target.mesh) else {
            continue;
        };
        let Some(selected) = mesh
            .attributes()
            .get_bool(kreate_mesh::attribute::names::SELECT_FACE)
        else {
            continue;
        };
        let offsets = mesh.face_offsets();
        let corners = mesh.corner_verts();
        let positions = mesh.positions();
        let normal_matrix = target.to_world.matrix3.inverse().transpose();
        for (face, on) in selected.iter().enumerate() {
            if !*on || face + 1 >= offsets.len() {
                continue;
            }
            let ring = &corners[offsets[face] as usize..offsets[face + 1] as usize];
            let mut n = glam::DVec3::ZERO;
            for (i, &a) in ring.iter().enumerate() {
                let p = positions[a as usize].as_dvec3();
                let q = positions[ring[(i + 1) % ring.len()] as usize].as_dvec3();
                n += glam::DVec3::new(
                    (p.y - q.y) * (p.z + q.z),
                    (p.z - q.z) * (p.x + q.x),
                    (p.x - q.x) * (p.y + q.y),
                );
            }
            sum += normal_matrix * n;
        }
    }
    let n = sum.normalize_or_zero();
    (n != glam::DVec3::ZERO).then_some(n)
}

// --- edge slide ---------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct EdgeSlideParams {
    /// How far across: one is all the way to the neighbouring vertices on
    /// one side, minus one on the other.
    #[param(min = -1.0, max = 1.0, unit = "factor")]
    pub factor: f64,
}

/// Blender's `G G`: the selected edges slid across the faces beside them.
pub struct EdgeSlide;

impl Operator for EdgeSlide {
    type Params = EdgeSlideParams;
    fn id() -> &'static str {
        "mesh.edge_slide"
    }
    fn label() -> &'static str {
        "Edge Slide"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let moved = with_bmesh(doc, |bm| Ok(edit::edge_slide(bm, params.factor as f32)))?;
        Ok(if moved == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

/// Where an edge slide of the active mesh's selection goes, in the world:
/// the selected vertices' middle, and the middle of where a factor of one
/// takes them. What the drag measures the pointer against.
pub fn edge_slide_handle(doc: &Document) -> Option<(glam::DVec3, glam::DVec3)> {
    let object = target_object(doc)?;
    let mesh = doc.mesh(target_mesh(doc)?)?;
    let bm = kreate_mesh::BMesh::from_mesh_data(mesh);
    let rails = edit::slide_rails(&bm);
    let plus: Vec<_> = rails
        .iter()
        .filter_map(|r| Some((r.from, r.plus?)))
        .collect();
    let pairs = if plus.is_empty() {
        rails
            .iter()
            .filter_map(|r| Some((r.from, r.minus?)))
            .collect()
    } else {
        plus
    };
    if pairs.is_empty() {
        return None;
    }
    let n = pairs.len() as f32;
    let from = pairs.iter().map(|p| p.0).sum::<kreate_math::Vec3>() / n;
    let to = pairs.iter().map(|p| p.1).sum::<kreate_math::Vec3>() / n;
    let to_world = doc.world_transform(object);
    Some((
        to_world.transform_point3(from.as_dvec3()),
        to_world.transform_point3(to.as_dvec3()),
    ))
}

// --- symmetrize -------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SymmetrizeParams {
    /// The axis across whose plane, through the object's origin.
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
    /// Which half is kept and copied over.
    #[param(variants("positive_to_negative", "negative_to_positive"))]
    pub direction: u32,
}

/// Blender's Symmetrize: one half of the mesh cut away and the other
/// reflected into its place, welded at the seam.
pub struct Symmetrize;

impl Operator for Symmetrize {
    type Params = SymmetrizeParams;
    fn id() -> &'static str {
        "mesh.symmetrize"
    }
    fn label() -> &'static str {
        "Symmetrize"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let done = with_bmesh(doc, |bm| {
            let mesh = bm.to_mesh_data();
            let out = edit::symmetrize(&mesh, params.axis, params.direction == 1);
            if out.num_faces() == 0 {
                return Ok(false);
            }
            *bm = kreate_mesh::BMesh::from_mesh_data(&out);
            Ok(true)
        })?;
        Ok(if done {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

/// The edge a loop cut starts from, as its two ends in the world: the first
/// selected edge of the mesh [`LoopCut`] cuts, in the order its `position`
/// runs — `0` at the first end, `1` at the second. What `Ctrl+R` slides the
/// cut along with the pointer; `None` when no edge is selected.
pub fn loop_cut_edge(doc: &Document) -> Option<(glam::DVec3, glam::DVec3)> {
    let mesh = doc.mesh(target_mesh(doc)?)?;
    let selected = mesh
        .attributes()
        .get_bool(kreate_mesh::attribute::names::SELECT_EDGE)?;
    let edge = selected.iter().position(|on| *on)?;
    edge_in_world(doc, edge as u32)
}

/// Edge `edge` of the active mesh as its two ends in the world, in the order
/// a loop cut's `position` runs.
pub fn edge_in_world(doc: &Document, edge: u32) -> Option<(glam::DVec3, glam::DVec3)> {
    let object = target_object(doc)?;
    let mesh = doc.mesh(target_mesh(doc)?)?;
    let [a, b] = *mesh.edges().get(edge as usize)?;
    let to_world = doc.world_transform(object);
    let positions = mesh.positions();
    let at = |v: u32| to_world.transform_point3(positions[v as usize].as_dvec3());
    Some((at(a), at(b)))
}

#[cfg(test)]
mod tests {

    /// Blender's falloff curves, halfway out and at the rim.
    #[test]
    fn the_falloff_curves_have_their_shapes() {
        let half = |curve| super::falloff(0.5, 1.0, curve);
        let expect = [0.5, 0.75f32.sqrt(), 0.5f32.sqrt(), 0.25, 0.5, 1.0];
        for (curve, want) in expect.iter().enumerate() {
            assert!(
                (half(curve as u32) - want).abs() < 1e-5,
                "{}: {}",
                super::FALLOFF_CURVES[curve],
                half(curve as u32)
            );
            assert_eq!(
                super::falloff(1.0, 1.0, curve as u32),
                0.0,
                "{} at the rim",
                super::FALLOFF_CURVES[curve]
            );
            assert_eq!(
                super::falloff(0.0, 1.0, curve as u32),
                1.0,
                "{} at the centre",
                super::FALLOFF_CURVES[curve]
            );
        }
    }

    use super::*;
    use crate::{History, Mode, OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_scene::document::{Object, ObjectData};

    fn registry() -> OperatorRegistry {
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        r
    }

    fn params(pairs: &[(&'static str, PropertyValue)]) -> PropertyMap {
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        p
    }

    /// A document holding one cube, selected, in edit mode.
    fn cube_doc() -> (Document, MeshId, Context) {
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
        let obj = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        (
            doc,
            mesh,
            Context {
                mode: Mode::EditMesh,
            },
        )
    }

    #[test]
    fn extruding_a_closed_solid_along_its_own_normal_moves_nothing() {
        // Not a defect: the face normals of a closed surface sum to zero, so the
        // region has no direction to be pushed along. The topology change still
        // happens — the shell is rebuilt and reselected, which is what lets the
        // user then move it by hand — but nothing shifts on its own.
        //
        // Worth pinning down, because the obvious expectation is that the cube
        // inflates, and a future "extrude along each face's own normal" variant
        // is a different operator rather than a fix to this one.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[
                ("action", PropertyValue::Enum(0)),
                ("mode", PropertyValue::Enum(2)),
            ]),
        )
        .unwrap();

        let before = doc.mesh(mesh).unwrap().clone();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.extrude_region",
            &params(&[("offset", PropertyValue::Float(1.0))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), before.num_faces());
        assert_eq!(after.num_verts(), before.num_verts());
        assert_eq!(
            after.bounds().max,
            before.bounds().max,
            "a closed region has no normal to travel along"
        );
        assert_eq!(after.validate(), Vec::new());
    }

    #[test]
    fn extruding_one_face_adds_the_walls_and_undoes_cleanly() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();

        // Select the top face by hand, through the mesh's own layers.
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let top = bm
                .face_ids()
                .max_by(|a, b| {
                    let za = bm.face_normal(*a).normalize_or_zero().z;
                    let zb = bm.face_normal(*b).normalize_or_zero().z;
                    za.partial_cmp(&zb).unwrap()
                })
                .unwrap();
            bm.set_face_selected(top, true);
            bm.flush_selection(SelectMode::Face);
            *m = bm.to_mesh_data();
        }

        let before = doc.mesh(mesh).unwrap().clone();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.extrude_region",
            &params(&[("offset", PropertyValue::Float(3.0))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), 10, "5 sides + lid + 4 walls");
        assert_eq!(after.num_verts(), 12);
        assert!(after.bounds().max.z > 3.9);

        assert!(h.undo(&mut doc));
        let restored = doc.mesh(mesh).unwrap();
        assert_eq!(restored.num_faces(), before.num_faces());
        assert_eq!(restored.num_verts(), before.num_verts());
        assert_eq!(restored.positions(), before.positions());
    }

    #[test]
    fn an_extrude_along_a_direction_goes_that_way_in_the_world() {
        // A cube turned a quarter turn about Z: its top face extruded along
        // world X lands two units along world X, whatever its own axes are.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        let obj = doc.selection.objects[0];
        doc.object_mut(obj).unwrap().transform.rotation =
            glam::DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let top = bm
                .face_ids()
                .max_by(|a, b| {
                    let za = bm.face_normal(*a).normalize_or_zero().z;
                    let zb = bm.face_normal(*b).normalize_or_zero().z;
                    za.partial_cmp(&zb).unwrap()
                })
                .unwrap();
            bm.set_face_selected(top, true);
            bm.flush_selection(SelectMode::Face);
            *m = bm.to_mesh_data();
        }
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.extrude_region",
            &params(&[
                ("offset", PropertyValue::Float(2.0)),
                ("direction", PropertyValue::Vec3([1.0, 0.0, 0.0])),
            ]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), 10);
        // World +X is the turned cube's local -Y.
        let b = after.bounds();
        assert!((b.min.y + 3.0).abs() < 1e-5, "{b:?}");
        assert!((b.max.z - 1.0).abs() < 1e-5, "{b:?}");
        assert_eq!(after.validate(), Vec::new());
    }

    #[test]
    fn a_selection_box_that_catches_nothing_says_where_things_are() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, _, ctx) = cube_doc();
        let err = h
            .dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.select_where",
                &params(&[
                    ("mode", PropertyValue::Enum(2)),
                    ("min", PropertyValue::Vec3([0.9, 0.9, 0.9])),
                    ("max", PropertyValue::Vec3([0.95, 0.95, 0.95])),
                ]),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("nothing matched"), "{err}");
        assert!(err.contains("nearest face center"), "{err}");
        // By direction alone, the top is found.
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select_where",
            &params(&[
                ("mode", PropertyValue::Enum(2)),
                ("normal", PropertyValue::Vec3([0.0, 0.0, 1.0])),
            ]),
        )
        .unwrap();
    }

    #[test]
    fn a_mesh_check_selects_what_it_found_and_nothing_else() {
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::plane(2.0));
        let obj = doc.add_object(Object::new("Plane", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.check_select",
            &params(&[("issue", PropertyValue::Enum(0))]),
        )
        .unwrap();
        let m = doc.mesh(mesh).unwrap();
        let picked = selection::selected(m, kreate_mesh::attribute::Domain::Edge);
        assert_eq!(picked.len(), 4, "a plane's rim is all open");
        // Nothing flipped on a plane: nothing to select, and no step left.
        let before = h.entries().len();
        let result = h
            .dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.check_select",
                &params(&[("issue", PropertyValue::Enum(2))]),
            )
            .unwrap();
        assert_eq!(result, OpResult::Cancelled);
        assert_eq!(h.entries().len(), before);
    }

    #[test]
    fn the_selections_middle_goes_where_it_is_typed() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        // Moved, so the typed place is the world's and not the mesh's.
        let obj = doc.selection.objects[0];
        doc.object_mut(obj).unwrap().transform.translation = glam::DVec3::new(10.0, 0.0, 0.0);
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let top = bm
                .face_ids()
                .max_by(|a, b| {
                    let za = bm.face_normal(*a).normalize_or_zero().z;
                    let zb = bm.face_normal(*b).normalize_or_zero().z;
                    za.partial_cmp(&zb).unwrap()
                })
                .unwrap();
            bm.set_face_selected(top, true);
            bm.flush_selection(SelectMode::Face);
            *m = bm.to_mesh_data();
        }
        let (median, count) = selection_median(&doc).unwrap();
        assert_eq!(count, 4);
        assert!((median - glam::DVec3::new(10.0, 0.0, 1.0)).length() < 1e-9);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.median_set",
            &params(&[("location", PropertyValue::Vec3([10.0, 0.0, 5.0]))]),
        )
        .unwrap();
        let (median, _) = selection_median(&doc).unwrap();
        assert!((median - glam::DVec3::new(10.0, 0.0, 5.0)).length() < 1e-6);
        assert!((doc.mesh(mesh).unwrap().bounds().max.z - 5.0).abs() < 1e-5);
        assert!(
            (doc.mesh(mesh).unwrap().bounds().min.z + 1.0).abs() < 1e-5,
            "the rest stays"
        );
    }

    #[test]
    fn extruding_with_nothing_selected_writes_no_history() {
        // A tool that records an empty step makes the next undo do nothing
        // visible, which users read as undo being broken.
        let r = registry();
        let mut h = History::new();
        let (mut doc, _, ctx) = cube_doc();

        let result = h
            .dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.extrude_region",
                &params(&[("offset", PropertyValue::Float(1.0))]),
            )
            .unwrap();
        assert_eq!(result, OpResult::Cancelled);
        assert_eq!(h.entries().len(), 0, "nothing to undo");
    }

    #[test]
    fn edit_mode_operators_do_not_poll_in_object_mode() {
        // The gate that stops a mesh keystroke firing while the user is moving
        // objects around.
        let (doc, _, _) = cube_doc();
        let object_mode = Context { mode: Mode::Object };
        assert!(!ExtrudeRegion::poll(&object_mode, &doc));
        assert!(!SelectElements::poll(&object_mode, &doc));
        assert!(ExtrudeRegion::poll(
            &Context {
                mode: Mode::EditMesh
            },
            &doc
        ));
    }

    #[test]
    fn the_selection_operator_round_trips_through_the_document() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[("action", PropertyValue::Enum(0))]),
        )
        .unwrap();
        let all = doc.mesh(mesh).unwrap().to_bmesh();
        assert_eq!(all.selected_verts().len(), 8);

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[("action", PropertyValue::Enum(1))]),
        )
        .unwrap();
        let none = doc.mesh(mesh).unwrap().to_bmesh();
        assert!(none.selected_verts().is_empty());
    }

    #[test]
    fn an_extrude_is_reproducible_from_its_recorded_parameters() {
        // The operator spine: redo re-runs `execute` from `(id, params)` alone.
        // If anything were read from outside the parameter block, the two runs
        // would diverge here.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[
                ("action", PropertyValue::Enum(0)),
                ("mode", PropertyValue::Enum(2)),
            ]),
        )
        .unwrap();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.extrude_region",
            &params(&[("offset", PropertyValue::Float(2.0))]),
        )
        .unwrap();
        let once = doc.mesh(mesh).unwrap().clone();

        assert!(h.undo(&mut doc));
        assert!(h.redo(&r, &mut doc, &ctx).unwrap());
        let twice = doc.mesh(mesh).unwrap();

        assert_eq!(twice.positions(), once.positions());
        assert_eq!(twice.corner_verts(), once.corner_verts());
        assert_eq!(twice.edges(), once.edges());
    }

    /// Selects the face whose normal points most nearly along +Z.
    fn select_top_face(doc: &mut Document, mesh: MeshId) {
        let m = doc.mesh_mut(mesh).unwrap();
        let mut bm = m.to_bmesh();
        let top = bm
            .face_ids()
            .max_by(|a, b| {
                let za = bm.face_normal(*a).normalize_or_zero().z;
                let zb = bm.face_normal(*b).normalize_or_zero().z;
                za.partial_cmp(&zb).unwrap()
            })
            .unwrap();
        bm.set_face_selected(top, true);
        bm.flush_selection(SelectMode::Face);
        *m = bm.to_mesh_data();
    }

    #[test]
    fn deleting_a_face_opens_the_solid_and_undoes_cleanly() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        select_top_face(&mut doc, mesh);
        let before = doc.mesh(mesh).unwrap().clone();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.delete",
            &params(&[("mode", PropertyValue::Enum(2))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), 5);
        assert_eq!(after.num_verts(), 8, "the border stays");

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), before.num_faces());
    }

    #[test]
    fn dissolving_an_edge_keeps_the_solid_closed() {
        // The distinction users constantly confuse, asserted at the operator
        // level: delete opens the cube, dissolve does not.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let e = bm.edge_ids().next().unwrap();
            bm.set_edge_selected(e, true);
            bm.flush_selection(SelectMode::Edge);
            *m = bm.to_mesh_data();
        }

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.dissolve",
            &params(&[("mode", PropertyValue::Enum(1))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), 5);
        assert_eq!(after.num_edges(), 11);
        assert_eq!(after.num_verts(), 8, "nothing was opened up");
        assert_eq!(after.validate(), Vec::new());
    }

    #[test]
    fn subdividing_multiplies_the_face_count_per_iteration() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[
                ("action", PropertyValue::Enum(0)),
                ("mode", PropertyValue::Enum(2)),
            ]),
        )
        .unwrap();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.subdivide",
            &params(&[("iterations", PropertyValue::Int(2))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), 96, "6 -> 24 -> 96");
        assert_eq!(after.validate(), Vec::new());

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 6);
    }

    #[test]
    fn an_absurd_iteration_count_is_clamped_rather_than_obeyed() {
        // The parameter arrives from scripts and saved keymaps too, and a
        // hundred iterations is not a request anyone can have meant.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[("action", PropertyValue::Enum(0))]),
        )
        .unwrap();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.subdivide",
            &params(&[("iterations", PropertyValue::Int(100))]),
        )
        .unwrap();
        assert!(
            doc.mesh(mesh).unwrap().num_faces() <= 6 * 4u32.pow(6),
            "clamped to six passes"
        );
    }

    #[test]
    fn every_edit_operator_stands_down_in_object_mode() {
        let (doc, _, _) = cube_doc();
        let object_mode = Context { mode: Mode::Object };
        assert!(!DeleteElements::poll(&object_mode, &doc));
        assert!(!DissolveElements::poll(&object_mode, &doc));
        assert!(!Subdivide::poll(&object_mode, &doc));
    }

    #[test]
    fn a_vertex_moved_onto_another_merges_with_auto_merge_on() {
        for (auto, want) in [(true, 6), (false, 8)] {
            let r = registry();
            let mut h = History::new();
            let mut doc = Document::new();
            // Two quads half a metre apart; the first's right side selected.
            let split = {
                let mut bm = kreate_mesh::bmesh::BMesh::new();
                let quad = |bm: &mut kreate_mesh::bmesh::BMesh, x: f32, pick: bool| {
                    let a = bm.add_vert(kreate_math::Vec3::new(x, 0.0, 0.0));
                    let b = bm.add_vert(kreate_math::Vec3::new(x + 1.0, 0.0, 0.0));
                    let c = bm.add_vert(kreate_math::Vec3::new(x + 1.0, 1.0, 0.0));
                    let d = bm.add_vert(kreate_math::Vec3::new(x, 1.0, 0.0));
                    bm.add_face(&[a, b, c, d]).unwrap();
                    if pick {
                        bm.set_vert_selected(b, true);
                        bm.set_vert_selected(c, true);
                    }
                };
                quad(&mut bm, 0.0, true);
                quad(&mut bm, 1.5, false);
                bm.to_mesh_data()
            };
            let mesh = doc.add_mesh(split);
            let obj = doc.add_object(Object::new("Split", ObjectData::Mesh(mesh)));
            doc.selection.add(obj);
            let ctx = Context {
                mode: Mode::EditMesh,
            };
            h.dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.translate",
                &params(&[
                    ("translation", PropertyValue::Vec3([0.5, 0.0, 0.0])),
                    ("auto_merge", PropertyValue::Bool(auto)),
                ]),
            )
            .unwrap();
            let after = doc.mesh(mesh).unwrap();
            assert_eq!(after.num_verts(), want, "auto merge {auto}");
            assert_eq!(after.validate(), Vec::new());
            // One step: undone, the gap and all eight are back.
            assert!(h.undo(&mut doc));
            assert_eq!(doc.mesh(mesh).unwrap().num_verts(), 8);
        }
    }

    /// A box that misses a face by a rounding still catches it; one that
    /// misses by more says what tolerance would, and takes it when given.
    #[test]
    fn select_where_allows_a_little_slack() {
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(0.04));
        let obj = doc.add_object(Object::new("Box", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };
        let at = |z: f64| {
            vec![
                ("mode", PropertyValue::Enum(2)),
                // The corners back to front on purpose: a box is its two
                // corners, whichever is given first.
                ("min", PropertyValue::Vec3([1.0, 1.0, z])),
                ("max", PropertyValue::Vec3([-1.0, -1.0, z])),
            ]
        };
        // The top's centre is at 0.02; a micrometre off is caught.
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select_where",
            &params(&at(0.020001)),
        )
        .unwrap();
        // A fifth of a millimetre off is not, and the error says what would.
        let err = h
            .dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.select_where",
                &params(&at(0.0198)),
            )
            .unwrap_err();
        assert!(err.to_string().contains("tolerance="), "{err}");
        let mut given = at(0.0198);
        given.push(("tolerance", PropertyValue::Float(0.0003)));
        h.dispatch(&r, &mut doc, &ctx, "mesh.select_where", &params(&given))
            .unwrap();
    }

    /// The mesh check says how many it found, whichever way it went: none on
    /// a closed box, the four open edges of a box without its top.
    #[test]
    fn check_select_says_how_many_it_found() {
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();
        let box_ = kreate_mesh::primitives::cube(1.0);
        let mesh = doc.add_mesh(box_.clone());
        let obj = doc.add_object(Object::new("Box", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };
        let _ = crate::report::take();
        let ran = h
            .dispatch(&r, &mut doc, &ctx, "mesh.check_select", &params(&[]))
            .unwrap();
        assert_eq!(ran, OpResult::Cancelled);
        assert_eq!(
            crate::report::take().as_deref(),
            Some("found 0 open edges: none")
        );
        let normals = box_.compute_face_normals();
        let keep: Vec<Vec<u32>> = (0..box_.num_faces())
            .filter(|f| normals[*f as usize].z < 0.9)
            .map(|f| box_.face_verts(f).to_vec())
            .collect();
        *doc.mesh_mut(mesh).unwrap() =
            kreate_mesh::MeshData::from_faces(box_.positions().to_vec(), &keep);
        h.dispatch(&r, &mut doc, &ctx, "mesh.check_select", &params(&[]))
            .unwrap();
        assert_eq!(
            crate::report::take().as_deref(),
            Some("found 4 open edges, selected")
        );
    }

    /// A distance weld on a closed boolean result: a small distance keeps it
    /// closed; one longer than its detail is refused, the mesh untouched.
    #[test]
    fn a_distance_merge_never_opens_a_closed_mesh() {
        use kreate_mesh::modifiers::boolean::{boolean, BooleanOp};
        let mut plate = kreate_mesh::primitives::cube(1.0);
        for p in plate.positions_mut() {
            *p *= kreate_math::Vec3::new(0.06, 0.04, 0.003);
        }
        let mut c = kreate_mesh::mech::internal_thread_cutter(0.003, 0.0005, 0.01, 0.0);
        for p in c.positions_mut() {
            *p += kreate_math::Vec3::new(0.015, 0.01, -0.005);
        }
        let cut = boolean(&plate, &c, BooleanOp::Difference);
        assert!(kreate_mesh::measure::is_closed(&cut));
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();
        let mesh = doc.add_mesh(cut);
        let obj = doc.add_object(Object::new("Plate", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };
        let faces = doc.mesh(mesh).unwrap().num_faces();
        let err = h
            .dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.merge",
                &params(&[("distance", PropertyValue::Float(1e-4))]),
            )
            .unwrap_err();
        assert!(err.to_string().contains("would open"), "{err}");
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), faces);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.merge",
            &params(&[("distance", PropertyValue::Float(1e-6))]),
        )
        .unwrap();
        assert!(kreate_mesh::measure::is_closed(doc.mesh(mesh).unwrap()));
    }

    #[test]
    fn welding_by_distance_joins_a_split_mesh_through_the_operator() {
        // The importer path, end to end: a mesh whose faces share no vertices
        // becomes one editable surface.
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();

        let split = {
            let mut bm = kreate_mesh::bmesh::BMesh::new();
            let quad = |bm: &mut kreate_mesh::bmesh::BMesh, x: f32| {
                let a = bm.add_vert(kreate_math::Vec3::new(x, 0.0, 0.0));
                let b = bm.add_vert(kreate_math::Vec3::new(x + 1.0, 0.0, 0.0));
                let c = bm.add_vert(kreate_math::Vec3::new(x + 1.0, 1.0, 0.0));
                let d = bm.add_vert(kreate_math::Vec3::new(x, 1.0, 0.0));
                bm.add_face(&[a, b, c, d]).unwrap();
            };
            quad(&mut bm, 0.0);
            quad(&mut bm, 1.0);
            bm.to_mesh_data()
        };
        assert_eq!(split.num_verts(), 8);

        let mesh = doc.add_mesh(split);
        let obj = doc.add_object(Object::new("Split", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.merge",
            &params(&[("distance", PropertyValue::Float(1e-4))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_verts(), 6, "the seam was welded");
        assert_eq!(after.num_faces(), 2);
        assert_eq!(after.validate(), Vec::new());

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_verts(), 8);
    }

    #[test]
    fn loop_select_grows_from_the_selected_edge() {
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::grid(4.0, 4, 4));
        let obj = doc.add_object(Object::new("Grid", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };

        // Seed one interior edge running along X.
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let seed = bm
                .edge_ids()
                .find(|&e| {
                    let [a, b] = bm.edge_verts(e).unwrap();
                    let (pa, pb) = (bm.position(a).unwrap(), bm.position(b).unwrap());
                    pa.y.abs() < 1e-4 && pb.y.abs() < 1e-4 && (pa.x - pb.x).abs() > 1e-4
                })
                .unwrap();
            bm.set_edge_selected(seed, true);
            *m = bm.to_mesh_data();
        }

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[("action", PropertyValue::Enum(3))]),
        )
        .unwrap();

        let bm = doc.mesh(mesh).unwrap().to_bmesh();
        assert_eq!(bm.selected_edges().len(), 4, "the whole row");
        assert_eq!(bm.selected_verts().len(), 5);
    }

    #[test]
    fn loop_select_with_nothing_selected_is_cancelled() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, _, ctx) = cube_doc();
        let result = h
            .dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.select",
                &params(&[("action", PropertyValue::Enum(3))]),
            )
            .unwrap();
        assert_eq!(result, OpResult::Cancelled);
        assert_eq!(h.entries().len(), 0);
    }

    #[test]
    fn insetting_frames_the_selected_faces() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        select_top_face(&mut doc, mesh);

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.inset",
            &params(&[
                ("thickness", PropertyValue::Float(0.3)),
                ("depth", PropertyValue::Float(-0.4)),
            ]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(
            after.num_faces(),
            5 + 1 + 4,
            "five sides, the panel, its frame"
        );
        assert_eq!(after.num_verts(), 12);
        assert!(
            after.bounds().max.z <= 1.0 + 1e-5,
            "the panel sank, not rose"
        );
        assert_eq!(after.validate(), Vec::new());

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 6);
    }

    #[test]
    fn an_absurd_inset_thickness_produces_a_sane_mesh() {
        // The clamp reaching through the operator. A parameter arriving from a
        // script has no idea what scale the geometry is.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[
                ("action", PropertyValue::Enum(0)),
                ("mode", PropertyValue::Enum(2)),
            ]),
        )
        .unwrap();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.inset",
            &params(&[
                ("thickness", PropertyValue::Float(1000.0)),
                ("individual", PropertyValue::Bool(true)),
            ]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.validate(), Vec::new());
        assert!(after.positions().iter().all(|p| p.is_finite()));
    }

    #[test]
    fn bridging_falls_back_to_the_meshs_own_open_borders() {
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();

        // Two loose rings, which is what an extrude with no cap leaves.
        let rings = {
            let mut bm = kreate_mesh::bmesh::BMesh::new();
            for z in [0.0f32, 2.0] {
                let verts: Vec<_> = (0..6)
                    .map(|i| {
                        let t = i as f32 / 6.0 * std::f32::consts::TAU;
                        bm.add_vert(kreate_math::Vec3::new(t.cos(), t.sin(), z))
                    })
                    .collect();
                for i in 0..6 {
                    bm.add_edge(verts[i], verts[(i + 1) % 6]).unwrap();
                }
            }
            bm
        };
        // `MeshData` cannot hold wire edges, so the rings go in as a document
        // mesh only after they carry faces. Bridge them in edit form instead,
        // which is the path the operator takes anyway.
        let mesh = doc.add_mesh(rings.to_mesh_data());
        let obj = doc.add_object(Object::new("Rings", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };

        // With no faces the rings did not survive the crossing, so this is the
        // documented no-op rather than a bridge. Asserted so the limitation is
        // visible rather than surprising.
        let result = h
            .dispatch(&r, &mut doc, &ctx, "mesh.bridge", &PropertyMap::new())
            .unwrap();
        assert_eq!(result, OpResult::Cancelled);
        assert_eq!(h.entries().len(), 0);
    }

    #[test]
    fn bridging_two_borders_of_a_real_mesh_builds_a_band() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();

        // Open the top and bottom of the cube.
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let mut faces: Vec<_> = bm.face_ids().collect();
            faces.sort_by(|a, b| {
                let za = bm.face_normal(*a).normalize_or_zero().z;
                let zb = bm.face_normal(*b).normalize_or_zero().z;
                za.partial_cmp(&zb).unwrap()
            });
            bm.remove_face(faces[0]);
            bm.remove_face(faces[faces.len() - 1]);
            *m = bm.to_mesh_data();
        }
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 4);

        h.dispatch(&r, &mut doc, &ctx, "mesh.bridge", &PropertyMap::new())
            .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), 8, "four walls plus four bridging quads");
        assert_eq!(after.validate(), Vec::new());
    }

    #[test]
    fn an_edge_slides_across_its_face_and_the_handle_says_which_way() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        let ends = {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let e = bm.edge_ids().next().unwrap();
            bm.set_edge_selected(e, true);
            *m = bm.to_mesh_data();
            m.edges()[0]
        };
        let before = doc.mesh(mesh).unwrap().positions().to_vec();
        let (from, to) = edge_slide_handle(&doc).expect("an edge to slide");
        assert!((to - from).length() > 1.0, "a cube's edge, two across");
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.edge_slide",
            &params(&[("factor", PropertyValue::Float(0.5))]),
        )
        .unwrap();
        let after = doc.mesh(mesh).unwrap().positions().to_vec();
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            let moved = (*a - *b).length();
            if ends.contains(&(i as u32)) {
                assert!(
                    (moved - 1.0).abs() < 1e-4,
                    "halfway along a side of two: {moved}"
                );
            } else {
                assert!(moved < 1e-6, "only the edge's ends move");
            }
        }
        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().positions(), &before[..]);
    }

    #[test]
    fn symmetrize_keeps_one_half_and_undoes() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        for p in doc.mesh_mut(mesh).unwrap().positions_mut() {
            p.x += 0.5;
        }
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.symmetrize",
            &params(&[
                ("axis", PropertyValue::Int(0)),
                ("direction", PropertyValue::Int(0)),
            ]),
        )
        .unwrap();
        let out = doc.mesh(mesh).unwrap();
        let v = kreate_mesh::measure::mass_properties(out, |p| p).volume;
        assert!((v - 12.0).abs() < 1e-3, "{v}");
        assert!(kreate_mesh::measure::is_closed(out));
        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 6);
    }

    #[test]
    fn a_loop_cut_adds_a_belt_and_undoes_cleanly() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();

        // Seed one edge, which is what hovering does in the interactive tool.
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let e = bm.edge_ids().next().unwrap();
            bm.set_edge_selected(e, true);
            *m = bm.to_mesh_data();
        }

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.loop_cut",
            &params(&[
                ("cuts", PropertyValue::Int(1)),
                ("position", PropertyValue::Float(0.5)),
            ]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), 10, "four of the six faces split in two");
        assert_eq!(after.num_verts(), 12);
        assert_eq!(after.validate(), Vec::new());

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 6);
    }

    /// What `Ctrl+R` slides by: the pointer measured along the edge from
    /// its first end as `edge_in_world` gives it. The cut has to move that
    /// way too, on every edge — it went against the pointer on half of them.
    #[test]
    fn a_loop_cut_slides_from_the_end_the_tool_measures_from() {
        let r = registry();
        let (base, mesh, ctx) = cube_doc();
        let count = base.mesh(mesh).unwrap().num_edges();
        for edge in 0..count {
            let mut doc = base.clone();
            let mut h = History::new();
            let (a, b) = edge_in_world(&doc, edge).unwrap();
            h.dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.loop_cut",
                &params(&[
                    ("cuts", PropertyValue::Int(1)),
                    ("edge", PropertyValue::Int(edge as i64)),
                    ("position", PropertyValue::Float(0.25)),
                ]),
            )
            .unwrap();
            let want = a.lerp(b, 0.25);
            let after = doc.mesh(mesh).unwrap();
            let landed = after
                .positions()
                .iter()
                .any(|p| p.as_dvec3().distance(want) < 1e-5);
            assert!(landed, "edge {edge}: no cut a quarter from its first end");
        }
    }

    #[test]
    fn sliding_re_runs_from_the_original_rather_than_compounding() {
        // The adjust-last path, which is what makes a slide exact. Two different
        // positions applied in turn must give the same result as the second one
        // applied alone — not the second applied to the first's output.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let e = bm.edge_ids().next().unwrap();
            bm.set_edge_selected(e, true);
            *m = bm.to_mesh_data();
        }
        let seeded = doc.clone();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.loop_cut",
            &params(&[("position", PropertyValue::Float(0.5))]),
        )
        .unwrap();
        h.adjust_last(
            &r,
            &mut doc,
            &ctx,
            &params(&[("position", PropertyValue::Float(0.25))]),
        )
        .unwrap();
        let slid = doc.mesh(mesh).unwrap().clone();

        let mut fresh = seeded;
        let mut h2 = History::new();
        h2.dispatch(
            &r,
            &mut fresh,
            &ctx,
            "mesh.loop_cut",
            &params(&[("position", PropertyValue::Float(0.25))]),
        )
        .unwrap();

        assert_eq!(slid.num_verts(), fresh.mesh(mesh).unwrap().num_verts());
        assert_eq!(slid.positions(), fresh.mesh(mesh).unwrap().positions());
    }

    #[test]
    fn a_loop_cut_with_no_seed_is_cancelled() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, _, ctx) = cube_doc();
        let result = h
            .dispatch(&r, &mut doc, &ctx, "mesh.loop_cut", &PropertyMap::new())
            .unwrap();
        assert_eq!(result, OpResult::Cancelled);
        assert_eq!(h.entries().len(), 0);
    }

    #[test]
    fn recalculating_normals_fixes_an_imported_mess() {
        // The commonest real use: a mesh arrives with inconsistent winding and
        // the user wants it fixed without selecting anything first.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let f = bm.face_ids().next().unwrap();
            kreate_mesh::edit::flip(&mut bm, &[f]);
            *m = bm.to_mesh_data();
        }

        let result = h
            .dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.normals",
                &params(&[("action", PropertyValue::Enum(0))]),
            )
            .unwrap();
        assert_eq!(result, OpResult::Finished);

        let bm = doc.mesh(mesh).unwrap().to_bmesh();
        for f in bm.face_ids() {
            let ring: Vec<kreate_math::Vec3> = bm
                .face_verts(f)
                .iter()
                .filter_map(|v| bm.position(*v))
                .collect();
            let mid: kreate_math::Vec3 =
                ring.iter().copied().sum::<kreate_math::Vec3>() / ring.len() as f32;
            assert!(bm.face_normal(f).dot(mid) > 0.0, "{f:?} still points in");
        }

        assert!(h.undo(&mut doc));
    }

    #[test]
    fn marking_sharp_by_angle_reaches_the_document() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.mark_sharp",
            &params(&[("angle", PropertyValue::Float(30f64.to_radians()))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        let sharp = after
            .attributes()
            .get_bool(kreate_mesh::attribute::names::SHARP_EDGE)
            .expect("the layer was created");
        assert_eq!(sharp.iter().filter(|x| **x).count(), 12);
        assert_eq!(after.validate(), Vec::new());

        assert!(h.undo(&mut doc));
        assert!(doc
            .mesh(mesh)
            .unwrap()
            .attributes()
            .get_bool(kreate_mesh::attribute::names::SHARP_EDGE)
            .is_none_or(|s| s.iter().all(|x| !*x)));
    }

    #[test]
    fn shading_flat_changes_what_the_renderer_would_draw() {
        // The operator has to reach the normals, not just an attribute nobody
        // reads. Compared through the same function the render extraction uses.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        let before = doc
            .mesh(mesh)
            .unwrap()
            .compute_corner_normals(std::f32::consts::PI);

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.shade",
            &params(&[("smooth", PropertyValue::Bool(false))]),
        )
        .unwrap();

        let after = doc
            .mesh(mesh)
            .unwrap()
            .compute_corner_normals(std::f32::consts::PI);
        assert_eq!(before.len(), after.len());
        let changed = before
            .iter()
            .zip(&after)
            .filter(|(a, b)| (**a - **b).length() > 1e-4)
            .count();
        assert!(changed > 0, "shading flat changed no normals at all");
    }

    #[test]
    fn filling_a_hole_closes_the_solid_through_the_operator() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let f = bm.face_ids().next().unwrap();
            bm.remove_face(f);
            *m = bm.to_mesh_data();
        }
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 5);

        h.dispatch(&r, &mut doc, &ctx, "mesh.fill", &PropertyMap::new())
            .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), 6);
        assert_eq!(after.num_verts(), 8, "an n-gon fill adds nothing");
        assert_eq!(after.validate(), Vec::new());

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 5);
    }

    #[test]
    fn two_holes_fail_with_a_message_the_user_can_act_on() {
        // A refusal is information. "Fill needs exactly one border, found 2"
        // tells the user to select one; a silent no-op does not.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            let mut faces: Vec<_> = bm.face_ids().collect();
            faces.sort_by(|a, b| {
                let za = bm.face_normal(*a).normalize_or_zero().z;
                let zb = bm.face_normal(*b).normalize_or_zero().z;
                za.partial_cmp(&zb).unwrap()
            });
            bm.remove_face(faces[0]);
            bm.remove_face(faces[faces.len() - 1]);
            *m = bm.to_mesh_data();
        }

        let result = h.dispatch(&r, &mut doc, &ctx, "mesh.fill", &PropertyMap::new());
        match result {
            Err(OpError::Failed(message)) => {
                assert!(
                    message.contains("found 2"),
                    "the message should say what was wrong: {message}"
                );
            }
            other => panic!("expected a reason, got {other:?}"),
        }
        assert_eq!(h.entries().len(), 0, "a refusal writes no history");
    }

    #[test]
    fn spinning_a_profile_builds_a_revolve() {
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();
        // A quad standing away from the Z axis.
        let profile = {
            let mut bm = kreate_mesh::bmesh::BMesh::new();
            let a = bm.add_vert(kreate_math::Vec3::new(2.0, 0.0, -0.5));
            let b = bm.add_vert(kreate_math::Vec3::new(3.0, 0.0, -0.5));
            let c = bm.add_vert(kreate_math::Vec3::new(3.0, 0.0, 0.5));
            let d = bm.add_vert(kreate_math::Vec3::new(2.0, 0.0, 0.5));
            bm.add_face(&[a, b, c, d]).unwrap();
            bm.select_all();
            bm.to_mesh_data()
        };
        let mesh = doc.add_mesh(profile);
        let obj = doc.add_object(Object::new("Profile", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.spin",
            &params(&[
                ("angle", PropertyValue::Float(std::f64::consts::TAU)),
                ("steps", PropertyValue::Int(8)),
                ("axis", PropertyValue::Enum(2)),
                ("close", PropertyValue::Bool(true)),
            ]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.validate(), Vec::new());
        // Eight bands of four walls, plus the two caps, less the welded ring.
        assert!(after.num_faces() >= 32, "{} faces", after.num_faces());
        let bounds = after.bounds();
        assert!(bounds.min.x < -2.9 && bounds.max.x > 2.9, "{bounds:?}");
        assert!(
            bounds.min.y < -2.9 && bounds.max.y > 2.9,
            "it went all the way round"
        );

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 1);
    }

    #[test]
    fn splitting_a_face_off_detaches_it_and_undoes_cleanly() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        select_top_face(&mut doc, mesh);
        let before = doc.mesh(mesh).unwrap().num_verts();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.split",
            &params(&[("mode", PropertyValue::Enum(0))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_verts(), before + 4, "four corners were shared");
        assert_eq!(after.num_faces(), 6, "no face was added or lost");
        assert_eq!(after.validate(), Vec::new());

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_verts(), before);
    }

    #[test]
    fn splitting_something_already_separate_is_cancelled() {
        // Nothing shared means nothing to detach. A history entry here would
        // make the user's next undo do nothing visible.
        let r = registry();
        let mut h = History::new();
        let (mut doc, _, ctx) = cube_doc();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[
                ("action", PropertyValue::Enum(0)),
                ("mode", PropertyValue::Enum(2)),
            ]),
        )
        .unwrap();

        let result = h
            .dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.split",
                &params(&[("mode", PropertyValue::Enum(0))]),
            )
            .unwrap();
        assert_eq!(result, OpResult::Cancelled);
    }

    #[test]
    fn ripping_a_row_of_edges_cuts_a_sheet_in_two() {
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::grid(4.0, 4, 4));
        let obj = doc.add_object(Object::new("Sheet", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };

        {
            let m = doc.mesh_mut(mesh).unwrap();
            let mut bm = m.to_bmesh();
            for e in bm.edge_ids().collect::<Vec<_>>() {
                if bm.edge_face_count(e) != 2 {
                    continue;
                }
                let [a, b] = bm.edge_verts(e).unwrap();
                let (pa, pb) = (bm.position(a).unwrap(), bm.position(b).unwrap());
                if pa.y.abs() < 1e-4 && pb.y.abs() < 1e-4 && (pa.x - pb.x).abs() > 1e-4 {
                    bm.set_edge_selected(e, true);
                }
            }
            *m = bm.to_mesh_data();
        }
        let before = doc.mesh(mesh).unwrap().num_verts();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.split",
            &params(&[("mode", PropertyValue::Enum(1))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_verts(), before + 5, "five vertices along the row");
        assert_eq!(after.validate(), Vec::new());

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_verts(), before);
    }

    #[test]
    fn bevelling_every_edge_chamfers_the_cube_and_undoes_cleanly() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[
                ("action", PropertyValue::Enum(0)),
                ("mode", PropertyValue::Enum(1)),
            ]),
        )
        .unwrap();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.bevel",
            &params(&[("width", PropertyValue::Float(0.25))]),
        )
        .unwrap();

        let after = doc.mesh(mesh).unwrap();
        assert_eq!(after.num_faces(), 6 + 12 + 8, "faces, chamfers and corners");
        assert_eq!(after.num_verts(), 24, "three per original corner");
        assert_eq!(after.validate(), Vec::new());

        assert!(h.undo(&mut doc));
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 6);
        assert_eq!(doc.mesh(mesh).unwrap().num_verts(), 8);
    }

    #[test]
    fn adjusting_a_bevel_width_re_runs_from_the_original() {
        // The adjust-last path. Compounding would chamfer the chamfer, and the
        // second value would mean something different from the first.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[
                ("action", PropertyValue::Enum(0)),
                ("mode", PropertyValue::Enum(1)),
            ]),
        )
        .unwrap();
        let seeded = doc.clone();

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.bevel",
            &params(&[("width", PropertyValue::Float(0.25))]),
        )
        .unwrap();
        h.adjust_last(
            &r,
            &mut doc,
            &ctx,
            &params(&[("width", PropertyValue::Float(0.1))]),
        )
        .unwrap();
        let adjusted = doc.mesh(mesh).unwrap().clone();

        let mut fresh = seeded;
        let mut h2 = History::new();
        h2.dispatch(
            &r,
            &mut fresh,
            &ctx,
            "mesh.bevel",
            &params(&[("width", PropertyValue::Float(0.1))]),
        )
        .unwrap();

        assert_eq!(adjusted.num_faces(), fresh.mesh(mesh).unwrap().num_faces());
        assert_eq!(adjusted.positions(), fresh.mesh(mesh).unwrap().positions());
    }

    #[test]
    fn bevelling_only_boundary_edges_is_cancelled() {
        let r = registry();
        let mut h = History::new();
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::plane(2.0));
        let obj = doc.add_object(Object::new("Plane", ObjectData::Mesh(mesh)));
        doc.selection.add(obj);
        let ctx = Context {
            mode: Mode::EditMesh,
        };
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select",
            &params(&[
                ("action", PropertyValue::Enum(0)),
                ("mode", PropertyValue::Enum(1)),
            ]),
        )
        .unwrap();

        let result = h
            .dispatch(
                &r,
                &mut doc,
                &ctx,
                "mesh.bevel",
                &params(&[("width", PropertyValue::Float(0.2))]),
            )
            .unwrap();
        assert_eq!(result, OpResult::Cancelled);
        assert_eq!(doc.mesh(mesh).unwrap().num_faces(), 1);
    }
    // --- click selection ----------------------------------------------------

    /// Runs `mesh.select_pick` on the given element.
    fn click(
        r: &OperatorRegistry,
        h: &mut History,
        d: &mut Document,
        c: &Context,
        element: u32,
        mode: u32,
        extend: bool,
    ) -> OpResult {
        let p = params(&[
            ("element", PropertyValue::Int(element as i64)),
            ("mode", PropertyValue::Enum(mode)),
            ("extend", PropertyValue::Bool(extend)),
        ]);
        h.dispatch(r, d, c, "mesh.select_pick", &p)
            .expect("mesh.select_pick")
    }

    /// A registry, a history and a document holding one cube in edit mode.
    fn edit_mode_cube() -> (OperatorRegistry, History, Document, Context) {
        let (doc, _, ctx) = cube_doc();
        (registry(), History::new(), doc, ctx)
    }

    /// Which layer of a domain is selected, by index.
    fn selected(d: &Document, layer: &str) -> Vec<usize> {
        let id = d.selection.active.unwrap();
        let mesh_id = d.object(id).unwrap().mesh().unwrap();
        let mesh = d.mesh(mesh_id).unwrap();
        match mesh.attributes().get_bool(layer) {
            Some(sel) => sel
                .iter()
                .enumerate()
                .filter(|(_, s)| **s)
                .map(|(i, _)| i)
                .collect(),
            None => Vec::new(),
        }
    }

    fn selected_faces(d: &Document) -> Vec<usize> {
        selected(d, kreate_mesh::attribute::names::SELECT_FACE)
    }

    #[test]
    fn a_click_selects_exactly_one_element() {
        let (r, mut h, mut d, c) = edit_mode_cube();
        assert_eq!(
            click(&r, &mut h, &mut d, &c, 3, 2, false),
            OpResult::Finished
        );
        assert_eq!(selected_faces(&d), vec![3]);
    }

    #[test]
    fn a_plain_click_replaces_the_selection() {
        // The default everywhere: clicking somewhere else means "that one
        // instead", not "that one as well".
        let (r, mut h, mut d, c) = edit_mode_cube();
        click(&r, &mut h, &mut d, &c, 1, 2, false);
        click(&r, &mut h, &mut d, &c, 4, 2, false);
        assert_eq!(selected_faces(&d), vec![4]);
    }

    #[test]
    fn an_extending_click_adds_and_a_second_one_removes() {
        // Shift+Click toggles, so a mis-click is undone by repeating it rather
        // than by starting the selection over.
        let (r, mut h, mut d, c) = edit_mode_cube();
        click(&r, &mut h, &mut d, &c, 1, 2, false);
        click(&r, &mut h, &mut d, &c, 4, 2, true);
        assert_eq!(selected_faces(&d), vec![1, 4]);

        click(&r, &mut h, &mut d, &c, 1, 2, true);
        assert_eq!(
            selected_faces(&d),
            vec![4],
            "the second click took it back out"
        );
    }

    #[test]
    fn selecting_a_face_marks_its_edges_and_vertices() {
        // Flushed across domains, or the next operator acts on a selection the
        // user cannot see — an extrude that does nothing because the vertices
        // it works from were never marked.
        let (r, mut h, mut d, c) = edit_mode_cube();
        click(&r, &mut h, &mut d, &c, 0, 2, false);

        use kreate_mesh::attribute::names;
        assert_eq!(selected(&d, names::SELECT_VERT).len(), 4, "a quad has four");
        assert_eq!(selected(&d, names::SELECT_EDGE).len(), 4);
    }

    #[test]
    fn an_element_the_mesh_does_not_have_is_refused() {
        // The id buffer is a frame or two old, so an index can name an element an
        // edit has since removed. Normal, not an error, and it must not panic.
        let (r, mut h, mut d, c) = edit_mode_cube();
        assert_eq!(
            click(&r, &mut h, &mut d, &c, 9_999, 2, false),
            OpResult::Cancelled
        );
        assert!(selected_faces(&d).is_empty());
    }

    #[test]
    fn a_click_is_one_undo_step() {
        // The reason this is an operator rather than a write from the viewport:
        // nothing here thought about undo, and a selection the user regrets comes
        // back in one step.
        let (r, mut h, mut d, c) = edit_mode_cube();
        click(&r, &mut h, &mut d, &c, 2, 2, false);
        assert_eq!(selected_faces(&d), vec![2]);

        assert!(h.undo(&mut d), "one step back");
        assert!(selected_faces(&d).is_empty());
    }

    #[test]
    fn clicking_empty_space_clears_the_selection() {
        let (r, mut h, mut d, c) = edit_mode_cube();
        click(&r, &mut h, &mut d, &c, 2, 2, false);
        let p = PropertyMap::new();
        assert_eq!(
            h.dispatch(&r, &mut d, &c, "mesh.select_nothing", &p)
                .expect("mesh.select_nothing"),
            OpResult::Finished
        );
        assert!(selected_faces(&d).is_empty());
    }

    #[test]
    fn clearing_an_empty_selection_leaves_no_history_step() {
        // Clicking empty space twice should not fill the history with steps that
        // undo to the same state.
        let (r, mut h, mut d, c) = edit_mode_cube();
        let p = PropertyMap::new();
        assert_eq!(
            h.dispatch(&r, &mut d, &c, "mesh.select_nothing", &p)
                .expect("mesh.select_nothing"),
            OpResult::Cancelled
        );
        assert!(!h.can_undo());
    }

    // --- box, lasso and circle selection ------------------------------------

    /// Runs `mesh.select_set` with a resolved element list.
    fn select_set(
        r: &OperatorRegistry,
        h: &mut History,
        d: &mut Document,
        c: &Context,
        elements: &[u32],
        extend: bool,
        deselect: bool,
    ) -> OpResult {
        let p = params(&[
            ("elements", PropertyValue::Ints(elements.to_vec())),
            ("mode", PropertyValue::Enum(2)),
            ("extend", PropertyValue::Bool(extend)),
            ("deselect", PropertyValue::Bool(deselect)),
        ]);
        h.dispatch(r, d, c, "mesh.select_set", &p)
            .expect("mesh.select_set")
    }

    #[test]
    fn a_resolved_set_selects_all_of_it_at_once() {
        let (r, mut h, mut d, c) = edit_mode_cube();
        assert_eq!(
            select_set(&r, &mut h, &mut d, &c, &[0, 2, 4], false, false),
            OpResult::Finished
        );
        assert_eq!(selected_faces(&d), vec![0, 2, 4]);
    }

    #[test]
    fn a_set_replaces_the_selection_unless_extending() {
        let (r, mut h, mut d, c) = edit_mode_cube();
        select_set(&r, &mut h, &mut d, &c, &[0, 1], false, false);
        select_set(&r, &mut h, &mut d, &c, &[4], false, false);
        assert_eq!(selected_faces(&d), vec![4]);

        select_set(&r, &mut h, &mut d, &c, &[0], true, false);
        assert_eq!(selected_faces(&d), vec![0, 4]);
    }

    #[test]
    fn a_deselecting_set_removes_only_what_it_names() {
        // Ctrl+drag: a box that takes elements out. It has to leave the rest
        // alone, or it is just a differently coloured replace.
        let (r, mut h, mut d, c) = edit_mode_cube();
        select_set(&r, &mut h, &mut d, &c, &[0, 1, 2, 3], false, false);
        select_set(&r, &mut h, &mut d, &c, &[1, 3], true, true);
        assert_eq!(selected_faces(&d), vec![0, 2]);
    }

    #[test]
    fn an_empty_box_that_replaces_clears_the_selection() {
        // Dragging over nothing means "select nothing", which is the same gesture
        // a user reaches for to clear.
        let (r, mut h, mut d, c) = edit_mode_cube();
        select_set(&r, &mut h, &mut d, &c, &[0, 1], false, false);
        assert_eq!(
            select_set(&r, &mut h, &mut d, &c, &[], false, false),
            OpResult::Finished
        );
        assert!(selected_faces(&d).is_empty());
    }

    #[test]
    fn an_empty_box_that_extends_does_nothing() {
        // Shift+drag over nothing should not touch the selection, and should not
        // leave a history step that undoes to the same state.
        let (r, mut h, mut d, c) = edit_mode_cube();
        select_set(&r, &mut h, &mut d, &c, &[0, 1], false, false);
        let steps = h.entries().len();
        assert_eq!(
            select_set(&r, &mut h, &mut d, &c, &[], true, false),
            OpResult::Cancelled
        );
        assert_eq!(selected_faces(&d), vec![0, 1]);
        assert_eq!(h.entries().len(), steps, "a refused operator left a step");
    }

    /// Dispatches an operator by id with the given parameters.
    fn run(
        r: &OperatorRegistry,
        h: &mut History,
        d: &mut Document,
        c: &Context,
        id: &str,
        pairs: &[(&'static str, PropertyValue)],
    ) -> OpResult {
        h.dispatch(r, d, c, id, &params(pairs)).expect(id)
    }

    /// The mesh an edit-mode document is holding.
    fn mesh_arc(d: &Document) -> std::sync::Arc<kreate_mesh::MeshData> {
        let id = d
            .object(d.selection.active.unwrap())
            .unwrap()
            .mesh()
            .unwrap();
        d.mesh_arc(id).unwrap().clone()
    }

    #[test]
    fn a_click_changes_nothing_but_the_selection() {
        // The claim the whole selection path was rewritten for. The renderer
        // decides what to re-upload by comparing attribute arrays for pointer
        // identity, so a click that rebuilds the mesh — which is what a round
        // trip through `BMesh` does — re-uploads a million vertices to change
        // three bits.
        //
        // Tested by array identity rather than by contents: a `BMesh` round trip
        // is the identity on contents, so a value comparison passes while every
        // cache in the system sees a brand-new mesh.
        let (r, mut h, mut d, c) = edit_mode_cube();
        let before = mesh_arc(&d);
        click(&r, &mut h, &mut d, &c, 2, 2, false);
        let after = mesh_arc(&d);

        assert!(
            before.only_selection_differs(&after),
            "a click changed something other than the selection"
        );
        assert_eq!(selected_faces(&d), vec![2], "and it did select something");
    }

    #[test]
    fn a_box_select_changes_nothing_but_the_selection() {
        // The same claim for the drag, which is the one that happens every frame
        // rather than once.
        let (r, mut h, mut d, c) = edit_mode_cube();
        let before = mesh_arc(&d);
        select_set(&r, &mut h, &mut d, &c, &[0, 1, 2], false, false);
        assert!(before.only_selection_differs(&mesh_arc(&d)));
    }

    #[test]
    fn select_all_and_invert_change_nothing_but_the_selection() {
        let (r, mut h, mut d, c) = edit_mode_cube();
        let before = mesh_arc(&d);
        run(&r, &mut h, &mut d, &c, "mesh.select", &[]);
        let all = mesh_arc(&d);
        assert!(before.only_selection_differs(&all), "select all");

        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "mesh.select",
            &[("action", PropertyValue::Enum(2))],
        );
        assert!(all.only_selection_differs(&mesh_arc(&d)), "invert");
    }

    #[test]
    fn a_geometry_edit_still_replaces_the_mesh() {
        // The companion: the shortcut must not swallow a real edit. An operator
        // that changes topology has to hand the renderer something it will
        // actually rebuild.
        let (r, mut h, mut d, c) = edit_mode_cube();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "mesh.select",
            &[("action", PropertyValue::Enum(0))],
        );
        let before = mesh_arc(&d);
        run(&r, &mut h, &mut d, &c, "mesh.subdivide", &[]);
        assert!(
            !before.only_selection_differs(&mesh_arc(&d)),
            "a subdivision was mistaken for a selection change"
        );
    }

    #[test]
    fn a_cancelled_operator_leaves_the_mesh_arrays_untouched() {
        // The claim worth pinning down, because getting it wrong is invisible:
        // an edit-mode operator round-trips the mesh through `BMesh` before it
        // can tell there is nothing to do, and that rebuilds every array. Left
        // in place, the renderer is handed a mesh it has never seen — a full
        // re-upload of geometry that did not change — with no history entry to
        // undo it.
        //
        // Tested by array identity rather than by contents: a `BMesh` round trip
        // is the identity on contents, so comparing values would pass while the
        // caches that matter all saw a brand-new mesh.
        let (r, mut h, mut d, c) = edit_mode_cube();
        select_set(&r, &mut h, &mut d, &c, &[0, 1], false, false);

        let mesh_id = d
            .object(d.selection.active.unwrap())
            .unwrap()
            .mesh()
            .unwrap();
        let before = d.mesh_arc(mesh_id).unwrap().clone();

        // A click on an element the mesh does not have: the operator has to
        // convert the mesh before it can tell, so this is a cancel that has
        // already rebuilt everything by the time it decides.
        assert_eq!(
            click(&r, &mut h, &mut d, &c, 9_999, 2, false),
            OpResult::Cancelled
        );

        let after = d.mesh_arc(mesh_id).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(&before, after),
            "a cancelled operator left a rebuilt mesh behind"
        );
    }

    #[test]
    fn indices_the_mesh_does_not_have_are_skipped() {
        // The id buffer is a frame or two old, so a drag that ends just after an
        // edit can name elements that have gone. The rest of the box still has
        // to land.
        let (r, mut h, mut d, c) = edit_mode_cube();
        select_set(&r, &mut h, &mut d, &c, &[1, 9_999, 3], false, false);
        assert_eq!(selected_faces(&d), vec![1, 3]);
    }

    #[test]
    fn a_box_select_is_one_undo_step() {
        let (r, mut h, mut d, c) = edit_mode_cube();
        select_set(&r, &mut h, &mut d, &c, &[0, 1, 2, 3, 4, 5], false, false);
        assert_eq!(selected_faces(&d).len(), 6);
        assert!(h.undo(&mut d), "one step back");
        assert!(selected_faces(&d).is_empty());
    }

    #[test]
    fn a_box_select_flushes_across_domains() {
        // Selecting faces has to mark their edges and vertices, once at the end
        // rather than once per element — a box over a thousand faces must not
        // flush a thousand times.
        let (r, mut h, mut d, c) = edit_mode_cube();
        select_set(&r, &mut h, &mut d, &c, &[0, 1, 2, 3, 4, 5], false, false);
        use kreate_mesh::attribute::names;
        assert_eq!(
            selected(&d, names::SELECT_VERT).len(),
            8,
            "every vertex of a fully selected cube"
        );
        assert_eq!(selected(&d, names::SELECT_EDGE).len(), 12);
    }

    #[test]
    fn a_recorded_box_select_replays_as_a_script_line() {
        // The reason the parameter is a resolved set rather than a screen
        // rectangle: a set replays exactly. A rectangle would depend on where the
        // camera was, and re-running it after the view moved would select
        // something else.
        let (r, mut h, mut d, c) = edit_mode_cube();
        select_set(&r, &mut h, &mut d, &c, &[1, 4], false, false);
        let line = h.last().expect("a step was recorded").to_script_call();
        assert!(line.contains("mesh.select_set"), "{line}");
        assert!(line.contains("[1, 4]"), "the set has to be in it: {line}");
    }

    #[test]
    fn a_click_selects_in_the_domain_it_was_asked_for() {
        let (r, mut h, mut d, c) = edit_mode_cube();
        click(&r, &mut h, &mut d, &c, 5, 0, false);

        assert_eq!(
            selected(&d, kreate_mesh::attribute::names::SELECT_VERT),
            vec![5],
            "vertex 5, and only vertex 5"
        );
    }

    #[test]
    fn separating_a_selection_makes_a_new_object() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = cube_doc();

        // Two faces, by index.
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select_set",
            &params(&[
                ("elements", PropertyValue::Ints(vec![0, 1])),
                ("mode", PropertyValue::Enum(2)),
            ]),
        )
        .unwrap();

        h.dispatch(&r, &mut doc, &ctx, "mesh.separate", &params(&[]))
            .expect("separate ran");

        assert_eq!(doc.objects().count(), 2, "no new object appeared");
        assert_eq!(
            doc.mesh(mesh).unwrap().num_faces(),
            4,
            "the source kept them"
        );
        let new_id = doc.selection.active.expect("the new object is active");
        let new_mesh = doc.object(new_id).and_then(|o| o.mesh()).unwrap();
        assert_eq!(doc.mesh(new_mesh).unwrap().num_faces(), 2);
    }

    #[test]
    fn the_separated_object_takes_the_sources_placement() {
        // A separate is a change of ownership, not of shape: the vertices were
        // copied in the source's local space, so reading them in any other
        // space would move the geometry.
        let r = registry();
        let mut h = History::new();
        let (mut doc, _, ctx) = cube_doc();
        let source = doc.selection.active.expect("the cube is active");
        if let Some(o) = doc.object_mut(source) {
            o.transform.translation = kreate_math::DVec3::new(3.0, -2.0, 1.0);
        }
        let placement = doc.object(source).unwrap().transform;

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "mesh.select_set",
            &params(&[
                ("elements", PropertyValue::Ints(vec![0])),
                ("mode", PropertyValue::Enum(2)),
            ]),
        )
        .unwrap();
        h.dispatch(&r, &mut doc, &ctx, "mesh.separate", &params(&[]))
            .expect("separate ran");

        let new_id = doc.selection.active.unwrap();
        assert_eq!(
            doc.object(new_id).unwrap().transform,
            placement,
            "the new object did not take the source's placement"
        );
    }

    #[test]
    fn separating_nothing_is_cancelled_and_leaves_no_object() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, _, ctx) = cube_doc();
        let before = doc.objects().count();
        let depth = h.entries().len();
        h.dispatch(&r, &mut doc, &ctx, "mesh.separate", &params(&[]))
            .expect("separate ran");
        assert_eq!(doc.objects().count(), before, "an object appeared anyway");
        assert_eq!(
            h.entries().len(),
            depth,
            "an empty step reached the history"
        );
    }
}
