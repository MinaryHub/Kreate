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

use crate::operator::{Context, Mode, OpError, OpResult, Operator};
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

/// Runs `f` over the mesh in edit form and writes the result back.
///
/// Takes the coarse undo path: every one of these changes topology, and a
/// declared write set over positions cannot describe a mesh that has grown
/// faces.
pub(crate) fn with_bmesh<T>(
    doc: &mut DocMut<'_>,
    f: impl FnOnce(&mut BMesh) -> Result<T, OpError>,
) -> Result<T, OpError> {
    let mesh_id =
        target_mesh(doc.document()).ok_or_else(|| OpError::Failed("no mesh to edit".into()))?;

    // The edit-mode structure, when edit mode is open and still agrees with the
    // document. Held across operators, so the decode happens once on entering
    // edit mode rather than once per extrude.
    if doc.edit_bmesh(mesh_id).is_some() {
        let out = {
            let bm = doc
                .edit_bmesh(mesh_id)
                .expect("checked on the line above, and nothing has borrowed since");
            f(bm)?
        };
        if !doc.bake_edit() {
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

/// True when there is a mesh to edit at all.
pub(crate) fn can_edit(ctx: &Context, doc: &Document) -> bool {
    ctx.mode == Mode::EditMesh && target_mesh(doc).is_some()
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

/// The mean position of the affected vertices: what a rotation or a scale
/// happens about by default.
///
/// The mean rather than the bounding-box centre. For a symmetric selection they
/// agree; for a lopsided one the mean follows where the geometry actually is,
/// and the bounding-box centre can sit in empty space away from every vertex —
/// which makes a rotation swing rather than turn.
fn median(bm: &BMesh, verts: &[VertId]) -> Option<glam::Vec3> {
    if verts.is_empty() {
        return None;
    }
    let mut total = glam::Vec3::ZERO;
    let mut count = 0.0;
    for v in verts {
        if let Some(p) = bm.position(*v) {
            total += p;
            count += 1.0;
        }
    }
    (count > 0.0).then(|| total / count)
}

/// How many distinct positions a selection occupies.
///
/// Used by rotate and scale to refuse a selection that has nothing to turn or
/// stretch: every affected vertex sitting on the pivot means both operations are
/// exactly the identity, and reporting success for a no-op puts an entry in the
/// history that undoes to the same thing.
fn distinct_positions(bm: &BMesh, verts: &[VertId]) -> usize {
    let mut keys: Vec<[i64; 3]> = verts
        .iter()
        .filter_map(|v| bm.position(*v))
        .map(|p| {
            // Quantised at a micrometre, so two vertices a float's breadth apart
            // count as one place. Without this a "single vertex" that happens to
            // be two welded ones would pass for something scalable.
            let q = |x: f32| (x as f64 * 1e6).round() as i64;
            [q(p.x), q(p.y), q(p.z)]
        })
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys.len()
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

/// How influence falls off with distance, for proportional editing.
///
/// Smooth by default. A modeller reaches for proportional editing to shape a
/// surface, and a linear falloff leaves a visible cone with a crease at its rim
/// — the very artefact the tool exists to avoid.
fn falloff(distance: f64, radius: f64) -> f32 {
    if radius <= 0.0 {
        return 0.0;
    }
    let t = (distance / radius).clamp(0.0, 1.0);
    // Smoothstep, flipped: 1 at the centre, 0 at the rim, and flat at both ends
    // so the moved region joins the rest of the mesh without a crease.
    let s = 1.0 - t;
    (s * s * (3.0 - 2.0 * s)) as f32
}

/// A weight per vertex: 1 for the selection, falling off for everything else.
///
/// The distance is to the *nearest selected vertex*, not to the selection's
/// median. That matters as soon as the selection is more than a point: dragging
/// a whole edge loop should pull the mesh evenly along its length, and a
/// distance to the median would pull hardest in the middle and leave the ends
/// behind.
///
/// O(V x S) as written, which is honest about what it is. A million-vertex mesh
/// with a large selection wants a spatial index over the selected vertices, and
/// that is the shape of the eventual fix; the operator's signature does not
/// change when it lands.
fn proportional_weights(bm: &BMesh, selected: &[VertId], radius: f64) -> Vec<(VertId, f32)> {
    let anchors: Vec<glam::Vec3> = selected.iter().filter_map(|v| bm.position(*v)).collect();
    // A fast path, not a rule: `falloff` already returns zero for a
    // non-positive radius, so the loop below would reach the same answer. What
    // it would also do is measure every vertex in the mesh against every
    // selected one to conclude that nothing moves — and proportional editing
    // being *off* is the common case, so that cost is paid on almost every
    // transform.
    //
    // Being redundant for correctness is why removing it changes no test. It is
    // kept for what it costs, not for what it decides.
    if anchors.is_empty() || radius <= 0.0 {
        return selected.iter().map(|v| (*v, 1.0)).collect();
    }
    let mut out: Vec<(VertId, f32)> = Vec::new();
    for v in bm.vert_ids() {
        if selected
            .binary_search_by_key(&v.index(), |s| s.index())
            .is_ok()
        {
            // Full weight, and not measured: a selected vertex is at distance
            // zero from itself, but saying so explicitly means a rounding error
            // can never give the selection less than 1.
            out.push((v, 1.0));
            continue;
        }
        let Some(p) = bm.position(v) else { continue };
        let nearest = anchors
            .iter()
            .map(|a| a.distance(p) as f64)
            .fold(f64::INFINITY, f64::min);
        let weight = falloff(nearest, radius);
        if weight > 0.0 {
            out.push((v, weight));
        }
    }
    out
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MeshTranslateParams {
    /// How far to move the selection, in object space.
    #[param(unit = "length")]
    pub translation: [f64; 3],
    /// Proportional editing radius. Zero is off.
    #[param(soft_min = 0.0, soft_max = 10.0, unit = "length")]
    pub proportional_size: f64,
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
        let offset = glam::Vec3::new(
            params.translation[0] as f32,
            params.translation[1] as f32,
            params.translation[2] as f32,
        );
        let radius = params.proportional_size;
        let moved = with_bmesh(doc, |bm| {
            let verts = selected_vertices(bm);
            if verts.is_empty() {
                return Ok(false);
            }
            for (v, weight) in proportional_weights(bm, &verts, radius) {
                if let Some(p) = bm.position(v) {
                    bm.set_position(v, p + offset * weight);
                }
            }
            Ok(true)
        })?;
        Ok(if moved {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
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
            0 => glam::Vec3::X,
            1 => glam::Vec3::Y,
            _ => glam::Vec3::Z,
        };
        let rotation = glam::Quat::from_axis_angle(axis, params.angle as f32);
        let radius = params.proportional_size;
        let turned = with_bmesh(doc, |bm| {
            let verts = selected_vertices(bm);
            // A single point has nothing to turn: every vertex is the pivot, so
            // the rotation is the identity.
            if distinct_positions(bm, &verts) < 2 {
                return Ok(false);
            }
            let Some(about) = median(bm, &verts) else {
                return Ok(false);
            };
            for (v, weight) in proportional_weights(bm, &verts, radius) {
                if let Some(p) = bm.position(v) {
                    // The rotation is interpolated towards the identity rather
                    // than the *result* being interpolated towards the original.
                    // The two differ: blending positions drags a partly-weighted
                    // vertex along the chord and shrinks the mesh towards the
                    // pivot, while blending the rotation turns it through a
                    // smaller angle, which is what the tool means.
                    let partial = glam::Quat::IDENTITY.slerp(rotation, weight);
                    bm.set_position(v, about + partial * (p - about));
                }
            }
            Ok(true)
        })?;
        Ok(if turned {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct MeshScaleParams {
    /// Scale factor per axis, about the selection's median.
    pub scale: [f64; 3],
    /// Proportional editing radius. Zero is off.
    #[param(soft_min = 0.0, soft_max = 10.0, unit = "length")]
    pub proportional_size: f64,
}

impl Default for MeshScaleParams {
    fn default() -> Self {
        Self {
            scale: [1.0; 3],
            proportional_size: 0.0,
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
        let factor = glam::Vec3::new(
            params.scale[0] as f32,
            params.scale[1] as f32,
            params.scale[2] as f32,
        );
        let radius = params.proportional_size;
        let scaled = with_bmesh(doc, |bm| {
            let verts = selected_vertices(bm);
            if distinct_positions(bm, &verts) < 2 {
                return Ok(false);
            }
            let Some(about) = median(bm, &verts) else {
                return Ok(false);
            };
            for (v, weight) in proportional_weights(bm, &verts, radius) {
                if let Some(p) = bm.position(v) {
                    // The factor is interpolated towards one.
                    //
                    // Unlike the rotation, this is *not* a correction: scaling is
                    // linear in the position, so blending the factor and blending
                    // the result are algebraically the same thing, and a test
                    // cannot tell them apart. Written this way for symmetry with
                    // the rotation above and because it says what is meant — a
                    // vertex at half weight is scaled half as much, rather than
                    // dragged half way to where a full scale would put it.
                    let partial = glam::Vec3::ONE.lerp(factor, weight);
                    bm.set_position(v, about + (p - about) * partial);
                }
            }
            Ok(true)
        })?;
        Ok(if scaled {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

// --- extrude ----------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ExtrudeParams {
    /// How far to move the new faces along the region normal.
    #[param(soft_min = -10.0, soft_max = 10.0, unit = "length")]
    pub offset: f64,
}

impl Default for ExtrudeParams {
    fn default() -> Self {
        Self { offset: 1.0 }
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
        let grew = with_bmesh(doc, |bm| {
            let faces = bm.selected_faces();
            if faces.is_empty() {
                return Ok(false);
            }
            edit::extrude_selected_faces(bm, params.offset as f32)
                .map_err(|e| OpError::Failed(e.to_string()))?;
            Ok(true)
        })?;

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
        let removed = with_bmesh(doc, |bm| Ok(edit::delete(bm, mode)))?;
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
        let out = with_bmesh(doc, |bm| {
            let result = match params.mode {
                0 => edit::dissolve_verts(bm, &bm.selected_verts()),
                2 => edit::dissolve_faces(bm, &bm.selected_faces(), &[]),
                _ => edit::dissolve_edges(bm, &bm.selected_edges()),
            };
            result.map_err(|e| OpError::Failed(e.to_string()))
        })?;

        // A dissolve that healed nothing but refused some groups is not a
        // failure the user should have to undo — it is a report. Telling them
        // apart matters: `Cancelled` leaves the history clean.
        Ok(if out.faces.is_empty() {
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
        let grew = with_bmesh(doc, |bm| {
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
        })?;
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
}

impl Default for BevelParams {
    fn default() -> Self {
        Self { width: 0.1 }
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
        let built = with_bmesh(doc, |bm| {
            edit::bevel_selected(bm, params.width as f32)
                .map(|out| out.faces.len())
                .map_err(|e| OpError::Failed(e.to_string()))
        })?;
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
        let opened = with_bmesh(doc, |bm| {
            let result = if params.mode == 1 {
                edit::rip_selected_edges(bm)
            } else {
                edit::split_selected_faces(bm)
            };
            result
                .map(|out| out.seams)
                .map_err(|e| OpError::Failed(e.to_string()))
        })?;
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
        let built = with_bmesh(doc, |bm| {
            let span = (params.span > 0).then_some(params.span as usize);
            match edit::fill_selected(bm, params.grid, span) {
                Ok(out) => Ok(out.faces.len()),
                // A refusal is information, not a crash: "two borders" and "an
                // odd border" both tell the user exactly what to change.
                Err(e) => Err(OpError::Failed(e.to_string())),
            }
        })?;
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
        let grew = with_bmesh(doc, |bm| {
            let out = edit::spin_selected(bm, centre, axis, params.angle as f32, params.steps)
                .map_err(|e| OpError::Failed(e.to_string()))?;
            if params.close {
                edit::close_spin(bm, &out).map_err(|e| OpError::Failed(e.to_string()))?;
            }
            Ok(out.verts.len())
        })?;
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
        let changed = with_bmesh(doc, |bm| {
            // The whole mesh when nothing is selected: recalculating normals is
            // something a user does to a model they have just imported, and
            // making them select everything first is friction with no purpose.
            let mut faces = bm.selected_faces();
            if faces.is_empty() {
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
        })?;
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
        let changed = with_bmesh(doc, |bm| {
            let mut faces = bm.selected_faces();
            if faces.is_empty() {
                faces = bm.face_ids().collect();
            }
            Ok(edit::set_shade_smooth(bm, &faces, params.smooth))
        })?;
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
        let changed = with_bmesh(doc, |bm| {
            Ok(if params.angle > 0.0 {
                edit::auto_smooth(bm, params.angle as f32)
            } else {
                edit::set_sharp_selected(bm, params.sharp)
            })
        })?;
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
}

impl Default for LoopCutParams {
    fn default() -> Self {
        Self {
            cuts: 1,
            position: 0.5,
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
        let out = with_bmesh(doc, |bm| {
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
        })?;
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
        let built = with_bmesh(doc, |bm| {
            // Fall back to every open border when nothing is selected: two open
            // ends is the common case, and making the user select both first is
            // friction with no purpose.
            if bm.selected_edges().is_empty() {
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
        })?;
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
        let out = with_bmesh(doc, |bm| {
            let result = if params.distance > 0.0 {
                // By distance: the selection if there is one, the whole mesh
                // otherwise. Welding a whole imported mesh is the common case,
                // and making the user select everything first is friction with
                // no purpose.
                let mut verts = bm.selected_verts();
                if verts.is_empty() {
                    verts = bm.vert_ids().collect();
                }
                edit::weld_by_distance(bm, &verts, params.distance as f32)
            } else {
                edit::merge_selected(bm, target)
            };
            result.map_err(|e| OpError::Failed(e.to_string()))
        })?;
        Ok(if out.removed == 0 {
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
    #[param(variants("all", "none", "invert", "loop", "ring", "face_loop"))]
    pub action: u32,
    /// Which element type an inversion works on.
    #[param(variants("vert", "edge", "face"))]
    pub mode: u32,
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
            let adjacency = edit_adjacency(doc);
            let grew = with_mesh(doc, |mesh| {
                match params.action {
                    1 => selection::select_none(mesh),
                    2 => match &adjacency {
                        Some(map) => selection::select_invert_with(mesh, mode, map),
                        None => selection::select_invert(mesh, mode),
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

        let grew = with_bmesh(doc, |bm| {
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
        })?;
        Ok(if grew {
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
        let adjacency = edit_adjacency(doc);
        let changed = with_mesh(doc, |mesh| {
            // An index the mesh does not have is not an error worth failing on:
            // the id buffer describes a frame the document may have moved on
            // from, which is normal a frame or two after an edit.
            if params.element as usize >= mesh.domain_len(domain) {
                return false;
            }
            let already = selection::is_selected(mesh, domain, params.element);
            // A plain click replaces the selection; Shift+Click toggles this one
            // element and leaves the rest.
            let on = if params.extend { !already } else { true };
            if !params.extend {
                selection::select_none(mesh);
            }
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
        let had = with_mesh(doc, |mesh| {
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
        let (name, transform, parent, parent_inverse) = {
            let Some(object) = doc.document().object(source) else {
                return Err(OpError::Failed("the object is missing".into()));
            };
            (
                format!("{}.part", object.name),
                object.transform,
                object.parent,
                object.parent_inverse,
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
        let new_id = d.add_object(object);

        // Selected and active, because the next thing a user does is move it.
        d.selection.objects = vec![new_id];
        d.selection.active = Some(new_id);
        Ok(OpResult::Finished)
    }
}

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<DeleteElements>();
    registry.register::<SelectPick>();
    registry.register::<SelectSet>();
    registry.register::<SelectNothing>();
    registry.register::<DissolveElements>();
    registry.register::<BevelEdges>();
    registry.register::<BridgeEdgeLoops>();
    registry.register::<ExtrudeRegion>();
    registry.register::<FillHole>();
    registry.register::<InsetFaces>();
    registry.register::<LoopCut>();
    registry.register::<MarkSharp>();
    registry.register::<MergeVerts>();
    registry.register::<SelectElements>();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{History, OperatorRegistry, PropertyMap, PropertyValue};
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
