//! UV operators.
//!
//! One operator for all four projections, because they differ only in which
//! closed-form function turns a position into a point on the map — and because
//! `mode` being a parameter is what lets the adjust-last-operation panel try
//! box and then sphere without re-invoking anything.
//!
//! # No camera reaches here
//!
//! "Project from view" is [`Projection::Planar`] with the camera's right and
//! up vectors as parameters. The operator never sees a camera, which is §1.6's
//! rule doing its job: `execute` reads parameters and nothing else, so the
//! same projection is reproducible from a script, replayable from history and
//! testable without a window. An operator that reached for the view would be
//! none of those, and the redo panel could not re-run it.

use kreate_mesh::arena::FaceId;
use kreate_mesh::edit::uv_edit;
use kreate_mesh::edit::uv_pack;
use kreate_mesh::edit::uv_project::{self, Fit, Projection};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

use super::mesh::{can_edit, with_bmesh};

/// Which projection to use.
///
/// The order is the order of the menu, which is roughly how often each is
/// reached for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Flat, along [`ProjectParams::right`] and [`ProjectParams::up`].
    #[default]
    Planar,
    /// Six-sided, chosen per face from its normal.
    Box,
    Sphere,
    Cylinder,
}

impl Mode {
    fn from_index(i: u32) -> Self {
        match i {
            1 => Mode::Box,
            2 => Mode::Sphere,
            3 => Mode::Cylinder,
            _ => Mode::Planar,
        }
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ProjectParams {
    /// Planar, box, sphere or cylinder.
    #[param(variants("planar", "box", "sphere", "cylinder"))]
    pub mode: u32,
    /// The map's horizontal axis, for a planar projection.
    ///
    /// Also the caller's way of saying "from this view": pass the camera's
    /// right and up. Ignored by the other three modes.
    pub right: [f64; 3],
    /// The map's vertical axis, for a planar projection.
    pub up: [f64; 3],
    /// The axis a spherical or cylindrical projection wraps around.
    pub axis: [f64; 3],
    /// The centre a spherical or cylindrical projection measures from.
    #[param(unit = "length")]
    pub center: [f64; 3],
    /// Fit the result into the unit square, preserving aspect.
    ///
    /// On by default because raw world coordinates put a two-metre wall far
    /// off the map and scaling it back would be the artist's first act. Off
    /// when projecting two selections into one shared space, where fitting
    /// each to its own bounds destroys the relationship between them.
    pub fit: bool,
    /// Which UV set to write.
    ///
    /// A UV set is a named Corner layer, so a second name is a second set and
    /// costs nothing. Empty means the default set.
    pub layer: String,
}

impl Default for ProjectParams {
    fn default() -> Self {
        Self {
            mode: Mode::Planar as u32,
            // A view looking down -Y, which is the default camera. Chosen so a
            // bare invocation from a script does something sensible rather
            // than projecting everything onto a point.
            right: [1.0, 0.0, 0.0],
            up: [0.0, 0.0, 1.0],
            axis: [0.0, 0.0, 1.0],
            center: [0.0, 0.0, 0.0],
            fit: true,
            layer: String::new(),
        }
    }
}

fn vec3(v: [f64; 3]) -> glam::Vec3 {
    glam::Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)
}

impl ProjectParams {
    /// The layer these parameters name.
    pub fn layer_name(&self) -> &str {
        if self.layer.is_empty() {
            uv_project::DEFAULT_LAYER
        } else {
            &self.layer
        }
    }

    fn projection(&self) -> Projection {
        match Mode::from_index(self.mode) {
            Mode::Planar => Projection::Planar {
                right: vec3(self.right),
                up: vec3(self.up),
            },
            Mode::Box => Projection::Box,
            Mode::Sphere => Projection::Sphere {
                axis: vec3(self.axis),
                center: vec3(self.center),
            },
            Mode::Cylinder => Projection::Cylinder {
                axis: vec3(self.axis),
                center: vec3(self.center),
            },
        }
    }
}

/// Projects UVs onto the selected faces.
pub struct Project;

impl Operator for Project {
    type Params = ProjectParams;

    fn id() -> &'static str {
        "uv.project"
    }
    fn label() -> &'static str {
        "Project UVs"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let how = params.projection();
        let layer = params.layer_name().to_string();
        let fit = if params.fit { Fit::Bounds } else { Fit::None };

        let written = with_bmesh(doc, |bm| {
            // The selection, or the whole mesh when there is none — the same
            // rule as shading and normals. Unwrapping the lot is what the user
            // means on a fresh primitive, and unwrapping a selection is what
            // they mean once they have made one.
            let mut faces = bm.selected_faces();
            if faces.is_empty() {
                faces = bm.face_ids().collect::<Vec<FaceId>>();
            }
            Ok(uv_project::project_with(bm, &faces, how, &layer, fit))
        })?;

        Ok(if written == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- selection ---------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectParams {
    /// All, none, invert, or grow to whole islands.
    #[param(variants("all", "none", "invert", "linked"))]
    pub action: u32,
    /// Which UV set to act on. Empty means the default set.
    pub layer: String,
}

/// Selects UVs wholesale: all, none, inverted, or grown to islands.
pub struct Select;

impl Operator for Select {
    type Params = SelectParams;

    fn id() -> &'static str {
        "uv.select"
    }
    fn label() -> &'static str {
        "Select UVs"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let layer = layer_or_default(&params.layer).to_string();
        let action = params.action;
        with_bmesh(doc, |bm| {
            match action {
                1 => uv_edit::select_all(bm, false),
                2 => uv_edit::select_invert(bm),
                3 => uv_edit::select_linked(bm, &layer),
                _ => uv_edit::select_all(bm, true),
            };
            Ok(())
        })?;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectSetParams {
    /// The corners to act on, by slot.
    ///
    /// The resolved *set*, not the rectangle that produced it — the same rule
    /// as the 3D box select, and for the same reasons: a region depends on a
    /// view, cannot be replayed headlessly, and re-running it after the view
    /// has moved silently picks something else.
    pub corners: Vec<u32>,
    /// Add to the selection instead of replacing it.
    pub extend: bool,
    /// Remove the listed corners instead of adding them.
    pub deselect: bool,
}

/// Selects a resolved set of corners — what a click or a box drag commits.
pub struct SelectSet;

impl Operator for SelectSet {
    type Params = SelectSetParams;

    fn id() -> &'static str {
        "uv.select_set"
    }
    fn label() -> &'static str {
        "Select UVs"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let action = if params.deselect {
            uv_edit::SelectAction::Subtract
        } else if params.extend {
            uv_edit::SelectAction::Extend
        } else {
            uv_edit::SelectAction::Replace
        };
        let corners = params.corners.clone();
        with_bmesh(doc, |bm| {
            // Slots to handles through the arena, so a stale index from a
            // selection captured before a topology edit is dropped rather than
            // aliasing whatever now occupies that slot.
            let ids: Vec<kreate_mesh::arena::LoopId> = corners
                .iter()
                .filter_map(|slot| bm.corner_at(*slot))
                .collect();
            uv_edit::set_selection(bm, &ids, action);
            Ok(())
        })?;
        Ok(OpResult::Finished)
    }
}

// --- transform ---------------------------------------------------------------

/// Where a UV transform turns and scales about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pivot {
    /// The median of the selection's distinct positions.
    #[default]
    Median,
    /// The map's origin.
    Origin,
}

/// Which of the three transforms this is.
///
/// A parameter rather than three operators, the way `mesh.select`'s `action`
/// is: they share a pivot, a layer and a selection, and splitting them would
/// mean maintaining the same wrapper three times. It also gives the keymap
/// something to bind — `G`, `R` and `S` reach one operator and differ only
/// here — and gives the redo panel a control that says which tool ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransformMode {
    #[default]
    Move,
    Rotate,
    Scale,
}

impl TransformMode {
    pub fn from_index(i: u32) -> Self {
        match i {
            1 => TransformMode::Rotate,
            2 => TransformMode::Scale,
            _ => TransformMode::Move,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TransformParams {
    /// Move, turn or scale.
    #[param(variants("move", "rotate", "scale"))]
    pub mode: u32,
    #[param(soft_min = -2.0, soft_max = 2.0)]
    pub offset_u: f64,
    #[param(soft_min = -2.0, soft_max = 2.0)]
    pub offset_v: f64,
    #[param(unit = "angle")]
    pub rotation: f64,
    /// Scale along U. Zero is read as one, so a bare invocation is a move.
    #[param(soft_min = -4.0, soft_max = 4.0)]
    pub scale_u: f64,
    /// Scale along V.
    #[param(soft_min = -4.0, soft_max = 4.0)]
    pub scale_v: f64,
    /// The selection's median, or the map's origin.
    #[param(variants("median", "origin"))]
    pub pivot: u32,
    /// Which UV set to act on. Empty means the default set.
    pub layer: String,
}

/// Moves, turns and scales the selected UVs.
pub struct Transform;

impl Operator for Transform {
    type Params = TransformParams;

    fn id() -> &'static str {
        "uv.transform"
    }
    fn label() -> &'static str {
        "Transform UVs"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let layer = layer_or_default(&params.layer).to_string();
        // Only the mode's own group is applied. Otherwise a redo panel left
        // showing a rotation from the last gesture would silently fold it into
        // the next move.
        let mode = TransformMode::from_index(params.mode);
        let offset = match mode {
            TransformMode::Move => glam::Vec2::new(params.offset_u as f32, params.offset_v as f32),
            _ => glam::Vec2::ZERO,
        };
        let rotation = match mode {
            TransformMode::Rotate => params.rotation as f32,
            _ => 0.0,
        };
        // Zero reads as one, because `#[derive(Default)]` gives zero and a
        // scale of zero collapses the selection to a point. A parameter whose
        // default silently destroys the map is a trap.
        let scale = match mode {
            TransformMode::Scale => glam::Vec2::new(
                if params.scale_u == 0.0 {
                    1.0
                } else {
                    params.scale_u as f32
                },
                if params.scale_v == 0.0 {
                    1.0
                } else {
                    params.scale_v as f32
                },
            ),
            _ => glam::Vec2::ONE,
        };
        let origin_pivot = params.pivot == Pivot::Origin as u32;

        let moved = with_bmesh(doc, |bm| {
            let corners = uv_edit::selected(bm);
            if corners.is_empty() {
                return Ok(0);
            }
            let pivot = if origin_pivot {
                glam::Vec2::ZERO
            } else {
                uv_edit::median(bm, &corners, &layer)
            };
            Ok(uv_edit::transform(
                bm,
                &corners,
                &layer,
                &uv_edit::Transform2 {
                    translation: offset,
                    rotation,
                    scale,
                    pivot,
                },
            ))
        })?;

        Ok(if moved == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

/// The layer a parameter names, or the default set.
fn layer_or_default(name: &str) -> &str {
    if name.is_empty() {
        uv_project::DEFAULT_LAYER
    } else {
        name
    }
}

// --- smart project and packing ----------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SmartParams {
    /// How far a face's normal may stray from its cluster's average before it
    /// starts a new island.
    #[param(min = 0.0, max = 3.15, unit = "angle")]
    pub angle: f64,
    /// The gap left between islands, as a fraction of the map.
    #[param(min = 0.0, soft_max = 0.05)]
    pub margin: f64,
    /// Which UV set to write. Empty means the default set.
    pub layer: String,
}

impl Default for SmartParams {
    fn default() -> Self {
        Self {
            angle: f64::from(uv_pack::DEFAULT_ANGLE),
            margin: f64::from(uv_pack::DEFAULT_MARGIN),
            layer: String::new(),
        }
    }
}

/// Unwraps the selection by clustering it and packing the pieces.
///
/// The one unwrap here that is not a closed-form projection, and the one that
/// makes a whole object usable in a single press: a projection puts every face
/// on one plane, and an object facing six ways needs six.
pub struct SmartProject;

impl Operator for SmartProject {
    type Params = SmartParams;

    fn id() -> &'static str {
        "uv.smart_project"
    }
    fn label() -> &'static str {
        "Smart UV Project"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let layer = layer_or_default(&params.layer).to_string();
        let angle = params.angle as f32;
        let margin = params.margin as f32;

        let written = with_bmesh(doc, |bm| {
            let mut faces = bm.selected_faces();
            if faces.is_empty() {
                faces = bm.face_ids().collect::<Vec<FaceId>>();
            }
            Ok(uv_pack::smart_project(bm, &faces, &layer, angle, margin))
        })?;

        Ok(if written == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct PackParams {
    /// The gap left between islands, as a fraction of the map.
    #[param(min = 0.0, soft_max = 0.05)]
    pub margin: f64,
    /// Which UV set to act on. Empty means the default set.
    pub layer: String,
}

impl Default for PackParams {
    fn default() -> Self {
        Self {
            margin: f64::from(uv_pack::DEFAULT_MARGIN),
            layer: String::new(),
        }
    }
}

/// Tidies the existing islands into the unit square without recutting them.
///
/// For the artist who has cut and arranged by hand: the cuts are theirs and
/// the arithmetic is not worth doing by eye.
pub struct Pack;

impl Operator for Pack {
    type Params = PackParams;

    fn id() -> &'static str {
        "uv.pack"
    }
    fn label() -> &'static str {
        "Pack Islands"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let layer = layer_or_default(&params.layer).to_string();
        let margin = params.margin as f32;

        let written = with_bmesh(doc, |bm| {
            // Every island, not the selection's: packing a subset would move
            // it on top of the islands that were left where they were.
            let islands = uv_edit::islands(bm, &layer);
            Ok(uv_pack::pack(bm, &layer, &islands, margin))
        })?;

        Ok(if written == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct UnwrapParams {
    /// The gap left between islands, as a fraction of the map.
    #[param(min = 0.0, soft_max = 0.05)]
    pub margin: f64,
    /// Which UV set to write. Empty means the default set.
    pub layer: String,
}

impl Default for UnwrapParams {
    fn default() -> Self {
        Self {
            margin: f64::from(uv_pack::DEFAULT_MARGIN),
            layer: String::new(),
        }
    }
}

/// Blender's Unwrap: the selected faces cut along their seams and laid flat
/// with the least stretch they allow (LSCM), then packed. Every face when
/// none is selected.
pub struct Unwrap;

impl Operator for Unwrap {
    type Params = UnwrapParams;

    fn id() -> &'static str {
        "uv.unwrap"
    }
    fn label() -> &'static str {
        "Unwrap"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        can_edit(ctx, doc)
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let layer = layer_or_default(&params.layer).to_string();
        let margin = params.margin as f32;
        let written = with_bmesh(doc, |bm| {
            let mut faces = bm.selected_faces();
            if faces.is_empty() {
                faces = bm.face_ids().collect::<Vec<FaceId>>();
            }
            Ok(kreate_mesh::edit::uv_unwrap::unwrap(
                bm, &faces, &layer, margin,
            ))
        })?;
        Ok(if written == 0 {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<Unwrap>();
    registry.register::<Project>();
    registry.register::<Select>();
    registry.register::<SelectSet>();
    registry.register::<Transform>();
    registry.register::<SmartProject>();
    registry.register::<Pack>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{History, OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_mesh::attribute::Domain;
    use kreate_mesh::primitives;
    use kreate_scene::document::{MeshId, Object, ObjectData};

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

    /// A document holding one mesh, selected, in edit mode.
    fn doc_with(mesh: kreate_mesh::MeshData) -> (Document, MeshId, Context) {
        let mut doc = Document::new();
        let id = doc.add_mesh(mesh);
        let obj = doc.add_object(Object::new("Mesh", ObjectData::Mesh(id)));
        doc.selection.add(obj);
        (
            doc,
            id,
            Context {
                mode: crate::operator::Mode::EditMesh,
            },
        )
    }

    /// A cube with its own coordinates taken away.
    ///
    /// Primitives arrive with a UV layer, so a test whose claim is that *this*
    /// operator created one has to start from a mesh that has none. Undo is the
    /// case it matters for: restoring values and leaving a layer behind looks
    /// like success, and only a mesh that had no layer can tell the difference.
    fn cube_without_uvs() -> kreate_mesh::MeshData {
        let mut mesh = primitives::cube(2.0);
        mesh.attributes_mut().remove(uv_project::DEFAULT_LAYER);
        mesh
    }

    fn uvs(doc: &Document, mesh: MeshId, layer: &str) -> Vec<glam::Vec2> {
        doc.mesh(mesh)
            .unwrap()
            .attributes()
            .get_f32x2(layer)
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }

    fn project(
        r: &OperatorRegistry,
        h: &mut History,
        doc: &mut Document,
        ctx: &Context,
        pairs: &[(&'static str, PropertyValue)],
    ) {
        h.dispatch(r, doc, ctx, "uv.project", &params(pairs))
            .expect("uv.project ran");
    }

    #[test]
    fn a_projection_reaches_the_document() {
        // The point of the operator as opposed to the kernel function it
        // wraps: the UVs land on the mesh in the document, through the same
        // edit-session crossing as every other mesh operator.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = doc_with(primitives::cube(2.0));
        project(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            &[("mode", PropertyValue::Enum(Mode::Box as u32))],
        );

        let got = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);
        assert_eq!(got.len(), 24, "one UV per corner");
        assert!(got.iter().all(|uv| uv.is_finite()));
        assert!(
            got.iter().any(|uv| *uv != glam::Vec2::ZERO),
            "every UV is zero, so nothing was projected"
        );
    }

    #[test]
    fn the_layer_is_a_corner_layer() {
        // Not a detail. A UV in the Point domain cannot represent a seam,
        // which is the whole reason the domain model exists — and a layer in
        // the wrong domain has the wrong length, so it would be silently
        // truncated crossing back to `MeshData`.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = doc_with(primitives::cube(2.0));
        project(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            &[("mode", PropertyValue::Enum(Mode::Box as u32))],
        );
        let layer = doc
            .mesh(mesh)
            .unwrap()
            .attributes()
            .get(uv_project::DEFAULT_LAYER)
            .expect("the layer exists");
        assert_eq!(layer.domain, Domain::Corner);
    }

    #[test]
    fn undo_removes_a_layer_the_projection_created() {
        // The case a snapshot undo can get wrong: a layer that did not exist
        // before the operator ran. Restoring the values and leaving the layer
        // behind would look like success and leave the mesh carrying UVs
        // nobody asked for.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = doc_with(cube_without_uvs());
        assert!(uvs(&doc, mesh, uv_project::DEFAULT_LAYER).is_empty());

        project(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            &[("mode", PropertyValue::Enum(Mode::Box as u32))],
        );
        assert_eq!(uvs(&doc, mesh, uv_project::DEFAULT_LAYER).len(), 24);

        assert!(h.undo(&mut doc), "there was nothing to undo");
        assert!(
            uvs(&doc, mesh, uv_project::DEFAULT_LAYER).is_empty(),
            "undo left the UV layer behind"
        );
    }

    #[test]
    fn each_mode_gives_a_different_answer() {
        // Guards the dispatch. A `mode` parameter that fell through to one
        // branch would still produce plausible UVs, and every other test here
        // would still pass.
        let mut results = Vec::new();
        for mode in 0..4 {
            let r = registry();
            let mut h = History::new();
            let (mut doc, mesh, ctx) = doc_with(primitives::uv_sphere(1.0, 12, 6));
            project(
                &r,
                &mut h,
                &mut doc,
                &ctx,
                &[("mode", PropertyValue::Enum(mode))],
            );
            results.push(uvs(&doc, mesh, uv_project::DEFAULT_LAYER));
        }
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                assert_ne!(results[i], results[j], "modes {i} and {j} agree");
            }
        }
    }

    #[test]
    fn the_operator_projects_the_selection_rather_than_the_whole_mesh() {
        // The gap a falsification sweep found: the kernel's own test proves a
        // *selection* is a local edit, but nothing proved the operator was
        // passing the selection through. Replacing `bm.selected_faces()` with
        // the whole mesh left every test here passing, because every test
        // projected everything.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = doc_with(primitives::cube(2.0));

        // Project everything first, so every corner holds a known value and
        // "unchanged" is a real observation. Counting corners still at the
        // layer's default cannot work: the fit puts the region's lowest corner
        // at exactly (0, 0), so a projected corner and an untouched one are
        // indistinguishable.
        project(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            &[("mode", PropertyValue::Enum(Mode::Sphere as u32))],
        );
        let before = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);

        // Two of the six faces, by index. The set rather than a screen region,
        // which is what makes this replayable at all.
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
        .expect("selected two faces");

        project(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            &[("mode", PropertyValue::Enum(Mode::Box as u32))],
        );

        let after = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);
        assert_eq!(after.len(), 24, "the layer covers every corner either way");
        let changed = before.iter().zip(&after).filter(|(a, b)| a != b).count();
        assert_eq!(
            changed, 8,
            "two quads is eight corners; {changed} changed, so the operator is              not passing the selection through"
        );
    }

    #[test]
    fn a_named_set_is_written_without_disturbing_the_default() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = doc_with(primitives::cube(2.0));
        project(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            &[("mode", PropertyValue::Enum(Mode::Box as u32))],
        );
        let first = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);

        project(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            &[
                ("mode", PropertyValue::Enum(Mode::Sphere as u32)),
                ("layer", PropertyValue::Text("uv_lightmap".into())),
            ],
        );
        assert_eq!(uvs(&doc, mesh, uv_project::DEFAULT_LAYER), first);
        assert_eq!(uvs(&doc, mesh, "uv_lightmap").len(), 24);
    }

    #[test]
    fn the_view_arrives_as_parameters_rather_than_as_a_camera() {
        // §1.6's rule, checked rather than asserted in a comment: two
        // different bases give two different projections, and neither needed a
        // camera to exist. That is what makes "project from view" replayable
        // from a script and adjustable in the redo panel.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = doc_with(primitives::cube(2.0));
        project(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            &[
                ("mode", PropertyValue::Enum(Mode::Planar as u32)),
                ("right", PropertyValue::Vec3([1.0, 0.0, 0.0])),
                ("up", PropertyValue::Vec3([0.0, 0.0, 1.0])),
            ],
        );
        let front = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);

        project(
            &r,
            &mut h,
            &mut doc,
            &ctx,
            &[
                ("mode", PropertyValue::Enum(Mode::Planar as u32)),
                ("right", PropertyValue::Vec3([0.0, 1.0, 0.0])),
                ("up", PropertyValue::Vec3([0.0, 0.0, 1.0])),
            ],
        );
        assert_ne!(front, uvs(&doc, mesh, uv_project::DEFAULT_LAYER));
    }

    fn selected_count(doc: &Document, mesh: MeshId) -> usize {
        doc.mesh(mesh)
            .unwrap()
            .attributes()
            .get_bool(kreate_mesh::attribute::names::SELECT_UV)
            .map(|f| f.iter().filter(|x| **x).count())
            .unwrap_or(0)
    }

    /// A cube with box-projected UVs and nothing selected.
    fn projected_cube(r: &OperatorRegistry, h: &mut History) -> (Document, MeshId, Context) {
        let (mut doc, mesh, ctx) = doc_with(primitives::cube(2.0));
        project(
            r,
            h,
            &mut doc,
            &ctx,
            &[("mode", PropertyValue::Enum(Mode::Box as u32))],
        );
        (doc, mesh, ctx)
    }

    #[test]
    fn select_all_none_and_invert_go_through_the_operator() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = projected_cube(&r, &mut h);

        let sel = |doc: &mut Document, h: &mut History, action: u32| {
            h.dispatch(
                &r,
                doc,
                &ctx,
                "uv.select",
                &params(&[("action", PropertyValue::Enum(action))]),
            )
            .expect("uv.select ran");
        };

        sel(&mut doc, &mut h, 0);
        assert_eq!(selected_count(&doc, mesh), 24);
        sel(&mut doc, &mut h, 1);
        assert_eq!(selected_count(&doc, mesh), 0);
        sel(&mut doc, &mut h, 2);
        assert_eq!(selected_count(&doc, mesh), 24);
    }

    #[test]
    fn a_resolved_set_of_corners_is_what_a_click_commits() {
        // The parameter is the set, not the rectangle. Same rule as the 3D box
        // select: a region depends on a view and cannot be replayed.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = projected_cube(&r, &mut h);

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select_set",
            &params(&[("corners", PropertyValue::Ints(vec![0, 1, 2, 3]))]),
        )
        .expect("select_set ran");
        assert_eq!(selected_count(&doc, mesh), 4);

        // Extend adds, replace replaces.
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select_set",
            &params(&[
                ("corners", PropertyValue::Ints(vec![4, 5])),
                ("extend", PropertyValue::Bool(true)),
            ]),
        )
        .expect("select_set extended");
        assert_eq!(selected_count(&doc, mesh), 6);

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select_set",
            &params(&[
                ("corners", PropertyValue::Ints(vec![0])),
                ("extend", PropertyValue::Bool(true)),
                ("deselect", PropertyValue::Bool(true)),
            ]),
        )
        .expect("select_set deselected");
        assert_eq!(selected_count(&doc, mesh), 5);

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select_set",
            &params(&[("corners", PropertyValue::Ints(vec![7]))]),
        )
        .expect("select_set replaced");
        assert_eq!(selected_count(&doc, mesh), 1);
    }

    #[test]
    fn a_slot_that_no_longer_holds_a_corner_is_dropped() {
        // A selection captured before a topology edit carries stale slots. The
        // arena's generation check is what keeps them from aliasing whatever
        // now occupies the slot, and `corner_at` is where that happens — an
        // out-of-range index must simply not select anything.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = projected_cube(&r, &mut h);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select_set",
            &params(&[("corners", PropertyValue::Ints(vec![0, 9_999]))]),
        )
        .expect("select_set ran");
        assert_eq!(selected_count(&doc, mesh), 1, "a stale slot was honoured");
    }

    #[test]
    fn select_linked_grows_to_whole_islands() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = projected_cube(&r, &mut h);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select_set",
            &params(&[("corners", PropertyValue::Ints(vec![0]))]),
        )
        .expect("select_set ran");
        assert_eq!(selected_count(&doc, mesh), 1);

        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select",
            &params(&[("action", PropertyValue::Enum(3))]),
        )
        .expect("select linked ran");
        // A box-projected cube's islands are single quads.
        assert_eq!(selected_count(&doc, mesh), 4);
    }

    #[test]
    fn a_transform_moves_only_the_selected_uvs() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = projected_cube(&r, &mut h);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select_set",
            &params(&[("corners", PropertyValue::Ints(vec![0, 1, 2, 3]))]),
        )
        .expect("select_set ran");

        let before = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.transform",
            &params(&[
                ("offset_u", PropertyValue::Float(0.25)),
                ("offset_v", PropertyValue::Float(-0.5)),
            ]),
        )
        .expect("transform ran");
        let after = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);

        let moved: Vec<usize> = before
            .iter()
            .zip(&after)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(moved.len(), 4, "{} corners moved", moved.len());
        for i in moved {
            let d = after[i] - before[i];
            assert!((d - glam::Vec2::new(0.25, -0.5)).length() < 1e-6, "{d:?}");
        }
    }

    #[test]
    fn a_zero_scale_parameter_reads_as_one() {
        // `#[derive(Default)]` gives zero, and a scale of zero collapses the
        // selection to a point. A parameter whose default silently destroys
        // the mesh is a trap, so zero means "not given".
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = projected_cube(&r, &mut h);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select",
            &params(&[("action", PropertyValue::Enum(0))]),
        )
        .expect("select all");

        let before = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);
        // In *scale* mode, which is where the rule bites: a move ignores the
        // scale group entirely, so asking in move mode would prove nothing.
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.transform",
            &params(&[
                ("mode", PropertyValue::Enum(2)),
                ("scale_v", PropertyValue::Float(1.5)),
            ]),
        )
        .expect("transform ran");
        let after = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);

        let distinct_before = {
            let mut v: Vec<String> = before.iter().map(|uv| format!("{uv:?}")).collect();
            v.sort();
            v.dedup();
            v.len()
        };
        let distinct_after = {
            let mut v: Vec<String> = after.iter().map(|uv| format!("{uv:?}")).collect();
            v.sort();
            v.dedup();
            v.len()
        };
        assert_eq!(
            distinct_before, distinct_after,
            "the map collapsed: a default scale of zero was applied"
        );
    }

    #[test]
    fn only_the_named_mode_is_applied() {
        // The redo panel keeps whatever the last gesture left in it. Without
        // this, a rotation from the previous tool would silently fold into the
        // next move — one operator with three parameter groups has to apply
        // exactly one of them.
        //
        // All three modes, each asked with all three groups populated. Testing
        // only the move case cannot see an offset that is applied
        // unconditionally, because in move mode that is the right answer:
        // a falsification sweep removed the guard and the test still passed.
        let r = registry();

        let run = |mode: u32| -> (Vec<glam::Vec2>, Vec<glam::Vec2>) {
            let mut h = History::new();
            let (mut doc, mesh, ctx) = projected_cube(&r, &mut h);
            h.dispatch(
                &r,
                &mut doc,
                &ctx,
                "uv.select",
                &params(&[("action", PropertyValue::Enum(0))]),
            )
            .expect("select all");
            let before = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);
            h.dispatch(
                &r,
                &mut doc,
                &ctx,
                "uv.transform",
                &params(&[
                    ("mode", PropertyValue::Enum(mode)),
                    ("offset_u", PropertyValue::Float(0.25)),
                    ("rotation", PropertyValue::Float(1.0)),
                    ("scale_u", PropertyValue::Float(3.0)),
                    ("scale_v", PropertyValue::Float(3.0)),
                ]),
            )
            .expect("transform ran");
            (before, uvs(&doc, mesh, uv_project::DEFAULT_LAYER))
        };

        let centroid = |v: &[glam::Vec2]| -> glam::Vec2 {
            v.iter().copied().sum::<glam::Vec2>() / v.len() as f32
        };
        let spread = |v: &[glam::Vec2]| -> f32 {
            let c = centroid(v);
            v.iter().map(|p| (*p - c).length()).sum::<f32>() / v.len() as f32
        };

        // Move: every corner shifts by exactly the offset. No turn, no scale.
        let (before, after) = run(0);
        for (a, b) in before.iter().zip(&after) {
            let d = *b - *a;
            assert!(
                (d - glam::Vec2::new(0.25, 0.0)).length() < 1e-5,
                "move: a corner moved by {d:?}; the rotation or scale leaked in"
            );
        }

        // Rotate: the centroid stays where it was — so no offset was applied —
        // and the corners have moved, so the rotation did happen.
        let (before, after) = run(1);
        let drift = (centroid(&after) - centroid(&before)).length();
        assert!(
            drift < 1e-5,
            "rotate: the centroid moved by {drift}; the offset leaked in"
        );
        assert!(
            (spread(&after) - spread(&before)).abs() < 1e-4,
            "rotate: the spread changed; the scale leaked in"
        );
        assert!(
            before.iter().zip(&after).any(|(a, b)| a != b),
            "rotate: nothing moved at all"
        );

        // Scale: the centroid stays put and the spread grows by the factor.
        let (before, after) = run(2);
        let drift = (centroid(&after) - centroid(&before)).length();
        assert!(
            drift < 1e-5,
            "scale: the centroid moved by {drift}; the offset leaked in"
        );
        let grew = spread(&after) / spread(&before);
        assert!(
            (grew - 3.0).abs() < 1e-3,
            "scale: the spread grew by {grew}, not three"
        );
    }

    #[test]
    fn the_pivot_choice_changes_where_a_rotation_turns_about() {
        let r = registry();
        let mut h = History::new();

        let run = |pivot: u32| -> Vec<glam::Vec2> {
            let mut h2 = History::new();
            let (mut doc, mesh, ctx) = projected_cube(&r, &mut h2);
            h2.dispatch(
                &r,
                &mut doc,
                &ctx,
                "uv.select",
                &params(&[("action", PropertyValue::Enum(0))]),
            )
            .expect("select all");
            h2.dispatch(
                &r,
                &mut doc,
                &ctx,
                "uv.transform",
                &params(&[
                    ("mode", PropertyValue::Enum(1)),
                    ("rotation", PropertyValue::Float(0.7)),
                    ("pivot", PropertyValue::Enum(pivot)),
                ]),
            )
            .expect("transform ran");
            uvs(&doc, mesh, uv_project::DEFAULT_LAYER)
        };

        let _ = &mut h;
        assert_ne!(run(0), run(1), "median and origin gave the same answer");
    }

    #[test]
    fn undo_restores_the_uvs_a_transform_moved() {
        // §1.7 on a layer the operator only *modifies*, as opposed to the
        // projection test, which covers a layer it creates.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = projected_cube(&r, &mut h);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.select",
            &params(&[("action", PropertyValue::Enum(0))]),
        )
        .expect("select all");

        let before = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.transform",
            &params(&[("offset_u", PropertyValue::Float(0.3))]),
        )
        .expect("transform ran");
        assert_ne!(uvs(&doc, mesh, uv_project::DEFAULT_LAYER), before);

        assert!(h.undo(&mut doc), "nothing to undo");
        assert_eq!(uvs(&doc, mesh, uv_project::DEFAULT_LAYER), before);
    }

    #[test]
    fn a_transform_with_nothing_selected_is_cancelled() {
        // Not an error, and not a no-op that still lands on the undo stack: a
        // tool that reports Finished with nothing to do puts an empty step in
        // the history, and the user's next undo does nothing visible.
        let r = registry();
        let mut h = History::new();
        let (mut doc, _, ctx) = projected_cube(&r, &mut h);
        let depth = h.entries().len();
        h.dispatch(
            &r,
            &mut doc,
            &ctx,
            "uv.transform",
            &params(&[("offset_u", PropertyValue::Float(0.3))]),
        )
        .expect("transform ran");
        assert_eq!(
            h.entries().len(),
            depth,
            "an empty transform reached the history"
        );
    }

    #[test]
    fn a_smart_project_unwraps_a_whole_cube_inside_the_map() {
        // The one press that makes an object usable. Six faces facing six ways
        // cannot go on one plane, which is what separates this from every other
        // mode of `uv.project`.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = doc_with(primitives::cube(2.0));
        h.dispatch(&r, &mut doc, &ctx, "uv.smart_project", &params(&[]))
            .expect("smart project ran");

        let got = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);
        assert_eq!(got.len(), 24);
        for uv in &got {
            assert!(uv.is_finite(), "{uv:?}");
            assert!(
                uv.x >= -1e-3 && uv.x <= 1.0 + 1e-3 && uv.y >= -1e-3 && uv.y <= 1.0 + 1e-3,
                "a corner landed outside the map at {uv:?}"
            );
        }
        // And it really did cut: a single plane would have put every corner on
        // one quad's worth of the map.
        let distinct = {
            let mut v: Vec<String> = got
                .iter()
                .map(|uv| format!("{:.3},{:.3}", uv.x, uv.y))
                .collect();
            v.sort();
            v.dedup();
            v.len()
        };
        assert!(
            distinct >= 12,
            "only {distinct} distinct UVs; nothing was cut"
        );
    }

    #[test]
    fn the_angle_parameter_changes_how_much_is_cut() {
        // Guards the test above. A smart project that ignored its angle would
        // still land inside the map and still look plausible.
        let r = registry();

        let distinct_for = |angle: f64| -> usize {
            let mut h = History::new();
            let (mut doc, mesh, ctx) = doc_with(primitives::cylinder(1.0, 2.0, 16));
            h.dispatch(
                &r,
                &mut doc,
                &ctx,
                "uv.smart_project",
                &params(&[("angle", PropertyValue::Float(angle))]),
            )
            .expect("smart project ran");
            let got = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);
            let mut v: Vec<String> = got
                .iter()
                .map(|uv| format!("{:.4},{:.4}", uv.x, uv.y))
                .collect();
            v.sort();
            v.dedup();
            v.len()
        };

        // A tight limit cuts the cylinder's side into many islands, so far more
        // corners end up at distinct UVs than a loose one, which keeps the
        // strip whole and shares corners along it.
        let tight = distinct_for(5.0_f64.to_radians());
        let loose = distinct_for(120.0_f64.to_radians());
        assert!(
            tight > loose,
            "the angle changed nothing: {tight} distinct at 5 degrees, {loose} at 120"
        );
    }

    #[test]
    fn packing_tidies_without_recutting() {
        // `uv.pack` is for the artist who has cut and arranged by hand. It must
        // move the islands and not merge them.
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = projected_cube(&r, &mut h);
        let before = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);

        h.dispatch(&r, &mut doc, &ctx, "uv.pack", &params(&[]))
            .expect("pack ran");
        let after = uvs(&doc, mesh, uv_project::DEFAULT_LAYER);

        assert_ne!(before, after, "packing moved nothing");
        for uv in &after {
            assert!(
                uv.x >= -1e-3 && uv.x <= 1.0 + 1e-3 && uv.y >= -1e-3 && uv.y <= 1.0 + 1e-3,
                "a corner landed outside the map at {uv:?}"
            );
        }
        // Six islands went in; six must come out. A packer that merged them
        // would still put everything inside the square.
        let (_, m) = doc.meshes().next().unwrap();
        let bm = kreate_mesh::bmesh::BMesh::from_mesh_data(m);
        assert_eq!(
            uv_edit::islands(&bm, uv_project::DEFAULT_LAYER).len(),
            6,
            "the islands were merged"
        );
    }

    #[test]
    fn packing_a_mesh_with_no_uvs_is_cancelled() {
        // Not an error, and not a history entry: there is nothing to pack, and
        // an empty step between the user and the edit they wanted back is
        // worse than nothing happening.
        let r = registry();
        let mut h = History::new();
        let (mut doc, _, ctx) = doc_with(cube_without_uvs());
        let depth = h.entries().len();
        h.dispatch(&r, &mut doc, &ctx, "uv.pack", &params(&[]))
            .expect("pack ran");
        assert_eq!(h.entries().len(), depth);
    }

    #[test]
    fn undo_takes_back_a_smart_project() {
        let r = registry();
        let mut h = History::new();
        let (mut doc, mesh, ctx) = doc_with(cube_without_uvs());
        assert!(uvs(&doc, mesh, uv_project::DEFAULT_LAYER).is_empty());
        h.dispatch(&r, &mut doc, &ctx, "uv.smart_project", &params(&[]))
            .expect("smart project ran");
        assert_eq!(uvs(&doc, mesh, uv_project::DEFAULT_LAYER).len(), 24);
        assert!(h.undo(&mut doc));
        assert!(uvs(&doc, mesh, uv_project::DEFAULT_LAYER).is_empty());
    }

    #[test]
    fn it_declines_outside_edit_mode() {
        let (doc, _, _) = doc_with(primitives::cube(2.0));
        let object_mode = Context {
            mode: crate::operator::Mode::Object,
        };
        assert!(!Project::poll(&object_mode, &doc));
    }
}
