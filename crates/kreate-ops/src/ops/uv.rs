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

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<Project>();
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
        let (mut doc, mesh, ctx) = doc_with(primitives::cube(2.0));
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

    #[test]
    fn it_declines_outside_edit_mode() {
        let (doc, _, _) = doc_with(primitives::cube(2.0));
        let object_mode = Context {
            mode: crate::operator::Mode::Object,
        };
        assert!(!Project::poll(&object_mode, &doc));
    }
}
