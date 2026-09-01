//! Object-mode operators.

use kreate_math::{DQuat, DVec3};
use kreate_scene::document::ObjectId;
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

/// The primitive shapes `object.add_primitive` can build.
///
/// Indices are part of the scripting and keymap surface, so entries are only
/// ever appended — reordering them would silently change what a saved keymap or
/// an old script does.
pub const PRIMITIVES: &[&str] = &[
    "plane",
    "cube",
    "grid",
    "uv_sphere",
    "ico_sphere",
    "cylinder",
    "cone",
    "torus",
];

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddPrimitiveParams {
    /// Which shape to add.
    #[param(variants(
        "plane",
        "cube",
        "grid",
        "uv_sphere",
        "ico_sphere",
        "cylinder",
        "cone",
        "torus"
    ))]
    pub shape: u32,
    /// Overall size, in metres.
    #[param(min = 0.0001, soft_max = 100.0, unit = "length")]
    pub size: f64,
    /// Segments around the main axis, for the rounded shapes.
    #[param(min = 3, max = 512, unit = "count")]
    pub segments: u32,
    /// Rings from pole to pole, or minor segments for a torus.
    #[param(min = 2, max = 512, unit = "count")]
    pub rings: u32,
    /// Where to place the new object.
    pub location: [f64; 3],
}

impl Default for AddPrimitiveParams {
    fn default() -> Self {
        Self {
            shape: 1, // cube
            size: 2.0,
            segments: 32,
            rings: 16,
            location: [0.0; 3],
        }
    }
}

/// Adds a primitive mesh object.
pub struct AddPrimitive;

impl Operator for AddPrimitive {
    type Params = AddPrimitiveParams;

    fn id() -> &'static str {
        "object.add_primitive"
    }
    fn label() -> &'static str {
        "Add Primitive"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_mesh::primitives as p;
        let size = params.size as f32;
        let seg = params.segments;
        let rings = params.rings;
        let name = PRIMITIVES
            .get(params.shape as usize)
            .copied()
            .unwrap_or("cube");

        let mesh = match name {
            "plane" => p::plane(size),
            "grid" => p::grid(size, seg.min(256), rings.min(256)),
            "uv_sphere" => p::uv_sphere(size * 0.5, seg, rings),
            "ico_sphere" => p::ico_sphere(size * 0.5, (rings as f32).log2().round() as u32),
            "cylinder" => p::cylinder(size * 0.5, size, seg),
            "cone" => p::cone(size * 0.5, size, seg),
            "torus" => p::torus(size * 0.5, size * 0.15, seg, rings),
            _ => p::cube(size),
        };

        // Validated here rather than trusted: a primitive generator that emits
        // invalid topology would seed corruption that only surfaces three
        // operators later, with no way to attribute it.
        let violations = mesh.validate();
        if !violations.is_empty() {
            return Err(OpError::Failed(format!(
                "generated {name} is invalid: {violations:?}"
            )));
        }

        let d = doc.document_mut();
        let id = d.add_mesh_object(capitalise(name), mesh);
        d.object_mut(id).expect("just added").transform.translation =
            DVec3::from_array(params.location);
        // A newly added object becomes the selection, which is what every
        // subsequent operator acts on.
        d.selection.clear();
        d.selection.add(id);
        Ok(OpResult::Finished)
    }
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DeleteParams {}

/// Deletes the selected objects.
pub struct Delete;

impl Operator for Delete {
    type Params = DeleteParams;

    fn id() -> &'static str {
        "object.delete"
    }
    fn label() -> &'static str {
        "Delete Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in targets {
            d.remove_object(id);
        }
        d.selection.clear();
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DuplicateParams {
    /// Share the original's geometry instead of copying it.
    ///
    /// A linked duplicate costs one transform; editing either copy changes both.
    pub linked: bool,
}

/// Duplicates the selected objects.
pub struct Duplicate;

impl Operator for Duplicate {
    type Params = DuplicateParams;

    fn id() -> &'static str {
        "object.duplicate"
    }
    fn label() -> &'static str {
        "Duplicate Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        let mut copies = Vec::with_capacity(targets.len());
        for id in targets {
            let copy = if params.linked {
                d.duplicate_linked(id)
            } else {
                d.duplicate_full(id)
            };
            if let Some(c) = copy {
                copies.push(c);
            }
        }
        if copies.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        // The copies become the selection, so a duplicate-then-move macro moves
        // the new objects rather than the originals.
        d.selection.clear();
        for c in copies {
            d.selection.add(c);
        }
        Ok(OpResult::Finished)
    }
}

/// Which point a rotation or scale happens about.
pub const PIVOTS: &[&str] = &["median", "individual", "active", "world_origin"];

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TranslateParams {
    /// Offset applied to every selected object, in metres.
    #[param(unit = "length")]
    pub translation: [f64; 3],
}

/// Moves the selected objects.
pub struct Translate;

impl Operator for Translate {
    type Params = TranslateParams;

    fn id() -> &'static str {
        "object.translate"
    }
    fn label() -> &'static str {
        "Move Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let offset = DVec3::from_array(params.translation);
        let d = doc.document_mut();
        for id in targets {
            if let Some(o) = d.object_mut(id) {
                o.transform.translation += offset;
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RotateParams {
    /// Angle in radians. The UI shows and parses degrees.
    #[param(unit = "angle")]
    pub angle: f64,
    /// Axis to rotate about.
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
    /// What the rotation happens about.
    #[param(variants("median", "individual", "active", "world_origin"))]
    pub pivot: u32,
}

/// Rotates the selected objects.
pub struct Rotate;

impl Operator for Rotate {
    type Params = RotateParams;

    fn id() -> &'static str {
        "object.rotate"
    }
    fn label() -> &'static str {
        "Rotate Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let axis = match params.axis {
            0 => DVec3::X,
            1 => DVec3::Y,
            _ => DVec3::Z,
        };
        let rotation = DQuat::from_axis_angle(axis, params.angle);
        let pivot = pivot_point(doc.document(), &targets, params.pivot);

        let d = doc.document_mut();
        for id in targets {
            // `individual` rotates each object about its own origin, so there is
            // no orbiting component.
            let about = if params.pivot == 1 {
                d.object(id).map(|o| o.transform.translation)
            } else {
                pivot
            };
            let Some(about) = about else { continue };
            if let Some(o) = d.object_mut(id) {
                let offset = o.transform.translation - about;
                o.transform.translation = about + rotation * offset;
                o.transform.rotation = (rotation * o.transform.rotation).normalize();
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ScaleParams {
    /// Scale factor per axis.
    pub scale: [f64; 3],
    /// What the scaling happens about.
    #[param(variants("median", "individual", "active", "world_origin"))]
    pub pivot: u32,
}

impl Default for ScaleParams {
    fn default() -> Self {
        Self {
            scale: [1.0; 3],
            pivot: 0,
        }
    }
}

/// Scales the selected objects.
pub struct Scale;

impl Operator for Scale {
    type Params = ScaleParams;

    fn id() -> &'static str {
        "object.scale"
    }
    fn label() -> &'static str {
        "Scale Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let targets = doc.document().selection.objects.clone();
        if targets.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let factor = DVec3::from_array(params.scale);
        let pivot = pivot_point(doc.document(), &targets, params.pivot);

        let d = doc.document_mut();
        for id in targets {
            let about = if params.pivot == 1 {
                d.object(id).map(|o| o.transform.translation)
            } else {
                pivot
            };
            let Some(about) = about else { continue };
            if let Some(o) = d.object_mut(id) {
                let offset = o.transform.translation - about;
                o.transform.translation = about + offset * factor;
                o.transform.scale *= factor;
            }
        }
        Ok(OpResult::Finished)
    }
}

/// Resolves a pivot mode to a point, or `None` for `individual`, which has a
/// different point per object.
fn pivot_point(doc: &Document, targets: &[ObjectId], mode: u32) -> Option<DVec3> {
    match mode {
        // median
        0 => {
            let mut sum = DVec3::ZERO;
            let mut count = 0.0;
            for id in targets {
                if let Some(o) = doc.object(*id) {
                    sum += o.transform.translation;
                    count += 1.0;
                }
            }
            (count > 0.0).then(|| sum / count)
        }
        1 => None,
        // active
        2 => doc
            .selection
            .active
            .and_then(|a| doc.object(a))
            .map(|o| o.transform.translation),
        _ => Some(DVec3::ZERO),
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectAllParams {
    /// What to do to the selection.
    #[param(variants("select", "deselect", "toggle", "invert"))]
    pub action: u32,
}

/// Selects, deselects, toggles or inverts the object selection.
pub struct SelectAll;

impl Operator for SelectAll {
    type Params = SelectAllParams;

    fn id() -> &'static str {
        "object.select_all"
    }
    fn label() -> &'static str {
        "Select All Objects"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let selectable: Vec<ObjectId> = doc
            .document()
            .objects()
            .filter(|(_, o)| o.selectable)
            .map(|(id, _)| id)
            .collect();
        let currently = doc.document().selection.objects.clone();

        // `toggle` means "select all unless everything already is", which is what
        // a single key doing double duty has to mean.
        let action = match params.action {
            2 => {
                if currently.len() == selectable.len() && !selectable.is_empty() {
                    1
                } else {
                    0
                }
            }
            other => other,
        };

        let d = doc.document_mut();
        match action {
            0 => {
                d.selection.clear();
                for id in selectable {
                    d.selection.add(id);
                }
            }
            1 => d.selection.clear(),
            _ => {
                d.selection.clear();
                for id in selectable.into_iter().filter(|id| !currently.contains(id)) {
                    d.selection.add(id);
                }
            }
        }
        // Selection changes are undoable: users expect Ctrl+Z to restore a
        // selection, and losing a careful one is more annoying than losing an edit.
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ParentParams {
    /// Clear the parent instead of setting it to the active object.
    pub clear: bool,
}

/// Parents the selected objects to the active one.
pub struct Parent;

impl Operator for Parent {
    type Params = ParentParams;

    fn id() -> &'static str {
        "object.parent"
    }
    fn label() -> &'static str {
        "Parent Objects"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let selection = doc.document().selection.clone();
        let parent = if params.clear { None } else { selection.active };
        if !params.clear && parent.is_none() {
            return Err(OpError::Failed("no active object to parent to".into()));
        }

        let d = doc.document_mut();
        let mut changed = false;
        let mut refused = 0;
        for id in selection.objects.iter().copied() {
            if Some(id) == parent {
                continue;
            }
            match d.set_parent(id, parent) {
                Ok(()) => changed = true,
                // A cycle is refused per object rather than failing the whole
                // operation: parenting a mixed selection where one member is an
                // ancestor should do the sensible part.
                Err(_) => refused += 1,
            }
        }
        if !changed {
            return if refused > 0 {
                Err(OpError::Failed(format!(
                    "{refused} object(s) would have created a cycle"
                )))
            } else {
                Ok(OpResult::Cancelled)
            };
        }
        Ok(OpResult::Finished)
    }
}

/// Registers every object-mode operator.
pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddPrimitive>();
    registry.register::<Delete>();
    registry.register::<Duplicate>();
    registry.register::<Translate>();
    registry.register::<Rotate>();
    registry.register::<Scale>();
    registry.register::<SelectAll>();
    registry.register::<Parent>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::OperatorRegistry;
    use crate::{PropertyMap, PropertyValue};

    fn setup() -> (OperatorRegistry, History, Document, Context) {
        let mut r = OperatorRegistry::new();
        register(&mut r);
        (r, History::new(), Document::new(), Context::default())
    }

    fn run(
        r: &OperatorRegistry,
        h: &mut History,
        d: &mut Document,
        c: &Context,
        id: &str,
        pairs: &[(&'static str, PropertyValue)],
    ) -> OpResult {
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        h.dispatch(r, d, c, id, &p).expect(id)
    }

    #[test]
    fn every_primitive_can_be_added_and_is_valid() {
        // The generators are validated inside the operator, so this also proves
        // no shape seeds invalid topology into a document.
        let (r, mut h, mut d, c) = setup();
        for (i, name) in PRIMITIVES.iter().enumerate() {
            let result = run(
                &r,
                &mut h,
                &mut d,
                &c,
                "object.add_primitive",
                &[("shape", PropertyValue::Enum(i as u32))],
            );
            assert_eq!(result, OpResult::Finished, "{name}");
        }
        assert_eq!(d.object_count(), PRIMITIVES.len());
    }

    #[test]
    fn adding_a_primitive_selects_it() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        assert_eq!(d.selection.objects.len(), 1);
        assert!(d.selection.active.is_some());
    }

    #[test]
    fn a_primitive_is_placed_at_the_requested_location() {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("location", PropertyValue::Vec3([1.0, 2.0, 3.0]))],
        );
        let id = d.selection.active.unwrap();
        assert_eq!(
            d.object(id).unwrap().transform.translation,
            DVec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn an_out_of_range_shape_index_falls_back_rather_than_failing() {
        // A saved keymap or script may name a shape that no longer exists.
        let (r, mut h, mut d, c) = setup();
        let result = run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("shape", PropertyValue::Enum(999))],
        );
        assert_eq!(result, OpResult::Finished);
        assert_eq!(d.object_count(), 1);
    }

    #[test]
    fn adding_then_undoing_leaves_an_empty_document() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        h.undo(&mut d);
        assert_eq!(d.object_count(), 0);
        assert!(d.selection.is_empty(), "the selection must be restored too");
    }

    #[test]
    fn deleting_removes_the_selection_and_undo_brings_it_back() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(&r, &mut h, &mut d, &c, "object.delete", &[]);
        assert_eq!(d.object_count(), 0);

        h.undo(&mut d);
        assert_eq!(d.object_count(), 1);
        assert_eq!(
            d.selection.objects.len(),
            1,
            "the selection comes back as well"
        );
    }

    #[test]
    fn a_linked_duplicate_shares_geometry_and_a_full_one_does_not() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.duplicate",
            &[("linked", PropertyValue::Bool(true))],
        );
        assert_eq!(d.object_count(), 2);
        assert_eq!(d.mesh_count(), 1, "linked must not copy geometry");

        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.duplicate",
            &[("linked", PropertyValue::Bool(false))],
        );
        assert_eq!(d.mesh_count(), 2);
    }

    #[test]
    fn duplicating_selects_the_copies() {
        // So that a duplicate-then-move macro moves the new objects.
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let original = d.selection.active.unwrap();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.duplicate",
            &[("linked", PropertyValue::Bool(true))],
        );
        assert_eq!(d.selection.objects.len(), 1);
        assert_ne!(d.selection.active, Some(original));
    }

    #[test]
    fn translate_moves_every_selected_object() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.duplicate",
            &[("linked", PropertyValue::Bool(true))],
        );
        run(&r, &mut h, &mut d, &c, "object.select_all", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.translate",
            &[("translation", PropertyValue::Vec3([0.0, 0.0, 5.0]))],
        );
        for (_, o) in d.objects() {
            assert_eq!(o.transform.translation.z, 5.0);
        }
    }

    #[test]
    fn rotating_about_the_median_orbits_the_objects() {
        let (r, mut h, mut d, c) = setup();
        // Two objects at x = 0 and x = 2, median x = 1.
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("location", PropertyValue::Vec3([2.0, 0.0, 0.0]))],
        );
        run(&r, &mut h, &mut d, &c, "object.select_all", &[]);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.rotate",
            &[
                ("angle", PropertyValue::Float(std::f64::consts::PI)),
                ("axis", PropertyValue::Enum(2)),
                ("pivot", PropertyValue::Enum(0)),
            ],
        );
        // A half turn about x = 1 swaps them.
        let xs: Vec<f64> = d
            .objects()
            .map(|(_, o)| (o.transform.translation.x * 1e6).round() / 1e6)
            .collect();
        assert!(xs.contains(&2.0) && xs.contains(&0.0), "got {xs:?}");
    }

    #[test]
    fn rotating_about_individual_origins_does_not_move_anything() {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("location", PropertyValue::Vec3([5.0, 0.0, 0.0]))],
        );
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.rotate",
            &[
                ("angle", PropertyValue::Float(1.0)),
                ("pivot", PropertyValue::Enum(1)),
            ],
        );
        let id = d.selection.active.unwrap();
        let o = d.object(id).unwrap();
        assert_eq!(o.transform.translation, DVec3::new(5.0, 0.0, 0.0));
        assert_ne!(
            o.transform.rotation,
            DQuat::IDENTITY,
            "it should still have turned"
        );
    }

    #[test]
    fn scaling_about_the_world_origin_moves_objects_outward() {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("location", PropertyValue::Vec3([2.0, 0.0, 0.0]))],
        );
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.scale",
            &[
                ("scale", PropertyValue::Vec3([2.0, 2.0, 2.0])),
                ("pivot", PropertyValue::Enum(3)),
            ],
        );
        let id = d.selection.active.unwrap();
        assert_eq!(d.object(id).unwrap().transform.translation.x, 4.0);
        assert_eq!(d.object(id).unwrap().transform.scale.x, 2.0);
    }

    #[test]
    fn select_all_toggles_between_all_and_none() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);

        // One is selected after the second add; toggle should select all.
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.select_all",
            &[("action", PropertyValue::Enum(2))],
        );
        assert_eq!(d.selection.objects.len(), 2);

        // Everything is selected, so toggle now deselects.
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.select_all",
            &[("action", PropertyValue::Enum(2))],
        );
        assert!(d.selection.is_empty());
    }

    #[test]
    fn select_all_can_invert() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let first = d.selection.active.unwrap();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let second = d.selection.active.unwrap();

        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.select_all",
            &[("action", PropertyValue::Enum(3))],
        );
        assert!(d.selection.contains(first));
        assert!(!d.selection.contains(second));
    }

    #[test]
    fn a_selection_change_is_undoable() {
        // Users expect Ctrl+Z to restore a selection; losing a careful one is
        // more annoying than losing an edit.
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let before = d.selection.clone();

        run(&r, &mut h, &mut d, &c, "object.select_all", &[]);
        assert_ne!(d.selection, before);

        h.undo(&mut d);
        assert_eq!(d.selection, before);
    }

    #[test]
    fn parenting_refuses_a_cycle_per_object_and_does_the_rest() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let a = d.selection.active.unwrap();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let b = d.selection.active.unwrap();
        d.set_parent(b, Some(a)).unwrap();

        // Select both with `a` active: `b` is fine, `a` itself is skipped.
        d.selection.clear();
        d.selection.add(b);
        d.selection.add(a);
        let result = run(&r, &mut h, &mut d, &c, "object.parent", &[]);
        assert_eq!(result, OpResult::Finished);
        assert_eq!(d.object(b).unwrap().parent, Some(a));
    }

    #[test]
    fn parenting_can_clear() {
        let (r, mut h, mut d, c) = setup();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let a = d.selection.active.unwrap();
        run(&r, &mut h, &mut d, &c, "object.add_primitive", &[]);
        let b = d.selection.active.unwrap();
        d.set_parent(b, Some(a)).unwrap();

        d.selection.clear();
        d.selection.add(b);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.parent",
            &[("clear", PropertyValue::Bool(true))],
        );
        assert_eq!(d.object(b).unwrap().parent, None);
    }

    #[test]
    fn operators_needing_a_selection_are_polled_out_when_there_is_none() {
        let (r, _h, d, c) = setup();
        for id in [
            "object.delete",
            "object.duplicate",
            "object.translate",
            "object.parent",
        ] {
            let info = r.get(id).expect(id);
            assert!(!info.poll(&c, &d), "{id} should be unavailable");
        }
        // Adding and select-all always work.
        assert!(r.get("object.add_primitive").unwrap().poll(&c, &d));
        assert!(r.get("object.select_all").unwrap().poll(&c, &d));
    }

    #[test]
    fn a_session_of_operators_reads_back_as_a_script() {
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("shape", PropertyValue::Enum(5))],
        );
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.translate",
            &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
        );
        let script = h.to_script();
        assert!(
            script.contains("kreate.ops.object.add_primitive("),
            "{script}"
        );
        assert!(
            script.contains("kreate.ops.object.translate(translation=(0.0, 0.0, 1.0))"),
            "{script}"
        );
    }

    #[test]
    fn adjusting_the_last_add_changes_the_shape_in_place() {
        // The F9 panel on a real operator: change the primitive after adding it.
        let (r, mut h, mut d, c) = setup();
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.add_primitive",
            &[("shape", PropertyValue::Enum(1))],
        );
        assert_eq!(d.object_count(), 1);
        let cube_faces = d
            .mesh(
                d.object(d.selection.active.unwrap())
                    .unwrap()
                    .mesh()
                    .unwrap(),
            )
            .unwrap()
            .num_faces();

        let mut p = PropertyMap::new();
        p.set("shape", PropertyValue::Enum(5)); // cylinder
        p.set("segments", PropertyValue::Int(8));
        h.adjust_last(&r, &mut d, &c, &p).unwrap();

        assert_eq!(d.object_count(), 1, "adjusting must replace, not add");
        let faces = d
            .mesh(
                d.object(d.selection.active.unwrap())
                    .unwrap()
                    .mesh()
                    .unwrap(),
            )
            .unwrap()
            .num_faces();
        assert_ne!(faces, cube_faces);
        assert_eq!(faces, 10, "8 sides plus two caps");
    }
}
