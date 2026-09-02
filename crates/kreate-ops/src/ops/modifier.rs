//! Operators that edit an object's modifier stack.
//!
//! The stack is document state, so every change to it goes through an operator
//! like every other document change — there is no "the panel writes it
//! directly" shortcut. That is what makes a modifier undoable, scriptable and
//! recordable without any of those being thought about here.
//!
//! # One setter per kind, rather than one generic one
//!
//! A single `modifier_set { index, field, value }` would need its own reflection
//! layer over the parameter structs and would give the properties panel nothing
//! to draw. A setter per kind carries that kind's parameters as ordinary
//! reflected fields, so the existing schema-driven property editor draws the
//! panel with the right ranges, units and tooltips, and adding a fourth modifier
//! adds its panel.
//!
//! Every one of these acts on the **active** object, which is the object whose
//! properties the panel is showing. Applying a modifier change to every selected
//! object would be a different operation and a surprising default: the user is
//! looking at one stack.

use kreate_mesh::modifiers::{ArrayParams, MirrorParams, SubdivideParams};
use kreate_ops_derive::OpParams;
use kreate_scene::modifier::{ModifierInstance, ModifierKind};
use kreate_scene::{DocMut, Document, ObjectId};

use crate::operator::{Context, OpError, OpResult, Operator};

/// The modifier kinds, in the order the "add" menu offers them.
pub const KINDS: &[&str] = &["mirror", "subdivide", "array"];

fn kind_of(index: u32) -> ModifierKind {
    match index {
        0 => ModifierKind::Mirror(MirrorParams::default()),
        1 => ModifierKind::Subdivide(SubdivideParams::default()),
        _ => ModifierKind::Array(ArrayParams::default()),
    }
}

/// The object a modifier operator acts on.
///
/// Every operator here polls on this, so the `Cancelled` branch each `execute`
/// has for a missing target is not reached through the registry. It is still not
/// a panic: `Operator::execute` is public, an add-on can call it, and crashing a
/// user's session over a stale selection would be the wrong trade.
fn target(doc: &Document) -> Option<ObjectId> {
    let id = doc.selection.active?;
    let object = doc.object(id)?;
    // A linked datablock may not be edited, but a modifier stack belongs to the
    // *object*, so a linked mesh can still be mirrored locally. It is the object
    // that has to be local.
    object.linkage.is_editable().then_some(id)
}

/// Whether an object has a modifier at `index`.
fn has(doc: &Document, id: ObjectId, index: u32) -> bool {
    doc.object(id)
        .is_some_and(|o| (index as usize) < o.modifiers.len())
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddParams {
    /// Which modifier to add.
    #[param(variants("mirror", "subdivide", "array"))]
    pub kind: u32,
}

/// Appends a modifier to the active object's stack.
pub struct Add;

impl Operator for Add {
    type Params = AddParams;

    fn id() -> &'static str {
        "object.modifier_add"
    }
    fn label() -> &'static str {
        "Add Modifier"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        target(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = target(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        // Appended rather than inserted at the top: the stack reads input at the
        // bottom and output at the top, so a new modifier acts on everything
        // already there, which is what a user adding one to a finished stack
        // means.
        doc.document_mut()
            .object_mut(id)
            .expect("target checked")
            .modifiers
            .push(ModifierInstance::new(kind_of(params.kind)));
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct IndexParams {
    /// Which entry in the stack, counting from the bottom.
    #[param(min = 0, unit = "count")]
    pub index: u32,
}

/// Removes one modifier from the active object's stack.
pub struct Remove;

impl Operator for Remove {
    type Params = IndexParams;

    fn id() -> &'static str {
        "object.modifier_remove"
    }
    fn label() -> &'static str {
        "Remove Modifier"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        target(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = target(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        if !has(doc.document(), id, params.index) {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut()
            .object_mut(id)
            .expect("target checked")
            .modifiers
            .remove(params.index as usize);
        Ok(OpResult::Finished)
    }
}

/// Switches one modifier on or off, keeping its parameters.
pub struct Toggle;

impl Operator for Toggle {
    type Params = IndexParams;

    fn id() -> &'static str {
        "object.modifier_toggle"
    }
    fn label() -> &'static str {
        "Toggle Modifier"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        target(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = target(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        if !has(doc.document(), id, params.index) {
            return Ok(OpResult::Cancelled);
        }
        let object = doc.document_mut().object_mut(id).expect("target checked");
        let entry = &mut object.modifiers[params.index as usize];
        entry.enabled = !entry.enabled;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MoveParams {
    /// Which entry to move, counting from the bottom.
    #[param(min = 0, unit = "count")]
    pub index: u32,
    /// How far to move it. Negative is down the stack, towards the input.
    #[param(min = -64, max = 64, unit = "count")]
    pub offset: i32,
}

/// Reorders the stack.
///
/// The order is the whole meaning of a linear stack — mirroring a subdivision is
/// not subdividing a mirror — so moving an entry is a document change like any
/// other rather than a view preference.
pub struct Move;

impl Operator for Move {
    type Params = MoveParams;

    fn id() -> &'static str {
        "object.modifier_move"
    }
    fn label() -> &'static str {
        "Move Modifier"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        target(doc).is_some_and(|id| doc.object(id).is_some_and(|o| o.modifiers.len() > 1))
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = target(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        if !has(doc.document(), id, params.index) || params.offset == 0 {
            return Ok(OpResult::Cancelled);
        }
        let object = doc.document_mut().object_mut(id).expect("target checked");
        let from = params.index as usize;
        // Clamped rather than refused: a user holding the "move up" button at the
        // top of the stack means "as far as it goes", and an error there would
        // read as the button being broken.
        let to = (from as i64 + params.offset as i64).clamp(0, object.modifiers.len() as i64 - 1)
            as usize;
        if to == from {
            return Ok(OpResult::Cancelled);
        }
        let entry = object.modifiers.remove(from);
        object.modifiers.insert(to, entry);
        Ok(OpResult::Finished)
    }
}

/// Writes a modifier's parameters, if the entry is of the expected kind.
///
/// The kind check is not paranoia: these operators are re-run from the adjust
/// panel and from scripts, where the stack may have changed since. Writing
/// mirror parameters over an array would be a silent replacement of one
/// modifier by another.
fn set_kind(doc: &mut DocMut<'_>, index: u32, kind: ModifierKind) -> Result<OpResult, OpError> {
    let Some(id) = target(doc.document()) else {
        return Ok(OpResult::Cancelled);
    };
    if !has(doc.document(), id, index) {
        return Ok(OpResult::Cancelled);
    }
    let object = doc.document_mut().object_mut(id).expect("target checked");
    let entry = &mut object.modifiers[index as usize];
    if entry.kind.id() != kind.id() {
        return Ok(OpResult::Cancelled);
    }
    entry.kind = kind;
    Ok(OpResult::Finished)
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetMirrorParams {
    #[param(min = 0, unit = "count")]
    pub index: u32,
    /// Reflect across the YZ plane.
    pub x: bool,
    /// Reflect across the XZ plane.
    pub y: bool,
    /// Reflect across the XY plane.
    pub z: bool,
    /// A vertex closer than this to a mirror plane is welded to its reflection.
    #[param(min = 0.0, soft_max = 0.1, unit = "length")]
    pub merge_threshold: f64,
}

impl Default for SetMirrorParams {
    fn default() -> Self {
        let d = MirrorParams::default();
        Self {
            index: 0,
            x: d.x,
            y: d.y,
            z: d.z,
            merge_threshold: d.merge_threshold as f64,
        }
    }
}

/// Sets a mirror modifier's parameters.
pub struct SetMirror;

impl Operator for SetMirror {
    type Params = SetMirrorParams;

    fn id() -> &'static str {
        "object.modifier_set_mirror"
    }
    fn label() -> &'static str {
        "Mirror Modifier"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        target(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        set_kind(
            doc,
            params.index,
            ModifierKind::Mirror(MirrorParams {
                x: params.x,
                y: params.y,
                z: params.z,
                merge_threshold: params.merge_threshold as f32,
            }),
        )
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetSubdivideParams {
    #[param(min = 0, unit = "count")]
    pub index: u32,
    /// Each level splits every edge in half.
    #[param(min = 0, max = 6, unit = "count")]
    pub levels: u32,
}

impl Default for SetSubdivideParams {
    fn default() -> Self {
        Self {
            index: 0,
            levels: SubdivideParams::default().levels,
        }
    }
}

/// Sets a subdivision modifier's parameters.
pub struct SetSubdivide;

impl Operator for SetSubdivide {
    type Params = SetSubdivideParams;

    fn id() -> &'static str {
        "object.modifier_set_subdivide"
    }
    fn label() -> &'static str {
        "Subdivision Modifier"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        target(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        set_kind(
            doc,
            params.index,
            ModifierKind::Subdivide(SubdivideParams {
                levels: params.levels,
            }),
        )
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetArrayParams {
    #[param(min = 0, unit = "count")]
    pub index: u32,
    /// Total copies including the original.
    #[param(min = 1, max = 1000, unit = "count")]
    pub count: u32,
    /// Offset as a fraction of the bounding box, per axis.
    #[param(soft_max = 4.0)]
    pub relative: [f64; 3],
    /// Offset in metres, added to the relative one.
    #[param(unit = "length")]
    pub constant: [f64; 3],
}

impl Default for SetArrayParams {
    fn default() -> Self {
        let d = ArrayParams::default();
        Self {
            index: 0,
            count: d.count,
            relative: [
                d.relative.x as f64,
                d.relative.y as f64,
                d.relative.z as f64,
            ],
            constant: [
                d.constant.x as f64,
                d.constant.y as f64,
                d.constant.z as f64,
            ],
        }
    }
}

/// Sets an array modifier's parameters.
pub struct SetArray;

impl Operator for SetArray {
    type Params = SetArrayParams;

    fn id() -> &'static str {
        "object.modifier_set_array"
    }
    fn label() -> &'static str {
        "Array Modifier"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        target(doc).is_some()
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let v = |a: [f64; 3]| kreate_math::Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32);
        set_kind(
            doc,
            params.index,
            ModifierKind::Array(ArrayParams {
                count: params.count,
                relative: v(params.relative),
                constant: v(params.constant),
            }),
        )
    }
}

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<Add>();
    registry.register::<Remove>();
    registry.register::<Toggle>();
    registry.register::<Move>();
    registry.register::<SetMirror>();
    registry.register::<SetSubdivide>();
    registry.register::<SetArray>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::OperatorRegistry;
    use crate::{PropertyMap, PropertyValue};
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::ModifierCache;

    /// A registry, a history and a document holding one selected cube.
    fn setup() -> (OperatorRegistry, History, Document, ObjectId, Context) {
        let mut r = OperatorRegistry::new();
        register(&mut r);
        let mut doc = Document::new();
        let id = doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        doc.selection.add(id);
        (r, History::new(), doc, id, Context::default())
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

    /// Adds a modifier by its index in [`KINDS`].
    fn add(
        r: &OperatorRegistry,
        h: &mut History,
        d: &mut Document,
        c: &Context,
        kind: u32,
    ) -> OpResult {
        run(
            r,
            h,
            d,
            c,
            "object.modifier_add",
            &[("kind", PropertyValue::Enum(kind))],
        )
    }

    fn kinds(doc: &Document, id: ObjectId) -> Vec<&'static str> {
        doc.object(id)
            .unwrap()
            .modifiers
            .iter()
            .map(|m| m.kind.id())
            .collect()
    }

    #[test]
    fn adding_a_modifier_appends_it() {
        // Appended rather than inserted at the bottom: a modifier added to a
        // finished stack should act on everything already there.
        let (r, mut h, mut d, id, c) = setup();
        add(&r, &mut h, &mut d, &c, 0);
        add(&r, &mut h, &mut d, &c, 2);
        assert_eq!(kinds(&d, id), vec!["mirror", "array"]);
    }

    #[test]
    fn a_modifier_change_is_one_undo_step() {
        // The reason these are operators at all: nothing here had to think about
        // undo, and a modifier the user regrets comes back off in one step.
        let (r, mut h, mut d, id, c) = setup();
        add(&r, &mut h, &mut d, &c, 1);
        assert_eq!(kinds(&d, id), vec!["subdivide"]);

        assert!(h.undo(&mut d), "one step back");
        assert!(d.object(id).unwrap().modifiers.is_empty());
    }

    #[test]
    fn removing_a_modifier_leaves_the_rest_in_order() {
        let (r, mut h, mut d, id, c) = setup();
        for kind in [0u32, 1, 2] {
            add(&r, &mut h, &mut d, &c, kind);
        }
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.modifier_remove",
            &[("index", PropertyValue::Int(1))],
        );
        assert_eq!(kinds(&d, id), vec!["mirror", "array"]);
    }

    #[test]
    fn an_index_past_the_end_is_refused_rather_than_panicking() {
        // These operators are re-run from the adjust panel and from scripts,
        // where the stack may have changed since.
        let (r, mut h, mut d, id, c) = setup();
        add(&r, &mut h, &mut d, &c, 0);
        let result = run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.modifier_remove",
            &[("index", PropertyValue::Int(7))],
        );
        assert_eq!(result, OpResult::Cancelled);
        assert_eq!(kinds(&d, id), vec!["mirror"]);
    }

    #[test]
    fn toggling_keeps_the_parameters() {
        let (r, mut h, mut d, id, c) = setup();
        add(&r, &mut h, &mut d, &c, 2);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.modifier_set_array",
            &[
                ("index", PropertyValue::Int(0)),
                ("count", PropertyValue::Int(5)),
            ],
        );
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.modifier_toggle",
            &[("index", PropertyValue::Int(0))],
        );

        let entry = d.object(id).unwrap().modifiers[0];
        assert!(!entry.enabled);
        assert_eq!(
            entry.kind,
            ModifierKind::Array(ArrayParams {
                count: 5,
                ..Default::default()
            }),
            "toggling off must not lose what the modifier was set to"
        );

        // And a disabled modifier really does nothing.
        let mut cache = ModifierCache::new();
        assert_eq!(cache.evaluated(&d, id).unwrap().num_verts(), 8);
    }

    #[test]
    fn moving_a_modifier_changes_the_result() {
        // The order is the whole meaning of a linear stack, which is why moving
        // an entry is a document change rather than a view preference.
        let (r, mut h, mut d, id, c) = setup();
        add(&r, &mut h, &mut d, &c, 0);
        add(&r, &mut h, &mut d, &c, 2);

        let mut cache = ModifierCache::new();
        let before = cache.evaluated(&d, id).unwrap().bounds();

        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.modifier_move",
            &[
                ("index", PropertyValue::Int(1)),
                ("offset", PropertyValue::Int(-1)),
            ],
        );
        assert_eq!(kinds(&d, id), vec!["array", "mirror"]);

        let after = cache.evaluated(&d, id).unwrap().bounds();
        assert_ne!(before.min.x, after.min.x, "reordering changed nothing");
    }

    #[test]
    fn moving_past_the_end_clamps_rather_than_failing() {
        // A user holding "move up" at the top of the stack means "as far as it
        // goes"; an error there reads as the button being broken.
        let (r, mut h, mut d, id, c) = setup();
        add(&r, &mut h, &mut d, &c, 0);
        add(&r, &mut h, &mut d, &c, 1);
        add(&r, &mut h, &mut d, &c, 2);

        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.modifier_move",
            &[
                ("index", PropertyValue::Int(0)),
                ("offset", PropertyValue::Int(99)),
            ],
        );
        assert_eq!(kinds(&d, id), vec!["subdivide", "array", "mirror"]);
    }

    #[test]
    fn a_setter_refuses_an_entry_of_another_kind() {
        // Writing mirror parameters over an array would silently replace one
        // modifier with another — and these operators are re-run from the adjust
        // panel, where the stack may have moved on.
        let (r, mut h, mut d, id, c) = setup();
        add(&r, &mut h, &mut d, &c, 2);
        let result = run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.modifier_set_mirror",
            &[
                ("index", PropertyValue::Int(0)),
                ("y", PropertyValue::Bool(true)),
            ],
        );
        assert_eq!(result, OpResult::Cancelled);
        assert_eq!(kinds(&d, id), vec!["array"]);
    }

    #[test]
    fn a_setter_writes_through_to_the_geometry() {
        let (r, mut h, mut d, id, c) = setup();
        add(&r, &mut h, &mut d, &c, 1);
        run(
            &r,
            &mut h,
            &mut d,
            &c,
            "object.modifier_set_subdivide",
            &[
                ("index", PropertyValue::Int(0)),
                ("levels", PropertyValue::Int(2)),
            ],
        );

        let mut cache = ModifierCache::new();
        assert_eq!(cache.evaluated(&d, id).unwrap().num_faces(), 96);
    }

    /// Dispatches an add, returning the error rather than unwrapping it.
    ///
    /// `poll` refusing is not a failure of the operator — it is the mechanism by
    /// which a menu entry greys out — so the two are distinguished here.
    fn try_add(
        r: &OperatorRegistry,
        h: &mut History,
        d: &mut Document,
        c: &Context,
    ) -> Result<OpResult, OpError> {
        let mut p = PropertyMap::new();
        p.set("kind", PropertyValue::Enum(0));
        h.dispatch(r, d, c, "object.modifier_add", &p)
    }

    #[test]
    fn nothing_happens_without_an_active_object() {
        // Refused by `poll`, so the operator never runs and nothing reaches the
        // history — which is also what greys the panel out.
        let (r, mut h, mut d, _, c) = setup();
        d.selection.clear();
        d.add_object(Object::new("Empty", ObjectData::Empty));
        assert!(matches!(
            try_add(&r, &mut h, &mut d, &c),
            Err(OpError::PollFailed(_))
        ));
        assert!(!h.can_undo(), "a refused operator must not leave a step");
    }

    #[test]
    fn a_linked_object_refuses_the_change() {
        let (r, mut h, mut d, id, c) = setup();
        d.object_mut(id).unwrap().linkage = kreate_scene::Linkage::Linked {
            library: "lib.kreate".into(),
            name: "Cube".into(),
        };
        assert!(matches!(
            try_add(&r, &mut h, &mut d, &c),
            Err(OpError::PollFailed(_))
        ));
        assert!(d.object(id).unwrap().modifiers.is_empty());
    }

    #[test]
    fn every_operator_is_registered_under_its_own_id() {
        let (r, ..) = setup();
        for id in [
            "object.modifier_add",
            "object.modifier_remove",
            "object.modifier_toggle",
            "object.modifier_move",
            "object.modifier_set_mirror",
            "object.modifier_set_subdivide",
            "object.modifier_set_array",
        ] {
            assert!(r.get(id).is_some(), "{id} is not registered");
        }
    }

    #[test]
    fn the_add_menu_and_the_kind_list_agree() {
        // The variant list drives the dropdown; `kind_of` turns the choice into a
        // modifier. They are two lists and would otherwise drift apart silently.
        assert_eq!(KINDS.len(), 3);
        for (i, name) in KINDS.iter().enumerate() {
            assert_eq!(kind_of(i as u32).id(), *name);
        }
        let schema = <AddParams as kreate_reflect::OpParams>::schema();
        let field = schema
            .iter()
            .find(|d| d.name == "kind")
            .expect("the kind field is reflected");
        match field.kind {
            kreate_reflect::PropertyKind::Enum { variants } => assert_eq!(variants, KINDS),
            other => panic!("expected an enum, got {other:?}"),
        }
    }
}
