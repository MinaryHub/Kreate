//! Keyframes: putting the scene's values into its curves.
//!
//! Blender's `I` and `Alt+I`. Inserting a key reads what the selected objects
//! are now and writes it into their actions at a frame; deleting one takes
//! the keys at a frame away. Both are edits — they undo, they replay — and
//! both carry the frame as a parameter, because the frame the timeline is on
//! is not something a replay can know: a script that says "key at frame 12"
//! has to mean frame 12 whatever the clock says when it runs.

use kreate_ops_derive::OpParams;
use kreate_scene::animation::{self, Action, Interpolation};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<InsertKeyframe>();
    registry.register::<DeleteKeyframe>();
    registry.register::<SelectKeysBox>();
    registry.register::<SelectAllKeys>();
    registry.register::<MoveKeys>();
    registry.register::<DeleteKeys>();
    registry.register::<SetKeyInterpolation>();
    registry.register::<KeyProperty>();
    registry.register::<UnkeyProperty>();
    registry.register::<SelectKeysRegion>();
    registry.register::<SetHandleType>();
    registry.register::<SetKeySlope>();
}

/// Which properties a key is inserted for.
///
/// Blender's keying sets, the four a person reaches for: everything, or one of
/// the three. The order is the `what` parameter's.
pub const KEYING_SETS: [&[&str]; 4] = [
    &["location", "rotation_euler", "scale"],
    &["location"],
    &["rotation_euler"],
    &["scale"],
];

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct InsertKeyframeParams {
    /// Which frame. The application fills in the one the timeline is on.
    pub frame: f64,
    /// What to key.
    #[param(variants("loc_rot_scale", "location", "rotation", "scale"))]
    pub what: u32,
}

impl Default for InsertKeyframeParams {
    fn default() -> Self {
        Self { frame: 1.0, what: 0 }
    }
}

/// Keys the selected objects' transforms at a frame.
///
/// An object with no action gets one, named after it, which is what Blender
/// does and what makes the first `I` on a fresh scene work.
pub struct InsertKeyframe;

impl Operator for InsertKeyframe {
    type Params = InsertKeyframeParams;
    fn id() -> &'static str {
        "anim.keyframe_insert"
    }
    fn label() -> &'static str {
        "Insert Keyframe"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if !params.frame.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        // In pose mode `I` keys the selected bones, as Blender's does — the
        // same key and the same keying sets, with a bone's rotation its
        // quaternion.
        if posing(ctx) {
            let properties = BONE_KEYING_SETS.get(params.what as usize).copied().unwrap_or(BONE_KEYING_SETS[0]);
            let Some(object) = crate::ops::pose::posed_object(doc.document()) else {
                return Ok(OpResult::Cancelled);
            };
            let bones = selected_bones(doc.document(), object);
            let keyed = key_bones(doc.document_mut(), object, &bones, params.frame, properties);
            return Ok(if keyed { OpResult::Finished } else { OpResult::Cancelled });
        }
        let properties = KEYING_SETS
            .get(params.what as usize)
            .copied()
            .unwrap_or(KEYING_SETS[0]);
        let targets = doc.document().selection.objects.clone();
        let keyed = key_objects(doc.document_mut(), &targets, params.frame, properties);
        Ok(if keyed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

/// The bone keying sets, in [`KEYING_SETS`]' order.
pub const BONE_KEYING_SETS: [&[&str]; 4] = [
    &["location", "rotation_quaternion", "scale"],
    &["location"],
    &["rotation_quaternion"],
    &["scale"],
];

fn posing(ctx: &Context) -> bool {
    ctx.mode.acts_on() == crate::mode::ActsOn::Bones
}

/// The names of the posed armature's selected bones, in bone order.
pub fn selected_bones(doc: &Document, object: kreate_scene::ObjectId) -> Vec<String> {
    let (Some((_, armature)), Some(o)) = (doc.armature_of(object), doc.object(object)) else {
        return Vec::new();
    };
    armature.bones.iter().filter(|b| o.pose.is_selected(&b.name)).map(|b| b.name.clone()).collect()
}

/// Keys `properties` of each of `bones` of `object`'s pose at `frame`, on the
/// object's action — made, named after it, if it has none.
///
/// A quaternion is keyed on the side of the sphere nearest the curve's value
/// there: `q` and `-q` are one rotation, and the curves draw a straight line
/// between their numbers — so keying the far one turns the bone the long way
/// round between two keys that look a degree apart.
pub fn key_bones(
    d: &mut Document,
    object: kreate_scene::ObjectId,
    bones: &[String],
    frame: f64,
    properties: &[&str],
) -> bool {
    let Some(o) = d.object(object) else { return false };
    if bones.is_empty() {
        return false;
    }
    let pose = o.pose.clone();
    let action = match o.action.filter(|a| d.action(*a).is_some()) {
        Some(action) => action,
        None => {
            let action = d.add_action(Action::new(format!("{}Action", o.name)));
            if let Some(o) = d.object_mut(object) {
                o.action = Some(action);
            }
            action
        }
    };
    let Some(action) = d.action_mut(action) else { return false };
    for bone in bones {
        let transform = pose.transform(bone);
        for property in properties {
            let Some(mut values) = animation::read_bone(&transform, property) else { continue };
            let path = animation::bone_path(bone, property);
            if *property == "rotation_quaternion" {
                let there: Vec<Option<f64>> = (0..4u8)
                    .map(|i| {
                        action
                            .curves
                            .iter()
                            .find(|c| c.path.property.as_ref() == path && c.path.index == Some(i))
                            .and_then(|c| c.evaluate(frame))
                    })
                    .collect();
                if there.iter().all(Option::is_some) {
                    let dot: f64 = there.iter().zip(&values).map(|(a, b)| a.unwrap_or(0.0) * b).sum();
                    if dot < 0.0 {
                        values.iter_mut().for_each(|v| *v = -*v);
                    }
                }
            }
            for (index, value) in values.into_iter().enumerate() {
                action.curve_mut(&path, index as u8).insert(frame, value, Interpolation::default());
            }
        }
    }
    true
}

/// Keys `properties` of each of `objects` at `frame`, as they stand.
///
/// The one place a key is written from an object's transform: `I` calls it,
/// and so does every object transform when auto-keying is on — so a key put
/// down by a move and a key put down by `I` are the same key. An object with
/// no action gets one named after it.
pub fn key_objects(
    d: &mut Document,
    objects: &[kreate_scene::ObjectId],
    frame: f64,
    properties: &[&str],
) -> bool {
    let mut keyed = false;
    for id in objects {
        let Some(object) = d.object(*id) else { continue };
        let transform = object.transform;
        let action = match object.action.filter(|a| d.action(*a).is_some()) {
            Some(action) => action,
            None => {
                let name = format!("{}Action", object.name);
                let action = d.add_action(Action::new(name));
                if let Some(o) = d.object_mut(*id) {
                    o.action = Some(action);
                }
                action
            }
        };
        let Some(action) = d.action_mut(action) else { continue };
        for property in properties {
            let Some(value) = animation::read_object(&transform, property) else {
                continue;
            };
            for index in 0..3u8 {
                action
                    .curve_mut(property, index)
                    .insert(frame, value[index as usize], Interpolation::default());
            }
        }
        keyed = true;
    }
    keyed
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DeleteKeyframeParams {
    pub frame: f64,
}

impl Default for DeleteKeyframeParams {
    fn default() -> Self {
        Self { frame: 1.0 }
    }
}

/// Takes away every key the selected objects have at a frame.
///
/// A curve left with no keys is removed too, so the property it drove stops
/// being animated rather than being held at a value nobody keyed.
pub struct DeleteKeyframe;

impl Operator for DeleteKeyframe {
    type Params = DeleteKeyframeParams;
    fn id() -> &'static str {
        "anim.keyframe_delete"
    }
    fn label() -> &'static str {
        "Delete Keyframe"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .objects
            .iter()
            .any(|id| doc.object(*id).is_some_and(|o| o.action.is_some()))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        ctx: &Context,
    ) -> Result<OpResult, OpError> {
        // In pose mode, the selected bones' keys and nothing else — Alt+I on
        // a hand must not take the character's placement key with it.
        if posing(ctx) {
            let Some(object) = crate::ops::pose::posed_object(doc.document()) else {
                return Ok(OpResult::Cancelled);
            };
            let bones = selected_bones(doc.document(), object);
            let Some(action) = doc.document().object(object).and_then(|o| o.action) else {
                return Ok(OpResult::Cancelled);
            };
            let ours = |c: &animation::FCurve| {
                animation::parse_bone_path(&c.path.property).is_some_and(|(b, _)| bones.iter().any(|n| n == b))
            };
            let any = doc.document().action(action).is_some_and(|a| {
                a.curves.iter().filter(|c| ours(c)).any(|c| c.keys.iter().any(|k| (k.frame - params.frame).abs() < animation::SAME_FRAME))
            });
            if !any {
                return Ok(OpResult::Cancelled);
            }
            if let Some(a) = doc.document_mut().action_mut(action) {
                for curve in a.curves.iter_mut().filter(|c| ours(c)) {
                    curve.remove(params.frame);
                }
                a.curves.retain(|c| !c.keys.is_empty());
            }
            return Ok(OpResult::Finished);
        }
        let actions: Vec<_> = {
            let d = doc.document();
            // Two selected objects sharing one action delete once.
            let mut seen = std::collections::HashSet::new();
            d.selection
                .objects
                .iter()
                .filter_map(|id| d.object(*id).and_then(|o| o.action))
                .filter(|a| seen.insert(*a))
                .collect()
        };
        // Asked before anything is borrowed mutably, so a frame with nothing
        // on it is a cancel with no step to undo.
        let any = actions.iter().any(|a| {
            doc.document()
                .action(*a)
                .is_some_and(|a| a.curves.iter().any(|c| c.keys.iter().any(|k| (k.frame - params.frame).abs() < animation::SAME_FRAME)))
        });
        if !any {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for action in actions {
            if let Some(action) = d.action_mut(action) {
                for curve in &mut action.curves {
                    curve.remove(params.frame);
                }
                action.curves.retain(|c| !c.keys.is_empty());
            }
        }
        Ok(OpResult::Finished)
    }
}

// --- the dope sheet's edits -------------------------------------------------
//
// These act on "the keys the dope sheet shows": the actions of the selected
// objects, which is what Blender's dope sheet shows with Only Show Selected —
// its default. So they need no list of which actions, and a replay acts on the
// same keys whenever the same objects are selected.

/// The actions the dope sheet shows, once each, in the selection's order:
/// each selected object's own, then its materials' and its light's.
pub fn shown_actions(doc: &Document) -> Vec<kreate_scene::ActionId> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for id in &doc.selection.objects {
        let Some(object) = doc.object(*id) else { continue };
        let mut push = |a: Option<kreate_scene::ActionId>| {
            if let Some(a) = a.filter(|a| doc.action(*a).is_some() && seen.insert(*a)) {
                out.push(a);
            }
        };
        push(object.action);
        if let kreate_scene::document::ObjectData::Mesh(mesh) = object.data {
            for slot in 0..doc.mesh_slots(mesh).len() {
                push(doc.material_at(*id, slot).and_then(|m| m.action));
            }
        }
        if let kreate_scene::document::ObjectData::Light(light) = object.data {
            push(doc.light(light).and_then(|l| l.action));
        }
    }
    out
}

fn any_shown(doc: &Document) -> bool {
    !shown_actions(doc).is_empty()
}

/// One dope-sheet row, as text: an action's identity, a slash, and the
/// property the row is for — empty for the row that stands for them all.
pub fn row_target(action_uid: u64, property: Option<&str>) -> String {
    format!("{action_uid}/{}", escape(property.unwrap_or("")))
}

/// A property as a target writes it: a bone's name can hold a space, which
/// separates targets, or a slash, which separates a target's parts.
fn escape(property: &str) -> String {
    property.replace('%', "%25").replace(' ', "%20").replace('/', "%2F")
}

fn unescape(property: &str) -> String {
    property.replace("%2F", "/").replace("%20", " ").replace("%25", "%")
}

fn parse_targets(text: &str) -> Vec<(u64, Option<String>)> {
    text.split_whitespace()
        .filter_map(|t| {
            let (uid, property) = t.split_once('/')?;
            let uid = uid.parse().ok()?;
            Some((uid, (!property.is_empty()).then(|| unescape(property))))
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectKeysBoxParams {
    /// Which rows, as `row_target` writes them, separated by spaces.
    ///
    /// Text rather than a list of numbers because a row is an action *and* a
    /// property, and because a script line should say what it selects: `12/location`.
    pub targets: String,
    pub frame_min: f64,
    pub frame_max: f64,
    /// Add to the selection rather than replacing it.
    pub extend: bool,
}

/// Selects the keys in some rows between two frames: a click is a box one
/// frame wide on one row.
pub struct SelectKeysBox;

impl Operator for SelectKeysBox {
    type Params = SelectKeysBoxParams;
    fn id() -> &'static str {
        "anim.keys_select_box"
    }
    fn label() -> &'static str {
        "Select Keys"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let shown = shown_actions(doc.document());
        let targets = parse_targets(&params.targets);
        let d = doc.document_mut();
        let mut changed = false;
        if !params.extend {
            for id in &shown {
                if let Some(a) = d.action_mut(*id) {
                    changed |= a.select_all_with(|_| false);
                }
            }
        }
        for (uid, property) in targets {
            // Only the actions on show: a target naming another is a stale
            // script line, and selecting keys nobody can see is a surprise.
            let Some(id) = d.action_by_uid(kreate_scene::Uid::from_raw(uid)).filter(|a| shown.contains(a)) else {
                continue;
            };
            if let Some(a) = d.action_mut(id) {
                changed |= a.select_range(property.as_deref(), params.frame_min, params.frame_max, true);
            }
        }
        Ok(if changed { OpResult::Finished } else { OpResult::Cancelled })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectAllKeysParams {
    #[param(variants("select", "deselect", "invert"))]
    pub action: u32,
}

/// Selects, deselects or inverts every key the dope sheet shows.
pub struct SelectAllKeys;

impl Operator for SelectAllKeys {
    type Params = SelectAllKeysParams;
    fn id() -> &'static str {
        "anim.keys_select_all"
    }
    fn label() -> &'static str {
        "Select All Keys"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let shown = shown_actions(doc.document());
        let d = doc.document_mut();
        let mut changed = false;
        for id in shown {
            if let Some(a) = d.action_mut(id) {
                changed |= match params.action {
                    0 => a.select_all_with(|_| true),
                    1 => a.select_all_with(|_| false),
                    _ => a.select_all_with(|s| !s),
                };
            }
        }
        Ok(if changed { OpResult::Finished } else { OpResult::Cancelled })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MoveKeysParams {
    /// How many frames. Whole in practice — the dope sheet rounds — but a
    /// script may say otherwise.
    pub offset: f64,
    /// How much to add to their values: the graph editor's vertical drag.
    pub value: f64,
}

/// Moves the selected keys along time.
pub struct MoveKeys;

impl Operator for MoveKeys {
    type Params = MoveKeysParams;
    fn id() -> &'static str {
        "anim.keys_move"
    }
    fn label() -> &'static str {
        "Move Keys"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if (params.offset == 0.0 && params.value == 0.0)
            || !params.offset.is_finite()
            || !params.value.is_finite()
        {
            return Ok(OpResult::Cancelled);
        }
        let shown = shown_actions(doc.document());
        if !shown.iter().any(|a| doc.document().action(*a).is_some_and(|a| a.any_selected())) {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in shown {
            if let Some(a) = d.action_mut(id) {
                if params.offset != 0.0 {
                    a.move_selected(params.offset);
                }
                if params.value != 0.0 {
                    a.offset_selected_values(params.value);
                }
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DeleteKeysParams {}

/// Deletes the selected keys.
pub struct DeleteKeys;

impl Operator for DeleteKeys {
    type Params = DeleteKeysParams;
    fn id() -> &'static str {
        "anim.keys_delete"
    }
    fn label() -> &'static str {
        "Delete Keys"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let shown = shown_actions(doc.document());
        if !shown.iter().any(|a| doc.document().action(*a).is_some_and(|a| a.any_selected())) {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in shown {
            if let Some(a) = d.action_mut(id) {
                a.delete_selected();
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetKeyInterpolationParams {
    #[param(variants("constant", "linear", "bezier"))]
    pub interpolation: u32,
}

/// Sets how the selected keys travel to the next: Blender's `T`.
pub struct SetKeyInterpolation;

impl Operator for SetKeyInterpolation {
    type Params = SetKeyInterpolationParams;
    fn id() -> &'static str {
        "anim.keys_interpolation"
    }
    fn label() -> &'static str {
        "Set Interpolation"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(kind) = Interpolation::ALL.get(params.interpolation as usize).copied() else {
            return Ok(OpResult::Cancelled);
        };
        let shown = shown_actions(doc.document());
        let any = shown.iter().any(|a| {
            doc.document().action(*a).is_some_and(|a| {
                a.curves.iter().any(|c| c.keys.iter().any(|k| k.selected && k.interpolation != kind))
            })
        });
        if !any {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in shown {
            if let Some(a) = d.action_mut(id) {
                a.set_selected_interpolation(kind);
            }
        }
        Ok(OpResult::Finished)
    }
}

// --- keying one property of a material or a light ---------------------------

/// Which datablock a property key is for, in `KeyProperty::datablock`'s order.
fn datablock_of(which: u32) -> kreate_scene::DatablockKind {
    match which {
        0 => kreate_scene::DatablockKind::Material,
        2 => kreate_scene::DatablockKind::Object,
        _ => kreate_scene::DatablockKind::Light,
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct KeyPropertyParams {
    pub frame: f64,
    /// A material of the active object, or its light.
    #[param(variants("material", "light", "object"))]
    pub datablock: u32,
    /// Which of the active object's material slots, for a material.
    #[param(min = 0.0)]
    pub slot: u32,
    /// The property, by the name `animation::animatable` lists.
    pub property: String,
}

impl Default for KeyPropertyParams {
    fn default() -> Self {
        Self { frame: 1.0, datablock: 0, slot: 0, property: String::new() }
    }
}

/// Keys one property of the active object's material or light, as it stands —
/// the diamond beside a value in the properties editor, Blender's `I` over a
/// field.
pub struct KeyProperty;

/// Reads the value and makes sure an action is there to take it: the
/// datablock's own, made and linked if it has none. `None` when there is no
/// such datablock or no such property.
fn property_action(
    d: &mut Document,
    params_datablock: u32,
    slot: u32,
    property: &str,
) -> Option<(kreate_scene::ActionId, Vec<f64>)> {
    let kind = datablock_of(params_datablock);
    let object = d.selection.active?;
    match kind {
        // The active object's own transform: a diamond in the Object tab.
        kreate_scene::DatablockKind::Object => {
            let o = d.object(object)?;
            let value = animation::read_object(&o.transform, property)?.to_array().to_vec();
            let action = match o.action.filter(|a| d.action(*a).is_some()) {
                Some(a) => a,
                None => {
                    let a = d.add_action(Action::new(format!("{}Action", o.name)));
                    d.object_mut(object)?.action = Some(a);
                    a
                }
            };
            Some((action, value))
        }
        kreate_scene::DatablockKind::Material => {
            let id = d.material_id_at(object, slot as usize)?;
            let m = d.material(id)?;
            let value = animation::read_material(m, property)?;
            let action = match m.action.filter(|a| d.action(*a).is_some()) {
                Some(a) => a,
                None => {
                    let a = d.add_action(Action::for_kind(format!("{}Action", m.name), kind));
                    d.material_mut(id)?.action = Some(a);
                    a
                }
            };
            Some((action, value))
        }
        _ => {
            let kreate_scene::document::ObjectData::Light(id) = d.object(object)?.data else {
                return None;
            };
            let l = d.light(id)?;
            let value = animation::read_light(l, property)?;
            let action = match l.action.filter(|a| d.action(*a).is_some()) {
                Some(a) => a,
                None => {
                    let a = d.add_action(Action::for_kind(format!("{}Action", l.name), kind));
                    d.light_mut(id)?.action = Some(a);
                    a
                }
            };
            Some((action, value))
        }
    }
}

impl Operator for KeyProperty {
    type Params = KeyPropertyParams;
    fn id() -> &'static str {
        "anim.keyframe_insert_property"
    }
    fn label() -> &'static str {
        "Insert Keyframe on Property"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection.active.is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let kind = datablock_of(params.datablock);
        // Checked against what the kind declares before anything is made: a
        // misspelt property must not leave an empty action behind.
        if !params.frame.is_finite()
            || !animation::animatable(kind).iter().any(|d| d.name == params.property)
        {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        let Some((action, value)) = property_action(d, params.datablock, params.slot, &params.property) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(action) = d.action_mut(action) else {
            return Ok(OpResult::Cancelled);
        };
        for (index, v) in value.iter().enumerate() {
            action
                .curve_mut(&params.property, index as u8)
                .insert(params.frame, *v, Interpolation::default());
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct UnkeyPropertyParams {
    pub frame: f64,
    #[param(variants("material", "light", "object"))]
    pub datablock: u32,
    #[param(min = 0.0)]
    pub slot: u32,
    pub property: String,
}

impl Default for UnkeyPropertyParams {
    fn default() -> Self {
        Self { frame: 1.0, datablock: 0, slot: 0, property: String::new() }
    }
}

/// Takes away one property's key at a frame: the diamond, clicked on a
/// property that is keyed here.
pub struct UnkeyProperty;

impl Operator for UnkeyProperty {
    type Params = UnkeyPropertyParams;
    fn id() -> &'static str {
        "anim.keyframe_delete_property"
    }
    fn label() -> &'static str {
        "Delete Keyframe on Property"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection.active.is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let Some(object) = d.selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let action = match datablock_of(params.datablock) {
            kreate_scene::DatablockKind::Material => {
                d.material_at(object, params.slot as usize).and_then(|m| m.action)
            }
            kreate_scene::DatablockKind::Object => d.object(object).and_then(|o| o.action),
            _ => match d.object(object).map(|o| &o.data) {
                Some(kreate_scene::document::ObjectData::Light(l)) => d.light(*l).and_then(|l| l.action),
                _ => None,
            },
        };
        let Some(action) = action else {
            return Ok(OpResult::Cancelled);
        };
        let has = d.action(action).is_some_and(|a| {
            a.curves.iter().any(|c| {
                c.path.property.as_ref() == params.property
                    && c.keys.iter().any(|k| (k.frame - params.frame).abs() < animation::SAME_FRAME)
            })
        });
        if !has {
            return Ok(OpResult::Cancelled);
        }
        if let Some(a) = doc.document_mut().action_mut(action) {
            for curve in a.curves.iter_mut().filter(|c| c.path.property.as_ref() == params.property) {
                curve.remove(params.frame);
            }
            a.curves.retain(|c| !c.keys.is_empty());
        }
        Ok(OpResult::Finished)
    }
}

// --- the graph editor's edits -------------------------------------------------

/// One curve, as text: an action's identity, the property and the component —
/// `12/location/0`.
pub fn curve_target(action_uid: u64, property: &str, index: u8) -> String {
    format!("{action_uid}/{}/{index}", escape(property))
}

pub(crate) fn parse_curve(text: &str) -> Option<(u64, String, u8)> {
    let mut parts = text.split('/');
    let uid = parts.next()?.parse().ok()?;
    let property = unescape(parts.next()?);
    let index = parts.next()?.parse().ok()?;
    Some((uid, property, index))
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectKeysRegionParams {
    /// Which curves, as `curve_target` writes them, separated by spaces.
    pub targets: String,
    pub frame_min: f64,
    pub frame_max: f64,
    pub value_min: f64,
    pub value_max: f64,
    pub extend: bool,
}

/// Selects the keys inside a box on the graph: a frame range and a value
/// range, on some curves. A click is a small box around one key.
pub struct SelectKeysRegion;

impl Operator for SelectKeysRegion {
    type Params = SelectKeysRegionParams;
    fn id() -> &'static str {
        "anim.keys_select_region"
    }
    fn label() -> &'static str {
        "Select Keys in Region"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let shown = shown_actions(doc.document());
        let d = doc.document_mut();
        let mut changed = false;
        if !params.extend {
            for id in &shown {
                if let Some(a) = d.action_mut(*id) {
                    changed |= a.select_all_with(|_| false);
                }
            }
        }
        for (uid, property, index) in params.targets.split_whitespace().filter_map(parse_curve) {
            let Some(id) = d.action_by_uid(kreate_scene::Uid::from_raw(uid)).filter(|a| shown.contains(a)) else {
                continue;
            };
            if let Some(a) = d.action_mut(id) {
                changed |= a.select_region(
                    &property,
                    index,
                    (params.frame_min, params.frame_max),
                    (params.value_min, params.value_max),
                );
            }
        }
        Ok(if changed { OpResult::Finished } else { OpResult::Cancelled })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetHandleTypeParams {
    /// Auto-clamped, or flat — Blender's `V` menu, the two a person reaches for.
    #[param(variants("auto", "flat"))]
    pub kind: u32,
}

/// Sets how the selected keys' handles are made.
pub struct SetHandleType;

impl Operator for SetHandleType {
    type Params = SetHandleTypeParams;
    fn id() -> &'static str {
        "anim.keys_handle_type"
    }
    fn label() -> &'static str {
        "Set Handle Type"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let slope = if params.kind == 1 { Some(0.0) } else { None };
        let shown = shown_actions(doc.document());
        let any = shown.iter().any(|a| {
            doc.document()
                .action(*a)
                .is_some_and(|a| a.curves.iter().any(|c| c.keys.iter().any(|k| k.selected && k.slope != slope)))
        });
        if !any {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in shown {
            if let Some(a) = d.action_mut(id) {
                a.set_selected_slope(slope);
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetKeySlopeParams {
    /// The curve, as `curve_target` writes it.
    pub target: String,
    /// Which key on it.
    pub frame: f64,
    /// Value per frame.
    pub slope: f64,
}

/// Sets one key's slope: what dragging its handle in the graph editor means.
pub struct SetKeySlope;

impl Operator for SetKeySlope {
    type Params = SetKeySlopeParams;
    fn id() -> &'static str {
        "anim.key_set_slope"
    }
    fn label() -> &'static str {
        "Set Key Slope"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((uid, property, index)) = parse_curve(&params.target) else {
            return Ok(OpResult::Cancelled);
        };
        if !params.slope.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let shown = shown_actions(doc.document());
        let Some(id) = doc
            .document()
            .action_by_uid(kreate_scene::Uid::from_raw(uid))
            .filter(|a| shown.contains(a))
        else {
            return Ok(OpResult::Cancelled);
        };
        let Some(a) = doc.document_mut().action_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(key) = a
            .curves
            .iter_mut()
            .filter(|c| c.path.property.as_ref() == property && c.path.index == Some(index))
            .flat_map(|c| c.keys.iter_mut())
            .find(|k| (k.frame - params.frame).abs() < animation::SAME_FRAME)
        else {
            return Ok(OpResult::Cancelled);
        };
        if key.slope == Some(params.slope) {
            return Ok(OpResult::Cancelled);
        }
        key.slope = Some(params.slope);
        Ok(OpResult::Finished)
    }
}

