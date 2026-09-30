//! Editing a skeleton: bones added, extruded, moved and deleted.
//!
//! Blender's armature Edit Mode. What is selected is a bone's two ends — its
//! head and its tail — and a whole bone is both. A connected bone's head *is*
//! its parent's tail, so there is one joint where two bones meet: selecting it
//! selects both, moving it moves both, and every edit ends with
//! [`Armature::reconnect`] putting that true again.
//!
//! The skeleton is edited at rest, in the armature object's space; `G`, `R`
//! and `S` hand over world-space changes, as they do everywhere, and these turn
//! them into that space once.
//!
//! A pose is not touched. A pose is relative to rest, so it rides along on
//! whatever the rest becomes — which is Blender's behaviour and the reason to
//! build a skeleton before posing it.

use kreate_math::{DAffine3, DQuat, DVec3};
use kreate_ops_derive::OpParams;
use kreate_scene::armature::{Armature, Bone};
use kreate_scene::document::ObjectId;
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<SelectPick>();
    registry.register::<SelectAll>();
    registry.register::<Translate>();
    registry.register::<Rotate>();
    registry.register::<Scale>();
    registry.register::<Extrude>();
    registry.register::<Delete>();
    registry.register::<AddBone>();
    registry.register::<SetDisplay>();
    registry.register::<SetBone>();
}

/// The armature object being edited.
pub fn edited_object(doc: &Document) -> Option<ObjectId> {
    let active = doc.selection.active?;
    doc.armature_of(active).map(|_| active)
}

fn in_edit_mode(ctx: &Context, doc: &Document) -> bool {
    ctx.mode.acts_on() == crate::mode::ActsOn::BoneJoints && edited_object(doc).is_some()
}

/// Every selected end, as `(bone, is_tail)`, once each: a connected head is
/// its parent's tail and is listed as that.
fn selected_ends(armature: &Armature) -> Vec<(usize, bool)> {
    let mut out = Vec::new();
    for (i, b) in armature.bones.iter().enumerate() {
        if b.select_tail {
            out.push((i, true));
        }
        if !(b.connected && b.parent.is_some()) && b.select_head {
            out.push((i, false));
        }
    }
    out
}

/// The median of the selected ends, in the world: what `R` and `S` turn and
/// scale about, and what the application measures the pointer from.
pub fn pivot(doc: &Document) -> Option<DVec3> {
    let object = edited_object(doc)?;
    let (_, armature) = doc.armature_of(object)?;
    let ends = selected_ends(armature);
    if ends.is_empty() {
        return None;
    }
    let world = doc.world_transform(object);
    let sum: DVec3 = ends
        .iter()
        .map(|&(i, tail)| world.transform_point3(if tail { armature.bones[i].tail } else { armature.bones[i].head }))
        .sum();
    Some(sum / ends.len() as f64)
}

/// Runs `edit` on the edited armature, then reconnects it.
///
/// A transform that changes nothing still finishes: `G` starts its drag with
/// the identity, and a transform that declined it could never be dragged at
/// all — which is what happened, to every `G`, `R`, `S` and `E` in this mode.
fn edit_armature(
    doc: &mut DocMut<'_>,
    edit: impl FnOnce(&mut Armature, &DAffine3) -> bool,
) -> Result<OpResult, OpError> {
    let Some(object) = edited_object(doc.document()) else { return Ok(OpResult::Cancelled) };
    let Some((id, _)) = doc.document().armature_of(object) else { return Ok(OpResult::Cancelled) };
    let world = doc.document().world_transform(object);
    let Some(armature) = doc.document().armature(id) else { return Ok(OpResult::Cancelled) };
    let mut edited = armature.clone();
    if !edit(&mut edited, &world) {
        return Ok(OpResult::Cancelled);
    }
    edited.reconnect();
    let Some(a) = doc.document_mut().armature_mut(id) else { return Ok(OpResult::Cancelled) };
    *a = edited;
    Ok(OpResult::Finished)
}

/// Moves every selected end by `change`, given in world space.
fn move_ends(doc: &mut DocMut<'_>, change: impl Fn(DVec3) -> DVec3) -> Result<OpResult, OpError> {
    edit_armature(doc, |a, world| {
        if world.matrix3.determinant().abs() < 1e-12 {
            return false;
        }
        let to_local = world.inverse();
        let ends = selected_ends(a);
        if ends.is_empty() {
            return false;
        }
        for (i, tail) in ends {
            let p = if tail { &mut a.bones[i].tail } else { &mut a.bones[i].head };
            *p = to_local.transform_point3(change(world.transform_point3(*p)));
        }
        true
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Part {
    #[default]
    Body,
    Head,
    Tail,
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectPickParams {
    pub bone: String,
    /// The whole bone, or one of its ends.
    #[param(variants("body", "head", "tail"))]
    pub part: u32,
    pub extend: bool,
}

/// Selects a bone, or one end of it.
pub struct SelectPick;

impl Operator for SelectPick {
    type Params = SelectPickParams;
    fn id() -> &'static str {
        "armature.select_pick"
    }
    fn label() -> &'static str {
        "Select Bone Joint"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_edit_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let part = match params.part {
            1 => Part::Head,
            2 => Part::Tail,
            _ => Part::Body,
        };
        let name = params.bone.clone();
        let extend = params.extend;
        edit_armature(doc, move |a, _| {
            let Some(i) = a.bone_index(&name) else { return false };
            let before = a.clone();
            let is_on = match part {
                Part::Body => a.bone_selected(i),
                Part::Head => a.head_selected(i),
                Part::Tail => a.bones[i].select_tail,
            };
            // A plain click selects only this; a Shift-click toggles it.
            let on = !(extend && is_on);
            if !extend {
                for b in &mut a.bones {
                    b.select_head = false;
                    b.select_tail = false;
                }
            }
            if matches!(part, Part::Body | Part::Head) {
                a.set_head_selected(i, on);
            }
            if matches!(part, Part::Body | Part::Tail) {
                a.bones[i].select_tail = on;
            }
            *a != before
        })
        .inspect(|result| {
            // The active bone is one thing in both modes, as in Blender: the
            // one the Bone tab shows.
            if *result == OpResult::Finished {
                if let Some(object) = edited_object(doc.document()) {
                    if let Some(o) = doc.document_mut().object_mut(object) {
                        o.pose.active = Some(params.bone.clone());
                    }
                }
            }
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectAllParams {
    #[param(variants("select", "deselect", "toggle", "invert"))]
    pub action: u32,
}

/// Selects every end, none, or the others.
pub struct SelectAll;

impl Operator for SelectAll {
    type Params = SelectAllParams;
    fn id() -> &'static str {
        "armature.select_all"
    }
    fn label() -> &'static str {
        "Select All Bones"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_edit_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let action = params.action;
        edit_armature(doc, move |a, _| {
            let before = a.clone();
            let any = a.bones.iter().any(|b| b.select_head || b.select_tail);
            for b in &mut a.bones {
                let set = |v: bool| match action {
                    0 => true,
                    1 => false,
                    2 => !any,
                    _ => !v,
                };
                b.select_head = set(b.select_head);
                b.select_tail = set(b.select_tail);
            }
            *a != before
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TranslateParams {
    #[param(unit = "length")]
    pub translation: [f64; 3],
}

/// Moves the selected ends.
pub struct Translate;

impl Operator for Translate {
    type Params = TranslateParams;
    fn id() -> &'static str {
        "armature.translate"
    }
    fn label() -> &'static str {
        "Move Bone Joints"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_edit_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let d = DVec3::from_array(params.translation);
        if !d.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        move_ends(doc, |p| p + d)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RotateParams {
    #[param(unit = "angle")]
    pub angle: f64,
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
}

/// Turns the selected ends about a world axis through their median.
pub struct Rotate;

impl Operator for Rotate {
    type Params = RotateParams;
    fn id() -> &'static str {
        "armature.rotate"
    }
    fn label() -> &'static str {
        "Rotate Bone Joints"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_edit_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        if !params.angle.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let Some(centre) = pivot(doc.document()) else { return Ok(OpResult::Cancelled) };
        let axis = [DVec3::X, DVec3::Y, DVec3::Z][(params.axis as usize).min(2)];
        let q = DQuat::from_axis_angle(axis, params.angle);
        move_ends(doc, |p| centre + q * (p - centre))
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ScaleParams {
    pub scale: [f64; 3],
}

impl Default for ScaleParams {
    fn default() -> Self {
        Self { scale: [1.0; 3] }
    }
}

/// Scales the selected ends about their median, along the world's axes.
pub struct Scale;

impl Operator for Scale {
    type Params = ScaleParams;
    fn id() -> &'static str {
        "armature.scale"
    }
    fn label() -> &'static str {
        "Scale Bone Joints"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_edit_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let s = DVec3::from_array(params.scale);
        if !s.is_finite() || s.min_element().abs() < 1e-9 {
            return Ok(OpResult::Cancelled);
        }
        let Some(centre) = pivot(doc.document()) else { return Ok(OpResult::Cancelled) };
        move_ends(doc, |p| centre + (p - centre) * s)
    }
}

/// Grows a new bone out of every selected tail — Blender's `E`.
///
/// Each new bone is connected to the one it grew from and half its length
/// along the same line, so a cancelled drag after it leaves a bone rather than
/// a point. Only the new tails are selected afterwards, which is what the drag
/// that follows moves.
pub struct Extrude;

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ExtrudeParams {}

impl Operator for Extrude {
    type Params = ExtrudeParams;
    fn id() -> &'static str {
        "armature.extrude"
    }
    fn label() -> &'static str {
        "Extrude Bones"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_edit_mode(ctx, doc)
    }
    fn execute(_params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        edit_armature(doc, |a, _| {
            let from: Vec<usize> = (0..a.bones.len()).filter(|i| a.bones[*i].select_tail).collect();
            if from.is_empty() {
                return false;
            }
            for b in &mut a.bones {
                b.select_head = false;
                b.select_tail = false;
            }
            for i in from {
                let parent = a.bones[i].clone();
                let dir = parent.tail - parent.head;
                let step = if dir.length() > 1e-9 { dir * 0.5 } else { DVec3::Z * 0.5 };
                let name = a.unique_name(&parent.name);
                a.add_bone(Bone {
                    name,
                    parent: Some(i),
                    head: parent.tail,
                    tail: parent.tail + step,
                    roll: parent.roll,
                    connected: true,
                    select_head: false,
                    select_tail: true,
                });
            }
            true
        })
    }
}

/// Deletes the selected bones — those with both ends selected.
pub struct Delete;

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DeleteParams {}

impl Operator for Delete {
    type Params = DeleteParams;
    fn id() -> &'static str {
        "armature.delete"
    }
    fn label() -> &'static str {
        "Delete Bones"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_edit_mode(ctx, doc)
    }
    fn execute(_params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        edit_armature(doc, |a, _| {
            let remove: Vec<bool> = (0..a.bones.len()).map(|i| a.bone_selected(i)).collect();
            // An armature with no bones is not a skeleton; the last one stays.
            if !remove.iter().any(|r| *r) || remove.iter().all(|r| *r) {
                return false;
            }
            a.remove_bones(&remove);
            true
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddBoneParams {
    /// Where its head goes, in the armature's space.
    #[param(unit = "length")]
    pub head: [f64; 3],
}

/// Adds a free bone a unit long, standing up — Blender's Shift+A.
pub struct AddBone;

impl Operator for AddBone {
    type Params = AddBoneParams;
    fn id() -> &'static str {
        "armature.bone_add"
    }
    fn label() -> &'static str {
        "Add Bone"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_edit_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let head = DVec3::from_array(params.head);
        if !head.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        edit_armature(doc, |a, _| {
            for b in &mut a.bones {
                b.select_head = false;
                b.select_tail = false;
            }
            let name = a.unique_name("Bone");
            a.add_bone(Bone {
                name,
                head,
                tail: head + DVec3::Z,
                select_head: true,
                select_tail: true,
                ..Bone::default()
            });
            true
        })
    }
}

// --- the Bone and Armature tabs ---------------------------------------------

fn active_armature(doc: &Document) -> Option<(ObjectId, kreate_scene::ArmatureId)> {
    let object = doc.selection.active?;
    doc.armature_of(object).map(|(id, _)| (object, id))
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetDisplayParams {
    /// Drawn over everything, so a skeleton inside its mesh can be seen.
    pub in_front: bool,
    #[param(variants("octahedral", "stick"))]
    pub display: u32,
}

/// What the Armature tab shows now, as the operator's parameters.
pub fn display_params(armature: &Armature) -> SetDisplayParams {
    SetDisplayParams {
        in_front: armature.in_front,
        display: match armature.display {
            kreate_scene::armature::BoneDisplay::Octahedral => 0,
            kreate_scene::armature::BoneDisplay::Stick => 1,
        },
    }
}

/// Sets how the active armature is drawn.
pub struct SetDisplay;

impl Operator for SetDisplay {
    type Params = SetDisplayParams;
    fn id() -> &'static str {
        "armature.set_display"
    }
    fn label() -> &'static str {
        "Armature Display"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_armature(doc).is_some()
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some((_, id)) = active_armature(doc.document()) else { return Ok(OpResult::Cancelled) };
        let display = if params.display == 1 {
            kreate_scene::armature::BoneDisplay::Stick
        } else {
            kreate_scene::armature::BoneDisplay::Octahedral
        };
        let Some(a) = doc.document().armature(id) else { return Ok(OpResult::Cancelled) };
        if a.in_front == params.in_front && a.display == display {
            return Ok(OpResult::Cancelled);
        }
        let Some(a) = doc.document_mut().armature_mut(id) else { return Ok(OpResult::Cancelled) };
        a.in_front = params.in_front;
        a.display = display;
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetBoneParams {
    /// Which bone, by the name it has now.
    #[param(hidden)]
    pub bone: String,
    pub name: String,
    /// Its parent's name; empty for none.
    pub parent: String,
    /// Its head held on its parent's tail.
    pub connected: bool,
    #[param(unit = "length")]
    pub head: [f64; 3],
    #[param(unit = "length")]
    pub tail: [f64; 3],
    #[param(unit = "angle")]
    pub roll: f64,
}

/// What the Bone tab shows for `bone`, as the operator's parameters.
pub fn bone_params(armature: &Armature, bone: usize) -> Option<SetBoneParams> {
    let b = armature.bones.get(bone)?;
    Some(SetBoneParams {
        bone: b.name.clone(),
        name: b.name.clone(),
        parent: b.parent.map(|p| armature.bones[p].name.clone()).unwrap_or_default(),
        connected: b.connected,
        head: b.head.to_array(),
        tail: b.tail.to_array(),
        roll: b.roll,
    })
}

/// Sets everything about one bone at rest: the Bone tab's fields.
///
/// A rename reaches the pose and the curves that name the bone, as it does in
/// Blender; a new parent that is the bone or below it is refused, because the
/// hierarchy would loop.
pub struct SetBone;

impl Operator for SetBone {
    type Params = SetBoneParams;
    fn id() -> &'static str {
        "armature.bone_set"
    }
    fn label() -> &'static str {
        "Set Bone"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_armature(doc).is_some()
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some((_, id)) = active_armature(doc.document()) else { return Ok(OpResult::Cancelled) };
        let Some(original) = doc.document().armature(id).cloned() else { return Ok(OpResult::Cancelled) };
        let Some(i) = original.bone_index(&params.bone) else { return Ok(OpResult::Cancelled) };
        let renaming = params.name != params.bone;
        if renaming && (params.name.trim().is_empty() || original.bone_index(&params.name).is_some()) {
            return Ok(OpResult::Cancelled);
        }
        let parent = match params.parent.as_str() {
            "" => None,
            name => match original.bone_index(name) {
                Some(p) if !kreate_scene::armature::is_ancestor(&original, i, p) => Some(p),
                _ => return Ok(OpResult::Cancelled),
            },
        };
        let (head, tail) = (DVec3::from_array(params.head), DVec3::from_array(params.tail));
        if !head.is_finite() || !tail.is_finite() || !params.roll.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let mut edited = original.clone();
        {
            let b = &mut edited.bones[i];
            b.parent = parent;
            b.connected = params.connected && parent.is_some();
            b.head = head;
            b.tail = tail;
            b.roll = params.roll;
        }
        if edited.sort_parents_first().is_none() {
            return Ok(OpResult::Cancelled);
        }
        edited.reconnect();
        if edited == original && !renaming {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        if let Some(a) = d.armature_mut(id) {
            *a = edited;
        }
        if renaming {
            d.rename_bone(id, &params.bone, &params.name);
        }
        Ok(OpResult::Finished)
    }
}
