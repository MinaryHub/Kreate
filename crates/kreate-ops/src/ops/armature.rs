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
    registry.register::<Symmetrize>();
    registry.register::<AddHumanoid>();
    registry.register::<Bind>();
    registry.register::<Unbind>();
    registry.register::<SetPreserveVolume>();
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
        .map(|&(i, tail)| {
            world.transform_point3(if tail {
                armature.bones[i].tail
            } else {
                armature.bones[i].head
            })
        })
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
    let Some(object) = edited_object(doc.document()) else {
        return Ok(OpResult::Cancelled);
    };
    let Some((id, _)) = doc.document().armature_of(object) else {
        return Ok(OpResult::Cancelled);
    };
    let world = doc.document().world_transform(object);
    let Some(armature) = doc.document().armature(id) else {
        return Ok(OpResult::Cancelled);
    };
    let mut edited = armature.clone();
    if !edit(&mut edited, &world) {
        return Ok(OpResult::Cancelled);
    }
    edited.reconnect();
    let Some(a) = doc.document_mut().armature_mut(id) else {
        return Ok(OpResult::Cancelled);
    };
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
            let p = if tail {
                &mut a.bones[i].tail
            } else {
                &mut a.bones[i].head
            };
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let part = match params.part {
            1 => Part::Head,
            2 => Part::Tail,
            _ => Part::Body,
        };
        let name = params.bone.clone();
        let extend = params.extend;
        edit_armature(doc, move |a, _| {
            let Some(i) = a.bone_index(&name) else {
                return false;
            };
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if !params.angle.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let Some(centre) = pivot(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let s = DVec3::from_array(params.scale);
        if !s.is_finite() || s.min_element().abs() < 1e-9 {
            return Ok(OpResult::Cancelled);
        }
        let Some(centre) = pivot(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
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
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        edit_armature(doc, |a, _| {
            let from: Vec<usize> = (0..a.bones.len())
                .filter(|i| a.bones[*i].select_tail)
                .collect();
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
                let step = if dir.length() > 1e-9 {
                    dir * 0.5
                } else {
                    DVec3::Z * 0.5
                };
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
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((_, id)) = active_armature(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let display = if params.display == 1 {
            kreate_scene::armature::BoneDisplay::Stick
        } else {
            kreate_scene::armature::BoneDisplay::Octahedral
        };
        let Some(a) = doc.document().armature(id) else {
            return Ok(OpResult::Cancelled);
        };
        if a.in_front == params.in_front && a.display == display {
            return Ok(OpResult::Cancelled);
        }
        let Some(a) = doc.document_mut().armature_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
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
        parent: b
            .parent
            .map(|p| armature.bones[p].name.clone())
            .unwrap_or_default(),
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((_, id)) = active_armature(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(original) = doc.document().armature(id).cloned() else {
            return Ok(OpResult::Cancelled);
        };
        let Some(i) = original.bone_index(&params.bone) else {
            return Ok(OpResult::Cancelled);
        };
        let renaming = params.name != params.bone;
        if renaming
            && (params.name.trim().is_empty() || original.bone_index(&params.name).is_some())
        {
            return Ok(OpResult::Cancelled);
        }
        let parent = match params.parent.as_str() {
            "" => None,
            name => match original.bone_index(name) {
                Some(p) if !kreate_scene::armature::is_ancestor(&original, i, p) => Some(p),
                _ => return Ok(OpResult::Cancelled),
            },
        };
        let (head, tail) = (
            DVec3::from_array(params.head),
            DVec3::from_array(params.tail),
        );
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

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SymmetrizeParams {
    /// Which side is copied: the bones on +X (Blender's left, `.L`) or on -X.
    #[param(variants("positive_x", "negative_x"))]
    pub from: u32,
}

/// Copies every sided bone — `.L`, `_l`, `Left…` — across X to the other
/// side's name, making it or moving it there: Blender's Symmetrize. A
/// mirrored bone's parent is its parent's mirror when there is one, the
/// same parent otherwise; its roll turns the other way.
pub struct Symmetrize;

impl Operator for Symmetrize {
    type Params = SymmetrizeParams;
    fn id() -> &'static str {
        "armature.symmetrize"
    }
    fn label() -> &'static str {
        "Symmetrize Bones"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_armature(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_scene::armature::mirrored_name;
        let Some((_, id)) = active_armature(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(original) = doc.document().armature(id).cloned() else {
            return Ok(OpResult::Cancelled);
        };
        let sign = if params.from == 1 { -1.0 } else { 1.0 };
        let flip = |p: DVec3| DVec3::new(-p.x, p.y, p.z);
        let mut edited = original.clone();
        let mut changed = 0;
        // Parents first, so a mirrored parent is there for its child.
        for i in 0..original.bones.len() {
            let bone = original.bones[i].clone();
            let Some(other) = mirrored_name(&bone.name) else {
                continue;
            };
            if (bone.head.x + bone.tail.x) * sign <= 0.0 {
                continue;
            }
            let parent = bone.parent.map(|p| {
                let name = &edited.bones[p].name;
                mirrored_name(name)
                    .and_then(|m| edited.bone_index(&m))
                    .unwrap_or(p)
            });
            let mirrored = Bone {
                name: other.clone(),
                parent,
                head: flip(bone.head),
                tail: flip(bone.tail),
                roll: -bone.roll,
                connected: bone.connected && parent.is_some(),
                ..Default::default()
            };
            match edited.bone_index(&other) {
                Some(j) => {
                    // Its parent may not be an earlier bone after all.
                    let keep = edited.bones[j].parent;
                    let b = &mut edited.bones[j];
                    b.head = mirrored.head;
                    b.tail = mirrored.tail;
                    b.roll = mirrored.roll;
                    b.parent = match mirrored.parent {
                        Some(p) if p < j => Some(p),
                        _ => keep,
                    };
                    b.connected = mirrored.connected && b.parent.is_some();
                }
                None => {
                    edited.add_bone(mirrored);
                }
            }
            changed += 1;
        }
        if changed == 0 {
            return Err(OpError::Failed(
                "no bone on that side has a sided name (.L, _l, Left…)".into(),
            ));
        }
        if edited.sort_parents_first().is_none() {
            return Ok(OpResult::Cancelled);
        }
        edited.reconnect();
        if edited == original {
            return Ok(OpResult::Cancelled);
        }
        if let Some(a) = doc.document_mut().armature_mut(id) {
            *a = edited;
        }
        Ok(OpResult::Finished)
    }
}

// --- binding a mesh to the skeleton -------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct BindParams {
    /// With weights by heat spreading along the surface (Blender's
    /// automatic), with the mesh's own — empty groups made for the bones
    /// that have none — or by distance alone, the quicker envelope.
    #[param(variants("automatic", "empty", "envelope"))]
    pub weights: u32,
    /// Automatic weights only from bones a vertex can see — not through
    /// skin: for a figure whose arms hang beside its body.
    pub visibility: bool,
}

/// Parents the selected meshes to the active armature and makes them follow
/// its bones — Blender's Ctrl+P, With Automatic Weights or With Empty Groups.
///
/// One step: the parent, the binding and the weights undo together, as they
/// were one choice.
pub struct Bind;

impl Operator for Bind {
    type Params = BindParams;
    fn id() -> &'static str {
        "object.bind_armature"
    }
    fn label() -> &'static str {
        "Parent with Armature Deform"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_armature(doc).is_some() && !meshes_to_bind(doc).is_empty()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some((rig, armature_id)) = active_armature(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let meshes = meshes_to_bind(doc.document());
        if meshes.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let Some(rig_uid) = doc.document().object_uid(rig) else {
            return Ok(OpResult::Cancelled);
        };
        let d = doc.document_mut();
        for object in meshes {
            // Refused only for a loop, which a mesh under an armature cannot
            // be — but the object is left unbound then rather than bound to a
            // rig it is not under.
            if d.set_parent(object, Some(rig)).is_err() {
                continue;
            }
            let Some(mesh_id) = d.object(object).and_then(|o| o.mesh()) else {
                continue;
            };
            let to_armature = d.world_transform(rig).inverse() * d.world_transform(object);
            let (Some(armature), Some(mesh)) = (d.armature(armature_id).cloned(), d.mesh(mesh_id))
            else {
                continue;
            };
            let layers: Vec<(String, Vec<f32>)> = if params.weights == 0 {
                kreate_scene::skin::heat_weights(mesh, &armature, to_armature, params.visibility)
            } else if params.weights == 2 {
                kreate_scene::skin::automatic_weights(mesh, &armature, to_armature)
            } else {
                armature
                    .bones
                    .iter()
                    .map(|b| kreate_scene::skin::weight_layer(&b.name))
                    .filter(|name| !mesh.attributes().contains(name))
                    .map(|name| (name, vec![0.0; mesh.num_verts() as usize]))
                    .collect()
            };
            if let Some(mesh) = d.mesh_mut(mesh_id) {
                for (name, weights) in layers {
                    mesh.attributes_mut().insert(
                        &name,
                        kreate_mesh::attribute::Domain::Point,
                        kreate_mesh::attribute::AttrData::F32(std::sync::Arc::new(weights)),
                    );
                }
            }
            if let Some(o) = d.object_mut(object) {
                o.skin = Some(rig_uid);
            }
        }
        Ok(OpResult::Finished)
    }
}

/// The selected mesh objects other than the active one.
fn meshes_to_bind(doc: &Document) -> Vec<ObjectId> {
    doc.selection
        .objects
        .iter()
        .copied()
        .filter(|id| {
            Some(*id) != doc.selection.active && doc.object(*id).is_some_and(|o| o.mesh().is_some())
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct PreserveVolumeParams {
    pub preserve_volume: bool,
}

/// Blender's Preserve Volume, on the selected bound meshes: dual-quaternion
/// skinning, which keeps a twisted joint round.
pub struct SetPreserveVolume;

impl Operator for SetPreserveVolume {
    type Params = PreserveVolumeParams;
    fn id() -> &'static str {
        "object.set_preserve_volume"
    }
    fn label() -> &'static str {
        "Preserve Volume"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .objects
            .iter()
            .any(|id| doc.object(*id).is_some_and(|o| o.skin.is_some()))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let changing: Vec<ObjectId> = doc
            .document()
            .selection
            .objects
            .iter()
            .copied()
            .filter(|id| {
                doc.document().object(*id).is_some_and(|o| {
                    o.skin.is_some() && o.preserve_volume != params.preserve_volume
                })
            })
            .collect();
        if changing.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in changing {
            if let Some(o) = d.object_mut(id) {
                o.preserve_volume = params.preserve_volume;
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct UnbindParams {}

/// Stops the selected meshes following their armature. Their weights stay:
/// binding again gives the same deformation back, as removing and re-adding
/// Blender's modifier does.
pub struct Unbind;

impl Operator for Unbind {
    type Params = UnbindParams;
    fn id() -> &'static str {
        "object.unbind_armature"
    }
    fn label() -> &'static str {
        "Remove Armature Deform"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .objects
            .iter()
            .any(|id| doc.object(*id).is_some_and(|o| o.skin.is_some()))
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let bound: Vec<ObjectId> = doc
            .document()
            .selection
            .objects
            .iter()
            .copied()
            .filter(|id| doc.document().object(*id).is_some_and(|o| o.skin.is_some()))
            .collect();
        if bound.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in bound {
            if let Some(o) = d.object_mut(id) {
                o.skin = None;
            }
        }
        Ok(OpResult::Finished)
    }
}

// --- a humanoid rig fitted to a mesh -----------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct HumanoidParams {
    /// Bind the mesh to it at once, with automatic weights.
    pub bind: bool,
    /// Fingers too: a thumb and four fingers a hand, three bones each,
    /// spread across the hand as it is measured. For a figure with open
    /// hands; a mitten or a fist is better left one bone.
    pub fingers: bool,
}

impl Default for HumanoidParams {
    fn default() -> Self {
        Self {
            bind: true,
            fingers: false,
        }
    }
}

/// A left hand's fingers, as [`humanoid_bones`] lays bones out: from where
/// the hand's points are past the wrist — its length along the arm and its
/// width across, the thumb on the front (-Y) side, as a figure stands with
/// its palms down.
pub fn finger_bones(points: &[DVec3], l: &Landmarks) -> Vec<(String, DVec3, DVec3, String, bool)> {
    let along = (l.fingertip - l.wrist).normalize_or(DVec3::X);
    let length = (l.fingertip - l.wrist).length();
    // Across the hand, level: the front of the figure is -Y.
    let mut across = along.cross(DVec3::Z).normalize_or(DVec3::NEG_Y);
    if across.y > 0.0 {
        across = -across;
    }
    let hand: Vec<&DVec3> = points
        .iter()
        .filter(|p| {
            let t = (**p - l.wrist).dot(along);
            t > 0.0 && t <= length * 1.02 && p.x > 0.0
        })
        .collect();
    let spread: Vec<f64> = hand.iter().map(|p| (**p - l.wrist).dot(across)).collect();
    let (lo, hi) = (
        spread.iter().copied().fold(f64::MAX, f64::min),
        spread.iter().copied().fold(f64::MIN, f64::max),
    );
    let (lo, hi) = if lo < hi {
        (lo, hi)
    } else {
        (-length * 0.25, length * 0.25)
    };
    let width = hi - lo;
    let middle = (lo + hi) / 2.0;
    let knuckle = l.wrist + along * length * 0.45;
    let mut out = Vec::new();
    let mut chain = |name: &str, start: DVec3, end: DVec3, parent: &str| {
        let joints = [0.0, 0.45, 0.75, 1.0].map(|t| start + (end - start) * t);
        for k in 0..3 {
            out.push((
                format!("{name}{}", k + 1),
                joints[k],
                joints[k + 1],
                if k == 0 {
                    parent.to_string()
                } else {
                    format!("{name}{k}")
                },
                k > 0,
            ));
        }
    };
    // Index nearest the thumb, at the front; pinky at the back.
    for (i, (name, reach)) in [
        ("Index", 0.92),
        ("Middle", 1.0),
        ("Ring", 0.95),
        ("Pinky", 0.8),
    ]
    .iter()
    .enumerate()
    {
        // `across` points to the front: the index furthest along it.
        let offset = across * (middle + width * (1.5 - i as f64) * 0.22 - width * 0.1);
        let start = knuckle + offset;
        let end = start + along * length * 0.55 * reach;
        chain(name, start, end, "Hand");
    }
    let thumb_root = l.wrist + along * length * 0.1 + across * (middle + width * 0.3);
    let thumb_tip = knuckle + along * length * 0.2 + across * (middle + width * 0.75);
    chain("Thumb", thumb_root, thumb_tip, "Hand");
    out
}

/// Where a standing figure's joints are, in the world: measured where a
/// measure is sure — its height, where its legs are, how far its arms reach,
/// which way its feet point — and by a figure's proportions elsewhere.
///
/// A start, not an answer: the joints are where most figures have them, and
/// `check_rig` says where this one differs.
#[derive(Clone, Debug, PartialEq)]
pub struct Landmarks {
    pub hips: DVec3,
    pub spine: DVec3,
    pub chest: DVec3,
    pub neck: DVec3,
    pub head: DVec3,
    pub top: DVec3,
    /// The left side's (+X); the right is its mirror.
    pub clavicle: DVec3,
    pub shoulder: DVec3,
    pub elbow: DVec3,
    pub wrist: DVec3,
    pub fingertip: DVec3,
    pub hip: DVec3,
    pub knee: DVec3,
    pub ankle: DVec3,
    pub toe: DVec3,
}

/// The landmarks of a figure standing up Z, facing -Y, its middle on X = 0.
pub fn humanoid_landmarks(points: &[DVec3]) -> Option<Landmarks> {
    let (lo, hi) = points.iter().fold(
        (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let height = hi.z - lo.z;
    if points.len() < 8 || height <= 0.0 {
        return None;
    }
    let at = |f: f64| lo.z + height * f;
    // The middle of a spread of numbers: halfway between its ends, which a
    // limb's outline gives whichever side has more points.
    let middle = |v: Vec<f64>| -> Option<f64> {
        let lo = v.iter().copied().reduce(f64::min)?;
        let hi = v.iter().copied().reduce(f64::max)?;
        Some((lo + hi) / 2.0)
    };
    let band = |from: f64, to: f64| {
        points
            .iter()
            .filter(move |p| p.z >= at(from) && p.z <= at(to))
    };
    // The body's front-to-back middle, from the chest.
    let mid_y = middle(band(0.6, 0.75).map(|p| p.y).collect()).unwrap_or((lo.y + hi.y) / 2.0);
    // A leg's middle, from just below the knee: where its points are on +X.
    let leg_x = middle(band(0.2, 0.3).filter(|p| p.x > 0.0).map(|p| p.x).collect())
        .unwrap_or(height * 0.05)
        .max(height * 0.02);
    // The arm's reach: the farthest point on +X above the hips.
    let tip = band(0.45, 1.0)
        .filter(|p| p.x > 0.0)
        .copied()
        .max_by(|a, b| a.x.total_cmp(&b.x))?;
    let shoulder = DVec3::new((height * 0.1).min(tip.x * 0.5), mid_y, at(0.81));
    let toward = (tip - shoulder).normalize_or(DVec3::X);
    let reach = (tip - shoulder).length();
    // Toes: the frontmost point low on the left foot.
    let toe_y = band(0.0, 0.06)
        .filter(|p| p.x > 0.0)
        .map(|p| p.y)
        .fold(f64::MAX, f64::min);
    let toe_y = if toe_y.is_finite() {
        toe_y
    } else {
        mid_y - height * 0.08
    };
    let column = |f: f64| DVec3::new(0.0, mid_y, at(f));
    Some(Landmarks {
        hips: column(0.50),
        spine: column(0.58),
        chest: column(0.68),
        neck: column(0.82),
        head: column(0.87),
        top: column(1.0),
        clavicle: DVec3::new(height * 0.02, mid_y, at(0.80)),
        shoulder,
        elbow: shoulder + toward * reach * 0.42,
        wrist: shoulder + toward * reach * 0.80,
        fingertip: tip,
        hip: DVec3::new(leg_x, mid_y, at(0.49)),
        knee: DVec3::new(leg_x, mid_y, at(0.27)),
        ankle: DVec3::new(leg_x, mid_y, at(0.045)),
        toe: DVec3::new(leg_x, toe_y, at(0.015)),
    })
}

/// The bones of a humanoid on `l`: name, head, tail, parent's name, and
/// whether its head is its parent's tail. The left side's, then their
/// mirrors.
pub fn humanoid_bones(l: &Landmarks) -> Vec<(String, DVec3, DVec3, Option<String>, bool)> {
    let flip = |p: DVec3| DVec3::new(-p.x, p.y, p.z);
    let mut bones: Vec<(String, DVec3, DVec3, Option<String>, bool)> = vec![
        ("Hips".into(), l.hips, l.spine, None, false),
        ("Spine".into(), l.spine, l.chest, Some("Hips".into()), true),
        ("Chest".into(), l.chest, l.neck, Some("Spine".into()), true),
        ("Neck".into(), l.neck, l.head, Some("Chest".into()), true),
        ("Head".into(), l.head, l.top, Some("Neck".into()), true),
    ];
    let side: Vec<(&str, DVec3, DVec3, &str, bool)> = vec![
        ("Shoulder", l.clavicle, l.shoulder, "Chest", false),
        ("UpperArm", l.shoulder, l.elbow, "Shoulder", true),
        ("ForeArm", l.elbow, l.wrist, "UpperArm", true),
        ("Hand", l.wrist, l.fingertip, "ForeArm", true),
        ("Thigh", l.hip, l.knee, "Hips", false),
        ("Shin", l.knee, l.ankle, "Thigh", true),
        ("Foot", l.ankle, l.toe, "Shin", true),
    ];
    for (suffix, mirror) in [(".L", false), (".R", true)] {
        for (name, head, tail, parent, connected) in &side {
            let place = |p: DVec3| if mirror { flip(p) } else { p };
            let parent = if *parent == "Chest" || *parent == "Hips" {
                parent.to_string()
            } else {
                format!("{parent}{suffix}")
            };
            bones.push((
                format!("{name}{suffix}"),
                place(*head),
                place(*tail),
                Some(parent),
                *connected,
            ));
        }
    }
    bones
}

/// An armature fitted to the active mesh, a figure standing up Z and facing
/// -Y with its arms out or down: Hips to Head up the middle, and on each side
/// a shoulder, arm and hand, a thigh, shin and foot, named `.L` and `.R`.
/// Bound at once unless asked not to; one step to undo.
pub struct AddHumanoid;

impl Operator for AddHumanoid {
    type Params = HumanoidParams;
    fn id() -> &'static str {
        "object.add_humanoid_rig"
    }
    fn label() -> &'static str {
        "Add Humanoid Rig"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .active
            .and_then(|id| doc.object(id))
            .is_some_and(|o| o.mesh().is_some())
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_scene::document::{Object, ObjectData};
        let d = doc.document();
        let Some(body) = d
            .selection
            .active
            .filter(|id| d.object(*id).is_some_and(|o| o.mesh().is_some()))
        else {
            return Ok(OpResult::Cancelled);
        };
        let Some(mesh) = d
            .object(body)
            .and_then(|o| o.mesh())
            .and_then(|m| d.mesh(m))
        else {
            return Ok(OpResult::Cancelled);
        };
        let world = d.world_transform(body);
        let points: Vec<DVec3> = mesh
            .positions()
            .iter()
            .map(|p| world.transform_point3(p.as_dvec3()))
            .collect();
        let landmarks = humanoid_landmarks(&points)
            .ok_or_else(|| OpError::Failed("the mesh is too small to read a figure from".into()))?;
        let mut bones = humanoid_bones(&landmarks);
        if params.fingers {
            // The hand ends at the knuckles, and the fingers go on from there,
            // each side's from its own side's points.
            let knuckle = landmarks.wrist + (landmarks.fingertip - landmarks.wrist) * 0.45;
            let flip = |p: DVec3| DVec3::new(-p.x, p.y, p.z);
            for bone in bones.iter_mut() {
                match bone.0.as_str() {
                    "Hand.L" => bone.2 = knuckle,
                    "Hand.R" => bone.2 = flip(knuckle),
                    _ => {}
                }
            }
            for (suffix, mirror) in [(".L", false), (".R", true)] {
                let side: Vec<DVec3> = if mirror {
                    points.iter().map(|p| flip(*p)).collect()
                } else {
                    points.clone()
                };
                for (name, head, tail, parent, connected) in finger_bones(&side, &landmarks) {
                    let place = |p: DVec3| if mirror { flip(p) } else { p };
                    bones.push((
                        format!("{name}{suffix}"),
                        place(head),
                        place(tail),
                        Some(format!("{parent}{suffix}")),
                        connected,
                    ));
                }
            }
        }
        let mut armature = Armature::new("Armature");
        armature.bones.clear();
        for (name, head, tail, parent, connected) in bones {
            let parent = parent.and_then(|p| armature.bone_index(&p));
            armature.add_bone(Bone {
                name,
                parent,
                head,
                tail,
                connected: connected && parent.is_some(),
                ..Default::default()
            });
        }
        armature.reconnect();
        let d = doc.document_mut();
        let armature = d.add_armature(armature);
        let rig = d.add_object(Object::new("Armature", ObjectData::Armature(armature)));
        d.selection.clear();
        d.selection.add(body);
        d.selection.add(rig);
        d.selection.active = Some(rig);
        if params.bind {
            // Arms hanging rather than held out: a bone's line to the body's
            // side then runs through the arm's own skin.
            let hanging = landmarks.fingertip.z
                < landmarks.shoulder.z - (landmarks.top.z - landmarks.ankle.z) * 0.15;
            Bind::execute(
                &BindParams {
                    weights: 0,
                    visibility: hanging,
                },
                doc,
                ctx,
            )?;
        }
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod symmetry_tests {
    use super::*;

    #[test]
    fn left_bones_are_mirrored_to_the_right_with_their_parents() {
        use kreate_scene::document::{Object, ObjectData};
        let mut doc = Document::new();
        let mut arm = Armature::new("Rig");
        arm.bones[0].name = "Spine".into();
        let upper = arm.add_bone(Bone {
            name: "Upper.L".into(),
            parent: Some(0),
            head: DVec3::new(0.2, 0.0, 1.0),
            tail: DVec3::new(0.6, 0.0, 1.0),
            roll: 0.3,
            ..Default::default()
        });
        arm.add_bone(Bone {
            name: "Lower.L".into(),
            parent: Some(upper),
            head: DVec3::new(0.6, 0.0, 1.0),
            tail: DVec3::new(0.9, 0.0, 1.0),
            connected: true,
            ..Default::default()
        });
        let armature = doc.add_armature(arm);
        let rig = doc.add_object(Object::new("Rig", ObjectData::Armature(armature)));
        doc.selection.add(rig);
        doc.selection.active = Some(rig);
        let mut edit = DocMut::new(&mut doc);
        Symmetrize::execute(&SymmetrizeParams::default(), &mut edit, &Context::default())
            .expect("ran");
        let a = doc.armature(armature).expect("armature");
        let upper_r = a.bone_index("Upper.R").expect("made");
        let lower_r = a.bone_index("Lower.R").expect("made");
        assert_eq!(a.bones[upper_r].head, DVec3::new(-0.2, 0.0, 1.0));
        assert_eq!(a.bones[upper_r].roll, -0.3);
        assert_eq!(a.bones[upper_r].parent, a.bone_index("Spine"));
        assert_eq!(
            a.bones[lower_r].parent,
            Some(upper_r),
            "its parent's mirror"
        );
        assert_eq!(a.bones.len(), 5);
    }

    /// A figure 1.8 m tall of boxes, each cut into rings up Z so every height
    /// has points: legs apart, arms straight out.
    fn figure() -> kreate_mesh::MeshData {
        let mut positions = Vec::new();
        let mut faces: Vec<Vec<u32>> = Vec::new();
        let mut boxed = |lo: glam::Vec3, hi: glam::Vec3, along_x: bool| {
            let rings = 12;
            let base = positions.len() as u32;
            for k in 0..=rings {
                let t = k as f32 / rings as f32;
                let corners: [glam::Vec3; 4] = if along_x {
                    let x = lo.x + (hi.x - lo.x) * t;
                    [
                        glam::Vec3::new(x, lo.y, lo.z),
                        glam::Vec3::new(x, hi.y, lo.z),
                        glam::Vec3::new(x, hi.y, hi.z),
                        glam::Vec3::new(x, lo.y, hi.z),
                    ]
                } else {
                    let z = lo.z + (hi.z - lo.z) * t;
                    [
                        glam::Vec3::new(lo.x, lo.y, z),
                        glam::Vec3::new(hi.x, lo.y, z),
                        glam::Vec3::new(hi.x, hi.y, z),
                        glam::Vec3::new(lo.x, hi.y, z),
                    ]
                };
                positions.extend(corners);
            }
            for k in 0..rings {
                for c in 0..4u32 {
                    let a = base + k * 4 + c;
                    let b = base + k * 4 + (c + 1) % 4;
                    faces.push(vec![a, b, b + 4, a + 4]);
                }
            }
        };
        let v = glam::Vec3::new;
        boxed(v(-0.18, -0.1, 0.9), v(0.18, 0.1, 1.45), false); // torso
        boxed(v(-0.1, -0.1, 1.5), v(0.1, 0.1, 1.8), false); // head
        boxed(v(0.05, -0.07, 0.0), v(0.17, 0.07, 0.9), false); // left leg
        boxed(v(-0.17, -0.07, 0.0), v(-0.05, 0.07, 0.9), false); // right leg
        boxed(v(0.18, -0.05, 1.35), v(0.85, 0.05, 1.45), true); // left arm
        boxed(v(-0.85, -0.05, 1.35), v(-0.18, 0.05, 1.45), true); // right arm
        kreate_mesh::MeshData::from_faces(positions, &faces)
    }

    #[test]
    fn a_humanoid_rig_is_fitted_to_a_figure_and_bound() {
        use kreate_scene::document::{Object, ObjectData};
        let mut doc = Document::new();
        let mesh = doc.add_mesh(figure());
        let body = doc.add_object(Object::new("Body", ObjectData::Mesh(mesh)));
        doc.selection.add(body);
        doc.selection.active = Some(body);
        let mut edit = DocMut::new(&mut doc);
        AddHumanoid::execute(&HumanoidParams::default(), &mut edit, &Context::default())
            .expect("ran");
        let (rig, armature) = doc
            .objects()
            .find_map(|(id, o)| match o.data {
                ObjectData::Armature(a) => Some((id, a)),
                _ => None,
            })
            .expect("a rig");
        let a = doc.armature(armature).expect("bones");
        assert_eq!(a.bones.len(), 19, "five up the middle, seven a side");
        let bone = |n: &str| &a.bones[a.bone_index(n).unwrap_or_else(|| panic!("{n}"))];
        // The hand ends at the arm's end, the thigh is in the leg, and the
        // right is the left's mirror.
        assert!(
            (bone("Hand.L").tail.x - 0.85).abs() < 0.02,
            "{:?}",
            bone("Hand.L").tail
        );
        assert!(
            (bone("Thigh.L").head.x - 0.11).abs() < 0.03,
            "{:?}",
            bone("Thigh.L").head
        );
        assert!((bone("Hand.R").tail.x + 0.85).abs() < 0.02);
        assert!(bone("Head").tail.z > 1.75 && bone("Foot.L").tail.z < 0.1);
        assert_eq!(bone("UpperArm.L").parent, a.bone_index("Shoulder.L"));
        // And bound: the body follows it, with a weight for every bone.
        let uid = doc.object(rig).expect("rig").uid;
        assert_eq!(doc.object(body).expect("body").skin, Some(uid));
        let m = doc.mesh(mesh).expect("mesh");
        assert!(m
            .attributes()
            .get_f32(&kreate_scene::skin::weight_layer("ForeArm.L"))
            .is_some_and(|w| w.iter().any(|x| *x > 0.5)));
    }

    #[test]
    fn fingers_spread_across_each_hand_when_asked() {
        use kreate_scene::document::{Object, ObjectData};
        let mut doc = Document::new();
        let mesh = doc.add_mesh(figure());
        let body = doc.add_object(Object::new("Body", ObjectData::Mesh(mesh)));
        doc.selection.add(body);
        doc.selection.active = Some(body);
        let mut edit = DocMut::new(&mut doc);
        AddHumanoid::execute(
            &HumanoidParams {
                bind: false,
                fingers: true,
            },
            &mut edit,
            &Context::default(),
        )
        .expect("ran");
        let a = doc
            .armatures()
            .next()
            .map(|(_, a)| a.clone())
            .expect("an armature");
        assert_eq!(a.bones.len(), 19 + 30, "five fingers of three a hand");
        let bone = |n: &str| a.bones[a.bone_index(n).unwrap_or_else(|| panic!("{n}"))].clone();
        let (index, pinky) = (bone("Index1.L"), bone("Pinky1.L"));
        assert!(
            index.head.y < pinky.head.y,
            "index at the front: {index:?} {pinky:?}"
        );
        assert!(bone("Middle3.L").tail.x <= 0.86 && bone("Middle3.L").tail.x > 0.75);
        assert_eq!(
            a.bones[bone("Index1.L").parent.expect("parent")].name,
            "Hand.L"
        );
        assert!(
            (bone("Index1.R").head.x + index.head.x).abs() < 1e-9,
            "mirrored"
        );
    }
}
