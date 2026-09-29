//! Armatures and their poses: adding one, selecting bones, moving them.
//!
//! Pose mode acts on the active object's armature, as Blender's does with one
//! armature in it. Bones are named by their names — the pose's own key, and
//! what a script can say.
//!
//! # Moving bones in the world's terms
//!
//! `G`, `R` and `S` hand these operators what they hand the object ones: a
//! world offset, a turn about a world axis through a pivot, a scale. A bone's
//! pose is in its own axes, so each operator works out where the bone's frame
//! should go in the world and asks [`armature::pose_for_frame`] what pose puts
//! it there — the one conversion, done exactly, whatever the bone's parents
//! are doing.
//!
//! A selected bone whose ancestor is also selected is left to the ancestor,
//! as in Blender: moving both would move the child twice.

use kreate_math::{DAffine3, DQuat, DTransform, DVec3};
use kreate_ops_derive::OpParams;
use kreate_scene::armature::{self, Armature};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddArmature>();
    registry.register::<SelectPick>();
    registry.register::<SelectAll>();
    registry.register::<Translate>();
    registry.register::<Rotate>();
    registry.register::<Scale>();
    registry.register::<ClearTransform>();
}

/// The armature object pose mode acts on.
pub fn posed_object(doc: &Document) -> Option<ObjectId> {
    let active = doc.selection.active?;
    doc.armature_of(active).map(|_| active)
}

/// The bones a transform moves: selected, with no selected ancestor.
fn moving_bones(armature: &Armature, object: &Object) -> Vec<usize> {
    let selected: Vec<usize> =
        (0..armature.bones.len()).filter(|i| object.pose.is_selected(&armature.bones[*i].name)).collect();
    selected
        .iter()
        .copied()
        .filter(|&i| {
            !selected
                .iter()
                .any(|&other| other != i && armature::is_ancestor(armature, other, i))
        })
        .collect()
}

/// Where the selected bones' heads are on average, in the world: what `R` and
/// `S` turn about, and what the application measures the pointer from.
pub fn pivot(doc: &Document) -> Option<DVec3> {
    let object = posed_object(doc)?;
    let (_, armature) = doc.armature_of(object)?;
    let o = doc.object(object)?;
    let frames = doc.posed_bones(object);
    let heads: Vec<DVec3> = moving_bones(armature, o).iter().map(|i| frames[*i].translation).collect();
    if heads.is_empty() {
        return None;
    }
    Some(heads.iter().copied().sum::<DVec3>() / heads.len() as f64)
}

/// Moves each moving bone's world frame by `change`, as a pose.
fn transform_bones(doc: &mut DocMut<'_>, change: impl Fn(DAffine3) -> DAffine3) -> Result<OpResult, OpError> {
    let Some(object) = posed_object(doc.document()) else {
        return Ok(OpResult::Cancelled);
    };
    let d = doc.document();
    let Some((_, armature)) = d.armature_of(object) else {
        return Ok(OpResult::Cancelled);
    };
    let armature = armature.clone();
    let Some(o) = d.object(object) else { return Ok(OpResult::Cancelled) };
    let moving = moving_bones(&armature, o);
    if moving.is_empty() {
        return Ok(OpResult::Cancelled);
    }
    let world = d.world_transform(object);
    // A zero-scaled armature has no way back from the world.
    if world.matrix3.determinant().abs() < 1e-12 {
        return Ok(OpResult::Cancelled);
    }
    let to_armature = world.inverse();
    let frames = armature::posed(&armature, &o.pose);
    let poses: Vec<(String, DTransform)> = moving
        .iter()
        .map(|&i| {
            let target = to_armature * change(world * frames[i]);
            (armature.bones[i].name.clone(), armature::pose_for_frame(&armature, &frames, i, target))
        })
        .collect();
    let Some(o) = doc.document_mut().object_mut(object) else { return Ok(OpResult::Cancelled) };
    for (name, pose) in poses {
        let entry = o.pose.entry(&name);
        entry.transform = DTransform { rotation: pose.rotation.normalize(), ..pose };
    }
    Ok(OpResult::Finished)
}

fn in_pose_mode(ctx: &Context, doc: &Document) -> bool {
    ctx.mode.acts_on() == crate::mode::ActsOn::Bones && posed_object(doc).is_some()
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddArmatureParams {
    #[param(unit = "length")]
    pub location: [f64; 3],
}

/// Adds an armature with one bone — Blender's Add ▸ Armature.
pub struct AddArmature;

impl Operator for AddArmature {
    type Params = AddArmatureParams;
    fn id() -> &'static str {
        "object.add_armature"
    }
    fn label() -> &'static str {
        "Add Armature"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let d = doc.document_mut();
        let datablock = d.add_armature(Armature::new("Armature"));
        let mut object = Object::new("Armature", ObjectData::Armature(datablock));
        object.transform.translation = DVec3::from_array(params.location);
        let id = d.add_object(object);
        d.selection.clear();
        d.selection.add(id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectPickParams {
    /// The bone, by name.
    pub bone: String,
    /// Add to the selection, or take the bone out of it if it was already the
    /// active one — Shift-click, as for objects.
    pub extend: bool,
}

/// Selects a bone of the posed armature.
pub struct SelectPick;

impl Operator for SelectPick {
    type Params = SelectPickParams;
    fn id() -> &'static str {
        "pose.select_pick"
    }
    fn label() -> &'static str {
        "Select Bone"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_pose_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else { return Ok(OpResult::Cancelled) };
        let Some((_, armature)) = doc.document().armature_of(object) else { return Ok(OpResult::Cancelled) };
        let names: Vec<String> = armature.bones.iter().map(|b| b.name.clone()).collect();
        if !names.contains(&params.bone) {
            return Ok(OpResult::Cancelled);
        }
        let Some(o) = doc.document_mut().object_mut(object) else { return Ok(OpResult::Cancelled) };
        let pose = &mut o.pose;
        if params.extend {
            let was_active = pose.active.as_deref() == Some(params.bone.as_str());
            let entry = pose.entry(&params.bone);
            if entry.selected && was_active {
                entry.selected = false;
                pose.active = None;
            } else {
                entry.selected = true;
                pose.active = Some(params.bone.clone());
            }
        } else {
            for name in &names {
                if pose.is_selected(name) {
                    pose.entry(name).selected = false;
                }
            }
            pose.entry(&params.bone).selected = true;
            pose.active = Some(params.bone.clone());
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectAllParams {
    #[param(variants("select", "deselect", "toggle", "invert"))]
    pub action: u32,
}

/// Selects every bone, none, or the other ones.
pub struct SelectAll;

impl Operator for SelectAll {
    type Params = SelectAllParams;
    fn id() -> &'static str {
        "pose.select_all"
    }
    fn label() -> &'static str {
        "Select All Bones"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_pose_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else { return Ok(OpResult::Cancelled) };
        let Some((_, armature)) = doc.document().armature_of(object) else { return Ok(OpResult::Cancelled) };
        let names: Vec<String> = armature.bones.iter().map(|b| b.name.clone()).collect();
        let before = doc.document().object(object).map(|o| o.pose.clone()).unwrap_or_default();
        let any = names.iter().any(|n| before.is_selected(n));
        let want = |n: &str| match params.action {
            0 => true,
            1 => false,
            2 => !any,
            _ => !before.is_selected(n),
        };
        if names.iter().all(|n| want(n) == before.is_selected(n)) {
            return Ok(OpResult::Cancelled);
        }
        let Some(o) = doc.document_mut().object_mut(object) else { return Ok(OpResult::Cancelled) };
        for n in &names {
            let selected = want(n);
            if selected != o.pose.is_selected(n) {
                o.pose.entry(n).selected = selected;
            }
        }
        if o.pose.active.as_ref().is_some_and(|a| !o.pose.is_selected(a)) {
            o.pose.active = None;
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TranslateParams {
    /// World offset, in metres.
    #[param(unit = "length")]
    pub translation: [f64; 3],
}

/// Moves the selected bones.
pub struct Translate;

impl Operator for Translate {
    type Params = TranslateParams;
    fn id() -> &'static str {
        "pose.translate"
    }
    fn label() -> &'static str {
        "Move Bones"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_pose_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let offset = DVec3::from_array(params.translation);
        if !offset.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        transform_bones(doc, |f| DAffine3::from_translation(offset) * f)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RotateParams {
    #[param(unit = "angle")]
    pub angle: f64,
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
}

/// Turns the selected bones about a world axis through their heads' median.
pub struct Rotate;

impl Operator for Rotate {
    type Params = RotateParams;
    fn id() -> &'static str {
        "pose.rotate"
    }
    fn label() -> &'static str {
        "Rotate Bones"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_pose_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        if !params.angle.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let Some(p) = pivot(doc.document()) else { return Ok(OpResult::Cancelled) };
        let axis = [DVec3::X, DVec3::Y, DVec3::Z][(params.axis as usize).min(2)];
        let turn = DAffine3::from_translation(p)
            * DAffine3::from_quat(DQuat::from_axis_angle(axis, params.angle))
            * DAffine3::from_translation(-p);
        transform_bones(doc, |f| turn * f)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ScaleParams {
    /// Along each of the bone's own axes.
    pub scale: [f64; 3],
}

impl Default for ScaleParams {
    fn default() -> Self {
        Self { scale: [1.0; 3] }
    }
}

/// Scales the selected bones, each about its own head in its own axes.
///
/// Its own axes rather than the world's: a scaled bone scales everything it
/// carries, and a world-axis scale on a turned bone is a shear nothing
/// downstream can hold. Which is also why the scale does not move the heads
/// towards a shared pivot — Blender's Individual Origins, the choice a posed
/// rig almost always wants.
pub struct Scale;

impl Operator for Scale {
    type Params = ScaleParams;
    fn id() -> &'static str {
        "pose.scale"
    }
    fn label() -> &'static str {
        "Scale Bones"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_pose_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let s = DVec3::from_array(params.scale);
        if !s.is_finite() || s.min_element().abs() < 1e-9 {
            return Ok(OpResult::Cancelled);
        }
        transform_bones(doc, |f| f * DAffine3::from_scale(s))
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ClearParams {
    #[param(variants("all", "location", "rotation", "scale"))]
    pub what: u32,
}

/// Puts the selected bones back at rest — Alt+G, Alt+R, Alt+S.
pub struct ClearTransform;

impl Operator for ClearTransform {
    type Params = ClearParams;
    fn id() -> &'static str {
        "pose.clear_transform"
    }
    fn label() -> &'static str {
        "Clear Pose Transform"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        in_pose_mode(ctx, doc)
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else { return Ok(OpResult::Cancelled) };
        let cleared = |t: DTransform| match params.what {
            1 => DTransform { translation: DVec3::ZERO, ..t },
            2 => DTransform { rotation: DQuat::IDENTITY, ..t },
            3 => DTransform { scale: DVec3::ONE, ..t },
            _ => DTransform::IDENTITY,
        };
        let Some(o) = doc.document().object(object) else { return Ok(OpResult::Cancelled) };
        if !o.pose.bones.iter().any(|b| b.selected && cleared(b.transform) != b.transform) {
            return Ok(OpResult::Cancelled);
        }
        let Some(o) = doc.document_mut().object_mut(object) else { return Ok(OpResult::Cancelled) };
        for b in o.pose.bones.iter_mut().filter(|b| b.selected) {
            b.transform = cleared(b.transform);
        }
        Ok(OpResult::Finished)
    }
}
