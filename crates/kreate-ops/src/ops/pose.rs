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
use kreate_scene::armature::{self, Armature, IkConstraint};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<AddIk>();
    registry.register::<SetIk>();
    registry.register::<RemoveIk>();
    registry.register::<AddConstraint>();
    registry.register::<SetConstraint>();
    registry.register::<RemoveConstraint>();
    registry.register::<AddArmature>();
    registry.register::<SelectPick>();
    registry.register::<SelectAll>();
    registry.register::<Translate>();
    registry.register::<Rotate>();
    registry.register::<Scale>();
    registry.register::<ClearTransform>();
    registry.register::<SetPoseBone>();
}

/// The armature object pose mode acts on.
pub fn posed_object(doc: &Document) -> Option<ObjectId> {
    let active = doc.selection.active?;
    doc.armature_of(active).map(|_| active)
}

/// The bones a transform moves: selected, with no selected ancestor.
fn moving_bones(armature: &Armature, object: &Object) -> Vec<usize> {
    let selected: Vec<usize> = (0..armature.bones.len())
        .filter(|i| object.pose.is_selected(&armature.bones[*i].name))
        .collect();
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
    let heads: Vec<DVec3> = moving_bones(armature, o)
        .iter()
        .map(|i| frames[*i].translation)
        .collect();
    if heads.is_empty() {
        return None;
    }
    Some(heads.iter().copied().sum::<DVec3>() / heads.len() as f64)
}

/// Moves each moving bone's world frame by `change`, as a pose.
fn transform_bones(
    doc: &mut DocMut<'_>,
    change: impl Fn(DAffine3) -> DAffine3,
) -> Result<OpResult, OpError> {
    let Some(object) = posed_object(doc.document()) else {
        return Ok(OpResult::Cancelled);
    };
    let d = doc.document();
    let Some((_, armature)) = d.armature_of(object) else {
        return Ok(OpResult::Cancelled);
    };
    let armature = armature.clone();
    let Some(o) = d.object(object) else {
        return Ok(OpResult::Cancelled);
    };
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
            (
                armature.bones[i].name.clone(),
                armature::pose_for_frame(&armature, &frames, i, target),
            )
        })
        .collect();
    let Some(o) = doc.document_mut().object_mut(object) else {
        return Ok(OpResult::Cancelled);
    };
    for (name, pose) in poses {
        let entry = o.pose.entry(&name);
        entry.transform = DTransform {
            rotation: pose.rotation.normalize(),
            ..pose
        };
    }
    Ok(OpResult::Finished)
}

/// Keys the bones a transform just moved, when auto-keying asked for it — in
/// the same step, so the move and its key undo together.
fn autokey(
    doc: &mut DocMut<'_>,
    done: OpResult,
    on: bool,
    frame: f64,
    properties: &[&str],
) -> Result<OpResult, OpError> {
    if on && done == OpResult::Finished && frame.is_finite() {
        if let Some(object) = posed_object(doc.document()) {
            let bones: Vec<String> = {
                let d = doc.document();
                match (d.armature_of(object), d.object(object)) {
                    (Some((_, a)), Some(o)) => moving_bones(a, o)
                        .into_iter()
                        .map(|i| a.bones[i].name.clone())
                        .collect(),
                    _ => Vec::new(),
                }
            };
            crate::ops::anim::key_bones(doc.document_mut(), object, &bones, frame, properties);
        }
    }
    Ok(done)
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some((_, armature)) = doc.document().armature_of(object) else {
            return Ok(OpResult::Cancelled);
        };
        let names: Vec<String> = armature.bones.iter().map(|b| b.name.clone()).collect();
        if !names.contains(&params.bone) {
            return Ok(OpResult::Cancelled);
        }
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some((_, armature)) = doc.document().armature_of(object) else {
            return Ok(OpResult::Cancelled);
        };
        let names: Vec<String> = armature.bones.iter().map(|b| b.name.clone()).collect();
        let before = doc
            .document()
            .object(object)
            .map(|o| o.pose.clone())
            .unwrap_or_default();
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
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        for n in &names {
            let selected = want(n);
            if selected != o.pose.is_selected(n) {
                o.pose.entry(n).selected = selected;
            }
        }
        if o.pose
            .active
            .as_ref()
            .is_some_and(|a| !o.pose.is_selected(a))
        {
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
    /// Key what this changed at `key_frame` — auto-keying, as on objects.
    #[param(hidden)]
    pub autokey: bool,
    #[param(hidden)]
    pub key_frame: f64,
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let offset = DVec3::from_array(params.translation);
        if !offset.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let done = transform_bones(doc, |f| DAffine3::from_translation(offset) * f)?;
        autokey(doc, done, params.autokey, params.key_frame, &["location"])
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RotateParams {
    #[param(unit = "angle")]
    pub angle: f64,
    #[param(variants("x", "y", "z"))]
    pub axis: u32,
    #[param(hidden)]
    pub autokey: bool,
    #[param(hidden)]
    pub key_frame: f64,
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if !params.angle.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let Some(p) = pivot(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let axis = [DVec3::X, DVec3::Y, DVec3::Z][(params.axis as usize).min(2)];
        let turn = DAffine3::from_translation(p)
            * DAffine3::from_quat(DQuat::from_axis_angle(axis, params.angle))
            * DAffine3::from_translation(-p);
        let done = transform_bones(doc, |f| turn * f)?;
        // The location too: a bone turned about a point that is not its head
        // has moved as well as turned.
        autokey(
            doc,
            done,
            params.autokey,
            params.key_frame,
            &["location", "rotation_quaternion"],
        )
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ScaleParams {
    /// Along each of the bone's own axes.
    pub scale: [f64; 3],
    #[param(hidden)]
    pub autokey: bool,
    #[param(hidden)]
    pub key_frame: f64,
}

impl Default for ScaleParams {
    fn default() -> Self {
        Self {
            scale: [1.0; 3],
            autokey: false,
            key_frame: 1.0,
        }
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let s = DVec3::from_array(params.scale);
        if !s.is_finite() || s.min_element().abs() < 1e-9 {
            return Ok(OpResult::Cancelled);
        }
        let done = transform_bones(doc, |f| f * DAffine3::from_scale(s))?;
        autokey(doc, done, params.autokey, params.key_frame, &["scale"])
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
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let cleared = |t: DTransform| match params.what {
            1 => DTransform {
                translation: DVec3::ZERO,
                ..t
            },
            2 => DTransform {
                rotation: DQuat::IDENTITY,
                ..t
            },
            3 => DTransform {
                scale: DVec3::ONE,
                ..t
            },
            _ => DTransform::IDENTITY,
        };
        let Some(o) = doc.document().object(object) else {
            return Ok(OpResult::Cancelled);
        };
        if !o
            .pose
            .bones
            .iter()
            .any(|b| b.selected && cleared(b.transform) != b.transform)
        {
            return Ok(OpResult::Cancelled);
        }
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        for b in o.pose.bones.iter_mut().filter(|b| b.selected) {
            b.transform = cleared(b.transform);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetPoseBoneParams {
    /// Which bone.
    #[param(hidden)]
    pub bone: String,
    #[param(unit = "length")]
    pub location: [f64; 3],
    /// As XYZ Euler angles, the numbers a person types; stored as the
    /// quaternion the pose keeps.
    #[param(unit = "angle")]
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
}

impl Default for SetPoseBoneParams {
    fn default() -> Self {
        Self {
            bone: String::new(),
            location: [0.0; 3],
            rotation: [0.0; 3],
            scale: [1.0; 3],
        }
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct IkParams {
    /// The bone whose tip reaches; empty for the active bone.
    pub bone: String,
    /// The bone it reaches for; empty for the other selected bone, as
    /// Blender's Shift+I takes it — or none yet, to pick in the Bone tab.
    pub target: String,
    /// How many bones turn, counting the bone itself; zero for all the way up.
    #[param(min = 0, max = 64, unit = "count")]
    pub chain: u32,
    pub enabled: bool,
    /// The bone the first joint bends towards; empty for none.
    pub pole: String,
}

impl Default for IkParams {
    fn default() -> Self {
        Self {
            bone: String::new(),
            target: String::new(),
            chain: 2,
            enabled: true,
            pole: String::new(),
        }
    }
}

/// The named bone, or the active one.
fn bone_or_active(doc: &Document, object: ObjectId, bone: &str) -> Option<String> {
    let (_, armature) = doc.armature_of(object)?;
    let name = if bone.is_empty() {
        doc.object(object)?.pose.active.clone()?
    } else {
        bone.to_string()
    };
    armature.bone_index(&name).map(|_| name)
}

/// Blender's Shift+I: an IK constraint on the active bone, reaching for the
/// other selected bone. One per bone; adding again replaces it.
pub struct AddIk;

impl Operator for AddIk {
    type Params = IkParams;
    fn id() -> &'static str {
        "pose.ik_add"
    }
    fn label() -> &'static str {
        "Add IK"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        posed_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let d = doc.document();
        let Some(bone) = bone_or_active(d, object, &params.bone) else {
            return Err(OpError::Failed("no active bone to put IK on".into()));
        };
        let target = if params.target.is_empty() {
            let (_, armature) = d.armature_of(object).expect("posed");
            let pose = &d.object(object).expect("posed").pose;
            armature
                .bones
                .iter()
                .map(|b| b.name.clone())
                .find(|n| *n != bone && pose.is_selected(n))
                .unwrap_or_default()
        } else {
            params.target.clone()
        };
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        o.pose.ik.retain(|c| c.bone != bone);
        o.pose.ik.push(IkConstraint {
            bone,
            target,
            chain: params.chain,
            enabled: params.enabled,
            pole: params.pole.clone(),
        });
        Ok(OpResult::Finished)
    }
}

/// Changes a bone's IK: its target, chain, whether it is on.
pub struct SetIk;

impl Operator for SetIk {
    type Params = IkParams;
    fn id() -> &'static str {
        "pose.ik_set"
    }
    fn label() -> &'static str {
        "Set IK"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        posed_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(bone) = bone_or_active(doc.document(), object, &params.bone) else {
            return Ok(OpResult::Cancelled);
        };
        let wanted = IkConstraint {
            bone: bone.clone(),
            target: params.target.clone(),
            chain: params.chain,
            enabled: params.enabled,
            pole: params.pole.clone(),
        };
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        match o.pose.ik.iter_mut().find(|c| c.bone == bone) {
            Some(c) if *c == wanted => Ok(OpResult::Cancelled),
            Some(c) => {
                *c = wanted;
                Ok(OpResult::Finished)
            }
            None => Ok(OpResult::Cancelled),
        }
    }
}

/// Takes a bone's IK off.
pub struct RemoveIk;

impl Operator for RemoveIk {
    type Params = IkParams;
    fn id() -> &'static str {
        "pose.ik_remove"
    }
    fn label() -> &'static str {
        "Remove IK"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        posed_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(bone) = bone_or_active(doc.document(), object, &params.bone) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        let before = o.pose.ik.len();
        o.pose.ik.retain(|c| c.bone != bone);
        Ok(if o.pose.ik.len() < before {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ConstraintParams {
    /// Which of the pose's constraints, for setting and removing; ignored when
    /// adding.
    #[param(hidden, min = 0, max = 4096, unit = "count")]
    pub index: u32,
    /// The bone it is on; empty for the active bone. Adding only.
    pub bone: String,
    #[param(variants(
        "copy_location",
        "copy_rotation",
        "copy_scale",
        "damped_track",
        "limit_rotation",
        "limit_location"
    ))]
    pub kind: u32,
    /// The bone it reads; empty when adding takes the other selected bone.
    pub target: String,
    #[param(min = 0.0, max = 1.0)]
    pub influence: f64,
    pub enabled: bool,
    /// A limit's lowest XYZ: angles in degrees for Limit Rotation, metres
    /// for Limit Location.
    pub min: [f64; 3],
    /// A limit's highest XYZ, in the same units.
    pub max: [f64; 3],
}

impl Default for ConstraintParams {
    fn default() -> Self {
        Self {
            index: 0,
            bone: String::new(),
            kind: 0,
            target: String::new(),
            influence: 1.0,
            enabled: true,
            min: [-180.0; 3],
            max: [180.0; 3],
        }
    }
}

impl ConstraintParams {
    /// The constraint these describe, on `bone`.
    fn constraint(&self, bone: String) -> Option<armature::BoneConstraint> {
        let mut kind = armature::ConstraintKind::from_index(self.kind)?;
        match &mut kind {
            armature::ConstraintKind::LimitRotation { min, max } => {
                *min = DVec3::from_array(self.min.map(f64::to_radians));
                *max = DVec3::from_array(self.max.map(f64::to_radians));
            }
            armature::ConstraintKind::LimitLocation { min, max } => {
                *min = DVec3::from_array(self.min);
                *max = DVec3::from_array(self.max);
            }
            _ => {}
        }
        Some(armature::BoneConstraint {
            bone,
            kind,
            target: self.target.clone(),
            influence: self.influence.clamp(0.0, 1.0),
            enabled: self.enabled,
        })
    }
}

/// What the Bone tab shows for one constraint, as the operator's parameters.
pub fn constraint_params(index: usize, c: &armature::BoneConstraint) -> ConstraintParams {
    let (min, max) = match &c.kind {
        armature::ConstraintKind::LimitRotation { min, max } => (
            min.to_array().map(f64::to_degrees),
            max.to_array().map(f64::to_degrees),
        ),
        armature::ConstraintKind::LimitLocation { min, max } => (min.to_array(), max.to_array()),
        _ => ([-180.0; 3], [180.0; 3]),
    };
    ConstraintParams {
        index: index as u32,
        bone: c.bone.clone(),
        kind: c.kind.index(),
        target: c.target.clone(),
        influence: c.influence,
        enabled: c.enabled,
        min,
        max,
    }
}

/// A bone constraint on the active bone: Copy Location, Copy Rotation, Copy
/// Scale, Damped Track or Limit Rotation. Blender's Shift+Ctrl+C. The target
/// is the other selected bone when none is named.
pub struct AddConstraint;

impl Operator for AddConstraint {
    type Params = ConstraintParams;
    fn id() -> &'static str {
        "pose.constraint_add"
    }
    fn label() -> &'static str {
        "Add Bone Constraint"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        posed_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let d = doc.document();
        let Some(bone) = bone_or_active(d, object, &params.bone) else {
            return Err(OpError::Failed("no active bone to constrain".into()));
        };
        let mut params = params.clone();
        if params.target.is_empty() {
            let (_, armature) = d.armature_of(object).expect("posed");
            let pose = &d.object(object).expect("posed").pose;
            params.target = armature
                .bones
                .iter()
                .map(|b| b.name.clone())
                .find(|n| *n != bone && pose.is_selected(n))
                .unwrap_or_default();
        }
        let Some(constraint) = params.constraint(bone) else {
            return Err(OpError::Failed("no such constraint".into()));
        };
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        o.pose.constraints.push(constraint);
        Ok(OpResult::Finished)
    }
}

/// Changes one of the pose's constraints, by its place in the stack.
pub struct SetConstraint;

impl Operator for SetConstraint {
    type Params = ConstraintParams;
    fn id() -> &'static str {
        "pose.constraint_set"
    }
    fn label() -> &'static str {
        "Set Bone Constraint"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        posed_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(c) = o.pose.constraints.get_mut(params.index as usize) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(wanted) = params.constraint(c.bone.clone()) else {
            return Err(OpError::Failed("no such constraint".into()));
        };
        if *c == wanted {
            return Ok(OpResult::Cancelled);
        }
        *c = wanted;
        Ok(OpResult::Finished)
    }
}

/// Takes one constraint off the pose, by its place in the stack.
pub struct RemoveConstraint;

impl Operator for RemoveConstraint {
    type Params = ConstraintParams;
    fn id() -> &'static str {
        "pose.constraint_remove"
    }
    fn label() -> &'static str {
        "Remove Bone Constraint"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        posed_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        if (params.index as usize) >= o.pose.constraints.len() {
            return Ok(OpResult::Cancelled);
        }
        o.pose.constraints.remove(params.index as usize);
        Ok(OpResult::Finished)
    }
}

/// What the Bone tab shows for a posed bone, as the operator's parameters.
pub fn pose_bone_params(pose: &kreate_scene::armature::Pose, bone: &str) -> SetPoseBoneParams {
    let t = pose.transform(bone);
    SetPoseBoneParams {
        bone: bone.to_string(),
        location: t.translation.to_array(),
        rotation: kreate_scene::animation::to_euler(t.rotation).to_array(),
        scale: t.scale.to_array(),
    }
}

/// Places one bone's pose exactly: the Bone tab in pose mode.
pub struct SetPoseBone;

impl Operator for SetPoseBone {
    type Params = SetPoseBoneParams;
    fn id() -> &'static str {
        "pose.set_bone"
    }
    fn label() -> &'static str {
        "Set Bone Pose"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        posed_object(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(object) = posed_object(doc.document()) else {
            return Ok(OpResult::Cancelled);
        };
        let exists = doc
            .document()
            .armature_of(object)
            .is_some_and(|(_, a)| a.bone_index(&params.bone).is_some());
        let (l, r, s) = (
            DVec3::from_array(params.location),
            DVec3::from_array(params.rotation),
            DVec3::from_array(params.scale),
        );
        if !exists || !l.is_finite() || !r.is_finite() || !s.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let wanted = DTransform {
            translation: l,
            rotation: kreate_scene::animation::from_euler(r).normalize(),
            scale: s,
        };
        let now = doc
            .document()
            .object(object)
            .map(|o| o.pose.transform(&params.bone));
        if now == Some(wanted) {
            return Ok(OpResult::Cancelled);
        }
        let Some(o) = doc.document_mut().object_mut(object) else {
            return Ok(OpResult::Cancelled);
        };
        o.pose.entry(&params.bone).transform = wanted;
        Ok(OpResult::Finished)
    }
}
