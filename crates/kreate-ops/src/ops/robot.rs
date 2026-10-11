//! Posing a robot by its joints.
//!
//! A joint is an object that turns about its own Z (or slides along it) from
//! where it rests — see `kreate_scene::robot`. Setting one is setting that
//! turn, held to the joint's limits, and the joints that mimic it follow, as
//! the robot's controller would make them.

use std::collections::HashSet;

use kreate_scene::document::ObjectId;
use kreate_scene::robot::{Joint, RobotPart};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<SetJoint>();
    registry.register::<ResetPose>();
}

fn joint_of(doc: &Document, id: ObjectId) -> Option<Joint> {
    match &doc.object(id)?.robot {
        Some(RobotPart::Joint(j)) => Some(j.clone()),
        _ => None,
    }
}

fn find_joint(doc: &Document, name: &str) -> Option<ObjectId> {
    if name.is_empty() {
        return doc
            .selection
            .active
            .filter(|id| joint_of(doc, *id).is_some());
    }
    doc.objects()
        .find(|(id, o)| o.name == name && joint_of(doc, *id).is_some())
        .map(|(id, _)| id)
}

/// Puts `id` at `position` (held to its limits), and every joint that
/// mimics it where it says — and theirs, once each.
fn set(doc: &mut Document, id: ObjectId, position: f64) -> bool {
    let mut changed = false;
    let mut todo = vec![(id, position)];
    let mut seen = HashSet::new();
    while let Some((id, position)) = todo.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(joint) = joint_of(doc, id) else {
            continue;
        };
        if !joint.kind.moves() {
            continue;
        }
        let held = joint.clamp(position);
        let placed = joint.place(held);
        let name = match doc.object_mut(id) {
            Some(o) => {
                if o.transform != placed {
                    o.transform = placed;
                    changed = true;
                }
                o.name.clone()
            }
            None => continue,
        };
        for (other, o) in doc.objects() {
            if let Some(RobotPart::Joint(j)) = &o.robot {
                if let Some(m) = j.mimic.as_ref().filter(|m| m.joint == name) {
                    todo.push((other, m.multiplier * held + m.offset));
                }
            }
        }
    }
    changed
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetJointParams {
    /// The joint's object, by name; empty for the active object.
    pub joint: String,
    /// Radians for a joint that turns, metres for one that slides. Held to
    /// the joint's limits.
    pub position: f64,
}

/// Turns (or slides) one joint, and those that mimic it.
pub struct SetJoint;

impl Operator for SetJoint {
    type Params = SetJointParams;
    fn id() -> &'static str {
        "robot.set_joint"
    }
    fn label() -> &'static str {
        "Set Joint"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.objects()
            .any(|(_, o)| matches!(o.robot, Some(RobotPart::Joint(_))))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if !params.position.is_finite() {
            return Err(OpError::Failed(
                "a joint's position must be a number".into(),
            ));
        }
        let Some(id) = find_joint(doc.document(), &params.joint) else {
            return Err(OpError::Failed(if params.joint.is_empty() {
                "the active object is not a robot's joint".into()
            } else {
                format!("no joint called `{}`", params.joint)
            }));
        };
        if set(doc.document_mut(), id, params.position) {
            Ok(OpResult::Finished)
        } else {
            Ok(OpResult::Cancelled)
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ResetPoseParams {}

/// Every joint back where it rests: the robot as its description draws it.
pub struct ResetPose;

impl Operator for ResetPose {
    type Params = ResetPoseParams;
    fn id() -> &'static str {
        "robot.reset_pose"
    }
    fn label() -> &'static str {
        "Reset Robot Pose"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        SetJoint::poll(ctx, doc)
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let joints: Vec<(ObjectId, Joint)> = doc
            .document()
            .objects()
            .filter_map(|(id, o)| match &o.robot {
                Some(RobotPart::Joint(j)) if j.kind.moves() => Some((id, j.clone())),
                _ => None,
            })
            .collect();
        let mut changed = false;
        for (id, j) in joints {
            if let Some(o) = doc.document_mut().object_mut(id) {
                let rest = j.place(j.clamp(0.0));
                if o.transform != rest {
                    o.transform = rest;
                    changed = true;
                }
            }
        }
        Ok(if changed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::{DQuat, DTransform, DVec3};
    use kreate_scene::robot::{JointKind, Mimic};
    use kreate_scene::{Object, ObjectData};

    fn joint(name: &str, mimic: Option<Mimic>) -> Object {
        let mut o = Object::new(name, ObjectData::Empty);
        o.robot = Some(RobotPart::Joint(Joint {
            kind: JointKind::Revolute,
            rest: DTransform {
                translation: DVec3::Z,
                rotation: DQuat::IDENTITY,
                scale: DVec3::ONE,
            },
            lower: -1.0,
            upper: 1.0,
            mimic,
            ..Joint::default()
        }));
        o.transform = DTransform {
            translation: DVec3::Z,
            ..DTransform::IDENTITY
        };
        o
    }

    #[test]
    fn a_joint_turns_within_its_limits_and_its_mimic_follows() {
        let mut doc = Document::new();
        let a = doc.add_object(joint("finger", None));
        let b = doc.add_object(joint(
            "finger_mirror",
            Some(Mimic {
                joint: "finger".into(),
                multiplier: -0.5,
                offset: 0.0,
            }),
        ));
        assert!(set(&mut doc, a, 5.0));
        let ja = joint_of(&doc, a).unwrap();
        let jb = joint_of(&doc, b).unwrap();
        let pa = ja.position_of(&doc.object(a).unwrap().transform);
        let pb = jb.position_of(&doc.object(b).unwrap().transform);
        assert!((pa - 1.0).abs() < 1e-9, "held to its upper limit: {pa}");
        assert!((pb + 0.5).abs() < 1e-9, "the mimic follows: {pb}");
        assert!(!set(&mut doc, a, 1.0), "already there");
    }
}
