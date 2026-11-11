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
    registry.register::<RigRobot>();
    registry.register::<AddJoint>();
    registry.register::<MotionCheck>();
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
        // The rig's bone with it: a rigged robot's joints follow its bones,
        // and a joint set without its bone would spring back.
        changed |= doc.hold_joint_in_rig(id, held);
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
            let d = doc.document_mut();
            if let Some(o) = d.object_mut(id) {
                let rest = j.place(j.clamp(0.0));
                if o.transform != rest {
                    o.transform = rest;
                    changed = true;
                }
            }
            changed |= d.hold_joint_in_rig(id, j.clamp(0.0));
        }
        Ok(if changed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RigRobotParams {}

/// Makes an armature that poses the robot the active object belongs to: a
/// bone for each joint that moves, its limits as the bone's, an IK target at
/// the end of each arm. See `Document::rig_robot`.
pub struct RigRobot;

impl Operator for RigRobot {
    type Params = RigRobotParams;
    fn id() -> &'static str {
        "robot.rig"
    }
    fn label() -> &'static str {
        "Make Armature from Robot"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .active
            .is_some_and(|a| doc.robot_root(a).is_some() && doc.robot_rig(a).is_none())
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(active) = doc.document().selection.active else {
            return Err(OpError::Failed("select a part of the robot".into()));
        };
        let d = doc.document_mut();
        let rig = d.rig_robot(active).map_err(OpError::Failed)?;
        d.selection.clear();
        d.selection.add(rig);
        d.selection.active = Some(rig);
        Ok(OpResult::Finished)
    }
}

// --- assembly joints --------------------------------------------------------
//
// A machine's moving parts are a robot's: a joint between two parts is the
// same joint object a robot description makes — an empty at the joint's
// origin, its Z the axis, the moving part under it — so it is posed by the
// same sliders, held to the same limits, followed by the same mimics, and
// written to the same URDF. No second kind of joint, and no solver: the
// joints make a tree, each part moving with the one it is joined to. A
// closed loop (a four-bar linkage) is not something this can hold.

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AddJointParams {
    #[param(variants("fixed", "revolute", "continuous", "prismatic"))]
    pub kind: u32,
    /// The limits: radians for a turn, metres for a slide.
    #[param(unit = "angle")]
    pub lower: f64,
    #[param(unit = "angle")]
    pub upper: f64,
    /// Put the joint at the 3D cursor, its axis the cursor's Z — a cursor
    /// put on a face or a hole's edge (Shift+S) sets where and which way.
    pub at_cursor: bool,
    /// The joint's origin, in the world, when not at the cursor.
    #[param(unit = "length")]
    pub origin: [f64; 3],
    /// Its axis, in the world, when not at the cursor.
    pub axis: [f64; 3],
}

impl Default for AddJointParams {
    fn default() -> Self {
        Self {
            kind: 1,
            lower: -std::f64::consts::PI,
            upper: std::f64::consts::PI,
            at_cursor: true,
            origin: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
        }
    }
}

/// Joins the active part to another selected one — or to the world, when
/// none is — by a joint at the cursor: the active part then turns or slides
/// about it, within its limits, from the Robot panel's slider or
/// `robot.set_joint`, and whatever is joined to it moves with it.
pub struct AddJoint;

impl Operator for AddJoint {
    type Params = AddJointParams;
    fn id() -> &'static str {
        "object.add_joint"
    }
    fn label() -> &'static str {
        "Add Joint"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection.active.is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_math::{DQuat, DTransform, DVec3};
        use kreate_scene::robot::JointKind;
        let d = doc.document();
        let part = d
            .selection
            .active
            .ok_or_else(|| OpError::Failed("select the part that moves".into()))?;
        let base = d
            .selection
            .objects
            .iter()
            .copied()
            .find(|id| *id != part && d.object(*id).is_some());
        if base.is_some_and(|b| d.is_ancestor(part, b)) {
            return Err(OpError::Failed(
                "the other part already moves with this one; join them the other way".into(),
            ));
        }
        if d.object(part)
            .and_then(|o| o.parent)
            .is_some_and(|p| joint_of(d, p).is_some())
        {
            return Err(OpError::Failed(
                "this part already has a joint; remove it first".into(),
            ));
        }
        let kind = match params.kind {
            0 => JointKind::Fixed,
            2 => JointKind::Continuous,
            3 => JointKind::Prismatic,
            _ => JointKind::Revolute,
        };
        let (origin, axis) = if params.at_cursor {
            (d.cursor, d.cursor_rotation * DVec3::Z)
        } else {
            (
                DVec3::from_array(params.origin),
                DVec3::from_array(params.axis),
            )
        };
        if !origin.is_finite() || !axis.is_finite() || axis.length_squared() < 1e-24 {
            return Err(OpError::Failed(
                "the joint needs an origin and an axis".into(),
            ));
        }
        // A slide's limits are metres. Left at the turn's default of half
        // a turn each way they were 3.14 m of travel, and a motion check
        // then stepped 17 cm at a time, over whatever was in the way.
        let pi = std::f64::consts::PI;
        let (lower, upper) = if kind == JointKind::Prismatic
            && (params.lower + pi).abs() < 1e-9
            && (params.upper - pi).abs() < 1e-9
        {
            (-0.1, 0.1)
        } else {
            (params.lower, params.upper)
        };
        let (lower, upper) = (lower.min(upper), lower.max(upper));
        let world = DTransform {
            translation: origin,
            rotation: DQuat::from_rotation_arc(DVec3::Z, axis.normalize()),
            scale: DVec3::ONE,
        };
        let part_name = d.object(part).map(|o| o.name.clone()).unwrap_or_default();
        let d = doc.document_mut();
        let name = d.unique_object_name(&format!("Joint {part_name}"));
        let mut o = kreate_scene::Object::new(name, kreate_scene::ObjectData::Empty);
        o.transform = world;
        let id = d.add_object(o);
        // Under the base where it stands, and the part under it where it
        // stands: nothing moves now.
        if let Some(b) = base {
            d.set_parent(id, Some(b))
                .map_err(|e| OpError::Failed(format!("{e:?}")))?;
        }
        d.set_parent(part, Some(id))
            .map_err(|e| OpError::Failed(format!("{e:?}")))?;
        let rest = d.object(id).map(|o| o.transform).unwrap_or(world);
        if let Some(o) = d.object_mut(id) {
            o.robot = Some(RobotPart::Joint(Joint {
                kind,
                rest,
                lower,
                upper,
                ..Joint::default()
            }));
        }
        d.selection.clear();
        d.selection.add(id);
        d.selection.active = Some(id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct MotionCheckParams {
    /// The joint's object, by name; empty for the active object.
    pub joint: String,
    /// Positions tried across its range.
    #[param(min = 2, max = 360, unit = "count")]
    pub steps: u32,
}

impl Default for MotionCheckParams {
    fn default() -> Self {
        Self {
            joint: String::new(),
            steps: 36,
        }
    }
}

/// What a motion check found: where along a joint's travel what it moves
/// runs into something else, and what.
#[derive(Clone, Debug, PartialEq)]
pub struct Collision {
    /// Radians or metres.
    pub position: f64,
    pub moving: String,
    pub other: String,
    /// Cubic metres.
    pub volume: f64,
}

/// The parts that move with `joint` and those that do not, each in the
/// world as `doc` has it posed.
fn collisions_at(doc: &Document, joint: ObjectId) -> Vec<(String, String, f64)> {
    use kreate_mesh::modifiers::boolean::boolean;
    use kreate_mesh::modifiers::BooleanOp;
    use kreate_scene::modifier::{EvalCtx, ModifierCache};
    let mut cache = ModifierCache::new();
    let parts: Vec<(ObjectId, bool, kreate_mesh::MeshData)> = doc
        .objects()
        .filter(|(id, o)| {
            o.mesh().is_some()
                && o.sketch.is_empty()
                && o.sketch_lines.is_empty()
                && doc.is_shown(*id)
                && !matches!(o.robot, Some(RobotPart::Collision))
        })
        .filter_map(|(id, _)| {
            let mesh = cache.evaluated(doc, id, &EvalCtx::still())?;
            let to = doc.world_transform(id);
            let mut m = (*mesh).clone();
            for p in m.positions_mut() {
                *p = to.transform_point3(p.as_dvec3()).as_vec3();
            }
            (m.num_faces() > 0).then(|| (id, doc.is_ancestor(joint, id), m))
        })
        .collect();
    let name = |id: ObjectId| doc.object(id).map(|o| o.name.clone()).unwrap_or_default();
    let mut out = Vec::new();
    for (a, moves_a, ma) in &parts {
        if !moves_a {
            continue;
        }
        let ba = ma.bounds();
        for (b, moves_b, mb) in &parts {
            if *moves_b {
                continue;
            }
            let bb = mb.bounds();
            if ba.min.cmpgt(bb.max).any() || bb.min.cmpgt(ba.max).any() {
                continue;
            }
            let overlap = boolean(ma, mb, BooleanOp::Intersect);
            if overlap.num_faces() == 0 {
                continue;
            }
            let v = kreate_mesh::measure::mass_properties(&overlap, |p| p)
                .volume
                .abs();
            // Less than a cubic tenth of a millimetre is touching.
            if v >= 1e-13 {
                out.push((name(*a), name(*b), v));
            }
        }
    }
    out
}

/// Every collision a joint's travel meets: the joint taken through its
/// range — its limits, a whole turn for one without, a span of its parts'
/// size each way for an unlimited slide — and at each step what moves with
/// it tried against everything that does not.
pub fn motion_collisions(doc: &Document, joint: ObjectId, steps: u32) -> Vec<Collision> {
    let Some(j) = joint_of(doc, joint) else {
        return Vec::new();
    };
    if !j.kind.moves() {
        return Vec::new();
    }
    let (lo, hi) = if j.kind.limited() && j.lower < j.upper {
        (j.lower, j.upper)
    } else if j.kind.slides() {
        (-0.1, 0.1)
    } else {
        (-std::f64::consts::PI, std::f64::consts::PI)
    };
    let steps = steps.clamp(2, 360);
    let mut scratch = doc.clone();
    let mut out = Vec::new();
    for k in 0..=steps {
        let position = lo + (hi - lo) * k as f64 / steps as f64;
        set(&mut scratch, joint, position);
        for (moving, other, volume) in collisions_at(&scratch, joint) {
            out.push(Collision {
                position,
                moving,
                other,
                volume,
            });
        }
    }
    out
}

/// Takes a joint through its travel and says where what it moves runs
/// into the rest — the first and last position of each pair's contact —
/// or that it is clear all the way. Changes nothing.
pub struct MotionCheck;

impl Operator for MotionCheck {
    type Params = MotionCheckParams;
    fn id() -> &'static str {
        "robot.motion_check"
    }
    fn label() -> &'static str {
        "Check Joint Motion"
    }
    fn poll(ctx: &Context, doc: &Document) -> bool {
        SetJoint::poll(ctx, doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let Some(id) = find_joint(d, &params.joint) else {
            return Err(OpError::Failed("select a joint to check".into()));
        };
        let joint = joint_of(d, id).expect("a joint");
        let show = |v: f64| {
            if joint.kind.slides() {
                format!("{:.1} mm", v * 1e3)
            } else {
                format!("{:.1}°", v.to_degrees())
            }
        };
        let found = motion_collisions(d, id, params.steps);
        if found.is_empty() {
            return Err(OpError::Failed(format!(
                "clear: nothing runs into anything across its travel ({} positions)",
                params.steps.clamp(2, 360) + 1
            )));
        }
        // Each pair's span of contact.
        let mut spans: Vec<(String, String, f64, f64, f64)> = Vec::new();
        for c in &found {
            match spans.iter_mut().find(|s| s.0 == c.moving && s.1 == c.other) {
                Some(s) => {
                    s.2 = s.2.min(c.position);
                    s.3 = s.3.max(c.position);
                    s.4 = s.4.max(c.volume);
                }
                None => spans.push((
                    c.moving.clone(),
                    c.other.clone(),
                    c.position,
                    c.position,
                    c.volume,
                )),
            }
        }
        let lines: Vec<String> = spans
            .iter()
            .map(|(a, b, from, to, v)| {
                format!(
                    "{a} hits {b} from {} to {} (up to {:.1} mm3)",
                    show(*from),
                    show(*to),
                    v * 1e9
                )
            })
            .collect();
        Err(OpError::Failed(lines.join("; ")))
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

    /// An arm joined to a post by a hinge at the cursor: it turns about the
    /// hinge from the slider's op, stays where it was when joined, and the
    /// motion check finds the wall it swings into — only on the side the
    /// wall is.
    #[test]
    fn a_hinged_arm_turns_about_its_joint_and_the_check_finds_the_wall() {
        use crate::operator::OperatorRegistry;
        use crate::History;
        use kreate_reflect::{PropertyMap, PropertyValue};
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let block = |doc: &mut Document, name: &str, size: DVec3, at: DVec3| {
            let mut m = kreate_mesh::primitives::cube(1.0);
            for p in m.positions_mut() {
                *p *= size.as_vec3();
            }
            let id = doc.add_mesh_object(name, m);
            doc.object_mut(id).unwrap().transform.translation = at;
            id
        };
        // A post at the origin; an arm 100 mm long reaching along +X from
        // it; a wall at +Y, 80 mm out, that the arm meets turning
        // anticlockwise but not clockwise.
        let post = block(&mut doc, "Post", DVec3::new(0.02, 0.02, 0.05), DVec3::ZERO);
        let arm = block(
            &mut doc,
            "Arm",
            DVec3::new(0.1, 0.01, 0.01),
            DVec3::new(0.06, 0.0, 0.03),
        );
        let _wall = block(
            &mut doc,
            "Wall",
            DVec3::new(0.2, 0.01, 0.1),
            DVec3::new(0.0, 0.08, 0.0),
        );
        doc.selection.clear();
        doc.selection.add(post);
        doc.selection.add(arm);
        doc.selection.active = Some(arm);
        doc.cursor = DVec3::new(0.0, 0.0, 0.03);
        let run = |r: &OperatorRegistry,
                   h: &mut History,
                   doc: &mut Document,
                   id: &str,
                   pairs: &[(&'static str, PropertyValue)]| {
            let mut p = PropertyMap::new();
            for (k, v) in pairs {
                p.set(k, v.clone());
            }
            h.dispatch(r, doc, &Context::default(), id, &p)
        };
        let before = doc.world_transform(arm);
        run(&r, &mut h, &mut doc, "object.add_joint", &[]).unwrap();
        let joint = doc.selection.active.unwrap();
        assert!(matches!(
            doc.object(joint).unwrap().robot,
            Some(RobotPart::Joint(_))
        ));
        let after = doc.world_transform(arm);
        assert!(
            (after.translation - before.translation).length() < 1e-9,
            "joining moved the arm"
        );
        // A quarter turn anticlockwise: the arm points up +Y.
        run(
            &r,
            &mut h,
            &mut doc,
            "robot.set_joint",
            &[(
                "position",
                PropertyValue::Float(std::f64::consts::FRAC_PI_2),
            )],
        )
        .unwrap();
        let tip = doc.world_transform(arm).translation;
        assert!(
            (tip - DVec3::new(0.0, 0.06, 0.03)).length() < 1e-6,
            "{tip:?}"
        );
        run(&r, &mut h, &mut doc, "robot.reset_pose", &[]).unwrap();
        let found = motion_collisions(&doc, joint, 36);
        assert!(!found.is_empty(), "the wall is in the arm's way");
        assert!(found.iter().all(|c| c.moving == "Arm" && c.other == "Wall"));
        // Only turning toward the wall, about a quarter turn either side
        // of it.
        assert!(found.iter().all(|c| c.position > 0.5 && c.position < 2.7));
        // The post the arm hangs from is not counted against it, and the
        // check left the pose as it was.
        assert!(
            doc.world_transform(arm)
                .translation
                .distance(before.translation)
                < 1e-9
        );
    }
}
