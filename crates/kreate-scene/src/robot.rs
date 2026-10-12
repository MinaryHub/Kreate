//! What an object is to a robot: a link, a joint, or a link's collision
//! shape — as a robot description (URDF) says them, kept so a robot can be
//! posed within its limits and written back out.
//!
//! # A joint is an empty that turns about its own Z
//!
//! A joint's object sits where the joint's origin is, turned so that the
//! joint's axis is its local Z, and its child link sits under it. Its
//! position — an angle for a revolute or continuous joint, a distance for a
//! prismatic one — is a turn about, or a slide along, that Z from its
//! [`Joint::rest`]. [`Joint::place`] is the one place that says how, and
//! [`Joint::position_of`] reads one back from a transform the user moved by
//! hand.

use std::collections::HashMap;

use kreate_math::{DAffine3, DMat3, DQuat, DTransform, DVec3};
use kreate_reflect::Uid;

use crate::armature::{
    self, Armature, Bone, BoneConstraint, ConstraintKind, IkConstraint, PoseBone,
};
use crate::document::{Object, ObjectData, ObjectId};
use crate::Document;

/// How a joint moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JointKind {
    #[default]
    Fixed,
    /// Turns, between its limits.
    Revolute,
    /// Turns, without limits.
    Continuous,
    /// Slides, between its limits.
    Prismatic,
    /// Moves every way: placed by hand.
    Floating,
    /// Slides in its plane: placed by hand.
    Planar,
}

impl JointKind {
    pub const ALL: [Self; 6] = [
        Self::Fixed,
        Self::Revolute,
        Self::Continuous,
        Self::Prismatic,
        Self::Floating,
        Self::Planar,
    ];

    /// URDF's name for it.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Revolute => "revolute",
            Self::Continuous => "continuous",
            Self::Prismatic => "prismatic",
            Self::Floating => "floating",
            Self::Planar => "planar",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.tag() == tag)
    }

    /// Whether it has one position to set: a turn or a slide.
    pub fn moves(self) -> bool {
        matches!(self, Self::Revolute | Self::Continuous | Self::Prismatic)
    }

    /// Whether its position is a distance rather than an angle.
    pub fn slides(self) -> bool {
        self == Self::Prismatic
    }

    /// Whether its limits hold it.
    pub fn limited(self) -> bool {
        matches!(self, Self::Revolute | Self::Prismatic)
    }
}

/// Another joint this one follows: `position = multiplier · theirs + offset`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mimic {
    /// The joint followed, by its object's name.
    pub joint: String,
    pub multiplier: f64,
    pub offset: f64,
}

/// A joint between two links.
#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    pub kind: JointKind,
    /// Where the joint's object sits at position zero, in its parent's
    /// space: the joint's origin, turned so its axis is local Z.
    pub rest: DTransform,
    /// Radians or metres.
    pub lower: f64,
    pub upper: f64,
    /// What the description says it can push and how fast: kept to be
    /// written back, not used here.
    pub effort: f64,
    pub velocity: f64,
    pub mimic: Option<Mimic>,
}

impl Default for Joint {
    fn default() -> Self {
        Self {
            kind: JointKind::Fixed,
            rest: DTransform::IDENTITY,
            lower: 0.0,
            upper: 0.0,
            effort: 0.0,
            velocity: 0.0,
            mimic: None,
        }
    }
}

impl Joint {
    /// `position` held to the limits, for a joint that has them.
    pub fn clamp(&self, position: f64) -> f64 {
        if self.kind.limited() && self.lower <= self.upper {
            position.clamp(self.lower, self.upper)
        } else {
            position
        }
    }

    /// The joint's object's transform at `position`.
    pub fn place(&self, position: f64) -> DTransform {
        let mut t = self.rest;
        match self.kind {
            JointKind::Revolute | JointKind::Continuous => {
                t.rotation = (self.rest.rotation * DQuat::from_rotation_z(position)).normalize();
            }
            JointKind::Prismatic => {
                t.translation = self.rest.translation + self.rest.rotation * (DVec3::Z * position);
            }
            _ => {}
        }
        t
    }

    /// The position a transform puts the joint at: the turn about the rest's
    /// Z (or the slide along it) that brings the rest there.
    pub fn position_of(&self, t: &DTransform) -> f64 {
        match self.kind {
            JointKind::Revolute | JointKind::Continuous => {
                let turn = self.rest.rotation.inverse() * t.rotation;
                // The twist about Z: the turn's Z part, as an angle.
                let (z, w) = (turn.z, turn.w);
                let angle = 2.0 * z.atan2(w);
                // Into (-π, π]: a turn and its negative quaternion agree.
                let pi = std::f64::consts::PI;
                (angle + pi).rem_euclid(2.0 * pi) - pi
            }
            JointKind::Prismatic => {
                (self.rest.rotation.inverse() * (t.translation - self.rest.translation)).z
            }
            _ => 0.0,
        }
    }
}

/// A link's mass, where its centre is and how it resists turning — kept to
/// be written back.
#[derive(Clone, Debug, PartialEq)]
pub struct Inertial {
    /// The centre of mass and the inertia's axes, in the link's space.
    pub origin: DTransform,
    /// Kilograms.
    pub mass: f64,
    /// ixx, ixy, ixz, iyy, iyz, izz.
    pub inertia: [f64; 6],
}

/// What an object is to a robot.
#[derive(Clone, Debug, PartialEq)]
pub enum RobotPart {
    /// A link: the frame its visuals and its child joints hang from.
    Link {
        inertial: Option<Inertial>,
    },
    Joint(Joint),
    /// A link's collision shape: a mesh, hidden, kept to be written back.
    Collision,
    /// An armature that poses the robot: a bone for each joint that moves,
    /// whose turn (or slide) is the joint's position. See
    /// [`Document::rig_robot`].
    Rig {
        joints: Vec<RigJoint>,
    },
}

/// One joint as a rig's pose puts it.
#[derive(Clone, Debug, PartialEq)]
pub struct RigReading {
    /// The joint's object.
    pub joint: ObjectId,
    /// Its bone.
    pub bone: String,
    pub kind: JointKind,
    /// Whether it mimics another joint rather than following its own bone.
    pub mimic: bool,
    /// Radians or metres, within its limits.
    pub position: f64,
    /// The joint's object's transform at that position.
    pub placed: DTransform,
}

/// One joint a rig's bone moves.
#[derive(Clone, Debug, PartialEq)]
pub struct RigJoint {
    /// The bone, by name, as a pose names it.
    pub bone: String,
    /// The joint's object, by identity: its name can change.
    pub joint: Uid,
}

/// A bone's own transform for a joint at `position`: a turn about the bone's
/// Z, or a slide along it. The rig's bones stand with Z along the joint's
/// axis, so this is the joint's own motion.
fn bone_motion(kind: JointKind, position: f64) -> DTransform {
    match kind {
        JointKind::Prismatic => DTransform {
            translation: DVec3::Z * position,
            ..DTransform::IDENTITY
        },
        _ => DTransform {
            rotation: DQuat::from_rotation_z(position),
            ..DTransform::IDENTITY
        },
    }
}

impl Document {
    /// The robot's root link at or above `id`: the topmost link on the way
    /// up.
    pub fn robot_root(&self, id: ObjectId) -> Option<ObjectId> {
        let mut at = Some(id);
        let mut root = None;
        for _ in 0..=self.object_count() {
            let Some(i) = at else { break };
            let o = self.object(i)?;
            if matches!(o.robot, Some(RobotPart::Link { .. })) {
                root = Some(i);
            }
            at = o.parent;
        }
        root
    }

    /// Where `id` stands with every joint of its robot at rest, in world
    /// space: the robot as its description draws it.
    pub fn robot_rest_world(&self, id: ObjectId) -> DAffine3 {
        let mut chain = Vec::new();
        let mut at = Some(id);
        while let Some(i) = at {
            if chain.len() > self.object_count() {
                break;
            }
            chain.push(i);
            at = self.object(i).and_then(|o| o.parent);
        }
        let mut world = DAffine3::IDENTITY;
        for i in chain.into_iter().rev() {
            let Some(o) = self.object(i) else { continue };
            let local = match &o.robot {
                Some(RobotPart::Joint(j)) => j.rest,
                _ => o.transform,
            };
            world = match o.parent {
                Some(_) => world * o.parent_inverse * local.to_affine(),
                None => local.to_affine(),
            };
        }
        world
    }

    /// The armature posing the robot `id` is part of, if it has one.
    pub fn robot_rig(&self, id: ObjectId) -> Option<ObjectId> {
        let root = self.robot_root(id)?;
        self.objects()
            .find(|(_, o)| o.parent == Some(root) && matches!(o.robot, Some(RobotPart::Rig { .. })))
            .map(|(i, _)| i)
    }

    /// Makes an armature that poses the robot `id` belongs to: a bone at each
    /// joint that moves, its head on the joint, its Z along the joint's axis,
    /// parented as the joints are; a fixed joint is no bone, its links riding
    /// on the bone above. The joints' limits become the bones' Limit Rotation
    /// (and, for a slider, Limit Location) constraints; each arm's last bone
    /// gets an IK target and an IK constraint, off until it is wanted. The
    /// bones start where the joints stand now, and from then on the bones
    /// are what the joints follow — see [`Document::sync_robot_rigs`].
    ///
    /// The robot's own objects are left where they are, linked as before:
    /// links under joints under links, which is what a URDF is written from.
    /// The armature hangs from the root link, so moving the robot moves it.
    pub fn rig_robot(&mut self, id: ObjectId) -> Result<ObjectId, String> {
        let root = self.robot_root(id).ok_or("not part of a robot")?;
        if self.robot_rig(root).is_some() {
            return Err("the robot already has an armature".into());
        }
        let root_world = self.world_transform(root);
        let to_root = root_world.inverse();

        // The joints that move, parents first, each with the nearest one above.
        struct Found {
            name: String,
            uid: Uid,
            joint: Joint,
            frame: DAffine3,
            parent: Option<usize>,
            position: f64,
        }
        let mut found: Vec<Found> = Vec::new();
        let mut index_of: HashMap<ObjectId, usize> = HashMap::new();
        for (oid, _) in self.hierarchy() {
            if oid == root || !self.is_ancestor(root, oid) {
                continue;
            }
            let Some(o) = self.object(oid) else { continue };
            let Some(RobotPart::Joint(j)) = &o.robot else {
                continue;
            };
            if !j.kind.moves() {
                continue;
            }
            let mut parent = None;
            let mut up = o.parent;
            while let Some(u) = up.filter(|u| *u != root) {
                if let Some(k) = index_of.get(&u) {
                    parent = Some(*k);
                    break;
                }
                up = self.object(u).and_then(|x| x.parent);
            }
            index_of.insert(oid, found.len());
            found.push(Found {
                name: o.name.clone(),
                uid: o.uid,
                joint: j.clone(),
                frame: to_root * self.robot_rest_world(oid),
                parent,
                position: j.clamp(j.position_of(&o.transform)),
            });
        }
        if found.is_empty() {
            return Err("the robot has no joints that move".into());
        }

        // Each joint's parts at rest, in the root's space: where a bone with
        // no joint below it points.
        let mut part_lo = vec![DVec3::splat(f64::MAX); found.len()];
        let mut part_hi = vec![DVec3::splat(f64::MIN); found.len()];
        let (mut all_lo, mut all_hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
        for (oid, o) in self.objects() {
            let Some(mesh) = o.mesh().and_then(|m| self.mesh(m)) else {
                continue;
            };
            if !o.visible || !self.is_ancestor(root, oid) {
                continue;
            }
            let b = mesh.bounds();
            if b.is_empty() {
                continue;
            }
            let at = to_root * self.robot_rest_world(oid);
            let mut owner = None;
            let mut up = Some(oid);
            while let Some(u) = up {
                if let Some(k) = index_of.get(&u) {
                    owner = Some(*k);
                    break;
                }
                up = self.object(u).and_then(|x| x.parent);
            }
            for corner in 0..8 {
                let c = DVec3::new(
                    if corner & 1 == 0 { b.min.x } else { b.max.x } as f64,
                    if corner & 2 == 0 { b.min.y } else { b.max.y } as f64,
                    if corner & 4 == 0 { b.min.z } else { b.max.z } as f64,
                );
                let p = at.transform_point3(c);
                all_lo = all_lo.min(p);
                all_hi = all_hi.max(p);
                if let Some(k) = owner {
                    part_lo[k] = part_lo[k].min(p);
                    part_hi[k] = part_hi[k].max(p);
                }
            }
        }
        let size = if all_hi.x >= all_lo.x {
            (all_hi - all_lo).length().max(1e-3)
        } else {
            1.0
        };

        let mut rig = Armature::new(format!(
            "{} Rig",
            self.object(root)
                .map(|o| o.name.as_str())
                .unwrap_or("Robot")
        ));
        rig.bones.clear();
        rig.in_front = true;
        for (k, f) in found.iter().enumerate() {
            let head = f.frame.translation;
            let axis = f.frame.matrix3.mul_vec3(DVec3::Z).normalize_or_zero();
            let flat = |v: DVec3| v - axis * v.dot(axis);
            // Towards the farthest joint below, seen across the axis; for the
            // last joint of an arm, towards its parts.
            let toward = found
                .iter()
                .filter(|c| c.parent == Some(k))
                .map(|c| c.frame.translation)
                .max_by(|a, b| {
                    flat(*a - head)
                        .length_squared()
                        .total_cmp(&flat(*b - head).length_squared())
                })
                .or_else(|| {
                    (part_hi[k].x >= part_lo[k].x).then(|| (part_lo[k] + part_hi[k]) * 0.5)
                });
            let (dir, length) = match toward {
                Some(p) if flat(p - head).length() > size * 1e-3 => (
                    flat(p - head).normalize(),
                    (p - head).length().max(size * 0.02),
                ),
                _ => (
                    flat(f.frame.matrix3.mul_vec3(DVec3::Y))
                        .try_normalize()
                        .unwrap_or_else(|| axis.any_orthonormal_vector()),
                    size * 0.05,
                ),
            };
            // The roll that makes the bone's Z the joint's axis.
            let want = DQuat::from_mat3(&DMat3::from_cols(dir.cross(axis), dir, axis)).normalize();
            let align = if dir.dot(DVec3::Y) < -1.0 + 1e-12 {
                DQuat::from_rotation_z(std::f64::consts::PI)
            } else {
                DQuat::from_rotation_arc(DVec3::Y, dir)
            };
            let rem = align.inverse() * want;
            let roll = 2.0 * rem.y.atan2(rem.w);
            rig.add_bone(Bone {
                name: f.name.clone(),
                parent: f.parent,
                head,
                tail: head + dir * length,
                roll,
                ..Bone::default()
            });
        }

        // The pose the robot stands in now, and the joints' limits.
        let mut pose = armature::Pose::default();
        for f in &found {
            pose.bones.push(PoseBone {
                name: f.name.clone(),
                transform: bone_motion(f.joint.kind, f.position),
                selected: false,
            });
            let held = DVec3::ZERO;
            let limit = |min: DVec3, max: DVec3| ConstraintKind::LimitRotation { min, max };
            let mut add = |kind| {
                pose.constraints.push(BoneConstraint {
                    bone: f.name.clone(),
                    kind,
                    target: String::new(),
                    influence: 1.0,
                    enabled: true,
                })
            };
            match f.joint.kind {
                JointKind::Revolute => add(limit(
                    DVec3::new(0.0, 0.0, f.joint.lower.min(f.joint.upper)),
                    DVec3::new(0.0, 0.0, f.joint.upper.max(f.joint.lower)),
                )),
                JointKind::Continuous => {
                    let pi = std::f64::consts::PI;
                    add(limit(DVec3::new(0.0, 0.0, -pi), DVec3::new(0.0, 0.0, pi)))
                }
                JointKind::Prismatic => {
                    add(limit(held, held));
                    add(ConstraintKind::LimitLocation {
                        min: DVec3::new(0.0, 0.0, f.joint.lower.min(f.joint.upper)),
                        max: DVec3::new(0.0, 0.0, f.joint.upper.max(f.joint.lower)),
                    });
                }
                _ => {}
            }
        }

        // An IK target at the end of each arm. The arm ends at a joint that
        // turns with none below it that does — a wrist, whatever gripper
        // slides on it — and the arm is the run up from there to a branch.
        // Its effector is a bone of its own, held fixed on the wrist and
        // reaching to the middle of what the wrist carries: a bone's own
        // length has to lie across its joint's axis, and a wrist's hand lies
        // along it. The IK is off: the bones turn by hand until it is on.
        let turns = |k: usize| {
            matches!(
                found[k].joint.kind,
                JointKind::Revolute | JointKind::Continuous
            ) && found[k].joint.mimic.is_none()
        };
        let descends = |d: usize, a: usize| {
            let mut at = Some(d);
            for _ in 0..=found.len() {
                match at {
                    Some(x) if x == a => return true,
                    Some(x) => at = found[x].parent,
                    None => return false,
                }
            }
            false
        };
        let carries_turn = |k: usize| (0..found.len()).any(|d| descends(d, k) && turns(d));
        let branches = |p: usize| {
            (0..found.len())
                .filter(|c| found[*c].parent == Some(p) && carries_turn(*c))
                .count()
        };
        let posed = armature::posed(&rig, &pose);
        let mut arms = Vec::new();
        for e in 0..found.len() {
            if !turns(e) || (0..found.len()).any(|d| d != e && descends(d, e) && turns(d)) {
                continue;
            }
            let mut chain = 1u32;
            let mut at = e;
            while let Some(p) = found[at].parent {
                if branches(p) > 1 {
                    break;
                }
                chain += 1;
                at = p;
            }
            if chain < 2 {
                continue;
            }
            let (mut lo, mut hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
            for d in (0..found.len()).filter(|d| descends(*d, e)) {
                lo = lo.min(part_lo[d]);
                hi = hi.max(part_hi[d]);
            }
            let head = found[e].frame.translation;
            let axis = found[e]
                .frame
                .matrix3
                .mul_vec3(DVec3::Z)
                .normalize_or_zero();
            let point = Some((lo + hi) * 0.5)
                .filter(|p| hi.x >= lo.x && p.distance(head) > size * 1e-3)
                .unwrap_or(head + axis * size * 0.05);
            arms.push((e, chain, head, point));
        }
        for (e, chain, head, point) in arms {
            let end = found[e].name.clone();
            let tip = rig.unique_name(&format!("{end} tip"));
            rig.add_bone(Bone {
                name: tip.clone(),
                parent: Some(e),
                head,
                tail: point,
                ..Bone::default()
            });
            pose.constraints.push(BoneConstraint {
                bone: tip.clone(),
                kind: ConstraintKind::LimitRotation {
                    min: DVec3::ZERO,
                    max: DVec3::ZERO,
                },
                target: String::new(),
                influence: 1.0,
                enabled: true,
            });
            // The target where the tip is now, the robot as it stands.
            let at = (posed[e] * rig.bones[e].rest().inverse()).transform_point3(point);
            let target = rig.unique_name(&format!("{end} IK"));
            rig.add_bone(Bone {
                name: target.clone(),
                parent: None,
                head: at,
                tail: at + DVec3::Z * size * 0.03,
                ..Bone::default()
            });
            pose.ik.push(IkConstraint {
                bone: tip,
                target,
                chain: chain + 1,
                enabled: false,
                pole: String::new(),
            });
        }

        let joints: Vec<RigJoint> = found
            .iter()
            .map(|f| RigJoint {
                bone: f.name.clone(),
                joint: f.uid,
            })
            .collect();
        let rig_name = rig.name.clone();
        let data = self.add_armature(rig);
        let mut object = Object::new(rig_name, ObjectData::Armature(data));
        object.transform = DTransform::from_affine(root_world);
        object.pose = pose;
        object.robot = Some(RobotPart::Rig { joints });
        let made = self.add_object(object);
        self.set_parent(made, Some(root))
            .map_err(|_| "the armature could not hang from the robot".to_string())?;
        Ok(made)
    }

    /// The joints rig `rig` moves, read from `pose`: each joint's position
    /// from its bone as posed — IK and limits included — and a joint that
    /// mimics another from that one's. What [`Document::sync_robot_rigs`]
    /// writes, and what a file's animation samples frame by frame.
    pub fn rig_readings(&self, rig: ObjectId, pose: &armature::Pose) -> Vec<RigReading> {
        let (Some((_, armature)), Some(RobotPart::Rig { joints })) = (
            self.armature_of(rig),
            self.object(rig).and_then(|o| o.robot.as_ref()),
        ) else {
            return Vec::new();
        };
        let frames = armature::posed(armature, pose);
        let mut found: Vec<(RigReading, String, Joint)> = Vec::new();
        for rj in joints {
            let (Some(id), Some(i)) = (self.object_by_uid(rj.joint), armature.bone_index(&rj.bone))
            else {
                continue;
            };
            let Some(o) = self.object(id) else { continue };
            let Some(RobotPart::Joint(j)) = &o.robot else {
                continue;
            };
            let local = armature::pose_base(armature, &frames, i).inverse() * frames[i];
            let position = match j.kind {
                JointKind::Prismatic => local.translation.z,
                _ => armature::twist_about_z(local.to_scale_rotation_translation().1),
            };
            let position = j.clamp(position);
            found.push((
                RigReading {
                    joint: id,
                    bone: rj.bone.clone(),
                    kind: j.kind,
                    mimic: j.mimic.is_some(),
                    position,
                    placed: j.place(position),
                },
                o.name.clone(),
                j.clone(),
            ));
        }
        // Mimics from what they follow — a chain of them settles in as many
        // passes as it is long.
        for _ in 0..found.len().max(1) {
            let mut changed = false;
            for k in 0..found.len() {
                let Some(m) = found[k].2.mimic.clone() else {
                    continue;
                };
                let leader = found
                    .iter()
                    .find(|f| f.1 == m.joint)
                    .map(|f| f.0.position)
                    .or_else(|| {
                        let (_, o) = self.objects().find(|(_, o)| o.name == m.joint)?;
                        match &o.robot {
                            Some(RobotPart::Joint(j)) => Some(j.position_of(&o.transform)),
                            _ => None,
                        }
                    });
                let Some(leader) = leader else { continue };
                let want = found[k].2.clamp(m.multiplier * leader + m.offset);
                if (want - found[k].0.position).abs() > 1e-12 {
                    found[k].0.position = want;
                    found[k].0.placed = found[k].2.place(want);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        found.into_iter().map(|f| f.0).collect()
    }

    /// Every rigged robot's joints put where its bones are (see
    /// [`Document::rig_readings`]), a mimicking joint's bone turned to match.
    /// Not an edit, like the drivers: what the pose says, made true. Returns
    /// how many objects moved.
    pub fn sync_robot_rigs(&mut self) -> usize {
        let rigs: Vec<ObjectId> = self
            .objects()
            .filter(|(_, o)| matches!(o.robot, Some(RobotPart::Rig { .. })))
            .map(|(id, _)| id)
            .collect();
        let mut moved = 0;
        for rig in rigs {
            let Some(mut pose) = self.object(rig).map(|o| o.pose.clone()) else {
                continue;
            };
            let readings = self.rig_readings(rig, &pose);
            let mut pose_changed = false;
            for r in &readings {
                if r.mimic {
                    let motion = bone_motion(r.kind, r.position);
                    let entry = pose.entry(&r.bone);
                    if !close(&entry.transform, &motion) {
                        entry.transform = motion;
                        pose_changed = true;
                    }
                }
                if let Some(o) = self.object_untouched_mut(r.joint) {
                    if !close(&o.transform, &r.placed) {
                        o.transform = r.placed;
                        moved += 1;
                    }
                }
            }
            if pose_changed {
                if let Some(o) = self.object_untouched_mut(rig) {
                    o.pose = pose;
                    moved += 1;
                }
            }
        }
        if moved > 0 {
            self.touch_version();
        }
        moved
    }

    /// Puts the rig's bone for `joint` (if a rig poses it) at `position`, so
    /// setting a joint by its number and turning its bone agree. Returns
    /// whether a bone moved.
    pub fn hold_joint_in_rig(&mut self, joint: ObjectId, position: f64) -> bool {
        let (Some(rig), Some(uid)) = (self.robot_rig(joint), self.object_uid(joint)) else {
            return false;
        };
        let kind = match self.object(joint).and_then(|o| o.robot.as_ref()) {
            Some(RobotPart::Joint(j)) => j.kind,
            _ => return false,
        };
        let bone = match self.object(rig).and_then(|o| o.robot.as_ref()) {
            Some(RobotPart::Rig { joints }) => joints
                .iter()
                .find(|j| j.joint == uid)
                .map(|j| j.bone.clone()),
            _ => None,
        };
        let Some(bone) = bone else { return false };
        let motion = bone_motion(kind, position);
        let Some(o) = self.object_mut(rig) else {
            return false;
        };
        let entry = o.pose.entry(&bone);
        if close(&entry.transform, &motion) {
            return false;
        }
        entry.transform = motion;
        true
    }
}

/// Two transforms the same but for rounding.
fn close(a: &DTransform, b: &DTransform) -> bool {
    a.translation.distance(b.translation) < 1e-12
        && a.rotation.dot(b.rotation).abs() > 1.0 - 1e-14
        && a.scale.distance(b.scale) < 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm() -> Joint {
        Joint {
            kind: JointKind::Revolute,
            rest: DTransform {
                translation: DVec3::new(0.0, 0.0, 1.0),
                rotation: DQuat::from_rotation_x(0.7),
                scale: DVec3::ONE,
            },
            lower: -1.0,
            upper: 2.0,
            ..Joint::default()
        }
    }

    #[test]
    fn a_turned_joint_reads_back_the_turn_it_was_given() {
        let j = arm();
        for a in [-1.0, 0.0, 0.5, 2.0] {
            let t = j.place(a);
            assert!((j.position_of(&t) - a).abs() < 1e-9, "{a}");
            assert_eq!(t.translation, j.rest.translation);
        }
        assert_eq!(j.clamp(3.0), 2.0);
        assert_eq!(j.clamp(-3.0), -1.0);
    }

    #[test]
    fn a_sliding_joint_slides_along_its_own_z() {
        let j = Joint {
            kind: JointKind::Prismatic,
            ..arm()
        };
        let t = j.place(0.25);
        let along = j.rest.rotation * DVec3::Z;
        assert!((t.translation - (j.rest.translation + along * 0.25)).length() < 1e-12);
        assert!((j.position_of(&t) - 0.25).abs() < 1e-12);
        assert_eq!(JointKind::from_tag("prismatic"), Some(JointKind::Prismatic));
    }

    #[test]
    fn a_continuous_joint_is_not_held() {
        let j = Joint {
            kind: JointKind::Continuous,
            ..arm()
        };
        assert_eq!(j.clamp(10.0), 10.0);
    }
}
