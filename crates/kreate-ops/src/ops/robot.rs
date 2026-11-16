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
    registry.register::<AssemblyJoint>();
    registry.register::<MotionLink>();
    registry.register::<Contact>();
    registry.register::<Explode>();
    registry.register::<NewComponent>();
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

/// Puts one of a joint's ways of moving — `index` in its kind's
/// [`kreate_scene::robot::JointKind::dofs`] — at `position`, held to its
/// limits, the others left where they are. The first of a one-way joint is
/// [`set`], with its mimics and its rig.
pub fn set_dof(doc: &mut Document, id: ObjectId, index: usize, position: f64) -> bool {
    let Some(joint) = joint_of(doc, id) else {
        return false;
    };
    if joint.kind.moves() && index == 0 {
        return set(doc, id, position);
    }
    let Some(o) = doc.object(id) else {
        return false;
    };
    let mut positions = joint.positions_of(&o.transform);
    let Some(slot) = positions.get_mut(index) else {
        return false;
    };
    *slot = joint.clamp_at(index, position);
    let placed = joint.place_all(&positions);
    match doc.object_mut(id) {
        Some(o) if o.transform != placed => {
            o.transform = placed;
            true
        }
        _ => false,
    }
}

/// Where a joint with contact on can go toward `position` before what it
/// moves goes into another part: as far as it can, found by halving the
/// way from where it is; `position` itself for a joint without contact,
/// or with nothing in the way.
pub fn stopped_by_contact(doc: &Document, id: ObjectId, index: usize, position: f64) -> f64 {
    let Some(joint) = joint_of(doc, id).filter(|j| j.contact) else {
        return position;
    };
    let Some(o) = doc.object(id) else {
        return position;
    };
    let from = joint
        .positions_of(&o.transform)
        .get(index)
        .copied()
        .unwrap_or(0.0);
    let at = |p: f64| {
        let mut scratch = doc.clone();
        set_dof(&mut scratch, id, index, p);
        collisions_at(&scratch, id).is_empty()
    };
    // Already in something where it is: let it move, out or further in;
    // contact holds a part out, it does not trap one.
    if !at(from) {
        return position;
    }
    // Along the way, not only at its end: a slide long enough goes
    // through a wall and out the far side, clear at both ends.
    const STEPS: usize = 32;
    let mut free = from;
    let mut hit = None;
    for k in 1..=STEPS {
        let p = from + (position - from) * k as f64 / STEPS as f64;
        if at(p) {
            free = p;
        } else {
            hit = Some(p);
            break;
        }
    }
    let Some(mut hit) = hit else {
        return position;
    };
    for _ in 0..10 {
        let mid = (free + hit) * 0.5;
        if at(mid) {
            free = mid;
        } else {
            hit = mid;
        }
    }
    free
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
    /// Which of its ways of moving, for a joint that has more than one: a
    /// cylindrical joint's turn (0) or slide (1), a planar one's X, Y and
    /// turn, a ball's three turns.
    pub index: u32,
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
        let position =
            stopped_by_contact(doc.document(), id, params.index as usize, params.position);
        if set_dof(doc.document_mut(), id, params.index as usize, position) {
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
    #[param(variants(
        "fixed",
        "revolute",
        "continuous",
        "prismatic",
        "cylindrical",
        "pin_slot",
        "planar",
        "ball"
    ))]
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
        let kind = add_joint_kind(params.kind);
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
        let world = DTransform {
            translation: origin,
            rotation: DQuat::from_rotation_arc(DVec3::Z, axis.normalize()),
            scale: DVec3::ONE,
        };
        join_at(
            doc.document_mut(),
            part,
            base,
            world,
            kind,
            (params.lower, params.upper),
        )?;
        Ok(OpResult::Finished)
    }
}

/// `object.add_joint`'s kinds, by their number.
fn add_joint_kind(kind: u32) -> kreate_scene::robot::JointKind {
    use kreate_scene::robot::JointKind;
    match kind {
        0 => JointKind::Fixed,
        2 => JointKind::Continuous,
        3 => JointKind::Prismatic,
        4 => JointKind::Cylindrical,
        5 => JointKind::PinSlot,
        6 => JointKind::Planar,
        7 => JointKind::Ball,
        _ => JointKind::Revolute,
    }
}

/// A joint's object made at `world` — its origin, its Z the axis — under
/// `base` where it stands, with `part` under it where that stands: nothing
/// moves, and from now on `part` turns or slides about it. Selected.
fn join_at(
    d: &mut Document,
    part: ObjectId,
    base: Option<ObjectId>,
    world: kreate_math::DTransform,
    kind: kreate_scene::robot::JointKind,
    (lower, upper): (f64, f64),
) -> Result<ObjectId, OpError> {
    use kreate_scene::robot::JointKind;
    // A slide's limits are metres. Left at the turn's default of half
    // a turn each way they were 3.14 m of travel, and a motion check
    // then stepped 17 cm at a time, over whatever was in the way.
    let pi = std::f64::consts::PI;
    let (lower, upper) =
        if kind == JointKind::Prismatic && (lower + pi).abs() < 1e-9 && (upper - pi).abs() < 1e-9 {
            (-0.1, 0.1)
        } else {
            (lower, upper)
        };
    let (lower, upper) = (lower.min(upper), lower.max(upper));
    let part_name = d.object(part).map(|o| o.name.clone()).unwrap_or_default();
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
            // A cylindrical joint's slide and a pin-slot's, free until
            // limited.
            ..Joint::default()
        }));
    }
    d.selection.clear();
    d.selection.add(id);
    d.selection.active = Some(id);
    Ok(id)
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct AssemblyJointParams {
    /// The part that moves, by its uid.
    pub uid: String,
    /// Where on it the joint is — a face's middle, a hole's centre — and
    /// which way that faces, in the world.
    #[param(unit = "length")]
    pub from: [f64; 3],
    pub from_normal: [f64; 3],
    /// The same on the part it is joined to.
    #[param(unit = "length")]
    pub to: [f64; 3],
    pub to_normal: [f64; 3],
    /// The part it is joined to, by its uid; empty for the world — the
    /// part is then grounded through the joint.
    pub target: String,
    /// Rigid holds it; revolute turns it about the axis; slider slides it
    /// along; cylindrical does both; pin-slot turns it and slides it across;
    /// planar slides it in the plane and turns it; ball turns it every way.
    #[param(variants(
        "rigid",
        "revolute",
        "slider",
        "cylindrical",
        "pin_slot",
        "planar",
        "ball"
    ))]
    pub kind: u32,
    /// The moving part the other way round on the axis: its face looking
    /// the same way as the other's rather than at it.
    pub flip: bool,
    /// A gap left along the axis, and a turn about it, from where the
    /// points meet.
    #[param(unit = "length")]
    pub offset: f64,
    #[param(unit = "angle")]
    pub angle: f64,
    /// The first way it moves held between these: radians for a turn,
    /// metres for a slide; equal for none.
    pub lower: f64,
    pub upper: f64,
}

impl Default for AssemblyJointParams {
    fn default() -> Self {
        Self {
            uid: String::new(),
            from: [0.0; 3],
            from_normal: [0.0, 0.0, 1.0],
            to: [0.0; 3],
            to_normal: [0.0, 0.0, 1.0],
            target: String::new(),
            kind: 1,
            flip: false,
            offset: 0.0,
            angle: 0.0,
            lower: 0.0,
            upper: 0.0,
        }
    }
}

/// Fusion's Joint: a point of one part — a face's middle, a hole's centre
/// — brought onto a point of another, the faces against each other, and
/// joined there in the motion asked for. The joint's axis is the second
/// face's normal; the part turns about it, slides along it, or is held.
pub struct AssemblyJoint;

impl Operator for AssemblyJoint {
    type Params = AssemblyJointParams;
    fn id() -> &'static str {
        "assembly.joint"
    }
    fn label() -> &'static str {
        "Joint"
    }
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_math::{DAffine3, DQuat, DTransform, DVec3};
        use kreate_scene::robot::JointKind;
        let d = doc.document();
        let by_uid = |text: &str| {
            text.trim()
                .parse::<u64>()
                .ok()
                .and_then(|raw| d.object_by_uid(kreate_scene::Uid::from_raw(raw)))
        };
        let part = by_uid(&params.uid).ok_or_else(|| OpError::Failed("no such part".into()))?;
        let base = by_uid(&params.target).filter(|b| *b != part);
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
        let (from, to) = (DVec3::from_array(params.from), DVec3::from_array(params.to));
        let nf = DVec3::from_array(params.from_normal).normalize_or_zero();
        let nt = DVec3::from_array(params.to_normal).normalize_or_zero();
        if nf == DVec3::ZERO || nt == DVec3::ZERO || !from.is_finite() || !to.is_finite() {
            return Err(OpError::Failed(
                "a joint needs two points and their faces' directions".into(),
            ));
        }
        // The moving face turned to look back at the other — or the same
        // way, flipped — then about the axis by the angle; its point onto
        // the other's, the gap away along the axis.
        let facing = if params.flip { nt } else { -nt };
        let turn = DQuat::from_axis_angle(nt, params.angle) * DQuat::from_rotation_arc(nf, facing);
        let origin = to + nt * params.offset;
        let motion = DAffine3::from_translation(origin)
            * DAffine3::from_quat(turn)
            * DAffine3::from_translation(-from);
        let object = d.object(part).expect("found");
        let above = d.world_transform(part) * object.transform.to_affine().inverse();
        let local = above.inverse() * motion * above * object.transform.to_affine();
        let (scale, rotation, translation) = local.to_scale_rotation_translation();
        if !translation.is_finite() || !rotation.is_finite() {
            return Err(OpError::Failed(
                "the parts cannot be put together that way".into(),
            ));
        }
        let kind = match params.kind {
            0 => JointKind::Fixed,
            2 => JointKind::Prismatic,
            3 => JointKind::Cylindrical,
            4 => JointKind::PinSlot,
            5 => JointKind::Planar,
            6 => JointKind::Ball,
            _ => JointKind::Revolute,
        };
        let d = doc.document_mut();
        {
            let o = d.object_mut(part).expect("found");
            o.transform.translation = translation;
            o.transform.rotation = rotation.normalize();
            o.transform.scale = scale;
            // A face-to-face mate is the joint's to keep now.
            o.mate = None;
        }
        // The joint's frame: its Z the axis, its X along the part's own
        // first direction across it, so a slot's slide is the part's way.
        let world = DTransform {
            translation: origin,
            rotation: DQuat::from_rotation_arc(DVec3::Z, nt),
            scale: DVec3::ONE,
        };
        // A revolute joint with no limits given turns all the way round.
        let kind = if kind == JointKind::Revolute && params.lower == params.upper {
            JointKind::Continuous
        } else {
            kind
        };
        join_at(d, part, base, world, kind, (params.lower, params.upper))?;
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

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct MotionLinkParams {
    /// The joint that drives, and the one that follows it, by name: empty
    /// for the selection — the other selected joint drives the active one.
    pub driver: String,
    pub driven: String,
    /// How far the follower goes for each of the driver's: a gear pair's
    /// teeth over teeth, a screw's pitch over a turn. Nought takes the link
    /// off.
    pub ratio: f64,
    /// Where the follower is when the driver is at nought.
    pub offset: f64,
}

impl Default for MotionLinkParams {
    fn default() -> Self {
        Self {
            driver: String::new(),
            driven: String::new(),
            ratio: 1.0,
            offset: 0.0,
        }
    }
}

/// Fusion's Motion Link: one joint follows another — a gear its mate,
/// a slide the screw that drives it. Moving the driver moves the follower
/// by the ratio; the follower is not moved by hand while it follows.
pub struct MotionLink;

impl Operator for MotionLink {
    type Params = MotionLinkParams;
    fn id() -> &'static str {
        "assembly.motion_link"
    }
    fn label() -> &'static str {
        "Motion Link"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.objects()
            .filter(|(_, o)| matches!(o.robot, Some(RobotPart::Joint(_))))
            .count()
            >= 2
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let d = doc.document();
        let named = |name: &str| {
            d.objects()
                .find(|(id, o)| o.name == name && joint_of(d, *id).is_some())
                .map(|(id, _)| id)
        };
        let driven = if params.driven.is_empty() {
            d.selection.active.filter(|id| joint_of(d, *id).is_some())
        } else {
            named(&params.driven)
        }
        .ok_or_else(|| OpError::Failed("select the joint that follows last".into()))?;
        let driver = if params.driver.is_empty() {
            d.selection
                .objects
                .iter()
                .copied()
                .find(|id| *id != driven && joint_of(d, *id).is_some())
        } else {
            named(&params.driver)
        }
        .filter(|id| *id != driven)
        .ok_or_else(|| {
            OpError::Failed("select the joint that drives, then the one that follows".into())
        })?;
        let follower = joint_of(d, driven).expect("a joint");
        if !follower.kind.moves() {
            return Err(OpError::Failed(
                "the follower has to be a joint that turns or slides one way".into(),
            ));
        }
        // A loop of links would have each move the other for ever.
        let driver_name = d.object(driver).map(|o| o.name.clone()).unwrap_or_default();
        let driven_name = d.object(driven).map(|o| o.name.clone()).unwrap_or_default();
        let mut at = driver_name.clone();
        for _ in 0..1000 {
            let Some(next) = d
                .objects()
                .find(|(_, o)| o.name == at)
                .and_then(|(id, _)| joint_of(d, id))
                .and_then(|j| j.mimic.map(|m| m.joint))
            else {
                break;
            };
            if next == driven_name {
                return Err(OpError::Failed(
                    "the driver already follows this joint: a loop of links".into(),
                ));
            }
            at = next;
        }
        let link =
            (params.ratio != 0.0 && params.ratio.is_finite()).then(|| kreate_scene::robot::Mimic {
                joint: driver_name,
                multiplier: params.ratio,
                offset: if params.offset.is_finite() {
                    params.offset
                } else {
                    0.0
                },
            });
        let d = doc.document_mut();
        if let Some(Some(RobotPart::Joint(j))) = d.object_mut(driven).map(|o| o.robot.as_mut()) {
            j.mimic = link;
        }
        // The follower where the driver now says.
        if let Some(position) = joint_of(d, driver).and_then(|j| {
            d.object(driver)
                .map(|o| j.positions_of(&o.transform).first().copied())
        }) {
            set(d, driver, position.unwrap_or(0.0));
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ContactParams {
    /// The joint, by name; empty for the active one.
    pub joint: String,
    /// Turn it over, on, or off: on, what the joint moves stops where it
    /// would go into another part.
    #[param(variants("toggle", "on", "off"))]
    pub set: u32,
}

/// Fusion's contact sets, by the joint: with contact on, what a joint
/// moves goes only as far as touching what is in its way.
pub struct Contact;

impl Operator for Contact {
    type Params = ContactParams;
    fn id() -> &'static str {
        "assembly.contact"
    }
    fn label() -> &'static str {
        "Contact"
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
        let id = find_joint(doc.document(), &params.joint)
            .ok_or_else(|| OpError::Failed("select a joint".into()))?;
        match doc
            .document_mut()
            .object_mut(id)
            .and_then(|o| o.robot.as_mut())
        {
            Some(RobotPart::Joint(j)) => {
                let on = match params.set {
                    1 => true,
                    2 => false,
                    _ => !j.contact,
                };
                if j.contact == on {
                    return Ok(OpResult::Cancelled);
                }
                j.contact = on;
                Ok(OpResult::Finished)
            }
            _ => Ok(OpResult::Cancelled),
        }
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ExplodeParams {
    /// How far apart: each part moved out from the assembly's middle by
    /// this much of how far it already is from it.
    #[param(min = 0.0, soft_max = 3.0)]
    pub amount: f64,
}

impl Default for ExplodeParams {
    fn default() -> Self {
        Self { amount: 0.6 }
    }
}

/// An exploded view: the selected parts — every part shown, with fewer
/// than two selected — moved out from their middle, each along the line
/// from it, so how they go together can be seen. Undo puts them back.
pub struct Explode;

impl Operator for Explode {
    type Params = ExplodeParams;
    fn id() -> &'static str {
        "assembly.explode"
    }
    fn label() -> &'static str {
        "Explode"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.objects().filter(|(_, o)| o.mesh().is_some()).count() >= 2
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_math::DVec3;
        let d = doc.document();
        let part = |id: ObjectId| {
            d.object(id).is_some_and(|o| {
                o.mesh().is_some()
                    && o.drawing.is_none()
                    && o.sketch_lines.is_empty()
                    && !matches!(o.robot, Some(RobotPart::Collision))
            }) && d.is_shown(id)
        };
        let chosen: Vec<ObjectId> = d
            .selection
            .objects
            .iter()
            .copied()
            .filter(|id| part(*id))
            .collect();
        let parts: Vec<ObjectId> = if chosen.len() >= 2 {
            chosen
        } else {
            d.objects()
                .map(|(id, _)| id)
                .filter(|id| part(*id))
                .collect()
        };
        if parts.len() < 2 {
            return Err(OpError::Failed(
                "an exploded view needs two parts or more".into(),
            ));
        }
        // Each part's middle, in the world, as it shows.
        let middles: Vec<(ObjectId, DVec3)> = parts
            .iter()
            .filter_map(|id| {
                let mesh = kreate_scene::modifier::shown_base(d, *id)?;
                if mesh.num_verts() == 0 {
                    return None;
                }
                let b = mesh.bounds();
                let c = (b.min + b.max).as_dvec3() * 0.5;
                Some((*id, d.world_transform(*id).transform_point3(c)))
            })
            .collect();
        let centre = middles.iter().map(|(_, c)| *c).sum::<DVec3>() / middles.len().max(1) as f64;
        // In each one's own parent's space, before anything is moved: a
        // part under another that moves is moved by its own amount only.
        let moves: Vec<(ObjectId, DVec3)> = middles
            .iter()
            .map(|(id, c)| {
                let delta = (*c - centre) * params.amount;
                let above = d.world_transform(*id)
                    * d.object(*id)
                        .expect("a part")
                        .transform
                        .to_affine()
                        .inverse();
                (*id, above.inverse().transform_vector3(delta))
            })
            .collect();
        // A child's own move less what its parent's moves already gave it.
        let given: std::collections::HashMap<ObjectId, DVec3> = middles
            .iter()
            .map(|(id, c)| (*id, (*c - centre) * params.amount))
            .collect();
        let d = doc.document_mut();
        for (id, local) in moves {
            let mut inherited = DVec3::ZERO;
            let mut up = d.object(id).and_then(|o| o.parent);
            while let Some(p) = up {
                if let Some(m) = given.get(&p) {
                    inherited += *m;
                }
                up = d.object(p).and_then(|o| o.parent);
            }
            let above = d.world_transform(id)
                * d.object(id)
                    .expect("a part")
                    .transform
                    .to_affine()
                    .inverse();
            let extra = above.inverse().transform_vector3(inherited);
            if let Some(o) = d.object_mut(id) {
                o.transform.translation += local - extra;
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct NewComponentParams {
    /// Its name; "Component" when none is given.
    pub name: String,
}

/// Fusion's New Component from bodies: the selected parts gathered under
/// one object at their middle — moved, hidden, duplicated and listed as
/// one, each kept where it is.
pub struct NewComponent;

impl Operator for NewComponent {
    type Params = NewComponentParams;
    fn id() -> &'static str {
        "assembly.component"
    }
    fn label() -> &'static str {
        "New Component"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.objects.is_empty()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        use kreate_math::DVec3;
        let d = doc.document();
        // The selected ones nothing else selected is above: the others go
        // with them.
        let picked: Vec<ObjectId> = d.selection.objects.clone();
        let tops: Vec<ObjectId> = picked
            .iter()
            .copied()
            .filter(|id| !picked.iter().any(|p| p != id && d.is_ancestor(*p, *id)))
            .collect();
        if tops.is_empty() {
            return Err(OpError::Failed("select the parts to gather".into()));
        }
        let middle = tops
            .iter()
            .map(|id| d.world_transform(*id).translation)
            .sum::<DVec3>()
            / tops.len() as f64;
        // Under what they were all under, if one thing: a component in a
        // component stays where it was in the tree.
        let parent = d.object(tops[0]).and_then(|o| o.parent);
        let shared = tops
            .iter()
            .all(|id| d.object(*id).and_then(|o| o.parent) == parent)
            .then_some(parent)
            .flatten();
        let name = if params.name.trim().is_empty() {
            "Component"
        } else {
            params.name.trim()
        };
        let d = doc.document_mut();
        let name = d.unique_object_name(name);
        let mut o = kreate_scene::Object::new(name, kreate_scene::ObjectData::Empty);
        o.transform.translation = middle;
        let id = d.add_object(o);
        if let Some(p) = shared {
            let _ = d.set_parent(id, Some(p));
        }
        for top in &tops {
            d.set_parent(*top, Some(id))
                .map_err(|e| OpError::Failed(format!("{e:?}")))?;
        }
        d.selection.clear();
        d.selection.add(id);
        d.selection.active = Some(id);
        Ok(OpResult::Finished)
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
    /// Fusion's Joint: a shaft's end brought into a plate's hole, joined
    /// cylindrical — it turns in the hole and slides along it, each way by
    /// its own slider; and a ball turns every way about its centre.
    #[test]
    fn a_shaft_is_joined_into_a_hole_and_turns_and_slides_there() {
        use crate::operator::OperatorRegistry;
        use crate::History;
        use kreate_reflect::{PropertyMap, PropertyValue};
        let mut r = OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let mut h = History::new();
        let mut doc = Document::new();
        let plate = doc.add_mesh_object("Plate", kreate_mesh::primitives::cube(0.1));
        let shaft = doc.add_mesh_object("Shaft", kreate_mesh::primitives::cube(0.01));
        doc.object_mut(shaft).unwrap().transform.translation = DVec3::new(0.5, 0.3, 0.2);
        let uid = |doc: &Document, id| doc.object_uid(id).unwrap().raw().to_string();
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
        // The shaft's bottom face's middle onto the plate's top's middle.
        let (sid, pid) = (uid(&doc, shaft), uid(&doc, plate));
        run(
            &r,
            &mut h,
            &mut doc,
            "assembly.joint",
            &[
                ("uid", PropertyValue::Text(sid)),
                ("target", PropertyValue::Text(pid)),
                ("from", PropertyValue::Vec3([0.5, 0.3, 0.195])),
                ("from_normal", PropertyValue::Vec3([0.0, 0.0, -1.0])),
                ("to", PropertyValue::Vec3([0.0, 0.0, 0.05])),
                ("to_normal", PropertyValue::Vec3([0.0, 0.0, 1.0])),
                ("kind", PropertyValue::Enum(3)),
            ],
        )
        .unwrap();
        let joint = doc.selection.active.unwrap();
        let Some(RobotPart::Joint(j)) = doc.object(joint).unwrap().robot.clone() else {
            panic!("no joint");
        };
        assert_eq!(j.kind, kreate_scene::robot::JointKind::Cylindrical);
        // Brought there: its bottom on the plate's top, its middle over
        // the plate's.
        let at = doc.world_transform(shaft).translation;
        assert!((at - DVec3::new(0.0, 0.0, 0.055)).length() < 1e-9, "{at:?}");
        assert_eq!(doc.object(shaft).unwrap().parent, Some(joint));
        // Its slide: 20 mm up the axis. Its turn kept.
        run(
            &r,
            &mut h,
            &mut doc,
            "robot.set_joint",
            &[
                ("position", PropertyValue::Float(0.02)),
                ("index", PropertyValue::Int(1)),
            ],
        )
        .unwrap();
        let at = doc.world_transform(shaft).translation;
        assert!((at - DVec3::new(0.0, 0.0, 0.075)).length() < 1e-9, "{at:?}");
        // And its turn: a quarter, the slide kept.
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
        let o = doc.object(joint).unwrap();
        let read = j.positions_of(&o.transform);
        assert!(
            (read[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "{read:?}"
        );
        assert!((read[1] - 0.02).abs() < 1e-9, "{read:?}");
        // Moving the plate takes the shaft with it.
        doc.object_mut(plate).unwrap().transform.translation = DVec3::new(1.0, 0.0, 0.0);
        let at = doc.world_transform(shaft).translation;
        assert!((at - DVec3::new(1.0, 0.0, 0.075)).length() < 1e-9, "{at:?}");
    }

    #[allow(clippy::type_complexity)]
    /// A registry, an empty history and a runner for the tests below.
    fn harness() -> (
        crate::operator::OperatorRegistry,
        crate::History,
        impl Fn(
            &crate::operator::OperatorRegistry,
            &mut crate::History,
            &mut Document,
            &str,
            &[(&'static str, kreate_reflect::PropertyValue)],
        ) -> Result<OpResult, OpError>,
    ) {
        let mut r = crate::operator::OperatorRegistry::new();
        crate::register_builtin(&mut r);
        let run = |r: &crate::operator::OperatorRegistry,
                   h: &mut crate::History,
                   doc: &mut Document,
                   id: &str,
                   pairs: &[(&'static str, kreate_reflect::PropertyValue)]| {
            let mut p = kreate_reflect::PropertyMap::new();
            for (k, v) in pairs {
                p.set(k, v.clone());
            }
            h.dispatch(r, doc, &Context::default(), id, &p)
        };
        (r, crate::History::new(), run)
    }

    /// Two joints linked: the follower turns twice as far as the driver,
    /// and a link that would loop is refused.
    #[test]
    fn a_motion_link_makes_one_joint_follow_another() {
        use kreate_reflect::PropertyValue as V;
        let (r, mut h, run) = harness();
        let mut doc = Document::new();
        let a = doc.add_object(joint("Gear A", None));
        let b = doc.add_object(joint("Gear B", None));
        for id in [a, b] {
            if let Some(Some(RobotPart::Joint(j))) = doc.object_mut(id).map(|o| o.robot.as_mut()) {
                j.kind = JointKind::Continuous;
            }
        }
        doc.selection.add(a);
        doc.selection.add(b);
        doc.selection.active = Some(b);
        run(
            &r,
            &mut h,
            &mut doc,
            "assembly.motion_link",
            &[("ratio", V::Float(-2.0))],
        )
        .unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "robot.set_joint",
            &[
                ("joint", V::Text("Gear A".into())),
                ("position", V::Float(0.5)),
            ],
        )
        .unwrap();
        let o = doc.object(b).unwrap();
        let j = joint_of(&doc, b).unwrap();
        assert!((j.position_of(&o.transform) + 1.0).abs() < 1e-9);
        // B already follows A: A following B would be a loop.
        doc.selection.active = Some(a);
        doc.selection.objects = vec![b, a];
        assert!(run(&r, &mut h, &mut doc, "assembly.motion_link", &[]).is_err());
    }

    /// With contact on, a slider goes only as far as touching the part in
    /// its way; with it off, through.
    #[test]
    fn contact_stops_a_slide_at_the_part_in_its_way() {
        use kreate_reflect::PropertyValue as V;
        let (r, mut h, run) = harness();
        let mut doc = Document::new();
        // A 10 mm block on a slide along X, a wall of 10 mm whose near
        // face is 30 mm along.
        let block = doc.add_mesh_object("Block", kreate_mesh::primitives::cube(0.01));
        let wall = doc.add_mesh_object("Wall", kreate_mesh::primitives::cube(0.01));
        doc.object_mut(wall).unwrap().transform.translation = DVec3::new(0.04, 0.0, 0.0);
        doc.selection.add(block);
        doc.selection.active = Some(block);
        run(
            &r,
            &mut h,
            &mut doc,
            "object.add_joint",
            &[
                ("kind", V::Enum(3)),
                ("at_cursor", V::Bool(false)),
                ("axis", V::Vec3([1.0, 0.0, 0.0])),
                ("lower", V::Float(-0.1)),
                ("upper", V::Float(0.1)),
            ],
        )
        .unwrap();
        let joint_id = doc.selection.active.unwrap();
        run(&r, &mut h, &mut doc, "assembly.contact", &[]).unwrap();
        assert!(joint_of(&doc, joint_id).unwrap().contact);
        run(
            &r,
            &mut h,
            &mut doc,
            "robot.set_joint",
            &[("position", V::Float(0.08))],
        )
        .unwrap();
        let x = doc.world_transform(block).translation.x;
        // Touching the wall: its far face at the wall's near one, 30 mm on.
        assert!((x - 0.03).abs() < 2e-4, "{x}");
        // Off, it goes through.
        run(&r, &mut h, &mut doc, "assembly.contact", &[]).unwrap();
        run(
            &r,
            &mut h,
            &mut doc,
            "robot.set_joint",
            &[("position", V::Float(0.08))],
        )
        .unwrap();
        let x = doc.world_transform(block).translation.x;
        assert!((x - 0.08).abs() < 1e-9, "{x}");
    }

    /// An exploded view moves each part out from the middle, by how far it
    /// is from it; undone, they are back.
    #[test]
    fn an_exploded_view_moves_the_parts_apart() {
        let (r, mut h, run) = harness();
        let mut doc = Document::new();
        let left = doc.add_mesh_object("Left", kreate_mesh::primitives::cube(0.01));
        let right = doc.add_mesh_object("Right", kreate_mesh::primitives::cube(0.01));
        doc.object_mut(left).unwrap().transform.translation = DVec3::new(-0.01, 0.0, 0.0);
        doc.object_mut(right).unwrap().transform.translation = DVec3::new(0.01, 0.0, 0.0);
        run(
            &r,
            &mut h,
            &mut doc,
            "assembly.explode",
            &[("amount", kreate_reflect::PropertyValue::Float(1.0))],
        )
        .unwrap();
        assert!((doc.world_transform(left).translation.x + 0.02).abs() < 1e-9);
        assert!((doc.world_transform(right).translation.x - 0.02).abs() < 1e-9);
        h.undo(&mut doc);
        assert!((doc.world_transform(left).translation.x + 0.01).abs() < 1e-9);
    }

    /// A component gathers the selected parts under one object, each left
    /// where it is, and moves them as one.
    #[test]
    fn a_component_gathers_the_parts_and_moves_them_as_one() {
        let (r, mut h, run) = harness();
        let mut doc = Document::new();
        let a = doc.add_mesh_object("A", kreate_mesh::primitives::cube(0.01));
        let b = doc.add_mesh_object("B", kreate_mesh::primitives::cube(0.01));
        doc.object_mut(b).unwrap().transform.translation = DVec3::new(0.1, 0.0, 0.0);
        doc.selection.add(a);
        doc.selection.add(b);
        run(&r, &mut h, &mut doc, "assembly.component", &[]).unwrap();
        let c = doc.selection.active.unwrap();
        assert_eq!(doc.object(a).unwrap().parent, Some(c));
        assert_eq!(doc.object(b).unwrap().parent, Some(c));
        assert!((doc.world_transform(b).translation.x - 0.1).abs() < 1e-9);
        assert!((doc.world_transform(c).translation.x - 0.05).abs() < 1e-9);
        doc.object_mut(c).unwrap().transform.translation.z += 1.0;
        assert!((doc.world_transform(a).translation.z - 1.0).abs() < 1e-9);
        assert!((doc.world_transform(b).translation.z - 1.0).abs() < 1e-9);
    }

    /// Each kind's positions read back as they were put.
    #[test]
    fn every_kind_reads_back_its_positions() {
        use kreate_scene::robot::JointKind;
        let rest = DTransform {
            translation: DVec3::new(1.0, 2.0, 3.0),
            rotation: DQuat::from_rotation_x(0.4),
            scale: DVec3::ONE,
        };
        for (kind, put) in [
            (JointKind::Revolute, vec![0.7]),
            (JointKind::Prismatic, vec![-0.03]),
            (JointKind::Cylindrical, vec![1.1, 0.05]),
            (JointKind::PinSlot, vec![-0.9, 0.02]),
            (JointKind::Planar, vec![0.01, -0.02, 0.5]),
            (JointKind::Ball, vec![0.3, -0.2, 0.25]),
        ] {
            let j = Joint {
                kind,
                rest,
                ..Joint::default()
            };
            let got = j.positions_of(&j.place_all(&put));
            for (a, b) in put.iter().zip(&got) {
                assert!((a - b).abs() < 1e-9, "{kind:?}: {put:?} read back {got:?}");
            }
        }
    }

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
