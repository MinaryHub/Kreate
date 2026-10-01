//! Armatures: bones at rest, and the pose an object holds them in.
//!
//! §2.4's model, Blender's in shape. The **armature** is a datablock — the
//! bones as they were built, their hierarchy and their rest placement — and
//! the **pose** is on the object, because two characters sharing one skeleton
//! stand differently. A bone's pose is a transform *relative to its rest*, in
//! the bone's own axes, so the identity pose is the skeleton as built and
//! "clear the pose" is setting every transform back to the identity.
//!
//! # A bone's axes
//!
//! Blender's: the bone's Y runs from its head to its tail, and the roll turns
//! X and Z about that. Rotating a bone about its own Y twists it; about X bends
//! it forward. That is the convention every rig tutorial, every exported
//! animation and every animator's hands assume, and a different one would make
//! each of those translate.
//!
//! # Where the numbers live
//!
//! Heads and tails are in the armature object's space. A pose is evaluated
//! into matrices in that same space by [`posed`]; the object's world transform
//! goes on top, as it does for a mesh's vertices.

use kreate_math::{DAffine3, DMat3, DQuat, DTransform, DVec3};
use kreate_reflect::Uid;

use crate::document::Linkage;

/// One bone at rest.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bone {
    pub name: String,
    /// Index into [`Armature::bones`]. Always an earlier bone, which
    /// [`Armature::add_bone`] keeps true and makes evaluation one pass.
    pub parent: Option<usize>,
    /// Armature space.
    pub head: DVec3,
    pub tail: DVec3,
    /// Radians about the bone's own Y.
    pub roll: f64,
    /// Whether its head is its parent's tail. Kept true by every edit: moving
    /// the parent's tail moves this head, and the other way round — Blender's
    /// Connected, what a chain extruded with `E` is.
    pub connected: bool,
    /// Edit-mode selection of the two ends. On the skeleton rather than the
    /// object because it is the skeleton being edited, as Blender keeps it.
    /// A connected bone's head is its parent's tail, and its flag follows.
    pub select_head: bool,
    pub select_tail: bool,
}

impl Bone {
    pub fn length(&self) -> f64 {
        (self.tail - self.head).length()
    }

    /// The bone's rest frame in armature space: origin at the head, Y along
    /// the bone, X and Z turned by the roll.
    ///
    /// Blender's construction: the rotation that takes Y to the bone's
    /// direction by the shortest arc, then the roll about the bone. Pointing
    /// straight down is the one direction with no shortest arc; it is a half
    /// turn about Z, which is what Blender picks.
    pub fn rest(&self) -> DAffine3 {
        let dir = (self.tail - self.head).normalize_or_zero();
        let align = if dir == DVec3::ZERO {
            DQuat::IDENTITY
        } else if dir.dot(DVec3::Y) < -1.0 + 1e-12 {
            DQuat::from_rotation_z(std::f64::consts::PI)
        } else {
            DQuat::from_rotation_arc(DVec3::Y, dir)
        };
        let rotation = align * DQuat::from_rotation_y(self.roll);
        DAffine3::from_rotation_translation(rotation, self.head)
    }
}

/// A skeleton: bones and their hierarchy, as built.
#[derive(Clone, Debug, PartialEq)]
pub struct Armature {
    pub name: String,
    pub linkage: Linkage,
    /// Assigned by the document. See [`Uid`].
    pub uid: Uid,
    /// Parents before children.
    pub bones: Vec<Bone>,
    /// Drawn over everything, so a skeleton inside its mesh can be seen and
    /// picked — Blender's In Front.
    pub in_front: bool,
    /// How the bones are drawn.
    pub display: BoneDisplay,
}

/// How an armature's bones are drawn: Blender's first two.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoneDisplay {
    /// A diamond a tenth of the way up, joined to both ends.
    #[default]
    Octahedral,
    /// A line from head to tail, for when the diamonds are in the way.
    Stick,
}

impl Armature {
    /// An armature with one bone standing a unit tall at the origin — what
    /// Blender's Add ▸ Armature makes.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            linkage: Linkage::Local,
            uid: Uid::NONE,
            bones: vec![Bone {
                name: "Bone".into(),
                parent: None,
                head: DVec3::ZERO,
                tail: DVec3::Z,
                ..Bone::default()
            }],
            in_front: false,
            display: BoneDisplay::Octahedral,
        }
    }

    /// Puts parents before children again after a parent changed, keeping
    /// every other bone in its order. `None` when the parents loop, which an
    /// edit must refuse before it gets here.
    pub fn sort_parents_first(&mut self) -> Option<()> {
        let n = self.bones.len();
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut placed = vec![false; n];
        while order.len() < n {
            let before = order.len();
            for i in 0..n {
                if !placed[i] && self.bones[i].parent.is_none_or(|p| placed[p]) {
                    placed[i] = true;
                    order.push(i);
                }
            }
            if order.len() == before {
                return None;
            }
        }
        let mut new_index = vec![0; n];
        for (k, &i) in order.iter().enumerate() {
            new_index[i] = k;
        }
        self.bones = order
            .iter()
            .map(|&i| {
                let mut b = self.bones[i].clone();
                b.parent = b.parent.map(|p| new_index[p]);
                b
            })
            .collect();
        Some(())
    }

    /// Adds a bone and returns its index. A parent that is not an earlier
    /// bone is dropped rather than stored: an index pointing forwards, or at
    /// nothing, is how a hierarchy comes to loop.
    pub fn add_bone(&mut self, mut bone: Bone) -> usize {
        if bone.parent.is_some_and(|p| p >= self.bones.len()) {
            bone.parent = None;
        }
        if bone.parent.is_none() {
            bone.connected = false;
        }
        self.bones.push(bone);
        self.bones.len() - 1
    }

    pub fn bone_index(&self, name: &str) -> Option<usize> {
        self.bones.iter().position(|b| b.name == name)
    }

    /// A name no bone has yet: `base`, then `base.001` and on, as Blender
    /// numbers them.
    pub fn unique_name(&self, base: &str) -> String {
        if self.bone_index(base).is_none() {
            return base.to_string();
        }
        // The number is Blender's, and is replaced rather than stacked:
        // extruding from `Bone.001` makes `Bone.002`, not `Bone.001.001`.
        let base = match base.rsplit_once('.') {
            Some((stem, n)) if n.len() == 3 && n.bytes().all(|b| b.is_ascii_digit()) => stem,
            _ => base,
        };
        (1..)
            .map(|i| format!("{base}.{i:03}"))
            .find(|n| self.bone_index(n).is_none())
            .expect("the names run out after the integers do")
    }
}

impl Armature {
    /// Whether the bone's head is selected, a connected head answering for
    /// its parent's tail.
    pub fn head_selected(&self, i: usize) -> bool {
        let b = &self.bones[i];
        match (b.connected, b.parent) {
            (true, Some(p)) => self.bones[p].select_tail,
            _ => b.select_head,
        }
    }

    /// Whether the whole bone is selected: both ends.
    pub fn bone_selected(&self, i: usize) -> bool {
        self.head_selected(i) && self.bones[i].select_tail
    }

    /// Sets a head's selection — its parent's tail, if connected.
    pub fn set_head_selected(&mut self, i: usize, selected: bool) {
        match (self.bones[i].connected, self.bones[i].parent) {
            (true, Some(p)) => self.bones[p].select_tail = selected,
            _ => self.bones[i].select_head = selected,
        }
    }

    /// Puts every connected head back on its parent's tail, parents first.
    pub fn reconnect(&mut self) {
        for i in 0..self.bones.len() {
            if let (true, Some(p)) = (self.bones[i].connected, self.bones[i].parent) {
                self.bones[i].head = self.bones[p].tail;
            }
        }
    }

    /// Takes bones out, their children going to the removed bone's parent,
    /// no longer connected — the joint they were connected at is gone.
    pub fn remove_bones(&mut self, remove: &[bool]) {
        let n = self.bones.len();
        // Where each old index lands, and each removed bone's surviving
        // ancestor, resolved parents first.
        let mut next = 0;
        let new_index: Vec<Option<usize>> = (0..n)
            .map(|i| {
                (!remove.get(i).copied().unwrap_or(false)).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        let mut heir: Vec<Option<usize>> = vec![None; n];
        for i in 0..n {
            heir[i] = match new_index[i] {
                Some(k) => Some(k),
                None => self.bones[i].parent.and_then(|p| heir[p]),
            };
        }
        let mut kept = Vec::with_capacity(next);
        for (i, bone) in self.bones.iter().enumerate() {
            if new_index[i].is_none() {
                continue;
            }
            let mut b = bone.clone();
            match b.parent {
                Some(p) if new_index[p].is_none() => {
                    b.parent = heir[p];
                    b.connected = false;
                }
                Some(p) => b.parent = new_index[p],
                None => {}
            }
            kept.push(b);
        }
        self.bones = kept;
    }
}

/// One bone's pose on one object.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseBone {
    /// The bone it moves, by name — Blender's rule, and what makes a pose
    /// survive a bone being added before it.
    pub name: String,
    /// Relative to the bone's rest, in its own axes.
    pub transform: DTransform,
    pub selected: bool,
}

/// How an object holds its armature's bones.
///
/// Sparse: a bone with no entry is at rest and unselected. Most bones of most
/// rigs are at rest most of the time, and a new bone needs no entry to exist.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pose {
    pub bones: Vec<PoseBone>,
    /// The bone the header and the sidebar show, by name.
    pub active: Option<String>,
    /// Inverse kinematics on this pose's bones, applied after the bones' own
    /// transforms, in order.
    pub ik: Vec<IkConstraint>,
}

/// Blender's IK constraint: the tip of `bone` reaches for the head of
/// `target`, turning `bone` and the `chain` bones above it (all the way to
/// the root when `chain` is zero). Both by name, as the pose is.
#[derive(Clone, Debug, PartialEq)]
pub struct IkConstraint {
    pub bone: String,
    pub target: String,
    pub chain: u32,
    pub enabled: bool,
    /// The pole: a bone whose head the chain's first joint — the elbow, the
    /// knee — bends towards. Empty for none, and the chain bends however its
    /// rest pose leans.
    pub pole: String,
}

impl Pose {
    pub fn get(&self, name: &str) -> Option<&PoseBone> {
        self.bones.iter().find(|b| b.name == name)
    }

    /// The entry for `name`, made at rest if there was none.
    pub fn entry(&mut self, name: &str) -> &mut PoseBone {
        match self.bones.iter().position(|b| b.name == name) {
            Some(i) => &mut self.bones[i],
            None => {
                self.bones.push(PoseBone { name: name.into(), transform: DTransform::IDENTITY, selected: false });
                self.bones.last_mut().expect("just pushed")
            }
        }
    }

    pub fn transform(&self, name: &str) -> DTransform {
        self.get(name).map(|b| b.transform).unwrap_or(DTransform::IDENTITY)
    }

    pub fn is_selected(&self, name: &str) -> bool {
        self.get(name).is_some_and(|b| b.selected)
    }
}

/// Every bone's posed frame, in armature space, in bone order.
///
/// A bone's frame is its parent's posed frame, then where it sits on the parent
/// at rest, then its own pose: `P_parent · (R_parent⁻¹ · R) · L`. For a root
/// the parent is the armature itself: `R · L`.
pub fn posed(armature: &Armature, pose: &Pose) -> Vec<DAffine3> {
    let mut frames = posed_fk(armature, pose);
    for ik in pose.ik.iter().filter(|c| c.enabled) {
        solve_ik(armature, pose, &mut frames, ik);
    }
    frames
}

/// The frames from the bones' own transforms alone, before any constraint.
pub fn posed_fk(armature: &Armature, pose: &Pose) -> Vec<DAffine3> {
    let mut out: Vec<DAffine3> = Vec::with_capacity(armature.bones.len());
    for bone in &armature.bones {
        let rest = bone.rest();
        let local = pose.transform(&bone.name).to_affine();
        let frame = match bone.parent.and_then(|p| Some((out.get(p)?, armature.bones.get(p)?))) {
            Some((parent_posed, parent)) => *parent_posed * parent.rest().inverse() * rest * local,
            None => rest * local,
        };
        out.push(frame);
    }
    out
}

/// How many passes the solver makes at most, and how close is close enough —
/// a hundred-thousandth of the chain's length.
const IK_PASSES: usize = 32;
const IK_CLOSE: f64 = 1e-5;

/// Turns the chain ending at `ik.bone` so its tip reaches the target's head
/// (cyclic coordinate descent: each bone in turn, from the tip up, turned
/// about its head to point the tip at the target), then carries every bone
/// hanging below the chain along with it.
fn solve_ik(armature: &Armature, pose: &Pose, frames: &mut [DAffine3], ik: &IkConstraint) {
    let (Some(tip_bone), Some(target)) = (armature.bone_index(&ik.bone), armature.bone_index(&ik.target)) else {
        return;
    };
    // The chain, tip first. A target inside its own chain would chase itself.
    let mut chain = vec![tip_bone];
    while ik.chain == 0 || chain.len() < ik.chain as usize {
        match armature.bones[*chain.last().expect("not empty")].parent {
            Some(p) => chain.push(p),
            None => break,
        }
    }
    if chain.contains(&target) || is_ancestor(armature, tip_bone, target) {
        return;
    }
    let goal = frames[target].translation;
    let length = armature.bones[tip_bone].length();
    let tip = |frames: &[DAffine3]| frames[tip_bone].transform_point3(DVec3::Y * length);
    let reach: f64 = chain.iter().map(|b| armature.bones[*b].length()).sum::<f64>().max(1e-9);
    // A straight chain with the goal on its own line gives the solver no
    // side to bend to: every bone already points at it. Bend the first
    // joint a little first — towards the pole when there is one.
    if chain.len() >= 2 {
        let root = frames[*chain.last().expect("not empty")].translation;
        let line = (tip(frames) - root).normalize_or_zero();
        let off_line = |p: DVec3| (p - root).cross(line).length();
        let straight = chain.iter().all(|b| off_line(frames[*b].translation) < 1e-6 * reach);
        if line != DVec3::ZERO && straight && off_line(goal) < 1e-6 * reach {
            let side = armature
                .bone_index(&ik.pole)
                .map(|p| frames[p].translation - root)
                .map(|v| v - line * v.dot(line))
                .filter(|v| v.length_squared() > 1e-18)
                .unwrap_or_else(|| line.any_orthonormal_vector());
            let axis = side.normalize().cross(line).normalize_or_zero();
            // The whole chain tipped off the line about its root: the
            // solver then has a side to fold the joints back from.
            let about = DAffine3::from_translation(root)
                * DAffine3::from_quat(DQuat::from_axis_angle(axis, -0.2))
                * DAffine3::from_translation(-root);
            for &c in &chain {
                frames[c] = about * frames[c];
            }
        }
    }
    for _ in 0..IK_PASSES {
        if tip(frames).distance(goal) < IK_CLOSE * reach {
            break;
        }
        for (k, &j) in chain.iter().enumerate() {
            let pivot = frames[j].translation;
            let (from, to) = (tip(frames) - pivot, goal - pivot);
            if from.length_squared() < 1e-18 || to.length_squared() < 1e-18 {
                continue;
            }
            let turn = DQuat::from_rotation_arc(from.normalize(), to.normalize());
            let about = DAffine3::from_translation(pivot) * DAffine3::from_quat(turn) * DAffine3::from_translation(-pivot);
            // Bone `j` and the chain below it, which hang from it.
            for &c in &chain[..=k] {
                frames[c] = about * frames[c];
            }
        }
    }
    // The pole: the whole chain turned about the line from its root to the
    // goal until its first joint is on the pole's side. The tip is on that
    // line, so it stays on the goal.
    if let Some(pole) = armature.bone_index(&ik.pole).filter(|p| !chain.contains(p)) {
        if chain.len() >= 2 {
            let root = frames[*chain.last().expect("not empty")].translation;
            let joint = frames[chain[chain.len() - 2]].translation;
            let axis = (goal - root).normalize_or_zero();
            let flat = |v: DVec3| v - axis * v.dot(axis);
            let (bend, want) = (flat(joint - root), flat(frames[pole].translation - root));
            if axis != DVec3::ZERO && bend.length_squared() > 1e-18 && want.length_squared() > 1e-18 {
                let (bend, want) = (bend.normalize(), want.normalize());
                let angle = bend.cross(want).dot(axis).atan2(bend.dot(want));
                let about = DAffine3::from_translation(root)
                    * DAffine3::from_quat(DQuat::from_axis_angle(axis, angle))
                    * DAffine3::from_translation(-root);
                for &c in &chain {
                    frames[c] = about * frames[c];
                }
            }
        }
    }
    // Everything else hanging from the chain follows it, as the forward pass
    // would have put it — parents come before children, so one sweep.
    let mut moved = vec![false; armature.bones.len()];
    for &c in &chain {
        moved[c] = true;
    }
    for (i, bone) in armature.bones.iter().enumerate() {
        if chain.contains(&i) {
            continue;
        }
        let Some(p) = bone.parent.filter(|p| moved[*p]) else { continue };
        let local = pose.transform(&bone.name).to_affine();
        frames[i] = frames[p] * armature.bones[p].rest().inverse() * bone.rest() * local;
        moved[i] = true;
    }
}

/// The pose transform that puts bone `index` at `frame` (armature space),
/// given the frames of the bones before it — the inverse of one step of
/// [`posed`], and what an edit in world terms turns into.
pub fn pose_for_frame(armature: &Armature, frames: &[DAffine3], index: usize, frame: DAffine3) -> DTransform {
    let bone = &armature.bones[index];
    let base = match bone.parent.and_then(|p| Some((frames.get(p)?, armature.bones.get(p)?))) {
        Some((parent_posed, parent)) => *parent_posed * parent.rest().inverse() * bone.rest(),
        None => bone.rest(),
    };
    DTransform::from_affine(base.inverse() * frame)
}

/// Whether `ancestor` is `bone` or above it.
pub fn is_ancestor(armature: &Armature, ancestor: usize, mut bone: usize) -> bool {
    for _ in 0..=armature.bones.len() {
        if bone == ancestor {
            return true;
        }
        match armature.bones.get(bone).and_then(|b| b.parent) {
            Some(p) => bone = p,
            None => return false,
        }
    }
    false
}

/// The bone's four corners at its widest, a tenth of the way up: the points
/// Blender's octahedral bone is drawn through. Armature space, from a frame.
pub fn octahedron(frame: &DAffine3, length: f64) -> [DVec3; 6] {
    let m: DMat3 = frame.matrix3;
    let head = frame.translation;
    let y = m * DVec3::Y * length;
    let w = 0.1 * length;
    let (x, z) = (m * DVec3::X * w, m * DVec3::Z * w);
    let ring = head + y * 0.1;
    [head, ring + x, ring + z, ring - x, ring - z, head + y]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: DVec3, b: DVec3) -> bool {
        (a - b).length() < 1e-9
    }

    fn arm() -> Armature {
        let mut a = Armature::new("Arm");
        a.bones[0].name = "Upper".into();
        a.add_bone(Bone { name: "Lower".into(), parent: Some(0), head: DVec3::Z, tail: DVec3::new(0.0, 0.0, 2.0), roll: 0.0, ..Default::default() });
        a
    }

    #[test]
    fn a_bones_y_runs_from_head_to_tail() {
        let b = Bone { name: "B".into(), parent: None, head: DVec3::new(1.0, 0.0, 0.0), tail: DVec3::new(1.0, 0.0, 3.0), roll: 0.0, ..Default::default() };
        let r = b.rest();
        assert!(close(r.translation, DVec3::new(1.0, 0.0, 0.0)));
        assert!(close(r.matrix3 * DVec3::Y, DVec3::Z));
        // Straight down, the one direction with no shortest arc.
        let down = Bone { tail: DVec3::new(1.0, 0.0, -2.0), ..b.clone() };
        assert!(close(down.rest().matrix3 * DVec3::Y, DVec3::NEG_Z));
        // The roll turns X about the bone.
        let rolled = Bone { roll: std::f64::consts::FRAC_PI_2, ..b };
        let x0 = b_x(&Bone { roll: 0.0, ..rolled.clone() });
        let x1 = b_x(&rolled);
        assert!(x0.dot(x1).abs() < 1e-9, "a quarter roll left X where it was");
    }

    fn b_x(b: &Bone) -> DVec3 {
        b.rest().matrix3 * DVec3::X
    }

    #[test]
    fn at_rest_the_pose_is_the_skeleton_as_built() {
        let a = arm();
        let frames = posed(&a, &Pose::default());
        assert!(close(frames[1].translation, DVec3::Z));
        assert!(close(frames[1].matrix3 * DVec3::Y, DVec3::Z));
    }

    #[test]
    fn bending_a_parent_carries_its_child() {
        let a = arm();
        let mut pose = Pose::default();
        // A quarter turn about the upper bone's own X bends it over.
        pose.entry("Upper").transform.rotation = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
        let frames = posed(&a, &pose);
        let elbow = frames[1].translation;
        let tip = frames[1].transform_point3(DVec3::Y * a.bones[1].length());
        assert!((elbow.z).abs() < 1e-9, "the child's head did not follow its parent's tail: {elbow:?}");
        assert!((elbow.length() - 1.0).abs() < 1e-9);
        assert!(((tip - elbow).length() - 1.0).abs() < 1e-9, "the child changed length");
        assert!(close(tip - elbow, elbow), "the child did not keep pointing along its parent");
    }

    #[test]
    fn a_frame_turns_back_into_the_pose_that_makes_it() {
        let a = arm();
        let mut pose = Pose::default();
        pose.entry("Upper").transform.rotation = DQuat::from_rotation_z(0.4);
        pose.entry("Lower").transform = DTransform {
            translation: DVec3::new(0.1, 0.2, 0.0),
            rotation: DQuat::from_rotation_x(0.3),
            scale: DVec3::splat(1.5),
        };
        let frames = posed(&a, &pose);
        let back = pose_for_frame(&a, &frames, 1, frames[1]);
        let want = pose.transform("Lower");
        assert!(close(back.translation, want.translation));
        assert!(back.rotation.dot(want.rotation).abs() > 1.0 - 1e-12);
        assert!(close(back.scale, want.scale));
    }

    #[test]
    fn names_are_numbered_and_parents_only_point_back() {
        let mut a = arm();
        assert_eq!(a.unique_name("Lower"), "Lower.001");
        assert_eq!(a.unique_name("Hand"), "Hand");
        // Numbered from the stem, not stacked on the number.
        a.bones[1].name = "Lower.001".into();
        assert_eq!(a.unique_name("Lower.001"), "Lower.002");
        let i = a.add_bone(Bone { name: "Loose".into(), parent: Some(99), head: DVec3::ZERO, tail: DVec3::Y, roll: 0.0, ..Default::default() });
        assert_eq!(a.bones[i].parent, None);
        assert!(is_ancestor(&a, 0, 1));
        assert!(!is_ancestor(&a, 1, 0));
    }

    /// The arm, a hand hanging off the forearm, and a loose IK target.
    fn rig() -> Armature {
        let mut a = arm();
        a.add_bone(Bone { name: "Hand".into(), parent: Some(1), head: DVec3::new(0.0, 0.0, 2.0), tail: DVec3::new(0.0, 0.0, 2.3), connected: true, ..Default::default() });
        a.add_bone(Bone { name: "Target".into(), parent: None, head: DVec3::new(1.2, 0.0, 0.9), tail: DVec3::new(1.2, 0.0, 1.2), ..Default::default() });
        a
    }

    fn ik(chain: u32) -> IkConstraint {
        IkConstraint { bone: "Lower".into(), target: "Target".into(), chain, enabled: true, pole: String::new() }
    }

    #[test]
    fn ik_brings_the_forearms_tip_to_the_target_and_the_hand_along() {
        let a = rig();
        let pose = Pose { ik: vec![ik(2)], ..Default::default() };
        let frames = posed(&a, &pose);
        let tip = frames[1].transform_point3(DVec3::Y * a.bones[1].length());
        assert!(tip.distance(DVec3::new(1.2, 0.0, 0.9)) < 1e-3, "{tip:?}");
        // The upper arm still starts where it did; the forearm starts at its tip.
        assert!(close(frames[0].translation, DVec3::ZERO));
        let elbow = frames[0].transform_point3(DVec3::Y * a.bones[0].length());
        assert!(elbow.distance(frames[1].translation) < 1e-9);
        // The hand is carried to the wrist.
        assert!(frames[2].translation.distance(tip) < 1e-9, "the hand stayed behind");
        // Off, the arm stands as it was.
        let off = Pose { ik: vec![IkConstraint { enabled: false, ..ik(2) }], ..Default::default() };
        assert!(close(posed(&a, &off)[1].translation, DVec3::Z));
    }

    #[test]
    fn out_of_reach_the_chain_points_straight_at_the_target() {
        let mut a = rig();
        a.bones[3].head = DVec3::new(10.0, 0.0, 0.0);
        a.bones[3].tail = DVec3::new(10.0, 0.0, 1.0);
        let frames = posed(&a, &Pose { ik: vec![ik(0)], ..Default::default() });
        let tip = frames[1].transform_point3(DVec3::Y * a.bones[1].length());
        assert!((tip - DVec3::new(2.0, 0.0, 0.0)).length() < 1e-2, "{tip:?}");
    }

    #[test]
    fn a_chain_of_one_turns_only_the_forearm() {
        let a = rig();
        let frames = posed(&a, &Pose { ik: vec![ik(1)], ..Default::default() });
        assert!(close(frames[0].matrix3 * DVec3::Y, DVec3::Z), "the upper arm moved");
        let dir = (frames[1].matrix3 * DVec3::Y).normalize();
        let want = (DVec3::new(1.2, 0.0, 0.9) - DVec3::Z).normalize();
        assert!(dir.dot(want) > 0.999, "{dir:?}");
    }

    #[test]
    fn a_pole_decides_which_way_the_elbow_bends() {
        // The target straight above-ish in X, so the arm has to bend; a pole
        // bone out on +Y and then on -Y puts the elbow on that side.
        let mut a = rig();
        a.bones[3].head = DVec3::new(0.0, 0.0, 1.5);
        a.bones[3].tail = DVec3::new(0.0, 0.0, 1.8);
        for side in [1.0, -1.0] {
            let mut rig = a.clone();
            rig.add_bone(Bone { name: "Pole".into(), head: DVec3::new(0.0, 3.0 * side, 1.0), tail: DVec3::new(0.0, 3.0 * side, 1.3), ..Default::default() });
            let pose = Pose { ik: vec![IkConstraint { pole: "Pole".into(), ..ik(2) }], ..Default::default() };
            let frames = posed(&rig, &pose);
            let elbow = frames[1].translation;
            assert!(elbow.y * side > 0.3, "side {side}: elbow at {elbow:?}");
            let tip = frames[1].transform_point3(DVec3::Y * rig.bones[1].length());
            assert!(tip.distance(DVec3::new(0.0, 0.0, 1.5)) < 1e-3, "the pole pulled the tip off the target: {tip:?}");
        }
    }
}

