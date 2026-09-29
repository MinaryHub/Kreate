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
        }
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
}
