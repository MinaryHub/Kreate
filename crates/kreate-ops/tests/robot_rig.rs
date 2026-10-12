//! A robot read from a URDF, posed by an armature: bones made from its joints,
//! the joints following the bones — turned by hand, set by number, dragged by
//! IK — and objects hanging from bones as they are.

use std::io::Cursor;
use std::path::Path;

use kreate_math::{DAffine3, DQuat, DVec3};
use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::armature::{self, ConstraintKind};
use kreate_scene::robot::{JointKind, RobotPart};
use kreate_scene::{Document, ObjectId};

/// A base turning on a yaw joint, two pitch joints up an arm, a tool fixed
/// on the end; and beside it a slider carrying a joint that mimics the yaw.
const ROBOT: &str = r#"<robot name="bot">
  <link name="base"><visual><geometry><box size="0.4 0.4 0.1"/></geometry></visual></link>
  <link name="turret"><visual><origin xyz="0 0 0.1"/><geometry><cylinder radius="0.08" length="0.2"/></geometry></visual></link>
  <link name="upper"><visual><origin xyz="0 0 0.25"/><geometry><box size="0.05 0.05 0.5"/></geometry></visual></link>
  <link name="fore"><visual><origin xyz="0 0 0.2"/><geometry><box size="0.04 0.04 0.4"/></geometry></visual></link>
  <link name="tool"><visual><origin xyz="0 0 0.05"/><geometry><box size="0.03 0.03 0.1"/></geometry></visual></link>
  <link name="carriage"><visual><geometry><box size="0.1 0.1 0.05"/></geometry></visual></link>
  <link name="flag"><visual><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.01 0.05"/></geometry></visual></link>
  <joint name="yaw" type="revolute">
    <parent link="base"/><child link="turret"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="shoulder" type="revolute">
    <parent link="turret"/><child link="upper"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/><child link="fore"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
  <joint name="mount" type="fixed">
    <parent link="fore"/><child link="tool"/>
    <origin xyz="0 0 0.4"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="carriage"/>
    <origin xyz="0.3 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.2" effort="1" velocity="1"/>
  </joint>
  <joint name="wave" type="revolute">
    <parent link="carriage"/><child link="flag"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
    <mimic joint="yaw" multiplier="0.5" offset="0.1"/>
  </joint>
</robot>"#;

fn robot() -> Document {
    let scene = kreate_io::urdf::parse_urdf(ROBOT, Path::new("."), &|p: &Path| {
        Err(format!("no file {}", p.display()))
    })
    .expect("parses");
    let mut doc = Document::new();
    kreate_io::build_into(&scene, &mut doc);
    doc
}

fn named(doc: &Document, name: &str) -> ObjectId {
    doc.objects()
        .find(|(_, o)| o.name == name)
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("no {name}"))
}

fn position(doc: &Document, joint: &str) -> f64 {
    let o = doc.object(named(doc, joint)).unwrap();
    match &o.robot {
        Some(RobotPart::Joint(j)) => j.position_of(&o.transform),
        _ => panic!("{joint} is not a joint"),
    }
}

fn wired() -> (OperatorRegistry, History) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (registry, History::new())
}

fn run(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    op: &str,
    params: &[(&'static str, PropertyValue)],
) {
    let mut p = PropertyMap::new();
    for (k, v) in params {
        p.set(k, v.clone());
    }
    history
        .dispatch(registry, doc, &Context::default(), op, &p)
        .unwrap_or_else(|e| panic!("{op}: {e}"));
}

/// The robot, rigged through the operator, with the rig's object.
fn rigged() -> (Document, ObjectId, OperatorRegistry, History) {
    let mut doc = robot();
    let (registry, mut history) = wired();
    let base = named(&doc, "base");
    doc.selection.clear();
    doc.selection.add(base);
    doc.selection.active = Some(base);
    run(&registry, &mut history, &mut doc, "robot.rig", &[]);
    let rig = doc.selection.active.expect("the rig is made active");
    (doc, rig, registry, history)
}

fn turn(doc: &mut Document, rig: ObjectId, bone: &str, rotation: DQuat) {
    doc.object_mut(rig)
        .unwrap()
        .pose
        .entry(bone)
        .transform
        .rotation = rotation;
    doc.sync_robot_rigs();
}

#[test]
fn every_moving_joint_is_a_bone_standing_on_its_axis() {
    let (doc, rig, _, _) = rigged();
    let (_, armature) = doc.armature_of(rig).expect("an armature");
    let names: Vec<&str> = armature.bones.iter().map(|b| b.name.as_str()).collect();
    for joint in ["yaw", "shoulder", "elbow", "slide", "wave"] {
        assert!(names.contains(&joint), "{joint} in {names:?}");
    }
    assert!(!names.contains(&"mount"), "a fixed joint is no bone");
    // Parented as the joints are; the arm's effector and its IK target are
    // the two other bones.
    let parent = |n: &str| {
        let b = &armature.bones[armature.bone_index(n).unwrap()];
        b.parent.map(|p| armature.bones[p].name.clone())
    };
    assert_eq!(parent("yaw"), None);
    assert_eq!(parent("shoulder").as_deref(), Some("yaw"));
    assert_eq!(parent("elbow").as_deref(), Some("shoulder"));
    assert_eq!(parent("wave").as_deref(), Some("slide"));
    assert_eq!(armature.bones.len(), 7, "{names:?}");
    assert_eq!(parent("elbow tip").as_deref(), Some("elbow"));
    // Each bone's head on its joint and its Z on the joint's axis, in the
    // world.
    let rig_world = doc.world_transform(rig);
    for joint in ["yaw", "shoulder", "elbow", "slide", "wave"] {
        let bone = &armature.bones[armature.bone_index(joint).unwrap()];
        let frame = rig_world * bone.rest();
        let joint_world = doc.world_transform(named(&doc, joint));
        assert!(
            frame.translation.distance(joint_world.translation) < 1e-6,
            "{joint}"
        );
        let axis = frame.matrix3.mul_vec3(DVec3::Z);
        let want = joint_world.matrix3.mul_vec3(DVec3::Z);
        assert!(
            axis.dot(want) > 1.0 - 1e-6,
            "{joint}: {axis} against {want}"
        );
    }
    let pose = &doc.object(rig).unwrap().pose;
    assert_eq!(pose.ik.len(), 1);
    assert_eq!(pose.ik[0].bone, "elbow tip");
    assert_eq!(pose.ik[0].chain, 4);
    assert!(!pose.ik[0].enabled, "IK waits to be switched on");
    assert!(pose
        .constraints
        .iter()
        .any(|c| c.bone == "slide" && matches!(c.kind, ConstraintKind::LimitLocation { .. })));
}

#[test]
fn a_turned_bone_turns_its_link_about_the_joint_axis_alone() {
    let (mut doc, rig, _, _) = rigged();
    let tool = named(&doc, "tool");
    let elbow_world = doc.world_transform(named(&doc, "elbow"));
    let before = doc.world_transform(tool);
    turn(&mut doc, rig, "elbow", DQuat::from_rotation_z(0.6));
    assert!((position(&doc, "elbow") - 0.6).abs() < 1e-6);
    let after = doc.world_transform(tool);
    // The tool turned 0.6 about the elbow's axis, through the elbow.
    let axis = elbow_world.matrix3.mul_vec3(DVec3::Z).normalize();
    let about = DAffine3::from_translation(elbow_world.translation)
        * DAffine3::from_quat(DQuat::from_axis_angle(axis, 0.6))
        * DAffine3::from_translation(-elbow_world.translation);
    let expected = about * before;
    assert!(
        after.abs_diff_eq(expected, 1e-6),
        "{after:?}\nagainst\n{expected:?}"
    );
    // Turned off its axis as well: the limit takes the tilt away, and what
    // is left is a turn about the axis.
    turn(
        &mut doc,
        rig,
        "elbow",
        DQuat::from_rotation_x(0.4) * DQuat::from_rotation_z(0.6),
    );
    let angle = position(&doc, "elbow");
    assert!(angle > 0.3 && angle < 0.9, "{angle}");
    let about = DAffine3::from_translation(elbow_world.translation)
        * DAffine3::from_quat(DQuat::from_axis_angle(axis, angle))
        * DAffine3::from_translation(-elbow_world.translation);
    assert!(doc.world_transform(tool).abs_diff_eq(about * before, 1e-6));
}

#[test]
fn the_limits_hold_and_a_slider_slides() {
    let (mut doc, rig, _, _) = rigged();
    turn(&mut doc, rig, "shoulder", DQuat::from_rotation_z(2.5));
    assert!(
        (position(&doc, "shoulder") - 1.5).abs() < 1e-6,
        "held at the limit"
    );
    turn(&mut doc, rig, "shoulder", DQuat::from_rotation_z(-2.5));
    assert!((position(&doc, "shoulder") + 1.5).abs() < 1e-6);
    // The slider: along its axis, to its stop, and nowhere else.
    let o = doc.object_mut(rig).unwrap();
    o.pose.entry("slide").transform.translation = DVec3::new(0.3, -0.2, 0.5);
    doc.sync_robot_rigs();
    assert!((position(&doc, "slide") - 0.2).abs() < 1e-6);
    let carriage = doc.world_transform(named(&doc, "carriage")).translation;
    assert!(
        carriage.distance(DVec3::new(0.5, 0.0, 0.0)) < 1e-6,
        "{carriage}"
    );
}

#[test]
fn a_mimic_follows_and_its_bone_with_it() {
    let (mut doc, rig, _, _) = rigged();
    turn(&mut doc, rig, "yaw", DQuat::from_rotation_z(0.8));
    assert!((position(&doc, "wave") - (0.5 * 0.8 + 0.1)).abs() < 1e-6);
    let bone = doc.object(rig).unwrap().pose.transform("wave");
    assert!((armature::twist_about_z(bone.rotation) - 0.5).abs() < 1e-6);
}

#[test]
fn setting_a_joint_by_number_turns_its_bone() {
    let (mut doc, rig, registry, mut history) = rigged();
    run(
        &registry,
        &mut history,
        &mut doc,
        "robot.set_joint",
        &[
            ("joint", PropertyValue::Text("shoulder".into())),
            ("position", PropertyValue::Float(0.7)),
        ],
    );
    doc.sync_robot_rigs();
    assert!(
        (position(&doc, "shoulder") - 0.7).abs() < 1e-6,
        "not sprung back"
    );
    let bone = doc.object(rig).unwrap().pose.transform("shoulder");
    assert!((armature::twist_about_z(bone.rotation) - 0.7).abs() < 1e-6);
    // And the reset puts both back.
    run(&registry, &mut history, &mut doc, "robot.reset_pose", &[]);
    doc.sync_robot_rigs();
    assert!(position(&doc, "shoulder").abs() < 1e-6);
}

#[test]
fn ik_drags_the_arm_to_its_target_through_its_hinges() {
    let (mut doc, rig, _, _) = rigged();
    // A pose the arm can reach, and where its tip is in it.
    for (bone, a) in [("yaw", 0.4), ("shoulder", 0.5), ("elbow", -0.7)] {
        doc.object_mut(rig)
            .unwrap()
            .pose
            .entry(bone)
            .transform
            .rotation = DQuat::from_rotation_z(a);
    }
    let (_, armature) = doc.armature_of(rig).unwrap();
    let armature = armature.clone();
    let elbow = armature.bone_index("elbow tip").unwrap();
    let length = armature.bones[elbow].length();
    let frames = armature::posed(&armature, &doc.object(rig).unwrap().pose);
    let goal = frames[elbow].transform_point3(DVec3::Y * length);
    // Back to rest, the target moved there, IK on.
    {
        let o = doc.object_mut(rig).unwrap();
        o.pose.bones.retain(|b| b.name.ends_with(" IK"));
        o.pose.ik[0].enabled = true;
    }
    let target = armature
        .bone_index(&doc.object(rig).unwrap().pose.ik[0].target)
        .unwrap();
    let frames = armature::posed(&armature, &doc.object(rig).unwrap().pose);
    let want = DAffine3::from_rotation_translation(
        armature.bones[target]
            .rest()
            .to_scale_rotation_translation()
            .1,
        goal,
    );
    let t = armature::pose_for_frame(&armature, &frames, target, want);
    doc.object_mut(rig)
        .unwrap()
        .pose
        .entry(&armature.bones[target].name)
        .transform = t;
    doc.sync_robot_rigs();
    let frames = armature::posed(&armature, &doc.object(rig).unwrap().pose);
    let tip = frames[elbow].transform_point3(DVec3::Y * length);
    assert!(tip.distance(goal) < 1e-3, "reached {tip} for {goal}");
    // Every joint turned about its own axis, within its limits: the hinges
    // are what was solved, so the bones' frames are the joints'.
    let rig_world = doc.world_transform(rig);
    for joint in ["yaw", "shoulder", "elbow"] {
        let bone_world = rig_world * frames[armature.bone_index(joint).unwrap()];
        let joint_world = doc.world_transform(named(&doc, joint));
        assert!(
            bone_world.translation.distance(joint_world.translation) < 1e-6,
            "{joint}"
        );
        let (b, j) = (
            bone_world.matrix3.mul_vec3(DVec3::Z),
            joint_world.matrix3.mul_vec3(DVec3::Z),
        );
        assert!(b.dot(j) > 1.0 - 1e-6, "{joint}");
    }
    assert!(position(&doc, "shoulder").abs() <= 1.5 + 1e-6);
    assert!(position(&doc, "elbow").abs() <= 2.0 + 1e-6);
}

#[test]
fn the_rig_survives_saving_and_undo_takes_it_away() {
    let (mut doc, rig, _, mut history) = rigged();
    turn(&mut doc, rig, "shoulder", DQuat::from_rotation_z(0.3));
    let bytes =
        kreate_io::container::write(&doc, &kreate_io::Manifest::default(), &[]).expect("writes");
    let mut back = kreate_io::container::read(Cursor::new(bytes))
        .expect("reads")
        .document;
    back.sync_robot_rigs();
    assert!((position(&back, "shoulder") - 0.3).abs() < 1e-6);
    let rig_back = named(&back, &doc.object(rig).unwrap().name);
    assert_eq!(
        back.object(rig_back).unwrap().robot,
        doc.object(rig).unwrap().robot
    );
    assert_eq!(
        back.object(rig_back).unwrap().pose.constraints,
        doc.object(rig).unwrap().pose.constraints
    );
    // Turning it after the trip still turns the robot.
    turn(&mut back, rig_back, "elbow", DQuat::from_rotation_z(-0.4));
    assert!((position(&back, "elbow") + 0.4).abs() < 1e-6);
    // Undo: no armature.
    assert!(history.undo(&mut doc));
    assert!(doc
        .objects()
        .all(|(_, o)| !matches!(o.robot, Some(RobotPart::Rig { .. }))));
    let _ = JointKind::Fixed;
}

#[test]
fn an_object_parented_to_a_bone_rides_it_through_save_copy_and_gltf() {
    let (mut doc, rig, registry, mut history) = rigged();
    // A marker hung from the elbow's bone, by the operator.
    let marker = doc.add_mesh_object("Marker", kreate_mesh::primitives::cube(0.02));
    doc.object_mut(marker).unwrap().transform.translation = DVec3::new(0.1, 0.0, 0.9);
    let at = doc.world_transform(marker);
    doc.object_mut(rig).unwrap().pose.active = Some("elbow".into());
    doc.selection.clear();
    doc.selection.add(marker);
    doc.selection.add(rig);
    doc.selection.active = Some(rig);
    run(
        &registry,
        &mut history,
        &mut doc,
        "object.parent",
        &[("bone", PropertyValue::Bool(true))],
    );
    let o = doc.object(marker).unwrap();
    assert_eq!(o.parent, Some(rig));
    assert_eq!(o.parent_bone.as_deref(), Some("elbow"));
    assert!(
        doc.world_transform(marker).abs_diff_eq(at, 1e-6),
        "stayed put"
    );
    // Turning the bone carries it, rigidly, about the elbow.
    let elbow = doc.world_transform(named(&doc, "elbow"));
    turn(&mut doc, rig, "elbow", DQuat::from_rotation_z(0.5));
    let axis = elbow.matrix3.mul_vec3(DVec3::Z).normalize();
    let about = DAffine3::from_translation(elbow.translation)
        * DAffine3::from_quat(DQuat::from_axis_angle(axis, 0.5))
        * DAffine3::from_translation(-elbow.translation);
    assert!(doc.world_transform(marker).abs_diff_eq(about * at, 1e-6));
    let moved = doc.world_transform(marker);
    // Saved and read back.
    let bytes =
        kreate_io::container::write(&doc, &kreate_io::Manifest::default(), &[]).expect("writes");
    let back = kreate_io::container::read(Cursor::new(bytes))
        .expect("reads")
        .document;
    let m = named(&back, "Marker");
    assert_eq!(
        back.object(m).unwrap().parent_bone.as_deref(),
        Some("elbow")
    );
    assert!(back.world_transform(m).abs_diff_eq(moved, 1e-6));
    // Copied with its armature, into another document.
    let copy = doc.extract_objects(&[rig, marker]);
    let mut other = Document::new();
    let added = other.merge_from(&copy);
    let m2 = added
        .iter()
        .copied()
        .find(|id| other.object(*id).unwrap().name.starts_with("Marker"))
        .unwrap();
    assert_eq!(
        other.object(m2).unwrap().parent_bone.as_deref(),
        Some("elbow")
    );
    // In a glTF, under the bone's node.
    let (scene, _) = kreate_io::scene_from_document(&doc);
    let marker_node = scene.nodes.iter().position(|n| n.name == "Marker").unwrap();
    let parent = scene.nodes[marker_node].parent.expect("a parent");
    assert_eq!(scene.nodes[parent].name, "elbow");
    // Undo the parenting: back on its own, where it was.
    assert!(history.undo(&mut doc));
    assert_eq!(doc.object(marker).unwrap().parent_bone, None);
}

#[test]
fn keyed_bones_play_the_robot_in_time_and_in_a_gltf() {
    let (mut doc, rig, _, _) = rigged();
    let bones = vec!["elbow".to_string()];
    let props = ["rotation_quaternion"];
    turn(&mut doc, rig, "elbow", DQuat::IDENTITY);
    kreate_ops::ops::anim::key_bones(&mut doc, rig, &bones, 1.0, &props);
    turn(&mut doc, rig, "elbow", DQuat::from_rotation_z(1.0));
    kreate_ops::ops::anim::key_bones(&mut doc, rig, &bones, 11.0, &props);
    // Played: the joint follows the keyed bone, frame by frame.
    doc.animate(1.0);
    assert!(position(&doc, "elbow").abs() < 1e-6);
    doc.animate(11.0);
    assert!((position(&doc, "elbow") - 1.0).abs() < 1e-6);
    // Written: the elbow joint's own node moves, so the arm moves with it in
    // a viewer that knows nothing of the rig.
    let (scene, _) = kreate_io::scene_from_document(&doc);
    let order: Vec<ObjectId> = doc.objects().map(|(id, _)| id).collect();
    let node = order
        .iter()
        .position(|id| *id == named(&doc, "elbow"))
        .unwrap();
    let track = scene
        .animations
        .iter()
        .flat_map(|a| a.tracks.iter())
        .find(|t| t.node == node)
        .expect("the elbow joint is animated");
    let (first, last) = (track.transforms[0], *track.transforms.last().unwrap());
    let turned = first.rotation.inverse() * last.rotation;
    assert!(
        (turned.angle_between(DQuat::IDENTITY) - 1.0).abs() < 1e-3,
        "{turned:?}"
    );
}
