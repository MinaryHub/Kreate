//! Building a skeleton through the operators.

use kreate_math::DVec3;
use kreate_ops::operator::Context;
use kreate_ops::{History, Mode, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

fn edit_ctx() -> Context {
    Context { mode: Mode::from_tag("edit_armature").expect("an armature edit mode") }
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
) -> bool {
    let mut p = PropertyMap::new();
    for (k, v) in params {
        p.set(k, v.clone());
    }
    let depth = history.entries().len();
    history.dispatch(registry, doc, &edit_ctx(), op, &p).expect("ran");
    history.entries().len() > depth
}

/// One armature, one bone standing up from (2, 0, 0).
fn rig() -> (Document, kreate_scene::ArmatureId) {
    let mut doc = Document::new();
    let id = doc.add_armature(kreate_scene::armature::Armature::new("Arm"));
    let mut object = Object::new("Rig", ObjectData::Armature(id));
    object.transform.translation = DVec3::new(2.0, 0.0, 0.0);
    let object = doc.add_object(object);
    doc.selection.add(object);
    (doc, id)
}

fn text(s: &str) -> PropertyValue {
    PropertyValue::Text(s.into())
}

#[test]
fn extruding_grows_a_connected_chain_and_moving_the_joint_moves_both_bones() {
    let (registry, mut history) = wired();
    let (mut doc, id) = rig();
    // Select the tail and extrude twice: three bones in a chain.
    assert!(run(&registry, &mut history, &mut doc, "armature.select_pick", &[("bone", text("Bone")), ("part", PropertyValue::Enum(2))]));
    assert!(run(&registry, &mut history, &mut doc, "armature.extrude", &[]));
    assert!(run(&registry, &mut history, &mut doc, "armature.extrude", &[]));
    let a = doc.armature(id).unwrap();
    assert_eq!(a.bones.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), vec!["Bone", "Bone.001", "Bone.002"]);
    assert!(a.bones[1].connected && a.bones[2].connected);
    assert_eq!(a.bones[1].head, a.bones[0].tail);
    assert!((a.bones[2].tail - DVec3::new(0.0, 0.0, 1.75)).length() < 1e-12, "{:?}", a.bones[2].tail);

    // The joint between the first two: selecting the second's head is the
    // first's tail, and moving it moves both.
    assert!(run(&registry, &mut history, &mut doc, "armature.select_pick", &[("bone", text("Bone.001")), ("part", PropertyValue::Enum(1))]));
    assert!(doc.armature(id).unwrap().bones[0].select_tail);
    assert!(run(&registry, &mut history, &mut doc, "armature.translate", &[("translation", PropertyValue::Vec3([0.0, 1.0, 0.0]))]));
    let a = doc.armature(id).unwrap();
    assert!((a.bones[0].tail - DVec3::new(0.0, 1.0, 1.0)).length() < 1e-12);
    assert_eq!(a.bones[1].head, a.bones[0].tail, "the chain came apart at the joint");
    assert_eq!(a.bones[0].head, DVec3::ZERO, "an end that was not selected moved");
}

#[test]
fn a_move_is_in_the_worlds_terms_on_a_turned_armature() {
    let (registry, mut history) = wired();
    let (mut doc, id) = rig();
    let object = doc.selection.active.unwrap();
    doc.object_mut(object).unwrap().transform.rotation = kreate_math::DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
    run(&registry, &mut history, &mut doc, "armature.select_pick", &[("bone", text("Bone")), ("part", PropertyValue::Enum(2))]);
    run(&registry, &mut history, &mut doc, "armature.translate", &[("translation", PropertyValue::Vec3([0.0, 1.0, 0.0]))]);
    // World +Y on an armature turned a quarter about Z is its own +X.
    let tail = doc.armature(id).unwrap().bones[0].tail;
    assert!((tail - DVec3::new(1.0, 0.0, 1.0)).length() < 1e-12, "{tail:?}");
    assert_eq!(kreate_ops::ops::armature::pivot(&doc).map(|p| (p - DVec3::new(2.0, 1.0, 1.0)).length() < 1e-12), Some(true));
}

#[test]
fn deleting_a_middle_bone_hands_its_child_to_its_parent_and_undo_brings_it_back() {
    let (registry, mut history) = wired();
    let (mut doc, id) = rig();
    run(&registry, &mut history, &mut doc, "armature.select_pick", &[("bone", text("Bone")), ("part", PropertyValue::Enum(2))]);
    run(&registry, &mut history, &mut doc, "armature.extrude", &[]);
    run(&registry, &mut history, &mut doc, "armature.extrude", &[]);
    let built = doc.armature(id).unwrap().clone();
    assert!(run(&registry, &mut history, &mut doc, "armature.select_pick", &[("bone", text("Bone.001"))]));
    assert!(run(&registry, &mut history, &mut doc, "armature.delete", &[]));
    let a = doc.armature(id).unwrap();
    assert_eq!(a.bones.len(), 2);
    assert_eq!(a.bones[1].name, "Bone.002");
    assert_eq!(a.bones[1].parent, Some(0));
    assert!(!a.bones[1].connected, "still connected to a joint that is gone");
    assert!(history.undo(&mut doc));
    assert!(history.undo(&mut doc));
    assert_eq!(doc.armature(id).unwrap().bones, built.bones);
}

#[test]
fn the_last_bone_is_not_deleted_and_a_new_one_can_be_added() {
    let (registry, mut history) = wired();
    let (mut doc, id) = rig();
    run(&registry, &mut history, &mut doc, "armature.select_all", &[("action", PropertyValue::Enum(0))]);
    assert!(!run(&registry, &mut history, &mut doc, "armature.delete", &[]));
    assert!(run(&registry, &mut history, &mut doc, "armature.bone_add", &[("head", PropertyValue::Vec3([1.0, 0.0, 0.0]))]));
    let a = doc.armature(id).unwrap();
    assert_eq!(a.bones.len(), 2);
    assert_eq!(a.bones[1].name, "Bone.001");
    assert_eq!(a.bones[1].parent, None);
    assert!(a.bone_selected(1) && !a.bone_selected(0), "only the new bone is selected");
}

#[test]
fn rotate_and_scale_act_about_the_selections_middle() {
    let (registry, mut history) = wired();
    let (mut doc, id) = rig();
    run(&registry, &mut history, &mut doc, "armature.select_pick", &[("bone", text("Bone"))]);
    // The bone's middle is (2, 0, 0.5) in the world; half a turn about X
    // there swaps its ends.
    assert!(run(&registry, &mut history, &mut doc, "armature.rotate", &[("angle", PropertyValue::Float(std::f64::consts::PI)), ("axis", PropertyValue::Enum(0))]));
    let b = &doc.armature(id).unwrap().bones[0];
    assert!((b.head - DVec3::Z).length() < 1e-9 && b.tail.length() < 1e-9, "{:?} {:?}", b.head, b.tail);
    assert!(run(&registry, &mut history, &mut doc, "armature.scale", &[("scale", PropertyValue::Vec3([2.0, 2.0, 2.0]))]));
    let b = &doc.armature(id).unwrap().bones[0];
    assert!((b.length() - 2.0).abs() < 1e-9);
}

#[test]
fn a_drag_can_start_from_nothing() {
    // `G`, `R` and `S` begin their drag with the identity. An operator that
    // declined "no change" could never be dragged; the app said "Move is not
    // possible here" after every E, and G never started at all.
    let (registry, mut history) = wired();
    let (mut doc, _) = rig();
    run(&registry, &mut history, &mut doc, "armature.select_all", &[("action", PropertyValue::Enum(0))]);
    for (op, key, value) in [
        ("armature.translate", "translation", PropertyValue::Vec3([0.0; 3])),
        ("armature.rotate", "angle", PropertyValue::Float(0.0)),
        ("armature.scale", "scale", PropertyValue::Vec3([1.0; 3])),
    ] {
        let mut p = PropertyMap::new();
        p.set(key, value);
        let session = history.begin_modal(&registry, &mut doc, None, &edit_ctx(), op, &p).expect("ran");
        assert!(session.is_some(), "{op} refused to start from the identity");
        History::cancel_modal(session.unwrap(), &mut doc);
    }
}

/// A three-bone chain, extruded from the one bone.
fn chain(registry: &OperatorRegistry, history: &mut History, doc: &mut Document) {
    run(registry, history, doc, "armature.select_pick", &[("bone", text("Bone")), ("part", PropertyValue::Enum(2))]);
    run(registry, history, doc, "armature.extrude", &[]);
    run(registry, history, doc, "armature.extrude", &[]);
}

fn set_bone(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    id: kreate_scene::ArmatureId,
    bone: &str,
    change: impl FnOnce(&mut kreate_ops::ops::armature::SetBoneParams),
) -> bool {
    let a = doc.armature(id).unwrap();
    let mut p = kreate_ops::ops::armature::bone_params(a, a.bone_index(bone).unwrap()).unwrap();
    change(&mut p);
    let map = kreate_reflect::OpParams::to_map(&p);
    let depth = history.entries().len();
    history.dispatch(registry, doc, &edit_ctx(), "armature.bone_set", &map).expect("ran");
    history.entries().len() > depth
}

#[test]
fn renaming_a_bone_carries_its_pose_and_its_curves() {
    let (registry, mut history) = wired();
    let (mut doc, id) = rig();
    chain(&registry, &mut history, &mut doc);
    let rig_object = doc.selection.active.unwrap();
    {
        let o = doc.object_mut(rig_object).unwrap();
        o.pose.entry("Bone.001").transform.translation.x = 0.5;
        o.pose.active = Some("Bone.001".into());
    }
    let mut action = kreate_scene::animation::Action::new("A");
    action
        .curve_mut(&kreate_scene::animation::bone_path("Bone.001", "location"), 0)
        .insert(1.0, 0.5, kreate_scene::animation::Interpolation::Linear);
    let action = doc.add_action(action);
    doc.object_mut(rig_object).unwrap().action = Some(action);

    assert!(set_bone(&registry, &mut history, &mut doc, id, "Bone.001", |p| p.name = "Forearm".into()));
    assert_eq!(doc.armature(id).unwrap().bones[1].name, "Forearm");
    let o = doc.object(rig_object).unwrap();
    assert_eq!(o.pose.transform("Forearm").translation.x, 0.5, "the pose stayed with the old name");
    assert_eq!(o.pose.active.as_deref(), Some("Forearm"));
    let path = &doc.action(action).unwrap().curves[0].path.property;
    assert_eq!(path.as_ref(), kreate_scene::animation::bone_path("Forearm", "location"));

    // A name another bone has is refused, and so is an empty one.
    assert!(!set_bone(&registry, &mut history, &mut doc, id, "Forearm", |p| p.name = "Bone".into()));
    assert!(!set_bone(&registry, &mut history, &mut doc, id, "Forearm", |p| p.name = " ".into()));
    // And undo gives the old name back everywhere.
    assert!(history.undo(&mut doc));
    assert_eq!(doc.armature(id).unwrap().bones[1].name, "Bone.001");
    assert_eq!(doc.object(rig_object).unwrap().pose.transform("Bone.001").translation.x, 0.5);
}

#[test]
fn a_parent_below_the_bone_is_refused_and_a_new_one_reorders_the_list() {
    let (registry, mut history) = wired();
    let (mut doc, id) = rig();
    chain(&registry, &mut history, &mut doc);
    // Bone under its own grandchild: a loop.
    assert!(!set_bone(&registry, &mut history, &mut doc, id, "Bone", |p| p.parent = "Bone.002".into()));
    // A free bone added last, made the parent of the first: it has to move up.
    run(&registry, &mut history, &mut doc, "armature.bone_add", &[("head", PropertyValue::Vec3([0.0, 0.0, -1.0]))]);
    assert!(set_bone(&registry, &mut history, &mut doc, id, "Bone", |p| p.parent = "Bone.003".into()));
    let a = doc.armature(id).unwrap();
    assert_eq!(a.bones[0].name, "Bone.003", "a parent after its child");
    let bone = a.bone_index("Bone").unwrap();
    assert_eq!(a.bones[bone].parent, Some(0));
    for (i, b) in a.bones.iter().enumerate() {
        assert!(b.parent.is_none_or(|p| p < i), "{} comes before its parent", b.name);
    }
    // Connected: the head jumps to the new parent's tail.
    assert!(set_bone(&registry, &mut history, &mut doc, id, "Bone", |p| p.connected = true));
    let a = doc.armature(id).unwrap();
    let bone = a.bone_index("Bone").unwrap();
    assert_eq!(a.bones[bone].head, a.bones[0].tail);
}

#[test]
fn the_armature_tab_and_the_pose_fields() {
    let (registry, mut history) = wired();
    let (mut doc, id) = rig();
    let mut p = PropertyMap::new();
    p.set("in_front", PropertyValue::Bool(true));
    p.set("display", PropertyValue::Enum(1));
    history.dispatch(&registry, &mut doc, &edit_ctx(), "armature.set_display", &p).expect("ran");
    let a = doc.armature(id).unwrap();
    assert!(a.in_front);
    assert_eq!(a.display, kreate_scene::armature::BoneDisplay::Stick);

    let rig_object = doc.selection.active.unwrap();
    let mut p = kreate_ops::ops::pose::pose_bone_params(&doc.object(rig_object).unwrap().pose, "Bone");
    p.rotation = [0.0, 0.0, 0.5];
    p.location = [1.0, 0.0, 0.0];
    history
        .dispatch(&registry, &mut doc, &Context::default(), "pose.set_bone", &kreate_reflect::OpParams::to_map(&p))
        .expect("ran");
    let t = doc.object(rig_object).unwrap().pose.transform("Bone");
    assert_eq!(t.translation, DVec3::X);
    assert!((kreate_scene::animation::to_euler(t.rotation).z - 0.5).abs() < 1e-12);
    // And the fields read back what was set.
    let back = kreate_ops::ops::pose::pose_bone_params(&doc.object(rig_object).unwrap().pose, "Bone");
    assert!((back.rotation[2] - 0.5).abs() < 1e-12);
}

#[test]
fn clicking_a_bone_in_edit_mode_makes_it_the_active_one() {
    let (registry, mut history) = wired();
    let (mut doc, _) = rig();
    chain(&registry, &mut history, &mut doc);
    run(&registry, &mut history, &mut doc, "armature.select_pick", &[("bone", text("Bone.001"))]);
    let rig_object = doc.selection.active.unwrap();
    assert_eq!(doc.object(rig_object).unwrap().pose.active.as_deref(), Some("Bone.001"));
}
