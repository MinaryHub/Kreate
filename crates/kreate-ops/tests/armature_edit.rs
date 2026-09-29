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
