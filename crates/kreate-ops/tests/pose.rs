//! Posing through the operators: select a bone, move it, and its children
//! come along.

use kreate_math::{DQuat, DVec3};
use kreate_ops::operator::Context;
use kreate_ops::{History, Mode, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::armature::{Armature, Bone};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

fn pose_ctx() -> Context {
    Context { mode: Mode::from_tag("pose").expect("a pose mode") }
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
    ctx: &Context,
    op: &str,
    params: &[(&'static str, PropertyValue)],
) -> bool {
    let mut p = PropertyMap::new();
    for (k, v) in params {
        p.set(k, v.clone());
    }
    let depth = history.entries().len();
    history.dispatch(registry, doc, ctx, op, &p).expect("ran");
    history.entries().len() > depth
}

/// An arm of two bones standing up from (5, 0, 0), active and selected.
fn arm() -> (Document, kreate_scene::ObjectId) {
    let mut doc = Document::new();
    let mut a = Armature::new("Arm");
    a.bones[0].name = "Upper".into();
    a.add_bone(Bone { name: "Lower".into(), parent: Some(0), head: DVec3::Z, tail: DVec3::new(0.0, 0.0, 2.0), roll: 0.0 });
    let id = doc.add_armature(a);
    let mut object = Object::new("Rig", ObjectData::Armature(id));
    object.transform.translation = DVec3::new(5.0, 0.0, 0.0);
    let object = doc.add_object(object);
    doc.selection.add(object);
    (doc, object)
}

fn text(s: &str) -> PropertyValue {
    PropertyValue::Text(s.into())
}

#[test]
fn pose_operators_need_pose_mode_and_an_armature() {
    let (registry, mut history) = wired();
    let (mut doc, _) = arm();
    let mut p = PropertyMap::new();
    p.set("bone", text("Upper"));
    let object_mode = Context::default();
    assert!(history.dispatch(&registry, &mut doc, &object_mode, "pose.select_pick", &p).is_err());
    let mut plain = Document::new();
    let cube = plain.add_object(Object::new("Empty", ObjectData::Empty));
    plain.selection.add(cube);
    assert!(history.dispatch(&registry, &mut plain, &pose_ctx(), "pose.select_pick", &p).is_err());
}

#[test]
fn turning_the_upper_bone_carries_the_lower_and_undo_brings_it_back() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    assert!(run(&registry, &mut history, &mut doc, &ctx, "pose.select_pick", &[("bone", text("Upper"))]));
    assert_eq!(kreate_ops::ops::pose::pivot(&doc), Some(DVec3::new(5.0, 0.0, 0.0)));
    // A quarter turn about world X through the shoulder lays the arm along -Y.
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.rotate",
        &[("angle", PropertyValue::Float(-std::f64::consts::FRAC_PI_2)), ("axis", PropertyValue::Enum(0))]
    ));
    let bones = doc.posed_bones(rig);
    let elbow = bones[1].translation;
    assert!((elbow - DVec3::new(5.0, 1.0, 0.0)).length() < 1e-9, "the lower bone stayed behind: {elbow:?}");
    assert!(history.undo(&mut doc));
    let elbow = doc.posed_bones(rig)[1].translation;
    assert!((elbow - DVec3::new(5.0, 0.0, 1.0)).length() < 1e-9);
}

#[test]
fn a_child_moved_with_its_parent_selected_is_moved_once() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    run(&registry, &mut history, &mut doc, &ctx, "pose.select_all", &[("action", PropertyValue::Enum(0))]);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.translate",
        &[("translation", PropertyValue::Vec3([0.0, 2.0, 0.0]))]
    ));
    let bones = doc.posed_bones(rig);
    assert!((bones[0].translation - DVec3::new(5.0, 2.0, 0.0)).length() < 1e-9);
    assert!((bones[1].translation - DVec3::new(5.0, 2.0, 1.0)).length() < 1e-9, "the child moved twice");
    // In the bone's own axes a world +Y is the upper bone's -Z (its Y is up).
    let upper = doc.object(rig).unwrap().pose.transform("Upper");
    assert!((upper.translation - DVec3::new(0.0, 0.0, -2.0)).length() < 1e-9, "{:?}", upper.translation);
    assert!(doc.object(rig).unwrap().pose.transform("Lower") == kreate_math::DTransform::IDENTITY);
}

#[test]
fn a_scale_is_in_the_bones_own_axes_and_clearing_undoes_it() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    run(&registry, &mut history, &mut doc, &ctx, "pose.select_pick", &[("bone", text("Upper"))]);
    run(&registry, &mut history, &mut doc, &ctx, "pose.rotate", &[("angle", PropertyValue::Float(0.3)), ("axis", PropertyValue::Enum(2))]);
    assert!(run(&registry, &mut history, &mut doc, &ctx, "pose.scale", &[("scale", PropertyValue::Vec3([1.0, 2.0, 1.0]))]));
    let upper = doc.object(rig).unwrap().pose.transform("Upper");
    assert!((upper.scale - DVec3::new(1.0, 2.0, 1.0)).length() < 1e-9, "{:?}", upper.scale);
    // Twice as long along itself: the elbow is two units from the shoulder.
    let bones = doc.posed_bones(rig);
    assert!(((bones[1].translation - bones[0].translation).length() - 2.0).abs() < 1e-9);

    assert!(run(&registry, &mut history, &mut doc, &ctx, "pose.clear_transform", &[("what", PropertyValue::Enum(3))]));
    let upper = doc.object(rig).unwrap().pose.transform("Upper");
    assert!((upper.scale - DVec3::ONE).length() < 1e-12);
    assert!(upper.rotation.dot(DQuat::IDENTITY).abs() < 1.0 - 1e-6, "clearing the scale cleared the rotation");
    assert!(run(&registry, &mut history, &mut doc, &ctx, "pose.clear_transform", &[("what", PropertyValue::Enum(0))]));
    assert_eq!(doc.object(rig).unwrap().pose.transform("Upper"), kreate_math::DTransform::IDENTITY);
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "pose.clear_transform", &[("what", PropertyValue::Enum(0))]),
        "clearing what is already clear is a step");
}

#[test]
fn shift_click_adds_and_a_second_takes_away() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    run(&registry, &mut history, &mut doc, &ctx, "pose.select_pick", &[("bone", text("Upper"))]);
    run(&registry, &mut history, &mut doc, &ctx, "pose.select_pick", &[("bone", text("Lower")), ("extend", PropertyValue::Bool(true))]);
    let pose = &doc.object(rig).unwrap().pose;
    assert!(pose.is_selected("Upper") && pose.is_selected("Lower"));
    assert_eq!(pose.active.as_deref(), Some("Lower"));
    run(&registry, &mut history, &mut doc, &ctx, "pose.select_pick", &[("bone", text("Lower")), ("extend", PropertyValue::Bool(true))]);
    let pose = &doc.object(rig).unwrap().pose;
    assert!(pose.is_selected("Upper") && !pose.is_selected("Lower"));
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "pose.select_pick", &[("bone", text("Tail"))]), "a bone that is not there");
}

#[test]
fn add_armature_makes_one_standing_bone_and_selects_it() {
    let (registry, mut history) = wired();
    let mut doc = Document::new();
    assert!(run(&registry, &mut history, &mut doc, &Context::default(), "object.add_armature", &[]));
    let rig = doc.selection.active.expect("the new armature is active");
    let (_, a) = doc.armature_of(rig).expect("an armature");
    assert_eq!(a.bones.len(), 1);
    assert!((doc.posed_bones(rig)[0].matrix3 * DVec3::Y - DVec3::Z).length() < 1e-12);
}
