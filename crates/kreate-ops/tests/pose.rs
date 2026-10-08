//! Posing through the operators: select a bone, move it, and its children
//! come along.

use kreate_math::{DQuat, DVec3};
use kreate_ops::operator::Context;
use kreate_ops::{History, Mode, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::armature::{Armature, Bone};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

fn pose_ctx() -> Context {
    Context {
        mode: Mode::from_tag("pose").expect("a pose mode"),
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
    a.add_bone(Bone {
        name: "Lower".into(),
        parent: Some(0),
        head: DVec3::Z,
        tail: DVec3::new(0.0, 0.0, 2.0),
        roll: 0.0,
        ..Default::default()
    });
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
    assert!(history
        .dispatch(&registry, &mut doc, &object_mode, "pose.select_pick", &p)
        .is_err());
    let mut plain = Document::new();
    let cube = plain.add_object(Object::new("Empty", ObjectData::Empty));
    plain.selection.add(cube);
    assert!(history
        .dispatch(&registry, &mut plain, &pose_ctx(), "pose.select_pick", &p)
        .is_err());
}

#[test]
fn turning_the_upper_bone_carries_the_lower_and_undo_brings_it_back() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[("bone", text("Upper"))]
    ));
    assert_eq!(
        kreate_ops::ops::pose::pivot(&doc),
        Some(DVec3::new(5.0, 0.0, 0.0))
    );
    // A quarter turn about world X through the shoulder lays the arm along -Y.
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.rotate",
        &[
            ("angle", PropertyValue::Float(-std::f64::consts::FRAC_PI_2)),
            ("axis", PropertyValue::Enum(0))
        ]
    ));
    let bones = doc.posed_bones(rig);
    let elbow = bones[1].translation;
    assert!(
        (elbow - DVec3::new(5.0, 1.0, 0.0)).length() < 1e-9,
        "the lower bone stayed behind: {elbow:?}"
    );
    assert!(history.undo(&mut doc));
    let elbow = doc.posed_bones(rig)[1].translation;
    assert!((elbow - DVec3::new(5.0, 0.0, 1.0)).length() < 1e-9);
}

#[test]
fn a_child_moved_with_its_parent_selected_is_moved_once() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_all",
        &[("action", PropertyValue::Enum(0))],
    );
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
    assert!(
        (bones[1].translation - DVec3::new(5.0, 2.0, 1.0)).length() < 1e-9,
        "the child moved twice"
    );
    // In the bone's own axes a world +Y is the upper bone's -Z (its Y is up).
    let upper = doc.object(rig).unwrap().pose.transform("Upper");
    assert!(
        (upper.translation - DVec3::new(0.0, 0.0, -2.0)).length() < 1e-9,
        "{:?}",
        upper.translation
    );
    assert!(doc.object(rig).unwrap().pose.transform("Lower") == kreate_math::DTransform::IDENTITY);
}

#[test]
fn a_scale_is_in_the_bones_own_axes_and_clearing_undoes_it() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[("bone", text("Upper"))],
    );
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.rotate",
        &[
            ("angle", PropertyValue::Float(0.3)),
            ("axis", PropertyValue::Enum(2)),
        ],
    );
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.scale",
        &[("scale", PropertyValue::Vec3([1.0, 2.0, 1.0]))]
    ));
    let upper = doc.object(rig).unwrap().pose.transform("Upper");
    assert!(
        (upper.scale - DVec3::new(1.0, 2.0, 1.0)).length() < 1e-9,
        "{:?}",
        upper.scale
    );
    // Twice as long along itself: the elbow is two units from the shoulder.
    let bones = doc.posed_bones(rig);
    assert!(((bones[1].translation - bones[0].translation).length() - 2.0).abs() < 1e-9);

    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.clear_transform",
        &[("what", PropertyValue::Enum(3))]
    ));
    let upper = doc.object(rig).unwrap().pose.transform("Upper");
    assert!((upper.scale - DVec3::ONE).length() < 1e-12);
    assert!(
        upper.rotation.dot(DQuat::IDENTITY).abs() < 1.0 - 1e-6,
        "clearing the scale cleared the rotation"
    );
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.clear_transform",
        &[("what", PropertyValue::Enum(0))]
    ));
    assert_eq!(
        doc.object(rig).unwrap().pose.transform("Upper"),
        kreate_math::DTransform::IDENTITY
    );
    assert!(
        !run(
            &registry,
            &mut history,
            &mut doc,
            &ctx,
            "pose.clear_transform",
            &[("what", PropertyValue::Enum(0))]
        ),
        "clearing what is already clear is a step"
    );
}

#[test]
fn shift_click_adds_and_a_second_takes_away() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[("bone", text("Upper"))],
    );
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[
            ("bone", text("Lower")),
            ("extend", PropertyValue::Bool(true)),
        ],
    );
    let pose = &doc.object(rig).unwrap().pose;
    assert!(pose.is_selected("Upper") && pose.is_selected("Lower"));
    assert_eq!(pose.active.as_deref(), Some("Lower"));
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[
            ("bone", text("Lower")),
            ("extend", PropertyValue::Bool(true)),
        ],
    );
    let pose = &doc.object(rig).unwrap().pose;
    assert!(pose.is_selected("Upper") && !pose.is_selected("Lower"));
    assert!(
        !run(
            &registry,
            &mut history,
            &mut doc,
            &ctx,
            "pose.select_pick",
            &[("bone", text("Tail"))]
        ),
        "a bone that is not there"
    );
}

#[test]
fn add_armature_makes_one_standing_bone_and_selects_it() {
    let (registry, mut history) = wired();
    let mut doc = Document::new();
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &Context::default(),
        "object.add_armature",
        &[]
    ));
    let rig = doc.selection.active.expect("the new armature is active");
    let (_, a) = doc.armature_of(rig).expect("an armature");
    assert_eq!(a.bones.len(), 1);
    assert!((doc.posed_bones(rig)[0].matrix3 * DVec3::Y - DVec3::Z).length() < 1e-12);
}

// --- keys ---------------------------------------------------------------------

fn key(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    ctx: &Context,
    op: &str,
    frame: f64,
) -> bool {
    run(
        registry,
        history,
        doc,
        ctx,
        op,
        &[("frame", PropertyValue::Float(frame))],
    )
}

#[test]
fn i_in_pose_mode_keys_the_selected_bones_and_the_pose_plays_back() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[("bone", text("Upper"))],
    );
    assert!(key(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.keyframe_insert",
        1.0
    ));
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.rotate",
        &[
            ("angle", PropertyValue::Float(1.0)),
            ("axis", PropertyValue::Enum(0)),
        ],
    );
    assert!(key(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.keyframe_insert",
        11.0
    ));

    let action = doc
        .object(rig)
        .unwrap()
        .action
        .expect("keying made an action");
    let paths: Vec<String> = doc
        .action(action)
        .unwrap()
        .curves
        .iter()
        .map(|c| c.path.property.to_string())
        .collect();
    assert_eq!(paths.len(), 10, "three, four and three curves: {paths:?}");
    assert!(
        paths
            .iter()
            .all(|p| p.starts_with("pose.bones[\"Upper\"].")),
        "{paths:?}"
    );

    // Halfway, the bone is halfway round, and the object has not moved.
    doc.animate(6.0);
    let turned = doc.object(rig).unwrap().pose.transform("Upper").rotation;
    let (axis, angle) = turned.to_axis_angle();
    assert!(
        (angle - 0.5).abs() < 0.05,
        "turned {angle} halfway between 0 and 1"
    );
    assert!(axis.x.abs() > 0.99);
    assert_eq!(
        doc.object(rig).unwrap().transform.translation,
        DVec3::new(5.0, 0.0, 0.0)
    );
    doc.animate(1.0);
    assert!(
        doc.object(rig)
            .unwrap()
            .pose
            .transform("Upper")
            .rotation
            .dot(DQuat::IDENTITY)
            .abs()
            > 1.0 - 1e-12
    );
}

#[test]
fn a_key_on_the_far_side_of_the_sphere_is_brought_back() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[("bone", text("Upper"))],
    );
    key(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.keyframe_insert",
        1.0,
    );
    // The same rotation, written as its negative: a straight line from +1 to
    // -1 in W goes through nothing, and the bone would spin all the way round.
    doc.object_mut(rig)
        .unwrap()
        .pose
        .entry("Upper")
        .transform
        .rotation = -DQuat::from_rotation_x(0.2);
    key(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.keyframe_insert",
        11.0,
    );
    let action = doc.object(rig).unwrap().action.unwrap();
    let w = doc
        .action(action)
        .unwrap()
        .curves
        .iter()
        .find(|c| c.path.property.ends_with("rotation_quaternion") && c.path.index == Some(0))
        .unwrap()
        .keys[1]
        .value;
    assert!(w > 0.9, "keyed on the far side: w = {w}");
}

#[test]
fn alt_i_in_pose_mode_leaves_the_objects_own_keys() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    // A key on the object first, in object mode.
    assert!(key(
        &registry,
        &mut history,
        &mut doc,
        &Context::default(),
        "anim.keyframe_insert",
        1.0
    ));
    let ctx = pose_ctx();
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[("bone", text("Lower"))],
    );
    key(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.keyframe_insert",
        1.0,
    );
    let action = doc.object(rig).unwrap().action.unwrap();
    assert_eq!(doc.action(action).unwrap().curves.len(), 19);
    assert!(key(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.keyframe_delete",
        1.0
    ));
    let left: Vec<String> = doc
        .action(action)
        .unwrap()
        .curves
        .iter()
        .map(|c| c.path.property.to_string())
        .collect();
    assert_eq!(left.len(), 9, "{left:?}");
    assert!(left.iter().all(|p| !p.starts_with("pose.")));
}

#[test]
fn auto_keying_keys_the_bones_a_turn_moved() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[("bone", text("Lower"))],
    );
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.rotate",
        &[
            ("angle", PropertyValue::Float(0.4)),
            ("axis", PropertyValue::Enum(0)),
            ("autokey", PropertyValue::Bool(true)),
            ("key_frame", PropertyValue::Float(7.0)),
        ],
    );
    let action = doc.object(rig).unwrap().action.expect("the turn keyed");
    let a = doc.action(action).unwrap();
    assert_eq!(a.curves.len(), 7, "location and rotation");
    assert!(a
        .curves
        .iter()
        .all(|c| c.keys.len() == 1 && c.keys[0].frame == 7.0));
    // One step: undoing the turn takes its key too.
    assert!(history.undo(&mut doc));
    assert!(doc.object(rig).unwrap().action.is_none());
}

#[test]
fn a_bone_with_a_space_in_its_name_can_be_picked_in_the_dope_sheet() {
    use kreate_ops::ops::anim::row_target;
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let ctx = pose_ctx();
    doc.armature_mut(doc.armature_of(rig).unwrap().0)
        .unwrap()
        .bones[1]
        .name = "Lower Arm".into();
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "pose.select_pick",
        &[("bone", text("Lower Arm"))],
    );
    key(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.keyframe_insert",
        1.0,
    );
    let action = doc.object(rig).unwrap().action.unwrap();
    let uid = doc.action_uid(action).unwrap().raw();
    run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.keys_select_all",
        &[("action", PropertyValue::Enum(1))],
    );
    let path = kreate_scene::animation::bone_path("Lower Arm", "location");
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.keys_select_box",
        &[
            ("targets", text(&row_target(uid, Some(&path)))),
            ("frame_min", PropertyValue::Float(0.5)),
            ("frame_max", PropertyValue::Float(1.5)),
        ]
    ));
    let selected: Vec<String> = doc
        .action(action)
        .unwrap()
        .curves
        .iter()
        .filter(|c| c.keys.iter().any(|k| k.selected))
        .map(|c| c.path.property.to_string())
        .collect();
    assert_eq!(selected, vec![path.clone(), path.clone(), path]);
}

#[test]
fn the_bone_tabs_diamond_keys_one_property_of_one_bone() {
    let (registry, mut history) = wired();
    let (mut doc, rig) = arm();
    let path = kreate_scene::animation::bone_path("Lower", "rotation_quaternion");
    let mut p = PropertyMap::new();
    p.set("frame", PropertyValue::Float(3.0));
    p.set("datablock", PropertyValue::Enum(2));
    p.set("property", PropertyValue::Text(path.clone()));
    history
        .dispatch(
            &registry,
            &mut doc,
            &Context::default(),
            "anim.keyframe_insert_property",
            &p,
        )
        .expect("ran");
    let action = doc.object(rig).unwrap().action.expect("an action");
    let curves: Vec<_> = doc
        .action(action)
        .unwrap()
        .curves
        .iter()
        .map(|c| c.path.property.to_string())
        .collect();
    assert_eq!(curves, vec![path.clone(); 4]);
    history
        .dispatch(
            &registry,
            &mut doc,
            &Context::default(),
            "anim.keyframe_delete_property",
            &p,
        )
        .expect("ran");
    assert!(doc.action(action).unwrap().curves.is_empty());
    // A bone the armature does not have keys nothing.
    p.set(
        "property",
        PropertyValue::Text(kreate_scene::animation::bone_path("Tail", "location")),
    );
    assert!(history
        .dispatch(
            &registry,
            &mut doc,
            &Context::default(),
            "anim.keyframe_insert_property",
            &p
        )
        .is_ok());
    assert!(doc.action(action).unwrap().curves.is_empty());
}

#[test]
fn shift_i_puts_ik_on_the_active_bone_toward_the_other_selected_one() {
    use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_scene::armature::{Armature, Bone};
    use kreate_scene::document::{Object, ObjectData};
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut doc = kreate_scene::Document::new();
    let mut a = Armature::new("Arm");
    a.bones[0].name = "Upper".into();
    a.add_bone(Bone {
        name: "Lower".into(),
        parent: Some(0),
        head: kreate_math::DVec3::Z,
        tail: kreate_math::DVec3::new(0.0, 0.0, 2.0),
        connected: true,
        ..Default::default()
    });
    a.add_bone(Bone {
        name: "Goal".into(),
        head: kreate_math::DVec3::new(1.2, 0.0, 0.9),
        tail: kreate_math::DVec3::new(1.2, 0.0, 1.3),
        ..Default::default()
    });
    let arm = doc.add_armature(a);
    let rig = doc.add_object(Object::new("Rig", ObjectData::Armature(arm)));
    doc.selection.add(rig);
    {
        let pose = &mut doc.object_mut(rig).unwrap().pose;
        pose.entry("Goal").selected = true;
        pose.entry("Lower").selected = true;
        pose.active = Some("Lower".into());
    }
    let mut history = History::new();
    let ctx = kreate_ops::operator::Context::default();
    history
        .dispatch(
            &registry,
            &mut doc,
            &ctx,
            "pose.ik_add",
            &PropertyMap::new(),
        )
        .unwrap();
    let ik = doc.object(rig).unwrap().pose.ik.clone();
    assert_eq!(ik.len(), 1);
    assert_eq!(
        (ik[0].bone.as_str(), ik[0].target.as_str(), ik[0].chain),
        ("Lower", "Goal", 2)
    );
    // The forearm's tip is on the goal now.
    let frames = doc.posed_bones(rig);
    let tip = frames[1].transform_point3(kreate_math::DVec3::Y);
    assert!(
        tip.distance(kreate_math::DVec3::new(1.2, 0.0, 0.9)) < 1e-3,
        "{tip:?}"
    );
    // Off, then gone.
    let mut p = PropertyMap::new();
    p.set("target", PropertyValue::Text("Goal".into()));
    p.set("enabled", PropertyValue::Bool(false));
    history
        .dispatch(&registry, &mut doc, &ctx, "pose.ik_set", &p)
        .unwrap();
    assert!(!doc.object(rig).unwrap().pose.ik[0].enabled);
    history
        .dispatch(
            &registry,
            &mut doc,
            &ctx,
            "pose.ik_remove",
            &PropertyMap::new(),
        )
        .unwrap();
    assert!(doc.object(rig).unwrap().pose.ik.is_empty());
    // Renaming the goal follows into the constraint.
    history
        .dispatch(
            &registry,
            &mut doc,
            &ctx,
            "pose.ik_add",
            &PropertyMap::new(),
        )
        .unwrap();
    assert!(doc.rename_bone(arm, "Goal", "Hand_IK"));
    assert_eq!(doc.object(rig).unwrap().pose.ik[0].target, "Hand_IK");
}

#[test]
fn a_bone_constraint_is_added_set_and_removed_by_operators() {
    use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_scene::armature::{Armature, Bone, ConstraintKind};
    use kreate_scene::document::{Object, ObjectData};
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut doc = kreate_scene::Document::new();
    let mut a = Armature::new("Arm");
    a.bones[0].name = "Head".into();
    a.add_bone(Bone {
        name: "Goal".into(),
        head: kreate_math::DVec3::new(2.0, 0.0, 0.0),
        tail: kreate_math::DVec3::new(2.0, 0.0, 1.0),
        ..Default::default()
    });
    let arm = doc.add_armature(a);
    let rig = doc.add_object(Object::new("Rig", ObjectData::Armature(arm)));
    doc.selection.add(rig);
    {
        let pose = &mut doc.object_mut(rig).unwrap().pose;
        pose.entry("Goal").selected = true;
        pose.entry("Head").selected = true;
        pose.active = Some("Head".into());
    }
    let mut history = History::new();
    let ctx = kreate_ops::operator::Context::default();
    // Damped Track, at the other selected bone.
    let mut p = PropertyMap::new();
    p.set("kind", PropertyValue::Int(3));
    history
        .dispatch(&registry, &mut doc, &ctx, "pose.constraint_add", &p)
        .unwrap();
    let c = doc.object(rig).unwrap().pose.constraints.clone();
    assert_eq!(c.len(), 1);
    assert_eq!((c[0].bone.as_str(), c[0].target.as_str()), ("Head", "Goal"));
    assert_eq!(c[0].kind, ConstraintKind::DampedTrack);
    // The head bone now points along +X, at the goal.
    let along = doc.posed_bones(rig)[0]
        .matrix3
        .mul_vec3(kreate_math::DVec3::Y);
    assert!(
        along.normalize().dot(kreate_math::DVec3::X) > 0.999,
        "{along:?}"
    );
    // Halfway, through the operator the Bone tab sends.
    let mut set =
        kreate_reflect::OpParams::to_map(&kreate_ops::ops::pose::constraint_params(0, &c[0]));
    set.set("influence", PropertyValue::Float(0.5));
    history
        .dispatch(&registry, &mut doc, &ctx, "pose.constraint_set", &set)
        .unwrap();
    assert_eq!(doc.object(rig).unwrap().pose.constraints[0].influence, 0.5);
    // Undone, then removed.
    history.undo(&mut doc);
    assert_eq!(doc.object(rig).unwrap().pose.constraints[0].influence, 1.0);
    let mut p = PropertyMap::new();
    p.set("index", PropertyValue::Int(0));
    history
        .dispatch(&registry, &mut doc, &ctx, "pose.constraint_remove", &p)
        .unwrap();
    assert!(doc.object(rig).unwrap().pose.constraints.is_empty());
}

#[test]
fn push_down_turns_the_action_into_a_strip_where_its_keys_are() {
    use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_scene::animation::{Action, Interpolation};
    use kreate_scene::document::{Object, ObjectData};
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut doc = kreate_scene::Document::new();
    let mut a = Action::new("Hop");
    let c = a.curve_mut("location", 2);
    c.insert(5.0, 0.0, Interpolation::Linear);
    c.insert(15.0, 1.0, Interpolation::Linear);
    let a = doc.add_action(a);
    let mut o = Object::new("Ball", ObjectData::Empty);
    o.action = Some(a);
    let id = doc.add_object(o);
    doc.selection.add(id);
    let mut history = History::new();
    let ctx = kreate_ops::operator::Context::default();
    history
        .dispatch(
            &registry,
            &mut doc,
            &ctx,
            "anim.nla_push_down",
            &PropertyMap::new(),
        )
        .unwrap();
    let o = doc.object(id).unwrap();
    assert!(o.action.is_none());
    assert_eq!(o.nla.len(), 1);
    assert_eq!(
        (o.nla[0].start, o.nla[0].action_start, o.nla[0].action_end),
        (5.0, 5.0, 15.0)
    );
    // It still plays.
    doc.animate(10.0);
    assert!((doc.object(id).unwrap().transform.translation.z - 0.5).abs() < 1e-9);
    // Twice as slow, through the operator the panel sends.
    let mut p = kreate_reflect::OpParams::to_map(&kreate_ops::ops::anim::nla_strip_params(
        0,
        &doc.object(id).unwrap().nla[0],
    ));
    p.set("scale", PropertyValue::Float(2.0));
    history
        .dispatch(&registry, &mut doc, &ctx, "anim.nla_strip_set", &p)
        .unwrap();
    doc.animate(15.0);
    assert!((doc.object(id).unwrap().transform.translation.z - 0.5).abs() < 1e-9);
}
