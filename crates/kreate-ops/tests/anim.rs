//! Keyframing through the operators, and the scene played back from them.

use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;
use kreate_math::DVec3;

fn wired() -> (OperatorRegistry, History, Document, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (registry, History::new(), Document::new(), Context::default())
}

fn key(registry: &OperatorRegistry, history: &mut History, doc: &mut Document, ctx: &Context, frame: f64) -> bool {
    let mut p = PropertyMap::new();
    p.set("frame", PropertyValue::Float(frame));
    let depth = history.entries().len();
    history
        .dispatch(registry, doc, ctx, "anim.keyframe_insert", &p)
        .expect("ran");
    history.entries().len() > depth
}

fn a_selected_empty(doc: &mut Document) -> kreate_scene::ObjectId {
    let id = doc.add_object(Object::new("Box", ObjectData::Empty));
    doc.selection.add(id);
    id
}

#[test]
fn two_keys_move_an_object_between_them() {
    let (registry, mut history, mut doc, ctx) = wired();
    let id = a_selected_empty(&mut doc);
    assert!(key(&registry, &mut history, &mut doc, &ctx, 1.0));
    doc.object_mut(id).unwrap().transform.translation = DVec3::new(10.0, 0.0, 0.0);
    assert!(key(&registry, &mut history, &mut doc, &ctx, 11.0));

    // The first key made an action, named after the object, and linked it.
    let action = doc.object(id).unwrap().action.expect("keying made an action");
    assert_eq!(doc.action(action).unwrap().name, "BoxAction");
    assert_eq!(doc.action(action).unwrap().key_frames(), vec![1.0, 11.0]);
    // Nine curves: three components of three properties.
    assert_eq!(doc.action(action).unwrap().curves.len(), 9);

    doc.animate(1.0);
    assert!(doc.object(id).unwrap().transform.translation.x.abs() < 1e-9);
    doc.animate(11.0);
    assert!((doc.object(id).unwrap().transform.translation.x - 10.0).abs() < 1e-9);
    doc.animate(6.0);
    let halfway = doc.object(id).unwrap().transform.translation.x;
    assert!((halfway - 5.0).abs() < 1e-9, "eased motion is symmetric about the middle: {halfway}");
    assert_eq!(doc.playback.frame, 6.0);
}

#[test]
fn undoing_the_first_key_takes_the_action_away_too() {
    let (registry, mut history, mut doc, ctx) = wired();
    let id = a_selected_empty(&mut doc);
    key(&registry, &mut history, &mut doc, &ctx, 1.0);
    assert!(history.undo(&mut doc));
    assert!(doc.object(id).unwrap().action.is_none());
    assert_eq!(doc.action_count(), 0, "the action outlived the undo");
}

#[test]
fn deleting_a_key_leaves_the_others_and_nothing_to_delete_is_a_cancel() {
    let (registry, mut history, mut doc, ctx) = wired();
    let id = a_selected_empty(&mut doc);
    key(&registry, &mut history, &mut doc, &ctx, 1.0);
    key(&registry, &mut history, &mut doc, &ctx, 20.0);
    let delete = |history: &mut History, doc: &mut Document, frame: f64| {
        let mut p = PropertyMap::new();
        p.set("frame", PropertyValue::Float(frame));
        let depth = history.entries().len();
        history.dispatch(&registry, doc, &ctx, "anim.keyframe_delete", &p).expect("ran");
        history.entries().len() > depth
    };
    assert!(delete(&mut history, &mut doc, 20.0));
    let action = doc.object(id).unwrap().action.unwrap();
    assert_eq!(doc.action(action).unwrap().key_frames(), vec![1.0]);
    assert!(!delete(&mut history, &mut doc, 7.0), "a frame with no key left a step to undo");
    // The last key gone: the curves go too, and the property is free again.
    assert!(delete(&mut history, &mut doc, 1.0));
    assert!(doc.action(action).unwrap().curves.is_empty());
}

#[test]
fn a_rotation_keyed_past_half_a_turn_plays_the_long_way_it_was_keyed() {
    // Euler curves, so two keys a full turn apart spin rather than stand still,
    // which is the reason rotation is keyed as angles and not a quaternion.
    let (registry, mut history, mut doc, ctx) = wired();
    let id = a_selected_empty(&mut doc);
    key(&registry, &mut history, &mut doc, &ctx, 1.0);
    let action = doc.object(id).unwrap().action.unwrap();
    doc.action_mut(action)
        .unwrap()
        .curve_mut("rotation_euler", 2)
        .insert(11.0, std::f64::consts::TAU, kreate_scene::animation::Interpolation::Linear);
    doc.animate(6.0);
    let q = doc.object(id).unwrap().transform.rotation;
    // Halfway through a full turn about Z: facing backwards.
    assert!((q * DVec3::X - DVec3::NEG_X).length() < 1e-6, "{q:?}");
}

#[test]
fn keying_replays_from_the_history() {
    let (registry, mut history, mut doc, ctx) = wired();
    a_selected_empty(&mut doc);
    let start = doc.clone();
    key(&registry, &mut history, &mut doc, &ctx, 3.0);
    let script: String = history.entries().iter().map(|e| e.to_script_call()).collect::<Vec<_>>().join("\n");
    let mut replay = start;
    let mut fresh = History::new();
    let outcomes = fresh.run_script(&registry, &mut replay, None, &ctx, &script);
    assert!(outcomes.iter().all(|o| o.result.is_ok()), "{script}");
    let frames = |d: &Document| d.actions().next().map(|(_, a)| a.key_frames());
    assert_eq!(frames(&doc), frames(&replay));
    assert_eq!(frames(&replay), Some(vec![3.0]));
}

// --- the dope sheet's edits --------------------------------------------------

fn run(registry: &OperatorRegistry, history: &mut History, doc: &mut Document, ctx: &Context, op: &str, params: &[(&'static str, PropertyValue)]) -> bool {
    let mut p = PropertyMap::new();
    for (k, v) in params {
        p.set(k, v.clone());
    }
    let depth = history.entries().len();
    history.dispatch(registry, doc, ctx, op, &p).expect("ran");
    history.entries().len() > depth
}

/// A selected object keyed at 1, 10 and 20, with nothing selected in its keys.
fn keyed(registry: &OperatorRegistry, history: &mut History, doc: &mut Document, ctx: &Context) -> (kreate_scene::ObjectId, u64) {
    let id = a_selected_empty(doc);
    for f in [1.0, 10.0, 20.0] {
        key(registry, history, doc, ctx, f);
    }
    run(registry, history, doc, ctx, "anim.keys_select_all", &[("action", PropertyValue::Enum(1))]);
    let action = doc.object(id).unwrap().action.unwrap();
    (id, doc.action_uid(action).unwrap().raw())
}

fn selected_frames(doc: &Document, id: kreate_scene::ObjectId, property: &str) -> Vec<f64> {
    let action = doc.object(id).unwrap().action.unwrap();
    let mut out: Vec<f64> = doc
        .action(action)
        .unwrap()
        .curves
        .iter()
        .filter(|c| c.path.property.as_ref() == property)
        .flat_map(|c| c.keys.iter().filter(|k| k.selected).map(|k| k.frame))
        .collect();
    out.sort_by(f64::total_cmp);
    out.dedup();
    out
}

#[test]
fn a_box_selects_its_rows_between_its_frames() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (id, uid) = keyed(&registry, &mut history, &mut doc, &ctx);
    assert!(run(&registry, &mut history, &mut doc, &ctx, "anim.keys_select_box", &[
        ("targets", PropertyValue::Text(kreate_ops::ops::anim::row_target(uid, Some("location")))),
        ("frame_min", PropertyValue::Float(5.0)),
        ("frame_max", PropertyValue::Float(25.0)),
    ]));
    assert_eq!(selected_frames(&doc, id, "location"), vec![10.0, 20.0]);
    assert!(selected_frames(&doc, id, "scale").is_empty());
    // A click without extend replaces; with extend it adds.
    run(&registry, &mut history, &mut doc, &ctx, "anim.keys_select_box", &[
        ("targets", PropertyValue::Text(kreate_ops::ops::anim::row_target(uid, None))),
        ("frame_min", PropertyValue::Float(0.5)),
        ("frame_max", PropertyValue::Float(1.5)),
        ("extend", PropertyValue::Bool(true)),
    ]);
    assert_eq!(selected_frames(&doc, id, "location"), vec![1.0, 10.0, 20.0]);
    assert_eq!(selected_frames(&doc, id, "scale"), vec![1.0]);
}

#[test]
fn moving_keys_moves_the_animation_and_undo_puts_them_back() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (id, uid) = keyed(&registry, &mut history, &mut doc, &ctx);
    run(&registry, &mut history, &mut doc, &ctx, "anim.keys_select_box", &[
        ("targets", PropertyValue::Text(kreate_ops::ops::anim::row_target(uid, None))),
        ("frame_min", PropertyValue::Float(20.0)),
        ("frame_max", PropertyValue::Float(20.0)),
    ]);
    assert!(run(&registry, &mut history, &mut doc, &ctx, "anim.keys_move", &[("offset", PropertyValue::Float(5.0))]));
    let action = doc.object(id).unwrap().action.unwrap();
    assert_eq!(doc.action(action).unwrap().key_frames(), vec![1.0, 10.0, 25.0]);
    assert!(history.undo(&mut doc));
    assert_eq!(doc.action(action).unwrap().key_frames(), vec![1.0, 10.0, 20.0]);
    // Nothing selected: a move is a cancel.
    run(&registry, &mut history, &mut doc, &ctx, "anim.keys_select_all", &[("action", PropertyValue::Enum(1))]);
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "anim.keys_move", &[("offset", PropertyValue::Float(5.0))]));
}

#[test]
fn delete_and_interpolation_act_on_the_selection_only() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (id, uid) = keyed(&registry, &mut history, &mut doc, &ctx);
    run(&registry, &mut history, &mut doc, &ctx, "anim.keys_select_box", &[
        ("targets", PropertyValue::Text(kreate_ops::ops::anim::row_target(uid, None))),
        ("frame_min", PropertyValue::Float(1.0)),
        ("frame_max", PropertyValue::Float(1.0)),
    ]);
    assert!(run(&registry, &mut history, &mut doc, &ctx, "anim.keys_interpolation", &[("interpolation", PropertyValue::Enum(0))]));
    let action = doc.object(id).unwrap().action.unwrap();
    for curve in &doc.action(action).unwrap().curves {
        assert_eq!(curve.keys[0].interpolation, kreate_scene::animation::Interpolation::Constant);
        assert_eq!(curve.keys[1].interpolation, kreate_scene::animation::Interpolation::Bezier);
    }
    assert!(run(&registry, &mut history, &mut doc, &ctx, "anim.keys_delete", &[]));
    assert_eq!(doc.action(action).unwrap().key_frames(), vec![10.0, 20.0]);
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "anim.keys_delete", &[]), "nothing selected, yet a step");
}

#[test]
fn a_new_key_is_selected_so_the_next_edit_acts_on_it() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (id, _) = keyed(&registry, &mut history, &mut doc, &ctx);
    key(&registry, &mut history, &mut doc, &ctx, 30.0);
    assert_eq!(selected_frames(&doc, id, "location"), vec![30.0]);
}
