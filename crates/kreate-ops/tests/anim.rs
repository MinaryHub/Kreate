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
