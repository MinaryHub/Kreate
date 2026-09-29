//! Drivers through the operators: made, edited, undone and replayed.

use kreate_ops::operator::Context;
use kreate_ops::ops::driver::driver_target;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

fn wired() -> (OperatorRegistry, History, Document, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (registry, History::new(), Document::new(), Context::default())
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

fn text(s: &str) -> PropertyValue {
    PropertyValue::Text(s.to_string())
}

/// A handle and a door, the door's Z rotation to be driven by the handle's X.
fn scene() -> (Document, kreate_scene::ObjectId, kreate_scene::ObjectId, String) {
    let mut doc = Document::new();
    let handle = doc.add_object(Object::new("Handle", ObjectData::Empty));
    let mut door_object = Object::new("Door", ObjectData::Empty);
    door_object.transform.rotation = kreate_math::DQuat::from_rotation_z(0.25);
    let door = doc.add_object(door_object);
    let target = driver_target(doc.object_uid(door).unwrap().raw(), "rotation_euler", 2);
    (doc, handle, door, target)
}

fn door_z(doc: &Document, door: kreate_scene::ObjectId) -> f64 {
    kreate_scene::animation::to_euler(doc.object(door).unwrap().transform.rotation).z
}

/// The full rig, through the operators, as a panel would build it.
fn rig(registry: &OperatorRegistry, history: &mut History, doc: &mut Document, ctx: &Context, handle: kreate_scene::ObjectId, target: &str) {
    assert!(run(registry, history, doc, ctx, "anim.driver_add", &[("target", text(target))]));
    assert!(run(registry, history, doc, ctx, "anim.driver_variable_add", &[("target", text(target))]));
    let uid = doc.object_uid(handle).unwrap().raw().to_string();
    assert!(run(
        registry,
        history,
        doc,
        ctx,
        "anim.driver_variable_set",
        &[
            ("target", text(target)),
            ("variable", PropertyValue::Int(0)),
            ("name", text("h")),
            ("object", text(&uid)),
            ("property", PropertyValue::Enum(0)),
            ("index", PropertyValue::Int(0)),
        ],
    ));
    assert!(run(
        registry,
        history,
        doc,
        ctx,
        "anim.driver_expression",
        &[("target", text(target)), ("expression", text("h * 2"))],
    ));
}

#[test]
fn a_new_driver_holds_the_value_it_found() {
    // Adding a driver is not a request to move anything: its first expression
    // is the property's value now.
    let (registry, mut history, mut doc, ctx) = wired();
    let (mut d, _, door, target) = scene();
    std::mem::swap(&mut doc, &mut d);
    assert!(run(&registry, &mut history, &mut doc, &ctx, "anim.driver_add", &[("target", text(&target))]));
    doc.drive();
    assert!((door_z(&doc, door) - 0.25).abs() < 1e-12, "adding the driver turned the door");
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "anim.driver_add", &[("target", text(&target))]),
        "a second driver on one component");
}

#[test]
fn a_rig_built_from_the_operators_follows_its_handle_and_undoes() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (mut d, handle, door, target) = scene();
    std::mem::swap(&mut doc, &mut d);
    rig(&registry, &mut history, &mut doc, &ctx, handle, &target);
    doc.object_mut(handle).unwrap().transform.translation.x = 0.4;
    doc.drive();
    assert!((door_z(&doc, door) - 0.8).abs() < 1e-9);

    // Four steps, and undoing the expression takes the door back to what the
    // first expression said.
    assert!(history.undo(&mut doc));
    doc.drive();
    assert!((door_z(&doc, door) - 0.25).abs() < 1e-9, "the old expression did not come back");
    assert!(history.undo(&mut doc));
    assert!(history.undo(&mut doc));
    assert!(history.undo(&mut doc));
    assert!(doc.object(door).unwrap().drivers.is_empty());
}

#[test]
fn a_rig_replays_from_the_history() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (mut d, handle, door, target) = scene();
    std::mem::swap(&mut doc, &mut d);
    let start = doc.clone();
    rig(&registry, &mut history, &mut doc, &ctx, handle, &target);
    let script: String = history.entries().iter().map(|e| e.to_script_call()).collect::<Vec<_>>().join("\n");
    let mut replay = start;
    let outcomes = History::new().run_script(&registry, &mut replay, None, &ctx, &script);
    assert!(outcomes.iter().all(|o| o.result.is_ok()), "{script}");
    assert_eq!(replay.object(door).unwrap().drivers, doc.object(door).unwrap().drivers);
}

#[test]
fn variables_are_numbered_names_and_come_off_again() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (mut d, _, door, target) = scene();
    std::mem::swap(&mut doc, &mut d);
    run(&registry, &mut history, &mut doc, &ctx, "anim.driver_add", &[("target", text(&target))]);
    for _ in 0..3 {
        run(&registry, &mut history, &mut doc, &ctx, "anim.driver_variable_add", &[("target", text(&target))]);
    }
    let names = |doc: &Document| -> Vec<String> {
        doc.object(door).unwrap().drivers[0].variables.iter().map(|v| v.name.clone()).collect()
    };
    assert_eq!(names(&doc), vec!["var", "var_1", "var_2"]);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        &ctx,
        "anim.driver_variable_remove",
        &[("target", text(&target)), ("variable", PropertyValue::Int(1))]
    ));
    assert_eq!(names(&doc), vec!["var", "var_2"]);
    assert!(run(&registry, &mut history, &mut doc, &ctx, "anim.driver_remove", &[("target", text(&target))]));
    assert!(doc.object(door).unwrap().drivers.is_empty());
}

#[test]
fn a_target_that_names_nothing_is_refused() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (mut d, _, door, _) = scene();
    std::mem::swap(&mut doc, &mut d);
    let uid = doc.object_uid(door).unwrap().raw();
    for bad in [driver_target(uid, "colour", 0), driver_target(uid, "location", 3), driver_target(999, "location", 0), "junk".into()] {
        assert!(!run(&registry, &mut history, &mut doc, &ctx, "anim.driver_add", &[("target", text(&bad))]), "{bad}");
    }
}
