//! Shape keys through the operators: added, edited through the trade edit
//! mode makes, turned up, and drawn.

use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::modifier::{EvalCtx, ModifierCache};
use kreate_scene::Document;

fn run(registry: &OperatorRegistry, history: &mut History, doc: &mut Document, op: &str, pairs: &[(&'static str, PropertyValue)]) {
    let mut p = PropertyMap::new();
    for (k, v) in pairs {
        p.set(k, v.clone());
    }
    history.dispatch(registry, doc, &Context::default(), op, &p).expect("ran");
}

#[test]
fn a_key_made_edited_and_turned_up_is_what_is_drawn() {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
    let id = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    doc.selection.add(id);

    run(&registry, &mut history, &mut doc, "object.shape_key_add", &[]);
    assert_eq!(doc.object(id).unwrap().active_shape.as_deref(), Some("Key 1"));
    // Into edit mode on the key, move the top up, and out again.
    run(&registry, &mut history, &mut doc, "object.shape_key_edit", &[("enter", PropertyValue::Bool(true))]);
    for p in doc.mesh_mut(mesh).unwrap().positions_mut() {
        if p.z > 0.0 {
            p.z += 2.0;
        }
    }
    run(&registry, &mut history, &mut doc, "object.shape_key_edit", &[("enter", PropertyValue::Bool(false))]);
    assert!(doc.object(id).unwrap().editing_shape.is_none());
    assert_eq!(doc.mesh(mesh).unwrap().bounds().max.z, 1.0, "the basis took the key's edit");

    let mut cache = ModifierCache::new();
    let drawn = |doc: &Document, cache: &mut ModifierCache| cache.evaluated(doc, id, &EvalCtx::still()).unwrap().bounds().max.z;
    assert_eq!(drawn(&doc, &mut cache), 1.0, "a key at nothing moved the mesh");
    run(&registry, &mut history, &mut doc, "object.shape_key_set", &[("name", PropertyValue::Text("Key 1".into())), ("value", PropertyValue::Float(0.5))]);
    assert!((drawn(&doc, &mut cache) - 2.0).abs() < 1e-5);
    // Renamed, it still drives the mesh.
    run(
        &registry,
        &mut history,
        &mut doc,
        "object.shape_key_set",
        &[("name", PropertyValue::Text("Key 1".into())), ("value", PropertyValue::Float(1.0)), ("new_name", PropertyValue::Text("Tall".into()))],
    );
    assert!((drawn(&doc, &mut cache) - 3.0).abs() < 1e-5);
    assert_eq!(doc.object(id).unwrap().active_shape.as_deref(), Some("Tall"));
    // Removed, the mesh is the basis again.
    run(&registry, &mut history, &mut doc, "object.shape_key_remove", &[("name", PropertyValue::Text("Tall".into()))]);
    assert_eq!(drawn(&doc, &mut cache), 1.0);
    assert!(!doc.mesh(mesh).unwrap().attributes().contains("shape:Tall"));
}

#[test]
fn a_keyed_shape_value_plays_back() {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
    let id = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    doc.selection.add(id);
    run(&registry, &mut history, &mut doc, "object.shape_key_add", &[]);
    let path = kreate_scene::animation::shape_path("Key 1");
    let key = |doc: &mut Document, history: &mut History, frame: f64, value: f64| {
        run(&registry, history, doc, "object.shape_key_set", &[("name", PropertyValue::Text("Key 1".into())), ("value", PropertyValue::Float(value))]);
        run(
            &registry,
            history,
            doc,
            "anim.keyframe_insert_property",
            &[("frame", PropertyValue::Float(frame)), ("datablock", PropertyValue::Enum(2)), ("property", PropertyValue::Text(path.clone()))],
        );
    };
    key(&mut doc, &mut history, 1.0, 0.0);
    key(&mut doc, &mut history, 11.0, 1.0);
    doc.animate(6.0);
    let v = doc.object(id).unwrap().shape_keys[0].value;
    assert!(v > 0.2 && v < 0.8, "{v}");
    doc.animate(11.0);
    assert_eq!(doc.object(id).unwrap().shape_keys[0].value, 1.0);
}

#[test]
fn a_sculpt_stroke_raises_the_surface_and_replays_from_its_line() {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::grid(4.0, 16, 16));
    let id = doc.add_object(Object::new("Grid", ObjectData::Mesh(mesh)));
    doc.selection.add(id);
    let start = doc.clone();
    run(
        &registry,
        &mut history,
        &mut doc,
        "sculpt.stroke",
        &[
            ("dabs", PropertyValue::Floats(vec![0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0])),
            ("strength", PropertyValue::Float(1.0)),
        ],
    );
    let top = |d: &Document| d.mesh(mesh).unwrap().positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    assert!(top(&doc) > 0.1);
    let script = history.entries()[0].to_script_call();
    let mut replay = start;
    let out = History::new().run_script(&registry, &mut replay, None, &Context::default(), &script);
    assert!(out.iter().all(|o| o.result.is_ok()), "{script}");
    assert_eq!(top(&replay), top(&doc));
}
