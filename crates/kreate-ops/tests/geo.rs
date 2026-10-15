//! Geometry nodes through their operators: on, built, wired, and evaluated
//! by the modifier cache as the viewport would draw them.

use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::modifier::{EvalCtx, ModifierCache};
use kreate_scene::Document;

fn run(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    op: &str,
    params: &[(&'static str, PropertyValue)],
) {
    let mut map = PropertyMap::new();
    for (k, v) in params {
        map.set(k, v.clone());
    }
    history
        .dispatch(registry, doc, &Context::default(), op, &map)
        .expect("ran");
}

#[test]
fn a_graph_built_by_operators_is_what_the_object_shows_and_undo_takes_it_back() {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
    let cube = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    doc.selection.add(cube);
    let faces = |doc: &Document| {
        ModifierCache::new()
            .evaluated(doc, cube, &EvalCtx::still())
            .unwrap()
            .num_faces()
    };
    run(&registry, &mut history, &mut doc, "geo.enable", &[]);
    assert_eq!(faces(&doc), 6, "the passthrough changes nothing");
    // An Array (kind 1) between the group's input and output.
    run(
        &registry,
        &mut history,
        &mut doc,
        "geo.add",
        &[("kind", PropertyValue::Enum(1))],
    );
    for (from, to, socket) in [(0, 2, 0), (2, 1, 0)] {
        run(
            &registry,
            &mut history,
            &mut doc,
            "geo.link",
            &[
                ("from_node", PropertyValue::Int(from)),
                ("to_node", PropertyValue::Int(to)),
                ("to_socket", PropertyValue::Int(socket)),
            ],
        );
    }
    assert_eq!(faces(&doc), 18, "three copies by default");
    run(
        &registry,
        &mut history,
        &mut doc,
        "geo.set_value",
        &[
            ("node", PropertyValue::Int(2)),
            ("field", PropertyValue::Int(0)),
            ("value", PropertyValue::Float(5.0)),
        ],
    );
    assert_eq!(faces(&doc), 30);
    assert!(history.undo(&mut doc));
    assert_eq!(faces(&doc), 18);
    // A wire that would loop is refused, and refusing changes nothing.
    let mut map = PropertyMap::new();
    map.set("from_node", PropertyValue::Int(2));
    map.set("to_node", PropertyValue::Int(2));
    map.set("to_socket", PropertyValue::Int(0));
    assert!(history
        .dispatch(&registry, &mut doc, &Context::default(), "geo.link", &map)
        .is_err());
    assert_eq!(faces(&doc), 18);
}

#[test]
fn every_kind_on_the_add_menu_is_the_one_added() {
    // The operator's list of kinds is the menu's: an entry past its end was
    // clamped to the last it knew, and a field node came in as a profile.
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    for (i, tag) in kreate_scene::geonodes::GeoNodeKind::ADDABLE
        .iter()
        .enumerate()
    {
        let mut history = History::new();
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
        let cube = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        doc.selection.add(cube);
        run(&registry, &mut history, &mut doc, "geo.enable", &[]);
        run(
            &registry,
            &mut history,
            &mut doc,
            "geo.add",
            &[("kind", PropertyValue::Enum(i as u32))],
        );
        let graph = doc.object(cube).unwrap().geometry_nodes.as_ref().unwrap();
        assert_eq!(graph.nodes.last().unwrap().kind.tag(), *tag);
    }
}
