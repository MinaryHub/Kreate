//! Editing a material's graph through its operators.
//!
//! What the node editor's gestures become. The editor is tested for what it
//! *reports*; these are tested for what the report *does* — and for the
//! promises every edit makes: it undoes, it replays, and it never leaves a
//! graph without its surface.

use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::shader::NodeKind;
use kreate_scene::{Document, Material, MaterialId};

fn wired() -> (OperatorRegistry, History, Document, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (registry, History::new(), Document::new(), Context::default())
}

/// A selected cube with a textured material in its first slot.
fn textured_cube(doc: &mut Document) -> (MaterialId, kreate_scene::document::ImageId) {
    let image = doc.add_image(kreate_scene::Image::new(
        "Bricks",
        4,
        4,
        kreate_scene::PixelFormat::Rgba8,
    ));
    let material = doc.add_material(Material {
        name: "Brick".into(),
        base_color_image: Some(image),
        ..Default::default()
    });
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    doc.set_mesh_slots(mesh, vec![Some(material)]);
    let object = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    doc.selection.add(object);
    (material, image)
}

fn run(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    ctx: &Context,
    op: &str,
    params: &[(&'static str, PropertyValue)],
) -> bool {
    let mut map = PropertyMap::new();
    for (k, v) in params {
        map.set(k, v.clone());
    }
    let depth = history.entries().len();
    history.dispatch(registry, doc, ctx, op, &map).expect("the operator ran");
    history.entries().len() > depth
}

fn graph(doc: &Document, material: MaterialId) -> kreate_scene::shader::ShaderGraph {
    doc.material(material)
        .and_then(|m| m.graph.clone())
        .expect("the material has a graph")
}

#[test]
fn use_nodes_builds_the_graph_the_material_already_was() {
    // Turning nodes on must change no pixel: the texture and its tint are
    // there as nodes. A fresh surface instead would drop the texture the
    // moment somebody opened the editor to look.
    let (registry, mut history, mut doc, ctx) = wired();
    let (material, image) = textured_cube(&mut doc);
    let before = doc.material(material).unwrap().shading_graph();
    assert!(run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[
        ("enabled", PropertyValue::Bool(true)),
    ]));
    assert_eq!(graph(&doc, material), before);
    assert!(graph(&doc, material).images().contains(&image));

    // Off takes it away, and the flat fields draw again; undo brings it back.
    assert!(run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[
        ("enabled", PropertyValue::Bool(false)),
    ]));
    assert!(doc.material(material).unwrap().graph.is_none());
    assert!(history.undo(&mut doc));
    assert!(doc.material(material).unwrap().graph.is_some());
    // And asking for what is already so leaves no step to undo.
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[
        ("enabled", PropertyValue::Bool(true)),
    ]));
}

#[test]
fn a_node_is_added_where_it_was_asked_and_undo_takes_it_away() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (material, _) = textured_cube(&mut doc);
    run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[]);
    let count = graph(&doc, material).nodes.len();
    for (index, tag) in NodeKind::ADDABLE.iter().enumerate() {
        assert!(run(&registry, &mut history, &mut doc, &ctx, "node.add", &[
            ("kind", PropertyValue::Enum(index as u32)),
            ("x", PropertyValue::Float(-300.0)),
            ("y", PropertyValue::Float(40.0)),
        ]));
        let g = graph(&doc, material);
        let added = g.nodes.last().unwrap();
        assert_eq!(added.kind.tag(), *tag, "the menu's index {index} added the wrong kind");
        assert_eq!(added.position, [-300.0, 40.0]);
    }
    assert_eq!(graph(&doc, material).nodes.len(), count + NodeKind::ADDABLE.len());
    assert!(history.undo(&mut doc));
    assert_eq!(graph(&doc, material).nodes.len(), count + NodeKind::ADDABLE.len() - 1);
}

#[test]
fn the_surface_can_be_neither_removed_nor_doubled() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (material, _) = textured_cube(&mut doc);
    run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[]);
    let surface = graph(&doc, material).output().unwrap() as u32;
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "node.remove", &[
        ("nodes", PropertyValue::Ints(vec![surface])),
    ]));
    assert!(graph(&doc, material).output().is_some());
}

#[test]
fn removing_nodes_takes_their_wires_and_keeps_the_rest_pointing_right() {
    // Several at once, in any order: each removal renumbers the nodes after
    // it, and the operator must not remove a neighbour of the one it meant.
    let (registry, mut history, mut doc, ctx) = wired();
    let (material, _) = textured_cube(&mut doc);
    run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[]);
    let g = graph(&doc, material);
    let names: Vec<String> = g.nodes.iter().map(|n| n.name.clone()).collect();
    let image = g.nodes.iter().position(|n| matches!(n.kind, NodeKind::Image { .. })).unwrap();
    let tint = g.nodes.iter().position(|n| matches!(n.kind, NodeKind::Color(_))).unwrap();
    assert!(run(&registry, &mut history, &mut doc, &ctx, "node.remove", &[
        ("nodes", PropertyValue::Ints(vec![tint as u32, image as u32])),
    ]));
    let after = graph(&doc, material);
    let left: Vec<String> = after.nodes.iter().map(|n| n.name.clone()).collect();
    let want: Vec<String> = names
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != image && *i != tint)
        .map(|(_, n)| n.clone())
        .collect();
    assert_eq!(left, want);
    for link in &after.links {
        assert!(link.from_node < after.nodes.len() && link.to_node < after.nodes.len());
    }
    // The multiply lost both its inputs, and the surface kept its wire from it.
    assert_eq!(after.links.len(), 1);
}

#[test]
fn a_wire_goes_where_it_is_dropped_and_a_wrong_one_is_refused_quietly() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (material, _) = textured_cube(&mut doc);
    run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[]);
    // A Value node, wired into the roughness.
    run(&registry, &mut history, &mut doc, &ctx, "node.add", &[("kind", PropertyValue::Enum(1))]);
    let g = graph(&doc, material);
    let value = (g.nodes.len() - 1) as u32;
    let surface = g.output().unwrap() as u32;
    let link = |from_socket: u32, to_socket: u32| {
        vec![
            ("from_node", PropertyValue::Int(value as i64)),
            ("from_socket", PropertyValue::Int(from_socket as i64)),
            ("to_node", PropertyValue::Int(surface as i64)),
            ("to_socket", PropertyValue::Int(to_socket as i64)),
        ]
    };
    assert!(run(&registry, &mut history, &mut doc, &ctx, "node.link", &link(0, 2)));
    assert!(graph(&doc, material).incoming(surface as usize, 2).is_some());
    // A number into a colour: refused, and nothing to undo.
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "node.link", &link(0, 0)));
    // The same wire again: nothing changes, nothing to undo.
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "node.link", &link(0, 2)));

    assert!(run(&registry, &mut history, &mut doc, &ctx, "node.unlink", &[
        ("to_node", PropertyValue::Int(surface as i64)),
        ("to_socket", PropertyValue::Int(2)),
    ]));
    assert!(graph(&doc, material).incoming(surface as usize, 2).is_none());
}

#[test]
fn a_drag_is_one_step_and_moves_every_node_once() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (material, _) = textured_cube(&mut doc);
    run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[]);
    let before = graph(&doc, material);
    assert!(run(&registry, &mut history, &mut doc, &ctx, "node.translate", &[
        // Named twice: moved once.
        ("nodes", PropertyValue::Ints(vec![0, 1, 1])),
        ("x", PropertyValue::Float(10.0)),
        ("y", PropertyValue::Float(-5.0)),
    ]));
    let after = graph(&doc, material);
    for i in [0usize, 1] {
        assert_eq!(after.nodes[i].position[0], before.nodes[i].position[0] + 10.0);
        assert_eq!(after.nodes[i].position[1], before.nodes[i].position[1] - 5.0);
    }
    assert_eq!(after.nodes[2].position, before.nodes[2].position);
}

#[test]
fn a_nodes_numbers_and_image_are_set_through_their_fields() {
    let (registry, mut history, mut doc, ctx) = wired();
    let (material, image) = textured_cube(&mut doc);
    run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[]);
    let surface = graph(&doc, material).output().unwrap() as i64;
    assert!(run(&registry, &mut history, &mut doc, &ctx, "node.set_value", &[
        ("node", PropertyValue::Int(surface)),
        ("field", PropertyValue::Int(2)),
        ("value", PropertyValue::Float(0.9)),
    ]));
    assert!(run(&registry, &mut history, &mut doc, &ctx, "node.set_color", &[
        ("node", PropertyValue::Int(surface)),
        ("field", PropertyValue::Int(0)),
        ("color", PropertyValue::Vec3([1.0, 0.0, 0.0])),
    ]));
    match graph(&doc, material).nodes[surface as usize].kind {
        NodeKind::Surface { roughness, base_color, .. } => {
            assert!((roughness - 0.9).abs() < 1e-6);
            assert_eq!(base_color.r, 1.0);
            assert_eq!(base_color.g, 0.0);
        }
        _ => panic!("the surface moved"),
    }
    // A field the node does not have: refused.
    assert!(!run(&registry, &mut history, &mut doc, &ctx, "node.set_value", &[
        ("node", PropertyValue::Int(surface)),
        ("field", PropertyValue::Int(7)),
        ("value", PropertyValue::Float(0.1)),
    ]));

    // An image node, emptied and filled again by identity.
    let node = graph(&doc, material)
        .nodes
        .iter()
        .position(|n| matches!(n.kind, NodeKind::Image { .. }))
        .unwrap() as i64;
    let uid = doc.image_uid(image).unwrap().raw().to_string();
    assert!(run(&registry, &mut history, &mut doc, &ctx, "node.set_image", &[
        ("node", PropertyValue::Int(node)),
        ("uid", PropertyValue::Text(String::new())),
    ]));
    assert!(graph(&doc, material).images().is_empty());
    assert!(run(&registry, &mut history, &mut doc, &ctx, "node.set_image", &[
        ("node", PropertyValue::Int(node)),
        ("uid", PropertyValue::Text(uid)),
    ]));
    assert_eq!(graph(&doc, material).images(), vec![image]);
}

#[test]
fn every_graph_edit_replays_from_the_history() {
    // The history's script, run on a fresh copy of where it started, ends
    // where the edits did — the promise every operator makes.
    let (registry, mut history, mut doc, ctx) = wired();
    textured_cube(&mut doc);
    let start = doc.clone();
    run(&registry, &mut history, &mut doc, &ctx, "material.use_nodes", &[]);
    run(&registry, &mut history, &mut doc, &ctx, "node.add", &[
        ("kind", PropertyValue::Enum(6)),
        ("x", PropertyValue::Float(-500.0)),
    ]);
    run(&registry, &mut history, &mut doc, &ctx, "node.translate", &[
        ("nodes", PropertyValue::Ints(vec![0])),
        ("x", PropertyValue::Float(3.0)),
    ]);
    let script: String = history
        .entries()
        .iter()
        .map(|e| e.to_script_call())
        .collect::<Vec<_>>()
        .join("\n");

    let mut replay = start;
    let mut fresh = History::new();
    let outcomes = fresh.run_script(&registry, &mut replay, None, &ctx, &script);
    assert!(outcomes.iter().all(|o| o.result.is_ok()), "{script}");
    let a = doc.materials().next().unwrap().1.graph.clone();
    let b = replay.materials().next().unwrap().1.graph.clone();
    assert_eq!(a, b, "the replay built a different graph:\n{script}");
}
