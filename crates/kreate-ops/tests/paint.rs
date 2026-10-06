//! Vertex groups and weight strokes through the operators.

use kreate_ops::operator::Context;
use kreate_ops::ops::paint::{groups, weights};
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{Document, MeshId};

fn wired() -> (OperatorRegistry, History, Document, MeshId) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let id = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    doc.selection.add(id);
    (registry, History::new(), doc, mesh)
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
    history
        .dispatch(registry, doc, &Context::default(), op, &p)
        .expect("ran");
    history.entries().len() > depth
}

fn text(s: &str) -> PropertyValue {
    PropertyValue::Text(s.into())
}

#[test]
fn groups_are_made_numbered_and_the_new_one_is_active() {
    let (registry, mut history, mut doc, mesh) = wired();
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "object.vertex_group_add",
        &[]
    ));
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "object.vertex_group_add",
        &[]
    ));
    assert_eq!(groups(&doc, mesh), vec!["Group", "Group.001"]);
    let active = |doc: &Document| {
        doc.object(doc.selection.active.unwrap())
            .unwrap()
            .active_group
            .clone()
    };
    assert_eq!(active(&doc).as_deref(), Some("Group.001"));
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "object.vertex_group_set_active",
        &[("name", text("Group"))]
    ));
    assert_eq!(active(&doc).as_deref(), Some("Group"));
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "object.vertex_group_remove",
        &[("name", text("Group"))]
    ));
    assert_eq!(groups(&doc, mesh), vec!["Group.001"]);
    assert_eq!(
        active(&doc).as_deref(),
        Some("Group.001"),
        "removing the active group left nothing active"
    );
}

#[test]
fn a_stroke_moves_its_vertices_towards_the_weight_once_and_replays() {
    let (registry, mut history, mut doc, mesh) = wired();
    let start = doc.clone();
    // Half strength towards one on vertices 0 and 2 — vertex 2 twice, as a
    // brush crossing back over it would list it.
    let stroke = [
        ("group", text("Arm")),
        ("vertices", PropertyValue::Ints(vec![0, 2, 2])),
        ("weight", PropertyValue::Float(1.0)),
        ("strength", PropertyValue::Float(0.5)),
    ];
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "paint.weight_stroke",
        &stroke
    ));
    let w = weights(&doc, mesh, "Arm");
    assert_eq!(
        (w[0], w[1]),
        (0.5, 0.0),
        "the stroke made the group and painted only what it crossed"
    );
    assert_eq!(
        w[2], 0.5,
        "a vertex the brush crossed twice moved twice rather than once"
    );
    // Replayed from its script line, the same weights.
    let script: String = history
        .entries()
        .iter()
        .map(|e| e.to_script_call())
        .collect::<Vec<_>>()
        .join("\n");
    let mut replay = start;
    let outcomes =
        History::new().run_script(&registry, &mut replay, None, &Context::default(), &script);
    assert!(outcomes.iter().all(|o| o.result.is_ok()), "{script}");
    assert_eq!(weights(&replay, mesh, "Arm"), w);
}

#[test]
fn normalize_makes_each_vertex_sum_to_one() {
    let (registry, mut history, mut doc, mesh) = wired();
    for (group, weight) in [("A", 1.0), ("B", 1.0)] {
        run(
            &registry,
            &mut history,
            &mut doc,
            "paint.weight_stroke",
            &[
                ("group", text(group)),
                ("vertices", PropertyValue::Ints(vec![0])),
                ("weight", PropertyValue::Float(weight)),
                ("strength", PropertyValue::Float(1.0)),
            ],
        );
    }
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "paint.weights_normalize",
        &[]
    ));
    assert_eq!(weights(&doc, mesh, "A")[0], 0.5);
    assert_eq!(weights(&doc, mesh, "B")[0], 0.5);
    assert_eq!(
        weights(&doc, mesh, "A")[1],
        0.0,
        "a vertex with no weight got some"
    );
    assert!(
        !run(
            &registry,
            &mut history,
            &mut doc,
            "paint.weights_normalize",
            &[]
        ),
        "normalizing twice changed something"
    );
}

#[test]
fn a_soft_brush_paints_its_rim_less_than_its_middle() {
    let (registry, mut history, mut doc, mesh) = wired();
    let stroke = [
        ("group", text("Arm")),
        ("vertices", PropertyValue::Ints(vec![0, 1, 1])),
        ("weight", PropertyValue::Float(1.0)),
        ("strength", PropertyValue::Float(1.0)),
        // Vertex 1 crossed twice: the stronger pass counts, once.
        ("amounts", PropertyValue::Floats(vec![1.0, 0.25, 0.5])),
    ];
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "paint.weight_stroke",
        &stroke
    ));
    let w = weights(&doc, mesh, "Arm");
    assert_eq!((w[0], w[1]), (1.0, 0.5));
}

#[test]
fn a_texture_stroke_paints_under_the_brush_and_undo_takes_only_its_tiles_back() {
    use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
    use kreate_scene::document::{Object, ObjectData};
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut doc = kreate_scene::Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let cube = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    doc.selection.add(cube);
    let ctx = kreate_ops::operator::Context::default();
    let mut history = History::new();
    let mut p = PropertyMap::new();
    p.set("width", PropertyValue::Int(256));
    p.set("height", PropertyValue::Int(256));
    history
        .dispatch(&registry, &mut doc, &ctx, "paint.new_texture", &p)
        .unwrap();
    // The cube's first material now shows a new white image.
    let material = doc.mesh_slots(mesh)[0].expect("a material");
    let image = doc.material(material).unwrap().base_color_image.expect("an image");
    let pixel = |doc: &kreate_scene::Document, x: u32, y: u32| {
        let img = doc.image(image).unwrap();
        let t = kreate_scene::image::TILE;
        let at = (((y % t) * t + (x % t)) * 4) as usize;
        img.tile(x / t, y / t).map(|tile| tile.bytes()[at..at + 4].to_vec())
    };
    assert_eq!(pixel(&doc, 128, 128).unwrap(), vec![255, 255, 255, 255]);
    let before = doc.image(image).unwrap().revision();
    let handles_before: Vec<_> = (0..2)
        .flat_map(|ty| (0..2).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| doc.image(image).unwrap().tile_handle(tx, ty))
        .collect();
    // A red dab in the middle of the top-left tile: a tenth of the image wide.
    let uid = doc.image_uid(image).unwrap().raw().to_string();
    let mut p = PropertyMap::new();
    p.set("image", PropertyValue::Text(uid));
    p.set("dabs", PropertyValue::Floats(vec![0.25, 0.25, 0.05, 1.0]));
    p.set("color", PropertyValue::Vec3([1.0, 0.0, 0.0]));
    history
        .dispatch(&registry, &mut doc, &ctx, "paint.texture_stroke", &p)
        .unwrap();
    // Red at the middle — the pixel's centre is half a pixel off the dab's,
    // where the falloff is a hair under full.
    let middle = pixel(&doc, 64, 64).unwrap();
    assert!(middle[0] == 255 && middle[1] <= 3 && middle[2] <= 3, "{middle:?}");
    assert_eq!(pixel(&doc, 64 + 20, 64).unwrap(), vec![255, 255, 255, 255], "outside, untouched");
    let edge = pixel(&doc, 64 + 10, 64).unwrap();
    assert!(edge[1] > 0 && edge[1] < 255, "the falloff is soft: {edge:?}");
    assert!(doc.image(image).unwrap().revision() > before);
    // Only the tile it touched was replaced.
    let after: Vec<_> = (0..2)
        .flat_map(|ty| (0..2).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| doc.image(image).unwrap().tile_handle(tx, ty))
        .collect();
    let replaced = handles_before
        .iter()
        .zip(&after)
        .filter(|(a, b)| !std::sync::Arc::ptr_eq(a.as_ref().unwrap(), b.as_ref().unwrap()))
        .count();
    assert_eq!(replaced, 1);
    assert!(history.undo(&mut doc));
    assert_eq!(pixel(&doc, 64, 64).unwrap(), vec![255, 255, 255, 255]);
}
