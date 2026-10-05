//! Everything the Add menu offers can actually be added.
//!
//! Blender's Add menu is Mesh ▸ eight shapes, Light ▸ four kinds, Camera,
//! Empty. Ours mirrors it, and each entry sends an enum index or an operator
//! id — so this holds the menu's indices to the operators' by adding every one
//! and looking at what arrived.

use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::ObjectData;
use kreate_scene::Document;

fn wired() -> (OperatorRegistry, History, Document, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (
        registry,
        History::new(),
        Document::new(),
        Context::default(),
    )
}

#[test]
fn every_primitive_the_menu_lists_adds_a_mesh_object_of_that_shape() {
    let (registry, mut history, mut doc, ctx) = wired();
    for (index, name) in kreate_ops::ops::object::PRIMITIVES.iter().enumerate() {
        let mut params = PropertyMap::new();
        params.set("shape", PropertyValue::Enum(index as u32));
        history
            .dispatch(&registry, &mut doc, &ctx, "object.add_primitive", &params)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let added = doc.selection.active.expect("the new object is active");
        let object = doc.object(added).expect("it exists");
        assert!(
            matches!(object.data, ObjectData::Mesh(_)),
            "{name} is not a mesh"
        );
        let mesh = doc.mesh(object.mesh().unwrap()).unwrap();
        assert!(mesh.num_faces() > 0, "{name} has no faces");
    }
    assert_eq!(
        doc.object_count(),
        kreate_ops::ops::object::PRIMITIVES.len()
    );
    // Distinct shapes, not eight cubes: the face counts must not all agree.
    let mut faces: Vec<u32> = doc
        .objects()
        .map(|(_, o)| doc.mesh(o.mesh().unwrap()).unwrap().num_faces())
        .collect();
    faces.sort_unstable();
    faces.dedup();
    assert!(
        faces.len() >= 5,
        "the shapes are too alike to be eight shapes: {faces:?}"
    );
}

#[test]
fn an_empty_can_be_added_and_is_selected() {
    // Blender's Add ▸ Empty ▸ Plain Axes: an object with a place and nothing
    // else, which is what a parent for a rig or a target for a constraint is.
    let (registry, mut history, mut doc, ctx) = wired();
    history
        .dispatch(
            &registry,
            &mut doc,
            &ctx,
            "object.add_empty",
            &PropertyMap::new(),
        )
        .expect("adds");
    let added = doc.selection.active.expect("the empty is active");
    assert!(matches!(doc.object(added).unwrap().data, ObjectData::Empty));
    assert_eq!(
        doc.selection.objects,
        vec![added],
        "the empty is the whole selection"
    );
    assert!(history.undo(&mut doc));
    assert_eq!(doc.object_count(), 0, "undo did not take it away");
}

#[test]
fn a_camera_from_the_menu_arrives_selected() {
    let (registry, mut history, mut doc, ctx) = wired();
    history
        .dispatch(
            &registry,
            &mut doc,
            &ctx,
            "object.add_camera",
            &PropertyMap::new(),
        )
        .expect("adds");
    let added = doc.selection.active.expect("active");
    assert!(matches!(
        doc.object(added).unwrap().data,
        ObjectData::Camera(_)
    ));
}
