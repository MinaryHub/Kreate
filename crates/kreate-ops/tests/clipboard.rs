//! Copy and paste, as the operator sees them.
//!
//! Copy is not an edit and leaves no history; it turns the selection into a
//! fragment and the application keeps the text. Paste *is* an edit, and it
//! carries the fragment's own bytes in its parameters — so it can be undone,
//! redone and replayed from the history a week later, with the copied objects
//! long gone, and produce what it produced the first time.

use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

fn wired() -> (OperatorRegistry, History, Document, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (registry, History::new(), Document::new(), Context::default())
}

fn cube(doc: &mut Document, name: &str) -> kreate_scene::ObjectId {
    doc.add_mesh_object(name, kreate_mesh::primitives::cube(2.0))
}

/// What Ctrl+C does: the selection, as text.
fn copy(doc: &Document) -> String {
    let fragment = doc.extract_objects(&doc.selection.objects);
    kreate_io::clipboard::encode(&fragment).expect("encodes")
}

/// What Ctrl+V does: the text, through the operator.
fn paste(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    ctx: &Context,
    text: &str,
) -> bool {
    let mut params = PropertyMap::new();
    params.set("data", PropertyValue::Text(text.to_string()));
    history
        .dispatch(registry, doc, ctx, "object.paste", &params)
        .is_ok()
}

#[test]
fn a_pasted_object_is_a_new_object_and_becomes_the_selection() {
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    let _b = cube(&mut doc, "B");
    doc.selection.add(a);
    let text = copy(&doc);

    assert!(paste(&registry, &mut history, &mut doc, &ctx, &text));
    assert_eq!(doc.object_count(), 3);
    // The paste is what is selected now, and only it — the next move moves
    // the paste, which is what a person pasting something is about to do.
    assert_eq!(doc.selection.objects.len(), 1);
    let pasted = doc.selection.objects[0];
    assert_ne!(pasted, a);
    assert_eq!(doc.selection.active, Some(pasted));
    assert_eq!(doc.object(pasted).unwrap().name, "A.001");
    assert_eq!(doc.object(a).unwrap().name, "A", "the original is untouched");
}

#[test]
fn pasting_twice_gives_two() {
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    doc.selection.add(a);
    let text = copy(&doc);
    assert!(paste(&registry, &mut history, &mut doc, &ctx, &text));
    assert!(paste(&registry, &mut history, &mut doc, &ctx, &text));
    assert_eq!(doc.object_count(), 3);
    let uids: std::collections::HashSet<_> = doc
        .objects()
        .map(|(id, _)| doc.object_uid(id).unwrap())
        .collect();
    assert_eq!(uids.len(), 3, "every paste has its own identity");
}

#[test]
fn a_paste_survives_the_originals_being_deleted() {
    // The reason the bytes ride in the parameters: the clipboard outlives what
    // was copied, and so does the history entry.
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    doc.selection.add(a);
    let text = copy(&doc);
    history
        .dispatch(&registry, &mut doc, &ctx, "object.delete", &PropertyMap::new())
        .expect("delete");
    assert_eq!(doc.object_count(), 0);

    assert!(paste(&registry, &mut history, &mut doc, &ctx, &text));
    assert_eq!(doc.object_count(), 1);
    assert_eq!(doc.objects().next().unwrap().1.name, "A.001");
}

#[test]
fn undo_takes_the_paste_back_and_redo_brings_it_again() {
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    doc.selection.add(a);
    let text = copy(&doc);
    assert!(paste(&registry, &mut history, &mut doc, &ctx, &text));
    assert_eq!(doc.object_count(), 2);
    assert!(history.undo(&mut doc));
    assert_eq!(doc.object_count(), 1, "undo did not remove the paste");
    assert!(history.redo(&registry, &mut doc, &ctx).is_ok());
    assert_eq!(doc.object_count(), 2, "redo did not bring it back");
}

#[test]
fn text_that_is_not_a_fragment_is_declined_without_a_history_step() {
    // Anything another application left on the clipboard. Declined, not an
    // error: "that was not ours" is the normal case for a paste.
    let (registry, mut history, mut doc, ctx) = wired();
    cube(&mut doc, "A");
    let depth = history.entries().len();
    assert!(paste(&registry, &mut history, &mut doc, &ctx, "not a fragment"));
    assert!(paste(&registry, &mut history, &mut doc, &ctx, ""));
    assert_eq!(doc.object_count(), 1);
    assert_eq!(history.entries().len(), depth, "a declined paste left a step to undo");
}

#[test]
fn a_material_and_its_texture_come_along() {
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    let image = doc.add_image(kreate_scene::Image::new(
        "Bricks",
        4,
        4,
        kreate_scene::PixelFormat::Rgba8,
    ));
    let material = doc.add_material(kreate_scene::Material {
        name: "Brick".into(),
        base_color_image: Some(image),
        ..Default::default()
    });
    let mesh = doc.object(a).unwrap().mesh().unwrap();
    doc.set_mesh_slots(mesh, vec![Some(material)]);
    doc.selection.add(a);
    let text = copy(&doc);

    assert!(paste(&registry, &mut history, &mut doc, &ctx, &text));
    let pasted = doc.selection.active.expect("the paste is active");
    let pasted_material = doc.material_at(pasted, 0).expect("the paste has a material");
    assert_eq!(pasted_material.name, "Brick");
    assert!(pasted_material.base_color_image.is_some(), "the texture did not come along");
    assert_ne!(
        doc.material_id_at(pasted, 0),
        Some(material),
        "the paste's material is its own, not the original's"
    );
}

#[test]
fn a_lamp_can_be_copied_and_pasted() {
    let (registry, mut history, mut doc, ctx) = wired();
    let light = doc.add_light(kreate_scene::Light::of_kind(
        "Bulb",
        kreate_scene::LightKind::BULB,
    ));
    let lamp = doc.add_object(Object::new("Lamp", ObjectData::Light(light)));
    doc.selection.add(lamp);
    let text = copy(&doc);
    assert!(paste(&registry, &mut history, &mut doc, &ctx, &text));
    assert_eq!(doc.light_count(), 2);
}

#[test]
fn every_map_comes_along_and_a_shared_image_stays_shared() {
    // One file used as both the colour and the packed map — an ordinary thing
    // to do with a grey mask. It is one image in the paste too, as it was here.
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    let mask = doc.add_image(kreate_scene::Image::new("Mask", 4, 4, kreate_scene::PixelFormat::Rgba8));
    let normal = doc.add_image(kreate_scene::Image::new("N", 4, 4, kreate_scene::PixelFormat::Rgba8));
    let material = doc.add_material(kreate_scene::Material {
        base_color_image: Some(mask),
        metallic_roughness_image: Some(mask),
        normal_image: Some(normal),
        ..Default::default()
    });
    let mesh = doc.object(a).unwrap().mesh().unwrap();
    doc.set_mesh_slots(mesh, vec![Some(material)]);
    doc.selection.add(a);
    let text = copy(&doc);
    let images_before = doc.images().count();

    assert!(paste(&registry, &mut history, &mut doc, &ctx, &text));
    let pasted = doc.selection.active.expect("the paste is active");
    let m = doc.material_at(pasted, 0).expect("a material");
    assert!(m.normal_image.is_some(), "the normal map did not come along");
    assert_eq!(m.base_color_image, m.metallic_roughness_image, "a shared image was split");
    assert_eq!(doc.images().count(), images_before + 2, "two images, not three");
}
