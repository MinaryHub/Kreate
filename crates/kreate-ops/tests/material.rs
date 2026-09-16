//! A material is a datablock, and a face points at a slot rather than at it.
//!
//! §1.4's argument again, with one addition the others did not need. A mesh's
//! faces already carry a `material_index`, and that index has to mean something
//! — so the question is what it indexes.
//!
//! **A per-mesh slot list, not the document's arena.** Two reasons, and either
//! alone would settle it: a `MeshData` has to be transferable between
//! documents, where an arena id means nothing; and `edit::join` already offsets
//! each part's indices as it merges meshes, which is only coherent against a
//! list that belongs to the mesh.
//!
//! **Slots on the mesh, overrides on the object.** A linked copy shares the
//! mesh, so it shares the slots — assigning a material to one copy assigns it
//! to all of them, which is what an artist means by "the material of this
//! chair". The override is how one copy differs, and it is modelled now
//! because retrofitting it means auditing every material lookup.

use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{Document, Material, MeshId};

fn cube(doc: &mut Document) -> (MeshId, kreate_scene::ObjectId) {
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let object = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    (mesh, object)
}

fn red(doc: &mut Document) -> kreate_scene::MaterialId {
    doc.add_material(Material {
        name: "Red".into(),
        base_color: kreate_math::SceneRgb::new(0.8, 0.1, 0.1),
        ..Material::default()
    })
}

fn blue(doc: &mut Document) -> kreate_scene::MaterialId {
    doc.add_material(Material {
        name: "Blue".into(),
        base_color: kreate_math::SceneRgb::new(0.1, 0.1, 0.8),
        ..Material::default()
    })
}

#[test]
fn a_fresh_mesh_has_no_material_and_that_is_a_state() {
    // Every cube anybody adds. Answering `None` rather than inventing a
    // default here keeps the decision in one place: the renderer draws an
    // unassigned surface with its own default, and the document does not
    // pretend somebody chose one.
    let mut doc = Document::new();
    let (mesh, object) = cube(&mut doc);
    assert!(doc.mesh_slots(mesh).is_empty());
    assert!(doc.material_at(object, 0).is_none());
}

#[test]
fn a_material_gets_an_identity_like_every_other_datablock() {
    // It had none until now, which meant a curve could not target one and a
    // file could not name one.
    let mut doc = Document::new();
    let id = red(&mut doc);
    let uid = doc.material_uid(id).expect("an identity");
    assert!(!uid.is_none());
    assert_eq!(doc.material_by_uid(uid), Some(id));
}

#[test]
fn materials_count_separately_from_every_other_kind() {
    // One counter per kind, so the numbers stay small and a person reading a
    // file can match a curve to its target by eye.
    let mut doc = Document::new();
    let material = red(&mut doc);
    let light = doc.add_light(kreate_scene::Light::new("Lamp"));
    assert_eq!(doc.material_uid(material), doc.light_uid(light));
}

#[test]
fn a_face_index_names_a_slot_and_the_slot_names_the_material() {
    // The indirection this commit exists for. A face carries a small integer,
    // the mesh carries the list, and the document carries the material.
    let mut doc = Document::new();
    let (mesh, object) = cube(&mut doc);
    let red = red(&mut doc);
    let blue = blue(&mut doc);

    assert!(doc.set_mesh_slots(mesh, vec![Some(red), Some(blue)]));
    assert_eq!(doc.material_at(object, 0).map(|m| m.name.as_str()), Some("Red"));
    assert_eq!(
        doc.material_at(object, 1).map(|m| m.name.as_str()),
        Some("Blue")
    );
}

#[test]
fn a_slot_past_the_end_is_nothing_rather_than_a_panic() {
    // Reachable from an imported mesh whose faces point past what the importer
    // filled in, and from a mesh whose slots were shortened. Nothing is the
    // same answer an unassigned slot gives, and the renderer already handles
    // that.
    let mut doc = Document::new();
    let (mesh, object) = cube(&mut doc);
    let red = red(&mut doc);
    doc.set_mesh_slots(mesh, vec![Some(red)]);

    assert!(doc.material_at(object, 1).is_none());
    assert!(doc.material_at(object, 99).is_none());
}

#[test]
fn assigning_a_high_slot_grows_the_list_rather_than_refusing() {
    // An imported mesh with a face pointing at slot three is exactly the case
    // this has to cope with. Refusing would lose the assignment rather than
    // the index, which is the wrong half to drop.
    let mut doc = Document::new();
    let (mesh, object) = cube(&mut doc);
    let blue = blue(&mut doc);

    assert!(doc.set_mesh_slot(mesh, 3, Some(blue)));
    assert_eq!(doc.mesh_slots(mesh).len(), 4);
    assert!(doc.material_at(object, 0).is_none());
    assert_eq!(
        doc.material_at(object, 3).map(|m| m.name.as_str()),
        Some("Blue")
    );
}

#[test]
fn two_linked_copies_share_their_mesh_and_so_share_its_materials() {
    // §1.4's whole argument, at the material. Assigning to one copy assigns to
    // both, which is what an artist means by "the material of this chair" —
    // and it is why the slots are on the mesh.
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let left = doc.add_object(Object::new("Left", ObjectData::Mesh(mesh)));
    let right = doc.add_object(Object::new("Right", ObjectData::Mesh(mesh)));
    let red = red(&mut doc);

    doc.set_mesh_slot(mesh, 0, Some(red));
    assert_eq!(doc.material_at(left, 0).map(|m| m.name.as_str()), Some("Red"));
    assert_eq!(
        doc.material_at(right, 0).map(|m| m.name.as_str()),
        Some("Red")
    );
}

#[test]
fn an_object_can_override_one_slot_and_leave_the_rest() {
    // The case the override exists for: two linked copies of a chair, one of
    // them upholstered differently. The override is per slot, so the other
    // slots still come from the mesh.
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let left = doc.add_object(Object::new("Left", ObjectData::Mesh(mesh)));
    let right = doc.add_object(Object::new("Right", ObjectData::Mesh(mesh)));
    let red = red(&mut doc);
    let blue = blue(&mut doc);
    doc.set_mesh_slots(mesh, vec![Some(red), Some(red)]);

    doc.object_mut(right).expect("right").material_overrides = vec![Some(blue)];

    assert_eq!(doc.material_at(left, 0).map(|m| m.name.as_str()), Some("Red"));
    assert_eq!(
        doc.material_at(right, 0).map(|m| m.name.as_str()),
        Some("Blue")
    );
    // The slot the object says nothing about still comes from the mesh.
    assert_eq!(
        doc.material_at(right, 1).map(|m| m.name.as_str()),
        Some("Red")
    );
}

#[test]
fn an_override_of_none_falls_through_to_the_mesh() {
    // `Some(None)` and a missing entry mean the same thing, which they have to:
    // the list is positional, so overriding slot one requires an entry for slot
    // zero, and that entry must not mean "this slot has no material".
    //
    // An object that wanted to *remove* a material would clear the mesh's slot
    // or be given a different mesh. Making a hole in the override list mean
    // "nothing" would leave no way to say "as the mesh says", which is the
    // answer needed far more often.
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let object = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    let red = red(&mut doc);
    let blue = blue(&mut doc);
    doc.set_mesh_slots(mesh, vec![Some(red), Some(red)]);

    doc.object_mut(object).expect("object").material_overrides = vec![None, Some(blue)];
    assert_eq!(
        doc.material_at(object, 0).map(|m| m.name.as_str()),
        Some("Red"),
        "a hole in the override list should fall through"
    );
    assert_eq!(
        doc.material_at(object, 1).map(|m| m.name.as_str()),
        Some("Blue")
    );
}

#[test]
fn an_object_with_no_mesh_has_no_material() {
    // An empty, a camera, a light. Not an error: asking is reasonable and the
    // answer is nothing.
    let mut doc = Document::new();
    let empty = doc.add_object(Object::new("Empty", ObjectData::Empty));
    assert!(doc.material_at(empty, 0).is_none());

    // But an override still answers, because an override is the object's and
    // does not go through the mesh. That is not a special case — it falls out
    // of the rule, and pinning it means the rule cannot be reordered without
    // somebody noticing.
    let red = red(&mut doc);
    doc.object_mut(empty).expect("empty").material_overrides = vec![Some(red)];
    assert_eq!(
        doc.material_at(empty, 0).map(|m| m.name.as_str()),
        Some("Red")
    );
}

#[test]
fn the_base_colour_is_a_reflectance_and_says_so() {
    // §2.1's discipline, applied to the field it was missing from. This was
    // `[f32; 4]`, which says nothing about the space it is in — and both of
    // the colour bugs that discipline was introduced after were a value in the
    // wrong one.
    let material = Material::default();
    assert!(material.base_color.is_display_representable());
    assert_eq!(material.alpha, 1.0);
    // Opacity travels beside the colour rather than in it: a transfer function
    // applies to three channels and never to the fourth.
    let glass = Material {
        base_color: kreate_math::SceneRgb::new(0.9, 0.95, 1.0),
        alpha: 0.1,
        ..Material::default()
    };
    assert_eq!(glass.base_color.to_array().len(), 3);
    assert_eq!(glass.alpha, 0.1);
}

#[test]
fn editing_a_material_edits_it_for_every_object_using_it() {
    // The datablock's whole point, and the thing a per-object copy would lose.
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let a = doc.add_object(Object::new("A", ObjectData::Mesh(mesh)));
    let b = doc.add_object(Object::new("B", ObjectData::Mesh(mesh)));
    let red = red(&mut doc);
    doc.set_mesh_slot(mesh, 0, Some(red));

    doc.material_mut(red).expect("red").roughness = 0.12;
    assert_eq!(doc.material_at(a, 0).map(|m| m.roughness), Some(0.12));
    assert_eq!(doc.material_at(b, 0).map(|m| m.roughness), Some(0.12));
    assert_eq!(doc.material_count(), 1, "the material was copied");
}

#[test]
fn slots_belong_to_the_mesh_and_not_to_the_document() {
    // Which is what makes a `MeshData` transferable: the face indices mean
    // something relative to the mesh they came with, and nothing relative to
    // an arena in some other document.
    let mut doc = Document::new();
    let one = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let two = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let red = red(&mut doc);
    let blue = blue(&mut doc);
    doc.set_mesh_slot(one, 0, Some(red));
    doc.set_mesh_slot(two, 0, Some(blue));

    let first = doc.add_object(Object::new("One", ObjectData::Mesh(one)));
    let second = doc.add_object(Object::new("Two", ObjectData::Mesh(two)));
    assert_eq!(
        doc.material_at(first, 0).map(|m| m.name.as_str()),
        Some("Red")
    );
    assert_eq!(
        doc.material_at(second, 0).map(|m| m.name.as_str()),
        Some("Blue")
    );
}

#[test]
fn setting_slots_on_a_mesh_that_is_gone_is_refused_rather_than_ignored() {
    // A caller holding a stale id should learn about it. Silently doing
    // nothing is how a material assignment disappears with no error.
    let mut doc = Document::new();
    let stranger = Document::new().add_mesh(kreate_mesh::primitives::cube(1.0));
    assert!(!doc.set_mesh_slots(stranger, vec![None]));
    assert!(!doc.set_mesh_slot(stranger, 0, None));
}

// --- the base colour texture ----------------------------------------------
//
// The application opens the file and adds the datablock; the operator makes the
// *link*. Everything below is about the link, because that is the part an undo
// step holds and the part a script can get wrong.

use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};

fn wired() -> (OperatorRegistry, History, Document, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (registry, History::new(), Document::new(), Context::default())
}

/// A cube whose first slot holds a material, selected and active.
fn a_cube_with_a_material(doc: &mut Document) -> kreate_scene::MaterialId {
    let (mesh, object) = cube(doc);
    let material = red(doc);
    doc.set_mesh_slot(mesh, 0, Some(material));
    doc.selection.add(object);
    material
}

fn an_image(doc: &mut Document) -> kreate_scene::document::ImageId {
    doc.add_image(kreate_scene::Image::new(
        "Bricks",
        4,
        4,
        kreate_scene::PixelFormat::Rgba8,
    ))
}

fn set_image(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    ctx: &Context,
    uid: &str,
) {
    let mut params = PropertyMap::new();
    params.set("uid", PropertyValue::Text(uid.to_string()));
    history
        .dispatch(registry, doc, ctx, "material.set_image", &params)
        .expect("the operator ran");
}

#[test]
fn a_texture_arrives_and_can_be_taken_off_again() {
    let (registry, mut history, mut doc, ctx) = wired();
    let material = a_cube_with_a_material(&mut doc);
    let image = an_image(&mut doc);
    let uid = doc.image_uid(image).expect("an identity").raw().to_string();

    set_image(&registry, &mut history, &mut doc, &ctx, &uid);
    assert_eq!(doc.material(material).unwrap().base_color_image, Some(image));

    // Empty rather than a second operator, the same way a world goes back to a
    // colour: one thing to undo and one place the rule lives.
    set_image(&registry, &mut history, &mut doc, &ctx, "");
    assert_eq!(doc.material(material).unwrap().base_color_image, None);
}

#[test]
fn undoing_gives_the_material_back_the_texture_it_had() {
    let (registry, mut history, mut doc, ctx) = wired();
    let material = a_cube_with_a_material(&mut doc);
    let image = an_image(&mut doc);
    let uid = doc.image_uid(image).expect("an identity").raw().to_string();

    set_image(&registry, &mut history, &mut doc, &ctx, &uid);
    set_image(&registry, &mut history, &mut doc, &ctx, "");
    assert!(history.undo(&mut doc), "there was nothing to undo");
    assert_eq!(
        doc.material(material).unwrap().base_color_image,
        Some(image),
        "undo did not give the texture back"
    );
    // And the datablock is still there either way — undoing a link must not
    // delete an image another material may have started using.
    assert!(doc.image(image).is_some());
}

#[test]
fn a_uid_naming_an_image_that_is_not_here_is_refused() {
    // A parameter from a script or a stale redo. A material pointing at nothing
    // is worse than one that declined to change: the first draws a hole and the
    // second draws what it drew before.
    let (registry, mut history, mut doc, ctx) = wired();
    let material = a_cube_with_a_material(&mut doc);
    let image = an_image(&mut doc);
    let uid = doc.image_uid(image).expect("an identity").raw().to_string();
    set_image(&registry, &mut history, &mut doc, &ctx, &uid);

    let depth = history.entries().len();
    set_image(&registry, &mut history, &mut doc, &ctx, "999999");
    set_image(&registry, &mut history, &mut doc, &ctx, "not a number");
    assert_eq!(
        doc.material(material).unwrap().base_color_image,
        Some(image),
        "a refused uid changed the material"
    );
    assert_eq!(
        history.entries().len(),
        depth,
        "a refused uid left a step to undo"
    );
}

#[test]
fn the_parameters_that_read_a_material_back_leave_it_alone() {
    // The panel writes what it read. If the two disagree, opening the tab
    // changes the document — which is how a value drifts with nobody touching
    // it.
    let (registry, mut history, mut doc, ctx) = wired();
    let material = a_cube_with_a_material(&mut doc);
    let image = an_image(&mut doc);
    let uid = doc.image_uid(image).expect("an identity").raw().to_string();
    set_image(&registry, &mut history, &mut doc, &ctx, &uid);

    let read = kreate_ops::ops::object::material_image_params(
        &doc,
        doc.material(material).expect("a material"),
    );
    assert_eq!(read.uid, uid);
    set_image(&registry, &mut history, &mut doc, &ctx, &read.uid);
    assert_eq!(doc.material(material).unwrap().base_color_image, Some(image));
}
