//! Materials, their slots, and their overrides, through a save and a load.
//!
//! Three links have to survive, and they are three different kinds of thing:
//!
//! - The **material** itself, with its identity, so a curve or a driver
//!   pointing at it still is.
//! - The **mesh's slot list**, which is what a face's `material_index` numbers.
//!   Stored beside the meshes rather than inside them for the same reason
//!   their identities are: a slot list belongs to a mesh, and a mesh is not
//!   owned by an object.
//! - The **object's overrides**, which are positional and where a hole means
//!   "as the mesh says" rather than "no material".

use kreate_io::container::{self, Manifest};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{Document, Material, MeshId};

fn round_trip(doc: &Document) -> Document {
    let bytes = container::write(doc, &Manifest::default(), &[]).expect("writes");
    container::read(std::io::Cursor::new(bytes))
        .expect("opens")
        .document
}

fn named(doc: &mut Document, name: &str, r: f32, g: f32, b: f32) -> kreate_scene::MaterialId {
    doc.add_material(Material {
        name: name.into(),
        base_color: kreate_math::SceneRgb::new(r, g, b),
        ..Material::default()
    })
}

fn cube(doc: &mut Document, name: &str) -> (MeshId, kreate_scene::ObjectId) {
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let object = doc.add_object(Object::new(name, ObjectData::Mesh(mesh)));
    (mesh, object)
}

#[test]
fn every_field_survives_with_nothing_at_its_default() {
    // Deliberately nothing at its default, so a writer that dropped a field
    // and a reader that filled it in from the default would not cancel out.
    // The camera file states this rule; the light file learned it from a
    // falsification sweep.
    let mut doc = Document::new();
    let id = doc.add_material(Material {
        name: "Brushed brass".into(),
        base_color: kreate_math::SceneRgb::new(0.71, 0.53, 0.19),
        alpha: 0.37,
        metallic: 0.83,
        roughness: 0.29,
        ..Material::default()
    });
    let (mesh, _) = cube(&mut doc, "Cube");
    doc.set_mesh_slot(mesh, 0, Some(id));

    let back = round_trip(&doc);
    let (_, got) = back.materials().next().expect("a material");
    assert_eq!(got.name, "Brushed brass");
    assert_eq!(got.base_color.to_array(), [0.71, 0.53, 0.19]);
    assert_eq!(got.alpha, 0.37);
    assert_eq!(got.metallic, 0.83);
    assert_eq!(got.roughness, 0.29);
}

#[test]
fn the_base_colour_crosses_linear_and_unchanged() {
    // Linear on both sides and in glTF too, so this is a copy. Encoding here
    // and decoding there would put a transfer function in the middle of a
    // round trip, which is the shape of both colour bugs §2.1 records.
    let mut doc = Document::new();
    let id = named(&mut doc, "Mid", 0.5, 0.25, 0.125);
    let (mesh, _) = cube(&mut doc, "Cube");
    doc.set_mesh_slot(mesh, 0, Some(id));

    let back = round_trip(&doc);
    let (_, got) = back.materials().next().expect("a material");
    assert_eq!(got.base_color.to_array(), [0.5, 0.25, 0.125]);
}

#[test]
fn a_materials_identity_survives_a_save() {
    let mut doc = Document::new();
    let id = named(&mut doc, "Red", 0.8, 0.1, 0.1);
    let (mesh, _) = cube(&mut doc, "Cube");
    doc.set_mesh_slot(mesh, 0, Some(id));
    let before = doc.material_uid(id).expect("an identity");
    assert!(!before.is_none(), "the writer had nothing to write");

    let back = round_trip(&doc);
    let (_, got) = back.materials().next().expect("a material");
    assert!(!got.uid.is_none(), "the loader issued nothing");
    assert_eq!(got.uid, before);
    assert!(back.material_by_uid(got.uid).is_some());
}

#[test]
fn identities_with_gaps_in_them_come_back_as_they_were() {
    // A re-issuing loader passes a test that saves two and gets two back,
    // because the numbers come out in the same order. It fails this one.
    let mut doc = Document::new();
    let first = doc.restore_material(Material::default(), kreate_scene::Uid::from_raw(2));
    let last = doc.restore_material(Material::default(), kreate_scene::Uid::from_raw(9));
    let (mesh, _) = cube(&mut doc, "Cube");
    doc.set_mesh_slots(mesh, vec![Some(first), Some(last)]);

    let back = round_trip(&doc);
    let mut uids: Vec<u64> = back.materials().map(|(_, m)| m.uid.raw()).collect();
    uids.sort_unstable();
    assert_eq!(uids, vec![2, 9]);

    // And a new material cannot collide with one the file skipped.
    let mut back = back;
    let fresh = back.add_material(Material::default());
    assert!(back.material_uid(fresh).unwrap().raw() > 9);
}

#[test]
fn a_meshs_slots_come_back_pointing_at_the_same_materials() {
    // The indirection, end to end: a face's index names a slot, the slot names
    // a material, and both halves have to cross.
    let mut doc = Document::new();
    let red = named(&mut doc, "Red", 0.8, 0.1, 0.1);
    let blue = named(&mut doc, "Blue", 0.1, 0.1, 0.8);
    let (mesh, _) = cube(&mut doc, "Cube");
    doc.set_mesh_slots(mesh, vec![Some(red), None, Some(blue)]);

    let back = round_trip(&doc);
    let (id, _) = back
        .objects()
        .find(|(_, o)| o.name == "Cube")
        .expect("the cube");
    assert_eq!(
        back.material_at(id, 0).map(|m| m.name.as_str()),
        Some("Red")
    );
    // A hole in the slot list is a slot nothing is assigned to, and it has to
    // stay a hole rather than closing up — the indices after it are face data.
    assert!(back.material_at(id, 1).is_none());
    assert_eq!(
        back.material_at(id, 2).map(|m| m.name.as_str()),
        Some("Blue")
    );
}

#[test]
fn two_meshes_keep_their_own_slot_lists() {
    // Slots belong to a mesh. A writer keyed by anything else would give both
    // meshes the same materials, and the scene would look plausible.
    let mut doc = Document::new();
    let red = named(&mut doc, "Red", 0.8, 0.1, 0.1);
    let blue = named(&mut doc, "Blue", 0.1, 0.1, 0.8);
    let (one, first) = cube(&mut doc, "One");
    let (two, second) = cube(&mut doc, "Two");
    doc.set_mesh_slot(one, 0, Some(red));
    doc.set_mesh_slot(two, 0, Some(blue));
    let (first_name, second_name) = (
        doc.objects()
            .find(|(id, _)| *id == first)
            .map(|(_, o)| o.name.clone())
            .unwrap(),
        doc.objects()
            .find(|(id, _)| *id == second)
            .map(|(_, o)| o.name.clone())
            .unwrap(),
    );

    let back = round_trip(&doc);
    for (id, object) in back.objects() {
        let want = if object.name == first_name {
            "Red"
        } else if object.name == second_name {
            "Blue"
        } else {
            continue;
        };
        assert_eq!(
            back.material_at(id, 0).map(|m| m.name.as_str()),
            Some(want),
            "{} got the wrong material",
            object.name
        );
    }
}

#[test]
fn two_linked_copies_still_share_one_mesh_and_one_slot_list() {
    // §1.4's argument has to survive the file too. A writer that saved the
    // mesh twice would give the pair two slot lists, and editing one would
    // stop editing both.
    let mut doc = Document::new();
    let red = named(&mut doc, "Red", 0.8, 0.1, 0.1);
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    doc.set_mesh_slot(mesh, 0, Some(red));
    doc.add_object(Object::new("Left", ObjectData::Mesh(mesh)));
    doc.add_object(Object::new("Right", ObjectData::Mesh(mesh)));

    let back = round_trip(&doc);
    let meshes: Vec<MeshId> = back.objects().filter_map(|(_, o)| o.mesh()).collect();
    assert_eq!(meshes.len(), 2);
    assert_eq!(meshes[0], meshes[1], "the mesh was saved twice");
    assert_eq!(back.material_count(), 1);
}

#[test]
fn an_objects_override_survives_and_a_hole_in_it_stays_a_hole() {
    // The positional rule: `null` means "as the mesh says". A writer that
    // dropped the holes would shift every later override by one, which is a
    // scene where the wrong chair is upholstered.
    let mut doc = Document::new();
    let red = named(&mut doc, "Red", 0.8, 0.1, 0.1);
    let blue = named(&mut doc, "Blue", 0.1, 0.1, 0.8);
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    doc.set_mesh_slots(mesh, vec![Some(red), Some(red)]);
    doc.add_object(Object::new("Plain", ObjectData::Mesh(mesh)));
    let odd = doc.add_object(Object::new("Odd", ObjectData::Mesh(mesh)));
    doc.object_mut(odd).expect("odd").material_overrides = vec![None, Some(blue)];

    let back = round_trip(&doc);
    for (id, object) in back.objects() {
        match object.name.as_str() {
            "Plain" => {
                assert!(object.material_overrides.is_empty());
                assert_eq!(
                    back.material_at(id, 1).map(|m| m.name.as_str()),
                    Some("Red")
                );
            }
            "Odd" => {
                assert_eq!(object.material_overrides.len(), 2);
                assert_eq!(
                    back.material_at(id, 0).map(|m| m.name.as_str()),
                    Some("Red"),
                    "the hole closed up"
                );
                assert_eq!(
                    back.material_at(id, 1).map(|m| m.name.as_str()),
                    Some("Blue")
                );
            }
            other => panic!("unexpected object {other}"),
        }
    }
}

#[test]
fn a_scene_with_no_materials_writes_none_and_reads_none() {
    // Every scene anybody starts. The record lists are skipped when empty, so
    // a file from a scene with no materials is byte-identical to one written
    // before materials existed — which is what keeps the format's minor
    // version honest.
    let mut doc = Document::new();
    cube(&mut doc, "Cube");
    let back = round_trip(&doc);
    assert_eq!(back.material_count(), 0);
    let (id, _) = back.objects().next().expect("the cube");
    assert!(back.material_at(id, 0).is_none());
}

#[test]
fn a_slot_naming_a_material_that_is_not_in_the_file_is_left_empty() {
    // Reachable from a hand-edited file or a later version. An empty slot is
    // what an unassigned one is, and the renderer already draws it.
    let mut doc = Document::new();
    let red = named(&mut doc, "Red", 0.8, 0.1, 0.1);
    let (mesh, _) = cube(&mut doc, "Cube");
    doc.set_mesh_slot(mesh, 0, Some(red));
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");

    let tampered = retarget_first_slot(&bytes, 7);
    let back = container::read(std::io::Cursor::new(tampered))
        .expect("a bad index must not fail the load")
        .document;
    let (id, _) = back.objects().next().expect("the cube");
    assert!(back.material_at(id, 0).is_none());
    // And the material itself is still there, so nothing was lost but the
    // link.
    assert_eq!(back.material_count(), 1);
}

// --- reaching into the archive --------------------------------------------

fn scene_text(archive: &[u8]) -> String {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive.to_vec())).expect("archive");
    let mut file = zip.by_name("scene.json").expect("scene.json");
    let mut text = String::new();
    std::io::Read::read_to_string(&mut file, &mut text).expect("utf-8");
    text
}

fn retarget_first_slot(archive: &[u8], to: u64) -> Vec<u8> {
    let mut scene: serde_json::Value = serde_json::from_str(&scene_text(archive)).expect("json");
    let slots = scene["mesh_slots"]
        .as_object_mut()
        .expect("mesh_slots")
        .values_mut()
        .next()
        .expect("one mesh");
    slots[0] = serde_json::Value::from(to);
    rewrite_scene(archive, &serde_json::to_string(&scene).expect("json"))
}

fn rewrite_scene(archive: &[u8], scene: &str) -> Vec<u8> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive.to_vec())).expect("archive");
    let mut out = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).expect("entry");
        let name = entry.name().expect("a named entry").into_owned();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut bytes).expect("read");
        out.start_file(&name, options).expect("start");
        let body = if name == "scene.json" {
            scene.as_bytes()
        } else {
            &bytes
        };
        std::io::Write::write_all(&mut out, body).expect("write");
    }
    out.finish().expect("finish").into_inner()
}
