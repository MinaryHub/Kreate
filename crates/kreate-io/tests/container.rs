//! The `.kreate` container's guarantees.
//!
//! Three of these cannot be added after files exist, because a file written
//! before the rule was in place does not obey it:
//!
//! - **Binary entries are STORED.** A deflated entry cannot be memory mapped,
//!   and mapping is the whole reason `.kmesh` is laid out the way it is. A file
//!   saved with its meshes compressed stays slow to open forever.
//! - **The bytes are deterministic**, so version control stores one blob for a
//!   document that has not changed.
//! - **Unknown entries and unknown JSON fields survive.** An archive from a
//!   newer build, opened and saved here, must keep what this build could not
//!   read — otherwise one save in an older version is silent data loss.

use kreate_io::container::{self, Manifest, UnknownEntry};
use kreate_scene::document::{Document, Object, ObjectData};
use std::io::Cursor;

/// A document with a hierarchy, two meshes and a linked duplicate.
///
/// Not one cube: a round trip over a single object would pass with parenting,
/// mesh sharing and transforms all broken, and those are the parts a file format
/// most easily gets wrong.
fn sample() -> Document {
    let mut doc = Document::new();
    let cube = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
    let sphere = doc.add_mesh(kreate_mesh::primitives::uv_sphere(1.0, 16, 8));

    let parent = doc.add_object(Object::new("parent", ObjectData::Mesh(cube)));
    let child = doc.add_object(Object::new("child", ObjectData::Mesh(sphere)));
    // A linked duplicate: a second object sharing the first mesh datablock. If
    // the writer wrote geometry per object instead of per datablock, this is
    // what would catch it.
    let twin = doc.add_object(Object::new("twin", ObjectData::Mesh(cube)));
    let empty = doc.add_object(Object::new("pivot", ObjectData::Empty));

    doc.set_parent(child, Some(parent)).expect("parent");
    doc.set_parent(twin, Some(empty)).expect("parent");

    if let Some(o) = doc.object_mut(parent) {
        o.transform.translation = glam::DVec3::new(1.0, 2.0, 3.0);
        o.transform.scale = glam::DVec3::new(2.0, 2.0, 2.0);
    }
    if let Some(o) = doc.object_mut(child) {
        o.transform.rotation = glam::DQuat::from_rotation_z(0.5);
        o.visible = false;
    }
    if let Some(o) = doc.object_mut(empty) {
        o.selectable = false;
    }
    doc
}

fn round_trip(doc: &Document) -> container::Loaded {
    let bytes = container::write(doc, &Manifest::default(), &[]).expect("writes");
    container::read(Cursor::new(bytes)).expect("reads")
}

#[test]
fn a_document_survives_a_round_trip() {
    let doc = sample();
    let loaded = round_trip(&doc);
    let after = &loaded.document;

    assert_eq!(
        after.objects().count(),
        doc.objects().count(),
        "object count"
    );
    assert_eq!(after.meshes().count(), doc.meshes().count(), "mesh count");

    let before: Vec<_> = doc.objects().map(|(_, o)| o.name.clone()).collect();
    let names: Vec<_> = after.objects().map(|(_, o)| o.name.clone()).collect();
    assert_eq!(before, names, "object names and order");

    for ((_, a), (_, b)) in doc.objects().zip(after.objects()) {
        assert_eq!(
            a.transform.translation, b.transform.translation,
            "{}",
            a.name
        );
        assert_eq!(a.transform.scale, b.transform.scale, "{}", a.name);
        assert!(
            a.transform
                .rotation
                .abs_diff_eq(b.transform.rotation, 1e-12),
            "{} rotation",
            a.name
        );
        assert_eq!(a.visible, b.visible, "{} visibility", a.name);
        assert_eq!(a.selectable, b.selectable, "{} selectable", a.name);
    }
}

#[test]
fn the_hierarchy_survives_a_round_trip() {
    // Parenting is stored as an index into the object list, not as an arena id:
    // an id carries a generation and means nothing outside the process that made
    // it. This is the test that says the translation works both ways.
    let doc = sample();
    let loaded = round_trip(&doc);

    let parent_names = |d: &Document| -> Vec<(String, Option<String>)> {
        let name_of: std::collections::BTreeMap<_, _> =
            d.objects().map(|(id, o)| (id, o.name.clone())).collect();
        d.objects()
            .map(|(_, o)| {
                (
                    o.name.clone(),
                    o.parent.and_then(|p| name_of.get(&p).cloned()),
                )
            })
            .collect()
    };
    assert_eq!(parent_names(&doc), parent_names(&loaded.document));
    assert_eq!(
        doc.roots().len(),
        loaded.document.roots().len(),
        "root count"
    );
}

#[test]
fn a_linked_duplicate_shares_one_mesh_entry_rather_than_two_copies() {
    // Two objects reference one datablock. If the writer emitted geometry per
    // object, the archive would hold three meshes for two — and the file would
    // grow with every duplicate the user made, which is exactly what
    // referencing datablocks by id exists to prevent.
    let doc = sample();
    assert_eq!(doc.meshes().count(), 2);
    let loaded = round_trip(&doc);
    assert_eq!(
        loaded.document.meshes().count(),
        2,
        "the duplicate was written as a second copy of the mesh"
    );

    // And the two objects still point at the same datablock afterwards.
    let mesh_of = |name: &str| -> Option<kreate_scene::MeshId> {
        loaded
            .document
            .objects()
            .find(|(_, o)| o.name == name)
            .and_then(|(_, o)| match o.data {
                ObjectData::Mesh(m) => Some(m),
                _ => None,
            })
    };
    assert_eq!(mesh_of("parent"), mesh_of("twin"), "the link was broken");
}

#[test]
fn mesh_entries_are_stored_uncompressed_so_they_can_be_mapped() {
    // The decision the whole `.kmesh` layout depends on. A deflated entry has to
    // be decompressed into a buffer before anything can look at it, which throws
    // away the one property that layout was designed for.
    let bytes = container::write(&sample(), &Manifest::default(), &[]).expect("writes");
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).expect("opens");

    let mut checked = 0;
    for i in 0..zip.len() {
        let entry = zip.by_index(i).expect("entry");
        let name = entry.name().to_string();
        if name.ends_with(".kmesh") {
            assert_eq!(
                entry.compression(),
                zip::CompressionMethod::Stored,
                "{name} is compressed, so it cannot be memory mapped"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "the archive had no meshes to check");
}

#[test]
fn saving_the_same_document_twice_produces_the_same_bytes() {
    // What lets version control store one blob for an unchanged file. The trap
    // is the zip's per-entry modification time, which comes from the clock
    // unless it is set — two saves a second apart would otherwise differ in
    // every entry.
    let doc = sample();
    let a = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    let b = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    assert_eq!(a, b, "two saves of one document differ");
}

#[test]
fn every_entry_carries_a_fixed_timestamp() {
    // Checked directly rather than inferred from two saves matching. `zip`
    // currently defaults to a fixed date only because it is built without its
    // `time` feature — with that feature on, the default is the clock, and a
    // falsification pass that swapped the explicit epoch for the crate default
    // changed nothing precisely because the feature happens to be off today.
    //
    // Feature unification can turn it on from anywhere in the dependency graph,
    // and the symptom would be every file differing from itself on every save.
    let bytes = container::write(&sample(), &Manifest::default(), &[]).expect("writes");
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).expect("opens");
    let epoch = zip::DateTime::default();
    for i in 0..zip.len() {
        let entry = zip.by_index(i).expect("entry");
        let name = entry.name().to_string();
        assert_eq!(
            entry.last_modified(),
            Some(epoch),
            "{name} carries a timestamp that is not the fixed epoch"
        );
    }
}

#[test]
fn saving_a_loaded_document_reproduces_the_file() {
    // The stronger statement: a load followed by a save is the identity. Without
    // it, opening a file and closing it without touching anything would still
    // show up as a change in version control.
    let doc = sample();
    let first = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    let loaded = container::read(Cursor::new(first.clone())).expect("reads");
    let second =
        container::write(&loaded.document, &loaded.manifest, &loaded.unknown).expect("writes");
    assert_eq!(first, second, "a load-and-save changed the file");
}

#[test]
fn an_entry_from_a_newer_build_survives_being_saved_by_this_one() {
    // The forward-compatibility rule. An archive holding something this build
    // has no idea about — a texture set, an animation track, a node graph — must
    // come back out untouched, or opening a file in an older version quietly
    // destroys it.
    let future = UnknownEntry {
        name: "animation/tracks.bin".to_string(),
        bytes: b"whatever a later version stores here".to_vec(),
    };
    let bytes = container::write(
        &sample(),
        &Manifest::default(),
        std::slice::from_ref(&future),
    )
    .expect("writes");
    let loaded = container::read(Cursor::new(bytes)).expect("reads");
    assert_eq!(loaded.unknown, vec![future.clone()], "the entry was lost");

    let again =
        container::write(&loaded.document, &loaded.manifest, &loaded.unknown).expect("writes");
    let reloaded = container::read(Cursor::new(again)).expect("reads");
    assert_eq!(reloaded.unknown, vec![future]);
}

#[test]
fn an_unknown_json_field_survives_a_round_trip() {
    // The same rule one level down. A newer build may add a field to the
    // manifest; this one keeps it rather than dropping it on the next save.
    let mut manifest = Manifest::default();
    manifest.extra.insert(
        "render_settings".to_string(),
        serde_json::json!({ "samples": 64, "denoise": true }),
    );

    let bytes = container::write(&sample(), &manifest, &[]).expect("writes");
    let loaded = container::read(Cursor::new(bytes)).expect("reads");
    assert_eq!(
        loaded.manifest.extra.get("render_settings"),
        manifest.extra.get("render_settings"),
        "the unknown field was dropped"
    );
}

#[test]
fn a_document_that_demands_a_newer_reader_is_refused() {
    // The file's own statement about who may read it. A reader that ignored it
    // would not fail — it would succeed and produce something wrong.
    let manifest = Manifest {
        min_reader_major: 99,
        ..Manifest::default()
    };
    let bytes = container::write(&sample(), &manifest, &[]).expect("writes");
    match container::read(Cursor::new(bytes)) {
        Err(container::ContainerError::TooNew { needed }) => assert_eq!(needed, 99),
        other => panic!("expected a refusal, got {:?}", other.err()),
    }
}

#[test]
fn a_newer_minor_version_is_still_readable() {
    // Minor bumps are additive. A reader that refused them would turn every
    // additive change into a compatibility break.
    let manifest = Manifest {
        version_minor: 7,
        ..Manifest::default()
    };
    let bytes = container::write(&sample(), &manifest, &[]).expect("writes");
    let loaded = container::read(Cursor::new(bytes)).expect("a newer minor version must open");
    assert_eq!(loaded.manifest.version_minor, 7);
}

#[test]
fn something_that_is_not_a_kreate_document_is_refused() {
    // A plain zip, and a file that is not a zip at all. Both arrive when a user
    // drags the wrong thing onto the window.
    let mut plain = Vec::new();
    {
        let mut zip = zip::ZipWriter::new(Cursor::new(&mut plain));
        zip.start_file("readme.txt", zip::write::SimpleFileOptions::default())
            .unwrap();
        use std::io::Write;
        zip.write_all(b"not a document").unwrap();
        zip.finish().unwrap();
    }
    assert!(container::read(Cursor::new(plain)).is_err());
    assert!(container::read(Cursor::new(b"not a zip at all".to_vec())).is_err());
}

#[test]
fn an_empty_document_round_trips() {
    let doc = Document::new();
    let loaded = round_trip(&doc);
    assert_eq!(loaded.document.objects().count(), 0);
    assert_eq!(loaded.document.meshes().count(), 0);
}

#[test]
fn a_truncated_archive_is_an_error_and_never_a_panic() {
    // Interrupted saves and partial downloads. The zip reader does most of the
    // work here, but a prefix that happens to contain a readable manifest and a
    // damaged mesh has to come back as an error rather than as a panic in the
    // cast-based mesh reader.
    let bytes = container::write(&sample(), &Manifest::default(), &[]).expect("writes");
    for len in (0..bytes.len()).step_by(7) {
        let _ = container::read(Cursor::new(bytes[..len].to_vec()));
    }
    assert!(container::read(Cursor::new(bytes)).is_ok());
}
