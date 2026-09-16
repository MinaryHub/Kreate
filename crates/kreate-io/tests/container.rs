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

// ---------------------------------------------------------------------------
// Images.
//
// `images/` was the entry the archive did not have, and three things were
// waiting behind it: an environment for §7.8's IBL, Phase 8's textures, and the
// world tab's background control. A world has been able to *name* its image by
// identity since minor 2 and that name has never resolved, because there was
// nothing in the file to resolve it to.

/// An image with a recognisable pixel, and a hole where a tile is not resident.
///
/// The hole is the point: an image that is fully resident passes whether the
/// residency survives or not, and residency is what the tiling exists for.
fn patchy_image() -> kreate_scene::Image {
    let mut image = kreate_scene::Image::new(
        "Sky",
        400,
        300,
        kreate_scene::PixelFormat::Rgba8,
    );
    image.color_space = kreate_scene::ColorSpace::Raw;
    let tile = image.tile_mut(0, 0).expect("in range");
    tile.bytes_mut()[0..4].copy_from_slice(&[9, 8, 7, 255]);
    let tile = image.tile_mut(2, 1).expect("in range");
    tile.bytes_mut()[0..4].copy_from_slice(&[1, 2, 3, 255]);
    image
}

#[test]
fn an_image_survives_a_round_trip_with_its_holes() {
    let mut doc = Document::new();
    let id = doc.add_image(patchy_image());
    let before = doc.image(id).expect("the image").clone();

    let loaded = round_trip(&doc);
    let (_, after) = loaded
        .document
        .images()
        .next()
        .expect("the image came back");
    assert_eq!(after.name, "Sky");
    assert_eq!(after.width(), 400);
    assert_eq!(after.height(), 300);
    assert_eq!(after.format, kreate_scene::PixelFormat::Rgba8);
    // The space is declared and never guessed (§2.1), so it has to be written
    // down — and `Raw` is the one that matters, because the default is `Srgb`
    // and a field that is never written passes a test that checks the default.
    assert_eq!(after.color_space, kreate_scene::ColorSpace::Raw);
    assert_eq!(
        after.resident_tiles(),
        2,
        "the image came back with {} of its 2 tiles",
        after.resident_tiles()
    );
    assert_eq!(after.tile_count(), before.tile_count());
    assert_eq!(
        after.tile(2, 1).map(|t| t.bytes()[0..4].to_vec()),
        Some(vec![1, 2, 3, 255])
    );
    assert!(after.tile(1, 1).is_none(), "a hole was filled in");
}

#[test]
fn a_worlds_background_finds_its_image_again() {
    // The whole reason the layout is worth having, and the thing
    // `WorldRecord::image_uid` has been written against for three minor
    // versions. A background that does not resolve falls back to a flat grey,
    // which is exactly what this test would see if the images were dropped.
    let mut doc = Document::new();
    let image = doc.add_image(patchy_image());
    let world = doc.add_world(kreate_scene::World {
        name: "Sky".into(),
        linkage: kreate_scene::document::Linkage::Local,
        uid: kreate_scene::Uid::NONE,
        background: kreate_scene::Background::Image(image),
        strength: 1.5,
        rotation: 0.25,
    });
    doc.set_active_world(Some(world));

    let loaded = round_trip(&doc);
    let (_, world) = loaded.document.worlds().next().expect("the world came back");
    let kreate_scene::Background::Image(id) = world.background else {
        panic!("the background came back as {:?}", world.background);
    };
    let image = loaded.document.image(id).expect("and the image it names");
    assert_eq!(image.name, "Sky");
    assert_eq!(
        image.tile(0, 0).map(|t| t.bytes()[0..4].to_vec()),
        Some(vec![9, 8, 7, 255]),
        "the background resolved to an image with the wrong pixels"
    );
}

#[test]
fn image_entries_are_stored_uncompressed_so_they_can_be_mapped() {
    // The same decision as the meshes, and it has to be made now for the same
    // reason: a file saved with its tiles deflated stays slow to open forever.
    let mut doc = Document::new();
    doc.add_image(patchy_image());
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).expect("opens");

    let mut checked = 0;
    for i in 0..zip.len() {
        let entry = zip.by_index(i).expect("entry");
        let name = entry.name().to_string();
        if name.ends_with(".ktile") {
            assert_eq!(
                entry.compression(),
                zip::CompressionMethod::Stored,
                "{name} is compressed, so it cannot be memory mapped"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 1, "the archive had {checked} tile files");
}

#[test]
fn a_declared_image_with_no_pixels_writes_no_tile_file() {
    // A 16K texture nobody has painted is a real state and an authored one —
    // the declaration is the thing, and the pixels arrive later. Writing an
    // entry for it would put sixteen thousand absences in the archive.
    let mut doc = Document::new();
    doc.add_image(kreate_scene::Image::new(
        "Declared",
        16384,
        16384,
        kreate_scene::PixelFormat::Rgba8,
    ));
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    // Asked of the archive's entries and not of its size. A tile file for an
    // image with nothing in it is forty-eight bytes of header, so a size test
    // passes whether the entry is there or not — and the claim is that the
    // entry is not there.
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes.clone())).expect("opens");
    let names: Vec<String> = zip.file_names().map(str::to_string).collect();
    assert!(
        !names.iter().any(|n| n.ends_with(".ktile")),
        "an image with no pixels wrote a tile file: {names:?}"
    );
    let _ = &mut zip;

    let loaded = container::read(Cursor::new(bytes)).expect("reads");
    let (_, image) = loaded.document.images().next().expect("still declared");
    assert_eq!(image.width(), 16384);
    assert_eq!(image.resident_tiles(), 0);
}

#[test]
fn an_images_identity_outlives_the_file() {
    // What a world's background names it by, and what an animation curve will.
    // An index would be a different thing: it says where the image sits today.
    //
    // The identity is given rather than assigned, and that is the whole of what
    // makes this test work. A document built by `add_image` numbers its images
    // from one, and so does a document built by a loader that forgot to restore
    // them — so the two agree by coincidence and the test passes with the
    // restoration gone. A document that came from a file has whatever numbers
    // that file had, which is what is being simulated here.
    let mut doc = Document::new();
    let id = doc.restore_image(patchy_image(), kreate_scene::Uid::from_raw(77));
    let uid = doc.image_uid(id).expect("restored");
    assert_eq!(uid.raw(), 77);

    let loaded = round_trip(&doc);
    let found = loaded
        .document
        .image_by_uid(uid)
        .expect("the image came back as a different datablock");
    assert_eq!(
        loaded.document.image(found).expect("its pixels").name,
        "Sky"
    );
}

#[test]
fn where_an_image_came_from_survives_even_though_its_pixels_also_do() {
    // The path is the authored thing. A user who edits `sky.hdr` outside and
    // reopens expects the new sky, and without it there would be nothing to
    // reload from — which is also what makes the thrifty version (leave the
    // pixels out, reload from here) an addition rather than a redesign.
    let mut doc = Document::new();
    let mut image = patchy_image();
    image.source = kreate_scene::Source::File("/textures/sky.hdr".into());
    doc.add_image(image);

    let loaded = round_trip(&doc);
    let (_, image) = loaded.document.images().next().expect("came back");
    match &image.source {
        kreate_scene::Source::File(path) => {
            assert_eq!(path.to_string_lossy(), "/textures/sky.hdr")
        }
        other => panic!("the path came back as {other:?}"),
    }
    // And the pixels are here too, which is what makes a `.kreate` a single
    // file somebody can send.
    assert_eq!(image.resident_tiles(), 2);
}

#[test]
fn a_decoded_file_survives_the_archive_unchanged() {
    // Both ends attached: a real PNG through the importer, into a document,
    // through a save, and back. Each half has its own tests and this is the
    // only thing that checks they meet.
    let mut pixels = image::RgbaImage::new(200, 140);
    for (x, y, p) in pixels.enumerate_pixels_mut() {
        *p = image::Rgba([(x % 256) as u8, (y % 256) as u8, 42, 255]);
    }
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(pixels)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("encodes");

    let decoded = kreate_io::load_image(
        &encoded.into_inner(),
        "Checker",
        kreate_scene::ColorSpace::Srgb,
    )
    .expect("decodes");
    let mut doc = Document::new();
    doc.add_image(decoded);

    let loaded = round_trip(&doc);
    let (_, image) = loaded.document.images().next().expect("came back");
    assert_eq!((image.width(), image.height()), (200, 140));
    assert_eq!(image.color_space, kreate_scene::ColorSpace::Srgb);
    // A pixel from the middle of a tile that is not the first one, so a walk
    // that lost a tile or shifted one is visible.
    let tile = image.tile(1, 1).expect("resident");
    let at = ((10 * kreate_scene::TILE) + 10) as usize * 4;
    assert_eq!(
        &tile.bytes()[at..at + 4],
        &[138, 138, 42, 255],
        "the pixel at (138, 138) is not the one that belongs there"
    );
}

#[test]
fn a_build_that_has_never_heard_of_images_does_not_destroy_one() {
    // The forwards-compatibility rule, applied to the addition this is. An
    // older reader keeps the record in its `extra` map and the tile file as an
    // unknown entry, and writes both back — so a document with an environment
    // survives a round trip through a build from before environments.
    //
    // Simulated the only way it can be from here: the entry and the field are
    // handed back exactly as `container::read` produced them, which is what
    // such a reader would do.
    let mut doc = Document::new();
    doc.add_image(patchy_image());
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");

    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).expect("opens");
    let mut carried = Vec::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).expect("entry");
        let name = entry.name().to_string();
        if name.ends_with(".ktile") {
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut bytes).expect("reads");
            carried.push(UnknownEntry { name, bytes });
        }
    }
    assert_eq!(carried.len(), 1);

    // A document with no images at all, saved with that entry carried through —
    // which is what an older build holds after opening the file above.
    let rewritten =
        container::write(&Document::new(), &Manifest::default(), &carried).expect("writes");
    let mut zip = zip::ZipArchive::new(Cursor::new(rewritten)).expect("opens");
    assert!(
        zip.by_name("images/000000.ktile").is_ok(),
        "the tile file was discarded by a build that did not understand it"
    );
}

// ---------------------------------------------------------------------------
// One name for the UV layer.

#[test]
fn a_file_written_with_the_old_uv_name_comes_back_with_the_new_one() {
    // **The two halves never met.** The importers wrote `uv_map` and the UV
    // editor and the unwrapper read `uv` — so a model imported from glTF had
    // coordinates the editor could not show, and coordinates unwrapped here did
    // not export. Every test on either side passed, because each was asking its
    // own half.
    //
    // A file saved before the names were joined still carries the old one, so
    // the reader renames it.
    use kreate_mesh::attribute::{names, AttrData, Domain};
    let mut mesh = kreate_mesh::primitives::cube(1.0);
    let corners = mesh.num_corners() as usize;
    let uvs: Vec<glam::Vec2> = (0..corners)
        .map(|c| glam::Vec2::new(c as f32 * 0.01, 0.5))
        .collect();
    // The primitive's own layer is taken away first: a file from before the
    // names were joined had `uv_map` and nothing else, and leaving both here
    // would be testing the *other* rule — that a mesh with both keeps both,
    // which the test below this one covers.
    mesh.attributes_mut().remove(names::UV);
    mesh.attributes_mut().insert(
        "uv_map",
        Domain::Corner,
        AttrData::F32x2(uvs.clone().into()),
    );

    let mut doc = Document::new();
    let id = doc.add_mesh(mesh);
    doc.add_object(Object::new("Cube", ObjectData::Mesh(id)));

    let loaded = round_trip(&doc);
    let (_, back) = loaded.document.meshes().next().expect("the mesh");
    let got = back
        .attributes()
        .get_f32x2(names::UV)
        .expect("the coordinates came back under the name everything reads");
    assert_eq!(got.len(), corners);
    assert_eq!(got[3], uvs[3], "the values changed on the way through");
    assert!(
        back.attributes().get("uv_map").is_none(),
        "the old layer was copied rather than renamed"
    );
}

#[test]
fn a_mesh_that_really_has_a_layer_called_uv_map_keeps_it() {
    // Somebody's own layer, which happens to be spelled the way the old one
    // was. Taking it away would be worse than leaving it where they put it —
    // so the rename only fires when there is nothing under the real name.
    use kreate_mesh::attribute::{names, AttrData, Domain};
    let mut mesh = kreate_mesh::primitives::cube(1.0);
    let corners = mesh.num_corners() as usize;
    let real: Vec<glam::Vec2> = (0..corners).map(|_| glam::Vec2::new(1.0, 1.0)).collect();
    let theirs: Vec<glam::Vec2> = (0..corners).map(|_| glam::Vec2::new(0.25, 0.75)).collect();
    mesh.attributes_mut()
        .insert(names::UV, Domain::Corner, AttrData::F32x2(real.into()));
    mesh.attributes_mut()
        .insert("uv_map", Domain::Corner, AttrData::F32x2(theirs.into()));

    let mut doc = Document::new();
    let id = doc.add_mesh(mesh);
    doc.add_object(Object::new("Cube", ObjectData::Mesh(id)));

    let loaded = round_trip(&doc);
    let (_, back) = loaded.document.meshes().next().expect("the mesh");
    assert_eq!(
        back.attributes().get_f32x2(names::UV).expect("the real one")[0],
        glam::Vec2::new(1.0, 1.0)
    );
    assert_eq!(
        back.attributes().get_f32x2("uv_map").expect("and theirs")[0],
        glam::Vec2::new(0.25, 0.75)
    );
}
