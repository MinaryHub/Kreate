//! A camera through the file and back.
//!
//! The datablock and its operator are tested in `kreate-ops`; what is here is
//! the format's half: that every physical parameter survives, that the identity
//! does, that a shared lens comes back as one datablock, and that a scene with
//! no camera in it writes nothing about cameras.

use kreate_io::container::{self, Manifest};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{Camera, Document, Projection, SensorFit};
use std::io::Cursor;

/// The camera datablock the one camera object in a document refers to.
fn only_camera(doc: &Document) -> &Camera {
    let id = doc
        .objects()
        .filter_map(|(_, o)| match o.data {
            ObjectData::Camera(id) => Some(id),
            _ => None,
        })
        .next()
        .expect("a camera object");
    doc.camera(id).expect("its datablock")
}

fn round_trip(doc: &Document) -> Document {
    let bytes = container::write(doc, &Manifest::default(), &[]).expect("writes");
    container::read(Cursor::new(bytes)).expect("opens").document
}

#[test]
fn every_physical_parameter_survives_a_save_and_a_load() {
    // Deliberately nothing at its default, so a writer that dropped a field
    // and a reader that filled it in from the default would not cancel out.
    let mut doc = Document::new();
    let lens = doc.add_camera(Camera {
        projection: Projection::Orthographic,
        focal_length_mm: 85.0,
        sensor_width_mm: 23.5,
        sensor_height_mm: 15.6,
        sensor_fit: SensorFit::Vertical,
        ortho_scale: 4.25,
        clip_near: 0.05,
        clip_far: 250.0,
        f_stop: Some(1.4),
        focus_distance: 2.75,
        shutter: 0.25,
        ..Camera::new("Anamorphic")
    });
    doc.add_object(Object::new("Shot", ObjectData::Camera(lens)));
    let before = doc.camera(lens).expect("the lens").clone();
    assert_ne!(
        before,
        Camera::new("Anamorphic"),
        "the fixture is all defaults"
    );

    let after = round_trip(&doc);
    assert_eq!(after.camera_count(), 1);
    assert_eq!(only_camera(&after), &before, "a camera came back different");

    // And the object still refers to it, rather than having become an empty.
    assert!(
        after
            .objects()
            .any(|(_, o)| matches!(o.data, ObjectData::Camera(_))),
        "the object lost its camera"
    );
}

#[test]
fn a_pinhole_stays_a_pinhole() {
    // `None` and `Some(0.0)` are different things — one skips the depth of
    // field sampling entirely — and a format that wrote the absent case as a
    // zero would turn every pinhole into an infinite aperture on load.
    let mut doc = Document::new();
    let lens = doc.add_camera(Camera {
        f_stop: None,
        ..Camera::new("Pinhole")
    });
    doc.add_object(Object::new("Shot", ObjectData::Camera(lens)));

    let after = round_trip(&doc);
    assert_eq!(only_camera(&after).f_stop, None);
    assert!(!only_camera(&after).has_depth_of_field());
}

#[test]
fn a_shared_lens_is_still_one_datablock_after_a_round_trip() {
    // The half a naive writer gets wrong: writing the camera inside each object
    // would come back as two lenses that no longer move together, and the
    // symptom is a stereo pair that drifts apart when one eye is adjusted.
    let mut doc = Document::new();
    let lens = doc.add_camera(Camera::new("Shared"));
    doc.add_object(Object::new("Left", ObjectData::Camera(lens)));
    doc.add_object(Object::new("Right", ObjectData::Camera(lens)));

    let after = round_trip(&doc);
    assert_eq!(after.objects().count(), 2);
    assert_eq!(after.camera_count(), 1, "the shared lens was duplicated");

    let ids: Vec<_> = after
        .objects()
        .filter_map(|(_, o)| match o.data {
            ObjectData::Camera(id) => Some(id),
            _ => None,
        })
        .collect();
    assert_eq!(ids.len(), 2);
    assert_eq!(ids[0], ids[1], "the two objects point at different lenses");
}

#[test]
fn a_camera_keeps_its_identity_across_a_save() {
    // A curve driving the focal length has to find the same datablock when the
    // scene is opened again. Checked with a gap in the numbering, because a
    // loader that issued fresh identities in order would otherwise agree.
    let mut doc = Document::new();
    let spare = doc.add_camera(Camera::new("Spare"));
    let lens = doc.add_camera(Camera::new("Keeper"));
    doc.add_object(Object::new("Shot", ObjectData::Camera(lens)));
    let _ = spare;
    let before = doc.camera_uid(lens).expect("an identity");
    assert_ne!(
        before,
        kreate_scene::Uid::FIRST,
        "the fixture's camera is the first one, so it cannot tell reading an \
         identity from issuing one"
    );

    let after = round_trip(&doc);
    // Only the referenced camera is written; the spare has no object and no
    // reference, so it is not part of the scene that comes back.
    assert_eq!(only_camera(&after).uid, before);
    // And it is a real identity on both sides. Without this the test passes
    // when *neither* side assigns one — nothing equals nothing — which is how
    // a sweep found this test agreeing with a document that issued no
    // identities at all.
    assert!(
        !only_camera(&after).uid.is_none(),
        "the camera came back with no identity"
    );
}

#[test]
fn a_scene_with_no_camera_writes_nothing_about_cameras() {
    // Which is what kept the golden files from moving when this was added. An
    // addition that costs an existing scene bytes is an addition that shows up
    // in every diff of every file forever.
    let mut doc = Document::new();
    doc.add_object(Object::new("Empty", ObjectData::Empty));
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        !text.contains("focal_length_mm"),
        "a camera-less scene wrote camera fields"
    );
}

#[test]
fn an_unknown_projection_opens_rather_than_refusing() {
    // A scene from a later version. One camera framed wrongly is worth
    // opening; a file that will not open is not.
    let mut doc = Document::new();
    let lens = doc.add_camera(Camera::new("Future"));
    doc.add_object(Object::new("Shot", ObjectData::Camera(lens)));
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");

    // Rewrite the projection to something this build has never heard of.
    let doctored = rewrite_entry(&bytes, "scene.json", |text| {
        text.replace("\"perspective\"", "\"panoramic-equirectangular\"")
    });
    let after = container::read(Cursor::new(doctored)).expect("opens anyway");
    assert_eq!(after.document.camera_count(), 1);
    assert_eq!(
        only_camera(&after.document).projection,
        Projection::Perspective,
        "an unknown projection should fall back rather than vanish"
    );
}

/// Rewrites one text entry of a zip archive, leaving the rest alone.
fn rewrite_entry(bytes: &[u8], name: &str, f: impl Fn(&str) -> String) -> Vec<u8> {
    let mut source = zip::ZipArchive::new(Cursor::new(bytes)).expect("archive");
    let mut out = Vec::new();
    {
        let mut writer = zip::ZipWriter::new(Cursor::new(&mut out));
        let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
        for i in 0..source.len() {
            let mut entry = source.by_index(i).expect("entry");
            let entry_name = entry.name().to_string();
            let mut content = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut content).expect("read");
            if entry_name == name {
                content = f(&String::from_utf8_lossy(&content)).into_bytes();
            }
            writer.start_file(entry_name, options).expect("start");
            std::io::Write::write_all(&mut writer, &content).expect("write");
        }
        writer.finish().expect("finish");
    }
    out
}
