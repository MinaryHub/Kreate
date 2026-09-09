//! A camera is a physical instrument, and a datablock.
//!
//! §3's seventh "decide now", checked from both directions: the operator that
//! creates one, and what the document then holds. The claim being tested is not
//! that the numbers are stored — that is a struct — but that the *focal length*
//! is the stored one and the field of view is derived from it. A tool that
//! stored the angle would pass every test about round-tripping a camera and
//! could not tell an exporter what lens it was.

use kreate_ops::operator::{Context, Mode};
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{Camera, Document};

fn setup() -> (OperatorRegistry, History, Document, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (
        registry,
        History::new(),
        Document::new(),
        Context { mode: Mode::Object },
    )
}

fn run(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    ctx: &Context,
    op: &str,
    pairs: &[(&'static str, PropertyValue)],
) -> kreate_ops::OpResult {
    let mut params = PropertyMap::new();
    for (k, v) in pairs {
        params.set(k, v.clone());
    }
    history
        .dispatch(registry, doc, ctx, op, &params)
        .expect("dispatch")
}

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

#[test]
fn adding_a_camera_makes_an_object_and_a_datablock() {
    // Two things, and the distinction is the point: the object is *where* the
    // camera is and the datablock is *what* it is. A bare `ObjectData::Camera`
    // tag — which is what this was — says an object is a camera and nothing
    // about the lens.
    let (r, mut h, mut d, c) = setup();
    run(&r, &mut h, &mut d, &c, "object.add_camera", &[]);

    assert_eq!(d.objects().count(), 1);
    assert_eq!(d.camera_count(), 1);
    let camera = only_camera(&d);
    assert_eq!(camera.focal_length_mm, 50.0);
    assert!(!camera.uid.is_none(), "the datablock has no identity");
}

#[test]
fn the_focal_length_is_what_is_stored() {
    // The decision, stated where it can fail. A camera created at 35mm holds
    // 35mm, and the angle follows from it — so an exporter has a lens to write
    // and a renderer has a number to divide by an f-stop.
    let (r, mut h, mut d, c) = setup();
    run(
        &r,
        &mut h,
        &mut d,
        &c,
        "object.add_camera",
        &[("focal_length_mm", PropertyValue::Float(35.0))],
    );
    let camera = only_camera(&d);
    assert_eq!(camera.focal_length_mm, 35.0);
    // 36mm of sensor at 35mm: `2·atan(36 / 70)` is 54.4°.
    let across = camera.fov_x(16.0 / 9.0).to_degrees();
    assert!((across - 54.43).abs() < 0.05, "{across}");
}

#[test]
fn a_zero_f_number_is_a_pinhole_and_not_an_infinite_aperture() {
    // Where the two forms meet. A slider cannot hold `None`, so the parameter
    // is a plain float and zero means off — and the datablock says so in its
    // type rather than carrying a zero that every consumer has to remember to
    // check before dividing by it.
    let (r, mut h, mut d, c) = setup();
    run(&r, &mut h, &mut d, &c, "object.add_camera", &[]);
    let camera = only_camera(&d);
    assert_eq!(camera.f_stop, None);
    assert!(!camera.has_depth_of_field());
    assert_eq!(camera.aperture_diameter_m(), None);

    let (r, mut h, mut d, c) = setup();
    run(
        &r,
        &mut h,
        &mut d,
        &c,
        "object.add_camera",
        &[("f_stop", PropertyValue::Float(2.8))],
    );
    let camera = only_camera(&d);
    assert_eq!(camera.f_stop, Some(2.8));
    assert!(camera.has_depth_of_field());
}

#[test]
fn a_camera_undoes_in_one_step_and_takes_its_datablock_with_it() {
    // A datablock left behind by an undone creation is an orphan nobody can
    // reach and nothing will collect — and it would be written to the next
    // save.
    let (r, mut h, mut d, c) = setup();
    run(&r, &mut h, &mut d, &c, "object.add_camera", &[]);
    assert_eq!(d.camera_count(), 1);

    assert!(h.undo(&mut d), "the creation should undo");
    assert_eq!(d.objects().count(), 0);
    assert_eq!(d.camera_count(), 0, "the datablock outlived its object");
}

#[test]
fn two_objects_can_share_one_lens() {
    // §1.4's argument, for cameras. A stereo pair or a set of matched angles
    // wants one lens and several positions, and an ownership model that could
    // not express that would have to be unpicked from every call site.
    let mut doc = Document::new();
    let lens = doc.add_camera(Camera::new("Shared"));
    let left = doc.add_object(Object::new("Left", ObjectData::Camera(lens)));
    let right = doc.add_object(Object::new("Right", ObjectData::Camera(lens)));
    assert_ne!(left, right);
    assert_eq!(doc.camera_count(), 1);

    // Changing the lens changes both, which is what sharing means.
    doc.camera_mut(lens).expect("the lens").focal_length_mm = 85.0;
    for (_, object) in doc.objects() {
        let ObjectData::Camera(id) = object.data else {
            panic!("not a camera")
        };
        assert_eq!(doc.camera(id).expect("lens").focal_length_mm, 85.0);
    }
}
