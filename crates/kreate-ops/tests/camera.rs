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

// --- exposure ---------------------------------------------------------------
//
// The bridge between §2.2's photometric lights and §2.1's display transform.
// A sun is twenty thousand lux and AgX expects scene-linear values where 0.18
// is mid grey with four stops of room above one; without an exposure, a
// physically correct scene saturates every pixel. The rendered viewport's
// first tests measured exactly that — 831 of 831 lit pixels clipped — which is
// how this came to be written.
//
// The tests below are photographic identities rather than numbers I chose.
// Anybody with a light meter can check them, which is the point of storing the
// aperture and the shutter rather than an abstract brightness.

/// The scene rate the shutter fraction is resolved against.
const RATE: f64 = 24.0;

/// A camera set the way a photographer would describe it.
fn set(f_stop: f64, shutter_seconds: f64, iso: f64) -> Camera {
    Camera {
        f_stop: Some(f_stop),
        // The stored shutter is a fraction of a frame, so a duration has to be
        // turned back into one. That indirection is deliberate — see
        // `Camera::shutter`, which keeps a 180° shutter a 180° shutter when
        // the rate changes.
        shutter: shutter_seconds * RATE,
        iso,
        ..Camera::new("Camera")
    }
}

#[test]
fn sunny_sixteen_exposes_a_grey_card_correctly() {
    // The oldest rule in photography: on a sunny day, f/16 at one over the ISO
    // gives a correct exposure. At ISO 100 that is 1/100 s, and "correct"
    // means an 18% grey card lands on 0.18.
    //
    // Direct sun is about 100,000 lux; a grey card reflects 18% of it back
    // diffusely, so its luminance is `100000 × 0.18 / π` nits.
    let camera = set(16.0, 1.0 / 100.0, 100.0);
    let card = 100_000.0 * 0.18 / std::f64::consts::PI;
    let exposed = card * camera.exposure_scale(RATE);
    assert!(
        (0.14..0.24).contains(&exposed),
        "a sunlit grey card exposed to {exposed}, not near 0.18"
    );
}

#[test]
fn each_leg_of_the_triangle_is_worth_its_stop() {
    // Opening one stop, halving the shutter speed and doubling the ISO are the
    // same change. A renderer that got one of the three wrong would look
    // right to anybody who only ever touched the others.
    let base = set(5.6, 1.0 / 60.0, 100.0);
    let one_stop = 2f64.sqrt();

    let wider = set(5.6 / one_stop, 1.0 / 60.0, 100.0);
    let slower = set(5.6, 1.0 / 30.0, 100.0);
    let faster_film = set(5.6, 1.0 / 60.0, 200.0);

    let brighter = base.exposure_scale(RATE) * 2.0;
    for (name, camera) in [
        ("aperture", wider),
        ("shutter", slower),
        ("sensitivity", faster_film),
    ] {
        let got = camera.exposure_scale(RATE);
        assert!(
            (got - brighter).abs() < brighter * 1e-6,
            "{name} was not worth a stop: {got} against {brighter}"
        );
    }
}

#[test]
fn the_compensation_dial_is_stops_and_nothing_else() {
    // The control a person actually reaches for. The triangle sets the look —
    // how much is in focus, how much is blurred — and this sets the brightness
    // without disturbing either.
    let mut camera = set(5.6, 1.0 / 60.0, 100.0);
    let flat = camera.exposure_scale(RATE);
    camera.exposure_compensation = 1.0;
    assert!((camera.exposure_scale(RATE) - flat * 2.0).abs() < flat * 1e-9);
    camera.exposure_compensation = -2.0;
    assert!((camera.exposure_scale(RATE) - flat / 4.0).abs() < flat * 1e-9);

    // And it does not touch the depth of field or the blur, which is the whole
    // reason it is a separate number rather than a nudge to the aperture.
    assert_eq!(camera.f_stop, Some(5.6));
    assert_eq!(camera.shutter, (1.0 / 60.0) * RATE);
}

#[test]
fn the_stops_form_and_the_scale_form_agree() {
    // Two spellings of one number, and the display transform takes the
    // logarithmic one. They are easy to confuse and the wrong one still
    // produces a picture.
    let camera = set(2.8, 1.0 / 250.0, 400.0);
    let scale = camera.exposure_scale(RATE);
    let stops = camera.display_exposure(RATE);
    assert!((2f64.powf(stops) - scale).abs() < scale * 1e-9);
}

#[test]
fn turning_off_the_depth_of_field_does_not_turn_off_the_light() {
    // `f_stop: None` means "do not simulate depth of field", which is a
    // rendering choice. Reading it as an infinite f-number would make a
    // pinhole camera see nothing at all, and the symptom would be a black
    // viewport with no error.
    let pinhole = Camera {
        f_stop: None,
        shutter: (1.0 / 60.0) * RATE,
        ..Camera::new("Pinhole")
    };
    let scale = pinhole.exposure_scale(RATE);
    assert!(scale.is_finite() && scale > 0.0, "{scale}");
    // It exposes as the stand-in aperture would.
    let stand_in = set(
        kreate_scene::camera::DEFAULT_EXPOSURE_F_STOP,
        1.0 / 60.0,
        100.0,
    );
    assert!((scale - stand_in.exposure_scale(RATE)).abs() < scale * 1e-9);
}

#[test]
fn an_instant_shutter_does_not_turn_off_the_light_either() {
    // Same argument: zero shutter is no motion blur, not a closed camera.
    let sharp = Camera {
        shutter: 0.0,
        f_stop: Some(5.6),
        ..Camera::new("Sharp")
    };
    let scale = sharp.exposure_scale(RATE);
    assert!(scale.is_finite() && scale > 0.0, "{scale}");
}

#[test]
fn nothing_a_file_can_hold_produces_a_nan() {
    // A hand-edited file, an importer, a script. An exposure of NaN is a black
    // or white frame with nothing to point at.
    for camera in [
        Camera {
            iso: 0.0,
            ..Camera::new("Zero ISO")
        },
        Camera {
            iso: -100.0,
            ..Camera::new("Negative ISO")
        },
        Camera {
            f_stop: Some(0.0),
            ..Camera::new("Zero aperture")
        },
        Camera {
            f_stop: Some(-2.8),
            ..Camera::new("Negative aperture")
        },
        Camera {
            shutter: -1.0,
            ..Camera::new("Negative shutter")
        },
    ] {
        let scale = camera.exposure_scale(RATE);
        assert!(
            scale.is_finite() && scale > 0.0,
            "{} gave {scale}",
            camera.name
        );
    }
    // And a rate of zero, which is what a document with no rate would hand
    // over.
    assert!(Camera::new("C").exposure_scale(0.0).is_finite());
}

#[test]
fn a_default_camera_frames_an_ordinary_room() {
    // Not a calibration, a sanity check: the camera an object arrives with
    // should make an ordinary interior look like something rather than like
    // black or white. Five hundred lux on a mid-grey wall is an office.
    let camera = Camera::new("Camera");
    let wall = 500.0 * 0.5 / std::f64::consts::PI;
    let exposed = wall * camera.exposure_scale(RATE);
    assert!(
        (0.001..1.0).contains(&exposed),
        "an office wall exposed to {exposed}"
    );
}
