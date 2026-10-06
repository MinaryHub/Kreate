//! The feel of the interactive transform, as arithmetic.
//!
//! `Gesture::params` is a pure function of a cursor position and a camera, which
//! is the whole reason it is shaped that way: "dragging a hundred pixels moves
//! the selection a hundred pixels' worth of world", "dragging up moves it up",
//! "a circular drag turns it the way the pointer goes" are all claims that can
//! be checked by numbers instead of by a person with a mouse.

use glam::DVec3;
use kreate::modal::{Axis, Gesture, Kind, Orientation, Snap};
use kreate::snap::SnapWorld;
use kreate_ops::PropertyValue;
use kreate_render::Camera;

const VIEWPORT: (f32, f32) = (800.0, 600.0);

/// A local basis that is the world's, so a test that says nothing about
/// orientation gets the same answer either way.
const WORLD_AXES: [glam::Vec3; 3] = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];

/// A camera looking along −Y at the origin from eight units away, so world X is
/// screen right and world Z is screen up. That makes every expectation below
/// readable rather than a matrix multiplication in a comment.
fn camera() -> Camera {
    Camera {
        pivot: DVec3::ZERO,
        distance: 8.0,
        // Identity looks down −Z, which is a top view. A quarter turn about X
        // puts the eye on −Y looking towards the origin.
        rotation: kreate_math::DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
        ..Camera::default()
    }
}

/// Moves the pointer and reads what the gesture now means.
///
/// A gesture *tracks* the pointer rather than being handed a position, because
/// rotation has to count whole turns as they happen — a single `atan2`
/// difference cannot express more than half of one. So every test drives it the
/// way the event loop does: report the motion, then read the parameters.
fn at(g: &mut Gesture, cursor: (f32, f32)) -> kreate_ops::PropertyMap {
    g.track(cursor);
    g.params()
}

fn translation(params: &kreate_ops::PropertyMap) -> [f64; 3] {
    match params.get("translation") {
        Some(PropertyValue::Vec3(v)) => *v,
        other => panic!("expected a translation, got {other:?}"),
    }
}

fn angle(params: &kreate_ops::PropertyMap) -> f64 {
    match params.get("angle") {
        Some(PropertyValue::Float(f)) => *f,
        other => panic!("expected an angle, got {other:?}"),
    }
}

fn scale(params: &kreate_ops::PropertyMap) -> [f64; 3] {
    match params.get("scale") {
        Some(PropertyValue::Vec3(v)) => *v,
        other => panic!("expected a scale, got {other:?}"),
    }
}

fn axis_index(params: &kreate_ops::PropertyMap) -> u32 {
    match params.get("axis") {
        Some(PropertyValue::Enum(i)) => *i,
        other => panic!("expected an axis, got {other:?}"),
    }
}

#[test]
fn a_gesture_that_has_not_moved_asks_for_nothing() {
    // The invariant the whole "restore then apply" design rests on: pressing the
    // key must not move anything. If the identity were not the identity, every
    // gesture would nudge the selection the instant it began.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    assert_eq!(translation(&at(&mut g, (400.0, 300.0))), [0.0, 0.0, 0.0]);

    let mut g = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    assert_eq!(angle(&at(&mut g, (500.0, 300.0))), 0.0);

    let mut g = Gesture::begin(
        Kind::Scale,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    assert_eq!(scale(&at(&mut g, (500.0, 300.0))), [1.0, 1.0, 1.0]);
}

#[test]
fn dragging_right_moves_the_selection_right() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let t = translation(&at(&mut g, (500.0, 300.0)));
    assert!(t[0] > 0.5, "dragging right gave {t:?}");
    assert!(t[2].abs() < 1e-6, "and should not have moved it vertically");
}

#[test]
fn dragging_up_moves_the_selection_up() {
    // Window coordinates grow downwards and world up does not. Getting the sign
    // wrong here is the classic "drag up, it goes down" bug, and it is invisible
    // in a screenshot of a symmetric object.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let t = translation(&at(&mut g, (400.0, 200.0)));
    assert!(t[2] > 0.5, "dragging up gave {t:?}");
}

#[test]
fn a_free_move_tracks_the_cursor_one_to_one() {
    // What makes a move feel attached to the pointer rather than geared. A
    // hundred pixels of cursor is a hundred pixels' worth of world at the
    // selection's depth, so the selection stays under the mouse.
    let camera = camera();
    let pivot = DVec3::ZERO;
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        pivot,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );

    // The world height of the viewport at the pivot's distance.
    let distance = (pivot - camera.eye()).length();
    let world_height = 2.0 * (camera.fov_y * 0.5).tan() * distance;
    let per_pixel = world_height / VIEWPORT.1 as f64;

    let t = translation(&at(&mut g, (400.0 + 150.0, 300.0)));
    assert!(
        (t[0] - 150.0 * per_pixel).abs() < 1e-4,
        "moved {} for 150 px, expected {}",
        t[0],
        150.0 * per_pixel
    );
}

#[test]
fn a_move_is_bigger_when_the_selection_is_further_away() {
    // The same consequence from the other side: the mapping is a world size per
    // pixel, so a distant selection covers more ground for the same drag. A
    // fixed scale would make far objects crawl and near ones fly.
    let camera = camera();
    let mut near = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let mut far = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::new(0.0, 40.0, 0.0),
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let a = translation(&at(&mut near, (500.0, 300.0)))[0];
    let b = translation(&at(&mut far, (500.0, 300.0)))[0];
    assert!(b > a * 2.0, "near {a}, far {b}");
}

#[test]
fn an_axis_constraint_confines_the_move_to_that_axis() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::Z);
    // A drag that is mostly sideways, so the unconstrained answer would be
    // mostly X — the constraint has to discard that part rather than rotate it.
    let t = translation(&at(&mut g, (520.0, 250.0)));
    assert_eq!(t[0], 0.0);
    assert_eq!(t[1], 0.0);
    assert!(t[2] > 0.0, "got {t:?}");
}

#[test]
fn an_axis_key_cycles_free_global_local_free() {
    // Two orientations, because both are wanted often and neither is obviously
    // the default: moving a wheel along the world's X is a different request
    // from moving it along the axle it is mounted on.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    assert_eq!(g.axis, None);

    g.cycle_axis(Axis::X);
    assert_eq!(
        (g.axis, g.orientation),
        (Some(Axis::X), Orientation::Global)
    );
    g.cycle_axis(Axis::X);
    assert_eq!((g.axis, g.orientation), (Some(Axis::X), Orientation::Local));
    g.cycle_axis(Axis::X);
    assert_eq!(g.axis, None, "a third press should free it again");
}

#[test]
fn a_different_axis_key_starts_its_own_cycle() {
    // Pressing Y after XX is a new request, not a continuation of X's. Carrying
    // the local orientation across would mean the second key did something the
    // user did not ask for and could not see.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    g.cycle_axis(Axis::X);
    assert_eq!(g.orientation, Orientation::Local);
    g.cycle_axis(Axis::Y);
    assert_eq!(
        (g.axis, g.orientation),
        (Some(Axis::Y), Orientation::Global),
        "a new axis should start from global"
    );
}

#[test]
fn a_local_constraint_uses_the_selections_own_axes() {
    // The whole point of the second press. With a basis turned a quarter turn
    // about Z, local X is world Y.
    let camera = camera();
    let turned = [glam::Vec3::Y, -glam::Vec3::X, glam::Vec3::Z];
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        turned,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    for c in "2".chars() {
        g.type_character(c);
    }
    let global = translation(&at(&mut g, (400.0, 300.0)));
    assert_eq!(global, [2.0, 0.0, 0.0], "global X is world X");

    g.cycle_axis(Axis::X);
    let local = translation(&at(&mut g, (400.0, 300.0)));
    assert_eq!(local, [0.0, 2.0, 0.0], "local X is this basis's X");
}

#[test]
fn a_local_axis_is_normalised() {
    // An object transform's axes are not unit vectors when the object is scaled.
    // Left as they are, a constrained move on a doubled object would travel
    // twice as far as the number said.
    let camera = camera();
    let doubled = [
        glam::Vec3::X * 2.0,
        glam::Vec3::Y * 2.0,
        glam::Vec3::Z * 2.0,
    ];
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        doubled,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    g.cycle_axis(Axis::X);
    g.type_character('3');
    assert_eq!(translation(&at(&mut g, (400.0, 300.0))), [3.0, 0.0, 0.0]);
}

#[test]
fn precision_mode_slows_the_gesture_without_moving_it() {
    // §2.2's rule, and the reason precision is measured from the moment the
    // modifier goes down: engaging it must not make the selection jump back
    // towards where the drag started.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let at_engage = translation(&at(&mut g, (500.0, 300.0)))[0];

    g.set_precise(true, (500.0, 300.0));
    let straight_after = translation(&at(&mut g, (500.0, 300.0)))[0];
    assert!(
        (at_engage - straight_after).abs() < 1e-6,
        "engaging precision moved it from {at_engage} to {straight_after}"
    );

    // A further hundred pixels now counts for a tenth as much.
    let coarse_hundred = at_engage;
    let precise = translation(&at(&mut g, (600.0, 300.0)))[0] - at_engage;
    assert!(
        (precise - coarse_hundred * 0.1).abs() < 1e-4,
        "a precise hundred pixels moved {precise}, expected {}",
        coarse_hundred * 0.1
    );
}

#[test]
fn releasing_precision_mode_does_not_move_it_either() {
    // The other half. A modifier that can be held and released mid-drag has two
    // transitions, and only checking one of them is how the second one ships
    // broken.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.set_precise(true, (500.0, 300.0));
    let before = translation(&at(&mut g, (600.0, 300.0)))[0];
    g.set_precise(false, (600.0, 300.0));
    let after = translation(&at(&mut g, (600.0, 300.0)))[0];
    assert!(
        (before - after).abs() < 1e-6,
        "releasing precision moved it from {before} to {after}"
    );

    // And it is coarse again.
    let further = translation(&at(&mut g, (700.0, 300.0)))[0] - after;
    let coarse_hundred = {
        let mut fresh = Gesture::begin(
            Kind::Move,
            (0.0, 0.0),
            DVec3::ZERO,
            &camera,
            VIEWPORT,
            WORLD_AXES,
            Snap::default(),
            SnapWorld::default(),
        );
        translation(&at(&mut fresh, (100.0, 0.0)))[0]
    };
    assert!(
        (further - coarse_hundred).abs() < 1e-4,
        "after releasing, a hundred pixels moved {further}, expected {coarse_hundred}"
    );
}

#[test]
fn a_typed_number_overrides_the_pointer_entirely() {
    // The bridge between the mouse and the keyboard: whatever the pointer is
    // doing, a typed value is exact. Without this a modal tool cannot be used
    // for anything dimensioned.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    for c in "2.5".chars() {
        assert!(g.type_character(c));
    }
    // A wild pointer position, to prove it is ignored.
    let t = translation(&at(&mut g, (-1000.0, 5000.0)));
    assert_eq!(t, [2.5, 0.0, 0.0]);
}

#[test]
fn the_numeric_buffer_takes_expressions() {
    // The buffer holds an expression, so operators, brackets and unit letters
    // all have to get in. Whether what has been typed *means* anything is
    // decided when it is evaluated, which is the only place that can decide.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    for c in "1.4/8".chars() {
        assert!(g.type_character(c), "{c} was refused");
    }
    let t = translation(&at(&mut g, (400.0, 300.0)));
    assert!((t[0] - 0.175).abs() < 1e-9, "got {t:?}");
}

#[test]
fn a_unit_suffix_is_understood_in_the_field_it_belongs_to() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::Y);
    for c in "50cm".chars() {
        g.type_character(c);
    }
    let t = translation(&at(&mut g, (400.0, 300.0)));
    assert!((t[1] - 0.5).abs() < 1e-9, "got {t:?}");
}

#[test]
fn a_unit_the_field_cannot_use_leaves_the_pointer_in_charge() {
    // `30deg` in a distance is a mistake. The expression refuses it, and a
    // refused expression hands control back to the pointer rather than reading
    // as zero — which would snap the selection back to where it started.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    for c in "30deg".chars() {
        g.type_character(c);
    }
    let t = translation(&at(&mut g, (500.0, 300.0)));
    assert!(t[0] > 0.5, "the pointer should still be driving: {t:?}");
}

#[test]
fn a_rotation_field_reads_degrees_and_accepts_radians() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    for c in "90".chars() {
        g.type_character(c);
    }
    let a = angle(&at(&mut g, (500.0, 300.0)));
    assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-9, "got {a}");

    for _ in 0..2 {
        g.backspace();
    }
    for c in "1rad".chars() {
        g.type_character(c);
    }
    let a = angle(&at(&mut g, (500.0, 300.0)));
    assert!((a - 1.0).abs() < 1e-9, "got {a}");
}

#[test]
fn the_numeric_buffer_takes_only_what_it_can_use() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    assert!(g.type_character('-'), "a sign belongs in an expression");
    assert!(g.type_character('4'));
    // A letter is taken, because `4cm` is a real thing to type. It is the
    // *evaluation* that rejects `4q`, not the keystroke.
    assert!(g.type_character('q'));
    g.cycle_axis(Axis::Y);
    assert_eq!(
        translation(&at(&mut g, (400.0, 300.0))),
        [0.0, 0.0, 0.0],
        "`-4q` does not evaluate, so nothing is typed and the pointer has not moved"
    );
    g.backspace();
    assert_eq!(translation(&at(&mut g, (400.0, 300.0))), [0.0, -4.0, 0.0]);
    // A key that is not part of any expression is refused outright.
    assert!(!g.type_character('%'));
}

#[test]
fn a_partial_number_is_not_treated_as_zero() {
    // Typing `-` then dragging must not snap the selection to the origin while
    // the buffer is unusable.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.type_character('-');
    let t = translation(&at(&mut g, (500.0, 300.0)));
    assert!(t[0] > 0.5, "the pointer should still be driving: {t:?}");
}

#[test]
fn a_circular_drag_rotates_the_way_the_pointer_goes() {
    // Following the pointer around the pivot, rather than reading horizontal
    // motion. It is what the gesture looks like it should do, and it keeps
    // working when the pivot is off to one side of the window.
    let camera = camera();
    let pivot_px = (400.0f32, 300.0f32);
    // Starting to the right of the pivot and sweeping upwards is anticlockwise
    // on screen, which is a positive turn.
    let mut g = Gesture::begin(
        Kind::Rotate,
        (pivot_px.0 + 100.0, pivot_px.1),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let up = angle(&at(&mut g, (pivot_px.0, pivot_px.1 - 100.0)));
    assert!(
        (up - std::f64::consts::FRAC_PI_2).abs() < 1e-3,
        "a quarter sweep gave {up} radians"
    );

    // And the other way round is the other sign.
    let down = angle(&at(&mut g, (pivot_px.0, pivot_px.1 + 100.0)));
    assert!(
        (down + std::f64::consts::FRAC_PI_2).abs() < 1e-3,
        "the opposite sweep gave {down} radians"
    );
}

#[test]
fn a_rotation_past_the_far_side_does_not_jump_a_full_turn() {
    // Two `atan2` results differ by the swept angle only if the difference is
    // taken the short way round. Without the wrap, sweeping past the opposite
    // side of the pivot flips the sign and the selection snaps through a whole
    // revolution.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let mut previous = 0.0;
    // Three quarters of a turn, in small steps. Going *past* half a turn is the
    // whole point: a first version of this test stopped just short of it, so the
    // wrap was never exercised and removing it changed nothing. No step may move
    // the angle by more than the step itself.
    let steps = 60;
    for step in 1..=steps {
        let theta = step as f32 * (1.5 * std::f32::consts::PI) / steps as f32;
        let cursor = (400.0 + 100.0 * theta.cos(), 300.0 - 100.0 * theta.sin());
        let a = angle(&at(&mut g, cursor));
        assert!(
            (a - previous).abs() < 0.5,
            "the angle jumped from {previous} to {a} at step {step}"
        );
        previous = a;
    }
}

#[test]
fn a_scale_measures_distance_from_the_pivot_not_sideways_travel() {
    // Dragging perpendicular to the arm between the pivot and the cursor still
    // increases the distance, so it still grows the selection.
    //
    // This is the case that separates a distance ratio from anything simpler. A
    // first version of this file only dragged straight towards and away from the
    // pivot, where a ratio and a plain horizontal delta give identical answers —
    // so replacing the ratio with `1 + dx/100` passed every scale test.
    let camera = camera();
    // Pivot projects to the centre of the viewport; start 100 px to its right.
    let mut g = Gesture::begin(
        Kind::Scale,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );

    // Straight up by 100 px: the cursor is now sqrt(2) times as far from the
    // pivot, while its horizontal displacement is zero.
    let s = scale(&at(&mut g, (500.0, 200.0)))[0];
    assert!(
        (s - std::f64::consts::SQRT_2).abs() < 1e-3,
        "a perpendicular drag scaled by {s}, expected {}",
        std::f64::consts::SQRT_2
    );
}

#[test]
fn a_scale_dragged_around_the_pivot_does_not_change() {
    // The other side of the same property: moving along a circle centred on the
    // pivot keeps the distance, so it keeps the scale. Any formula reading a
    // displacement rather than a radius fails this.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Scale,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    for step in 0..8 {
        let theta = step as f32 * std::f32::consts::FRAC_PI_4;
        let cursor = (400.0 + 100.0 * theta.cos(), 300.0 - 100.0 * theta.sin());
        let s = scale(&at(&mut g, cursor))[0];
        assert!(
            (s - 1.0).abs() < 1e-3,
            "circling the pivot at step {step} scaled by {s}"
        );
    }
}

#[test]
fn a_rotation_started_on_the_pivot_does_nothing_rather_than_something_wild() {
    // Two nearly coincident points have no meaningful angle between them, and an
    // `atan2` of noise would spin the selection.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Rotate,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    assert_eq!(angle(&at(&mut g, (400.5, 300.5))), 0.0);
}

#[test]
fn a_rotation_defaults_to_the_axis_facing_the_viewer() {
    // A circular drag looks like it turns the thing about the axis pointing out
    // of the screen. With this camera on −Y, that is Y.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    assert_eq!(axis_index(&at(&mut g, (500.0, 200.0))), Axis::Y.index());
}

#[test]
fn a_typed_rotation_is_in_degrees() {
    // Degrees at the keyboard, radians in the parameter. Nobody types radians.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    for c in "90".chars() {
        g.type_character(c);
    }
    let a = angle(&at(&mut g, (500.0, 300.0)));
    assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-9, "got {a}");
}

#[test]
fn dragging_away_from_the_pivot_grows_the_selection() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Scale,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    // Twice as far from the pivot as the start.
    let s = scale(&at(&mut g, (600.0, 300.0)));
    assert!((s[0] - 2.0).abs() < 1e-3, "got {s:?}");
    assert_eq!(s[0], s[1], "a free scale is uniform");
    assert_eq!(s[1], s[2]);
}

#[test]
fn dragging_towards_the_pivot_shrinks_the_selection() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Scale,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let s = scale(&at(&mut g, (450.0, 300.0)));
    assert!((s[0] - 0.5).abs() < 1e-3, "got {s:?}");
}

#[test]
fn a_constrained_scale_stretches_one_axis_only() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Scale,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    let s = scale(&at(&mut g, (600.0, 300.0)));
    assert!((s[0] - 2.0).abs() < 1e-3, "got {s:?}");
    assert_eq!(s[1], 1.0);
    assert_eq!(s[2], 1.0);
}

#[test]
fn a_scale_started_on_the_pivot_holds_at_one() {
    // There is no reference length to take a ratio against. Held at 1 rather
    // than dividing by nothing.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Scale,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    assert_eq!(scale(&at(&mut g, (900.0, 300.0)))[0], 1.0);
}

#[test]
fn the_operator_a_gesture_runs_depends_on_the_mode() {
    // The same three keys, one set of operators per mode. Object mode moves
    // objects; edit mode moves the elements inside one.
    use kreate_ops::operator::Mode;
    assert_eq!(Kind::Move.op_id(Mode::Object), "object.translate");
    assert_eq!(Kind::Move.op_id(Mode::EditMesh), "mesh.translate");
    assert_eq!(Kind::Rotate.op_id(Mode::Object), "object.rotate");
    assert_eq!(Kind::Rotate.op_id(Mode::EditMesh), "mesh.rotate");
    assert_eq!(Kind::Scale.op_id(Mode::Object), "object.scale");
    assert_eq!(Kind::Scale.op_id(Mode::EditMesh), "mesh.scale");

    // And no two answers coincide. Three keys times the modes should be that
    // many distinct operators; a `match` arm copied and not edited would
    // otherwise pass every line above.
    // A brush mode moves nothing: its card borrows the object's three, and no
    // key in its context reaches them.
    let mut ids: Vec<&str> = Mode::ALL
        .iter()
        .filter(|mode| {
            !matches!(
                mode.acts_on(),
                kreate_ops::mode::ActsOn::Weights
                    | kreate_ops::mode::ActsOn::Sculpt
                    | kreate_ops::mode::ActsOn::Texture
            )
        })
        .flat_map(|mode| {
            [Kind::Move, Kind::Rotate, Kind::Scale]
                .into_iter()
                .map(move |kind| kind.op_id(*mode))
        })
        .collect();
    let total = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), total, "two gestures name the same operator");
}

#[test]
fn every_operator_a_gesture_names_exists() {
    // A gesture that names a missing operator fails at the moment the user
    // presses the key, which is the worst time to find out.
    let mut registry = kreate_ops::OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    // Every mode, from `Mode::ALL` rather than a list written here: a mode
    // added without an operator behind it is exactly the failure this test is
    // for, and a hardcoded pair would not see it.
    for kind in [
        Kind::Move,
        Kind::Rotate,
        Kind::Scale,
        Kind::Extrude,
        Kind::ExtrudeFree,
        Kind::Inset,
        Kind::Bevel,
        Kind::LoopCut,
    ] {
        for mode in kreate_ops::operator::Mode::ALL {
            let id = kind.op_id(mode);
            assert!(
                registry.get(id).is_some(),
                "{id} is not registered ({kind:?} in {mode:?})"
            );
        }
    }
}

#[test]
fn the_status_line_says_what_is_happening() {
    // The modal header is where the application talks to the user for the whole
    // duration of a gesture: without it, a constrained or precise drag looks
    // like the tool misbehaving.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::Z);
    g.set_precise(true, (400.0, 300.0));
    for c in "1.5".chars() {
        g.type_character(c);
    }
    g.track((450.0, 280.0));
    let status = g.status();
    assert!(status.contains("Move"), "{status}");
    assert!(status.contains('Z'), "{status}");
    assert!(status.contains("1.5"), "{status}");
    assert!(status.contains("precise"), "{status}");
}

#[test]
fn the_parameters_a_gesture_produces_are_the_operators_own() {
    // The claim that makes the adjust panel, scripting and macro replay work: a
    // confirmed drag is indistinguishable from those numbers having been typed.
    // Checked by handing the gesture's map to the operator's schema and finding
    // every key accounted for.
    let camera = camera();
    let mut registry = kreate_ops::OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);

    for (kind, mode) in [
        Kind::Move,
        Kind::Rotate,
        Kind::Scale,
        Kind::Extrude,
        Kind::ExtrudeFree,
        Kind::Inset,
        Kind::Bevel,
        Kind::LoopCut,
    ]
    .into_iter()
    .flat_map(|kind| {
        kreate_ops::operator::Mode::ALL
            .into_iter()
            .map(move |mode| (kind, mode))
    }) {
        let mut g = Gesture::begin(
            kind,
            (500.0, 300.0),
            DVec3::ZERO,
            &camera,
            VIEWPORT,
            WORLD_AXES,
            Snap::default(),
            SnapWorld::default(),
        );
        let params = at(&mut g, (560.0, 260.0));
        let info = registry.get(kind.op_id(mode)).expect("registered");
        let names: Vec<&str> = info.schema.iter().map(|d| d.name).collect();
        for (key, _) in params.iter() {
            assert!(
                names.contains(&key),
                "{} produced `{key}`, which {} does not accept (it takes {names:?})",
                kind.label(),
                kind.op_id(mode)
            );
        }
    }
}

#[test]
fn precision_mode_during_a_rotation_does_not_jump_the_angle() {
    // Rotation accumulates increments, and precision mode changes the arithmetic
    // that produces them. If the bearing is not re-seeded across the transition,
    // the first increment afterwards is measured against a bearing computed under
    // the old scale — and the selection snaps.
    //
    // Only rotation is exposed to this: move and scale read the current
    // displacement, so they have nothing to carry across. The precision tests
    // for those two passed against a version that did not re-seed at all.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    // A sixth of a turn, anticlockwise on screen.
    let quarter = (400.0 + 100.0 * 0.5, 300.0 - 100.0 * 0.866);
    let before = angle(&at(&mut g, quarter));
    assert!(before > 0.5, "the sweep did not happen: {before}");

    g.set_precise(true, quarter);
    let after = angle(&at(&mut g, quarter));
    assert!(
        (before - after).abs() < 1e-9,
        "engaging precision moved the angle from {before} to {after}"
    );

    // And the sweep is now a tenth as fast.
    let further = (400.0 + 100.0 * 0.0, 300.0 - 100.0);
    let precise = angle(&at(&mut g, further)) - after;
    g.set_precise(false, further);
    let released = angle(&at(&mut g, further));
    assert!(
        (released - (after + precise)).abs() < 1e-9,
        "releasing precision moved the angle from {} to {released}",
        after + precise
    );
    // The same motion on a fresh, coarse gesture, for comparison.
    let mut coarse = Gesture::begin(
        Kind::Rotate,
        quarter,
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let unscaled = angle(&at(&mut coarse, further));

    // Nonzero, or the transition swallowed the sweep entirely — which a version
    // that re-seeded the bearing without re-accumulating did, and which the
    // upper bound alone happily accepted.
    assert!(
        precise.abs() > 1e-6,
        "the precise sweep was lost across the transition"
    );
    // Roughly a tenth, and deliberately not asserted as exactly a tenth.
    // Precision scales the *pointer displacement*, and the angle swept to a
    // point is not a linear function of that point — so a tenth of the motion
    // sweeps a bit under a tenth of the angle. That is the correct behaviour:
    // the selection follows the (slowed) pointer, which is what the modifier
    // promises. A test demanding exactly a tenth would be asserting a
    // relationship that does not hold and would have to be loosened until it
    // measured nothing.
    let ratio = precise / unscaled;
    assert!(
        (0.05..0.15).contains(&ratio),
        "a precise sweep moved {precise} of the coarse {unscaled}, a ratio of {ratio}"
    );
}

#[test]
fn reporting_the_same_pointer_position_twice_changes_nothing() {
    // Tracking has to be idempotent in position: the event loop may report the
    // same coordinates twice — a modifier press carrying the last known cursor,
    // a repaint, a platform that coalesces oddly — and none of that is motion.
    //
    // This is also the only thing that distinguishes re-seeding the bearing at a
    // precision transition from re-seeding *and* re-accumulating there. Without
    // the second half, the increment between the transition and the next motion
    // report is silently dropped; with an intervening same-position report it is
    // not, which is exactly why the earlier precision test could not see it.
    let camera = camera();
    let one = (450.0, 260.0);
    let two = (400.0, 200.0);

    let mut direct = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    direct.track(one);
    direct.set_precise(true, one);
    let straight = angle(&at(&mut direct, two));

    let mut doubled = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    doubled.track(one);
    doubled.set_precise(true, one);
    doubled.track(one);
    let repeated = angle(&at(&mut doubled, two));

    assert!(
        (straight - repeated).abs() < 1e-12,
        "a repeated position changed the angle: {straight} against {repeated}"
    );
}

// --- snapping ---------------------------------------------------------------

fn snapping() -> Snap {
    Snap {
        enabled: true,
        ..Snap::default()
    }
}

#[test]
fn a_snapped_move_lands_on_the_increment() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        snapping(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    // Whatever the pointer says, the answer is a multiple of the step.
    for x in [420.0, 437.0, 455.0, 512.0] {
        let t = translation(&at(&mut g, (x, 300.0)))[0];
        let steps = t / 0.1;
        assert!(
            (steps - steps.round()).abs() < 1e-9,
            "{t} is not a multiple of the snap step"
        );
    }
}

#[test]
fn ctrl_inverts_snapping_rather_than_enabling_it() {
    // §2.2's rule and the reason there is one modifier rather than two: someone
    // with snapping on wants Ctrl to let them off it, and someone with it off
    // wants Ctrl to turn it on.
    let camera = camera();
    let free_pointer = (473.0, 300.0);

    let mut on = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        snapping(),
        SnapWorld::default(),
    );
    on.cycle_axis(Axis::X);
    let snapped = translation(&at(&mut on, free_pointer))[0];
    on.set_snap_inverted(true);
    let released = translation(&at(&mut on, free_pointer))[0];
    assert_ne!(snapped, released, "Ctrl did not let the user off snapping");
    assert!(
        (released / 0.1 - (released / 0.1).round()).abs() > 1e-6,
        "the released value is still on the increment: {released}"
    );

    let mut off = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    off.cycle_axis(Axis::X);
    off.set_snap_inverted(true);
    let engaged = translation(&at(&mut off, free_pointer))[0];
    assert!(
        (engaged / 0.1 - (engaged / 0.1).round()).abs() < 1e-9,
        "Ctrl did not turn snapping on: {engaged}"
    );
}

#[test]
fn b_turns_snapping_on_for_the_rest_of_the_drag_and_off_again() {
    // Blender's `B` during a transform. A toggle, not a hold: `Ctrl` is the
    // hold. Someone laying twenty panels on the grid presses `B` once per
    // drag rather than holding a key through each.
    let camera = camera();
    let free_pointer = (473.0, 300.0);
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    let on_increment = |t: f64| (t / 0.1 - (t / 0.1).round()).abs() < 1e-9;

    let free = translation(&at(&mut g, free_pointer))[0];
    assert!(
        !on_increment(free),
        "the fixture pointer must land off the grid: {free}"
    );

    // With the header on increment, B snaps to geometry instead (see
    // `b_snaps_the_move_to_the_vertex_under_the_pointer`); with nothing under
    // the pointer the move stays free.
    g.toggle_snap();
    let snapped = translation(&at(&mut g, free_pointer))[0];
    assert_eq!(snapped, free, "nothing is near, so nothing is caught");
    assert!(
        g.status().contains("snap"),
        "the header does not say snapping is on: {}",
        g.status()
    );

    g.toggle_snap();
    let released = translation(&at(&mut g, free_pointer))[0];
    assert_eq!(released, free, "a second B did not turn snapping off again");
    assert!(!g.status().contains("snap"));
}

#[test]
fn ctrl_still_inverts_after_b_has_turned_snapping_on() {
    // The two compose the way the header states them: B sets the drag's
    // setting, Ctrl inverts whatever the setting is.
    let camera = camera();
    let free_pointer = (473.0, 300.0);
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    g.toggle_snap();
    g.set_snap_inverted(true);
    let t = translation(&at(&mut g, free_pointer))[0];
    assert!(
        (t / 0.1 - (t / 0.1).round()).abs() > 1e-6,
        "Ctrl should let the user off the snapping B turned on: {t}"
    );
}

#[test]
fn a_typed_value_is_never_snapped() {
    // Someone who wrote a number meant that number. Rounding it would be the
    // tool arguing with an instruction.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        snapping(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::X);
    for c in "0.137".chars() {
        g.type_character(c);
    }
    assert_eq!(translation(&at(&mut g, (900.0, 300.0)))[0], 0.137);
}

#[test]
fn a_snapped_rotation_lands_on_the_angle_increment() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        snapping(),
        SnapWorld::default(),
    );
    let step = std::f64::consts::FRAC_PI_2 / 6.0;
    for theta in [0.3f32, 0.9, 1.7] {
        let cursor = (400.0 + 100.0 * theta.cos(), 300.0 - 100.0 * theta.sin());
        let a = angle(&at(&mut g, cursor));
        let steps = a / step;
        assert!(
            (steps - steps.round()).abs() < 1e-9,
            "{a} radians is not a multiple of {step}"
        );
    }
}

#[test]
fn a_snapped_scale_lands_on_the_factor_increment() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Scale,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        snapping(),
        SnapWorld::default(),
    );
    for x in [530.0, 617.0, 683.0] {
        let f = scale(&at(&mut g, (x, 300.0)))[0];
        let steps = f / 0.1;
        assert!(
            (steps - steps.round()).abs() < 1e-9,
            "{f} is not a multiple of the snap step"
        );
    }
}

// --- proportional editing ---------------------------------------------------

#[test]
fn a_proportional_radius_rides_along_as_a_parameter() {
    // The gesture names the radius and the operator does the falloff, so this
    // module never has to know where the vertices are.
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    assert!(
        at(&mut g, (500.0, 300.0))
            .get("proportional_size")
            .is_none(),
        "off should say nothing rather than say zero"
    );

    g.set_proportional(Some(1.5));
    match at(&mut g, (500.0, 300.0)).get("proportional_size") {
        Some(PropertyValue::Float(r)) => assert_eq!(*r, 1.5),
        other => panic!("expected a radius, got {other:?}"),
    }
}

#[test]
fn the_status_line_reports_the_new_modes() {
    let camera = camera();
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera,
        VIEWPORT,
        WORLD_AXES,
        snapping(),
        SnapWorld::default(),
    );
    g.cycle_axis(Axis::Z);
    g.cycle_axis(Axis::Z);
    g.set_proportional(Some(0.75));
    g.track((450.0, 280.0));
    let status = g.status();
    assert!(
        status.contains("ZZ"),
        "a local constraint should show: {status}"
    );
    assert!(status.contains("snap"), "{status}");
    assert!(status.contains("falloff"), "{status}");
}

/// An extrude gesture whose faces face screen-up: world Z, in this camera.
fn extrude() -> Gesture {
    let mut g = Gesture::begin(
        Kind::Extrude,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera(),
        VIEWPORT,
        // What the application hands over: the faces' normal as local Z.
        [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z],
        Snap::default(),
        SnapWorld::default(),
    );
    g.axis = Some(Axis::Z);
    g.orientation = Orientation::Local;
    g
}

fn offset(params: &kreate_ops::PropertyMap) -> f64 {
    match params.get("offset") {
        Some(PropertyValue::Float(f)) => *f,
        other => panic!("expected an offset, got {other:?}"),
    }
}

#[test]
fn an_extrude_follows_the_pointer_along_the_normal_and_only_along_it() {
    // Blender's E: the new faces come out under the pointer, along their
    // normal. Starting at nothing, so pressing E moves nothing.
    let mut g = extrude();
    assert_eq!(offset(&at(&mut g, (400.0, 300.0))), 0.0);
    let up = offset(&at(&mut g, (400.0, 200.0)));
    assert!(
        up > 0.5,
        "dragging towards the normal did not pull the faces out: {up}"
    );
    let down = offset(&at(&mut g, (400.0, 400.0)));
    assert!(down < -0.5, "dragging away did not push them in: {down}");
    let across = offset(&at(&mut g, (500.0, 300.0)));
    assert!(
        across.abs() < 1e-6,
        "a drag across the normal moved along it: {across}"
    );
    // Axis keys do not take it off the normal.
    g.cycle_axis(Axis::X);
    assert_eq!(g.axis, Some(Axis::Z));
    assert_eq!(g.orientation, Orientation::Local);
}

#[test]
fn a_typed_extrude_distance_is_exact() {
    let mut g = extrude();
    for c in "0.25".chars() {
        assert!(g.type_character(c));
    }
    assert_eq!(offset(&at(&mut g, (430.0, 170.0))), 0.25);
}

/// A gesture of `kind` begun 200 pixels right of the pivot, which is at the
/// middle of the screen.
fn pulled(kind: Kind) -> Gesture {
    Gesture::begin(
        kind,
        (600.0, 300.0),
        DVec3::ZERO,
        &camera(),
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    )
}

fn float(params: &kreate_ops::PropertyMap, name: &str) -> f64 {
    match params.get(name) {
        Some(PropertyValue::Float(f)) => *f,
        other => panic!("expected {name}, got {other:?}"),
    }
}

#[test]
fn an_inset_thickens_as_the_pointer_moves_in_and_starts_at_nothing() {
    // Blender's I: nothing at the press, thicker towards the middle.
    let mut g = pulled(Kind::Inset);
    assert_eq!(float(&at(&mut g, (600.0, 300.0)), "thickness"), 0.0);
    let inward = float(&at(&mut g, (500.0, 300.0)), "thickness");
    assert!(inward > 0.1, "moving in did not inset: {inward}");
    let further = float(&at(&mut g, (450.0, 300.0)), "thickness");
    assert!(further > inward);
    // Out past where it started is nothing, not negative.
    assert_eq!(float(&at(&mut g, (700.0, 300.0)), "thickness"), 0.0);
    // Axis keys mean nothing to it.
    g.cycle_axis(Axis::X);
    assert_eq!(g.axis, None);
}

#[test]
fn a_bevel_widens_as_the_pointer_moves_out_and_never_asks_for_nothing() {
    // Blender's Ctrl+B. The start is not zero — the operator declines a bevel
    // of no width, which would refuse the gesture before it began.
    let mut g = pulled(Kind::Bevel);
    let start = float(&at(&mut g, (600.0, 300.0)), "width");
    assert!(start > 0.0 && start < 1e-3, "{start}");
    let out = float(&at(&mut g, (700.0, 300.0)), "width");
    assert!(out > 0.1, "moving out did not widen: {out}");
    assert_eq!(float(&at(&mut g, (500.0, 300.0)), "width"), start);
    for c in "0.2".chars() {
        g.type_character(c);
    }
    assert_eq!(float(&at(&mut g, (650.0, 300.0)), "width"), 0.2);
}

#[test]
fn a_loop_cut_slides_along_its_edge_and_the_wheel_or_a_number_sets_how_many() {
    let mut g = pulled(Kind::LoopCut);
    // An edge along world X, 2 units: screen right.
    g.set_slide_edge(DVec3::new(-1.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0));
    let cuts = |p: &kreate_ops::PropertyMap| match p.get("cuts") {
        Some(PropertyValue::Int(n)) => *n,
        other => panic!("expected cuts, got {other:?}"),
    };
    let p = at(&mut g, (600.0, 300.0));
    assert_eq!((float(&p, "position"), cuts(&p)), (0.5, 1));
    let right = float(&at(&mut g, (630.0, 300.0)), "position");
    assert!(
        right > 0.5 && right < 1.0,
        "sliding towards the second end: {right}"
    );
    let left = float(&at(&mut g, (570.0, 300.0)), "position");
    assert!(
        (left - (1.0 - right)).abs() < 1e-6,
        "not symmetric: {left} {right}"
    );
    // Across the edge does not slide it; far past an end stops at the end.
    assert_eq!(float(&at(&mut g, (600.0, 200.0)), "position"), 0.5);
    assert_eq!(float(&at(&mut g, (2000.0, 300.0)), "position"), 1.0);
    // The wheel adds loops, never below one.
    g.add_cuts(2);
    assert_eq!(cuts(&at(&mut g, (600.0, 300.0))), 3);
    g.add_cuts(-10);
    assert_eq!(g.cuts, 1);
    // A typed number is how many.
    g.type_character('4');
    assert_eq!(cuts(&at(&mut g, (600.0, 300.0))), 4);
}

#[test]
fn a_loop_cut_begun_on_a_hovered_edge_names_it() {
    let mut g = pulled(Kind::LoopCut);
    g.edge = Some(7);
    g.cuts = 3;
    let p = at(&mut g, (600.0, 300.0));
    assert_eq!(p.get("edge"), Some(&PropertyValue::Int(7)));
    assert_eq!(p.get("cuts"), Some(&PropertyValue::Int(3)));
    // Without one, the operator's own default — the selection — stands.
    assert_eq!(
        at(&mut pulled(Kind::LoopCut), (600.0, 300.0)).get("edge"),
        None
    );
}

#[test]
fn ctrl_during_an_inset_holds_the_thickness_and_drives_the_depth() {
    let mut g = pulled(Kind::Inset);
    let thick = float(&at(&mut g, (500.0, 300.0)), "thickness");
    assert!(thick > 0.1);
    g.set_snap_inverted(true);
    // Out from the middle while held: the thickness stays, the face comes out.
    let p = at(&mut g, (560.0, 300.0));
    assert_eq!(float(&p, "thickness"), thick);
    let depth = float(&p, "depth");
    assert!(depth > 0.1, "{depth}");
    // Released: the depth stays where it was left.
    g.set_snap_inverted(false);
    assert!((float(&at(&mut g, (560.0, 300.0)), "depth") - depth).abs() < 1e-9);
}

/// `B` mid-drag, Blender's: the drag began with nothing to catch on, and from
/// the press on it lands on the vertex under the pointer.
#[test]
fn b_snaps_the_move_to_the_vertex_under_the_pointer() {
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera(),
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    let target = DVec3::new(2.0, 0.0, 1.0);
    let on_it = kreate::snap::project(target, &camera(), VIEWPORT).unwrap();
    let near = (on_it.0 + 3.0, on_it.1 - 2.0);
    let unsnapped = translation(&at(&mut g, near));
    assert!(
        (DVec3::from_array(unsnapped) - target).length() > 1e-3,
        "nothing to snap to yet"
    );

    g.toggle_snap();
    assert!(g.wants_world(), "increment snapping gathered no geometry");
    g.set_world(SnapWorld {
        verts: vec![DVec3::new(-3.0, 0.0, 0.0), target],
        ..SnapWorld::default()
    });
    assert!(!g.wants_world());
    let snapped = translation(&at(&mut g, near));
    assert!(
        (DVec3::from_array(snapped) - target).length() < 1e-9,
        "{snapped:?}"
    );
    let (point, element) = g.snap_point().expect("a marker to draw");
    assert_eq!(element, kreate::snap::Element::Vertex);
    assert!((point - target).length() < 1e-9);

    // And again lets go.
    g.toggle_snap();
    let free = translation(&at(&mut g, near));
    assert!((DVec3::from_array(free) - DVec3::from_array(unsnapped)).length() < 1e-9);
    assert!(g.snap_point().is_none());
}

/// The point of the selection nearest the target is what lands on it —
/// Blender's snap base "Closest" — not the selection's middle, which would
/// sink a cube halfway into what it was put on.
#[test]
fn the_nearest_moving_point_lands_on_the_target() {
    let mut g = Gesture::begin(
        Kind::Move,
        (400.0, 300.0),
        DVec3::ZERO,
        &camera(),
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld::default(),
    );
    g.set_sources(vec![DVec3::new(0.5, 0.0, 0.5), DVec3::new(-0.2, 0.0, -0.9)]);
    g.toggle_snap();
    let target = DVec3::new(2.0, 0.0, 1.0);
    g.set_world(SnapWorld {
        verts: vec![target],
        ..SnapWorld::default()
    });
    let on_it = kreate::snap::project(target, &camera(), VIEWPORT).unwrap();
    let moved = translation(&at(&mut g, on_it));
    assert!(
        (DVec3::from_array(moved) - DVec3::new(1.5, 0.0, 0.5)).length() < 1e-9,
        "{moved:?}"
    );
}

/// A rotation has no destination, so `B` there means its increments, as
/// before, and nothing is marked.
#[test]
fn a_rotation_is_not_snapped_to_elements() {
    let mut g = Gesture::begin(
        Kind::Rotate,
        (500.0, 300.0),
        DVec3::ZERO,
        &camera(),
        VIEWPORT,
        WORLD_AXES,
        Snap::default(),
        SnapWorld {
            verts: vec![DVec3::new(1.0, 0.0, 0.0)],
            ..SnapWorld::default()
        },
    );
    g.toggle_snap();
    at(
        &mut g,
        kreate::snap::project(DVec3::new(1.0, 0.0, 0.0), &camera(), VIEWPORT).unwrap(),
    );
    assert!(g.snap_point().is_none());
}
