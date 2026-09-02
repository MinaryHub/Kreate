//! The navigation gizmo: the axis widget in the viewport's corner.
//!
//! Six balls on the ends of three axes, drawn in the top-right. Dragging it
//! orbits; clicking a ball snaps to that axis view.
//!
//! # Why the hit-test is analytic
//!
//! The plan is explicit that gizmo picking is a CPU screen-space distance test
//! rather than a GPU id buffer. Three reasons, all of which apply here: it is
//! deterministic, it costs no readback stall on the frame the user clicked, and
//! it puts occlusion under direct control — a ball behind the origin must still
//! be clickable when nothing nearer covers it, which is a rule about intent
//! rather than about depth.
//!
//! # Layout is separated from drawing
//!
//! [`layout`] and [`hit`] are pure and take no painter, so "does clicking the
//! `+X` ball give a `+X` view" is a unit test rather than something only a human
//! can check. The drawing function is thin on purpose.

use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};
use kreate_math::{DQuat, DVec3};
use kreate_render::ViewAxis;

/// One ball, positioned and depth-sorted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    pub axis: ViewAxis,
    pub centre: Pos2,
    /// How near the viewer this ball is: larger is nearer. Used for draw order
    /// and to break a tie when two balls overlap under the cursor.
    pub depth: f32,
}

/// How big the widget is, as a fraction of the shorter side of the viewport,
/// clamped so it neither vanishes on a large screen nor dominates a small one.
pub fn radius_for(viewport: Rect) -> f32 {
    (viewport.width().min(viewport.height()) * 0.075).clamp(28.0, 52.0)
}

/// Where the widget sits: the top-right corner, inset by its own radius.
pub fn centre_for(viewport: Rect, radius: f32) -> Pos2 {
    let inset = radius + 12.0;
    Pos2::new(viewport.max.x - inset, viewport.min.y + inset)
}

/// Ball radius, as a fraction of the widget's.
pub fn ball_radius(radius: f32) -> f32 {
    (radius * 0.22).max(7.0)
}

/// Projects the six axes into the widget, nearest last.
///
/// `rotation` is the camera's, not its inverse: the widget shows the world's
/// axes as seen *from* the camera, so the world directions are taken into view
/// space by the inverse of the camera's rotation.
pub fn layout(rotation: DQuat, centre: Pos2, radius: f32) -> Vec<Ball> {
    let view = rotation.inverse();
    let mut balls: Vec<Ball> = ViewAxis::ALL
        .iter()
        .map(|&axis| {
            let v: DVec3 = view * axis.direction();
            Ball {
                axis,
                // Screen `y` grows downward, so the view-space `y` is negated.
                centre: centre + Vec2::new(v.x as f32, -v.y as f32) * radius,
                // The camera looks down `-Z`, so a larger `z` is nearer.
                depth: v.z as f32,
            }
        })
        .collect();
    // Painter's order: far first, so the near balls draw over them.
    balls.sort_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    balls
}

/// Which ball is under a point, if any.
///
/// Nearest-to-the-viewer wins a tie rather than nearest-to-the-cursor: when two
/// balls overlap, the one drawn on top is the one the user believes they are
/// clicking, and disagreeing with the picture is worse than being a pixel off.
pub fn hit(balls: &[Ball], pos: Pos2, ball_radius: f32) -> Option<ViewAxis> {
    let mut best: Option<&Ball> = None;
    for ball in balls {
        if (ball.centre - pos).length() > ball_radius {
            continue;
        }
        if best.is_none_or(|b| ball.depth > b.depth) {
            best = Some(ball);
        }
    }
    best.map(|b| b.axis)
}

/// Whether a point is inside the widget's disc, where a drag orbits.
pub fn contains(centre: Pos2, radius: f32, pos: Pos2) -> bool {
    (centre - pos).length() <= radius + 4.0
}

/// Axis colours, matching the grid's.
fn axis_colour(axis: ViewAxis, near: bool) -> Color32 {
    let base = match axis.axis_index() {
        0 => Color32::from_rgb(214, 74, 84),
        1 => Color32::from_rgb(106, 186, 84),
        _ => Color32::from_rgb(74, 128, 214),
    };
    if near {
        base
    } else {
        // The far end of each axis is hollow and dimmer. Drawing both ends
        // identically makes the widget ambiguous: there is no way to tell which
        // way round the model is from a symmetrical picture.
        Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 150)
    }
}

/// Draws the widget.
pub fn draw(
    painter: &Painter,
    centre: Pos2,
    radius: f32,
    balls: &[Ball],
    hovered: Option<ViewAxis>,
) {
    let r = ball_radius(radius);
    for ball in balls {
        let positive = ball.axis.is_positive();
        let colour = axis_colour(ball.axis, positive);
        // Only the positive end gets a spoke: two spokes per axis would draw a
        // line straight through the widget and make the near end hard to read.
        if positive {
            painter.line_segment(
                [centre, ball.centre],
                Stroke::new(2.0, axis_colour(ball.axis, true)),
            );
        }
        let is_hovered = hovered == Some(ball.axis);
        let size = if is_hovered { r * 1.25 } else { r };
        if positive {
            painter.circle_filled(ball.centre, size, colour);
            painter.text(
                ball.centre,
                Align2::CENTER_CENTER,
                ball.axis.label(),
                FontId::proportional(r * 1.05),
                Color32::from_rgb(20, 20, 22),
            );
        } else {
            painter.circle_filled(ball.centre, size, Color32::from_rgb(38, 38, 42));
            painter.circle_stroke(ball.centre, size, Stroke::new(1.5, colour));
        }
        if is_hovered {
            painter.circle_stroke(ball.centre, size + 2.0, Stroke::new(1.5, Color32::WHITE));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_render::Camera;

    fn widget() -> (Pos2, f32) {
        (Pos2::new(400.0, 100.0), 40.0)
    }

    #[test]
    fn the_six_balls_are_laid_out_around_the_centre() {
        let (centre, radius) = widget();
        let balls = layout(DQuat::IDENTITY, centre, radius);
        assert_eq!(balls.len(), 6);
        for ball in &balls {
            let offset = (ball.centre - centre).length();
            assert!(
                (offset - radius).abs() < 1e-3 || offset < 1e-3,
                "a ball is {offset} from the centre, not {radius}"
            );
        }
        // Opposite ends are opposite.
        let x = balls.iter().find(|b| b.axis == ViewAxis::PosX).unwrap();
        let nx = balls.iter().find(|b| b.axis == ViewAxis::NegX).unwrap();
        assert!(((x.centre - centre) + (nx.centre - centre)).length() < 1e-3);
    }

    #[test]
    fn the_widget_shows_the_world_from_the_camera() {
        // With the identity rotation the camera looks down `-Z`, so `+Z` points
        // at the viewer and its ball sits in the middle rather than off to a
        // side. Getting the inverse the wrong way round puts every axis in the
        // mirror position, which looks plausible until the model turns.
        let (centre, radius) = widget();
        let balls = layout(DQuat::IDENTITY, centre, radius);
        let z = balls.iter().find(|b| b.axis == ViewAxis::PosZ).unwrap();
        assert!((z.centre - centre).length() < 1e-3);
        assert!(z.depth > 0.9, "+Z should be facing the viewer");

        let x = balls.iter().find(|b| b.axis == ViewAxis::PosX).unwrap();
        assert!(x.centre.x > centre.x, "+X should be to the right");
        let y = balls.iter().find(|b| b.axis == ViewAxis::PosY).unwrap();
        assert!(y.centre.y < centre.y, "+Y should be up the screen");
    }

    #[test]
    fn balls_are_ordered_far_to_near() {
        let camera = Camera::default();
        let balls = layout(camera.rotation, widget().0, widget().1);
        for pair in balls.windows(2) {
            assert!(pair[0].depth <= pair[1].depth, "not sorted for painting");
        }
    }

    #[test]
    fn clicking_a_ball_names_its_axis() {
        // A three-quarter view, so that no two balls coincide. Under the
        // identity rotation the two `Z` balls sit on top of each other and the
        // near one correctly answers for both — which is the overlap rule, not
        // a counterexample to this one.
        let (centre, radius) = widget();
        let balls = layout(Camera::default().rotation, centre, radius);
        for ball in &balls {
            assert_eq!(
                hit(&balls, ball.centre, ball_radius(radius)),
                Some(ball.axis),
                "clicking {:?} at {:?}",
                ball.axis,
                ball.centre
            );
        }
    }

    #[test]
    fn clicking_between_the_balls_hits_nothing() {
        let (centre, radius) = widget();
        let balls = layout(DQuat::IDENTITY, centre, radius);
        // Well outside the widget.
        assert_eq!(
            hit(
                &balls,
                centre + Vec2::new(radius * 3.0, 0.0),
                ball_radius(radius)
            ),
            None
        );
    }

    #[test]
    fn the_nearer_of_two_overlapping_balls_wins() {
        // The one drawn on top is the one the user believes they are clicking.
        // Disagreeing with the picture is worse than being a pixel off.
        let (centre, radius) = widget();
        // Looking straight down `-Z` puts `+Z` and `-Z` on top of each other.
        let balls = layout(DQuat::IDENTITY, centre, radius);
        let picked = hit(&balls, centre, ball_radius(radius)).unwrap();
        assert_eq!(picked, ViewAxis::PosZ, "the near end, not the far one");
    }

    #[test]
    fn the_disc_is_where_a_drag_orbits() {
        let (centre, radius) = widget();
        assert!(contains(centre, radius, centre));
        assert!(contains(centre, radius, centre + Vec2::new(radius, 0.0)));
        assert!(!contains(
            centre,
            radius,
            centre + Vec2::new(radius * 2.0, 0.0)
        ));
    }

    #[test]
    fn the_widget_stays_inside_the_viewport() {
        for size in [(320.0, 240.0), (1920.0, 1080.0), (600.0, 2000.0)] {
            let rect = Rect::from_min_size(Pos2::new(120.0, 30.0), Vec2::new(size.0, size.1));
            let radius = radius_for(rect);
            let centre = centre_for(rect, radius);
            assert!(
                centre.x + radius <= rect.max.x && centre.y - radius >= rect.min.y,
                "the widget escaped a {size:?} viewport"
            );
            assert!((28.0..=52.0).contains(&radius), "radius {radius}");
        }
    }

    #[test]
    fn snapping_to_an_axis_looks_along_it() {
        // The other half of the click: the camera the widget asks for.
        let camera = Camera::default();
        for axis in ViewAxis::ALL {
            let snapped = camera.looking_along(axis);
            let forward = snapped.forward();
            assert!(
                (forward + axis.direction()).length() < 1e-9,
                "{axis:?}: looking {forward:?}, wanted {:?}",
                -axis.direction()
            );
            assert_eq!(snapped.pivot, camera.pivot, "the subject must not move");
            assert_eq!(snapped.distance, camera.distance);
        }
    }

    #[test]
    fn a_top_view_is_not_rolled_to_an_arbitrary_angle() {
        // The gimbal collapse: using world `+Z` as the up reference for every
        // axis leaves the top and bottom views degenerate, and they come out
        // rolled by whatever the arithmetic happened to produce.
        let camera = Camera::default();
        for axis in [ViewAxis::PosZ, ViewAxis::NegZ] {
            let snapped = camera.looking_along(axis);
            let up = snapped.up();
            assert!(
                up.dot(DVec3::Y).abs() > 0.99,
                "{axis:?}: up is {up:?}, which is not the world +Y a top view wants"
            );
        }
        for axis in [
            ViewAxis::PosX,
            ViewAxis::NegX,
            ViewAxis::PosY,
            ViewAxis::NegY,
        ] {
            let up = camera.looking_along(axis).up();
            assert!(
                up.dot(DVec3::Z) > 0.99,
                "{axis:?}: a side view's up should be world +Z, got {up:?}"
            );
        }
    }

    #[test]
    fn every_axis_view_is_a_valid_orientation() {
        let camera = Camera::default();
        for axis in ViewAxis::ALL {
            let s = camera.looking_along(axis);
            assert!(
                (s.rotation.length() - 1.0).abs() < 1e-9,
                "{axis:?} unnormalised"
            );
            let (f, r, u) = (s.forward(), s.right(), s.up());
            assert!(f.dot(r).abs() < 1e-9 && f.dot(u).abs() < 1e-9 && r.dot(u).abs() < 1e-9);
            assert!(
                r.cross(u).dot(-f) > 0.99,
                "{axis:?} is left-handed, which mirrors everything"
            );
        }
    }

    #[test]
    fn clicking_the_ball_that_faces_you_is_a_no_op() {
        // The widget must agree with itself: after snapping to an axis, that
        // axis's ball is the one facing the viewer.
        let camera = Camera::default();
        for axis in ViewAxis::ALL {
            let snapped = camera.looking_along(axis);
            let (centre, radius) = widget();
            let balls = layout(snapped.rotation, centre, radius);
            let facing = balls
                .iter()
                .max_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap())
                .unwrap();
            assert_eq!(facing.axis, axis, "after snapping to {axis:?}");
        }
    }
}
