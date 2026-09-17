//! Putting a world point on the screen, for the interface's half of the lamp
//! and camera glyphs.
//!
//! The glyphs themselves are drawn by the renderer, in its own pass against
//! its depth, so a lamp behind a wall is behind the wall — see
//! `kreate_render::markers`. What is left to the interface is the click: the
//! renderer's id buffer has never heard of a lamp, so a click on one has to be
//! answered here, by projecting the lamp the same way the renderer did and
//! asking whether the pointer is on its body.

use egui::{Pos2, Rect};
use kreate_math::{DMat4, DQuat, DVec3};

/// What the interface knows about the camera, enough to put a world point on
/// the screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projector {
    pub eye: DVec3,
    pub rotation: DQuat,
    /// `Camera::projection_matrix(aspect)`, f64 so a far-off object projects
    /// where the renderer put it rather than a pixel away.
    pub projection: DMat4,
    /// The rectangle the picture fills, in interface points.
    pub rect: Rect,
}

impl Projector {
    /// Where a world point lands on screen, or `None` behind the camera.
    pub fn project(&self, world: DVec3) -> Option<Pos2> {
        // Camera-relative first, the way the renderer does it: the subtraction
        // in f64 is what keeps a distant object's glyph on its object.
        let view = self.rotation.inverse() * (world - self.eye);
        let clip = self.projection * view.extend(1.0);
        if clip.w <= 1e-9 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Pos2::new(
            self.rect.min.x + (ndc.x as f32 * 0.5 + 0.5) * self.rect.width(),
            self.rect.min.y + (0.5 - ndc.y as f32 * 0.5) * self.rect.height(),
        ))
    }
}

/// The radius of a glyph's body on screen, in points: the same whatever the
/// distance, the way a widget is, so a far lamp is still findable. Kept equal
/// to what the renderer draws — see `kreate::glyphs`.
const BODY: f32 = 7.0;

/// How close a click has to land to a glyph's body to be a click on it.
///
/// A little larger than the body, because a seven-point circle is a small
/// thing to hit with a pointer and a miss by two points reads as "lamps
/// cannot be selected" rather than as a miss.
pub const HIT_RADIUS: f32 = BODY + 4.0;

/// Whether a click at `at` lands on the glyph of an object at `world`.
///
/// The same projection the glyph was drawn with, which is what makes the
/// answer match the picture: the body is drawn at the object's projected
/// position at a constant screen size, so that is what is tested. The ground
/// line, the cone and the camera's frame are not part of the target — they
/// are long and thin, and a click on them is more often a click on whatever
/// is behind.
pub fn hits(projector: &Projector, world: DVec3, at: Pos2) -> bool {
    match projector.project(world) {
        Some(centre) => projector.rect.contains(centre) && centre.distance(at) <= HIT_RADIUS,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projector() -> Projector {
        // A camera at +Z looking down -Z at the origin, 90° vertical fov, over
        // a 200×100 rectangle.
        let aspect = 2.0;
        let f = 1.0 / (std::f64::consts::FRAC_PI_4).tan();
        // A plain perspective matrix, +Y up, w = -z.
        let projection = DMat4::from_cols_array(&[
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, -1.0, -1.0,
            0.0, 0.0, -0.1, 0.0,
        ]);
        Projector {
            eye: DVec3::new(0.0, 0.0, 5.0),
            rotation: DQuat::IDENTITY,
            projection,
            rect: Rect::from_min_size(Pos2::new(10.0, 20.0), egui::Vec2::new(200.0, 100.0)),
        }
    }

    #[test]
    fn the_point_the_camera_looks_at_lands_in_the_middle() {
        let p = projector();
        let at = p.project(DVec3::ZERO).expect("in front");
        assert!((at.x - 110.0).abs() < 0.01 && (at.y - 70.0).abs() < 0.01, "{at:?}");
    }

    #[test]
    fn up_in_the_world_is_up_on_the_screen() {
        // Screen y grows downward; a point above the pivot must land above
        // the centre, not below it — the sign the naive mapping gets wrong.
        let p = projector();
        let centre = p.project(DVec3::ZERO).unwrap();
        let above = p.project(DVec3::new(0.0, 1.0, 0.0)).unwrap();
        assert!(above.y < centre.y, "above {above:?} vs centre {centre:?}");
        let right = p.project(DVec3::new(1.0, 0.0, 0.0)).unwrap();
        assert!(right.x > centre.x);
    }

    #[test]
    fn a_point_behind_the_camera_is_not_drawn() {
        let p = projector();
        assert_eq!(p.project(DVec3::new(0.0, 0.0, 6.0)), None);
    }

    #[test]
    fn a_click_on_the_body_hits_and_one_beside_it_does_not() {
        let p = projector();
        let world = DVec3::ZERO;
        let centre = p.project(world).unwrap();
        assert!(hits(&p, world, centre));
        assert!(hits(&p, world, centre + egui::vec2(HIT_RADIUS - 0.5, 0.0)));
        assert!(!hits(&p, world, centre + egui::vec2(HIT_RADIUS + 1.0, 0.0)));
        // Behind the camera there is no glyph to hit.
        assert!(!hits(&p, DVec3::new(0.0, 0.0, 6.0), centre));
    }
}
