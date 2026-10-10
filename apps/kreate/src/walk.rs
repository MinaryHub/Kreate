//! Walking through the scene, as Blender's Walk Navigation does: the keys
//! move, the mouse turns the head.
//!
//! `W` `A` `S` `D` (or the arrows) move over the ground — the view's heading
//! with its tilt taken out, so looking down at a model and walking forward
//! does not dive into the floor — and `E` and `Q` rise and sink. `Shift`
//! runs, `Alt` creeps, and the wheel sets the pace. The application owns the
//! window: it hides and holds the pointer, feeds the mouse's motion to
//! [`look`], the held keys to [`Walk::set`], and the time since the last
//! frame to [`Walk::step`]. A click or `Enter` keeps the view; a right click
//! or `Esc` goes back to [`Walk::before`].

use glam::{DQuat, DVec3};
use kreate_render::{Camera, Projection};

/// Radians the head turns for a point of mouse movement.
pub const SENSITIVITY: f64 = 0.0025;
/// Metres a second at the start: Blender's walking pace.
pub const PACE: f64 = 2.5;
/// How much faster `Shift` is, and how much slower `Alt`.
pub const RUN: f64 = 4.0;
pub const CREEP: f64 = 0.2;

/// A way to go, held on a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Forward,
    Back,
    Left,
    Right,
    Up,
    Down,
}

impl Move {
    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A walk under way.
#[derive(Clone, Debug)]
pub struct Walk {
    /// The view it started from, which cancelling returns to.
    pub before: Camera,
    /// Metres a second, before `Shift` or `Alt`.
    pub pace: f64,
    held: u8,
}

impl Walk {
    /// Starts from `camera`, which it readies for walking: a perspective
    /// view whose up points skywards. A top view, or one rolled over, is
    /// stood up facing where it looked, tilted down a little.
    pub fn begin(camera: &mut Camera) -> Self {
        let before = *camera;
        camera.projection = Projection::Perspective;
        if camera.up().z < 0.2 {
            let eye = camera.eye();
            let ahead = camera.forward();
            let heading = DVec3::new(ahead.x, ahead.y, 0.0);
            let heading = if heading.length() > 1e-6 {
                heading.normalize()
            } else {
                // Straight down: the way the top of the picture pointed.
                let up = camera.up();
                DVec3::new(up.x, up.y, 0.0)
                    .try_normalize()
                    .unwrap_or(DVec3::Y)
            };
            // Level, facing `heading`: the view looks along -Z, so turn -Z
            // onto the heading with Y up, then tip it down.
            let level = DQuat::from_mat3(&glam::DMat3::from_cols(
                heading.cross(DVec3::Z),
                DVec3::Z,
                -heading,
            ));
            camera.rotation = (level * DQuat::from_rotation_x(-0.35)).normalize();
            camera.pivot = eye - camera.rotation * (DVec3::Z * camera.distance);
        }
        Self {
            before,
            pace: PACE,
            held: 0,
        }
    }

    /// A key for `way` went down or came up.
    pub fn set(&mut self, way: Move, down: bool) {
        if down {
            self.held |= way.bit();
        } else {
            self.held &= !way.bit();
        }
    }

    pub fn is_held(&self, way: Move) -> bool {
        self.held & way.bit() != 0
    }

    /// Whether any key is moving the view, so the window keeps drawing.
    pub fn moving(&self) -> bool {
        self.held != 0
    }

    /// The wheel, `notches` up to quicken and down to slow.
    pub fn quicken(&mut self, notches: f64) {
        self.pace = (self.pace * 1.25f64.powf(notches)).clamp(0.01, 1000.0);
    }

    /// Moves `camera` for `seconds` of the keys held. Whether it moved.
    pub fn step(&self, camera: &mut Camera, seconds: f64, run: bool, creep: bool) -> bool {
        let axis = |plus: Move, minus: Move| {
            self.is_held(plus) as i32 as f64 - self.is_held(minus) as i32 as f64
        };
        let ahead = axis(Move::Forward, Move::Back);
        let side = axis(Move::Right, Move::Left);
        let rise = axis(Move::Up, Move::Down);
        if ahead == 0.0 && side == 0.0 && rise == 0.0 {
            return false;
        }
        // Over the ground: the heading without its tilt. Looking straight
        // down, the heading is where the top of the picture points.
        let flat = |v: DVec3| DVec3::new(v.x, v.y, 0.0).try_normalize();
        let forward = flat(camera.forward())
            .or_else(|| flat(camera.up()))
            .unwrap_or(DVec3::Y);
        let right = forward.cross(DVec3::Z);
        let way = (forward * ahead + right * side + DVec3::Z * rise).normalize_or_zero();
        let mut pace = self.pace;
        if run {
            pace *= RUN;
        }
        if creep {
            pace *= CREEP;
        }
        camera.pivot += way * pace * seconds;
        true
    }
}

/// Turns `camera`'s head by a mouse movement of `dx`, `dy` points: right
/// turns right, down looks down.
pub fn look(camera: &mut Camera, dx: f64, dy: f64) {
    camera.look(-dx * SENSITIVITY, -dy * SENSITIVITY);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: DVec3, b: DVec3) -> bool {
        a.distance(b) < 1e-9
    }

    #[test]
    fn forward_goes_over_the_ground_the_way_the_view_faces() {
        let mut camera = Camera::default();
        let mut walk = Walk::begin(&mut camera);
        let eye = camera.eye();
        let facing = DVec3::new(camera.forward().x, camera.forward().y, 0.0).normalize();
        walk.set(Move::Forward, true);
        assert!(walk.step(&mut camera, 2.0, false, false));
        let went = camera.eye() - eye;
        assert!(went.z.abs() < 1e-9, "dived: {went}");
        assert!(close(went, facing * PACE * 2.0), "{went}");
        // Up is up, whatever the view.
        walk.set(Move::Forward, false);
        walk.set(Move::Up, true);
        let from = camera.eye();
        walk.step(&mut camera, 1.0, true, false);
        assert!(close(camera.eye() - from, DVec3::Z * PACE * RUN));
        // And nothing held is nothing done.
        walk.set(Move::Up, false);
        assert!(!walk.step(&mut camera, 1.0, false, false));
    }

    #[test]
    fn looking_turns_the_head_and_not_the_place() {
        let mut camera = Camera::default();
        Walk::begin(&mut camera);
        let eye = camera.eye();
        let heading = camera.forward();
        look(&mut camera, 200.0, 0.0);
        assert!(close(camera.eye(), eye), "the eye moved");
        // Right is clockwise from above.
        assert!(heading.cross(camera.forward()).z < 0.0, "turned left");
        let before = camera.forward().z;
        look(&mut camera, 0.0, 100.0);
        assert!(camera.forward().z < before, "did not look down");
        // Never over the top, however far the mouse goes.
        look(&mut camera, 0.0, -100_000.0);
        assert!(camera.up().z > 0.0, "{}", camera.up());
        assert!(close(camera.eye(), eye));
    }

    #[test]
    fn a_top_view_is_stood_up_where_it_was() {
        let mut camera = Camera::default().looking_along(kreate_render::camera::ViewAxis::PosZ);
        camera.projection = Projection::Orthographic;
        let eye = camera.eye();
        let walk = Walk::begin(&mut camera);
        assert_eq!(walk.before.projection, Projection::Orthographic);
        assert_eq!(camera.projection, Projection::Perspective);
        assert!(close(camera.eye(), eye));
        assert!(camera.up().z > 0.5, "{}", camera.up());
        assert!(camera.forward().z < 0.0, "looks down a little");
    }

    #[test]
    fn the_wheel_sets_the_pace_within_reason() {
        let mut camera = Camera::default();
        let mut walk = Walk::begin(&mut camera);
        walk.quicken(1.0);
        assert!(walk.pace > PACE);
        walk.quicken(-1000.0);
        assert!(walk.pace >= 0.01);
    }
}
