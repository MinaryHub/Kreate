//! Camera easing: smoothing, and timed moves.
//!
//! Two separate things that both sit between what navigation asks for and what
//! is drawn.
//!
//! **Smoothing** takes the edge off orbit, pan and zoom. Subtle by design — a
//! time constant of about 70 ms, which reads as weight rather than as lag — and
//! switchable off, because a professional often wants strict one-to-one and will
//! notice anything else immediately.
//!
//! **Timed moves** are Frame Selected and the navigation gizmo's axis snaps.
//! These must never teleport: a camera that jumps costs the user their sense of
//! where they are, and they then have to orbit around to rebuild it. A quarter
//! of a second of smoothstep is the whole difference.
//!
//! # First-order, not a spring
//!
//! The plan says critically-damped spring, and this is exponential smoothing —
//! first order, no velocity state. The deviation is deliberate and worth naming:
//! a second-order spring needs a velocity per component *including the
//! quaternion*, where "velocity" has no single agreed meaning; the extra
//! smoothness it buys is at the very start of a motion and is not perceptible at
//! 70 ms; and both are overshoot-free, which is the property that actually
//! matters. What is not negotiable is frame-rate independence, and
//! `1 - exp(-dt/τ)` gives that exactly rather than approximately.

use kreate_math::{DQuat, DVec3};

use crate::camera::Camera;

/// A timed move from one framing to another.
#[derive(Clone, Copy, Debug)]
struct Transition {
    from: Camera,
    to: Camera,
    elapsed: f64,
    duration: f64,
}

/// Where the camera is going, and how it gets there.
#[derive(Clone, Copy, Debug)]
pub struct CameraMotion {
    /// What navigation writes to. The drawn camera follows it.
    pub target: Camera,
    /// Time constant in seconds. Zero disables smoothing entirely.
    pub smoothing: f64,
    transition: Option<Transition>,
}

/// The plan's figure, and about as much as reads as weight rather than as lag.
pub const DEFAULT_SMOOTHING: f64 = 0.07;

/// Long enough to follow, short enough not to wait for.
pub const DEFAULT_TRANSITION: f64 = 0.25;

/// Below this, a camera has arrived. Chosen relative to the orbit distance so
/// the same threshold works at a millimetre and at a kilometre.
const SETTLED: f64 = 1e-4;

impl CameraMotion {
    pub fn new(camera: Camera) -> Self {
        Self {
            target: camera,
            smoothing: DEFAULT_SMOOTHING,
            transition: None,
        }
    }

    /// Starts a timed move, overriding any smoothing for its duration.
    ///
    /// Also moves the target, so navigation during the move blends into it
    /// rather than fighting it.
    pub fn animate_to(&mut self, from: Camera, to: Camera, duration: f64) {
        self.target = to;
        self.transition = Some(Transition {
            from,
            to,
            elapsed: 0.0,
            duration: duration.max(1e-3),
        });
    }

    /// Cancels a timed move, leaving the camera where it is.
    ///
    /// What a user's own navigation input does: taking hold of the view must
    /// stop it drifting somewhere on its own.
    pub fn interrupt(&mut self, current: &Camera) {
        if self.transition.take().is_some() {
            self.target = *current;
        }
    }

    pub fn is_animating(&self) -> bool {
        self.transition.is_some()
    }

    /// Advances `current` toward the target. Returns `true` while still moving.
    ///
    /// The return value drives redrawing: an idle window must use no CPU and no
    /// GPU, so "is anything still in flight" has to be answerable rather than
    /// assumed.
    pub fn advance(&mut self, current: &mut Camera, dt: f64) -> bool {
        if let Some(t) = &mut self.transition {
            t.elapsed += dt;
            let progress = (t.elapsed / t.duration).clamp(0.0, 1.0);
            // Smoothstep: zero velocity at both ends, so the move neither jerks
            // into motion nor stops dead.
            let eased = progress * progress * (3.0 - 2.0 * progress);
            *current = blend(&t.from, &t.to, eased);
            if progress >= 1.0 {
                *current = t.to;
                self.transition = None;
                return false;
            }
            return true;
        }

        if self.smoothing <= 0.0 {
            let moved = !settled(current, &self.target);
            *current = self.target;
            return moved;
        }
        if settled(current, &self.target) {
            // Snapped rather than approached forever: an exponential never
            // exactly arrives, and a camera that is always *nearly* still keeps
            // the window redrawing at full rate on battery.
            *current = self.target;
            return false;
        }
        let alpha = 1.0 - (-dt / self.smoothing).exp();
        *current = blend(current, &self.target, alpha);
        true
    }
}

/// Interpolates between two framings.
///
/// Distance moves *geometrically*, matching the dolly: halving then halving
/// again is what a zoom does, so a linear midpoint between 1 m and 100 m would
/// spend most of the move far away and then rush in.
fn blend(a: &Camera, b: &Camera, t: f64) -> Camera {
    let mut out = *a;
    out.pivot = a.pivot.lerp(b.pivot, t);
    out.rotation = slerp(a.rotation, b.rotation, t);
    out.distance = if a.distance > 0.0 && b.distance > 0.0 {
        a.distance * (b.distance / a.distance).powf(t)
    } else {
        a.distance + (b.distance - a.distance) * t
    };
    out.fov_y = a.fov_y + (b.fov_y - a.fov_y) * t;
    // Not interpolable: switching projection or orbit style mid-move should
    // take effect at once rather than passing through something in between.
    out.projection = b.projection;
    out.orbit_style = b.orbit_style;
    out
}

/// Shortest-arc slerp.
///
/// The sign fix matters: two quaternions describing nearly the same orientation
/// can differ in sign, and interpolating them naively sends the camera the long
/// way round — a full barrel roll where the user asked for a nudge.
fn slerp(a: DQuat, b: DQuat, t: f64) -> DQuat {
    let b = if a.dot(b) < 0.0 { -b } else { b };
    a.slerp(b, t).normalize()
}

/// Whether two framings are close enough to call equal.
fn settled(a: &Camera, b: &Camera) -> bool {
    let scale = a.distance.max(b.distance).max(1e-6);
    (a.pivot - b.pivot).length() < SETTLED * scale
        && (a.distance - b.distance).abs() < SETTLED * scale
        && a.rotation.dot(b.rotation).abs() > 1.0 - SETTLED
}

/// A framing that fits a bounding sphere, for the caller to animate to.
pub fn framing_for(from: &Camera, centre: DVec3, radius: f64, aspect: f64) -> Camera {
    let mut out = *from;
    out.frame_sphere(centre, radius, aspect);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn other() -> Camera {
        let mut c = Camera {
            pivot: DVec3::new(10.0, -4.0, 2.0),
            distance: 40.0,
            ..Camera::default()
        };
        c.orbit(1.2, -0.4);
        c
    }

    #[test]
    fn with_smoothing_off_the_camera_is_exactly_the_target() {
        // The preference a professional reaches for first. "Almost immediate" is
        // not the same as immediate, and they will notice.
        let mut current = Camera::default();
        let mut motion = CameraMotion::new(current);
        motion.smoothing = 0.0;
        motion.target = other();

        assert!(motion.advance(&mut current, 1.0 / 60.0));
        assert_eq!(current.pivot, motion.target.pivot);
        assert_eq!(current.distance, motion.target.distance);
        assert!(
            !motion.advance(&mut current, 1.0 / 60.0),
            "and then it rests"
        );
    }

    #[test]
    fn smoothing_converges_and_then_stops_asking_for_frames() {
        // An exponential never exactly arrives. Without a settle threshold the
        // window redraws forever at full rate, which on a laptop is a flat
        // battery for a camera that has visibly stopped.
        let mut current = Camera::default();
        let mut motion = CameraMotion::new(current);
        motion.target = other();

        let mut frames = 0;
        while motion.advance(&mut current, 1.0 / 60.0) {
            frames += 1;
            assert!(frames < 600, "never settled");
        }
        assert!(frames > 3, "it should take a few frames, not one");
        assert!(
            frames < 120,
            "two seconds to follow a nudge is lag: {frames}"
        );
        assert_eq!(current.pivot, motion.target.pivot, "it lands exactly");
    }

    #[test]
    fn smoothing_is_frame_rate_independent() {
        // The property that makes the setting mean the same thing on a 60 Hz
        // laptop and a 240 Hz monitor. Half the step at twice the rate must
        // reach the same place.
        let target = other();
        let run = |steps: u32, dt: f64| {
            let mut current = Camera::default();
            let mut motion = CameraMotion::new(current);
            motion.target = target;
            for _ in 0..steps {
                motion.advance(&mut current, dt);
            }
            current
        };
        let slow = run(6, 1.0 / 60.0);
        let fast = run(24, 1.0 / 240.0);
        assert!(
            (slow.distance - fast.distance).abs() / slow.distance < 0.02,
            "{} against {}",
            slow.distance,
            fast.distance
        );
        assert!((slow.pivot - fast.pivot).length() / slow.distance < 0.02);
    }

    #[test]
    fn smoothing_never_overshoots() {
        // The one property that matters more than the exact curve. An overshoot
        // reads as the view bouncing, which nobody reads as polish.
        let mut current = Camera::default();
        let start = current.distance;
        let mut motion = CameraMotion::new(current);
        motion.target.distance = start * 4.0;

        while motion.advance(&mut current, 1.0 / 60.0) {
            assert!(
                current.distance >= start - 1e-9 && current.distance <= start * 4.0 + 1e-9,
                "distance left the interval: {}",
                current.distance
            );
        }
    }

    #[test]
    fn a_timed_move_takes_its_duration_and_lands_exactly() {
        let mut current = Camera::default();
        let mut motion = CameraMotion::new(current);
        let destination = other();
        motion.animate_to(current, destination, 0.25);

        let dt = 1.0 / 60.0;
        let mut elapsed = 0.0;
        while motion.advance(&mut current, dt) {
            elapsed += dt;
            assert!(elapsed < 1.0, "the move ran long");
        }
        assert!(elapsed >= 0.2, "the move ran short: {elapsed}");
        assert_eq!(current.pivot, destination.pivot);
        assert_eq!(current.distance, destination.distance);
        assert!(!motion.is_animating());
    }

    #[test]
    fn a_timed_move_eases_at_both_ends() {
        // Smoothstep, stated as what it is for: the move must not jerk into
        // motion or stop dead, because both read as a glitch rather than as a
        // camera.
        let mut current = Camera::default();
        let mut motion = CameraMotion::new(current);
        let destination = other();
        motion.animate_to(current, destination, 0.3);

        let dt = 1.0 / 120.0;
        let mut speeds = Vec::new();
        let mut previous = current.pivot;
        while motion.advance(&mut current, dt) {
            speeds.push((current.pivot - previous).length());
            previous = current.pivot;
        }
        assert!(speeds.len() > 10);
        let first = speeds[0];
        let middle = speeds[speeds.len() / 2];
        let last = *speeds.last().unwrap();
        assert!(first < middle * 0.5, "it started abruptly");
        assert!(last < middle * 0.5, "it stopped dead");
    }

    #[test]
    fn distance_moves_geometrically() {
        // A linear midpoint between one metre and a hundred would spend most of
        // the move far away and then rush in. Zoom is geometric, and so is this.
        let mut current = Camera {
            distance: 1.0,
            ..Camera::default()
        };
        let mut destination = current;
        destination.distance = 100.0;

        let mut motion = CameraMotion::new(current);
        motion.animate_to(current, destination, 0.2);
        let dt = 0.1;
        motion.advance(&mut current, dt);

        assert!(
            current.distance < 25.0,
            "halfway should be near the geometric mean of 10, not 50: {}",
            current.distance
        );
    }

    #[test]
    fn a_move_takes_the_short_way_round() {
        // Two quaternions for nearly the same orientation can differ in sign.
        // Interpolating them naively rolls the camera all the way over — a
        // barrel roll where the user asked for a nudge.
        let mut current = Camera::default();
        let mut destination = current;
        destination.orbit(0.2, 0.0);
        destination.rotation = -destination.rotation;

        let mut motion = CameraMotion::new(current);
        motion.animate_to(current, destination, 0.2);

        let mut travelled = 0.0;
        let mut previous = current.forward();
        while motion.advance(&mut current, 1.0 / 120.0) {
            travelled += previous.angle_between(current.forward());
            previous = current.forward();
        }
        assert!(
            travelled < 0.5,
            "the camera turned {travelled} radians to cover 0.2"
        );
    }

    #[test]
    fn taking_hold_of_the_view_stops_a_move() {
        // A camera that keeps drifting after the user grabs it is a camera
        // fighting them.
        let mut current = Camera::default();
        let mut motion = CameraMotion::new(current);
        motion.animate_to(current, other(), 1.0);
        motion.advance(&mut current, 0.1);
        assert!(motion.is_animating());

        let caught = current;
        motion.interrupt(&current);
        assert!(!motion.is_animating());
        assert!(!motion.advance(&mut current, 1.0 / 60.0));
        assert_eq!(current.pivot, caught.pivot);
    }

    #[test]
    fn framing_animates_rather_than_teleporting() {
        // Frame Selected. The plan is explicit: never teleport, because the user
        // loses their sense of where they are and has to orbit to rebuild it.
        let mut current = Camera::default();
        let destination = framing_for(&current, DVec3::new(20.0, 0.0, 0.0), 3.0, 16.0 / 9.0);
        let mut motion = CameraMotion::new(current);
        motion.animate_to(current, destination, DEFAULT_TRANSITION);

        let mut frames = 0;
        while motion.advance(&mut current, 1.0 / 60.0) {
            frames += 1;
        }
        assert!(
            frames > 5,
            "a quarter second at 60 Hz is about fifteen frames"
        );
        assert!((current.pivot - DVec3::new(20.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn an_idle_camera_asks_for_no_frames_at_all() {
        // The on-demand rendering contract. An idle window must use no CPU and
        // no GPU; a motion layer that always claims to be moving would defeat it
        // entirely and nothing else would notice.
        let mut current = Camera::default();
        let mut motion = CameraMotion::new(current);
        for _ in 0..10 {
            assert!(!motion.advance(&mut current, 1.0 / 60.0));
        }
        motion.smoothing = 0.0;
        assert!(!motion.advance(&mut current, 1.0 / 60.0));
    }
}
