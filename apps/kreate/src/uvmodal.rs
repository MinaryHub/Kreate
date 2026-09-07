//! The UV editor's modal transform: `G`, `R`, `S` in two dimensions.
//!
//! The same shape as the viewport's gesture in `modal`, and deliberately not
//! the same code. A 2D transform has no camera, no orientation to cycle
//! through and nothing to snap to, and the parts of the 3D gesture that handle
//! those are most of it. What is shared is what matters: §2.2's rule that the
//! transform is always *original → transform → current* and never
//! incremental, and §1.7's rule that nothing reaches the undo stack until the
//! gesture confirms. Both of those live in `History`'s modal session, which
//! this drives.
//!
//! # The key starts it; the mouse does the work
//!
//! `G` does not move anything. It begins a gesture that follows the pointer
//! until the user confirms or cancels — the same contract the viewport's
//! transform has, because a user who has learned one has learned the other.

use kreate_ops::{PropertyMap, PropertyValue};

use crate::expr;

/// Which transform is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Move,
    Rotate,
    Scale,
}

impl Kind {
    /// The label the header shows, before translation.
    pub fn label(self) -> &'static str {
        match self {
            Kind::Move => "uv-move",
            Kind::Rotate => "uv-rotate",
            Kind::Scale => "uv-scale",
        }
    }
}

/// Which axis the transform is constrained to, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    U,
    V,
}

impl Axis {
    pub fn label(self) -> &'static str {
        match self {
            Axis::U => "U",
            Axis::V => "V",
        }
    }
}

/// A transform being dragged in the UV editor.
#[derive(Clone, Debug)]
pub struct UvGesture {
    kind: Kind,
    /// Where the pointer was when the gesture began, in UV units.
    origin: glam::Vec2,
    /// Where it is now, in UV units.
    cursor: glam::Vec2,
    /// The pivot: the selection's median, in UV units.
    pivot: glam::Vec2,
    /// The constrained axis, if any.
    axis: Option<Axis>,
    /// What the user has typed, if anything.
    numeric: String,
    /// Where precision mode was engaged, if it is.
    ///
    /// The point rather than a flag, because §2.2's precision is measured
    /// *from where the modifier was pressed* — engaging it must not make the
    /// selection jump.
    precise_from: Option<glam::Vec2>,
    /// How far the pointer has turned about the pivot since the gesture began.
    ///
    /// Accumulated rather than differenced from the origin, because the angle
    /// between two directions is only ever the short way round: a drag past
    /// half a turn would otherwise flip sign. The same fix the 3D rotation
    /// needed.
    swept: f64,
    /// The bearing the sweep was last measured from.
    bearing: f64,
}

impl UvGesture {
    /// Begins a gesture at `cursor`, turning about `pivot`.
    pub fn begin(kind: Kind, cursor: glam::Vec2, pivot: glam::Vec2) -> Self {
        Self {
            kind,
            origin: cursor,
            cursor,
            pivot,
            axis: None,
            numeric: String::new(),
            precise_from: None,
            swept: 0.0,
            bearing: bearing_of(cursor - pivot),
        }
    }

    /// Follows the pointer.
    pub fn track(&mut self, cursor: glam::Vec2) {
        if self.kind == Kind::Rotate {
            let now = bearing_of(cursor - self.pivot);
            // Only the step between frames is normalised, and it is then
            // added. That is what lets a drag exceed half a turn.
            self.swept += normalise(now - self.bearing);
            self.bearing = now;
        }
        self.cursor = cursor;
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn axis(&self) -> Option<Axis> {
        self.axis
    }

    /// Constrains to an axis, or releases the constraint if it is already on
    /// that one.
    ///
    /// A toggle rather than a set, so the key that applied it takes it away —
    /// which is what the viewport's transform does, and the reason a user does
    /// not have to remember a separate "release" key.
    pub fn constrain(&mut self, axis: Axis) {
        self.axis = if self.axis == Some(axis) {
            None
        } else {
            Some(axis)
        };
    }

    /// Engages or releases precision mode.
    pub fn set_precise(&mut self, on: bool) {
        match (on, self.precise_from) {
            (true, None) => self.precise_from = Some(self.cursor),
            (false, Some(_)) => {
                // Fold what precision mode contributed into the origin, so
                // releasing the modifier does not undo it. Without this the
                // selection jumps back the moment Shift comes up.
                let raw = self.cursor - self.precise_from.unwrap_or(self.cursor);
                self.origin += raw - raw * PRECISION;
                self.precise_from = None;
            }
            _ => {}
        }
    }

    /// Accepts a character into the numeric buffer.
    ///
    /// Returns whether it was taken, so the caller can fall through to the
    /// keymap for anything that is not part of a number.
    pub fn type_char(&mut self, c: char) -> bool {
        if c.is_ascii_digit() || matches!(c, '-' | '.' | '/' | '*' | '+' | '(' | ')' | ' ') {
            self.numeric.push(c);
            true
        } else if matches!(c, 'm' | 'c' | 'd' | 'e' | 'g' | 'r' | 'a' | 'n' | '°')
            && !self.numeric.is_empty()
        {
            // Unit suffixes, but only once a number has been started —
            // otherwise `g` and `r` and `s` would be swallowed instead of
            // starting the next tool.
            self.numeric.push(c);
            true
        } else {
            false
        }
    }

    /// Removes the last typed character.
    pub fn backspace(&mut self) -> bool {
        self.numeric.pop().is_some()
    }

    /// The typed value, if the buffer holds a complete expression.
    fn typed(&self) -> Option<f64> {
        if self.numeric.is_empty() {
            return None;
        }
        let want = match self.kind {
            Kind::Rotate => expr::Unit::Angle,
            // UV space is dimensionless: an offset is a fraction of the map and
            // a scale is a ratio, so neither takes a length.
            Kind::Move | Kind::Scale => expr::Unit::Factor,
        };
        expr::eval(&self.numeric, want).ok()
    }

    /// The pointer's displacement from the origin, with precision applied.
    fn delta(&self) -> glam::Vec2 {
        match self.precise_from {
            None => self.cursor - self.origin,
            Some(from) => (from - self.origin) + (self.cursor - from) * PRECISION,
        }
    }

    /// The parameters this gesture currently asks for.
    ///
    /// A pure function of the gesture's state, which is what lets the operator
    /// be re-run from the restored original every frame instead of being
    /// nudged. It is also what the redo panel edits after the gesture ends.
    pub fn params(&self) -> PropertyMap {
        let mut p = PropertyMap::new();
        let typed = self.typed();
        match self.kind {
            Kind::Move => {
                let d = self.delta();
                let (u, v) = match (self.axis, typed) {
                    (Some(Axis::U), Some(t)) => (t, 0.0),
                    (Some(Axis::V), Some(t)) => (0.0, t),
                    (None, Some(t)) => (t, 0.0),
                    (Some(Axis::U), None) => (d.x as f64, 0.0),
                    (Some(Axis::V), None) => (0.0, d.y as f64),
                    (None, None) => (d.x as f64, d.y as f64),
                };
                p.set("offset_u", PropertyValue::Float(u));
                p.set("offset_v", PropertyValue::Float(v));
            }
            Kind::Rotate => {
                let radians = typed.unwrap_or(self.swept);
                p.set("rotation", PropertyValue::Float(radians));
            }
            Kind::Scale => {
                let factor = typed.unwrap_or_else(|| self.distance_ratio());
                let (u, v) = match self.axis {
                    Some(Axis::U) => (factor, 1.0),
                    Some(Axis::V) => (1.0, factor),
                    None => (factor, factor),
                };
                p.set("scale_u", PropertyValue::Float(u));
                p.set("scale_v", PropertyValue::Float(v));
            }
        }
        p
    }

    /// How much further from the pivot the pointer has moved, as a ratio.
    ///
    /// One when the gesture began, whatever the pointer's distance was — so a
    /// scale started far from the pivot does not immediately jump.
    fn distance_ratio(&self) -> f64 {
        let from = (self.origin - self.pivot).length() as f64;
        let to = (self.cursor - self.pivot).length() as f64;
        if from <= f64::EPSILON {
            // Started on the pivot: there is no radius to compare against, and
            // any answer would be arbitrary. One leaves the selection alone
            // until the user moves off it.
            return 1.0;
        }
        let ratio = to / from;
        match self.precise_from {
            None => ratio,
            Some(_) => 1.0 + (ratio - 1.0) * PRECISION as f64,
        }
    }

    /// The line the header shows: tool, constraint, precision, value.
    pub fn status(&self) -> String {
        let axis = match self.axis {
            Some(a) => format!(" {}", a.label()),
            None => String::new(),
        };
        let precise = if self.precise_from.is_some() {
            "  precise"
        } else {
            ""
        };
        let typed = if self.numeric.is_empty() {
            String::new()
        } else {
            format!("  [{}]", self.numeric)
        };
        let value = match self.kind {
            Kind::Move => {
                let d = self.delta();
                match self.axis {
                    Some(Axis::U) => format!("{:.4}", d.x),
                    Some(Axis::V) => format!("{:.4}", d.y),
                    None => format!("{:.4} {:.4}", d.x, d.y),
                }
            }
            Kind::Rotate => {
                let radians = self.typed().unwrap_or(self.swept);
                format!("{:.1} deg", radians.to_degrees())
            }
            Kind::Scale => {
                let f = self.typed().unwrap_or_else(|| self.distance_ratio());
                format!("{f:.4}")
            }
        };
        format!("{}{axis}{precise}  {value}{typed}", self.kind.label())
    }
}

/// How much finer precision mode is.
const PRECISION: f32 = 0.1;

/// The bearing of a direction, in radians.
fn bearing_of(d: glam::Vec2) -> f64 {
    if d.length_squared() <= f32::EPSILON {
        0.0
    } else {
        (d.y as f64).atan2(d.x as f64)
    }
}

/// A difference reduced to `-pi..pi`.
fn normalise(a: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut a = a % tau;
    if a > std::f64::consts::PI {
        a -= tau;
    } else if a < -std::f64::consts::PI {
        a += tau;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn float(p: &PropertyMap, name: &str) -> f64 {
        match p.get(name) {
            Some(PropertyValue::Float(v)) => *v,
            other => panic!("{name} is {other:?}"),
        }
    }

    #[test]
    fn a_gesture_that_has_not_moved_asks_for_nothing() {
        // Pressing the key must not move anything. If the identity were not
        // the identity, every gesture would nudge the selection the instant it
        // began — and the user would see the map twitch on every keypress.
        let g = UvGesture::begin(Kind::Move, Vec2::new(0.4, 0.6), Vec2::splat(0.5));
        let p = g.params();
        assert_eq!(float(&p, "offset_u"), 0.0);
        assert_eq!(float(&p, "offset_v"), 0.0);

        let g = UvGesture::begin(Kind::Rotate, Vec2::new(0.4, 0.6), Vec2::splat(0.5));
        assert_eq!(float(&g.params(), "rotation"), 0.0);

        let g = UvGesture::begin(Kind::Scale, Vec2::new(0.4, 0.6), Vec2::splat(0.5));
        assert_eq!(float(&g.params(), "scale_u"), 1.0);
        assert_eq!(float(&g.params(), "scale_v"), 1.0);
    }

    #[test]
    fn a_move_follows_the_pointer() {
        let mut g = UvGesture::begin(Kind::Move, Vec2::new(0.2, 0.2), Vec2::splat(0.5));
        g.track(Vec2::new(0.5, 0.1));
        let p = g.params();
        assert!((float(&p, "offset_u") - 0.3).abs() < 1e-5);
        assert!((float(&p, "offset_v") + 0.1).abs() < 1e-5);
    }

    #[test]
    fn a_constraint_drops_the_other_axis_and_toggles_off() {
        let mut g = UvGesture::begin(Kind::Move, Vec2::new(0.2, 0.2), Vec2::splat(0.5));
        g.track(Vec2::new(0.5, 0.6));

        g.constrain(Axis::U);
        let p = g.params();
        assert!((float(&p, "offset_u") - 0.3).abs() < 1e-5);
        assert_eq!(float(&p, "offset_v"), 0.0);

        g.constrain(Axis::V);
        let p = g.params();
        assert_eq!(float(&p, "offset_u"), 0.0);
        assert!((float(&p, "offset_v") - 0.4).abs() < 1e-5);

        // The same key again releases it, so the user does not need a second
        // key to mean "never mind".
        g.constrain(Axis::V);
        assert_eq!(g.axis(), None);
        let p = g.params();
        assert!((float(&p, "offset_u") - 0.3).abs() < 1e-5);
        assert!((float(&p, "offset_v") - 0.4).abs() < 1e-5);
    }

    #[test]
    fn a_rotation_accumulates_past_half_a_turn() {
        // The bug this gesture's `swept` field exists to prevent. The angle
        // between two directions is always the short way round, so a drag past
        // half a turn differenced from the origin flips sign — the selection
        // snaps back the other way mid-drag.
        let pivot = Vec2::splat(0.5);
        let mut g = UvGesture::begin(Kind::Rotate, Vec2::new(1.0, 0.5), pivot);
        // Three quarters of a turn, in twelve steps.
        for i in 1..=12 {
            let a = std::f32::consts::TAU * 0.75 * (i as f32 / 12.0);
            g.track(pivot + Vec2::new(a.cos(), a.sin()) * 0.5);
        }
        let turned = float(&g.params(), "rotation");
        assert!(
            (turned - std::f64::consts::TAU * 0.75).abs() < 1e-3,
            "swept {turned}, expected three quarters of a turn"
        );
    }

    #[test]
    fn a_scale_starts_at_one_however_far_the_pointer_is() {
        // Otherwise starting a scale with the pointer far from the pivot jumps
        // the selection before the user has moved at all.
        for radius in [0.05, 0.5, 5.0] {
            let pivot = Vec2::splat(0.5);
            let g = UvGesture::begin(Kind::Scale, pivot + Vec2::new(radius, 0.0), pivot);
            assert!(
                (float(&g.params(), "scale_u") - 1.0).abs() < 1e-9,
                "{radius}"
            );
        }
    }

    #[test]
    fn a_scale_doubles_when_the_pointer_doubles_its_distance() {
        let pivot = Vec2::splat(0.5);
        let mut g = UvGesture::begin(Kind::Scale, pivot + Vec2::new(0.2, 0.0), pivot);
        g.track(pivot + Vec2::new(0.4, 0.0));
        assert!((float(&g.params(), "scale_u") - 2.0).abs() < 1e-5);
    }

    #[test]
    fn a_scale_started_on_the_pivot_leaves_the_selection_alone() {
        // There is no radius to compare against, so any ratio would be
        // arbitrary — and dividing by it would be worse than arbitrary.
        let pivot = Vec2::splat(0.5);
        let mut g = UvGesture::begin(Kind::Scale, pivot, pivot);
        g.track(pivot + Vec2::new(0.3, 0.1));
        let f = float(&g.params(), "scale_u");
        assert!(f.is_finite(), "{f}");
        assert_eq!(f, 1.0);
    }

    #[test]
    fn precision_is_measured_from_where_it_was_engaged() {
        // §2.2's rule. Engaging precision mode must not move the selection: it
        // changes the rate from that point on, and nothing else.
        let mut g = UvGesture::begin(Kind::Move, Vec2::ZERO, Vec2::splat(0.5));
        g.track(Vec2::new(0.4, 0.0));
        let before = float(&g.params(), "offset_u");
        g.set_precise(true);
        assert!(
            (float(&g.params(), "offset_u") - before).abs() < 1e-9,
            "engaging precision moved the selection"
        );

        // And from there it moves at a tenth of the rate.
        g.track(Vec2::new(0.5, 0.0));
        let after = float(&g.params(), "offset_u");
        assert!(
            (after - (before + 0.1 * f64::from(PRECISION))).abs() < 1e-5,
            "{before} -> {after}"
        );
    }

    #[test]
    fn releasing_precision_keeps_what_it_contributed() {
        // Otherwise letting go of Shift snaps the selection to where it would
        // have been at full rate, which is a jump the user did not ask for.
        let mut g = UvGesture::begin(Kind::Move, Vec2::ZERO, Vec2::splat(0.5));
        g.track(Vec2::new(0.4, 0.0));
        g.set_precise(true);
        g.track(Vec2::new(0.5, 0.0));
        let during = float(&g.params(), "offset_u");
        g.set_precise(false);
        assert!(
            (float(&g.params(), "offset_u") - during).abs() < 1e-5,
            "releasing precision jumped: {during} -> {}",
            float(&g.params(), "offset_u")
        );
    }

    #[test]
    fn a_typed_number_wins_over_the_pointer() {
        let mut g = UvGesture::begin(Kind::Move, Vec2::ZERO, Vec2::splat(0.5));
        g.track(Vec2::new(0.4, 0.3));
        for c in "0.25".chars() {
            assert!(g.type_char(c), "{c} was not taken");
        }
        g.constrain(Axis::V);
        let p = g.params();
        assert_eq!(float(&p, "offset_u"), 0.0);
        assert!((float(&p, "offset_v") - 0.25).abs() < 1e-9);
    }

    #[test]
    fn an_expression_is_evaluated_and_a_partial_one_is_ignored() {
        let mut g = UvGesture::begin(Kind::Move, Vec2::ZERO, Vec2::splat(0.5));
        for c in "1/8".chars() {
            g.type_char(c);
        }
        assert!((float(&g.params(), "offset_u") - 0.125).abs() < 1e-9);

        // A trailing operator is not a number yet, and must not be treated as
        // zero: that would jump the selection to the pivot mid-typing.
        let mut g = UvGesture::begin(Kind::Move, Vec2::ZERO, Vec2::splat(0.5));
        g.track(Vec2::new(0.4, 0.0));
        for c in "2*".chars() {
            g.type_char(c);
        }
        assert!(
            (float(&g.params(), "offset_u") - 0.4).abs() < 1e-5,
            "an incomplete expression displaced the pointer's value"
        );
    }

    #[test]
    fn a_rotation_takes_degrees_and_a_move_does_not_take_a_length() {
        // UV space is dimensionless: an offset is a fraction of the map and a
        // scale is a ratio, so `2m` is not a UV offset and must be refused
        // rather than silently read as two.
        let mut g = UvGesture::begin(Kind::Rotate, Vec2::new(1.0, 0.5), Vec2::splat(0.5));
        for c in "45deg".chars() {
            g.type_char(c);
        }
        let turned = float(&g.params(), "rotation");
        assert!(
            (turned - std::f64::consts::FRAC_PI_4).abs() < 1e-9,
            "{turned}"
        );

        let mut g = UvGesture::begin(Kind::Move, Vec2::ZERO, Vec2::splat(0.5));
        g.track(Vec2::new(0.4, 0.0));
        for c in "2m".chars() {
            g.type_char(c);
        }
        assert!(
            (float(&g.params(), "offset_u") - 0.4).abs() < 1e-5,
            "a length was accepted as a UV offset"
        );
    }

    #[test]
    fn a_tool_key_is_not_swallowed_by_an_empty_numeric_buffer() {
        // `g`, `r` and `s` are unit letters *and* tool keys. Taking them
        // before a number has been started would mean the second `G` of a
        // double press vanished into the buffer.
        let mut g = UvGesture::begin(Kind::Move, Vec2::ZERO, Vec2::splat(0.5));
        assert!(!g.type_char('g'));
        assert!(!g.type_char('r'));
        assert!(g.type_char('4'));
        assert!(g.type_char('d'));
    }

    #[test]
    fn backspace_removes_what_was_typed_and_then_reports_nothing_left() {
        let mut g = UvGesture::begin(Kind::Move, Vec2::ZERO, Vec2::splat(0.5));
        g.type_char('7');
        assert!(g.backspace());
        assert!(!g.backspace());
    }

    #[test]
    fn the_status_line_says_the_tool_the_axis_and_the_value() {
        let mut g = UvGesture::begin(Kind::Move, Vec2::ZERO, Vec2::splat(0.5));
        g.track(Vec2::new(0.25, 0.0));
        g.constrain(Axis::U);
        let s = g.status();
        assert!(s.contains("uv-move"), "{s}");
        assert!(s.contains('U'), "{s}");
        assert!(s.contains("0.25"), "{s}");
    }
}
