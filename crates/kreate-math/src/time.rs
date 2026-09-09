//! When a scene is, and in what unit.
//!
//! Time is the second thing after colour that a `f64` cannot say enough about.
//! A bare `time: f64` in an evaluation context is either a frame number or a
//! quantity of seconds, and the two differ by the scene's rate — so a modifier
//! reading it as one while a curve wrote the other is off by a factor of
//! twenty-four with nothing to blame. [`SceneTime`] carries both and the rate
//! that relates them.
//!
//! # Stored as a frame, available as seconds
//!
//! The frame is the authored quantity, and that follows production practice
//! rather than tidiness. An artist puts a key "on frame 12", and if the stored
//! value were seconds then changing the delivery rate from 24 to 30 would land
//! most keys *between* frames — which breaks stepped interpolation and any pose
//! that was meant to hold for exactly one frame. Every tool stores frames for
//! this reason.
//!
//! Seconds are what anything physical needs: a shutter interval for motion
//! blur, a simulation step, a driver expression about velocity. Those get
//! [`SceneTime::seconds`], which is the frame divided by the rate, and they
//! never see a frame number.
//!
//! The consequence, stated so that nobody discovers it: **changing the scene's
//! rate changes the duration of an animation, not its frame numbering.** A
//! twenty-four frame action is one second at 24 fps and eight tenths at 30. That
//! is what an artist expects from a frame-authored tool, and it is the opposite
//! of what a seconds-authored one would do.
//!
//! # Fractional frames are the normal case
//!
//! `frame` is `f64` and not an integer. Motion blur samples between frames,
//! playback at a rate the display cannot divide lands between them, and a
//! sub-frame step is how a simulation is asked for a state nobody keyed. An
//! integer frame would push every one of those into a separate mechanism.

/// A point in a scene's time.
///
/// Copy and comparable, because it is a cache key: the evaluated scene is a
/// function of the document *and* this, and an evaluation cache that could not
/// compare two times would serve one frame's geometry for another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneTime {
    frame: f64,
    rate: f64,
}

impl Default for SceneTime {
    /// Frame one at twenty-four per second.
    ///
    /// One rather than zero because that is where every timeline in this
    /// industry starts, and 24 because it is the only rate that is a default
    /// anywhere rather than a house style.
    fn default() -> Self {
        Self {
            frame: 1.0,
            rate: 24.0,
        }
    }
}

impl SceneTime {
    /// A time at `frame`, in a scene running at `rate` frames a second.
    ///
    /// A rate that is not a positive finite number is refused rather than
    /// stored: everything below divides by it, and one `NaN` frame rate would
    /// turn every evaluated position into a `NaN` with the cause several
    /// systems away.
    pub fn new(frame: f64, rate: f64) -> Option<Self> {
        if !rate.is_finite() || rate <= 0.0 || !frame.is_finite() {
            return None;
        }
        Some(Self { frame, rate })
    }

    /// The authored quantity: which frame, possibly between two.
    pub fn frame(&self) -> f64 {
        self.frame
    }

    /// Frames a second.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// The physical quantity: how long since frame zero.
    ///
    /// What a shutter, a simulation step or an expression about speed wants.
    /// Measured from frame zero rather than from frame one, so that a duration
    /// is a subtraction and needs no correction — `b.seconds() - a.seconds()`
    /// is the interval between two times whatever the timeline starts at.
    pub fn seconds(&self) -> f64 {
        self.frame / self.rate
    }

    /// The same instant, `frames` later.
    pub fn advanced_by(&self, frames: f64) -> Self {
        Self {
            frame: self.frame + frames,
            rate: self.rate,
        }
    }

    /// The same instant, `seconds` later.
    pub fn after(&self, seconds: f64) -> Self {
        self.advanced_by(seconds * self.rate)
    }

    /// The same frame, at a different rate.
    ///
    /// What changing a scene's delivery rate does: the keys stay where they were
    /// authored and the animation's duration changes. See the module docs for
    /// why that is the right way round.
    pub fn at_rate(&self, rate: f64) -> Option<Self> {
        Self::new(self.frame, rate)
    }

    /// This time snapped to a whole frame, towards zero.
    ///
    /// For a display that shows a frame number, and for a stepped curve. Not
    /// for evaluation: rounding the time before evaluating is how a motion blur
    /// sample ends up on the same frame as the one before it.
    pub fn whole_frame(&self) -> i64 {
        self.frame.floor() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_are_the_frame_divided_by_the_rate() {
        let t = SceneTime::new(24.0, 24.0).expect("a valid time");
        assert!((t.seconds() - 1.0).abs() < 1e-12);

        let t = SceneTime::new(15.0, 30.0).expect("a valid time");
        assert!((t.seconds() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_duration_is_a_subtraction_whatever_the_timeline_starts_at() {
        // Why seconds are measured from frame zero rather than from frame one.
        // A shutter interval and a simulation step are both differences, and a
        // representation needing a correction for where the timeline begins
        // would have every one of them carry it.
        let a = SceneTime::new(1.0, 24.0).expect("valid");
        let b = a.advanced_by(24.0);
        assert!((b.seconds() - a.seconds() - 1.0).abs() < 1e-12);
        assert_eq!(b.frame(), 25.0);
    }

    #[test]
    fn changing_the_rate_keeps_the_frame_and_changes_the_duration() {
        // The consequence of storing the frame, stated as a test because it is
        // the half of the decision somebody will be surprised by. A key on
        // frame 12 stays on frame 12; what changes is how long it takes to get
        // there.
        let at_24 = SceneTime::new(12.0, 24.0).expect("valid");
        let at_30 = at_24.at_rate(30.0).expect("valid");
        assert_eq!(at_30.frame(), 12.0, "the key moved");
        assert!((at_24.seconds() - 0.5).abs() < 1e-12);
        assert!((at_30.seconds() - 0.4).abs() < 1e-12);
    }

    #[test]
    fn a_key_authored_in_seconds_would_have_landed_between_frames() {
        // The other way round, for comparison — and the reason it is not the
        // way round chosen. Frame 13 at 24 fps is 13/24 of a second, which at
        // 30 fps is frame 16.25: a pose meant to hold for exactly one frame now
        // starts a quarter of a frame late, and stepped interpolation has
        // nothing to step on.
        let authored = SceneTime::new(13.0, 24.0).expect("valid");
        let as_seconds = authored.seconds();
        let renumbered = as_seconds * 30.0;
        assert!(
            (renumbered - 16.25).abs() < 1e-9,
            "the fixture should land between frames, landed on {renumbered}"
        );
        assert!(
            renumbered.fract() > 1e-6,
            "a seconds-authored key should not fall on a frame here"
        );
    }

    #[test]
    fn a_rate_that_cannot_be_divided_by_is_refused() {
        // Everything above divides by the rate. One `NaN` here becomes a `NaN`
        // position several systems away, with nothing pointing back.
        assert!(SceneTime::new(1.0, 0.0).is_none());
        assert!(SceneTime::new(1.0, -24.0).is_none());
        assert!(SceneTime::new(1.0, f64::NAN).is_none());
        assert!(SceneTime::new(1.0, f64::INFINITY).is_none());
        assert!(SceneTime::new(f64::NAN, 24.0).is_none());
        assert!(SceneTime::new(1.0, 23.976).is_some());
    }

    #[test]
    fn a_fractional_frame_is_an_ordinary_time() {
        // Motion blur samples between frames, and so does playback at a rate
        // the display cannot divide. An integer frame would push both into a
        // separate mechanism.
        let half = SceneTime::new(12.5, 25.0).expect("valid");
        assert!((half.seconds() - 0.5).abs() < 1e-12);
        assert_eq!(half.whole_frame(), 12);

        // And a shutter offset asked for in seconds lands wherever it lands.
        let sample = half.after(0.01);
        assert!((sample.frame() - 12.75).abs() < 1e-9);
    }

    #[test]
    fn two_times_compare_by_frame_and_by_rate() {
        // It is a cache key. Two times that differ in either part are different
        // times, and a cache that thought otherwise would serve one frame's
        // geometry for another.
        let a = SceneTime::new(12.0, 24.0).expect("valid");
        assert_eq!(a, SceneTime::new(12.0, 24.0).expect("valid"));
        assert_ne!(a, SceneTime::new(13.0, 24.0).expect("valid"));
        assert_ne!(
            a,
            SceneTime::new(12.0, 30.0).expect("valid"),
            "the same frame at another rate is another instant"
        );
    }

    #[test]
    fn the_default_is_the_first_frame() {
        let t = SceneTime::default();
        assert_eq!(t.frame(), 1.0);
        assert_eq!(t.rate(), 24.0);
    }
}
