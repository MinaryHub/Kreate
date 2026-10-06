//! Nonlinear animation: actions laid on a timeline as strips.
//!
//! Blender's NLA, its first useful part. An object plays its strips, bottom
//! to top, each one an action moved to where it starts, slowed or sped up,
//! repeated, and mixed in by an influence; then its active action on top, as
//! Blender does. A walk keyed once becomes a walk for as long as the shot by
//! repeating its strip; two takes become one by laying them end to end.
//!
//! Strips replace what is under them by their influence — a straight line
//! from below to theirs, the turn by the shorter arc. Blender's Add, Subtract
//! and Multiply blends are not here yet.

use crate::document::ActionId;
use kreate_math::{DQuat, DTransform};

/// What a strip does outside its own stretch of the timeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NlaExtend {
    /// Nothing: the strip plays only between its ends.
    Nothing,
    /// Holds its last frame after it ends: Blender's default for a strip
    /// above another.
    #[default]
    HoldForward,
    /// Holds its first frame before and its last after: Blender's default
    /// for the bottom strip.
    Hold,
}

impl NlaExtend {
    pub const ALL: [Self; 3] = [Self::Nothing, Self::HoldForward, Self::Hold];

    pub fn tag(self) -> &'static str {
        match self {
            Self::Nothing => "nothing",
            Self::HoldForward => "hold_forward",
            Self::Hold => "hold",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|e| e.tag() == tag)
    }

    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|e| *e == self).unwrap_or(0) as u32
    }
}

/// One action, placed on the timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct NlaStrip {
    pub name: String,
    pub action: ActionId,
    /// The scene frame it starts at.
    pub start: f64,
    /// The stretch of the action it plays — its keys' range when it is made.
    pub action_start: f64,
    pub action_end: f64,
    /// How many scene frames one of the action's takes; above one is slower.
    pub scale: f64,
    /// How many times it plays, end to end.
    pub repeat: f64,
    /// How much of it shows, 0 to 1.
    pub influence: f64,
    pub extend: NlaExtend,
    pub mute: bool,
}

impl NlaStrip {
    pub fn new(name: impl Into<String>, action: ActionId, start: f64, range: (f64, f64)) -> Self {
        Self {
            name: name.into(),
            action,
            start,
            action_start: range.0,
            action_end: range.1,
            scale: 1.0,
            repeat: 1.0,
            influence: 1.0,
            extend: NlaExtend::HoldForward,
            mute: false,
        }
    }

    /// The scene frame it ends at.
    pub fn end(&self) -> f64 {
        self.start + self.length() * self.scale.max(1e-6) * self.repeat.max(1e-6)
    }

    fn length(&self) -> f64 {
        (self.action_end - self.action_start).max(0.0)
    }

    /// The action's own frame at scene frame `frame`, or `None` where the
    /// strip shows nothing.
    pub fn action_frame(&self, frame: f64) -> Option<f64> {
        if self.mute {
            return None;
        }
        let (start, end) = (self.start, self.end());
        let frame = if frame < start {
            match self.extend {
                NlaExtend::Hold => start,
                _ => return None,
            }
        } else if frame > end {
            match self.extend {
                NlaExtend::Nothing => return None,
                _ => end,
            }
        } else {
            frame
        };
        let length = self.length();
        if length <= 0.0 {
            return Some(self.action_start);
        }
        let along = (frame - start) / self.scale.max(1e-6);
        // Repeats wrap; the very end is the last frame, not the first again.
        let total = length * self.repeat.max(1e-6);
        let within = if along >= total {
            length
        } else {
            along.rem_euclid(length)
        };
        Some(self.action_start + within)
    }
}

/// `a` towards `b` by `t`: positions and scales in straight lines, the turn
/// by the shorter arc.
pub fn mix_transform(a: DTransform, b: DTransform, t: f64) -> DTransform {
    if t >= 1.0 {
        return b;
    }
    let to = if a.rotation.dot(b.rotation) < 0.0 { -b.rotation } else { b.rotation };
    DTransform {
        translation: a.translation.lerp(b.translation, t),
        rotation: DQuat::slerp(a.rotation, to, t).normalize(),
        scale: a.scale.lerp(b.scale, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip() -> NlaStrip {
        let action = crate::Document::new().add_action(crate::animation::Action::new("Walk"));
        NlaStrip::new("Walk", action, 10.0, (1.0, 21.0))
    }

    #[test]
    fn a_strip_plays_its_action_from_where_it_starts() {
        let s = strip();
        assert_eq!(s.end(), 30.0);
        assert_eq!(s.action_frame(10.0), Some(1.0));
        assert_eq!(s.action_frame(20.0), Some(11.0));
        // Held after, nothing before — the default.
        assert_eq!(s.action_frame(40.0), Some(21.0));
        assert_eq!(s.action_frame(5.0), None);
    }

    #[test]
    fn scale_and_repeat_stretch_and_loop_it() {
        let mut s = strip();
        s.scale = 2.0;
        s.repeat = 3.0;
        assert_eq!(s.end(), 10.0 + 20.0 * 2.0 * 3.0);
        assert_eq!(s.action_frame(30.0), Some(11.0));
        // The second time round starts again from the beginning.
        assert_eq!(s.action_frame(50.0), Some(1.0));
        assert_eq!(s.action_frame(60.0), Some(6.0));
        assert_eq!(s.action_frame(s.end()), Some(21.0));
    }

    #[test]
    fn hold_holds_both_ways_and_nothing_neither_and_mute_silences() {
        let mut s = strip();
        s.extend = NlaExtend::Hold;
        assert_eq!(s.action_frame(0.0), Some(1.0));
        s.extend = NlaExtend::Nothing;
        assert_eq!(s.action_frame(31.0), None);
        s.extend = NlaExtend::Hold;
        s.mute = true;
        assert_eq!(s.action_frame(20.0), None);
    }

    /// An object whose X runs 0 to 10 over frames 1 to 11 in one action and
    /// sits at 100 in another.
    fn scene() -> (crate::Document, crate::document::ObjectId, ActionId, ActionId) {
        use crate::animation::{Action, Interpolation};
        use crate::document::{Object, ObjectData};
        let mut doc = crate::Document::new();
        let mut run = Action::new("Run");
        let c = run.curve_mut("location", 0);
        c.insert(1.0, 0.0, Interpolation::Linear);
        c.insert(11.0, 10.0, Interpolation::Linear);
        let run = doc.add_action(run);
        let mut stand = Action::new("Stand");
        stand.curve_mut("location", 0).insert(1.0, 100.0, Interpolation::Linear);
        let stand = doc.add_action(stand);
        let id = doc.add_object(Object::new("Mover", ObjectData::Empty));
        (doc, id, run, stand)
    }

    fn x(doc: &mut crate::Document, id: crate::document::ObjectId, frame: f64) -> f64 {
        doc.animate(frame);
        doc.object(id).unwrap().transform.translation.x
    }

    #[test]
    fn strips_play_where_they_are_placed_repeat_and_mix_by_influence() {
        let (mut doc, id, run, stand) = scene();
        let mut walk = NlaStrip::new("Run", run, 21.0, (1.0, 11.0));
        walk.repeat = 2.0;
        walk.extend = NlaExtend::Hold;
        doc.object_mut(id).unwrap().nla.push(walk);
        // Held before it starts, then played from frame 21, twice.
        assert!((x(&mut doc, id, 5.0) - 0.0).abs() < 1e-9);
        assert!((x(&mut doc, id, 26.0) - 5.0).abs() < 1e-9);
        assert!((x(&mut doc, id, 36.0) - 5.0).abs() < 1e-9);
        // A strip above, at half influence, from frame 30 on: halfway to 100.
        let mut over = NlaStrip::new("Stand", stand, 30.0, (1.0, 1.0));
        over.influence = 0.5;
        doc.object_mut(id).unwrap().nla.push(over);
        assert!((x(&mut doc, id, 36.0) - (5.0 + (100.0 - 5.0) * 0.5)).abs() < 1e-9);
        assert!((x(&mut doc, id, 26.0) - 5.0).abs() < 1e-9, "not before it starts");
        // Muted, it is gone.
        doc.object_mut(id).unwrap().nla[1].mute = true;
        assert!((x(&mut doc, id, 36.0) - 5.0).abs() < 1e-9);
        // The active action plays on top of every strip.
        doc.object_mut(id).unwrap().action = Some(stand);
        assert!((x(&mut doc, id, 26.0) - 100.0).abs() < 1e-9);
    }
}
