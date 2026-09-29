//! Keyframes, curves and actions: values over time.
//!
//! §2.4's data, and the first thing that makes the scene a function of time
//! rather than a still. An [`Action`] is a datablock holding [`FCurve`]s; an
//! object plays the action it is given; a curve is one number over frames,
//! written as [`Keyframe`]s and read by [`FCurve::evaluate`].
//!
//! # A curve's path is relative to whoever plays it
//!
//! A curve names a property — `location`, component 0 — and not a target: the
//! [`PropertyPath`]'s target is [`Uid::NONE`], which here means "the object
//! playing this action". §2.4 wants an action shareable — a walk cycle on two
//! characters — and a curve that named one object by identity would move that
//! object whoever played it. The identity half of the path is still what a
//! driver or a constraint will use, where a target is the point.
//!
//! # Rotation is keyed as Euler angles
//!
//! Blender's default and every animator's expectation: three curves in
//! degrees-worth of radians, X then Y then Z, that can wind past a full turn.
//! The transform stores a quaternion, which cannot say "two turns", so keying
//! reads the quaternion out as angles and playing writes them back. The cost
//! is the known one: a key reads the rotation as it stands, so a multi-turn
//! spin is made by editing a key's value rather than by rotating past 360°.
//!
//! # Interpolation
//!
//! Constant, linear, and Bézier with Blender's *auto-clamped* handles — the
//! default a new key gets, because it is the one that looks like motion: a
//! curve eases through its keys and never overshoots one that is a peak or a
//! trough. Handles are derived from the neighbouring keys rather than stored,
//! which is what "auto" means; a graph editor that lets somebody drag a
//! handle is where stored handles arrive.

use kreate_math::{DQuat, DTransform, DVec3};
use kreate_reflect::{AnimatableDesc, DatablockKind, PropertyKind, PropertyPath, Uid};

use crate::document::Linkage;

/// How the curve travels from a key to the next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// Holds the key's value until the next key: a cut, a blink.
    Constant,
    Linear,
    /// Eases in and out with auto-clamped handles.
    #[default]
    Bezier,
}

impl Interpolation {
    pub const ALL: [Interpolation; 3] = [Self::Constant, Self::Linear, Self::Bezier];

    /// A stable name for files and scripts.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Linear => "linear",
            Self::Bezier => "bezier",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|i| i.tag() == tag)
    }
}

/// One key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keyframe {
    pub frame: f64,
    pub value: f64,
    /// How the curve leaves this key toward the next.
    pub interpolation: Interpolation,
    /// Whether the dope sheet has it selected.
    ///
    /// On the key, as Blender keeps it: selection is part of the document, so
    /// it saves, it undoes, and the operators that act on "the selected keys"
    /// can be replayed from the history without being told which they were.
    pub selected: bool,
}

/// Two keys closer than this are the same key: a key inserted where one
/// already is replaces it rather than stacking a second.
pub const SAME_FRAME: f64 = 1e-4;

/// One number over time.
#[derive(Clone, Debug, PartialEq)]
pub struct FCurve {
    pub path: PropertyPath,
    /// Sorted by frame, never two at one frame. Kept so by [`FCurve::insert`];
    /// a file that says otherwise is sorted on the way in.
    pub keys: Vec<Keyframe>,
}

impl FCurve {
    pub fn new(path: PropertyPath) -> Self {
        Self { path, keys: Vec::new() }
    }

    /// An empty curve for one component of a property of whoever plays it —
    /// the kind every action curve is. See the module docs on the target.
    pub fn owned(property: &str, index: u8) -> Self {
        Self::new(PropertyPath::component(DatablockKind::Object, Uid::NONE, property, index))
    }

    /// Puts a key at `frame`, replacing one already there.
    pub fn insert(&mut self, frame: f64, value: f64, interpolation: Interpolation) {
        // A key just put down is selected, so the next move or delete in the
        // dope sheet acts on what was just keyed.
        let key = Keyframe { frame, value, interpolation, selected: true };
        match self.keys.iter().position(|k| (k.frame - frame).abs() < SAME_FRAME) {
            Some(at) => self.keys[at] = key,
            None => {
                let at = self.keys.partition_point(|k| k.frame < frame);
                self.keys.insert(at, key);
            }
        }
    }

    /// Takes away the key at `frame`, if there is one.
    pub fn remove(&mut self, frame: f64) -> bool {
        let before = self.keys.len();
        self.keys.retain(|k| (k.frame - frame).abs() >= SAME_FRAME);
        self.keys.len() != before
    }

    /// Sorts, and drops keys that share a frame — for keys read from a file.
    pub fn normalise(&mut self) {
        self.keys.retain(|k| k.frame.is_finite() && k.value.is_finite());
        self.keys.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        self.keys.dedup_by(|b, a| (b.frame - a.frame).abs() < SAME_FRAME);
    }

    /// The curve's value at `frame`, or `None` for a curve with no keys.
    ///
    /// Before the first key and after the last, the value holds: Blender's
    /// constant extrapolation, and the only one that does not send an object
    /// flying off after its last key.
    pub fn evaluate(&self, frame: f64) -> Option<f64> {
        let first = self.keys.first()?;
        let last = self.keys.last()?;
        if frame <= first.frame {
            return Some(first.value);
        }
        if frame >= last.frame {
            return Some(last.value);
        }
        let next = self.keys.partition_point(|k| k.frame <= frame);
        let (i, a, b) = (next - 1, self.keys[next - 1], self.keys[next]);
        let span = b.frame - a.frame;
        let t = (frame - a.frame) / span;
        Some(match a.interpolation {
            Interpolation::Constant => a.value,
            Interpolation::Linear => a.value + (b.value - a.value) * t,
            Interpolation::Bezier => {
                // Hermite with the auto-clamped slopes. Blender's Bézier with
                // handles a third of the way along each side is exactly this:
                // its frame is linear in the curve parameter, so the value is
                // a cubic in the frame with these end slopes.
                let (m0, m1) = (self.slope(i) * span, self.slope(next) * span);
                let t2 = t * t;
                let t3 = t2 * t;
                (2.0 * t3 - 3.0 * t2 + 1.0) * a.value
                    + (t3 - 2.0 * t2 + t) * m0
                    + (-2.0 * t3 + 3.0 * t2) * b.value
                    + (t3 - t2) * m1
            }
        })
    }

    /// The auto-clamped slope at key `i`, in value per frame.
    ///
    /// Flat at the two ends and at any key that is a peak or a trough — the
    /// clamp, which is what keeps a bouncing ball from dipping through the
    /// floor between two keys that both say "on the floor". Otherwise the
    /// slope of the line through the two neighbours.
    fn slope(&self, i: usize) -> f64 {
        let (Some(prev), Some(next)) = (i.checked_sub(1).and_then(|p| self.keys.get(p)), self.keys.get(i + 1)) else {
            return 0.0;
        };
        let here = self.keys[i].value;
        if (here - prev.value) * (next.value - here) <= 0.0 {
            return 0.0;
        }
        (next.value - prev.value) / (next.frame - prev.frame)
    }
}

/// A set of curves, played by whatever is given it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Action {
    pub name: String,
    pub linkage: Linkage,
    /// Assigned by the document. See [`Uid`].
    pub uid: Uid,
    pub curves: Vec<FCurve>,
}

impl Action {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// The curve for one component of one property, made if it is not there.
    pub fn curve_mut(&mut self, property: &str, index: u8) -> &mut FCurve {
        let at = self
            .curves
            .iter()
            .position(|c| c.path.property.as_ref() == property && c.path.index == Some(index));
        match at {
            Some(at) => &mut self.curves[at],
            None => {
                self.curves.push(FCurve::owned(property, index));
                self.curves.last_mut().expect("just pushed")
            }
        }
    }

    /// Selects (or, with `select` false, deselects) the keys of `property` —
    /// or of every curve — whose frames are within `min..=max`.
    pub fn select_range(&mut self, property: Option<&str>, min: f64, max: f64, select: bool) -> bool {
        let mut changed = false;
        for curve in self
            .curves
            .iter_mut()
            .filter(|c| property.is_none_or(|p| c.path.property.as_ref() == p))
        {
            for key in &mut curve.keys {
                if key.frame >= min - 1e-9 && key.frame <= max + 1e-9 && key.selected != select {
                    key.selected = select;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Sets every key's selection with `f`. Returns whether anything changed.
    pub fn select_all_with(&mut self, f: impl Fn(bool) -> bool) -> bool {
        let mut changed = false;
        for curve in &mut self.curves {
            for key in &mut curve.keys {
                let to = f(key.selected);
                changed |= to != key.selected;
                key.selected = to;
            }
        }
        changed
    }

    pub fn any_selected(&self) -> bool {
        self.curves.iter().any(|c| c.keys.iter().any(|k| k.selected))
    }

    /// Moves the selected keys by `offset` frames.
    ///
    /// A moved key landing where an unselected key is replaces it, as it does
    /// in Blender: two keys at one frame is not a state a curve can be in, and
    /// the one being moved is the one somebody meant.
    pub fn move_selected(&mut self, offset: f64) -> bool {
        let mut moved = false;
        for curve in &mut self.curves {
            let (mut going, staying): (Vec<Keyframe>, Vec<Keyframe>) =
                curve.keys.iter().partition(|k| k.selected);
            if going.is_empty() {
                continue;
            }
            for key in &mut going {
                key.frame += offset;
            }
            let mut keys: Vec<Keyframe> = staying
                .into_iter()
                .filter(|k| !going.iter().any(|g| (g.frame - k.frame).abs() < SAME_FRAME))
                .collect();
            keys.extend(going);
            keys.sort_by(|a, b| a.frame.total_cmp(&b.frame));
            curve.keys = keys;
            moved = true;
        }
        moved
    }

    /// Takes away the selected keys, and any curve left with none.
    pub fn delete_selected(&mut self) -> bool {
        let before: usize = self.curves.iter().map(|c| c.keys.len()).sum();
        for curve in &mut self.curves {
            curve.keys.retain(|k| !k.selected);
        }
        self.curves.retain(|c| !c.keys.is_empty());
        let after: usize = self.curves.iter().map(|c| c.keys.len()).sum();
        after != before
    }

    /// Sets how the selected keys travel to the next.
    pub fn set_selected_interpolation(&mut self, interpolation: Interpolation) -> bool {
        let mut changed = false;
        for curve in &mut self.curves {
            for key in curve.keys.iter_mut().filter(|k| k.selected) {
                changed |= key.interpolation != interpolation;
                key.interpolation = interpolation;
            }
        }
        changed
    }

    /// Every frame some curve has a key on, sorted, once each.
    pub fn key_frames(&self) -> Vec<f64> {
        let mut frames: Vec<f64> = self.curves.iter().flat_map(|c| c.keys.iter().map(|k| k.frame)).collect();
        frames.sort_by(f64::total_cmp);
        frames.dedup_by(|b, a| (*b - *a).abs() < SAME_FRAME);
        frames
    }
}

/// The scene's clock: where it is, the range it plays, how fast.
///
/// Document state, saved with the file, as Blender's frame range is. Changing
/// the frame is not an edit — nothing is undone by stepping back through the
/// timeline — so it is written directly rather than through an operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Playback {
    pub frame: f64,
    pub start: i32,
    pub end: i32,
    pub fps: f64,
}

impl Default for Playback {
    /// Frames 1 to 250 at 24 a second: Blender's defaults, which is to say
    /// ten seconds of film.
    fn default() -> Self {
        Self { frame: 1.0, start: 1, end: 250, fps: 24.0 }
    }
}

impl Playback {
    /// The frame after `dt` seconds of playing, looping at the end.
    pub fn advanced(&self, dt: f64) -> f64 {
        let (start, end) = (self.start as f64, self.end.max(self.start) as f64);
        let next = self.frame + dt * self.fps;
        if next > end + 1.0 - 1e-9 {
            start + (next - start).rem_euclid(end + 1.0 - start)
        } else {
            next
        }
    }
}

// --- what an object animates ------------------------------------------------

/// The properties of an object a curve can drive.
pub const OBJECT_ANIMATABLE: &[AnimatableDesc] = &[
    AnimatableDesc {
        name: "location",
        kind: PropertyKind::Vec3,
        components: 3,
        doc: "Where the object is, relative to its parent.",
    },
    AnimatableDesc {
        name: "rotation_euler",
        kind: PropertyKind::Vec3,
        components: 3,
        doc: "Its rotation as angles about X, then Y, then Z, in radians.",
    },
    AnimatableDesc {
        name: "scale",
        kind: PropertyKind::Vec3,
        components: 3,
        doc: "Its size along each of its own axes.",
    },
];

/// A rotation as Blender's XYZ Euler angles: X applied first, then Y, then Z.
pub fn to_euler(q: DQuat) -> DVec3 {
    let (z, y, x) = q.to_euler(kreate_math::glam::EulerRot::ZYX);
    DVec3::new(x, y, z)
}

/// The inverse of [`to_euler`].
pub fn from_euler(angles: DVec3) -> DQuat {
    DQuat::from_euler(kreate_math::glam::EulerRot::ZYX, angles.z, angles.y, angles.x)
}

/// A transform's animatable values, as the curves see them.
pub fn read_object(t: &DTransform, property: &str) -> Option<DVec3> {
    Some(match property {
        "location" => t.translation,
        "rotation_euler" => to_euler(t.rotation),
        "scale" => t.scale,
        _ => return None,
    })
}

/// Plays `action` onto `transform` at `frame`: every keyed component is set,
/// and every component nothing keys is left as it was.
pub fn apply_to_object(action: &Action, frame: f64, transform: &mut DTransform) {
    let mut location = transform.translation;
    let mut euler: Option<DVec3> = None;
    let mut scale = transform.scale;
    for curve in &action.curves {
        let (Some(index), Some(value)) = (curve.path.index, curve.evaluate(frame)) else {
            continue;
        };
        let i = index as usize;
        if i > 2 {
            continue;
        }
        match curve.path.property.as_ref() {
            "location" => location[i] = value,
            "scale" => scale[i] = value,
            "rotation_euler" => {
                let e = euler.get_or_insert_with(|| to_euler(transform.rotation));
                e[i] = value;
            }
            _ => {}
        }
    }
    transform.translation = location;
    transform.scale = scale;
    if let Some(e) = euler {
        transform.rotation = from_euler(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(keys: &[(f64, f64, Interpolation)]) -> FCurve {
        let mut c = FCurve::new(PropertyPath::component(DatablockKind::Object, Uid::NONE, "location", 0));
        for (f, v, i) in keys {
            c.insert(*f, *v, *i);
        }
        c
    }

    #[test]
    fn a_key_lands_in_order_and_replaces_one_at_its_frame() {
        let mut c = curve(&[(10.0, 1.0, Interpolation::Linear), (1.0, 0.0, Interpolation::Linear)]);
        c.insert(5.0, 3.0, Interpolation::Linear);
        c.insert(10.0, 7.0, Interpolation::Linear);
        let frames: Vec<f64> = c.keys.iter().map(|k| k.frame).collect();
        assert_eq!(frames, vec![1.0, 5.0, 10.0]);
        assert_eq!(c.keys[2].value, 7.0, "a second key at a frame stacked instead of replacing");
        assert!(c.remove(5.0));
        assert!(!c.remove(5.0));
        assert_eq!(c.keys.len(), 2);
    }

    #[test]
    fn each_interpolation_travels_the_way_it_says() {
        use Interpolation::*;
        let at = |i| curve(&[(0.0, 0.0, i), (10.0, 10.0, i)]).evaluate(2.5).unwrap();
        assert_eq!(at(Constant), 0.0);
        assert!((at(Linear) - 2.5).abs() < 1e-12);
        // Eased: slower than linear leaving the first key.
        let eased = at(Bezier);
        assert!(eased < 2.5 && eased > 0.0, "{eased}");
        // And it passes through every key exactly.
        let c = curve(&[(0.0, 0.0, Bezier), (5.0, 4.0, Bezier), (10.0, 1.0, Bezier)]);
        for k in &c.keys {
            assert!((c.evaluate(k.frame).unwrap() - k.value).abs() < 1e-12);
        }
    }

    #[test]
    fn before_the_first_key_and_after_the_last_the_value_holds() {
        let c = curve(&[(5.0, 2.0, Interpolation::Linear), (9.0, 6.0, Interpolation::Linear)]);
        assert_eq!(c.evaluate(-100.0), Some(2.0));
        assert_eq!(c.evaluate(1000.0), Some(6.0));
        assert_eq!(FCurve::new(c.path.clone()).evaluate(3.0), None);
    }

    #[test]
    fn a_peak_is_not_overshot() {
        // The clamp. A ball keyed on the floor, in the air, on the floor must
        // never go below the floor or above its highest key between them.
        let c = curve(&[
            (0.0, 0.0, Interpolation::Bezier),
            (10.0, 5.0, Interpolation::Bezier),
            (20.0, 0.0, Interpolation::Bezier),
            (22.0, 0.3, Interpolation::Bezier),
        ]);
        for step in 0..=220 {
            let v = c.evaluate(step as f64 / 10.0).unwrap();
            assert!((-1e-9..=5.0 + 1e-9).contains(&v), "frame {}: {v}", step as f64 / 10.0);
        }
    }

    #[test]
    fn euler_angles_go_there_and_back() {
        for angles in [
            DVec3::new(0.3, -0.2, 1.1),
            DVec3::new(-1.0, 0.4, 0.0),
            DVec3::new(0.0, 0.0, -2.5),
        ] {
            let back = to_euler(from_euler(angles));
            assert!((back - angles).length() < 1e-9, "{angles:?} came back as {back:?}");
        }
        // X first, then Y, then Z: rotating +X by 90° about Z after nothing
        // else lands on +Y.
        let q = from_euler(DVec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
        assert!((q * DVec3::X - DVec3::Y).length() < 1e-9);
    }

    #[test]
    fn playing_keyed_components_leaves_the_rest_alone() {
        let mut action = Action::new("Move");
        action.curve_mut("location", 0).insert(1.0, 0.0, Interpolation::Linear);
        action.curve_mut("location", 0).insert(11.0, 10.0, Interpolation::Linear);
        let mut t = DTransform {
            translation: DVec3::new(99.0, 5.0, 6.0),
            ..DTransform::IDENTITY
        };
        apply_to_object(&action, 6.0, &mut t);
        assert!((t.translation.x - 5.0).abs() < 1e-12);
        assert_eq!((t.translation.y, t.translation.z), (5.0, 6.0));
        assert_eq!(t.rotation, DQuat::IDENTITY, "an unkeyed rotation was rewritten");
    }

    #[test]
    fn playback_loops_at_the_end_of_the_range() {
        let p = Playback { frame: 249.0, start: 1, end: 250, fps: 24.0 };
        // One frame on: 250, still inside.
        assert!((p.advanced(1.0 / 24.0) - 250.0).abs() < 1e-9);
        // Two on: past the end, back to the start.
        assert!((p.advanced(2.0 / 24.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_file_with_keys_out_of_order_is_put_right() {
        let mut c = curve(&[]);
        c.keys = vec![
            Keyframe { frame: 5.0, value: 1.0, interpolation: Interpolation::Linear, selected: false },
            Keyframe { frame: 1.0, value: 0.0, interpolation: Interpolation::Linear, selected: false },
            Keyframe { frame: 5.0, value: 9.0, interpolation: Interpolation::Linear, selected: false },
            Keyframe { frame: f64::NAN, value: 0.0, interpolation: Interpolation::Linear, selected: false },
        ];
        c.normalise();
        let frames: Vec<f64> = c.keys.iter().map(|k| k.frame).collect();
        assert_eq!(frames, vec![1.0, 5.0]);
    }

    fn two_curve_action() -> Action {
        let mut a = Action::new("A");
        for (f, v) in [(1.0, 0.0), (10.0, 5.0), (20.0, 2.0)] {
            a.curve_mut("location", 0).insert(f, v, Interpolation::Linear);
            a.curve_mut("scale", 1).insert(f, v, Interpolation::Linear);
        }
        a.select_all_with(|_| false);
        a
    }

    fn selected(a: &Action, property: &str) -> Vec<f64> {
        a.curves
            .iter()
            .filter(|c| c.path.property.as_ref() == property)
            .flat_map(|c| c.keys.iter().filter(|k| k.selected).map(|k| k.frame))
            .collect()
    }

    #[test]
    fn a_range_selects_only_its_property_and_its_frames() {
        let mut a = two_curve_action();
        assert!(a.select_range(Some("location"), 9.5, 20.5, true));
        assert_eq!(selected(&a, "location"), vec![10.0, 20.0]);
        assert!(selected(&a, "scale").is_empty(), "a location row selected scale keys");
        assert!(!a.select_range(Some("location"), 9.5, 20.5, true), "reselecting reported a change");
    }

    #[test]
    fn moving_selected_keys_replaces_what_they_land_on_and_keeps_order() {
        let mut a = two_curve_action();
        a.select_range(None, 10.0, 10.0, true);
        assert!(a.move_selected(10.0));
        for curve in &a.curves {
            let frames: Vec<f64> = curve.keys.iter().map(|k| k.frame).collect();
            assert_eq!(frames, vec![1.0, 20.0], "{}", curve.path.property);
            // The key that moved is the one that stayed at 20: value 5, not 2.
            assert_eq!(curve.keys[1].value, 5.0);
        }
    }

    #[test]
    fn deleting_the_selection_leaves_the_rest_and_drops_empty_curves() {
        let mut a = two_curve_action();
        a.select_range(Some("scale"), 0.0, 100.0, true);
        a.select_range(Some("location"), 1.0, 1.0, true);
        assert!(a.delete_selected());
        assert_eq!(a.curves.len(), 1, "the emptied scale curve stayed");
        assert_eq!(a.curves[0].keys.len(), 2);
        assert!(!a.delete_selected(), "nothing selected, and something was deleted");
    }

    #[test]
    fn interpolation_changes_only_the_selected_keys() {
        let mut a = two_curve_action();
        a.select_range(None, 1.0, 1.0, true);
        assert!(a.set_selected_interpolation(Interpolation::Constant));
        for curve in &a.curves {
            assert_eq!(curve.keys[0].interpolation, Interpolation::Constant);
            assert_eq!(curve.keys[1].interpolation, Interpolation::Linear);
        }
    }
}

