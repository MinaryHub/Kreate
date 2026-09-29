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
    /// A slope set by hand, in value per frame, for a Bézier key — the graph
    /// editor's handle. `None` is auto-clamped: derived from the neighbours.
    ///
    /// A slope and not a pair of handle points: the handles stay a third of
    /// the way to each neighbour, as auto handles are, so the curve is still a
    /// cubic in the frame and one number says everything a person dragging a
    /// handle changes. Blender's handles can also be lengthened; that is the
    /// weighted-tangent feature, and it is not this one.
    pub slope: Option<f64>,
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
        let key = Keyframe { frame, value, interpolation, selected: true, slope: None };
        match self.keys.iter().position(|k| (k.frame - frame).abs() < SAME_FRAME) {
            // A handle somebody set survives the key being set again: keying a
            // new value is not a request to forget how the curve bends there.
            Some(at) => self.keys[at] = Keyframe { slope: self.keys[at].slope, ..key },
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
    pub fn slope(&self, i: usize) -> f64 {
        if let Some(slope) = self.keys.get(i).and_then(|k| k.slope) {
            return slope;
        }
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
#[derive(Clone, Debug, PartialEq)]
pub struct Action {
    pub name: String,
    pub linkage: Linkage,
    /// Assigned by the document. See [`Uid`].
    pub uid: Uid,
    /// What kind of datablock plays it: an object's transform, a material's
    /// surface, a light's colour and brightness. Its curves' properties are
    /// that kind's — see [`animatable`].
    pub animates: DatablockKind,
    pub curves: Vec<FCurve>,
}

impl Default for Action {
    fn default() -> Self {
        Self {
            name: String::new(),
            linkage: Linkage::Local,
            uid: Uid::NONE,
            animates: DatablockKind::Object,
            curves: Vec::new(),
        }
    }
}

impl Action {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// An action for a kind of datablock other than an object.
    pub fn for_kind(name: impl Into<String>, animates: DatablockKind) -> Self {
        Self {
            animates,
            ..Self::new(name)
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
                self.curves.push(FCurve::new(PropertyPath::component(
                    self.animates,
                    Uid::NONE,
                    property,
                    index,
                )));
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

    /// Selects the keys of one component of one property within a frame
    /// range and a value range: the graph editor's box, where a key is a point
    /// on a plane and not a column on a row.
    pub fn select_region(
        &mut self,
        property: &str,
        index: u8,
        frames: (f64, f64),
        values: (f64, f64),
    ) -> bool {
        let mut changed = false;
        for curve in self
            .curves
            .iter_mut()
            .filter(|c| c.path.property.as_ref() == property && c.path.index == Some(index))
        {
            for key in &mut curve.keys {
                let inside = key.frame >= frames.0 - 1e-9
                    && key.frame <= frames.1 + 1e-9
                    && key.value >= values.0
                    && key.value <= values.1;
                if inside && !key.selected {
                    key.selected = true;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Changes the value of the selected keys by `by`.
    pub fn offset_selected_values(&mut self, by: f64) -> bool {
        let mut changed = false;
        for curve in &mut self.curves {
            for key in curve.keys.iter_mut().filter(|k| k.selected) {
                key.value += by;
                changed |= by != 0.0;
            }
        }
        changed
    }

    /// Sets the hand-set slope of the selected keys: `None` for auto.
    pub fn set_selected_slope(&mut self, slope: Option<f64>) -> bool {
        let mut changed = false;
        for curve in &mut self.curves {
            for key in curve.keys.iter_mut().filter(|k| k.selected) {
                changed |= key.slope != slope;
                key.slope = slope;
            }
        }
        changed
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

/// The properties of a material a curve can drive.
///
/// The flat surface's numbers. A material that uses a node graph is shaded by
/// the graph, so animating these changes what it would be without one — the
/// graph's own values are keyed when the node editor learns to key them.
pub const MATERIAL_ANIMATABLE: &[AnimatableDesc] = &[
    AnimatableDesc { name: "base_color", kind: PropertyKind::Color, components: 3, doc: "The surface's colour, linear." },
    AnimatableDesc { name: "metallic", kind: PropertyKind::Float, components: 1, doc: "Metal or not." },
    AnimatableDesc { name: "roughness", kind: PropertyKind::Float, components: 1, doc: "How blurred reflections are." },
    AnimatableDesc { name: "alpha", kind: PropertyKind::Float, components: 1, doc: "Opacity." },
    AnimatableDesc { name: "normal_strength", kind: PropertyKind::Float, components: 1, doc: "How far the normal map tilts the surface." },
];

/// The properties of a light a curve can drive.
pub const LIGHT_ANIMATABLE: &[AnimatableDesc] = &[
    AnimatableDesc { name: "color", kind: PropertyKind::Color, components: 3, doc: "The light's tint, linear." },
    AnimatableDesc { name: "intensity", kind: PropertyKind::Float, components: 1, doc: "How much light, in the kind's unit." },
    AnimatableDesc { name: "exposure", kind: PropertyKind::Float, components: 1, doc: "Brightness in stops, on top of the intensity." },
];

/// What a kind of datablock can have animated.
pub fn animatable(kind: DatablockKind) -> &'static [AnimatableDesc] {
    match kind {
        DatablockKind::Object => OBJECT_ANIMATABLE,
        DatablockKind::Material => MATERIAL_ANIMATABLE,
        DatablockKind::Light => LIGHT_ANIMATABLE,
        _ => &[],
    }
}

/// A material's animatable value, as its components.
pub fn read_material(m: &crate::document::Material, property: &str) -> Option<Vec<f64>> {
    Some(match property {
        "base_color" => m.base_color.to_array().iter().map(|c| *c as f64).collect(),
        "metallic" => vec![m.metallic as f64],
        "roughness" => vec![m.roughness as f64],
        "alpha" => vec![m.alpha as f64],
        "normal_strength" => vec![m.normal_strength as f64],
        _ => return None,
    })
}

/// Plays `action` onto a material at `frame`.
pub fn apply_to_material(action: &Action, frame: f64, m: &mut crate::document::Material) {
    let mut colour = m.base_color.to_array();
    for curve in &action.curves {
        let (Some(i), Some(v)) = (curve.path.index, curve.evaluate(frame)) else { continue };
        let v = v as f32;
        match (curve.path.property.as_ref(), i) {
            ("base_color", 0..=2) => colour[i as usize] = v.max(0.0),
            ("metallic", 0) => m.metallic = v.clamp(0.0, 1.0),
            ("roughness", 0) => m.roughness = v.clamp(0.0, 1.0),
            ("alpha", 0) => m.alpha = v.clamp(0.0, 1.0),
            ("normal_strength", 0) => m.normal_strength = v.max(0.0),
            _ => {}
        }
    }
    m.base_color = kreate_math::SceneRgb::new(colour[0], colour[1], colour[2]);
}

/// A light's animatable value, as its components.
pub fn read_light(l: &crate::light::Light, property: &str) -> Option<Vec<f64>> {
    Some(match property {
        "color" => l.color.to_array().iter().map(|c| *c as f64).collect(),
        "intensity" => vec![l.intensity],
        "exposure" => vec![l.exposure],
        _ => return None,
    })
}

/// Plays `action` onto a light at `frame`.
pub fn apply_to_light(action: &Action, frame: f64, l: &mut crate::light::Light) {
    let mut colour = l.color.to_array();
    for curve in &action.curves {
        let (Some(i), Some(v)) = (curve.path.index, curve.evaluate(frame)) else { continue };
        match (curve.path.property.as_ref(), i) {
            ("color", 0..=2) => colour[i as usize] = (v as f32).max(0.0),
            // A light cannot give out less than nothing: a Bézier easing to
            // zero may undershoot, and a negative lamp darkens what it lights.
            ("intensity", 0) => l.intensity = v.max(0.0),
            ("exposure", 0) => l.exposure = v,
            _ => {}
        }
    }
    l.color = kreate_math::SceneRgb::new(colour[0], colour[1], colour[2]);
}

// --- what a pose animates ---------------------------------------------------

/// The properties of a bone's pose a curve can drive: Blender's pose-bone
/// channels, with the rotation a quaternion as Blender's default is — a bone
/// turns in every direction, and Euler angles lock up exactly where a
/// shoulder does.
pub const BONE_ANIMATABLE: &[AnimatableDesc] = &[
    AnimatableDesc { name: "location", kind: PropertyKind::Vec3, components: 3, doc: "Where the bone is moved from rest, in its own axes." },
    AnimatableDesc {
        name: "rotation_quaternion",
        kind: PropertyKind::Float,
        components: 4,
        doc: "How it is turned from rest, as W, X, Y, Z.",
    },
    AnimatableDesc { name: "scale", kind: PropertyKind::Vec3, components: 3, doc: "Its size along its own axes." },
];

/// The path a curve on a bone's pose has: Blender's, so it reads as the same
/// thing to anyone who has seen one — `pose.bones["Upper"].location`.
///
/// On the *object's* action, as Blender keeps them: one action animates a
/// character, its placement and its bones together.
pub fn bone_path(bone: &str, property: &str) -> String {
    format!("pose.bones[\"{bone}\"].{property}")
}

/// The bone and the property of a pose path, if it is one.
pub fn parse_bone_path(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_prefix("pose.bones[\"")?;
    let (bone, property) = rest.split_once("\"].")?;
    Some((bone, property))
}

/// What a curve's property is, on an action that animates `kind`: an entry of
/// [`animatable`], or of [`BONE_ANIMATABLE`] for a pose path.
pub fn describe(kind: DatablockKind, property: &str) -> Option<&'static AnimatableDesc> {
    if kind == DatablockKind::Object {
        if let Some((_, p)) = parse_bone_path(property) {
            return BONE_ANIMATABLE.iter().find(|d| d.name == p);
        }
    }
    animatable(kind).iter().find(|d| d.name == property)
}

/// A bone's pose values, as the curves see them. The quaternion is W, X, Y, Z.
pub fn read_bone(t: &DTransform, property: &str) -> Option<Vec<f64>> {
    Some(match property {
        "location" => t.translation.to_array().to_vec(),
        "rotation_quaternion" => vec![t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z],
        "scale" => t.scale.to_array().to_vec(),
        _ => return None,
    })
}

/// Plays the pose curves of `action` onto `pose` at `frame`. A bone nothing
/// keys is left as it is; a quaternion is normalised after its four curves
/// are read, since four numbers each moving in a line are not a rotation
/// between the keys.
pub fn apply_to_pose(action: &Action, frame: f64, pose: &mut crate::armature::Pose) {
    let mut turned: Vec<(String, [f64; 4])> = Vec::new();
    for curve in &action.curves {
        let Some((bone, property)) = parse_bone_path(&curve.path.property) else { continue };
        let (Some(index), Some(value)) = (curve.path.index, curve.evaluate(frame)) else { continue };
        let i = index as usize;
        match property {
            "location" if i < 3 => pose.entry(bone).transform.translation[i] = value,
            "scale" if i < 3 => pose.entry(bone).transform.scale[i] = value,
            "rotation_quaternion" if i < 4 => {
                let at = match turned.iter().position(|(b, _)| b == bone) {
                    Some(at) => at,
                    None => {
                        let q = pose.transform(bone).rotation;
                        turned.push((bone.to_string(), [q.w, q.x, q.y, q.z]));
                        turned.len() - 1
                    }
                };
                turned[at].1[i] = value;
            }
            _ => {}
        }
    }
    for (bone, [w, x, y, z]) in turned {
        let q = DQuat::from_xyzw(x, y, z, w);
        if q.length_squared() > 1e-24 {
            pose.entry(&bone).transform.rotation = q.normalize();
        }
    }
}

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
            Keyframe { frame: 5.0, value: 1.0, interpolation: Interpolation::Linear, selected: false, slope: None },
            Keyframe { frame: 1.0, value: 0.0, interpolation: Interpolation::Linear, selected: false, slope: None },
            Keyframe { frame: 5.0, value: 9.0, interpolation: Interpolation::Linear, selected: false, slope: None },
            Keyframe { frame: f64::NAN, value: 0.0, interpolation: Interpolation::Linear, selected: false, slope: None },
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

    #[test]
    fn a_hand_set_slope_bends_the_curve_and_auto_gives_it_back() {
        let mut c = curve(&[(0.0, 0.0, Interpolation::Bezier), (10.0, 0.0, Interpolation::Bezier)]);
        // Flat keys at the same value: auto is a straight line.
        assert!(c.evaluate(5.0).unwrap().abs() < 1e-12);
        c.keys[0].slope = Some(1.0);
        // Leaving upwards now, so the middle is above.
        assert!(c.evaluate(5.0).unwrap() > 0.5);
        assert_eq!(c.slope(0), 1.0);
        c.keys[0].slope = None;
        assert_eq!(c.slope(0), 0.0);
        // Keying the frame again keeps the handle.
        c.keys[0].slope = Some(2.0);
        c.insert(0.0, 3.0, Interpolation::Bezier);
        assert_eq!(c.keys[0].slope, Some(2.0));
        assert_eq!(c.keys[0].value, 3.0);
    }

    #[test]
    fn a_region_selects_by_value_as_well_as_frame() {
        let mut a = two_curve_action();
        // Keys of location X: (1,0) (10,5) (20,2). A box over values 1..6
        // between frames 5 and 25 takes 10 and 20, not 1.
        assert!(a.select_region("location", 0, (5.0, 25.0), (1.0, 6.0)));
        assert_eq!(selected(&a, "location"), vec![10.0, 20.0]);
        assert!(a.offset_selected_values(1.0));
        let x = a.curves.iter().find(|c| c.path.property.as_ref() == "location").unwrap();
        assert_eq!(x.keys.iter().map(|k| k.value).collect::<Vec<_>>(), vec![0.0, 6.0, 3.0]);
    }

    #[test]
    fn a_bone_path_reads_back_and_plays_onto_the_pose() {
        let path = bone_path("Upper Arm", "rotation_quaternion");
        assert_eq!(path, "pose.bones[\"Upper Arm\"].rotation_quaternion");
        assert_eq!(parse_bone_path(&path), Some(("Upper Arm", "rotation_quaternion")));
        assert_eq!(parse_bone_path("location"), None);
        assert_eq!(describe(DatablockKind::Object, &path).map(|d| d.components), Some(4));
        assert_eq!(describe(DatablockKind::Object, "location").map(|d| d.components), Some(3));

        // Half a turn about X over ten frames, keyed as a quaternion.
        let mut a = Action::new("Wave");
        let q1 = DQuat::from_rotation_x(1.0);
        for (i, v) in [1.0, 0.0, 0.0, 0.0].into_iter().enumerate() {
            a.curve_mut(&path, i as u8).insert(1.0, v, Interpolation::Linear);
        }
        for (i, v) in [q1.w, q1.x, q1.y, q1.z].into_iter().enumerate() {
            a.curve_mut(&path, i as u8).insert(11.0, v, Interpolation::Linear);
        }
        a.curve_mut(&bone_path("Upper Arm", "location"), 2).insert(1.0, 0.5, Interpolation::Linear);
        let mut pose = crate::armature::Pose::default();
        apply_to_pose(&a, 11.0, &mut pose);
        let t = pose.transform("Upper Arm");
        assert!(t.rotation.dot(q1).abs() > 1.0 - 1e-12);
        assert_eq!(t.translation.z, 0.5);
        // Between the keys it is still a rotation.
        apply_to_pose(&a, 6.0, &mut pose);
        assert!((pose.transform("Upper Arm").rotation.length() - 1.0).abs() < 1e-12);
        // And an object transform ignores the bone's curves.
        let mut t = DTransform::IDENTITY;
        apply_to_object(&a, 11.0, &mut t);
        assert_eq!(t, DTransform::IDENTITY);
    }
}

