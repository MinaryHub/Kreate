//! The interactive transform: press a key, move the mouse, confirm or cancel.
//!
//! # The shape §2.2 asks for
//!
//! `M`, `R` and `S` do not move anything by themselves. They *start* a gesture:
//! from then until the user confirms, the pointer drives the operator's
//! parameters and the viewport shows the result. Confirm commits one history
//! entry; cancel restores and records nothing.
//!
//! Two rules do most of the work, and both are §2.2's:
//!
//! **Always `original → transform → current`, never incremental.** Every mouse
//! move computes the parameters afresh from where the drag *started*, and
//! `History::update_modal` restores the document before applying them. A long
//! drag therefore accumulates no drift, a cancel is one restore, and no operator
//! needs an incremental update path.
//!
//! **The mouse and the keyboard end in the same place.** A gesture produces a
//! `PropertyMap`, and that map is the whole of what the operator sees. Confirming
//! a drag is exactly equivalent to having typed those numbers — which is what
//! makes the adjust panel, scripting and macro replay work without any of them
//! knowing a pointer was involved.
//!
//! # Why the mapping lives in a pure function
//!
//! [`Gesture::params`] reads the gesture's own state and a camera and returns a
//! parameter map. No document, no event loop, no window. That is what lets the
//! feel of the tool — a drag being 1:1 with the cursor, a rotation following the
//! pointer around the pivot, a scale being the ratio of two screen distances —
//! be tested by arithmetic rather than by a person moving a mouse and squinting.
//!
//! # Why the pointer is *tracked* rather than sampled
//!
//! [`Gesture::track`] is called once per pointer event and `params` reads what it
//! accumulated. Move and scale would be happy with a bare cursor position, but
//! rotation is not: the difference of two `atan2` results can only ever express
//! half a turn, so a drag that keeps going round has to have its whole turns
//! counted as it goes.
//!
//! Taking the angle from the start position each frame instead looks simpler and
//! is wrong — found by a test that swept three quarters of a turn and watched the
//! angle flip from +175 degrees to −185. Half a turn is not an unreasonable
//! rotation to ask for.
//!
//! # What is deliberately not here yet
//!
//! The local-axis cycle (`X`, `XX`, `X` again), expression and unit entry
//! (`2*3.5`, `30deg`), snapping, and proportional editing. Each is named in §2.2
//! and each is additive: they change what `params` returns, not the structure
//! around it. The axis constraint, precision hold and plain numeric entry are
//! here, because they are what a transform is unusable without.

use glam::{DVec3, Vec3};
use kreate_ops::{PropertyMap, PropertyValue};
use kreate_render::Camera;

/// Which transform a gesture is driving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Move,
    Rotate,
    Scale,
}

impl Kind {
    /// The operator to run, which depends on what is being edited.
    pub fn op_id(self, edit_mode: bool) -> &'static str {
        match (self, edit_mode) {
            (Kind::Move, false) => "object.translate",
            (Kind::Rotate, false) => "object.rotate",
            (Kind::Scale, false) => "object.scale",
            (Kind::Move, true) => "mesh.translate",
            (Kind::Rotate, true) => "mesh.rotate",
            (Kind::Scale, true) => "mesh.scale",
        }
    }

    /// The name for the status line.
    pub fn label(self) -> &'static str {
        match self {
            Kind::Move => "Move",
            Kind::Rotate => "Rotate",
            Kind::Scale => "Scale",
        }
    }
}

/// A world axis the gesture is constrained to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn direction(self) -> Vec3 {
        match self {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }

    /// The index the rotate and scale operators use for an axis.
    pub fn index(self) -> u32 {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Axis::X => "X",
            Axis::Y => "Y",
            Axis::Z => "Z",
        }
    }
}

/// How much finer a held precision modifier makes the gesture.
///
/// A tenth, measured from the moment the modifier went down rather than from the
/// start of the drag — so pressing it does not make the selection jump back
/// towards the origin. §2.2 is explicit about that, and it is the difference
/// between a precision mode that is usable and one that is a surprise.
const PRECISION: f32 = 0.1;

/// One interactive transform in progress.
#[derive(Clone, Debug)]
pub struct Gesture {
    pub kind: Kind,
    /// Where the pointer was when the gesture began, in viewport pixels.
    start: (f32, f32),
    /// The selection's pivot in world space: what a rotation turns about and a
    /// scale grows from, and the point a screen-space angle is measured around.
    pivot: DVec3,
    /// The pivot's position on screen, in the same pixel space as `start`.
    pivot_px: (f32, f32),
    /// How many world units one pixel spans at the pivot's depth.
    ///
    /// Computed once, at the start. Recomputing it per frame would make a move
    /// along the view direction change its own sensitivity mid-drag.
    units_per_pixel: f32,
    /// The camera's right and up in world space, for a free move.
    right: Vec3,
    up: Vec3,
    pub axis: Option<Axis>,
    /// Where precision mode was engaged, if it is engaged.
    precise_from: Option<(f32, f32)>,
    /// Motion accumulated before precision mode was engaged.
    ///
    /// Held separately so that releasing and re-engaging the modifier does not
    /// discard what came before it.
    coarse: (f32, f32),
    /// Digits the user has typed, if any. Overrides the pointer entirely.
    pub numeric: String,
    /// Where the pointer is now, as last reported to [`Gesture::track`].
    cursor: (f32, f32),
    /// The total angle swept around the pivot, whole turns included.
    ///
    /// Accumulated rather than recomputed, because a single `atan2` difference
    /// cannot express more than half a turn in either direction.
    swept: f64,
    /// The pointer's bearing from the pivot at the last `track`, for the next
    /// increment.
    bearing: Option<f64>,
}

impl Gesture {
    /// Starts a gesture. `cursor` and the viewport are in the same pixel space.
    ///
    /// `pivot` is where the selection is: the median of the selected objects or
    /// mesh elements. The caller computes it, because what counts as the
    /// selection differs between object and edit mode and this module should not
    /// have an opinion about the document.
    pub fn begin(
        kind: Kind,
        cursor: (f32, f32),
        pivot: DVec3,
        camera: &Camera,
        viewport: (f32, f32),
    ) -> Self {
        // The world size of one pixel at the pivot's distance. This is what makes
        // a free move track the cursor: drag the mouse a hundred pixels and the
        // selection moves a hundred pixels' worth of world.
        let distance = (pivot - camera.eye()).length().max(1e-6);
        let height = viewport.1.max(1.0) as f64;
        let units_per_pixel = (2.0 * (camera.fov_y * 0.5).tan() * distance / height) as f32;

        let pivot_px =
            project(pivot, camera, viewport).unwrap_or((viewport.0 * 0.5, viewport.1 * 0.5));

        let mut gesture = Self {
            kind,
            start: cursor,
            pivot,
            pivot_px,
            units_per_pixel,
            right: camera.right().as_vec3(),
            up: camera.up().as_vec3(),
            axis: None,
            precise_from: None,
            coarse: (0.0, 0.0),
            numeric: String::new(),
            cursor,
            swept: 0.0,
            bearing: None,
        };
        // Seeded from the starting position, so the *first* motion counts. Left
        // unseeded, the first increment has nothing to measure against and is
        // silently discarded — which for a rotation means the initial movement
        // of every gesture does nothing.
        gesture.accumulate();
        gesture
    }

    /// Follows the pointer. Called once per motion event, before reading
    /// [`Gesture::params`].
    ///
    /// Everything a gesture means is derived from what this accumulates, which
    /// is what lets a rotation pass half a turn and keep going.
    pub fn track(&mut self, cursor: (f32, f32)) {
        self.cursor = cursor;
        self.accumulate();
    }

    /// Adds the increment since the last bearing to the swept total.
    ///
    /// Also called after a change that moves the *effective* pointer without the
    /// physical one moving — engaging precision mode, say — so that the bearing
    /// and the total never disagree.
    fn accumulate(&mut self) {
        let (dx, dy) = self.effective_delta(self.cursor);
        let arm = (
            self.start.0 + dx - self.pivot_px.0,
            self.start.1 + dy - self.pivot_px.1,
        );
        // Too close to the pivot to have a bearing at all. Left alone rather
        // than guessed at: an `atan2` of noise here would spin the selection.
        if arm.0.hypot(arm.1) < 1.0 {
            return;
        }
        let bearing = (arm.1 as f64).atan2(arm.0 as f64);
        if let Some(previous) = self.bearing {
            // Screen Y grows downwards, so a sweep that looks anticlockwise is a
            // decreasing `atan2` and has to be negated to turn the way it looks.
            self.swept += -shortest(bearing - previous);
        }
        self.bearing = Some(bearing);
    }

    /// Toggles an axis constraint, as pressing the same key twice releases it.
    pub fn toggle_axis(&mut self, axis: Axis) {
        self.axis = if self.axis == Some(axis) {
            None
        } else {
            Some(axis)
        };
    }

    /// Engages or releases precision mode at the current cursor.
    pub fn set_precise(&mut self, on: bool, cursor: (f32, f32)) {
        self.cursor = cursor;
        match (on, self.precise_from) {
            (true, None) => {
                // Bank what has happened so far, so the selection does not jump.
                self.coarse = self.raw_delta(cursor);
                self.precise_from = Some(cursor);
            }
            (false, Some(_)) => {
                self.coarse = self.effective_delta(cursor);
                self.precise_from = None;
                // The banked total is now measured from the start again, so the
                // start moves to wherever the cursor is.
                self.start = (cursor.0 - self.coarse.0, cursor.1 - self.coarse.1);
                self.coarse = (0.0, 0.0);
            }
            _ => {}
        }
        // Neither transition changes where the selection is, so neither may add
        // to the swept angle. The bearing is re-seeded from the new arithmetic
        // instead: without this, the first increment after the transition is
        // measured against a bearing computed under the old one, and the
        // selection jumps.
        self.bearing = None;
        self.accumulate();
    }

    /// Accepts a typed character for the numeric buffer. `true` if it was taken.
    pub fn type_character(&mut self, c: char) -> bool {
        if c.is_ascii_digit() || c == '.' || (c == '-' && self.numeric.is_empty()) {
            self.numeric.push(c);
            true
        } else {
            false
        }
    }

    /// Removes the last typed character.
    pub fn backspace(&mut self) {
        self.numeric.pop();
    }

    /// The pointer displacement, ignoring precision mode.
    fn raw_delta(&self, cursor: (f32, f32)) -> (f32, f32) {
        (cursor.0 - self.start.0, cursor.1 - self.start.1)
    }

    /// The displacement that drives the transform, precision applied.
    fn effective_delta(&self, cursor: (f32, f32)) -> (f32, f32) {
        match self.precise_from {
            None => self.raw_delta(cursor),
            Some(from) => (
                self.coarse.0 + (cursor.0 - from.0) * PRECISION,
                self.coarse.1 + (cursor.1 - from.1) * PRECISION,
            ),
        }
    }

    /// The typed value, if the buffer holds a usable number.
    fn typed(&self) -> Option<f64> {
        if self.numeric.is_empty() || self.numeric == "-" || self.numeric == "." {
            return None;
        }
        self.numeric.parse().ok()
    }

    /// The operator parameters this gesture currently means.
    ///
    /// Pure: a cursor position and a camera in, a parameter map out. No document
    /// is consulted and none is changed, which is what makes the feel of each
    /// tool testable as arithmetic.
    pub fn params(&self) -> PropertyMap {
        let mut params = PropertyMap::new();
        let (dx, dy) = self.effective_delta(self.cursor);

        match self.kind {
            Kind::Move => {
                let offset = match (self.axis, self.typed()) {
                    // A typed distance along a chosen axis is exact.
                    (Some(axis), Some(value)) => axis.direction() * value as f32,
                    // Typed with no axis: along the axis the drag most resembles,
                    // so that typing a number after starting to drag does what
                    // the drag was already doing.
                    (None, Some(value)) => self.dominant_axis(dx, dy).direction() * value as f32,
                    (Some(axis), None) => {
                        // The pointer's motion projected onto the axis, so the
                        // selection follows the cursor as closely as a
                        // constrained move can.
                        let free = self.screen_to_world(dx, dy);
                        axis.direction() * free.dot(axis.direction())
                    }
                    // Free: straight across the view plane, one pixel to one
                    // pixel's worth of world.
                    (None, None) => self.screen_to_world(dx, dy),
                };
                params.set(
                    "translation",
                    PropertyValue::Vec3([offset.x as f64, offset.y as f64, offset.z as f64]),
                );
            }
            Kind::Rotate => {
                let axis = self.axis.unwrap_or_else(|| self.view_axis());
                let angle = match self.typed() {
                    Some(degrees) => degrees.to_radians(),
                    // The angle the pointer has swept around the pivot on
                    // screen, whole turns included. Following the pointer around
                    // the pivot rather than reading horizontal motion: it is what
                    // the gesture looks like it should do, and it keeps working
                    // when the pivot is off to one side.
                    None => self.swept,
                };
                params.set("angle", PropertyValue::Float(angle));
                params.set("axis", PropertyValue::Enum(axis.index()));
            }
            Kind::Scale => {
                let factor = match self.typed() {
                    Some(value) => value,
                    // The ratio of the pointer's distance from the pivot now to
                    // what it was at the start. Dragging away grows, dragging
                    // through the pivot shrinks towards nothing.
                    None => self.distance_ratio() as f64,
                };
                let scale = match self.axis {
                    None => [factor, factor, factor],
                    Some(Axis::X) => [factor, 1.0, 1.0],
                    Some(Axis::Y) => [1.0, factor, 1.0],
                    Some(Axis::Z) => [1.0, 1.0, factor],
                };
                params.set("scale", PropertyValue::Vec3(scale));
            }
        }
        params
    }

    /// A screen displacement as a world vector across the view plane.
    fn screen_to_world(&self, dx: f32, dy: f32) -> Vec3 {
        // Y is negated because window coordinates grow downwards and the
        // camera's up does not. Getting this wrong is the classic "dragging up
        // moves it down" bug.
        self.right * (dx * self.units_per_pixel) - self.up * (dy * self.units_per_pixel)
    }

    /// Whichever world axis the drag is closest to, for a typed free move.
    fn dominant_axis(&self, dx: f32, dy: f32) -> Axis {
        let world = self.screen_to_world(dx, dy);
        let (x, y, z) = (world.x.abs(), world.y.abs(), world.z.abs());
        if x >= y && x >= z {
            Axis::X
        } else if y >= z {
            Axis::Y
        } else {
            Axis::Z
        }
    }

    /// The world axis most nearly aligned with the view direction.
    ///
    /// The default for a rotation with no explicit axis: turning about the axis
    /// pointing at the viewer is what a circular drag looks like it is doing.
    fn view_axis(&self) -> Axis {
        let forward = self.right.cross(self.up);
        let (x, y, z) = (forward.x.abs(), forward.y.abs(), forward.z.abs());
        if x >= y && x >= z {
            Axis::X
        } else if y >= z {
            Axis::Y
        } else {
            Axis::Z
        }
    }

    /// How much further from the pivot the pointer is than when it started.
    fn distance_ratio(&self) -> f32 {
        let (dx, dy) = self.effective_delta(self.cursor);
        let then = (self.start.0 - self.pivot_px.0).hypot(self.start.1 - self.pivot_px.1);
        // A gesture started on top of the pivot has no reference length, so the
        // ratio is undefined. Held at 1 rather than exploding.
        if then < 1.0 {
            return 1.0;
        }
        let now = (self.start.0 + dx - self.pivot_px.0).hypot(self.start.1 + dy - self.pivot_px.1);
        now / then
    }

    /// What the status line says while the gesture runs.
    pub fn status(&self) -> String {
        let axis = match self.axis {
            Some(axis) => format!(" {}", axis.label()),
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
                let (dx, dy) = self.effective_delta(self.cursor);
                let v = match (self.axis, self.typed()) {
                    (Some(a), Some(t)) => a.direction() * t as f32,
                    (None, Some(t)) => self.dominant_axis(dx, dy).direction() * t as f32,
                    (Some(a), None) => {
                        a.direction() * self.screen_to_world(dx, dy).dot(a.direction())
                    }
                    (None, None) => self.screen_to_world(dx, dy),
                };
                format!("{:.3} {:.3} {:.3}", v.x, v.y, v.z)
            }
            Kind::Rotate => {
                let radians = self.typed().map(f64::to_radians).unwrap_or(self.swept);
                format!("{:.1} deg", radians.to_degrees())
            }
            Kind::Scale => {
                let f = self.typed().unwrap_or_else(|| self.distance_ratio() as f64);
                format!("{f:.3}")
            }
        };
        format!("{}{axis}  {value}{typed}{precise}", self.kind.label())
    }

    /// The pivot, for a caller that wants to draw it.
    pub fn pivot(&self) -> DVec3 {
        self.pivot
    }
}

/// The shortest signed way round between two bearings.
///
/// Used on each *increment*, never on a total. Two `atan2` results differ by the
/// angle between them only if the difference is taken the short way; applied to a
/// whole gesture instead, this is what makes a sweep past half a turn flip sign
/// and the selection snap through a revolution.
fn shortest(mut angle: f64) -> f64 {
    while angle > std::f64::consts::PI {
        angle -= std::f64::consts::TAU;
    }
    while angle < -std::f64::consts::PI {
        angle += std::f64::consts::TAU;
    }
    angle
}

/// A world point in viewport pixels, or `None` when it is behind the camera.
fn project(point: DVec3, camera: &Camera, viewport: (f32, f32)) -> Option<(f32, f32)> {
    let aspect = (viewport.0 / viewport.1.max(1.0)) as f64;
    let clip = camera.view_projection(aspect)
        * glam::Vec4::new(
            (point.x - camera.eye().x) as f32,
            (point.y - camera.eye().y) as f32,
            (point.z - camera.eye().z) as f32,
            1.0,
        );
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some((
        (ndc.x * 0.5 + 0.5) * viewport.0,
        (0.5 - ndc.y * 0.5) * viewport.1,
    ))
}
