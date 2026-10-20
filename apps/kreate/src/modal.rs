//! The interactive transform: press a key, move the mouse, confirm or cancel.
//!
//! # The shape §2.2 asks for
//!
//! `G`, `R` and `S` do not move anything by themselves. They *start* a gesture:
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
//! # The axis cycle
//!
//! Pressing an axis key walks `free -> global -> local -> free`. Two presses
//! rather than one because both orientations are wanted often and neither is
//! obviously the default: moving a wheel along the world's X is a different
//! request from moving it along the axle it is mounted on, and a modeller wants
//! whichever the object happens to be turned to.
//!
//! # Snapping
//!
//! Held [`Snap`] settings with `Ctrl` **inverting** them rather than enabling
//! them. §2.2 is specific about the direction: a user who has turned snapping on
//! wants `Ctrl` to let them off it for one drag, and a user who has it off wants
//! `Ctrl` to turn it on. One modifier, and it always means "the other way".
//!
//! # Proportional editing
//!
//! The radius rides along as a parameter, so the operator does the falloff. That
//! keeps this module free of the document — the gesture says "within this
//! radius, this curve" and the mesh operator is what knows where the vertices
//! are.
//!
//! Element snapping — to a vertex, an edge or a face — lands the selection *on*
//! something rather than rounding how far it went, so it replaces the offset
//! instead of adjusting it. The candidates come from [`crate::snap`], which
//! excludes everything the drag is about to move.

use glam::{DVec3, Vec3};
use kreate_ops::{PropertyMap, PropertyValue};
use kreate_render::Camera;

/// Which transform a gesture is driving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Move,
    Rotate,
    Scale,
    /// Blender's `E`: the selected faces extruded, and the new ones dragged out
    /// along their normal — the extrude's own `offset`, so the whole gesture is
    /// one operator and one step to undo.
    Extrude,
    /// `E` on edges or vertices: pulled out and moved freely, as `G` moves.
    ExtrudeFree,
    /// Blender's `I`: the pointer moving in towards the selection's middle
    /// makes the inset thicker.
    Inset,
    /// Blender's `Ctrl+B`: the pointer moving away from the selection's middle
    /// makes the chamfer wider.
    Bevel,
    /// Blender's `Ctrl+R`: the new loop slides along the selected edge with the
    /// pointer, and a typed number is how many loops.
    LoopCut,
    /// Blender's `G G`: the selected edges slide across the faces beside
    /// them, the pointer along the way they go; a typed number is the factor.
    EdgeSlide,
}

impl Kind {
    /// The operator to run, which depends on what is being edited.
    ///
    /// Exhaustive on the mode rather than taking a boolean. "Which operator
    /// does `G` run in this mode" is a real question that a new mode has to
    /// answer, and a boolean lets it be defaulted — silently, to whichever of
    /// the two it happens not to equal.
    pub fn op_id(self, mode: kreate_ops::operator::Mode) -> &'static str {
        let [translate, rotate, scale] = mode.info().transform;
        match self {
            Kind::Move => translate,
            Kind::Rotate => rotate,
            Kind::Scale => scale,
            // Not the mode's: only a mesh's faces extrude.
            Kind::Extrude => "mesh.extrude_region",
            Kind::ExtrudeFree => "mesh.extrude_move",
            Kind::Inset => "mesh.inset",
            Kind::Bevel => "mesh.bevel",
            Kind::LoopCut => "mesh.loop_cut",
            Kind::EdgeSlide => "mesh.edge_slide",
        }
    }

    /// The name for the status line.
    pub fn label(self) -> &'static str {
        match self {
            Kind::Move => "Move",
            Kind::Rotate => "Rotate",
            Kind::Scale => "Scale",
            Kind::Extrude | Kind::ExtrudeFree => "Extrude",
            Kind::Inset => "Inset",
            Kind::Bevel => "Bevel",
            Kind::LoopCut => "Loop Cut",
            Kind::EdgeSlide => "Edge Slide",
        }
    }

    /// Whether the gesture is one of the three transforms, which take an axis.
    pub fn transforms(self) -> bool {
        matches!(
            self,
            Kind::Move | Kind::Rotate | Kind::Scale | Kind::ExtrudeFree
        )
    }
}

/// Which set of axes a constraint is expressed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// The world's own axes.
    Global,
    /// The selection's: an object's own rotation, or in edit mode the object the
    /// elements belong to.
    Local,
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

/// The narrowest bevel a gesture asks for: where it starts, and the least it
/// goes back to. Far below anything visible.
const MIN_BEVEL: f64 = 1e-4;

/// The most loops one loop cut makes — the operator's own limit.
const MAX_CUTS: u32 = 64;

/// The degree sign, accepted in an angle field.
const DEGREE_SIGN: char = '\u{b0}';

/// What a gesture rounds its value to.
///
/// Increments rather than a grid in world space, because a transform is a
/// *change*: snapping a move to a tenth of a metre means the object ends up a
/// round distance from where it was, which is what someone laying out a panel
/// asks for. Snapping to absolute world positions is a different feature and
/// belongs with element snapping.
#[derive(Clone, Copy, Debug)]
pub struct Snap {
    /// Whether snapping is on by default. `Ctrl` inverts it.
    pub enabled: bool,
    /// What it snaps to.
    pub mode: crate::snap::Mode,
    /// Metres per step for a move.
    pub length: f64,
    /// Radians per step for a rotation. A sixth of a right angle by default,
    /// which is fifteen degrees — the increment a modeller reaches for.
    pub angle: f64,
    /// Steps per unit for a scale, so a tenth means 1.1, 1.2, and so on.
    pub factor: f64,
}

impl Default for Snap {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: crate::snap::Mode::Increment,
            length: 0.1,
            angle: std::f64::consts::FRAC_PI_2 / 6.0,
            factor: 0.1,
        }
    }
}

impl Snap {
    /// Rounds `value` to the nearest multiple of `step`.
    fn round(value: f64, step: f64) -> f64 {
        if step <= 0.0 || !value.is_finite() {
            return value;
        }
        (value / step).round() * step
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
    /// Which axes the constraint is in. Only meaningful while `axis` is set.
    pub orientation: Orientation,
    /// What the first press of an axis key means — the header's Transform
    /// Orientation. The second press is the other one, the third lets go.
    first_orientation: Orientation,
    /// The selection's own axes, for a local constraint.
    ///
    /// Captured at the start rather than read per frame: the transform may be
    /// rotating the very thing this describes, and a constraint that followed
    /// its own result would curve.
    basis: [Vec3; 3],
    /// The object's own axes, for an extrude: its `basis` is the faces'
    /// normal frame, and a local X, Y or Z means the object's.
    object_basis: [Vec3; 3],
    /// An extrude travelling along its faces' normal — what `E` starts as —
    /// rather than along an axis the user picked.
    along_normal: bool,
    /// Each face along its own normal (Alt+E ▸ Along Normals). Axis keys do
    /// nothing then: there is no one direction to replace.
    along_normals: bool,
    /// The selection has a region normal to measure the pointer along. A
    /// closed solid has none, and the pointer's distance from the middle is
    /// read instead.
    has_normal: bool,
    /// Snapping, as configured. `snap_inverted` flips it for this drag.
    pub snap: Snap,
    pub snap_inverted: bool,
    /// What may be snapped to: everything that is not moving.
    ///
    /// Built once, at the start. Nothing outside the selection moves during a
    /// transform, so the set cannot change — rebuilding it per frame would be
    /// the same answer at a hundred times the cost.
    world: crate::snap::SnapWorld,
    /// The points of what is moving, in the world, where they were when the
    /// drag began — Blender's snap base "Closest": the one nearest the target
    /// is what lands on it. Empty means the pivot does.
    sources: Vec<DVec3>,
    /// The point of what moves that lands on the target, when the user has
    /// picked one — Blender's `B` during a move: this corner onto that one.
    /// It replaces the nearest of `sources`.
    base: Option<DVec3>,
    /// While the base is being picked: what moves, where it was, as the
    /// candidates. The selection holds still meanwhile.
    picking: Option<crate::snap::SnapWorld>,
    /// The camera and viewport, for resolving a snap in screen space.
    ///
    /// Held rather than passed, so that `params` stays a function of the
    /// gesture's own state. The camera may orbit mid-drag; the copy here is
    /// refreshed by [`Gesture::set_view`] when it does.
    camera: Camera,
    viewport: (f32, f32),
    /// Proportional editing radius in world units, or `None` for off.
    pub proportional: Option<f64>,
    /// Its falloff curve.
    falloff: u32,
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
    /// A loop cut's edge on screen, first end to second — what the pointer
    /// slides the cut along. `None` for every other kind.
    slide: Option<((f32, f32), (f32, f32))>,
    /// The same edge's two ends in the world, for a distance along it.
    slide_world: Option<(DVec3, DVec3)>,
    /// The end a loop cut is measured from after `B`: `false` the first,
    /// `true` the second. While set, a typed number is a distance from it.
    pub anchor: Option<bool>,
    /// How many loops a loop cut makes, before anything is typed. The wheel
    /// changes it.
    pub cuts: u32,
    /// The edge a loop cut runs across when the pointer chose it rather than
    /// the selection — passed on to the operator, so the cut replays.
    pub edge: Option<u32>,
    /// An inset's depth, Blender's Ctrl: while Ctrl is held the thickness
    /// stays where it was and the pointer pushes the inset face in or out.
    /// The thickness it was held at, the pull it was held from, and the depth
    /// banked from earlier holds.
    depth_hold: Option<(f64, f64)>,
    depth: f64,
}

impl Gesture {
    /// Starts a gesture. `cursor` and the viewport are in the same pixel space.
    ///
    /// `pivot` is where the selection is: the median of the selected objects or
    /// mesh elements. The caller computes it, because what counts as the
    /// selection differs between object and edit mode and this module should not
    /// have an opinion about the document.
    // Eight arguments, and each is a distinct fact the gesture cannot work out
    // for itself: what it is, where the pointer is, where the selection is, how
    // the scene is being looked at, how big the viewport is, which way the
    // selection faces, how the user has snapping set, and what may be snapped
    // to. Bundling them into a struct would move the same eight words to a
    // different place and add a type whose only job is to be unpacked here.
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        kind: Kind,
        cursor: (f32, f32),
        pivot: DVec3,
        camera: &Camera,
        viewport: (f32, f32),
        basis: [Vec3; 3],
        snap: Snap,
        world: crate::snap::SnapWorld,
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
            orientation: Orientation::Global,
            first_orientation: Orientation::Global,
            basis,
            object_basis: basis,
            along_normal: false,
            along_normals: false,
            has_normal: true,
            snap,
            snap_inverted: false,
            world,
            sources: Vec::new(),
            base: None,
            picking: None,
            camera: *camera,
            viewport,
            proportional: None,
            falloff: 0,
            precise_from: None,
            coarse: (0.0, 0.0),
            numeric: String::new(),
            cursor,
            swept: 0.0,
            bearing: None,
            slide: None,
            slide_world: None,
            anchor: None,
            cuts: 1,
            edge: None,
            depth_hold: None,
            depth: 0.0,
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

    /// Walks the axis cycle: free, then global, then local, then free again.
    ///
    /// Pressing a *different* axis key starts that axis's cycle from global
    /// rather than inheriting the previous one's orientation — the second press
    /// is a new request, not a continuation.
    pub fn cycle_axis(&mut self, axis: Axis) {
        // An extrude starts along its faces' normal. X, Y or Z sends it along
        // that axis instead — the header's orientation first, the other one
        // on a second press — and a third press brings it back to the normal.
        if self.kind == Kind::Extrude && self.along_normals {
            return;
        }
        if self.kind == Kind::Extrude {
            let first = self.first_orientation;
            let second = match first {
                Orientation::Global => Orientation::Local,
                Orientation::Local => Orientation::Global,
            };
            match self.axis {
                Some(current)
                    if !self.along_normal && current == axis && self.orientation == first =>
                {
                    self.orientation = second;
                }
                Some(current) if !self.along_normal && current == axis => {
                    self.extrude_along_normal();
                }
                _ => {
                    self.along_normal = false;
                    self.axis = Some(axis);
                    self.orientation = first;
                }
            }
            return;
        }
        // An inset, a bevel and a loop cut have no direction to hold.
        if !self.kind.transforms() {
            return;
        }
        let first = self.first_orientation;
        let second = match first {
            Orientation::Global => Orientation::Local,
            Orientation::Local => Orientation::Global,
        };
        match self.axis {
            Some(current) if current == axis && self.orientation == first => {
                self.orientation = second;
            }
            Some(current) if current == axis => {
                self.axis = None;
                self.orientation = first;
            }
            _ => {
                self.axis = Some(axis);
                self.orientation = first;
            }
        }
    }

    /// The object's own axes, which a local constraint on an extrude follows.
    pub fn set_object_basis(&mut self, basis: [Vec3; 3]) {
        self.object_basis = basis;
    }

    /// Holds an extrude to its faces' normal: the local Z of the frame it was
    /// begun with.
    pub fn extrude_along_normal(&mut self) {
        self.along_normal = true;
        self.axis = Some(Axis::Z);
        self.orientation = Orientation::Local;
    }

    /// Makes an extrude move each face along its own normal. `has_normal` says
    /// whether the region has one direction the pointer can be measured along.
    pub fn extrude_along_normals(&mut self, has_normal: bool) {
        self.extrude_along_normal();
        self.along_normals = true;
        self.has_normal = has_normal;
    }

    /// Sets what the first press of an axis key constrains to.
    pub fn set_first_orientation(&mut self, orientation: Orientation) {
        self.first_orientation = orientation;
    }

    /// The proportional falloff curve, one of
    /// `kreate_ops::ops::mesh::FALLOFF_CURVES`.
    pub fn set_proportional_falloff(&mut self, curve: u32) {
        self.falloff = curve;
    }

    /// The axis a transform is held to, when it is one of the objects' own:
    /// what the operators take to apply each along its own axes. Not an
    /// extrude's normal frame, which is the faces', not the objects'.
    fn local_axis(&self) -> Option<Axis> {
        let axis = self.axis?;
        (self.orientation == Orientation::Local
            && matches!(self.kind, Kind::Move | Kind::Rotate | Kind::Scale))
        .then_some(axis)
    }

    /// The constrained direction in world space.
    fn axis_direction(&self, axis: Axis) -> Vec3 {
        match self.orientation {
            Orientation::Global => axis.direction(),
            // Normalised, because the basis comes from an object transform and a
            // scaled object's axes are not unit vectors. Without this a
            // constrained move on a scaled object travels the wrong distance.
            Orientation::Local => {
                // An extrude sent along a local axis means the object's, not
                // the faces' normal frame it was begun with.
                let basis = if self.kind == Kind::Extrude && !self.along_normal {
                    &self.object_basis
                } else {
                    &self.basis
                };
                basis[axis.index() as usize].normalize_or(axis.direction())
            }
        }
    }

    /// The edge a loop cut slides along, as its two ends in the world, first
    /// the end `position` 0 is at. Projected once, as the pivot is.
    pub fn set_slide_edge(&mut self, a: DVec3, b: DVec3) {
        self.slide =
            project(a, &self.camera, self.viewport).zip(project(b, &self.camera, self.viewport));
        self.slide_world = Some((a, b));
    }

    /// `B` in a loop cut: the cut snaps to the end of its edge nearer the
    /// pointer, and from there it is measured — a number typed now is how
    /// far from that end it cuts. Again, the other end.
    pub fn anchor_slide(&mut self) {
        let Some(((ax, ay), (bx, by))) = self.slide else {
            return;
        };
        self.anchor = Some(match self.anchor {
            Some(end) => !end,
            None => {
                let (cx, cy) = self.cursor;
                (cx - bx).hypot(cy - by) < (cx - ax).hypot(cy - ay)
            }
        });
        // The slide goes on from the end, from where the pointer is now.
        self.start = self.cursor;
        self.precise_from = None;
        self.coarse = (0.0, 0.0);
    }

    /// The loop cut's edge length in the world.
    fn slide_length(&self) -> Option<f64> {
        self.slide_world
            .map(|(a, b)| a.distance(b))
            .filter(|l| *l > 1e-12)
    }

    /// How far from the anchored end the cut is: typed, or the pointer's
    /// slide as a length — rounded to the increment when snapping.
    fn anchored_distance(&self, dx: f32, dy: f32) -> Option<f64> {
        let end = self.anchor?;
        let length = self.slide_length()?;
        if let Some(typed) = self.typed() {
            return Some(typed.clamp(0.0, length));
        }
        let along = self.slide_position_from(if end { 1.0 } else { 0.0 }, dx, dy);
        let from_end = if end { 1.0 - along } else { along } * length;
        let d = if self.snapping() {
            Snap::round(from_end, self.snap.length)
        } else {
            from_end
        };
        Some(d.clamp(0.0, length))
    }

    /// Where the loop cut sits along its edge, 0 at the first end: the
    /// anchored distance when there is one, otherwise the slide.
    fn cut_position(&self, dx: f32, dy: f32) -> f64 {
        match (
            self.anchor,
            self.anchored_distance(dx, dy),
            self.slide_length(),
        ) {
            (Some(end), Some(d), Some(length)) => {
                let f = d / length;
                if end {
                    1.0 - f
                } else {
                    f
                }
            }
            _ => self.slide_position(dx, dy),
        }
    }

    /// Adds `step` loops to a loop cut, one at the least — the wheel.
    pub fn add_cuts(&mut self, step: i32) {
        self.cuts = (self.cuts as i32 + step).clamp(1, MAX_CUTS as i32) as u32;
    }

    /// Updates the view the gesture resolves snaps against.
    ///
    /// Called when the camera moves mid-drag. Orbiting during a transform is
    /// allowed — looking at what you are doing is not a conflict — and a snap
    /// resolved against a stale camera would catch on whatever used to be under
    /// the cursor.
    pub fn set_view(&mut self, camera: &Camera, viewport: (f32, f32)) {
        self.camera = *camera;
        self.viewport = viewport;
    }

    /// Where the cursor is pointing, if a snap catches.
    ///
    /// `None` for increment snapping, which adjusts the value rather than
    /// choosing a destination, and for a cursor that is not near anything.
    fn snap_target(&self) -> Option<DVec3> {
        self.snap_hit().map(|hit| hit.point)
    }

    fn snap_hit(&self) -> Option<crate::snap::Hit> {
        if !self.snapping()
            || !self.snap.mode.needs_geometry()
            || !matches!(self.kind, Kind::Move | Kind::ExtrudeFree | Kind::Extrude)
        {
            return None;
        }
        crate::snap::resolve(
            &self.world,
            self.snap.mode,
            self.cursor,
            &self.camera,
            self.viewport,
        )
    }

    /// What a snap caught, for drawing a marker on it — or, while a base is
    /// being picked, the point of what moves the pointer is on.
    pub fn snap_point(&self) -> Option<(DVec3, crate::snap::Element)> {
        if self.picking.is_some() {
            return self.base_hit().map(|hit| (hit.point, hit.element));
        }
        if self.typed().is_some() {
            return None;
        }
        self.snap_hit().map(|hit| (hit.point, hit.element))
    }

    /// Whether this gesture can have its snap base picked: a move.
    pub fn takes_base(&self) -> bool {
        matches!(self.kind, Kind::Move | Kind::ExtrudeFree)
    }

    /// Whether the user is picking the snap base now.
    pub fn picking_base(&self) -> bool {
        self.picking.is_some()
    }

    /// Starts picking the snap base among `own` — the geometry that moves,
    /// where it was. The selection goes back to where it started and waits.
    pub fn begin_base_pick(&mut self, own: crate::snap::SnapWorld) {
        self.picking = Some(own);
    }

    /// The point of what moves under the pointer, by the element snap set
    /// (or any element, when the setting is increment).
    fn base_hit(&self) -> Option<crate::snap::Hit> {
        let own = self.picking.as_ref()?;
        let mode = if self.snap.mode.needs_geometry() {
            self.snap.mode
        } else {
            crate::snap::Mode::Element
        };
        crate::snap::resolve(own, mode, self.cursor, &self.camera, self.viewport)
    }

    /// Takes the point under the pointer as the snap base and goes on
    /// moving, snapping on: from here the base lands on what the pointer
    /// catches. Pointing at nothing keeps the base there was. The move starts
    /// again from where the pointer is, so nothing jumps.
    pub fn finish_base_pick(&mut self) -> bool {
        let hit = self.base_hit();
        self.picking = None;
        if let Some(hit) = &hit {
            self.base = Some(hit.point);
        }
        if self.snap.mode == crate::snap::Mode::Increment {
            self.snap.mode = crate::snap::Mode::Element;
        }
        if !self.snapping() {
            self.snap.enabled = !self.snap.enabled;
        }
        self.start = self.cursor;
        self.precise_from = None;
        self.coarse = (0.0, 0.0);
        hit.is_some()
    }

    /// Whether snapping wants geometry it has not been given yet — `B` in a
    /// drag that began with increment snapping has no candidates.
    pub fn wants_world(&self) -> bool {
        self.snapping() && self.snap.mode.needs_geometry() && self.world.is_empty()
    }

    /// The geometry to snap to, given after the start.
    pub fn set_world(&mut self, world: crate::snap::SnapWorld) {
        self.world = world;
    }

    /// The moving points a snap may land, where they were at the start.
    pub fn set_sources(&mut self, sources: Vec<DVec3>) {
        self.sources = sources;
    }

    /// The point of the selection that lands on `target`: the moving point
    /// nearest it as the pointer has carried things so far, or the pivot.
    fn snap_source(&self, target: DVec3, free: DVec3) -> DVec3 {
        if let Some(base) = self.base {
            return base;
        }
        self.sources
            .iter()
            .copied()
            .min_by(|a, b| {
                let da = (*a + free).distance_squared(target);
                let db = (*b + free).distance_squared(target);
                da.total_cmp(&db)
            })
            .unwrap_or(self.pivot)
    }

    /// Inverts the snap setting for this drag, as a held `Ctrl` does.
    pub fn set_snap_inverted(&mut self, inverted: bool) {
        // An inset has another use for Ctrl: depth, as in Blender.
        if self.kind == Kind::Inset {
            match (inverted, self.depth_hold) {
                (true, None) => self.depth_hold = Some((self.inset_thickness(), self.pull())),
                (false, Some(_)) => {
                    self.depth = self.inset_depth();
                    self.depth_hold = None;
                }
                _ => {}
            }
            return;
        }
        self.snap_inverted = inverted;
    }

    /// The inset's depth: what earlier holds banked, plus this hold's pull —
    /// the pointer moving out from the middle pushes the face out.
    fn inset_depth(&self) -> f64 {
        match self.depth_hold {
            Some((_, from)) => self.depth + (from - self.pull()),
            None => self.depth,
        }
    }

    /// Turns snapping on, or off again, for the rest of this drag — `B`.
    ///
    /// Blender's key during a transform, and the reason it is a toggle and
    /// not a hold: `Ctrl` is the hold. A modeller who wants to lay a row of
    /// panels down on the grid holds `Ctrl` for one and presses `B` for
    /// twenty, and it is the drag that remembers, not the scene — the setting
    /// in the header is what the next drag starts from.
    ///
    /// Turning it on when the setting is increment snaps to geometry instead —
    /// the vertex, edge or face under the pointer, as Blender's `B` does.
    pub fn toggle_snap(&mut self) {
        let on = !self.snapping();
        if on && self.snap.mode == crate::snap::Mode::Increment {
            self.snap.mode = crate::snap::Mode::Element;
        }
        // Ctrl may be inverting it; flip what decides, so B always flips
        // what the user sees.
        self.snap.enabled = !self.snap.enabled;
    }

    /// Whether snapping applies right now.
    fn snapping(&self) -> bool {
        self.snap.enabled != self.snap_inverted
    }

    /// Sets or clears the proportional editing radius.
    pub fn set_proportional(&mut self, radius: Option<f64>) {
        self.proportional = radius;
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
    ///
    /// Permissive, because the buffer holds an expression rather than a number:
    /// operators, brackets and unit letters all have to get in. Whether what has
    /// been typed *means* anything is decided when it is evaluated, which is
    /// also the only place that can decide — `2m` is fine and `2q` is not, and
    /// neither is knowable one character at a time.
    pub fn type_character(&mut self, c: char) -> bool {
        let usable = c.is_ascii_digit()
            || c.is_ascii_alphabetic()
            || matches!(c, '.' | '-' | '+' | '*' | '/' | '(' | ')' | ' ')
            || c == DEGREE_SIGN;
        if usable {
            self.numeric.push(c);
        }
        usable
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

    /// The typed value, if the buffer holds something that evaluates.
    ///
    /// Through the expression parser, so `2*3.5`, `1/8`, `2cm` and `30deg` are
    /// all accepted where a number is. A buffer that does not evaluate yet — or
    /// at all — hands control back to the pointer rather than reading as zero.
    fn typed(&self) -> Option<f64> {
        crate::expr::eval(&self.numeric, self.unit()).ok()
    }

    /// The unit the numeric field is in, which decides how a bare number and a
    /// suffix are read.
    fn unit(&self) -> crate::expr::Unit {
        match self.kind {
            Kind::Move | Kind::ExtrudeFree | Kind::Extrude | Kind::Inset | Kind::Bevel => {
                crate::expr::Unit::Length
            }
            Kind::Rotate => crate::expr::Unit::Angle,
            // Anchored, a loop cut's number is a distance from the end.
            Kind::LoopCut if self.anchor.is_some() => crate::expr::Unit::Length,
            Kind::Scale | Kind::LoopCut | Kind::EdgeSlide => crate::expr::Unit::Factor,
        }
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
            Kind::Move | Kind::ExtrudeFree => {
                let offset = self.offset(dx, dy);
                params.set(
                    "translation",
                    PropertyValue::Vec3([offset.x, offset.y, offset.z]),
                );
                // Held to a local axis: the distance along it, for each
                // object to take along its own — the world offset is only the
                // active object's.
                if let Some(axis) = self.local_axis() {
                    let along = offset.dot(self.axis_direction(axis).as_dvec3());
                    let mut own = [0.0; 3];
                    own[axis.index() as usize] = along;
                    params.set("orientation", PropertyValue::Enum(1));
                    params.set("local_translation", PropertyValue::Vec3(own));
                }
            }
            Kind::Extrude => {
                params.set(
                    "offset",
                    PropertyValue::Float(self.extrude_distance(dx, dy)),
                );
                if self.along_normals {
                    params.set("along_normals", PropertyValue::Bool(true));
                } else if !self.along_normal {
                    let d = self.extrude_direction().as_dvec3();
                    params.set("direction", PropertyValue::Vec3([d.x, d.y, d.z]));
                }
            }
            Kind::Inset => {
                let thickness = match self.depth_hold {
                    Some((held, _)) => held,
                    None => self.inset_thickness(),
                };
                params.set("thickness", PropertyValue::Float(thickness));
                params.set("depth", PropertyValue::Float(self.inset_depth()));
            }
            Kind::Bevel => {
                params.set("width", PropertyValue::Float(self.bevel_width()));
            }
            Kind::LoopCut => {
                params.set("cuts", PropertyValue::Int(self.loop_cuts() as i64));
                if let Some(edge) = self.edge {
                    params.set("edge", PropertyValue::Int(edge as i64));
                }
                params.set("position", PropertyValue::Float(self.cut_position(dx, dy)));
            }
            Kind::EdgeSlide => {
                let factor = self
                    .typed()
                    .unwrap_or_else(|| self.slide_factor(dx, dy))
                    .clamp(-1.0, 1.0);
                params.set("factor", PropertyValue::Float(factor));
            }
            Kind::Rotate => {
                let axis = self.axis.unwrap_or_else(|| self.view_axis());
                let angle = match self.typed() {
                    // Already in radians: the expression parser converts, so a
                    // bare `30` and `30deg` mean the same thing and `0.5rad`
                    // means what it says.
                    Some(radians) => radians,
                    // The angle the pointer has swept around the pivot on
                    // screen, whole turns included. Following the pointer around
                    // the pivot rather than reading horizontal motion: it is what
                    // the gesture looks like it should do, and it keeps working
                    // when the pivot is off to one side.
                    None => self.swept,
                };
                // Typed values are never snapped: someone who wrote a number
                // meant that number, and rounding it would be the tool arguing.
                // Angles snap to their increment whichever element mode is
                // set: there is no sensible "rotate until this vertex is on
                // that one" without a second reference point, and the increment
                // is what a rotation wants anyway.
                let angle = if self.snapping() && self.typed().is_none() {
                    Snap::round(angle, self.snap.angle)
                } else {
                    angle
                };
                params.set("angle", PropertyValue::Float(angle));
                params.set("axis", PropertyValue::Enum(axis.index()));
                // About each object's own axis when held to a local one.
                if self.local_axis().is_some() {
                    params.set("orientation", PropertyValue::Enum(1));
                }
            }
            Kind::Scale => {
                let factor = match self.typed() {
                    Some(value) => value,
                    // The ratio of the pointer's distance from the pivot now to
                    // what it was at the start. Dragging away grows, dragging
                    // through the pivot shrinks towards nothing.
                    None => self.distance_ratio() as f64,
                };
                let factor = if self.snapping() && self.typed().is_none() {
                    Snap::round(factor, self.snap.factor)
                } else {
                    factor
                };
                let scale = match self.axis {
                    None => [factor, factor, factor],
                    Some(Axis::X) => [factor, 1.0, 1.0],
                    Some(Axis::Y) => [1.0, factor, 1.0],
                    Some(Axis::Z) => [1.0, 1.0, factor],
                };
                params.set("scale", PropertyValue::Vec3(scale));
                // Along each object's own axes when held to a local one; the
                // world's otherwise, which is the operators' default.
                if self.local_axis().is_some() {
                    params.set("orientation", PropertyValue::Enum(1));
                }
            }
        }
        // Carried on every transform, so an operator that supports it sees it and
        // one that does not ignores it. The radius is in world units; absent
        // means off, which is what an operator that has never heard of it
        // defaults to anyway.
        if let Some(radius) = self.proportional {
            params.set("proportional_size", PropertyValue::Float(radius));
            params.set("proportional_falloff", PropertyValue::Enum(self.falloff));
        }
        params
    }

    /// The translation this gesture means, constrained and snapped.
    ///
    /// In `f64`, because that is what comes out the other end: the parameter is
    /// `f64` and an object's transform is `f64` for the large-scene precision
    /// §4.10 depends on. Computing in `f32` and widening at the last moment
    /// looked harmless and was not — a typed `1.4/8` came out as
    /// 0.17499999701976776, which is a rounding error the user can see in the
    /// header and in the saved file.
    fn offset(&self, dx: f32, dy: f32) -> DVec3 {
        // Held where it started while the base is picked: the points to pick
        // from are where they were.
        if self.picking.is_some() {
            return DVec3::ZERO;
        }
        // Only in increment mode. The element modes have already had their say
        // above, and rounding *their* answer would move the selection off the
        // vertex it was just snapped to.
        let increment = self.snapping() && self.snap.mode == crate::snap::Mode::Increment;
        let snap = |v: f64| {
            if increment {
                Snap::round(v, self.snap.length)
            } else {
                v
            }
        };
        // An element snap names a *destination*, so it replaces the offset rather
        // than rounding it: the selection lands on the thing the cursor is over.
        // Under an axis constraint only the component along that axis is taken,
        // so a constrained move still travels only along its axis and stops
        // level with what was snapped to.
        //
        // A typed value still wins. Someone who wrote a number is not asking to
        // be moved somewhere else.
        if self.typed().is_none() {
            if let Some(target) = self.snap_target() {
                let free = self.screen_to_world(dx, dy).as_dvec3();
                let to_target = target - self.snap_source(target, free);
                return match self.axis {
                    None => to_target,
                    Some(axis) => {
                        let direction = self.axis_direction(axis).as_dvec3();
                        direction * to_target.dot(direction)
                    }
                };
            }
        }
        match (self.axis, self.typed()) {
            // A typed distance along a chosen axis is exact, and unsnapped.
            (Some(axis), Some(value)) => self.axis_direction(axis).as_dvec3() * value,
            // Typed with no axis: along the axis the drag most resembles, so
            // that typing a number after starting to drag does what the drag was
            // already doing.
            (None, Some(value)) => {
                let axis = self.dominant_axis(dx, dy);
                self.axis_direction(axis).as_dvec3() * value
            }
            (Some(axis), None) => {
                // The pointer's motion projected onto the axis, so the selection
                // follows the cursor as closely as a constrained move can.
                let direction = self.axis_direction(axis).as_dvec3();
                let free = self.screen_to_world(dx, dy).as_dvec3();
                direction * snap(free.dot(direction))
            }
            // Free: straight across the view plane, one pixel to one pixel's
            // worth of world. Snapped per component, so a snapped free move lands
            // on round numbers in every direction rather than only along one.
            (None, None) => {
                let free = self.screen_to_world(dx, dy).as_dvec3();
                DVec3::new(snap(free.x), snap(free.y), snap(free.z))
            }
        }
    }

    /// How far an extrude has gone: a typed number as it is, otherwise the
    /// constrained move's offset measured along the way it travels.
    fn extrude_distance(&self, dx: f32, dy: f32) -> f64 {
        if let Some(value) = self.typed() {
            return value;
        }
        if !self.has_normal {
            // Out from the middle grows it, as a bevel's width does.
            return self.length_value(-self.pull());
        }
        self.offset(dx, dy).dot(self.extrude_direction().as_dvec3())
    }

    /// Which way an extrude travels, in the world: its faces' normal, or the
    /// axis it was sent along.
    fn extrude_direction(&self) -> Vec3 {
        match self.axis {
            Some(axis) if !self.along_normal => self.axis_direction(axis),
            _ => self.basis[2].normalize_or(Vec3::Z),
        }
    }

    /// How much nearer the selection's middle the pointer is than it started,
    /// in world units at the pivot's depth — signed, so an inset reads it as is
    /// and a bevel the other way round.
    fn pull(&self) -> f64 {
        let (dx, dy) = self.effective_delta(self.cursor);
        let then = (self.start.0 - self.pivot_px.0).hypot(self.start.1 - self.pivot_px.1);
        let now = (self.start.0 + dx - self.pivot_px.0).hypot(self.start.1 + dy - self.pivot_px.1);
        ((then - now) * self.units_per_pixel) as f64
    }

    /// A length the pointer made, rounded to the increment when snapping. A
    /// typed number is taken as it is.
    fn length_value(&self, pointer: f64) -> f64 {
        match self.typed() {
            Some(value) => value,
            None if self.snapping() => Snap::round(pointer, self.snap.length),
            None => pointer,
        }
    }

    /// An inset's thickness: none at the start, more as the pointer moves in.
    fn inset_thickness(&self) -> f64 {
        self.length_value(self.pull().max(0.0)).max(0.0)
    }

    /// A bevel's width: the pointer moving out from the middle. Never quite
    /// nothing — a bevel of no width is no bevel, and the operator declines it,
    /// which would stop the gesture where it starts.
    fn bevel_width(&self) -> f64 {
        self.length_value((-self.pull()).max(0.0)).max(MIN_BEVEL)
    }

    /// How many loops: a typed whole number, otherwise what the wheel set.
    fn loop_cuts(&self) -> u32 {
        if self.anchor.is_some() {
            return self.cuts;
        }
        match self.typed() {
            Some(n) if n.is_finite() => (n.round() as i64).clamp(1, MAX_CUTS as i64) as u32,
            _ => self.cuts,
        }
    }

    /// Where a loop cut sits along its edge: the middle to begin with, then
    /// wherever the pointer has slid it — the motion along the edge on screen,
    /// as a fraction of the edge's length there.
    fn slide_position(&self, dx: f32, dy: f32) -> f64 {
        // Off the very ends: a cut slid onto a corner is a sliver of a face.
        self.slide_position_from(0.5, dx, dy).clamp(0.02, 0.98)
    }

    /// How far an edge slide has gone: the motion along the slide's way on
    /// screen, as a share of how far one whole slide reaches there — minus
    /// for the other side.
    fn slide_factor(&self, dx: f32, dy: f32) -> f64 {
        let Some(((ax, ay), (bx, by))) = self.slide else {
            return 0.0;
        };
        let (ex, ey) = (bx - ax, by - ay);
        let length2 = ex * ex + ey * ey;
        if length2 < 1.0 {
            return 0.0;
        }
        (((dx * ex + dy * ey) / length2) as f64).clamp(-1.0, 1.0)
    }

    /// The same, from `from` along the edge rather than its middle.
    fn slide_position_from(&self, from: f64, dx: f32, dy: f32) -> f64 {
        let Some(((ax, ay), (bx, by))) = self.slide else {
            return from;
        };
        let (ex, ey) = (bx - ax, by - ay);
        let length2 = ex * ex + ey * ey;
        // An edge seen end on has no length to slide along.
        if length2 < 1.0 {
            return from;
        }
        (from + ((dx * ex + dy * ey) / length2) as f64).clamp(0.0, 1.0)
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
    /// Which axis the gesture is constrained to, if any.
    ///
    /// Exposed so the modal header can wear that axis's colour. The status
    /// string already *says* the axis, but a colour is read without reading,
    /// and during a drag the user's attention is on the model.
    pub fn axis_index(&self) -> Option<usize> {
        self.axis.map(|a| a.index() as usize)
    }

    pub fn status(&self) -> String {
        if self.picking.is_some() {
            let on = match self.base_hit() {
                Some(hit) => format!("  on {}", hit.element.label()),
                None => String::new(),
            };
            return format!(
                "{}  snap base: point at what moves and click (B or Enter){on}",
                self.kind.label()
            );
        }
        let len = kreate_reflect::format_length;
        let axis = match (self.axis, self.orientation) {
            _ if self.kind == Kind::Extrude && self.along_normals => " along normals".to_string(),
            _ if self.kind == Kind::Extrude && self.along_normal => " normal".to_string(),
            (Some(axis), Orientation::Global) => format!(" {}", axis.label()),
            // Doubled, the way it was typed, so the header says which of the two
            // constraints is active rather than leaving it to be inferred from
            // the result.
            (Some(axis), Orientation::Local) => format!(" {0}{0}", axis.label()),
            (None, _) => String::new(),
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
            Kind::Extrude => {
                let (dx, dy) = self.effective_delta(self.cursor);
                len(self.extrude_distance(dx, dy))
            }
            Kind::Inset => {
                let thickness = self
                    .depth_hold
                    .map(|(held, _)| held)
                    .unwrap_or_else(|| self.inset_thickness());
                format!(
                    "thickness {}  depth {}  (Ctrl: depth)",
                    len(thickness),
                    len(self.inset_depth())
                )
            }
            Kind::Bevel => format!("width {}", len(self.bevel_width())),
            Kind::EdgeSlide => {
                let (dx, dy) = self.effective_delta(self.cursor);
                let factor = self
                    .typed()
                    .unwrap_or_else(|| self.slide_factor(dx, dy))
                    .clamp(-1.0, 1.0);
                format!("factor {factor:.3}  (type a factor, -1 to 1)")
            }
            Kind::LoopCut => {
                let (dx, dy) = self.effective_delta(self.cursor);
                match (self.anchor, self.anchored_distance(dx, dy)) {
                    (Some(end), Some(d)) => format!(
                        "cuts {}  {} from {} end  (type a distance; B: other end)",
                        self.loop_cuts(),
                        len(d),
                        if end { "second" } else { "first" },
                    ),
                    _ => format!(
                        "cuts {}  position {:.3}  (wheel: cuts; B: measure from an end)",
                        self.loop_cuts(),
                        self.slide_position(dx, dy)
                    ),
                }
            }
            Kind::Move | Kind::ExtrudeFree => {
                let (dx, dy) = self.effective_delta(self.cursor);
                // Each axis and how far in all, in the scene's unit: what a
                // part is being moved by is a dimension.
                let v = self.offset(dx, dy);
                format!(
                    "Dx {}  Dy {}  Dz {}  ({})",
                    len(v.x),
                    len(v.y),
                    len(v.z),
                    len(v.length())
                )
            }
            Kind::Rotate => {
                let radians = self.typed().unwrap_or(self.swept);
                format!("{:.1} deg", radians.to_degrees())
            }
            Kind::Scale => {
                let f = self.typed().unwrap_or_else(|| self.distance_ratio() as f64);
                format!("{f:.3}")
            }
        };
        let snap = if self.snapping() {
            let caught = if self.snap_target().is_some() {
                "*"
            } else {
                ""
            };
            match self.snap_hit() {
                Some(hit) if self.snap.mode == crate::snap::Mode::Element => {
                    format!("  snap {}{caught}", hit.element.label())
                }
                _ => format!("  snap {}{caught}", self.snap.mode.label()),
            }
        } else {
            String::new()
        };
        let proportional = match self.proportional {
            Some(r) => format!("  falloff {r:.2}"),
            None => String::new(),
        };
        format!(
            "{}{axis}  {value}{typed}{precise}{snap}{proportional}",
            self.kind.label()
        )
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
