//! Sculpting: a brush stroke as a function of the mesh and where the brush
//! went.
//!
//! A stroke is a list of dabs — a centre, a radius and a pen pressure each, in
//! the mesh's own space — and a brush with its settings. Applied in order to
//! the stroke's starting mesh, the same dabs make the same shape, which is
//! what lets the stroke be one operator: replayable, one step to undo, and
//! re-run from the start while the pointer is still moving.
//!
//! Every brush reads the normals of the mesh as the stroke found it, so a dab
//! does not chase what the dab before it did; the shaping brushes (flatten,
//! clay) read the surface as it is now, because a plane is where the surface
//! *is*.
//!
//! # What a brush leaves alone
//!
//! - The **mask** ([`MASK`], a point layer 0 to 1): a vertex is moved
//!   `1 − mask` as much. The mask brush paints it.
//! - **Hidden** vertices (the edit-mode hide layer, which face sets hide
//!   through): never moved.
//! - With **front faces only**, vertices whose normal faces away from the view.
//!
//! # Cost
//!
//! A dab touches the vertices under it, found through a grid over the
//! stroke's starting positions rather than by measuring every vertex. And a
//! stroke re-run with more dabs than last time picks up where it stopped: a
//! [`StrokeCache`] holds the result so far, so a growing stroke costs its new
//! dabs, not all of them again on every frame.

use std::sync::Arc;

use kreate_math::{Quat, Vec3};

use crate::attribute::{names, AttrData, Domain};
use crate::MeshData;

/// The mask a brush respects: a point layer, 0 untouched to 1 protected.
pub const MASK: &str = ".sculpt_mask";
/// Which face set each face is in: a face layer, 0 for none.
pub const FACE_SET: &str = ".sculpt_face_set";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Brush {
    /// Raises the surface along its normal under the brush; inverted, carves.
    #[default]
    Draw,
    /// Evens the surface out towards its neighbours.
    Smooth,
    /// Drags what is under the brush where the pointer goes.
    Grab,
    /// Swells each vertex along its own normal; inverted, shrinks.
    Inflate,
    /// Builds up toward a plane just above the surface, as strips of clay.
    ClayStrips,
    /// Cuts a sharp groove and pulls its sides in; inverted, a sharp ridge.
    Crease,
    /// Presses the surface onto the plane through it.
    Flatten,
    /// Planes off what stands above the surface's plane.
    Scrape,
    /// Fills in what lies below the surface's plane.
    Fill,
    /// Draws the surface in toward the brush's middle; inverted, pushes out.
    Pinch,
    /// Raises the surface to one even height however long the stroke goes on.
    Layer,
    /// Pulls the surface out along the stroke, like drawing out clay.
    SnakeHook,
    /// Paints the mask; inverted, clears it.
    Mask,
    /// Paints the faces under the brush into a face set.
    FaceSet,
}

impl Brush {
    /// Every brush, in the order scripts and the panel number them.
    pub const ALL: [Brush; 14] = [
        Brush::Draw,
        Brush::Smooth,
        Brush::Grab,
        Brush::Inflate,
        Brush::ClayStrips,
        Brush::Crease,
        Brush::Flatten,
        Brush::Scrape,
        Brush::Fill,
        Brush::Pinch,
        Brush::Layer,
        Brush::SnakeHook,
        Brush::Mask,
        Brush::FaceSet,
    ];

    pub fn from_index(i: u32) -> Self {
        Self::ALL.get(i as usize).copied().unwrap_or_default()
    }

    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|b| *b == self).unwrap_or(0) as u32
    }
}

/// How a brush's effect fades from its middle to its rim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Falloff {
    /// `3t² − 2t³`: full in the middle, easing to nothing. Blender's default.
    #[default]
    Smooth,
    /// A dome: broad, then dropping at the rim.
    Sphere,
    /// Sharp in the middle, a long soft skirt.
    Root,
    /// A point in the middle, little elsewhere.
    Sharp,
    /// Straight from full to nothing.
    Linear,
    /// Full all the way to the rim.
    Constant,
}

impl Falloff {
    pub const ALL: [Falloff; 6] = [
        Falloff::Smooth,
        Falloff::Sphere,
        Falloff::Root,
        Falloff::Sharp,
        Falloff::Linear,
        Falloff::Constant,
    ];

    pub fn from_index(i: u32) -> Self {
        Self::ALL.get(i as usize).copied().unwrap_or_default()
    }

    fn weight(self, distance: f32, radius: f32) -> f32 {
        if distance >= radius || radius <= 0.0 {
            return 0.0;
        }
        let t = 1.0 - distance / radius;
        match self {
            Falloff::Smooth => t * t * (3.0 - 2.0 * t),
            Falloff::Sphere => (1.0 - (1.0 - t) * (1.0 - t)).max(0.0).sqrt(),
            Falloff::Root => t.sqrt(),
            Falloff::Sharp => t * t,
            Falloff::Linear => t,
            Falloff::Constant => 1.0,
        }
    }
}

/// A grey picture stamped through the brush: where it is dark the brush does
/// less. Sampled with wrapping, across the brush's width, square to the view.
#[derive(Clone, Debug, PartialEq)]
pub struct Alpha {
    pub width: u32,
    pub height: u32,
    /// Row by row, 0 to 1.
    pub values: Vec<f32>,
}

impl Alpha {
    /// The value at `(u, v)`, both 0 to 1 across the picture, bilinear.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let (w, h) = (self.width.max(1) as usize, self.height.max(1) as usize);
        if self.values.len() < w * h {
            return 1.0;
        }
        let x = u.rem_euclid(1.0) * w as f32 - 0.5;
        let y = v.rem_euclid(1.0) * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |x: f32, y: f32| {
            let xi = (x as i64).rem_euclid(w as i64) as usize;
            let yi = (y as i64).rem_euclid(h as i64) as usize;
            self.values[yi * w + xi]
        };
        let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1.0, y0) * fx;
        let bottom = at(x0, y0 + 1.0) * (1.0 - fx) + at(x0 + 1.0, y0 + 1.0) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// One touch of the brush: where, how wide, and how hard the pen pressed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dab {
    pub centre: Vec3,
    pub radius: f32,
    /// 0 to 1; a mouse is always 1.
    pub pressure: f32,
}

impl Dab {
    pub fn new(centre: Vec3, radius: f32) -> Self {
        Self {
            centre,
            radius,
            pressure: 1.0,
        }
    }
}

/// What a stroke does besides its dabs.
#[derive(Clone, Debug)]
pub struct Stroke {
    pub brush: Brush,
    /// How hard, 0 to 1.
    pub strength: f32,
    /// The other way: carve instead of raise, shrink instead of swell.
    pub invert: bool,
    /// For a grab: how far the pointer has dragged, in the mesh's space.
    pub grab: Vec3,
    /// Mirrored across X, Y and Z as well: Blender's symmetry.
    pub mirror: [bool; 3],
    /// Repeated this many times round the Z axis; 1 is once.
    pub radial: u32,
    pub falloff: Falloff,
    /// The way the view looks, in the mesh's space; zero when unknown.
    pub view: Vec3,
    /// Only what faces the view is touched.
    pub front_only: bool,
    /// How much each dab also smooths what it touched, 0 to 1.
    pub auto_smooth: f32,
    /// A picture stamped through the brush.
    pub alpha: Option<Arc<Alpha>>,
    /// The face set the face set brush paints.
    pub face_set: i32,
}

impl Default for Stroke {
    fn default() -> Self {
        Self {
            brush: Brush::Draw,
            strength: 0.5,
            invert: false,
            grab: Vec3::ZERO,
            mirror: [false; 3],
            radial: 1,
            falloff: Falloff::Smooth,
            view: Vec3::ZERO,
            front_only: false,
            auto_smooth: 0.0,
            alpha: None,
            face_set: 1,
        }
    }
}

impl PartialEq for Stroke {
    /// The alpha by identity: the same picture is the same `Arc`, and
    /// comparing a picture's pixels on every frame of a stroke would cost
    /// more than the dab.
    fn eq(&self, other: &Self) -> bool {
        self.brush == other.brush
            && self.strength == other.strength
            && self.invert == other.invert
            && self.grab == other.grab
            && self.mirror == other.mirror
            && self.radial == other.radial
            && self.falloff == other.falloff
            && self.view == other.view
            && self.front_only == other.front_only
            && self.auto_smooth == other.auto_smooth
            && self.face_set == other.face_set
            && match (&self.alpha, &other.alpha) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

impl Stroke {
    /// Each copy of a dab the symmetry makes — the dab itself first — as the
    /// transform taking the original to it: a sign per axis, then a turn
    /// about Z.
    fn copies(&self) -> Vec<(Vec3, Quat)> {
        let mut signs = vec![Vec3::ONE];
        for (axis, on) in self.mirror.iter().enumerate() {
            if *on {
                let flipped: Vec<Vec3> = signs
                    .iter()
                    .map(|s| {
                        let mut s = *s;
                        s[axis] = -s[axis];
                        s
                    })
                    .collect();
                signs.extend(flipped);
            }
        }
        let n = self.radial.max(1);
        let mut out = Vec::new();
        for k in 0..n {
            let turn = Quat::from_rotation_z(std::f32::consts::TAU * k as f32 / n as f32);
            for s in &signs {
                out.push((*s, turn));
            }
        }
        out
    }
}

/// A falloff's weight at `distance` in a brush of `radius`.
pub(crate) fn falloff_of(f: Falloff, distance: f32, radius: f32) -> f32 {
    f.weight(distance, radius)
}

/// The copies a stroke's symmetry makes of each dab.
pub(crate) fn copies_of(stroke: &Stroke) -> Vec<(Vec3, Quat)> {
    stroke.copies()
}

/// Area-weighted vertex normals.
fn normals_of(positions: &[Vec3], mesh: &MeshData) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; positions.len()];
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        let mut n = Vec3::ZERO;
        for i in 0..ring.len() {
            n += positions[ring[i] as usize].cross(positions[ring[(i + 1) % ring.len()] as usize]);
        }
        for v in ring {
            out[*v as usize] += n;
        }
    }
    out.into_iter().map(|n| n.normalize_or_zero()).collect()
}

/// Each vertex's neighbours along edges, packed: vertex `v`'s are
/// `items[start[v]..start[v + 1]]`.
struct Neighbours {
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Neighbours {
    fn new(mesh: &MeshData) -> Self {
        let n = mesh.num_verts() as usize;
        let mut start = vec![0u32; n + 1];
        for [a, b] in mesh.edges() {
            start[*a as usize + 1] += 1;
            start[*b as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; start[n] as usize];
        for [a, b] in mesh.edges() {
            items[fill[*a as usize] as usize] = *b;
            fill[*a as usize] += 1;
            items[fill[*b as usize] as usize] = *a;
            fill[*b as usize] += 1;
        }
        Self { start, items }
    }

    fn of(&self, v: usize) -> &[u32] {
        &self.items[self.start[v] as usize..self.start[v + 1] as usize]
    }
}

/// Which faces each vertex is on, packed as [`Neighbours`] is.
struct VertexFaces {
    start: Vec<u32>,
    items: Vec<u32>,
}

impl VertexFaces {
    fn new(mesh: &MeshData) -> Self {
        let n = mesh.num_verts() as usize;
        let mut start = vec![0u32; n + 1];
        for f in 0..mesh.num_faces() {
            for v in mesh.face_verts(f) {
                start[*v as usize + 1] += 1;
            }
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; start[n] as usize];
        for f in 0..mesh.num_faces() {
            for v in mesh.face_verts(f) {
                items[fill[*v as usize] as usize] = f;
                fill[*v as usize] += 1;
            }
        }
        Self { start, items }
    }

    fn of(&self, v: usize) -> &[u32] {
        &self.items[self.start[v] as usize..self.start[v + 1] as usize]
    }
}

/// A uniform grid over points, for finding the ones within a radius without
/// measuring all of them.
struct Grid {
    min: Vec3,
    cell: f32,
    dims: [u32; 3],
    /// Cell `c`'s points are `items[start[c]..start[c + 1]]`.
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Grid {
    /// Cells about `cell` wide — the brush's radius is the natural size —
    /// made coarser when that would mean more cells than about twice the
    /// points.
    fn new(points: &[Vec3], cell: f32) -> Self {
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in points.iter().filter(|p| p.is_finite()) {
            min = min.min(*p);
            max = max.max(*p);
        }
        if min.x > max.x {
            min = Vec3::ZERO;
            max = Vec3::ZERO;
        }
        let extent = max - min;
        let budget = (points.len() as f64 * 2.0).max(64.0);
        let mut cell = if cell.is_finite() && cell > 0.0 {
            cell
        } else {
            extent.max_element()
        };
        cell = cell.max(extent.max_element() / 1024.0).max(1e-6);
        let dims_for =
            |cell: f32| [extent.x, extent.y, extent.z].map(|e| (e / cell).floor() as u32 + 1);
        while dims_for(cell).iter().map(|d| *d as f64).product::<f64>() > budget {
            cell *= 1.5;
        }
        let mut grid = Self {
            min,
            cell,
            dims: dims_for(cell),
            start: Vec::new(),
            items: Vec::new(),
        };
        let cells = grid.dims.iter().map(|d| *d as usize).product::<usize>();
        let slot: Vec<Option<usize>> = points
            .iter()
            .map(|p| p.is_finite().then(|| grid.index(grid.cell_of(*p))))
            .collect();
        let mut start = vec![0u32; cells + 1];
        for c in slot.iter().flatten() {
            start[c + 1] += 1;
        }
        for i in 0..cells {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; start[cells] as usize];
        for (i, c) in slot.iter().enumerate() {
            if let Some(c) = c {
                items[fill[*c] as usize] = i as u32;
                fill[*c] += 1;
            }
        }
        grid.start = start;
        grid.items = items;
        grid
    }

    fn cell_of(&self, p: Vec3) -> [u32; 3] {
        let c = (p - self.min) / self.cell;
        let axis = |v: f32, d: u32| (v.floor().max(0.0) as u32).min(d - 1);
        [
            axis(c.x, self.dims[0]),
            axis(c.y, self.dims[1]),
            axis(c.z, self.dims[2]),
        ]
    }

    fn index(&self, [x, y, z]: [u32; 3]) -> usize {
        (z as usize * self.dims[1] as usize + y as usize) * self.dims[0] as usize + x as usize
    }

    /// Every point within `radius` of `centre`, with its distance.
    fn within(&self, points: &[Vec3], centre: Vec3, radius: f32, mut f: impl FnMut(usize, f32)) {
        if radius.is_nan() || radius <= 0.0 || !centre.is_finite() {
            return;
        }
        let lo = self.cell_of(centre - Vec3::splat(radius));
        let hi = self.cell_of(centre + Vec3::splat(radius));
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let c = self.index([x, y, z]);
                    for &i in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
                        let d = points[i as usize].distance(centre);
                        if d < radius {
                            f(i as usize, d);
                        }
                    }
                }
            }
        }
    }
}

/// Two axes square to `n`.
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let n = n.normalize_or(Vec3::Z);
    let other = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let u = other.cross(n).normalize();
    (u, n.cross(u))
}

/// A stroke part-way through: the mesh as it found it, what it has done so
/// far, and the tables every dab reads.
///
/// [`StrokeCache::fits`] says whether a call can pick up from here — the same
/// mesh, the same brush settings, and the dabs so far at the front of the new
/// list — and [`StrokeCache::run`] then applies only what is new. The answer
/// is the one [`apply`] gives from scratch.
pub struct StrokeCache {
    original: Vec<Vec3>,
    corners: Vec<u32>,
    normal: Vec<Vec3>,
    /// Only the brushes that need them build them.
    near: Option<Neighbours>,
    faces_of: Option<VertexFaces>,
    grid: Grid,
    current: Vec<Vec3>,
    mask: Vec<f32>,
    hidden: Vec<bool>,
    face_sets: Vec<i32>,
    /// The mask, hidden vertices and face sets the stroke began with, which
    /// the ones above move away from as a mask or face-set brush paints.
    started_with: (Vec<f32>, Vec<bool>, Vec<i32>),
    /// The stroke settings, its grab left out: a grab is re-done from the
    /// start every time anyway.
    key: Stroke,
    applied: Vec<Dab>,
    moved: bool,
    masked: bool,
    painted_sets: bool,
}

impl StrokeCache {
    /// Starts a stroke on `mesh`. `radius` sizes the search grid; the brush's
    /// is right.
    pub fn new(mesh: &MeshData, stroke: &Stroke, radius: f32) -> Self {
        let original = mesh.positions().to_vec();
        let started_with = Self::protection(mesh);
        let (mask, hidden, face_sets) = started_with.clone();
        let smoothing = stroke.brush == Brush::Smooth || stroke.auto_smooth > 0.0;
        Self {
            grid: Grid::new(&original, radius),
            normal: normals_of(&original, mesh),
            corners: mesh.corner_verts().to_vec(),
            near: smoothing.then(|| Neighbours::new(mesh)),
            faces_of: (stroke.brush == Brush::FaceSet).then(|| VertexFaces::new(mesh)),
            current: original.clone(),
            original,
            mask,
            hidden,
            face_sets,
            started_with,
            key: Self::key(stroke),
            applied: Vec::new(),
            moved: false,
            masked: false,
            painted_sets: false,
        }
    }

    /// What keeps vertices from a brush: the mask, the hidden ones, and the
    /// face sets — as the mesh has them now, with the defaults where it has
    /// none.
    fn protection(mesh: &MeshData) -> (Vec<f32>, Vec<bool>, Vec<i32>) {
        let n = mesh.positions().len();
        let attrs = mesh.attributes();
        let mask = attrs
            .get_f32(MASK)
            .filter(|m| m.len() == n)
            .map(|m| m.to_vec())
            .unwrap_or_else(|| vec![0.0; n]);
        let hidden = attrs
            .get_bool(names::HIDE_VERT)
            .filter(|h| h.len() == n)
            .map(|h| h.to_vec())
            .unwrap_or_else(|| vec![false; n]);
        let nf = mesh.num_faces() as usize;
        let face_sets = attrs
            .get_i32(FACE_SET)
            .filter(|s| s.len() == nf)
            .map(|s| s.to_vec())
            .unwrap_or_else(|| vec![0; nf]);
        (mask, hidden, face_sets)
    }

    fn key(stroke: &Stroke) -> Stroke {
        Stroke {
            grab: Vec3::ZERO,
            ..stroke.clone()
        }
    }

    /// Whether `dabs` on `mesh` with `stroke` continues what this has done.
    ///
    /// The mask, the hidden vertices and the face sets as well as the shape:
    /// masking everything, stroking, then inverting the mask and stroking
    /// the same dabs again left the positions and the dabs as they were, so
    /// the second stroke picked up the first's state — the old mask — and
    /// moved nothing.
    pub fn fits(&self, mesh: &MeshData, stroke: &Stroke, dabs: &[Dab]) -> bool {
        self.key == Self::key(stroke)
            && dabs.starts_with(&self.applied)
            && mesh.corner_verts() == self.corners.as_slice()
            && mesh.positions() == self.original.as_slice()
            && Self::protection(mesh) == self.started_with
    }

    /// Applies what of `dabs` is new and writes the result into `mesh`, which
    /// must be the mesh this began on (see [`StrokeCache::fits`]). `false`
    /// when nothing was under any dab.
    pub fn run(&mut self, mesh: &mut MeshData, dabs: &[Dab], stroke: &Stroke) -> bool {
        if dabs.is_empty() || self.original.is_empty() {
            return false;
        }
        if stroke.brush == Brush::Grab {
            self.grab(dabs[0], stroke);
        } else {
            let start = self.applied.len().min(dabs.len());
            let copies = stroke.copies();
            for (k, d) in dabs.iter().enumerate().skip(start) {
                let before = k.checked_sub(1).map(|p| dabs[p]);
                for (sign, turn) in &copies {
                    let place = |p: Vec3| *turn * (p * *sign);
                    let dab = Dab {
                        centre: place(d.centre),
                        ..*d
                    };
                    let from = before.map(|b| place(b.centre));
                    self.dab(dab, from, stroke, *sign, *turn);
                }
            }
            self.applied = dabs.to_vec();
        }
        if self.moved {
            mesh.positions_mut().copy_from_slice(&self.current);
        }
        if self.masked {
            mesh.attributes_mut().insert(
                MASK,
                Domain::Point,
                AttrData::F32(Arc::new(self.mask.clone())),
            );
        }
        if self.painted_sets {
            mesh.attributes_mut().insert(
                FACE_SET,
                Domain::Face,
                AttrData::I32(Arc::new(self.face_sets.clone())),
            );
        }
        self.moved || self.masked || self.painted_sets
    }

    /// How much vertex `i` takes of a dab that reaches it with falloff `w`:
    /// none if it is hidden, masked, facing away, or under a dark stretch of
    /// the alpha.
    fn reach(&self, i: usize, w: f32, dab: &Dab, stroke: &Stroke, sign: Vec3, turn: Quat) -> f32 {
        if self.hidden[i] {
            return 0.0;
        }
        let mut w = w * dab.pressure.clamp(0.0, 1.0);
        if stroke.brush != Brush::Mask {
            w *= 1.0 - self.mask[i].clamp(0.0, 1.0);
        }
        if stroke.front_only && stroke.view != Vec3::ZERO {
            let view = turn * (stroke.view * sign);
            if self.normal[i].dot(view) > 0.0 {
                return 0.0;
            }
        }
        if let Some(alpha) = &stroke.alpha {
            let view = if stroke.view == Vec3::ZERO {
                Vec3::Z
            } else {
                turn * (stroke.view * sign)
            };
            let (u, v) = basis(view);
            let d = self.original[i] - dab.centre;
            let size = 2.0 * dab.radius.max(1e-6);
            w *= alpha.sample(d.dot(u) / size + 0.5, d.dot(v) / size + 0.5);
        }
        w
    }

    /// One grab: what was under the first dab, carried by the drag.
    fn grab(&mut self, first: Dab, stroke: &Stroke) {
        self.current.copy_from_slice(&self.original);
        self.moved = false;
        for (sign, turn) in stroke.copies() {
            let dab = Dab {
                centre: turn * (first.centre * sign),
                ..first
            };
            let delta = turn * (stroke.grab * sign);
            let mut under = Vec::new();
            self.grid
                .within(&self.original, dab.centre, dab.radius, |i, d| {
                    under.push((i, d))
                });
            for (i, d) in under {
                let w = self.reach(
                    i,
                    stroke.falloff.weight(d, dab.radius),
                    &dab,
                    stroke,
                    sign,
                    turn,
                );
                if w > 0.0 {
                    self.current[i] += delta * w;
                    self.moved = true;
                }
            }
        }
    }

    fn dab(&mut self, dab: Dab, from: Option<Vec3>, stroke: &Stroke, sign: Vec3, turn: Quat) {
        let strength = stroke.strength.clamp(0.0, 1.0);
        let invert = if stroke.invert { -1.0 } else { 1.0 };
        // Who is under the brush, and how much. The snake hook takes hold of
        // the surface where it is now — it has been pulled along — and every
        // other brush where the stroke found it.
        let mut under: Vec<(usize, f32)> = Vec::new();
        if stroke.brush == Brush::SnakeHook {
            for (i, p) in self.current.iter().enumerate() {
                let d = p.distance(dab.centre);
                let w = stroke.falloff.weight(d, dab.radius);
                if w > 0.0 {
                    under.push((i, w));
                }
            }
        } else {
            let mut found = Vec::new();
            self.grid
                .within(&self.original, dab.centre, dab.radius, |i, d| {
                    found.push((i, d))
                });
            for (i, d) in found {
                let w = stroke.falloff.weight(d, dab.radius);
                if w > 0.0 {
                    under.push((i, w));
                }
            }
        }
        let under: Vec<(usize, f32)> = under
            .into_iter()
            .map(|(i, w)| (i, self.reach(i, w, &dab, stroke, sign, turn)))
            .filter(|(_, w)| *w > 0.0)
            .collect();
        if under.is_empty() {
            return;
        }
        match stroke.brush {
            Brush::Grab => {}
            Brush::Mask => {
                let step = strength * invert;
                for (i, w) in &under {
                    self.mask[*i] = (self.mask[*i] + step * *w).clamp(0.0, 1.0);
                }
                self.masked = true;
            }
            Brush::FaceSet => {
                if let Some(faces) = &self.faces_of {
                    for (i, _) in &under {
                        for f in faces.of(*i) {
                            self.face_sets[*f as usize] = stroke.face_set;
                        }
                    }
                    self.painted_sets = true;
                }
            }
            _ => {
                let near = self.near.as_ref();
                let empty: &[u32] = &[];
                self.moved |= displace(
                    stroke,
                    dab,
                    from,
                    &under,
                    &mut self.current,
                    &self.original,
                    &self.normal,
                    |i| near.map_or(empty, |n| n.of(i)),
                );
            }
        }
    }
}

/// What a shaping brush does to the vertices `under` it — each with how much
/// it takes — given where they are, where the stroke found them, their
/// normals, and each one's neighbours along edges. Shared by the fixed
/// topology of [`StrokeCache`] and by dynamic topology, which cuts the
/// surface finer before it calls this. `true` when anything moved.
#[allow(clippy::too_many_arguments)]
pub(crate) fn displace<'n>(
    stroke: &Stroke,
    dab: Dab,
    from: Option<Vec3>,
    under: &[(usize, f32)],
    current: &mut [Vec3],
    original: &[Vec3],
    normal: &[Vec3],
    near: impl Fn(usize) -> &'n [u32],
) -> bool {
    if under.is_empty() {
        return false;
    }
    let strength = stroke.strength.clamp(0.0, 1.0);
    let invert = if stroke.invert { -1.0 } else { 1.0 };
    // The surface under the brush: its normal and middle, weighted as the
    // brush is.
    let area_normal = under
        .iter()
        .fold(Vec3::ZERO, |s, (i, w)| s + normal[*i] * *w)
        .normalize_or_zero();
    let total: f32 = under.iter().map(|(_, w)| w).sum();
    let middle = under
        .iter()
        .fold(Vec3::ZERO, |s, (i, w)| s + current[*i] * *w)
        / total.max(1e-12);
    let r = dab.radius;
    let mut moved = false;
    match stroke.brush {
        Brush::Grab | Brush::Mask | Brush::FaceSet => {}
        Brush::Draw => {
            let step = r * 0.15 * strength * invert;
            for (i, w) in under {
                current[*i] += area_normal * step * *w;
            }
            moved = true;
        }
        Brush::Inflate => {
            let step = r * 0.1 * strength * invert;
            for (i, w) in under {
                current[*i] += normal[*i] * step * *w;
            }
            moved = true;
        }
        Brush::Smooth => moved |= smooth(under, strength, current, &near),
        Brush::ClayStrips | Brush::Flatten | Brush::Scrape | Brush::Fill => {
            // The plane: through the surface for flatten, scrape and fill;
            // a little above it for clay, which is what builds up.
            let lift = if stroke.brush == Brush::ClayStrips {
                r * 0.15 * strength * invert
            } else {
                0.0
            };
            let plane = middle + area_normal * lift;
            for (i, w) in under {
                let p = current[*i];
                let height = (p - plane).dot(area_normal);
                let takes = match (stroke.brush, stroke.invert) {
                    (Brush::Flatten, _) => true,
                    // Scrape cuts what is above; inverted it is fill.
                    (Brush::Scrape, false) | (Brush::Fill, true) => height > 0.0,
                    (Brush::Fill, false) | (Brush::Scrape, true) => height < 0.0,
                    // Clay only ever adds toward its plane.
                    (Brush::ClayStrips, false) => height < 0.0,
                    (Brush::ClayStrips, true) => height > 0.0,
                    _ => false,
                };
                if takes {
                    current[*i] -= area_normal * height * (*w * strength);
                }
            }
            moved = true;
        }
        Brush::Pinch | Brush::Crease => {
            let pinch = if stroke.brush == Brush::Crease {
                0.5
            } else {
                1.0
            } * strength
                * invert;
            for (i, w) in under {
                let p = current[*i];
                let to = dab.centre - p;
                let along = to - area_normal * to.dot(area_normal);
                current[*i] += along * (0.25 * pinch * *w);
            }
            if stroke.brush == Brush::Crease {
                // Into the surface; inverted, out of it.
                let step = -r * 0.1 * strength * invert;
                for (i, w) in under {
                    current[*i] += area_normal * step * *w * *w;
                }
            }
            moved = true;
        }
        Brush::Layer => {
            let height = r * 0.1 * strength.max(0.05) * 2.0 * invert;
            for (i, w) in under {
                let n = normal[*i];
                let have = (current[*i] - original[*i]).dot(n);
                let want = height * *w;
                let more = if invert > 0.0 {
                    (want - have).max(0.0)
                } else {
                    (want - have).min(0.0)
                };
                current[*i] += n * more;
            }
            moved = true;
        }
        Brush::SnakeHook => {
            if let Some(from) = from {
                let delta = dab.centre - from;
                for (i, w) in under {
                    current[*i] += delta * *w;
                }
                moved = true;
            }
        }
    }
    if stroke.auto_smooth > 0.0
        && !matches!(stroke.brush, Brush::Smooth | Brush::Mask | Brush::FaceSet)
    {
        moved |= smooth(under, stroke.auto_smooth.clamp(0.0, 1.0), current, &near);
    }
    moved
}

/// Moves each vertex under the brush toward the middle of its neighbours.
/// Everything is read before anything is written: each moves toward where
/// its neighbours were before this dab.
fn smooth<'n>(
    under: &[(usize, f32)],
    strength: f32,
    current: &mut [Vec3],
    near: &impl Fn(usize) -> &'n [u32],
) -> bool {
    let next: Vec<(usize, Vec3)> = under
        .iter()
        .filter_map(|(i, w)| {
            let w = w * strength;
            let n = near(*i);
            if w <= 0.0 || n.is_empty() {
                return None;
            }
            let average =
                n.iter().fold(Vec3::ZERO, |s, k| s + current[*k as usize]) / n.len() as f32;
            Some((*i, current[*i].lerp(average, w)))
        })
        .collect();
    let moved = !next.is_empty();
    for (i, p) in next {
        current[i] = p;
    }
    moved
}

/// Applies a stroke to `mesh`. `false` when nothing was under any dab.
pub fn apply(mesh: &mut MeshData, dabs: &[Dab], stroke: &Stroke) -> bool {
    if dabs.is_empty() || mesh.num_verts() == 0 {
        return false;
    }
    StrokeCache::new(mesh, stroke, dabs[0].radius).run(mesh, dabs, stroke)
}

/// The mask as it is, one value a vertex; zeros when there is none.
pub fn mask_of(mesh: &MeshData) -> Vec<f32> {
    let n = mesh.num_verts() as usize;
    mesh.attributes()
        .get_f32(MASK)
        .filter(|m| m.len() == n)
        .map(|m| m.to_vec())
        .unwrap_or_else(|| vec![0.0; n])
}

/// The face sets as they are, one a face; zeros when there are none.
pub fn face_sets_of(mesh: &MeshData) -> Vec<i32> {
    let n = mesh.num_faces() as usize;
    mesh.attributes()
        .get_i32(FACE_SET)
        .filter(|s| s.len() == n)
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![0; n])
}

/// What the mask operator does to the whole mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskAction {
    Clear,
    Invert,
    Fill,
}

/// Clears, inverts or fills the mask. `false` when nothing changed.
pub fn edit_mask(mesh: &mut MeshData, action: MaskAction) -> bool {
    let before = mask_of(mesh);
    let after: Vec<f32> = before
        .iter()
        .map(|m| match action {
            MaskAction::Clear => 0.0,
            MaskAction::Invert => 1.0 - m.clamp(0.0, 1.0),
            MaskAction::Fill => 1.0,
        })
        .collect();
    if after == before {
        return false;
    }
    if action == MaskAction::Clear {
        mesh.attributes_mut().remove(MASK);
    } else {
        mesh.attributes_mut()
            .insert(MASK, Domain::Point, AttrData::F32(Arc::new(after)));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn grid() -> MeshData {
        primitives::grid(4.0, 16, 16)
    }

    fn stroke(brush: Brush) -> Stroke {
        Stroke {
            brush,
            strength: 1.0,
            ..Stroke::default()
        }
    }

    fn dab(x: f32, y: f32, radius: f32) -> Dab {
        Dab::new(Vec3::new(x, y, 0.0), radius)
    }

    fn top(m: &MeshData) -> f32 {
        m.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max)
    }

    fn bottom(m: &MeshData) -> f32 {
        m.positions().iter().map(|p| p.z).fold(f32::MAX, f32::min)
    }

    fn at(m: &MeshData, x: f32, y: f32) -> Vec3 {
        *m.positions()
            .iter()
            .min_by(|a, b| {
                (a.truncate() - kreate_math::Vec2::new(x, y))
                    .length()
                    .total_cmp(&(b.truncate() - kreate_math::Vec2::new(x, y)).length())
            })
            .unwrap()
    }

    #[test]
    fn draw_raises_under_the_brush_and_nowhere_else_and_inverted_carves() {
        let mut m = grid();
        assert!(apply(&mut m, &[dab(0.0, 0.0, 1.0)], &stroke(Brush::Draw)));
        assert!(top(&m) > 0.1, "{}", top(&m));
        assert!(m
            .positions()
            .iter()
            .filter(|p| p.x.abs() > 1.1 || p.y.abs() > 1.1)
            .all(|p| p.z == 0.0));
        let mut m = grid();
        apply(
            &mut m,
            &[dab(0.0, 0.0, 1.0)],
            &Stroke {
                invert: true,
                ..stroke(Brush::Draw)
            },
        );
        assert!(bottom(&m) < -0.1);
    }

    #[test]
    fn smooth_takes_a_spike_down() {
        let mut m = grid();
        apply(&mut m, &[dab(0.0, 0.0, 0.3)], &stroke(Brush::Draw));
        let spike = top(&m);
        apply(&mut m, &[dab(0.0, 0.0, 1.0); 4], &stroke(Brush::Smooth));
        assert!(top(&m) < spike * 0.7, "{} from {spike}", top(&m));
    }

    #[test]
    fn grab_carries_what_was_under_the_first_dab() {
        let mut m = grid();
        let s = Stroke {
            grab: Vec3::new(0.0, 0.0, 2.0),
            ..stroke(Brush::Grab)
        };
        apply(&mut m, &[dab(0.0, 0.0, 1.0)], &s);
        assert!((top(&m) - 2.0).abs() < 0.05, "{}", top(&m));
    }

    #[test]
    fn symmetry_does_the_same_on_each_mirrored_side_and_round_the_axis() {
        let mut m = grid();
        let s = Stroke {
            mirror: [true, true, false],
            ..stroke(Brush::Draw)
        };
        apply(&mut m, &[dab(1.0, 0.75, 0.5)], &s);
        let z = at(&m, 1.0, 0.75).z;
        assert!(z > 0.0);
        for (x, y) in [(-1.0, 0.75), (1.0, -0.75), (-1.0, -0.75)] {
            assert!((at(&m, x, y).z - z).abs() < 1e-5, "({x}, {y})");
        }
        let mut m = grid();
        let s = Stroke {
            radial: 4,
            ..stroke(Brush::Draw)
        };
        apply(&mut m, &[dab(1.0, 0.0, 0.4)], &s);
        let z = at(&m, 1.0, 0.0).z;
        for (x, y) in [(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)] {
            assert!(
                (at(&m, x, y).z - z).abs() < 1e-4,
                "({x}, {y}): {} vs {z}",
                at(&m, x, y).z
            );
        }
    }

    #[test]
    fn the_mask_protects_and_the_mask_brush_paints_it() {
        let mut m = grid();
        assert!(apply(&mut m, &[dab(1.0, 0.0, 0.8)], &stroke(Brush::Mask)));
        let mask = mask_of(&m);
        assert!(mask.iter().any(|v| *v > 0.9), "painted");
        assert!(top(&m) == 0.0, "the mask brush moves nothing");
        apply(&mut m, &[dab(0.0, 0.0, 3.0)], &stroke(Brush::Draw));
        assert!(
            at(&m, 1.0, 0.0).z < at(&m, -1.0, 0.0).z * 0.2,
            "masked side barely moved"
        );
        // Inverted, it clears — wider, since the draw lifted the mask's edge
        // away from where it was painted.
        apply(
            &mut m,
            &[dab(1.0, 0.0, 1.5); 3],
            &Stroke {
                invert: true,
                ..stroke(Brush::Mask)
            },
        );
        assert!(mask_of(&m).iter().all(|v| *v < 0.01));
    }

    #[test]
    fn the_mask_is_cleared_inverted_and_filled() {
        let mut m = grid();
        assert!(edit_mask(&mut m, MaskAction::Fill));
        assert!(mask_of(&m).iter().all(|v| *v == 1.0));
        assert!(edit_mask(&mut m, MaskAction::Invert));
        assert!(mask_of(&m).iter().all(|v| *v == 0.0));
        edit_mask(&mut m, MaskAction::Fill);
        assert!(edit_mask(&mut m, MaskAction::Clear));
        assert!(!m.attributes().contains(MASK));
        assert!(!edit_mask(&mut m, MaskAction::Clear), "nothing to clear");
    }

    #[test]
    fn hidden_vertices_are_never_moved() {
        let mut m = grid();
        let hide: Vec<bool> = m.positions().iter().map(|p| p.x > 0.0).collect();
        m.attributes_mut().insert(
            names::HIDE_VERT,
            Domain::Point,
            AttrData::Bool(Arc::new(hide)),
        );
        apply(&mut m, &[dab(0.0, 0.0, 2.0)], &stroke(Brush::Draw));
        assert!(m
            .positions()
            .iter()
            .filter(|p| p.x > 0.0)
            .all(|p| p.z == 0.0));
        assert!(top(&m) > 0.0);
    }

    #[test]
    fn flatten_presses_a_bump_down_and_scrape_only_cuts_what_stands_up() {
        let mut bumpy = grid();
        apply(&mut bumpy, &[dab(0.0, 0.0, 0.5)], &stroke(Brush::Draw));
        apply(
            &mut bumpy,
            &[dab(1.0, 0.0, 0.5)],
            &Stroke {
                invert: true,
                ..stroke(Brush::Draw)
            },
        );
        let (hill, pit) = (top(&bumpy), bottom(&bumpy));
        let mut m = bumpy.clone();
        apply(&mut m, &[dab(0.5, 0.0, 1.5); 6], &stroke(Brush::Flatten));
        assert!(
            top(&m) < hill * 0.5 && bottom(&m) > pit * 0.5,
            "{} {}",
            top(&m),
            bottom(&m)
        );
        let mut m = bumpy.clone();
        apply(&mut m, &[dab(0.5, 0.0, 1.5); 6], &stroke(Brush::Scrape));
        assert!(top(&m) < hill * 0.7, "cut down: {}", top(&m));
        assert!(bottom(&m) <= pit * 0.9, "the pit left: {}", bottom(&m));
    }

    #[test]
    fn clay_builds_up_and_layer_stops_at_its_height() {
        let mut m = grid();
        apply(&mut m, &[dab(0.0, 0.0, 1.0); 3], &stroke(Brush::ClayStrips));
        assert!(top(&m) > 0.05, "{}", top(&m));
        let mut once = grid();
        apply(&mut once, &[dab(0.0, 0.0, 1.0)], &stroke(Brush::Layer));
        let mut many = grid();
        apply(&mut many, &[dab(0.0, 0.0, 1.0); 10], &stroke(Brush::Layer));
        assert!(top(&once) > 0.0);
        assert!(
            (top(&many) - top(&once)).abs() < 1e-5,
            "one height: {} vs {}",
            top(&many),
            top(&once)
        );
    }

    #[test]
    fn pinch_draws_in_and_crease_cuts_a_groove() {
        let mut m = grid();
        let before = at(&m, 0.75, 0.0);
        apply(&mut m, &[dab(0.0, 0.0, 1.2)], &stroke(Brush::Pinch));
        let moved = m
            .positions()
            .iter()
            .find(|p| (p.y - before.y).abs() < 1e-4 && p.x > 0.5 && p.x < 0.8)
            .copied();
        assert!(moved.unwrap().x < before.x, "toward the middle");
        let mut m = grid();
        apply(&mut m, &[dab(0.0, 0.0, 1.0)], &stroke(Brush::Crease));
        assert!(bottom(&m) < -0.01, "a groove: {}", bottom(&m));
    }

    #[test]
    fn snake_hook_pulls_the_surface_out_along_the_stroke() {
        let mut m = grid();
        let dabs: Vec<Dab> = (0..8)
            .map(|k| Dab::new(Vec3::new(0.0, 0.0, k as f32 * 0.1), 0.5))
            .collect();
        apply(&mut m, &dabs, &stroke(Brush::SnakeHook));
        assert!(top(&m) > 0.4, "pulled up with the stroke: {}", top(&m));
    }

    #[test]
    fn the_face_set_brush_paints_the_faces_under_it() {
        let mut m = grid();
        apply(
            &mut m,
            &[dab(0.0, 0.0, 0.6)],
            &Stroke {
                face_set: 7,
                ..stroke(Brush::FaceSet)
            },
        );
        let sets = face_sets_of(&m);
        assert!(sets.contains(&7) && sets.contains(&0));
        assert_eq!(top(&m), 0.0, "and moves nothing");
    }

    #[test]
    fn pressure_and_falloff_and_alpha_change_how_much() {
        let full = {
            let mut m = grid();
            apply(&mut m, &[dab(0.0, 0.0, 1.0)], &stroke(Brush::Draw));
            top(&m)
        };
        let light = {
            let mut m = grid();
            apply(
                &mut m,
                &[Dab {
                    pressure: 0.25,
                    ..dab(0.0, 0.0, 1.0)
                }],
                &stroke(Brush::Draw),
            );
            top(&m)
        };
        assert!((light - full * 0.25).abs() < 1e-4, "{light} vs {full}");
        // Constant reaches the rim at full strength; smooth has nothing there.
        let rim = |falloff| {
            let mut m = grid();
            apply(
                &mut m,
                &[dab(0.0, 0.0, 1.0)],
                &Stroke {
                    falloff,
                    ..stroke(Brush::Inflate)
                },
            );
            at(&m, 0.75, 0.0).z
        };
        assert!(rim(Falloff::Constant) > rim(Falloff::Smooth) * 2.0);
        // A black alpha stops it altogether.
        let black = Arc::new(Alpha {
            width: 2,
            height: 2,
            values: vec![0.0; 4],
        });
        let mut m = grid();
        assert!(!apply(
            &mut m,
            &[dab(0.0, 0.0, 1.0)],
            &Stroke {
                alpha: Some(black),
                view: -Vec3::Z,
                ..stroke(Brush::Draw)
            }
        ));
    }

    #[test]
    fn front_faces_only_leaves_the_back_of_a_sphere() {
        let mut m = primitives::uv_sphere(1.0, 24, 12);
        let s = Stroke {
            front_only: true,
            view: -Vec3::Z,
            ..stroke(Brush::Inflate)
        };
        apply(&mut m, &[Dab::new(Vec3::ZERO, 3.0)], &s);
        let base = primitives::uv_sphere(1.0, 24, 12);
        for (p, q) in m.positions().iter().zip(base.positions()) {
            if q.z < -0.2 {
                assert_eq!(p, q, "the far side, unmoved");
            }
        }
        assert!(m.positions().iter().any(|p| p.z > 1.01));
    }

    #[test]
    fn auto_smooth_softens_a_draw() {
        let sharp = {
            let mut m = grid();
            apply(&mut m, &[dab(0.0, 0.0, 0.4); 3], &stroke(Brush::Draw));
            top(&m)
        };
        let soft = {
            let mut m = grid();
            apply(
                &mut m,
                &[dab(0.0, 0.0, 0.4); 3],
                &Stroke {
                    auto_smooth: 0.8,
                    ..stroke(Brush::Draw)
                },
            );
            top(&m)
        };
        assert!(soft < sharp, "{soft} < {sharp}");
    }

    /// A stroke picked up from its cache, a dab or two at a time, lands
    /// exactly where the same stroke run from scratch does — for every brush,
    /// mirrored or not.
    #[test]
    fn a_stroke_continued_from_its_cache_matches_one_run_from_scratch() {
        let base = primitives::uv_sphere(1.0, 48, 24);
        let dabs: Vec<Dab> = (0..9)
            .map(|i| {
                Dab::new(
                    Vec3::new((i as f32 * 0.3).cos(), (i as f32 * 0.3).sin(), 0.2),
                    0.35,
                )
            })
            .collect();
        for brush in Brush::ALL {
            for mirror_x in [false, true] {
                let s = Stroke {
                    mirror: [mirror_x, false, false],
                    grab: Vec3::new(0.1, 0.0, 0.3),
                    ..stroke(brush)
                };
                let mut whole = base.clone();
                apply(&mut whole, &dabs, &s);

                let mut cache = StrokeCache::new(&base, &s, dabs[0].radius);
                let mut last = base.clone();
                for n in [1, 2, 5, 9] {
                    let mut m = base.clone();
                    assert!(cache.fits(&m, &s, &dabs[..n]), "{brush:?} {n}");
                    cache.run(&mut m, &dabs[..n], &s);
                    last = m;
                }
                assert_eq!(
                    last.positions(),
                    whole.positions(),
                    "{brush:?} mirror {mirror_x}"
                );
                assert_eq!(mask_of(&last), mask_of(&whole), "{brush:?}");
                assert_eq!(face_sets_of(&last), face_sets_of(&whole), "{brush:?}");
            }
        }
    }

    #[test]
    fn a_cache_does_not_fit_another_mesh_or_other_settings_or_fewer_dabs() {
        let base = grid();
        let s = stroke(Brush::Draw);
        let dabs = [dab(0.0, 0.0, 1.0), dab(1.0, 0.0, 1.0)];
        let mut cache = StrokeCache::new(&base, &s, 1.0);
        let mut m = base.clone();
        cache.run(&mut m, &dabs, &s);
        assert!(!cache.fits(&m, &s, &dabs), "the mesh it already changed");
        assert!(!cache.fits(&base, &s, &dabs[..1]), "fewer dabs");
        let harder = Stroke {
            strength: 0.5,
            ..s.clone()
        };
        assert!(!cache.fits(&base, &harder, &dabs));
        assert!(cache.fits(&base, &s, &dabs));
    }

    /// The grid finds exactly what measuring every vertex finds.
    #[test]
    fn the_grid_finds_what_a_full_scan_finds() {
        let m = primitives::uv_sphere(2.0, 40, 20);
        let p = m.positions();
        for cell in [0.05, 0.3, 5.0] {
            let g = Grid::new(p, cell);
            for (c, r) in [
                (Vec3::ZERO, 2.5),
                (Vec3::new(2.0, 0.0, 0.0), 0.4),
                (Vec3::new(9.0, 9.0, 9.0), 1.0),
                (Vec3::new(0.0, -2.0, 0.1), 0.05),
            ] {
                let mut found = Vec::new();
                g.within(p, c, r, |i, _| found.push(i));
                found.sort_unstable();
                let all: Vec<usize> = (0..p.len()).filter(|i| p[*i].distance(c) < r).collect();
                assert_eq!(found, all, "cell {cell} centre {c} radius {r}");
            }
        }
    }

    #[test]
    fn inflate_swells_a_sphere() {
        let mut m = primitives::uv_sphere(1.0, 16, 8);
        let r0 = m.positions().iter().map(|p| p.length()).fold(0.0, f32::max);
        apply(&mut m, &[Dab::new(Vec3::Z, 0.8)], &stroke(Brush::Inflate));
        let r1 = m.positions().iter().map(|p| p.length()).fold(0.0, f32::max);
        assert!(r1 > r0 + 0.01);
    }

    #[test]
    fn an_alpha_samples_bilinear_and_wraps() {
        let a = Alpha {
            width: 2,
            height: 1,
            values: vec![0.0, 1.0],
        };
        assert!((a.sample(0.25, 0.5) - 0.0).abs() < 1e-6);
        assert!((a.sample(0.5, 0.5) - 0.5).abs() < 1e-6);
        assert!(
            (a.sample(1.25, 0.5) - a.sample(0.25, 0.5)).abs() < 1e-6,
            "wraps"
        );
    }
}
