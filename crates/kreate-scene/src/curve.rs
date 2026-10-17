//! Curve objects: Bézier, polyline and B-spline paths with handles, and the
//! mesh they are shown as — a wire, a tube, a ribbon, or a filled outline.
//!
//! A curve object is a mesh object that carries a [`Curve`]: what it shows
//! is built from the curve, as a body is built from its features, and the
//! mesh it holds is only what the last build left. That keeps a curve in
//! every place a mesh goes — the file, the clipboard, a shared session, the
//! picker — without a kind of its own.
//!
//! Points and handles are in the object's own space. A Bézier point's two
//! handles are absolute positions, as Blender keeps them, so moving a point
//! moves its handles with it only when the mover says so.

use kreate_math::{DVec3, Vec3};
use kreate_mesh::MeshData;

/// How a spline passes its points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SplineKind {
    /// Cubic Bézier segments between the points, shaped by their handles.
    #[default]
    Bezier,
    /// Straight lines from point to point.
    Poly,
    /// A uniform cubic B-spline: smooth, near the points but not through
    /// them — Blender's NURBS at its default order.
    Nurbs,
}

impl SplineKind {
    pub const ALL: [SplineKind; 3] = [SplineKind::Bezier, SplineKind::Poly, SplineKind::Nurbs];

    pub fn tag(self) -> &'static str {
        match self {
            SplineKind::Bezier => "bezier",
            SplineKind::Poly => "poly",
            SplineKind::Nurbs => "nurbs",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.tag() == tag)
    }
}

/// One point of a spline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvePoint {
    pub position: DVec3,
    /// The Bézier handles, as positions: the one the curve comes in by and
    /// the one it leaves by. Unused by the other kinds.
    pub left: DVec3,
    pub right: DVec3,
    /// Scales the bevel here: 1 is the curve's own depth.
    pub radius: f64,
    /// Turns the profile about the curve here, in radians.
    pub tilt: f64,
}

impl CurvePoint {
    /// A point with its handles on it: a corner, until they are pulled out.
    pub fn at(position: DVec3) -> Self {
        Self {
            position,
            left: position,
            right: position,
            radius: 1.0,
            tilt: 0.0,
        }
    }

    /// A point with handles laid `tangent` either side of it.
    pub fn with_handles(position: DVec3, tangent: DVec3) -> Self {
        Self {
            position,
            left: position - tangent,
            right: position + tangent,
            radius: 1.0,
            tilt: 0.0,
        }
    }

    /// Moves the point, its handles with it.
    pub fn translate(&mut self, by: DVec3) {
        self.position += by;
        self.left += by;
        self.right += by;
    }
}

/// One run of points.
#[derive(Clone, Debug, PartialEq)]
pub struct Spline {
    pub kind: SplineKind,
    pub points: Vec<CurvePoint>,
    /// Back to the first point at the end.
    pub closed: bool,
}

impl Spline {
    pub fn new(kind: SplineKind, points: Vec<CurvePoint>, closed: bool) -> Self {
        Self {
            kind,
            points,
            closed,
        }
    }

    /// The segments: one fewer than the points, or as many when closed.
    pub fn segment_count(&self) -> usize {
        let n = self.points.len();
        if n < 2 {
            0
        } else if self.closed {
            n
        } else {
            n - 1
        }
    }

    /// The spline as points along it, `resolution` steps a segment, each
    /// with its radius and tilt. A closed spline does not repeat its first
    /// point at the end.
    pub fn sample(&self, resolution: u32) -> Vec<Sampled> {
        let n = self.points.len();
        let steps = resolution.max(1) as usize;
        let segments = self.segment_count();
        if n == 0 {
            return Vec::new();
        }
        if segments == 0 {
            return vec![Sampled::from(&self.points[0])];
        }
        let mut out = Vec::with_capacity(segments * steps + 1);
        match self.kind {
            SplineKind::Poly => {
                for p in &self.points {
                    out.push(Sampled::from(p));
                }
                if !self.closed {
                    // Nothing more: a polyline is its points.
                }
            }
            SplineKind::Bezier => {
                for s in 0..segments {
                    let a = &self.points[s];
                    let b = &self.points[(s + 1) % n];
                    let last = s + 1 == segments && !self.closed;
                    let count = if last { steps + 1 } else { steps };
                    for k in 0..count {
                        let t = k as f64 / steps as f64;
                        out.push(Sampled {
                            position: bezier(a.position, a.right, b.left, b.position, t),
                            radius: a.radius + (b.radius - a.radius) * t,
                            tilt: a.tilt + (b.tilt - a.tilt) * t,
                        });
                    }
                }
            }
            SplineKind::Nurbs => {
                // Uniform cubic B-spline over the control points. Open: the
                // ends are repeated so the curve reaches them, as Blender's
                // Endpoint does.
                let control: Vec<&CurvePoint> = if self.closed {
                    self.points.iter().collect()
                } else {
                    let first = &self.points[0];
                    let last = &self.points[n - 1];
                    std::iter::once(first)
                        .chain(std::iter::once(first))
                        .chain(self.points.iter())
                        .chain(std::iter::once(last))
                        .chain(std::iter::once(last))
                        .collect()
                };
                let m = control.len();
                let spans = if self.closed { m } else { m - 3 };
                for s in 0..spans {
                    let p = |i: usize| control[(s + i) % m];
                    let last = s + 1 == spans && !self.closed;
                    let count = if last { steps + 1 } else { steps };
                    for k in 0..count {
                        let t = k as f64 / steps as f64;
                        let (t2, t3) = (t * t, t * t * t);
                        let w = [
                            (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0,
                            (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                            (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
                            t3 / 6.0,
                        ];
                        let mut position = DVec3::ZERO;
                        let (mut radius, mut tilt) = (0.0, 0.0);
                        for (i, wi) in w.iter().enumerate() {
                            position += p(i).position * *wi;
                            radius += p(i).radius * wi;
                            tilt += p(i).tilt * wi;
                        }
                        out.push(Sampled {
                            position,
                            radius,
                            tilt,
                        });
                    }
                }
            }
        }
        out
    }
}

/// A point along a spline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampled {
    pub position: DVec3,
    pub radius: f64,
    pub tilt: f64,
}

impl From<&CurvePoint> for Sampled {
    fn from(p: &CurvePoint) -> Self {
        Self {
            position: p.position,
            radius: p.radius,
            tilt: p.tilt,
        }
    }
}

fn bezier(p0: DVec3, p1: DVec3, p2: DVec3, p3: DVec3, t: f64) -> DVec3 {
    let u = 1.0 - t;
    p0 * (u * u * u) + p1 * (3.0 * u * u * t) + p2 * (3.0 * u * t * t) + p3 * (t * t * t)
}

/// An object carried along a curve object as the clock runs: Blender's
/// Follow Path. From `start`, the whole curve takes `frames` frames;
/// `offset` shifts where along it the object is, as a fraction.
#[derive(Clone, Debug, PartialEq)]
pub struct FollowPath {
    /// The curve object, by identity.
    pub curve: kreate_reflect::Uid,
    pub start: f64,
    pub frames: f64,
    pub offset: f64,
    /// Turned to face along the curve: its Y along the tangent.
    pub follow: bool,
    /// Round again past the end, rather than stopping there.
    pub cyclic: bool,
}

impl Default for FollowPath {
    fn default() -> Self {
        Self {
            curve: kreate_reflect::Uid::NONE,
            start: 1.0,
            frames: 100.0,
            offset: 0.0,
            follow: true,
            cyclic: false,
        }
    }
}

impl Curve {
    /// Where the curve is `t` of the way along, by length, with the tangent
    /// there: what a follower is put at. Over the first spline.
    pub fn along(&self, t: f64) -> Option<(DVec3, DVec3)> {
        let (pts, closed) = self.sampled().into_iter().next()?;
        let mut positions: Vec<DVec3> = pts.iter().map(|p| p.position).collect();
        if closed && positions.len() > 1 {
            positions.push(positions[0]);
        }
        if positions.len() < 2 {
            return Some((positions.first().copied()?, DVec3::Y));
        }
        let lengths: Vec<f64> = positions.windows(2).map(|w| w[0].distance(w[1])).collect();
        let total: f64 = lengths.iter().sum();
        if total <= 0.0 {
            return Some((positions[0], DVec3::Y));
        }
        let mut want = t.clamp(0.0, 1.0) * total;
        for (i, len) in lengths.iter().enumerate() {
            if want <= *len || i + 1 == lengths.len() {
                let f = if *len > 0.0 {
                    (want / len).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let tangent = (positions[i + 1] - positions[i]).normalize_or(DVec3::Y);
                return Some((positions[i].lerp(positions[i + 1], f), tangent));
            }
            want -= len;
        }
        None
    }
}

/// The curve, and how it is shown.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub splines: Vec<Spline>,
    /// Steps a segment when it is sampled.
    pub resolution: u32,
    /// A round profile of this radius swept along it: a tube, a rope, a
    /// pipe. Nothing is a wire.
    pub bevel_depth: f64,
    /// Segments round the bevel's profile.
    pub bevel_resolution: u32,
    /// A ribbon this wide across the curve, in its normal; with a bevel,
    /// a rounded slab.
    pub extrude: f64,
    /// A tube's ends closed.
    pub fill_caps: bool,
    /// A closed spline's inside filled as a face — and, extruded, a slab.
    pub fill: bool,
}

impl Default for Curve {
    fn default() -> Self {
        Self {
            splines: Vec::new(),
            resolution: 12,
            bevel_depth: 0.0,
            bevel_resolution: 8,
            extrude: 0.0,
            fill_caps: true,
            fill: false,
        }
    }
}

impl Curve {
    /// A Bézier curve through `points`, handles laid along the run.
    pub fn bezier_through(points: &[DVec3], closed: bool) -> Self {
        let n = points.len();
        let pts = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let before = if i > 0 {
                    points[i - 1]
                } else if closed && n > 1 {
                    points[n - 1]
                } else {
                    *p
                };
                let after = if i + 1 < n {
                    points[i + 1]
                } else if closed && n > 1 {
                    points[0]
                } else {
                    *p
                };
                // A third of the way to the neighbours: Catmull-Rom's handles.
                let tangent = (after - before) / 6.0;
                CurvePoint::with_handles(*p, tangent)
            })
            .collect();
        Self {
            splines: vec![Spline::new(SplineKind::Bezier, pts, closed)],
            ..Default::default()
        }
    }

    /// A Bézier circle of `radius` in the XY plane: four points, handles
    /// at the usual 0.5523 of the radius.
    pub fn circle(radius: f64) -> Self {
        const K: f64 = 0.552_284_75;
        let pts = (0..4)
            .map(|i| {
                let a = i as f64 * std::f64::consts::FRAC_PI_2;
                let (s, c) = a.sin_cos();
                let position = DVec3::new(c, s, 0.0) * radius;
                let tangent = DVec3::new(-s, c, 0.0) * radius * K;
                CurvePoint::with_handles(position, tangent)
            })
            .collect();
        Self {
            splines: vec![Spline::new(SplineKind::Bezier, pts, true)],
            ..Default::default()
        }
    }

    /// A polyline through `points`.
    pub fn poly(points: &[DVec3], closed: bool) -> Self {
        let pts = points.iter().map(|p| CurvePoint::at(*p)).collect();
        Self {
            splines: vec![Spline::new(SplineKind::Poly, pts, closed)],
            ..Default::default()
        }
    }

    /// Whether it is shown as a wire alone: nothing a mesh can be.
    pub fn is_wire(&self) -> bool {
        self.bevel_depth <= 0.0 && self.extrude <= 0.0 && !self.fill
    }

    /// Every spline sampled, in the object's space.
    pub fn sampled(&self) -> Vec<(Vec<Sampled>, bool)> {
        self.splines
            .iter()
            .filter(|s| !s.points.is_empty())
            .map(|s| (s.sample(self.resolution), s.closed))
            .collect()
    }

    /// The lines the curve is drawn by when it is a wire, as segments in
    /// the object's space.
    pub fn wire(&self) -> Vec<[DVec3; 2]> {
        let mut out = Vec::new();
        for (pts, closed) in self.sampled() {
            for w in pts.windows(2) {
                out.push([w[0].position, w[1].position]);
            }
            if closed && pts.len() > 2 {
                out.push([pts[pts.len() - 1].position, pts[0].position]);
            }
        }
        out
    }

    /// The mesh the curve is shown as: a tube or ribbon along it, a filled
    /// outline, or nothing for a wire.
    pub fn to_mesh(&self) -> MeshData {
        let mut positions: Vec<Vec3> = Vec::new();
        let mut faces: Vec<Vec<u32>> = Vec::new();
        for (pts, closed) in self.sampled() {
            if pts.len() < 2 {
                continue;
            }
            let swept = self.bevel_depth > 0.0 || self.extrude > 0.0;
            if swept && !(self.fill && closed && self.bevel_depth <= 0.0) {
                self.sweep(&pts, closed, &mut positions, &mut faces);
            }
            if self.fill && closed && pts.len() >= 3 {
                self.fill_outline(&pts, &mut positions, &mut faces);
            }
        }
        if faces.is_empty() {
            return MeshData::empty();
        }
        MeshData::from_faces(positions, &faces)
    }

    /// The profile swept along `pts`: a ring of `bevel_resolution` points
    /// for a bevel, two for an extrude, and a rounded slab for both.
    fn sweep(
        &self,
        pts: &[Sampled],
        closed: bool,
        positions: &mut Vec<Vec3>,
        faces: &mut Vec<Vec<u32>>,
    ) {
        let profile = self.profile();
        let frames = frames(pts, closed);
        let base = positions.len() as u32;
        let ring = profile.len() as u32;
        for (i, (p, (t, n, b))) in pts.iter().zip(&frames).enumerate() {
            let _ = i;
            let (s, c) = p.tilt.sin_cos();
            // The frame turned by the tilt about the tangent.
            let n2 = *n * c + *b * s;
            let b2 = *b * c - *n * s;
            let _ = t;
            for (x, y) in &profile {
                let r = p.radius.max(0.0);
                let at = p.position + n2 * (x * r) + b2 * (y * r);
                positions.push(at.as_vec3());
            }
        }
        let count = pts.len() as u32;
        let spans = if closed { count } else { count - 1 };
        for i in 0..spans {
            let (a, b) = (i, (i + 1) % count);
            for k in 0..ring {
                let k2 = (k + 1) % ring;
                // An extrude's profile is open: no face across its back.
                if ring == 2 && k2 == 0 {
                    continue;
                }
                faces.push(vec![
                    base + a * ring + k,
                    base + a * ring + k2,
                    base + b * ring + k2,
                    base + b * ring + k,
                ]);
            }
        }
        if !closed && self.fill_caps && ring > 2 {
            faces.push((0..ring).rev().map(|k| base + k).collect());
            faces.push((0..ring).map(|k| base + (count - 1) * ring + k).collect());
        }
    }

    /// The profile across the curve, in the frame's (normal, binormal).
    fn profile(&self) -> Vec<(f64, f64)> {
        let d = self.bevel_depth.max(0.0);
        let e = self.extrude.max(0.0);
        if d > 0.0 {
            let n = self.bevel_resolution.clamp(3, 256) as usize;
            if e > 0.0 {
                // A slab: a half ring each side of the extrude.
                let half = (n / 2).max(2);
                let mut out = Vec::new();
                for side in [1.0, -1.0] {
                    for k in 0..=half {
                        let a = std::f64::consts::PI * k as f64 / half as f64
                            - std::f64::consts::FRAC_PI_2;
                        let (s, c) = a.sin_cos();
                        out.push((c * d * side, e * side + s * d * side));
                    }
                }
                return out;
            }
            (0..n)
                .map(|k| {
                    let a = std::f64::consts::TAU * k as f64 / n as f64;
                    (a.cos() * d, a.sin() * d)
                })
                .collect()
        } else {
            vec![(0.0, e), (0.0, -e)]
        }
    }

    /// A closed outline's inside, as faces: triangulated in the plane it
    /// lies nearest, and extruded into a slab when the curve is.
    fn fill_outline(&self, pts: &[Sampled], positions: &mut Vec<Vec3>, faces: &mut Vec<Vec<u32>>) {
        let ring: Vec<DVec3> = pts.iter().map(|p| p.position).collect();
        let centre = ring.iter().copied().sum::<DVec3>() / ring.len() as f64;
        let mut normal = DVec3::ZERO;
        for i in 0..ring.len() {
            let (p, q) = (ring[i], ring[(i + 1) % ring.len()]);
            normal += DVec3::new(
                (p.y - q.y) * (p.z + q.z),
                (p.z - q.z) * (p.x + q.x),
                (p.x - q.x) * (p.y + q.y),
            );
        }
        let normal = normal.normalize_or(DVec3::Z);
        let x = normal.any_orthonormal_vector();
        let y = normal.cross(x);
        let flat: Vec<kreate_math::DVec2> = ring
            .iter()
            .map(|p| {
                let d = *p - centre;
                kreate_math::DVec2::new(d.dot(x), d.dot(y))
            })
            .collect();
        let tris = kreate_mesh::earcut::triangulate(&[flat]);
        if tris.is_empty() {
            return;
        }
        let e = self.extrude.max(0.0);
        let base = positions.len() as u32;
        let n = ring.len() as u32;
        if e > 0.0 {
            for p in &ring {
                positions.push((*p + normal * e).as_vec3());
            }
            for p in &ring {
                positions.push((*p - normal * e).as_vec3());
            }
            for t in &tris {
                faces.push(vec![
                    base + t[0] as u32,
                    base + t[1] as u32,
                    base + t[2] as u32,
                ]);
                faces.push(vec![
                    base + n + t[2] as u32,
                    base + n + t[1] as u32,
                    base + n + t[0] as u32,
                ]);
            }
            for i in 0..n {
                let j = (i + 1) % n;
                faces.push(vec![base + n + i, base + n + j, base + j, base + i]);
            }
        } else {
            for p in &ring {
                positions.push(p.as_vec3());
            }
            for t in &tris {
                faces.push(vec![
                    base + t[0] as u32,
                    base + t[1] as u32,
                    base + t[2] as u32,
                ]);
            }
        }
    }

    /// What the shown mesh depends on, for a cache.
    pub fn key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{self:?}").hash(&mut h);
        h.finish()
    }
}

/// A frame at each point: tangent, normal, binormal — carried along the
/// curve so the profile does not twist on its own (parallel transport), and
/// for a closed curve the twist it gathers spread back out so the ends meet.
fn frames(pts: &[Sampled], closed: bool) -> Vec<(DVec3, DVec3, DVec3)> {
    let n = pts.len();
    let tangent = |i: usize| -> DVec3 {
        let prev = if i > 0 {
            pts[i - 1].position
        } else if closed {
            pts[n - 1].position
        } else {
            pts[i].position
        };
        let next = if i + 1 < n {
            pts[i + 1].position
        } else if closed {
            pts[0].position
        } else {
            pts[i].position
        };
        (next - prev).normalize_or(DVec3::Z)
    };
    let mut out = Vec::with_capacity(n);
    let t0 = tangent(0);
    let mut normal = t0.any_orthonormal_vector();
    // The first frame's normal as near the world's up as it can be: a
    // ribbon on the ground lies flat.
    let up = if t0.z.abs() > 0.99 {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let flat = (up - t0 * up.dot(t0)).normalize_or(normal);
    normal = flat;
    let mut prev_t = t0;
    for i in 0..n {
        let t = tangent(i);
        // Rotate the normal as the tangent turned.
        let axis = prev_t.cross(t);
        let s = axis.length();
        let c = prev_t.dot(t).clamp(-1.0, 1.0);
        if s > 1e-12 {
            let axis = axis / s;
            let angle = s.atan2(c);
            normal = rotate(normal, axis, angle);
        }
        normal = (normal - t * normal.dot(t)).normalize_or(t.any_orthonormal_vector());
        out.push((t, normal, t.cross(normal)));
        prev_t = t;
    }
    if closed && n > 2 {
        // The twist left between the last frame and the first, undone a
        // little more at each point.
        let (t_end, n_end, _) = out[n - 1];
        let t_first = out[0].0;
        let axis = t_end.cross(t_first);
        let s = axis.length();
        let mut carried = n_end;
        if s > 1e-12 {
            carried = rotate(n_end, axis / s, s.atan2(t_end.dot(t_first)));
        }
        carried = (carried - t_first * carried.dot(t_first)).normalize_or(out[0].1);
        let n0 = out[0].1;
        let twist = carried.cross(n0).dot(t_first).atan2(carried.dot(n0));
        for (i, (t, nn, bb)) in out.iter_mut().enumerate() {
            let a = twist * i as f64 / n as f64;
            *nn = rotate(*nn, *t, a);
            *bb = t.cross(*nn);
        }
    }
    out
}

fn rotate(v: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    let (s, c) = angle.sin_cos();
    v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(m: &MeshData) -> f64 {
        kreate_mesh::measure::mass_properties(m, |p| p).volume
    }

    fn closed(m: &MeshData) -> bool {
        kreate_mesh::measure::is_closed(m)
    }

    #[test]
    fn a_circle_is_round_and_a_bezier_reaches_its_points() {
        let c = Curve::circle(2.0);
        let (pts, _) = &c.sampled()[0];
        assert_eq!(pts.len(), 4 * 12);
        for p in pts {
            assert!((p.position.length() - 2.0).abs() < 0.01, "{p:?}");
        }
        let through = [DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0)];
        let c = Curve::bezier_through(&through, false);
        let (pts, _) = &c.sampled()[0];
        assert_eq!(pts.len(), 2 * 12 + 1);
        assert!((pts[0].position - through[0]).length() < 1e-12);
        assert!((pts[12].position - through[1]).length() < 1e-12);
        assert!((pts[24].position - through[2]).length() < 1e-12);
    }

    #[test]
    fn a_wire_has_no_mesh_and_lines_to_draw() {
        let c = Curve::poly(&[DVec3::ZERO, DVec3::X, DVec3::Y], true);
        assert!(c.is_wire());
        assert_eq!(c.to_mesh().num_faces(), 0);
        assert_eq!(c.wire().len(), 3);
    }

    #[test]
    fn a_bevelled_line_is_a_closed_tube_of_the_right_volume() {
        let mut c = Curve::poly(
            &[DVec3::new(1.0, 2.0, 3.0), DVec3::new(5.0, 2.0, 3.0)],
            false,
        );
        c.bevel_depth = 0.5;
        c.bevel_resolution = 64;
        let m = c.to_mesh();
        assert!(closed(&m), "capped");
        let want = std::f64::consts::PI * 0.25 * 4.0;
        assert!((volume(&m) - want).abs() < want * 0.01, "{}", volume(&m));
    }

    #[test]
    fn a_bevelled_circle_is_a_closed_ring() {
        let mut c = Curve::circle(3.0);
        c.bevel_depth = 0.25;
        c.bevel_resolution = 64;
        c.resolution = 24;
        let m = c.to_mesh();
        assert!(closed(&m), "a ring has no ends");
        // A torus: 2π² R r².
        let want = 2.0 * std::f64::consts::PI.powi(2) * 3.0 * 0.0625;
        assert!((volume(&m) - want).abs() < want * 0.03, "{}", volume(&m));
    }

    #[test]
    fn a_filled_outline_extruded_is_a_slab() {
        let square = [
            DVec3::ZERO,
            DVec3::X * 2.0,
            DVec3::new(2.0, 2.0, 0.0),
            DVec3::Y * 2.0,
        ];
        let mut c = Curve::poly(&square, true);
        c.fill = true;
        assert_eq!(c.to_mesh().num_faces(), 2, "two triangles");
        c.extrude = 0.5;
        let m = c.to_mesh();
        assert!(closed(&m));
        assert!((volume(&m) - 4.0).abs() < 1e-6, "{}", volume(&m));
    }

    #[test]
    fn a_nurbs_is_smooth_and_reaches_its_ends() {
        let pts = [DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y];
        let mut c = Curve::poly(&pts, false);
        c.splines[0].kind = SplineKind::Nurbs;
        let (s, _) = &c.sampled()[0];
        assert!((s[0].position - pts[0]).length() < 1e-9);
        assert!((s[s.len() - 1].position - pts[3]).length() < 1e-9);
        // Inside the hull of its points.
        assert!(s
            .iter()
            .all(|p| p.position.x >= -1e-9 && p.position.x <= 1.0 + 1e-9));
    }

    #[test]
    fn the_kinds_have_tags() {
        for k in SplineKind::ALL {
            assert_eq!(SplineKind::from_tag(k.tag()), Some(k));
        }
    }
}
