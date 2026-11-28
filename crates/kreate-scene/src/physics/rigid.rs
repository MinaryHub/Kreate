//! Rigid bodies: shapes that fall, hit, slide and come to rest.
//!
//! # The method
//!
//! Sequential impulses (Catto), over several small steps a frame. Each step:
//! gravity and damping change the velocities, the contacts are found, each
//! contact's impulse is solved a few times round — never pulling, and
//! friction never more than the friction coefficient times the push — and the
//! bodies move. A small velocity bias pushes overlapping bodies apart.
//!
//! # The contacts
//!
//! A convex body is held two ways at once: as the planes bounding it, and as
//! points on its corners and edges. A contact is a point of one body inside
//! the other's planes, so a box lying on a box touches at four corners, which
//! is what lets a stack stand. What that misses is two edges crossing with no
//! corner inside either body; with a box's edge midpoints among its points the
//! cases that matter for falling and stacking are held.
//!
//! A convex hull is approximated by the 26 planes of a k-DOP — the axes, the
//! diagonals of the faces and of the box — with its extreme vertices as its
//! points. That is a little fuller than the true hull at its corners and is
//! what a renderer's eye cannot tell from it once the object is moving.
//!
//! A passive body's own triangles (`CollisionShape::Mesh`) are a two-sided
//! surface: a point is in contact when it is behind the triangle nearest it,
//! as seen from the moving body's middle.

use kreate_math::{DQuat, DVec3, Vec3};

use super::collide::Collider;

/// A body's shape, in its own frame — its centre at the origin.
#[derive(Clone, Debug)]
pub enum Shape {
    Sphere {
        radius: f64,
    },
    Hull {
        /// Corners (and edge middles) tested against other bodies.
        points: Vec<DVec3>,
        /// Outward unit normal and distance from the centre, `n·x <= d` inside.
        planes: Vec<(DVec3, f64)>,
        /// Half the box round it.
        half: DVec3,
        /// Its edges, end to end, where they are known — a box's — for the
        /// contact two edges make crossing with no corner inside either.
        edges: Vec<[DVec3; 2]>,
    },
    /// Triangles in the world. Passive only.
    Mesh(Box<Collider>),
}

/// One body.
#[derive(Clone, Debug)]
pub struct Body {
    /// Centre of mass, in the world.
    pub pos: DVec3,
    pub rot: DQuat,
    pub vel: DVec3,
    /// Angular velocity, radians a second, in the world.
    pub spin: DVec3,
    pub inv_mass: f64,
    /// The inverse of the inertia about the body's own axes.
    pub inv_inertia: DVec3,
    pub shape: Shape,
    pub friction: f64,
    pub bounciness: f64,
    pub linear_damping: f64,
    pub angular_damping: f64,
    /// The fraction of the world's gravity that pulls it.
    pub gravity: f64,
    /// A ball round the shape, for a quick "can these two touch".
    reach: f64,
    /// Resting dynamic bodies sleep until disturbed by an awake contact or joint motor.
    pub sleeping: bool,
    sleep_timer: f64,
}

/// The 26 directions of a k-DOP: the axes, and the diagonals of the faces and
/// of the box.
fn dop_directions() -> Vec<DVec3> {
    let mut out = Vec::new();
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                if (x, y, z) != (0, 0, 0) {
                    out.push(DVec3::new(x as f64, y as f64, z as f64).normalize());
                }
            }
        }
    }
    out
}

impl Body {
    /// A box of half size `half`, centred at `pos`.
    pub fn cuboid(pos: DVec3, rot: DQuat, half: DVec3, mass: f64) -> Self {
        let h = half.max(DVec3::splat(1e-4));
        let mut points = Vec::new();
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    points.push(h * DVec3::new(x, y, z));
                }
            }
        }
        // The edges' middles: a box landing on an edge.
        let pairs = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        let edges: Vec<[DVec3; 2]> = pairs
            .iter()
            .map(|(a, b)| [points[*a], points[*b]])
            .collect();
        for (a, b) in pairs {
            points.push((points[a] + points[b]) * 0.5);
        }
        let planes = vec![
            (DVec3::X, h.x),
            (-DVec3::X, h.x),
            (DVec3::Y, h.y),
            (-DVec3::Y, h.y),
            (DVec3::Z, h.z),
            (-DVec3::Z, h.z),
        ];
        Self::with_shape(
            pos,
            rot,
            Shape::Hull {
                points,
                planes,
                half: h,
                edges,
            },
            mass,
        )
    }

    /// A ball.
    pub fn ball(pos: DVec3, radius: f64, mass: f64) -> Self {
        Self::with_shape(
            pos,
            DQuat::IDENTITY,
            Shape::Sphere {
                radius: radius.max(1e-4),
            },
            mass,
        )
    }

    /// The convex shape round `points`, given in the world; the body is
    /// turned as `rot` says, and centred on the middle of the points' box in
    /// that frame.
    pub fn hull(points: &[DVec3], rot: DQuat, mass: f64) -> Self {
        let inv = rot.inverse();
        let local: Vec<DVec3> = points.iter().map(|p| inv * *p).collect();
        let (lo, hi) = local.iter().fold(
            (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        if local.is_empty() {
            return Self::cuboid(DVec3::ZERO, rot, DVec3::splat(0.5), mass);
        }
        let centre = (lo + hi) * 0.5;
        let local: Vec<DVec3> = local.iter().map(|p| *p - centre).collect();
        let half = ((hi - lo) * 0.5).max(DVec3::splat(1e-4));
        let dirs = dop_directions();
        let planes: Vec<(DVec3, f64)> = dirs
            .iter()
            .map(|d| (*d, local.iter().map(|p| p.dot(*d)).fold(f64::MIN, f64::max)))
            .collect();
        // The extreme vertices in many directions: the hull's corners.
        let mut extremes: Vec<DVec3> = Vec::new();
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let mut sample = dirs.clone();
        for i in 0..64 {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / 64.0;
            let r = (1.0 - z * z).sqrt();
            sample.push(DVec3::new(
                r * (golden * i as f64).cos(),
                r * (golden * i as f64).sin(),
                z,
            ));
        }
        for d in sample {
            let best = local
                .iter()
                .copied()
                .max_by(|a, b| a.dot(d).total_cmp(&b.dot(d)))
                .unwrap_or(DVec3::ZERO);
            if !extremes.iter().any(|e| e.distance_squared(best) < 1e-12) {
                extremes.push(best);
            }
        }
        Self::with_shape(
            rot * centre,
            rot,
            Shape::Hull {
                points: extremes,
                planes,
                half,
                edges: Vec::new(),
            },
            mass,
        )
    }

    /// A passive body of triangles.
    pub fn mesh(collider: Collider) -> Self {
        let b = collider.bounds();
        let centre = ((b.min + b.max) * 0.5).as_dvec3();
        let mut body = Self::with_shape(
            centre,
            DQuat::IDENTITY,
            Shape::Mesh(Box::new(collider)),
            0.0,
        );
        body.reach = ((b.max - b.min) * 0.5).as_dvec3().length();
        body
    }

    fn with_shape(pos: DVec3, rot: DQuat, shape: Shape, mass: f64) -> Self {
        let (inertia, reach) = match &shape {
            Shape::Sphere { radius } => (DVec3::splat(0.4 * mass * radius * radius), *radius),
            Shape::Hull { half, points, .. } => {
                let h2 = *half * *half;
                (
                    DVec3::new(h2.y + h2.z, h2.x + h2.z, h2.x + h2.y) * (mass / 3.0),
                    points
                        .iter()
                        .map(|p| p.length())
                        .fold(half.length(), f64::max),
                )
            }
            Shape::Mesh(_) => (DVec3::ZERO, 0.0),
        };
        let dynamic = mass > 0.0 && !matches!(shape, Shape::Mesh(_));
        Self {
            pos,
            rot,
            vel: DVec3::ZERO,
            spin: DVec3::ZERO,
            inv_mass: if dynamic { 1.0 / mass } else { 0.0 },
            inv_inertia: if dynamic {
                DVec3::ONE / inertia.max(DVec3::splat(1e-12))
            } else {
                DVec3::ZERO
            },
            shape,
            friction: 0.5,
            bounciness: 0.0,
            linear_damping: 0.04,
            angular_damping: 0.1,
            gravity: 1.0,
            reach,
            sleeping: false,
            sleep_timer: 0.0,
        }
    }

    /// Makes it not move: a passive body.
    pub fn fixed(mut self) -> Self {
        self.inv_mass = 0.0;
        self.inv_inertia = DVec3::ZERO;
        self
    }

    pub fn is_dynamic(&self) -> bool {
        self.inv_mass > 0.0
    }

    /// Wakes a sleeping dynamic body so it participates in the next step.
    pub fn wake(&mut self) {
        self.sleeping = false;
        self.sleep_timer = 0.0;
    }

    /// Conservative world-space axis-aligned bounding box for broadphase culling.
    fn aabb(&self) -> (DVec3, DVec3) {
        match &self.shape {
            Shape::Mesh(m) => {
                let b = m.bounds();
                (b.min.as_dvec3(), b.max.as_dvec3())
            }
            Shape::Sphere { radius } => {
                let r = DVec3::splat(*radius);
                (self.pos - r, self.pos + r)
            }
            Shape::Hull { .. } => {
                let r = DVec3::splat(self.reach);
                (self.pos - r, self.pos + r)
            }
        }
    }

    fn inv_inertia_world(&self, v: DVec3) -> DVec3 {
        self.rot * ((self.rot.inverse() * v) * self.inv_inertia)
    }

    fn velocity_at(&self, r: DVec3) -> DVec3 {
        self.vel + self.spin.cross(r)
    }

    fn apply(&mut self, impulse: DVec3, r: DVec3) {
        self.vel += impulse * self.inv_mass;
        self.spin += self.inv_inertia_world(r.cross(impulse));
    }
}

/// Where two bodies touch: `normal` points from `b` to `a`.
#[derive(Clone, Copy, Debug)]
struct Contact {
    a: usize,
    b: usize,
    point: DVec3,
    normal: DVec3,
    depth: f64,
}

/// How far `p` is inside a hull body, and the face it is nearest leaving by.
fn inside_hull(body: &Body, planes: &[(DVec3, f64)], p: DVec3) -> Option<(DVec3, f64)> {
    let local = body.rot.inverse() * (p - body.pos);
    let mut best = (DVec3::Z, f64::MIN);
    for (n, d) in planes {
        let s = n.dot(local) - d;
        if s >= 0.0 {
            return None;
        }
        if s > best.1 {
            best = (*n, s);
        }
    }
    Some((body.rot * best.0, -best.1))
}

/// How far a sphere's centre `c` is outside a hull, and the face it is over.
fn hull_distance(body: &Body, planes: &[(DVec3, f64)], c: DVec3) -> (DVec3, f64) {
    let local = body.rot.inverse() * (c - body.pos);
    let mut best = (DVec3::Z, f64::MIN);
    for (n, d) in planes {
        let s = n.dot(local) - d;
        if s > best.1 {
            best = (*n, s);
        }
    }
    (body.rot * best.0, best.1)
}

/// A point of a moving body against a passive surface: the normal turned to
/// face the body's middle, and how deep the point is behind it.
fn against_mesh(collider: &Collider, centre: DVec3, p: DVec3, reach: f64) -> Option<(DVec3, f64)> {
    let (q, n) = collider.nearest(p.as_vec3(), reach as f32)?;
    let (q, mut n) = (q.as_dvec3(), n.as_dvec3());
    if (centre - q).dot(n) < 0.0 {
        n = -n;
    }
    let s = (p - q).dot(n);
    (s < 0.0).then_some((n, -s))
}

fn contacts_between(bodies: &[Body], a: usize, b: usize, out: &mut Vec<Contact>) {
    let (ba, bb) = (&bodies[a], &bodies[b]);
    let mut push = |point: DVec3, normal: DVec3, depth: f64| {
        out.push(Contact {
            a,
            b,
            point,
            normal,
            depth,
        })
    };
    match (&ba.shape, &bb.shape) {
        (Shape::Sphere { radius: ra }, Shape::Sphere { radius: rb }) => {
            let d = ba.pos - bb.pos;
            let dist = d.length();
            if dist < ra + rb {
                let n = if dist > 1e-9 { d / dist } else { DVec3::Z };
                push(ba.pos - n * *ra, n, ra + rb - dist);
            }
        }
        (Shape::Sphere { radius }, Shape::Hull { planes, .. }) => {
            let (n, s) = hull_distance(bb, planes, ba.pos);
            if s < *radius {
                push(ba.pos - n * *radius, n, radius - s);
            }
        }
        (Shape::Hull { planes, .. }, Shape::Sphere { radius }) => {
            let (n, s) = hull_distance(ba, planes, bb.pos);
            if s < *radius {
                push(bb.pos + n * *radius, -n, radius - s);
            }
        }
        (
            Shape::Hull {
                points: pa,
                planes: qa,
                edges: ea,
                ..
            },
            Shape::Hull {
                points: pb,
                planes: qb,
                edges: eb,
                ..
            },
        ) => {
            let world_edges = |b: &Body, e: &[[DVec3; 2]]| -> Vec<[DVec3; 2]> {
                e.iter()
                    .map(|[p, q]| [b.pos + b.rot * *p, b.pos + b.rot * *q])
                    .collect()
            };
            let (wea, web) = (world_edges(ba, ea), world_edges(bb, eb));
            // Edge against edge: their directions' crosses are axes too, and
            // the one that separates two crossed bars.
            let mut cross_axes: Vec<DVec3> = Vec::new();
            for [p, q] in &wea {
                for [r, s] in &web {
                    let c = (*q - *p).cross(*s - *r);
                    if c.length_squared() > 1e-18 {
                        let c = c.normalize();
                        if !cross_axes.iter().any(|x| x.dot(c).abs() > 1.0 - 1e-9) {
                            cross_axes.push(c);
                        }
                    }
                }
            }
            let wa: Vec<DVec3> = pa.iter().map(|p| ba.pos + ba.rot * *p).collect();
            let wb: Vec<DVec3> = pb.iter().map(|p| bb.pos + bb.rot * *p).collect();
            // One normal for the pair: the face direction, of either body,
            // along which they overlap least — the separating-axis answer.
            // Choosing per point instead goes wrong exactly where it matters:
            // a box squarely on a box has its corners on the other's side
            // faces, where "the nearest face" is a coin toss and a sideways
            // push holds nothing up.
            let along = |points: &[DVec3], n: DVec3| {
                points.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
                    (lo.min(p.dot(n)), hi.max(p.dot(n)))
                })
            };
            let mut best: Option<(DVec3, f64)> = None;
            let mut edge_axis = false;
            for (n, from_edges) in qa
                .iter()
                .map(|(n, _)| (ba.rot * *n, false))
                .chain(qb.iter().map(|(n, _)| (bb.rot * *n, false)))
                .chain(cross_axes.iter().map(|n| (*n, true)))
            {
                let (alo, ahi) = along(&wa, n);
                let (blo, bhi) = along(&wb, n);
                let overlap = (ahi.min(bhi) - alo.max(blo)).max(0.0);
                if ahi.min(bhi) < alo.max(blo) {
                    return;
                }
                // Turned to point from b to a.
                let n = if (ba.pos - bb.pos).dot(n) >= 0.0 {
                    n
                } else {
                    -n
                };
                if best.is_none_or(|(_, o)| overlap < o - 1e-9) {
                    best = Some((n, overlap));
                    edge_axis = from_edges;
                }
            }
            let Some((n, overlap)) = best else { return };
            let (_, b_top) = along(&wb, n);
            let (a_bottom, _) = along(&wa, n);
            let mut found = false;
            if !edge_axis {
                for w in &wa {
                    if inside_hull(bb, qb, *w).is_some() {
                        push(*w, n, (b_top - w.dot(n)).max(0.0));
                        found = true;
                    }
                }
                for w in &wb {
                    if inside_hull(ba, qa, *w).is_some() {
                        push(*w, n, (w.dot(n) - a_bottom).max(0.0));
                        found = true;
                    }
                }
            }
            // No corner in either, or the overlap is least across two
            // edges: the edges' nearest points, the contact half way between.
            if !found {
                let mut nearest: Option<(f64, DVec3)> = None;
                for [p, q] in &wea {
                    for [r, s] in &web {
                        let (x, y) = closest_on_segments(*p, *q, *r, *s);
                        let d = x.distance(y);
                        if d <= overlap + 1e-6 && nearest.is_none_or(|(b, _)| d < b) {
                            nearest = Some((d, (x + y) * 0.5));
                        }
                    }
                }
                if let Some((_, at)) = nearest {
                    push(at, n, overlap);
                }
            }
        }
        (Shape::Hull { points, .. }, Shape::Mesh(m)) => {
            for p in points {
                let w = ba.pos + ba.rot * *p;
                if let Some((n, depth)) = against_mesh(m, ba.pos, w, ba.reach * 0.5) {
                    push(w, n, depth);
                }
            }
        }
        (Shape::Mesh(m), Shape::Hull { points, .. }) => {
            for p in points {
                let w = bb.pos + bb.rot * *p;
                if let Some((n, depth)) = against_mesh(m, bb.pos, w, bb.reach * 0.5) {
                    push(w, -n, depth);
                }
            }
        }
        (Shape::Sphere { radius }, Shape::Mesh(m)) => {
            if let Some((q, n)) = m.nearest(ba.pos.as_vec3(), *radius as f32) {
                let (q, n) = (q.as_dvec3(), n.as_dvec3());
                let d = ba.pos - q;
                let dist = d.length();
                let n = if dist > 1e-9 {
                    d / dist
                } else if n.dot(d) >= 0.0 {
                    n
                } else {
                    -n
                };
                push(ba.pos - n * *radius, n, radius - dist);
            }
        }
        (Shape::Mesh(m), Shape::Sphere { radius }) => {
            if let Some((q, n)) = m.nearest(bb.pos.as_vec3(), *radius as f32) {
                let (q, n) = (q.as_dvec3(), n.as_dvec3());
                let d = bb.pos - q;
                let dist = d.length();
                let n = if dist > 1e-9 {
                    d / dist
                } else if n.dot(d) >= 0.0 {
                    n
                } else {
                    -n
                };
                push(bb.pos - n * *radius, -n, radius - dist);
            }
        }
        (Shape::Mesh(_), Shape::Mesh(_)) => {}
    }
}

/// The nearest points of segments `pq` and `rs`.
fn closest_on_segments(p: DVec3, q: DVec3, r: DVec3, s: DVec3) -> (DVec3, DVec3) {
    let (d1, d2, w) = (q - p, s - r, p - r);
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(w));
    let (b, c) = (d1.dot(d2), d1.dot(w));
    let denom = a * e - b * b;
    let mut sc = if denom > 1e-18 {
        ((b * f - c * e) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut tc = if e > 1e-18 { (b * sc + f) / e } else { 0.0 };
    if tc < 0.0 {
        tc = 0.0;
        sc = if a > 1e-18 {
            (-c / a).clamp(0.0, 1.0)
        } else {
            0.0
        };
    } else if tc > 1.0 {
        tc = 1.0;
        sc = if a > 1e-18 {
            ((b - c) / a).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    (p + d1 * sc, r + d2 * tc)
}

/// PD servo motor driving a hinge (radians) or slider (metres) joint toward a target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointMotor {
    /// Target angle (radians) for a hinge, or travel offset (metres) for a slider.
    pub target: f64,
    /// Proportional gain (N·m/rad or N/m).
    pub stiffness: f64,
    /// Derivative damping gain (N·m·s/rad or N·s/m).
    pub damping: f64,
    /// Maximum torque (N·m) or force (N) the actuator can exert.
    pub max_force: f64,
}

/// Degree-of-freedom kind of an articulated rigid-body joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RigidJointKind {
    /// Welds body `b` rigidly to body `a`.
    Fixed,
    /// Revolute hinge rotating about `axis` with optional `[lower, upper]` limits (radians).
    Hinge {
        lower: Option<f64>,
        upper: Option<f64>,
    },
    /// Prismatic slider translating along `axis` with optional `[lower, upper]` travel limits (metres).
    Slider {
        lower: Option<f64>,
        upper: Option<f64>,
    },
}

/// An articulated constraint connecting two bodies in a [`RigidWorld`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidJoint {
    pub body_a: usize,
    pub body_b: usize,
    /// Joint anchor point in body `a`'s local frame.
    pub anchor_a: DVec3,
    /// Joint anchor point in body `b`'s local frame.
    pub anchor_b: DVec3,
    /// Joint axis in body `a`'s local frame (unit vector).
    pub axis_a: DVec3,
    /// Joint axis in body `b`'s local frame (unit vector).
    pub axis_b: DVec3,
    /// Relative orientation of body `b` in body `a`'s frame at zero hinge angle.
    pub rest_rot: DQuat,
    pub kind: RigidJointKind,
    pub motor: Option<JointMotor>,
}

impl RigidJoint {
    /// A fixed weld joint between `body_a` and `body_b` at their respective local anchors.
    pub fn fixed(body_a: usize, body_b: usize, anchor_a: DVec3, anchor_b: DVec3) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a: DVec3::Z,
            axis_b: DVec3::Z,
            rest_rot: DQuat::IDENTITY,
            kind: RigidJointKind::Fixed,
            motor: None,
        }
    }

    /// A revolute hinge joint rotating about `axis` (expressed in both local frames).
    pub fn hinge(
        body_a: usize,
        body_b: usize,
        anchor_a: DVec3,
        anchor_b: DVec3,
        axis: DVec3,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        let u = axis.normalize_or_zero();
        let u = if u.length_squared() > 0.5 {
            u
        } else {
            DVec3::Z
        };
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a: u,
            axis_b: u,
            rest_rot: DQuat::IDENTITY,
            kind: RigidJointKind::Hinge { lower, upper },
            motor: None,
        }
    }

    /// A prismatic slider joint translating along `axis` (expressed in both local frames).
    pub fn slider(
        body_a: usize,
        body_b: usize,
        anchor_a: DVec3,
        anchor_b: DVec3,
        axis: DVec3,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        let u = axis.normalize_or_zero();
        let u = if u.length_squared() > 0.5 {
            u
        } else {
            DVec3::Z
        };
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a: u,
            axis_b: u,
            rest_rot: DQuat::IDENTITY,
            kind: RigidJointKind::Slider { lower, upper },
            motor: None,
        }
    }
}

/// The bodies, articulated joints, and what pulls them.
#[derive(Clone, Debug)]
pub struct RigidWorld {
    pub bodies: Vec<Body>,
    pub joints: Vec<RigidJoint>,
    pub gravity: DVec3,
    pub iterations: u32,
}

/// A contact made ready to solve.
struct Prepared {
    c: Contact,
    ra: DVec3,
    rb: DVec3,
    normal_mass: f64,
    target: f64,
    friction: f64,
    pushed: f64,
}

const SLEEP_LIN_SPEED: f64 = 0.03;
const SLEEP_ANG_SPEED: f64 = 0.05;
const SLEEP_TIME: f64 = 0.5;
const WAKE_LIN_SPEED: f64 = 0.06;
const WAKE_ANG_SPEED: f64 = 0.1;

fn ortho_basis(n: DVec3) -> (DVec3, DVec3) {
    let ref_axis = if n.x.abs() < 0.8 { DVec3::X } else { DVec3::Y };
    let t1 = n.cross(ref_axis).normalize_or_zero();
    let t2 = n.cross(t1).normalize_or_zero();
    (t1, t2)
}

fn effective_inv_mass(b: &Body) -> f64 {
    if b.is_dynamic() && !b.sleeping {
        b.inv_mass
    } else {
        0.0
    }
}

fn effective_inv_inertia_world(b: &Body, v: DVec3) -> DVec3 {
    if b.is_dynamic() && !b.sleeping {
        b.inv_inertia_world(v)
    } else {
        DVec3::ZERO
    }
}

fn apply_awake(b: &mut Body, impulse: DVec3, r: DVec3) {
    if b.is_dynamic() && !b.sleeping {
        b.apply(impulse, r);
    }
}

fn solve_linear_axis(
    bodies: &mut [Body],
    pair: (usize, usize),
    arms: (DVec3, DVec3),
    dir: DVec3,
    target_rel_vel: f64,
    limits: (f64, f64),
) {
    let (ia, ib) = pair;
    let (ra, rb) = arms;
    let (min_impulse, max_impulse) = limits;
    let (ba, bb) = (&bodies[ia], &bodies[ib]);
    let rel_v = (bb.velocity_at(rb) - ba.velocity_at(ra)).dot(dir);
    let k = effective_inv_mass(ba)
        + effective_inv_mass(bb)
        + effective_inv_inertia_world(ba, ra.cross(dir))
            .cross(ra)
            .dot(dir)
        + effective_inv_inertia_world(bb, rb.cross(dir))
            .cross(rb)
            .dot(dir);
    if k <= 1e-12 {
        return;
    }
    let lambda = ((target_rel_vel - rel_v) / k).clamp(min_impulse, max_impulse);
    apply_awake(&mut bodies[ia], -dir * lambda, ra);
    apply_awake(&mut bodies[ib], dir * lambda, rb);
}

fn solve_angular_axis(
    bodies: &mut [Body],
    ia: usize,
    ib: usize,
    axis: DVec3,
    target_rel_omega: f64,
    min_impulse: f64,
    max_impulse: f64,
) {
    let (ba, bb) = (&bodies[ia], &bodies[ib]);
    let rel_w = (bb.spin - ba.spin).dot(axis);
    let k = effective_inv_inertia_world(ba, axis).dot(axis)
        + effective_inv_inertia_world(bb, axis).dot(axis);
    if k <= 1e-12 {
        return;
    }
    let lambda = ((target_rel_omega - rel_w) / k).clamp(min_impulse, max_impulse);
    if bodies[ia].is_dynamic() && !bodies[ia].sleeping {
        let dw = bodies[ia].inv_inertia_world(axis * lambda);
        bodies[ia].spin -= dw;
    }
    if bodies[ib].is_dynamic() && !bodies[ib].sleeping {
        let dw = bodies[ib].inv_inertia_world(axis * lambda);
        bodies[ib].spin += dw;
    }
}

fn joint_hinge_angle(ba: &Body, bb: &Body, j: &RigidJoint) -> f64 {
    let q_rel = (ba.rot * j.rest_rot).inverse() * bb.rot;
    let proj = q_rel.xyz().dot(j.axis_a);
    let angle = 2.0 * proj.atan2(q_rel.w);
    // Normalize to [-PI, PI].
    if angle > std::f64::consts::PI {
        angle - std::f64::consts::TAU
    } else if angle < -std::f64::consts::PI {
        angle + std::f64::consts::TAU
    } else {
        angle
    }
}

fn joint_slider_offset(ba: &Body, bb: &Body, j: &RigidJoint) -> f64 {
    let pa = ba.pos + ba.rot * j.anchor_a;
    let pb = bb.pos + bb.rot * j.anchor_b;
    let axis = (ba.rot * j.axis_a).normalize_or_zero();
    (pb - pa).dot(axis)
}

impl RigidWorld {
    pub fn new(gravity: DVec3, iterations: u32) -> Self {
        Self {
            bodies: Vec::new(),
            joints: Vec::new(),
            gravity,
            iterations: iterations.max(1),
        }
    }

    /// Current hinge angle (radians) of joint `index`, or `0.0` if out of range.
    pub fn joint_angle(&self, index: usize) -> f64 {
        let Some(j) = self.joints.get(index) else {
            return 0.0;
        };
        let (Some(ba), Some(bb)) = (self.bodies.get(j.body_a), self.bodies.get(j.body_b)) else {
            return 0.0;
        };
        joint_hinge_angle(ba, bb, j)
    }

    /// Current slider travel offset (metres) of joint `index`, or `0.0` if out of range.
    pub fn joint_offset(&self, index: usize) -> f64 {
        let Some(j) = self.joints.get(index) else {
            return 0.0;
        };
        let (Some(ba), Some(bb)) = (self.bodies.get(j.body_a), self.bodies.get(j.body_b)) else {
            return 0.0;
        };
        joint_slider_offset(ba, bb, j)
    }

    /// One frame of `dt` seconds, in `substeps` steps.
    pub fn advance(&mut self, dt: f64, substeps: u32) {
        let n = substeps.max(1);
        for _ in 0..n {
            self.step(dt / n as f64);
        }
    }

    /// X-axis Sweep-and-Prune AABB broadphase (`O(N log N)`) followed by narrowphase.
    fn contacts(&self) -> Vec<Contact> {
        let n = self.bodies.len();
        if n < 2 {
            return Vec::new();
        }
        let mut entries: Vec<(usize, DVec3, DVec3)> = self
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let (lo, hi) = b.aabb();
                (i, lo, hi)
            })
            .collect();
        entries.sort_by(|a, b| a.1.x.total_cmp(&b.1.x));

        let mut out = Vec::new();
        for i in 0..n {
            let (ia, lo_a, hi_a) = entries[i];
            let ba = &self.bodies[ia];
            let a_awake_dyn = ba.is_dynamic() && !ba.sleeping;
            for &(ib, lo_b, hi_b) in &entries[i + 1..] {
                if lo_b.x > hi_a.x {
                    break;
                }
                if hi_a.y < lo_b.y || hi_b.y < lo_a.y || hi_a.z < lo_b.z || hi_b.z < lo_a.z {
                    continue;
                }
                let bb = &self.bodies[ib];
                let b_awake_dyn = bb.is_dynamic() && !bb.sleeping;
                // Skip static-static, static-sleeping, and sleeping-sleeping pairs.
                if !a_awake_dyn && !b_awake_dyn {
                    continue;
                }
                let (first, second) = if ia < ib { (ia, ib) } else { (ib, ia) };
                if self.joints.iter().any(|j| {
                    (j.body_a == first && j.body_b == second)
                        || (j.body_a == second && j.body_b == first)
                }) {
                    continue;
                }
                let (bf, bs) = (&self.bodies[first], &self.bodies[second]);
                let near = match (&bf.shape, &bs.shape) {
                    (Shape::Mesh(m), _) => overlaps_box(m, bs.pos, bs.reach),
                    (_, Shape::Mesh(m)) => overlaps_box(m, bf.pos, bf.reach),
                    _ => bf.pos.distance(bs.pos) <= bf.reach + bs.reach,
                };
                if near {
                    contacts_between(&self.bodies, first, second, &mut out);
                }
            }
        }
        out
    }

    /// One step of `dt` seconds.
    pub fn step(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let n_bodies = self.bodies.len();
        let mut motor_active = vec![false; n_bodies];
        // Wake bodies driven by active joint motors or linked to fast-moving bodies.
        for j in &self.joints {
            if j.body_a >= n_bodies || j.body_b >= n_bodies || j.body_a == j.body_b {
                continue;
            }
            if j.motor.is_some() {
                if self.bodies[j.body_a].is_dynamic() {
                    self.bodies[j.body_a].wake();
                    motor_active[j.body_a] = true;
                }
                if self.bodies[j.body_b].is_dynamic() {
                    self.bodies[j.body_b].wake();
                    motor_active[j.body_b] = true;
                }
            } else {
                let a_fast = self.bodies[j.body_a].is_dynamic()
                    && !self.bodies[j.body_a].sleeping
                    && (self.bodies[j.body_a].vel.length() > WAKE_LIN_SPEED
                        || self.bodies[j.body_a].spin.length() > WAKE_ANG_SPEED);
                let b_fast = self.bodies[j.body_b].is_dynamic()
                    && !self.bodies[j.body_b].sleeping
                    && (self.bodies[j.body_b].vel.length() > WAKE_LIN_SPEED
                        || self.bodies[j.body_b].spin.length() > WAKE_ANG_SPEED);
                if a_fast && self.bodies[j.body_b].is_dynamic() {
                    self.bodies[j.body_b].wake();
                }
                if b_fast && self.bodies[j.body_a].is_dynamic() {
                    self.bodies[j.body_a].wake();
                }
            }
        }

        for b in self
            .bodies
            .iter_mut()
            .filter(|b| b.is_dynamic() && !b.sleeping)
        {
            b.vel += self.gravity * b.gravity * dt;
            b.vel *= (1.0 - b.linear_damping.clamp(0.0, 1.0)).powf(dt);
            b.spin *= (1.0 - b.angular_damping.clamp(0.0, 1.0)).powf(dt);
        }

        // Apply PD servo motor forces/torques once per substep (with implicit damping stabilization).
        for j in &self.joints {
            let Some(motor) = j.motor else { continue };
            if j.body_a >= n_bodies || j.body_b >= n_bodies || j.body_a == j.body_b {
                continue;
            }
            match j.kind {
                RigidJointKind::Hinge { .. } => {
                    let (ba, bb) = (&self.bodies[j.body_a], &self.bodies[j.body_b]);
                    let axis = (ba.rot * j.axis_a).normalize_or_zero();
                    if axis.length_squared() < 0.5 {
                        continue;
                    }
                    let angle = joint_hinge_angle(ba, bb, j);
                    let rel_w = (bb.spin - ba.spin).dot(axis);
                    let k = effective_inv_inertia_world(ba, axis).dot(axis)
                        + effective_inv_inertia_world(bb, axis).dot(axis);
                    let denom = 1.0 + motor.damping * dt * k;
                    let tau = ((motor.stiffness * (motor.target - angle) - motor.damping * rel_w)
                        / denom)
                        .clamp(-motor.max_force, motor.max_force);
                    let imp = axis * (tau * dt);
                    if self.bodies[j.body_a].is_dynamic() && !self.bodies[j.body_a].sleeping {
                        let dw = self.bodies[j.body_a].inv_inertia_world(imp);
                        self.bodies[j.body_a].spin -= dw;
                    }
                    if self.bodies[j.body_b].is_dynamic() && !self.bodies[j.body_b].sleeping {
                        let dw = self.bodies[j.body_b].inv_inertia_world(imp);
                        self.bodies[j.body_b].spin += dw;
                    }
                }
                RigidJointKind::Slider { .. } => {
                    let (ba, bb) = (&self.bodies[j.body_a], &self.bodies[j.body_b]);
                    let axis = (ba.rot * j.axis_a).normalize_or_zero();
                    if axis.length_squared() < 0.5 {
                        continue;
                    }
                    let ra = ba.rot * j.anchor_a;
                    let rb = bb.rot * j.anchor_b;
                    let offset = joint_slider_offset(ba, bb, j);
                    let rel_v = (bb.velocity_at(rb) - ba.velocity_at(ra)).dot(axis);
                    let k = effective_inv_mass(ba) + effective_inv_mass(bb);
                    let denom = 1.0 + motor.damping * dt * k;
                    let force = ((motor.stiffness * (motor.target - offset)
                        - motor.damping * rel_v)
                        / denom)
                        .clamp(-motor.max_force, motor.max_force);
                    let imp = axis * (force * dt);
                    apply_awake(&mut self.bodies[j.body_a], -imp, ra);
                    apply_awake(&mut self.bodies[j.body_b], imp, rb);
                }
                RigidJointKind::Fixed => {}
            }
        }

        let raw_contacts = self.contacts();
        // Wake sleeping bodies struck by a moving awake dynamic body.
        for c in &raw_contacts {
            let a_awake_fast = self.bodies[c.a].is_dynamic()
                && !self.bodies[c.a].sleeping
                && (self.bodies[c.a].vel.length() > WAKE_LIN_SPEED
                    || self.bodies[c.a].spin.length() > WAKE_ANG_SPEED);
            let b_awake_fast = self.bodies[c.b].is_dynamic()
                && !self.bodies[c.b].sleeping
                && (self.bodies[c.b].vel.length() > WAKE_LIN_SPEED
                    || self.bodies[c.b].spin.length() > WAKE_ANG_SPEED);
            if a_awake_fast && self.bodies[c.b].is_dynamic() && self.bodies[c.b].sleeping {
                self.bodies[c.b].wake();
            }
            if b_awake_fast && self.bodies[c.a].is_dynamic() && self.bodies[c.a].sleeping {
                self.bodies[c.a].wake();
            }
        }

        let mut prepared: Vec<Prepared> = raw_contacts
            .into_iter()
            .map(|c| {
                let (a, b) = (&self.bodies[c.a], &self.bodies[c.b]);
                let ra = c.point - a.pos;
                let rb = c.point - b.pos;
                let n = c.normal;
                let k = effective_inv_mass(a)
                    + effective_inv_mass(b)
                    + effective_inv_inertia_world(a, ra.cross(n)).cross(ra).dot(n)
                    + effective_inv_inertia_world(b, rb.cross(n)).cross(rb).dot(n);
                let vn = (a.velocity_at(ra) - b.velocity_at(rb)).dot(n);
                let bounce = a.bounciness.max(b.bounciness).clamp(0.0, 1.0);
                // Coming in fast enough to bounce; slower is resting, and a
                // resting contact that bounced would never settle.
                let rebound = if vn < -0.5 { -bounce * vn } else { 0.0 };
                let push_out = (0.2 / dt * (c.depth - 0.002).max(0.0)).min(2.0);
                Prepared {
                    c,
                    ra,
                    rb,
                    normal_mass: if k > 1e-12 { 1.0 / k } else { 0.0 },
                    target: rebound.max(push_out),
                    friction: (a.friction * b.friction).max(0.0).sqrt(),
                    pushed: 0.0,
                }
            })
            .collect();

        let joints_snapshot = self.joints.clone();
        let baumgarte = 0.25 / dt;
        for _ in 0..self.iterations {
            for p in prepared.iter_mut() {
                let (ia, ib) = (p.c.a, p.c.b);
                let n = p.c.normal;
                let rel = self.bodies[ia].velocity_at(p.ra) - self.bodies[ib].velocity_at(p.rb);
                let vn = rel.dot(n);
                let total = (p.pushed + (p.target - vn) * p.normal_mass).max(0.0);
                let dl = total - p.pushed;
                p.pushed = total;
                apply_awake(&mut self.bodies[ia], n * dl, p.ra);
                apply_awake(&mut self.bodies[ib], -n * dl, p.rb);
                // Friction, against the sliding, no more than μ times the push.
                let rel = self.bodies[ia].velocity_at(p.ra) - self.bodies[ib].velocity_at(p.rb);
                let tangential = rel - n * rel.dot(n);
                let speed = tangential.length();
                if speed > 1e-9 {
                    let t = tangential / speed;
                    let (a, b) = (&self.bodies[ia], &self.bodies[ib]);
                    let k = effective_inv_mass(a)
                        + effective_inv_mass(b)
                        + effective_inv_inertia_world(a, p.ra.cross(t))
                            .cross(p.ra)
                            .dot(t)
                        + effective_inv_inertia_world(b, p.rb.cross(t))
                            .cross(p.rb)
                            .dot(t);
                    if k > 1e-12 {
                        let jt = (speed / k).min(p.friction * p.pushed);
                        apply_awake(&mut self.bodies[ia], -t * jt, p.ra);
                        apply_awake(&mut self.bodies[ib], t * jt, p.rb);
                    }
                }
            }

            // Solve articulated joint constraints (position, orientation alignment, and travel limits).
            for j in &joints_snapshot {
                let (ia, ib) = (j.body_a, j.body_b);
                if ia >= n_bodies || ib >= n_bodies || ia == ib {
                    continue;
                }
                if !self.bodies[ia].is_dynamic() && !self.bodies[ib].is_dynamic() {
                    continue;
                }
                let ra = self.bodies[ia].rot * j.anchor_a;
                let rb = self.bodies[ib].rot * j.anchor_b;
                let pa = self.bodies[ia].pos + ra;
                let pb = self.bodies[ib].pos + rb;
                let pos_err = pb - pa;

                match j.kind {
                    RigidJointKind::Fixed => {
                        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                            let target_v = -baumgarte * pos_err.dot(axis);
                            solve_linear_axis(
                                &mut self.bodies,
                                (ia, ib),
                                (ra, rb),
                                axis,
                                target_v,
                                (f64::NEG_INFINITY, f64::INFINITY),
                            );
                        }
                        let ref_rot = self.bodies[ia].rot * j.rest_rot;
                        let q_err = ref_rot.inverse() * self.bodies[ib].rot;
                        let sign = if q_err.w < 0.0 { -1.0 } else { 1.0 };
                        let ang_err = ref_rot * (q_err.xyz() * (2.0 * sign));
                        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                            let target_w = -baumgarte * ang_err.dot(axis);
                            solve_angular_axis(
                                &mut self.bodies,
                                ia,
                                ib,
                                axis,
                                target_w,
                                f64::NEG_INFINITY,
                                f64::INFINITY,
                            );
                        }
                    }
                    RigidJointKind::Hinge { lower, upper } => {
                        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                            let target_v = -baumgarte * pos_err.dot(axis);
                            solve_linear_axis(
                                &mut self.bodies,
                                (ia, ib),
                                (ra, rb),
                                axis,
                                target_v,
                                (f64::NEG_INFINITY, f64::INFINITY),
                            );
                        }
                        let axis_a_w = (self.bodies[ia].rot * j.axis_a).normalize_or_zero();
                        let axis_b_w = (self.bodies[ib].rot * j.axis_b).normalize_or_zero();
                        if axis_a_w.length_squared() > 0.5 && axis_b_w.length_squared() > 0.5 {
                            let perp_err = axis_a_w.cross(axis_b_w);
                            let (u1, u2) = ortho_basis(axis_a_w);
                            for perp in [u1, u2] {
                                let target_w = -baumgarte * perp_err.dot(perp);
                                solve_angular_axis(
                                    &mut self.bodies,
                                    ia,
                                    ib,
                                    perp,
                                    target_w,
                                    f64::NEG_INFINITY,
                                    f64::INFINITY,
                                );
                            }
                            let angle = joint_hinge_angle(&self.bodies[ia], &self.bodies[ib], j);
                            if let Some(lo) = lower {
                                if angle < lo {
                                    solve_angular_axis(
                                        &mut self.bodies,
                                        ia,
                                        ib,
                                        axis_a_w,
                                        baumgarte * (lo - angle),
                                        0.0,
                                        f64::INFINITY,
                                    );
                                }
                            }
                            if let Some(hi) = upper {
                                if angle > hi {
                                    solve_angular_axis(
                                        &mut self.bodies,
                                        ia,
                                        ib,
                                        axis_a_w,
                                        baumgarte * (hi - angle),
                                        f64::NEG_INFINITY,
                                        0.0,
                                    );
                                }
                            }
                        }
                    }
                    RigidJointKind::Slider { lower, upper } => {
                        let axis_w = (self.bodies[ia].rot * j.axis_a).normalize_or_zero();
                        if axis_w.length_squared() > 0.5 {
                            let (u1, u2) = ortho_basis(axis_w);
                            for perp in [u1, u2] {
                                let target_v = -baumgarte * pos_err.dot(perp);
                                solve_linear_axis(
                                    &mut self.bodies,
                                    (ia, ib),
                                    (ra, rb),
                                    perp,
                                    target_v,
                                    (f64::NEG_INFINITY, f64::INFINITY),
                                );
                            }
                            let offset = pos_err.dot(axis_w);
                            if let Some(lo) = lower {
                                if offset < lo {
                                    solve_linear_axis(
                                        &mut self.bodies,
                                        (ia, ib),
                                        (ra, rb),
                                        axis_w,
                                        baumgarte * (lo - offset),
                                        (0.0, f64::INFINITY),
                                    );
                                }
                            }
                            if let Some(hi) = upper {
                                if offset > hi {
                                    solve_linear_axis(
                                        &mut self.bodies,
                                        (ia, ib),
                                        (ra, rb),
                                        axis_w,
                                        baumgarte * (hi - offset),
                                        (f64::NEG_INFINITY, 0.0),
                                    );
                                }
                            }
                        }
                        // Lock relative rotation on a prismatic slider.
                        let ref_rot = self.bodies[ia].rot * j.rest_rot;
                        let q_err = ref_rot.inverse() * self.bodies[ib].rot;
                        let sign = if q_err.w < 0.0 { -1.0 } else { 1.0 };
                        let ang_err = ref_rot * (q_err.xyz() * (2.0 * sign));
                        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                            let target_w = -baumgarte * ang_err.dot(axis);
                            solve_angular_axis(
                                &mut self.bodies,
                                ia,
                                ib,
                                axis,
                                target_w,
                                f64::NEG_INFINITY,
                                f64::INFINITY,
                            );
                        }
                    }
                }
            }
        }

        for (idx, b) in self.bodies.iter_mut().enumerate() {
            if !b.is_dynamic() || b.sleeping {
                continue;
            }
            b.pos += b.vel * dt;
            let w = DQuat::from_xyzw(b.spin.x, b.spin.y, b.spin.z, 0.0);
            let dq = w * b.rot;
            b.rot = DQuat::from_xyzw(
                b.rot.x + 0.5 * dq.x * dt,
                b.rot.y + 0.5 * dq.y * dt,
                b.rot.z + 0.5 * dq.z * dt,
                b.rot.w + 0.5 * dq.w * dt,
            )
            .normalize();

            if !motor_active[idx]
                && b.vel.length_squared() < SLEEP_LIN_SPEED * SLEEP_LIN_SPEED
                && b.spin.length_squared() < SLEEP_ANG_SPEED * SLEEP_ANG_SPEED
            {
                b.sleep_timer += dt;
                if b.sleep_timer >= SLEEP_TIME {
                    b.sleeping = true;
                    b.vel = DVec3::ZERO;
                    b.spin = DVec3::ZERO;
                }
            } else {
                b.sleep_timer = 0.0;
            }
        }
    }
}

fn overlaps_box(m: &Collider, centre: DVec3, reach: f64) -> bool {
    let b = m.bounds();
    let r = Vec3::splat(reach as f32);
    let c = centre.as_vec3();
    (c + r).cmpge(b.min).all() && (c - r).cmple(b.max).all()
}

#[cfg(test)]
mod tests {
    /// Two bars crossed, each turned on its length to stand on an edge: they
    /// meet edge to edge, no corner of either inside the other. The upper
    /// one comes to rest on the lower's edge rather than falling through.
    #[test]
    fn crossed_bars_meet_edge_to_edge() {
        let mut w = world();
        let turn = |axis: DVec3| DQuat::from_axis_angle(axis, std::f64::consts::FRAC_PI_4);
        let half = 0.1 * std::f64::consts::SQRT_2;
        w.bodies.push(
            Body::cuboid(DVec3::ZERO, turn(DVec3::X), DVec3::new(1.0, 0.1, 0.1), 0.0).fixed(),
        );
        // The upper bar along Y, its lower edge a little above the lower's top.
        let mut upper = Body::cuboid(
            DVec3::new(0.0, 0.0, 2.0 * half + 0.02),
            turn(DVec3::Y),
            DVec3::new(0.1, 1.0, 0.1),
            1.0,
        );
        upper.angular_damping = 5.0;
        w.bodies.push(upper);
        run(&mut w, 0.6);
        let z = w.bodies[1].pos.z;
        assert!(z > 2.0 * half - 0.03, "fell through: {z}");
    }

    use super::*;
    use kreate_math::DAffine3;

    fn world() -> RigidWorld {
        RigidWorld::new(DVec3::new(0.0, 0.0, -9.81), 10)
    }

    fn run(w: &mut RigidWorld, seconds: f64) {
        let frames = (seconds * 24.0) as usize;
        for _ in 0..frames {
            w.advance(1.0 / 24.0, 10);
        }
    }

    #[test]
    fn a_box_falls_onto_a_box_and_stays_on_it() {
        let mut w = world();
        w.bodies.push(
            Body::cuboid(DVec3::ZERO, DQuat::IDENTITY, DVec3::new(2.0, 2.0, 0.5), 0.0).fixed(),
        );
        w.bodies.push(Body::cuboid(
            DVec3::new(0.0, 0.0, 3.0),
            DQuat::IDENTITY,
            DVec3::splat(0.5),
            1.0,
        ));
        run(&mut w, 3.0);
        let b = &w.bodies[1];
        assert!(
            (b.pos.z - 1.0).abs() < 0.03,
            "rests on the top at 1.0: {}",
            b.pos.z
        );
        assert!(b.vel.length() < 0.05, "at rest: {:?}", b.vel);
        assert!(
            b.pos.truncate().length() < 0.05,
            "did not wander: {:?}",
            b.pos
        );
    }

    #[test]
    fn a_tilted_box_lands_flat() {
        let mut w = world();
        w.bodies.push(
            Body::cuboid(DVec3::ZERO, DQuat::IDENTITY, DVec3::new(3.0, 3.0, 0.5), 0.0).fixed(),
        );
        let tilt = DQuat::from_rotation_x(0.5) * DQuat::from_rotation_y(0.3);
        w.bodies.push(Body::cuboid(
            DVec3::new(0.0, 0.0, 2.0),
            tilt,
            DVec3::splat(0.5),
            1.0,
        ));
        run(&mut w, 4.0);
        let b = &w.bodies[1];
        assert!((b.pos.z - 1.0).abs() < 0.05, "on a face: {}", b.pos.z);
        // A face down: the body's up is one of its axes.
        let up = b.rot.inverse() * DVec3::Z;
        let biggest = up.abs().max_element();
        assert!(biggest > 0.99, "flat: {up:?}");
    }

    #[test]
    fn three_boxes_stand_in_a_stack() {
        let mut w = world();
        w.bodies.push(
            Body::cuboid(DVec3::ZERO, DQuat::IDENTITY, DVec3::new(3.0, 3.0, 0.5), 0.0).fixed(),
        );
        for i in 0..3 {
            w.bodies.push(Body::cuboid(
                DVec3::new(0.0, 0.0, 1.0 + i as f64 * 1.01),
                DQuat::IDENTITY,
                DVec3::splat(0.5),
                1.0,
            ));
        }
        run(&mut w, 3.0);
        for i in 0..3 {
            let b = &w.bodies[i + 1];
            assert!(
                (b.pos.z - (1.0 + i as f64)).abs() < 0.08,
                "box {i} at {}",
                b.pos.z
            );
            assert!(b.pos.truncate().length() < 0.1, "box {i} slid: {:?}", b.pos);
        }
    }

    #[test]
    fn a_ball_rests_on_a_mesh_floor_and_a_bouncy_one_bounces() {
        let floor = Collider::new(
            &kreate_mesh::primitives::plane(10.0),
            DAffine3::IDENTITY,
            0.0,
            0.5,
            0.0,
        );
        let mut w = world();
        w.bodies.push(Body::mesh(floor.clone()));
        w.bodies
            .push(Body::ball(DVec3::new(0.0, 0.0, 2.0), 0.5, 1.0));
        run(&mut w, 3.0);
        assert!(
            (w.bodies[1].pos.z - 0.5).abs() < 0.03,
            "{}",
            w.bodies[1].pos.z
        );

        let mut w = world();
        w.bodies.push(Body::mesh(floor));
        let mut ball = Body::ball(DVec3::new(0.0, 0.0, 2.0), 0.5, 1.0);
        ball.bounciness = 0.8;
        w.bodies.push(ball);
        // Down for about 0.6 s; then on the way up again.
        let mut lowest = f64::MAX;
        let mut rose = false;
        for _ in 0..36 {
            w.advance(1.0 / 24.0, 10);
            let z = w.bodies[1].pos.z;
            if z < lowest {
                lowest = z;
            } else if z > lowest + 0.3 {
                rose = true;
            }
        }
        assert!(rose, "it bounced");
    }

    #[test]
    fn a_hull_of_a_box_mesh_behaves_as_the_box() {
        let mesh = kreate_mesh::primitives::cube(1.0);
        let points: Vec<DVec3> = mesh
            .positions()
            .iter()
            .map(|p| p.as_dvec3() + DVec3::new(0.0, 0.0, 3.0))
            .collect();
        let mut w = world();
        w.bodies.push(
            Body::cuboid(DVec3::ZERO, DQuat::IDENTITY, DVec3::new(2.0, 2.0, 0.5), 0.0).fixed(),
        );
        w.bodies.push(Body::hull(&points, DQuat::IDENTITY, 1.0));
        run(&mut w, 3.0);
        assert!(
            (w.bodies[1].pos.z - 1.0).abs() < 0.03,
            "{}",
            w.bodies[1].pos.z
        );
    }

    #[test]
    fn friction_holds_a_box_on_a_slope_that_a_slippery_one_slides_down() {
        let slide = |friction: f64| {
            let mut w = world();
            let slope = DQuat::from_rotation_y(0.3);
            let mut ground =
                Body::cuboid(DVec3::ZERO, slope, DVec3::new(5.0, 5.0, 0.5), 0.0).fixed();
            ground.friction = friction;
            w.bodies.push(ground);
            let mut b = Body::cuboid(
                slope * DVec3::new(0.0, 0.0, 1.0),
                slope,
                DVec3::splat(0.5),
                1.0,
            );
            b.friction = friction;
            let start = b.pos;
            w.bodies.push(b);
            run(&mut w, 1.0);
            w.bodies[1].pos.distance(start)
        };
        let held = slide(1.0);
        let slid = slide(0.0);
        assert!(held < 0.1, "held: {held}");
        assert!(slid > 1.0, "slid: {slid}");
    }

    #[test]
    fn broadphase_and_sleeping_settle_many_bodies() {
        let mut w = world();
        // Wide static floor.
        w.bodies.push(
            Body::cuboid(
                DVec3::ZERO,
                DQuat::IDENTITY,
                DVec3::new(30.0, 30.0, 0.5),
                0.0,
            )
            .fixed(),
        );
        // 16 separated cubes dropped onto the floor.
        for i in 0..16 {
            let x = (i % 4) as f64 * 2.5 - 3.75;
            let y = (i / 4) as f64 * 2.5 - 3.75;
            w.bodies.push(Body::cuboid(
                DVec3::new(x, y, 1.4),
                DQuat::IDENTITY,
                DVec3::splat(0.4),
                1.0,
            ));
        }
        run(&mut w, 2.5);
        for (idx, b) in w.bodies[1..].iter().enumerate() {
            assert!(
                (b.pos.z - 0.9).abs() < 0.04,
                "box {idx} did not rest on floor: z={}",
                b.pos.z
            );
            assert!(b.sleeping, "box {idx} should have fallen asleep");
        }
    }

    #[test]
    fn hinge_joint_motor_drives_link_to_target_angle() {
        let mut w = RigidWorld::new(DVec3::ZERO, 12);
        // Static base at origin, arm link hinged about Z at (0, 0, 0) with arm centre at (0.5, 0, 0).
        w.bodies
            .push(Body::cuboid(DVec3::ZERO, DQuat::IDENTITY, DVec3::splat(0.2), 0.0).fixed());
        w.bodies.push(Body::cuboid(
            DVec3::new(0.5, 0.0, 0.0),
            DQuat::IDENTITY,
            DVec3::new(0.5, 0.1, 0.1),
            1.0,
        ));
        let mut hinge = RigidJoint::hinge(
            0,
            1,
            DVec3::ZERO,
            DVec3::new(-0.5, 0.0, 0.0),
            DVec3::Z,
            Some(-1.5),
            Some(1.5),
        );
        let target = std::f64::consts::FRAC_PI_3;
        hinge.motor = Some(JointMotor {
            target,
            stiffness: 40.0,
            damping: 8.0,
            max_force: 50.0,
        });
        w.joints.push(hinge);

        run(&mut w, 2.0);
        let angle = w.joint_angle(0);
        assert!(
            (angle - target).abs() < 0.05,
            "hinge angle {angle:.4} did not reach target {target:.4}"
        );
        // Anchor must remain attached to the origin.
        let anchor_world = w.bodies[1].pos + w.bodies[1].rot * DVec3::new(-0.5, 0.0, 0.0);
        assert!(
            anchor_world.length() < 0.03,
            "hinge anchor drifted: {anchor_world:?}"
        );
    }

    #[test]
    fn slider_joint_constrains_motion_to_axis() {
        let mut w = RigidWorld::new(DVec3::new(0.0, 0.0, -9.81), 12);
        // Static rail + dynamic carriage constrained to slide along +X in [-0.2, 0.6].
        w.bodies
            .push(Body::cuboid(DVec3::ZERO, DQuat::IDENTITY, DVec3::splat(0.2), 0.0).fixed());
        w.bodies.push(Body::cuboid(
            DVec3::ZERO,
            DQuat::IDENTITY,
            DVec3::splat(0.2),
            1.0,
        ));
        let mut slider = RigidJoint::slider(
            0,
            1,
            DVec3::ZERO,
            DVec3::ZERO,
            DVec3::X,
            Some(-0.2),
            Some(0.6),
        );
        slider.motor = Some(JointMotor {
            target: 0.45,
            stiffness: 60.0,
            damping: 12.0,
            max_force: 80.0,
        });
        w.joints.push(slider);

        run(&mut w, 2.0);
        let pos = w.bodies[1].pos;
        let offset = w.joint_offset(0);
        assert!(
            (offset - 0.45).abs() < 0.03,
            "slider offset {offset:.4} did not reach 0.45"
        );
        assert!(
            pos.y.abs() < 0.03 && pos.z.abs() < 0.05,
            "slider drifted off X axis under gravity: {pos:?}"
        );
    }
}
