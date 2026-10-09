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
        for (a, b) in [
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
        ] {
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
                ..
            },
            Shape::Hull {
                points: pb,
                planes: qb,
                ..
            },
        ) => {
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
            for n in qa
                .iter()
                .map(|(n, _)| ba.rot * *n)
                .chain(qb.iter().map(|(n, _)| bb.rot * *n))
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
                }
            }
            let Some((n, _)) = best else { return };
            let (_, b_top) = along(&wb, n);
            let (a_bottom, _) = along(&wa, n);
            for w in &wa {
                if inside_hull(bb, qb, *w).is_some() {
                    push(*w, n, (b_top - w.dot(n)).max(0.0));
                }
            }
            for w in &wb {
                if inside_hull(ba, qa, *w).is_some() {
                    push(*w, n, (w.dot(n) - a_bottom).max(0.0));
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

/// The bodies, and what pulls them.
#[derive(Clone, Debug)]
pub struct RigidWorld {
    pub bodies: Vec<Body>,
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

impl RigidWorld {
    pub fn new(gravity: DVec3, iterations: u32) -> Self {
        Self {
            bodies: Vec::new(),
            gravity,
            iterations: iterations.max(1),
        }
    }

    /// One frame of `dt` seconds, in `substeps` steps.
    pub fn advance(&mut self, dt: f64, substeps: u32) {
        let n = substeps.max(1);
        for _ in 0..n {
            self.step(dt / n as f64);
        }
    }

    fn contacts(&self) -> Vec<Contact> {
        let mut out = Vec::new();
        for a in 0..self.bodies.len() {
            for b in a + 1..self.bodies.len() {
                let (ba, bb) = (&self.bodies[a], &self.bodies[b]);
                if !ba.is_dynamic() && !bb.is_dynamic() {
                    continue;
                }
                let near = match (&ba.shape, &bb.shape) {
                    (Shape::Mesh(m), _) => overlaps_box(m, bb.pos, bb.reach),
                    (_, Shape::Mesh(m)) => overlaps_box(m, ba.pos, ba.reach),
                    _ => ba.pos.distance(bb.pos) <= ba.reach + bb.reach,
                };
                if near {
                    contacts_between(&self.bodies, a, b, &mut out);
                }
            }
        }
        out
    }

    /// One step of `dt` seconds.
    pub fn step(&mut self, dt: f64) {
        for b in self.bodies.iter_mut().filter(|b| b.is_dynamic()) {
            b.vel += self.gravity * b.gravity * dt;
            b.vel *= (1.0 - b.linear_damping.clamp(0.0, 1.0)).powf(dt);
            b.spin *= (1.0 - b.angular_damping.clamp(0.0, 1.0)).powf(dt);
        }
        let mut prepared: Vec<Prepared> = self
            .contacts()
            .into_iter()
            .map(|c| {
                let (a, b) = (&self.bodies[c.a], &self.bodies[c.b]);
                let ra = c.point - a.pos;
                let rb = c.point - b.pos;
                let n = c.normal;
                let k = a.inv_mass
                    + b.inv_mass
                    + a.inv_inertia_world(ra.cross(n)).cross(ra).dot(n)
                    + b.inv_inertia_world(rb.cross(n)).cross(rb).dot(n);
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
        for _ in 0..self.iterations {
            for p in prepared.iter_mut() {
                let (ia, ib) = (p.c.a, p.c.b);
                let n = p.c.normal;
                let rel = self.bodies[ia].velocity_at(p.ra) - self.bodies[ib].velocity_at(p.rb);
                let vn = rel.dot(n);
                let total = (p.pushed + (p.target - vn) * p.normal_mass).max(0.0);
                let dl = total - p.pushed;
                p.pushed = total;
                self.bodies[ia].apply(n * dl, p.ra);
                self.bodies[ib].apply(-n * dl, p.rb);
                // Friction, against the sliding, no more than μ times the push.
                let rel = self.bodies[ia].velocity_at(p.ra) - self.bodies[ib].velocity_at(p.rb);
                let tangential = rel - n * rel.dot(n);
                let speed = tangential.length();
                if speed > 1e-9 {
                    let t = tangential / speed;
                    let (a, b) = (&self.bodies[ia], &self.bodies[ib]);
                    let k = a.inv_mass
                        + b.inv_mass
                        + a.inv_inertia_world(p.ra.cross(t)).cross(p.ra).dot(t)
                        + b.inv_inertia_world(p.rb.cross(t)).cross(p.rb).dot(t);
                    if k > 1e-12 {
                        let jt = (speed / k).min(p.friction * p.pushed);
                        self.bodies[ia].apply(-t * jt, p.ra);
                        self.bodies[ib].apply(t * jt, p.rb);
                    }
                }
            }
        }
        for b in self.bodies.iter_mut().filter(|b| b.is_dynamic()) {
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
}
