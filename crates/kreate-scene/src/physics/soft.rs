//! Cloth and soft bodies: a mesh's vertices as particles held by its edges.
//!
//! # The method
//!
//! Position-based dynamics in small steps (Macklin et al., *Small Steps in
//! Physics Simulation*): each step moves every vertex by its velocity, then
//! pulls the positions back toward what the constraints want, then takes the
//! velocity from how far each vertex actually went. Stable at any stiffness,
//! which is what lets a slider go from silk to plywood without the bake
//! exploding.
//!
//! The constraints:
//! - **edges** keep their rest length — stretch;
//! - across each pair of triangles sharing an edge, the two far corners keep
//!   their distance — bending;
//! - a soft body's vertices are drawn toward its rest shape, turned and moved
//!   to fit where it is (shape matching, Müller et al. 2005) — springing back;
//! - a closed soft body keeps its volume, times one plus the pressure.
//!
//! Pinned vertices are not simulated: they are where the object, as animated,
//! puts them.

use kreate_math::glam::DMat3;
use kreate_math::{DAffine3, DQuat, DVec3, Vec3};
use kreate_mesh::attribute::{AttrData, Domain};
use kreate_mesh::MeshData;

use super::collide::Collider;
use super::{SoftBody, SoftKind};

/// One cloth or soft body being simulated.
#[derive(Clone, Debug)]
pub struct SoftSim {
    /// Positions in the world.
    pub x: Vec<DVec3>,
    prev: Vec<DVec3>,
    v: Vec<DVec3>,
    inv_mass: Vec<f64>,
    /// Where each vertex is in the object's own space, at rest.
    rest: Vec<DVec3>,
    pinned: Vec<bool>,
    edges: Vec<(u32, u32, f64)>,
    bends: Vec<(u32, u32, f64)>,
    tris: Vec<[u32; 3]>,
    /// The rest volume, for a closed mesh; `None` for an open one.
    volume: Option<f64>,
    /// Shape matching: each vertex's rest offset from the centre of mass.
    offsets: Vec<DVec3>,
    turn: DQuat,
    stretch: f64,
    bend: f64,
    damping: f64,
    shape: f64,
    pressure: f64,
}

/// The weights of the vertex group `group`, if the mesh has it.
pub fn group_weights<'a>(mesh: &'a MeshData, group: &str) -> Option<&'a [f32]> {
    let layer = mesh.attributes().get(&crate::skin::weight_layer(group))?;
    match (&layer.domain, &layer.data) {
        (Domain::Point, AttrData::F32(w)) if w.len() == mesh.num_verts() as usize => Some(w),
        _ => None,
    }
}

impl SoftSim {
    /// The mesh `mesh` (in its own space) placed by `world`, as `settings`
    /// says.
    pub fn new(mesh: &MeshData, world: DAffine3, settings: &SoftBody) -> Self {
        let n = mesh.num_verts() as usize;
        let rest: Vec<DVec3> = mesh.positions().iter().map(|p| p.as_dvec3()).collect();
        let x: Vec<DVec3> = rest.iter().map(|p| world.transform_point3(*p)).collect();
        let weights = settings
            .pin_group
            .as_deref()
            .and_then(|g| group_weights(mesh, g));
        let pinned: Vec<bool> = (0..n)
            .map(|i| weights.is_some_and(|w| w[i] > 0.5))
            .collect();
        let each = settings.mass.max(1e-6) / n.max(1) as f64;
        let inv_mass = pinned.iter().map(|p| if *p { 0.0 } else { 1.0 / each }).collect();
        let edges = mesh
            .edges()
            .iter()
            .map(|[a, b]| (*a, *b, x[*a as usize].distance(x[*b as usize])))
            .collect();
        let tri = mesh.triangulate();
        let (tris, _) = tri.indices.as_chunks::<3>();
        let tris = tris.to_vec();
        // The far corners of each pair of triangles sharing an edge.
        let mut by_edge: std::collections::HashMap<(u32, u32), Vec<u32>> = Default::default();
        for t in &tris {
            for k in 0..3 {
                let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                by_edge.entry((a.min(b), a.max(b))).or_default().push(c);
            }
        }
        let mut bends = Vec::new();
        let mut closed = !tris.is_empty();
        for far in by_edge.values() {
            closed &= far.len() == 2;
            if let [c, d] = far[..] {
                if c != d {
                    bends.push((c, d, x[c as usize].distance(x[d as usize])));
                }
            }
        }
        let soft = settings.kind == SoftKind::Soft;
        let centre = x.iter().copied().sum::<DVec3>() / n.max(1) as f64;
        let mut sim = Self {
            offsets: x.iter().map(|p| *p - centre).collect(),
            prev: x.clone(),
            v: vec![DVec3::ZERO; n],
            x,
            inv_mass,
            rest,
            pinned,
            edges,
            bends,
            tris,
            volume: None,
            turn: DQuat::IDENTITY,
            stretch: settings.stiffness.clamp(0.0, 1.0),
            bend: settings.bending.clamp(0.0, 1.0),
            damping: settings.damping.clamp(0.0, 1.0),
            shape: if soft { settings.shape_stiffness.clamp(0.0, 1.0) } else { 0.0 },
            pressure: if soft { settings.pressure.clamp(-1.0, 1.0) } else { 0.0 },
        };
        if soft && closed {
            sim.volume = Some(sim.current_volume());
        }
        sim
    }

    fn current_volume(&self) -> f64 {
        self.tris
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| self.x[i as usize]);
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    /// The vertices in the object's own space, for `world` the object's
    /// placement now.
    pub fn local(&self, world: DAffine3) -> Vec<Vec3> {
        let back = world.inverse();
        self.x
            .iter()
            .map(|p| back.transform_point3(*p).as_vec3())
            .collect()
    }

    /// One frame of `dt` seconds in `substeps` steps; `world` is where the
    /// object is now, for the pinned vertices.
    pub fn advance(
        &mut self,
        dt: f64,
        substeps: u32,
        iterations: u32,
        gravity: DVec3,
        world: DAffine3,
        colliders: &[&Collider],
    ) {
        let n = substeps.max(1);
        for _ in 0..n {
            self.step(dt / n as f64, iterations.max(1), gravity, world, colliders);
        }
    }

    fn step(&mut self, dt: f64, iterations: u32, gravity: DVec3, world: DAffine3, colliders: &[&Collider]) {
        let keep = (1.0 - self.damping).powf(dt);
        for i in 0..self.x.len() {
            self.prev[i] = self.x[i];
            if self.pinned[i] {
                self.x[i] = world.transform_point3(self.rest[i]);
                continue;
            }
            self.v[i] = (self.v[i] + gravity * dt) * keep;
            self.x[i] += self.v[i] * dt;
        }
        for _ in 0..iterations {
            for &(a, b, rest) in &self.edges {
                distance(&mut self.x, &self.inv_mass, a, b, rest, self.stretch);
            }
            for &(a, b, rest) in &self.bends {
                distance(&mut self.x, &self.inv_mass, a, b, rest, self.bend);
            }
            if let Some(rest) = self.volume {
                self.keep_volume(rest * (1.0 + self.pressure));
            }
        }
        if self.shape > 0.0 {
            self.match_shape();
        }
        for i in 0..self.x.len() {
            if self.inv_mass[i] == 0.0 {
                continue;
            }
            for c in colliders {
                let (from, to) = (self.prev[i].as_vec3(), self.x[i].as_vec3());
                if let Some(hit) = c.resolve(from, to) {
                    // Pushed out, and the sliding along the surface slowed by
                    // its friction.
                    let moved = (hit.point - from).as_dvec3();
                    let n = hit.normal.as_dvec3();
                    let along = moved - n * moved.dot(n);
                    self.x[i] = hit.point.as_dvec3() - along * c.friction as f64;
                }
            }
        }
        for i in 0..self.x.len() {
            self.v[i] = (self.x[i] - self.prev[i]) / dt;
        }
    }

    fn keep_volume(&mut self, target: f64) {
        let volume = self.current_volume();
        let mut grads = vec![DVec3::ZERO; self.x.len()];
        for t in &self.tris {
            let [a, b, c] = t.map(|i| self.x[i as usize]);
            grads[t[0] as usize] += b.cross(c) / 6.0;
            grads[t[1] as usize] += c.cross(a) / 6.0;
            grads[t[2] as usize] += a.cross(b) / 6.0;
        }
        let w: f64 = grads
            .iter()
            .zip(&self.inv_mass)
            .map(|(g, m)| g.length_squared() * m)
            .sum();
        if w < 1e-12 {
            return;
        }
        let lambda = -(volume - target) / w;
        for (i, g) in grads.iter().enumerate() {
            self.x[i] += *g * (lambda * self.inv_mass[i]);
        }
    }

    /// Draws the vertices toward the rest shape, moved and turned to where
    /// the body is.
    fn match_shape(&mut self) {
        let total: f64 = self.inv_mass.iter().map(|w| if *w > 0.0 { 1.0 / w } else { 0.0 }).sum();
        if total <= 0.0 {
            return;
        }
        let centre = self
            .x
            .iter()
            .zip(&self.inv_mass)
            .map(|(p, w)| if *w > 0.0 { *p / *w } else { DVec3::ZERO })
            .sum::<DVec3>()
            / total;
        let mut a = DMat3::ZERO;
        for (p, q) in self.x.iter().zip(&self.offsets) {
            let d = *p - centre;
            a += DMat3::from_cols(d * q.x, d * q.y, d * q.z);
        }
        self.turn = rotation_of(a, self.turn);
        for i in 0..self.x.len() {
            if self.inv_mass[i] > 0.0 {
                let goal = centre + self.turn * self.offsets[i];
                let to = goal - self.x[i];
                self.x[i] += to * self.shape;
            }
        }
    }
}

fn distance(x: &mut [DVec3], w: &[f64], a: u32, b: u32, rest: f64, stiffness: f64) {
    let (a, b) = (a as usize, b as usize);
    let total = w[a] + w[b];
    if total <= 0.0 {
        return;
    }
    let d = x[a] - x[b];
    let len = d.length();
    if len < 1e-12 {
        return;
    }
    let shift = d / len * ((len - rest) / total * stiffness);
    x[a] -= shift * w[a];
    x[b] += shift * w[b];
}

/// The rotation in `a`, found by iterating from `guess` (Müller et al.,
/// *A Robust Method to Extract the Rotational Part of Deformations*, 2016).
fn rotation_of(a: DMat3, guess: DQuat) -> DQuat {
    let mut q = guess;
    for _ in 0..20 {
        let r = DMat3::from_quat(q);
        let omega = r.x_axis.cross(a.x_axis) + r.y_axis.cross(a.y_axis) + r.z_axis.cross(a.z_axis);
        let denom = (r.x_axis.dot(a.x_axis) + r.y_axis.dot(a.y_axis) + r.z_axis.dot(a.z_axis)).abs() + 1e-9;
        let omega = omega / denom;
        let angle = omega.length();
        if angle < 1e-9 {
            break;
        }
        q = (DQuat::from_axis_angle(omega / angle, angle) * q).normalize();
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: DVec3 = DVec3::new(0.0, 0.0, -9.81);

    fn cloth(divs: u32) -> MeshData {
        kreate_mesh::primitives::grid(2.0, divs, divs)
    }

    fn run(sim: &mut SoftSim, frames: usize, world: DAffine3, colliders: &[&Collider]) {
        for _ in 0..frames {
            sim.advance(1.0 / 24.0, 10, 2, G, world, colliders);
        }
    }

    #[test]
    fn cloth_pinned_along_one_edge_hangs_from_it() {
        let mut mesh = cloth(10);
        // Pin the row along y = +1.
        let w: Vec<f32> = mesh.positions().iter().map(|p| if p.y > 0.99 { 1.0 } else { 0.0 }).collect();
        mesh.attributes_mut().insert(
            &crate::skin::weight_layer("Pin"),
            Domain::Point,
            AttrData::F32(std::sync::Arc::new(w)),
        );
        // Damped hard, so it has stopped swinging when it is looked at.
        let settings = SoftBody {
            pin_group: Some("Pin".into()),
            damping: 0.9,
            ..SoftBody::new(SoftKind::Cloth)
        };
        let mut sim = SoftSim::new(&mesh, DAffine3::IDENTITY, &settings);
        run(&mut sim, 144, DAffine3::IDENTITY, &[]);
        let lowest = sim.x.iter().map(|p| p.z).fold(f64::MAX, f64::min);
        assert!(lowest < -1.5, "it hangs: {lowest}");
        assert!(lowest > -2.3, "and does not stretch far: {lowest}");
        for (p, r) in sim.x.iter().zip(&sim.rest) {
            if r.y > 0.99 {
                assert!(p.distance(*r) < 1e-9, "a pinned vertex stays");
            }
        }
    }

    #[test]
    fn cloth_falls_onto_a_box_and_drapes_over_it() {
        let mesh = cloth(16);
        let lift = DAffine3::from_translation(DVec3::new(0.0, 0.0, 1.5));
        let mut sim = SoftSim::new(&mesh, lift, &SoftBody::new(SoftKind::Cloth));
        let cube = kreate_mesh::primitives::cube(1.0);
        let box_ = Collider::new(&cube, DAffine3::IDENTITY, 0.02, 0.5, 0.0);
        run(&mut sim, 48, lift, &[&box_]);
        // The middle rests on the lid, the corners hang past the sides.
        let middle = sim
            .x
            .iter()
            .filter(|p| p.x.abs() < 0.2 && p.y.abs() < 0.2)
            .map(|p| p.z)
            .fold(f64::MAX, f64::min);
        assert!(middle > 0.45, "held up by the lid: {middle}");
        let lowest = sim.x.iter().map(|p| p.z).fold(f64::MAX, f64::min);
        assert!(lowest < 0.0, "the edges fall past it: {lowest}");
    }

    #[test]
    fn a_soft_cube_lands_and_keeps_its_shape() {
        let cube = kreate_mesh::primitives::cube(1.0);
        let lift = DAffine3::from_translation(DVec3::new(0.0, 0.0, 2.0));
        let mut settings = SoftBody::new(SoftKind::Soft);
        settings.shape_stiffness = 0.5;
        let mut sim = SoftSim::new(&cube, lift, &settings);
        let floor = Collider::new(&kreate_mesh::primitives::plane(10.0), DAffine3::IDENTITY, 0.01, 0.5, 0.0);
        run(&mut sim, 72, lift, &[&floor]);
        let (lo, hi) = sim
            .x
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(p.z), hi.max(p.z)));
        assert!(lo > -0.05, "on the floor: {lo}");
        assert!(hi - lo > 0.7, "still a cube, not a puddle: {}", hi - lo);
    }

    #[test]
    fn the_rotation_of_a_turned_matrix_is_found() {
        let q = DQuat::from_rotation_z(0.7) * DQuat::from_rotation_x(0.3);
        let a = DMat3::from_quat(q) * DMat3::from_diagonal(DVec3::new(2.0, 1.0, 0.5));
        let r = rotation_of(a, DQuat::IDENTITY);
        assert!(r.dot(q).abs() > 0.9999, "{r:?} vs {q:?}");
    }
}
