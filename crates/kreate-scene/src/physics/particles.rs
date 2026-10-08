//! Particles and hair: points born on an emitter's surface, and strands grown
//! from it — and the geometry each is drawn as.
//!
//! Both scatter over the surface by area with a seeded sequence, so the same
//! settings make the same scatter every time — a bake repeats, and hair does
//! not crawl when an unrelated setting changes.
//!
//! Drawn as geometry laid onto the emitter's evaluated mesh, in its own
//! space: an instance object's mesh (or a small diamond) at every particle,
//! and a thin ribbon along every strand. A ribbon rather than a line because
//! a mesh's faces are what the renderer draws.

use kreate_math::{DAffine3, DVec3, Vec3};
use kreate_mesh::MeshData;

use super::collide::Collider;
use super::soft::group_weights;
use super::{Particle, ParticleKind, ParticleSettings, Rng};

/// An emitter's triangles, ready to be scattered over by area.
#[derive(Clone, Debug)]
pub struct Surface {
    tris: Vec<[Vec3; 3]>,
    /// Running total of the triangles' areas, for picking one by area.
    cumulative: Vec<f64>,
}

impl Surface {
    pub fn of(mesh: &MeshData) -> Self {
        let tri = mesh.triangulate();
        let (idx, _) = tri.indices.as_chunks::<3>();
        let p = mesh.positions();
        let mut tris = Vec::with_capacity(idx.len());
        let mut cumulative = Vec::with_capacity(idx.len());
        let mut total = 0.0;
        for t in idx {
            let [a, b, c] = t.map(|i| p[i as usize]);
            total += 0.5 * (b - a).cross(c - a).length() as f64;
            tris.push([a, b, c]);
            cumulative.push(total);
        }
        Self { tris, cumulative }
    }

    pub fn is_empty(&self) -> bool {
        self.cumulative.last().is_none_or(|t| *t <= 0.0)
    }

    /// A point on the surface, by area, and the normal there.
    pub fn sample(&self, rng: &mut Rng) -> (DVec3, DVec3) {
        let total = *self.cumulative.last().unwrap_or(&0.0);
        let pick = rng.unit() * total;
        let i = self
            .cumulative
            .partition_point(|c| *c < pick)
            .min(self.tris.len().saturating_sub(1));
        let Some([a, b, c]) = self.tris.get(i).copied() else {
            return (DVec3::ZERO, DVec3::Z);
        };
        let (mut u, mut v) = (rng.unit(), rng.unit());
        if u + v > 1.0 {
            (u, v) = (1.0 - u, 1.0 - v);
        }
        let (a, b, c) = (a.as_dvec3(), b.as_dvec3(), c.as_dvec3());
        let p = a + (b - a) * u + (c - a) * v;
        let n = (b - a).cross(c - a).normalize_or(DVec3::Z);
        (p, n)
    }
}

/// A particle while it is alive.
#[derive(Clone, Copy, Debug)]
struct Live {
    birth: f64,
    x: DVec3,
    v: DVec3,
}

/// One emitter's particles being simulated.
#[derive(Clone, Debug)]
pub struct ParticleSim {
    settings: ParticleSettings,
    surface: Surface,
    rng: Rng,
    /// Each particle's birth frame, in order.
    births: Vec<f64>,
    next: usize,
    live: Vec<Live>,
}

impl ParticleSim {
    pub fn new(mesh: &MeshData, settings: &ParticleSettings) -> Self {
        let count = settings.count as usize;
        let (start, end) = (
            settings.frame_start as f64,
            settings.frame_end.max(settings.frame_start) as f64,
        );
        let births = (0..count)
            .map(|i| start + (end - start) * i as f64 / count.max(1) as f64)
            .collect();
        Self {
            settings: settings.clone(),
            surface: Surface::of(mesh),
            rng: Rng::new(settings.seed as u64 + 1),
            births,
            next: 0,
            live: Vec::new(),
        }
    }

    /// From frame `frame - 1` to `frame`, `dt` seconds, the emitter placed by
    /// `world`.
    pub fn advance(
        &mut self,
        frame: f64,
        dt: f64,
        substeps: u32,
        gravity: DVec3,
        world: DAffine3,
        colliders: &[&Collider],
    ) {
        if !self.surface.is_empty() {
            let turn = world.matrix3;
            while self.next < self.births.len() && self.births[self.next] <= frame {
                let (p, n) = self.surface.sample(&mut self.rng);
                let n = (turn * n).normalize_or(DVec3::Z);
                let v = n * self.settings.normal_speed
                    + self.rng.direction() * self.settings.random_speed;
                self.live.push(Live {
                    birth: self.births[self.next],
                    x: world.transform_point3(p),
                    v,
                });
                self.next += 1;
            }
        }
        let lifetime = self.settings.lifetime.max(0.0);
        self.live.retain(|p| frame - p.birth <= lifetime);
        let n = substeps.max(1);
        let h = dt / n as f64;
        let pull = gravity * self.settings.gravity;
        let keep = (1.0 - self.settings.drag.clamp(0.0, 1.0)).powf(h);
        for _ in 0..n {
            for p in self.live.iter_mut() {
                p.v = (p.v + pull * h) * keep;
                let to = p.x + p.v * h;
                let mut at = to;
                for c in colliders {
                    if let Some(hit) = c.resolve(p.x.as_vec3(), at.as_vec3()) {
                        at = hit.point.as_dvec3();
                        let normal = hit.normal.as_dvec3();
                        let vn = p.v.dot(normal);
                        if vn < 0.0 {
                            let along = p.v - normal * vn;
                            p.v = along * (1.0 - c.friction as f64)
                                - normal * (vn * c.bounciness as f64);
                        }
                    }
                }
                p.x = at;
            }
        }
    }

    /// The particles alive, in the emitter's own space.
    pub fn snapshot(&self, world: DAffine3) -> Vec<Particle> {
        let back = world.inverse();
        self.live
            .iter()
            .map(|p| Particle {
                position: back.transform_point3(p.x).as_vec3(),
                heading: (back.matrix3 * p.v).normalize_or_zero().as_vec3(),
            })
            .collect()
    }
}

/// Every strand of a hair system at rest, in the emitter's own space: its
/// root and `segments` points on along the surface's normal, strand after
/// strand. Where a vertex group has the system's name, hair grows only where
/// it is painted.
pub fn hair_rest(mesh: &MeshData, settings: &ParticleSettings) -> Vec<Particle> {
    let surface = Surface::of(mesh);
    if surface.is_empty() {
        return Vec::new();
    }
    let segments = settings.hair_segments.clamp(1, 64) as usize;
    let mut rng = Rng::new(settings.seed as u64 + 1);
    let density = group_weights(mesh, &settings.name);
    let mut out = Vec::with_capacity(settings.count as usize * (segments + 1));
    let mut made = 0;
    let mut tries = 0;
    while made < settings.count && tries < settings.count * 8 {
        tries += 1;
        let (root, n) = surface.sample(&mut rng);
        if let Some(w) = density {
            // The weight of the vertex nearest the root, as a chance of
            // growing there.
            let near = mesh
                .positions()
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.as_dvec3()
                        .distance_squared(root)
                        .total_cmp(&b.1.as_dvec3().distance_squared(root))
                })
                .map(|(i, _)| w[i])
                .unwrap_or(1.0);
            if rng.unit() as f32 >= near {
                continue;
            }
        }
        for k in 0..=segments {
            out.push(Particle {
                position: (root + n * (settings.hair_length * k as f64 / segments as f64))
                    .as_vec3(),
                heading: n.as_vec3(),
            });
        }
        made += 1;
    }
    out
}

/// A hair system's strands, swinging: each point after the root held to the
/// one before by its length, the root where the emitter puts it.
#[derive(Clone, Debug)]
pub struct HairSim {
    segments: usize,
    rest: Vec<DVec3>,
    x: Vec<DVec3>,
    prev: Vec<DVec3>,
    v: Vec<DVec3>,
    length: f64,
    stiffness: f64,
    gravity: f64,
    drag: f64,
}

impl HairSim {
    pub fn new(mesh: &MeshData, settings: &ParticleSettings, world: DAffine3) -> Self {
        let rest: Vec<DVec3> = hair_rest(mesh, settings)
            .iter()
            .map(|p| p.position.as_dvec3())
            .collect();
        let x: Vec<DVec3> = rest.iter().map(|p| world.transform_point3(*p)).collect();
        let segments = settings.hair_segments.clamp(1, 64) as usize;
        Self {
            segments,
            length: settings.hair_length / segments as f64,
            prev: x.clone(),
            v: vec![DVec3::ZERO; x.len()],
            x,
            rest,
            stiffness: settings.hair_stiffness.clamp(0.0, 1.0),
            gravity: settings.gravity,
            drag: settings.drag.clamp(0.0, 1.0),
        }
    }

    pub fn advance(
        &mut self,
        dt: f64,
        substeps: u32,
        gravity: DVec3,
        world: DAffine3,
        colliders: &[&Collider],
    ) {
        let n = substeps.max(1);
        let h = dt / n as f64;
        let per = self.segments + 1;
        let keep = (1.0 - self.drag.max(0.05)).powf(h);
        for _ in 0..n {
            for i in 0..self.x.len() {
                self.prev[i] = self.x[i];
                if i % per == 0 {
                    self.x[i] = world.transform_point3(self.rest[i]);
                    continue;
                }
                self.v[i] = (self.v[i] + gravity * self.gravity * h) * keep;
                self.x[i] += self.v[i] * h;
            }
            for _ in 0..2 {
                for s in 0..self.x.len() / per {
                    let base = s * per;
                    for k in 1..per {
                        // Toward the strand's rest shape, as the emitter now
                        // holds it: what a stiff strand springs back to.
                        let goal = world.transform_point3(self.rest[base + k]);
                        let pull = self.stiffness * 0.05;
                        let to = goal - self.x[base + k];
                        self.x[base + k] += to * pull;
                        // Its length from the point before, which is held.
                        let a = self.x[base + k - 1];
                        let d = self.x[base + k] - a;
                        let len = d.length().max(1e-12);
                        self.x[base + k] = a + d / len * self.length;
                    }
                }
            }
            for i in 0..self.x.len() {
                if i % per == 0 {
                    continue;
                }
                for c in colliders {
                    if let Some(hit) = c.resolve(self.prev[i].as_vec3(), self.x[i].as_vec3()) {
                        self.x[i] = hit.point.as_dvec3();
                    }
                }
                self.v[i] = (self.x[i] - self.prev[i]) / h;
            }
        }
    }

    pub fn snapshot(&self, world: DAffine3) -> Vec<Particle> {
        let back = world.inverse();
        let per = self.segments + 1;
        (0..self.x.len())
            .map(|i| {
                let here = self.x[i];
                let next = if i % per < self.segments {
                    self.x[i + 1]
                } else {
                    here + (here - self.x[i - 1])
                };
                Particle {
                    position: back.transform_point3(here).as_vec3(),
                    heading: (back.matrix3 * (next - here)).normalize_or_zero().as_vec3(),
                }
            })
            .collect()
    }
}

/// A small diamond: what a particle is without an instance object.
fn diamond(r: f32) -> (Vec<Vec3>, Vec<Vec<u32>>) {
    let p = vec![
        Vec3::X * r,
        -Vec3::X * r,
        Vec3::Y * r,
        -Vec3::Y * r,
        Vec3::Z * r,
        -Vec3::Z * r,
    ];
    let f = vec![
        vec![0, 2, 4],
        vec![2, 1, 4],
        vec![1, 3, 4],
        vec![3, 0, 4],
        vec![2, 0, 5],
        vec![1, 2, 5],
        vec![3, 1, 5],
        vec![0, 3, 5],
    ];
    (p, f)
}

/// A copy of `instance` — or a diamond — at every particle.
pub fn particle_mesh(particles: &[Particle], size: f64, instance: Option<&MeshData>) -> MeshData {
    let (shape, faces): (Vec<Vec3>, Vec<Vec<u32>>) = match instance {
        Some(m) => (
            m.positions().iter().map(|p| *p * size as f32).collect(),
            (0..m.num_faces())
                .map(|f| m.face_verts(f).to_vec())
                .collect(),
        ),
        None => diamond(size as f32),
    };
    let mut positions = Vec::with_capacity(particles.len() * shape.len());
    let mut all = Vec::with_capacity(particles.len() * faces.len());
    for p in particles {
        let base = positions.len() as u32;
        positions.extend(shape.iter().map(|s| p.position + *s));
        all.extend(
            faces
                .iter()
                .map(|f| f.iter().map(|i| i + base).collect::<Vec<u32>>()),
        );
    }
    MeshData::from_faces(positions, &all)
}

/// A thin ribbon along every strand, narrowing to its tip.
pub fn hair_mesh(points: &[Particle], segments: u32, width: f64) -> MeshData {
    let per = segments.clamp(1, 64) as usize + 1;
    let mut positions = Vec::with_capacity(points.len() * 2);
    let mut faces = Vec::new();
    for strand in points.chunks_exact(per) {
        let base = positions.len() as u32;
        let root_heading = strand[0].heading;
        // Square to the strand and to something it is not along.
        let other = if root_heading.z.abs() < 0.9 {
            Vec3::Z
        } else {
            Vec3::X
        };
        let side = root_heading.cross(other).normalize_or(Vec3::X);
        for (k, p) in strand.iter().enumerate() {
            let w = width as f32 * 0.5 * (1.0 - 0.8 * k as f32 / (per - 1) as f32);
            positions.push(p.position - side * w);
            positions.push(p.position + side * w);
        }
        for k in 0..per as u32 - 1 {
            let a = base + 2 * k;
            faces.push(vec![a, a + 1, a + 3, a + 2]);
        }
    }
    MeshData::from_faces(positions, &faces)
}

/// What a particle system adds to its emitter at a frame: particles if it
/// has baked ones, hair baked or at rest. `None` when there is nothing.
pub fn system_mesh(
    mesh: &MeshData,
    settings: &ParticleSettings,
    baked: Option<&[Particle]>,
    instance: Option<&MeshData>,
) -> Option<MeshData> {
    match settings.kind {
        ParticleKind::Emitter => baked.map(|p| particle_mesh(p, settings.size, instance)),
        ParticleKind::Hair => {
            let rest;
            let points = match baked {
                Some(p) => p,
                None => {
                    rest = hair_rest(mesh, settings);
                    &rest
                }
            };
            (!points.is_empty()).then(|| hair_mesh(points, settings.hair_segments, settings.size))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: DVec3 = DVec3::new(0.0, 0.0, -9.81);

    #[test]
    fn particles_are_born_over_the_range_and_die_at_their_lifetime() {
        let plane = kreate_mesh::primitives::plane(2.0);
        let mut s = ParticleSettings::new("P", ParticleKind::Emitter);
        (s.count, s.frame_start, s.frame_end, s.lifetime) = (100, 1, 10, 5.0);
        let mut sim = ParticleSim::new(&plane, &s);
        sim.advance(1.0, 1.0 / 24.0, 4, G, DAffine3::IDENTITY, &[]);
        assert!(
            !sim.live.is_empty() && sim.live.len() <= 11,
            "a tenth at the first frame: {}",
            sim.live.len()
        );
        for f in 2..=10 {
            sim.advance(f as f64, 1.0 / 24.0, 4, G, DAffine3::IDENTITY, &[]);
        }
        // Born up to frame 10, each living 5 frames: those of frames 5–10.
        assert!(
            sim.live.len() < 70 && sim.live.len() > 40,
            "{}",
            sim.live.len()
        );
        assert_eq!(sim.next, 100, "all born");
        for f in 11..=20 {
            sim.advance(f as f64, 1.0 / 24.0, 4, G, DAffine3::IDENTITY, &[]);
        }
        assert!(sim.live.is_empty());
    }

    #[test]
    fn particles_shot_up_fall_back_and_bounce_off_a_floor() {
        let emitter = kreate_mesh::primitives::plane(0.2);
        let mut s = ParticleSettings::new("P", ParticleKind::Emitter);
        (s.count, s.frame_start, s.frame_end, s.lifetime) = (20, 1, 1, 500.0);
        (s.normal_speed, s.random_speed) = (3.0, 0.0);
        let lift = DAffine3::from_translation(DVec3::new(0.0, 0.0, 1.0));
        let floor = Collider::new(
            &kreate_mesh::primitives::plane(20.0),
            DAffine3::IDENTITY,
            0.01,
            0.0,
            0.5,
        );
        let mut sim = ParticleSim::new(&emitter, &s);
        let mut highest: f64 = 0.0;
        for f in 1..=96 {
            sim.advance(f as f64, 1.0 / 24.0, 4, G, lift, &[&floor]);
            highest = highest.max(sim.live[0].x.z);
            for p in &sim.live {
                assert!(p.x.z > -1e-3, "never below the floor: {}", p.x.z);
            }
        }
        assert!(highest > 1.3, "went up: {highest}");
        assert!(sim.live.iter().all(|p| p.x.z < 0.5), "came down");
    }

    #[test]
    fn hair_stands_on_its_surface_and_the_same_seed_grows_the_same() {
        let plane = kreate_mesh::primitives::plane(2.0);
        let s = ParticleSettings::new("Hair", ParticleKind::Hair);
        let a = hair_rest(&plane, &s);
        assert_eq!(a.len(), (s.count * (s.hair_segments + 1)) as usize);
        assert_eq!(a, hair_rest(&plane, &s));
        let tip = a[s.hair_segments as usize];
        assert!(
            (tip.position.z - s.hair_length as f32).abs() < 1e-5,
            "{:?}",
            tip.position
        );
        let mesh = hair_mesh(&a, s.hair_segments, s.size);
        assert_eq!(mesh.num_faces(), s.count * s.hair_segments);
    }

    #[test]
    fn dynamic_hair_droops_under_gravity_and_its_roots_stay() {
        let plane = kreate_mesh::primitives::plane(2.0);
        let mut s = ParticleSettings::new("Hair", ParticleKind::Hair);
        (s.count, s.hair_stiffness, s.hair_length) = (20, 0.0, 0.5);
        // Sideways: the hair grows out along +X and gravity pulls it down.
        let turn = DAffine3::from_rotation_y(std::f64::consts::FRAC_PI_2);
        let mut sim = HairSim::new(&plane, &s, turn);
        for _ in 0..48 {
            sim.advance(1.0 / 24.0, 8, G, turn, &[]);
        }
        let per = (s.hair_segments + 1) as usize;
        for strand in sim.x.chunks(per) {
            assert!(
                strand[per - 1].z < strand[0].z - 0.3,
                "the tip hangs below the root"
            );
        }
        let at = sim.snapshot(turn);
        let rest = hair_rest(&plane, &s);
        assert!(
            at[0].position.distance(rest[0].position) < 1e-5,
            "roots stay"
        );
    }

    #[test]
    fn a_particle_is_drawn_as_its_instance() {
        let cube = kreate_mesh::primitives::cube(2.0);
        let ps = [
            Particle {
                position: Vec3::ZERO,
                heading: Vec3::Z,
            },
            Particle {
                position: Vec3::X * 5.0,
                heading: Vec3::Z,
            },
        ];
        let m = particle_mesh(&ps, 0.5, Some(&cube));
        assert_eq!(m.num_verts(), 16);
        assert_eq!(m.num_faces(), 12);
        let far = m.positions()[8..]
            .iter()
            .map(|p| p.x)
            .fold(f32::MIN, f32::max);
        assert!((far - 5.5).abs() < 1e-5, "half-size cube at x = 5: {far}");
        assert_eq!(particle_mesh(&ps, 0.1, None).num_faces(), 16, "diamonds");
    }
}
