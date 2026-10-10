//! Physics: what an object is to the simulation, the scene's world, and the
//! baked result.
//!
//! # Baked, never live
//!
//! A simulation's next frame comes from its last one, so unlike an animation
//! curve it is not a function of the frame: scrubbing backwards, or jumping to
//! frame 200, cannot be answered without running every frame before it. So it
//! is run once, from the start of the range to its end — the bake — and what
//! plays is the record. That is Blender's model, and the plan's (§2.4: what
//! several frames want at once is an explicit bake).
//!
//! # Where the record goes
//!
//! - A **rigid body** moves as a whole, so its record is a placement per
//!   frame, written into the object's transform as the frame changes —
//!   [`crate::document::Document::animate`] — exactly as an action's is. Bake
//!   to Keyframes turns it into an action for anything that reads curves
//!   (glTF).
//! - **Cloth** and **soft bodies** change shape, so theirs is the vertices per
//!   frame, laid over the evaluated mesh — see
//!   [`crate::modifier::ModifierCache`].
//! - **Particles** and **hair** are geometry the emitter grows, also laid on
//!   in evaluation.
//! - A **liquid**'s record is its surface, a mesh per frame, which its
//!   domain shows in place of its own box — see [`fluid`].
//!
//! The record is the document's, behind an `Arc`, so undo keeps it for the
//! price of a pointer and a bake is one undo step. It is saved with the file
//! (`kreate_io::kbake`), so a baked scene opens baked.
//!
//! # What touches what
//!
//! As in Blender: rigid bodies collide with one another; cloth, soft bodies
//! and particles collide with objects that have [`Collision`] — where the
//! simulation has them, if they are rigid bodies too. A passive rigid body
//! is not moved by the simulation but follows its object's animation, and
//! carries what it touches.

pub mod bake;
pub mod collide;
pub mod fluid;
pub mod particles;
pub mod rigid;
pub mod soft;

use std::sync::Arc;

use kreate_math::{DTransform, DVec3, Vec3};
use kreate_reflect::Uid;

/// Whether a rigid body is moved by the simulation or only gets in the way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BodyKind {
    /// Falls, is pushed, tumbles.
    #[default]
    Active,
    /// Stays where it is: a floor, a wall, a table.
    Passive,
}

/// The shape a rigid body collides as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CollisionShape {
    /// The mesh's outline, made convex. What most things are close enough to.
    #[default]
    ConvexHull,
    /// The mesh's bounding box.
    Box,
    /// A ball round the mesh's bounding box.
    Sphere,
    /// The mesh's own triangles, holes and hollows included. Passive bodies
    /// only — an active one collides as its hull.
    Mesh,
}

/// An object as a rigid body: Blender's Rigid Body panel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidBody {
    pub kind: BodyKind,
    pub shape: CollisionShape,
    /// Kilograms.
    pub mass: f64,
    /// How much it resists sliding, 0 to 1.
    pub friction: f64,
    /// How much of a hit's speed comes back out, 0 to 1.
    pub bounciness: f64,
    /// The fraction of its speed it loses each second, moving and turning.
    pub linear_damping: f64,
    pub angular_damping: f64,
    /// How much gravity pulls it, as a fraction of the scene's: 1 falls as
    /// everything does, 0 floats — still pushed by what hits it, still in the
    /// way — and below 0 rises.
    pub gravity: f64,
}

impl Default for RigidBody {
    /// Blender's: a kilogram, half friction, no bounce.
    fn default() -> Self {
        Self {
            kind: BodyKind::Active,
            shape: CollisionShape::ConvexHull,
            mass: 1.0,
            friction: 0.5,
            bounciness: 0.0,
            linear_damping: 0.04,
            angular_damping: 0.1,
            gravity: 1.0,
        }
    }
}

/// An object cloth, soft bodies and particles collide with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Collision {
    /// How far from the surface something is kept, in metres.
    pub thickness: f64,
    /// 0 slides, 1 sticks.
    pub friction: f64,
    /// How much of a particle's speed it bounces off with.
    pub bounciness: f64,
}

impl Default for Collision {
    fn default() -> Self {
        Self {
            thickness: 0.02,
            friction: 0.3,
            bounciness: 0.3,
        }
    }
}

/// Which deformable body an object is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SoftKind {
    /// Thin and floppy: hangs, drapes, folds.
    #[default]
    Cloth,
    /// A solid that gives: wobbles back to its shape.
    Soft,
}

/// An object as cloth or a soft body.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftBody {
    pub kind: SoftKind,
    /// The whole object's mass, kilograms.
    pub mass: f64,
    /// How much its edges resist stretching, 0 to 1.
    pub stiffness: f64,
    /// How much it resists folding, 0 to 1.
    pub bending: f64,
    /// The fraction of its speed it loses each second.
    pub damping: f64,
    /// A soft body: how hard it springs back to its own shape, 0 to 1.
    pub shape_stiffness: f64,
    /// A soft body: how much it is blown up, -1 to 1; a closed mesh only.
    pub pressure: f64,
    /// The vertex group whose vertices are pinned — held where the object
    /// puts them, as a curtain on its rail. `None` pins nothing.
    pub pin_group: Option<String>,
    /// It collides with itself: a fold does not pass through the cloth
    /// under it.
    pub self_collision: bool,
    /// How far apart its own surfaces are kept, metres.
    pub self_distance: f64,
}

impl SoftBody {
    /// Cloth's defaults, or a soft body's.
    pub fn new(kind: SoftKind) -> Self {
        match kind {
            SoftKind::Cloth => Self {
                kind,
                mass: 1.0,
                stiffness: 1.0,
                bending: 0.05,
                damping: 0.1,
                shape_stiffness: 0.0,
                pressure: 0.0,
                pin_group: None,
                self_collision: false,
                self_distance: 0.015,
            },
            SoftKind::Soft => Self {
                kind,
                mass: 1.0,
                stiffness: 0.9,
                bending: 0.5,
                damping: 0.2,
                shape_stiffness: 0.3,
                pressure: 0.0,
                pin_group: None,
                self_collision: false,
                self_distance: 0.015,
            },
        }
    }
}

impl Default for SoftBody {
    fn default() -> Self {
        Self::new(SoftKind::Cloth)
    }
}

/// Whether a particle system throws particles or grows hair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParticleKind {
    #[default]
    Emitter,
    Hair,
}

/// One particle system on an emitter: Blender's Particles panel, cut to what
/// most uses need.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleSettings {
    pub name: String,
    pub kind: ParticleKind,
    /// How many particles, or hair strands.
    pub count: u32,
    /// Which random scatter: two systems with different seeds differ.
    pub seed: u32,
    /// Emitter: the frames particles are born between.
    pub frame_start: i32,
    pub frame_end: i32,
    /// Emitter: how many frames a particle lives.
    pub lifetime: f64,
    /// Emitter: speed out of the surface, metres a second.
    pub normal_speed: f64,
    /// Emitter: speed in a random direction, metres a second.
    pub random_speed: f64,
    /// How much gravity pulls it, as a fraction of the world's.
    pub gravity: f64,
    /// The fraction of its speed it loses each second to the air.
    pub drag: f64,
    /// A particle's size, or a hair's width, metres.
    pub size: f64,
    /// The object drawn at each particle, by identity; `None` draws a small
    /// diamond.
    pub instance: Option<Uid>,
    /// Hair: each strand's length, metres.
    pub hair_length: f64,
    /// Hair: how many pieces a strand bends at.
    pub hair_segments: u32,
    /// Hair: moved by gravity and collisions when baked, rather than standing
    /// straight.
    pub hair_dynamic: bool,
    /// Hair: how hard a strand holds its shape, 0 to 1.
    pub hair_stiffness: f64,
    /// Whether the emitter itself is still drawn.
    pub show_emitter: bool,
}

impl ParticleSettings {
    pub fn new(name: impl Into<String>, kind: ParticleKind) -> Self {
        Self {
            name: name.into(),
            kind,
            count: if kind == ParticleKind::Hair {
                500
            } else {
                1000
            },
            seed: 0,
            frame_start: 1,
            frame_end: 200,
            lifetime: 50.0,
            normal_speed: 1.0,
            random_speed: 0.2,
            gravity: 1.0,
            drag: 0.0,
            size: if kind == ParticleKind::Hair {
                0.005
            } else {
                0.03
            },
            instance: None,
            hair_length: 0.2,
            hair_segments: 5,
            hair_dynamic: false,
            hair_stiffness: 0.5,
            show_emitter: true,
        }
    }
}

/// What a liquid's flow object does: Blender's Flow Behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlowKind {
    /// Liquid where the mesh is, when the simulation starts.
    #[default]
    Geometry,
    /// Liquid keeps pouring out of where the mesh is: a tap, a hose.
    Inflow,
    /// Liquid that runs into where the mesh is drains away.
    Outflow,
}

/// The box a liquid is simulated in: Blender's fluid Domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidDomain {
    /// Cells along the box's longest side. More is finer and slower.
    pub resolution: u32,
    /// 0 to 1: how much of its own motion the liquid keeps from step to
    /// step. High is lively and splashy, low is calm and smooth.
    pub flip: f64,
}

impl Default for FluidDomain {
    fn default() -> Self {
        Self {
            resolution: 32,
            flip: 0.95,
        }
    }
}

/// Where liquid comes from or goes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FluidFlow {
    pub kind: FlowKind,
    /// The speed it starts with, metres a second, in the world.
    pub velocity: DVec3,
}

/// An object's part in a liquid simulation. Obstacles are objects with
/// [`Collision`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fluid {
    Domain(FluidDomain),
    Flow(FluidFlow),
}

/// Everything physical about one object. Empty for nearly all of them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjectPhysics {
    pub rigid: Option<RigidBody>,
    pub collision: Option<Collision>,
    pub soft: Option<SoftBody>,
    pub particles: Vec<ParticleSettings>,
    pub fluid: Option<Fluid>,
}

impl ObjectPhysics {
    pub fn is_empty(&self) -> bool {
        self.rigid.is_none()
            && self.collision.is_none()
            && self.soft.is_none()
            && self.particles.is_empty()
            && self.fluid.is_none()
    }
}

/// The scene's physics: gravity, how finely time is cut, and the frames a
/// bake covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicsWorld {
    /// Metres per second squared; Earth's, down.
    pub gravity: DVec3,
    /// Steps per frame. More is steadier and slower.
    pub substeps: u32,
    /// Solver passes per step. More is stiffer and slower.
    pub iterations: u32,
    pub frame_start: i32,
    pub frame_end: i32,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self {
            gravity: DVec3::new(0.0, 0.0, -9.81),
            substeps: 10,
            iterations: 10,
            frame_start: 1,
            frame_end: 250,
        }
    }
}

/// A rigid body's baked placements: one local transform per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct RigidTrack {
    pub object: Uid,
    pub frames: Vec<DTransform>,
}

/// A cloth or soft body's baked vertices, in the object's own space.
#[derive(Clone, Debug)]
pub struct SoftTrack {
    pub object: Uid,
    pub frames: Vec<Arc<[Vec3]>>,
}

/// One particle at one frame, in the emitter's own space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    /// Which way it is going, unit length; zero when still.
    pub heading: Vec3,
}

/// A particle system's baked frames: the particles alive at each, or for hair
/// every strand's points end to end.
#[derive(Clone, Debug)]
pub struct ParticleTrack {
    pub object: Uid,
    pub system: usize,
    pub frames: Vec<Arc<[Particle]>>,
}

/// A liquid domain's baked surface: a mesh a frame, in the domain object's
/// own space, which is what the domain shows in place of its box.
#[derive(Clone, Debug)]
pub struct FluidTrack {
    pub object: Uid,
    pub frames: Vec<Arc<kreate_mesh::MeshData>>,
}

/// The baked result: see the module notes.
#[derive(Clone, Debug, Default)]
pub struct PhysicsCache {
    /// The frame the first entry of every track is.
    pub start: i32,
    pub rigid: Vec<RigidTrack>,
    pub soft: Vec<SoftTrack>,
    pub particles: Vec<ParticleTrack>,
    pub fluid: Vec<FluidTrack>,
}

impl PhysicsCache {
    pub fn is_empty(&self) -> bool {
        self.rigid.is_empty()
            && self.soft.is_empty()
            && self.particles.is_empty()
            && self.fluid.is_empty()
    }

    /// The liquid surface a domain has at `frame`.
    pub fn fluid_at(&self, object: Uid, frame: f64) -> Option<&Arc<kreate_mesh::MeshData>> {
        let track = self.fluid.iter().find(|t| t.object == object)?;
        track.frames.get(self.index(frame, track.frames.len())?)
    }

    /// Which entry of a track of `len` frames plays at `frame`: held at the
    /// first before the range and at the last after it.
    pub fn index(&self, frame: f64, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        let i = (frame.round() as i64 - self.start as i64).clamp(0, len as i64 - 1);
        Some(i as usize)
    }

    pub fn rigid_of(&self, object: Uid) -> Option<&RigidTrack> {
        self.rigid.iter().find(|t| t.object == object)
    }

    pub fn soft_of(&self, object: Uid) -> Option<&SoftTrack> {
        self.soft.iter().find(|t| t.object == object)
    }

    /// The vertices a cloth or soft body has at `frame`.
    pub fn soft_at(&self, object: Uid, frame: f64) -> Option<&Arc<[Vec3]>> {
        let track = self.soft_of(object)?;
        track.frames.get(self.index(frame, track.frames.len())?)
    }

    /// The particles a system has at `frame`.
    pub fn particles_at(&self, object: Uid, system: usize, frame: f64) -> Option<&Arc<[Particle]>> {
        let track = self
            .particles
            .iter()
            .find(|t| t.object == object && t.system == system)?;
        track.frames.get(self.index(frame, track.frames.len())?)
    }
}

/// A small, fixed random sequence: the same scatter every run and on every
/// machine, which a bake has to be.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }

    /// splitmix64.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A direction, uniform over the sphere.
    pub fn direction(&mut self) -> DVec3 {
        let z = self.unit() * 2.0 - 1.0;
        let a = self.unit() * std::f64::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        DVec3::new(r * a.cos(), r * a.sin(), z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_track_holds_its_ends_outside_the_range() {
        let cache = PhysicsCache {
            start: 10,
            ..Default::default()
        };
        assert_eq!(cache.index(5.0, 4), Some(0));
        assert_eq!(cache.index(11.4, 4), Some(1));
        assert_eq!(cache.index(99.0, 4), Some(3));
        assert_eq!(cache.index(10.0, 0), None);
    }

    #[test]
    fn the_random_sequence_is_the_same_every_time() {
        let a: Vec<u64> = (0..4)
            .map({
                let mut r = Rng::new(7);
                move |_| r.next_u64()
            })
            .collect();
        let b: Vec<u64> = (0..4)
            .map({
                let mut r = Rng::new(7);
                move |_| r.next_u64()
            })
            .collect();
        assert_eq!(a, b);
        let mut r = Rng::new(1);
        for _ in 0..100 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
            assert!((r.direction().length() - 1.0).abs() < 1e-9);
        }
    }
}
