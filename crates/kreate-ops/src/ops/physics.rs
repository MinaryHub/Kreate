//! Physics: what the selected objects are to the simulation, the scene's
//! world, and the bake. See `kreate_scene::physics`.
//!
//! Every setting goes through an operator, as everything does, so a change
//! is an undo step and a script line; the bake is one too, so a bad bake is
//! undone like anything else.

use std::sync::Arc;

use kreate_math::DVec3;
use kreate_scene::animation::{self, Interpolation};
use kreate_scene::physics::{
    BodyKind, Collision, CollisionShape, ObjectPhysics, ParticleKind, ParticleSettings,
    PhysicsCache, PhysicsWorld, RigidBody, SoftBody, SoftKind,
};
use kreate_scene::{DocMut, Document, ObjectId};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<RigidBodySet>();
    registry.register::<CollisionSet>();
    registry.register::<SoftSet>();
    registry.register::<ParticlesAdd>();
    registry.register::<ParticlesRemove>();
    registry.register::<ParticlesSet>();
    registry.register::<WorldSet>();
    registry.register::<Bake>();
    registry.register::<FreeBake>();
    registry.register::<BakeToKeys>();
}

/// The selected objects with a mesh: what physics settings go on.
fn selected_meshes(doc: &Document) -> Vec<ObjectId> {
    doc.selection
        .objects
        .iter()
        .copied()
        .filter(|id| doc.object(*id).is_some_and(|o| o.mesh().is_some()))
        .collect()
}

fn active_physics(doc: &Document) -> Option<&ObjectPhysics> {
    let o = doc.object(doc.selection.active?)?;
    o.mesh().map(|_| &o.physics)
}

/// Sets something on every selected mesh — or on the active object alone —
/// or cancels when nothing changes.
fn on_selected(
    doc: &mut DocMut<'_>,
    active_only: bool,
    mut set: impl FnMut(&mut ObjectPhysics),
) -> OpResult {
    let ids = if active_only {
        let d = doc.document();
        d.selection
            .active
            .filter(|id| d.object(*id).is_some_and(|o| o.mesh().is_some()))
            .into_iter()
            .collect()
    } else {
        selected_meshes(doc.document())
    };
    let changing: Vec<ObjectId> = ids
        .into_iter()
        .filter(|id| {
            doc.document().object(*id).is_some_and(|o| {
                let mut p = o.physics.clone();
                set(&mut p);
                p != o.physics
            })
        })
        .collect();
    if changing.is_empty() {
        return OpResult::Cancelled;
    }
    let d = doc.document_mut();
    for id in changing {
        if let Some(o) = d.object_mut(id) {
            set(&mut o.physics);
        }
    }
    OpResult::Finished
}

// --- rigid body ------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct RigidBodyParams {
    /// Whether it is a rigid body at all.
    pub enabled: bool,
    /// Active falls and tumbles; passive stays put and is in the way.
    #[param(variants("active", "passive"))]
    pub kind: u32,
    /// The shape it collides as.
    #[param(variants("convex_hull", "box", "sphere", "mesh"))]
    pub shape: u32,
    /// Kilograms.
    #[param(min = 0.001, soft_max = 100.0)]
    pub mass: f64,
    /// How much it resists sliding.
    #[param(min = 0.0, max = 1.0)]
    pub friction: f64,
    /// How much of a hit's speed comes back.
    #[param(min = 0.0, max = 1.0)]
    pub bounciness: f64,
    /// The fraction of its speed lost each second.
    #[param(min = 0.0, max = 1.0)]
    pub linear_damping: f64,
    /// The fraction of its turning lost each second.
    #[param(min = 0.0, max = 1.0)]
    pub angular_damping: f64,
    /// The active object only, rather than every selected mesh: what the
    /// Physics tab sets, because the tab shows the active object's settings —
    /// applying them to the whole selection turned off the physics of every
    /// other selected object that the active one did not have.
    #[param(hidden)]
    pub active_only: bool,
}

impl Default for RigidBodyParams {
    fn default() -> Self {
        rigid_params(None)
    }
}

/// The parameters that leave a rigid body as it is; `None` is "not one",
/// with the defaults it would become.
pub fn rigid_params(body: Option<&RigidBody>) -> RigidBodyParams {
    let b = body.copied().unwrap_or_default();
    RigidBodyParams {
        enabled: body.is_some(),
        kind: match b.kind {
            BodyKind::Active => 0,
            BodyKind::Passive => 1,
        },
        shape: match b.shape {
            CollisionShape::ConvexHull => 0,
            CollisionShape::Box => 1,
            CollisionShape::Sphere => 2,
            CollisionShape::Mesh => 3,
        },
        mass: b.mass,
        friction: b.friction,
        bounciness: b.bounciness,
        linear_damping: b.linear_damping,
        angular_damping: b.angular_damping,
        active_only: false,
    }
}

/// Makes the selected meshes rigid bodies — or not — with these settings.
pub struct RigidBodySet;

impl Operator for RigidBodySet {
    type Params = RigidBodyParams;
    fn id() -> &'static str {
        "physics.rigid_body_set"
    }
    fn label() -> &'static str {
        "Rigid Body"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !selected_meshes(doc).is_empty()
    }
    fn current(
        _ctx: &Context,
        doc: &Document,
        _given: &crate::PropertyMap,
    ) -> Option<Self::Params> {
        Some(rigid_params(active_physics(doc)?.rigid.as_ref()))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let body = params.enabled.then(|| RigidBody {
            kind: if params.kind == 1 {
                BodyKind::Passive
            } else {
                BodyKind::Active
            },
            shape: match params.shape {
                1 => CollisionShape::Box,
                2 => CollisionShape::Sphere,
                3 => CollisionShape::Mesh,
                _ => CollisionShape::ConvexHull,
            },
            mass: params.mass.max(0.001),
            friction: params.friction.clamp(0.0, 1.0),
            bounciness: params.bounciness.clamp(0.0, 1.0),
            linear_damping: params.linear_damping.clamp(0.0, 1.0),
            angular_damping: params.angular_damping.clamp(0.0, 1.0),
        });
        Ok(on_selected(doc, params.active_only, |p| p.rigid = body))
    }
}

// --- collision ---------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct CollisionParams {
    /// Whether cloth, soft bodies and particles collide with it.
    pub enabled: bool,
    /// How far from the surface they are kept, metres.
    #[param(min = 0.0, soft_max = 0.2, unit = "length")]
    pub thickness: f64,
    /// 0 slides, 1 sticks.
    #[param(min = 0.0, max = 1.0)]
    pub friction: f64,
    /// How much of a particle's speed bounces off.
    #[param(min = 0.0, max = 1.0)]
    pub bounciness: f64,
    /// The active object only, rather than every selected mesh: what the
    /// Physics tab sets, because the tab shows the active object's settings —
    /// applying them to the whole selection turned off the physics of every
    /// other selected object that the active one did not have.
    #[param(hidden)]
    pub active_only: bool,
}

impl Default for CollisionParams {
    fn default() -> Self {
        collision_params(None)
    }
}

pub fn collision_params(c: Option<&Collision>) -> CollisionParams {
    let d = c.copied().unwrap_or_default();
    CollisionParams {
        enabled: c.is_some(),
        thickness: d.thickness,
        friction: d.friction,
        bounciness: d.bounciness,
        active_only: false,
    }
}

/// Makes the selected meshes something cloth and particles collide with.
pub struct CollisionSet;

impl Operator for CollisionSet {
    type Params = CollisionParams;
    fn id() -> &'static str {
        "physics.collision_set"
    }
    fn label() -> &'static str {
        "Collision"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !selected_meshes(doc).is_empty()
    }
    fn current(
        _ctx: &Context,
        doc: &Document,
        _given: &crate::PropertyMap,
    ) -> Option<Self::Params> {
        Some(collision_params(active_physics(doc)?.collision.as_ref()))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let c = params.enabled.then(|| Collision {
            thickness: params.thickness.max(0.0),
            friction: params.friction.clamp(0.0, 1.0),
            bounciness: params.bounciness.clamp(0.0, 1.0),
        });
        Ok(on_selected(doc, params.active_only, |p| p.collision = c))
    }
}

// --- cloth and soft body -----------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SoftParams {
    /// None, cloth, or a soft body.
    #[param(variants("none", "cloth", "soft_body"))]
    pub kind: u32,
    /// The whole object's mass, kilograms.
    #[param(min = 0.001, soft_max = 50.0)]
    pub mass: f64,
    /// How much its edges resist stretching.
    #[param(min = 0.0, max = 1.0)]
    pub stiffness: f64,
    /// How much it resists folding.
    #[param(min = 0.0, max = 1.0)]
    pub bending: f64,
    /// The fraction of its speed lost each second.
    #[param(min = 0.0, max = 1.0)]
    pub damping: f64,
    /// A soft body: how hard it springs back to its shape.
    #[param(min = 0.0, max = 1.0)]
    pub shape_stiffness: f64,
    /// A soft body: how much it is blown up; a closed mesh only.
    #[param(min = -1.0, max = 1.0)]
    pub pressure: f64,
    /// The vertex group whose vertices stay where the object puts them.
    /// Empty pins nothing.
    pub pin_group: String,
    /// The active object only, rather than every selected mesh: what the
    /// Physics tab sets, because the tab shows the active object's settings —
    /// applying them to the whole selection turned off the physics of every
    /// other selected object that the active one did not have.
    #[param(hidden)]
    pub active_only: bool,
}

impl Default for SoftParams {
    fn default() -> Self {
        soft_params(None)
    }
}

pub fn soft_params(s: Option<&SoftBody>) -> SoftParams {
    let d = s.cloned().unwrap_or_default();
    SoftParams {
        kind: match s.map(|s| s.kind) {
            None => 0,
            Some(SoftKind::Cloth) => 1,
            Some(SoftKind::Soft) => 2,
        },
        mass: d.mass,
        stiffness: d.stiffness,
        bending: d.bending,
        damping: d.damping,
        shape_stiffness: d.shape_stiffness,
        pressure: d.pressure,
        pin_group: d.pin_group.unwrap_or_default(),
        active_only: false,
    }
}

/// Which of the soft parameters a kind has: the panel shows only these.
pub fn soft_param_applies(kind: u32, name: &str) -> bool {
    match name {
        "kind" => true,
        "shape_stiffness" | "pressure" => kind == 2,
        _ => kind != 0,
    }
}

/// Makes the selected meshes cloth, soft bodies, or neither.
pub struct SoftSet;

impl Operator for SoftSet {
    type Params = SoftParams;
    fn id() -> &'static str {
        "physics.soft_set"
    }
    fn label() -> &'static str {
        "Cloth / Soft Body"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !selected_meshes(doc).is_empty()
    }
    fn current(
        _ctx: &Context,
        doc: &Document,
        _given: &crate::PropertyMap,
    ) -> Option<Self::Params> {
        Some(soft_params(active_physics(doc)?.soft.as_ref()))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let kind = match params.kind {
            1 => Some(SoftKind::Cloth),
            2 => Some(SoftKind::Soft),
            _ => None,
        };
        let pin = Some(params.pin_group.trim().to_string()).filter(|g| !g.is_empty());
        let given = SoftBody {
            kind: kind.unwrap_or_default(),
            mass: params.mass.max(0.001),
            stiffness: params.stiffness.clamp(0.0, 1.0),
            bending: params.bending.clamp(0.0, 1.0),
            damping: params.damping.clamp(0.0, 1.0),
            shape_stiffness: params.shape_stiffness.clamp(0.0, 1.0),
            pressure: params.pressure.clamp(-1.0, 1.0),
            pin_group: pin.clone(),
        };
        Ok(on_selected(doc, params.active_only, |p| {
            p.soft = match kind {
                None => None,
                // Becoming another kind starts from that kind's own settings:
                // a cloth's floppiness carried into a soft body is a puddle.
                Some(k)
                    if p.soft.as_ref().is_some_and(|s| s.kind != k)
                        || p.soft.is_none() && given.kind != SoftKind::Cloth =>
                {
                    Some(SoftBody {
                        pin_group: pin.clone(),
                        ..SoftBody::new(k)
                    })
                }
                Some(_) => Some(given.clone()),
            }
        }))
    }
}

// --- particles ---------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ParticlesAddParams {
    /// Particles thrown from the surface, or hair grown on it.
    #[param(variants("emitter", "hair"))]
    pub kind: u32,
}

/// Adds a particle system to the active object.
pub struct ParticlesAdd;

impl Operator for ParticlesAdd {
    type Params = ParticlesAddParams;
    fn id() -> &'static str {
        "physics.particles_add"
    }
    fn label() -> &'static str {
        "Add Particle System"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_physics(doc).is_some()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = doc.document().selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let kind = if params.kind == 1 {
            ParticleKind::Hair
        } else {
            ParticleKind::Emitter
        };
        let d = doc.document_mut();
        let Some(o) = d.object_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        let base = if kind == ParticleKind::Hair {
            "Hair"
        } else {
            "Particles"
        };
        let mut name = base.to_string();
        let mut n = 1;
        while o.physics.particles.iter().any(|s| s.name == name) {
            name = format!("{base}.{n:03}");
            n += 1;
        }
        o.physics.particles.push(ParticleSettings::new(name, kind));
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct ParticlesRemoveParams {
    pub index: u32,
}

/// Removes one of the active object's particle systems.
pub struct ParticlesRemove;

impl Operator for ParticlesRemove {
    type Params = ParticlesRemoveParams;
    fn id() -> &'static str {
        "physics.particles_remove"
    }
    fn label() -> &'static str {
        "Remove Particle System"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_physics(doc).is_some_and(|p| !p.particles.is_empty())
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = doc.document().selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let i = params.index as usize;
        if doc
            .document()
            .object(id)
            .is_none_or(|o| i >= o.physics.particles.len())
        {
            return Ok(OpResult::Cancelled);
        }
        if let Some(o) = doc.document_mut().object_mut(id) {
            o.physics.particles.remove(i);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct ParticlesSetParams {
    /// Which system, counting from zero.
    #[param(hidden)]
    pub index: u32,
    pub name: String,
    #[param(variants("emitter", "hair"))]
    pub kind: u32,
    /// How many particles, or hair strands.
    #[param(min = 0, soft_max = 10000)]
    pub count: u32,
    /// A different scatter for each number.
    pub seed: u32,
    /// The first frame particles are born on.
    pub frame_start: i32,
    /// The last frame particles are born on.
    pub frame_end: i32,
    /// How many frames a particle lives.
    #[param(min = 0.0, soft_max = 500.0)]
    pub lifetime: f64,
    /// Speed out of the surface, metres a second.
    #[param(soft_min = -10.0, soft_max = 10.0)]
    pub normal_speed: f64,
    /// Speed in a random direction, metres a second.
    #[param(min = 0.0, soft_max = 10.0)]
    pub random_speed: f64,
    /// How much gravity pulls, as a fraction of the world's.
    #[param(soft_min = -1.0, soft_max = 2.0)]
    pub gravity: f64,
    /// The fraction of speed lost to the air each second.
    #[param(min = 0.0, max = 1.0)]
    pub drag: f64,
    /// A particle's size, or a hair's width, metres.
    #[param(min = 0.0, soft_max = 1.0, unit = "length")]
    pub size: f64,
    /// The object drawn at each particle, by name; empty draws a small
    /// diamond.
    pub instance: String,
    /// Each strand's length, metres.
    #[param(min = 0.0, soft_max = 2.0, unit = "length")]
    pub hair_length: f64,
    /// How many pieces a strand bends at.
    #[param(min = 1, max = 64)]
    pub hair_segments: u32,
    /// Moved by gravity and collisions when baked.
    pub hair_dynamic: bool,
    /// How hard a strand holds its shape.
    #[param(min = 0.0, max = 1.0)]
    pub hair_stiffness: f64,
    /// Whether the emitter itself is still drawn.
    pub show_emitter: bool,
}

impl Default for ParticlesSetParams {
    fn default() -> Self {
        let d = ParticleSettings::new("Particles", ParticleKind::Emitter);
        particle_params(&Document::new(), 0, &d)
    }
}

/// The parameters that leave system `index` as it is.
pub fn particle_params(doc: &Document, index: u32, s: &ParticleSettings) -> ParticlesSetParams {
    ParticlesSetParams {
        index,
        name: s.name.clone(),
        kind: if s.kind == ParticleKind::Hair { 1 } else { 0 },
        count: s.count,
        seed: s.seed,
        frame_start: s.frame_start,
        frame_end: s.frame_end,
        lifetime: s.lifetime,
        normal_speed: s.normal_speed,
        random_speed: s.random_speed,
        gravity: s.gravity,
        drag: s.drag,
        size: s.size,
        instance: s
            .instance
            .and_then(|u| doc.object_by_uid(u))
            .and_then(|id| doc.object(id))
            .map(|o| o.name.clone())
            .unwrap_or_default(),
        hair_length: s.hair_length,
        hair_segments: s.hair_segments,
        hair_dynamic: s.hair_dynamic,
        hair_stiffness: s.hair_stiffness,
        show_emitter: s.show_emitter,
    }
}

/// Which particle parameters a kind has.
pub fn particle_param_applies(kind: u32, name: &str) -> bool {
    let hair = kind == 1;
    match name {
        "frame_start" | "frame_end" | "lifetime" | "normal_speed" | "random_speed" | "instance" => {
            !hair
        }
        "hair_length" | "hair_segments" | "hair_dynamic" | "hair_stiffness" => hair,
        _ => true,
    }
}

/// Sets one of the active object's particle systems.
pub struct ParticlesSet;

impl Operator for ParticlesSet {
    type Params = ParticlesSetParams;
    fn id() -> &'static str {
        "physics.particles_set"
    }
    fn label() -> &'static str {
        "Particle Settings"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_physics(doc).is_some_and(|p| !p.particles.is_empty())
    }
    fn current(_ctx: &Context, doc: &Document, given: &crate::PropertyMap) -> Option<Self::Params> {
        let index = match given.get("index") {
            Some(kreate_reflect::PropertyValue::Int(i)) => *i as u32,
            _ => 0,
        };
        let s = active_physics(doc)?.particles.get(index as usize)?;
        Some(particle_params(doc, index, s))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = doc.document().selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let i = params.index as usize;
        let instance = if params.instance.trim().is_empty() {
            None
        } else {
            let d = doc.document();
            let found = d
                .objects()
                .find(|(_, o)| o.name == params.instance.trim())
                .map(|(_, o)| o.uid);
            match found {
                Some(uid) => Some(uid),
                None => {
                    return Err(OpError::Failed(format!(
                        "no object is called \"{}\"",
                        params.instance.trim()
                    )))
                }
            }
        };
        let Some(before) = doc
            .document()
            .object(id)
            .and_then(|o| o.physics.particles.get(i).cloned())
        else {
            return Ok(OpResult::Cancelled);
        };
        let kind = if params.kind == 1 {
            ParticleKind::Hair
        } else {
            ParticleKind::Emitter
        };
        let after = ParticleSettings {
            name: Some(params.name.trim().to_string())
                .filter(|n| !n.is_empty())
                .unwrap_or(before.name.clone()),
            kind,
            count: params.count.min(1_000_000),
            seed: params.seed,
            frame_start: params.frame_start,
            frame_end: params.frame_end.max(params.frame_start),
            lifetime: params.lifetime.max(0.0),
            normal_speed: params.normal_speed,
            random_speed: params.random_speed.max(0.0),
            gravity: params.gravity,
            drag: params.drag.clamp(0.0, 1.0),
            size: params.size.max(0.0),
            instance,
            hair_length: params.hair_length.max(0.0),
            hair_segments: params.hair_segments.clamp(1, 64),
            hair_dynamic: params.hair_dynamic,
            hair_stiffness: params.hair_stiffness.clamp(0.0, 1.0),
            show_emitter: params.show_emitter,
        };
        if after == before {
            return Ok(OpResult::Cancelled);
        }
        if let Some(o) = doc.document_mut().object_mut(id) {
            o.physics.particles[i] = after;
        }
        Ok(OpResult::Finished)
    }
}

// --- the world and the bake ---------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct WorldParams {
    /// Metres per second squared.
    #[param(unit = "length")]
    pub gravity: [f64; 3],
    /// Steps per frame: more is steadier and slower.
    #[param(min = 1, max = 100)]
    pub substeps: u32,
    /// Solver passes per step: more is stiffer and slower.
    #[param(min = 1, max = 100)]
    pub iterations: u32,
    /// The first frame a bake covers.
    pub frame_start: i32,
    /// The last frame a bake covers.
    pub frame_end: i32,
}

impl Default for WorldParams {
    fn default() -> Self {
        world_params(&PhysicsWorld::default())
    }
}

pub fn world_params(w: &PhysicsWorld) -> WorldParams {
    WorldParams {
        gravity: w.gravity.to_array(),
        substeps: w.substeps,
        iterations: w.iterations,
        frame_start: w.frame_start,
        frame_end: w.frame_end,
    }
}

/// The scene's gravity, steps and bake range.
pub struct WorldSet;

impl Operator for WorldSet {
    type Params = WorldParams;
    fn id() -> &'static str {
        "physics.world_set"
    }
    fn label() -> &'static str {
        "Physics World"
    }
    fn current(
        _ctx: &Context,
        doc: &Document,
        _given: &crate::PropertyMap,
    ) -> Option<Self::Params> {
        Some(world_params(&doc.physics))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let w = PhysicsWorld {
            gravity: DVec3::from_array(params.gravity),
            substeps: params.substeps.clamp(1, 100),
            iterations: params.iterations.clamp(1, 100),
            frame_start: params.frame_start,
            frame_end: params.frame_end.max(params.frame_start),
        };
        if w == doc.document().physics {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut().physics = w;
        Ok(OpResult::Finished)
    }
}

/// Whether anything in the scene takes part in a simulation.
fn anything_physical(doc: &Document) -> bool {
    doc.objects().any(|(_, o)| !o.physics.is_empty())
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct BakeParams {}

/// Runs every simulation over the physics frame range and keeps the result.
pub struct Bake;

impl Operator for Bake {
    type Params = BakeParams;
    fn id() -> &'static str {
        "physics.bake"
    }
    fn label() -> &'static str {
        "Bake Physics"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        anything_physical(doc)
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let (cache, summary) = kreate_scene::physics::bake::bake(doc.document());
        if summary.rigid + summary.soft + summary.particles == 0 {
            return Err(OpError::Failed(
                "nothing to bake: no active rigid body, cloth, soft body or emitter".into(),
            ));
        }
        let d = doc.document_mut();
        restore_rigid(d);
        d.physics_cache = Arc::new(cache);
        let frame = d.playback.frame;
        d.animate(frame);
        Ok(OpResult::Finished)
    }
}

/// Puts every baked rigid body back where the bake found it.
fn restore_rigid(d: &mut Document) {
    let cache = d.physics_cache.clone();
    for track in &cache.rigid {
        if let (Some(id), Some(first)) = (d.object_by_uid(track.object), track.frames.first()) {
            if let Some(o) = d.object_mut(id) {
                o.transform = *first;
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct FreeBakeParams {}

/// Throws the bake away; rigid bodies go back to where they started.
pub struct FreeBake;

impl Operator for FreeBake {
    type Params = FreeBakeParams;
    fn id() -> &'static str {
        "physics.free_bake"
    }
    fn label() -> &'static str {
        "Free Bake"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.physics_cache.is_empty()
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if doc.document().physics_cache.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        restore_rigid(d);
        d.physics_cache = Arc::new(PhysicsCache::default());
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct BakeToKeysParams {}

/// Turns the baked rigid bodies into keyframes — a key every frame — and
/// stops them being rigid bodies, as Blender's Bake To Keyframes does: what
/// reads curves (glTF, the graph editor) then has the motion.
pub struct BakeToKeys;

impl Operator for BakeToKeys {
    type Params = BakeToKeysParams;
    fn id() -> &'static str {
        "physics.bake_to_keys"
    }
    fn label() -> &'static str {
        "Bake to Keyframes"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.physics_cache.rigid.is_empty()
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let cache = doc.document().physics_cache.clone();
        if cache.rigid.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for track in &cache.rigid {
            let Some(id) = d.object_by_uid(track.object) else {
                continue;
            };
            let Some(object) = d.object(id) else { continue };
            let action = match object.action.filter(|a| d.action(*a).is_some()) {
                Some(a) => a,
                None => {
                    let a = d.add_action(animation::Action::new(format!("{}Action", object.name)));
                    if let Some(o) = d.object_mut(id) {
                        o.action = Some(a);
                    }
                    a
                }
            };
            let Some(action) = d.action_mut(action) else {
                continue;
            };
            // Each angle kept within half a turn of the frame before, so a
            // tumbling body does not spin back round between two keys.
            let mut last: Option<DVec3> = None;
            for (k, t) in track.frames.iter().enumerate() {
                let frame = (cache.start + k as i32) as f64;
                let mut euler = animation::read_object(t, "rotation_euler").unwrap_or_default();
                if let Some(prev) = last {
                    for axis in 0..3 {
                        let tau = std::f64::consts::TAU;
                        euler[axis] += ((prev[axis] - euler[axis]) / tau).round() * tau;
                    }
                }
                last = Some(euler);
                for axis in 0..3u8 {
                    action.curve_mut("location", axis).insert(
                        frame,
                        t.translation[axis as usize],
                        Interpolation::Linear,
                    );
                    action.curve_mut("rotation_euler", axis).insert(
                        frame,
                        euler[axis as usize],
                        Interpolation::Linear,
                    );
                }
            }
            if let Some(o) = d.object_mut(id) {
                o.physics.rigid = None;
            }
        }
        let mut rest = (*cache).clone();
        rest.rigid.clear();
        d.physics_cache = Arc::new(rest);
        let frame = d.playback.frame;
        d.animate(frame);
        Ok(OpResult::Finished)
    }
}
