//! The bake: every simulation in the scene, run from the first frame of the
//! range to the last, into a [`PhysicsCache`].
//!
//! Run on a copy of the document, played frame by frame as the timeline
//! would — so an animated collider, a pinned corner on an animated object
//! and an emitter on the move all are where their keys put them — and
//! evaluated without the last bake, so a bake never reads its own output.

use std::sync::Arc;

use kreate_math::{DAffine3, DQuat, DTransform, DVec3};
use kreate_reflect::Uid;

use super::collide::Collider;
use super::particles::{HairSim, ParticleSim};
use super::rigid::{Body, RigidWorld};
use super::soft::SoftSim;
use super::{
    BodyKind, CollisionShape, ParticleKind, ParticleTrack, PhysicsCache, RigidTrack, SoftTrack,
};
use crate::document::{Document, ObjectId};
use crate::modifier::{EvalCtx, ModifierCache};

/// What a bake did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BakeSummary {
    pub frames: usize,
    pub rigid: usize,
    pub soft: usize,
    pub particles: usize,
}

/// A rigid body being baked, and how to turn where it is into its object's
/// transform.
struct Tracked {
    object: ObjectId,
    uid: Uid,
    body: usize,
    /// Where the body's frame was, and the object, when the bake began.
    centre: DVec3,
    turn: DQuat,
    world: DAffine3,
    frames: Vec<DTransform>,
}

/// Baked frames of vertices, and of particles.
type Frames<T> = Vec<Arc<[T]>>;
/// A cloth or soft body being baked.
type Baking = (ObjectId, Uid, SoftSim, Frames<kreate_math::Vec3>);
/// A particle system being baked: its object, which system, the sim.
type Emitting<S> = (ObjectId, Uid, usize, S, Frames<super::Particle>);

/// The object's transform that puts it at `world`, given its parent.
fn local_for(doc: &Document, id: ObjectId, world: DAffine3) -> DTransform {
    let Some(o) = doc.object(id) else {
        return DTransform::from_affine(world);
    };
    let above = match o.parent {
        Some(p) => doc.world_transform(p) * o.parent_inverse,
        None => DAffine3::IDENTITY,
    };
    DTransform::from_affine(above.inverse() * world)
}

/// Runs every simulation in `doc` over its physics frame range.
pub fn bake(doc: &Document) -> (PhysicsCache, BakeSummary) {
    let settings = doc.physics;
    let start = settings.frame_start;
    let end = settings.frame_end.max(start);
    let dt = 1.0 / doc.playback.fps.max(1.0);
    let substeps = settings.substeps.clamp(1, 100);
    let soft_iterations = (settings.iterations / 5).clamp(1, 20);

    let mut scene = doc.clone();
    scene.physics_cache = Arc::new(PhysicsCache::default());
    scene.animate(start as f64);
    let mut eval = ModifierCache::new();
    let ctx = EvalCtx::still();

    let ids: Vec<ObjectId> = scene
        .objects()
        .filter(|(id, o)| o.mesh().is_some() && !o.physics.is_empty() && scene.is_shown(*id))
        .map(|(id, _)| id)
        .collect();

    // What collides with cloth, soft bodies and particles.
    let mut colliders: Vec<(ObjectId, Collider)> = Vec::new();
    // Rigid bodies.
    let mut world = RigidWorld::new(settings.gravity, settings.iterations);
    let mut tracked: Vec<Tracked> = Vec::new();
    let mut softs: Vec<Baking> = Vec::new();
    let mut emitters: Vec<Emitting<ParticleSim>> = Vec::new();
    let mut hairs: Vec<Emitting<HairSim>> = Vec::new();

    for &id in &ids {
        let Some(o) = scene.object(id) else { continue };
        let Some(mesh) = eval.evaluated_base(&scene, id, &ctx) else {
            continue;
        };
        let w = scene.world_transform(id);
        let p = o.physics.clone();
        if let Some(c) = p.collision {
            colliders.push((id, Collider::new(&mesh, w, c.thickness, c.friction, c.bounciness)));
        }
        if let Some(r) = p.rigid {
            let (_, turn, _) = w.to_scale_rotation_translation();
            let points: Vec<DVec3> = mesh
                .positions()
                .iter()
                .map(|p| w.transform_point3(p.as_dvec3()))
                .collect();
            let mass = r.mass.max(1e-3);
            let box_of = || {
                let inv = turn.inverse();
                let (lo, hi) = points.iter().fold(
                    (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
                    |(lo, hi), p| (lo.min(inv * *p), hi.max(inv * *p)),
                );
                (turn * ((lo + hi) * 0.5), (hi - lo) * 0.5)
            };
            let mut body = match (r.shape, r.kind) {
                (CollisionShape::Mesh, BodyKind::Passive) => {
                    Body::mesh(Collider::new(&mesh, w, 0.0, r.friction, r.bounciness))
                }
                (CollisionShape::Box, _) => {
                    let (centre, half) = box_of();
                    Body::cuboid(centre, turn, half, mass)
                }
                (CollisionShape::Sphere, _) => {
                    let (centre, half) = box_of();
                    Body::ball(centre, half.max_element(), mass)
                }
                _ => Body::hull(&points, turn, mass),
            };
            if r.kind == BodyKind::Passive {
                body = body.fixed();
            }
            body.friction = r.friction;
            body.bounciness = r.bounciness;
            body.linear_damping = r.linear_damping;
            body.angular_damping = r.angular_damping;
            let (centre, rot) = (body.pos, body.rot);
            world.bodies.push(body);
            if r.kind == BodyKind::Active {
                tracked.push(Tracked {
                    object: id,
                    uid: o.uid,
                    body: world.bodies.len() - 1,
                    centre,
                    turn: rot,
                    world: w,
                    frames: vec![o.transform],
                });
            }
        }
        if let Some(s) = &p.soft {
            let sim = SoftSim::new(&mesh, w, s);
            let first: Arc<[kreate_math::Vec3]> = sim.local(w).into();
            softs.push((id, o.uid, sim, vec![first]));
        }
        for (i, s) in p.particles.iter().enumerate() {
            match s.kind {
                ParticleKind::Emitter => {
                    let mut sim = ParticleSim::new(&mesh, s);
                    // Those born on the first frame, not yet moved.
                    sim.advance(start as f64, 0.0, 1, settings.gravity, w, &[]);
                    let first: Arc<[super::Particle]> = sim.snapshot(w).into();
                    emitters.push((id, o.uid, i, sim, vec![first]));
                }
                ParticleKind::Hair if s.hair_dynamic => {
                    let sim = HairSim::new(&mesh, s, w);
                    let first: Arc<[super::Particle]> = sim.snapshot(w).into();
                    hairs.push((id, o.uid, i, sim, vec![first]));
                }
                ParticleKind::Hair => {}
            }
        }
    }

    for frame in start + 1..=end {
        scene.animate(frame as f64);
        for (id, c) in colliders.iter_mut() {
            c.place(scene.world_transform(*id));
        }
        let near: Vec<&Collider> = colliders.iter().map(|(_, c)| c).collect();
        if !tracked.is_empty() {
            world.advance(dt, substeps);
            for t in tracked.iter_mut() {
                let b = &world.bodies[t.body];
                let moved = DAffine3::from_rotation_translation(b.rot * t.turn.inverse(), b.pos)
                    * DAffine3::from_translation(-t.centre)
                    * t.world;
                t.frames.push(local_for(&scene, t.object, moved));
            }
        }
        for (id, _, sim, frames) in softs.iter_mut() {
            // Each collider but itself: a cloth that is also a collider does
            // not collide with its own rest shape.
            let others: Vec<&Collider> = colliders
                .iter()
                .filter(|(c, _)| c != id)
                .map(|(_, c)| c)
                .collect();
            let w = scene.world_transform(*id);
            sim.advance(dt, substeps, soft_iterations, settings.gravity, w, &others);
            frames.push(sim.local(w).into());
        }
        for (id, _, _, sim, frames) in emitters.iter_mut() {
            let w = scene.world_transform(*id);
            sim.advance(frame as f64, dt, substeps.min(10), settings.gravity, w, &near);
            frames.push(sim.snapshot(w).into());
        }
        for (id, _, _, sim, frames) in hairs.iter_mut() {
            let w = scene.world_transform(*id);
            sim.advance(dt, substeps, settings.gravity, w, &near);
            frames.push(sim.snapshot(w).into());
        }
    }

    let summary = BakeSummary {
        frames: (end - start + 1) as usize,
        rigid: tracked.len(),
        soft: softs.len(),
        particles: emitters.len() + hairs.len(),
    };
    let cache = PhysicsCache {
        start,
        rigid: tracked
            .into_iter()
            .map(|t| RigidTrack {
                object: t.uid,
                frames: t.frames,
            })
            .collect(),
        soft: softs
            .into_iter()
            .map(|(_, uid, _, frames)| SoftTrack { object: uid, frames })
            .collect(),
        particles: emitters
            .into_iter()
            .map(|(_, uid, system, _, frames)| (uid, system, frames))
            .chain(hairs.into_iter().map(|(_, uid, system, _, frames)| (uid, system, frames)))
            .map(|(object, system, frames)| ParticleTrack {
                object,
                system,
                frames,
            })
            .collect(),
    };
    (cache, summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Object, ObjectData};
    use crate::physics::{Collision, RigidBody, SoftBody, SoftKind};

    fn add(doc: &mut Document, name: &str, mesh: kreate_mesh::MeshData, at: DVec3) -> ObjectId {
        let m = doc.add_mesh(mesh);
        let mut o = Object::new(name, ObjectData::Mesh(m));
        o.transform.translation = at;
        doc.add_object(o)
    }

    fn short(doc: &mut Document, frames: i32) {
        doc.physics.frame_start = 1;
        doc.physics.frame_end = frames;
    }

    #[test]
    fn a_baked_cube_falls_onto_the_floor_and_plays_back() {
        let mut doc = Document::new();
        short(&mut doc, 60);
        let floor = add(&mut doc, "Floor", kreate_mesh::primitives::plane(10.0), DVec3::ZERO);
        let cube = add(&mut doc, "Cube", kreate_mesh::primitives::cube(1.0), DVec3::new(0.0, 0.0, 3.0));
        doc.object_mut(floor).unwrap().physics.rigid = Some(RigidBody {
            kind: BodyKind::Passive,
            shape: CollisionShape::Mesh,
            ..RigidBody::default()
        });
        doc.object_mut(cube).unwrap().physics.rigid = Some(RigidBody::default());
        let (cache, summary) = bake(&doc);
        assert_eq!(summary.rigid, 1, "the floor does not move");
        assert_eq!(cache.rigid[0].frames.len(), 60);
        assert_eq!(cache.rigid[0].frames[0].translation.z, 3.0, "starts where it was");
        let last = cache.rigid[0].frames[59].translation.z;
        assert!((last - 0.5).abs() < 0.05, "rests on the floor: {last}");

        doc.physics_cache = Arc::new(cache);
        doc.animate(60.0);
        assert!((doc.object(cube).unwrap().transform.translation.z - 0.5).abs() < 0.05);
        doc.animate(1.0);
        assert_eq!(doc.object(cube).unwrap().transform.translation.z, 3.0, "and back");
    }

    #[test]
    fn a_child_body_is_written_back_through_its_parent() {
        let mut doc = Document::new();
        short(&mut doc, 48);
        let floor = add(&mut doc, "Floor", kreate_mesh::primitives::plane(10.0), DVec3::ZERO);
        doc.object_mut(floor).unwrap().physics.rigid = Some(RigidBody {
            kind: BodyKind::Passive,
            shape: CollisionShape::Box,
            ..RigidBody::default()
        });
        let holder = add(&mut doc, "Holder", kreate_mesh::primitives::cube(0.1), DVec3::new(5.0, 0.0, 0.0));
        let cube = add(&mut doc, "Cube", kreate_mesh::primitives::cube(1.0), DVec3::new(5.0, 0.0, 2.0));
        doc.set_parent(cube, Some(holder)).unwrap();
        doc.object_mut(cube).unwrap().physics.rigid = Some(RigidBody {
            shape: CollisionShape::Box,
            ..RigidBody::default()
        });
        let (cache, _) = bake(&doc);
        doc.physics_cache = Arc::new(cache);
        doc.animate(48.0);
        let at = doc.world_transform(cube).translation;
        assert!((at.z - 0.5).abs() < 0.05 && (at.x - 5.0).abs() < 0.05, "{at:?}");
    }

    #[test]
    fn baked_cloth_drapes_and_is_laid_onto_the_evaluated_mesh() {
        let mut doc = Document::new();
        short(&mut doc, 36);
        let block = add(&mut doc, "Block", kreate_mesh::primitives::cube(1.0), DVec3::ZERO);
        doc.object_mut(block).unwrap().physics.collision = Some(Collision::default());
        let cloth = add(&mut doc, "Cloth", kreate_mesh::primitives::grid(2.0, 12, 12), DVec3::new(0.0, 0.0, 1.2));
        doc.object_mut(cloth).unwrap().physics.soft = Some(SoftBody::new(SoftKind::Cloth));
        let (cache, summary) = bake(&doc);
        assert_eq!(summary.soft, 1);
        doc.physics_cache = Arc::new(cache);
        doc.animate(36.0);
        let mut eval = ModifierCache::new();
        let mesh = eval.evaluated(&doc, cloth, &EvalCtx::still()).unwrap();
        // In the cloth's own space, lifted 1.2: the lowest corner hangs below
        // the lid at 0.5, so below -0.7 here.
        let lowest = mesh.positions().iter().map(|p| p.z).fold(f32::MAX, f32::min);
        assert!(lowest < -0.8, "draped: {lowest}");
        let middle = mesh
            .positions()
            .iter()
            .filter(|p| p.x.abs() < 0.2 && p.y.abs() < 0.2)
            .map(|p| p.z)
            .fold(f32::MAX, f32::min);
        assert!(middle > -0.75, "held by the lid: {middle}");
        // The same frame again is the same mesh: nothing is redone.
        let again = eval.evaluated(&doc, cloth, &EvalCtx::still()).unwrap();
        assert!(Arc::ptr_eq(&mesh, &again));
    }

    #[test]
    fn particles_and_hair_appear_on_their_emitter() {
        let mut doc = Document::new();
        short(&mut doc, 10);
        let e = add(&mut doc, "Emitter", kreate_mesh::primitives::plane(1.0), DVec3::ZERO);
        let mut spray = crate::physics::ParticleSettings::new("Spray", ParticleKind::Emitter);
        (spray.count, spray.frame_start, spray.frame_end) = (50, 1, 5);
        let hair = crate::physics::ParticleSettings::new("Fur", ParticleKind::Hair);
        doc.object_mut(e).unwrap().physics.particles = vec![spray, hair.clone()];
        let mut eval = ModifierCache::new();
        let before = eval.evaluated(&doc, e, &EvalCtx::still()).unwrap();
        let plane_faces = 1;
        assert_eq!(
            before.num_faces(),
            plane_faces + hair.count * hair.hair_segments,
            "hair at rest before any bake, no particles"
        );
        let (cache, _) = bake(&doc);
        doc.physics_cache = Arc::new(cache);
        doc.animate(10.0);
        let after = eval.evaluated(&doc, e, &EvalCtx::still()).unwrap();
        assert_eq!(after.num_faces(), before.num_faces() + 50 * 8, "fifty diamonds");
    }
}
