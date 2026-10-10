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
use super::fluid::{world_triangles, LiquidSim};
use super::gas::{Emission, GasSettings, GasSim};
use super::particles::{HairSim, ParticleSim};
use super::rigid::{Body, RigidWorld};
use super::soft::SoftSim;
use super::{
    BodyKind, CollisionShape, FluidTrack, GasFrame, GasTrack, ParticleKind, ParticleTrack,
    PhysicsCache, RigidTrack, SoftTrack, Substance,
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
    pub fluid: usize,
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
    /// Where the simulation has the object now, for what follows it.
    now: DAffine3,
}

/// A passive rigid body that follows its object's animation: a door that
/// swings, a paddle that sweeps. Where it was when the bake began, to move
/// it by as much as the object has moved since.
struct Kinematic {
    object: ObjectId,
    body: usize,
    pos: DVec3,
    rot: DQuat,
    world: DAffine3,
}

/// Baked frames of vertices, and of particles.
type Frames<T> = Vec<Arc<[T]>>;
/// A cloth or soft body being baked.
type Baking = (ObjectId, Uid, SoftSim, Frames<kreate_math::Vec3>);
/// A particle system being baked: its object, which system, the sim.
type Emitting<S> = (ObjectId, Uid, usize, S, Frames<super::Particle>);

/// A gas domain being baked.
struct Gas {
    object: ObjectId,
    uid: Uid,
    sim: GasSim,
    frames: Vec<GasFrame>,
}

/// A liquid domain being baked.
struct Liquid {
    object: ObjectId,
    uid: Uid,
    sim: LiquidSim,
    frames: Vec<Arc<kreate_mesh::MeshData>>,
}

/// The liquid's surface in the domain object's own space: empty when there
/// is no liquid, so an emptied domain shows nothing rather than its box.
fn surface_of(sim: &LiquidSim, world: DAffine3) -> Arc<kreate_mesh::MeshData> {
    let mut mesh = sim
        .surface()
        .unwrap_or_else(|| kreate_mesh::MeshData::from_faces(Vec::new(), &[]));
    let back = world.inverse();
    for p in mesh.positions_mut() {
        *p = back.transform_point3(p.as_dvec3()).as_vec3();
    }
    Arc::new(mesh)
}

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
    bake_with(doc, |_, _| true).expect("a bake that is never stopped finishes")
}

/// The same, telling `progress` how many frames of how many are done after
/// each one; it answers whether to go on. `None` when it said to stop.
pub fn bake_with(
    doc: &Document,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Option<(PhysicsCache, BakeSummary)> {
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

    // Liquid flows count hidden too: a tap is usually kept out of sight.
    let ids: Vec<ObjectId> = scene
        .objects()
        .filter(|(id, o)| {
            o.mesh().is_some()
                && !o.physics.is_empty()
                && (scene.is_shown(*id) || o.physics.fluid.is_some())
        })
        .map(|(id, _)| id)
        .collect();

    // What collides with cloth, soft bodies and particles.
    let mut colliders: Vec<(ObjectId, Collider)> = Vec::new();
    // Rigid bodies.
    let mut world = RigidWorld::new(settings.gravity, settings.iterations);
    let mut tracked: Vec<Tracked> = Vec::new();
    let mut kinematic: Vec<Kinematic> = Vec::new();
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
            colliders.push((
                id,
                Collider::new(&mesh, w, c.thickness, c.friction, c.bounciness),
            ));
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
            body.gravity = r.gravity;
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
                    now: w,
                });
            } else {
                kinematic.push(Kinematic {
                    object: id,
                    body: world.bodies.len() - 1,
                    pos: centre,
                    rot,
                    world: w,
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

    // Liquids: each domain with every flow in the scene, and every collision
    // object but itself for its obstacles.
    let mut obstacles: Vec<(ObjectId, Arc<kreate_mesh::MeshData>)> = Vec::new();
    let mut flows: Vec<(super::FluidFlow, Vec<[kreate_math::Vec3; 3]>)> = Vec::new();
    for &id in &ids {
        let Some(o) = scene.object(id) else { continue };
        let Some(mesh) = eval.evaluated_base(&scene, id, &ctx) else {
            continue;
        };
        if o.physics.collision.is_some() {
            obstacles.push((id, mesh.clone()));
        }
        if let Some(super::Fluid::Flow(flow)) = o.physics.fluid {
            flows.push((flow, world_triangles(&mesh, scene.world_transform(id))));
        }
    }
    let mut liquids: Vec<Liquid> = Vec::new();
    let mut gases: Vec<Gas> = Vec::new();
    for &id in &ids {
        let Some(o) = scene.object(id) else { continue };
        let Some(super::Fluid::Domain(domain)) = o.physics.fluid else {
            continue;
        };
        let Some(mesh) = eval.evaluated_base(&scene, id, &ctx) else {
            continue;
        };
        let w = scene.world_transform(id);
        let bounds = kreate_math::Aabb::from_points(
            mesh.positions()
                .iter()
                .map(|p| w.transform_point3(p.as_dvec3()).as_vec3()),
        );
        if bounds.is_empty() {
            continue;
        }
        let walls: Vec<Vec<[kreate_math::Vec3; 3]>> = obstacles
            .iter()
            .filter(|(o, _)| *o != id)
            .map(|(o, m)| world_triangles(m, scene.world_transform(*o)))
            .collect();
        if domain.kind == super::DomainKind::Gas {
            let mut sim = GasSim::new(
                bounds,
                domain.resolution,
                GasSettings {
                    buoyancy: domain.buoyancy as f32,
                    vorticity: domain.vorticity as f32,
                    dissipation: domain.dissipation as f32,
                    ..GasSettings::default()
                },
            );
            sim.set_obstacles(&walls);
            for (flow, tris) in flows
                .iter()
                .filter(|(f, _)| f.substance != Substance::Liquid)
            {
                let heat = flow.temperature.clamp(0.0, 2.0) as f32;
                let emission = match flow.substance {
                    Substance::Smoke => Emission {
                        smoke: 1.0,
                        heat,
                        fuel: 0.0,
                        velocity: flow.velocity.as_vec3(),
                    },
                    Substance::Fire => Emission {
                        smoke: 0.0,
                        heat,
                        fuel: 1.0,
                        velocity: flow.velocity.as_vec3(),
                    },
                    // Fire that smokes: a little smoke of its own, so the
                    // flame shows through it; its burning makes the rest.
                    _ => Emission {
                        smoke: 0.25,
                        heat,
                        fuel: 1.0,
                        velocity: flow.velocity.as_vec3(),
                    },
                };
                match flow.kind {
                    super::FlowKind::Geometry => sim.puff(tris, emission),
                    super::FlowKind::Inflow => sim.add_source(tris, emission),
                    super::FlowKind::Outflow => sim.add_sink(tris),
                }
            }
            let (smoke, flame) = sim.snapshot();
            gases.push(Gas {
                object: id,
                uid: o.uid,
                frames: vec![GasFrame {
                    smoke: smoke.into(),
                    flame: flame.into(),
                }],
                sim,
            });
            continue;
        }
        let mut sim = LiquidSim::new(bounds, domain.resolution, domain.flip, o.uid.raw());
        sim.set_obstacles(&walls);
        for (flow, tris) in flows
            .iter()
            .filter(|(f, _)| f.substance == Substance::Liquid)
        {
            let speed = flow.velocity.as_vec3();
            match flow.kind {
                super::FlowKind::Geometry => sim.fill(tris, speed),
                super::FlowKind::Inflow => sim.add_inflow(tris, speed),
                super::FlowKind::Outflow => sim.add_outflow(tris),
            }
        }
        let first = surface_of(&sim, w);
        liquids.push(Liquid {
            object: id,
            uid: o.uid,
            sim,
            frames: vec![first],
        });
    }

    let total = (end - start) as usize;
    for frame in start + 1..=end {
        if !progress((frame - start - 1) as usize, total) {
            return None;
        }
        scene.animate(frame as f64);
        // Passive bodies where their objects' keys put them, moving at the
        // speed that took them there, so what they push is pushed.
        for k in &kinematic {
            let w = scene.world_transform(k.object);
            let moved = w * k.world.inverse();
            let (_, turn, _) = moved.to_scale_rotation_translation();
            let b = &mut world.bodies[k.body];
            let (pos, rot) = (moved.transform_point3(k.pos), (turn * k.rot).normalize());
            b.vel = (pos - b.pos) / dt;
            let (axis, angle) = (rot * b.rot.inverse()).to_axis_angle();
            let angle = if angle > std::f64::consts::PI {
                angle - std::f64::consts::TAU
            } else {
                angle
            };
            b.spin = axis * (angle / dt);
            b.pos = pos;
            b.rot = rot;
            if let super::rigid::Shape::Mesh(c) = &mut b.shape {
                c.place(w);
            }
        }
        if !tracked.is_empty() {
            world.advance(dt, substeps);
            for t in tracked.iter_mut() {
                let b = &world.bodies[t.body];
                let moved = DAffine3::from_rotation_translation(b.rot * t.turn.inverse(), b.pos)
                    * DAffine3::from_translation(-t.centre)
                    * t.world;
                t.now = moved;
                t.frames.push(local_for(&scene, t.object, moved));
            }
        }
        // Colliders where they are this frame: a collider that is also a
        // simulated rigid body where the simulation has it, so cloth lands
        // on a box that is still falling.
        for (id, c) in colliders.iter_mut() {
            let w = tracked
                .iter()
                .find(|t| t.object == *id)
                .map_or_else(|| scene.world_transform(*id), |t| t.now);
            c.place(w);
        }
        let near: Vec<&Collider> = colliders.iter().map(|(_, c)| c).collect();
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
            sim.advance(
                frame as f64,
                dt,
                substeps.min(10),
                settings.gravity,
                w,
                &near,
            );
            frames.push(sim.snapshot(w).into());
        }
        for (id, _, _, sim, frames) in hairs.iter_mut() {
            let w = scene.world_transform(*id);
            sim.advance(dt, substeps, settings.gravity, w, &near);
            frames.push(sim.snapshot(w).into());
        }
        for liquid in liquids.iter_mut() {
            // Obstacles where they are this frame: a paddle sweeps.
            let walls: Vec<Vec<[kreate_math::Vec3; 3]>> = obstacles
                .iter()
                .filter(|(o, _)| *o != liquid.object)
                .map(|(o, m)| {
                    let w = tracked
                        .iter()
                        .find(|t| t.object == *o)
                        .map_or_else(|| scene.world_transform(*o), |t| t.now);
                    world_triangles(m, w)
                })
                .collect();
            if !walls.is_empty() {
                liquid.sim.set_obstacles(&walls);
            }
            liquid.sim.advance(dt as f32, settings.gravity.as_vec3());
            let w = scene.world_transform(liquid.object);
            liquid.frames.push(surface_of(&liquid.sim, w));
        }
        for gas in gases.iter_mut() {
            let walls: Vec<Vec<[kreate_math::Vec3; 3]>> = obstacles
                .iter()
                .filter(|(o, _)| *o != gas.object)
                .map(|(o, m)| {
                    let w = tracked
                        .iter()
                        .find(|t| t.object == *o)
                        .map_or_else(|| scene.world_transform(*o), |t| t.now);
                    world_triangles(m, w)
                })
                .collect();
            if !walls.is_empty() {
                gas.sim.set_obstacles(&walls);
            }
            gas.sim.advance(dt as f32, settings.gravity.as_vec3());
            let (smoke, flame) = gas.sim.snapshot();
            gas.frames.push(GasFrame {
                smoke: smoke.into(),
                flame: flame.into(),
            });
        }
    }

    let summary = BakeSummary {
        frames: (end - start + 1) as usize,
        rigid: tracked.len(),
        soft: softs.len(),
        particles: emitters.len() + hairs.len(),
        fluid: liquids.len() + gases.len(),
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
            .map(|(_, uid, _, frames)| SoftTrack {
                object: uid,
                frames,
            })
            .collect(),
        particles: emitters
            .into_iter()
            .map(|(_, uid, system, _, frames)| (uid, system, frames))
            .chain(
                hairs
                    .into_iter()
                    .map(|(_, uid, system, _, frames)| (uid, system, frames)),
            )
            .map(|(object, system, frames)| ParticleTrack {
                object,
                system,
                frames,
            })
            .collect(),
        fluid: liquids
            .into_iter()
            .map(|l| FluidTrack {
                object: l.uid,
                frames: l.frames,
            })
            .collect(),
        gas: gases
            .into_iter()
            .map(|g| GasTrack {
                object: g.uid,
                cells: g.sim.cells(),
                bounds: g.sim.bounds(),
                frames: g.frames,
            })
            .collect(),
    };
    progress(total, total);
    Some((cache, summary))
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
    fn a_bake_reports_its_frames_and_stops_when_told() {
        let mut doc = Document::new();
        short(&mut doc, 20);
        let cube = add(
            &mut doc,
            "Cube",
            kreate_mesh::primitives::cube(1.0),
            DVec3::new(0.0, 0.0, 3.0),
        );
        doc.object_mut(cube).unwrap().physics.rigid = Some(RigidBody::default());
        let mut seen = Vec::new();
        assert!(bake_with(&doc, |done, total| {
            seen.push((done, total));
            true
        })
        .is_some());
        assert_eq!(seen.first(), Some(&(0, 19)));
        assert_eq!(seen.last(), Some(&(19, 19)));
        assert!(bake_with(&doc, |done, _| done < 5).is_none(), "stopped");
    }

    #[test]
    fn a_baked_cube_falls_onto_the_floor_and_plays_back() {
        let mut doc = Document::new();
        short(&mut doc, 60);
        let floor = add(
            &mut doc,
            "Floor",
            kreate_mesh::primitives::plane(10.0),
            DVec3::ZERO,
        );
        let cube = add(
            &mut doc,
            "Cube",
            kreate_mesh::primitives::cube(1.0),
            DVec3::new(0.0, 0.0, 3.0),
        );
        doc.object_mut(floor).unwrap().physics.rigid = Some(RigidBody {
            kind: BodyKind::Passive,
            shape: CollisionShape::Mesh,
            ..RigidBody::default()
        });
        doc.object_mut(cube).unwrap().physics.rigid = Some(RigidBody::default());
        let (cache, summary) = bake(&doc);
        assert_eq!(summary.rigid, 1, "the floor does not move");
        assert_eq!(cache.rigid[0].frames.len(), 60);
        assert_eq!(
            cache.rigid[0].frames[0].translation.z, 3.0,
            "starts where it was"
        );
        let last = cache.rigid[0].frames[59].translation.z;
        assert!((last - 0.5).abs() < 0.05, "rests on the floor: {last}");

        doc.physics_cache = Arc::new(cache);
        doc.animate(60.0);
        assert!((doc.object(cube).unwrap().transform.translation.z - 0.5).abs() < 0.05);
        doc.animate(1.0);
        assert_eq!(
            doc.object(cube).unwrap().transform.translation.z,
            3.0,
            "and back"
        );
    }

    #[test]
    fn an_animated_passive_body_moves_and_carries_what_sits_on_it() {
        let mut doc = Document::new();
        short(&mut doc, 48);
        // A slab that slides along X under a box resting on it.
        let slab = add(
            &mut doc,
            "Slab",
            kreate_mesh::primitives::cube(1.0),
            DVec3::new(0.0, 0.0, 0.5),
        );
        {
            let o = doc.object_mut(slab).unwrap();
            o.physics.rigid = Some(RigidBody {
                kind: BodyKind::Passive,
                shape: CollisionShape::Box,
                friction: 1.0,
                ..RigidBody::default()
            });
            o.transform.scale = DVec3::new(4.0, 2.0, 1.0);
        }
        let mut action = crate::animation::Action::new("Slide");
        for (frame, x) in [(1.0, 0.0), (48.0, 2.0)] {
            action.curve_mut("location", 0).insert(
                frame,
                x,
                crate::animation::Interpolation::Linear,
            );
            action.curve_mut("location", 2).insert(
                frame,
                0.5,
                crate::animation::Interpolation::Linear,
            );
        }
        let a = doc.add_action(action);
        doc.object_mut(slab).unwrap().action = Some(a);
        let rider = add(
            &mut doc,
            "Rider",
            kreate_mesh::primitives::cube(0.5),
            DVec3::new(0.0, 0.0, 1.26),
        );
        doc.object_mut(rider).unwrap().physics.rigid = Some(RigidBody {
            shape: CollisionShape::Box,
            friction: 1.0,
            ..RigidBody::default()
        });
        let (cache, _) = bake(&doc);
        let x = cache.rigid[0].frames.last().unwrap().translation.x;
        assert!(x > 1.0, "carried along by the slab: {x}");
    }

    #[test]
    fn cloth_lands_on_a_box_that_is_still_falling() {
        let mut doc = Document::new();
        short(&mut doc, 36);
        let floor = add(
            &mut doc,
            "Floor",
            kreate_mesh::primitives::plane(10.0),
            DVec3::ZERO,
        );
        {
            let o = doc.object_mut(floor).unwrap();
            o.physics.rigid = Some(RigidBody {
                kind: BodyKind::Passive,
                shape: CollisionShape::Mesh,
                ..RigidBody::default()
            });
            o.physics.collision = Some(Collision::default());
        }
        let block = add(
            &mut doc,
            "Block",
            kreate_mesh::primitives::cube(1.0),
            DVec3::new(0.0, 0.0, 2.0),
        );
        {
            let o = doc.object_mut(block).unwrap();
            o.physics.rigid = Some(RigidBody {
                shape: CollisionShape::Box,
                ..RigidBody::default()
            });
            o.physics.collision = Some(Collision::default());
        }
        let cloth = add(
            &mut doc,
            "Cloth",
            kreate_mesh::primitives::grid(2.0, 12, 12),
            DVec3::new(0.0, 0.0, 2.8),
        );
        doc.object_mut(cloth).unwrap().physics.soft = Some(SoftBody::new(SoftKind::Cloth));
        let (cache, _) = bake(&doc);
        let block_z = cache.rigid[0].frames.last().unwrap().translation.z;
        assert!((block_z - 0.5).abs() < 0.1, "{block_z}");
        let last = cache.soft[0].frames.last().unwrap();
        let middle = last
            .iter()
            .filter(|p| p.x.abs() < 0.2 && p.y.abs() < 0.2)
            .map(|p| p.z + 2.8)
            .fold(f32::MAX, f32::min);
        assert!(
            middle > 0.95,
            "on the lid where the block came to rest: {middle}"
        );
    }

    #[test]
    fn a_child_body_is_written_back_through_its_parent() {
        let mut doc = Document::new();
        short(&mut doc, 48);
        let floor = add(
            &mut doc,
            "Floor",
            kreate_mesh::primitives::plane(10.0),
            DVec3::ZERO,
        );
        doc.object_mut(floor).unwrap().physics.rigid = Some(RigidBody {
            kind: BodyKind::Passive,
            shape: CollisionShape::Box,
            ..RigidBody::default()
        });
        let holder = add(
            &mut doc,
            "Holder",
            kreate_mesh::primitives::cube(0.1),
            DVec3::new(5.0, 0.0, 0.0),
        );
        let cube = add(
            &mut doc,
            "Cube",
            kreate_mesh::primitives::cube(1.0),
            DVec3::new(5.0, 0.0, 2.0),
        );
        doc.set_parent(cube, Some(holder)).unwrap();
        doc.object_mut(cube).unwrap().physics.rigid = Some(RigidBody {
            shape: CollisionShape::Box,
            ..RigidBody::default()
        });
        let (cache, _) = bake(&doc);
        doc.physics_cache = Arc::new(cache);
        doc.animate(48.0);
        let at = doc.world_transform(cube).translation;
        assert!(
            (at.z - 0.5).abs() < 0.05 && (at.x - 5.0).abs() < 0.05,
            "{at:?}"
        );
    }

    #[test]
    fn a_body_with_no_gravity_floats_and_is_still_knocked_on() {
        // Two cubes side by side in the air: one falls, one with no gravity
        // stays — and a third, thrown at the floater, moves it.
        let mut doc = Document::new();
        short(&mut doc, 24);
        let faller = add(
            &mut doc,
            "Faller",
            kreate_mesh::primitives::cube(1.0),
            DVec3::new(-3.0, 0.0, 3.0),
        );
        let floater = add(
            &mut doc,
            "Floater",
            kreate_mesh::primitives::cube(1.0),
            DVec3::new(3.0, 0.0, 3.0),
        );
        for (id, gravity) in [(faller, 1.0), (floater, 0.0)] {
            doc.object_mut(id).unwrap().physics.rigid = Some(RigidBody {
                shape: CollisionShape::Box,
                gravity,
                ..RigidBody::default()
            });
        }
        let (cache, _) = bake(&doc);
        doc.physics_cache = Arc::new(cache);
        doc.animate(24.0);
        assert!(
            doc.world_transform(faller).translation.z < 2.0,
            "the faller did not fall"
        );
        let floats = doc.world_transform(floater).translation;
        assert!(
            (floats - DVec3::new(3.0, 0.0, 3.0)).length() < 1e-3,
            "the floater moved by itself: {floats:?}"
        );
    }

    #[test]
    fn baked_cloth_drapes_and_is_laid_onto_the_evaluated_mesh() {
        let mut doc = Document::new();
        short(&mut doc, 36);
        let block = add(
            &mut doc,
            "Block",
            kreate_mesh::primitives::cube(1.0),
            DVec3::ZERO,
        );
        doc.object_mut(block).unwrap().physics.collision = Some(Collision::default());
        let cloth = add(
            &mut doc,
            "Cloth",
            kreate_mesh::primitives::grid(2.0, 12, 12),
            DVec3::new(0.0, 0.0, 1.2),
        );
        doc.object_mut(cloth).unwrap().physics.soft = Some(SoftBody::new(SoftKind::Cloth));
        let (cache, summary) = bake(&doc);
        assert_eq!(summary.soft, 1);
        doc.physics_cache = Arc::new(cache);
        doc.animate(36.0);
        let mut eval = ModifierCache::new();
        let mesh = eval.evaluated(&doc, cloth, &EvalCtx::still()).unwrap();
        // In the cloth's own space, lifted 1.2: the lowest corner hangs below
        // the lid at 0.5, so below -0.7 here.
        let lowest = mesh
            .positions()
            .iter()
            .map(|p| p.z)
            .fold(f32::MAX, f32::min);
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
        let e = add(
            &mut doc,
            "Emitter",
            kreate_mesh::primitives::plane(1.0),
            DVec3::ZERO,
        );
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
        assert_eq!(
            after.num_faces(),
            before.num_faces() + 50 * 8,
            "fifty diamonds"
        );
    }

    #[test]
    fn a_liquid_falls_in_its_domain_and_the_domain_shows_it() {
        use crate::physics::{Fluid, FluidDomain, FluidFlow};
        let mut doc = Document::new();
        short(&mut doc, 24);
        let domain = add(
            &mut doc,
            "Domain",
            kreate_mesh::primitives::cube(2.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        let water = add(
            &mut doc,
            "Water",
            kreate_mesh::primitives::cube(0.8),
            DVec3::new(-0.5, 0.0, 1.4),
        );
        doc.object_mut(domain).unwrap().physics.fluid = Some(Fluid::Domain(FluidDomain {
            resolution: 16,
            ..Default::default()
        }));
        doc.object_mut(water).unwrap().physics.fluid = Some(Fluid::Flow(FluidFlow::default()));
        let (cache, summary) = bake(&doc);
        assert_eq!(summary.fluid, 1);
        assert_eq!(cache.fluid[0].frames.len(), 24);
        let top =
            |m: &kreate_mesh::MeshData| m.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        // In the domain's own space, which sits a metre up.
        let (first, last) = (&cache.fluid[0].frames[0], &cache.fluid[0].frames[23]);
        assert!(first.num_faces() > 0 && last.num_faces() > 0);
        assert!(
            top(last) < top(first) - 0.2,
            "{} from {}",
            top(last),
            top(first)
        );

        let uid = doc.object(domain).unwrap().uid;
        doc.physics_cache = Arc::new(cache);
        doc.animate(24.0);
        let shown = ModifierCache::new()
            .evaluated(&doc, domain, &EvalCtx::still())
            .unwrap();
        assert!(Arc::ptr_eq(
            &shown,
            doc.physics_cache.fluid_at(uid, 24.0).unwrap()
        ));
    }

    #[test]
    fn smoke_rises_in_its_domain_and_the_box_is_not_drawn() {
        use crate::physics::{DomainKind, FlowKind, Fluid, FluidDomain, FluidFlow, Substance};
        let mut doc = Document::new();
        short(&mut doc, 24);
        let domain = add(
            &mut doc,
            "Domain",
            kreate_mesh::primitives::cube(2.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        let fire = add(
            &mut doc,
            "Fire",
            kreate_mesh::primitives::cube(0.4),
            DVec3::new(0.0, 0.0, 0.3),
        );
        doc.object_mut(domain).unwrap().physics.fluid = Some(Fluid::Domain(FluidDomain {
            kind: DomainKind::Gas,
            resolution: 16,
            ..Default::default()
        }));
        doc.object_mut(fire).unwrap().physics.fluid = Some(Fluid::Flow(FluidFlow {
            kind: FlowKind::Inflow,
            substance: Substance::SmokeAndFire,
            ..Default::default()
        }));
        let (cache, summary) = bake(&doc);
        assert_eq!(summary.fluid, 1);
        assert!(cache.fluid.is_empty(), "a gas is not a liquid surface");
        let track = &cache.gas[0];
        assert_eq!(track.frames.len(), 24);
        let cells = track.cells.iter().product::<usize>();
        let last = &track.frames[23];
        assert_eq!(last.smoke.len(), cells);
        assert!(last.flame.iter().any(|f| *f > 64), "it burns");
        // Smoke in the upper half by the end.
        let [nx, ny, nz] = track.cells;
        let high = (nz / 2 * ny * nx..cells)
            .filter(|c| last.smoke[*c] > 10)
            .count();
        assert!(high > 0, "nothing rose");

        let uid = doc.object(domain).unwrap().uid;
        doc.physics_cache = Arc::new(cache);
        doc.animate(12.0);
        assert!(doc.physics_cache.gas_at(uid, 12.0).is_some());
        let shown = ModifierCache::new()
            .evaluated(&doc, domain, &EvalCtx::still())
            .unwrap();
        assert_eq!(shown.num_faces(), 0, "the box is not drawn");
    }
}
