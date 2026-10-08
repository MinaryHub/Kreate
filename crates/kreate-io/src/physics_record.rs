//! An object's physics settings and the scene's physics world, as a scene
//! file writes them. See `kreate_scene::physics`.
//!
//! Enumerations are words, not numbers, so a file reads as what it means and
//! an enumeration that grows does not renumber what is already written. A
//! word this build does not know reads as the default — a file from a newer
//! build loses that one setting, not the object.
//!
//! The bake is not here: it is large and a bake makes it again.

use kreate_scene::physics::{
    BodyKind, Collision, CollisionShape, ObjectPhysics, ParticleKind, ParticleSettings,
    PhysicsWorld, RigidBody, SoftBody, SoftKind,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RigidRecord {
    pub kind: String,
    pub shape: String,
    pub mass: f64,
    pub friction: f64,
    pub bounciness: f64,
    pub linear_damping: f64,
    pub angular_damping: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollisionRecord {
    pub thickness: f64,
    pub friction: f64,
    pub bounciness: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftRecord {
    pub kind: String,
    pub mass: f64,
    pub stiffness: f64,
    pub bending: f64,
    pub damping: f64,
    pub shape_stiffness: f64,
    pub pressure: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin_group: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticleRecord {
    pub name: String,
    pub kind: String,
    pub count: u32,
    pub seed: u32,
    pub frame_start: i32,
    pub frame_end: i32,
    pub lifetime: f64,
    pub normal_speed: f64,
    pub random_speed: f64,
    pub gravity: f64,
    pub drag: f64,
    pub size: f64,
    /// The instance object, by identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<u64>,
    pub hair_length: f64,
    pub hair_segments: u32,
    pub hair_dynamic: bool,
    pub hair_stiffness: f64,
    pub show_emitter: bool,
}

/// An object's physics; absent from the file when it has none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigid: Option<RigidRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision: Option<CollisionRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soft: Option<SoftRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub particles: Vec<ParticleRecord>,
}

/// The scene's physics world; absent when it is the default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldRecord {
    pub gravity: [f64; 3],
    pub substeps: u32,
    pub iterations: u32,
    pub frame_start: i32,
    pub frame_end: i32,
}

pub fn write(p: &ObjectPhysics) -> Option<PhysicsRecord> {
    if p.is_empty() {
        return None;
    }
    Some(PhysicsRecord {
        rigid: p.rigid.map(|r| RigidRecord {
            kind: match r.kind {
                BodyKind::Active => "active",
                BodyKind::Passive => "passive",
            }
            .into(),
            shape: match r.shape {
                CollisionShape::ConvexHull => "convex_hull",
                CollisionShape::Box => "box",
                CollisionShape::Sphere => "sphere",
                CollisionShape::Mesh => "mesh",
            }
            .into(),
            mass: r.mass,
            friction: r.friction,
            bounciness: r.bounciness,
            linear_damping: r.linear_damping,
            angular_damping: r.angular_damping,
        }),
        collision: p.collision.map(|c| CollisionRecord {
            thickness: c.thickness,
            friction: c.friction,
            bounciness: c.bounciness,
        }),
        soft: p.soft.as_ref().map(|s| SoftRecord {
            kind: match s.kind {
                SoftKind::Cloth => "cloth",
                SoftKind::Soft => "soft_body",
            }
            .into(),
            mass: s.mass,
            stiffness: s.stiffness,
            bending: s.bending,
            damping: s.damping,
            shape_stiffness: s.shape_stiffness,
            pressure: s.pressure,
            pin_group: s.pin_group.clone(),
        }),
        particles: p
            .particles
            .iter()
            .map(|s| ParticleRecord {
                name: s.name.clone(),
                kind: match s.kind {
                    ParticleKind::Emitter => "emitter",
                    ParticleKind::Hair => "hair",
                }
                .into(),
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
                instance: s.instance.map(|u| u.raw()),
                hair_length: s.hair_length,
                hair_segments: s.hair_segments,
                hair_dynamic: s.hair_dynamic,
                hair_stiffness: s.hair_stiffness,
                show_emitter: s.show_emitter,
            })
            .collect(),
    })
}

pub fn read(r: &PhysicsRecord) -> ObjectPhysics {
    ObjectPhysics {
        rigid: r.rigid.as_ref().map(|b| RigidBody {
            kind: if b.kind == "passive" {
                BodyKind::Passive
            } else {
                BodyKind::Active
            },
            shape: match b.shape.as_str() {
                "box" => CollisionShape::Box,
                "sphere" => CollisionShape::Sphere,
                "mesh" => CollisionShape::Mesh,
                _ => CollisionShape::ConvexHull,
            },
            mass: b.mass.max(0.001),
            friction: b.friction.clamp(0.0, 1.0),
            bounciness: b.bounciness.clamp(0.0, 1.0),
            linear_damping: b.linear_damping.clamp(0.0, 1.0),
            angular_damping: b.angular_damping.clamp(0.0, 1.0),
        }),
        collision: r.collision.as_ref().map(|c| Collision {
            thickness: c.thickness.max(0.0),
            friction: c.friction.clamp(0.0, 1.0),
            bounciness: c.bounciness.clamp(0.0, 1.0),
        }),
        soft: r.soft.as_ref().map(|s| SoftBody {
            kind: if s.kind == "soft_body" {
                SoftKind::Soft
            } else {
                SoftKind::Cloth
            },
            mass: s.mass.max(0.001),
            stiffness: s.stiffness.clamp(0.0, 1.0),
            bending: s.bending.clamp(0.0, 1.0),
            damping: s.damping.clamp(0.0, 1.0),
            shape_stiffness: s.shape_stiffness.clamp(0.0, 1.0),
            pressure: s.pressure.clamp(-1.0, 1.0),
            pin_group: s.pin_group.clone(),
        }),
        particles: r
            .particles
            .iter()
            .map(|p| ParticleSettings {
                name: p.name.clone(),
                kind: if p.kind == "hair" {
                    ParticleKind::Hair
                } else {
                    ParticleKind::Emitter
                },
                // A count a file can make absurd is brought into range.
                count: p.count.min(1_000_000),
                seed: p.seed,
                frame_start: p.frame_start,
                frame_end: p.frame_end.max(p.frame_start),
                lifetime: p.lifetime.max(0.0),
                normal_speed: p.normal_speed,
                random_speed: p.random_speed.max(0.0),
                gravity: p.gravity,
                drag: p.drag.clamp(0.0, 1.0),
                size: p.size.max(0.0),
                instance: p.instance.map(kreate_scene::Uid::from_raw),
                hair_length: p.hair_length.max(0.0),
                hair_segments: p.hair_segments.clamp(1, 64),
                hair_dynamic: p.hair_dynamic,
                hair_stiffness: p.hair_stiffness.clamp(0.0, 1.0),
                show_emitter: p.show_emitter,
            })
            .collect(),
    }
}

pub fn write_world(w: &PhysicsWorld) -> Option<WorldRecord> {
    (*w != PhysicsWorld::default()).then(|| WorldRecord {
        gravity: w.gravity.to_array(),
        substeps: w.substeps,
        iterations: w.iterations,
        frame_start: w.frame_start,
        frame_end: w.frame_end,
    })
}

pub fn read_world(r: &WorldRecord) -> PhysicsWorld {
    PhysicsWorld {
        gravity: kreate_math::DVec3::from_array(r.gravity),
        substeps: r.substeps.clamp(1, 100),
        iterations: r.iterations.clamp(1, 100),
        frame_start: r.frame_start,
        frame_end: r.frame_end.max(r.frame_start),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_setting_comes_back_and_none_are_at_their_defaults() {
        let p = ObjectPhysics {
            rigid: Some(RigidBody {
                kind: BodyKind::Passive,
                shape: CollisionShape::Sphere,
                mass: 3.5,
                friction: 0.25,
                bounciness: 0.75,
                linear_damping: 0.125,
                angular_damping: 0.375,
            }),
            collision: Some(Collision {
                thickness: 0.07,
                friction: 0.8,
                bounciness: 0.6,
            }),
            soft: Some(SoftBody {
                kind: SoftKind::Soft,
                mass: 2.0,
                stiffness: 0.3,
                bending: 0.2,
                damping: 0.4,
                shape_stiffness: 0.9,
                pressure: 0.5,
                pin_group: Some("Top".into()),
            }),
            particles: vec![ParticleSettings {
                instance: Some(kreate_scene::Uid::from_raw(42)),
                count: 77,
                seed: 9,
                hair_dynamic: true,
                show_emitter: false,
                ..ParticleSettings::new("Fur", ParticleKind::Hair)
            }],
        };
        let r = write(&p).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: PhysicsRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(read(&back), p);
        assert!(write(&ObjectPhysics::default()).is_none());

        let w = PhysicsWorld {
            gravity: kreate_math::DVec3::new(0.0, 0.0, -1.62),
            substeps: 20,
            iterations: 7,
            frame_start: 10,
            frame_end: 99,
        };
        assert_eq!(read_world(&write_world(&w).unwrap()), w);
        assert!(write_world(&PhysicsWorld::default()).is_none());
    }
}
