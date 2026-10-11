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
    BodyKind, Collision, CollisionShape, DomainKind, FlowKind, Fluid, FluidDomain, FluidFlow,
    ObjectPhysics, ParticleKind, ParticleSettings, PhysicsWorld, RigidBody, SoftBody, SoftKind,
    Substance,
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
    /// Absent in files from before it was settable: the world's full pull.
    #[serde(default = "full_gravity", skip_serializing_if = "is_full_gravity")]
    pub gravity: f64,
}

fn full_gravity() -> f64 {
    1.0
}

fn is_full_gravity(g: &f64) -> bool {
    *g == 1.0
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
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub self_collision: bool,
    #[serde(default = "default_self_distance")]
    pub self_distance: f64,
}

fn default_self_distance() -> f64 {
    0.015
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

/// An object's part in a liquid: a domain's settings, or a flow's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FluidRecord {
    /// `domain` or `flow`.
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flip: Option<f64>,
    /// A flow's: `geometry`, `inflow` or `outflow`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flow: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub velocity: Option<[f64; 3]>,
    /// A domain's: `liquid` or `gas`. Absent is a liquid, as every domain
    /// was before gases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub buoyancy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vorticity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dissipation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smoke_color: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turbulence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<f64>,
    /// A flow's: `liquid`, `smoke`, `fire` or `smoke_and_fire`. Absent is
    /// liquid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub substance: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fluid: Option<FluidRecord>,
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
            gravity: r.gravity,
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
            self_collision: s.self_collision,
            self_distance: s.self_distance,
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
        fluid: p.fluid.map(|f| match f {
            Fluid::Domain(d) => FluidRecord {
                role: "domain".into(),
                resolution: Some(d.resolution),
                flip: Some(d.flip),
                flow: None,
                velocity: None,
                domain: Some(
                    match d.kind {
                        DomainKind::Liquid => "liquid",
                        DomainKind::Gas => "gas",
                    }
                    .into(),
                ),
                buoyancy: Some(d.buoyancy),
                vorticity: Some(d.vorticity),
                dissipation: Some(d.dissipation),
                smoke_color: Some(d.smoke_color.to_array()),
                turbulence: Some(d.turbulence),
                detail: Some(d.detail),
                substance: None,
                temperature: None,
            },
            Fluid::Flow(f) => FluidRecord {
                role: "flow".into(),
                resolution: None,
                flip: None,
                flow: Some(
                    match f.kind {
                        FlowKind::Geometry => "geometry",
                        FlowKind::Inflow => "inflow",
                        FlowKind::Outflow => "outflow",
                    }
                    .into(),
                ),
                velocity: Some(f.velocity.to_array()),
                domain: None,
                buoyancy: None,
                vorticity: None,
                dissipation: None,
                smoke_color: None,
                turbulence: None,
                detail: None,
                substance: Some(
                    match f.substance {
                        Substance::Liquid => "liquid",
                        Substance::Smoke => "smoke",
                        Substance::Fire => "fire",
                        Substance::SmokeAndFire => "smoke_and_fire",
                    }
                    .into(),
                ),
                temperature: Some(f.temperature),
            },
        }),
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
            gravity: if b.gravity.is_finite() {
                b.gravity.clamp(-10.0, 10.0)
            } else {
                1.0
            },
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
            self_collision: s.self_collision,
            self_distance: s.self_distance.max(0.0),
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
        fluid: r.fluid.as_ref().map(|f| {
            if f.role == "domain" {
                let d = FluidDomain::default();
                let finite = |v: Option<f64>, or: f64| v.filter(|v| v.is_finite()).unwrap_or(or);
                Fluid::Domain(FluidDomain {
                    kind: if f.domain.as_deref() == Some("gas") {
                        DomainKind::Gas
                    } else {
                        DomainKind::Liquid
                    },
                    resolution: f
                        .resolution
                        .unwrap_or(d.resolution)
                        .clamp(4, kreate_scene::physics::fluid::MAX_RESOLUTION),
                    flip: finite(f.flip, d.flip).clamp(0.0, 1.0),
                    buoyancy: finite(f.buoyancy, d.buoyancy).clamp(-50.0, 50.0),
                    vorticity: finite(f.vorticity, d.vorticity).clamp(0.0, 10.0),
                    dissipation: finite(f.dissipation, d.dissipation).clamp(0.0, 10.0),
                    smoke_color: f
                        .smoke_color
                        .map(kreate_math::DVec3::from_array)
                        .filter(|c| c.is_finite())
                        .unwrap_or(d.smoke_color)
                        .clamp(kreate_math::DVec3::ZERO, kreate_math::DVec3::ONE),
                    turbulence: finite(f.turbulence, d.turbulence).clamp(0.0, 10.0),
                    detail: finite(f.detail, d.detail).clamp(0.0, 1.0),
                })
            } else {
                Fluid::Flow(FluidFlow {
                    kind: match f.flow.as_deref() {
                        Some("inflow") => FlowKind::Inflow,
                        Some("outflow") => FlowKind::Outflow,
                        _ => FlowKind::Geometry,
                    },
                    velocity: f
                        .velocity
                        .map(kreate_math::DVec3::from_array)
                        .filter(|v| v.is_finite())
                        .unwrap_or_default(),
                    substance: match f.substance.as_deref() {
                        Some("smoke") => Substance::Smoke,
                        Some("fire") => Substance::Fire,
                        Some("smoke_and_fire") => Substance::SmokeAndFire,
                        _ => Substance::Liquid,
                    },
                    temperature: f
                        .temperature
                        .filter(|t| t.is_finite())
                        .unwrap_or(1.0)
                        .clamp(0.0, 2.0),
                })
            }
        }),
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
                gravity: 0.25,
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
                self_collision: true,
                self_distance: 0.04,
            }),
            particles: vec![ParticleSettings {
                instance: Some(kreate_scene::Uid::from_raw(42)),
                count: 77,
                seed: 9,
                hair_dynamic: true,
                show_emitter: false,
                ..ParticleSettings::new("Fur", ParticleKind::Hair)
            }],
            fluid: Some(Fluid::Flow(FluidFlow {
                kind: FlowKind::Inflow,
                substance: Substance::SmokeAndFire,
                velocity: kreate_math::DVec3::new(0.5, 0.0, -2.0),
                temperature: 1.5,
            })),
        };
        let domain = ObjectPhysics {
            fluid: Some(Fluid::Domain(FluidDomain {
                kind: DomainKind::Gas,
                resolution: 48,
                flip: 0.8,
                buoyancy: 3.5,
                vorticity: 0.75,
                dissipation: 0.25,
                smoke_color: kreate_math::DVec3::new(0.25, 0.5, 0.75),
                turbulence: 1.25,
                detail: 0.375,
            })),
            ..Default::default()
        };
        assert_eq!(read(&write(&domain).unwrap()), domain);
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
