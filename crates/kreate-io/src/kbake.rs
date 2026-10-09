//! `.kbake`: a physics bake in a `.kreate` archive.
//!
//! So a baked scene opens baked: the rigid bodies' placements, the cloth and
//! soft bodies' vertices and the particles, frame by frame. Little-endian,
//! counted, and read defensively — a bake is a convenience, so one that is
//! damaged or from a newer build is skipped with the scene still opening,
//! and a bake makes it again.
//!
//! ```text
//! "KBAKE\0" u32 version  i32 start
//! u32 rigid   { u64 uid  u32 frames  frames × (3 f64 translation, 4 f64 rotation, 3 f64 scale) }
//! u32 soft    { u64 uid  u32 frames  u32 verts  frames × verts × 3 f32 }
//! u32 systems { u64 uid  u32 system  u32 frames  frames × (u32 n  n × 6 f32) }
//! ```

use std::sync::Arc;

use kreate_math::{DQuat, DTransform, DVec3, Vec3};
use kreate_scene::physics::{Particle, ParticleTrack, PhysicsCache, RigidTrack, SoftTrack};
use kreate_scene::Uid;

const MAGIC: &[u8; 6] = b"KBAKE\0";
const VERSION: u32 = 1;

pub fn write(cache: &PhysicsCache) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&cache.start.to_le_bytes());
    let u32_ = |out: &mut Vec<u8>, v: usize| out.extend_from_slice(&(v as u32).to_le_bytes());
    u32_(&mut out, cache.rigid.len());
    for t in &cache.rigid {
        out.extend_from_slice(&t.object.raw().to_le_bytes());
        u32_(&mut out, t.frames.len());
        for f in &t.frames {
            let r = f.rotation;
            for v in [
                f.translation.x,
                f.translation.y,
                f.translation.z,
                r.x,
                r.y,
                r.z,
                r.w,
                f.scale.x,
                f.scale.y,
                f.scale.z,
            ] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    u32_(&mut out, cache.soft.len());
    for t in &cache.soft {
        out.extend_from_slice(&t.object.raw().to_le_bytes());
        u32_(&mut out, t.frames.len());
        u32_(&mut out, t.frames.first().map_or(0, |f| f.len()));
        for f in &t.frames {
            for p in f.iter() {
                for v in p.to_array() {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    u32_(&mut out, cache.particles.len());
    for t in &cache.particles {
        out.extend_from_slice(&t.object.raw().to_le_bytes());
        u32_(&mut out, t.system);
        u32_(&mut out, t.frames.len());
        for f in &t.frames {
            u32_(&mut out, f.len());
            for p in f.iter() {
                for v in [p.position.to_array(), p.heading.to_array()].concat() {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    out
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let end = self.at.checked_add(n)?;
        let out = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(out)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    /// A count, refused if what it counts could not fit in what is left.
    fn count(&mut self, each: usize) -> Option<usize> {
        let n = self.u32()? as usize;
        (n.checked_mul(each.max(1))? <= self.bytes.len() - self.at).then_some(n)
    }
}

/// The bake in `bytes`, or `None` when it is not one this build can read.
pub fn read(bytes: &[u8]) -> Option<PhysicsCache> {
    let mut r = Reader { bytes, at: 0 };
    if r.take(6)? != MAGIC || r.u32()? != VERSION {
        return None;
    }
    let start = r.i32()?;
    let mut cache = PhysicsCache {
        start,
        ..Default::default()
    };
    for _ in 0..r.count(12)? {
        let object = Uid::from_raw(r.u64()?);
        let n = r.count(80)?;
        let mut frames = Vec::with_capacity(n);
        for _ in 0..n {
            let t = DVec3::new(r.f64()?, r.f64()?, r.f64()?);
            let q = DQuat::from_xyzw(r.f64()?, r.f64()?, r.f64()?, r.f64()?);
            let s = DVec3::new(r.f64()?, r.f64()?, r.f64()?);
            frames.push(DTransform {
                translation: t,
                rotation: q,
                scale: s,
            });
        }
        cache.rigid.push(RigidTrack { object, frames });
    }
    for _ in 0..r.count(16)? {
        let object = Uid::from_raw(r.u64()?);
        let n = r.u32()? as usize;
        let verts = r.u32()? as usize;
        if n.checked_mul(verts)?.checked_mul(12)? > bytes.len() - r.at {
            return None;
        }
        let mut frames = Vec::with_capacity(n);
        for _ in 0..n {
            let mut f = Vec::with_capacity(verts);
            for _ in 0..verts {
                f.push(Vec3::new(r.f32()?, r.f32()?, r.f32()?));
            }
            frames.push(Arc::<[Vec3]>::from(f));
        }
        cache.soft.push(SoftTrack { object, frames });
    }
    for _ in 0..r.count(16)? {
        let object = Uid::from_raw(r.u64()?);
        let system = r.u32()? as usize;
        let n = r.count(4)?;
        let mut frames = Vec::with_capacity(n);
        for _ in 0..n {
            let count = r.count(24)?;
            let mut f = Vec::with_capacity(count);
            for _ in 0..count {
                f.push(Particle {
                    position: Vec3::new(r.f32()?, r.f32()?, r.f32()?),
                    heading: Vec3::new(r.f32()?, r.f32()?, r.f32()?),
                });
            }
            frames.push(Arc::<[Particle]>::from(f));
        }
        cache.particles.push(ParticleTrack {
            object,
            system,
            frames,
        });
    }
    Some(cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PhysicsCache {
        PhysicsCache {
            start: 7,
            rigid: vec![RigidTrack {
                object: Uid::from_raw(3),
                frames: vec![
                    DTransform::IDENTITY,
                    DTransform {
                        translation: DVec3::new(1.0, -2.0, 3.5),
                        rotation: DQuat::from_rotation_z(0.4),
                        scale: DVec3::splat(2.0),
                    },
                ],
            }],
            soft: vec![SoftTrack {
                object: Uid::from_raw(9),
                frames: vec![
                    Arc::from(vec![Vec3::X, Vec3::Y]),
                    Arc::from(vec![Vec3::Z, Vec3::ONE]),
                ],
            }],
            particles: vec![ParticleTrack {
                object: Uid::from_raw(9),
                system: 1,
                frames: vec![
                    Arc::from(Vec::<Particle>::new()),
                    Arc::from(vec![Particle {
                        position: Vec3::ONE,
                        heading: Vec3::Z,
                    }]),
                ],
            }],
        }
    }

    #[test]
    fn a_bake_comes_back_as_it_went() {
        let cache = sample();
        let back = read(&write(&cache)).expect("read");
        assert_eq!(back.start, 7);
        assert_eq!(back.rigid, cache.rigid);
        assert_eq!(back.soft[0].object, cache.soft[0].object);
        assert_eq!(&*back.soft[0].frames[1], &*cache.soft[0].frames[1]);
        assert_eq!(back.particles[0].system, 1);
        assert_eq!(
            &*back.particles[0].frames[1],
            &*cache.particles[0].frames[1]
        );
    }

    #[test]
    fn a_damaged_bake_is_refused_not_trusted() {
        let bytes = write(&sample());
        for cut in [0, 5, 20, bytes.len() / 2, bytes.len() - 1] {
            assert!(read(&bytes[..cut]).is_none(), "cut at {cut}");
        }
        let mut wrong = bytes.clone();
        wrong[6] = 99;
        assert!(read(&wrong).is_none(), "another version");
        // A count far bigger than the file is not allocated for.
        let mut huge = bytes;
        huge[14..18].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(read(&huge).is_none());
    }
}
