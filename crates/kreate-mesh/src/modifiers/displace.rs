//! The displace modifier: every vertex pushed along its normal by a
//! strength, roughened by noise — a plain surface made a hillside, a sheet
//! made hammered, without painting a thing.

use kreate_math::Vec3;

use crate::MeshData;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplaceParams {
    /// How far along the normal, in the mesh's units; negative sinks it.
    pub strength: f32,
    /// How far the noise reaches, as a fraction of the strength: 0 is a
    /// plain push, 1 is all noise.
    pub noise: f32,
    /// How big the noise's lumps are, in the mesh's units.
    pub scale: f32,
}

impl Default for DisplaceParams {
    fn default() -> Self {
        Self {
            strength: 0.1,
            noise: 0.0,
            scale: 1.0,
        }
    }
}

/// `input` with its vertices moved along their normals.
pub fn displace(input: &MeshData, params: &DisplaceParams) -> MeshData {
    if input.is_empty() || params.strength == 0.0 || !params.strength.is_finite() {
        return input.clone();
    }
    let normals = vertex_normals(input);
    let noise = params.noise.clamp(0.0, 1.0);
    let scale = if params.scale.is_finite() && params.scale > 0.0 {
        params.scale
    } else {
        1.0
    };
    let mut out = input.clone();
    for (p, n) in out.positions_mut().iter_mut().zip(&normals) {
        let lump = if noise > 0.0 {
            value_noise(*p / scale) * 2.0 - 1.0
        } else {
            0.0
        };
        let amount = params.strength * ((1.0 - noise) + noise * lump);
        *p += *n * amount;
    }
    out
}

/// Area-weighted vertex normals: each face's cross products summed onto its
/// corners, so a large face weighs more than a sliver.
pub fn vertex_normals(mesh: &MeshData) -> Vec<Vec3> {
    let positions = mesh.positions();
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        let mut n = Vec3::ZERO;
        for i in 0..ring.len() {
            let (a, b) = (
                positions[ring[i] as usize],
                positions[ring[(i + 1) % ring.len()] as usize],
            );
            n += a.cross(b);
        }
        for v in ring {
            normals[*v as usize] += n;
        }
    }
    for n in &mut normals {
        *n = n.normalize_or_zero();
    }
    normals
}

/// Smooth value noise in 0..1: a hash on the lattice, blended with a
/// smoothstep. The same everywhere for the same point, which is what a
/// modifier needs — no state, no seed.
pub fn value_noise(p: Vec3) -> f32 {
    let f = p.floor();
    let t = p - f;
    let s = t * t * (Vec3::splat(3.0) - 2.0 * t);
    let h = |x: f32, y: f32, z: f32| -> f32 {
        let mut n = (x as i64).wrapping_mul(73_856_093)
            ^ (y as i64).wrapping_mul(19_349_663)
            ^ (z as i64).wrapping_mul(83_492_791);
        n = (n ^ (n >> 13)).wrapping_mul(1_274_126_177);
        ((n ^ (n >> 16)) & 0xffff) as f32 / 65_535.0
    };
    let l = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x0 = l(h(f.x, f.y, f.z), h(f.x + 1.0, f.y, f.z), s.x);
    let x1 = l(h(f.x, f.y + 1.0, f.z), h(f.x + 1.0, f.y + 1.0, f.z), s.x);
    let x2 = l(h(f.x, f.y, f.z + 1.0), h(f.x + 1.0, f.y, f.z + 1.0), s.x);
    let x3 = l(
        h(f.x, f.y + 1.0, f.z + 1.0),
        h(f.x + 1.0, f.y + 1.0, f.z + 1.0),
        s.x,
    );
    l(l(x0, x1, s.y), l(x2, x3, s.y), s.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_plain_push_grows_a_sphere_by_the_strength() {
        let sphere = primitives::uv_sphere(1.0, 24, 12);
        let out = displace(
            &sphere,
            &DisplaceParams {
                strength: 0.25,
                noise: 0.0,
                scale: 1.0,
            },
        );
        for p in out.positions() {
            assert!((p.length() - 1.25).abs() < 0.02, "{p:?}");
        }
        assert_eq!(out.num_faces(), sphere.num_faces());
    }

    #[test]
    fn noise_roughens_within_the_strength_and_repeats_itself() {
        let sphere = primitives::uv_sphere(1.0, 24, 12);
        let p = DisplaceParams {
            strength: 0.2,
            noise: 1.0,
            scale: 0.5,
        };
        let out = displace(&sphere, &p);
        let again = displace(&sphere, &p);
        assert_eq!(out.positions(), again.positions(), "the same every time");
        let radii: Vec<f32> = out.positions().iter().map(|p| p.length()).collect();
        let (lo, hi) = radii
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), r| (lo.min(*r), hi.max(*r)));
        assert!(lo >= 0.8 - 0.02 && hi <= 1.2 + 0.02, "{lo} {hi}");
        assert!(hi - lo > 0.1, "roughened: {lo} {hi}");
    }
}
