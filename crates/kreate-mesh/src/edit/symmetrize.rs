//! Symmetrize: one half of a mesh made the whole of it — the other half cut
//! away at the object's mirror plane and the kept half reflected over, its
//! seam welded. Blender's Mesh ▸ Symmetrize.
//!
//! Rebuilt from positions and faces: the faces the plane crosses are new, so
//! what was painted or unwrapped on the mesh is not carried over.

use std::collections::HashMap;

use kreate_math::Vec3;

use crate::modifiers::mirror::{mirror, MirrorParams};
use crate::MeshData;

/// Closer to the plane than this, a vertex is on it.
const ON_PLANE: f32 = 1e-5;

/// `input` made symmetric across the plane through its origin normal to
/// `axis` (0 X, 1 Y, 2 Z), the half on the positive side kept — or the
/// negative one with `from_negative`.
pub fn symmetrize(input: &MeshData, axis: u32, from_negative: bool) -> MeshData {
    let a = axis.min(2) as usize;
    let sign = if from_negative { -1.0 } else { 1.0 };
    let side = |p: Vec3| p[a] * sign;
    let positions = input.positions();

    let mut out: Vec<Vec3> = Vec::new();
    // The kept vertices, and the new ones where an edge crosses, each once.
    let mut kept: HashMap<u32, u32> = HashMap::new();
    let mut crossing: HashMap<(u32, u32), u32> = HashMap::new();
    let mut keep = |v: u32, out: &mut Vec<Vec3>| -> u32 {
        *kept.entry(v).or_insert_with(|| {
            let mut p = positions[v as usize];
            if side(p).abs() <= ON_PLANE {
                p[a] = 0.0;
            }
            out.push(p);
            out.len() as u32 - 1
        })
    };
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for f in 0..input.num_faces() {
        let ring = input.face_verts(f);
        let mut face: Vec<u32> = Vec::with_capacity(ring.len() + 2);
        for i in 0..ring.len() {
            let (u, w) = (ring[i], ring[(i + 1) % ring.len()]);
            let (pu, pw) = (positions[u as usize], positions[w as usize]);
            let (su, sw) = (side(pu), side(pw));
            if su >= -ON_PLANE {
                face.push(keep(u, &mut out));
            }
            // Strictly across: one end in, the other out.
            let across = (su > ON_PLANE && sw < -ON_PLANE) || (su < -ON_PLANE && sw > ON_PLANE);
            if across {
                let key = (u.min(w), u.max(w));
                let at = *crossing.entry(key).or_insert_with(|| {
                    let t = su / (su - sw);
                    let mut p = pu.lerp(pw, t);
                    p[a] = 0.0;
                    out.push(p);
                    out.len() as u32 - 1
                });
                face.push(at);
            }
        }
        face.dedup();
        if face.len() > 1 && face.first() == face.last() {
            face.pop();
        }
        // A face wholly on the plane would come back twice from the mirror.
        let flat = face.iter().all(|v| out[*v as usize][a] == 0.0);
        if face.len() >= 3 && !flat {
            faces.push(face);
        }
    }
    let half = MeshData::from_faces(out, &faces);
    mirror(
        &half,
        &MirrorParams {
            x: a == 0,
            y: a == 1,
            z: a == 2,
            merge_threshold: ON_PLANE * 10.0,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn moved(mut m: MeshData, by: Vec3) -> MeshData {
        for p in m.positions_mut() {
            *p += by;
        }
        m
    }

    #[test]
    fn a_cube_off_centre_becomes_its_positive_half_twice() {
        // From -0.5 to 1.5 along X: the plane cuts it a quarter of the way.
        let cube = moved(primitives::cube(2.0), Vec3::new(0.5, 0.0, 0.0));
        let out = symmetrize(&cube, 0, false);
        let (lo, hi) = out
            .positions()
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                (lo.min(p.x), hi.max(p.x))
            });
        assert!(
            (lo + 1.5).abs() < 1e-5 && (hi - 1.5).abs() < 1e-5,
            "{lo} {hi}"
        );
        assert!(crate::measure::is_closed(&out), "welded at the seam");
        let v = crate::measure::mass_properties(&out, |p| p).volume;
        // Two halves of 1.5 x 2 x 2.
        assert!((v - 12.0).abs() < 1e-3, "{v}");
    }

    #[test]
    fn the_negative_half_can_be_the_kept_one() {
        let cube = moved(primitives::cube(2.0), Vec3::new(0.5, 0.0, 0.0));
        let out = symmetrize(&cube, 0, true);
        let v = crate::measure::mass_properties(&out, |p| p).volume;
        // Two of 0.5 x 2 x 2.
        assert!((v - 4.0).abs() < 1e-3, "{v}");
        assert!(crate::measure::is_closed(&out));
    }

    #[test]
    fn a_symmetric_mesh_comes_back_the_same_size() {
        let sphere = primitives::uv_sphere(1.0, 16, 8);
        let out = symmetrize(&sphere, 1, false);
        let before = crate::measure::mass_properties(&sphere, |p| p).volume;
        let after = crate::measure::mass_properties(&out, |p| p).volume;
        assert!((before - after).abs() < 1e-3, "{before} {after}");
        assert!(crate::measure::is_closed(&out));
    }
}
