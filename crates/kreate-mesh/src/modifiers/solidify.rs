//! The solidify modifier: a surface given thickness.
//!
//! Two shells — the surface pushed out and pulled in along its vertex normals —
//! with the inner one turned inside out, and the open borders joined by a band
//! of quads. The everyday use is walls, panels, cloth and anything modelled as
//! a single sheet that has to read as solid.
//!
//! `offset` is Blender's: −1 grows the thickness inwards only (the original
//! surface is the outside), +1 outwards only, 0 half each way.

use kreate_math::Vec3;

use crate::MeshData;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidifyParams {
    /// How thick, in the mesh's units.
    pub thickness: f32,
    /// Where the original surface sits in the thickness, −1 to 1.
    pub offset: f32,
}

impl Default for SolidifyParams {
    fn default() -> Self {
        Self {
            thickness: 0.1,
            offset: -1.0,
        }
    }
}

/// Gives `input` thickness.
pub fn solidify(input: &MeshData, params: &SolidifyParams) -> MeshData {
    if input.is_empty() || params.thickness == 0.0 || !params.thickness.is_finite() {
        return input.clone();
    }
    let n = input.num_verts() as usize;
    let positions = input.positions();
    // Area-weighted vertex normals: a face's cross products summed onto its
    // corners, which weighs a large face more than a sliver.
    let mut normals = vec![Vec3::ZERO; n];
    for f in 0..input.num_faces() {
        let ring = input.face_verts(f);
        let mut face_normal = Vec3::ZERO;
        for i in 0..ring.len() {
            let (a, b) = (
                positions[ring[i] as usize],
                positions[ring[(i + 1) % ring.len()] as usize],
            );
            face_normal += a.cross(b);
        }
        for v in ring {
            normals[*v as usize] += face_normal;
        }
    }
    let offset = params.offset.clamp(-1.0, 1.0);
    let outwards = params.thickness * (1.0 + offset) * 0.5;
    let inwards = params.thickness * (1.0 - offset) * 0.5;
    let mut out_positions = Vec::with_capacity(n * 2);
    for (p, normal) in positions.iter().zip(&normals) {
        out_positions.push(*p + normal.normalize_or_zero() * outwards);
    }
    for (p, normal) in positions.iter().zip(&normals) {
        out_positions.push(*p - normal.normalize_or_zero() * inwards);
    }

    let inner = |v: u32| v + n as u32;
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(input.num_faces() as usize * 2);
    for f in 0..input.num_faces() {
        faces.push(input.face_verts(f).to_vec());
    }
    for f in 0..input.num_faces() {
        faces.push(
            input
                .face_verts(f)
                .iter()
                .rev()
                .map(|v| inner(*v))
                .collect(),
        );
    }
    // The rim: every edge on exactly one face, walked the way that face walks
    // it, gets a quad back across to the inner shell.
    let mut uses: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for f in 0..input.num_faces() {
        let ring = input.face_verts(f);
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            *uses.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    for f in 0..input.num_faces() {
        let ring = input.face_verts(f);
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            if uses.get(&(a.min(b), a.max(b))) == Some(&1) {
                faces.push(vec![b, a, inner(a), inner(b)]);
            }
        }
    }
    MeshData::from_faces(out_positions, &faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_plane_becomes_a_closed_slab_of_the_thickness_asked_for() {
        let slab = solidify(
            &primitives::plane(2.0),
            &SolidifyParams {
                thickness: 0.5,
                offset: -1.0,
            },
        );
        assert_eq!(slab.num_verts(), 8);
        assert_eq!(slab.num_faces(), 6, "two shells and four sides");
        let (lo, hi) = (slab.bounds().min, slab.bounds().max);
        assert!(
            (hi.z - 0.0).abs() < 1e-6 && (lo.z + 0.5).abs() < 1e-6,
            "{lo:?} {hi:?}"
        );
        // Closed: every edge on two faces.
        let mut uses = std::collections::HashMap::new();
        for f in 0..slab.num_faces() {
            let ring = slab.face_verts(f);
            for i in 0..ring.len() {
                let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(uses.values().all(|u| *u == 2));
    }

    #[test]
    fn a_closed_mesh_gets_no_rim_and_offset_moves_the_shells() {
        let shell = solidify(
            &primitives::cube(2.0),
            &SolidifyParams {
                thickness: 0.2,
                offset: 1.0,
            },
        );
        assert_eq!(shell.num_faces(), 12);
        assert!(
            shell.bounds().max.x > 1.05,
            "offset 1 did not grow outwards"
        );
    }
}
