//! Limited dissolve: faces that lie in one plane, merged — Blender's
//! X ▸ Limited Dissolve, and the tidying a boolean's result wants.
//!
//! Two faces across an edge merge when their normals are within `angle` of
//! each other and they share a material. Faces are grouped by those edges
//! alone, so a group never reaches across a crease to a face that only
//! touches it, and each group becomes one n-gon. A group whose outline is not
//! one ring — a flat region with a hole — is left as it was, as
//! [`crate::edit::dissolve_faces`] leaves it: no n-gon can hold a hole.

use crate::arena::{ArenaIndex, FaceId};
use crate::attribute::names;
use crate::bmesh::{BMesh, BmError};

/// Merges flat neighbours. Returns how many faces there were less of.
pub fn limited_dissolve(mesh: &mut BMesh, angle: f32) -> Result<usize, BmError> {
    let faces: Vec<FaceId> = mesh.face_ids().collect();
    let slot = |f: FaceId| f.index() as usize;
    let slots = mesh.slot_count(crate::Domain::Face);
    let normals: Vec<_> = {
        let mut n = vec![kreate_math::Vec3::ZERO; slots];
        for f in &faces {
            n[slot(*f)] = mesh.face_normal(*f).normalize_or_zero();
        }
        n
    };
    let material = |f: FaceId| {
        mesh.attributes()
            .get_i32(names::MATERIAL_INDEX)
            .and_then(|m| m.get(slot(f)).copied())
            .unwrap_or(0)
    };
    let cos = angle.cos();

    // Union-find over the faces, joined only across flat edges.
    let mut parent: Vec<usize> = (0..slots).collect();
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut joined = false;
    for e in mesh.edge_ids().collect::<Vec<_>>() {
        let [a, b] = mesh.edge_faces(e)[..] else {
            continue;
        };
        let (na, nb) = (normals[slot(a)], normals[slot(b)]);
        if na == kreate_math::Vec3::ZERO || nb == kreate_math::Vec3::ZERO {
            continue;
        }
        if na.dot(nb) >= cos && material(a) == material(b) {
            let (ra, rb) = (root(&mut parent, slot(a)), root(&mut parent, slot(b)));
            if ra != rb {
                parent[ra] = rb;
                joined = true;
            }
        }
    }
    if !joined {
        return Ok(0);
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<FaceId>> = Default::default();
    for f in &faces {
        groups
            .entry(root(&mut parent, slot(*f)))
            .or_default()
            .push(*f);
    }
    let before = mesh.num_faces();
    for group in groups.into_values().filter(|g| g.len() > 1) {
        crate::edit::dissolve_faces(mesh, &group, &[])?;
    }
    Ok(before - mesh.num_faces())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_subdivided_plane_becomes_one_face_and_a_cube_stays_a_cube() {
        let mut plane = BMesh::from_mesh_data(&primitives::grid(2.0, 4, 4));
        let gone = limited_dissolve(&mut plane, 0.01).unwrap();
        assert_eq!(plane.num_faces(), 1, "{gone}");
        plane.assert_valid();

        let mut cube = BMesh::from_mesh_data(&primitives::cube(2.0));
        assert_eq!(limited_dissolve(&mut cube, 0.01).unwrap(), 0);
        assert_eq!(cube.num_faces(), 6);
    }

    #[test]
    fn a_subdivided_cube_goes_back_to_six_faces() {
        let mut m = primitives::cube(2.0).to_bmesh();
        let all: Vec<FaceId> = m.face_ids().collect();
        crate::edit::subdivide_faces(&mut m, &all).unwrap();
        assert!(m.num_faces() > 6);
        limited_dissolve(&mut m, 0.01).unwrap();
        assert_eq!(m.num_faces(), 6);
        m.assert_valid();
    }
}
