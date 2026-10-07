//! Is this mesh fit to be made? The checks a 3D-print toolbox runs before a
//! part goes to a slicer.
//!
//! Each finds the elements at fault rather than a yes or no, so the editor
//! can select them and the user can see where: an open edge is a hole, a
//! non-manifold edge a fin or a seam three faces share, a flipped edge two
//! faces wound against each other, a degenerate face one with no area, a
//! loose vertex one no face uses, an intersecting face one that passes
//! through another, a thin face one whose wall is thinner than asked.
//!
//! In `f64` space given by the caller — the world, so a thickness in
//! millimetres means millimetres of the part as placed.

use kreate_math::{Aabb, Bvh, DVec3, Ray, Vec3};

use crate::MeshData;

/// What a check found, as element indices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Edges with one face: the rims of holes.
    pub open_edges: Vec<u32>,
    /// Edges with three or more faces.
    pub non_manifold_edges: Vec<u32>,
    /// Edges whose two faces run along them the same way: one is wound
    /// inside out against the other.
    pub flipped_edges: Vec<u32>,
    /// Faces with (next to) no area.
    pub degenerate_faces: Vec<u32>,
    /// Vertices no face uses.
    pub loose_verts: Vec<u32>,
    /// Faces passing through another face that is not their neighbour.
    pub intersecting_faces: Vec<u32>,
    /// Faces whose wall, measured straight in, is thinner than asked.
    pub thin_faces: Vec<u32>,
}

impl Report {
    /// Nothing at fault.
    pub fn is_clean(&self) -> bool {
        *self == Self::default()
    }
}

/// Runs the checks. `thickness` is the thinnest wall allowed, in the same
/// units as `to_space` places the points; zero skips that check, and
/// `intersections` false skips the costliest one.
pub fn check(
    mesh: &MeshData,
    to_space: impl Fn(DVec3) -> DVec3,
    thickness: f64,
    intersections: bool,
) -> Report {
    let points: Vec<DVec3> = mesh
        .positions()
        .iter()
        .map(|p| to_space(p.as_dvec3()))
        .collect();
    let mut report = Report::default();

    // How each edge is used: by how many faces, and in which direction.
    let edges = mesh.edges();
    let corner_edges = mesh.corner_edges();
    let corner_verts = mesh.corner_verts();
    let offsets = mesh.face_offsets();
    let mut uses = vec![0u32; edges.len()];
    let mut forward = vec![0u32; edges.len()];
    let mut used_vert = vec![false; points.len()];
    for f in 0..mesh.num_faces() as usize {
        let (start, end) = (offsets[f] as usize, offsets[f + 1] as usize);
        for c in start..end {
            let e = corner_edges[c] as usize;
            let a = corner_verts[c];
            used_vert[a as usize] = true;
            uses[e] += 1;
            // The face runs a -> next; the edge is stored [x, y].
            if edges[e][0] == a {
                forward[e] += 1;
            }
        }
    }
    for (e, n) in uses.iter().enumerate() {
        match n {
            1 => report.open_edges.push(e as u32),
            2 if forward[e] != 1 => report.flipped_edges.push(e as u32),
            n if *n > 2 => report.non_manifold_edges.push(e as u32),
            _ => {}
        }
    }
    report.loose_verts = (0..points.len() as u32)
        .filter(|v| !used_vert[*v as usize])
        .collect();

    // Scale for "no area": relative to the part, so a millimetre part and a
    // building are judged alike.
    let (lo, hi) = points.iter().fold(
        (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let size = if points.is_empty() { 1.0 } else { (hi - lo).length().max(1e-12) };
    let tiny_area = (size * 1e-7) * (size * 1e-7);

    // Triangles, for the area, the intersections and the walls. Fan per face.
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut tri_face: Vec<u32> = Vec::new();
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        let mut area = DVec3::ZERO;
        for i in 0..ring.len() {
            area += points[ring[i] as usize].cross(points[ring[(i + 1) % ring.len()] as usize]);
        }
        if ring.len() < 3 || area.length() * 0.5 <= tiny_area {
            report.degenerate_faces.push(f);
            continue;
        }
        for i in 1..ring.len() - 1 {
            tris.push([ring[0], ring[i], ring[i + 1]]);
            tri_face.push(f);
        }
    }
    if tris.is_empty() {
        return report;
    }
    // The tree works in f32 about the part's own middle, which keeps the
    // precision where the part is.
    let middle = (lo + hi) * 0.5;
    let local: Vec<Vec3> = points.iter().map(|p| (*p - middle).as_vec3()).collect();
    let bvh = Bvh::build(&local, &tris);

    // Faces passing through each other.
    let mut crossing = vec![false; mesh.num_faces() as usize];
    let mut near = Vec::new();
    for (i, t) in tris.iter().enumerate().filter(|_| intersections) {
        let corners = t.map(|v| local[v as usize]);
        let bounds = Aabb::from_points(corners).expanded(size as f32 * 1e-6);
        near.clear();
        bvh.overlaps(&bounds, &mut near);
        for &j in near.iter() {
            let j = j as usize;
            if j <= i || tri_face[i] == tri_face[j] {
                continue;
            }
            let other = tris[j];
            // Neighbours meet along an edge or at a corner; that is touching,
            // not passing through.
            if t.iter().any(|v| other.contains(v)) {
                continue;
            }
            if triangles_cross(corners, other.map(|v| local[v as usize])) {
                crossing[tri_face[i] as usize] = true;
                crossing[tri_face[j] as usize] = true;
            }
        }
    }
    report.intersecting_faces = (0..crossing.len() as u32)
        .filter(|f| crossing[*f as usize])
        .collect();

    // Walls: straight in from each face's middle, against its normal, to the
    // next surface. Closer than asked is thin.
    if thickness > 0.0 {
        let mut thin = vec![false; mesh.num_faces() as usize];
        for f in 0..mesh.num_faces() {
            if thin[f as usize] || report.degenerate_faces.binary_search(&f).is_ok() {
                continue;
            }
            let ring = mesh.face_verts(f);
            let centre = ring.iter().map(|v| points[*v as usize]).sum::<DVec3>() / ring.len() as f64;
            let mut normal = DVec3::ZERO;
            for i in 0..ring.len() {
                normal += points[ring[i] as usize].cross(points[ring[(i + 1) % ring.len()] as usize]);
            }
            let normal = normal.normalize_or_zero();
            if normal == DVec3::ZERO {
                continue;
            }
            // Started a hair inside, so the face itself is not the answer.
            let start = centre - normal * (size * 1e-6) - middle;
            let ray = Ray::new(start.as_vec3(), (-normal).as_vec3());
            if let Some(hit) = bvh.raycast(&local, &tris, &ray, thickness as f32) {
                if tri_face[hit.primitive as usize] != f && (hit.t as f64) < thickness {
                    thin[f as usize] = true;
                }
            }
        }
        report.thin_faces = (0..thin.len() as u32).filter(|f| thin[*f as usize]).collect();
    }
    report
}

/// Whether two triangles pass through each other: an edge of one crosses the
/// other. Touching at a point counts; lying in one plane does not.
fn triangles_cross(a: [Vec3; 3], b: [Vec3; 3]) -> bool {
    let edge_hits = |t: [Vec3; 3], o: [Vec3; 3]| {
        (0..3).any(|i| {
            let (p, q) = (t[i], t[(i + 1) % 3]);
            let d = q - p;
            let len = d.length();
            len > 0.0
                && Ray::new(p, d / len)
                    .intersect_triangle(o[0], o[1], o[2])
                    .is_some_and(|(s, _, _)| s <= len)
        })
    };
    edge_hits(a, b) || edge_hits(b, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn same(p: DVec3) -> DVec3 {
        p
    }

    #[test]
    fn a_closed_cube_is_clean() {
        let r = check(&primitives::cube(2.0), same, 0.1, true);
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn a_hole_is_found_by_its_rim() {
        let r = check(&primitives::plane(2.0), same, 0.0, true);
        assert_eq!(r.open_edges.len(), 4);
        assert!(r.non_manifold_edges.is_empty() && r.flipped_edges.is_empty());
    }

    #[test]
    fn a_face_turned_inside_out_shows_at_its_edges() {
        let mut m = primitives::cube(2.0);
        let mut bm = m.to_bmesh();
        let f = bm.face_ids().next().unwrap();
        crate::edit::flip(&mut bm, &[f]);
        m = bm.to_mesh_data();
        let r = check(&m, same, 0.0, true);
        assert_eq!(r.flipped_edges.len(), 4, "{r:?}");
    }

    #[test]
    fn two_cubes_through_each_other_intersect_and_a_thin_slab_is_thin() {
        let a = primitives::cube(2.0);
        let b = primitives::cube(2.0);
        let (joined, _) = crate::edit::join(&[
            crate::edit::Part {
                mesh: &a,
                transform: kreate_math::DAffine3::IDENTITY,
            },
            crate::edit::Part {
                mesh: &b,
                transform: kreate_math::DAffine3::from_translation(DVec3::new(1.0, 0.5, 0.25)),
            },
        ]);
        let r = check(&joined, same, 0.0, true);
        assert!(!r.intersecting_faces.is_empty());

        let mut slab = primitives::cube(2.0);
        for p in slab.positions_mut() {
            p.z *= 0.01; // 2 cm thick, 2 m wide
        }
        let r = check(&slab, same, 0.05, true);
        assert!(r.thin_faces.len() >= 2, "{r:?}");
        let r = check(&slab, same, 0.01, true);
        assert!(r.thin_faces.is_empty(), "thicker than asked is not thin: {r:?}");
    }

    #[test]
    fn loose_vertices_and_flat_faces_are_found() {
        let mut m = primitives::plane(2.0);
        let mut bm = m.to_bmesh();
        bm.add_vert(Vec3::new(5.0, 5.0, 5.0));
        let v = bm.add_vert(Vec3::ZERO);
        let w = bm.add_vert(Vec3::ZERO);
        let x = bm.add_vert(Vec3::ZERO);
        bm.add_face(&[v, w, x]).unwrap();
        m = bm.to_mesh_data();
        let r = check(&m, same, 0.0, true);
        assert_eq!(r.loose_verts.len(), 1);
        assert_eq!(r.degenerate_faces.len(), 1);
    }
}
