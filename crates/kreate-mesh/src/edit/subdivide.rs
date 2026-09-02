//! Subdivide.
//!
//! Splits every edge of the selected faces at its midpoint and replaces each
//! face with one quad per corner. An n-gon becomes n quads, a quad becomes four,
//! a triangle becomes three — the same rule for every face count, which is what
//! makes it safe to run on a mixed mesh.
//!
//! # The neighbours matter more than the region
//!
//! Splitting an edge shared with an *unselected* face changes that face too: its
//! edge no longer exists. Leaving it alone produces a T-junction — a vertex
//! sitting on the middle of a neighbouring face's edge, belonging to neither
//! side. T-junctions look harmless and then crack open under any deformation,
//! shade wrongly along the seam, and break every later operator that assumes
//! shared edges. So neighbours are rebuilt with the midpoint inserted, turning a
//! quad into a pentagon rather than into a defect.
//!
//! # Cuts
//!
//! One cut per call. More cuts on a quad is a genuinely different construction —
//! an (n+1)² grid, not a repeated halving — and the operator that wants it says
//! so rather than looping this.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, EdgeId, FaceId, VertId};
use crate::bmesh::{BMesh, BmError};
use crate::Domain;

use super::region::Region;

/// What a subdivision produced.
#[derive(Clone, Debug, Default)]
pub struct Subdivision {
    /// The faces that replaced the region.
    pub faces: Vec<FaceId>,
    /// One per split edge.
    pub midpoints: Vec<VertId>,
    /// One per subdivided face.
    pub centres: Vec<VertId>,
    /// Neighbouring faces rebuilt to take a midpoint they did not ask for.
    pub patched_neighbours: usize,
}

/// Splits every edge of `faces` and replaces each face with one quad per corner.
pub fn subdivide_faces(mesh: &mut BMesh, faces: &[FaceId]) -> Result<Subdivision, BmError> {
    let region = Region::new(mesh, faces);
    if region.is_empty() {
        return Ok(Subdivision::default());
    }

    // --- read ---------------------------------------------------------------
    // Rings and their edges, per face, in winding order. Read in full first: the
    // loop cycles stop existing the moment the old faces are removed, and every
    // one of them is needed to place the new quads.
    let mut rings: Vec<(Vec<VertId>, Vec<EdgeId>)> = Vec::with_capacity(region.faces.len());
    let mut to_split: Vec<EdgeId> = Vec::new();
    for &f in &region.faces {
        let loops = mesh.face_loops(f);
        let mut verts = Vec::with_capacity(loops.len());
        let mut edges = Vec::with_capacity(loops.len());
        for l in loops {
            let (Some(v), Some(e)) = (mesh.loop_vert(l), mesh.loop_edge(l)) else {
                continue;
            };
            verts.push(v);
            edges.push(e);
            to_split.push(e);
        }
        rings.push((verts, edges));
    }
    to_split.sort_unstable_by_key(|e| e.index());
    to_split.dedup();

    // Faces outside the region that share a split edge, and will have to be
    // rebuilt around the new midpoint.
    let mut neighbours: Vec<FaceId> = Vec::new();
    for &e in &to_split {
        for g in mesh.edge_faces(e) {
            if !region.contains(g) && !neighbours.contains(&g) {
                neighbours.push(g);
            }
        }
    }
    neighbours.sort_unstable_by_key(|f| f.index());
    let neighbour_rings: Vec<Vec<(VertId, EdgeId)>> = neighbours
        .iter()
        .map(|&f| {
            mesh.face_loops(f)
                .into_iter()
                .filter_map(|l| Some((mesh.loop_vert(l)?, mesh.loop_edge(l)?)))
                .collect()
        })
        .collect();

    // --- write --------------------------------------------------------------
    let mut mid_of = vec![None; mesh.slot_count(Domain::Edge)];
    let mut out = Subdivision::default();
    for &e in &to_split {
        let Some([a, b]) = mesh.edge_verts(e) else {
            continue;
        };
        let (Some(pa), Some(pb)) = (mesh.position(a), mesh.position(b)) else {
            continue;
        };
        let m = mesh.add_vert((pa + pb) * 0.5);
        mid_of[e.index() as usize] = Some(m);
        out.midpoints.push(m);
    }
    let midpoint = |e: EdgeId, table: &[Option<VertId>]| -> Option<VertId> {
        table.get(e.index() as usize).copied().flatten()
    };

    for (verts, edges) in &rings {
        let n = verts.len();
        if n < 3 {
            continue;
        }
        let centre_pos: Vec3 =
            verts.iter().filter_map(|v| mesh.position(*v)).sum::<Vec3>() / n as f32;
        let centre = mesh.add_vert(centre_pos);
        out.centres.push(centre);

        for i in 0..n {
            let prev = (i + n - 1) % n;
            let (Some(before), Some(after)) =
                (midpoint(edges[prev], &mid_of), midpoint(edges[i], &mid_of))
            else {
                continue;
            };
            // `before → corner → after → centre` keeps the parent face's
            // winding: `edges[prev]` runs into the corner and `edges[i]` runs
            // out of it.
            out.faces
                .push(mesh.add_face(&[before, verts[i], after, centre])?);
        }
    }

    // Neighbours: the same ring, with each split edge's midpoint spliced in
    // after the vertex the edge leaves.
    let mut rebuilt_neighbours = Vec::with_capacity(neighbours.len());
    for ring in &neighbour_rings {
        let mut patched = Vec::with_capacity(ring.len() * 2);
        for &(v, e) in ring {
            patched.push(v);
            if let Some(m) = midpoint(e, &mid_of) {
                patched.push(m);
            }
        }
        rebuilt_neighbours.push(patched);
    }

    for &f in &region.faces {
        mesh.remove_face(f);
    }
    for &f in &neighbours {
        mesh.remove_face(f);
    }
    // The originals are gone, so their edges are wires now; clearing them before
    // rebuilding keeps the neighbour's new ring from finding a stale edge
    // between two vertices that are no longer adjacent.
    for &e in &to_split {
        if mesh.edge(e).is_some() && mesh.is_wire_edge(e) {
            mesh.remove_edge(e);
        }
    }
    for ring in rebuilt_neighbours {
        if ring.len() >= 3 {
            mesh.add_face(&ring)?;
            out.patched_neighbours += 1;
        }
    }

    mesh.select_none();
    for f in &out.faces {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(crate::SelectMode::Face);
    Ok(out)
}

/// Subdivides the selected faces.
pub fn subdivide_selected(mesh: &mut BMesh) -> Result<Subdivision, BmError> {
    let faces = mesh.selected_faces();
    subdivide_faces(mesh, &faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_quad_becomes_four_quads() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let f = m.face_ids().next().unwrap();
        let out = subdivide_faces(&mut m, &[f]).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 4);
        assert_eq!(out.midpoints.len(), 4);
        assert_eq!(out.centres.len(), 1);
        assert_eq!(m.num_faces(), 4);
        assert_eq!(m.num_verts(), 9, "4 corners, 4 midpoints, 1 centre");
        assert_eq!(m.num_edges(), 12);
        for f in m.face_ids() {
            assert_eq!(m.face(f).unwrap().len(), 4);
        }
    }

    #[test]
    fn a_triangle_becomes_three_quads() {
        // The same rule for every face count, which is what makes it safe on a
        // mixed mesh. A triangle-specific pattern is where subdivision code
        // usually starts and where it usually goes wrong.
        let mut m = BMesh::new();
        let a = m.add_vert(Vec3::ZERO);
        let b = m.add_vert(Vec3::X);
        let c = m.add_vert(Vec3::Y);
        let f = m.add_face(&[a, b, c]).unwrap();

        let out = subdivide_faces(&mut m, &[f]).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 3);
        for f in m.face_ids() {
            assert_eq!(m.face(f).unwrap().len(), 4, "quads, not smaller triangles");
        }
        assert_eq!(m.num_verts(), 7);
    }

    #[test]
    fn a_cube_subdivides_into_twenty_four_quads_and_stays_closed() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let all: Vec<FaceId> = m.face_ids().collect();
        let out = subdivide_faces(&mut m, &all).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 24);
        assert_eq!(m.num_faces(), 24);
        assert_eq!(
            m.num_verts(),
            8 + 12 + 6,
            "corners, edge midpoints, centres"
        );
        assert_eq!(m.num_edges(), 48);
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "still watertight");
        }
        assert_eq!(
            m.num_verts() as i64 - m.num_edges() as i64 + m.num_faces() as i64,
            2
        );
    }

    #[test]
    fn a_split_edge_is_shared_rather_than_duplicated() {
        // Two faces subdivided together must meet along the same midpoint. A
        // per-face midpoint would split the mesh in half along the seam, which
        // looks identical until something moves.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 1));
        assert_eq!(m.num_faces(), 2);
        let all: Vec<FaceId> = m.face_ids().collect();
        subdivide_faces(&mut m, &all).unwrap();
        m.assert_valid();

        let shared = m.edge_ids().filter(|e| m.edge_face_count(*e) == 2).count();
        assert!(shared > 0, "the two halves are still joined");
        assert_eq!(m.num_faces(), 8);
        // 3x2 original grid vertices, 7 midpoints, 2 centres.
        assert_eq!(m.num_verts(), 6 + 7 + 2);
    }

    #[test]
    fn an_unselected_neighbour_gains_the_midpoint_instead_of_a_t_junction() {
        // The defect this guards is the one that does not show up until later: a
        // vertex sitting on the middle of a neighbour's edge, belonging to
        // neither face. It renders correctly, and then cracks open under any
        // deformation.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 1));
        let faces: Vec<FaceId> = m.face_ids().collect();
        let out = subdivide_faces(&mut m, &[faces[0]]).unwrap();
        m.assert_valid();

        assert_eq!(out.patched_neighbours, 1);
        assert_eq!(m.num_faces(), 5, "four new quads and the rebuilt neighbour");

        // Every vertex belongs to every face whose edge it lies on: no vertex
        // has only one face on a shared edge.
        let neighbour = m
            .face_ids()
            .find(|f| m.face(*f).unwrap().len() == 5)
            .expect("the neighbour became a pentagon");
        assert_eq!(m.face_verts(neighbour).len(), 5);
        for e in m.edge_ids() {
            assert!(
                m.edge_face_count(e) <= 2,
                "the patch must not create a non-manifold edge"
            );
        }
    }

    #[test]
    fn subdividing_twice_quadruples_again() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        m.select_all();
        subdivide_selected(&mut m).unwrap();
        m.assert_valid();
        assert_eq!(m.num_faces(), 4);

        subdivide_selected(&mut m).unwrap();
        m.assert_valid();
        assert_eq!(m.num_faces(), 16, "each iteration halves every edge");
        for f in m.face_ids() {
            assert_eq!(m.face(f).unwrap().len(), 4);
        }
    }

    #[test]
    fn the_new_faces_are_what_ends_up_selected() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        m.select_all();
        let out = subdivide_selected(&mut m).unwrap();
        assert_eq!(m.selected_faces(), out.faces);
        assert_eq!(m.selected_verts().len(), 9, "every vertex of the new patch");
    }

    #[test]
    fn midpoints_land_on_the_edges_they_split() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        let ring: Vec<Vec3> = m
            .face_verts(f)
            .iter()
            .filter_map(|v| m.position(*v))
            .collect();
        let out = subdivide_faces(&mut m, &[f]).unwrap();

        for mid in &out.midpoints {
            let p = m.position(*mid).unwrap();
            let on_an_edge = (0..ring.len()).any(|i| {
                let a = ring[i];
                let b = ring[(i + 1) % ring.len()];
                ((a + b) * 0.5 - p).length() < 1e-5
            });
            assert!(on_an_edge, "midpoint {p:?} is not on any edge of the face");
        }
        let centre = m.position(out.centres[0]).unwrap();
        let want: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
        assert!((centre - want).length() < 1e-5);
    }

    #[test]
    fn the_new_faces_keep_the_parent_winding() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m
            .face_ids()
            .max_by(|a, b| {
                let za = m.face_normal(*a).normalize_or_zero().z;
                let zb = m.face_normal(*b).normalize_or_zero().z;
                za.partial_cmp(&zb).unwrap()
            })
            .unwrap();
        let before = m.face_normal(f).normalize_or_zero();
        let out = subdivide_faces(&mut m, &[f]).unwrap();

        for g in &out.faces {
            let n = m.face_normal(*g).normalize_or_zero();
            assert!(
                n.dot(before) > 0.99,
                "child normal {n:?} disagrees with the parent's {before:?}"
            );
        }
    }

    #[test]
    fn subdividing_nothing_is_a_no_op() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        let out = subdivide_faces(&mut m, &[]).unwrap();
        assert!(out.faces.is_empty());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
        m.assert_valid();
    }

    #[test]
    fn a_subdivided_mesh_still_bakes_to_storage_form() {
        let mut m = BMesh::from_mesh_data(&primitives::uv_sphere(1.0, 8, 4));
        m.select_all();
        subdivide_selected(&mut m).unwrap();
        m.assert_valid();

        let baked = m.to_mesh_data();
        assert_eq!(baked.validate(), Vec::new());
        assert_eq!(baked.num_faces() as usize, m.num_faces());
        let again = BMesh::from_mesh_data(&baked);
        again.assert_valid();
        assert_eq!(again.selected_faces().len(), m.selected_faces().len());
    }
}
