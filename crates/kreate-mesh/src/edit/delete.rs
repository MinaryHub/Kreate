//! Delete and dissolve.
//!
//! Two operations users constantly confuse, so the distinction is worth stating
//! precisely: **delete** removes geometry and leaves a hole; **dissolve**
//! removes geometry and heals what is left into a single larger face. Deleting
//! an edge of a cube opens it; dissolving that edge merges the two faces into a
//! hexagon and the cube stays closed.
//!
//! Dissolve is where non-manifold tolerance earns its keep. Merging a region
//! into one n-gon goes through an intermediate state with the old faces gone and
//! the new one not yet built, and the outline itself can be a shape no
//! half-edge structure would accept.

use crate::arena::{ArenaIndex, EdgeId, FaceId, VertId};
use crate::bmesh::{BMesh, BmError};
use crate::Domain;

use super::region::Region;

/// What a delete should take with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DeleteMode {
    /// Vertices, plus every edge and face touching them.
    Verts,
    /// Edges, plus the faces on them. Vertices stay.
    Edges,
    /// Faces, plus the edges and vertices nothing else uses.
    #[default]
    Faces,
    /// Faces only — the outline stays behind as loose edges.
    ///
    /// Blender's "Only Faces". Used to punch a hole while keeping the border to
    /// build from.
    FacesOnly,
}

/// Removes selected geometry.
///
/// Returns how many faces, edges and vertices went.
pub fn delete(mesh: &mut BMesh, mode: DeleteMode) -> (usize, usize, usize) {
    let before = (mesh.num_faces(), mesh.num_edges(), mesh.num_verts());
    match mode {
        DeleteMode::Verts => {
            for v in mesh.selected_verts() {
                mesh.remove_vert(v);
            }
        }
        DeleteMode::Edges => {
            for e in mesh.selected_edges() {
                mesh.remove_edge(e);
            }
        }
        DeleteMode::Faces => {
            let faces = mesh.selected_faces();
            let touched: Vec<VertId> = Region::new(mesh, &faces).verts(mesh);
            for f in faces {
                mesh.remove_face(f);
            }
            // Only what the deleted faces alone were holding up. A blanket
            // `remove_loose` would also take loose geometry the user built
            // deliberately somewhere else in the mesh.
            for v in touched {
                if mesh.vert_faces(v).is_empty() {
                    mesh.remove_vert(v);
                }
            }
        }
        DeleteMode::FacesOnly => {
            for f in mesh.selected_faces() {
                mesh.remove_face(f);
            }
        }
    }
    mesh.select_none();
    (
        before.0 - mesh.num_faces(),
        before.1 - mesh.num_edges(),
        before.2 - mesh.num_verts(),
    )
}

/// What a dissolve produced.
#[derive(Clone, Debug, Default)]
pub struct Dissolution {
    /// The merged faces, one per connected group that could be healed.
    pub faces: Vec<FaceId>,
    /// Groups left untouched because their outline could not be chained into a
    /// single ring — a region with a hole, or one pinched at a vertex.
    pub skipped: usize,
}

/// Merges each connected group of faces into one n-gon.
///
/// `exclude` names vertices to leave out of the result, which is what makes
/// dissolving a *vertex* the same operation as dissolving the faces around it.
///
/// A group whose outline is not a single ring is skipped rather than guessed at:
/// a region with a hole has two rings and no n-gon can represent it, and one
/// pinched at a vertex has an outline that can be chained two ways. Producing
/// something in those cases means producing a face turned inside out through the
/// middle of the model, which the user finds much later and cannot attribute.
pub fn dissolve_faces(
    mesh: &mut BMesh,
    faces: &[FaceId],
    exclude: &[VertId],
) -> Result<Dissolution, BmError> {
    let region = Region::new(mesh, faces);
    if region.is_empty() {
        return Ok(Dissolution::default());
    }
    let mut excluded = vec![false; mesh.slot_count(Domain::Point)];
    for v in exclude {
        if let Some(slot) = excluded.get_mut(v.index() as usize) {
            *slot = true;
        }
    }

    let mut out = Dissolution::default();
    for group in region.components(mesh) {
        let part = Region::new(mesh, &group);
        let interior = part.verts(mesh);
        // The group's edges, kept so the ones the merge makes redundant can be
        // cleared afterwards. Without this the dissolved edge survives as a wire
        // lying inside the new face — invisible in the solid view, and quietly
        // wrong to every later operator that walks the topology.
        let mut group_edges: Vec<EdgeId> = group
            .iter()
            .flat_map(|f| mesh.face_loops(*f))
            .filter_map(|l| mesh.loop_edge(l))
            .collect();
        group_edges.sort_unstable_by_key(|e| e.index());
        group_edges.dedup();

        let Some(loops) = part.boundary_loops(mesh) else {
            out.skipped += 1;
            continue;
        };
        if loops.len() != 1 {
            out.skipped += 1;
            continue;
        }
        let ring: Vec<VertId> = loops[0]
            .iter()
            .copied()
            .filter(|v| !excluded[v.index() as usize])
            .collect();
        if ring.len() < 3 {
            out.skipped += 1;
            continue;
        }

        for f in &group {
            mesh.remove_face(*f);
        }
        out.faces.push(mesh.add_face(&ring)?);

        // Everything the group alone was holding. Edges first: a vertex is only
        // loose once the wires hanging off it are gone.
        //
        // Tested for wireness rather than assumed: an edge interior to the
        // region can still carry a third face on non-manifold geometry, and
        // removing that would take a face the user never selected.
        for e in group_edges {
            if mesh.edge(e).is_some() && mesh.is_wire_edge(e) {
                mesh.remove_edge(e);
            }
        }
        for v in interior {
            if mesh.vert(v).is_some() && mesh.vert_faces(v).is_empty() {
                mesh.remove_vert(v);
            }
        }
    }

    mesh.select_none();
    for f in &out.faces {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(crate::SelectMode::Face);
    Ok(out)
}

/// Removes edges by merging the faces that share them.
///
/// The faces on every named edge form the region, so dissolving two edges of one
/// face merges all three faces into a single n-gon — which is what the user
/// asked for, rather than two separate merges that fight over the middle.
///
/// Edges with fewer than two faces are ignored: there is nothing on the other
/// side to merge with, and removing them would be a delete, not a dissolve.
pub fn dissolve_edges(mesh: &mut BMesh, edges: &[EdgeId]) -> Result<Dissolution, BmError> {
    let mut faces: Vec<FaceId> = Vec::new();
    for &e in edges {
        if mesh.edge_face_count(e) < 2 {
            continue;
        }
        for f in mesh.edge_faces(e) {
            if !faces.contains(&f) {
                faces.push(f);
            }
        }
    }
    dissolve_faces(mesh, &faces, &[])
}

/// Removes vertices by merging the faces around them.
///
/// The same operation as dissolving those faces, with the vertices left out of
/// the result — which handles the interior case (the vertex vanishes into the
/// merged face) and the corner case (the vertex is dropped from the ring) with
/// one rule instead of two.
pub fn dissolve_verts(mesh: &mut BMesh, verts: &[VertId]) -> Result<Dissolution, BmError> {
    let mut faces: Vec<FaceId> = Vec::new();
    for &v in verts {
        for f in mesh.vert_faces(v) {
            if !faces.contains(&f) {
                faces.push(f);
            }
        }
    }
    let out = dissolve_faces(mesh, &faces, verts)?;
    // A vertex only used by wire edges after the merge is left over from the
    // dissolve rather than something the user built.
    for &v in verts {
        if mesh.vert(v).is_some() && mesh.vert_faces(v).is_empty() {
            mesh.remove_vert(v);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;
    use crate::SelectMode;

    fn cube() -> BMesh {
        BMesh::from_mesh_data(&primitives::cube(2.0))
    }

    fn select_face(mesh: &mut BMesh, f: FaceId) {
        mesh.select_none();
        mesh.set_face_selected(f, true);
        mesh.flush_selection(SelectMode::Face);
    }

    // --- delete -------------------------------------------------------------

    #[test]
    fn deleting_a_face_leaves_a_hole_with_its_border_intact() {
        let mut m = cube();
        let f = m.face_ids().next().unwrap();
        select_face(&mut m, f);
        let (faces, edges, verts) = delete(&mut m, DeleteMode::Faces);
        m.assert_valid();

        assert_eq!((faces, edges, verts), (1, 0, 0), "only the face goes");
        assert_eq!(m.num_faces(), 5);
        assert_eq!(
            m.num_verts(),
            8,
            "every corner is still held by other faces"
        );
        assert_eq!(m.num_edges(), 12);
    }

    #[test]
    fn deleting_a_vertex_takes_everything_touching_it() {
        let mut m = cube();
        let v = m.vert_ids().next().unwrap();
        m.select_none();
        m.set_vert_selected(v, true);
        let (faces, edges, verts) = delete(&mut m, DeleteMode::Verts);
        m.assert_valid();

        assert_eq!(verts, 1);
        assert_eq!(faces, 3, "a cube corner belongs to three faces");
        assert_eq!(edges, 3);
        assert_eq!(m.num_verts(), 7);
    }

    #[test]
    fn deleting_only_faces_keeps_the_outline_as_wire() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        m.select_all();
        delete(&mut m, DeleteMode::FacesOnly);
        m.assert_valid();

        assert_eq!(m.num_faces(), 0);
        assert_eq!(m.num_edges(), 4, "the border survives to build from");
        assert_eq!(m.num_verts(), 4);
        assert!(m.edge_ids().all(|e| m.is_wire_edge(e)));
    }

    #[test]
    fn deleting_faces_reclaims_only_what_they_alone_held() {
        // The middle face of a 3x3 grid is surrounded, so nothing is reclaimed.
        // A corner face holds one vertex on its own.
        let mut m = BMesh::from_mesh_data(&primitives::grid(6.0, 3, 3));
        let corner = m
            .face_ids()
            .find(|f| {
                m.face_loops(*f)
                    .iter()
                    .filter(|l| {
                        m.loop_edge(**l)
                            .map(|e| m.edge_face_count(e) == 1)
                            .unwrap_or(false)
                    })
                    .count()
                    == 2
            })
            .expect("a grid has corner faces");
        select_face(&mut m, corner);
        let (_, _, verts) = delete(&mut m, DeleteMode::Faces);
        m.assert_valid();
        assert_eq!(
            verts, 1,
            "only the outer corner was held by this face alone"
        );
        assert_eq!(m.num_faces(), 8);
    }

    // --- dissolve -----------------------------------------------------------

    #[test]
    fn dissolving_an_edge_of_a_cube_merges_two_faces_and_keeps_it_closed() {
        // The headline difference from delete: the solid stays watertight.
        let mut m = cube();
        let e = m.edge_ids().next().unwrap();
        let out = dissolve_edges(&mut m, &[e]).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 1);
        assert_eq!(out.skipped, 0);
        assert_eq!(m.num_faces(), 5);
        assert_eq!(m.num_edges(), 11);
        assert_eq!(m.num_verts(), 8);
        assert_eq!(
            m.face(out.faces[0]).unwrap().len(),
            6,
            "two quads sharing an edge merge into a hexagon"
        );
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "still a closed solid");
        }
    }

    #[test]
    fn dissolving_two_edges_of_one_face_merges_all_three() {
        // Dissolving them one at a time would give a different, worse answer:
        // the second merge would act on a face the first had already replaced.
        let mut m = cube();
        let f = m.face_ids().next().unwrap();
        let loops = m.face_loops(f);
        let a = m.loop_edge(loops[0]).unwrap();
        let b = m.loop_edge(loops[2]).unwrap();

        let out = dissolve_edges(&mut m, &[a, b]).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 1);
        assert_eq!(m.num_faces(), 4, "three faces became one");
        assert_eq!(m.face(out.faces[0]).unwrap().len(), 8);
    }

    #[test]
    fn dissolving_a_boundary_edge_is_ignored() {
        // Nothing on the other side to merge with. Removing it would be a
        // delete, and quietly turning one into the other is how a tool loses a
        // user's trust.
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let e = m.edge_ids().next().unwrap();
        let out = dissolve_edges(&mut m, &[e]).unwrap();
        assert!(out.faces.is_empty());
        assert_eq!(m.num_faces(), 1);
        assert_eq!(m.num_edges(), 4);
        m.assert_valid();
    }

    #[test]
    fn dissolving_an_interior_vertex_removes_it_and_heals_the_hole() {
        // The centre of a 3x3 grid: four faces meet there and become one quad.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        let centre = m
            .vert_ids()
            .find(|v| m.vert_edge_count(*v) == 4)
            .expect("a 2x2 grid has one interior vertex");

        let out = dissolve_verts(&mut m, &[centre]).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 1);
        assert_eq!(m.num_faces(), 1);
        assert_eq!(m.face(out.faces[0]).unwrap().len(), 8, "the grid's outline");
        assert!(m.vert(centre).is_none(), "the dissolved vertex is gone");
        assert_eq!(m.num_verts(), 8);
    }

    #[test]
    fn dissolving_a_corner_vertex_drops_it_from_the_face() {
        // The other half of the same rule: the vertex is on the outline, so it
        // is left out of the merged ring rather than vanishing into it.
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let corner = m.vert_ids().next().unwrap();
        let out = dissolve_verts(&mut m, &[corner]).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 1);
        assert_eq!(
            m.face(out.faces[0]).unwrap().len(),
            3,
            "a quad minus a corner is a triangle"
        );
        assert!(m.vert(corner).is_none());
        assert_eq!(m.num_verts(), 3);
    }

    #[test]
    fn dissolving_a_region_with_a_hole_is_refused_not_guessed() {
        // The ring of eight faces around the centre of a 3x3 grid has two
        // boundary cycles. No n-gon can represent that, and healing it would
        // weld the hole shut without being asked.
        let m0 = BMesh::from_mesh_data(&primitives::grid(6.0, 3, 3));
        let middle = m0
            .face_ids()
            .find(|f| {
                m0.face_loops(*f).iter().all(|l| {
                    m0.loop_edge(*l)
                        .map(|e| m0.edge_face_count(e) == 2)
                        .unwrap_or(false)
                })
            })
            .unwrap();
        let ring: Vec<FaceId> = m0.face_ids().filter(|f| *f != middle).collect();

        let mut m = m0.clone();
        let out = dissolve_faces(&mut m, &ring, &[]).unwrap();
        m.assert_valid();

        assert_eq!(out.skipped, 1);
        assert!(out.faces.is_empty());
        assert_eq!(m.num_faces(), 9, "nothing was changed");
    }

    #[test]
    fn two_separate_patches_dissolve_into_two_faces() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(8.0, 4, 4));
        let faces: Vec<FaceId> = m.face_ids().collect();
        // Two 2x2 blocks at opposite corners of a 4x4 grid.
        let left = [faces[0], faces[1], faces[4], faces[5]];
        let right = [faces[10], faces[11], faces[14], faces[15]];
        let both: Vec<FaceId> = left.iter().chain(&right).copied().collect();

        let out = dissolve_faces(&mut m, &both, &[]).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 2, "two patches, two n-gons");
        assert_eq!(out.skipped, 0);
        assert_eq!(m.num_faces(), 16 - 8 + 2);
        for f in &out.faces {
            assert_eq!(m.face(*f).unwrap().len(), 8);
        }
    }

    #[test]
    fn a_dissolved_face_keeps_the_winding_it_came_from() {
        // A merged face wound the other way is lit from inside and reads as a
        // shading bug. The outline is built from the region's own loop
        // directions, so this holds by construction — and this is the test that
        // says so.
        let mut m = cube();
        let e = m.edge_ids().next().unwrap();
        let originals: Vec<kreate_math::Vec3> = m
            .edge_faces(e)
            .iter()
            .map(|f| m.face_normal(*f).normalize_or_zero())
            .collect();
        let out = dissolve_edges(&mut m, &[e]).unwrap();

        let merged = m.face_normal(out.faces[0]).normalize_or_zero();
        assert!(
            originals.iter().all(|n| merged.dot(*n) > 0.0),
            "merged normal {merged:?} disagrees with its sources {originals:?}"
        );
    }

    #[test]
    fn the_merged_face_ends_up_selected() {
        let mut m = cube();
        let e = m.edge_ids().next().unwrap();
        let out = dissolve_edges(&mut m, &[e]).unwrap();
        assert_eq!(m.selected_faces(), out.faces);
        assert_eq!(m.selected_verts().len(), 6);
    }

    #[test]
    fn dissolving_nothing_is_a_no_op() {
        let mut m = cube();
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        assert!(dissolve_faces(&mut m, &[], &[]).unwrap().faces.is_empty());
        assert!(dissolve_edges(&mut m, &[]).unwrap().faces.is_empty());
        assert!(dissolve_verts(&mut m, &[]).unwrap().faces.is_empty());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
        m.assert_valid();
    }

    #[test]
    fn dissolving_every_edge_of_a_solid_is_refused() {
        // The whole surface is the region, so there is no outline at all and no
        // ring to build from. Must be reported, not attempted.
        let mut m = cube();
        let all: Vec<EdgeId> = m.edge_ids().collect();
        let out = dissolve_edges(&mut m, &all).unwrap();
        m.assert_valid();
        assert_eq!(out.skipped, 1);
        assert!(out.faces.is_empty());
        assert_eq!(m.num_faces(), 6);
    }
}
