//! Extrude.
//!
//! The most-used operator in a polygon modeller, and the one that decides
//! whether the rest of the toolset feels trustworthy.
//!
//! # Region, not per-face
//!
//! Extruding a region of faces produces *one* raised shell with walls only
//! around its outline. Extruding each face separately produces a shell per face
//! with walls between them — a different operation, and the one users complain
//! about when a tool gets this wrong. The whole difference is which edges count
//! as boundary: here, an edge is boundary when exactly one of the faces on it is
//! in the region.
//!
//! # Order of operations
//!
//! Everything is *read* before anything is written. The boundary directions come
//! from the original faces' loop cycles, and those cycles stop existing the
//! moment the faces are removed — so removal happens last, after the new shell
//! is already built.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, FaceId, VertId};
use crate::bmesh::{BMesh, BmError, SelectMode};

/// What an extrusion produced.
#[derive(Clone, Debug, Default)]
pub struct Extrusion {
    /// The raised copies of the original faces. The new selection.
    pub new_faces: Vec<FaceId>,
    /// The walls joining the shell to what it was lifted from.
    pub side_faces: Vec<FaceId>,
    /// The vertices that moved, which is what a transform then drags.
    pub new_verts: Vec<VertId>,
    /// The region's area-weighted normal, before the offset was applied.
    pub normal: Vec3,
}

/// Extrudes a region of faces, offsetting the new shell by `offset`.
///
/// Passing `Vec3::ZERO` gives the topology change alone, which is exactly what
/// an interactive extrude wants: build once on mouse-down, then let the modal
/// transform move `new_verts` every frame.
pub fn extrude_faces(
    mesh: &mut BMesh,
    faces: &[FaceId],
    offset: Vec3,
) -> Result<Extrusion, BmError> {
    let faces: Vec<FaceId> = faces
        .iter()
        .copied()
        .filter(|f| mesh.face(*f).is_some())
        .collect();
    if faces.is_empty() {
        return Ok(Extrusion::default());
    }

    // --- read ---------------------------------------------------------------
    let normal = mesh.region_normal(&faces);

    // Membership by slot: a linear `contains` over the region would make the
    // whole operator quadratic, which shows up the first time someone extrudes
    // a whole character rather than one face.
    let mut in_region = vec![false; mesh.slot_count(crate::Domain::Face)];
    for f in &faces {
        in_region[f.index() as usize] = true;
    }

    let mut rings: Vec<Vec<VertId>> = Vec::with_capacity(faces.len());
    let mut boundary: Vec<(VertId, VertId)> = Vec::new();
    for &f in &faces {
        let loops = mesh.face_loops(f);
        let mut ring = Vec::with_capacity(loops.len());
        for &l in &loops {
            let Some(v) = mesh.loop_vert(l) else { continue };
            ring.push(v);

            let Some(e) = mesh.loop_edge(l) else { continue };
            let inside = mesh
                .edge_faces(e)
                .into_iter()
                .filter(|g| in_region[g.index() as usize])
                .count();
            if inside == 1 {
                // Direction taken from this face's own winding, so the wall it
                // grows is wound consistently with it. Reading it from the edge
                // instead gives an arbitrary orientation and half the walls come
                // out inside-out.
                let Some(next) = mesh.loop_next(l) else {
                    continue;
                };
                let Some(b) = mesh.loop_vert(next) else {
                    continue;
                };
                boundary.push((v, b));
            }
        }
        rings.push(ring);
    }

    // --- write --------------------------------------------------------------
    let mut originals: Vec<VertId> = rings.iter().flatten().copied().collect();
    originals.sort_unstable_by_key(|v| v.index());
    originals.dedup();

    let mut copy_of = vec![None; mesh.slot_count(crate::Domain::Point)];
    let mut new_verts = Vec::with_capacity(originals.len());
    for &v in &originals {
        let Some(p) = mesh.position(v) else { continue };
        let n = mesh.add_vert(p + offset);
        // `add_vert` can grow the point domain, so the table is sized after.
        if copy_of.len() <= v.index() as usize {
            copy_of.resize(v.index() as usize + 1, None);
        }
        copy_of[v.index() as usize] = Some(n);
        new_verts.push(n);
    }
    let copied =
        |v: VertId| -> Option<VertId> { copy_of.get(v.index() as usize).copied().flatten() };

    let mut new_faces = Vec::with_capacity(faces.len());
    for ring in &rings {
        let lifted: Vec<VertId> = ring.iter().filter_map(|v| copied(*v)).collect();
        if lifted.len() == ring.len() {
            new_faces.push(mesh.add_face(&lifted)?);
        }
    }

    let mut side_faces = Vec::with_capacity(boundary.len());
    for (a, b) in boundary {
        let (Some(a2), Some(b2)) = (copied(a), copied(b)) else {
            continue;
        };
        // `(a, b, b', a')`: with `a → b` the original winding, this quad's normal
        // points away from the solid. Reversing it produces a shell that is lit
        // from the inside and reads as a shading bug rather than a winding one.
        side_faces.push(mesh.add_face(&[a, b, b2, a2])?);
    }

    for f in &faces {
        mesh.remove_face(*f);
    }

    // A vertex that the region alone used is now interior to nothing: the shell
    // it belonged to has moved up and the walls do not reach it. Leaving it
    // behind litters the mesh with points that no operator will ever touch again
    // and that the user cannot see to delete.
    for v in originals {
        if mesh.vert_faces(v).is_empty() {
            mesh.remove_vert(v);
        }
    }

    // The new shell is what the user goes on to move, so it is what must be
    // selected — anything else and the follow-up transform grabs the wrong thing.
    mesh.select_none();
    for f in &new_faces {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(SelectMode::Face);

    Ok(Extrusion {
        new_faces,
        side_faces,
        new_verts,
        normal,
    })
}

/// Extrudes the selected faces along the region normal.
///
/// The form a keystroke reaches: `E` extrudes along the normal, and the modal
/// transform that follows constrains to that axis by default.
pub fn extrude_selected_faces(mesh: &mut BMesh, distance: f32) -> Result<Extrusion, BmError> {
    let faces = mesh.selected_faces();
    let normal = mesh.region_normal(&faces);
    extrude_faces(mesh, &faces, normal * distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn cube() -> BMesh {
        BMesh::from_mesh_data(&primitives::cube(2.0))
    }

    /// The face whose normal points most nearly along `dir`.
    fn face_facing(mesh: &BMesh, dir: Vec3) -> FaceId {
        mesh.face_ids()
            .max_by(|a, b| {
                let da = mesh.face_normal(*a).normalize_or_zero().dot(dir);
                let db = mesh.face_normal(*b).normalize_or_zero().dot(dir);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
    }

    #[test]
    fn extruding_one_face_of_a_cube_makes_a_box_on_a_box() {
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        let out = extrude_faces(&mut m, &[top], Vec3::Z * 3.0).unwrap();
        m.assert_valid();

        assert_eq!(out.new_faces.len(), 1);
        assert_eq!(out.side_faces.len(), 4, "one wall per boundary edge");
        assert_eq!(out.new_verts.len(), 4);

        assert_eq!(
            m.num_faces(),
            5 + 1 + 4,
            "five original, the lid, four walls"
        );
        assert_eq!(m.num_verts(), 12);
        assert_eq!(m.num_edges(), 20);

        // Still a closed solid: every edge has exactly two faces.
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "the result must stay watertight");
        }
        assert_eq!(
            m.num_verts() as i64 - m.num_edges() as i64 + m.num_faces() as i64,
            2,
            "Euler characteristic of a genus-0 closed surface"
        );

        let lifted = m.face_verts(out.new_faces[0]);
        for v in lifted {
            assert!((m.position(v).unwrap().z - 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn the_walls_face_outward() {
        // The classic silent failure. A reversed side quad still draws, still
        // validates, and still exports — it just lights from the inside, and the
        // report that comes back is "the shading looks wrong".
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        let out = extrude_faces(&mut m, &[top], Vec3::Z * 2.0).unwrap();

        let centre = Vec3::new(0.0, 0.0, 2.0);
        for f in &out.side_faces {
            let n = m.face_normal(*f).normalize_or_zero();
            let verts = m.face_verts(*f);
            let mid: Vec3 =
                verts.iter().filter_map(|v| m.position(*v)).sum::<Vec3>() / verts.len() as f32;
            assert!(
                n.dot(mid - centre) > 0.0,
                "wall {f:?} points inward: normal {n:?} at {mid:?}"
            );
        }
        let lid = m.face_normal(out.new_faces[0]).normalize_or_zero();
        assert!(lid.dot(Vec3::Z) > 0.99, "the lid kept the original winding");
    }

    #[test]
    fn a_region_grows_walls_only_around_its_outline() {
        // The region-versus-per-face distinction, stated as a count. Two adjacent
        // faces share an edge; a per-face extrude would put a wall on it and
        // leave a membrane inside the solid.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        assert_eq!(m.num_faces(), 4);
        let all: Vec<FaceId> = m.face_ids().collect();

        let out = extrude_faces(&mut m, &all, Vec3::Z).unwrap();
        m.assert_valid();
        assert_eq!(out.new_faces.len(), 4);
        assert_eq!(
            out.side_faces.len(),
            8,
            "the grid's outline is eight edges; the four interior edges get no wall"
        );
        assert_eq!(m.num_faces(), 12, "no leftover faces from the old region");
    }

    #[test]
    fn the_interior_of_an_extruded_region_leaves_nothing_behind() {
        // A 3x3 grid has one vertex used only by the region. After the region
        // lifts away, that vertex belongs to nothing.
        let mut m = BMesh::from_mesh_data(&primitives::grid(6.0, 3, 3));
        let before = m.num_verts();
        let all: Vec<FaceId> = m.face_ids().collect();
        extrude_faces(&mut m, &all, Vec3::Z).unwrap();
        m.assert_valid();

        // 16 vertices, of which the 4 interior ones are used only by the region.
        assert_eq!(m.num_verts(), before + 16 - 4);
        for v in m.vert_ids() {
            assert!(
                !m.vert_faces(v).is_empty(),
                "{v:?} was left with no faces at all"
            );
        }
    }

    #[test]
    fn a_zero_offset_extrude_changes_topology_and_nothing_else() {
        // What an interactive extrude does on mouse-down: build the shell, then
        // let the modal transform move it. If this moved anything, the first
        // frame of every extrude would jump.
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        let before: Vec<Vec3> = m.vert_ids().filter_map(|v| m.position(v)).collect();

        let out = extrude_faces(&mut m, &[top], Vec3::ZERO).unwrap();
        m.assert_valid();

        for v in &out.new_verts {
            let p = m.position(*v).unwrap();
            assert!(
                before.iter().any(|q| (*q - p).length() < 1e-6),
                "the copy sits exactly on its original"
            );
        }
        assert_eq!(out.side_faces.len(), 4);
    }

    #[test]
    fn the_new_shell_is_what_ends_up_selected() {
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        m.set_face_selected(top, true);
        m.flush_selection(SelectMode::Face);

        let out = extrude_selected_faces(&mut m, 2.0).unwrap();
        assert_eq!(m.selected_faces(), out.new_faces);
        assert_eq!(m.selected_verts().len(), 4);
        assert!(
            !m.selected_faces().contains(&top),
            "the original is gone, not merely deselected"
        );
    }

    #[test]
    fn extruding_a_boundary_face_of_an_open_surface_works() {
        // A plane has no second face on any edge, so every edge is boundary and
        // the extrusion is an open box with no lid on the bottom.
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let f = m.face_ids().next().unwrap();
        let out = extrude_faces(&mut m, &[f], Vec3::Z * 2.0).unwrap();
        m.assert_valid();

        assert_eq!(out.side_faces.len(), 4);
        assert_eq!(m.num_faces(), 5);
        assert_eq!(m.num_verts(), 8, "the original four are held by the walls");
    }

    #[test]
    fn extruding_nothing_is_a_no_op_rather_than_an_error() {
        let mut m = cube();
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        let out = extrude_faces(&mut m, &[], Vec3::Z).unwrap();
        assert!(out.new_faces.is_empty());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
        m.assert_valid();
    }

    #[test]
    fn repeated_extrusions_stack() {
        let mut m = cube();
        let top = face_facing(&m, Vec3::Z);
        m.set_face_selected(top, true);
        m.flush_selection(SelectMode::Face);

        for _ in 0..8 {
            extrude_selected_faces(&mut m, 1.0).unwrap();
            m.assert_valid();
        }
        assert_eq!(m.num_faces(), 5 + 1 + 8 * 4);
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "still watertight after eight");
        }
        let top_z = m
            .selected_verts()
            .iter()
            .filter_map(|v| m.position(*v))
            .map(|p| p.z)
            .fold(f32::MIN, f32::max);
        assert!(
            (top_z - 9.0).abs() < 1e-4,
            "eight metres above the cube's lid"
        );
    }

    #[test]
    fn extruding_every_face_of_a_solid_grows_a_shell() {
        // The whole surface is the region, so there is no boundary at all and no
        // walls: the result is the original solid with a second, detached shell
        // outside it. Unusual, but it must not corrupt anything.
        let mut m = cube();
        let all: Vec<FaceId> = m.face_ids().collect();
        let out = extrude_faces(&mut m, &all, Vec3::ZERO).unwrap();
        m.assert_valid();

        assert!(out.side_faces.is_empty(), "a closed region has no outline");
        assert_eq!(out.new_faces.len(), 6);
        assert_eq!(m.num_faces(), 6, "the original faces were replaced");
        assert_eq!(m.num_verts(), 8, "and their vertices reclaimed");
    }
}
