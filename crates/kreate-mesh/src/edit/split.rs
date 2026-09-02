//! Split and rip: taking geometry apart.
//!
//! Two operations that users tell apart by what stays connected. **Split**
//! detaches a region of faces from the rest, leaving it exactly where it was as
//! a separate shell. **Rip** opens a seam along selected edges, so the surface
//! stays one piece but can be pulled apart along that line.
//!
//! # Both are the same algorithm
//!
//! For each vertex on the cut, work out which faces should keep it and which
//! should get a copy, then move those faces' corners onto the copy. What differs
//! is only how the fan around a vertex is grouped: split groups by "in the
//! region or not", rip groups by "reachable without crossing a selected edge".
//!
//! Sharing the implementation is not tidiness. Corner data survives here because
//! [`BMesh::replace_vert_in_face`] keeps the loops, and a second implementation
//! would be the one that forgets — leaving a seam where the UVs tear.

use crate::arena::{ArenaIndex, EdgeId, FaceId, VertId};
use crate::bmesh::{BMesh, BmError};
use crate::Domain;

use super::region::Region;

/// What a split or rip produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Split {
    /// The duplicated vertices.
    pub verts: Vec<VertId>,
    /// How many seams were opened, i.e. vertices that had to be duplicated.
    pub seams: usize,
}

/// Duplicates `v` and moves the given faces' corners onto the copy.
fn detach(mesh: &mut BMesh, v: VertId, faces: &[FaceId], out: &mut Split) -> Result<(), BmError> {
    let Some(position) = mesh.position(v) else {
        return Ok(());
    };
    let copy = mesh.add_vert(position);
    let mut moved = false;
    for &f in faces {
        moved |= mesh.replace_vert_in_face(f, v, copy);
    }
    if moved {
        out.verts.push(copy);
        out.seams += 1;
    } else {
        // Nothing took the copy — a corner that would have folded onto an edge,
        // which `replace_vert_in_face` refuses. Removing it keeps the mesh free
        // of a stray point the user cannot see and did not ask for.
        mesh.remove_vert(copy);
    }
    Ok(())
}

/// Removes every edge around the given vertices that no face uses any more.
///
/// A split leaves two kinds of wire behind, and gathering the candidates *after*
/// the work — not before it — is what catches both. The obvious one is the
/// original edge, abandoned once every face using it has moved to a copy. The
/// other is transient: moving a face's corners one at a time creates
/// part-original, part-copy edges along the way — `(v0', v1)` before `v1` has
/// moved too — which the very next step abandons. A list gathered beforehand
/// contains neither, and they go on joining the two halves of a split that looks
/// separated in every other respect.
///
/// `around` is the *original* vertices only. That is sufficient rather than
/// lucky: every wire either edge is one the plan touched, or was created by
/// moving a corner off a plan vertex, so it has a plan vertex at one end. The
/// debug assertion below is what will catch a future change that breaks that.
fn sweep_wires(mesh: &mut BMesh, around: &[VertId], copies: &[VertId]) {
    let gather = |mesh: &BMesh, from: &[VertId]| -> Vec<EdgeId> {
        let mut out: Vec<EdgeId> = from
            .iter()
            .filter(|v| mesh.vert(**v).is_some())
            .flat_map(|v| mesh.vert_edges(*v))
            .collect();
        out.sort_unstable_by_key(|e| e.index());
        out.dedup();
        out
    };
    for e in gather(mesh, around) {
        if mesh.edge(e).is_some() && mesh.is_wire_edge(e) {
            mesh.remove_edge(e);
        }
    }
    debug_assert!(
        gather(mesh, copies)
            .into_iter()
            .all(|e| !mesh.is_wire_edge(e)),
        "a wire edge survived with only copied endpoints; sweeping the originals          is no longer sufficient"
    );
}

/// Detaches a region of faces from the rest of the mesh.
///
/// The region stays exactly where it was; it simply no longer shares any vertex
/// with what surrounds it. Blender's `Y`.
pub fn split_region(mesh: &mut BMesh, faces: &[FaceId]) -> Result<Split, BmError> {
    let region = Region::new(mesh, faces);
    let mut out = Split::default();
    if region.is_empty() {
        return Ok(out);
    }

    // Read the whole plan before changing anything: the fans are read from the
    // topology that is about to change underneath them.
    let mut plan: Vec<(VertId, Vec<FaceId>)> = Vec::new();
    for v in region.verts(mesh) {
        let fan = mesh.vert_faces(v);
        let inside: Vec<FaceId> = fan
            .iter()
            .copied()
            .filter(|f| region.contains(*f))
            .collect();
        // Only a vertex the region *shares* needs splitting. One used by the
        // region alone is already detached.
        if inside.len() == fan.len() {
            continue;
        }
        plan.push((v, inside));
    }
    let involved: Vec<VertId> = plan.iter().map(|(v, _)| *v).collect();
    for (v, faces) in plan {
        detach(mesh, v, &faces, &mut out)?;
    }
    sweep_wires(mesh, &involved, &out.verts);
    Ok(out)
}

/// Opens the mesh along the given edges.
///
/// The surface stays one piece where the edges do not reach, and separates where
/// they do — so a chain of edges across a sheet becomes a slit, and a closed loop
/// becomes a full separation. Blender's `V` in edge mode, and what its Edge Split
/// modifier does.
///
/// Edges with fewer than two faces are ignored: there is nothing joined across
/// them to open.
pub fn split_edges(mesh: &mut BMesh, edges: &[EdgeId]) -> Result<Split, BmError> {
    let mut cut = vec![false; mesh.slot_count(Domain::Edge)];
    let mut affected: Vec<EdgeId> = Vec::new();
    for &e in edges {
        if mesh.edge(e).is_none() || mesh.edge_face_count(e) < 2 {
            continue;
        }
        cut[e.index() as usize] = true;
        affected.push(e);
    }
    let mut out = Split::default();
    if affected.is_empty() {
        return Ok(out);
    }

    // Which vertices the cut passes through.
    let mut touched: Vec<VertId> = affected
        .iter()
        .filter_map(|e| mesh.edge_verts(*e))
        .flatten()
        .collect();
    touched.sort_unstable_by_key(|v| v.index());
    touched.dedup();

    // The fan around each touched vertex, grouped by what the cut leaves
    // connected. Read in full first, for the same reason as above.
    let mut plan: Vec<(VertId, Vec<Vec<FaceId>>)> = Vec::new();
    for v in touched {
        let groups = fan_groups(mesh, v, &cut);
        if groups.len() < 2 {
            // The cut does not separate anything here. Either the chain stops
            // at this vertex, or the fan around it stays connected the other way
            // — which is why cutting one edge of a cube opens nothing.
            continue;
        }
        plan.push((v, groups));
    }

    let involved: Vec<VertId> = plan.iter().map(|(v, _)| *v).collect();
    for (v, groups) in plan {
        // The first group keeps the original vertex; every other gets a copy.
        // Which group is "first" is arbitrary but deterministic, because the fan
        // is walked in slot order.
        for group in groups.iter().skip(1) {
            detach(mesh, v, group, &mut out)?;
        }
    }
    sweep_wires(mesh, &involved, &out.verts);
    Ok(out)
}

/// The faces around `v`, grouped by what remains connected once the cut edges
/// are treated as gaps.
///
/// Two faces are in the same group when they share an edge at `v` that is not
/// cut. Deterministic: the fan is read in slot order and the walk seeds from the
/// lowest-numbered face, so the same input always splits the same way.
fn fan_groups(mesh: &BMesh, v: VertId, cut: &[bool]) -> Vec<Vec<FaceId>> {
    let mut fan = mesh.vert_faces(v);
    fan.sort_unstable_by_key(|f| f.index());

    let mut groups: Vec<Vec<FaceId>> = Vec::new();
    let mut assigned: Vec<FaceId> = Vec::new();
    for &seed in &fan {
        if assigned.contains(&seed) {
            continue;
        }
        let mut group = vec![seed];
        assigned.push(seed);
        let mut queue = vec![seed];
        while let Some(current) = queue.pop() {
            for l in mesh.face_loops(current) {
                let Some(e) = mesh.loop_edge(l) else { continue };
                // Only the two edges at `v` connect faces around it.
                if !mesh.edge_verts(e).is_some_and(|pair| pair.contains(&v)) {
                    continue;
                }
                if cut.get(e.index() as usize).copied().unwrap_or(false) {
                    continue;
                }
                for g in mesh.edge_faces(e) {
                    if fan.contains(&g) && !assigned.contains(&g) {
                        assigned.push(g);
                        group.push(g);
                        queue.push(g);
                    }
                }
            }
        }
        group.sort_unstable_by_key(|f| f.index());
        groups.push(group);
    }
    groups
}

/// Detaches the selected faces from the rest.
pub fn split_selected_faces(mesh: &mut BMesh) -> Result<Split, BmError> {
    let faces = mesh.selected_faces();
    split_region(mesh, &faces)
}

/// Opens the mesh along the selected edges.
pub fn rip_selected_edges(mesh: &mut BMesh) -> Result<Split, BmError> {
    let edges = mesh.selected_edges();
    split_edges(mesh, &edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;
    use crate::SelectMode;
    use kreate_math::{Vec2, Vec3};

    /// How many connected pieces the mesh has, by walking shared vertices.
    fn shells(mesh: &BMesh) -> usize {
        let mut seen: Vec<VertId> = Vec::new();
        let mut count = 0;
        for start in mesh.vert_ids() {
            if seen.contains(&start) {
                continue;
            }
            count += 1;
            let mut queue = vec![start];
            seen.push(start);
            while let Some(v) = queue.pop() {
                for e in mesh.vert_edges(v) {
                    if let Some(w) = mesh.edge_other_vert(e, v) {
                        if !seen.contains(&w) {
                            seen.push(w);
                            queue.push(w);
                        }
                    }
                }
            }
        }
        count
    }

    #[test]
    fn splitting_a_face_off_a_cube_leaves_two_shells() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        assert_eq!(shells(&m), 1);
        let f = m.face_ids().next().unwrap();
        let before = m.face_verts(f);

        let out = split_region(&mut m, &[f]).unwrap();
        m.assert_valid();

        assert_eq!(out.seams, 4, "all four corners were shared");
        assert_eq!(m.num_verts(), 12, "four new copies");
        assert_eq!(m.num_faces(), 6, "no face was added or lost");
        assert_eq!(shells(&m), 2, "the face is a piece of its own");

        // And it did not move.
        let after = m.face_verts(f);
        for (a, b) in before.iter().zip(&after) {
            let _ = a;
            assert!(m.position(*b).is_some());
        }
    }

    #[test]
    fn a_split_face_stays_exactly_where_it_was() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        let before: Vec<Vec3> = m
            .face_verts(f)
            .iter()
            .filter_map(|v| m.position(*v))
            .collect();

        split_region(&mut m, &[f]).unwrap();
        let after: Vec<Vec3> = m
            .face_verts(f)
            .iter()
            .filter_map(|v| m.position(*v))
            .collect();
        assert_eq!(before.len(), after.len());
        for p in &after {
            assert!(
                before.iter().any(|q| (*q - *p).length() < 1e-6),
                "a corner moved to {p:?}"
            );
        }
    }

    #[test]
    fn splitting_preserves_per_corner_data() {
        // The reason this goes through `replace_vert_in_face`. Rebuilding the
        // faces would clear the UVs of everything along the seam, and the user
        // finds out when the texture tears.
        use crate::attribute::AttrData;
        use std::sync::Arc;

        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let corners = m.slot_count(Domain::Corner);
        m.attributes_mut().insert(
            "uv",
            Domain::Corner,
            AttrData::F32x2(Arc::new(
                (0..corners).map(|i| Vec2::new(i as f32, 0.0)).collect(),
            )),
        );
        let before: Vec<(u32, f32)> = m
            .loop_ids()
            .map(|l| {
                (
                    l.index(),
                    m.attributes().get_f32x2("uv").unwrap()[l.index() as usize].x,
                )
            })
            .collect();

        let f = m.face_ids().next().unwrap();
        split_region(&mut m, &[f]).unwrap();
        m.assert_valid();

        for (slot, value) in before {
            assert_eq!(
                m.attributes().get_f32x2("uv").unwrap()[slot as usize].x,
                value,
                "corner {slot} lost its UV"
            );
        }
    }

    #[test]
    fn splitting_leaves_no_wire_behind() {
        // The original edges survive wherever every face using them moved to a
        // copy. Invisible in a solid view, and quietly wrong to every operator
        // that walks the topology afterwards.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        split_region(&mut m, &[f]).unwrap();
        m.assert_valid();
        assert!(
            m.edge_ids().all(|e| !m.is_wire_edge(e)),
            "a wire edge was left behind"
        );
        assert_eq!(m.num_edges(), 16, "12 original plus the copy's four");
    }

    #[test]
    fn splitting_a_whole_mesh_changes_nothing() {
        // Every vertex is used by the region alone, so nothing is shared and
        // nothing needs a copy.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let all: Vec<FaceId> = m.face_ids().collect();
        let out = split_region(&mut m, &all).unwrap();
        m.assert_valid();
        assert_eq!(out, Split::default());
        assert_eq!(m.num_verts(), 8);
    }

    // --- rip ----------------------------------------------------------------

    #[test]
    fn ripping_an_edge_loop_separates_a_cylinder() {
        // A closed loop of interior edges cuts the tube into two open bands.
        let mut m = BMesh::from_mesh_data(&primitives::cylinder(1.0, 2.0, 8));
        let caps: Vec<FaceId> = m
            .face_ids()
            .filter(|f| m.face(*f).unwrap().len() == 8)
            .collect();
        for f in caps {
            m.remove_face(f);
        }
        // Cut around the middle: there is no middle ring on a plain cylinder, so
        // add one first.
        let vertical = m
            .edge_ids()
            .find(|&e| {
                let [a, b] = m.edge_verts(e).unwrap();
                (m.position(a).unwrap().z - m.position(b).unwrap().z).abs() > 0.5
            })
            .unwrap();
        let cut = super::super::loopcut::loop_cut(&mut m, vertical, 1, 0.5).unwrap();
        m.assert_valid();
        assert_eq!(shells(&m), 1);

        // The new loop's edges are the ones joining consecutive cut vertices.
        let mut loop_edges: Vec<EdgeId> = Vec::new();
        for &a in &cut.verts {
            for e in m.vert_edges(a) {
                let Some(b) = m.edge_other_vert(e, a) else {
                    continue;
                };
                if cut.verts.contains(&b) && !loop_edges.contains(&e) {
                    loop_edges.push(e);
                }
            }
        }
        assert_eq!(loop_edges.len(), 8, "a ring of eight");

        let out = split_edges(&mut m, &loop_edges).unwrap();
        m.assert_valid();
        assert_eq!(out.seams, 8, "one copy per vertex on the ring");
        assert_eq!(shells(&m), 2, "the tube is now two bands");
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }

    /// The interior edges of one row of a grid, running along X at `y == 0`.
    fn middle_row(mesh: &BMesh) -> Vec<EdgeId> {
        mesh.edge_ids()
            .filter(|&e| {
                if mesh.edge_face_count(e) != 2 {
                    return false;
                }
                let [a, b] = mesh.edge_verts(e).unwrap();
                let (pa, pb) = (mesh.position(a).unwrap(), mesh.position(b).unwrap());
                pa.y.abs() < 1e-4 && pb.y.abs() < 1e-4 && (pa.x - pb.x).abs() > 1e-4
            })
            .collect()
    }

    #[test]
    fn ripping_part_of_a_row_makes_a_slit_the_sheet_still_holds() {
        // The case a full-loop test would miss. The chain stops inside the
        // sheet, so only the vertex the cut passes *through* separates — the two
        // ends stay shared, and that is what makes the slit a zip rather than a
        // cut in half.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
        let mut row = middle_row(&m);
        row.sort_by(|a, b| {
            let ax = m.position(m.edge_verts(*a).unwrap()[0]).unwrap().x;
            let bx = m.position(m.edge_verts(*b).unwrap()[0]).unwrap().x;
            ax.partial_cmp(&bx).unwrap()
        });
        assert_eq!(row.len(), 4);
        let chain = &row[1..3];

        let before = m.num_verts();
        let out = split_edges(&mut m, chain).unwrap();
        m.assert_valid();

        assert_eq!(
            out.seams, 1,
            "only the vertex between the two cut edges separates"
        );
        assert_eq!(m.num_verts(), before + 1);
        assert_eq!(shells(&m), 1, "a slit, not a separation");
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }

    #[test]
    fn ripping_a_row_that_spans_the_sheet_cuts_it_in_two() {
        // The contrast. A chain reaching both borders is a separation, and every
        // vertex on it splits — including its ends, whose fans are arcs rather
        // than closed rings.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
        let row = middle_row(&m);
        assert_eq!(row.len(), 4);

        let out = split_edges(&mut m, &row).unwrap();
        m.assert_valid();

        assert_eq!(out.seams, 5, "all five vertices along the row");
        assert_eq!(shells(&m), 2, "the sheet is now two halves");
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }

    #[test]
    fn ripping_a_boundary_edge_does_nothing() {
        // Nothing is joined across it to open. Silently doing something else
        // would be worse than doing nothing.
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let e = m.edge_ids().next().unwrap();
        let out = split_edges(&mut m, &[e]).unwrap();
        assert_eq!(out, Split::default());
        assert_eq!(m.num_verts(), 4);
        m.assert_valid();
    }

    #[test]
    fn ripping_one_interior_edge_of_a_cube_opens_a_seam() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let e = m.edge_ids().next().unwrap();
        let out = split_edges(&mut m, &[e]).unwrap();
        m.assert_valid();

        // A cube's edge has three faces at each end, and cutting one edge leaves
        // the fan connected the other way round — so neither end separates.
        assert_eq!(
            out.seams, 0,
            "the fans stay connected around the other side"
        );
        assert_eq!(shells(&m), 1);
        assert_eq!(m.num_verts(), 8);
    }

    #[test]
    fn ripping_the_selection_goes_through_the_same_path() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
        m.select_none();
        for e in middle_row(&m) {
            m.set_edge_selected(e, true);
        }
        m.flush_selection(SelectMode::Edge);

        let out = rip_selected_edges(&mut m).unwrap();
        m.assert_valid();
        assert_eq!(out.seams, 5);
        assert_eq!(shells(&m), 2);
    }

    #[test]
    fn splitting_nothing_is_a_no_op() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        assert_eq!(split_region(&mut m, &[]).unwrap(), Split::default());
        assert_eq!(split_edges(&mut m, &[]).unwrap(), Split::default());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
        m.assert_valid();
    }

    #[test]
    fn the_grouping_is_the_same_on_every_run() {
        // Which side of a rip keeps the original vertex is arbitrary but must be
        // stable, or the same file produces different geometry twice running.
        let build = || {
            let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
            let chain = middle_row(&m);
            split_edges(&mut m, &chain).unwrap();
            m.to_mesh_data().corner_verts().to_vec()
        };
        let first = build();
        for _ in 0..5 {
            assert_eq!(build(), first);
        }
    }
}
