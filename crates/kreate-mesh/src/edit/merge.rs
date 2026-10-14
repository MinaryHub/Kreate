//! Merge and weld.
//!
//! Collapsing several vertices into one. The topology work is the same for both;
//! they differ only in how the groups are chosen — by selection, or by distance.
//!
//! # Why this is not just "move them to the same place"
//!
//! Two vertices at identical coordinates are still two vertices. They shade with
//! a seam, they pull apart under any deformation, they export as a split mesh,
//! and every operator that walks the topology treats them as unconnected. Weld
//! is the step that makes an imported triangle soup editable at all, and it is
//! the one importers most often skip.
//!
//! # What collapsing does to faces
//!
//! A face whose corners collapse together stops being a face. Two of its corners
//! merging leaves a triangle where a quad was — fine, and kept. All but two
//! merging leaves a sliver with no area — dropped. And a face whose *opposite*
//! corners merge is a bowtie: it repeats a vertex, which no valid face may do.
//! Those are dropped and counted rather than repaired, because the repair (a
//! split into two faces) is a guess about what the user meant.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, FaceId, VertId};
use crate::bmesh::{BMesh, BmError};
use crate::Domain;

/// Where a merged group ends up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergeTarget {
    /// The average of the group.
    #[default]
    Centre,
    /// The first vertex in slot order, which is stable across runs.
    First,
    /// The last vertex in slot order.
    Last,
}

/// What a merge produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Merge {
    /// Vertices that went.
    pub removed: usize,
    /// Faces that collapsed to nothing, or to a shape that repeats a vertex.
    pub faces_dropped: usize,
    /// Groups that were collapsed.
    pub groups: usize,
}

/// Collapses each group of vertices into one.
///
/// Groups are handled together rather than one at a time so that a face touching
/// two different groups sees both collapses at once — sequential merges would
/// rebuild it twice and the second rebuild would work on a ring the first had
/// already changed.
pub fn merge_groups(
    mesh: &mut BMesh,
    groups: &[Vec<VertId>],
    target: MergeTarget,
) -> Result<Merge, BmError> {
    let mut out = Merge::default();
    let slots = mesh.slot_count(Domain::Point);
    // `remap[slot]` is the vertex that slot's vertex becomes. Identity where a
    // vertex is not being merged.
    let mut remap: Vec<Option<VertId>> = vec![None; slots];
    let mut doomed: Vec<VertId> = Vec::new();

    for group in groups {
        let mut members: Vec<VertId> = group
            .iter()
            .copied()
            .filter(|v| mesh.vert(*v).is_some())
            .collect();
        members.sort_unstable_by_key(|v| v.index());
        members.dedup();
        if members.len() < 2 {
            continue;
        }
        out.groups += 1;

        let survivor = match target {
            MergeTarget::First | MergeTarget::Centre => members[0],
            MergeTarget::Last => members[members.len() - 1],
        };
        let place = match target {
            MergeTarget::Centre => {
                let sum: Vec3 = members.iter().filter_map(|v| mesh.position(*v)).sum();
                sum / members.len() as f32
            }
            _ => mesh.position(survivor).unwrap_or(Vec3::ZERO),
        };
        mesh.set_position(survivor, place);

        for &v in &members {
            if v != survivor {
                remap[v.index() as usize] = Some(survivor);
                doomed.push(v);
            }
        }
    }
    if doomed.is_empty() {
        return Ok(out);
    }

    let mapped = |v: VertId, table: &[Option<VertId>]| -> VertId {
        table
            .get(v.index() as usize)
            .copied()
            .flatten()
            .unwrap_or(v)
    };

    // --- read ---------------------------------------------------------------
    // Every face touching anything that is about to move, with its ring. Read in
    // full before any removal: the loop cycles are gone the moment the faces
    // are.
    let mut affected: Vec<FaceId> = Vec::new();
    for &v in &doomed {
        for f in mesh.vert_faces(v) {
            if !affected.contains(&f) {
                affected.push(f);
            }
        }
    }
    affected.sort_unstable_by_key(|f| f.index());
    let rings: Vec<Vec<VertId>> = affected.iter().map(|&f| mesh.face_verts(f)).collect();

    // --- write --------------------------------------------------------------
    for &f in &affected {
        mesh.remove_face(f);
    }

    for ring in rings {
        let mut rebuilt: Vec<VertId> = Vec::with_capacity(ring.len());
        for v in ring {
            let m = mapped(v, &remap);
            // Consecutive duplicates are the ordinary case — two adjacent
            // corners merging — and they simply shorten the face.
            if rebuilt.last() != Some(&m) {
                rebuilt.push(m);
            }
        }
        if rebuilt.len() > 1 && rebuilt.first() == rebuilt.last() {
            rebuilt.pop();
        }
        let mut sorted = rebuilt.clone();
        sorted.sort_unstable_by_key(|v| v.index());
        let before = sorted.len();
        sorted.dedup();
        // A vertex repeated non-consecutively is a bowtie: the face crosses
        // itself. Dropping it is a loss, but making one up is worse — the repair
        // is a guess about which of two faces the user wanted.
        if rebuilt.len() < 3 || sorted.len() != before {
            out.faces_dropped += 1;
            continue;
        }
        mesh.add_face(&rebuilt)?;
    }

    for v in doomed {
        if mesh.vert(v).is_some() {
            mesh.remove_vert(v);
            out.removed += 1;
        }
    }

    // Wire edges left where a collapsed face's edges used to be.
    let wires: Vec<_> = mesh.edge_ids().filter(|&e| mesh.is_wire_edge(e)).collect();
    for e in wires {
        mesh.remove_edge(e);
    }

    mesh.select_none();
    Ok(out)
}

/// Collapses the selected vertices into one.
pub fn merge_selected(mesh: &mut BMesh, target: MergeTarget) -> Result<Merge, BmError> {
    let group = mesh.selected_verts();
    merge_groups(mesh, &[group], target)
}

/// Merges vertices closer together than `threshold`.
///
/// The step that makes an imported triangle soup editable. Without it a glTF or
/// STL mesh has three unconnected vertices per triangle, and every modelling
/// operation shatters it.
///
/// Grouping is by a grid whose cell is the threshold, with the eight
/// neighbouring cells also searched — a pair straddling a cell boundary is
/// otherwise missed, and the result would depend on where the model happened to
/// sit relative to the origin.
pub fn weld_by_distance(
    mesh: &mut BMesh,
    verts: &[VertId],
    threshold: f32,
) -> Result<Merge, BmError> {
    if threshold <= 0.0 {
        return Ok(Merge::default());
    }
    let cell = threshold.max(f32::MIN_POSITIVE);
    let key = |p: Vec3| -> [i32; 3] {
        [
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        ]
    };

    // A sorted table rather than a hash map: the grouping decides the output
    // geometry, and a hasher would make it depend on iteration order.
    let mut cells: Vec<([i32; 3], VertId)> = verts
        .iter()
        .copied()
        .filter_map(|v| Some((key(mesh.position(v)?), v)))
        .collect();
    cells.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.index().cmp(&b.1.index())));

    let lookup = |k: [i32; 3]| -> &[([i32; 3], VertId)] {
        let lo = cells.partition_point(|(c, _)| *c < k);
        let hi = cells.partition_point(|(c, _)| *c <= k);
        &cells[lo..hi]
    };

    let slots = mesh.slot_count(Domain::Point);
    let mut group_of: Vec<Option<usize>> = vec![None; slots];
    let mut groups: Vec<Vec<VertId>> = Vec::new();
    let t2 = threshold * threshold;

    // In slot order, so the same input always produces the same grouping.
    let mut ordered: Vec<VertId> = verts.to_vec();
    ordered.sort_unstable_by_key(|v| v.index());
    ordered.dedup();

    for v in ordered {
        if group_of[v.index() as usize].is_some() {
            continue;
        }
        let Some(p) = mesh.position(v) else { continue };
        let k = key(p);
        let index = groups.len();
        let mut group = vec![v];
        group_of[v.index() as usize] = Some(index);

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let near = [k[0] + dx, k[1] + dy, k[2] + dz];
                    for &(_, other) in lookup(near) {
                        if group_of[other.index() as usize].is_some() {
                            continue;
                        }
                        let Some(q) = mesh.position(other) else {
                            continue;
                        };
                        if (q - p).length_squared() <= t2 {
                            group_of[other.index() as usize] = Some(index);
                            group.push(other);
                        }
                    }
                }
            }
        }
        if group.len() > 1 {
            groups.push(group);
        } else {
            groups.push(Vec::new());
        }
    }

    let groups: Vec<Vec<VertId>> = groups.into_iter().filter(|g| g.len() > 1).collect();
    merge_groups(mesh, &groups, MergeTarget::Centre)
}

/// Merges each of `moved` with whatever other vertex now sits in its place,
/// within `threshold`: Blender's Auto Merge, run after a vertex is moved or
/// snapped onto another. Only groups with a moved vertex in them merge —
/// two vertices left overlapping on purpose elsewhere in the mesh stay two.
pub fn weld_onto(mesh: &mut BMesh, moved: &[VertId], threshold: f32) -> Result<Merge, BmError> {
    if threshold <= 0.0 || moved.is_empty() {
        return Ok(Merge::default());
    }
    let cell = threshold.max(f32::MIN_POSITIVE);
    let key = |p: Vec3| -> [i32; 3] {
        [
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        ]
    };
    let mut cells: Vec<([i32; 3], VertId)> = mesh
        .vert_ids()
        .filter_map(|v| Some((key(mesh.position(v)?), v)))
        .collect();
    cells.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.index().cmp(&b.1.index())));
    let lookup = |k: [i32; 3]| -> &[([i32; 3], VertId)] {
        let lo = cells.partition_point(|(c, _)| *c < k);
        let hi = cells.partition_point(|(c, _)| *c <= k);
        &cells[lo..hi]
    };
    let t2 = threshold * threshold;
    let mut taken: std::collections::HashSet<VertId> = std::collections::HashSet::new();
    let mut ordered: Vec<VertId> = moved.to_vec();
    ordered.sort_unstable_by_key(|v| v.index());
    ordered.dedup();
    let mut groups: Vec<Vec<VertId>> = Vec::new();
    for v in ordered {
        if taken.contains(&v) {
            continue;
        }
        let Some(p) = mesh.position(v) else { continue };
        let k = key(p);
        let mut group = vec![v];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &(_, other) in lookup([k[0] + dx, k[1] + dy, k[2] + dz]) {
                        if other == v || taken.contains(&other) {
                            continue;
                        }
                        if mesh
                            .position(other)
                            .is_some_and(|q| (q - p).length_squared() <= t2)
                        {
                            group.push(other);
                        }
                    }
                }
            }
        }
        if group.len() > 1 {
            taken.extend(group.iter().copied());
            groups.push(group);
        }
    }
    merge_groups(mesh, &groups, MergeTarget::Centre)
}

/// Welds every vertex of the mesh.
pub fn weld_all(mesh: &mut BMesh, threshold: f32) -> Result<Merge, BmError> {
    let all: Vec<VertId> = mesh.vert_ids().collect();
    weld_by_distance(mesh, &all, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;
    use crate::SelectMode;

    #[test]
    fn a_vertex_moved_onto_another_merges_and_others_overlapping_stay() {
        // Two quads side by side, not sharing their middle edge.
        let mut m = BMesh::from_mesh_data(&crate::MeshData::from_faces(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        ));
        let ids: Vec<VertId> = m.vert_ids().collect();
        // Vertex 1 "moved": it merges with 4, and 2 and 7 stay two.
        let out = weld_onto(&mut m, &[ids[1]], 1e-5).unwrap();
        assert_eq!(out.removed, 1);
        assert_eq!(m.vert_ids().count(), 7);
        // Nothing near: nothing merges.
        let ids: Vec<VertId> = m.vert_ids().collect();
        assert_eq!(weld_onto(&mut m, &[ids[0]], 1e-5).unwrap().removed, 0);
    }

    #[test]
    fn merging_two_corners_of_a_quad_leaves_a_triangle() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let f = m.face_ids().next().unwrap();
        let ring = m.face_verts(f);

        let out = merge_groups(&mut m, &[vec![ring[0], ring[1]]], MergeTarget::Centre).unwrap();
        m.assert_valid();

        assert_eq!(out.removed, 1);
        assert_eq!(out.faces_dropped, 0);
        assert_eq!(m.num_verts(), 3);
        assert_eq!(m.num_faces(), 1);
        assert_eq!(m.face(m.face_ids().next().unwrap()).unwrap().len(), 3);
    }

    #[test]
    fn merging_opposite_corners_drops_the_bowtie_rather_than_inventing_a_face() {
        // A quad whose opposite corners merge repeats a vertex, which no valid
        // face may do. Splitting it into two triangles would be a guess about
        // which pair the user meant.
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let f = m.face_ids().next().unwrap();
        let ring = m.face_verts(f);

        let out = merge_groups(&mut m, &[vec![ring[0], ring[2]]], MergeTarget::Centre).unwrap();
        m.assert_valid();

        assert_eq!(out.faces_dropped, 1);
        assert_eq!(m.num_faces(), 0);
        assert_eq!(out.removed, 1);
    }

    #[test]
    fn merging_a_whole_face_removes_it() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let ring = m.face_verts(m.face_ids().next().unwrap());
        let out = merge_groups(&mut m, &[ring], MergeTarget::Centre).unwrap();
        m.assert_valid();

        assert_eq!(out.removed, 3);
        assert_eq!(out.faces_dropped, 1, "nothing with area is left");
        assert_eq!(m.num_verts(), 1);
        assert_eq!(m.num_faces(), 0);
        assert_eq!(m.num_edges(), 0, "no wires left dangling");
    }

    #[test]
    fn the_survivor_lands_where_the_target_says() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let ring = m.face_verts(m.face_ids().next().unwrap());
        let a = m.position(ring[0]).unwrap();
        let b = m.position(ring[1]).unwrap();

        let mut centre = m.clone();
        merge_groups(&mut centre, &[vec![ring[0], ring[1]]], MergeTarget::Centre).unwrap();
        let p = centre.position(ring[0]).unwrap();
        assert!((p - (a + b) * 0.5).length() < 1e-5);

        let mut last = m.clone();
        merge_groups(&mut last, &[vec![ring[0], ring[1]]], MergeTarget::Last).unwrap();
        assert!((last.position(ring[1]).unwrap() - b).length() < 1e-5);
        assert!(last.vert(ring[0]).is_none(), "the earlier one went");

        merge_groups(&mut m, &[vec![ring[0], ring[1]]], MergeTarget::First).unwrap();
        assert!((m.position(ring[0]).unwrap() - a).length() < 1e-5);
    }

    #[test]
    fn merging_an_edge_of_a_cube_keeps_it_closed() {
        // Collapsing one edge of a cube gives a solid with two triangles where
        // the two adjacent quads were, and it must still be watertight.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let e = m.edge_ids().next().unwrap();
        let [a, b] = m.edge_verts(e).unwrap();

        let out = merge_groups(&mut m, &[vec![a, b]], MergeTarget::Centre).unwrap();
        m.assert_valid();

        assert_eq!(out.removed, 1);
        assert_eq!(m.num_verts(), 7);
        assert_eq!(m.num_faces(), 6, "no face collapsed entirely");
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "still watertight");
        }
        let tris = m
            .face_ids()
            .filter(|f| m.face(*f).unwrap().len() == 3)
            .count();
        assert_eq!(tris, 2, "the two faces on the collapsed edge");
    }

    // --- weld ---------------------------------------------------------------

    #[test]
    fn welding_joins_a_split_seam() {
        // The importer case: two quads that look joined but share no vertices.
        // Before welding, every modelling operation shatters the mesh.
        let mut m = BMesh::new();
        let a = m.add_vert(Vec3::new(0.0, 0.0, 0.0));
        let b = m.add_vert(Vec3::new(1.0, 0.0, 0.0));
        let c = m.add_vert(Vec3::new(1.0, 1.0, 0.0));
        let d = m.add_vert(Vec3::new(0.0, 1.0, 0.0));
        m.add_face(&[a, b, c, d]).unwrap();
        // A second quad sharing the b-c edge geometrically but not topologically.
        let b2 = m.add_vert(Vec3::new(1.0, 0.0, 0.0));
        let c2 = m.add_vert(Vec3::new(1.0, 1.0, 0.0));
        let e = m.add_vert(Vec3::new(2.0, 0.0, 0.0));
        let f = m.add_vert(Vec3::new(2.0, 1.0, 0.0));
        m.add_face(&[b2, e, f, c2]).unwrap();

        assert_eq!(m.num_verts(), 8);
        assert!(
            m.edge_ids().all(|e| m.edge_face_count(e) == 1),
            "nothing is joined yet"
        );

        let out = weld_all(&mut m, 1e-4).unwrap();
        m.assert_valid();

        assert_eq!(out.removed, 2);
        assert_eq!(out.faces_dropped, 0);
        assert_eq!(m.num_verts(), 6);
        assert_eq!(m.num_faces(), 2);
        assert_eq!(
            m.edge_ids().filter(|&e| m.edge_face_count(e) == 2).count(),
            1,
            "the seam is now one shared edge"
        );
    }

    #[test]
    fn welding_finds_pairs_that_straddle_a_cell_boundary() {
        // The bug a naive grid has: two points either side of a cell line are
        // within the threshold but land in different buckets, so whether they
        // weld depends on where the model happens to sit relative to the origin.
        let mut m = BMesh::new();
        let eps = 1e-3;
        for offset in [0.0f32, 1.0, 2.5, -7.25] {
            let a = m.add_vert(Vec3::new(offset, 0.0, 0.0));
            let b = m.add_vert(Vec3::new(offset + eps * 0.4, 0.0, 0.0));
            let c = m.add_vert(Vec3::new(offset, 1.0, 0.0));
            m.add_face(&[a, b, c]).unwrap();
        }
        assert_eq!(m.num_verts(), 12);

        let out = weld_all(&mut m, eps).unwrap();
        m.assert_valid();
        assert_eq!(
            out.groups, 4,
            "every pair must be found regardless of where it sits"
        );
        assert_eq!(out.removed, 4);
    }

    #[test]
    fn welding_below_the_threshold_changes_nothing() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        let out = weld_all(&mut m, 1e-6).unwrap();
        m.assert_valid();
        assert_eq!(out, Merge::default());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
    }

    #[test]
    fn a_zero_threshold_is_a_no_op_rather_than_welding_everything() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        assert_eq!(weld_all(&mut m, 0.0).unwrap(), Merge::default());
        assert_eq!(weld_all(&mut m, -1.0).unwrap(), Merge::default());
        assert_eq!(m.num_verts(), 8);
    }

    #[test]
    fn welding_is_the_same_on_every_run() {
        // The grouping decides the output geometry, so it must not depend on
        // iteration order. Ten runs of the same input, compared exactly.
        let build = || {
            let mut m = BMesh::from_mesh_data(&primitives::uv_sphere(1.0, 12, 6));
            let _ = weld_all(&mut m, 0.35);
            m.positions()
        };
        let first = build();
        for _ in 0..9 {
            assert_eq!(build(), first);
        }
    }

    #[test]
    fn a_welded_mesh_still_bakes_to_storage_form() {
        let mut m = BMesh::from_mesh_data(&primitives::uv_sphere(1.0, 16, 8));
        weld_all(&mut m, 0.25).unwrap();
        m.assert_valid();
        let baked = m.to_mesh_data();
        assert_eq!(baked.validate(), Vec::new());
    }

    #[test]
    fn merging_the_selection_goes_through_the_same_path() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let ring = m.face_verts(m.face_ids().next().unwrap());
        m.set_vert_selected(ring[0], true);
        m.set_vert_selected(ring[1], true);
        m.flush_selection(SelectMode::Vert);

        let out = merge_selected(&mut m, MergeTarget::Centre).unwrap();
        assert_eq!(out.removed, 1);
        assert_eq!(m.num_verts(), 3);
        assert!(
            m.selected_verts().is_empty(),
            "the selection is stale, so cleared"
        );
        m.assert_valid();
    }

    #[test]
    fn merging_nothing_is_a_no_op() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        assert_eq!(
            merge_groups(&mut m, &[], MergeTarget::Centre).unwrap(),
            Merge::default()
        );
        let one = vec![m.vert_ids().next().unwrap()];
        assert_eq!(
            merge_groups(&mut m, &[one], MergeTarget::Centre).unwrap(),
            Merge::default(),
            "a group of one has nothing to merge with"
        );
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
        m.assert_valid();
    }
}
