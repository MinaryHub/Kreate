//! Face regions: connectivity and boundary.
//!
//! Shared by every operator that works on "the selected faces" as one thing
//! rather than face by face — extrude, dissolve, and later inset and bridge.
//! Kept in one place because the definition of *boundary* is the whole
//! difference between the region behaviour and the per-face behaviour, and two
//! operators disagreeing about it is a bug nobody would think to look for.

use crate::arena::{ArenaIndex, EdgeId, FaceId, VertId};
use crate::bmesh::BMesh;
use crate::Domain;

/// A face-region membership test, by slot.
///
/// A linear scan over the region would make every caller quadratic, which shows
/// up the first time someone selects a whole character rather than one face.
pub struct Region {
    member: Vec<bool>,
    pub faces: Vec<FaceId>,
}

impl Region {
    pub fn new(mesh: &BMesh, faces: &[FaceId]) -> Self {
        let mut member = vec![false; mesh.slot_count(Domain::Face)];
        let mut kept = Vec::with_capacity(faces.len());
        for &f in faces {
            let slot = f.index() as usize;
            if mesh.face(f).is_some() && !member[slot] {
                member[slot] = true;
                kept.push(f);
            }
        }
        Self {
            member,
            faces: kept,
        }
    }

    #[inline]
    pub fn contains(&self, f: FaceId) -> bool {
        self.member
            .get(f.index() as usize)
            .copied()
            .unwrap_or(false)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// How many of an edge's faces are in the region.
    ///
    /// One means the edge is on the outline. Zero is impossible for an edge of a
    /// region face; two or more means it is interior — or, on non-manifold
    /// geometry, that three region faces meet there, which is why this counts
    /// rather than returning a bool.
    pub fn faces_on(&self, mesh: &BMesh, e: EdgeId) -> usize {
        mesh.edge_faces(e)
            .into_iter()
            .filter(|f| self.contains(*f))
            .count()
    }

    /// The region's outline, as directed edges taken from each face's winding.
    ///
    /// Direction matters: a wall or a replacement n-gon built from these must be
    /// wound consistently with the faces it came from, and reading the direction
    /// off the edge instead gives an arbitrary orientation.
    pub fn boundary_edges(&self, mesh: &BMesh) -> Vec<(VertId, VertId)> {
        let mut out = Vec::new();
        for &f in &self.faces {
            for l in mesh.face_loops(f) {
                let (Some(a), Some(e)) = (mesh.loop_vert(l), mesh.loop_edge(l)) else {
                    continue;
                };
                if self.faces_on(mesh, e) != 1 {
                    continue;
                }
                let Some(b) = mesh.loop_next(l).and_then(|n| mesh.loop_vert(n)) else {
                    continue;
                };
                out.push((a, b));
            }
        }
        out
    }

    /// The outline as ordered vertex cycles, or `None` if it does not form any.
    ///
    /// `None` means a pinch: a vertex the outline passes through twice, which
    /// happens when a region touches itself at a corner. There is no single
    /// correct way to chain such an outline, so the caller is told rather than
    /// handed one of the two arbitrary answers — a dissolve that silently picks
    /// wrong produces a face turned inside out through the middle of a model.
    pub fn boundary_loops(&self, mesh: &BMesh) -> Option<Vec<Vec<VertId>>> {
        let n = mesh.slot_count(Domain::Point);
        let mut succ: Vec<Option<VertId>> = vec![None; n];
        let mut starts: Vec<VertId> = Vec::new();
        for (a, b) in self.boundary_edges(mesh) {
            let slot = a.index() as usize;
            if slot >= n || succ[slot].is_some() {
                return None;
            }
            succ[slot] = Some(b);
            starts.push(a);
        }
        if starts.is_empty() {
            return Some(Vec::new());
        }
        starts.sort_unstable_by_key(|v| v.index());

        let mut visited = vec![false; n];
        let mut loops = Vec::new();
        for &start in &starts {
            if visited[start.index() as usize] {
                continue;
            }
            let mut cycle = Vec::new();
            let mut v = start;
            // Bounded by the outline's own length: a corrupt `succ` chain must
            // not hang the editor.
            for _ in 0..=starts.len() {
                visited[v.index() as usize] = true;
                cycle.push(v);
                v = succ.get(v.index() as usize).copied().flatten()?;
                if v == start {
                    break;
                }
            }
            if v != start || cycle.len() < 3 {
                return None;
            }
            loops.push(cycle);
        }
        Some(loops)
    }

    /// Splits the region into groups of faces joined through shared edges.
    ///
    /// Dissolving two separate patches must produce two n-gons, not one face
    /// that leaps across the gap between them. Breadth-first from the
    /// lowest-numbered face, so the grouping is the same on every run.
    pub fn components(&self, mesh: &BMesh) -> Vec<Vec<FaceId>> {
        let mut seen = vec![false; mesh.slot_count(Domain::Face)];
        let mut out = Vec::new();
        for &seed in &self.faces {
            if seen[seed.index() as usize] {
                continue;
            }
            let mut group = Vec::new();
            let mut queue = vec![seed];
            seen[seed.index() as usize] = true;
            while let Some(f) = queue.pop() {
                group.push(f);
                for l in mesh.face_loops(f) {
                    let Some(e) = mesh.loop_edge(l) else { continue };
                    for g in mesh.edge_faces(e) {
                        let slot = g.index() as usize;
                        if self.contains(g) && !seen[slot] {
                            seen[slot] = true;
                            queue.push(g);
                        }
                    }
                }
            }
            group.sort_unstable_by_key(|f| f.index());
            out.push(group);
        }
        out
    }

    /// Every vertex the region uses, in slot order.
    pub fn verts(&self, mesh: &BMesh) -> Vec<VertId> {
        let mut out: Vec<VertId> = self
            .faces
            .iter()
            .flat_map(|f| mesh.face_verts(*f))
            .collect();
        out.sort_unstable_by_key(|v| v.index());
        out.dedup();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_grid_region_knows_its_outline() {
        let m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        let all: Vec<FaceId> = m.face_ids().collect();
        let r = Region::new(&m, &all);

        assert_eq!(r.boundary_edges(&m).len(), 8);
        let loops = r.boundary_loops(&m).expect("a grid outline is one cycle");
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 8);
    }

    #[test]
    fn a_closed_solid_has_no_outline() {
        let m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let all: Vec<FaceId> = m.face_ids().collect();
        let r = Region::new(&m, &all);
        assert!(r.boundary_edges(&m).is_empty());
        assert_eq!(r.boundary_loops(&m), Some(Vec::new()));
    }

    #[test]
    fn a_region_with_a_hole_reports_two_cycles() {
        // The ring of eight faces around the centre of a 3x3 grid: an outer
        // outline and an inner one. An operator that assumes a single cycle
        // would weld the hole shut without being asked.
        let m = BMesh::from_mesh_data(&primitives::grid(6.0, 3, 3));
        let faces: Vec<FaceId> = m.face_ids().collect();
        assert_eq!(faces.len(), 9);
        // The middle face is the one whose every edge has two faces.
        let middle = faces
            .iter()
            .copied()
            .find(|f| {
                m.face_loops(*f).iter().all(|l| {
                    m.loop_edge(*l)
                        .map(|e| m.edge_face_count(e) == 2)
                        .unwrap_or(false)
                })
            })
            .unwrap();
        let ring: Vec<FaceId> = faces.into_iter().filter(|f| *f != middle).collect();

        let r = Region::new(&m, &ring);
        let loops = r.boundary_loops(&m).expect("both cycles chain cleanly");
        assert_eq!(loops.len(), 2, "an outer outline and the hole");
        let mut lens: Vec<usize> = loops.iter().map(|l| l.len()).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![4, 12]);
    }

    #[test]
    fn separate_patches_are_separate_components() {
        let m = BMesh::from_mesh_data(&primitives::grid(6.0, 3, 3));
        let faces: Vec<FaceId> = m.face_ids().collect();
        // Two opposite corners of the grid share neither an edge nor anything
        // else the walk can cross.
        let pair = vec![faces[0], faces[8]];
        let r = Region::new(&m, &pair);
        let parts = r.components(&m);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].len(), 1);
        assert_eq!(parts[1].len(), 1);

        let joined = Region::new(&m, &[faces[0], faces[1]]);
        assert_eq!(joined.components(&m).len(), 1, "these share an edge");
    }

    #[test]
    fn a_pinched_outline_is_refused_rather_than_guessed() {
        // Two faces of a grid meeting only at a corner: the outline passes
        // through that vertex twice, and there is no non-arbitrary way to chain
        // it. Reporting the ambiguity is the whole point — picking one produces
        // a face turned inside out through the middle of the model.
        let m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        let faces: Vec<FaceId> = m.face_ids().collect();

        // Pick the two faces that share only a vertex.
        let mut pinched = None;
        for &a in &faces {
            for &b in &faces {
                if a.index() >= b.index() {
                    continue;
                }
                let shares_edge = m.face_loops(a).iter().any(|l| {
                    m.loop_edge(*l)
                        .map(|e| m.edge_faces(e).contains(&b))
                        .unwrap_or(false)
                });
                let shares_vert = m.face_verts(a).iter().any(|v| m.face_verts(b).contains(v));
                if !shares_edge && shares_vert {
                    pinched = Some(vec![a, b]);
                }
            }
        }
        let pinched = pinched.expect("a 2x2 grid has a diagonal pair");
        let r = Region::new(&m, &pinched);
        assert_eq!(
            r.boundary_loops(&m),
            None,
            "the outline visits the shared corner twice"
        );
        assert_eq!(r.components(&m).len(), 2, "they are still two components");
    }

    #[test]
    fn duplicates_in_the_input_are_ignored() {
        let m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let f = m.face_ids().next().unwrap();
        let r = Region::new(&m, &[f, f, f]);
        assert_eq!(r.faces.len(), 1);
        assert_eq!(r.boundary_edges(&m).len(), 4);
    }
}
