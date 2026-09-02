//! Edge loops and edge rings.
//!
//! These are the operators the plan singles out as the real test of the
//! structure: neither can be written without the radial cycle, and both go
//! wrong quietly on anything but a perfect quad grid. If loop select is right,
//! the topology is right.
//!
//! ```text
//!   ┌───┬───┬───┐        loop: along the edges, across vertices
//!   │   │   │   │        ring: across the faces, between parallel edges
//!   ├───┼───┼───┤
//!   │   │   │   │        ═══  an edge loop
//!   ├═══┼═══┼═══┤        ║ ║  the ring it belongs to
//!   │   │   │   │
//!   └───┴───┴───┘
//! ```
//!
//! # Where they stop
//!
//! Both walks stop at irregularity rather than guessing through it — a pole, a
//! triangle, a boundary, a non-manifold fan. That is not a limitation to be
//! fixed later: a loop that continues past a pole picks one of several equally
//! defensible continuations, and the user discovers the wrong choice only after
//! the loop cut has already gone somewhere they did not want.

use crate::arena::{EdgeId, FaceId, LoopId, VertId};
use crate::bmesh::BMesh;

/// Whether two edges share a face.
fn share_a_face(mesh: &BMesh, a: EdgeId, b: EdgeId) -> bool {
    let fa = mesh.edge_faces(a);
    mesh.edge_faces(b).into_iter().any(|f| fa.contains(&f))
}

/// The next edge of a loop, leaving `edge` through `vert`.
///
/// The rule for a regular interior vertex: of the four edges there, one shares
/// no face with the edge we arrived on, and that is the continuation. Any other
/// valence has no such edge, or more than one, and the walk ends — which is why
/// this returns an `Option` rather than picking.
fn loop_step(mesh: &BMesh, edge: EdgeId, vert: VertId) -> Option<EdgeId> {
    let mut candidates = mesh
        .vert_edges(vert)
        .into_iter()
        .filter(|&e| e != edge && !share_a_face(mesh, e, edge));
    let next = candidates.next()?;
    // More than one continuation means the vertex is a pole and there is no
    // non-arbitrary answer.
    if candidates.next().is_some() {
        return None;
    }
    Some(next)
}

/// One direction of a walk: what it found, and whether it came back to the
/// start.
///
/// The flag is not a detail. A closed loop is reachable from both ends, so
/// walking both directions and concatenating lists every edge twice — and the
/// obvious guard, comparing where the two halves ended, does not fire because
/// they end at *opposite* ends of the same cycle. Asking the first walk whether
/// it closed is the only test that works for both open and closed loops.
#[derive(Default)]
struct Walk {
    edges: Vec<EdgeId>,
    closed: bool,
}

/// Joins the two directions of a walk into one ordered list including `start`.
fn join(mut backward: Walk, forward: Walk, start: EdgeId) -> Vec<EdgeId> {
    if backward.closed {
        // One direction already went all the way round.
        let mut out = Vec::with_capacity(backward.edges.len() + 1);
        out.push(start);
        out.append(&mut backward.edges);
        return out;
    }
    backward.edges.reverse();
    backward.edges.push(start);
    backward.edges.extend(forward.edges);
    backward.edges
}

/// The edge loop through `start`, in walk order.
///
/// Always contains `start`, and never lists an edge twice.
pub fn edge_loop(mesh: &BMesh, start: EdgeId) -> Vec<EdgeId> {
    let Some([a, b]) = mesh.edge_verts(start) else {
        return Vec::new();
    };
    let backward = walk_loop(mesh, start, a);
    if backward.closed {
        return join(backward, Walk::default(), start);
    }
    let forward = walk_loop(mesh, start, b);
    join(backward, forward, start)
}

/// Walks one direction, not including `start`.
fn walk_loop(mesh: &BMesh, start: EdgeId, from: VertId) -> Walk {
    let mut out = Walk::default();
    let mut edge = start;
    let mut vert = from;
    // Bounded by the edge count: a corrupt disk or radial cycle must not hang
    // the editor, and this walk runs on whatever the user has open.
    for _ in 0..mesh.num_edges() {
        let Some(next) = loop_step(mesh, edge, vert) else {
            break;
        };
        if next == start {
            out.closed = true;
            break;
        }
        let Some(v) = mesh.edge_other_vert(next, vert) else {
            break;
        };
        out.edges.push(next);
        edge = next;
        vert = v;
    }
    out
}

/// The edge opposite `edge` in a quad.
///
/// Only defined for quads: on a triangle or an n-gon there is no opposite edge,
/// and that is exactly where a ring must stop.
fn opposite_in_quad(mesh: &BMesh, face: FaceId, edge: EdgeId) -> Option<EdgeId> {
    if mesh.face(face)?.len() != 4 {
        return None;
    }
    let entry: LoopId = mesh
        .face_loops(face)
        .into_iter()
        .find(|&l| mesh.loop_edge(l) == Some(edge))?;
    let across = mesh.loop_next(mesh.loop_next(entry)?)?;
    mesh.loop_edge(across)
}

/// The edge ring through `start`, in walk order.
///
/// Steps across quads rather than along edges, so it collects the parallel edges
/// a loop cut would slide between.
pub fn edge_ring(mesh: &BMesh, start: EdgeId) -> Vec<EdgeId> {
    let faces = mesh.edge_faces(start);
    let backward = match faces.first() {
        Some(&f) => walk_ring(mesh, start, f),
        None => Walk::default(),
    };
    if backward.closed {
        return join(backward, Walk::default(), start);
    }
    let forward = match faces.get(1) {
        Some(&f) => walk_ring(mesh, start, f),
        None => Walk::default(),
    };
    join(backward, forward, start)
}

/// Walks one direction across faces, not including `start`.
fn walk_ring(mesh: &BMesh, start: EdgeId, from: FaceId) -> Walk {
    let mut out = Walk::default();
    let mut edge = start;
    let mut face = from;
    for _ in 0..mesh.num_edges() {
        let Some(next) = opposite_in_quad(mesh, face, edge) else {
            break;
        };
        if next == start {
            out.closed = true;
            break;
        }
        out.edges.push(next);
        // Across to the face on the other side. Exactly one other face, or the
        // ring has reached a boundary or a non-manifold fan and stops.
        let others: Vec<FaceId> = mesh
            .edge_faces(next)
            .into_iter()
            .filter(|&f| f != face)
            .collect();
        let [other] = others[..] else { break };
        edge = next;
        face = other;
    }
    out
}

/// The faces a ring passes through, in walk order.
///
/// What a face loop select gives, and what a loop cut needs in order to know
/// which faces it is splitting.
pub fn face_loop(mesh: &BMesh, start: EdgeId) -> Vec<FaceId> {
    let ring = edge_ring(mesh, start);
    let mut out = Vec::new();
    let add_between = |a: EdgeId, b: EdgeId, out: &mut Vec<FaceId>| {
        let fa = mesh.edge_faces(a);
        for f in mesh.edge_faces(b) {
            if fa.contains(&f) && !out.contains(&f) {
                out.push(f);
            }
        }
    };
    for pair in ring.windows(2) {
        add_between(pair[0], pair[1], &mut out);
    }
    // A ring that went all the way round has one more face than it has gaps
    // between consecutive edges: the one closing the cycle. Without this, a face
    // loop around a cylinder is one face short, and the missing face is the one
    // the walk started from — so it looks like an off-by-one in the wrong place.
    if let (Some(&first), Some(&last)) = (ring.first(), ring.last()) {
        if ring.len() > 2 {
            add_between(last, first, &mut out);
        }
    }
    out
}

impl BMesh {
    /// Adds the edge loop through `edge` to the selection.
    pub fn select_edge_loop(&mut self, edge: EdgeId) -> usize {
        let found = edge_loop(self, edge);
        for e in &found {
            self.set_edge_selected(*e, true);
        }
        self.flush_selection(crate::SelectMode::Edge);
        found.len()
    }

    /// Adds the edge ring through `edge` to the selection.
    pub fn select_edge_ring(&mut self, edge: EdgeId) -> usize {
        let found = edge_ring(self, edge);
        for e in &found {
            self.set_edge_selected(*e, true);
        }
        self.flush_selection(crate::SelectMode::Edge);
        found.len()
    }

    /// Adds the face loop through `edge` to the selection.
    pub fn select_face_loop(&mut self, edge: EdgeId) -> usize {
        let found = face_loop(self, edge);
        for f in &found {
            self.set_face_selected(*f, true);
        }
        self.flush_selection(crate::SelectMode::Face);
        found.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;
    use kreate_math::Vec3;

    /// Whether two edges of a face touch each other end to end.
    fn share_a_face_ends(mesh: &BMesh, a: EdgeId, b: EdgeId) -> bool {
        let Some(va) = mesh.edge_verts(a) else {
            return false;
        };
        let Some(vb) = mesh.edge_verts(b) else {
            return false;
        };
        va.iter().any(|x| vb.contains(x))
    }

    /// An edge of a grid running along X, at the given row.
    fn grid_edge_along_x(mesh: &BMesh, y: f32) -> EdgeId {
        mesh.edge_ids()
            .find(|&e| {
                let Some([a, b]) = mesh.edge_verts(e) else {
                    return false;
                };
                let (Some(pa), Some(pb)) = (mesh.position(a), mesh.position(b)) else {
                    return false;
                };
                (pa.y - y).abs() < 1e-4 && (pb.y - y).abs() < 1e-4 && (pa.x - pb.x).abs() > 1e-4
            })
            .expect("no edge along x at that row")
    }

    #[test]
    fn a_loop_runs_the_width_of_a_grid() {
        // A 4x4 grid: an interior edge running along X continues across three
        // more, giving a loop of four.
        let m = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
        let e = grid_edge_along_x(&m, 0.0);
        let found = edge_loop(&m, e);

        assert_eq!(found.len(), 4, "one edge per column");
        assert!(found.contains(&e));
        for &g in &found {
            let [a, b] = m.edge_verts(g).unwrap();
            assert!(
                (m.position(a).unwrap().y).abs() < 1e-4 && (m.position(b).unwrap().y).abs() < 1e-4,
                "the loop wandered off its row"
            );
        }
    }

    #[test]
    fn a_loop_on_a_cylinder_closes_on_itself() {
        // A closed loop is the case that lists every edge twice if the two
        // half-walks are not noticed to have met.
        let m = BMesh::from_mesh_data(&primitives::cylinder(1.0, 2.0, 12));
        // A vertical edge of the side wall, whose loop runs around the tube.
        let side = m
            .edge_ids()
            .find(|&e| {
                let [a, b] = m.edge_verts(e).unwrap();
                let (pa, pb) = (m.position(a).unwrap(), m.position(b).unwrap());
                (pa.z - pb.z).abs() > 0.5
            })
            .expect("a cylinder has vertical edges");

        let ring = edge_ring(&m, side);
        assert_eq!(ring.len(), 12, "once around the tube, each edge once");
        let mut sorted = ring.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ring.len(), "no edge listed twice");
    }

    #[test]
    fn a_ring_crosses_the_faces_a_loop_runs_along() {
        let m = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
        let e = grid_edge_along_x(&m, 0.0);
        let ring = edge_ring(&m, e);
        // Four rows of the grid, so the ring perpendicular to this edge has five
        // edges: the row lines above and below it, all the way out.
        assert_eq!(ring.len(), 5);
        assert!(ring.contains(&e));

        let faces = face_loop(&m, e);
        assert_eq!(faces.len(), 4, "one face between each pair of ring edges");
    }

    #[test]
    fn a_face_loop_around_a_cylinder_is_not_one_face_short() {
        // The gap-counting bug: N edges in a closed ring have N gaps, not N-1,
        // and the one that closes the cycle is the face the walk started from.
        // Off by one, and always the same face, which makes it look like a bug
        // somewhere else entirely.
        let m = BMesh::from_mesh_data(&primitives::cylinder(1.0, 2.0, 12));
        let side = m
            .edge_ids()
            .find(|&e| {
                let [a, b] = m.edge_verts(e).unwrap();
                let (pa, pb) = (m.position(a).unwrap(), m.position(b).unwrap());
                (pa.z - pb.z).abs() > 0.5
            })
            .unwrap();

        let ring = edge_ring(&m, side);
        let faces = face_loop(&m, side);
        assert_eq!(ring.len(), 12);
        assert_eq!(faces.len(), 12, "one face per gap, and the cycle has 12");
    }

    #[test]
    fn a_loop_stops_at_a_pole_rather_than_guessing() {
        // The reason both walks return early. A pole has several equally
        // defensible continuations, and picking one sends a later loop cut
        // somewhere the user did not ask for — discovered long after the fact.
        let m = BMesh::from_mesh_data(&primitives::uv_sphere(1.0, 8, 4));
        // A meridian edge next to the pole: the loop along it must stop there.
        let near_pole = m
            .edge_ids()
            .find(|&e| {
                let [a, b] = m.edge_verts(e).unwrap();
                m.vert_edge_count(a) == 8 || m.vert_edge_count(b) == 8
            })
            .expect("a uv sphere has a pole");

        let found = edge_loop(&m, near_pole);
        assert!(
            found.len() < m.num_edges(),
            "the walk must not run away across the whole sphere"
        );
        assert!(found.contains(&near_pole));
    }

    #[test]
    fn a_ring_stops_at_a_triangle() {
        // A cone's cap is a fan of triangles. There is no opposite edge in a
        // triangle, so the ring ends there rather than choosing one.
        let m = BMesh::from_mesh_data(&primitives::cone(1.0, 2.0, 8));
        for e in m.edge_ids() {
            let ring = edge_ring(&m, e);
            assert!(
                ring.len() <= m.num_edges(),
                "the walk must terminate on every edge of a cone"
            );
            assert!(ring.contains(&e));
        }
    }

    #[test]
    fn a_boundary_edge_is_its_own_loop_or_runs_the_border() {
        let m = BMesh::from_mesh_data(&primitives::plane(2.0));
        for e in m.edge_ids() {
            let found = edge_loop(&m, e);
            assert!(found.contains(&e));
            assert!(found.len() <= 4);
            let ring = edge_ring(&m, e);
            assert_eq!(
                ring.len(),
                2,
                "a ring through one quad is the edge and the one opposite it"
            );
            assert!(ring.contains(&e));
            // And the two really are parallel, rather than adjacent.
            let other = *ring.iter().find(|&&g| g != e).unwrap();
            assert!(
                !share_a_face_ends(&m, e, other),
                "the ring picked an adjacent edge, not the opposite one"
            );
        }
    }

    #[test]
    fn a_non_manifold_fan_stops_a_ring() {
        // Three faces on one edge: there is no single "other side" to cross to,
        // so the walk ends instead of choosing arbitrarily.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 1));
        let shared = m
            .edge_ids()
            .find(|&e| m.edge_face_count(e) == 2)
            .expect("the two quads share an edge");
        let [a, b] = m.edge_verts(shared).unwrap();
        let c = m.add_vert(Vec3::new(0.0, 0.0, 2.0));
        let d = m.add_vert(Vec3::new(1.0, 0.0, 2.0));
        m.add_face(&[a, b, d, c]).unwrap();
        assert_eq!(m.edge_face_count(shared), 3);

        let ring = edge_ring(&m, shared);
        assert!(
            ring.len() <= 3,
            "the ring must not wander through the fan: {ring:?}"
        );
        m.assert_valid();
    }

    #[test]
    fn selecting_a_loop_flushes_to_its_vertices() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
        let e = grid_edge_along_x(&m, 0.0);
        let n = m.select_edge_loop(e);

        assert_eq!(n, 4);
        assert_eq!(m.selected_edges().len(), 4);
        assert_eq!(m.selected_verts().len(), 5, "five vertices span four edges");
        assert!(m.selected_faces().is_empty(), "a line is not a face");
    }

    #[test]
    fn selecting_a_face_loop_selects_the_strip() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 4, 4));
        let e = grid_edge_along_x(&m, 0.0);
        let n = m.select_face_loop(e);
        assert_eq!(n, 4);
        assert_eq!(m.selected_faces().len(), 4);
        m.assert_valid();
    }

    #[test]
    fn every_edge_of_every_primitive_terminates() {
        // The property that matters more than any individual answer: these walks
        // run on whatever the user has open, including geometry no one designed
        // them for.
        let meshes = [
            primitives::cube(2.0),
            primitives::grid(4.0, 5, 3),
            primitives::uv_sphere(1.0, 9, 5),
            primitives::ico_sphere(1.0, 1),
            primitives::cylinder(1.0, 2.0, 7),
            primitives::cone(1.0, 2.0, 5),
            primitives::torus(1.0, 0.4, 9, 6),
        ];
        for mesh in meshes {
            let m = BMesh::from_mesh_data(&mesh);
            for e in m.edge_ids() {
                let l = edge_loop(&m, e);
                let r = edge_ring(&m, e);
                assert!(l.contains(&e) && r.contains(&e));
                assert!(l.len() <= m.num_edges() && r.len() <= m.num_edges());
                let mut sorted = l.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), l.len(), "loop listed an edge twice");
                let mut sorted = r.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), r.len(), "ring listed an edge twice");
            }
        }
    }
}
