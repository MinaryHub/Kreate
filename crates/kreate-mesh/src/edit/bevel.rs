//! Bevel: replacing an edge with a face.
//!
//! The plan calls this the hardest of the core operators and pins its scope down
//! precisely: **edge bevel, one segment, no profile, simple miter**. Several
//! segments, a profile curve, miter types and general self-intersection handling
//! are all deferred. What is here has to be right for the ordinary cases and
//! honest about the rest.
//!
//! # The vertex is the whole problem
//!
//! Placing the new quad along an edge is arithmetic. What makes bevel hard is
//! what happens where several edges meet, and the answer is not one rule but
//! three — chosen per *face corner*, by how many of **that face's own** two edges
//! at the vertex are being bevelled:
//!
//! - **Both** — the corner insets on both sides and becomes one point. Chamfer
//!   every edge of a cube and this is every corner of every original face, which
//!   is why those faces stay quads instead of becoming octagons.
//! - **One** — the corner insets away from that edge only, along the other one,
//!   and becomes one point.
//! - **Neither, but some other edge at the vertex is bevelled** — the corner is
//!   *cut* into two points, one along each of its edges. This is the case that is
//!   easy to miss: bevel one edge of a cube and the third face at each end, whose
//!   own edges are untouched, still has to open up. Leaving it at the original
//!   vertex tears the mesh, because its neighbours have moved and no longer share
//!   an edge with it.
//!
//! The three rules agree wherever they meet, which is what lets the pieces close
//! up: the two-point form of one face uses exactly the points its neighbours
//! arrived at by the one-point rule.
//!
//! # What v1 refuses
//!
//! Boundary and non-manifold edges. A bevel needs two faces to put the new quad
//! between; with one there is nothing on the other side, and with three there is
//! no single answer. Both are skipped and counted rather than guessed at.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, EdgeId, FaceId, VertId};
use crate::bmesh::{BMesh, BmError, SelectMode};
use crate::Domain;

/// What a bevel produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bevel {
    /// The quads that replaced the edges. Part of the new selection.
    pub faces: Vec<FaceId>,
    /// The caps closing the corners where enough bevelled edges met to leave a
    /// hole.
    pub caps: Vec<FaceId>,
    /// Every vertex created.
    pub verts: Vec<VertId>,
    /// Edges skipped for want of exactly two faces.
    pub skipped: usize,
    /// True if the width had to be reduced to keep the offsets from crossing.
    pub clamped: bool,
}

/// A new vertex offset from `vertex` along `along`.
///
/// Shared between every face that arrives at it, which is what makes the pieces
/// meet rather than merely touch.
struct EdgePoint {
    vertex: VertId,
    along: EdgeId,
    created: VertId,
}

/// A new vertex inset from `vertex` on both sides, belonging to one face.
struct CornerPoint {
    vertex: VertId,
    face: FaceId,
    created: VertId,
}

/// Bevels the given edges by `width`, measured along the adjacent edges.
///
/// The offset is measured *along the neighbouring edge* rather than as a
/// perpendicular distance from the bevelled one. The two agree wherever corners
/// are square; on an oblique corner this reads as a slightly wider chamfer than
/// the number suggests. That is a v1 simplification, and the exact-distance
/// variant belongs with the profile work.
pub fn bevel_edges(mesh: &mut BMesh, edges: &[EdgeId], width: f32) -> Result<Bevel, BmError> {
    let mut out = Bevel::default();
    if width <= 0.0 {
        return Ok(out);
    }

    // --- what can be bevelled ----------------------------------------------
    let mut is_bevelled = vec![false; mesh.slot_count(Domain::Edge)];
    let mut chosen: Vec<EdgeId> = Vec::new();
    for &e in edges {
        if mesh.edge(e).is_none() {
            continue;
        }
        if mesh.edge_face_count(e) != 2 {
            out.skipped += 1;
            continue;
        }
        if !is_bevelled[e.index() as usize] {
            is_bevelled[e.index() as usize] = true;
            chosen.push(e);
        }
    }
    if chosen.is_empty() {
        return Ok(out);
    }
    let bevelled = |e: EdgeId| {
        is_bevelled
            .get(e.index() as usize)
            .copied()
            .unwrap_or(false)
    };

    let mut touched: Vec<VertId> = chosen
        .iter()
        .filter_map(|e| mesh.edge_verts(*e))
        .flatten()
        .collect();
    touched.sort_unstable_by_key(|v| v.index());
    touched.dedup();

    // --- how far it can go --------------------------------------------------
    // One width for the whole operation, clamped by the tightest place: an
    // offset longer than half an edge would pass the offset coming from the
    // other end and the quad between them would be inside out. A per-edge clamp
    // would keep the bevel legal and make it visibly uneven, which reads as a
    // bug rather than as a limit.
    let mut limit = f32::INFINITY;
    for &v in &touched {
        for g in mesh.vert_edges(v) {
            let Some([a, b]) = mesh.edge_verts(g) else {
                continue;
            };
            let (Some(pa), Some(pb)) = (mesh.position(a), mesh.position(b)) else {
                continue;
            };
            let length = (pb - pa).length();
            if length > 1e-6 {
                // Halved again for the doubly-inset corners: those move by the
                // width along *both* of a face's edges, so half an edge each way
                // is already the limit before they meet.
                limit = limit.min(length * 0.49 * 0.5);
            }
        }
    }
    let applied = width.min(limit);
    if applied < width {
        out.clamped = true;
    }

    // --- the plan ------------------------------------------------------------
    // Read in full before anything is written: every rule below reads the loop
    // cycles, and those stop existing the moment the old faces are removed.
    let mut affected: Vec<FaceId> = Vec::new();
    for &v in &touched {
        for f in mesh.vert_faces(v) {
            if !affected.contains(&f) {
                affected.push(f);
            }
        }
    }
    affected.sort_unstable_by_key(|f| f.index());

    /// Which of a face's corner replacements a vertex needs.
    enum Corner {
        /// One point, inset along the named edge.
        Along(EdgeId),
        /// One point, inset along both of the face's edges at this vertex.
        Both,
        /// Two points, one along each edge, in traversal order.
        Split(EdgeId, EdgeId),
        /// Untouched.
        Keep,
    }

    let mut plan: Vec<(FaceId, Vec<(VertId, Corner)>)> = Vec::with_capacity(affected.len());
    for &f in &affected {
        let mut corners = Vec::new();
        for l in mesh.face_loops(f) {
            let Some(v) = mesh.loop_vert(l) else { continue };
            let Some((incoming, outgoing)) = corner_edges(mesh, f, v) else {
                corners.push((v, Corner::Keep));
                continue;
            };
            let corner = match (bevelled(incoming), bevelled(outgoing)) {
                (true, true) => Corner::Both,
                // Inset away from the bevelled edge, which is a step along the
                // other one.
                (true, false) => Corner::Along(outgoing),
                (false, true) => Corner::Along(incoming),
                (false, false) if touched.contains(&v) => Corner::Split(incoming, outgoing),
                (false, false) => Corner::Keep,
            };
            corners.push((v, corner));
        }
        plan.push((f, corners));
    }

    // --- the new vertices ---------------------------------------------------
    let mut edge_points: Vec<EdgePoint> = Vec::new();
    let mut corner_points: Vec<CornerPoint> = Vec::new();
    let offset_along = |mesh: &BMesh, v: VertId, g: EdgeId| -> Option<Vec3> {
        let from = mesh.position(v)?;
        let to = mesh.position(mesh.edge_other_vert(g, v)?)?;
        let direction = (to - from).normalize_or_zero();
        (direction != Vec3::ZERO).then(|| direction * applied)
    };

    for (f, corners) in &plan {
        for (v, corner) in corners {
            match corner {
                Corner::Keep => {}
                Corner::Along(g) => {
                    if edge_points.iter().any(|p| p.vertex == *v && p.along == *g) {
                        continue;
                    }
                    let (Some(base), Some(step)) = (mesh.position(*v), offset_along(mesh, *v, *g))
                    else {
                        continue;
                    };
                    let created = mesh.add_vert(base + step);
                    out.verts.push(created);
                    edge_points.push(EdgePoint {
                        vertex: *v,
                        along: *g,
                        created,
                    });
                }
                Corner::Split(a, b) => {
                    for g in [a, b] {
                        if edge_points.iter().any(|p| p.vertex == *v && p.along == *g) {
                            continue;
                        }
                        let (Some(base), Some(step)) =
                            (mesh.position(*v), offset_along(mesh, *v, *g))
                        else {
                            continue;
                        };
                        let created = mesh.add_vert(base + step);
                        out.verts.push(created);
                        edge_points.push(EdgePoint {
                            vertex: *v,
                            along: *g,
                            created,
                        });
                    }
                }
                Corner::Both => {
                    let Some((incoming, outgoing)) = corner_edges(mesh, *f, *v) else {
                        continue;
                    };
                    let (Some(base), Some(a), Some(b)) = (
                        mesh.position(*v),
                        offset_along(mesh, *v, incoming),
                        offset_along(mesh, *v, outgoing),
                    ) else {
                        continue;
                    };
                    let created = mesh.add_vert(base + a + b);
                    out.verts.push(created);
                    corner_points.push(CornerPoint {
                        vertex: *v,
                        face: *f,
                        created,
                    });
                }
            }
        }
    }

    let along_point = |v: VertId, g: EdgeId| -> Option<VertId> {
        edge_points
            .iter()
            .find(|p| p.vertex == v && p.along == g)
            .map(|p| p.created)
    };
    let both_point = |v: VertId, f: FaceId| -> Option<VertId> {
        corner_points
            .iter()
            .find(|p| p.vertex == v && p.face == f)
            .map(|p| p.created)
    };

    // The rings for the rebuilt faces, and the single point each face presents
    // at each bevelled corner — which is what the walls are built from.
    let mut rebuilt: Vec<Vec<VertId>> = Vec::with_capacity(plan.len());
    for (f, corners) in &plan {
        let mut ring: Vec<VertId> = Vec::new();
        for (v, corner) in corners {
            match corner {
                Corner::Keep => ring.push(*v),
                Corner::Along(g) => ring.extend(along_point(*v, *g)),
                Corner::Both => ring.extend(both_point(*v, *f)),
                Corner::Split(a, b) => {
                    ring.extend(along_point(*v, *a));
                    ring.extend(along_point(*v, *b));
                }
            }
        }
        rebuilt.push(ring);
    }

    // A face with a bevelled edge at `v` presents exactly one point there, and
    // that is the wall's corner.
    let wall_corner = |mesh: &BMesh, f: FaceId, v: VertId, e: EdgeId| -> Option<VertId> {
        let (incoming, outgoing) = corner_edges(mesh, f, v)?;
        match (bevelled(incoming), bevelled(outgoing)) {
            (true, true) => both_point(v, f),
            (true, false) => along_point(v, outgoing),
            (false, true) => along_point(v, incoming),
            (false, false) => {
                let _ = e;
                None
            }
        }
    };

    /// Whether `face` walks `edge` from `from` toward its other end.
    fn walks_forward(mesh: &BMesh, face: FaceId, edge: EdgeId, from: VertId) -> bool {
        mesh.face_loops(face)
            .into_iter()
            .any(|l| mesh.loop_edge(l) == Some(edge) && mesh.loop_vert(l) == Some(from))
    }

    let mut walls: Vec<[VertId; 4]> = Vec::new();
    for &e in &chosen {
        let Some([a, b]) = mesh.edge_verts(e) else {
            continue;
        };
        let faces = mesh.edge_faces(e);
        let [f1, f2] = faces[..] else { continue };
        // Which face walks the edge forwards decides the wall's winding, and on
        // a consistently wound mesh exactly one of them does. Taking whichever
        // face the radial cycle happened to list first gives a wall that is
        // inside out half the time — and a wall lit from behind reads as a
        // shading fault, not a winding one.
        let (forward, backward) = if walks_forward(mesh, f1, e, a) {
            (f1, f2)
        } else {
            (f2, f1)
        };
        let (Some(near_a), Some(near_b), Some(far_a), Some(far_b)) = (
            wall_corner(mesh, forward, a, e),
            wall_corner(mesh, forward, b, e),
            wall_corner(mesh, backward, a, e),
            wall_corner(mesh, backward, b, e),
        ) else {
            continue;
        };
        // Against the forward face's direction along its own new edge, then with
        // the backward face's — which is what it means for all three to agree
        // across the edges they share.
        walls.push([near_b, near_a, far_a, far_b]);
    }

    // --- write --------------------------------------------------------------
    for &f in &affected {
        mesh.remove_face(f);
    }
    for ring in rebuilt {
        let mut cleaned = ring;
        cleaned.dedup();
        if cleaned.len() > 1 && cleaned.first() == cleaned.last() {
            cleaned.pop();
        }
        if cleaned.len() >= 3 {
            mesh.add_face(&cleaned)?;
        }
    }
    for wall in walls {
        let mut ring = wall.to_vec();
        ring.dedup();
        if ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() >= 3 {
            out.faces.push(mesh.add_face(&ring)?);
        }
    }

    // Whatever is left of the originals is litter.
    let mut stale: Vec<EdgeId> = touched
        .iter()
        .filter(|v| mesh.vert(**v).is_some())
        .flat_map(|v| mesh.vert_edges(*v))
        .collect();
    stale.sort_unstable_by_key(|e| e.index());
    stale.dedup();
    for e in stale {
        if mesh.edge(e).is_some() && mesh.is_wire_edge(e) {
            mesh.remove_edge(e);
        }
    }
    for &v in &touched {
        if mesh.vert(v).is_some() && mesh.vert_faces(v).is_empty() {
            mesh.remove_vert(v);
        }
    }

    // --- corners ------------------------------------------------------------
    // Where enough bevelled edges met, the walls and the shrunken faces leave a
    // hole. Found by chaining the border it left rather than constructed
    // directly: the border is already there, and `fill` knows how to wind a face
    // to agree with its neighbours. Two points leave no hole — a pair of points
    // is not a polygon — which is why bevelling one edge, or a ring of parallel
    // ones, needs no caps at all.
    for &v in &touched {
        let ring: Vec<VertId> = edge_points
            .iter()
            .filter(|p| p.vertex == v)
            .map(|p| p.created)
            .chain(
                corner_points
                    .iter()
                    .filter(|p| p.vertex == v)
                    .map(|p| p.created),
            )
            .filter(|w| mesh.vert(*w).is_some())
            .collect();
        if ring.len() < 3 {
            continue;
        }
        let mut border: Vec<EdgeId> = ring
            .iter()
            .flat_map(|w| mesh.vert_edges(*w))
            .filter(|e| {
                mesh.edge_face_count(*e) == 1
                    && mesh
                        .edge_verts(*e)
                        .is_some_and(|[x, y]| ring.contains(&x) && ring.contains(&y))
            })
            .collect();
        border.sort_unstable_by_key(|e| e.index());
        border.dedup();
        if border.len() < 3 {
            continue;
        }
        let Some(chains) = super::bridge::chain_borders(mesh, &border) else {
            continue;
        };
        let [cycle] = &chains[..] else { continue };
        if cycle.len() < 3 {
            continue;
        }
        let cycle = cycle.clone();
        if let Ok(cap) = super::fill::fill(mesh, &cycle) {
            out.caps.extend(cap.faces);
        }
    }

    mesh.select_none();
    for f in out.faces.iter().chain(&out.caps) {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(SelectMode::Face);
    Ok(out)
}

/// The two edges of `face` that meet at `vertex`, as `(incoming, outgoing)`.
fn corner_edges(mesh: &BMesh, face: FaceId, vertex: VertId) -> Option<(EdgeId, EdgeId)> {
    let l = mesh
        .face_loops(face)
        .into_iter()
        .find(|&l| mesh.loop_vert(l) == Some(vertex))?;
    let previous = mesh.loop_prev(l)?;
    Some((mesh.loop_edge(previous)?, mesh.loop_edge(l)?))
}

/// Bevels the selected edges.
pub fn bevel_selected(mesh: &mut BMesh, width: f32) -> Result<Bevel, BmError> {
    let edges = mesh.selected_edges();
    bevel_edges(mesh, &edges, width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn cube() -> BMesh {
        BMesh::from_mesh_data(&primitives::cube(2.0))
    }

    /// Every edge has two faces, i.e. the solid is still closed.
    fn watertight(mesh: &BMesh) -> bool {
        mesh.edge_ids().all(|e| mesh.edge_face_count(e) == 2)
    }

    fn euler(mesh: &BMesh) -> i64 {
        mesh.num_verts() as i64 - mesh.num_edges() as i64 + mesh.num_faces() as i64
    }

    #[test]
    fn bevelling_one_edge_chamfers_it() {
        // A plane cut across one edge. The two faces along it keep four corners;
        // the face at each *end* opens from a corner into an edge, becoming a
        // pentagon — that is the third rule, and the case that tears the mesh if
        // it is missed.
        let mut m = cube();
        let e = m.edge_ids().next().unwrap();
        let out = bevel_edges(&mut m, &[e], 0.3).unwrap();
        m.assert_valid();

        assert_eq!(out.skipped, 0);
        assert!(!out.clamped);
        assert_eq!(out.faces.len(), 1, "one edge, one quad");
        assert!(out.caps.is_empty(), "a single chamfer leaves no hole");
        assert_eq!(m.num_faces(), 7, "six faces plus the chamfer");
        assert_eq!(m.num_verts(), 10, "each end became two points");
        assert_eq!(m.num_edges(), 15);
        assert!(watertight(&m));
        assert_eq!(euler(&m), 2);

        let pentagons = m
            .face_ids()
            .filter(|f| m.face(*f).unwrap().len() == 5)
            .count();
        assert_eq!(pentagons, 2, "the face at each end opened up");
    }

    #[test]
    fn the_new_quad_sits_where_the_edge_was_and_faces_outward() {
        // A reversed chamfer still draws and still validates; it is lit from
        // inside, which the user reports as a shading bug.
        let mut m = cube();
        let e = m.edge_ids().next().unwrap();
        let [a, b] = m.edge_verts(e).unwrap();
        let midpoint = (m.position(a).unwrap() + m.position(b).unwrap()) * 0.5;

        let out = bevel_edges(&mut m, &[e], 0.3).unwrap();
        let ring: Vec<Vec3> = m
            .face_verts(out.faces[0])
            .iter()
            .filter_map(|v| m.position(*v))
            .collect();
        let centre: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
        assert!(
            (centre - midpoint).length() < 0.4,
            "the quad centre {centre:?} is nowhere near the old edge at {midpoint:?}"
        );
        let normal = m.face_normal(out.faces[0]).normalize_or_zero();
        assert!(
            normal.dot(centre) > 0.0,
            "the quad at {centre:?} faces {normal:?}, which is inward"
        );
    }

    #[test]
    fn bevelling_every_edge_of_a_cube_keeps_the_faces_square() {
        // The case that decides whether the first rule is right. Both of each
        // original face's edges at every corner are bevelled, so each corner
        // insets on both sides into *one* point — and the faces stay quads. Get
        // that wrong and they come out as octagons, with twice the vertices and
        // a hexagonal cap on every corner.
        let mut m = cube();
        let all: Vec<EdgeId> = m.edge_ids().collect();
        let out = bevel_edges(&mut m, &all, 0.25).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 12, "one quad per edge");
        assert_eq!(out.caps.len(), 8, "one cap per corner");
        assert_eq!(m.num_verts(), 24, "three per original corner");
        assert_eq!(m.num_faces(), 6 + 12 + 8);
        assert_eq!(m.num_edges(), 48);
        assert!(watertight(&m));
        assert_eq!(euler(&m), 2);

        let quads = m
            .face_ids()
            .filter(|f| m.face(*f).unwrap().len() == 4)
            .count();
        let triangles = m
            .face_ids()
            .filter(|f| m.face(*f).unwrap().len() == 3)
            .count();
        assert_eq!(quads, 18, "six originals and twelve chamfers");
        assert_eq!(triangles, 8, "the corner caps");
    }

    #[test]
    fn a_chamfered_cube_is_convex() {
        // The strongest cheap check that nothing folded: every face of a convex
        // solid faces away from its centre.
        let mut m = cube();
        let all: Vec<EdgeId> = m.edge_ids().collect();
        bevel_edges(&mut m, &all, 0.25).unwrap();

        for f in m.face_ids() {
            let ring: Vec<Vec3> = m
                .face_verts(f)
                .iter()
                .filter_map(|v| m.position(*v))
                .collect();
            let mid: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
            assert!(
                m.face_normal(f)
                    .normalize_or_zero()
                    .dot(mid.normalize_or_zero())
                    > 0.5,
                "a face at {mid:?} is turned the wrong way"
            );
        }
    }

    #[test]
    fn bevelling_a_ring_of_parallel_edges_needs_no_caps() {
        // The third rule again, from the other side. The cube's top face has
        // neither of its own edges bevelled at any corner, yet every corner is
        // touched — so it opens from a quad into an octagon, and the two points
        // it opens into are the ones its neighbours already arrived at.
        let mut m = cube();
        let waist: Vec<EdgeId> = m
            .edge_ids()
            .filter(|&e| {
                let [a, b] = m.edge_verts(e).unwrap();
                let (pa, pb) = (m.position(a).unwrap(), m.position(b).unwrap());
                (pa.z - pb.z).abs() > 1.0
            })
            .collect();
        assert_eq!(waist.len(), 4, "the four vertical edges");

        let out = bevel_edges(&mut m, &waist, 0.3).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 4);
        assert!(out.caps.is_empty(), "two points leave no hole to cap");
        assert_eq!(m.num_faces(), 10, "six plus four");
        assert_eq!(m.num_verts(), 16, "two per original corner");
        assert!(watertight(&m));
        assert_eq!(euler(&m), 2);

        let octagons = m
            .face_ids()
            .filter(|f| m.face(*f).unwrap().len() == 8)
            .count();
        assert_eq!(octagons, 2, "the top and bottom opened at all four corners");
    }

    #[test]
    fn too_wide_a_bevel_is_clamped_rather_than_folded() {
        let mut m = cube();
        let all: Vec<EdgeId> = m.edge_ids().collect();
        let out = bevel_edges(&mut m, &all, 50.0).unwrap();
        m.assert_valid();

        assert!(out.clamped);
        for f in m.face_ids() {
            assert!(
                m.face_normal(f).length() > 1e-9,
                "a bevel produced a zero-area face"
            );
        }
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }

    #[test]
    fn the_width_is_the_same_everywhere() {
        // Clamping per edge would keep the bevel legal and make it visibly
        // uneven, which reads as a bug rather than as a limit.
        let mut m = cube();
        for v in m.vert_ids().collect::<Vec<_>>() {
            let p = m.position(v).unwrap();
            m.set_position(v, Vec3::new(p.x, p.y, p.z * 0.15));
        }
        let all: Vec<EdgeId> = m.edge_ids().collect();
        let out = bevel_edges(&mut m, &all, 0.5).unwrap();
        m.assert_valid();
        assert!(out.clamped, "the short edges must set the limit");

        let widths: Vec<f32> = out
            .faces
            .iter()
            .map(|f| {
                let ring: Vec<Vec3> = m
                    .face_verts(*f)
                    .iter()
                    .filter_map(|v| m.position(*v))
                    .collect();
                (ring[0] - ring[ring.len() - 1]).length()
            })
            .collect();
        let first = widths[0];
        for w in &widths {
            assert!((w - first).abs() < 1e-4, "the bevel is uneven: {widths:?}");
        }
    }

    #[test]
    fn a_boundary_edge_is_skipped_and_counted() {
        // A bevel needs two faces to put the quad between. Doing something else
        // silently would be worse than doing nothing.
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let all: Vec<EdgeId> = m.edge_ids().collect();
        let out = bevel_edges(&mut m, &all, 0.2).unwrap();
        assert_eq!(out.skipped, 4);
        assert!(out.faces.is_empty());
        assert_eq!(m.num_faces(), 1, "nothing was changed");
        m.assert_valid();
    }

    #[test]
    fn a_non_manifold_edge_is_skipped() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 1));
        let shared = m.edge_ids().find(|&e| m.edge_face_count(e) == 2).unwrap();
        let [a, b] = m.edge_verts(shared).unwrap();
        let c = m.add_vert(Vec3::new(0.0, 0.0, 2.0));
        let d = m.add_vert(Vec3::new(1.0, 0.0, 2.0));
        m.add_face(&[a, b, d, c]).unwrap();
        assert_eq!(m.edge_face_count(shared), 3);

        let out = bevel_edges(&mut m, &[shared], 0.2).unwrap();
        assert_eq!(out.skipped, 1);
        assert!(out.faces.is_empty());
        m.assert_valid();
    }

    #[test]
    fn a_zero_or_negative_width_is_a_no_op() {
        let mut m = cube();
        let all: Vec<EdgeId> = m.edge_ids().collect();
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        assert_eq!(bevel_edges(&mut m, &all, 0.0).unwrap(), Bevel::default());
        assert_eq!(bevel_edges(&mut m, &all, -1.0).unwrap(), Bevel::default());
        assert_eq!(bevel_edges(&mut m, &[], 0.3).unwrap(), Bevel::default());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
        m.assert_valid();
    }

    #[test]
    fn the_new_faces_are_what_ends_up_selected() {
        let mut m = cube();
        let all: Vec<EdgeId> = m.edge_ids().collect();
        let out = bevel_edges(&mut m, &all, 0.25).unwrap();
        let mut selected = m.selected_faces();
        let mut expected: Vec<FaceId> = out.faces.iter().chain(&out.caps).copied().collect();
        selected.sort_unstable();
        expected.sort_unstable();
        assert_eq!(selected, expected);
    }

    #[test]
    fn bevelling_the_selection_goes_through_the_same_path() {
        let mut m = cube();
        m.select_none();
        let e = m.edge_ids().next().unwrap();
        m.set_edge_selected(e, true);
        m.flush_selection(SelectMode::Edge);

        let out = bevel_selected(&mut m, 0.3).unwrap();
        assert_eq!(out.faces.len(), 1);
        m.assert_valid();
    }

    #[test]
    fn every_primitive_survives_an_aggressive_bevel() {
        // The clamp is what makes this safe without knowing the scale of the
        // geometry. What must not happen is a degenerate face or an invalid
        // mesh; the shape itself is allowed to look extreme.
        for mesh in [
            primitives::cube(2.0),
            primitives::grid(4.0, 3, 3),
            primitives::uv_sphere(1.0, 10, 5),
            primitives::ico_sphere(1.0, 1),
            primitives::cylinder(1.0, 2.0, 7),
            primitives::cone(1.0, 2.0, 6),
            primitives::torus(1.0, 0.4, 8, 6),
        ] {
            let mut m = BMesh::from_mesh_data(&mesh);
            let interior: Vec<EdgeId> = m
                .edge_ids()
                .filter(|&e| m.edge_face_count(e) == 2)
                .collect();
            let out = bevel_edges(&mut m, &interior, 100.0).unwrap();
            m.assert_valid();
            let _ = out;
            for f in m.face_ids() {
                assert!(
                    m.face_normal(f).length() > 1e-12,
                    "a bevel produced a zero-area face"
                );
            }
            assert_eq!(m.to_mesh_data().validate(), Vec::new());
        }
    }

    #[test]
    fn a_bevelled_mesh_still_bakes_to_storage_form() {
        let mut m = cube();
        let all: Vec<EdgeId> = m.edge_ids().collect();
        bevel_edges(&mut m, &all, 0.3).unwrap();
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }
}
