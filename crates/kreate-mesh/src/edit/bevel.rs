//! Bevel: replacing an edge with a face.
//!
//! The plan calls this the hardest of the core operators and pins its scope down
//! precisely: **edge bevel, simple miter**, now with any number of segments on
//! a round profile. A profile shape other than round, miter types and general
//! self-intersection handling are deferred. What is here has to be right for
//! the ordinary cases and honest about the rest.
//!
//! # Segments
//!
//! One segment is a flat chamfer: each bevelled edge becomes a quad. More
//! segments bend that quad into strips along an arc that leaves one face and
//! arrives at the other tangent to both — a rounded edge, which is what
//! furniture, a phone, a bar of soap have. See [`arc`] for the curve and why
//! neighbouring walls share theirs.
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
    bevel_edges_segments(mesh, edges, width, 1)
}

/// Bevels the given edges by `width` into `segments` strips on a round
/// profile. One segment is [`bevel_edges`]' flat chamfer.
pub fn bevel_edges_segments(
    mesh: &mut BMesh,
    edges: &[EdgeId],
    width: f32,
    segments: u32,
) -> Result<Bevel, BmError> {
    let segments = segments.clamp(1, 64);
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

    // What each new point is: the original vertex, moved along these edges.
    // The arc between two of them bends about the part they share — see `arc`.
    let basis = |created: VertId| -> Option<(VertId, Vec<EdgeId>)> {
        if let Some(p) = edge_points.iter().find(|p| p.created == created) {
            return Some((p.vertex, vec![p.along]));
        }
        let p = corner_points.iter().find(|p| p.created == created)?;
        let (incoming, outgoing) = corner_edges(mesh, p.face, p.vertex)?;
        Some((p.vertex, vec![incoming, outgoing]))
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

    // The profile between two wall corners at one original vertex: the
    // points strictly between them, from `from` to `to`. Shared by every wall
    // that runs between the same two corners — where two bevelled edges turn
    // a corner together, both walls end on the same pair, and a curve of each
    // their own would leave a sliver open between them.
    let mut arcs: Vec<((VertId, VertId), Vec<Vec3>)> = Vec::new();
    let mut profile = |mesh: &BMesh, from: VertId, to: VertId| -> Vec<Vec3> {
        if segments < 2 || from == to {
            return Vec::new();
        }
        if let Some((_, points)) = arcs.iter().find(|(k, _)| *k == (from, to)) {
            return points.clone();
        }
        if let Some((_, points)) = arcs.iter().find(|(k, _)| *k == (to, from)) {
            return points.iter().rev().copied().collect();
        }
        let points = (|| {
            let (v, a) = basis(from)?;
            let (_, b) = basis(to)?;
            let shared: Vec3 = a
                .iter()
                .filter(|g| b.contains(g))
                .filter_map(|g| offset_along(mesh, v, *g))
                .sum();
            let corner = mesh.position(v)? + shared;
            Some(arc(
                corner,
                mesh.position(from)?,
                mesh.position(to)?,
                segments,
            ))
        })()
        .unwrap_or_default();
        arcs.push(((from, to), points.clone()));
        points
    };

    let mut walls: Vec<[VertId; 4]> = Vec::new();
    /// A wall bent into strips: the corner pairs at each end of its edge.
    struct Strip {
        a: (VertId, VertId),
        b: (VertId, VertId),
        vertex_a: VertId,
        vertex_b: VertId,
    }
    let mut strips: Vec<Strip> = Vec::new();
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
        if segments < 2 {
            walls.push([near_b, near_a, far_a, far_b]);
            continue;
        }
        // The arcs at each end, as positions for now: vertices are made once
        // per arc, below, so two walls sharing one share its vertices too.
        let at_a = profile(mesh, near_a, far_a);
        let at_b = profile(mesh, near_b, far_b);
        if at_a.len() + 1 != segments as usize || at_b.len() + 1 != segments as usize {
            walls.push([near_b, near_a, far_a, far_b]);
            continue;
        }
        strips.push(Strip {
            a: (near_a, far_a),
            b: (near_b, far_b),
            vertex_a: a,
            vertex_b: b,
        });
    }

    // Each arc's vertices, made once.
    let mut arc_verts: Vec<((VertId, VertId), Vec<VertId>, VertId)> = Vec::new();
    let mut rail = |mesh: &mut BMesh,
                    out: &mut Bevel,
                    ends: (VertId, VertId),
                    vertex: VertId|
     -> Vec<VertId> {
        if let Some((_, made, _)) = arc_verts.iter().find(|(k, _, _)| *k == ends) {
            return made.clone();
        }
        if let Some((_, made, _)) = arc_verts.iter().find(|(k, _, _)| *k == (ends.1, ends.0)) {
            return made.iter().rev().copied().collect();
        }
        let points = arcs
            .iter()
            .find(|(k, _)| *k == ends)
            .map(|(_, p)| p.clone())
            .or_else(|| {
                arcs.iter()
                    .find(|(k, _)| *k == (ends.1, ends.0))
                    .map(|(_, p)| p.iter().rev().copied().collect())
            })
            .unwrap_or_default();
        let made: Vec<VertId> = points.into_iter().map(|p| mesh.add_vert(p)).collect();
        out.verts.extend(&made);
        arc_verts.push((ends, made.clone(), vertex));
        made
    };
    let mut strip_rings: Vec<Vec<VertId>> = Vec::new();
    for strip in &strips {
        let mut at_a = vec![strip.a.0];
        at_a.extend(rail(mesh, &mut out, strip.a, strip.vertex_a));
        at_a.push(strip.a.1);
        let mut at_b = vec![strip.b.0];
        at_b.extend(rail(mesh, &mut out, strip.b, strip.vertex_b));
        at_b.push(strip.b.1);
        for k in 0..segments as usize {
            strip_rings.push(vec![at_b[k], at_a[k], at_a[k + 1], at_b[k + 1]]);
        }
    }

    // Where each corner's round is centred: the original vertex moved in along
    // every bevelled edge that meets there. On a cube bevelled all round, every
    // arc at the corner lies on a sphere about this point — which is what lets
    // the cap be bent onto that sphere rather than left flat.
    let centres: Vec<(VertId, Vec3)> = touched
        .iter()
        .filter_map(|&v| {
            let moved: Vec3 = mesh
                .vert_edges(v)
                .into_iter()
                .filter(|g| bevelled(*g))
                .filter_map(|g| offset_along(mesh, v, g))
                .sum();
            Some((v, mesh.position(v)? + moved))
        })
        .collect();

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
    for ring in strip_rings {
        let mut ring = ring;
        ring.dedup();
        if ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() >= 3 {
            out.faces.push(mesh.add_face(&ring)?);
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
            .chain(
                arc_verts
                    .iter()
                    .filter(|(_, _, vertex)| *vertex == v)
                    .flat_map(|(_, made, _)| made.iter().copied()),
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
            if segments < 2 {
                out.caps.extend(cap.faces);
                continue;
            }
            // Rounded edges around a flat cap read as a cut corner. Where
            // three rounds meet with an even count of segments, a grid of
            // quads on the round; otherwise a fan to one point pushed out
            // onto it.
            let centre = centres.iter().find(|(w, _)| *w == v).map(|(_, c)| *c);
            let corners: Vec<VertId> = corner_points
                .iter()
                .filter(|p| p.vertex == v)
                .map(|p| p.created)
                .collect();
            for face in cap.faces {
                let patched =
                    centre.and_then(|c| grid_onto_round(mesh, face, c, &corners, segments));
                match patched.or_else(|| centre.and_then(|c| poke_onto_round(mesh, face, c))) {
                    Some(made) => out.caps.extend(made),
                    None => out.caps.push(face),
                }
            }
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

/// Replaces a corner's cap with a grid of quads on the round about `centre`,
/// when the cap is three rounds of `segments` strips meeting — a cube's
/// corner bevelled all round — and `segments` is even. `None`, leaving the
/// face, otherwise.
///
/// The cap is cut into three four-sided pieces at the middle of each round
/// and a point in the middle of the cap; each piece is filled as a Coons
/// patch of its four sides and every point carried out onto the round. Even
/// only because the pieces meet at the rounds' middles, which an odd count
/// does not have a vertex at.
fn grid_onto_round(
    mesh: &mut BMesh,
    face: FaceId,
    centre: Vec3,
    corners: &[VertId],
    segments: u32,
) -> Option<Vec<FaceId>> {
    let n = segments as usize;
    if n < 2 || !n.is_multiple_of(2) || corners.len() != 3 {
        return None;
    }
    let ring = mesh.face_verts(face);
    if ring.len() != 3 * n {
        return None;
    }
    // Where the rounds meet, spaced a round apart round the ring.
    let first = ring.iter().position(|v| corners.contains(v))?;
    if !(0..3).all(|k| corners.contains(&ring[(first + k * n) % ring.len()])) {
        return None;
    }
    let at = |i: usize| ring[(first + i) % ring.len()];
    let position = |mesh: &BMesh, v: VertId| mesh.position(v);
    let points: Vec<Vec3> = ring.iter().filter_map(|v| position(mesh, *v)).collect();
    if points.len() != ring.len() {
        return None;
    }
    let radius = points.iter().map(|p| (*p - centre).length()).sum::<f32>() / points.len() as f32;
    let onto = |p: Vec3| centre + (p - centre).normalize_or_zero() * radius;
    let middle = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    if (middle - centre).length() < 1e-9 || radius <= 0.0 {
        return None;
    }

    let m = n / 2;
    mesh.remove_face(face);
    let apex = mesh.add_vert(onto(middle));
    // The three spokes, from each round's middle to the apex, `m` steps.
    let mut spokes: Vec<Vec<VertId>> = Vec::with_capacity(3);
    for k in 0..3 {
        let from = at(k * n + m);
        let (a, b) = (position(mesh, from)?, position(mesh, apex)?);
        let mut spoke = vec![from];
        for j in 1..m {
            spoke.push(mesh.add_vert(onto(a.lerp(b, j as f32 / m as f32))));
        }
        spoke.push(apex);
        spokes.push(spoke);
    }

    let mut made = Vec::new();
    for k in 0..3 {
        // Corners (0,0) the round's start, (m,0) its middle, (m,m) the apex,
        // (0,m) the previous round's middle.
        let bottom: Vec<VertId> = (0..=m).map(|u| at(k * n + u)).collect();
        let left: Vec<VertId> = (0..=m).map(|v| at((k * n + 3 * n - v) % (3 * n))).collect();
        let right = spokes[k].clone();
        let top = spokes[(k + 2) % 3].clone();
        let side = |mesh: &BMesh, vs: &[VertId]| -> Option<Vec<Vec3>> {
            vs.iter().map(|v| position(mesh, *v)).collect()
        };
        let (pb, pl, pr, pt) = (
            side(mesh, &bottom)?,
            side(mesh, &left)?,
            side(mesh, &right)?,
            side(mesh, &top)?,
        );
        let mut grid = vec![vec![apex; m + 1]; m + 1];
        for u in 0..=m {
            for v in 0..=m {
                grid[u][v] = if v == 0 {
                    bottom[u]
                } else if u == 0 {
                    left[v]
                } else if u == m {
                    right[v]
                } else if v == m {
                    top[u]
                } else {
                    let (s, t) = (u as f32 / m as f32, v as f32 / m as f32);
                    let p = pl[v] * (1.0 - s) + pr[v] * s + pb[u] * (1.0 - t) + pt[u] * t
                        - (pb[0] * (1.0 - s) * (1.0 - t)
                            + pb[m] * s * (1.0 - t)
                            + pt[m] * s * t
                            + pt[0] * (1.0 - s) * t);
                    mesh.add_vert(onto(p))
                };
            }
        }
        for u in 0..m {
            for v in 0..m {
                made.push(
                    mesh.add_face(&[
                        grid[u][v],
                        grid[u + 1][v],
                        grid[u + 1][v + 1],
                        grid[u][v + 1],
                    ])
                    .ok()?,
                );
            }
        }
    }
    Some(made)
}

/// Replaces `face` with a fan of triangles to a new point on the round about
/// `centre`: the face's own middle, carried out from `centre` to the average
/// distance its corners are. `None`, leaving the face, if it cannot be done.
fn poke_onto_round(mesh: &mut BMesh, face: FaceId, centre: Vec3) -> Option<Vec<FaceId>> {
    let ring = mesh.face_verts(face);
    let points: Vec<Vec3> = ring.iter().filter_map(|v| mesh.position(*v)).collect();
    if points.len() != ring.len() || ring.len() < 4 {
        return None;
    }
    let middle = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let radius = points.iter().map(|p| (*p - centre).length()).sum::<f32>() / points.len() as f32;
    let outward = (middle - centre).normalize_or_zero();
    if outward == Vec3::ZERO || radius <= 0.0 {
        return None;
    }
    mesh.remove_face(face);
    let apex = mesh.add_vert(centre + outward * radius);
    let mut fan = Vec::with_capacity(ring.len());
    for i in 0..ring.len() {
        fan.push(
            mesh.add_face(&[ring[i], ring[(i + 1) % ring.len()], apex])
                .ok()?,
        );
    }
    Some(fan)
}

/// Bevels the selected edges.
pub fn bevel_selected(mesh: &mut BMesh, width: f32) -> Result<Bevel, BmError> {
    bevel_selected_segments(mesh, width, 1)
}

/// Bevels the selected edges into `segments` strips on a round profile.
pub fn bevel_selected_segments(
    mesh: &mut BMesh,
    width: f32,
    segments: u32,
) -> Result<Bevel, BmError> {
    let edges = mesh.selected_edges();
    bevel_edges_segments(mesh, &edges, width, segments)
}

/// The points strictly between `from` and `to` on a round profile about
/// `corner`, for `segments` strips.
///
/// `from - corner` and `to - corner` are the two offsets the bevel made, one
/// in each face; the curve leaves `from` heading back along the first and
/// arrives at `to` along the second, so it meets both faces tangentially. On
/// a square corner with equal offsets it is a quarter circle; on any other it
/// is the same quarter circle carried by the affine map that takes the axes
/// to the two offsets, which keeps both tangencies.
///
/// `corner` is the original vertex moved along whatever edges the two ends
/// share: nothing for a single edge's wall, the bevelled edge itself where its
/// neighbours are bevelled too, as on a cube bevelled all round. It depends
/// only on the two ends, which is what lets two walls ending on the same pair
/// agree on the curve between them.
fn arc(corner: Vec3, from: Vec3, to: Vec3, segments: u32) -> Vec<Vec3> {
    let (u, v) = (from - corner, to - corner);
    (1..segments)
        .map(|i| {
            let t = std::f32::consts::FRAC_PI_2 * i as f32 / segments as f32;
            corner + u * (1.0 - t.sin()) + v * (1.0 - t.cos())
        })
        .collect()
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

    /// Every new strip faces away from the cube's centre.
    fn strips_face_out(mesh: &BMesh, faces: &[FaceId]) -> bool {
        faces.iter().all(|f| {
            let ring: Vec<Vec3> = mesh
                .face_verts(*f)
                .iter()
                .filter_map(|v| mesh.position(*v))
                .collect();
            let centre: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
            mesh.face_normal(*f).dot(centre) > 0.0
        })
    }

    #[test]
    fn segments_round_one_edge_into_a_quarter_cylinder() {
        let mut m = cube();
        let e = m.edge_ids().next().unwrap();
        let [a, b] = m.edge_verts(e).unwrap();
        let (pa, pb) = (m.position(a).unwrap(), m.position(b).unwrap());
        let out = bevel_edges_segments(&mut m, &[e], 0.4, 4).unwrap();
        m.assert_valid();
        assert!(!out.clamped);
        assert_eq!(out.faces.len(), 4, "four strips");
        assert!(watertight(&m));
        assert_eq!(euler(&m), 2);
        assert!(strips_face_out(&m, &out.faces));

        // Every point on the round lies half a width from the axis the round
        // turns about: the old edge moved in by the width along both faces.
        let along = (pb - pa).normalize();
        let inward = -(pa + pb) * 0.5;
        let inward = (inward - along * inward.dot(along)).normalize();
        let axis_point = pa + inward * (0.4 * std::f32::consts::SQRT_2);
        for v in out.verts.iter().filter_map(|v| m.position(*v)) {
            let d = v - axis_point;
            let radial = (d - along * d.dot(along)).length();
            assert!(
                (radial - 0.4).abs() < 1e-4,
                "{v:?} is {radial} from the axis"
            );
        }
    }

    #[test]
    fn segments_round_every_edge_of_a_cube_and_close_the_corners() {
        let mut m = cube();
        let edges: Vec<EdgeId> = m.edge_ids().collect();
        let out = bevel_edges_segments(&mut m, &edges, 0.3, 3).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 12 * 3);
        // Each corner: three arcs of three strips meet, nine points round a
        // fan of nine triangles bent onto the sphere.
        assert_eq!(out.caps.len(), 8 * 9, "a fan on every corner");
        for f in &out.caps {
            for v in m.face_verts(*f) {
                let p = m.position(v).unwrap();
                let centre = p.signum() * (1.0 - 0.3);
                assert!(
                    ((p - centre).length() - 0.3).abs() < 0.02,
                    "{p:?} is off the round"
                );
            }
        }
        assert!(watertight(&m), "a corner was left open");
        assert_eq!(euler(&m), 2);
        assert!(strips_face_out(&m, &out.faces));
    }

    #[test]
    fn an_even_count_of_segments_caps_each_corner_with_a_grid_of_quads() {
        let mut m = cube();
        let edges: Vec<EdgeId> = m.edge_ids().collect();
        let out = bevel_edges_segments(&mut m, &edges, 0.3, 4).unwrap();
        m.assert_valid();
        // Three pieces of two by two at each corner.
        assert_eq!(out.caps.len(), 8 * 12);
        assert!(
            out.caps.iter().all(|f| m.face_verts(*f).len() == 4),
            "the caps are not all quads"
        );
        assert!(watertight(&m), "a corner was left open");
        assert_eq!(euler(&m), 2);
        for f in &out.caps {
            let ring: Vec<Vec3> = m
                .face_verts(*f)
                .iter()
                .filter_map(|v| m.position(*v))
                .collect();
            for p in &ring {
                let centre = p.signum() * (1.0 - 0.3);
                assert!(
                    ((*p - centre).length() - 0.3).abs() < 0.02,
                    "{p:?} is off the round"
                );
            }
            // And facing out of the cube.
            let mid: Vec3 = ring.iter().copied().sum::<Vec3>() / 4.0;
            assert!(m.face_normal(*f).dot(mid) > 0.0, "a cap quad faces in");
        }
    }

    #[test]
    fn segments_round_a_rim_with_the_walls_meeting_at_its_corners() {
        // The top four edges, as on a table top: at each corner two rounded
        // walls turn together, and must share their profile rather than leave
        // a sliver between them.
        let mut m = cube();
        let top: Vec<EdgeId> = m
            .edge_ids()
            .filter(|e| {
                let [a, b] = m.edge_verts(*e).unwrap();
                m.position(a).unwrap().z > 0.5 && m.position(b).unwrap().z > 0.5
            })
            .collect();
        assert_eq!(top.len(), 4);
        let out = bevel_edges_segments(&mut m, &top, 0.4, 5).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 4 * 5);
        assert!(
            out.caps.is_empty(),
            "the walls meet; nothing is left to cap"
        );
        assert!(watertight(&m));
        assert_eq!(euler(&m), 2);
        assert!(strips_face_out(&m, &out.faces));
    }
}
