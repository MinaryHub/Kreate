//! The knife: cutting new edges along a path drawn over the surface.
//!
//! Blender's `K`. The user clicks points on the mesh; each pair of points is a
//! segment, and the segment cuts every face it passes over *as seen from the
//! view* — the cut runs through the plane that holds the segment and the view
//! direction, which is what makes a straight stroke on screen a straight cut
//! on the model however the surface curves under it.
//!
//! # A walk, not a slice
//!
//! The plane alone would also cut the back of the model, which the user never
//! drew over. So each segment is followed across the surface instead: from the
//! face the first point is on, over the edge the plane leaves it by, into the
//! next face, until the face the second point is on. Only the faces walked
//! through are cut.
//!
//! # What is made
//!
//! Every edge crossed gets a vertex where the plane meets it, and every face
//! walked through is split in two along the cut. A click inside a face (rather
//! than on an edge) is a bend in the cut: the path runs through it, and the
//! face is split along the bent line. A path that *ends* inside a face — its
//! first or last point not on an edge — leaves that last piece uncut: a face
//! cannot be split by a line that stops in its middle.

use kreate_math::Vec3;

use crate::arena::{EdgeId, FaceId, VertId};
use crate::bmesh::{BMesh, BmError, SelectMode};

/// What a knife cut made.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Knife {
    /// The vertices the cut runs through, new and old.
    pub verts: Vec<VertId>,
    /// The faces the cut split, as they are now.
    pub faces: Vec<FaceId>,
}

/// Where a path point is on the mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Loc {
    Vert(VertId),
    Edge(EdgeId),
    Face(FaceId),
}

#[derive(Clone, Copy, Debug)]
struct Pos {
    loc: Loc,
    at: Vec3,
}

/// Cuts along `path`, points on the surface in the mesh's own space, seen
/// along `view`.
pub fn knife(mesh: &mut BMesh, path: &[Vec3], view: Vec3) -> Result<Knife, BmError> {
    let mut out = Knife::default();
    if path.len() < 2 || view.length_squared() == 0.0 {
        return Ok(out);
    }
    let size = extent(mesh).max(1e-6);
    let near = size * 1e-4;

    // --- read: where each point is, and what each segment crosses ------------
    let located: Vec<Pos> = path.iter().filter_map(|p| locate(mesh, *p, near)).collect();
    // One entry per step across a face: the face and the two ends, in order.
    let mut steps: Vec<(FaceId, Pos, Pos)> = Vec::new();
    for pair in located.windows(2) {
        walk(mesh, pair[0], pair[1], view, near, &mut steps);
    }
    if steps.is_empty() {
        return Ok(out);
    }

    // --- write: vertices on the crossed edges ---------------------------------
    // Grouped per edge, so an edge crossed twice is split once into three.
    let mut on_edges: Vec<(EdgeId, Vec<Vec3>)> = Vec::new();
    for (_, a, b) in &steps {
        for p in [a, b] {
            if let Loc::Edge(e) = p.loc {
                match on_edges.iter_mut().find(|(x, _)| *x == e) {
                    Some((_, list)) if list.iter().any(|q| q.distance(p.at) < near) => {}
                    Some((_, list)) => list.push(p.at),
                    None => on_edges.push((e, vec![p.at])),
                }
            }
        }
    }
    let mut made: Vec<(Vec3, VertId)> = Vec::new();
    for (e, points) in on_edges {
        for (at, v) in split_edge(mesh, e, &points)? {
            made.push((at, v));
        }
    }
    // Points inside faces get their vertex now; they are joined in below.
    let mut interior: Vec<(Vec3, VertId)> = Vec::new();
    let vert_of = |p: &Pos, mesh: &mut BMesh, interior: &mut Vec<(Vec3, VertId)>| -> VertId {
        match p.loc {
            Loc::Vert(v) => v,
            Loc::Edge(_) => made
                .iter()
                .min_by(|a, b| a.0.distance(p.at).total_cmp(&b.0.distance(p.at)))
                .map(|(_, v)| *v)
                .expect("every edge point was made a vertex"),
            Loc::Face(_) => match interior.iter().find(|(q, _)| q.distance(p.at) < near) {
                Some((_, v)) => *v,
                None => {
                    let v = mesh.add_vert(p.at);
                    interior.push((p.at, v));
                    v
                }
            },
        }
    };

    // --- write: split each face along its piece of the path ------------------
    // Consecutive steps in the same face are one piece: entry, bends, exit.
    let mut pieces: Vec<Vec<(VertId, bool)>> = Vec::new();
    let mut last_face: Option<FaceId> = None;
    for (face, a, b) in &steps {
        let va = vert_of(a, mesh, &mut interior);
        let vb = vert_of(b, mesh, &mut interior);
        let inner = |p: &Pos| matches!(p.loc, Loc::Face(_));
        match pieces.last_mut() {
            Some(piece) if last_face == Some(*face) && piece.last().map(|x| x.0) == Some(va) => {
                piece.push((vb, inner(b)));
            }
            _ => pieces.push(vec![(va, inner(a)), (vb, inner(b))]),
        }
        last_face = Some(*face);
    }
    for piece in pieces {
        // From the first point on the face's outline to the last; a piece that
        // does not reach the outline twice cannot split anything.
        let Some(first) = piece.iter().position(|(_, inner)| !inner) else {
            continue;
        };
        let Some(last) = piece.iter().rposition(|(_, inner)| !inner) else {
            continue;
        };
        if last <= first {
            continue;
        }
        let chain: Vec<VertId> = piece[first..=last].iter().map(|(v, _)| *v).collect();
        if let Some([fa, fb]) = split_face(mesh, &chain)? {
            out.faces.push(fa);
            out.faces.push(fb);
        }
        for v in chain {
            if !out.verts.contains(&v) {
                out.verts.push(v);
            }
        }
    }
    // Bends that no face took — the loose end of a path that stopped inside a
    // face — are not left behind as stray vertices.
    for (_, v) in interior {
        if mesh.vert_edge_count(v) == 0 {
            mesh.remove_vert(v);
            out.verts.retain(|x| *x != v);
        }
    }
    out.faces.retain(|f| mesh.face(*f).is_some());

    // The cut is selected, as edges, so the next key acts on it.
    mesh.select_none();
    for w in out.verts.windows(2) {
        if let Some(e) = mesh.edge_between(w[0], w[1]) {
            mesh.set_edge_selected(e, true);
        }
    }
    mesh.flush_selection(SelectMode::Edge);
    Ok(out)
}

/// The diagonal of the mesh's bounds: the yardstick for "close enough".
fn extent(mesh: &BMesh) -> f32 {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for v in mesh.vert_ids() {
        if let Some(p) = mesh.position(v) {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    (hi - lo).length()
}

/// Which vertex, edge or face `p` is on, trying them in that order.
fn locate(mesh: &BMesh, p: Vec3, near: f32) -> Option<Pos> {
    if let Some(v) = mesh
        .vert_ids()
        .find(|v| mesh.position(*v).is_some_and(|q| q.distance(p) < near))
    {
        return Some(Pos {
            loc: Loc::Vert(v),
            at: mesh.position(v)?,
        });
    }
    let mut best: Option<(f32, EdgeId, Vec3)> = None;
    for e in mesh.edge_ids() {
        let Some([a, b]) = mesh.edge_verts(e) else {
            continue;
        };
        let (Some(pa), Some(pb)) = (mesh.position(a), mesh.position(b)) else {
            continue;
        };
        let q = closest_on_segment(p, pa, pb);
        let d = q.distance(p);
        if d < near && best.is_none_or(|(bd, _, _)| d < bd) {
            best = Some((d, e, q));
        }
    }
    if let Some((_, e, q)) = best {
        return Some(Pos {
            loc: Loc::Edge(e),
            at: q,
        });
    }
    // Inside a face: on one of its fan triangles, within reach of its plane.
    let mut best: Option<(f32, FaceId)> = None;
    for f in mesh.face_ids() {
        let ring: Vec<Vec3> = mesh
            .face_verts(f)
            .iter()
            .filter_map(|v| mesh.position(*v))
            .collect();
        for i in 1..ring.len().saturating_sub(1) {
            if let Some(d) = on_triangle(p, ring[0], ring[i], ring[i + 1], near * 10.0) {
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, f));
                }
            }
        }
    }
    best.map(|(_, f)| Pos {
        loc: Loc::Face(f),
        at: p,
    })
}

fn closest_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 == 0.0 {
        return a;
    }
    a + ab * ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
}

/// How far `p` is from the triangle's plane, if it is over the triangle and
/// within `reach` of it.
fn on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3, reach: f32) -> Option<f32> {
    let n = (b - a).cross(c - a);
    let area2 = n.length_squared();
    if area2 == 0.0 {
        return None;
    }
    let dist = (p - a).dot(n) / area2.sqrt();
    if dist.abs() > reach {
        return None;
    }
    let q = p - n * ((p - a).dot(n) / area2);
    let u = (b - q).cross(c - q).dot(n) / area2;
    let v = (c - q).cross(a - q).dot(n) / area2;
    let w = 1.0 - u - v;
    let slack = -1e-4;
    (u >= slack && v >= slack && w >= slack).then_some(dist.abs())
}

/// The faces a position is on.
fn faces_at(mesh: &BMesh, loc: Loc) -> Vec<FaceId> {
    match loc {
        Loc::Vert(v) => mesh.vert_faces(v),
        Loc::Edge(e) => mesh.edge_faces(e),
        Loc::Face(f) => vec![f],
    }
}

/// Follows the segment `a → b` across the surface, appending one step per
/// face walked through.
fn walk(mesh: &BMesh, a: Pos, b: Pos, view: Vec3, near: f32, steps: &mut Vec<(FaceId, Pos, Pos)>) {
    let d = b.at - a.at;
    let len2 = d.length_squared();
    let normal = d.cross(view);
    if len2 == 0.0 || normal.length_squared() == 0.0 {
        return;
    }
    let normal = normal.normalize();
    let target = faces_at(mesh, b.loc);
    let mut here = a;
    let mut s_here = 0.0f32;
    let mut came_from: Option<FaceId> = None;
    // A walk crosses at most every face once; a bound, not an expectation.
    for _ in 0..mesh.num_faces() + 1 {
        let around: Vec<FaceId> = faces_at(mesh, here.loc)
            .into_iter()
            .filter(|f| Some(*f) != came_from)
            .collect();
        if let Some(f) = around.iter().find(|f| target.contains(f)) {
            steps.push((*f, here, b));
            return;
        }
        // The nearest edge ahead, among the faces here, where the plane
        // crosses it.
        let mut best: Option<(f32, FaceId, Pos)> = None;
        for &f in &around {
            for l in mesh.face_loops(f) {
                let Some(e) = mesh.loop_edge(l) else { continue };
                let touches = match here.loc {
                    Loc::Edge(x) => x == e,
                    Loc::Vert(v) => mesh.edge_verts(e).is_some_and(|ends| ends.contains(&v)),
                    Loc::Face(_) => false,
                };
                if touches {
                    continue;
                }
                let Some([p, q]) = mesh.edge_verts(e) else {
                    continue;
                };
                let (Some(pp), Some(pq)) = (mesh.position(p), mesh.position(q)) else {
                    continue;
                };
                let (sp, sq) = (normal.dot(pp - a.at), normal.dot(pq - a.at));
                if (sp > 0.0 && sq > 0.0) || (sp < 0.0 && sq < 0.0) || sp == sq {
                    continue;
                }
                let t = sp / (sp - sq);
                let x = pp.lerp(pq, t);
                let s = d.dot(x - a.at) / len2;
                if s <= s_here + 1e-6 || s > 1.0 + 1e-4 {
                    continue;
                }
                let loc = if x.distance(pp) < near {
                    Pos {
                        loc: Loc::Vert(p),
                        at: pp,
                    }
                } else if x.distance(pq) < near {
                    Pos {
                        loc: Loc::Vert(q),
                        at: pq,
                    }
                } else {
                    Pos {
                        loc: Loc::Edge(e),
                        at: x,
                    }
                };
                if best.as_ref().is_none_or(|(bs, _, _)| s < *bs) {
                    best = Some((s, f, loc));
                }
            }
        }
        let Some((s, f, next)) = best else { return };
        steps.push((f, here, next));
        came_from = Some(f);
        here = next;
        s_here = s;
    }
}

/// Splits edge `e` at `points`, one new vertex each, and puts them into every
/// face along it. Returns each point with its vertex.
fn split_edge(
    mesh: &mut BMesh,
    e: EdgeId,
    points: &[Vec3],
) -> Result<Vec<(Vec3, VertId)>, BmError> {
    let Some([p, q]) = mesh.edge_verts(e) else {
        return Ok(Vec::new());
    };
    let Some(pp) = mesh.position(p) else {
        return Ok(Vec::new());
    };
    let mut sorted: Vec<Vec3> = points.to_vec();
    sorted.sort_by(|a, b| a.distance(pp).total_cmp(&b.distance(pp)));
    let faces = mesh.edge_faces(e);
    let rings: Vec<Vec<VertId>> = faces.iter().map(|f| mesh.face_verts(*f)).collect();
    for f in &faces {
        mesh.remove_face(*f);
    }
    mesh.remove_edge(e);
    let new: Vec<VertId> = sorted.iter().map(|at| mesh.add_vert(*at)).collect();
    for ring in rings {
        let n = ring.len();
        let mut patched = Vec::with_capacity(n + new.len());
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            patched.push(a);
            if a == p && b == q {
                patched.extend(new.iter().copied());
            } else if a == q && b == p {
                patched.extend(new.iter().rev().copied());
            }
        }
        mesh.add_face(&patched)?;
    }
    if faces.is_empty() {
        // A wire edge: a chain of wire edges through the new vertices.
        let mut prev = p;
        for v in new.iter().copied().chain(std::iter::once(q)) {
            mesh.add_edge(prev, v)?;
            prev = v;
        }
    }
    Ok(sorted.into_iter().zip(new).collect())
}

/// Splits the face that has both ends of `chain` on its outline, along the
/// chain. `None` when no face has, or when the chain is already an edge.
fn split_face(mesh: &mut BMesh, chain: &[VertId]) -> Result<Option<[FaceId; 2]>, BmError> {
    let (first, last) = (chain[0], chain[chain.len() - 1]);
    if first == last {
        return Ok(None);
    }
    let inner = &chain[1..chain.len() - 1];
    let Some(face) = mesh
        .vert_faces(first)
        .into_iter()
        .find(|f| mesh.face_verts(*f).contains(&last))
    else {
        return Ok(None);
    };
    let ring = mesh.face_verts(face);
    let n = ring.len();
    let a = ring
        .iter()
        .position(|v| *v == first)
        .expect("the face has it");
    let b = ring
        .iter()
        .position(|v| *v == last)
        .expect("the face has it");
    if inner.is_empty() && ((a + 1) % n == b || (b + 1) % n == a) {
        return Ok(None);
    }
    // Round the outline from `a` to `b`, back along the chain; and from `b`
    // to `a`, forward along it. Both keep the face's winding.
    let mut one: Vec<VertId> = Vec::new();
    let mut i = a;
    loop {
        one.push(ring[i]);
        if i == b {
            break;
        }
        i = (i + 1) % n;
    }
    one.extend(inner.iter().rev().copied());
    let mut two: Vec<VertId> = Vec::new();
    let mut i = b;
    loop {
        two.push(ring[i]);
        if i == a {
            break;
        }
        i = (i + 1) % n;
    }
    two.extend(inner.iter().copied());
    if one.len() < 3 || two.len() < 3 {
        return Ok(None);
    }
    mesh.remove_face(face);
    let fa = mesh.add_face(&one)?;
    let fb = mesh.add_face(&two)?;
    Ok(Some([fa, fb]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    /// A 2-unit plane in XY, seen from above.
    fn plane() -> BMesh {
        BMesh::from_mesh_data(&primitives::plane(2.0))
    }

    #[test]
    fn a_stroke_from_edge_to_edge_splits_the_face_in_two() {
        let mut m = plane();
        let out = knife(
            &mut m,
            &[Vec3::new(-1.0, 0.2, 0.0), Vec3::new(1.0, -0.3, 0.0)],
            Vec3::NEG_Z,
        )
        .unwrap();
        m.assert_valid();
        assert_eq!(m.num_faces(), 2);
        assert_eq!(m.num_verts(), 6);
        assert_eq!(out.faces.len(), 2);
        assert_eq!(out.verts.len(), 2);
        // Both halves face the way the plane did.
        for f in m.face_ids() {
            assert!(m.face_normal(f).z > 0.0);
        }
    }

    #[test]
    fn a_stroke_across_a_grid_cuts_every_face_it_passes_over_and_only_those() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(3.0, 3, 3));
        let before = m.num_faces();
        // Along y = 0.1, across the middle row of three quads.
        knife(
            &mut m,
            &[Vec3::new(-1.5, 0.1, 0.0), Vec3::new(1.5, 0.1, 0.0)],
            Vec3::NEG_Z,
        )
        .unwrap();
        m.assert_valid();
        assert_eq!(m.num_faces(), before + 3);
    }

    #[test]
    fn a_bend_inside_a_face_is_part_of_the_cut() {
        let mut m = plane();
        knife(
            &mut m,
            &[
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
            ],
            Vec3::NEG_Z,
        )
        .unwrap();
        m.assert_valid();
        assert_eq!(m.num_faces(), 2);
        assert_eq!(m.num_verts(), 7, "two on the edges and the bend");
    }

    #[test]
    fn a_stroke_that_stops_inside_a_face_cuts_nothing_and_leaves_nothing() {
        let mut m = plane();
        knife(
            &mut m,
            &[Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.2, 0.1, 0.0)],
            Vec3::NEG_Z,
        )
        .unwrap();
        m.assert_valid();
        assert_eq!(m.num_faces(), 1);
        // The crossed edge keeps its new vertex; the stray end does not.
        assert_eq!(m.num_verts(), 5);
    }

    #[test]
    fn a_cube_is_cut_on_the_side_that_was_drawn_over_and_not_behind() {
        // Seen from the front (-Y), a stroke across the front face from its
        // left edge to its right edge. The back face is in the same plane of
        // the cut and must not be touched.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        knife(
            &mut m,
            &[Vec3::new(-1.0, -1.0, 0.3), Vec3::new(1.0, -1.0, 0.3)],
            Vec3::Y,
        )
        .unwrap();
        m.assert_valid();
        assert_eq!(m.num_faces(), 7);
        // The side faces had their shared edges split, so they are pentagons
        // now; the back face is still a quad.
        let back = m
            .face_ids()
            .find(|f| m.face_normal(*f).normalize().y > 0.9)
            .expect("a back face");
        assert_eq!(m.face_verts(back).len(), 4);
    }

    #[test]
    fn a_stroke_around_a_corner_follows_the_surface() {
        // From the front face over the top edge onto the top face: the cut
        // walks both.
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let before = m.num_faces();
        knife(
            &mut m,
            &[
                Vec3::new(-1.0, -1.0, 0.5),
                Vec3::new(0.0, -1.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
            ],
            (Vec3::Y - Vec3::Z).normalize(),
        )
        .unwrap();
        m.assert_valid();
        assert!(m.num_faces() >= before + 2, "{}", m.num_faces());
    }
}
