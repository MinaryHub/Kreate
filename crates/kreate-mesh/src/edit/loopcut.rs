//! Loop cut.
//!
//! Runs one or more new edge loops across a ring of quads. The plan calls its
//! user value very high, and it is: a loop cut is how a modeller adds resolution
//! exactly where it is needed instead of subdividing everything.
//!
//! # It is the ring walk plus one careful detail
//!
//! The topology is easy once [`edge_ring`](super::walk::edge_ring) has given the
//! strip of faces. The detail that is not easy is **orientation**. Each ring edge
//! is shared by two faces, and the two see it wound in opposite directions. If
//! each face places its cut point at `t` from whichever endpoint it happens to
//! meet first, the cut alternates sides down the strip and comes out as a
//! zigzag — which still validates, still renders, and is obviously wrong only
//! once someone looks at it.
//!
//! The fix is one line, and it is worth naming precisely because it looks like a
//! typo: a face reads its **entry** edge forwards, `a → b`, and its **exit** edge
//! *backwards*, `d → c`. Since the next face along sees that same edge wound the
//! other way, its own `a → b` is this face's `d → c`, and the two agree without
//! any coordination. Reading the exit edge forwards instead is exactly the
//! zigzag.
//!
//! # Neighbours
//!
//! A ring can end against a face that is not part of it — a triangle, an n-gon,
//! anything the walk refused to cross. Splitting the shared edge would leave
//! that face with a vertex sitting in the middle of one of its edges and
//! belonging to neither side. Those faces are rebuilt with the new vertices
//! inserted, the same rule subdivide uses.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, EdgeId, FaceId, LoopId, VertId};
use crate::bmesh::{BMesh, BmError, SelectMode};
use crate::Domain;

use super::walk::edge_ring;

/// What a loop cut produced.
#[derive(Clone, Debug, Default)]
pub struct LoopCut {
    /// The faces that replaced the ring's.
    pub faces: Vec<FaceId>,
    /// The new vertices, in ring order, one group per cut.
    pub verts: Vec<VertId>,
    /// Non-ring faces rebuilt to take a vertex they did not ask for.
    pub patched_neighbours: usize,
}

/// The split vertices along one edge, remembered with the direction they were
/// laid out in.
struct Splits {
    from: VertId,
    verts: Vec<VertId>,
}

/// Cuts `cuts` loops across the ring through `seed`.
///
/// `position` is where the cut sits between the ring edges' endpoints, from
/// zero to one. With more than one cut they are spread evenly and `position`
/// shifts the whole group, which is what the interactive slide does.
pub fn loop_cut(
    mesh: &mut BMesh,
    seed: EdgeId,
    cuts: u32,
    position: f32,
) -> Result<LoopCut, BmError> {
    let mut out = LoopCut::default();
    let cuts = cuts.max(1) as usize;
    if mesh.edge(seed).is_none() {
        return Ok(out);
    }

    let ring = edge_ring(mesh, seed);
    if ring.len() < 2 {
        return Ok(out);
    }

    // --- read ---------------------------------------------------------------
    // Consecutive ring edges, each pair naming one face of the strip. Derived
    // from the ring rather than from `face_loop` so that *which* of a face's two
    // ring edges is the entry follows the walk order.
    //
    // Picking the entry as whichever ring edge the loop cycle happens to list
    // first is the subtle version of the zigzag: each face is then internally
    // straight but disagrees with its neighbour about which end of the shared
    // edge is which, and the cut jumps once per face.
    let mut pairs: Vec<(EdgeId, EdgeId)> = ring.windows(2).map(|w| (w[0], w[1])).collect();
    if ring.len() > 2 {
        let (first, last) = (ring[0], ring[ring.len() - 1]);
        let closes = mesh
            .edge_faces(last)
            .into_iter()
            .any(|f| mesh.edge_faces(first).contains(&f));
        if closes {
            pairs.push((last, first));
        }
    }

    // Each strip as `(entry loop, exit loop)`. Read before anything is written:
    // the loop cycles stop existing the moment the faces are removed.
    let mut strips: Vec<(LoopId, LoopId)> = Vec::with_capacity(pairs.len());
    let mut faces: Vec<FaceId> = Vec::with_capacity(pairs.len());
    for (e0, e1) in pairs {
        let shared: Vec<FaceId> = mesh
            .edge_faces(e0)
            .into_iter()
            .filter(|f| mesh.edge_faces(e1).contains(f))
            .collect();
        let [f] = shared[..] else { continue };
        if mesh.face(f).map(|x| x.len()) != Some(4) {
            continue;
        }
        let Some(entry) = mesh
            .face_loops(f)
            .into_iter()
            .find(|&l| mesh.loop_edge(l) == Some(e0))
        else {
            continue;
        };
        let Some(exit) = mesh.loop_next(entry).and_then(|l| mesh.loop_next(l)) else {
            continue;
        };
        if mesh.loop_edge(exit) != Some(e1) {
            continue;
        }
        strips.push((entry, exit));
        faces.push(f);
    }
    if strips.is_empty() {
        return Ok(out);
    }
    let mut in_loop = vec![false; mesh.slot_count(Domain::Face)];
    for &f in &faces {
        in_loop[f.index() as usize] = true;
    }

    // Corners, per strip, read while the cycles are intact.
    let corners: Vec<[VertId; 4]> = strips
        .iter()
        .filter_map(|&(entry, exit)| {
            Some([
                mesh.loop_vert(entry)?,
                mesh.loop_vert(mesh.loop_next(entry)?)?,
                mesh.loop_vert(exit)?,
                mesh.loop_vert(mesh.loop_next(exit)?)?,
            ])
        })
        .collect();
    let strip_edges: Vec<[EdgeId; 2]> = strips
        .iter()
        .filter_map(|&(entry, exit)| Some([mesh.loop_edge(entry)?, mesh.loop_edge(exit)?]))
        .collect();
    if corners.len() != strips.len() || strip_edges.len() != strips.len() {
        return Ok(out);
    }

    // The cut positions along an edge. Evenly spread, then shifted by
    // `position - 0.5` so that a single cut at 0.5 sits in the middle and the
    // slider moves the whole group without reordering it.
    //
    // `position` runs from the seed edge's first end. The strips lay each
    // ring edge out from the end their face's winding reaches first, which
    // for the seed is that end or the other one by chance — and the slide,
    // which measures the pointer along the seed from its first end, went
    // the wrong way half the time. Measured from the other end, the same
    // place is `1 - position`: the evenly spread group is symmetric about
    // the middle.
    let seed_from = strip_edges.iter().zip(&corners).find_map(|([e0, e1], c)| {
        if *e0 == seed {
            Some(c[0])
        } else if *e1 == seed {
            Some(c[3])
        } else {
            None
        }
    });
    let position = position.clamp(0.0, 1.0);
    let position = match (seed_from, mesh.edge_verts(seed)) {
        (Some(from), Some([first, _])) if from != first => 1.0 - position,
        _ => position,
    };
    let shift = position - 0.5;
    let ts: Vec<f32> = (0..cuts)
        .map(|i| ((i + 1) as f32 / (cuts + 1) as f32 + shift).clamp(0.02, 0.98))
        .collect();

    // --- write --------------------------------------------------------------
    let mut splits: Vec<Option<Splits>> =
        (0..mesh.slot_count(Domain::Edge)).map(|_| None).collect();

    let place = |mesh: &mut BMesh,
                 splits: &mut Vec<Option<Splits>>,
                 e: EdgeId,
                 from: VertId,
                 to: VertId|
     -> Vec<VertId> {
        let slot = e.index() as usize;
        if let Some(existing) = &splits[slot] {
            // Two faces share every interior ring edge, and by the entry/exit
            // rule above they always ask for it from the same end. Asserted
            // rather than silently corrected: a fallback here would hide a
            // broken walk behind a cut that merely looks plausible, and the
            // assertion names the invariant for whoever changes this next.
            debug_assert_eq!(
                existing.from, from,
                "a ring edge was laid out from both ends; the entry/exit \
                 orientation rule has been broken"
            );
            return existing.verts.clone();
        }
        let (Some(a), Some(b)) = (mesh.position(from), mesh.position(to)) else {
            return Vec::new();
        };
        let verts: Vec<VertId> = ts.iter().map(|t| mesh.add_vert(a.lerp(b, *t))).collect();
        splits[slot] = Some(Splits {
            from,
            verts: verts.clone(),
        });
        verts
    };

    let mut new_rings: Vec<(Vec<VertId>, Vec<VertId>, [VertId; 4])> =
        Vec::with_capacity(strips.len());
    for (i, [a, b, c, d]) in corners.iter().copied().enumerate() {
        let [e0, e1] = strip_edges[i];
        // `a → b` on the entry edge, and `d → c` on the exit edge: `d` is the
        // corner adjacent to `a`, so laying both out from those ends is what
        // makes the cut run straight across rather than diagonally.
        let near = place(mesh, &mut splits, e0, a, b);
        let far = place(mesh, &mut splits, e1, d, c);
        if near.len() != cuts || far.len() != cuts {
            continue;
        }
        new_rings.push((near, far, [a, b, c, d]));
    }
    for (near, far, _) in &new_rings {
        // Both sides. Recording only the near one loses the last ring edge's
        // vertices — the strips overlap everywhere except at the two ends — and
        // the caller then selects an incomplete loop.
        out.verts.extend(near.iter().copied());
        out.verts.extend(far.iter().copied());
    }
    out.verts.sort_unstable_by_key(|v| v.index());
    out.verts.dedup();

    // Neighbouring faces that share a split edge but are not part of the ring.
    let split_edges: Vec<EdgeId> = ring
        .iter()
        .copied()
        .filter(|e| splits[e.index() as usize].is_some())
        .collect();
    let mut neighbours: Vec<FaceId> = Vec::new();
    for &e in &split_edges {
        for f in mesh.edge_faces(e) {
            if !in_loop[f.index() as usize] && !neighbours.contains(&f) {
                neighbours.push(f);
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

    for &f in &faces {
        mesh.remove_face(f);
    }
    for &f in &neighbours {
        mesh.remove_face(f);
    }
    for &e in &split_edges {
        if mesh.edge(e).is_some() && mesh.is_wire_edge(e) {
            mesh.remove_edge(e);
        }
    }

    for (near, far, [a, b, c, d]) in new_rings {
        // Strips from the `a`/`d` side across to the `b`/`c` side.
        let mut left = a;
        let mut right = d;
        for k in 0..cuts {
            out.faces
                .push(mesh.add_face(&[left, near[k], far[k], right])?);
            left = near[k];
            right = far[k];
        }
        out.faces.push(mesh.add_face(&[left, b, c, right])?);
    }

    for ring in neighbour_rings {
        let mut patched = Vec::with_capacity(ring.len() * 2);
        for (v, e) in ring {
            patched.push(v);
            if let Some(s) = &splits[e.index() as usize] {
                let mut extra = s.verts.clone();
                if s.from != v {
                    extra.reverse();
                }
                patched.extend(extra);
            }
        }
        if patched.len() >= 3 {
            mesh.add_face(&patched)?;
            out.patched_neighbours += 1;
        }
    }

    mesh.select_none();
    for v in &out.verts {
        mesh.set_vert_selected(*v, true);
    }
    mesh.flush_selection(SelectMode::Vert);
    Ok(out)
}

/// Cuts a loop across the ring through the first selected edge.
pub fn loop_cut_selected(mesh: &mut BMesh, cuts: u32, position: f32) -> Result<LoopCut, BmError> {
    let Some(seed) = mesh.selected_edges().first().copied() else {
        return Ok(LoopCut::default());
    };
    loop_cut(mesh, seed, cuts, position)
}

/// The vertices a loop cut would create, without changing anything.
///
/// What the interactive preview draws before the user commits, and what makes
/// the tool feel like it is answering rather than guessing.
pub fn preview(mesh: &BMesh, seed: EdgeId, position: f32) -> Vec<Vec3> {
    let ring = edge_ring(mesh, seed);
    let t = position.clamp(0.02, 0.98);
    ring.iter()
        .filter_map(|&e| {
            let [a, b] = mesh.edge_verts(e)?;
            Some(mesh.position(a)?.lerp(mesh.position(b)?, t))
        })
        .collect()
}

/// The ring a loop cut through edge `seed` of a document mesh would cross: each
/// ring edge with the end its cut is measured from, in walk order, and whether
/// the ring closes on itself.
///
/// On the [`MeshData`] rather than the edit structure, because it runs on every
/// pointer move while `Ctrl+R` is looking for an edge — a decode per move would
/// be the whole cost of the hover. The same walk as the cut's: across quads to
/// the opposite edge, stopping at anything that is not a quad or at an edge
/// with other than two faces.
pub fn ring_on(mesh: &crate::MeshData, seed: u32) -> (Vec<(u32, u32)>, bool) {
    let Some(&[a, _]) = mesh.edges().get(seed as usize) else {
        return (Vec::new(), false);
    };
    let corner_edges = mesh.corner_edges();
    let mut faces_of: Vec<Vec<u32>> = vec![Vec::new(); mesh.num_edges() as usize];
    for f in 0..mesh.num_faces() {
        for c in mesh.face_corner_range(f) {
            faces_of[corner_edges[c] as usize].push(f);
        }
    }
    // One step: from `edge`, measured from `start`, across `face`.
    let step = |edge: u32, start: u32, face: u32| -> Option<(u32, u32)> {
        let range = mesh.face_corner_range(face);
        if range.len() != 4 {
            return None;
        }
        let verts = mesh.face_verts(face);
        let i = range.clone().position(|c| corner_edges[c] == edge)?;
        let next_start = if verts[i] == start {
            verts[(i + 3) % 4]
        } else {
            verts[(i + 2) % 4]
        };
        Some((corner_edges[range.start + (i + 2) % 4], next_start))
    };
    let mut forward = vec![(seed, a)];
    let mut closed = false;
    let mut backward = Vec::new();
    let seed_faces = faces_of[seed as usize].clone();
    if seed_faces.len() > 2 {
        return (forward, false);
    }
    for (side, &first) in seed_faces.iter().enumerate() {
        let out = if side == 0 {
            &mut forward
        } else {
            &mut backward
        };
        let (mut edge, mut start, mut face) = (seed, a, first);
        while let Some((next, next_start)) = step(edge, start, face) {
            if next == seed {
                closed = true;
                break;
            }
            // A walk that comes back on itself without closing — a ring that
            // crosses its own path — stops where it would repeat.
            if out.iter().any(|(e, _)| *e == next) {
                break;
            }
            out.push((next, next_start));
            let around = &faces_of[next as usize];
            match around.iter().find(|f| **f != face) {
                Some(f) if around.len() == 2 => {
                    (edge, start, face) = (next, next_start, *f);
                }
                _ => break,
            }
        }
        if closed {
            return (forward, true);
        }
    }
    backward.reverse();
    backward.extend(forward);
    (backward, false)
}

/// Where `cuts` loops through edge `seed` would run, one line each, in the
/// mesh's own space: what `Ctrl+R` draws under the pointer before the click.
/// Evenly spread along the ring's edges, as a cut at the middle places them.
pub fn preview_lines(mesh: &crate::MeshData, seed: u32, cuts: u32) -> Vec<Vec<Vec3>> {
    let (ring, closed) = ring_on(mesh, seed);
    if ring.len() < 2 {
        return Vec::new();
    }
    let positions = mesh.positions();
    let edges = mesh.edges();
    let cuts = cuts.max(1);
    (0..cuts)
        .map(|k| {
            let t = (k + 1) as f32 / (cuts + 1) as f32;
            let mut line: Vec<Vec3> = ring
                .iter()
                .map(|(e, start)| {
                    let [p, q] = edges[*e as usize];
                    let end = if p == *start { q } else { p };
                    positions[*start as usize].lerp(positions[end as usize], t)
                })
                .collect();
            if closed {
                line.push(line[0]);
            }
            line
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    /// An edge of a grid running along X at the given row.
    fn edge_along_x(mesh: &BMesh, y: f32) -> EdgeId {
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
    fn one_cut_across_a_grid_adds_a_row() {
        // The cut runs *across* the ring the seed belongs to, which is
        // perpendicular to the seed itself: hovering a horizontal edge gives a
        // vertical cut, the way `Ctrl+R` does.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        assert_eq!(m.num_faces(), 4);
        let seed = edge_along_x(&m, 0.0);

        let out = loop_cut(&mut m, seed, 1, 0.5).unwrap();
        m.assert_valid();

        assert_eq!(out.faces.len(), 4, "each of the two ring faces became two");
        assert_eq!(m.num_faces(), 6);
        assert_eq!(out.verts.len(), 3, "one per ring edge, ends included");
        assert_eq!(m.num_verts(), 9 + 3);
        for f in m.face_ids() {
            assert_eq!(m.face(f).unwrap().len(), 4, "still all quads");
        }
    }

    /// Where a cut lands on each ring edge, as a fraction from a fixed end.
    ///
    /// The measure a zigzag fails: a straight cut is at the same fraction along
    /// every ring edge, a zigzagging one alternates between `t` and `1 - t`.
    fn cut_fractions(before: &BMesh, after: &BMesh, cut: &LoopCut, seed: EdgeId) -> Vec<f32> {
        let ring = edge_ring(before, seed);
        let mut out = Vec::new();
        for e in ring {
            let Some([a, b]) = before.edge_verts(e) else {
                continue;
            };
            let (Some(pa), Some(pb)) = (before.position(a), before.position(b)) else {
                continue;
            };
            let span = pb - pa;
            let length2 = span.length_squared();
            if length2 < 1e-9 {
                continue;
            }
            for v in &cut.verts {
                let Some(p) = after.position(*v) else {
                    continue;
                };
                let t = (p - pa).dot(span) / length2;
                if (0.0..=1.0).contains(&t) && ((pa + span * t) - p).length() < 1e-4 {
                    out.push(t);
                }
            }
        }
        out
    }

    #[test]
    fn the_cut_sits_at_the_same_fraction_along_every_ring_edge() {
        // The zigzag, measured directly. Reading a face's exit edge forwards
        // instead of backwards puts alternate cuts at `1 - t`, which on a
        // symmetric grid is invisible at `t = 0.5` and obvious anywhere else —
        // so this asks for an asymmetric position on purpose.
        let before = BMesh::from_mesh_data(&primitives::grid(8.0, 4, 4));
        let seed = edge_along_x(&before, 0.0);
        let mut m = before.clone();
        let out = loop_cut(&mut m, seed, 1, 0.2).unwrap();
        m.assert_valid();

        let fractions = cut_fractions(&before, &m, &out, seed);
        assert!(fractions.len() >= 4, "got {fractions:?}");
        let first = fractions[0];
        for t in &fractions {
            assert!(
                (t - first).abs() < 1e-3,
                "the cut alternates sides: {fractions:?}"
            );
        }
    }

    #[test]
    fn the_cut_runs_straight_rather_than_zigzagging() {
        // The orientation bug this file is mostly about. Each ring edge is seen
        // wound opposite ways by its two faces; placing `t` from whichever end a
        // face meets first makes the cut alternate sides down the strip. It
        // still validates and still renders — it is only visibly wrong.
        let mut m = BMesh::from_mesh_data(&primitives::grid(8.0, 4, 4));
        let seed = edge_along_x(&m, 0.0);
        let out = loop_cut(&mut m, seed, 1, 0.25).unwrap();
        m.assert_valid();

        // The seed runs along X, so the ring is the parallel edges in the rows
        // above and below and the cut runs along Y at a constant X.
        let xs: Vec<f32> = out
            .verts
            .iter()
            .filter_map(|v| m.position(*v))
            .map(|p| p.x)
            .collect();
        assert!(xs.len() >= 4, "got {xs:?}");
        let first = xs[0];
        for x in &xs {
            assert!(
                (x - first).abs() < 1e-4,
                "the cut wandered: {xs:?} — a zigzag, not a loop"
            );
        }
    }

    #[test]
    fn the_position_places_the_cut() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        let seed = edge_along_x(&m, 0.0);
        let [a, b] = m.edge_verts(seed).unwrap();
        let (pa, pb) = (m.position(a).unwrap(), m.position(b).unwrap());

        let out = loop_cut(&mut m, seed, 1, 0.25).unwrap();
        m.assert_valid();
        let x = m.position(out.verts[0]).unwrap().x;

        // A quarter along the seed from its first end — the end the slide
        // measures the pointer from. Either end used to do, and the cut then
        // slid against the pointer on every edge the walk took backwards.
        let fraction = (x - pa.x) / (pb.x - pa.x);
        assert!(
            (fraction - 0.25).abs() < 1e-4,
            "the cut sits at {fraction} of the way along the seed"
        );
    }

    /// Every edge of a cube, each way round the walk may take it: a quarter
    /// from its first end, every time.
    #[test]
    fn the_position_runs_from_the_seeds_first_end_on_every_edge() {
        let base = BMesh::from_mesh_data(&primitives::cube(2.0));
        let edges: Vec<EdgeId> = base.edge_ids().collect();
        for seed in edges {
            let mut m = base.clone();
            let [a, b] = m.edge_verts(seed).unwrap();
            let (pa, pb) = (m.position(a).unwrap(), m.position(b).unwrap());
            let out = loop_cut(&mut m, seed, 1, 0.25).unwrap();
            let on_seed = out
                .verts
                .iter()
                .filter_map(|v| m.position(*v))
                .find(|p| {
                    let d = pb - pa;
                    let t = (*p - pa).dot(d) / d.length_squared();
                    (pa + d * t).distance(*p) < 1e-5 && (0.0..=1.0).contains(&t)
                })
                .expect("a cut vertex on the seed");
            let fraction = (on_seed - pa).length() / (pb - pa).length();
            assert!(
                (fraction - 0.25).abs() < 1e-4,
                "{seed:?}: {fraction} of the way from its first end"
            );
        }
    }

    #[test]
    fn several_cuts_are_spread_evenly() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 1, 1));
        assert_eq!(m.num_faces(), 1);
        let seed = m.edge_ids().next().unwrap();

        let out = loop_cut(&mut m, seed, 3, 0.5).unwrap();
        m.assert_valid();
        assert_eq!(out.faces.len(), 4, "one quad split into four strips");
        assert_eq!(out.verts.len(), 6, "three per ring edge, two ring edges");
        let mut xs: Vec<f32> = out
            .verts
            .iter()
            .filter_map(|v| m.position(*v))
            .map(|p| p.x)
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
        assert_eq!(xs.len(), 3, "three distinct cut lines: {xs:?}");
        let gaps: Vec<f32> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(
            (gaps[0] - gaps[1]).abs() < 1e-4,
            "the cuts are not evenly spread: {gaps:?}"
        );
        for f in m.face_ids() {
            assert_eq!(m.face(f).unwrap().len(), 4);
        }
    }

    #[test]
    fn a_cut_around_a_cylinder_closes_on_itself() {
        let mut m = BMesh::from_mesh_data(&primitives::cylinder(1.0, 2.0, 8));
        let vertical = m
            .edge_ids()
            .find(|&e| {
                let [a, b] = m.edge_verts(e).unwrap();
                (m.position(a).unwrap().z - m.position(b).unwrap().z).abs() > 0.5
            })
            .unwrap();

        let out = loop_cut(&mut m, vertical, 1, 0.5).unwrap();
        m.assert_valid();

        assert_eq!(
            out.verts.len(),
            8,
            "one per vertical edge, all the way round"
        );
        assert_eq!(out.faces.len(), 16, "each of the eight walls became two");
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "the solid stayed closed");
        }
        // Every new vertex sits at the same height, halfway up the wall.
        let bounds = m.to_mesh_data().bounds();
        let middle = (bounds.min.z + bounds.max.z) * 0.5;
        for v in &out.verts {
            assert!(
                (m.position(*v).unwrap().z - middle).abs() < 1e-4,
                "a cut vertex is off the belt"
            );
        }
    }

    #[test]
    fn a_cut_around_a_cube_makes_a_belt() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let seed = m.edge_ids().next().unwrap();
        let out = loop_cut(&mut m, seed, 1, 0.5).unwrap();
        m.assert_valid();

        assert_eq!(out.verts.len(), 4, "a cube's ring is four edges");
        assert_eq!(m.num_faces(), 6 + 4, "four of the six faces split");
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2);
        }
        assert_eq!(
            m.num_verts() as i64 - m.num_edges() as i64 + m.num_faces() as i64,
            2
        );
    }

    #[test]
    fn a_neighbour_outside_the_ring_gains_the_vertex() {
        // A ring that ends against a triangle: splitting the shared edge would
        // leave a vertex in the middle of the triangle's edge, belonging to
        // neither. The neighbour is rebuilt instead — the same rule subdivide
        // uses, and for the same reason.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 1));
        // Attach a triangle to one end of the strip.
        let border: Vec<EdgeId> = m.edge_ids().filter(|&e| m.is_boundary_edge(e)).collect();
        let pick = border
            .iter()
            .copied()
            .find(|&e| {
                let [a, b] = m.edge_verts(e).unwrap();
                let (pa, pb) = (m.position(a).unwrap(), m.position(b).unwrap());
                (pa.x - pb.x).abs() < 1e-4
            })
            .expect("a vertical border edge");
        let [a, b] = m.edge_verts(pick).unwrap();
        let tip = m.add_vert(Vec3::new(4.0, 0.0, 0.0));
        m.add_face(&[a, b, tip]).unwrap();
        m.assert_valid();

        let seed = m
            .edge_ids()
            .find(|&e| {
                let [x, y] = m.edge_verts(e).unwrap();
                let (px, py) = (m.position(x).unwrap(), m.position(y).unwrap());
                (px.y - py.y).abs() < 1e-4 && (px.x - py.x).abs() > 1e-4
            })
            .unwrap();
        let out = loop_cut(&mut m, seed, 1, 0.5).unwrap();
        m.assert_valid();

        for e in m.edge_ids() {
            assert!(
                m.edge_face_count(e) <= 2,
                "patching must not create a non-manifold edge"
            );
        }
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
        let _ = out;
    }

    #[test]
    fn a_ring_that_cannot_be_walked_cuts_nothing() {
        // A cone's cap is triangles, so there is no strip of quads to cut.
        let mut m = BMesh::from_mesh_data(&primitives::cone(1.0, 2.0, 8));
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        let cap_edge = m
            .edge_ids()
            .find(|&e| {
                m.edge_faces(e)
                    .iter()
                    .all(|f| m.face(*f).unwrap().len() == 3)
            })
            .expect("a cone has triangle-only edges");

        let out = loop_cut(&mut m, cap_edge, 1, 0.5).unwrap();
        m.assert_valid();
        assert!(out.faces.is_empty());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
    }

    #[test]
    fn the_new_loop_is_what_ends_up_selected() {
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        let seed = edge_along_x(&m, 0.0);
        let out = loop_cut(&mut m, seed, 1, 0.5).unwrap();

        let selected = m.selected_verts();
        assert_eq!(selected.len(), out.verts.len());
        for v in &out.verts {
            assert!(m.is_vert_selected(*v));
        }
        assert!(
            !m.selected_edges().is_empty(),
            "consecutive cut vertices make edges, which must flush selected"
        );
    }

    #[test]
    fn the_preview_matches_where_the_cut_lands() {
        // What the interactive tool draws before the user commits. If it
        // disagreed with the result, the tool would feel like it was guessing.
        let m0 = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        let seed = edge_along_x(&m0, 0.0);
        let shown = preview(&m0, seed, 0.3);

        let mut m = m0.clone();
        let out = loop_cut(&mut m, seed, 1, 0.3).unwrap();
        let landed: Vec<Vec3> = out.verts.iter().filter_map(|v| m.position(*v)).collect();

        assert_eq!(shown.len(), landed.len());
        for p in &landed {
            assert!(
                shown.iter().any(|q| (*q - *p).length() < 1e-4),
                "the cut landed at {p:?}, which the preview did not show"
            );
        }
    }

    #[test]
    fn cutting_a_mesh_leaves_it_bakeable() {
        for mesh in [
            primitives::cube(2.0),
            primitives::grid(4.0, 4, 3),
            primitives::cylinder(1.0, 2.0, 9),
            primitives::torus(1.0, 0.4, 8, 6),
        ] {
            let base = BMesh::from_mesh_data(&mesh);
            for seed in base.edge_ids() {
                let mut m = base.clone();
                let _ = loop_cut(&mut m, seed, 2, 0.5).unwrap();
                m.assert_valid();
                assert_eq!(m.to_mesh_data().validate(), Vec::new());
            }
        }
    }

    #[test]
    fn the_hover_ring_on_the_document_mesh_goes_all_the_way_round() {
        // A cube's edge rings are four edges round, and close.
        let cube = primitives::cube(2.0);
        let (ring, closed) = ring_on(&cube, 0);
        assert!(closed);
        assert_eq!(ring.len(), 4);
        let lines = preview_lines(&cube, 0, 2);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].len(), 5, "a closed ring returns to its start");
        // Every point sits a third of the way along a cube edge: two
        // coordinates at the cube's half size, one at a third between.
        for p in &lines[0] {
            let on_face = [p.x, p.y, p.z]
                .iter()
                .filter(|c| (c.abs() - 1.0).abs() < 1e-5)
                .count();
            assert_eq!(on_face, 2, "{p:?}");
        }
        // On an open grid the ring runs from border to border, both ways from
        // the edge it started at.
        let grid = primitives::grid(3.0, 3, 3);
        let middle = (0..grid.num_edges())
            .find(|e| {
                let [a, b] = grid.edges()[*e as usize];
                let (pa, pb) = (grid.positions()[a as usize], grid.positions()[b as usize]);
                pa.x.abs() < 0.6 && pb.x.abs() < 0.6 && pa.y.abs() < 0.6 && pb.y.abs() < 0.6
            })
            .expect("an inner edge");
        let (ring, closed) = ring_on(&grid, middle);
        assert!(!closed);
        assert_eq!(ring.len(), 4, "three quads across: four edges");
    }
}
