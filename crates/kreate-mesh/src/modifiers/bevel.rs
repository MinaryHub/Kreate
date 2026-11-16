//! The bevel modifier: sharp edges chamfered, without touching the cage.
//!
//! The same chamfer as the edit-mode bevel (`Ctrl+B`), applied to every edge
//! whose faces meet at more than `angle` — Blender's default Angle limit, which
//! is what makes a hard-surface model catch the light at its corners while the
//! mesh the user edits stays simple.

use crate::bmesh::BMesh;
use crate::MeshData;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BevelParams {
    /// How wide the chamfer is.
    pub width: f32,
    /// Edges whose faces turn by more than this, in radians, are bevelled.
    pub angle: f32,
    /// Strips across each edge: one is a flat chamfer, more round it.
    pub segments: u32,
}

impl Default for BevelParams {
    fn default() -> Self {
        Self {
            width: 0.1,
            angle: 30f32.to_radians(),
            segments: 1,
        }
    }
}

/// How many strips a fillet takes: enough that the round reads as round,
/// one more for every 15° of turn.
pub const FILLET_SEGMENTS: u32 = 6;

/// Every edge of a solid where its faces turn by more than 30° rounded to
/// `radius` — or, for a chamfer, cut back `radius` along each face. What a
/// machined part's "break all sharp edges" means; exact for edges at right
/// angles, which is what nearly every edge of one is.
pub fn round_edges(input: &MeshData, radius: f32, chamfer: bool) -> MeshData {
    bevel(
        input,
        &BevelParams {
            width: radius,
            angle: 30f32.to_radians(),
            segments: if chamfer { 1 } else { FILLET_SEGMENTS },
        },
    )
}

/// The edges of `mesh` where its faces turn by more than 30°, as its
/// vertices' pairs: what a CAD part calls its edges, the facets of a round
/// left out. An edge on one face only is one too.
pub fn sharp_edges(mesh: &MeshData) -> Vec<[u32; 2]> {
    let p = mesh.positions();
    let mut normals: std::collections::HashMap<(u32, u32), Vec<glam::Vec3>> = Default::default();
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        let mut n = glam::Vec3::ZERO;
        for i in 0..ring.len() {
            n += p[ring[i] as usize].cross(p[ring[(i + 1) % ring.len()] as usize]);
        }
        let n = n.normalize_or_zero();
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            normals.entry((a.min(b), a.max(b))).or_default().push(n);
        }
    }
    let cos_limit = 30f32.to_radians().cos();
    let mut out: Vec<[u32; 2]> = normals
        .into_iter()
        .filter(|(_, ns)| match ns.as_slice() {
            [a, b] => a.dot(*b) < cos_limit,
            _ => true,
        })
        .map(|((a, b), _)| [a, b])
        .collect();
    out.sort_unstable();
    out
}

/// The sharp edges that run on from `start` — through each vertex to the
/// one that carries on its way, turning by less than 30° — both ways: a
/// round edge's facets, a slot's straight and round edges, taken as one, as
/// a click on any of them takes the whole in Fusion. Indices into `edges`.
pub fn tangent_chain(mesh: &MeshData, edges: &[[u32; 2]], start: usize) -> Vec<usize> {
    let p = mesh.positions();
    let mut at: std::collections::HashMap<u32, Vec<usize>> = Default::default();
    for (i, [a, b]) in edges.iter().enumerate() {
        at.entry(*a).or_default().push(i);
        at.entry(*b).or_default().push(i);
    }
    let cos_limit = 30f32.to_radians().cos();
    let mut chain = vec![start];
    let mut seen: std::collections::HashSet<usize> = [start].into_iter().collect();
    for end in [edges[start][1], edges[start][0]] {
        // Walk on from this end, the way the edge leaves it.
        let (mut edge, mut v) = (start, end);
        loop {
            let [a, b] = edges[edge];
            let from = if b == v { a } else { b };
            let along = (p[v as usize] - p[from as usize]).normalize_or_zero();
            let next = at.get(&v).and_then(|list| {
                list.iter()
                    .copied()
                    .filter(|e| *e != edge && !seen.contains(e))
                    .map(|e| {
                        let [c, d] = edges[e];
                        let to = if c == v { d } else { c };
                        (
                            e,
                            to,
                            (p[to as usize] - p[v as usize])
                                .normalize_or_zero()
                                .dot(along),
                        )
                    })
                    .filter(|(_, _, cos)| *cos > cos_limit)
                    .max_by(|x, y| x.2.total_cmp(&y.2))
            });
            let Some((e, to, _)) = next else { break };
            seen.insert(e);
            chain.push(e);
            edge = e;
            v = to;
        }
    }
    chain
}

/// The sharp edge of `mesh` nearest `point`, within `within`: an index into
/// `edges`.
pub fn nearest_edge(
    mesh: &MeshData,
    edges: &[[u32; 2]],
    point: glam::Vec3,
    within: f32,
) -> Option<usize> {
    let p = mesh.positions();
    edges
        .iter()
        .enumerate()
        .map(|(i, [a, b])| {
            let (a, b) = (p[*a as usize], p[*b as usize]);
            let ab = b - a;
            let t = ((point - a).dot(ab) / ab.length_squared().max(1e-30)).clamp(0.0, 1.0);
            (i, (a + ab * t).distance(point))
        })
        .filter(|(_, d)| *d <= within)
        .min_by(|x, y| x.1.total_cmp(&y.1))
        .map(|(i, _)| i)
}

/// The edges a fillet on `points` takes: each point's nearest sharp edge
/// and the chain it runs on in, as segments. Nothing for a point on none.
pub fn chosen_edges(mesh: &MeshData, points: &[glam::Vec3]) -> Vec<[glam::Vec3; 2]> {
    let edges = sharp_edges(mesh);
    let p = mesh.positions();
    let size = mesh.bounds().size().length().max(1e-9);
    let mut taken: std::collections::BTreeSet<usize> = Default::default();
    for point in points {
        if let Some(start) = nearest_edge(mesh, &edges, *point, size * 1e-3) {
            taken.extend(tangent_chain(mesh, &edges, start));
        }
    }
    taken
        .into_iter()
        .map(|i| [p[edges[i][0] as usize], p[edges[i][1] as usize]])
        .collect()
}

/// `input` with only the edges at `points` rounded to `radius` — or cut
/// back, for a chamfer — each with the chain of edges it runs on in. The
/// chosen-edges fillet of a CAD program; with no points, [`round_edges`].
pub fn round_chosen(
    input: &MeshData,
    radius: f32,
    chamfer: bool,
    points: &[glam::Vec3],
) -> MeshData {
    if points.is_empty() {
        return round_edges(input, radius, chamfer);
    }
    if input.is_empty() || radius.is_nan() || radius <= 0.0 {
        return input.clone();
    }
    let segments = chosen_edges(input, points);
    if segments.is_empty() {
        return input.clone();
    }
    let mut bm = BMesh::from_mesh_data(input);
    let size = input.bounds().size().length().max(1e-9);
    let same = |a: glam::Vec3, b: glam::Vec3| a.distance(b) <= size * 1e-6;
    let edges: Vec<_> = bm
        .edge_ids()
        .filter(|e| {
            let Some([a, b]) = bm.edge_verts(*e) else {
                return false;
            };
            let (Some(pa), Some(pb)) = (bm.position(a), bm.position(b)) else {
                return false;
            };
            segments
                .iter()
                .any(|[x, y]| (same(pa, *x) && same(pb, *y)) || (same(pa, *y) && same(pb, *x)))
        })
        .collect();
    if edges.is_empty() {
        return input.clone();
    }
    let strips = if chamfer { 1 } else { FILLET_SEGMENTS };
    match crate::edit::bevel_edges_segments(&mut bm, &edges, radius, strips) {
        Ok(_) => bm.to_mesh_data(),
        Err(_) => input.clone(),
    }
}

/// Chamfers `input`'s sharp edges.
pub fn bevel(input: &MeshData, params: &BevelParams) -> MeshData {
    if input.is_empty() || params.width.is_nan() || params.width <= 0.0 {
        return input.clone();
    }
    let mut bm = BMesh::from_mesh_data(input);
    let cos_limit = params.angle.clamp(0.0, std::f32::consts::PI).cos();
    let edges: Vec<_> = bm
        .edge_ids()
        .filter(|e| {
            let faces = bm.edge_faces(*e);
            let [a, b] = faces[..] else { return false };
            let (na, nb) = (
                bm.face_normal(a).normalize_or_zero(),
                bm.face_normal(b).normalize_or_zero(),
            );
            na.dot(nb) < cos_limit
        })
        .collect();
    if edges.is_empty() {
        return input.clone();
    }
    match crate::edit::bevel_edges_segments(&mut bm, &edges, params.width, params.segments) {
        Ok(_) => bm.to_mesh_data(),
        // A bevel that cannot be built draws the cage rather than nothing.
        Err(_) => input.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_cube_gets_its_twelve_edges_chamfered_and_a_plane_nothing() {
        let cube = primitives::cube(2.0);
        let out = bevel(&cube, &BevelParams::default());
        assert!(out.num_faces() > cube.num_faces(), "{}", out.num_faces());
        // Still inside the cube's bounds: a chamfer cuts, it does not grow.
        assert!(out.bounds().max.x <= 1.0 + 1e-5);
        let plane = primitives::plane(2.0);
        assert_eq!(bevel(&plane, &BevelParams::default()).num_faces(), 1);
    }

    /// A cube's twelve edges are its sharp ones; a click on one top edge
    /// takes that edge alone (the next turns 90°); a cylinder's rim is
    /// taken whole from a click on one facet of it.
    #[test]
    fn a_click_takes_an_edge_and_the_edges_it_runs_on_in() {
        let cube = primitives::cube(2.0);
        assert_eq!(sharp_edges(&cube).len(), 12);
        let top = chosen_edges(&cube, &[glam::Vec3::new(0.0, 1.0, 1.0)]);
        assert_eq!(top.len(), 1, "{top:?}");
        let rounded = round_chosen(&cube, 0.2, false, &[glam::Vec3::new(0.0, 1.0, 1.0)]);
        assert!(rounded.num_faces() > cube.num_faces());
        assert!(crate::measure::is_closed(&rounded));
        // Only that edge: the opposite top edge's corner is still there.
        assert!(rounded
            .positions()
            .iter()
            .any(|p| p.distance(glam::Vec3::new(1.0, -1.0, 1.0)) < 1e-5));
        assert!(!rounded
            .positions()
            .iter()
            .any(|p| p.distance(glam::Vec3::new(0.0, 1.0, 1.0)) < 1e-3
                && (p.y - 1.0).abs() < 1e-5
                && (p.z - 1.0).abs() < 1e-5));

        let cylinder = primitives::cylinder(1.0, 2.0, 32);
        let rim = chosen_edges(&cylinder, &[glam::Vec3::new(1.0, 0.0, 1.0)]);
        assert_eq!(rim.len(), 32, "the whole top rim");
        // No points: every sharp edge, as before.
        assert_eq!(
            round_chosen(&cube, 0.2, true, &[]).num_faces(),
            round_edges(&cube, 0.2, true).num_faces()
        );
    }
}
