//! Uniform Catmull–Clark subdivision.
//!
//! The smooth one. Simple subdivision refines topology and leaves the shape
//! alone; this pulls the surface toward a limit surface, which is what a user
//! reaching for "subdivision surface" means. Both are kept, because a modeller
//! wants simple subdivision too — for adding topology to a shape that is already
//! right.
//!
//! # The rules, and why each is what it is
//!
//! One level replaces every n-gon with n quads, built from three new kinds of
//! point:
//!
//! - a **face point** at the centroid of each face;
//! - an **edge point** at `(v1 + v2 + f1 + f2) / 4` — the two endpoints and the
//!   two adjacent face points;
//! - a **vertex point** moved to `(F + 2R + (n − 3) v) / n`, where `F` averages
//!   the adjacent face points, `R` averages the adjacent edge midpoints, and `n`
//!   is the valence.
//!
//! Those weights are not tunable choices: they are what makes the limit surface
//! `C²` everywhere except at irregular vertices, where it is `C¹`. Changing them
//! produces a surface that looks nearly right and develops creases that nobody
//! put there.
//!
//! # Boundaries and creases share one rule
//!
//! An open boundary and a creased edge want the same thing: the surface should
//! follow a curve rather than being pulled inward by a face that is not there.
//! So both use the cubic B-spline curve rule, `(a + 6v + b) / 8`, along the
//! sharp edges — and a vertex with three or more of them is a corner and does
//! not move at all.
//!
//! Treating boundaries as ordinary interior vertices is the single most visible
//! way to get this wrong: an open mesh shrinks away from its own border on every
//! level, and a plane subdivided four times ends up noticeably smaller than the
//! plane.
//!
//! # Partial creases
//!
//! A crease of 1, an edge marked sharp, a boundary and a non-manifold edge are
//! sharp at every level. A crease between 0 and 1 is sharp for a while: it is
//! worth [`CREASE_LEVELS`] levels at full weight, and each level spends one —
//! the semi-sharp creases of Pixar's subdivision, which is what a softly
//! rounded edge on a cushion or a phone wants. While it lasts, a sharpness of
//! one or more takes the sharp rule; a fraction of one blends the smooth and
//! the sharp results by that fraction, so the edge relaxes rather than
//! snapping smooth. A vertex takes the average sharpness of its creased edges.
//!
//! # What is deliberately not here
//!
//! - **Adaptive or GPU evaluation.** This is the CPU cage evaluator. The plan
//!   puts limit-surface evaluation and adaptive tessellation in OpenSubdiv's
//!   territory, as a separate project.
//! - **Smooth UV subdivision.** Corner data is interpolated linearly, which is
//!   what most tools default to. Subdividing UVs with the surface rules keeps
//!   texture distortion lower on curved surfaces and is worth having later.

use kreate_math::Vec3;

use crate::attribute::{names, Domain};
use crate::MeshData;

/// How many levels a crease just under 1 stays sharp for. A crease of `c`
/// below 1 has a sharpness of `c * CREASE_LEVELS`, and each level of
/// subdivision spends one.
pub const CREASE_LEVELS: f32 = 4.0;

/// How sharp an edge with crease weight `c` is: infinite at 1 and over.
fn crease_sharpness(c: f32) -> f32 {
    if c >= 1.0 {
        f32::INFINITY
    } else {
        c.max(0.0) * CREASE_LEVELS
    }
}

/// The crease an edge's halves carry to the next level: one level's
/// sharpness spent, or unchanged at full.
fn crease_after_a_level(c: f32) -> f32 {
    if c >= 1.0 {
        c
    } else {
        (c - 1.0 / CREASE_LEVELS).max(0.0)
    }
}

/// Applies one level of Catmull–Clark subdivision.
pub fn subdivide_once(input: &MeshData) -> MeshData {
    let nv = input.num_verts() as usize;
    let ne = input.num_edges() as usize;
    let nf = input.num_faces() as usize;
    if nv == 0 || nf == 0 {
        return input.clone();
    }

    let positions = input.positions();
    let topology = Topology::of(input);

    // Output vertices in three blocks: the moved originals, then one per edge,
    // then one per face. A fixed layout rather than an incremental push, because
    // every later step needs to name a point before it has been computed, and
    // "vertex 7's edge point" has to mean the same thing everywhere.
    let face_base = nv + ne;

    let face_points: Vec<Vec3> = (0..nf)
        .map(|f| {
            let ring = input.face_verts(f as u32);
            if ring.is_empty() {
                return Vec3::ZERO;
            }
            ring.iter().map(|v| positions[*v as usize]).sum::<Vec3>() / ring.len() as f32
        })
        .collect();

    let edge_points: Vec<Vec3> = (0..ne)
        .map(|e| {
            let [a, b] = input.edges()[e];
            let midpoint = (positions[a as usize] + positions[b as usize]) * 0.5;
            // A sharp or boundary edge takes the midpoint: its point must lie on
            // the polyline, or the crease is not a crease. An interior smooth
            // edge is pulled toward the two faces it joins; a partly sharp one
            // goes part of the way.
            let sharpness = topology.sharpness[e];
            if sharpness >= 1.0 || topology.edge_face_count[e] != 2 {
                return midpoint;
            }
            let smooth = (positions[a as usize]
                + positions[b as usize]
                + face_points[topology.edge_faces[e][0] as usize]
                + face_points[topology.edge_faces[e][1] as usize])
                * 0.25;
            smooth.lerp(midpoint, sharpness)
        })
        .collect();

    let vertex_points: Vec<Vec3> = (0..nv)
        .map(|v| vertex_point(input, &topology, &face_points, v))
        .collect();

    let mut out_positions = Vec::with_capacity(nv + ne + nf);
    out_positions.extend(vertex_points);
    out_positions.extend(edge_points);
    out_positions.extend(face_points);

    // Every n-gon becomes n quads, each spanning a corner: the vertex, the edge
    // point ahead of it, the face point, and the edge point behind. Wound in that
    // order so the quads keep the original face's orientation.
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(input.num_corners() as usize);
    // Source corners per output corner, for interpolating corner data.
    let mut corner_offsets: Vec<u32> = vec![0];
    let mut corner_sources: Vec<u32> = Vec::new();
    // Source face per output face.
    let mut face_order: Vec<u32> = Vec::with_capacity(faces.capacity());

    for f in 0..nf {
        let ring = input.face_verts(f as u32);
        let corners = input.face_corner_range(f as u32);
        let n = ring.len();
        if n < 3 {
            continue;
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            // The index is the subject here, not the vertex: the body also walks
            // `corner_edge` and both neighbours by position.
            let previous = (i + n - 1) % n;
            let next = (i + 1) % n;
            let ahead = topology.corner_edge[corners.start + i] as usize;
            let behind = topology.corner_edge[corners.start + previous] as usize;
            faces.push(vec![
                ring[i],
                (nv + ahead) as u32,
                (face_base + f) as u32,
                (nv + behind) as u32,
            ]);
            face_order.push(f as u32);

            let c = |k: usize| (corners.start + k) as u32;
            // The corner sitting on an original vertex takes that corner's value
            // exactly; the two on edge points take the mean of the corners either
            // side; the one at the centre takes the mean of the whole face.
            let quad: [&[u32]; 4] = [&[c(i)], &[c(i), c(next)], &[], &[c(previous), c(i)]];
            for (slot, sources) in quad.iter().enumerate() {
                if slot == 2 {
                    corner_sources.extend((0..n).map(c));
                } else {
                    corner_sources.extend_from_slice(sources);
                }
                corner_offsets.push(corner_sources.len() as u32);
            }
        }
    }

    let mut out = MeshData::from_faces(out_positions, &faces);
    carry_point_data(&mut out, input, &topology);
    carry_corner_data(&mut out, input, &corner_offsets, &corner_sources);
    carry_face_data(&mut out, input, &face_order);
    carry_edge_data(&mut out, input, nv, ne);
    // A partial crease spends a level's worth of its sharpness.
    if let Some(crease) = out.attributes_mut().make_mut_f32(names::CREASE) {
        for c in crease.iter_mut() {
            *c = crease_after_a_level(*c);
        }
    }
    out
}

/// Where one original vertex moves to.
fn vertex_point(input: &MeshData, topology: &Topology, face_points: &[Vec3], v: usize) -> Vec3 {
    let positions = input.positions();
    let here = positions[v];
    let edges = topology.vertex_edges(v);
    let valence = edges.len();

    // A vertex with fewer than three edges has no neighbourhood to average over
    // — a wire end, or a lone point. Leaving it alone is the only answer that is
    // not arbitrary.
    if valence < 3 && topology.sharp_at(v) < 2 {
        return here;
    }

    let sharp: Vec<u32> = edges
        .iter()
        .copied()
        .filter(|e| topology.creased(*e as usize))
        .collect();

    // Three or more creases meeting is a corner: it stays put. Anything else
    // rounds it off, and a cube whose corners have been creased on all three
    // edges would still come out spherical.
    //
    // Exactly two: the vertex lies along a crease or a boundary, and follows the
    // cubic B-spline curve through its two neighbours. Without this an open mesh
    // shrinks away from its own border on every level.
    let rule = match sharp.len() {
        0 | 1 => None,
        2 => {
            let mut sum = here * 6.0;
            for e in &sharp {
                let [a, b] = input.edges()[*e as usize];
                let other = if a as usize == v { b } else { a };
                sum += positions[other as usize];
            }
            Some(sum / 8.0)
        }
        _ => Some(here),
    };
    // How sharp the vertex is: its creased edges' average, where a boundary or
    // a full crease is sharp outright.
    let sharpness = if sharp.is_empty() {
        0.0
    } else {
        sharp.iter().map(|e| topology.edge_sharpness(*e as usize)).sum::<f32>() / sharp.len() as f32
    };
    match rule {
        Some(rule) if sharpness >= 1.0 => return rule,
        Some(rule) => return smooth_vertex(input, topology, face_points, v).lerp(rule, sharpness),
        None => {}
    }
    smooth_vertex(input, topology, face_points, v)
}

/// The interior rule for one vertex: where it moves with no crease about.
fn smooth_vertex(input: &MeshData, topology: &Topology, face_points: &[Vec3], v: usize) -> Vec3 {
    let positions = input.positions();
    let here = positions[v];
    let edges = topology.vertex_edges(v);
    let valence = edges.len();
    if valence == 0 {
        return here;
    }

    // The interior rule. These weights are what make the limit surface C² away
    // from irregular vertices; they are not a tunable blend.
    let n = valence as f32;
    let mut face_sum = Vec3::ZERO;
    for f in topology.vertex_faces(v) {
        face_sum += face_points[*f as usize];
    }
    let faces = topology.vertex_faces(v).len() as f32;
    let average_face = if faces > 0.0 { face_sum / faces } else { here };

    let mut edge_sum = Vec3::ZERO;
    for e in edges {
        let [a, b] = input.edges()[*e as usize];
        edge_sum += (positions[a as usize] + positions[b as usize]) * 0.5;
    }
    let average_edge = edge_sum / n;

    (average_face + average_edge * 2.0 + here * (n - 3.0)) / n
}

/// The adjacency one level of subdivision needs, built once.
struct Topology {
    /// Which edge each corner walks along, to the next corner of its face.
    corner_edge: Vec<u32>,
    /// Up to two faces per edge. A third is ignored for the *smoothing* rules —
    /// see [`Topology::of`].
    edge_faces: Vec<[u32; 2]>,
    edge_face_count: Vec<u32>,
    /// How sharp each edge is at this level: zero smooth, infinite for a full
    /// crease, a mark or a non-manifold fan, and a number of levels between.
    sharpness: Vec<f32>,
    vertex_edge_offsets: Vec<u32>,
    vertex_edge_list: Vec<u32>,
    vertex_face_offsets: Vec<u32>,
    vertex_face_list: Vec<u32>,
}

impl Topology {
    fn of(input: &MeshData) -> Self {
        let nv = input.num_verts() as usize;
        let ne = input.num_edges() as usize;
        let nf = input.num_faces() as usize;

        // Edges by sorted vertex pair, with binary search rather than a hash
        // map: the output geometry depends on this lookup, and a hasher with a
        // per-run seed would make the result depend on which run it was.
        let mut by_pair: Vec<([u32; 2], u32)> = input
            .edges()
            .iter()
            .enumerate()
            .map(|(i, [a, b])| {
                let mut key = [*a, *b];
                key.sort_unstable();
                (key, i as u32)
            })
            .collect();
        by_pair.sort_unstable();
        let find = |a: u32, b: u32| -> Option<u32> {
            let mut key = [a, b];
            key.sort_unstable();
            by_pair
                .binary_search_by_key(&key, |(k, _)| *k)
                .ok()
                .map(|i| by_pair[i].1)
        };

        let mut corner_edge = vec![u32::MAX; input.num_corners() as usize];
        let mut edge_faces = vec![[u32::MAX; 2]; ne];
        let mut edge_face_count = vec![0u32; ne];
        let mut vertex_edges: Vec<Vec<u32>> = vec![Vec::new(); nv];
        let mut vertex_faces: Vec<Vec<u32>> = vec![Vec::new(); nv];

        for f in 0..nf {
            let ring = input.face_verts(f as u32);
            let corners = input.face_corner_range(f as u32);
            for (i, v) in ring.iter().enumerate() {
                let next = ring[(i + 1) % ring.len()];
                if let Some(e) = find(*v, next) {
                    corner_edge[corners.start + i] = e;
                    let slot = edge_face_count[e as usize] as usize;
                    // A third face on one edge is non-manifold. The smoothing
                    // rules have no meaning there — "the two adjacent faces" is
                    // not a thing — so the edge is treated as a crease further
                    // down, and only the first two faces are recorded here.
                    if slot < 2 {
                        edge_faces[e as usize][slot] = f as u32;
                    }
                    edge_face_count[e as usize] += 1;
                }
                vertex_faces[*v as usize].push(f as u32);
            }
        }
        for (e, [a, b]) in input.edges().iter().enumerate() {
            vertex_edges[*a as usize].push(e as u32);
            vertex_edges[*b as usize].push(e as u32);
        }

        let crease = input.attributes().get_f32(names::CREASE);
        let marked = input.attributes().get_bool(names::SHARP_EDGE);
        let sharpness: Vec<f32> = (0..ne)
            .map(|e| {
                let flagged = marked.and_then(|m| m.get(e)).copied().unwrap_or(false);
                // Non-manifold counts as sharp: the interior rule needs exactly
                // two faces, and averaging over "the two of the three" would pick
                // an answer out of the face order.
                if flagged || edge_face_count[e] > 2 {
                    return f32::INFINITY;
                }
                crease_sharpness(crease.and_then(|c| c.get(e)).copied().unwrap_or(0.0))
            })
            .collect();

        let (vertex_edge_offsets, vertex_edge_list) = flatten(&vertex_edges);
        let (vertex_face_offsets, vertex_face_list) = flatten(&vertex_faces);
        Self {
            corner_edge,
            edge_faces,
            edge_face_count,
            sharpness,
            vertex_edge_offsets,
            vertex_edge_list,
            vertex_face_offsets,
            vertex_face_list,
        }
    }

    fn vertex_edges(&self, v: usize) -> &[u32] {
        slice_of(&self.vertex_edge_offsets, &self.vertex_edge_list, v)
    }

    fn vertex_faces(&self, v: usize) -> &[u32] {
        slice_of(&self.vertex_face_offsets, &self.vertex_face_list, v)
    }

    /// Whether an edge takes part in a crease rule: sharp at all, or a
    /// boundary.
    fn creased(&self, e: usize) -> bool {
        self.sharpness[e] > 0.0 || self.edge_face_count[e] != 2
    }

    /// An edge's sharpness, with a boundary as sharp as it gets.
    fn edge_sharpness(&self, e: usize) -> f32 {
        if self.edge_face_count[e] != 2 {
            f32::INFINITY
        } else {
            self.sharpness[e]
        }
    }

    /// How many sharp or boundary edges meet at a vertex.
    fn sharp_at(&self, v: usize) -> usize {
        self.vertex_edges(v)
            .iter()
            .filter(|e| self.creased(**e as usize))
            .count()
    }
}

fn flatten(lists: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(lists.len() + 1);
    let mut flat = Vec::new();
    offsets.push(0);
    for list in lists {
        flat.extend_from_slice(list);
        offsets.push(flat.len() as u32);
    }
    (offsets, flat)
}

fn slice_of<'a>(offsets: &[u32], flat: &'a [u32], i: usize) -> &'a [u32] {
    let start = offsets[i] as usize;
    let end = offsets[i + 1] as usize;
    &flat[start..end]
}

/// Point layers: originals keep their value, edge and face points blend.
fn carry_point_data(out: &mut MeshData, input: &MeshData, topology: &Topology) {
    let nv = input.num_verts() as usize;
    let ne = input.num_edges() as usize;
    let nf = input.num_faces() as usize;

    let mut offsets: Vec<u32> = Vec::with_capacity(nv + ne + nf + 1);
    let mut sources: Vec<u32> = Vec::new();
    offsets.push(0);
    for v in 0..nv {
        sources.push(v as u32);
        offsets.push(sources.len() as u32);
    }
    for e in 0..ne {
        let [a, b] = input.edges()[e];
        sources.extend_from_slice(&[a, b]);
        offsets.push(sources.len() as u32);
    }
    for f in 0..nf {
        sources.extend_from_slice(input.face_verts(f as u32));
        offsets.push(sources.len() as u32);
    }
    let _ = topology;

    let mut source = input.attributes().clone();
    // Position is what the rules just computed; it must not be blended over the
    // top of them.
    source.remove(names::POSITION);
    for layer in source.iter_domain(Domain::Point) {
        let blended = layer.data.interpolate(&offsets, &sources);
        out.attributes_mut()
            .insert(&layer.name, Domain::Point, blended);
    }
}

/// Corner layers: interpolated linearly within each face.
fn carry_corner_data(out: &mut MeshData, input: &MeshData, offsets: &[u32], sources: &[u32]) {
    let expected = out.num_corners() as usize;
    // The corner map is built alongside the faces, so a mismatch means the two
    // walks disagreed — and silently carrying the wrong data would put UVs a
    // corner out of step, which shows only once something is textured.
    debug_assert_eq!(
        offsets.len().saturating_sub(1),
        expected,
        "corner interpolation map does not match the output"
    );
    if offsets.len().saturating_sub(1) != expected {
        return;
    }
    for layer in input.attributes().iter_domain(Domain::Corner) {
        let blended = layer.data.interpolate(offsets, sources);
        out.attributes_mut()
            .insert(&layer.name, Domain::Corner, blended);
    }
}

/// Face layers: every quad inherits from the face it came from.
fn carry_face_data(out: &mut MeshData, input: &MeshData, order: &[u32]) {
    for layer in input.attributes().iter_domain(Domain::Face) {
        let gathered = layer.data.gather(order);
        out.attributes_mut()
            .insert(&layer.name, Domain::Face, gathered);
    }
}

/// Edge layers: the two halves of an original edge inherit it; the edges from an
/// edge point to a face point are new and take the layer's default.
///
/// Worked out from the endpoint blocks rather than by matching vertex pairs: an
/// output edge joining an original vertex to an edge point *is* half of that
/// edge, and which half is not something any layer cares about. A crease that
/// did not survive here would vanish on the first subdivision, which is the one
/// place a user is sure to notice.
fn carry_edge_data(out: &mut MeshData, input: &MeshData, nv: usize, ne: usize) {
    if input
        .attributes()
        .iter_domain(Domain::Edge)
        .next()
        .is_none()
    {
        return;
    }
    let order: Vec<u32> = out
        .edges()
        .iter()
        .map(|[a, b]| {
            let (a, b) = (*a as usize, *b as usize);
            let original = |i: usize| i < nv;
            let edge_point = |i: usize| (nv..nv + ne).contains(&i);
            match (original(a), edge_point(b), original(b), edge_point(a)) {
                (true, true, ..) => (b - nv) as u32,
                (.., true, true) => (a - nv) as u32,
                _ => u32::MAX,
            }
        })
        .collect();

    for layer in input.attributes().iter_domain(Domain::Edge) {
        let gathered = layer.data.gather(&order);
        out.attributes_mut()
            .insert(&layer.name, Domain::Edge, gathered);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::AttrData;
    use crate::primitives;
    use std::sync::Arc;

    /// A single quad in the XY plane.
    fn quad() -> MeshData {
        MeshData::from_faces(
            vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2, 3]],
        )
    }

    #[test]
    fn one_level_turns_each_ngon_into_n_quads() {
        let out = subdivide_once(&primitives::cube(2.0));
        out.assert_valid();
        assert_eq!(out.num_faces(), 24, "six quads become four each");
        for face in out.faces() {
            assert_eq!(face.len(), 4, "every output face is a quad");
        }
        // 8 original + 12 edge + 6 face points.
        assert_eq!(out.num_verts(), 26);
    }

    #[test]
    fn a_cube_is_pulled_toward_a_sphere() {
        // The whole difference from simple subdivision. A cube of side 2 has its
        // corners at a distance of sqrt(3); Catmull-Clark pulls them well inside
        // that, and repeated levels converge on a sphere-like surface.
        let cube = primitives::cube(2.0);
        let once = subdivide_once(&cube);
        let twice = subdivide_once(&once);

        let radius = |m: &MeshData| {
            m.positions()
                .iter()
                .map(|p| p.length())
                .fold(0.0f32, f32::max)
        };
        assert!(
            radius(&once) < radius(&cube) - 0.2,
            "the surface did not move inward: {} then {}",
            radius(&cube),
            radius(&once)
        );
        assert!(radius(&twice) < radius(&once));
    }

    #[test]
    fn the_original_vertices_move() {
        // The difference from simple subdivision, stated where it is visible:
        // simple subdivision leaves every original vertex exactly where it was
        // and only adds new ones. Catmull-Clark moves them.
        //
        // Not tested through the bounding box: a cube's box is unchanged by
        // subdivision, because the six face points sit at the face centres and
        // still reach ±1 on their own axis. The box would have made this look
        // like nothing happened.
        let cube = primitives::cube(2.0);
        let out = subdivide_once(&cube);
        for (v, before) in cube.positions().iter().enumerate() {
            let after = out.positions()[v];
            assert!(
                (after - *before).length() > 0.1,
                "vertex {v} did not move: {before:?} to {after:?}"
            );
        }
    }

    #[test]
    fn a_face_point_stays_on_the_original_surface() {
        // Face points are centroids of the original face and are not moved by
        // anything. That is what keeps a subdivided surface inscribed in its cage
        // rather than drifting off it.
        let cube = primitives::cube(2.0);
        let out = subdivide_once(&cube);
        let face_base = cube.num_verts() as usize + cube.num_edges() as usize;
        for f in 0..cube.num_faces() as usize {
            let point = out.positions()[face_base + f];
            let centroid: kreate_math::Vec3 = cube
                .face_verts(f as u32)
                .iter()
                .map(|v| cube.positions()[*v as usize])
                .sum::<kreate_math::Vec3>()
                / cube.face_verts(f as u32).len() as f32;
            assert!(
                (point - centroid).length() < 1e-6,
                "face {f}'s point drifted: {point:?} against {centroid:?}"
            );
        }
    }

    #[test]
    fn a_face_point_is_the_centroid() {
        let out = subdivide_once(&quad());
        // Layout: 4 originals, 4 edge points, then the face point.
        let face_point = out.positions()[8];
        assert!(
            face_point.length() < 1e-6,
            "the centre of a symmetric quad is the origin, got {face_point:?}"
        );
    }

    #[test]
    fn a_straight_boundary_stays_straight() {
        // The most visible way to get this wrong. Treated as interior vertices, a
        // boundary is pulled inward on every level and an open mesh shrinks away
        // from its own border.
        //
        // Tested on the straight *runs* of the border rather than on the whole
        // bounding box, because the corners are supposed to round: a border is a
        // closed cubic B-spline, so a square outline legitimately becomes a
        // rounded square. Only a vertex whose neighbours are collinear with it
        // has to stay put.
        let mut mesh = primitives::grid(2.0, 4, 4);
        let before = mesh.bounds();
        for _ in 0..3 {
            mesh = subdivide_once(&mesh);
        }

        // The midpoint of each border run: still on the border line, and not
        // pulled inward.
        let reaches = |axis: usize, value: f32| {
            mesh.positions()
                .iter()
                .any(|p| (p[axis] - value).abs() < 1e-4)
        };
        assert!(
            reaches(0, before.max.x) && reaches(0, before.min.x),
            "the border along X moved inward: {:?}",
            mesh.bounds()
        );
        assert!(
            reaches(1, before.max.y) && reaches(1, before.min.y),
            "the border along Y moved inward: {:?}",
            mesh.bounds()
        );
    }

    #[test]
    fn a_boundary_corner_rounds_off() {
        // The other half of the boundary rule, and the reason the test above
        // measures runs rather than the box: a border is a closed curve, so its
        // corners are smoothed along it. A rule that pinned every boundary vertex
        // would leave an open mesh permanently polygonal at its edges.
        let mesh = subdivide_once(&quad());
        let corner = mesh.positions()[0];
        assert!(
            corner.length() < quad().positions()[0].length() - 0.1,
            "the corner of the border did not round: {corner:?}"
        );
    }

    #[test]
    fn a_boundary_curve_is_smoothed_along_itself() {
        // A boundary follows a cubic B-spline through its neighbours: it keeps
        // its extent but rounds off. A rule that simply pinned every boundary
        // vertex would leave the border polygonal for ever.
        let mesh = MeshData::from_faces(
            vec![
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
            ],
            &[vec![0, 1, 2, 3]],
        );
        let out = subdivide_once(&mesh);
        // The peak was at y = 1 and should have been drawn in toward its
        // neighbours, which sit at y = 0.
        let peak = out.positions()[1];
        assert!(
            peak.y < 0.9 && peak.y > 0.5,
            "the boundary vertex did not follow the curve: {peak:?}"
        );
    }

    #[test]
    fn a_creased_edge_holds_its_line() {
        // A crease should behave exactly like a boundary: the surface follows the
        // creased polyline instead of being pulled off it.
        let mut mesh = primitives::cube(2.0);
        let edges = mesh.num_edges() as usize;
        // Crease the four edges around the cube's top face.
        let top: Vec<bool> = mesh
            .edges()
            .iter()
            .map(|[a, b]| {
                mesh.positions()[*a as usize].z > 0.9 && mesh.positions()[*b as usize].z > 0.9
            })
            .collect();
        assert_eq!(top.iter().filter(|x| **x).count(), 4, "four top edges");
        mesh.attributes_mut().insert(
            names::CREASE,
            Domain::Edge,
            AttrData::F32(Arc::new(
                (0..edges).map(|i| if top[i] { 1.0 } else { 0.0 }).collect(),
            )),
        );

        let creased = subdivide_once(&mesh);
        let plain = subdivide_once(&primitives::cube(2.0));
        // The creased ring holds its square outline, so the top of the cube stays
        // wider than it does without the crease.
        let width = |m: &MeshData| {
            m.positions()
                .iter()
                .filter(|p| p.z > 0.5)
                .map(|p| p.x.abs().max(p.y.abs()))
                .fold(0.0f32, f32::max)
        };
        assert!(
            width(&creased) > width(&plain) + 0.05,
            "the crease was ignored: {} against {}",
            width(&creased),
            width(&plain)
        );
    }

    #[test]
    fn a_crease_survives_the_level_it_is_applied_at() {
        // Two levels of a creased mesh must still be creased at the second, so the
        // layer has to reach the output. A crease that vanished after one level is
        // the one failure a user is certain to see.
        let mut mesh = primitives::cube(2.0);
        let edges = mesh.num_edges() as usize;
        mesh.attributes_mut().insert(
            names::CREASE,
            Domain::Edge,
            AttrData::F32(Arc::new(vec![1.0; edges])),
        );
        let out = subdivide_once(&subdivide_once(&mesh));
        let crease = out
            .attributes()
            .get_f32(names::CREASE)
            .expect("the layer survived");
        assert_eq!(crease.len(), out.num_edges() as usize);
        assert!(
            crease.iter().filter(|c| **c >= 1.0).count() >= 24,
            "the creases did not propagate: {} of {}",
            crease.iter().filter(|c| **c >= 1.0).count(),
            crease.len()
        );
    }

    #[test]
    fn creasing_every_edge_leaves_the_shape_alone() {
        // The limit of the crease rule: a fully creased cube has three creases at
        // every corner, so every corner is a corner and nothing moves. It is also
        // the check that the corner rule exists at all — without it a fully
        // creased cube still rounds off.
        let mut mesh = primitives::cube(2.0);
        let edges = mesh.num_edges() as usize;
        mesh.attributes_mut().insert(
            names::SHARP_EDGE,
            Domain::Edge,
            AttrData::Bool(Arc::new(vec![true; edges])),
        );
        let out = subdivide_once(&mesh);
        // Compared vertex by vertex, not by the bounding box: a cube's box is
        // unchanged by subdivision either way, because the six face points sit
        // at the face centres and still reach ±1. The box would have made the
        // corner rule look load-bearing when it was not being exercised at all.
        for (v, before) in mesh.positions().iter().enumerate() {
            assert_eq!(
                out.positions()[v],
                *before,
                "corner {v} of a fully creased cube moved"
            );
        }
    }

    #[test]
    fn corner_data_is_interpolated_within_its_face() {
        // UVs, and the reason they are corner data: the value at a new corner
        // comes from the corners of the same face, never from across a seam.
        let mut mesh = quad();
        let corners = mesh.num_corners() as usize;
        mesh.attributes_mut().insert(
            "uv",
            Domain::Corner,
            AttrData::F32x2(Arc::new(vec![
                kreate_math::Vec2::new(0.0, 0.0),
                kreate_math::Vec2::new(1.0, 0.0),
                kreate_math::Vec2::new(1.0, 1.0),
                kreate_math::Vec2::new(0.0, 1.0),
            ])),
        );
        assert_eq!(corners, 4);

        let out = subdivide_once(&mesh);
        let uv = out
            .attributes()
            .get_f32x2("uv")
            .expect("the layer survived");
        assert_eq!(uv.len(), out.num_corners() as usize);
        // Every quad has one corner at the face centre, which must be the mean of
        // the four source corners.
        let centre = kreate_math::Vec2::new(0.5, 0.5);
        let centres = uv.iter().filter(|p| (**p - centre).length() < 1e-5).count();
        assert_eq!(centres, 4, "one per output quad, got {centres}");
    }

    #[test]
    fn face_data_is_inherited_by_every_quad() {
        let mut mesh = primitives::cube(2.0);
        let count = mesh.num_faces() as usize;
        mesh.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(Arc::new((0..count as i32).collect())),
        );
        let out = subdivide_once(&mesh);
        let materials = out
            .attributes()
            .get_i32(names::MATERIAL_INDEX)
            .expect("the layer survived");
        assert_eq!(materials.len(), out.num_faces() as usize);
        for material in 0..count as i32 {
            assert_eq!(
                materials.iter().filter(|m| **m == material).count(),
                4,
                "each face's four quads keep its material"
            );
        }
    }

    #[test]
    fn a_selection_does_not_bleed_outward() {
        // Selection is a Bool layer, and blending it with "any source" would grow
        // the selection by a ring on every level — which looks like the
        // subdivision selecting things by itself.
        let mut mesh = primitives::cube(2.0);
        let verts = mesh.num_verts() as usize;
        mesh.attributes_mut().insert(
            names::SELECT_VERT,
            Domain::Point,
            AttrData::Bool(Arc::new((0..verts).map(|i| i == 0).collect())),
        );
        let out = subdivide_once(&mesh);
        let selected = out
            .attributes()
            .get_bool(names::SELECT_VERT)
            .expect("the layer survived");
        assert_eq!(
            selected.iter().filter(|s| **s).count(),
            1,
            "one selected vertex should stay one"
        );
    }

    #[test]
    fn an_empty_mesh_subdivides_to_nothing() {
        let out = subdivide_once(&MeshData::empty());
        assert!(out.is_empty());
    }

    #[test]
    fn a_triangle_becomes_three_quads() {
        // Not four: n-gons become n quads, and a triangle is an n-gon. Getting
        // this wrong is how a subdivision surface ends up with a triangle left in
        // it that never becomes a quad.
        let mesh = MeshData::from_faces(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            &[vec![0, 1, 2]],
        );
        let out = subdivide_once(&mesh);
        out.assert_valid();
        assert_eq!(out.num_faces(), 3);
        for face in out.faces() {
            assert_eq!(face.len(), 4);
        }
    }

    #[test]
    fn the_result_stays_valid_over_several_levels() {
        // The topology walk is where an off-by-one hides, and an invalid mesh
        // three levels in is a crash somewhere else entirely.
        let mut mesh = primitives::uv_sphere(1.0, 8, 4);
        for level in 0..3 {
            mesh = subdivide_once(&mesh);
            assert_eq!(mesh.validate(), Vec::new(), "level {level} produced junk");
        }
    }

    #[test]
    fn winding_is_preserved() {
        // The quads are wound (vertex, ahead, centre, behind). Reversed, every
        // normal on the model flips and the surface renders inside out.
        let cube = primitives::cube(2.0);
        let out = subdivide_once(&cube);
        let normals = out.compute_face_normals();
        for (index, normal) in normals.iter().enumerate() {
            let ring: Vec<Vec3> = out
                .face_verts(index as u32)
                .iter()
                .map(|v| out.positions()[*v as usize])
                .collect();
            let mid: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
            assert!(
                normal.normalize_or_zero().dot(mid.normalize_or_zero()) > 0.5,
                "face {index} at {mid:?} faces inward: {normal:?}"
            );
        }
    }

    #[test]
    fn a_partial_crease_lands_between_smooth_and_sharp_and_relaxes() {
        // The top ring of a cube at no crease, half a crease and a full one: the
        // half crease holds the outline wider than smooth and narrower than
        // sharp — and over levels it spends its sharpness and relaxes.
        let top_ring = |weight: f32| {
            let mut mesh = primitives::cube(2.0);
            let ring: Vec<f32> = mesh
                .edges()
                .iter()
                .map(|[a, b]| {
                    let up = mesh.positions()[*a as usize].z > 0.9 && mesh.positions()[*b as usize].z > 0.9;
                    if up { weight } else { 0.0 }
                })
                .collect();
            mesh.attributes_mut()
                .insert(names::CREASE, Domain::Edge, AttrData::F32(Arc::new(ring)));
            mesh
        };
        let width = |m: &MeshData| {
            m.positions()
                .iter()
                .filter(|p| p.z > 0.5)
                .map(|p| p.x.abs().max(p.y.abs()))
                .fold(0.0f32, f32::max)
        };
        let once = |w: f32| width(&subdivide_once(&top_ring(w)));
        let (smooth, half, sharp) = (once(0.0), once(0.1), once(1.0));
        assert!(smooth < half && half < sharp, "{smooth} < {half} < {sharp}");

        // 0.1 is less than one level's worth: it is gone after the first.
        let after = subdivide_once(&top_ring(0.1));
        let crease = after.attributes().get_f32(names::CREASE).unwrap();
        assert!(crease.iter().all(|c| *c == 0.0), "the sharpness was not spent");
        let full = subdivide_once(&top_ring(1.0));
        assert!(full.attributes().get_f32(names::CREASE).unwrap().iter().any(|c| *c == 1.0));
    }
}
