//! The boolean modifier: one mesh cut by, joined with, or kept inside another.
//!
//! By binary space partitioning, the method of Thibault and Naylor that
//! `csg.js` made familiar: each solid's polygons build a tree of the planes
//! they lie in, and each solid is clipped against the other's tree — what is
//! inside the other kept or thrown away depending on the operation. It is
//! exact up to floating point, needs no special cases for coplanar faces, and
//! is short.
//!
//! # What it asks of its input
//!
//! Both meshes should be closed. A hole lets "inside" leak, and the result is
//! wrong in the way every boolean is wrong on an open mesh — Blender's too.
//!
//! # What it gives back
//!
//! A closed mesh, with each input face kept whole wherever the cut did not
//! reach it. The BSP alone gives neither: it splits a face along every plane of
//! the other solid's tree, cut or not — a cube's side came back as a row of
//! strips, one per facet of a cylinder that never touched it — and the pieces
//! of two neighbouring faces split at different places meet in T-junctions,
//! so the mesh had open edges everywhere and a bevel or a subdivision put on
//! top of it tore. [`assemble`] puts that right: pieces are welded, every
//! vertex lying on another piece's edge is put into that edge, and the pieces
//! of one input face are merged back into as few polygons as stay simple.
//! Attributes (UVs, materials) are not carried; the result is geometry.
//!
//! Everything is in `f64` — a plane test in `f32` misclassifies points near a
//! face, and a boolean is nothing but plane tests.

use kreate_math::{DVec3, Vec3};

use crate::MeshData;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BooleanOp {
    /// What is in the first and not in the second — cutting a hole.
    #[default]
    Difference,
    /// What is in either.
    Union,
    /// What is in both.
    Intersect,
}

const EPSILON: f64 = 1e-6;

#[derive(Clone, Copy, Debug)]
struct Plane {
    normal: DVec3,
    w: f64,
}

impl Plane {
    fn from_points(a: DVec3, b: DVec3, c: DVec3) -> Option<Plane> {
        let n = (b - a).cross(c - a);
        let len = n.length();
        if len < 1e-12 {
            return None;
        }
        let normal = n / len;
        Some(Plane {
            normal,
            w: normal.dot(a),
        })
    }

    fn flip(&mut self) {
        self.normal = -self.normal;
        self.w = -self.w;
    }
}

#[derive(Clone, Debug)]
struct Polygon {
    verts: Vec<DVec3>,
    plane: Plane,
    /// Which input face this is, or a piece of — A's faces first, then B's.
    /// What lets the pieces be merged back afterwards.
    source: u32,
}

impl Polygon {
    fn new(verts: Vec<DVec3>, source: u32) -> Option<Polygon> {
        // The plane of the whole polygon, from its Newell normal, so a thin
        // first triangle does not decide it.
        let mut n = DVec3::ZERO;
        for i in 0..verts.len() {
            let (p, q) = (verts[i], verts[(i + 1) % verts.len()]);
            n += DVec3::new(
                (p.y - q.y) * (p.z + q.z),
                (p.z - q.z) * (p.x + q.x),
                (p.x - q.x) * (p.y + q.y),
            );
        }
        let len = n.length();
        if verts.len() < 3 {
            return None;
        }
        if len < 1e-12 {
            return Plane::from_points(verts[0], verts[1], verts[2]).map(|plane| Polygon {
                verts,
                plane,
                source,
            });
        }
        let normal = n / len;
        let centre = verts.iter().fold(DVec3::ZERO, |s, p| s + *p) / verts.len() as f64;
        Some(Polygon {
            plane: Plane {
                normal,
                w: normal.dot(centre),
            },
            verts,
            source,
        })
    }

    fn flip(&mut self) {
        self.verts.reverse();
        self.plane.flip();
    }
}

const COPLANAR: u8 = 0;
const FRONT: u8 = 1;
const BACK: u8 = 2;
const SPANNING: u8 = 3;

/// Sorts `polygon` against `plane` into the four lists.
fn split(
    plane: &Plane,
    polygon: &Polygon,
    coplanar_front: &mut Vec<Polygon>,
    coplanar_back: &mut Vec<Polygon>,
    front: &mut Vec<Polygon>,
    back: &mut Vec<Polygon>,
) {
    let mut kind = 0u8;
    let types: Vec<u8> = polygon
        .verts
        .iter()
        .map(|v| {
            let t = plane.normal.dot(*v) - plane.w;
            let k = if t < -EPSILON {
                BACK
            } else if t > EPSILON {
                FRONT
            } else {
                COPLANAR
            };
            kind |= k;
            k
        })
        .collect();
    match kind {
        COPLANAR => {
            if plane.normal.dot(polygon.plane.normal) > 0.0 {
                coplanar_front.push(polygon.clone());
            } else {
                coplanar_back.push(polygon.clone());
            }
        }
        FRONT => front.push(polygon.clone()),
        BACK => back.push(polygon.clone()),
        _ => {
            let (mut f, mut b) = (Vec::new(), Vec::new());
            let n = polygon.verts.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let (ti, tj) = (types[i], types[j]);
                let (vi, vj) = (polygon.verts[i], polygon.verts[j]);
                if ti != BACK {
                    f.push(vi);
                }
                if ti != FRONT {
                    b.push(vi);
                }
                if (ti | tj) == SPANNING {
                    let t = (plane.w - plane.normal.dot(vi)) / plane.normal.dot(vj - vi);
                    let v = vi.lerp(vj, t);
                    f.push(v);
                    b.push(v);
                }
            }
            if f.len() >= 3 {
                front.push(Polygon {
                    verts: f,
                    plane: polygon.plane,
                    source: polygon.source,
                });
            }
            if b.len() >= 3 {
                back.push(Polygon {
                    verts: b,
                    plane: polygon.plane,
                    source: polygon.source,
                });
            }
        }
    }
}

/// A BSP tree, its nodes in an arena so that nothing about it recurses — a
/// badly balanced tree over a few thousand polygons is thousands deep.
#[derive(Default)]
struct Tree {
    nodes: Vec<Node>,
}

#[derive(Default)]
struct Node {
    plane: Option<Plane>,
    front: Option<usize>,
    back: Option<usize>,
    polygons: Vec<Polygon>,
}

impl Tree {
    fn new(polygons: Vec<Polygon>) -> Tree {
        let mut tree = Tree {
            nodes: vec![Node::default()],
        };
        tree.build(0, polygons);
        tree
    }

    fn build(&mut self, root: usize, polygons: Vec<Polygon>) {
        let mut work = vec![(root, polygons)];
        while let Some((at, polygons)) = work.pop() {
            if polygons.is_empty() {
                continue;
            }
            let plane = *self.nodes[at].plane.get_or_insert(polygons[0].plane);
            let (mut cf, mut cb, mut f, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for p in &polygons {
                split(&plane, p, &mut cf, &mut cb, &mut f, &mut b);
            }
            self.nodes[at].polygons.extend(cf);
            self.nodes[at].polygons.extend(cb);
            if !f.is_empty() {
                let child = self.child(at, true);
                work.push((child, f));
            }
            if !b.is_empty() {
                let child = self.child(at, false);
                work.push((child, b));
            }
        }
    }

    fn child(&mut self, at: usize, front: bool) -> usize {
        let existing = if front {
            self.nodes[at].front
        } else {
            self.nodes[at].back
        };
        if let Some(c) = existing {
            return c;
        }
        self.nodes.push(Node::default());
        let c = self.nodes.len() - 1;
        if front {
            self.nodes[at].front = Some(c);
        } else {
            self.nodes[at].back = Some(c);
        }
        c
    }

    /// Turns the solid inside out: every plane and polygon flipped, and front
    /// and back swapped.
    fn invert(&mut self) {
        for node in &mut self.nodes {
            for p in &mut node.polygons {
                p.flip();
            }
            if let Some(plane) = &mut node.plane {
                plane.flip();
            }
            std::mem::swap(&mut node.front, &mut node.back);
        }
    }

    /// What of `polygons` is outside this solid.
    fn clip_polygons(&self, polygons: Vec<Polygon>) -> Vec<Polygon> {
        let mut out = Vec::new();
        let mut work = vec![(0usize, polygons)];
        while let Some((at, polygons)) = work.pop() {
            let node = &self.nodes[at];
            let Some(plane) = node.plane else {
                out.extend(polygons);
                continue;
            };
            let (mut cf, mut cb, mut f, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for p in &polygons {
                split(&plane, p, &mut cf, &mut cb, &mut f, &mut b);
            }
            f.extend(cf);
            b.extend(cb);
            match node.front {
                Some(c) => work.push((c, f)),
                None => out.extend(f),
            }
            // Behind a leaf is inside: dropped.
            if let Some(c) = node.back {
                work.push((c, b));
            }
        }
        out
    }

    /// Removes from this tree's polygons everything inside `other`.
    fn clip_to(&mut self, other: &Tree) {
        for i in 0..self.nodes.len() {
            let polygons = std::mem::take(&mut self.nodes[i].polygons);
            self.nodes[i].polygons = other.clip_polygons(polygons);
        }
    }

    fn all_polygons(&self) -> Vec<Polygon> {
        self.nodes
            .iter()
            .flat_map(|n| n.polygons.iter().cloned())
            .collect()
    }
}

/// A mesh's faces as polygons, their sources numbered from `first`.
fn polygons_of(mesh: &MeshData, first: u32) -> Vec<Polygon> {
    let positions = mesh.positions();
    (0..mesh.num_faces())
        .filter_map(|f| {
            let verts: Vec<DVec3> = mesh
                .face_verts(f)
                .iter()
                .map(|v| positions[*v as usize].as_dvec3())
                .collect();
            Polygon::new(verts, first + f)
        })
        .collect()
}

/// `a` combined with `b`, both in the same space.
pub fn boolean(a: &MeshData, b: &MeshData, op: BooleanOp) -> MeshData {
    if a.is_empty() {
        return match op {
            BooleanOp::Union => b.clone(),
            _ => a.clone(),
        };
    }
    if b.is_empty() {
        return match op {
            BooleanOp::Intersect => MeshData::from_faces(Vec::new(), &[]),
            _ => a.clone(),
        };
    }
    let mut ta = Tree::new(polygons_of(a, 0));
    let mut tb = Tree::new(polygons_of(b, a.num_faces()));
    let polygons = match op {
        BooleanOp::Union => {
            ta.clip_to(&tb);
            tb.clip_to(&ta);
            tb.invert();
            tb.clip_to(&ta);
            tb.invert();
            let mut all = ta.all_polygons();
            all.extend(tb.all_polygons());
            all
        }
        BooleanOp::Difference => {
            ta.invert();
            ta.clip_to(&tb);
            tb.clip_to(&ta);
            tb.invert();
            tb.clip_to(&ta);
            tb.invert();
            let mut all = ta.all_polygons();
            all.extend(tb.all_polygons());
            let mut result = Tree::new(all);
            result.invert();
            result.all_polygons()
        }
        BooleanOp::Intersect => {
            ta.invert();
            tb.clip_to(&ta);
            tb.invert();
            ta.clip_to(&tb);
            tb.clip_to(&ta);
            let mut all = ta.all_polygons();
            all.extend(tb.all_polygons());
            let mut result = Tree::new(all);
            result.invert();
            result.all_polygons()
        }
    };
    assemble(polygons)
}

/// How close two points are to be the same point, and a point to lie on an
/// edge. Absolute, in the mesh's units: a boolean's numbers are plane tests
/// accurate to [`EPSILON`], and this is ten times looser than that.
const WELD: f64 = 1e-5;

/// The BSP's polygons to a closed mesh: see the module docs.
fn assemble(polygons: Vec<Polygon>) -> MeshData {
    let mut welder = Welder::default();
    let mut pieces: Vec<(u32, Vec<u32>)> = Vec::with_capacity(polygons.len());
    for p in &polygons {
        let mut ring: Vec<u32> = Vec::with_capacity(p.verts.len());
        for v in &p.verts {
            let i = welder.index(*v);
            if ring.last() != Some(&i) {
                ring.push(i);
            }
        }
        while ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() >= 3 {
            pieces.push((p.source, ring));
        }
    }
    let positions = welder.positions;

    split_t_junctions(&positions, &mut pieces);

    // The pieces of each input face, merged back together.
    pieces.sort_by_key(|(source, _)| *source);
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut at = 0;
    while at < pieces.len() {
        let source = pieces[at].0;
        let end = at
            + pieces[at..]
                .iter()
                .take_while(|(s, _)| *s == source)
                .count();
        let group: Vec<Vec<u32>> = pieces[at..end].iter().map(|(_, r)| r.clone()).collect();
        faces.extend(merge_pieces(group));
        at = end;
    }

    dissolve_straight_vertices(&positions, &mut faces);

    // Only the vertices still in use, renumbered in the order faces meet them.
    let mut renumber = vec![u32::MAX; positions.len()];
    let mut kept: Vec<Vec3> = Vec::new();
    for face in &mut faces {
        for v in face.iter_mut() {
            if renumber[*v as usize] == u32::MAX {
                renumber[*v as usize] = kept.len() as u32;
                kept.push(positions[*v as usize].as_vec3());
            }
            *v = renumber[*v as usize];
        }
    }
    MeshData::from_faces(kept, &faces)
}

/// Points to indices, a point within [`WELD`] of one already seen taking its
/// index. Searches the neighbouring cells too: rounding alone puts two points
/// a hair apart on either side of a cell boundary into different vertices.
#[derive(Default)]
struct Welder {
    positions: Vec<DVec3>,
    cells: std::collections::HashMap<[i64; 3], Vec<u32>>,
}

impl Welder {
    fn cell(p: DVec3) -> [i64; 3] {
        [
            (p.x / WELD).floor() as i64,
            (p.y / WELD).floor() as i64,
            (p.z / WELD).floor() as i64,
        ]
    }

    fn index(&mut self, p: DVec3) -> u32 {
        let c = Self::cell(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = self.cells.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                        if let Some(&i) = list
                            .iter()
                            .find(|i| self.positions[**i as usize].distance(p) <= WELD)
                        {
                            return i;
                        }
                    }
                }
            }
        }
        let i = self.positions.len() as u32;
        self.positions.push(p);
        self.cells.entry(c).or_default().push(i);
        i
    }
}

/// Puts every vertex that lies on a piece's edge into that edge.
///
/// A face split at a point its neighbour across the edge was not split at is
/// a T-junction: the mesh looks closed and is not, and the pieces of one face
/// cannot be merged along an edge they do not share exactly. Candidates come
/// from whichever axis the edge spans least, so a short or axis-aligned edge
/// looks at a handful of vertices rather than all of them.
fn split_t_junctions(positions: &[DVec3], pieces: &mut [(u32, Vec<u32>)]) {
    let sorted: [Vec<u32>; 3] = std::array::from_fn(|axis| {
        let mut order: Vec<u32> = (0..positions.len() as u32).collect();
        order.sort_by(|a, b| positions[*a as usize][axis].total_cmp(&positions[*b as usize][axis]));
        order
    });
    let mut on_edge: Vec<(f64, u32)> = Vec::new();
    for (_, ring) in pieces.iter_mut() {
        let n = ring.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let (a, b) = (ring[k], ring[(k + 1) % n]);
            out.push(a);
            let (pa, pb) = (positions[a as usize], positions[b as usize]);
            let d = pb - pa;
            let len2 = d.length_squared();
            if len2 <= WELD * WELD {
                continue;
            }
            let extent = d.abs();
            let axis = if extent.x <= extent.y && extent.x <= extent.z {
                0
            } else if extent.y <= extent.z {
                1
            } else {
                2
            };
            let (lo, hi) = (pa[axis].min(pb[axis]) - WELD, pa[axis].max(pb[axis]) + WELD);
            let list = &sorted[axis];
            let from = list.partition_point(|i| positions[*i as usize][axis] < lo);
            on_edge.clear();
            for &v in list[from..]
                .iter()
                .take_while(|i| positions[**i as usize][axis] <= hi)
            {
                if v == a || v == b {
                    continue;
                }
                let p = positions[v as usize];
                let t = (p - pa).dot(d) / len2;
                if t <= 0.0 || t >= 1.0 {
                    continue;
                }
                if (pa + d * t).distance(p) <= WELD {
                    on_edge.push((t, v));
                }
            }
            on_edge.sort_by(|x, y| x.0.total_cmp(&y.0));
            out.extend(on_edge.iter().map(|(_, v)| *v));
        }
        *ring = out;
    }
}

/// The pieces of one face, merged into as few polygons as stay simple.
///
/// Two pieces merge when they share one unbroken run of edges, which is
/// exactly when their union has no hole in it: a face with a hole through it
/// cannot be one polygon, and stays as however many it takes. Untouched by
/// the cut, the pieces come back together as the face they were.
fn merge_pieces(mut rings: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    let mut i = 0;
    while i < rings.len() {
        let mut j = i + 1;
        while j < rings.len() {
            if let Some(union) = merge_two(&rings[i], &rings[j]) {
                rings[i] = union;
                rings.swap_remove(j);
                // Grown, it may now meet pieces it did not before.
                j = i + 1;
            } else {
                j += 1;
            }
        }
        i += 1;
    }
    rings
}

/// `p` and `q` as one polygon, if they share one unbroken run of edges and
/// the result is simple.
fn merge_two(p: &[u32], q: &[u32]) -> Option<Vec<u32>> {
    let (n, m) = (p.len(), q.len());
    // Shared: p's edge a→b is q's edge b→a.
    let q_edges: std::collections::HashSet<(u32, u32)> =
        (0..m).map(|k| (q[k], q[(k + 1) % m])).collect();
    let shared: Vec<bool> = (0..n)
        .map(|k| q_edges.contains(&(p[(k + 1) % n], p[k])))
        .collect();
    let count = shared.iter().filter(|s| **s).count();
    if count == 0 || count == n {
        return None;
    }
    // One run: exactly one place where an unshared edge is followed by a shared one.
    let starts: Vec<usize> = (0..n)
        .filter(|k| shared[*k] && !shared[(k + n - 1) % n])
        .collect();
    if starts.len() != 1 {
        return None;
    }
    let s = starts[0];
    // p[s] .. p[s + count] is the shared chain; q runs it backwards, ending at p[s].
    let first = p[s];
    let last = p[(s + count) % n];
    let at = q.iter().position(|v| *v == first)?;
    let mut union = Vec::with_capacity(n + m - 2 * count);
    // p from the chain's end round to its start...
    for k in 0..=(n - count) {
        union.push(p[(s + count + k) % n]);
    }
    // ...then q from just after the chain's start round to just before its end.
    for k in 1..m {
        let v = q[(at + k) % m];
        if v == last {
            break;
        }
        union.push(v);
    }
    if union.len() != n + m - 2 * count {
        return None;
    }
    // A vertex twice is a pinch: the two touched at a point as well.
    let mut seen = union.clone();
    seen.sort_unstable();
    seen.dedup();
    (seen.len() == union.len()).then_some(union)
}

/// Takes out every vertex that only passes along a straight edge.
///
/// What the T-junction repair put in where a neighbour was split and the
/// merge then made unnecessary. A vertex with exactly two neighbours, in line
/// with them, is on one edge between two faces, so taking it out of both
/// keeps the mesh closed.
fn dissolve_straight_vertices(positions: &[DVec3], faces: &mut [Vec<u32>]) {
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
    for face in faces.iter() {
        let n = face.len();
        for k in 0..n {
            let (a, b) = (face[k], face[(k + 1) % n]);
            for (x, y) in [(a, b), (b, a)] {
                if !neighbours[x as usize].contains(&y) {
                    neighbours[x as usize].push(y);
                }
            }
        }
    }
    let mut straight: Vec<bool> = (0..positions.len())
        .map(|v| {
            let [a, b] = neighbours[v][..] else {
                return false;
            };
            let (pa, pb, p) = (positions[a as usize], positions[b as usize], positions[v]);
            let d = pb - pa;
            let t = (p - pa).dot(d) / d.length_squared().max(f64::MIN_POSITIVE);
            t > 0.0 && t < 1.0 && (pa + d * t).distance(p) <= WELD
        })
        .collect();
    // Out of every face or none: a face left too few corners keeps its own,
    // and the face across the edge has to keep them too.
    for face in faces.iter() {
        if face.iter().filter(|v| !straight[**v as usize]).count() < 3 {
            for v in face {
                straight[*v as usize] = false;
            }
        }
    }
    for face in faces.iter_mut() {
        face.retain(|v| !straight[*v as usize]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn moved(mut m: MeshData, by: Vec3) -> MeshData {
        for p in m.positions_mut() {
            *p += by;
        }
        m
    }

    /// The volume a closed mesh encloses, by the divergence theorem.
    fn volume(m: &MeshData) -> f64 {
        let p = m.positions();
        let mut v = 0.0;
        for f in 0..m.num_faces() {
            let ring = m.face_verts(f);
            for i in 1..ring.len() - 1 {
                let (a, b, c) = (
                    p[ring[0] as usize].as_dvec3(),
                    p[ring[i] as usize].as_dvec3(),
                    p[ring[i + 1] as usize].as_dvec3(),
                );
                v += a.dot(b.cross(c)) / 6.0;
            }
        }
        v
    }

    /// Edges not matched by the same edge the other way round: holes and
    /// T-junctions. Zero for a closed mesh.
    fn open_edges(m: &MeshData) -> usize {
        let mut e: std::collections::HashMap<(u32, u32), i32> = std::collections::HashMap::new();
        for f in 0..m.num_faces() {
            let r = m.face_verts(f);
            for i in 0..r.len() {
                let (a, b) = (r[i], r[(i + 1) % r.len()]);
                *e.entry((a, b)).or_default() += 1;
                *e.entry((b, a)).or_default() -= 1;
            }
        }
        e.values().filter(|v| **v != 0).count() / 2
    }

    /// The volume of what the renderer would draw: through the real
    /// triangulation, so a merged non-convex face that ear clipping got wrong
    /// shows up here even though its polygon is right.
    fn drawn_volume(m: &MeshData) -> f64 {
        let p = m.positions();
        m.triangulate()
            .indices
            .chunks(3)
            .map(|t| {
                let (a, b, c) = (
                    p[t[0] as usize].as_dvec3(),
                    p[t[1] as usize].as_dvec3(),
                    p[t[2] as usize].as_dvec3(),
                );
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    fn rotated_to_x(mut m: MeshData) -> MeshData {
        for p in m.positions_mut() {
            *p = Vec3::new(p.z, p.y, -p.x);
        }
        m
    }

    #[test]
    fn the_result_is_closed_and_draws_the_volume_it_encloses() {
        let cube = primitives::cube(2.0);
        let rod = primitives::cylinder(0.5, 4.0, 32);
        let cases = [
            ("hole", boolean(&cube, &rod, BooleanOp::Difference)),
            (
                "sphere out",
                boolean(
                    &cube,
                    &primitives::uv_sphere(1.2, 32, 16),
                    BooleanOp::Difference,
                ),
            ),
            (
                "sphere in",
                boolean(
                    &cube,
                    &primitives::uv_sphere(1.2, 32, 16),
                    BooleanOp::Intersect,
                ),
            ),
            (
                "pocket",
                boolean(
                    &cube,
                    &moved(primitives::cube(1.0), Vec3::new(0.0, 0.0, 0.5)),
                    BooleanOp::Difference,
                ),
            ),
            (
                "stacked",
                boolean(
                    &cube,
                    &moved(primitives::cube(1.0), Vec3::new(0.0, 0.0, 1.5)),
                    BooleanOp::Union,
                ),
            ),
            (
                "torus",
                boolean(
                    &cube,
                    &primitives::torus(1.0, 0.3, 24, 12),
                    BooleanOp::Difference,
                ),
            ),
        ];
        for (name, m) in &cases {
            assert_eq!(open_edges(m), 0, "{name}: open edges");
            assert!(
                (drawn_volume(m) - volume(m)).abs() < 1e-3,
                "{name}: drawn {} encloses {}",
                drawn_volume(m),
                volume(m)
            );
        }
        // A second hole across the first, on the first's result: what a stack
        // of two booleans does.
        let twice = boolean(
            &cases[0].1,
            &rotated_to_x(rod.clone()),
            BooleanOp::Difference,
        );
        assert_eq!(open_edges(&twice), 0, "second hole: open edges");
        assert!((drawn_volume(&twice) - volume(&twice)).abs() < 1e-3);
    }

    #[test]
    fn faces_the_cut_does_not_reach_stay_whole() {
        // A rod through the top and bottom: the four sides are untouched and
        // come back as the four quads they were, not as strips cut along the
        // rod's facets.
        let out = boolean(
            &primitives::cube(2.0),
            &primitives::cylinder(0.5, 4.0, 32),
            BooleanOp::Difference,
        );
        let p = out.positions();
        let sides = (0..out.num_faces())
            .filter(|f| {
                let ring = out.face_verts(*f);
                let n = crate::tessellate::newell_normal(
                    &ring.iter().map(|v| p[*v as usize]).collect::<Vec<_>>(),
                );
                n.z.abs() < 0.5
                    && ring
                        .iter()
                        .all(|v| p[*v as usize].x.abs() > 0.999 || p[*v as usize].y.abs() > 0.999)
            })
            .count();
        assert_eq!(sides, 4, "the sides came back in pieces");
        // The top and bottom each have a hole, so each is a few polygons
        // rather than one — but a few, not the fan of 32 slivers it was.
        assert!(
            out.num_faces() < 6 + 32 + 2 * 12,
            "{} faces",
            out.num_faces()
        );
    }

    #[test]
    fn two_unit_overlapping_cubes_give_the_volumes_arithmetic_says() {
        // Two 2-unit cubes, the second moved one unit along X: they overlap in
        // a 1 x 2 x 2 box.
        let a = primitives::cube(2.0);
        let b = moved(primitives::cube(2.0), Vec3::X);
        let near = |got: f64, want: f64| (got - want).abs() < 1e-3;
        let difference = boolean(&a, &b, BooleanOp::Difference);
        assert!(
            near(volume(&difference), 4.0),
            "difference {}",
            volume(&difference)
        );
        let union = boolean(&a, &b, BooleanOp::Union);
        assert!(near(volume(&union), 12.0), "union {}", volume(&union));
        let intersect = boolean(&a, &b, BooleanOp::Intersect);
        assert!(
            near(volume(&intersect), 4.0),
            "intersect {}",
            volume(&intersect)
        );
        assert!(intersect.bounds().min.x > -0.01 && intersect.bounds().max.x < 1.01);
    }

    #[test]
    fn a_cutter_that_misses_leaves_the_mesh_as_it_was() {
        let a = primitives::cube(2.0);
        let b = moved(primitives::cube(1.0), Vec3::new(10.0, 0.0, 0.0));
        let out = boolean(&a, &b, BooleanOp::Difference);
        assert!((volume(&out) - 8.0).abs() < 1e-4);
        assert!((volume(&boolean(&a, &b, BooleanOp::Intersect))).abs() < 1e-6);
    }

    #[test]
    fn a_hole_through_a_cube_takes_out_what_it_should() {
        // A 0.5-wide square rod right through a 2-unit cube: 8 - 0.25 * 2.
        let a = primitives::cube(2.0);
        let mut rod = primitives::cube(1.0);
        for p in rod.positions_mut() {
            p.x *= 0.5;
            p.y *= 0.5;
            p.z *= 4.0;
        }
        let out = boolean(&a, &rod, BooleanOp::Difference);
        assert!((volume(&out) - 7.5).abs() < 1e-3, "{}", volume(&out));
    }
}
