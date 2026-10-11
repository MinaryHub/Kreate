//! Quad remesh: the surface rebuilt as squares that follow its shape —
//! Blender's QuadriFlow, by the method of Instant Meshes (Jakob, Tarini,
//! Panozzo and Sorkine-Hornung, 2015).
//!
//! A voxel remesh makes even squares, but they line up with the world's axes
//! and stair-step across anything turned. A generated or sculpted mesh needs
//! squares that run along it: rings round a limb, lines along a crease.
//!
//! # How
//!
//! 1. **Fine enough**: edges longer than half a square are split, so every
//!    square has several vertices under it to decide where it goes.
//! 2. **Which way**: at every vertex a direction in its tangent plane, the
//!    same as its neighbours' up to a quarter turn — a cross field — smoothed
//!    from coarse to fine over a hierarchy of merged vertices. Open edges and
//!    sharp creases hold their own direction, so squares run along them.
//! 3. **Where**: at every vertex a grid of squares in that direction, offset
//!    so that it agrees with its neighbours' — smoothed the same way.
//! 4. **Squares**: vertices whose nearest grid corner is the same corner
//!    become one vertex; neighbours one grid step apart become an edge; and
//!    the faces are the loops of those edges round each vertex.
//! 5. **Even out**: the new vertices are relaxed along the surface and put
//!    back onto it.
//!
//! Where the field turns round a point — the poles of a sphere, a cube's
//! corners — a face can be a triangle or a five-sided one. Everything else is
//! squares.

use std::collections::HashMap;

use kreate_math::{Aabb, Bvh, Vec3};

use crate::remesh::closest_on_triangle;
use crate::MeshData;

/// The most triangles the surface is split into before the field is solved.
/// Squares finer than this would take that many — refused rather than tried.
pub const MAX_TRIANGLES: usize = 4_000_000;

/// A crease sharper than this, in degrees between the faces' normals, is one
/// the squares run along.
const SHARP_DEGREES: f32 = 50.0;

/// Why a quad remesh could not be done.
#[derive(Clone, Debug, PartialEq)]
pub enum QuadRemeshError {
    /// There is no surface to rebuild.
    Empty,
    /// The squares are too small for the surface: it would be split into
    /// this many triangles.
    TooFine(usize),
    /// The field gave no faces — the squares are bigger than the shape.
    NoFaces,
}

impl std::fmt::Display for QuadRemeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuadRemeshError::Empty => write!(f, "the mesh has no faces to remesh"),
            QuadRemeshError::TooFine(n) => write!(
                f,
                "squares that small need {n} triangles to place, more than {MAX_TRIANGLES}: \
                 ask for fewer faces"
            ),
            QuadRemeshError::NoFaces => write!(
                f,
                "squares that size make no faces on this shape: ask for more faces"
            ),
        }
    }
}

impl std::error::Error for QuadRemeshError {}

/// The square size that gives about `faces` squares over `mesh`'s surface.
pub fn edge_for_faces(mesh: &MeshData, faces: usize) -> f32 {
    let p = mesh.positions();
    let mut area = 0.0f32;
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        for k in 1..ring.len().saturating_sub(1) {
            let (a, b, c) = (
                p[ring[0] as usize],
                p[ring[k] as usize],
                p[ring[k + 1] as usize],
            );
            area += 0.5 * (b - a).cross(c - a).length();
        }
    }
    (area / faces.max(1) as f32).sqrt()
}

/// The surface of `mesh` rebuilt as squares about `edge` across, running
/// along its shape.
pub fn quad_remesh(mesh: &MeshData, edge: f32) -> Result<MeshData, QuadRemeshError> {
    let tri = mesh.triangulate();
    let (tris, _) = tri.indices.as_chunks::<3>();
    let p = mesh.positions();
    let tris: Vec<[u32; 3]> = tris
        .iter()
        .copied()
        .filter(|t| {
            let [a, b, c] = t.map(|i| p[i as usize]);
            [a, b, c].iter().all(|v| v.is_finite()) && (b - a).cross(c - a).length_squared() > 0.0
        })
        .collect();
    if tris.is_empty() || edge.is_nan() || edge <= 0.0 {
        return Err(QuadRemeshError::Empty);
    }
    let source_positions = p.to_vec();
    let source_closed = closed(&tris);
    // The field is laid out at twice the size asked for, and every face
    // it makes is split into squares at the end: where the grid cannot be
    // one grid all over a curved surface, a face can come out with three
    // sides or five, and split, those are squares too.
    let grid = 2.0 * edge;
    // 1. Fine enough.
    let (positions, tris) = refine(p.to_vec(), tris, 0.5 * grid)?;
    let finest = Level::from_triangles(&positions, &tris);
    // 2 and 3. The fields, coarse to fine.
    let mut levels = vec![finest];
    while levels.len() < 24 {
        let last = levels.last().expect("one level");
        if last.len() <= 32 {
            break;
        }
        let (coarse, parent) = last.coarsen();
        if coarse.len() as f32 > 0.9 * last.len() as f32 {
            break;
        }
        levels.last_mut().expect("one level").parent = parent;
        levels.push(coarse);
    }
    let (q, o) = solve(&levels, grid);
    // 4. Faces, the holes between them closed where the shape was closed,
    // each split into squares.
    let (points, mut faces) = extract(&levels[0], &q, &o, grid);
    if faces.is_empty() {
        return Err(QuadRemeshError::NoFaces);
    }
    if source_closed {
        fill_holes(&mut faces);
    }
    let (points, faces) = subdivide(points, &faces);
    // 5. Even out, onto the surface as it was.
    let source_tris: Vec<[u32; 3]> = {
        let (t, _) = tri.indices.as_chunks::<3>();
        t.to_vec()
    };
    let points = relax(points, &faces, &source_positions, &source_tris, edge);
    Ok(MeshData::from_faces(points, &faces))
}

/// Whether every edge of `tris` is between two of them.
fn closed(tris: &[[u32; 3]]) -> bool {
    let mut count: HashMap<(u32, u32), u32> = HashMap::new();
    for t in tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    count.values().all(|c| *c == 2)
}

/// Each hole of a dozen sides or fewer — where the field could not decide
/// on faces — closed with one face.
fn fill_holes(faces: &mut Vec<Vec<u32>>) {
    let mut sides: HashMap<(u32, u32), u32> = HashMap::new();
    for f in faces.iter() {
        for k in 0..f.len() {
            *sides.entry((f[k], f[(k + 1) % f.len()])).or_insert(0) += 1;
        }
    }
    // Round a hole the other way from the faces beside it.
    let mut after: HashMap<u32, Option<u32>> = HashMap::new();
    for &(a, b) in sides.keys() {
        if !sides.contains_key(&(b, a)) {
            after
                .entry(b)
                .and_modify(|next| *next = None) // more than one way on
                .or_insert(Some(a));
        }
    }
    let mut starts: Vec<u32> = after.keys().copied().collect();
    starts.sort_unstable();
    let mut done = std::collections::HashSet::new();
    for start in starts {
        if done.contains(&start) {
            continue;
        }
        let mut ring = vec![start];
        let mut at = start;
        let whole = loop {
            let Some(Some(next)) = after.get(&at).copied() else {
                break false;
            };
            if next == start {
                break true;
            }
            if ring.len() > 12 || ring.contains(&next) {
                break false;
            }
            ring.push(next);
            at = next;
        };
        for v in &ring {
            done.insert(*v);
        }
        if whole && ring.len() >= 3 {
            faces.push(ring);
        }
    }
}

/// Every face split into squares, one for each corner: from the corner to
/// the middles of its two sides and the face's middle.
fn subdivide(points: Vec<Vec3>, faces: &[Vec<u32>]) -> (Vec<Vec3>, Vec<Vec<u32>>) {
    let mut out = points;
    let mut middle: HashMap<(u32, u32), u32> = HashMap::new();
    let mut split = Vec::with_capacity(faces.len() * 4);
    for f in faces {
        let centre = out.len() as u32;
        out.push(f.iter().map(|&v| out[v as usize]).sum::<Vec3>() / f.len() as f32);
        let mids: Vec<u32> = (0..f.len())
            .map(|k| {
                let (a, b) = (f[k], f[(k + 1) % f.len()]);
                *middle.entry((a.min(b), a.max(b))).or_insert_with(|| {
                    out.push(0.5 * (out[a as usize] + out[b as usize]));
                    out.len() as u32 - 1
                })
            })
            .collect();
        for k in 0..f.len() {
            let before = mids[(k + f.len() - 1) % f.len()];
            split.push(vec![f[k], mids[k], centre, before]);
        }
    }
    (out, split)
}

// --- 1. fine enough ------------------------------------------------------------

/// Every edge longer than `fine` split at its middle, as often as it takes.
fn refine(
    mut positions: Vec<Vec3>,
    mut tris: Vec<[u32; 3]>,
    fine: f32,
) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), QuadRemeshError> {
    // What the splitting will come to, said before it is done.
    let area: f32 = tris
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            0.5 * (b - a).cross(c - a).length()
        })
        .sum();
    let estimate = (area / (0.25 * fine * fine)) as usize;
    if estimate.max(tris.len()) > MAX_TRIANGLES {
        return Err(QuadRemeshError::TooFine(estimate.max(tris.len())));
    }
    let fine2 = fine * fine;
    for _ in 0..24 {
        let mut middle: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &tris {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let key = (a.min(b), a.max(b));
                if middle.contains_key(&key) {
                    continue;
                }
                let (pa, pb) = (positions[a as usize], positions[b as usize]);
                if pa.distance_squared(pb) > fine2 {
                    middle.insert(key, positions.len() as u32);
                    positions.push(0.5 * (pa + pb));
                }
            }
        }
        if middle.is_empty() {
            break;
        }
        if tris.len() * 4 > MAX_TRIANGLES {
            return Err(QuadRemeshError::TooFine(tris.len() * 4));
        }
        let mid = |a: u32, b: u32| middle.get(&(a.min(b), a.max(b))).copied();
        let mut next = Vec::with_capacity(tris.len() * 2);
        for &[a, b, c] in &tris {
            match (mid(a, b), mid(b, c), mid(c, a)) {
                (None, None, None) => next.push([a, b, c]),
                (Some(ab), Some(bc), Some(ca)) => {
                    next.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
                }
                // One split: in two.
                (Some(m), None, None) => next.extend([[a, m, c], [m, b, c]]),
                (None, Some(m), None) => next.extend([[b, m, a], [m, c, a]]),
                (None, None, Some(m)) => next.extend([[c, m, b], [m, a, b]]),
                // Two: the corner between them cut off, the rest in two.
                (Some(ab), Some(bc), None) => next.extend([[ab, b, bc], [a, ab, bc], [a, bc, c]]),
                (None, Some(bc), Some(ca)) => next.extend([[bc, c, ca], [b, bc, ca], [b, ca, a]]),
                (Some(ab), None, Some(ca)) => next.extend([[ca, a, ab], [c, ca, ab], [c, ab, b]]),
            }
        }
        tris = next;
    }
    Ok((positions, tris))
}

// --- the hierarchy -------------------------------------------------------------

/// The vertices at one coarseness: where, which way they face, how much
/// surface they stand for, who they touch, and what direction — if any —
/// the shape holds them to.
struct Level {
    v: Vec<Vec3>,
    n: Vec<Vec3>,
    area: Vec<f32>,
    start: Vec<u32>,
    adj: Vec<u32>,
    held: Vec<Option<Vec3>>,
    /// Each vertex's vertex at the next level up; empty at the top.
    parent: Vec<u32>,
}

impl Level {
    fn len(&self) -> usize {
        self.v.len()
    }

    /// How far apart neighbours are, typically.
    fn spacing(&self) -> f32 {
        let mut sum = 0.0f64;
        for i in 0..self.len() {
            for &j in self.neighbours(i) {
                sum += self.v[i].distance(self.v[j as usize]) as f64;
            }
        }
        (sum / self.adj.len().max(1) as f64) as f32
    }

    fn neighbours(&self, i: usize) -> &[u32] {
        &self.adj[self.start[i] as usize..self.start[i + 1] as usize]
    }

    fn from_triangles(positions: &[Vec3], tris: &[[u32; 3]]) -> Self {
        let count = positions.len();
        let mut n = vec![Vec3::ZERO; count];
        let mut area = vec![0.0f32; count];
        // Each edge, with the faces either side of it: one face is an open
        // edge, two that meet at an angle a crease.
        let mut edges: HashMap<(u32, u32), (Vec3, u32, Vec3)> = HashMap::new();
        for t in tris {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            let cross = (b - a).cross(c - a);
            let face_n = cross.normalize_or_zero();
            for k in 0..3 {
                n[t[k] as usize] += cross;
                area[t[k] as usize] += cross.length() / 6.0;
                let (i, j) = (t[k], t[(k + 1) % 3]);
                let e = edges.entry((i.min(j), i.max(j))).or_insert((
                    positions[j as usize] - positions[i as usize],
                    0,
                    face_n,
                ));
                e.1 += 1;
                if e.1 == 2 {
                    e.2 = if e.2.dot(face_n) < SHARP_DEGREES.to_radians().cos() {
                        Vec3::ZERO // marks a crease
                    } else {
                        face_n
                    };
                }
            }
        }
        let n: Vec<Vec3> = n
            .into_iter()
            .map(|n| n.try_normalize().unwrap_or(Vec3::Z))
            .collect();
        // What holds each vertex: the open edges or creases through it,
        // when there are one or two — a line to run along. At a corner,
        // where more meet, nothing: the field turns there.
        let mut along: Vec<(u32, Vec3)> = vec![(0, Vec3::ZERO); count];
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(edges.len() * 2);
        for (&(i, j), &(dir, faces, face_n)) in &edges {
            pairs.push((i, j));
            pairs.push((j, i));
            if faces == 1 || (faces == 2 && face_n == Vec3::ZERO) {
                for v in [i, j] {
                    let (k, sum) = &mut along[v as usize];
                    // Lined up with what is there so far, so two halves of
                    // one line add rather than cancel.
                    let d = dir.normalize_or_zero();
                    *sum += if sum.dot(d) < 0.0 { -d } else { d };
                    *k += 1;
                }
            }
        }
        let held = along
            .iter()
            .zip(&n)
            .map(|(&(k, sum), n)| {
                if !(1..=2).contains(&k) {
                    return None;
                }
                (sum - *n * n.dot(sum)).try_normalize()
            })
            .collect();
        pairs.sort_unstable();
        let (start, adj) = csr(count, &pairs);
        Level {
            v: positions.to_vec(),
            n,
            area,
            start,
            adj,
            held,
            parent: Vec::new(),
        }
    }

    /// Each vertex not yet taken gathering the neighbours not yet taken that
    /// face about the same way: the next level up, and which vertex of it
    /// each vertex here went into. In order, so the same mesh coarsens the
    /// same way.
    fn coarsen(&self) -> (Level, Vec<u32>) {
        let mut parent = vec![u32::MAX; self.len()];
        let mut v = Vec::new();
        let mut n = Vec::new();
        let mut area = Vec::new();
        let mut held = Vec::new();
        for i in 0..self.len() {
            if parent[i] != u32::MAX {
                continue;
            }
            let id = v.len() as u32;
            parent[i] = id;
            let mut members = vec![i];
            for &j in self.neighbours(i) {
                let j = j as usize;
                if parent[j] == u32::MAX && self.n[i].dot(self.n[j]) > 0.7 {
                    parent[j] = id;
                    members.push(j);
                }
            }
            let a: f32 = members.iter().map(|&k| self.area[k]).sum();
            let middle = if a > 0.0 {
                members
                    .iter()
                    .map(|&k| self.v[k] * self.area[k])
                    .sum::<Vec3>()
                    / a
            } else {
                members.iter().map(|&k| self.v[k]).sum::<Vec3>() / members.len() as f32
            };
            let normal = members
                .iter()
                .map(|&k| self.n[k] * self.area[k].max(1e-20))
                .sum::<Vec3>()
                .try_normalize()
                .unwrap_or(self.n[i]);
            v.push(middle);
            n.push(normal);
            area.push(a);
            held.push(
                members
                    .iter()
                    .find_map(|&k| self.held[k])
                    .and_then(|d| (d - normal * normal.dot(d)).try_normalize()),
            );
        }
        let mut pairs = Vec::new();
        for i in 0..self.len() {
            for &j in self.neighbours(i) {
                let (a, b) = (parent[i], parent[j as usize]);
                if a != b {
                    pairs.push((a, b));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let (start, adj) = csr(v.len(), &pairs);
        (
            Level {
                v,
                n,
                area,
                start,
                adj,
                held,
                parent: Vec::new(),
            },
            parent,
        )
    }
}

/// Sorted, distinct `(from, to)` pairs as offsets and a flat list.
fn csr(count: usize, pairs: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let mut start = vec![0u32; count + 1];
    let mut adj = Vec::with_capacity(pairs.len());
    let mut last = None;
    for &(a, b) in pairs {
        if last == Some((a, b)) {
            continue;
        }
        last = Some((a, b));
        start[a as usize + 1] += 1;
        adj.push(b);
    }
    for i in 0..count {
        start[i + 1] += start[i];
    }
    (start, adj)
}

// --- 2 and 3. the fields -------------------------------------------------------

/// Of `q0` and `q1`, each turned by some quarter turn about its normal, the
/// two that point most nearly the same way.
fn compat_orientation(q0: Vec3, n0: Vec3, q1: Vec3, n1: Vec3) -> (Vec3, Vec3) {
    let a = [q0, n0.cross(q0)];
    let b = [q1, n1.cross(q1)];
    let mut best = (f32::MIN, 0, 0);
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            let s = ai.dot(*bj).abs();
            if s > best.0 {
                best = (s, i, j);
            }
        }
    }
    let d = a[best.1].dot(b[best.2]);
    (a[best.1], b[best.2] * if d < 0.0 { -1.0 } else { 1.0 })
}

/// The point nearest both tangent planes, between the two vertices.
fn middle_point(p0: Vec3, n0: Vec3, p1: Vec3, n1: Vec3) -> Vec3 {
    let n0p0 = n0.dot(p0);
    let n0p1 = n0.dot(p1);
    let n1p1 = n1.dot(p1);
    let n1p0 = n1.dot(p0);
    let n0n1 = n0.dot(n1);
    let denom = 1.0 / (1.0 - n0n1 * n0n1 + 1e-4);
    let l0 = 2.0 * (n0p1 - n0p0 - n0n1 * (n1p0 - n1p1)) * denom;
    let l1 = 2.0 * (n1p0 - n1p1 - n0n1 * (n0p1 - n0p0)) * denom;
    0.5 * (p0 + p1) - 0.25 * (n0 * l0 + n1 * l1)
}

/// The grid corner below `p` on the grid through `o` along `q`, in steps.
fn floor_index(o: Vec3, q: Vec3, n: Vec3, p: Vec3, inv: f32) -> [i32; 2] {
    let t = n.cross(q);
    let d = p - o;
    [
        (q.dot(d) * inv).floor() as i32,
        (t.dot(d) * inv).floor() as i32,
    ]
}

fn corner(o: Vec3, q: Vec3, n: Vec3, at: [i32; 2], scale: f32) -> Vec3 {
    o + q * (at[0] as f32 * scale) + n.cross(q) * (at[1] as f32 * scale)
}

/// The two grids' corners, one from each, nearest each other near the two
/// vertices — in each grid's own steps.
#[allow(clippy::too_many_arguments)]
fn compat_position_index(
    v0: Vec3,
    n0: Vec3,
    q0: Vec3,
    o0: Vec3,
    v1: Vec3,
    n1: Vec3,
    q1: Vec3,
    o1: Vec3,
    scale: f32,
) -> ([i32; 2], [i32; 2]) {
    let inv = 1.0 / scale;
    let middle = middle_point(v0, n0, v1, n1);
    let base0 = floor_index(o0, q0, n0, middle, inv);
    let base1 = floor_index(o1, q1, n1, middle, inv);
    let mut best = (f32::MAX, base0, base1);
    for i in 0..4 {
        let a = [base0[0] + (i & 1), base0[1] + (i >> 1)];
        let pa = corner(o0, q0, n0, a, scale);
        for j in 0..4 {
            let b = [base1[0] + (j & 1), base1[1] + (j >> 1)];
            let cost = pa.distance_squared(corner(o1, q1, n1, b, scale));
            if cost < best.0 {
                best = (cost, a, b);
            }
        }
    }
    (best.1, best.2)
}

/// `o` moved by whole steps of its grid to the corner nearest `v`.
fn round_to(o: Vec3, q: Vec3, n: Vec3, v: Vec3, scale: f32) -> Vec3 {
    let t = n.cross(q);
    let d = v - o;
    let inv = 1.0 / scale;
    o + q * ((q.dot(d) * inv).round() * scale) + t * ((t.dot(d) * inv).round() * scale)
}

/// The cross field and the grid positions at the finest level.
fn solve(levels: &[Level], scale: f32) -> (Vec<Vec3>, Vec<Vec3>) {
    const SWEEPS: usize = 20;
    let top = levels.last().expect("one level");
    let mut q: Vec<Vec3> = (0..top.len())
        .map(|i| top.held[i].unwrap_or_else(|| top.n[i].any_orthonormal_vector()))
        .collect();
    for at in (0..levels.len()).rev() {
        let level = &levels[at];
        if at + 1 < levels.len() {
            // Down from the level above.
            q = (0..level.len())
                .map(|i| {
                    let n = level.n[i];
                    level.held[i].unwrap_or_else(|| {
                        let up = q[level.parent[i] as usize];
                        (up - n * n.dot(up))
                            .try_normalize()
                            .unwrap_or_else(|| n.any_orthonormal_vector())
                    })
                })
                .collect();
        }
        for _ in 0..SWEEPS {
            for i in 0..level.len() {
                if level.held[i].is_some() {
                    continue;
                }
                let n = level.n[i];
                let mut sum = q[i];
                let mut weight = 0.0f32;
                for &j in level.neighbours(i) {
                    let j = j as usize;
                    let (a, b) = compat_orientation(sum, n, q[j], level.n[j]);
                    sum = a * weight + b;
                    sum -= n * n.dot(sum);
                    weight += 1.0;
                    sum = sum.try_normalize().unwrap_or(q[i]);
                }
                q[i] = sum;
            }
        }
    }
    // Positions, the same way, with the finished field at each level: the
    // finest field carried up by its parents.
    let mut fields: Vec<Vec<Vec3>> = vec![q];
    for at in 1..levels.len() {
        let below = &levels[at - 1];
        let level = &levels[at];
        let mut up = vec![Vec3::ZERO; level.len()];
        #[allow(clippy::needless_range_loop)] // several arrays, one vertex
        for i in 0..below.len() {
            let p = below.parent[i] as usize;
            let mine = fields[at - 1][i];
            up[p] = if up[p] == Vec3::ZERO {
                mine
            } else {
                let (a, b) = compat_orientation(up[p], level.n[p], mine, below.n[i]);
                a + b
            };
        }
        let up = up
            .iter()
            .zip(&level.n)
            .zip(&level.held)
            .map(|((u, n), held)| {
                held.unwrap_or_else(|| {
                    (*u - *n * n.dot(*u))
                        .try_normalize()
                        .unwrap_or_else(|| n.any_orthonormal_vector())
                })
            })
            .collect();
        fields.push(up);
    }
    // Only levels whose vertices are well under a square apart: coarser,
    // the grid between two vertices is a guess, and a wrong guess there is a
    // fault in the grid that no smoothing below can undo.
    let first = (0..levels.len())
        .rev()
        .find(|&at| levels[at].spacing() < 2.0 * scale)
        .unwrap_or(0);
    let mut o: Vec<Vec3> = levels[first].v.clone();
    for at in (0..=first).rev() {
        let level = &levels[at];
        let q = &fields[at];
        if at < first {
            o = (0..level.len())
                .map(|i| {
                    let up = o[level.parent[i] as usize];
                    let n = level.n[i];
                    up - n * n.dot(up - level.v[i])
                })
                .collect();
        }
        for _ in 0..SWEEPS {
            for i in 0..level.len() {
                let (v, n) = (level.v[i], level.n[i]);
                let mut sum = o[i];
                let mut weight = 0.0f32;
                for &j in level.neighbours(i) {
                    let j = j as usize;
                    let (a, b) = compat_position_index(
                        v, n, q[i], sum, level.v[j], level.n[j], q[j], o[j], scale,
                    );
                    let pa = corner(sum, q[i], n, a, scale);
                    let pb = corner(o[j], q[j], level.n[j], b, scale);
                    sum = (pa * weight + pb) / (weight + 1.0);
                    weight += 1.0;
                    sum -= n * n.dot(sum - v);
                }
                if level.held[i].is_some() {
                    // A grid line along the edge it is held to.
                    let t = n.cross(q[i]);
                    sum += t * t.dot(v - sum);
                }
                o[i] = round_to(sum, q[i], n, v, scale);
            }
        }
    }
    (fields.swap_remove(0), o)
}

// --- 4. squares ----------------------------------------------------------------

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let up = parent[parent[i as usize] as usize];
        parent[i as usize] = up;
        i = up;
    }
    i
}

/// The new vertices and the faces between them.
fn extract(level: &Level, q: &[Vec3], o: &[Vec3], scale: f32) -> (Vec<Vec3>, Vec<Vec<u32>>) {
    let count = level.len();
    let mut set: Vec<u32> = (0..count as u32).collect();
    let mut links: Vec<(u32, u32)> = Vec::new();
    for i in 0..count {
        for &j in level.neighbours(i) {
            if (j as usize) < i {
                continue;
            }
            let j = j as usize;
            let (n0, n1) = (level.n[i], level.n[j]);
            let (si, sj) = compat_position_index(
                level.v[i], n0, q[i], o[i], level.v[j], n1, q[j], o[j], scale,
            );
            // j's steps in i's directions: q[j] is one of ±q[i], ±t[i].
            let t0 = n0.cross(q[i]);
            let (dq, dt) = (q[j].dot(q[i]), q[j].dot(t0));
            let (a, b) = if dq.abs() > dt.abs() {
                (dq.signum() as i32, 0)
            } else {
                (0, dt.signum() as i32)
            };
            let turned = [sj[0] * a - sj[1] * b, sj[0] * b + sj[1] * a];
            let diff = [si[0] - turned[0], si[1] - turned[1]];
            match diff[0].abs() + diff[1].abs() {
                0 => {
                    let (ri, rj) = (find(&mut set, i as u32), find(&mut set, j as u32));
                    if ri != rj {
                        set[ri.max(rj) as usize] = ri.min(rj);
                    }
                }
                1 => links.push((i as u32, j as u32)),
                _ => {}
            }
        }
    }
    // Where the field is not quite one grid, one corner can be taken by two
    // sets that never touch — and two vertices a fraction of a square apart
    // turn the squares round them into triangles. Sets whose corners are
    // that close, facing the same way, are one.
    {
        let mut centre: HashMap<u32, (Vec3, Vec3, f32)> = HashMap::new();
        #[allow(clippy::needless_range_loop)] // several arrays, one vertex
        for i in 0..count {
            let r = find(&mut set, i as u32);
            let c = centre.entry(r).or_insert((Vec3::ZERO, Vec3::ZERO, 0.0));
            c.0 += o[i];
            c.1 += level.n[i];
            c.2 += 1.0;
        }
        let mut roots: Vec<(u32, Vec3, Vec3)> = centre
            .into_iter()
            .map(|(r, c)| (r, c.0 / c.2, c.1.normalize_or_zero()))
            .collect();
        roots.sort_unstable_by_key(|r| r.0);
        let cell = 0.5 * scale;
        let key = |p: Vec3| (p / cell).floor().as_ivec3();
        let mut grid: HashMap<kreate_math::glam::IVec3, Vec<usize>> = HashMap::new();
        for (k, r) in roots.iter().enumerate() {
            grid.entry(key(r.1)).or_default().push(k);
        }
        for (k, &(r, p, n)) in roots.iter().enumerate() {
            let at = key(p);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let Some(near) =
                            grid.get(&(at + kreate_math::glam::IVec3::new(dx, dy, dz)))
                        else {
                            continue;
                        };
                        for &m in near {
                            let (r2, p2, n2) = roots[m];
                            if m <= k || p.distance(p2) > cell || n.dot(n2) < 0.5 {
                                continue;
                            }
                            let (a, b) = (find(&mut set, r), find(&mut set, r2));
                            if a != b {
                                set[a.max(b) as usize] = a.min(b);
                            }
                        }
                    }
                }
            }
        }
    }
    // One vertex for each set: where its grid corners are, on average.
    let mut index = vec![u32::MAX; count];
    let mut sum: Vec<(Vec3, Vec3, f32)> = Vec::new();
    #[allow(clippy::needless_range_loop)] // several arrays, one vertex
    for i in 0..count {
        let r = find(&mut set, i as u32) as usize;
        if index[r] == u32::MAX {
            index[r] = sum.len() as u32;
            sum.push((Vec3::ZERO, Vec3::ZERO, 0.0));
        }
        let s = &mut sum[index[r] as usize];
        s.0 += o[i];
        s.1 += level.n[i];
        s.2 += 1.0;
    }
    let node = |set: &mut [u32], i: u32| index[find(set, i) as usize];
    let points: Vec<Vec3> = sum.iter().map(|s| s.0 / s.2).collect();
    let normals: Vec<Vec3> = sum
        .iter()
        .map(|s| s.1.try_normalize().unwrap_or(Vec3::Z))
        .collect();
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for (i, j) in links {
        let (a, b) = (node(&mut set, i), node(&mut set, j));
        if a != b {
            pairs.push((a, b));
            pairs.push((b, a));
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    // Each new vertex's field direction, gathered from its members.
    let mut axis = vec![Vec3::ZERO; points.len()];
    let mut taken = vec![0.0f32; points.len()];
    #[allow(clippy::needless_range_loop)] // several arrays, one vertex
    for i in 0..count {
        let a = node(&mut set, i as u32) as usize;
        let n = normals[a];
        axis[a] = if taken[a] == 0.0 {
            q[i]
        } else {
            let (mine, theirs) = compat_orientation(axis[a], n, q[i], level.n[i]);
            (mine * taken[a] + theirs) / (taken[a] + 1.0)
        };
        axis[a] = (axis[a] - n * n.dot(axis[a])).normalize_or_zero();
        taken[a] += 1.0;
    }
    // A grid corner has one neighbour each way along the grid, no more. Where
    // the field is not one grid, a corner can be linked to two the same way —
    // and those links cut squares into triangles. The one most nearly along
    // the grid is kept, and a link only if both ends keep it.
    let way = |a: u32, b: u32| -> (usize, f32) {
        let (n, x) = (normals[a as usize], axis[a as usize]);
        let d = points[b as usize] - points[a as usize];
        let d = (d - n * n.dot(d)).normalize_or_zero();
        let (cx, cy) = (d.dot(x), d.dot(n.cross(x)));
        if cx.abs() >= cy.abs() {
            (if cx >= 0.0 { 0 } else { 2 }, cx.abs())
        } else {
            (if cy >= 0.0 { 1 } else { 3 }, cy.abs())
        }
    };
    let mut best: HashMap<(u32, usize), (f32, u32)> = HashMap::new();
    for &(a, b) in &pairs {
        let (k, score) = way(a, b);
        let slot = best.entry((a, k)).or_insert((f32::MIN, b));
        if score > slot.0 {
            *slot = (score, b);
        }
    }
    let keeps = |a: u32, b: u32| best.get(&(a, way(a, b).0)).is_some_and(|s| s.1 == b);
    let pairs: Vec<(u32, u32)> = pairs
        .iter()
        .copied()
        .filter(|&(a, b)| keeps(a, b) && keeps(b, a))
        .collect();
    let (start, mut adj) = csr(points.len(), &pairs);
    // Round each vertex anticlockwise, seen from outside.
    for v in 0..points.len() {
        let n = normals[v];
        let x = n.any_orthonormal_vector();
        let y = n.cross(x);
        let ring = &mut adj[start[v] as usize..start[v + 1] as usize];
        ring.sort_by(|&a, &b| {
            let angle = |w: u32| {
                let d = points[w as usize] - points[v];
                d.dot(y).atan2(d.dot(x))
            };
            angle(a).total_cmp(&angle(b))
        });
    }
    let ring = |v: u32| &adj[start[v as usize] as usize..start[v as usize + 1] as usize];
    // The edge after `from → v` round the face on its left: v's neighbour
    // just before `from`, going anticlockwise.
    let next = |from: u32, v: u32| -> Option<u32> {
        let r = ring(v);
        let k = r.iter().position(|&w| w == from)?;
        Some(r[(k + r.len() - 1) % r.len()])
    };
    let slot = |a: u32, b: u32| -> usize {
        let r = ring(a);
        start[a as usize] as usize + r.iter().position(|&w| w == b).expect("an edge")
    };
    let mut used = vec![false; adj.len()];
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for a in 0..points.len() as u32 {
        for &b in ring(a) {
            if used[slot(a, b)] {
                continue;
            }
            let mut loop_ = vec![a, b];
            let (mut from, mut at) = (a, b);
            let closed = loop {
                let Some(w) = next(from, at) else { break false };
                if at == a && w == b {
                    loop_.pop();
                    break true;
                }
                if loop_.len() > 6 {
                    break false;
                }
                loop_.push(w);
                (from, at) = (at, w);
            };
            if !closed || loop_.len() < 3 {
                continue;
            }
            // Faces the way the surface does — not the outside of an open
            // edge, or a fold.
            let mut newell = Vec3::ZERO;
            let mut normal = Vec3::ZERO;
            for k in 0..loop_.len() {
                let (p, r) = (
                    points[loop_[k] as usize],
                    points[loop_[(k + 1) % loop_.len()] as usize],
                );
                newell += p.cross(r);
                normal += normals[loop_[k] as usize];
            }
            if newell.dot(normal) <= 0.0 {
                continue;
            }
            for k in 0..loop_.len() {
                used[slot(loop_[k], loop_[(k + 1) % loop_.len()])] = true;
            }
            faces.push(loop_);
        }
    }
    // Only the vertices some face uses.
    let mut keep = vec![u32::MAX; points.len()];
    let mut kept = Vec::new();
    for f in &mut faces {
        for v in f.iter_mut() {
            if keep[*v as usize] == u32::MAX {
                keep[*v as usize] = kept.len() as u32;
                kept.push(points[*v as usize]);
            }
            *v = keep[*v as usize];
        }
    }
    (kept, faces)
}

// --- 5. even out ---------------------------------------------------------------

/// Each vertex moved toward the middle of its neighbours, along the surface,
/// and put back on the surface it came from.
fn relax(
    mut points: Vec<Vec3>,
    faces: &[Vec<u32>],
    positions: &[Vec3],
    tris: &[[u32; 3]],
    scale: f32,
) -> Vec<Vec3> {
    let bvh = Bvh::build(positions, tris);
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); points.len()];
    let mut open: HashMap<(u32, u32), u32> = HashMap::new();
    for f in faces {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            neighbours[a as usize].push(b);
            neighbours[b as usize].push(a);
            *open.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    // The vertices on an open edge stay where they are along it.
    let mut pinned = vec![false; points.len()];
    for (&(a, b), &count) in &open {
        if count == 1 {
            pinned[a as usize] = true;
            pinned[b as usize] = true;
        }
    }
    for list in &mut neighbours {
        list.sort_unstable();
        list.dedup();
    }
    let mut hits = Vec::new();
    let nearest = |p: Vec3, hits: &mut Vec<u32>| -> Option<Vec3> {
        let mut reach = scale;
        for _ in 0..4 {
            hits.clear();
            bvh.overlaps(&Aabb::new(p, p).expanded(reach), hits);
            let best = hits
                .iter()
                .map(|&t| {
                    let [a, b, c] = tris[t as usize].map(|i| positions[i as usize]);
                    closest_on_triangle(p, a, b, c)
                })
                .min_by(|a, b| a.distance_squared(p).total_cmp(&b.distance_squared(p)));
            if best.is_some() {
                return best;
            }
            reach *= 2.0;
        }
        None
    };
    for p in points.iter_mut() {
        if let Some(on) = nearest(*p, &mut hits) {
            *p = on;
        }
    }
    for _ in 0..4 {
        let before = points.clone();
        for (i, p) in points.iter_mut().enumerate() {
            if pinned[i] || neighbours[i].is_empty() {
                continue;
            }
            let middle = neighbours[i]
                .iter()
                .map(|&j| before[j as usize])
                .sum::<Vec3>()
                / neighbours[i].len() as f32;
            let moved = before[i].lerp(middle, 0.5);
            if let Some(on) = nearest(moved, &mut hits) {
                *p = on;
            }
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    struct Stats {
        faces: usize,
        quads: usize,
        open_edges: usize,
        edges: usize,
    }

    fn stats(m: &MeshData) -> Stats {
        let mut count = HashMap::new();
        let mut quads = 0;
        for f in 0..m.num_faces() {
            let ring = m.face_verts(f);
            if ring.len() == 4 {
                quads += 1;
            }
            for k in 0..ring.len() {
                let (a, b) = (ring[k], ring[(k + 1) % ring.len()]);
                *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        Stats {
            faces: m.num_faces() as usize,
            quads,
            open_edges: count.values().filter(|c| **c != 2).count(),
            edges: count.len(),
        }
    }

    #[test]
    fn a_sphere_becomes_squares_on_its_surface() {
        let ball = primitives::uv_sphere(1.0, 48, 24);
        let m = quad_remesh(&ball, 0.15).unwrap();
        let s = stats(&m);
        eprintln!(
            "sphere: {} faces, {} quads, {} of {} edges not between two",
            s.faces, s.quads, s.open_edges, s.edges
        );
        assert!(s.quads as f32 > 0.9 * s.faces as f32, "mostly squares");
        assert!(
            (s.open_edges as f32) < 0.02 * s.edges as f32,
            "nearly closed"
        );
        let expected = 4.0 * std::f32::consts::PI / (0.15 * 0.15);
        assert!(
            (s.faces as f32 - expected).abs() < 0.3 * expected,
            "{} faces near {expected}",
            s.faces
        );
        for p in m.positions() {
            assert!(
                (p.length() - 1.0).abs() < 0.02,
                "on the surface: {}",
                p.length()
            );
        }
    }

    #[test]
    fn a_cube_comes_back_in_squares_along_its_edges() {
        let cube = primitives::cube(2.0);
        let m = quad_remesh(&cube, 0.25).unwrap();
        let s = stats(&m);
        eprintln!(
            "cube: {} faces, {} quads, {} of {} edges not between two",
            s.faces, s.quads, s.open_edges, s.edges
        );
        assert!(s.quads as f32 > 0.95 * s.faces as f32, "squares");
        assert_eq!(s.open_edges, 0, "closed");
        // Every vertex on a face of the cube, its edges kept.
        for p in m.positions() {
            assert!(
                (p.abs().max_element() - 1.0).abs() < 1e-3,
                "on the cube: {p}"
            );
        }
    }

    #[test]
    fn an_open_sheet_keeps_its_edge() {
        let sheet = primitives::grid(2.0, 4, 4);
        let m = quad_remesh(&sheet, 0.2).unwrap();
        let s = stats(&m);
        eprintln!("sheet: {} faces, {} quads", s.faces, s.quads);
        assert!(s.quads as f32 > 0.95 * s.faces as f32, "squares");
        let (lo, hi) = m
            .positions()
            .iter()
            .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| {
                (lo.min(*p), hi.max(*p))
            });
        assert!(
            lo.x < -0.9 && hi.x > 0.9 && lo.y < -0.9 && hi.y > 0.9,
            "{lo} {hi}"
        );
    }

    #[test]
    fn too_fine_and_nothing_are_said() {
        let cube = primitives::cube(2.0);
        assert!(matches!(
            quad_remesh(&cube, 0.0002),
            Err(QuadRemeshError::TooFine(_))
        ));
        assert!(matches!(
            quad_remesh(&MeshData::empty(), 0.1),
            Err(QuadRemeshError::Empty)
        ));
    }

    #[test]
    fn the_face_count_asked_for_sets_the_size() {
        let cube = primitives::cube(2.0);
        let e = edge_for_faces(&cube, 600);
        assert!((e - 0.2).abs() < 1e-4, "{e}");
    }
}
