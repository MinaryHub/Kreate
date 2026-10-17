//! The decimate modifier: fewer triangles, the shape kept — edges collapsed
//! cheapest first, the cost of each the squared distance its new vertex
//! would stand from the planes of the faces it had (Garland and Heckbert's
//! quadric error metric). A scan or a sculpt brought down to what a game
//! or a web page can carry.

use kreate_math::{DVec3, Vec3};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::MeshData;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecimateParams {
    /// The share of the triangles kept, 0 to 1.
    pub ratio: f32,
}

impl Default for DecimateParams {
    fn default() -> Self {
        Self { ratio: 0.5 }
    }
}

/// A symmetric 4 × 4 quadric, the upper triangle.
#[derive(Clone, Copy, Debug, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn plane(n: DVec3, d: f64) -> Self {
        let (a, b, c) = (n.x, n.y, n.z);
        Self([
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        ])
    }

    fn add(&mut self, other: &Quadric) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a += b;
        }
    }

    fn cost(&self, v: DVec3) -> f64 {
        let q = &self.0;
        let (x, y, z) = (v.x, v.y, v.z);
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    cost: f64,
    a: u32,
    b: u32,
    at: Vec3,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // The heap pops the largest: reversed, so the cheapest comes first.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// `input` with about `ratio` of its triangles left.
///
/// Triangulated first; what comes out is triangles. A collapse that would
/// turn a triangle inside out is passed over, which keeps the surface from
/// folding on itself.
pub fn decimate(input: &MeshData, params: &DecimateParams) -> MeshData {
    let ratio = params.ratio.clamp(0.0, 1.0);
    if input.is_empty() || ratio >= 1.0 {
        return input.clone();
    }
    let tri = input.triangulate();
    let (tris, _) = tri.indices.as_chunks::<3>();
    let mut pos: Vec<Vec3> = input.positions().to_vec();
    let mut tris: Vec<[u32; 3]> = tris.to_vec();
    let mut alive = vec![true; tris.len()];
    let target = ((tris.len() as f32 * ratio).round() as usize).max(2);
    let mut live = tris.len();
    if live <= target {
        return input.clone();
    }

    // The quadric at each vertex: the planes of its triangles.
    let mut quadrics = vec![Quadric::default(); pos.len()];
    for t in &tris {
        let (a, b, c) = (
            pos[t[0] as usize].as_dvec3(),
            pos[t[1] as usize].as_dvec3(),
            pos[t[2] as usize].as_dvec3(),
        );
        let n = (b - a).cross(c - a);
        let len = n.length();
        if len < 1e-18 {
            continue;
        }
        let n = n / len;
        let q = Quadric::plane(n, -n.dot(a));
        for v in t {
            quadrics[*v as usize].add(&q);
        }
    }
    // The triangles at each vertex.
    let mut vtris: Vec<Vec<u32>> = vec![Vec::new(); pos.len()];
    for (i, t) in tris.iter().enumerate() {
        for v in t {
            vtris[*v as usize].push(i as u32);
        }
    }
    // Where a vertex went when it was collapsed away.
    let mut merged: Vec<u32> = (0..pos.len() as u32).collect();
    let find = |merged: &Vec<u32>, mut v: u32| {
        while merged[v as usize] != v {
            v = merged[v as usize];
        }
        v
    };

    let candidate = |a: u32, b: u32, pos: &[Vec3], quadrics: &[Quadric]| -> Candidate {
        let mut q = quadrics[a as usize];
        q.add(&quadrics[b as usize]);
        let (pa, pb) = (pos[a as usize], pos[b as usize]);
        // The cheapest of the two ends and their middle: no matrix to
        // invert, and never a vertex flung off the surface.
        let mid = (pa + pb) * 0.5;
        let best = [pa, pb, mid]
            .into_iter()
            .map(|p| (q.cost(p.as_dvec3()), p))
            .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal))
            .expect("three");
        Candidate {
            cost: best.0.max(0.0),
            a,
            b,
            at: best.1,
        }
    };
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for t in &tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            if seen.insert(key) {
                heap.push(candidate(key.0, key.1, &pos, &quadrics));
            }
        }
    }

    while live > target {
        let Some(c) = heap.pop() else { break };
        let (a, b) = (find(&merged, c.a), find(&merged, c.b));
        if a == b {
            continue;
        }
        // Still an edge?
        if !vtris[a as usize]
            .iter()
            .any(|t| alive[*t as usize] && tris[*t as usize].contains(&b))
        {
            continue;
        }
        // The link condition: the two ends share, as neighbours, only the
        // far corners of the triangles on the edge itself. Any other shared
        // neighbour would leave two triangles over one edge — a fin — and
        // the surface no longer closed.
        let neighbours = |v: u32| -> Vec<u32> {
            let mut out = Vec::new();
            for t in &vtris[v as usize] {
                if alive[*t as usize] {
                    for w in tris[*t as usize] {
                        if w != v && !out.contains(&w) {
                            out.push(w);
                        }
                    }
                }
            }
            out
        };
        let (na, nb) = (neighbours(a), neighbours(b));
        let shared = na.iter().filter(|v| nb.contains(v)).count();
        let across = vtris[a as usize]
            .iter()
            .filter(|t| alive[**t as usize] && tris[**t as usize].contains(&b))
            .count();
        if shared != across {
            continue;
        }
        // Would any triangle round `a` or `b` that survives turn over?
        let mut flips = false;
        for v in [a, b] {
            for t in &vtris[v as usize] {
                let tri = tris[*t as usize];
                if !alive[*t as usize] || (tri.contains(&a) && tri.contains(&b)) {
                    continue;
                }
                let before = normal_of(&tri, &pos, None);
                let after = normal_of(&tri, &pos, Some((v, c.at)));
                if before.dot(after) < 0.0 {
                    flips = true;
                    break;
                }
            }
            if flips {
                break;
            }
        }
        if flips {
            continue;
        }
        // Collapse b into a, at the chosen place.
        pos[a as usize] = c.at;
        let qb = quadrics[b as usize];
        quadrics[a as usize].add(&qb);
        merged[b as usize] = a;
        let moved: Vec<u32> = vtris[b as usize].clone();
        for t in moved {
            if !alive[t as usize] {
                continue;
            }
            let tri = &mut tris[t as usize];
            if tri.contains(&a) {
                alive[t as usize] = false;
                live -= 1;
                continue;
            }
            for v in tri.iter_mut() {
                if *v == b {
                    *v = a;
                }
            }
            vtris[a as usize].push(t);
        }
        vtris[b as usize].clear();
        // The edges round the survivor, costed again.
        let mut around: Vec<u32> = Vec::new();
        for t in &vtris[a as usize] {
            if alive[*t as usize] {
                for v in tris[*t as usize] {
                    if v != a && !around.contains(&v) {
                        around.push(v);
                    }
                }
            }
        }
        for v in around {
            heap.push(candidate(a.min(v), a.max(v), &pos, &quadrics));
        }
    }

    // Out: the live triangles over the vertices still in use.
    let mut index = vec![u32::MAX; pos.len()];
    let mut out_pos: Vec<Vec3> = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(live);
    for (i, t) in tris.iter().enumerate() {
        if !alive[i] {
            continue;
        }
        let mut face = Vec::with_capacity(3);
        for v in t {
            let v = *v as usize;
            if index[v] == u32::MAX {
                index[v] = out_pos.len() as u32;
                out_pos.push(pos[v]);
            }
            face.push(index[v]);
        }
        if face[0] != face[1] && face[1] != face[2] && face[0] != face[2] {
            faces.push(face);
        }
    }
    MeshData::from_faces(out_pos, &faces)
}

fn normal_of(tri: &[u32; 3], pos: &[Vec3], moved: Option<(u32, Vec3)>) -> Vec3 {
    let at = |v: u32| match moved {
        Some((m, p)) if m == v => p,
        _ => pos[v as usize],
    };
    let (a, b, c) = (at(tri[0]), at(tri[1]), at(tri[2]));
    (b - a).cross(c - a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_sphere_comes_down_to_about_the_ratio_and_stays_a_sphere() {
        let sphere = primitives::uv_sphere(1.0, 48, 24);
        let out = decimate(&sphere, &DecimateParams { ratio: 0.25 });
        let before = sphere.triangulate().indices.len() / 3;
        let after = out.num_faces() as usize;
        assert!(
            after < before / 3 && after > before / 6,
            "{before} -> {after}"
        );
        assert!(out.faces().all(|f| f.len() == 3));
        for p in out.positions() {
            assert!((p.length() - 1.0).abs() < 0.05, "{p:?}");
        }
        assert!(crate::measure::is_closed(&out), "still closed");
    }

    #[test]
    fn a_ratio_of_one_leaves_the_mesh_alone() {
        let cube = primitives::cube(2.0);
        let out = decimate(&cube, &DecimateParams { ratio: 1.0 });
        assert_eq!(out.num_faces(), 6);
    }
}
