//! Dynamic topology: a sculpt stroke that cuts the surface finer where the
//! brush goes — Blender's Dyntopo.
//!
//! Before each dab touches the surface, the edges under it longer than the
//! detail size are split and those much shorter are collapsed, so the brush
//! always has about the same density of triangles to work with: a crease cut
//! into a coarse ball is a crease, not a jagged line between its few
//! vertices. Then the dab moves the vertices as any stroke does
//! ([`crate::sculpt::displace`]).
//!
//! The surface becomes triangles, as Blender's does, and a stroke with
//! dynamic topology leaves the mesh with nothing but its positions: UVs,
//! materials, the mask and face sets are not carried through a surface that
//! is being re-cut.
//!
//! Like [`crate::sculpt::StrokeCache`] a [`DynCache`] picks up where the
//! stroke stopped, so a growing stroke costs its new dabs.

use kreate_math::Vec3;

use crate::sculpt::{displace, Brush, Dab, Stroke};
use crate::MeshData;

/// Edges shorter than this fraction of the detail size are collapsed.
const COLLAPSE: f32 = 0.4;
/// At most this many vertices are made by one dab, so a detail size set
/// absurdly small slows the stroke rather than eating the memory.
const MAX_NEW: usize = 20_000;

/// A stroke with dynamic topology, part-way through.
pub struct DynCache {
    original: Vec<Vec3>,
    corners: Vec<u32>,
    key: (Stroke, u32),
    applied: Vec<Dab>,
    pos: Vec<Vec3>,
    /// Where each vertex was when it was made — at the stroke's start, or
    /// when an edge was split — which the layer brush measures from.
    born: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    alive: Vec<bool>,
    /// The triangles at each vertex.
    vtris: Vec<Vec<u32>>,
}

impl DynCache {
    pub fn new(mesh: &MeshData, stroke: &Stroke, detail: f32) -> Self {
        let tri = mesh.triangulate();
        let (t, _) = tri.indices.as_chunks::<3>();
        let pos = mesh.positions().to_vec();
        let mut vtris = vec![Vec::new(); pos.len()];
        for (i, tri) in t.iter().enumerate() {
            for v in tri {
                vtris[*v as usize].push(i as u32);
            }
        }
        Self {
            original: pos.clone(),
            corners: mesh.corner_verts().to_vec(),
            key: (Self::key(stroke), detail.to_bits()),
            applied: Vec::new(),
            born: pos.clone(),
            pos,
            alive: vec![true; t.len()],
            tris: t.to_vec(),
            vtris,
        }
    }

    fn key(stroke: &Stroke) -> Stroke {
        Stroke {
            grab: Vec3::ZERO,
            ..stroke.clone()
        }
    }

    /// Whether `dabs` on `mesh` continues what this has done.
    pub fn fits(&self, mesh: &MeshData, stroke: &Stroke, detail: f32, dabs: &[Dab]) -> bool {
        self.key == (Self::key(stroke), detail.to_bits())
            && dabs.starts_with(&self.applied)
            && mesh.corner_verts() == self.corners.as_slice()
            && mesh.positions() == self.original.as_slice()
    }

    /// Applies the new dabs, re-cutting under each, and replaces `mesh` with
    /// the result. `detail` is the edge length aimed at.
    pub fn run(&mut self, mesh: &mut MeshData, dabs: &[Dab], stroke: &Stroke, detail: f32) -> bool {
        let start = self.applied.len().min(dabs.len());
        let mut any = false;
        let copies = stroke_copies(stroke);
        for (k, d) in dabs.iter().enumerate().skip(start) {
            let before = k.checked_sub(1).map(|p| dabs[p]);
            for (sign, turn) in &copies {
                let place = |p: Vec3| *turn * (p * *sign);
                let dab = Dab { centre: place(d.centre), ..*d };
                let from = before.map(|b| place(b.centre));
                any |= self.dab(dab, from, stroke, detail.max(1e-5));
            }
        }
        self.applied = dabs.to_vec();
        if any {
            *mesh = self.to_mesh();
        }
        any
    }

    fn edge_tris(&self, a: u32, b: u32) -> Vec<u32> {
        self.vtris[a as usize]
            .iter()
            .copied()
            .filter(|t| self.alive[*t as usize] && self.tris[*t as usize].contains(&b))
            .collect()
    }

    fn split(&mut self, a: u32, b: u32) -> bool {
        let shared = self.edge_tris(a, b);
        if shared.is_empty() {
            return false;
        }
        let m = self.pos.len() as u32;
        self.pos.push((self.pos[a as usize] + self.pos[b as usize]) * 0.5);
        self.born.push((self.born[a as usize] + self.born[b as usize]) * 0.5);
        self.vtris.push(Vec::new());
        for t in shared {
            let tri = self.tris[t as usize];
            let first = tri.map(|v| if v == b { m } else { v });
            let second = tri.map(|v| if v == a { m } else { v });
            self.tris[t as usize] = first;
            self.vtris[b as usize].retain(|x| *x != t);
            self.vtris[m as usize].push(t);
            let new = self.tris.len() as u32;
            self.tris.push(second);
            self.alive.push(true);
            for v in second {
                self.vtris[v as usize].push(new);
            }
        }
        true
    }

    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut out: Vec<u32> = self.vtris[v as usize]
            .iter()
            .filter(|t| self.alive[**t as usize])
            .flat_map(|t| self.tris[*t as usize])
            .filter(|w| *w != v)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn on_border(&self, v: u32) -> bool {
        self.neighbours(v).iter().any(|w| self.edge_tris(v, *w).len() != 2)
    }

    fn normal_of(&self, t: [u32; 3]) -> Vec3 {
        let [a, b, c] = t.map(|i| self.pos[i as usize]);
        (b - a).cross(c - a)
    }

    /// Merges `b` into `a`, at their middle, when that leaves the surface a
    /// surface: an inside edge, the two triangles either side the only
    /// shared neighbours, and no triangle turned over.
    fn collapse(&mut self, a: u32, b: u32) -> bool {
        let shared = self.edge_tris(a, b);
        if shared.len() != 2 || self.on_border(a) || self.on_border(b) {
            return false;
        }
        let (na, nb) = (self.neighbours(a), self.neighbours(b));
        let common = na.iter().filter(|v| nb.contains(v)).count();
        if common != 2 {
            return false;
        }
        let mid = (self.pos[a as usize] + self.pos[b as usize]) * 0.5;
        // Every triangle that keeps going must not flip.
        for v in [a, b] {
            for t in &self.vtris[v as usize] {
                if !self.alive[*t as usize] || shared.contains(t) {
                    continue;
                }
                let before = self.normal_of(self.tris[*t as usize]);
                let after_tri = self.tris[*t as usize].map(|x| if x == a || x == b { u32::MAX } else { x });
                let [p, q, r] = after_tri.map(|x| if x == u32::MAX { mid } else { self.pos[x as usize] });
                let after = (q - p).cross(r - p);
                if before.dot(after) <= 0.0 {
                    return false;
                }
            }
        }
        for t in &shared {
            self.alive[*t as usize] = false;
        }
        let moving: Vec<u32> = self.vtris[b as usize]
            .iter()
            .copied()
            .filter(|t| self.alive[*t as usize])
            .collect();
        for t in moving {
            for v in self.tris[t as usize].iter_mut() {
                if *v == b {
                    *v = a;
                }
            }
            self.vtris[a as usize].push(t);
        }
        self.vtris[b as usize].clear();
        self.born[a as usize] = (self.born[a as usize] + self.born[b as usize]) * 0.5;
        self.pos[a as usize] = mid;
        true
    }

    /// The edges of the living triangles touching `verts`, once each.
    fn edges_near(&self, verts: &[u32]) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for v in verts {
            for t in &self.vtris[*v as usize] {
                if !self.alive[*t as usize] {
                    continue;
                }
                let tri = self.tris[*t as usize];
                for k in 0..3 {
                    let (x, y) = (tri[k], tri[(k + 1) % 3]);
                    out.push((x.min(y), x.max(y)));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn within(&self, centre: Vec3, radius: f32) -> Vec<u32> {
        (0..self.pos.len() as u32)
            .filter(|v| !self.vtris[*v as usize].is_empty() && self.pos[*v as usize].distance(centre) < radius)
            .collect()
    }

    fn dab(&mut self, dab: Dab, from: Option<Vec3>, stroke: &Stroke, detail: f32) -> bool {
        // Re-cut under the brush: long edges split, longest first, a few
        // rounds over; then short ones collapsed.
        let reach = dab.radius * 1.1;
        let made = self.pos.len();
        for _ in 0..4 {
            let near = self.within(dab.centre, reach);
            let mut long: Vec<(f32, u32, u32)> = self
                .edges_near(&near)
                .into_iter()
                .filter_map(|(a, b)| {
                    let (pa, pb) = (self.pos[a as usize], self.pos[b as usize]);
                    let len = pa.distance(pb);
                    (len > detail && ((pa + pb) * 0.5).distance(dab.centre) < reach).then_some((len, a, b))
                })
                .collect();
            if long.is_empty() || self.pos.len() - made > MAX_NEW {
                break;
            }
            long.sort_by(|x, y| y.0.total_cmp(&x.0));
            for (_, a, b) in long {
                if self.pos.len() - made > MAX_NEW {
                    break;
                }
                // Still an edge, still long: an earlier split may have cut it.
                if self.pos[a as usize].distance(self.pos[b as usize]) > detail {
                    self.split(a, b);
                }
            }
        }
        let near = self.within(dab.centre, dab.radius);
        for (a, b) in self.edges_near(&near) {
            if self.vtris[a as usize].is_empty() || self.vtris[b as usize].is_empty() {
                continue;
            }
            if self.pos[a as usize].distance(self.pos[b as usize]) < detail * COLLAPSE {
                self.collapse(a, b);
            }
        }
        // The brush, on the surface as it now is.
        let near = if stroke.brush == Brush::SnakeHook {
            (0..self.pos.len() as u32)
                .filter(|v| !self.vtris[*v as usize].is_empty())
                .collect()
        } else {
            self.within(dab.centre, dab.radius)
        };
        let under: Vec<(usize, f32)> = near
            .iter()
            .map(|v| {
                let d = self.pos[*v as usize].distance(dab.centre);
                (*v as usize, falloff_weight(stroke, d, dab.radius) * dab.pressure.clamp(0.0, 1.0))
            })
            .filter(|(_, w)| *w > 0.0)
            .collect();
        if under.is_empty() {
            return self.pos.len() != made;
        }
        let mut normal = vec![Vec3::ZERO; self.pos.len()];
        for (v, _) in &under {
            let n = self.vtris[*v]
                .iter()
                .filter(|t| self.alive[**t as usize])
                .fold(Vec3::ZERO, |s, t| s + self.normal_of(self.tris[*t as usize]));
            normal[*v] = n.normalize_or_zero();
        }
        let neighbours: std::collections::HashMap<usize, Vec<u32>> = if matches!(stroke.brush, Brush::Smooth)
            || stroke.auto_smooth > 0.0
        {
            under.iter().map(|(v, _)| (*v, self.neighbours(*v as u32))).collect()
        } else {
            Default::default()
        };
        let empty: &[u32] = &[];
        displace(
            stroke,
            dab,
            from,
            &under,
            &mut self.pos,
            &self.born,
            &normal,
            |i| neighbours.get(&i).map_or(empty, |n| n.as_slice()),
        );
        true
    }

    /// The surface as a mesh: the living triangles, over the vertices they
    /// use. Built straight into the mesh's own arrays — a list per triangle
    /// was most of a frame's cost on a dense surface.
    fn to_mesh(&self) -> MeshData {
        let mut map = vec![u32::MAX; self.pos.len()];
        let mut positions = Vec::with_capacity(self.pos.len());
        let mut corners = Vec::with_capacity(self.tris.len() * 3);
        for (t, tri) in self.tris.iter().enumerate() {
            if !self.alive[t] {
                continue;
            }
            for v in tri {
                if map[*v as usize] == u32::MAX {
                    map[*v as usize] = positions.len() as u32;
                    positions.push(self.pos[*v as usize]);
                }
                corners.push(map[*v as usize]);
            }
        }
        let offsets: Vec<u32> = (0..=corners.len() as u32 / 3).map(|f| f * 3).collect();
        let mut mesh = MeshData::from_csr(positions.len() as u32, offsets, corners, Vec::new())
            .expect("every corner names a vertex just made");
        mesh.positions_mut().copy_from_slice(&positions);
        mesh
    }
}

/// The falloff a stroke uses, for dynamic topology's own search.
fn falloff_weight(stroke: &Stroke, distance: f32, radius: f32) -> f32 {
    crate::sculpt::falloff_of(stroke.falloff, distance, radius)
}

/// The copies the stroke's symmetry makes of each dab.
fn stroke_copies(stroke: &Stroke) -> Vec<(Vec3, kreate_math::Quat)> {
    crate::sculpt::copies_of(stroke)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn stroke(brush: Brush) -> Stroke {
        Stroke {
            brush,
            strength: 1.0,
            ..Stroke::default()
        }
    }

    fn edges(m: &MeshData) -> Vec<f32> {
        m.edges()
            .iter()
            .map(|[a, b]| m.positions()[*a as usize].distance(m.positions()[*b as usize]))
            .collect()
    }

    #[test]
    fn a_dab_cuts_the_surface_under_it_to_the_detail_size() {
        let base = primitives::grid(2.0, 4, 4);
        let mut m = base.clone();
        let mut cache = DynCache::new(&m, &stroke(Brush::Draw), 0.1);
        assert!(cache.run(&mut m, &[Dab::new(Vec3::ZERO, 0.5)], &stroke(Brush::Draw), 0.1));
        assert!(m.num_faces() > base.num_faces() * 4, "{} faces", m.num_faces());
        // Under the brush the edges are near the detail size; far away the
        // grid is as it was.
        let fine = m
            .edges()
            .iter()
            .filter(|[a, b]| {
                let mid = (m.positions()[*a as usize] + m.positions()[*b as usize]) * 0.5;
                mid.truncate().length() < 0.2
            })
            .map(|[a, b]| m.positions()[*a as usize].distance(m.positions()[*b as usize]))
            .fold(0.0f32, f32::max);
        assert!(fine <= 0.15, "the longest edge under the brush: {fine}");
        assert!(edges(&m).iter().any(|l| *l > 0.45), "the far grid untouched");
        assert!(m.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max) > 0.02, "and drawn");
    }

    #[test]
    #[ignore]
    fn cost() {
        let base = primitives::uv_sphere(1.0, 128, 64);
        let s = stroke(Brush::Draw);
        let mut cache = DynCache::new(&base, &s, 0.03);
        let mut dabs = Vec::new();
        let (mut run, mut out) = (0.0, 0.0);
        for i in 0..40 {
            let a = i as f32 * 0.03;
            dabs.push(Dab::new(Vec3::new(a.cos(), a.sin(), 0.0), 0.15));
            let start = cache.applied.len();
            let t = std::time::Instant::now();
            let copies = stroke_copies(&s);
            for d in &dabs[start..] {
                for (sign, turn) in &copies {
                    let dab = Dab { centre: *turn * (d.centre * *sign), ..*d };
                    cache.dab(dab, None, &s, 0.03);
                }
            }
            cache.applied = dabs.clone();
            run += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let _ = cache.to_mesh();
            out += t.elapsed().as_secs_f64();
        }
        eprintln!("per frame: dabs {:.2} ms, to_mesh {:.2} ms", run * 25.0, out * 25.0);
    }

    #[test]
    fn the_surface_stays_a_surface() {
        // A closed ball, cut and collapsed under a long stroke: every edge
        // between two triangles still.
        let base = primitives::ico_sphere(1.0, 2);
        let mut m = base.clone();
        let s = stroke(Brush::Draw);
        let dabs: Vec<Dab> = (0..12)
            .map(|i| {
                let a = i as f32 * 0.2;
                Dab::new(Vec3::new(a.cos(), a.sin(), 0.0), 0.4)
            })
            .collect();
        let mut cache = DynCache::new(&base, &s, 0.08);
        cache.run(&mut m, &dabs, &s, 0.08);
        let mut count = std::collections::HashMap::new();
        for f in 0..m.num_faces() {
            let ring = m.face_verts(f);
            for k in 0..3 {
                let (a, b) = (ring[k], ring[(k + 1) % 3]);
                *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(count.values().all(|c| *c == 2), "closed and manifold");
    }

    #[test]
    fn a_continued_stroke_matches_one_run_whole() {
        let base = primitives::uv_sphere(1.0, 16, 8);
        let s = stroke(Brush::Inflate);
        let dabs: Vec<Dab> = (0..6).map(|i| Dab::new(Vec3::new(0.0, (i as f32 * 0.3).sin(), 1.0), 0.4)).collect();
        let mut whole = base.clone();
        DynCache::new(&base, &s, 0.1).run(&mut whole, &dabs, &s, 0.1);
        let mut cache = DynCache::new(&base, &s, 0.1);
        let mut last = base.clone();
        for n in [1, 3, 6] {
            let mut m = base.clone();
            assert!(cache.fits(&m, &s, 0.1, &dabs[..n]));
            cache.run(&mut m, &dabs[..n], &s, 0.1);
            last = m;
        }
        assert_eq!(last.positions(), whole.positions());
        assert!(!cache.fits(&base, &s, 0.2, &dabs), "another detail size is another stroke");
    }
}
