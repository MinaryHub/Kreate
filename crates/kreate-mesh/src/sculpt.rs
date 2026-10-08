//! Sculpting: a brush stroke as a function of the mesh and where the brush
//! went.
//!
//! A stroke is a list of dabs — a centre and a radius each, in the mesh's own
//! space — and a brush. Applied in order to the stroke's starting mesh, the
//! same dabs make the same shape, which is what lets the stroke be one
//! operator: replayable, one step to undo, and re-run from the start while
//! the pointer is still moving.
//!
//! Falloff is Blender's Smooth curve (`3t² − 2t³`, full in the middle, nothing
//! at the rim), and every brush reads the normals and neighbours of the mesh
//! as the stroke found it, so a dab does not chase what the dab before it did.
//!
//! # Cost
//!
//! A dab touches the vertices under it, found through a grid over the
//! stroke's starting positions rather than by measuring every vertex. And a
//! stroke re-run with more dabs than last time picks up where it stopped: a
//! [`StrokeCache`] holds the result so far, so a growing stroke costs its new
//! dabs, not all of them again on every frame.

use kreate_math::Vec3;

use crate::MeshData;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Brush {
    /// Raises the surface along its normal under the brush; inverted, carves.
    #[default]
    Draw,
    /// Evens the surface out towards its neighbours.
    Smooth,
    /// Drags what is under the brush where the pointer goes.
    Grab,
    /// Swells each vertex along its own normal; inverted, shrinks.
    Inflate,
}

/// One touch of the brush: where, and how wide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dab {
    pub centre: Vec3,
    pub radius: f32,
}

/// What a stroke does besides its dabs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    pub brush: Brush,
    /// How hard, 0 to 1.
    pub strength: f32,
    /// The other way: carve instead of raise, shrink instead of swell.
    pub invert: bool,
    /// For a grab: how far the pointer has dragged, in the mesh's space.
    pub grab: Vec3,
    /// Mirrored across X as well, Blender's X symmetry.
    pub mirror_x: bool,
}

fn falloff(distance: f32, radius: f32) -> f32 {
    if distance >= radius || radius <= 0.0 {
        return 0.0;
    }
    let t = 1.0 - distance / radius;
    t * t * (3.0 - 2.0 * t)
}

fn mirrored(d: Dab) -> Dab {
    Dab {
        centre: Vec3::new(-d.centre.x, d.centre.y, d.centre.z),
        ..d
    }
}

/// Area-weighted vertex normals.
fn normals(mesh: &MeshData) -> Vec<Vec3> {
    let p = mesh.positions();
    let mut out = vec![Vec3::ZERO; p.len()];
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        let mut n = Vec3::ZERO;
        for i in 0..ring.len() {
            n += p[ring[i] as usize].cross(p[ring[(i + 1) % ring.len()] as usize]);
        }
        for v in ring {
            out[*v as usize] += n;
        }
    }
    out.into_iter().map(|n| n.normalize_or_zero()).collect()
}

/// Each vertex's neighbours along edges, packed: vertex `v`'s are
/// `items[start[v]..start[v + 1]]`.
struct Neighbours {
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Neighbours {
    fn new(mesh: &MeshData) -> Self {
        let n = mesh.num_verts() as usize;
        let mut start = vec![0u32; n + 1];
        for [a, b] in mesh.edges() {
            start[*a as usize + 1] += 1;
            start[*b as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; start[n] as usize];
        for [a, b] in mesh.edges() {
            items[fill[*a as usize] as usize] = *b;
            fill[*a as usize] += 1;
            items[fill[*b as usize] as usize] = *a;
            fill[*b as usize] += 1;
        }
        Self { start, items }
    }

    fn of(&self, v: usize) -> &[u32] {
        &self.items[self.start[v] as usize..self.start[v + 1] as usize]
    }
}

/// A uniform grid over points, for finding the ones within a radius without
/// measuring all of them.
struct Grid {
    min: Vec3,
    cell: f32,
    dims: [u32; 3],
    /// Cell `c`'s points are `items[start[c]..start[c + 1]]`.
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Grid {
    /// Cells about `cell` wide — the brush's radius is the natural size —
    /// made coarser when that would mean more cells than about twice the
    /// points.
    fn new(points: &[Vec3], cell: f32) -> Self {
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in points.iter().filter(|p| p.is_finite()) {
            min = min.min(*p);
            max = max.max(*p);
        }
        if min.x > max.x {
            min = Vec3::ZERO;
            max = Vec3::ZERO;
        }
        let extent = max - min;
        let budget = (points.len() as f64 * 2.0).max(64.0);
        let mut cell = if cell.is_finite() && cell > 0.0 {
            cell
        } else {
            extent.max_element()
        };
        cell = cell.max(extent.max_element() / 1024.0).max(1e-6);
        let dims_for =
            |cell: f32| [extent.x, extent.y, extent.z].map(|e| (e / cell).floor() as u32 + 1);
        while dims_for(cell).iter().map(|d| *d as f64).product::<f64>() > budget {
            cell *= 1.5;
        }
        let mut grid = Self {
            min,
            cell,
            dims: dims_for(cell),
            start: Vec::new(),
            items: Vec::new(),
        };
        let cells = grid.dims.iter().map(|d| *d as usize).product::<usize>();
        let slot: Vec<Option<usize>> = points
            .iter()
            .map(|p| p.is_finite().then(|| grid.index(grid.cell_of(*p))))
            .collect();
        let mut start = vec![0u32; cells + 1];
        for c in slot.iter().flatten() {
            start[c + 1] += 1;
        }
        for i in 0..cells {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; start[cells] as usize];
        for (i, c) in slot.iter().enumerate() {
            if let Some(c) = c {
                items[fill[*c] as usize] = i as u32;
                fill[*c] += 1;
            }
        }
        grid.start = start;
        grid.items = items;
        grid
    }

    fn cell_of(&self, p: Vec3) -> [u32; 3] {
        let c = (p - self.min) / self.cell;
        let axis = |v: f32, d: u32| (v.floor().max(0.0) as u32).min(d - 1);
        [
            axis(c.x, self.dims[0]),
            axis(c.y, self.dims[1]),
            axis(c.z, self.dims[2]),
        ]
    }

    fn index(&self, [x, y, z]: [u32; 3]) -> usize {
        (z as usize * self.dims[1] as usize + y as usize) * self.dims[0] as usize + x as usize
    }

    /// Every point within `radius` of `centre`, with its distance.
    fn within(&self, points: &[Vec3], centre: Vec3, radius: f32, mut f: impl FnMut(usize, f32)) {
        if radius.is_nan() || radius <= 0.0 || !centre.is_finite() {
            return;
        }
        let lo = self.cell_of(centre - Vec3::splat(radius));
        let hi = self.cell_of(centre + Vec3::splat(radius));
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let c = self.index([x, y, z]);
                    for &i in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
                        let d = points[i as usize].distance(centre);
                        if d < radius {
                            f(i as usize, d);
                        }
                    }
                }
            }
        }
    }
}

/// A stroke part-way through: the mesh as it found it, what it has done so
/// far, and the tables every dab reads.
///
/// [`StrokeCache::fits`] says whether a call can pick up from here — the same
/// mesh, the same brush settings, and the dabs so far at the front of the new
/// list — and [`StrokeCache::run`] then applies only what is new. The answer
/// is the one [`apply`] gives from scratch.
pub struct StrokeCache {
    original: Vec<Vec3>,
    corners: Vec<u32>,
    normal: Vec<Vec3>,
    /// Only a smoothing stroke needs it.
    near: Option<Neighbours>,
    grid: Grid,
    current: Vec<Vec3>,
    /// The stroke settings, its grab left out: a grab is re-done from the
    /// start every time anyway.
    key: Stroke,
    applied: Vec<Dab>,
    moved: bool,
}

impl StrokeCache {
    /// Starts a stroke on `mesh`. `radius` sizes the search grid; the brush's
    /// is right.
    pub fn new(mesh: &MeshData, stroke: &Stroke, radius: f32) -> Self {
        let original = mesh.positions().to_vec();
        Self {
            grid: Grid::new(&original, radius),
            normal: normals(mesh),
            corners: mesh.corner_verts().to_vec(),
            near: (stroke.brush == Brush::Smooth).then(|| Neighbours::new(mesh)),
            current: original.clone(),
            original,
            key: Self::key(stroke),
            applied: Vec::new(),
            moved: false,
        }
    }

    fn key(stroke: &Stroke) -> Stroke {
        Stroke {
            grab: Vec3::ZERO,
            ..*stroke
        }
    }

    /// Whether `dabs` on `mesh` with `stroke` continues what this has done.
    pub fn fits(&self, mesh: &MeshData, stroke: &Stroke, dabs: &[Dab]) -> bool {
        self.key == Self::key(stroke)
            && dabs.starts_with(&self.applied)
            && mesh.corner_verts() == self.corners.as_slice()
            && mesh.positions() == self.original.as_slice()
    }

    /// Applies what of `dabs` is new and writes the result into `mesh`, which
    /// must be the mesh this began on (see [`StrokeCache::fits`]). `false`
    /// when nothing was under any dab.
    pub fn run(&mut self, mesh: &mut MeshData, dabs: &[Dab], stroke: &Stroke) -> bool {
        if dabs.is_empty() || self.original.is_empty() {
            return false;
        }
        if stroke.brush == Brush::Grab {
            self.grab(dabs[0], stroke);
        } else {
            let start = self.applied.len().min(dabs.len());
            for d in &dabs[start..] {
                self.dab(*d, stroke);
                if stroke.mirror_x {
                    self.dab(mirrored(*d), stroke);
                }
            }
            self.applied = dabs.to_vec();
        }
        if self.moved {
            mesh.positions_mut().copy_from_slice(&self.current);
        }
        self.moved
    }

    /// One grab: what was under the first dab, carried by the drag.
    fn grab(&mut self, first: Dab, stroke: &Stroke) {
        self.current.copy_from_slice(&self.original);
        self.moved = false;
        let mut grabs = vec![(first, stroke.grab)];
        if stroke.mirror_x {
            let g = stroke.grab;
            grabs.push((mirrored(first), Vec3::new(-g.x, g.y, g.z)));
        }
        for (dab, delta) in grabs {
            let (current, moved) = (&mut self.current, &mut self.moved);
            self.grid
                .within(&self.original, dab.centre, dab.radius, |i, d| {
                    let w = falloff(d, dab.radius);
                    if w > 0.0 {
                        current[i] += delta * w;
                        *moved = true;
                    }
                });
        }
    }

    fn dab(&mut self, dab: Dab, stroke: &Stroke) {
        let strength = stroke.strength.clamp(0.0, 1.0);
        let sign = if stroke.invert { -1.0 } else { 1.0 };
        // Who is under the brush, and how much.
        let mut under: Vec<(usize, f32)> = Vec::new();
        self.grid
            .within(&self.original, dab.centre, dab.radius, |i, d| {
                let w = falloff(d, dab.radius);
                if w > 0.0 {
                    under.push((i, w));
                }
            });
        if under.is_empty() {
            return;
        }
        match stroke.brush {
            Brush::Grab => {}
            Brush::Draw => {
                // The surface's normal under the brush, weighted as the brush is.
                let area = under
                    .iter()
                    .fold(Vec3::ZERO, |s, (i, w)| s + self.normal[*i] * *w)
                    .normalize_or_zero();
                let step = dab.radius * 0.15 * strength * sign;
                for (i, w) in &under {
                    self.current[*i] += area * step * *w;
                }
                self.moved = true;
            }
            Brush::Inflate => {
                let step = dab.radius * 0.1 * strength * sign;
                for (i, w) in &under {
                    self.current[*i] += self.normal[*i] * step * *w;
                }
                self.moved = true;
            }
            Brush::Smooth => {
                let Some(near) = self.near.as_ref() else {
                    return;
                };
                // Read everything before writing anything: each vertex moves
                // towards where its neighbours were before this dab.
                let next: Vec<(usize, Vec3)> = under
                    .iter()
                    .filter_map(|(i, w)| {
                        let w = w * strength;
                        let n = near.of(*i);
                        if w <= 0.0 || n.is_empty() {
                            return None;
                        }
                        let average = n
                            .iter()
                            .fold(Vec3::ZERO, |s, k| s + self.current[*k as usize])
                            / n.len() as f32;
                        Some((*i, self.current[*i].lerp(average, w)))
                    })
                    .collect();
                if !next.is_empty() {
                    self.moved = true;
                }
                for (i, p) in next {
                    self.current[i] = p;
                }
            }
        }
    }
}

/// Applies a stroke to `mesh`. `false` when nothing was under any dab.
pub fn apply(mesh: &mut MeshData, dabs: &[Dab], stroke: &Stroke) -> bool {
    if dabs.is_empty() || mesh.num_verts() == 0 {
        return false;
    }
    StrokeCache::new(mesh, stroke, dabs[0].radius).run(mesh, dabs, stroke)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn grid() -> MeshData {
        primitives::grid(4.0, 16, 16)
    }

    fn stroke(brush: Brush) -> Stroke {
        Stroke {
            brush,
            strength: 1.0,
            invert: false,
            grab: Vec3::ZERO,
            mirror_x: false,
        }
    }

    fn top(m: &MeshData) -> f32 {
        m.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max)
    }

    #[test]
    fn draw_raises_under_the_brush_and_nowhere_else_and_inverted_carves() {
        let mut m = grid();
        let dab = Dab {
            centre: Vec3::ZERO,
            radius: 1.0,
        };
        assert!(apply(&mut m, &[dab], &stroke(Brush::Draw)));
        assert!(top(&m) > 0.1, "{}", top(&m));
        // Outside the radius, untouched.
        assert!(m
            .positions()
            .iter()
            .filter(|p| p.x.abs() > 1.1 || p.y.abs() > 1.1)
            .all(|p| p.z == 0.0));
        let mut m = grid();
        apply(
            &mut m,
            &[dab],
            &Stroke {
                invert: true,
                ..stroke(Brush::Draw)
            },
        );
        assert!(m.positions().iter().map(|p| p.z).fold(f32::MAX, f32::min) < -0.1);
    }

    #[test]
    fn smooth_takes_a_spike_down() {
        let mut m = grid();
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::ZERO,
                radius: 0.3,
            }],
            &stroke(Brush::Draw),
        );
        let spike = top(&m);
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::ZERO,
                radius: 1.0,
            }; 4],
            &stroke(Brush::Smooth),
        );
        assert!(top(&m) < spike * 0.7, "{} from {spike}", top(&m));
    }

    #[test]
    fn grab_carries_what_was_under_the_first_dab() {
        let mut m = grid();
        let s = Stroke {
            grab: Vec3::new(0.0, 0.0, 2.0),
            ..stroke(Brush::Grab)
        };
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::ZERO,
                radius: 1.0,
            }],
            &s,
        );
        assert!((top(&m) - 2.0).abs() < 0.05, "{}", top(&m));
    }

    #[test]
    fn x_symmetry_does_the_same_on_the_other_side() {
        let mut m = grid();
        let s = Stroke {
            mirror_x: true,
            ..stroke(Brush::Draw)
        };
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::new(1.0, 0.0, 0.0),
                radius: 0.5,
            }],
            &s,
        );
        let at = |x: f32| {
            m.positions()
                .iter()
                .find(|p| (p.x - x).abs() < 1e-4 && p.y.abs() < 1e-4)
                .unwrap()
                .z
        };
        assert!(at(1.0) > 0.0);
        assert!(
            (at(1.0) - at(-1.0)).abs() < 1e-5,
            "{} vs {}",
            at(1.0),
            at(-1.0)
        );
    }

    /// A stroke picked up from its cache, a dab or two at a time, lands
    /// exactly where the same stroke run from scratch does — for every brush,
    /// mirrored or not.
    #[test]
    fn a_stroke_continued_from_its_cache_matches_one_run_from_scratch() {
        let base = primitives::uv_sphere(1.0, 48, 24);
        let dabs: Vec<Dab> = (0..9)
            .map(|i| Dab {
                centre: Vec3::new((i as f32 * 0.3).cos(), (i as f32 * 0.3).sin(), 0.2),
                radius: 0.35,
            })
            .collect();
        for brush in [Brush::Draw, Brush::Smooth, Brush::Inflate, Brush::Grab] {
            for mirror_x in [false, true] {
                let s = Stroke {
                    mirror_x,
                    grab: Vec3::new(0.1, 0.0, 0.3),
                    ..stroke(brush)
                };
                let mut whole = base.clone();
                apply(&mut whole, &dabs, &s);

                let mut cache = StrokeCache::new(&base, &s, dabs[0].radius);
                let mut last = base.clone();
                for n in [1, 2, 5, 9] {
                    // As the operator sees it: the starting mesh every time.
                    let mut m = base.clone();
                    assert!(cache.fits(&m, &s, &dabs[..n]), "{brush:?} {n}");
                    cache.run(&mut m, &dabs[..n], &s);
                    last = m;
                }
                assert_eq!(
                    last.positions(),
                    whole.positions(),
                    "{brush:?} mirror {mirror_x}"
                );
            }
        }
    }

    #[test]
    fn a_cache_does_not_fit_another_mesh_or_other_settings_or_fewer_dabs() {
        let base = grid();
        let s = stroke(Brush::Draw);
        let dabs = [
            Dab {
                centre: Vec3::ZERO,
                radius: 1.0,
            },
            Dab {
                centre: Vec3::X,
                radius: 1.0,
            },
        ];
        let mut cache = StrokeCache::new(&base, &s, 1.0);
        let mut m = base.clone();
        cache.run(&mut m, &dabs, &s);
        assert!(!cache.fits(&m, &s, &dabs), "the mesh it already changed");
        assert!(!cache.fits(&base, &s, &dabs[..1]), "fewer dabs");
        let harder = Stroke { strength: 0.5, ..s };
        assert!(!cache.fits(&base, &harder, &dabs));
        assert!(cache.fits(&base, &s, &dabs));
    }

    /// The grid finds exactly what measuring every vertex finds.
    #[test]
    fn the_grid_finds_what_a_full_scan_finds() {
        let m = primitives::uv_sphere(2.0, 40, 20);
        let p = m.positions();
        for cell in [0.05, 0.3, 5.0] {
            let g = Grid::new(p, cell);
            for (c, r) in [
                (Vec3::ZERO, 2.5),
                (Vec3::new(2.0, 0.0, 0.0), 0.4),
                (Vec3::new(9.0, 9.0, 9.0), 1.0),
                (Vec3::new(0.0, -2.0, 0.1), 0.05),
            ] {
                let mut found = Vec::new();
                g.within(p, c, r, |i, _| found.push(i));
                found.sort_unstable();
                let all: Vec<usize> = (0..p.len()).filter(|i| p[*i].distance(c) < r).collect();
                assert_eq!(found, all, "cell {cell} centre {c} radius {r}");
            }
        }
    }

    #[test]
    fn inflate_swells_a_sphere() {
        let mut m = primitives::uv_sphere(1.0, 16, 8);
        let r0 = m.positions().iter().map(|p| p.length()).fold(0.0, f32::max);
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::Z,
                radius: 0.8,
            }],
            &stroke(Brush::Inflate),
        );
        let r1 = m.positions().iter().map(|p| p.length()).fold(0.0, f32::max);
        assert!(r1 > r0 + 0.01);
    }
}
