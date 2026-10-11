//! Liquid: FLIP on a grid, as Blender's Mantaflow liquid is.
//!
//! The liquid is particles, and its motion is solved on a grid. Each step
//! the particles' velocities are spread onto the faces of the grid's cells
//! (a MAC grid: each face holds the speed across it), gravity is added, and
//! the pressure that makes the flow lose no volume is solved for and taken
//! off. The particles then take back the grid's *change* — FLIP, lively —
//! blended with a little of its value — PIC, steady — and move through it.
//! A particle carrying its own velocity between steps is what keeps a splash
//! a splash; the grid alone smears it away.
//!
//! What fills the domain at the start, what keeps pouring in and what
//! drains are meshes, made into cells by winding (`kreate_mesh::remesh`).
//! Obstacles — objects with Collision — are cells the liquid cannot enter,
//! made again every frame so an animated one sweeps through it. The surface
//! drawn is the particles' density, smoothed, made a mesh by surface nets.
//!
//! In the world, along the world's axes: the domain is the box its object's
//! mesh fills.

use kreate_math::{Aabb, DAffine3, Vec3};
use kreate_mesh::remesh::{surface_nets, winding, Field};
use kreate_mesh::MeshData;

use super::Rng;

/// The most cells along a domain's longest side.
pub const MAX_RESOLUTION: u32 = 128;

/// Particles per cell along each axis when a cell is filled: eight a cell.
const SEED: usize = 2;

#[derive(Clone, Copy, Debug)]
struct Drop {
    pos: Vec3,
    vel: Vec3,
}

/// A mesh made into cells of the grid, and what it does there.
#[derive(Clone, Debug)]
pub struct Region {
    cells: Vec<bool>,
    /// The speed liquid poured from it starts with, metres a second.
    velocity: Vec3,
}

/// A liquid being simulated.
#[derive(Clone, Debug)]
pub struct LiquidSim {
    /// Cells along X, Y and Z.
    n: [usize; 3],
    /// A cell's side, metres.
    h: f32,
    /// The domain's low corner in the world.
    origin: Vec3,
    u: Vec<f32>,
    v: Vec<f32>,
    w: Vec<f32>,
    solid: Vec<bool>,
    drops: Vec<Drop>,
    /// 0 is PIC, 1 is FLIP.
    flip: f32,
    inflows: Vec<Region>,
    outflows: Vec<Vec<bool>>,
    rng: Rng,
}

/// Fluid cell, air, or solid: what a cell holds this step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Air,
    Liquid,
    Solid,
}

/// The triangles of `mesh` placed in the world by `world`.
pub fn world_triangles(mesh: &MeshData, world: DAffine3) -> Vec<[Vec3; 3]> {
    let tri = mesh.triangulate();
    let (tris, _) = tri.indices.as_chunks::<3>();
    let p: Vec<Vec3> = mesh
        .positions()
        .iter()
        .map(|p| world.transform_point3(p.as_dvec3()).as_vec3())
        .collect();
    tris.iter()
        .map(|t| t.map(|i| p[i as usize]))
        .filter(|t| t.iter().all(|v| v.is_finite()))
        .collect()
}

impl LiquidSim {
    /// An empty domain filling `bounds`, `resolution` cells along its
    /// longest side.
    pub fn new(bounds: Aabb, resolution: u32, flip: f64, seed: u64) -> Self {
        let size = (bounds.max - bounds.min).max(Vec3::splat(1e-3));
        let resolution = resolution.clamp(4, MAX_RESOLUTION) as f32;
        let h = size.max_element() / resolution;
        let n = [size.x, size.y, size.z].map(|s| ((s / h).round() as usize).max(1));
        let cells = n[0] * n[1] * n[2];
        Self {
            n,
            h,
            origin: bounds.min,
            u: vec![0.0; (n[0] + 1) * n[1] * n[2]],
            v: vec![0.0; n[0] * (n[1] + 1) * n[2]],
            w: vec![0.0; n[0] * n[1] * (n[2] + 1)],
            solid: vec![false; cells],
            drops: Vec::new(),
            flip: flip.clamp(0.0, 1.0) as f32,
            inflows: Vec::new(),
            outflows: Vec::new(),
            rng: Rng::new(seed),
        }
    }

    pub fn cells(&self) -> [usize; 3] {
        self.n
    }

    pub fn cell_size(&self) -> f32 {
        self.h
    }

    pub fn particle_count(&self) -> usize {
        self.drops.len()
    }

    /// Where the particles are, in the world.
    pub fn particles(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.drops.iter().map(|d| d.pos)
    }

    fn cell(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.n[1] + j) * self.n[0] + i
    }

    /// Which cells of the grid a closed surface fills: by their middles.
    pub fn cells_inside(&self, tris: &[[Vec3; 3]]) -> Vec<bool> {
        if tris.is_empty() {
            return vec![false; self.n.iter().product()];
        }
        winding(
            tris,
            self.origin + Vec3::splat(self.h * 0.5),
            self.h,
            self.n,
        )
    }

    /// Fills what `tris` encloses with liquid moving at `velocity`.
    pub fn fill(&mut self, tris: &[[Vec3; 3]], velocity: Vec3) {
        let inside = self.cells_inside(tris);
        for k in 0..self.n[2] {
            for j in 0..self.n[1] {
                for i in 0..self.n[0] {
                    let c = self.cell(i, j, k);
                    if inside[c] && !self.solid[c] {
                        self.seed_cell(i, j, k, velocity);
                    }
                }
            }
        }
    }

    /// Something that keeps pouring liquid in where `tris` encloses.
    pub fn add_inflow(&mut self, tris: &[[Vec3; 3]], velocity: Vec3) {
        let cells = self.cells_inside(tris);
        self.inflows.push(Region { cells, velocity });
    }

    /// Something that drains whatever liquid enters what `tris` encloses.
    pub fn add_outflow(&mut self, tris: &[[Vec3; 3]]) {
        let cells = self.cells_inside(tris);
        self.outflows.push(cells);
    }

    /// The cells obstacles fill this frame. Liquid in one is pushed out.
    pub fn set_obstacles(&mut self, obstacles: &[Vec<[Vec3; 3]>]) {
        let mut solid = vec![false; self.solid.len()];
        for tris in obstacles {
            for (s, inside) in solid.iter_mut().zip(self.cells_inside(tris)) {
                *s |= inside;
            }
        }
        self.solid = solid;
    }

    fn seed_cell(&mut self, i: usize, j: usize, k: usize, velocity: Vec3) {
        let step = self.h / SEED as f32;
        for c in 0..SEED * SEED * SEED {
            let (a, b, d) = (c % SEED, (c / SEED) % SEED, c / (SEED * SEED));
            let jitter = Vec3::new(
                self.rng.unit() as f32,
                self.rng.unit() as f32,
                self.rng.unit() as f32,
            );
            let at = Vec3::new(
                (i * SEED + a) as f32,
                (j * SEED + b) as f32,
                (k * SEED + d) as f32,
            );
            self.drops.push(Drop {
                pos: self.origin + (at + jitter) * step,
                vel: velocity,
            });
        }
    }

    /// The cell a world point is in, if it is in the domain.
    fn cell_of(&self, p: Vec3) -> Option<(usize, usize, usize)> {
        let g = (p - self.origin) / self.h;
        if g.min_element() < 0.0 {
            return None;
        }
        let (i, j, k) = (g.x as usize, g.y as usize, g.z as usize);
        (i < self.n[0] && j < self.n[1] && k < self.n[2]).then_some((i, j, k))
    }

    /// Advances `seconds` under `gravity`, in steps short enough that
    /// nothing crosses more than a cell in one.
    pub fn advance(&mut self, seconds: f32, gravity: Vec3) {
        let mut left = seconds;
        let mut steps = 0;
        while left > 1e-6 && steps < 64 {
            let fastest = self
                .drops
                .iter()
                .map(|d| d.vel.length())
                .fold(0.0f32, f32::max)
                + (gravity.length() * left).sqrt();
            let dt = (self.h / fastest.max(1e-3)).min(left);
            // The last steps of a frame evened out, so a frame does not end
            // on a sliver.
            let dt = if dt < left && dt > left * 0.5 {
                left * 0.5
            } else {
                dt
            };
            self.step(dt, gravity);
            left -= dt;
            steps += 1;
        }
    }

    fn step(&mut self, dt: f32, gravity: Vec3) {
        self.pour();
        let kinds = self.classify();
        let (mut u, mut v, mut w) = self.to_grid();
        let (u0, v0, w0) = (u.clone(), v.clone(), w.clone());
        for x in u.iter_mut() {
            *x += gravity.x * dt;
        }
        for x in v.iter_mut() {
            *x += gravity.y * dt;
        }
        for x in w.iter_mut() {
            *x += gravity.z * dt;
        }
        self.walls(&kinds, &mut u, &mut v, &mut w);
        self.project(&kinds, &mut u, &mut v, &mut w);
        self.extrapolate(&kinds, &mut u, &mut v, &mut w);
        self.walls(&kinds, &mut u, &mut v, &mut w);
        // FLIP: each particle takes the grid's change; PIC: its value.
        let du: Vec<f32> = u.iter().zip(&u0).map(|(a, b)| a - b).collect();
        let dv: Vec<f32> = v.iter().zip(&v0).map(|(a, b)| a - b).collect();
        let dw: Vec<f32> = w.iter().zip(&w0).map(|(a, b)| a - b).collect();
        let flip = self.flip;
        let mut drops = std::mem::take(&mut self.drops);
        for d in drops.iter_mut() {
            let pic = self.sample(&u, &v, &w, d.pos);
            let change = self.sample(&du, &dv, &dw, d.pos);
            d.vel = pic * (1.0 - flip) + (d.vel + change) * flip;
        }
        self.u = u;
        self.v = v;
        self.w = w;
        // Through the new flow, midpoint first.
        for d in drops.iter_mut() {
            let mid = d.pos + self.velocity(d.pos) * (0.5 * dt);
            let to = d.pos + self.velocity(mid) * dt;
            d.pos = self.keep_out(d.pos, to, &mut d.vel);
        }
        drops.retain(|d| {
            self.cell_of(d.pos).is_none_or(|(i, j, k)| {
                let c = self.cell(i, j, k);
                !self.outflows.iter().any(|o| o[c])
            })
        });
        self.drops = drops;
    }

    /// Inflows topped up: a cell of one that has emptied is filled again.
    fn pour(&mut self) {
        if self.inflows.is_empty() {
            return;
        }
        let mut count = vec![0u32; self.solid.len()];
        for d in &self.drops {
            if let Some((i, j, k)) = self.cell_of(d.pos) {
                count[self.cell(i, j, k)] += 1;
            }
        }
        let inflows = std::mem::take(&mut self.inflows);
        for region in &inflows {
            for k in 0..self.n[2] {
                for j in 0..self.n[1] {
                    for i in 0..self.n[0] {
                        let c = self.cell(i, j, k);
                        if region.cells[c] && !self.solid[c] && count[c] < 2 {
                            self.seed_cell(i, j, k, region.velocity);
                            count[c] += (SEED * SEED * SEED) as u32;
                        }
                    }
                }
            }
            // And what is in it keeps the inflow's speed: a hose, not a
            // puddle that happens to be in the hose.
            for d in self.drops.iter_mut() {
                let g = (d.pos - self.origin) / self.h;
                if g.min_element() < 0.0 {
                    continue;
                }
                let (i, j, k) = (g.x as usize, g.y as usize, g.z as usize);
                if i < self.n[0] && j < self.n[1] && k < self.n[2] {
                    let c = (k * self.n[1] + j) * self.n[0] + i;
                    if region.cells[c] {
                        d.vel = region.velocity;
                    }
                }
            }
        }
        self.inflows = inflows;
    }

    fn classify(&self) -> Vec<Cell> {
        let mut kinds: Vec<Cell> = self
            .solid
            .iter()
            .map(|s| if *s { Cell::Solid } else { Cell::Air })
            .collect();
        for d in &self.drops {
            if let Some((i, j, k)) = self.cell_of(d.pos) {
                let c = self.cell(i, j, k);
                if kinds[c] == Cell::Air {
                    kinds[c] = Cell::Liquid;
                }
            }
        }
        kinds
    }

    /// The particles' velocities spread onto the faces they are near.
    fn to_grid(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.n;
        let mut out = [
            (vec![0.0f32; self.u.len()], vec![0.0f32; self.u.len()]),
            (vec![0.0f32; self.v.len()], vec![0.0f32; self.v.len()]),
            (vec![0.0f32; self.w.len()], vec![0.0f32; self.w.len()]),
        ];
        for d in &self.drops {
            let g = (d.pos - self.origin) / self.h;
            for (axis, (sum, weight)) in out.iter_mut().enumerate() {
                let dims = face_dims(n, axis);
                // A face of this axis sits at the cell's low side on it and
                // the middle of the other two.
                let mut q = g - Vec3::splat(0.5);
                q[axis] += 0.5;
                let speed = d.vel[axis];
                for_trilinear(q, dims, |at, wgt| {
                    sum[at] += speed * wgt;
                    weight[at] += wgt;
                });
            }
        }
        let [(su, wu), (sv, wv), (sw, ww)] = out;
        let finish = |s: Vec<f32>, w: Vec<f32>| -> Vec<f32> {
            s.into_iter()
                .zip(w)
                .map(|(s, w)| if w > 1e-6 { s / w } else { 0.0 })
                .collect()
        };
        (finish(su, wu), finish(sv, wv), finish(sw, ww))
    }

    /// No flow through the domain's walls or into an obstacle.
    fn walls(&self, kinds: &[Cell], u: &mut [f32], v: &mut [f32], w: &mut [f32]) {
        let n = self.n;
        for (axis, field) in [&mut *u, &mut *v, &mut *w].into_iter().enumerate() {
            let dims = face_dims(n, axis);
            for k in 0..dims[2] {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        let at = [i, j, k];
                        let side = at[axis];
                        let blocked = side == 0 || side == n[axis] || {
                            let mut lo = at;
                            lo[axis] -= 1;
                            kinds[self.cell(lo[0], lo[1], lo[2])] == Cell::Solid
                                || kinds[self.cell(i, j, k)] == Cell::Solid
                        };
                        if blocked {
                            field[(k * dims[1] + j) * dims[0] + i] = 0.0;
                        }
                    }
                }
            }
        }
    }

    /// Takes off the pressure that would squeeze or stretch the liquid:
    /// afterwards as much leaves each liquid cell as enters it.
    fn project(&self, kinds: &[Cell], u: &mut [f32], v: &mut [f32], w: &mut [f32]) {
        let n = self.n;
        let cells = kinds.len();
        // The unknowns: one pressure per liquid cell.
        let mut index = vec![usize::MAX; cells];
        let mut liquid = Vec::new();
        for (c, kind) in kinds.iter().enumerate() {
            if *kind == Cell::Liquid {
                index[c] = liquid.len();
                liquid.push(c);
            }
        }
        if liquid.is_empty() {
            return;
        }
        let at = |c: usize| (c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1]));
        let uix = |i: usize, j: usize, k: usize| (k * n[1] + j) * (n[0] + 1) + i;
        let vix = |i: usize, j: usize, k: usize| (k * (n[1] + 1) + j) * n[0] + i;
        let wix = |i: usize, j: usize, k: usize| (k * n[1] + j) * n[0] + i;
        // Each liquid cell's neighbours that are not solid (or wall): the
        // diagonal, and the liquid ones among them, the rest of the row.
        let mut diag = vec![0.0f32; liquid.len()];
        let mut nbrs: Vec<[usize; 6]> = vec![[usize::MAX; 6]; liquid.len()];
        let mut rhs = vec![0.0f32; liquid.len()];
        for (r, &c) in liquid.iter().enumerate() {
            let (i, j, k) = at(c);
            let div = u[uix(i + 1, j, k)] - u[uix(i, j, k)] + v[vix(i, j + 1, k)] - v[vix(i, j, k)]
                + w[wix(i, j, k + 1)]
                - w[wix(i, j, k)];
            rhs[r] = -div;
            let around = [
                (i > 0).then(|| (i - 1, j, k)),
                (i + 1 < n[0]).then(|| (i + 1, j, k)),
                (j > 0).then(|| (i, j - 1, k)),
                (j + 1 < n[1]).then(|| (i, j + 1, k)),
                (k > 0).then(|| (i, j, k - 1)),
                (k + 1 < n[2]).then(|| (i, j, k + 1)),
            ];
            for (s, nb) in around.into_iter().enumerate() {
                let Some((a, b, d)) = nb else { continue };
                let nc = self.cell(a, b, d);
                match kinds[nc] {
                    Cell::Solid => {}
                    Cell::Air => diag[r] += 1.0,
                    Cell::Liquid => {
                        diag[r] += 1.0;
                        nbrs[r][s] = index[nc];
                    }
                }
            }
        }
        let apply = |x: &[f32], out: &mut [f32]| {
            for r in 0..x.len() {
                let mut s = diag[r] * x[r];
                for nb in nbrs[r] {
                    if nb != usize::MAX {
                        s -= x[nb];
                    }
                }
                out[r] = s;
            }
        };
        let q = conjugate_gradient(&apply, &diag, &rhs, 400, 1e-5);
        let pressure = |c: usize| -> f32 {
            match kinds[c] {
                Cell::Liquid => q[index[c]],
                _ => 0.0,
            }
        };
        // Each face between two cells, at least one liquid, neither solid.
        for (axis, field) in [&mut *u, &mut *v, &mut *w].into_iter().enumerate() {
            let dims = face_dims(n, axis);
            for k in 0..dims[2] {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        let hi = [i, j, k];
                        if hi[axis] == 0 || hi[axis] == n[axis] {
                            continue;
                        }
                        let mut lo = hi;
                        lo[axis] -= 1;
                        let (a, b) = (self.cell(lo[0], lo[1], lo[2]), self.cell(i, j, k));
                        if kinds[a] == Cell::Solid || kinds[b] == Cell::Solid {
                            continue;
                        }
                        if kinds[a] != Cell::Liquid && kinds[b] != Cell::Liquid {
                            continue;
                        }
                        field[(k * dims[1] + j) * dims[0] + i] -= pressure(b) - pressure(a);
                    }
                }
            }
        }
    }

    /// Velocities carried out from the liquid into the air beside it, so a
    /// particle near the surface reads the liquid's flow rather than still
    /// air.
    fn extrapolate(&self, kinds: &[Cell], u: &mut [f32], v: &mut [f32], w: &mut [f32]) {
        let n = self.n;
        for (axis, field) in [&mut *u, &mut *v, &mut *w].into_iter().enumerate() {
            let dims = face_dims(n, axis);
            let fix = |i: usize, j: usize, k: usize| (k * dims[1] + j) * dims[0] + i;
            let mut known = vec![false; field.len()];
            for k in 0..dims[2] {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        let hi = [i, j, k];
                        let mut lo = hi;
                        let liquid_hi =
                            hi[axis] < n[axis] && kinds[self.cell(i, j, k)] == Cell::Liquid;
                        let liquid_lo = hi[axis] > 0 && {
                            lo[axis] -= 1;
                            kinds[self.cell(lo[0], lo[1], lo[2])] == Cell::Liquid
                        };
                        known[fix(i, j, k)] = liquid_hi || liquid_lo;
                    }
                }
            }
            for _ in 0..3 {
                let before = known.clone();
                let values = field.to_vec();
                for k in 0..dims[2] {
                    for j in 0..dims[1] {
                        for i in 0..dims[0] {
                            let at = fix(i, j, k);
                            if before[at] {
                                continue;
                            }
                            let (mut sum, mut count) = (0.0, 0.0);
                            let p = [i as isize, j as isize, k as isize];
                            for (da, sign) in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)] {
                                let mut q = p;
                                q[da] += sign;
                                if q[da] < 0 || q[da] >= dims[da] as isize {
                                    continue;
                                }
                                let nb = fix(q[0] as usize, q[1] as usize, q[2] as usize);
                                if before[nb] {
                                    sum += values[nb];
                                    count += 1.0;
                                }
                            }
                            if count > 0.0 {
                                field[at] = sum / count;
                                known[at] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    fn sample(&self, u: &[f32], v: &[f32], w: &[f32], p: Vec3) -> Vec3 {
        let g = (p - self.origin) / self.h;
        let mut out = Vec3::ZERO;
        for (axis, field) in [u, v, w].into_iter().enumerate() {
            let dims = face_dims(self.n, axis);
            let mut q = g - Vec3::splat(0.5);
            q[axis] += 0.5;
            let mut s = 0.0;
            for_trilinear(q, dims, |at, wgt| s += field[at] * wgt);
            out[axis] = s;
        }
        out
    }

    fn velocity(&self, p: Vec3) -> Vec3 {
        self.sample(&self.u, &self.v, &self.w, p)
    }

    /// Where a particle going from `from` to `to` ends up: inside the
    /// domain, out of the obstacles.
    fn keep_out(&self, from: Vec3, to: Vec3, vel: &mut Vec3) -> Vec3 {
        let margin = self.h * 0.01;
        let lo = self.origin + Vec3::splat(margin);
        let hi = self.origin
            + Vec3::new(self.n[0] as f32, self.n[1] as f32, self.n[2] as f32) * self.h
            - Vec3::splat(margin);
        let mut p = to;
        for a in 0..3 {
            if p[a] < lo[a] {
                p[a] = lo[a];
                vel[a] = vel[a].max(0.0);
            } else if p[a] > hi[a] {
                p[a] = hi[a];
                vel[a] = vel[a].min(0.0);
            }
        }
        let solid = |q: Vec3| {
            self.cell_of(q)
                .is_some_and(|(i, j, k)| self.solid[self.cell(i, j, k)])
        };
        if solid(p) {
            // Back where it came from, if that is clear; else out the
            // nearest open side of its cell.
            if !solid(from) {
                *vel = Vec3::ZERO;
                return from;
            }
            if let Some((i, j, k)) = self.cell_of(p) {
                let centre = self.origin
                    + (Vec3::new(i as f32, j as f32, k as f32) + Vec3::splat(0.5)) * self.h;
                for (axis, sign) in [
                    (2, 1.0),
                    (0, -1.0),
                    (0, 1.0),
                    (1, -1.0),
                    (1, 1.0),
                    (2, -1.0),
                ] {
                    let mut q = centre;
                    q[axis] += sign * self.h;
                    if !solid(q) && self.cell_of(q).is_some() {
                        *vel = Vec3::ZERO;
                        return q;
                    }
                }
            }
        }
        p
    }

    /// The liquid's surface in the world: the particles' density, smoothed,
    /// at half the cell size. `None` when there is no liquid.
    pub fn surface(&self) -> Option<MeshData> {
        if self.drops.is_empty() {
            return None;
        }
        let s = self.h * 0.5;
        // A point of room past each wall, so the surface closes there.
        let dims = self.n.map(|c| c * 2 + 3);
        let origin = self.origin - Vec3::splat(s);
        let mut density = vec![0.0f32; dims.iter().product()];
        let index = |i: usize, j: usize, k: usize| (k * dims[1] + j) * dims[0] + i;
        let reach = 2.0 * s;
        let kernel = |d2: f32| {
            let x = 1.0 - d2 / (reach * reach);
            if x > 0.0 {
                x * x * x
            } else {
                0.0
            }
        };
        for d in &self.drops {
            let g = (d.pos - origin) / s;
            let lo = (g - Vec3::splat(2.0)).ceil().max(Vec3::splat(1.0));
            let hi = (g + Vec3::splat(2.0)).floor().min(Vec3::new(
                dims[0] as f32 - 2.0,
                dims[1] as f32 - 2.0,
                dims[2] as f32 - 2.0,
            ));
            for k in lo.z as usize..=hi.z.max(0.0) as usize {
                for j in lo.y as usize..=hi.y.max(0.0) as usize {
                    for i in lo.x as usize..=hi.x.max(0.0) as usize {
                        let q = origin + Vec3::new(i as f32, j as f32, k as f32) * s;
                        density[index(i, j, k)] += kernel(q.distance_squared(d.pos));
                    }
                }
            }
        }
        // What a point deep in the liquid reads: particles `h / SEED` apart
        // all round it, which is `s` here. Half of it is the surface.
        let full = {
            let mut sum = 0.0;
            for k in -3i32..=3 {
                for j in -3i32..=3 {
                    for i in -3i32..=3 {
                        let q = Vec3::new(i as f32, j as f32, k as f32) * s;
                        sum += kernel(q.length_squared());
                    }
                }
            }
            sum
        };
        // Two passes of smoothing, so single particles do not stand as
        // beads and a flat pool is flat.
        let mut smooth = density;
        for _ in 0..2 {
            let from = smooth.clone();
            for k in 1..dims[2] - 1 {
                for j in 1..dims[1] - 1 {
                    for i in 1..dims[0] - 1 {
                        let c = from[index(i, j, k)];
                        let around = from[index(i - 1, j, k)]
                            + from[index(i + 1, j, k)]
                            + from[index(i, j - 1, k)]
                            + from[index(i, j + 1, k)]
                            + from[index(i, j, k - 1)]
                            + from[index(i, j, k + 1)];
                        smooth[index(i, j, k)] = 0.5 * c + around / 12.0;
                    }
                }
            }
        }
        // A third of full, not half: a sheet one or two particles thick —
        // what a splash thins to — reads as a sheet rather than as lace. A
        // full pool's edge moves out by a fraction of a cell for it.
        let iso = 0.33 * full;
        let field = Field {
            origin,
            h: s,
            dims,
            values: smooth.into_iter().map(|d| iso - d).collect(),
        };
        let mut mesh = surface_nets(&field)?;
        taubin(&mut mesh, 4);
        // Shaded smooth: water has no facets.
        let faces = mesh.num_faces() as usize;
        mesh.attributes_mut().insert(
            kreate_mesh::attribute::names::SHADE_SMOOTH,
            kreate_mesh::attribute::Domain::Face,
            kreate_mesh::attribute::AttrData::Bool(std::sync::Arc::new(vec![true; faces])),
        );
        Some(mesh)
    }
}

/// Taubin's smoothing: each vertex towards its neighbours' middle, then a
/// little further back out, `rounds` times. The steps a grid leaves on a
/// surface go; the volume, which plain averaging shrinks, stays.
fn taubin(mesh: &mut MeshData, rounds: usize) {
    let n = mesh.num_verts() as usize;
    let mut around: Vec<Vec<u32>> = vec![Vec::new(); n];
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        for k in 0..ring.len() {
            let (a, b) = (ring[k], ring[(k + 1) % ring.len()]);
            if !around[a as usize].contains(&b) {
                around[a as usize].push(b);
            }
            if !around[b as usize].contains(&a) {
                around[b as usize].push(a);
            }
        }
    }
    for _ in 0..rounds {
        for factor in [0.5f32, -0.53] {
            let from = mesh.positions().to_vec();
            for (v, p) in mesh.positions_mut().iter_mut().enumerate() {
                let near = &around[v];
                if near.is_empty() {
                    continue;
                }
                let middle =
                    near.iter().map(|u| from[*u as usize]).sum::<Vec3>() / near.len() as f32;
                *p = from[v] + (middle - from[v]) * factor;
            }
        }
    }
}

/// The dimensions of the faces across `axis` of a grid of `n` cells.
pub(super) fn face_dims(n: [usize; 3], axis: usize) -> [usize; 3] {
    let mut d = n;
    d[axis] += 1;
    d
}

/// Calls `f` with each of the eight points around `q` (in grid points, of
/// a grid `dims` points along each axis) and its trilinear weight; points
/// past the edge are left out.
pub(super) fn for_trilinear(q: Vec3, dims: [usize; 3], mut f: impl FnMut(usize, f32)) {
    let max = Vec3::new(
        dims[0] as f32 - 1.0,
        dims[1] as f32 - 1.0,
        dims[2] as f32 - 1.0,
    );
    let q = q.clamp(Vec3::ZERO, max);
    let base = q.floor();
    let t = q - base;
    let (i, j, k) = (base.x as usize, base.y as usize, base.z as usize);
    for c in 0..8 {
        let (a, b, d) = (c & 1, (c >> 1) & 1, c >> 2);
        let (x, y, z) = (i + a, j + b, k + d);
        if x >= dims[0] || y >= dims[1] || z >= dims[2] {
            continue;
        }
        let wgt = (if a == 1 { t.x } else { 1.0 - t.x })
            * (if b == 1 { t.y } else { 1.0 - t.y })
            * (if d == 1 { t.z } else { 1.0 - t.z });
        if wgt > 0.0 {
            f((z * dims[1] + y) * dims[0] + x, wgt);
        }
    }
}

/// Solves `A x = b` for a symmetric positive (semi)definite `A`, given as
/// `apply`, by conjugate gradients preconditioned by its diagonal.
pub(super) fn conjugate_gradient(
    apply: &impl Fn(&[f32], &mut [f32]),
    diag: &[f32],
    b: &[f32],
    iterations: usize,
    tolerance: f32,
) -> Vec<f32> {
    let n = b.len();
    let mut x = vec![0.0f32; n];
    let mut r = b.to_vec();
    let norm_b = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if norm_b < 1e-12 {
        return x;
    }
    let precondition = |r: &[f32], z: &mut [f32]| {
        for i in 0..n {
            z[i] = if diag[i] > 0.0 { r[i] / diag[i] } else { 0.0 };
        }
    };
    let mut z = vec![0.0f32; n];
    precondition(&r, &mut z);
    let mut p = z.clone();
    let mut rz: f64 = r
        .iter()
        .zip(&z)
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let mut ap = vec![0.0f32; n];
    for _ in 0..iterations {
        apply(&p, &mut ap);
        let pap: f64 = p
            .iter()
            .zip(&ap)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        if pap.abs() < 1e-30 {
            break;
        }
        let alpha = (rz / pap) as f32;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if r.iter().map(|v| v.abs()).fold(0.0f32, f32::max) <= tolerance * norm_b {
            break;
        }
        precondition(&r, &mut z);
        let rz_new: f64 = r
            .iter()
            .zip(&z)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let beta = (rz_new / rz) as f32;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::DVec3;

    fn cube_at(centre: Vec3, size: f32) -> Vec<[Vec3; 3]> {
        let mesh = kreate_mesh::primitives::cube(size);
        world_triangles(&mesh, DAffine3::from_translation(centre.as_dvec3()))
    }

    fn bounds(lo: Vec3, hi: Vec3) -> Aabb {
        Aabb { min: lo, max: hi }
    }

    #[test]
    fn a_block_of_water_falls_and_spreads_over_the_floor_keeping_its_volume() {
        let mut sim = LiquidSim::new(bounds(Vec3::ZERO, Vec3::new(2.0, 1.0, 1.0)), 20, 0.95, 1);
        // A block in one end, standing tall.
        sim.fill(&cube_at(Vec3::new(0.3, 0.5, 0.45), 0.5), Vec3::ZERO);
        let count = sim.particle_count();
        assert!(count > 0);
        let height = |sim: &LiquidSim| sim.particles().map(|p| p.z).fold(0.0f32, f32::max);
        let spread = |sim: &LiquidSim| sim.particles().map(|p| p.x).fold(0.0f32, f32::max);
        let (h0, x0) = (height(&sim), spread(&sim));
        for _ in 0..48 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        assert_eq!(sim.particle_count(), count, "none lost");
        assert!(
            height(&sim) < h0 * 0.8,
            "it fell: {} from {h0}",
            height(&sim)
        );
        assert!(
            spread(&sim) > x0 + 0.5,
            "it ran along: {} from {x0}",
            spread(&sim)
        );
        for p in sim.particles() {
            assert!(
                p.min_element() >= 0.0 && p.x <= 2.0 && p.y <= 1.0 && p.z <= 1.0,
                "{p}"
            );
        }
        // Liquid lies in a layer, not a cloud: the particles packed about
        // as densely as they started, give or take a squeeze.
        let cells = sim
            .classify()
            .iter()
            .filter(|c| **c == Cell::Liquid)
            .count();
        let per_cell = count as f32 / cells as f32;
        assert!(per_cell > 3.0, "{per_cell} a liquid cell: it boiled away");
    }

    #[test]
    fn liquid_stands_still_in_a_box() {
        let mut sim = LiquidSim::new(bounds(Vec3::ZERO, Vec3::ONE), 16, 0.95, 2);
        sim.fill(
            &world_triangles(
                &kreate_mesh::primitives::cube(1.0),
                DAffine3::from_scale_rotation_translation(
                    DVec3::new(1.0, 1.0, 0.4),
                    Default::default(),
                    DVec3::new(0.5, 0.5, 0.2),
                ),
            ),
            Vec3::ZERO,
        );
        for _ in 0..24 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        let fastest = sim
            .drops
            .iter()
            .map(|d| d.vel.length())
            .fold(0.0f32, f32::max);
        assert!(fastest < 0.5, "still water stirs at {fastest} m/s");
        let top = sim.particles().map(|p| p.z).fold(0.0f32, f32::max);
        assert!((top - 0.4).abs() < 0.1, "its level is {top}");
    }

    #[test]
    fn an_inflow_pours_and_an_outflow_drains() {
        let mut sim = LiquidSim::new(bounds(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)), 12, 0.95, 3);
        sim.add_inflow(
            &cube_at(Vec3::new(0.5, 0.5, 0.85), 0.2),
            Vec3::new(0.0, 0.0, -1.0),
        );
        for _ in 0..12 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        let poured = sim.particle_count();
        assert!(poured > 0, "nothing poured");
        assert!(sim.particles().any(|p| p.z < 0.5), "it did not fall");
        sim.add_outflow(&cube_at(Vec3::new(0.5, 0.5, 0.1), 1.0));
        sim.inflows.clear();
        for _ in 0..24 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        assert!(
            sim.particle_count() < poured / 4,
            "{} left of {poured}",
            sim.particle_count()
        );
    }

    #[test]
    fn an_obstacle_keeps_the_liquid_out() {
        let mut sim = LiquidSim::new(bounds(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)), 16, 0.95, 4);
        // A shelf across the middle, and water above it.
        let shelf = world_triangles(
            &kreate_mesh::primitives::cube(1.0),
            DAffine3::from_scale_rotation_translation(
                DVec3::new(2.0, 2.0, 0.2),
                Default::default(),
                DVec3::new(0.5, 0.5, 0.4),
            ),
        );
        sim.set_obstacles(&[shelf]);
        sim.fill(&cube_at(Vec3::new(0.5, 0.5, 0.75), 0.3), Vec3::ZERO);
        for _ in 0..24 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        assert!(
            sim.particles().all(|p| p.z > 0.45),
            "it went through the shelf: lowest {}",
            sim.particles().map(|p| p.z).fold(f32::MAX, f32::min)
        );
    }

    #[test]
    fn the_surface_is_a_closed_skin_round_the_liquid() {
        let mut sim = LiquidSim::new(bounds(Vec3::ZERO, Vec3::ONE), 16, 0.95, 5);
        sim.fill(&cube_at(Vec3::splat(0.5), 0.5), Vec3::ZERO);
        let skin = sim.surface().expect("a surface");
        let (lo, hi) = skin.positions().iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        // About the block it was filled from.
        assert!(lo.distance(Vec3::splat(0.25)) < 0.12, "{lo}");
        assert!(hi.distance(Vec3::splat(0.75)) < 0.12, "{hi}");
        assert!(LiquidSim::new(bounds(Vec3::ZERO, Vec3::ONE), 8, 0.9, 0)
            .surface()
            .is_none());
    }

    #[test]
    fn a_still_pool_has_a_flat_smooth_top() {
        let mut sim = LiquidSim::new(bounds(Vec3::ZERO, Vec3::ONE), 16, 0.95, 6);
        sim.fill(
            &world_triangles(
                &kreate_mesh::primitives::cube(1.0),
                DAffine3::from_scale_rotation_translation(
                    DVec3::new(1.0, 1.0, 0.4),
                    Default::default(),
                    DVec3::new(0.5, 0.5, 0.2),
                ),
            ),
            Vec3::ZERO,
        );
        for _ in 0..12 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        let skin = sim.surface().expect("a surface");
        // The top: vertices well inside the walls and above the middle.
        let top: Vec<f32> = skin
            .positions()
            .iter()
            .filter(|p| p.z > 0.25 && p.x > 0.2 && p.x < 0.8 && p.y > 0.2 && p.y < 0.8)
            .map(|p| p.z)
            .collect();
        assert!(top.len() > 20, "{} top vertices", top.len());
        let mean = top.iter().sum::<f32>() / top.len() as f32;
        let rough = top.iter().map(|z| (z - mean).abs()).fold(0.0f32, f32::max);
        assert!(
            rough < sim.cell_size() * 0.35,
            "the top is {rough} rough, cells {}",
            sim.cell_size()
        );
        let smooth = skin
            .attributes()
            .get(kreate_mesh::attribute::names::SHADE_SMOOTH)
            .is_some();
        assert!(smooth, "not shaded smooth");
    }

    #[test]
    fn a_sheet_one_particle_thick_is_one_piece() {
        // What a splash thins to: particles spread one deep across a plane.
        let mut sim = LiquidSim::new(bounds(Vec3::ZERO, Vec3::ONE), 16, 0.95, 7);
        let step = sim.cell_size() / SEED as f32;
        let mut y = 0.2;
        while y < 0.8 {
            let mut x = 0.2;
            while x < 0.8 {
                sim.drops.push(Drop {
                    pos: Vec3::new(x, y, 0.5),
                    vel: Vec3::ZERO,
                });
                x += step;
            }
            y += step;
        }
        let skin = sim.surface().expect("a surface");
        // Pieces, by faces sharing vertices.
        let n = skin.num_verts() as usize;
        let mut root: Vec<usize> = (0..n).collect();
        fn find(root: &mut [usize], mut v: usize) -> usize {
            while root[v] != v {
                root[v] = root[root[v]];
                v = root[v];
            }
            v
        }
        for f in 0..skin.num_faces() {
            let ring = skin.face_verts(f);
            for w in ring.windows(2) {
                let (a, b) = (
                    find(&mut root, w[0] as usize),
                    find(&mut root, w[1] as usize),
                );
                root[a] = b;
            }
        }
        let pieces = (0..n).filter(|v| find(&mut root, *v) == *v).count();
        assert_eq!(pieces, 1, "the sheet came apart into {pieces} pieces");
    }
}
