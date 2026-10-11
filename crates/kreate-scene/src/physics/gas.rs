//! Smoke and fire: a gas on a grid, as Blender's Mantaflow gas is.
//!
//! The gas is fields on the cells — how much smoke, how hot, how much fuel,
//! how brightly it burns — carried along by a velocity on the cells' faces.
//! Each step the sources put smoke, heat and fuel where they are; heat and
//! smoke push upward (buoyancy), the swirls the grid would smear away are
//! given back (vorticity confinement); the pressure that keeps the flow
//! from piling up is taken off, with the domain's sides open so smoke
//! leaves through them; then everything is carried along the flow, looking
//! back along it from each cell (semi-Lagrangian), and smoke thins and heat
//! fades. Fuel burns where it is hot or where it was lit: the burning is
//! the flame, and makes heat and smoke.
//!
//! What a bake keeps of a frame is the smoke and the flame, a byte a cell
//! each, which the viewport draws as a volume.

use kreate_math::{Aabb, Vec3};
use kreate_mesh::remesh::winding;

use rayon::prelude::*;

use super::fluid::{conjugate_gradient, face_dims, for_trilinear, poisson_apply};

/// The most cells along a domain's longest side.
pub const MAX_RESOLUTION: u32 = 128;

/// How the gas behaves: the domain's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasSettings {
    /// How strongly heat lifts the gas, metres a second squared at full heat.
    pub buoyancy: f32,
    /// How strongly smoke itself sinks (positive) or rises (negative).
    pub weight: f32,
    /// Swirl given back each step. 0 is smooth, higher curls.
    pub vorticity: f32,
    /// The fraction of the smoke that thins away each second.
    pub dissipation: f32,
    /// The fraction of the heat lost each second.
    pub cooling: f32,
    /// The fraction of fuel that burns each second once lit.
    pub burn_rate: f32,
    /// How hard the gas is stirred by small eddies of its own: breaks a
    /// plume's smooth sides into curls. 0 is none.
    pub turbulence: f32,
}

impl Default for GasSettings {
    fn default() -> Self {
        Self {
            buoyancy: 2.0,
            weight: 0.0,
            vorticity: 0.3,
            dissipation: 0.05,
            cooling: 0.6,
            burn_rate: 2.5,
            turbulence: 0.5,
        }
    }
}

/// What a source puts into the cells it fills.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emission {
    pub smoke: f32,
    pub heat: f32,
    pub fuel: f32,
    pub velocity: Vec3,
}

#[derive(Clone, Debug)]
struct Source {
    cells: Vec<bool>,
    emission: Emission,
}

/// A gas being simulated.
#[derive(Clone, Debug)]
pub struct GasSim {
    n: [usize; 3],
    h: f32,
    origin: Vec3,
    u: Vec<f32>,
    v: Vec<f32>,
    w: Vec<f32>,
    pub density: Vec<f32>,
    pub heat: Vec<f32>,
    pub fuel: Vec<f32>,
    /// How brightly each cell burned in the last step, 0 to 1.
    pub flame: Vec<f32>,
    solid: Vec<bool>,
    /// The last step's pressure in each cell, where the next solve starts.
    pressure: Vec<f32>,
    sources: Vec<Source>,
    /// Cells that take away whatever gas enters them.
    sinks: Vec<Vec<bool>>,
    settings: GasSettings,
    /// What stirs the sources, the same every bake.
    rng: super::Rng,
    /// Seconds simulated, which the eddies drift with.
    time: f32,
}

impl GasSim {
    pub fn new(bounds: Aabb, resolution: u32, settings: GasSettings) -> Self {
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
            density: vec![0.0; cells],
            heat: vec![0.0; cells],
            fuel: vec![0.0; cells],
            flame: vec![0.0; cells],
            solid: vec![false; cells],
            pressure: vec![0.0; cells],
            sources: Vec::new(),
            sinks: Vec::new(),
            settings,
            rng: super::Rng::new(0x5EED),
            time: 0.0,
        }
    }

    pub fn cells(&self) -> [usize; 3] {
        self.n
    }

    /// The box the cells fill, in the world.
    pub fn bounds(&self) -> Aabb {
        Aabb {
            min: self.origin,
            max: self.origin
                + Vec3::new(self.n[0] as f32, self.n[1] as f32, self.n[2] as f32) * self.h,
        }
    }

    fn cell(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.n[1] + j) * self.n[0] + i
    }

    fn cells_inside(&self, tris: &[[Vec3; 3]]) -> Vec<bool> {
        if tris.is_empty() {
            return vec![false; self.density.len()];
        }
        winding(
            tris,
            self.origin + Vec3::splat(self.h * 0.5),
            self.h,
            self.n,
        )
    }

    /// Something that keeps putting smoke, heat or fuel where `tris`
    /// encloses.
    pub fn add_source(&mut self, tris: &[[Vec3; 3]], emission: Emission) {
        let cells = self.cells_inside(tris);
        self.sources.push(Source { cells, emission });
    }

    /// Puts smoke, heat or fuel where `tris` encloses, once: a puff.
    pub fn puff(&mut self, tris: &[[Vec3; 3]], emission: Emission) {
        let cells = self.cells_inside(tris);
        let source = Source { cells, emission };
        let kept = std::mem::replace(&mut self.sources, vec![source]);
        self.emit();
        self.sources = kept;
    }

    /// Something that takes away whatever gas enters what `tris` encloses.
    pub fn add_sink(&mut self, tris: &[[Vec3; 3]]) {
        let cells = self.cells_inside(tris);
        self.sinks.push(cells);
    }

    /// The cells obstacles fill this frame.
    pub fn set_obstacles(&mut self, obstacles: &[Vec<[Vec3; 3]>]) {
        let mut solid = vec![false; self.solid.len()];
        for tris in obstacles {
            for (s, inside) in solid.iter_mut().zip(self.cells_inside(tris)) {
                *s |= inside;
            }
        }
        for (c, s) in solid.iter().enumerate() {
            if *s {
                self.density[c] = 0.0;
                self.heat[c] = 0.0;
                self.fuel[c] = 0.0;
            }
        }
        self.solid = solid;
    }

    /// Advances `seconds` under `gravity`, in steps short enough that
    /// nothing is carried more than a couple of cells in one.
    pub fn advance(&mut self, seconds: f32, gravity: Vec3) {
        let mut left = seconds;
        let mut steps = 0;
        while left > 1e-6 && steps < 32 {
            let fastest = self
                .u
                .iter()
                .chain(&self.v)
                .chain(&self.w)
                .fold(0.0f32, |m, x| m.max(x.abs()))
                .max(1e-3);
            let dt = (2.0 * self.h / fastest).min(left);
            self.step(dt, gravity);
            left -= dt;
            steps += 1;
        }
    }

    fn step(&mut self, dt: f32, gravity: Vec3) {
        self.emit();
        self.burn(dt);
        self.forces(dt, gravity);
        self.project();
        self.advect(dt);
        self.time += dt;
        let s = self.settings;
        let thin = (1.0 - s.dissipation * dt).clamp(0.0, 1.0);
        let cool = (1.0 - s.cooling * dt).clamp(0.0, 1.0);
        for d in self.density.iter_mut() {
            *d *= thin;
        }
        for t in self.heat.iter_mut() {
            *t *= cool;
        }
        for sink in &self.sinks {
            for (c, inside) in sink.iter().enumerate() {
                if *inside {
                    self.density[c] = 0.0;
                    self.heat[c] = 0.0;
                    self.fuel[c] = 0.0;
                    self.flame[c] = 0.0;
                }
            }
        }
    }

    fn emit(&mut self) {
        for source in &self.sources {
            let e = source.emission;
            for (c, inside) in source.cells.iter().enumerate() {
                if !*inside || self.solid[c] {
                    continue;
                }
                // Each cell a little more or less than the next: a real
                // source is never even, and the swirls grow from the
                // difference — a perfectly even one rises as a smooth blob.
                let wobble = 0.75 + 0.5 * self.rng.unit() as f32;
                self.density[c] = self.density[c].max(e.smoke * wobble);
                self.heat[c] = self.heat[c].max(e.heat * wobble);
                self.fuel[c] = self.fuel[c].max(e.fuel * wobble);
            }
        }
        // And the flow in it moves as the source says: a jet, not a
        // puff that happens to sit in the jet.
        let n = self.n;
        for source in &self.sources {
            let e = source.emission;
            if e.velocity == Vec3::ZERO {
                continue;
            }
            for k in 0..n[2] {
                for j in 0..n[1] {
                    for i in 0..n[0] {
                        if !source.cells[self.cell(i, j, k)] {
                            continue;
                        }
                        let fu = (k * n[1] + j) * (n[0] + 1) + i;
                        let fv = (k * (n[1] + 1) + j) * n[0] + i;
                        let fw = (k * n[1] + j) * n[0] + i;
                        self.u[fu] = e.velocity.x;
                        self.u[fu + 1] = e.velocity.x;
                        self.v[fv] = e.velocity.y;
                        self.v[fv + n[0]] = e.velocity.y;
                        self.w[fw] = e.velocity.z;
                        self.w[fw + n[0] * n[1]] = e.velocity.z;
                    }
                }
            }
        }
    }

    /// Fuel burns once lit — where it is hot — into flame, heat and smoke.
    fn burn(&mut self, dt: f32) {
        let rate = (self.settings.burn_rate * dt).clamp(0.0, 1.0);
        for c in 0..self.fuel.len() {
            let fuel = self.fuel[c];
            if fuel <= 1e-4 {
                self.flame[c] = 0.0;
                continue;
            }
            // Fuel needs heat to burn; a source of fuel is lit where it is.
            let lit = (self.heat[c] * 2.0).clamp(0.0, 1.0).max(0.5);
            let burned = fuel * rate * lit;
            self.fuel[c] = fuel - burned;
            self.flame[c] = (burned / rate.max(1e-6)).clamp(0.0, 1.0);
            self.heat[c] = (self.heat[c] + burned * 2.0).min(2.0);
            self.density[c] = (self.density[c] + burned * 0.6).min(2.0);
        }
    }

    /// Buoyancy and vorticity confinement, onto the faces.
    fn forces(&mut self, dt: f32, gravity: Vec3) {
        let n = self.n;
        let s = self.settings;
        let up = -gravity.normalize_or(Vec3::NEG_Z);
        // Cell-centred force, then averaged onto each face.
        let mut force: Vec<Vec3> = (0..self.density.len())
            .into_par_iter()
            .map(|c| {
                if self.solid[c] {
                    Vec3::ZERO
                } else {
                    up * (s.buoyancy * self.heat[c] - s.weight * self.density[c])
                }
            })
            .collect();
        // The flow at each cell's middle, once, for both the swirls and the
        // eddies.
        let centre = if s.vorticity > 0.0 || s.turbulence > 0.0 {
            self.centred()
        } else {
            Vec::new()
        };
        if s.vorticity > 0.0 {
            let at = |i: isize, j: isize, k: isize| -> Vec3 {
                let i = i.clamp(0, n[0] as isize - 1) as usize;
                let j = j.clamp(0, n[1] as isize - 1) as usize;
                let k = k.clamp(0, n[2] as isize - 1) as usize;
                centre[(k * n[1] + j) * n[0] + i]
            };
            let h2 = 2.0 * self.h;
            let place = |c: usize| {
                (
                    (c % n[0]) as isize,
                    ((c / n[0]) % n[1]) as isize,
                    (c / (n[0] * n[1])) as isize,
                )
            };
            let curl: Vec<Vec3> = (0..centre.len())
                .into_par_iter()
                .map(|c| {
                    let (i, j, k) = place(c);
                    let dx = (at(i + 1, j, k) - at(i - 1, j, k)) / h2;
                    let dy = (at(i, j + 1, k) - at(i, j - 1, k)) / h2;
                    let dz = (at(i, j, k + 1) - at(i, j, k - 1)) / h2;
                    Vec3::new(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x)
                })
                .collect();
            let len = |i: isize, j: isize, k: isize| -> f32 {
                let i = i.clamp(0, n[0] as isize - 1) as usize;
                let j = j.clamp(0, n[1] as isize - 1) as usize;
                let k = k.clamp(0, n[2] as isize - 1) as usize;
                curl[(k * n[1] + j) * n[0] + i].length()
            };
            let h = self.h;
            force.par_iter_mut().enumerate().for_each(|(c, force)| {
                let (i, j, k) = place(c);
                let grad = Vec3::new(
                    len(i + 1, j, k) - len(i - 1, j, k),
                    len(i, j + 1, k) - len(i, j - 1, k),
                    len(i, j, k + 1) - len(i, j, k - 1),
                ) / h2;
                if let Some(towards) = grad.try_normalize() {
                    // Fedkiw's ε·h·(N × ω).
                    *force += 2.0 * s.vorticity * h * towards.cross(curl[c]);
                }
            });
        }
        // Eddies: the curl of a smooth noise a few cells across, drifting
        // with time — swirls that turn without squeezing, so they add no
        // pressure for the projection to fight. Only where there is gas;
        // still air stirred is work for nothing.
        if s.turbulence > 0.0 {
            let across = 4.0 * self.h;
            let drift = up * (-0.4 * self.time / across);
            // They turn the flow, not speed it: each cell's velocity is
            // swung about the noise's direction, as fast as the flow crosses
            // an eddy — so still smoke is left still and a fast plume is torn
            // at its edges — and at right angles to itself, so no energy is
            // added. A push along the noise instead fed on the speed it made,
            // and a hot fire's plume ran away with it.
            let flow = &centre;
            let (solid, density, heat, fuel, h) =
                (&self.solid, &self.density, &self.heat, &self.fuel, self.h);
            force.par_iter_mut().enumerate().for_each(|(c, force)| {
                if solid[c] || density[c] + heat[c] + fuel[c] < 0.01 {
                    return;
                }
                let (i, j, k) = (c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1]));
                let p = Vec3::new(i as f32, j as f32, k as f32) * (h / across) + drift;
                let speed = flow[c].length();
                let Some(axis) = curl_noise(p).try_normalize() else {
                    return;
                };
                // Radians a second, held below a quarter turn a step.
                let turn = (0.3 * s.turbulence * speed / across).min(0.25 / dt.max(1e-4));
                *force += axis.cross(flow[c]) * turn;
            });
        }
        for (axis, field) in [&mut self.u, &mut self.v, &mut self.w]
            .into_iter()
            .enumerate()
        {
            let dims = face_dims(n, axis);
            for k in 0..dims[2] {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        let at = [i, j, k];
                        let mut sum = 0.0;
                        let mut count = 0.0;
                        if at[axis] < n[axis] {
                            sum += force[(k * n[1] + j) * n[0] + i][axis];
                            count += 1.0;
                        }
                        if at[axis] > 0 {
                            let mut lo = at;
                            lo[axis] -= 1;
                            sum += force[(lo[2] * n[1] + lo[1]) * n[0] + lo[0]][axis];
                            count += 1.0;
                        }
                        if count > 0.0 {
                            field[(k * dims[1] + j) * dims[0] + i] += sum / count * dt;
                        }
                    }
                }
            }
        }
    }

    /// The velocity at each cell's middle: the average of its faces.
    fn centred(&self) -> Vec<Vec3> {
        let n = self.n;
        let mut out = vec![Vec3::ZERO; self.density.len()];
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let fu = (k * n[1] + j) * (n[0] + 1) + i;
                    let fv = (k * (n[1] + 1) + j) * n[0] + i;
                    let fw = (k * n[1] + j) * n[0] + i;
                    out[self.cell(i, j, k)] = Vec3::new(
                        0.5 * (self.u[fu] + self.u[fu + 1]),
                        0.5 * (self.v[fv] + self.v[fv + n[0]]),
                        0.5 * (self.w[fw] + self.w[fw + n[0] * n[1]]),
                    );
                }
            }
        }
        out
    }

    /// The pressure that keeps the flow from piling up, taken off: every
    /// open cell an unknown, the world past the sides at no pressure, an
    /// obstacle's faces closed.
    fn project(&mut self) {
        let n = self.n;
        let cells = self.density.len();
        let mut index = vec![usize::MAX; cells];
        let mut open = Vec::new();
        for (c, at) in index.iter_mut().enumerate() {
            if !self.solid[c] {
                *at = open.len();
                open.push(c);
            }
        }
        if open.is_empty() {
            return;
        }
        // Closed faces first: into an obstacle there is no flow.
        self.close_obstacles();
        let uix = |i: usize, j: usize, k: usize| (k * n[1] + j) * (n[0] + 1) + i;
        let vix = |i: usize, j: usize, k: usize| (k * (n[1] + 1) + j) * n[0] + i;
        let wix = |i: usize, j: usize, k: usize| (k * n[1] + j) * n[0] + i;
        let mut diag = vec![0.0f32; open.len()];
        let mut nbrs: Vec<[usize; 6]> = vec![[usize::MAX; 6]; open.len()];
        let mut rhs = vec![0.0f32; open.len()];
        for (r, &c) in open.iter().enumerate() {
            let (i, j, k) = (c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1]));
            rhs[r] = -(self.u[uix(i + 1, j, k)] - self.u[uix(i, j, k)] + self.v[vix(i, j + 1, k)]
                - self.v[vix(i, j, k)]
                + self.w[wix(i, j, k + 1)]
                - self.w[wix(i, j, k)]);
            let around = [
                (i > 0).then(|| (i - 1, j, k)),
                (i + 1 < n[0]).then(|| (i + 1, j, k)),
                (j > 0).then(|| (i, j - 1, k)),
                (j + 1 < n[1]).then(|| (i, j + 1, k)),
                (k > 0).then(|| (i, j, k - 1)),
                (k + 1 < n[2]).then(|| (i, j, k + 1)),
            ];
            for (s, nb) in around.into_iter().enumerate() {
                match nb {
                    // Past the side: the open world, at no pressure.
                    None => diag[r] += 1.0,
                    Some((a, b, d)) => {
                        let nc = self.cell(a, b, d);
                        if !self.solid[nc] {
                            diag[r] += 1.0;
                            nbrs[r][s] = index[nc];
                        }
                    }
                }
            }
        }
        let apply = |x: &[f32], out: &mut [f32]| poisson_apply(&diag, &nbrs, x, out);
        let guess = open.iter().map(|c| self.pressure[*c]).collect();
        let q = conjugate_gradient(&apply, &diag, &rhs, guess, 200, 1e-4);
        for (r, c) in open.iter().enumerate() {
            self.pressure[*c] = q[r];
        }
        let solid = &self.solid;
        let pressure = |c: Option<usize>| -> f32 {
            match c {
                Some(c) if !solid[c] => q[index[c]],
                _ => 0.0,
            }
        };
        for (axis, field) in [&mut self.u, &mut self.v, &mut self.w]
            .into_iter()
            .enumerate()
        {
            let dims = face_dims(n, axis);
            for k in 0..dims[2] {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        let hi = [i, j, k];
                        let b = (hi[axis] < n[axis]).then(|| (k * n[1] + j) * n[0] + i);
                        let a = (hi[axis] > 0).then(|| {
                            let mut lo = hi;
                            lo[axis] -= 1;
                            (lo[2] * n[1] + lo[1]) * n[0] + lo[0]
                        });
                        if a.is_some_and(|c| solid[c]) || b.is_some_and(|c| solid[c]) {
                            continue;
                        }
                        field[(k * dims[1] + j) * dims[0] + i] -= pressure(b) - pressure(a);
                    }
                }
            }
        }
    }

    fn close_obstacles(&mut self) {
        if !self.solid.iter().any(|s| *s) {
            return;
        }
        let n = self.n;
        let solid = self.solid.clone();
        for (axis, field) in [&mut self.u, &mut self.v, &mut self.w]
            .into_iter()
            .enumerate()
        {
            let dims = face_dims(n, axis);
            for k in 0..dims[2] {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        let hi = [i, j, k];
                        let b = hi[axis] < n[axis] && solid[(k * n[1] + j) * n[0] + i];
                        let a = hi[axis] > 0 && {
                            let mut lo = hi;
                            lo[axis] -= 1;
                            solid[(lo[2] * n[1] + lo[1]) * n[0] + lo[0]]
                        };
                        if a || b {
                            field[(k * dims[1] + j) * dims[0] + i] = 0.0;
                        }
                    }
                }
            }
        }
    }

    /// The velocity at a point, in grid units of cells.
    fn velocity_at(&self, g: Vec3) -> Vec3 {
        let mut out = Vec3::ZERO;
        for (axis, field) in [&self.u, &self.v, &self.w].into_iter().enumerate() {
            let dims = face_dims(self.n, axis);
            let mut q = g - Vec3::splat(0.5);
            q[axis] += 0.5;
            let mut s = 0.0;
            for_trilinear(q, dims, |at, wgt| s += field[at] * wgt);
            out[axis] = s;
        }
        out
    }

    fn scalar_at(&self, field: &[f32], g: Vec3) -> f32 {
        let mut s = 0.0;
        for_trilinear(g - Vec3::splat(0.5), self.n, |at, wgt| s += field[at] * wgt);
        s
    }

    /// Everything carried along the flow: each cell takes what was where
    /// the flow brought it from, traced back at the midpoint.
    fn advect(&mut self, dt: f32) {
        let n = self.n;
        let back = |sim: &GasSim, g: Vec3| -> Vec3 {
            let v = sim.velocity_at(g) / sim.h;
            let mid = g - v * (0.5 * dt);
            g - sim.velocity_at(mid) / sim.h * dt
        };
        // Every cell and every face looks back on its own, from the flow as
        // it stood: every core takes a share.
        let size = Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32);
        let carried: Vec<[f32; 3]> = (0..self.density.len())
            .into_par_iter()
            .map(|c| {
                if self.solid[c] {
                    return [0.0; 3];
                }
                let (i, j, k) = (c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1]));
                let g = Vec3::new(i as f32, j as f32, k as f32) + Vec3::splat(0.5);
                let from = back(self, g);
                // From outside the domain comes clear air.
                if from.min_element() < 0.0 || from.cmpgt(size).any() {
                    return [0.0; 3];
                }
                [
                    self.scalar_at(&self.density, from),
                    self.scalar_at(&self.heat, from),
                    self.scalar_at(&self.fuel, from),
                ]
            })
            .collect();
        let scalars = [0, 1, 2].map(|s| carried.iter().map(|c| c[s]).collect::<Vec<f32>>());
        let faces = [0, 1, 2].map(|axis| {
            let dims = face_dims(n, axis);
            (0..dims.iter().product::<usize>())
                .into_par_iter()
                .map(|at| {
                    let (i, j, k) = (
                        at % dims[0],
                        (at / dims[0]) % dims[1],
                        at / (dims[0] * dims[1]),
                    );
                    let mut g = Vec3::new(i as f32, j as f32, k as f32) + Vec3::splat(0.5);
                    g[axis] -= 0.5;
                    let from = back(self, g);
                    self.velocity_at(from)[axis]
                })
                .collect::<Vec<f32>>()
        });
        let [density, heat, fuel] = scalars;
        self.density = density;
        self.heat = heat;
        self.fuel = fuel;
        let [u, v, w] = faces;
        self.u = u;
        self.v = v;
        self.w = w;
        self.close_obstacles();
    }

    /// The smoke and the flame as bytes, a cell each, X fastest: what a bake
    /// keeps and the viewport draws.
    pub fn snapshot(&self) -> (Vec<u8>, Vec<u8>) {
        let byte = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
        (
            self.density.iter().map(|d| byte(*d)).collect(),
            self.flame.iter().map(|f| byte(*f)).collect(),
        )
    }
}

/// A lattice point's value, from -1 to 1: a hash, the same everywhere.
fn lattice(x: i32, y: i32, z: i32, channel: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ (z as u32).wrapping_mul(0xCB1A_B31F)
        ^ channel.wrapping_mul(0x1656_67B1);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5BD1_E995);
    h ^= h >> 15;
    (h & 0xFFFF) as f32 / 32767.5 - 1.0
}

/// Smooth value noise: the lattice blended with a quintic, so its slope is
/// smooth too — its curl, which the eddies are, has no seams.
fn value_noise(p: Vec3, channel: u32) -> f32 {
    let base = p.floor();
    let f = p - base;
    let s = f * f * f * (f * (f * 6.0 - Vec3::splat(15.0)) + Vec3::splat(10.0));
    let (x, y, z) = (base.x as i32, base.y as i32, base.z as i32);
    let mut out = 0.0;
    for c in 0..8 {
        let (a, b, d) = (c & 1, (c >> 1) & 1, c >> 2);
        let w = (if a == 1 { s.x } else { 1.0 - s.x })
            * (if b == 1 { s.y } else { 1.0 - s.y })
            * (if d == 1 { s.z } else { 1.0 - s.z });
        out += w * lattice(x + a, y + b, z + d, channel);
    }
    out
}

/// The curl of three value noises: a flow that swirls and neither gathers
/// nor spreads.
fn curl_noise(p: Vec3) -> Vec3 {
    let e = 0.05;
    let d = |axis: usize, channel: u32| {
        let mut step = Vec3::ZERO;
        step[axis] = e;
        (value_noise(p + step, channel) - value_noise(p - step, channel)) / (2.0 * e)
    };
    Vec3::new(d(1, 2) - d(2, 1), d(2, 0) - d(0, 2), d(0, 1) - d(1, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::DAffine3;

    fn cube_at(centre: Vec3, size: f32) -> Vec<[Vec3; 3]> {
        super::super::fluid::world_triangles(
            &kreate_mesh::primitives::cube(size),
            DAffine3::from_translation(centre.as_dvec3()),
        )
    }

    fn domain() -> Aabb {
        Aabb {
            min: Vec3::ZERO,
            max: Vec3::new(1.0, 1.0, 2.0),
        }
    }

    /// Where the smoke is, on average, and how much there is.
    fn centre_of(sim: &GasSim) -> (Vec3, f32) {
        let n = sim.n;
        let (mut at, mut total) = (Vec3::ZERO, 0.0);
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let d = sim.density[sim.cell(i, j, k)];
                    at += Vec3::new(i as f32, j as f32, k as f32) * d;
                    total += d;
                }
            }
        }
        (at / total.max(1e-6) * sim.h, total)
    }

    #[test]
    fn hot_smoke_rises_from_its_source() {
        let mut sim = GasSim::new(domain(), 16, GasSettings::default());
        sim.add_source(
            &cube_at(Vec3::new(0.5, 0.5, 0.2), 0.3),
            Emission {
                smoke: 1.0,
                heat: 1.0,
                fuel: 0.0,
                velocity: Vec3::ZERO,
            },
        );
        sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        let (low, _) = centre_of(&sim);
        for _ in 0..36 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        let (high, amount) = centre_of(&sim);
        assert!(amount > 1.0, "smoke made: {amount}");
        assert!(high.z > low.z + 0.3, "rose from {} to {}", low.z, high.z);
        assert!(
            (high.x - 0.5).abs() < 0.2 && (high.y - 0.5).abs() < 0.2,
            "{high}"
        );
        assert!(sim.density.iter().all(|d| d.is_finite() && *d >= 0.0));
    }

    #[test]
    fn cold_smoke_with_nothing_to_lift_it_stays_put_and_thins() {
        let mut sim = GasSim::new(
            domain(),
            12,
            GasSettings {
                dissipation: 0.5,
                ..GasSettings::default()
            },
        );
        let block = cube_at(Vec3::new(0.5, 0.5, 1.0), 0.4);
        let mut puff = GasSim::new(domain(), 12, GasSettings::default());
        puff.add_source(
            &block,
            Emission {
                smoke: 1.0,
                heat: 0.0,
                fuel: 0.0,
                velocity: Vec3::ZERO,
            },
        );
        puff.emit();
        sim.density = puff.density.clone();
        let (before, amount) = centre_of(&sim);
        for _ in 0..24 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        let (after, left) = centre_of(&sim);
        assert!(after.distance(before) < 0.05, "drifted {before} → {after}");
        assert!(
            left < amount * 0.7 && left > amount * 0.4,
            "{left} of {amount}"
        );
    }

    #[test]
    fn fuel_burns_into_flame_heat_and_smoke() {
        let mut sim = GasSim::new(domain(), 12, GasSettings::default());
        sim.add_source(
            &cube_at(Vec3::new(0.5, 0.5, 0.2), 0.3),
            Emission {
                smoke: 0.0,
                heat: 0.0,
                fuel: 1.0,
                velocity: Vec3::ZERO,
            },
        );
        for _ in 0..36 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        assert!(sim.flame.iter().any(|f| *f > 0.3), "no flame");
        assert!(sim.heat.iter().any(|t| *t > 0.3), "no heat");
        // The smoke the burning makes is carried up by its heat.
        let n = sim.n;
        let top = (0..sim.density.len())
            .filter(|c| sim.density[*c] > 0.05)
            .map(|c| c / (n[0] * n[1]))
            .max()
            .unwrap_or(0) as f32
            * sim.h;
        assert!(top > 0.8, "smoke only reached {top}");
    }

    #[test]
    fn an_obstacle_parts_the_plume() {
        let mut sim = GasSim::new(domain(), 16, GasSettings::default());
        sim.set_obstacles(&[cube_at(Vec3::new(0.5, 0.5, 1.0), 0.5)]);
        sim.add_source(
            &cube_at(Vec3::new(0.5, 0.5, 0.2), 0.3),
            Emission {
                smoke: 1.0,
                heat: 1.0,
                fuel: 0.0,
                velocity: Vec3::ZERO,
            },
        );
        for _ in 0..72 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        for (c, solid) in sim.solid.iter().enumerate() {
            if *solid {
                assert_eq!(sim.density[c], 0.0, "smoke inside the obstacle");
            }
        }
        // Held under it, then round its sides: up past its middle by now.
        let n = sim.n;
        let beside: f32 = (0..sim.density.len())
            .filter(|c| c / (n[0] * n[1]) > (1.0 / sim.h) as usize)
            .map(|c| sim.density[c])
            .sum();
        assert!(beside > 0.1, "{beside} past the obstacle's middle");
    }

    #[test]
    fn a_snapshot_is_a_byte_a_cell() {
        let mut sim = GasSim::new(domain(), 8, GasSettings::default());
        sim.density[0] = 2.0;
        sim.density[1] = 0.5;
        sim.flame[2] = 1.0;
        let (smoke, flame) = sim.snapshot();
        assert_eq!(smoke.len(), sim.density.len());
        assert_eq!((smoke[0], smoke[1], flame[2]), (255, 128, 255));
    }

    #[test]
    fn turbulence_breaks_a_plume_up_and_it_still_rises() {
        let source = cube_at(Vec3::new(0.5, 0.5, 0.2), 0.3);
        let plume = |turbulence: f32| {
            let mut sim = GasSim::new(
                domain(),
                16,
                GasSettings {
                    turbulence,
                    vorticity: 0.0,
                    ..GasSettings::default()
                },
            );
            sim.add_source(
                &source,
                Emission {
                    smoke: 1.0,
                    heat: 1.0,
                    fuel: 0.0,
                    velocity: Vec3::ZERO,
                },
            );
            for _ in 0..36 {
                sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
            }
            sim
        };
        let (calm, stirred) = (plume(0.0), plume(1.0));
        // How far the smoke is from mirror-symmetric across x = 0.5.
        let lopsided = |sim: &GasSim| {
            let n = sim.n;
            let mut apart = 0.0;
            for k in 0..n[2] {
                for j in 0..n[1] {
                    for i in 0..n[0] {
                        let a = sim.density[sim.cell(i, j, k)];
                        let b = sim.density[sim.cell(n[0] - 1 - i, j, k)];
                        apart += (a - b).abs();
                    }
                }
            }
            apart
        };
        assert!(
            lopsided(&stirred) > lopsided(&calm) * 2.0 + 0.5,
            "stirred {} calm {}",
            lopsided(&stirred),
            lopsided(&calm)
        );
        let (low, _) = centre_of(&calm);
        let (high, _) = centre_of(&stirred);
        // Mixing slows a plume, but it still climbs well clear of its
        // source, which sits at 0.2.
        assert!(
            high.z > 0.5 && high.z > low.z * 0.6,
            "it stopped rising: {} vs {}",
            high.z,
            low.z
        );
        // And the same every time.
        assert_eq!(plume(1.0).density, stirred.density);
    }

    #[test]
    fn a_hot_fire_stirred_hard_does_not_run_away() {
        let mut sim = GasSim::new(
            domain(),
            16,
            GasSettings {
                turbulence: 3.0,
                ..GasSettings::default()
            },
        );
        sim.add_source(
            &cube_at(Vec3::new(0.5, 0.5, 0.2), 0.3),
            Emission {
                smoke: 0.25,
                heat: 2.0,
                fuel: 1.0,
                velocity: Vec3::ZERO,
            },
        );
        let started = std::time::Instant::now();
        for _ in 0..48 {
            sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
        }
        let fastest = sim
            .u
            .iter()
            .chain(&sim.v)
            .chain(&sim.w)
            .fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            fastest.is_finite() && fastest < 20.0,
            "the flow ran away to {fastest} m/s"
        );
        assert!(
            started.elapsed().as_secs() < 30,
            "took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_bake_is_the_same_on_one_core_as_on_many() {
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut sim = GasSim::new(domain(), 12, GasSettings::default());
                sim.add_source(
                    &cube_at(Vec3::new(0.5, 0.5, 0.2), 0.3),
                    Emission {
                        smoke: 0.5,
                        heat: 1.0,
                        fuel: 1.0,
                        velocity: Vec3::ZERO,
                    },
                );
                for _ in 0..12 {
                    sim.advance(1.0 / 24.0, Vec3::new(0.0, 0.0, -9.81));
                }
                (sim.density, sim.flame)
            })
        };
        assert_eq!(run(1), run(4));
    }
}
