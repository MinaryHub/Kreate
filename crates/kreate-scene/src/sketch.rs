//! Sketch constraints: dimensions and relations on a sketch's points, and the
//! solver that moves the points until they hold.
//!
//! A sketch is a flat mesh in its object's XY plane (see
//! `kreate_mesh::profile`); a constraint names its vertices by index. The
//! solver is damped least squares (Levenberg–Marquardt) over the points the
//! constraints name: each step moves them as little as it can toward every
//! constraint at once, so an underconstrained sketch keeps the shape the user
//! drew where nothing says otherwise, and a contradictory one ends as close
//! as it can get and says so.

use kreate_math::DVec2;

/// One constraint, on the sketch's vertices by index.
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    /// Two points this far apart.
    Distance { a: u32, b: u32, value: f64 },
    /// Two points level: the same Y.
    Horizontal { a: u32, b: u32 },
    /// Two points one above the other: the same X.
    Vertical { a: u32, b: u32 },
    /// A point held where it is.
    Fixed { a: u32, at: [f64; 2] },
    /// The angle from line ab to line cd, in radians.
    Angle {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        value: f64,
    },
    /// Line ab parallel to line cd.
    Parallel { a: u32, b: u32, c: u32, d: u32 },
    /// Line ab square to line cd.
    Perpendicular { a: u32, b: u32, c: u32, d: u32 },
    /// Line ab as long as line cd.
    Equal { a: u32, b: u32, c: u32, d: u32 },
    /// Two points one: an end on an end, a line's end on a circle's centre
    /// (two circles' centres are concentric this way).
    Coincident { a: u32, b: u32 },
    /// Point `p` on the line through `a` and `b`.
    OnLine { p: u32, a: u32, b: u32 },
    /// A circle: `ring` evenly round `centre`, its radius however far the
    /// first is. The radius is a [`Constraint::Distance`] from the centre to
    /// the first, when it is dimensioned.
    Circle { centre: u32, ring: Vec<u32> },
    /// An arc: `ring` evenly round `centre`, anticlockwise from the first to
    /// the last, all as far from it as the first is. Its ends are free; how
    /// far round it goes is wherever they are.
    Arc { centre: u32, ring: Vec<u32> },
    /// Line ab touching the circle or arc about `centre` through `on`: the
    /// line as far from the centre as the radius.
    Tangent {
        a: u32,
        b: u32,
        centre: u32,
        on: u32,
    },
    /// A smooth curve through `through`, in order: `ring` are the points
    /// along it between each pair, evenly by parameter — Fusion's fit-point
    /// spline. Closed, it comes back round to the first.
    Spline {
        through: Vec<u32>,
        ring: Vec<u32>,
        closed: bool,
    },
}

/// Where a fit-point spline's ring points lie: `count` points spread evenly
/// by parameter along a Catmull-Rom through `through`, between each pair of
/// its points — what the ring is held to.
pub fn spline_ring(through: &[DVec2], count: usize, closed: bool) -> Vec<DVec2> {
    let n = through.len();
    if n < 2 || count == 0 {
        return Vec::new();
    }
    let spans = if closed { n } else { n - 1 };
    let per = count / spans;
    let at = |i: isize| -> DVec2 {
        if closed {
            through[i.rem_euclid(n as isize) as usize]
        } else {
            through[i.clamp(0, n as isize - 1) as usize]
        }
    };
    let mut out = Vec::with_capacity(count);
    for s in 0..spans {
        let (p0, p1, p2, p3) = (
            at(s as isize - 1),
            at(s as isize),
            at(s as isize + 1),
            at(s as isize + 2),
        );
        for j in 1..=per {
            let t = j as f64 / (per + 1) as f64;
            let (t2, t3) = (t * t, t * t * t);
            out.push(
                (p1 * 2.0
                    + (p2 - p0) * t
                    + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
                    + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3)
                    * 0.5,
            );
        }
    }
    out
}

/// The kinds, in the order the operators number them.
pub const KINDS: [&str; 14] = [
    "distance",
    "horizontal",
    "vertical",
    "fixed",
    "angle",
    "parallel",
    "perpendicular",
    "equal",
    "coincident",
    "on_line",
    "circle",
    "arc",
    "tangent",
    "spline",
];

impl Constraint {
    pub fn tag(&self) -> &'static str {
        KINDS[self.kind()]
    }

    /// Its place in [`KINDS`].
    pub fn kind(&self) -> usize {
        match self {
            Constraint::Distance { .. } => 0,
            Constraint::Horizontal { .. } => 1,
            Constraint::Vertical { .. } => 2,
            Constraint::Fixed { .. } => 3,
            Constraint::Angle { .. } => 4,
            Constraint::Parallel { .. } => 5,
            Constraint::Perpendicular { .. } => 6,
            Constraint::Equal { .. } => 7,
            Constraint::Coincident { .. } => 8,
            Constraint::OnLine { .. } => 9,
            Constraint::Circle { .. } => 10,
            Constraint::Arc { .. } => 11,
            Constraint::Tangent { .. } => 12,
            Constraint::Spline { .. } => 13,
        }
    }

    /// The points it names.
    pub fn points(&self) -> Vec<u32> {
        match self {
            Constraint::Distance { a, b, .. }
            | Constraint::Horizontal { a, b }
            | Constraint::Vertical { a, b }
            | Constraint::Coincident { a, b } => vec![*a, *b],
            Constraint::Fixed { a, .. } => vec![*a],
            Constraint::OnLine { p, a, b } => vec![*p, *a, *b],
            Constraint::Angle { a, b, c, d, .. }
            | Constraint::Parallel { a, b, c, d }
            | Constraint::Perpendicular { a, b, c, d }
            | Constraint::Equal { a, b, c, d } => vec![*a, *b, *c, *d],
            Constraint::Circle { centre, ring } | Constraint::Arc { centre, ring } => {
                std::iter::once(*centre)
                    .chain(ring.iter().copied())
                    .collect()
            }
            Constraint::Tangent { a, b, centre, on } => vec![*a, *b, *centre, *on],
            Constraint::Spline { through, ring, .. } => {
                through.iter().chain(ring.iter()).copied().collect()
            }
        }
    }

    /// The same constraint on other points: each index through `map`. What
    /// deleting or merging a sketch's points does to what names them.
    pub fn remapped(&self, map: &impl Fn(u32) -> u32) -> Self {
        let mut out = self.clone();
        match &mut out {
            Constraint::Distance { a, b, .. }
            | Constraint::Horizontal { a, b }
            | Constraint::Vertical { a, b }
            | Constraint::Coincident { a, b } => {
                *a = map(*a);
                *b = map(*b);
            }
            Constraint::Fixed { a, .. } => *a = map(*a),
            Constraint::OnLine { p, a, b } => {
                *p = map(*p);
                *a = map(*a);
                *b = map(*b);
            }
            Constraint::Angle { a, b, c, d, .. }
            | Constraint::Parallel { a, b, c, d }
            | Constraint::Perpendicular { a, b, c, d }
            | Constraint::Equal { a, b, c, d } => {
                *a = map(*a);
                *b = map(*b);
                *c = map(*c);
                *d = map(*d);
            }
            Constraint::Circle { centre, ring } | Constraint::Arc { centre, ring } => {
                *centre = map(*centre);
                for r in ring.iter_mut() {
                    *r = map(*r);
                }
            }
            Constraint::Tangent { a, b, centre, on } => {
                *a = map(*a);
                *b = map(*b);
                *centre = map(*centre);
                *on = map(*on);
            }
            Constraint::Spline { through, ring, .. } => {
                for v in through.iter_mut().chain(ring.iter_mut()) {
                    *v = map(*v);
                }
            }
        }
        out
    }

    /// Its number, for the ones that have one: a length, or an angle in
    /// radians.
    pub fn value(&self) -> Option<f64> {
        match self {
            Constraint::Distance { value, .. } | Constraint::Angle { value, .. } => Some(*value),
            _ => None,
        }
    }

    /// The same constraint with its number changed.
    pub fn with_value(&self, v: f64) -> Self {
        let mut out = self.clone();
        match &mut out {
            Constraint::Distance { value, .. } | Constraint::Angle { value, .. } => *value = v,
            _ => {}
        }
        out
    }

    /// From its tag, its points and its number, as a file stores it. `None`
    /// for a tag this build does not know, or too few points.
    pub fn from_parts(tag: &str, p: &[u32], value: f64, at: [f64; 2]) -> Option<Self> {
        let get = |i: usize| p.get(i).copied();
        Some(match tag {
            "distance" => Constraint::Distance {
                a: get(0)?,
                b: get(1)?,
                value,
            },
            "horizontal" => Constraint::Horizontal {
                a: get(0)?,
                b: get(1)?,
            },
            "vertical" => Constraint::Vertical {
                a: get(0)?,
                b: get(1)?,
            },
            "fixed" => Constraint::Fixed { a: get(0)?, at },
            "angle" => Constraint::Angle {
                a: get(0)?,
                b: get(1)?,
                c: get(2)?,
                d: get(3)?,
                value,
            },
            "parallel" => Constraint::Parallel {
                a: get(0)?,
                b: get(1)?,
                c: get(2)?,
                d: get(3)?,
            },
            "perpendicular" => Constraint::Perpendicular {
                a: get(0)?,
                b: get(1)?,
                c: get(2)?,
                d: get(3)?,
            },
            "equal" => Constraint::Equal {
                a: get(0)?,
                b: get(1)?,
                c: get(2)?,
                d: get(3)?,
            },
            "coincident" => Constraint::Coincident {
                a: get(0)?,
                b: get(1)?,
            },
            "on_line" => Constraint::OnLine {
                p: get(0)?,
                a: get(1)?,
                b: get(2)?,
            },
            "circle" | "arc" => {
                let centre = get(0)?;
                let ring = p.get(1..).filter(|r| r.len() >= 2)?.to_vec();
                if tag == "circle" {
                    Constraint::Circle { centre, ring }
                } else {
                    Constraint::Arc { centre, ring }
                }
            }
            "tangent" => Constraint::Tangent {
                a: get(0)?,
                b: get(1)?,
                centre: get(2)?,
                on: get(3)?,
            },
            // How many points it passes through, then those, then the ring;
            // `value` says whether it closes.
            "spline" => {
                let n = get(0)? as usize;
                let through = p.get(1..1 + n).filter(|t| t.len() >= 2)?.to_vec();
                let ring = p.get(1 + n..).unwrap_or(&[]).to_vec();
                Constraint::Spline {
                    through,
                    ring,
                    closed: value != 0.0,
                }
            }
            _ => return None,
        })
    }

    /// What `from_parts` reads back as this one: the points as it lists
    /// them, and the value.
    pub fn parts(&self) -> (Vec<u32>, f64) {
        match self {
            Constraint::Spline {
                through,
                ring,
                closed,
            } => (
                std::iter::once(through.len() as u32)
                    .chain(through.iter().copied())
                    .chain(ring.iter().copied())
                    .collect(),
                if *closed { 1.0 } else { 0.0 },
            ),
            other => (other.points(), other.value().unwrap_or(0.0)),
        }
    }

    /// Where a Fixed one holds its point.
    pub fn at(&self) -> [f64; 2] {
        match self {
            Constraint::Fixed { at, .. } => *at,
            _ => [0.0; 2],
        }
    }

    /// How far from holding it is, as residuals (zero when it holds).
    fn residuals(&self, p: &impl Fn(u32) -> DVec2, out: &mut Vec<f64>) {
        let line = |a: u32, b: u32| p(b) - p(a);
        let along = |angle: f64| DVec2::new(angle.cos(), angle.sin());
        match *self {
            Constraint::Coincident { a, b } => {
                let d = p(b) - p(a);
                out.extend([d.x, d.y]);
            }
            Constraint::OnLine { p: q, a, b } => {
                let u = line(a, b);
                out.push(u.perp_dot(p(q) - p(a)) / u.length().max(1e-12));
            }
            Constraint::Circle { centre, ref ring } => {
                let c = p(centre);
                let first = p(ring[0]) - c;
                let r = first.length();
                let start = first.y.atan2(first.x);
                let n = ring.len() as f64;
                for (i, v) in ring.iter().enumerate().skip(1) {
                    let want = c + along(start + std::f64::consts::TAU * i as f64 / n) * r;
                    let d = p(*v) - want;
                    out.extend([d.x, d.y]);
                }
            }
            Constraint::Arc { centre, ref ring } => {
                let c = p(centre);
                let first = p(ring[0]) - c;
                let last = p(*ring.last().expect("two or more")) - c;
                let r = first.length();
                let start = first.y.atan2(first.x);
                let sweep = (last.y.atan2(last.x) - start).rem_euclid(std::f64::consts::TAU);
                out.push(last.length() - r);
                let n = (ring.len() - 1) as f64;
                for (i, v) in ring.iter().enumerate().take(ring.len() - 1).skip(1) {
                    let want = c + along(start + sweep * i as f64 / n) * r;
                    let d = p(*v) - want;
                    out.extend([d.x, d.y]);
                }
            }
            Constraint::Tangent { a, b, centre, on } => {
                let u = line(a, b);
                let off = u.perp_dot(p(centre) - p(a)).abs() / u.length().max(1e-12);
                out.push(off - (p(on) - p(centre)).length());
            }
            Constraint::Spline {
                ref through,
                ref ring,
                closed,
            } => {
                let pts: Vec<DVec2> = through.iter().map(|v| p(*v)).collect();
                for (v, want) in ring.iter().zip(spline_ring(&pts, ring.len(), closed)) {
                    let d = p(*v) - want;
                    out.extend([d.x, d.y]);
                }
            }
            Constraint::Distance { a, b, value } => out.push(line(a, b).length() - value),
            Constraint::Horizontal { a, b } => out.push(line(a, b).y),
            Constraint::Vertical { a, b } => out.push(line(a, b).x),
            Constraint::Fixed { a, at } => {
                let d = p(a) - DVec2::from_array(at);
                out.extend([d.x, d.y]);
            }
            Constraint::Angle { a, b, c, d, value } => {
                let (u, v) = (line(a, b), line(c, d));
                let turn = u.perp_dot(v).atan2(u.dot(v));
                let diff = turn - value;
                // The nearer way round.
                out.push(
                    (diff + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                        - std::f64::consts::PI,
                );
            }
            Constraint::Parallel { a, b, c, d } => {
                let (u, v) = (line(a, b), line(c, d));
                out.push(u.perp_dot(v) / (u.length() * v.length()).max(1e-12));
            }
            Constraint::Perpendicular { a, b, c, d } => {
                let (u, v) = (line(a, b), line(c, d));
                out.push(u.dot(v) / (u.length() * v.length()).max(1e-12));
            }
            Constraint::Equal { a, b, c, d } => out.push(line(a, b).length() - line(c, d).length()),
        }
    }
}

/// How a solve ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solve {
    /// The constraints hold, to a hair.
    pub converged: bool,
    /// What is left of them: the residuals' length.
    pub error: f64,
}

/// Moves `points` until `constraints` hold, as little as it can. Points no
/// constraint names stay where they are; constraints naming points that are
/// not there are skipped.
pub fn solve(points: &mut [DVec2], constraints: &[Constraint]) -> Solve {
    let live: Vec<&Constraint> = constraints
        .iter()
        .filter(|c| c.points().iter().all(|i| (*i as usize) < points.len()))
        .collect();
    let mut vars: Vec<u32> = live.iter().flat_map(|c| c.points()).collect();
    vars.sort_unstable();
    vars.dedup();
    if vars.is_empty() {
        return Solve {
            converged: true,
            error: 0.0,
        };
    }
    let slot = |i: u32| vars.binary_search(&i).expect("a named point");
    let mut x: Vec<f64> = vars
        .iter()
        .flat_map(|i| {
            let p = points[*i as usize];
            [p.x, p.y]
        })
        .collect();
    let residuals = |x: &[f64]| {
        let p = |i: u32| {
            let s = slot(i);
            DVec2::new(x[2 * s], x[2 * s + 1])
        };
        let mut r = Vec::new();
        for c in &live {
            c.residuals(&p, &mut r);
        }
        r
    };
    let norm = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-3);
    let tolerance = scale * 1e-10;

    let n = x.len();
    let mut r = residuals(&x);
    let mut lambda = 1e-3;
    for _ in 0..200 {
        if norm(&r) <= tolerance {
            break;
        }
        // The Jacobian, by central differences: every residual is a few
        // lines of arithmetic, and n is small.
        let h = scale * 1e-7;
        let m = r.len();
        let mut jac = vec![0.0; m * n];
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let (rp, rm) = (residuals(&xp), residuals(&xm));
            for i in 0..m {
                jac[i * n + j] = (rp[i] - rm[i]) / (2.0 * h);
            }
        }
        // (JᵀJ + λ diag) δ = -Jᵀr
        let mut a = vec![0.0; n * n];
        let mut g = vec![0.0; n];
        for i in 0..m {
            for j in 0..n {
                let jij = jac[i * n + j];
                if jij == 0.0 {
                    continue;
                }
                g[j] -= jij * r[i];
                for k in 0..n {
                    a[j * n + k] += jij * jac[i * n + k];
                }
            }
        }
        let mut improved = false;
        for _ in 0..12 {
            let mut damped = a.clone();
            for j in 0..n {
                damped[j * n + j] += lambda * (1.0 + a[j * n + j]);
            }
            let Some(step) = gauss(damped, g.clone(), n) else {
                lambda *= 10.0;
                continue;
            };
            let trial: Vec<f64> = x.iter().zip(&step).map(|(a, b)| a + b).collect();
            let rt = residuals(&trial);
            if norm(&rt) < norm(&r) {
                x = trial;
                r = rt;
                lambda = (lambda / 3.0).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 4.0;
        }
        if !improved {
            break;
        }
    }
    for (k, i) in vars.iter().enumerate() {
        points[*i as usize] = DVec2::new(x[2 * k], x[2 * k + 1]);
    }
    let error = norm(&r);
    Solve {
        converged: error <= scale * 1e-6,
        error,
    }
}

/// Which points the constraints pin down completely: Fusion's black (here
/// the text colour) against its blue, for a point that can still move.
///
/// A point is pinned when no motion of the sketch that keeps every
/// constraint as it is moves it: when its two coordinates are zero in every
/// vector of the null space of the constraints' Jacobian, at the points as
/// they are. A sketch with nothing fixed is never pinned — the whole of it
/// can slide — which is what a fixed point, or a line drawn from the origin,
/// is for. Points no constraint names are free.
pub fn pinned(points: &[DVec2], constraints: &[Constraint]) -> Vec<bool> {
    let n = points.len() * 2;
    if n == 0 {
        return Vec::new();
    }
    let live: Vec<&Constraint> = constraints
        .iter()
        .filter(|c| c.points().iter().all(|i| (*i as usize) < points.len()))
        .collect();
    let residuals = |x: &[f64]| {
        let p = |i: u32| DVec2::new(x[2 * i as usize], x[2 * i as usize + 1]);
        let mut r = Vec::new();
        for c in &live {
            c.residuals(&p, &mut r);
        }
        r
    };
    let x: Vec<f64> = points.iter().flat_map(|p| [p.x, p.y]).collect();
    let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-3);
    let h = scale * 1e-7;
    let m = residuals(&x).len();
    // The Jacobian, a row per residual.
    let mut jac = vec![vec![0.0; n]; m];
    for j in 0..n {
        let (mut xp, mut xm) = (x.clone(), x.clone());
        xp[j] += h;
        xm[j] -= h;
        let (rp, rm) = (residuals(&xp), residuals(&xm));
        for i in 0..m {
            jac[i][j] = (rp[i] - rm[i]) / (2.0 * h);
        }
    }
    // Reduced row echelon form; the columns without a pivot are the free
    // directions, each the start of one null-space vector.
    let mut pivots: Vec<(usize, usize)> = Vec::new();
    let mut row = 0;
    for col in 0..n {
        let Some(best) = (row..m).max_by(|a, b| jac[*a][col].abs().total_cmp(&jac[*b][col].abs()))
        else {
            break;
        };
        if jac[best][col].abs() < 1e-9 {
            continue;
        }
        jac.swap(row, best);
        let lead = jac[row][col];
        for v in jac[row].iter_mut() {
            *v /= lead;
        }
        let pivot_row = jac[row].clone();
        for (other, r) in jac.iter_mut().enumerate() {
            if other != row {
                let f = r[col];
                if f != 0.0 {
                    for (v, p) in r.iter_mut().zip(&pivot_row) {
                        *v -= f * p;
                    }
                }
            }
        }
        pivots.push((row, col));
        row += 1;
        if row == m {
            break;
        }
    }
    let pivot_cols: std::collections::HashSet<usize> = pivots.iter().map(|p| p.1).collect();
    let mut moves = vec![false; n];
    for free in (0..n).filter(|c| !pivot_cols.contains(c)) {
        moves[free] = true;
        for (r, c) in &pivots {
            if jac[*r][free].abs() > 1e-7 {
                moves[*c] = true;
            }
        }
    }
    (0..points.len())
        .map(|i| !moves[2 * i] && !moves[2 * i + 1])
        .collect()
}

/// Solves `a x = b` (n × n) by elimination with partial pivoting.
fn gauss(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let pivot =
            (col..n).max_by(|i, j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[pivot * n + col].abs() < 1e-300 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..n {
            let f = a[row * n + col] / a[col * n + col];
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| a[row * n + k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row * n + row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rough_square() -> Vec<DVec2> {
        vec![
            DVec2::new(0.02, -0.01),
            DVec2::new(1.1, 0.05),
            DVec2::new(0.95, 0.9),
            DVec2::new(-0.05, 1.08),
        ]
    }

    #[test]
    fn a_rough_square_comes_out_square() {
        let mut p = rough_square();
        let c = vec![
            Constraint::Fixed {
                a: 0,
                at: [0.0, 0.0],
            },
            Constraint::Horizontal { a: 0, b: 1 },
            Constraint::Vertical { a: 1, b: 2 },
            Constraint::Horizontal { a: 2, b: 3 },
            Constraint::Vertical { a: 3, b: 0 },
            Constraint::Distance {
                a: 0,
                b: 1,
                value: 2.0,
            },
            Constraint::Equal {
                a: 0,
                b: 1,
                c: 1,
                d: 2,
            },
        ];
        let s = solve(&mut p, &c);
        assert!(s.converged, "{s:?}");
        let near = |a: DVec2, b: DVec2| (a - b).length() < 1e-7;
        assert!(near(p[0], DVec2::ZERO), "{p:?}");
        assert!(near(p[1], DVec2::new(2.0, 0.0)), "{p:?}");
        assert!(near(p[2], DVec2::new(2.0, 2.0)), "{p:?}");
        assert!(near(p[3], DVec2::new(0.0, 2.0)), "{p:?}");
    }

    #[test]
    fn an_angle_a_parallel_and_a_perpendicular_hold() {
        let mut p = vec![
            DVec2::ZERO,
            DVec2::new(1.0, 0.1),
            DVec2::new(0.0, 1.0),
            DVec2::new(0.8, 1.3),
            DVec2::new(2.0, 0.0),
            DVec2::new(2.1, 1.0),
        ];
        let c = vec![
            Constraint::Fixed {
                a: 0,
                at: [0.0, 0.0],
            },
            Constraint::Fixed {
                a: 1,
                at: [1.0, 0.0],
            },
            Constraint::Angle {
                a: 0,
                b: 1,
                c: 2,
                d: 3,
                value: 30f64.to_radians(),
            },
            Constraint::Perpendicular {
                a: 0,
                b: 1,
                c: 4,
                d: 5,
            },
        ];
        assert!(solve(&mut p, &c).converged);
        let u = p[3] - p[2];
        assert!((u.y.atan2(u.x).to_degrees() - 30.0).abs() < 1e-6);
        assert!(
            (p[5] - p[4]).x.abs() < 1e-7,
            "square to a level line is upright"
        );
        let mut q = vec![
            DVec2::ZERO,
            DVec2::new(1.0, 0.0),
            DVec2::new(0.0, 1.0),
            DVec2::new(1.0, 1.4),
        ];
        let c = vec![
            Constraint::Fixed {
                a: 0,
                at: [0.0, 0.0],
            },
            Constraint::Fixed {
                a: 1,
                at: [1.0, 0.0],
            },
            Constraint::Parallel {
                a: 0,
                b: 1,
                c: 2,
                d: 3,
            },
        ];
        assert!(solve(&mut q, &c).converged);
        assert!((q[3] - q[2]).y.abs() < 1e-7);
    }

    /// A rectangle drawn roughly: corners level and upright, two sides
    /// dimensioned, one corner on the origin.
    fn rectangle() -> (Vec<DVec2>, Vec<Constraint>) {
        let p = vec![
            DVec2::new(0.01, 0.02),
            DVec2::new(0.041, -0.003),
            DVec2::new(0.039, 0.021),
            DVec2::new(-0.002, 0.019),
        ];
        let c = vec![
            Constraint::Horizontal { a: 0, b: 1 },
            Constraint::Vertical { a: 1, b: 2 },
            Constraint::Horizontal { a: 2, b: 3 },
            Constraint::Vertical { a: 3, b: 0 },
            Constraint::Distance {
                a: 0,
                b: 1,
                value: 0.04,
            },
            Constraint::Distance {
                a: 1,
                b: 2,
                value: 0.02,
            },
        ];
        (p, c)
    }

    #[test]
    fn a_dimensioned_rectangle_is_pinned_only_once_it_is_tied_down() {
        let (mut p, mut c) = rectangle();
        assert!(solve(&mut p, &c).converged);
        // Its shape is set, but the whole of it can slide.
        assert!(pinned(&p, &c).iter().all(|f| !f), "{:?}", pinned(&p, &c));
        c.push(Constraint::Fixed {
            a: 0,
            at: [0.0, 0.0],
        });
        assert!(solve(&mut p, &c).converged);
        assert!(pinned(&p, &c).iter().all(|f| *f), "{:?}", pinned(&p, &c));
        assert!((p[2] - DVec2::new(0.04, 0.02)).length() < 1e-9, "{p:?}");
        // One side's dimension gone: the far corners are free again, the
        // tied one is not.
        c.remove(5);
        let f = pinned(&p, &c);
        assert!(f[0] && f[1] && !f[2] && !f[3], "{f:?}");
    }

    #[test]
    fn changing_a_dimension_moves_the_sketch_to_it() {
        let (mut p, mut c) = rectangle();
        c.push(Constraint::Fixed {
            a: 0,
            at: [0.0, 0.0],
        });
        assert!(solve(&mut p, &c).converged);
        c[4] = c[4].with_value(0.06);
        assert!(solve(&mut p, &c).converged);
        assert!((p[2] - DVec2::new(0.06, 0.02)).length() < 1e-9, "{p:?}");
    }

    /// A circle of eight round a centre: round, its radius dimensioned, a
    /// line made tangent to it and a second circle concentric.
    #[test]
    fn a_circle_is_round_takes_its_radius_and_a_tangent_touches_it() {
        let mut p: Vec<DVec2> = (0..8)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 8.0;
                DVec2::new(a.cos(), a.sin()) * (1.0 + 0.05 * (i % 3) as f64)
            })
            .collect();
        p.push(DVec2::new(0.05, -0.02)); // 8: the centre
        p.push(DVec2::new(-2.0, 1.3)); // 9, 10: a line
        p.push(DVec2::new(2.0, 1.2));
        p.push(DVec2::new(0.01, 0.01)); // 11: another centre
        let ring: Vec<u32> = (0..8).collect();
        let c = vec![
            Constraint::Fixed {
                a: 8,
                at: [0.0, 0.0],
            },
            Constraint::Circle {
                centre: 8,
                ring: ring.clone(),
            },
            Constraint::Distance {
                a: 8,
                b: 0,
                value: 0.5,
            },
            Constraint::Horizontal { a: 9, b: 10 },
            Constraint::Tangent {
                a: 9,
                b: 10,
                centre: 8,
                on: 0,
            },
            Constraint::Coincident { a: 8, b: 11 },
        ];
        let s = solve(&mut p, &c);
        assert!(s.converged, "{s:?}");
        for v in &ring {
            assert!((p[*v as usize].length() - 0.5).abs() < 1e-8, "{p:?}");
        }
        assert!((p[9].y.abs() - 0.5).abs() < 1e-8, "the line touches: {p:?}");
        assert!(p[11].length() < 1e-9, "concentric");
        // The ring is pinned but for turning about its centre — which is a
        // motion, so it is not pinned; the centre is.
        let f = pinned(&p, &c);
        assert!(f[8] && f[11]);
    }

    #[test]
    fn an_arc_keeps_its_ends_on_its_circle() {
        let mut p = vec![
            DVec2::new(1.0, 0.0),
            DVec2::new(0.75, 0.62),
            DVec2::new(0.1, 1.05),
            DVec2::new(-0.4, 0.95),
            DVec2::ZERO,
        ];
        let c = vec![
            Constraint::Fixed {
                a: 4,
                at: [0.0, 0.0],
            },
            Constraint::Fixed {
                a: 0,
                at: [1.0, 0.0],
            },
            Constraint::Arc {
                centre: 4,
                ring: vec![0, 1, 2, 3],
            },
        ];
        assert!(solve(&mut p, &c).converged);
        for v in 0..4 {
            assert!((p[v].length() - 1.0).abs() < 1e-8, "{p:?}");
        }
        let ang = |v: DVec2| v.y.atan2(v.x);
        let step = ang(p[1]) - ang(p[0]);
        assert!((ang(p[2]) - ang(p[1]) - step).abs() < 1e-7, "evenly: {p:?}");
    }

    #[test]
    fn a_new_constraint_survives_the_file_and_a_remap() {
        let c = Constraint::Circle {
            centre: 4,
            ring: vec![0, 1, 2],
        };
        let back = Constraint::from_parts(c.tag(), &c.points(), 0.0, [0.0; 2]);
        assert_eq!(back, Some(c.clone()));
        assert_eq!(
            c.remapped(&|i| i + 10),
            Constraint::Circle {
                centre: 14,
                ring: vec![10, 11, 12]
            }
        );
        for c in [
            Constraint::Coincident { a: 1, b: 2 },
            Constraint::OnLine { p: 1, a: 2, b: 3 },
            Constraint::Tangent {
                a: 0,
                b: 1,
                centre: 2,
                on: 3,
            },
        ] {
            assert_eq!(
                Constraint::from_parts(c.tag(), &c.points(), 0.0, [0.0; 2]),
                Some(c)
            );
        }
    }

    #[test]
    fn what_no_constraint_names_stays_put_and_a_contradiction_says_so() {
        let mut p = vec![DVec2::ZERO, DVec2::new(1.0, 0.0), DVec2::new(5.0, 5.0)];
        let s = solve(
            &mut p,
            &[Constraint::Distance {
                a: 0,
                b: 1,
                value: 3.0,
            }],
        );
        assert!(s.converged);
        assert_eq!(p[2], DVec2::new(5.0, 5.0));
        assert!(((p[1] - p[0]).length() - 3.0).abs() < 1e-8);
        let mut p = vec![DVec2::ZERO, DVec2::new(1.0, 0.0)];
        let s = solve(
            &mut p,
            &[
                Constraint::Distance {
                    a: 0,
                    b: 1,
                    value: 1.0,
                },
                Constraint::Distance {
                    a: 0,
                    b: 1,
                    value: 2.0,
                },
            ],
        );
        assert!(!s.converged);
    }
}
