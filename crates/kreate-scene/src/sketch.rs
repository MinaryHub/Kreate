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
}

/// The kinds, in the order the operators number them.
pub const KINDS: [&str; 8] = [
    "distance",
    "horizontal",
    "vertical",
    "fixed",
    "angle",
    "parallel",
    "perpendicular",
    "equal",
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
        }
    }

    /// The points it names.
    pub fn points(&self) -> Vec<u32> {
        match *self {
            Constraint::Distance { a, b, .. }
            | Constraint::Horizontal { a, b }
            | Constraint::Vertical { a, b } => vec![a, b],
            Constraint::Fixed { a, .. } => vec![a],
            Constraint::Angle { a, b, c, d, .. }
            | Constraint::Parallel { a, b, c, d }
            | Constraint::Perpendicular { a, b, c, d }
            | Constraint::Equal { a, b, c, d } => vec![a, b, c, d],
        }
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
            _ => return None,
        })
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
        match *self {
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
