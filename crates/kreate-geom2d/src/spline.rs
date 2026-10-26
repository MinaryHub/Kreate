//! B-splines and NURBS, as DXF's SPLINE holds them: a degree, control
//! points, a knot vector, and weights when it is rational.

use crate::DVec2;

#[derive(Clone, Debug, PartialEq)]
pub struct Spline {
    pub degree: usize,
    pub ctrl: Vec<DVec2>,
    /// `ctrl.len() + degree + 1` of them, not decreasing.
    pub knots: Vec<f64>,
    /// One per control point, or none for a plain B-spline.
    pub weights: Option<Vec<f64>>,
    /// The points it was drawn through, when it was drawn that way: what
    /// the person placed, and what editing it moves. Empty for a spline
    /// given by its control points.
    pub fit: Vec<DVec2>,
}

impl Spline {
    /// A clamped, uniform B-spline over `ctrl`: it starts at the first
    /// point and ends at the last, its parameter running over `[0, 1]`.
    pub fn clamped(degree: usize, ctrl: Vec<DVec2>) -> Spline {
        let n = ctrl.len();
        let degree = degree.min(n.saturating_sub(1)).max(1);
        let inner = n - degree - 1;
        let mut knots = vec![0.0; degree + 1];
        for i in 1..=inner {
            knots.push(i as f64 / (inner + 1) as f64);
        }
        knots.extend(std::iter::repeat_n(1.0, degree + 1));
        Spline {
            degree,
            ctrl,
            knots,
            weights: None,
            fit: Vec::new(),
        }
    }

    /// A cubic through `fit` in order — the curve a person draws by
    /// clicking points along it. Its knots are spaced by chord length, its
    /// ends are free (no bend at the ends), and it is solved for exactly,
    /// so the curve passes through every point.
    pub fn through(fit: &[DVec2]) -> Spline {
        let n = fit.len();
        if n < 3 {
            let mut s = Spline::clamped(1, fit.to_vec());
            s.fit = fit.to_vec();
            return s;
        }
        // Chord-length parameters for the fit points.
        let mut u = vec![0.0; n];
        for i in 1..n {
            u[i] = u[i - 1] + fit[i].distance(fit[i - 1]).max(1e-12);
        }
        let total = u[n - 1];
        for v in &mut u {
            *v /= total;
        }
        // Knots: clamped, with the fit parameters' interior ones as the
        // interior knots, plus two more for the free end conditions.
        let degree = 3;
        let mut knots = vec![0.0; degree + 1];
        knots.extend_from_slice(&u[1..n - 1]);
        knots.extend(std::iter::repeat_n(1.0, degree + 1));
        let m = n + 2; // control points
                       // Rows: one per fit point (the curve passes through it), and one
                       // at each end (zero second derivative there).
        let mut a = vec![0.0; m * m];
        let mut bx = vec![0.0; m];
        let mut by = vec![0.0; m];
        let probe = Spline {
            degree,
            ctrl: vec![DVec2::ZERO; m],
            knots: knots.clone(),
            weights: None,
            fit: Vec::new(),
        };
        for (row, (&t, p)) in u.iter().zip(fit).enumerate() {
            for (j, v) in probe.basis(t, 0).into_iter().enumerate() {
                a[row * m + j] = v;
            }
            bx[row] = p.x;
            by[row] = p.y;
        }
        for (row, t) in [(n, 0.0), (n + 1, 1.0)] {
            for (j, v) in probe.basis(t, 2).into_iter().enumerate() {
                a[row * m + j] = v;
            }
        }
        let (xs, ys) = match (solve(a.clone(), bx, m), solve(a, by, m)) {
            (Some(x), Some(y)) => (x, y),
            _ => return Spline::clamped(1, fit.to_vec()),
        };
        Spline {
            degree,
            ctrl: xs
                .into_iter()
                .zip(ys)
                .map(|(x, y)| DVec2::new(x, y))
                .collect(),
            knots,
            weights: None,
            fit: fit.to_vec(),
        }
    }

    /// The uniform Catmull–Rom curve through `through`, exactly: each span
    /// is the cubic Bézier between two of the points, its handles a sixth
    /// of the way to the neighbours, and the spans one after another are a
    /// B-spline whose inner knots are each three times over. What a sketch
    /// of format 1.4 drew a spline as, so its migration keeps its shape.
    /// Closed, it comes back round to the first point.
    pub fn catmull_rom(through: &[DVec2], closed: bool) -> Spline {
        let n = through.len();
        if n < 2 {
            return Spline::clamped(1, through.to_vec());
        }
        let spans = if closed { n } else { n - 1 };
        let at = |i: isize| -> DVec2 {
            if closed {
                through[i.rem_euclid(n as isize) as usize]
            } else {
                through[i.clamp(0, n as isize - 1) as usize]
            }
        };
        let mut ctrl = vec![at(0)];
        for k in 0..spans as isize {
            let (p0, p1, p2, p3) = (at(k - 1), at(k), at(k + 1), at(k + 2));
            ctrl.push(p1 + (p2 - p0) / 6.0);
            ctrl.push(p2 - (p3 - p1) / 6.0);
            ctrl.push(p2);
        }
        let mut knots = vec![0.0; 4];
        for k in 1..spans {
            let t = k as f64 / spans as f64;
            knots.extend([t, t, t]);
        }
        knots.extend([1.0; 4]);
        let mut fit = through.to_vec();
        if closed {
            fit.push(through[0]);
        }
        Spline {
            degree: 3,
            ctrl,
            knots,
            weights: None,
            fit,
        }
    }

    pub fn domain(&self) -> (f64, f64) {
        (
            self.knots[self.degree],
            self.knots[self.knots.len() - self.degree - 1],
        )
    }

    /// The knot span holding `t`.
    fn span(&self, t: f64) -> usize {
        let n = self.ctrl.len();
        let (_, hi) = self.domain();
        if t >= hi {
            // The last non-empty span.
            let mut k = n - 1;
            while k > self.degree && self.knots[k] >= hi {
                k -= 1;
            }
            return k;
        }
        let mut k = self.degree;
        while k + 1 < n && self.knots[k + 1] <= t {
            k += 1;
        }
        k
    }

    /// Every basis function's `d`-th derivative at `t`, one per control
    /// point. Cox–de Boor, by differentiating the recursion.
    fn basis(&self, t: f64, d: usize) -> Vec<f64> {
        let n = self.ctrl.len();
        let p = self.degree;
        let k = &self.knots;
        // Degree-0 functions, then raised a degree at a time.
        let s = self.span(t);
        let mut nfun = vec![0.0; n + p];
        nfun[s] = 1.0;
        let mut levels = vec![nfun.clone()];
        for q in 1..=p {
            let prev = &levels[q - 1];
            let mut next = vec![0.0; n + p];
            for i in 0..n + p - q {
                let mut v = 0.0;
                let d1 = k[i + q] - k[i];
                if d1 > 0.0 {
                    v += (t - k[i]) / d1 * prev[i];
                }
                let d2 = k[i + q + 1] - k[i + 1];
                if d2 > 0.0 {
                    v += (k[i + q + 1] - t) / d2 * prev[i + 1];
                }
                next[i] = v;
            }
            levels.push(next);
        }
        if d == 0 {
            return levels[p][..n].to_vec();
        }
        // The derivative of degree-p functions from degree p-1 ones,
        // applied d times down the levels.
        let mut out = levels[p - d.min(p)].clone();
        for q in (p - d.min(p) + 1)..=p {
            let mut next = vec![0.0; n + p];
            for i in 0..n + p - q {
                let mut v = 0.0;
                let d1 = k[i + q] - k[i];
                if d1 > 0.0 {
                    v += q as f64 / d1 * out[i];
                }
                let d2 = k[i + q + 1] - k[i + 1];
                if d2 > 0.0 {
                    v -= q as f64 / d2 * out[i + 1];
                }
                next[i] = v;
            }
            out = next;
        }
        out[..n].to_vec()
    }

    pub fn point(&self, t: f64) -> DVec2 {
        let b = self.basis(t, 0);
        match &self.weights {
            None => b.iter().zip(&self.ctrl).map(|(w, p)| *p * *w).sum(),
            Some(weights) => {
                let mut num = DVec2::ZERO;
                let mut den = 0.0;
                for ((bi, p), w) in b.iter().zip(&self.ctrl).zip(weights) {
                    num += *p * (bi * w);
                    den += bi * w;
                }
                if den.abs() < 1e-300 {
                    self.ctrl[0]
                } else {
                    num / den
                }
            }
        }
    }

    pub fn derivative(&self, t: f64) -> DVec2 {
        match &self.weights {
            None => self
                .basis(t, 1)
                .iter()
                .zip(&self.ctrl)
                .map(|(w, p)| *p * *w)
                .sum(),
            Some(_) => {
                // Rational: by central difference, which is accurate
                // enough for tangents and Newton steps.
                let (lo, hi) = self.domain();
                let h = (hi - lo) * 1e-7;
                let (a, b) = ((t - h).max(lo), (t + h).min(hi));
                (self.point(b) - self.point(a)) / (b - a)
            }
        }
    }
}

/// Dense Gaussian elimination with partial pivoting: the small systems a
/// spline fit makes.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let pivot =
            (col..n).max_by(|i, j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[pivot * n + col].abs() < 1e-14 {
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
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row * n + k] * x[k];
        }
        x[row] = s / a[row * n + row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clamped cubic over four points is their Bézier curve.
    #[test]
    fn a_clamped_cubic_on_four_points_is_their_bezier() {
        let p = [
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 2.0),
            DVec2::new(3.0, 2.0),
            DVec2::new(4.0, 0.0),
        ];
        let s = Spline::clamped(3, p.to_vec());
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let u = 1.0 - t;
            let want = p[0] * u.powi(3)
                + p[1] * (3.0 * u * u * t)
                + p[2] * (3.0 * u * t * t)
                + p[3] * t.powi(3);
            assert!(s.point(t).distance(want) < 1e-12, "{t}");
            let dwant = (p[1] - p[0]) * (3.0 * u * u)
                + (p[2] - p[1]) * (6.0 * u * t)
                + (p[3] - p[2]) * (3.0 * t * t);
            assert!(s.derivative(t).distance(dwant) < 1e-9, "derivative at {t}");
        }
    }

    /// A curve drawn through points passes through every one of them.
    #[test]
    fn a_spline_through_points_passes_through_them() {
        let fit = [
            DVec2::new(0.0, 0.0),
            DVec2::new(0.01, 0.01),
            DVec2::new(0.02, 0.0),
            DVec2::new(0.03, 0.01),
            DVec2::new(0.05, 0.0),
        ];
        let s = Spline::through(&fit);
        // Each fit point is somewhere on the curve: sample densely.
        let (lo, hi) = s.domain();
        for p in fit {
            let nearest = (0..=4000)
                .map(|i| s.point(lo + (hi - lo) * i as f64 / 4000.0).distance(p))
                .fold(f64::MAX, f64::min);
            assert!(nearest < 1e-5, "{p} is {nearest} off");
        }
        assert!(s.point(lo).distance(fit[0]) < 1e-12);
        assert!(s.point(hi).distance(fit[4]) < 1e-12);
    }

    /// The Catmull–Rom spline is the sketch's: at a quarter of each span
    /// it is where `kreate_scene::sketch::spline_ring` would put a point.
    #[test]
    fn a_catmull_rom_spline_passes_its_points_with_their_tangents() {
        let p = [
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(2.0, 0.0),
            DVec2::new(3.0, 1.0),
        ];
        let s = Spline::catmull_rom(&p, false);
        for (k, q) in p.iter().enumerate() {
            let t = k as f64 / 3.0;
            assert!(s.point(t).distance(*q) < 1e-12, "{k}");
        }
        // The Catmull–Rom formula at a quarter of the middle span.
        let t = 0.25;
        let (p0, p1, p2, p3) = (p[0], p[1], p[2], p[3]);
        let want = (p1 * 2.0
            + (p2 - p0) * t
            + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (t * t)
            + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * (t * t * t))
            * 0.5;
        let got = s.point((1.0 + t) / 3.0);
        assert!(got.distance(want) < 1e-12, "{got} against {want}");
        // Closed, it comes back.
        let c = Spline::catmull_rom(&p, true);
        let (lo, hi) = c.domain();
        assert!(c.point(lo).distance(c.point(hi)) < 1e-12);
    }

    /// Weights of one make a NURBS the plain B-spline.
    #[test]
    fn unit_weights_change_nothing() {
        let ctrl = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(2.0, -1.0),
            DVec2::new(3.0, 0.0),
            DVec2::new(4.0, 1.0),
        ];
        let plain = Spline::clamped(3, ctrl.clone());
        let mut rational = plain.clone();
        rational.weights = Some(vec![1.0; ctrl.len()]);
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            assert!(plain.point(t).distance(rational.point(t)) < 1e-12);
        }
    }
}
