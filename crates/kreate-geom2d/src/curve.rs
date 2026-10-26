//! The curves a drawing is made of.

use crate::prim::Prim;
use crate::spline::Spline;
use crate::{DVec2, POINT_EPS, TAU};

#[derive(Clone, Debug, PartialEq)]
pub enum Curve2D {
    Line {
        a: DVec2,
        b: DVec2,
    },
    Circle {
        c: DVec2,
        r: f64,
    },
    /// From `start` through `sweep` (radians): anticlockwise when `sweep`
    /// is positive, as DXF's ARC always is, and clockwise when negative —
    /// which an arc run backwards along a region's boundary needs. Writers
    /// for formats with only anticlockwise arcs turn it round.
    Arc {
        c: DVec2,
        r: f64,
        start: f64,
        sweep: f64,
    },
    /// Centred on `c`, `major` the half-axis from the centre, `ratio` the
    /// minor half-axis over the major, and the part from `start` through
    /// `sweep` of the ellipse's own angle — DXF's ELLIPSE.
    Ellipse {
        c: DVec2,
        major: DVec2,
        ratio: f64,
        start: f64,
        sweep: f64,
    },
    /// Vertices with the bulge of the segment that leaves each — DXF's
    /// LWPOLYLINE. Closed, the last vertex's bulge is the segment back to
    /// the first.
    Polyline {
        verts: Vec<(DVec2, f64)>,
        closed: bool,
    },
    Spline(Spline),
}

impl Curve2D {
    /// The parameter's range: see the crate notes.
    pub fn domain(&self) -> (f64, f64) {
        match self {
            Curve2D::Polyline { .. } => (0.0, self.prims().len() as f64),
            Curve2D::Spline(s) => s.domain(),
            _ => (0.0, 1.0),
        }
    }

    /// The exact pieces of a curve that has them — a line, a circle, an
    /// arc, a polyline — or none for an ellipse or a spline.
    pub fn prims(&self) -> Vec<Prim> {
        match self {
            Curve2D::Line { a, b } => vec![Prim::Line { a: *a, b: *b }],
            Curve2D::Circle { c, r } => vec![Prim::Arc {
                c: *c,
                r: *r,
                start: 0.0,
                sweep: TAU,
            }],
            Curve2D::Arc { c, r, start, sweep } => vec![Prim::Arc {
                c: *c,
                r: *r,
                start: *start,
                sweep: *sweep,
            }],
            Curve2D::Polyline { verts, closed } => {
                let n = verts.len();
                let segments = if *closed { n } else { n.saturating_sub(1) };
                (0..segments)
                    .map(|i| {
                        let (p, bulge) = verts[i];
                        let q = verts[(i + 1) % n].0;
                        Prim::from_bulge(p, q, bulge)
                    })
                    .collect()
            }
            Curve2D::Ellipse { .. } | Curve2D::Spline(_) => Vec::new(),
        }
    }

    /// For a curve made of prims: which one parameter `t` falls on, and the
    /// parameter within it.
    fn locate(&self, t: f64) -> Option<(Prim, f64)> {
        let prims = self.prims();
        if prims.is_empty() {
            return None;
        }
        match self {
            Curve2D::Polyline { .. } => {
                let i = (t.floor().max(0.0) as usize).min(prims.len() - 1);
                Some((prims[i], (t - i as f64).clamp(0.0, 1.0)))
            }
            _ => Some((prims[0], t)),
        }
    }

    fn ellipse_angle(start: f64, sweep: f64, t: f64) -> f64 {
        start + sweep * t
    }

    pub fn point(&self, t: f64) -> DVec2 {
        match self {
            Curve2D::Ellipse {
                c,
                major,
                ratio,
                start,
                sweep,
            } => {
                let a = Self::ellipse_angle(*start, *sweep, t);
                *c + *major * a.cos() + major.perp() * (*ratio * a.sin())
            }
            Curve2D::Spline(s) => s.point(t),
            _ => match self.locate(t) {
                Some((p, u)) => p.point(u),
                None => DVec2::ZERO,
            },
        }
    }

    /// The derivative by the parameter.
    pub fn derivative(&self, t: f64) -> DVec2 {
        match self {
            Curve2D::Ellipse {
                major,
                ratio,
                start,
                sweep,
                ..
            } => {
                let a = Self::ellipse_angle(*start, *sweep, t);
                (*major * -a.sin() + major.perp() * (*ratio * a.cos())) * *sweep
            }
            Curve2D::Spline(s) => s.derivative(t),
            _ => match self.locate(t) {
                Some((p, u)) => p.derivative(u),
                None => DVec2::ZERO,
            },
        }
    }

    pub fn start(&self) -> DVec2 {
        self.point(self.domain().0)
    }

    pub fn end(&self) -> DVec2 {
        self.point(self.domain().1)
    }

    /// Whether it comes back to where it began: a circle, a closed
    /// polyline, a full ellipse, a spline whose ends meet.
    pub fn is_closed(&self) -> bool {
        match self {
            Curve2D::Circle { .. } => true,
            Curve2D::Polyline { closed, .. } => *closed,
            Curve2D::Ellipse { sweep, .. } => sweep.abs() >= TAU - 1e-12,
            _ => self.start().distance(self.end()) < POINT_EPS,
        }
    }

    /// Its length: exact for prims, by Gauss–Legendre quadrature for the
    /// rest.
    pub fn length(&self) -> f64 {
        let prims = self.prims();
        if !prims.is_empty() {
            return prims.iter().map(Prim::length).sum();
        }
        let (lo, hi) = self.domain();
        let pieces = 64;
        let mut total = 0.0;
        for k in 0..pieces {
            let (a, b) = (
                lo + (hi - lo) * k as f64 / pieces as f64,
                lo + (hi - lo) * (k + 1) as f64 / pieces as f64,
            );
            total += gauss_legendre(|t| self.derivative(t).length(), a, b);
        }
        total
    }

    /// Points drawing it within `tol` of the true curve, from start to end.
    pub fn tessellate(&self, tol: f64) -> Vec<DVec2> {
        let mut out = Vec::new();
        let prims = self.prims();
        if !prims.is_empty() {
            for p in &prims {
                p.tessellate(tol, &mut out);
            }
            return out;
        }
        // Adaptive: halve a span until its midpoint is within `tol` of the
        // chord, which for curves this smooth bounds the error.
        let (lo, hi) = self.domain();
        let start_spans = 16;
        out.push(self.point(lo));
        for k in 0..start_spans {
            let a = lo + (hi - lo) * k as f64 / start_spans as f64;
            let b = lo + (hi - lo) * (k + 1) as f64 / start_spans as f64;
            self.refine(a, b, tol, 0, &mut out);
        }
        out
    }

    fn refine(&self, a: f64, b: f64, tol: f64, depth: u32, out: &mut Vec<DVec2>) {
        let (pa, pb) = (self.point(a), self.point(b));
        let m = (a + b) / 2.0;
        let pm = self.point(m);
        // The midpoint's distance from the chord, and a quarter point's
        // too, so an S-bend across the span is not missed.
        let off = |p: DVec2| {
            let d = pb - pa;
            let len = d.length();
            if len < POINT_EPS {
                p.distance(pa)
            } else {
                (p - pa).perp_dot(d).abs() / len
            }
        };
        let q = self.point((a + m) / 2.0);
        if depth >= 20 || (off(pm) <= tol && off(q) <= tol) {
            out.push(pb);
            return;
        }
        self.refine(a, m, tol, depth + 1, out);
        self.refine(m, b, tol, depth + 1, out);
    }

    /// The parameter of the point on the curve nearest `p`, and that point.
    pub fn closest(&self, p: DVec2) -> (f64, DVec2) {
        let prims = self.prims();
        if !prims.is_empty() {
            let mut best = (0.0, prims[0].point(0.0), f64::MAX);
            for (i, prim) in prims.iter().enumerate() {
                let (u, q) = prim.closest(p);
                let d = q.distance_squared(p);
                if d < best.2 {
                    let t = match self {
                        Curve2D::Polyline { .. } => i as f64 + u,
                        _ => u,
                    };
                    best = (t, q, d);
                }
            }
            return (best.0, best.1);
        }
        // Sample, then Newton on the squared distance's derivative.
        let (lo, hi) = self.domain();
        let n = 256;
        let mut t = lo;
        let mut best = f64::MAX;
        for i in 0..=n {
            let s = lo + (hi - lo) * i as f64 / n as f64;
            let d = self.point(s).distance_squared(p);
            if d < best {
                best = d;
                t = s;
            }
        }
        for _ in 0..20 {
            let c = self.point(t);
            let d1 = self.derivative(t);
            let h = (hi - lo) * 1e-6;
            let d2 =
                (self.derivative((t + h).min(hi)) - self.derivative((t - h).max(lo))) / (2.0 * h);
            let f = (c - p).dot(d1);
            let df = d1.dot(d1) + (c - p).dot(d2);
            if df.abs() < 1e-300 {
                break;
            }
            let next = (t - f / df).clamp(lo, hi);
            if (next - t).abs() < 1e-14 * (hi - lo) {
                t = next;
                break;
            }
            t = next;
        }
        (t, self.point(t))
    }

    /// The axis-aligned box that holds it: lo, hi.
    pub fn bounds(&self) -> (DVec2, DVec2) {
        let prims = self.prims();
        if !prims.is_empty() {
            return prims.iter().map(Prim::bounds).fold(
                (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)),
                |(lo, hi), (a, b)| (lo.min(a), hi.max(b)),
            );
        }
        let pts = self.tessellate(self.size_hint() * 1e-6);
        pts.iter().fold(
            (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        )
    }

    /// A rough size, for tolerances relative to the curve.
    fn size_hint(&self) -> f64 {
        match self {
            Curve2D::Ellipse { major, .. } => major.length(),
            Curve2D::Spline(s) => {
                let (lo, hi) = s.ctrl.iter().fold(
                    (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)),
                    |(lo, hi), p| (lo.min(*p), hi.max(*p)),
                );
                (hi - lo).length().max(POINT_EPS)
            }
            _ => 1.0,
        }
    }
}

/// Five-point Gauss–Legendre over `[a, b]`.
fn gauss_legendre(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    const X: [f64; 5] = [
        0.0,
        -0.538_469_310_105_683_1,
        0.538_469_310_105_683_1,
        -0.906_179_845_938_664,
        0.906_179_845_938_664,
    ];
    const W: [f64; 5] = [
        0.568_888_888_888_888_9,
        0.478_628_670_499_366_5,
        0.478_628_670_499_366_5,
        0.236_926_885_056_189_1,
        0.236_926_885_056_189_1,
    ];
    let (m, h) = ((a + b) / 2.0, (b - a) / 2.0);
    X.iter().zip(W).map(|(x, w)| w * f(m + h * x)).sum::<f64>() * h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    #[test]
    fn lengths_are_exact_where_they_can_be() {
        assert!(
            (Curve2D::Circle {
                c: v(1.0, 1.0),
                r: 2.0
            }
            .length()
                - 4.0 * PI)
                .abs()
                < 1e-12
        );
        let arc = Curve2D::Arc {
            c: v(0.0, 0.0),
            r: 1.0,
            start: 0.0,
            sweep: PI / 2.0,
        };
        assert!((arc.length() - PI / 2.0).abs() < 1e-12);
        // An ellipse with ratio one is a circle.
        let e = Curve2D::Ellipse {
            c: v(0.0, 0.0),
            major: v(3.0, 0.0),
            ratio: 1.0,
            start: 0.0,
            sweep: TAU,
        };
        assert!((e.length() - 6.0 * PI).abs() < 1e-9, "{}", e.length());
    }

    /// A bulge of one is a half circle, anticlockwise; minus one the other
    /// half.
    #[test]
    fn a_bulge_of_one_is_a_half_circle() {
        let ccw = Curve2D::Polyline {
            verts: vec![(v(0.0, 0.0), 1.0), (v(2.0, 0.0), 0.0)],
            closed: false,
        };
        // Anticlockwise from (0,0) to (2,0) round (1,0) passes below.
        assert!(
            ccw.point(0.5).distance(v(1.0, -1.0)) < 1e-12,
            "{}",
            ccw.point(0.5)
        );
        assert!((ccw.length() - PI).abs() < 1e-12);
        let cw = Curve2D::Polyline {
            verts: vec![(v(0.0, 0.0), -1.0), (v(2.0, 0.0), 0.0)],
            closed: false,
        };
        assert!(
            cw.point(0.5).distance(v(1.0, 1.0)) < 1e-12,
            "{}",
            cw.point(0.5)
        );
        // A small bulge stays near the chord, on the right side.
        let small = Curve2D::Polyline {
            verts: vec![(v(0.0, 0.0), 0.1), (v(2.0, 0.0), 0.0)],
            closed: false,
        };
        let mid = small.point(0.5);
        assert!(mid.y < 0.0 && mid.y > -0.3, "{mid}");
        assert!(small.end().distance(v(2.0, 0.0)) < 1e-12);
    }

    #[test]
    fn the_nearest_point_is_on_the_curve_and_nearest() {
        let arc = Curve2D::Arc {
            c: v(0.0, 0.0),
            r: 1.0,
            start: 0.0,
            sweep: PI / 2.0,
        };
        let (_, q) = arc.closest(v(2.0, 2.0));
        assert!(q.distance(v(1.0, 1.0).normalize()) < 1e-12);
        // Past its end, the end.
        let (t, q) = arc.closest(v(-1.0, -0.1));
        assert!(q.distance(v(1.0, 0.0)) < 1e-12 || q.distance(v(0.0, 1.0)) < 1e-12);
        assert!(t == 0.0 || t == 1.0);
        // An ellipse, by Newton: the end of its minor axis.
        let e = Curve2D::Ellipse {
            c: v(0.0, 0.0),
            major: v(2.0, 0.0),
            ratio: 0.5,
            start: 0.0,
            sweep: TAU,
        };
        let (_, q) = e.closest(v(0.0, 3.0));
        assert!(q.distance(v(0.0, 1.0)) < 1e-9, "{q}");
    }

    /// The drawing of a curve strays from it by no more than asked.
    #[test]
    fn a_tessellation_stays_within_its_tolerance() {
        let tol = 1e-4;
        for curve in [
            Curve2D::Circle {
                c: v(0.0, 0.0),
                r: 0.05,
            },
            Curve2D::Ellipse {
                c: v(0.0, 0.0),
                major: v(0.08, 0.02),
                ratio: 0.3,
                start: 0.3,
                sweep: 4.0,
            },
            Curve2D::Spline(Spline::through(&[
                v(0.0, 0.0),
                v(0.02, 0.03),
                v(0.05, -0.01),
                v(0.08, 0.02),
            ])),
        ] {
            let pts = curve.tessellate(tol);
            let (lo, hi) = curve.domain();
            for i in 0..=2000 {
                let p = curve.point(lo + (hi - lo) * i as f64 / 2000.0);
                // The distance from the curve's point to the polyline.
                let d = pts
                    .windows(2)
                    .map(|w| Prim::Line { a: w[0], b: w[1] }.closest(p).1.distance(p))
                    .fold(f64::MAX, f64::min);
                assert!(d <= tol * 1.05, "{curve:?}: {d}");
            }
            assert!(pts.len() < 400, "{} points", pts.len());
        }
    }

    #[test]
    fn bounds_hold_an_arcs_extremes() {
        let arc = Curve2D::Arc {
            c: v(0.0, 0.0),
            r: 1.0,
            start: -0.5,
            sweep: 1.0,
        };
        let (lo, hi) = arc.bounds();
        assert!((hi.x - 1.0).abs() < 1e-12, "the east extreme: {hi}");
        assert!(lo.x > 0.8);
    }
}
