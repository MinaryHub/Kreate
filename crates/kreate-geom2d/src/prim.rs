//! The two pieces every exact curve breaks into: a straight segment and a
//! circular arc. A circle is an arc of a full turn, and a polyline is its
//! segments, each one or the other by its bulge.

use crate::{DVec2, POINT_EPS, TAU};

/// A line segment or a circular arc, parameterised over `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prim {
    Line {
        a: DVec2,
        b: DVec2,
    },
    /// From `start` (radians) through `sweep`: positive anticlockwise,
    /// negative clockwise, `±TAU` a whole circle.
    Arc {
        c: DVec2,
        r: f64,
        start: f64,
        sweep: f64,
    },
}

impl Prim {
    /// The arc a polyline segment from `p` to `q` with `bulge` makes, or the
    /// line when the bulge is nothing. A bulge is the tangent of a quarter
    /// of the arc's sweep, positive anticlockwise — DXF's LWPOLYLINE.
    pub fn from_bulge(p: DVec2, q: DVec2, bulge: f64) -> Prim {
        let chord = q - p;
        let len = chord.length();
        if bulge.abs() < 1e-12 || len < POINT_EPS {
            return Prim::Line { a: p, b: q };
        }
        let sweep = 4.0 * bulge.atan();
        // The centre sits on the chord's perpendicular bisector, at the
        // distance that makes the chord subtend `sweep`.
        let r = len / (2.0 * (sweep / 2.0).sin()).abs();
        let mid = (p + q) / 2.0;
        let normal = chord.perp() / len;
        let h = (r * r - (len / 2.0).powi(2)).max(0.0).sqrt();
        // Anticlockwise and less than half a turn: the centre is to the
        // left of the chord.
        let left = (sweep > 0.0) == (sweep.abs() < std::f64::consts::PI);
        let c = mid + normal * if left { h } else { -h };
        let start = (p - c).to_angle();
        Prim::Arc { c, r, start, sweep }
    }

    pub fn point(&self, t: f64) -> DVec2 {
        match *self {
            Prim::Line { a, b } => a + (b - a) * t,
            Prim::Arc { c, r, start, sweep } => c + DVec2::from_angle(start + sweep * t) * r,
        }
    }

    /// The derivative by the parameter.
    pub fn derivative(&self, t: f64) -> DVec2 {
        match *self {
            Prim::Line { a, b } => b - a,
            Prim::Arc {
                r, start, sweep, ..
            } => DVec2::from_angle(start + sweep * t).perp() * (r * sweep),
        }
    }

    pub fn length(&self) -> f64 {
        match *self {
            Prim::Line { a, b } => a.distance(b),
            Prim::Arc { r, sweep, .. } => r * sweep.abs(),
        }
    }

    /// The parameter of the point on the prim nearest `p`, and that point.
    pub fn closest(&self, p: DVec2) -> (f64, DVec2) {
        match *self {
            Prim::Line { a, b } => {
                let d = b - a;
                let len2 = d.length_squared();
                let t = if len2 < POINT_EPS * POINT_EPS {
                    0.0
                } else {
                    ((p - a).dot(d) / len2).clamp(0.0, 1.0)
                };
                (t, self.point(t))
            }
            Prim::Arc { c, .. } => {
                let t = match self.param_of_angle((p - c).to_angle()) {
                    Some(t) => t,
                    // Outside the arc's span: whichever end is nearer.
                    None => {
                        if self.point(0.0).distance_squared(p)
                            <= self.point(1.0).distance_squared(p)
                        {
                            0.0
                        } else {
                            1.0
                        }
                    }
                };
                (t, self.point(t))
            }
        }
    }

    /// For an arc, the parameter at which it passes the direction `angle`
    /// from its centre, if it does within `[0, 1]` (with a sliver of
    /// tolerance at the ends). Lines have none.
    pub fn param_of_angle(&self, angle: f64) -> Option<f64> {
        let Prim::Arc {
            r, start, sweep, ..
        } = *self
        else {
            return None;
        };
        // How far round from the start, in the arc's own direction.
        let along = if sweep >= 0.0 {
            (angle - start).rem_euclid(TAU)
        } else {
            (start - angle).rem_euclid(TAU)
        };
        let span = sweep.abs();
        if span >= TAU - 1e-12 {
            return Some(along / TAU);
        }
        // A point exactly at the start can come back as a full turn.
        let slack = POINT_EPS / r.max(POINT_EPS);
        if along <= span + slack {
            Some((along / span).min(1.0))
        } else if along >= TAU - slack {
            Some(0.0)
        } else {
            None
        }
    }

    /// The axis-aligned box that holds it: lo, hi.
    pub fn bounds(&self) -> (DVec2, DVec2) {
        match *self {
            Prim::Line { a, b } => (a.min(b), a.max(b)),
            Prim::Arc { c, r, .. } => {
                let (mut lo, mut hi) = (self.point(0.0), self.point(0.0));
                let e = self.point(1.0);
                lo = lo.min(e);
                hi = hi.max(e);
                // The extremes it passes: east, north, west, south.
                for k in 0..4 {
                    let a = k as f64 * std::f64::consts::FRAC_PI_2;
                    if self.param_of_angle(a).is_some() {
                        let q = c + DVec2::from_angle(a) * r;
                        lo = lo.min(q);
                        hi = hi.max(q);
                    }
                }
                (lo, hi)
            }
        }
    }

    /// Points drawing it within `tol` of the true curve, ends included.
    pub fn tessellate(&self, tol: f64, out: &mut Vec<DVec2>) {
        match *self {
            Prim::Line { a, b } => {
                if out.last() != Some(&a) {
                    out.push(a);
                }
                out.push(b);
            }
            Prim::Arc { r, sweep, .. } => {
                // A chord over angle θ strays r(1 - cos θ/2) from the arc.
                let tol = tol.max(r * 1e-9);
                let step = if tol >= r {
                    std::f64::consts::PI
                } else {
                    2.0 * (1.0 - tol / r).acos()
                };
                let n = ((sweep.abs() / step).ceil() as usize).clamp(1, 4096);
                let first = self.point(0.0);
                if out.last() != Some(&first) {
                    out.push(first);
                }
                for i in 1..=n {
                    out.push(self.point(i as f64 / n as f64));
                }
            }
        }
    }
}
