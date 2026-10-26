//! What drafting does to a curve: take part of it, offset it, round the
//! corner between two.

use crate::curve::Curve2D;
use crate::intersect::{prim_prim, Intersection};
use crate::prim::Prim;
use crate::spline::Spline;
use crate::{DVec2, POINT_EPS, TAU};

impl Curve2D {
    /// The part from parameter `t0` to `t1`, as a curve of the same kind
    /// where there is one (an arc of a circle, a shorter line, the segments
    /// of a polyline between). On a closed curve `t1 < t0` goes the long
    /// way round, through the seam — what a trim between two crossings of
    /// a circle leaves.
    pub fn sub(&self, t0: f64, t1: f64) -> Curve2D {
        match self {
            Curve2D::Line { .. } => Curve2D::Line {
                a: self.point(t0),
                b: self.point(t1),
            },
            Curve2D::Circle { c, r } => {
                let span = if t1 > t0 { t1 - t0 } else { t1 + 1.0 - t0 };
                Curve2D::Arc {
                    c: *c,
                    r: *r,
                    start: (TAU * t0).rem_euclid(TAU),
                    sweep: TAU * span,
                }
            }
            Curve2D::Arc { c, r, start, sweep } => Curve2D::Arc {
                c: *c,
                r: *r,
                start: start + sweep * t0,
                sweep: sweep * (t1 - t0),
            },
            Curve2D::Ellipse {
                c,
                major,
                ratio,
                start,
                sweep,
            } => {
                let closed = sweep.abs() >= TAU - 1e-12;
                let span = if t1 > t0 || !closed {
                    t1 - t0
                } else {
                    t1 + 1.0 - t0
                };
                Curve2D::Ellipse {
                    c: *c,
                    major: *major,
                    ratio: *ratio,
                    start: start + sweep * t0,
                    sweep: sweep * span,
                }
            }
            Curve2D::Polyline { closed, .. } => sub_polyline(self, *closed, t0, t1),
            Curve2D::Spline(s) => Curve2D::Spline(s.sub(t0, t1)),
        }
    }

    /// The same curve run the other way.
    pub fn reversed(&self) -> Curve2D {
        match self {
            Curve2D::Line { a, b } => Curve2D::Line { a: *b, b: *a },
            Curve2D::Circle { c, r } => Curve2D::Arc {
                c: *c,
                r: *r,
                start: 0.0,
                sweep: -TAU,
            },
            Curve2D::Arc { c, r, start, sweep } => Curve2D::Arc {
                c: *c,
                r: *r,
                start: start + sweep,
                sweep: -sweep,
            },
            Curve2D::Ellipse {
                c,
                major,
                ratio,
                start,
                sweep,
            } => Curve2D::Ellipse {
                c: *c,
                major: *major,
                ratio: *ratio,
                start: start + sweep,
                sweep: -sweep,
            },
            Curve2D::Polyline { verts, closed } => {
                // The segment that arrived at a vertex now leaves it, its
                // bulge negated. Closed, the first vertex stays first and
                // the way back to it from the old last one leads.
                let n = verts.len();
                let mut out: Vec<(DVec2, f64)> = Vec::with_capacity(n);
                if *closed && n > 0 {
                    out.push((verts[0].0, -verts[n - 1].1));
                    for k in (1..n).rev() {
                        out.push((verts[k].0, -verts[k - 1].1));
                    }
                } else {
                    for k in (0..n).rev() {
                        let bulge = if k > 0 { -verts[k - 1].1 } else { 0.0 };
                        out.push((verts[k].0, bulge));
                    }
                }
                Curve2D::Polyline {
                    verts: out,
                    closed: *closed,
                }
            }
            Curve2D::Spline(s) => Curve2D::Spline(s.reversed()),
        }
    }

    /// The curve `distance` to the left of its direction of travel (to the
    /// right when negative), or `None` when nothing is left of it — a
    /// circle offset inward past its centre. Lines, arcs and circles are
    /// exact; a polyline's segments are offset and their neighbours
    /// extended or cut to meet, as a drafting program does; an ellipse or a
    /// spline becomes a spline through offset points, within `tol`.
    pub fn offset(&self, distance: f64, tol: f64) -> Option<Curve2D> {
        match self {
            Curve2D::Line { a, b } => {
                let n = (*b - *a).normalize_or_zero().perp();
                Some(Curve2D::Line {
                    a: *a + n * distance,
                    b: *b + n * distance,
                })
            }
            // Anticlockwise, the left is the inside.
            Curve2D::Circle { c, r } => (r - distance > POINT_EPS).then_some(Curve2D::Circle {
                c: *c,
                r: r - distance,
            }),
            Curve2D::Arc { c, r, start, sweep } => {
                let r2 = r - distance * sweep.signum();
                (r2 > POINT_EPS).then_some(Curve2D::Arc {
                    c: *c,
                    r: r2,
                    start: *start,
                    sweep: *sweep,
                })
            }
            Curve2D::Polyline { closed, .. } => offset_polyline(self, *closed, distance),
            Curve2D::Ellipse { .. } | Curve2D::Spline(_) => {
                let pts = self.tessellate(tol / 4.0);
                if pts.len() < 2 {
                    return None;
                }
                let (lo, hi) = self.domain();
                let n = (pts.len() / 4).clamp(8, 200);
                let fit: Vec<DVec2> = (0..=n)
                    .map(|i| {
                        let t = lo + (hi - lo) * i as f64 / n as f64;
                        let d = self.derivative(t).normalize_or_zero();
                        self.point(t) + d.perp() * distance
                    })
                    .collect();
                Some(Curve2D::Spline(Spline::through(&fit)))
            }
        }
    }
}

/// The bulge of a prim, for writing it back as a polyline segment.
fn bulge_of(p: &Prim) -> f64 {
    match *p {
        Prim::Line { .. } => 0.0,
        Prim::Arc { sweep, .. } => (sweep / 4.0).tan(),
    }
}

fn sub_prim(p: &Prim, t0: f64, t1: f64) -> Prim {
    match *p {
        Prim::Line { .. } => Prim::Line {
            a: p.point(t0),
            b: p.point(t1),
        },
        Prim::Arc { c, r, start, sweep } => Prim::Arc {
            c,
            r,
            start: start + sweep * t0,
            sweep: sweep * (t1 - t0),
        },
    }
}

fn sub_polyline(curve: &Curve2D, closed: bool, t0: f64, t1: f64) -> Curve2D {
    let prims = curve.prims();
    let n = prims.len();
    if n == 0 {
        return curve.clone();
    }
    let total = n as f64;
    // Through the seam on a closed one: as far as the end, then from 0.
    let end = if t1 < t0 && closed { t1 + total } else { t1 };
    let mut out: Vec<(DVec2, f64)> = Vec::new();
    let mut t = t0;
    while t < end - 1e-12 {
        let i = (t.floor() as usize) % n;
        let seg_start = t.floor();
        let local0 = t - seg_start;
        let local1 = (end - seg_start).min(1.0);
        let piece = sub_prim(&prims[i], local0, local1);
        out.push((piece.point(0.0), bulge_of(&piece)));
        t = seg_start + 1.0;
        if local1 < 1.0 {
            break;
        }
    }
    out.push((curve.point(t1), 0.0));
    Curve2D::Polyline {
        verts: out,
        closed: false,
    }
}

fn offset_polyline(curve: &Curve2D, closed: bool, distance: f64) -> Option<Curve2D> {
    let prims = curve.prims();
    if prims.is_empty() {
        return None;
    }
    let offset: Vec<Prim> = prims
        .iter()
        .filter_map(|p| match *p {
            Prim::Line { a, b } => {
                let n = (b - a).normalize_or_zero().perp();
                Some(Prim::Line {
                    a: a + n * distance,
                    b: b + n * distance,
                })
            }
            Prim::Arc { c, r, start, sweep } => {
                let r2 = r - distance * sweep.signum();
                (r2 > POINT_EPS).then_some(Prim::Arc {
                    c,
                    r: r2,
                    start,
                    sweep,
                })
            }
        })
        .collect();
    if offset.len() != prims.len() {
        return None;
    }
    // Where each pair of neighbours meets: their ends where they already
    // touch, else the crossing of the two extended, else a straight join.
    let n = offset.len();
    let joints = if closed { n } else { n - 1 };
    let mut corner: Vec<DVec2> = Vec::with_capacity(joints);
    for i in 0..joints {
        let (a, b) = (offset[i], offset[(i + 1) % n]);
        let (ea, sb) = (a.point(1.0), b.point(0.0));
        if ea.distance(sb) < POINT_EPS {
            corner.push(ea);
            continue;
        }
        let meet = extended_meet(&a, &b, ea, sb);
        corner.push(meet.unwrap_or((ea + sb) / 2.0));
    }
    // Rebuild: each segment runs from the corner before it to the one
    // after, keeping its bulge's sense (an arc's centre and radius stay;
    // its ends move along it).
    let mut verts: Vec<(DVec2, f64)> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let start = if i == 0 && !closed {
            offset[0].point(0.0)
        } else {
            corner[(i + joints - 1) % joints.max(1)]
        };
        let end = if i == n - 1 && !closed {
            offset[n - 1].point(1.0)
        } else {
            corner[i % joints.max(1)]
        };
        let bulge = match offset[i] {
            Prim::Line { .. } => 0.0,
            Prim::Arc { c, sweep, .. } => {
                let a0 = (start - c).to_angle();
                let a1 = (end - c).to_angle();
                let mut s = a1 - a0;
                if sweep > 0.0 {
                    s = s.rem_euclid(TAU);
                } else {
                    s = -(-s).rem_euclid(TAU);
                }
                (s / 4.0).tan()
            }
        };
        verts.push((start, bulge));
        if i == n - 1 && !closed {
            verts.push((end, 0.0));
        }
    }
    Some(Curve2D::Polyline { verts, closed })
}

/// Where two prims would meet if each ran on past its end: lines as
/// infinite lines, arcs as whole circles. The meeting nearest both ends.
fn extended_meet(a: &Prim, b: &Prim, ea: DVec2, sb: DVec2) -> Option<DVec2> {
    let grow = |p: &Prim| -> Prim {
        match *p {
            Prim::Line { a, b } => {
                let d = (b - a) * 1e6;
                Prim::Line { a: a - d, b: b + d }
            }
            Prim::Arc { c, r, .. } => Prim::Arc {
                c,
                r,
                start: 0.0,
                sweep: TAU,
            },
        }
    };
    prim_prim(&grow(a), &grow(b))
        .into_iter()
        .filter_map(|h| match h {
            Intersection::Point { p, .. } => Some(p),
            _ => None,
        })
        .min_by(|x, y| {
            (x.distance(ea) + x.distance(sb)).total_cmp(&(y.distance(ea) + y.distance(sb)))
        })
}

/// A round corner between two curves.
#[derive(Clone, Debug, PartialEq)]
pub struct Fillet {
    /// The rounding arc, anticlockwise.
    pub arc: Curve2D,
    /// Where it touches the first curve, as that curve's parameter.
    pub ta: f64,
    /// Where it touches the second.
    pub tb: f64,
}

/// The arc of radius `r` tangent to `a` and `b`, in the corner nearest the
/// points `near_a` on `a` and `near_b` on `b` — where the user clicked, as
/// AutoCAD's FILLET asks. Lines and arcs; `None` for curves that cannot be
/// rounded this way, or a radius that does not fit.
pub fn fillet(a: &Curve2D, near_a: DVec2, b: &Curve2D, near_b: DVec2, r: f64) -> Option<Fillet> {
    let (pa, pb) = (a.prims(), b.prims());
    if pa.len() != 1 || pb.len() != 1 || r <= POINT_EPS {
        return None;
    }
    let (pa, pb) = (pa[0], pb[0]);
    // The centre lies at distance r from both: on one of each curve's two
    // offsets. Try the four pairs and keep the centre whose tangent points
    // are nearest the clicks.
    let offsets = |p: &Prim| -> Vec<Prim> {
        [r, -r]
            .iter()
            .filter_map(|d| match *p {
                Prim::Line { a, b } => {
                    let n = (b - a).normalize_or_zero().perp();
                    let grow = (b - a) * 1e6;
                    Some(Prim::Line {
                        a: a + n * *d - grow,
                        b: b + n * *d + grow,
                    })
                }
                Prim::Arc { c, r: rr, .. } => {
                    let r2 = rr + d;
                    (r2 > POINT_EPS).then_some(Prim::Arc {
                        c,
                        r: r2,
                        start: 0.0,
                        sweep: TAU,
                    })
                }
            })
            .collect()
    };
    let foot = |p: &Prim, c: DVec2| -> DVec2 {
        match *p {
            Prim::Line { a, b } => {
                let d = b - a;
                a + d * ((c - a).dot(d) / d.length_squared())
            }
            Prim::Arc { c: k, r: rr, .. } => k + (c - k).normalize_or_zero() * rr,
        }
    };
    let mut best: Option<(f64, DVec2, DVec2, DVec2)> = None;
    for oa in offsets(&pa) {
        for ob in offsets(&pb) {
            for hit in prim_prim(&oa, &ob) {
                let Intersection::Point { p: centre, .. } = hit else {
                    continue;
                };
                let (ta, tb) = (foot(&pa, centre), foot(&pb, centre));
                let score = ta.distance(near_a) + tb.distance(near_b);
                if best.is_none_or(|(s, ..)| score < s) {
                    best = Some((score, centre, ta, tb));
                }
            }
        }
    }
    let (_, centre, qa, qb) = best?;
    // The short way round between the two tangent points.
    let (a0, a1) = ((qa - centre).to_angle(), (qb - centre).to_angle());
    let ccw = (a1 - a0).rem_euclid(TAU);
    let (start, sweep) = if ccw <= std::f64::consts::PI {
        (a0, ccw)
    } else {
        (a1, TAU - ccw)
    };
    Some(Fillet {
        arc: Curve2D::Arc {
            c: centre,
            r,
            start: start.rem_euclid(TAU),
            sweep,
        },
        ta: a.closest(qa).0,
        tb: b.closest(qb).0,
    })
}

impl Spline {
    /// The part between two parameters, by inserting knots until both are
    /// ends: the same curve, exactly, over the shorter span.
    ///
    /// With a knot of multiplicity `p` (the degree) at `t`, the curve
    /// passes through a control point there — the one just before the
    /// knot's first occurrence. So once both ends have it, the part is the
    /// control points between those two, under knots clamped at the ends.
    pub fn sub(&self, t0: f64, t1: f64) -> Spline {
        let (t0, t1) = (t0.min(t1), t0.max(t1));
        let p = self.degree;
        let mut s = self.clone();
        for t in [t0, t1] {
            let have = s.knots.iter().filter(|k| **k == t).count();
            for _ in have..p {
                s = s.insert_knot(t);
            }
        }
        // The control point at t0: just before the last p occurrences of
        // t0 (at the domain's start that is the first control point).
        let last_t0 = s.knots.iter().rposition(|k| *k == t0).unwrap_or(p);
        let a = last_t0 + 1 - p;
        // The control point at t1: just before its first occurrence.
        let b = s
            .knots
            .iter()
            .position(|k| *k == t1)
            .unwrap_or(s.ctrl.len());
        let ctrl = s.ctrl[a - 1..b].to_vec();
        let weights = s.weights.as_ref().map(|w| w[a - 1..b].to_vec());
        let mut knots = vec![t0; p + 1];
        knots.extend(s.knots.iter().copied().filter(|k| *k > t0 && *k < t1));
        knots.extend(std::iter::repeat_n(t1, p + 1));
        Spline {
            degree: p,
            ctrl,
            knots,
            weights,
        }
    }

    /// Boehm's knot insertion: one more knot at `t`, the curve unchanged.
    pub fn insert_knot(&self, t: f64) -> Spline {
        let p = self.degree;
        let k = self.knots.iter().rposition(|x| *x <= t).unwrap_or(p);
        let k = k.min(self.ctrl.len() - 1).max(p);
        // Homogeneous coordinates, so a rational one stays exact.
        let w: Vec<f64> = self
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0; self.ctrl.len()]);
        let hom: Vec<(DVec2, f64)> = self
            .ctrl
            .iter()
            .zip(&w)
            .map(|(c, w)| (*c * *w, *w))
            .collect();
        let mut out: Vec<(DVec2, f64)> = Vec::with_capacity(hom.len() + 1);
        for i in 0..=hom.len() {
            if i + p <= k {
                out.push(hom[i]);
            } else if i > k {
                out.push(hom[i - 1]);
            } else {
                let denom = self.knots[i + p] - self.knots[i];
                let alpha = if denom.abs() < 1e-300 {
                    0.0
                } else {
                    (t - self.knots[i]) / denom
                };
                let (a, wa) = hom[i - 1];
                let (b, wb) = hom[i];
                out.push((
                    a * (1.0 - alpha) + b * alpha,
                    wa * (1.0 - alpha) + wb * alpha,
                ));
            }
        }
        let mut knots = self.knots.clone();
        knots.insert(k + 1, t);
        let rational = self.weights.is_some();
        Spline {
            degree: p,
            ctrl: out.iter().map(|(c, w)| *c / *w).collect(),
            knots,
            weights: rational.then(|| out.iter().map(|(_, w)| *w).collect()),
        }
    }

    /// The same curve run the other way, over the same domain.
    pub fn reversed(&self) -> Spline {
        let (lo, hi) = self.domain();
        let knots: Vec<f64> = self.knots.iter().rev().map(|k| lo + hi - k).collect();
        Spline {
            degree: self.degree,
            ctrl: self.ctrl.iter().rev().copied().collect(),
            knots,
            weights: self
                .weights
                .as_ref()
                .map(|w| w.iter().rev().copied().collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    /// Any curve's part is the same points over the shorter span.
    #[test]
    fn a_part_of_a_curve_is_that_part() {
        let curves = [
            Curve2D::Line {
                a: v(0.0, 0.0),
                b: v(2.0, 1.0),
            },
            Curve2D::Arc {
                c: v(1.0, 1.0),
                r: 0.5,
                start: 0.3,
                sweep: 2.0,
            },
            Curve2D::Polyline {
                verts: vec![
                    (v(0.0, 0.0), 0.4),
                    (v(1.0, 0.0), 0.0),
                    (v(1.0, 1.0), -0.3),
                    (v(0.0, 1.0), 0.0),
                ],
                closed: false,
            },
            Curve2D::Spline(Spline::through(&[
                v(0.0, 0.0),
                v(1.0, 1.0),
                v(2.0, -1.0),
                v(3.0, 0.5),
            ])),
            Curve2D::Ellipse {
                c: v(0.0, 0.0),
                major: v(2.0, 1.0),
                ratio: 0.4,
                start: 0.2,
                sweep: 3.0,
            },
        ];
        for curve in curves {
            let (lo, hi) = curve.domain();
            let (t0, t1) = (lo + (hi - lo) * 0.2, lo + (hi - lo) * 0.7);
            let part = curve.sub(t0, t1);
            assert!(part.start().distance(curve.point(t0)) < 1e-9, "{curve:?}");
            assert!(part.end().distance(curve.point(t1)) < 1e-9, "{curve:?}");
            // Every point of the part is on the curve.
            let (plo, phi) = part.domain();
            for i in 0..=50 {
                let p = part.point(plo + (phi - plo) * i as f64 / 50.0);
                let (_, q) = curve.closest(p);
                assert!(
                    q.distance(p) < 1e-7,
                    "{curve:?}: {p} is {} off",
                    q.distance(p)
                );
            }
        }
    }

    /// Through a circle's seam: the long way round.
    #[test]
    fn a_circle_cut_across_its_seam_keeps_the_long_way() {
        let circle = Curve2D::Circle {
            c: v(0.0, 0.0),
            r: 1.0,
        };
        let part = circle.sub(0.75, 0.25);
        let Curve2D::Arc { sweep, .. } = part else {
            panic!()
        };
        assert!((sweep - PI).abs() < 1e-12);
        assert!(
            part.point(0.5).distance(v(1.0, 0.0)) < 1e-12,
            "through angle zero"
        );
    }

    #[test]
    fn offsets_stay_their_distance_away() {
        let line = Curve2D::Line {
            a: v(0.0, 0.0),
            b: v(1.0, 0.0),
        };
        let up = line.offset(0.1, 1e-6).unwrap();
        assert!(
            up.start().distance(v(0.0, 0.1)) < 1e-12,
            "left of travel is up"
        );
        let arc = Curve2D::Arc {
            c: v(0.0, 0.0),
            r: 1.0,
            start: 0.0,
            sweep: FRAC_PI_2,
        };
        let Curve2D::Arc { r, .. } = arc.offset(0.1, 1e-6).unwrap() else {
            panic!()
        };
        assert!(
            (r - 0.9).abs() < 1e-12,
            "left of an anticlockwise arc is in"
        );
        assert!(Curve2D::Circle {
            c: v(0.0, 0.0),
            r: 0.1
        }
        .offset(0.2, 1e-6)
        .is_none());
        // A spline's offset is within tolerance of its distance away.
        let s = Curve2D::Spline(Spline::through(&[v(0.0, 0.0), v(0.05, 0.03), v(0.1, 0.0)]));
        let o = s.offset(0.005, 1e-5).unwrap();
        let (lo, hi) = o.domain();
        for i in 1..20 {
            let p = o.point(lo + (hi - lo) * i as f64 / 20.0);
            let d = s.closest(p).1.distance(p);
            assert!((d - 0.005).abs() < 5e-5, "{d}");
        }
    }

    /// A square's offset outward is a bigger square, its corners extended
    /// to meet as a drafting program's are.
    #[test]
    fn a_closed_polyline_offsets_with_sharp_corners() {
        // Anticlockwise, so left is in; negative goes out.
        let square = Curve2D::Polyline {
            verts: vec![
                (v(0.0, 0.0), 0.0),
                (v(1.0, 0.0), 0.0),
                (v(1.0, 1.0), 0.0),
                (v(0.0, 1.0), 0.0),
            ],
            closed: true,
        };
        let out = square.offset(-0.1, 1e-6).unwrap();
        let Curve2D::Polyline { verts, closed } = &out else {
            panic!()
        };
        assert!(closed);
        let want = [v(-0.1, -0.1), v(1.1, -0.1), v(1.1, 1.1), v(-0.1, 1.1)];
        for p in want {
            assert!(
                verts.iter().any(|(q, _)| q.distance(p) < 1e-9),
                "{p} in {verts:?}"
            );
        }
        assert!((out.length() - 4.8).abs() < 1e-9);
    }

    /// Two lines at a right angle rounded: the arc is tangent to both, at
    /// r from the corner along each.
    #[test]
    fn a_right_corner_rounds_tangent_to_both() {
        let a = Curve2D::Line {
            a: v(0.0, 0.0),
            b: v(2.0, 0.0),
        };
        let b = Curve2D::Line {
            a: v(2.0, 0.0),
            b: v(2.0, 2.0),
        };
        let f = fillet(&a, v(1.5, 0.0), &b, v(2.0, 0.5), 0.5).unwrap();
        let Curve2D::Arc { c, r, sweep, .. } = f.arc else {
            panic!()
        };
        assert!(c.distance(v(1.5, 0.5)) < 1e-9, "{c}");
        assert!((r - 0.5).abs() < 1e-12);
        assert!((sweep - FRAC_PI_2).abs() < 1e-9);
        assert!((a.point(f.ta).x - 1.5).abs() < 1e-9);
        assert!((b.point(f.tb).y - 0.5).abs() < 1e-9);
        // A line and a circle: the arc touches the circle.
        let circle = Curve2D::Circle {
            c: v(0.0, 2.0),
            r: 1.0,
        };
        let floor = Curve2D::Line {
            a: v(-3.0, 0.0),
            b: v(3.0, 0.0),
        };
        // The gap is 1, so the rounding needs a diameter of more than that.
        assert!(fillet(&floor, v(1.0, 0.0), &circle, v(0.7, 1.3), 0.3).is_none());
        let f = fillet(&floor, v(1.0, 0.0), &circle, v(0.7, 1.3), 0.6).unwrap();
        let Curve2D::Arc { c, .. } = f.arc else {
            panic!()
        };
        assert!((c.y - 0.6).abs() < 1e-9, "tangent to the floor: {c}");
        assert!(
            (c.distance(v(0.0, 2.0)) - 1.6).abs() < 1e-9,
            "outside the circle"
        );
        assert!(c.x > 0.0, "on the side clicked");
    }

    #[test]
    fn a_spline_with_a_knot_inserted_is_the_same_curve() {
        let s = Spline::through(&[
            v(0.0, 0.0),
            v(1.0, 2.0),
            v(2.0, 0.0),
            v(3.0, 1.0),
            v(4.0, 0.0),
        ]);
        let t = s.insert_knot(0.37);
        for i in 0..=40 {
            let u = i as f64 / 40.0;
            assert!(s.point(u).distance(t.point(u)) < 1e-12, "{u}");
        }
        let r = s.reversed();
        let (lo, hi) = s.domain();
        for i in 0..=10 {
            let u = lo + (hi - lo) * i as f64 / 10.0;
            assert!(s.point(u).distance(r.point(lo + hi - u)) < 1e-12);
        }
    }

    /// Every curve reversed passes the same points the other way.
    #[test]
    fn a_reversed_curve_runs_back_over_the_same_points() {
        let curves = [
            Curve2D::Arc {
                c: v(0.0, 0.0),
                r: 1.0,
                start: 0.2,
                sweep: 2.0,
            },
            Curve2D::Circle {
                c: v(1.0, 0.0),
                r: 0.5,
            },
            Curve2D::Ellipse {
                c: v(0.0, 0.0),
                major: v(2.0, 0.5),
                ratio: 0.3,
                start: 0.5,
                sweep: 2.5,
            },
        ];
        for c in curves {
            let r = c.reversed();
            let (lo, hi) = c.domain();
            for i in 0..=10 {
                let t = lo + (hi - lo) * i as f64 / 10.0;
                let back = r.point(lo + hi - t);
                assert!(c.point(t).distance(back) < 1e-12, "{c:?} at {t}");
            }
            assert!((c.length() - r.length()).abs() < 1e-9);
        }
    }

    #[test]
    fn a_polyline_reversed_is_the_same_path() {
        let p = Curve2D::Polyline {
            verts: vec![(v(0.0, 0.0), 0.5), (v(1.0, 0.0), 0.0), (v(1.0, 1.0), 0.0)],
            closed: false,
        };
        let r = p.reversed();
        assert!(r.start().distance(p.end()) < 1e-12);
        assert!(r.end().distance(p.start()) < 1e-12);
        assert!((r.length() - p.length()).abs() < 1e-12);
        assert!(
            r.point(1.5).distance(p.point(0.5)) < 1e-12,
            "the bulge is kept"
        );
    }
}
