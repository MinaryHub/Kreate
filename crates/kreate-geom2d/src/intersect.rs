//! Where two curves meet.
//!
//! Exact between the prims — a line and a line, a line and an arc, two
//! arcs — and for a curve that has none (an ellipse, a spline) found on a
//! fine drawing of it and then settled onto both true curves by Newton's
//! method, so the point is on the curves and not on their chords.

use crate::curve::Curve2D;
use crate::prim::Prim;
use crate::{DVec2, POINT_EPS};

/// One place two curves meet.
#[derive(Clone, Debug, PartialEq)]
pub enum Intersection {
    /// They cross or touch at `p`, at parameter `ta` on the first and `tb`
    /// on the second.
    Point { p: DVec2, ta: f64, tb: f64 },
    /// They run together — two lines on one line, two arcs on one circle —
    /// over these spans of each one's parameter.
    Overlap { a: (f64, f64), b: (f64, f64) },
}

/// Every place `a` and `b` meet, in order along `a`.
pub fn intersect(a: &Curve2D, b: &Curve2D) -> Vec<Intersection> {
    let (pa, pb) = (a.prims(), b.prims());
    let mut out = if !pa.is_empty() && !pb.is_empty() {
        let mut out = Vec::new();
        for (i, x) in pa.iter().enumerate() {
            for (j, y) in pb.iter().enumerate() {
                for hit in prim_prim(x, y) {
                    out.push(match hit {
                        Intersection::Point { p, ta, tb } => Intersection::Point {
                            p,
                            ta: lift(a, i, ta),
                            tb: lift(b, j, tb),
                        },
                        Intersection::Overlap { a: s, b: t } => Intersection::Overlap {
                            a: (lift(a, i, s.0), lift(a, i, s.1)),
                            b: (lift(b, j, t.0), lift(b, j, t.1)),
                        },
                    });
                }
            }
        }
        out
    } else {
        approximate(a, b)
    };
    // A polyline's vertex is the end of one segment and the start of the
    // next: the same crossing found twice.
    dedup(&mut out);
    out.sort_by(|x, y| first_ta(x).total_cmp(&first_ta(y)));
    out
}

fn first_ta(i: &Intersection) -> f64 {
    match i {
        Intersection::Point { ta, .. } => *ta,
        Intersection::Overlap { a, .. } => a.0.min(a.1),
    }
}

/// A prim's own parameter as the curve's: segment `i` of a polyline is
/// `[i, i + 1]`.
fn lift(curve: &Curve2D, i: usize, t: f64) -> f64 {
    match curve {
        Curve2D::Polyline { .. } => i as f64 + t,
        _ => t,
    }
}

fn dedup(hits: &mut Vec<Intersection>) {
    let mut kept: Vec<Intersection> = Vec::new();
    for h in hits.drain(..) {
        let duplicate = match &h {
            Intersection::Point { p, .. } => kept.iter().any(|k| match k {
                Intersection::Point { p: q, .. } => q.distance(*p) < POINT_EPS * 10.0,
                _ => false,
            }),
            _ => false,
        };
        if !duplicate {
            kept.push(h);
        }
    }
    *hits = kept;
}

/// The exact meeting of two prims.
pub fn prim_prim(a: &Prim, b: &Prim) -> Vec<Intersection> {
    match (*a, *b) {
        (Prim::Line { a: p, b: q }, Prim::Line { a: r, b: s }) => line_line(p, q, r, s),
        (Prim::Line { .. }, Prim::Arc { .. }) => line_arc(a, b, false),
        (Prim::Arc { .. }, Prim::Line { .. }) => line_arc(b, a, true),
        (Prim::Arc { .. }, Prim::Arc { .. }) => arc_arc(a, b),
    }
}

fn line_line(p: DVec2, q: DVec2, r: DVec2, s: DVec2) -> Vec<Intersection> {
    let (d, e) = (q - p, s - r);
    let denom = d.perp_dot(e);
    let scale = d.length() * e.length();
    if scale < POINT_EPS * POINT_EPS {
        return Vec::new();
    }
    if denom.abs() <= 1e-12 * scale {
        // Parallel. On one line only if r is on p→q's line.
        let off = (r - p).perp_dot(d).abs() / d.length();
        if off > POINT_EPS {
            return Vec::new();
        }
        let len2 = d.length_squared();
        let (t0, t1) = ((r - p).dot(d) / len2, (s - p).dot(d) / len2);
        let (lo, hi) = (t0.min(t1).max(0.0), t0.max(t1).min(1.0));
        let eps = POINT_EPS / d.length();
        if hi < lo - eps {
            return Vec::new();
        }
        let back = |t: f64| (p + d * t - r).dot(e) / e.length_squared();
        if hi - lo <= eps {
            // End to end: a single point.
            let t = (lo + hi) / 2.0;
            return vec![Intersection::Point {
                p: p + d * t,
                ta: t,
                tb: back(t).clamp(0.0, 1.0),
            }];
        }
        return vec![Intersection::Overlap {
            a: (lo, hi),
            b: (back(lo), back(hi)),
        }];
    }
    let ta = (r - p).perp_dot(e) / denom;
    let tb = (r - p).perp_dot(d) / denom;
    let (ea, eb) = (POINT_EPS / d.length(), POINT_EPS / e.length());
    if ta < -ea || ta > 1.0 + ea || tb < -eb || tb > 1.0 + eb {
        return Vec::new();
    }
    let (ta, tb) = (ta.clamp(0.0, 1.0), tb.clamp(0.0, 1.0));
    vec![Intersection::Point {
        p: p + d * ta,
        ta,
        tb,
    }]
}

/// A line against an arc; `swap` returns the parameters arc-first.
fn line_arc(line: &Prim, arc: &Prim, swap: bool) -> Vec<Intersection> {
    let (Prim::Line { a: p, b: q }, Prim::Arc { c, r, .. }) = (*line, *arc) else {
        return Vec::new();
    };
    let d = q - p;
    let len = d.length();
    if len < POINT_EPS {
        return Vec::new();
    }
    // The foot of the perpendicular from the centre, and how far either
    // side of it the circle crosses the line.
    let u = d / len;
    let foot_t = (c - p).dot(u);
    let foot = p + u * foot_t;
    let h = foot.distance(c);
    if h > r + POINT_EPS {
        return Vec::new();
    }
    let half = if (h - r).abs() <= POINT_EPS {
        0.0
    } else {
        (r * r - h * h).max(0.0).sqrt()
    };
    let along: &[f64] = if half == 0.0 { &[0.0] } else { &[-half, half] };
    let mut out = Vec::new();
    for s in along {
        let t = (foot_t + s) / len;
        let eps = POINT_EPS / len;
        if t < -eps || t > 1.0 + eps {
            continue;
        }
        let t = t.clamp(0.0, 1.0);
        let at = p + d * t;
        let Some(ta_arc) = arc.param_of_angle((at - c).to_angle()) else {
            continue;
        };
        out.push(if swap {
            Intersection::Point {
                p: at,
                ta: ta_arc,
                tb: t,
            }
        } else {
            Intersection::Point {
                p: at,
                ta: t,
                tb: ta_arc,
            }
        });
    }
    out
}

fn arc_arc(a: &Prim, b: &Prim) -> Vec<Intersection> {
    let (Prim::Arc { c: c1, r: r1, .. }, Prim::Arc { c: c2, r: r2, .. }) = (*a, *b) else {
        return Vec::new();
    };
    let d = c1.distance(c2);
    if d < POINT_EPS && (r1 - r2).abs() < POINT_EPS {
        return arc_overlap(a, b, c1);
    }
    if d < POINT_EPS || d > r1 + r2 + POINT_EPS || d < (r1 - r2).abs() - POINT_EPS {
        return Vec::new();
    }
    // Along the line of centres to the chord through both crossings, then
    // either way along the chord.
    let x = (d * d + r1 * r1 - r2 * r2) / (2.0 * d);
    let y2 = r1 * r1 - x * x;
    let u = (c2 - c1) / d;
    let base = c1 + u * x;
    let points: Vec<DVec2> = if y2 <= POINT_EPS * r1.max(1.0) {
        vec![base]
    } else {
        let y = y2.sqrt();
        vec![base + u.perp() * y, base - u.perp() * y]
    };
    points
        .into_iter()
        .filter_map(|p| {
            let ta = a.param_of_angle((p - c1).to_angle())?;
            let tb = b.param_of_angle((p - c2).to_angle())?;
            Some(Intersection::Point { p, ta, tb })
        })
        .collect()
}

/// Two arcs on one circle: where their spans share angles.
fn arc_overlap(a: &Prim, b: &Prim, c: DVec2) -> Vec<Intersection> {
    // Walk b's ends and a's ends: the shared stretch is bounded by
    // whichever ends lie on the other arc. Sampling b finely is enough to
    // find the runs; their ends are then exact (they are ends).
    let n = 720;
    let mut runs: Vec<(f64, f64)> = Vec::new();
    let mut open: Option<f64> = None;
    for i in 0..=n {
        let tb = i as f64 / n as f64;
        let on = a.param_of_angle((b.point(tb) - c).to_angle()).is_some();
        match (on, open) {
            (true, None) => open = Some(tb),
            (false, Some(s)) => {
                runs.push((s, (i - 1) as f64 / n as f64));
                open = None;
            }
            _ => {}
        }
    }
    if let Some(s) = open {
        runs.push((s, 1.0));
    }
    runs.into_iter()
        .map(|(b0, b1)| {
            let ta = |tb: f64| {
                a.param_of_angle((b.point(tb) - c).to_angle())
                    .unwrap_or(0.0)
            };
            if (b1 - b0).abs() < 1.0 / n as f64 {
                Intersection::Point {
                    p: b.point(b0),
                    ta: ta(b0),
                    tb: b0,
                }
            } else {
                Intersection::Overlap {
                    a: (ta(b0), ta(b1)),
                    b: (b0, b1),
                }
            }
        })
        .collect()
}

/// For curves with no prims: cross their fine drawings, then settle each
/// crossing onto both true curves.
fn approximate(a: &Curve2D, b: &Curve2D) -> Vec<Intersection> {
    let size = {
        let (lo, hi) = a.bounds();
        let (lo2, hi2) = b.bounds();
        (hi.max(hi2) - lo.min(lo2)).length().max(POINT_EPS)
    };
    let tol = size * 1e-4;
    let sample = |c: &Curve2D| -> Vec<(f64, DVec2)> {
        // Parameters as well as points: a fine uniform sampling, since the
        // tessellation does not report its parameters.
        let (lo, hi) = c.domain();
        let n = ((c.length() / tol).ceil() as usize).clamp(64, 20_000);
        (0..=n)
            .map(|i| {
                let t = lo + (hi - lo) * i as f64 / n as f64;
                (t, c.point(t))
            })
            .collect()
    };
    let (sa, sb) = (sample(a), sample(b));
    let mut out = Vec::new();
    for wa in sa.windows(2) {
        let (alo, ahi) = (wa[0].1.min(wa[1].1), wa[0].1.max(wa[1].1));
        for wb in sb.windows(2) {
            let (blo, bhi) = (wb[0].1.min(wb[1].1), wb[0].1.max(wb[1].1));
            if ahi.x < blo.x || bhi.x < alo.x || ahi.y < blo.y || bhi.y < alo.y {
                continue;
            }
            for hit in line_line(wa[0].1, wa[1].1, wb[0].1, wb[1].1) {
                let Intersection::Point { ta, tb, .. } = hit else {
                    continue;
                };
                let ta = wa[0].0 + (wa[1].0 - wa[0].0) * ta;
                let tb = wb[0].0 + (wb[1].0 - wb[0].0) * tb;
                if let Some(found) = settle(a, b, ta, tb) {
                    out.push(found);
                }
            }
        }
    }
    out
}

/// Newton's method on `a(s) - b(t) = 0`.
fn settle(a: &Curve2D, b: &Curve2D, mut s: f64, mut t: f64) -> Option<Intersection> {
    let (alo, ahi) = a.domain();
    let (blo, bhi) = b.domain();
    for _ in 0..30 {
        let f = a.point(s) - b.point(t);
        if f.length() < POINT_EPS * 1e-3 {
            break;
        }
        let (da, db) = (a.derivative(s), b.derivative(t));
        // [da, -db] [ds, dt]ᵀ = -f
        let det = da.x * -db.y - (-db.x) * da.y;
        if det.abs() < 1e-300 {
            break;
        }
        let ds = (-f.x * -db.y - (-db.x) * -f.y) / det;
        let dt = (da.x * -f.y - (-f.x) * da.y) / det;
        s = (s + ds).clamp(alo, ahi);
        t = (t + dt).clamp(blo, bhi);
    }
    let p = a.point(s);
    (p.distance(b.point(t)) < POINT_EPS * 10.0).then_some(Intersection::Point { p, ta: s, tb: t })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Spline, TAU};
    use std::f64::consts::PI;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    fn points(hits: &[Intersection]) -> Vec<DVec2> {
        hits.iter()
            .filter_map(|h| match h {
                Intersection::Point { p, .. } => Some(*p),
                _ => None,
            })
            .collect()
    }

    fn line(a: DVec2, b: DVec2) -> Curve2D {
        Curve2D::Line { a, b }
    }

    #[test]
    fn two_lines_cross_where_they_cross_and_not_past_their_ends() {
        let hits = intersect(
            &line(v(0.0, 0.0), v(2.0, 2.0)),
            &line(v(0.0, 2.0), v(2.0, 0.0)),
        );
        assert_eq!(points(&hits), vec![v(1.0, 1.0)]);
        let Intersection::Point { ta, tb, .. } = hits[0] else {
            panic!()
        };
        assert!((ta - 0.5).abs() < 1e-12 && (tb - 0.5).abs() < 1e-12);
        // Short of each other: nothing.
        assert!(intersect(
            &line(v(0.0, 0.0), v(0.9, 0.9)),
            &line(v(0.0, 2.0), v(2.0, 0.0))
        )
        .is_empty());
        // End on end counts.
        assert_eq!(
            points(&intersect(
                &line(v(0.0, 0.0), v(1.0, 0.0)),
                &line(v(1.0, 0.0), v(1.0, 1.0))
            )),
            vec![v(1.0, 0.0)]
        );
    }

    #[test]
    fn lines_on_one_line_overlap() {
        let hits = intersect(
            &line(v(0.0, 0.0), v(4.0, 0.0)),
            &line(v(3.0, 0.0), v(1.0, 0.0)),
        );
        assert_eq!(hits.len(), 1);
        let Intersection::Overlap { a, b } = hits[0] else {
            panic!("{hits:?}")
        };
        assert!((a.0 - 0.25).abs() < 1e-12 && (a.1 - 0.75).abs() < 1e-12);
        assert!((b.0 - 1.0).abs() < 1e-12 && b.1.abs() < 1e-12);
        // Parallel and apart: nothing.
        assert!(intersect(
            &line(v(0.0, 0.0), v(4.0, 0.0)),
            &line(v(0.0, 1.0), v(4.0, 1.0))
        )
        .is_empty());
    }

    #[test]
    fn a_line_crosses_a_circle_twice_and_touches_it_once() {
        let circle = Curve2D::Circle {
            c: v(0.0, 0.0),
            r: 1.0,
        };
        let hits = intersect(&line(v(-2.0, 0.0), v(2.0, 0.0)), &circle);
        assert_eq!(points(&hits).len(), 2);
        for p in points(&hits) {
            assert!((p.length() - 1.0).abs() < 1e-12);
        }
        // Tangent: one point.
        let touch = intersect(&line(v(-2.0, 1.0), v(2.0, 1.0)), &circle);
        assert_eq!(points(&touch), vec![v(0.0, 1.0)]);
        // An arc only where it is.
        let upper = Curve2D::Arc {
            c: v(0.0, 0.0),
            r: 1.0,
            start: 0.0,
            sweep: PI,
        };
        let hits = intersect(&line(v(0.0, -2.0), v(0.0, 2.0)), &upper);
        assert_eq!(points(&hits), vec![v(0.0, 1.0)]);
        let Intersection::Point { tb, .. } = hits[0] else {
            panic!()
        };
        assert!((tb - 0.5).abs() < 1e-12, "half way round the arc: {tb}");
    }

    #[test]
    fn two_circles_cross_at_two_points() {
        let a = Curve2D::Circle {
            c: v(0.0, 0.0),
            r: 1.0,
        };
        let b = Curve2D::Circle {
            c: v(1.0, 0.0),
            r: 1.0,
        };
        let hits = points(&intersect(&a, &b));
        assert_eq!(hits.len(), 2);
        for p in hits {
            assert!((p.x - 0.5).abs() < 1e-12);
            assert!((p.y.abs() - 0.75f64.sqrt()).abs() < 1e-12);
        }
        // Touching from outside.
        let c = Curve2D::Circle {
            c: v(2.0, 0.0),
            r: 1.0,
        };
        assert_eq!(points(&intersect(&a, &c)).len(), 1);
        // One inside the other: none.
        let small = Curve2D::Circle {
            c: v(0.1, 0.0),
            r: 0.2,
        };
        assert!(intersect(&a, &small).is_empty());
    }

    #[test]
    fn arcs_on_one_circle_overlap_where_they_share_it() {
        let a = Curve2D::Arc {
            c: v(0.0, 0.0),
            r: 1.0,
            start: 0.0,
            sweep: PI,
        };
        let b = Curve2D::Arc {
            c: v(0.0, 0.0),
            r: 1.0,
            start: PI / 2.0,
            sweep: PI,
        };
        let hits = intersect(&a, &b);
        assert_eq!(hits.len(), 1, "{hits:?}");
        let Intersection::Overlap { a: span, .. } = hits[0] else {
            panic!("{hits:?}")
        };
        assert!(
            (span.0 - 0.5).abs() < 0.01 && (span.1 - 1.0).abs() < 0.01,
            "{span:?}"
        );
    }

    /// A polyline's crossing at a vertex is one crossing, not two.
    #[test]
    fn a_crossing_at_a_polyline_vertex_is_found_once() {
        let zig = Curve2D::Polyline {
            verts: vec![(v(0.0, 0.0), 0.0), (v(1.0, 1.0), 0.0), (v(2.0, 0.0), 0.0)],
            closed: false,
        };
        let hits = intersect(&zig, &line(v(1.0, 0.0), v(1.0, 2.0)));
        assert_eq!(points(&hits), vec![v(1.0, 1.0)]);
        let Intersection::Point { ta, .. } = hits[0] else {
            panic!()
        };
        assert!((ta - 1.0).abs() < 1e-12);
    }

    /// An ellipse and a spline are crossed on their true curves.
    #[test]
    fn an_ellipse_and_a_spline_meet_on_the_true_curves() {
        let e = Curve2D::Ellipse {
            c: v(0.0, 0.0),
            major: v(2.0, 0.0),
            ratio: 0.5,
            start: 0.0,
            sweep: TAU,
        };
        let hits = intersect(&e, &line(v(0.0, -2.0), v(0.0, 2.0)));
        let pts = points(&hits);
        assert_eq!(pts.len(), 2, "{hits:?}");
        for p in pts {
            assert!((p.y.abs() - 1.0).abs() < 1e-8 && p.x.abs() < 1e-8, "{p}");
        }
        let s = Curve2D::Spline(Spline::through(&[
            v(-1.0, -1.0),
            v(0.0, 0.5),
            v(1.0, -1.0),
            v(2.0, 0.5),
        ]));
        let horizon = line(v(-2.0, 0.0), v(3.0, 0.0));
        let hits = intersect(&s, &horizon);
        assert_eq!(points(&hits).len(), 3, "{hits:?}");
        for h in &hits {
            let Intersection::Point { ta, p, .. } = h else {
                panic!()
            };
            assert!(s.point(*ta).distance(*p) < 1e-9);
            assert!(p.y.abs() < 1e-9, "{p}");
        }
    }
}
