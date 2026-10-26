//! The closed regions a set of curves makes: what a profile is for an
//! extrude, and what a hatch fills.
//!
//! The curves are cut wherever they meet, the pieces joined at their ends
//! into a planar graph, and its faces traced. Each region keeps its
//! boundary as the pieces of the curves themselves — an arc stays an arc —
//! so a region can be drawn, filled or swept to whatever error its user
//! wants, and a boundary piece still says which curve it came from.
//!
//! What does not bound anything is left out: a line with a free end, the
//! tail of a line that runs past a corner.

use crate::curve::Curve2D;
use crate::intersect::{intersect, Intersection};
use crate::{DVec2, POINT_EPS};

/// One piece of a region's boundary: part of curve `source`, running the
/// way the boundary goes.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryPiece {
    pub source: usize,
    pub curve: Curve2D,
}

/// A closed loop of pieces, anticlockwise for a region's outside,
/// clockwise for a hole.
pub type Loop = Vec<BoundaryPiece>;

#[derive(Clone, Debug)]
pub struct Region {
    pub outer: Loop,
    pub holes: Vec<Loop>,
    /// The outer loop's area less its holes'.
    pub area: f64,
    /// The outer loop drawn, for containment tests.
    outline: Vec<DVec2>,
    hole_outlines: Vec<Vec<DVec2>>,
}

impl Region {
    /// Whether `p` is inside: in the outer loop and in no hole.
    pub fn contains(&self, p: DVec2) -> bool {
        inside(&self.outline, p) && !self.hole_outlines.iter().any(|h| inside(h, p))
    }

    /// The outer loop as points, within `tol`.
    pub fn outline(&self, tol: f64) -> Vec<DVec2> {
        loop_points(&self.outer, tol)
    }

    /// Each hole as points, within `tol`.
    pub fn hole_outlines(&self, tol: f64) -> Vec<Vec<DVec2>> {
        self.holes.iter().map(|h| loop_points(h, tol)).collect()
    }
}

fn loop_points(l: &Loop, tol: f64) -> Vec<DVec2> {
    let mut out: Vec<DVec2> = Vec::new();
    for piece in l {
        for p in piece.curve.tessellate(tol) {
            if out.last().is_none_or(|q| q.distance(p) > POINT_EPS) {
                out.push(p);
            }
        }
    }
    if out.len() > 1 && out[0].distance(out[out.len() - 1]) <= POINT_EPS {
        out.pop();
    }
    out
}

fn signed_area(pts: &[DVec2]) -> f64 {
    let n = pts.len();
    (0..n)
        .map(|i| pts[i].perp_dot(pts[(i + 1) % n]))
        .sum::<f64>()
        / 2.0
}

fn inside(poly: &[DVec2], p: DVec2) -> bool {
    let mut odd = false;
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a.y > p.y) != (b.y > p.y) {
            let x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if p.x < x {
                odd = !odd;
            }
        }
    }
    odd
}

/// A piece of the graph: part of a source curve between two vertices.
struct Edge {
    source: usize,
    curve: Curve2D,
    from: usize,
    to: usize,
}

/// The regions `curves` close, each with its holes. `tol` is the error the
/// regions' areas and containment are worked to.
pub fn regions(curves: &[Curve2D], tol: f64) -> Vec<Region> {
    // Every curve as the pieces that have no self-crossings: a polyline's
    // segments one by one, everything else whole.
    let mut parts: Vec<(usize, Curve2D)> = Vec::new();
    for (i, c) in curves.iter().enumerate() {
        match c {
            Curve2D::Polyline { .. } => {
                let (_, hi) = c.domain();
                for k in 0..hi as usize {
                    parts.push((i, c.sub(k as f64, k as f64 + 1.0)));
                }
            }
            _ => parts.push((i, c.clone())),
        }
    }
    // Where each is cut: its ends, and every place another meets it.
    let bounds: Vec<(DVec2, DVec2)> = parts.iter().map(|(_, c)| c.bounds()).collect();
    let mut cuts: Vec<Vec<f64>> = parts
        .iter()
        .map(|(_, c)| {
            let (lo, hi) = c.domain();
            vec![lo, hi]
        })
        .collect();
    for i in 0..parts.len() {
        for j in i + 1..parts.len() {
            let (a, b) = (bounds[i], bounds[j]);
            let pad = POINT_EPS * 10.0;
            if a.1.x + pad < b.0.x
                || b.1.x + pad < a.0.x
                || a.1.y + pad < b.0.y
                || b.1.y + pad < a.0.y
            {
                continue;
            }
            for hit in intersect(&parts[i].1, &parts[j].1) {
                match hit {
                    Intersection::Point { ta, tb, .. } => {
                        cuts[i].push(ta);
                        cuts[j].push(tb);
                    }
                    Intersection::Overlap { a, b } => {
                        cuts[i].extend([a.0, a.1]);
                        cuts[j].extend([b.0, b.1]);
                    }
                }
            }
        }
    }
    // Vertices: points within a whisker of each other are one.
    let mut vertices: Vec<DVec2> = Vec::new();
    let mut vertex = |p: DVec2| -> usize {
        if let Some(i) = vertices
            .iter()
            .position(|q| q.distance(p) < POINT_EPS * 100.0)
        {
            return i;
        }
        vertices.push(p);
        vertices.len() - 1
    };
    let mut edges: Vec<Edge> = Vec::new();
    for ((source, curve), mut ts) in parts.iter().zip(cuts) {
        ts.sort_by(f64::total_cmp);
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        // A closed curve's pieces run from its seam round to it again; the
        // seam is a vertex like any other, where two of them meet.
        let pieces: Vec<(f64, f64)> = ts.windows(2).map(|w| (w[0], w[1])).collect();
        for (t0, t1) in pieces {
            let piece = curve.sub(t0, t1);
            if piece.length() < POINT_EPS * 10.0 {
                continue;
            }
            let (from, to) = (vertex(piece.start()), vertex(piece.end()));
            if from == to && !piece.is_closed() {
                continue;
            }
            // The same stretch twice (an overlap) is one edge.
            let mid = piece.point(piece_mid(&piece));
            if edges.iter().any(|e| {
                ((e.from, e.to) == (from, to) || (e.from, e.to) == (to, from))
                    && e.curve.point(piece_mid(&e.curve)).distance(mid) < POINT_EPS * 100.0
            }) {
                continue;
            }
            edges.push(Edge {
                source: *source,
                curve: piece,
                from,
                to,
            });
        }
    }
    // A whole circle, uncut, is a loop on one vertex: a region by itself.
    let mut out: Vec<Region> = Vec::new();
    let mut loops: Vec<(Loop, f64, Vec<DVec2>)> = Vec::new();
    edges.retain(|e| {
        if e.from == e.to {
            let l = vec![BoundaryPiece {
                source: e.source,
                curve: e.curve.clone(),
            }];
            let pts = loop_points(&l, tol);
            let a = signed_area(&pts);
            // Anticlockwise for the region, and its reverse for the hole
            // it leaves in whatever surrounds it.
            let (l, pts, a) = if a < 0.0 {
                let r = vec![BoundaryPiece {
                    source: e.source,
                    curve: e.curve.reversed(),
                }];
                let p = loop_points(&r, tol);
                (r, p, -a)
            } else {
                (l, pts, a)
            };
            let mut hole = l.clone();
            hole[0].curve = hole[0].curve.reversed();
            let hole_pts = loop_points(&hole, tol);
            loops.push((l, a, pts));
            loops.push((hole, -a, hole_pts));
            false
        } else {
            true
        }
    });
    // Free ends bound nothing: take them off until none is left.
    loop {
        let mut degree = vec![0usize; vertices.len()];
        for e in &edges {
            degree[e.from] += 1;
            degree[e.to] += 1;
        }
        let before = edges.len();
        edges.retain(|e| degree[e.from] > 1 && degree[e.to] > 1);
        if edges.len() == before {
            break;
        }
    }
    // Half-edges, and at each vertex the ones leaving it by angle.
    let half: Vec<(usize, bool)> = (0..edges.len())
        .flat_map(|i| [(i, true), (i, false)])
        .collect();
    let leaving = |h: (usize, bool)| -> (usize, f64) {
        let e = &edges[h.0];
        let (v, dir) = if h.1 {
            (e.from, depart(&e.curve, false))
        } else {
            (e.to, depart(&e.curve, true))
        };
        (v, dir.to_angle())
    };
    let mut around: Vec<Vec<usize>> = vec![Vec::new(); vertices.len()];
    for (k, h) in half.iter().enumerate() {
        around[leaving(*h).0].push(k);
    }
    for list in &mut around {
        list.sort_by(|a, b| leaving(half[*a]).1.total_cmp(&leaving(half[*b]).1));
    }
    let head = |h: (usize, bool)| if h.1 { edges[h.0].to } else { edges[h.0].from };
    let twin = |k: usize| k ^ 1;
    // Trace each face, keeping it on the left: from a half-edge's head,
    // leave by the one just clockwise of the way back.
    let mut used = vec![false; half.len()];
    for start in 0..half.len() {
        if used[start] {
            continue;
        }
        let mut l: Loop = Vec::new();
        let mut k = start;
        let mut ok = true;
        loop {
            used[k] = true;
            let (e, forward) = half[k];
            let curve = if forward {
                edges[e].curve.clone()
            } else {
                edges[e].curve.reversed()
            };
            l.push(BoundaryPiece {
                source: edges[e].source,
                curve,
            });
            let v = head(half[k]);
            let list = &around[v];
            let Some(at) = list.iter().position(|x| *x == twin(k)) else {
                ok = false;
                break;
            };
            k = list[(at + list.len() - 1) % list.len()];
            if k == start {
                break;
            }
            if used[k] || l.len() > half.len() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let pts = loop_points(&l, tol);
        let a = signed_area(&pts);
        loops.push((l, a, pts));
    }
    // Anticlockwise loops are regions; clockwise ones are the outsides of
    // connected pieces, and a hole in the smallest region around them.
    let mut faces: Vec<(Loop, f64, Vec<DVec2>)> = Vec::new();
    let mut outsides: Vec<(Loop, f64, Vec<DVec2>)> = Vec::new();
    for (l, a, pts) in loops {
        if a > POINT_EPS * POINT_EPS {
            faces.push((l, a, pts));
        } else if a < -POINT_EPS * POINT_EPS {
            outsides.push((l, a, pts));
        }
    }
    let mut holes: Vec<Vec<usize>> = vec![Vec::new(); faces.len()];
    for (h, (_, ha, hp)) in outsides.iter().enumerate() {
        // A point just inside the outside loop's own region, so it is not
        // on any face's boundary: the loop's first point, nudged along the
        // inward normal.
        let probe = probe_inside(hp);
        let host = faces
            .iter()
            .enumerate()
            .filter(|(_, (_, fa, fp))| *fa > -ha + POINT_EPS && inside(fp, probe))
            .filter(|(_, (_, _, fp))| !same_loop(fp, hp))
            .min_by(|x, y| x.1 .1.total_cmp(&y.1 .1))
            .map(|(i, _)| i);
        if let Some(f) = host {
            holes[f].push(h);
        }
    }
    for (i, (l, a, pts)) in faces.into_iter().enumerate() {
        let hole_loops: Vec<Loop> = holes[i].iter().map(|h| outsides[*h].0.clone()).collect();
        let hole_outlines: Vec<Vec<DVec2>> =
            holes[i].iter().map(|h| outsides[*h].2.clone()).collect();
        let hole_area: f64 = holes[i].iter().map(|h| -outsides[*h].1).sum();
        out.push(Region {
            outer: l,
            holes: hole_loops,
            area: a - hole_area,
            outline: pts,
            hole_outlines,
        });
    }
    out
}

/// The middle of a piece's parameter range.
fn piece_mid(c: &Curve2D) -> f64 {
    let (lo, hi) = c.domain();
    (lo + hi) / 2.0
}

/// The direction a piece leaves its start (or, `at_end`, its end, going
/// backwards): taken a short way along, so two pieces leaving along one
/// tangent are told apart by how they bend.
fn depart(c: &Curve2D, at_end: bool) -> DVec2 {
    let (lo, hi) = c.domain();
    let len = c.length().max(POINT_EPS);
    let step = ((hi - lo) * (1e-4f64).max(POINT_EPS * 1e3 / len)).min((hi - lo) / 2.0);
    if at_end {
        c.point(hi - step) - c.end()
    } else {
        c.point(lo + step) - c.start()
    }
}

/// The same outline, as a hole's reverse is of its face's.
fn same_loop(a: &[DVec2], b: &[DVec2]) -> bool {
    a.len() == b.len()
        && a.iter()
            .all(|p| b.iter().any(|q| q.distance(*p) < POINT_EPS * 100.0))
}

/// A point a little inside the region a clockwise outline bounds from
/// outside — that is, just outside the outline itself, on its left.
fn probe_inside(pts: &[DVec2]) -> DVec2 {
    if pts.len() < 2 {
        return pts.first().copied().unwrap_or(DVec2::ZERO);
    }
    let (a, b) = (pts[0], pts[1]);
    let mid = (a + b) / 2.0;
    let n = (b - a).normalize_or_zero().perp();
    let scale = pts
        .iter()
        .fold(0.0f64, |m, p| m.max((*p - pts[0]).length()))
        .max(POINT_EPS);
    mid + n * (scale * 1e-6).max(POINT_EPS * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Spline;
    use std::f64::consts::PI;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    fn line(a: DVec2, b: DVec2) -> Curve2D {
        Curve2D::Line { a, b }
    }

    fn areas(rs: &[Region]) -> Vec<f64> {
        let mut a: Vec<f64> = rs.iter().map(|r| r.area).collect();
        a.sort_by(f64::total_cmp);
        a
    }

    #[test]
    fn four_lines_close_a_square() {
        let c = [
            line(v(0.0, 0.0), v(1.0, 0.0)),
            line(v(1.0, 0.0), v(1.0, 1.0)),
            line(v(1.0, 1.0), v(0.0, 1.0)),
            line(v(0.0, 1.0), v(0.0, 0.0)),
        ];
        let rs = regions(&c, 1e-6);
        assert_eq!(areas(&rs), vec![1.0]);
        assert!(rs[0].contains(v(0.5, 0.5)));
        assert!(!rs[0].contains(v(1.5, 0.5)));
        assert_eq!(rs[0].outer.len(), 4);
    }

    /// A plate with a hole: one region, the hole's area off it, the hole a
    /// circle still.
    #[test]
    fn a_circle_inside_a_rectangle_is_a_hole_and_a_disc() {
        let c = [
            Curve2D::Polyline {
                verts: vec![
                    (v(0.0, 0.0), 0.0),
                    (v(4.0, 0.0), 0.0),
                    (v(4.0, 2.0), 0.0),
                    (v(0.0, 2.0), 0.0),
                ],
                closed: true,
            },
            Curve2D::Circle {
                c: v(2.0, 1.0),
                r: 0.5,
            },
        ];
        let rs = regions(&c, 1e-7);
        assert_eq!(rs.len(), 2, "the plate and the disc");
        let disc_area = PI * 0.25;
        let a = areas(&rs);
        assert!((a[0] - disc_area).abs() < 1e-6, "{a:?}");
        assert!((a[1] - (8.0 - disc_area)).abs() < 1e-6, "{a:?}");
        let plate = rs.iter().find(|r| r.area > 1.0).unwrap();
        assert_eq!(plate.holes.len(), 1);
        assert!(matches!(
            plate.holes[0][0].curve,
            Curve2D::Circle { .. } | Curve2D::Arc { .. }
        ));
        assert!(plate.contains(v(0.5, 0.5)));
        assert!(!plate.contains(v(2.0, 1.0)), "not in the hole");
    }

    /// Two overlapping circles make three regions: two crescents and the
    /// lens between.
    #[test]
    fn two_overlapping_circles_make_three_regions() {
        let c = [
            Curve2D::Circle {
                c: v(0.0, 0.0),
                r: 1.0,
            },
            Curve2D::Circle {
                c: v(1.0, 0.0),
                r: 1.0,
            },
        ];
        let rs = regions(&c, 1e-7);
        assert_eq!(rs.len(), 3, "{:?}", areas(&rs));
        let lens = 2.0 * PI / 3.0 - 3f64.sqrt() / 2.0;
        let a = areas(&rs);
        assert!((a[0] - lens).abs() < 1e-5, "{a:?}");
        assert!((a[1] - (PI - lens)).abs() < 1e-5);
        assert!((a.iter().sum::<f64>() - (2.0 * PI - lens)).abs() < 1e-5);
    }

    /// A line running past a corner, and one with a free end, bound
    /// nothing more than the square.
    #[test]
    fn tails_and_free_ends_bound_nothing() {
        let c = [
            line(v(-1.0, 0.0), v(2.0, 0.0)),
            line(v(1.0, -1.0), v(1.0, 1.0)),
            line(v(1.0, 1.0), v(0.0, 1.0)),
            line(v(0.0, 1.0), v(0.0, 0.0)),
            line(v(0.5, 0.5), v(3.0, 3.0)),
        ];
        let rs = regions(&c, 1e-6);
        assert_eq!(rs.len(), 1, "{:?}", areas(&rs));
        assert!((rs[0].area - 1.0).abs() < 1e-9);
    }

    /// A slot: two half circles and two lines.
    #[test]
    fn a_slot_closes_with_its_arcs() {
        let c = [
            line(v(0.0, -1.0), v(4.0, -1.0)),
            Curve2D::Arc {
                c: v(4.0, 0.0),
                r: 1.0,
                start: -PI / 2.0,
                sweep: PI,
            },
            line(v(4.0, 1.0), v(0.0, 1.0)),
            Curve2D::Arc {
                c: v(0.0, 0.0),
                r: 1.0,
                start: PI / 2.0,
                sweep: PI,
            },
        ];
        let rs = regions(&c, 1e-7);
        assert_eq!(rs.len(), 1);
        assert!((rs[0].area - (8.0 + PI)).abs() < 1e-5, "{}", rs[0].area);
    }

    /// A closed spline is a region of its own; a line across it cuts it
    /// in two.
    #[test]
    fn a_line_across_a_closed_curve_cuts_it_in_two() {
        let blob = Curve2D::Spline(Spline::through(&[
            v(1.0, 0.0),
            v(0.0, 1.0),
            v(-1.0, 0.0),
            v(0.0, -1.0),
            v(1.0, 0.0),
        ]));
        let whole = regions(std::slice::from_ref(&blob), 1e-6);
        assert_eq!(whole.len(), 1);
        let cut = regions(&[blob, line(v(-2.0, 0.0), v(2.0, 0.0))], 1e-6);
        assert_eq!(cut.len(), 2, "{:?}", areas(&cut));
        let sum: f64 = cut.iter().map(|r| r.area).sum();
        assert!(
            (sum - whole[0].area).abs() < 1e-4,
            "{sum} against {}",
            whole[0].area
        );
    }

    /// Nested: a square in a square in a square — each ring its own
    /// region, each with one hole.
    #[test]
    fn nested_loops_hold_each_other_as_holes() {
        let square = |s: f64| Curve2D::Polyline {
            verts: vec![
                (v(-s, -s), 0.0),
                (v(s, -s), 0.0),
                (v(s, s), 0.0),
                (v(-s, s), 0.0),
            ],
            closed: true,
        };
        let rs = regions(&[square(3.0), square(2.0), square(1.0)], 1e-6);
        assert_eq!(rs.len(), 3);
        let a = areas(&rs);
        assert!(
            (a[0] - 4.0).abs() < 1e-9 && (a[1] - 12.0).abs() < 1e-9 && (a[2] - 20.0).abs() < 1e-9,
            "{a:?}"
        );
        let middle = rs.iter().find(|r| (r.area - 12.0).abs() < 1e-6).unwrap();
        assert_eq!(middle.holes.len(), 1);
        assert!(middle.contains(v(1.5, 0.0)));
        assert!(!middle.contains(v(0.0, 0.0)));
        assert!(!middle.contains(v(2.5, 0.0)));
    }
}
