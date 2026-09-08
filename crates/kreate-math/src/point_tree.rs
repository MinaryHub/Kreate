//! A k-d tree over points, for "what is the nearest one to here".
//!
//! # Why this exists rather than a grid
//!
//! A uniform grid answers "is there a point within `r`" well and "which point
//! is nearest" badly, and the difference is not academic. Proportional editing
//! asks the second question for every vertex of the mesh against every selected
//! one, and a modeller's selection is a *patch*: a couple of thousand vertices
//! inside a square a fortieth of the surface across. Every one of them lands in
//! the same handful of grid cells, so a cell lookup narrows two thousand
//! candidates down to two thousand candidates — **43 ms a frame, measured, at
//! two hundred thousand vertices** — while the same query against a tree visits
//! a few dozen nodes.
//!
//! The grid is not wrong, it is answering a different question. Welding by
//! distance really does want "everything within `r`", and
//! `kreate_mesh::edit::weld_by_distance` uses a grid for it correctly.
//!
//! # Shape
//!
//! An implicit balanced tree over a permuted array, in the manner of a heap: the
//! node for the range `[lo, hi)` is its median element, at `(lo + hi) / 2`, and
//! its children are the two halves. That costs no child pointers at all, and it
//! makes the build a sequence of `select_nth_unstable` calls — O(n log n) with
//! no allocation per node.
//!
//! Splitting on the widest axis of the range rather than cycling x, y, z. A
//! mesh is routinely flat in one axis — a sheet, a floor plan, a lofted
//! surface — and cycling would spend every third level splitting a dimension
//! with nothing in it.
//!
//! **No test covers that rule, and a sweep that replaced it with cycling axes
//! found nothing that failed** — not the correctness tests, which cannot see
//! it, and not the drag budget, which stayed inside its threshold. The bounds
//! below do the pruning either way, and the choice of split axis only decides
//! how quickly they tighten. It is kept because it is the right rule and now
//! costs nothing (the bounds are computed anyway, and the widest axis falls out
//! of them), not because anything demonstrates it is needed.
//!
//! # Every node carries its bounds, and the textbook version that does not is
//! # the reason
//!
//! A plain k-d tree prunes on the splitting plane: skip the far half when the
//! plane is further away than the best found. **That prunes nothing for a thin
//! set**, and a modeller’s selection is routinely thin — an edge loop, a strip
//! of faces, a row of a grid. The widest axis of a strip is its length, so the
//! root splits along it, and a query point sitting just off to the side has a
//! plane distance of nearly zero at every level near the root. The search
//! descends almost the whole tree and rejects at the leaves.
//!
//! That is not a hypothesis. The version without bounds measured **40 ms** on
//! the case a grid did in 3.5, which is what sent this back to be rewritten.
//! With a box per node the same query rejects the entire tree at the root,
//! because the box is far in the axis the planes never split.
//!
//! Twenty-four bytes a point, and worth every one.

use glam::Vec3;

use crate::Aabb;

/// Below this, a range is scanned rather than split: the branch and the
/// bounds test cost more than measuring a handful of points.
const LEAF: usize = 8;

/// A static k-d tree over a set of points.
///
/// Static on purpose. Every caller so far builds one, asks it a great many
/// questions and throws it away, and insertion would cost a rebalancing story
/// for nobody.
#[derive(Clone, Debug, Default)]
pub struct PointTree {
    /// The points, permuted into tree order.
    points: Vec<Vec3>,
    /// Each point's index in the slice the tree was built from, so a caller can
    /// find out *which* point was nearest and not merely where it is.
    origin: Vec<u32>,
    /// The split axis of the node at each position. Only meaningful at a
    /// position that is some range's median, which is exactly where it is read.
    axes: Vec<u8>,
    /// The bounds of the *whole range* whose median sits at each position — not
    /// of the point there. Read at the same positions as `axes`, and the reason
    /// a query over a thin set is not a linear scan.
    bounds: Vec<Aabb>,
}

/// What a query found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nearest {
    /// The point's index in the slice the tree was built from.
    pub index: u32,
    /// Distance from the query point.
    pub distance: f32,
}

impl PointTree {
    /// Builds a tree over `points`.
    ///
    /// The indices a query reports are positions in this slice, and they stay
    /// valid however the tree permutes its own copy.
    pub fn build(points: &[Vec3]) -> Self {
        let mut ordered: Vec<(Vec3, u32)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (*p, i as u32))
            .collect();
        let n = ordered.len();
        let mut axes = vec![0u8; n];
        let mut bounds = vec![Aabb::new(Vec3::ZERO, Vec3::ZERO); n];
        split(&mut ordered, &mut axes, &mut bounds, 0, n);

        let mut tree = Self {
            points: Vec::with_capacity(n),
            origin: Vec::with_capacity(n),
            axes,
            bounds,
        };
        for (p, i) in ordered {
            tree.points.push(p);
            tree.origin.push(i);
        }
        tree
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// The nearest point to `p`, or `None` if there is none within `limit`.
    ///
    /// The limit is not a convenience: it is what lets a query over a large
    /// mesh reject most of the tree at the root. A falloff radius is exactly
    /// such a limit, and passing `f32::INFINITY` to mean "no limit" works but
    /// gives up that pruning.
    pub fn nearest(&self, p: Vec3, limit: f32) -> Option<Nearest> {
        // A NaN limit named explicitly rather than caught by a negated
        // comparison: `!(limit >= 0.0)` is true for NaN and reads like it is
        // testing for a negative, which is how a reader comes to believe a NaN
        // reaches the search.
        if self.points.is_empty() || limit.is_nan() || limit < 0.0 {
            return None;
        }
        // Squared throughout, so the search never takes a square root it does
        // not report.
        //
        // The limit is kept apart from the running best rather than seeded into
        // it, so that it can be *inclusive* while an improvement stays strict.
        // Folding the two together makes the choice: seeding the best with the
        // limit rejects a point at exactly that distance — which for a falloff
        // means a vertex sitting on the selection is not part of it — while
        // accepting ties lets a later equidistant point displace an earlier one,
        // and which point that is depends on how the tree happened to permute
        // them. Neither is acceptable, so they are separate numbers.
        let limit2 = limit * limit;
        let mut best = (f32::INFINITY, u32::MAX);
        self.search(0, self.points.len(), p, limit2, &mut best);
        (best.1 != u32::MAX).then(|| Nearest {
            index: best.1,
            distance: best.0.sqrt(),
        })
    }

    fn search(&self, lo: usize, hi: usize, p: Vec3, limit2: f32, best: &mut (f32, u32)) {
        if hi <= lo {
            return;
        }
        // The whole range, rejected in one test. This is the pruning that
        // matters: the splitting planes below can only separate along one axis
        // at a time, and for a thin set that is never the axis the query is far
        // in.
        let mid = (lo + hi) / 2;
        if distance_squared_to(&self.bounds[mid], p) > best.0.min(limit2) {
            return;
        }
        if hi - lo <= LEAF {
            for i in lo..hi {
                let d = self.points[i].distance_squared(p);
                if d < best.0 && d <= limit2 {
                    *best = (d, self.origin[i]);
                }
            }
            return;
        }

        let d = self.points[mid].distance_squared(p);
        if d < best.0 && d <= limit2 {
            *best = (d, self.origin[mid]);
        }

        let axis = self.axes[mid] as usize;
        let offset = p[axis] - self.points[mid][axis];
        // The near side first: finding a close point there is what makes the
        // far side's bounds test succeed, and a search that took them in a
        // fixed order would visit both halves nearly every time.
        let (near, far) = if offset < 0.0 {
            ((lo, mid), (mid + 1, hi))
        } else {
            ((mid + 1, hi), (lo, mid))
        };
        self.search(near.0, near.1, p, limit2, best);
        // The far side can only hold something worth having if the splitting
        // plane is closer than both what we have and what we would accept. Kept
        // alongside the box test rather than replaced by it: this one costs a
        // subtract and a multiply and rejects before the recursive call.
        if offset * offset <= best.0.min(limit2) {
            self.search(far.0, far.1, p, limit2, best);
        }
    }
}

fn split(ordered: &mut [(Vec3, u32)], axes: &mut [u8], bounds: &mut [Aabb], lo: usize, hi: usize) {
    if hi <= lo {
        return;
    }
    // Recorded for every range the recursion produces, leaves included, because
    // the search tests it before it looks at the size.
    let mid = (lo + hi) / 2;
    let box_ = Aabb::from_points(ordered[lo..hi].iter().map(|(p, _)| *p));
    bounds[mid] = box_;
    if hi - lo <= LEAF {
        return;
    }
    // Out of the bounds that were needed anyway, so the rule is free.
    let axis = widest(box_);
    // `total_cmp` rather than `partial_cmp`, because a mesh under an operator
    // legitimately contains a NaN for as long as it takes something to report
    // it, and an inconsistent comparator turns that into a panic in a place
    // that has nothing to do with the cause.
    ordered[lo..hi].select_nth_unstable_by(mid - lo, |a, b| a.0[axis].total_cmp(&b.0[axis]));
    axes[mid] = axis as u8;
    split(ordered, axes, bounds, lo, mid);
    split(ordered, axes, bounds, mid + 1, hi);
}

/// How far `p` is from the nearest point of `box`, squared. Zero inside.
fn distance_squared_to(bounds: &Aabb, p: Vec3) -> f32 {
    let outside = (bounds.min - p).max(p - bounds.max).max(Vec3::ZERO);
    outside.length_squared()
}

/// The axis over which a box is longest.
fn widest(bounds: Aabb) -> usize {
    let size = bounds.max - bounds.min;
    if size.x >= size.y && size.x >= size.z {
        0
    } else if size.y >= size.z {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic spread of points, and deliberately not a symmetric one:
    /// a lattice or a sphere makes every candidate equidistant, and a tree that
    /// returned any of them would pass.
    fn scattered(n: u32) -> Vec<Vec3> {
        let mut state = 0x2545_F491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..n)
            .map(|_| Vec3::new(next() * 10.0, next() * 3.0, next() * 0.4))
            .collect()
    }

    /// Query points, deliberately *not* drawn from the set being searched.
    /// The first version reused `scattered`, whose sequence is deterministic,
    /// so every query sat exactly on a point: the nearest distance was zero and
    /// no test of the limit could say anything.
    fn queries(n: u32) -> Vec<Vec3> {
        scattered(n * 2)
            .into_iter()
            .map(|p| p + Vec3::new(0.031, -0.017, 0.009))
            .take(n as usize)
            .collect()
    }

    fn brute(points: &[Vec3], p: Vec3, limit: f32) -> Option<Nearest> {
        points
            .iter()
            .enumerate()
            .map(|(i, q)| (i as u32, q.distance(p)))
            .filter(|(_, d)| *d <= limit)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, distance)| Nearest { index, distance })
    }

    #[test]
    fn the_tree_agrees_with_measuring_every_point() {
        // The only claim that matters, checked against the definition. Sizes
        // either side of the leaf threshold, so both the scan and the descent
        // are exercised.
        for n in [1u32, 7, 8, 9, 64, 1000] {
            let points = scattered(n);
            let tree = PointTree::build(&points);
            assert_eq!(tree.len(), n as usize);
            for q in queries(50) {
                let want = brute(&points, q, f32::INFINITY);
                let got = tree.nearest(q, f32::INFINITY);
                match (want, got) {
                    (Some(w), Some(g)) => assert!(
                        (w.distance - g.distance).abs() < 1e-5,
                        "n={n} q={q:?}: tree found {g:?}, nearest is {w:?}"
                    ),
                    (None, None) => {}
                    _ => panic!("n={n} q={q:?}: {want:?} vs {got:?}"),
                }
            }
        }
    }

    #[test]
    fn the_limit_is_honoured_in_both_directions() {
        // Both directions, because a limit that rejected everything would pass
        // half of this test and a limit that rejected nothing would pass the
        // other half.
        let points = scattered(500);
        let tree = PointTree::build(&points);
        let mut inside = 0;
        let mut outside = 0;
        for q in queries(80) {
            let exact = brute(&points, q, f32::INFINITY).expect("500 points");
            let generous = tree.nearest(q, exact.distance * 1.01);
            assert!(generous.is_some(), "a limit above the answer refused it");
            inside += 1;

            let mean = tree.nearest(q, exact.distance * 0.99);
            assert!(mean.is_none(), "a limit below the answer returned {mean:?}");
            outside += 1;
        }
        assert!(inside > 0 && outside > 0);
    }

    #[test]
    fn a_point_at_exactly_the_limit_is_found() {
        // The boundary, which the first version got wrong by seeding the
        // running best with the limit. It matters: the falloff hands the limit
        // and the radius the same number, so a vertex exactly a radius away
        // was silently outside the neighbourhood.
        let points = vec![Vec3::new(3.0, 0.0, 0.0)];
        let tree = PointTree::build(&points);
        let q = Vec3::ZERO;
        assert_eq!(
            tree.nearest(q, 3.0),
            Some(Nearest {
                index: 0,
                distance: 3.0
            }),
            "a point at exactly the limit was refused"
        );
        assert_eq!(tree.nearest(q, 2.999), None);
    }

    #[test]
    fn an_empty_tree_finds_nothing_rather_than_panicking() {
        let tree = PointTree::build(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.nearest(Vec3::ZERO, 1.0), None);
    }

    #[test]
    fn a_flat_sheet_is_searched_correctly() {
        // A surface with nothing in z, which is the shape a modeller’s selection
        // usually has. This checks the *answers*, which is all it can: the
        // split-axis rule shows up as work done rather than as a wrong result,
        // and a falsification sweep confirmed that replacing the rule with
        // cycling axes fails nothing here or in the drag budget. See the module
        // header — the rule is kept on its merits, not on this test.
        let points: Vec<Vec3> = (0..4096)
            .map(|i| Vec3::new((i % 64) as f32, (i / 64) as f32, 0.0))
            .collect();
        let tree = PointTree::build(&points);
        // Every point is its own nearest, found exactly.
        for (i, p) in points.iter().enumerate().step_by(97) {
            let got = tree.nearest(*p, 0.5).expect("a point finds itself");
            assert_eq!(got.index, i as u32);
            assert_eq!(got.distance, 0.0);
        }
    }

    #[test]
    fn coincident_points_do_not_confuse_the_split() {
        // A degenerate range: every point identical, so no axis has any extent
        // and the median split cannot separate anything. It must still build
        // and still answer.
        let points = vec![Vec3::new(2.0, 2.0, 2.0); 100];
        let tree = PointTree::build(&points);
        let got = tree.nearest(Vec3::new(2.0, 2.0, 2.5), 1.0).expect("found");
        assert!((got.distance - 0.5).abs() < 1e-6, "{got:?}");
    }

    #[test]
    fn a_nan_does_not_panic_the_build() {
        // A mesh mid-operator legitimately holds a NaN until something reports
        // it, and a spatial index that panicked would report it from the wrong
        // place entirely.
        let mut points = scattered(200);
        points[73] = Vec3::new(f32::NAN, 1.0, 2.0);
        let tree = PointTree::build(&points);
        assert_eq!(tree.len(), 200);
        // The answer for a well-behaved query is still a real point.
        let got = tree.nearest(Vec3::new(0.0, 0.0, 0.0), f32::INFINITY);
        assert!(got.is_some_and(|n| n.distance.is_finite()), "{got:?}");
    }
}
