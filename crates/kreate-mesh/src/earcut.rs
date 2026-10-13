//! Polygons with holes into triangles: Mapbox's earcut, carried over.
//!
//! Ear clipping on its own goes wrong on the corridor a hole's bridge makes
//! and on a run of points along a straight side, and both are everyday in a
//! CAD face — a plate with a row of holes, an edge cut into many points by
//! its neighbour. Earcut is the version of the algorithm that a great many
//! files have been through: holes bridged to the right vertex, three passes
//! that each loosen what counts as an ear, and a split into two polygons as
//! the last resort.
//!
//! One change: earcut drops a point that lies on the straight line between
//! its neighbours, and here every boundary point is a vertex the face shares
//! with the next one, so a dropped point would leave a crack. Each is noted
//! and put back afterwards, on the edge it was dropped from.

use std::collections::HashMap;

use glam::DVec2;

#[derive(Clone, Copy)]
struct Node {
    /// The point's index in the input.
    i: usize,
    p: DVec2,
    prev: usize,
    next: usize,
    steiner: bool,
}

struct Earcut {
    nodes: Vec<Node>,
    out: Vec<[usize; 3]>,
    /// Points dropped from a straight run, as (before, point, after).
    dropped: Vec<(usize, usize, usize)>,
}

/// Twice the signed area of `p q r`, negative when they turn anticlockwise
/// — earcut's own convention.
fn area(p: DVec2, q: DVec2, r: DVec2) -> f64 {
    (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
}

fn point_in_triangle(a: DVec2, b: DVec2, c: DVec2, p: DVec2) -> bool {
    (c.x - p.x) * (a.y - p.y) >= (a.x - p.x) * (c.y - p.y)
        && (a.x - p.x) * (b.y - p.y) >= (b.x - p.x) * (a.y - p.y)
        && (b.x - p.x) * (c.y - p.y) >= (c.x - p.x) * (b.y - p.y)
}

fn sign(v: f64) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

fn on_segment(p: DVec2, q: DVec2, r: DVec2) -> bool {
    q.x <= p.x.max(r.x) && q.x >= p.x.min(r.x) && q.y <= p.y.max(r.y) && q.y >= p.y.min(r.y)
}

fn intersects(p1: DVec2, q1: DVec2, p2: DVec2, q2: DVec2) -> bool {
    let o1 = sign(area(p1, q1, p2));
    let o2 = sign(area(p1, q1, q2));
    let o3 = sign(area(p2, q2, p1));
    let o4 = sign(area(p2, q2, q1));
    (o1 != o2 && o3 != o4)
        || (o1 == 0 && on_segment(p1, p2, q1))
        || (o2 == 0 && on_segment(p1, q2, q1))
        || (o3 == 0 && on_segment(p2, p1, q2))
        || (o4 == 0 && on_segment(p2, q1, q2))
}

impl Earcut {
    fn n(&self, k: usize) -> Node {
        self.nodes[k]
    }
    fn p(&self, k: usize) -> DVec2 {
        self.nodes[k].p
    }
    fn next(&self, k: usize) -> usize {
        self.nodes[k].next
    }
    fn prev(&self, k: usize) -> usize {
        self.nodes[k].prev
    }

    fn insert(&mut self, i: usize, p: DVec2, last: Option<usize>) -> usize {
        let k = self.nodes.len();
        match last {
            None => self.nodes.push(Node {
                i,
                p,
                prev: k,
                next: k,
                steiner: false,
            }),
            Some(l) => {
                let after = self.next(l);
                self.nodes.push(Node {
                    i,
                    p,
                    prev: l,
                    next: after,
                    steiner: false,
                });
                self.nodes[after].prev = k;
                self.nodes[l].next = k;
            }
        }
        k
    }

    fn remove(&mut self, k: usize) {
        let (prev, next) = (self.prev(k), self.next(k));
        self.nodes[next].prev = prev;
        self.nodes[prev].next = next;
    }

    /// A ring as a linked list, anticlockwise for the outside and clockwise
    /// for a hole.
    fn ring(&mut self, pts: &[DVec2], first: usize, outside: bool) -> Option<usize> {
        if pts.is_empty() {
            return None;
        }
        let mut twice = 0.0;
        for k in 0..pts.len() {
            let j = (k + pts.len() - 1) % pts.len();
            twice += (pts[j].x - pts[k].x) * (pts[k].y + pts[j].y);
        }
        let mut last = None;
        if outside == (twice > 0.0) {
            for (k, p) in pts.iter().enumerate() {
                last = Some(self.insert(first + k, *p, last));
            }
        } else {
            for (k, p) in pts.iter().enumerate().rev() {
                last = Some(self.insert(first + k, *p, last));
            }
        }
        let last = last?;
        if self.p(last) == self.p(self.next(last)) {
            let next = self.next(last);
            self.remove(last);
            return Some(next);
        }
        Some(last)
    }

    /// Drops repeated points and points on a straight run, noting each one
    /// that lay between its neighbours so it can go back.
    fn filter(&mut self, start: usize, end: Option<usize>) -> usize {
        let mut end = end.unwrap_or(start);
        let mut p = start;
        loop {
            let mut again = false;
            let node = self.n(p);
            let (a, c) = (self.p(node.prev), self.p(node.next));
            if !node.steiner && (node.p == c || area(a, node.p, c) == 0.0) {
                if node.p != c
                    && node.p != a
                    && (node.p - a).dot(c - a) > 0.0
                    && (node.p - c).dot(a - c) > 0.0
                {
                    self.dropped
                        .push((self.n(node.prev).i, node.i, self.n(node.next).i));
                }
                self.remove(p);
                p = node.prev;
                end = p;
                if p == self.next(p) {
                    break;
                }
                again = true;
            } else {
                p = node.next;
            }
            if !again && p == end {
                break;
            }
        }
        end
    }

    fn is_ear(&self, ear: usize) -> bool {
        let (a, b, c) = (self.prev(ear), ear, self.next(ear));
        let (pa, pb, pc) = (self.p(a), self.p(b), self.p(c));
        if area(pa, pb, pc) >= 0.0 {
            return false;
        }
        let lo = pa.min(pb).min(pc);
        let hi = pa.max(pb).max(pc);
        let mut p = self.next(c);
        while p != a {
            let q = self.p(p);
            if q.x >= lo.x
                && q.x <= hi.x
                && q.y >= lo.y
                && q.y <= hi.y
                && point_in_triangle(pa, pb, pc, q)
                && area(self.p(self.prev(p)), q, self.p(self.next(p))) >= 0.0
            {
                return false;
            }
            p = self.next(p);
        }
        true
    }

    fn linked(&mut self, ear: Option<usize>, pass: u8) {
        let Some(mut ear) = ear else { return };
        let mut stop = ear;
        while self.prev(ear) != self.next(ear) {
            let (prev, next) = (self.prev(ear), self.next(ear));
            if self.is_ear(ear) {
                self.out
                    .push([self.n(prev).i, self.n(ear).i, self.n(next).i]);
                self.remove(ear);
                ear = self.next(next);
                stop = ear;
                continue;
            }
            ear = next;
            if ear == stop {
                match pass {
                    0 => {
                        let e = self.filter(ear, None);
                        self.linked(Some(e), 1);
                    }
                    1 => {
                        let e = self.filter(ear, None);
                        let e = self.cure(e);
                        self.linked(Some(e), 2);
                    }
                    _ => self.split(ear),
                }
                break;
            }
        }
    }

    fn cure(&mut self, start: usize) -> usize {
        let mut start = start;
        let mut p = start;
        loop {
            let a = self.prev(p);
            let b = self.next(self.next(p));
            let pn = self.next(p);
            if self.p(a) != self.p(b)
                && intersects(self.p(a), self.p(p), self.p(pn), self.p(b))
                && self.locally_inside(a, b)
                && self.locally_inside(b, a)
            {
                self.out.push([self.n(a).i, self.n(p).i, self.n(b).i]);
                self.remove(p);
                self.remove(pn);
                p = b;
                start = b;
            }
            p = self.next(p);
            if p == start {
                break;
            }
        }
        self.filter(p, None)
    }

    fn split(&mut self, start: usize) {
        let mut a = start;
        loop {
            let mut b = self.next(self.next(a));
            while b != self.prev(a) {
                if self.n(a).i != self.n(b).i && self.valid_diagonal(a, b) {
                    let c = self.split_polygon(a, b);
                    let a2 = self.filter(a, Some(self.next(a)));
                    let c2 = self.filter(c, Some(self.next(c)));
                    self.linked(Some(a2), 0);
                    self.linked(Some(c2), 0);
                    return;
                }
                b = self.next(b);
            }
            a = self.next(a);
            if a == start {
                break;
            }
        }
    }

    fn holes(&mut self, holes: Vec<usize>, mut outer: usize) -> usize {
        let mut queue: Vec<usize> = holes.into_iter().map(|h| self.leftmost(h)).collect();
        queue.sort_by(|a, b| self.p(*a).x.total_cmp(&self.p(*b).x));
        for h in queue {
            outer = self.hole(h, outer);
        }
        outer
    }

    fn hole(&mut self, hole: usize, outer: usize) -> usize {
        let Some(bridge) = self.bridge(hole, outer) else {
            return outer;
        };
        let reverse = self.split_polygon(bridge, hole);
        let rn = self.next(reverse);
        self.filter(reverse, Some(rn));
        let bn = self.next(bridge);
        self.filter(bridge, Some(bn))
    }

    /// The outside vertex a hole is joined to: the one its leftmost point
    /// sees first looking left (David Eberly's method).
    fn bridge(&self, hole: usize, outer: usize) -> Option<usize> {
        let h = self.p(hole);
        let mut p = outer;
        let mut qx = f64::NEG_INFINITY;
        let mut m = None;
        loop {
            let (a, b) = (self.p(p), self.p(self.next(p)));
            if h.y <= a.y && h.y >= b.y && b.y != a.y {
                let x = a.x + (h.y - a.y) * (b.x - a.x) / (b.y - a.y);
                if x <= h.x && x > qx {
                    qx = x;
                    m = Some(if a.x < b.x { p } else { self.next(p) });
                    if x == h.x {
                        return m;
                    }
                }
            }
            p = self.next(p);
            if p == outer {
                break;
            }
        }
        let mut m = m?;
        let stop = m;
        let mp = self.p(m);
        let mut tan_min = f64::INFINITY;
        p = m;
        loop {
            let q = self.p(p);
            let (t1, t3) = if h.y < mp.y {
                (DVec2::new(h.x, h.y), DVec2::new(qx, h.y))
            } else {
                (DVec2::new(qx, h.y), DVec2::new(h.x, h.y))
            };
            if h.x >= q.x && q.x >= mp.x && h.x != q.x && point_in_triangle(t1, mp, t3, q) {
                let tan = (h.y - q.y).abs() / (h.x - q.x);
                let mpx = self.p(m);
                if self.locally_inside(p, hole)
                    && (tan < tan_min
                        || (tan == tan_min
                            && (q.x > mpx.x || (q.x == mpx.x && self.sector_contains(m, p)))))
                {
                    m = p;
                    tan_min = tan;
                }
            }
            p = self.next(p);
            if p == stop {
                break;
            }
        }
        Some(m)
    }

    fn sector_contains(&self, m: usize, p: usize) -> bool {
        area(self.p(self.prev(m)), self.p(m), self.p(self.prev(p))) < 0.0
            && area(self.p(self.next(p)), self.p(m), self.p(self.next(m))) < 0.0
    }

    fn leftmost(&self, start: usize) -> usize {
        let mut p = start;
        let mut best = start;
        loop {
            let (q, b) = (self.p(p), self.p(best));
            if q.x < b.x || (q.x == b.x && q.y < b.y) {
                best = p;
            }
            p = self.next(p);
            if p == start {
                break;
            }
        }
        best
    }

    fn valid_diagonal(&self, a: usize, b: usize) -> bool {
        let (na, nb) = (self.n(a), self.n(b));
        let (pa, pb) = (na.p, nb.p);
        self.n(na.next).i != nb.i
            && self.n(na.prev).i != nb.i
            && !self.intersects_polygon(a, b)
            && ((self.locally_inside(a, b)
                && self.locally_inside(b, a)
                && self.middle_inside(a, b)
                && (area(self.p(na.prev), pa, self.p(nb.prev)) != 0.0
                    || area(pa, self.p(nb.prev), pb) != 0.0))
                || (pa == pb
                    && area(self.p(na.prev), pa, self.p(na.next)) > 0.0
                    && area(self.p(nb.prev), pb, self.p(nb.next)) > 0.0))
    }

    fn intersects_polygon(&self, a: usize, b: usize) -> bool {
        let (ia, ib) = (self.n(a).i, self.n(b).i);
        let mut p = a;
        loop {
            let n = self.next(p);
            let (ip, inx) = (self.n(p).i, self.n(n).i);
            if ip != ia
                && inx != ia
                && ip != ib
                && inx != ib
                && intersects(self.p(p), self.p(n), self.p(a), self.p(b))
            {
                return true;
            }
            p = n;
            if p == a {
                return false;
            }
        }
    }

    fn locally_inside(&self, a: usize, b: usize) -> bool {
        let (pp, pa, pn, pb) = (
            self.p(self.prev(a)),
            self.p(a),
            self.p(self.next(a)),
            self.p(b),
        );
        if area(pp, pa, pn) < 0.0 {
            area(pa, pb, pn) >= 0.0 && area(pa, pp, pb) >= 0.0
        } else {
            area(pa, pb, pp) < 0.0 || area(pa, pn, pb) < 0.0
        }
    }

    fn middle_inside(&self, a: usize, b: usize) -> bool {
        let mid = (self.p(a) + self.p(b)) * 0.5;
        let mut inside = false;
        let mut p = a;
        loop {
            let (q, r) = (self.p(p), self.p(self.next(p)));
            if (q.y > mid.y) != (r.y > mid.y)
                && r.y != q.y
                && mid.x < (r.x - q.x) * (mid.y - q.y) / (r.y - q.y) + q.x
            {
                inside = !inside;
            }
            p = self.next(p);
            if p == a {
                return inside;
            }
        }
    }

    /// Joins `a` and `b` with a diagonal, splitting the ring in two; returns
    /// a node of the second.
    fn split_polygon(&mut self, a: usize, b: usize) -> usize {
        let (na, nb) = (self.n(a), self.n(b));
        let a2 = self.nodes.len();
        self.nodes.push(Node {
            prev: a2,
            next: a2,
            ..na
        });
        let b2 = self.nodes.len();
        self.nodes.push(Node {
            prev: b2,
            next: b2,
            ..nb
        });
        let (an, bp) = (na.next, nb.prev);
        self.nodes[a].next = b;
        self.nodes[b].prev = a;
        self.nodes[a2].next = an;
        self.nodes[an].prev = a2;
        self.nodes[b2].next = a2;
        self.nodes[a2].prev = b2;
        self.nodes[bp].next = b2;
        self.nodes[b2].prev = bp;
        b2
    }
}

/// Triangles over `rings[0]` less the other rings. Indices are into the
/// rings laid end to end; every triangle runs anticlockwise, and every point
/// of every ring is a corner of some triangle.
pub fn triangulate(rings: &[Vec<DVec2>]) -> Vec<[usize; 3]> {
    let mut e = Earcut {
        nodes: Vec::new(),
        out: Vec::new(),
        dropped: Vec::new(),
    };
    let Some(first) = rings.first() else {
        return Vec::new();
    };
    if first.len() < 3 {
        return Vec::new();
    }
    let Some(mut outer) = e.ring(first, 0, true) else {
        return Vec::new();
    };
    if e.next(outer) == e.prev(outer) {
        return Vec::new();
    }
    let mut base = first.len();
    let mut holes = Vec::new();
    for r in &rings[1..] {
        if r.len() >= 3 {
            if let Some(h) = e.ring(r, base, false) {
                if h == e.next(h) {
                    e.nodes[h].steiner = true;
                }
                holes.push(h);
            }
        }
        base += r.len();
    }
    if !holes.is_empty() {
        outer = e.holes(holes, outer);
    }
    e.linked(Some(outer), 0);

    // The dropped points back, last first: a point dropped from an edge
    // that was itself left by an earlier drop goes onto that edge once it
    // is back.
    let mut tris = e.out;
    let mut by_edge: HashMap<(usize, usize), usize> = HashMap::new();
    for (t, tri) in tris.iter().enumerate() {
        for k in 0..3 {
            by_edge.insert((tri[k], tri[(k + 1) % 3]), t);
        }
    }
    for (a, b, c) in e.dropped.into_iter().rev() {
        let Some(t) = by_edge.remove(&(a, c)) else {
            continue;
        };
        let tri = tris[t];
        let Some(k) = (0..3).find(|k| tri[*k] == a && tri[(k + 1) % 3] == c) else {
            continue;
        };
        let x = tri[(k + 2) % 3];
        tris[t] = [a, b, x];
        let t2 = tris.len();
        tris.push([b, c, x]);
        by_edge.insert((a, b), t);
        by_edge.insert((b, x), t);
        by_edge.insert((b, c), t2);
        by_edge.insert((c, x), t2);
        by_edge.insert((x, b), t2);
        by_edge.insert((x, a), t);
    }
    tris
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area_of(rings: &[Vec<DVec2>], tris: &[[usize; 3]]) -> f64 {
        let pts: Vec<DVec2> = rings.iter().flatten().copied().collect();
        tris.iter()
            .map(|t| (pts[t[1]] - pts[t[0]]).perp_dot(pts[t[2]] - pts[t[0]]) * 0.5)
            .sum()
    }

    #[test]
    fn a_square_with_a_hole_has_its_area_and_every_point() {
        let outer = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(2.0, 0.0),
            DVec2::new(4.0, 0.0),
            DVec2::new(4.0, 4.0),
            DVec2::new(0.0, 4.0),
        ];
        let hole = vec![
            DVec2::new(1.0, 1.0),
            DVec2::new(1.0, 2.0),
            DVec2::new(2.0, 2.0),
            DVec2::new(2.0, 1.0),
        ];
        let rings = vec![outer, hole];
        let tris = triangulate(&rings);
        assert!((area_of(&rings, &tris) - 15.0).abs() < 1e-9);
        assert!(tris.iter().all(|t| {
            let pts: Vec<DVec2> = rings.iter().flatten().copied().collect();
            (pts[t[1]] - pts[t[0]]).perp_dot(pts[t[2]] - pts[t[0]]) >= 0.0
        }));
        // The point on the straight bottom edge is used.
        assert!(tris.iter().any(|t| t.contains(&1)));
    }

    #[test]
    fn many_holes_in_a_long_strip_do_not_overlap() {
        let mut rings = vec![vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(100.0, 0.0),
            DVec2::new(100.0, 3.0),
            DVec2::new(0.0, 3.0),
        ]];
        let mut hole_area = 0.0;
        for k in 0..12 {
            let c = DVec2::new(5.0 + 8.0 * k as f64, 1.5);
            let r = 0.4 + 0.05 * k as f64;
            rings.push(
                (0..24)
                    .map(|i| {
                        let a = i as f64 / 24.0 * std::f64::consts::TAU;
                        c + DVec2::new(a.cos(), a.sin()) * r
                    })
                    .collect(),
            );
            hole_area += 0.5 * 24.0 * r * r * (std::f64::consts::TAU / 24.0).sin();
        }
        let tris = triangulate(&rings);
        let total = area_of(&rings, &tris);
        assert!((total - (300.0 - hole_area)).abs() < 1e-6, "{total}");
    }
}
