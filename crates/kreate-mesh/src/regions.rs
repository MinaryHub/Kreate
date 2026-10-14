//! The regions a sketch's lines enclose, as Fusion picks them: the drawing
//! split wherever lines cross, and each area bounded by it a profile of its
//! own — two overlapping circles are three regions, a circle inside a square
//! is a disc and a square with a hole.
//!
//! The lines are the sketch's edges in its XY plane; faces are not needed.
//! Every region is an outline and the holes in it, ready to extrude or turn.

use kreate_math::{Vec2, Vec3};

use crate::MeshData;

/// One enclosed area: its outline anticlockwise, its holes clockwise.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub outline: Vec<Vec2>,
    pub holes: Vec<Vec<Vec2>>,
}

impl Region {
    /// Its area, holes taken out.
    pub fn area(&self) -> f32 {
        signed(&self.outline) + self.holes.iter().map(|h| signed(h)).sum::<f32>()
    }

    /// Whether `p` is in it: inside the outline and in no hole.
    pub fn contains(&self, p: Vec2) -> bool {
        inside(&self.outline, p) && !self.holes.iter().any(|h| inside(h, p))
    }
}

fn signed(l: &[Vec2]) -> f32 {
    (0..l.len())
        .map(|i| l[i].perp_dot(l[(i + 1) % l.len()]))
        .sum::<f32>()
        * 0.5
}

/// Even-odd containment.
fn inside(polygon: &[Vec2], p: Vec2) -> bool {
    let mut odd = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let (a, b) = (polygon[i], polygon[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            odd = !odd;
        }
        j = i;
    }
    odd
}

/// The regions of a sketch mesh: its edges in its XY plane.
pub fn sketch_regions(mesh: &MeshData) -> Vec<Region> {
    let points: Vec<Vec2> = mesh
        .positions()
        .iter()
        .map(|p| Vec2::new(p.x, p.y))
        .collect();
    regions(&points, mesh.edges())
}

/// The regions `edges` between `points` enclose, crossings split.
pub fn regions(points: &[Vec2], edges: &[[u32; 2]]) -> Vec<Region> {
    let (nodes, links) = arrange(points, edges);
    let links = prune(nodes.len(), links);
    if links.is_empty() {
        return Vec::new();
    }
    // Each node's neighbours by angle.
    let mut around: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for &[a, b] in &links {
        around[a].push(b);
        around[b].push(a);
    }
    for (n, list) in around.iter_mut().enumerate() {
        let here = nodes[n];
        list.sort_by(|a, b| {
            let (da, db) = (nodes[*a] - here, nodes[*b] - here);
            da.y.atan2(da.x).total_cmp(&db.y.atan2(db.x))
        });
        list.dedup();
    }
    // Walk every directed edge once, keeping the face on the left: at each
    // node, leave by the edge just clockwise of the one arrived by.
    let mut used = std::collections::HashSet::new();
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for &[a, b] in &links {
        for (from, to) in [(a, b), (b, a)] {
            if used.contains(&(from, to)) {
                continue;
            }
            let mut cycle = Vec::new();
            let (mut u, mut v) = (from, to);
            let mut guard = 0;
            loop {
                used.insert((u, v));
                cycle.push(u);
                let list = &around[v];
                let back = list.iter().position(|w| *w == u).expect("a link both ways");
                let next = list[(back + list.len() - 1) % list.len()];
                u = v;
                v = next;
                guard += 1;
                if (u, v) == (from, to) || guard > links.len() * 2 + 4 {
                    break;
                }
            }
            if cycle.len() >= 3 {
                cycles.push(cycle);
            }
        }
    }
    let polygon = |c: &[usize]| c.iter().map(|i| nodes[*i]).collect::<Vec<Vec2>>();
    // Anticlockwise cycles bound areas; clockwise ones are the outsides of
    // connected pieces, which go as holes in whatever area holds them.
    let mut faces: Vec<Region> = Vec::new();
    let mut outsides: Vec<Vec<Vec2>> = Vec::new();
    for c in &cycles {
        let p = polygon(c);
        let a = signed(&p);
        if a > 1e-12 {
            faces.push(Region {
                outline: p,
                holes: Vec::new(),
            });
        } else if a < -1e-12 {
            outsides.push(p);
        }
    }
    for outside in outsides {
        let probe = outside[0];
        // The smallest area round it that is not itself: a piece's own faces
        // share its points, so they hold none of them strictly inside.
        let holder = faces
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                outside
                    .iter()
                    .all(|p| inside(&f.outline, *p) && !f.outline.contains(p))
            })
            .min_by(|(_, a), (_, b)| signed(&a.outline).total_cmp(&signed(&b.outline)))
            .map(|(i, _)| i);
        let _ = probe;
        if let Some(i) = holder {
            faces[i].holes.push(outside);
        }
    }
    faces
}

/// The drawing as a plane graph: points merged where they meet, edges split
/// where they cross or where a point lies on them.
fn arrange(points: &[Vec2], edges: &[[u32; 2]]) -> (Vec<Vec2>, Vec<[usize; 2]>) {
    let extent = points
        .iter()
        .fold(0.0f32, |m, p| m.max(p.x.abs()).max(p.y.abs()))
        .max(1e-6);
    let eps = extent * 1e-5;
    let segs: Vec<(Vec2, Vec2)> = edges
        .iter()
        .filter_map(|e| Some((*points.get(e[0] as usize)?, *points.get(e[1] as usize)?)))
        .filter(|(a, b)| a.distance(*b) > eps)
        .collect();
    // Where along each segment it is cut, from 0 to 1.
    let mut cuts: Vec<Vec<f32>> = segs.iter().map(|_| vec![0.0, 1.0]).collect();
    for i in 0..segs.len() {
        for j in i + 1..segs.len() {
            let (p, r) = (segs[i].0, segs[i].1 - segs[i].0);
            let (q, s) = (segs[j].0, segs[j].1 - segs[j].0);
            let denom = r.perp_dot(s);
            if denom.abs() > 1e-12 {
                let t = (q - p).perp_dot(s) / denom;
                let u = (q - p).perp_dot(r) / denom;
                let (ti, tj) = (eps / r.length(), eps / s.length());
                if (-ti..=1.0 + ti).contains(&t) && (-tj..=1.0 + tj).contains(&u) {
                    cuts[i].push(t.clamp(0.0, 1.0));
                    cuts[j].push(u.clamp(0.0, 1.0));
                }
            } else {
                // Parallel: where one's ends lie on the other.
                for (k, other) in [(i, j), (j, i)] {
                    let (a, d) = (segs[k].0, segs[k].1 - segs[k].0);
                    for end in [segs[other].0, segs[other].1] {
                        let t = (end - a).dot(d) / d.length_squared();
                        let off = d.perp_dot(end - a).abs() / d.length();
                        if off < eps && (0.0..=1.0).contains(&t) {
                            cuts[k].push(t);
                        }
                    }
                }
            }
        }
    }
    let mut nodes: Vec<Vec2> = Vec::new();
    let node_of = |p: Vec2, nodes: &mut Vec<Vec2>| -> usize {
        if let Some(i) = nodes.iter().position(|q| q.distance(p) <= eps) {
            return i;
        }
        nodes.push(p);
        nodes.len() - 1
    };
    let mut links: Vec<[usize; 2]> = Vec::new();
    for (k, (a, b)) in segs.iter().enumerate() {
        let mut ts = cuts[k].clone();
        ts.sort_by(f32::total_cmp);
        let ids: Vec<usize> = ts
            .iter()
            .map(|t| node_of(*a + (*b - *a) * *t, &mut nodes))
            .collect();
        for w in ids.windows(2) {
            if w[0] != w[1] {
                let link = [w[0].min(w[1]), w[0].max(w[1])];
                if !links.contains(&link) {
                    links.push(link);
                }
            }
        }
    }
    (nodes, links)
}

/// Without the lines that go nowhere: an end with one line on it bounds
/// nothing, and neither does the line, once it is gone, to its other end.
fn prune(nodes: usize, mut links: Vec<[usize; 2]>) -> Vec<[usize; 2]> {
    loop {
        let mut degree = vec![0usize; nodes];
        for [a, b] in &links {
            degree[*a] += 1;
            degree[*b] += 1;
        }
        let before = links.len();
        links.retain(|[a, b]| degree[*a] > 1 && degree[*b] > 1);
        if links.len() == before {
            return links;
        }
    }
}

/// Which region holds `p`: the innermost, when one is inside another's hole.
pub fn region_at(regions: &[Region], p: Vec2) -> Option<usize> {
    regions
        .iter()
        .enumerate()
        .filter(|(_, r)| r.contains(p))
        .min_by(|(_, a), (_, b)| a.area().total_cmp(&b.area()))
        .map(|(i, _)| i)
}

/// A region's caps as triangles over its points (the outline then each hole,
/// laid end to end, as `loops_of` numbers them), each anticlockwise. Through
/// earcut, which fills an outline with holes whatever their shapes — the
/// keyhole filler before it returned nothing for a plate of one width with
/// a round hole, and its walls stood without a top or a bottom.
///
/// Then merged back into as few faces as hold no hole: the wireframe draws
/// every edge, and a plate's top drawn as earcut's fan of slivers reads as
/// streaks across a flat face. A plate without a hole comes back as one
/// face, one with a hole as two or three.
fn caps(region: &Region) -> Vec<Vec<u32>> {
    let rings: Vec<Vec<kreate_math::DVec2>> = std::iter::once(&region.outline)
        .chain(&region.holes)
        .map(|l| l.iter().map(|p| p.as_dvec2()).collect())
        .collect();
    let triangles: Vec<Vec<u32>> = crate::earcut::triangulate(&rings)
        .into_iter()
        .map(|[a, b, c]| vec![a as u32, b as u32, c as u32])
        .collect();
    merged(triangles)
}

/// Anticlockwise faces of one plane merged across their shared edges while
/// each stays a simple ring — one that never comes back to a point. Across
/// an edge `u → v` of one face and `v → u` of the other, the merged ring is
/// the first from `v` round to `u` and then the second's points between.
pub(crate) fn merged(mut faces: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    loop {
        let mut joined = false;
        let mut owner: std::collections::HashMap<(u32, u32), usize> = Default::default();
        for (i, f) in faces.iter().enumerate() {
            for k in 0..f.len() {
                owner.insert((f[k], f[(k + 1) % f.len()]), i);
            }
        }
        let mut gone = vec![false; faces.len()];
        for i in 0..faces.len() {
            if gone[i] {
                continue;
            }
            let mut k = 0;
            while k < faces[i].len() {
                let (u, v) = (faces[i][k], faces[i][(k + 1) % faces[i].len()]);
                let Some(&j) = owner.get(&(v, u)) else {
                    k += 1;
                    continue;
                };
                if j == i || gone[j] {
                    k += 1;
                    continue;
                }
                let a = &faces[i];
                let b = &faces[j];
                let start_a = (k + 1) % a.len();
                let ring_a = (0..a.len()).map(|t| a[(start_a + t) % a.len()]);
                let at_u = b.iter().position(|x| *x == u).unwrap_or(0);
                let ring_b = (1..b.len() - 1).map(|t| b[(at_u + t) % b.len()]);
                let ring: Vec<u32> = ring_a.chain(ring_b).collect();
                let mut seen = std::collections::HashSet::new();
                if !ring.iter().all(|x| seen.insert(*x)) {
                    k += 1;
                    continue;
                }
                for t in 0..ring.len() {
                    owner.insert((ring[t], ring[(t + 1) % ring.len()]), i);
                }
                faces[i] = ring;
                gone[j] = true;
                joined = true;
                k = 0;
            }
        }
        faces = faces
            .into_iter()
            .zip(gone)
            .filter(|(_, g)| !g)
            .map(|(f, _)| f)
            .collect();
        if !joined {
            return faces;
        }
    }
}

/// Points and loops for a region: the outline then each hole, as indices
/// into the points.
fn loops_of(region: &Region) -> (Vec<Vec2>, Vec<Vec<u32>>) {
    let mut points = Vec::new();
    let mut loops = Vec::new();
    for l in std::iter::once(&region.outline).chain(&region.holes) {
        let start = points.len() as u32;
        points.extend(l.iter().copied());
        loops.push((start..start + l.len() as u32).collect());
    }
    (points, loops)
}

/// A region pushed from `z0` to `z1` along Z: a closed solid, holes through.
pub fn extrude_region(region: &Region, z0: f32, z1: f32) -> MeshData {
    let (points, loops) = loops_of(region);
    if (z1 - z0).abs() < 1e-9 || points.len() < 3 {
        return MeshData::empty();
    }
    let (lo, hi) = (z0.min(z1), z0.max(z1));
    let n = points.len() as u32;
    let mut verts: Vec<Vec3> = points.iter().map(|p| Vec3::new(p.x, p.y, lo)).collect();
    verts.extend(points.iter().map(|p| Vec3::new(p.x, p.y, hi)));
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for cap in caps(region) {
        faces.push(cap.iter().rev().copied().collect());
        faces.push(cap.iter().map(|v| v + n).collect());
    }
    for l in &loops {
        for k in 0..l.len() {
            let (a, b) = (l[k], l[(k + 1) % l.len()]);
            faces.push(vec![a, b, b + n, a + n]);
        }
    }
    outward(MeshData::from_faces(verts, &faces))
}

/// A region turned `angle` radians about the line through `origin` along
/// `axis`, both in its plane, in `segments` steps for a whole turn: a lathe,
/// closed with the region itself at both ends when it does not go all the
/// way round.
pub fn revolve_region(
    region: &Region,
    origin: Vec2,
    axis: Vec2,
    angle: f32,
    segments: u32,
) -> MeshData {
    let (points, loops) = loops_of(region);
    let axis = axis.normalize_or_zero();
    if axis == Vec2::ZERO || points.len() < 3 || angle.abs() < 1e-6 {
        return MeshData::empty();
    }
    let side = Vec2::new(-axis.y, axis.x);
    let full = angle.abs() >= std::f32::consts::TAU - 1e-4;
    let turn = if full { std::f32::consts::TAU } else { angle };
    let steps =
        ((segments.max(3) as f32 * turn.abs() / std::f32::consts::TAU).ceil() as u32).max(1);
    let rings = if full { steps } else { steps + 1 };
    let n = points.len() as u32;
    let mut verts: Vec<Vec3> = Vec::new();
    for s in 0..rings {
        let a = turn * s as f32 / steps as f32;
        for p in &points {
            let d = *p - origin;
            let along = d.dot(axis);
            let out = d.dot(side);
            let flat = origin + axis * along + side * (out * a.cos());
            verts.push(Vec3::new(flat.x, flat.y, out * a.sin()));
        }
    }
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for s in 0..steps {
        let (r0, r1) = (s * n, ((s + 1) % rings) * n);
        for l in &loops {
            for k in 0..l.len() {
                let (a, b) = (l[k], l[(k + 1) % l.len()]);
                faces.push(vec![r0 + a, r0 + b, r1 + b, r1 + a]);
            }
        }
    }
    if !full {
        let last = steps * n;
        for cap in caps(region) {
            faces.push(cap.iter().rev().copied().collect());
            faces.push(cap.iter().map(|v| v + last).collect());
        }
    }
    outward(MeshData::from_faces(verts, &faces))
}

/// Turned to face out: a solid's signed volume is positive when its faces
/// do, whichever way round the loops were walked.
fn outward(mesh: MeshData) -> MeshData {
    let p = mesh.positions();
    let mut volume = 0.0f64;
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        for i in 1..ring.len().saturating_sub(1) {
            let (a, b, c) = (
                p[ring[0] as usize].as_dvec3(),
                p[ring[i] as usize].as_dvec3(),
                p[ring[i + 1] as usize].as_dvec3(),
            );
            volume += a.dot(b.cross(c));
        }
    }
    if volume >= 0.0 {
        return mesh;
    }
    let faces: Vec<Vec<u32>> = (0..mesh.num_faces())
        .map(|f| mesh.face_verts(f).iter().rev().copied().collect())
        .collect();
    MeshData::from_faces(p.to_vec(), &faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::{is_closed, mass_properties};

    /// A sketch's lines as points and edges: each loop closed.
    fn drawing(loops: &[Vec<Vec2>]) -> (Vec<Vec2>, Vec<[u32; 2]>) {
        let mut points = Vec::new();
        let mut edges = Vec::new();
        for l in loops {
            let s = points.len() as u32;
            points.extend(l.iter().copied());
            for i in 0..l.len() as u32 {
                edges.push([s + i, s + (i + 1) % l.len() as u32]);
            }
        }
        (points, edges)
    }

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Vec2> {
        vec![
            Vec2::new(x0, y0),
            Vec2::new(x1, y0),
            Vec2::new(x1, y1),
            Vec2::new(x0, y1),
        ]
    }

    fn circle(c: Vec2, r: f32, n: usize) -> Vec<Vec2> {
        (0..n)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / n as f32;
                c + Vec2::new(a.cos(), a.sin()) * r
            })
            .collect()
    }

    #[test]
    fn a_square_is_one_region() {
        let (p, e) = drawing(&[rect(0.0, 0.0, 2.0, 1.0)]);
        let r = regions(&p, &e);
        assert_eq!(r.len(), 1);
        assert!((r[0].area() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn a_circle_in_a_square_is_a_disc_and_a_square_with_a_hole() {
        let (p, e) = drawing(&[
            rect(0.0, 0.0, 4.0, 2.0),
            circle(Vec2::new(2.0, 1.0), 0.5, 32),
        ]);
        let r = regions(&p, &e);
        assert_eq!(r.len(), 2, "{r:?}");
        let disc = region_at(&r, Vec2::new(2.0, 1.0)).expect("the disc");
        let plate = region_at(&r, Vec2::new(0.3, 0.3)).expect("the plate");
        assert_ne!(disc, plate);
        assert_eq!(r[plate].holes.len(), 1);
        let circle_area = r[disc].area();
        assert!((r[plate].area() - (8.0 - circle_area)).abs() < 1e-4);
    }

    #[test]
    fn two_overlapping_squares_are_three_regions() {
        let (p, e) = drawing(&[rect(0.0, 0.0, 2.0, 2.0), rect(1.0, 1.0, 3.0, 3.0)]);
        let r = regions(&p, &e);
        assert_eq!(r.len(), 3, "{r:?}");
        let middle = region_at(&r, Vec2::new(1.5, 1.5)).expect("the overlap");
        assert!((r[middle].area() - 1.0).abs() < 1e-5);
        let total: f32 = r.iter().map(|x| x.area()).sum();
        assert!((total - 7.0).abs() < 1e-4);
    }

    #[test]
    fn a_line_across_a_square_splits_it_and_a_loose_line_bounds_nothing() {
        let mut l = drawing(&[rect(0.0, 0.0, 2.0, 1.0)]);
        // A line from the bottom edge's middle to the top's, and a stray.
        let s = l.0.len() as u32;
        l.0.extend([
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(5.0, 5.0),
            Vec2::new(6.0, 5.0),
        ]);
        l.1.extend([[s, s + 1], [s + 2, s + 3]]);
        let r = regions(&l.0, &l.1);
        assert_eq!(r.len(), 2, "{r:?}");
        for x in &r {
            assert!((x.area() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn a_plate_with_a_hole_extrudes_to_its_volume() {
        let (p, e) = drawing(&[
            rect(0.0, 0.0, 0.04, 0.02),
            circle(Vec2::new(0.02, 0.01), 0.005, 48),
        ]);
        let r = regions(&p, &e);
        let plate = region_at(&r, Vec2::new(0.002, 0.002)).expect("plate");
        let solid = extrude_region(&r[plate], 0.0, 0.01);
        assert!(is_closed(&solid));
        let v = mass_properties(&solid, |p| p).volume;
        let want = r[plate].area() as f64 * 0.01;
        assert!((v - want).abs() < want * 1e-4, "{v} against {want}");
        // Symmetric: both ways, the same volume.
        let both = extrude_region(&r[plate], -0.005, 0.005);
        assert!((mass_properties(&both, |p| p).volume - want).abs() < want * 1e-4);
    }

    #[test]
    fn caps_are_a_few_faces_not_a_fan_of_slivers() {
        let (p, e) = drawing(&[rect(0.0, 0.0, 2.0, 1.0)]);
        let r = regions(&p, &e);
        // A box: two caps and four walls.
        assert_eq!(extrude_region(&r[0], 0.0, 1.0).num_faces(), 6);

        let (p, e) = drawing(&[
            rect(0.0, 0.0, 0.04, 0.02),
            circle(Vec2::new(0.02, 0.01), 0.005, 48),
        ]);
        let r = regions(&p, &e);
        let plate = region_at(&r, Vec2::new(0.002, 0.002)).expect("plate");
        let solid = extrude_region(&r[plate], 0.0, 0.01);
        let walls = 4 + 48;
        let caps = solid.num_faces() as usize - walls;
        // No face can hold a hole, so a cap is at least two; earcut's fan
        // was 52 a side.
        assert!((4..=8).contains(&caps), "{caps} cap faces");
        assert!(is_closed(&solid));
        let v = mass_properties(&solid, |p| p).volume;
        let want = r[plate].area() as f64 * 0.01;
        assert!((v - want).abs() < want * 1e-4, "{v} against {want}");
    }

    #[test]
    fn a_square_turned_round_an_axis_is_a_ring_and_a_quarter_turn_a_quarter() {
        let (p, e) = drawing(&[rect(1.0, 0.0, 2.0, 1.0)]);
        let r = regions(&p, &e);
        let full = revolve_region(&r[0], Vec2::ZERO, Vec2::Y, std::f32::consts::TAU, 128);
        assert!(is_closed(&full));
        let v = mass_properties(&full, |p| p).volume;
        let want = std::f64::consts::PI * (4.0 - 1.0) * 1.0;
        assert!((v - want).abs() < want * 2e-3, "{v} against {want}");
        let quarter = revolve_region(&r[0], Vec2::ZERO, Vec2::Y, std::f32::consts::FRAC_PI_2, 128);
        assert!(is_closed(&quarter));
        let q = mass_properties(&quarter, |p| p).volume;
        assert!((q - want / 4.0).abs() < want * 2e-3, "{q}");
    }
}
