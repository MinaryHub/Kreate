//! A section: the mesh cut by a plane, one side kept, the cut capped.
//!
//! What a section view on a drawing shows, and what a designer looks at to
//! see a wall's thickness or a part's inside. Every face is clipped to the
//! kept side; where the plane passed through the solid, the cut's outline is
//! closed with a cap — holes and all, so a tube's section is a ring and not
//! a disc.
//!
//! The cap is found from the cut itself: each clipped face contributes the
//! segment it was cut along, the segments meet at the points they share (an
//! edge is cut once, so its point is one vertex), and they chain into loops.
//! With the surface wound outward, an outline's loops run one way and its
//! holes the other, which is how each hole finds its outline.

use std::collections::HashMap;

use kreate_math::{Vec2, Vec3};

use crate::MeshData;

/// `mesh` with everything on the positive side of the plane (`normal`
/// pointing at it) removed, and the cut capped when `cap` is on.
pub fn section(mesh: &MeshData, point: Vec3, normal: Vec3, cap: bool) -> MeshData {
    let n = normal.normalize_or_zero();
    if n == Vec3::ZERO {
        return mesh.clone();
    }
    let positions = mesh.positions();
    let size = {
        let (lo, hi) = positions.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        (hi - lo).length().max(1e-6)
    };
    // A vertex on the plane is taken as kept, a hair inside, so no cut point
    // lands exactly on an existing vertex.
    let tol = size * 1e-6;
    let side: Vec<f32> = positions
        .iter()
        .map(|p| {
            let d = (*p - point).dot(n);
            if d.abs() < tol {
                -tol
            } else {
                d
            }
        })
        .collect();
    if side.iter().all(|d| *d <= 0.0) {
        return mesh.clone();
    }

    let mut out: Vec<Vec3> = Vec::new();
    let mut kept: Vec<Option<u32>> = vec![None; positions.len()];
    let mut keep = |v: u32, out: &mut Vec<Vec3>| -> u32 {
        *kept[v as usize].get_or_insert_with(|| {
            out.push(positions[v as usize]);
            out.len() as u32 - 1
        })
    };
    let mut cut_points: HashMap<(u32, u32), u32> = HashMap::new();
    let mut cut = |a: u32, b: u32, out: &mut Vec<Vec3>| -> u32 {
        let key = (a.min(b), a.max(b));
        *cut_points.entry(key).or_insert_with(|| {
            let (da, db) = (side[key.0 as usize], side[key.1 as usize]);
            let t = da / (da - db);
            let (pa, pb) = (positions[key.0 as usize], positions[key.1 as usize]);
            out.push(pa + (pb - pa) * t);
            out.len() as u32 - 1
        })
    };

    let mut faces: Vec<Vec<u32>> = Vec::new();
    // Directed along the cap's own winding: the reverse of the clipped face's
    // edge along the cut.
    let mut segments: Vec<(u32, u32)> = Vec::new();
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        let mut poly: Vec<u32> = Vec::with_capacity(ring.len() + 2);
        let mut exit: Option<u32> = None;
        let mut first_entry: Option<u32> = None;
        let mut pending_exit_at_start = false;
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            let (ina, inb) = (side[a as usize] <= 0.0, side[b as usize] <= 0.0);
            if ina {
                poly.push(keep(a, &mut out));
            }
            if ina != inb {
                let x = cut(a, b, &mut out);
                poly.push(x);
                if ina {
                    exit = Some(x);
                } else {
                    // Entering: the face ran exit -> x along the cut.
                    match exit.take() {
                        Some(e) => segments.push((x, e)),
                        None => {
                            first_entry = Some(x);
                            pending_exit_at_start = true;
                        }
                    }
                }
            }
        }
        // An exit late in the ring pairs with the entry at its start.
        if let (true, Some(e), Some(x)) = (pending_exit_at_start, exit, first_entry) {
            segments.push((x, e));
        }
        if poly.len() >= 3 {
            faces.push(poly);
        }
    }

    if cap {
        cap_faces(&out, &segments, n, &mut faces);
    }
    MeshData::from_faces(out, &faces)
}

/// Chains the cut's segments into loops and fills them, holes inside their
/// outlines, as triangles.
fn cap_faces(points: &[Vec3], segments: &[(u32, u32)], normal: Vec3, faces: &mut Vec<Vec<u32>>) {
    let next: HashMap<u32, u32> = segments.iter().copied().collect();
    let mut used: HashMap<u32, bool> = HashMap::new();
    let mut loops: Vec<Vec<u32>> = Vec::new();
    for &(start, _) in segments {
        if used.get(&start).copied().unwrap_or(false) {
            continue;
        }
        let mut chain = vec![start];
        used.insert(start, true);
        let mut at = start;
        let closed = loop {
            let Some(&to) = next.get(&at) else {
                break false;
            };
            if to == start {
                break true;
            }
            if used.get(&to).copied().unwrap_or(false) {
                break false;
            }
            used.insert(to, true);
            chain.push(to);
            at = to;
        };
        // An open chain is a cut through an open surface: nothing to close.
        if closed && chain.len() >= 3 {
            loops.push(chain);
        }
    }
    if loops.is_empty() {
        return;
    }
    // A basis in the plane, with the cap's normal as its up.
    let u = normal.any_orthonormal_vector();
    let v = normal.cross(u);
    let flat = |i: u32| {
        let p = points[i as usize];
        Vec2::new(p.dot(u), p.dot(v))
    };
    for [a, b, c] in fill_loops(loops, &flat) {
        faces.push(vec![a, b, c]);
    }
}

/// Fills closed loops lying in a plane, as triangles: loops running
/// counter-clockwise (as `flat` sees them) are outlines, the others holes,
/// each hole in the smallest outline around it.
pub(crate) fn fill_loops(loops: Vec<Vec<u32>>, flat: &impl Fn(u32) -> Vec2) -> Vec<[u32; 3]> {
    let area = |l: &[u32]| -> f32 {
        (0..l.len())
            .map(|i| {
                let (a, b) = (flat(l[i]), flat(l[(i + 1) % l.len()]));
                a.x * b.y - b.x * a.y
            })
            .sum::<f32>()
            * 0.5
    };
    let (outlines, holes): (Vec<_>, Vec<_>) = loops.into_iter().partition(|l| area(l) > 0.0);
    let mut groups: Vec<(Vec<u32>, Vec<Vec<u32>>)> =
        outlines.into_iter().map(|o| (o, Vec::new())).collect();
    for hole in holes {
        let probe = flat(hole[0]);
        let home = groups
            .iter_mut()
            .filter(|(o, _)| contains(&o.iter().map(|i| flat(*i)).collect::<Vec<_>>(), probe))
            .min_by(|a, b| area(&a.0).total_cmp(&area(&b.0)));
        if let Some((_, hs)) = home {
            hs.push(hole);
        }
    }
    let mut out = Vec::new();
    for (outline, holes) in groups {
        let polygon = bridge(outline, holes, flat);
        let flat_points: Vec<Vec2> = polygon.iter().map(|i| flat(*i)).collect();
        for [a, b, c] in ear_clip(&flat_points) {
            out.push([polygon[a], polygon[b], polygon[c]]);
        }
    }
    out
}

/// Even-odd containment.
fn contains(polygon: &[Vec2], p: Vec2) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let (a, b) = (polygon[i], polygon[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// One polygon from an outline and its holes: each hole joined to the
/// outline by a pair of edges from its rightmost point to the nearest outline
/// point it can see — the keyhole, which an ear clipper then fills.
fn bridge(outline: Vec<u32>, mut holes: Vec<Vec<u32>>, flat: &impl Fn(u32) -> Vec2) -> Vec<u32> {
    let mut polygon = outline;
    // Rightmost holes first, so a later bridge never crosses an earlier one.
    holes.sort_by(|a, b| {
        let ra = a.iter().map(|i| flat(*i).x).fold(f32::MIN, f32::max);
        let rb = b.iter().map(|i| flat(*i).x).fold(f32::MIN, f32::max);
        rb.total_cmp(&ra)
    });
    for hole in holes {
        let (hi, _) = hole
            .iter()
            .enumerate()
            .max_by(|a, b| flat(*a.1).x.total_cmp(&flat(*b.1).x))
            .expect("a hole has points");
        let h = flat(hole[hi]);
        // The nearest outline point to the right that no edge blocks; failing
        // that, simply the nearest.
        let blocked = |p: Vec2, polygon: &[u32]| {
            (0..polygon.len()).any(|k| {
                let (a, b) = (flat(polygon[k]), flat(polygon[(k + 1) % polygon.len()]));
                segments_cross(h, p, a, b)
            })
        };
        let candidates: Vec<usize> = (0..polygon.len()).collect();
        let best = candidates
            .iter()
            .copied()
            .filter(|k| !blocked(flat(polygon[*k]), &polygon))
            .min_by(|a, b| {
                flat(polygon[*a])
                    .distance_squared(h)
                    .total_cmp(&flat(polygon[*b]).distance_squared(h))
            })
            .or_else(|| {
                candidates.iter().copied().min_by(|a, b| {
                    flat(polygon[*a])
                        .distance_squared(h)
                        .total_cmp(&flat(polygon[*b]).distance_squared(h))
                })
            })
            .expect("an outline has points");
        // outline .. best, hole from hi round to hi, best again, outline on.
        let mut joined = polygon[..=best].to_vec();
        for k in 0..=hole.len() {
            joined.push(hole[(hi + k) % hole.len()]);
        }
        joined.extend_from_slice(&polygon[best..]);
        polygon = joined;
    }
    polygon
}

/// Whether segments pq and ab cross at a point inside both (touching at an
/// end does not count).
fn segments_cross(p: Vec2, q: Vec2, a: Vec2, b: Vec2) -> bool {
    let d = |o: Vec2, x: Vec2, y: Vec2| (x - o).perp_dot(y - o);
    let (d1, d2) = (d(p, q, a), d(p, q, b));
    let (d3, d4) = (d(a, b, p), d(a, b, q));
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// Ear clipping for a counter-clockwise polygon that may visit a point twice
/// (a keyhole's bridge): a point that coincides with an ear's corner does not
/// block it.
fn ear_clip(points: &[Vec2]) -> Vec<[usize; 3]> {
    let mut idx: Vec<usize> = (0..points.len()).collect();
    let mut out = Vec::new();
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let mut guard = idx.len() * idx.len() + 8;
    while idx.len() > 3 && guard > 0 {
        guard -= 1;
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let (ia, ib, ic) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (a, b, c) = (points[ia], points[ib], points[ic]);
            if cross(a, b, c) <= 0.0 {
                continue;
            }
            let blocks = idx.iter().any(|&k| {
                let p = points[k];
                if p == a || p == b || p == c {
                    return false;
                }
                cross(a, b, p) >= 0.0 && cross(b, c, p) >= 0.0 && cross(c, a, p) >= 0.0
            });
            if blocks {
                continue;
            }
            out.push([ia, ib, ic]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        let (a, b, c) = (points[idx[0]], points[idx[1]], points[idx[2]]);
        if cross(a, b, c) > 0.0 {
            out.push([idx[0], idx[1], idx[2]]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::mass_properties;
    use crate::primitives;

    fn volume(m: &MeshData) -> f64 {
        mass_properties(m, |p| p).volume
    }

    #[test]
    fn half_a_cube_is_half_its_volume_and_closed() {
        let m = section(&primitives::cube(2.0), Vec3::ZERO, Vec3::X, true);
        assert!((volume(&m) - 4.0).abs() < 1e-4, "{}", volume(&m));
        assert!(crate::measure::is_closed(&m));
        assert!(m.positions().iter().all(|p| p.x <= 1e-4));
        assert_eq!(m.validate(), Vec::new());
    }

    #[test]
    fn without_a_cap_the_cut_is_open() {
        let m = section(&primitives::cube(2.0), Vec3::ZERO, Vec3::X, false);
        assert!(!crate::measure::is_closed(&m));
    }

    #[test]
    fn a_plane_that_misses_leaves_the_mesh_alone() {
        let cube = primitives::cube(2.0);
        let m = section(&cube, Vec3::new(5.0, 0.0, 0.0), Vec3::X, true);
        assert_eq!(m.num_faces(), cube.num_faces());
    }

    #[test]
    fn a_tubes_section_is_a_ring_not_a_disc() {
        // A thick-walled tube: a torus cut through its middle has two round
        // holes' worth of cap, and a ring's cap where it is cut across the
        // ring. Here: a torus cut by the plane through its axis gives two
        // discs; cut across (normal along the axis) through the middle it
        // gives a ring, whose cap must leave the middle open.
        let torus = primitives::torus(1.0, 0.3, 48, 24);
        let full = volume(&torus);
        let half = section(&torus, Vec3::ZERO, Vec3::Z, true);
        assert!(
            crate::measure::is_closed(&half),
            "the ring is capped all round"
        );
        assert!(
            (volume(&half) - full * 0.5).abs() / full < 0.01,
            "{} of {full}",
            volume(&half)
        );
        // Nothing of the cap reaches the hole in the middle.
        let middle = half
            .positions()
            .iter()
            .filter(|p| p.z.abs() < 1e-4)
            .map(|p| (p.x * p.x + p.y * p.y).sqrt())
            .fold(f32::MAX, f32::min);
        assert!(middle > 0.6, "the cap stays on the ring: {middle}");
        let m = half.triangulate();
        let covered_hole = m.indices.as_chunks::<3>().0.iter().any(|t| {
            let c = t
                .iter()
                .map(|i| half.positions()[*i as usize])
                .sum::<Vec3>()
                / 3.0;
            c.z.abs() < 1e-4 && (c.x * c.x + c.y * c.y).sqrt() < 0.6
        });
        assert!(!covered_hole, "a triangle of the cap lies over the hole");
    }
}
