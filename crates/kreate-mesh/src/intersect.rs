//! Whether two solids run into each other.
//!
//! What a bounding box cannot answer: a chair pushed under a table overlaps it
//! as boxes and touches nothing, and a vase standing on the table touches it
//! along a face without going into it. The question here is the real one —
//! do the surfaces cross, or is one inside the other — on triangles in world
//! space, so the caller has already placed both.
//!
//! # Touching is not crossing
//!
//! Two surfaces that only meet — a face resting on a face, an edge along a
//! face — are not reported. A segment that lies in the other triangle's plane
//! is parallel to it and does not cross it, and a crossing within a hair of a
//! segment's end or of a triangle's edge is a contact. That is the answer a
//! person placing objects wants: the vase is on the table, not in it.
//!
//! # Cost
//!
//! Only the triangles inside the region where the two boxes overlap take part,
//! and the second solid's are binned on a grid over that region, so each
//! triangle of the first is tested against its neighbours rather than all of
//! them.

use kreate_math::Vec3;

/// A triangle in world space.
pub type Triangle = [Vec3; 3];

/// How two solids meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contact {
    /// Apart, or only touching.
    Clear,
    /// Their surfaces cross.
    Crossing,
    /// One is wholly inside the other: `true` when the first is inside the
    /// second.
    Inside { first_in_second: bool },
}

/// Distances below this, relative to the region's size, count as touching.
const TOUCH: f32 = 1e-5;
/// Grid cells along each axis of the overlap region.
const CELLS: usize = 16;

/// How solids `a` and `b`, given as world-space triangles, meet.
///
/// "Inside" asks whether a point of one is enclosed by the other, which means
/// something only for a closed surface; for an open one (a plane) it is never
/// reported.
pub fn contact(a: &[Triangle], b: &[Triangle]) -> Contact {
    let (Some(box_a), Some(box_b)) = (bounds(a), bounds(b)) else {
        return Contact::Clear;
    };
    let lo = box_a.0.max(box_b.0);
    let hi = box_a.1.min(box_b.1);
    let size = (box_a.1 - box_a.0).max(box_b.1 - box_b.0).max_element().max(1e-6);
    let eps = TOUCH * size;
    if (hi - lo).min_element() < -eps {
        return Contact::Clear;
    }

    if surfaces_cross(a, b, lo - Vec3::splat(eps), hi + Vec3::splat(eps), eps) {
        return Contact::Crossing;
    }
    // No crossing: either apart, or one entirely within the other. A point
    // of the smaller decides — and only a point that is inside the other's
    // box can be inside it at all.
    if let Some(p) = a.first().map(|t| t[0]).filter(|p| inside_box(*p, box_b)) {
        if encloses(b, p, size) {
            return Contact::Inside { first_in_second: true };
        }
    }
    if let Some(p) = b.first().map(|t| t[0]).filter(|p| inside_box(*p, box_a)) {
        if encloses(a, p, size) {
            return Contact::Inside { first_in_second: false };
        }
    }
    Contact::Clear
}

fn bounds(tris: &[Triangle]) -> Option<(Vec3, Vec3)> {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for t in tris {
        for p in t {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    (lo.x <= hi.x).then_some((lo, hi))
}

fn inside_box(p: Vec3, (lo, hi): (Vec3, Vec3)) -> bool {
    p.cmpge(lo).all() && p.cmple(hi).all()
}

fn tri_box(t: &Triangle) -> (Vec3, Vec3) {
    (t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]))
}

fn boxes_meet(a: (Vec3, Vec3), b: (Vec3, Vec3)) -> bool {
    a.0.cmple(b.1).all() && b.0.cmple(a.1).all()
}

/// Whether any triangle of `a` crosses any of `b` within the region.
fn surfaces_cross(a: &[Triangle], b: &[Triangle], lo: Vec3, hi: Vec3, eps: f32) -> bool {
    let region = (lo, hi);
    let near_a: Vec<&Triangle> = a.iter().filter(|t| boxes_meet(tri_box(t), region)).collect();
    let near_b: Vec<&Triangle> = b.iter().filter(|t| boxes_meet(tri_box(t), region)).collect();
    if near_a.is_empty() || near_b.is_empty() {
        return false;
    }

    // The second solid's triangles, binned by the cells their boxes cover.
    let extent = (hi - lo).max(Vec3::splat(1e-9));
    let cell_of = |p: Vec3| {
        let c = ((p - lo) / extent * CELLS as f32).floor();
        [0, 1, 2].map(|i| (c[i].max(0.0) as usize).min(CELLS - 1))
    };
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); CELLS * CELLS * CELLS];
    let index = |c: [usize; 3]| (c[2] * CELLS + c[1]) * CELLS + c[0];
    for (i, t) in near_b.iter().enumerate() {
        let (tlo, thi) = tri_box(t);
        let (c0, c1) = (cell_of(tlo), cell_of(thi));
        for z in c0[2]..=c1[2] {
            for y in c0[1]..=c1[1] {
                for x in c0[0]..=c1[0] {
                    grid[index([x, y, z])].push(i as u32);
                }
            }
        }
    }

    let mut seen = vec![u32::MAX; near_b.len()];
    for (ia, ta) in near_a.iter().enumerate() {
        let (tlo, thi) = tri_box(ta);
        let (c0, c1) = (cell_of(tlo), cell_of(thi));
        for z in c0[2]..=c1[2] {
            for y in c0[1]..=c1[1] {
                for x in c0[0]..=c1[0] {
                    for &ib in &grid[index([x, y, z])] {
                        if seen[ib as usize] == ia as u32 {
                            continue;
                        }
                        seen[ib as usize] = ia as u32;
                        if triangles_cross(ta, near_b[ib as usize], eps) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Whether two triangles cross: an edge of either passes through the other's
/// interior. Two triangles that cross always do so along a segment whose ends
/// are where edges of one or the other pierce the other.
fn triangles_cross(a: &Triangle, b: &Triangle, eps: f32) -> bool {
    if !boxes_meet(tri_box(a), tri_box(b)) {
        return false;
    }
    let pierces = |t: &Triangle, s: &Triangle| {
        (0..3).any(|i| segment_pierces(s[i], s[(i + 1) % 3], t, eps))
    };
    pierces(a, b) || pierces(b, a)
}

/// Whether segment `p`–`q` passes through the interior of triangle `t`, not
/// merely touching it at an end or along an edge.
fn segment_pierces(p: Vec3, q: Vec3, t: &Triangle, eps: f32) -> bool {
    let d = q - p;
    let length = d.length();
    if length <= eps {
        return false;
    }
    let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
    let h = d.cross(e2);
    let det = e1.dot(h);
    // Parallel, which includes lying in the triangle's plane: resting on it.
    let scale = e1.length() * e2.length() * length;
    if det.abs() <= 1e-7 * scale {
        return false;
    }
    let inv = 1.0 / det;
    let s = p - t[0];
    let u = s.dot(h) * inv;
    let qv = s.cross(e1);
    let v = d.dot(qv) * inv;
    let at = e2.dot(qv) * inv;
    // Strictly inside, by a margin: a hit on an edge or a vertex is contact.
    let margin = eps / length;
    let edge = 1e-4;
    at > margin && at < 1.0 - margin && u > edge && v > edge && u + v < 1.0 - edge
}

/// Whether the closed surface `tris` encloses `p`: a ray from it crosses the
/// surface an odd number of times. A slanted direction, so a ray grazing an
/// edge of an axis-aligned box is not the usual case.
fn encloses(tris: &[Triangle], p: Vec3, size: f32) -> bool {
    let dir = Vec3::new(0.5773, 0.5775, 0.5771).normalize();
    let far = p + dir * size * 4.0;
    let mut crossings = 0;
    for t in tris {
        if segment_hits(p, far, t) {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

/// Whether a segment meets a triangle at all, edges included.
fn segment_hits(p: Vec3, q: Vec3, t: &Triangle) -> bool {
    let d = q - p;
    let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
    let h = d.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-12 {
        return false;
    }
    let inv = 1.0 / det;
    let s = p - t[0];
    let u = s.dot(h) * inv;
    let qv = s.cross(e1);
    let v = d.dot(qv) * inv;
    let at = e2.dot(qv) * inv;
    at > 0.0 && at <= 1.0 && u >= 0.0 && v >= 0.0 && u + v <= 1.0
}

/// A mesh's triangles in world space, carried through `to_world`.
pub fn world_triangles(mesh: &crate::MeshData, to_world: kreate_math::DAffine3) -> Vec<Triangle> {
    let world: Vec<Vec3> = mesh
        .positions()
        .iter()
        .map(|p| to_world.transform_point3(p.as_dvec3()).as_vec3())
        .collect();
    let mut out = Vec::new();
    for face in mesh.faces() {
        // A fan: right for the convex faces a modeller makes, and a crossing
        // test does not need the triangulation to be pretty.
        for i in 1..face.len().saturating_sub(1) {
            out.push([world[face[0] as usize], world[face[i] as usize], world[face[i + 1] as usize]]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;
    use kreate_math::{DAffine3, DVec3};

    fn cube_at(size: f32, at: [f64; 3]) -> Vec<Triangle> {
        world_triangles(&primitives::cube(size), DAffine3::from_translation(DVec3::from_array(at)))
    }

    #[test]
    fn two_cubes_pushed_into_each_other_cross() {
        assert_eq!(contact(&cube_at(2.0, [0.0; 3]), &cube_at(2.0, [1.5, 0.3, 0.2])), Contact::Crossing);
    }

    #[test]
    fn apart_is_clear() {
        assert_eq!(contact(&cube_at(2.0, [0.0; 3]), &cube_at(2.0, [3.0, 0.0, 0.0])), Contact::Clear);
    }

    #[test]
    fn one_resting_on_the_other_is_only_touching() {
        // A small cube standing on a big one, face on face.
        assert_eq!(contact(&cube_at(2.0, [0.0; 3]), &cube_at(1.0, [0.2, 0.1, 1.5])), Contact::Clear);
        // And side against side.
        assert_eq!(contact(&cube_at(2.0, [0.0; 3]), &cube_at(2.0, [2.0, 0.5, 0.0])), Contact::Clear);
    }

    #[test]
    fn one_inside_the_other_is_inside() {
        assert_eq!(
            contact(&cube_at(0.5, [0.1, 0.0, 0.0]), &cube_at(2.0, [0.0; 3])),
            Contact::Inside { first_in_second: true }
        );
        assert_eq!(
            contact(&cube_at(2.0, [0.0; 3]), &cube_at(0.5, [0.1, 0.0, 0.0])),
            Contact::Inside { first_in_second: false }
        );
    }

    #[test]
    fn boxes_that_overlap_without_the_shapes_meeting_are_clear() {
        // A thin ring around a post: their boxes overlap, the shapes do not.
        let ring = world_triangles(&primitives::torus(1.0, 0.1, 32, 8), DAffine3::IDENTITY);
        let post = world_triangles(&primitives::cylinder(0.3, 3.0, 16), DAffine3::IDENTITY);
        assert_eq!(contact(&ring, &post), Contact::Clear);
        let fat = world_triangles(&primitives::cylinder(1.2, 3.0, 16), DAffine3::IDENTITY);
        assert_eq!(contact(&ring, &fat), Contact::Inside { first_in_second: true });
        let crossing = world_triangles(&primitives::cylinder(0.95, 3.0, 32), DAffine3::IDENTITY);
        assert_eq!(contact(&ring, &crossing), Contact::Crossing);
    }
}
