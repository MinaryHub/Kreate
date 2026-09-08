//! Triangulation of n-gons, for the GPU and for triangle-only file formats.
//!
//! `MeshData` stores faces as authored. Nothing in the editable representation is
//! ever triangulated — a quad is a quad, and the user's topology is preserved
//! exactly. Triangles are produced here, at the boundary, and the mapping back to
//! the originating face is carried alongside so that picking, per-face attributes,
//! and lossless n-gon round-tripping all keep working.

use glam::{Vec2, Vec3};

use crate::MeshData;

/// Triangulated geometry plus the mapping back to source faces.
#[derive(Clone, Debug, Default)]
pub struct Triangulation {
    /// Vertex indices, three per triangle.
    pub indices: Vec<u32>,
    /// Source face index, one per triangle. This is what makes GPU id-buffer
    /// picking able to report a *face* rather than a triangle, and what lets an
    /// exporter reconstruct n-gons.
    pub tri_to_face: Vec<u32>,
}

impl Triangulation {
    pub fn num_triangles(&self) -> usize {
        self.tri_to_face.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tri_to_face.is_empty()
    }
}

/// The same triangulation, in *corner* indices rather than vertex indices.
///
/// A corner says both which vertex a triangle vertex is and which face it came
/// through; a vertex index says only the first. Anything that needs a per-corner
/// attribute — a UV, a split normal, a vertex colour — needs the second half,
/// and recovering it from a vertex index means searching the face for it.
///
/// That search was real: the renderer's extraction did it for every triangle
/// corner, six million times on a million faces, to find out something
/// [`MeshData::triangulate`] had discarded one line earlier.
#[derive(Clone, Debug, Default)]
pub struct CornerTriangulation {
    /// Corner indices, three per triangle.
    pub corners: Vec<u32>,
    /// Source face per triangle.
    pub tri_to_face: Vec<u32>,
}

impl CornerTriangulation {
    pub fn num_triangles(&self) -> usize {
        self.tri_to_face.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tri_to_face.is_empty()
    }
}

/// Best-fit normal of a polygon via Newell's method.
///
/// Correct for non-planar n-gons, where any single cross product would depend on
/// which corner happened to be sampled.
pub fn newell_normal(points: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        n += (a - b).cross(a + b);
    }
    n.normalize_or_zero()
}

/// Builds an orthonormal basis whose Z is `n`, then projects `points` into it.
///
/// Ear clipping is a 2D algorithm; this is how a face in space gets there. The
/// tangent is chosen from whichever cardinal axis is least aligned with `n`, so
/// the basis never degenerates.
fn project_to_plane(points: &[Vec3], n: Vec3) -> Vec<Vec2> {
    let helper = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let t = helper.cross(n).normalize_or_zero();
    let b = n.cross(t);
    points
        .iter()
        .map(|p| Vec2::new(p.dot(t), p.dot(b)))
        .collect()
}

#[inline]
fn cross2(o: Vec2, a: Vec2, b: Vec2) -> f32 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Point-in-triangle test using the same orientation predicate as the ear test,
/// so a point exactly on an edge is treated consistently by both.
fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = cross2(a, b, p);
    let d2 = cross2(b, c, p);
    let d3 = cross2(c, a, p);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Triangulates one polygon, appending `(n - 2)` triangles' worth of *local*
/// indices (into `points`) to `out`.
///
/// Uses a fan for triangles and for convex quads, and ear clipping otherwise.
/// Ear clipping is O(n²) but handles concave polygons, which a fan does not — a
/// fan across an L-shaped n-gon produces triangles outside the polygon, and that
/// is a visible bug on the very first non-convex face a user draws.
///
/// Returns the number of triangles emitted. A degenerate polygon (fewer than 3
/// points, or zero area) emits nothing rather than emitting garbage.
pub fn triangulate_polygon(points: &[Vec3], out: &mut Vec<[u32; 3]>) -> usize {
    let n = points.len();
    if n < 3 {
        return 0;
    }
    if n == 3 {
        out.push([0, 1, 2]);
        return 1;
    }

    let normal = newell_normal(points);
    if normal == Vec3::ZERO {
        return 0; // fully degenerate face
    }
    let p2 = project_to_plane(points, normal);

    // Signed area tells us the winding in the projected basis; ear clipping
    // below assumes counter-clockwise, so reverse if needed.
    let area2: f32 = (0..n)
        .map(|i| {
            let a = p2[i];
            let b = p2[(i + 1) % n];
            a.x * b.y - b.x * a.y
        })
        .sum();

    let mut idx: Vec<u32> = (0..n as u32).collect();
    if area2 < 0.0 {
        idx.reverse();
    }

    let start = out.len();
    // Guard against pathological input: each successful clip removes a vertex,
    // so more iterations than that means no ear was found and we must bail
    // rather than spin. Falling back to a fan is wrong for concave polygons but
    // is strictly better than hanging the application.
    let mut guard = n * n;
    while idx.len() > 3 && guard > 0 {
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let (ia, ib, ic) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (a, b, c) = (p2[ia as usize], p2[ib as usize], p2[ic as usize]);
            if cross2(a, b, c) <= 0.0 {
                continue; // reflex corner, not an ear
            }
            // An ear must contain no other vertex of the polygon.
            let contains_other = idx.iter().any(|&other| {
                other != ia
                    && other != ib
                    && other != ic
                    && point_in_triangle(p2[other as usize], a, b, c)
            });
            if contains_other {
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
        guard -= 1;
    }
    if idx.len() >= 3 {
        for i in 1..idx.len() - 1 {
            out.push([idx[0], idx[i], idx[i + 1]]);
        }
    }
    out.len() - start
}

impl MeshData {
    /// Triangulates every face.
    ///
    /// Cached by the caller: this is invalidated by topology changes, **not** by
    /// position changes. A vertex drag can technically flip the valid
    /// triangulation of a concave n-gon, but the artifact during a drag is
    /// negligible; retriangulating affected faces once on mouse-up is correct and
    /// costs nothing per frame.
    pub fn triangulate(&self) -> Triangulation {
        let corners = self.triangulate_corners();
        let corner_verts = self.corner_verts();
        Triangulation {
            indices: corners
                .corners
                .iter()
                .map(|c| corner_verts[*c as usize])
                .collect(),
            tri_to_face: corners.tri_to_face,
        }
    }

    /// Triangulates every face, in corner indices.
    ///
    /// The primitive of the two: [`Self::triangulate`] is this plus one pass
    /// through `corner_verts`. A caller that wants per-corner attributes should
    /// use this and skip the round trip, because recovering a corner from a
    /// vertex index means searching the face for it.
    ///
    /// Cached by the caller: this is invalidated by topology changes, **not** by
    /// position changes. A vertex drag can technically flip the valid
    /// triangulation of a concave n-gon, but the artifact during a drag is
    /// negligible; retriangulating affected faces once on mouse-up is correct and
    /// costs nothing per frame.
    pub fn triangulate_corners(&self) -> CornerTriangulation {
        let pos = self.positions();
        let mut tri = CornerTriangulation::default();
        let mut local: Vec<[u32; 3]> = Vec::new();
        let mut points: Vec<Vec3> = Vec::new();

        for f in 0..self.num_faces() {
            let verts = self.face_verts(f);
            if verts.len() < 3 {
                continue;
            }
            local.clear();
            points.clear();
            points.extend(verts.iter().map(|&v| pos[v as usize]));
            let count = triangulate_polygon(&points, &mut local);
            // Local indices are into `verts`, which is the face's corner range:
            // corner `start + i` is the vertex `verts[i]`.
            let start = self.face_corner_range(f).start as u32;
            for t in &local {
                tri.corners
                    .extend_from_slice(&[start + t[0], start + t[1], start + t[2]]);
            }
            tri.tri_to_face.extend(std::iter::repeat_n(f, count));
        }
        tri
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshData;

    /// A mesh whose faces have different sizes and whose vertices are shared,
    /// so a corner index and a vertex index cannot be confused for each other.
    /// On a mesh of one triangle they coincide, and the test would prove
    /// nothing.
    fn mixed() -> MeshData {
        let p = |x: f32, y: f32| Vec3::new(x, y, 0.0);
        MeshData::from_faces(
            vec![
                p(0.0, 0.0),
                p(1.0, 0.0),
                p(2.0, 0.0),
                p(2.0, 1.0),
                p(1.0, 1.0),
                p(0.0, 1.0),
                p(3.0, 0.5),
            ],
            &[
                vec![0, 1, 4, 5],
                vec![1, 2, 3, 4],
                vec![2, 6, 3],
                vec![0, 1, 2, 6, 3, 4, 5],
            ],
        )
    }

    #[test]
    fn the_corner_triangulation_names_the_same_vertices() {
        // The two have to agree, because `triangulate` is now built on the
        // other one and the renderer uses the corner form directly.
        let mesh = mixed();
        let verts = mesh.triangulate();
        let corners = mesh.triangulate_corners();

        assert_eq!(verts.tri_to_face, corners.tri_to_face);
        assert_eq!(verts.indices.len(), corners.corners.len());
        let corner_verts = mesh.corner_verts();
        for (i, (&v, &c)) in verts.indices.iter().zip(&corners.corners).enumerate() {
            assert_eq!(
                corner_verts[c as usize], v,
                "triangle vertex {i}: corner {c} is vertex {} not {v}",
                corner_verts[c as usize]
            );
        }
    }

    #[test]
    fn a_reported_corner_belongs_to_its_own_triangle_s_face() {
        // The property the renderer depends on and a vertex index cannot
        // supply: the corner identifies *which face* the triangle vertex came
        // through, which is what makes a per-corner attribute — a split normal,
        // a UV — readable without searching the face for the vertex.
        //
        // A shared vertex is the case that distinguishes this from "some corner
        // with the right vertex", so `mixed` shares every vertex between at
        // least two faces.
        let mesh = mixed();
        let corners = mesh.triangulate_corners();
        for (t, tri) in corners.corners.as_chunks::<3>().0.iter().enumerate() {
            let face = corners.tri_to_face[t];
            let range = mesh.face_corner_range(face);
            for &c in tri {
                assert!(
                    range.contains(&(c as usize)),
                    "triangle {t} of face {face} names corner {c}, outside {range:?}"
                );
            }
        }
    }

    fn area_of(points: &[Vec3], tris: &[[u32; 3]]) -> f32 {
        tris.iter()
            .map(|t| {
                let (a, b, c) = (
                    points[t[0] as usize],
                    points[t[1] as usize],
                    points[t[2] as usize],
                );
                (b - a).cross(c - a).length() * 0.5
            })
            .sum()
    }

    #[test]
    fn triangle_is_passed_through() {
        let pts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let mut out = Vec::new();
        assert_eq!(triangulate_polygon(&pts, &mut out), 1);
        assert_eq!(out, [[0, 1, 2]]);
    }

    #[test]
    fn quad_becomes_two_triangles_covering_it() {
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut out = Vec::new();
        assert_eq!(triangulate_polygon(&pts, &mut out), 2);
        assert!((area_of(&pts, &out) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn concave_polygon_is_covered_exactly() {
        // An L shape. A naive fan would emit triangles outside the polygon and
        // overshoot the true area of 3.
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let mut out = Vec::new();
        assert_eq!(triangulate_polygon(&pts, &mut out), 4);
        assert!(
            (area_of(&pts, &out) - 3.0).abs() < 1e-4,
            "expected area 3, got {}",
            area_of(&pts, &out)
        );
    }

    #[test]
    fn clockwise_polygon_is_handled() {
        let ccw = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let cw: Vec<Vec3> = ccw.iter().rev().copied().collect();
        let mut out = Vec::new();
        assert_eq!(triangulate_polygon(&cw, &mut out), 2);
        assert!((area_of(&cw, &out) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn polygon_off_the_xy_plane_is_handled() {
        // The projection basis must not depend on the polygon lying on XY.
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let mut out = Vec::new();
        assert_eq!(triangulate_polygon(&pts, &mut out), 2);
        assert!((area_of(&pts, &out) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn degenerate_polygons_emit_nothing() {
        let mut out = Vec::new();
        assert_eq!(triangulate_polygon(&[Vec3::ZERO, Vec3::X], &mut out), 0);
        // Collinear points have no plane and therefore no triangulation.
        let line = vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0, Vec3::X * 3.0];
        assert_eq!(triangulate_polygon(&line, &mut out), 0);
        assert!(out.is_empty());
    }

    #[test]
    fn mesh_triangulation_maps_back_to_faces() {
        let m = MeshData::from_faces(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(2.0, 0.5, 0.0),
            ],
            &[vec![0, 1, 2, 3], vec![1, 4, 2]],
        );
        let t = m.triangulate();
        assert_eq!(t.num_triangles(), 3); // quad -> 2, tri -> 1
        assert_eq!(t.indices.len(), 9);
        assert_eq!(t.tri_to_face, vec![0, 0, 1]);
        assert!(t.indices.iter().all(|&i| i < m.num_verts()));
    }

    #[test]
    fn triangle_count_is_n_minus_2_for_convex_ngons() {
        for n in 3..12usize {
            let pts: Vec<Vec3> = (0..n)
                .map(|i| {
                    let a = i as f32 / n as f32 * std::f32::consts::TAU;
                    Vec3::new(a.cos(), a.sin(), 0.0)
                })
                .collect();
            let mut out = Vec::new();
            assert_eq!(triangulate_polygon(&pts, &mut out), n - 2, "n = {n}");
        }
    }
}
