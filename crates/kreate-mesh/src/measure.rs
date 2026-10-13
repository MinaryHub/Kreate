//! What a solid weighs: its volume, its surface, where its middle of mass is.
//!
//! The volume and the centroid come from the divergence theorem over the
//! surface — each triangle with the origin makes a signed tetrahedron, and
//! the signed ones sum to the solid. That is exact for a closed surface wound
//! outward, and meaningless for an open one, so the answer says whether the
//! surface is closed rather than leaving the number to look right.
//!
//! In `f64`, in whatever space the points are given: the caller hands world
//! positions, so a scaled object weighs what it looks like.

use kreate_math::DVec3;

use crate::MeshData;

/// A solid's measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    /// Cubic units; negative when the surface is wound inward.
    pub volume: f64,
    /// Square units: every face, whether or not the surface is closed.
    pub area: f64,
    /// The middle of the volume — of the mass, for a uniform material.
    pub centroid: DVec3,
    /// Every edge has exactly two faces: the volume means something.
    pub closed: bool,
}

/// Measures `mesh` with its points placed by `to_space`.
pub fn mass_properties(mesh: &MeshData, to_space: impl Fn(DVec3) -> DVec3) -> MassProperties {
    let points: Vec<DVec3> = mesh
        .positions()
        .iter()
        .map(|p| to_space(p.as_dvec3()))
        .collect();
    let mut volume = 0.0;
    let mut moment = DVec3::ZERO;
    let mut area = 0.0;
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        if ring.len() < 3 {
            continue;
        }
        let a = points[ring[0] as usize];
        for i in 1..ring.len() - 1 {
            let (b, c) = (points[ring[i] as usize], points[ring[i + 1] as usize]);
            // Six times the signed tetrahedron (origin, a, b, c).
            let six = a.dot(b.cross(c));
            volume += six;
            moment += (a + b + c) * six;
            area += (b - a).cross(c - a).length() * 0.5;
        }
    }
    let volume = volume / 6.0;
    // Each tetrahedron's centroid is (a + b + c) / 4 (the origin the fourth
    // corner), weighted by its volume six / 6.
    let centroid = if volume.abs() > 1e-300 {
        moment / (24.0 * volume)
    } else {
        DVec3::ZERO
    };
    MassProperties {
        volume,
        area,
        centroid,
        closed: is_closed(mesh),
    }
}

/// Every edge borders exactly two faces.
pub fn is_closed(mesh: &MeshData) -> bool {
    let mut uses = vec![0u32; mesh.num_edges() as usize];
    let edges = mesh.corner_edges();
    for e in edges.iter() {
        if let Some(u) = uses.get_mut(*e as usize) {
            *u += 1;
        }
    }
    mesh.num_faces() > 0 && uses.iter().all(|u| *u == 2)
}

/// Whether `point` is inside the closed surface `mesh`, both in the mesh's
/// own space: rays out from it along three slanted directions, each counting
/// the faces it crosses — odd is inside — and the three voting, so a ray that
/// grazes an edge or a corner is outvoted rather than trusted.
pub fn contains(mesh: &MeshData, point: DVec3) -> bool {
    let tri = mesh.triangulate();
    let positions = mesh.positions();
    let (tris, _) = tri.indices.as_chunks::<3>();
    let corners: Vec<[DVec3; 3]> = tris
        .iter()
        .map(|t| t.map(|v| positions[v as usize].as_dvec3()))
        .collect();
    let directions = [
        DVec3::new(1.0, 0.123_456, 0.065_432).normalize(),
        DVec3::new(-0.087_654, 1.0, 0.154_321).normalize(),
        DVec3::new(0.111_111, -0.098_765, 1.0).normalize(),
    ];
    let inside = directions
        .iter()
        .filter(|d| {
            corners
                .iter()
                .filter(|[a, b, c]| ray_hits(point, **d, *a, *b, *c))
                .count()
                % 2
                == 1
        })
        .count();
    inside >= 2
}

/// Möller–Trumbore: whether the ray from `o` along `d` crosses triangle
/// `a b c` ahead of `o`.
fn ray_hits(o: DVec3, d: DVec3, a: DVec3, b: DVec3, c: DVec3) -> bool {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-14 {
        return false;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    e2.dot(q) * inv > 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_cube_has_its_volume_area_and_middle() {
        let m = primitives::cube(2.0);
        let p = mass_properties(&m, |p| p + DVec3::new(5.0, 0.0, -1.0));
        assert!((p.volume - 8.0).abs() < 1e-9, "{}", p.volume);
        assert!((p.area - 24.0).abs() < 1e-9);
        assert!((p.centroid - DVec3::new(5.0, 0.0, -1.0)).length() < 1e-9);
        assert!(p.closed);
    }

    #[test]
    fn a_scaled_space_scales_the_answer() {
        let m = primitives::cube(1.0);
        let p = mass_properties(&m, |p| p * DVec3::new(0.01, 0.02, 0.03));
        assert!((p.volume - 6e-6).abs() < 1e-15);
    }

    #[test]
    fn a_sphere_comes_close_to_its_formula() {
        let m = primitives::uv_sphere(1.0, 128, 64);
        let p = mass_properties(&m, |p| p);
        let exact = 4.0 / 3.0 * std::f64::consts::PI;
        assert!((p.volume - exact).abs() / exact < 0.005, "{}", p.volume);
        assert!((p.area - 4.0 * std::f64::consts::PI).abs() < 0.05);
        assert!(p.centroid.length() < 1e-6);
        assert!(p.closed);
    }

    #[test]
    fn a_plane_is_not_closed() {
        let p = mass_properties(&primitives::plane(2.0), |p| p);
        assert!(!p.closed);
        assert!((p.area - 4.0).abs() < 1e-9);
    }

    #[test]
    fn a_point_is_inside_a_cube_or_not() {
        let cube = crate::primitives::cube(2.0);
        assert!(super::contains(&cube, DVec3::ZERO));
        assert!(super::contains(&cube, DVec3::new(0.9, -0.9, 0.5)));
        assert!(!super::contains(&cube, DVec3::new(1.5, 0.0, 0.0)));
        assert!(!super::contains(&cube, DVec3::new(0.0, 0.0, -3.0)));
    }
}
