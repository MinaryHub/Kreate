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
}
