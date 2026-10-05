//! The bevel modifier: sharp edges chamfered, without touching the cage.
//!
//! The same chamfer as the edit-mode bevel (`Ctrl+B`), applied to every edge
//! whose faces meet at more than `angle` — Blender's default Angle limit, which
//! is what makes a hard-surface model catch the light at its corners while the
//! mesh the user edits stays simple.

use crate::bmesh::BMesh;
use crate::MeshData;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BevelParams {
    /// How wide the chamfer is.
    pub width: f32,
    /// Edges whose faces turn by more than this, in radians, are bevelled.
    pub angle: f32,
    /// Strips across each edge: one is a flat chamfer, more round it.
    pub segments: u32,
}

impl Default for BevelParams {
    fn default() -> Self {
        Self {
            width: 0.1,
            angle: 30f32.to_radians(),
            segments: 1,
        }
    }
}

/// Chamfers `input`'s sharp edges.
pub fn bevel(input: &MeshData, params: &BevelParams) -> MeshData {
    if input.is_empty() || params.width.is_nan() || params.width <= 0.0 {
        return input.clone();
    }
    let mut bm = BMesh::from_mesh_data(input);
    let cos_limit = params.angle.clamp(0.0, std::f32::consts::PI).cos();
    let edges: Vec<_> = bm
        .edge_ids()
        .filter(|e| {
            let faces = bm.edge_faces(*e);
            let [a, b] = faces[..] else { return false };
            let (na, nb) = (
                bm.face_normal(a).normalize_or_zero(),
                bm.face_normal(b).normalize_or_zero(),
            );
            na.dot(nb) < cos_limit
        })
        .collect();
    if edges.is_empty() {
        return input.clone();
    }
    match crate::edit::bevel_edges_segments(&mut bm, &edges, params.width, params.segments) {
        Ok(_) => bm.to_mesh_data(),
        // A bevel that cannot be built draws the cage rather than nothing.
        Err(_) => input.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn a_cube_gets_its_twelve_edges_chamfered_and_a_plane_nothing() {
        let cube = primitives::cube(2.0);
        let out = bevel(&cube, &BevelParams::default());
        assert!(out.num_faces() > cube.num_faces(), "{}", out.num_faces());
        // Still inside the cube's bounds: a chamfer cuts, it does not grow.
        assert!(out.bounds().max.x <= 1.0 + 1e-5);
        let plane = primitives::plane(2.0);
        assert_eq!(bevel(&plane, &BevelParams::default()).num_faces(), 1);
    }
}
