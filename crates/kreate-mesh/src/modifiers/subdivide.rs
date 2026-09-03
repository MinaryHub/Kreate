//! The subdivision modifier.
//!
//! Splits every face into quads, `levels` times over — smoothly by default.
//!
//! # Two kinds, and the default is the smooth one
//!
//! **Catmull-Clark** pulls the surface toward a limit surface: this is what
//! "subdivision surface" means and what a user reaching for this modifier
//! expects. See [`crate::modifiers::catmull_clark`] for the rules.
//!
//! **Simple** subdivision refines the topology and leaves the shape exactly
//! where it was. That is not a lesser version of the same thing — it is a
//! different tool, for adding topology to a shape that is already right, and a
//! modeller wants both.
//!
//! The default is smooth because being handed refined-but-unchanged geometry
//! when you asked for a subdivision surface is the more surprising of the two.

use crate::bmesh::BMesh;
use crate::MeshData;

/// How many times to subdivide, and how.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubdivideParams {
    /// Each level splits every edge in half.
    pub levels: u32,
    /// Smooth the surface toward its limit rather than only refining topology.
    pub smooth: bool,
}

impl Default for SubdivideParams {
    fn default() -> Self {
        Self {
            levels: 1,
            smooth: true,
        }
    }
}

/// The most levels a modifier will apply.
///
/// A viewport modifier is re-evaluated whenever its input changes, so an
/// unreasonable level is not merely slow — it is slow on every edit.
pub const MAX_LEVELS: u32 = 6;

/// Subdivides `input`.
pub fn subdivide(input: &MeshData, params: &SubdivideParams) -> MeshData {
    let levels = params.levels.min(MAX_LEVELS);
    if levels == 0 || input.is_empty() {
        return input.clone();
    }
    if params.smooth {
        let mut mesh = input.clone();
        for _ in 0..levels {
            mesh = crate::modifiers::catmull_clark::subdivide_once(&mesh);
        }
        return mesh;
    }

    // Simple subdivision goes through the edit operator, which already knows how
    // to split an n-gon and keeps the operator and the modifier producing the
    // same topology from the same input.
    let mut mesh = BMesh::from_mesh_data(input);
    for _ in 0..levels {
        let faces: Vec<_> = mesh.face_ids().collect();
        if crate::edit::subdivide_faces(&mut mesh, &faces).is_err() {
            break;
        }
    }
    mesh.to_mesh_data()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn simple(levels: u32) -> SubdivideParams {
        SubdivideParams {
            levels,
            smooth: false,
        }
    }

    fn smooth(levels: u32) -> SubdivideParams {
        SubdivideParams {
            levels,
            smooth: true,
        }
    }

    #[test]
    fn one_level_turns_each_face_into_quads() {
        for params in [simple(1), smooth(1)] {
            let out = subdivide(&primitives::cube(2.0), &params);
            out.assert_valid();
            assert_eq!(out.num_faces(), 24, "four per original face: {params:?}");
            for face in out.faces() {
                assert_eq!(face.len(), 4);
            }
        }
    }

    #[test]
    fn levels_compound() {
        for params in [simple(2), smooth(2)] {
            let out = subdivide(&primitives::cube(2.0), &params);
            out.assert_valid();
            assert_eq!(out.num_faces(), 96, "{params:?}");
        }
    }

    #[test]
    fn zero_levels_is_the_identity() {
        let input = primitives::cube(2.0);
        for params in [simple(0), smooth(0)] {
            let out = subdivide(&input, &params);
            assert_eq!(out.positions(), input.positions(), "{params:?}");
            assert_eq!(out.corner_verts(), input.corner_verts());
        }
    }

    #[test]
    fn an_absurd_level_is_clamped() {
        // A viewport modifier is re-evaluated whenever its input changes, so an
        // unreasonable level is slow on every edit rather than once.
        for params in [simple(100), smooth(100)] {
            let out = subdivide(&primitives::cube(2.0), &params);
            out.assert_valid();
            assert!(out.num_faces() <= 6 * 4u32.pow(MAX_LEVELS), "{params:?}");
        }
    }

    #[test]
    fn the_default_smooths() {
        // What a user reaching for "subdivision surface" gets. Being handed
        // refined-but-unchanged geometry is the more surprising default of the
        // two, so it is not the one on offer.
        assert!(SubdivideParams::default().smooth);

        let input = primitives::cube(2.0);
        let out = subdivide(&input, &SubdivideParams::default());
        let moved = input
            .positions()
            .iter()
            .zip(out.positions())
            .filter(|(a, b)| (**a - **b).length() > 0.1)
            .count();
        assert_eq!(
            moved,
            input.num_verts() as usize,
            "every original vertex should have been pulled inward"
        );
    }

    #[test]
    fn simple_subdivision_leaves_the_shape_alone() {
        // The other tool, and why it is kept: adding topology to a shape that is
        // already right.
        let input = primitives::cube(2.0);
        let out = subdivide(&input, &simple(2));
        for (v, before) in input.positions().iter().enumerate() {
            assert_eq!(
                out.positions()[v],
                *before,
                "simple subdivision moved vertex {v}"
            );
        }
    }

    #[test]
    fn an_empty_mesh_subdivides_to_nothing() {
        for params in [simple(1), smooth(1)] {
            let out = subdivide(&MeshData::empty(), &params);
            assert!(out.is_empty(), "{params:?}");
        }
    }
}
