//! The subdivision modifier.
//!
//! Splits every face into quads, `levels` times over.
//!
//! # This is simple subdivision, not Catmull-Clark
//!
//! The surface is refined but not *smoothed*: new vertices land on the edges and
//! face centres of the cage rather than being pulled toward a limit surface. That
//! is a real difference and it is stated rather than implied — a user reaching
//! for "subdivision surface" wants the smooth one, and finding out by looking is
//! worse than being told.
//!
//! Catmull-Clark is the plan's target for this slot and is a separate piece of
//! work: face, edge and vertex point rules, boundary rules and crease handling,
//! each of which has to be right or the surface pulls away from the cage in a way
//! that only shows on curved geometry. Simple subdivision is here now because the
//! *stack* is what this phase is for, and a modifier that refines topology
//! exercises every part of it.

use crate::bmesh::BMesh;
use crate::MeshData;

/// How many times to subdivide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubdivideParams {
    /// Each level splits every edge in half.
    pub levels: u32,
}

impl Default for SubdivideParams {
    fn default() -> Self {
        Self { levels: 1 }
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

    #[test]
    fn one_level_turns_each_face_into_quads() {
        let input = primitives::cube(2.0);
        let out = subdivide(&input, &SubdivideParams::default());
        out.assert_valid();
        assert_eq!(out.num_faces(), 24, "four per original face");
        for face in out.faces() {
            assert_eq!(face.len(), 4);
        }
    }

    #[test]
    fn levels_compound() {
        let input = primitives::cube(2.0);
        let out = subdivide(&input, &SubdivideParams { levels: 2 });
        out.assert_valid();
        assert_eq!(out.num_faces(), 96);
    }

    #[test]
    fn zero_levels_is_the_identity() {
        let input = primitives::cube(2.0);
        let out = subdivide(&input, &SubdivideParams { levels: 0 });
        assert_eq!(out.positions(), input.positions());
        assert_eq!(out.corner_verts(), input.corner_verts());
    }

    #[test]
    fn an_absurd_level_is_clamped() {
        // A viewport modifier is re-evaluated whenever its input changes, so an
        // unreasonable level is slow on every edit rather than once.
        let input = primitives::cube(2.0);
        let out = subdivide(&input, &SubdivideParams { levels: 100 });
        out.assert_valid();
        assert!(out.num_faces() <= 6 * 4u32.pow(MAX_LEVELS));
    }

    #[test]
    fn the_cage_is_not_smoothed() {
        // Stated rather than implied: this refines topology and leaves the shape
        // alone.
        let input = primitives::cube(2.0);
        let out = subdivide(&input, &SubdivideParams { levels: 2 });
        assert_eq!(
            out.bounds().max,
            input.bounds().max,
            "simple subdivision must not pull the surface inward"
        );
    }

    #[test]
    fn an_empty_mesh_subdivides_to_nothing() {
        let out = subdivide(&MeshData::empty(), &SubdivideParams::default());
        assert!(out.is_empty());
    }
}
