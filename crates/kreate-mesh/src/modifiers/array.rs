//! The array modifier.
//!
//! Repeats the mesh along an offset. The everyday use is anything regular — a row
//! of windows, a chain, a staircase — where editing the original updates every
//! copy.
//!
//! # The offset is relative by default
//!
//! An offset given as a fraction of the bounding box means the copies sit end to
//! end whatever the size of the mesh, so a user editing the original does not
//! have to retune the spacing every time. A constant offset is added on top, and
//! is what a user wants when the spacing is the thing being designed.

use kreate_math::Vec3;

use crate::MeshData;

/// How many copies, and how far apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArrayParams {
    /// Total copies including the original. One is the identity.
    pub count: u32,
    /// Offset as a fraction of the bounding box, per axis.
    pub relative: Vec3,
    /// Offset in metres, added to the relative one.
    pub constant: Vec3,
}

impl Default for ArrayParams {
    fn default() -> Self {
        Self {
            count: 2,
            relative: Vec3::X,
            constant: Vec3::ZERO,
        }
    }
}

/// The most copies a modifier will make.
///
/// Not a memory guarantee — a thousand copies of a million-vertex mesh is still
/// far too much — but a guard against the two ways an absurd count arrives: a
/// spinner dragged off the end of its range, and a script passing something
/// unchecked. Without it `count` comes straight from a `u32` and the modifier
/// tries to allocate for four billion copies.
pub const MAX_COUNT: u32 = 1000;

/// Repeats `input` along the offset.
pub fn array(input: &MeshData, params: &ArrayParams) -> MeshData {
    let copies = params.count.clamp(1, MAX_COUNT);
    if copies == 1 || input.is_empty() {
        return input.clone();
    }
    let size = input.bounds().size();
    let step = params.relative * size + params.constant;

    let source_verts = input.num_verts() as usize;
    let source_faces = input.num_faces() as usize;
    let source_corners = input.num_corners() as usize;

    let mut positions: Vec<Vec3> = Vec::with_capacity(source_verts * copies as usize);
    let mut point_order: Vec<u32> = Vec::with_capacity(positions.capacity());
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(source_faces * copies as usize);
    let mut face_order: Vec<u32> = Vec::with_capacity(faces.capacity());
    let mut corner_order: Vec<u32> = Vec::with_capacity(source_corners * copies as usize);

    for copy in 0..copies {
        let shift = step * copy as f32;
        let base = (copy as usize * source_verts) as u32;
        for (i, p) in input.positions().iter().enumerate() {
            positions.push(*p + shift);
            point_order.push(i as u32);
        }
        for (index, face) in input.faces().enumerate() {
            faces.push(face.iter().map(|v| v + base).collect());
            face_order.push(index as u32);
            corner_order.extend(input.face_corner_range(index as u32).map(|c| c as u32));
        }
    }

    let mut out = MeshData::from_faces(positions, &faces);
    super::carry_attributes(&mut out, input, &point_order, &corner_order, &face_order);
    // `point_order` is already output-vertex to source-vertex, so every copy's
    // edges pair against the same source edges — which is what a repeated crease
    // should be. A source-to-output map would only reach the first copy.
    super::gather_edges_by_pair(&mut out, input, &point_order);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::{names, AttrData, Domain};
    use crate::primitives;
    use std::sync::Arc;

    #[test]
    fn two_copies_sit_end_to_end() {
        let input = primitives::cube(2.0);
        let out = array(&input, &ArrayParams::default());
        out.assert_valid();

        assert_eq!(out.num_verts(), 16);
        assert_eq!(out.num_faces(), 12);
        let bounds = out.bounds();
        assert!(
            (bounds.max.x - 3.0).abs() < 1e-5,
            "the second copy should start where the first ends: {bounds:?}"
        );
        assert!((bounds.min.x + 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_relative_offset_follows_the_mesh_size() {
        // The reason it is relative by default: a user editing the original must
        // not have to retune the spacing every time.
        let small = array(&primitives::cube(1.0), &ArrayParams::default());
        let large = array(&primitives::cube(4.0), &ArrayParams::default());
        assert!((small.bounds().size().x - 2.0).abs() < 1e-5);
        assert!((large.bounds().size().x - 8.0).abs() < 1e-5);
    }

    #[test]
    fn a_constant_offset_adds_a_gap() {
        let out = array(
            &primitives::cube(2.0),
            &ArrayParams {
                count: 3,
                relative: Vec3::X,
                constant: Vec3::X * 0.5,
            },
        );
        out.assert_valid();
        assert_eq!(out.num_verts(), 24);
        assert!((out.bounds().size().x - (2.0 + 2.5 * 2.0)).abs() < 1e-4);
    }

    #[test]
    fn one_copy_is_the_identity() {
        let input = primitives::cube(2.0);
        let out = array(
            &input,
            &ArrayParams {
                count: 1,
                ..Default::default()
            },
        );
        assert_eq!(out.positions(), input.positions());
        assert_eq!(out.corner_verts(), input.corner_verts());
    }

    #[test]
    fn zero_copies_is_treated_as_one() {
        // A parameter of zero arrives from a script and from a spinner dragged
        // past its minimum. An empty mesh looks like the tool broke.
        let input = primitives::cube(2.0);
        let out = array(
            &input,
            &ArrayParams {
                count: 0,
                ..Default::default()
            },
        );
        assert_eq!(out.num_verts(), input.num_verts());
    }

    #[test]
    fn every_copy_keeps_the_attributes() {
        let mut input = primitives::cube(2.0);
        let corners = input.num_corners() as usize;
        input.attributes_mut().insert(
            "uv",
            Domain::Corner,
            AttrData::F32x2(Arc::new(
                (0..corners)
                    .map(|i| kreate_math::Vec2::new(i as f32, 0.0))
                    .collect(),
            )),
        );
        let edges = input.num_edges() as usize;
        input.attributes_mut().insert(
            names::SHARP_EDGE,
            Domain::Edge,
            AttrData::Bool(Arc::new((0..edges).map(|i| i % 3 == 0).collect())),
        );

        let out = array(
            &input,
            &ArrayParams {
                count: 3,
                ..Default::default()
            },
        );
        out.assert_valid();

        let uvs = out.attributes().get_f32x2("uv").unwrap();
        assert_eq!(uvs.len(), out.num_corners() as usize);
        for copy in 0..3 {
            for i in 0..corners {
                assert_eq!(
                    uvs[copy * corners + i].x,
                    i as f32,
                    "copy {copy} corner {i} lost its UV"
                );
            }
        }

        let sharp = out.attributes().get_bool(names::SHARP_EDGE).unwrap();
        let marked = input
            .attributes()
            .get_bool(names::SHARP_EDGE)
            .unwrap()
            .iter()
            .filter(|x| **x)
            .count();
        assert_eq!(
            sharp.iter().filter(|x| **x).count(),
            marked * 3,
            "each copy should carry the same creases"
        );
    }

    #[test]
    fn a_zero_offset_stacks_the_copies_without_corrupting_anything() {
        // Unusual, and not something to be clever about: the copies land on top
        // of each other. What must not happen is an invalid mesh.
        let out = array(
            &primitives::cube(2.0),
            &ArrayParams {
                count: 3,
                relative: Vec3::ZERO,
                constant: Vec3::ZERO,
            },
        );
        out.assert_valid();
        assert_eq!(out.num_verts(), 24);
        assert_eq!(out.validate(), Vec::new());
    }

    #[test]
    fn an_absurd_count_is_clamped() {
        // `count` is a `u32`, so a script or a dragged spinner can ask for four
        // billion copies and the modifier would try to allocate for them.
        let out = array(
            &primitives::cube(2.0),
            &ArrayParams {
                count: u32::MAX,
                ..Default::default()
            },
        );
        assert_eq!(out.num_verts(), 8 * MAX_COUNT);
    }

    #[test]
    fn an_empty_mesh_arrays_to_nothing() {
        let out = array(&MeshData::empty(), &ArrayParams::default());
        assert!(out.is_empty());
    }
}
