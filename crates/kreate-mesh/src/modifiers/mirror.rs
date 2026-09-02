//! The mirror modifier.
//!
//! Reflects the mesh across one or more axis planes and welds what lands on the
//! plane. The everyday use is symmetrical modelling: build half a character,
//! mirror the other half, and never edit it directly.
//!
//! # Reflecting reverses the winding
//!
//! A reflection flips handedness, so a mirrored face copied corner-for-corner
//! faces *inward*. Reversing the ring is not a detail — a mirror that forgets it
//! produces a model whose left half is lit from inside, and the report that comes
//! back is "the shading is broken on one side".
//!
//! # Welding is a map, not a search
//!
//! A vertex within the threshold of the mirror plane maps to *itself* rather than
//! to a copy. That makes the seam exact and needs no spatial query: the plane is
//! at the origin along the chosen axis, so "on the plane" is one comparison. A
//! distance-based weld after the fact would also join vertices that happen to be
//! near each other for unrelated reasons.

use kreate_math::Vec3;

use crate::MeshData;

/// Which planes to reflect across, and how close counts as on the plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirrorParams {
    /// Reflect across the YZ plane, i.e. negate X.
    pub x: bool,
    pub y: bool,
    pub z: bool,
    /// A vertex closer than this to a mirror plane is welded to its reflection.
    pub merge_threshold: f32,
}

impl Default for MirrorParams {
    fn default() -> Self {
        Self {
            x: true,
            y: false,
            z: false,
            merge_threshold: 1e-4,
        }
    }
}

impl MirrorParams {
    /// The axes to reflect across, in a fixed order.
    fn axes(&self) -> Vec<usize> {
        let mut out = Vec::new();
        if self.x {
            out.push(0);
        }
        if self.y {
            out.push(1);
        }
        if self.z {
            out.push(2);
        }
        out
    }
}

/// Reflects `input` across the requested planes.
///
/// Each axis is applied in turn, so mirroring across X and then Y gives four
/// quadrants — the same result as Blender, and the reason the axes are applied
/// sequentially rather than as one combined reflection.
pub fn mirror(input: &MeshData, params: &MirrorParams) -> MeshData {
    let mut current = input.clone();
    for axis in params.axes() {
        current = mirror_one(&current, axis, params.merge_threshold.max(0.0));
    }
    current
}

fn mirror_one(input: &MeshData, axis: usize, threshold: f32) -> MeshData {
    let positions = input.positions();
    let count = positions.len();

    // Where each source vertex's reflection lives. A vertex on the plane is its
    // own reflection, which is what makes the seam exact.
    let mut reflection: Vec<u32> = Vec::with_capacity(count);
    let mut new_positions: Vec<Vec3> = positions.to_vec();
    // The source vertex each output vertex came from, for the point layers.
    let mut point_order: Vec<u32> = (0..count as u32).collect();
    for (i, p) in positions.iter().enumerate() {
        if p[axis].abs() <= threshold {
            reflection.push(i as u32);
            continue;
        }
        let mut mirrored = *p;
        mirrored[axis] = -mirrored[axis];
        reflection.push(new_positions.len() as u32);
        new_positions.push(mirrored);
        point_order.push(i as u32);
    }

    let mut faces: Vec<Vec<u32>> = input.faces().map(|f| f.to_vec()).collect();
    let mut face_order: Vec<u32> = (0..input.num_faces()).collect();
    let mut corner_order: Vec<u32> = (0..input.num_corners()).collect();

    for (index, face) in input.faces().enumerate() {
        let mut ring: Vec<u32> = face.iter().map(|v| reflection[*v as usize]).collect();
        let mut corners: Vec<u32> = input
            .face_corner_range(index as u32)
            .map(|c| c as u32)
            .collect();
        // A reflection flips handedness, so the copy has to be wound the other
        // way. Copied corner-for-corner it faces inward, and half the model is
        // lit from inside.
        ring.reverse();
        corners.reverse();

        // A face lying entirely on the plane would be duplicated on top of
        // itself, giving every one of its edges four faces. Skipped rather than
        // merged, because the original is already there.
        //
        // Compared as sets, not element by element: the reflected ring is also
        // reversed, so `ring == face` is false for every face but a degenerate
        // one and the duplicate slips through.
        let mut mapped = ring.clone();
        mapped.sort_unstable();
        debug_assert!(
            mapped.windows(2).all(|w| w[0] != w[1]),
            "reflection is injective — an on-plane vertex maps to itself and every \
             other to a fresh index — so a face's distinct vertices stay distinct"
        );
        let mut original: Vec<u32> = face.to_vec();
        original.sort_unstable();
        if mapped == original {
            continue;
        }

        faces.push(ring);
        face_order.push(index as u32);
        corner_order.extend(corners);
    }

    let mut out = MeshData::from_faces(new_positions, &faces);
    // The corner order must line up with how `from_faces` laid the corners out,
    // which is face by face in the order given — the same order they were pushed.
    super::carry_attributes(&mut out, input, &point_order, &corner_order, &face_order);
    // `point_order`, not `reflection`: the pairing wants output-to-source, and
    // `reflection` is source-to-*copy*, which would move every crease onto the
    // mirrored edge and leave the original smooth.
    super::gather_edges_by_pair(&mut out, input, &point_order);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::{names, AttrData, Domain};
    use crate::primitives;
    use kreate_math::Vec2;
    use std::sync::Arc;

    /// Half a box: the +X half of a cube, open on the mirror plane.
    fn half_box() -> MeshData {
        let positions = vec![
            Vec3::new(0.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(0.0, 1.0, -1.0),
            Vec3::new(0.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ];
        let faces = vec![
            vec![0, 3, 2, 1], // -Z
            vec![4, 5, 6, 7], // +Z
            vec![1, 2, 6, 5], // +X
            vec![0, 1, 5, 4], // -Y
            vec![2, 3, 7, 6], // +Y
        ];
        MeshData::from_faces(positions, &faces)
    }

    #[test]
    fn mirroring_a_half_box_closes_it() {
        let input = half_box();
        let out = mirror(&input, &MirrorParams::default());
        out.assert_valid();

        assert_eq!(out.num_verts(), 12, "the four seam vertices are shared");
        assert_eq!(out.num_faces(), 10, "five each side");
        assert_eq!(
            out.bounds().min.x,
            -1.0,
            "the copy reaches the other side: {:?}",
            out.bounds()
        );
    }

    #[test]
    fn the_mirrored_half_faces_outward() {
        // A reflection flips handedness. Copied corner-for-corner the copy faces
        // inward, and the report that comes back is "the shading is broken on one
        // side".
        let out = mirror(&half_box(), &MirrorParams::default());
        let normals = out.compute_face_normals();
        for (index, normal) in normals.iter().enumerate() {
            let ring: Vec<Vec3> = out
                .face_verts(index as u32)
                .iter()
                .map(|v| out.positions()[*v as usize])
                .collect();
            let mid: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
            assert!(
                normal.normalize_or_zero().dot(mid.normalize_or_zero()) > 0.4,
                "face {index} at {mid:?} faces {normal:?}, which is inward"
            );
        }
    }

    #[test]
    fn vertices_on_the_plane_are_welded_rather_than_doubled() {
        // Doubling them leaves a seam that looks closed and pulls apart under any
        // deformation — and every edge along it carries only one face.
        let out = mirror(&half_box(), &MirrorParams::default());
        let on_plane = out.positions().iter().filter(|p| p.x.abs() < 1e-6).count();
        assert_eq!(on_plane, 4, "four seam vertices, not eight");
        assert_eq!(out.validate(), Vec::new());
    }

    #[test]
    fn corner_layers_survive_the_mirror() {
        // Mirroring a cube is arithmetic; carrying its UVs through is where a
        // modifier is either right or quietly destructive, and it is invisible
        // until someone textures the result.
        //
        // Each corner's UV records the vertex it sits on, so every output corner
        // can be checked against its own vertex rather than merely against "some
        // corner of the original". Reversing the ring without reversing the
        // corner order passes the weaker check and shifts every mirrored UV by
        // one corner around its face — a seam that only shows once textured.
        let mut input = half_box();
        let corner_verts = input.corner_verts().to_vec();
        input.attributes_mut().insert(
            "uv",
            Domain::Corner,
            AttrData::F32x2(Arc::new(
                corner_verts
                    .iter()
                    .map(|v| Vec2::new(*v as f32, 0.0))
                    .collect(),
            )),
        );

        let out = mirror(&input, &MirrorParams::default());
        let uvs = out
            .attributes()
            .get_f32x2("uv")
            .expect("the layer survived");
        assert_eq!(uvs.len(), out.num_corners() as usize);

        // Which source vertex each output vertex came from. The half box's
        // vertices are distinct in (|x|, y, z), so the reflection is unambiguous.
        let source_of = |p: Vec3| -> u32 {
            input
                .positions()
                .iter()
                .position(|q| {
                    (Vec3::new(q.x.abs(), q.y, q.z) - Vec3::new(p.x.abs(), p.y, p.z)).length()
                        < 1e-6
                })
                .expect("every output vertex is a source vertex or its reflection")
                as u32
        };

        for (corner, vert) in out.corner_verts().iter().enumerate() {
            let expected = source_of(out.positions()[*vert as usize]);
            assert_eq!(
                uvs[corner].x, expected as f32,
                "corner {corner} sits on source vertex {expected} but carries the UV of {}",
                uvs[corner].x
            );
        }
    }

    #[test]
    fn face_layers_survive_the_mirror() {
        let mut input = half_box();
        let count = input.num_faces() as usize;
        input.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(Arc::new((0..count as i32).collect())),
        );

        let out = mirror(&input, &MirrorParams::default());
        let materials = out.attributes().get_i32(names::MATERIAL_INDEX).unwrap();
        assert_eq!(materials.len(), out.num_faces() as usize);
        for (i, value) in materials.iter().take(count).enumerate() {
            assert_eq!(*value, i as i32);
        }
    }

    /// Marks one edge of `input` sharp and returns the endpoint positions of
    /// every sharp edge of the mirrored result, as unordered pairs.
    fn creases_after_mirroring(mut input: MeshData, edge: usize) -> (Vec<Vec3>, Vec<[Vec3; 2]>) {
        let count = input.num_edges() as usize;
        input.attributes_mut().insert(
            names::SHARP_EDGE,
            Domain::Edge,
            AttrData::Bool(Arc::new((0..count).map(|i| i == edge).collect())),
        );
        let marked = input.edges()[edge];
        let marked_positions = vec![
            input.positions()[marked[0] as usize],
            input.positions()[marked[1] as usize],
        ];

        let out = mirror(&input, &MirrorParams::default());
        let sharp = out.attributes().get_bool(names::SHARP_EDGE).unwrap();
        let found = out
            .edges()
            .iter()
            .zip(sharp)
            .filter(|(_, s)| **s)
            .map(|(e, _)| {
                [
                    out.positions()[e[0] as usize],
                    out.positions()[e[1] as usize],
                ]
            })
            .collect();
        (marked_positions, found)
    }

    /// Are two edges the same, whichever way round their endpoints are given?
    fn same_edge(a: &[Vec3; 2], b: &[Vec3]) -> bool {
        (a[0] - b[0]).length() < 1e-6 && (a[1] - b[1]).length() < 1e-6
            || (a[0] - b[1]).length() < 1e-6 && (a[1] - b[0]).length() < 1e-6
    }

    #[test]
    fn an_edge_layer_follows_its_edge_and_not_its_index() {
        // `MeshData` derives its edge list from its faces in a canonical order,
        // so output indices have nothing to do with input ones. Copying edge
        // layers positionally puts every crease on the wrong edge — and looks
        // right on any mesh whose orders happen to agree.
        //
        // Edge 3 of the half box runs from the seam out to +X, so its reflection
        // is a different edge and both ends of the mirror should be creased: a
        // crease that survives on one half only shades the model asymmetrically,
        // which is the same complaint as a mirrored face wound inward.
        let (marked, found) = creases_after_mirroring(half_box(), 3);
        assert_eq!(found.len(), 2, "the edge and its reflection: {found:?}");

        let mut reflected: Vec<Vec3> = marked.clone();
        for p in &mut reflected {
            p.x = -p.x;
        }
        assert!(
            found.iter().any(|e| same_edge(e, &marked)),
            "the original lost its crease: {found:?} against {marked:?}"
        );
        assert!(
            found.iter().any(|e| same_edge(e, &reflected)),
            "the reflection never got the crease: {found:?} against {reflected:?}"
        );
    }

    #[test]
    fn a_crease_on_the_seam_is_not_counted_twice() {
        // An edge with both ends on the plane is its own reflection, so it is one
        // edge in the output and must not be double-marked or lose the layer.
        let input = half_box();
        let seam = input
            .edges()
            .iter()
            .position(|[a, b]| {
                input.positions()[*a as usize].x.abs() < 1e-6
                    && input.positions()[*b as usize].x.abs() < 1e-6
            })
            .expect("the half box is open on the plane, so it has seam edges");

        let (marked, found) = creases_after_mirroring(input, seam);
        assert_eq!(found.len(), 1, "one seam edge, creased once: {found:?}");
        assert!(same_edge(&found[0], &marked));
    }

    #[test]
    fn mirroring_two_axes_gives_four_quadrants() {
        // Applied one axis at a time, which is what makes X-then-Y produce four
        // quadrants rather than two.
        let input = half_box();
        let out = mirror(
            &input,
            &MirrorParams {
                x: true,
                y: true,
                z: false,
                merge_threshold: 1e-4,
            },
        );
        out.assert_valid();
        let bounds = out.bounds();
        assert!(bounds.min.x < -0.9 && bounds.max.x > 0.9);
        assert!(bounds.min.y < -0.9 && bounds.max.y > 0.9);
        assert!(out.num_faces() > input.num_faces() * 2);
    }

    #[test]
    fn mirroring_a_closed_mesh_that_straddles_the_plane_doubles_it() {
        // Not an error, and not something to be clever about: a cube centred on
        // the origin mirrors to a second cube on top of the first. The user asked
        // for a reflection and got one.
        let input = primitives::cube(2.0);
        let out = mirror(&input, &MirrorParams::default());
        out.assert_valid();
        assert_eq!(out.num_faces(), 12);
        assert_eq!(out.bounds().min.x, -1.0);
    }

    #[test]
    fn a_face_lying_on_the_plane_is_not_duplicated_onto_itself() {
        // Its reflection is itself, so copying it would give every one of its
        // edges four faces — a non-manifold mesh from an operation the user
        // thinks of as harmless.
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let input = MeshData::from_faces(positions, &[vec![0, 1, 2, 3]]);
        let out = mirror(&input, &MirrorParams::default());
        out.assert_valid();
        assert_eq!(out.num_faces(), 1, "the face is its own reflection");
        assert_eq!(out.num_verts(), 4);
    }

    #[test]
    fn no_axes_is_the_identity() {
        let input = half_box();
        let out = mirror(
            &input,
            &MirrorParams {
                x: false,
                y: false,
                z: false,
                merge_threshold: 1e-4,
            },
        );
        assert_eq!(out.positions(), input.positions());
        assert_eq!(out.corner_verts(), input.corner_verts());
    }

    #[test]
    fn a_zero_threshold_still_welds_what_is_exactly_on_the_plane() {
        let out = mirror(
            &half_box(),
            &MirrorParams {
                merge_threshold: 0.0,
                ..Default::default()
            },
        );
        out.assert_valid();
        assert_eq!(out.num_verts(), 12, "exact zeros are still on the plane");
    }
}
