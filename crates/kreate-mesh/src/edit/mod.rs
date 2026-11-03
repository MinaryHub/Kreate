//! Mesh editing operations on [`BMesh`](crate::BMesh).
//!
//! These are the topology half of the operators the plan lists for v1. They take
//! a `&mut BMesh` and element handles and know nothing about documents, undo or
//! parameters — `kreate-ops` wraps each one in an [`Operator`] and supplies
//! those. The split is what lets the whole modelling toolset be tested without a
//! document, a GPU or a mouse.
//!
//! [`Operator`]: https://docs.rs/kreate-ops

pub mod bevel;
pub mod bridge;
pub mod delete;
pub mod extrude;
pub mod fill;
pub mod inset;
pub mod join;
pub mod knife;
pub mod limited;
pub mod loopcut;
pub mod merge;
pub mod normals;
pub mod region;
pub mod slide;
pub mod spin;
pub mod split;
pub mod subdivide;
pub mod symmetrize;
pub mod uv_edit;
pub mod uv_pack;
pub mod uv_project;
pub mod uv_unwrap;
pub mod walk;

pub use bevel::{
    bevel_edges, bevel_edges_segments, bevel_selected, bevel_selected_segments, Bevel,
};
pub use bridge::{bridge_loops, bridge_selected, open_borders, Bridge, BridgeError};
pub use delete::{delete, dissolve_edges, dissolve_faces, dissolve_verts, DeleteMode, Dissolution};
pub use extrude::{
    extrude_edges_and_verts, extrude_faces, extrude_selected_faces,
    extrude_selected_faces_along_normals, Extrusion, WireExtrusion,
};
pub use fill::{fill, fill_selected, grid_fill, the_only_border, Fill, FillError};
pub use inset::{inset_individual, inset_region, inset_selected, Inset};
pub use join::{join, separate_selected, Part, Report as JoinReport};
pub use knife::{knife, Knife};
pub use limited::limited_dissolve;
pub use loopcut::{loop_cut, loop_cut_selected, LoopCut};
pub use merge::{
    merge_groups, merge_selected, weld_all, weld_by_distance, weld_onto, Merge, MergeTarget,
};
pub use normals::{
    auto_smooth, flip, recalculate, set_crease, set_shade_smooth, set_sharp, set_sharp_selected,
    Recalculation,
};
pub use region::Region;
pub use slide::{edge_slide, rails as slide_rails, Rail};
pub use spin::{close_spin, spin, spin_selected, Spin};
pub use split::{rip_selected_edges, split_edges, split_region, split_selected_faces, Split};
pub use subdivide::{subdivide_faces, subdivide_selected, Subdivision};
pub use symmetrize::symmetrize;
pub use uv_edit::{islands as uv_islands, SelectAction as UvSelectAction, Transform2};
pub use uv_pack::{cluster_by_angle, pack as uv_pack, smart_project};
pub use uv_project::{project as uv_project, project_with as uv_project_with, Fit, Projection};
pub use walk::{edge_loop, edge_ring, face_loop};

/// How far `x` lies along the segment from `a` to `b`, `0..=1`: where a
/// vertex placed on an edge sits, for its corner values to be blended to.
pub(crate) fn along(
    mesh: &crate::BMesh,
    a: crate::VertId,
    b: crate::VertId,
    x: crate::VertId,
) -> f32 {
    let (Some(pa), Some(pb), Some(px)) = (mesh.position(a), mesh.position(b), mesh.position(x))
    else {
        return 0.5;
    };
    let d = pb - pa;
    let len2 = d.length_squared();
    if len2 < 1e-20 {
        return 0.5;
    }
    ((px - pa).dot(d) / len2).clamp(0.0, 1.0)
}

/// A face's corner values blended to a point inside it: the barycentric blend
/// in the fan triangle around the first corner that holds the point (or comes
/// nearest to), in the face's plane. Exact for anything that varies linearly
/// over a convex face, which is what a UV map does.
pub(crate) fn values_inside(
    corners: &[(kreate_math::Vec3, crate::ElementValues)],
    normal: kreate_math::Vec3,
    p: kreate_math::Vec3,
) -> crate::ElementValues {
    let n = corners.len();
    if n == 0 {
        return crate::ElementValues::default();
    }
    if n < 3 {
        return corners[0].1.clone();
    }
    let (o, _) = &corners[0];
    let mut best: Option<(f32, usize, [f32; 3])> = None;
    for i in 1..n - 1 {
        let (a, b) = (corners[i].0, corners[i + 1].0);
        let area = (a - *o).cross(b - *o).dot(normal);
        if area.abs() < 1e-12 {
            continue;
        }
        // Each corner's weight is the area of the triangle opposite it.
        let w_o = (b - a).cross(p - a).dot(normal) / area;
        let w_a = (*o - b).cross(p - b).dot(normal) / area;
        let w_b = 1.0 - w_o - w_a;
        let worst = w_o.min(w_a).min(w_b);
        if best.is_none_or(|(w, ..)| worst > w) {
            best = Some((worst, i, [w_o, w_a, w_b]));
        }
    }
    match best {
        Some((_, i, [wo, wa, wb])) => crate::ElementValues::blend(&[
            (&corners[0].1, wo),
            (&corners[i].1, wa),
            (&corners[i + 1].1, wb),
        ]),
        None => corners[0].1.clone(),
    }
}

/// What the operator tests check is carried onto rebuilt faces.
#[cfg(test)]
pub(crate) mod carried {
    use crate::attribute::{names, AttrType, Domain};
    use crate::{BMesh, FaceId};
    use kreate_math::Vec2;

    /// Material 3, smooth shading and a UV of (5 + x, 5 + y) on every face
    /// and corner of `mesh` — none of them a layer's default, so a face born
    /// without them shows.
    pub fn paint(mesh: &mut BMesh) {
        use crate::arena::ArenaIndex;
        let faces = mesh.slot_count(Domain::Face);
        let corners = mesh.slot_count(Domain::Corner);
        let a = mesh.attributes_mut();
        a.ensure(names::MATERIAL_INDEX, Domain::Face, AttrType::I32, faces);
        a.ensure(names::SHADE_SMOOTH, Domain::Face, AttrType::Bool, faces);
        a.ensure(names::UV, Domain::Corner, AttrType::F32x2, corners);
        a.make_mut_i32(names::MATERIAL_INDEX).unwrap().fill(3);
        a.make_mut_bool(names::SHADE_SMOOTH).unwrap().fill(true);
        let uv: Vec<(usize, Vec2)> = mesh
            .loop_ids()
            .filter_map(|l| {
                let p = mesh.position(mesh.loop_vert(l)?)?;
                Some((l.index() as usize, Vec2::new(5.0 + p.x, 5.0 + p.y)))
            })
            .collect();
        let layer = mesh.attributes_mut().make_mut_f32x2(names::UV).unwrap();
        for (i, v) in uv {
            layer[i] = v;
        }
    }

    /// Every corner's UV is still `paint`'s function of where its vertex
    /// is: true of an operator that cuts a face where it lies, whose new
    /// corners take UVs interpolated from the old ones.
    pub fn assert_uv_follows_position(mesh: &BMesh) {
        use crate::arena::ArenaIndex;
        let uv = mesh.attributes().get_f32x2(names::UV).unwrap();
        for l in mesh.loop_ids() {
            let p = mesh.position(mesh.loop_vert(l).unwrap()).unwrap();
            let want = Vec2::new(5.0 + p.x, 5.0 + p.y);
            let got = uv[l.index() as usize];
            assert!(
                got.distance(want) < 1e-4,
                "a corner at {p} has UV {got}, not {want}"
            );
        }
    }

    /// Every one of `faces` has the material, the smoothing and a UV on each
    /// corner that `paint` gave.
    pub fn assert_kept(mesh: &BMesh, faces: &[FaceId]) {
        use crate::arena::ArenaIndex;
        assert!(!faces.is_empty());
        let a = mesh.attributes();
        let material = a.get_i32(names::MATERIAL_INDEX).unwrap();
        let smooth = a.get_bool(names::SHADE_SMOOTH).unwrap();
        let uv = a.get_f32x2(names::UV).unwrap();
        for f in faces {
            let i = f.index() as usize;
            assert_eq!(material[i], 3, "face {i} lost its material");
            assert!(smooth[i], "face {i} lost its smooth shading");
            for l in mesh.face_loops(*f) {
                let u = uv[l.index() as usize];
                assert!(u != Vec2::ZERO, "a corner of face {i} lost its UV");
            }
        }
    }
}
