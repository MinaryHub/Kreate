//! Editing a UV layer: selection, islands, and the 2D transform.
//!
//! The UV editor is a second viewport over the same mesh, and this is what it
//! stands on. Three things live here and nowhere else:
//!
//! **Selection is per corner.** One vertex is two UVs on opposite sides of a
//! seam, and selecting one must not select the other — so UV selection cannot
//! be the vertex selection wearing a different hat. It gets its own Corner
//! layer, [`names::SELECT_UV`], which is the same trick the three 3D
//! selections use: an attribute layer rather than a side table, so it survives
//! the `MeshData` round trip and shares structurally under undo with no extra
//! machinery.
//!
//! **Islands.** A UV map is a set of pieces cut apart along seams, and the
//! unit an artist actually grabs is the piece, not the corner. [`islands`]
//! finds them by walking faces across edges that are *UV-continuous* — both
//! ends agree on both sides — which is the definition of "not cut here". This
//! is what "select linked" means in UV space, and without it the editor is
//! unusable on anything real: dragging a shirt means clicking two hundred
//! corners.
//!
//! **The transform is a closed-form function of the original.** §2.2's rule,
//! and the reason a modal drag can be re-run from its parameters: apply
//! rotation and scale about a pivot, then translate. Never incrementally,
//! because a long drag accumulates float error and cancelling has to be a
//! restore rather than an inverse.

use glam::Vec2;

use crate::arena::{ArenaIndex, FaceId, LoopId};
use crate::attribute::{names, AttrType, Domain};
use crate::bmesh::BMesh;

/// How close two UVs must be to count as the same point.
///
/// Absolute rather than relative: UV space is the unit square by convention,
/// so there is a natural scale here that world space does not have. Loose
/// enough to survive an f32 round trip through a file, tight enough that two
/// deliberately separate islands a texel apart stay separate.
const WELD_EPSILON: f32 = 1e-5;

/// Ensures the UV selection layer exists, and returns its length.
fn ensure_selection(mesh: &mut BMesh) -> usize {
    let len = mesh.slot_count(Domain::Corner);
    mesh.attributes_mut()
        .ensure(names::SELECT_UV, Domain::Corner, AttrType::Bool, len);
    len
}

/// The selected corners, in ascending slot order.
///
/// Ascending because operator output must not depend on iteration accident —
/// the same rule that keeps golden files reproducible.
pub fn selected(mesh: &BMesh) -> Vec<LoopId> {
    let Some(flags) = mesh.attributes().get_bool(names::SELECT_UV) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for f in mesh.face_ids() {
        for l in mesh.face_loops(f) {
            let slot = l.index() as usize;
            if flags.get(slot).copied().unwrap_or(false) {
                out.push(l);
            }
        }
    }
    out.sort_by_key(|l| l.index());
    out.dedup_by_key(|l| l.index());
    out
}

/// What [`set_selection`] does with the corners it is given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelectAction {
    /// Replace the selection with these corners.
    #[default]
    Replace,
    /// Add these corners to the selection.
    Extend,
    /// Remove these corners from the selection.
    Subtract,
}

/// Applies `corners` to the UV selection, returning how many corners are
/// selected afterwards.
pub fn set_selection(mesh: &mut BMesh, corners: &[LoopId], action: SelectAction) -> usize {
    ensure_selection(mesh);
    let Some(flags) = mesh.attributes_mut().make_mut_bool(names::SELECT_UV) else {
        return 0;
    };
    if action == SelectAction::Replace {
        flags.iter_mut().for_each(|f| *f = false);
    }
    let value = action != SelectAction::Subtract;
    for l in corners {
        let slot = l.index() as usize;
        if slot < flags.len() {
            flags[slot] = value;
        }
    }
    flags.iter().filter(|f| **f).count()
}

/// Selects every corner, or none, or the complement.
pub fn select_all(mesh: &mut BMesh, selected: bool) -> usize {
    ensure_selection(mesh);
    // Only corners that belong to a face: a free slot in the arena has a UV
    // value but no geometry, and selecting it would put the transform's pivot
    // somewhere nothing is.
    let live: Vec<LoopId> = mesh.face_ids().flat_map(|f| mesh.face_loops(f)).collect();
    if selected {
        set_selection(mesh, &live, SelectAction::Replace)
    } else {
        set_selection(mesh, &[], SelectAction::Replace)
    }
}

/// Inverts the UV selection.
pub fn select_invert(mesh: &mut BMesh) -> usize {
    ensure_selection(mesh);
    let live: Vec<LoopId> = mesh.face_ids().flat_map(|f| mesh.face_loops(f)).collect();
    let was: Vec<bool> = {
        let flags = mesh.attributes().get_bool(names::SELECT_UV).unwrap();
        live.iter()
            .map(|l| flags.get(l.index() as usize).copied().unwrap_or(false))
            .collect()
    };
    let wanted: Vec<LoopId> = live
        .iter()
        .zip(&was)
        .filter(|(_, on)| !**on)
        .map(|(l, _)| *l)
        .collect();
    set_selection(mesh, &wanted, SelectAction::Replace)
}

/// Whether two faces are joined across `edge` in the UV map.
///
/// Continuous when both ends of the edge agree on both sides. That is exactly
/// what "there is no seam here" means, and it is a stronger test than "the
/// faces touch": two faces sharing an edge in 3D are routinely cut apart in
/// UV, which is the entire purpose of a seam.
fn uv_continuous(mesh: &BMesh, a: LoopId, b: LoopId, uvs: &[Vec2]) -> bool {
    let (Some(a_next), Some(b_next)) = (mesh.loop_next(a), mesh.loop_next(b)) else {
        return false;
    };
    let (Some(av), Some(bv)) = (mesh.loop_vert(a), mesh.loop_vert(b)) else {
        return false;
    };
    let at = |l: LoopId| uvs.get(l.index() as usize).copied().unwrap_or(Vec2::ZERO);
    let same = |x: Vec2, y: Vec2| (x - y).abs().max_element() <= WELD_EPSILON;

    // The two loops run the edge in opposite directions on a manifold pair, so
    // `a` pairs with `b_next` and `a_next` with `b`. Checked against the
    // vertices rather than assumed, because a non-manifold or mis-wound pair
    // reaches here too and must simply answer "not continuous" instead of
    // producing a wrong island.
    if mesh.loop_vert(b_next) == Some(av) && mesh.loop_vert(a_next) == Some(bv) {
        same(at(a), at(b_next)) && same(at(a_next), at(b))
    } else if mesh.loop_vert(b) == Some(av) && mesh.loop_vert(b_next) == Some(bv) {
        // Same winding on both sides: the surface is inconsistently wound
        // here. Still answerable.
        same(at(a), at(b)) && same(at(a_next), at(b_next))
    } else {
        false
    }
}

/// The UV islands of a layer, each as its corners in ascending slot order.
///
/// An island is a maximal set of faces reachable from one another across
/// UV-continuous edges — the piece an artist grabs. Islands come back in a
/// deterministic order, seeded by ascending face index, so a script that acts
/// on "the third island" acts on the same one twice.
pub fn islands(mesh: &BMesh, layer: &str) -> Vec<Vec<LoopId>> {
    let Some(uvs) = mesh.attributes().get_f32x2(layer) else {
        return Vec::new();
    };
    let faces: Vec<FaceId> = mesh.face_ids().collect();
    let slot_count = mesh.slot_count(Domain::Face);
    let mut seen = vec![false; slot_count];
    let mut out = Vec::new();

    for seed in &faces {
        if seen.get(seed.index() as usize).copied().unwrap_or(true) {
            continue;
        }
        let mut stack = vec![*seed];
        seen[seed.index() as usize] = true;
        let mut corners: Vec<LoopId> = Vec::new();

        while let Some(f) = stack.pop() {
            let loops = mesh.face_loops(f);
            corners.extend(loops.iter().copied());
            for l in &loops {
                let Some(e) = mesh.loop_edge(*l) else {
                    continue;
                };
                for other in mesh.edge_loops(e) {
                    if other.index() == l.index() {
                        continue;
                    }
                    let Some(nf) = mesh.loop_face(other) else {
                        continue;
                    };
                    if seen.get(nf.index() as usize).copied().unwrap_or(true) {
                        continue;
                    }
                    if uv_continuous(mesh, *l, other, uvs) {
                        seen[nf.index() as usize] = true;
                        stack.push(nf);
                    }
                }
            }
        }
        corners.sort_by_key(|l| l.index());
        corners.dedup_by_key(|l| l.index());
        out.push(corners);
    }
    out
}

/// Grows the selection to whole islands.
///
/// "Select linked", in UV space. An island with one selected corner becomes
/// wholly selected; an island with none is left alone.
pub fn select_linked(mesh: &mut BMesh, layer: &str) -> usize {
    let groups = islands(mesh, layer);
    ensure_selection(mesh);
    let flags: Vec<bool> = mesh
        .attributes()
        .get_bool(names::SELECT_UV)
        .map(|f| f.to_vec())
        .unwrap_or_default();
    let mut wanted: Vec<LoopId> = Vec::new();
    for group in &groups {
        let touched = group
            .iter()
            .any(|l| flags.get(l.index() as usize).copied().unwrap_or(false));
        if touched {
            wanted.extend(group.iter().copied());
        }
    }
    set_selection(mesh, &wanted, SelectAction::Extend)
}

/// A 2D similarity transform of the UV plane.
///
/// Scale and rotation about `pivot`, then translation. Held as parameters
/// rather than as a matrix so an operator can carry it, a script can write it,
/// and the redo panel can show it — the same reason [`crate::edit`]'s other
/// tools take numbers rather than transforms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform2 {
    pub translation: Vec2,
    /// Radians, counter-clockwise.
    pub rotation: f32,
    pub scale: Vec2,
    pub pivot: Vec2,
}

impl Default for Transform2 {
    fn default() -> Self {
        Self {
            translation: Vec2::ZERO,
            rotation: 0.0,
            scale: Vec2::ONE,
            pivot: Vec2::ZERO,
        }
    }
}

impl Transform2 {
    /// Where one point goes.
    pub fn apply(&self, uv: Vec2) -> Vec2 {
        let d = uv - self.pivot;
        let scaled = d * self.scale;
        let (sin, cos) = self.rotation.sin_cos();
        let rotated = Vec2::new(
            scaled.x * cos - scaled.y * sin,
            scaled.x * sin + scaled.y * cos,
        );
        self.pivot + rotated + self.translation
    }

    /// Whether this transform is the identity, and so worth skipping.
    pub fn is_identity(&self) -> bool {
        self.translation == Vec2::ZERO && self.rotation == 0.0 && self.scale == Vec2::ONE
    }
}

/// The median of a set of corners' UVs — the default pivot.
///
/// The median of the *distinct* positions, not of the corners: a corner is
/// duplicated wherever several faces meet, and averaging over corners would
/// pull the pivot toward whichever vertex happens to have the most faces on
/// it. The same reasoning as the 3D transform's pivot.
pub fn median(mesh: &BMesh, corners: &[LoopId], layer: &str) -> Vec2 {
    let Some(uvs) = mesh.attributes().get_f32x2(layer) else {
        return Vec2::ZERO;
    };
    let mut distinct: Vec<Vec2> = Vec::new();
    for l in corners {
        let Some(uv) = uvs.get(l.index() as usize).copied() else {
            continue;
        };
        if !distinct
            .iter()
            .any(|p| (*p - uv).abs().max_element() <= WELD_EPSILON)
        {
            distinct.push(uv);
        }
    }
    if distinct.is_empty() {
        return Vec2::ZERO;
    }
    distinct.iter().copied().sum::<Vec2>() / distinct.len() as f32
}

/// Transforms the given corners' UVs, returning how many were written.
///
/// Applied to whatever is in the layer now. A modal drag re-runs this from the
/// restored original every frame rather than compounding — see §2.2 — so this
/// function has no history of its own to get wrong.
pub fn transform(mesh: &mut BMesh, corners: &[LoopId], layer: &str, how: &Transform2) -> usize {
    if corners.is_empty() || how.is_identity() {
        return 0;
    }
    let Some(uvs) = mesh.attributes_mut().make_mut_f32x2(layer) else {
        return 0;
    };
    let mut count = 0;
    for l in corners {
        let slot = l.index() as usize;
        if slot < uvs.len() {
            uvs[slot] = how.apply(uvs[slot]);
            count += 1;
        }
    }
    count
}

/// The bounding box of a set of corners' UVs, as `(min, max)`.
///
/// `None` when there is nothing to bound. The UV editor frames the view with
/// it and the packer measures islands with it.
pub fn bounds(mesh: &BMesh, corners: &[LoopId], layer: &str) -> Option<(Vec2, Vec2)> {
    let uvs = mesh.attributes().get_f32x2(layer)?;
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    let mut any = false;
    for l in corners {
        if let Some(uv) = uvs.get(l.index() as usize) {
            min = min.min(*uv);
            max = max.max(*uv);
            any = true;
        }
    }
    if any {
        Some((min, max))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::uv_project::{self, Fit, Projection};
    use crate::primitives;
    use crate::MeshData;
    use glam::Vec3;

    fn cube_with_uvs() -> BMesh {
        let mesh = primitives::cube(2.0);
        let mut bm = BMesh::from_mesh_data(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        uv_project::project(&mut bm, &faces, Projection::Box, uv_project::DEFAULT_LAYER);
        bm
    }

    fn uvs(bm: &BMesh) -> Vec<Vec2> {
        bm.attributes()
            .get_f32x2(uv_project::DEFAULT_LAYER)
            .unwrap()
            .to_vec()
    }

    #[test]
    fn selection_is_per_corner_and_not_per_vertex() {
        // The premise of the whole module. One vertex of a box-projected cube
        // is three corners on three different sides of the map; selecting one
        // must leave the other two alone, or a seam cannot be dragged apart.
        let mut bm = cube_with_uvs();
        let first_face = bm.face_ids().next().unwrap();
        let one = bm.face_loops(first_face)[0];
        let vertex = bm.loop_vert(one).unwrap();

        assert_eq!(set_selection(&mut bm, &[one], SelectAction::Replace), 1);

        // Every other corner on that same vertex.
        let siblings: Vec<LoopId> = bm
            .face_ids()
            .flat_map(|f| bm.face_loops(f))
            .filter(|l| *l != one && bm.loop_vert(*l) == Some(vertex))
            .collect();
        assert!(
            !siblings.is_empty(),
            "the test needs a vertex shared by several faces"
        );
        let chosen = selected(&bm);
        assert_eq!(chosen.len(), 1);
        for s in &siblings {
            assert!(
                !chosen.iter().any(|c| c.index() == s.index()),
                "selecting one corner selected another on the same vertex"
            );
        }
    }

    #[test]
    fn select_all_none_and_invert_agree_with_each_other() {
        let mut bm = cube_with_uvs();
        let total = 24;
        assert_eq!(select_all(&mut bm, true), total);
        assert_eq!(select_all(&mut bm, false), 0);

        let first = bm.face_loops(bm.face_ids().next().unwrap());
        set_selection(&mut bm, &first, SelectAction::Replace);
        assert_eq!(selected(&bm).len(), 4);
        assert_eq!(select_invert(&mut bm), total - 4);
        assert_eq!(select_invert(&mut bm), 4);
    }

    #[test]
    fn extend_and_subtract_do_what_they_say() {
        let mut bm = cube_with_uvs();
        let faces: Vec<FaceId> = bm.face_ids().collect();
        let a = bm.face_loops(faces[0]);
        let b = bm.face_loops(faces[1]);

        set_selection(&mut bm, &a, SelectAction::Replace);
        assert_eq!(set_selection(&mut bm, &b, SelectAction::Extend), 8);
        assert_eq!(set_selection(&mut bm, &a, SelectAction::Subtract), 4);
        // And replace really replaces rather than adding.
        assert_eq!(set_selection(&mut bm, &a, SelectAction::Replace), 4);
    }

    #[test]
    fn a_box_projected_cube_is_six_islands() {
        // Every side lands on its own patch of the map, so nothing is
        // continuous with anything: six faces, six islands. This is the case
        // that tells island-finding apart from "the faces touch in 3D", where
        // the answer would be one.
        let bm = cube_with_uvs();
        let groups = islands(&bm, uv_project::DEFAULT_LAYER);
        assert_eq!(
            groups.len(),
            6,
            "{:?}",
            groups.iter().map(Vec::len).collect::<Vec<_>>()
        );
        for g in &groups {
            assert_eq!(g.len(), 4);
        }
    }

    #[test]
    fn a_planar_projected_grid_is_one_island() {
        // The other half of the same claim. A flat grid projected along its own
        // normal is continuous everywhere, so it is one island — and if the
        // continuity test were wrong in the permissive direction the cube test
        // alone would not say so.
        let mesh = primitives::grid(2.0, 4, 4);
        let mut bm = BMesh::from_mesh_data(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        uv_project::project_with(
            &mut bm,
            &faces,
            Projection::Planar {
                right: Vec3::X,
                up: Vec3::Y,
            },
            uv_project::DEFAULT_LAYER,
            Fit::Bounds,
        );
        let groups = islands(&bm, uv_project::DEFAULT_LAYER);
        assert_eq!(
            groups.len(),
            1,
            "a continuous map came back as {} islands",
            groups.len()
        );
        assert_eq!(groups[0].len(), faces.len() * 4);
    }

    #[test]
    fn moving_one_face_apart_splits_its_island() {
        // Islands are a property of the *UVs*, not of the topology. Dragging
        // one face off its neighbours has to cut it out, or "select linked"
        // would keep grabbing geometry the artist has already separated.
        let mesh = primitives::grid(2.0, 3, 3);
        let mut bm = BMesh::from_mesh_data(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        uv_project::project_with(
            &mut bm,
            &faces,
            Projection::Planar {
                right: Vec3::X,
                up: Vec3::Y,
            },
            uv_project::DEFAULT_LAYER,
            Fit::None,
        );
        assert_eq!(islands(&bm, uv_project::DEFAULT_LAYER).len(), 1);

        let one = bm.face_loops(faces[0]);
        transform(
            &mut bm,
            &one,
            uv_project::DEFAULT_LAYER,
            &Transform2 {
                translation: Vec2::new(10.0, 0.0),
                ..Default::default()
            },
        );
        assert_eq!(islands(&bm, uv_project::DEFAULT_LAYER).len(), 2);
    }

    #[test]
    fn select_linked_grows_to_the_island_and_no_further() {
        let mut bm = cube_with_uvs();
        let faces: Vec<FaceId> = bm.face_ids().collect();
        let one = bm.face_loops(faces[0])[0];
        set_selection(&mut bm, &[one], SelectAction::Replace);
        assert_eq!(select_linked(&mut bm, uv_project::DEFAULT_LAYER), 4);
        // A box-projected cube's islands are single faces, so growing again
        // must change nothing rather than creeping outward.
        assert_eq!(select_linked(&mut bm, uv_project::DEFAULT_LAYER), 4);
    }

    #[test]
    fn islands_come_back_in_a_stable_order() {
        // Operator output must not depend on iteration accident. Two runs over
        // the same mesh must agree, or a script acting on "island 3" acts on a
        // different one each time and a golden file goes flaky.
        let bm = cube_with_uvs();
        let a = islands(&bm, uv_project::DEFAULT_LAYER);
        let b = islands(&bm, uv_project::DEFAULT_LAYER);
        let ids = |g: &Vec<Vec<LoopId>>| -> Vec<Vec<u32>> {
            g.iter()
                .map(|island| island.iter().map(|l| l.index()).collect())
                .collect()
        };
        assert_eq!(ids(&a), ids(&b));
    }

    #[test]
    fn a_translation_moves_the_selection_and_nothing_else() {
        let mut bm = cube_with_uvs();
        let before = uvs(&bm);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        let one = bm.face_loops(faces[0]);

        let n = transform(
            &mut bm,
            &one,
            uv_project::DEFAULT_LAYER,
            &Transform2 {
                translation: Vec2::new(0.25, -0.5),
                ..Default::default()
            },
        );
        assert_eq!(n, 4);

        let after = uvs(&bm);
        let moved: Vec<usize> = before
            .iter()
            .zip(&after)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(moved.len(), 4);
        for i in moved {
            assert!((after[i] - before[i] - Vec2::new(0.25, -0.5)).length() < 1e-6);
        }
    }

    #[test]
    fn rotation_and_scale_happen_about_the_pivot() {
        // A quarter turn about the selection's own median sends each corner to
        // where its neighbour was. Checked as a set, because which corner
        // lands where depends on winding and is not the claim.
        let mut bm = cube_with_uvs();
        let faces: Vec<FaceId> = bm.face_ids().collect();
        let one = bm.face_loops(faces[0]);
        let pivot = median(&bm, &one, uv_project::DEFAULT_LAYER);
        let before: Vec<Vec2> = one.iter().map(|l| uvs(&bm)[l.index() as usize]).collect();

        transform(
            &mut bm,
            &one,
            uv_project::DEFAULT_LAYER,
            &Transform2 {
                rotation: std::f32::consts::FRAC_PI_2,
                pivot,
                ..Default::default()
            },
        );
        let after: Vec<Vec2> = one.iter().map(|l| uvs(&bm)[l.index() as usize]).collect();

        for a in &after {
            assert!(
                before.iter().any(|b| (*b - *a).length() < 1e-5),
                "a quarter turn about the median did not permute the corners: \\
                 {before:?} -> {after:?}"
            );
        }
        // And the median has not moved, which is what "about the pivot" means.
        let moved = median(&bm, &one, uv_project::DEFAULT_LAYER);
        assert!((moved - pivot).length() < 1e-5, "{pivot:?} -> {moved:?}");
    }

    #[test]
    fn the_pivot_is_the_median_of_distinct_positions() {
        // Not of corners. A corner is duplicated wherever faces meet, so
        // averaging corners drags the pivot toward the busiest vertex — the
        // same mistake the 3D transform's pivot avoids.
        //
        // The selection has to be *asymmetric* for this to say anything. The
        // first version of this test used all four faces of a 2x2 grid, where
        // symmetry puts both averages at the centre — so replacing the
        // distinct-position median with a corner average changed nothing and
        // the test passed against the bug. Three faces of the four is the
        // smallest shape where the two answers differ.
        let mesh = primitives::grid(2.0, 2, 2);
        let mut bm = BMesh::from_mesh_data(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        assert_eq!(faces.len(), 4);
        uv_project::project_with(
            &mut bm,
            &faces,
            Projection::Planar {
                right: Vec3::X,
                up: Vec3::Y,
            },
            uv_project::DEFAULT_LAYER,
            Fit::Bounds,
        );

        let chosen: Vec<LoopId> = faces[..3].iter().flat_map(|f| bm.face_loops(*f)).collect();
        let uvs = bm
            .attributes()
            .get_f32x2(uv_project::DEFAULT_LAYER)
            .unwrap();
        let at = |l: &LoopId| uvs[l.index() as usize];

        // What a corner average would say.
        let corner_mean = chosen.iter().map(at).sum::<Vec2>() / chosen.len() as f32;
        // What the distinct positions say.
        let mut distinct: Vec<Vec2> = Vec::new();
        for uv in chosen.iter().map(at) {
            if !distinct.iter().any(|p| (*p - uv).length() < 1e-6) {
                distinct.push(uv);
            }
        }
        let distinct_mean = distinct.iter().copied().sum::<Vec2>() / distinct.len() as f32;

        // The premise: on this selection the two genuinely disagree, so the
        // assertion below can fail.
        assert!(
            (corner_mean - distinct_mean).length() > 1e-3,
            "the selection is too symmetric to tell the two apart: \
             {corner_mean:?} vs {distinct_mean:?}"
        );
        let pivot = median(&bm, &chosen, uv_project::DEFAULT_LAYER);
        assert!(
            (pivot - distinct_mean).length() < 1e-5,
            "pivot {pivot:?} is not the distinct-position median {distinct_mean:?}"
        );
    }

    #[test]
    fn a_slot_whose_corner_is_gone_resolves_to_nothing() {
        // `BMesh::corner_at` is the bridge from an outside index — a click in
        // the UV editor, a selection restored from a file — to a handle, and
        // the generation check is the whole reason it exists. An index that is
        // still *in range* but whose slot has been freed is the dangerous
        // case: a length check cannot catch it, so without the generation it
        // would alias whatever now lives there.
        //
        // The first attempt at this used an index of 9999, which the length
        // check in `set_selection` rejects on its own — so replacing the
        // generation check with a bare `from_parts` left every test passing.
        let mut bm = cube_with_uvs();
        let doomed = bm.face_ids().next().unwrap();
        let slots: Vec<u32> = bm.face_loops(doomed).iter().map(|l| l.index()).collect();
        assert_eq!(slots.len(), 4);
        for slot in &slots {
            assert!(bm.corner_at(*slot).is_some(), "slot {slot} starts occupied");
        }

        bm.set_face_selected(doomed, true);
        crate::edit::delete(&mut bm, crate::edit::DeleteMode::Faces);

        for slot in &slots {
            assert!(
                (*slot as usize) < bm.slot_count(Domain::Corner),
                "slot {slot} left the arena, so a length check would catch it \
                 and this test would prove nothing"
            );
            assert!(
                bm.corner_at(*slot).is_none(),
                "slot {slot} still resolves to a corner after its face was deleted"
            );
        }
    }

    #[test]
    fn an_identity_transform_writes_nothing() {
        // Cheap, and it is what lets a modal drag that has not moved yet leave
        // the mesh alone rather than rewriting every selected UV to itself.
        let mut bm = cube_with_uvs();
        let all: Vec<LoopId> = bm.face_ids().flat_map(|f| bm.face_loops(f)).collect();
        assert_eq!(
            transform(
                &mut bm,
                &all,
                uv_project::DEFAULT_LAYER,
                &Transform2::default()
            ),
            0
        );
    }

    #[test]
    fn the_selection_survives_a_round_trip_through_mesh_data() {
        // The reason selection is an attribute layer rather than a side table.
        // Leaving edit mode and coming back must not lose what was selected,
        // and undo shares the layer structurally with no extra machinery.
        let mut bm = cube_with_uvs();
        let faces: Vec<FaceId> = bm.face_ids().collect();
        let one = bm.face_loops(faces[0]);
        set_selection(&mut bm, &one, SelectAction::Replace);

        let data: MeshData = bm.to_mesh_data();
        let back = BMesh::from_mesh_data(&data);
        assert_eq!(selected(&back).len(), 4, "the UV selection was lost");
    }

    #[test]
    fn bounds_frame_the_selection() {
        let bm = cube_with_uvs();
        let faces: Vec<FaceId> = bm.face_ids().collect();
        let one = bm.face_loops(faces[0]);
        let (min, max) = bounds(&bm, &one, uv_project::DEFAULT_LAYER).unwrap();
        assert!(min.x < max.x && min.y < max.y, "{min:?} {max:?}");
        assert!(bounds(&bm, &[], uv_project::DEFAULT_LAYER).is_none());
    }
}
