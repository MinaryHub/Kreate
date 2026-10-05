//! Normal operations: winding, smoothing, and sharp edges.
//!
//! None of these move a vertex. They change how the surface is *interpreted* —
//! which way a face points, and where the shading is allowed to blend — and that
//! is why they are worth having as first-class operators rather than as
//! properties buried in a panel. A model with inconsistent winding looks broken
//! in a way users report as a rendering bug, and a model with no sharp edges
//! looks like a bag of soap.
//!
//! # The canonical form
//!
//! Following the plan's interoperability model, smoothing is stored as
//! per-**edge** `sharp_edge`, per-**face** `shade_smooth`, and an optional
//! auto-smooth angle. Every format's own idea — OBJ's smoothing groups, FBX's
//! per-polygon-vertex layers, glTF's split vertices — converts to and from that
//! one representation at the import boundary. Which means these operators are
//! also what makes those conversions round-trip.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, EdgeId, FaceId};
use crate::attribute::{names, AttrType, Domain};
use crate::bmesh::BMesh;

use super::region::Region;

/// What a recalculation did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Recalculation {
    /// Faces whose winding was reversed.
    pub flipped: usize,
    /// Connected groups considered.
    pub groups: usize,
    /// Groups whose outside could not be determined, and were only made
    /// *consistent* rather than outward.
    pub ambiguous: usize,
}

/// Reverses the winding of the given faces.
pub fn flip(mesh: &mut BMesh, faces: &[FaceId]) -> usize {
    let region = Region::new(mesh, faces);
    let mut flipped = 0;
    for f in region.faces {
        if mesh.reverse_face(f) {
            flipped += 1;
        }
    }
    flipped
}

/// Makes winding consistent within each connected group, then points it outward.
///
/// Two steps, and they fail differently. Making a group *consistent* is exact:
/// two faces sharing an edge agree if their loops traverse that edge in opposite
/// directions, and a walk propagates that from any seed. Deciding which way is
/// *out* needs a closed surface, where signed volume answers it; on an open one
/// there is no inside, and the fallback is a heuristic.
///
/// Both are reported. A tool that quietly guesses on an open surface leaves the
/// user to notice that half their model is lit from behind.
pub fn recalculate(mesh: &mut BMesh, faces: &[FaceId], inward: bool) -> Recalculation {
    let region = Region::new(mesh, faces);
    let mut out = Recalculation::default();

    for group in region.components(mesh) {
        out.groups += 1;
        make_consistent(mesh, &group, &mut out);

        let (volume, ambiguous) = orientation_of(mesh, &group);
        if ambiguous {
            out.ambiguous += 1;
        }
        // A positive signed volume means the winding already points out.
        let wants_flip = if inward { volume > 0.0 } else { volume < 0.0 };
        if wants_flip {
            for &f in &group {
                if mesh.reverse_face(f) {
                    out.flipped += 1;
                }
            }
        }
    }
    out
}

/// Propagates one winding through a group of faces joined by shared edges.
fn make_consistent(mesh: &mut BMesh, group: &[FaceId], out: &mut Recalculation) {
    let mut in_group = vec![false; mesh.slot_count(Domain::Face)];
    for f in group {
        in_group[f.index() as usize] = true;
    }
    let mut settled = vec![false; mesh.slot_count(Domain::Face)];

    // Seeded from the lowest-numbered face of each connected piece, so the
    // result does not depend on iteration order.
    for &seed in group {
        if settled[seed.index() as usize] {
            continue;
        }
        settled[seed.index() as usize] = true;
        let mut queue = vec![seed];
        while let Some(current) = queue.pop() {
            for l in mesh.face_loops(current) {
                let (Some(edge), Some(a)) = (mesh.loop_edge(l), mesh.loop_vert(l)) else {
                    continue;
                };
                let Some(b) = mesh.loop_next(l).and_then(|n| mesh.loop_vert(n)) else {
                    continue;
                };
                for g in mesh.edge_faces(edge) {
                    if g == current || !in_group[g.index() as usize] {
                        continue;
                    }
                    if settled[g.index() as usize] {
                        continue;
                    }
                    settled[g.index() as usize] = true;
                    // Consistent when the neighbour crosses the shared edge the
                    // *other* way. Same direction means one of them is
                    // inside-out.
                    let same_direction = mesh
                        .face_loops(g)
                        .into_iter()
                        .any(|m| mesh.loop_edge(m) == Some(edge) && mesh.loop_vert(m) == Some(a));
                    if same_direction && mesh.reverse_face(g) {
                        out.flipped += 1;
                    }
                    let _ = b;
                    queue.push(g);
                }
            }
        }
    }
}

/// The group's signed volume, and whether that answer is trustworthy.
///
/// Signed volume by the divergence theorem: a closed surface wound outward
/// encloses a positive volume. On an open surface the figure is meaningless —
/// the "volume" depends on where the origin happens to be — so a fallback picks
/// the face furthest from the group's centre and asks whether it faces away from
/// it. That is a heuristic and is reported as one.
fn orientation_of(mesh: &BMesh, group: &[FaceId]) -> (f32, bool) {
    let mut centre = Vec3::ZERO;
    let mut samples = 0.0f32;
    let mut volume = 0.0f32;
    let mut scale = 0.0f32;

    for &f in group {
        let ring: Vec<Vec3> = mesh
            .face_verts(f)
            .iter()
            .filter_map(|v| mesh.position(*v))
            .collect();
        if ring.len() < 3 {
            continue;
        }
        let mid: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
        centre += mid;
        samples += 1.0;
        // Twice the area, times the normal.
        let normal = mesh.face_normal(f);
        volume += mid.dot(normal) / 6.0;
        scale += normal.length();
    }
    if samples == 0.0 {
        return (0.0, true);
    }
    centre /= samples;

    // A closed group encloses a volume comfortably larger than its own surface
    // scale allows for rounding; an open one does not.
    let closed = group
        .iter()
        .flat_map(|f| mesh.face_loops(*f))
        .filter_map(|l| mesh.loop_edge(l))
        .all(|e| mesh.edge_face_count(e) >= 2);
    if closed && volume.abs() > scale * 1e-4 {
        return (volume, false);
    }

    // Open: take the face whose centre is furthest from the group's, and ask
    // whether it faces away. Right for a bowl or a shell, and a guess for
    // anything genuinely ambiguous — which is why the caller is told.
    let mut best = 0.0f32;
    let mut sign = 0.0f32;
    for &f in group {
        let ring: Vec<Vec3> = mesh
            .face_verts(f)
            .iter()
            .filter_map(|v| mesh.position(*v))
            .collect();
        if ring.len() < 3 {
            continue;
        }
        let mid: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
        let away = mid - centre;
        if away.length() > best {
            best = away.length();
            sign = away.dot(mesh.face_normal(f));
        }
    }
    (sign, true)
}

/// Sets `shade_smooth` on the given faces.
pub fn set_shade_smooth(mesh: &mut BMesh, faces: &[FaceId], smooth: bool) -> usize {
    let region = Region::new(mesh, faces);
    if region.is_empty() {
        return 0;
    }
    let len = mesh.slot_count(Domain::Face);
    mesh.attributes_mut()
        .ensure(names::SHADE_SMOOTH, Domain::Face, AttrType::Bool, len);
    // New faces default to `false`, and a mesh with no layer at all renders as
    // smooth, so creating the layer must fill it with what was being shown
    // rather than with the type's default — otherwise marking one face smooth
    // turns every other face flat.
    let existed = mesh
        .attributes()
        .get_bool(names::SHADE_SMOOTH)
        .is_some_and(|s| s.iter().any(|x| *x));
    let slots: Vec<u32> = region.faces.iter().map(|f| f.index()).collect();
    let all: Vec<u32> = mesh.face_ids().map(|f| f.index()).collect();
    if let Some(values) = mesh.attributes_mut().make_mut_bool(names::SHADE_SMOOTH) {
        if !existed {
            for slot in &all {
                values[*slot as usize] = true;
            }
        }
        for slot in &slots {
            values[*slot as usize] = smooth;
        }
    }
    slots.len()
}

/// Marks or clears `sharp_edge` on the given edges.
pub fn set_sharp(mesh: &mut BMesh, edges: &[EdgeId], sharp: bool) -> usize {
    let live: Vec<u32> = edges
        .iter()
        .filter(|e| mesh.edge(**e).is_some())
        .map(|e| e.index())
        .collect();
    if live.is_empty() {
        return 0;
    }
    let len = mesh.slot_count(Domain::Edge);
    mesh.attributes_mut()
        .ensure(names::SHARP_EDGE, Domain::Edge, AttrType::Bool, len);
    if let Some(values) = mesh.attributes_mut().make_mut_bool(names::SHARP_EDGE) {
        for slot in &live {
            values[*slot as usize] = sharp;
        }
    }
    live.len()
}

/// Sets the subdivision crease of `edges`: 0 smooth, 1 sharp at every level,
/// between them sharp for a while — see the Catmull–Clark module.
pub fn set_crease(mesh: &mut BMesh, edges: &[EdgeId], weight: f32) -> usize {
    let live: Vec<u32> = edges
        .iter()
        .filter(|e| mesh.edge(**e).is_some())
        .map(|e| e.index())
        .collect();
    if live.is_empty() {
        return 0;
    }
    let len = mesh.slot_count(Domain::Edge);
    mesh.attributes_mut()
        .ensure(names::CREASE, Domain::Edge, AttrType::F32, len);
    if let Some(values) = mesh.attributes_mut().make_mut_f32(names::CREASE) {
        for slot in &live {
            values[*slot as usize] = weight.clamp(0.0, 1.0);
        }
    }
    live.len()
}

/// Marks every edge whose two faces meet at more than `angle` as sharp.
///
/// The auto-smooth operation, applied once rather than evaluated every frame.
/// Baking it into marks rather than leaving it as a render setting is what makes
/// it exportable: OBJ smoothing groups and FBX layers are marks, and a threshold
/// that only exists in the viewport does not survive a save.
///
/// Boundary and non-manifold edges are marked too: an edge with one face has no
/// second face to blend with, and one with three has no single answer.
pub fn auto_smooth(mesh: &mut BMesh, angle: f32) -> usize {
    let cos_threshold = angle.cos();
    let mut sharp: Vec<EdgeId> = Vec::new();
    let mut smooth: Vec<EdgeId> = Vec::new();
    for e in mesh.edge_ids().collect::<Vec<_>>() {
        let faces = mesh.edge_faces(e);
        let is_sharp = match faces.as_slice() {
            [a, b] => {
                let (na, nb) = (
                    mesh.face_normal(*a).normalize_or_zero(),
                    mesh.face_normal(*b).normalize_or_zero(),
                );
                na.dot(nb) < cos_threshold
            }
            // One face or a fan: nothing to blend with, or no single answer.
            _ => true,
        };
        if is_sharp {
            sharp.push(e);
        } else {
            smooth.push(e);
        }
    }
    // Both directions, so re-running with a wider angle *unmarks* what no longer
    // qualifies. Only marking would make the operation one-way and the angle
    // impossible to adjust downward.
    set_sharp(mesh, &smooth, false);
    set_sharp(mesh, &sharp, true)
}

/// Marks the selected edges sharp, or clears them.
pub fn set_sharp_selected(mesh: &mut BMesh, sharp: bool) -> usize {
    let edges = mesh.selected_edges();
    set_sharp(mesh, &edges, sharp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn cube() -> BMesh {
        BMesh::from_mesh_data(&primitives::cube(2.0))
    }

    /// Whether every face of a closed solid points away from its centre.
    fn all_outward(mesh: &BMesh) -> bool {
        mesh.face_ids().all(|f| {
            let ring: Vec<Vec3> = mesh
                .face_verts(f)
                .iter()
                .filter_map(|v| mesh.position(*v))
                .collect();
            let mid: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
            mesh.face_normal(f).dot(mid) > 0.0
        })
    }

    #[test]
    fn flipping_reverses_the_normal_and_keeps_the_topology() {
        let mut m = cube();
        let f = m.face_ids().next().unwrap();
        let before = m.face_normal(f).normalize_or_zero();
        let ring = m.face_verts(f);

        assert_eq!(flip(&mut m, &[f]), 1);
        m.assert_valid();

        let after = m.face_normal(f).normalize_or_zero();
        assert!(
            after.dot(before) < -0.99,
            "{after:?} should oppose {before:?}"
        );
        assert_eq!(m.num_faces(), 6);
        assert_eq!(m.num_edges(), 12);
        let reversed = m.face_verts(f);
        assert_eq!(reversed.len(), ring.len());
        for v in &ring {
            assert!(reversed.contains(v), "the same corners, in the other order");
        }
    }

    #[test]
    fn flipping_preserves_per_corner_data() {
        // The reason `reverse_face` works in place. Rebuilding the face would be
        // simpler and would clear its UVs, which the user finds when the texture
        // on one flipped face goes blank.
        use crate::attribute::AttrData;
        use kreate_math::Vec2;
        use std::sync::Arc;

        let mut m = cube();
        let corners = m.slot_count(Domain::Corner);
        m.attributes_mut().insert(
            "uv",
            Domain::Corner,
            AttrData::F32x2(Arc::new(
                (0..corners).map(|i| Vec2::new(i as f32, 0.0)).collect(),
            )),
        );
        let f = m.face_ids().next().unwrap();
        let before: Vec<(u32, f32)> = m
            .face_loops(f)
            .iter()
            .map(|l| {
                (
                    l.index(),
                    m.attributes().get_f32x2("uv").unwrap()[l.index() as usize].x,
                )
            })
            .collect();

        flip(&mut m, &[f]);
        m.assert_valid();

        for (slot, value) in before {
            assert_eq!(
                m.attributes().get_f32x2("uv").unwrap()[slot as usize].x,
                value,
                "corner {slot} lost its UV"
            );
        }
    }

    #[test]
    fn flipping_twice_is_the_identity() {
        let mut m = cube();
        let f = m.face_ids().next().unwrap();
        let before = m.face_verts(f);
        flip(&mut m, &[f]);
        flip(&mut m, &[f]);
        m.assert_valid();
        assert_eq!(m.face_verts(f), before);
    }

    #[test]
    fn recalculating_fixes_one_inverted_face() {
        let mut m = cube();
        assert!(all_outward(&m), "the primitive starts correct");
        let f = m.face_ids().next().unwrap();
        flip(&mut m, &[f]);
        assert!(!all_outward(&m));

        let all: Vec<FaceId> = m.face_ids().collect();
        let out = recalculate(&mut m, &all, false);
        m.assert_valid();

        assert_eq!(out.groups, 1);
        assert_eq!(out.ambiguous, 0, "a closed solid is not ambiguous");
        assert!(out.flipped >= 1);
        assert!(all_outward(&m), "every face should point out again");
    }

    #[test]
    fn recalculating_a_fully_inverted_solid_turns_it_back() {
        // Consistent but inside-out: the walk finds nothing to fix, so the
        // decision rests entirely on the signed volume.
        let mut m = cube();
        let all: Vec<FaceId> = m.face_ids().collect();
        flip(&mut m, &all);
        assert!(!all_outward(&m));

        let out = recalculate(&mut m, &all, false);
        m.assert_valid();
        assert_eq!(out.flipped, 6, "all six, in one go");
        assert!(all_outward(&m));
    }

    #[test]
    fn recalculating_inside_points_them_all_in() {
        let mut m = cube();
        let all: Vec<FaceId> = m.face_ids().collect();
        recalculate(&mut m, &all, true);
        m.assert_valid();
        assert!(
            m.face_ids().all(|f| {
                let ring: Vec<Vec3> = m
                    .face_verts(f)
                    .iter()
                    .filter_map(|v| m.position(*v))
                    .collect();
                let mid: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
                m.face_normal(f).dot(mid) < 0.0
            }),
            "every face should now face inward"
        );
    }

    #[test]
    fn an_open_surface_is_made_consistent_and_reported_as_ambiguous() {
        // There is no inside to a flat sheet. Making it *consistent* is exact
        // and worth doing; deciding which way is out is a guess, and a tool that
        // guesses silently leaves the user to notice half their model is lit
        // from behind.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        let faces: Vec<FaceId> = m.face_ids().collect();
        flip(&mut m, &[faces[0]]);

        let out = recalculate(&mut m, &faces, false);
        m.assert_valid();
        assert_eq!(out.ambiguous, 1, "an open surface has no outside");

        // Consistency is what must hold: every pair of neighbours agrees.
        let reference = m.face_normal(faces[0]).normalize_or_zero();
        for f in m.face_ids() {
            let n = m.face_normal(f).normalize_or_zero();
            assert!(
                n.dot(reference) > 0.99,
                "the sheet is still inconsistent: {n:?} against {reference:?}"
            );
        }
    }

    #[test]
    fn two_separate_shells_are_handled_separately() {
        let mut m = cube();
        let second = BMesh::from_mesh_data(&primitives::cube(2.0));
        // Graft a second cube far away by rebuilding it inside `m`.
        let offset = Vec3::new(10.0, 0.0, 0.0);
        let mut mapping = Vec::new();
        for v in second.vert_ids() {
            mapping.push((v, m.add_vert(second.position(v).unwrap() + offset)));
        }
        for f in second.face_ids() {
            let ring: Vec<_> = second
                .face_verts(f)
                .iter()
                .map(|v| mapping.iter().find(|(a, _)| a == v).unwrap().1)
                .collect();
            m.add_face(&ring).unwrap();
        }
        m.assert_valid();

        let all: Vec<FaceId> = m.face_ids().collect();
        // Invert only the far shell.
        let far: Vec<FaceId> = all
            .iter()
            .copied()
            .filter(|f| {
                m.face_verts(*f)
                    .iter()
                    .all(|v| m.position(*v).unwrap().x > 5.0)
            })
            .collect();
        assert_eq!(far.len(), 6);
        flip(&mut m, &far);

        let out = recalculate(&mut m, &all, false);
        m.assert_valid();
        assert_eq!(out.groups, 2, "two shells, two decisions");
        assert_eq!(out.ambiguous, 0);
        for f in m.face_ids() {
            let ring: Vec<Vec3> = m
                .face_verts(f)
                .iter()
                .filter_map(|v| m.position(*v))
                .collect();
            let mid: Vec3 = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
            let centre = if mid.x > 5.0 { offset } else { Vec3::ZERO };
            assert!(
                m.face_normal(f).dot(mid - centre) > 0.0,
                "a face of one shell points inward"
            );
        }
    }

    // --- smoothing ----------------------------------------------------------

    #[test]
    fn marking_one_face_flat_leaves_the_others_smooth() {
        // The layer does not exist until something sets it, and a mesh without
        // it renders as smooth. Creating it must therefore fill it with what was
        // being shown — otherwise marking one face flat turns every other face
        // flat too, which looks like the operator doing far more than it said.
        let mut m = cube();
        assert!(m.attributes().get_bool(names::SHADE_SMOOTH).is_none());
        let f = m.face_ids().next().unwrap();

        assert_eq!(set_shade_smooth(&mut m, &[f], false), 1);
        m.assert_valid();

        let values = m.attributes().get_bool(names::SHADE_SMOOTH).unwrap();
        assert!(!values[f.index() as usize], "the marked face is flat");
        for g in m.face_ids().filter(|g| *g != f) {
            assert!(values[g.index() as usize], "{g:?} was turned flat too");
        }
    }

    #[test]
    fn marking_edges_sharp_and_clearing_them_again() {
        let mut m = cube();
        let edges: Vec<EdgeId> = m.edge_ids().take(3).collect();
        assert_eq!(set_sharp(&mut m, &edges, true), 3);
        let values = m.attributes().get_bool(names::SHARP_EDGE).unwrap();
        assert_eq!(values.iter().filter(|x| **x).count(), 3);

        set_sharp(&mut m, &edges, false);
        let values = m.attributes().get_bool(names::SHARP_EDGE).unwrap();
        assert_eq!(values.iter().filter(|x| **x).count(), 0);
        m.assert_valid();
    }

    #[test]
    fn auto_smooth_marks_a_cubes_edges_and_not_a_spheres() {
        // The operation, stated as what a user expects of it: a cube is all
        // creases and a fine sphere is none.
        let mut cube = cube();
        let marked = auto_smooth(&mut cube, 30f32.to_radians());
        assert_eq!(marked, 12, "every edge of a cube is a right angle");

        let mut sphere = BMesh::from_mesh_data(&primitives::uv_sphere(1.0, 32, 16));
        let interior = sphere
            .edge_ids()
            .filter(|e| sphere.edge_face_count(*e) == 2)
            .count();
        let marked = auto_smooth(&mut sphere, 30f32.to_radians());
        assert!(
            marked < interior / 4,
            "a fine sphere should be nearly all smooth: {marked} of {interior}"
        );
    }

    #[test]
    fn auto_smooth_can_be_widened_again() {
        // One-way marking would make the angle impossible to adjust downward,
        // and the user would have to clear every edge by hand to try a different
        // value.
        let mut m = BMesh::from_mesh_data(&primitives::uv_sphere(1.0, 16, 8));
        let tight = auto_smooth(&mut m, 5f32.to_radians());
        let wide = auto_smooth(&mut m, 80f32.to_radians());
        assert!(
            wide < tight,
            "widening the angle must unmark edges: {tight} then {wide}"
        );
    }

    #[test]
    fn auto_smooth_marks_boundary_and_non_manifold_edges() {
        // An edge with one face has nothing to blend with; one with three has no
        // single answer. Leaving either smooth produces a normal averaged from
        // an arbitrary subset.
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        assert_eq!(auto_smooth(&mut m, 30f32.to_radians()), 4);
        let values = m.attributes().get_bool(names::SHARP_EDGE).unwrap();
        assert!(values.iter().all(|x| *x));
    }

    #[test]
    fn a_sharp_edge_actually_breaks_the_shading() {
        // The end-to-end point of all of this: a mark on an edge must change the
        // normals the renderer gets. Without the smooth-group walk in
        // `compute_corner_normals` this passes the mark along and changes
        // nothing, which reads as the mark not working.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 1));
        // Bend the sheet so its two faces meet at a shallow angle: shallow
        // enough that the auto-smooth threshold alone would keep them smooth.
        let seam: Vec<_> = m
            .vert_ids()
            .filter(|v| m.position(*v).unwrap().x.abs() < 1e-4)
            .collect();
        assert_eq!(seam.len(), 2);
        for v in &seam {
            let p = m.position(*v).unwrap();
            m.set_position(*v, p + Vec3::Z * 0.3);
        }

        let smooth_normals = m.to_mesh_data().compute_corner_normals(60f32.to_radians());
        let shared = m.edge_between(seam[0], seam[1]).unwrap();
        set_sharp(&mut m, &[shared], true);
        let sharp_normals = m.to_mesh_data().compute_corner_normals(60f32.to_radians());

        assert_eq!(smooth_normals.len(), sharp_normals.len());
        let changed = smooth_normals
            .iter()
            .zip(&sharp_normals)
            .filter(|(a, b)| (**a - **b).length() > 1e-4)
            .count();
        assert!(
            changed >= 4,
            "marking the seam sharp changed {changed} corner normals; it should \
             split the four corners along it"
        );
    }

    #[test]
    fn smoothing_marks_survive_the_round_trip_to_storage_form() {
        // They are attribute layers, so this is the domain model working — but
        // it is also what makes OBJ smoothing groups and FBX layers round-trip,
        // so it is worth pinning down here rather than only at the IO boundary.
        let mut m = cube();
        auto_smooth(&mut m, 30f32.to_radians());
        let f = m.face_ids().next().unwrap();
        set_shade_smooth(&mut m, &[f], false);

        let baked = m.to_mesh_data();
        assert_eq!(baked.validate(), Vec::new());
        let again = BMesh::from_mesh_data(&baked);
        assert_eq!(
            again
                .attributes()
                .get_bool(names::SHARP_EDGE)
                .map(|s| s.iter().filter(|x| **x).count()),
            Some(12)
        );
        assert_eq!(
            again
                .attributes()
                .get_bool(names::SHADE_SMOOTH)
                .map(|s| s.iter().filter(|x| !**x).count()),
            Some(1)
        );
    }

    #[test]
    fn nothing_selected_is_a_no_op() {
        let mut m = cube();
        assert_eq!(flip(&mut m, &[]), 0);
        assert_eq!(set_shade_smooth(&mut m, &[], true), 0);
        assert_eq!(set_sharp(&mut m, &[], true), 0);
        assert_eq!(recalculate(&mut m, &[], false), Recalculation::default());
        m.assert_valid();
    }
}
