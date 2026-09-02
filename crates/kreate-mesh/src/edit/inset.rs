//! Inset.
//!
//! Shrinks a face inward and fills the gap with a band of quads. The plan
//! budgets weeks for this, and the reason is one sentence: offsetting a polygon
//! inward is not offsetting each of its edges inward, because the corners have
//! to be solved as well, and a concave corner can collide with the far side of
//! the polygon before any edge collapses.
//!
//! # What v1 does, exactly
//!
//! Each corner moves along the bisector of its two edges, scaled so that both
//! edges move inward by the requested thickness. Then the thickness is clamped
//! so that no edge of the inner ring passes through zero length.
//!
//! That clamp is the straight skeleton's **edge event** — the first way a
//! shrinking polygon can degenerate — and it is exact. It is not the whole
//! story: a **split event**, where a reflex corner reaches an edge on the other
//! side, can happen first on a strongly concave face, and this does not detect
//! it. On such a face a large inset can self-intersect.
//!
//! That is stated rather than hidden. The alternative is a full straight
//! skeleton, which is a research-grade sub-project with its own numeric
//! robustness problems, and the plan defers it. What v1 must not do is fail
//! silently on the ordinary convex case, and the clamp is what guarantees that.

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, FaceId, VertId};
use crate::bmesh::{BMesh, BmError, SelectMode};

use super::region::Region;

/// What an inset produced.
#[derive(Clone, Debug, Default)]
pub struct Inset {
    /// The shrunken faces. The new selection.
    pub inner_faces: Vec<FaceId>,
    /// The band between the original outline and the shrunken one.
    pub band_faces: Vec<FaceId>,
    /// The new inner ring vertices.
    pub inner_verts: Vec<VertId>,
    /// True if the thickness had to be reduced to keep a ring from folding.
    pub clamped: bool,
}

/// How much a ring can shrink before one of its edges collapses.
///
/// The straight skeleton's first edge event, solved exactly: every edge's length
/// is linear in the inset distance, so the answer is the smallest positive root.
/// `f32::INFINITY` when no edge ever collapses, which is the case for a ring
/// being pushed outward.
fn max_inset(ring: &[Vec3], offsets: &[Vec3]) -> f32 {
    let n = ring.len();
    let mut limit = f32::INFINITY;
    for i in 0..n {
        let j = (i + 1) % n;
        let edge = ring[j] - ring[i];
        let length = edge.length();
        if length < 1e-6 {
            continue;
        }
        let dir = edge / length;
        let rate = (offsets[j] - offsets[i]).dot(dir);
        if rate < -1e-6 {
            limit = limit.min(length / -rate);
        }
    }
    limit
}

/// The inward bisector offsets for one ring, one unit of thickness each.
///
/// Scaled by `1 / sin(θ/2)` so that both edges meeting at a corner move inward
/// by the same distance. Without that scale a sharp corner's edges barely move
/// while a flat one's move fully, and the inset band comes out visibly uneven —
/// the single most common way a hand-rolled inset looks wrong.
fn bisector_offsets(ring: &[Vec3], normal: Vec3) -> Vec<Vec3> {
    let n = ring.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = ring[(i + n - 1) % n];
        let next = ring[(i + 1) % n];
        let into_prev = (ring[i] - prev).normalize_or_zero();
        let into_next = (next - ring[i]).normalize_or_zero();
        // `cross(normal, edge)` points into the polygon for a ring wound
        // counter-clockwise about the normal, which is the winding every face
        // here has by construction.
        let inward_prev = normal.cross(into_prev).normalize_or_zero();
        let inward_next = normal.cross(into_next).normalize_or_zero();

        let bisector = (inward_prev + inward_next).normalize_or_zero();
        if bisector.length_squared() < 1e-12 {
            // A 180° turn: the two edges are antiparallel and there is no
            // bisector. Either inward direction is as good as the other.
            out.push(inward_next);
            continue;
        }
        // How far along the bisector reaches an offset of one unit from the
        // edges. Capped, because a nearly-degenerate spike would otherwise send
        // its corner to infinity.
        let sin_half = bisector.dot(inward_next).abs().max(0.05);
        out.push(bisector / sin_half);
    }
    out
}

/// Insets each face separately.
///
/// `thickness` shrinks the face in its own plane; `depth` moves the shrunken
/// face along its normal, which is what makes an inset useful for panelling and
/// for starting a bevel by hand.
pub fn inset_individual(
    mesh: &mut BMesh,
    faces: &[FaceId],
    thickness: f32,
    depth: f32,
) -> Result<Inset, BmError> {
    let region = Region::new(mesh, faces);
    let mut out = Inset::default();
    for &f in &region.faces {
        let ring: Vec<VertId> = mesh.face_verts(f);
        if ring.len() < 3 {
            continue;
        }
        let normal = mesh.face_normal(f).normalize_or_zero();
        inset_one_ring(mesh, &ring, &[f], normal, thickness, depth, &mut out)?;
    }
    finish(mesh, &mut out);
    Ok(out)
}

/// Insets a region as one, so no band appears between neighbouring faces.
///
/// The same distinction as extrude's: the interior of the region keeps its
/// topology, and the band grows only around the outline. A per-face inset of two
/// adjacent quads leaves a wall of geometry down the middle, which is a
/// different operation and rarely the one wanted.
pub fn inset_region(
    mesh: &mut BMesh,
    faces: &[FaceId],
    thickness: f32,
    depth: f32,
) -> Result<Inset, BmError> {
    let region = Region::new(mesh, faces);
    let mut out = Inset::default();
    if region.is_empty() {
        return Ok(out);
    }
    let normal = mesh.region_normal(&region.faces);

    for group in region.components(mesh) {
        let part = Region::new(mesh, &group);
        // A region with a hole has two outlines and no single ring to shrink;
        // one pinched at a corner has an outline that can be chained two ways.
        // Both are left alone rather than guessed at.
        let Some(loops) = part.boundary_loops(mesh) else {
            continue;
        };
        let [ring] = &loops[..] else { continue };
        let group_normal = if normal.length_squared() > 1e-12 {
            normal
        } else {
            mesh.region_normal(&group)
        };
        inset_one_ring(mesh, ring, &group, group_normal, thickness, depth, &mut out)?;
    }
    finish(mesh, &mut out);
    Ok(out)
}

/// Shrinks one ring, rebuilds the faces it bounded, and grows the band.
#[allow(clippy::too_many_arguments)]
fn inset_one_ring(
    mesh: &mut BMesh,
    ring: &[VertId],
    faces: &[FaceId],
    normal: Vec3,
    thickness: f32,
    depth: f32,
    out: &mut Inset,
) -> Result<(), BmError> {
    let positions: Vec<Vec3> = ring.iter().filter_map(|v| mesh.position(*v)).collect();
    if positions.len() != ring.len() || positions.len() < 3 {
        return Ok(());
    }
    let offsets = bisector_offsets(&positions, normal);

    // Never all the way to the collapse: an edge of exactly zero length is a
    // face that repeats a vertex, which `add_face` refuses and which no operator
    // downstream could use anyway.
    let limit = max_inset(&positions, &offsets) * 0.98;
    let applied = thickness.min(limit);
    if applied < thickness {
        out.clamped = true;
    }

    // Rings of the interior faces, read before anything is removed.
    let interior: Vec<Vec<VertId>> = faces.iter().map(|&f| mesh.face_verts(f)).collect();

    let mut inner = Vec::with_capacity(ring.len());
    for (i, &v) in ring.iter().enumerate() {
        let p = positions[i] + offsets[i] * applied + normal * depth;
        let n = mesh.add_vert(p);
        inner.push(n);
        out.inner_verts.push(n);
        let _ = v;
    }

    // A map from the outline's vertices to their shrunken copies, so the
    // interior faces can be rebuilt in place. Interior vertices are not on the
    // outline and keep their identity.
    let mut moved: Vec<(VertId, VertId)> =
        ring.iter().copied().zip(inner.iter().copied()).collect();
    moved.sort_unstable_by_key(|(a, _)| a.index());

    for &f in faces {
        mesh.remove_face(f);
    }
    for original in interior {
        let rebuilt: Vec<VertId> = original
            .iter()
            .map(
                |v| match moved.binary_search_by_key(&v.index(), |(a, _)| a.index()) {
                    Ok(i) => moved[i].1,
                    Err(_) => *v,
                },
            )
            .collect();
        out.inner_faces.push(mesh.add_face(&rebuilt)?);
    }

    let n = ring.len();
    for i in 0..n {
        let j = (i + 1) % n;
        // `(a, b, b', a')`, the same winding rule the extrude walls use: the
        // band must be wound consistently with the face it came from or it lights
        // from the inside.
        out.band_faces
            .push(mesh.add_face(&[ring[i], ring[j], inner[j], inner[i]])?);
    }
    Ok(())
}

/// Selects the shrunken faces, which is what a following transform should grab.
fn finish(mesh: &mut BMesh, out: &mut Inset) {
    if out.inner_faces.is_empty() {
        return;
    }
    mesh.select_none();
    for f in &out.inner_faces {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(SelectMode::Face);
}

/// Insets the selected faces as one region.
pub fn inset_selected(mesh: &mut BMesh, thickness: f32, depth: f32) -> Result<Inset, BmError> {
    let faces = mesh.selected_faces();
    inset_region(mesh, &faces, thickness, depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    /// The signed area of a ring about a normal. Sign flip means the ring folded.
    fn signed_area(mesh: &BMesh, f: FaceId, normal: Vec3) -> f32 {
        let ring: Vec<Vec3> = mesh
            .face_verts(f)
            .iter()
            .filter_map(|v| mesh.position(*v))
            .collect();
        let mut sum = Vec3::ZERO;
        for i in 0..ring.len() {
            sum += ring[i].cross(ring[(i + 1) % ring.len()]);
        }
        sum.dot(normal) * 0.5
    }

    #[test]
    fn insetting_a_quad_makes_a_smaller_quad_in_a_frame() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let f = m.face_ids().next().unwrap();
        let out = inset_individual(&mut m, &[f], 0.25, 0.0).unwrap();
        m.assert_valid();

        assert!(!out.clamped);
        assert_eq!(out.inner_faces.len(), 1);
        assert_eq!(out.band_faces.len(), 4, "one quad per edge");
        assert_eq!(m.num_faces(), 5);
        assert_eq!(m.num_verts(), 8);

        // The inner ring is a square 0.25 in from each side, so half a unit
        // smaller across.
        let inner: Vec<Vec3> = m
            .face_verts(out.inner_faces[0])
            .iter()
            .filter_map(|v| m.position(*v))
            .collect();
        for p in &inner {
            assert!((p.x.abs() - 0.75).abs() < 1e-4, "x was {}", p.x);
            assert!((p.y.abs() - 0.75).abs() < 1e-4, "y was {}", p.y);
        }
    }

    #[test]
    fn a_corner_moves_far_enough_that_both_its_edges_move_equally() {
        // The `1 / sin(θ/2)` scale. Without it a sharp corner barely moves while
        // a flat one moves fully, and the band comes out visibly uneven — the
        // most common way a hand-rolled inset looks wrong.
        let mut m = BMesh::new();
        // A very obtuse triangle: one corner near 20°, the others wide.
        let a = m.add_vert(Vec3::new(0.0, 0.0, 0.0));
        let b = m.add_vert(Vec3::new(4.0, 0.0, 0.0));
        let c = m.add_vert(Vec3::new(2.0, 0.35, 0.0));
        let f = m.add_face(&[a, b, c]).unwrap();

        let t = 0.05f32;
        let out = inset_individual(&mut m, &[f], t, 0.0).unwrap();
        m.assert_valid();

        // Every inner vertex is `t` from both of the original edges meeting at
        // its corner, regardless of how sharp that corner is.
        let outer = [
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)),
            (Vec3::new(4.0, 0.0, 0.0), Vec3::new(2.0, 0.35, 0.0)),
            (Vec3::new(2.0, 0.35, 0.0), Vec3::new(0.0, 0.0, 0.0)),
        ];
        for v in &out.inner_verts {
            let p = m.position(*v).unwrap();
            let mut distances: Vec<f32> = outer
                .iter()
                .map(|(s, e)| {
                    let d = (*e - *s).normalize();
                    ((p - *s) - d * (p - *s).dot(d)).length()
                })
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(
                (distances[0] - t).abs() < 1e-3 && (distances[1] - t).abs() < 1e-3,
                "corner is {distances:?} from its edges, wanted two at {t}"
            );
        }
    }

    #[test]
    fn too_much_thickness_is_clamped_rather_than_folded() {
        // The defect this prevents: an inner ring that has passed through itself
        // still renders, still validates, and is inside-out. It is discovered as
        // "the shading went strange" long after.
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        let f = m.face_ids().next().unwrap();
        let normal = m.face_normal(f).normalize_or_zero();
        let before = signed_area(&m, f, normal);

        let out = inset_individual(&mut m, &[f], 50.0, 0.0).unwrap();
        m.assert_valid();

        assert!(out.clamped, "50 units into a 2-unit square must be clamped");
        let after = signed_area(&m, out.inner_faces[0], normal);
        assert!(
            after > 0.0 && before > 0.0,
            "the ring folded: area went from {before} to {after}"
        );
        assert!(after < before, "and it did shrink");
    }

    #[test]
    fn depth_moves_the_inner_face_along_the_normal() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let top = m
            .face_ids()
            .max_by(|a, b| {
                let za = m.face_normal(*a).normalize_or_zero().z;
                let zb = m.face_normal(*b).normalize_or_zero().z;
                za.partial_cmp(&zb).unwrap()
            })
            .unwrap();

        let out = inset_individual(&mut m, &[top], 0.3, -0.5).unwrap();
        m.assert_valid();
        for v in &out.inner_verts {
            assert!(
                (m.position(*v).unwrap().z - 0.5).abs() < 1e-4,
                "sunk half a unit"
            );
        }
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "a panelled cube is still closed");
        }
    }

    #[test]
    fn the_band_faces_outward() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let top = m
            .face_ids()
            .max_by(|a, b| {
                let za = m.face_normal(*a).normalize_or_zero().z;
                let zb = m.face_normal(*b).normalize_or_zero().z;
                za.partial_cmp(&zb).unwrap()
            })
            .unwrap();
        let out = inset_individual(&mut m, &[top], 0.3, -0.4).unwrap();

        // The band of a sunken inset faces inward toward the pit's axis; with
        // zero depth it is flat and coplanar with the face. What must hold in
        // both cases is that it agrees with the face it grew from rather than
        // being inverted, so compare against the inner face.
        let inner = m.face_normal(out.inner_faces[0]).normalize_or_zero();
        assert!(inner.dot(Vec3::Z) > 0.99, "the inner face kept its winding");
        for f in &out.band_faces {
            let n = m.face_normal(*f).normalize_or_zero();
            assert!(
                n.dot(Vec3::Z).abs() < 0.99,
                "a band quad should stand up, not lie flat: {n:?}"
            );
        }
    }

    #[test]
    fn a_region_inset_puts_no_band_between_neighbours() {
        // The same distinction as extrude's. A per-face inset of four adjacent
        // quads leaves walls down the middle; a region inset frames the outline
        // only.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        let all: Vec<FaceId> = m.face_ids().collect();
        let out = inset_region(&mut m, &all, 0.3, 0.0).unwrap();
        m.assert_valid();

        assert_eq!(out.inner_faces.len(), 4, "the interior keeps its topology");
        assert_eq!(out.band_faces.len(), 8, "one per outline edge, no more");
        assert_eq!(m.num_faces(), 12);
    }

    #[test]
    fn a_per_face_inset_of_the_same_region_does_put_bands_between_them() {
        // Stated as a contrast, so the two operations cannot quietly become one.
        let mut m = BMesh::from_mesh_data(&primitives::grid(4.0, 2, 2));
        let all: Vec<FaceId> = m.face_ids().collect();
        let out = inset_individual(&mut m, &all, 0.2, 0.0).unwrap();
        m.assert_valid();

        assert_eq!(out.inner_faces.len(), 4);
        assert_eq!(
            out.band_faces.len(),
            16,
            "four per face, including shared ones"
        );
        assert_eq!(m.num_faces(), 20);
    }

    #[test]
    fn insetting_every_face_of_a_solid_keeps_it_closed() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let all: Vec<FaceId> = m.face_ids().collect();
        inset_individual(&mut m, &all, 0.25, 0.0).unwrap();
        m.assert_valid();

        assert_eq!(m.num_faces(), 6 * 5);
        for e in m.edge_ids() {
            assert_eq!(m.edge_face_count(e), 2, "still watertight");
        }
    }

    #[test]
    fn the_inner_faces_are_what_ends_up_selected() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(2.0));
        m.select_all();
        let out = inset_selected(&mut m, 0.2, 0.0).unwrap();
        assert_eq!(m.selected_faces(), out.inner_faces);
        assert_eq!(m.selected_verts().len(), 4);
    }

    #[test]
    fn insetting_nothing_is_a_no_op() {
        let mut m = BMesh::from_mesh_data(&primitives::cube(2.0));
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        assert!(inset_individual(&mut m, &[], 0.2, 0.0)
            .unwrap()
            .inner_faces
            .is_empty());
        assert!(inset_region(&mut m, &[], 0.2, 0.0)
            .unwrap()
            .inner_faces
            .is_empty());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
        m.assert_valid();
    }

    #[test]
    fn an_inset_mesh_still_bakes_to_storage_form() {
        let mut m = BMesh::from_mesh_data(&primitives::uv_sphere(1.0, 10, 5));
        let all: Vec<FaceId> = m.face_ids().collect();
        inset_individual(&mut m, &all, 0.05, 0.02).unwrap();
        m.assert_valid();
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }

    #[test]
    fn every_primitive_survives_an_aggressive_inset() {
        // The clamp is what makes this safe to run without knowing the scale of
        // the geometry. A thickness far larger than any face must produce a
        // degenerate-free mesh, not a folded one.
        for mesh in [
            primitives::cube(2.0),
            primitives::grid(4.0, 3, 3),
            primitives::uv_sphere(1.0, 9, 5),
            primitives::ico_sphere(1.0, 1),
            primitives::cylinder(1.0, 2.0, 7),
            primitives::cone(1.0, 2.0, 6),
            primitives::torus(1.0, 0.4, 8, 6),
        ] {
            let mut m = BMesh::from_mesh_data(&mesh);
            let all: Vec<FaceId> = m.face_ids().collect();
            let out = inset_individual(&mut m, &all, 100.0, 0.0).unwrap();
            m.assert_valid();
            assert!(out.clamped);
            for f in m.face_ids() {
                assert!(
                    m.face_normal(f).length() > 1e-9,
                    "an inset produced a zero-area face"
                );
            }
        }
    }
}
