//! Spin: revolving a profile.
//!
//! Repeated extrude-and-rotate. That is not a shortcut — it is what spin *is*,
//! and building it out of [`extrude_faces`](super::extrude::extrude_faces)
//! rather than beside it means the two cannot disagree about winding, about
//! which vertices are reclaimed, or about what ends up selected. A separate
//! implementation would drift on all three.
//!
//! # Closing the circle
//!
//! A full revolution leaves the last ring sitting exactly on top of the first,
//! as two coincident rings rather than one. Welding them is a *separate*
//! decision — the same distinction merge draws — so this reports that the ends
//! meet and leaves the join to the caller. A spin that silently welds cannot be
//! used to build a spiral, and one that silently does not leaves a seam the user
//! finds later.

use kreate_math::{DQuat, Vec3};

use crate::arena::{ArenaIndex, FaceId, VertId};
use crate::bmesh::{BMesh, BmError};

/// What a spin produced.
#[derive(Clone, Debug, Default)]
pub struct Spin {
    /// The faces at the end of the revolution — the new selection.
    pub faces: Vec<FaceId>,
    /// Every vertex the spin created.
    pub verts: Vec<VertId>,
    /// The rings, in order, **starting with the profile's own**.
    ///
    /// Including the original matters: a full revolution lands on the profile,
    /// not on the first band, so a caller welding the ends has to join the last
    /// ring to the profile. Leaving it out makes the join one step short, and the
    /// seam it leaves is a ring of doubled vertices in the middle of the shape.
    pub rings: Vec<Vec<VertId>>,
    /// True when the last step landed back on the first ring, i.e. the profile
    /// went all the way round.
    pub closed: bool,
}

/// Revolves the given faces about an axis.
///
/// `steps` segments spanning `angle` radians in total. Each step extrudes the
/// current end and rotates the new vertices, so a quarter turn in six steps is
/// six bands of fifteen degrees.
pub fn spin(
    mesh: &mut BMesh,
    faces: &[FaceId],
    centre: Vec3,
    axis: Vec3,
    angle: f32,
    steps: u32,
) -> Result<Spin, BmError> {
    let mut out = Spin::default();
    let steps = steps.max(1);
    let unit = axis.normalize_or_zero();
    if unit == Vec3::ZERO || faces.is_empty() {
        return Ok(out);
    }
    let per_step = angle / steps as f32;

    // Composed in `f64`. A spin of sixty steps applies this rotation sixty
    // times, and in `f32` the accumulated error leaves a full revolution
    // visibly short of closing — the last ring lands a fraction of a degree
    // from the first, which the eye catches before any measurement does.
    let step_rotation = DQuat::from_axis_angle(unit.as_dvec3(), per_step as f64);
    let centre = centre.as_dvec3();

    // The profile's own vertices, captured before anything is extruded: the
    // input faces do not survive the first step, because an extrude replaces the
    // region it lifts.
    let mut origin: Vec<VertId> = faces.iter().flat_map(|f| mesh.face_verts(*f)).collect();
    origin.sort_unstable_by_key(|v| v.index());
    origin.dedup();
    let scale = origin
        .iter()
        .filter_map(|v| mesh.position(*v))
        .map(|p| (p.as_dvec3() - centre).length() as f32)
        .fold(0.0f32, f32::max)
        .max(1e-6);
    out.rings.push(origin);

    let mut current: Vec<FaceId> = faces.to_vec();
    for _ in 0..steps {
        let grown = super::extrude::extrude_faces(mesh, &current, Vec3::ZERO)?;
        if grown.new_faces.is_empty() {
            break;
        }

        // One step, from where the copy landed. An extrude with no offset puts
        // the new ring exactly on the previous one, so a single increment takes
        // it to the next angle — applying the *accumulated* angle instead
        // compounds, because the geometry it is applied to has already turned.
        // Six fifteen-degree steps then finish at 315°, not 90°.
        let ring: Vec<VertId> = grown.new_verts.clone();
        for &v in &ring {
            let Some(p) = mesh.position(v) else { continue };
            let local = p.as_dvec3() - centre;
            mesh.set_position(v, (centre + step_rotation * local).as_vec3());
        }
        out.verts.extend(ring.iter().copied());
        out.rings.push(ring);
        current = grown.new_faces;
    }

    // Did the profile come back to where it started? Asked geometrically rather
    // than by comparing the angle against `TAU`, so two half-turns and a wrapped
    // angle are caught as well. The tolerance is relative to the radius, because
    // a millimetre is a closed circle at one scale and a gap at another.
    if out.rings.len() > 1 {
        let first = out.rings[0].clone();
        let last = out.rings[out.rings.len() - 1].clone();
        out.closed = first.len() == last.len()
            && last.iter().all(|v| {
                let Some(p) = mesh.position(*v) else {
                    return false;
                };
                first.iter().any(|w| {
                    mesh.position(*w)
                        .is_some_and(|q| (q - p).length() < scale * 1e-3)
                })
            });
    }

    out.faces = current;
    mesh.select_none();
    for f in &out.faces {
        mesh.set_face_selected(*f, true);
    }
    mesh.flush_selection(crate::SelectMode::Face);
    Ok(out)
}

/// Revolves the selected faces.
pub fn spin_selected(
    mesh: &mut BMesh,
    centre: Vec3,
    axis: Vec3,
    angle: f32,
    steps: u32,
) -> Result<Spin, BmError> {
    let faces = mesh.selected_faces();
    spin(mesh, &faces, centre, axis, angle, steps)
}

/// Welds a closed spin's two ends together.
///
/// Separate from [`spin`] on purpose: a spin that silently welds cannot build a
/// spiral, and one that silently does not leaves a seam. Doing nothing when the
/// ends do not meet keeps a partial revolution from being pulled shut.
pub fn close_spin(mesh: &mut BMesh, spun: &Spin) -> Result<usize, BmError> {
    if !spun.closed || spun.rings.len() < 2 {
        return Ok(0);
    }
    let (Some(first), Some(last)) = (spun.rings.first(), spun.rings.last()) else {
        return Ok(0);
    };
    let mut groups: Vec<Vec<VertId>> = Vec::new();
    for &v in last {
        let Some(p) = mesh.position(v) else { continue };
        let nearest = first.iter().copied().filter(|w| *w != v).min_by(|a, b| {
            let da = mesh
                .position(*a)
                .map(|q| (q - p).length())
                .unwrap_or(f32::MAX);
            let db = mesh
                .position(*b)
                .map(|q| (q - p).length())
                .unwrap_or(f32::MAX);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(w) = nearest {
            groups.push(vec![v, w]);
        }
    }
    let merged = super::merge::merge_groups(mesh, &groups, super::merge::MergeTarget::Centre)?;
    Ok(merged.removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    /// A single quad standing away from the Z axis, to revolve.
    fn profile() -> (BMesh, FaceId) {
        let mut m = BMesh::new();
        let a = m.add_vert(Vec3::new(2.0, 0.0, -0.5));
        let b = m.add_vert(Vec3::new(3.0, 0.0, -0.5));
        let c = m.add_vert(Vec3::new(3.0, 0.0, 0.5));
        let d = m.add_vert(Vec3::new(2.0, 0.0, 0.5));
        let f = m.add_face(&[a, b, c, d]).unwrap();
        (m, f)
    }

    #[test]
    fn a_quarter_turn_in_six_steps_makes_six_bands() {
        let (mut m, f) = profile();
        let out = spin(
            &mut m,
            &[f],
            Vec3::ZERO,
            Vec3::Z,
            std::f32::consts::FRAC_PI_2,
            6,
        )
        .unwrap();
        m.assert_valid();

        assert_eq!(out.rings.len(), 7, "the profile plus six bands");
        assert_eq!(out.verts.len(), 24, "four vertices per band");
        assert!(!out.closed, "a quarter turn does not close");
        assert_eq!(out.faces.len(), 1, "one face at the end");

        // The end ring is a quarter turn round from the start.
        for v in out.rings.last().unwrap() {
            let p = m.position(*v).unwrap();
            assert!(
                p.x.abs() < 1e-3,
                "a quarter turn about Z should leave X at zero, got {p:?}"
            );
        }
    }

    #[test]
    fn the_profile_keeps_its_distance_from_the_axis() {
        // Rotating each step from the *previous* one accumulates drift and the
        // profile slowly spirals inward. Rotating from the original by the
        // accumulated angle does not, and at sixty steps the difference is
        // visible.
        let (mut m, f) = profile();
        let before: Vec<f32> = m
            .face_verts(f)
            .iter()
            .map(|v| m.position(*v).unwrap().truncate().length())
            .collect();

        let out = spin(&mut m, &[f], Vec3::ZERO, Vec3::Z, std::f32::consts::TAU, 60).unwrap();
        m.assert_valid();

        for ring in out.rings.iter().skip(1) {
            let radii: Vec<f32> = ring
                .iter()
                .map(|v| m.position(*v).unwrap().truncate().length())
                .collect();
            for r in &radii {
                assert!(
                    before.iter().any(|b| (b - r).abs() < 1e-3),
                    "a ring drifted to radius {r}, away from {before:?}"
                );
            }
        }
    }

    #[test]
    fn a_full_turn_reports_that_its_ends_meet() {
        // And does not weld them on its own. A spin that silently welds cannot
        // build a spiral; one that silently does not leaves a seam.
        let (mut m, f) = profile();
        let out = spin(&mut m, &[f], Vec3::ZERO, Vec3::Z, std::f32::consts::TAU, 12).unwrap();
        m.assert_valid();

        assert!(out.closed, "a full revolution's ends coincide");
        let before = m.num_verts();
        let welded = close_spin(&mut m, &out).unwrap();
        m.assert_valid();
        assert_eq!(welded, 4, "one ring's worth of duplicates");
        assert_eq!(m.num_verts(), before - 4);
    }

    #[test]
    fn a_welded_full_turn_is_a_closed_torus() {
        let (mut m, f) = profile();
        let out = spin(&mut m, &[f], Vec3::ZERO, Vec3::Z, std::f32::consts::TAU, 12).unwrap();
        // The original profile face and the end face are the two caps of the
        // tube; removing them and welding leaves a closed torus.
        for g in out.faces.iter().copied().chain(std::iter::once(f)) {
            m.remove_face(g);
        }
        close_spin(&mut m, &out).unwrap();
        m.assert_valid();

        for e in m.edge_ids() {
            assert_eq!(
                m.edge_face_count(e),
                2,
                "a revolved and welded profile should be watertight"
            );
        }
        // Euler characteristic of a torus is zero.
        assert_eq!(
            m.num_verts() as i64 - m.num_edges() as i64 + m.num_faces() as i64,
            0
        );
    }

    #[test]
    fn a_partial_turn_is_not_pulled_shut() {
        let (mut m, f) = profile();
        let out = spin(&mut m, &[f], Vec3::ZERO, Vec3::Z, std::f32::consts::PI, 8).unwrap();
        assert_eq!(out.rings.len(), 9, "the profile plus eight bands");
        assert!(!out.closed);
        let before = m.num_verts();
        assert_eq!(close_spin(&mut m, &out).unwrap(), 0, "nothing to weld");
        assert_eq!(m.num_verts(), before);
        m.assert_valid();
    }

    /// Whether every interior edge's two faces traverse it in opposite
    /// directions, which is what consistent winding means.
    fn consistently_wound(mesh: &BMesh) -> bool {
        mesh.edge_ids().all(|e| {
            let loops = mesh.edge_loops(e);
            if loops.len() != 2 {
                return true;
            }
            mesh.loop_vert(loops[0]) != mesh.loop_vert(loops[1])
        })
    }

    #[test]
    fn a_spun_shell_is_consistently_wound() {
        // Inherited from extrude rather than reimplemented, which is the point
        // of building spin on top of it — but worth asserting, because a spin is
        // where a winding mistake would be repeated sixty times.
        //
        // Consistency is the property, not outwardness. Which way the shell
        // faces depends on the spin direction *relative to the profile's own
        // normal*: revolving a face against its normal turns the walls inside
        // out, exactly as extruding it backwards would. That is a real property
        // of the operation rather than a defect, and the next test pins it down.
        let (mut m, f) = profile();
        spin(&mut m, &[f], Vec3::ZERO, Vec3::Z, std::f32::consts::TAU, 12).unwrap();
        m.assert_valid();
        assert!(consistently_wound(&m), "the shell contradicts itself");
    }

    #[test]
    fn reversing_the_spin_reverses_the_shell() {
        // The direction property, stated. A user who gets an inside-out revolve
        // has spun against the profile's normal, and the fix is the sign of the
        // angle — not a recalculate.
        let make = |angle: f32| {
            let (mut m, f) = profile();
            let out = spin(&mut m, &[f], Vec3::ZERO, Vec3::Z, angle, 8).unwrap();
            // A wall on the bottom edge of the profile, at the first band.
            let wall = m
                .face_ids()
                .find(|g| {
                    let ring = m.face_verts(*g);
                    ring.len() == 4
                        && ring
                            .iter()
                            .all(|v| (m.position(*v).unwrap().z + 0.5).abs() < 1e-4)
                })
                .expect("a wall along the profile's bottom edge");
            let n = m.face_normal(wall).normalize_or_zero();
            let _ = out;
            n
        };
        let forward = make(std::f32::consts::FRAC_PI_2);
        let backward = make(-std::f32::consts::FRAC_PI_2);
        assert!(
            forward.dot(backward) < -0.9,
            "reversing the angle should reverse the wall: {forward:?} against {backward:?}"
        );
    }

    #[test]
    fn a_degenerate_axis_or_empty_selection_is_a_no_op() {
        let (mut m, f) = profile();
        let before = (m.num_verts(), m.num_edges(), m.num_faces());
        assert!(spin(&mut m, &[f], Vec3::ZERO, Vec3::ZERO, 1.0, 4)
            .unwrap()
            .verts
            .is_empty());
        assert!(spin(&mut m, &[], Vec3::ZERO, Vec3::Z, 1.0, 4)
            .unwrap()
            .verts
            .is_empty());
        assert_eq!((m.num_verts(), m.num_edges(), m.num_faces()), before);
        m.assert_valid();
    }

    #[test]
    fn zero_steps_is_treated_as_one_rather_than_as_nothing() {
        // A parameter of zero arrives from a script and from a spinner the user
        // dragged past its minimum. One band is the nearest sensible answer;
        // doing nothing looks like the tool is broken.
        let (mut m, f) = profile();
        let out = spin(&mut m, &[f], Vec3::ZERO, Vec3::Z, 0.5, 0).unwrap();
        assert_eq!(out.rings.len(), 2, "the profile plus one band");
        m.assert_valid();
    }

    #[test]
    fn a_spun_mesh_still_bakes_to_storage_form() {
        let mut m = BMesh::from_mesh_data(&primitives::plane(1.0));
        let f = m.face_ids().next().unwrap();
        spin(&mut m, &[f], Vec3::new(0.0, -3.0, 0.0), Vec3::Y, 1.5, 5).unwrap();
        m.assert_valid();
        assert_eq!(m.to_mesh_data().validate(), Vec::new());
    }
}
