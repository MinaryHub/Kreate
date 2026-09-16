//! Projection unwrapping: the UV a modeller can get without solving anything.
//!
//! Four projections — planar, box, spherical, cylindrical — each a closed-form
//! function from a position to a point on the map. That is deliberately the
//! whole of it: a real unwrap minimises distortion over a mesh, which is an
//! optimisation problem (LSCM, ABF) and a Phase 7 subject. Projection covers
//! the cases where the artist already knows the shape — a wall, a crate, a
//! planet, a pipe — and covers them exactly rather than approximately.
//!
//! Two things make the difference between this and a naive
//! `uv = position.xy`:
//!
//! **The seam.** A spherical or cylindrical projection wraps: longitude runs
//! 0 to 1 and then starts again. A face straddling that line gets one corner
//! at 0.99 and the next at 0.01, and unless something intervenes it is drawn
//! stretched across the entire map — a bright wrong band that shows up in
//! every render. Each face is therefore unwrapped *as a face*: corners are
//! pulled into the same period as the first one, so a straddling face reads
//! 0.99 to 1.01 and stays a face. See [`unwrap_period`].
//!
//! **Per-corner, not per-vertex.** UVs live in the Corner domain, so a vertex
//! shared by faces on opposite sides of a seam holds a different UV in each —
//! which is exactly the representation a seam needs, and why the domain model
//! was chosen in the first place.
//!
//! Poles are the honest weakness: a spherical projection's top and bottom
//! collapse to a point, so triangles touching a pole are degenerate in UV
//! space. Nothing here can fix that — it is a property of the sphere map, not
//! of this code — and a real unwrap is the answer when it matters.

use glam::{Vec2, Vec3};

use crate::arena::{ArenaIndex, FaceId, LoopId};
use crate::attribute::{names, AttrType, Domain};
use crate::bmesh::BMesh;

/// Where a projection puts a point on the map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    /// Flat, along a given basis. The two axes span the map; anything along
    /// their cross product is discarded.
    ///
    /// This is also "project from view": the caller passes the camera's right
    /// and up vectors and gets what the screen sees. Passing the basis rather
    /// than a camera is what keeps the operator pure — no view state reaches
    /// the mesh kernel, and a headless test can do the same projection without
    /// a window.
    Planar { right: Vec3, up: Vec3 },
    /// Six planar projections, one per signed axis, chosen per face by which
    /// axis its normal points most nearly along.
    ///
    /// The classic crate unwrap. Each face lands on exactly one of the six
    /// sides, so a box comes out perfect and anything boxy comes out usable.
    Box,
    /// Longitude and latitude about an axis, from a centre.
    Sphere { axis: Vec3, center: Vec3 },
    /// Longitude and height about an axis, from a centre.
    Cylinder { axis: Vec3, center: Vec3 },
}

/// How the result is fitted into the unit square.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Fit {
    /// Scale and centre the projected region into `0..1`, preserving aspect.
    ///
    /// The default because it is what makes the result *usable* rather than
    /// merely correct: raw world coordinates put a two-metre wall miles off
    /// the map, and the artist's first act would be to scale it back.
    #[default]
    Bounds,
    /// Leave the projected coordinates as they are.
    ///
    /// For a projection whose output is already normalised by construction —
    /// spherical longitude, or a caller doing its own packing — and for
    /// projecting two objects into one shared space, where fitting each to its
    /// own bounds would destroy the relationship.
    None,
}

/// Which layer the UVs are written to.
pub const DEFAULT_LAYER: &str = names::UV;

/// Projects `faces` into the named Corner layer, returning how many corners
/// were written.
///
/// Faces outside `faces` keep whatever they had: the layer is created if
/// absent but never cleared, so projecting a selection is a local edit rather
/// than a whole-mesh replacement. That is what lets a character be unwrapped
/// in parts.
pub fn project(mesh: &mut BMesh, faces: &[FaceId], how: Projection, layer: &str) -> usize {
    project_with(mesh, faces, how, layer, Fit::default())
}

/// [`project`], with control over the fit.
pub fn project_with(
    mesh: &mut BMesh,
    faces: &[FaceId],
    how: Projection,
    layer: &str,
    fit: Fit,
) -> usize {
    // Gathered before touching the store, because computing a UV needs the
    // geometry and holding a mutable borrow of the attributes at the same time
    // would not compile — and because the fit needs every value before any of
    // them can be written.
    let mut written: Vec<(LoopId, Vec2)> = Vec::new();
    for f in faces {
        let loops = mesh.face_loops(*f);
        if loops.is_empty() {
            continue;
        }
        let basis = face_basis(mesh, *f, how);
        let mut face_uvs: Vec<Vec2> = Vec::with_capacity(loops.len());
        // How far each corner is from the projection axis. Only the wrapping
        // projections need it, and they need it because a corner *on* the axis
        // has no longitude at all — see `unwrap_period`.
        let mut radii: Vec<f32> = Vec::with_capacity(loops.len());
        for l in &loops {
            let Some(v) = mesh.loop_vert(*l) else {
                continue;
            };
            let Some(p) = mesh.position(v) else {
                continue;
            };
            face_uvs.push(project_point(p, how, basis));
            radii.push(axis_radius(p, how));
        }
        if face_uvs.len() != loops.len() {
            // A loop with no vertex is a broken cycle; the validator's problem,
            // not this function's. Skipping the face is the one safe answer:
            // writing a partial face would leave UVs that look right in the
            // count and wrong on screen.
            continue;
        }
        if wraps(how) {
            unwrap_period(&mut face_uvs, &radii);
        }
        written.extend(loops.iter().copied().zip(face_uvs));
    }

    if written.is_empty() {
        return 0;
    }
    if fit == Fit::Bounds {
        fit_to_unit_square(&mut written);
    }

    let len = mesh.slot_count(Domain::Corner);
    mesh.attributes_mut()
        .ensure(layer, Domain::Corner, AttrType::F32x2, len);
    let Some(values) = mesh.attributes_mut().make_mut_f32x2(layer) else {
        return 0;
    };
    let mut count = 0;
    for (l, uv) in &written {
        let slot = l.index() as usize;
        if slot < values.len() {
            values[slot] = *uv;
            count += 1;
        }
    }
    count
}

/// The per-face choice a projection makes, if it makes one.
///
/// Only [`Projection::Box`] does: it picks one of six planar bases from the
/// face normal. Computing it once per face rather than once per corner is not
/// only cheaper, it is *required* — a per-corner choice could put two corners
/// of one face on different sides of the box.
#[derive(Clone, Copy)]
struct Basis {
    right: Vec3,
    up: Vec3,
}

fn face_basis(mesh: &BMesh, f: FaceId, how: Projection) -> Basis {
    match how {
        Projection::Planar { right, up } => Basis { right, up },
        Projection::Box => box_basis(mesh.face_normal(f)),
        // The wrapping projections need no basis; their axis is fixed.
        Projection::Sphere { .. } | Projection::Cylinder { .. } => Basis {
            right: Vec3::X,
            up: Vec3::Y,
        },
    }
}

/// The side of the box a normal faces, as a planar basis.
///
/// Signed, so the two faces of a thin plate land on opposite sides of the map
/// rather than on top of each other mirrored — which reads as a texture
/// printed backwards on one side.
fn box_basis(normal: Vec3) -> Basis {
    // `right x up == normal`, on every side. That is the condition, and it is
    // not a detail: get it wrong and the face's winding in UV space flips, so
    // the texture on that side of the box reads mirrored. Four of the six were
    // wrong when this was first written and the winding test is what said so.
    let a = normal.abs();
    if a.x >= a.y && a.x >= a.z {
        if normal.x >= 0.0 {
            Basis {
                right: Vec3::Y,
                up: Vec3::Z,
            }
        } else {
            Basis {
                right: -Vec3::Y,
                up: Vec3::Z,
            }
        }
    } else if a.y >= a.z {
        if normal.y >= 0.0 {
            Basis {
                right: -Vec3::X,
                up: Vec3::Z,
            }
        } else {
            Basis {
                right: Vec3::X,
                up: Vec3::Z,
            }
        }
    } else if normal.z >= 0.0 {
        Basis {
            right: Vec3::X,
            up: Vec3::Y,
        }
    } else {
        Basis {
            right: Vec3::X,
            up: -Vec3::Y,
        }
    }
}

fn project_point(p: Vec3, how: Projection, basis: Basis) -> Vec2 {
    match how {
        Projection::Planar { .. } | Projection::Box => {
            Vec2::new(p.dot(basis.right), p.dot(basis.up))
        }
        Projection::Sphere { axis, center } => {
            let (u, v, w) = frame(axis);
            let d = p - center;
            let (x, y, z) = (d.dot(u), d.dot(v), d.dot(w));
            let longitude = angle_fraction(x, y);
            // From the axis, not from the centre: `atan2` of the height
            // against the radius gives latitude that is stable at the equator
            // and saturates at the poles, where `acos` of a normalised vector
            // loses precision.
            let radius = (x * x + y * y).sqrt();
            let latitude = z.atan2(radius) / std::f32::consts::PI + 0.5;
            Vec2::new(longitude, latitude)
        }
        Projection::Cylinder { axis, center } => {
            let (u, v, w) = frame(axis);
            let d = p - center;
            Vec2::new(angle_fraction(d.dot(u), d.dot(v)), d.dot(w))
        }
    }
}

/// A right-handed frame whose third vector is `axis`.
///
/// The reference vector is chosen away from the axis, because a cross product
/// with a nearly parallel vector loses all its precision — and an axis of
/// exactly `Z` is the common case, not the exception.
fn frame(axis: Vec3) -> (Vec3, Vec3, Vec3) {
    let w = axis.normalize_or_zero();
    let w = if w == Vec3::ZERO { Vec3::Z } else { w };
    let reference = if w.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let u = reference.cross(w).normalize();
    let v = w.cross(u);
    (u, v, w)
}

/// The angle of `(x, y)` as a fraction of a full turn, in `0..1`.
fn angle_fraction(x: f32, y: f32) -> f32 {
    let a = y.atan2(x) / std::f32::consts::TAU;
    if a < 0.0 {
        a + 1.0
    } else {
        a
    }
}

fn wraps(how: Projection) -> bool {
    matches!(how, Projection::Sphere { .. } | Projection::Cylinder { .. })
}

/// Pulls a face's longitudes into one period.
///
/// The seam fix. Longitude is periodic with period 1, so `0.99` and `0.01`
/// name points a hundredth of a turn apart — but as numbers they are almost a
/// whole turn apart, and a face holding both is drawn across the entire map.
/// Shifting each corner into whichever period puts it nearest the anchor turns
/// `0.01` into `1.01`, and the face is a face again. UVs outside `0..1` are
/// ordinary and correct: the texture repeats.
///
/// **The anchor is the corner furthest from the axis, not the first one.** A
/// corner *on* the axis — a sphere's pole, the apex of a cone — has no
/// longitude: `atan2(0, 0)` is zero, which is a number rather than an answer.
/// Anchoring a pole triangle on its pole shifts the two corners that do have
/// longitudes to the far side of the map, which is the same visible defect the
/// seam fix exists to prevent, arriving by a different route. The corner
/// furthest out has the best-conditioned longitude there is on the face.
///
/// Degenerate corners then take the anchor's longitude outright. That leaves a
/// pole triangle a sliver in `u`, which is what a sphere map does at a pole
/// and not something this function can improve on.
fn unwrap_period(uvs: &mut [Vec2], radii: &[f32]) {
    debug_assert_eq!(uvs.len(), radii.len());
    let Some((anchor, largest)) = radii
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return;
    };
    if largest.is_nan() || largest <= 0.0 {
        // Every corner is on the axis. Nothing here has a longitude and there
        // is nothing to anchor to.
        return;
    }
    let reference = uvs[anchor].x;
    // Relative to the face's own size, so it holds for a metre-wide sphere and
    // a millimetre-wide one alike.
    let degenerate = largest * 1e-4;
    for (i, uv) in uvs.iter_mut().enumerate() {
        if i == anchor {
            continue;
        }
        if radii[i] <= degenerate {
            uv.x = reference;
        } else {
            uv.x -= (uv.x - reference).round();
        }
    }
}

/// How far a point is from a wrapping projection's axis.
///
/// Zero for the projections that do not wrap: they have no axis, and the value
/// is never read for them.
fn axis_radius(p: Vec3, how: Projection) -> f32 {
    match how {
        Projection::Sphere { axis, center } | Projection::Cylinder { axis, center } => {
            let (u, v, _) = frame(axis);
            let d = p - center;
            let (x, y) = (d.dot(u), d.dot(v));
            (x * x + y * y).sqrt()
        }
        Projection::Planar { .. } | Projection::Box => 0.0,
    }
}

/// Scales and centres the projected region into the unit square.
///
/// Aspect-preserving: a two-by-one wall comes out two-by-one, using half the
/// map's height. Stretching it to fill would put a square texture on it
/// squashed, which is exactly the thing an artist notices first.
fn fit_to_unit_square(written: &mut [(LoopId, Vec2)]) {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for (_, uv) in written.iter() {
        min = min.min(*uv);
        max = max.max(*uv);
    }
    let extent = max - min;
    let span = extent.x.max(extent.y);
    if !span.is_finite() || span <= 0.0 {
        // Every corner projected to one point: a face seen exactly edge-on, or
        // a degenerate selection. Any scale is arbitrary, so leave it centred
        // rather than dividing by zero.
        for (_, uv) in written.iter_mut() {
            *uv = Vec2::splat(0.5);
        }
        return;
    }
    // Centre the *region*, so a wide selection is centred vertically and a
    // tall one horizontally.
    let offset = (Vec2::splat(span) - extent) * 0.5;
    for (_, uv) in written.iter_mut() {
        *uv = (*uv - min + offset) / span;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::AttrType;
    use crate::primitives;
    use crate::MeshData;

    fn bmesh(mesh: &MeshData) -> BMesh {
        BMesh::from_mesh_data(mesh)
    }

    fn uvs(bm: &BMesh, layer: &str) -> Vec<Vec2> {
        bm.attributes().get_f32x2(layer).unwrap().to_vec()
    }

    /// The UVs of one face, in loop order.
    fn face_uvs(bm: &BMesh, f: FaceId, layer: &str) -> Vec<Vec2> {
        let all = bm.attributes().get_f32x2(layer).unwrap();
        bm.face_loops(f)
            .iter()
            .map(|l| all[l.index() as usize])
            .collect()
    }

    #[test]
    fn a_planar_projection_of_a_plane_fills_the_map() {
        // The simplest thing that must be exactly right. A flat quad projected
        // along its own normal is the identity up to the fit, so its corners
        // are the corners of the unit square.
        let mesh = primitives::plane(2.0);
        let mut bm = bmesh(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        let n = project(
            &mut bm,
            &faces,
            Projection::Planar {
                right: Vec3::X,
                up: Vec3::Y,
            },
            DEFAULT_LAYER,
        );
        assert_eq!(n, 4);

        let got = uvs(&bm, DEFAULT_LAYER);
        let mut sorted: Vec<(f32, f32)> = got.iter().map(|uv| (uv.x, uv.y)).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, vec![(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]);
    }

    #[test]
    fn the_fit_preserves_aspect() {
        // A two-by-one quad must come out two-by-one. Stretching it to fill
        // the map would put a square texture on it squashed — the first thing
        // an artist notices and the reason `Fit::Bounds` is aspect-preserving
        // rather than a per-axis normalise.
        let mut mesh = primitives::plane(1.0);
        {
            let positions = mesh.positions_mut();
            for p in positions.iter_mut() {
                p.x *= 4.0;
            }
        }
        let mut bm = bmesh(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        project(
            &mut bm,
            &faces,
            Projection::Planar {
                right: Vec3::X,
                up: Vec3::Y,
            },
            DEFAULT_LAYER,
        );

        let got = uvs(&bm, DEFAULT_LAYER);
        let width = got.iter().map(|uv| uv.x).fold(f32::MIN, f32::max)
            - got.iter().map(|uv| uv.x).fold(f32::MAX, f32::min);
        let height = got.iter().map(|uv| uv.y).fold(f32::MIN, f32::max)
            - got.iter().map(|uv| uv.y).fold(f32::MAX, f32::min);
        assert!((width - 1.0).abs() < 1e-5, "width {width}");
        assert!((height - 0.25).abs() < 1e-5, "height {height}");
        // And it is centred in the direction it does not fill.
        let mid = got.iter().map(|uv| uv.y).sum::<f32>() / got.len() as f32;
        assert!((mid - 0.5).abs() < 1e-5, "not centred: {mid}");
    }

    #[test]
    fn a_box_projection_puts_each_side_of_a_cube_on_its_own_axis() {
        // Six faces, six bases. What is checked is not the exact numbers but
        // the property that makes a box projection worth having: every face is
        // unwrapped without distortion, so every face's UV area is the same.
        let mesh = primitives::cube(2.0);
        let mut bm = bmesh(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        assert_eq!(faces.len(), 6);
        project(&mut bm, &faces, Projection::Box, DEFAULT_LAYER);

        let areas: Vec<f32> = faces
            .iter()
            .map(|f| polygon_area(&face_uvs(&bm, *f, DEFAULT_LAYER)))
            .collect();
        let first = areas[0].abs();
        assert!(first > 1e-4, "faces collapsed: {areas:?}");
        for a in &areas {
            assert!(
                (a.abs() - first).abs() < 1e-4,
                "a cube's faces must unwrap equally: {areas:?}"
            );
        }
    }

    #[test]
    fn a_box_projection_does_not_mirror_opposite_faces() {
        // Signed bases, not `abs`. With unsigned axes the two faces of a plate
        // land on top of each other with opposite winding, and the texture on
        // one of them reads backwards. Checked as winding, which is what
        // "backwards" means in UV space.
        let mesh = primitives::cube(2.0);
        let mut bm = bmesh(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        project(&mut bm, &faces, Projection::Box, DEFAULT_LAYER);

        let signs: Vec<f32> = faces
            .iter()
            .map(|f| polygon_area(&face_uvs(&bm, *f, DEFAULT_LAYER)).signum())
            .collect();
        assert!(
            signs.iter().all(|s| *s == signs[0]),
            "faces disagree on winding, so some texture reads mirrored: {signs:?}"
        );
    }

    #[test]
    fn a_face_across_the_seam_stays_a_face() {
        // The bug this module's seam handling exists to prevent. A cylinder's
        // longitude wraps, and the face straddling 0/1 would otherwise hold
        // corners at 0.99 and 0.01 — drawn as a band across the whole map.
        //
        // Checked as UV width per face: no face may be wider than a fraction
        // of the map, because a cylinder of 16 sides has faces a sixteenth of
        // a turn wide.
        let mesh = primitives::cylinder(1.0, 2.0, 16);
        let mut bm = bmesh(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        project_with(
            &mut bm,
            &faces,
            Projection::Cylinder {
                axis: Vec3::Z,
                center: Vec3::ZERO,
            },
            DEFAULT_LAYER,
            Fit::None,
        );

        // The sides only. A cap is a disc whose corners really do go all the
        // way around the axis, so under a cylindrical projection it really
        // does span the whole map — that is what cylindrical projection *is*,
        // not a seam defect, and the artist projects caps separately. Checking
        // it here would be asserting that a correct answer is wrong.
        let mut sides = 0;
        let mut caps = 0;
        for f in &faces {
            if bm.face_normal(*f).dot(Vec3::Z).abs() > 0.9 {
                caps += 1;
                continue;
            }
            sides += 1;
            let face = face_uvs(&bm, *f, DEFAULT_LAYER);
            let lo = face.iter().map(|uv| uv.x).fold(f32::MAX, f32::min);
            let hi = face.iter().map(|uv| uv.x).fold(f32::MIN, f32::max);
            assert!(
                // A sixteen-sided cylinder's sides are a sixteenth of a turn
                // wide, so anything near a half is the seam, not geometry.
                hi - lo < 0.15,
                "a face spans {:.3} of the map: the seam is not being unwrapped",
                hi - lo
            );
        }
        assert_eq!((sides, caps), (16, 2), "not the cylinder this test assumes");
    }

    #[test]
    fn the_seam_fix_is_what_makes_that_true() {
        // Guards the test above. Without the period unwrap there really is a
        // face spanning nearly the whole map, so the assertion is not vacuous
        // — a cylinder built with its first vertex elsewhere would pass by
        // luck rather than by the mechanism.
        let mesh = primitives::cylinder(1.0, 2.0, 16);
        let bm = bmesh(&mesh);
        let mut widest: f32 = 0.0;
        for f in bm.face_ids() {
            let raw: Vec<f32> = bm
                .face_loops(f)
                .iter()
                .filter_map(|l| bm.loop_vert(*l))
                .filter_map(|v| bm.position(v))
                .map(|p| angle_fraction(p.x, p.y))
                .collect();
            if raw.is_empty() {
                continue;
            }
            let lo = raw.iter().copied().fold(f32::MAX, f32::min);
            let hi = raw.iter().copied().fold(f32::MIN, f32::max);
            widest = widest.max(hi - lo);
        }
        assert!(
            widest > 0.5,
            "no face straddles the seam, so the seam test proves nothing: {widest}"
        );
    }

    #[test]
    fn a_spherical_projection_spans_the_map_once() {
        let mesh = primitives::uv_sphere(1.0, 24, 12);
        let mut bm = bmesh(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        project_with(
            &mut bm,
            &faces,
            Projection::Sphere {
                axis: Vec3::Z,
                center: Vec3::ZERO,
            },
            DEFAULT_LAYER,
            Fit::None,
        );
        let got = uvs(&bm, DEFAULT_LAYER);
        // Latitude covers the poles exactly: 0 at the bottom, 1 at the top.
        let lo = got.iter().map(|uv| uv.y).fold(f32::MAX, f32::min);
        let hi = got.iter().map(|uv| uv.y).fold(f32::MIN, f32::max);
        assert!(lo < 0.01, "latitude does not reach the bottom pole: {lo}");
        assert!(hi > 0.99, "latitude does not reach the top pole: {hi}");
        // And no face straddles the longitude seam.
        for f in &faces {
            let face = face_uvs(&bm, *f, DEFAULT_LAYER);
            let flo = face.iter().map(|uv| uv.x).fold(f32::MAX, f32::min);
            let fhi = face.iter().map(|uv| uv.x).fold(f32::MIN, f32::max);
            assert!(fhi - flo < 0.5, "face spans {:.3}", fhi - flo);
        }
    }

    #[test]
    fn the_axis_can_be_any_direction() {
        // `frame` picks its reference vector away from the axis. An axis of
        // exactly Z is the common case and an axis of exactly X used to be the
        // one that produced a zero-length cross product and NaN everywhere.
        for axis in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            -Vec3::X,
            Vec3::new(1.0, 1.0, 1.0),
        ] {
            let mesh = primitives::uv_sphere(1.0, 12, 6);
            let mut bm = bmesh(&mesh);
            let faces: Vec<FaceId> = bm.face_ids().collect();
            project(
                &mut bm,
                &faces,
                Projection::Sphere {
                    axis,
                    center: Vec3::ZERO,
                },
                DEFAULT_LAYER,
            );
            let got = uvs(&bm, DEFAULT_LAYER);
            assert!(
                got.iter().all(|uv| uv.is_finite()),
                "axis {axis:?} produced a non-finite UV"
            );
        }
    }

    #[test]
    fn projecting_a_selection_leaves_the_rest_alone() {
        // What makes it possible to unwrap a character in parts. The layer is
        // created if missing but never cleared, so an unselected face keeps
        // whatever it had.
        let mesh = primitives::cube(2.0);
        let mut bm = bmesh(&mesh);
        let len = bm.slot_count(Domain::Corner);
        bm.attributes_mut()
            .ensure(DEFAULT_LAYER, Domain::Corner, AttrType::F32x2, len);
        if let Some(values) = bm.attributes_mut().make_mut_f32x2(DEFAULT_LAYER) {
            for v in values.iter_mut() {
                *v = Vec2::new(-7.0, -7.0);
            }
        }

        let all: Vec<FaceId> = bm.face_ids().collect();
        let one = vec![all[0]];
        let n = project(&mut bm, &one, Projection::Box, DEFAULT_LAYER);
        assert_eq!(n, 4);

        let touched = bm.face_loops(all[0]);
        let got = uvs(&bm, DEFAULT_LAYER);
        for l in &touched {
            assert_ne!(got[l.index() as usize], Vec2::new(-7.0, -7.0));
        }
        let untouched: usize = all[1..]
            .iter()
            .flat_map(|f| bm.face_loops(*f))
            .filter(|l| got[l.index() as usize] == Vec2::new(-7.0, -7.0))
            .count();
        assert_eq!(untouched, 20, "the unselected faces were overwritten");
    }

    #[test]
    fn a_named_layer_is_written_and_the_default_is_left_alone() {
        // UV sets are named Corner layers, so a second projection into a
        // second name must not disturb the first. This is the whole of "more
        // than one UV set" as far as this module is concerned.
        let mesh = primitives::plane(2.0);
        let mut bm = bmesh(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        project(
            &mut bm,
            &faces,
            Projection::Planar {
                right: Vec3::X,
                up: Vec3::Y,
            },
            DEFAULT_LAYER,
        );
        let first = uvs(&bm, DEFAULT_LAYER);

        project(
            &mut bm,
            &faces,
            Projection::Planar {
                right: Vec3::Y,
                up: Vec3::X,
            },
            "uv_lightmap",
        );
        assert_eq!(uvs(&bm, DEFAULT_LAYER), first, "the first set moved");
        assert_ne!(
            uvs(&bm, "uv_lightmap"),
            first,
            "the second set is a copy of the first"
        );
    }

    #[test]
    fn an_edge_on_face_does_not_divide_by_zero() {
        // A quad projected along its own plane collapses to a line, and a
        // selection of one such face has zero extent in one axis — or in both,
        // for a face seen exactly edge-on. The fit must not produce NaN, which
        // would reach the GPU as a hole in the mesh.
        let mesh = primitives::plane(2.0);
        let mut bm = bmesh(&mesh);
        let faces: Vec<FaceId> = bm.face_ids().collect();
        project(
            &mut bm,
            &faces,
            // Both basis vectors perpendicular to the plane: everything
            // projects to the origin.
            Projection::Planar {
                right: Vec3::Z,
                up: Vec3::Z,
            },
            DEFAULT_LAYER,
        );
        let got = uvs(&bm, DEFAULT_LAYER);
        assert!(got.iter().all(|uv| uv.is_finite()), "{got:?}");
        assert!(got.iter().all(|uv| *uv == Vec2::splat(0.5)));
    }

    #[test]
    fn an_empty_selection_writes_nothing_and_creates_no_layer() {
        // A primitive arrives with coordinates of its own, and this test is
        // about the layer being created here — so it is taken away first.
        let mut mesh = primitives::cube(2.0);
        mesh.attributes_mut().remove(DEFAULT_LAYER);
        let mut bm = bmesh(&mesh);
        assert_eq!(project(&mut bm, &[], Projection::Box, DEFAULT_LAYER), 0);
        assert!(
            bm.attributes().get(DEFAULT_LAYER).is_none(),
            "an empty projection left a layer behind"
        );
    }

    /// Twice the signed area of a polygon, by the shoelace formula.
    fn polygon_area(uvs: &[Vec2]) -> f32 {
        let mut sum = 0.0;
        for i in 0..uvs.len() {
            let a = uvs[i];
            let b = uvs[(i + 1) % uvs.len()];
            sum += a.x * b.y - b.x * a.y;
        }
        sum * 0.5
    }
}
