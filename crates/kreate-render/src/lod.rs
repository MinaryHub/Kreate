//! How much of an object to draw, given how big it is on screen.
//!
//! Two decisions live here, and the plan puts them first in the large-scene
//! phase because they are the safest things on the list and among the largest
//! wins:
//!
//! 1. **Screen-size culling.** An object two pixels across cannot be seen and
//!    cannot be clicked, so skipping it loses the user nothing.
//! 2. **Overlay density LOD.** When a mesh has less than a pixel of screen per
//!    vertex, the vertex dots are a grey smear that hides the shading
//!    underneath and cannot be aimed at. Fading them out is not a compromise —
//!    users read it as the viewport getting *more* legible.
//!
//! The second is the one that matters. Solid shading ten million triangles is
//! easy; drawing ten million vertex dots and edges with per-element state is
//! not, and the overlay path is where an editor actually spends its frame. The
//! plan's estimate is a fivefold speedup on dense scenes.
//!
//! # What is never simplified
//!
//! The object being edited, and anything selected. Both are protected by
//! [`Detail::of`]'s `protected` flag rather than by a threshold, because the
//! failure is not a slow frame — it is a user who selected something and
//! cannot see what they selected. An editor may lie about how *much* detail it
//! draws; it may never lie about *what is there*.
//!
//! Picking is unaffected either way: it reads an id buffer rendered from the
//! same geometry, so what looks clickable is clickable, and the CPU BVH that
//! tools query is always full resolution.

use crate::camera::{Camera, Projection};
use crate::mesh_gpu::MeshCounts;
use glam::{DAffine3, DVec3};
use kreate_math::Aabb;

/// How large a draw is on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewSize {
    /// The world-space bounding sphere's diameter, in device pixels.
    pub diameter_px: f32,
    /// Screen area it covers, in square pixels, as a disc of that diameter.
    ///
    /// Area rather than diameter is what the overlay thresholds want: "pixels
    /// per vertex" is the quantity that decides whether a field of dots is
    /// readable, and that is an area divided by a count.
    pub area_px2: f32,
}

impl ViewSize {
    /// Everything, for the cases where measuring is meaningless — an empty
    /// mesh's bounds, a camera inside the object, a degenerate viewport.
    ///
    /// Deliberately the *unculled* answer. Every uncertainty in this module
    /// resolves towards drawing, because a wrongly hidden object is a bug the
    /// user reports as data loss and a wrongly drawn one costs a fraction of a
    /// millisecond.
    pub const EVERYTHING: Self = Self {
        diameter_px: f32::INFINITY,
        area_px2: f32::INFINITY,
    };

    /// Measures a mesh's bounds under a transform, as seen by a camera.
    ///
    /// `viewport_h_px` is the render target's height in device pixels — the
    /// same resolution the pass runs at, not the interface's point size, or
    /// dynamic resolution scaling would change what is culled.
    pub fn measure(
        bounds: Aabb,
        transform: &DAffine3,
        camera: &Camera,
        viewport_h_px: u32,
    ) -> Self {
        if bounds.is_empty() || viewport_h_px == 0 {
            return Self::EVERYTHING;
        }

        // A bounding sphere under a rotation-and-scale is a sphere scaled by the
        // largest scale factor, and for a TRS transform that is exactly the
        // longest column of the linear part: rotation keeps the columns
        // orthogonal, so the singular values *are* the column lengths. Only a
        // parent chain that composes non-uniform scale with rotation introduces
        // shear and makes this a slight underestimate — bounded, and at these
        // thresholds inconsequential.
        let linear = transform.matrix3;
        let scale = linear
            .x_axis
            .length()
            .max(linear.y_axis.length())
            .max(linear.z_axis.length());
        let radius = bounds.bounding_radius() as f64 * scale;
        if !radius.is_finite() || radius <= 0.0 {
            return Self::EVERYTHING;
        }

        let center = bounds.center();
        let center = transform.transform_point3(DVec3::new(
            center.x as f64,
            center.y as f64,
            center.z as f64,
        ));

        let Some(diameter) =
            sphere_diameter_px(center - camera.eye(), radius, camera, viewport_h_px)
        else {
            return Self::EVERYTHING;
        };
        let diameter_px = diameter as f32;
        let radius_px = diameter_px * 0.5;
        Self {
            diameter_px,
            area_px2: std::f32::consts::PI * radius_px * radius_px,
        }
    }
}

/// How many device pixels across a bounding sphere appears.
///
/// `centre` is camera-relative — the vector from the eye to the sphere's centre
/// — because that is the form everything downstream uses: the GPU is handed
/// camera-relative positions so that a scene ten kilometres from the origin
/// keeps its float precision.
///
/// `None` means the question is meaningless and the caller should draw: a
/// degenerate viewport, a camera inside the sphere, an orthographic view with no
/// height. Every uncertainty in this module resolves towards drawing.
///
/// One implementation rather than two, because the cluster cull needs exactly
/// this and having a second copy of a projection formula is how the CPU
/// reference and the shader drift apart.
pub fn sphere_diameter_px(
    centre: DVec3,
    radius: f64,
    camera: &Camera,
    viewport_h_px: u32,
) -> Option<f64> {
    if viewport_h_px == 0 || !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    let height = viewport_h_px as f64;
    let diameter = match camera.projection {
        Projection::Orthographic => {
            let view_height = camera.ortho_height();
            if view_height <= 0.0 {
                return None;
            }
            2.0 * radius / view_height * height
        }
        Projection::Perspective => {
            let distance = centre.length();
            // The camera is inside the sphere, so it covers the screen. Also the
            // guard against dividing by zero.
            if distance <= radius {
                return None;
            }
            let half_extent = distance * (camera.fov_y * 0.5).tan();
            if half_extent <= 0.0 {
                return None;
            }
            radius / half_extent * height
        }
    };
    diameter.is_finite().then_some(diameter)
}

/// The thresholds, so they are one named thing rather than constants scattered
/// through the draw loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    /// Below this diameter in pixels, skip the object entirely.
    pub cull_px: f32,
    /// Square pixels per vertex at which vertex dots are drawn at full alpha.
    pub point_full_px2: f32,
    /// Square pixels per vertex at which they are gone.
    pub point_none_px2: f32,
    /// Square pixels per edge at which the wireframe is at full alpha.
    pub wire_full_px2: f32,
    /// Square pixels per edge at which it is gone.
    pub wire_none_px2: f32,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            // Two pixels across is about four square pixels of coverage, which
            // is the plan's figure. At that size an object is a speck that
            // cannot be aimed at.
            cull_px: 2.0,
            // Derived from the dot, not fitted to a scene. `Viewport::point_size`
            // is 6 pixels, so one dot covers about pi*3^2 = 28 square pixels.
            // Give each vertex that much screen and the dots sit edge to edge,
            // which is the last density at which they read as separate handles:
            // full alpha. Give it a quarter of that and every pixel is under
            // four overlapping dots, which is a solid mass carrying no
            // information at all: gone.
            //
            // Worth saying why this is derived rather than tuned. The band
            // started at [1, 6], from the plan's "under a pixel per vertex",
            // and timing it showed that figure describes where information is
            // provably *absent* rather than where it stops being readable — a
            // quarter-million-vertex mesh filling the frame sat mid-band, still
            // issuing every instance, and measured a speedup of exactly 1.0x.
            // Two rounds of nudging constants at that one scene followed, which
            // is how a calibration becomes an overfit; the dot's own geometry is
            // a reason that holds for any scene.
            //
            // Safe to be this aggressive because of who it applies to: vertex
            // dots are only drawn in edit mode, and the mesh *being* edited is
            // protected. So this only ever fades the dots of neighbouring
            // objects — context, never the target.
            point_full_px2: 28.0,
            point_none_px2: 7.0,
            // Edges survive density better than dots, and the reason is
            // geometric: a one-pixel line needs about a pixel of area to be
            // visible where a dot needs twenty-eight. A dense wireframe still
            // shows silhouette and creases long after a dot field has become a
            // smear, so it fades far later — only once it has turned into a flat
            // grey fill over the shading it is meant to annotate.
            //
            // Kept well below half the vertex figures, and that is the whole
            // point: a quad mesh has about *twice* as many edges as vertices, so
            // bands that merely look lower here would still fade the wireframe
            // first. Half of the vertex band would make the two fade in
            // lockstep; going below that is what actually orders them. Verified
            // over a density sweep rather than trusted — the first calibration
            // got this backwards and the sweep is what said so.
            wire_full_px2: 2.0,
            wire_none_px2: 0.25,
        }
    }
}

/// What to draw for one object this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detail {
    /// Whether to draw the object at all.
    pub draw: bool,
    /// Multiplier on the wireframe's alpha. Zero means skip the draw.
    pub wire_alpha: f32,
    /// Multiplier on the vertex points' alpha. Zero means skip the draw.
    pub point_alpha: f32,
}

impl Detail {
    /// Everything at full strength.
    pub const FULL: Self = Self {
        draw: true,
        wire_alpha: 1.0,
        point_alpha: 1.0,
    };

    /// Decides what one object gets.
    ///
    /// `protected` opts an object out of every simplification: the mesh being
    /// edited, and anything selected. See the module docs for why that is a
    /// flag and not a threshold.
    pub fn of(size: ViewSize, counts: MeshCounts, budget: &Budget, protected: bool) -> Self {
        if protected {
            return Self::FULL;
        }
        if size.diameter_px < budget.cull_px {
            return Self {
                draw: false,
                wire_alpha: 0.0,
                point_alpha: 0.0,
            };
        }
        Self {
            draw: true,
            wire_alpha: fade(
                per_element(size.area_px2, counts.edges),
                budget.wire_none_px2,
                budget.wire_full_px2,
            ),
            point_alpha: fade(
                per_element(size.area_px2, counts.verts),
                budget.point_none_px2,
                budget.point_full_px2,
            ),
        }
    }
}

/// Screen area available to each of `count` elements.
///
/// An element count of zero returns infinity rather than dividing: there is
/// nothing to fade, and the caller skips the draw on the count anyway.
fn per_element(area_px2: f32, count: u32) -> f32 {
    if count == 0 {
        return f32::INFINITY;
    }
    area_px2 / count as f32
}

/// Alpha below which an overlay is treated as absent.
///
/// The fade only reaches exactly zero at the very bottom of its band, so without
/// a floor an overlay spends a stretch of the range at one or two percent alpha:
/// invisible, and still issuing every instance. Measuring the density LOD is
/// what exposed this — a quarter-million-vertex mesh sat at 0.5% alpha and
/// measured a speedup of 1.0x, because the draw was still being made.
///
/// Two percent is under one step of an eight-bit channel over a dark surface, so
/// snapping it away is a discontinuity nobody can see. That matters: the whole
/// reason the ramp is smooth is to avoid a *perceptible* pop as the user orbits,
/// and a step this small does not qualify.
const VISIBLE_FLOOR: f32 = 0.02;

/// Smooth ramp from 0 at `none` to 1 at `full`.
///
/// Smooth and not a step because the alternative is visible: a hard cutoff makes
/// the wireframe blink on and off as the user orbits, and a blink reads as a
/// rendering bug where a fade reads as depth of field. The metric is continuous
/// in distance, so the fade is too — apart from the deliberate snap at
/// [`VISIBLE_FLOOR`], which is what turns the bottom of the fade from cheap-
/// looking into actually cheap.
fn fade(value: f32, none: f32, full: f32) -> f32 {
    if !value.is_finite() {
        return 1.0;
    }
    if full <= none {
        // A degenerate budget must not silently hide geometry.
        return 1.0;
    }
    let t = ((value - none) / (full - none)).clamp(0.0, 1.0);
    // Smoothstep, so the ramp leaves and arrives flat and the transition has no
    // perceptible edge at either end.
    let alpha = t * t * (3.0 - 2.0 * t);
    if alpha < VISIBLE_FLOOR {
        return 0.0;
    }
    alpha
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn counts(verts: u32, edges: u32) -> MeshCounts {
        MeshCounts {
            verts,
            edges,
            faces: 1,
            triangles: 2,
        }
    }

    fn unit_cube() -> Aabb {
        Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5))
    }

    fn camera_at(distance: f64) -> Camera {
        Camera {
            pivot: DVec3::ZERO,
            distance,
            ..Camera::default()
        }
    }

    #[test]
    fn a_nearby_object_is_measured_larger_than_a_distant_one() {
        let near = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(5.0), 1080);
        let far = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(500.0), 1080);
        assert!(near.diameter_px > far.diameter_px);
        // Perspective, so a hundred times the distance is a hundredth the size.
        let ratio = near.diameter_px / far.diameter_px;
        assert!((ratio - 100.0).abs() < 1.0, "ratio was {ratio}");
    }

    #[test]
    fn scaling_an_object_scales_its_measured_size() {
        let plain = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(50.0), 1080);
        let scaled = ViewSize::measure(
            unit_cube(),
            &DAffine3::from_scale(DVec3::splat(10.0)),
            &camera_at(50.0),
            1080,
        );
        let ratio = scaled.diameter_px / plain.diameter_px;
        assert!((ratio - 10.0).abs() < 0.01, "ratio was {ratio}");
    }

    #[test]
    fn a_rotation_does_not_change_the_measured_size() {
        // The metric is a bounding *sphere*, so it must be rotation invariant.
        // If it were not, an object would pop in and out of existence as it
        // turned, which is the exact artefact a bounding sphere is chosen to
        // avoid.
        let plain = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(400.0), 1080);
        let turned = ViewSize::measure(
            unit_cube(),
            &DAffine3::from_axis_angle(DVec3::new(1.0, 2.0, 3.0).normalize(), 0.7),
            &camera_at(400.0),
            1080,
        );
        assert!((plain.diameter_px - turned.diameter_px).abs() < 1e-4);
    }

    #[test]
    fn a_non_uniformly_scaled_object_is_measured_by_its_largest_axis() {
        // Underestimating the size would cull something the user can see, so the
        // largest scale factor is the one that has to win.
        let size = ViewSize::measure(
            unit_cube(),
            &DAffine3::from_scale(DVec3::new(1.0, 1.0, 40.0)),
            &camera_at(500.0),
            1080,
        );
        let uniform = ViewSize::measure(
            unit_cube(),
            &DAffine3::from_scale(DVec3::splat(40.0)),
            &camera_at(500.0),
            1080,
        );
        assert!((size.diameter_px - uniform.diameter_px).abs() < 1e-4);
    }

    #[test]
    fn a_speck_is_culled_and_a_visible_object_is_not() {
        let budget = Budget::default();
        let speck = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(2000.0), 1080);
        assert!(speck.diameter_px < budget.cull_px, "{speck:?}");
        assert!(!Detail::of(speck, counts(8, 12), &budget, false).draw);

        let visible = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(20.0), 1080);
        assert!(Detail::of(visible, counts(8, 12), &budget, false).draw);
    }

    #[test]
    fn a_culled_object_also_reports_zero_overlay_alpha() {
        // Relied on by the draw loop, so pinned here. The overlay passes test
        // both `draw` and their alpha, and that redundancy is only harmless
        // because a culled object reports zero for both — a falsification sweep
        // confirmed that removing `draw` from those two branches changes
        // nothing on screen.
        //
        // If `Detail::of` ever stopped zeroing the alphas on a cull, those
        // branches would become load-bearing rather than redundant, and this
        // test is what says so rather than leaving it to be discovered.
        let culled = Detail::of(
            ViewSize {
                diameter_px: 0.5,
                area_px2: 0.2,
            },
            counts(8, 12),
            &Budget::default(),
            false,
        );
        assert!(!culled.draw);
        assert_eq!(culled.wire_alpha, 0.0);
        assert_eq!(culled.point_alpha, 0.0);
    }

    #[test]
    fn a_selected_speck_is_still_drawn() {
        // The whole point of the protected flag. A user who selects something
        // and then cannot see it has been told a lie about what exists, which is
        // categorically worse than a slow frame.
        let budget = Budget::default();
        let speck = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(2000.0), 1080);
        assert_eq!(
            Detail::of(speck, counts(8, 12), &budget, true),
            Detail::FULL
        );
    }

    #[test]
    fn a_dense_mesh_loses_its_vertex_dots_before_its_wireframe() {
        // Ordering claim, not just a pair of numbers: dots become unreadable
        // sooner than edges do, so at some density the wireframe must still be
        // there after the dots have gone.
        let budget = Budget::default();
        // 100x100 pixels of screen for a million vertices: 0.01 px² each.
        let size = ViewSize {
            diameter_px: 113.0,
            area_px2: 10_000.0,
        };
        let detail = Detail::of(size, counts(1_000_000, 2_000_000), &budget, false);
        assert!(detail.draw);
        assert_eq!(detail.point_alpha, 0.0);
        assert_eq!(detail.wire_alpha, 0.0);

        // A milder density: 4 px² per vertex, 2 per edge.
        let size = ViewSize {
            diameter_px: 500.0,
            area_px2: 40_000.0,
        };
        let detail = Detail::of(size, counts(10_000, 20_000), &budget, false);
        assert!(
            detail.point_alpha < detail.wire_alpha,
            "dots should fade first: {detail:?}"
        );
    }

    #[test]
    fn dots_never_outlast_the_wireframe_at_any_density() {
        // The ordering above stated as an invariant over the whole range, on the
        // element ratio a quad mesh actually has: twice as many edges as
        // vertices.
        //
        // Worth a sweep and not a sample, because the first calibration failed
        // exactly here. Bands that read as "lower" for edges still faded the
        // wireframe first, since the edge count divides the same screen area
        // twice as many ways. A single density would have been chosen to pass.
        let budget = Budget::default();
        let verts = 10_000u32;
        for i in 0..=200 {
            let per_vert = 0.05 * i as f32;
            let size = ViewSize {
                diameter_px: 500.0,
                area_px2: per_vert * verts as f32,
            };
            let detail = Detail::of(size, counts(verts, verts * 2), &budget, false);
            assert!(
                detail.point_alpha <= detail.wire_alpha,
                "at {per_vert} px2/vert the dots outlasted the wireframe: {detail:?}"
            );
        }
    }

    #[test]
    fn a_sparse_mesh_is_untouched() {
        // The common case, and the one a regression here would ruin: an ordinary
        // mesh at an ordinary distance must be bit-for-bit unaffected.
        let budget = Budget::default();
        let size = ViewSize {
            diameter_px: 400.0,
            area_px2: 125_000.0,
        };
        assert_eq!(
            Detail::of(size, counts(500, 1_000), &budget, false),
            Detail::FULL
        );
    }

    #[test]
    fn the_fade_is_continuous_across_its_whole_range() {
        // A blink reads as a bug; a fade reads as intent. Checked by walking the
        // band and bounding the step between neighbouring samples, because the
        // artefact this prevents is a *discontinuity*, not a wrong endpoint.
        //
        // The tolerance also has to accommodate the deliberate snap at
        // `VISIBLE_FLOOR`, which is a step of two percent — the one
        // discontinuity in here, and small enough that no orbit reveals it.
        let budget = Budget::default();
        let steps = 400;
        let mut previous = None;
        for i in 0..=steps {
            let per_vert = budget.point_none_px2
                + (budget.point_full_px2 - budget.point_none_px2) * i as f32 / steps as f32;
            let size = ViewSize {
                diameter_px: 500.0,
                area_px2: per_vert * 10_000.0,
            };
            let alpha = Detail::of(size, counts(10_000, 1), &budget, false).point_alpha;
            if let Some(prev) = previous {
                assert!(
                    (alpha - prev) < 0.05,
                    "alpha jumped from {prev} to {alpha} at step {i}"
                );
            }
            previous = Some(alpha);
        }
        assert_eq!(previous, Some(1.0));
    }

    #[test]
    fn an_invisible_fade_becomes_no_draw_at_all() {
        // The floor, and why it exists: without it the bottom of the band is a
        // stretch where an overlay is invisible and fully paid for. Measured as
        // exactly a 1.0x speedup on a quarter-million-vertex mesh before this
        // was added.
        let budget = Budget::default();
        let alpha = |per_vert: f32| {
            Detail::of(
                ViewSize {
                    diameter_px: 500.0,
                    area_px2: per_vert * 10_000.0,
                },
                counts(10_000, 1),
                &budget,
                false,
            )
            .point_alpha
        };
        // Just inside the band, where smoothstep is a fraction of a percent.
        assert_eq!(alpha(budget.point_none_px2 + 0.4), 0.0);
        // And far enough in to be worth drawing, so the floor has not eaten the
        // whole lower half of the ramp.
        assert!(alpha(budget.point_none_px2 + 2.0) > VISIBLE_FLOOR);
    }

    #[test]
    fn the_fade_reaches_both_ends_exactly() {
        let budget = Budget::default();
        let at = |per_vert: f32| {
            Detail::of(
                ViewSize {
                    diameter_px: 500.0,
                    area_px2: per_vert * 1_000.0,
                },
                counts(1_000, 1),
                &budget,
                false,
            )
            .point_alpha
        };
        assert_eq!(at(budget.point_none_px2), 0.0);
        assert_eq!(at(budget.point_none_px2 * 0.5), 0.0);
        assert_eq!(at(budget.point_full_px2), 1.0);
        assert_eq!(at(budget.point_full_px2 * 4.0), 1.0);
    }

    #[test]
    fn an_empty_mesh_measures_as_everything_rather_than_as_nothing() {
        // An empty AABB has an infinite min and a negative-infinite max, so a
        // radius computed from it is garbage. Resolving towards drawing means
        // the failure is a wasted draw of a mesh with nothing in it, not a
        // silently vanished object.
        let size = ViewSize::measure(Aabb::EMPTY, &DAffine3::IDENTITY, &camera_at(10.0), 1080);
        assert_eq!(size, ViewSize::EVERYTHING);
        assert_eq!(
            Detail::of(size, counts(0, 0), &Budget::default(), false),
            Detail::FULL
        );
    }

    #[test]
    fn a_camera_inside_an_object_sees_all_of_it() {
        // Distance minus radius goes through zero here, so the naive formula
        // divides by zero and then reports a negative size — which would cull
        // the object the camera is standing inside.
        let size = ViewSize::measure(
            unit_cube(),
            &DAffine3::from_scale(DVec3::splat(100.0)),
            &camera_at(1.0),
            1080,
        );
        assert_eq!(size, ViewSize::EVERYTHING);
    }

    #[test]
    fn a_zero_height_viewport_measures_as_everything() {
        // Reached during a resize, when the target is momentarily degenerate.
        let size = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(10.0), 0);
        assert_eq!(size, ViewSize::EVERYTHING);
    }

    #[test]
    fn a_taller_viewport_makes_everything_bigger() {
        // The metric is in device pixels, so it has to follow the render
        // resolution. Dynamic resolution scaling changes this number, and it
        // should: at 70% scale there genuinely are fewer pixels per vertex.
        let small = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(100.0), 540);
        let large = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &camera_at(100.0), 1080);
        assert!((large.diameter_px / small.diameter_px - 2.0).abs() < 1e-3);
    }

    #[test]
    fn an_orthographic_camera_measures_by_its_view_height() {
        let mut cam = camera_at(10.0);
        cam.projection = Projection::Orthographic;
        let size = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &cam, 1080);
        // The bounding *sphere* of a unit cube has diameter sqrt(3), not 1: the
        // metric measures the sphere that encloses the box, so its corners are
        // inside it.
        let expected = 3.0f64.sqrt() / cam.ortho_height() * 1080.0;
        assert!(
            (size.diameter_px as f64 - expected).abs() < 1e-3,
            "{} vs {expected}",
            size.diameter_px
        );
    }

    #[test]
    fn dollying_an_orthographic_camera_still_changes_the_measured_size() {
        // Not the obvious answer, and worth pinning. An orthographic projection
        // does not shrink things with distance, so the temptation is to assert
        // that dollying changes nothing — which is what the first version of
        // this test asserted, and it was wrong.
        //
        // This camera derives its ortho view height *from* `distance`
        // (`ortho_height` = 2 d tan(fov/2)), because that is how zooming an
        // orthographic viewport works: there is nothing else to zoom. So the
        // measured size does follow distance, through the view height.
        let mut near = camera_at(10.0);
        near.projection = Projection::Orthographic;
        let mut far = near;
        far.distance = 1000.0;
        let near_px = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &near, 1080).diameter_px;
        let far_px = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &far, 1080).diameter_px;
        let ratio = near_px / far_px;
        assert!((ratio - 100.0).abs() < 0.01, "ratio was {ratio}");
    }

    #[test]
    fn switching_projection_does_not_change_the_size_at_the_pivot() {
        // The camera picks its ortho height precisely so that toggling
        // projection keeps the framing at the pivot unchanged, so an object
        // sitting at the pivot has to measure identically under both. If it did
        // not, pressing the ortho toggle could cull something, or drop a
        // wireframe, for no reason the user could see.
        let perspective = camera_at(40.0);
        let mut orthographic = perspective;
        orthographic.projection = Projection::Orthographic;
        let a = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &perspective, 1080);
        let b = ViewSize::measure(unit_cube(), &DAffine3::IDENTITY, &orthographic, 1080);
        assert!(
            (a.diameter_px - b.diameter_px).abs() < 1e-3,
            "{a:?} vs {b:?}"
        );
    }
}
