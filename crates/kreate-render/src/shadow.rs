//! What a light sees.
//!
//! A shadow map is a depth buffer rendered from a light, so the first question
//! is where the light's camera is and what it frames. That question is
//! answerable without a device, which is why it is here on its own: a frustum
//! that is too small crops shadows at its edge, and one that is too large
//! spends its precision on empty space and stipples the contact shadows. Both
//! look like a shadow bug and neither is.
//!
//! # Eye-relative, like everything else
//!
//! The main pass has the fragment's position in eye-relative metres and the
//! lights packed the same way (see [`crate::lights`]), so the shadow matrix is
//! built there too. A shadow map in absolute world space would mean the sampling
//! shader converting back, in `f32`, at the exact magnitudes eye-relative space
//! exists to avoid.
//!
//! # Reversed-Z, like everything else
//!
//! Depth increases toward the light, cleared to zero, compared with
//! `GreaterEqual`. The float depth buffer then has its precision where the
//! geometry is, which is the whole reason §v1 chose it — and a shadow pass that
//! used the other convention would be a depth test that always passed.
//!
//! # What is not here
//!
//! Cascades. One frustum over the whole scene is the first step and it is
//! enough to see a shadow; a sun over a large scene wants three or four of
//! them, split by view distance, and the split is a policy question rather than
//! a geometric one. The shape below takes a bounding box, so a cascade is the
//! same function called with a smaller one.

use kreate_math::Aabb;

/// The camera a shadow map is rendered from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowView {
    /// Eye-relative world to the light's clip space, reversed-Z.
    pub view_proj: glam::Mat4,
    /// The near and far planes the projection was built with, in metres.
    ///
    /// Carried because the sampling pass needs them to turn a stored depth
    /// back into a distance, and recomputing them there would be a second
    /// place for the numbers to live.
    pub near: f32,
    pub far: f32,
}

/// A little room around the fitted box.
///
/// Geometry exactly on the boundary is geometry that half of its pixels fall
/// outside, and the edge of a shadow map is where a sample reads nothing.
const MARGIN: f32 = 1.0e-3;

/// The smallest near plane a perspective shadow will use, in metres.
///
/// Reversed-Z puts its precision near the near plane, so a near plane at a
/// millimetre wastes most of the buffer on the first centimetre. A tenth of a
/// metre is closer than any light usefully sits to what it lights.
const MIN_NEAR: f32 = 0.1;

/// Builds an orthographic projection in reversed-Z.
///
/// Written out rather than taken from `glam`, because the reversed-Z form is
/// not what any library's `orthographic_rh` produces and a silently normal-Z
/// matrix is a depth test that always passes.
fn ortho_reversed(half: glam::Vec3, near: f32, far: f32) -> glam::Mat4 {
    let depth = (far - near).max(1e-6);
    // `near` and `far` are distances along the looking direction, so the view
    // coordinates they name are `-near` and `-far`. This maps the first to 1
    // and the second to 0:
    //
    //     depth = (z_view + far) / (far - near)
    glam::Mat4::from_cols(
        glam::Vec4::new(1.0 / half.x, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, 1.0 / half.y, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0 / depth, 0.0),
        glam::Vec4::new(0.0, 0.0, far / depth, 1.0),
    )
}

/// A perspective projection in reversed-Z, with an infinite far plane.
///
/// The same form the main camera uses, so a shadow's depth means what the
/// viewport's does.
fn perspective_reversed(fov_y: f32, near: f32) -> glam::Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    glam::Mat4::from_cols(
        glam::Vec4::new(f, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, f, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 0.0, -1.0),
        glam::Vec4::new(0.0, 0.0, near, 0.0),
    )
}

/// A rotation that looks along `forward`.
///
/// Picks the up vector rather than taking one, because a shadow map has no
/// preferred roll — and a caller that had to supply one would have to know to
/// avoid the degenerate case, which is the whole difficulty.
fn look_along(forward: glam::Vec3) -> glam::Mat4 {
    let forward = forward.try_normalize().unwrap_or(glam::Vec3::NEG_Z);
    // Any axis not parallel to the direction. Z unless the light points along
    // it, which is the common case for a sun and exactly where a fixed up
    // vector degenerates.
    let up = if forward.z.abs() > 0.99 {
        glam::Vec3::Y
    } else {
        glam::Vec3::Z
    };
    let right = up.cross(forward).normalize();
    let up = forward.cross(right);
    // World to light: the transpose of the light's basis.
    glam::Mat4::from_cols(
        glam::Vec4::new(right.x, up.x, -forward.x, 0.0),
        glam::Vec4::new(right.y, up.y, -forward.y, 0.0),
        glam::Vec4::new(right.z, up.z, -forward.z, 0.0),
        glam::Vec4::W,
    )
}

/// What a parallel light sees: an orthographic box around the scene.
///
/// Fitted to the bounds *rotated into the light's frame*, not to the bounds
/// themselves. A box fitted to the world-axis extents is bigger than it needs
/// to be at every angle but one, and the excess comes out of the shadow's
/// resolution.
///
/// `None` for an empty scene, which is not an error: there is nothing to cast
/// and nothing to catch, and a pass that ran anyway would render a map nothing
/// samples.
pub fn sun_view(direction: glam::Vec3, bounds: Aabb) -> Option<ShadowView> {
    if bounds.is_empty() {
        return None;
    }
    let view = look_along(direction);
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for corner in bounds.corners() {
        let p = view.transform_point3(corner);
        min = min.min(p);
        max = max.max(p);
    }
    // The camera looks down -Z, so in front is negative and *nearer* is the
    // larger z. The distances the projection wants are the negated extents.
    //
    // Not clamped to zero, which was the first attempt and was wrong: a
    // parallel light has no position, so the frame's origin is arbitrary and
    // the scene can sit at any z at all — including entirely "behind" it. What
    // matters is only that the range covers the scene, and `far - near` is the
    // extent either way.
    let near = -max.z - MARGIN;
    let far = -min.z + MARGIN;
    let half = glam::Vec3::new(
        (max.x - min.x).max(1e-4) * 0.5 + MARGIN,
        (max.y - min.y).max(1e-4) * 0.5 + MARGIN,
        0.0,
    );
    // Centred on the scene, which is what stops the box wasting half its width
    // when the scene is off to one side.
    let centre = glam::Vec3::new((min.x + max.x) * 0.5, (min.y + max.y) * 0.5, 0.0);
    let recentre = glam::Mat4::from_translation(-centre);
    Some(ShadowView {
        view_proj: ortho_reversed(half, near, far) * recentre * view,
        near,
        far,
    })
}

/// What a spot sees: a perspective frustum as wide as its cone.
///
/// `cone_angle` is the full angle, as [`crate::lights::PlacedLight`] stores it.
/// The frustum is square and covers the cone's circle, so the corners of the
/// map are outside the cone and shade nothing — which is the price of a square
/// texture and is cheaper than the alternatives.
pub fn spot_view(
    position: glam::Vec3,
    direction: glam::Vec3,
    cone_angle: f32,
    bounds: Aabb,
) -> Option<ShadowView> {
    if bounds.is_empty() {
        return None;
    }
    let view = look_along(direction) * glam::Mat4::from_translation(-position);
    // How far the scene reaches in front of the light, which is what the far
    // plane has to cover.
    let mut far = 0.0f32;
    for corner in bounds.corners() {
        far = far.max(-view.transform_point3(corner).z);
    }
    if far <= 0.0 {
        // Every corner is behind the light. Nothing it lights is in the scene.
        return None;
    }
    let near = MIN_NEAR.min(far * 0.5);
    // A square frustum that contains the cone's circle: the diagonal of the
    // square is what the cone's full angle has to fit inside.
    let fov = (cone_angle.clamp(1e-3, std::f32::consts::PI * 0.99)).min(3.0);
    Some(ShadowView {
        view_proj: perspective_reversed(fov, near) * view,
        near,
        far: far + MARGIN,
    })
}

/// One face of a point light's cube.
///
/// Six 90° frustums. `face` is 0 to 5 in the order `+X -X +Y -Y +Z -Z`, which
/// is the order a cube texture's layers are.
pub fn point_view(position: glam::Vec3, face: usize, bounds: Aabb) -> Option<ShadowView> {
    if bounds.is_empty() {
        return None;
    }
    let direction = match face % 6 {
        0 => glam::Vec3::X,
        1 => glam::Vec3::NEG_X,
        2 => glam::Vec3::Y,
        3 => glam::Vec3::NEG_Y,
        4 => glam::Vec3::Z,
        _ => glam::Vec3::NEG_Z,
    };
    let view = look_along(direction) * glam::Mat4::from_translation(-position);
    let mut far = 0.0f32;
    for corner in bounds.corners() {
        far = far.max(-view.transform_point3(corner).z);
    }
    if far <= 0.0 {
        return None;
    }
    let near = MIN_NEAR.min(far * 0.5);
    Some(ShadowView {
        // Exactly 90°, so the six faces meet without a gap or an overlap.
        view_proj: perspective_reversed(std::f32::consts::FRAC_PI_2, near) * view,
        near,
        far: far + MARGIN,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> Aabb {
        Aabb::new(
            glam::Vec3::new(-3.0, -2.0, 0.0),
            glam::Vec3::new(4.0, 5.0, 2.0),
        )
    }

    /// Where a point lands in the shadow's clip space, after the divide.
    fn project(view: &ShadowView, p: glam::Vec3) -> glam::Vec3 {
        let clip = view.view_proj * p.extend(1.0);
        clip.truncate() / clip.w
    }

    fn inside(p: glam::Vec3) -> bool {
        p.x.abs() <= 1.0001 && p.y.abs() <= 1.0001 && (-0.0001..=1.0001).contains(&p.z)
    }

    #[test]
    fn a_suns_box_contains_the_whole_scene() {
        // A frustum too small crops the shadow at its edge, and the symptom is
        // a shadow that ends in mid-air.
        for direction in [
            glam::Vec3::NEG_Z,
            glam::Vec3::new(0.3, -0.5, -0.81).normalize(),
            glam::Vec3::X,
            glam::Vec3::new(-0.7, 0.7, 0.1).normalize(),
        ] {
            let view = sun_view(direction, scene()).expect("a scene");
            for corner in scene().corners() {
                let p = project(&view, corner);
                assert!(
                    inside(p),
                    "{direction} left {corner} at {p} outside the shadow"
                );
            }
        }
    }

    #[test]
    fn a_suns_box_is_no_bigger_than_it_needs_to_be() {
        // The other half, and the one that shows as stippled contact shadows
        // rather than as a missing shadow: a box fitted to the world-axis
        // extents is larger than the rotated scene at every angle but one, and
        // the excess comes out of the resolution.
        //
        // Measured by how far the scene actually spreads across the map. A
        // snug fit touches both edges.
        let view = sun_view(glam::Vec3::new(0.3, -0.5, -0.81).normalize(), scene())
            .expect("a scene");
        let (mut lo, mut hi) = (glam::Vec2::splat(f32::INFINITY), glam::Vec2::splat(f32::NEG_INFINITY));
        for corner in scene().corners() {
            let p = project(&view, corner);
            lo = lo.min(p.truncate());
            hi = hi.max(p.truncate());
        }
        assert!(lo.x < -0.98 && lo.y < -0.98, "the box is loose: {lo}");
        assert!(hi.x > 0.98 && hi.y > 0.98, "the box is loose: {hi}");
    }

    #[test]
    fn depth_increases_toward_the_light() {
        // Reversed-Z, like the rest of the renderer. The other convention is a
        // depth test that always passes, and a scene with no shadows in it at
        // all.
        let direction = glam::Vec3::NEG_Z;
        let view = sun_view(direction, scene()).expect("a scene");
        // The light looks down -Z, so the top of the scene is nearer to it.
        let near_point = glam::Vec3::new(0.0, 0.0, 2.0);
        let far_point = glam::Vec3::new(0.0, 0.0, 0.0);
        assert!(
            project(&view, near_point).z > project(&view, far_point).z,
            "depth does not increase toward the light"
        );
        // And the ends land on the range's ends, or the precision is being
        // spent somewhere the geometry is not.
        assert!(project(&view, near_point).z > 0.99);
        assert!(project(&view, far_point).z < 0.01);
    }

    #[test]
    fn a_sun_has_no_position_and_its_view_does_not_depend_on_one() {
        // A parallel light is a direction. A `sun_view` that took a position
        // would be a frustum somebody had to place, and placing it wrongly
        // crops the scene.
        let a = sun_view(glam::Vec3::NEG_Z, scene()).expect("a scene");
        let shifted = Aabb::new(
            glam::Vec3::new(-3.0, -2.0, 10.0),
            glam::Vec3::new(4.0, 5.0, 12.0),
        );
        let b = sun_view(glam::Vec3::NEG_Z, shifted).expect("a scene");
        // The two frames differ only in depth: the scene moved along the
        // light's axis, so the box slid rather than resized.
        for corner in shifted.corners() {
            assert!(inside(project(&b, corner)));
        }
        assert!(a.far - a.near > 0.0 && b.far - b.near > 0.0);
    }

    #[test]
    fn an_empty_scene_gives_no_shadow_camera() {
        // Nothing to cast and nothing to catch. A pass that ran anyway would
        // render a map nothing samples, every frame.
        assert!(sun_view(glam::Vec3::NEG_Z, Aabb::EMPTY).is_none());
        assert!(spot_view(glam::Vec3::ZERO, glam::Vec3::NEG_Z, 1.0, Aabb::EMPTY).is_none());
        assert!(point_view(glam::Vec3::ZERO, 0, Aabb::EMPTY).is_none());
    }

    #[test]
    fn a_spot_frames_what_is_in_front_of_it() {
        // On the axis at any distance the scene reaches, and not behind.
        let position = glam::Vec3::new(0.0, 0.0, 8.0);
        let view = spot_view(position, glam::Vec3::NEG_Z, 60f32.to_radians(), scene())
            .expect("a scene");
        for z in [6.0f32, 4.0, 1.0] {
            let p = project(&view, glam::Vec3::new(0.0, 0.0, z));
            assert!(inside(p), "a point on the axis at {z} landed at {p}");
        }
        // Behind the light is outside, and the divide does not make it look
        // inside — which is what a missing `w` check does.
        let behind = view.view_proj * glam::Vec3::new(0.0, 0.0, 10.0).extend(1.0);
        assert!(behind.w <= 0.0, "a point behind the light had w {}", behind.w);
    }

    #[test]
    fn a_spot_pointing_away_from_everything_has_nothing_to_draw() {
        // Reachable: a lamp aimed out of a window. Rendering a map for it is
        // a pass whose every fragment is clipped.
        let view = spot_view(
            glam::Vec3::new(0.0, 0.0, 8.0),
            glam::Vec3::Z,
            1.0,
            scene(),
        );
        assert!(view.is_none());
    }

    #[test]
    fn a_wider_cone_frames_more() {
        // The cone angle reaches the projection. A spot whose map ignored it
        // would have the same resolution however narrow the beam, which is
        // the resolution being wasted where it is most visible.
        let position = glam::Vec3::new(0.0, 0.0, 8.0);
        let off_axis = glam::Vec3::new(2.0, 0.0, 4.0);
        let narrow = spot_view(position, glam::Vec3::NEG_Z, 20f32.to_radians(), scene())
            .expect("a scene");
        let wide = spot_view(position, glam::Vec3::NEG_Z, 100f32.to_radians(), scene())
            .expect("a scene");
        let n = project(&narrow, off_axis).x.abs();
        let w = project(&wide, off_axis).x.abs();
        assert!(w < n, "the wider cone did not frame more: {w} against {n}");
    }

    #[test]
    fn the_six_faces_of_a_point_light_cover_every_direction() {
        // Six 90° frustums meeting without a gap. A gap is a wedge of the
        // scene that casts no shadow, and it moves with the light.
        let position = glam::Vec3::new(0.5, 1.0, 1.0);
        let big = Aabb::new(glam::Vec3::splat(-20.0), glam::Vec3::splat(20.0));
        let mut probes = Vec::new();
        for i in 0..200 {
            // A spread of directions from a deterministic spiral, so the test
            // is not testing a random number generator.
            let t = i as f32 / 200.0;
            let phi = t * std::f32::consts::TAU * 7.0;
            let z = 1.0 - 2.0 * t;
            let r = (1.0 - z * z).max(0.0).sqrt();
            probes.push(position + glam::Vec3::new(r * phi.cos(), r * phi.sin(), z) * 5.0);
        }
        for probe in probes {
            let seen = (0..6).any(|face| {
                point_view(position, face, big)
                    .map(|view| inside(project(&view, probe)))
                    .unwrap_or(false)
            });
            assert!(seen, "no face of the cube sees {probe}");
        }
    }

    #[test]
    fn the_near_plane_is_not_a_millimetre() {
        // Reversed-Z puts its precision near the near plane, so a near plane
        // at a millimetre spends most of the buffer on the first centimetre —
        // which is exactly where nothing is.
        let view = spot_view(
            glam::Vec3::new(0.0, 0.0, 8.0),
            glam::Vec3::NEG_Z,
            1.0,
            scene(),
        )
        .expect("a scene");
        assert!(view.near >= MIN_NEAR * 0.99, "near is {}", view.near);
        assert!(view.far > view.near);
    }

    #[test]
    fn a_light_pointing_along_the_up_axis_still_has_a_frame() {
        // The degenerate case a fixed up vector produces, and the commonest
        // light in any scene: a sun pointing straight down.
        for direction in [glam::Vec3::Z, glam::Vec3::NEG_Z] {
            let view = sun_view(direction, scene()).expect("a scene");
            for corner in scene().corners() {
                let p = project(&view, corner);
                assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
                assert!(inside(p), "{direction} left {corner} at {p}");
            }
        }
    }
}
