//! The viewport camera.
//!
//! Three decisions here are made once and cannot practically be revisited later,
//! because reversing any of them means touching every shader and every transform
//! path in the application.
//!
//! **f64 state.** Pivot, orientation and distance are double precision. Positions
//! inside a mesh are `f32` in object space, which is plenty; what needs range is
//! the world, and a scene placed 10 km from the origin visibly jitters if the
//! camera is single precision.
//!
//! **Camera-relative rendering.** The view matrix carries rotation only. Object
//! transforms are rebased onto the camera *in f64* before being handed to the
//! GPU as `f32`, so the numbers a shader sees are always small regardless of
//! where the scene sits. This costs nothing on day one and is a rewrite later.
//!
//! **Reversed-Z with an infinite far plane.** Depth 1 at the near plane falling
//! toward 0 at infinity. Combined with a float depth buffer, precision is close
//! to uniform in world space, which is what lets one scene hold a 1 mm bevel and
//! a 10 km backdrop without depth fighting. It also removes the far plane as a
//! thing users can be surprised by.

use glam::{DMat4, DQuat, DVec3, Mat4, Vec3};

/// How orbiting behaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrbitStyle {
    /// Keeps world +Z up. What modellers overwhelmingly expect, and what makes
    /// a scene stay level after a long editing session.
    Turntable,
    /// Free rotation with no preferred up. A minority preference, but a strong
    /// one among users coming from certain tools.
    Trackball,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Perspective,
    Orthographic,
}

/// One of the six axis views the navigation gizmo snaps to.
///
/// Named for the side the camera moves *to*: `PosX` puts the eye on the `+X`
/// side looking back at the pivot, which is what a user means when they click
/// the `+X` ball.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewAxis {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl ViewAxis {
    /// All six, in a stable order: `+X, -X, +Y, -Y, +Z, -Z`.
    pub const ALL: [ViewAxis; 6] = [
        ViewAxis::PosX,
        ViewAxis::NegX,
        ViewAxis::PosY,
        ViewAxis::NegY,
        ViewAxis::PosZ,
        ViewAxis::NegZ,
    ];

    /// The direction from the pivot toward the eye.
    pub fn direction(self) -> DVec3 {
        match self {
            ViewAxis::PosX => DVec3::X,
            ViewAxis::NegX => DVec3::NEG_X,
            ViewAxis::PosY => DVec3::Y,
            ViewAxis::NegY => DVec3::NEG_Y,
            ViewAxis::PosZ => DVec3::Z,
            ViewAxis::NegZ => DVec3::NEG_Z,
        }
    }

    /// Whether this is the positive end of its axis, which is what decides
    /// whether the gizmo draws it filled and labelled.
    pub fn is_positive(self) -> bool {
        matches!(self, ViewAxis::PosX | ViewAxis::PosY | ViewAxis::PosZ)
    }

    /// `0` for X, `1` for Y, `2` for Z — the index a colour table is keyed on.
    pub fn axis_index(self) -> usize {
        match self {
            ViewAxis::PosX | ViewAxis::NegX => 0,
            ViewAxis::PosY | ViewAxis::NegY => 1,
            ViewAxis::PosZ | ViewAxis::NegZ => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ViewAxis::PosX | ViewAxis::NegX => "X",
            ViewAxis::PosY | ViewAxis::NegY => "Y",
            ViewAxis::PosZ | ViewAxis::NegZ => "Z",
        }
    }
}

/// An orbit camera.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// The point orbit, pan, and dolly are all relative to.
    pub pivot: DVec3,
    pub rotation: DQuat,
    /// Distance from `pivot` to the eye. Always positive.
    pub distance: f64,
    /// Vertical field of view in radians.
    pub fov_y: f64,
    pub projection: Projection,
    pub orbit_style: OrbitStyle,
}

impl Default for Camera {
    fn default() -> Self {
        let mut c = Self {
            pivot: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            distance: 8.0,
            fov_y: 45f64.to_radians(),
            projection: Projection::Perspective,
            orbit_style: OrbitStyle::Turntable,
        };
        // Identity looks straight down -Z, i.e. a top view. Start at a three
        // quarter view instead, which is what a modeller wants on a new scene.
        c.orbit(-0.6, 1.1);
        c
    }
}

impl Camera {
    /// Smallest allowed distance. Zero would make orbiting undefined and the
    /// near plane collapse, and a UI drag can reach it.
    const MIN_DISTANCE: f64 = 1e-4;

    /// Eye position in world space.
    #[inline]
    pub fn eye(&self) -> DVec3 {
        self.pivot + self.rotation * (DVec3::Z * self.distance)
    }

    /// Unit vector from the eye toward the pivot.
    #[inline]
    pub fn forward(&self) -> DVec3 {
        self.rotation * DVec3::NEG_Z
    }

    #[inline]
    pub fn right(&self) -> DVec3 {
        self.rotation * DVec3::X
    }

    #[inline]
    pub fn up(&self) -> DVec3 {
        self.rotation * DVec3::Y
    }

    /// Near plane distance, derived from the orbit distance.
    ///
    /// A fixed near plane is wrong at both ends: it clips geometry when the user
    /// zooms in on a detail, and wastes precision when they pull back. Scaling it
    /// with distance keeps roughly the same relative precision at every zoom.
    #[inline]
    pub fn near(&self) -> f64 {
        (self.distance * 1e-4).max(1e-6)
    }

    /// Rotates about the pivot. Deltas are in radians.
    pub fn orbit(&mut self, yaw: f64, pitch: f64) {
        match self.orbit_style {
            OrbitStyle::Turntable => {
                // Yaw pre-multiplied (world space) and pitch post-multiplied
                // (camera space). Doing both in camera space is what makes naive
                // orbit cameras slowly roll until the horizon is crooked.
                self.rotation = DQuat::from_rotation_z(yaw) * self.rotation;
                self.rotation *= DQuat::from_rotation_x(pitch);
            }
            OrbitStyle::Trackball => {
                self.rotation *= DQuat::from_rotation_y(yaw) * DQuat::from_rotation_x(pitch);
            }
        }
        self.rotation = self.rotation.normalize();
    }

    /// Turns the head, the eye staying where it is: walking's look, where
    /// orbiting turns about the pivot. Yaw about the world's up, pitch about
    /// the view's right, in radians; positive pitch looks up. A pitch that
    /// would tip the view over the top is not taken.
    pub fn look(&mut self, yaw: f64, pitch: f64) {
        let eye = self.eye();
        let turned = (DQuat::from_rotation_z(yaw) * self.rotation).normalize();
        let tilted = (turned * DQuat::from_rotation_x(pitch)).normalize();
        self.rotation = if (tilted * DVec3::Y).z > 0.01 {
            tilted
        } else {
            turned
        };
        self.pivot = eye - self.rotation * (DVec3::Z * self.distance);
    }

    /// Translates the pivot in the view plane.
    ///
    /// Deltas are a fraction of viewport height. Scaling by distance (and by the
    /// field of view) is what makes panning feel identical whether zoomed into a
    /// bolt or looking at a building — without it, panning is unusably fast when
    /// far away and unusably slow when close.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        let world_per_unit = match self.projection {
            Projection::Perspective => 2.0 * self.distance * (self.fov_y * 0.5).tan(),
            Projection::Orthographic => self.ortho_height(),
        };
        self.pivot += (self.right() * -dx + self.up() * dy) * world_per_unit;
    }

    /// Moves the eye toward or away from the pivot.
    ///
    /// Geometric rather than linear, so each notch of the wheel covers the same
    /// *proportion* of the remaining distance and zoom never stalls or overshoots.
    pub fn dolly(&mut self, amount: f64) {
        self.distance = (self.distance * 1.1f64.powf(-amount)).max(Self::MIN_DISTANCE);
    }

    /// A framing that looks along one of the six world axes.
    ///
    /// `axis` names the side the **eye** moves to, not the direction of view:
    /// `PosZ` is the top view, looking down. Read the other way round it silently
    /// reverses which of two stacked objects is in front, and every symmetric
    /// test still passes.
    ///
    /// The pivot and the distance are kept, so an axis snap changes where the
    /// camera is looking *from* and nothing else — the subject stays the same
    /// size and in the same place, which is what makes the six views usable as
    /// a set rather than as six separate jumps.
    ///
    /// The up direction is chosen per axis rather than fixed: world `+Z` is up
    /// for the four side views, but a top view has no meaningful "up" along `Z`
    /// and needs `+Y` instead. Using one vector for all six leaves the top and
    /// bottom views degenerate — the classic gimbal collapse, which shows up as
    /// a top view rolled to an arbitrary angle.
    pub fn looking_along(&self, axis: ViewAxis) -> Camera {
        let toward_eye = axis.direction();
        // A top or bottom view: `Z` cannot also be up.
        let up_reference = if toward_eye.dot(DVec3::Z).abs() > 0.99 {
            DVec3::Y
        } else {
            DVec3::Z
        };
        let right = up_reference.cross(toward_eye).normalize();
        let up = toward_eye.cross(right);
        let basis = glam::DMat3::from_cols(right, up, toward_eye);
        Camera {
            rotation: DQuat::from_mat3(&basis).normalize(),
            ..*self
        }
    }

    /// Moves the pivot to a depth along the view axis without moving the eye.
    ///
    /// **Auto-depth**, the single most valuable navigation feature there is.
    /// Orbiting about a fixed origin while working close to a surface swings the
    /// model wildly for a small mouse movement — the drunk feeling every
    /// hand-rolled 3D viewport has. Orbiting about what is under the cursor
    /// removes it entirely.
    ///
    /// The pivot goes to the given depth **along the view direction**, not to the
    /// hit point itself. Moving it sideways to the exact hit would drag the eye
    /// with it and the view would jump the instant the drag began, which reads as
    /// a bug rather than as a feature.
    pub fn set_pivot_depth(&mut self, depth: f64) {
        let eye = self.eye();
        let d = depth.max(Self::MIN_DISTANCE);
        self.distance = d;
        self.pivot = eye + self.forward() * d;
    }

    /// The depth of a world point along the view axis, or `None` if it is behind
    /// the eye.
    pub fn depth_of(&self, point: DVec3) -> Option<f64> {
        let d = (point - self.eye()).dot(self.forward());
        (d > Self::MIN_DISTANCE).then_some(d)
    }

    /// Dollies toward a direction rather than toward the pivot.
    ///
    /// Zoom-to-cursor: the point under the pointer stays under it as the view
    /// closes in, so a user zooms into a detail by pointing at it instead of
    /// alternating zoom and pan. `direction` is a unit vector from the eye,
    /// normally the ray through the cursor.
    pub fn dolly_towards(&mut self, amount: f64, direction: DVec3) {
        let before = self.distance;
        self.dolly(amount);
        let travelled = before - self.distance;
        if travelled.abs() < f64::EPSILON {
            return;
        }
        // Slide the pivot sideways by as much as the eye advanced *off* the view
        // axis. With `direction` equal to the view direction this is exactly
        // zero, so plain dolly stays plain.
        let along = direction.dot(self.forward());
        let sideways = direction - self.forward() * along;
        self.pivot += sideways * travelled;
    }

    /// Height of the orthographic view volume, chosen so switching projections
    /// keeps the framing at the pivot unchanged.
    #[inline]
    pub fn ortho_height(&self) -> f64 {
        2.0 * self.distance * (self.fov_y * 0.5).tan()
    }

    /// Frames a bounding sphere, leaving a small margin.
    ///
    /// `aspect` is width divided by height. It is required rather than optional
    /// because framing on the vertical field of view alone clips the subject
    /// horizontally the moment a viewport is taller than it is wide — which is
    /// exactly what happens when side panels take most of the width. The
    /// constraint is whichever axis is *tighter*.
    pub fn frame_sphere(&mut self, center: DVec3, radius: f64, aspect: f64) {
        self.pivot = center;
        let r = radius.max(1e-6);
        let aspect = if aspect.is_finite() && aspect > 1e-6 {
            aspect
        } else {
            1.0
        };
        let half_v = self.fov_y * 0.5;
        let half_h = (half_v.tan() * aspect).atan();
        let half = half_v.min(half_h);
        self.distance = (r / half.sin()) * 1.15;
    }

    /// View matrix with the translation removed.
    ///
    /// Used together with [`Camera::model_matrix_relative`]: the camera sits at
    /// the origin and the world is moved to meet it.
    pub fn view_rotation(&self) -> Mat4 {
        DMat4::from_quat(self.rotation.inverse()).as_mat4()
    }

    /// Rebases a world-space model transform onto the camera, in f64, then
    /// narrows to `f32` for the GPU.
    ///
    /// The subtraction is the entire point: `model_translation - eye` is small
    /// even when both operands are huge, so the `f32` the shader receives has
    /// full relative precision.
    pub fn model_matrix_relative(&self, model: &glam::DAffine3) -> Mat4 {
        let mut rebased = *model;
        rebased.translation -= self.eye();
        DMat4::from(rebased).as_mat4()
    }

    /// Projection matrix, reversed-Z. Depth is 1 at the near plane.
    ///
    /// For perspective this is the infinite-far form: there is no far plane to
    /// tune and nothing to clip against in the distance.
    pub fn projection_matrix(&self, aspect: f64) -> Mat4 {
        let aspect = if aspect.is_finite() && aspect > 1e-6 {
            aspect
        } else {
            1.0
        };
        match self.projection {
            Projection::Perspective => {
                let f = 1.0 / (self.fov_y * 0.5).tan();
                let n = self.near();
                // Column-major. Gives z_clip = n and w_clip = -z_view, so
                // depth = n / -z_view: 1 at the near plane, tending to 0.
                DMat4::from_cols(
                    glam::DVec4::new(f / aspect, 0.0, 0.0, 0.0),
                    glam::DVec4::new(0.0, f, 0.0, 0.0),
                    glam::DVec4::new(0.0, 0.0, 0.0, -1.0),
                    glam::DVec4::new(0.0, 0.0, n, 0.0),
                )
                .as_mat4()
            }
            Projection::Orthographic => {
                // Orthographic has no perspective divide to exploit, so it needs
                // a real far plane. Deriving it from the distance keeps the
                // depth range tight around whatever is being looked at.
                let h = self.ortho_height() * 0.5;
                let w = h * aspect;
                let near = -self.distance * 10.0;
                let far = self.distance * 10.0;
                let d = far - near;
                DMat4::from_cols(
                    glam::DVec4::new(1.0 / w, 0.0, 0.0, 0.0),
                    glam::DVec4::new(0.0, 1.0 / h, 0.0, 0.0),
                    glam::DVec4::new(0.0, 0.0, 1.0 / d, 0.0),
                    glam::DVec4::new(0.0, 0.0, far / d, 1.0),
                )
                .as_mat4()
            }
        }
    }

    /// Combined view-projection for camera-relative geometry.
    pub fn view_projection(&self, aspect: f64) -> Mat4 {
        self.projection_matrix(aspect) * self.view_rotation()
    }

    /// A world-space ray through a viewport pixel, expressed relative to the eye.
    ///
    /// `ndc` is in `[-1, 1]` with +Y up.
    pub fn ray_direction(&self, ndc: glam::Vec2, aspect: f64) -> Vec3 {
        let t = (self.fov_y * 0.5).tan() as f32;
        let dir_camera = Vec3::new(ndc.x * t * aspect as f32, ndc.y * t, -1.0);
        let rot = glam::Quat::from_xyzw(
            self.rotation.x as f32,
            self.rotation.y as f32,
            self.rotation.z as f32,
            self.rotation.w as f32,
        );
        (rot * dir_camera).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DAffine3, Vec4};

    /// Depth of a point given in *camera-relative* world space.
    fn depth_of(cam: &Camera, world_relative: DVec3, aspect: f64) -> f32 {
        let clip: Vec4 = cam.view_projection(aspect) * world_relative.as_vec3().extend(1.0);
        clip.z / clip.w
    }

    #[test]
    fn eye_sits_at_distance_from_pivot() {
        let c = Camera::default();
        assert!((c.eye() - c.pivot).length() - c.distance < 1e-9);
    }

    #[test]
    fn forward_points_at_the_pivot() {
        let c = Camera::default();
        let to_pivot = (c.pivot - c.eye()).normalize();
        assert!((c.forward() - to_pivot).length() < 1e-9);
    }

    #[test]
    fn orbit_preserves_distance() {
        let mut c = Camera::default();
        let d = c.distance;
        for _ in 0..50 {
            c.orbit(0.31, -0.17);
            assert!((c.distance - d).abs() < 1e-12);
            assert!((c.eye() - c.pivot).length() - d < 1e-9);
        }
    }

    #[test]
    fn turntable_keeps_the_horizon_level() {
        // The failure this guards: applying yaw in camera space instead of world
        // space slowly rolls the camera until the scene looks crooked. The right
        // vector must stay perpendicular to world up.
        let mut c = Camera::default();
        for _ in 0..200 {
            c.orbit(0.37, 0.21);
            assert!(
                c.right().dot(DVec3::Z).abs() < 1e-9,
                "camera rolled: right = {:?}",
                c.right()
            );
        }
    }

    #[test]
    fn trackball_is_allowed_to_roll() {
        let mut c = Camera {
            orbit_style: OrbitStyle::Trackball,
            ..Camera::default()
        };
        for _ in 0..20 {
            c.orbit(0.5, 0.5);
        }
        assert!(
            c.right().dot(DVec3::Z).abs() > 1e-6,
            "trackball should not be constrained"
        );
    }

    #[test]
    fn dolly_is_geometric_and_never_reaches_zero() {
        let mut c = Camera::default();
        let start = c.distance;
        c.dolly(1.0);
        let once = c.distance;
        c.dolly(1.0);
        // Equal steps must multiply by the same factor, not subtract the same
        // amount, or zooming stalls when close and overshoots when far.
        assert!(((once / start) - (c.distance / once)).abs() < 1e-12);

        for _ in 0..10_000 {
            c.dolly(1.0);
        }
        assert!(c.distance > 0.0, "distance must stay positive");
    }

    #[test]
    fn pan_scales_with_distance() {
        let mut near = Camera {
            distance: 1.0,
            ..Camera::default()
        };
        let mut far = Camera {
            distance: 100.0,
            ..Camera::default()
        };
        near.pan(0.1, 0.0);
        far.pan(0.1, 0.0);
        let near_moved = (near.pivot
            - Camera {
                distance: 1.0,
                ..Camera::default()
            }
            .pivot)
            .length();
        let far_moved = (far.pivot
            - Camera {
                distance: 100.0,
                ..Camera::default()
            }
            .pivot)
            .length();
        assert!(
            (far_moved / near_moved - 100.0).abs() < 1e-6,
            "panning must cover the same fraction of the screen at any zoom"
        );
    }

    #[test]
    fn reversed_z_puts_one_at_the_near_plane() {
        let c = Camera::default();
        let n = c.near();
        // A point exactly on the near plane, in camera-relative world space.
        let p = c.forward() * n;
        let d = depth_of(&c, p, 16.0 / 9.0);
        assert!(
            (d - 1.0).abs() < 1e-4,
            "near plane depth should be 1, got {d}"
        );
    }

    #[test]
    fn reversed_z_decreases_monotonically_toward_zero() {
        let c = Camera::default();
        let aspect = 16.0 / 9.0;
        let mut last = f32::INFINITY;
        for exp in 0..12 {
            let dist = c.near() * 10f64.powi(exp);
            let d = depth_of(&c, c.forward() * dist, aspect);
            assert!(d < last, "depth must decrease with distance (exp {exp})");
            assert!(
                (0.0..=1.0).contains(&d),
                "depth {d} out of range at exp {exp}"
            );
            last = d;
        }
        assert!(last < 1e-6, "depth should approach 0 far away, got {last}");
    }

    #[test]
    fn there_is_no_far_plane_to_clip_against() {
        // 10,000 km away must still be in front of the camera rather than culled.
        let c = Camera::default();
        let d = depth_of(&c, c.forward() * 1.0e7, 1.0);
        assert!(
            d > 0.0 && d < 1e-9,
            "distant geometry should survive, got depth {d}"
        );
    }

    #[test]
    fn geometry_behind_the_camera_is_clipped() {
        let c = Camera::default();
        let clip: Vec4 = c.view_projection(1.0) * (c.forward() * -5.0).as_vec3().extend(1.0);
        assert!(clip.w < 0.0, "points behind the eye must have negative w");
    }

    #[test]
    fn orthographic_depth_is_also_reversed() {
        let c = Camera {
            projection: Projection::Orthographic,
            ..Camera::default()
        };
        let near = depth_of(&c, c.forward() * 1.0, 1.0);
        let far = depth_of(&c, c.forward() * 50.0, 1.0);
        assert!(
            near > far,
            "reversed-Z: nearer geometry must have greater depth"
        );
        assert!((0.0..=1.0).contains(&far));
    }

    #[test]
    fn camera_relative_rebasing_keeps_precision_far_from_the_origin() {
        // The bug this prevents: at 10 km with f32 world coordinates, a 1 mm
        // feature lands below the representable step and the model visibly
        // jitters as the camera moves.
        let far_away = DVec3::new(10_000.0, 10_000.0, 10_000.0);
        let c = Camera {
            pivot: far_away,
            ..Camera::default()
        };

        let model = DAffine3::from_translation(far_away + DVec3::new(0.001, 0.0, 0.0));
        let m = c.model_matrix_relative(&model);

        // Rebased, the translation is a small number and the millimetre survives
        // essentially intact.
        let rebased = m.w_axis.truncate();
        let expected = (far_away + DVec3::new(0.001, 0.0, 0.0) - c.eye()).as_vec3();
        let rebased_err = (rebased - expected).length();
        assert!(
            rebased_err < 1e-5,
            "rebased error {rebased_err} is too large"
        );

        // Contrast: the same offset applied in f32 *world* space. At 10 km an
        // f32 step is about 1 mm, so the offset does not vanish — it snaps to the
        // nearest representable value. That quantisation is the jitter: as the
        // camera moves, the feature flicks between adjacent steps.
        let naive = (far_away.as_vec3() + Vec3::new(0.001, 0.0, 0.0)) - far_away.as_vec3();
        let naive_err = (naive.x - 0.001).abs();
        assert!(
            naive_err / 0.001 > 0.01,
            "f32 world coordinates should quantise the millimetre noticeably, \
             but the error was only {naive_err}"
        );
        assert!(
            rebased_err < naive_err,
            "rebasing must be more accurate than working in f32 world space"
        );
    }

    #[test]
    fn frame_sphere_fits_the_sphere_in_view() {
        let mut c = Camera::default();
        c.frame_sphere(DVec3::new(1.0, 2.0, 3.0), 5.0, 16.0 / 9.0);
        assert_eq!(c.pivot, DVec3::new(1.0, 2.0, 3.0));
        // The half-angle to the sphere's silhouette must be inside the fov.
        let half_angle = (5.0f64 / c.distance).asin();
        assert!(half_angle < c.fov_y * 0.5, "sphere should fit with margin");
    }

    #[test]
    fn framing_fits_the_sphere_on_both_axes_at_any_aspect() {
        // The bug this guards: framing on the vertical field of view alone clips
        // the subject horizontally as soon as a viewport is taller than it is
        // wide, which is what side panels do to it.
        for aspect in [0.25, 0.45, 1.0, 1.13, 2.7, 8.0] {
            let mut c = Camera::default();
            let r = 5.0;
            c.frame_sphere(DVec3::ZERO, r, aspect);

            let half_v = c.fov_y * 0.5;
            let half_h = (half_v.tan() * aspect).atan();
            let subtended = (r / c.distance).asin();
            assert!(
                subtended < half_v && subtended < half_h,
                "at aspect {aspect} the sphere subtends {subtended} but the half                  angles are v={half_v} h={half_h}"
            );
        }
    }

    #[test]
    fn a_narrow_viewport_pulls_the_camera_further_back() {
        // The concrete consequence: a tall, narrow viewport needs more distance
        // for the same subject.
        let mut wide = Camera::default();
        wide.frame_sphere(DVec3::ZERO, 1.0, 2.0);
        let mut narrow = Camera::default();
        narrow.frame_sphere(DVec3::ZERO, 1.0, 0.4);
        assert!(
            narrow.distance > wide.distance * 1.5,
            "narrow {} should be much further than wide {}",
            narrow.distance,
            wide.distance
        );
    }

    #[test]
    fn frame_sphere_handles_degenerate_input() {
        let mut c = Camera::default();
        c.frame_sphere(DVec3::ZERO, 0.0, 1.0);
        assert!(c.distance.is_finite() && c.distance > 0.0);
        // A zero-width viewport happens during a resize or with a collapsed panel.
        for aspect in [0.0, -1.0, f64::NAN] {
            let mut c = Camera::default();
            c.frame_sphere(DVec3::ZERO, 1.0, aspect);
            assert!(
                c.distance.is_finite() && c.distance > 0.0,
                "aspect {aspect}"
            );
        }
    }

    #[test]
    fn ray_through_the_centre_is_the_forward_vector() {
        let c = Camera::default();
        let r = c.ray_direction(glam::Vec2::ZERO, 1.6);
        assert!((r - c.forward().as_vec3()).length() < 1e-6);
    }

    #[test]
    fn rays_spread_with_the_field_of_view() {
        let c = Camera::default();
        let edge = c.ray_direction(glam::Vec2::new(0.0, 1.0), 1.0);
        let angle = edge.dot(c.forward().as_vec3()).acos() as f64;
        assert!(
            (angle - c.fov_y * 0.5).abs() < 1e-5,
            "top-edge ray should sit at half the fov"
        );
    }

    #[test]
    fn degenerate_aspect_does_not_produce_nan() {
        // A viewport can be zero pixels wide during a resize or when a panel is
        // collapsed; a NaN matrix there blanks the whole frame.
        let c = Camera::default();
        for aspect in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let m = c.projection_matrix(aspect);
            assert!(
                m.to_cols_array().iter().all(|v| v.is_finite()),
                "aspect {aspect}"
            );
        }
    }

    #[test]
    fn auto_depth_moves_the_pivot_without_moving_the_eye() {
        // The whole point: the orbit centre changes, the picture does not. A
        // pivot change that shifted the view would announce itself as a jump the
        // instant the user pressed the button.
        let mut c = Camera::default();
        let eye = c.eye();
        let forward = c.forward();

        c.set_pivot_depth(2.5);
        assert!((c.eye() - eye).length() < 1e-9, "the eye moved");
        assert!((c.forward() - forward).length() < 1e-9, "the view turned");
        assert!((c.distance - 2.5).abs() < 1e-9);
        assert!((c.pivot - (eye + forward * 2.5)).length() < 1e-9);
    }

    #[test]
    fn orbiting_after_auto_depth_turns_about_the_new_centre() {
        let mut c = Camera::default();
        c.set_pivot_depth(2.0);
        let centre = c.pivot;
        c.orbit(0.5, 0.2);
        assert!(
            (c.pivot - centre).length() < 1e-9,
            "orbit must leave the pivot alone"
        );
        assert!(
            ((c.eye() - centre).length() - 2.0).abs() < 1e-9,
            "and keep the eye at the same radius from it"
        );
    }

    #[test]
    fn auto_depth_refuses_to_collapse_onto_the_eye() {
        // A ray that hits something touching the near plane would otherwise set
        // the distance to zero, which makes orbit undefined and the near plane
        // collapse.
        let mut c = Camera::default();
        c.set_pivot_depth(0.0);
        assert!(c.distance > 0.0);
        c.set_pivot_depth(-5.0);
        assert!(c.distance > 0.0);
        assert!(c.near() > 0.0);
    }

    #[test]
    fn a_point_behind_the_eye_has_no_depth() {
        let c = Camera::default();
        let behind = c.eye() - c.forward() * 3.0;
        assert_eq!(c.depth_of(behind), None);
        let ahead = c.eye() + c.forward() * 3.0;
        assert!((c.depth_of(ahead).unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn zooming_along_the_view_axis_is_a_plain_dolly() {
        // The property that keeps wheel-zoom at the centre of the screen exactly
        // as it was: pointing straight ahead must not slide the pivot at all.
        let mut cursor = Camera::default();
        let mut plain = cursor;
        let forward = cursor.forward();

        cursor.dolly_towards(1.0, forward);
        plain.dolly(1.0);
        assert!((cursor.pivot - plain.pivot).length() < 1e-9);
        assert!((cursor.distance - plain.distance).abs() < 1e-12);
    }

    #[test]
    fn zooming_toward_the_cursor_brings_that_point_closer_to_the_centre() {
        // Zoom-to-cursor, stated as what the user sees: the thing they are
        // pointing at ends up nearer the middle of the screen than it was.
        let mut c = Camera::default();
        let eye = c.eye();
        // A direction off to one side of the view.
        let direction = (c.forward() + c.right() * 0.3).normalize();
        let target = eye + direction * c.distance;

        let offset_before = {
            let to = target - c.pivot;
            (to - c.forward() * to.dot(c.forward())).length()
        };
        c.dolly_towards(3.0, direction);
        let offset_after = {
            let to = target - c.pivot;
            (to - c.forward() * to.dot(c.forward())).length()
        };
        assert!(
            offset_after < offset_before,
            "the target went from {offset_before} to {offset_after} off-axis"
        );
        assert!(c.distance < 8.0, "and the view did close in");
    }
}
