//! Foundational math for Kreate.
//!
//! # Conventions
//!
//! Kreate is **Z-up, -Y forward, right-handed**, and its base unit is the metre
//! (matching Blender, USD, and most CAD). Every file format gets exactly one
//! [`AxisConvention`] matrix applied at the import/export IR boundary and nowhere
//! else. A handedness flip must also reverse face winding — see `kreate-io`.
//!
//! # Precision
//!
//! Mesh positions are `f32` in **object-local** space; the mesh kernel is
//! deliberately not generic over its scalar type. World-space quantities that
//! must survive a large scene — object transforms and the camera — are `f64`,
//! and are converted to camera-relative space in `f64` before being downcast for
//! the GPU. Without that, a scene 10 km from the origin visibly jitters.
//!
//! # Colour
//!
//! The working space is **scene-linear with Rec.709 primaries**, and which space
//! a colour is in is carried in its type — [`Srgb`], [`DisplayRgb`],
//! [`SceneRgb`]. See [`color`] for why that is a type and not a convention.
//! Colour lives here, beside the axis convention and the unit of length, because
//! it is the same kind of thing: an agreement every crate above has to share,
//! and one that cannot be settled twice.
//!
//! # Time
//!
//! The same again, and for the same reason: a bare `f64` is either a frame or a
//! quantity of seconds, and the two differ by the scene's rate. [`SceneTime`]
//! carries the frame it was authored as, the rate, and the seconds that follow
//! from both. See [`time`] for why the frame is the stored half.

pub use glam;
pub use glam::{
    Affine3A, DAffine3, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4, Mat3, Mat3A, Mat4, Quat, Vec2,
    Vec3, Vec3A, Vec4,
};

mod aabb;
mod agx;
mod axis;
mod bvh;
mod color;
mod plane;
mod point_tree;
mod ray;
mod time;
mod transform;

pub use aabb::Aabb;
/// The display transform. `MIN_EV`/`MAX_EV` are renamed on the way out:
/// they are AgX's exposure range and nothing else's, and a bare `MIN_EV`
/// at the crate root would read as a property of the working space.
pub use agx::{DisplayTransform, ViewTransform, MAX_EV as AGX_MAX_EV, MIN_EV as AGX_MIN_EV};
pub use axis::{Axis, AxisConvention, Handedness, UpAxis};
pub use bvh::{Bvh, Hit};
pub use color::{linear_to_srgb, srgb_to_linear, DisplayRgb, SceneRgb, Srgb};
pub use plane::Plane;
pub use point_tree::{Nearest, PointTree};
pub use ray::Ray;
pub use time::SceneTime;
pub use transform::{DTransform, Transform};

/// Comparison epsilon for positions expressed in metres.
pub const POS_EPSILON: f32 = 1e-6;

/// Returns `true` when `a` and `b` are within `eps` of each other.
#[inline]
pub fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}
