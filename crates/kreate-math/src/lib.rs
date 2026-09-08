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

pub use glam;
pub use glam::{
    Affine3A, DAffine3, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4, Mat3, Mat3A, Mat4, Quat, Vec2,
    Vec3, Vec3A, Vec4,
};

mod aabb;
mod axis;
mod bvh;
mod plane;
mod point_tree;
mod ray;
mod transform;

pub use aabb::Aabb;
pub use axis::{Axis, AxisConvention, Handedness, UpAxis};
pub use bvh::{Bvh, Hit};
pub use plane::Plane;
pub use point_tree::{Nearest, PointTree};
pub use ray::Ray;
pub use transform::{DTransform, Transform};

/// Comparison epsilon for positions expressed in metres.
pub const POS_EPSILON: f32 = 1e-6;

/// Returns `true` when `a` and `b` are within `eps` of each other.
#[inline]
pub fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}
