use glam::{Affine3A, DAffine3, DQuat, DVec3, Mat3, Quat, Vec3};

/// A TRS transform.
///
/// Stored decomposed rather than as a matrix. A matrix cannot answer "what is
/// this object's rotation" without an ambiguous decomposition, so gimbal-aware
/// rotation UI, non-uniform-scale semantics, and clean interpolation all need
/// the components kept apart. World matrices are a *cache* derived from this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    #[inline]
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    #[inline]
    pub fn to_affine(&self) -> Affine3A {
        Affine3A::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    #[inline]
    pub fn from_affine(m: Affine3A) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        let rotation = if rotation.is_finite() {
            rotation
        } else {
            Quat::IDENTITY
        };
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// The matrix that transforms normals: the inverse-transpose of the linear part.
    ///
    /// Only equal to the rotation when the scale is uniform. Getting this wrong
    /// under non-uniform scale tilts every normal, which reads as broken shading
    /// rather than as a transform bug.
    #[inline]
    pub fn normal_matrix(&self) -> Mat3 {
        Mat3::from(self.to_affine().matrix3).inverse().transpose()
    }

    /// `true` when the transform flips handedness, in which case face winding
    /// must be reversed for backface culling to stay correct.
    #[inline]
    pub fn is_mirrored(&self) -> bool {
        self.scale.x * self.scale.y * self.scale.z < 0.0
    }

    #[inline]
    #[must_use]
    pub fn mul_transform(&self, child: &Self) -> Self {
        Self::from_affine(self.to_affine() * child.to_affine())
    }

    #[inline]
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.to_affine().transform_point3(p)
    }

    #[inline]
    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        self.to_affine().transform_vector3(v)
    }

    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self::from_affine(self.to_affine().inverse())
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn affine_roundtrip() {
        let t = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_z(FRAC_PI_2),
            scale: Vec3::new(2.0, 2.0, 2.0),
        };
        let back = Transform::from_affine(t.to_affine());
        assert!((back.translation - t.translation).length() < 1e-5);
        assert!((back.scale - t.scale).length() < 1e-5);
        assert!(back.rotation.dot(t.rotation).abs() > 1.0 - 1e-5);
    }

    #[test]
    fn inverse_undoes_transform() {
        let t = Transform {
            translation: Vec3::new(-3.0, 0.5, 7.0),
            rotation: Quat::from_rotation_y(0.7),
            scale: Vec3::splat(3.0),
        };
        let p = Vec3::new(1.0, -2.0, 0.25);
        let back = t.inverse().transform_point(t.transform_point(p));
        assert!((back - p).length() < 1e-4, "{back} != {p}");
    }

    #[test]
    fn normal_matrix_is_not_the_rotation_under_nonuniform_scale() {
        // A normal on a squashed sphere must not simply rotate with the object.
        let t = Transform {
            scale: Vec3::new(1.0, 1.0, 0.1),
            ..Transform::IDENTITY
        };
        let n = t.normal_matrix() * Vec3::new(0.0, 0.0, 1.0);
        assert!(
            n.z > 9.0,
            "expected the squashed axis to be amplified, got {n}"
        );
    }

    #[test]
    fn detects_mirroring() {
        assert!(!Transform::IDENTITY.is_mirrored());
        assert!(Transform::from_scale(Vec3::new(-1.0, 1.0, 1.0)).is_mirrored());
        // Two flips cancel and are a rotation, not a mirror.
        assert!(!Transform::from_scale(Vec3::new(-1.0, -1.0, 1.0)).is_mirrored());
    }
}

/// A TRS transform in double precision.
///
/// Object transforms are `f64` while mesh positions are `f32`. That split is
/// deliberate: what needs range is the *world*, not the interior of a mesh. A
/// scene placed 10 km from the origin needs double precision to stay stable,
/// while a 1 mm feature measured from its own object's origin does not — and
/// making the mesh generic over its scalar type would wreck both compile times
/// and API ergonomics for no gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DTransform {
    pub translation: DVec3,
    pub rotation: DQuat,
    pub scale: DVec3,
}

impl DTransform {
    pub const IDENTITY: Self = Self {
        translation: DVec3::ZERO,
        rotation: DQuat::IDENTITY,
        scale: DVec3::ONE,
    };

    #[inline]
    pub fn from_translation(translation: DVec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    #[inline]
    pub fn from_rotation(rotation: DQuat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    #[inline]
    pub fn from_scale(scale: DVec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    #[inline]
    pub fn to_affine(&self) -> DAffine3 {
        DAffine3::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    #[inline]
    pub fn from_affine(m: DAffine3) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        // A matrix scaled to nothing — a particle before it appears — has no
        // turn to recover, and a NaN one spread through every child, every
        // bound and every frame. Nothing is turned, then.
        let rotation = if rotation.is_finite() {
            rotation
        } else {
            DQuat::IDENTITY
        };
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// `true` when the transform flips handedness, so face winding must be
    /// reversed for backface culling to stay correct.
    #[inline]
    pub fn is_mirrored(&self) -> bool {
        self.scale.x * self.scale.y * self.scale.z < 0.0
    }

    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self::from_affine(self.to_affine().inverse())
    }

    #[inline]
    pub fn transform_point(&self, p: DVec3) -> DVec3 {
        self.to_affine().transform_point3(p)
    }

    /// Narrows to the single-precision form, for handing to the GPU *after*
    /// rebasing onto the camera.
    #[inline]
    pub fn as_transform(&self) -> Transform {
        Transform {
            translation: self.translation.as_vec3(),
            rotation: self.rotation.as_quat(),
            scale: self.scale.as_vec3(),
        }
    }
}

impl Default for DTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod dtransform_tests {
    use super::*;

    #[test]
    fn affine_roundtrip() {
        let t = DTransform {
            translation: DVec3::new(1.0, 2.0, 3.0),
            rotation: DQuat::from_rotation_z(0.7),
            scale: DVec3::splat(2.0),
        };
        let back = DTransform::from_affine(t.to_affine());
        assert!((back.translation - t.translation).length() < 1e-12);
        assert!((back.scale - t.scale).length() < 1e-12);
    }

    #[test]
    fn keeps_precision_far_from_the_origin() {
        // The reason object transforms are f64: at 10 km an f32 step is about
        // 1 mm, so a millimetre offset would quantise and the object would
        // visibly jitter as the camera moves.
        let far = DVec3::splat(10_000.0);
        let t = DTransform::from_translation(far);
        let moved = t.transform_point(DVec3::new(0.001, 0.0, 0.0));
        assert!((moved.x - 10_000.001).abs() < 1e-9, "got {}", moved.x);
    }

    #[test]
    fn inverse_undoes_the_transform() {
        let t = DTransform {
            translation: DVec3::new(-3.0, 0.5, 7.0),
            rotation: DQuat::from_rotation_y(0.7),
            scale: DVec3::splat(3.0),
        };
        let p = DVec3::new(1.0, -2.0, 0.25);
        assert!((t.inverse().transform_point(t.transform_point(p)) - p).length() < 1e-12);
    }

    #[test]
    fn detects_mirroring() {
        assert!(!DTransform::IDENTITY.is_mirrored());
        assert!(DTransform::from_scale(DVec3::new(-1.0, 1.0, 1.0)).is_mirrored());
        assert!(!DTransform::from_scale(DVec3::new(-1.0, -1.0, 1.0)).is_mirrored());
    }

    #[test]
    fn a_matrix_scaled_to_nothing_comes_apart_without_nan() {
        let m = DAffine3::from_scale_rotation_translation(
            DVec3::ZERO,
            DQuat::from_rotation_z(0.7),
            DVec3::new(1.0, 2.0, 3.0),
        );
        let t = DTransform::from_affine(m);
        assert!(t.rotation.is_finite() && t.scale == DVec3::ZERO);
        assert_eq!(t.translation, DVec3::new(1.0, 2.0, 3.0));
    }
}
