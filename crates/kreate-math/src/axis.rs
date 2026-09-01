use glam::{Mat3, Vec3};

/// A signed principal axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl Axis {
    #[inline]
    pub fn as_vec3(self) -> Vec3 {
        match self {
            Axis::PosX => Vec3::X,
            Axis::NegX => Vec3::NEG_X,
            Axis::PosY => Vec3::Y,
            Axis::NegY => Vec3::NEG_Y,
            Axis::PosZ => Vec3::Z,
            Axis::NegZ => Vec3::NEG_Z,
        }
    }

    #[inline]
    pub fn flipped(self) -> Self {
        match self {
            Axis::PosX => Axis::NegX,
            Axis::NegX => Axis::PosX,
            Axis::PosY => Axis::NegY,
            Axis::NegY => Axis::PosY,
            Axis::PosZ => Axis::NegZ,
            Axis::NegZ => Axis::PosZ,
        }
    }
}

/// Which axis a coordinate system treats as "up".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UpAxis {
    /// Kreate's own convention, shared with Blender, USD, and most CAD.
    ZUp,
    /// glTF, Maya, Unity, and most realtime engines.
    YUp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handedness {
    Right,
    Left,
}

/// Converts between a file format's coordinate convention and Kreate's.
///
/// **This is applied at the import/export IR boundary and nowhere else.** Every
/// axis bug in every 3D application comes from applying the conversion twice, or
/// in the middle of a pipeline where half the data has already been converted.
///
/// A handedness flip also requires reversing face winding and flipping the
/// normal matrix; [`AxisConvention::flips_winding`] reports when the caller must
/// do so. Skipping it is the classic silent bug: geometry looks right until
/// backface culling or shading is switched on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisConvention {
    pub up: UpAxis,
    pub handedness: Handedness,
}

impl AxisConvention {
    /// Kreate's native convention: Z-up, -Y forward, right-handed.
    pub const KREATE: Self = Self {
        up: UpAxis::ZUp,
        handedness: Handedness::Right,
    };
    /// glTF 2.0: Y-up, right-handed.
    pub const GLTF: Self = Self {
        up: UpAxis::YUp,
        handedness: Handedness::Right,
    };

    /// Matrix taking a point *from* this convention *into* Kreate space.
    pub fn to_kreate(self) -> Mat3 {
        let m = match self.up {
            UpAxis::ZUp => Mat3::IDENTITY,
            // (x, y, z)_yup -> (x, -z, y)_zup
            UpAxis::YUp => Mat3::from_cols(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, -1.0, 0.0),
            ),
        };
        match self.handedness {
            Handedness::Right => m,
            // Mirror one axis; the caller must reverse winding to match.
            Handedness::Left => m * Mat3::from_diagonal(Vec3::new(1.0, 1.0, -1.0)),
        }
    }

    /// Matrix taking a point *from* Kreate space *into* this convention.
    pub fn from_kreate(self) -> Mat3 {
        self.to_kreate().inverse()
    }

    /// Whether converting through this convention reverses triangle winding.
    #[inline]
    pub fn flips_winding(self) -> bool {
        self.handedness == Handedness::Left
    }
}

impl Default for AxisConvention {
    fn default() -> Self {
        Self::KREATE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kreate_convention_is_identity() {
        assert_eq!(AxisConvention::KREATE.to_kreate(), Mat3::IDENTITY);
        assert!(!AxisConvention::KREATE.flips_winding());
    }

    #[test]
    fn gltf_up_becomes_kreate_up() {
        // glTF's +Y (up) must land on Kreate's +Z (up).
        let m = AxisConvention::GLTF.to_kreate();
        assert!((m * Vec3::Y - Vec3::Z).length() < 1e-6);
        // glTF's +Z (toward viewer) becomes Kreate's -Y (toward viewer).
        assert!((m * Vec3::Z - Vec3::NEG_Y).length() < 1e-6);
        assert!((m * Vec3::X - Vec3::X).length() < 1e-6);
    }

    #[test]
    fn conversion_roundtrips() {
        for conv in [AxisConvention::KREATE, AxisConvention::GLTF] {
            let p = Vec3::new(1.0, 2.0, 3.0);
            assert!((conv.from_kreate() * (conv.to_kreate() * p) - p).length() < 1e-6);
        }
    }

    #[test]
    fn right_handed_conversions_preserve_orientation() {
        // A right-handed -> right-handed change of basis must not mirror, or every
        // imported mesh would be inside out.
        assert!(AxisConvention::GLTF.to_kreate().determinant() > 0.0);
        assert!(!AxisConvention::GLTF.flips_winding());
    }

    #[test]
    fn left_handed_conversion_mirrors_and_reports_it() {
        let lh = AxisConvention {
            up: UpAxis::YUp,
            handedness: Handedness::Left,
        };
        assert!(lh.to_kreate().determinant() < 0.0);
        assert!(lh.flips_winding());
    }
}
