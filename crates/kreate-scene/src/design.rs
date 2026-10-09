//! What a scene keeps for being used as a design rather than a picture: the
//! unit it is drawn in, and — as they arrive — the measurements left on it,
//! the material it is made of, the plane it is cut by.
//!
//! The scene's, as Blender keeps its units: saved with the file, set by
//! operators so a change is one undo step.

pub use kreate_reflect::LengthUnit;

use kreate_math::{DQuat, DVec3};

/// A section view: the scene cut by a plane, the far side taken away, and
/// the cut capped so walls read as solid. What is drawn, not what is saved:
/// the meshes are untouched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Section {
    /// Which plane: across X, Y or Z, or 3 for the work plane (the 3D
    /// cursor's Z).
    pub axis: u32,
    /// How far along the plane's normal from the origin (or the cursor).
    pub offset: f64,
    /// Keep the other side.
    pub flip: bool,
    /// Close the cut.
    pub cap: bool,
}

impl Section {
    /// The plane in the world: a point on it, and the normal pointing at the
    /// side taken away.
    pub fn plane(&self, cursor: DVec3, cursor_rotation: DQuat) -> (DVec3, DVec3) {
        let (origin, normal) = match self.axis {
            0 => (DVec3::ZERO, DVec3::X),
            1 => (DVec3::ZERO, DVec3::Y),
            3 => (cursor, cursor_rotation * DVec3::Z),
            _ => (DVec3::ZERO, DVec3::Z),
        };
        let normal = if self.flip { -normal } else { normal };
        let along = if self.flip { -self.offset } else { self.offset };
        (origin + normal * along, normal)
    }
}

/// A dimension left on the scene: two points and the distance between
/// them, or three and the angle at the middle one. In the world, so it stays
/// where it was measured when what it measured moves — as a ruler laid on a
/// drawing does.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub points: Vec<DVec3>,
}

impl Measurement {
    /// The distance from the first point to the last, for two.
    pub fn length(&self) -> Option<f64> {
        match self.points.as_slice() {
            [a, b] => Some(a.distance(*b)),
            _ => None,
        }
    }

    /// The angle at the middle point, in radians, for three.
    pub fn angle(&self) -> Option<f64> {
        match self.points.as_slice() {
            [a, b, c] => {
                let (u, v) = (*a - *b, *c - *b);
                (u.length() > 0.0 && v.length() > 0.0).then(|| u.angle_between(v))
            }
            _ => None,
        }
    }

    /// Two or three finite points: anything else is not a measurement.
    pub fn is_valid(&self) -> bool {
        matches!(self.points.len(), 2 | 3) && self.points.iter().all(|p| p.is_finite())
    }
}

/// Water's density, kilograms per cubic metre: what a new scene assumes a
/// part is made of until it is told.
pub const DEFAULT_DENSITY: f64 = 1000.0;

/// The scene's design settings.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignSettings {
    /// The unit lengths are shown and typed in. Lengths are stored in metres
    /// whatever this says; see [`LengthUnit`].
    pub length_unit: LengthUnit,
    /// The dimensions left on the scene, in the order they were taken.
    pub measurements: Vec<Measurement>,
    /// What the parts are made of, in kilograms per cubic metre, for their
    /// mass: PLA is about 1240, aluminium 2700, steel 7850.
    pub density: f64,
    /// The section view, when it is on.
    pub section: Option<Section>,
}

impl Default for DesignSettings {
    fn default() -> Self {
        Self {
            length_unit: LengthUnit::default(),
            measurements: Vec::new(),
            density: DEFAULT_DENSITY,
            section: None,
        }
    }
}

impl DesignSettings {
    /// Nothing set: what a file without design settings reads as.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_measurement_reads_a_length_or_an_angle() {
        let d = Measurement {
            points: vec![DVec3::ZERO, DVec3::new(3.0, 4.0, 0.0)],
        };
        assert_eq!(d.length(), Some(5.0));
        assert_eq!(d.angle(), None);
        let a = Measurement {
            points: vec![DVec3::X, DVec3::ZERO, DVec3::Y],
        };
        assert!((a.angle().unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!(!Measurement {
            points: vec![DVec3::ZERO]
        }
        .is_valid());
    }
}
