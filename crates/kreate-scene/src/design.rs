//! What a scene keeps for being used as a design rather than a picture: the
//! unit it is drawn in, and — as they arrive — the measurements left on it,
//! the material it is made of, the plane it is cut by.
//!
//! The scene's, as Blender keeps its units: saved with the file, set by
//! operators so a change is one undo step.

pub use kreate_reflect::LengthUnit;

/// The scene's design settings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DesignSettings {
    /// The unit lengths are shown and typed in. Lengths are stored in metres
    /// whatever this says; see [`LengthUnit`].
    pub length_unit: LengthUnit,
}

impl DesignSettings {
    /// Nothing set: what a file without design settings reads as.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}
