//! What an object is to a robot: a link, a joint, or a link's collision
//! shape — as a robot description (URDF) says them, kept so a robot can be
//! posed within its limits and written back out.
//!
//! # A joint is an empty that turns about its own Z
//!
//! A joint's object sits where the joint's origin is, turned so that the
//! joint's axis is its local Z, and its child link sits under it. Its
//! position — an angle for a revolute or continuous joint, a distance for a
//! prismatic one — is a turn about, or a slide along, that Z from its
//! [`Joint::rest`]. [`Joint::place`] is the one place that says how, and
//! [`Joint::position_of`] reads one back from a transform the user moved by
//! hand.

use kreate_math::{DQuat, DTransform, DVec3};

/// How a joint moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JointKind {
    #[default]
    Fixed,
    /// Turns, between its limits.
    Revolute,
    /// Turns, without limits.
    Continuous,
    /// Slides, between its limits.
    Prismatic,
    /// Moves every way: placed by hand.
    Floating,
    /// Slides in its plane: placed by hand.
    Planar,
}

impl JointKind {
    pub const ALL: [Self; 6] = [
        Self::Fixed,
        Self::Revolute,
        Self::Continuous,
        Self::Prismatic,
        Self::Floating,
        Self::Planar,
    ];

    /// URDF's name for it.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Revolute => "revolute",
            Self::Continuous => "continuous",
            Self::Prismatic => "prismatic",
            Self::Floating => "floating",
            Self::Planar => "planar",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.tag() == tag)
    }

    /// Whether it has one position to set: a turn or a slide.
    pub fn moves(self) -> bool {
        matches!(self, Self::Revolute | Self::Continuous | Self::Prismatic)
    }

    /// Whether its position is a distance rather than an angle.
    pub fn slides(self) -> bool {
        self == Self::Prismatic
    }

    /// Whether its limits hold it.
    pub fn limited(self) -> bool {
        matches!(self, Self::Revolute | Self::Prismatic)
    }
}

/// Another joint this one follows: `position = multiplier · theirs + offset`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mimic {
    /// The joint followed, by its object's name.
    pub joint: String,
    pub multiplier: f64,
    pub offset: f64,
}

/// A joint between two links.
#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    pub kind: JointKind,
    /// Where the joint's object sits at position zero, in its parent's
    /// space: the joint's origin, turned so its axis is local Z.
    pub rest: DTransform,
    /// Radians or metres.
    pub lower: f64,
    pub upper: f64,
    /// What the description says it can push and how fast: kept to be
    /// written back, not used here.
    pub effort: f64,
    pub velocity: f64,
    pub mimic: Option<Mimic>,
}

impl Default for Joint {
    fn default() -> Self {
        Self {
            kind: JointKind::Fixed,
            rest: DTransform::IDENTITY,
            lower: 0.0,
            upper: 0.0,
            effort: 0.0,
            velocity: 0.0,
            mimic: None,
        }
    }
}

impl Joint {
    /// `position` held to the limits, for a joint that has them.
    pub fn clamp(&self, position: f64) -> f64 {
        if self.kind.limited() && self.lower <= self.upper {
            position.clamp(self.lower, self.upper)
        } else {
            position
        }
    }

    /// The joint's object's transform at `position`.
    pub fn place(&self, position: f64) -> DTransform {
        let mut t = self.rest;
        match self.kind {
            JointKind::Revolute | JointKind::Continuous => {
                t.rotation = (self.rest.rotation * DQuat::from_rotation_z(position)).normalize();
            }
            JointKind::Prismatic => {
                t.translation = self.rest.translation + self.rest.rotation * (DVec3::Z * position);
            }
            _ => {}
        }
        t
    }

    /// The position a transform puts the joint at: the turn about the rest's
    /// Z (or the slide along it) that brings the rest there.
    pub fn position_of(&self, t: &DTransform) -> f64 {
        match self.kind {
            JointKind::Revolute | JointKind::Continuous => {
                let turn = self.rest.rotation.inverse() * t.rotation;
                // The twist about Z: the turn's Z part, as an angle.
                let (z, w) = (turn.z, turn.w);
                let angle = 2.0 * z.atan2(w);
                // Into (-π, π]: a turn and its negative quaternion agree.
                let pi = std::f64::consts::PI;
                (angle + pi).rem_euclid(2.0 * pi) - pi
            }
            JointKind::Prismatic => {
                (self.rest.rotation.inverse() * (t.translation - self.rest.translation)).z
            }
            _ => 0.0,
        }
    }
}

/// A link's mass, where its centre is and how it resists turning — kept to
/// be written back.
#[derive(Clone, Debug, PartialEq)]
pub struct Inertial {
    /// The centre of mass and the inertia's axes, in the link's space.
    pub origin: DTransform,
    /// Kilograms.
    pub mass: f64,
    /// ixx, ixy, ixz, iyy, iyz, izz.
    pub inertia: [f64; 6],
}

/// What an object is to a robot.
#[derive(Clone, Debug, PartialEq)]
pub enum RobotPart {
    /// A link: the frame its visuals and its child joints hang from.
    Link {
        inertial: Option<Inertial>,
    },
    Joint(Joint),
    /// A link's collision shape: a mesh, hidden, kept to be written back.
    Collision,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm() -> Joint {
        Joint {
            kind: JointKind::Revolute,
            rest: DTransform {
                translation: DVec3::new(0.0, 0.0, 1.0),
                rotation: DQuat::from_rotation_x(0.7),
                scale: DVec3::ONE,
            },
            lower: -1.0,
            upper: 2.0,
            ..Joint::default()
        }
    }

    #[test]
    fn a_turned_joint_reads_back_the_turn_it_was_given() {
        let j = arm();
        for a in [-1.0, 0.0, 0.5, 2.0] {
            let t = j.place(a);
            assert!((j.position_of(&t) - a).abs() < 1e-9, "{a}");
            assert_eq!(t.translation, j.rest.translation);
        }
        assert_eq!(j.clamp(3.0), 2.0);
        assert_eq!(j.clamp(-3.0), -1.0);
    }

    #[test]
    fn a_sliding_joint_slides_along_its_own_z() {
        let j = Joint {
            kind: JointKind::Prismatic,
            ..arm()
        };
        let t = j.place(0.25);
        let along = j.rest.rotation * DVec3::Z;
        assert!((t.translation - (j.rest.translation + along * 0.25)).length() < 1e-12);
        assert!((j.position_of(&t) - 0.25).abs() < 1e-12);
        assert_eq!(JointKind::from_tag("prismatic"), Some(JointKind::Prismatic));
    }

    #[test]
    fn a_continuous_joint_is_not_held() {
        let j = Joint {
            kind: JointKind::Continuous,
            ..arm()
        };
        assert_eq!(j.clamp(10.0), 10.0);
    }
}
