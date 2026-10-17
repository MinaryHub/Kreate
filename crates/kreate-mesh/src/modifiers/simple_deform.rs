//! The simple deform modifier: the mesh twisted, bent, tapered or stretched
//! along one of its axes by an amount — Blender's Simple Deform.
//!
//! Each works on where a vertex is along the axis, from the mesh's bounds:
//! at the bottom nothing happens, at the top the whole amount does, and
//! between them it is spread out.

use kreate_math::Vec3;

use crate::MeshData;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeformMode {
    /// Turned about the axis, more the further along it.
    #[default]
    Twist,
    /// Curled round, as a sheet bends.
    Bend,
    /// Narrowed towards one end.
    Taper,
    /// Pulled longer and narrowed to keep its bulk, or squashed.
    Stretch,
}

impl DeformMode {
    pub const ALL: [DeformMode; 4] = [
        DeformMode::Twist,
        DeformMode::Bend,
        DeformMode::Taper,
        DeformMode::Stretch,
    ];

    pub fn tag(self) -> &'static str {
        match self {
            DeformMode::Twist => "twist",
            DeformMode::Bend => "bend",
            DeformMode::Taper => "taper",
            DeformMode::Stretch => "stretch",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.tag() == tag)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimpleDeformParams {
    pub mode: DeformMode,
    /// The axis it works along: 0 X, 1 Y, 2 Z.
    pub axis: u32,
    /// Radians for a twist or a bend; a factor for a taper (1 is a point)
    /// or a stretch (1 is twice as long).
    pub amount: f32,
}

impl Default for SimpleDeformParams {
    fn default() -> Self {
        Self {
            mode: DeformMode::Twist,
            axis: 2,
            amount: std::f32::consts::FRAC_PI_4,
        }
    }
}

/// `input` deformed along its axis.
pub fn simple_deform(input: &MeshData, params: &SimpleDeformParams) -> MeshData {
    if input.is_empty() || params.amount == 0.0 || !params.amount.is_finite() {
        return input.clone();
    }
    let axis = (params.axis as usize).min(2);
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let positions = input.positions();
    let (lo, hi) = positions.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
        (lo.min(p[axis]), hi.max(p[axis]))
    });
    let span = (hi - lo).max(1e-9);
    let mut out = input.clone();
    for p in out.positions_mut() {
        let t = (p[axis] - lo) / span;
        let mut q = *p;
        match params.mode {
            DeformMode::Twist => {
                let (s, c) = (params.amount * t).sin_cos();
                let (a, b) = (p[u], p[v]);
                q[u] = a * c - b * s;
                q[v] = a * s + b * c;
            }
            DeformMode::Bend => {
                // The axis curled into an arc of the amount in the plane of
                // the axis and `u`: a flat sheet becomes a tube's wall.
                let angle = params.amount;
                let radius = span / angle;
                let along = p[axis] - lo;
                let theta = along / radius;
                let r = radius - p[u];
                q[axis] = lo + r * theta.sin();
                q[u] = radius - r * theta.cos();
            }
            DeformMode::Taper => {
                let k = 1.0 - params.amount.min(1.0) * t;
                q[u] = p[u] * k;
                q[v] = p[v] * k;
            }
            DeformMode::Stretch => {
                let k = 1.0 + params.amount;
                q[axis] = lo + (p[axis] - lo) * k;
                // Thinner in the middle as it is pulled, the bulk kept.
                let pinch = 1.0 / k.sqrt().max(1e-6);
                let bulge = 1.0 + (1.0 - pinch) * (1.0 - (2.0 * t - 1.0).abs());
                q[u] = p[u] * pinch * bulge;
                q[v] = p[v] * pinch * bulge;
            }
        }
        *p = q;
    }
    let _ = Vec3::ZERO;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn tall() -> MeshData {
        let mut m = primitives::cube(1.0);
        for p in m.positions_mut() {
            p.z *= 4.0;
        }
        m
    }

    #[test]
    fn a_twist_turns_the_top_and_leaves_the_bottom() {
        let m = tall();
        let out = simple_deform(
            &m,
            &SimpleDeformParams {
                mode: DeformMode::Twist,
                axis: 2,
                amount: std::f32::consts::FRAC_PI_2,
            },
        );
        for (a, b) in m.positions().iter().zip(out.positions()) {
            if a.z < -1.9 {
                assert!(
                    (a.truncate() - b.truncate()).length() < 1e-6,
                    "the bottom stays"
                );
            } else {
                // A quarter turn: (x, y) -> (-y, x).
                assert!(
                    (b.x + a.y).abs() < 1e-5 && (b.y - a.x).abs() < 1e-5,
                    "{a:?} {b:?}"
                );
            }
            assert!((a.z - b.z).abs() < 1e-6);
        }
    }

    #[test]
    fn a_taper_to_a_point_and_a_stretch_by_a_half() {
        let m = tall();
        let taper = simple_deform(
            &m,
            &SimpleDeformParams {
                mode: DeformMode::Taper,
                axis: 2,
                amount: 1.0,
            },
        );
        let top: Vec<_> = taper.positions().iter().filter(|p| p.z > 1.9).collect();
        assert!(top.iter().all(|p| p.x.abs() < 1e-6 && p.y.abs() < 1e-6));
        let stretch = simple_deform(
            &m,
            &SimpleDeformParams {
                mode: DeformMode::Stretch,
                axis: 2,
                amount: 0.5,
            },
        );
        let hi = stretch
            .positions()
            .iter()
            .map(|p| p.z)
            .fold(f32::MIN, f32::max);
        assert!((hi - 4.0).abs() < 1e-5, "{hi}");
    }

    #[test]
    fn a_bend_by_a_half_turn_brings_the_ends_level() {
        let m = tall();
        let out = simple_deform(
            &m,
            &SimpleDeformParams {
                mode: DeformMode::Bend,
                axis: 2,
                amount: std::f32::consts::PI,
            },
        );
        // Bent round half a circle, the top's centre line sits at the same
        // height as the bottom's, across from it.
        let bottom_z = out
            .positions()
            .iter()
            .zip(m.positions())
            .filter(|(_, a)| a.z < -1.9 && a.x.abs() < 0.6)
            .map(|(b, _)| b.z)
            .fold(0.0, f32::max);
        let top_z = out
            .positions()
            .iter()
            .zip(m.positions())
            .filter(|(_, a)| a.z > 1.9)
            .map(|(b, _)| b.z)
            .fold(0.0, f32::max);
        assert!((top_z - bottom_z).abs() < 0.7, "{top_z} vs {bottom_z}");
    }
}
