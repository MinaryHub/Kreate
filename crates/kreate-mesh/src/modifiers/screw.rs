//! The screw modifier: the mesh taken as a profile — its edges — and swept
//! round an axis, rising by a pitch each turn. A profile line makes a
//! surface of revolution; with a pitch, a thread, a spring, a spiral stair.
//! Blender's Screw.

use kreate_math::{Quat, Vec3};

use crate::MeshData;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrewParams {
    /// The axis swept round, through the mesh's origin: 0 X, 1 Y, 2 Z.
    pub axis: u32,
    /// How far round, in radians; a whole turn closes a ring.
    pub angle: f32,
    /// How far along the axis one whole turn rises.
    pub pitch: f32,
    /// Steps round one whole turn.
    pub steps: u32,
    /// Whole turns on top of the angle, for a spring.
    pub turns: u32,
}

impl Default for ScrewParams {
    fn default() -> Self {
        Self {
            axis: 2,
            angle: std::f32::consts::TAU,
            pitch: 0.0,
            steps: 32,
            turns: 0,
        }
    }
}

/// `input`'s edges swept round the axis.
///
/// Each step is a copy of the profile turned and raised; a quad joins each
/// edge to its copy at the next step. A whole turn with no pitch joins the
/// last step back to the first, so a circle's profile makes a closed ring.
pub fn screw(input: &MeshData, params: &ScrewParams) -> MeshData {
    // The edges the way their faces walk them, each once: the quads swept
    // from them then all face the same way, so a closed profile makes a
    // solid rather than a wall turned inside out here and there.
    let mut edges: Vec<[u32; 2]> = Vec::new();
    let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for f in 0..input.num_faces() {
        let ring = input.face_verts(f);
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            if a != b && seen.insert((a.min(b), a.max(b))) {
                edges.push([a, b]);
            }
        }
    }
    let total = params.angle + std::f32::consts::TAU * params.turns as f32;
    if input.is_empty() || edges.is_empty() || !total.is_finite() || total.abs() < 1e-6 {
        return input.clone();
    }
    let axis = match params.axis.min(2) {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    };
    let per_turn = params.steps.clamp(3, 512) as f32;
    let steps = ((total.abs() / std::f32::consts::TAU) * per_turn)
        .ceil()
        .max(1.0) as usize;
    let closes = params.pitch == 0.0 && (total.abs() - std::f32::consts::TAU).abs() < 1e-5;
    let rings = if closes { steps } else { steps + 1 };
    let n = input.num_verts();
    let positions = input.positions();
    let mut out_positions: Vec<Vec3> = Vec::with_capacity(rings * n as usize);
    for r in 0..rings {
        let t = r as f32 / steps as f32;
        let turn = Quat::from_axis_angle(axis, total * t);
        let rise = axis * (params.pitch * total * t / std::f32::consts::TAU);
        for p in positions {
            out_positions.push(turn * *p + rise);
        }
    }
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(steps * edges.len());
    for r in 0..steps {
        let next = (r + 1) % rings;
        let (base, base_next) = (r as u32 * n, next as u32 * n);
        for [a, b] in &edges {
            faces.push(vec![base + a, base + b, base_next + b, base_next + a]);
        }
    }
    let out = MeshData::from_faces(out_positions, &faces);
    // Facing out: the sweep's winding depends on which way the profile
    // runs and which way it turns, so it is settled by the volume's sign.
    let p = out.positions();
    let mut volume = 0.0f64;
    for f in 0..out.num_faces() {
        let ring = out.face_verts(f);
        for i in 1..ring.len().saturating_sub(1) {
            let (a, b, c) = (
                p[ring[0] as usize].as_dvec3(),
                p[ring[i] as usize].as_dvec3(),
                p[ring[i + 1] as usize].as_dvec3(),
            );
            volume += a.dot(b.cross(c));
        }
    }
    if volume >= 0.0 || !closes {
        return out;
    }
    let flipped: Vec<Vec<u32>> = (0..out.num_faces())
        .map(|f| out.face_verts(f).iter().rev().copied().collect())
        .collect();
    MeshData::from_faces(p.to_vec(), &flipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A profile: a square of edges standing off the axis, as a mesh of
    /// one face whose edges are the profile.
    fn square_profile() -> MeshData {
        MeshData::from_faces(
            vec![
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 1.0),
                Vec3::new(2.0, 0.0, 1.0),
            ],
            &[vec![0, 1, 2, 3]],
        )
    }

    #[test]
    fn a_whole_turn_of_a_square_is_a_closed_ring_of_the_right_volume() {
        let out = screw(
            &square_profile(),
            &ScrewParams {
                axis: 2,
                angle: std::f32::consts::TAU,
                pitch: 0.0,
                steps: 96,
                turns: 0,
            },
        );
        assert!(crate::measure::is_closed(&out));
        let v = crate::measure::mass_properties(&out, |p| p).volume;
        // A square 1 x 1 swept round at mean radius 2.5: 2π · 2.5 · 1.
        let want = std::f64::consts::TAU * 2.5;
        assert!((v - want).abs() < want * 0.01, "{v} vs {want}");
    }

    #[test]
    fn a_pitch_makes_a_spring_that_rises_by_it_each_turn() {
        let out = screw(
            &square_profile(),
            &ScrewParams {
                axis: 2,
                angle: 0.0,
                pitch: 2.0,
                steps: 16,
                turns: 3,
            },
        );
        assert!(!crate::measure::is_closed(&out), "open at both ends");
        let top = out.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        assert!(
            (top - 7.0).abs() < 1e-4,
            "three turns of two, on a profile one tall: {top}"
        );
    }
}
