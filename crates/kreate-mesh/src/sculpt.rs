//! Sculpting: a brush stroke as a function of the mesh and where the brush
//! went.
//!
//! A stroke is a list of dabs — a centre and a radius each, in the mesh's own
//! space — and a brush. Applied in order to the stroke's starting mesh, the
//! same dabs make the same shape, which is what lets the stroke be one
//! operator: replayable, one step to undo, and re-run from the start while
//! the pointer is still moving.
//!
//! Falloff is Blender's Smooth curve (`3t² − 2t³`, full in the middle, nothing
//! at the rim), and every brush reads the normals and neighbours of the mesh
//! as the stroke found it, so a dab does not chase what the dab before it did.

use std::sync::Arc;

use kreate_math::Vec3;

use crate::MeshData;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Brush {
    /// Raises the surface along its normal under the brush; inverted, carves.
    #[default]
    Draw,
    /// Evens the surface out towards its neighbours.
    Smooth,
    /// Drags what is under the brush where the pointer goes.
    Grab,
    /// Swells each vertex along its own normal; inverted, shrinks.
    Inflate,
}

/// One touch of the brush: where, and how wide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dab {
    pub centre: Vec3,
    pub radius: f32,
}

/// What a stroke does besides its dabs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    pub brush: Brush,
    /// How hard, 0 to 1.
    pub strength: f32,
    /// The other way: carve instead of raise, shrink instead of swell.
    pub invert: bool,
    /// For a grab: how far the pointer has dragged, in the mesh's space.
    pub grab: Vec3,
    /// Mirrored across X as well, Blender's X symmetry.
    pub mirror_x: bool,
}

fn falloff(distance: f32, radius: f32) -> f32 {
    if distance >= radius || radius <= 0.0 {
        return 0.0;
    }
    let t = 1.0 - distance / radius;
    t * t * (3.0 - 2.0 * t)
}

/// Area-weighted vertex normals.
fn normals(mesh: &MeshData) -> Vec<Vec3> {
    let p = mesh.positions();
    let mut out = vec![Vec3::ZERO; p.len()];
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        let mut n = Vec3::ZERO;
        for i in 0..ring.len() {
            n += p[ring[i] as usize].cross(p[ring[(i + 1) % ring.len()] as usize]);
        }
        for v in ring {
            out[*v as usize] += n;
        }
    }
    out.into_iter().map(|n| n.normalize_or_zero()).collect()
}

/// Each vertex's neighbours along edges.
fn neighbours(mesh: &MeshData) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new(); mesh.num_verts() as usize];
    for [a, b] in mesh.edges() {
        out[*a as usize].push(*b);
        out[*b as usize].push(*a);
    }
    out
}

/// Applies a stroke to `mesh`. `false` when nothing was under any dab.
pub fn apply(mesh: &mut MeshData, dabs: &[Dab], stroke: &Stroke) -> bool {
    if dabs.is_empty() || mesh.num_verts() == 0 {
        return false;
    }
    let strength = stroke.strength.clamp(0.0, 1.0);
    let sign = if stroke.invert { -1.0 } else { 1.0 };
    let normal = normals(mesh);
    let original: Arc<Vec<Vec3>> = Arc::new(mesh.positions().to_vec());
    let near = neighbours(mesh);
    // The dabs, and their reflections when mirrored.
    let mut all: Vec<(Dab, bool)> = dabs.iter().map(|d| (*d, false)).collect();
    if stroke.mirror_x {
        all.extend(dabs.iter().map(|d| {
            (
                Dab {
                    centre: Vec3::new(-d.centre.x, d.centre.y, d.centre.z),
                    ..*d
                },
                true,
            )
        }));
    }
    let mut moved = false;
    match stroke.brush {
        Brush::Grab => {
            // One grab: what was under the first dab, carried by the drag.
            let first = dabs[0];
            let mut grabs = vec![(first, stroke.grab)];
            if stroke.mirror_x {
                let flip = |v: Vec3| Vec3::new(-v.x, v.y, v.z);
                grabs.push((
                    Dab {
                        centre: flip(first.centre),
                        ..first
                    },
                    flip(stroke.grab),
                ));
            }
            for (dab, delta) in &grabs {
                let (dab, delta) = (*dab, *delta);
                let p = mesh.positions_mut();
                for (i, o) in original.iter().enumerate() {
                    let w = falloff(o.distance(dab.centre), dab.radius);
                    if w > 0.0 {
                        p[i] += delta * w;
                        moved = true;
                    }
                }
            }
        }
        Brush::Draw => {
            for (dab, _) in &all {
                // The surface's normal under the brush, weighted as the brush is.
                let mut area = Vec3::ZERO;
                for (i, o) in original.iter().enumerate() {
                    area += normal[i] * falloff(o.distance(dab.centre), dab.radius);
                }
                let area = area.normalize_or_zero();
                let step = dab.radius * 0.15 * strength * sign;
                let p = mesh.positions_mut();
                for (i, o) in original.iter().enumerate() {
                    let w = falloff(o.distance(dab.centre), dab.radius);
                    if w > 0.0 {
                        p[i] += area * step * w;
                        moved = true;
                    }
                }
            }
        }
        Brush::Inflate => {
            for (dab, _) in &all {
                let step = dab.radius * 0.1 * strength * sign;
                let p = mesh.positions_mut();
                for (i, o) in original.iter().enumerate() {
                    let w = falloff(o.distance(dab.centre), dab.radius);
                    if w > 0.0 {
                        p[i] += normal[i] * step * w;
                        moved = true;
                    }
                }
            }
        }
        Brush::Smooth => {
            for (dab, _) in &all {
                let now = mesh.positions().to_vec();
                let p = mesh.positions_mut();
                for (i, o) in original.iter().enumerate() {
                    let w = falloff(o.distance(dab.centre), dab.radius) * strength;
                    if w <= 0.0 || near[i].is_empty() {
                        continue;
                    }
                    let average = near[i].iter().fold(Vec3::ZERO, |s, n| s + now[*n as usize])
                        / near[i].len() as f32;
                    p[i] = now[i].lerp(average, w);
                    moved = true;
                }
            }
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn grid() -> MeshData {
        primitives::grid(4.0, 16, 16)
    }

    fn stroke(brush: Brush) -> Stroke {
        Stroke {
            brush,
            strength: 1.0,
            invert: false,
            grab: Vec3::ZERO,
            mirror_x: false,
        }
    }

    fn top(m: &MeshData) -> f32 {
        m.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max)
    }

    #[test]
    fn draw_raises_under_the_brush_and_nowhere_else_and_inverted_carves() {
        let mut m = grid();
        let dab = Dab {
            centre: Vec3::ZERO,
            radius: 1.0,
        };
        assert!(apply(&mut m, &[dab], &stroke(Brush::Draw)));
        assert!(top(&m) > 0.1, "{}", top(&m));
        // Outside the radius, untouched.
        assert!(m
            .positions()
            .iter()
            .filter(|p| p.x.abs() > 1.1 || p.y.abs() > 1.1)
            .all(|p| p.z == 0.0));
        let mut m = grid();
        apply(
            &mut m,
            &[dab],
            &Stroke {
                invert: true,
                ..stroke(Brush::Draw)
            },
        );
        assert!(m.positions().iter().map(|p| p.z).fold(f32::MAX, f32::min) < -0.1);
    }

    #[test]
    fn smooth_takes_a_spike_down() {
        let mut m = grid();
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::ZERO,
                radius: 0.3,
            }],
            &stroke(Brush::Draw),
        );
        let spike = top(&m);
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::ZERO,
                radius: 1.0,
            }; 4],
            &stroke(Brush::Smooth),
        );
        assert!(top(&m) < spike * 0.7, "{} from {spike}", top(&m));
    }

    #[test]
    fn grab_carries_what_was_under_the_first_dab() {
        let mut m = grid();
        let s = Stroke {
            grab: Vec3::new(0.0, 0.0, 2.0),
            ..stroke(Brush::Grab)
        };
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::ZERO,
                radius: 1.0,
            }],
            &s,
        );
        assert!((top(&m) - 2.0).abs() < 0.05, "{}", top(&m));
    }

    #[test]
    fn x_symmetry_does_the_same_on_the_other_side() {
        let mut m = grid();
        let s = Stroke {
            mirror_x: true,
            ..stroke(Brush::Draw)
        };
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::new(1.0, 0.0, 0.0),
                radius: 0.5,
            }],
            &s,
        );
        let at = |x: f32| {
            m.positions()
                .iter()
                .find(|p| (p.x - x).abs() < 1e-4 && p.y.abs() < 1e-4)
                .unwrap()
                .z
        };
        assert!(at(1.0) > 0.0);
        assert!(
            (at(1.0) - at(-1.0)).abs() < 1e-5,
            "{} vs {}",
            at(1.0),
            at(-1.0)
        );
    }

    #[test]
    fn inflate_swells_a_sphere() {
        let mut m = primitives::uv_sphere(1.0, 16, 8);
        let r0 = m.positions().iter().map(|p| p.length()).fold(0.0, f32::max);
        apply(
            &mut m,
            &[Dab {
                centre: Vec3::Z,
                radius: 0.8,
            }],
            &stroke(Brush::Inflate),
        );
        let r1 = m.positions().iter().map(|p| p.length()).fold(0.0, f32::max);
        assert!(r1 > r0 + 0.01);
    }
}
