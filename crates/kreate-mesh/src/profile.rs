//! Solids from a sketch: a closed outline drawn in a plane, pushed out
//! straight or turned about an axis.
//!
//! A sketch is a mesh of edges with no faces, lying in its own XY plane —
//! the plane of the object that holds it, which is where the work plane put
//! it. Its closed loops are the profile: loops that run anticlockwise (seen
//! from +Z) are outlines, the others holes, so a ring drawn as two circles is
//! a tube when it is extruded.

use std::collections::HashMap;

use kreate_math::{Vec2, Vec3};

use crate::MeshData;

/// The closed loops of a sketch: the boundary of its faces, as vertex
/// indices in order. A sketch is flat faces in its XY plane — a filled
/// outline, as F or a plane makes one — and its edges with one face are the
/// profile. A face drawn inside another is a hole (see `extrude_profile`).
pub fn profile_loops(mesh: &MeshData) -> Vec<Vec<u32>> {
    let edges = mesh.corner_edges();
    let mut uses = vec![0u32; mesh.num_edges() as usize];
    for e in edges.iter() {
        uses[*e as usize] += 1;
    }
    // Each boundary edge, directed as its one face runs along it.
    let offsets = mesh.face_offsets();
    let corners = mesh.corner_verts();
    let mut next: HashMap<u32, u32> = HashMap::new();
    for f in 0..mesh.num_faces() as usize {
        let (a, b) = (offsets[f] as usize, offsets[f + 1] as usize);
        for c in a..b {
            if uses[edges[c] as usize] == 1 {
                let to = if c + 1 < b {
                    corners[c + 1]
                } else {
                    corners[a]
                };
                next.insert(corners[c], to);
            }
        }
    }
    let mut seen: HashMap<u32, bool> = HashMap::new();
    let mut loops = Vec::new();
    let mut starts: Vec<u32> = next.keys().copied().collect();
    starts.sort_unstable();
    for start in starts {
        if seen.get(&start).copied().unwrap_or(false) {
            continue;
        }
        let mut chain = vec![start];
        seen.insert(start, true);
        let mut at = next[&start];
        let closed = loop {
            if at == start {
                break true;
            }
            if seen.get(&at).copied().unwrap_or(false) {
                break false;
            }
            let Some(&to) = next.get(&at) else {
                break false;
            };
            seen.insert(at, true);
            chain.push(at);
            at = to;
        };
        if closed && chain.len() >= 3 {
            loops.push(chain);
        }
    }
    loops
}

fn flat(p: Vec3) -> Vec2 {
    Vec2::new(p.x, p.y)
}

fn signed_area(points: &[Vec3], l: &[u32]) -> f32 {
    (0..l.len())
        .map(|i| {
            let (a, b) = (
                flat(points[l[i] as usize]),
                flat(points[l[(i + 1) % l.len()] as usize]),
            );
            a.x * b.y - b.x * a.y
        })
        .sum::<f32>()
        * 0.5
}

/// The sketch's profile pushed `height` along +Z (or -Z, for a negative
/// height): a closed solid, holes and all. Empty when the sketch has no
/// closed loop.
pub fn extrude_profile(mesh: &MeshData, height: f32) -> MeshData {
    let mut loops = profile_loops(mesh);
    if loops.is_empty() || height == 0.0 {
        return MeshData::empty();
    }
    let source = mesh.positions();
    // Which loops are holes: those inside an odd number of others. Each is
    // then turned to run the way its role wants, whichever way it was drawn.
    let flat_loop = |l: &[u32]| {
        l.iter()
            .map(|v| flat(source[*v as usize]))
            .collect::<Vec<Vec2>>()
    };
    let shapes: Vec<Vec<Vec2>> = loops.iter().map(|l| flat_loop(l)).collect();
    for (i, l) in loops.iter_mut().enumerate() {
        let probe = shapes[i][0];
        let depth = shapes
            .iter()
            .enumerate()
            .filter(|(j, s)| *j != i && inside(s, probe))
            .count();
        let outline = depth % 2 == 0;
        if (signed_area(source, l) > 0.0) != outline {
            l.reverse();
        }
    }

    // Bottom ring then top ring, per vertex of the sketch.
    let up = Vec3::new(0.0, 0.0, height);
    let n = source.len() as u32;
    let mut points: Vec<Vec3> = source.to_vec();
    points.extend(source.iter().map(|p| *p + up));
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let caps = crate::section::fill_loops(loops.clone(), &|i: u32| flat(source[i as usize]));
    // Wound for +Z height: the bottom faces down, the top up, the sides out.
    let flip = height < 0.0;
    for [a, b, c] in &caps {
        // The sketch's own cap faces away from the solid, the pushed one
        // away from it on the other side.
        if flip {
            faces.push(vec![*a, *b, *c]);
            faces.push(vec![*a + n, *c + n, *b + n]);
        } else {
            faces.push(vec![*a, *c, *b]);
            faces.push(vec![*a + n, *b + n, *c + n]);
        }
    }
    for l in &loops {
        for k in 0..l.len() {
            let (a, b) = (l[k], l[(k + 1) % l.len()]);
            faces.push(if flip {
                vec![a, a + n, b + n, b]
            } else {
                vec![a, b, b + n, a + n]
            });
        }
    }
    // Only the vertices in use.
    let mut renumber = vec![u32::MAX; points.len()];
    let mut kept = Vec::new();
    for f in &mut faces {
        for v in f.iter_mut() {
            if renumber[*v as usize] == u32::MAX {
                renumber[*v as usize] = kept.len() as u32;
                kept.push(points[*v as usize]);
            }
            *v = renumber[*v as usize];
        }
    }
    MeshData::from_faces(kept, &faces)
}

/// The sketch's profile turned a full turn about its Y axis, `segments`
/// steps round: a lathe. The profile's X is the radius (its distance from
/// the axis), so it should lie to one side of it.
pub fn revolve_profile(mesh: &MeshData, segments: u32) -> MeshData {
    let loops = profile_loops(mesh);
    let source = mesh.positions();
    let mut parts: Vec<MeshData> = Vec::new();
    for l in loops {
        // Closed: the lathe walks the profile end to end, so the first point
        // is given again at the end.
        let profile: Vec<(f32, f32)> = l
            .iter()
            .chain(l.first())
            .map(|v| {
                let p = source[*v as usize];
                (p.x.abs(), p.y)
            })
            .collect();
        parts.push(crate::primitives::lathe(&profile, segments.max(3)));
    }
    match parts.len() {
        0 => MeshData::empty(),
        1 => parts.pop().expect("one"),
        _ => {
            let pieces: Vec<crate::edit::Part<'_>> = parts
                .iter()
                .map(|m| crate::edit::Part {
                    mesh: m,
                    transform: kreate_math::DAffine3::IDENTITY,
                })
                .collect();
            crate::edit::join(&pieces).0
        }
    }
}

/// Even-odd containment.
fn inside(polygon: &[Vec2], p: Vec2) -> bool {
    let mut odd = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let (a, b) = (polygon[i], polygon[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            odd = !odd;
        }
        j = i;
    }
    odd
}

/// A sketch of closed outlines, for tests and for starting one: each loop
/// its points in order, filled.
pub fn sketch_of(loops: &[Vec<Vec2>]) -> MeshData {
    let mut points = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for l in loops {
        let start = points.len() as u32;
        points.extend(l.iter().map(|p| Vec3::new(p.x, p.y, 0.0)));
        faces.push((start..start + l.len() as u32).collect());
    }
    MeshData::from_faces(points, &faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::{is_closed, mass_properties};

    fn square(half: f32) -> Vec<Vec2> {
        vec![
            Vec2::new(-half, -half),
            Vec2::new(half, -half),
            Vec2::new(half, half),
            Vec2::new(-half, half),
        ]
    }

    #[test]
    fn a_square_pushed_up_is_a_box() {
        let sketch = sketch_of(&[square(1.0)]);
        assert_eq!(profile_loops(&sketch).len(), 1);
        let solid = extrude_profile(&sketch, 3.0);
        let p = mass_properties(&solid, |p| p);
        assert!((p.volume - 12.0).abs() < 1e-4, "{}", p.volume);
        assert!(is_closed(&solid));
        assert_eq!(solid.validate(), Vec::new());
    }

    #[test]
    fn a_square_drawn_the_other_way_or_pushed_down_still_faces_out() {
        let mut backwards = square(1.0);
        backwards.reverse();
        let sketch = sketch_of(&[backwards]);
        for h in [2.0, -2.0] {
            let v = mass_properties(&extrude_profile(&sketch, h), |p| p).volume;
            assert!((v - 8.0).abs() < 1e-4, "height {h}: {v}");
        }
    }

    #[test]
    fn a_square_inside_a_square_is_a_hollow_section() {
        let sketch = sketch_of(&[square(2.0), square(1.0)]);
        let solid = extrude_profile(&sketch, 1.0);
        let v = mass_properties(&solid, |p| p).volume;
        assert!((v - 12.0).abs() < 1e-4, "16 - 4: {v}");
        assert!(is_closed(&solid));
    }

    #[test]
    fn a_profile_turned_about_y_is_a_ring() {
        // A unit square from x 2 to 3: a ring of radius 2 to 3, 1 tall.
        let sketch = sketch_of(&[vec![
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(3.0, 1.0),
            Vec2::new(2.0, 1.0),
        ]]);
        let solid = revolve_profile(&sketch, 128);
        let v = mass_properties(&solid, |p| p).volume.abs();
        let exact = std::f64::consts::PI * (9.0 - 4.0);
        assert!((v - exact).abs() / exact < 0.01, "{v} against {exact}");
    }
}
