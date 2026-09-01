//! Primitive mesh generators.
//!
//! These build **quad-dominant** topology wherever the shape allows it. A cube is
//! six quads, not twelve triangles, and a UV sphere is quads everywhere except
//! the pole rings. That is what a modeller expects to subdivide, loop-cut, and
//! bevel, and it is also the first real exercise of n-gon storage: if a primitive
//! silently triangulates, so will everything downstream.
//!
//! All primitives are built in Kreate space (Z-up) and centred on the origin
//! unless noted.

use std::f32::consts::{PI, TAU};

use glam::Vec3;

use crate::MeshData;

/// A `size` x `size` quad on the XY plane, centred on the origin, facing +Z.
pub fn plane(size: f32) -> MeshData {
    let h = size * 0.5;
    MeshData::from_faces(
        vec![
            Vec3::new(-h, -h, 0.0),
            Vec3::new(h, -h, 0.0),
            Vec3::new(h, h, 0.0),
            Vec3::new(-h, h, 0.0),
        ],
        &[vec![0, 1, 2, 3]],
    )
}

/// An `x` by `y` grid of quads on the XY plane, centred on the origin.
///
/// `x_divs` and `y_divs` are quad counts, clamped to at least 1.
pub fn grid(size: f32, x_divs: u32, y_divs: u32) -> MeshData {
    let (nx, ny) = (x_divs.max(1), y_divs.max(1));
    let h = size * 0.5;
    let mut positions = Vec::with_capacity(((nx + 1) * (ny + 1)) as usize);
    for j in 0..=ny {
        for i in 0..=nx {
            positions.push(Vec3::new(
                -h + size * i as f32 / nx as f32,
                -h + size * j as f32 / ny as f32,
                0.0,
            ));
        }
    }
    let vid = |i: u32, j: u32| j * (nx + 1) + i;
    let mut faces = Vec::with_capacity((nx * ny) as usize);
    for j in 0..ny {
        for i in 0..nx {
            faces.push(vec![
                vid(i, j),
                vid(i + 1, j),
                vid(i + 1, j + 1),
                vid(i, j + 1),
            ]);
        }
    }
    MeshData::from_faces(positions, &faces)
}

/// An axis-aligned cube of edge length `size`, as six quads, wound outward.
pub fn cube(size: f32) -> MeshData {
    let h = size * 0.5;
    let positions = vec![
        Vec3::new(-h, -h, -h), // 0
        Vec3::new(h, -h, -h),  // 1
        Vec3::new(h, h, -h),   // 2
        Vec3::new(-h, h, -h),  // 3
        Vec3::new(-h, -h, h),  // 4
        Vec3::new(h, -h, h),   // 5
        Vec3::new(h, h, h),    // 6
        Vec3::new(-h, h, h),   // 7
    ];
    let faces = vec![
        vec![0, 3, 2, 1], // -Z
        vec![4, 5, 6, 7], // +Z
        vec![0, 1, 5, 4], // -Y
        vec![2, 3, 7, 6], // +Y
        vec![1, 2, 6, 5], // +X
        vec![0, 4, 7, 3], // -X
    ];
    MeshData::from_faces(positions, &faces)
}

/// A UV sphere with `segments` around the Z axis and `rings` from pole to pole.
///
/// Quads everywhere except the two pole rings, which are necessarily triangles.
/// `segments` is clamped to >= 3 and `rings` to >= 2.
pub fn uv_sphere(radius: f32, segments: u32, rings: u32) -> MeshData {
    let seg = segments.max(3);
    let ring = rings.max(2);

    let mut positions = vec![Vec3::new(0.0, 0.0, radius)]; // north pole = 0
    for r in 1..ring {
        let phi = PI * r as f32 / ring as f32;
        let (sp, cp) = phi.sin_cos();
        for s in 0..seg {
            let theta = TAU * s as f32 / seg as f32;
            let (st, ct) = theta.sin_cos();
            positions.push(Vec3::new(radius * sp * ct, radius * sp * st, radius * cp));
        }
    }
    positions.push(Vec3::new(0.0, 0.0, -radius));
    let south = (positions.len() - 1) as u32;

    // Ring `r` (1-based) starts at this vertex index.
    let ring_start = |r: u32| 1 + (r - 1) * seg;
    let mut faces = Vec::new();

    // Winding note: rings run from the north pole downward, so increasing theta
    // is counter-clockwise seen from +Z but clockwise seen from -Z. The two pole
    // caps therefore wind in opposite directions.
    for s in 0..seg {
        let a = ring_start(1) + s;
        let b = ring_start(1) + (s + 1) % seg;
        faces.push(vec![0, a, b]);
    }
    for r in 1..ring - 1 {
        for s in 0..seg {
            let s1 = (s + 1) % seg;
            faces.push(vec![
                ring_start(r) + s,
                ring_start(r + 1) + s,
                ring_start(r + 1) + s1,
                ring_start(r) + s1,
            ]);
        }
    }
    for s in 0..seg {
        let a = ring_start(ring - 1) + s;
        let b = ring_start(ring - 1) + (s + 1) % seg;
        faces.push(vec![south, b, a]);
    }

    MeshData::from_faces(positions, &faces)
}

/// A cylinder along Z, with n-gon caps.
///
/// The caps are single n-gons rather than fans. That is the whole point of
/// supporting n-gons: a 32-sided cylinder cap should be one face the user can
/// select and inset, not 30 triangles.
pub fn cylinder(radius: f32, height: f32, segments: u32) -> MeshData {
    let seg = segments.max(3);
    let h = height * 0.5;
    let mut positions = Vec::with_capacity((seg * 2) as usize);
    for s in 0..seg {
        let a = TAU * s as f32 / seg as f32;
        let (sa, ca) = a.sin_cos();
        positions.push(Vec3::new(radius * ca, radius * sa, -h));
    }
    for s in 0..seg {
        let a = TAU * s as f32 / seg as f32;
        let (sa, ca) = a.sin_cos();
        positions.push(Vec3::new(radius * ca, radius * sa, h));
    }

    let mut faces = Vec::with_capacity((seg + 2) as usize);
    for s in 0..seg {
        let s1 = (s + 1) % seg;
        faces.push(vec![s, s1, seg + s1, seg + s]);
    }
    faces.push((0..seg).rev().collect()); // bottom cap, wound outward (-Z)
    faces.push((seg..seg * 2).collect()); // top cap (+Z)
    MeshData::from_faces(positions, &faces)
}

/// A cone along Z with an n-gon base, apex at `+height/2`.
pub fn cone(radius: f32, height: f32, segments: u32) -> MeshData {
    let seg = segments.max(3);
    let h = height * 0.5;
    let mut positions = Vec::with_capacity((seg + 1) as usize);
    for s in 0..seg {
        let a = TAU * s as f32 / seg as f32;
        let (sa, ca) = a.sin_cos();
        positions.push(Vec3::new(radius * ca, radius * sa, -h));
    }
    positions.push(Vec3::new(0.0, 0.0, h));
    let apex = seg;

    let mut faces: Vec<Vec<u32>> = (0..seg).map(|s| vec![s, (s + 1) % seg, apex]).collect();
    faces.push((0..seg).rev().collect()); // base
    MeshData::from_faces(positions, &faces)
}

/// A torus in the XY plane, all quads.
pub fn torus(major_radius: f32, minor_radius: f32, major_segs: u32, minor_segs: u32) -> MeshData {
    let (nm, nn) = (major_segs.max(3), minor_segs.max(3));
    let mut positions = Vec::with_capacity((nm * nn) as usize);
    for i in 0..nm {
        let u = TAU * i as f32 / nm as f32;
        let (su, cu) = u.sin_cos();
        for j in 0..nn {
            let v = TAU * j as f32 / nn as f32;
            let (sv, cv) = v.sin_cos();
            let r = major_radius + minor_radius * cv;
            positions.push(Vec3::new(r * cu, r * su, minor_radius * sv));
        }
    }
    let vid = |i: u32, j: u32| (i % nm) * nn + (j % nn);
    let mut faces = Vec::with_capacity((nm * nn) as usize);
    for i in 0..nm {
        for j in 0..nn {
            faces.push(vec![
                vid(i, j),
                vid(i + 1, j),
                vid(i + 1, j + 1),
                vid(i, j + 1),
            ]);
        }
    }
    MeshData::from_faces(positions, &faces)
}

/// An icosphere: an icosahedron subdivided `subdivisions` times and projected
/// onto the sphere. All triangles, but with far more uniform vertex density than
/// a UV sphere and no pole singularity.
pub fn ico_sphere(radius: f32, subdivisions: u32) -> MeshData {
    // Golden-ratio icosahedron.
    let t = (1.0 + 5.0f32.sqrt()) * 0.5;
    let mut positions: Vec<Vec3> = vec![
        Vec3::new(-1.0, t, 0.0),
        Vec3::new(1.0, t, 0.0),
        Vec3::new(-1.0, -t, 0.0),
        Vec3::new(1.0, -t, 0.0),
        Vec3::new(0.0, -1.0, t),
        Vec3::new(0.0, 1.0, t),
        Vec3::new(0.0, -1.0, -t),
        Vec3::new(0.0, 1.0, -t),
        Vec3::new(t, 0.0, -1.0),
        Vec3::new(t, 0.0, 1.0),
        Vec3::new(-t, 0.0, -1.0),
        Vec3::new(-t, 0.0, 1.0),
    ];
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for _ in 0..subdivisions.min(7) {
        // Midpoints are cached by edge key so shared edges produce one vertex;
        // without this the mesh would be a triangle soup with cracks in it.
        let mut midpoints: ahash::AHashMap<(u32, u32), u32> = ahash::AHashMap::new();
        let mut next = Vec::with_capacity(tris.len() * 4);
        for t in &tris {
            let mut mid = |a: u32, b: u32, positions: &mut Vec<Vec3>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                *midpoints.entry(key).or_insert_with(|| {
                    let p = (positions[a as usize] + positions[b as usize]) * 0.5;
                    positions.push(p);
                    (positions.len() - 1) as u32
                })
            };
            let a = mid(t[0], t[1], &mut positions);
            let b = mid(t[1], t[2], &mut positions);
            let c = mid(t[2], t[0], &mut positions);
            next.extend_from_slice(&[[t[0], a, c], [t[1], b, a], [t[2], c, b], [a, b, c]]);
        }
        tris = next;
    }

    for p in &mut positions {
        *p = p.normalize() * radius;
    }
    let faces: Vec<Vec<u32>> = tris.iter().map(|t| t.to_vec()).collect();
    MeshData::from_faces(positions, &faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tessellate::newell_normal;

    /// Every face must be wound so its normal points away from the solid's
    /// interior. A single inverted face is invisible in flat shading but shows
    /// up immediately once backface culling is on, so it is worth asserting.
    ///
    /// `interior` maps a face centroid to the nearest point *inside* the solid.
    /// For a star-shaped solid that is just the origin; a torus needs the centre
    /// of its tube, because the inner equator's outward normal points *toward*
    /// the Z axis and an origin-based test would call it inverted.
    fn assert_outward(m: &MeshData, name: &str, interior: impl Fn(Vec3) -> Vec3) {
        let pos = m.positions();
        for f in 0..m.num_faces() {
            let verts = m.face_verts(f);
            let pts: Vec<Vec3> = verts.iter().map(|&v| pos[v as usize]).collect();
            let n = newell_normal(&pts);
            let centroid = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
            let outward = centroid - interior(centroid);
            assert!(
                n.dot(outward) > 0.0,
                "{name}: face {f} is wound inward (n = {n}, centroid = {centroid})"
            );
        }
    }

    #[test]
    fn all_primitives_are_valid() {
        let all: Vec<(&str, MeshData)> = vec![
            ("plane", plane(2.0)),
            ("grid", grid(2.0, 4, 3)),
            ("cube", cube(2.0)),
            ("uv_sphere", uv_sphere(1.0, 12, 6)),
            ("cylinder", cylinder(1.0, 2.0, 16)),
            ("cone", cone(1.0, 2.0, 12)),
            ("torus", torus(1.0, 0.3, 12, 8)),
            ("ico_sphere", ico_sphere(1.0, 2)),
        ];
        for (name, m) in &all {
            let v = m.validate();
            assert!(v.is_empty(), "{name} failed validation: {v:#?}");
            assert!(m.num_faces() > 0, "{name} has no faces");
        }
    }

    #[test]
    fn cube_is_six_quads_not_twelve_triangles() {
        let c = cube(2.0);
        assert_eq!(c.num_verts(), 8);
        assert_eq!(c.num_faces(), 6);
        assert_eq!(c.num_edges(), 12, "Euler: 8 - 12 + 6 = 2");
        assert!((0..6).all(|f| c.face_len(f) == 4));
    }

    #[test]
    fn closed_primitives_satisfy_euler_characteristic() {
        // V - E + F = 2 for any closed surface of genus 0.
        for (name, m) in [
            ("cube", cube(1.0)),
            ("uv_sphere", uv_sphere(1.0, 10, 5)),
            ("cylinder", cylinder(1.0, 2.0, 12)),
            ("cone", cone(1.0, 2.0, 9)),
            ("ico_sphere", ico_sphere(1.0, 2)),
        ] {
            let chi = m.num_verts() as i64 - m.num_edges() as i64 + m.num_faces() as i64;
            assert_eq!(chi, 2, "{name}: V - E + F = {chi}, expected 2");
        }
    }

    #[test]
    fn torus_has_genus_one() {
        // V - E + F = 0 for a torus. This catches a seam that failed to close.
        let m = torus(1.0, 0.3, 16, 8);
        let chi = m.num_verts() as i64 - m.num_edges() as i64 + m.num_faces() as i64;
        assert_eq!(
            chi, 0,
            "torus should have Euler characteristic 0, got {chi}"
        );
    }

    #[test]
    fn closed_primitives_are_wound_outward() {
        let origin = |_: Vec3| Vec3::ZERO;
        assert_outward(&cube(2.0), "cube", origin);
        assert_outward(&uv_sphere(1.0, 12, 6), "uv_sphere", origin);
        assert_outward(&cylinder(1.0, 2.0, 16), "cylinder", origin);
        assert_outward(&cone(1.0, 2.0, 12), "cone", origin);
        assert_outward(&ico_sphere(1.0, 1), "ico_sphere", origin);

        // For the torus the interior reference is the centre of the tube: the
        // point on the major circle nearest this face.
        let major = 1.0f32;
        assert_outward(&torus(major, 0.3, 12, 8), "torus", move |c| {
            let radial = Vec3::new(c.x, c.y, 0.0);
            radial.normalize_or_zero() * major
        });
    }

    #[test]
    fn cylinder_caps_are_ngons() {
        let c = cylinder(1.0, 2.0, 16);
        // 16 side quads + 2 caps
        assert_eq!(c.num_faces(), 18);
        assert_eq!(c.face_len(16), 16, "bottom cap should be a single n-gon");
        assert_eq!(c.face_len(17), 16, "top cap should be a single n-gon");
    }

    #[test]
    fn sphere_points_lie_on_the_sphere() {
        let r = 2.5;
        for m in [uv_sphere(r, 16, 8), ico_sphere(r, 2)] {
            for p in m.positions() {
                assert!((p.length() - r).abs() < 1e-4, "{p} is not on radius {r}");
            }
        }
    }

    #[test]
    fn ico_sphere_subdivision_quadruples_faces() {
        assert_eq!(ico_sphere(1.0, 0).num_faces(), 20);
        assert_eq!(ico_sphere(1.0, 1).num_faces(), 80);
        assert_eq!(ico_sphere(1.0, 2).num_faces(), 320);
    }

    #[test]
    fn grid_dimensions_are_respected() {
        let g = grid(4.0, 4, 3);
        assert_eq!(g.num_faces(), 12);
        assert_eq!(g.num_verts(), 5 * 4);
        let b = g.bounds();
        assert!((b.size() - Vec3::new(4.0, 4.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn degenerate_parameters_are_clamped_not_invalid() {
        // A UI slider can reach 0; producing an invalid mesh instead of a
        // minimal one would be a crash waiting to happen.
        for m in [
            grid(1.0, 0, 0),
            uv_sphere(1.0, 0, 0),
            cylinder(1.0, 1.0, 0),
            cone(1.0, 1.0, 1),
            torus(1.0, 0.3, 0, 0),
        ] {
            m.assert_valid();
            assert!(m.num_faces() > 0);
        }
    }
}
