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

use glam::{Vec2, Vec3};

use crate::attribute::{names, AttrData, Domain};
use crate::MeshData;

// --- surface coordinates ---------------------------------------------------
//
// Every primitive with an obvious parameterisation carries one. Without it a
// cube added from the menu has no UV layer, and a texture assigned to it
// samples a single texel — which is the state this was in until the renderer
// could sample anything at all.
//
// The coordinates are per *corner*, not per vertex, and that is what makes a
// seam expressible: the column where a sphere wraps needs `u = 1` on one side
// of an edge and `u = 0` on the other, at the same position. A per-vertex layer
// cannot say that, and the usual workaround — duplicating the column of
// vertices — puts the seam in the topology the user edits.
//
// **Built from the indices that built the geometry**, not recovered from the
// positions. Recovering means `atan2`, and `atan2` cannot tell the wrap from a
// face that happens to span it — the first version of this guessed, by calling
// any face covering more than half the image a seam, and a cylinder's cap is a
// face that covers all of it on purpose. A ring index is exact and knows
// nothing about halves.

/// Writes a corner-domain UV layer.
///
/// `uvs` is in corner order: every face's corners in turn, in the order the
/// faces were given.
fn set_uvs(mesh: &mut MeshData, uvs: Vec<Vec2>) {
    debug_assert_eq!(uvs.len(), mesh.num_corners() as usize);
    mesh.attributes_mut()
        .insert(names::UV, Domain::Corner, AttrData::F32x2(uvs.into()));
}

/// A disc projection, for a cap.
///
/// Segment `s` of `n` around a circle, laid on the image as the circle it is.
/// A cap cannot use the angle-and-height mapping its sides use: it *is* the
/// whole turn, so that mapping would run the texture round the rim and leave
/// the middle undefined.
fn disc(s: u32, n: u32) -> Vec2 {
    let a = TAU * s as f32 / n as f32;
    let (sa, ca) = a.sin_cos();
    Vec2::new(0.5 + 0.5 * ca, 0.5 + 0.5 * sa)
}

/// A `size` x `size` quad on the XY plane, centred on the origin, facing +Z.
pub fn plane(size: f32) -> MeshData {
    let h = size * 0.5;
    let mut mesh = MeshData::from_faces(
        vec![
            Vec3::new(-h, -h, 0.0),
            Vec3::new(h, -h, 0.0),
            Vec3::new(h, h, 0.0),
            Vec3::new(-h, h, 0.0),
        ],
        &[vec![0, 1, 2, 3]],
    );
    set_uvs(
        &mut mesh,
        vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ],
    );
    mesh
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
    let mut uvs = Vec::with_capacity((nx * ny * 4) as usize);
    let (fx, fy) = (1.0 / nx as f32, 1.0 / ny as f32);
    for j in 0..ny {
        for i in 0..nx {
            faces.push(vec![
                vid(i, j),
                vid(i + 1, j),
                vid(i + 1, j + 1),
                vid(i, j + 1),
            ]);
            uvs.extend_from_slice(&[
                Vec2::new(i as f32 * fx, j as f32 * fy),
                Vec2::new((i + 1) as f32 * fx, j as f32 * fy),
                Vec2::new((i + 1) as f32 * fx, (j + 1) as f32 * fy),
                Vec2::new(i as f32 * fx, (j + 1) as f32 * fy),
            ]);
        }
    }
    let mut mesh = MeshData::from_faces(positions, &faces);
    set_uvs(&mut mesh, uvs);
    mesh
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
    let mut mesh = MeshData::from_faces(positions, &faces);
    // Every face takes the whole square, which is what a checker on a fresh
    // cube should look like. The alternative — six faces laid out as a cross in
    // one image — is an unwrap, and an unwrap is a choice the user makes rather
    // than one a primitive makes for them.
    let square = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];
    set_uvs(&mut mesh, faces.iter().flat_map(|_| square).collect());
    mesh
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
    // Longitude from the segment index and latitude from the ring index, so the
    // face that wraps gets `u = 1` on its far side rather than `u = 0`. The top
    // row of an image is the north pole, the same way up as an environment map.
    let mut uvs: Vec<Vec2> = Vec::new();
    let (fu, fv) = (1.0 / seg as f32, 1.0 / ring as f32);

    // Winding note: rings run from the north pole downward, so increasing theta
    // is counter-clockwise seen from +Z but clockwise seen from -Z. The two pole
    // caps therefore wind in opposite directions.
    for s in 0..seg {
        let a = ring_start(1) + s;
        let b = ring_start(1) + (s + 1) % seg;
        faces.push(vec![0, a, b]);
        // The pole sits between its two neighbours. Left at zero it would reach
        // back to the left edge of the image and the cap would be a star.
        uvs.extend_from_slice(&[
            Vec2::new((s as f32 + 0.5) * fu, 0.0),
            Vec2::new(s as f32 * fu, fv),
            Vec2::new((s + 1) as f32 * fu, fv),
        ]);
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
            uvs.extend_from_slice(&[
                Vec2::new(s as f32 * fu, r as f32 * fv),
                Vec2::new(s as f32 * fu, (r + 1) as f32 * fv),
                Vec2::new((s + 1) as f32 * fu, (r + 1) as f32 * fv),
                Vec2::new((s + 1) as f32 * fu, r as f32 * fv),
            ]);
        }
    }
    for s in 0..seg {
        let a = ring_start(ring - 1) + s;
        let b = ring_start(ring - 1) + (s + 1) % seg;
        faces.push(vec![south, b, a]);
        uvs.extend_from_slice(&[
            Vec2::new((s as f32 + 0.5) * fu, 1.0),
            Vec2::new((s + 1) as f32 * fu, 1.0 - fv),
            Vec2::new(s as f32 * fu, 1.0 - fv),
        ]);
    }

    let mut mesh = MeshData::from_faces(positions, &faces);
    set_uvs(&mut mesh, uvs);
    mesh
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
    let mut uvs: Vec<Vec2> = Vec::new();
    let fu = 1.0 / seg as f32;
    for s in 0..seg {
        let s1 = (s + 1) % seg;
        faces.push(vec![s, s1, seg + s1, seg + s]);
        uvs.extend_from_slice(&[
            Vec2::new(s as f32 * fu, 0.0),
            Vec2::new((s + 1) as f32 * fu, 0.0),
            Vec2::new((s + 1) as f32 * fu, 1.0),
            Vec2::new(s as f32 * fu, 1.0),
        ]);
    }
    // The caps are discs, not more of the side. A cap is the whole turn, so the
    // angle-and-height mapping would run the texture round its rim and leave
    // the middle with nowhere to come from.
    faces.push((0..seg).rev().collect()); // bottom cap, wound outward (-Z)
    uvs.extend((0..seg).rev().map(|s| disc(s, seg)));
    faces.push((seg..seg * 2).collect()); // top cap (+Z)
    uvs.extend((0..seg).map(|s| disc(s, seg)));
    let mut mesh = MeshData::from_faces(positions, &faces);
    set_uvs(&mut mesh, uvs);
    mesh
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
    let fu = 1.0 / seg as f32;
    let mut uvs: Vec<Vec2> = Vec::new();
    for s in 0..seg {
        // The apex is on the axis and shared by every side, so it has no
        // longitude of its own. Each triangle gives it the middle of its own
        // span, which is what stops the cap fanning across the whole image.
        uvs.extend_from_slice(&[
            Vec2::new(s as f32 * fu, 0.0),
            Vec2::new((s + 1) as f32 * fu, 0.0),
            Vec2::new((s as f32 + 0.5) * fu, 1.0),
        ]);
    }
    faces.push((0..seg).rev().collect()); // base
    uvs.extend((0..seg).rev().map(|s| disc(s, seg)));
    let mut mesh = MeshData::from_faces(positions, &faces);
    set_uvs(&mut mesh, uvs);
    mesh
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
    let mut uvs: Vec<Vec2> = Vec::with_capacity((nm * nn * 4) as usize);
    // Both angles wrap, and both come from their own index — so both seams get
    // `1` on one side and `0` on the other without anything having to work out
    // which face is the seam.
    let (fu, fv) = (1.0 / nm as f32, 1.0 / nn as f32);
    for i in 0..nm {
        for j in 0..nn {
            faces.push(vec![
                vid(i, j),
                vid(i + 1, j),
                vid(i + 1, j + 1),
                vid(i, j + 1),
            ]);
            uvs.extend_from_slice(&[
                Vec2::new(i as f32 * fu, j as f32 * fv),
                Vec2::new((i + 1) as f32 * fu, j as f32 * fv),
                Vec2::new((i + 1) as f32 * fu, (j + 1) as f32 * fv),
                Vec2::new(i as f32 * fu, (j + 1) as f32 * fv),
            ]);
        }
    }
    let mut mesh = MeshData::from_faces(positions, &faces);
    set_uvs(&mut mesh, uvs);
    mesh
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

/// A profile turned about Z: a lathe, for a vase, a cup, a bottle, a lamp.
///
/// `profile` is the silhouette as (radius, height) points, read from one end
/// to the other — bottom to top for the outside of a vase. A point on the axis
/// (radius zero) becomes one pole vertex, so a profile that starts and ends on
/// the axis gives a closed solid; one that ends off it stays open there, which
/// is the mouth of a cup. A profile that comes back down the inside gives the
/// wall a thickness.
///
/// Faces point the way the profile's right-hand side faces when walked from
/// the first point to the last with the axis on the left, which for a profile
/// drawn upward on the outside is outward.
pub fn lathe(profile: &[(f32, f32)], segments: u32) -> MeshData {
    let seg = segments.max(3);
    const ON_AXIS: f32 = 1e-6;
    // Each profile point's ring: one vertex on the axis, `seg` off it.
    let mut rings: Vec<Vec<u32>> = Vec::with_capacity(profile.len());
    let mut positions = Vec::new();
    for &(r, z) in profile {
        let r = r.max(0.0);
        let start = positions.len() as u32;
        if r <= ON_AXIS {
            positions.push(Vec3::new(0.0, 0.0, z));
            rings.push(vec![start; seg as usize]);
        } else {
            for s in 0..seg {
                let a = TAU * s as f32 / seg as f32;
                let (sa, ca) = a.sin_cos();
                positions.push(Vec3::new(r * ca, r * sa, z));
            }
            rings.push((start..start + seg).collect());
        }
    }
    // V along the profile by length, so a texture is not stretched where the
    // points are far apart.
    let mut along = vec![0.0f32];
    for pair in profile.windows(2) {
        let (a, b) = (
            Vec2::new(pair[0].0, pair[0].1),
            Vec2::new(pair[1].0, pair[1].1),
        );
        along.push(along.last().copied().unwrap_or(0.0) + a.distance(b));
    }
    let total = along.last().copied().unwrap_or(0.0).max(1e-9);

    let mut faces = Vec::new();
    let mut uvs = Vec::new();
    let fu = 1.0 / seg as f32;
    for i in 0..profile.len().saturating_sub(1) {
        let (lo, hi) = (&rings[i], &rings[i + 1]);
        let (v0, v1) = (along[i] / total, along[i + 1] / total);
        for s in 0..seg as usize {
            let s1 = (s + 1) % seg as usize;
            let quad = [lo[s], lo[s1], hi[s1], hi[s]];
            let corners = [
                Vec2::new(s as f32 * fu, v0),
                Vec2::new((s + 1) as f32 * fu, v0),
                Vec2::new((s + 1) as f32 * fu, v1),
                Vec2::new(s as f32 * fu, v1),
            ];
            // A pole collapses one side of the quad to a point: a triangle.
            let mut face = Vec::with_capacity(4);
            let mut face_uvs = Vec::with_capacity(4);
            for (k, v) in quad.iter().enumerate() {
                if !face.contains(v) {
                    face.push(*v);
                    face_uvs.push(corners[k]);
                }
            }
            if face.len() >= 3 {
                faces.push(face);
                uvs.extend(face_uvs);
            }
        }
    }
    let mut mesh = MeshData::from_faces(positions, &faces);
    set_uvs(&mut mesh, uvs);
    mesh
}

/// What [`sweep`] carries along its path.
#[derive(Clone, Debug)]
pub struct Sweep<'a> {
    /// The points the path passes through. Joined by a smooth curve through
    /// each of them, `resolution` steps between neighbours; a resolution of
    /// one joins them with straight lines.
    pub path: &'a [Vec3],
    /// The cross-section, in the plane across the path: x along the path's
    /// normal, y along its binormal. Closed: the last point joins the first.
    pub profile: &'a [Vec2],
    pub resolution: u32,
    /// The path is a loop: its end joins its start.
    pub closed: bool,
    /// The profile's size at the start and at the end, blended by distance
    /// along the path — a spout that narrows, a horn.
    pub scale: (f32, f32),
    /// Close the two ends of an open path with a face each.
    pub caps: bool,
}

/// A circle of `radius` in `segments`, the commonest profile: a tube.
pub fn circle_profile(radius: f32, segments: u32) -> Vec<Vec2> {
    let n = segments.max(3);
    (0..n)
        .map(|i| {
            let a = TAU * i as f32 / n as f32;
            Vec2::new(a.cos(), a.sin()) * radius
        })
        .collect()
}

/// A profile swept along a path: a pipe, a handle, a spout, a picture frame.
///
/// The cross-section is carried along the path on rotation-minimising frames
/// (Wang et al.'s double reflection), so it does not twist about the path
/// except where the path itself turns — the naive Frenet frame flips over at
/// every inflection and has no answer at all on a straight run. A closed path
/// has whatever twist is left over when the frame comes back round spread
/// evenly along it, so the ends meet.
pub fn sweep(s: &Sweep<'_>) -> MeshData {
    let points = smooth_path(s.path, s.resolution.max(1), s.closed);
    let mut profile: Vec<Vec2> = s.profile.to_vec();
    if points.len() < 2 || profile.len() < 2 {
        return MeshData::from_faces(Vec::new(), &[]);
    }
    // Counter-clockwise, so the walls face out whatever order it came in.
    let area: f32 = (0..profile.len())
        .map(|i| {
            let (a, b) = (profile[i], profile[(i + 1) % profile.len()]);
            a.x * b.y - b.x * a.y
        })
        .sum();
    if area < 0.0 {
        profile.reverse();
    }

    let n = points.len();
    let tangent = |i: usize| -> Vec3 {
        let (prev, next) = if s.closed {
            (points[(i + n - 1) % n], points[(i + 1) % n])
        } else {
            (points[i.saturating_sub(1)], points[(i + 1).min(n - 1)])
        };
        (next - prev).normalize_or_zero()
    };
    let tangents: Vec<Vec3> = (0..n).map(tangent).collect();
    let first = tangents[0];
    let up = if first.z.abs() < 0.9 {
        Vec3::Z
    } else {
        Vec3::X
    };
    let mut normals = vec![first.cross(up).cross(first).normalize_or_zero(); n];
    for i in 0..n - 1 {
        let v1 = points[i + 1] - points[i];
        let c1 = v1.dot(v1);
        if c1 < 1e-20 {
            normals[i + 1] = normals[i];
            continue;
        }
        let r_l = normals[i] - v1 * (2.0 / c1 * v1.dot(normals[i]));
        let t_l = tangents[i] - v1 * (2.0 / c1 * v1.dot(tangents[i]));
        let v2 = tangents[i + 1] - t_l;
        let c2 = v2.dot(v2);
        normals[i + 1] = if c2 < 1e-20 {
            r_l
        } else {
            r_l - v2 * (2.0 / c2 * v2.dot(r_l))
        };
    }
    if s.closed {
        // Carried once more round to the start, the frame comes back turned by
        // some angle; undo it a little at each step.
        let v1 = points[0] - points[n - 1];
        let c1 = v1.dot(v1).max(1e-20);
        let r_l = normals[n - 1] - v1 * (2.0 / c1 * v1.dot(normals[n - 1]));
        let t_l = tangents[n - 1] - v1 * (2.0 / c1 * v1.dot(tangents[n - 1]));
        let v2 = tangents[0] - t_l;
        let c2 = v2.dot(v2);
        let back = if c2 < 1e-20 {
            r_l
        } else {
            r_l - v2 * (2.0 / c2 * v2.dot(r_l))
        };
        let angle = back
            .cross(normals[0])
            .dot(tangents[0])
            .atan2(back.dot(normals[0]));
        for (i, normal) in normals.iter_mut().enumerate() {
            let turn = glam::Quat::from_axis_angle(tangents[i], angle * i as f32 / n as f32);
            *normal = turn * *normal;
        }
    }

    // Distance along the path, for the taper and for V.
    let mut along = vec![0.0f32; n];
    for i in 1..n {
        along[i] = along[i - 1] + points[i].distance(points[i - 1]);
    }
    let total = if s.closed {
        along[n - 1] + points[0].distance(points[n - 1])
    } else {
        along[n - 1]
    }
    .max(1e-9);

    let m = profile.len();
    let mut positions = Vec::with_capacity(n * m);
    for i in 0..n {
        let t = along[i] / total;
        let size = s.scale.0 + (s.scale.1 - s.scale.0) * t;
        let normal = normals[i].normalize_or_zero();
        let binormal = tangents[i].cross(normal).normalize_or_zero();
        for p in &profile {
            positions.push(points[i] + (normal * p.x + binormal * p.y) * size);
        }
    }

    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();
    let rings = if s.closed { n } else { n - 1 };
    let v_at = |i: usize| if i == n { 1.0 } else { along[i] / total };
    for i in 0..rings {
        let next = (i + 1) % n;
        for j in 0..m {
            let k = (j + 1) % m;
            let at = |ring: usize, p: usize| (ring * m + p) as u32;
            faces.push(vec![at(i, j), at(i, k), at(next, k), at(next, j)]);
            let (u0, u1) = (j as f32 / m as f32, (j + 1) as f32 / m as f32);
            let (v0, v1) = (v_at(i), v_at(i + 1));
            uvs.extend_from_slice(&[
                Vec2::new(u0, v0),
                Vec2::new(u1, v0),
                Vec2::new(u1, v1),
                Vec2::new(u0, v1),
            ]);
        }
    }
    if s.caps && !s.closed && m >= 3 {
        let bounds = profile.iter().fold(
            (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        let span = (bounds.1 - bounds.0).max(Vec2::splat(1e-9));
        let flat = |p: &Vec2| (*p - bounds.0) / span;
        // The start faces back along the path, the end along it.
        faces.push((0..m as u32).rev().collect());
        uvs.extend(profile.iter().rev().map(flat));
        faces.push(((n - 1) * m..n * m).map(|v| v as u32).collect());
        uvs.extend(profile.iter().map(flat));
    }
    let mut mesh = MeshData::from_faces(positions, &faces);
    set_uvs(&mut mesh, uvs);
    mesh
}

/// The path's points with `resolution` steps of a Catmull–Rom curve between
/// each pair: through every given point, with no overshoot to speak of.
pub fn smooth_path(points: &[Vec3], resolution: u32, closed: bool) -> Vec<Vec3> {
    let n = points.len();
    if n < 2 || resolution <= 1 {
        let mut out: Vec<Vec3> = points.to_vec();
        out.dedup();
        return out;
    }
    let at = |i: isize| -> Vec3 {
        if closed {
            points[i.rem_euclid(n as isize) as usize]
        } else {
            points[i.clamp(0, n as isize - 1) as usize]
        }
    };
    let spans = if closed { n } else { n - 1 };
    let mut out = Vec::with_capacity(spans * resolution as usize + 1);
    for span in 0..spans as isize {
        let (p0, p1, p2, p3) = (at(span - 1), at(span), at(span + 1), at(span + 2));
        for step in 0..resolution {
            let t = step as f32 / resolution as f32;
            let (t2, t3) = (t * t, t * t * t);
            out.push(
                (p1 * 2.0
                    + (p2 - p0) * t
                    + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
                    + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3)
                    * 0.5,
            );
        }
    }
    if !closed {
        out.push(points[n - 1]);
    }
    out
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

    // --- surface coordinates -------------------------------------------

    /// Every primitive that should carry coordinates, and the one that does not.
    fn uv_bearing() -> Vec<(&'static str, MeshData)> {
        vec![
            ("plane", plane(2.0)),
            ("grid", grid(2.0, 4, 3)),
            ("cube", cube(2.0)),
            ("uv_sphere", uv_sphere(1.0, 16, 8)),
            ("cylinder", cylinder(1.0, 2.0, 12)),
            ("cone", cone(1.0, 2.0, 12)),
            ("torus", torus(1.0, 0.3, 16, 8)),
        ]
    }

    #[test]
    fn a_primitive_arrives_with_somewhere_to_put_a_texture() {
        // A cube added from the menu had no UV layer at all, so a texture
        // assigned to it sampled one texel. Found by the renderer's first test
        // that put a picture on a surface — every test below this line in the
        // stack passed while it was true.
        for (name, mesh) in uv_bearing() {
            let uvs = mesh
                .attributes()
                .get_f32x2(names::UV)
                .unwrap_or_else(|| panic!("{name} has no UV layer"));
            assert_eq!(
                uvs.len(),
                mesh.num_corners() as usize,
                "{name}: one coordinate per corner"
            );
            // And they are not all the same point, which is what a layer of
            // zeroes would be — the state this is replacing.
            let first = uvs[0];
            assert!(
                uvs.iter().any(|uv| (*uv - first).length() > 1e-4),
                "{name}: every corner has the same coordinate"
            );
        }
    }

    #[test]
    fn the_coordinates_stay_inside_the_image() {
        // Sampling outside zero to one is a repeat, which is a real thing to
        // want and not what a primitive should arrive doing. The seam unwrap
        // pushes coordinates up by one on the face that spans the wrap, so its
        // right edge lands exactly on one rather than past it.
        for (name, mesh) in uv_bearing() {
            let uvs = mesh.attributes().get_f32x2(names::UV).expect("a layer");
            for uv in uvs {
                assert!(
                    (-1e-5..=1.0 + 1e-5).contains(&uv.x) && (-1e-5..=1.0 + 1e-5).contains(&uv.y),
                    "{name}: {uv:?} is outside the image"
                );
            }
        }
    }

    #[test]
    fn the_seam_does_not_run_backwards() {
        // **The reason coordinates are per corner.** A sphere's wrap needs
        // `u = 1` on one side of an edge and `u = 0` on the other, at the same
        // position — and without the unwrap the face there gets `0.98` and
        // `0.01`, which squeezes the whole texture backwards through one column
        // of quads. It is a thin stripe on a model, and it looks like a
        // texture problem rather than a mesh one.
        for (name, mesh) in [
            ("uv_sphere", uv_sphere(1.0, 16, 8)),
            ("cylinder", cylinder(1.0, 2.0, 12)),
            ("cone", cone(1.0, 2.0, 12)),
            ("torus", torus(1.0, 0.3, 16, 8)),
        ] {
            let uvs = mesh.attributes().get_f32x2(names::UV).expect("a layer");
            for f in 0..mesh.num_faces() {
                let range = mesh.face_corner_range(f);
                // Sides only. A cap is a disc and covers the image on purpose,
                // which is what this asked of it at first — the claim is about
                // the faces that use the angle, and those are the quads and
                // triangles the sides are made of.
                if range.len() > 4 {
                    continue;
                }
                let mut lo = f32::MAX;
                let mut hi = f32::MIN;
                for c in range.clone() {
                    lo = lo.min(uvs[c].x);
                    hi = hi.max(uvs[c].x);
                }
                // A side spans one segment, so anything near half the image is
                // the seam running the wrong way.
                assert!(
                    hi - lo < 0.5,
                    "{name}: face {f} spans {:.3} of the image, which is the seam",
                    hi - lo
                );
            }
        }
    }

    #[test]
    fn every_face_of_a_cube_gets_the_whole_square() {
        // What a checker on a fresh cube should look like. An unwrap that laid
        // the six faces out in one image is a choice the user makes; a
        // primitive making it for them would be a layout they then have to
        // undo.
        let mesh = cube(2.0);
        let uvs = mesh.attributes().get_f32x2(names::UV).expect("a layer");
        for f in 0..mesh.num_faces() {
            let corners: Vec<Vec2> = mesh.face_corner_range(f).map(|c| uvs[c]).collect();
            assert_eq!(corners.len(), 4);
            for wanted in [
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ] {
                assert!(
                    corners.iter().any(|c| (*c - wanted).length() < 1e-5),
                    "face {f} is missing the corner {wanted:?}"
                );
            }
        }
    }

    #[test]
    fn a_plane_is_the_image_itself() {
        // The simplest claim there is, and the one a person checks first: a
        // texture on a plane is that texture, the right way up.
        let mesh = plane(2.0);
        let uvs = mesh.attributes().get_f32x2(names::UV).expect("a layer");
        let positions = mesh.positions();
        for (c, v) in mesh.corner_verts().iter().enumerate() {
            let p = positions[*v as usize];
            let wanted = Vec2::new(p.x * 0.5 + 0.5, p.y * 0.5 + 0.5);
            assert!((uvs[c] - wanted).length() < 1e-5, "{:?}", uvs[c]);
        }
    }

    #[test]
    fn a_spheres_poles_do_not_fan_across_the_image() {
        // A pole is on the axis and has no longitude of its own —
        // `atan2(0, 0)` is zero, not "the same as its neighbours". Left at
        // zero, every triangle of the cap would reach back to the left edge of
        // the image and the top of a sphere would be a star.
        let mesh = uv_sphere(1.0, 16, 8);
        let uvs = mesh.attributes().get_f32x2(names::UV).expect("a layer");
        let positions = mesh.positions();
        for f in 0..mesh.num_faces() {
            let range = mesh.face_corner_range(f);
            for c in range.clone() {
                let p = positions[mesh.corner_verts()[c] as usize];
                if (p.x * p.x + p.y * p.y).sqrt() >= 1e-6 {
                    continue;
                }
                // A pole corner. Its `u` sits between its neighbours', and its
                // `v` is the edge of the image.
                let others: Vec<f32> = range
                    .clone()
                    .filter(|o| *o != c)
                    .map(|o| uvs[o].x)
                    .collect();
                let lo = others.iter().cloned().fold(f32::MAX, f32::min);
                let hi = others.iter().cloned().fold(f32::MIN, f32::max);
                assert!(
                    uvs[c].x >= lo - 1e-5 && uvs[c].x <= hi + 1e-5,
                    "the pole corner of face {f} is at {:.3}, outside its neighbours {lo:.3}..{hi:.3}",
                    uvs[c].x
                );
            }
        }
    }

    #[test]
    fn an_icosphere_has_no_coordinates_and_that_is_said_rather_than_faked() {
        // It has no natural parameterisation — that is the point of it, and the
        // reason it has no pole singularity. A spherical projection could be
        // put on one, and it would arrive with a seam and a pinch that somebody
        // then has to fix, which is worse than arriving with nothing and being
        // unwrapped on purpose.
        let mesh = ico_sphere(1.0, 2);
        assert!(mesh.attributes().get_f32x2(names::UV).is_none());
    }

    #[test]
    fn a_lathe_from_axis_to_axis_is_closed_and_faces_out() {
        // A vase: closed at the bottom, round, closed at the top.
        let mesh = lathe(
            &[(0.0, 0.0), (0.4, 0.0), (0.5, 0.6), (0.2, 1.0), (0.0, 1.0)],
            16,
        );
        mesh.assert_valid();
        assert_eq!(mesh.num_verts(), 2 + 3 * 16, "one vertex per pole");
        assert_eq!(mesh.num_faces(), 4 * 16);
        let normals = mesh.compute_face_normals();
        let centre = Vec3::new(0.0, 0.0, 0.5);
        for (f, n) in normals.iter().enumerate() {
            let ring = mesh.face_verts(f as u32);
            let mid: Vec3 = ring
                .iter()
                .map(|v| mesh.positions()[*v as usize])
                .sum::<Vec3>()
                / ring.len() as f32;
            assert!(n.dot(mid - centre) > 0.0, "face {f} points in");
        }
    }

    #[test]
    fn a_lathe_that_ends_off_the_axis_is_open_there() {
        // A cup: a bottom, a wall, no lid.
        let mesh = lathe(&[(0.0, 0.0), (0.3, 0.0), (0.35, 0.8)], 12);
        mesh.assert_valid();
        assert_eq!(mesh.num_faces(), 2 * 12);
        let b = mesh.bounds();
        assert!((b.max.z - 0.8).abs() < 1e-6 && (b.max.x - 0.35).abs() < 1e-6);
    }

    #[test]
    fn a_tube_swept_along_a_bend_is_closed_and_faces_out() {
        let path = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.5),
            Vec3::new(1.5, 0.0, 1.5),
            Vec3::new(1.2, 0.3, 2.2),
        ];
        let profile = circle_profile(0.1, 12);
        let mesh = sweep(&Sweep {
            path: &path,
            profile: &profile,
            resolution: 6,
            closed: false,
            scale: (1.0, 0.5),
            caps: true,
        });
        mesh.assert_valid();
        let rings = 3 * 6 + 1;
        assert_eq!(mesh.num_verts() as usize, rings * 12);
        assert_eq!(mesh.num_faces() as usize, (rings - 1) * 12 + 2);
        // Every edge has two faces: capped, it is a solid.
        let mut uses = std::collections::HashMap::new();
        for f in mesh.faces() {
            for i in 0..f.len() {
                let (a, b) = (f[i], f[(i + 1) % f.len()]);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(uses.values().all(|c| *c == 2));
        // The walls face away from the path: each wall's normal points from the
        // nearest path point out to the wall.
        let normals = mesh.compute_face_normals();
        let dense = smooth_path(&path, 6, false);
        for (f, normal) in normals.iter().enumerate().take((rings - 1) * 12) {
            let ring = mesh.face_verts(f as u32);
            let mid: Vec3 = ring
                .iter()
                .map(|v| mesh.positions()[*v as usize])
                .sum::<Vec3>()
                / 4.0;
            let near = dense
                .iter()
                .copied()
                .min_by(|a, b| a.distance(mid).total_cmp(&b.distance(mid)))
                .unwrap();
            assert!(normal.dot(mid - near) > 0.0, "wall {f} faces in");
        }
        // And it tapers: the last ring is half the first.
        let ring_size = |r: usize| mesh.positions()[r * 12].distance(mesh.positions()[r * 12 + 6]);
        assert!((ring_size(rings - 1) / ring_size(0) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn a_frame_swept_round_a_loop_closes_without_a_seam() {
        // A square profile round a rectangle: a picture frame.
        let path = [
            Vec3::new(-1.0, 0.0, -0.7),
            Vec3::new(1.0, 0.0, -0.7),
            Vec3::new(1.0, 0.0, 0.7),
            Vec3::new(-1.0, 0.0, 0.7),
        ];
        let profile = [
            Vec2::new(-0.05, -0.02),
            Vec2::new(0.05, -0.02),
            Vec2::new(0.05, 0.02),
            Vec2::new(-0.05, 0.02),
        ];
        let mesh = sweep(&Sweep {
            path: &path,
            profile: &profile,
            resolution: 1,
            closed: true,
            scale: (1.0, 1.0),
            caps: true,
        });
        mesh.assert_valid();
        assert_eq!(
            mesh.num_faces(),
            16,
            "four sides of four faces, no caps on a loop"
        );
        let mut uses = std::collections::HashMap::new();
        for f in mesh.faces() {
            for i in 0..f.len() {
                let (a, b) = (f[i], f[(i + 1) % f.len()]);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(uses.values().all(|c| *c == 2), "the loop's ends met");
    }
}
