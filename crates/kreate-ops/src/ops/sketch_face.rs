//! Where a sketch on a face goes: the plane of a flat face of a part, its
//! axes square to the face's longest edge, and the outline of the face — and
//! of any hole in it — to draw against, as Fusion projects a face's edges
//! into a sketch made on it.
//!
//! "The face" is what a person means by it, not one polygon: a flat top
//! that a boolean or an import left as many triangles is all of those
//! triangles that lie in its plane and touch one another.

use kreate_math::{DAffine3, DMat3, DQuat, DVec2, DVec3, Vec3};
use kreate_mesh::MeshData;

/// Faces meeting at no more than this cosine's angle — thirty-one degrees,
/// a twelve-sided cylinder's — are one curved surface, not two flat ones.
const SMOOTH: f64 = 0.857;

/// A sketch plane on a face, in the world.
#[derive(Clone, Debug)]
pub struct FaceFrame {
    pub origin: DVec3,
    /// The sketch's X, Y and normal as the columns of its turn.
    pub rotation: DQuat,
    /// The face's outline and its holes' outlines, in the sketch's plane,
    /// the outer one first.
    pub loops: Vec<Vec<DVec2>>,
}

/// The flat face of `mesh` (placed in the world by `to_world`) that `face` is
/// part of, as a sketch plane. Its origin is `at` brought onto the plane, or
/// the middle of the outline when `at` is `None`. An error for a face that is
/// not flat — a cylinder's side — which no flat sketch lies on.
pub fn face_frame(
    mesh: &MeshData,
    to_world: DAffine3,
    face: u32,
    at: Option<DVec3>,
) -> Result<FaceFrame, String> {
    if face >= mesh.num_faces() {
        return Err(format!("the mesh has no face {face}"));
    }
    let world: Vec<DVec3> = mesh
        .positions()
        .iter()
        .map(|p| to_world.transform_point3(p.as_dvec3()))
        .collect();
    let (lo, hi) = world.iter().fold(
        (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let tol = (hi - lo).length().max(1e-9) * 1e-6;
    let newell = |f: u32| {
        let ring = mesh.face_verts(f);
        let mut n = DVec3::ZERO;
        for i in 0..ring.len() {
            let (p, q) = (
                world[ring[i] as usize],
                world[ring[(i + 1) % ring.len()] as usize],
            );
            n += p.cross(q);
        }
        n
    };
    let normal = newell(face).normalize_or_zero();
    if normal == DVec3::ZERO {
        return Err(format!("face {face} has no area"));
    }
    let seed = world[mesh.face_verts(face)[0] as usize];
    let on_plane = |f: u32| {
        mesh.face_verts(f)
            .iter()
            .all(|v| (world[*v as usize] - seed).dot(normal).abs() <= tol)
    };
    if !on_plane(face) {
        return Err(format!(
            "face {face} is not flat: a sketch goes on a flat face (put the 3D cursor on a curved one and sketch on the cursor's plane)"
        ));
    }
    // The faces in this plane that touch the one picked, across edges.
    let mut owner: std::collections::HashMap<(u32, u32), Vec<u32>> = Default::default();
    for f in 0..mesh.num_faces() {
        let r = mesh.face_verts(f);
        for i in 0..r.len() {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            owner.entry((a.min(b), a.max(b))).or_default().push(f);
        }
    }
    let mut patch = vec![false; mesh.num_faces() as usize];
    let mut work = vec![face];
    patch[face as usize] = true;
    while let Some(f) = work.pop() {
        let r = mesh.face_verts(f);
        for i in 0..r.len() {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            for g in &owner[&(a.min(b), a.max(b))] {
                if !patch[*g as usize]
                    && newell(*g).normalize_or_zero().dot(normal) > 1.0 - 1e-6
                    && on_plane(*g)
                {
                    patch[*g as usize] = true;
                    work.push(*g);
                }
            }
        }
    }
    // A facet of a curved surface — a cylinder's side cut into strips — is
    // flat by itself but meets its neighbours at a slight angle: a sketch
    // on it would be a sketch on one strip of a round wall.
    let curved = (0..mesh.num_faces())
        .filter(|f| patch[*f as usize])
        .any(|f| {
            let r = mesh.face_verts(f);
            (0..r.len()).any(|i| {
                let (a, b) = (r[i], r[(i + 1) % r.len()]);
                owner[&(a.min(b), a.max(b))].iter().any(|g| {
                    !patch[*g as usize] && {
                        let c = newell(*g).normalize_or_zero().dot(normal);
                        c > SMOOTH && c < 1.0 - 1e-6
                    }
                })
            })
        });
    if curved {
        return Err(format!(
            "face {face} is part of a curved surface: a sketch goes on a flat face (put the 3D cursor on a curved one and sketch on the cursor's plane)"
        ));
    }
    // Its rim: the edges of the patch no other face of it shares, chained.
    let mut next: std::collections::HashMap<u32, u32> = Default::default();
    for f in (0..mesh.num_faces()).filter(|f| patch[*f as usize]) {
        let r = mesh.face_verts(f);
        for i in 0..r.len() {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            let inner = owner[&(a.min(b), a.max(b))]
                .iter()
                .filter(|g| patch[**g as usize])
                .count()
                > 1;
            if !inner {
                next.insert(a, b);
            }
        }
    }
    let mut loops: Vec<Vec<u32>> = Vec::new();
    while let Some(&start) = next.keys().next() {
        let mut ring = vec![start];
        let mut at_v = start;
        while let Some(b) = next.remove(&at_v) {
            if b == start {
                break;
            }
            ring.push(b);
            at_v = b;
        }
        if ring.len() >= 3 {
            loops.push(ring);
        }
    }
    if loops.is_empty() {
        return Err("the face has no outline".into());
    }
    // The outer outline is the longest way round.
    let length = |l: &Vec<u32>| {
        (0..l.len())
            .map(|i| world[l[i] as usize].distance(world[l[(i + 1) % l.len()] as usize]))
            .sum::<f64>()
    };
    loops.sort_by(|a, b| length(b).total_cmp(&length(a)));
    // X along the outline's longest edge, laid in the plane.
    let outer = &loops[0];
    let (mut x, mut longest) = (DVec3::ZERO, 0.0);
    for i in 0..outer.len() {
        let d = world[outer[(i + 1) % outer.len()] as usize] - world[outer[i] as usize];
        let d = d - normal * d.dot(normal);
        if d.length() > longest {
            longest = d.length();
            x = d / longest;
        }
    }
    let y = normal.cross(x);
    let rotation = DQuat::from_mat3(&DMat3::from_cols(x, y, normal)).normalize();
    let middle = outer.iter().map(|v| world[*v as usize]).sum::<DVec3>() / outer.len() as f64;
    let origin = match at {
        Some(p) => p - normal * (p - seed).dot(normal),
        None => middle - normal * (middle - seed).dot(normal),
    };
    let to_plane = |p: DVec3| {
        let d = p - origin;
        DVec2::new(d.dot(x), d.dot(y))
    };
    Ok(FaceFrame {
        origin,
        rotation,
        loops: loops
            .iter()
            .map(|l| l.iter().map(|v| to_plane(world[*v as usize])).collect())
            .collect(),
    })
}

/// The face of `mesh` (in the world by `to_world`) that `point` lies on: in
/// its plane and inside its outline. The nearest such face, when the point
/// is near an edge two share.
pub fn face_at(mesh: &MeshData, to_world: DAffine3, point: DVec3) -> Option<u32> {
    let world: Vec<DVec3> = mesh
        .positions()
        .iter()
        .map(|p| to_world.transform_point3(p.as_dvec3()))
        .collect();
    let (lo, hi) = world.iter().fold(
        (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let tol = (hi - lo).length().max(1e-9) * 1e-4;
    let mut best: Option<(f64, u32)> = None;
    let mut tris = Vec::new();
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        let mut n = DVec3::ZERO;
        for i in 0..ring.len() {
            n += world[ring[i] as usize].cross(world[ring[(i + 1) % ring.len()] as usize]);
        }
        let n = n.normalize_or_zero();
        if n == DVec3::ZERO {
            continue;
        }
        let gap = (point - world[ring[0] as usize]).dot(n).abs();
        if gap > tol || best.is_some_and(|(g, _)| g <= gap) {
            continue;
        }
        // Inside: in one of the face's triangles, seen along its normal.
        let pts: Vec<Vec3> = ring.iter().map(|v| world[*v as usize].as_vec3()).collect();
        tris.clear();
        kreate_mesh::tessellate::triangulate_polygon(&pts, &mut tris);
        let p = point - n * (point - world[ring[0] as usize]).dot(n);
        let inside = tris.iter().any(|t| {
            let [a, b, c] = t.map(|i| world[ring[i as usize] as usize]);
            let s = |u: DVec3, v: DVec3| (v - u).cross(p - u).dot(n);
            let (s1, s2, s3) = (s(a, b), s(b, c), s(c, a));
            let e = -tol * tol;
            (s1 >= e && s2 >= e && s3 >= e) || (s1 <= -e && s2 <= -e && s3 <= -e)
        });
        if inside {
            best = Some((gap, f));
        }
    }
    best.map(|(_, f)| f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The top of a box made of triangles is one face: its outline the
    /// box's square, the sketch's normal up, the origin where it was asked.
    #[test]
    fn a_triangulated_top_is_one_face_with_its_square_outline() {
        let cube = kreate_mesh::primitives::cube(2.0);
        let tri = cube.triangulate();
        let (t, _) = tri.indices.as_chunks::<3>();
        let faces: Vec<Vec<u32>> = t.iter().map(|t| t.to_vec()).collect();
        let mesh = MeshData::from_faces(cube.positions().to_vec(), &faces);
        let up = DAffine3::from_translation(DVec3::new(5.0, 0.0, 0.0));
        let at = DVec3::new(5.5, 0.2, 1.0);
        let face = face_at(&mesh, up, at).expect("a face under the point");
        let frame = face_frame(&mesh, up, face, Some(at)).unwrap();
        assert!((frame.rotation * DVec3::Z - DVec3::Z).length() < 1e-9);
        assert!(frame.origin.distance(at) < 1e-9);
        assert_eq!(frame.loops.len(), 1);
        assert_eq!(frame.loops[0].len(), 4, "{:?}", frame.loops);
        // The square, two across, around (-0.5, -0.2) from the origin.
        let (lo, hi) = frame.loops[0].iter().fold(
            (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        assert!(((hi - lo) - DVec2::splat(2.0)).length() < 1e-9, "{lo} {hi}");
    }

    /// A plate with a hole through it: the top is its outline and the
    /// hole's; the hole's wall is no flat face to sketch on.
    #[test]
    fn a_top_with_a_hole_has_both_outlines_and_a_round_wall_is_refused() {
        let mut plate = kreate_mesh::primitives::cube(1.0);
        for p in plate.positions_mut() {
            *p *= Vec3::new(0.06, 0.04, 0.003);
        }
        let hole = kreate_mesh::primitives::cylinder(0.005, 0.01, 24);
        let cut = kreate_mesh::modifiers::boolean::boolean(
            &plate,
            &hole,
            kreate_mesh::modifiers::BooleanOp::Difference,
        );
        let id = DAffine3::IDENTITY;
        let top = face_at(&cut, id, DVec3::new(0.02, 0.0, 0.0015)).unwrap();
        let frame = face_frame(&cut, id, top, None).unwrap();
        assert_eq!(frame.loops.len(), 2, "outline and hole");
        assert_eq!(frame.loops[1].len(), 24);
        let wall = face_at(&cut, id, DVec3::new(0.005, 0.0, 0.0)).expect("the hole's wall");
        assert!(face_frame(&cut, id, wall, None)
            .unwrap_err()
            .contains("curved"));
    }
}
