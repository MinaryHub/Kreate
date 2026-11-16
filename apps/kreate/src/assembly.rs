//! Where a joint goes on a part: Fusion's joint origins.
//!
//! A click on a part's face is caught by the place a joint is wanted at —
//! the middle of a hole the face has, when the click is by it; else the
//! middle of the flat face it is on, however many of the mesh's faces that
//! flat face is drawn as. Each with the way the face looks, which is the
//! joint's axis.

use kreate_math::{DAffine3, DVec3};
use kreate_mesh::MeshData;

/// A place a joint can go: a point and the way the face there looks, in
/// the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Origin {
    pub at: DVec3,
    pub normal: DVec3,
    /// Whether it is a hole's or a round edge's centre, not a face's middle.
    pub round: bool,
}

/// The joint origin a click at `hit` on face `face` of `mesh`, placed by
/// `world`, is caught by. `None` for a face that is not one.
pub fn origin_at(mesh: &MeshData, world: &DAffine3, face: u32, hit: DVec3) -> Option<Origin> {
    if face >= mesh.num_faces() {
        return None;
    }
    let p: Vec<DVec3> = mesh
        .positions()
        .iter()
        .map(|v| world.transform_point3(v.as_dvec3()))
        .collect();
    let normal_of = |f: u32| {
        let ring = mesh.face_verts(f);
        let mut n = DVec3::ZERO;
        for i in 0..ring.len() {
            n += p[ring[i] as usize].cross(p[ring[(i + 1) % ring.len()] as usize]);
        }
        n.normalize_or_zero()
    };
    let normal = normal_of(face);
    if normal == DVec3::ZERO {
        return None;
    }
    let ring0 = mesh.face_verts(face);
    let plane = p[ring0[0] as usize].dot(normal);
    let size = p
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs().max_element()))
        .max(1e-9);
    let flat_with = |f: u32| {
        normal_of(f).dot(normal) > 0.9999
            && mesh
                .face_verts(f)
                .iter()
                .all(|v| (p[*v as usize].dot(normal) - plane).abs() <= size * 1e-6)
    };
    // The flat face the click is on: the faces joined to it edge by edge
    // that lie in its plane and look its way.
    let mut edge_faces: std::collections::HashMap<(u32, u32), Vec<u32>> = Default::default();
    for f in 0..mesh.num_faces() {
        let ring = mesh.face_verts(f);
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            edge_faces.entry((a.min(b), a.max(b))).or_default().push(f);
        }
    }
    let mut region = vec![face];
    let mut seen: std::collections::HashSet<u32> = [face].into_iter().collect();
    let mut k = 0;
    while k < region.len() {
        let ring = mesh.face_verts(region[k]);
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            for &g in &edge_faces[&(a.min(b), a.max(b))] {
                if !seen.contains(&g) && flat_with(g) {
                    seen.insert(g);
                    region.push(g);
                }
            }
        }
        k += 1;
    }
    // Its middle, by area.
    let mut area = 0.0;
    let mut middle = DVec3::ZERO;
    for &f in &region {
        let ring = mesh.face_verts(f);
        let a = p[ring[0] as usize];
        for i in 1..ring.len().saturating_sub(1) {
            let (b, c) = (p[ring[i] as usize], p[ring[i + 1] as usize]);
            let t = (b - a).cross(c - a).dot(normal) * 0.5;
            area += t;
            middle += (a + b + c) / 3.0 * t;
        }
    }
    let middle = if area.abs() > 1e-30 {
        middle / area
    } else {
        p[ring0[0] as usize]
    };
    // Its edges that are its alone — its outline and its holes' rims —
    // chained into loops.
    let mut boundary: Vec<(u32, u32)> = Vec::new();
    for &f in &region {
        let ring = mesh.face_verts(f);
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            let inside = edge_faces[&(a.min(b), a.max(b))]
                .iter()
                .filter(|g| seen.contains(g))
                .count();
            if inside == 1 {
                boundary.push((a, b));
            }
        }
    }
    let mut loops: Vec<Vec<u32>> = Vec::new();
    let mut left = boundary.clone();
    while let Some((start, mut next)) = left.pop() {
        let mut ring = vec![start];
        while next != start {
            ring.push(next);
            let Some(i) = left.iter().position(|(a, _)| *a == next) else {
                break;
            };
            next = left.swap_remove(i).1;
        }
        if ring.len() >= 3 {
            loops.push(ring);
        }
    }
    // A loop that is a circle: its points all as far from its middle.
    let mut best: Option<(f64, DVec3)> = None;
    for ring in &loops {
        if ring.len() < 8 {
            continue;
        }
        let c = ring.iter().map(|v| p[*v as usize]).sum::<DVec3>() / ring.len() as f64;
        let radii: Vec<f64> = ring.iter().map(|v| p[*v as usize].distance(c)).collect();
        let r = radii.iter().sum::<f64>() / radii.len() as f64;
        if r <= 0.0 || radii.iter().any(|x| (x - r).abs() > r * 0.03) {
            continue;
        }
        // Caught when the click is within half again its radius.
        let d = hit.distance(c);
        if d <= r * 1.5 && best.is_none_or(|(b, _)| d < b) {
            best = Some((d, c));
        }
    }
    Some(match best {
        Some((_, c)) => Origin {
            at: c,
            normal,
            round: true,
        },
        None => Origin {
            at: middle,
            normal,
            round: false,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::Vec3;

    /// A plate's top with a round hole through it, as faces: a fan of
    /// quads from the hole's rim out to the plate's square outline.
    fn plate_with_hole() -> MeshData {
        let n = 24;
        let mut points: Vec<Vec3> = Vec::new();
        for i in 0..n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            points.push(Vec3::new(a.cos() * 0.005, a.sin() * 0.005, 0.0));
        }
        for i in 0..n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            let s = a.cos().abs().max(a.sin().abs());
            points.push(Vec3::new(a.cos() / s * 0.02, a.sin() / s * 0.02, 0.0));
        }
        let n = n as u32;
        let faces: Vec<Vec<u32>> = (0..n)
            .map(|i| vec![i, n + i, n + (i + 1) % n, (i + 1) % n])
            .collect();
        MeshData::from_faces(points, &faces)
    }

    /// A click by the hole is caught by its centre; one out on the plate,
    /// by the plate's middle — which, with the hole in the middle, is the
    /// same place, so the plate is moved off the origin to tell them apart.
    #[test]
    fn a_click_by_a_hole_finds_its_centre_and_elsewhere_the_faces_middle() {
        let mesh = plate_with_hole();
        let world = DAffine3::from_translation(DVec3::new(1.0, 2.0, 3.0));
        let by_hole = origin_at(&mesh, &world, 0, DVec3::new(1.006, 2.0, 3.0)).unwrap();
        assert!(by_hole.round);
        assert!(
            by_hole.at.distance(DVec3::new(1.0, 2.0, 3.0)) < 1e-6,
            "{by_hole:?}"
        );
        assert!((by_hole.normal.z.abs() - 1.0).abs() < 1e-9);
        let out = origin_at(&mesh, &world, 3, DVec3::new(1.018, 2.017, 3.0)).unwrap();
        assert!(!out.round);
        // The square plate's middle, all its faces taken together.
        assert!(out.at.distance(DVec3::new(1.0, 2.0, 3.0)) < 1e-4, "{out:?}");
    }

    /// A box's top is caught at the middle of the whole top, not of the
    /// one face clicked.
    #[test]
    fn a_flat_face_drawn_as_several_is_one_face() {
        let points = vec![
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(2.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 1.0, 1.0),
        ];
        let mesh = MeshData::from_faces(points, &[vec![0, 1, 4, 3], vec![1, 2, 5, 4]]);
        let o = origin_at(&mesh, &DAffine3::IDENTITY, 1, DVec3::new(1.5, 0.5, 1.0)).unwrap();
        assert!(o.at.distance(DVec3::new(1.0, 0.5, 1.0)) < 1e-9, "{o:?}");
    }
}
