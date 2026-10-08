//! What cloth, soft bodies and particles collide with: a collision object's
//! triangles in the world, with a tree to find the near ones.
//!
//! Kept in the object's own space and moved with it frame by frame: the tree
//! is refitted rather than rebuilt (`Bvh::refit`), because a collider that is
//! animated moves every frame and the shape of its tree does not change.

use kreate_math::{Aabb, Bvh, DAffine3, Ray, Vec3};
use kreate_mesh::MeshData;

/// One collision object, where it is this frame.
#[derive(Clone, Debug)]
pub struct Collider {
    local: Vec<Vec3>,
    world: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    normals: Vec<Vec3>,
    bvh: Bvh,
    pub thickness: f32,
    pub friction: f32,
    pub bounciness: f32,
}

/// Where something touched a collider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Where it should be: on the surface, `thickness` out on the side it
    /// came from.
    pub point: Vec3,
    /// The surface's normal, turned to the side it came from.
    pub normal: Vec3,
}

impl Collider {
    /// A collider from a mesh in its own space, placed by `world`.
    pub fn new(
        mesh: &MeshData,
        world: DAffine3,
        thickness: f64,
        friction: f64,
        bounciness: f64,
    ) -> Self {
        let tri = mesh.triangulate();
        let (tris, _) = tri.indices.as_chunks::<3>();
        let tris = tris.to_vec();
        let local = mesh.positions().to_vec();
        let mut c = Self {
            world: local.clone(),
            bvh: Bvh::build(&local, &tris),
            local,
            normals: vec![Vec3::Z; tris.len()],
            tris,
            thickness: thickness.max(0.0) as f32,
            friction: friction.clamp(0.0, 1.0) as f32,
            bounciness: bounciness.clamp(0.0, 1.0) as f32,
        };
        c.place(world);
        c
    }

    /// Moves it to where `world` puts it.
    pub fn place(&mut self, world: DAffine3) {
        for (w, l) in self.world.iter_mut().zip(&self.local) {
            *w = world.transform_point3(l.as_dvec3()).as_vec3();
        }
        for (n, t) in self.normals.iter_mut().zip(&self.tris) {
            let [a, b, c] = t.map(|i| self.world[i as usize]);
            *n = (b - a).cross(c - a).normalize_or_zero();
        }
        self.bvh.refit(&self.world, &self.tris);
    }

    pub fn bounds(&self) -> Aabb {
        self.bvh.bounds()
    }

    /// The nearest point of the surface within `reach` of `p`, and the
    /// triangle's normal there.
    pub fn nearest(&self, p: Vec3, reach: f32) -> Option<(Vec3, Vec3)> {
        let query = Aabb {
            min: p - Vec3::splat(reach),
            max: p + Vec3::splat(reach),
        };
        let mut near = Vec::new();
        self.bvh.overlaps(&query, &mut near);
        let mut best: Option<(f32, Vec3, Vec3)> = None;
        for t in near {
            let [a, b, c] = self.tris[t as usize].map(|i| self.world[i as usize]);
            let q = closest_on_triangle(p, a, b, c);
            let d = q.distance_squared(p);
            if d <= reach * reach && best.is_none_or(|(bd, _, _)| d < bd) {
                best = Some((d, q, self.normals[t as usize]));
            }
        }
        best.map(|(_, q, n)| (q, n))
    }

    /// Whether going from `from` to `to` crosses the surface, and where.
    pub fn crossing(&self, from: Vec3, to: Vec3) -> Option<(Vec3, Vec3)> {
        let d = to - from;
        let len = d.length();
        if len < 1e-9 {
            return None;
        }
        let hit = self
            .bvh
            .raycast(&self.world, &self.tris, &Ray::new(from, d / len), len)?;
        Some((hit.point, self.normals[hit.primitive as usize]))
    }

    /// Where a point that went from `from` to `to` has to be instead, if the
    /// surface is in its way: through it, or nearer than the thickness.
    pub fn resolve(&self, from: Vec3, to: Vec3) -> Option<Contact> {
        let gap = self.thickness.max(1e-4);
        if let Some((hit, n)) = self.crossing(from, to) {
            // Out on the side it came from.
            let side = if (from - hit).dot(n) >= 0.0 { n } else { -n };
            return Some(Contact {
                point: hit + side * gap,
                normal: side,
            });
        }
        let (q, n) = self.nearest(to, gap)?;
        let away = to - q;
        // Which side it belongs on is where it was a moment ago, so a point
        // grazing a thin sheet is not flipped through it.
        let side = if (from - q).dot(n) >= 0.0 { n } else { -n };
        if away.dot(side) >= gap {
            return None;
        }
        Some(Contact {
            point: to + side * (gap - away.dot(side)),
            normal: side,
        })
    }
}

/// The point of triangle `abc` nearest `p`. Ericson's, §5.1.5.
pub fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor() -> Collider {
        let mesh = kreate_mesh::primitives::plane(4.0);
        Collider::new(&mesh, DAffine3::IDENTITY, 0.05, 0.0, 0.0)
    }

    #[test]
    fn the_nearest_point_of_a_triangle() {
        let (a, b, c) = (Vec3::ZERO, Vec3::X, Vec3::Y);
        assert!(
            closest_on_triangle(Vec3::new(0.2, 0.2, 5.0), a, b, c)
                .distance(Vec3::new(0.2, 0.2, 0.0))
                < 1e-6
        );
        assert_eq!(closest_on_triangle(Vec3::new(-1.0, -1.0, 0.0), a, b, c), a);
        assert_eq!(
            closest_on_triangle(Vec3::new(0.5, -3.0, 0.0), a, b, c),
            Vec3::new(0.5, 0.0, 0.0)
        );
    }

    #[test]
    fn a_point_falling_through_the_floor_is_put_back_on_top() {
        let f = floor();
        let c = f
            .resolve(Vec3::new(0.3, 0.1, 0.2), Vec3::new(0.3, 0.1, -0.4))
            .expect("it went through");
        assert!((c.point.z - 0.05).abs() < 1e-4, "{:?}", c.point);
        assert!(c.normal.z > 0.99);
    }

    #[test]
    fn a_point_from_below_stays_below_and_one_far_away_is_left_alone() {
        let f = floor();
        let c = f
            .resolve(Vec3::new(0.0, 0.0, -0.2), Vec3::new(0.0, 0.0, -0.01))
            .expect("inside the thickness");
        assert!((c.point.z + 0.05).abs() < 1e-4, "{:?}", c.point);
        assert!(f
            .resolve(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 0.5))
            .is_none());
        assert!(
            f.resolve(Vec3::new(9.0, 0.0, 1.0), Vec3::new(9.0, 0.0, -1.0))
                .is_none(),
            "beside the floor"
        );
    }

    #[test]
    fn a_moved_collider_is_where_it_was_moved() {
        let mut f = floor();
        f.place(DAffine3::from_translation(kreate_math::DVec3::new(
            0.0, 0.0, 2.0,
        )));
        let c = f
            .resolve(Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 1.0))
            .expect("hit");
        assert!((c.point.z - 2.05).abs() < 1e-4);
    }
}
