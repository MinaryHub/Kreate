use crate::Aabb;
use glam::Vec3;

/// A ray with a normalised direction.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
    /// Componentwise `1.0 / dir`, cached for slab tests. Infinities are intended
    /// here: an axis-parallel ray yields `inf`, and the slab test's min/max
    /// ordering handles that correctly, whereas branching on zero does not.
    inv_dir: Vec3,
}

impl Ray {
    /// Builds a ray, normalising `dir`.
    #[inline]
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        Self::new_normalized(origin, dir.normalize())
    }

    /// Builds a ray from an already-normalised direction.
    #[inline]
    pub fn new_normalized(origin: Vec3, dir: Vec3) -> Self {
        debug_assert!(
            (dir.length_squared() - 1.0).abs() < 1e-4,
            "direction must be normalized, got {dir}"
        );
        Self {
            origin,
            dir,
            inv_dir: dir.recip(),
        }
    }

    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }

    /// Slab test against an AABB. Returns the entry/exit parameters clipped to
    /// `[t_min, t_max]`, or `None` when the ray misses.
    #[inline]
    pub fn intersect_aabb(&self, aabb: &Aabb, t_min: f32, t_max: f32) -> Option<(f32, f32)> {
        let t0 = (aabb.min - self.origin) * self.inv_dir;
        let t1 = (aabb.max - self.origin) * self.inv_dir;
        let near = t0.min(t1);
        let far = t0.max(t1);
        let enter = near.max_element().max(t_min);
        let exit = far.min_element().min(t_max);
        (enter <= exit).then_some((enter, exit))
    }

    /// Möller-Trumbore ray/triangle intersection.
    ///
    /// Returns `(t, u, v)` where `u`/`v` are barycentric coordinates of `b`/`c`.
    /// Backfaces hit too: an editor must be able to click a face from behind, and
    /// visibility is decided by the depth buffer, not here.
    pub fn intersect_triangle(&self, a: Vec3, b: Vec3, c: Vec3) -> Option<(f32, f32, f32)> {
        const EPS: f32 = 1e-8;
        let e1 = b - a;
        let e2 = c - a;
        let pvec = self.dir.cross(e2);
        let det = e1.dot(pvec);
        if det.abs() < EPS {
            return None; // ray parallel to the triangle plane
        }
        let inv_det = 1.0 / det;
        let tvec = self.origin - a;
        let u = tvec.dot(pvec) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let qvec = tvec.cross(e1);
        let v = self.dir.dot(qvec) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = e2.dot(qvec) * inv_det;
        (t >= 0.0).then_some((t, u, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri() -> (Vec3, Vec3, Vec3) {
        (Vec3::ZERO, Vec3::X, Vec3::Y)
    }

    #[test]
    fn hits_triangle_centroid() {
        let (a, b, c) = tri();
        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), -Vec3::Z);
        let (t, u, v) = ray.intersect_triangle(a, b, c).expect("should hit");
        assert!((t - 1.0).abs() < 1e-5);
        assert!((u - 0.25).abs() < 1e-5 && (v - 0.25).abs() < 1e-5);
    }

    #[test]
    fn misses_outside_triangle() {
        let (a, b, c) = tri();
        let ray = Ray::new(Vec3::new(0.9, 0.9, 1.0), -Vec3::Z);
        assert!(ray.intersect_triangle(a, b, c).is_none());
    }

    #[test]
    fn hits_backfaces() {
        let (a, b, c) = tri();
        let ray = Ray::new(Vec3::new(0.25, 0.25, -1.0), Vec3::Z);
        assert!(ray.intersect_triangle(a, b, c).is_some());
    }

    #[test]
    fn axis_parallel_ray_hits_aabb() {
        // The inv_dir infinities must not poison the slab test.
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, -10.0), Vec3::Z);
        let (enter, exit) = ray
            .intersect_aabb(&aabb, 0.0, f32::INFINITY)
            .expect("should hit");
        assert!((enter - 9.0).abs() < 1e-4 && (exit - 11.0).abs() < 1e-4);
    }

    #[test]
    fn ray_pointing_away_misses_aabb() {
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, -10.0), -Vec3::Z);
        assert!(ray.intersect_aabb(&aabb, 0.0, f32::INFINITY).is_none());
    }
}
