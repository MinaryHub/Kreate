use glam::Vec3;

/// An oriented plane, stored as a unit normal and a signed offset such that
/// points on the plane satisfy `normal · p + d == 0`.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    #[inline]
    pub fn new(normal: Vec3, d: f32) -> Self {
        Self { normal, d }
    }

    #[inline]
    pub fn from_point_normal(point: Vec3, normal: Vec3) -> Self {
        let normal = normal.normalize();
        Self {
            normal,
            d: -normal.dot(point),
        }
    }

    /// Plane through three points, wound counter-clockwise when viewed from the
    /// normal side. Returns `None` for degenerate (collinear) input.
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Self> {
        let n = (b - a).cross(c - a);
        let len = n.length();
        (len > 1e-12).then(|| {
            let normal = n / len;
            Self {
                normal,
                d: -normal.dot(a),
            }
        })
    }

    /// Signed distance: positive in front of the plane, negative behind.
    #[inline]
    pub fn distance(&self, p: Vec3) -> f32 {
        self.normal.dot(p) + self.d
    }

    #[inline]
    pub fn project(&self, p: Vec3) -> Vec3 {
        p - self.normal * self.distance(p)
    }

    /// Parameter `t` at which a ray crosses the plane, or `None` when parallel.
    /// May be negative — the caller decides whether a hit behind the origin counts.
    #[inline]
    pub fn intersect_ray(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        let denom = self.normal.dot(dir);
        (denom.abs() > 1e-8).then(|| -(self.normal.dot(origin) + self.d) / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_plane_distance_is_height() {
        let p = Plane::from_point_normal(Vec3::ZERO, Vec3::Z);
        assert!((p.distance(Vec3::new(3.0, -2.0, 5.0)) - 5.0).abs() < 1e-6);
        assert!((p.distance(Vec3::new(0.0, 0.0, -1.5)) + 1.5).abs() < 1e-6);
    }

    #[test]
    fn from_points_is_ccw() {
        let p = Plane::from_points(Vec3::ZERO, Vec3::X, Vec3::Y).unwrap();
        assert!(p.normal.dot(Vec3::Z) > 0.99);
    }

    #[test]
    fn collinear_points_have_no_plane() {
        assert!(Plane::from_points(Vec3::ZERO, Vec3::X, Vec3::X * 2.0).is_none());
    }

    #[test]
    fn project_lands_on_plane() {
        let p = Plane::from_point_normal(Vec3::new(0.0, 0.0, 2.0), Vec3::Z);
        assert!(p.distance(p.project(Vec3::new(1.0, 1.0, 9.0))).abs() < 1e-6);
    }

    #[test]
    fn parallel_ray_does_not_intersect() {
        let p = Plane::from_point_normal(Vec3::ZERO, Vec3::Z);
        assert!(p.intersect_ray(Vec3::new(0.0, 0.0, 1.0), Vec3::X).is_none());
    }
}
