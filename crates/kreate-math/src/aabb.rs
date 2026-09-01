use glam::Vec3;

/// An axis-aligned bounding box.
///
/// The empty box is represented by `min > max` on every axis, which makes
/// [`Aabb::EMPTY`] a correct identity for [`Aabb::union`] and lets
/// [`Aabb::extend`] build a box by folding without a special first-element case.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        points.into_iter().fold(Self::EMPTY, |b, p| b.extend(p))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    #[inline]
    #[must_use]
    pub fn extend(self, p: Vec3) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    #[inline]
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Full extent along each axis. Zero for an empty box rather than negative.
    #[inline]
    pub fn size(&self) -> Vec3 {
        (self.max - self.min).max(Vec3::ZERO)
    }

    /// Radius of the bounding sphere centred on [`Aabb::center`].
    #[inline]
    pub fn bounding_radius(&self) -> f32 {
        self.size().length() * 0.5
    }

    /// Index of the longest axis (0 = X, 1 = Y, 2 = Z). Used by BVH splitting.
    #[inline]
    pub fn max_extent_axis(&self) -> usize {
        let s = self.size();
        if s.x >= s.y && s.x >= s.z {
            0
        } else if s.y >= s.z {
            1
        } else {
            2
        }
    }

    /// Surface area, the cost metric for SAH BVH construction. Zero when empty.
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let s = self.size();
        2.0 * (s.x * s.y + s.y * s.z + s.z * s.x)
    }

    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.cmple(other.max).all() && self.max.cmpge(other.min).all()
    }

    #[inline]
    #[must_use]
    pub fn expanded(self, amount: f32) -> Self {
        Self {
            min: self.min - Vec3::splat(amount),
            max: self.max + Vec3::splat(amount),
        }
    }

    /// The 8 corners, in the bit order `x | y<<1 | z<<2`.
    pub fn corners(&self) -> [Vec3; 8] {
        let (n, x) = (self.min, self.max);
        [
            Vec3::new(n.x, n.y, n.z),
            Vec3::new(x.x, n.y, n.z),
            Vec3::new(n.x, x.y, n.z),
            Vec3::new(x.x, x.y, n.z),
            Vec3::new(n.x, n.y, x.z),
            Vec3::new(x.x, n.y, x.z),
            Vec3::new(n.x, x.y, x.z),
            Vec3::new(x.x, x.y, x.z),
        ]
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_union_identity() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(2.0));
        assert_eq!(Aabb::EMPTY.union(b), b);
        assert_eq!(b.union(Aabb::EMPTY), b);
        assert!(Aabb::EMPTY.is_empty());
    }

    #[test]
    fn empty_box_has_no_negative_size() {
        assert_eq!(Aabb::EMPTY.size(), Vec3::ZERO);
        assert_eq!(Aabb::EMPTY.surface_area(), 0.0);
    }

    #[test]
    fn from_points_bounds_all() {
        let pts = [Vec3::new(1.0, -2.0, 3.0), Vec3::new(-4.0, 5.0, 0.0)];
        let b = Aabb::from_points(pts);
        assert_eq!(b.min, Vec3::new(-4.0, -2.0, 0.0));
        assert_eq!(b.max, Vec3::new(1.0, 5.0, 3.0));
        assert!(pts.iter().all(|p| b.contains(*p)));
    }

    #[test]
    fn max_extent_axis_picks_longest() {
        let b = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 5.0, 2.0));
        assert_eq!(b.max_extent_axis(), 1);
    }
}
