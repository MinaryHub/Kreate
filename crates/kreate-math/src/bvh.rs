//! A bounding volume hierarchy over triangles.
//!
//! Written rather than taken from a physics crate, for reasons specific to a
//! modeller. A BVH here is not only a ray accelerator: it answers "what is the
//! nearest vertex to this point", it must **refit** in place when only positions
//! change (which is every frame of a drag), and its node layout has to stay
//! small enough to live beside a ten-million-triangle mesh. A physics library's
//! API fights all three.
//!
//! # What it is for
//!
//! The plan is explicit that this is a *separate service* from GPU id-buffer
//! picking, not a replacement for it. The id buffer answers "what did the user
//! click, as they see it", with occlusion for free. This answers "where does
//! this ray meet the surface, exactly", which is what snapping, knife
//! projection, loop-cut placement, measurement, headless scripting and — first —
//! the camera's auto-depth pivot all need. Neither can do the other's job.
//!
//! # Build
//!
//! Binned surface-area heuristic, 12 bins on the widest axis of the centroid
//! bounds. SAH rather than a median split because a modeller's geometry is not
//! uniformly distributed: a character mesh is mostly head and hands, and a
//! median split builds a tree that is balanced by count and terrible by area.
//!
//! Binning rather than a full sweep because a full sweep is `O(n log² n)` with a
//! sort at every level, and at a million triangles that is the difference
//! between a build the user does not notice and one they do.

use crate::{Aabb, Ray, Vec3};

/// A node. Interior nodes store the index of their left child; the right child
/// is always the next one, so only one link is needed.
#[derive(Clone, Copy, Debug)]
struct Node {
    bounds: Aabb,
    /// Leaf: the first primitive in [`Bvh::order`]. Interior: the left child.
    first: u32,
    /// Number of primitives. Zero marks an interior node.
    count: u32,
}

impl Node {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

/// What a ray hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Index into the triangle list the tree was built over.
    pub primitive: u32,
    /// Distance along the ray.
    pub t: f32,
    /// Position of the hit.
    pub point: Vec3,
    /// Barycentric coordinates within the triangle, as `(u, v)` for corners
    /// `b` and `c`. The `a` weight is `1 - u - v`.
    ///
    /// Carried because it is what a snap or a knife needs next, and recovering
    /// it afterwards means redoing the intersection.
    pub barycentric: (f32, f32),
}

/// A bounding volume hierarchy over an indexed triangle list.
#[derive(Clone, Debug, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    /// Primitive indices, permuted by the build so a leaf is a contiguous range.
    order: Vec<u32>,
}

/// Leaves below this size are not worth splitting: the traversal cost of the
/// extra level exceeds the intersection cost of testing them all.
const MAX_LEAF: usize = 4;

/// Bins per split candidate. Twelve is the usual sweet spot — the SAH cost
/// curve is flat enough that more bins buy almost nothing and cost a pass.
const BINS: usize = 12;

impl Bvh {
    /// Builds a tree over `indices` interpreted as triangles into `positions`.
    ///
    /// Degenerate and out-of-range triangles are skipped rather than rejected: a
    /// mesh mid-edit legitimately contains both, and a spatial index that
    /// refuses to build is a tool that stops working exactly when the user is
    /// in the middle of something.
    pub fn build(positions: &[Vec3], indices: &[[u32; 3]]) -> Self {
        let mut order: Vec<u32> = Vec::with_capacity(indices.len());
        let mut bounds: Vec<Aabb> = Vec::with_capacity(indices.len());
        let mut centroids: Vec<Vec3> = Vec::with_capacity(indices.len());
        for (i, tri) in indices.iter().enumerate() {
            let Some([a, b, c]) = fetch(positions, *tri) else {
                continue;
            };
            if !a.is_finite() || !b.is_finite() || !c.is_finite() {
                continue;
            }
            order.push(i as u32);
            let bb = Aabb::EMPTY.extend(a).extend(b).extend(c);
            centroids.push(bb.center());
            bounds.push(bb);
        }
        if order.is_empty() {
            return Self::default();
        }

        // `bounds` and `centroids` are indexed by *position in `order`*, so they
        // permute along with it and no lookup is needed during the build.
        let mut tree = Self {
            nodes: Vec::with_capacity(order.len() * 2),
            order,
        };
        tree.nodes.push(Node {
            bounds: Aabb::EMPTY,
            first: 0,
            count: tree.order.len() as u32,
        });
        let count = tree.order.len();
        tree.split(0, 0, count, &mut bounds, &mut centroids);
        tree
    }

    /// Recomputes every node's bounds from moved positions, keeping the tree.
    ///
    /// The reason this exists: during a drag the topology does not change, so
    /// the tree's *shape* is still valid even though its boxes are not. A refit
    /// is a linear bottom-up pass; a rebuild is not, and doing one per frame is
    /// how an interactive tool starts to stutter on a large mesh.
    ///
    /// The shape degrades as the geometry moves far from where it was built. A
    /// caller dragging a long way should rebuild when the drag ends, which is
    /// exactly when there is time to.
    pub fn refit(&mut self, positions: &[Vec3], indices: &[[u32; 3]]) {
        if self.nodes.is_empty() {
            return;
        }
        let order = &self.order;
        let leaf = |node: &mut Node| {
            if !node.is_leaf() {
                return;
            }
            let mut bb = Aabb::EMPTY;
            for slot in node.first..node.first + node.count {
                let Some(&prim) = order.get(slot as usize) else {
                    continue;
                };
                let Some(tri) = indices.get(prim as usize) else {
                    continue;
                };
                if let Some([a, b, c]) = fetch(positions, *tri) {
                    bb = bb.extend(a).extend(b).extend(c);
                }
            }
            node.bounds = bb;
        };
        // The leaves are nearly all the work — a gather of every triangle's
        // corners — and each depends on nothing but the positions, so a large
        // tree has them shared out across the cores. A sculpt stroke refits
        // every frame, and on one core a quarter-million triangles is most of
        // the frame.
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(8);
        if threads > 1 && self.nodes.len() >= 8192 {
            let chunk = self.nodes.len().div_ceil(threads);
            std::thread::scope(|scope| {
                for part in self.nodes.chunks_mut(chunk) {
                    scope.spawn(move || part.iter_mut().for_each(leaf));
                }
            });
        } else {
            self.nodes.iter_mut().for_each(leaf);
        }
        // Children always come after their parent, so a reverse pass sees
        // them already updated.
        for i in (0..self.nodes.len()).rev() {
            let node = self.nodes[i];
            if !node.is_leaf() {
                let left = node.first as usize;
                self.nodes[i].bounds = self.nodes[left].bounds.union(self.nodes[left + 1].bounds);
            }
        }
    }

    #[inline]
    /// Every node, flattened for a GPU walk: its bounds, then the first
    /// index into [`Bvh::order`] for a leaf or the left child for an interior
    /// node (the right child is the next one), then the leaf's primitive
    /// count, zero for an interior node.
    pub fn flat_nodes(&self) -> Vec<(Aabb, u32, u32)> {
        self.nodes
            .iter()
            .map(|n| (n.bounds, n.first, n.count))
            .collect()
    }

    /// The primitives in leaf order: a leaf's are a contiguous run of these.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of triangles the tree covers.
    #[inline]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// The root's bounds, or an empty box.
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map(|n| n.bounds).unwrap_or(Aabb::EMPTY)
    }

    /// Bytes held by the tree's own arrays.
    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<Node>() + self.order.len() * 4
    }

    /// The nearest triangle the ray meets, if any.
    ///
    /// Front-to-back traversal with the current best distance as the cutoff, so
    /// a ray into a dense mesh visits a handful of nodes rather than the tree.
    pub fn raycast(
        &self,
        positions: &[Vec3],
        indices: &[[u32; 3]],
        ray: &Ray,
        t_max: f32,
    ) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<Hit> = None;
        let mut limit = t_max;
        // An explicit stack rather than recursion: the depth is data-dependent,
        // and a pathological mesh must not overflow the thread's stack.
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);

        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if ray.intersect_aabb(&node.bounds, 0.0, limit).is_none() {
                continue;
            }
            if node.is_leaf() {
                for slot in node.first..node.first + node.count {
                    let prim = self.order[slot as usize];
                    let Some(tri) = indices.get(prim as usize) else {
                        continue;
                    };
                    let Some([a, b, c]) = fetch(positions, *tri) else {
                        continue;
                    };
                    let Some((t, u, v)) = ray.intersect_triangle(a, b, c) else {
                        continue;
                    };
                    if t >= 0.0 && t < limit {
                        limit = t;
                        best = Some(Hit {
                            primitive: prim,
                            t,
                            point: ray.at(t),
                            barycentric: (u, v),
                        });
                    }
                }
                continue;
            }
            // Nearer child last, so it comes off the stack first and its hit
            // tightens the cutoff before the far child is tested at all.
            let left = node.first;
            let right = node.first + 1;
            let dl = self.nodes[left as usize]
                .bounds
                .center()
                .distance_squared(ray.origin);
            let dr = self.nodes[right as usize]
                .bounds
                .center()
                .distance_squared(ray.origin);
            if dl < dr {
                stack.push(right);
                stack.push(left);
            } else {
                stack.push(left);
                stack.push(right);
            }
        }
        best
    }

    /// Candidate triangles for a box query.
    ///
    /// A **broad phase**: it returns everything in any leaf the query reaches,
    /// so a few triangles outside the box come back too. That is the intended
    /// trade — testing each primitive's box during traversal costs more than
    /// letting the caller's exact test reject a handful. What it never does is
    /// miss a triangle that really overlaps.
    pub fn overlaps(&self, query: &Aabb, out: &mut Vec<u32>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0u32];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if !node.bounds.intersects(query) {
                continue;
            }
            if node.is_leaf() {
                out.extend(
                    self.order[node.first as usize..(node.first + node.count) as usize]
                        .iter()
                        .copied(),
                );
            } else {
                stack.push(node.first);
                stack.push(node.first + 1);
            }
        }
    }

    /// Splits node `index`, whose primitives are `order[first .. first + count]`.
    fn split(
        &mut self,
        index: usize,
        first: usize,
        count: usize,
        bounds: &mut [Aabb],
        centroids: &mut [Vec3],
    ) {
        let mut node_bounds = Aabb::EMPTY;
        let mut centroid_bounds = Aabb::EMPTY;
        for i in first..first + count {
            node_bounds = node_bounds.union(bounds[i]);
            centroid_bounds = centroid_bounds.extend(centroids[i]);
        }
        self.nodes[index].bounds = node_bounds;

        if count <= MAX_LEAF {
            return;
        }
        let axis = centroid_bounds.max_extent_axis();
        let lo = centroid_bounds.min[axis];
        let hi = centroid_bounds.max[axis];
        if hi - lo < 1e-9 {
            // Every centroid is at the same place along the widest axis, so no
            // split can separate them. A median split here would loop forever;
            // a leaf is the honest answer.
            return;
        }
        let scale = BINS as f32 / (hi - lo);

        let mut bin_bounds = [Aabb::EMPTY; BINS];
        let mut bin_count = [0usize; BINS];
        for i in first..first + count {
            let b = bin_of(centroids[i][axis], lo, scale);
            bin_bounds[b] = bin_bounds[b].union(bounds[i]);
            bin_count[b] += 1;
        }

        // Sweep the bin boundaries once from each side, so each candidate's
        // left and right costs are available without an inner loop.
        let mut left_area = [0.0f32; BINS - 1];
        let mut left_count = [0usize; BINS - 1];
        let mut running = Aabb::EMPTY;
        let mut running_count = 0usize;
        for i in 0..BINS - 1 {
            running = running.union(bin_bounds[i]);
            running_count += bin_count[i];
            left_area[i] = running.surface_area();
            left_count[i] = running_count;
        }
        let mut right_area = [0.0f32; BINS - 1];
        let mut right_count = [0usize; BINS - 1];
        running = Aabb::EMPTY;
        running_count = 0;
        for i in (1..BINS).rev() {
            running = running.union(bin_bounds[i]);
            running_count += bin_count[i];
            right_area[i - 1] = running.surface_area();
            right_count[i - 1] = running_count;
        }

        let mut best_cost = f32::INFINITY;
        let mut best_split = 0usize;
        for i in 0..BINS - 1 {
            if left_count[i] == 0 || right_count[i] == 0 {
                continue;
            }
            let cost = left_area[i] * left_count[i] as f32 + right_area[i] * right_count[i] as f32;
            if cost < best_cost {
                best_cost = cost;
                best_split = i;
            }
        }
        // A leaf costs its own area times its count. If no split beats that,
        // splitting makes traversal slower, not faster.
        let leaf_cost = node_bounds.surface_area() * count as f32;
        if !best_cost.is_finite() || best_cost >= leaf_cost {
            return;
        }

        // Partition in place, permuting the parallel arrays in lockstep.
        let mut mid = first;
        for i in first..first + count {
            if bin_of(centroids[i][axis], lo, scale) <= best_split {
                self.order.swap(i, mid);
                bounds.swap(i, mid);
                centroids.swap(i, mid);
                mid += 1;
            }
        }
        let left_len = mid - first;
        if left_len == 0 || left_len == count {
            return;
        }

        let left = self.nodes.len() as u32;
        self.nodes.push(Node {
            bounds: Aabb::EMPTY,
            first: first as u32,
            count: left_len as u32,
        });
        self.nodes.push(Node {
            bounds: Aabb::EMPTY,
            first: mid as u32,
            count: (count - left_len) as u32,
        });
        self.nodes[index].first = left;
        self.nodes[index].count = 0;

        self.split(left as usize, first, left_len, bounds, centroids);
        self.split(left as usize + 1, mid, count - left_len, bounds, centroids);
    }
}

#[inline]
fn bin_of(value: f32, lo: f32, scale: f32) -> usize {
    (((value - lo) * scale) as usize).min(BINS - 1)
}

#[inline]
fn fetch(positions: &[Vec3], tri: [u32; 3]) -> Option<[Vec3; 3]> {
    Some([
        *positions.get(tri[0] as usize)?,
        *positions.get(tri[1] as usize)?,
        *positions.get(tri[2] as usize)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid of `n × n` quads in the XY plane, as triangles.
    fn grid(n: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        let idx = |x: usize, y: usize| (y * (n + 1) + x) as u32;
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                indices.push([idx(x, y), idx(x + 1, y), idx(x + 1, y + 1)]);
                indices.push([idx(x, y), idx(x + 1, y + 1), idx(x, y + 1)]);
            }
        }
        (positions, indices)
    }

    /// The answer a tree must agree with: test every triangle.
    fn brute_force(
        positions: &[Vec3],
        indices: &[[u32; 3]],
        ray: &Ray,
        t_max: f32,
    ) -> Option<(u32, f32)> {
        let mut best: Option<(u32, f32)> = None;
        for (i, tri) in indices.iter().enumerate() {
            let Some([a, b, c]) = fetch(positions, *tri) else {
                continue;
            };
            if let Some((t, _, _)) = ray.intersect_triangle(a, b, c) {
                if t >= 0.0 && t < t_max && best.is_none_or(|(_, bt)| t < bt) {
                    best = Some((i as u32, t));
                }
            }
        }
        best
    }

    #[test]
    fn an_empty_tree_answers_nothing() {
        let tree = Bvh::build(&[], &[]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(
            tree.raycast(&[], &[], &Ray::new(Vec3::ZERO, Vec3::Z), 1.0),
            None
        );
    }

    #[test]
    fn a_single_triangle_is_hit_and_missed_correctly() {
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let indices = vec![[0u32, 1, 2]];
        let tree = Bvh::build(&positions, &indices);

        let down = Ray::new(Vec3::new(0.2, 0.2, 1.0), -Vec3::Z);
        let hit = tree
            .raycast(&positions, &indices, &down, 10.0)
            .expect("straight through it");
        assert_eq!(hit.primitive, 0);
        assert!((hit.t - 1.0).abs() < 1e-5);
        assert!((hit.point - Vec3::new(0.2, 0.2, 0.0)).length() < 1e-5);

        let past = Ray::new(Vec3::new(5.0, 5.0, 1.0), -Vec3::Z);
        assert!(tree.raycast(&positions, &indices, &past, 10.0).is_none());
    }

    #[test]
    fn the_barycentric_coordinates_reconstruct_the_hit() {
        // Carried because a snap or a knife needs them next, and recovering them
        // afterwards means redoing the intersection.
        let positions = vec![Vec3::ZERO, Vec3::X * 3.0, Vec3::Y * 2.0];
        let indices = vec![[0u32, 1, 2]];
        let tree = Bvh::build(&positions, &indices);
        let ray = Ray::new(Vec3::new(0.9, 0.4, 1.0), -Vec3::Z);
        let hit = tree.raycast(&positions, &indices, &ray, 10.0).unwrap();

        let (u, v) = hit.barycentric;
        let rebuilt = positions[0] * (1.0 - u - v) + positions[1] * u + positions[2] * v;
        assert!(
            (rebuilt - hit.point).length() < 1e-4,
            "{rebuilt:?} should be {:?}",
            hit.point
        );
    }

    #[test]
    fn the_tree_agrees_with_brute_force_on_every_ray() {
        // The only test that really matters. An acceleration structure that is
        // fast and subtly wrong is worse than none: the errors are rare, ray
        // dependent, and show up as a snap that occasionally jumps.
        let (positions, indices) = grid(8);
        let tree = Bvh::build(&positions, &indices);
        assert_eq!(tree.len(), indices.len());

        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            (seed.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f32 / (1u64 << 53) as f32
        };

        for _ in 0..400 {
            let origin = Vec3::new(next() * 10.0 - 1.0, next() * 10.0 - 1.0, 2.0);
            let target = Vec3::new(next() * 10.0 - 1.0, next() * 10.0 - 1.0, -2.0);
            let ray = Ray::new(origin, target - origin);
            let theirs = brute_force(&positions, &indices, &ray, 100.0);
            let ours = tree.raycast(&positions, &indices, &ray, 100.0);
            match (theirs, ours) {
                (None, None) => {}
                (Some((_, t)), Some(hit)) => assert!(
                    (t - hit.t).abs() < 1e-4,
                    "distance disagrees: {t} vs {}",
                    hit.t
                ),
                (a, b) => panic!("disagreement: brute force {a:?}, tree {b:?}"),
            }
        }
    }

    #[test]
    fn a_refit_tracks_moved_positions() {
        // What a drag does. The tree keeps its shape and only its boxes change.
        let (mut positions, indices) = grid(4);
        let mut tree = Bvh::build(&positions, &indices);
        let nodes_before = tree.nodes.len();

        // Moved sideways, out of the boxes entirely. Moving it along the ray
        // instead would prove nothing: the boxes are a culling structure, so a
        // ray that still passes through the stale ones finds the moved triangles
        // anyway and a stale tree gives the right answer by luck.
        for p in positions.iter_mut() {
            p.x += 50.0;
        }
        let ray = Ray::new(Vec3::new(51.5, 1.5, 10.0), -Vec3::Z);
        assert!(
            tree.raycast(&positions, &indices, &ray, 100.0).is_none(),
            "the stale tree must cull the ray away, or this test proves nothing"
        );

        tree.refit(&positions, &indices);
        assert_eq!(tree.nodes.len(), nodes_before, "a refit does not rebuild");
        let hit = tree
            .raycast(&positions, &indices, &ray, 100.0)
            .expect("the surface moved with the mesh");
        assert!((hit.t - 10.0).abs() < 1e-4, "hit at t = {}", hit.t);
    }

    #[test]
    fn a_refit_matches_a_rebuild_on_answers() {
        let (mut positions, indices) = grid(6);
        let mut refitted = Bvh::build(&positions, &indices);
        for (i, p) in positions.iter_mut().enumerate() {
            p.z += (i as f32 * 0.37).sin() * 0.5;
        }
        refitted.refit(&positions, &indices);
        let rebuilt = Bvh::build(&positions, &indices);

        for x in 0..12 {
            for y in 0..12 {
                let ray = Ray::new(Vec3::new(x as f32 * 0.5, y as f32 * 0.5, 5.0), -Vec3::Z);
                let a = refitted.raycast(&positions, &indices, &ray, 100.0);
                let b = rebuilt.raycast(&positions, &indices, &ray, 100.0);
                match (a, b) {
                    (None, None) => {}
                    (Some(a), Some(b)) => assert!((a.t - b.t).abs() < 1e-4),
                    _ => panic!("a refit and a rebuild disagree at ({x}, {y})"),
                }
            }
        }
    }

    /// A tree big enough that its leaves are refitted on several threads:
    /// every box still holds what is under it, and rays agree with a rebuild.
    #[test]
    fn a_large_refit_shared_across_threads_is_whole() {
        let (mut positions, indices) = grid(120);
        let mut refitted = Bvh::build(&positions, &indices);
        assert!(
            refitted.nodes.len() >= 8192,
            "{} nodes: not the threaded path",
            refitted.nodes.len()
        );
        for (i, p) in positions.iter_mut().enumerate() {
            p.z += (i as f32 * 0.37).sin() * 0.5;
        }
        refitted.refit(&positions, &indices);
        for node in &refitted.nodes {
            if node.is_leaf() {
                for slot in node.first..node.first + node.count {
                    let tri = indices[refitted.order[slot as usize] as usize];
                    for v in tri {
                        let p = positions[v as usize];
                        assert!(p.cmpge(node.bounds.min).all() && p.cmple(node.bounds.max).all());
                    }
                }
            } else {
                let left = node.first as usize;
                for child in [left, left + 1] {
                    let c = refitted.nodes[child].bounds;
                    assert!(
                        c.min.cmpge(node.bounds.min).all() && c.max.cmple(node.bounds.max).all()
                    );
                }
            }
        }
        let rebuilt = Bvh::build(&positions, &indices);
        for x in 0..40 {
            for y in 0..40 {
                let ray = Ray::new(
                    Vec3::new(x as f32 * 1.5 + 0.3, y as f32 * 1.5 + 0.3, 5.0),
                    -Vec3::Z,
                );
                let a = refitted
                    .raycast(&positions, &indices, &ray, 100.0)
                    .map(|h| h.t);
                let b = rebuilt
                    .raycast(&positions, &indices, &ray, 100.0)
                    .map(|h| h.t);
                assert_eq!(a.is_some(), b.is_some(), "({x}, {y})");
                if let (Some(a), Some(b)) = (a, b) {
                    assert!((a - b).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn degenerate_input_builds_rather_than_refusing() {
        // A mesh mid-edit legitimately holds zero-area triangles and stale
        // indices. A spatial index that refuses to build is a tool that stops
        // working exactly when the user is in the middle of something.
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(f32::NAN, 0.0, 0.0)];
        let indices = vec![
            [0u32, 1, 2],
            [0, 0, 0],  // zero area
            [0, 1, 99], // out of range
            [0, 1, 3],  // NaN corner
        ];
        let tree = Bvh::build(&positions, &indices);
        assert_eq!(tree.len(), 2, "the good triangle and the degenerate one");

        let ray = Ray::new(Vec3::new(0.2, 0.2, 1.0), -Vec3::Z);
        let hit = tree.raycast(&positions, &indices, &ray, 10.0).unwrap();
        assert_eq!(hit.primitive, 0);
    }

    #[test]
    fn coincident_centroids_do_not_hang_the_build() {
        // Every centroid at the same point: no split can separate them, and a
        // median split would recurse forever on the same set.
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let indices = vec![[0u32, 1, 2]; 64];
        let tree = Bvh::build(&positions, &indices);
        assert_eq!(tree.len(), 64);
        let ray = Ray::new(Vec3::new(0.2, 0.2, 1.0), -Vec3::Z);
        assert!(tree.raycast(&positions, &indices, &ray, 10.0).is_some());
    }

    #[test]
    fn the_split_is_chosen_by_area_not_by_count() {
        // The point of the SAH. A cluster of small triangles beside one large
        // one must not be split down the middle by count: the tree that results
        // is balanced and slow.
        let mut positions = vec![
            Vec3::new(-100.0, -100.0, 0.0),
            Vec3::new(100.0, -100.0, 0.0),
            Vec3::new(0.0, 100.0, 0.0),
        ];
        let mut indices = vec![[0u32, 1, 2]];
        for i in 0..200 {
            let x = i as f32 * 0.01;
            let base = positions.len() as u32;
            positions.push(Vec3::new(x, 0.0, 1.0));
            positions.push(Vec3::new(x + 0.005, 0.0, 1.0));
            positions.push(Vec3::new(x, 0.005, 1.0));
            indices.push([base, base + 1, base + 2]);
        }
        let tree = Bvh::build(&positions, &indices);

        // A ray far from the cluster must not have to walk into it.
        let ray = Ray::new(Vec3::new(-50.0, -50.0, 5.0), -Vec3::Z);
        let hit = tree.raycast(&positions, &indices, &ray, 100.0).unwrap();
        assert_eq!(hit.primitive, 0, "the big triangle");
        // And the tree really did branch rather than collapsing to one leaf.
        assert!(tree.nodes.len() > 3, "{} nodes", tree.nodes.len());
    }

    #[test]
    fn overlapping_boxes_are_gathered() {
        let (positions, indices) = grid(4);
        let tree = Bvh::build(&positions, &indices);
        let mut found = Vec::new();

        tree.overlaps(
            &Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(5.0, 5.0, 1.0)),
            &mut found,
        );
        assert_eq!(found.len(), indices.len(), "everything is inside");

        let query = Aabb::new(Vec3::new(0.1, 0.1, -0.1), Vec3::new(0.9, 0.9, 0.1));
        tree.overlaps(&query, &mut found);
        assert!(
            found.len() < indices.len(),
            "a small query must cull something"
        );

        // The contract is no false negatives. False positives are expected and
        // are the caller's exact test to reject.
        for (i, tri) in indices.iter().enumerate() {
            let bb = Aabb::from_points(fetch(&positions, *tri).unwrap());
            if bb.intersects(&query) {
                assert!(
                    found.contains(&(i as u32)),
                    "triangle {i} overlaps the query and was not returned"
                );
            }
        }
    }

    #[test]
    fn the_tree_stays_inside_its_per_triangle_budget() {
        // A BVH that costs as much again as the geometry is one a user turns
        // off. Measured per triangle rather than against the mesh's total,
        // because the ratio to the mesh depends on how many vertices the
        // triangles happen to share and is not a property of the tree.
        //
        // 32 B/tri is the budget: a node is 32 bytes and a well-formed tree has
        // well under one node per triangle.
        const BUDGET_PER_TRI: usize = 32;
        for n in [8usize, 32, 64] {
            let (positions, indices) = grid(n);
            let tree = Bvh::build(&positions, &indices);
            let per_tri = tree.memory_bytes() as f64 / indices.len() as f64;
            assert!(
                per_tri <= BUDGET_PER_TRI as f64,
                "grid {n}: {per_tri:.1} B/tri over a budget of {BUDGET_PER_TRI}"
            );
        }
    }
}
