//! Smart UV project: cluster by angle, project each cluster, pack the results.
//!
//! The one unwrap here that is not a closed-form projection. It is still not a
//! solver — a real unwrap minimises distortion over a surface and belongs to
//! Phase 7 — but it is the tool that makes a whole object usable in one press,
//! which projection alone cannot: a projection puts every face on one plane,
//! and an object with faces pointing six ways needs six planes.
//!
//! Three steps, and each is separately testable:
//!
//! 1. **Cluster.** Grow regions of faces whose normals stay within an angle of
//!    the region's average. Adjacent faces only, so a cluster is a connected
//!    patch rather than every face that happens to point the same way — two
//!    parallel walls at opposite ends of a building are two islands, and
//!    packing them as one would put a seam through the middle of the texture.
//! 2. **Project.** Each cluster planar, along its own average normal.
//! 3. **Pack.** Fit the islands into the unit square, side by side.
//!
//! # Uniform texel density
//!
//! Every island is scaled by *one* factor, not fitted to its own box. That is
//! the difference between a usable atlas and a decorative one: scaling each
//! island to fill its slot gives a small face the same number of texels as a
//! large one, so the texture is sharp in some places and blurred in others for
//! no reason the artist can see or control.

use glam::{Vec2, Vec3};

use crate::arena::{ArenaIndex, FaceId, LoopId};
use crate::attribute::{AttrType, Domain};
use crate::bmesh::BMesh;

use super::uv_project::{self, Fit, Projection};

/// How near two faces' normals must be to share a cluster, by default.
///
/// Sixty-six degrees, which is what every tool of this kind uses. Loose enough
/// that a curved surface stays in one piece; tight enough that a box comes
/// apart at its edges, which is where an artist would have cut it.
pub const DEFAULT_ANGLE: f32 = 66.0_f32 * std::f32::consts::PI / 180.0;

/// The gap left between islands, as a fraction of the map.
///
/// Not decoration: a texture is filtered, and two islands touching bleed into
/// each other along the join. The default is about one texel at 512 across,
/// which is the smallest size anyone is likely to bake at.
pub const DEFAULT_MARGIN: f32 = 0.002;

/// How many rotations each island is tried at while packing.
///
/// A heuristic, and named as one. Turning an island to minimise its bounding
/// box is what lets a long diagonal strip lie flat instead of reserving a
/// square, and eight steps over a quarter turn is enough to find most of that
/// — a finer search buys very little because the boxes are then packed into
/// shelves anyway.
const ROTATION_STEPS: usize = 8;

/// Groups faces into connected patches whose normals stay within `angle`.
///
/// Compared against the *cluster's running average* rather than against the
/// seed: a cylinder's side should come off as one strip, and each face on it is
/// only a few degrees from its neighbour but ninety from the first one. Against
/// the average, the cluster follows the curve until it turns too far.
///
/// Seeded in ascending face order and returned in that order, so the result is
/// reproducible — the same rule the islands walk follows, and for the same
/// reason.
pub fn cluster_by_angle(mesh: &BMesh, faces: &[FaceId], angle: f32) -> Vec<Vec<FaceId>> {
    let limit = angle.cos();
    let slots = mesh.slot_count(Domain::Face);
    let mut wanted = vec![false; slots];
    for f in faces {
        if let Some(slot) = wanted.get_mut(f.index() as usize) {
            *slot = true;
        }
    }
    let mut taken = vec![false; slots];
    let mut out = Vec::new();

    for seed in faces {
        if taken.get(seed.index() as usize).copied().unwrap_or(true) {
            continue;
        }
        taken[seed.index() as usize] = true;
        let mut cluster = vec![*seed];
        let mut average = mesh.face_normal(*seed).normalize_or_zero();
        let mut frontier = vec![*seed];

        while let Some(f) = frontier.pop() {
            for l in mesh.face_loops(f) {
                let Some(e) = mesh.loop_edge(l) else {
                    continue;
                };
                for other in mesh.edge_loops(e) {
                    let Some(nf) = mesh.loop_face(other) else {
                        continue;
                    };
                    let slot = nf.index() as usize;
                    if !wanted.get(slot).copied().unwrap_or(false)
                        || taken.get(slot).copied().unwrap_or(true)
                    {
                        continue;
                    }
                    let n = mesh.face_normal(nf).normalize_or_zero();
                    if average.dot(n) < limit {
                        continue;
                    }
                    taken[slot] = true;
                    cluster.push(nf);
                    // Weighted by nothing: an average over faces, not over
                    // area. A large flat face and a small one contribute
                    // equally, which keeps a fan of small faces from being
                    // outvoted by the slab next to it.
                    average = (average + n).normalize_or_zero();
                    if average == Vec3::ZERO {
                        average = n;
                    }
                    frontier.push(nf);
                }
            }
        }
        cluster.sort_by_key(|f| f.index());
        out.push(cluster);
    }
    out
}

/// One island being packed.
struct Placed {
    corners: Vec<LoopId>,
    /// The rotation chosen, in radians.
    rotation: f32,
    /// Extent after rotation, before scaling.
    size: Vec2,
    /// Where the island's lower-left corner goes, before scaling.
    at: Vec2,
}

/// Projects each cluster and packs the results into the unit square.
///
/// Returns how many corners were written.
pub fn smart_project(
    mesh: &mut BMesh,
    faces: &[FaceId],
    layer: &str,
    angle: f32,
    margin: f32,
) -> usize {
    // No `is_empty` guard: with nothing selected there are no clusters, the
    // projection loop does nothing, and `pack` returns zero when the layer
    // does not exist. An early return here changed no behaviour, which a
    // falsification sweep proved by deleting it — so it is gone rather than
    // left as a branch no test can reach.
    let clusters = cluster_by_angle(mesh, faces, angle);

    // Each cluster onto its own plane, at world scale. `Fit::None` is the whole
    // point: fitting each cluster to the unit square here would throw away the
    // relative sizes that the single packing scale is about to preserve.
    for cluster in &clusters {
        let normal = cluster
            .iter()
            .map(|f| mesh.face_normal(*f).normalize_or_zero())
            .fold(Vec3::ZERO, |a, b| a + b)
            .normalize_or_zero();
        let normal = if normal == Vec3::ZERO {
            Vec3::Z
        } else {
            normal
        };
        let (right, up) = plane_basis(normal);
        uv_project::project_with(
            mesh,
            cluster,
            Projection::Planar { right, up },
            layer,
            Fit::None,
        );
    }

    let islands: Vec<Vec<LoopId>> = clusters
        .iter()
        .map(|c| {
            let mut corners: Vec<LoopId> = c.iter().flat_map(|f| mesh.face_loops(*f)).collect();
            corners.sort_by_key(|l| l.index());
            corners.dedup_by_key(|l| l.index());
            corners
        })
        .collect();

    pack(mesh, layer, &islands, margin)
}

/// Packs the given islands into the unit square, in place.
///
/// Useful on its own: an artist who has cut and moved islands by hand wants
/// them tidied without losing the cuts.
pub fn pack(mesh: &mut BMesh, layer: &str, islands: &[Vec<LoopId>], margin: f32) -> usize {
    let Some(uvs) = mesh.attributes().get_f32x2(layer) else {
        return 0;
    };
    let uvs = uvs.to_vec();
    let margin = margin.clamp(0.0, 0.2);

    // --- choose each island's rotation and measure it ----------------------
    let mut placed: Vec<Placed> = Vec::new();
    for corners in islands {
        if corners.is_empty() {
            continue;
        }
        let points: Vec<Vec2> = corners
            .iter()
            .filter_map(|l| uvs.get(l.index() as usize).copied())
            .collect();
        if points.is_empty() {
            continue;
        }
        let (rotation, size) = best_rotation(&points);
        placed.push(Placed {
            corners: corners.clone(),
            rotation,
            size,
            at: Vec2::ZERO,
        });
    }
    if placed.is_empty() {
        return 0;
    }

    // --- shelf-pack, tallest first -----------------------------------------
    //
    // Tallest first, because a shelf is as tall as its tallest member: a tall
    // island on a shelf of short ones wastes the difference across the whole
    // row. That is the standard result for shelf packing, and it is why the
    // sort is here.
    //
    // **It is not measured.** Reversing this order changes the fill ratio by
    // nothing at all on a cube, a slab, a sixteen- and a forty-eight-sided
    // cylinder, and a sphere — the layouts come out identical, because with
    // one to six islands the strip width decides everything and the order
    // decides nothing. The sort is kept because it is correct for the case it
    // is for — a character with forty pieces of varied height — and that case
    // is not in these tests. Anyone tempted to trust it should know it rests
    // on the literature and not on a test here.
    //
    // Deterministic either way, which *is* tested: an artist who repacks twice
    // must get the same layout, or a texture already painted against it is
    // invalidated.
    let order = {
        let mut idx: Vec<usize> = (0..placed.len()).collect();
        idx.sort_by(|a, b| {
            placed[*b]
                .size
                .y
                .partial_cmp(&placed[*a].size.y)
                .unwrap_or(std::cmp::Ordering::Equal)
                // By index on a tie, so equal islands do not swap between runs.
                .then(a.cmp(b))
        });
        idx
    };

    // A strip about as wide as the total area is deep. Squarish shelves waste
    // less than a very wide or very tall strip, and the whole thing is scaled
    // to fit afterwards, so the absolute width is arbitrary.
    let total_area: f32 = placed.iter().map(|p| p.size.x * p.size.y).sum();
    let widest = placed.iter().map(|p| p.size.x).fold(0.0, f32::max);
    let strip = total_area.sqrt().max(widest).max(f32::MIN_POSITIVE);

    // The gap goes *between* the islands, in layout units, so that scaling
    // the whole layout down carries it along. Inset only the outer square —
    // which is what this did at first — and adjacent islands touch exactly:
    // their shared corners coincide, the island walk finds them continuous
    // again, and six islands come back as five. Filtering would bleed across
    // the join too, which is the reason a margin exists at all.
    //
    // Expressed against the strip so the pre-scale gap survives the scaling
    // in roughly the proportion asked for; `used` ends up close to `strip`,
    // so the final gap is close to `margin`. Close enough is the honest claim
    // here — exact would need the layout and the scale to be solved together.
    let pad = margin * strip;

    let mut cursor = Vec2::ZERO;
    let mut shelf_height = 0.0_f32;
    let mut used = Vec2::ZERO;
    for i in &order {
        let size = placed[*i].size;
        if cursor.x > 0.0 && cursor.x + size.x > strip {
            cursor = Vec2::new(0.0, cursor.y + shelf_height);
            shelf_height = 0.0;
        }
        placed[*i].at = cursor;
        cursor.x += size.x + pad;
        shelf_height = shelf_height.max(size.y + pad);
        used.x = used.x.max(cursor.x);
        used.y = used.y.max(cursor.y + size.y + pad);
    }

    // --- one scale for everything ------------------------------------------
    //
    // `used` already carries one pad past each island, so the outer margin
    // falls out of centring the result: half the trailing pad on each side.
    // Against a floor rather than against zero. Every island being a point —
    // a cluster seen exactly edge-on — leaves `used` at the denormal scraps of
    // `f32::MIN_POSITIVE`, and one divided by a denormal overflows to infinity
    // rather than failing a `> 0.0` test. That reached the layer as NaN, which
    // reaches the GPU as a hole in the mesh.
    const FLOOR: f32 = 1e-9;
    let scale = if used.x <= FLOOR || used.y <= FLOOR {
        1.0
    } else {
        (1.0 / used.x).min(1.0 / used.y)
    };
    debug_assert!(scale.is_finite(), "packing scale is {scale}");

    // Centred, so a map that does not fill the square is not jammed into one
    // corner.
    let extent = used * scale;
    let origin = (Vec2::ONE - extent) * 0.5;

    // --- write --------------------------------------------------------------
    let Some(values) = mesh.attributes_mut().make_mut_f32x2(layer) else {
        return 0;
    };
    let mut count = 0;
    for p in &placed {
        let points: Vec<Vec2> = p
            .corners
            .iter()
            .filter_map(|l| uvs.get(l.index() as usize).copied())
            .collect();
        if points.is_empty() {
            continue;
        }
        let (sin, cos) = p.rotation.sin_cos();
        let turn = |v: Vec2| Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos);
        let mut min = Vec2::splat(f32::INFINITY);
        for v in &points {
            min = min.min(turn(*v));
        }
        for l in &p.corners {
            let slot = l.index() as usize;
            let Some(uv) = uvs.get(slot).copied() else {
                continue;
            };
            if slot >= values.len() {
                continue;
            }
            let local = turn(uv) - min;
            values[slot] = origin + (p.at + local) * scale;
            count += 1;
        }
    }
    count
}

/// The rotation, in `0..pi/2`, whose bounding box has the least area.
///
/// A quarter turn is enough: past it the box repeats, only transposed.
fn best_rotation(points: &[Vec2]) -> (f32, Vec2) {
    let mut best = (0.0_f32, extent_at(points, 0.0));
    let mut best_area = best.1.x * best.1.y;
    for step in 1..ROTATION_STEPS {
        let angle = step as f32 * std::f32::consts::FRAC_PI_2 / ROTATION_STEPS as f32;
        let size = extent_at(points, angle);
        let area = size.x * size.y;
        if area < best_area {
            best = (angle, size);
            best_area = area;
        }
    }
    best
}

fn extent_at(points: &[Vec2], angle: f32) -> Vec2 {
    let (sin, cos) = angle.sin_cos();
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for v in points {
        let t = Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos);
        min = min.min(t);
        max = max.max(t);
    }
    (max - min).max(Vec2::ZERO)
}

/// Two axes spanning the plane perpendicular to `normal`.
///
/// The reference is chosen away from the normal, because a cross product with
/// a nearly parallel vector loses all its precision — and a normal of exactly
/// `Z` is the common case, not the exception.
fn plane_basis(normal: Vec3) -> (Vec3, Vec3) {
    let reference = if normal.z.abs() < 0.9 {
        Vec3::Z
    } else {
        Vec3::X
    };
    let right = reference.cross(normal).normalize_or_zero();
    let right = if right == Vec3::ZERO { Vec3::X } else { right };
    let up = normal.cross(right);
    (right, up)
}

/// Ensures the layer exists, so a smart project can be the first thing that
/// ever touched this mesh's UVs.
pub fn ensure_layer(mesh: &mut BMesh, layer: &str) {
    let len = mesh.slot_count(Domain::Corner);
    mesh.attributes_mut()
        .ensure(layer, Domain::Corner, AttrType::F32x2, len);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    const LAYER: &str = uv_project::DEFAULT_LAYER;

    fn bmesh(mesh: &crate::MeshData) -> BMesh {
        BMesh::from_mesh_data(mesh)
    }

    fn all_faces(bm: &BMesh) -> Vec<FaceId> {
        bm.face_ids().collect()
    }

    fn uvs(bm: &BMesh) -> Vec<Vec2> {
        bm.attributes().get_f32x2(LAYER).unwrap().to_vec()
    }

    /// The axis-aligned box of one island's UVs.
    fn island_box(bm: &BMesh, faces: &[FaceId]) -> (Vec2, Vec2) {
        let uvs = uvs(bm);
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for f in faces {
            for l in bm.face_loops(*f) {
                let uv = uvs[l.index() as usize];
                min = min.min(uv);
                max = max.max(uv);
            }
        }
        (min, max)
    }

    /// The area of a face in world space.
    ///
    /// Newell's method rather than a triangle's cross product: an n-gon here
    /// is routinely a little non-planar, and the test compares against it.
    fn face_area(mesh: &BMesh, f: FaceId) -> f32 {
        let verts = mesh.face_verts(f);
        if verts.len() < 3 {
            return 0.0;
        }
        let mut normal = Vec3::ZERO;
        for i in 0..verts.len() {
            let a = mesh.position(verts[i]).unwrap_or(Vec3::ZERO);
            let b = mesh
                .position(verts[(i + 1) % verts.len()])
                .unwrap_or(Vec3::ZERO);
            normal += a.cross(b);
        }
        normal.length() * 0.5
    }

    /// Twice the signed area of a face in UV space.
    fn uv_area(bm: &BMesh, f: FaceId) -> f32 {
        let uvs = uvs(bm);
        let ring: Vec<Vec2> = bm
            .face_loops(f)
            .iter()
            .map(|l| uvs[l.index() as usize])
            .collect();
        let mut sum = 0.0;
        for i in 0..ring.len() {
            let a = ring[i];
            let b = ring[(i + 1) % ring.len()];
            sum += a.x * b.y - b.x * a.y;
        }
        (sum * 0.5).abs()
    }

    #[test]
    fn a_cube_clusters_into_six() {
        // Adjacent faces are ninety degrees apart, which is past any sensible
        // limit, so a box comes apart at its edges — where an artist would
        // have cut it.
        let mut bm = bmesh(&primitives::cube(2.0));
        let faces = all_faces(&bm);
        let clusters = cluster_by_angle(&bm, &faces, DEFAULT_ANGLE);
        assert_eq!(
            clusters.len(),
            6,
            "{:?}",
            clusters.iter().map(Vec::len).collect::<Vec<_>>()
        );
        let _ = &mut bm;
    }

    #[test]
    fn a_flat_grid_clusters_into_one() {
        let bm = bmesh(&primitives::grid(2.0, 4, 4));
        let faces = all_faces(&bm);
        let clusters = cluster_by_angle(&bm, &faces, DEFAULT_ANGLE);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), faces.len());
    }

    #[test]
    fn a_cylinder_side_stays_one_cluster_and_the_caps_come_off() {
        // The case that decides whether the average is tracked or the seed is.
        // Each side face is a few degrees from its neighbour and ninety from
        // the first one: against the seed the strip would break into pieces
        // the artist would then have to sew back together.
        let bm = bmesh(&primitives::cylinder(1.0, 2.0, 24));
        let faces = all_faces(&bm);
        let clusters = cluster_by_angle(&bm, &faces, DEFAULT_ANGLE);

        let sides: Vec<&Vec<FaceId>> = clusters.iter().filter(|c| c.len() > 2).collect();
        assert_eq!(
            sides.len(),
            1,
            "the side came off as {} pieces: {:?}",
            sides.len(),
            clusters.iter().map(Vec::len).collect::<Vec<_>>()
        );
        assert_eq!(sides[0].len(), 24, "the side is not whole");
        // And the two caps are their own clusters, because they face along the
        // axis rather than across it.
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn the_angle_limit_is_what_decides() {
        // Guards the test above: at a limit tight enough, even the cylinder's
        // side has to come apart. Without this, a clustering that ignored the
        // angle entirely and returned one cluster per connected component
        // would pass every other test here.
        let bm = bmesh(&primitives::cylinder(1.0, 2.0, 24));
        let faces = all_faces(&bm);
        let tight = cluster_by_angle(&bm, &faces, 5.0_f32.to_radians());
        assert!(
            tight.len() > 3,
            "a five-degree limit still gave {} clusters",
            tight.len()
        );
        // And at a limit loose enough, everything joins.
        let loose = cluster_by_angle(&bm, &faces, 179.0_f32.to_radians());
        assert_eq!(loose.len(), 1, "a 179-degree limit gave {}", loose.len());
    }

    #[test]
    fn every_island_lands_inside_the_unit_square() {
        let mut bm = bmesh(&primitives::cube(2.0));
        let faces = all_faces(&bm);
        let n = smart_project(&mut bm, &faces, LAYER, DEFAULT_ANGLE, DEFAULT_MARGIN);
        assert_eq!(n, 24);

        for uv in uvs(&bm) {
            assert!(
                uv.x >= -1e-4 && uv.x <= 1.0 + 1e-4 && uv.y >= -1e-4 && uv.y <= 1.0 + 1e-4,
                "a corner landed outside the map at {uv:?}"
            );
        }
    }

    #[test]
    fn islands_do_not_overlap() {
        // Checked as disjoint bounding boxes, which is what a shelf packer
        // guarantees and therefore what is honest to assert. Two overlapping
        // islands share texels, and painting one paints the other.
        let mut bm = bmesh(&primitives::cube(2.0));
        let faces = all_faces(&bm);
        let clusters = cluster_by_angle(&bm, &faces, DEFAULT_ANGLE);
        smart_project(&mut bm, &faces, LAYER, DEFAULT_ANGLE, DEFAULT_MARGIN);

        let boxes: Vec<(Vec2, Vec2)> = clusters.iter().map(|c| island_box(&bm, c)).collect();
        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                let (amin, amax) = boxes[i];
                let (bmin, bmax) = boxes[j];
                let apart = amax.x <= bmin.x + 1e-4
                    || bmax.x <= amin.x + 1e-4
                    || amax.y <= bmin.y + 1e-4
                    || bmax.y <= amin.y + 1e-4;
                assert!(
                    apart,
                    "islands {i} and {j} overlap: {amin:?}..{amax:?} vs {bmin:?}..{bmax:?}"
                );
            }
        }
    }

    #[test]
    fn texel_density_is_uniform_across_the_islands() {
        // The claim that makes the atlas usable rather than decorative. Two
        // faces of the same area in 3D must have the same area in UV space,
        // whichever island they ended up on — otherwise the texture is sharp
        // in places and blurred in others for no reason the artist can see.
        //
        // A cube's six faces are all the same size, so their UV areas must be
        // too. Fitting each island to its own slot would also pass on a cube's
        // *equal* islands, which is why the next test uses unequal ones.
        let mut bm = bmesh(&primitives::cube(2.0));
        let faces = all_faces(&bm);
        smart_project(&mut bm, &faces, LAYER, DEFAULT_ANGLE, DEFAULT_MARGIN);

        let areas: Vec<f32> = faces.iter().map(|f| uv_area(&bm, *f)).collect();
        let first = areas[0];
        assert!(first > 1e-5, "the faces collapsed: {areas:?}");
        for a in &areas {
            assert!(
                (a - first).abs() < first * 0.02,
                "a cube's faces got different texel densities: {areas:?}"
            );
        }
    }

    /// A slab: six faces, three different areas, six separate clusters.
    ///
    /// A cube cannot tell a uniform scale from a per-island fit, because its
    /// six islands are the same size either way — which is exactly how two
    /// falsification breaks survived the first version of these tests.
    fn slab() -> BMesh {
        let mut mesh = primitives::cube(2.0);
        {
            let positions = mesh.positions_mut();
            for p in positions.iter_mut() {
                p.y *= 0.25;
                p.z *= 2.5;
            }
        }
        bmesh(&mesh)
    }

    #[test]
    fn unequal_islands_keep_their_relative_sizes() {
        // The claim a per-island fit gets wrong, and the reason the whole
        // packer uses one scale: a face four times the area in 3D must be four
        // times the area on the map, whichever island it landed on. Fitting
        // each island to its own slot gives a small face as many texels as a
        // large one, so the texture is sharp in places and blurred in others
        // for no reason the artist can see.
        let mut bm = slab();
        let faces = all_faces(&bm);
        let clusters = cluster_by_angle(&bm, &faces, DEFAULT_ANGLE);
        assert_eq!(clusters.len(), 6, "the slab is not six islands");

        // The premise: the islands genuinely differ in size, so the assertion
        // below can fail.
        let world: Vec<f32> = faces.iter().map(|f| face_area(&bm, *f)).collect();
        let largest = world.iter().copied().fold(0.0, f32::max);
        let smallest = world.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            largest / smallest > 3.0,
            "the slab's faces are too alike to tell the two scalings apart: \
             {smallest} to {largest}"
        );

        smart_project(&mut bm, &faces, LAYER, DEFAULT_ANGLE, DEFAULT_MARGIN);

        let ratios: Vec<f32> = faces
            .iter()
            .map(|f| uv_area(&bm, *f) / face_area(&bm, *f))
            .collect();
        let first = ratios[0];
        assert!(first > 1e-9, "the faces collapsed: {ratios:?}");
        for r in &ratios {
            assert!(
                (r - first).abs() < first * 0.05,
                "texel density is not uniform across unequal islands: {ratios:?}"
            );
        }
    }

    #[test]
    fn the_atlas_is_not_mostly_wasted() {
        // A regression guard, and deliberately only that. It is *not* a
        // measure of packing quality: reversing the shelf order leaves these
        // numbers unchanged, so this cannot tell a good order from a bad one
        // — see the comment on the sort.
        //
        // What it catches is the packer becoming much worse than it is: a
        // scale computed from the wrong extent, a pad applied per island
        // instead of once, an island placed outside the strip.
        //
        // The floors are what the packer measures at today, less a margin.
        // Named per shape because they differ for a real reason: a cylinder's
        // side is one very wide island, and a shelf as wide as the widest
        // member leaves the rest of that shelf empty. That is a limitation of
        // shelf packing rather than a defect, and hiding it behind a single
        // loose bound would misrepresent it.
        for (name, mut bm, floor) in [
            ("cube", bmesh(&primitives::cube(2.0)), 0.60),
            ("slab", slab(), 0.50),
            // One wide island and two small ones: a third of the map, and
            // that is the shape of the answer, not a bug.
            ("cylinder", bmesh(&primitives::cylinder(1.0, 2.0, 16)), 0.30),
            ("sphere", bmesh(&primitives::uv_sphere(1.0, 16, 8)), 0.90),
        ] {
            let faces = all_faces(&bm);
            let clusters = cluster_by_angle(&bm, &faces, DEFAULT_ANGLE);
            smart_project(&mut bm, &faces, LAYER, DEFAULT_ANGLE, 0.0);
            let used: f32 = clusters
                .iter()
                .map(|c| {
                    let (min, max) = island_box(&bm, c);
                    let size = max - min;
                    size.x * size.y
                })
                .sum();
            assert!(
                used > floor,
                "{name}: the islands fill {:.0}% of the map, against a floor of \
                 {:.0}%",
                used * 100.0,
                floor * 100.0
            );
        }
    }

    #[test]
    fn a_large_face_gets_more_of_the_map_than_a_small_one() {
        // The other half of uniform density, and the half a per-island fit
        // gets wrong: the *ratio* between islands has to survive packing. Two
        // planes, one four times the area of the other, in one selection.
        // A 2x2 grid with one quadrant stretched, so its four faces have
        // genuinely different areas.
        let mut mesh = primitives::grid(2.0, 2, 2);
        {
            let positions = mesh.positions_mut();
            for p in positions.iter_mut() {
                if p.x > 0.0 {
                    p.x *= 3.0;
                }
                if p.y > 0.0 {
                    p.y *= 3.0;
                }
            }
        }
        let mut bm = bmesh(&mesh);
        let faces = all_faces(&bm);
        smart_project(&mut bm, &faces, LAYER, DEFAULT_ANGLE, DEFAULT_MARGIN);

        // Every face is on the same island here, so this is really a check
        // that the projection kept relative sizes — but that is exactly the
        // property `Fit::None` inside `smart_project` is there for, and
        // switching it to `Fit::Bounds` breaks it.
        let mut ratios: Vec<f32> = Vec::new();
        for f in &faces {
            let world = face_area(&bm, *f);
            let uv = uv_area(&bm, *f);
            assert!(world > 1e-6 && uv > 1e-9, "{world} {uv}");
            ratios.push(uv / world);
        }
        let first = ratios[0];
        for r in &ratios {
            assert!(
                (r - first).abs() < first * 0.02,
                "UV area is not proportional to world area: {ratios:?}"
            );
        }
    }

    #[test]
    fn the_margin_separates_the_islands_and_not_only_the_edge() {
        // The gap between islands is what a margin is *for*: a texture is
        // filtered, and two islands touching bleed into each other along the
        // join. The first version of this test only looked at the outer edge,
        // and the first version of the packer only inset the outer square — so
        // adjacent islands touched exactly, their shared corners coincided,
        // and the island walk merged them back together.
        let mut bm = bmesh(&primitives::cube(2.0));
        let faces = all_faces(&bm);
        let margin = 0.02;
        let clusters = cluster_by_angle(&bm, &faces, DEFAULT_ANGLE);
        smart_project(&mut bm, &faces, LAYER, DEFAULT_ANGLE, margin);

        for uv in uvs(&bm) {
            assert!(
                uv.x > margin * 0.3 && uv.x < 1.0 - margin * 0.3,
                "a corner sat on the edge at {uv:?}"
            );
        }

        // And between: the closest pair of island boxes is at least most of a
        // margin apart on the axis that separates them.
        let boxes: Vec<(Vec2, Vec2)> = clusters.iter().map(|c| island_box(&bm, c)).collect();
        let mut closest = f32::INFINITY;
        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                let (amin, amax) = boxes[i];
                let (bmin, bmax) = boxes[j];
                // Distance between the boxes, zero if they touch or overlap.
                let dx = (bmin.x - amax.x).max(amin.x - bmax.x);
                let dy = (bmin.y - amax.y).max(amin.y - bmax.y);
                closest = closest.min(dx.max(dy));
            }
        }
        assert!(
            closest > margin * 0.3,
            "the closest islands are {closest} apart, against a margin of {margin}"
        );

        // Which is what keeps them separate islands at all.
        assert_eq!(
            super::super::uv_edit::islands(&bm, LAYER).len(),
            6,
            "packing merged islands back together"
        );
    }

    #[test]
    fn packing_twice_gives_the_same_layout() {
        // An artist who repacks must get the same answer. A shelf order that
        // depended on iteration accident would move the whole atlas every
        // time, invalidating any texture already painted against it.
        let run = || {
            let mut bm = bmesh(&primitives::cylinder(1.0, 2.0, 12));
            let faces = all_faces(&bm);
            smart_project(&mut bm, &faces, LAYER, DEFAULT_ANGLE, DEFAULT_MARGIN);
            uvs(&bm)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn rotating_an_island_is_tried_and_helps() {
        // A long diagonal strip reserves a square box unrotated. Checked as a
        // property of `best_rotation` rather than through the packer, so the
        // claim is about the mechanism and not about a layout.
        let diagonal: Vec<Vec2> = (0..10)
            .map(|i| Vec2::new(i as f32 * 0.1, i as f32 * 0.1))
            .collect();
        let (angle, size) = best_rotation(&diagonal);
        assert!(angle > 0.0, "no rotation was chosen for a diagonal strip");
        let unrotated = extent_at(&diagonal, 0.0);
        assert!(
            size.x * size.y < unrotated.x * unrotated.y,
            "the chosen rotation is not smaller: {size:?} vs {unrotated:?}"
        );
    }

    #[test]
    fn an_empty_selection_writes_nothing() {
        let mut bm = bmesh(&primitives::cube(2.0));
        assert_eq!(
            smart_project(&mut bm, &[], LAYER, DEFAULT_ANGLE, DEFAULT_MARGIN),
            0
        );
        assert!(bm.attributes().get(LAYER).is_none());
    }

    #[test]
    fn a_degenerate_island_does_not_produce_a_nan() {
        // A cluster seen exactly edge-on projects to a line, and one of zero
        // extent to a point. Dividing by that extent would put NaN in the
        // layer, which reaches the GPU as a hole in the mesh.
        let mut bm = bmesh(&primitives::grid(2.0, 2, 2));
        let faces = all_faces(&bm);
        ensure_layer(&mut bm, LAYER);
        if let Some(v) = bm.attributes_mut().make_mut_f32x2(LAYER) {
            for uv in v.iter_mut() {
                *uv = Vec2::splat(0.25);
            }
        }
        let islands: Vec<Vec<LoopId>> =
            vec![faces.iter().flat_map(|f| bm.face_loops(*f)).collect()];
        pack(&mut bm, LAYER, &islands, DEFAULT_MARGIN);
        for uv in uvs(&bm) {
            assert!(uv.is_finite(), "{uv:?}");
        }
    }

    #[test]
    fn packing_an_existing_layout_keeps_the_cuts() {
        // `pack` on its own is for an artist who has cut and arranged islands
        // by hand and wants them tidied. It must not merge them: the corners
        // of each island stay together.
        let mut bm = bmesh(&primitives::cube(2.0));
        let faces = all_faces(&bm);
        uv_project::project(&mut bm, &faces, Projection::Box, LAYER);
        let clusters = cluster_by_angle(&bm, &faces, DEFAULT_ANGLE);
        let islands: Vec<Vec<LoopId>> = clusters
            .iter()
            .map(|c| c.iter().flat_map(|f| bm.face_loops(*f)).collect())
            .collect();

        assert_eq!(pack(&mut bm, LAYER, &islands, DEFAULT_MARGIN), 24);
        let boxes: Vec<(Vec2, Vec2)> = clusters.iter().map(|c| island_box(&bm, c)).collect();
        // Six islands, six boxes, none of them the whole map.
        for (min, max) in &boxes {
            assert!(
                (max.x - min.x) < 0.9 && (max.y - min.y) < 0.9,
                "an island spans the map: {min:?}..{max:?}"
            );
        }
    }
}
