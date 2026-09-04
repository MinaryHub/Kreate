//! Clusters: the unit the static path culls and draws.
//!
//! A ten-million-triangle scene cannot be culled per object — the objects are
//! too big — and it must not be culled per triangle. The unit in between is a
//! cluster of at most 128 triangles with a bounding sphere and a **normal
//! cone**, and it is what makes the whole large-scene design work: one compute
//! dispatch tests every cluster in the scene and appends the survivors to a
//! list, and one `draw_indirect` draws them.
//!
//! # Why a single indirect draw and not multi-draw
//!
//! `MULTI_DRAW_INDIRECT_COUNT` does not exist on Metal. An architecture built
//! on it makes macOS a rewrite, and by the time that is discovered a hundred
//! call sites assume it. So every cluster lives in one index buffer and the
//! instance *is* the cluster: the vertex shader reads `visible[instance_index]`,
//! looks up the cluster record, and finds its own indices. `draw_indirect` is
//! WebGPU core, so this runs on Vulkan, Metal, DX12 — and in a browser.
//!
//! # What the normal cone buys
//!
//! Exact, safe backface culling a cluster at a time. Not an approximation and
//! not a change of topology, so it is completely safe in an editor: a cluster is
//! rejected only when *every* triangle in it faces away. On a closed mesh that
//! is typically 40–50% of them, for one dot product.
//!
//! # What this is not
//!
//! Not a level-of-detail DAG. [`Cluster::lod`] exists and is always zero;
//! simplification is Phase 7 and only for objects the user has locked, because
//! a topology edit invalidates the whole build. The bake here is the input that
//! work would need, which is why deferring it costs nothing.

use crate::mesh_gpu::RenderMeshCpu;
use bytemuck::{Pod, Zeroable};
use glam::{DAffine3, DVec3, Mat4, Vec3, Vec4, Vec4Swizzles};

/// Triangles per cluster.
///
/// The draw issues this many triangles for *every* surviving cluster, so it is
/// also the fixed vertex count of the indirect draw. 128 is the figure the
/// plan settles on and what meshoptimizer is tuned around.
pub const MAX_TRIANGLES: usize = 128;

/// Distinct vertices a cluster may touch.
///
/// Generous, and deliberately so. The usual figure is 64, which is what a mesh
/// shader's workgroup can hold — but there is no mesh shader here, the indices
/// are flattened back to the mesh's own numbering, and nothing downstream cares
/// how many distinct vertices a cluster touches.
///
/// What it does control is which cap the clusterer hits first. At 64 vertices
/// the *vertex* cap binds long before the triangle cap and clusters come out
/// averaging 95 of their 128 triangles — while the draw issues a fixed 384
/// vertices regardless, so a quarter of every cluster's draw was surplus.
/// Measured: raising this from 64 to 192 cut the cluster count from 108,736 to
/// 79,616 on the same geometry and took the cluster path from 8x slower than a
/// direct draw to 2.4x slower.
///
/// Clusters do get spatially larger, which loosens their bounding spheres and
/// cones a little. That is the trade, and at a fixed draw size it is clearly
/// worth it.
pub const MAX_VERTICES: usize = 192;

/// How much the clusterer should favour tight normal cones over tight spheres.
///
/// Zero optimises purely for spatial locality, one purely for direction. The
/// cone is worth real weight here because backface culling is half the win and
/// costs one dot product, but pushing it too far inflates the spheres and
/// weakens frustum and occlusion culling, which are the other half.
const CONE_WEIGHT: f32 = 0.5;

/// One cluster, in the layout the cull shader reads.
///
/// `repr(C)` and 48 bytes: four-vector aligned throughout, so the same bytes
/// serve as a WGSL `array<Cluster>` with no padding surprises. A ten-million
/// triangle scene is about 78,000 clusters, so under 4 MB.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Cluster {
    /// Object-space bounding sphere: centre in `xyz`, radius in `w`.
    ///
    /// Object space, not world: geometry is shared between linked duplicates and
    /// only the transform differs, so a world-space sphere would have to be
    /// rebaked per instance. The cull transforms it instead.
    pub sphere: [f32; 4],
    /// Normal cone: axis in `xyz`, cosine cutoff in `w`.
    pub cone: [f32; 4],
    /// Where this cluster's indices start in [`ClusterMesh::indices`].
    pub index_offset: u32,
    /// How many triangles it actually has, at most [`MAX_TRIANGLES`].
    ///
    /// The draw always issues `MAX_TRIANGLES`, and the vertex shader collapses
    /// the surplus to a degenerate triangle — zero area, rejected by the
    /// rasteriser at no cost. Cheaper than the alternative, which is a draw per
    /// cluster.
    pub triangle_count: u32,
    /// Level of detail. Always zero; see the module docs.
    pub lod: u32,
    pub _pad: u32,
}

/// A mesh reorganised into clusters.
#[derive(Clone, Debug, Default)]
pub struct ClusterMesh {
    pub clusters: Vec<Cluster>,
    /// Triangle indices grouped by cluster, into the mesh's own vertex buffer.
    ///
    /// Flattened out of meshoptimizer's two-level form on purpose. The
    /// two-level layout exists to feed a mesh shader; this design draws from a
    /// flat index buffer, so carrying the indirection would cost a lookup per
    /// vertex and buy nothing.
    pub indices: Vec<u32>,
}

impl ClusterMesh {
    /// Splits a mesh into clusters and measures each one.
    pub fn bake(cpu: &RenderMeshCpu) -> Self {
        if cpu.indices.is_empty() || cpu.positions.is_empty() {
            return Self {
                clusters: Vec::new(),
                indices: Vec::new(),
            };
        }

        let positions = bytemuck::cast_slice::<_, u8>(&cpu.positions);
        let stride = std::mem::size_of_val(&cpu.positions[0]);
        let Ok(adapter) = meshopt::VertexDataAdapter::new(positions, stride, 0) else {
            // Only reachable if the position buffer stops being a tight array of
            // three floats. Falling back to one cluster per mesh keeps the scene
            // drawable rather than making a layout change a blank viewport.
            return Self::single_cluster(cpu);
        };

        let meshlets = meshopt::build_meshlets(
            &cpu.indices,
            &adapter,
            MAX_VERTICES,
            MAX_TRIANGLES,
            CONE_WEIGHT,
        );

        let mut clusters = Vec::with_capacity(meshlets.len());
        let mut indices = Vec::with_capacity(cpu.indices.len());
        for meshlet in meshlets.iter() {
            let bounds = meshopt::compute_meshlet_bounds(meshlet, &adapter);
            let triangle_count = meshlet.triangles.len() / 3;
            let index_offset = indices.len() as u32;
            // Local indices back to the mesh's own numbering.
            indices.extend(
                meshlet
                    .triangles
                    .iter()
                    .map(|local| meshlet.vertices[*local as usize]),
            );
            clusters.push(Cluster {
                sphere: [
                    bounds.center[0],
                    bounds.center[1],
                    bounds.center[2],
                    bounds.radius,
                ],
                cone: [
                    bounds.cone_axis[0],
                    bounds.cone_axis[1],
                    bounds.cone_axis[2],
                    bounds.cone_cutoff,
                ],
                index_offset,
                triangle_count: triangle_count as u32,
                lod: 0,
                _pad: 0,
            });
        }

        Self { clusters, indices }
    }

    /// The whole mesh as one cluster, for the fallback path.
    ///
    /// A cone cutoff of 1.0 disables backface rejection: the cluster faces every
    /// direction, so it is never entirely backfacing. Getting that wrong here
    /// would cull whole meshes.
    fn single_cluster(cpu: &RenderMeshCpu) -> Self {
        let centre = cpu.bounds.center();
        Self {
            clusters: vec![Cluster {
                sphere: [centre.x, centre.y, centre.z, cpu.bounds.bounding_radius()],
                cone: [0.0, 0.0, 1.0, 1.0],
                index_offset: 0,
                triangle_count: (cpu.indices.len() / 3) as u32,
                lod: 0,
                _pad: 0,
            }],
            indices: cpu.indices.clone(),
        }
    }

    /// Bytes the cluster records and index stream occupy.
    pub fn memory_bytes(&self) -> usize {
        self.clusters.len() * std::mem::size_of::<Cluster>() + self.indices.len() * 4
    }
}

/// A world-space bounding sphere, ready to test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSphere {
    pub centre: Vec3,
    pub radius: f32,
}

impl WorldSphere {
    /// Transforms an object-space sphere into camera-relative world space.
    ///
    /// Camera-relative because everything the renderer hands the GPU is: an
    /// absolute world position ten kilometres from the origin has visibly less
    /// float precision than the geometry deserves, and the cull has to agree
    /// with the draw about where things are or it culls the wrong ones.
    ///
    /// The radius scales by the longest column of the linear part, which for a
    /// TRS transform is exactly the largest scale factor — rotation keeps the
    /// columns orthogonal, so the singular values *are* the column lengths.
    pub fn of(sphere: [f32; 4], transform: &DAffine3, eye: DVec3) -> Self {
        let centre = transform.transform_point3(DVec3::new(
            sphere[0] as f64,
            sphere[1] as f64,
            sphere[2] as f64,
        )) - eye;
        let m = transform.matrix3;
        let scale = m
            .x_axis
            .length()
            .max(m.y_axis.length())
            .max(m.z_axis.length());
        Self {
            centre: centre.as_vec3(),
            radius: sphere[3] * scale as f32,
        }
    }
}

/// A plane every point is inside, for padding an unused frustum slot.
///
/// A zero normal with a large positive distance: `dot(0, p) + w` is `w`, which
/// clears any radius. The obvious alternative — a zeroed slot — rejects
/// everything.
const ACCEPT_EVERYTHING: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1e30);

/// The frustum's planes, as `xyz = normal`, `w = distance`.
///
/// Inward-facing: a point is inside when `dot(normal, p) + w >= 0`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Frustum {
    planes: [Vec4; Frustum::MAX_PLANES],
    count: usize,
}

impl Frustum {
    /// Six candidates, of which an infinite far plane leaves five usable.
    pub const MAX_PLANES: usize = 6;

    /// Extracts the planes from a view-projection matrix.
    ///
    /// Gribb–Hartmann: each clip-space inequality is a row combination of the
    /// matrix. Degenerate candidates are dropped rather than normalised, which
    /// is what makes this work with the renderer's reversed-Z infinite far
    /// plane — that plane genuinely does not exist, and a zero-length normal is
    /// how the matrix says so.
    pub fn from_view_projection(view_proj: &Mat4) -> Self {
        let r0 = view_proj.row(0);
        let r1 = view_proj.row(1);
        let r2 = view_proj.row(2);
        let r3 = view_proj.row(3);
        // WebGPU clips to 0 <= z <= w, unlike OpenGL's -w <= z <= w, so the two
        // depth planes are `r2` and `r3 - r2` rather than a symmetric pair.
        // Reversed-Z then maps the near plane to z = w and infinity to z = 0,
        // which is why `r2` is the degenerate one here and not `r3 - r2`.
        let candidates = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2, r2];

        let mut planes = [Vec4::ZERO; Self::MAX_PLANES];
        let mut count = 0;
        for plane in candidates {
            let length = plane.xyz().length();
            // A plane whose normal has no direction constrains nothing. Keeping
            // it would divide by zero and reject everything.
            if length > 1e-6 && length.is_finite() {
                planes[count] = plane / length;
                count += 1;
            }
        }
        Self { planes, count }
    }

    pub fn plane_count(&self) -> usize {
        self.count
    }

    /// The planes as a fixed array of six, the shape the cull shader reads.
    ///
    /// Unused slots are filled with a plane that accepts everything rather than
    /// left zeroed. A zero plane has `dot(0, p) + 0 >= -radius` false for every
    /// point, so leaving the slot empty would reject the entire scene — the
    /// loop in the shader has a fixed trip count and cannot know which slots
    /// are real.
    pub fn to_padded_planes(&self) -> [Vec4; Self::MAX_PLANES] {
        let mut out = [ACCEPT_EVERYTHING; Self::MAX_PLANES];
        out[..self.count].copy_from_slice(&self.planes[..self.count]);
        out
    }

    /// Whether any part of a sphere is inside.
    ///
    /// Conservative: a sphere straddling a corner can pass all six plane tests
    /// while lying outside the frustum. That direction of error costs a wasted
    /// cluster; the other direction punches a hole in the model.
    pub fn intersects(&self, sphere: WorldSphere) -> bool {
        self.planes[..self.count]
            .iter()
            .all(|p| p.xyz().dot(sphere.centre) + p.w >= -sphere.radius)
    }
}

/// Whether every triangle in a cluster faces away from the eye.
///
/// The formula is meshoptimizer's, in the form that uses the bounding sphere
/// rather than the cone apex:
///
/// ```text
/// dot(centre - eye, axis) >= cutoff * length(centre - eye) + radius
/// ```
///
/// Chosen over the apex form for two reasons. The sphere is already here for
/// frustum culling, so it needs no extra data; and this form has no singularity
/// when the eye sits at the centre, which is a real case — the camera can be
/// inside a mesh.
///
/// `centre` is camera-relative, so `eye` is the origin and the vector is the
/// centre itself.
pub fn cone_is_backfacing(cluster: &Cluster, sphere: WorldSphere, transform: &DAffine3) -> bool {
    // The cone axis is a direction, so it rotates but must not translate, and
    // under non-uniform scale it does not transform like a normal at all —
    // hence the inverse transpose. Skipped when the scale is uniform, which is
    // almost always, because the inverse of a 3x3 is not free.
    let m = transform.matrix3;
    let axis = Vec3::new(cluster.cone[0], cluster.cone[1], cluster.cone[2]);
    let axis = (m.inverse().transpose() * axis.as_dvec3())
        .as_vec3()
        .normalize_or_zero();
    if axis == Vec3::ZERO {
        // A degenerate cone claims no direction, so it can never be proven
        // backfacing. Resolving towards drawing: the cost is a cluster that
        // should have been rejected, and the alternative is a hole.
        return false;
    }
    let cutoff = cluster.cone[3];
    sphere.centre.dot(axis) >= cutoff * sphere.centre.length() + sphere.radius
}

/// A depth pyramid as plain data, for the CPU reference.
pub struct PyramidData<'a> {
    /// One entry per level, level 0 first.
    pub levels: &'a [Vec<f32>],
    pub size: (u32, u32),
}

/// Whether a cluster is entirely behind what the pyramid already holds.
///
/// The specification the compute shader implements. Mirrors it step for step,
/// including the deliberate over-estimates: the box enclosing the sphere rather
/// than the sphere, the level rounded up, the nearest depth taken from the box's
/// corners. Every one of those errs towards keeping the cluster, and that
/// asymmetry is the point — a wrongly kept cluster costs a fraction of a
/// millisecond, a wrongly rejected one is a hole in the model.
pub fn is_occluded(sphere: WorldSphere, view_proj: &Mat4, pyramid: &PyramidData<'_>) -> bool {
    let mut lo = glam::Vec2::splat(1e30);
    let mut hi = glam::Vec2::splat(-1e30);
    let mut nearest = -1e30f32;
    for i in 0..8u32 {
        let offset = Vec3::new(
            if i & 1 != 0 {
                sphere.radius
            } else {
                -sphere.radius
            },
            if i & 2 != 0 {
                sphere.radius
            } else {
                -sphere.radius
            },
            if i & 4 != 0 {
                sphere.radius
            } else {
                -sphere.radius
            },
        );
        let clip = *view_proj * (sphere.centre + offset).extend(1.0);
        // The box crosses the plane through the eye, so its projection folds and
        // a screen extent is meaningless.
        if clip.w <= 1e-6 {
            return false;
        }
        let ndc = clip.truncate() / clip.w;
        lo = lo.min(ndc.truncate());
        hi = hi.max(ndc.truncate());
        // Reversed-Z: larger is nearer.
        nearest = nearest.max(ndc.z);
    }

    let size = glam::Vec2::new(pyramid.size.0 as f32, pyramid.size.1 as f32);
    // NDC to texture coordinates. Y flips: NDC has +Y up, a texture +V down.
    let uv_lo = (glam::Vec2::new(lo.x, -hi.y) * 0.5 + 0.5).clamp(glam::Vec2::ZERO, glam::Vec2::ONE);
    let uv_hi = (glam::Vec2::new(hi.x, -lo.y) * 0.5 + 0.5).clamp(glam::Vec2::ZERO, glam::Vec2::ONE);

    let span = (uv_hi - uv_lo) * size;
    let levels = pyramid.levels.len() as f32;
    let level = span
        .x
        .max(span.y)
        .max(1.0)
        .log2()
        .ceil()
        .clamp(0.0, levels - 1.0);
    let li = level as usize;
    let level_size = (size / 2f32.powf(level)).floor().max(glam::Vec2::ONE);
    let last = level_size - glam::Vec2::ONE;
    let t_lo = (uv_lo * level_size).floor().clamp(glam::Vec2::ZERO, last);
    let t_hi = (uv_hi * level_size).floor().clamp(glam::Vec2::ZERO, last);

    let width = level_size.x as usize;
    let at = |x: f32, y: f32| pyramid.levels[li][y as usize * width + x as usize];
    let farthest = at(t_lo.x, t_lo.y)
        .min(at(t_hi.x, t_lo.y))
        .min(at(t_lo.x, t_hi.y))
        .min(at(t_hi.x, t_hi.y));

    // Strictly greater, so a cluster exactly coplanar with what is in front of
    // it survives. Ties happen constantly and rejecting them makes surfaces
    // flicker.
    farthest > nearest
}

/// Whether a cluster survives every cull, decided on the CPU.
///
/// This is the *specification* the compute shader implements, and it exists so
/// the shader can be tested against something rather than against a screenshot.
/// A cull that is wrong on the GPU shows up as geometry that vanishes at certain
/// angles, which is close to undebuggable by eye; a set difference against this
/// function names the exact cluster and the exact camera.
///
/// See `crates/kreate-render/tests/cluster_cull.rs`, which asserts the two agree
/// cluster for cluster over a range of viewpoints.
pub fn survives(
    cluster: &Cluster,
    transform: &DAffine3,
    camera: &crate::camera::Camera,
    frustum: &Frustum,
    viewport_h_px: u32,
    cull_px: f32,
) -> bool {
    let sphere = WorldSphere::of(cluster.sphere, transform, camera.eye());
    if !frustum.intersects(sphere) {
        return false;
    }
    if cone_is_backfacing(cluster, sphere, transform) {
        return false;
    }
    match crate::lod::sphere_diameter_px(
        sphere.centre.as_dvec3(),
        sphere.radius as f64,
        camera,
        viewport_h_px,
    ) {
        // Meaningless to measure, so it is drawn — the same resolution the
        // shader reaches by returning the full viewport height.
        None => true,
        Some(diameter) => diameter >= cull_px as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::mesh_gpu::RenderMeshCpu;

    fn sphere_mesh() -> RenderMeshCpu {
        RenderMeshCpu::extract(&kreate_mesh::primitives::uv_sphere(1.0, 64, 32))
    }

    #[test]
    fn every_triangle_survives_the_bake_exactly_once() {
        // The invariant the whole feature rests on. A clusterer that drops or
        // duplicates a triangle produces a model with holes or with z-fighting,
        // and either is the kind of bug that gets blamed on the shader for a
        // week.
        //
        // Compared as multisets of sorted corner triples: clustering is free to
        // reorder the triangles and to rotate a triangle's corners, but not to
        // change which vertices meet.
        let cpu = sphere_mesh();
        let baked = ClusterMesh::bake(&cpu);

        let mut before: Vec<[u32; 3]> = cpu
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let mut t = *t;
                t.sort_unstable();
                t
            })
            .collect();
        let mut after: Vec<[u32; 3]> = baked
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let mut t = *t;
                t.sort_unstable();
                t
            })
            .collect();
        before.sort_unstable();
        after.sort_unstable();
        assert_eq!(before, after);
    }

    #[test]
    fn the_index_ranges_tile_the_index_buffer_without_gaps_or_overlap() {
        // Each cluster's range has to start where the last one ended, because
        // the vertex shader finds its indices by offset. An off-by-one here
        // draws a neighbouring cluster's geometry, which looks like corrupt
        // topology rather than like a bad offset.
        let baked = ClusterMesh::bake(&sphere_mesh());
        let mut expected = 0u32;
        for cluster in &baked.clusters {
            assert_eq!(cluster.index_offset, expected);
            expected += cluster.triangle_count * 3;
        }
        assert_eq!(expected as usize, baked.indices.len());
    }

    #[test]
    fn no_cluster_exceeds_the_triangle_cap() {
        // The draw issues exactly MAX_TRIANGLES per cluster, so a cluster with
        // more would have its surplus silently dropped.
        let baked = ClusterMesh::bake(&sphere_mesh());
        assert!(!baked.clusters.is_empty());
        for cluster in &baked.clusters {
            assert!(
                cluster.triangle_count as usize <= MAX_TRIANGLES,
                "a cluster held {} triangles",
                cluster.triangle_count
            );
            assert!(cluster.triangle_count > 0);
        }
    }

    #[test]
    fn every_bounding_sphere_contains_its_own_triangles() {
        // Culling reads these spheres and nothing else, so a sphere that does
        // not contain its geometry culls visible triangles.
        let cpu = sphere_mesh();
        let baked = ClusterMesh::bake(&cpu);
        for (i, cluster) in baked.clusters.iter().enumerate() {
            let centre = Vec3::new(cluster.sphere[0], cluster.sphere[1], cluster.sphere[2]);
            let radius = cluster.sphere[3];
            let start = cluster.index_offset as usize;
            let end = start + cluster.triangle_count as usize * 3;
            for index in &baked.indices[start..end] {
                let p = Vec3::from_array(cpu.positions[*index as usize].0);
                let distance = (p - centre).length();
                assert!(
                    distance <= radius + 1e-3,
                    "cluster {i}: vertex {distance} from a centre with radius {radius}"
                );
            }
        }
    }

    #[test]
    fn the_normal_cone_contains_its_own_triangle_normals() {
        // The property that makes backface rejection *safe* rather than
        // approximate. Every triangle normal in the cluster has to lie inside
        // the cone, or the cull can reject a cluster that has a visible face.
        let cpu = sphere_mesh();
        let baked = ClusterMesh::bake(&cpu);
        let position = |i: u32| Vec3::from_array(cpu.positions[i as usize].0);

        for (i, cluster) in baked.clusters.iter().enumerate() {
            let axis = Vec3::new(cluster.cone[0], cluster.cone[1], cluster.cone[2]);
            if axis.length() < 1e-6 {
                continue;
            }
            let axis = axis.normalize();
            // cone_cutoff is sin-like in meshoptimizer's formulation: the
            // half-angle is acos of the complement. What has to hold is simply
            // that no normal points further from the axis than the cutoff
            // allows, expressed the way the cull expresses it.
            let limit = (1.0 - cluster.cone[3] * cluster.cone[3]).max(0.0).sqrt();
            let start = cluster.index_offset as usize;
            for t in baked.indices[start..start + cluster.triangle_count as usize * 3].chunks(3) {
                let (a, b, c) = (position(t[0]), position(t[1]), position(t[2]));
                let normal = (b - a).cross(c - a);
                if normal.length() < 1e-9 {
                    continue;
                }
                let dot = normal.normalize().dot(axis);
                assert!(
                    dot >= limit - 1e-2,
                    "cluster {i}: a normal at {dot} against a cone limit of {limit}"
                );
            }
        }
    }

    #[test]
    fn backface_rejection_never_discards_a_cluster_with_a_visible_triangle() {
        // The safety property, and the one worth the most: a conservative cull
        // that errs the wrong way punches holes in the model. Checked
        // exhaustively over every cluster of a sphere from several eye
        // positions, because a single viewpoint could easily miss the cluster
        // that straddles the silhouette.
        let cpu = sphere_mesh();
        let baked = ClusterMesh::bake(&cpu);
        let position = |i: u32| Vec3::from_array(cpu.positions[i as usize].0);

        let mut rejected = 0usize;
        let mut total = 0usize;
        for eye in [
            DVec3::new(0.0, 0.0, 5.0),
            DVec3::new(3.0, 4.0, 0.0),
            DVec3::new(-2.0, 1.0, -6.0),
            DVec3::new(1.2, -1.2, 1.2),
        ] {
            for cluster in &baked.clusters {
                total += 1;
                let sphere = WorldSphere::of(cluster.sphere, &DAffine3::IDENTITY, eye);
                if !cone_is_backfacing(cluster, sphere, &DAffine3::IDENTITY) {
                    continue;
                }
                rejected += 1;
                // Rejected, so every triangle in it must face away.
                let start = cluster.index_offset as usize;
                let end = start + cluster.triangle_count as usize * 3;
                for t in baked.indices[start..end].chunks(3) {
                    let (a, b, c) = (position(t[0]), position(t[1]), position(t[2]));
                    let normal = (b - a).cross(c - a);
                    if normal.length() < 1e-9 {
                        continue;
                    }
                    // Camera-relative: the eye is the origin, so the direction
                    // from the eye to the triangle is the vertex itself.
                    let to_eye = (eye.as_vec3() - a).normalize_or_zero();
                    assert!(
                        normal.normalize().dot(to_eye) <= 1e-3,
                        "a rejected cluster held a triangle facing the eye at {eye:?}"
                    );
                }
            }
        }

        // And it has to reject a useful share, or the cone is dead weight. The
        // plan's figure for a closed mesh is 40-50%.
        let share = rejected as f64 / total as f64;
        assert!(
            share > 0.3,
            "backface rejection discarded only {:.0}% of clusters",
            share * 100.0
        );
    }

    #[test]
    fn a_cluster_behind_the_camera_is_outside_the_frustum() {
        let camera = Camera {
            pivot: DVec3::ZERO,
            distance: 10.0,
            ..Camera::default()
        };
        let frustum = Frustum::from_view_projection(&camera.view_projection(1.0));
        // The camera looks at the origin from +distance along its forward axis,
        // so a sphere well behind the eye is out.
        let behind = camera.eye() - camera.forward() * 5.0;
        let sphere = WorldSphere::of(
            [0.0, 0.0, 0.0, 0.5],
            &DAffine3::from_translation(behind),
            camera.eye(),
        );
        assert!(!frustum.intersects(sphere));
    }

    #[test]
    fn a_cluster_at_the_pivot_is_inside_the_frustum() {
        let camera = Camera {
            pivot: DVec3::ZERO,
            distance: 10.0,
            ..Camera::default()
        };
        let frustum = Frustum::from_view_projection(&camera.view_projection(1.0));
        let sphere = WorldSphere::of([0.0, 0.0, 0.0, 0.5], &DAffine3::IDENTITY, camera.eye());
        assert!(frustum.intersects(sphere));
    }

    #[test]
    fn a_cluster_far_off_to_the_side_is_outside_the_frustum() {
        let camera = Camera {
            pivot: DVec3::ZERO,
            distance: 10.0,
            ..Camera::default()
        };
        let frustum = Frustum::from_view_projection(&camera.view_projection(1.0));
        let aside = camera.right() * 500.0;
        let sphere = WorldSphere::of(
            [0.0, 0.0, 0.0, 1.0],
            &DAffine3::from_translation(aside),
            camera.eye(),
        );
        assert!(!frustum.intersects(sphere));
    }

    #[test]
    fn an_infinite_far_plane_leaves_five_planes_and_keeps_distant_clusters() {
        // The renderer uses reversed-Z with an infinite far plane, so the sixth
        // candidate is degenerate. Normalising it instead of dropping it would
        // divide by a zero-length normal and reject the entire scene — which is
        // why this is a test and not a comment.
        let camera = Camera {
            pivot: DVec3::ZERO,
            distance: 10.0,
            ..Camera::default()
        };
        let frustum = Frustum::from_view_projection(&camera.view_projection(1.0));
        assert_eq!(frustum.plane_count(), 5);

        let far = camera.pivot - camera.forward() * -100_000.0;
        let sphere = WorldSphere::of(
            [0.0, 0.0, 0.0, 10.0],
            &DAffine3::from_translation(far),
            camera.eye(),
        );
        assert!(
            frustum.intersects(sphere),
            "an infinite far plane must not clip distant geometry"
        );
    }

    #[test]
    fn the_padded_plane_slots_accept_everything() {
        // The shader loops a fixed six times and cannot know which slots are
        // real, so the padding has to be inert. A zeroed slot is not inert: it
        // rejects every point, and the whole scene would vanish.
        let camera = Camera {
            pivot: DVec3::ZERO,
            distance: 10.0,
            ..Camera::default()
        };
        let real = Frustum::from_view_projection(&camera.view_projection(1.0));
        let padded = real.to_padded_planes();

        // Rebuilt as a frustum that uses all six slots, so the padding is
        // actually exercised rather than inspected.
        let all_six = Frustum {
            planes: padded,
            count: Frustum::MAX_PLANES,
        };
        let inside = WorldSphere::of([0.0, 0.0, 0.0, 0.5], &DAffine3::IDENTITY, camera.eye());
        assert!(
            all_six.intersects(inside),
            "the padded slots rejected a sphere at the pivot"
        );
        // And the real planes still do their job with the padding present.
        let aside = WorldSphere::of(
            [0.0, 0.0, 0.0, 1.0],
            &DAffine3::from_translation(camera.right() * 500.0),
            camera.eye(),
        );
        assert!(!all_six.intersects(aside));
    }

    #[test]
    fn a_scaled_transform_scales_the_bounding_sphere() {
        let sphere = WorldSphere::of(
            [0.0, 0.0, 0.0, 2.0],
            &DAffine3::from_scale(DVec3::new(1.0, 1.0, 5.0)),
            DVec3::ZERO,
        );
        assert_eq!(sphere.radius, 10.0, "the largest scale factor has to win");
    }

    #[test]
    fn a_rotated_cluster_is_still_backfacing_from_the_other_side() {
        // The cone axis is a direction and has to follow the transform. Without
        // that, rotating an object would leave its backface culling pointing
        // the way the mesh was authored, and half the model would vanish.
        let cpu = sphere_mesh();
        let baked = ClusterMesh::bake(&cpu);
        let cluster = baked
            .clusters
            .iter()
            .max_by(|a, b| a.cone[2].total_cmp(&b.cone[2]))
            .expect("a sphere has clusters");

        let eye = DVec3::new(0.0, 0.0, 6.0);
        let facing = WorldSphere::of(cluster.sphere, &DAffine3::IDENTITY, eye);
        assert!(
            !cone_is_backfacing(cluster, facing, &DAffine3::IDENTITY),
            "a cluster whose cone points at the eye is not backfacing"
        );

        // Turn the object half a revolution: the same cluster now faces away.
        let turned = DAffine3::from_axis_angle(DVec3::Y, std::f64::consts::PI);
        let sphere = WorldSphere::of(cluster.sphere, &turned, eye);
        assert!(
            cone_is_backfacing(cluster, sphere, &turned),
            "the cone axis did not follow the transform"
        );
    }

    #[test]
    fn an_empty_mesh_bakes_to_nothing_rather_than_panicking() {
        let cpu = RenderMeshCpu::extract(&kreate_mesh::MeshData::default());
        let baked = ClusterMesh::bake(&cpu);
        assert!(baked.clusters.is_empty());
        assert!(baked.indices.is_empty());
        assert_eq!(baked.memory_bytes(), 0);
    }

    #[test]
    fn the_cluster_record_is_the_size_the_shader_expects() {
        // The bytes are handed straight to WGSL as `array<Cluster>`, so a field
        // added without a matching shader change reads garbage cones and culls
        // at random. Pinned here because the symptom is invisible geometry
        // rather than a compile error.
        assert_eq!(std::mem::size_of::<Cluster>(), 48);
        assert_eq!(std::mem::align_of::<Cluster>(), 4);
    }

    #[test]
    fn a_dense_mesh_clusters_near_the_triangle_cap() {
        // Clusters far below the cap would mean the fixed-size draw wastes most
        // of its vertices on degenerate triangles, which is the one way this
        // design can be slower than a draw per object.
        let cpu = sphere_mesh();
        let baked = ClusterMesh::bake(&cpu);
        let triangles: u32 = baked.clusters.iter().map(|c| c.triangle_count).sum();
        let average = triangles as f64 / baked.clusters.len() as f64;
        assert!(
            average > MAX_TRIANGLES as f64 * 0.7,
            "clusters averaged {average:.0} triangles against a cap of {MAX_TRIANGLES}"
        );
    }
}
