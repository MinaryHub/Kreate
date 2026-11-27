//! High-poly to low-poly texture baker (Wave D1).
//!
//! Rasterizes each UV triangle of a low-poly target mesh, fires inward/outward
//! cage rays against a high-poly source BVH in parallel (`rayon`), and bakes:
//! - Tangent-space normal maps (OpenGL +Y or DirectX -Y, matching `tangents.rs`)
//! - Ambient occlusion (cosine-weighted hemisphere raytracing)
//! - Base color (transferring high-poly material/texture/vertex color)
//! - Metallic-Roughness (glTF G=roughness, B=metallic packing)
//! - Height (normalized signed displacement along the low-poly normal)
//! - Curvature (mean dihedral curvature: 0.5 flat, >0.5 convex, <0.5 concave)
//!
//! Finished bakes run an iterative UV margin dilation pass so mipmaps never
//! bleed background black across UV island seams.

use glam::{DAffine3, Vec2, Vec3, Vec4};
use kreate_math::{linear_to_srgb, srgb_to_linear, Bvh, Ray};
use rayon::prelude::*;

use crate::attribute::{names, AttrData, Domain};
use crate::tangents::{barycentric_2d, compute_corner_tangents, tangent_basis};
use crate::MeshData;

/// Which texture channel/map to bake.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BakeKind {
    #[default]
    TangentNormal,
    AmbientOcclusion,
    BaseColor,
    MetallicRoughness,
    Height,
    Curvature,
}

impl BakeKind {
    pub const ALL: [BakeKind; 6] = [
        BakeKind::TangentNormal,
        BakeKind::AmbientOcclusion,
        BakeKind::BaseColor,
        BakeKind::MetallicRoughness,
        BakeKind::Height,
        BakeKind::Curvature,
    ];

    pub fn from_index(i: u32) -> Self {
        Self::ALL
            .get(i as usize)
            .copied()
            .unwrap_or(Self::TangentNormal)
    }

    pub fn suffix(self) -> &'static str {
        match self {
            BakeKind::TangentNormal => "Normal",
            BakeKind::AmbientOcclusion => "AO",
            BakeKind::BaseColor => "BaseColor",
            BakeKind::MetallicRoughness => "ORM",
            BakeKind::Height => "Height",
            BakeKind::Curvature => "Curvature",
        }
    }

    pub fn is_srgb(self) -> bool {
        matches!(self, BakeKind::BaseColor)
    }
}

/// Baker configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BakeConfig {
    pub width: u32,
    pub height: u32,
    pub kind: BakeKind,
    /// Outward offset along the low-poly normal before firing inward rays, in metres.
    pub cage_extrusion: f32,
    /// Maximum ray search distance (from the extruded cage), in metres.
    pub max_ray_distance: f32,
    /// UV island seam padding in pixels (dilation passes).
    pub margin: u32,
    /// Hemisphere rays per texel for `AmbientOcclusion`.
    pub ao_samples: u32,
    /// Ray length for `AmbientOcclusion`, in metres.
    pub ao_distance: f32,
    /// Invert green channel for DirectX normal maps (false = OpenGL +Y, glTF standard).
    pub flip_normal_y: bool,
    /// Auto-smooth angle (radians) used when meshes lack authored custom normals.
    pub auto_smooth: f32,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            kind: BakeKind::TangentNormal,
            cage_extrusion: 0.05,
            max_ray_distance: 0.5,
            margin: 16,
            ao_samples: 16,
            ao_distance: 1.0,
            flip_normal_y: false,
            auto_smooth: 30f32.to_radians(),
        }
    }
}

/// Optional flat RGBA8 texture borrowed from a high-poly source material.
#[derive(Clone, Copy)]
pub struct BakeImage<'a> {
    pub width: u32,
    pub height: u32,
    pub rgba8: &'a [u8],
}

impl<'a> BakeImage<'a> {
    fn sample(&self, uv: Vec2) -> [f32; 4] {
        if self.width == 0 || self.height == 0 || self.rgba8.len() < 4 {
            return [1.0; 4];
        }
        let u = uv.x.rem_euclid(1.0);
        // Kreate UV origin is bottom-left; image rows are top-to-bottom.
        let v = 1.0 - uv.y.rem_euclid(1.0);
        let x = ((u * self.width as f32).floor() as u32).min(self.width - 1);
        let y = ((v * self.height as f32).floor() as u32).min(self.height - 1);
        let at = ((y * self.width + x) * 4) as usize;
        if let Some(px) = self.rgba8.get(at..at + 4) {
            [
                px[0] as f32 / 255.0,
                px[1] as f32 / 255.0,
                px[2] as f32 / 255.0,
                px[3] as f32 / 255.0,
            ]
        } else {
            [1.0; 4]
        }
    }
}

/// Material info per slot of a high-poly source mesh.
#[derive(Clone, Copy)]
pub struct BakeMaterial<'a> {
    /// Linear base color `[r, g, b]`.
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub base_color_image: Option<BakeImage<'a>>,
    pub metallic_roughness_image: Option<BakeImage<'a>>,
}

impl Default for BakeMaterial<'_> {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            base_color_image: None,
            metallic_roughness_image: None,
        }
    }
}

/// One high-poly source mesh contributing to the bake.
pub struct BakeSource<'a> {
    pub mesh: &'a MeshData,
    pub world: DAffine3,
    pub materials: Vec<BakeMaterial<'a>>,
}

/// Result of a bake: RGBA8 pixels top-to-bottom + hit statistics.
#[derive(Clone, Debug)]
pub struct BakeOutput {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub covered_texels: usize,
    pub hit_texels: usize,
}

struct HighMeshData<'a> {
    positions: Vec<Vec3>,
    tri_indices: Vec<[u32; 3]>,
    tri_normals: Vec<[Vec3; 3]>,
    tri_uvs: Vec<[Vec2; 3]>,
    tri_colors: Vec<[Vec4; 3]>,
    tri_curvatures: Vec<[f32; 3]>,
    tri_materials: Vec<BakeMaterial<'a>>,
    bvh: Bvh,
}

fn transform_point(m: &DAffine3, p: Vec3) -> Vec3 {
    m.transform_point3(p.as_dvec3()).as_vec3()
}

fn normal_matrix(m: &DAffine3) -> glam::Mat3 {
    let inv = m.matrix3.inverse().transpose();
    inv.as_mat3()
}

fn vertex_colors_per_corner(mesh: &MeshData) -> Option<Vec<Vec4>> {
    let layer = mesh.attributes().get(names::COLOR)?;
    let corners = mesh.corner_verts();
    match (&layer.domain, &layer.data) {
        (Domain::Corner, AttrData::F32x4(v)) => Some(v.to_vec()),
        (Domain::Point, AttrData::F32x4(v)) => {
            Some(corners.iter().map(|&p| v[p as usize]).collect())
        }
        (Domain::Corner, AttrData::F32x3(v)) => Some(v.iter().map(|c| c.extend(1.0)).collect()),
        (Domain::Point, AttrData::F32x3(v)) => {
            Some(corners.iter().map(|&p| v[p as usize].extend(1.0)).collect())
        }
        _ => None,
    }
}

/// Computes signed mean curvature per vertex from dihedral angles around
/// incident edges: positive on convex ridges, negative in concave valleys.
fn vertex_curvatures(mesh: &MeshData, world_pos: &[Vec3]) -> Vec<f32> {
    let n_verts = world_pos.len();
    let mut sum = vec![0.0f32; n_verts];
    let mut count = vec![0u32; n_verts];
    let face_normals: Vec<Vec3> = mesh
        .compute_face_normals()
        .into_iter()
        .map(|n| n.normalize_or_zero())
        .collect();
    let corner_edges = mesh.corner_edges();
    let edges = mesh.edges();
    let mut edge_faces: Vec<[u32; 2]> = vec![[u32::MAX; 2]; edges.len()];
    for f in 0..mesh.num_faces() {
        for c in mesh.face_corner_range(f) {
            let e = corner_edges[c] as usize;
            if e < edge_faces.len() {
                if edge_faces[e][0] == u32::MAX {
                    edge_faces[e][0] = f;
                } else if edge_faces[e][1] == u32::MAX {
                    edge_faces[e][1] = f;
                }
            }
        }
    }
    let mut face_center = Vec::with_capacity(mesh.num_faces() as usize);
    for f in 0..mesh.num_faces() {
        let verts = mesh.face_verts(f);
        let c =
            verts.iter().map(|&v| world_pos[v as usize]).sum::<Vec3>() / verts.len().max(1) as f32;
        face_center.push(c);
    }
    for (e_idx, &[va, vb]) in edges.iter().enumerate() {
        let [f0, f1] = edge_faces[e_idx];
        if f0 == u32::MAX || f1 == u32::MAX {
            continue;
        }
        let n0 = face_normals[f0 as usize];
        let n1 = face_normals[f1 as usize];
        let angle = n0.dot(n1).clamp(-1.0, 1.0).acos();
        // Determine convex vs concave: if f1's center is behind f0's plane, edge is convex.
        let sign = if (face_center[f1 as usize] - face_center[f0 as usize]).dot(n0) <= 0.0 {
            1.0
        } else {
            -1.0
        };
        let signed = sign * angle;
        for v in [va as usize, vb as usize] {
            if v < n_verts {
                sum[v] += signed;
                count[v] += 1;
            }
        }
    }
    sum.into_iter()
        .zip(count)
        .map(|(s, c)| if c > 0 { s / c as f32 } else { 0.0 })
        .collect()
}

fn build_high<'a>(sources: &[BakeSource<'a>], auto_smooth: f32) -> HighMeshData<'a> {
    let mut positions = Vec::new();
    let mut tri_indices = Vec::new();
    let mut tri_normals = Vec::new();
    let mut tri_uvs = Vec::new();
    let mut tri_colors = Vec::new();
    let mut tri_curvatures = Vec::new();
    let mut tri_materials = Vec::new();

    for src in sources {
        let base_v = positions.len() as u32;
        let mesh = src.mesh;
        let nmat = normal_matrix(&src.world);
        let world_pos: Vec<Vec3> = mesh
            .positions()
            .iter()
            .map(|&p| transform_point(&src.world, p))
            .collect();
        let curv = vertex_curvatures(mesh, &world_pos);
        positions.extend_from_slice(&world_pos);

        let corner_normals = mesh.resolved_corner_normals(auto_smooth);
        let uvs = mesh.attributes().get_f32x2(names::UV);
        let colors = vertex_colors_per_corner(mesh);
        let mat_idx = mesh.attributes().get_i32(names::MATERIAL_INDEX);
        let corner_verts = mesh.corner_verts();
        let tri = mesh.triangulate_corners();

        for t in 0..tri.num_triangles() {
            let f = tri.tri_to_face[t] as usize;
            let c0 = tri.corners[t * 3] as usize;
            let c1 = tri.corners[t * 3 + 1] as usize;
            let c2 = tri.corners[t * 3 + 2] as usize;
            let v0 = corner_verts[c0];
            let v1 = corner_verts[c1];
            let v2 = corner_verts[c2];
            tri_indices.push([base_v + v0, base_v + v1, base_v + v2]);
            let mut ns = [Vec3::Z; 3];
            for (k, c) in [c0, c1, c2].into_iter().enumerate() {
                ns[k] = (nmat * corner_normals[c]).normalize_or_zero();
                if ns[k].length_squared() < 0.5 {
                    ns[k] = Vec3::Z;
                }
            }
            tri_normals.push(ns);
            tri_uvs.push(match uvs {
                Some(u) => [u[c0], u[c1], u[c2]],
                None => [Vec2::ZERO; 3],
            });
            tri_colors.push(match &colors {
                Some(col) => [col[c0], col[c1], col[c2]],
                None => [Vec4::ONE; 3],
            });
            tri_curvatures.push([curv[v0 as usize], curv[v1 as usize], curv[v2 as usize]]);
            let slot = mat_idx
                .and_then(|m| m.get(f).copied())
                .filter(|&s| s >= 0)
                .unwrap_or(0) as usize;
            let mat = src.materials.get(slot).copied().unwrap_or_default();
            tri_materials.push(mat);
        }
    }

    let bvh = Bvh::build(&positions, &tri_indices);
    HighMeshData {
        positions,
        tri_indices,
        tri_normals,
        tri_uvs,
        tri_colors,
        tri_curvatures,
        tri_materials,
        bvh,
    }
}

/// Deterministic PCG hash for hemisphere AO sampling.
#[inline]
fn hash_u32(mut x: u32) -> u32 {
    x = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Cosine-weighted hemisphere direction around `n` from uniform `(u1, u2) ∈ [0,1)²`.
#[inline]
fn cosine_hemisphere(n: Vec3, u1: f32, u2: f32) -> Vec3 {
    let r = u1.clamp(0.0, 1.0).sqrt();
    let phi = std::f32::consts::TAU * u2;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - u1).max(0.0).sqrt();
    let ref_axis = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    let t = ref_axis.cross(n).normalize();
    let b = n.cross(t);
    (t * x + b * y + n * z).normalize()
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Bakes `sources` (high-poly meshes) onto `target` (low-poly mesh with UVs).
///
/// Returns `None` only if `target` has no `UV` layer or zero dimensions.
pub fn bake_textures(
    target: &MeshData,
    target_world: DAffine3,
    sources: &[BakeSource<'_>],
    config: &BakeConfig,
) -> Option<BakeOutput> {
    let w = config.width.clamp(16, 8192);
    let h = config.height.clamp(16, 8192);
    let uvs = target.attributes().get_f32x2(names::UV)?;
    if target.num_faces() == 0 || sources.is_empty() {
        return None;
    }

    let high = build_high(sources, config.auto_smooth);
    if high.tri_indices.is_empty() {
        return None;
    }

    // Prepare low-poly world-space positions, normals, and MikkTSpace tangents.
    let nmat = normal_matrix(&target_world);
    let low_pos: Vec<Vec3> = target
        .positions()
        .iter()
        .map(|&p| transform_point(&target_world, p))
        .collect();
    let low_normals_local = target.resolved_corner_normals(config.auto_smooth);
    let low_tangents_local = compute_corner_tangents(target, &low_normals_local);
    let low_normals: Vec<Vec3> = low_normals_local
        .iter()
        .map(|&n| {
            let wn = (nmat * n).normalize_or_zero();
            if wn.length_squared() > 0.5 {
                wn
            } else {
                Vec3::Z
            }
        })
        .collect();
    let low_tangents: Vec<[f32; 4]> = low_tangents_local
        .iter()
        .map(|t| {
            let wt =
                (target_world.matrix3.as_mat3() * Vec3::new(t[0], t[1], t[2])).normalize_or_zero();
            [wt.x, wt.y, wt.z, t[3]]
        })
        .collect();

    let corner_verts = target.corner_verts();
    let tri = target.triangulate_corners();
    let num_tris = tri.num_triangles();

    // Rasterize low-poly UV triangles to collect per-texel low-poly geometry.
    // Using a compact slot array (one entry per texel) makes parallel raytracing
    // over covered texels deterministic and lock-free.
    #[derive(Clone, Copy)]
    struct TexelSample {
        pos: Vec3,
        normal: Vec3,
        tangent: [f32; 4],
    }

    let total = (w as usize) * (h as usize);
    let mut coverage: Vec<Option<TexelSample>> = vec![None; total];
    let wf = w as f32;
    let hf = h as f32;

    for t in 0..num_tris {
        let c0 = tri.corners[t * 3] as usize;
        let c1 = tri.corners[t * 3 + 1] as usize;
        let c2 = tri.corners[t * 3 + 2] as usize;

        // Convert bottom-left UVs to top-left pixel coordinates without
        // wrapping 1.0 back to 0.0 (which would collapse full [0,1] quads).
        let to_px = |uv: Vec2| Vec2::new(uv.x * wf, (1.0 - uv.y) * hf);
        let a = to_px(uvs[c0]);
        let b = to_px(uvs[c1]);
        let c = to_px(uvs[c2]);

        let min_x = a.x.min(b.x).min(c.x).floor().max(0.0) as u32;
        let max_x = a.x.max(b.x).max(c.x).ceil().min(wf - 1.0).max(0.0) as u32;
        let min_y = a.y.min(b.y).min(c.y).floor().max(0.0) as u32;
        let max_y = a.y.max(b.y).max(c.y).ceil().min(hf - 1.0).max(0.0) as u32;

        let p0 = low_pos[corner_verts[c0] as usize];
        let p1 = low_pos[corner_verts[c1] as usize];
        let p2 = low_pos[corner_verts[c2] as usize];
        let n0 = low_normals[c0];
        let n1 = low_normals[c1];
        let n2 = low_normals[c2];
        let t0 = low_tangents[c0];
        let t1 = low_tangents[c1];
        let t2 = low_tangents[c2];

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let p = Vec2::new(px as f32 + 0.5, py as f32 + 0.5);
                let Some([u, v, wb]) = barycentric_2d(p, a, b, c) else {
                    continue;
                };
                // Tiny negative tolerance catches shared triangle edges without holes.
                if u < -1e-4 || v < -1e-4 || wb < -1e-4 {
                    continue;
                }
                let pos = p0 * u + p1 * v + p2 * wb;
                let normal = (n0 * u + n1 * v + n2 * wb).normalize_or_zero();
                let normal = if normal.length_squared() > 0.5 {
                    normal
                } else {
                    n0
                };
                let tv = Vec3::new(t0[0], t0[1], t0[2]) * u
                    + Vec3::new(t1[0], t1[1], t1[2]) * v
                    + Vec3::new(t2[0], t2[1], t2[2]) * wb;
                let sign = if (t0[3] * u + t1[3] * v + t2[3] * wb) < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                coverage[(py * w + px) as usize] = Some(TexelSample {
                    pos,
                    normal,
                    tangent: [tv.x, tv.y, tv.z, sign],
                });
            }
        }
    }

    let bg_color: [u8; 4] = match config.kind {
        BakeKind::TangentNormal => [128, 128, 255, 255],
        BakeKind::AmbientOcclusion => [255, 255, 255, 255],
        BakeKind::BaseColor => [0, 0, 0, 255],
        BakeKind::MetallicRoughness => [255, 128, 0, 255],
        BakeKind::Height | BakeKind::Curvature => [128, 128, 128, 255],
    };

    let cage = config.cage_extrusion.max(1e-4);
    let max_dist = config.max_ray_distance.max(cage * 2.0);
    let ao_samples = config.ao_samples.clamp(1, 256);
    let ao_dist = config.ao_distance.max(1e-4);

    // Parallel raycast & shade per texel.
    let results: Vec<([u8; 4], bool, bool)> = coverage
        .par_iter()
        .enumerate()
        .map(|(idx, sample)| {
            let Some(s) = sample else {
                return (bg_color, false, false);
            };
            // Primary inward ray from extruded cage along -N.
            let origin_in = s.pos + s.normal * cage;
            let ray_in = Ray::new(origin_in, -s.normal);
            let mut hit = high
                .bvh
                .raycast(&high.positions, &high.tri_indices, &ray_in, max_dist);
            let mut signed_height = hit.map(|h| cage - h.t);

            // Fallback outward ray if the high-poly surface extends past the cage.
            if hit.is_none() {
                let ray_out = Ray::new(s.pos - s.normal * (cage * 0.25), s.normal);
                if let Some(h_out) =
                    high.bvh
                        .raycast(&high.positions, &high.tri_indices, &ray_out, max_dist)
                {
                    signed_height = Some(h_out.t - cage * 0.25);
                    hit = Some(h_out);
                }
            }

            let Some(hit) = hit else {
                return (bg_color, true, false);
            };

            let tri_i = hit.primitive as usize;
            let (bu, bv) = hit.barycentric;
            let bw = 1.0 - bu - bv;
            let hn_arr = high.tri_normals[tri_i];
            let hit_normal = (hn_arr[0] * bw + hn_arr[1] * bu + hn_arr[2] * bv).normalize_or_zero();
            let hit_normal = if hit_normal.length_squared() > 0.5 {
                hit_normal
            } else {
                s.normal
            };

            let rgba = match config.kind {
                BakeKind::TangentNormal => {
                    let (t, b, n) = tangent_basis(s.normal, s.tangent);
                    let ts_x = hit_normal.dot(t);
                    let mut ts_y = hit_normal.dot(b);
                    if config.flip_normal_y {
                        ts_y = -ts_y;
                    }
                    let ts_z = hit_normal.dot(n);
                    let ts = Vec3::new(ts_x, ts_y, ts_z).normalize_or_zero();
                    [
                        to_u8(ts.x * 0.5 + 0.5),
                        to_u8(ts.y * 0.5 + 0.5),
                        to_u8(ts.z * 0.5 + 0.5),
                        255,
                    ]
                }
                BakeKind::AmbientOcclusion => {
                    let origin = hit.point + hit_normal * (ao_dist * 1e-3);
                    let mut unoccluded = 0u32;
                    for k in 0..ao_samples {
                        let seed = (idx as u32).wrapping_mul(1664525) ^ k.wrapping_mul(1013904223);
                        let u1 = (hash_u32(seed) as f32) * (1.0 / 4294967296.0);
                        let u2 =
                            (hash_u32(seed.wrapping_add(0x9e3779b9)) as f32) * (1.0 / 4294967296.0);
                        let dir = cosine_hemisphere(hit_normal, u1, u2);
                        let ray = Ray::new(origin, dir);
                        if high
                            .bvh
                            .raycast(&high.positions, &high.tri_indices, &ray, ao_dist)
                            .is_none()
                        {
                            unoccluded += 1;
                        }
                    }
                    let ao = to_u8(unoccluded as f32 / ao_samples as f32);
                    [ao, ao, ao, 255]
                }
                BakeKind::BaseColor => {
                    let uv_arr = high.tri_uvs[tri_i];
                    let hit_uv = uv_arr[0] * bw + uv_arr[1] * bu + uv_arr[2] * bv;
                    let col_arr = high.tri_colors[tri_i];
                    let vcol = col_arr[0] * bw + col_arr[1] * bu + col_arr[2] * bv;
                    let mat = high.tri_materials[tri_i];
                    let mut c = [
                        mat.base_color[0] * vcol.x,
                        mat.base_color[1] * vcol.y,
                        mat.base_color[2] * vcol.z,
                    ];
                    if let Some(img) = mat.base_color_image {
                        let tex = img.sample(hit_uv);
                        // Image texture is sRGB encoded; multiply in linear space.
                        for ch in 0..3 {
                            c[ch] *= srgb_to_linear(tex[ch]);
                        }
                    }
                    [
                        to_u8(linear_to_srgb(c[0].clamp(0.0, 1.0))),
                        to_u8(linear_to_srgb(c[1].clamp(0.0, 1.0))),
                        to_u8(linear_to_srgb(c[2].clamp(0.0, 1.0))),
                        255,
                    ]
                }
                BakeKind::MetallicRoughness => {
                    let uv_arr = high.tri_uvs[tri_i];
                    let hit_uv = uv_arr[0] * bw + uv_arr[1] * bu + uv_arr[2] * bv;
                    let mat = high.tri_materials[tri_i];
                    let mut roughness = mat.roughness;
                    let mut metallic = mat.metallic;
                    if let Some(img) = mat.metallic_roughness_image {
                        let tex = img.sample(hit_uv);
                        roughness *= tex[1];
                        metallic *= tex[2];
                    }
                    // glTF packing: R=1.0 (unused/AO), G=Roughness, B=Metallic.
                    [255, to_u8(roughness), to_u8(metallic), 255]
                }
                BakeKind::Height => {
                    let d = signed_height.unwrap_or(0.0);
                    let norm = (d / (cage * 2.0) + 0.5).clamp(0.0, 1.0);
                    let v = to_u8(norm);
                    [v, v, v, 255]
                }
                BakeKind::Curvature => {
                    let c_arr = high.tri_curvatures[tri_i];
                    let curv = c_arr[0] * bw + c_arr[1] * bu + c_arr[2] * bv;
                    // Map [-pi/2, +pi/2] -> [0, 1] centered at 0.5 (flat).
                    let v = to_u8((curv / std::f32::consts::FRAC_PI_2 * 0.5 + 0.5).clamp(0.0, 1.0));
                    [v, v, v, 255]
                }
            };
            (rgba, true, true)
        })
        .collect();

    let mut rgba8 = vec![0u8; total * 4];
    let mut written = vec![false; total];
    let mut covered_texels = 0usize;
    let mut hit_texels = 0usize;

    for (i, (px, cov, hit)) in results.into_iter().enumerate() {
        rgba8[i * 4..i * 4 + 4].copy_from_slice(&px);
        if cov {
            covered_texels += 1;
        }
        if hit {
            hit_texels += 1;
            written[i] = true;
        }
    }

    // UV margin dilation: expand island border texels outward to prevent mipmap
    // seam bleeding.
    dilate_margin(
        &mut rgba8,
        &mut written,
        w as usize,
        h as usize,
        config.margin,
    );

    Some(BakeOutput {
        width: w,
        height: h,
        rgba8,
        covered_texels,
        hit_texels,
    })
}

/// Expands written texels outward by `margin` pixels using 8-connected neighbor
/// averaging.
fn dilate_margin(rgba: &mut [u8], written: &mut [bool], w: usize, h: usize, margin: u32) {
    if margin == 0 || w == 0 || h == 0 {
        return;
    }
    let mut next_rgba = rgba.to_vec();
    let mut next_written = written.to_vec();
    for _ in 0..margin {
        let mut any = false;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if written[i] {
                    continue;
                }
                let mut sum = [0u32; 4];
                let mut count = 0u32;
                let y0 = y.saturating_sub(1);
                let y1 = (y + 1).min(h - 1);
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(w - 1);
                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        let ni = ny * w + nx;
                        if written[ni] {
                            for c in 0..4 {
                                sum[c] += rgba[ni * 4 + c] as u32;
                            }
                            count += 1;
                        }
                    }
                }
                if let Some(n) = std::num::NonZeroU32::new(count) {
                    for c in 0..4 {
                        next_rgba[i * 4 + c] = (sum[c] / n) as u8;
                    }
                    next_written[i] = true;
                    any = true;
                }
            }
        }
        rgba.copy_from_slice(&next_rgba);
        written.copy_from_slice(&next_written);
        if !any {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn bake_tangent_normal_of_sphere_onto_cube_produces_varied_normals_and_dilates() {
        let low = primitives::cube(2.0);
        let high = primitives::uv_sphere(1.05, 24, 16);
        let src = BakeSource {
            mesh: &high,
            world: DAffine3::IDENTITY,
            materials: vec![BakeMaterial::default()],
        };
        let cfg = BakeConfig {
            width: 64,
            height: 64,
            kind: BakeKind::TangentNormal,
            cage_extrusion: 0.4,
            max_ray_distance: 2.0,
            margin: 4,
            ..Default::default()
        };
        let out = bake_textures(&low, DAffine3::IDENTITY, &[src], &cfg).expect("bakes");
        assert!(out.hit_texels > 1000, "hits={}", out.hit_texels);
        // Baked normals should deviate from flat [128, 128, 255] because the
        // sphere curves across the flat cube faces.
        let varied = out
            .rgba8
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| (px[0] as i32 - 128).abs() > 15 || (px[1] as i32 - 128).abs() > 15)
            .count();
        assert!(
            varied > 500,
            "expected curved tangent normals, got {varied}"
        );
    }

    #[test]
    fn bake_ao_basecolor_height_and_curvature_succeed() {
        let low = primitives::cube(2.0);
        let high = primitives::uv_sphere(1.02, 16, 12);
        let src = BakeSource {
            mesh: &high,
            world: DAffine3::IDENTITY,
            materials: vec![BakeMaterial {
                base_color: [0.8, 0.1, 0.1],
                metallic: 0.9,
                roughness: 0.2,
                ..Default::default()
            }],
        };
        for kind in [
            BakeKind::AmbientOcclusion,
            BakeKind::BaseColor,
            BakeKind::MetallicRoughness,
            BakeKind::Height,
            BakeKind::Curvature,
        ] {
            let cfg = BakeConfig {
                width: 32,
                height: 32,
                kind,
                cage_extrusion: 0.4,
                max_ray_distance: 2.0,
                margin: 2,
                ao_samples: 8,
                ..Default::default()
            };
            let out = bake_textures(&low, DAffine3::IDENTITY, std::slice::from_ref(&src), &cfg)
                .expect("bakes");
            assert!(out.hit_texels > 100, "{kind:?} hits={}", out.hit_texels);
            if kind == BakeKind::BaseColor {
                // Red channel must dominate green/blue.
                let red_dominant = out
                    .rgba8
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|px| px[0] > 180 && px[1] < 120)
                    .count();
                assert!(red_dominant > 100, "BaseColor red transfer failed");
            }
            if kind == BakeKind::MetallicRoughness {
                let orm_match = out
                    .rgba8
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|px| px[1] < 70 && px[2] > 200)
                    .count();
                assert!(orm_match > 100, "ORM metallic/roughness packing failed");
            }
        }
    }
}
