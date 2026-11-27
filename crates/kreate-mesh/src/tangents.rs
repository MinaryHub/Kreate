//! Per-corner tangent frames (glTF / MikkTSpace convention).
//!
//! Each corner receives `[tx, ty, tz, sign]` where `T` is orthogonalized
//! against the corner normal `N` and `B = sign * (N × T)` points along
//! increasing texture V (glTF 2.0 §3.7.2.1). Reused by both the glTF exporter
//! and the high-to-low texture baker so baked normal maps and exported tangent
//! attributes agree to the bit.

use glam::{Vec2, Vec3};

use crate::attribute::names;
use crate::MeshData;

/// Computes one `[tx, ty, tz, sign]` tangent per corner of `mesh`.
///
/// Accumulates area-weighted triangle tangents and bitangents onto corners that
/// share `(vertex, normal, uv)` (so UV seams and hard edges stay split), then
/// Gram-Schmidt orthogonalizes against `corner_normals[c]`. Corners whose
/// triangles have degenerate UVs receive a deterministic orthonormal fallback.
pub fn compute_corner_tangents(mesh: &MeshData, corner_normals: &[Vec3]) -> Vec<[f32; 4]> {
    let num_corners = mesh.num_corners() as usize;
    if num_corners == 0 {
        return Vec::new();
    }
    let positions = mesh.positions();
    let corner_verts = mesh.corner_verts();
    let uvs = mesh.attributes().get_f32x2(names::UV);

    let mut tan_sum = vec![Vec3::ZERO; num_corners];
    let mut bitan_sum = vec![Vec3::ZERO; num_corners];

    if let Some(uvs) = uvs {
        let tri = mesh.triangulate_corners();
        // Accumulate per triangle onto its three corners.
        for t in 0..tri.num_triangles() {
            let c0 = tri.corners[t * 3] as usize;
            let c1 = tri.corners[t * 3 + 1] as usize;
            let c2 = tri.corners[t * 3 + 2] as usize;
            let p0 = positions[corner_verts[c0] as usize];
            let p1 = positions[corner_verts[c1] as usize];
            let p2 = positions[corner_verts[c2] as usize];
            let uv0 = uvs[c0];
            let uv1 = uvs[c1];
            let uv2 = uvs[c2];

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let duv1 = uv1 - uv0;
            let duv2 = uv2 - uv0;
            let det = duv1.x * duv2.y - duv1.y * duv2.x;
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            // Area-weighted (multiply by triangle area = 0.5 * |e1 × e2|).
            let area = e1.cross(e2).length() * 0.5;
            if area <= 1e-12 {
                continue;
            }
            let t_dir = ((e1 * duv2.y - e2 * duv1.y) * inv).normalize_or_zero() * area;
            let b_dir = ((e2 * duv1.x - e1 * duv2.x) * inv).normalize_or_zero() * area;
            for c in [c0, c1, c2] {
                tan_sum[c] += t_dir;
                bitan_sum[c] += b_dir;
            }
        }

        // Smooth tangents across corners at the same vertex that agree on normal
        // and UV (and handedness) so shared smooth vertices have one continuous
        // tangent frame while UV seams / mirrored UVs stay cleanly split.
        let mut by_vert: Vec<Vec<usize>> = vec![Vec::new(); positions.len()];
        for (c, &v) in corner_verts.iter().enumerate() {
            if (v as usize) < by_vert.len() {
                by_vert[v as usize].push(c);
            }
        }
        let raw_t = tan_sum.clone();
        let raw_b = bitan_sum.clone();
        for group in &by_vert {
            if group.len() <= 1 {
                continue;
            }
            for &c in group {
                let nc = corner_normals.get(c).copied().unwrap_or(Vec3::Z);
                let uvc = uvs[c];
                let tc = raw_t[c];
                let bc = raw_b[c];
                let hc = nc.cross(tc).dot(bc);
                let mut acc_t = Vec3::ZERO;
                let mut acc_b = Vec3::ZERO;
                for &other in group {
                    let no = corner_normals.get(other).copied().unwrap_or(Vec3::Z);
                    if nc.dot(no) < 0.999 {
                        continue;
                    }
                    let uvo = uvs[other];
                    if (uvc - uvo).length_squared() > 1e-10 {
                        continue;
                    }
                    let to = raw_t[other];
                    let bo = raw_b[other];
                    let ho = no.cross(to).dot(bo);
                    if hc * ho < 0.0 {
                        continue;
                    }
                    acc_t += to;
                    acc_b += bo;
                }
                if acc_t.length_squared() > 1e-18 {
                    tan_sum[c] = acc_t;
                    bitan_sum[c] = acc_b;
                }
            }
        }
    }

    let mut out = Vec::with_capacity(num_corners);
    for c in 0..num_corners {
        let n = corner_normals
            .get(c)
            .copied()
            .unwrap_or(Vec3::Z)
            .normalize_or_zero();
        let n = if n.length_squared() > 0.5 { n } else { Vec3::Z };
        let t_raw = tan_sum[c];
        let b_raw = bitan_sum[c];
        // Gram-Schmidt orthogonalize T against N.
        let t_orth = t_raw - n * n.dot(t_raw);
        let (t, sign) = if t_orth.length_squared() > 1e-12 {
            let t = t_orth.normalize();
            let sign = if n.cross(t).dot(b_raw) < 0.0 {
                -1.0
            } else {
                1.0
            };
            (t, sign)
        } else {
            (fallback_tangent(n), 1.0)
        };
        out.push([t.x, t.y, t.z, sign]);
    }
    out
}

/// Deterministic unit vector perpendicular to `n`.
#[inline]
pub fn fallback_tangent(n: Vec3) -> Vec3 {
    let ref_axis = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    ref_axis.cross(n).normalize_or_zero()
}

/// Reconstructs the orthonormal `(T, B, N)` basis from a normal and a glTF
/// `[tx, ty, tz, sign]` tangent.
#[inline]
pub fn tangent_basis(n: Vec3, tangent: [f32; 4]) -> (Vec3, Vec3, Vec3) {
    let n = n.normalize_or_zero();
    let n = if n.length_squared() > 0.5 { n } else { Vec3::Z };
    let t_raw = Vec3::new(tangent[0], tangent[1], tangent[2]);
    let t_orth = t_raw - n * n.dot(t_raw);
    let t = if t_orth.length_squared() > 1e-10 {
        t_orth.normalize()
    } else {
        fallback_tangent(n)
    };
    let sign = if tangent[3] < 0.0 { -1.0 } else { 1.0 };
    let b = n.cross(t) * sign;
    (t, b, n)
}

/// Helper: computes barycentric weights `(w0, w1, w2)` of `p` in 2D triangle
/// `(a, b, c)`, or `None` if degenerate.
#[inline]
pub(crate) fn barycentric_2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<[f32; 3]> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let den = v0.x * v1.y - v1.x * v0.y;
    if den.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / den;
    let v = (v2.x * v1.y - v1.x * v2.y) * inv;
    let w = (v0.x * v2.y - v2.x * v0.y) * inv;
    let u = 1.0 - v - w;
    Some([u, v, w])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn cube_tangents_are_orthonormal_to_normals() {
        let cube = primitives::cube(1.0);
        let normals = cube.compute_corner_normals(30f32.to_radians());
        let tangents = compute_corner_tangents(&cube, &normals);
        assert_eq!(tangents.len(), cube.num_corners() as usize);
        for (n, t) in normals.iter().zip(&tangents) {
            let tv = Vec3::new(t[0], t[1], t[2]);
            assert!((tv.length() - 1.0).abs() < 1e-4, "unit tangent: {tv:?}");
            assert!(
                n.dot(tv).abs() < 1e-4,
                "orthogonal to normal: {n:?} vs {tv:?}"
            );
            assert!(t[3] == 1.0 || t[3] == -1.0, "handedness sign: {}", t[3]);
        }
    }
}
