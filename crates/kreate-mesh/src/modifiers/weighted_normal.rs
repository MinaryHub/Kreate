//! Weighted Normals modifier: face-area and corner-angle weighted split
//! normals stored into `names::CUSTOM_NORMAL`.
//!
//! Hard-surface game models with bevelled edges shade flat across large faces
//! while curving smoothly over narrow bevel strips when each corner normal is
//! weighted by `area^weight` (Blender's Weighted Normal modifier). Writing the
//! result into `names::CUSTOM_NORMAL` lets both the viewport and glTF/USD
//! exporters preserve the exact shading.

use std::sync::Arc;

use glam::Vec3;

use crate::attribute::{names, AttrData, Domain};
use crate::MeshData;

/// How strongly larger faces dominate smaller bevels at shared vertices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedNormalParams {
    /// Area weighting factor (Blender's `weight` 1..=100; default 50).
    /// Higher values make large planar faces dominate thin bevel strips.
    pub weight: f32,
    /// Angle threshold in radians above which an edge splits smoothing groups
    /// when `keep_sharp` is enabled (default 60°).
    pub threshold: f32,
    /// Respect authored `SHARP_EDGE` marks and angle breaks.
    pub keep_sharp: bool,
    /// Weight corners by interior face angle in addition to face area.
    pub face_angle: bool,
}

impl Default for WeightedNormalParams {
    fn default() -> Self {
        Self {
            weight: 50.0,
            threshold: 60f32.to_radians(),
            keep_sharp: true,
            face_angle: true,
        }
    }
}

/// Computes face-area (and optionally corner-angle) weighted corner normals
/// and stores them in `names::CUSTOM_NORMAL`.
pub fn weighted_normal(mesh: &MeshData, params: &WeightedNormalParams) -> MeshData {
    let mut out = mesh.clone();
    let num_corners = out.num_corners() as usize;
    let num_faces = out.num_faces() as usize;
    if num_corners == 0 || num_faces == 0 {
        return out;
    }

    let positions = out.positions();
    let corner_verts = out.corner_verts();
    let corner_edges = out.corner_edges();
    let sharp = out.attributes().get_bool(names::SHARP_EDGE);

    // Precompute face normals (unit) and face areas.
    let raw_face_normals = out.compute_face_normals();
    let mut face_unit = Vec::with_capacity(num_faces);
    let mut face_area = Vec::with_capacity(num_faces);
    for n in &raw_face_normals {
        let len = n.length();
        if len > 1e-12 {
            face_unit.push(*n / len);
            face_area.push(len * 0.5);
        } else {
            face_unit.push(Vec3::Z);
            face_area.push(0.0);
        }
    }

    // Map weight [1, 100] -> exponent [0.25, 4.0] so 50 -> ~2.0 (area^2),
    // strongly favouring large planar panels over thin bevel strips.
    let exp = (params.weight.clamp(1.0, 100.0) / 25.0).clamp(0.25, 4.0);
    let cos_thresh = params.threshold.clamp(0.0, std::f32::consts::PI).cos();

    // Map each corner to its face and compute its interior corner angle.
    let mut corner_face = vec![0u32; num_corners];
    let mut corner_angle = vec![1.0f32; num_corners];
    let mut vert_corners: Vec<Vec<usize>> = vec![Vec::new(); positions.len()];
    let mut edge_sharp_at_vert: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
    if params.keep_sharp {
        if let Some(sharp_flags) = sharp {
            for (e_idx, [a, b]) in out.edges().iter().enumerate() {
                if sharp_flags.get(e_idx).copied().unwrap_or(false) {
                    if (*a as usize) < edge_sharp_at_vert.len() {
                        edge_sharp_at_vert[*a as usize].push(e_idx as u32);
                    }
                    if (*b as usize) < edge_sharp_at_vert.len() {
                        edge_sharp_at_vert[*b as usize].push(e_idx as u32);
                    }
                }
            }
        }
    }

    for f in 0..num_faces {
        let range = out.face_corner_range(f as u32);
        let len = range.len();
        for (i, c) in range.clone().enumerate() {
            corner_face[c] = f as u32;
            let v = corner_verts[c] as usize;
            if v < vert_corners.len() {
                vert_corners[v].push(c);
            }
            if params.face_angle && len >= 3 {
                let c_prev = range.start + (i + len - 1) % len;
                let c_next = range.start + (i + 1) % len;
                let p = positions[v];
                let a = (positions[corner_verts[c_prev] as usize] - p).normalize_or_zero();
                let b = (positions[corner_verts[c_next] as usize] - p).normalize_or_zero();
                let angle = a.dot(b).clamp(-1.0, 1.0).acos();
                corner_angle[c] = angle.max(1e-4);
            }
        }
    }

    // Build per-edge face adjacency so sharp edges cleanly block smoothing
    // between their two adjacent faces.
    let mut edge_faces: Vec<[u32; 2]> = vec![[u32::MAX; 2]; out.edges().len()];
    for f in 0..num_faces {
        let range = out.face_corner_range(f as u32);
        for c in range {
            let e = corner_edges[c] as usize;
            if e < edge_faces.len() {
                if edge_faces[e][0] == u32::MAX {
                    edge_faces[e][0] = f as u32;
                } else if edge_faces[e][1] == u32::MAX {
                    edge_faces[e][1] = f as u32;
                }
            }
        }
    }

    let mut custom = vec![Vec3::Z; num_corners];
    for c in 0..num_corners {
        let v = corner_verts[c] as usize;
        let f0 = corner_face[c] as usize;
        let n0 = face_unit[f0];

        let mut sum = Vec3::ZERO;
        for &other_c in &vert_corners[v] {
            let fo = corner_face[other_c] as usize;
            let no = face_unit[fo];
            if params.keep_sharp && fo != f0 {
                if n0.dot(no) < cos_thresh {
                    continue;
                }
                // If an explicitly sharp edge separates f0 and fo, split them.
                let split_by_sharp = edge_sharp_at_vert[v].iter().any(|&e| {
                    let pair = edge_faces[e as usize];
                    (pair[0] == f0 as u32 && pair[1] == fo as u32)
                        || (pair[1] == f0 as u32 && pair[0] == fo as u32)
                });
                if split_by_sharp {
                    continue;
                }
            }
            let w_area = face_area[fo].max(1e-12).powf(exp);
            let w = if params.face_angle {
                w_area * corner_angle[other_c]
            } else {
                w_area
            };
            sum += no * w;
        }
        custom[c] = if sum.length_squared() > 1e-18 {
            sum.normalize()
        } else {
            n0
        };
    }

    out.attributes_mut().insert(
        names::CUSTOM_NORMAL,
        Domain::Corner,
        AttrData::F32x3(Arc::new(custom)),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modifiers::bevel::{bevel, BevelParams};
    use crate::primitives;

    #[test]
    fn weighted_normal_keeps_large_bevelled_cube_faces_flat() {
        let cube = primitives::cube(2.0);
        let bev = bevel(
            &cube,
            &BevelParams {
                width: 0.05,
                segments: 1,
                angle: 30f32.to_radians(),
            },
        );
        let wn = weighted_normal(&bev, &WeightedNormalParams::default());
        let normals = wn
            .attributes()
            .get_f32x3(names::CUSTOM_NORMAL)
            .expect("custom normals written");
        assert_eq!(normals.len(), wn.num_corners() as usize);

        // Find the largest face (one of the 6 main cube panels) and verify its
        // corner normals stay nearly aligned with its face normal (> 0.99)
        // despite sharing vertices with narrow bevel faces.
        let face_normals = wn.compute_face_normals();
        let (big_face, big_n) = face_normals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.length_squared().total_cmp(&b.1.length_squared()))
            .unwrap();
        let unit_big = big_n.normalize();
        for c in wn.face_corner_range(big_face as u32) {
            let dot = normals[c].dot(unit_big);
            assert!(
                dot > 0.99,
                "weighted normal on large panel drifted from flat face: dot={dot}"
            );
        }
    }
}
