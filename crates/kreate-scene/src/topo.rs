//! Persistent topological naming (`FaceRef`, `EdgeRef`) for parametric CAD.
//!
//! Instead of binding fillets, chamfers, shells, or face-sketches to brittle
//! integer face/edge indices that shift whenever an upstream feature changes
//! topology, callers record a [`FaceRef`] or [`EdgeRef`] capturing the
//! entity's birth feature order and geometric invariants (normal/tangent,
//! surface/curve kind, centroid/midpoint, and area/length). On rebuild,
//! [`FaceRef::resolve`] and [`EdgeRef::resolve`] match against either OCCT
//! B-rep signatures or tessellated `MeshData` signatures.

use kreate_math::DVec3;
use kreate_mesh::MeshData;
use kreate_occt::{EdgeSignature, FaceSignature};

/// Persistent reference to a B-rep or mesh face across parametric rebuilds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceRef {
    /// Timeline order of the feature that birthed this face (0 if unknown).
    pub feature_order: u64,
    /// Outward unit normal in body space.
    pub normal: [f64; 3],
    /// Centroid in body space.
    pub center: [f64; 3],
    /// Reference surface area in square metres.
    pub area: f64,
    /// Surface classification (`0` = plane, `1` = cylinder, `2` = cone, `3` = sphere, `4` = torus, `255` = mesh).
    pub surface_kind: u8,
}

/// Persistent reference to a B-rep or mesh edge across parametric rebuilds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeRef {
    /// Timeline order of the feature that birthed this edge (0 if unknown).
    pub feature_order: u64,
    pub start: [f64; 3],
    pub mid: [f64; 3],
    pub end: [f64; 3],
    pub length: f64,
    /// Curve classification (`0` = line, `1` = circle/arc, `2` = ellipse, ..., `255` = mesh).
    pub curve_kind: u8,
}

impl FaceRef {
    pub fn from_signature(feature_order: u64, sig: FaceSignature) -> Self {
        Self {
            feature_order,
            normal: sig.normal,
            center: sig.center,
            area: sig.area,
            surface_kind: sig.surface_kind,
        }
    }

    /// Resolves this persistent reference against a freshly rebuilt set of
    /// face signatures, returning the best matching face.
    ///
    /// Prioritizes matching surface kind and outward normal alignment so a face
    /// shifted along its own normal (e.g. changing an extrude depth) resolves
    /// reliably even when other faces are added or reordered.
    pub fn resolve(&self, candidates: &[FaceSignature]) -> Option<FaceSignature> {
        if candidates.is_empty() {
            return None;
        }
        let want_n = DVec3::from_array(self.normal).normalize_or_zero();
        let want_c = DVec3::from_array(self.center);
        let scale = self.area.abs().sqrt().max(1e-3);

        let mut best: Option<(f64, FaceSignature)> = None;
        for &cand in candidates {
            let n = DVec3::from_array(cand.normal).normalize_or_zero();
            let c = DVec3::from_array(cand.center);
            let normal_err = (1.0 - want_n.dot(n)).clamp(0.0, 2.0);
            // Penalize perpendicular drift far more than drift along the face
            // normal (changing extrude depth translates a top face along `want_n`).
            let dc = c - want_c;
            let axial = dc.dot(want_n).abs();
            let lateral = (dc - want_n * dc.dot(want_n)).length();
            let dist_err = (lateral * 3.0 + axial * 0.35) / scale;
            let area_ratio = if self.area > 1e-12 && cand.area > 1e-12 {
                (self.area / cand.area).ln().abs()
            } else {
                0.0
            };
            let kind_penalty = if self.surface_kind == 255
                || cand.surface_kind == 255
                || self.surface_kind == cand.surface_kind
            {
                0.0
            } else {
                5.0
            };
            let score = kind_penalty + normal_err * 8.0 + dist_err + area_ratio * 0.5;
            if best.is_none_or(|(b, _)| score < b) {
                best = Some((score, cand));
            }
        }
        best.map(|(_, sig)| sig)
    }
}

impl EdgeRef {
    pub fn from_signature(feature_order: u64, sig: EdgeSignature) -> Self {
        Self {
            feature_order,
            start: sig.start,
            mid: sig.mid,
            end: sig.end,
            length: sig.length,
            curve_kind: sig.curve_kind,
        }
    }

    /// Resolves this persistent edge reference against rebuilt edge signatures.
    pub fn resolve(&self, candidates: &[EdgeSignature]) -> Option<EdgeSignature> {
        if candidates.is_empty() {
            return None;
        }
        let want_a = DVec3::from_array(self.start);
        let want_m = DVec3::from_array(self.mid);
        let want_b = DVec3::from_array(self.end);
        let want_dir = (want_b - want_a).normalize_or_zero();
        let scale = self.length.max(1e-3);

        let mut best: Option<(f64, EdgeSignature)> = None;
        for &cand in candidates {
            let a = DVec3::from_array(cand.start);
            let m = DVec3::from_array(cand.mid);
            let b = DVec3::from_array(cand.end);
            let dir = (b - a).normalize_or_zero();
            let dir_err = if want_dir != DVec3::ZERO && dir != DVec3::ZERO {
                1.0 - want_dir.dot(dir).abs()
            } else {
                0.0
            };
            let mid_err = m.distance(want_m) / scale;
            let len_err = if self.length > 1e-12 && cand.length > 1e-12 {
                (self.length / cand.length).ln().abs()
            } else {
                0.0
            };
            let kind_penalty = if self.curve_kind == 255
                || cand.curve_kind == 255
                || self.curve_kind == cand.curve_kind
            {
                0.0
            } else {
                4.0
            };
            let score = kind_penalty + dir_err * 6.0 + mid_err * 2.0 + len_err * 0.75;
            if best.is_none_or(|(s, _)| score < s) {
                best = Some((score, cand));
            }
        }
        best.map(|(_, sig)| sig)
    }
}

/// Extracts face signatures directly from a tessellated `MeshData` when the
/// OCCT B-rep kernel is not active.
pub fn face_signatures_from_mesh(mesh: &MeshData) -> Vec<FaceSignature> {
    let pos = mesh.positions();
    let mut out = Vec::with_capacity(mesh.num_faces() as usize);
    for f in 0..mesh.num_faces() {
        let verts = mesh.face_verts(f);
        if verts.len() < 3 {
            continue;
        }
        let mut center = DVec3::ZERO;
        for &v in verts {
            center += pos[v as usize].as_dvec3();
        }
        center /= verts.len() as f64;
        let p0 = pos[verts[0] as usize].as_dvec3();
        let mut area_vec = DVec3::ZERO;
        for i in 1..verts.len() - 1 {
            let a = pos[verts[i] as usize].as_dvec3() - p0;
            let b = pos[verts[i + 1] as usize].as_dvec3() - p0;
            area_vec += a.cross(b);
        }
        let area = 0.5 * area_vec.length();
        let normal = area_vec.normalize_or_zero();
        out.push(FaceSignature {
            index: f,
            center: center.to_array(),
            normal: normal.to_array(),
            area,
            surface_kind: 255,
        });
    }
    out
}

/// Extracts edge signatures directly from a tessellated `MeshData`.
pub fn edge_signatures_from_mesh(mesh: &MeshData) -> Vec<EdgeSignature> {
    let pos = mesh.positions();
    mesh.edges()
        .iter()
        .enumerate()
        .map(|(i, &[a, b])| {
            let pa = pos[a as usize].as_dvec3();
            let pb = pos[b as usize].as_dvec3();
            let mid = (pa + pb) * 0.5;
            EdgeSignature {
                index: i as u32,
                start: pa.to_array(),
                mid: mid.to_array(),
                end: pb.to_array(),
                length: pa.distance(pb),
                curve_kind: 255,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_ref_tracks_top_face_after_extrude_height_and_topology_change() {
        let box_before = kreate_mesh::primitives::cube(2.0);
        let sigs_before = face_signatures_from_mesh(&box_before);
        let top = sigs_before
            .iter()
            .copied()
            .find(|s| s.normal[2] > 0.9)
            .expect("top face");
        let face_ref = FaceRef::from_signature(1, top);

        // Simulate rebuilding the part with doubled Z height and shifted face ordering.
        let mut taller = kreate_mesh::primitives::cube(2.0);
        for p in taller.positions_mut() {
            if p.z > 0.0 {
                p.z += 3.0;
            }
        }
        let mut sigs_after = face_signatures_from_mesh(&taller);
        sigs_after.reverse();
        let resolved = face_ref.resolve(&sigs_after).expect("resolved");
        assert!(resolved.normal[2] > 0.99);
        assert!((resolved.center[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn edge_ref_tracks_target_edge_when_edge_order_reverses() {
        let cube = kreate_mesh::primitives::cube(2.0);
        let sigs = edge_signatures_from_mesh(&cube);
        let picked = sigs[3];
        let eref = EdgeRef::from_signature(1, picked);
        let mut shuffled = sigs.clone();
        shuffled.reverse();
        let resolved = eref.resolve(&shuffled).expect("resolved");
        assert_eq!(resolved.index, picked.index);
    }
}
