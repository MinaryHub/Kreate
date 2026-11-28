//! Multi-constraint 3D assembly mate solver, non-destructive exploded states,
//! and indented Bill of Materials (BOM) generation.

use std::collections::{BTreeMap, HashMap};

use kreate_math::{DAffine3, DQuat, DVec3};
use kreate_reflect::Uid;

use crate::document::{Document, ObjectId};

/// Geometric constraint between two assembly parts (`body_a` and `body_b`).
#[derive(Clone, Debug, PartialEq)]
pub enum AssemblyMateKind {
    /// Planar contact: face point `anchor_a` / normal `axis_a` on `body_a`
    /// lies at `offset` metres along `axis_b` from `anchor_b` on `body_b`,
    /// with normals anti-parallel (`flip = true`, face-to-face) or parallel
    /// (`flip = false`, flush).
    Planar { offset: f64, flip: bool },
    /// Concentric axes: line `(anchor_a, axis_a)` aligns coaxially with
    /// `(anchor_b, axis_b)`. When `axial_offset` is `Some(d)`, also pins the
    /// axial distance along `axis_b` to `d`.
    Concentric { axial_offset: Option<f64> },
    /// Pins the Euclidean distance between `anchor_a` and `anchor_b`.
    Distance { distance: f64 },
    /// Keeps unit directions `axis_a` and `axis_b` parallel.
    Parallel,
    /// Keeps unit directions `axis_a` and `axis_b` perpendicular (90°).
    Perpendicular,
    /// Constrains the angle between `axis_a` and `axis_b` to `radians`.
    Angle { radians: f64 },
}

/// One assembly mate linking two bodies in their respective local frames.
#[derive(Clone, Debug, PartialEq)]
pub struct AssemblyMate {
    pub body_a: Uid,
    pub anchor_a: DVec3,
    pub axis_a: DVec3,
    pub body_b: Uid,
    pub anchor_b: DVec3,
    pub axis_b: DVec3,
    pub kind: AssemblyMateKind,
}

/// Rigid pose (position + orientation) of a body in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidPose {
    pub translation: DVec3,
    pub rotation: DQuat,
}

impl RigidPose {
    pub fn transform_point(&self, p: DVec3) -> DVec3 {
        self.translation + self.rotation * p
    }
    pub fn transform_dir(&self, d: DVec3) -> DVec3 {
        (self.rotation * d).normalize_or_zero()
    }
}

/// Outcome of the multi-constraint 3D rigid-body assembly solver.
#[derive(Clone, Debug)]
pub struct SolveReport {
    pub poses: HashMap<Uid, RigidPose>,
    pub iterations: usize,
    pub max_residual: f64,
    pub converged: bool,
}

/// Evaluates scalar residuals for all `mates` under `poses`.
fn eval_residuals(mates: &[AssemblyMate], poses: &HashMap<Uid, RigidPose>) -> Vec<f64> {
    let mut r = Vec::with_capacity(mates.len() * 4);
    for m in mates {
        let (Some(pa), Some(pb)) = (poses.get(&m.body_a), poses.get(&m.body_b)) else {
            continue;
        };
        let wa = pa.transform_point(m.anchor_a);
        let wb = pb.transform_point(m.anchor_b);
        let na = pa.transform_dir(m.axis_a);
        let nb = pb.transform_dir(m.axis_b);
        let diff = wa - wb;
        match m.kind {
            AssemblyMateKind::Planar { offset, flip } => {
                // Distance from anchor_a to plane of B at `offset` along nb.
                r.push(diff.dot(nb) - offset);
                // Align normals: na == -nb if flip else na == nb.
                let target = if flip { -nb } else { nb };
                let cross = na.cross(target);
                r.push(cross.x);
                r.push(cross.y);
                r.push(cross.z);
                // Keep sign consistent when nearly anti-aligned.
                let dot_err = (1.0 - na.dot(target)).max(0.0);
                if dot_err > 1.5 {
                    r.push(dot_err);
                } else {
                    r.push(0.0);
                }
            }
            AssemblyMateKind::Concentric { axial_offset } => {
                // Radial displacement perpendicular to nb must be zero.
                let radial = diff - nb * diff.dot(nb);
                r.push(radial.x);
                r.push(radial.y);
                r.push(radial.z);
                // Axes must be parallel (cross product == 0).
                let cross = na.cross(nb);
                r.push(cross.x);
                r.push(cross.y);
                r.push(cross.z);
                if let Some(ax) = axial_offset {
                    r.push(diff.dot(nb) - ax);
                }
            }
            AssemblyMateKind::Distance { distance } => {
                r.push(diff.length() - distance);
            }
            AssemblyMateKind::Parallel => {
                let cross = na.cross(nb);
                r.push(cross.x);
                r.push(cross.y);
                r.push(cross.z);
            }
            AssemblyMateKind::Perpendicular => {
                r.push(na.dot(nb));
            }
            AssemblyMateKind::Angle { radians } => {
                r.push(na.dot(nb) - radians.cos());
            }
        }
    }
    r
}

/// Perturbs a single scalar DOF `dof` (0..3 translation, 3..6 small-angle
/// rotation on `SO(3)`) of `pose` by `eps`.
fn perturb(pose: RigidPose, dof: usize, eps: f64) -> RigidPose {
    match dof {
        0 => RigidPose {
            translation: pose.translation + DVec3::new(eps, 0.0, 0.0),
            rotation: pose.rotation,
        },
        1 => RigidPose {
            translation: pose.translation + DVec3::new(0.0, eps, 0.0),
            rotation: pose.rotation,
        },
        2 => RigidPose {
            translation: pose.translation + DVec3::new(0.0, 0.0, eps),
            rotation: pose.rotation,
        },
        3 => RigidPose {
            translation: pose.translation,
            rotation: (DQuat::from_rotation_x(eps) * pose.rotation).normalize(),
        },
        4 => RigidPose {
            translation: pose.translation,
            rotation: (DQuat::from_rotation_y(eps) * pose.rotation).normalize(),
        },
        _ => RigidPose {
            translation: pose.translation,
            rotation: (DQuat::from_rotation_z(eps) * pose.rotation).normalize(),
        },
    }
}

/// Solves a dense symmetric positive-definite linear system `A x = b` via
/// Gaussian elimination with partial pivoting.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for (row, rvec) in a.iter().enumerate().skip(col + 1) {
            if rvec[col].abs() > best {
                best = rvec[col].abs();
                pivot = row;
            }
        }
        if best < 1e-15 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let diag = a[col][col];
        for row in (col + 1)..n {
            let f = a[row][col] / diag;
            if f.abs() < 1e-18 {
                continue;
            }
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in (row + 1)..n {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

/// Solves a multi-constraint 3D assembly simultaneously using damped
/// Levenberg-Marquardt over `SE(3)` poses for all ungrounded bodies.
///
/// Bodies listed in `grounded` stay fixed at their initial poses.
pub fn solve_assembly(
    initial: &HashMap<Uid, RigidPose>,
    grounded: &[Uid],
    mates: &[AssemblyMate],
    max_iters: usize,
) -> SolveReport {
    let mut poses = initial.clone();
    let mut free_bodies: Vec<Uid> = initial
        .keys()
        .copied()
        .filter(|u| !grounded.contains(u))
        .collect();
    free_bodies.sort_by_key(|u| u.raw());
    let n_vars = free_bodies.len() * 6;
    if n_vars == 0 || mates.is_empty() {
        let res = eval_residuals(mates, &poses);
        let max_residual = res.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        return SolveReport {
            poses,
            iterations: 0,
            max_residual,
            converged: max_residual < 1e-6,
        };
    }

    let eps = 1e-6;
    let mut lambda = 1e-3;
    let mut iterations = 0usize;
    let mut r = eval_residuals(mates, &poses);
    let mut cost: f64 = r.iter().map(|v| v * v).sum();

    for iter in 0..max_iters {
        iterations = iter + 1;
        let max_res = r.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        if max_res < 1e-7 {
            return SolveReport {
                poses,
                iterations: iter,
                max_residual: max_res,
                converged: true,
            };
        }
        let m = r.len();
        // Numerical Jacobian J (m x n_vars), stored column-major for fast J^T J.
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n_vars);
        for &uid in &free_bodies {
            let base = poses[&uid];
            for dof in 0..6 {
                poses.insert(uid, perturb(base, dof, eps));
                let r_plus = eval_residuals(mates, &poses);
                let col: Vec<f64> = r_plus
                    .iter()
                    .zip(&r)
                    .map(|(rp, r0)| (rp - r0) / eps)
                    .collect();
                cols.push(col);
            }
            poses.insert(uid, base);
        }

        // Normal equations: (J^T J + lambda I) dx = -J^T r
        let mut jtj = vec![vec![0.0; n_vars]; n_vars];
        let mut jtr = vec![0.0; n_vars];
        for i in 0..n_vars {
            let mut rhs = 0.0;
            for k in 0..m {
                rhs -= cols[i][k] * r[k];
            }
            jtr[i] = rhs;
            for j in i..n_vars {
                let mut dot = 0.0;
                for k in 0..m {
                    dot += cols[i][k] * cols[j][k];
                }
                jtj[i][j] = dot;
                jtj[j][i] = dot;
            }
            jtj[i][i] += lambda * jtj[i][i].max(1.0);
        }

        let Some(dx) = solve_linear(jtj, jtr) else {
            lambda *= 10.0;
            continue;
        };

        let mut trial = poses.clone();
        for (bi, &uid) in free_bodies.iter().enumerate() {
            let p = trial[&uid];
            let dt = DVec3::new(dx[bi * 6], dx[bi * 6 + 1], dx[bi * 6 + 2]);
            let dw = DVec3::new(dx[bi * 6 + 3], dx[bi * 6 + 4], dx[bi * 6 + 5]);
            let angle = dw.length();
            let dq = if angle > 1e-12 {
                DQuat::from_axis_angle(dw / angle, angle)
            } else {
                DQuat::IDENTITY
            };
            trial.insert(
                uid,
                RigidPose {
                    translation: p.translation + dt,
                    rotation: (dq * p.rotation).normalize(),
                },
            );
        }

        let r_trial = eval_residuals(mates, &trial);
        let cost_trial: f64 = r_trial.iter().map(|v| v * v).sum();
        if cost_trial < cost {
            poses = trial;
            r = r_trial;
            cost = cost_trial;
            lambda = (lambda * 0.3).max(1e-9);
        } else {
            lambda = (lambda * 5.0).min(1e6);
        }
    }

    let max_residual = r.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    SolveReport {
        poses,
        iterations,
        max_residual,
        converged: max_residual < 1e-5,
    }
}

/// Non-destructive exploded view state: stores per-object displacement vectors
/// without altering the authored `object.transform`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExplodedState {
    /// Active explosion factor in `0.0..=1.0`.
    pub factor: f64,
    /// Per-object world displacement at `factor = 1.0`.
    pub offsets: BTreeMap<Uid, DVec3>,
}

impl ExplodedState {
    /// Generates radial explosion vectors away from the assembly centroid for
    /// every visible mesh object in `doc`.
    pub fn auto_radial(doc: &Document, spread: f64) -> Self {
        let mut centers: Vec<(Uid, DVec3)> = Vec::new();
        for (id, obj) in doc.objects() {
            if !doc.is_shown(id) || obj.mesh().is_none() {
                continue;
            }
            let w = doc.world_transform(id);
            centers.push((obj.uid, w.translation));
        }
        if centers.is_empty() {
            return Self::default();
        }
        let mean = centers.iter().map(|(_, c)| *c).sum::<DVec3>() / centers.len() as f64;
        let mut offsets = BTreeMap::new();
        for (i, (uid, c)) in centers.into_iter().enumerate() {
            let v = c - mean;
            let dir = if v.length() > 1e-6 {
                v.normalize()
            } else {
                // Break exact coincidence deterministically along Z.
                DVec3::new(0.0, 0.0, (i as f64 + 1.0) * 0.25)
            };
            offsets.insert(uid, dir * spread);
        }
        Self {
            factor: 1.0,
            offsets,
        }
    }

    /// Evaluates the non-destructive exploded world transform of `id`.
    pub fn world_transform(&self, doc: &Document, id: ObjectId) -> DAffine3 {
        let mut w = doc.world_transform(id);
        let f = self.factor.clamp(0.0, 1.0);
        if f > 0.0 {
            if let Some(uid) = doc.object_uid(id) {
                if let Some(&delta) = self.offsets.get(&uid) {
                    w.translation += delta * f;
                }
            }
        }
        w
    }
}

/// One row of an indented hierarchical Bill of Materials (BOM).
#[derive(Clone, Debug, PartialEq)]
pub struct BomRow {
    /// Hierarchical item number (e.g. `"1"`, `"1.1"`, `"1.2"`, `"2"`).
    pub item_no: String,
    pub depth: usize,
    pub name: String,
    pub quantity: u32,
    pub material: String,
    /// Bounding box dimensions `[dx, dy, dz]` in millimetres.
    pub size_mm: [f64; 3],
    /// Estimated single-part volume in cubic centimetres.
    pub volume_cm3: f64,
}

/// Builds an indented hierarchical Bill of Materials from the visible scene tree.
pub fn build_bom(doc: &Document) -> Vec<BomRow> {
    fn visit(
        doc: &Document,
        children: &[ObjectId],
        prefix: &str,
        depth: usize,
        out: &mut Vec<BomRow>,
    ) {
        // Group sibling leaf parts that share the same mesh and material.
        let mut idx = 0u32;
        let mut emitted_mesh: HashMap<(Option<Uid>, String), usize> = HashMap::new();
        for &id in children {
            let Some(obj) = doc.object(id) else {
                continue;
            };
            if !obj.visible {
                continue;
            }
            let kids: Vec<ObjectId> = doc
                .objects()
                .filter(|(_, o)| o.parent == Some(id))
                .map(|(cid, _)| cid)
                .collect();
            let mat_name = doc
                .material_at(id, 0)
                .map(|m| m.name.clone())
                .unwrap_or_default();

            if kids.is_empty() {
                let mesh_uid = obj.mesh().and_then(|m| doc.mesh_uid(m));
                if let Some(muid) = mesh_uid {
                    if let Some(&row_idx) = emitted_mesh.get(&(Some(muid), mat_name.clone())) {
                        out[row_idx].quantity += 1;
                        continue;
                    }
                }
                idx += 1;
                let item_no = if prefix.is_empty() {
                    format!("{idx}")
                } else {
                    format!("{prefix}.{idx}")
                };
                let (size_mm, volume_cm3) = obj
                    .mesh()
                    .and_then(|m| doc.mesh(m))
                    .map(|mesh| {
                        let b = mesh.bounds();
                        let s = (b.max - b.min) * 1000.0;
                        let vol_m3 = kreate_mesh::measure::mass_properties(mesh, |p| p)
                            .volume
                            .abs();
                        ([s.x as f64, s.y as f64, s.z as f64], vol_m3 * 1e6)
                    })
                    .unwrap_or(([0.0; 3], 0.0));
                let row_idx = out.len();
                out.push(BomRow {
                    item_no,
                    depth,
                    name: obj.name.clone(),
                    quantity: 1,
                    material: mat_name.clone(),
                    size_mm,
                    volume_cm3,
                });
                if let Some(muid) = mesh_uid {
                    emitted_mesh.insert((Some(muid), mat_name), row_idx);
                }
            } else {
                idx += 1;
                let item_no = if prefix.is_empty() {
                    format!("{idx}")
                } else {
                    format!("{prefix}.{idx}")
                };
                out.push(BomRow {
                    item_no: item_no.clone(),
                    depth,
                    name: obj.name.clone(),
                    quantity: 1,
                    material: mat_name,
                    size_mm: [0.0; 3],
                    volume_cm3: 0.0,
                });
                visit(doc, &kids, &item_no, depth + 1, out);
            }
        }
    }

    let roots: Vec<ObjectId> = doc
        .objects()
        .filter(|(_, o)| o.parent.is_none())
        .map(|(id, _)| id)
        .collect();
    let mut rows = Vec::new();
    visit(doc, &roots, "", 0, &mut rows);
    rows
}

/// Formats an indented BOM as RFC-4180 CSV.
pub fn format_bom_csv(rows: &[BomRow]) -> String {
    let mut csv = String::from("Item,Name,Qty,Material,SizeX_mm,SizeY_mm,SizeZ_mm,Volume_cm3\n");
    for r in rows {
        let name = r.name.replace('"', "\"\"");
        let mat = r.material.replace('"', "\"\"");
        use std::fmt::Write as _;
        let _ = writeln!(
            csv,
            "{},\"{}\",{},\"{}\",{:.2},{:.2},{:.2},{:.3}",
            r.item_no,
            name,
            r.quantity,
            mat,
            r.size_mm[0],
            r.size_mm[1],
            r.size_mm[2],
            r.volume_cm3
        );
    }
    csv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_constraint_solver_aligns_plate_with_two_concentric_bolts_and_planar_mate() {
        let base_uid = Uid::from_raw(1);
        let plate_uid = Uid::from_raw(2);
        let mut init = HashMap::new();
        init.insert(
            base_uid,
            RigidPose {
                translation: DVec3::ZERO,
                rotation: DQuat::IDENTITY,
            },
        );
        // Start plate offset & tilted away from its mated position.
        init.insert(
            plate_uid,
            RigidPose {
                translation: DVec3::new(0.05, -0.04, 0.08),
                rotation: DQuat::from_euler(glam::EulerRot::XYZ, 0.25, -0.2, 0.4),
            },
        );

        let mates = vec![
            // Bottom of plate (z=0, normal -Z) flush against top of base (z=0.01, normal +Z)
            AssemblyMate {
                body_a: plate_uid,
                anchor_a: DVec3::ZERO,
                axis_a: -DVec3::Z,
                body_b: base_uid,
                anchor_b: DVec3::new(0.0, 0.0, 0.01),
                axis_b: DVec3::Z,
                kind: AssemblyMateKind::Planar {
                    offset: 0.0,
                    flip: true,
                },
            },
            // Bolt hole 1 at (0, 0) concentric
            AssemblyMate {
                body_a: plate_uid,
                anchor_a: DVec3::ZERO,
                axis_a: DVec3::Z,
                body_b: base_uid,
                anchor_b: DVec3::ZERO,
                axis_b: DVec3::Z,
                kind: AssemblyMateKind::Concentric { axial_offset: None },
            },
            // Bolt hole 2 at (0.05, 0) concentric (locks Z rotation too!)
            AssemblyMate {
                body_a: plate_uid,
                anchor_a: DVec3::new(0.05, 0.0, 0.0),
                axis_a: DVec3::Z,
                body_b: base_uid,
                anchor_b: DVec3::new(0.05, 0.0, 0.0),
                axis_b: DVec3::Z,
                kind: AssemblyMateKind::Concentric { axial_offset: None },
            },
        ];

        let report = solve_assembly(&init, &[base_uid], &mates, 60);
        assert!(
            report.converged,
            "solver should converge, residual={}",
            report.max_residual
        );
        let p = report.poses[&plate_uid];
        assert!(
            p.translation.distance(DVec3::new(0.0, 0.0, 0.01)) < 1e-5,
            "got {:?}",
            p.translation
        );
        assert!(p.transform_dir(DVec3::X).dot(DVec3::X) > 0.9999);
        assert!(p.transform_dir(DVec3::Z).dot(DVec3::Z) > 0.9999);
    }

    #[test]
    fn exploded_state_and_indented_bom_preserve_scene_transforms() {
        let mut doc = Document::new();
        let sub = doc.add_object(crate::Object::new("Gearbox", crate::ObjectData::Empty));
        let bolt_mesh = doc.add_mesh(kreate_mesh::primitives::cube(0.01));
        let mut b1 = crate::Object::new("M4_Bolt", crate::ObjectData::Mesh(bolt_mesh));
        b1.transform.translation = DVec3::new(0.02, 0.0, 0.0);
        let b1_id = doc.add_object(b1);
        let mut b2 = crate::Object::new("M4_Bolt", crate::ObjectData::Mesh(bolt_mesh));
        b2.transform.translation = DVec3::new(-0.02, 0.0, 0.0);
        let b2_id = doc.add_object(b2);
        doc.set_parent(b1_id, Some(sub)).unwrap();
        doc.set_parent(b2_id, Some(sub)).unwrap();

        let bom = build_bom(&doc);
        assert_eq!(bom.len(), 2);
        assert_eq!(bom[0].item_no, "1");
        assert_eq!(bom[0].name, "Gearbox");
        assert_eq!(bom[1].item_no, "1.1");
        assert_eq!(bom[1].quantity, 2);
        assert!(format_bom_csv(&bom).contains("1.1,\"M4_Bolt\",2"));

        let exploded = ExplodedState::auto_radial(&doc, 0.1);
        let w_b1 = exploded.world_transform(&doc, b1_id);
        assert!((w_b1.translation.x - 0.12).abs() < 1e-6);
        // Authored transform stays untouched.
        assert!((doc.object(b1_id).unwrap().transform.translation.x - 0.02).abs() < 1e-9);
    }
}
