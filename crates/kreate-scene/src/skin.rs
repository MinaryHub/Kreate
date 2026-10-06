//! Skinning: a mesh following an armature's bones.
//!
//! # Weights are the mesh's
//!
//! How much each vertex follows each bone is a point attribute on the mesh,
//! one float layer per bone, named after the bone ([`weight_layer`]) —
//! Blender's vertex groups. On the mesh and not the object, because the
//! weights describe the shape: linked duplicates share them as they share the
//! vertices, and edit mode carries them through every operation the way it
//! carries any other attribute.
//!
//! # Linear blend, normalised
//!
//! Each vertex goes where every bone that weighs on it would put it, averaged
//! by the weights — Blender's Armature modifier with its default settings,
//! weights divided by their total so a vertex whose weights do not sum to one
//! still lands between its bones rather than towards the origin. A vertex no
//! bone weighs on does not move.
//!
//! # Where it runs
//!
//! As the first step of evaluating a bound object, before its modifier stack,
//! which is where Blender users put the Armature modifier. On the CPU for now:
//! the plan's §3-8 puts the per-frame work on the GPU, and this is the
//! reference that pass has to match — the weights and the matrices here are
//! exactly what it will upload.

use kreate_math::{DAffine3, DMat3, DQuat, DVec3};
use kreate_mesh::attribute::{AttrData, Domain};
use kreate_mesh::MeshData;

use crate::armature::{posed, Armature, Pose};

/// The prefix every weight layer's name starts with.
pub const WEIGHT_PREFIX: &str = "weight:";

/// The point attribute holding how much each vertex follows `bone`.
pub fn weight_layer(bone: &str) -> String {
    format!("{WEIGHT_PREFIX}{bone}")
}

/// Each bone's skinning matrix, in the mesh's own space: from where a vertex
/// is at rest to where the bone's pose puts it.
///
/// `mesh_to_armature` takes the mesh object's space to the armature object's:
/// `W_armature⁻¹ · W_mesh`. The matrix is that, then the bone's rest undone and
/// its pose applied, then back: `M⁻¹ · P · R⁻¹ · M`.
pub fn skin_matrices(
    armature: &Armature,
    pose: &Pose,
    mesh_to_armature: DAffine3,
) -> Vec<DAffine3> {
    let back = mesh_to_armature.inverse();
    posed(armature, pose)
        .iter()
        .zip(&armature.bones)
        .map(|(p, bone)| back * *p * bone.rest().inverse() * mesh_to_armature)
        .collect()
}

/// `mesh` with its vertices where the bones put them, or `None` when no bone
/// of `armature` weighs on any vertex — then the mesh is its own answer and
/// nothing needs copying.
pub fn skin(mesh: &MeshData, armature: &Armature, matrices: &[DAffine3]) -> Option<MeshData> {
    skin_with(mesh, armature, matrices, false)
}

/// A bone matrix taken apart for dual-quaternion skinning: the rotation and
/// translation as a unit dual quaternion, and what is left — scale and any
/// shear — as a matrix applied first. `M = T · R · S` exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualBone {
    pub real: DQuat,
    pub dual: DQuat,
    pub rest: DMat3,
}

impl DualBone {
    pub fn of(m: &DAffine3) -> Self {
        let (_, r, t) = m.to_scale_rotation_translation();
        let r = r.normalize();
        let dual = DQuat::from_xyzw(t.x, t.y, t.z, 0.0) * r * 0.5;
        Self {
            real: r,
            dual,
            rest: DMat3::from_quat(r.inverse()) * m.matrix3,
        }
    }
}

/// Where the blend of `bones` (weight, bone) puts `p`: Kavan's dual
/// quaternion skinning with the scales blended linearly first. A joint bent
/// far keeps its volume where the linear blend pinches it to a candy wrapper.
pub fn dual_blend(p: DVec3, bones: &[(f64, DualBone)]) -> DVec3 {
    let Some((_, first)) = bones.first() else {
        return p;
    };
    let mut real = DQuat::from_xyzw(0.0, 0.0, 0.0, 0.0);
    let mut dual = DQuat::from_xyzw(0.0, 0.0, 0.0, 0.0);
    let mut rest = DMat3::ZERO;
    let mut total = 0.0;
    for (w, b) in bones {
        // The shorter way round: q and -q are one turn, and adding them cancels.
        let s = if b.real.dot(first.real) < 0.0 { -w } else { *w };
        real += b.real * s;
        dual += b.dual * s;
        rest += b.rest * *w;
        total += w;
    }
    let length = real.length();
    if length < 1e-12 || total <= 0.0 {
        return p;
    }
    let (real, dual) = (real * (1.0 / length), dual * (1.0 / length));
    let q = real.xyz();
    let d = dual.xyz();
    let t = 2.0 * (real.w * d - dual.w * q + q.cross(d));
    real * ((rest * (1.0 / total)) * p) + t
}

/// [`skin`], by dual quaternions when `preserve_volume` is set — Blender's
/// Preserve Volume on the Armature modifier.
pub fn skin_with(
    mesh: &MeshData,
    armature: &Armature,
    matrices: &[DAffine3],
    preserve_volume: bool,
) -> Option<MeshData> {
    let n = mesh.num_verts() as usize;
    // Each bone's weights, if the mesh has a layer for it of the right length.
    let layers: Vec<(usize, &[f32])> = armature
        .bones
        .iter()
        .enumerate()
        .filter_map(|(i, bone)| {
            let layer = mesh.attributes().get(&weight_layer(&bone.name))?;
            match (&layer.domain, &layer.data) {
                (Domain::Point, AttrData::F32(w)) if w.len() == n => Some((i, w.as_slice())),
                _ => None,
            }
        })
        .collect();
    if layers.is_empty() {
        return None;
    }
    let mut out = mesh.clone();
    let positions = out.positions_mut();
    if preserve_volume {
        let duals: Vec<DualBone> = matrices.iter().map(DualBone::of).collect();
        let mut on = Vec::new();
        for (v, p) in positions.iter_mut().enumerate() {
            on.clear();
            for (bone, weights) in &layers {
                let w = weights[v] as f64;
                if w > 0.0 {
                    if let Some(d) = duals.get(*bone) {
                        on.push((w, *d));
                    }
                }
            }
            if !on.is_empty() {
                let at = dual_blend(p.as_dvec3(), &on);
                *p = glam::Vec3::new(at.x as f32, at.y as f32, at.z as f32);
            }
        }
        return Some(out);
    }
    for (v, p) in positions.iter_mut().enumerate() {
        let rest = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
        let mut sum = DVec3::ZERO;
        let mut total = 0.0f64;
        for (bone, weights) in &layers {
            let w = weights[v] as f64;
            if w > 0.0 {
                if let Some(m) = matrices.get(*bone) {
                    sum += m.transform_point3(rest) * w;
                    total += w;
                }
            }
        }
        if total > 0.0 {
            let at = sum / total;
            *p = glam::Vec3::new(at.x as f32, at.y as f32, at.z as f32);
        }
    }
    Some(out)
}

/// The influences [`skin`] reads, as tables: for each vertex, the bones with a
/// positive weight on it and those weights, CSR over the vertices. What a GPU
/// skinning pass uploads once, so it can do per pose exactly what [`skin`]
/// does. `None` when no bone weighs on the mesh, as for [`skin`].
pub fn influences(mesh: &MeshData, armature: &Armature) -> Option<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    let n = mesh.num_verts() as usize;
    let layers: Vec<(u32, &[f32])> = armature
        .bones
        .iter()
        .enumerate()
        .filter_map(|(i, bone)| {
            let layer = mesh.attributes().get(&weight_layer(&bone.name))?;
            match (&layer.domain, &layer.data) {
                (Domain::Point, AttrData::F32(w)) if w.len() == n => Some((i as u32, w.as_slice())),
                _ => None,
            }
        })
        .collect();
    if layers.is_empty() {
        return None;
    }
    let mut offsets = Vec::with_capacity(n + 1);
    let mut bones = Vec::new();
    let mut weights = Vec::new();
    offsets.push(0);
    for v in 0..n {
        for (bone, w) in &layers {
            if w[v] > 0.0 {
                bones.push(*bone);
                weights.push(w[v]);
            }
        }
        offsets.push(bones.len() as u32);
    }
    Some((offsets, bones, weights))
}

/// Weights from distance: each vertex follows the bones nearest it.
///
/// Blender's "With Automatic Weights" solves heat diffusion over the surface;
/// this is the envelope it falls back to — the inverse fourth power of the
/// distance to each bone's segment, so the nearest bone dominates and a
/// vertex midway between two shares them, kept to the four nearest and
/// normalised. It is a starting point for weight painting, which is what
/// Blender's is too.
///
/// `mesh_to_armature` as for [`skin_matrices`]. One layer per bone, every
/// bone's, so a bone that ends up with no weight still has a group to paint.
pub fn automatic_weights(
    mesh: &MeshData,
    armature: &Armature,
    mesh_to_armature: DAffine3,
) -> Vec<(String, Vec<f32>)> {
    let segments: Vec<(DVec3, DVec3)> = armature.bones.iter().map(|b| (b.head, b.tail)).collect();
    let mut layers: Vec<Vec<f32>> = vec![vec![0.0; mesh.num_verts() as usize]; segments.len()];
    for (v, p) in mesh.positions().iter().enumerate() {
        let at = mesh_to_armature.transform_point3(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
        let mut near: Vec<(usize, f64)> = segments
            .iter()
            .enumerate()
            .map(|(i, (a, b))| (i, distance_to_segment(at, *a, *b)))
            .collect();
        near.sort_by(|x, y| x.1.total_cmp(&y.1));
        near.truncate(4);
        let raw: Vec<(usize, f64)> = near
            .iter()
            .map(|&(i, d)| (i, 1.0 / d.max(1e-4).powi(4)))
            .collect();
        let total: f64 = raw.iter().map(|(_, w)| w).sum();
        if total > 0.0 {
            for (i, w) in raw {
                layers[i][v] = (w / total) as f32;
            }
        }
    }
    armature
        .bones
        .iter()
        .map(|b| weight_layer(&b.name))
        .zip(layers)
        .collect()
}

fn distance_to_segment(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = if ab.length_squared() > 0.0 {
        ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (a + ab * t - p).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::armature::Bone;
    use kreate_math::DQuat;

    /// Two bones standing on each other, and a column of vertices beside them.
    fn arm() -> Armature {
        let mut a = Armature::new("Arm");
        a.bones[0].name = "Upper".into();
        a.add_bone(Bone {
            name: "Lower".into(),
            parent: Some(0),
            head: DVec3::Z,
            tail: DVec3::new(0.0, 0.0, 2.0),
            connected: true,
            ..Default::default()
        });
        a
    }

    fn column() -> MeshData {
        let mut m = kreate_mesh::primitives::cube(0.2);
        // Stretch the cube into a column two units tall, from z = 0.
        for p in m.positions_mut() {
            p.z = (p.z + 0.1) * 10.0;
        }
        m
    }

    #[test]
    fn dual_quaternions_agree_with_the_linear_blend_for_one_bone_and_keep_volume_for_two() {
        // A column with rings all the way up, so the joint has vertices.
        let mut m = kreate_mesh::primitives::uv_sphere(0.5, 16, 24);
        for p in m.positions_mut() {
            p.x *= 0.4;
            p.y *= 0.4;
            p.z = (p.z + 0.5) * 2.0;
        }
        let a = arm();
        let mut weighted = m.clone();
        for (name, w) in automatic_weights(&m, &a, DAffine3::IDENTITY) {
            weighted
                .attributes_mut()
                .insert(&name, Domain::Point, AttrData::F32(std::sync::Arc::new(w)));
        }
        // A twist and a scale on the lower bone: one bone alone is the same
        // matrix either way.
        let mut pose = Pose::default();
        pose.entry("Lower").transform.rotation = DQuat::from_rotation_y(2.4);
        pose.entry("Lower").transform.scale = DVec3::new(1.3, 0.8, 1.1);
        let matrices = skin_matrices(&a, &pose, DAffine3::IDENTITY);
        let p = DVec3::new(0.3, -0.2, 1.7);
        let by_dq = dual_blend(p, &[(1.0, DualBone::of(&matrices[1]))]);
        assert!(by_dq.distance(matrices[1].transform_point3(p)) < 1e-9);
        let linear = skin(&weighted, &a, &matrices).unwrap();
        let dual = skin_with(&weighted, &a, &matrices, true).unwrap();
        // At the joint the twist pinches the linear blend towards the bone;
        // the dual blend keeps it out. Measured as the mean distance from the
        // column's axis of the vertices around the joint.
        let spread = |mesh: &MeshData| {
            let near: Vec<f32> = m
                .positions()
                .iter()
                .zip(mesh.positions())
                .filter(|(rest, _)| (rest.z - 1.0).abs() < 0.3)
                .map(|(_, now)| (now.x * now.x + now.y * now.y).sqrt())
                .collect();
            near.iter().sum::<f32>() / near.len() as f32
        };
        let rest = spread(&weighted);
        assert!(spread(&linear) < rest * 0.8, "the linear blend should pinch: {} of {rest}", spread(&linear));
        assert!(spread(&dual) > spread(&linear) * 1.15, "{} vs {}", spread(&dual), spread(&linear));
    }

    #[test]
    fn automatic_weights_follow_the_nearest_bone_and_sum_to_one() {
        let m = column();
        let weights = automatic_weights(&m, &arm(), DAffine3::IDENTITY);
        assert_eq!(
            weights.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["weight:Upper", "weight:Lower"]
        );
        for (v, p) in m.positions().iter().enumerate() {
            let (up, low) = (weights[0].1[v], weights[1].1[v]);
            assert!(((up + low) - 1.0).abs() < 1e-5);
            if p.z < 0.5 {
                assert!(
                    up > 0.9,
                    "a vertex at the bottom follows the upper bone: {up}"
                );
            }
            if p.z > 1.5 {
                assert!(
                    low > 0.9,
                    "a vertex at the top follows the lower bone: {low}"
                );
            }
        }
    }

    #[test]
    fn bending_the_lower_bone_bends_the_top_of_the_column_and_leaves_the_bottom() {
        let mut m = column();
        for (name, w) in automatic_weights(&m, &arm(), DAffine3::IDENTITY) {
            m.attributes_mut()
                .insert(&name, Domain::Point, AttrData::F32(std::sync::Arc::new(w)));
        }
        let a = arm();
        let mut pose = Pose::default();
        // A quarter turn about the lower bone's own X, at the elbow.
        pose.entry("Lower").transform.rotation =
            DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
        let bent = skin(&m, &a, &skin_matrices(&a, &pose, DAffine3::IDENTITY))
            .expect("the mesh is weighted");
        for (rest, now) in m.positions().iter().zip(bent.positions()) {
            if rest.z < 0.5 {
                assert!(
                    (*rest - *now).length() < 0.05,
                    "the bottom moved: {rest:?} -> {now:?}"
                );
            }
            if rest.z > 1.9 {
                // Turned about the elbow: the top swings out sideways and down.
                assert!(
                    now.z < 1.2,
                    "the top did not bend over: {rest:?} -> {now:?}"
                );
            }
        }
    }

    #[test]
    fn at_rest_nothing_moves_and_an_unweighted_mesh_is_left_alone() {
        let mut m = column();
        let a = arm();
        assert!(skin(
            &m,
            &a,
            &skin_matrices(&a, &Pose::default(), DAffine3::IDENTITY)
        )
        .is_none());
        for (name, w) in automatic_weights(&m, &a, DAffine3::IDENTITY) {
            m.attributes_mut()
                .insert(&name, Domain::Point, AttrData::F32(std::sync::Arc::new(w)));
        }
        let still = skin(
            &m,
            &a,
            &skin_matrices(&a, &Pose::default(), DAffine3::IDENTITY),
        )
        .unwrap();
        for (a, b) in m.positions().iter().zip(still.positions()) {
            assert!((*a - *b).length() < 1e-5);
        }
    }

    #[test]
    fn the_matrices_carry_the_objects_placement() {
        // The armature a metre to the side of the mesh: at rest the skin must
        // still be the identity, and a pose must turn about the bone where it
        // really is.
        let a = arm();
        let mesh_to_armature = DAffine3::from_translation(DVec3::new(-1.0, 0.0, 0.0));
        let rest = skin_matrices(&a, &Pose::default(), mesh_to_armature);
        for m in &rest {
            assert!(m.abs_diff_eq(DAffine3::IDENTITY, 1e-12));
        }
        let mut pose = Pose::default();
        pose.entry("Upper").transform.rotation = DQuat::from_rotation_x(std::f64::consts::PI);
        let turned = skin_matrices(&a, &pose, mesh_to_armature)[0];
        // The bone's head, in the mesh's space, is at x = 1: it stays put.
        assert!((turned.transform_point3(DVec3::X) - DVec3::X).length() < 1e-9);
    }
}

#[cfg(test)]
mod cost {
    use super::*;

    /// How long skinning a large mesh takes, for the plan's numbers. Ignored:
    /// a timing is a measurement, not a test.
    #[test]
    #[ignore]
    fn skinning_cost() {
        let mut m = kreate_mesh::primitives::grid(2.0, 447, 447);
        let n = m.num_verts() as usize;
        let a = {
            let mut a = Armature::new("Arm");
            a.add_bone(crate::armature::Bone {
                name: "B".into(),
                parent: Some(0),
                head: DVec3::Z,
                tail: DVec3::Z * 2.0,
                connected: true,
                ..Default::default()
            });
            a
        };
        for (name, w) in automatic_weights(&m, &a, DAffine3::IDENTITY) {
            m.attributes_mut()
                .insert(&name, Domain::Point, AttrData::F32(std::sync::Arc::new(w)));
        }
        let mut pose = Pose::default();
        pose.entry("B").transform.rotation = kreate_math::DQuat::from_rotation_x(0.5);
        let matrices = skin_matrices(&a, &pose, DAffine3::IDENTITY);
        let start = std::time::Instant::now();
        let runs = 10;
        for _ in 0..runs {
            std::hint::black_box(skin(&m, &a, &matrices));
        }
        eprintln!(
            "skin {n} vertices: {:.2} ms",
            start.elapsed().as_secs_f64() * 1000.0 / runs as f64
        );
    }
}
