//! A rig, tried: each bone bent in turn, and the skin it moves measured —
//! what an assistant rigging a model needs to know whether it worked, and
//! where it did not, without a person turning every joint.
//!
//! The measure is the one that found Mixamo's inverted bind matrices: how
//! much the mesh's edges stretch. A limb that bends well keeps its edges'
//! lengths; a vertex left on the wrong bone, or on none, tears away from its
//! neighbours. Alongside: vertices no bone moves, and bones that move none.

use kreate_scene::document::ObjectData;
use kreate_scene::modifier::{EvalCtx, ModifierCache};
use kreate_scene::{Document, ObjectId};

/// One bone, bent.
#[derive(Clone, Debug, PartialEq)]
pub struct BoneResult {
    pub bone: String,
    /// How many vertices moved when it bent.
    pub moved: usize,
    /// The most any edge grew, as a ratio of its rest length (1 is
    /// unchanged); edges shorter than a tenth of a millimetre aside.
    pub worst_ratio: f64,
    /// How many edges grew past half again: a vertex left behind by the
    /// bone its neighbours follow — the mark of a weight in the wrong place.
    pub torn: usize,
    /// How many shrank under a third: the inside of a bend folding, which
    /// every skin does at a sharp angle, and a joint too hard if many do.
    pub pinched: usize,
    /// Where the worst edge is, in the world, at rest.
    pub worst_at: Option<[f64; 3]>,
}

/// What the rig did.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub rig: String,
    pub meshes: Vec<String>,
    pub bones: Vec<BoneResult>,
    /// Vertices whose weights add to almost nothing: they stay behind.
    pub unweighted: usize,
    /// A few of them, in the world, for looking at.
    pub unweighted_sample: Vec<[f64; 3]>,
}

/// Bends each bone of `rig` by `degrees` about its own X, one at a time,
/// and measures the meshes it moves. The document is not changed.
pub fn check(doc: &Document, rig: ObjectId, degrees: f64) -> Result<Report, String> {
    let object = doc.object(rig).ok_or("no such object")?;
    let ObjectData::Armature(armature_id) = object.data else {
        return Err(format!("`{}` is not an armature", object.name));
    };
    let armature = doc.armature(armature_id).ok_or("the armature is missing")?;
    let uid = object.uid;
    let meshes: Vec<ObjectId> = doc
        .objects()
        .filter(|(_, o)| o.skin == Some(uid) && o.mesh().is_some())
        .map(|(id, _)| id)
        .collect();
    if meshes.is_empty() {
        return Err(format!(
            "nothing follows `{}`: bind a mesh to it first (object.bind_armature)",
            object.name
        ));
    }
    // At rest: no action, no pose — the shape the bend is measured from.
    let mut rest_doc = doc.clone();
    if let Some(o) = rest_doc.object_mut(rig) {
        o.action = None;
        o.nla.clear();
        o.pose.bones.clear();
        o.pose.ik.clear();
    }
    let ctx = EvalCtx::still();
    let shape = |d: &Document, id: ObjectId| -> Option<Vec<glam::DVec3>> {
        let mesh = ModifierCache::new().evaluated(d, id, &ctx)?;
        let world = d.world_transform(id);
        Some(
            mesh.positions()
                .iter()
                .map(|p| world.transform_point3(p.as_dvec3()))
                .collect(),
        )
    };
    let rest: Vec<(ObjectId, Vec<glam::DVec3>, Vec<[u32; 2]>)> = meshes
        .iter()
        .filter_map(|id| {
            let edges = rest_doc
                .object(*id)
                .and_then(|o| o.mesh())
                .and_then(|m| rest_doc.mesh(m))
                .map(|m| m.edges().to_vec())?;
            Some((*id, shape(&rest_doc, *id)?, edges))
        })
        .collect();

    let turn = kreate_math::DQuat::from_rotation_x(degrees.to_radians());
    let mut bones = Vec::new();
    for bone in &armature.bones {
        let mut bent = rest_doc.clone();
        if let Some(o) = bent.object_mut(rig) {
            o.pose.entry(&bone.name).transform.rotation = turn;
        }
        let mut result = BoneResult {
            bone: bone.name.clone(),
            moved: 0,
            worst_ratio: 1.0,
            torn: 0,
            pinched: 0,
            worst_at: None,
        };
        let mut worst = 0.0f64;
        for (id, before, edges) in &rest {
            let Some(after) = shape(&bent, *id) else {
                continue;
            };
            result.moved += before
                .iter()
                .zip(&after)
                .filter(|(a, b)| a.distance(**b) > 1e-6)
                .count();
            for e in edges {
                let (a, b) = (e[0] as usize, e[1] as usize);
                let (Some(p), Some(q), Some(r), Some(s)) =
                    (before.get(a), before.get(b), after.get(a), after.get(b))
                else {
                    continue;
                };
                let at_rest = p.distance(*q);
                if at_rest < 1e-4 {
                    continue;
                }
                let ratio = r.distance(*s) / at_rest;
                if ratio > 1.5 {
                    result.torn += 1;
                } else if ratio < 1.0 / 3.0 {
                    result.pinched += 1;
                }
                if ratio > worst {
                    worst = ratio;
                    result.worst_ratio = ratio;
                    result.worst_at = Some(((*p + *q) * 0.5).to_array());
                }
            }
        }
        bones.push(result);
    }
    // Worst first: what to fix is at the top.
    bones.sort_by(|a, b| b.worst_ratio.total_cmp(&a.worst_ratio));

    // Vertices no bone holds.
    let mut unweighted = 0;
    let mut sample = Vec::new();
    for (id, before, _) in &rest {
        let Some(mesh) = rest_doc
            .object(*id)
            .and_then(|o| o.mesh())
            .and_then(|m| rest_doc.mesh(m))
        else {
            continue;
        };
        let layers: Vec<&[f32]> = armature
            .bones
            .iter()
            .filter_map(|b| {
                mesh.attributes()
                    .get_f32(&kreate_scene::skin::weight_layer(&b.name))
            })
            .collect();
        for (v, p) in before.iter().enumerate() {
            let total: f32 = layers.iter().filter_map(|l| l.get(v)).sum();
            if total < 0.01 {
                unweighted += 1;
                if sample.len() < 8 {
                    sample.push(p.to_array());
                }
            }
        }
    }
    Ok(Report {
        rig: object.name.clone(),
        meshes: meshes
            .iter()
            .filter_map(|id| doc.object(*id).map(|o| o.name.clone()))
            .collect(),
        bones,
        unweighted,
        unweighted_sample: sample,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::DVec3;
    use kreate_mesh::attribute::{AttrData, Domain};
    use kreate_scene::armature::{Armature, Bone};
    use kreate_scene::document::Object;

    /// A column three units tall on a two-bone arm, weighted by `weigh`.
    fn arm(weigh: impl Fn(f32) -> (f32, f32)) -> (Document, ObjectId) {
        let mut doc = Document::new();
        let mut arm = Armature::new("Arm");
        arm.bones[0].name = "Upper".into();
        arm.bones[0].head = DVec3::ZERO;
        arm.bones[0].tail = DVec3::new(0.0, 0.0, 1.5);
        arm.add_bone(Bone {
            name: "Lower".into(),
            parent: Some(0),
            head: DVec3::new(0.0, 0.0, 1.5),
            tail: DVec3::new(0.0, 0.0, 3.0),
            connected: true,
            ..Default::default()
        });
        let mut column = kreate_mesh::primitives::uv_sphere(0.5, 12, 24);
        for p in column.positions_mut() {
            p.x *= 0.3;
            p.y *= 0.3;
            p.z = (p.z + 0.5) * 3.0;
        }
        let (upper, lower): (Vec<f32>, Vec<f32>) =
            column.positions().iter().map(|p| weigh(p.z)).unzip();
        for (name, w) in [("Upper", upper), ("Lower", lower)] {
            column.attributes_mut().insert(
                &kreate_scene::skin::weight_layer(name),
                Domain::Point,
                AttrData::F32(std::sync::Arc::new(w)),
            );
        }
        let armature = doc.add_armature(arm);
        let rig = doc.add_object(Object::new("Rig", ObjectData::Armature(armature)));
        let mesh = doc.add_mesh(column);
        let body = doc.add_object(Object::new("Body", ObjectData::Mesh(mesh)));
        doc.set_parent(body, Some(rig)).expect("parented");
        let uid = doc.object(rig).expect("rig").uid;
        doc.object_mut(body).expect("body").skin = Some(uid);
        (doc, rig)
    }

    #[test]
    fn a_good_rig_bends_without_tearing_and_a_bad_one_is_found() {
        // Blended across the joint: a sleeve.
        let (good, rig) = arm(|z| {
            let t = ((z - 1.2) / 0.6).clamp(0.0, 1.0);
            (1.0 - t, t)
        });
        let report = check(&good, rig, 45.0).expect("checked");
        assert_eq!(report.meshes, ["Body"]);
        assert_eq!(report.unweighted, 0);
        assert!(report.bones.iter().all(|b| b.moved > 0), "{report:?}");
        assert!(report.bones.iter().all(|b| b.torn == 0), "{report:?}");

        // One stray vertex high on the arm left on the upper bone, and the
        // top cap on none.
        let (bad, rig) = arm(|z| {
            if z > 2.95 {
                (0.0, 0.0)
            } else if (2.2..2.3).contains(&z) {
                (1.0, 0.0)
            } else if z > 1.5 {
                (0.0, 1.0)
            } else {
                (1.0, 0.0)
            }
        });
        let report = check(&bad, rig, 45.0).expect("checked");
        assert!(report.unweighted > 0, "the cap is left behind");
        let worst = &report.bones[0];
        assert!(worst.torn > 0, "{report:?}");
        let at = worst.worst_at.expect("somewhere");
        assert!(at[2] > 1.5, "the tear is up the arm: {at:?}");
    }

    #[test]
    fn a_rig_nothing_follows_says_so() {
        let mut doc = Document::new();
        let armature = doc.add_armature(Armature::new("Alone"));
        let rig = doc.add_object(Object::new("Alone", ObjectData::Armature(armature)));
        let said = check(&doc, rig, 45.0).expect_err("nothing bound");
        assert!(said.contains("bind"), "{said}");
    }
}
