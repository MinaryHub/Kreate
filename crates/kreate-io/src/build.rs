//! Turning a read scene into a document.
//!
//! The step that did not exist: an [`ImportScene`] became a list of
//! `(name, MeshData)` pairs, every one of them placed at the origin, and a
//! light could not be imported at all because a light *is* a placement and
//! there was nowhere to put one.
//!
//! Here rather than in the application, because it is the other half of the
//! fidelity contract §5.7 asks for: what a file said and what the document ends
//! up holding are compared in tests, and a conversion living in the shell could
//! not be.

use crate::ir::{
    ExportNode, ExportScene, ImportCamera, ImportImage, ImportLight, ImportLightKind,
    ImportMaterial, ImportNodeData, ImportScene, LossyNote,
};
use kreate_scene::document::{ImageId, MaterialId, Object, ObjectData, ObjectId};
use kreate_scene::{ColorSpace, Document, MaterialMap};

/// What was lost or guessed while building the document.
///
/// Returned rather than logged, for the same reason [`crate::ir::LossyNote`] is:
/// a fidelity contract that reports into a log is a contract nothing can test.
#[derive(Clone, Debug, PartialEq)]
pub enum BuildNote {
    /// A node named a parent that is not in the file, or a chain that loops.
    /// The node is placed at the root with its own transform.
    BrokenParent { node: String },
    /// A node pointed at a mesh, light or camera the file does not contain.
    MissingData { node: String },
    /// A spot's cone came through; its blend was derived from the two angles
    /// glTF writes, which is not the same parameterisation.
    ConeBlend { light: String },
    /// A material named a texture that is not in the file, or one that could
    /// not be read. The slot is left empty.
    MissingImage { material: String },
    /// A second clip moved a node the first had already given an action. An
    /// object plays one action, so the later clip's movement of it was left
    /// out.
    LaterClip { animation: String, node: String },
    /// Something hung from a joint, which is a bone here and not an object:
    /// it was parented to the armature instead, and no longer follows the
    /// bone.
    HungFromJoint { node: String },
}

/// Adds everything in `scene` to `doc`, placed as the file placed it.
///
/// Returns the objects it made, in node order, so a caller can select what it
/// just imported.
pub fn build_into(scene: &ImportScene, doc: &mut Document) -> (Vec<ObjectId>, Vec<BuildNote>) {
    let mut notes = Vec::new();

    // The datablocks first, because a node refers to one by index and a
    // document refers to one by id.
    let materials = build_materials(scene, doc, &mut notes);

    // A face in the file names a material in the *file's* table; a face in the
    // document names a slot in its *mesh's* list (§7.5). So each mesh gets the
    // materials it actually uses as its slots, in the order of their file
    // numbers, and its faces are renumbered into that list. Handing every mesh
    // the whole table would work and would give a one-material prop fifty
    // empty-looking slots in a file with fifty materials.
    let mut working = scene.clone();
    let mut slot_lists: Vec<Vec<Option<MaterialId>>> = Vec::with_capacity(working.meshes.len());
    for mesh in &mut working.meshes {
        let Some(faces) = &mut mesh.face_materials else {
            slot_lists.push(Vec::new());
            continue;
        };
        let mut used: Vec<i32> = faces.iter().copied().filter(|m| *m >= 0).collect();
        used.sort_unstable();
        used.dedup();
        for face in faces.iter_mut() {
            *face = used
                .binary_search(face)
                .map(|slot| slot as i32)
                .unwrap_or(-1);
        }
        slot_lists.push(
            used.iter()
                .map(|m| materials.get(*m as usize).copied())
                .collect(),
        );
    }
    let meshes: Vec<_> = {
        let mut out = Vec::with_capacity(working.meshes.len());
        let (converted, _) = working.into_mesh_data();
        for ((_, mesh), slots) in converted.into_iter().zip(slot_lists) {
            let id = doc.add_mesh(mesh);
            if !slots.is_empty() {
                doc.set_mesh_slots(id, slots);
            }
            out.push(id);
        }
        out
    };

    let lights: Vec<_> = scene
        .lights
        .iter()
        .map(|light| {
            let kind = match light.kind {
                ImportLightKind::Point => kreate_scene::LightKind::Point {
                    // The file says nothing about how big the bulb is, and a
                    // source of no size casts a shadow nothing real casts. The
                    // datablock's own default is what a lamp looks like.
                    radius: match kreate_scene::LightKind::BULB {
                        kreate_scene::LightKind::Point { radius } => radius,
                        _ => 0.0,
                    },
                },
                ImportLightKind::Directional => kreate_scene::LightKind::SUN,
                ImportLightKind::Spot { inner, outer } => {
                    notes.push(BuildNote::ConeBlend {
                        light: light.name.clone(),
                    });
                    kreate_scene::LightKind::Spot {
                        radius: 0.05,
                        // glTF writes half-angles; the datablock stores the
                        // full cone, which is what a cone is called everywhere
                        // a person meets one.
                        cone_angle: (outer * 2.0) as f64,
                        // And a blend as the fraction of the cone that fades,
                        // which glTF expresses as a second angle instead. The
                        // two parameterisations carry the same shape and the
                        // conversion is exact in both directions.
                        blend: if outer > 0.0 {
                            ((outer - inner) / outer).clamp(0.0, 1.0) as f64
                        } else {
                            0.0
                        },
                    }
                }
            };
            let mut out = kreate_scene::Light::of_kind(light.name.clone(), kind);
            out.intensity = light.intensity as f64;
            // Linear on both sides: glTF's colour is a tint in linear space and
            // so is the datablock's, so this is a copy and not a decode.
            out.color = kreate_math::SceneRgb::new(light.color[0], light.color[1], light.color[2]);
            doc.add_light(out)
        })
        .collect();

    let cameras: Vec<_> = scene
        .cameras
        .iter()
        .map(|camera| {
            let mut out = kreate_scene::Camera::new(camera.name.clone());
            match (camera.fov_y, camera.ortho_width) {
                (Some(fov), _) => {
                    out.projection = kreate_scene::Projection::Perspective;
                    // The file says an angle and the datablock stores a lens.
                    // The sensor is the datablock's own — a file that writes a
                    // field of view is not saying anything about a sensor, and
                    // the focal length that follows is the one that gives that
                    // angle on the sensor the document has.
                    out.focal_length_mm = (out.sensor_height_mm * 0.5) / (fov as f64 * 0.5).tan();
                }
                (None, Some(width)) => {
                    out.projection = kreate_scene::Projection::Orthographic;
                    out.ortho_scale = (width * scene.source.unit_scale) as f64;
                }
                (None, None) => {}
            }
            out.clip_near = (camera.clip_near * scene.source.unit_scale) as f64;
            if let Some(far) = camera.clip_far {
                out.clip_far = (far * scene.source.unit_scale) as f64;
            }
            doc.add_camera(out)
        })
        .collect();

    // A file with no hierarchy at all — OBJ, STL — still has meshes, and they
    // belong at the origin because that is where the file put them. Inventing a
    // node each would claim a structure the file does not have.
    if scene.nodes.is_empty() {
        let made = meshes
            .iter()
            .zip(&scene.meshes)
            .map(|(id, mesh)| doc.add_object(Object::new(mesh.name.clone(), ObjectData::Mesh(*id))))
            .collect();
        return (made, notes);
    }

    // Joints become bones, not objects: each skin is an armature.
    let rigs = build_armatures(scene, doc);
    let rig_of_joint = |node: usize| rigs.iter().find(|r| r.joints.contains(&node));

    let made: Vec<Option<ObjectId>> = scene
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            if rig_of_joint(index).is_some() {
                return None;
            }
            let data = match node.data {
                ImportNodeData::Mesh(i) => match meshes.get(i) {
                    Some(id) => ObjectData::Mesh(*id),
                    None => {
                        notes.push(BuildNote::MissingData {
                            node: node.name.clone(),
                        });
                        ObjectData::Empty
                    }
                },
                ImportNodeData::Light(i) => match lights.get(i) {
                    Some(id) => ObjectData::Light(*id),
                    None => {
                        notes.push(BuildNote::MissingData {
                            node: node.name.clone(),
                        });
                        ObjectData::Empty
                    }
                },
                ImportNodeData::Camera(i) => match cameras.get(i) {
                    Some(id) => ObjectData::Camera(*id),
                    None => {
                        notes.push(BuildNote::MissingData {
                            node: node.name.clone(),
                        });
                        ObjectData::Empty
                    }
                },
                ImportNodeData::Empty => ObjectData::Empty,
            };
            // Placed at the origin for now. The transforms go on *after* the
            // parents, and the order is load-bearing: `set_parent` remembers
            // where the object was so that reparenting does not move it, which
            // is right when a user drags one thing onto another and wrong here.
            // A file's child transform is relative to its parent already.
            //
            // Parenting everything while it is still at the origin makes that
            // memory an identity, and then the transforms mean what the file
            // meant.
            Some(doc.add_object(Object::new(node.name.clone(), data)))
        })
        .collect();

    for (index, node) in scene.nodes.iter().enumerate() {
        let Some(Some(child_id)) = made.get(index) else { continue };
        // A skinned mesh hangs from its armature, placed where its node is
        // in the world — the space its armature's bones were built in.
        if let Some(rig) = node.skin.and_then(|s| rigs.iter().find(|r| r.skin == s)) {
            let _ = doc.set_parent(*child_id, Some(rig.object));
            continue;
        }
        let Some(parent) = node.parent else { continue };
        let parent_id = match made.get(parent) {
            Some(Some(p)) => Some(*p),
            Some(None) => {
                let rig = rig_of_joint(parent).map(|r| r.object);
                notes.push(BuildNote::HungFromJoint {
                    node: node.name.clone(),
                });
                rig
            }
            None => None,
        };
        let Some(parent_id) = parent_id else {
            notes.push(BuildNote::BrokenParent {
                node: node.name.clone(),
            });
            continue;
        };
        // `set_parent` refuses a cycle, which a file can express and this
        // cannot fix. Refused means the node stays at the root with its own
        // transform, which is the honest partial answer.
        if doc.set_parent(*child_id, Some(parent_id)).is_err() {
            notes.push(BuildNote::BrokenParent {
                node: node.name.clone(),
            });
        }
    }

    // And now the placements, with every parent link already in place.
    for (index, node) in scene.nodes.iter().enumerate() {
        let Some(Some(id)) = made.get(index) else { continue };
        if node.skin.is_some_and(|s| rigs.iter().any(|r| r.skin == s)) {
            if let (Some(world), Some(object)) = (scene.world_transform(index), doc.object_mut(*id)) {
                object.transform = to_transform(world);
            }
            continue;
        }
        if let Some(object) = doc.object_mut(*id) {
            object.transform = to_transform(node.transform.to_kreate(&scene.source));
        }
    }

    // Bound: the weights as the bones' vertex groups, and the mesh following
    // its armature.
    for (index, node) in scene.nodes.iter().enumerate() {
        let (Some(Some(id)), ImportNodeData::Mesh(m)) = (made.get(index), node.data) else {
            continue;
        };
        let Some(rig) = node.skin.and_then(|s| rigs.iter().find(|r| r.skin == s)) else {
            continue;
        };
        let (Some(Some(skin)), Some(mesh)) = (scene.vertex_skins.get(m), meshes.get(m)) else {
            continue;
        };
        let rig_uid = doc.object(rig.object).map(|o| o.uid);
        if let Some(data) = doc.mesh_mut(*mesh) {
            let n = data.num_verts() as usize;
            for (k, bone) in rig.bone_names.iter().enumerate() {
                let layer = kreate_scene::skin::weight_layer(bone);
                if data.attributes().get_f32(&layer).is_some() {
                    continue;
                }
                let mut w = vec![0.0f32; n];
                for (v, slot) in w.iter_mut().enumerate() {
                    if let (Some(j), Some(wt)) = (skin.joints.get(v), skin.weights.get(v)) {
                        for i in 0..4 {
                            if j[i] as usize == k {
                                *slot += wt[i];
                            }
                        }
                    }
                }
                data.attributes_mut().insert(
                    &layer,
                    kreate_mesh::attribute::Domain::Point,
                    kreate_mesh::attribute::AttrData::F32(std::sync::Arc::new(w)),
                );
            }
        }
        if let Some(object) = doc.object_mut(*id) {
            object.skin = rig_uid;
        }
    }

    build_animations(scene, &made, doc, &mut notes);
    for rig in &rigs {
        build_bone_animation(scene, rig, doc);
    }

    let mut out: Vec<ObjectId> = made.into_iter().flatten().collect();
    out.extend(rigs.iter().map(|r| r.object));
    (out, notes)
}

/// One skin, built as an armature.
struct Rig {
    skin: usize,
    object: ObjectId,
    /// Node indices, in the skin's joint order — a vertex's joint number.
    joints: Vec<usize>,
    /// The bone each joint became, in the same order.
    bone_names: Vec<String>,
    /// The bone each joint became, by index into the armature's bones.
    bone_of_joint: Vec<usize>,
}

/// Every skin as an armature object at the world origin: a bone per joint,
/// its frame the joint's frame at the bind, so a joint's turn is the bone's.
fn build_armatures(scene: &ImportScene, doc: &mut Document) -> Vec<Rig> {
    use kreate_math::{DAffine3, DQuat, DVec3};
    use kreate_scene::armature::{Armature, Bone};
    let mut rigs = Vec::new();
    let mut claimed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (s, skin) in scene.skins.iter().enumerate() {
        // A joint in two skins is a bone in the first.
        let joints: Vec<usize> = skin
            .joints
            .iter()
            .copied()
            .filter(|j| *j < scene.nodes.len())
            .collect();
        if joints.is_empty() || joints.iter().any(|j| claimed.contains(j)) {
            continue;
        }
        claimed.extend(joints.iter().copied());
        // Where each joint stood at the bind, in Kreate's space. An inverse
        // bind matrix takes the mesh's own coordinates into the joint's, so
        // its inverse is the joint in the mesh's space — and the mesh is
        // where its node puts it. glTF ignores that placement when it draws a
        // skin; it is still the space the matrices were written against, by
        // every exporter that writes one, ours included. When the file has no
        // matrices the joints' nodes are the bind.
        let mesh_world = scene
            .nodes
            .iter()
            .position(|n| n.skin == Some(s))
            .and_then(|n| scene.world_transform(n))
            .unwrap_or(DAffine3::IDENTITY);
        let bind: Vec<DAffine3> = joints
            .iter()
            .enumerate()
            .map(|(k, node)| {
                match skin.inverse_bind.as_ref().and_then(|m| m.get(k)) {
                    Some(m) => mesh_world * scene.matrix_to_kreate(m.inverse()),
                    None => scene.world_transform(*node).unwrap_or(DAffine3::IDENTITY),
                }
            })
            .collect();
        // A bone's frame has no scale.
        let frames: Vec<(DQuat, DVec3)> = bind
            .iter()
            .map(|m| {
                let (_, r, t) = m.to_scale_rotation_translation();
                (r.normalize(), t)
            })
            .collect();
        // The nearest ancestor that is a joint of this skin.
        let parent_joint = |k: usize| -> Option<usize> {
            let mut at = scene.nodes[joints[k]].parent;
            for _ in 0..=scene.nodes.len() {
                let p = at?;
                if let Some(pk) = joints.iter().position(|j| *j == p) {
                    return Some(pk);
                }
                at = scene.nodes.get(p)?.parent;
            }
            None
        };
        let parents: Vec<Option<usize>> = (0..joints.len()).map(parent_joint).collect();
        // Parents first: by depth among the joints.
        let depth = |mut k: usize| {
            let mut d = 0;
            while let Some(p) = parents[k] {
                d += 1;
                k = p;
                if d > joints.len() {
                    break;
                }
            }
            d
        };
        let mut order: Vec<usize> = (0..joints.len()).collect();
        order.sort_by_key(|k| depth(*k));
        let mut armature = Armature::new(skin.name.clone());
        armature.bones.clear();
        let mut bone_of_joint = vec![0usize; joints.len()];
        let mut lengths = vec![0.0f64; joints.len()];
        for &k in &order {
            let (rotation, head) = frames[k];
            let along = rotation * DVec3::Y;
            // As long as the farthest child joint lies along it; otherwise
            // its parent's length, or a tenth of a unit.
            let reach = (0..joints.len())
                .filter(|c| parents[*c] == Some(k))
                .map(|c| (frames[c].1 - head).dot(along))
                .fold(0.0f64, f64::max);
            let length = if reach > 1e-4 {
                reach
            } else {
                parents[k].map(|p| lengths[p]).filter(|l| *l > 0.0).unwrap_or(0.1)
            };
            lengths[k] = length;
            // The roll that turns the shortest-arc frame onto the joint's.
            let align = if along.dot(DVec3::Y) < -1.0 + 1e-12 {
                DQuat::from_rotation_z(std::f64::consts::PI)
            } else {
                DQuat::from_rotation_arc(DVec3::Y, along)
            };
            let twist = align.inverse() * rotation;
            let roll = 2.0 * twist.y.atan2(twist.w);
            let name = armature.unique_name(&scene.nodes[joints[k]].name);
            bone_of_joint[k] = armature.add_bone(Bone {
                name,
                parent: parents[k].map(|p| bone_of_joint[p]),
                head,
                tail: head + along * length,
                roll,
                ..Default::default()
            });
        }
        let bone_names: Vec<String> = bone_of_joint
            .iter()
            .map(|b| armature.bones[*b].name.clone())
            .collect();
        let name = if skin.name.is_empty() {
            "Armature".to_string()
        } else {
            skin.name.clone()
        };
        let id = doc.add_armature(armature);
        let object = doc.add_object(Object::new(name, ObjectData::Armature(id)));
        let rig = Rig {
            skin: s,
            object,
            joints,
            bone_names,
            bone_of_joint,
        };
        // Standing as the file's nodes stand, which need not be the bind.
        let pose = rig_pose(scene, &rig, doc, None);
        if let Some(o) = doc.object_mut(object) {
            for (name, t) in pose {
                if !t.to_affine().abs_diff_eq(DAffine3::IDENTITY, 1e-6) {
                    o.pose.entry(&name).transform = t;
                }
            }
        }
        rigs.push(rig);
    }
    rigs
}

/// Each bone's pose that puts it where its joint is at time `t` (seconds),
/// or as the nodes stand when `None`.
fn rig_pose(
    scene: &ImportScene,
    rig: &Rig,
    doc: &Document,
    t: Option<f32>,
) -> Vec<(String, kreate_math::DTransform)> {
    use kreate_math::DAffine3;
    let Some((_, armature)) = doc.armature_of(rig.object) else {
        return Vec::new();
    };
    // A node's placement under its parent at `t`.
    let local = |node: usize| -> DAffine3 {
        let n = &scene.nodes[node];
        let mut placed = n.transform;
        if let Some(t) = t {
            for animation in &scene.animations {
                for c in animation.channels.iter().filter(|c| c.node == node && c.key_count() > 0) {
                    match c.values {
                        crate::ir::ChannelValues::Translation(_) => {
                            placed.translation = c.vec3_at(t).unwrap_or(placed.translation)
                        }
                        crate::ir::ChannelValues::Rotation(_) => {
                            placed.rotation = c.quat_at(t).unwrap_or(placed.rotation)
                        }
                        crate::ir::ChannelValues::Scale(_) => {
                            placed.scale = c.vec3_at(t).unwrap_or(placed.scale)
                        }
                    }
                }
            }
        }
        placed.to_kreate(&scene.source)
    };
    let world = |node: usize| -> DAffine3 {
        let mut out = DAffine3::IDENTITY;
        let mut at = Some(node);
        for _ in 0..=scene.nodes.len() {
            let Some(n) = at else { break };
            out = local(n) * out;
            at = scene.nodes[n].parent;
        }
        out
    };
    // Bones are parents first, so each parent's posed frame is ready.
    let mut posed: Vec<DAffine3> = vec![DAffine3::IDENTITY; armature.bones.len()];
    let mut joint_of_bone = vec![usize::MAX; armature.bones.len()];
    for (k, b) in rig.bone_of_joint.iter().enumerate() {
        joint_of_bone[*b] = k;
    }
    let mut out = Vec::new();
    for (i, bone) in armature.bones.iter().enumerate() {
        let Some(&k) = joint_of_bone.get(i).filter(|k| **k != usize::MAX) else {
            continue;
        };
        let at = world(rig.joints[k]);
        let base = match bone.parent {
            Some(p) => posed[p] * armature.bones[p].rest().inverse() * bone.rest(),
            None => bone.rest(),
        };
        let pose = kreate_math::DTransform::from_affine(base.inverse() * at);
        posed[i] = base * pose.to_affine();
        out.push((bone.name.clone(), pose));
    }
    out
}

/// The clip's movement of a skin's joints, as its armature's action: every
/// frame of the moving range, each bone's pose keyed.
fn build_bone_animation(scene: &ImportScene, rig: &Rig, doc: &mut Document) {
    use kreate_scene::animation::{bone_path, Action, Interpolation};
    let times: Vec<f32> = scene
        .animations
        .iter()
        .flat_map(|a| a.channels.iter())
        .filter(|c| rig.joints.contains(&c.node))
        .flat_map(|c| c.times[..c.key_count()].iter().copied())
        .collect();
    let (Some(first), Some(last)) = (
        times.iter().copied().reduce(f32::min),
        times.iter().copied().reduce(f32::max),
    ) else {
        return;
    };
    let fps = if doc.playback.fps > 0.0 { doc.playback.fps } else { 24.0 };
    let start = doc.playback.start as f64;
    let frame_of = |t: f32| start + t as f64 * fps;
    let (a, b) = (frame_of(first), frame_of(last));
    let mut frames = vec![a];
    let mut f = a.floor() + 1.0;
    while f < b {
        frames.push(f);
        f += 1.0;
    }
    if b > a {
        frames.push(b);
    }
    let name = doc.object(rig.object).map(|o| o.name.clone()).unwrap_or_default();
    let mut action = Action::new(format!("{name}Action"));
    let mut previous: std::collections::HashMap<String, kreate_math::DQuat> =
        std::collections::HashMap::new();
    for frame in &frames {
        let t = ((frame - start) / fps) as f32;
        for (bone, pose) in rig_pose(scene, rig, doc, Some(t)) {
            for i in 0..3u8 {
                action.curve_mut(&bone_path(&bone, "location"), i).insert(
                    *frame,
                    pose.translation[i as usize],
                    Interpolation::Linear,
                );
                action.curve_mut(&bone_path(&bone, "scale"), i).insert(
                    *frame,
                    pose.scale[i as usize],
                    Interpolation::Linear,
                );
            }
            // On the previous key's side: q and -q are one turn, and the
            // curves draw straight lines between numbers.
            let mut q = pose.rotation;
            if previous.get(&bone).is_some_and(|p| p.dot(q) < 0.0) {
                q = -q;
            }
            previous.insert(bone.clone(), q);
            for (i, v) in [q.w, q.x, q.y, q.z].into_iter().enumerate() {
                action
                    .curve_mut(&bone_path(&bone, "rotation_quaternion"), i as u8)
                    .insert(*frame, v, Interpolation::Linear);
            }
        }
    }
    action.select_all_with(|_| false);
    let id = doc.add_action(action);
    if let Some(o) = doc.object_mut(rig.object) {
        o.action = Some(id);
    }
    if b.ceil() as i32 > doc.playback.end {
        doc.playback.end = b.ceil() as i32;
    }
    let frame = doc.playback.frame;
    doc.animate(frame);
}

/// The file's clips as actions: one per animated node, keyed on the frames
/// the file keyed.
///
/// glTF keys a node's whole placement, in its own axes, with a quaternion; the
/// document keys location, Euler angles and scale on Z-up. So every key is the
/// whole placement at that instant, put through the same conversion the rest
/// placement went through — asking the channels where the node is rather than
/// converting their numbers one by one, which the axis change would scramble.
///
/// Where the file's curve is not one the document can draw between keys —
/// a cubic spline, or a rotation (a slerp is not a straight line in Euler
/// angles) — it is sampled every frame. Straight lines and holds are exact on
/// the file's own keys.
fn build_animations(
    scene: &ImportScene,
    made: &[Option<ObjectId>],
    doc: &mut Document,
    notes: &mut Vec<BuildNote>,
) {
    use crate::ir::ChannelInterpolation as I;
    use kreate_scene::animation::{Action, Interpolation};
    let fps = if doc.playback.fps > 0.0 {
        doc.playback.fps
    } else {
        24.0
    };
    let start = doc.playback.start as f64;
    let mut animated: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut last_frame = f64::NEG_INFINITY;
    for animation in &scene.animations {
        let mut nodes: Vec<usize> = animation.channels.iter().map(|c| c.node).collect();
        nodes.sort_unstable();
        nodes.dedup();
        for node in nodes {
            let (Some(Some(object)), Some(rest)) = (made.get(node), scene.nodes.get(node)) else {
                continue;
            };
            if !animated.insert(node) {
                notes.push(BuildNote::LaterClip {
                    animation: animation.name.clone(),
                    node: rest.name.clone(),
                });
                continue;
            }
            let channels: Vec<_> = animation
                .channels
                .iter()
                .filter(|c| c.node == node && c.key_count() > 0)
                .collect();
            if channels.is_empty() {
                continue;
            }
            let mut times: Vec<f32> = channels
                .iter()
                .flat_map(|c| c.times[..c.key_count()].iter().copied())
                .collect();
            times.sort_by(f32::total_cmp);
            times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            let step = channels.iter().all(|c| c.interpolation == I::Step);
            let sampled = channels.iter().any(|c| {
                c.interpolation == I::CubicSpline
                    || (c.interpolation == I::Linear
                        && matches!(c.values, crate::ir::ChannelValues::Rotation(_))
                        && c.key_count() > 1)
            });
            let frame_of = |t: f32| start + t as f64 * fps;
            let mut frames: Vec<f64> = times.iter().map(|t| frame_of(*t)).collect();
            if sampled {
                let (first, last) = (frames[0], frames[frames.len() - 1]);
                let mut f = first.ceil();
                while f < last {
                    frames.push(f);
                    f += 1.0;
                }
                frames.sort_by(f64::total_cmp);
                frames.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            }
            let (mut moves_t, mut moves_r, mut moves_s) = (false, false, false);
            for c in &channels {
                match c.values {
                    crate::ir::ChannelValues::Translation(_) => moves_t = true,
                    crate::ir::ChannelValues::Rotation(_) => moves_r = true,
                    crate::ir::ChannelValues::Scale(_) => moves_s = true,
                }
            }
            let interpolation = if step {
                Interpolation::Constant
            } else {
                Interpolation::Linear
            };
            let mut action = Action::new(format!("{}Action", rest.name));
            let mut previous_euler: Option<kreate_math::DVec3> = None;
            for frame in &frames {
                let t = ((frame - start) / fps) as f32;
                let mut placed = rest.transform;
                for c in &channels {
                    match c.values {
                        crate::ir::ChannelValues::Translation(_) => {
                            placed.translation = c.vec3_at(t).unwrap_or(placed.translation)
                        }
                        crate::ir::ChannelValues::Rotation(_) => {
                            placed.rotation = c.quat_at(t).unwrap_or(placed.rotation)
                        }
                        crate::ir::ChannelValues::Scale(_) => {
                            placed.scale = c.vec3_at(t).unwrap_or(placed.scale)
                        }
                    }
                }
                let local = to_transform(placed.to_kreate(&scene.source));
                if moves_t {
                    for i in 0..3u8 {
                        action.curve_mut("location", i).insert(
                            *frame,
                            local.translation[i as usize],
                            interpolation,
                        );
                    }
                }
                if moves_r {
                    let euler = continuous_euler(local.rotation, previous_euler);
                    previous_euler = Some(euler);
                    for i in 0..3u8 {
                        action.curve_mut("rotation_euler", i).insert(
                            *frame,
                            euler[i as usize],
                            interpolation,
                        );
                    }
                }
                if moves_s {
                    for i in 0..3u8 {
                        action.curve_mut("scale", i).insert(
                            *frame,
                            local.scale[i as usize],
                            interpolation,
                        );
                    }
                }
                last_frame = last_frame.max(*frame);
            }
            action.select_all_with(|_| false);
            let id = doc.add_action(action);
            if let Some(o) = doc.object_mut(*object) {
                o.action = Some(id);
            }
        }
    }
    // The scene's range grows to hold what arrived, and never shrinks: a clip
    // that ends past frame 250 should play to its end, and one shorter than
    // the range leaves the author's range alone.
    if last_frame.is_finite() && last_frame.ceil() as i32 > doc.playback.end {
        doc.playback.end = last_frame.ceil() as i32;
    }
    if !animated.is_empty() {
        let frame = doc.playback.frame;
        doc.animate(frame);
    }
}

/// Euler angles for `q`, chosen to be nearest the previous key's.
///
/// Every rotation has two XYZ Euler triples, and each angle any number of whole
/// turns; the plain conversion picks within ±π, so a spin past half a turn
/// would jump back by a full one between two keys and the curve would play the
/// long way round. Of the two triples, each unwrapped to the previous one, the
/// closer wins.
fn continuous_euler(
    q: kreate_math::DQuat,
    previous: Option<kreate_math::DVec3>,
) -> kreate_math::DVec3 {
    use std::f64::consts::{PI, TAU};
    let e = kreate_scene::animation::to_euler(q);
    let Some(p) = previous else { return e };
    let unwrap = |v: kreate_math::DVec3| {
        kreate_math::DVec3::new(
            v.x + ((p.x - v.x) / TAU).round() * TAU,
            v.y + ((p.y - v.y) / TAU).round() * TAU,
            v.z + ((p.z - v.z) / TAU).round() * TAU,
        )
    };
    let a = unwrap(e);
    let b = unwrap(kreate_math::DVec3::new(e.x + PI, PI - e.y, e.z + PI));
    if (a - p).length_squared() <= (b - p).length_squared() {
        a
    } else {
        b
    }
}

/// The file's materials as datablocks, in the file's order, with their images.
///
/// An image becomes one datablock per *space* it is used in: glTF lets one
/// picture be both a base colour and a roughness map, and the document
/// declares a space per image (§2.1), so the two uses are two datablocks. The
/// usual case — each image in one slot — is one each.
fn build_materials(
    scene: &ImportScene,
    doc: &mut Document,
    notes: &mut Vec<BuildNote>,
) -> Vec<MaterialId> {
    let mut made: std::collections::HashMap<(usize, ColorSpace), ImageId> =
        std::collections::HashMap::new();
    scene
        .materials
        .iter()
        .map(|m| {
            let mut image_for = |index: Option<usize>, space: ColorSpace| -> Option<ImageId> {
                let index = index?;
                if let Some(id) = made.get(&(index, space)) {
                    return Some(*id);
                }
                let built = scene.images.get(index).and_then(|image| {
                    if image.pixels.is_empty() {
                        return None;
                    }
                    crate::image_import::image_from_rgba8(
                        &image.name,
                        image.width,
                        image.height,
                        &image.pixels,
                        space,
                    )
                    .ok()
                });
                match built {
                    Some(image) => {
                        let id = doc.add_image(image);
                        made.insert((index, space), id);
                        Some(id)
                    }
                    None => {
                        notes.push(BuildNote::MissingImage {
                            material: m.name.clone(),
                        });
                        None
                    }
                }
            };
            let base_color_image =
                image_for(m.base_color_texture, MaterialMap::BaseColor.color_space());
            let metallic_roughness_image = image_for(
                m.metallic_roughness_texture,
                MaterialMap::MetallicRoughness.color_space(),
            );
            let normal_image = image_for(m.normal_texture, MaterialMap::Normal.color_space());
            doc.add_material(kreate_scene::Material {
                name: m.name.clone(),
                // Linear on both sides — glTF's factor is linear, and so is the
                // datablock's colour — so this is a copy, not a decode.
                base_color: kreate_math::SceneRgb::new(
                    m.base_color[0],
                    m.base_color[1],
                    m.base_color[2],
                ),
                alpha: m.base_color[3],
                metallic: m.metallic,
                roughness: m.roughness,
                base_color_image,
                metallic_roughness_image,
                normal_image,
                normal_strength: m.normal_scale,
                ..Default::default()
            })
        })
        .collect()
}

/// A transform in the shape the document stores.
fn to_transform(affine: kreate_math::DAffine3) -> kreate_math::DTransform {
    let (scale, rotation, translation) = affine.to_scale_rotation_translation();
    kreate_math::DTransform {
        translation,
        rotation,
        scale,
    }
}

/// Turns a document into a scene an exporter can write.
///
/// The inverse of [`build_into`], and the half that did not exist: an export
/// took a list of `(name, &MeshData)` pairs, so a lit scene could arrive and
/// could not leave. §5.7's fidelity contract only held in one direction, which
/// is not a contract — a round trip either closes or it is a one-way door with
/// a hopeful name.
///
/// Notes are [`LossyNote`]s rather than [`BuildNote`]s because this is the
/// export direction and they are what an exporter already returns: a caller
/// writing a file gets one list to show, not two.
pub fn scene_from_document(doc: &Document) -> (ExportScene<'_>, Vec<LossyNote>) {
    let mut notes = Vec::new();

    // The material table, and where each id landed in it. Every material, not
    // only the used ones: a document's material list is authored data and a
    // slot pointing at one nothing happens to use today is still the author's.
    let mut material_index: std::collections::HashMap<MaterialId, usize> =
        std::collections::HashMap::new();
    let mut materials = Vec::new();
    // Each image once, however many materials and slots use it, so a file
    // carries one copy of a shared texture — the sharing the document had.
    let mut images: Vec<ImportImage> = Vec::new();
    let mut image_index: std::collections::HashMap<ImageId, usize> =
        std::collections::HashMap::new();
    let mut texture = |map: MaterialMap, id: Option<ImageId>, notes: &mut Vec<LossyNote>| {
        let id = id?;
        if let Some(index) = image_index.get(&id) {
            return Some(*index);
        }
        let image = doc.image(id)?;
        if image.format == kreate_scene::PixelFormat::Rgba16F {
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: a float image was written as eight bits, and lost its range above one",
                    image.name
                ),
            });
        }
        let index = images.len();
        images.push(ImportImage {
            name: image.name.clone(),
            width: image.width(),
            height: image.height(),
            pixels: crate::image_import::rgba8_of(image, map.color_space()),
        });
        image_index.insert(id, index);
        Some(index)
    };
    for (id, material) in doc.materials() {
        if material.graph.is_some() {
            // glTF has no node graphs; a material is a standard surface and
            // nothing else. What goes out is the flat surface the material
            // had, which is what the graph was made from — and may no longer
            // be what it draws.
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: uses a node graph, which glTF cannot hold; its flat surface was written",
                    material.name
                ),
            });
        }
        material_index.insert(id, materials.len());
        materials.push(ImportMaterial {
            name: material.name.clone(),
            base_color: [
                material.base_color.r,
                material.base_color.g,
                material.base_color.b,
                material.alpha,
            ],
            metallic: material.metallic,
            roughness: material.roughness,
            // Not a field on the datablock yet: §2.3's node graph is where
            // emission will live, and inventing a value here would make the
            // file claim one.
            emissive: glam::Vec3::ZERO,
            base_color_texture: texture(
                MaterialMap::BaseColor,
                material.base_color_image,
                &mut notes,
            ),
            metallic_roughness_texture: texture(
                MaterialMap::MetallicRoughness,
                material.metallic_roughness_image,
                &mut notes,
            ),
            normal_texture: texture(MaterialMap::Normal, material.normal_image, &mut notes),
            normal_scale: material.normal_strength,
        });
    }

    let mut meshes: Vec<(String, &kreate_mesh::MeshData)> = Vec::new();
    let mut mesh_slots: Vec<Vec<Option<usize>>> = Vec::new();
    let mut mesh_index = std::collections::HashMap::new();
    let mut lights = Vec::new();
    let mut light_index = std::collections::HashMap::new();
    let mut cameras = Vec::new();
    let mut camera_index = std::collections::HashMap::new();
    let mut nodes: Vec<ExportNode> = Vec::new();
    let mut node_of = std::collections::HashMap::new();

    for (id, object) in doc.objects() {
        node_of.insert(id, nodes.len());
        let data = match object.data {
            // A mesh id the document no longer has. Resolved here rather than
            // when the list is handed over, because dropping an entry later
            // would shift every index after it and hang the wrong geometry on
            // every node that followed.
            ObjectData::Mesh(mesh) => match doc.mesh(mesh) {
                None => {
                    notes.push(LossyNote::Unsupported {
                        what: format!("{}: its mesh is not in the document", object.name),
                    });
                    ImportNodeData::Empty
                }
                // Deduplicated on the id, which is what a linked duplicate is:
                // two objects and one mesh. glTF says the same thing the same
                // way — several nodes pointing at one mesh — so the sharing
                // survives the file rather than becoming two copies.
                Some(data) => {
                    let index = *mesh_index.entry(mesh).or_insert_with(|| {
                        let slots = doc
                            .mesh_slots(mesh)
                            .iter()
                            .map(|slot| slot.and_then(|id| material_index.get(&id).copied()))
                            .collect();
                        mesh_slots.push(slots);
                        meshes.push((object.name.clone(), data));
                        meshes.len() - 1
                    });
                    ImportNodeData::Mesh(index)
                }
            },
            ObjectData::Light(light) => {
                let index = *light_index.entry(light).or_insert_with(|| {
                    lights.push(match doc.light(light) {
                        Some(light) => export_light(light, &mut notes),
                        None => {
                            notes.push(LossyNote::Unsupported {
                                what: format!("{}: its light is not in the document", object.name),
                            });
                            ImportLight {
                                name: object.name.clone(),
                                kind: ImportLightKind::Point,
                                color: [1.0; 3],
                                intensity: 0.0,
                                range: None,
                            }
                        }
                    });
                    lights.len() - 1
                });
                ImportNodeData::Light(index)
            }
            ObjectData::Camera(camera) => {
                let index = *camera_index.entry(camera).or_insert_with(|| {
                    cameras.push(match doc.camera(camera) {
                        Some(camera) => export_camera(camera),
                        None => {
                            notes.push(LossyNote::Unsupported {
                                what: format!("{}: its camera is not in the document", object.name),
                            });
                            ImportCamera {
                                name: object.name.clone(),
                                fov_y: Some(std::f32::consts::FRAC_PI_4),
                                ortho_width: None,
                                clip_near: 0.1,
                                clip_far: None,
                            }
                        }
                    });
                    cameras.len() - 1
                });
                ImportNodeData::Camera(index)
            }
            ObjectData::Empty => ImportNodeData::Empty,
            // The armature's own node; its bones follow every object's, as
            // nodes of their own under it. See below.
            ObjectData::Armature(_) => ImportNodeData::Empty,
        };
        nodes.push(ExportNode {
            name: object.name.clone(),
            // Filled in below: an object may name a parent that comes after it
            // in the arena, and an index into a half-built list would be wrong.
            parent: None,
            // **Not `object.transform`.** The document composes a world
            // transform as `parent_world · parent_inverse · local`, where the
            // inverse is where the parent stood at the moment of parenting —
            // that is what keeps dragging one object onto another from
            // teleporting it. A file's child transform is relative to its
            // parent with nothing in between, so the two have to be folded
            // together here or every parented object leaves at the wrong place.
            transform: if object.parent.is_some() {
                kreate_math::DTransform::from_affine(
                    object.parent_inverse * object.transform.to_affine(),
                )
            } else {
                object.transform
            },
            data,
            skin: None,
        });
        if !object.visible {
            // Written anyway. Visibility is a viewport state and not a property
            // of the model, and a user who hid a wall to work behind it would
            // not expect the wall to be missing from the file.
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: hidden in the viewport, and written anyway",
                    object.name
                ),
            });
        }
    }

    for (index, (_, object)) in doc.objects().enumerate() {
        let Some(parent) = object.parent else {
            continue;
        };
        nodes[index].parent = node_of.get(&parent).copied();
        if nodes[index].parent.is_none() {
            notes.push(LossyNote::Unsupported {
                what: format!("{}: its parent is not in the document", object.name),
            });
        }
    }

    // Bones, after every object so that object `i` is still node `i`: a node
    // each, under its parent bone or the armature, placed as posed now. A
    // joint in glTF is a node like any other, and that is what a reader that
    // knows nothing of skins sees too: the skeleton, standing as it stands.
    let mut bone_nodes: std::collections::HashMap<ObjectId, Vec<usize>> =
        std::collections::HashMap::new();
    for (id, object) in doc.objects() {
        let Some((_, armature)) = doc.armature_of(id) else {
            continue;
        };
        let posed = kreate_scene::armature::posed(armature, &object.pose);
        let mut made = Vec::with_capacity(armature.bones.len());
        for (i, bone) in armature.bones.iter().enumerate() {
            let (parent, relative) = match bone.parent {
                Some(p) => (made.get(p).copied(), posed[p].inverse() * posed[i]),
                None => (node_of.get(&id).copied(), posed[i]),
            };
            made.push(nodes.len());
            nodes.push(ExportNode {
                name: bone.name.clone(),
                parent,
                transform: kreate_math::DTransform::from_affine(relative),
                data: ImportNodeData::Empty,
                skin: None,
            });
        }
        bone_nodes.insert(id, made);
    }

    // Skins: each bound mesh names its armature's bones as joints.
    let mut skins = Vec::new();
    for (id, object) in doc.objects() {
        let Some(rig) = object.skin.and_then(|uid| doc.object_by_uid(uid)) else {
            continue;
        };
        let (Some((_, armature)), Some(joints), Some(node)) =
            (doc.armature_of(rig), bone_nodes.get(&rig), node_of.get(&id))
        else {
            continue;
        };
        let rig_world = doc.world_transform(rig);
        let mesh_world = doc.world_transform(id);
        nodes[*node].skin = Some(skins.len());
        skins.push(crate::ir::ExportSkin {
            name: doc.object(rig).map(|o| o.name.clone()).unwrap_or_default(),
            joints: joints.clone(),
            joint_names: armature.bones.iter().map(|b| b.name.clone()).collect(),
            inverse_bind: armature
                .bones
                .iter()
                .map(|b| (rig_world * b.rest()).inverse() * mesh_world)
                .collect(),
        });
    }

    let animations = export_animations(doc, &bone_nodes, &mut notes);

    (
        ExportScene {
            meshes,
            materials,
            images,
            mesh_slots,
            lights,
            cameras,
            nodes,
            animations,
            skins,
        },
        notes,
    )
}

/// The objects' actions as one clip, sampled from the document itself.
///
/// One clip for the scene rather than one per action, because that is what
/// plays: a viewer given several clips plays one, and an object missing from
/// it stands still.
///
/// Holds and straight lines on location and scale are exact on the union of
/// the keys, and written only there. Anything else — a Bézier, or a rotation,
/// which glTF interpolates as a quaternion and the document as Euler angles —
/// is sampled every frame, which is what Blender's exporter does by default.
///
/// Frame `start` is second zero, so a clip reimported into a scene with the
/// same range and rate lands on the frames it left from.
fn export_animations(
    doc: &Document,
    bone_nodes: &std::collections::HashMap<ObjectId, Vec<usize>>,
    notes: &mut Vec<LossyNote>,
) -> Vec<crate::ir::ExportAnimation> {
    use kreate_scene::animation::Interpolation;
    const MOVES: [&str; 3] = ["location", "rotation_euler", "scale"];
    let fps = if doc.playback.fps > 0.0 {
        doc.playback.fps
    } else {
        24.0
    };
    // Keys before the range would be negative seconds; the clip starts at
    // whichever is earlier.
    let mut origin = doc.playback.start as f64;
    for (_, object) in doc.objects() {
        if let Some(action) = object.action.and_then(|a| doc.action(a)) {
            // Bones' keys too: their tracks share the clip's clock.
            for curve in action.curves.iter().filter(|c| {
                MOVES.contains(&c.path.property.as_ref())
                    || kreate_scene::animation::parse_bone_path(&c.path.property).is_some()
            }) {
                if let Some(first) = curve.keys.first() {
                    origin = origin.min(first.frame);
                }
            }
        }
    }
    if origin < doc.playback.start as f64 {
        notes.push(LossyNote::Unsupported {
            what: format!(
                "keys before frame {} moved the clip's start to frame {origin}",
                doc.playback.start
            ),
        });
    }
    let mut tracks = Vec::new();
    for (index, (_, object)) in doc.objects().enumerate() {
        let Some(action) = object.action.and_then(|a| doc.action(a)) else {
            continue;
        };
        let curves: Vec<_> = action
            .curves
            .iter()
            .filter(|c| MOVES.contains(&c.path.property.as_ref()) && !c.keys.is_empty())
            .collect();
        if curves.is_empty() {
            continue;
        }
        let moves = |p: &str| curves.iter().any(|c| c.path.property.as_ref() == p);
        // The interpolation that matters is each key's but the last's: it is
        // how the curve leaves a key, and nothing leaves the last one.
        let all = |kind: Interpolation| {
            curves.iter().all(|c| {
                c.keys[..c.keys.len() - 1]
                    .iter()
                    .all(|k| k.interpolation == kind)
            })
        };
        let step = all(Interpolation::Constant);
        let exact = step || (all(Interpolation::Linear) && !moves("rotation_euler"));
        let mut frames: Vec<f64> = curves
            .iter()
            .flat_map(|c| c.keys.iter().map(|k| k.frame))
            .collect();
        frames.sort_by(f64::total_cmp);
        frames.dedup_by(|a, b| (*a - *b).abs() < kreate_scene::animation::SAME_FRAME);
        if !exact {
            let (first, last) = (frames[0], frames[frames.len() - 1]);
            let mut f = first.ceil();
            while f < last {
                frames.push(f);
                f += 1.0;
            }
            frames.sort_by(f64::total_cmp);
            frames.dedup_by(|a, b| (*a - *b).abs() < kreate_scene::animation::SAME_FRAME);
        }
        let transforms = frames
            .iter()
            .map(|frame| {
                let mut t = object.transform;
                kreate_scene::animation::apply_to_object(action, *frame, &mut t);
                // Folded as the rest placement is: see `ExportNode::transform`.
                if object.parent.is_some() {
                    kreate_math::DTransform::from_affine(object.parent_inverse * t.to_affine())
                } else {
                    t
                }
            })
            .collect();
        tracks.push(crate::ir::ExportTrack {
            node: index,
            times: frames.iter().map(|f| ((f - origin) / fps) as f32).collect(),
            transforms,
            step,
            translation: moves("location"),
            rotation: moves("rotation_euler"),
            scale: moves("scale"),
        });
    }
    // Bones: every frame of the keyed range, each bone's placement under its
    // parent as the document poses it — IK and constraints included, which a
    // file has no place for and a viewer then plays as they looked.
    for (id, object) in doc.objects() {
        let (Some(action), Some((_, armature)), Some(joints)) = (
            object.action.and_then(|a| doc.action(a)),
            doc.armature_of(id),
            bone_nodes.get(&id),
        ) else {
            continue;
        };
        let keyed: Vec<f64> = action
            .curves
            .iter()
            .filter(|c| kreate_scene::animation::parse_bone_path(&c.path.property).is_some())
            .flat_map(|c| c.keys.iter().map(|k| k.frame))
            .collect();
        let (Some(first), Some(last)) = (
            keyed.iter().copied().reduce(f64::min),
            keyed.iter().copied().reduce(f64::max),
        ) else {
            continue;
        };
        let mut frames = vec![first];
        let mut f = first.floor() + 1.0;
        while f < last {
            frames.push(f);
            f += 1.0;
        }
        if last > first {
            frames.push(last);
        }
        let poses: Vec<Vec<kreate_math::DAffine3>> = frames
            .iter()
            .map(|frame| {
                let mut pose = object.pose.clone();
                kreate_scene::animation::apply_to_pose(action, *frame, &mut pose);
                kreate_scene::armature::posed(armature, &pose)
            })
            .collect();
        for (i, bone) in armature.bones.iter().enumerate() {
            let Some(node) = joints.get(i) else { continue };
            tracks.push(crate::ir::ExportTrack {
                node: *node,
                times: frames.iter().map(|f| ((f - origin) / fps) as f32).collect(),
                transforms: poses
                    .iter()
                    .map(|p| {
                        kreate_math::DTransform::from_affine(match bone.parent {
                            Some(parent) => p[parent].inverse() * p[i],
                            None => p[i],
                        })
                    })
                    .collect(),
                step: false,
                translation: true,
                rotation: true,
                scale: true,
            });
        }
    }
    for (_, o) in doc.objects() {
        if !o.drivers.is_empty() {
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: its drivers have no place in glTF; it was written as it stands now",
                    o.name
                ),
            });
        }
    }
    for (_, m) in doc.materials() {
        if m.action.is_some() {
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: a material's animation has no place in glTF and was not written",
                    m.name
                ),
            });
        }
    }
    for (_, l) in doc.lights() {
        if l.action.is_some() {
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: a light's animation has no place in glTF and was not written",
                    l.name
                ),
            });
        }
    }
    if tracks.is_empty() {
        return Vec::new();
    }
    vec![crate::ir::ExportAnimation {
        name: "Animation".into(),
        tracks,
    }]
}

/// One light, in the units an interchange format writes.
///
/// §2.2's decision paying off in the direction it was made for: the datablock's
/// candela and lux go out unconverted, so a file that comes back says the same
/// numbers it arrived with.
fn export_light(light: &kreate_scene::Light, notes: &mut Vec<LossyNote>) -> ImportLight {
    let kind = match light.kind {
        kreate_scene::LightKind::Point { .. } => ImportLightKind::Point,
        kreate_scene::LightKind::Sun { .. } => ImportLightKind::Directional,
        kreate_scene::LightKind::Spot {
            cone_angle, blend, ..
        } => {
            let outer = (cone_angle * 0.5) as f32;
            ImportLightKind::Spot {
                // The inverse of what `build_into` does on the way in: a full
                // cone and a blend fraction become two half-angles. Exact in
                // both directions, which is why the shape survives even though
                // the parameterisation does not.
                inner: outer * (1.0 - blend.clamp(0.0, 1.0) as f32),
                outer,
            }
        }
        kreate_scene::LightKind::Area { .. } => {
            // glTF has no area light, and the nearest thing that is not a lie
            // is the point light §7.3 already reduces it to for the raster
            // pass: the same axial candela from the same place. What is lost is
            // the shape, and with it the soft shadow that was the reason to
            // reach for an area light.
            notes.push(LossyNote::Unsupported {
                what: format!(
                    "{}: an area light was written as a point light of the same axial intensity",
                    light.name
                ),
            });
            ImportLightKind::Point
        }
    };
    if light.exposure != 0.0 {
        // glTF has one number where the datablock has two. Folded rather than
        // dropped — the amount of light is what the file is for — and said,
        // because a round trip brings back the product with the stops at zero.
        notes.push(LossyNote::Unsupported {
            what: format!(
                "{}: its {:+.2} stops were folded into the intensity",
                light.name, light.exposure
            ),
        });
    }
    if light.kind.is_soft() {
        notes.push(LossyNote::Unsupported {
            what: format!(
                "{}: the source's size was dropped; glTF's lights have no size",
                light.name
            ),
        });
    }
    ImportLight {
        name: light.name.clone(),
        kind,
        color: [light.color.r, light.color.g, light.color.b],
        // An area light's nits become candela here, which is the one kind whose
        // unit changes on the way out — because the kind it is being written as
        // is a different kind.
        intensity: light.axial_candela().unwrap_or(light.effective_intensity()) as f32,
        // No cutoff. The datablock has no artist-set distance yet (§7.8), and
        // the one the application derives from the display's black point is a
        // property of how the picture is being shown, not of the lamp.
        range: None,
    }
}

/// One camera, as much of it as glTF holds.
fn export_camera(camera: &kreate_scene::Camera) -> ImportCamera {
    // The camera's own sensor is the frame to measure the angle against. Any
    // other aspect would be the aspect of whatever this session happens to be
    // rendering, which is not a property of the camera.
    let aspect = camera.sensor_width_mm / camera.sensor_height_mm;
    let perspective = matches!(camera.projection, kreate_scene::Projection::Perspective);
    ImportCamera {
        name: camera.name.clone(),
        fov_y: perspective.then(|| camera.fov_y(aspect) as f32),
        ortho_width: (!perspective).then_some(camera.ortho_scale as f32),
        clip_near: camera.clip_near as f32,
        clip_far: Some(camera.clip_far as f32),
    }
}
