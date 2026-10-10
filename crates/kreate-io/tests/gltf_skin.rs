//! A rigged, skinned, animated mesh through glTF and back.
//!
//! The claim is the one a user would check by eye: the mesh is where it was,
//! at rest and mid-pose, after the round trip. Compared vertex by vertex in
//! the world, because a file is free to reorder the vertices and to place the
//! skeleton differently — what it may not do is put the surface elsewhere.

use kreate_io::build::build_into;
use kreate_io::gltf_import::{parse_gltf, GltfOptions};
use kreate_math::{DAffine3, DQuat, DTransform, DVec3};
use kreate_mesh::attribute::{AttrData, Domain};
use kreate_scene::animation::{bone_path, Action, Interpolation};
use kreate_scene::armature::{Armature, Bone};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::modifier::{EvalCtx, ModifierCache};
use kreate_scene::Document;

/// A three-bone arm, placed and turned in the world, with a column bound to
/// it, its weights from distance; the middle bone bends between frames 1 and
/// 10 and the hand turns about its own axis.
fn rigged() -> (Document, ObjectId) {
    let mut doc = Document::new();
    let mut arm = Armature::new("Arm");
    arm.bones[0].name = "Upper".into();
    arm.bones[0].roll = 0.4;
    for (name, parent, z) in [("Lower", 0, 2.0), ("Hand", 1, 3.0)] {
        arm.add_bone(Bone {
            name: name.into(),
            parent: Some(parent),
            head: DVec3::new(0.0, 0.0, z - 1.0),
            tail: DVec3::new(0.0, 0.1, z),
            roll: -0.7,
            connected: true,
            ..Default::default()
        });
    }
    arm.reconnect();
    let mut column = kreate_mesh::primitives::uv_sphere(0.5, 12, 18);
    for p in column.positions_mut() {
        p.x *= 0.4;
        p.y *= 0.4;
        p.z = (p.z + 0.5) * 3.0;
    }
    for (name, w) in kreate_scene::skin::automatic_weights(&column, &arm, DAffine3::IDENTITY) {
        column
            .attributes_mut()
            .insert(&name, Domain::Point, AttrData::F32(std::sync::Arc::new(w)));
    }
    let armature = doc.add_armature(arm);
    let mut rig = Object::new("Rig", ObjectData::Armature(armature));
    rig.transform = DTransform {
        translation: DVec3::new(1.0, -2.0, 0.5),
        rotation: DQuat::from_rotation_z(0.6),
        scale: DVec3::ONE,
    };
    let mut action = Action::new("Bend");
    for (frame, bend, twist) in [(1.0, 0.0, 0.0), (10.0, 1.1, 0.8)] {
        for (bone, q) in [
            ("Lower", DQuat::from_rotation_x(bend)),
            ("Hand", DQuat::from_rotation_y(twist)),
        ] {
            for (i, v) in [q.w, q.x, q.y, q.z].into_iter().enumerate() {
                action
                    .curve_mut(&bone_path(bone, "rotation_quaternion"), i as u8)
                    .insert(frame, v, Interpolation::Linear);
            }
        }
    }
    rig.action = Some(doc.add_action(action));
    let rig = doc.add_object(rig);
    let mesh = doc.add_mesh(column);
    let body = doc.add_object(Object::new("Body", ObjectData::Mesh(mesh)));
    doc.set_parent(body, Some(rig)).unwrap();
    // Placed off its armature, so the mesh-to-armature change is not the
    // identity — the term an inverse bind matrix carries for it.
    doc.object_mut(body).unwrap().transform = DTransform {
        translation: DVec3::new(0.3, -0.2, 0.4),
        rotation: DQuat::from_rotation_x(0.25),
        scale: DVec3::new(1.0, 1.2, 0.9),
    };
    let uid = doc.object(rig).unwrap().uid;
    doc.object_mut(body).unwrap().skin = Some(uid);
    (doc, body)
}

/// Where the mesh's vertices are in the world at `frame`, skinned.
fn surface(doc: &mut Document, body: ObjectId, frame: f64) -> Vec<DVec3> {
    doc.animate(frame);
    let mesh = ModifierCache::new()
        .evaluated(doc, body, &EvalCtx::still())
        .expect("a mesh");
    let world = doc.world_transform(body);
    mesh.positions()
        .iter()
        .map(|p| world.transform_point3(p.as_dvec3()))
        .collect()
}

/// The farthest any vertex of `a` is from the nearest of `b`, both ways.
fn apart(a: &[DVec3], b: &[DVec3]) -> f64 {
    let one_way = |a: &[DVec3], b: &[DVec3]| {
        a.iter()
            .map(|p| b.iter().map(|q| p.distance(*q)).fold(f64::MAX, f64::min))
            .fold(0.0, f64::max)
    };
    one_way(a, b).max(one_way(b, a))
}

fn round_trip(doc: &Document) -> (Document, Vec<ObjectId>) {
    let (scene, _) = kreate_io::scene_from_document(doc);
    let (bytes, _) = kreate_io::write_gltf_scene(&scene, &kreate_io::GltfExportOptions::default());
    let read = parse_gltf(&bytes, &GltfOptions::default()).expect("what we wrote parses");
    let mut out = Document::new();
    let (made, _) = build_into(&read, &mut out);
    (out, made)
}

#[test]
fn a_skinned_mesh_comes_back_where_it_was_at_rest_and_posed() {
    let (mut doc, body) = rigged();
    let (mut back, made) = round_trip(&doc);
    let body_back = *made
        .iter()
        .find(|id| back.object(**id).is_some_and(|o| o.name == "Body"))
        .expect("the mesh came back");
    assert!(back.object(body_back).unwrap().skin.is_some(), "and bound");
    let rig_back = back
        .object_by_uid(back.object(body_back).unwrap().skin.unwrap())
        .unwrap();
    let (_, arm) = back.armature_of(rig_back).expect("an armature");
    assert_eq!(
        arm.bones
            .iter()
            .map(|b| b.name.as_str())
            .collect::<Vec<_>>(),
        ["Upper", "Lower", "Hand"]
    );
    for frame in [1.0, 5.5, 10.0] {
        let want = surface(&mut doc, body, frame);
        let got = surface(&mut back, body_back, frame);
        let d = apart(&want, &got);
        assert!(d < 1e-3, "frame {frame}: the surface moved by {d}");
    }
    // And the pose really moves it, or the comparison above says nothing.
    let rest = surface(&mut doc, body, 1.0);
    let bent = surface(&mut doc, body, 10.0);
    assert!(apart(&rest, &bent) > 0.5);
}

#[test]
fn the_file_says_skin_joints_and_weights() {
    let (doc, _) = rigged();
    let (scene, _) = kreate_io::scene_from_document(&doc);
    let options = kreate_io::GltfExportOptions {
        binary: false,
        ..Default::default()
    };
    let (bytes, _) = kreate_io::write_gltf_scene(&scene, &options);
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let skins = json["skins"].as_array().expect("a skin");
    assert_eq!(skins.len(), 1);
    assert_eq!(skins[0]["joints"].as_array().unwrap().len(), 3);
    assert!(skins[0]["inverseBindMatrices"].is_u64());
    let body = json["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["name"] == "Body")
        .unwrap();
    assert_eq!(body["skin"], 0);
    let attributes = &json["meshes"][0]["primitives"][0]["attributes"];
    assert!(attributes["JOINTS_0"].is_u64() && attributes["WEIGHTS_0"].is_u64());
    // The joints are nodes named for the bones, under the armature's node.
    let names: Vec<&str> = skins[0]["joints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| {
            json["nodes"][j.as_u64().unwrap() as usize]["name"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(names, ["Upper", "Lower", "Hand"]);
    // The bones move in the clip.
    let channels = json["animations"][0]["channels"].as_array().unwrap();
    let joint_nodes: Vec<u64> = skins[0]["joints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j.as_u64().unwrap())
        .collect();
    assert!(channels
        .iter()
        .any(|c| joint_nodes.contains(&c["target"]["node"].as_u64().unwrap())));
}

/// A file with two takes on one rig — an idle and a walk, say — keeps both:
/// the first plays, the second is a muted strip beside it with its own
/// keys. Read together they were one action, the last take over the first.
#[test]
fn every_take_on_a_rig_is_kept_as_its_own_action() {
    let (doc, _) = rigged();
    let (scene, _) = kreate_io::scene_from_document(&doc);
    let (bytes, _) = kreate_io::write_gltf_scene(&scene, &kreate_io::GltfExportOptions::default());
    let mut read = parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    assert_eq!(read.animations.len(), 1, "the rig exports one take");
    // A second take: the same joints, twice as long.
    let mut second = read.animations[0].clone();
    second.name = "Slow".into();
    for channel in &mut second.channels {
        for t in &mut channel.times {
            *t *= 2.0;
        }
    }
    read.animations[0].name = "Bend".into();
    read.animations.push(second);
    let mut out = Document::new();
    build_into(&read, &mut out);

    let rig = out
        .objects()
        .find(|(_, o)| matches!(o.data, ObjectData::Armature(_)))
        .map(|(_, o)| o)
        .expect("a rig");
    let playing = out.action(rig.action.expect("an action")).expect("it");
    assert!(playing.name.ends_with("|Bend"), "{}", playing.name);
    assert_eq!(rig.nla.len(), 1, "the second take as a strip");
    let strip = &rig.nla[0];
    assert!(strip.mute, "the strip waits to be switched on");
    let slow = out.action(strip.action).expect("the strip's action");
    assert!(slow.name.ends_with("|Slow"), "{}", slow.name);
    // Its own keys: twice the frames of the first.
    let span = |a: &kreate_scene::animation::Action| {
        let frames: Vec<f64> = a
            .curves
            .iter()
            .flat_map(|c| c.keys.iter().map(|k| k.frame))
            .collect();
        frames.iter().copied().fold(f64::MIN, f64::max)
            - frames.iter().copied().fold(f64::MAX, f64::min)
    };
    let (fast, long) = (span(playing), span(slow));
    assert!((long / fast - 2.0).abs() < 0.2, "{fast} and {long} frames");
}

/// A skin's own node scaled to a hundredth, as Sketchfab's files hang
/// them: glTF draws a skin by its joints alone, so the surface is where it
/// was. Read against that node, the body stood a hundredth the size of its
/// bones.
#[test]
fn a_skins_own_node_does_not_place_it() {
    let (mut doc, body) = rigged();
    let (scene, _) = kreate_io::scene_from_document(&doc);
    let (bytes, _) = kreate_io::write_gltf_scene(&scene, &kreate_io::GltfExportOptions::default());
    let mut read = parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    let skinned = read
        .nodes
        .iter()
        .position(|n| n.skin.is_some())
        .expect("a skinned node");
    read.nodes[skinned].transform.scale *= 0.01;
    let mut back = Document::new();
    let (made, _) = build_into(&read, &mut back);
    let body_back = *made
        .iter()
        .find(|id| back.object(**id).is_some_and(|o| o.name == "Body"))
        .expect("the body");
    for frame in [1.0, 10.0] {
        let d = apart(
            &surface(&mut doc, body, frame),
            &surface(&mut back, body_back, frame),
        );
        assert!(d < 1e-3, "frame {frame}: {d}");
    }
}

/// A cube hung from the hand's joint, not skinned: it goes where the hand
/// goes, not staying behind while the arm bends.
#[test]
fn a_mesh_hung_from_a_joint_follows_its_bone() {
    let (doc, _) = rigged();
    let (scene, _) = kreate_io::scene_from_document(&doc);
    let (bytes, _) = kreate_io::write_gltf_scene(&scene, &kreate_io::GltfExportOptions::default());
    let mut read = parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    let hand = read
        .nodes
        .iter()
        .position(|n| n.name == "Hand")
        .expect("the hand joint");
    read.meshes.push(kreate_io::ir::ImportMesh {
        name: "Glove".into(),
        positions: vec![
            glam::Vec3::new(0.0, 0.0, 0.0),
            glam::Vec3::new(0.1, 0.0, 0.0),
            glam::Vec3::new(0.0, 0.1, 0.0),
        ],
        faces: vec![vec![0, 1, 2]],
        corner_uvs: None,
        corner_normals: None,
        face_materials: None,
        smoothing_groups: None,
    });
    let mut glove = kreate_io::ir::ImportNode::new("Glove");
    glove.parent = Some(hand);
    glove.data = kreate_io::ir::ImportNodeData::Mesh(read.meshes.len() - 1);
    read.nodes.push(glove);
    let glove_node = read.nodes.len() - 1;
    let mut back = Document::new();
    let (made, notes) = build_into(&read, &mut back);
    assert!(
        !notes
            .iter()
            .any(|n| format!("{n:?}").contains("HungFromJoint")),
        "{notes:?}"
    );
    let glove_id = *made
        .iter()
        .find(|id| back.object(**id).is_some_and(|o| o.name == "Glove"))
        .expect("the glove");
    // Where the file puts it at rest, and that it moves with the bend.
    let rest_world = read.world_transform(glove_node).expect("placed");
    let at = |back: &mut Document, frame: f64| surface(back, glove_id, frame)[0];
    let first = at(&mut back, 1.0);
    let want = rest_world.transform_point3(DVec3::ZERO);
    assert!(first.distance(want) < 1e-3, "{first} against {want}");
    assert!(
        at(&mut back, 10.0).distance(first) > 0.1,
        "it followed the hand"
    );
}
