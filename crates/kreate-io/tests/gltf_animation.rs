//! Movement through a glTF file and back.
//!
//! The document keys location, XYZ Euler angles and scale on Z-up; glTF keys a
//! node's translation, quaternion and scale on Y-up, in seconds. Most of what
//! can go wrong is in that gap, so most of these ask the only question that
//! does not care how the gap was crossed: *is the object in the same place at
//! the same frame?*

use kreate_io::build::{build_into, BuildNote};
use kreate_io::gltf_import::{parse_gltf, GltfOptions};
use kreate_io::ir::{
    ChannelInterpolation, ChannelValues, ImportAnimation, ImportChannel, ImportNode, ImportScene,
};
use kreate_io::SourceMetadata;
use kreate_math::{DQuat, DVec3};
use kreate_scene::animation::{Action, Interpolation};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::Document;

fn round_trip(doc: &Document) -> (Document, Vec<ObjectId>, Vec<u8>) {
    let (scene, _) = kreate_io::scene_from_document(doc);
    let (bytes, _) = kreate_io::write_gltf_scene(&scene, &kreate_io::GltfExportOptions::default());
    let read = parse_gltf(&bytes, &GltfOptions::default()).expect("what we wrote parses");
    let mut out = Document::new();
    let (made, _) = build_into(&read, &mut out);
    (out, made, bytes)
}

/// The same written as JSON, to read what the file says.
fn json_of(doc: &Document) -> serde_json::Value {
    let (scene, _) = kreate_io::scene_from_document(doc);
    let options = kreate_io::GltfExportOptions {
        binary: false,
        ..Default::default()
    };
    let (bytes, _) = kreate_io::write_gltf_scene(&scene, &options);
    serde_json::from_slice(&bytes).expect("json")
}

/// An empty object with an action, and the action's id.
fn animated(doc: &mut Document, name: &str, action: Action) -> ObjectId {
    let id = doc.add_action(action);
    let mut object = Object::new(name, ObjectData::Empty);
    object.action = Some(id);
    doc.add_object(object)
}

/// Where each object is at each frame, in both documents, compared.
fn same_motion(
    a: &mut Document,
    a_ids: &[ObjectId],
    b: &mut Document,
    b_ids: &[ObjectId],
    frames: &[f64],
) {
    for frame in frames {
        a.animate(*frame);
        b.animate(*frame);
        for (x, y) in a_ids.iter().zip(b_ids) {
            let (p, q) = (a.world_transform(*x), b.world_transform(*y));
            assert!(
                (p.translation - q.translation).length() < 1e-4,
                "frame {frame}: was at {:?}, came back at {:?}",
                p.translation,
                q.translation
            );
            let (rp, rq) = (
                p.to_scale_rotation_translation().1,
                q.to_scale_rotation_translation().1,
            );
            assert!(
                rp.dot(rq).abs() > 1.0 - 1e-6,
                "frame {frame}: turned {rp:?} and came back turned {rq:?}"
            );
        }
    }
}

fn curve_frames(
    doc: &Document,
    object: ObjectId,
    property: &str,
    index: u8,
) -> Vec<(f64, f64, Interpolation)> {
    let action = doc
        .object(object)
        .and_then(|o| o.action)
        .and_then(|a| doc.action(a))
        .expect("an action");
    let curve = action
        .curves
        .iter()
        .find(|c| c.path.property.as_ref() == property && c.path.index == Some(index))
        .unwrap_or_else(|| panic!("no {property}[{index}] curve"));
    curve
        .keys
        .iter()
        .map(|k| (k.frame, k.value, k.interpolation))
        .collect()
}

#[test]
fn straight_lines_come_back_on_the_same_keys() {
    // Exact, not sampled: a line between two keys is a line in either axis
    // convention, so the file needs only the keys and the reimport has only
    // those. Extra keys every frame would be a curve nobody can edit.
    let mut doc = Document::new();
    let mut action = Action::new("Slide");
    for (frame, x, y) in [(1.0, 0.0, 0.0), (13.0, 2.0, 1.0), (25.0, 2.0, -3.0)] {
        action
            .curve_mut("location", 0)
            .insert(frame, x, Interpolation::Linear);
        action
            .curve_mut("location", 1)
            .insert(frame, y, Interpolation::Linear);
    }
    let a = animated(&mut doc, "Slider", action);
    let (mut out, made, _) = round_trip(&doc);

    let x = curve_frames(&out, made[0], "location", 0);
    assert_eq!(
        x.iter().map(|k| k.0).collect::<Vec<_>>(),
        vec![1.0, 13.0, 25.0],
        "keys were added or lost"
    );
    assert!(x.iter().all(|k| k.2 == Interpolation::Linear));
    assert!((x[1].1 - 2.0).abs() < 1e-5);
    // Y is the one the axis change moves about: Kreate's Y is glTF's -Z.
    let y = curve_frames(&out, made[0], "location", 1);
    assert!((y[2].1 + 3.0).abs() < 1e-5, "Y came back as {}", y[2].1);
    same_motion(
        &mut doc,
        &[a],
        &mut out,
        &made,
        &[1.0, 5.0, 13.0, 20.5, 25.0, 40.0],
    );
}

#[test]
fn a_full_turn_comes_back_a_full_turn() {
    // The trap on both sides. On the way out a quaternion that changes sign
    // between samples is a reader slerping the long way; on the way in the
    // plain Euler conversion answers within ±π, so a spin past half a turn
    // would jump back by a whole one and the curve would unwind.
    let mut doc = Document::new();
    let mut action = Action::new("Spin");
    action
        .curve_mut("rotation_euler", 2)
        .insert(1.0, 0.0, Interpolation::Linear);
    action.curve_mut("rotation_euler", 2).insert(
        25.0,
        std::f64::consts::TAU,
        Interpolation::Linear,
    );
    let a = animated(&mut doc, "Turntable", action);
    let (mut out, made, _) = round_trip(&doc);

    let z = curve_frames(&out, made[0], "rotation_euler", 2);
    let last = z.last().expect("keys").1;
    assert!(
        (last - std::f64::consts::TAU).abs() < 1e-3,
        "a full turn came back as {last}"
    );
    // And every step is small: no key jumps back round.
    for pair in z.windows(2) {
        assert!(
            (pair[1].1 - pair[0].1).abs() < 0.5,
            "a jump between {:?} and {:?}",
            pair[0],
            pair[1]
        );
    }
    let frames: Vec<f64> = (1..=25).map(f64::from).collect();
    same_motion(&mut doc, &[a], &mut out, &made, &frames);
}

#[test]
fn a_bezier_is_sampled_and_plays_the_same() {
    let mut doc = Document::new();
    let mut action = Action::new("Bounce");
    for (frame, z) in [(1.0, 0.0), (12.0, 3.0), (24.0, 0.0)] {
        action
            .curve_mut("location", 2)
            .insert(frame, z, Interpolation::Bezier);
        action
            .curve_mut("scale", 0)
            .insert(frame, 1.0 + z * 0.1, Interpolation::Bezier);
    }
    let a = animated(&mut doc, "Ball", action);
    let (mut out, made, _) = round_trip(&doc);
    // Sampled on whole frames, where the document is exact; between them the
    // lines are an approximation and asserted as one.
    let frames: Vec<f64> = (1..=24).map(f64::from).collect();
    same_motion(&mut doc, &[a], &mut out, &made, &frames);
    let z = curve_frames(&out, made[0], "location", 2);
    assert_eq!(z.len(), 24, "one key a frame from the first to the last");
}

#[test]
fn a_hold_stays_a_hold() {
    let mut doc = Document::new();
    let mut action = Action::new("Blink");
    action
        .curve_mut("scale", 0)
        .insert(1.0, 1.0, Interpolation::Constant);
    action
        .curve_mut("scale", 0)
        .insert(10.0, 0.2, Interpolation::Constant);
    let a = animated(&mut doc, "Eye", action);
    let (mut out, made, _) = round_trip(&doc);
    let x = curve_frames(&out, made[0], "scale", 0);
    assert_eq!(x.len(), 2);
    assert_eq!(x[0].2, Interpolation::Constant);
    out.animate(9.5);
    let held = out.object(made[0]).expect("the eye").transform.scale.x;
    assert!((held - 1.0).abs() < 1e-5, "the hold became a ramp: {held}");
    same_motion(
        &mut doc,
        &[a],
        &mut out,
        &made,
        &[1.0, 5.0, 9.9, 10.0, 12.0],
    );
}

#[test]
fn a_child_of_a_turning_parent_moves_the_same_in_the_world() {
    // The parent's inverse has to be folded into every sample the way it is
    // into the rest placement, or a parented object's track is in a different
    // space from its own node.
    let mut doc = Document::new();
    let mut spin = Action::new("Spin");
    spin.curve_mut("rotation_euler", 2)
        .insert(1.0, 0.0, Interpolation::Linear);
    spin.curve_mut("rotation_euler", 2)
        .insert(20.0, 1.5, Interpolation::Linear);
    let mut parent_object = Object::new("Turntable", ObjectData::Empty);
    parent_object.transform.translation = DVec3::new(3.0, 0.0, 0.0);
    parent_object.transform.rotation = DQuat::from_rotation_z(0.7);
    parent_object.action = Some(doc.add_action(spin));
    let parent = doc.add_object(parent_object);
    let mut lift = Action::new("Lift");
    lift.curve_mut("location", 2)
        .insert(1.0, 0.0, Interpolation::Linear);
    lift.curve_mut("location", 2)
        .insert(20.0, 2.0, Interpolation::Linear);
    let mut child_object = Object::new("Arm", ObjectData::Empty);
    child_object.transform.translation = DVec3::new(0.0, 2.0, 0.0);
    child_object.action = Some(doc.add_action(lift));
    let child = doc.add_object(child_object);
    doc.animate(1.0);
    doc.set_parent(child, Some(parent)).expect("a fresh chain");

    let (mut out, made, _) = round_trip(&doc);
    same_motion(
        &mut doc,
        &[parent, child],
        &mut out,
        &made,
        &[1.0, 7.0, 14.0, 20.0],
    );
}

#[test]
fn the_file_says_what_moves_and_nothing_else() {
    // Checked in the JSON, because our own reader would accept a file that
    // other readers refuse: a sampler input without its min and max, or a
    // rotation channel for an object that only slides.
    let mut doc = Document::new();
    let mut action = Action::new("Slide");
    action
        .curve_mut("location", 0)
        .insert(1.0, 0.0, Interpolation::Linear);
    action
        .curve_mut("location", 0)
        .insert(49.0, 1.0, Interpolation::Linear);
    animated(&mut doc, "Slider", action);
    doc.add_object(Object::new("Still", ObjectData::Empty));
    let json = json_of(&doc);
    let animations = json["animations"].as_array().expect("an animations table");
    assert_eq!(animations.len(), 1, "one clip for the scene");
    let channels = animations[0]["channels"].as_array().expect("channels");
    assert_eq!(
        channels.len(),
        1,
        "only the translation moves: {channels:?}"
    );
    assert_eq!(channels[0]["target"]["path"], "translation");
    assert_eq!(channels[0]["target"]["node"], 0);
    let sampler = &animations[0]["samplers"][0];
    let input = &json["accessors"][sampler["input"].as_u64().expect("an input") as usize];
    // Frame 1 is second zero; frame 49 at 24 a second is second two.
    assert_eq!(input["min"][0].as_f64(), Some(0.0));
    assert!((input["max"][0].as_f64().expect("a max") - 2.0).abs() < 1e-6);
    assert_eq!(sampler["interpolation"], "LINEAR");
}

#[test]
fn a_still_scene_writes_no_animations() {
    let mut doc = Document::new();
    doc.add_object(Object::new("Still", ObjectData::Empty));
    let json = json_of(&doc);
    assert!(json
        .get("animations")
        .is_none_or(|a| a.as_array().is_some_and(|a| a.is_empty())));
}

#[test]
fn a_materials_animation_is_said_to_be_left_behind() {
    let mut doc = Document::new();
    let mut action = Action::for_kind("Fade", kreate_scene::DatablockKind::Material);
    action
        .curve_mut("alpha", 0)
        .insert(1.0, 1.0, Interpolation::Linear);
    let action = doc.add_action(action);
    doc.add_material(kreate_scene::Material {
        name: "Glass".into(),
        action: Some(action),
        ..Default::default()
    });
    let (_, notes) = kreate_io::scene_from_document(&doc);
    assert!(
        notes.iter().any(|n| format!("{n:?}").contains("Glass")),
        "nothing said the material's animation stayed behind: {notes:?}"
    );
}

#[test]
fn a_long_clip_stretches_the_range_and_a_short_one_does_not_shrink_it() {
    let mut doc = Document::new();
    let mut action = Action::new("Long");
    action
        .curve_mut("location", 0)
        .insert(1.0, 0.0, Interpolation::Linear);
    action
        .curve_mut("location", 0)
        .insert(400.0, 1.0, Interpolation::Linear);
    animated(&mut doc, "Walker", action);
    let (out, _, _) = round_trip(&doc);
    assert_eq!(out.playback.end, 400);

    let mut doc = Document::new();
    let mut action = Action::new("Short");
    action
        .curve_mut("location", 0)
        .insert(1.0, 0.0, Interpolation::Linear);
    action
        .curve_mut("location", 0)
        .insert(10.0, 1.0, Interpolation::Linear);
    animated(&mut doc, "Hopper", action);
    let (out, _, _) = round_trip(&doc);
    assert_eq!(
        out.playback.end, 250,
        "a short clip shrank the author's range"
    );
}

/// A scene of one Y-up node and one clip, by hand.
fn one_node(channels: Vec<ImportChannel>) -> ImportScene {
    let mut source = SourceMetadata::new("gltf");
    source.axis = kreate_math::AxisConvention::GLTF;
    let mut scene = ImportScene::new(source);
    scene.nodes = vec![ImportNode::new("Thing")];
    scene.animations = vec![ImportAnimation {
        name: "Clip".into(),
        channels,
    }];
    scene
}

#[test]
fn a_cubic_spline_is_followed_between_its_keys() {
    // Hermite in the file's seconds: one key at rest leaving upwards at 3 units
    // a second, one at one unit arriving flat. Halfway the spline says
    // 0.5 + (3 - 0) * 1/8 = 0.875, which a straight line (0.5) does not.
    let channel = ImportChannel {
        node: 0,
        interpolation: ChannelInterpolation::CubicSpline,
        times: vec![0.0, 1.0],
        values: ChannelValues::Translation(vec![
            glam::Vec3::ZERO,
            glam::Vec3::ZERO,
            glam::Vec3::new(0.0, 3.0, 0.0),
            glam::Vec3::ZERO,
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::ZERO,
        ]),
    };
    assert!((channel.vec3_at(0.5).expect("a value").y - 0.875).abs() < 1e-6);
    let mut doc = Document::new();
    let (made, _) = build_into(&one_node(vec![channel]), &mut doc);
    // Second 0.5 at 24 a second from frame 1 is frame 13; glTF's Y is Z here.
    doc.animate(13.0);
    let z = doc
        .object(made[0])
        .expect("the thing")
        .transform
        .translation
        .z;
    assert!((z - 0.875).abs() < 1e-5, "the spline was drawn as {z}");
}

#[test]
fn a_second_clip_on_the_same_node_is_said_and_left() {
    let clip = |name: &str, x: f32| ImportAnimation {
        name: name.into(),
        channels: vec![ImportChannel {
            node: 0,
            interpolation: ChannelInterpolation::Linear,
            times: vec![0.0, 1.0],
            values: ChannelValues::Translation(vec![
                glam::Vec3::ZERO,
                glam::Vec3::new(x, 0.0, 0.0),
            ]),
        }],
    };
    let mut scene = one_node(Vec::new());
    scene.animations = vec![clip("Walk", 1.0), clip("Run", 5.0)];
    let mut doc = Document::new();
    let (made, notes) = build_into(&scene, &mut doc);
    assert!(notes.contains(&BuildNote::LaterClip {
        animation: "Run".into(),
        node: "Thing".into()
    }));
    doc.animate(25.0);
    let x = doc
        .object(made[0])
        .expect("the thing")
        .transform
        .translation
        .x;
    assert!((x - 1.0).abs() < 1e-5, "the second clip won: {x}");
}
