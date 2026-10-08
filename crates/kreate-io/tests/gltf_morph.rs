//! Shape keys through glTF and back, as morph targets.
//!
//! What a user would check: the keys come back with their names, turned up
//! as far as they were, and the surface is where it was at every frame of
//! their animation.

use kreate_io::build::build_into;
use kreate_io::gltf_import::{parse_gltf, GltfOptions};
use kreate_math::{DQuat, DTransform, DVec3};
use kreate_scene::animation::{shape_path, Action, Interpolation};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::modifier::{EvalCtx, ModifierCache};
use kreate_scene::shape::{layer, ShapeKey};
use kreate_scene::Document;

/// A turned, scaled sphere with two keys: one that stretches the top up
/// (turned a quarter on), one that widens it along X (animated off to on).
fn shaped() -> (Document, ObjectId) {
    let mut doc = Document::new();
    let mut mesh = kreate_mesh::primitives::uv_sphere(1.0, 12, 8);
    let n = mesh.num_verts() as usize;
    let tall: Vec<glam::Vec3> = mesh
        .positions()
        .iter()
        .map(|p| glam::Vec3::new(0.0, 0.3 * p.y, (p.z.max(0.0)) * 0.8))
        .collect();
    let wide: Vec<glam::Vec3> = mesh
        .positions()
        .iter()
        .map(|p| glam::Vec3::new(p.x * 0.5, 0.0, 0.0))
        .collect();
    for (name, offsets) in [("Tall", tall), ("Wide", wide)] {
        assert_eq!(offsets.len(), n);
        mesh.attributes_mut().insert(
            &layer(name),
            kreate_mesh::attribute::Domain::Point,
            kreate_mesh::attribute::AttrData::F32x3(std::sync::Arc::new(offsets)),
        );
    }
    let mesh = doc.add_mesh(mesh);
    let mut object = Object::new("Blob", ObjectData::Mesh(mesh));
    object.transform = DTransform {
        translation: DVec3::new(1.0, 2.0, 0.5),
        rotation: DQuat::from_rotation_x(0.7),
        scale: DVec3::new(1.0, 2.0, 1.0),
    };
    object.shape_keys = vec![
        ShapeKey {
            name: "Tall".into(),
            value: 0.25,
        },
        ShapeKey {
            name: "Wide".into(),
            value: 0.0,
        },
    ];
    let mut action = Action::new("Breathe");
    for (frame, v) in [(1.0, 0.0), (10.0, 1.0)] {
        action
            .curve_mut(&shape_path("Wide"), 0)
            .insert(frame, v, Interpolation::Linear);
    }
    object.action = Some(doc.add_action(action));
    let id = doc.add_object(object);
    (doc, id)
}

fn surface(doc: &mut Document, id: ObjectId, frame: f64) -> Vec<DVec3> {
    doc.animate(frame);
    let mesh = ModifierCache::new()
        .evaluated(doc, id, &EvalCtx::still())
        .expect("a mesh");
    let world = doc.world_transform(id);
    mesh.positions()
        .iter()
        .map(|p| world.transform_point3(p.as_dvec3()))
        .collect()
}

fn apart(a: &[DVec3], b: &[DVec3]) -> f64 {
    let one_way = |a: &[DVec3], b: &[DVec3]| {
        a.iter()
            .map(|p| b.iter().map(|q| p.distance(*q)).fold(f64::MAX, f64::min))
            .fold(0.0, f64::max)
    };
    one_way(a, b).max(one_way(b, a))
}

fn written(doc: &Document, binary: bool) -> Vec<u8> {
    let (scene, _) = kreate_io::scene_from_document(doc);
    let options = kreate_io::GltfExportOptions {
        binary,
        ..Default::default()
    };
    kreate_io::write_gltf_scene(&scene, &options).0
}

#[test]
fn shape_keys_come_back_named_turned_up_and_moving() {
    let (mut doc, id) = shaped();
    let read = parse_gltf(&written(&doc, true), &GltfOptions::default()).expect("parses");
    let mut back = Document::new();
    let (made, _) = build_into(&read, &mut back);
    let blob = *made
        .iter()
        .find(|m| back.object(**m).is_some_and(|o| o.name == "Blob"))
        .expect("the object came back");
    let keys = back.object(blob).unwrap().shape_keys.clone();
    assert_eq!(
        keys.iter().map(|k| k.name.as_str()).collect::<Vec<_>>(),
        ["Tall", "Wide"]
    );
    assert!((keys[0].value - 0.25).abs() < 1e-6, "{keys:?}");
    for frame in [1.0, 4.0, 10.0] {
        let want = surface(&mut doc, id, frame);
        let got = surface(&mut back, blob, frame);
        let d = apart(&want, &got);
        assert!(d < 1e-4, "frame {frame}: the surface moved by {d}");
    }
    // The keys really change the surface, or the comparison says nothing.
    let a = surface(&mut doc, id, 1.0);
    let b = surface(&mut doc, id, 10.0);
    assert!(apart(&a, &b) > 0.2);
}

#[test]
fn the_file_names_its_targets_and_animates_their_weights() {
    let (doc, _) = shaped();
    let json: serde_json::Value = serde_json::from_slice(&written(&doc, false)).unwrap();
    let mesh = &json["meshes"][0];
    assert_eq!(
        mesh["extras"]["targetNames"],
        serde_json::json!(["Tall", "Wide"])
    );
    assert_eq!(
        mesh["primitives"][0]["targets"].as_array().unwrap().len(),
        2
    );
    let node = json["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["name"] == "Blob")
        .unwrap();
    assert_eq!(node["weights"], serde_json::json!([0.25, 0.0]));
    let channels = json["animations"][0]["channels"].as_array().unwrap();
    assert!(channels.iter().any(|c| c["target"]["path"] == "weights"));
}

#[test]
fn a_still_key_comes_back_as_far_on_as_it_was() {
    // Without an action, so the value can only come from the node's weights:
    // a clip writes every target's weight and would bring it back anyway.
    let (mut doc, id) = shaped();
    doc.object_mut(id).unwrap().action = None;
    let read = parse_gltf(&written(&doc, true), &GltfOptions::default()).expect("parses");
    let mut back = Document::new();
    let (made, _) = build_into(&read, &mut back);
    let blob = *made
        .iter()
        .find(|m| back.object(**m).is_some_and(|o| o.name == "Blob"))
        .unwrap();
    let keys = &back.object(blob).unwrap().shape_keys;
    assert_eq!((keys[0].value, keys[1].value), (0.25, 0.0));
}
