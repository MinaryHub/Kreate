//! What a file's hierarchy is worth, and whether it survives.
//!
//! The IR had no scene graph: meshes arrived as `(name, MeshData)` pairs with
//! their placement dropped, and a light could not be imported at all because a
//! light *is* a placement and there was nowhere to put one. These check the
//! thing that was missing rather than the geometry, which other files already
//! cover.

use kreate_io::build::{build_into, BuildNote};
use kreate_io::gltf_import::{parse_gltf, GltfOptions};
use kreate_io::ir::{
    ImportLight, ImportLightKind, ImportNode, ImportNodeData, ImportScene, ImportTransform,
};
use kreate_io::SourceMetadata;
use kreate_scene::document::ObjectData;
use kreate_scene::Document;

/// A scene with a node hierarchy, built by hand.
///
/// By hand rather than through a file, because what is being checked is the
/// IR's own arithmetic: a format's parser has its own tests and would put a
/// second thing between the assertion and the rule.
fn a_hierarchy() -> ImportScene {
    let mut scene = ImportScene::new(SourceMetadata::new("test"));
    let mut root = ImportNode::new("Root");
    root.transform = ImportTransform {
        translation: glam::Vec3::new(1.0, 0.0, 0.0),
        ..ImportTransform::default()
    };
    let mut child = ImportNode::new("Child");
    child.parent = Some(0);
    child.transform = ImportTransform {
        translation: glam::Vec3::new(0.0, 2.0, 0.0),
        ..ImportTransform::default()
    };
    scene.nodes = vec![root, child];
    scene
}

#[test]
fn a_child_is_placed_by_its_parent_as_well_as_itself() {
    // The whole reason a hierarchy is worth importing. Baking the world
    // transform into each node would put everything in the right place once and
    // throw away the structure — and then moving the parent would move nothing.
    let scene = a_hierarchy();
    let world = scene.world_transform(1).expect("a chain");
    assert!(
        (world.translation - kreate_math::DVec3::new(1.0, 2.0, 0.0)).length() < 1e-9,
        "the child landed at {:?}",
        world.translation
    );

    // And the document keeps it as a link rather than as a baked number: the
    // child's *own* transform is still its own.
    let mut doc = Document::new();
    let (made, _) = build_into(&scene, &mut doc);
    let child = doc.object(made[1]).expect("the child");
    assert!(
        (child.transform.translation - kreate_math::DVec3::new(0.0, 2.0, 0.0)).length() < 1e-9,
        "the child's own transform was baked: {:?}",
        child.transform.translation
    );
    assert_eq!(child.parent, Some(made[0]));
    assert!(
        (doc.world_transform(made[1]).translation - kreate_math::DVec3::new(1.0, 2.0, 0.0))
            .length()
            < 1e-9
    );
}

#[test]
fn a_loop_is_reported_rather_than_followed() {
    // A file can say anything, and a parent chain that comes back round would
    // be an infinite walk. The node lands at the root with its own transform,
    // which is the honest partial answer.
    let mut scene = a_hierarchy();
    scene.nodes[0].parent = Some(1);
    assert!(scene.world_transform(1).is_none(), "the loop was followed");

    let mut doc = Document::new();
    let (_, notes) = build_into(&scene, &mut doc);
    assert!(
        notes
            .iter()
            .any(|n| matches!(n, BuildNote::BrokenParent { .. })),
        "a loop was built without a word: {notes:?}"
    );
}

#[test]
fn a_rotation_is_turned_into_kreate_space_and_not_merely_moved() {
    // The subtlety the axis change hides. A Y-up rotation has to be conjugated
    // — rotated into Kreate space and back out on the other side — or a
    // turntable in a glTF file spins about the floor.
    //
    // Checked by what it *does*: a quarter turn about the file's up axis has to
    // come out as a quarter turn about Kreate's up axis, which is what a
    // turntable means in both.
    let mut scene = ImportScene::new(SourceMetadata::new("test"));
    scene.source.axis = kreate_math::AxisConvention::GLTF;
    let mut node = ImportNode::new("Turntable");
    node.transform = ImportTransform {
        // A quarter turn about +Y, which is up in a glTF file.
        rotation: glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        ..ImportTransform::default()
    };
    scene.nodes = vec![node];

    let world = scene.world_transform(0).expect("a node");
    // Kreate is Z-up, so the same turn is about +Z. Following it through: a
    // quarter turn about glTF's +Y sends its +X to its -Z, and glTF's -Z is
    // Kreate's +Y — so the file's side axis ends up on Kreate's forward one,
    // which is a quarter turn about up in both.
    let turned = world.transform_vector3(kreate_math::DVec3::X);
    assert!(
        (turned - kreate_math::DVec3::Y).length() < 1e-6,
        "a quarter turn about up came out as {turned:?}"
    );
    // And it is still a rotation: the axis change must not have left a scale
    // behind.
    assert!((turned.length() - 1.0).abs() < 1e-6);
}

#[test]
fn a_light_arrives_with_its_units_unchanged() {
    // §2.2 chose glTF's units for the datablock so that this boundary is a copy
    // rather than a conversion — a conversion is where rounding accumulates
    // across a round trip.
    let mut scene = ImportScene::new(SourceMetadata::new("test"));
    scene.lights.push(ImportLight {
        name: "Key".into(),
        kind: ImportLightKind::Point,
        color: [0.9, 0.8, 0.7],
        intensity: 1234.5,
        range: None,
    });
    let mut node = ImportNode::new("Key");
    node.data = ImportNodeData::Light(0);
    node.transform = ImportTransform {
        translation: glam::Vec3::new(0.0, 3.0, 0.0),
        ..ImportTransform::default()
    };
    scene.source.axis = kreate_math::AxisConvention::GLTF;
    scene.nodes = vec![node];

    let mut doc = Document::new();
    let (made, _) = build_into(&scene, &mut doc);
    let object = doc.object(made[0]).expect("the lamp");
    let ObjectData::Light(id) = object.data else {
        panic!("the lamp is not a light");
    };
    let light = doc.light(id).expect("its datablock");
    assert_eq!(light.intensity, 1234.5, "candela did not survive");
    assert!(matches!(
        light.kind,
        kreate_scene::LightKind::Point { .. }
    ));

    // Three metres up in the file is three metres up here, which is a different
    // axis and the same place.
    assert!(
        (object.transform.translation - kreate_math::DVec3::new(0.0, 0.0, 3.0)).length() < 1e-6,
        "the lamp landed at {:?}",
        object.transform.translation
    );
}

#[test]
fn a_spots_cone_changes_parameterisation_and_says_so() {
    // glTF writes two half-angles and the datablock stores a full angle and the
    // fraction of it that fades. The shape is the same and the conversion is
    // exact; it is noted anyway, because "the same shape" is a claim somebody
    // should be able to check rather than take.
    let mut scene = ImportScene::new(SourceMetadata::new("test"));
    scene.lights.push(ImportLight {
        name: "Spot".into(),
        kind: ImportLightKind::Spot {
            inner: 0.3,
            outer: 0.6,
        },
        color: [1.0; 3],
        intensity: 500.0,
        range: None,
    });
    let mut node = ImportNode::new("Spot");
    node.data = ImportNodeData::Light(0);
    scene.nodes = vec![node];

    let mut doc = Document::new();
    let (made, notes) = build_into(&scene, &mut doc);
    let ObjectData::Light(id) = doc.object(made[0]).expect("the spot").data else {
        panic!("not a light");
    };
    match doc.light(id).expect("its datablock").kind {
        kreate_scene::LightKind::Spot {
            cone_angle, blend, ..
        } => {
            assert!((cone_angle - 1.2).abs() < 1e-6, "cone is {cone_angle}");
            assert!((blend - 0.5).abs() < 1e-6, "blend is {blend}");
        }
        other => panic!("asked for a spot and got {other:?}"),
    }
    assert!(notes
        .iter()
        .any(|n| matches!(n, BuildNote::ConeBlend { .. })));
}

#[test]
fn a_format_with_no_hierarchy_still_places_its_meshes() {
    // OBJ and STL have no nodes at all, and their meshes belong at the origin
    // because that is where the file puts them. Inventing a node each would
    // claim a structure the file does not have.
    let mut scene = ImportScene::new(SourceMetadata::new("obj"));
    scene.meshes.push(kreate_io::ImportMesh {
        name: "Thing".into(),
        positions: vec![glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Y],
        faces: vec![vec![0, 1, 2]],
        corner_uvs: None,
        corner_normals: None,
        face_materials: None,
        smoothing_groups: None,
    });
    assert!(scene.nodes.is_empty());

    let mut doc = Document::new();
    let (made, notes) = build_into(&scene, &mut doc);
    assert_eq!(made.len(), 1);
    assert!(notes.is_empty(), "{notes:?}");
    let object = doc.object(made[0]).expect("the mesh");
    assert_eq!(object.name, "Thing");
    assert_eq!(object.transform.translation, kreate_math::DVec3::ZERO);
}

/// The smallest glTF that has a light in it, written out by hand.
const LAMP_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["KHR_lights_punctual"],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        { "name": "Key", "type": "point", "color": [1.0, 0.9, 0.8], "intensity": 800.0 }
      ]
    }
  },
  "scene": 0,
  "scenes": [ { "nodes": [0] } ],
  "nodes": [
    {
      "name": "Key",
      "translation": [1.0, 5.0, -2.0],
      "extensions": { "KHR_lights_punctual": { "light": 0 } }
    }
  ]
}"#;

#[test]
fn a_light_comes_out_of_a_real_gltf_file() {
    // The parser's end of it. Everything above is the IR's own arithmetic; this
    // is whether the reader fills it in at all, which it did not before —
    // `document.meshes()` was walked and the node tree thrown away.
    let scene = parse_gltf(LAMP_GLTF.as_bytes(), &GltfOptions::default()).expect("parses");
    assert_eq!(scene.lights.len(), 1, "no light came through");
    assert_eq!(scene.lights[0].intensity, 800.0);
    assert_eq!(scene.nodes.len(), 1);
    assert_eq!(scene.nodes[0].data, ImportNodeData::Light(0));

    let mut doc = Document::new();
    let (made, _) = build_into(&scene, &mut doc);
    let object = doc.object(made[0]).expect("the lamp");
    // glTF is Y-up: five up and two toward the viewer becomes five along
    // Kreate's up and two along its forward.
    assert!(
        (object.transform.translation - kreate_math::DVec3::new(1.0, 2.0, 5.0)).length() < 1e-6,
        "the lamp landed at {:?}",
        object.transform.translation
    );
}
