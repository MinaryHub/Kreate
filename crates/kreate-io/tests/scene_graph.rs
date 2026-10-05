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
use kreate_scene::document::{ObjectData, ObjectId};
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
    assert!(matches!(light.kind, kreate_scene::LightKind::Point { .. }));

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

// ---------------------------------------------------------------------------
// The way back out.
//
// Everything above is the import half. These close the loop §5.7 asks for: a
// document goes into a file and comes back, and what changed is compared
// against what the format can hold. Until the exporter took a scene it took a
// list of meshes, so a lit document could arrive and could not leave — and a
// fidelity contract that only holds in one direction is not one.

/// A document, written to a `.glb` and read back into a fresh one.
///
/// Through the real bytes rather than through the IR alone: the IR agreeing
/// with itself is what the unit tests check, and the thing that can silently go
/// wrong here is what `serde` writes and `gltf` reads.
fn round_trip(doc: &Document) -> (Document, Vec<ObjectId>) {
    let (scene, _) = kreate_io::scene_from_document(doc);
    let (bytes, _) = kreate_io::write_gltf_scene(&scene, &kreate_io::GltfExportOptions::default());
    let read = parse_gltf(&bytes, &GltfOptions::default()).expect("what we wrote parses");
    let mut out = Document::new();
    let (made, _) = build_into(&read, &mut out);
    (out, made)
}

#[test]
fn a_lamp_goes_out_to_a_file_and_comes_back_the_same_lamp() {
    // The half that was missing, end to end. §2.2 chose glTF's units so that
    // this is a copy in both directions — so the numbers are asserted exactly,
    // not within a tolerance for a conversion that should not be happening.
    let mut doc = Document::new();
    let mut lamp =
        kreate_scene::Light::of_kind("Key", kreate_scene::LightKind::Point { radius: 0.05 });
    lamp.intensity = 1234.5;
    lamp.color = kreate_math::SceneRgb::new(0.9, 0.8, 0.7);
    let id = doc.add_light(lamp);
    let mut object = kreate_scene::document::Object::new("Key", ObjectData::Light(id));
    object.transform.translation = kreate_math::DVec3::new(1.0, 2.0, 5.0);
    doc.add_object(object);

    let (out, made) = round_trip(&doc);
    let object = out.object(made[0]).expect("the lamp came back");
    assert_eq!(object.name, "Key");
    assert!(
        (object.transform.translation - kreate_math::DVec3::new(1.0, 2.0, 5.0)).length() < 1e-5,
        "the lamp came back at {:?}",
        object.transform.translation
    );
    let ObjectData::Light(id) = object.data else {
        panic!("it came back as something that is not a light");
    };
    let light = out.light(id).expect("its datablock");
    assert_eq!(light.intensity, 1234.5, "candela did not survive the file");
    assert!((light.color.r - 0.9).abs() < 1e-6);
    assert!((light.color.g - 0.8).abs() < 1e-6);
    assert!((light.color.b - 0.7).abs() < 1e-6);
}

#[test]
fn a_spots_cone_is_the_same_cone_on_the_way_back() {
    // The one conversion in the light path that is not a copy: a full angle and
    // a blend fraction become two half-angles, and back. Exact in both
    // directions, which is the claim — and the one place a "close enough"
    // tolerance would hide an off-by-a-half.
    let mut doc = Document::new();
    let id = doc.add_light(kreate_scene::Light::of_kind(
        "Spot",
        kreate_scene::LightKind::Spot {
            radius: 0.05,
            cone_angle: 1.2,
            // Not a half. A blend of one half makes `blend` and `1 - blend` the
            // same number, so the fixture agreed with the conversion written
            // either way round — and the sweep walked straight through it.
            blend: 0.25,
        },
    ));
    doc.add_object(kreate_scene::document::Object::new(
        "Spot",
        ObjectData::Light(id),
    ));

    let (out, made) = round_trip(&doc);
    let ObjectData::Light(id) = out.object(made[0]).expect("the spot").data else {
        panic!("not a light");
    };
    match out.light(id).expect("its datablock").kind {
        kreate_scene::LightKind::Spot {
            cone_angle, blend, ..
        } => {
            assert!((cone_angle - 1.2).abs() < 1e-6, "cone is {cone_angle}");
            assert!((blend - 0.25).abs() < 1e-6, "blend is {blend}");
        }
        other => panic!("a spot came back as {other:?}"),
    }
}

#[test]
fn the_suns_lux_is_not_quietly_turned_into_candela() {
    // Three kinds and three units, and the sun is the one whose unit is not
    // candela. An exporter that folded them all through one path would write a
    // plausible number, and the sun would come back as an indoor lamp.
    let mut doc = Document::new();
    let mut sun = kreate_scene::Light::sun("Sun");
    sun.intensity = 20_000.0;
    let id = doc.add_light(sun);
    doc.add_object(kreate_scene::document::Object::new(
        "Sun",
        ObjectData::Light(id),
    ));

    let (out, made) = round_trip(&doc);
    let ObjectData::Light(id) = out.object(made[0]).expect("the sun").data else {
        panic!("not a light");
    };
    let light = out.light(id).expect("its datablock");
    assert!(matches!(light.kind, kreate_scene::LightKind::Sun { .. }));
    assert_eq!(light.intensity, 20_000.0, "lux did not survive the file");
}

#[test]
fn a_lights_stops_leave_with_it_rather_than_staying_behind() {
    // glTF holds one number where the datablock holds an intensity and a number
    // of stops. Folding is the only honest answer — the amount of light is what
    // the file is for — and the note says so, because a round trip brings back
    // the product with the stops at zero.
    let mut doc = Document::new();
    let mut lamp =
        kreate_scene::Light::of_kind("Key", kreate_scene::LightKind::Point { radius: 0.05 });
    lamp.intensity = 100.0;
    lamp.exposure = 2.0;
    let id = doc.add_light(lamp);
    doc.add_object(kreate_scene::document::Object::new(
        "Key",
        ObjectData::Light(id),
    ));

    let (scene, notes) = kreate_io::scene_from_document(&doc);
    assert!(
        notes.iter().any(|n| format!("{n:?}").contains("stops")),
        "the stops were folded without a word: {notes:?}"
    );
    assert!(
        (scene.lights[0].intensity - 400.0).abs() < 1e-3,
        "two stops on a hundred candela came out as {}",
        scene.lights[0].intensity
    );
}

#[test]
fn a_child_is_still_a_child_on_the_far_side_of_a_file() {
    // The hierarchy, which is the thing the IR did not have.
    //
    // **A position alone cannot tell a link from a baked number**, and that is
    // the whole difficulty: an exporter that wrote each object's world
    // transform and no parents would put everything in exactly the right place,
    // and the file would be wrong in a way no still picture shows. So the test
    // moves the parent afterwards and asks whether the child followed.
    let mut doc = Document::new();
    let parent = doc.add_object(kreate_scene::document::Object::new(
        "Root",
        ObjectData::Empty,
    ));
    let mut child_object = kreate_scene::document::Object::new("Child", ObjectData::Empty);
    child_object.transform.translation = kreate_math::DVec3::new(0.0, 2.0, 0.0);
    let child = doc.add_object(child_object);
    doc.set_parent(child, Some(parent)).expect("a fresh chain");

    let (mut out, made) = round_trip(&doc);
    assert_eq!(
        out.object(made[1]).expect("the child").parent,
        Some(made[0])
    );
    assert!(
        (out.world_transform(made[1]).translation - doc.world_transform(child).translation)
            .length()
            < 1e-5,
        "the child landed at {:?}",
        out.world_transform(made[1]).translation
    );

    out.object_mut(made[0])
        .expect("the root")
        .transform
        .translation = kreate_math::DVec3::new(5.0, 0.0, 0.0);
    let moved = out.world_transform(made[1]).translation;
    assert!(
        (moved - kreate_math::DVec3::new(5.0, 2.0, 0.0)).length() < 1e-5,
        "the parent moved and the child stayed at {moved:?}"
    );
}

#[test]
fn parenting_in_the_application_does_not_move_the_child_through_a_file() {
    // The subtlety that is the mirror image of `build_into`'s ordering trap.
    // `set_parent` remembers where the parent stood so that dragging one object
    // onto another does not teleport it, and the document composes
    // `parent_world · parent_inverse · local`. A file's child transform has
    // nothing in between, so the two have to be folded on the way out — and a
    // parent that was already somewhere when the child was attached is exactly
    // the case that catches it.
    let mut doc = Document::new();
    let mut parent_object = kreate_scene::document::Object::new("Root", ObjectData::Empty);
    parent_object.transform.translation = kreate_math::DVec3::new(10.0, 0.0, 0.0);
    let parent = doc.add_object(parent_object);
    let mut child_object = kreate_scene::document::Object::new("Child", ObjectData::Empty);
    child_object.transform.translation = kreate_math::DVec3::new(0.0, 0.0, 3.0);
    let child = doc.add_object(child_object);
    // Attached after both are placed, which is what a user dragging one onto
    // the other does — and it leaves a non-identity `parent_inverse`.
    doc.set_parent(child, Some(parent)).expect("a fresh chain");
    let before = doc.world_transform(child).translation;

    let (out, made) = round_trip(&doc);
    let after = out.world_transform(made[1]).translation;
    assert!(
        (after - before).length() < 1e-5,
        "the child was at {before:?} and came back at {after:?}"
    );
}

#[test]
fn a_linked_duplicate_is_still_one_mesh_in_the_file() {
    // What a linked duplicate means: two objects and one mesh. glTF says the
    // same thing the same way, so the sharing is expressible — and writing the
    // geometry twice would double the file and quietly turn one datablock into
    // two on the way back, which is the thing §1.4 exists to prevent.
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    doc.add_object(kreate_scene::document::Object::new(
        "Left",
        ObjectData::Mesh(mesh),
    ));
    let mut right = kreate_scene::document::Object::new("Right", ObjectData::Mesh(mesh));
    right.transform.translation = kreate_math::DVec3::new(3.0, 0.0, 0.0);
    doc.add_object(right);

    let (scene, _) = kreate_io::scene_from_document(&doc);
    assert_eq!(scene.meshes.len(), 1, "the mesh was written twice");
    assert_eq!(scene.nodes.len(), 2);

    let (out, made) = round_trip(&doc);
    assert_eq!(made.len(), 2);
    let ids: Vec<_> = made
        .iter()
        .map(|id| match out.object(*id).expect("an object").data {
            ObjectData::Mesh(mesh) => mesh,
            ref other => panic!("expected a mesh and got {other:?}"),
        })
        .collect();
    assert_eq!(ids[0], ids[1], "one mesh came back as two");
    assert!(
        (out.object(made[1])
            .expect("the right one")
            .transform
            .translation
            - kreate_math::DVec3::new(3.0, 0.0, 0.0))
        .length()
            < 1e-5
    );
}

#[test]
fn a_camera_comes_back_looking_through_the_same_angle() {
    // glTF holds an angle and the datablock holds a lens on a sensor. The angle
    // is what survives, and the rest is noted rather than dropped in silence:
    // the aperture and the shutter are what the exposure is computed from, and
    // a file that came back with a different one would be a different picture.
    let mut doc = Document::new();
    let camera = kreate_scene::Camera::new("Camera");
    let wanted = camera.fov_y(camera.sensor_width_mm / camera.sensor_height_mm);
    let id = doc.add_camera(camera);
    doc.add_object(kreate_scene::document::Object::new(
        "Camera",
        ObjectData::Camera(id),
    ));

    // The note comes from the writer and not from `scene_from_document`,
    // deliberately: an aperture is something *glTF* has no room for, and USD
    // does. A note raised where the document is read would claim the loss for
    // every format at once.
    let (scene, _) = kreate_io::scene_from_document(&doc);
    let (_, notes) = kreate_io::write_gltf_scene(&scene, &kreate_io::GltfExportOptions::default());
    assert!(
        notes.iter().any(|n| format!("{n:?}").contains("shutter")),
        "the photographic parameters went missing quietly: {notes:?}"
    );

    let (out, made) = round_trip(&doc);
    let ObjectData::Camera(id) = out.object(made[0]).expect("the camera").data else {
        panic!("it came back as something that is not a camera");
    };
    let back = out.camera(id).expect("its datablock");
    let got = back.fov_y(back.sensor_width_mm / back.sensor_height_mm);
    assert!(
        (got - wanted).abs() < 1e-4,
        "{wanted} radians came back as {got}"
    );
}

#[test]
fn two_meshes_do_not_swap_materials_on_the_way_out() {
    // A face carries a *slot* and a slot is per-mesh (§7.5), so slot 0 of one
    // mesh and slot 0 of another are different materials. An exporter keyed on
    // the face's own number gets this right for one mesh and wrong for every
    // mesh after it, which is why the fixture has two.
    let mut doc = Document::new();
    let red = doc.add_material(kreate_scene::document::Material {
        name: "Red".into(),
        base_color: kreate_math::SceneRgb::new(1.0, 0.0, 0.0),
        ..Default::default()
    });
    let blue = doc.add_material(kreate_scene::document::Material {
        name: "Blue".into(),
        base_color: kreate_math::SceneRgb::new(0.0, 0.0, 1.0),
        ..Default::default()
    });
    let first = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let second = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    doc.set_mesh_slots(first, vec![Some(red)]);
    doc.set_mesh_slots(second, vec![Some(blue)]);
    doc.add_object(kreate_scene::document::Object::new(
        "First",
        ObjectData::Mesh(first),
    ));
    doc.add_object(kreate_scene::document::Object::new(
        "Second",
        ObjectData::Mesh(second),
    ));

    let (scene, _) = kreate_io::scene_from_document(&doc);
    // Both meshes' faces say slot 0, and the two slot 0s are different
    // materials. That is the whole of it.
    assert_ne!(
        scene.material_of(0, 0),
        scene.material_of(1, 0),
        "the two meshes resolved slot 0 to the same material"
    );
    let a = scene.materials[scene.material_of(0, 0).expect("a material")]
        .name
        .clone();
    let b = scene.materials[scene.material_of(1, 0).expect("a material")]
        .name
        .clone();
    assert_eq!((a.as_str(), b.as_str()), ("Red", "Blue"));
}

#[test]
fn a_mesh_list_with_no_hierarchy_still_writes_one_node_each() {
    // What the converter hands over and what a format with no nodes means. The
    // mesh-only entry point kept its shape through all of this, and the reason
    // it is checked here is that it is now a thin wrapper over the scene path —
    // so the thing that could break it is the scene path growing a requirement.
    let cube = kreate_mesh::primitives::cube(1.0);
    let (bytes, _) = kreate_io::write_gltf(
        &[("Cube".into(), &cube)],
        &[],
        &kreate_io::GltfExportOptions::default(),
    );
    let scene = parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(scene.nodes.len(), 1);
    assert_eq!(scene.nodes[0].data, ImportNodeData::Mesh(0));
    assert_eq!(scene.nodes[0].parent, None);
}

#[test]
fn an_objects_turn_and_size_survive_the_file() {
    // Every fixture above places things by translation alone, and a translation
    // is the one kind of transform that survives almost any mistake: the node
    // writer could drop the rotation and the scale entirely and every one of
    // them would still pass. A turn and a squash are what make the axis
    // conversion's conjugation do any work at all.
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let mut object = kreate_scene::document::Object::new("Thing", ObjectData::Mesh(mesh));
    object.transform.translation = kreate_math::DVec3::new(1.0, -2.0, 0.5);
    // About X, deliberately: a turn about *up* is the one the axis conversion
    // gets right even when it is applied on one side only.
    object.transform.rotation = kreate_math::DQuat::from_rotation_x(0.7);
    // Non-uniform, because a uniform scale commutes with everything and hides a
    // conjugation applied to the wrong factor.
    object.transform.scale = kreate_math::DVec3::new(2.0, 0.5, 1.25);
    let id = doc.add_object(object);
    let before = doc.world_transform(id);

    let (out, made) = round_trip(&doc);
    let after = out.world_transform(made[0]);
    // Compared by where the corners land rather than field by field: a rotation
    // and a scale can be written two ways that mean the same thing, and what
    // has to match is the placement, not the spelling.
    for corner in [
        kreate_math::DVec3::new(1.0, 1.0, 1.0),
        kreate_math::DVec3::new(-1.0, 1.0, -1.0),
        kreate_math::DVec3::new(0.0, 0.0, 1.0),
    ] {
        let want = before.transform_point3(corner);
        let got = after.transform_point3(corner);
        assert!(
            (want - got).length() < 1e-4,
            "{corner:?} was at {want:?} and came back at {got:?}"
        );
    }
}

#[test]
fn a_rotated_parent_places_its_child_the_same_way_on_both_sides() {
    // `parent_inverse` has to be folded in on the left, and every fixture with
    // pure translations passes whichever side it is folded on — translations
    // commute. A rotated parent is what tells the two apart, and it is not an
    // exotic case: it is a lamp parented to a turntable.
    let mut doc = Document::new();
    let mut parent_object = kreate_scene::document::Object::new("Turntable", ObjectData::Empty);
    parent_object.transform.translation = kreate_math::DVec3::new(4.0, 0.0, 0.0);
    parent_object.transform.rotation =
        kreate_math::DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
    let parent = doc.add_object(parent_object);
    let mut child_object = kreate_scene::document::Object::new("Arm", ObjectData::Empty);
    child_object.transform.translation = kreate_math::DVec3::new(0.0, 3.0, 1.0);
    let child = doc.add_object(child_object);
    doc.set_parent(child, Some(parent)).expect("a fresh chain");
    let before = doc.world_transform(child).translation;

    let (out, made) = round_trip(&doc);
    let after = out.world_transform(made[1]).translation;
    assert!(
        (after - before).length() < 1e-4,
        "the arm was at {before:?} and came back at {after:?}"
    );
}

#[test]
fn a_mesh_with_no_faces_does_not_shift_every_node_after_it() {
    // A mesh with nothing in it is skipped, so the scene's numbering and the
    // file's stop agreeing — and a node that assumed they ran in step hangs the
    // wrong geometry on everything after the gap. One empty mesh in front of a
    // real one is the whole of the fixture.
    let mut doc = Document::new();
    let empty = doc.add_mesh(kreate_mesh::MeshData::empty());
    let cube = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    doc.add_object(kreate_scene::document::Object::new(
        "Nothing",
        ObjectData::Mesh(empty),
    ));
    let mut solid = kreate_scene::document::Object::new("Cube", ObjectData::Mesh(cube));
    solid.transform.translation = kreate_math::DVec3::new(0.0, 0.0, 2.0);
    doc.add_object(solid);

    let (out, made) = round_trip(&doc);
    assert_eq!(made.len(), 2, "a node went missing along with its mesh");
    let solid = out.object(made[1]).expect("the cube");
    assert_eq!(solid.name, "Cube");
    let ObjectData::Mesh(id) = solid.data else {
        panic!("the cube came back without its mesh");
    };
    assert!(
        out.mesh(id).expect("its geometry").num_faces() > 0,
        "the cube came back empty, which is the empty mesh wearing its name"
    );
}

#[test]
fn an_orthographic_camera_is_still_orthographic_on_the_far_side() {
    // Two numbers and a kind, and the kind is the one a test that only compares
    // an angle cannot see: an orthographic camera that comes back as a
    // perspective one keeps its focal length, so its field of view still
    // matches while the picture it takes is a different picture.
    let mut doc = Document::new();
    let mut camera = kreate_scene::Camera::new("Ortho");
    camera.projection = kreate_scene::Projection::Orthographic;
    camera.ortho_scale = 7.5;
    camera.clip_near = 0.25;
    camera.clip_far = 250.0;
    let id = doc.add_camera(camera);
    doc.add_object(kreate_scene::document::Object::new(
        "Ortho",
        ObjectData::Camera(id),
    ));

    let (out, made) = round_trip(&doc);
    let ObjectData::Camera(id) = out.object(made[0]).expect("the camera").data else {
        panic!("it came back as something that is not a camera");
    };
    let back = out.camera(id).expect("its datablock");
    assert!(
        matches!(back.projection, kreate_scene::Projection::Orthographic),
        "it came back as a {:?} camera",
        back.projection
    );
    assert!(
        (back.ortho_scale - 7.5).abs() < 1e-4,
        "a view 7.5 metres across came back {} across",
        back.ortho_scale
    );
    // The clipping planes, which nothing else checks — and a near plane of zero
    // is the one value a depth buffer cannot use.
    assert!(
        (back.clip_near - 0.25).abs() < 1e-5,
        "near is {}",
        back.clip_near
    );
    assert!(
        (back.clip_far - 250.0).abs() < 1e-2,
        "far is {}",
        back.clip_far
    );
}

#[test]
fn an_area_light_leaves_as_a_point_light_of_the_same_axial_intensity() {
    // glTF has no area light, and §7.3 already reduces one to a point light for
    // the raster pass: the same candela along the normal, from the same place.
    // The conversion is nits times area, and the second panel below is what
    // makes it a conversion — at one square metre, multiplying and doing
    // nothing give the same number.
    let mut doc = Document::new();
    let mut panel = kreate_scene::Light::of_kind(
        "Panel",
        kreate_scene::LightKind::Area {
            shape: kreate_scene::AreaShape::Rect {
                width: 2.0,
                height: 0.5,
            },
            spread: std::f64::consts::FRAC_PI_2,
        },
    );
    panel.intensity = 500.0;
    let id = doc.add_light(panel);
    doc.add_object(kreate_scene::document::Object::new(
        "Panel",
        ObjectData::Light(id),
    ));

    let (scene, notes) = kreate_io::scene_from_document(&doc);
    assert!(
        notes
            .iter()
            .any(|n| format!("{n:?}").contains("area light")),
        "the shape went missing quietly: {notes:?}"
    );
    assert!(
        (scene.lights[0].intensity - 500.0).abs() < 1e-3,
        "500 nits over a square metre came out as {}",
        scene.lights[0].intensity
    );

    let mut doc = Document::new();
    let mut panel = kreate_scene::Light::of_kind(
        "Panel",
        kreate_scene::LightKind::Area {
            shape: kreate_scene::AreaShape::Rect {
                width: 2.0,
                height: 2.0,
            },
            spread: std::f64::consts::FRAC_PI_2,
        },
    );
    panel.intensity = 500.0;
    let id = doc.add_light(panel);
    doc.add_object(kreate_scene::document::Object::new(
        "Panel",
        ObjectData::Light(id),
    ));
    let (scene, _) = kreate_io::scene_from_document(&doc);
    assert!(
        (scene.lights[0].intensity - 2000.0).abs() < 1e-2,
        "four square metres at 500 nits came out as {}",
        scene.lights[0].intensity
    );
}

/// The file as a reader that is not us would see it.
///
/// Everything else here goes out through the writer and back through our own
/// importer, and the two agree about more than the specification requires: the
/// importer walks `document.nodes()` and never reads `scenes`, and the `gltf`
/// crate parses the light table whether or not the file declares the extension.
/// So a whole class of mistake — a file that is wrong for everybody else and
/// right for us — is invisible from inside the round trip.
fn written_json(doc: &Document) -> serde_json::Value {
    let (scene, _) = kreate_io::scene_from_document(doc);
    let (bytes, _) = kreate_io::write_gltf_scene(
        &scene,
        &kreate_io::GltfExportOptions {
            binary: false,
            ..kreate_io::GltfExportOptions::default()
        },
    );
    serde_json::from_slice(&bytes).expect("what we wrote is json")
}

#[test]
fn the_file_declares_the_light_extension_it_uses() {
    // A reader is entitled to skip an extension the file does not declare, and
    // several do. Ours does not notice, which is exactly why this is asserted
    // against the bytes and against the spelled-out name: a typo in the
    // constant produces a file that works here and has no lights anywhere else.
    let mut doc = Document::new();
    let id = doc.add_light(kreate_scene::Light::of_kind(
        "Key",
        kreate_scene::LightKind::Point { radius: 0.05 },
    ));
    doc.add_object(kreate_scene::document::Object::new(
        "Key",
        ObjectData::Light(id),
    ));

    let json = written_json(&doc);
    let used = json["extensionsUsed"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        used.iter().any(|v| v == "KHR_lights_punctual"),
        "the file uses lights and declares {used:?}"
    );
    // And not as required: a reader that ignores it still gets the whole model,
    // and a file nobody else can open is a bad trade for a round trip with
    // ourselves.
    assert!(
        json.get("extensionsRequired").is_none(),
        "the lights were made mandatory"
    );
}

#[test]
fn the_scene_lists_its_roots_and_not_every_node() {
    // glTF's scene names the roots; children hang off `children`. Listing them
    // all makes every child appear twice, which readers variously ignore, draw
    // twice, or reject — and our own importer never reads the list at all, so
    // nothing else here can see it.
    let mut doc = Document::new();
    let parent = doc.add_object(kreate_scene::document::Object::new(
        "Root",
        ObjectData::Empty,
    ));
    let child = doc.add_object(kreate_scene::document::Object::new(
        "Child",
        ObjectData::Empty,
    ));
    doc.set_parent(child, Some(parent)).expect("a fresh chain");

    let json = written_json(&doc);
    let roots = json["scenes"][0]["nodes"].as_array().expect("a scene");
    assert_eq!(roots.len(), 1, "the scene lists {roots:?}");
    let children = json["nodes"][0]["children"]
        .as_array()
        .expect("the root's children");
    assert_eq!(children.len(), 1);
}
