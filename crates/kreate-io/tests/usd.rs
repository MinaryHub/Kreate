//! The scene as a USD stage, in text.
//!
//! No USD library is at hand to read the result, so the claims are about the
//! text: it is balanced, it says what the scene has, a mesh's arrays add up,
//! a binding names a material that is there, and a child sits inside its
//! parent.

use kreate_math::{DQuat, DTransform, DVec3};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{Document, Material};

fn scene() -> Document {
    let mut doc = Document::new();
    let red = doc.add_material(Material {
        name: "Red Paint".into(),
        base_color: kreate_math::SceneRgb::new(0.8, 0.1, 0.1),
        roughness: 0.3,
        ..Default::default()
    });
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
    doc.set_mesh_slots(mesh, vec![Some(red)]);
    let mut cube = Object::new("Cube.001", ObjectData::Mesh(mesh));
    cube.transform = DTransform {
        translation: DVec3::new(1.0, 2.0, 3.0),
        rotation: DQuat::from_rotation_z(0.5),
        scale: DVec3::ONE,
    };
    let cube = doc.add_object(cube);
    let lamp = doc.add_light(kreate_scene::Light::of_kind(
        "Lamp",
        kreate_scene::LightKind::BULB,
    ));
    let lamp = doc.add_object(Object::new("Lamp", ObjectData::Light(lamp)));
    doc.set_parent(lamp, Some(cube)).unwrap();
    let camera = doc.add_camera(kreate_scene::Camera::new("Camera"));
    doc.add_object(Object::new("Camera", ObjectData::Camera(camera)));
    doc
}

fn usda(doc: &Document) -> String {
    let (scene, _) = kreate_io::scene_from_document(doc);
    kreate_io::usd::write_usda(&scene).0
}

#[test]
fn the_stage_is_balanced_z_up_and_has_everything() {
    let text = usda(&scene());
    assert!(text.starts_with("#usda 1.0"));
    assert!(text.contains("upAxis = \"Z\""));
    for (open, close) in [('{', '}'), ('(', ')'), ('[', ']')] {
        assert_eq!(
            text.matches(open).count(),
            text.matches(close).count(),
            "unbalanced {open}{close}"
        );
    }
    for wanted in [
        "def Xform \"Cube_001\"",
        "def Mesh \"Cube_001_mesh\"",
        "def SphereLight \"Lamp\"",
        "def Camera \"Camera\"",
        "def Material \"Red_Paint\"",
        "uniform token info:id = \"UsdPreviewSurface\"",
        "rel material:binding = </World/Looks/Red_Paint>",
        "float inputs:roughness = 0.3",
    ] {
        assert!(text.contains(wanted), "missing {wanted}\n{text}");
    }
}

#[test]
fn a_mesh_s_arrays_add_up() {
    let text = usda(&scene());
    let array = |key: &str| -> Vec<String> {
        let at = text.find(key).unwrap_or_else(|| panic!("no {key}"));
        let open = at + text[at..].find('[').unwrap();
        let close = open + text[open..].find(']').unwrap();
        text[open + 1..close]
            .split(", ")
            .map(str::to_string)
            .collect()
    };
    let counts: Vec<usize> = array("faceVertexCounts")
        .iter()
        .map(|c| c.parse().unwrap())
        .collect();
    let indices = array("faceVertexIndices");
    assert_eq!(counts, vec![4; 6]);
    assert_eq!(indices.len(), counts.iter().sum::<usize>());
    // Normals and texture coordinates are per corner.
    let line = text[text.find("normal3f[] normals").unwrap()..]
        .lines()
        .next()
        .unwrap();
    let open = line.find("= [").unwrap();
    let normals = line[open..open + line[open..].find(']').unwrap()]
        .matches('(')
        .count();
    assert_eq!(normals, 24);
}

#[test]
fn a_child_is_nested_inside_its_parent() {
    let text = usda(&scene());
    let parent = text.find("def Xform \"Cube_001\"").unwrap();
    let child = text.find("def SphereLight \"Lamp\"").unwrap();
    assert!(child > parent);
    // Between the two, more braces opened than closed: the lamp is inside.
    let between = &text[parent..child];
    assert!(between.matches('{').count() > between.matches('}').count());
    // And the parent's translation is the object's.
    assert!(text.contains("(1, 2, 3, 1) )"), "{text}");
}
