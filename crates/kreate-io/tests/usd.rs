//! The scene as a USD stage, in text.
//!
//! The claims are about the text — it is balanced, it says what the scene
//! has, a mesh's arrays add up, a binding names a material that is there, and
//! a child sits inside its parent — and about the round trip: what Kreate
//! writes, Kreate's own reader brings back.

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

#[test]
fn keyed_motion_and_a_baked_body_go_out_as_time_samples() {
    use kreate_scene::animation::{Action, Interpolation};
    let mut doc = scene();
    let cube = doc.objects().find(|(_, o)| o.name == "Cube.001").unwrap().0;
    let mut action = Action::new("Slide");
    for (frame, x) in [(1.0, 0.0), (24.0, 5.0)] {
        action
            .curve_mut("location", 0)
            .insert(frame, x, Interpolation::Linear);
    }
    let a = doc.add_action(action);
    doc.object_mut(cube).unwrap().action = Some(a);
    let text = usda(&doc);
    assert!(text.contains("timeCodesPerSecond = 24"), "{text}");
    assert!(text.contains("xformOp:transform.timeSamples"));
    assert_eq!(
        text.matches('{').count(),
        text.matches('}').count(),
        "balanced"
    );

    // A rigid body, baked: its fall, sampled every frame.
    let mut doc = Document::new();
    doc.physics.frame_end = 12;
    let m = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let mut o = Object::new("Box", ObjectData::Mesh(m));
    o.transform.translation.z = 3.0;
    o.physics.rigid = Some(kreate_scene::physics::RigidBody::default());
    doc.add_object(o);
    let (cache, _) = kreate_scene::physics::bake::bake(&doc);
    doc.physics_cache = std::sync::Arc::new(cache);
    let text = usda(&doc);
    let samples = text
        .split("xformOp:transform.timeSamples")
        .nth(1)
        .expect("sampled");
    assert_eq!(
        samples.split('}').next().unwrap().matches("( (").count(),
        12,
        "one a frame"
    );
}

#[test]
fn baked_cloth_goes_out_as_moving_points() {
    let mut doc = Document::new();
    doc.playback.end = 10;
    doc.physics.frame_end = 10;
    let m = doc.add_mesh(kreate_mesh::primitives::grid(1.0, 3, 3));
    let mut o = Object::new("Sheet", ObjectData::Mesh(m));
    o.physics.soft = Some(kreate_scene::physics::SoftBody::default());
    doc.add_object(o);
    let (cache, _) = kreate_scene::physics::bake::bake(&doc);
    doc.physics_cache = std::sync::Arc::new(cache);
    let text = usda(&doc);
    assert!(text.contains("points.timeSamples"), "{text}");
    let samples = text.split("points.timeSamples").nth(1).unwrap();
    assert_eq!(
        samples
            .split("\n        }")
            .next()
            .unwrap()
            .matches(": [")
            .count(),
        10
    );
}

#[test]
fn what_is_written_reads_back_in_place_and_in_colour() {
    let doc = scene();
    let text = usda(&doc);
    let back = kreate_io::usd_read::parse_usda_text(&text).expect("reads");
    let mut read = Document::new();
    kreate_io::build::build_into(&back, &mut read);
    let find = |d: &Document, name: &str| {
        d.objects()
            .find(|(_, o)| o.name.replace('.', "_") == name.replace('.', "_"))
            .map(|(id, _)| id)
            .unwrap_or_else(|| panic!("{name} came back"))
    };
    let (a, b) = (find(&doc, "Cube.001"), find(&read, "Cube_001"));
    let (wa, wb) = (doc.world_transform(a), read.world_transform(b));
    assert!(
        (wa.translation - wb.translation).length() < 1e-5,
        "{} {}",
        wa.translation,
        wb.translation
    );
    // The cube's mesh sits under its transform; its corners land where they
    // did.
    let corners = |d: &Document| -> Vec<DVec3> {
        let mut out: Vec<DVec3> = d
            .objects()
            .filter_map(|(id, o)| match o.data {
                ObjectData::Mesh(m) => Some((d.world_transform(id), d.mesh(m)?)),
                _ => None,
            })
            .flat_map(|(w, m)| {
                m.positions()
                    .iter()
                    .map(move |p| w.transform_point3(p.as_dvec3()))
                    .collect::<Vec<_>>()
            })
            .collect();
        out.sort_by(|p, q| {
            p.x.total_cmp(&q.x)
                .then(p.y.total_cmp(&q.y))
                .then(p.z.total_cmp(&q.z))
        });
        out
    };
    let (ca, cb) = (corners(&doc), corners(&read));
    assert_eq!(ca.len(), cb.len());
    for (p, q) in ca.iter().zip(&cb) {
        assert!((*p - *q).length() < 1e-4, "{p} {q}");
    }
    let red = read
        .materials()
        .find(|(_, m)| m.name.contains("Red"))
        .map(|(_, m)| m.clone())
        .expect("the paint came back");
    assert!((red.roughness - 0.3).abs() < 1e-5);
    assert!((red.base_color.r - 0.8).abs() < 1e-5);
    assert!(read
        .objects()
        .any(|(_, o)| matches!(o.data, ObjectData::Light(_))));
    assert!(read
        .objects()
        .any(|(_, o)| matches!(o.data, ObjectData::Camera(_))));
}

#[test]
fn a_usdz_carries_its_pictures_and_reads_back_with_them() {
    let mut doc = Document::new();
    let mut pixels = Vec::new();
    for y in 0..3u8 {
        for x in 0..5u8 {
            pixels.extend_from_slice(&[x * 40, y * 60, 200, 255]);
        }
    }
    let picture = kreate_io::image_import::image_from_rgba8(
        "Bricks",
        5,
        3,
        &pixels,
        kreate_scene::ColorSpace::Srgb,
    )
    .expect("an image");
    let picture = doc.add_image(picture);
    let packed = kreate_io::image_import::image_from_rgba8(
        "MetalRough",
        5,
        3,
        &pixels,
        kreate_scene::ColorSpace::Raw,
    )
    .expect("an image");
    let packed = doc.add_image(packed);
    let brick = doc.add_material(Material {
        name: "Brick".into(),
        base_color_image: Some(picture),
        metallic_roughness_image: Some(packed),
        ..Default::default()
    });
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    doc.set_mesh_slots(mesh, vec![Some(brick)]);
    doc.add_object(Object::new("Wall", ObjectData::Mesh(mesh)));

    let (scene, _) = kreate_io::scene_from_document(&doc);
    let (bytes, notes) = kreate_io::usd::write_usdz(&scene);
    assert!(
        notes.iter().all(|n| !format!("{n:?}").contains("whole")),
        "{notes:?}"
    );
    // Every file's data on a 64-byte boundary, stored as it is.
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&bytes)).expect("a zip");
    assert_eq!(
        archive.by_index(0).unwrap().name(),
        "scene.usda",
        "the stage first"
    );
    for i in 0..archive.len() {
        let f = archive.by_index(i).unwrap();
        assert_eq!(f.compression(), zip::CompressionMethod::Stored);
        assert_eq!(f.data_start() % 64, 0, "{}", f.name());
    }

    let path = std::env::temp_dir().join(format!("kreate-usdz-{}.usdz", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    if let Ok(keep) = std::env::var("KREATE_KEEP_USDZ") {
        std::fs::write(keep, &bytes).unwrap();
    }
    let back = kreate_io::usd_read::parse_usd_file(&path);
    let _ = std::fs::remove_file(&path);
    let back = back.expect("reads");
    let m = back
        .materials
        .iter()
        .find(|m| m.name == "Brick")
        .expect("the brick came back");
    let t = m.base_color_texture.expect("with its picture");
    assert_eq!((back.images[t].width, back.images[t].height), (5, 3));
    assert_eq!(&back.images[t].pixels[..4], &pixels[..4]);
    assert!(
        m.metallic_roughness_texture.is_some(),
        "metal and roughness, packed as they went"
    );
}
