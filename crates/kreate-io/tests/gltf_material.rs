//! Materials and their textures, out to glTF and back.
//!
//! Until this, a material could leave as factors and could not arrive at all:
//! the importer read the file's material table and nothing turned it into
//! datablocks, so every object imported grey, and a texture was a name nobody
//! resolved. The round trip is through the real bytes, for the reason
//! `scene_graph.rs` gives — what `serde` writes and `gltf` reads is exactly
//! the part the IR agreeing with itself cannot check.

use kreate_io::build::{build_into, BuildNote};
use kreate_io::{parse_gltf, GltfExportOptions, GltfOptions};
use kreate_scene::document::{ImageId, MaterialId, Object, ObjectData};
use kreate_scene::{ColorSpace, Document, Image, Material, PixelFormat};

/// An image whose every texel says where it is, so a flip or a transpose
/// shows up as a value in the wrong place rather than as a wash.
fn numbered(doc: &mut Document, name: &str, tag: u8, space: ColorSpace) -> ImageId {
    let (width, height) = (5u32, 3u32);
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            pixels.extend_from_slice(&[x as u8 * 40, y as u8 * 60, tag, 255]);
        }
    }
    let image = kreate_io::image_import::image_from_rgba8(name, width, height, &pixels, space)
        .expect("a small image");
    doc.add_image(image)
}

fn cube_with(doc: &mut Document, name: &str, material: MaterialId) {
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    doc.set_mesh_slots(mesh, vec![Some(material)]);
    doc.add_object(Object::new(name, ObjectData::Mesh(mesh)));
}

fn write(doc: &Document, binary: bool) -> Vec<u8> {
    let (scene, _) = kreate_io::scene_from_document(doc);
    let (bytes, _) = kreate_io::write_gltf_scene(
        &scene,
        &GltfExportOptions {
            binary,
            ..GltfExportOptions::default()
        },
    );
    bytes
}

fn read(bytes: &[u8]) -> (Document, Vec<BuildNote>) {
    let scene = parse_gltf(bytes, &GltfOptions::default()).expect("what we wrote parses");
    let mut out = Document::new();
    let (_, notes) = build_into(&scene, &mut out);
    (out, notes)
}

/// The JSON half of a `.glb`, for looking at what other readers will see.
fn glb_json(bytes: &[u8]) -> serde_json::Value {
    let length = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[20..20 + length]).expect("the chunk is json")
}

/// A document with one fully mapped material, nothing at its default.
fn mapped() -> Document {
    let mut doc = Document::new();
    let colour = numbered(&mut doc, "Albedo", 11, ColorSpace::Srgb);
    let packed = numbered(&mut doc, "ORM", 22, ColorSpace::Raw);
    let normal = numbered(&mut doc, "Normal", 33, ColorSpace::Raw);
    let material = doc.add_material(Material {
        name: "Brick".into(),
        base_color: kreate_math::SceneRgb::new(0.7, 0.5, 0.3),
        alpha: 0.9,
        metallic: 0.25,
        roughness: 0.75,
        base_color_image: Some(colour),
        metallic_roughness_image: Some(packed),
        normal_image: Some(normal),
        normal_strength: 0.6,
        ..Default::default()
    });
    cube_with(&mut doc, "Wall", material);
    doc
}

#[test]
fn a_material_arrives_as_a_material() {
    // The import half that did not exist. Factors are linear on both sides, so
    // they are asserted exactly.
    let (back, _) = read(&write(&mapped(), true));
    let (_, m) = back.materials().next().expect("the material arrived");
    assert_eq!(m.name, "Brick");
    assert_eq!(m.base_color.to_array(), [0.7, 0.5, 0.3]);
    assert_eq!((m.alpha, m.metallic, m.roughness), (0.9, 0.25, 0.75));
    assert_eq!(m.normal_strength, 0.6);
}

#[test]
fn every_map_comes_back_with_its_pixels_and_its_space() {
    // Each map to its own slot — a slot read from its neighbour would show as
    // the wrong tag — and the space the slot declares: the colour sRGB, the
    // others raw, which decides the atlas it is drawn from.
    let (back, _) = read(&write(&mapped(), true));
    let (_, m) = back.materials().next().expect("the material arrived");
    for (slot, tag, space) in [
        (m.base_color_image, 11, ColorSpace::Srgb),
        (m.metallic_roughness_image, 22, ColorSpace::Raw),
        (m.normal_image, 33, ColorSpace::Raw),
    ] {
        let image = back.image(slot.expect("the slot is filled")).expect("the image exists");
        assert_eq!(image.color_space, space);
        assert_eq!((image.width(), image.height()), (5, 3));
        let pixels = kreate_io::image_import::rgba8_of(image, space);
        // The corner the rows start at, and the far one: a flipped or
        // transposed image moves both.
        assert_eq!(&pixels[..4], &[0, 0, tag, 255]);
        let last = (3 * 5 - 1) * 4;
        assert_eq!(&pixels[last..last + 4], &[160, 120, tag, 255]);
    }
}

#[test]
fn a_gltf_with_a_data_uri_carries_its_textures_too() {
    // The text form is self-contained as well: its buffer is a data URI and
    // the images live in the buffer. Nothing is written beside the file.
    let (back, _) = read(&write(&mapped(), false));
    let (_, m) = back.materials().next().expect("the material arrived");
    assert!(m.base_color_image.is_some() && m.normal_image.is_some());
    assert_eq!(back.images().count(), 3);
}

#[test]
fn the_file_says_it_the_way_other_readers_expect() {
    // What Blender or three.js will read, checked in the JSON rather than
    // through our own importer, which is forgiving in ways they may not be.
    let json = glb_json(&write(&mapped(), true));
    assert_eq!(json["images"].as_array().map(Vec::len), Some(3));
    for image in json["images"].as_array().unwrap() {
        assert_eq!(image["mimeType"], "image/png");
        assert!(image["bufferView"].is_u64(), "an image is not in the buffer: {image}");
        assert!(image.get("uri").is_none(), "an image points outside the file");
    }
    assert_eq!(json["textures"].as_array().map(Vec::len), Some(3));
    let material = &json["materials"][0];
    let pbr = &material["pbrMetallicRoughness"];
    let texture_of = |v: &serde_json::Value| v["index"].as_u64().expect("an index") as usize;
    let image_name = |texture: usize| {
        let source = json["textures"][texture]["source"].as_u64().unwrap() as usize;
        json["images"][source]["name"].as_str().unwrap().to_string()
    };
    assert_eq!(image_name(texture_of(&pbr["baseColorTexture"])), "Albedo");
    assert_eq!(image_name(texture_of(&pbr["metallicRoughnessTexture"])), "ORM");
    assert_eq!(image_name(texture_of(&material["normalTexture"])), "Normal");
    let scale = material["normalTexture"]["scale"].as_f64().unwrap();
    assert!((scale - 0.6).abs() < 1e-6);
}

#[test]
fn a_shared_image_is_written_once() {
    // Two materials on one texture. Written twice, the file doubles and the
    // import makes two datablocks where the document had one.
    let mut doc = Document::new();
    let shared = numbered(&mut doc, "Shared", 5, ColorSpace::Srgb);
    for name in ["A", "B"] {
        let material = doc.add_material(Material {
            name: name.into(),
            base_color_image: Some(shared),
            ..Default::default()
        });
        cube_with(&mut doc, name, material);
    }
    let bytes = write(&doc, true);
    assert_eq!(glb_json(&bytes)["images"].as_array().map(Vec::len), Some(1));
    let (back, _) = read(&bytes);
    assert_eq!(back.images().count(), 1, "the import split one image into two");
}

#[test]
fn each_mesh_gets_the_materials_it_uses_as_its_own_slots() {
    // A face in the file names the *file's* material table; a face in the
    // document names its *mesh's* slot list. Two meshes with a material each
    // must come back with one slot each, pointing at the right one — not with
    // the whole table, and not both at the first.
    let mut doc = Document::new();
    let red = doc.add_material(Material {
        name: "Red".into(),
        base_color: kreate_math::SceneRgb::new(1.0, 0.0, 0.0),
        ..Default::default()
    });
    let blue = doc.add_material(Material {
        name: "Blue".into(),
        base_color: kreate_math::SceneRgb::new(0.0, 0.0, 1.0),
        ..Default::default()
    });
    cube_with(&mut doc, "RedCube", red);
    cube_with(&mut doc, "BlueCube", blue);

    let (back, _) = read(&write(&doc, true));
    for (object_name, material_name) in [("RedCube", "Red"), ("BlueCube", "Blue")] {
        let (id, _) = back
            .objects()
            .find(|(_, o)| o.name == object_name)
            .expect("the object arrived");
        let mesh = back.object(id).unwrap().mesh().expect("a mesh");
        assert_eq!(back.mesh_slots(mesh).len(), 1, "{object_name} has the wrong slot count");
        assert_eq!(
            back.material_at(id, 0).map(|m| m.name.as_str()),
            Some(material_name),
            "{object_name} wears the wrong material"
        );
        // And the faces point at that slot.
        let faces = back
            .mesh(mesh)
            .unwrap()
            .attributes()
            .get_i32(kreate_mesh::attribute::names::MATERIAL_INDEX)
            .expect("faces carry a slot")
            .to_vec();
        assert!(faces.iter().all(|f| *f == 0), "{object_name}: {faces:?}");
    }
}

#[test]
fn one_picture_in_two_spaces_is_two_datablocks() {
    // glTF lets one image be a base colour and a roughness map at once; the
    // document declares a space per image. So the import makes one of each,
    // and each reads its bytes the way its slot means them.
    let mut doc = Document::new();
    let mask = numbered(&mut doc, "Mask", 9, ColorSpace::Srgb);
    let material = doc.add_material(Material {
        name: "Both".into(),
        base_color_image: Some(mask),
        metallic_roughness_image: Some(mask),
        ..Default::default()
    });
    cube_with(&mut doc, "Cube", material);
    let bytes = write(&doc, true);
    assert_eq!(glb_json(&bytes)["images"].as_array().map(Vec::len), Some(1));

    let (back, _) = read(&bytes);
    let (_, m) = back.materials().next().unwrap();
    let colour = back.image(m.base_color_image.unwrap()).unwrap();
    let data = back.image(m.metallic_roughness_image.unwrap()).unwrap();
    assert_eq!(colour.color_space, ColorSpace::Srgb);
    assert_eq!(data.color_space, ColorSpace::Raw);
}

#[test]
fn a_float_image_goes_out_as_eight_bits_and_says_so() {
    // glTF's core has no float textures. The picture goes out clamped and
    // encoded for its slot, and the export's notes say the range was lost.
    let mut doc = Document::new();
    let mut image = Image::new("Sky", 2, 2, PixelFormat::Rgba16F);
    let half = |v: f32| half::f16::from_f32(v).to_le_bytes();
    let texel: Vec<u8> = [half(4.0), half(0.5), half(0.0), half(1.0)].concat();
    image.tile_mut(0, 0).unwrap().bytes_mut()[..8].copy_from_slice(&texel);
    let id = doc.add_image(image);
    let material = doc.add_material(Material {
        name: "Hot".into(),
        base_color_image: Some(id),
        ..Default::default()
    });
    cube_with(&mut doc, "Cube", material);

    let (scene, notes) = kreate_io::scene_from_document(&doc);
    assert!(
        notes.iter().any(|n| format!("{n:?}").contains("float")),
        "the lost range was not reported: {notes:?}"
    );
    // Clamped to one, then sRGB-encoded for a colour slot: 255, and 0.5 of
    // the light is 188.
    assert_eq!(&scene.images[0].pixels[..4], &[255, 188, 0, 255]);
}
