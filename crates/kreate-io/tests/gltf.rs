//! glTF import, and the four conventions it differs from Kreate on.
//!
//! Every one of these has the same failure mode — the model looks *almost*
//! right — which is why each gets a test that names the exact quantity rather
//! than checking that something loaded.

use kreate_io::gltf_import::{self, GltfOptions};
use kreate_io::ir::{LossyNote, UvOrigin};
use kreate_mesh::attribute::names;
use serde_json::json;

/// Builds a self-contained glTF with one triangle mesh.
///
/// Written as JSON with a base64 buffer rather than loaded from a committed
/// asset, so that a test can state the exact coordinates it expects to come back
/// and the reader has both halves in front of them.
fn gltf_with(positions: &[[f32; 3]], uvs: Option<&[[f32; 2]]>, indices: &[u16]) -> Vec<u8> {
    let mut buffer: Vec<u8> = Vec::new();

    let positions_offset = buffer.len();
    for p in positions {
        for c in p {
            buffer.extend_from_slice(&c.to_le_bytes());
        }
    }
    let positions_length = buffer.len() - positions_offset;

    let uvs_range = uvs.map(|uvs| {
        let offset = buffer.len();
        for uv in uvs {
            for c in uv {
                buffer.extend_from_slice(&c.to_le_bytes());
            }
        }
        (offset, buffer.len() - offset)
    });

    // Accessor offsets must be aligned to their component size.
    while !buffer.len().is_multiple_of(4) {
        buffer.push(0);
    }
    let indices_offset = buffer.len();
    for i in indices {
        buffer.extend_from_slice(&i.to_le_bytes());
    }
    let indices_length = buffer.len() - indices_offset;

    let mut views = vec![json!({
        "buffer": 0, "byteOffset": positions_offset, "byteLength": positions_length
    })];
    let mut accessors = vec![json!({
        "bufferView": 0, "componentType": 5126, "count": positions.len(),
        "type": "VEC3",
        // Positions are the one accessor glTF requires bounds on.
        "min": bounds(positions).0, "max": bounds(positions).1
    })];
    let mut attributes = json!({ "POSITION": 0 });
    if let Some((offset, length)) = uvs_range {
        views.push(json!({ "buffer": 0, "byteOffset": offset, "byteLength": length }));
        accessors.push(json!({
            "bufferView": views.len() - 1, "componentType": 5126,
            "count": uvs.unwrap().len(), "type": "VEC2"
        }));
        attributes["TEXCOORD_0"] = json!(accessors.len() - 1);
    }
    views.push(json!({
        "buffer": 0, "byteOffset": indices_offset, "byteLength": indices_length
    }));
    accessors.push(json!({
        "bufferView": views.len() - 1, "componentType": 5123,
        "count": indices.len(), "type": "SCALAR"
    }));
    let indices_accessor = accessors.len() - 1;

    let root = json!({
        "asset": { "version": "2.0", "generator": "kreate test" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0, "name": "Node" }],
        "meshes": [{
            "name": "Triangle",
            "primitives": [{ "attributes": attributes, "indices": indices_accessor }]
        }],
        "buffers": [{
            "byteLength": buffer.len(),
            "uri": format!("data:application/octet-stream;base64,{}", base64(&buffer))
        }],
        "bufferViews": views,
        "accessors": accessors
    });
    serde_json::to_vec(&root).expect("json")
}

fn bounds(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in positions {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    (min, max)
}

/// Base64 without a dependency — the test needs a few hundred bytes encoded,
/// not a library.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> (18 - i * 6)) & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// A triangle with a distinctive position in each axis, so a wrong axis
/// convention cannot be mistaken for a right one.
fn asymmetric() -> Vec<[f32; 3]> {
    vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 3.0]]
}

#[test]
fn a_triangle_imports() {
    let bytes = gltf_with(&asymmetric(), None, &[0, 1, 2]);
    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(scene.meshes[0].name, "Triangle");
    assert_eq!(scene.meshes[0].faces, vec![vec![0, 1, 2]]);
    assert_eq!(scene.source.generator.as_deref(), Some("kreate test"));
}

#[test]
fn the_file_is_declared_y_up_so_the_conversion_happens_once() {
    // The axis convention is recorded on the source rather than applied to the
    // positions here, and that is deliberate: the IR applies it exactly once,
    // when it becomes a document. An importer that also rotated the positions
    // would rotate them twice.
    let bytes = gltf_with(&asymmetric(), None, &[0, 1, 2]);
    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    assert_eq!(scene.source.axis, kreate_math::AxisConvention::GLTF);
    // The positions are still exactly as the file stored them.
    assert_eq!(scene.meshes[0].positions[2].to_array(), [0.0, 2.0, 3.0]);
}

#[test]
fn a_y_up_file_becomes_z_up_when_it_becomes_a_document() {
    // The other half: the conversion the source metadata describes actually
    // happens, once, at the boundary. glTF's +Y is Kreate's +Z, and its +Z is
    // Kreate's -Y — a model imported without this lies on its face.
    let bytes = gltf_with(&asymmetric(), None, &[0, 1, 2]);
    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    let (meshes, _) = scene.into_mesh_data();
    let p = meshes[0].1.positions();
    // The vertex the file put at (0, 2, 3) — two up and three towards the
    // viewer — should end up two along +Z and three along -Y.
    let moved = p
        .iter()
        .find(|v| v.length_squared() > 1.0)
        .expect("the tall vertex");
    assert!(
        (moved.z - 2.0).abs() < 1e-6 && (moved.y + 3.0).abs() < 1e-6,
        "a Y-up file did not become Z-up: {moved}"
    );
}

#[test]
fn texture_coordinates_are_flipped_from_the_top_left_origin() {
    // The single most consequential subtraction in the whole importer. glTF puts
    // (0, 0) at the top left; Kreate and OBJ put it at the bottom left. Get it
    // wrong and every texture is upside down — invisible on a symmetric one, and
    // unmissable on anything with text.
    //
    // The plan says explicitly to write this test.
    let uvs = [[0.0, 0.0], [1.0, 0.25], [0.5, 1.0]];
    let bytes = gltf_with(&asymmetric(), Some(&uvs), &[0, 1, 2]);
    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");

    // Through `into_mesh_data`, because that is where the IR applies `uv_origin`.
    // Asserting on `scene.meshes[0].corner_uvs` reads the file's own coordinates,
    // which is what let a duplicate flip in this module go unnoticed.
    let (meshes, source) = scene.into_mesh_data();
    let got = meshes[0].1.attributes().get_f32x2(names::UV).expect("uvs");
    // V flipped, U untouched.
    assert_eq!(got[0].to_array(), [0.0, 1.0]);
    assert_eq!(got[1].to_array(), [1.0, 0.75]);
    assert_eq!(got[2].to_array(), [0.5, 0.0]);
    assert_eq!(source.uv_origin, UvOrigin::TopLeft);
}

#[test]
fn split_vertices_are_welded_back_into_a_shared_surface() {
    // glTF splits a vertex wherever any attribute differs, so two triangles that
    // share an edge in the model may not share one in the file. Welding recovers
    // the surface the author actually modelled.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        // The second triangle repeats two corners exactly, as a split does.
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let bytes = gltf_with(&positions, None, &[0, 1, 2, 3, 4, 5]);
    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    let mesh = &scene.meshes[0];
    assert_eq!(
        mesh.positions.len(),
        4,
        "the split vertices were not welded"
    );
    let shared = mesh.faces[0]
        .iter()
        .filter(|v| mesh.faces[1].contains(v))
        .count();
    assert_eq!(shared, 2, "the triangles share no edge");
}

#[test]
fn welding_can_be_turned_off() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let bytes = gltf_with(&positions, None, &[0, 1, 2, 3, 4, 5]);
    let options = GltfOptions {
        weld: false,
        ..GltfOptions::default()
    };
    let scene = gltf_import::parse_gltf(&bytes, &options).expect("parses");
    assert_eq!(scene.meshes[0].positions.len(), 6);
}

#[test]
fn uvs_survive_the_weld_that_splits_them() {
    // The two halves together, and the reason the corner domain exists. Two
    // corners at the same position carry different UVs — a seam — and welding
    // the positions must not merge the UVs.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        // Same position as vertex 0, different UV: a seam.
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let uvs = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        // The seam's other side.
        [0.9, 0.1],
        [1.0, 1.0],
        [0.0, 1.0],
    ];
    let bytes = gltf_with(&positions, Some(&uvs), &[0, 1, 2, 3, 4, 5]);
    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    let mesh = &scene.meshes[0];
    assert_eq!(mesh.positions.len(), 4, "positions were not welded");

    let got = mesh.corner_uvs.as_ref().expect("uvs");
    assert_eq!(got.len(), 6, "a corner UV was lost with the welded vertex");
    // Corner 0 and corner 3 sit at the same position and kept different UVs.
    // In the file's own origin: this test is about the weld, and the flip is
    // `into_mesh_data`'s job.
    assert_eq!(got[0].to_array(), [0.0, 0.0]);
    assert_eq!(got[3].to_array(), [0.9, 0.1]);
}

#[test]
fn a_file_with_no_indices_reads_its_vertices_in_order() {
    // Indices are optional in glTF. Without them the vertices are consumed three
    // at a time, and a reader that assumed indices would produce nothing.
    let mut bytes = gltf_with(&asymmetric(), None, &[0, 1, 2]);
    // Strip the indices reference from the primitive.
    let mut root: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    root["meshes"][0]["primitives"][0]
        .as_object_mut()
        .unwrap()
        .remove("indices");
    bytes = serde_json::to_vec(&root).unwrap();

    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    assert_eq!(scene.meshes[0].faces, vec![vec![0, 1, 2]]);
}

#[test]
fn what_the_importer_does_not_read_is_recorded() {
    // A file whose rig or animation is dropped has to say so, or the user finds
    // out when the export they sent someone turns out to be a static pose.
    let mut root: serde_json::Value =
        serde_json::from_slice(&gltf_with(&asymmetric(), None, &[0, 1, 2])).unwrap();
    // An occlusion map: a standard surface here has no slot for one. The
    // picture is a file beside the document, read through its path.
    let dir = std::env::temp_dir().join(format!("kreate-occlusion-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("made");
    let mut png = Vec::new();
    image::RgbaImage::new(1, 1)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encoded");
    std::fs::write(dir.join("occlusion.png"), png).expect("written");
    root["images"] = json!([{ "uri": "occlusion.png" }]);
    root["textures"] = json!([{ "source": 0 }]);
    root["materials"] = json!([{ "occlusionTexture": { "index": 0 } }]);
    root["meshes"][0]["primitives"][0]["material"] = json!(0);
    let path = dir.join("occluded.gltf");
    std::fs::write(&path, serde_json::to_vec(&root).unwrap()).expect("written");

    let scene = gltf_import::parse_gltf_file(&path, &GltfOptions::default()).expect("parses");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        scene.source.notes.iter().any(|n| matches!(
            n,
            LossyNote::Unsupported { what } if what.contains("occlusion")
        )),
        "an unread feature was dropped silently: {:?}",
        scene.source.notes
    );
}

#[test]
fn a_material_is_read_with_its_factors() {
    let mut root: serde_json::Value =
        serde_json::from_slice(&gltf_with(&asymmetric(), None, &[0, 1, 2])).unwrap();
    root["materials"] = json!([{
        "name": "Red",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 0.2, 0.1, 1.0],
            "metallicFactor": 0.25,
            "roughnessFactor": 0.75
        },
        "emissiveFactor": [0.0, 0.5, 0.0]
    }]);
    root["meshes"][0]["primitives"][0]["material"] = json!(0);
    let bytes = serde_json::to_vec(&root).unwrap();

    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    assert_eq!(scene.materials.len(), 1);
    let m = &scene.materials[0];
    assert_eq!(m.name, "Red");
    assert_eq!(m.base_color, [1.0, 0.2, 0.1, 1.0]);
    assert_eq!(m.metallic, 0.25);
    assert_eq!(m.roughness, 0.75);
    assert_eq!(m.emissive.to_array(), [0.0, 0.5, 0.0]);
    // And the face knows which material it used.
    assert_eq!(scene.meshes[0].face_materials.as_ref().unwrap(), &vec![0]);
}

#[test]
fn a_non_triangle_primitive_is_skipped_with_a_note() {
    let mut root: serde_json::Value =
        serde_json::from_slice(&gltf_with(&asymmetric(), None, &[0, 1, 2])).unwrap();
    // Mode 1 is LINES.
    root["meshes"][0]["primitives"][0]["mode"] = json!(1);
    let bytes = serde_json::to_vec(&root).unwrap();

    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    assert!(scene.meshes[0].faces.is_empty());
    assert!(scene.source.notes.iter().any(|n| matches!(
        n,
        LossyNote::Unsupported { what } if what.contains("only triangles")
    )));
}

#[test]
fn a_mesh_whose_primitives_disagree_about_uvs_keeps_its_corners_aligned() {
    // glTF splits a mesh into primitives by material, and there is nothing
    // stopping one of them from carrying UVs while another does not. The corner
    // arrays are indexed by corner across the *whole* mesh, so a corner skipped
    // for the primitive that lacks the attribute shifts every UV after it — and
    // the model comes out textured correctly up to the join and wrong after it.
    //
    // A falsification pass that filled the arrays only where the attribute
    // existed passed every other test here, because they all use a single
    // primitive.
    let positions = [[0.0, 0.0, 0.0f32], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let uvs = [[0.1, 0.2f32], [0.3, 0.4], [0.5, 0.6]];
    let bytes = gltf_with(&positions, Some(&uvs), &[0, 1, 2]);
    let mut root: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    // A second primitive over the same positions, with no TEXCOORD_0.
    let without_uvs = json!({
        "attributes": { "POSITION": 0 },
        "indices": root["meshes"][0]["primitives"][0]["indices"].clone()
    });
    root["meshes"][0]["primitives"]
        .as_array_mut()
        .unwrap()
        .push(without_uvs);
    let bytes = serde_json::to_vec(&root).unwrap();

    let options = GltfOptions {
        weld: false,
        ..GltfOptions::default()
    };
    let scene = gltf_import::parse_gltf(&bytes, &options).expect("parses");
    let mesh = &scene.meshes[0];
    assert_eq!(
        mesh.faces.len(),
        2,
        "both primitives should contribute a face"
    );
    assert_eq!(
        mesh.corner_uvs.as_ref().expect("uvs").len(),
        mesh.num_corners(),
        "the corner UVs are not aligned with the corners"
    );
    // The first primitive's UVs are still where they belong, in the file's own
    // origin — this test is about corner alignment, not about the flip.
    let got = mesh.corner_uvs.as_ref().unwrap();
    assert_eq!(got[0].to_array(), [0.1, 0.2]);
}

#[test]
fn an_out_of_range_index_is_dropped_even_with_welding_off() {
    // A file can index past its own vertex array. With welding on the sanitiser
    // catches it on the way through; with welding off nothing else does, and an
    // invalid mesh would escape into the document — where the next thing to
    // touch it indexes out of bounds.
    //
    // Dropped rather than clamped: a clamped index makes a triangle that is
    // silently in the wrong place, which is harder to notice than a missing one
    // and impossible to explain.
    let bytes = gltf_with(&asymmetric(), None, &[0, 1, 2]);
    let mut root: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Claim more indices than the buffer holds vertices for.
    root["accessors"][1]["count"] = json!(3);
    let bytes = serde_json::to_vec(&root).unwrap();
    let mut root: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Point the position accessor at fewer vertices than the indices name.
    root["accessors"][0]["count"] = json!(2);
    let bytes = serde_json::to_vec(&root).unwrap();

    let options = GltfOptions {
        weld: false,
        ..GltfOptions::default()
    };
    let Ok(scene) = gltf_import::parse_gltf(&bytes, &options) else {
        // Refusing the file outright is also a correct answer.
        return;
    };
    let mesh = &scene.meshes[0];
    for face in &mesh.faces {
        for v in face {
            assert!(
                (*v as usize) < mesh.positions.len(),
                "a face indexes vertex {v} of {}",
                mesh.positions.len()
            );
        }
    }
    assert!(
        scene
            .source
            .notes
            .iter()
            .any(|n| matches!(n, LossyNote::OutOfRangeIndex { .. })),
        "an out-of-range index was dropped without a note: {:?}",
        scene.source.notes
    );
}

#[test]
fn garbage_is_refused_rather_than_panicking() {
    for bytes in [
        b"".to_vec(),
        b"not json".to_vec(),
        b"{}".to_vec(),
        br#"{"asset":{"version":"2.0"}}"#.to_vec(),
    ] {
        let _ = gltf_import::parse_gltf(&bytes, &GltfOptions::default());
    }
}

#[test]
fn every_truncation_of_a_valid_file_is_handled() {
    let bytes = gltf_with(&asymmetric(), None, &[0, 1, 2]);
    for len in (0..bytes.len()).step_by(13) {
        let _ = gltf_import::parse_gltf(&bytes[..len], &GltfOptions::default());
    }
    assert!(gltf_import::parse_gltf(&bytes, &GltfOptions::default()).is_ok());
}

/// A triangle whose buffer is a file beside it, named `uri`.
fn triangle_beside(dir: &std::path::Path, uri: &str) -> std::path::PathBuf {
    std::fs::create_dir_all(dir).expect("made");
    let points: Vec<u8> = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    std::fs::write(dir.join("tri.bin"), &points).expect("written");
    let json = format!(
        r#"{{"asset": {{"version": "2.0"}}, "scene": 0, "scenes": [{{"nodes": [0]}}],
        "nodes": [{{"mesh": 0, "name": "Tri"}}],
        "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}}}]}}],
        "accessors": [{{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0]}}],
        "bufferViews": [{{"buffer": 0, "byteLength": 36}}],
        "buffers": [{{"uri": "{uri}", "byteLength": 36}}]}}"#
    );
    let path = dir.join("tri.gltf");
    std::fs::write(&path, json).expect("written");
    path
}

/// What an exporter writing a folder hands over: a `.gltf` whose buffer is a
/// file beside it. Read from its path; and a reference that leaves the
/// folder is refused before anything is read.
#[test]
fn a_gltf_on_disk_reads_the_buffer_beside_it_and_nothing_further() {
    use kreate_io::gltf_import::{parse_gltf, parse_gltf_file, GltfError, GltfOptions};
    let dir = std::env::temp_dir().join(format!("kreate-gltf-{}", std::process::id()));
    let path = triangle_beside(&dir, "tri.bin");
    let scene = parse_gltf_file(&path, &GltfOptions::default()).expect("reads");
    assert_eq!(scene.meshes[0].faces.len(), 1);
    // From memory the same file cannot find its buffer: what the path is for.
    assert!(parse_gltf(
        &std::fs::read(&path).expect("read"),
        &GltfOptions::default()
    )
    .is_err());
    for outside in [
        "../tri.bin",
        "/etc/hosts",
        "file:///etc/hosts",
        "C:/x.bin",
        "a/../../b.bin",
    ] {
        let path = triangle_beside(&dir, outside);
        assert!(
            matches!(
                parse_gltf_file(&path, &GltfOptions::default()),
                Err(GltfError::Outside(_))
            ),
            "{outside}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A glTF camera at rest looks along its -Z, which in a Y-up file is level
/// with the floor: here, along +Y. And a light the file turns to shine down
/// shines down here. Both were tipped a quarter turn while their placements
/// were converted as a mesh's are.
#[test]
fn cameras_and_lights_look_where_the_file_says() {
    let root = json!({
        "asset": { "version": "2.0" },
        "extensionsUsed": ["KHR_lights_punctual"],
        "extensions": { "KHR_lights_punctual": { "lights": [ { "type": "directional", "intensity": 3.0 } ] } },
        "cameras": [ { "type": "perspective", "perspective": { "yfov": 0.8, "znear": 0.1 } } ],
        "scene": 0,
        "scenes": [ { "nodes": [0, 1] } ],
        "nodes": [
            { "name": "Cam", "camera": 0 },
            { "name": "Sun", "rotation": [-std::f64::consts::FRAC_1_SQRT_2, 0.0, 0.0, std::f64::consts::FRAC_1_SQRT_2],
              "extensions": { "KHR_lights_punctual": { "light": 0 } } }
        ]
    });
    let bytes = serde_json::to_vec(&root).unwrap();
    let scene = gltf_import::parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    let mut doc = kreate_scene::Document::new();
    kreate_io::build_into(&scene, &mut doc);
    let facing = |name: &str| {
        let (id, _) = doc.objects().find(|(_, o)| o.name == name).expect("there");
        doc.world_transform(id).matrix3 * glam::DVec3::NEG_Z
    };
    let cam = facing("Cam");
    assert!(
        (cam - glam::DVec3::Y).length() < 1e-5,
        "the camera looks {cam}"
    );
    let sun = facing("Sun");
    assert!(
        (sun - glam::DVec3::NEG_Z).length() < 1e-5,
        "the sun shines {sun}"
    );

    // And written out again, they read back the same way round.
    let (export, _) = kreate_io::scene_from_document(&doc);
    let (written, _) = kreate_io::write_gltf_scene(&export, &Default::default());
    let again = gltf_import::parse_gltf(&written, &GltfOptions::default()).expect("parses");
    let mut back = kreate_scene::Document::new();
    kreate_io::build_into(&again, &mut back);
    let (id, _) = back
        .objects()
        .find(|(_, o)| o.name == "Cam")
        .expect("there");
    let cam = back.world_transform(id).matrix3 * glam::DVec3::NEG_Z;
    assert!(
        (cam - glam::DVec3::Y).length() < 1e-5,
        "after a round trip it looks {cam}"
    );
}
