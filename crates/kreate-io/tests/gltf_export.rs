//! glTF export, and the round trip that is the reason it exists.
//!
//! The tests that matter here are the ones that go all the way out and back.
//! An exporter can be checked field by field against the specification and still
//! produce a file that comes home as a different model — the failures are in the
//! conversions (axis, UV origin, winding, vertex splitting), and every one of
//! them is invisible until something reads the file.

use glam::{Vec2, Vec3};
use kreate_io::gltf_export::{write_gltf, GltfExportOptions};
use kreate_io::gltf_import::{parse_gltf, GltfOptions};
use kreate_io::ir::LossyNote;
use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_mesh::MeshData;

/// A unit cube as six quads over eight vertices.
///
/// Quads on purpose: the n-gon path is what most of this file is about, and a
/// triangulated cube would exercise none of it.
fn cube() -> MeshData {
    let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
    let positions = vec![
        p(0.0, 0.0, 0.0),
        p(1.0, 0.0, 0.0),
        p(1.0, 1.0, 0.0),
        p(0.0, 1.0, 0.0),
        p(0.0, 0.0, 1.0),
        p(1.0, 0.0, 1.0),
        p(1.0, 1.0, 1.0),
        p(0.0, 1.0, 1.0),
    ];
    let faces = vec![
        vec![0, 3, 2, 1],
        vec![4, 5, 6, 7],
        vec![0, 1, 5, 4],
        vec![1, 2, 6, 5],
        vec![2, 3, 7, 6],
        vec![3, 0, 4, 7],
    ];
    MeshData::from_faces(positions, &faces)
}

fn round_trip(mesh: &MeshData, options: &GltfExportOptions) -> kreate_io::ir::ImportScene {
    let (bytes, _) = write_gltf(&[("cube".to_string(), mesh)], &[], options);
    parse_gltf(&bytes, &GltfOptions::default()).expect("what we wrote, we can read")
}

#[test]
fn a_cube_of_quads_comes_back_a_cube_of_quads() {
    // The headline claim. glTF stores triangles, so without KREATE_ngons this
    // returns twelve triangles and the modeller's topology is gone.
    let scene = round_trip(&cube(), &GltfExportOptions::default());
    let mesh = &scene.meshes[0];
    assert_eq!(mesh.faces.len(), 6, "faces: {:?}", mesh.faces);
    assert!(mesh.faces.iter().all(|f| f.len() == 4));
    assert_eq!(
        mesh.positions.len(),
        8,
        "the exporter's vertex split should have been undone by welding"
    );
}

#[test]
fn the_json_form_round_trips_too() {
    // Two containers, one writer. The base64 data URI path is the one with a
    // hand-rolled encoder in it, so it gets its own trip rather than being
    // assumed to match the glb.
    let options = GltfExportOptions {
        binary: false,
        ..GltfExportOptions::default()
    };
    let (bytes, _) = write_gltf(&[("cube".to_string(), &cube())], &[], &options);
    assert_eq!(&bytes[..1], b"{", "the json form should be json");
    let scene = parse_gltf(&bytes, &GltfOptions::default()).expect("reads");
    assert_eq!(scene.meshes[0].faces.len(), 6);
    assert_eq!(scene.meshes[0].positions.len(), 8);
}

#[test]
fn the_binary_form_is_a_valid_glb_container() {
    let (bytes, _) = write_gltf(
        &[("cube".to_string(), &cube())],
        &[],
        &GltfExportOptions::default(),
    );
    assert_eq!(&bytes[0..4], b"glTF");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
        bytes.len(),
        "the declared length should be the actual length"
    );
    assert_eq!(
        bytes.len() % 4,
        0,
        "a glb is padded to four bytes; a reader walking chunks will misread otherwise"
    );
}

#[test]
fn positions_survive_the_axis_change() {
    // Kreate is Z-up, glTF is Y-up. The conversion happens twice — once out,
    // once back — and the only way to catch a wrong matrix is a point that is
    // different in all three coordinates.
    let positions = vec![
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(-4.0, 5.0, -6.0),
        Vec3::new(7.0, -8.0, 9.0),
    ];
    let mesh = MeshData::from_faces(positions.clone(), &[vec![0, 1, 2]]);
    let scene = round_trip(&mesh, &GltfExportOptions::default());
    // Through `into_mesh_data`, because that is where the importer applies the
    // file's axis convention: the IR deliberately holds source-space positions
    // plus the matrix, so that one conversion happens in one place. Reading
    // `scene.meshes[0].positions` directly reads glTF space and would fail here
    // for the wrong reason.
    let (meshes, _) = scene.into_mesh_data();
    let got = meshes[0].1.positions();
    assert_eq!(got.len(), 3);
    for (a, b) in positions.iter().zip(got) {
        assert!(
            a.distance(*b) < 1e-5,
            "{a} came back as {b}; the axis conversion is not its own inverse"
        );
    }
}

#[test]
fn winding_is_not_flipped() {
    // Z-up to Y-up is a rotation, so it must not reverse face winding. If it
    // did, every exported model would be inside out — visible only with
    // backface culling on, which is exactly when a user notices and cannot
    // explain it.
    let mesh = MeshData::from_faces(vec![Vec3::ZERO, Vec3::X, Vec3::Y], &[vec![0, 1, 2]]);
    let before = mesh.compute_face_normals()[0].normalize();
    let scene = round_trip(&mesh, &GltfExportOptions::default());
    let (meshes, _) = scene.into_mesh_data();
    let after = meshes[0].1.compute_face_normals()[0].normalize();
    assert!(
        before.dot(after) > 0.99,
        "normal went from {before} to {after}"
    );
}

#[test]
fn uvs_come_back_the_same_way_up() {
    // glTF's origin is the top left and Kreate's is the bottom left, so the
    // exporter flips V and the importer flips it back. Getting one of the two
    // wrong is invisible on a symmetric texture and catastrophic on lettering.
    let mut mesh = MeshData::from_faces(
        vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ONE],
        &[vec![0, 1, 2, 3]],
    );
    let uvs = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.25),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 0.75),
    ];
    mesh.attributes_mut().insert(
        names::UV,
        Domain::Corner,
        AttrData::F32x2(uvs.clone().into()),
    );

    let scene = round_trip(&mesh, &GltfExportOptions::default());
    // Through `into_mesh_data`, where the IR applies `uv_origin`. Reading the
    // IR's own coordinates here would compare a half-converted value, and did:
    // this test passed against an importer that flipped V a second time, because
    // the two flips cancelled at exactly the point it was looking.
    let (meshes, _) = scene.into_mesh_data();
    let got = meshes[0]
        .1
        .attributes()
        .get_f32x2(names::UV)
        .expect("uvs survived");
    assert_eq!(got.len(), 4);
    for (a, b) in uvs.iter().zip(got) {
        assert!(
            a.distance(*b) < 1e-6,
            "uv {a} came back as {b}; the V flip is not its own inverse"
        );
    }
}

#[test]
fn a_seam_splits_vertices_on_the_way_out() {
    // Two faces sharing an edge but disagreeing about its UVs. glTF stores one
    // UV per vertex, so the shared vertices have to be split or the seam is
    // smeared across both faces. The split is undone by welding on the way back,
    // which is why the round trip alone cannot see it — the file has to be read
    // for its vertex count.
    let mut mesh = MeshData::from_faces(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        ],
        &[vec![0, 1, 2, 3], vec![1, 4, 5, 2]],
    );
    // Corners 1 and 2 (of the first face) sit on vertices 1 and 2; corners 4 and
    // 7 (of the second) sit on the same vertices with different UVs.
    let uvs = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(0.5, 0.0),
        Vec2::new(0.9, 0.0),
        Vec2::new(0.9, 1.0),
        Vec2::new(0.5, 1.0),
    ];
    mesh.attributes_mut()
        .insert(names::UV, Domain::Corner, AttrData::F32x2(uvs.into()));

    let options = GltfExportOptions {
        normals: false,
        ..GltfExportOptions::default()
    };
    let (bytes, _) = write_gltf(&[("seam".to_string(), &mesh)], &[], &options);
    // Read without welding so the file's own vertex count is what comes back.
    let scene = parse_gltf(
        &bytes,
        &GltfOptions {
            weld: false,
            ..GltfOptions::default()
        },
    )
    .expect("reads");
    assert_eq!(
        scene.meshes[0].positions.len(),
        8,
        "six vertices, two of them split by the seam, should be eight in the file"
    );
    let uvs = scene.meshes[0].corner_uvs.as_ref().expect("uvs");
    assert_eq!(uvs.len(), 8);
    assert!(
        (uvs[1].x - 1.0).abs() < 1e-6 && (uvs[4].x - 0.5).abs() < 1e-6,
        "the two sides of the seam kept different UVs: {:?}",
        uvs
    );
}

#[test]
fn a_cube_splits_its_normals_per_face() {
    // The other half of the split. A cube's corner has three normals, and a
    // vertex carries one, so eight vertices have to become twenty-four. Without
    // the split a cube shades as a sphere — the single most recognisable
    // exporter bug there is.
    let (bytes, _) = write_gltf(
        &[("cube".to_string(), &cube())],
        &[],
        &GltfExportOptions::default(),
    );
    let scene = parse_gltf(
        &bytes,
        &GltfOptions {
            weld: false,
            ..GltfOptions::default()
        },
    )
    .expect("reads");
    assert_eq!(scene.meshes[0].positions.len(), 24);
    let normals = scene.meshes[0].corner_normals.as_ref().expect("normals");
    // Every corner of a face shares that face's normal, and the six faces point
    // six different ways.
    let mut distinct: Vec<[i32; 3]> = normals
        .iter()
        .map(|n| [n.x.round() as i32, n.y.round() as i32, n.z.round() as i32])
        .collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 6, "got {distinct:?}");
}

#[test]
fn faces_are_split_into_primitives_by_material() {
    let mut mesh = cube();
    let materials: Vec<i32> = vec![0, 0, 1, 1, 2, 2];
    mesh.attributes_mut().insert(
        names::MATERIAL_INDEX,
        Domain::Face,
        AttrData::I32(materials.clone().into()),
    );
    let scene = round_trip(&mesh, &GltfExportOptions::default());
    let got = scene.meshes[0]
        .face_materials
        .as_ref()
        .expect("materials survived");
    // The faces are regrouped by material on the way out, so the order changes
    // but the partition does not.
    let mut counts = [0usize; 3];
    for m in got {
        counts[*m as usize] += 1;
    }
    assert_eq!(counts, [2, 2, 2], "got {got:?}");
    assert_eq!(scene.materials.len(), 3);
}

#[test]
fn a_material_index_with_no_table_still_partitions_the_faces() {
    // `write_gltf` is given an empty material table here, which is what the CLI
    // does. The indices are still real authored data, so they get placeholder
    // materials rather than being dropped — a face that loses which material it
    // belonged to cannot be given it back.
    let mut mesh = cube();
    mesh.attributes_mut().insert(
        names::MATERIAL_INDEX,
        Domain::Face,
        AttrData::I32(vec![0, 0, 0, 7, 7, 7].into()),
    );
    let (bytes, _) = write_gltf(&[("cube".to_string(), &mesh)], &[], &Default::default());
    let scene = parse_gltf(&bytes, &GltfOptions::default()).expect("reads");
    assert_eq!(scene.materials.len(), 2);
    let got = scene.meshes[0].face_materials.as_ref().unwrap();
    let mut sorted = got.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 0, 0, 1, 1, 1], "got {got:?}");
}

#[test]
fn without_the_extension_a_quad_comes_back_as_triangles() {
    // The negative control for the headline test. If this also returned quads,
    // the round trip would be proving nothing about KREATE_ngons.
    let options = GltfExportOptions {
        record_ngons: false,
        ..GltfExportOptions::default()
    };
    let scene = round_trip(&cube(), &options);
    assert_eq!(scene.meshes[0].faces.len(), 12);
    assert!(scene.meshes[0].faces.iter().all(|f| f.len() == 3));
}

#[test]
fn dropping_the_ngons_is_reported() {
    let options = GltfExportOptions {
        record_ngons: false,
        ..GltfExportOptions::default()
    };
    let (_, notes) = write_gltf(&[("cube".to_string(), &cube())], &[], &options);
    assert!(
        notes
            .iter()
            .any(|n| matches!(n, LossyNote::Unsupported { what } if what.contains("n-gons"))),
        "{notes:?}"
    );
}

#[test]
fn a_damaged_ngon_record_falls_back_to_the_triangles() {
    // The extension is an accelerator, not a requirement. A record whose two
    // arrays disagree would read every face shifted, so it has to be refused —
    // and refusing it must mean "use the triangles", not "fail to open".
    let (bytes, _) = write_gltf(
        &[("cube".to_string(), &cube())],
        &[],
        &GltfExportOptions {
            binary: false,
            ..GltfExportOptions::default()
        },
    );
    let mut root: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let sizes = root["meshes"][0]["primitives"][0]["extensions"]["KREATE_ngons"]["faceSizes"]
        .as_u64()
        .expect("the record is there to damage") as usize;
    // Claim one face fewer than the vertex list accounts for.
    let count = root["accessors"][sizes]["count"].as_u64().unwrap();
    root["accessors"][sizes]["count"] = serde_json::json!(count - 1);
    let bytes = serde_json::to_vec(&root).unwrap();

    let scene = parse_gltf(&bytes, &GltfOptions::default()).expect("still opens");
    assert_eq!(
        scene.meshes[0].faces.len(),
        12,
        "the triangles should have been used"
    );
    assert!(
        scene.source.notes.iter().any(|n| matches!(
            n,
            LossyNote::Unsupported { what } if what.contains("KREATE_ngons")
        )),
        "the fallback was silent: {:?}",
        scene.source.notes
    );
}

#[test]
fn the_extension_is_used_but_not_required() {
    // A file that lists KREATE_ngons as required is a file no other program will
    // open, in exchange for nothing — the triangles are complete and correct
    // without it.
    let (bytes, _) = write_gltf(
        &[("cube".to_string(), &cube())],
        &[],
        &GltfExportOptions {
            binary: false,
            ..GltfExportOptions::default()
        },
    );
    let root: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        root["extensionsUsed"],
        serde_json::json!(["KREATE_ngons"]),
        "{}",
        root["extensionsUsed"]
    );
    assert!(root.get("extensionsRequired").is_none());
}

#[test]
fn buffer_views_are_four_byte_aligned() {
    // Accessor offsets must be aligned to their component size, and a reader
    // that maps the buffer rather than copying it faults on a misaligned view.
    //
    // This checks the property on real output. It cannot currently fail, because
    // every buffer this exporter emits is already a multiple of four bytes; the
    // padding itself is tested directly in `gltf_export`'s own test module. This
    // one starts earning its keep the day an attribute arrives that is not.
    let mut mesh = cube();
    // An odd-length scalar buffer between the vector ones is what makes the
    // padding matter; without one every view lands on 4 by accident.
    mesh.attributes_mut().insert(
        names::MATERIAL_INDEX,
        Domain::Face,
        AttrData::I32(vec![0, 1, 0, 1, 0, 1].into()),
    );
    let (bytes, _) = write_gltf(
        &[("cube".to_string(), &mesh)],
        &[],
        &GltfExportOptions {
            binary: false,
            ..GltfExportOptions::default()
        },
    );
    let root: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for view in root["bufferViews"].as_array().unwrap() {
        let offset = view["byteOffset"].as_u64().unwrap_or(0);
        assert_eq!(offset % 4, 0, "view at {offset} is not aligned");
    }
}

#[test]
fn positions_declare_their_bounds() {
    // Required by the specification for POSITION, and used: a viewer frames the
    // model from these without decoding the buffer.
    let (bytes, _) = write_gltf(
        &[("cube".to_string(), &cube())],
        &[],
        &GltfExportOptions {
            binary: false,
            ..GltfExportOptions::default()
        },
    );
    let root: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let position = root["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        .as_u64()
        .unwrap() as usize;
    let accessor = &root["accessors"][position];
    assert_eq!(accessor["min"], serde_json::json!([0.0, 0.0, -1.0]));
    assert_eq!(accessor["max"], serde_json::json!([1.0, 1.0, 0.0]));
}

#[test]
fn creases_and_seams_are_reported_as_dropped() {
    let mut mesh = cube();
    mesh.rebuild_edges();
    let edges = mesh.num_edges() as usize;
    mesh.attributes_mut().insert(
        names::SEAM,
        Domain::Edge,
        AttrData::Bool(vec![false; edges].into()),
    );
    let (_, notes) = write_gltf(&[("cube".to_string(), &mesh)], &[], &Default::default());
    assert!(
        notes
            .iter()
            .any(|n| matches!(n, LossyNote::Unsupported { what } if what.contains("seams"))),
        "{notes:?}"
    );
}

#[test]
fn an_empty_mesh_produces_a_readable_file() {
    let (bytes, notes) = write_gltf(
        &[("nothing".to_string(), &MeshData::empty())],
        &[],
        &Default::default(),
    );
    assert!(!notes.is_empty(), "silence about an empty export");
    let scene = parse_gltf(&bytes, &GltfOptions::default()).expect("an empty file still opens");
    assert!(scene.meshes.is_empty());
}

#[test]
fn two_meshes_each_get_a_node() {
    let (bytes, _) = write_gltf(
        &[
            ("first".to_string(), &cube()),
            ("second".to_string(), &cube()),
        ],
        &[],
        &Default::default(),
    );
    let scene = parse_gltf(&bytes, &GltfOptions::default()).expect("reads");
    assert_eq!(scene.meshes.len(), 2);
    assert_eq!(scene.meshes[0].name, "first");
    assert_eq!(scene.meshes[1].name, "second");
}

#[test]
fn the_same_mesh_writes_the_same_bytes() {
    // Determinism is a file-format property, not a nicety: §5.3 wants an
    // unchanged document to hash the same so version control stores one blob.
    //
    // Several materials, because that is what there is to put in the wrong
    // order. A single-material mesh has one primitive and writes the same bytes
    // however the grouping is stored — which it did, and the test passed while a
    // hash-ordered grouping went undetected.
    let mut mesh = cube();
    mesh.attributes_mut().insert(
        names::MATERIAL_INDEX,
        Domain::Face,
        AttrData::I32(vec![3, 1, 4, 1, 5, 9].into()),
    );
    let a = write_gltf(&[("cube".to_string(), &mesh)], &[], &Default::default()).0;
    let b = write_gltf(&[("cube".to_string(), &mesh)], &[], &Default::default()).0;
    assert_eq!(a, b);
}
