//! Vertex colours through glTF and back: painted per corner, written as
//! COLOR_0, read back onto the same corners.

use kreate_io::build::build_into;
use kreate_io::gltf_import::{parse_gltf, GltfOptions};
use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_scene::Document;

#[test]
fn a_painted_cube_comes_back_painted() {
    let mut cube = kreate_mesh::primitives::cube(2.0);
    // Each face its own colour, so a corner of the cube is three colours at
    // one point — the case a per-vertex colour cannot hold.
    let mut colours = Vec::new();
    for f in 0..cube.num_faces() {
        let c = glam::Vec4::new((f % 2) as f32, ((f / 2) % 2) as f32, (f as f32) / 5.0, 1.0);
        for _ in cube.face_corner_range(f) {
            colours.push(c);
        }
    }
    cube.attributes_mut().insert(
        names::COLOR,
        Domain::Corner,
        AttrData::F32x4(std::sync::Arc::new(colours)),
    );
    let mut doc = Document::new();
    doc.add_mesh_object("Painted", cube.clone());

    let (scene, _) = kreate_io::scene_from_document(&doc);
    let (bytes, _) = kreate_io::write_gltf_scene(&scene, &kreate_io::GltfExportOptions::default());
    let read = parse_gltf(&bytes, &GltfOptions::default()).expect("parses");
    let mut back = Document::new();
    build_into(&read, &mut back);
    let (_, painted) = back
        .objects()
        .find(|(_, o)| o.name == "Painted")
        .expect("the cube");
    let mesh = back.mesh(painted.mesh().unwrap()).unwrap();
    let got = mesh
        .attributes()
        .get(names::COLOR)
        .expect("its colours came back");
    let AttrData::F32x4(got) = &got.data else {
        panic!("colours as four channels")
    };
    // Face by face: each face one colour, the one it was given.
    let mut seen = std::collections::BTreeSet::new();
    for f in 0..mesh.num_faces() {
        let range = mesh.face_corner_range(f);
        let first = got[range.start];
        assert!(
            range.clone().all(|c| got[c] == first),
            "face {f} came back mottled"
        );
        seen.insert(first.to_array().map(f32::to_bits));
    }
    assert_eq!(seen.len(), 6, "six faces, six colours");
}
