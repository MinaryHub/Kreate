//! Parts linked from a library file, in a shared session.

use kreate_collab::replica::Replica;
use kreate_scene::Document;

/// A part linked from a library file reaches a participant who does not
/// have that file: the session carries its geometry, and it stays marked
/// read-only.
#[test]
fn a_linked_part_arrives_whole_without_its_library() {
    use kreate_scene::document::Linkage;
    let mut a = Document::new();
    let part = a.add_mesh_object("Bolt", kreate_mesh::primitives::cube(0.01));
    a.object_mut(part).unwrap().linkage = Linkage::Linked {
        library: "C:/nowhere/parts-library.kreate".into(),
        name: "Bolt".into(),
    };
    let mut ra = Replica::new(&mut a, 1);
    let seed = ra.seed(&a).unwrap();
    ra.ack();
    let mut b = Document::new();
    let mut rb = Replica::new(&mut b, 2);
    rb.adopt_snapshot(&mut b, &seed).unwrap();

    let check = |b: &Document, faces: u32| {
        let (_, o) = b
            .objects()
            .find(|(_, o)| o.name == "Bolt")
            .expect("arrived");
        assert!(
            matches!(&o.linkage, Linkage::Linked { name, .. } if name == "Bolt"),
            "still read-only: {:?}",
            o.linkage
        );
        let mesh = b.mesh(o.mesh().expect("a mesh")).expect("its geometry");
        assert_eq!(mesh.num_faces(), faces);
    };
    check(&b, 6);

    // A later patch to it — the library updated on A's side — carries the
    // new geometry too.
    let mesh = a.object(part).unwrap().mesh().unwrap();
    *a.mesh_mut(mesh).unwrap() = kreate_mesh::primitives::uv_sphere(0.01, 8, 6);
    let faces = a.mesh(mesh).unwrap().num_faces();
    let patch = ra.outgoing(&a).expect("the change");
    ra.ack();
    rb.incoming(&mut b, 1, &patch).unwrap().expect("arrives");
    check(&b, faces);
}
