//! Physics settings through the file and back.

use kreate_io::container::{self, Manifest};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::physics::{ParticleKind, ParticleSettings, RigidBody, SoftBody, SoftKind};
use kreate_scene::Document;
use std::io::Cursor;

fn round_trip(doc: &Document) -> Document {
    let bytes = container::write(doc, &Manifest::default(), &[]).expect("writes");
    container::read(Cursor::new(bytes)).expect("opens").document
}

#[test]
fn an_objects_physics_and_the_world_are_saved() {
    let mut doc = Document::new();
    let bead = doc.add_mesh(kreate_mesh::primitives::ico_sphere(0.1, 1));
    let bead = doc.add_object(Object::new("Bead", ObjectData::Mesh(bead)));
    let bead_uid = doc.object(bead).unwrap().uid;
    let m = doc.add_mesh(kreate_mesh::primitives::grid(2.0, 4, 4));
    let mut o = Object::new("Sheet", ObjectData::Mesh(m));
    o.physics.rigid = Some(RigidBody {
        mass: 4.0,
        ..RigidBody::default()
    });
    o.physics.soft = Some(SoftBody {
        pin_group: Some("Top".into()),
        ..SoftBody::new(SoftKind::Soft)
    });
    o.physics.particles = vec![ParticleSettings {
        instance: Some(bead_uid),
        ..ParticleSettings::new("Spray", ParticleKind::Emitter)
    }];
    doc.add_object(o);
    doc.physics.frame_end = 123;
    doc.physics.gravity.z = -3.0;

    let back = round_trip(&doc);
    let (_, sheet) = back.objects().find(|(_, o)| o.name == "Sheet").unwrap();
    let (_, original) = doc.objects().find(|(_, o)| o.name == "Sheet").unwrap();
    assert_eq!(sheet.physics, original.physics);
    let instance = back
        .object_by_uid(sheet.physics.particles[0].instance.unwrap())
        .unwrap();
    assert_eq!(
        back.object(instance).unwrap().name,
        "Bead",
        "the instance by identity"
    );
    assert_eq!(back.physics, doc.physics);
}

#[test]
fn a_scene_without_physics_writes_nothing_of_it() {
    let bytes = container::write(&Document::new(), &Manifest::default(), &[]).expect("writes");
    assert!(!String::from_utf8_lossy(&bytes).contains("\"physics\""));
}
