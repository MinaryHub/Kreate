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
        random_size: 0.4,
        random_rotation: 0.75,
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
fn a_baked_scene_opens_baked() {
    use kreate_scene::physics::{BodyKind, CollisionShape};
    let mut doc = Document::new();
    doc.physics.frame_end = 30;
    let floor = doc.add_mesh(kreate_mesh::primitives::plane(10.0));
    let mut o = Object::new("Floor", ObjectData::Mesh(floor));
    o.physics.rigid = Some(RigidBody {
        kind: BodyKind::Passive,
        shape: CollisionShape::Mesh,
        ..RigidBody::default()
    });
    doc.add_object(o);
    let cube = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let mut o = Object::new("Cube", ObjectData::Mesh(cube));
    o.transform.translation.z = 3.0;
    o.physics.rigid = Some(RigidBody::default());
    doc.add_object(o);
    let (cache, _) = kreate_scene::physics::bake::bake(&doc);
    doc.physics_cache = std::sync::Arc::new(cache);

    let mut back = round_trip(&doc);
    assert_eq!(back.physics_cache.rigid, doc.physics_cache.rigid);
    back.animate(30.0);
    let (_, cube) = back.objects().find(|(_, o)| o.name == "Cube").unwrap();
    assert!(
        (cube.transform.translation.z - 0.5).abs() < 0.05,
        "plays from the file"
    );
}

#[test]
fn a_scene_without_physics_writes_nothing_of_it() {
    let bytes = container::write(&Document::new(), &Manifest::default(), &[]).expect("writes");
    assert!(!String::from_utf8_lossy(&bytes).contains("\"physics\""));
}
