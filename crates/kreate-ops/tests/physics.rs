//! Physics through the operators: settings on the selection, a bake that is
//! one undo step, playback, and Bake to Keyframes.

use kreate_math::DVec3;
use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::physics::{BodyKind, CollisionShape, ParticleKind, SoftKind};
use kreate_scene::{Document, ObjectId};

fn wired() -> (OperatorRegistry, History, Document) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut doc = Document::new();
    doc.physics.frame_end = 48;
    (registry, History::new(), doc)
}

fn add(doc: &mut Document, name: &str, mesh: kreate_mesh::MeshData, z: f64) -> ObjectId {
    let m = doc.add_mesh(mesh);
    let mut o = Object::new(name, ObjectData::Mesh(m));
    o.transform.translation = DVec3::new(0.0, 0.0, z);
    doc.add_object(o)
}

fn select(doc: &mut Document, id: ObjectId) {
    doc.selection.objects.clear();
    doc.selection.add(id);
    doc.selection.active = Some(id);
}

fn run(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    op: &str,
    params: &[(&'static str, PropertyValue)],
) -> bool {
    let mut p = PropertyMap::new();
    for (k, v) in params {
        p.set(k, v.clone());
    }
    let depth = history.entries().len();
    history
        .dispatch(registry, doc, &Context::default(), op, &p)
        .expect("ran");
    history.entries().len() > depth
}

/// A floor and a cube above it, both rigid bodies.
fn falling_cube(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
) -> ObjectId {
    let floor = add(doc, "Floor", kreate_mesh::primitives::plane(10.0), 0.0);
    let cube = add(doc, "Cube", kreate_mesh::primitives::cube(1.0), 3.0);
    select(doc, floor);
    assert!(run(
        registry,
        history,
        doc,
        "physics.rigid_body_set",
        &[
            ("enabled", PropertyValue::Bool(true)),
            ("kind", PropertyValue::Enum(1)),
            ("shape", PropertyValue::Enum(3)),
        ]
    ));
    select(doc, cube);
    assert!(run(
        registry,
        history,
        doc,
        "physics.rigid_body_set",
        &[("enabled", PropertyValue::Bool(true))]
    ));
    cube
}

#[test]
fn settings_land_on_the_selection_and_switching_off_removes_them() {
    let (registry, mut history, mut doc) = wired();
    let cube = falling_cube(&registry, &mut history, &mut doc);
    let floor = doc.objects().find(|(_, o)| o.name == "Floor").unwrap().0;
    let body = doc.object(floor).unwrap().physics.rigid.unwrap();
    assert_eq!(
        (body.kind, body.shape),
        (BodyKind::Passive, CollisionShape::Mesh)
    );
    assert_eq!(
        doc.object(cube).unwrap().physics.rigid.unwrap().kind,
        BodyKind::Active
    );
    assert!(
        !run(
            &registry,
            &mut history,
            &mut doc,
            "physics.rigid_body_set",
            &[("enabled", PropertyValue::Bool(true))]
        ),
        "the same again is no step"
    );
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.rigid_body_set",
        &[("enabled", PropertyValue::Bool(false))]
    ));
    assert!(doc.object(cube).unwrap().physics.rigid.is_none());
}

#[test]
fn a_bake_plays_back_is_one_undo_step_and_free_puts_things_back() {
    let (registry, mut history, mut doc) = wired();
    let cube = falling_cube(&registry, &mut history, &mut doc);
    assert!(run(&registry, &mut history, &mut doc, "physics.bake", &[]));
    doc.animate(48.0);
    let z = doc.object(cube).unwrap().transform.translation.z;
    assert!((z - 0.5).abs() < 0.05, "landed: {z}");

    assert!(history.undo(&mut doc));
    assert!(doc.physics_cache.is_empty(), "the bake is undone");

    // After an undo the new step takes the redo's place, so the history is
    // no longer: ask the document instead.
    run(&registry, &mut history, &mut doc, "physics.bake", &[]);
    assert!(!doc.physics_cache.is_empty());
    doc.animate(30.0);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.free_bake",
        &[]
    ));
    assert!(doc.physics_cache.is_empty());
    assert_eq!(
        doc.object(cube).unwrap().transform.translation.z,
        3.0,
        "back where it began"
    );
}

#[test]
fn bake_to_keyframes_leaves_an_action_and_no_rigid_body() {
    let (registry, mut history, mut doc) = wired();
    let cube = falling_cube(&registry, &mut history, &mut doc);
    assert!(run(&registry, &mut history, &mut doc, "physics.bake", &[]));
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.bake_to_keys",
        &[]
    ));
    let o = doc.object(cube).unwrap();
    assert!(o.physics.rigid.is_none());
    assert!(doc.physics_cache.rigid.is_empty());
    let action = doc.action(o.action.expect("an action")).unwrap();
    assert_eq!(action.key_frames().len(), 48);
    doc.animate(48.0);
    let z = doc.object(cube).unwrap().transform.translation.z;
    assert!((z - 0.5).abs() < 0.05, "the keys play the fall: {z}");
}

#[test]
fn cloth_kind_switches_take_that_kinds_settings() {
    let (registry, mut history, mut doc) = wired();
    let sheet = add(
        &mut doc,
        "Sheet",
        kreate_mesh::primitives::grid(2.0, 4, 4),
        1.0,
    );
    select(&mut doc, sheet);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.soft_set",
        &[("kind", PropertyValue::Enum(2))]
    ));
    let s = doc.object(sheet).unwrap().physics.soft.clone().unwrap();
    assert_eq!(s.kind, SoftKind::Soft);
    assert!(s.shape_stiffness > 0.0, "a soft body's own defaults");
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.soft_set",
        &[
            ("kind", PropertyValue::Enum(2)),
            ("pin_group", PropertyValue::Text("Top".into())),
        ]
    ));
    assert_eq!(
        doc.object(sheet)
            .unwrap()
            .physics
            .soft
            .clone()
            .unwrap()
            .pin_group
            .as_deref(),
        Some("Top")
    );
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.soft_set",
        &[("kind", PropertyValue::Enum(0))]
    ));
    assert!(doc.object(sheet).unwrap().physics.soft.is_none());
}

#[test]
fn particle_systems_are_added_named_set_and_removed() {
    let (registry, mut history, mut doc) = wired();
    let e = add(
        &mut doc,
        "Emitter",
        kreate_mesh::primitives::plane(1.0),
        0.0,
    );
    let _bead = add(
        &mut doc,
        "Bead",
        kreate_mesh::primitives::ico_sphere(1.0, 1),
        5.0,
    );
    select(&mut doc, e);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.particles_add",
        &[]
    ));
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.particles_add",
        &[("kind", PropertyValue::Enum(1))]
    ));
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.particles_add",
        &[]
    ));
    let names: Vec<String> = doc
        .object(e)
        .unwrap()
        .physics
        .particles
        .iter()
        .map(|s| s.name.clone())
        .collect();
    assert_eq!(names, ["Particles", "Hair", "Particles.001"]);
    assert_eq!(
        doc.object(e).unwrap().physics.particles[1].kind,
        ParticleKind::Hair
    );

    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.particles_set",
        &[
            ("index", PropertyValue::Int(0)),
            ("count", PropertyValue::Int(25)),
            ("instance", PropertyValue::Text("Bead".into())),
        ]
    ));
    let s = &doc.object(e).unwrap().physics.particles[0];
    assert_eq!(s.count, 25);
    assert!(s.instance.is_some());
    let mut p = PropertyMap::new();
    p.set("index", PropertyValue::Int(0));
    p.set("instance", PropertyValue::Text("Nobody".into()));
    assert!(
        history
            .dispatch(
                &registry,
                &mut doc,
                &Context::default(),
                "physics.particles_set",
                &p
            )
            .is_err(),
        "an instance that is not there is said"
    );

    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.particles_remove",
        &[("index", PropertyValue::Int(1))]
    ));
    assert_eq!(doc.object(e).unwrap().physics.particles.len(), 2);
}

#[test]
fn a_bake_with_nothing_moving_says_so() {
    let (registry, mut history, mut doc) = wired();
    let floor = add(&mut doc, "Floor", kreate_mesh::primitives::plane(4.0), 0.0);
    select(&mut doc, floor);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.collision_set",
        &[("enabled", PropertyValue::Bool(true))]
    ));
    let p = PropertyMap::new();
    let err = history
        .dispatch(&registry, &mut doc, &Context::default(), "physics.bake", &p)
        .unwrap_err();
    assert!(err.to_string().contains("nothing to bake"), "{err}");
}

#[test]
fn the_world_takes_its_settings() {
    let (registry, mut history, mut doc) = wired();
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.world_set",
        &[
            ("gravity", PropertyValue::Vec3([0.0, 0.0, -1.62])),
            ("frame_end", PropertyValue::Int(100)),
        ]
    ));
    assert_eq!(doc.physics.gravity.z, -1.62);
    assert_eq!(doc.physics.frame_end, 100);
}

#[test]
fn a_liquid_is_set_up_baked_and_undone_by_operators() {
    let (registry, mut history, mut doc) = wired();
    doc.physics.frame_end = 12;
    let tank = add(&mut doc, "Tank", kreate_mesh::primitives::cube(2.0), 1.0);
    let water = add(&mut doc, "Water", kreate_mesh::primitives::cube(0.8), 1.4);
    select(&mut doc, tank);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.fluid_set",
        &[
            ("role", PropertyValue::Enum(1)),
            ("resolution", PropertyValue::Int(12)),
        ]
    ));
    select(&mut doc, water);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.fluid_set",
        &[("role", PropertyValue::Enum(2))]
    ));
    assert!(matches!(
        doc.object(tank).unwrap().physics.fluid,
        Some(kreate_scene::physics::Fluid::Domain(d)) if d.resolution == 12
    ));
    // Given water, having no material of its own; the water block keeps
    // none, being a flow.
    let slots = doc.mesh_slots(doc.object(tank).unwrap().mesh().unwrap());
    let given = doc.material(slots[0].expect("a material")).unwrap();
    assert_eq!(given.name, "Water");
    assert!(given.transmission > 0.9 && (given.ior - 1.333).abs() < 1e-3);
    assert!(doc
        .mesh_slots(doc.object(water).unwrap().mesh().unwrap())
        .iter()
        .all(|s| s.is_none()));
    assert!(run(&registry, &mut history, &mut doc, "physics.bake", &[]));
    assert_eq!(doc.physics_cache.fluid.len(), 1);
    assert_eq!(doc.physics_cache.fluid[0].frames.len(), 12);
    assert!(history.undo(&mut doc));
    assert!(doc.physics_cache.is_empty(), "the bake is undone");
    // And the water is nothing again. (After an undo the new step takes the
    // redo's place, so ask the document rather than the history.)
    select(&mut doc, water);
    run(
        &registry,
        &mut history,
        &mut doc,
        "physics.fluid_set",
        &[("role", PropertyValue::Enum(0))],
    );
    assert!(doc.object(water).unwrap().physics.fluid.is_none());
}

#[test]
fn smoke_and_fire_are_set_up_and_baked_by_operators() {
    let (registry, mut history, mut doc) = wired();
    doc.physics.frame_end = 8;
    let room = add(&mut doc, "Room", kreate_mesh::primitives::cube(2.0), 1.0);
    let torch = add(&mut doc, "Torch", kreate_mesh::primitives::cube(0.4), 0.3);
    select(&mut doc, room);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.fluid_set",
        &[
            ("role", PropertyValue::Enum(1)),
            ("domain", PropertyValue::Enum(1)),
            ("resolution", PropertyValue::Int(12)),
        ]
    ));
    select(&mut doc, torch);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.fluid_set",
        &[
            ("role", PropertyValue::Enum(2)),
            ("substance", PropertyValue::Enum(3)),
            ("flow", PropertyValue::Enum(1)),
        ]
    ));
    assert!(run(&registry, &mut history, &mut doc, "physics.bake", &[]));
    assert!(doc.physics_cache.fluid.is_empty());
    assert_eq!(doc.physics_cache.gas.len(), 1);
    assert_eq!(doc.physics_cache.gas[0].frames.len(), 8);
}

#[test]
fn one_simulation_bakes_and_frees_on_its_own() {
    let (registry, mut history, mut doc) = wired();
    let cube = falling_cube(&registry, &mut history, &mut doc);
    let emitter = add(
        &mut doc,
        "Emitter",
        kreate_mesh::primitives::plane(1.0),
        6.0,
    );
    select(&mut doc, emitter);
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.particles_add",
        &[]
    ));
    // Everything, then the particles freed: the falling cube's bake stays.
    assert!(run(&registry, &mut history, &mut doc, "physics.bake", &[]));
    let rigid = doc.physics_cache.rigid.clone();
    assert!(!rigid.is_empty() && !doc.physics_cache.particles.is_empty());
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.free_bake",
        &[("only", PropertyValue::Enum(3))]
    ));
    assert!(doc.physics_cache.particles.is_empty());
    assert_eq!(
        doc.physics_cache.rigid, rigid,
        "the rigid bodies' bake is kept"
    );
    doc.animate(48.0);
    let z = doc.object(cube).unwrap().transform.translation.z;
    assert!((z - 0.5).abs() < 0.05, "still landed: {z}");
    // The particles alone, baked again: the rigid bake untouched.
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.bake",
        &[("only", PropertyValue::Enum(3))]
    ));
    assert!(!doc.physics_cache.particles.is_empty());
    assert_eq!(doc.physics_cache.rigid, rigid);
    // Rigid bodies alone, from nothing baked: the particles' bake is kept.
    let particles = doc.physics_cache.particles.len();
    let first = doc.physics_cache.particles[0].frames.len();
    assert!(run(
        &registry,
        &mut history,
        &mut doc,
        "physics.bake",
        &[("only", PropertyValue::Enum(1))]
    ));
    assert_eq!(doc.physics_cache.particles.len(), particles);
    assert_eq!(doc.physics_cache.particles[0].frames.len(), first);
    assert!(!doc.physics_cache.rigid.is_empty());
}
