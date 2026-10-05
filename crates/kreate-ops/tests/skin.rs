//! A mesh bound to an armature, through the operators: it follows the bones.

use kreate_math::{DQuat, DVec3};
use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::armature::{Armature, Bone};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::modifier::{EvalCtx, ModifierCache};
use kreate_scene::Document;

fn wired() -> (OperatorRegistry, History) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (registry, History::new())
}

/// A column two units tall at (3, 0, 0), and an arm of two bones standing
/// inside it; the column selected, the rig active.
fn scene() -> (Document, ObjectId, ObjectId) {
    let mut doc = Document::new();
    let mut mesh = kreate_mesh::primitives::cube(0.2);
    for p in mesh.positions_mut() {
        p.z = (p.z + 0.1) * 10.0;
    }
    let mesh = doc.add_mesh(mesh);
    let mut column = Object::new("Column", ObjectData::Mesh(mesh));
    column.transform.translation = DVec3::new(3.0, 0.0, 0.0);
    let column = doc.add_object(column);
    let mut a = Armature::new("Arm");
    a.bones[0].name = "Upper".into();
    a.add_bone(Bone {
        name: "Lower".into(),
        parent: Some(0),
        head: DVec3::Z,
        tail: DVec3::new(0.0, 0.0, 2.0),
        connected: true,
        ..Default::default()
    });
    let arm = doc.add_armature(a);
    let mut rig = Object::new("Rig", ObjectData::Armature(arm));
    rig.transform.translation = DVec3::new(3.0, 0.0, 0.0);
    let rig = doc.add_object(rig);
    doc.selection.add(column);
    doc.selection.add(rig);
    (doc, column, rig)
}

fn bind(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    weights: u32,
) -> bool {
    let mut p = PropertyMap::new();
    p.set("weights", PropertyValue::Enum(weights));
    let depth = history.entries().len();
    history
        .dispatch(
            registry,
            doc,
            &Context::default(),
            "object.bind_armature",
            &p,
        )
        .expect("ran");
    history.entries().len() > depth
}

fn top(doc: &Document, cache: &mut ModifierCache, column: ObjectId) -> f32 {
    let mesh = cache
        .evaluated(doc, column, &EvalCtx::still())
        .expect("a mesh");
    mesh.positions()
        .iter()
        .map(|p| p.z)
        .fold(f32::MIN, f32::max)
}

#[test]
fn a_bound_column_bends_with_the_lower_bone() {
    let (registry, mut history) = wired();
    let (mut doc, column, rig) = scene();
    assert!(bind(&registry, &mut history, &mut doc, 0));
    let o = doc.object(column).unwrap();
    assert_eq!(o.parent, Some(rig));
    assert_eq!(o.skin, doc.object_uid(rig));
    // Parented without moving.
    assert!((doc.world_transform(column).translation - DVec3::new(3.0, 0.0, 0.0)).length() < 1e-9);
    let mesh = doc.mesh(o.mesh().unwrap()).unwrap();
    assert!(
        mesh.attributes().contains("weight:Upper") && mesh.attributes().contains("weight:Lower")
    );

    let mut cache = ModifierCache::new();
    assert!(
        (top(&doc, &mut cache, column) - 2.0).abs() < 1e-4,
        "at rest the column moved"
    );
    doc.object_mut(rig)
        .unwrap()
        .pose
        .entry("Lower")
        .transform
        .rotation = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    let bent = top(&doc, &mut cache, column);
    assert!(
        bent < 1.3,
        "the top of the column did not bend over: {bent}"
    );

    // The same pose asked again is the cache's, not a second skinning.
    let a = cache.evaluated(&doc, column, &EvalCtx::still()).unwrap();
    let b = cache.evaluated(&doc, column, &EvalCtx::still()).unwrap();
    assert!(std::sync::Arc::ptr_eq(&a, &b));

    // Unbound, it stands straight again whatever the pose; undone, it bends.
    history
        .dispatch(
            &registry,
            &mut doc,
            &Context::default(),
            "object.unbind_armature",
            &PropertyMap::new(),
        )
        .expect("ran");
    assert!((top(&doc, &mut cache, column) - 2.0).abs() < 1e-4);
    assert!(history.undo(&mut doc));
    doc.object_mut(rig)
        .unwrap()
        .pose
        .entry("Lower")
        .transform
        .rotation = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    assert!(top(&doc, &mut cache, column) < 1.3);
}

#[test]
fn empty_groups_bind_without_moving_anything() {
    let (registry, mut history) = wired();
    let (mut doc, column, rig) = scene();
    assert!(bind(&registry, &mut history, &mut doc, 1));
    doc.object_mut(rig)
        .unwrap()
        .pose
        .entry("Upper")
        .transform
        .rotation = DQuat::from_rotation_x(1.0);
    let mut cache = ModifierCache::new();
    assert!(
        (top(&doc, &mut cache, column) - 2.0).abs() < 1e-4,
        "an empty group moved the mesh"
    );
}

#[test]
fn renaming_a_bone_renames_the_group_that_follows_it() {
    let (registry, mut history) = wired();
    let (mut doc, column, rig) = scene();
    bind(&registry, &mut history, &mut doc, 0);
    let (arm, _) = doc.armature_of(rig).unwrap();
    assert!(doc.rename_bone(arm, "Lower", "Forearm"));
    let mesh = doc
        .mesh(doc.object(column).unwrap().mesh().unwrap())
        .unwrap();
    assert!(mesh.attributes().contains("weight:Forearm"));
    assert!(!mesh.attributes().contains("weight:Lower"));
    // And it still bends.
    doc.object_mut(rig)
        .unwrap()
        .pose
        .entry("Forearm")
        .transform
        .rotation = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    let mut cache = ModifierCache::new();
    assert!(top(&doc, &mut cache, column) < 1.3);
}

#[test]
fn binding_needs_an_armature_and_a_mesh() {
    let (registry, mut history) = wired();
    let (mut doc, column, _) = scene();
    doc.selection.clear();
    doc.selection.add(column);
    let mut p = PropertyMap::new();
    p.set("weights", PropertyValue::Enum(0));
    assert!(history
        .dispatch(
            &registry,
            &mut doc,
            &Context::default(),
            "object.bind_armature",
            &p
        )
        .is_err());
}
