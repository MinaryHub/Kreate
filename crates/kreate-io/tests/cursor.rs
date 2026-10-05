//! The 3D cursor through the file and back.

use kreate_io::container::{self, Manifest};
use kreate_math::DVec3;
use kreate_scene::Document;
use std::io::Cursor;

fn round_trip(doc: &Document) -> Document {
    let bytes = container::write(doc, &Manifest::default(), &[]).expect("writes");
    container::read(Cursor::new(bytes)).expect("opens").document
}

#[test]
fn the_cursor_is_saved_where_it_was() {
    let mut doc = Document::new();
    doc.cursor = DVec3::new(1.5, -2.25, 0.125);
    assert_eq!(round_trip(&doc).cursor, doc.cursor);
}

#[test]
fn a_cursor_at_the_origin_writes_nothing() {
    // The default costs no bytes, which is what keeps the golden files still.
    let doc = Document::new();
    let bytes = container::write(&doc, &Manifest::default(), &[]).expect("writes");
    let text = String::from_utf8_lossy(&bytes);
    assert!(!text.contains("\"cursor\""));
    assert_eq!(round_trip(&doc).cursor, DVec3::ZERO);
}

#[test]
fn collections_are_saved_with_what_is_in_them_and_whether_hidden() {
    use kreate_scene::document::{Collection, Object, ObjectData};
    let mut doc = Document::new();
    let a = doc.add_object(Object::new("A", ObjectData::Empty));
    let _b = doc.add_object(Object::new("B", ObjectData::Empty));
    let set = doc.add_collection(Collection { name: "Set".into(), hide: true, ..Default::default() });
    doc.move_to_collection(a, Some(set));
    let back = round_trip(&doc);
    let id = back.find_collection("Set").expect("the collection came back");
    let c = back.collection(id).unwrap();
    assert!(c.hide);
    assert_eq!(c.objects.len(), 1);
    assert_eq!(back.object(c.objects[0]).unwrap().name, "A");
}

#[test]
fn every_modifier_is_saved_and_a_boolean_still_finds_its_cutter() {
    use kreate_mesh::modifiers::*;
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::modifier::{BooleanModifier, ModifierInstance, ModifierKind};
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let cutter = doc.add_object(Object::new("Cutter", ObjectData::Mesh(mesh)));
    let target = doc.add_object(Object::new("Target", ObjectData::Mesh(mesh)));
    let operand = doc.object_uid(cutter).unwrap();
    let stack = vec![
        ModifierInstance::new(ModifierKind::Mirror(MirrorParams { x: false, y: true, z: true, merge_threshold: 0.01 })),
        ModifierInstance::new(ModifierKind::Subdivide(SubdivideParams { levels: 2, smooth: false })),
        ModifierInstance::new(ModifierKind::Array(ArrayParams {
            count: 3,
            relative: kreate_math::Vec3::Y,
            constant: kreate_math::Vec3::new(0.5, 0.0, 0.0),
        })),
        ModifierInstance::new(ModifierKind::Solidify(SolidifyParams { thickness: 0.3, offset: 0.5 })),
        ModifierInstance { kind: ModifierKind::Bevel(BevelParams { width: 0.05, angle: 0.4, segments: 3 }), enabled: false },
        ModifierInstance::new(ModifierKind::Boolean(BooleanModifier { operand, op: BooleanOp::Intersect })),
    ];
    doc.object_mut(target).unwrap().modifiers = stack.clone();
    let back = round_trip(&doc);
    let t = back.objects().find(|(_, o)| o.name == "Target").map(|(id, _)| id).unwrap();
    assert_eq!(back.object(t).unwrap().modifiers, stack);
    let ModifierKind::Boolean(b) = back.object(t).unwrap().modifiers[5].kind else { unreachable!() };
    let found = back.object_by_uid(b.operand).and_then(|id| back.object(id)).map(|o| o.name.clone());
    assert_eq!(found.as_deref(), Some("Cutter"));
}

#[test]
fn ik_is_saved_with_the_pose() {
    use kreate_scene::armature::{Armature, IkConstraint};
    use kreate_scene::document::{Object, ObjectData};
    let mut doc = Document::new();
    let arm = doc.add_armature(Armature::new("Arm"));
    let rig = doc.add_object(Object::new("Rig", ObjectData::Armature(arm)));
    let ik = IkConstraint { bone: "Bone".into(), target: "Goal".into(), chain: 3, enabled: false, pole: "Knee".into() };
    doc.object_mut(rig).unwrap().pose.ik.push(ik.clone());
    let back = round_trip(&doc);
    let (_, o) = back.objects().find(|(_, o)| o.name == "Rig").unwrap();
    assert_eq!(o.pose.ik, vec![ik]);
}

#[test]
fn shape_keys_are_saved_with_their_offsets() {
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::shape::{add_layer, ShapeKey};
    let mut doc = Document::new();
    let mut mesh = kreate_mesh::primitives::cube(2.0);
    add_layer(&mut mesh, "Tall", &[]);
    let mesh = doc.add_mesh(mesh);
    let id = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    let o = doc.object_mut(id).unwrap();
    o.shape_keys = vec![ShapeKey { name: "Tall".into(), value: 0.25 }];
    o.active_shape = Some("Tall".into());
    let back = round_trip(&doc);
    let (_, o) = back.objects().find(|(_, o)| o.name == "Cube").unwrap();
    assert_eq!(o.shape_keys, vec![ShapeKey { name: "Tall".into(), value: 0.25 }]);
    assert_eq!(o.active_shape.as_deref(), Some("Tall"));
    assert!(back.mesh(o.mesh().unwrap()).unwrap().attributes().contains("shape:Tall"));
}
