//! Align Faces as an assembly mate: placed, and held.

use kreate_math::DVec3;
use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{Document, ObjectId};

fn part(doc: &mut Document, name: &str, at: DVec3) -> ObjectId {
    let m = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let mut o = Object::new(name, ObjectData::Mesh(m));
    o.transform.translation = at;
    doc.add_object(o)
}

#[test]
fn a_mated_part_sits_on_the_other_and_moves_with_it() {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    let base = part(&mut doc, "Base", DVec3::ZERO);
    let lid = part(&mut doc, "Lid", DVec3::new(5.0, 0.0, 0.0));
    let uid = |doc: &Document, id| doc.object_uid(id).unwrap().raw().to_string();
    let mut p = PropertyMap::new();
    p.set("uid", PropertyValue::Text(uid(&doc, lid)));
    // The lid's bottom face onto the base's top.
    p.set("from", PropertyValue::Vec3([5.0, 0.0, -0.5]));
    p.set("from_normal", PropertyValue::Vec3([0.0, 0.0, -1.0]));
    p.set("to", PropertyValue::Vec3([0.0, 0.0, 0.5]));
    p.set("to_normal", PropertyValue::Vec3([0.0, 0.0, 1.0]));
    p.set("target", PropertyValue::Text(uid(&doc, base)));
    history
        .dispatch(&registry, &mut doc, &Context::default(), "object.mate", &p)
        .unwrap();
    let at = doc.world_transform(lid).translation;
    assert!(
        at.distance(DVec3::new(0.0, 0.0, 1.0)) < 1e-9,
        "on top: {at:?}"
    );
    assert_eq!(doc.object(lid).unwrap().parent, Some(base), "attached");
    // The base moves; the lid comes with it.
    doc.object_mut(base).unwrap().transform.translation = DVec3::new(2.0, 3.0, 0.0);
    let at = doc.world_transform(lid).translation;
    assert!(
        at.distance(DVec3::new(2.0, 3.0, 1.0)) < 1e-9,
        "held: {at:?}"
    );
}
