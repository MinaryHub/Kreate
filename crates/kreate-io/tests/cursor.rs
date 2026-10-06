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
    let set = doc.add_collection(Collection {
        name: "Set".into(),
        hide: true,
        ..Default::default()
    });
    doc.move_to_collection(a, Some(set));
    let back = round_trip(&doc);
    let id = back
        .find_collection("Set")
        .expect("the collection came back");
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
        ModifierInstance::new(ModifierKind::Mirror(MirrorParams {
            x: false,
            y: true,
            z: true,
            merge_threshold: 0.01,
        })),
        ModifierInstance::new(ModifierKind::Subdivide(SubdivideParams {
            levels: 2,
            smooth: false,
        })),
        ModifierInstance::new(ModifierKind::Array(ArrayParams {
            count: 3,
            relative: kreate_math::Vec3::Y,
            constant: kreate_math::Vec3::new(0.5, 0.0, 0.0),
        })),
        ModifierInstance::new(ModifierKind::Solidify(SolidifyParams {
            thickness: 0.3,
            offset: 0.5,
        })),
        ModifierInstance {
            kind: ModifierKind::Bevel(BevelParams {
                width: 0.05,
                angle: 0.4,
                segments: 3,
            }),
            enabled: false,
        },
        ModifierInstance::new(ModifierKind::Boolean(BooleanModifier {
            operand,
            op: BooleanOp::Intersect,
        })),
    ];
    doc.object_mut(target).unwrap().modifiers = stack.clone();
    let back = round_trip(&doc);
    let t = back
        .objects()
        .find(|(_, o)| o.name == "Target")
        .map(|(id, _)| id)
        .unwrap();
    assert_eq!(back.object(t).unwrap().modifiers, stack);
    let ModifierKind::Boolean(b) = back.object(t).unwrap().modifiers[5].kind else {
        unreachable!()
    };
    let found = back
        .object_by_uid(b.operand)
        .and_then(|id| back.object(id))
        .map(|o| o.name.clone());
    assert_eq!(found.as_deref(), Some("Cutter"));
}

#[test]
fn ik_is_saved_with_the_pose() {
    use kreate_scene::armature::{Armature, IkConstraint};
    use kreate_scene::document::{Object, ObjectData};
    let mut doc = Document::new();
    let arm = doc.add_armature(Armature::new("Arm"));
    let rig = doc.add_object(Object::new("Rig", ObjectData::Armature(arm)));
    let ik = IkConstraint {
        bone: "Bone".into(),
        target: "Goal".into(),
        chain: 3,
        enabled: false,
        pole: "Knee".into(),
    };
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
    o.shape_keys = vec![ShapeKey {
        name: "Tall".into(),
        value: 0.25,
    }];
    o.active_shape = Some("Tall".into());
    let back = round_trip(&doc);
    let (_, o) = back.objects().find(|(_, o)| o.name == "Cube").unwrap();
    assert_eq!(
        o.shape_keys,
        vec![ShapeKey {
            name: "Tall".into(),
            value: 0.25
        }]
    );
    assert_eq!(o.active_shape.as_deref(), Some("Tall"));
    assert!(back
        .mesh(o.mesh().unwrap())
        .unwrap()
        .attributes()
        .contains("shape:Tall"));
}

#[test]
fn bone_constraints_are_saved_with_the_pose() {
    use kreate_scene::armature::{Armature, BoneConstraint, ConstraintKind};
    use kreate_scene::document::{Object, ObjectData};
    let mut doc = Document::new();
    let arm = doc.add_armature(Armature::new("Arm"));
    let rig = doc.add_object(Object::new("Rig", ObjectData::Armature(arm)));
    // Nothing at its default, so a field read back from the wrong place
    // cannot pass by landing on its fallback.
    let constraints = vec![
        BoneConstraint {
            bone: "Bone".into(),
            kind: ConstraintKind::DampedTrack,
            target: "Goal".into(),
            influence: 0.35,
            enabled: false,
        },
        BoneConstraint {
            bone: "Hand".into(),
            kind: ConstraintKind::LimitRotation {
                min: DVec3::new(-0.5, -0.25, -0.125),
                max: DVec3::new(0.75, 0.375, 1.5),
            },
            target: String::new(),
            influence: 0.8,
            enabled: true,
        },
    ];
    doc.object_mut(rig).unwrap().pose.constraints = constraints.clone();
    let back = round_trip(&doc);
    let (_, o) = back.objects().find(|(_, o)| o.name == "Rig").unwrap();
    assert_eq!(o.pose.constraints, constraints);
}

#[test]
fn preserve_volume_is_saved_with_the_object() {
    use kreate_scene::document::{Object, ObjectData};
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let id = doc.add_object(Object::new("Body", ObjectData::Mesh(mesh)));
    doc.object_mut(id).unwrap().preserve_volume = true;
    let back = round_trip(&doc);
    let (_, o) = back.objects().find(|(_, o)| o.name == "Body").unwrap();
    assert!(o.preserve_volume);
}

#[test]
fn nla_strips_are_saved_with_their_object() {
    use kreate_scene::animation::{Action, Interpolation};
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::nla::{NlaExtend, NlaStrip};
    let mut doc = Document::new();
    let mut walk = Action::new("Walk");
    walk.curve_mut("location", 0).insert(1.0, 2.0, Interpolation::Linear);
    let walk = doc.add_action(walk);
    let id = doc.add_object(Object::new("Mover", ObjectData::Empty));
    // Nothing at its default.
    let strip = NlaStrip {
        name: "Walk".into(),
        action: walk,
        start: 12.0,
        action_start: 3.0,
        action_end: 27.0,
        scale: 1.5,
        repeat: 2.25,
        influence: 0.75,
        extend: NlaExtend::Nothing,
        mute: true,
    };
    doc.object_mut(id).unwrap().nla.push(strip.clone());
    let back = round_trip(&doc);
    let (_, o) = back.objects().find(|(_, o)| o.name == "Mover").unwrap();
    assert_eq!(o.nla.len(), 1);
    let got = &o.nla[0];
    assert_eq!(back.action(got.action).unwrap().name, "Walk");
    assert_eq!(
        NlaStrip { action: walk, ..got.clone() },
        strip
    );
}

#[test]
fn geometry_nodes_are_saved_with_their_object() {
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::geonodes::{GeoGraph, GeoNode, GeoNodeKind};
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    let id = doc.add_object(Object::new("Body", ObjectData::Mesh(mesh)));
    let mut graph = GeoGraph::passthrough();
    graph.nodes.push(GeoNode {
        name: "Spin".into(),
        kind: GeoNodeKind::Transform {
            translate: [1.0, 2.0, 3.0],
            rotate: [10.0, 20.0, 30.0],
            scale: [1.5, 1.0, 0.5],
        },
        position: [12.0, -40.0],
    });
    graph.link(0, 2, 0).unwrap();
    graph.link(2, 1, 0).unwrap();
    doc.object_mut(id).unwrap().geometry_nodes = Some(graph.clone());
    let back = round_trip(&doc);
    let (_, o) = back.objects().find(|(_, o)| o.name == "Body").unwrap();
    assert_eq!(o.geometry_nodes.as_ref(), Some(&graph));
}
