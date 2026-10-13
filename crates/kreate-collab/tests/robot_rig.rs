//! A rigged robot edited by two people: a joint one of them turns stays
//! turned for the other, rather than being put back by the rig.

use kreate_collab::replica::Replica;
use kreate_math::{DTransform, DVec3};
use kreate_scene::robot::{Joint, JointKind, RobotPart};
use kreate_scene::{Document, Object, ObjectData, ObjectId};

/// A base, two turning joints one above the other, a link on each.
fn two_axis_robot(doc: &mut Document) -> ObjectId {
    let link = |doc: &mut Document, name: &str, parent: Option<ObjectId>| {
        let mut o = Object::new(name, ObjectData::Empty);
        o.robot = Some(RobotPart::Link { inertial: None });
        let id = doc.add_object(o);
        doc.set_parent(id, parent).unwrap();
        id
    };
    let joint = |doc: &mut Document, name: &str, parent: ObjectId| {
        let rest = DTransform {
            translation: DVec3::Z,
            ..DTransform::IDENTITY
        };
        let mut o = Object::new(name, ObjectData::Empty);
        o.robot = Some(RobotPart::Joint(Joint {
            kind: JointKind::Revolute,
            rest,
            lower: -2.0,
            upper: 2.0,
            ..Joint::default()
        }));
        o.transform = rest;
        let id = doc.add_object(o);
        doc.set_parent(id, Some(parent)).unwrap();
        id
    };
    let base = link(doc, "base", None);
    let j1 = joint(doc, "shoulder", base);
    let l1 = link(doc, "upper", Some(j1));
    let j2 = joint(doc, "elbow", l1);
    link(doc, "lower", Some(j2));
    base
}

fn named(doc: &Document, name: &str) -> ObjectId {
    doc.objects().find(|(_, o)| o.name == name).unwrap().0
}

/// The joint's position, as its object stands and as its bone says.
fn reading(doc: &Document, name: &str) -> (f64, f64) {
    let id = named(doc, name);
    let o = doc.object(id).unwrap();
    let Some(RobotPart::Joint(j)) = &o.robot else {
        panic!("{name} is not a joint")
    };
    let rig = doc.robot_rig(id).expect("rigged");
    let pose = doc.object(rig).unwrap().pose.clone();
    let bone = doc
        .rig_readings(rig, &pose)
        .into_iter()
        .find(|r| r.joint == id)
        .expect("a reading")
        .position;
    (j.position_of(&o.transform), bone)
}

#[test]
fn a_joint_turned_by_one_participant_stays_turned_for_the_other() {
    let mut a = Document::new();
    let base = two_axis_robot(&mut a);
    a.rig_robot(base).expect("rigs");
    a.sync_robot_rigs();
    let mut ra = Replica::new(&mut a, 1);
    let seed = ra.seed(&a).unwrap();
    ra.ack();
    let mut b = Document::new();
    let mut rb = Replica::new(&mut b, 2);
    rb.adopt_snapshot(&mut b, &seed).unwrap();

    // B turns the shoulder by its object alone: the patch carries the joint
    // and not the rig.
    let shoulder = named(&b, "shoulder");
    let turned = match &b.object(shoulder).unwrap().robot {
        Some(RobotPart::Joint(j)) => j.place(0.7),
        _ => unreachable!(),
    };
    b.object_mut(shoulder).unwrap().transform = turned;
    let patch = rb.outgoing(&b).expect("B's edit");
    rb.ack();

    // A takes it in, as the session's turn does, then runs what a frame
    // runs, ten times over.
    let changes = ra
        .incoming(&mut a, 2, &patch)
        .unwrap()
        .expect("something changed");
    if a.hold_remote_joints(&changes) > 0 {
        ra.absorb(&a);
    }
    for _ in 0..10 {
        a.drive();
        a.sync_robot_rigs();
        if let Some(back) = ra.outgoing(&a) {
            let (_, sent) = kreate_io::sync::decode(&back).unwrap();
            panic!("A sent something back: {sent:?}");
        }
    }
    assert_eq!(ra.waiting(), 0, "A sends nothing back");
    let (joint, bone) = reading(&a, "shoulder");
    assert!((joint - 0.7).abs() < 1e-9, "the joint went back: {joint}");
    assert!((bone - 0.7).abs() < 1e-9, "the bone did not follow: {bone}");
    // The other joint is where it was.
    let (joint, bone) = reading(&a, "elbow");
    assert!(joint.abs() < 1e-9 && bone.abs() < 1e-9, "{joint} {bone}");
}
