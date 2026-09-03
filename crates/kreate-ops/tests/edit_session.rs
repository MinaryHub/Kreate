//! Operators work in the edit session when one is open.
//!
//! Holding the edit structure across operations is the whole point: without it
//! every operator decodes the mesh on the way in and encodes it on the way out,
//! so an extrude on a large mesh spends more time converting than extruding.
//!
//! The claim is awkward to test directly — "it did not convert" leaves no trace
//! in the result — so it is tested by *identity*: an operator that used the
//! session leaves the session holding the geometry it produced, and one that
//! converted for itself leaves the session untouched and stale.

use kreate_ops::operator::{Context, Mode};
use kreate_ops::{EditSession, History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{MeshId, Object, ObjectData};
use kreate_scene::Document;

fn setup() -> (OperatorRegistry, History, Document, MeshId, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
    let object = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    doc.selection.add(object);
    (
        registry,
        History::new(),
        doc,
        mesh,
        Context {
            mode: Mode::EditMesh,
        },
    )
}

fn params(pairs: &[(&'static str, PropertyValue)]) -> PropertyMap {
    let mut map = PropertyMap::new();
    for (name, value) in pairs {
        map.set(name, value.clone());
    }
    map
}

/// Selects everything, so the geometry operators have something to act on.
fn select_all(r: &OperatorRegistry, h: &mut History, d: &mut Document, c: &Context) {
    h.dispatch(r, d, c, "mesh.select", &PropertyMap::new())
        .expect("mesh.select");
}

#[test]
fn an_operator_works_in_the_session_when_one_is_open() {
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    let mut session = EditSession::open(&d, mesh).expect("the mesh is there");

    h.dispatch_in_session(
        &r,
        &mut d,
        Some(&mut session),
        &c,
        "mesh.subdivide",
        &PropertyMap::new(),
    )
    .expect("mesh.subdivide");

    // The session holds what the operator produced, which is what says it worked
    // there rather than on a copy of its own.
    assert_eq!(d.mesh(mesh).unwrap().num_faces(), 24);
    assert_eq!(session.bmesh_mut().face_ids().count(), 24);
    assert!(
        session.matches(&d, mesh),
        "the session should be current with what it just wrote"
    );
}

#[test]
fn the_session_carries_across_several_operators() {
    // The decode happens once on opening, not once per operator. Three
    // subdivisions in a row, and the session is current before each.
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    let mut session = EditSession::open(&d, mesh).expect("the mesh is there");

    for level in 0..3 {
        assert!(
            session.matches(&d, mesh),
            "session went stale before level {level}"
        );
        h.dispatch_in_session(
            &r,
            &mut d,
            Some(&mut session),
            &c,
            "mesh.subdivide",
            &PropertyMap::new(),
        )
        .expect("mesh.subdivide");
    }
    assert_eq!(d.mesh(mesh).unwrap().num_faces(), 6 * 4 * 4 * 4);
}

#[test]
fn an_operator_still_works_without_a_session() {
    // Scripts, tests and anything outside edit mode have none, and must not have
    // to care.
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    h.dispatch(&r, &mut d, &c, "mesh.subdivide", &PropertyMap::new())
        .expect("mesh.subdivide");
    assert_eq!(d.mesh(mesh).unwrap().num_faces(), 24);
}

#[test]
fn a_session_for_another_mesh_is_not_used() {
    // Two objects, and the session belongs to the one that is not being edited.
    // Writing its topology over the target would replace one model with another.
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    let other = d.add_mesh(kreate_mesh::primitives::uv_sphere(1.0, 8, 4));
    let mut session = EditSession::open(&d, other).expect("the other mesh is there");
    let before = d.mesh(other).unwrap().num_faces();

    h.dispatch_in_session(
        &r,
        &mut d,
        Some(&mut session),
        &c,
        "mesh.subdivide",
        &PropertyMap::new(),
    )
    .expect("mesh.subdivide");

    assert_eq!(
        d.mesh(mesh).unwrap().num_faces(),
        24,
        "the target was edited"
    );
    assert_eq!(
        d.mesh(other).unwrap().num_faces(),
        before,
        "the other mesh must not have been touched"
    );
}

#[test]
fn an_operator_sees_a_selection_made_outside_the_session() {
    // A click writes the document mesh directly — it does not go through the
    // session — and it deliberately does *not* invalidate it, because treating
    // every click as staleness costs a full decode on the next operation.
    //
    // So the structure has to be told. An extrude that acted on the selection
    // the session was holding when it opened would do nothing at all.
    let (r, mut h, mut d, mesh, c) = setup();
    let mut session = EditSession::open(&d, mesh).expect("the mesh is there");

    h.dispatch(
        &r,
        &mut d,
        &c,
        "mesh.select_pick",
        &params(&[
            ("element", PropertyValue::Int(0)),
            ("mode", PropertyValue::Enum(2)),
        ]),
    )
    .expect("mesh.select_pick");
    assert!(
        session.matches(&d, mesh),
        "a click is not a geometry change and must not invalidate the session"
    );

    h.dispatch_in_session(
        &r,
        &mut d,
        Some(&mut session),
        &c,
        "mesh.extrude_region",
        &params(&[("offset", PropertyValue::Float(1.0))]),
    )
    .expect("mesh.extrude_region");

    assert!(
        d.mesh(mesh).unwrap().num_faces() > 6,
        "the extrude did nothing, so the session did not see the selection"
    );
}

#[test]
fn a_session_stale_from_a_geometry_change_reopens_rather_than_overwriting() {
    // The dangerous case, and the one the staleness check exists for: something
    // changed the geometry by another route. Writing back what the session holds
    // would silently revert it.
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    let mut session = EditSession::open(&d, mesh).expect("the mesh is there");

    // A move, applied straight to the document — a transform, or an undo.
    for p in d.mesh_mut(mesh).unwrap().positions_mut() {
        p.z += 5.0;
    }
    assert!(!session.matches(&d, mesh), "a geometry change is staleness");

    h.dispatch_in_session(
        &r,
        &mut d,
        Some(&mut session),
        &c,
        "mesh.subdivide",
        &PropertyMap::new(),
    )
    .expect("mesh.subdivide");

    // The move survived: the session reopened and subdivided the moved mesh
    // rather than writing its own copy back over it.
    let lowest = d
        .mesh(mesh)
        .unwrap()
        .positions()
        .iter()
        .map(|p| p.z)
        .fold(f32::INFINITY, f32::min);
    assert!(
        lowest > 3.0,
        "the session wrote stale geometry over the move: lowest z is {lowest}"
    );
}

#[test]
fn undo_still_restores_through_a_session() {
    // The session is a staging area, not a second source of truth. Undo restores
    // the document, and the next operator has to see that rather than the
    // topology the session was holding.
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    let mut session = EditSession::open(&d, mesh).expect("the mesh is there");

    h.dispatch_in_session(
        &r,
        &mut d,
        Some(&mut session),
        &c,
        "mesh.subdivide",
        &PropertyMap::new(),
    )
    .expect("mesh.subdivide");
    assert_eq!(d.mesh(mesh).unwrap().num_faces(), 24);

    assert!(h.undo(&mut d), "one step back");
    assert_eq!(d.mesh(mesh).unwrap().num_faces(), 6);

    // The session is stale now, and the next operator must act on the restored
    // mesh — six faces subdividing to twenty-four, not twenty-four to ninety-six.
    h.dispatch_in_session(
        &r,
        &mut d,
        Some(&mut session),
        &c,
        "mesh.subdivide",
        &PropertyMap::new(),
    )
    .expect("mesh.subdivide");
    assert_eq!(
        d.mesh(mesh).unwrap().num_faces(),
        24,
        "the session wrote stale topology over an undo"
    );
}
