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
use kreate_ops::{EditSessions, History, OperatorRegistry, PropertyMap, PropertyValue};
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
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);

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
    assert_eq!(
        session
            .get_mut(mesh)
            .expect("a session")
            .bmesh_mut()
            .face_ids()
            .count(),
        24
    );
    assert!(
        session.get(mesh).is_some_and(|s| s.matches(&d, mesh)),
        "the session should be current with what it just wrote"
    );
}

#[test]
fn the_session_carries_across_several_operators() {
    // The decode happens once on opening, not once per operator. Three
    // subdivisions in a row, and the session is current before each.
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);

    for level in 0..3 {
        assert!(
            session.get(mesh).is_some_and(|s| s.matches(&d, mesh)),
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
    let mut session = EditSessions::new();
    session.open_for(&d, &[other]);
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
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);

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
        session.get(mesh).is_some_and(|s| s.matches(&d, mesh)),
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
fn a_move_made_outside_the_session_is_adopted_rather_than_overwritten() {
    // The dangerous case, and the one the staleness check exists for: something
    // changed the mesh by another route. Writing back what the session holds
    // would silently revert it — which for a transform means the user’s drag
    // vanishing on their next extrude.
    //
    // A move is *not* staleness, though, and that distinction is the point.
    // Positions carry no connectivity, so the structure the session holds is
    // still the right one; it only has the wrong numbers in it. Copying four
    // arrays across is a fraction of decoding the mesh again — measured at 35 ms
    // against 81 for an extrude after a drag on two hundred thousand vertices.
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);

    // A move, applied straight to the document — a transform, or an undo.
    for p in d.mesh_mut(mesh).unwrap().positions_mut() {
        p.z += 5.0;
    }
    assert!(
        session.get(mesh).is_some_and(|s| s.matches(&d, mesh)),
        "a move is not a change of connectivity"
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

    // The move survived: the operator subdivided the moved mesh rather than
    // writing the session’s own copy back over it.
    let lowest = d
        .mesh(mesh)
        .unwrap()
        .positions()
        .iter()
        .map(|p| p.z)
        .fold(f32::INFINITY, f32::min);
    assert!(
        lowest > 3.0,
        "the session wrote stale positions over the move: lowest z is {lowest}"
    );
}

#[test]
fn a_session_stale_from_a_topology_change_reopens_rather_than_overwriting() {
    // The other side of the same line. Connectivity replaced by another route
    // — an undo across an extrude, a modifier applied, a file reloaded — is a
    // different mesh, and no amount of copying arrays across makes the held
    // structure describe it. This one has to be decoded again.
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);

    // A cylinder where the cube was: a different face count, so the difference
    // cannot be papered over by adopting anything.
    *d.mesh_mut(mesh).unwrap() = kreate_mesh::primitives::cylinder(1.0, 2.0, 12);
    kreate_mesh::selection::select_all(d.mesh_mut(mesh).unwrap());
    let replaced_faces = d.mesh(mesh).unwrap().num_faces();
    assert!(
        !session.get(mesh).is_some_and(|s| s.matches(&d, mesh)),
        "a change of connectivity is staleness"
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

    // Subdividing the *cylinder*, not the cube the session still held.
    let faces = d.mesh(mesh).unwrap().num_faces();
    assert!(
        faces > replaced_faces,
        "{faces} faces after subdividing {replaced_faces}"
    );
    assert_ne!(
        faces, 24,
        "the session wrote the cube back over the replacement"
    );
}

#[test]
fn positions_are_adopted_by_rank_even_when_the_arena_has_holes() {
    // The rule `adopt_positions` has to get right, and the one a symmetric
    // fixture cannot check. The document mesh is dense: vertex `i` is the
    // `i`-th one. The edit structure is not — a delete frees arena slots and
    // leaves gaps — so pairing by slot index puts every position after the
    // first gap on the wrong vertex.
    //
    // Off by one is the whole failure, and it is invisible on any mesh where
    // the vertices are interchangeable. So each vertex is given a position
    // nothing else could have, and the operator that follows is one that
    // changes no topology: every position must come out exactly where the move
    // put it.
    let (r, mut h, mut d, mesh, c) = setup();

    // A delete, to make the gaps: one vertex, which frees its arena slot and
    // takes the faces around it with it.
    {
        let m = d.mesh_mut(mesh).unwrap();
        m.ensure_selection_layers();
        kreate_mesh::selection::set_selected(m, kreate_mesh::attribute::Domain::Point, 0, true);
    }
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);
    h.dispatch_in_session(
        &r,
        &mut d,
        Some(&mut session),
        &c,
        "mesh.delete",
        &params(&[("mode", PropertyValue::Enum(0))]),
    )
    .expect("mesh.delete");
    let gaps = session
        .get(mesh)
        .expect("a session")
        .bmesh()
        .slot_count(kreate_mesh::attribute::Domain::Point)
        - d.mesh(mesh).unwrap().num_verts() as usize;
    assert!(
        gaps > 0,
        "the fixture has no gaps, so it cannot tell slot from rank"
    );

    // A distinctive position per vertex, written straight to the document the
    // way a transform writes them.
    let expected: Vec<glam::Vec3> = {
        let m = d.mesh_mut(mesh).unwrap();
        for (i, p) in m.positions_mut().iter_mut().enumerate() {
            *p = glam::Vec3::new(i as f32 * 100.0, 0.0, 0.0);
        }
        m.positions().to_vec()
    };

    // An operator that touches attributes and no connectivity, so the vertex
    // order it writes back is the order it read.
    kreate_mesh::selection::select_all(d.mesh_mut(mesh).unwrap());
    h.dispatch_in_session(
        &r,
        &mut d,
        Some(&mut session),
        &c,
        "mesh.mark_sharp",
        &params(&[("sharp", PropertyValue::Bool(true))]),
    )
    .expect("mesh.mark_sharp");

    assert_eq!(
        d.mesh(mesh).unwrap().positions(),
        expected.as_slice(),
        "positions were paired with the wrong vertices"
    );
}

#[test]
fn undo_still_restores_through_a_session() {
    // The session is a staging area, not a second source of truth. Undo restores
    // the document, and the next operator has to see that rather than the
    // topology the session was holding.
    let (r, mut h, mut d, mesh, c) = setup();
    select_all(&r, &mut h, &mut d, &c);
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);

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

/// The face of `mesh` whose centroid is highest.
fn top_face(d: &Document, mesh: MeshId) -> (u32, f32) {
    let m = d.mesh(mesh).unwrap();
    let positions = m.positions();
    let corner_verts = m.corner_verts();
    (0..m.num_faces())
        .map(|f| {
            let z = m
                .face_corner_range(f)
                .map(|c| positions[corner_verts[c] as usize].z)
                .sum::<f32>()
                / m.face_len(f) as f32;
            (f, z)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

#[test]
fn pressing_extrude_twice_extrudes_the_same_face_twice() {
    // The report: in edit mode, each press of E extruded a different face.
    // The first press was right; the second grabbed whichever face happened
    // to share a slot number with the new lid once the arena had a hole in
    // it. Through the session, because the session is where the selection is
    // adopted onto the edit structure — the path a script without one never
    // takes, which is why no test had seen it.
    let (r, mut h, mut d, mesh, c) = setup();
    let (top, z0) = top_face(&d, mesh);
    kreate_mesh::selection::set_selected(
        d.mesh_mut(mesh).unwrap(),
        kreate_mesh::attribute::Domain::Face,
        top,
        true,
    );
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);

    let faces_before = d.mesh(mesh).unwrap().num_faces();
    for press in 1..=2 {
        h.dispatch_in_session(
            &r,
            &mut d,
            Some(&mut session),
            &c,
            "mesh.extrude_region",
            &params(&[("offset", PropertyValue::Float(1.0))]),
        )
        .expect("mesh.extrude_region");

        let m = d.mesh(mesh).unwrap();
        // The extruded face is deleted and a lid and four walls appear: four
        // more faces per press, and — the part that matters here — a hole in
        // the edit structure's arena from the first press on.
        assert_eq!(
            m.num_faces(),
            faces_before + 4 * press,
            "press {press}: a lid and four walls for the face that was there"
        );
        // The selection is the new lid and nothing else, and the lid sits one
        // unit higher per press — which it can only do if the second press
        // extruded the first press's lid rather than some other face.
        let selected: Vec<u32> = m
            .attributes()
            .get_bool(kreate_mesh::attribute::names::SELECT_FACE)
            .map(|s| (0..s.len() as u32).filter(|i| s[*i as usize]).collect())
            .unwrap_or_default();
        let (lid, z) = top_face(&d, mesh);
        assert_eq!(selected, vec![lid], "press {press}: the selection is not the lid alone");
        assert!(
            (z - (z0 + press as f32)).abs() < 1e-4,
            "press {press}: the lid is at z={z}, expected {}",
            z0 + press as f32
        );
    }
}

