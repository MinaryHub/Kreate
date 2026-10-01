//! `F` on selected edges builds its face from *those* edges, through the same
//! path a click and a key take in the application: picks in edge mode, then
//! the fill in the edit session.

use kreate_math::Vec3;
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
    (registry, History::new(), doc, mesh, Context { mode: Mode::EditMesh })
}

fn params(pairs: &[(&'static str, PropertyValue)]) -> PropertyMap {
    let mut map = PropertyMap::new();
    for (name, value) in pairs {
        map.set(name, value.clone());
    }
    map
}

/// The edge whose two ends both satisfy `at`.
fn edge_where(doc: &Document, mesh: MeshId, at: impl Fn(Vec3) -> bool) -> u32 {
    let m = doc.mesh(mesh).unwrap();
    let p = m.positions();
    m.edges()
        .iter()
        .position(|[a, b]| at(p[*a as usize]) && at(p[*b as usize]))
        .expect("such an edge") as u32
}

fn pick_edge(r: &OperatorRegistry, h: &mut History, d: &mut Document, c: &Context, edge: u32, extend: bool) {
    h.dispatch(
        r,
        d,
        c,
        "mesh.select_pick",
        &params(&[
            ("element", PropertyValue::Int(edge as i64)),
            ("mode", PropertyValue::Enum(1)),
            ("extend", PropertyValue::Bool(extend)),
        ]),
    )
    .expect("mesh.select_pick");
}

/// Every face, as its corners' positions.
fn faces(doc: &Document, mesh: MeshId) -> Vec<Vec<Vec3>> {
    let m = doc.mesh(mesh).unwrap();
    (0..m.num_faces())
        .map(|f| m.face_verts(f).iter().map(|v| m.positions()[*v as usize]).collect())
        .collect()
}

/// The faces in the mesh now that were not in `before`. By shape, not by
/// number: a new face may take a slot an earlier edit freed, anywhere in the
/// order.
fn new_faces(doc: &Document, mesh: MeshId, before: &[Vec<Vec3>]) -> Vec<Vec<Vec3>> {
    faces(doc, mesh)
        .into_iter()
        .filter(|f| !before.iter().any(|b| same_points(b, f)))
        .collect()
}

fn ends(doc: &Document, mesh: MeshId, edges: &[u32]) -> Vec<Vec3> {
    let m = doc.mesh(mesh).unwrap();
    edges
        .iter()
        .flat_map(|e| m.edges()[*e as usize])
        .map(|v| m.positions()[v as usize])
        .collect()
}

fn same_points(a: &[Vec3], b: &[Vec3]) -> bool {
    a.len() == b.len() && a.iter().all(|p| b.iter().any(|q| p.distance(*q) < 1e-5))
}

#[test]
fn f_on_two_picked_edges_spans_those_two_edges() {
    let (r, mut h, mut d, mesh, c) = setup();
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);
    // Top front and bottom back: no face in common, no corner in common.
    let top_front = edge_where(&d, mesh, |p| p.z > 0.5 && p.y < -0.5);
    let bottom_back = edge_where(&d, mesh, |p| p.z < -0.5 && p.y > 0.5);
    pick_edge(&r, &mut h, &mut d, &c, top_front, false);
    pick_edge(&r, &mut h, &mut d, &c, bottom_back, true);
    let corners = ends(&d, mesh, &[top_front, bottom_back]);
    let before = faces(&d, mesh);

    h.dispatch_in_session(&r, &mut d, Some(&mut session), &c, "mesh.fill", &PropertyMap::new())
        .expect("mesh.fill");

    let made = new_faces(&d, mesh, &before);
    assert_eq!(made.len(), 1, "{made:?}");
    assert!(same_points(&made[0], &corners), "made {:?}, selected {:?}", made[0], corners);
}

#[test]
fn f_on_two_edges_of_an_open_strip_closes_between_them() {
    // Two faces deleted from the cube leave a ring; picking the two border
    // edges across a gap and pressing F bridges that gap and nothing else.
    let (r, mut h, mut d, mesh, c) = setup();
    {
        let m = d.mesh_mut(mesh).unwrap();
        let mut bm = m.to_bmesh();
        let top = bm
            .face_ids()
            .max_by(|a, b| bm.face_normal(*a).z.total_cmp(&bm.face_normal(*b).z))
            .unwrap();
        bm.remove_face(top);
        *m = bm.to_mesh_data();
    }
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);
    let front = edge_where(&d, mesh, |p| p.z > 0.5 && p.y < -0.5);
    let back = edge_where(&d, mesh, |p| p.z > 0.5 && p.y > 0.5);
    pick_edge(&r, &mut h, &mut d, &c, front, false);
    pick_edge(&r, &mut h, &mut d, &c, back, true);
    let corners = ends(&d, mesh, &[front, back]);
    let before = faces(&d, mesh);

    h.dispatch_in_session(&r, &mut d, Some(&mut session), &c, "mesh.fill", &PropertyMap::new())
        .expect("mesh.fill");

    let made = new_faces(&d, mesh, &before);
    assert_eq!(made.len(), 1, "{made:?}");
    assert!(same_points(&made[0], &corners), "made {:?}, selected {:?}", made[0], corners);
}

/// The report: delete a face, pick two edges, F — and the face came out on
/// edges nobody picked. A baked mesh numbers its edges in its own order, not
/// the edit structure's, and after the delete the two disagree everywhere;
/// the picked edges' selection was carried across by number.
#[test]
fn f_after_a_delete_in_the_session_still_spans_the_picked_edges() {
    let (r, mut h, mut d, mesh, c) = setup();
    let mut session = EditSessions::new();
    session.open_for(&d, &[mesh]);
    // The top face, deleted in the session.
    let top = {
        let m = d.mesh(mesh).unwrap();
        (0..m.num_faces())
            .find(|f| m.face_verts(*f).iter().all(|v| m.positions()[*v as usize].z > 0.5))
            .unwrap()
    };
    h.dispatch(
        &r,
        &mut d,
        &c,
        "mesh.select_pick",
        &params(&[("element", PropertyValue::Int(top as i64)), ("mode", PropertyValue::Enum(2))]),
    )
    .expect("mesh.select_pick");
    h.dispatch_in_session(
        &r,
        &mut d,
        Some(&mut session),
        &c,
        "mesh.delete",
        &params(&[("mode", PropertyValue::Enum(3))]),
    )
    .expect("mesh.delete");

    // The hole's front and back edges.
    let front = edge_where(&d, mesh, |p| p.z > 0.5 && p.y < -0.5);
    let back = edge_where(&d, mesh, |p| p.z > 0.5 && p.y > 0.5);
    pick_edge(&r, &mut h, &mut d, &c, front, false);
    pick_edge(&r, &mut h, &mut d, &c, back, true);
    let corners = ends(&d, mesh, &[front, back]);
    let before = faces(&d, mesh);

    h.dispatch_in_session(&r, &mut d, Some(&mut session), &c, "mesh.fill", &PropertyMap::new())
        .expect("mesh.fill");

    let made = new_faces(&d, mesh, &before);
    assert_eq!(made.len(), 1, "{made:?}");
    assert!(same_points(&made[0], &corners), "made {:?}, selected {:?}", made[0], corners);
}
