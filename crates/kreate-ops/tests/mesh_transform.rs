//! Moving, rotating and scaling a vertex, edge or face selection.
//!
//! One set of operators serves all three element types, because an edge is its
//! two vertices and a face is its ring. These tests are mostly about the places
//! that unity could go wrong: a vertex shared by two selected faces being moved
//! twice, a rotation about a pivot that sits in the wrong place, and the two
//! operations that have nothing to do on a single point.

use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_mesh::MeshData;
use kreate_ops::operator::{Context, Mode};
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

fn registry() -> OperatorRegistry {
    let mut r = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut r);
    r
}

/// A unit grid of two quads sharing an edge, in the XY plane.
///
/// Two faces on purpose: the shared edge is what makes a double-applied
/// translation visible.
fn two_quads() -> MeshData {
    let p = |x: f32, y: f32| glam::Vec3::new(x, y, 0.0);
    MeshData::from_faces(
        vec![
            p(0.0, 0.0),
            p(1.0, 0.0),
            p(1.0, 1.0),
            p(0.0, 1.0),
            p(2.0, 0.0),
            p(2.0, 1.0),
        ],
        &[vec![0, 1, 2, 3], vec![1, 4, 5, 2]],
    )
}

/// A document in edit mode on `mesh`, with the named selection layers set.
fn scene(mesh: MeshData) -> (Document, Context) {
    let mut doc = Document::new();
    let id = doc.add_mesh(mesh);
    let object = doc.add_object(Object::new("mesh", ObjectData::Mesh(id)));
    doc.selection.add(object);
    (
        doc,
        Context {
            mode: Mode::EditMesh,
        },
    )
}

fn select(doc: &mut Document, layer: &str, domain: Domain, on: &[usize]) {
    let mesh_id = doc.meshes().next().expect("a mesh").0;
    let len = {
        let mesh = doc.mesh(mesh_id).expect("mesh");
        mesh.domain_len(domain)
    };
    let mut flags = vec![false; len];
    for i in on {
        flags[*i] = true;
    }
    let mesh = doc.mesh_mut(mesh_id).expect("mesh");
    mesh.ensure_selection_layers();
    mesh.attributes_mut()
        .insert(layer, domain, AttrData::Bool(flags.into()));
}

fn positions(doc: &Document) -> Vec<glam::Vec3> {
    let id = doc.meshes().next().expect("a mesh").0;
    doc.mesh(id).expect("mesh").positions().to_vec()
}

fn run(
    doc: &mut Document,
    ctx: &Context,
    op: &str,
    pairs: &[(&'static str, PropertyValue)],
) -> Result<kreate_ops::OpResult, String> {
    let registry = registry();
    let mut history = History::new();
    let mut params = PropertyMap::new();
    for (k, v) in pairs {
        params.set(k, v.clone());
    }
    history
        .dispatch(&registry, doc, ctx, op, &params)
        .map_err(|e| e.to_string())
}

#[test]
fn moving_a_selected_vertex_moves_only_it() {
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0]);
    let before = positions(&doc);

    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
    )
    .expect("translate");

    let after = positions(&doc);
    assert_eq!(after[0], before[0] + glam::Vec3::Z);
    for i in 1..before.len() {
        assert_eq!(after[i], before[i], "vertex {i} moved and should not have");
    }
}

#[test]
fn moving_a_selected_edge_moves_both_its_ends() {
    // Edge 0 is whichever edge `rebuild_edges` numbered first; the claim is
    // about the count, not about which one — exactly two vertices move.
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_EDGE, Domain::Edge, &[0]);
    let before = positions(&doc);

    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
    )
    .expect("translate");

    let after = positions(&doc);
    let moved: Vec<usize> = (0..before.len())
        .filter(|i| after[*i] != before[*i])
        .collect();
    assert_eq!(moved.len(), 2, "moved {moved:?}");
    for i in moved {
        assert_eq!(after[i], before[i] + glam::Vec3::Z);
    }
}

#[test]
fn moving_a_selected_face_moves_its_whole_ring() {
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = positions(&doc);

    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
    )
    .expect("translate");

    let after = positions(&doc);
    let moved: Vec<usize> = (0..before.len())
        .filter(|i| after[*i] != before[*i])
        .collect();
    assert_eq!(moved, vec![0, 1, 2, 3], "the first quad's four corners");
}

#[test]
fn a_vertex_shared_by_two_selected_faces_moves_once() {
    // The bug de-duplication exists to prevent. Vertices 1 and 2 belong to both
    // quads, so a naive pass over each face's ring would translate them twice
    // and tear the mesh apart along the shared edge.
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0, 1]);
    let before = positions(&doc);

    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
    )
    .expect("translate");

    let after = positions(&doc);
    for i in 0..before.len() {
        assert_eq!(
            after[i],
            before[i] + glam::Vec3::Z,
            "vertex {i} was displaced by {:?} rather than one unit",
            after[i] - before[i]
        );
    }
}

#[test]
fn a_selection_spanning_two_element_types_moves_the_union() {
    // A selection is not required to live in one domain, and the transform does
    // not ask which mode the user is in — it moves whatever is selected.
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[4]);
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = positions(&doc);

    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
    )
    .expect("translate");

    let after = positions(&doc);
    let moved: Vec<usize> = (0..before.len())
        .filter(|i| after[*i] != before[*i])
        .collect();
    assert_eq!(moved, vec![0, 1, 2, 3, 4]);
}

#[test]
fn moving_nothing_is_cancelled_rather_than_recorded() {
    // An empty selection has nothing to move. Reported as cancelled so it does
    // not land in the history — otherwise the user's next undo reaches a step
    // that changed nothing instead of the edit they wanted back.
    let (mut doc, ctx) = scene(two_quads());
    let result = run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
    )
    .expect("dispatch");
    assert!(matches!(result, kreate_ops::OpResult::Cancelled));
}

#[test]
fn rotating_a_face_turns_it_about_its_own_centre() {
    // A quarter turn about Z of the unit quad at the origin: its centre stays
    // put and its corners go round. Checked by the centre rather than by each
    // corner's new coordinates, because which corner lands where is a property
    // of the winding and not of the rotation.
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = positions(&doc);
    let centre_before = (before[0] + before[1] + before[2] + before[3]) / 4.0;

    run(
        &mut doc,
        &ctx,
        "mesh.rotate",
        &[
            ("angle", PropertyValue::Float(std::f64::consts::FRAC_PI_2)),
            ("axis", PropertyValue::Enum(2)),
        ],
    )
    .expect("rotate");

    let after = positions(&doc);
    let centre_after = (after[0] + after[1] + after[2] + after[3]) / 4.0;
    assert!(
        centre_before.distance(centre_after) < 1e-5,
        "the face's centre moved from {centre_before} to {centre_after}"
    );
    // And it really turned: every corner is somewhere else.
    for i in 0..4 {
        assert!(
            before[i].distance(after[i]) > 0.1,
            "corner {i} did not move"
        );
    }
    // The far quad is untouched.
    assert_eq!(after[4], before[4]);
    assert_eq!(after[5], before[5]);
}

#[test]
fn a_rotation_preserves_distances_within_the_selection() {
    // What makes it a rotation rather than a shear.
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = positions(&doc);

    run(
        &mut doc,
        &ctx,
        "mesh.rotate",
        &[
            ("angle", PropertyValue::Float(0.7)),
            ("axis", PropertyValue::Enum(1)),
        ],
    )
    .expect("rotate");

    let after = positions(&doc);
    for a in 0..4 {
        for b in (a + 1)..4 {
            let was = before[a].distance(before[b]);
            let now = after[a].distance(after[b]);
            assert!(
                (was - now).abs() < 1e-4,
                "the distance from {a} to {b} went from {was} to {now}"
            );
        }
    }
}

#[test]
fn scaling_a_face_moves_its_corners_and_keeps_its_centre() {
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = positions(&doc);
    let centre_before = (before[0] + before[1] + before[2] + before[3]) / 4.0;

    run(
        &mut doc,
        &ctx,
        "mesh.scale",
        &[("scale", PropertyValue::Vec3([2.0, 2.0, 2.0]))],
    )
    .expect("scale");

    let after = positions(&doc);
    let centre_after = (after[0] + after[1] + after[2] + after[3]) / 4.0;
    assert!(centre_before.distance(centre_after) < 1e-5);
    for i in 0..4 {
        let was = before[i].distance(centre_before);
        let now = after[i].distance(centre_after);
        assert!(
            (now - was * 2.0).abs() < 1e-4,
            "corner {i} is {now} from the centre, expected {}",
            was * 2.0
        );
    }
}

#[test]
fn scaling_a_single_vertex_does_nothing_and_says_so() {
    // A lone vertex *is* its own median, so scaling about it is exactly the
    // identity. Refused rather than reported as done, because an entry in the
    // history that undoes to the same mesh is worse than no entry.
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0]);
    let before = positions(&doc);

    let result = run(
        &mut doc,
        &ctx,
        "mesh.scale",
        &[("scale", PropertyValue::Vec3([2.0, 2.0, 2.0]))],
    )
    .expect("dispatch");

    assert!(matches!(result, kreate_ops::OpResult::Cancelled));
    assert_eq!(positions(&doc), before, "the mesh changed");
}

#[test]
fn rotating_a_single_vertex_does_nothing_and_says_so() {
    // The same argument as scale: there is nothing to turn about a pivot that is
    // the only point in the selection.
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0]);
    let before = positions(&doc);

    let result = run(
        &mut doc,
        &ctx,
        "mesh.rotate",
        &[
            ("angle", PropertyValue::Float(1.0)),
            ("axis", PropertyValue::Enum(2)),
        ],
    )
    .expect("dispatch");

    assert!(matches!(result, kreate_ops::OpResult::Cancelled));
    assert_eq!(positions(&doc), before);
}

#[test]
fn scaling_several_selected_vertices_does_work() {
    // The reason a single vertex is refused by *count of distinct positions*
    // rather than by a ban on vertex mode. Tapering a selected ring of vertices
    // is a core modelling move, and it has to keep working.
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0, 1, 2, 3]);
    let before = positions(&doc);

    let result = run(
        &mut doc,
        &ctx,
        "mesh.scale",
        &[("scale", PropertyValue::Vec3([2.0, 2.0, 1.0]))],
    )
    .expect("dispatch");

    assert!(matches!(result, kreate_ops::OpResult::Finished));
    let after = positions(&doc);
    for i in 0..4 {
        assert_ne!(after[i], before[i], "vertex {i} did not move");
    }
    assert_eq!(after[4], before[4], "an unselected vertex moved");
}

#[test]
fn a_single_vertex_still_moves() {
    // Translation has no pivot, so a lone vertex is a perfectly good thing to
    // move. Asserted so that the guard on rotate and scale cannot be copied
    // onto translate by mistake.
    let (mut doc, ctx) = scene(two_quads());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0]);
    let result = run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([1.0, 0.0, 0.0]))],
    )
    .expect("dispatch");
    assert!(matches!(result, kreate_ops::OpResult::Finished));
}

#[test]
fn a_transform_is_undone_exactly() {
    // The property §9 calls the most valuable single test: apply, undo, and the
    // document is what it was.
    let mut doc = Document::new();
    let id = doc.add_mesh(two_quads());
    let object = doc.add_object(Object::new("mesh", ObjectData::Mesh(id)));
    doc.selection.add(object);
    let ctx = Context {
        mode: Mode::EditMesh,
    };
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = positions(&doc);

    let registry = registry();
    let mut history = History::new();
    let mut params = PropertyMap::new();
    params.set("scale", PropertyValue::Vec3([3.0, 3.0, 3.0]));
    history
        .dispatch(&registry, &mut doc, &ctx, "mesh.scale", &params)
        .expect("scale");
    assert_ne!(positions(&doc), before);

    assert!(history.undo(&mut doc), "there was nothing to undo");
    assert_eq!(positions(&doc), before, "undo did not restore the mesh");
}

#[test]
fn the_transforms_are_refused_outside_edit_mode() {
    // They act on a mesh element selection, and there is not one in object mode.
    let (mut doc, _) = scene(two_quads());
    let object_mode = Context { mode: Mode::Object };
    let registry = registry();
    for op in ["mesh.translate", "mesh.rotate", "mesh.scale"] {
        let info = registry.get(op).expect("registered");
        assert!(
            !info.poll(&object_mode, &doc),
            "{op} polls true in object mode"
        );
    }
    let _ = &mut doc;
}
