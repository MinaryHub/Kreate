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

// --- proportional editing ---------------------------------------------------

/// A row of six vertex pairs a unit apart along X.
///
/// A strip rather than a patch, so distance from the selection is just the
/// index and every expectation below reads as a number rather than a diagram.
fn strip() -> MeshData {
    let positions: Vec<glam::Vec3> = (0..6)
        .flat_map(|i| {
            [
                glam::Vec3::new(i as f32, 0.0, 0.0),
                glam::Vec3::new(i as f32, 1.0, 0.0),
            ]
        })
        .collect();
    let faces: Vec<Vec<u32>> = (0..5)
        .map(|i| vec![i * 2, i * 2 + 2, i * 2 + 3, i * 2 + 1])
        .collect();
    MeshData::from_faces(positions, &faces)
}

#[test]
fn proportional_editing_drags_the_neighbourhood_along() {
    // The reason §2.2 calls this central rather than a nicety: without it every
    // vertex move is a spike, and shaping a surface means moving each vertex by
    // hand.
    let (mut doc, ctx) = scene(strip());
    // The pair at x = 0.
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0, 1]);
    let before = positions(&doc);

    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[
            ("translation", PropertyValue::Vec3([0.0, 0.0, 1.0])),
            ("proportional_size", PropertyValue::Float(2.5)),
        ],
    )
    .expect("translate");

    let after = positions(&doc);
    let lift = |i: usize| (after[i].z - before[i].z) as f64;

    assert!((lift(0) - 1.0).abs() < 1e-6, "the selection moves fully");
    assert!((lift(1) - 1.0).abs() < 1e-6);
    // Inside the radius, falling off with distance.
    assert!(lift(2) > 0.0 && lift(2) < 1.0, "x=1 lifted {}", lift(2));
    assert!(lift(4) > 0.0 && lift(4) < lift(2), "x=2 lifted {}", lift(4));
    // Outside it, untouched.
    assert!(
        lift(6).abs() < 1e-6,
        "x=3 is beyond 2.5 and lifted {}",
        lift(6)
    );
    assert!(lift(10).abs() < 1e-6, "x=5 lifted {}", lift(10));
}

#[test]
fn the_falloff_is_flat_at_both_ends() {
    // A linear falloff leaves a visible cone with a crease at its rim, which is
    // the artefact proportional editing exists to avoid. Checked as the shape of
    // the curve: the change per step is small near the selection and near the
    // rim, and largest in between.
    let (mut doc, ctx) = scene(strip());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0, 1]);
    let before = positions(&doc);
    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[
            ("translation", PropertyValue::Vec3([0.0, 0.0, 1.0])),
            ("proportional_size", PropertyValue::Float(5.0)),
        ],
    )
    .expect("translate");
    let after = positions(&doc);
    let lift = |i: usize| (after[i * 2].z - before[i * 2].z) as f64;

    // x = 1..5, weights along a smoothstep from 1 down to 0.
    let steps: Vec<f64> = (1..5).map(|i| lift(i) - lift(i + 1)).collect();
    let middle = steps[1].max(steps[2]);
    // With a margin, not a bare `<`. A linear falloff makes every step exactly
    // equal, and `0.2 < 0.2` on `f32`-derived values comes out true about as
    // often as not — so the strict comparison passed against the very thing it
    // was written to reject.
    assert!(
        steps[0] < middle * 0.9 && steps[3] < middle * 0.9,
        "the falloff is not flattened at its ends: {steps:?}"
    );
}

#[test]
fn a_zero_radius_moves_only_the_selection() {
    let (mut doc, ctx) = scene(strip());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0, 1]);
    let before = positions(&doc);
    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[
            ("translation", PropertyValue::Vec3([0.0, 0.0, 1.0])),
            ("proportional_size", PropertyValue::Float(0.0)),
        ],
    )
    .expect("translate");
    let after = positions(&doc);
    for i in 2..before.len() {
        assert_eq!(after[i], before[i], "vertex {i} moved with no radius");
    }
    assert_ne!(after[0], before[0]);
}

#[test]
fn distance_is_measured_to_the_nearest_selected_vertex() {
    // Not to the selection's median. Dragging a whole edge loop should pull the
    // mesh evenly along its length; a distance to the median would pull hardest
    // in the middle and leave the ends behind.
    let (mut doc, ctx) = scene(strip());
    // The two ends, leaving a gap in the middle.
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0, 1, 10, 11]);
    let before = positions(&doc);
    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[
            ("translation", PropertyValue::Vec3([0.0, 0.0, 1.0])),
            ("proportional_size", PropertyValue::Float(1.5)),
        ],
    )
    .expect("translate");
    let after = positions(&doc);
    let lift = |i: usize| (after[i * 2].z - before[i * 2].z) as f64;

    // x = 1 is next to the left end and x = 4 next to the right: both move, and
    // by the same amount. Measured from a median at x = 2.5 they would instead
    // be the two *closest* points and move most, with the ends least.
    assert!(lift(1) > 0.1, "x=1 lifted {}", lift(1));
    assert!(lift(4) > 0.1, "x=4 lifted {}", lift(4));
    assert!((lift(1) - lift(4)).abs() < 1e-6, "the two sides differ");
    // And the middle, further than the radius from either end, does not move.
    assert!(lift(2).abs() < 1e-6, "x=2 lifted {}", lift(2));
}

#[test]
fn a_proportional_rotation_turns_by_a_smaller_angle_rather_than_moving_less() {
    // The two are different, and the difference is visible. Interpolating the
    // *result* drags a partly-weighted vertex along the chord between where it
    // was and where a full turn would put it — which pulls it towards the pivot
    // and shrinks the mesh. Interpolating the *rotation* turns it through a
    // smaller angle, which keeps its distance and is what the tool means.
    let (mut doc, ctx) = scene(strip());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0, 1]);
    let before = positions(&doc);
    run(
        &mut doc,
        &ctx,
        "mesh.rotate",
        &[
            ("angle", PropertyValue::Float(std::f64::consts::FRAC_PI_2)),
            ("axis", PropertyValue::Enum(2)),
            ("proportional_size", PropertyValue::Float(4.0)),
        ],
    )
    .expect("rotate");
    let after = positions(&doc);

    // The pivot is the selection's median, at (0, 0.5, 0). A partly-weighted
    // vertex has to keep its distance from it.
    let pivot = (before[0] + before[1]) * 0.5;
    for i in [2usize, 4, 6] {
        let was = before[i].distance(pivot);
        let now = after[i].distance(pivot);
        assert!(
            (was - now).abs() < 1e-4,
            "vertex {i} moved from {was} to {now} from the pivot; the rotation \
             was blended as positions rather than as an angle"
        );
        assert!(
            before[i].distance(after[i]) > 1e-4,
            "vertex {i} did not move at all"
        );
    }
}

#[test]
fn a_proportional_scale_keeps_the_selection_at_full_strength() {
    // Note that this cannot distinguish blending the factor from blending the
    // result: scaling is linear in the position, so the two are algebraically
    // identical. The rotation above is the one where the difference is real.
    let (mut doc, ctx) = scene(strip());
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0, 1, 10, 11]);
    let before = positions(&doc);
    run(
        &mut doc,
        &ctx,
        "mesh.scale",
        &[
            ("scale", PropertyValue::Vec3([2.0, 1.0, 1.0])),
            ("proportional_size", PropertyValue::Float(1.5)),
        ],
    )
    .expect("scale");
    let after = positions(&doc);

    // The pivot is the median of the four selected, at x = 2.5.
    let pivot_x = 2.5f32;
    for i in [0usize, 10] {
        let was = before[i].x - pivot_x;
        let now = after[i].x - pivot_x;
        assert!(
            (now - was * 2.0).abs() < 1e-4,
            "selected vertex {i} scaled by {} rather than 2",
            now / was
        );
    }
    // A neighbour scales by less than the full factor but more than not at all.
    let was = before[2].x - pivot_x;
    let now = after[2].x - pivot_x;
    let factor = now / was;
    assert!(
        factor > 1.0 && factor < 2.0,
        "the neighbour scaled by {factor}"
    );
}
