//! Moving, rotating and scaling a vertex, edge or face selection.
//!
//! One set of operators serves all three element types, because an edge is its
//! two vertices and a face is its ring. These tests are mostly about the places
//! that unity could go wrong: a vertex shared by two selected faces being moved
//! twice, a rotation about a pivot that sits in the wrong place, and the two
//! operations that have nothing to do on a single point.

use kreate_math::glam::{DQuat, DVec3};
use kreate_math::DTransform;
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

/// Gives the scene's one object a transform.
///
/// The whole reason the tests below exist: every other fixture in this file
/// leaves it at the identity, where local and world space are the same thing
/// and a transform applied in the wrong one is indistinguishable from a correct
/// one.
fn place(doc: &mut Document, transform: DTransform) {
    let id = doc.objects().next().expect("an object").0;
    doc.object_mut(id).expect("object").transform = transform;
}

/// Where the mesh's vertices are in the world.
fn world_positions(doc: &Document) -> Vec<DVec3> {
    let object = doc.objects().next().expect("an object").0;
    let to_world = doc.world_transform(object);
    let id = doc.meshes().next().expect("a mesh").0;
    doc.mesh(id)
        .expect("mesh")
        .positions()
        .iter()
        .map(|p| to_world.transform_point3(p.as_dvec3()))
        .collect()
}

/// Compares world positions to what a mesh of `f32` positions can hold.
///
/// The arithmetic is `f64` throughout — the parameters, the object transform and
/// the round trip through world space — but the answer is written back into a
/// `Vec3`, so a vertex on a scaled object several units out comes back a few
/// parts in 10⁸ from the exact value. That is storage resolution and nothing
/// else: every failure these tests are looking for is off by a whole unit or at
/// a right angle.
fn assert_close(label: &str, got: &[DVec3], want: &[DVec3]) {
    assert_eq!(got.len(), want.len(), "{label}: vertex count");
    let mut worst = 0.0f64;
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let d = (*a - *b).length();
        worst = worst.max(d);
        assert!(
            d < 1e-6,
            "{label}: vertex {i} is at {a:?}, wanted {b:?} ({d:e} apart)"
        );
    }
    eprintln!("{label}: worst {worst:e}");
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

// --- the space a transform happens in ---------------------------------------
//
// The parameters come from a pointer travelling across the view, or from a
// typed number, and both of those mean *world* units. They were being written
// straight into local mesh positions, so a quarter-turned object's vertices
// moved at a right angle to the drag — and nothing noticed, because every
// fixture in this file until now left the object at the identity.

#[test]
fn a_move_on_a_rotated_object_goes_where_the_pointer_went() {
    let (mut doc, ctx) = scene(two_quads());
    // A quarter turn about Z: the object's +X now points along world +Y.
    place(
        &mut doc,
        DTransform {
            rotation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            ..DTransform::default()
        },
    );
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = world_positions(&doc);

    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([1.0, 0.0, 0.0]))],
    )
    .expect("translate");

    let want: Vec<DVec3> = before
        .iter()
        .enumerate()
        // Face 0 is vertices 0..=3; 4 and 5 belong to the other quad.
        .map(|(i, p)| if i < 4 { *p + DVec3::X } else { *p })
        .collect();
    assert_close("a move on a rotated object", &world_positions(&doc), &want);
}

#[test]
fn a_move_on_a_scaled_object_travels_the_distance_asked_for() {
    // The failure a rotation cannot show: under a scale of two, writing the
    // world offset into local positions moves the vertex *twice* as far as the
    // pointer went. Which reads as the tool being oversensitive rather than as
    // a coordinate bug, and is the sort of thing that gets "fixed" with a fudge
    // factor.
    let (mut doc, ctx) = scene(two_quads());
    place(
        &mut doc,
        DTransform {
            scale: DVec3::new(2.0, 4.0, 0.5),
            ..DTransform::default()
        },
    );
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = world_positions(&doc);

    run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([0.5, -0.25, 3.0]))],
    )
    .expect("translate");

    let offset = DVec3::new(0.5, -0.25, 3.0);
    let want: Vec<DVec3> = before
        .iter()
        .enumerate()
        .map(|(i, p)| if i < 4 { *p + offset } else { *p })
        .collect();
    assert_close("a move on a scaled object", &world_positions(&doc), &want);
}

#[test]
fn a_rotation_on_a_moved_object_turns_about_the_selection_not_the_origin() {
    // The pivot is the selection's median, and it has to be the median in the
    // world: an object translated ten units away would otherwise turn about a
    // point ten units from where the gizmo is drawn, which swings the mesh
    // across the screen instead of turning it.
    let (mut doc, ctx) = scene(two_quads());
    place(
        &mut doc,
        DTransform {
            translation: DVec3::new(10.0, -4.0, 2.0),
            rotation: DQuat::from_rotation_x(0.6),
            scale: DVec3::splat(1.5),
        },
    );
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = world_positions(&doc);
    let pivot: DVec3 = before[..4].iter().copied().sum::<DVec3>() / 4.0;

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

    let turn = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
    let want: Vec<DVec3> = before
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i < 4 {
                pivot + turn * (*p - pivot)
            } else {
                *p
            }
        })
        .collect();
    assert_close(
        "a rotation on a moved object",
        &world_positions(&doc),
        &want,
    );
}

#[test]
fn a_scale_on_a_sheared_object_is_still_exact() {
    // The case that rules out folding the object's transform into the operation
    // instead of converting per vertex. A world-axis scale conjugated into the
    // space of a non-uniformly scaled and rotated object is a shear, which no
    // axis-and-factor can express — so `run_transform` goes to world and back
    // rather than trying to rewrite the gesture.
    let (mut doc, ctx) = scene(two_quads());
    place(
        &mut doc,
        DTransform {
            translation: DVec3::new(-3.0, 7.0, 1.0),
            rotation: DQuat::from_euler(kreate_math::glam::EulerRot::XYZ, 0.4, -0.9, 1.3),
            scale: DVec3::new(3.0, 0.5, 1.75),
        },
    );
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = world_positions(&doc);
    let pivot: DVec3 = before[..4].iter().copied().sum::<DVec3>() / 4.0;

    let factor = DVec3::new(2.0, 0.25, 1.0);
    run(
        &mut doc,
        &ctx,
        "mesh.scale",
        &[("scale", PropertyValue::Vec3([factor.x, factor.y, factor.z]))],
    )
    .expect("scale");

    let want: Vec<DVec3> = before
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i < 4 {
                pivot + (*p - pivot) * factor
            } else {
                *p
            }
        })
        .collect();
    assert_close("a scale on a sheared object", &world_positions(&doc), &want);
}

#[test]
fn the_proportional_radius_is_a_world_length() {
    // The circle the viewport draws is a world circle and the wheel adjusts a
    // world length, so the falloff has to be measured there. On an object
    // scaled by four, a radius measured in local space reaches a quarter as far
    // as the circle says — and differently in each axis under a non-uniform
    // scale, which is not a discrepancy anyone could reason about.
    //
    // The fixture: two quads three world units wide once the object is scaled,
    // one vertex selected, and a radius that reaches its neighbour in world
    // space but would not in local space.
    let scale = 4.0;
    let mut reached = Vec::new();
    for local_space in [false, true] {
        let (mut doc, ctx) = scene(two_quads());
        place(
            &mut doc,
            DTransform {
                scale: DVec3::splat(if local_space { 1.0 } else { scale }),
                ..DTransform::default()
            },
        );
        select(&mut doc, names::SELECT_VERT, Domain::Point, &[0]);
        let before = positions(&doc);
        // 1.5 world units: on the unscaled object that reaches vertex 1 (one
        // unit away) and not vertex 4 (two away). On the object scaled by four,
        // vertex 1 is four world units away and nothing but the selection
        // should move.
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
        reached.push(
            before
                .iter()
                .zip(&after)
                .filter(|(a, b)| (**a - **b).length() > 1e-6)
                .count(),
        );
    }
    let (scaled, unscaled) = (reached[0], reached[1]);
    // The selection plus its three neighbours within 1.5 units: (1,0) and (0,1)
    // at one, and (1,1) at 1.414. Not (2,0) or (2,1), at two and 2.236.
    assert_eq!(
        unscaled, 4,
        "on an unscaled object a 1.5-unit radius should reach the three          neighbours within it"
    );
    assert_eq!(
        scaled, 1,
        "on an object scaled by {scale} the same radius reaches nothing but          the selection, because the neighbour is now {scale} world units away"
    );
}

#[test]
fn an_object_flattened_to_nothing_is_refused_rather_than_written_as_nan() {
    // A zero in the scale makes the object-to-world transform singular: there
    // is no way back from world space, and every vertex of the mesh is already
    // at the same world point. Writing the inverse anyway fills the positions
    // array with NaN, which is a mesh no undo can rescue because the *next*
    // operation reads it too.
    let (mut doc, ctx) = scene(two_quads());
    place(
        &mut doc,
        DTransform {
            scale: DVec3::new(1.0, 0.0, 1.0),
            ..DTransform::default()
        },
    );
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let before = positions(&doc);

    let result = run(
        &mut doc,
        &ctx,
        "mesh.translate",
        &[("translation", PropertyValue::Vec3([1.0, 1.0, 1.0]))],
    )
    .expect("translate");

    assert_eq!(result, kreate_ops::OpResult::Cancelled);
    assert_eq!(positions(&doc), before, "the mesh was written anyway");
    assert!(
        positions(&doc).iter().all(|p| p.is_finite()),
        "the mesh has NaN in it"
    );
}

#[test]
fn e_drags_along_the_selected_faces_normal_in_the_world() {
    // A cube turned a quarter about X: its top face's normal, which is +Z in
    // the mesh, points along −Y in the world — and that is the line E pulls
    // the new faces out along.
    let (mut doc, _) = scene(kreate_mesh::primitives::cube(1.0));
    let top = {
        let mesh = doc.meshes().next().unwrap().1;
        let offsets = mesh.face_offsets();
        let corners = mesh.corner_verts();
        (0..mesh.num_faces() as usize)
            .find(|f| corners[offsets[*f] as usize..offsets[f + 1] as usize].iter().all(|v| mesh.positions()[*v as usize].z > 0.0))
            .expect("a top face")
    };
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[top]);
    let object = doc.selection.active.unwrap();
    doc.object_mut(object).unwrap().transform = DTransform {
        rotation: DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
        scale: DVec3::splat(3.0),
        ..DTransform::IDENTITY
    };
    let n = kreate_ops::ops::mesh::selected_faces_normal(&doc).expect("a face is selected");
    assert!((n - DVec3::NEG_Y).length() < 1e-9, "{n:?}");
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[]);
    assert!(kreate_ops::ops::mesh::selected_faces_normal(&doc).is_none());
}

#[test]
fn ctrl_r_slides_from_the_first_end_of_the_edge_it_reports() {
    // The gesture slides the cut along the edge `loop_cut_edge` names, so a
    // position near 0 has to land near that edge's first end — in the world,
    // wherever the object is.
    let (mut doc, ctx) = scene(kreate_mesh::primitives::cube(1.0));
    select(&mut doc, names::SELECT_EDGE, Domain::Edge, &[0]);
    let object = doc.selection.active.unwrap();
    doc.object_mut(object).unwrap().transform.translation = DVec3::new(5.0, 0.0, 0.0);
    let (a, b) = kreate_ops::ops::mesh::loop_cut_edge(&doc).expect("an edge is selected");
    run(&mut doc, &ctx, "mesh.loop_cut", &[("cuts", PropertyValue::Int(1)), ("position", PropertyValue::Float(0.2))]).unwrap();
    let want = a.lerp(b, 0.2) - DVec3::new(5.0, 0.0, 0.0);
    assert!(
        positions(&doc).iter().any(|p| (p.as_dvec3() - want).length() < 1e-5),
        "no new vertex a fifth of the way from the first end"
    );
    select(&mut doc, names::SELECT_EDGE, Domain::Edge, &[]);
    assert!(kreate_ops::ops::mesh::loop_cut_edge(&doc).is_none());
}

#[test]
fn inset_and_bevel_start_where_their_gestures_start() {
    // I begins at no thickness and Ctrl+B at a hair's width; the operators
    // must run there, or the gesture is refused before the pointer moves.
    let (mut doc, ctx) = scene(kreate_mesh::primitives::cube(1.0));
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    let inset = run(&mut doc, &ctx, "mesh.inset", &[("thickness", PropertyValue::Float(0.0))]).unwrap();
    assert_eq!(inset, kreate_ops::OpResult::Finished);
    let (mut doc, ctx) = scene(kreate_mesh::primitives::cube(1.0));
    select(&mut doc, names::SELECT_EDGE, Domain::Edge, &[0]);
    let bevel = run(&mut doc, &ctx, "mesh.bevel", &[("width", PropertyValue::Float(1e-4))]).unwrap();
    assert_eq!(bevel, kreate_ops::OpResult::Finished);
}

fn flags(doc: &Document, layer: &str) -> Vec<bool> {
    let id = doc.meshes().next().expect("a mesh").0;
    doc.mesh(id).unwrap().attributes().get_bool(layer).map(|l| l.to_vec()).unwrap_or_default()
}

#[test]
fn h_hides_the_selected_face_and_alt_h_brings_it_back_selected() {
    let (mut doc, ctx) = scene(kreate_mesh::primitives::cube(1.0));
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[2]);
    run(&mut doc, &ctx, "mesh.hide", &[("mode", PropertyValue::Enum(2))]).unwrap();
    assert!(flags(&doc, names::HIDE_FACE)[2]);
    assert!(!flags(&doc, names::SELECT_FACE).iter().any(|s| *s), "the hidden face stayed selected");
    // Select all cannot reach it.
    run(&mut doc, &ctx, "mesh.select", &[("action", PropertyValue::Enum(0))]).unwrap();
    assert!(!flags(&doc, names::SELECT_FACE)[2]);
    // Nothing more to hide is a no-op, not an error.
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[]);
    assert_eq!(
        run(&mut doc, &ctx, "mesh.hide", &[("mode", PropertyValue::Enum(2))]).unwrap(),
        kreate_ops::OpResult::Cancelled
    );
    run(&mut doc, &ctx, "mesh.reveal", &[]).unwrap();
    assert!(flags(&doc, names::HIDE_FACE).is_empty(), "the hide layer outlived the reveal");
    assert!(flags(&doc, names::SELECT_FACE)[2]);
}

#[test]
fn ctrl_plus_and_ctrl_l_grow_through_the_operator() {
    let (mut doc, ctx) = scene(kreate_mesh::primitives::cube(1.0));
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    run(&mut doc, &ctx, "mesh.select", &[("action", PropertyValue::Enum(6)), ("mode", PropertyValue::Enum(2))]).unwrap();
    // A cube's face touches every face but the opposite one.
    assert_eq!(flags(&doc, names::SELECT_FACE).iter().filter(|s| **s).count(), 5);
    run(&mut doc, &ctx, "mesh.select", &[("action", PropertyValue::Enum(7)), ("mode", PropertyValue::Enum(2))]).unwrap();
    assert_eq!(flags(&doc, names::SELECT_FACE).iter().filter(|s| **s).count(), 1);
    run(&mut doc, &ctx, "mesh.select", &[("action", PropertyValue::Enum(8))]).unwrap();
    assert!(flags(&doc, names::SELECT_FACE).iter().all(|s| *s));
}

#[test]
fn a_loop_cut_can_name_its_edge_and_needs_no_selection() {
    // What Ctrl+R's hover hands over: the edge under the pointer, by number,
    // so the cut replays whatever is selected at the time.
    let (mut doc, ctx) = scene(kreate_mesh::primitives::cube(1.0));
    select(&mut doc, names::SELECT_EDGE, Domain::Edge, &[]);
    let before = positions(&doc).len();
    run(&mut doc, &ctx, "mesh.loop_cut", &[("cuts", PropertyValue::Int(2)), ("edge", PropertyValue::Int(3))]).unwrap();
    assert_eq!(positions(&doc).len(), before + 8, "two loops of four round a cube");
    // An edge the mesh does not have does nothing.
    let (mut doc, ctx) = scene(kreate_mesh::primitives::cube(1.0));
    let r = run(&mut doc, &ctx, "mesh.loop_cut", &[("edge", PropertyValue::Int(999))]).unwrap();
    assert_eq!(r, kreate_ops::OpResult::Cancelled);
}

#[test]
fn a_rotation_can_turn_about_the_3d_cursor() {
    // Even a lone vertex, which about its own middle would not move at all.
    let (mut doc, ctx) = scene(kreate_mesh::primitives::cube(1.0));
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[0]);
    doc.cursor = DVec3::new(3.0, 0.0, 0.0);
    let p0 = positions(&doc)[0].as_dvec3();
    run(
        &mut doc,
        &ctx,
        "mesh.rotate",
        &[("angle", PropertyValue::Float(std::f64::consts::PI)), ("axis", PropertyValue::Enum(2)), ("pivot", PropertyValue::Enum(1))],
    )
    .unwrap();
    let p1 = positions(&doc)[0].as_dvec3();
    let expected = doc.cursor + DQuat::from_rotation_z(std::f64::consts::PI) * (p0 - doc.cursor);
    assert!((p1 - expected).length() < 1e-5, "{p1:?} vs {expected:?}");
}

#[test]
fn k_cuts_along_its_path_and_replays_from_the_script_line() {
    let (mut doc, ctx) = scene(kreate_mesh::primitives::plane(2.0));
    let start = doc.clone();
    let points = vec![-1.0, 0.2, 0.0, 1.0, -0.3, 0.0];
    let registry = registry();
    let mut history = History::new();
    let mut p = PropertyMap::new();
    p.set("points", PropertyValue::Floats(points));
    p.set("view", PropertyValue::Vec3([0.0, 0.0, -1.0]));
    history.dispatch(&registry, &mut doc, &ctx, "mesh.knife", &p).unwrap();
    let mesh = |d: &Document| d.mesh(d.meshes().next().unwrap().0).unwrap().num_faces();
    assert_eq!(mesh(&doc), 2);
    let script = history.entries()[0].to_script_call();
    let mut replay = start;
    let outcomes = History::new().run_script(&registry, &mut replay, None, &ctx, &script);
    assert!(outcomes.iter().all(|o| o.result.is_ok()), "{script}");
    assert_eq!(mesh(&replay), 2, "{script}");
}

#[test]
fn l_adds_the_piece_under_the_pointer_and_shift_l_takes_it_away() {
    use kreate_mesh::edit::join::{join, Part};
    let a = kreate_mesh::primitives::cube(1.0);
    let mut b = kreate_mesh::primitives::cube(1.0);
    for p in b.positions_mut() {
        p.x += 3.0;
    }
    let part = |mesh| Part { mesh, transform: glam::DAffine3::IDENTITY };
    let (joined, _) = join(&[part(&a), part(&b)]);
    let (mut doc, ctx) = scene(joined);
    // Face 7 is on the second cube; one face of the first is already selected.
    select(&mut doc, names::SELECT_FACE, Domain::Face, &[0]);
    run(&mut doc, &ctx, "mesh.select_linked_pick", &[("element", PropertyValue::Int(7)), ("mode", PropertyValue::Enum(2))]).unwrap();
    let faces = |d: &Document| flags(d, names::SELECT_FACE);
    assert!(faces(&doc)[6..12].iter().all(|s| *s), "the second cube is not all selected");
    assert!(faces(&doc)[0] && !faces(&doc)[1], "the first cube's selection changed");
    run(
        &mut doc,
        &ctx,
        "mesh.select_linked_pick",
        &[("element", PropertyValue::Int(7)), ("mode", PropertyValue::Enum(2)), ("deselect", PropertyValue::Bool(true))],
    )
    .unwrap();
    assert!(faces(&doc)[6..12].iter().all(|s| !*s));
}

#[test]
fn alt_click_selects_the_loop_through_the_edge() {
    let (mut doc, ctx) = scene(kreate_mesh::primitives::grid(3.0, 3, 3));
    // An inner edge along X in the middle row: its loop runs across the grid.
    let mesh = doc.mesh(doc.meshes().next().unwrap().0).unwrap().clone();
    let edge = (0..mesh.num_edges())
        .find(|e| {
            let [a, b] = mesh.edges()[*e as usize];
            let (pa, pb) = (mesh.positions()[a as usize], mesh.positions()[b as usize]);
            (pa.y - 0.5).abs() < 1e-4 && (pb.y - 0.5).abs() < 1e-4 && pa.x.abs() < 0.6 && pb.x.abs() < 0.6
        })
        .expect("a middle edge");
    run(&mut doc, &ctx, "mesh.select_loop_pick", &[("edge", PropertyValue::Int(edge as i64))]).unwrap();
    assert_eq!(flags(&doc, names::SELECT_EDGE).iter().filter(|s| **s).count(), 3);
    // The ring instead: the edges across it.
    run(&mut doc, &ctx, "mesh.select_loop_pick", &[("edge", PropertyValue::Int(edge as i64)), ("ring", PropertyValue::Bool(true))]).unwrap();
    assert_eq!(flags(&doc, names::SELECT_EDGE).iter().filter(|s| **s).count(), 4);
}

#[test]
fn e_on_an_edge_pulls_out_a_face_and_moves_it() {
    let (mut doc, ctx) = scene(kreate_mesh::primitives::plane(2.0));
    select(&mut doc, names::SELECT_EDGE, Domain::Edge, &[0]);
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[]);
    let mesh = doc.mesh(doc.meshes().next().unwrap().0).unwrap().clone();
    let [a, b] = mesh.edges()[0];
    {
        let id = doc.meshes().next().unwrap().0;
        let m = doc.mesh_mut(id).unwrap();
        let verts = m.attributes_mut().make_mut_bool(names::SELECT_VERT).unwrap();
        verts[a as usize] = true;
        verts[b as usize] = true;
    }
    run(&mut doc, &ctx, "mesh.extrude_move", &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))]).unwrap();
    let after = doc.mesh(doc.meshes().next().unwrap().0).unwrap();
    assert_eq!(after.num_faces(), 2);
    assert_eq!(after.positions().iter().filter(|p| (p.z - 1.0).abs() < 1e-6).count(), 2);
    // Nothing selected: nothing to extrude.
    let (mut doc, ctx) = scene(kreate_mesh::primitives::plane(2.0));
    select(&mut doc, names::SELECT_VERT, Domain::Point, &[]);
    assert_eq!(run(&mut doc, &ctx, "mesh.extrude_move", &[]).unwrap(), kreate_ops::OpResult::Cancelled);
}
