//! Edit mode at a million vertices.
//!
//! The plan calls this the benchmark that must not be deferred, and it is right
//! about why: the performance character of the whole application is decided by
//! the mesh path, and every decision made without measuring it is a guess. A
//! design that cannot carry a million vertices is worth discovering in month two
//! rather than month eight.
//!
//! # What these numbers are
//!
//! Absolute budgets from the plan's own table, deliberately loose. They are here
//! to catch a change of *shape* — an operation that quietly became a pass over
//! the whole mesh — not to police milliseconds on a busy machine. Every one of
//! them was over budget when first measured, and each has a comment saying by
//! how much and what fixed it.
//!
//! # How loose, and why that number
//!
//! Looser than they look like they should be, and the reason is measured. The
//! mutex below keeps these benchmarks from overlapping, but it cannot keep the
//! *machine* out of it: adding three tests to this file moved
//! `entering_edit_mode` from a steady 229–268 ms to a steady 371–404, with no
//! change to the code it measures. Run alone it goes back to 251–268. A few
//! seconds of sustained load is enough for this processor to clock down, and a
//! median over three runs does not help because the whole run is slower.
//!
//! So a threshold set at 1.5x the measured figure does not test the code, it
//! tests how many other benchmarks ran first — and it fails on a busy machine,
//! which is the one behaviour a budget test must not have. These are set to
//! catch the regressions they were written for: each says what the number was
//! before the fix, and the threshold sits below that and well above the spread.
//!
//! Run in release; in a debug build the mesh arithmetic is roughly fifty times
//! slower and the numbers mean nothing.

use std::time::{Duration, Instant};

use kreate_ops::operator::{Context, Mode};
use kreate_ops::{EditSessions, History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{MeshId, Object, ObjectData};
use kreate_scene::Document;

/// Grid divisions per side. 1000 gives 1,002,001 vertices and a million faces.
const DIVISIONS: u32 = 1000;

fn large_scene() -> (Document, MeshId) {
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::grid(10.0, DIVISIONS, DIVISIONS));
    let object = doc.add_object(Object::new("Grid", ObjectData::Mesh(mesh)));
    doc.selection.add(object);
    (doc, mesh)
}

fn edit_ctx() -> Context {
    Context {
        mode: Mode::EditMesh,
    }
}

fn registry() -> OperatorRegistry {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    registry
}

fn params(pairs: &[(&'static str, PropertyValue)]) -> PropertyMap {
    let mut map = PropertyMap::new();
    for (name, value) in pairs {
        map.set(name, value.clone());
    }
    map
}

/// The median of several runs.
///
/// A single sample is not a measurement here. `entering_edit_mode` was written
/// with one and read 272 to 301 ms over four runs, which looked repeatable
/// enough; once the suite grew two more benchmarks the same call read **294,
/// 422 and 641 ms** on consecutive runs of the same binary, and the test began
/// failing for no reason but the order it ran in. The mutex around these
/// benchmarks stops them overlapping, and stops nothing else — a machine that
/// has just built and dropped a million-vertex mesh is not in the state it was
/// in a moment before.
///
/// The median rather than the minimum: the minimum is the machine on its best
/// day, which is not the number a budget should be checked against.
fn median_time(runs: usize, mut f: impl FnMut()) -> Duration {
    let mut samples: Vec<Duration> = (0..runs.max(1)).map(|_| time(&mut f)).collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn time(f: impl FnOnce()) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

/// Held for the whole of any test that measures or allocates a large mesh.
///
/// Rust runs the tests in one binary on several threads, and every test here
/// builds a million-vertex grid. They were measuring each other: entering edit
/// mode timed 328 ms run alone and 667 ms run beside the others — a two-fold
/// swing on identical code, which is more than the change of *shape* these
/// budgets exist to detect.
///
/// That is also why the thresholds used to be set so far above the measured
/// numbers: a budget that has to survive an unknown amount of interference
/// cannot be tight enough to catch anything. With this, the numbers are
/// repeatable and the thresholds can mean something.
static BENCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Takes the lock, ignoring a previous test's panic.
///
/// A poisoned lock is a *failed* benchmark, not a reason to fail the rest: the
/// first failure is the one worth reading.
///
/// Nothing asserts that this lock is taken, and nothing can: its purpose is
/// repeatability, and a timing test that failed when the lock was removed would
/// be a timing test that fails on a busy machine. A falsification sweep
/// confirmed as much — stubbing it out left every threshold passing. What it
/// buys is that the numbers printed above mean the same thing twice.
fn serially() -> std::sync::MutexGuard<'static, ()> {
    BENCH.lock().unwrap_or_else(|e| e.into_inner())
}

/// Skips in a debug build, where the numbers say nothing about the design.
fn measurable() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("skipping: mesh arithmetic is ~50x slower without optimisation");
        return false;
    }
    true
}

#[test]
fn entering_edit_mode_stays_within_budget() {
    // The plan's figure: a million vertices into edit mode in under 300 ms.
    //
    // **Met, at 272 to 301 ms over four runs.** It was 1.5 s.
    //
    // The cost was never the eight million arena inserts; those are `Vec`
    // pushes. It was that `add_edge` asked `edge_between`, which walks a
    // vertex's disk cycle and allocates a `Vec` to do it — six million times,
    // to answer a question the source mesh had already answered. The cycles are
    // now built in bulk from counting sorts, and `corner_edges` the same way.
    // See `BMesh::build_elements`.
    //
    // Paid once on entering edit mode rather than per operator, which is what
    // the session bought.
    //
    // Measured 229 to 460 ms depending on what else has run — see the header:
    // the spread is the processor, not the code. The threshold is set below the
    // 1.5 s this cost before the fix and above anything the machine has
    // produced since.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let (doc, mesh) = large_scene();
    let mut session = None;
    let elapsed = median_time(3, || {
        session = {
            let mut set = EditSessions::new();
            set.open_for(&doc, &[mesh]);
            Some(set)
        }
    });
    assert!(session.is_some());
    // Reported, not just asserted. A budget test that does not say what it
    // measured makes the next round of tuning blind — this one was tightened
    // from 2500 ms with nothing to go on but a comment.
    eprintln!("entering edit mode: {elapsed:?}");
    assert!(
        elapsed < Duration::from_millis(900),
        "entering edit mode took {elapsed:?}, against the plan's 300 ms budget"
    );
}

#[test]
fn a_click_is_not_a_pass_over_the_mesh() {
    // The one that matters most: clicking is what a user does constantly, and it
    // has to land in the frame it happened in.
    //
    // Measured at 422 ms when this test was written — a click on a million-face
    // mesh rebuilt a sorted table of every edge, every time. The session now
    // holds that map, because it is topology and a selection does not change it.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let (mut doc, mesh) = large_scene();
    let registry = registry();
    let mut history = History::new();
    let ctx = edit_ctx();
    let mut session = EditSessions::new();
    session.open_for(&doc, &[mesh]);

    // The first click warms nothing that later ones rely on; measured on its own.
    let elapsed = time(|| {
        history
            .dispatch_in_session(
                &registry,
                &mut doc,
                Some(&mut session),
                &ctx,
                "mesh.select_pick",
                &params(&[
                    ("element", PropertyValue::Int(500)),
                    ("mode", PropertyValue::Enum(2)),
                ]),
            )
            .expect("mesh.select_pick");
    });
    assert!(
        elapsed < Duration::from_millis(60),
        "a click took {elapsed:?}, which is a pass over the mesh rather than a \
         write to three layers"
    );
}

#[test]
fn a_click_stays_cheap_when_repeated() {
    // A user clicks continuously. If the first click is cheap and the second is
    // not, something is being rebuilt per click and the session is not doing its
    // job.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let (mut doc, mesh) = large_scene();
    let registry = registry();
    let mut history = History::new();
    let ctx = edit_ctx();
    let mut session = EditSessions::new();
    session.open_for(&doc, &[mesh]);

    let mut click = |element: i64, session: &mut EditSessions, doc: &mut Document| {
        history
            .dispatch_in_session(
                &registry,
                doc,
                Some(session),
                &ctx,
                "mesh.select_pick",
                &params(&[
                    ("element", PropertyValue::Int(element)),
                    ("mode", PropertyValue::Enum(2)),
                ]),
            )
            .expect("mesh.select_pick");
    };

    let total = time(|| {
        for i in 0..10 {
            click(i * 37, &mut session, &mut doc);
        }
    });
    assert!(
        total < Duration::from_millis(400),
        "ten clicks took {total:?}; the adjacency is being rebuilt per click"
    );
}

#[test]
fn a_topology_edit_writes_back_within_budget() {
    // The gap the other benchmarks left. A *selection* click writes three
    // attribute layers straight to the document and never bakes, which is why
    // `a_click_is_not_a_pass_over_the_mesh` is fast. An edit that changes
    // topology has to bake: the edit structure is written back through
    // `to_mesh_data`, and that is a pass over everything.
    //
    // Nothing measured it until now, so the cost of an actual edit on a
    // million-face mesh was simply unknown.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let (mut doc, mesh) = large_scene();
    let registry = registry();
    let mut history = History::new();
    let ctx = edit_ctx();
    let mut session = EditSessions::new();
    session.open_for(&doc, &[mesh]);

    // One face selected, so the edit itself is tiny and what is left is the
    // write-back. That is deliberate: the question is what the *crossing*
    // costs, not what the operator does.
    history
        .dispatch_in_session(
            &registry,
            &mut doc,
            Some(&mut session),
            &ctx,
            "mesh.select_pick",
            &params(&[
                ("element", PropertyValue::Int(500)),
                ("mode", PropertyValue::Enum(2)),
            ]),
        )
        .expect("mesh.select_pick");

    // Three of them, median. Repeating is sound here because an extrude leaves
    // the new face selected, so each run is another one-face extrude on a
    // mesh of the same size — and the write-back, which is the whole cost, is
    // a pass over that size either way.
    let elapsed = median_time(3, || {
        history
            .dispatch_in_session(
                &registry,
                &mut doc,
                Some(&mut session),
                &ctx,
                "mesh.extrude_region",
                &params(&[("offset", PropertyValue::Float(0.1))]),
            )
            .expect("mesh.extrude_region");
    });
    // A reference pass over the same mesh: one visit to every corner, with
    // arithmetic, allocating its result. Not a model of the write-back — just
    // something of the same order that the same processor has to do at the same
    // moment.
    let reference = median_time(3, || {
        let mesh = doc.mesh(mesh).expect("a mesh");
        std::hint::black_box(mesh.compute_face_normals());
    });
    let ratio = elapsed.as_secs_f64() / reference.as_secs_f64();
    eprintln!(
        "a one-face extrude on a million faces: {elapsed:?} \
         (reference pass {reference:?}, ratio {ratio:.1})"
    );

    // Measured as a ratio rather than in milliseconds, and that is the point of
    // this block. The absolute figure has been 222 ms and 484 ms for the same
    // code within a few minutes of each other, purely by machine load — which
    // is how the 500 ms bound that used to be here came to fail. Its own
    // comment had already admitted it could not both survive a slow run and
    // catch the 622 ms it was written against.
    //
    // A ratio cannot be thrown by that: whatever throttling does to the
    // write-back it does to the reference in the same breath. Sixteen when this
    // was written, against roughly forty-two for the version the test exists to
    // reject, so twenty-five sits between them with room on both sides.
    //
    // What it catches is still the *shape* — a write-back that became several
    // passes over the mesh where it was a few — and now it catches it on a busy
    // afternoon too.
    assert!(
        ratio < 25.0,
        "a topology edit cost {ratio:.1} reference passes ({elapsed:?} against \
         {reference:?}); the write-back is the cost, not the edit"
    );
    // And a catastrophe net, loose enough to be about correctness rather than
    // speed: something that has stopped being linear in the mesh at all.
    assert!(
        elapsed < Duration::from_secs(5),
        "a topology edit took {elapsed:?}; that is not a slow machine"
    );
}

/// Divisions giving 203,401 vertices — the plan’s two-hundred-thousand figure
/// for a drag, which is a different budget from the million-vertex ones above.
const DRAG_DIVISIONS: u32 = 450;

/// A mesh and a session with a patch of faces selected, ready to drag.
fn drag_scene() -> (Document, MeshId, EditSessions, OperatorRegistry, History) {
    drag_scene_selecting((0..2_000).collect())
}

/// The same, with the selection named: the shape of it changes how a spatial
/// index behaves, so the tests that care have to choose.
fn drag_scene_selecting(
    elements: Vec<u32>,
) -> (Document, MeshId, EditSessions, OperatorRegistry, History) {
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::grid(
        10.0,
        DRAG_DIVISIONS,
        DRAG_DIVISIONS,
    ));
    let object = doc.add_object(Object::new("Grid", ObjectData::Mesh(mesh)));
    doc.selection.add(object);

    let registry = registry();
    let mut history = History::new();
    let mut session = EditSessions::new();
    session.open_for(&doc, &[mesh]);
    history
        .dispatch_in_session(
            &registry,
            &mut doc,
            Some(&mut session),
            &edit_ctx(),
            "mesh.select_set",
            &params(&[
                ("elements", PropertyValue::Ints(elements)),
                ("mode", PropertyValue::Enum(2)),
            ]),
        )
        .expect("mesh.select_set");
    (doc, mesh, session, registry, history)
}

fn dragged(offset: f64, radius: f64) -> PropertyMap {
    params(&[
        ("translation", PropertyValue::Vec3([offset, 0.0, 0.0])),
        ("proportional_size", PropertyValue::Float(radius)),
    ])
}

#[test]
fn a_drag_frame_stays_within_budget() {
    // The plan’s figure, and one of Phase 2’s gates: dragging two hundred
    // thousand vertices costs the CPU at most a millisecond a frame. It is the
    // budget that decides whether the application feels like a modeller, because
    // a frame is a pointer position and a frame that takes 95 ms is a drag at
    // ten frames a second.
    //
    // **95 ms when first measured, and it was the mesh encode.** Every operator
    // in `ops::mesh` crossed into `BMesh` and wrote the result back, which is
    // right for the ones that change topology and pure cost for a transform:
    // moving positions needs no adjacency. `to_mesh_data` rebuilt every array,
    // every frame, to change one of them.
    //
    // Now 1.1 to 1.3 ms, through `DocMut::edit_positions`. What remains is the
    // copy-on-write clone of the positions array — 2.4 MB, and the price of the
    // snapshot undo that makes a cancelled drag free — and the pass over the
    // selection layers that finds which vertices move. Both are invariant across
    // a gesture, so both could be hoisted out of the frame; the budget is met
    // closely enough that neither is worth the state it would take to cache.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let (mut doc, _, mut session, registry, mut history) = drag_scene();
    let ctx = edit_ctx();

    let mut modal = history
        .begin_modal(
            &registry,
            &mut doc,
            Some(&mut session),
            &ctx,
            "mesh.translate",
            &dragged(0.0, 0.0),
        )
        .expect("dispatch")
        .expect("a patch of faces is a selection");
    // The median of several frames rather than one: the first pays for warming
    // the caches, and a single sample on a busy machine measures the machine.
    let mut frames: Vec<Duration> = Vec::new();
    for frame in 1..=9 {
        let step = dragged(frame as f64 * 0.01, 0.0);
        frames.push(time(|| {
            history
                .update_modal(
                    &registry,
                    &mut doc,
                    Some(&mut session),
                    &ctx,
                    &mut modal,
                    &step,
                )
                .expect("update");
        }));
    }
    frames.sort_unstable();
    let median = frames[frames.len() / 2];
    eprintln!("a drag frame at {DRAG_DIVISIONS} divisions: {median:?}");

    // Loose against the plan’s 1 ms, as every budget here is: what it catches
    // is a change of shape — a per-frame pass over the whole mesh coming back —
    // rather than a millisecond on a busy machine. A ten-times margin still
    // fails at 95.
    assert!(
        median < Duration::from_millis(10),
        "one frame of a drag took {median:?}; the transform is doing something \
         proportional to the mesh rather than to the selection"
    );
    History::cancel_modal(modal, &mut doc);
}

/// The median of five drag frames at `radius`, on an already-built scene.
///
/// Extracted so the baseline and the measurements it is compared against are
/// the same code: a baseline measured differently is not a baseline.
#[allow(clippy::too_many_arguments)]
fn drag_median(
    registry: &OperatorRegistry,
    doc: &mut Document,
    session: &mut EditSessions,
    ctx: &Context,
    history: &mut History,
    radius: f64,
) -> Duration {
    let mut modal = history
        .begin_modal(
            registry,
            doc,
            Some(session),
            ctx,
            "mesh.translate",
            &dragged(0.0, radius),
        )
        .expect("dispatch")
        .expect("a selection");
    let mut frames: Vec<Duration> = Vec::new();
    for frame in 1..=5 {
        let step = dragged(frame as f64 * 0.01, radius);
        frames.push(time(|| {
            history
                .update_modal(registry, doc, Some(session), ctx, &mut modal, &step)
                .expect("update");
        }));
    }
    History::cancel_modal(modal, doc);
    frames.sort_unstable();
    frames[frames.len() / 2]
}

#[test]
fn a_proportional_drag_frame_stays_within_budget() {
    // The same budget as the plain drag, and the plan puts proportional editing
    // squarely inside it: "200k vert drag + proportional editing, zero dropped
    // frames". It is not an advanced option — it is how a modeller shapes a
    // surface, and a tool where turning it on drops to two frames a second has
    // not shipped it.
    //
    // **450 ms a frame when first measured**, because the falloff asked every
    // vertex for its distance to every selected one. Three structures later
    // (see `weigh_neighbourhood`) it is 4 to 16 ms depending on how much of the
    // mesh the radius actually reaches — and at the largest radius here nearly
    // every vertex moves, so most of that is output rather than search.
    //
    // Two selection shapes, because they prune differently and one of them
    // caught a real defect: the first tree pruned on splitting planes alone,
    // which does nothing for a *thin* selection — a strip, an edge loop —
    // because the widest axis is its length. It measured 40 ms where a grid did
    // 3.5. Node bounds fixed it. A test with only the compact patch would have
    // passed the broken version.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let ctx = edit_ctx();

    for (shape, elements) in [
        // The first 2,000 faces of a grid: four rows, so ten units long and a
        // tenth of a unit wide.
        ("a strip", (0..2_000u32).collect::<Vec<u32>>()),
        // A square patch of the same area, in the middle.
        (
            "a patch",
            (0..45u32)
                .flat_map(|r| (0..45u32).map(move |c| r * DRAG_DIVISIONS + c + 90_000))
                .collect(),
        ),
    ] {
        let (mut doc, _, mut session, registry, mut history) = drag_scene_selecting(elements);

        // The same drag with the falloff switched off, measured here so the
        // comparison below is against a number from this machine at this
        // moment. See the note above the assertion.
        let plain = drag_median(&registry, &mut doc, &mut session, &ctx, &mut history, 0.0);
        eprintln!("{shape}, no falloff: {plain:?} a frame");

        for radius in [0.2, 1.0, 4.0] {
            let mut modal = history
                .begin_modal(
                    &registry,
                    &mut doc,
                    Some(&mut session),
                    &ctx,
                    "mesh.translate",
                    &dragged(0.0, radius),
                )
                .expect("dispatch")
                .expect("a selection");
            let mut frames: Vec<Duration> = Vec::new();
            for frame in 1..=5 {
                let step = dragged(frame as f64 * 0.01, radius);
                frames.push(time(|| {
                    history
                        .update_modal(
                            &registry,
                            &mut doc,
                            Some(&mut session),
                            &ctx,
                            &mut modal,
                            &step,
                        )
                        .expect("update");
                }));
            }
            frames.sort_unstable();
            let median = frames[frames.len() / 2];
            eprintln!("{shape}, radius {radius}: {median:?} a frame");
            History::cancel_modal(modal, &mut doc);

            // A ratio against the same drag without the falloff, not an
            // absolute time.
            //
            // The absolute form was 40 ms, and running the whole workspace
            // moved this measurement from 18 ms to 46 ms: cargo runs one test
            // binary per crate in parallel, and `serially()` only serialises
            // within this one. The threshold sat inside the machine's load
            // spread.
            //
            // A ratio between *radii* would have been the wrong fix, and this
            // is worth stating because it is the tempting one: the original
            // defect asked every vertex for its distance to every selected
            // one, which costs the same whatever the radius — both radii would
            // have measured 450 ms and the ratio would have been one. Against
            // a drag with the falloff *off*, whose cost the defect does not
            // touch, the broken version is some four hundred times over and
            // this one is under twenty.
            let ratio = median.as_secs_f64() / plain.as_secs_f64().max(1e-6);
            assert!(
                ratio < 60.0,
                "{shape} at radius {radius} took {median:?} a frame against \
                 {plain:?} without the falloff — {ratio:.0} times; the falloff \
                 is measuring against the whole selection again"
            );
            // And a catastrophe is still a failure even if the baseline
            // inflated with it.
            assert!(
                median < Duration::from_millis(250),
                "{shape} at radius {radius} took {median:?} a frame"
            );
        }
    }
}

#[test]
fn a_transform_records_only_what_it_moved() {
    // §1.7’s first mitigation, and the reason the transform declares its write
    // set. A snapshot is cheap to *make* because every array is shared, but an
    // operator that rewrites the positions array makes the snapshot hold a
    // private copy of it: a million vertices is 12 MB an undo step, and a
    // modelling session is thousands of steps.
    //
    // Measured as bytes rather than as timing, so it holds in a debug build too.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let (mut doc, mesh, mut session, registry, mut history) = drag_scene();
    let ctx = edit_ctx();

    let moved = history
        .dispatch_in_session(
            &registry,
            &mut doc,
            Some(&mut session),
            &ctx,
            "mesh.translate",
            &dragged(0.5, 0.0),
        )
        .expect("mesh.translate");
    assert_eq!(moved, kreate_ops::operator::OpResult::Finished);

    let entry = history.entries().last().expect("one entry");
    let bytes = entry.bytes();
    let verts = doc.mesh(mesh).expect("the mesh is there").num_verts();
    eprintln!("a transform of {verts} vertices recorded {bytes} bytes");

    // The selection is a patch of 2,000 faces, so a little over 2,000 vertices
    // at 16 bytes each: an index and a position. The whole mesh would be 2.4 MB.
    assert!(
        bytes < 100_000,
        "a transform recorded {bytes} bytes; the write set is not declared"
    );
}

#[test]
fn a_box_select_commits_within_budget() {
    // The plan's figure: a box select over a million vertices commits in under
    // 50 ms. The resolve happens on the GPU; this is the part that applies it.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let (mut doc, mesh) = large_scene();
    let registry = registry();
    let mut history = History::new();
    let ctx = edit_ctx();
    let mut session = EditSessions::new();
    session.open_for(&doc, &[mesh]);

    // A hundred thousand faces, which is what a generous drag over a dense mesh
    // resolves to.
    let elements: Vec<u32> = (0..100_000).collect();
    let elapsed = time(|| {
        history
            .dispatch_in_session(
                &registry,
                &mut doc,
                Some(&mut session),
                &ctx,
                "mesh.select_set",
                &params(&[
                    ("elements", PropertyValue::Ints(elements)),
                    ("mode", PropertyValue::Enum(2)),
                ]),
            )
            .expect("mesh.select_set");
    });
    assert!(
        elapsed < Duration::from_millis(200),
        "committing a box select took {elapsed:?}"
    );
}

#[test]
fn the_memory_per_vertex_holds() {
    // The plan's budgets: the document form around fifty bytes a vertex, the edit
    // structure at most 250. Memory regressions in this domain are silent right
    // up until they are fatal — a million vertices at 500 bytes is half a
    // gigabyte for one mesh.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let (doc, mesh) = large_scene();
    let data = doc.mesh(mesh).expect("the mesh is there");
    let verts = data.num_verts() as usize;

    let document_form = data.memory_bytes() / verts;
    assert!(
        document_form <= 64,
        "the document mesh is {document_form} bytes a vertex"
    );

    let edit_form = data.to_bmesh().memory_bytes() / verts;
    assert!(
        edit_form <= 250,
        "the edit structure is {edit_form} bytes a vertex"
    );
}
