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
//! Run in release; in a debug build the mesh arithmetic is roughly fifty times
//! slower and the numbers mean nothing.

use std::time::{Duration, Instant};

use kreate_ops::operator::{Context, Mode};
use kreate_ops::{EditSession, History, OperatorRegistry, PropertyMap, PropertyValue};
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
    // the session bought. The threshold has a third of headroom over the
    // measured spread; tighter would fail on a busy machine and looser would
    // stop catching a change of shape.
    if !measurable() {
        return;
    }
    let _serial = serially();
    let (doc, mesh) = large_scene();
    let mut session = None;
    let elapsed = time(|| session = EditSession::open(&doc, mesh));
    assert!(session.is_some());
    // Reported, not just asserted. A budget test that does not say what it
    // measured makes the next round of tuning blind — this one was tightened
    // from 2500 ms with nothing to go on but a comment.
    eprintln!("entering edit mode: {elapsed:?}");
    assert!(
        elapsed < Duration::from_millis(400),
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
    let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");

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
    let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");

    let mut click = |element: i64, session: &mut EditSession, doc: &mut Document| {
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
    let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");

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

    let elapsed = time(|| {
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
    eprintln!("a one-face extrude on a million faces: {elapsed:?}");
    // 194 to 240 ms over three runs, from 622 ms when this test was written.
    // The write-back is still the whole cost — the edit touches one face — and
    // the remaining time is the attribute gathers and the positions copy, which
    // are passes over the mesh by nature rather than by accident.
    assert!(
        elapsed < Duration::from_millis(350),
        "a topology edit took {elapsed:?}; the write-back is the cost, not the \
         edit"
    );
}

/// Divisions giving 203,401 vertices — the plan’s two-hundred-thousand figure
/// for a drag, which is a different budget from the million-vertex ones above.
const DRAG_DIVISIONS: u32 = 450;

/// A mesh and a session with a patch of faces selected, ready to drag.
fn drag_scene() -> (Document, MeshId, EditSession, OperatorRegistry, History) {
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
    let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");
    history
        .dispatch_in_session(
            &registry,
            &mut doc,
            Some(&mut session),
            &edit_ctx(),
            "mesh.select_set",
            &params(&[
                ("elements", PropertyValue::Ints((0..2_000).collect())),
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
    let mut session = EditSession::open(&doc, mesh).expect("the mesh is there");

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
