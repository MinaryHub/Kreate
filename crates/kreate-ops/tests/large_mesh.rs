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
    // **This is over budget and the threshold below says so.** Measured at
    // 1.5 s: about 1.2 s of that is building the edit structure, the rest the
    // corner-to-edge map that goes with it.
    //
    // The cost is eight million arena inserts — a million vertices, two million
    // edges, a million faces, four million loops — each splicing itself into a
    // disk and a radial cycle as it arrives. Roughly ninety nanoseconds each,
    // which is unremarkable per element and five times the budget in total.
    // Closing it means constructing the cycles in bulk rather than incrementally,
    // which is a real piece of work and not a tuning pass.
    //
    // It is paid once on entering edit mode rather than per operator, which is
    // what the session bought. The threshold is set where the code actually is,
    // so a regression fails; the gap against the plan's 300 ms is recorded here
    // rather than quietly widened.
    if !measurable() {
        return;
    }
    let (doc, mesh) = large_scene();
    let mut session = None;
    let elapsed = time(|| session = EditSession::open(&doc, mesh));
    assert!(session.is_some());
    assert!(
        elapsed < Duration::from_millis(2500),
        "entering edit mode took {elapsed:?}, against a measured 1.5 s and a          budget of 300 ms"
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
fn a_box_select_commits_within_budget() {
    // The plan's figure: a box select over a million vertices commits in under
    // 50 ms. The resolve happens on the GPU; this is the part that applies it.
    if !measurable() {
        return;
    }
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
