//! The outliner at scale.
//!
//! A modelling scene reaches tens or hundreds of thousands of objects, and every
//! part of the path from document to panel has an obvious implementation that is
//! quadratic:
//!
//! - finding each node's children by searching every object;
//! - asking how many objects share a mesh, once per row;
//! - resolving a selection by searching for each key's object.
//!
//! All three are invisible at twenty objects and take minutes at a hundred
//! thousand. This is the test that says so.
//!
//! # Absolute budgets, not ratios
//!
//! The tempting test is a ratio: quadruple the scene, and the work must not grow
//! sixteenfold. It does not work here, and the reason is worth recording.
//!
//! An `Object` is a few hundred bytes, so eight thousand of them fit in cache
//! and thirty-two thousand do not. Measured across that boundary a *linear*
//! walk costs about ten times more for four times the objects — almost exactly
//! what a quadratic one would cost. The ratio is measuring the cache hierarchy,
//! not the algorithm, and no tolerance separates the two.
//!
//! An absolute budget at a hundred thousand objects does separate them, by four
//! orders of magnitude: linear is milliseconds, and the quadratic version this
//! file was written to catch takes tens of seconds. The budgets below are
//! therefore loose on purpose — they are there to catch a change of *shape*,
//! not to police a few milliseconds on a loaded machine.

use std::time::{Duration, Instant};

use kreate::shell::{object_key, outliner_rows, OutlinerOrder};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

/// A flat scene of `n` objects, all sharing one mesh.
///
/// Sharing is the harder case for the mesh-user count, and it is also the
/// realistic one: a scene with a hundred thousand objects got there by
/// duplicating something.
fn flat_scene(n: usize) -> Document {
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    for i in 0..n {
        doc.add_object(Object::new(format!("Object{i}"), ObjectData::Mesh(mesh)));
    }
    doc
}

/// A scene of `n` objects in one long parent chain.
///
/// The worst case for the hierarchy walk: maximum depth, so every node's
/// children have to be found.
fn deep_scene(n: usize) -> Document {
    // Parented at construction rather than with `set_parent`, which asks for the
    // parent's world transform and so builds a chain in quadratic time — which
    // would swamp the thing being measured.
    let mut doc = Document::new();
    let mut parent = None;
    for i in 0..n {
        let mut object = Object::new(format!("Object{i}"), ObjectData::Empty);
        object.parent = parent;
        parent = Some(doc.add_object(object));
    }
    doc
}

fn time(f: impl FnOnce()) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

/// The fastest of several runs.
///
/// A single measurement of something in the tens of microseconds is mostly
/// scheduler noise, and a ratio between two noisy numbers is worse than either.
/// The minimum is the standard answer: the run least interrupted by everything
/// else on the machine is the one that reflects the work.
fn fastest(runs: usize, mut f: impl FnMut()) -> Duration {
    (0..runs).map(|_| time(&mut f)).min().unwrap_or_default()
}

/// The scene size the budgets are stated at.
///
/// The plan's Phase 3 gate names a hundred thousand rows, and that is also where
/// the difference between linear and quadratic stops being arguable.
const SCALE: usize = 100_000;

/// What a whole-scene pass may cost.
///
/// Generous: these run when the document changes, not per frame. The quadratic
/// versions take tens of seconds, so this catches them with room to spare on a
/// machine doing other things.
const WHOLE_SCENE_BUDGET: Duration = Duration::from_millis(500);

#[test]
fn the_hierarchy_walk_stays_within_budget() {
    // Searching every object for the children of each node is the obvious way to
    // write this, and at a hundred thousand objects it is ten billion
    // comparisons — for a walk the outliner asks for whenever the scene changes.
    //
    // A deep chain rather than a flat one: it is the worst case, because every
    // node has children to find.
    let doc = deep_scene(SCALE);
    let elapsed = fastest(3, || {
        std::hint::black_box(doc.hierarchy());
    });
    assert_eq!(doc.hierarchy().len(), SCALE);
    assert!(
        elapsed < WHOLE_SCENE_BUDGET,
        "walking a hierarchy of {SCALE} took {elapsed:?}, which is the shape of \
         a quadratic search rather than an indexed walk"
    );
}

#[test]
fn building_the_row_order_stays_within_budget() {
    // The mesh-user count is the usual culprit here: asking how many objects
    // share a mesh, once per row, is the same quadratic shape.
    let doc = flat_scene(SCALE);
    let elapsed = fastest(3, || {
        let mut order = OutlinerOrder::default();
        order.sync(&doc);
        std::hint::black_box(order.len());
    });
    assert!(
        elapsed < WHOLE_SCENE_BUDGET,
        "ordering {SCALE} objects took {elapsed:?}"
    );
}

#[test]
fn resolving_a_whole_scene_selection_stays_within_budget() {
    // Selecting everything is one keystroke, and resolving it looks up one
    // object per key. A search per key is quadratic; the key is the arena slot,
    // so it is a lookup.
    let doc = flat_scene(SCALE);
    let keys: Vec<u32> = doc.objects().map(|(id, _)| object_key(id)).collect();
    let elapsed = fastest(3, || {
        for key in &keys {
            std::hint::black_box(kreate::shell::object_from_key(&doc, *key));
        }
    });
    assert!(
        elapsed < WHOLE_SCENE_BUDGET,
        "resolving {SCALE} keys took {elapsed:?}"
    );
}

#[test]
fn drawing_a_screenful_costs_the_same_at_any_scale() {
    // The claim the whole design rests on: a hundred thousand objects must cost
    // the same to draw as twenty, because only what is on screen is *built*.
    //
    // Rows are materialised one at a time by index; handing the widget a `Vec` of
    // rows instead would put a hundred thousand `String` clones in every frame.
    let small = flat_scene(100);
    let large = flat_scene(SCALE);
    let mut small_order = OutlinerOrder::default();
    let mut large_order = OutlinerOrder::default();
    small_order.sync(&small);
    large_order.sync(&large);

    // A screenful, drawn from the middle of the list.
    let screenful = 40;
    fn draw(order: &OutlinerOrder, doc: &Document, from: usize, count: usize) {
        for index in from..from + count {
            std::hint::black_box(order.row(doc, index));
        }
    }

    let cheap = fastest(9, || draw(&small_order, &small, 30, screenful));
    let dear = fastest(9, || draw(&large_order, &large, 50_000, screenful));

    // A ratio is the right shape of test *here*, unlike the whole-scene ones:
    // both sides do the same fixed amount of work — forty rows — so the claim is
    // that the scene's size does not enter into it at all. Some slack for the
    // larger scene's colder cache, and none for scanning the list per row.
    let ratio = dear.as_secs_f64() / cheap.as_secs_f64().max(1e-9);
    assert!(
        ratio < 20.0,
        "a screenful cost {ratio:.1}x more in the large scene ({cheap:?} then \
         {dear:?}); something is scanning the whole list per row"
    );
}

#[test]
fn the_order_is_reused_until_the_document_changes() {
    // The other half of the cost story: syncing an unchanged document has to be
    // free, or caching the order buys nothing.
    let doc = flat_scene(50_000);
    let mut order = OutlinerOrder::default();
    let first = time(|| order.sync(&doc));
    let again = time(|| order.sync(&doc));
    assert!(
        again < first / 10 || again < Duration::from_micros(50),
        "an unchanged document was re-ordered: {first:?} then {again:?}"
    );
}

#[test]
fn the_order_is_rebuilt_when_the_document_changes() {
    // And the half that keeps it correct. A cache that never invalidates shows
    // the user a scene that no longer exists.
    let mut doc = flat_scene(10);
    let mut order = OutlinerOrder::default();
    order.sync(&doc);
    assert_eq!(order.len(), 10);

    doc.add_object(Object::new("Another", ObjectData::Empty));
    order.sync(&doc);
    assert_eq!(order.len(), 11);
}

#[test]
fn the_rows_agree_with_the_document() {
    // Correctness alongside the cost: the lazy path has to produce what the eager
    // one did. Names, depth, sharing.
    let doc = deep_scene(50);
    let rows = outliner_rows(&doc);
    assert_eq!(rows.len(), 50);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row.name, format!("Object{i}"));
        assert_eq!(row.depth, i, "a parent chain indents one per level");
    }

    let shared = outliner_rows(&flat_scene(7));
    for row in &shared {
        assert_eq!(row.mesh_users, 7, "all seven objects share one mesh");
    }
}
