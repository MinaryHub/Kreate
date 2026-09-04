//! The uncommitted transaction an interactive operator runs inside.
//!
//! §1.7's rule: while a modal operator is running it owns a transaction that has
//! not been committed. Nothing reaches the undo stack until it finishes, a cancel
//! restores and records nothing, and a whole gesture commits as one entry.
//!
//! These are the tests for that contract rather than for the mouse mapping,
//! which lives in the application and is tested as arithmetic. What is checked
//! here is what a drag does to the *document* and the *history* — which is where
//! a mistake costs someone their work rather than their patience.

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

fn quad() -> MeshData {
    let p = |x: f32, y: f32| glam::Vec3::new(x, y, 0.0);
    MeshData::from_faces(
        vec![p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)],
        &[vec![0, 1, 2, 3]],
    )
}

fn edit_scene() -> (Document, Context) {
    let mut doc = Document::new();
    let id = doc.add_mesh(quad());
    let object = doc.add_object(Object::new("mesh", ObjectData::Mesh(id)));
    doc.selection.add(object);
    let mesh = doc.mesh_mut(id).expect("mesh");
    mesh.ensure_selection_layers();
    mesh.attributes_mut().insert(
        names::SELECT_FACE,
        Domain::Face,
        AttrData::Bool(vec![true].into()),
    );
    (
        doc,
        Context {
            mode: Mode::EditMesh,
        },
    )
}

fn object_scene() -> (Document, Context) {
    let mut doc = Document::new();
    let id = doc.add_mesh(quad());
    let object = doc.add_object(Object::new("thing", ObjectData::Mesh(id)));
    doc.selection.add(object);
    (doc, Context { mode: Mode::Object })
}

fn positions(doc: &Document) -> Vec<glam::Vec3> {
    let id = doc.meshes().next().expect("a mesh").0;
    doc.mesh(id).expect("mesh").positions().to_vec()
}

fn moved(x: f64) -> PropertyMap {
    let mut p = PropertyMap::new();
    p.set("translation", PropertyValue::Vec3([x, 0.0, 0.0]));
    p
}

#[test]
fn beginning_a_gesture_puts_nothing_in_the_history() {
    // The rule §1.7 states. A key press is not an edit, and an entry that
    // appeared on the press would be an undo step for something the user has not
    // done yet.
    let (mut doc, ctx) = edit_scene();
    let registry = registry();
    let mut history = History::new();
    let session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "mesh.translate",
            &moved(0.0),
        )
        .expect("dispatch")
        .expect("the operator should accept a selection");
    assert!(!history.can_undo(), "the history grew on the key press");
    let _ = session;
}

#[test]
fn a_whole_drag_commits_as_one_entry() {
    // Two hundred frames of pointer motion is one thing the user did, and has to
    // undo in one step.
    let (mut doc, ctx) = edit_scene();
    let registry = registry();
    let mut history = History::new();
    let mut session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "mesh.translate",
            &moved(0.0),
        )
        .expect("dispatch")
        .expect("accepted");

    for step in 1..=200 {
        history
            .update_modal(
                &registry,
                &mut doc,
                None,
                &ctx,
                &mut session,
                &moved(step as f64 * 0.01),
            )
            .expect("update");
    }
    history.commit_modal(session);

    assert_eq!(history.entries().len(), 1, "one gesture, one entry");
    assert!(
        (positions(&doc)[0].x - 2.0).abs() < 1e-4,
        "{:?}",
        positions(&doc)[0]
    );
}

#[test]
fn a_long_drag_does_not_accumulate_drift() {
    // The reason §2.2 forbids incremental application. Each update restores and
    // re-applies from the original, so a thousand small steps ending at the same
    // value as one big step give the same answer — to the last bit.
    let registry = registry();

    let (mut incremental_doc, ctx) = edit_scene();
    let mut history = History::new();
    let mut session = history
        .begin_modal(
            &registry,
            &mut incremental_doc,
            None,
            &ctx,
            "mesh.translate",
            &moved(0.0),
        )
        .expect("dispatch")
        .expect("accepted");
    for step in 1..=1000 {
        history
            .update_modal(
                &registry,
                &mut incremental_doc,
                None,
                &ctx,
                &mut session,
                &moved(step as f64 * 0.001),
            )
            .expect("update");
    }
    history.commit_modal(session);

    let (mut direct_doc, ctx) = edit_scene();
    let mut direct = History::new();
    direct
        .dispatch(
            &registry,
            &mut direct_doc,
            &ctx,
            "mesh.translate",
            &moved(1.0),
        )
        .expect("direct");

    assert_eq!(
        positions(&incremental_doc),
        positions(&direct_doc),
        "a thousand steps to 1.0 differ from one step to 1.0"
    );
}

#[test]
fn cancelling_restores_the_document_and_records_nothing() {
    let (mut doc, ctx) = edit_scene();
    let before = positions(&doc);
    let registry = registry();
    let mut history = History::new();
    let mut session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "mesh.translate",
            &moved(0.0),
        )
        .expect("dispatch")
        .expect("accepted");
    history
        .update_modal(&registry, &mut doc, None, &ctx, &mut session, &moved(5.0))
        .expect("update");
    assert_ne!(positions(&doc), before, "the drag should have moved it");

    History::cancel_modal(session, &mut doc);
    assert_eq!(positions(&doc), before, "cancel did not restore the mesh");
    assert!(!history.can_undo(), "cancel left an undo step behind");
    assert_eq!(history.entries().len(), 0);
}

#[test]
fn a_cancelled_gesture_does_not_eat_the_previous_undo_step() {
    // The failure this design exists to prevent. If a gesture wrote a history
    // entry on the key press and removed it on cancel — or worse, adjusted the
    // entry already on top — then pressing M and pressing Escape would cost the
    // user the edit they made before it.
    let (mut doc, ctx) = edit_scene();
    let registry = registry();
    let mut history = History::new();

    // A real edit first.
    history
        .dispatch(&registry, &mut doc, &ctx, "mesh.translate", &moved(3.0))
        .expect("the edit before the gesture");
    let after_edit = positions(&doc);
    assert_eq!(history.entries().len(), 1);

    // A gesture, driven and then abandoned.
    let mut session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "mesh.translate",
            &moved(0.0),
        )
        .expect("dispatch")
        .expect("accepted");
    history
        .update_modal(&registry, &mut doc, None, &ctx, &mut session, &moved(99.0))
        .expect("update");
    History::cancel_modal(session, &mut doc);

    assert_eq!(
        positions(&doc),
        after_edit,
        "the earlier edit was disturbed"
    );
    assert_eq!(
        history.entries().len(),
        1,
        "the earlier entry is still there"
    );
    assert!(history.undo(&mut doc), "and it still undoes");
}

#[test]
fn a_committed_gesture_undoes_in_one_step() {
    let (mut doc, ctx) = edit_scene();
    let before = positions(&doc);
    let registry = registry();
    let mut history = History::new();
    let mut session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "mesh.translate",
            &moved(0.0),
        )
        .expect("dispatch")
        .expect("accepted");
    for step in 1..=50 {
        history
            .update_modal(
                &registry,
                &mut doc,
                None,
                &ctx,
                &mut session,
                &moved(step as f64 * 0.1),
            )
            .expect("update");
    }
    history.commit_modal(session);

    assert!(history.undo(&mut doc), "undo");
    assert_eq!(
        positions(&doc),
        before,
        "one undo did not undo the whole drag"
    );
    assert!(!history.can_undo(), "there was more than one step");
}

#[test]
fn the_committed_entry_replays_as_a_script() {
    // The claim that makes the mouse and the keyboard the same thing: a
    // confirmed drag records `(op_id, params)` and nothing else, so replaying
    // that line reproduces it exactly. Without this, macro recording and
    // scripting would need a separate path for anything done with a pointer.
    let (mut doc, ctx) = edit_scene();
    let registry = registry();
    let mut history = History::new();
    let mut session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "mesh.translate",
            &moved(0.0),
        )
        .expect("dispatch")
        .expect("accepted");
    history
        .update_modal(&registry, &mut doc, None, &ctx, &mut session, &moved(1.25))
        .expect("update");
    let recorded = session.params().clone();
    history.commit_modal(session);
    let dragged = positions(&doc);

    // The same parameters, dispatched as a script would.
    let (mut replayed_doc, ctx) = edit_scene();
    let mut replay = History::new();
    replay
        .dispatch(
            &registry,
            &mut replayed_doc,
            &ctx,
            "mesh.translate",
            &recorded,
        )
        .expect("replay");
    assert_eq!(positions(&replayed_doc), dragged);
}

#[test]
fn a_gesture_the_operator_declines_does_not_start() {
    // A single selected vertex cannot be scaled. `begin_modal` must say so
    // rather than hand back a session — a session whose operator never ran would
    // apply its updates to whatever the top of the history happened to be.
    let mut doc = Document::new();
    let id = doc.add_mesh(quad());
    let object = doc.add_object(Object::new("mesh", ObjectData::Mesh(id)));
    doc.selection.add(object);
    let mesh = doc.mesh_mut(id).expect("mesh");
    mesh.ensure_selection_layers();
    mesh.attributes_mut().insert(
        names::SELECT_VERT,
        Domain::Point,
        AttrData::Bool(vec![true, false, false, false].into()),
    );
    let ctx = Context {
        mode: Mode::EditMesh,
    };

    let registry = registry();
    let mut history = History::new();
    let mut params = PropertyMap::new();
    params.set("scale", PropertyValue::Vec3([1.0, 1.0, 1.0]));
    let session = history
        .begin_modal(&registry, &mut doc, None, &ctx, "mesh.scale", &params)
        .expect("dispatch");
    assert!(
        session.is_none(),
        "a single vertex should not start a scale"
    );
    assert_eq!(history.entries().len(), 0);
}

/// An operator that declines one particular value.
///
/// Registered only by this test, because no shipped operator declines on the
/// strength of its *parameters* — the mesh transforms decline on the shape of the
/// selection, which does not change mid-drag. A first version of the test below
/// used `mesh.rotate` at angle zero and believed it was exercising the declined
/// path; it was not, and removing the path entirely changed nothing.
///
/// The path is worth keeping even so. An operator that reports nothing-to-do for
/// some values is a perfectly reasonable operator — an inset with zero
/// thickness, a bevel with zero width — and a drag that passes through such a
/// value must not be aborted by it.
mod declining {
    use kreate_ops::operator::{Context, OpError, OpResult, Operator};
    use kreate_ops::OpParams;
    use kreate_scene::{DocMut, Document};

    #[derive(Clone, Debug, Default, PartialEq, OpParams)]
    pub struct Params {
        pub amount: f64,
    }

    pub struct DeclinesZero;

    impl Operator for DeclinesZero {
        type Params = Params;

        fn id() -> &'static str {
            "test.declines_zero"
        }
        fn label() -> &'static str {
            "Declines Zero"
        }
        fn poll(_ctx: &Context, _doc: &Document) -> bool {
            true
        }
        fn execute(
            params: &Self::Params,
            doc: &mut DocMut<'_>,
            _ctx: &Context,
        ) -> Result<OpResult, OpError> {
            // Mutates *first*, then decides — which is not perversity, it is
            // what the real mesh operators do. `with_bmesh` converts the mesh,
            // rebuilds every array and bakes the result before the operator can
            // tell whether there was anything to do.
            //
            // That ordering is the whole reason `update_modal` restores after a
            // declined value. A test operator that checked its parameter before
            // touching anything left nothing to restore, so removing the restore
            // changed nothing and the branch went untested.
            let d = doc.document_mut();
            let ids: Vec<_> = d.objects().map(|(id, _)| id).collect();
            for id in ids {
                if let Some(object) = d.object_mut(id) {
                    object.transform.translation.x = params.amount;
                }
            }
            if params.amount == 0.0 {
                return Ok(OpResult::Cancelled);
            }
            Ok(OpResult::Finished)
        }
    }
}

#[test]
fn a_value_the_operator_declines_mid_drag_does_not_end_the_gesture() {
    // Dragging through a value the operator reports as nothing-to-do. The
    // gesture has to survive it: ending there would abort someone's drag because
    // their pointer passed through a particular position.
    let (mut doc, ctx) = object_scene();
    let mut registry = registry();
    registry.register::<declining::DeclinesZero>();
    let mut history = History::new();
    let id = doc.selection.objects[0];

    let amount = |v: f64| {
        let mut p = PropertyMap::new();
        p.set("amount", PropertyValue::Float(v));
        p
    };
    let mut session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "test.declines_zero",
            &amount(1.0),
        )
        .expect("dispatch")
        .expect("accepted");

    // Straight through the declined value and out the other side.
    for value in [2.0, 0.0, -3.0] {
        history
            .update_modal(
                &registry,
                &mut doc,
                None,
                &ctx,
                &mut session,
                &amount(value),
            )
            .expect("the gesture should survive a declined value");
    }
    // The last value that worked is the one that stands.
    assert_eq!(
        doc.object(id).expect("object").transform.translation.x,
        -3.0
    );
    history.commit_modal(session);
    assert_eq!(history.entries().len(), 1);
}

#[test]
fn a_declined_value_leaves_the_document_at_the_original() {
    // While the pointer sits on a value the operator will not apply, the
    // selection has to be *somewhere* — and the only defensible somewhere is
    // where it started. Leaving the previous frame's result in place would make
    // the document disagree with the parameters the session is holding.
    let (mut doc, ctx) = object_scene();
    let mut registry = registry();
    registry.register::<declining::DeclinesZero>();
    let mut history = History::new();
    let id = doc.selection.objects[0];
    // A starting position that is not the declined value, or the two would be
    // indistinguishable and the assertion would hold either way.
    if let Some(object) = doc.object_mut(id) {
        object.transform.translation.x = 7.0;
    }
    let original = doc.object(id).expect("object").transform.translation.x;
    assert_eq!(original, 7.0);

    let amount = |v: f64| {
        let mut p = PropertyMap::new();
        p.set("amount", PropertyValue::Float(v));
        p
    };
    let mut session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "test.declines_zero",
            &amount(5.0),
        )
        .expect("dispatch")
        .expect("accepted");
    history
        .update_modal(&registry, &mut doc, None, &ctx, &mut session, &amount(0.0))
        .expect("update");
    assert_eq!(
        doc.object(id).expect("object").transform.translation.x,
        original,
        "a declined value left the previous frame's result behind"
    );
    History::cancel_modal(session, &mut doc);
}

#[test]
fn the_transaction_works_the_same_in_object_mode() {
    // The same machinery, a different operator. Object mode moves the object's
    // transform rather than the mesh's vertices, and the transaction knows
    // nothing about the difference.
    let (mut doc, ctx) = object_scene();
    let registry = registry();
    let mut history = History::new();
    let id = doc.selection.objects[0];
    let before = doc.object(id).expect("object").transform.translation;

    let mut session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "object.translate",
            &moved(0.0),
        )
        .expect("dispatch")
        .expect("accepted");
    history
        .update_modal(&registry, &mut doc, None, &ctx, &mut session, &moved(4.0))
        .expect("update");
    let during = doc.object(id).expect("object").transform.translation;
    assert!((during.x - before.x - 4.0).abs() < 1e-9, "{during:?}");

    History::cancel_modal(session, &mut doc);
    assert_eq!(
        doc.object(id).expect("object").transform.translation,
        before,
        "cancel did not restore the object"
    );
}

#[test]
fn a_gesture_discards_a_redo_branch_like_any_other_edit() {
    // Committing after an undo has to drop the redo branch, or redo would apply
    // an operator to a document that is no longer the one it saw.
    let (mut doc, ctx) = edit_scene();
    let registry = registry();
    let mut history = History::new();
    history
        .dispatch(&registry, &mut doc, &ctx, "mesh.translate", &moved(1.0))
        .expect("first");
    history
        .dispatch(&registry, &mut doc, &ctx, "mesh.translate", &moved(2.0))
        .expect("second");
    history.undo(&mut doc);
    assert!(history.can_redo());

    let mut session = history
        .begin_modal(
            &registry,
            &mut doc,
            None,
            &ctx,
            "mesh.translate",
            &moved(0.0),
        )
        .expect("dispatch")
        .expect("accepted");
    history
        .update_modal(&registry, &mut doc, None, &ctx, &mut session, &moved(9.0))
        .expect("update");
    history.commit_modal(session);

    assert!(
        !history.can_redo(),
        "the redo branch survived a committed gesture"
    );
    assert_eq!(history.entries().len(), 2);
}
