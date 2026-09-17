//! Hiding an object is an edit, and it is addressed by identity.
//!
//! The outliner's eye toggles a row that may not be selected, so the operator
//! cannot act on the selection the way delete does. It names the object by its
//! uid — the thing a file and a script agree on — rather than by an arena slot,
//! which is reused the moment an object is deleted.

use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

fn wired() -> (OperatorRegistry, History, Document, Context) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    (registry, History::new(), Document::new(), Context::default())
}

fn cube(doc: &mut Document, name: &str) -> kreate_scene::ObjectId {
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
    doc.add_object(Object::new(name, ObjectData::Mesh(mesh)))
}

fn set_visible(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    ctx: &Context,
    uid: &str,
    visible: bool,
) -> bool {
    let mut params = PropertyMap::new();
    params.set("uid", PropertyValue::Text(uid.to_string()));
    params.set("visible", PropertyValue::Bool(visible));
    history
        .dispatch(registry, doc, ctx, "object.set_visible", &params)
        .is_ok()
}

#[test]
fn an_object_can_be_hidden_and_shown_by_name() {
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    let b = cube(&mut doc, "B");
    let uid = doc.object_uid(a).expect("an identity").raw().to_string();

    assert!(set_visible(&registry, &mut history, &mut doc, &ctx, &uid, false));
    assert!(!doc.object(a).unwrap().visible, "A is still visible");
    assert!(doc.object(b).unwrap().visible, "B was hidden too");

    assert!(set_visible(&registry, &mut history, &mut doc, &ctx, &uid, true));
    assert!(doc.object(a).unwrap().visible);
}

#[test]
fn hiding_does_not_need_the_object_to_be_selected() {
    // The eye is on the row, not on the selection: a user hides the thing in
    // the way, which is by definition not the thing they are working on.
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    let b = cube(&mut doc, "B");
    doc.selection.add(b);
    let uid = doc.object_uid(a).expect("an identity").raw().to_string();
    assert!(set_visible(&registry, &mut history, &mut doc, &ctx, &uid, false));
    assert!(!doc.object(a).unwrap().visible);
    assert!(doc.selection.contains(b), "the selection was disturbed");
}

#[test]
fn hiding_is_one_undo_step() {
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    let uid = doc.object_uid(a).expect("an identity").raw().to_string();
    assert!(set_visible(&registry, &mut history, &mut doc, &ctx, &uid, false));
    assert!(history.undo(&mut doc));
    assert!(doc.object(a).unwrap().visible, "undo did not show it again");
}

#[test]
fn a_uid_naming_nothing_is_refused() {
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    let depth = history.entries().len();
    set_visible(&registry, &mut history, &mut doc, &ctx, "424242", false);
    set_visible(&registry, &mut history, &mut doc, &ctx, "not a number", false);
    assert!(doc.object(a).unwrap().visible);
    assert_eq!(history.entries().len(), depth, "a refusal left a step to undo");
}

#[test]
fn setting_what_is_already_so_is_not_an_edit() {
    // Clicking the eye of a visible object to show it should not make an undo
    // step that does nothing: the user pressing undo afterwards expects their
    // last *change* to come back, not a no-op to be replayed.
    let (registry, mut history, mut doc, ctx) = wired();
    let a = cube(&mut doc, "A");
    let uid = doc.object_uid(a).expect("an identity").raw().to_string();
    let depth = history.entries().len();
    set_visible(&registry, &mut history, &mut doc, &ctx, &uid, true);
    assert_eq!(history.entries().len(), depth, "a no-op made an undo step");
}
