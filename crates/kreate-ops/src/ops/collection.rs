//! Collections: named groups of objects, Blender's `M`.
//!
//! One level under the scene, and an object in at most one — what `M`
//! (Move to Collection) gives, and what most scenes need: "the set", "the
//! lights", "the reference". A collection can be hidden, and everything in it
//! goes with it. Named by name in the parameters, so a script reads as the
//! outliner does.

use kreate_scene::document::Collection;
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<MoveToCollection>();
    registry.register::<SetCollectionVisible>();
    registry.register::<RemoveCollection>();
    registry.register::<RenameCollection>();
}

/// `name`, or `name.001` and so on when it is taken.
fn unique(doc: &Document, name: &str) -> String {
    let base = if name.trim().is_empty() { "Collection" } else { name.trim() };
    if doc.find_collection(base).is_none() {
        return base.to_string();
    }
    (1..)
        .map(|i| format!("{base}.{i:03}"))
        .find(|n| doc.find_collection(n).is_none())
        .expect("names run out after integers")
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MoveToCollectionParams {
    /// The collection, by name; empty for the scene itself.
    pub collection: String,
    /// Make a new collection — called `collection`, numbered if taken — and
    /// move into that.
    pub new: bool,
    /// One object, by identity in decimal, instead of the selection — the
    /// outliner's drag and drop. Empty for the selection.
    #[param(hidden)]
    pub uid: String,
}

/// Moves the selected objects into a collection — Blender's `M`.
pub struct MoveToCollection;

impl Operator for MoveToCollection {
    type Params = MoveToCollectionParams;
    fn id() -> &'static str {
        "object.move_to_collection"
    }
    fn label() -> &'static str {
        "Move to Collection"
    }
    fn poll(ctx: &Context, _doc: &Document) -> bool {
        ctx.mode.acts_on() == crate::mode::ActsOn::Objects
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let selected = if params.uid.is_empty() {
            doc.document().selection.objects.clone()
        } else {
            let Ok(raw) = params.uid.trim().parse::<u64>() else { return Ok(OpResult::Cancelled) };
            doc.document().object_by_uid(kreate_scene::Uid::from_raw(raw)).into_iter().collect()
        };
        if selected.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        let target = if params.new {
            let name = unique(doc.document(), &params.collection);
            Some(doc.document_mut().add_collection(Collection { name, ..Default::default() }))
        } else if params.collection.is_empty() {
            None
        } else {
            match doc.document().find_collection(&params.collection) {
                Some(id) => Some(id),
                None => return Ok(OpResult::Cancelled),
            }
        };
        let d = doc.document_mut();
        if !params.new && selected.iter().all(|o| d.collection_of(*o) == target) {
            return Ok(OpResult::Cancelled);
        }
        for object in selected {
            d.move_to_collection(object, target);
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetCollectionVisibleParams {
    pub name: String,
    pub visible: bool,
}

impl Default for SetCollectionVisibleParams {
    fn default() -> Self {
        Self { name: String::new(), visible: true }
    }
}

/// Shows or hides a collection, and everything in it — the outliner's eye.
pub struct SetCollectionVisible;

impl Operator for SetCollectionVisible {
    type Params = SetCollectionVisibleParams;
    fn id() -> &'static str {
        "collection.set_visible"
    }
    fn label() -> &'static str {
        "Show or Hide Collection"
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(id) = doc.document().find_collection(&params.name) else { return Ok(OpResult::Cancelled) };
        if doc.document().collection(id).is_some_and(|c| c.hide != params.visible) {
            return Ok(OpResult::Cancelled);
        }
        if let Some(c) = doc.document_mut().collection_mut(id) {
            c.hide = !params.visible;
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct CollectionNameParams {
    pub name: String,
}

/// Removes a collection; what was in it stays, in the scene itself.
pub struct RemoveCollection;

impl Operator for RemoveCollection {
    type Params = CollectionNameParams;
    fn id() -> &'static str {
        "collection.remove"
    }
    fn label() -> &'static str {
        "Remove Collection"
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(id) = doc.document().find_collection(&params.name) else { return Ok(OpResult::Cancelled) };
        doc.document_mut().remove_collection(id);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RenameCollectionParams {
    pub name: String,
    pub new_name: String,
}

/// Renames a collection, numbering the new name if another has it.
pub struct RenameCollection;

impl Operator for RenameCollection {
    type Params = RenameCollectionParams;
    fn id() -> &'static str {
        "collection.rename"
    }
    fn label() -> &'static str {
        "Rename Collection"
    }
    fn execute(params: &Self::Params, doc: &mut DocMut<'_>, _ctx: &Context) -> Result<OpResult, OpError> {
        let Some(id) = doc.document().find_collection(&params.name) else { return Ok(OpResult::Cancelled) };
        if params.new_name.trim().is_empty() || params.new_name == params.name {
            return Ok(OpResult::Cancelled);
        }
        let name = unique(doc.document(), &params.new_name);
        if let Some(c) = doc.document_mut().collection_mut(id) {
            c.name = name;
        }
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::History;
    use crate::operator::OperatorRegistry;
    use crate::{PropertyMap, PropertyValue};
    use kreate_scene::document::{Object, ObjectData};

    fn run(doc: &mut Document, op: &str, pairs: &[(&'static str, PropertyValue)]) -> bool {
        let mut r = OperatorRegistry::new();
        crate::ops::register_builtin(&mut r);
        let mut p = PropertyMap::new();
        for (k, v) in pairs {
            p.set(k, v.clone());
        }
        let mut h = History::new();
        h.dispatch(&r, doc, &Context::default(), op, &p).expect("ran");
        !h.entries().is_empty()
    }

    fn text(s: &str) -> PropertyValue {
        PropertyValue::Text(s.into())
    }

    #[test]
    fn m_moves_the_selection_into_a_new_collection_and_back_out() {
        let mut doc = Document::new();
        let a = doc.add_object(Object::new("A", ObjectData::Empty));
        let b = doc.add_object(Object::new("B", ObjectData::Empty));
        doc.selection.add(a);
        assert!(run(&mut doc, "object.move_to_collection", &[("collection", text("Props")), ("new", PropertyValue::Bool(true))]));
        let props = doc.find_collection("Props").expect("made");
        assert_eq!(doc.collection_of(a), Some(props));
        assert_eq!(doc.collection_of(b), None);
        // A second new one with the same name is numbered, and the object moves.
        assert!(run(&mut doc, "object.move_to_collection", &[("collection", text("Props")), ("new", PropertyValue::Bool(true))]));
        assert_eq!(doc.collection_of(a), doc.find_collection("Props.001"));
        assert!(doc.collection(props).unwrap().objects.is_empty(), "it is in two at once");
        // Back to the scene; again is nothing.
        assert!(run(&mut doc, "object.move_to_collection", &[("collection", text(""))]));
        assert_eq!(doc.collection_of(a), None);
        assert!(!run(&mut doc, "object.move_to_collection", &[("collection", text(""))]));
    }

    #[test]
    fn hiding_a_collection_hides_what_is_in_it() {
        let mut doc = Document::new();
        let a = doc.add_object(Object::new("A", ObjectData::Empty));
        doc.selection.add(a);
        run(&mut doc, "object.move_to_collection", &[("collection", text("Set")), ("new", PropertyValue::Bool(true))]);
        assert!(doc.is_shown(a));
        assert!(run(&mut doc, "collection.set_visible", &[("name", text("Set")), ("visible", PropertyValue::Bool(false))]));
        assert!(!doc.is_shown(a));
        assert!(run(&mut doc, "collection.rename", &[("name", text("Set")), ("new_name", text("Stage"))]));
        assert!(run(&mut doc, "collection.remove", &[("name", text("Stage"))]));
        assert!(doc.is_shown(a), "removing the collection left its object hidden");
        assert!(doc.object(a).is_some(), "removing the collection removed its object");
    }
}
