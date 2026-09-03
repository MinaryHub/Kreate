//! The modifier stack: an object's geometry as a function of its datablock.
//!
//! Three modifiers, applied in order, and that is deliberate. The point of this
//! module is not the features — mirror, subdivision and array are each a few
//! hundred lines in `kreate-mesh` — it is to force the separation between the
//! mesh a user *edits* and the mesh that gets *drawn* to exist while the
//! codebase is still small enough to change. Without it every operator, every
//! renderer path and every picking query quietly assumes "the mesh on screen is
//! the mesh in the document", and by the time a modifier makes that false there
//! are hundreds of call sites saying so.
//!
//! # The cache does not live in the document
//!
//! It would be natural to hang the evaluated mesh off `Document`. It is wrong:
//! `Document` is cloned for every undo snapshot and *restored* from one, so a
//! cache inside it travels backwards in time along with the geometry. The cache
//! therefore lives here, is owned by the application, and takes the document as
//! an argument.
//!
//! # Keyed on the input handle and the stack, not on the object
//!
//! An entry is looked up by the identity of its input `Arc` plus the stack that
//! was applied to it. Two linked duplicates with the same stack therefore share
//! one evaluated mesh — one evaluation, one GPU upload — exactly as two linked
//! duplicates with *no* stack share the datablock itself. Keying on `ObjectId`
//! would evaluate the same thing once per object and turn a thousand instances
//! back into a thousand meshes.
//!
//! The stack is compared for equality rather than hashed. A parameter hash is
//! smaller, but a collision shows the user the wrong geometry for no visible
//! reason, and a stack is a handful of numbers — there is nothing to save.

use std::sync::Arc;

use kreate_mesh::modifiers::{
    array, mirror, subdivide, ArrayParams, MirrorParams, SubdivideParams,
};
use kreate_mesh::MeshData;

use crate::document::{Document, ObjectId};

/// One modifier and its parameters.
///
/// An enum rather than a boxed trait: the set is closed, small, and needs to be
/// serialisable, comparable and copyable. A trait object would need each of
/// those bolted back on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModifierKind {
    Mirror(MirrorParams),
    Subdivide(SubdivideParams),
    Array(ArrayParams),
}

impl ModifierKind {
    /// The stable identifier, for keymaps, scripting and the file format.
    pub fn id(&self) -> &'static str {
        match self {
            ModifierKind::Mirror(_) => "mirror",
            ModifierKind::Subdivide(_) => "subdivide",
            ModifierKind::Array(_) => "array",
        }
    }

    /// The translation key for the user-visible name.
    ///
    /// Never the name itself: a bare string here is exactly what the CI gate
    /// forbids, and a modifier's name is on screen whenever its panel is.
    pub fn name_key(&self) -> &'static str {
        match self {
            ModifierKind::Mirror(_) => "modifier-mirror",
            ModifierKind::Subdivide(_) => "modifier-subdivide",
            ModifierKind::Array(_) => "modifier-array",
        }
    }

    fn apply(&self, input: &MeshData) -> MeshData {
        match self {
            ModifierKind::Mirror(p) => mirror(input, p),
            ModifierKind::Subdivide(p) => subdivide(input, p),
            ModifierKind::Array(p) => array(input, p),
        }
    }
}

/// A modifier in a stack: what it is, and whether it is on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModifierInstance {
    pub kind: ModifierKind,
    /// Disabled modifiers stay in the stack. Toggling one off to see the cage
    /// underneath is the commonest thing a user does with the panel, and it must
    /// not lose the parameters.
    pub enabled: bool,
}

impl ModifierInstance {
    pub fn new(kind: ModifierKind) -> Self {
        Self {
            kind,
            enabled: true,
        }
    }
}

/// Applies a stack to a mesh.
///
/// Linear and in order, which is not a simplification to be apologised for:
/// mirror-then-subdivide and subdivide-then-mirror are different results, and a
/// stack whose order did not matter would be hiding that from the user.
pub fn apply_stack(input: &MeshData, stack: &[ModifierInstance]) -> MeshData {
    let mut current: Option<MeshData> = None;
    for modifier in stack.iter().filter(|m| m.enabled) {
        let next = modifier.kind.apply(current.as_ref().unwrap_or(input));
        current = Some(next);
    }
    current.unwrap_or_else(|| input.clone())
}

/// Whether a stack would change its input at all.
///
/// An empty stack — or one with everything switched off — must return the input
/// handle itself rather than a copy of it, so that an object with no modifiers
/// is indistinguishable from one before modifiers existed. That is what keeps
/// linked duplicates sharing a single GPU mesh.
fn is_identity(stack: &[ModifierInstance]) -> bool {
    !stack.iter().any(|m| m.enabled)
}

struct Entry {
    input: Arc<MeshData>,
    stack: Vec<ModifierInstance>,
    output: Arc<MeshData>,
    /// The sweep generation this entry was last asked for.
    used: u64,
}

/// Evaluated meshes, kept between frames.
///
/// Owned by the application, not by the document — see the module docs.
#[derive(Default)]
pub struct ModifierCache {
    entries: Vec<Entry>,
    generation: u64,
    /// Counters, so "did it re-evaluate more than it had to" is answerable. It
    /// is invisible at a glance and shows up only as a stutter on a scene big
    /// enough that it is already too late to notice.
    hits: u64,
    misses: u64,
}

impl ModifierCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The evaluated mesh for one object, evaluating it if necessary.
    ///
    /// `None` when the object has no mesh, or references one that is not in the
    /// document.
    pub fn evaluated(&mut self, doc: &Document, id: ObjectId) -> Option<Arc<MeshData>> {
        let object = doc.object(id)?;
        let input = doc.mesh_arc(object.mesh()?)?;
        Some(self.evaluate(input, &object.modifiers))
    }

    /// The evaluated result of applying `stack` to `input`.
    pub fn evaluate(&mut self, input: &Arc<MeshData>, stack: &[ModifierInstance]) -> Arc<MeshData> {
        if is_identity(stack) {
            return input.clone();
        }
        // The whole invalidation mechanism, in one comparison. It is exact and
        // free, and it works only because the entry *holds* the input handle:
        // `Arc::make_mut` copies on write only while a handle is shared, so an
        // entry storing an id would watch the document mutate in place behind an
        // unchanged pointer and go on serving stale geometry.
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| Arc::ptr_eq(&e.input, input) && e.stack == stack)
        {
            entry.used = self.generation;
            self.hits += 1;
            return entry.output.clone();
        }

        let output = Arc::new(apply_stack(input, stack));
        self.misses += 1;
        self.entries.push(Entry {
            input: input.clone(),
            stack: stack.to_vec(),
            output: output.clone(),
            used: self.generation,
        });
        output
    }

    /// Every visible mesh object's evaluated geometry, in document order.
    ///
    /// The renderer's and the picker's view of the scene: neither of them sees a
    /// `MeshId` any more, which is the separation this module exists to create.
    pub fn evaluate_scene(&mut self, doc: &Document) -> Vec<(ObjectId, Arc<MeshData>)> {
        self.evaluate_scene_with_cage(doc, None)
    }

    /// The same, with one object left unevaluated.
    ///
    /// Edit mode shows and edits the **cage**: the mesh as it is in the document,
    /// not the result of its modifier stack. That is not a simplification — it is
    /// what makes element selection mean anything. An element index taken from
    /// the id buffer is an index into whatever was drawn, so drawing a mirrored
    /// copy would have a click on the reflection name a face on the other side,
    /// and a click on a subdivided surface name a face that does not exist in the
    /// document at all.
    ///
    /// Everything else in the scene still shows its modifier result, so the
    /// object being edited stays in context.
    pub fn evaluate_scene_with_cage(
        &mut self,
        doc: &Document,
        cage: Option<ObjectId>,
    ) -> Vec<(ObjectId, Arc<MeshData>)> {
        let ids: Vec<ObjectId> = doc
            .objects()
            .filter(|(_, o)| o.visible && o.mesh().is_some())
            .map(|(id, _)| id)
            .collect();
        ids.into_iter()
            .filter_map(|id| {
                if Some(id) == cage {
                    let mesh = doc.object(id)?.mesh()?;
                    return doc.mesh_arc(mesh).map(|m| (id, m.clone()));
                }
                self.evaluated(doc, id).map(|m| (id, m))
            })
            .collect()
    }

    /// Drops entries that were not asked for since the last sweep.
    ///
    /// Called once a frame, after evaluation. Without it the cache keeps one
    /// entry per parameter value the user has ever dragged through, which on a
    /// subdivision modifier is a copy of the mesh per frame of the drag.
    pub fn sweep(&mut self) {
        let generation = self.generation;
        self.entries.retain(|e| e.used == generation);
        self.generation += 1;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hits and misses since the cache was created.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Roughly how much the evaluated meshes occupy.
    ///
    /// Only the outputs: the inputs are the document's and are counted there.
    pub fn memory_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.output.memory_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Object, ObjectData};
    use kreate_mesh::primitives;

    fn doc_with_cube() -> (Document, ObjectId) {
        let mut doc = Document::new();
        let id = doc.add_mesh_object("Cube", primitives::cube(2.0));
        (doc, id)
    }

    fn mirror_x() -> ModifierInstance {
        ModifierInstance::new(ModifierKind::Mirror(MirrorParams::default()))
    }

    #[test]
    fn an_object_with_no_modifiers_evaluates_to_its_own_datablock() {
        // Not merely to an equal mesh: to the *same handle*. An object with no
        // modifiers has to be indistinguishable from one before modifiers
        // existed, or every linked duplicate becomes its own GPU upload.
        let (doc, id) = doc_with_cube();
        let mut cache = ModifierCache::new();
        let evaluated = cache.evaluated(&doc, id).unwrap();
        let datablock = doc
            .mesh_arc(doc.object(id).unwrap().mesh().unwrap())
            .unwrap();
        assert!(Arc::ptr_eq(&evaluated, datablock));
        assert!(cache.is_empty(), "the identity stack needs no entry");
    }

    #[test]
    fn a_stack_changes_the_evaluated_mesh_and_leaves_the_datablock_alone() {
        // The separation, stated as a test: the document still holds the cage the
        // user edits, and the thing on screen is something else.
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Subdivide(
                SubdivideParams {
                    levels: 1,
                    ..Default::default()
                },
            )));

        let mut cache = ModifierCache::new();
        let evaluated = cache.evaluated(&doc, id).unwrap();
        assert_eq!(evaluated.num_faces(), 24);
        let mesh_id = doc.object(id).unwrap().mesh().unwrap();
        assert_eq!(doc.mesh(mesh_id).unwrap().num_faces(), 6, "the cage moved");
    }

    #[test]
    fn a_disabled_modifier_keeps_its_parameters_and_does_nothing() {
        let (mut doc, id) = doc_with_cube();
        let mut instance = mirror_x();
        instance.enabled = false;
        doc.object_mut(id).unwrap().modifiers.push(instance);

        let mut cache = ModifierCache::new();
        let evaluated = cache.evaluated(&doc, id).unwrap();
        assert_eq!(evaluated.num_faces(), 6);
        assert_eq!(
            doc.object(id).unwrap().modifiers[0].kind,
            ModifierKind::Mirror(MirrorParams::default()),
            "toggling a modifier off must not lose what it was set to"
        );
    }

    #[test]
    fn the_order_of_the_stack_matters() {
        // Arraying a mirror and mirroring an array are different meshes: the
        // array's offset is a fraction of the bounding box, and mirroring first
        // doubles that box. A stack whose order did not matter would be hiding
        // that from the user.
        //
        // Compared by bounds rather than by face count, because the counts agree
        // — both orders make four copies of the input — which is exactly why a
        // count is not enough to tell these two apart.
        let cube = primitives::cube(2.0);
        let array = ModifierInstance::new(ModifierKind::Array(ArrayParams::default()));
        let mirror = ModifierInstance::new(ModifierKind::Mirror(MirrorParams::default()));
        let a = apply_stack(&cube, &[mirror, array]);
        let b = apply_stack(&cube, &[array, mirror]);

        assert_eq!(a.num_faces(), b.num_faces());
        assert_ne!(
            a.bounds().min.x,
            b.bounds().min.x,
            "the order made no difference: {:?} against {:?}",
            a.bounds(),
            b.bounds()
        );
    }

    #[test]
    fn an_unchanged_document_is_a_cache_hit() {
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id).unwrap().modifiers.push(mirror_x());

        let mut cache = ModifierCache::new();
        let first = cache.evaluated(&doc, id).unwrap();
        let second = cache.evaluated(&doc, id).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.stats(), (1, 1));
    }

    #[test]
    fn editing_the_input_mesh_invalidates_the_entry() {
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id).unwrap().modifiers.push(mirror_x());
        let mut cache = ModifierCache::new();
        let before = cache.evaluated(&doc, id).unwrap();

        let mesh_id = doc.object(id).unwrap().mesh().unwrap();
        doc.mesh_mut(mesh_id).unwrap().positions_mut()[0].y += 1.0;

        let after = cache.evaluated(&doc, id).unwrap();
        assert!(!Arc::ptr_eq(&before, &after), "stale geometry survived");
        assert_eq!(cache.stats().1, 2, "the second call re-evaluated");
    }

    #[test]
    fn changing_a_parameter_invalidates_the_entry() {
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Array(ArrayParams {
                count: 2,
                ..Default::default()
            })));
        let mut cache = ModifierCache::new();
        let before = cache.evaluated(&doc, id).unwrap();

        doc.object_mut(id).unwrap().modifiers[0].kind = ModifierKind::Array(ArrayParams {
            count: 3,
            ..Default::default()
        });
        let after = cache.evaluated(&doc, id).unwrap();
        assert_eq!(before.num_verts(), 16);
        assert_eq!(after.num_verts(), 24);
    }

    #[test]
    fn two_linked_duplicates_with_the_same_stack_share_one_evaluated_mesh() {
        // The reason for keying on the input handle rather than on the object:
        // per-object keying evaluates the same geometry once per instance and
        // turns a thousand linked duplicates back into a thousand meshes.
        let (mut doc, first) = doc_with_cube();
        doc.object_mut(first).unwrap().modifiers.push(mirror_x());
        let second = doc.duplicate_linked(first).unwrap();

        let mut cache = ModifierCache::new();
        let a = cache.evaluated(&doc, first).unwrap();
        let b = cache.evaluated(&doc, second).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(cache.len(), 1, "one entry for both instances");
    }

    #[test]
    fn two_linked_duplicates_with_different_stacks_do_not() {
        let (mut doc, first) = doc_with_cube();
        doc.object_mut(first).unwrap().modifiers.push(mirror_x());
        let second = doc.duplicate_linked(first).unwrap();
        doc.object_mut(second).unwrap().modifiers[0].kind = ModifierKind::Mirror(MirrorParams {
            x: false,
            y: true,
            ..Default::default()
        });

        let mut cache = ModifierCache::new();
        let a = cache.evaluated(&doc, first).unwrap();
        let b = cache.evaluated(&doc, second).unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn the_sweep_drops_what_nobody_asked_for() {
        // Otherwise the cache keeps one entry per parameter value the user has
        // ever dragged through — on a subdivision modifier, a copy of the mesh
        // for every frame of the drag.
        let (mut doc, id) = doc_with_cube();
        let mut cache = ModifierCache::new();
        for count in 2..8u32 {
            doc.object_mut(id).unwrap().modifiers =
                vec![ModifierInstance::new(ModifierKind::Array(ArrayParams {
                    count,
                    ..Default::default()
                }))];
            cache.evaluated(&doc, id).unwrap();
        }
        assert_eq!(cache.len(), 6, "every parameter value left an entry");

        cache.sweep();
        cache.evaluated(&doc, id).unwrap();
        cache.sweep();
        assert_eq!(cache.len(), 1, "only the live one survives");
    }

    #[test]
    fn a_swept_entry_is_re_evaluated_rather_than_lost() {
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id).unwrap().modifiers.push(mirror_x());
        let mut cache = ModifierCache::new();
        let before = cache.evaluated(&doc, id).unwrap();
        cache.sweep();
        cache.sweep();
        assert!(cache.is_empty());

        let after = cache.evaluated(&doc, id).unwrap();
        assert_eq!(before.num_faces(), after.num_faces());
    }

    #[test]
    fn evaluating_the_scene_skips_hidden_objects() {
        let (mut doc, id) = doc_with_cube();
        let hidden = doc.duplicate_linked(id).unwrap();
        doc.object_mut(hidden).unwrap().visible = false;
        doc.add_object(Object::new("Empty", ObjectData::Empty));

        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc);
        assert_eq!(scene.len(), 1);
        assert_eq!(scene[0].0, id);
    }

    #[test]
    fn the_edited_object_shows_its_cage_while_the_rest_show_their_result() {
        // What makes element selection mean anything: an element index from the
        // id buffer indexes whatever was drawn, so a click on a mirrored copy
        // would otherwise name a face on the other side of the model.
        let (mut doc, edited) = doc_with_cube();
        doc.object_mut(edited)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Subdivide(
                SubdivideParams {
                    levels: 1,
                    ..Default::default()
                },
            )));
        let other = doc.add_mesh_object("Other", primitives::cube(2.0));
        doc.object_mut(other)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Subdivide(
                SubdivideParams {
                    levels: 1,
                    ..Default::default()
                },
            )));

        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene_with_cage(&doc, Some(edited));

        let of = |id: ObjectId| {
            scene
                .iter()
                .find(|(o, _)| *o == id)
                .map(|(_, m)| m.num_faces())
                .unwrap()
        };
        assert_eq!(of(edited), 6, "the edited object shows its cage");
        assert_eq!(of(other), 24, "everything else shows its result");
    }

    #[test]
    fn the_cage_is_the_datablock_handle_itself() {
        // Not a copy of it: the GPU cache and the picker key on handle identity,
        // and a fresh clone every frame would re-upload the mesh on every frame
        // of edit mode.
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id).unwrap().modifiers.push(mirror_x());
        let mut cache = ModifierCache::new();

        let scene = cache.evaluate_scene_with_cage(&doc, Some(id));
        let datablock = doc
            .mesh_arc(doc.object(id).unwrap().mesh().unwrap())
            .unwrap();
        assert!(Arc::ptr_eq(&scene[0].1, datablock));

        let again = cache.evaluate_scene_with_cage(&doc, Some(id));
        assert!(
            Arc::ptr_eq(&scene[0].1, &again[0].1),
            "a new handle every frame"
        );
    }

    #[test]
    fn an_object_with_no_mesh_evaluates_to_nothing() {
        let mut doc = Document::new();
        let id = doc.add_object(Object::new("Empty", ObjectData::Empty));
        let mut cache = ModifierCache::new();
        assert!(cache.evaluated(&doc, id).is_none());
    }
}
