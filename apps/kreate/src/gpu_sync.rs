//! Deciding what the GPU needs re-uploaded.
//!
//! The naive rule — "the document version changed, rebuild everything" — is
//! correct and ruinous. Coarse invalidation is the leading cause of hitches in an
//! application of this kind: adding one object re-uploads every other, so the
//! cost of an edit grows with the size of the scene rather than with the size of
//! the edit.
//!
//! The right test is [`Arc::ptr_eq`] on the mesh handle. It is free, exact, and
//! needs no invalidation discipline to forget — but only if the cache **holds a
//! handle**. `Arc::make_mut` copies only when the handle is shared, so a cache
//! storing just an id would watch the document mutate in place behind an
//! unchanged pointer and never notice. Holding the handle is also what makes the
//! renderer's view genuinely immutable rather than aspirationally so.
//!
//! # Keyed on the evaluated mesh, not on the datablock
//!
//! Once objects have modifier stacks, the geometry to draw is no longer the
//! datablock: two objects sharing a mesh can have different stacks, so a cache
//! keyed on `MeshId` would hand one object the other's geometry. Keying on the
//! evaluated handle's identity is the fix, and it keeps instancing intact for
//! free — an object with no modifiers evaluates to the datablock itself, so a
//! thousand linked duplicates still share one upload, and so do a thousand that
//! happen to share a stack.
//!
//! It also removes a state. "Rebuild" existed because a `MeshId` outlived its
//! contents; an evaluated handle does not, so a changed mesh is simply a handle
//! the cache has never seen.
//!
//! # A selection change is not a new mesh
//!
//! One state is added, though, and it is the one that matters for interaction.
//! Selection is stored as ordinary attribute layers — which is what makes it
//! survive the round trip through edit mode and share structurally under undo —
//! so a click produces a new `MeshData` and a cache that could only ask "is this
//! the same handle" would re-upload the geometry of a million vertices because
//! one face was clicked. That is the classic mistake in this part of a modelling
//! tool: it makes rubber-band selection a slideshow, and it is invisible until
//! the mesh is big enough that it is too late to restructure.
//!
//! So a handle the cache has not seen is checked against the one it holds with
//! [`MeshData::only_selection_differs`], and when that is all that changed the
//! plan says to rewrite three small flag buffers instead.

use std::sync::Arc;

use kreate_mesh::MeshData;
use kreate_scene::document::ObjectId;

/// What the cache should look like after this frame.
#[derive(Clone, Debug, Default)]
pub struct SyncPlan {
    /// Cached slots that are still wanted, in the order they should end up in.
    /// Any slot not named here is dropped.
    pub keep: Vec<usize>,
    /// Slots whose mesh differs only in its selection, with the handle to adopt.
    ///
    /// The slot keeps its GPU buffers and has its flags rewritten. These slots
    /// also appear in `keep`, because that is what says they survive.
    pub reflags: Vec<(usize, Arc<MeshData>)>,
    /// Meshes that are not on the GPU yet.
    pub uploads: Vec<Arc<MeshData>>,
}

impl SyncPlan {
    /// Whether anything has to happen at all.
    ///
    /// The overwhelmingly common case is a scene that has not changed, and it
    /// should cost one comparison per object and no GPU work whatever.
    pub fn needs_work(&self, cached: usize) -> bool {
        !self.uploads.is_empty() || !self.reflags.is_empty() || self.keep.len() != cached
    }

    /// How many meshes the cache will hold once the plan is applied.
    pub fn total(&self) -> usize {
        self.keep.len() + self.uploads.len()
    }
}

/// Works out what changed, given what the cache is currently holding.
///
/// `scene` is the evaluated geometry of the visible objects; `cached` is the
/// handles the GPU cache holds, in slot order.
///
/// Pure, so the decision can be tested without a device — which matters,
/// because "did it upload more than it had to" is invisible at a glance and only
/// shows up as a stutter on a scene large enough that it is too late to notice.
pub fn plan_sync(cached: &[Arc<MeshData>], scene: &[(ObjectId, Arc<MeshData>)]) -> SyncPlan {
    let mut plan = SyncPlan::default();
    // Deduplicated by handle identity, so instances of one mesh — linked
    // duplicates, or objects that happen to share a modifier stack — are one
    // entry rather than one each.
    let mut wanted: Vec<&Arc<MeshData>> = Vec::new();

    for (_, mesh) in scene {
        if wanted.iter().any(|m| Arc::ptr_eq(m, mesh)) {
            continue;
        }
        wanted.push(mesh);

        // The whole mechanism, in one comparison.
        if let Some(slot) = cached.iter().position(|held| Arc::ptr_eq(held, mesh)) {
            plan.keep.push(slot);
            continue;
        }

        // A handle the cache has not seen, but perhaps not new geometry. Only
        // slots not already claimed this frame are candidates: an instance whose
        // stack diverged is a genuinely different mesh that happens to share most
        // of its arrays, and adopting it into the original's slot would leave
        // both objects drawing the same thing.
        let selection_only = cached
            .iter()
            .enumerate()
            .find(|(slot, held)| !plan.keep.contains(slot) && held.only_selection_differs(mesh));
        match selection_only {
            Some((slot, _)) => {
                plan.keep.push(slot);
                plan.reflags.push((slot, mesh.clone()));
            }
            None => plan.uploads.push(mesh.clone()),
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::document::{Object, ObjectData};
    use kreate_scene::modifier::{ModifierInstance, ModifierKind};
    use kreate_scene::{Document, ModifierCache};

    fn doc_with(count: usize) -> Document {
        let mut doc = Document::new();
        for i in 0..count {
            let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0 + i as f32));
            doc.add_object(Object::new(format!("Cube{i}"), ObjectData::Mesh(mesh)));
        }
        doc
    }

    /// What a cache would be holding after applying a plan.
    fn apply(cached: &[Arc<MeshData>], plan: &SyncPlan) -> Vec<Arc<MeshData>> {
        let mut out: Vec<Arc<MeshData>> = plan.keep.iter().map(|i| cached[*i].clone()).collect();
        out.extend(plan.uploads.iter().cloned());
        out
    }

    #[test]
    fn an_empty_cache_uploads_everything() {
        let doc = doc_with(3);
        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc);
        let plan = plan_sync(&[], &scene);
        assert_eq!(plan.uploads.len(), 3);
        assert!(plan.keep.is_empty());
        assert!(plan.needs_work(0));
    }

    #[test]
    fn an_unchanged_scene_uploads_nothing() {
        // The case that matters: it is what happens on almost every frame, and
        // it must cost one comparison per object and no GPU work at all.
        let doc = doc_with(3);
        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc);
        let held = apply(&[], &plan_sync(&[], &scene));

        let plan = plan_sync(&held, &scene);
        assert!(plan.uploads.is_empty());
        assert_eq!(plan.keep, vec![0, 1, 2]);
        assert!(!plan.needs_work(held.len()));
    }

    #[test]
    fn editing_one_mesh_uploads_only_that_one() {
        // The reason for handle identity rather than a document version: with a
        // version, editing one object re-uploads every other, and the cost of an
        // edit grows with the size of the scene.
        let mut doc = doc_with(3);
        let mut cache = ModifierCache::new();
        let held = apply(&[], &plan_sync(&[], &cache.evaluate_scene(&doc)));

        let target = doc.objects().next().unwrap().1.mesh().unwrap();
        doc.mesh_mut(target).unwrap().positions_mut()[0].y += 1.0;

        let plan = plan_sync(&held, &cache.evaluate_scene(&doc));
        assert_eq!(plan.uploads.len(), 1, "only the edited mesh");
        assert_eq!(plan.keep.len(), 2);
    }

    #[test]
    fn a_removed_object_drops_its_mesh() {
        let mut doc = doc_with(3);
        let mut cache = ModifierCache::new();
        let held = apply(&[], &plan_sync(&[], &cache.evaluate_scene(&doc)));

        let victim = doc.objects().next().unwrap().0;
        doc.remove_object(victim);

        let plan = plan_sync(&held, &cache.evaluate_scene(&doc));
        assert_eq!(plan.total(), 2);
        assert!(plan.uploads.is_empty());
        assert!(plan.needs_work(held.len()), "a slot has to be released");
    }

    #[test]
    fn a_hidden_object_drops_its_mesh() {
        let mut doc = doc_with(2);
        let mut cache = ModifierCache::new();
        let held = apply(&[], &plan_sync(&[], &cache.evaluate_scene(&doc)));

        let id = doc.objects().next().unwrap().0;
        doc.object_mut(id).unwrap().visible = false;

        let plan = plan_sync(&held, &cache.evaluate_scene(&doc));
        assert_eq!(plan.total(), 1);
    }

    #[test]
    fn linked_duplicates_share_one_upload() {
        // A thousand instances cost a thousand transforms, not a thousand
        // meshes. This is the payoff for datablock indirection and it has to
        // survive the move to evaluated geometry.
        let mut doc = doc_with(1);
        let original = doc.objects().next().unwrap().0;
        for _ in 0..9 {
            doc.duplicate_linked(original).unwrap();
        }
        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc);
        assert_eq!(scene.len(), 10, "ten objects");

        let plan = plan_sync(&[], &scene);
        assert_eq!(plan.uploads.len(), 1, "one mesh");
    }

    #[test]
    fn instances_sharing_a_modifier_stack_also_share_one_upload() {
        let mut doc = doc_with(1);
        let original = doc.objects().next().unwrap().0;
        doc.object_mut(original)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Mirror(
                Default::default(),
            )));
        doc.duplicate_linked(original).unwrap();

        let mut cache = ModifierCache::new();
        let plan = plan_sync(&[], &cache.evaluate_scene(&doc));
        assert_eq!(plan.uploads.len(), 1);
    }

    #[test]
    fn instances_with_different_stacks_do_not() {
        // And this is why the cache cannot be keyed on `MeshId`: these two
        // objects share a datablock and must not share geometry.
        let mut doc = doc_with(1);
        let first = doc.objects().next().unwrap().0;
        doc.object_mut(first)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Mirror(
                Default::default(),
            )));
        let second = doc.duplicate_linked(first).unwrap();
        doc.object_mut(second).unwrap().modifiers.clear();

        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc);
        let plan = plan_sync(&[], &scene);
        assert_eq!(plan.uploads.len(), 2, "one mirrored, one not");
    }

    #[test]
    fn a_modifier_parameter_change_uploads_the_new_geometry() {
        let mut doc = doc_with(1);
        let id = doc.objects().next().unwrap().0;
        doc.object_mut(id)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Array(
                kreate_mesh::modifiers::ArrayParams {
                    count: 2,
                    ..Default::default()
                },
            )));
        let mut cache = ModifierCache::new();
        let held = apply(&[], &plan_sync(&[], &cache.evaluate_scene(&doc)));

        doc.object_mut(id).unwrap().modifiers[0].kind =
            ModifierKind::Array(kreate_mesh::modifiers::ArrayParams {
                count: 5,
                ..Default::default()
            });

        let plan = plan_sync(&held, &cache.evaluate_scene(&doc));
        assert_eq!(plan.uploads.len(), 1);
        assert!(plan.keep.is_empty());
        assert_eq!(plan.uploads[0].num_verts(), 8 * 5);
    }

    #[test]
    fn a_selection_change_rewrites_flags_instead_of_uploading() {
        // The load-bearing case for interaction. Selection lives in attribute
        // layers, so a click produces a new `MeshData` — and re-uploading the
        // geometry of a large mesh on every click is what makes rubber-band
        // selection a slideshow.
        let mut doc = doc_with(1);
        let mut cache = ModifierCache::new();
        let held = apply(&[], &plan_sync(&[], &cache.evaluate_scene(&doc)));

        let target = doc.objects().next().unwrap().1.mesh().unwrap();
        {
            let mesh = doc.mesh_mut(target).unwrap();
            let faces = mesh.num_faces() as usize;
            mesh.attributes_mut().insert(
                kreate_mesh::attribute::names::SELECT_FACE,
                kreate_mesh::attribute::Domain::Face,
                kreate_mesh::attribute::AttrData::Bool(Arc::new(vec![true; faces])),
            );
        }

        let plan = plan_sync(&held, &cache.evaluate_scene(&doc));
        assert!(
            plan.uploads.is_empty(),
            "a click re-uploaded the geometry: {} uploads",
            plan.uploads.len()
        );
        assert_eq!(plan.reflags.len(), 1, "the flags should be rewritten");
        assert_eq!(plan.reflags[0].0, 0, "in the slot it already occupies");
        assert_eq!(plan.keep, vec![0], "and the slot survives");
        assert!(plan.needs_work(held.len()), "there is still work to do");
    }

    #[test]
    fn a_click_in_edit_mode_reaches_the_flags_only_path() {
        // The end-to-end version of the claim, through the operator the viewport
        // actually dispatches. The pieces were each correct in isolation and the
        // composition was not: selection went through `BMesh`, which rebuilt
        // every array, so the flags-only path never fired for the one operation
        // a user performs constantly.
        use kreate_ops::operator::{Context, Mode};
        use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};

        let mut doc = doc_with(1);
        let object = doc.objects().next().unwrap().0;
        doc.selection.add(object);

        let mut cache = ModifierCache::new();
        let held = apply(&[], &plan_sync(&[], &cache.evaluate_scene(&doc)));

        let mut registry = OperatorRegistry::new();
        kreate_ops::register_builtin(&mut registry);
        let mut history = History::new();
        let mut params = PropertyMap::new();
        params.set("element", PropertyValue::Int(2));
        params.set("mode", PropertyValue::Enum(2));
        params.set("extend", PropertyValue::Bool(false));
        history
            .dispatch(
                &registry,
                &mut doc,
                &Context {
                    mode: Mode::EditMesh,
                },
                "mesh.select_pick",
                &params,
            )
            .expect("mesh.select_pick");

        let plan = plan_sync(&held, &cache.evaluate_scene(&doc));
        assert!(
            plan.uploads.is_empty(),
            "a click re-uploaded the geometry: {} uploads",
            plan.uploads.len()
        );
        assert_eq!(plan.reflags.len(), 1, "the flags should be rewritten");
    }

    #[test]
    fn a_geometry_change_still_uploads() {
        // The companion: the shortcut must not swallow a real edit, or the screen
        // shows the old shape until something else forces a rebuild.
        let mut doc = doc_with(1);
        let mut cache = ModifierCache::new();
        let held = apply(&[], &plan_sync(&[], &cache.evaluate_scene(&doc)));

        let target = doc.objects().next().unwrap().1.mesh().unwrap();
        doc.mesh_mut(target).unwrap().positions_mut()[0].y += 1.0;

        let plan = plan_sync(&held, &cache.evaluate_scene(&doc));
        assert_eq!(plan.uploads.len(), 1);
        assert!(plan.reflags.is_empty());
    }

    #[test]
    fn an_instance_whose_stack_diverged_gets_its_own_upload() {
        // Two objects sharing a datablock, one of them modified. The evaluated
        // meshes share most of their arrays, so a careless "differs only by
        // selection" check would adopt one into the other's slot and both
        // objects would draw the same thing.
        let mut doc = doc_with(1);
        let first = doc.objects().next().unwrap().0;
        let second = doc.duplicate_linked(first).unwrap();
        let mut cache = ModifierCache::new();
        let held = apply(&[], &plan_sync(&[], &cache.evaluate_scene(&doc)));
        assert_eq!(held.len(), 1, "they share one mesh to begin with");

        doc.object_mut(second)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Array(
                kreate_mesh::modifiers::ArrayParams::default(),
            )));

        let plan = plan_sync(&held, &cache.evaluate_scene(&doc));
        assert_eq!(plan.keep, vec![0], "the original keeps its slot");
        assert_eq!(plan.uploads.len(), 1, "the arrayed copy is new geometry");
        assert!(plan.reflags.is_empty());
    }

    #[test]
    fn moving_an_object_uploads_nothing() {
        // A transform is not geometry. Uploading on a move would make orbiting a
        // parented rig re-upload the scene.
        let mut doc = doc_with(2);
        let mut cache = ModifierCache::new();
        let held = apply(&[], &plan_sync(&[], &cache.evaluate_scene(&doc)));

        let id = doc.objects().next().unwrap().0;
        doc.object_mut(id).unwrap().transform.translation = kreate_math::DVec3::Y;

        let plan = plan_sync(&held, &cache.evaluate_scene(&doc));
        assert!(!plan.needs_work(held.len()));
    }

    #[test]
    fn the_plan_is_in_document_order() {
        // So that a run is reproducible, and so that a palette assigned by slot
        // does not reshuffle every frame.
        let doc = doc_with(4);
        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc);
        let held = apply(&[], &plan_sync(&[], &scene));
        let plan = plan_sync(&held, &scene);
        assert_eq!(plan.keep, vec![0, 1, 2, 3]);
    }
}
