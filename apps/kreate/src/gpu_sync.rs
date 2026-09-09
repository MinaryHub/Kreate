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
//!
//! # Nor is a moved mesh
//!
//! The same argument, for the thing a drag does. A transform rewrites the
//! positions array and leaves every other array shared, which is the whole
//! point of it declaring its write set — and a plan that could only say "new
//! handle, upload it" answered a pointer position by building new buffers for
//! the indices, the triangle-to-face map, the edge list and the flags, every
//! one of them byte-for-byte what was already there.
//!
//! §4.5 asks for exactly this by name: a change to positions must not
//! invalidate the triangulation, the edge list or the index buffers. So there is
//! a third state, and [`MeshData::only_positions_differ`] recognises it.
//!
//! The three are mutually exclusive by construction, and a change that is
//! *both* — a drag that also moved the selection — matches neither and takes
//! the full upload. Conservative, and correct: a plan that guessed would leave
//! one of the two sets of buffers stale.

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
    /// Slots whose mesh differs only in where its vertices are.
    ///
    /// The slot keeps its GPU buffers and has its positions and normals
    /// rewritten — the two buffers a drag actually changes. Like `reflags`,
    /// these also appear in `keep`.
    pub repositions: Vec<(usize, Arc<MeshData>)>,
    /// Meshes that are not on the GPU yet.
    pub uploads: Vec<Arc<MeshData>>,
}

impl SyncPlan {
    /// Whether anything has to happen at all.
    ///
    /// The overwhelmingly common case is a scene that has not changed, and it
    /// should cost one comparison per object and no GPU work whatever.
    pub fn needs_work(&self, cached: usize) -> bool {
        !self.uploads.is_empty()
            || !self.reflags.is_empty()
            || !self.repositions.is_empty()
            || self.keep.len() != cached
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
        let unclaimed = |slot: &usize| !plan.keep.contains(slot);
        let selection_only = cached
            .iter()
            .enumerate()
            .find(|(slot, held)| unclaimed(slot) && held.only_selection_differs(mesh));
        if let Some((slot, _)) = selection_only {
            plan.keep.push(slot);
            plan.reflags.push((slot, mesh.clone()));
            continue;
        }
        let moved_only = cached
            .iter()
            .enumerate()
            .find(|(slot, held)| unclaimed(slot) && held.only_positions_differ(mesh));
        match moved_only {
            Some((slot, _)) => {
                plan.keep.push(slot);
                plan.repositions.push((slot, mesh.clone()));
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

    /// Moves one vertex of a document's first mesh, the way a transform does:
    /// through the positions array and nothing else.
    fn nudge(doc: &mut Document) {
        let id = doc.meshes().next().expect("a mesh").0;
        doc.mesh_mut(id).expect("a mesh").positions_mut()[0].x += 0.25;
    }

    #[test]
    fn a_moved_mesh_rewrites_positions_rather_than_uploading() {
        // What a drag frame costs. A transform rewrites the positions array and
        // leaves the indices, the triangle-to-face map, the edge list and the
        // flags shared, so a plan that scheduled an upload would rebuild every
        // one of them — byte-for-byte what was already there — on every pointer
        // position. §4.5 asks for this by name.
        let mut doc = doc_with(2);
        let mut cache = ModifierCache::new();
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

        nudge(&mut doc);
        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still());
        let plan = plan_sync(&held, &scene);

        assert!(
            plan.uploads.is_empty(),
            "a move re-uploaded {} mesh(es)",
            plan.uploads.len()
        );
        assert_eq!(plan.repositions.len(), 1, "one mesh moved");
        assert_eq!(plan.repositions[0].0, 0, "the slot it was already in");
        assert!(plan.reflags.is_empty(), "nothing was selected");
        assert_eq!(plan.keep, vec![0, 1], "both slots survive");
        assert!(
            plan.needs_work(held.len()),
            "the positions have to be written"
        );
    }

    #[test]
    fn the_moved_slot_adopts_the_new_handle() {
        // The bug this shape of code had once before, for `reflags`: the draw
        // loop finds a slot by `Arc::ptr_eq` against the evaluated mesh, so a
        // slot left holding the *previous* handle matches nothing and the object
        // silently stops being drawn. The plan has to carry the handle to adopt,
        // and applying it has to leave the cache holding that one.
        let mut doc = doc_with(1);
        let mut cache = ModifierCache::new();
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

        nudge(&mut doc);
        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still());
        let plan = plan_sync(&held, &scene);

        let (_, adopted) = &plan.repositions[0];
        assert!(
            Arc::ptr_eq(adopted, &scene[0].1),
            "the plan carries a handle the draw loop will not find"
        );

        // And once adopted, the next frame is free.
        let after = vec![adopted.clone()];
        let quiet = plan_sync(&after, &scene);
        assert!(!quiet.needs_work(after.len()), "an idle frame did work");
    }

    #[test]
    fn a_move_and_a_selection_together_take_the_full_upload() {
        // The two states are mutually exclusive, and a change that is both
        // matches neither. That is the conservative answer and the only correct
        // one: acting on either predicate alone would leave the other set of
        // buffers stale, and a mesh drawn with last frame’s selection or last
        // frame’s positions is a bug nobody would look for here.
        let mut doc = doc_with(1);
        let mut cache = ModifierCache::new();
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

        let id = doc.meshes().next().expect("a mesh").0;
        {
            let mesh = doc.mesh_mut(id).expect("a mesh");
            mesh.ensure_selection_layers();
            kreate_mesh::selection::select_all(mesh);
            mesh.positions_mut()[0].x += 0.25;
        }
        let mut cache = ModifierCache::new();
        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );

        assert_eq!(
            plan.uploads.len(),
            1,
            "a mixed change must not be guessed at"
        );
        assert!(plan.reflags.is_empty());
        assert!(plan.repositions.is_empty());
    }

    #[test]
    fn a_topology_change_is_not_mistaken_for_a_move() {
        // The guard on the whole thing. `only_positions_differ` rests on the
        // topology arrays being pointer-identical, so a mesh whose connectivity
        // changed must fall through — rewriting positions into buffers sized for
        // a different mesh is how a renderer draws garbage.
        let mut doc = doc_with(1);
        let mut cache = ModifierCache::new();
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

        let id = doc.meshes().next().expect("a mesh").0;
        *doc.mesh_mut(id).expect("a mesh") = kreate_mesh::primitives::uv_sphere(1.0, 8, 4);
        let mut cache = ModifierCache::new();
        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );

        assert_eq!(plan.uploads.len(), 1, "a new topology needs new buffers");
        assert!(plan.repositions.is_empty());
    }

    #[test]
    fn an_empty_cache_uploads_everything() {
        let doc = doc_with(3);
        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still());
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
        let scene = cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still());
        let held = apply(&[], &plan_sync(&[], &scene));

        let plan = plan_sync(&held, &scene);
        assert!(plan.uploads.is_empty());
        assert_eq!(plan.keep, vec![0, 1, 2]);
        assert!(!plan.needs_work(held.len()));
    }

    #[test]
    fn editing_one_mesh_touches_only_that_one() {
        // The reason for handle identity rather than a document version: with a
        // version, editing one object re-uploads every other, and the cost of an
        // edit grows with the size of the scene.
        //
        // Counted as "how many meshes does the plan act on" rather than "how
        // many does it upload", because which path the edited one takes is a
        // separate question — a move now has its own — and this test is about
        // the two it does *not* touch.
        let mut doc = doc_with(3);
        let mut cache = ModifierCache::new();
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

        let target = doc.objects().next().unwrap().1.mesh().unwrap();
        doc.mesh_mut(target).unwrap().positions_mut()[0].y += 1.0;

        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
        let acted_on = plan.uploads.len() + plan.reflags.len() + plan.repositions.len();
        assert_eq!(acted_on, 1, "only the edited mesh");
        assert_eq!(plan.keep.len(), 3, "all three slots survive");
    }

    #[test]
    fn a_removed_object_drops_its_mesh() {
        let mut doc = doc_with(3);
        let mut cache = ModifierCache::new();
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

        let victim = doc.objects().next().unwrap().0;
        doc.remove_object(victim);

        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
        assert_eq!(plan.total(), 2);
        assert!(plan.uploads.is_empty());
        assert!(plan.needs_work(held.len()), "a slot has to be released");
    }

    #[test]
    fn a_hidden_object_drops_its_mesh() {
        let mut doc = doc_with(2);
        let mut cache = ModifierCache::new();
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

        let id = doc.objects().next().unwrap().0;
        doc.object_mut(id).unwrap().visible = false;

        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
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
        let scene = cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still());
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
        let plan = plan_sync(
            &[],
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
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
        let scene = cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still());
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
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

        doc.object_mut(id).unwrap().modifiers[0].kind =
            ModifierKind::Array(kreate_mesh::modifiers::ArrayParams {
                count: 5,
                ..Default::default()
            });

        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
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
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

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

        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
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
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

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

        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
        assert!(
            plan.uploads.is_empty(),
            "a click re-uploaded the geometry: {} uploads",
            plan.uploads.len()
        );
        assert_eq!(plan.reflags.len(), 1, "the flags should be rewritten");
    }

    #[test]
    fn a_geometry_change_is_never_silently_kept() {
        // The companion to every shortcut here: none of them may swallow a real
        // edit, or the screen shows the old shape until something else forces a
        // rebuild. The failure mode is a slot that lands in `keep` with nothing
        // scheduled against it, which is indistinguishable from an idle frame.
        //
        // Both kinds of change, because they take different paths and the
        // shortcut that could swallow each is a different one.
        let mut doc = doc_with(1);
        let mut cache = ModifierCache::new();
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );
        let target = doc.objects().next().unwrap().1.mesh().unwrap();

        doc.mesh_mut(target).unwrap().positions_mut()[0].y += 1.0;
        let moved = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
        assert!(
            moved.needs_work(held.len()),
            "a move was reported as an idle frame"
        );
        assert_eq!(moved.repositions.len(), 1);
        assert!(moved.reflags.is_empty(), "nothing was selected");

        *doc.mesh_mut(target).unwrap() = kreate_mesh::primitives::uv_sphere(1.0, 8, 4);
        let replaced = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
        assert!(replaced.needs_work(held.len()));
        assert_eq!(
            replaced.uploads.len(),
            1,
            "new connectivity needs new buffers"
        );
        assert!(replaced.repositions.is_empty());
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
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );
        assert_eq!(held.len(), 1, "they share one mesh to begin with");

        doc.object_mut(second)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Array(
                kreate_mesh::modifiers::ArrayParams::default(),
            )));

        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
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
        let held = apply(
            &[],
            &plan_sync(
                &[],
                &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
            ),
        );

        let id = doc.objects().next().unwrap().0;
        doc.object_mut(id).unwrap().transform.translation = kreate_math::DVec3::Y;

        let plan = plan_sync(
            &held,
            &cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still()),
        );
        assert!(!plan.needs_work(held.len()));
    }

    #[test]
    fn the_plan_is_in_document_order() {
        // So that a run is reproducible, and so that a palette assigned by slot
        // does not reshuffle every frame.
        let doc = doc_with(4);
        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc, &kreate_scene::modifier::EvalCtx::still());
        let held = apply(&[], &plan_sync(&[], &scene));
        let plan = plan_sync(&held, &scene);
        assert_eq!(plan.keep, vec![0, 1, 2, 3]);
    }
}
