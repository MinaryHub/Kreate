//! The undo stack and operator history.
//!
//! # Restore and re-run, not invert
//!
//! Undo restores the recorded pre-state. Redo *re-runs the operator* with the
//! same parameters rather than replaying a stored post-state. That is possible
//! only because [`Operator::execute`] reads nothing but its parameters, and it
//! buys three things:
//!
//! - no operator has to implement a provably correct inverse,
//! - the adjust-last-operation panel is the same code path with edited
//!   parameters, so it needs no separate mechanism,
//! - a redo cannot drift out of step with the operator's current behaviour.
//!
//! The cost is that an operator must be fast enough to re-run. That is a much
//! weaker requirement than being incrementally updatable, and it removes an
//! entire class of incremental-update bugs.
//!
//! # A history entry is a script line
//!
//! Each entry holds `(op_id, params)`, which is exactly what a script call needs.
//! "Copy as script" from the history is therefore free rather than a feature, and
//! so is macro recording.

use kreate_scene::{DocMut, Document, UndoPayload};

use crate::operator::{Context, OpError, OpResult, OperatorRegistry};
use crate::property::PropertyMap;

/// An interactive operator in progress: an uncommitted transaction.
///
/// Held by the application for as long as the user is dragging. Deliberately not
/// inside [`History`]: the history is a record of what happened, and a gesture
/// that has not been confirmed has not happened yet. Keeping it outside also
/// makes the "no history entry until `Finished`" rule structural rather than
/// something every path has to remember.
pub struct ModalSession {
    op_id: String,
    label: String,
    /// The document before the operator ran at all.
    before: UndoPayload,
    /// The last parameters that were applied successfully.
    params: PropertyMap,
    /// What the last successful apply recorded, which becomes the history
    /// entry's payload on commit.
    payload: UndoPayload,
    /// Hidden parameters at their defaults. See `HistoryEntry::quiet`.
    quiet: Vec<&'static str>,
}

impl ModalSession {
    pub fn op_id(&self) -> &str {
        &self.op_id
    }

    /// The parameters as last applied. What the header shows the user.
    pub fn params(&self) -> &PropertyMap {
        &self.params
    }
}

/// One recorded operation.
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub op_id: String,
    pub params: PropertyMap,
    /// Human-readable label for the history panel.
    pub label: String,
    /// What to restore in order to undo this.
    payload: UndoPayload,
    /// Hidden parameters that were at their defaults, left out of the script
    /// line.
    ///
    /// The record keeps every parameter — replay must not depend on today's
    /// defaults — but a script line that said `autokey=False, key_frame=0.0`
    /// after every move would be noise about a feature that was off. An
    /// auto-keyed move still says so, because then the values are not the
    /// defaults.
    quiet: Vec<&'static str>,
}

/// The hidden parameters of `info` that `full` leaves at their defaults.
fn quiet_names(info: &crate::operator::OperatorInfo, full: &PropertyMap) -> Vec<&'static str> {
    let defaults = info.defaults();
    info.schema
        .iter()
        .filter(|d| d.hidden && full.get(d.name) == defaults.get(d.name))
        .map(|d| d.name)
        .collect()
}

impl HistoryEntry {
    /// The script line that reproduces this operation.
    pub fn to_script_call(&self) -> String {
        if self.quiet.is_empty() {
            return self.params.to_script_call(&self.op_id);
        }
        let mut shown = PropertyMap::new();
        for (name, value) in self.params.iter().filter(|(n, _)| !self.quiet.contains(n)) {
            shown.set(name, value.clone());
        }
        shown.to_script_call(&self.op_id)
    }

    /// The script line for a person to read: any text value longer than
    /// `max` bytes cut short and its length said.
    ///
    /// What the console and the status bar show. A paste carries the whole of
    /// what was pasted — an imported model is megabytes of it — and laying
    /// that out as text every frame stopped the application and filled the
    /// font atlas until every label was garbled. The line itself, for
    /// replaying, is [`HistoryEntry::to_script_call`], whole.
    pub fn to_script_preview(&self, max: usize) -> String {
        let mut shown = PropertyMap::new();
        for (name, value) in self.params.iter().filter(|(n, _)| !self.quiet.contains(n)) {
            let value = match value {
                // By bytes, which is free to ask: counting the characters of
                // megabytes every frame is the cost this is here to avoid.
                kreate_reflect::PropertyValue::Text(t) if t.len() > max => {
                    let head: String = t.chars().take(max).collect();
                    kreate_reflect::PropertyValue::Text(format!("{head}… ({} bytes)", t.len()))
                }
                other => other.clone(),
            };
            shown.set(name, value);
        }
        shown.to_script_call(&self.op_id)
    }

    pub fn bytes(&self) -> usize {
        self.payload.bytes() + self.op_id.len() + self.label.len()
    }
}

/// The undo stack.
///
/// Entries above `cursor` are undone-but-redoable. Dispatching a new operation
/// discards them, which is the conventional and expected behaviour.
pub struct History {
    entries: Vec<HistoryEntry>,
    /// Number of entries currently applied to the document.
    cursor: usize,
    budget_bytes: usize,
    /// Entries dropped by the budget, for reporting rather than silence.
    evicted: usize,
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

impl History {
    /// Default memory budget.
    ///
    /// A budget exists at all because "millions of elements" plus snapshot undo
    /// is how a process quietly reaches 16 GB and starts swapping. Eviction from
    /// the *oldest* end keeps recent undo depth, which is what users actually
    /// reach for.
    pub const DEFAULT_BUDGET: usize = 2 << 30;

    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            cursor: 0,
            budget_bytes: Self::DEFAULT_BUDGET,
            evicted: 0,
        }
    }

    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            ..Self::new()
        }
    }

    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        self.cursor < self.entries.len()
    }

    /// Entries in chronological order. Those at or past [`History::cursor`] are
    /// undone.
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// How many entries the memory budget has discarded.
    pub fn evicted(&self) -> usize {
        self.evicted
    }

    pub fn bytes(&self) -> usize {
        self.entries.iter().map(HistoryEntry::bytes).sum()
    }

    /// The most recent applied entry, for the adjust-last-operation panel.
    pub fn last(&self) -> Option<&HistoryEntry> {
        self.cursor.checked_sub(1).and_then(|i| self.entries.get(i))
    }

    /// Dispatches an operator and records it if it finished.
    ///
    /// The single entry point for anything that changes the document. `Running`,
    /// `Cancelled` and `PassThrough` record nothing: a modal operator owns an
    /// uncommitted transaction until it confirms, and a cancelled one must leave
    /// no trace in the history.
    pub fn dispatch(
        &mut self,
        registry: &OperatorRegistry,
        doc: &mut Document,
        ctx: &Context,
        op_id: &str,
        params: &PropertyMap,
    ) -> Result<OpResult, OpError> {
        self.dispatch_in_session(registry, doc, None, ctx, op_id, params)
    }

    /// The same, with an edit session the operator may work in.
    ///
    /// Separate rather than a parameter on `dispatch`, because most callers —
    /// scripts, tests, anything outside edit mode — have no session and should
    /// not have to say so.
    pub fn dispatch_in_session(
        &mut self,
        registry: &OperatorRegistry,
        doc: &mut Document,
        sessions: Option<&mut crate::EditSessions>,
        ctx: &Context,
        op_id: &str,
        params: &PropertyMap,
    ) -> Result<OpResult, OpError> {
        let info = registry
            .get(op_id)
            .ok_or_else(|| OpError::UnknownOperator(op_id.to_string()))?;
        let label = info.label.to_string();
        // Start from the defaults so a partial map still yields a complete,
        // reproducible record — the history must not depend on today's defaults
        // to be replayable tomorrow.
        let mut full = info.defaults();
        full.overlay(params);

        let mut edit = DocMut::with_sessions(doc, sessions);
        let result = registry.execute(op_id, &full, &mut edit, ctx);
        let payload = edit.finish();

        debug_assert_meshes_valid(doc, op_id);

        match result {
            Ok(OpResult::Finished) => {
                // A redo branch that is no longer reachable.
                self.entries.truncate(self.cursor);
                let quiet = quiet_names(info, &full);
                self.entries.push(HistoryEntry {
                    op_id: op_id.to_string(),
                    params: full,
                    label,
                    payload,
                    quiet,
                });
                self.cursor = self.entries.len();
                self.enforce_budget();
                Ok(OpResult::Finished)
            }
            // Anything else did not happen, so the document goes back to what it
            // was — here, once, rather than in every operator.
            //
            // This is not tidiness. An operator that decides there is nothing to
            // do has usually already run its conversion: an edit-mode operator
            // round-trips the mesh through `BMesh` before it can tell, and that
            // rebuilds every array. Left in place, a "nothing to do" outcome
            // hands the renderer a mesh it has never seen — a full re-upload of
            // geometry that did not change — and does it with no history entry
            // to undo. Restoring puts the original arrays back, so the caches
            // that key on array identity see nothing happened, because nothing
            // did.
            other => {
                payload.restore(doc);
                other
            }
        }
    }

    // --- interactive operators ----------------------------------------------

    /// Starts an interactive operator, holding an **uncommitted** transaction.
    ///
    /// Nothing reaches the undo stack until [`History::commit_modal`], and
    /// [`History::cancel_modal`] writes nothing at all. That is §1.7's rule and
    /// it is what makes a cancelled drag leave no trace — including no entry
    /// that a later undo would walk into.
    ///
    /// `None` when the operator declined to run: an empty selection, a
    /// single-point scale. The caller must not start a drag in that case, and
    /// getting this wrong is the specific bug where the interactive session
    /// silently retunes whatever entry happened to be at the top of the history
    /// instead of its own.
    pub fn begin_modal(
        &mut self,
        registry: &OperatorRegistry,
        doc: &mut Document,
        sessions: Option<&mut crate::EditSessions>,
        ctx: &Context,
        op_id: &str,
        params: &PropertyMap,
    ) -> Result<Option<ModalSession>, OpError> {
        let info = registry
            .get(op_id)
            .ok_or_else(|| OpError::UnknownOperator(op_id.to_string()))?;
        let label = info.label.to_string();

        // The state to come back to, taken before the operator runs. A whole
        // document, which is microseconds and a handful of pointer bumps because
        // every array is behind an `Arc` — the same property that makes snapshot
        // undo affordable at all.
        let before = UndoPayload::Snapshot(Box::new(doc.clone()));

        let mut full = info.defaults();
        full.overlay(params);
        let mut edit = DocMut::with_sessions(doc, sessions);
        let result = registry.execute(op_id, &full, &mut edit, ctx);
        let payload = edit.finish();

        match result {
            Ok(OpResult::Finished) => Ok(Some(ModalSession {
                op_id: op_id.to_string(),
                label,
                before,
                quiet: quiet_names(info, &full),
                params: full,
                payload,
            })),
            Ok(_) => {
                // Back to where it was, then say so. Same reasoning as
                // `dispatch`: the operator may already have rebuilt every array
                // deciding it had nothing to do.
                payload.restore(doc);
                Ok(None)
            }
            Err(e) => {
                payload.restore(doc);
                Err(e)
            }
        }
    }

    /// Re-runs the interactive operator with new parameters.
    ///
    /// **Restore, then apply from the original.** Never incremental. §2.2 makes
    /// this a rule rather than an implementation choice, and it buys three
    /// things: a long drag accumulates no floating point drift, a cancel is a
    /// single restore, and the operator needs no incremental update path — which
    /// is an entire class of bug that then does not exist.
    ///
    /// It also means the mouse and the keyboard reach the same code. A drag that
    /// ends with a final set of parameters is indistinguishable from someone
    /// having typed those parameters, which is what makes the adjust panel,
    /// scripting and macro replay work without any of them knowing a mouse was
    /// involved.
    pub fn update_modal(
        &mut self,
        registry: &OperatorRegistry,
        doc: &mut Document,
        sessions: Option<&mut crate::EditSessions>,
        ctx: &Context,
        modal: &mut ModalSession,
        params: &PropertyMap,
    ) -> Result<(), OpError> {
        modal.before.restore(doc);

        let info = registry
            .get(&modal.op_id)
            .ok_or_else(|| OpError::UnknownOperator(modal.op_id.clone()))?;
        let mut full = info.defaults();
        full.overlay(params);

        let mut edit = DocMut::with_sessions(doc, sessions);
        let result = registry.execute(&modal.op_id, &full, &mut edit, ctx);
        let payload = edit.finish();

        match result {
            Ok(OpResult::Finished) => {
                modal.quiet = quiet_names(info, &full);
                modal.params = full;
                modal.payload = payload;
                Ok(())
            }
            Ok(_) => {
                // The operator declined *this* value — a scale dragged back
                // through 1.0, say. The document is left at the original, the
                // session keeps the last value that worked, and the drag carries
                // on. Ending it here would abort a gesture because the pointer
                // passed through a degenerate position.
                payload.restore(doc);
                Ok(())
            }
            Err(e) => {
                payload.restore(doc);
                Err(e)
            }
        }
    }

    /// Commits the interactive operator as one history entry.
    ///
    /// One entry for the whole gesture, carrying the final parameters. A drag
    /// across two hundred frames has to undo in one step, and has to read in the
    /// history as the single thing the user did.
    pub fn commit_modal(&mut self, modal: ModalSession) {
        self.entries.truncate(self.cursor);
        self.entries.push(HistoryEntry {
            op_id: modal.op_id,
            params: modal.params,
            label: modal.label,
            payload: modal.payload,
            quiet: modal.quiet,
        });
        self.cursor = self.entries.len();
        self.enforce_budget();
    }

    /// Abandons the interactive operator, restoring the document.
    ///
    /// Takes no `&mut self`: there is nothing in the history to unwind, because
    /// nothing was ever put there. A cancelled drag is not an undo step.
    pub fn cancel_modal(modal: ModalSession, doc: &mut Document) {
        modal.before.restore(doc);
    }

    /// Undoes the most recent applied entry.
    pub fn undo(&mut self, doc: &mut Document) -> bool {
        let Some(index) = self.cursor.checked_sub(1) else {
            return false;
        };
        let Some(entry) = self.entries.get(index) else {
            return false;
        };
        entry.payload.restore(doc);
        self.cursor = index;
        debug_assert_meshes_valid(doc, "<undo>");
        true
    }

    /// Redoes the next entry by re-running its operator.
    ///
    /// Correct because the document has been restored to exactly the state the
    /// operator originally saw, and the operator reads nothing but its
    /// parameters.
    pub fn redo(
        &mut self,
        registry: &OperatorRegistry,
        doc: &mut Document,
        ctx: &Context,
    ) -> Result<bool, OpError> {
        let Some(entry) = self.entries.get(self.cursor) else {
            return Ok(false);
        };
        let (op_id, params) = (entry.op_id.clone(), entry.params.clone());

        let mut edit = DocMut::new(doc);
        let result = registry.execute(&op_id, &params, &mut edit, ctx);
        let payload = edit.finish();
        result?;

        // The re-run may record a different payload than the first run did — for
        // instance if the mesh has since been compacted — so the stored one is
        // replaced rather than trusted.
        self.entries[self.cursor].payload = payload;
        self.cursor += 1;
        Ok(true)
    }

    /// Re-runs the last operation with new parameters.
    ///
    /// The adjust-last-operation panel. Restore, then run again: the same path
    /// redo takes, which is why editing a parameter after the fact needs no
    /// operator-specific support.
    pub fn adjust_last(
        &mut self,
        registry: &OperatorRegistry,
        doc: &mut Document,
        ctx: &Context,
        params: &PropertyMap,
    ) -> Result<bool, OpError> {
        let Some(index) = self.cursor.checked_sub(1) else {
            return Ok(false);
        };
        let op_id = self.entries[index].op_id.clone();

        self.entries[index].payload.restore(doc);

        let mut full = registry
            .get(&op_id)
            .ok_or_else(|| OpError::UnknownOperator(op_id.clone()))?
            .defaults();
        full.overlay(params);

        let mut edit = DocMut::new(doc);
        let result = registry.execute(&op_id, &full, &mut edit, ctx);
        let payload = edit.finish();

        match result {
            Ok(_) => {
                self.entries[index].params = full;
                self.entries[index].payload = payload;
                Ok(true)
            }
            Err(e) => {
                // The document is back at the pre-operation state and the entry
                // no longer describes anything applied, so the cursor must move
                // with it rather than claiming an applied entry that is not.
                self.cursor = index;
                Err(e)
            }
        }
    }

    /// A script that reproduces the applied history, one call per line.
    ///
    /// Macro recording, with no separate machinery.
    pub fn to_script(&self) -> String {
        self.entries[..self.cursor]
            .iter()
            .map(HistoryEntry::to_script_call)
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.cursor = 0;
    }

    /// Drops the oldest entries until the budget is met.
    ///
    /// The most recent entry is always kept, however large: a budget that could
    /// discard the operation the user just performed would make undo
    /// untrustworthy, which is worse than exceeding a soft limit.
    fn enforce_budget(&mut self) {
        while self.bytes() > self.budget_bytes && self.entries.len() > 1 {
            self.entries.remove(0);
            self.cursor = self.cursor.saturating_sub(1);
            self.evicted += 1;
        }
    }
}

/// Validates every mesh in the document, in debug builds only.
///
/// The single highest-return line in the project. A mesh kernel does not fail
/// loudly: an operator leaves a subtly invalid mesh, three operators later
/// something crashes or the file is silently wrong, and the trail back to the
/// cause is gone. Checking at the dispatcher means the report names the
/// operator that did it, at the moment it did it.
///
/// Compiled out of release builds: the check is O(V + E + F) with a duplicate
/// scan, which is nothing beside an operator but is not free on a 10 M-vertex
/// mesh at interactive rates.
#[inline]
fn debug_assert_meshes_valid(doc: &Document, op_id: &str) {
    #[cfg(debug_assertions)]
    for (id, mesh) in doc.meshes() {
        let violations = mesh.validate();
        assert!(
            violations.is_empty(),
            "`{op_id}` left {id:?} invalid ({} problem(s)):\n{}",
            violations.len(),
            violations
                .iter()
                .map(|v| format!("  - {v:?}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (doc, op_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;
    use crate::{OpParams as OpParamsDerive, PropertyValue};
    use kreate_scene::document::{MeshId, Object, ObjectData};

    #[derive(Clone, Debug, Default, PartialEq, OpParamsDerive)]
    struct MoveParams {
        /// Distance to move the selected vertices along Z.
        #[param(unit = "length")]
        offset: f64,
    }

    /// Moves selected vertices, through the declared write-set path.
    struct MoveSelected;

    impl Operator for MoveSelected {
        type Params = MoveParams;
        fn id() -> &'static str {
            "mesh.move_selected"
        }
        fn label() -> &'static str {
            "Move Selected"
        }
        fn execute(
            params: &Self::Params,
            doc: &mut DocMut<'_>,
            _ctx: &Context,
        ) -> Result<OpResult, OpError> {
            let mesh =
                first_mesh(doc.document()).ok_or_else(|| OpError::Failed("no mesh".into()))?;
            let indices = doc.selected_vertices(mesh);
            if indices.is_empty() {
                return Ok(OpResult::Cancelled);
            }
            let offset = params.offset as f32;
            doc.edit_positions(mesh, &indices, |_, p| p.z += offset)
                .map_err(|e| OpError::Failed(e.to_string()))?;
            Ok(OpResult::Finished)
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, OpParamsDerive)]
    struct CorruptParams {
        /// Present only because every operator has a parameter block.
        unused: f64,
    }

    /// Deliberately leaves the mesh structurally invalid.
    ///
    /// Exists so the safety net can be shown to be armed. A guard that is never
    /// observed to fire is indistinguishable from one that was accidentally
    /// compiled out, and this one is the project's single most important
    /// defence.
    struct CorruptMesh;

    impl Operator for CorruptMesh {
        type Params = CorruptParams;
        fn id() -> &'static str {
            "test.corrupt_mesh"
        }
        fn label() -> &'static str {
            "Corrupt Mesh"
        }
        fn execute(
            _params: &Self::Params,
            doc: &mut DocMut<'_>,
            _ctx: &Context,
        ) -> Result<OpResult, OpError> {
            let mesh =
                first_mesh(doc.document()).ok_or_else(|| OpError::Failed("no mesh".into()))?;
            let indices = vec![0u32];
            doc.edit_positions(mesh, &indices, |_, p| {
                *p = kreate_math::Vec3::splat(f32::NAN)
            })
            .map_err(|e| OpError::Failed(e.to_string()))?;
            Ok(OpResult::Finished)
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, OpParamsDerive)]
    struct AddCubeParams {
        /// Edge length of the new cube.
        #[param(min = 0.001, unit = "length")]
        size: f64,
    }

    /// Adds an object, through the coarse snapshot path.
    struct AddCube;

    impl Operator for AddCube {
        type Params = AddCubeParams;
        fn id() -> &'static str {
            "object.add_cube"
        }
        fn label() -> &'static str {
            "Add Cube"
        }
        fn execute(
            params: &Self::Params,
            doc: &mut DocMut<'_>,
            _ctx: &Context,
        ) -> Result<OpResult, OpError> {
            let size = if params.size > 0.0 { params.size } else { 1.0 };
            doc.document_mut()
                .add_mesh_object("Cube", kreate_mesh::primitives::cube(size as f32));
            Ok(OpResult::Finished)
        }
    }

    struct AlwaysFails;

    impl Operator for AlwaysFails {
        type Params = AddCubeParams;
        fn id() -> &'static str {
            "test.fail"
        }
        fn label() -> &'static str {
            "Always Fails"
        }
        fn execute(
            _params: &Self::Params,
            _doc: &mut DocMut<'_>,
            _ctx: &Context,
        ) -> Result<OpResult, OpError> {
            Err(OpError::Failed("nope".into()))
        }
    }

    fn first_mesh(doc: &Document) -> Option<MeshId> {
        doc.objects().find_map(|(_, o)| o.mesh())
    }

    fn registry() -> OperatorRegistry {
        let mut r = OperatorRegistry::new();
        r.register::<MoveSelected>();
        r.register::<AddCube>();
        r.register::<AlwaysFails>();
        r
    }

    /// A grid with vertices 0 and 1 selected.
    fn doc_with_selection() -> Document {
        let mut d = Document::new();
        let mesh = d.add_mesh(kreate_mesh::primitives::grid(4.0, 3, 3));
        d.add_object(Object::new("Grid", ObjectData::Mesh(mesh)));
        let n = d.mesh(mesh).unwrap().num_verts() as usize;
        let mut sel = vec![false; n];
        sel[0] = true;
        sel[1] = true;
        d.mesh_mut(mesh).unwrap().attributes_mut().insert(
            kreate_mesh::attribute::names::SELECT_VERT,
            kreate_mesh::attribute::Domain::Point,
            kreate_mesh::attribute::AttrData::Bool(std::sync::Arc::new(sel)),
        );
        d
    }

    fn params(offset: f64) -> PropertyMap {
        let mut m = PropertyMap::new();
        m.set("offset", PropertyValue::Float(offset));
        m
    }

    fn z_of(doc: &Document, index: usize) -> f32 {
        let mesh = first_mesh(doc).unwrap();
        doc.mesh(mesh).unwrap().positions()[index].z
    }

    // --- the property the whole architecture rests on ----------------------

    #[test]
    fn apply_then_undo_restores_the_document_exactly() {
        let mut doc = doc_with_selection();
        let before: Vec<_> = doc
            .mesh(first_mesh(&doc).unwrap())
            .unwrap()
            .positions()
            .to_vec();

        let r = registry();
        let mut h = History::new();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(2.0),
        )
        .unwrap();
        assert_ne!(z_of(&doc, 0), before[0].z);

        assert!(h.undo(&mut doc));
        assert_eq!(
            doc.mesh(first_mesh(&doc).unwrap()).unwrap().positions(),
            before.as_slice(),
            "undo must restore the document exactly"
        );
    }

    #[test]
    fn undo_and_redo_are_symmetric_over_many_steps() {
        let mut doc = doc_with_selection();
        let original: Vec<_> = doc
            .mesh(first_mesh(&doc).unwrap())
            .unwrap()
            .positions()
            .to_vec();
        let r = registry();
        let mut h = History::new();

        for i in 1..=5 {
            h.dispatch(
                &r,
                &mut doc,
                &Context::default(),
                "mesh.move_selected",
                &params(i as f64),
            )
            .unwrap();
        }
        let after_all = z_of(&doc, 0);

        for _ in 0..5 {
            assert!(h.undo(&mut doc));
        }
        assert_eq!(
            doc.mesh(first_mesh(&doc).unwrap()).unwrap().positions(),
            original.as_slice()
        );
        assert!(!h.can_undo());

        for _ in 0..5 {
            assert!(h.redo(&r, &mut doc, &Context::default()).unwrap());
        }
        assert!(
            (z_of(&doc, 0) - after_all).abs() < 1e-6,
            "redo must reach the same state"
        );
        assert!(!h.can_redo());
    }

    #[test]
    fn a_new_operation_discards_the_redo_branch() {
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::new();

        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.0),
        )
        .unwrap();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.0),
        )
        .unwrap();
        h.undo(&mut doc);
        assert!(h.can_redo());

        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(5.0),
        )
        .unwrap();
        assert!(!h.can_redo(), "the unreachable branch must be discarded");
        assert_eq!(h.entries().len(), 2);
    }

    #[test]
    fn undoing_a_topology_change_restores_the_object_graph() {
        let mut doc = Document::new();
        let r = registry();
        let mut h = History::new();
        let mut p = PropertyMap::new();
        p.set("size", PropertyValue::Float(2.0));

        h.dispatch(&r, &mut doc, &Context::default(), "object.add_cube", &p)
            .unwrap();
        assert_eq!(doc.object_count(), 1);

        h.undo(&mut doc);
        assert_eq!(doc.object_count(), 0, "the object must be gone");

        h.redo(&r, &mut doc, &Context::default()).unwrap();
        assert_eq!(doc.object_count(), 1);
    }

    // --- adjust last operation --------------------------------------------

    #[test]
    fn adjusting_the_last_operation_replaces_its_effect() {
        // The F9 panel. Restore, then re-run with edited values — the same path
        // redo takes, which is why this needs no operator-specific support.
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::new();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.0),
        )
        .unwrap();
        assert!((z_of(&doc, 0) - 1.0).abs() < 1e-6);

        h.adjust_last(&r, &mut doc, &Context::default(), &params(7.0))
            .unwrap();
        assert!(
            (z_of(&doc, 0) - 7.0).abs() < 1e-6,
            "the effect must be replaced, not compounded"
        );
        assert_eq!(
            h.entries().len(),
            1,
            "adjusting must not add a history entry"
        );
    }

    #[test]
    fn adjusting_repeatedly_does_not_accumulate() {
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::new();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.0),
        )
        .unwrap();
        for v in [2.0, 3.0, 0.5] {
            h.adjust_last(&r, &mut doc, &Context::default(), &params(v))
                .unwrap();
        }
        assert!((z_of(&doc, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn undo_after_adjusting_restores_the_original_state() {
        let mut doc = doc_with_selection();
        let original = z_of(&doc, 0);
        let r = registry();
        let mut h = History::new();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.0),
        )
        .unwrap();
        h.adjust_last(&r, &mut doc, &Context::default(), &params(9.0))
            .unwrap();
        h.undo(&mut doc);
        assert!((z_of(&doc, 0) - original).abs() < 1e-6);
    }

    #[test]
    fn adjusting_with_nothing_in_the_history_is_a_no_op() {
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::new();
        assert!(!h
            .adjust_last(&r, &mut doc, &Context::default(), &params(1.0))
            .unwrap());
    }

    // --- what must not be recorded ----------------------------------------

    #[test]
    fn a_cancelled_operator_leaves_no_trace() {
        // A modal operator owns an uncommitted transaction until it confirms.
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::grid(1.0, 1, 1));
        doc.add_object(Object::new("Grid", ObjectData::Mesh(mesh)));

        let r = registry();
        let mut h = History::new();
        // Nothing is selected, so the operator cancels.
        let result = h
            .dispatch(
                &r,
                &mut doc,
                &Context::default(),
                "mesh.move_selected",
                &params(1.0),
            )
            .unwrap();
        assert_eq!(result, OpResult::Cancelled);
        assert!(h.entries().is_empty());
        assert!(!h.can_undo());
    }

    #[test]
    fn a_failing_operator_is_not_recorded() {
        let mut doc = Document::new();
        let r = registry();
        let mut h = History::new();
        assert!(h
            .dispatch(
                &r,
                &mut doc,
                &Context::default(),
                "test.fail",
                &PropertyMap::new()
            )
            .is_err());
        assert!(h.entries().is_empty());
    }

    #[test]
    fn dispatching_an_unknown_operator_is_an_error_and_records_nothing() {
        let mut doc = Document::new();
        let r = registry();
        let mut h = History::new();
        assert!(h
            .dispatch(
                &r,
                &mut doc,
                &Context::default(),
                "nope.nope",
                &PropertyMap::new()
            )
            .is_err());
        assert!(h.entries().is_empty());
    }

    // --- the history as a script ------------------------------------------

    #[test]
    fn the_history_reads_as_a_script() {
        // Macro recording and "copy as script", with no separate machinery: an
        // entry already holds exactly what a script call needs.
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::new();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.5),
        )
        .unwrap();
        let mut p = PropertyMap::new();
        p.set("size", PropertyValue::Float(2.0));
        h.dispatch(&r, &mut doc, &Context::default(), "object.add_cube", &p)
            .unwrap();

        assert_eq!(
            h.to_script(),
            "kreate.ops.mesh.move_selected(offset=1.5)\n\
             kreate.ops.object.add_cube(size=2.0)"
        );
    }

    #[test]
    fn an_undone_operation_is_not_in_the_script() {
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::new();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.0),
        )
        .unwrap();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(2.0),
        )
        .unwrap();
        h.undo(&mut doc);
        assert_eq!(h.to_script().lines().count(), 1);
    }

    #[test]
    fn a_recorded_entry_is_complete_even_from_a_partial_call() {
        // The record must not depend on today's defaults to be replayable
        // tomorrow, so a partial dispatch is expanded before it is stored.
        let mut doc = Document::new();
        let r = registry();
        let mut h = History::new();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "object.add_cube",
            &PropertyMap::new(),
        )
        .unwrap();
        assert_eq!(h.last().unwrap().params.len(), 1);
        assert!(h.last().unwrap().params.get("size").is_some());
    }

    // --- memory budget ----------------------------------------------------

    #[test]
    fn the_budget_evicts_the_oldest_entries() {
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::with_budget(200);

        for i in 1..=20 {
            h.dispatch(
                &r,
                &mut doc,
                &Context::default(),
                "mesh.move_selected",
                &params(i as f64),
            )
            .unwrap();
        }
        assert!(
            h.evicted() > 0,
            "the budget should have discarded something"
        );
        assert!(h.bytes() <= 200 || h.entries().len() == 1);
        // Recent undo depth is what users reach for, so it is what survives.
        assert!(h.can_undo());
    }

    #[test]
    fn the_most_recent_entry_is_never_evicted() {
        // A budget that could discard the operation just performed would make
        // undo untrustworthy, which is worse than exceeding a soft limit.
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::with_budget(0);
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.0),
        )
        .unwrap();
        assert_eq!(h.entries().len(), 1);
        assert!(h.can_undo());
    }

    #[test]
    fn undo_and_redo_report_availability_correctly() {
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::new();
        assert!(!h.can_undo() && !h.can_redo());
        assert!(!h.undo(&mut doc));

        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.0),
        )
        .unwrap();
        assert!(h.can_undo() && !h.can_redo());

        h.undo(&mut doc);
        assert!(!h.can_undo() && h.can_redo());
        assert!(h.redo(&r, &mut doc, &Context::default()).is_ok());
        assert!(h.can_undo() && !h.can_redo());
    }

    #[test]
    fn a_long_text_value_is_shortened_for_reading_and_kept_for_replaying() {
        let mut params = PropertyMap::new();
        params.set(
            "data",
            kreate_reflect::PropertyValue::Text("ab".repeat(500_000)),
        );
        params.set("name", kreate_reflect::PropertyValue::Text("Chair".into()));
        let entry = HistoryEntry {
            op_id: "object.paste".into(),
            params,
            label: "Paste".into(),
            payload: UndoPayload::Snapshot(Box::new(Document::new())),
            quiet: Vec::new(),
        };
        let shown = entry.to_script_preview(20);
        assert!(shown.len() < 200, "{shown}");
        assert!(shown.contains("(1000000 bytes)"));
        assert!(shown.contains("Chair"));
        assert!(entry.to_script_call().len() > 1_000_000);
    }

    #[test]
    fn a_history_entry_carries_a_label_for_the_panel() {
        let mut doc = doc_with_selection();
        let r = registry();
        let mut h = History::new();
        h.dispatch(
            &r,
            &mut doc,
            &Context::default(),
            "mesh.move_selected",
            &params(1.0),
        )
        .unwrap();
        assert_eq!(h.last().unwrap().label, "Move Selected");
    }

    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "the validation hook is compiled out of release builds"
    )]
    #[should_panic(expected = "test.corrupt_mesh")]
    fn the_dispatcher_catches_an_operator_that_corrupts_a_mesh() {
        // The plan's first-listed risk, made observable: an operator leaves bad
        // data, and the failure names *that* operator rather than surfacing three
        // operations later as an unattributable crash.
        let mut registry = OperatorRegistry::new();
        registry.register::<AddCube>();
        registry.register::<CorruptMesh>();
        let mut history = History::new();
        let mut doc = Document::new();
        let ctx = Context::default();

        let mut size = PropertyMap::new();
        size.set("size", PropertyValue::Float(2.0));
        history
            .dispatch(&registry, &mut doc, &ctx, "object.add_cube", &size)
            .unwrap();

        let _ = history.dispatch(
            &registry,
            &mut doc,
            &ctx,
            "test.corrupt_mesh",
            &PropertyMap::new(),
        );
    }
}
