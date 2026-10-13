//! The operator trait and registry.
//!
//! # The spine rule
//!
//! > Every document mutation goes through an operator. The UI never touches a
//! > `Document` directly — it dispatches `(op_id, params)`.
//!
//! No exceptions, gizmo drags included. This one rule buys undo, macro
//! recording, scripting, plugin parity, and headless testability at the same
//! time, from the same machinery. Violate it once and the rot starts, because
//! the next feature that needs any of those four has to be special-cased.
//!
//! # Interactive and pure are separate
//!
//! [`Operator::execute`] reads *only* its parameters. An interactive tool reads
//! the mouse, but on confirm it must be exactly reproducible by calling
//! `execute` with the final parameters. That separation is what gives, for free:
//!
//! - the adjust-last-operation panel (restore, then re-run with edited values),
//! - scripting and macro recording (a history entry *is* the script line),
//! - deterministic headless tests (no mouse simulation),
//! - a semantic history a user can read.
//!
//! An operator therefore does not need to be incrementally updatable — only fast
//! enough to re-run. That eliminates an entire class of incremental-update bugs.

use kreate_scene::{DocMut, Document};

use crate::property::{OpParams, PropertyDesc, PropertyMap};

/// What an operator did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpResult {
    /// Work is done and should be recorded in the history.
    Finished,
    /// A modal operator is still running; nothing is recorded yet.
    Running,
    /// Abandoned. The document is unchanged and nothing is recorded.
    Cancelled,
    /// The operator declined the event; the dispatcher should keep looking.
    PassThrough,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum OpError {
    #[error("no operator registered with id `{0}`")]
    UnknownOperator(String),
    /// The operator, and why not and what would let it — said for a person
    /// and for a script's author alike, who otherwise retry blind.
    #[error("`{0}` cannot run now: {1}")]
    PollFailed(String, String),
    #[error("{0}")]
    Failed(String),
}

pub use crate::mode::Mode;

/// Why an operator would not run, and what would let it: what its family
/// needs, the mode and the active object as they are, and the lines that
/// would put things right.
pub fn poll_reason(id: &str, ctx: &Context, doc: &Document) -> String {
    use kreate_scene::document::ObjectData;
    let active = doc.selection.active.and_then(|a| doc.object(a));
    let what = match active {
        None => "nothing is active".to_string(),
        Some(o) => {
            let kind = match o.data {
                ObjectData::Mesh(_) => "a mesh",
                ObjectData::Camera(_) => "a camera",
                ObjectData::Light(_) => "a light",
                ObjectData::Armature(_) => "an armature",
                ObjectData::Empty => "an empty",
            };
            format!("the active object is \"{}\", {kind}", o.name)
        }
    };
    let mode = ctx.mode.tag();
    let family = id.split('.').next().unwrap_or("");
    let active_mesh = active.is_some_and(|o| o.mesh().is_some());
    let select = "kreate.ops.object.select(name=\"...\")";
    let edit = "kreate.ops.object.mode_set(mode=\"edit\")";
    let (needs, fix) = match family {
        "mesh" | "uv" => (
            "it works on a mesh in edit mode".to_string(),
            if !active_mesh {
                format!("select a mesh with {select}, then {edit}")
            } else if !ctx.mode.edits_elements() {
                format!("enter edit mode with {edit}")
            } else {
                "select the elements it acts on first".to_string()
            },
        ),
        "pose" => (
            "it works on an armature in pose mode".to_string(),
            format!(
                "select the armature with {select}, then kreate.ops.object.mode_set(mode=\"pose\")"
            ),
        ),
        "armature" => (
            "it works on an armature in its edit mode".to_string(),
            format!("select the armature with {select}, then {edit}"),
        ),
        "sculpt" => (
            "it works on a mesh in sculpt mode".to_string(),
            format!(
                "select a mesh with {select}, then kreate.ops.object.mode_set(mode=\"sculpt\")"
            ),
        ),
        "object" if ctx.mode != Mode::RESTING => (
            "it works in object mode".to_string(),
            "leave with kreate.ops.object.mode_set(mode=\"object\")".to_string(),
        ),
        _ => (
            "what it acts on is not there".to_string(),
            format!("check the selection with get_scene, or select with {select}"),
        ),
    };
    format!("{needs}; the mode is {mode} and {what}. To fix: {fix}")
}

/// What an operator is allowed to know about the world beyond the document.
///
/// Deliberately small. Anything an operator reads from here rather than from its
/// parameters is something a script cannot control and a test cannot set, so the
/// bar for adding a field is high.
#[derive(Clone, Debug, Default)]
pub struct Context {
    pub mode: Mode,
}

/// A statically typed operator.
///
/// Implementors are unit-like types; all state lives in `Params`. That is what
/// makes an operator reproducible from `(id, params)` alone.
pub trait Operator: 'static {
    type Params: OpParams;

    /// Stable identifier, e.g. `"mesh.bevel"`. Part of the scripting API and of
    /// saved keymaps, so it must not change once released.
    fn id() -> &'static str;

    /// Fallback English label, used when no translation is loaded.
    fn label() -> &'static str;

    /// Whether the operator can run at all right now.
    ///
    /// Checked before `execute` and used to grey out menu items. Cheap by
    /// contract: it runs for every visible menu entry, every frame a menu is open.
    fn poll(_ctx: &Context, _doc: &Document) -> bool {
        true
    }

    /// The parameters that would leave the document as it is, for an
    /// operator whose parameters *are* the state it sets — a transform typed
    /// into a panel, a material's numbers.
    ///
    /// A script line names only what it changes. Without this, everything it
    /// left out would come from the defaults, and setting a material's colour
    /// would quietly reset its roughness. `given` is what the line did say,
    /// for an operator that needs it to know what it is about (a slot).
    fn current(_ctx: &Context, _doc: &Document, _given: &PropertyMap) -> Option<Self::Params> {
        None
    }

    /// Performs the work. Reads only `params` and the document.
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        ctx: &Context,
    ) -> Result<OpResult, OpError>;
}

/// A type-erased operator, as stored in the registry.
///
/// Function pointers rather than trait objects: there is no per-operator state to
/// box, dispatch happens once per user action rather than per element, and this
/// keeps the registry `Clone` and cheap.
#[derive(Clone)]
pub struct OperatorInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub schema: &'static [PropertyDesc],
    defaults: fn() -> PropertyMap,
    poll: fn(&Context, &Document) -> bool,
    current: fn(&Context, &Document, &PropertyMap) -> Option<PropertyMap>,
    exec: fn(&PropertyMap, &mut DocMut<'_>, &Context) -> Result<OpResult, OpError>,
}

impl OperatorInfo {
    fn of<O: Operator>() -> Self {
        Self {
            id: O::id(),
            label: O::label(),
            schema: O::Params::schema(),
            defaults: O::Params::defaults_map,
            poll: O::poll,
            current: |ctx, doc, given| O::current(ctx, doc, given).map(|p| p.to_map()),
            exec: |map, doc, ctx| {
                // The one place parameters cross from runtime-typed to typed.
                // `from_map` fills in missing entries from the defaults and
                // clamps to the schema, so a script or a stale keymap cannot
                // hand an operator a value it has to defend against.
                let params = O::Params::from_map(map);
                O::execute(&params, doc, ctx)
            },
        }
    }

    pub fn defaults(&self) -> PropertyMap {
        (self.defaults)()
    }

    pub fn poll(&self, ctx: &Context, doc: &Document) -> bool {
        (self.poll)(ctx, doc)
    }

    /// Fills what `given` leaves out from the document as it stands, for an
    /// operator that says what that is (see [`Operator::current`]). What is
    /// left after that still comes from the defaults.
    pub fn fill_from_current(&self, ctx: &Context, doc: &Document, given: &mut PropertyMap) {
        let Some(current) = (self.current)(ctx, doc, given) else {
            return;
        };
        for (name, value) in current.iter() {
            if given.get(name).is_none() {
                given.set(name, value.clone());
            }
        }
    }

    /// i18n key for the operator's own name.
    pub fn name_key(&self) -> String {
        crate::property::i18n_key(&["op", self.id, "name"])
    }

    /// i18n key for the operator's description.
    pub fn desc_key(&self) -> String {
        crate::property::i18n_key(&["op", self.id, "desc"])
    }
}

/// The set of known operators.
///
/// Kept sorted by id: the listing feeds menus, the command palette and the
/// scripting surface, all of which must be stable between runs.
#[derive(Clone, Default)]
pub struct OperatorRegistry {
    entries: Vec<OperatorInfo>,
}

impl OperatorRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an operator, replacing any previous one with the same id.
    ///
    /// Replacement rather than refusal, because an add-on overriding a built-in
    /// is a legitimate thing to want. Returns whether something was replaced, so
    /// the caller can log it.
    pub fn register<O: Operator>(&mut self) -> bool {
        let info = OperatorInfo::of::<O>();
        match self.entries.binary_search_by(|e| e.id.cmp(info.id)) {
            Ok(i) => {
                self.entries[i] = info;
                true
            }
            Err(i) => {
                self.entries.insert(i, info);
                false
            }
        }
    }

    pub fn get(&self, id: &str) -> Option<&OperatorInfo> {
        self.entries
            .binary_search_by(|e| e.id.cmp(id))
            .ok()
            .map(|i| &self.entries[i])
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All operators, in id order.
    pub fn iter(&self) -> impl Iterator<Item = &OperatorInfo> {
        self.entries.iter()
    }

    /// Operators whose id starts with `prefix`, e.g. `"mesh."`.
    pub fn in_namespace<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a OperatorInfo> {
        self.entries
            .iter()
            .filter(move |e| e.id.starts_with(prefix))
    }

    /// Runs an operator by id against a recording document handle.
    ///
    /// The single dispatch point. Everything — a menu item, a keystroke, a gizmo
    /// drag on confirm, a script call — arrives here, which is why undo and
    /// history need no cooperation from individual operators.
    pub fn execute(
        &self,
        id: &str,
        params: &PropertyMap,
        doc: &mut DocMut<'_>,
        ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let info = self
            .get(id)
            .ok_or_else(|| OpError::UnknownOperator(id.to_string()))?;
        if !info.poll(ctx, doc.document()) {
            return Err(OpError::PollFailed(
                id.to_string(),
                poll_reason(id, ctx, doc.document()),
            ));
        }
        (info.exec)(params, doc, ctx)
    }

    /// Runs an operator with its default parameters.
    pub fn execute_default(
        &self,
        id: &str,
        doc: &mut DocMut<'_>,
        ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let defaults = self
            .get(id)
            .ok_or_else(|| OpError::UnknownOperator(id.to_string()))?
            .defaults();
        self.execute(id, &defaults, doc, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpParams as OpParamsDerive;
    use kreate_scene::document::{Object, ObjectData};

    #[derive(Clone, Debug, Default, PartialEq, OpParamsDerive)]
    struct RenameParams {
        /// New name for the active object.
        name: String,
    }

    struct Rename;

    impl Operator for Rename {
        type Params = RenameParams;
        fn id() -> &'static str {
            "object.rename"
        }
        fn label() -> &'static str {
            "Rename Object"
        }
        fn poll(_ctx: &Context, doc: &Document) -> bool {
            doc.selection.active.is_some()
        }
        fn execute(
            params: &Self::Params,
            doc: &mut DocMut<'_>,
            _ctx: &Context,
        ) -> Result<OpResult, OpError> {
            let active = doc
                .document()
                .selection
                .active
                .ok_or_else(|| OpError::Failed("nothing selected".into()))?;
            let d = doc.document_mut();
            d.object_mut(active)
                .ok_or_else(|| OpError::Failed("active object is gone".into()))?
                .name = params.name.clone();
            Ok(OpResult::Finished)
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, OpParamsDerive)]
    struct NoopParams {}

    struct EditModeOnly;

    impl Operator for EditModeOnly {
        type Params = NoopParams;
        fn id() -> &'static str {
            "mesh.noop"
        }
        fn label() -> &'static str {
            "No-op"
        }
        fn poll(ctx: &Context, _doc: &Document) -> bool {
            ctx.mode == Mode::EditMesh
        }
        fn execute(
            _params: &Self::Params,
            _doc: &mut DocMut<'_>,
            _ctx: &Context,
        ) -> Result<OpResult, OpError> {
            Ok(OpResult::Finished)
        }
    }

    fn registry() -> OperatorRegistry {
        let mut r = OperatorRegistry::new();
        r.register::<Rename>();
        r.register::<EditModeOnly>();
        r
    }

    fn doc_with_selection() -> Document {
        let mut d = Document::new();
        let id = d.add_object(Object::new("Original", ObjectData::Empty));
        d.selection.add(id);
        d
    }

    #[test]
    fn registration_exposes_the_schema_and_label() {
        let r = registry();
        let info = r.get("object.rename").expect("registered");
        assert_eq!(info.label, "Rename Object");
        assert_eq!(info.schema.len(), 1);
        assert_eq!(info.schema[0].name, "name");
        assert_eq!(info.schema[0].doc, "New name for the active object.");
    }

    #[test]
    fn listing_is_sorted_regardless_of_registration_order() {
        // Menus, the command palette and the scripting surface all read this
        // listing; it must not vary between runs.
        let mut a = OperatorRegistry::new();
        a.register::<Rename>();
        a.register::<EditModeOnly>();
        let mut b = OperatorRegistry::new();
        b.register::<EditModeOnly>();
        b.register::<Rename>();

        let ids = |r: &OperatorRegistry| r.iter().map(|e| e.id).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
        assert_eq!(ids(&a), vec!["mesh.noop", "object.rename"]);
    }

    #[test]
    fn dispatching_by_id_runs_the_operator() {
        let mut doc = doc_with_selection();
        let r = registry();
        let mut params = PropertyMap::new();
        params.set("name", crate::PropertyValue::Text("Renamed".into()));

        let mut edit = DocMut::new(&mut doc);
        assert_eq!(
            r.execute("object.rename", &params, &mut edit, &Context::default()),
            Ok(OpResult::Finished)
        );
        drop(edit);

        let active = doc.selection.active.unwrap();
        assert_eq!(doc.object(active).unwrap().name, "Renamed");
    }

    #[test]
    fn an_unknown_id_is_an_error_rather_than_a_panic() {
        // Keymaps and scripts reference operators by name, so an id that no
        // longer exists must be reported, not fatal.
        let mut doc = Document::new();
        let mut edit = DocMut::new(&mut doc);
        let err = registry()
            .execute(
                "mesh.does_not_exist",
                &PropertyMap::new(),
                &mut edit,
                &Context::default(),
            )
            .unwrap_err();
        assert_eq!(err, OpError::UnknownOperator("mesh.does_not_exist".into()));
    }

    #[test]
    fn poll_gates_execution_on_context() {
        let mut doc = Document::new();
        let r = registry();

        let mut edit = DocMut::new(&mut doc);
        assert_eq!(
            r.execute_default("mesh.noop", &mut edit, &Context { mode: Mode::Object }),
            Err(OpError::PollFailed(
                "mesh.noop".into(),
                poll_reason(
                    "mesh.noop",
                    &Context { mode: Mode::Object },
                    &Document::new()
                )
            ))
        );
        assert_eq!(
            r.execute_default(
                "mesh.noop",
                &mut edit,
                &Context {
                    mode: Mode::EditMesh
                }
            ),
            Ok(OpResult::Finished)
        );
    }

    #[test]
    fn poll_gates_execution_on_document_state() {
        let mut doc = Document::new(); // nothing selected
        let r = registry();
        let mut edit = DocMut::new(&mut doc);
        assert!(matches!(
            r.execute_default("object.rename", &mut edit, &Context::default()),
            Err(OpError::PollFailed(..))
        ));
    }

    #[test]
    fn missing_parameters_come_from_the_defaults() {
        // The durability property, now end to end: a keymap written before a
        // parameter existed must still dispatch.
        let mut doc = doc_with_selection();
        let r = registry();
        let mut edit = DocMut::new(&mut doc);
        r.execute(
            "object.rename",
            &PropertyMap::new(),
            &mut edit,
            &Context::default(),
        )
        .unwrap();
        drop(edit);
        let active = doc.selection.active.unwrap();
        assert_eq!(
            doc.object(active).unwrap().name,
            "",
            "default is an empty name"
        );
    }

    #[test]
    fn registering_the_same_id_replaces_it() {
        // An add-on overriding a built-in is legitimate; the caller is told so it
        // can log it.
        let mut r = OperatorRegistry::new();
        assert!(!r.register::<Rename>());
        assert!(r.register::<Rename>());
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn namespace_filtering_finds_a_menu_group() {
        let r = registry();
        assert_eq!(r.in_namespace("mesh.").count(), 1);
        assert_eq!(r.in_namespace("object.").count(), 1);
        assert_eq!(r.in_namespace("nothing.").count(), 0);
    }

    #[test]
    fn an_operator_that_fails_reports_why() {
        // Distinct from PollFailed: poll passed, then the work itself could not
        // be done. The two need different UI treatment.
        let mut doc = Document::new();
        let id = doc.add_object(Object::new("A", ObjectData::Empty));
        doc.selection.add(id);
        doc.remove_object(id); // selection cleared, so poll now fails
        let mut edit = DocMut::new(&mut doc);
        assert!(matches!(
            registry().execute_default("object.rename", &mut edit, &Context::default()),
            Err(OpError::PollFailed(..))
        ));
    }

    #[test]
    fn i18n_keys_exist_for_the_operator_and_its_parameters() {
        let r = registry();
        let info = r.get("object.rename").unwrap();
        assert_eq!(info.name_key(), "op-object-rename-name");
        assert_eq!(
            info.schema[0].name_key(info.id),
            "op-object-rename-param-name-name"
        );
    }
}

#[cfg(test)]
mod mode_tests {
    use super::Mode;

    #[test]
    fn every_mode_has_a_distinct_tag_and_survives_a_round_trip() {
        // The tags are what a keymap, a script and a saved file name a mode by.
        // Serialising the discriminant instead would make all three depend on
        // declaration order.
        let mut seen: Vec<&str> = Vec::new();
        for mode in Mode::ALL {
            let tag = mode.tag();
            assert!(!seen.contains(&tag), "duplicate tag {tag}");
            seen.push(tag);
            assert_eq!(Mode::from_tag(tag), Some(mode));
            assert!(
                mode.label_key().starts_with("mode-"),
                "{tag} has key {}",
                mode.label_key()
            );
        }
        assert_eq!(Mode::from_tag("nonesuch"), None);
        assert_eq!(Mode::ALL.len(), 8, "ALL must list every variant");
    }

    #[test]
    fn the_modes_do_not_all_answer_the_same_way() {
        // The point of asking the mode rather than comparing it. A set of
        // derivations that answered the same for every mode would be a constant
        // wearing a match, and a test that only checked one mode would not
        // notice.
        assert!(Mode::ALL.iter().any(|m| m.needs_mesh()));
        assert!(Mode::ALL.iter().any(|m| !m.needs_mesh()));
        assert!(Mode::ALL.iter().any(|m| m.shows_elements()));
        assert!(Mode::ALL.iter().any(|m| !m.shows_elements()));

        // And object mode in particular, which is the resting state: it needs
        // nothing selected to enter and shows no element overlays.
        assert!(!Mode::Object.needs_mesh());
        assert!(!Mode::Object.shows_elements());
        assert!(Mode::EditMesh.needs_mesh());
        assert!(Mode::EditMesh.shows_elements());
    }

    #[test]
    fn a_mode_that_needs_a_mesh_cannot_be_entered_without_one() {
        // The rule the header and the application both ask, so that they cannot
        // disagree about it. A menu entry the user can click that then does
        // nothing is what disagreement looks like.
        use crate::mode::Has;
        let mesh = Has {
            mesh: true,
            armature: false,
        };
        let armature = Has {
            mesh: false,
            armature: true,
        };
        assert!(Mode::Object.can_enter(Has::NOTHING));
        assert!(Mode::Object.can_enter(mesh));
        assert!(!Mode::EditMesh.can_enter(Has::NOTHING));
        assert!(Mode::EditMesh.can_enter(mesh));
        assert!(!Mode::EditMesh.can_enter(armature));
        assert!(!Mode::Pose.can_enter(mesh));
        assert!(Mode::Pose.can_enter(armature));

        // Every mode is reachable with everything present, or it could never
        // be entered at all.
        for mode in Mode::ALL {
            assert!(
                mode.can_enter(Has::EVERYTHING),
                "{mode:?} can never be entered"
            );
        }
    }
}
