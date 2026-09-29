//! The keymap: what an input event means, given where it happened.
//!
//! # Why this is a data structure and not a `match`
//!
//! Remapping is the feature every professional tool is judged on, and it is not
//! achievable by adding a settings screen to a hard-coded dispatcher. Four
//! things have to be true from the start, and each of them is a decision about
//! the *shape* of this file:
//!
//! 1. **Layered resolution.** A preset, the user's changes on top, then add-ons.
//!    Flattening them at load time is what makes a later preset improvement
//!    invisible to anyone who ever touched their keymap.
//! 2. **The user's changes are a diff against stable ids**, not a copy of the
//!    whole table. Same reason.
//! 3. **Contexts nest, innermost first.** Edit mode's `X` must shadow object
//!    mode's without either knowing about the other.
//! 4. **Navigation is bound here too.** The Maya preset moves orbit from the
//!    middle button to `Alt` + left, and that is impossible if navigation is a
//!    branch in the event handler.
//!
//! # What a binding resolves *to*
//!
//! An [`Action`] — an operator id and a parameter override — not a function
//! pointer. The same `(id, params)` pair the history records and a script emits,
//! so a keystroke, a menu item, a script line and a redo are provably the same
//! thing rather than three implementations that drift.
//!
//! Where an action is *sent* is the application's business: ids beginning
//! `view.` change the viewport camera, which is per-view interface state rather
//! than document data and so is not an operator that undo should record.
//! Everything else goes to the operator registry. One table, two destinations —
//! deliberate, because what matters for remapping is that both are named and
//! parameterised, not that both mutate the document.

use std::collections::BTreeMap;

use kreate_reflect::{PropertyMap, PropertyValue};

/// A physical input.
///
/// Keys are named rather than taken from a window library's enum: this crate
/// must not depend on `winit`, the names are what a keymap file stores, and a
/// key that a future backend spells differently should not silently rebind
/// itself.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Input {
    /// A named key, e.g. `"E"`, `"Tab"`, `"F1"`, `"Digit1"`.
    Key(String),
    Mouse(MouseButton),
    /// Wheel, ignoring direction — a binding acts on both and reads the sign
    /// from the event.
    Wheel,
}

impl Input {
    pub fn key(name: &str) -> Self {
        Input::Key(name.to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Modifier state. `false` means "must not be held", not "don't care" — an
/// unqualified binding really must lose to a modified one, or `Ctrl+X` would
/// fire `X` as well.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

impl Modifiers {
    pub const NONE: Self = Self {
        shift: false,
        ctrl: false,
        alt: false,
    };

    pub fn shift() -> Self {
        Self {
            shift: true,
            ..Self::NONE
        }
    }
    pub fn ctrl() -> Self {
        Self {
            ctrl: true,
            ..Self::NONE
        }
    }
    pub fn alt() -> Self {
        Self {
            alt: true,
            ..Self::NONE
        }
    }
    /// Ctrl and Shift together. The "and the other one" half of a pair of
    /// commands — Save and Save As, Undo and Redo on platforms that spell Redo
    /// that way.
    pub fn ctrl_shift() -> Self {
        Self {
            ctrl: true,
            shift: true,
            ..Self::NONE
        }
    }

    /// How specific this combination is, for ranking two matches.
    pub(crate) fn weight(self) -> u32 {
        self.shift as u32 + self.ctrl as u32 + self.alt as u32
    }
}

/// When a binding fires.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventKind {
    #[default]
    Press,
    Release,
    /// Held and moved. What navigation binds to.
    Drag,
}

/// Where a binding applies. Resolved innermost first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Context {
    /// Everywhere, including inside a modal operator.
    #[default]
    Global,
    /// The 3D viewport, in any mode.
    Viewport,
    ObjectMode,
    EditMesh,
    /// Pose mode: `G` moves bones, not objects.
    Pose,
    /// The pointer is over the UV editor.
    ///
    /// Its own context, and deliberately *not* inside `Viewport`: `G` in the
    /// UV editor moves UVs and `G` in the viewport moves geometry, and a
    /// context that inherited the viewport's would have to decide which one
    /// won on every key. Keeping them siblings makes the answer "whichever
    /// panel the pointer is over", which is what the user already believes.
    UvEditor,
    /// Inside a running modal tool. First-class so `Transform Modal Map` can
    /// bind keys to *modal events* rather than to operators — the reason
    /// remapping works in Blender and not in most of its clones.
    Modal,
}

impl Context {
    /// The chain to search, innermost first.
    ///
    /// Edit mode sits inside the viewport, which sits inside global. A binding
    /// found earlier wins, which is how edit mode's `X` shadows object mode's
    /// without either mentioning the other.
    pub fn chain(self) -> &'static [Context] {
        match self {
            Context::Global => &[Context::Global],
            Context::Viewport => &[Context::Viewport, Context::Global],
            Context::ObjectMode => &[Context::ObjectMode, Context::Viewport, Context::Global],
            Context::EditMesh => &[Context::EditMesh, Context::Viewport, Context::Global],
            Context::Pose => &[Context::Pose, Context::Viewport, Context::Global],
            Context::UvEditor => &[Context::UvEditor, Context::Global],
            // A modal tool owns the keyboard: nothing outside it may fire, or a
            // stray `S` would start a scale inside a running transform.
            Context::Modal => &[Context::Modal],
        }
    }

    /// A stable name, which is how a mode names the context its keys resolve
    /// in without the operator crate depending on this one.
    pub fn tag(self) -> &'static str {
        match self {
            Context::Global => "global",
            Context::Viewport => "viewport",
            Context::ObjectMode => "object_mode",
            Context::EditMesh => "edit_mesh",
            Context::Pose => "pose",
            Context::UvEditor => "uv_editor",
            Context::Modal => "modal",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|c| c.tag() == tag)
    }

    /// Every context there is.
    pub const ALL: &'static [Context] = &[
        Context::Global,
        Context::Viewport,
        Context::ObjectMode,
        Context::EditMesh,
        Context::Pose,
        Context::UvEditor,
        Context::Modal,
    ];

    /// The contexts a keystroke is ever dispatched *in*.
    ///
    /// `Global` and `Viewport` are shared parents, never the active context —
    /// the application is always in object mode, edit mode or a modal tool. So
    /// a binding in a parent is only reachable through a leaf, which is what
    /// makes [`Keymap::conflicts`] able to tell a shadowed binding from a dead
    /// one.
    ///
    /// Derived from [`Self::chain`] rather than listed: a context that appears
    /// in someone else's chain is an ancestor, and anything else is a leaf. A
    /// new context therefore cannot be forgotten here.
    pub fn leaves() -> impl Iterator<Item = Context> {
        Self::ALL.iter().copied().filter(|c| {
            !Self::ALL
                .iter()
                .any(|other| other != c && other.chain().contains(c))
        })
    }
}

/// A stable identifier for a binding.
///
/// The whole point of the layering: the user's changes are stored against these,
/// so a later preset can move, retrigger or relabel a binding and the user's
/// override follows it. A diff against positions in a list could not.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingId(pub String);

impl BindingId {
    pub fn new(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// What a binding does.
#[derive(Clone, Debug, PartialEq)]
pub struct Action {
    pub op_id: String,
    pub params: PropertyMap,
}

impl Action {
    pub fn new(op_id: &str) -> Self {
        Self {
            op_id: op_id.to_string(),
            params: PropertyMap::new(),
        }
    }

    pub fn with(mut self, name: &'static str, value: PropertyValue) -> Self {
        self.params.set(name, value);
        self
    }

    /// True for actions the application routes to the viewport camera rather
    /// than to the operator registry.
    pub fn is_view(&self) -> bool {
        self.op_id.starts_with("view.")
    }
}

/// One entry in a keymap.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    pub id: BindingId,
    pub context: Context,
    pub input: Input,
    pub modifiers: Modifiers,
    pub event: EventKind,
    pub action: Action,
    /// Set by a user override that removes a binding without deleting it, so a
    /// later preset change still reaches them.
    pub disabled: bool,
}

impl Binding {
    pub fn new(id: &str, context: Context, input: Input, action: Action) -> Self {
        Self {
            id: BindingId::new(id),
            context,
            input,
            modifiers: Modifiers::NONE,
            event: EventKind::Press,
            action,
            disabled: false,
        }
    }

    pub fn with_modifiers(mut self, modifiers: Modifiers) -> Self {
        self.modifiers = modifiers;
        self
    }

    /// How this binding reads to a user: `Ctrl+Shift+M`, `Tab`, `MMB`.
    ///
    /// The modifier order is fixed rather than following the struct's fields,
    /// because a shortcut is read as a phrase: `Shift+Ctrl+S` looks like a
    /// different shortcut from `Ctrl+Shift+S` even though it is the same one.
    pub fn describe(&self) -> String {
        let mut text = String::new();
        for (held, name) in [
            (self.modifiers.ctrl, "Ctrl"),
            (self.modifiers.shift, "Shift"),
            (self.modifiers.alt, "Alt"),
        ] {
            if held {
                text.push_str(name);
                text.push('+');
            }
        }
        match &self.input {
            Input::Key(name) => text.push_str(name),
            Input::Mouse(button) => text.push_str(match button {
                MouseButton::Left => "LMB",
                MouseButton::Middle => "MMB",
                MouseButton::Right => "RMB",
            }),
            Input::Wheel => text.push_str("Wheel"),
        }
        if self.event == EventKind::Drag {
            text.push_str(" drag");
        }
        text
    }

    pub fn on(mut self, event: EventKind) -> Self {
        self.event = event;
        self
    }

    fn matches(&self, input: &Input, modifiers: Modifiers, event: EventKind) -> bool {
        !self.disabled && self.input == *input && self.modifiers == modifiers && self.event == event
    }
}

/// A user's change to one binding, stored against its stable id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Override {
    pub input: Option<Input>,
    pub modifiers: Option<Modifiers>,
    pub event: Option<EventKind>,
    pub disabled: Option<bool>,
}

/// The resolved keymap: a preset, the user's diff, and any add-on layers.
#[derive(Clone, Debug, Default)]
pub struct Keymap {
    preset: Vec<Binding>,
    overrides: BTreeMap<BindingId, Override>,
    /// Add-on bindings, searched before the preset so an add-on can take a key.
    addons: Vec<Binding>,
}

impl Keymap {
    pub fn new(preset: Vec<Binding>) -> Self {
        Self {
            preset,
            overrides: BTreeMap::new(),
            addons: Vec::new(),
        }
    }

    pub fn add_addon(&mut self, binding: Binding) {
        self.addons.push(binding);
    }

    /// Records a user change against a stable id.
    pub fn set_override(&mut self, id: &BindingId, change: Override) {
        self.overrides.insert(id.clone(), change);
    }

    pub fn clear_override(&mut self, id: &BindingId) {
        self.overrides.remove(id);
    }

    pub fn overrides(&self) -> &BTreeMap<BindingId, Override> {
        &self.overrides
    }

    /// A binding as the user will experience it: the preset entry with their
    /// change applied.
    fn effective(&self, binding: &Binding) -> Binding {
        let mut out = binding.clone();
        if let Some(change) = self.overrides.get(&binding.id) {
            if let Some(input) = &change.input {
                out.input = input.clone();
            }
            if let Some(m) = change.modifiers {
                out.modifiers = m;
            }
            if let Some(e) = change.event {
                out.event = e;
            }
            if let Some(d) = change.disabled {
                out.disabled = d;
            }
        }
        out
    }

    /// Every binding, add-ons first, with overrides applied.
    pub fn bindings(&self) -> impl Iterator<Item = Binding> + '_ {
        self.addons
            .iter()
            .chain(self.preset.iter())
            .map(|b| self.effective(b))
    }

    /// What an input event means in a context, or `None`.
    ///
    /// Searched innermost context first, and within a context the most specific
    /// modifier combination wins. Without that ranking a plain `A` bound in the
    /// same context as `Ctrl+A` would shadow it or be shadowed depending on
    /// declaration order, which is exactly the kind of behaviour that makes a
    /// keymap feel haunted.
    pub fn resolve(
        &self,
        context: Context,
        input: &Input,
        modifiers: Modifiers,
        event: EventKind,
    ) -> Option<Action> {
        for scope in context.chain() {
            let mut best: Option<Binding> = None;
            for binding in self.bindings() {
                if binding.context != *scope || !binding.matches(input, modifiers, event) {
                    continue;
                }
                let better = best
                    .as_ref()
                    .is_none_or(|b| binding.modifiers.weight() > b.modifiers.weight());
                if better {
                    best = Some(binding);
                }
            }
            if let Some(b) = best {
                return Some(b.action);
            }
        }
        None
    }

    /// The shortcut that runs `op_id` in `context`, if there is one.
    ///
    /// Resolved through the same context chain as a keystroke, so what is
    /// reported is what would actually fire: a tooltip that promised `X` while
    /// an inner context had rebound it would be worse than saying nothing.
    ///
    /// Among several, the *simplest* wins — fewest modifiers, then alphabetical
    /// for stability. An operator reachable by both `E` and `Ctrl+Shift+E` is
    /// introduced by `E`, and a tie that resolved by hash order would make the
    /// tooltip change between runs.
    pub fn shortcut_for(&self, op_id: &str, context: Context) -> Option<String> {
        self.shortcut_for_action(op_id, &PropertyMap::new(), context)
    }

    /// The shortcut that runs `op_id` **with these parameters**, if there is one.
    ///
    /// A binding qualifies when it agrees with `params` on every key `params`
    /// names. It may carry more of its own — the menu entry for Extrude says
    /// nothing about `offset`, and `E` is still its shortcut — but it may not
    /// disagree.
    ///
    /// The distinction is not academic. `mesh.normals` is bound twice: `Ctrl+N`
    /// recalculates and `Shift+N` flips. Looked up by operator alone, the
    /// "Recalculate Normals" row would be labelled with whichever of the two
    /// sorted first, and a menu that tells the user to press the key for a
    /// *different* command is worse than one that says nothing.
    pub fn shortcut_for_action(
        &self,
        op_id: &str,
        params: &PropertyMap,
        context: Context,
    ) -> Option<String> {
        let chain = context.chain();
        let mut best: Option<(usize, u32, String)> = None;
        for binding in self.bindings() {
            if binding.disabled || binding.action.op_id != op_id {
                continue;
            }
            let agrees = params
                .iter()
                .all(|(name, value)| binding.action.params.get(name) == Some(value));
            if !agrees {
                continue;
            }
            let Some(depth) = chain.iter().position(|c| *c == binding.context) else {
                continue;
            };
            let text = binding.describe();
            let weight = binding.modifiers.weight();
            let candidate = (depth, weight, text);
            if best.as_ref().is_none_or(|current| candidate < *current) {
                best = Some(candidate);
            }
        }
        best.map(|(_, _, text)| text)
    }

    /// Bindings that can never fire, each paired with what takes its trigger.
    ///
    /// Two ways a binding dies, and only one of them is obvious. The obvious
    /// one is a duplicate in the same context: the loser is unreachable
    /// everywhere.
    ///
    /// The other is invisible, and it cost this project its grid toggle. An
    /// *outer* binding whose trigger is taken in every inner context that
    /// leads to it is unreachable too: `Viewport` is never the active context,
    /// so a `Viewport` binding that both `ObjectMode` and `EditMesh` rebind can
    /// never fire — and a check that only compared like contexts said nothing
    /// at all. That is precisely what happened when Move moved onto `G`.
    ///
    /// Cross-context shadowing is not itself a fault and is not reported: `X`
    /// deletes an object in one mode and vertices in the other, and each is
    /// reachable in its own mode. What is reported is a binding reachable in
    /// **no** mode.
    ///
    /// `matches` demands exact modifiers, so only an identical trigger
    /// competes: `Shift+M` neither shadows nor is shadowed by `M`.
    ///
    /// Reported rather than prevented: a shadowed binding is usually what
    /// the user meant when they remapped something, and refusing the change
    /// would be worse than telling them. The pair is `(shadowed, by)`.
    pub fn conflicts(&self) -> Vec<(BindingId, BindingId)> {
        let all: Vec<Binding> = self.bindings().filter(|b| !b.disabled).collect();
        let mut out = Vec::new();
        for (i, b) in all.iter().enumerate() {
            let mut in_play = false;
            let mut reachable = false;
            let mut beaten_by: Option<&BindingId> = None;
            for leaf in Context::leaves() {
                let chain = leaf.chain();
                let Some(depth) = chain.iter().position(|c| *c == b.context) else {
                    continue;
                };
                in_play = true;
                // Who else answers this exact trigger, in a scope this leaf
                // searches first? Within one scope `resolve` keeps the first it
                // finds among equals, so an earlier position wins there.
                let winner = all.iter().enumerate().find(|(j, a)| {
                    *j != i
                        && a.input == b.input
                        && a.modifiers == b.modifiers
                        && a.event == b.event
                        && match chain.iter().position(|c| *c == a.context) {
                            Some(d) => d < depth || (d == depth && *j < i),
                            None => false,
                        }
                });
                match winner {
                    Some((_, a)) => beaten_by = beaten_by.or(Some(&a.id)),
                    None => reachable = true,
                }
            }
            if in_play && !reachable {
                if let Some(other) = beaten_by {
                    out.push((b.id.clone(), other.clone()));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;

    fn map() -> Keymap {
        Keymap::new(vec![
            Binding::new(
                "object.add",
                Context::ObjectMode,
                Input::key("A"),
                Action::new("object.add_primitive"),
            ),
            Binding::new(
                "object.select_all",
                Context::ObjectMode,
                Input::key("A"),
                Action::new("object.select_all"),
            )
            .with_modifiers(Modifiers::ctrl()),
            Binding::new(
                "mesh.extrude",
                Context::EditMesh,
                Input::key("E"),
                Action::new("mesh.extrude_region"),
            ),
            Binding::new(
                "global.undo",
                Context::Global,
                Input::key("Z"),
                Action::new("edit.undo"),
            )
            .with_modifiers(Modifiers::ctrl()),
        ])
    }

    #[test]
    fn a_modified_binding_beats_the_plain_one() {
        // Declaration order must not decide this. If it did, `Ctrl+A` would fire
        // `A` as well, depending on which was written first.
        let k = map();
        assert_eq!(
            k.resolve(
                Context::ObjectMode,
                &Input::key("A"),
                Modifiers::ctrl(),
                EventKind::Press
            )
            .unwrap()
            .op_id,
            "object.select_all"
        );
        assert_eq!(
            k.resolve(
                Context::ObjectMode,
                &Input::key("A"),
                Modifiers::NONE,
                EventKind::Press
            )
            .unwrap()
            .op_id,
            "object.add_primitive"
        );
    }

    #[test]
    fn an_outer_context_is_reached_from_an_inner_one() {
        let k = map();
        assert_eq!(
            k.resolve(
                Context::EditMesh,
                &Input::key("Z"),
                Modifiers::ctrl(),
                EventKind::Press
            )
            .unwrap()
            .op_id,
            "edit.undo",
            "undo is global and must work in every mode"
        );
    }

    #[test]
    fn an_edit_mode_binding_does_not_leak_into_object_mode() {
        let k = map();
        assert!(k
            .resolve(
                Context::ObjectMode,
                &Input::key("E"),
                Modifiers::NONE,
                EventKind::Press
            )
            .is_none());
        assert!(k
            .resolve(
                Context::EditMesh,
                &Input::key("E"),
                Modifiers::NONE,
                EventKind::Press
            )
            .is_some());
    }

    #[test]
    fn a_modal_context_admits_nothing_from_outside() {
        // A running transform owns the keyboard. Without this a stray `E` would
        // start an extrude in the middle of a move.
        let k = map();
        assert!(k
            .resolve(
                Context::Modal,
                &Input::key("Z"),
                Modifiers::ctrl(),
                EventKind::Press
            )
            .is_none());
    }

    #[test]
    fn a_user_override_moves_a_binding_and_survives_a_preset_change() {
        // The reason overrides are a diff against stable ids. A user who remaps
        // one key must still receive every later improvement to the rest.
        let mut k = map();
        k.set_override(
            &BindingId::new("mesh.extrude"),
            Override {
                input: Some(Input::key("Q")),
                ..Default::default()
            },
        );
        assert!(k
            .resolve(
                Context::EditMesh,
                &Input::key("E"),
                Modifiers::NONE,
                EventKind::Press
            )
            .is_none());
        assert_eq!(
            k.resolve(
                Context::EditMesh,
                &Input::key("Q"),
                Modifiers::NONE,
                EventKind::Press
            )
            .unwrap()
            .op_id,
            "mesh.extrude_region"
        );

        // A newer preset changes what that binding *does*, keeping its id.
        let mut newer = map();
        newer.preset[2].action = Action::new("mesh.extrude_individual");
        newer.overrides = k.overrides.clone();
        assert_eq!(
            newer
                .resolve(
                    Context::EditMesh,
                    &Input::key("Q"),
                    Modifiers::NONE,
                    EventKind::Press
                )
                .unwrap()
                .op_id,
            "mesh.extrude_individual",
            "the user's key change followed the binding to its new behaviour"
        );
    }

    #[test]
    fn a_disabled_binding_stops_firing_without_being_deleted() {
        let mut k = map();
        k.set_override(
            &BindingId::new("mesh.extrude"),
            Override {
                disabled: Some(true),
                ..Default::default()
            },
        );
        assert!(k
            .resolve(
                Context::EditMesh,
                &Input::key("E"),
                Modifiers::NONE,
                EventKind::Press
            )
            .is_none());

        k.clear_override(&BindingId::new("mesh.extrude"));
        assert!(k
            .resolve(
                Context::EditMesh,
                &Input::key("E"),
                Modifiers::NONE,
                EventKind::Press
            )
            .is_some());
    }

    #[test]
    fn an_addon_can_take_a_key_from_the_preset() {
        let mut k = map();
        k.add_addon(Binding::new(
            "addon.thing",
            Context::EditMesh,
            Input::key("E"),
            Action::new("addon.do_thing"),
        ));
        assert_eq!(
            k.resolve(
                Context::EditMesh,
                &Input::key("E"),
                Modifiers::NONE,
                EventKind::Press
            )
            .unwrap()
            .op_id,
            "addon.do_thing"
        );
        assert_eq!(k.conflicts().len(), 1, "and the collision is reported");
    }

    #[test]
    fn leaves_are_the_contexts_a_keystroke_arrives_in() {
        let leaves: Vec<Context> = Context::leaves().collect();
        assert!(leaves.contains(&Context::ObjectMode));
        assert!(leaves.contains(&Context::EditMesh));
        assert!(leaves.contains(&Context::Modal));
        assert!(leaves.contains(&Context::UvEditor));
        // Parents, never active on their own — which is the whole reason
        // `conflicts` can tell a shadowed binding from a dead one.
        assert!(!leaves.contains(&Context::Viewport));
        assert!(!leaves.contains(&Context::Global));
    }

    #[test]
    fn an_outer_binding_both_modes_rebind_is_reported_dead() {
        // The case that cost the grid toggle. `Viewport` is never the active
        // context, so a binding there that *both* inner modes take can never
        // fire — and a same-context comparison sees nothing wrong, because no
        // two of the three share a context.
        let k = Keymap::new(vec![
            Binding::new(
                "view.grid",
                Context::Viewport,
                Input::key("G"),
                Action::new("view.toggle_grid"),
            ),
            Binding::new(
                "object.translate",
                Context::ObjectMode,
                Input::key("G"),
                Action::new("object.translate"),
            ),
            Binding::new(
                "mesh.translate",
                Context::EditMesh,
                Input::key("G"),
                Action::new("mesh.translate"),
            ),
            // Every mode, or the one left out still reaches the outer key.
            Binding::new(
                "pose.translate",
                Context::Pose,
                Input::key("G"),
                Action::new("pose.translate"),
            ),
        ]);
        let conflicts = k.conflicts();
        assert_eq!(
            conflicts
                .iter()
                .map(|(a, _)| a.0.as_str())
                .collect::<Vec<_>>(),
            vec!["view.grid"],
            "the dead binding is the outer one, and only it"
        );
    }

    #[test]
    fn an_outer_binding_only_one_mode_rebinds_is_still_alive() {
        // Half the previous case. The grid toggle would still be reachable in
        // edit mode, so reporting it would be crying wolf — and a keymap panel
        // that flags working bindings gets ignored, which is how the real one
        // gets missed.
        let k = Keymap::new(vec![
            Binding::new(
                "view.grid",
                Context::Viewport,
                Input::key("G"),
                Action::new("view.toggle_grid"),
            ),
            Binding::new(
                "object.translate",
                Context::ObjectMode,
                Input::key("G"),
                Action::new("object.translate"),
            ),
        ]);
        assert!(k.conflicts().is_empty(), "{:?}", k.conflicts());
        // And it is genuinely reachable, in the mode that did not take the key.
        assert_eq!(
            k.resolve(
                Context::EditMesh,
                &Input::key("G"),
                Modifiers::NONE,
                EventKind::Press
            )
            .unwrap()
            .op_id,
            "view.toggle_grid"
        );
    }

    #[test]
    fn the_same_key_meaning_two_things_in_two_modes_is_not_a_conflict() {
        // `X` deletes an object in one mode and vertices in the other. Both are
        // reachable in their own mode, and neither is shadowed.
        let k = Keymap::new(vec![
            Binding::new(
                "object.delete",
                Context::ObjectMode,
                Input::key("X"),
                Action::new("object.delete"),
            ),
            Binding::new(
                "mesh.delete",
                Context::EditMesh,
                Input::key("X"),
                Action::new("mesh.delete"),
            ),
        ]);
        assert!(k.conflicts().is_empty(), "{:?}", k.conflicts());
    }

    #[test]
    fn a_modifier_neither_shadows_nor_is_shadowed() {
        // `matches` demands exact modifiers, so `M` and `Shift+M` do not
        // compete — which is what lets merge and weld share a letter.
        let k = Keymap::new(vec![
            Binding::new(
                "mesh.merge",
                Context::EditMesh,
                Input::key("M"),
                Action::new("mesh.merge"),
            ),
            Binding::new(
                "mesh.weld",
                Context::EditMesh,
                Input::key("M"),
                Action::new("mesh.merge"),
            )
            .with_modifiers(Modifiers::shift()),
        ]);
        assert!(k.conflicts().is_empty(), "{:?}", k.conflicts());
    }

    #[test]
    fn a_clean_preset_has_no_conflicts() {
        for preset in presets::ALL.iter().copied() {
            let k = presets::load(preset);
            let conflicts = k.conflicts();
            assert!(
                conflicts.is_empty(),
                "{preset:?} has shadowed bindings: {conflicts:?}"
            );
        }
    }

    #[test]
    fn every_preset_can_orbit_pan_and_zoom() {
        // The point of binding navigation here at all. A preset that cannot
        // navigate is not a preset, and the check is cheap.
        for preset in presets::ALL.iter().copied() {
            let k = presets::load(preset);
            let found: Vec<String> = k
                .bindings()
                .filter(|b| b.action.is_view())
                .map(|b| b.action.op_id.clone())
                .collect();
            for wanted in ["view.orbit", "view.pan", "view.dolly"] {
                assert!(
                    found.iter().any(|f| f == wanted),
                    "{preset:?} has no {wanted}: {found:?}"
                );
            }
        }
    }

    #[test]
    fn the_maya_preset_orbits_with_alt_and_the_left_button() {
        // The concrete thing that is impossible when navigation is a branch in
        // the event handler rather than a binding.
        let k = presets::load(presets::Preset::Maya);
        let action = k
            .resolve(
                Context::Viewport,
                &Input::Mouse(MouseButton::Left),
                Modifiers::alt(),
                EventKind::Drag,
            )
            .expect("Alt+LMB must orbit in the Maya preset");
        assert_eq!(action.op_id, "view.orbit");

        // And the two presets really differ — shown on the middle button rather
        // than on Alt+LMB, which every preset now binds so that navigation works
        // on a mouse without a usable middle button.
        let kreate = presets::load(presets::Preset::Kreate);
        let plain_middle = |k: &Keymap| {
            k.resolve(
                Context::Viewport,
                &Input::Mouse(MouseButton::Middle),
                Modifiers::NONE,
                EventKind::Drag,
            )
            .map(|a| a.op_id)
        };
        assert_eq!(plain_middle(&kreate).as_deref(), Some("view.orbit"));
        assert_eq!(
            plain_middle(&k),
            None,
            "Maya navigates only with Alt held, which is the difference"
        );
    }

    #[test]
    fn navigation_is_addressed_the_same_way_as_everything_else() {
        // One table, two destinations. What matters for remapping is that both
        // are named and parameterised, not that both mutate the document.
        let k = presets::load(presets::Preset::Kreate);
        let orbit = k
            .resolve(
                Context::Viewport,
                &Input::Mouse(MouseButton::Middle),
                Modifiers::NONE,
                EventKind::Drag,
            )
            .unwrap();
        assert!(orbit.is_view());
        let extrude = k
            .resolve(
                Context::EditMesh,
                &Input::key("E"),
                Modifiers::NONE,
                EventKind::Press,
            )
            .unwrap();
        assert!(!extrude.is_view());
    }
}

#[cfg(test)]
mod shortcut_tests {
    use super::*;
    use crate::presets::{load, Preset};

    #[test]
    fn a_shortcut_reads_as_a_phrase() {
        let b = Binding::new(
            "test",
            Context::Global,
            Input::key("S"),
            Action::new("file.save"),
        )
        .with_modifiers(Modifiers::ctrl_shift());
        // Ctrl before Shift before Alt, always. `Shift+Ctrl+S` looks like a
        // different shortcut from `Ctrl+Shift+S` even though it is the same one.
        assert_eq!(b.describe(), "Ctrl+Shift+S");
    }

    #[test]
    fn a_mouse_binding_reads_as_a_button() {
        let b = Binding::new(
            "test",
            Context::Viewport,
            Input::Mouse(MouseButton::Middle),
            Action::new("view.orbit"),
        )
        .on(EventKind::Drag);
        assert_eq!(b.describe(), "MMB drag");
    }

    #[test]
    fn the_shortcut_for_an_operator_is_found() {
        let k = load(Preset::Kreate);
        assert_eq!(
            k.shortcut_for("edit.undo", Context::Viewport).as_deref(),
            Some("Ctrl+Z")
        );
        assert_eq!(
            k.shortcut_for("mesh.extrude_region", Context::EditMesh)
                .as_deref(),
            Some("E"),
            "the binding is `mesh.extrude`; the *operator* it runs is              `mesh.extrude_region`, and the lookup is by operator"
        );
    }

    #[test]
    fn the_simplest_of_several_shortcuts_is_reported() {
        // An operator reachable by both `E` and `Ctrl+Shift+E` is introduced by
        // `E`. A tie broken by iteration order would also make a tooltip change
        // between runs, which is why the rule is fewest modifiers then
        // alphabetical rather than "whichever came first".
        let mut k = load(Preset::Kreate);
        k.add_addon(
            Binding::new(
                "test.awkward",
                Context::EditMesh,
                Input::key("E"),
                Action::new("mesh.extrude_region"),
            )
            .with_modifiers(Modifiers::ctrl_shift()),
        );
        assert_eq!(
            k.shortcut_for("mesh.extrude_region", Context::EditMesh)
                .as_deref(),
            Some("E")
        );
    }

    #[test]
    fn a_shortcut_from_an_unreachable_context_is_not_reported() {
        // An edit-mode binding does not apply in object mode, and a tooltip that
        // promised one would be telling the user to press a key that does
        // nothing.
        let k = load(Preset::Kreate);
        assert!(k
            .shortcut_for("mesh.extrude_region", Context::ObjectMode)
            .is_none());
    }

    #[test]
    fn an_operator_with_no_binding_reports_none() {
        let k = load(Preset::Kreate);
        assert!(k
            .shortcut_for("nothing.bound.to.this", Context::Global)
            .is_none());
    }

    #[test]
    fn a_disabled_binding_is_not_reported() {
        // A user who removed a shortcut should not be told it still works.
        let mut k = load(Preset::Kreate);
        k.set_override(
            &BindingId::new("edit.undo"),
            Override {
                disabled: Some(true),
                ..Default::default()
            },
        );
        assert!(k.shortcut_for("edit.undo", Context::Viewport).is_none());
    }

    #[test]
    fn a_remapped_binding_reports_its_new_key() {
        // The whole point of resolving through the keymap rather than hardcoding
        // a table: a user who rebound a key must see the key they chose.
        let mut k = load(Preset::Kreate);
        k.set_override(
            &BindingId::new("mesh.extrude"),
            Override {
                input: Some(Input::key("J")),
                modifiers: Some(Modifiers::alt()),
                ..Default::default()
            },
        );
        assert_eq!(
            k.shortcut_for("mesh.extrude_region", Context::EditMesh)
                .as_deref(),
            Some("Alt+J")
        );
    }
}
