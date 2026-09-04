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
    fn weight(self) -> u32 {
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
            // A modal tool owns the keyboard: nothing outside it may fire, or a
            // stray `S` would start a scale inside a running transform.
            Context::Modal => &[Context::Modal],
        }
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

    /// Bindings that can never fire because an earlier one takes the same
    /// trigger in the same context.
    ///
    /// Reported rather than prevented: a shadowed binding is usually what the
    /// user meant when they remapped something, and refusing the change would be
    /// worse than telling them. The pair is `(shadowed, by)`.
    pub fn conflicts(&self) -> Vec<(BindingId, BindingId)> {
        let all: Vec<Binding> = self.bindings().filter(|b| !b.disabled).collect();
        let mut out = Vec::new();
        for (i, b) in all.iter().enumerate() {
            for a in all.iter().take(i) {
                if a.context == b.context
                    && a.input == b.input
                    && a.modifiers == b.modifiers
                    && a.event == b.event
                {
                    out.push((b.id.clone(), a.id.clone()));
                    break;
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
