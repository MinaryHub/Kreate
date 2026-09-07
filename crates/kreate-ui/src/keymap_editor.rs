//! The remapping panel.
//!
//! A list of bindings with the preset above it. Two things make it work, and
//! both are properties of the data rather than of this file:
//!
//! - A change is stored as a **diff against a stable id**, so it survives a
//!   preset improvement. The panel therefore edits [`Override`]s, never the
//!   bindings themselves.
//! - A binding that has been changed says so and can be reset, individually.
//!   Without that, a user who has touched anything cannot tell what they have
//!   touched, and "reset everything" becomes the only way back.
//!
//! Conflicts are **shown, not prevented**. A shadowed binding is usually what
//! the user meant when they took a key for something else, and refusing the
//! change would be worse than telling them about it.

use std::collections::BTreeMap;

use egui::{RichText, Ui};
use kreate_input::keymap::{
    Binding, BindingId, Context, EventKind, Input, Keymap, Modifiers, MouseButton, Override,
};
use kreate_input::Preset;

/// What the panel asks the application to do.
#[derive(Clone, Debug, PartialEq)]
pub enum KeymapEdit {
    /// Switch preset. Discards nothing: the user's diff applies to the new one.
    UsePreset(Preset),
    /// Record or replace one binding's override.
    Set(BindingId, Override),
    /// Drop one binding's override, returning it to the preset.
    Reset(BindingId),
    /// Drop every override.
    ResetAll,
}

/// A human-readable trigger, e.g. `Ctrl+Shift+E` or `Alt+Middle drag`.
pub fn trigger_text(binding: &Binding) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if binding.modifiers.ctrl {
        parts.push("Ctrl");
    }
    if binding.modifiers.shift {
        parts.push("Shift");
    }
    if binding.modifiers.alt {
        parts.push("Alt");
    }
    let input = match &binding.input {
        Input::Key(name) => name.clone(),
        Input::Mouse(MouseButton::Left) => "Left".into(),
        Input::Mouse(MouseButton::Middle) => "Middle".into(),
        Input::Mouse(MouseButton::Right) => "Right".into(),
        Input::Wheel => "Wheel".into(),
    };
    let mut out = parts.join("+");
    if !out.is_empty() {
        out.push('+');
    }
    out.push_str(&input);
    match binding.event {
        EventKind::Press => {}
        EventKind::Release => out.push_str(" release"),
        EventKind::Drag => out.push_str(" drag"),
    }
    out
}

/// The translation key for a context's heading.
///
/// A key rather than the text: this crate reads a view model and emits
/// commands, and prose is the caller's to supply. The CI gate that forbids bare
/// strings in the interface is what keeps that true.
pub fn context_key(context: Context) -> &'static str {
    match context {
        Context::Global => "keymap-context-global",
        Context::Viewport => "keymap-context-viewport",
        Context::ObjectMode => "keymap-context-object",
        Context::EditMesh => "keymap-context-edit",
        Context::Modal => "keymap-context-modal",
    }
}

/// Everything the panel says, resolved by the caller.
///
/// The counted lines arrive already interpolated: Korean needs the number and
/// the noun to agree in a way a format string cannot express, which is the whole
/// reason the catalogue is Fluent rather than printf.
#[derive(Clone, Debug, Default)]
pub struct KeymapText {
    pub preset: String,
    /// "3 changed".
    pub changed: String,
    pub reset_all: String,
    /// "2 bindings shadowed". Empty when there are none.
    pub shadowed: String,
    pub reset_hint: String,
    pub shadowed_hint: String,
    pub capture_hint: String,
    /// Shown on the button of the binding that is listening.
    pub capturing: String,
    /// One per entry of [`CONTEXT_ORDER`], in the same order.
    pub contexts: Vec<String>,
}

/// The order contexts are listed in: outermost first, matching how a user thinks
/// about scope rather than how resolution walks it.
pub const CONTEXT_ORDER: [Context; 5] = [
    Context::Global,
    Context::Viewport,
    Context::ObjectMode,
    Context::EditMesh,
    Context::Modal,
];

/// Draws the panel, appending anything the user did to `edits`.
#[allow(clippy::too_many_arguments)]
pub fn keymap_editor(
    ui: &mut Ui,
    keymap: &Keymap,
    preset: Preset,
    overrides: &BTreeMap<BindingId, Override>,
    conflicts: &[(BindingId, BindingId)],
    text: &KeymapText,
    capturing: &mut Option<BindingId>,
    edits: &mut Vec<KeymapEdit>,
) {
    ui.horizontal(|ui| {
        ui.label(&text.preset);
        for candidate in kreate_input::presets::ALL.iter().copied() {
            if ui
                .selectable_label(candidate == preset, candidate.name())
                .clicked()
            {
                edits.push(KeymapEdit::UsePreset(candidate));
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label(&text.changed);
        if ui
            .add_enabled(!overrides.is_empty(), egui::Button::new(&text.reset_all))
            .clicked()
        {
            edits.push(KeymapEdit::ResetAll);
        }
    });

    if !conflicts.is_empty() {
        ui.colored_label(crate::theme::from_ui(ui).accent, &text.shadowed);
    }

    ui.separator();

    let bindings: Vec<Binding> = keymap.bindings().collect();
    egui::ScrollArea::vertical().show(ui, |ui| {
        for (index, context) in CONTEXT_ORDER.into_iter().enumerate() {
            let group: Vec<&Binding> = bindings.iter().filter(|b| b.context == context).collect();
            if group.is_empty() {
                continue;
            }
            ui.add_space(4.0);
            let heading = text.contexts.get(index).cloned().unwrap_or_default();
            ui.label(RichText::new(heading).strong());
            for binding in group {
                row(ui, binding, overrides, conflicts, text, capturing, edits);
            }
        }
    });
}

fn row(
    ui: &mut Ui,
    binding: &Binding,
    overrides: &BTreeMap<BindingId, Override>,
    conflicts: &[(BindingId, BindingId)],
    text: &KeymapText,
    capturing: &mut Option<BindingId>,
    edits: &mut Vec<KeymapEdit>,
) {
    let changed = overrides.contains_key(&binding.id);
    let shadowed = conflicts.iter().any(|(a, _)| *a == binding.id);
    let waiting = capturing.as_ref() == Some(&binding.id);
    let theme = crate::theme::from_ui(ui);

    ui.horizontal(|ui| {
        // The operator, not the binding id: the id is a compatibility handle,
        // and showing it here would be showing the user an implementation
        // detail in place of the thing they are looking for.
        let mut label = RichText::new(&binding.action.op_id);
        if binding.disabled {
            label = label.weak().strikethrough();
        }
        if changed {
            // Informational, not a warning: an overridden binding is the user
            // having done something deliberate.
            label = label.color(theme.info);
        }
        ui.label(label);

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // i18n-exempt: a reset arrow is a symbol, not prose
            if ui
                .add_enabled(changed, egui::Button::new("↺"))
                .on_hover_text(crate::text::wrap_tooltip(ui, &text.reset_hint))
                .clicked()
            {
                edits.push(KeymapEdit::Reset(binding.id.clone()));
            }

            let enabled_now = !binding.disabled;
            let mut enabled = enabled_now;
            if ui.checkbox(&mut enabled, "").changed() {
                let mut change = overrides.get(&binding.id).cloned().unwrap_or_default();
                change.disabled = Some(!enabled);
                edits.push(KeymapEdit::Set(binding.id.clone(), change));
            }

            let trigger = if waiting {
                text.capturing.clone()
            } else {
                trigger_text(binding)
            };
            let mut button = egui::Button::new(trigger);
            if shadowed {
                button = button.fill(theme.accent_wash);
            }
            if ui
                .add(button)
                .on_hover_text(if shadowed {
                    &text.shadowed_hint
                } else {
                    &text.capture_hint
                })
                .clicked()
            {
                // Toggling: clicking the same row again abandons the capture,
                // so a user who changed their mind is not stuck with the next
                // key they happen to press.
                *capturing = if waiting {
                    None
                } else {
                    Some(binding.id.clone())
                };
            }
        });
    });
}

/// Turns a captured key into the override to record.
///
/// Pure, so "does pressing Ctrl+Q while capturing produce Ctrl+Q" is a test
/// rather than something only a human can check. Keeps whatever else the user
/// had already changed about the binding.
pub fn capture(existing: Option<&Override>, key: &str, modifiers: Modifiers) -> Override {
    let mut change = existing.cloned().unwrap_or_default();
    change.input = Some(Input::Key(key.to_string()));
    change.modifiers = Some(modifiers);
    change
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_input::keymap::Action;

    fn binding(input: Input, modifiers: Modifiers, event: EventKind) -> Binding {
        Binding::new(
            "test.thing",
            Context::EditMesh,
            input,
            Action::new("mesh.extrude_region"),
        )
        .with_modifiers(modifiers)
        .on(event)
    }

    #[test]
    fn a_trigger_reads_the_way_a_user_would_say_it() {
        assert_eq!(
            trigger_text(&binding(Input::key("E"), Modifiers::NONE, EventKind::Press)),
            "E"
        );
        assert_eq!(
            trigger_text(&binding(
                Input::key("E"),
                Modifiers::ctrl(),
                EventKind::Press
            )),
            "Ctrl+E"
        );
        assert_eq!(
            trigger_text(&binding(
                Input::Mouse(MouseButton::Middle),
                Modifiers::alt(),
                EventKind::Drag
            )),
            "Alt+Middle drag"
        );
        assert_eq!(
            trigger_text(&binding(Input::Wheel, Modifiers::NONE, EventKind::Press)),
            "Wheel"
        );
    }

    #[test]
    fn modifiers_are_always_named_in_the_same_order() {
        // Two rows showing `Ctrl+Shift+E` and `Shift+Ctrl+E` for the same
        // combination make a list impossible to scan.
        let all = Modifiers {
            shift: true,
            ctrl: true,
            alt: true,
        };
        assert_eq!(
            trigger_text(&binding(Input::key("E"), all, EventKind::Press)),
            "Ctrl+Shift+Alt+E"
        );
    }

    #[test]
    fn capturing_a_key_records_it_with_its_modifiers() {
        let change = capture(None, "Q", Modifiers::ctrl());
        assert_eq!(change.input, Some(Input::key("Q")));
        assert_eq!(change.modifiers, Some(Modifiers::ctrl()));
        assert_eq!(change.disabled, None, "capturing must not disable anything");
    }

    #[test]
    fn capturing_keeps_what_the_user_had_already_changed() {
        // A binding the user disabled and then remapped must stay disabled.
        // Rebuilding the override from scratch silently re-enables it.
        let existing = Override {
            disabled: Some(true),
            event: Some(EventKind::Drag),
            ..Default::default()
        };
        let change = capture(Some(&existing), "Q", Modifiers::NONE);
        assert_eq!(change.disabled, Some(true));
        assert_eq!(change.event, Some(EventKind::Drag));
        assert_eq!(change.input, Some(Input::key("Q")));
    }

    #[test]
    fn every_context_has_a_name_and_a_place_in_the_list() {
        // A context missing from the order would have its bindings silently
        // absent from the panel — invisible, because the panel would still look
        // complete.
        for context in CONTEXT_ORDER {
            assert!(!context_key(context).is_empty());
        }
        assert_eq!(
            CONTEXT_ORDER.len(),
            5,
            "a new context must be added to the order, or its bindings vanish"
        );
    }

    #[test]
    fn every_preset_binding_appears_in_some_group() {
        // The same failure from the other side: nothing in any preset may fall
        // outside the listed contexts.
        for preset in kreate_input::presets::ALL.iter().copied() {
            let keymap = kreate_input::presets::load(preset);
            for binding in keymap.bindings() {
                assert!(
                    CONTEXT_ORDER.contains(&binding.context),
                    "{preset:?}: `{}` is in {:?}, which the panel does not list",
                    binding.id.0,
                    binding.context
                );
            }
        }
    }
}
