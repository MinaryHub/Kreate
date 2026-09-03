//! Viewport navigation is reachable on any mouse.
//!
//! The bindings themselves are data, so this checks the thing that actually
//! matters to someone holding a mouse: can the view be turned, moved and zoomed,
//! whatever hardware they have and whichever preset they chose.
//!
//! The failure this exists for is not subtle but it is easy to ship: navigation
//! bound only to the middle button. Plenty of mice have a wheel that clicks
//! badly or not at all, and a modeller whose view cannot be turned is not a
//! modeller.

use kreate_input::keymap::{Action, EventKind, Modifiers, MouseButton};
use kreate_input::{presets, Context, Input, Preset};

const PRESETS: [Preset; 4] = [Preset::Kreate, Preset::Blender, Preset::Maya, Preset::Max];

fn drag(preset: Preset, button: MouseButton, modifiers: Modifiers) -> Option<Action> {
    presets::load(preset).resolve(
        Context::Viewport,
        &Input::Mouse(button),
        modifiers,
        EventKind::Drag,
    )
}

fn op(preset: Preset, button: MouseButton, modifiers: Modifiers) -> Option<String> {
    drag(preset, button, modifiers).map(|a| a.op_id)
}

#[test]
fn every_preset_can_orbit_pan_and_dolly_without_a_middle_button() {
    // The whole point. Alt with the three buttons works on any mouse, and on a
    // trackpad, and is what Maya users already have in their hands.
    for preset in PRESETS {
        assert_eq!(
            op(preset, MouseButton::Left, Modifiers::alt()).as_deref(),
            Some("view.orbit"),
            "{preset:?} cannot orbit without a middle button"
        );
        assert_eq!(
            op(preset, MouseButton::Right, Modifiers::alt()).as_deref(),
            Some("view.dolly"),
            "{preset:?} cannot dolly without a middle button"
        );
    }
}

#[test]
fn every_preset_zooms_on_the_wheel() {
    // Universal, and nobody's muscle memory disagrees about it.
    for preset in PRESETS {
        let action = presets::load(preset).resolve(
            Context::Viewport,
            &Input::Wheel,
            Modifiers::NONE,
            EventKind::Press,
        );
        assert_eq!(
            action.map(|a| a.op_id).as_deref(),
            Some("view.dolly"),
            "{preset:?} does not zoom on the wheel"
        );
    }
}

#[test]
fn a_plain_left_drag_is_never_navigation() {
    // It is the selection rubber band. A preset that took it for navigation
    // would leave the user unable to select anything by dragging.
    for preset in PRESETS {
        assert_eq!(
            op(preset, MouseButton::Left, Modifiers::NONE),
            None,
            "{preset:?} bound a plain left drag to navigation"
        );
    }
}

#[test]
fn the_default_preset_keeps_its_middle_button_bindings() {
    // The fallback is an addition, not a replacement: anyone with the habit
    // keeps it.
    assert_eq!(
        op(Preset::Kreate, MouseButton::Middle, Modifiers::NONE).as_deref(),
        Some("view.orbit")
    );
    assert_eq!(
        op(Preset::Kreate, MouseButton::Middle, Modifiers::shift()).as_deref(),
        Some("view.pan")
    );
    assert_eq!(
        op(Preset::Kreate, MouseButton::Middle, Modifiers::ctrl()).as_deref(),
        Some("view.dolly")
    );
}

#[test]
fn each_preset_keeps_the_habit_it_exists_for() {
    // A preset that quietly became the same as the others would be worse than no
    // preset: the point is that a user's fingers already know one of them.
    assert_eq!(
        op(Preset::Max, MouseButton::Middle, Modifiers::NONE).as_deref(),
        Some("view.pan"),
        "3ds Max pans on the middle button"
    );
    assert_eq!(
        op(Preset::Maya, MouseButton::Middle, Modifiers::NONE),
        None,
        "Maya navigates only with Alt held"
    );
}

#[test]
fn no_navigation_binding_is_shadowed_by_another() {
    // A binding that can never fire reads to the user as navigation being
    // broken, and it is the kind of thing adding a fallback causes.
    for preset in PRESETS {
        let keymap = presets::load(preset);
        let shadowed: Vec<String> = keymap
            .conflicts()
            .into_iter()
            .filter(|(id, _)| id.0.starts_with("view."))
            .map(|(id, by)| format!("{} taken by {}", id.0, by.0))
            .collect();
        assert!(
            shadowed.is_empty(),
            "{preset:?} has navigation that can never fire: {shadowed:?}"
        );
    }
}
