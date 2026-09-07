//! The right-click menu and the tooltips, as data.
//!
//! What is checked here is the *content*: that a command's name, sentence and
//! shortcut are all found, that the shortcut is the one the keymap would
//! actually fire, and that the menu offers nothing that cannot run. Whether the
//! popup lands under the cursor is a question for a person looking at the
//! screen; whether it offers the right things is not, and it is the half that
//! goes wrong silently.

use kreate_i18n::Catalog;
use kreate_input::{Context, Keymap};
use kreate_ui::command_info::{describe, description_key, name_key, CommandInfo};

fn catalog() -> Catalog {
    kreate::shell::load_catalog("ko-KR")
}

fn english() -> Catalog {
    kreate::shell::load_catalog("en-US")
}

fn keymap() -> Keymap {
    kreate_input::presets::load(kreate_input::presets::Preset::Kreate)
}

/// Everything the menu and the toolbar can put in front of a user.
///
/// One list, so a command added to either place is covered by every test below
/// rather than by whichever one somebody remembered.
/// Each is the key a menu row or toolbar button resolves to, which for an
/// operator with variants includes the parameters that pick one — see
/// `command_info::variant_id`.
const COMMANDS: &[&str] = &[
    "mesh.translate",
    "mesh.rotate",
    "mesh.scale",
    "mesh.extrude_region",
    "mesh.inset",
    "mesh.bevel",
    "mesh.subdivide",
    "mesh.loop_cut",
    "mesh.bridge",
    "mesh.fill",
    "mesh.delete",
    "mesh.dissolve",
    "mesh.merge",
    "mesh.split",
    "mesh.shade.smooth.true",
    "mesh.shade.smooth.false",
    "mesh.mark_sharp",
    "mesh.normals.action.0",
    "mesh.normals.action.2",
    "mesh.spin",
    "object.translate",
    "object.rotate",
    "object.scale",
    "object.add_primitive",
    "object.duplicate",
    "object.duplicate.linked.false",
    "object.duplicate.linked.true",
    "object.delete",
];

#[test]
fn every_offered_command_has_a_name_and_a_description() {
    // A menu row with no sentence teaches nothing, and one showing a raw
    // operator id teaches the wrong thing. Both are what a missing catalogue
    // entry looks like, so the entries are checked rather than assumed.
    let catalog = catalog();
    for id in COMMANDS {
        assert!(
            catalog.has(&name_key(id)),
            "{id} has no name in the catalogue ({})",
            name_key(id)
        );
        assert!(
            catalog.has(&description_key(id)),
            "{id} has no description ({})",
            description_key(id)
        );
    }
}

#[test]
fn every_offered_command_is_translated_into_korean() {
    // The Korean catalogue is the one this is being built for, so an entry that
    // exists only in English is a gap rather than a fallback working correctly.
    let ko = catalog();
    for id in COMMANDS {
        let name = ko.get(&name_key(id));
        assert_ne!(&name, &name_key(id), "{id} falls back to its key");
        assert!(
            name.chars().any(|c| ('\u{ac00}'..='\u{d7a3}').contains(&c)),
            "{id} is named {name:?}, which has no Korean in it"
        );
    }
}

#[test]
fn a_missing_entry_falls_back_to_the_operators_own_label() {
    // A partly translated build has to stay usable, and a key on screen is not
    // usable. The fallback is the English label the operator carries.
    let info = describe(&catalog(), "nothing.like.this", "Fallback Label", None);
    assert_eq!(info.name, "Fallback Label");
    assert!(info.description.is_empty());
}

#[test]
fn a_description_is_a_sentence_about_the_model() {
    // Not a restatement of the name. "Extrude Region" is the name; what someone
    // who does not know the word needs is what it does to their mesh.
    let en = english();
    for id in COMMANDS {
        let text = en.get(&description_key(id));
        assert!(
            text.len() > 15,
            "{id}'s description is {text:?}, which is too short to say anything"
        );
        assert!(
            text.ends_with('.'),
            "{id}'s description is not a sentence: {text:?}"
        );
    }
}

#[test]
fn a_command_with_a_shortcut_reports_it() {
    let info = describe(
        &catalog(),
        "mesh.extrude_region",
        "Extrude",
        keymap().shortcut_for("mesh.extrude_region", Context::EditMesh),
    );
    assert_eq!(info.shortcut, "E");
    assert!(info.has_tooltip());
    assert!(
        info.tooltip().ends_with("E"),
        "the shortcut should end the tooltip: {:?}",
        info.tooltip()
    );
}

#[test]
fn the_shortcut_shown_is_the_one_that_would_fire() {
    // Resolved through the keymap in the context the command runs in, so a
    // rebound key is reported as the user set it. A hardcoded table would be a
    // second place the bindings live and would go stale the first time anyone
    // touched one — which §2.5 makes a first-class thing to do.
    let mut keymap = keymap();
    keymap.set_override(
        &kreate_input::BindingId::new("mesh.extrude"),
        kreate_input::Override {
            input: Some(kreate_input::Input::key("J")),
            ..Default::default()
        },
    );
    let info = describe(
        &catalog(),
        "mesh.extrude_region",
        "Extrude",
        keymap.shortcut_for("mesh.extrude_region", Context::EditMesh),
    );
    assert_eq!(info.shortcut, "J");
}

#[test]
fn a_command_with_no_shortcut_still_has_a_tooltip() {
    // The sentence alone is worth showing. A command reachable only from the
    // menu is exactly the one a user is least likely to know about.
    let info = describe(
        &catalog(),
        "mesh.split",
        "Split",
        keymap().shortcut_for("mesh.split", Context::EditMesh),
    );
    assert!(!info.description.is_empty());
    assert!(info.has_tooltip());
}

#[test]
fn a_shortcut_is_reported_in_the_mode_it_applies_to() {
    // `mesh.extrude_region` has a key in edit mode and none in object mode. A
    // tooltip that offered `E` on the object-mode toolbar would be telling the
    // user to press a key that does nothing.
    let keymap = keymap();
    assert!(keymap
        .shortcut_for("mesh.extrude_region", Context::EditMesh)
        .is_some());
    assert!(keymap
        .shortcut_for("mesh.extrude_region", Context::ObjectMode)
        .is_none());
}

#[test]
fn a_tooltip_that_would_say_nothing_is_not_shown() {
    let bare = CommandInfo {
        name: "Something".into(),
        description: String::new(),
        shortcut: String::new(),
    };
    assert!(!bare.has_tooltip());
}

#[test]
fn every_offered_command_is_a_registered_operator() {
    // A menu row naming a command that does not exist fails at the click, which
    // is the worst moment to find out — the user has already decided what they
    // want.
    let mut registry = kreate_ops::OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    for id in COMMANDS {
        // The operator is the first two segments; anything after is the
        // parameters that pick a variant.
        let operator: String = id.split('.').take(2).collect::<Vec<_>>().join(".");
        assert!(
            registry.get(&operator).is_some(),
            "{id} names {operator}, which is not a registered operator"
        );
    }
}

#[test]
fn the_menu_groups_are_named() {
    let catalog = catalog();
    for key in [
        "menu-group-transform",
        "menu-group-shape",
        "menu-group-remove",
        "menu-group-shading",
        "menu-group-object",
    ] {
        assert!(catalog.has(key), "{key} is missing");
    }
}

#[test]
fn turning_off_tooltip_translation_leaves_the_interface_alone() {
    // §3.5 ships the two as independent switches because many Korean
    // professionals want an English interface with Korean explanations — the
    // menu names are what they read in tutorials and remember as shortcuts.
    // This checks the switches really are independent.
    let mut catalog = catalog();
    catalog.options = kreate_i18n::TranslationOptions {
        ui: true,
        tooltips: false,
        new_data_names: false,
    };
    let info = describe(&catalog, "mesh.extrude_region", "Extrude", None);
    assert_ne!(
        info.name,
        name_key("mesh.extrude_region"),
        "the interface should still be translated"
    );
    assert_eq!(
        info.description,
        description_key("mesh.extrude_region"),
        "an untranslated tooltip resolves to its key"
    );
}

#[test]
fn a_variant_key_distinguishes_two_rows_of_one_operator() {
    // `mesh.shade` smooths or facets depending on one boolean, and the two are
    // different commands to a user with different names. Keyed by operator
    // alone they would collide and both would fall back to the same label.
    use kreate_reflect::{PropertyMap, PropertyValue};

    let mut smooth = PropertyMap::new();
    smooth.set("smooth", PropertyValue::Bool(true));
    let mut flat = PropertyMap::new();
    flat.set("smooth", PropertyValue::Bool(false));

    let a = kreate_ui::command_info::variant_id("mesh.shade", &smooth);
    let b = kreate_ui::command_info::variant_id("mesh.shade", &flat);
    assert_ne!(a, b, "the two variants share a key");
    assert_eq!(a, "mesh.shade.smooth.true");

    let catalog = catalog();
    let smoothed = describe(&catalog, &a, "fallback", None);
    let faceted = describe(&catalog, &b, "fallback", None);
    assert_ne!(smoothed.name, faceted.name, "both rows are named the same");
    assert_ne!(smoothed.name, "fallback", "the smooth variant has no entry");
    assert_ne!(faceted.name, "fallback", "the flat variant has no entry");
}

#[test]
fn a_command_with_no_variant_parameters_keys_as_itself() {
    use kreate_reflect::PropertyMap;
    assert_eq!(
        kreate_ui::command_info::variant_id("mesh.extrude_region", &PropertyMap::new()),
        "mesh.extrude_region"
    );
}

#[test]
fn a_setting_is_not_a_variant() {
    // An offset or a width is how one command was run, not a different command.
    // Putting it in the key would give every distinct offset its own name, and
    // none of them would resolve.
    use kreate_reflect::{PropertyMap, PropertyValue};
    let mut params = PropertyMap::new();
    params.set("offset", PropertyValue::Float(1.5));
    assert_eq!(
        kreate_ui::command_info::variant_id("mesh.extrude_region", &params),
        "mesh.extrude_region"
    );
}

#[test]
fn each_variant_finds_its_own_shortcut_rather_than_the_others() {
    // `mesh.normals` is bound twice: Ctrl+N recalculates and Shift+N flips.
    // Looked up by operator alone, one of the two rows would be labelled with
    // the key for the *other* command — worse than saying nothing, because the
    // user would press it and get something they did not ask for.
    use kreate_reflect::{PropertyMap, PropertyValue};
    let keymap = keymap();

    let mut recalculate = PropertyMap::new();
    recalculate.set("action", PropertyValue::Enum(0));
    let mut flip = PropertyMap::new();
    flip.set("action", PropertyValue::Enum(2));

    assert_eq!(
        keymap
            .shortcut_for_action("mesh.normals", &recalculate, Context::EditMesh)
            .as_deref(),
        Some("Ctrl+N")
    );
    assert_eq!(
        keymap
            .shortcut_for_action("mesh.normals", &flip, Context::EditMesh)
            .as_deref(),
        Some("Shift+N")
    );
}

#[test]
fn a_row_that_names_no_parameters_takes_the_simplest_binding() {
    // The Extrude row says nothing about `offset`, and the binding carries one.
    // A binding may hold more than the row names; it may not disagree with what
    // the row does name.
    use kreate_reflect::PropertyMap;
    assert_eq!(
        keymap()
            .shortcut_for_action(
                "mesh.extrude_region",
                &PropertyMap::new(),
                Context::EditMesh
            )
            .as_deref(),
        Some("E")
    );
}

#[test]
fn a_variant_with_no_binding_of_its_own_reports_no_shortcut() {
    // Rather than borrowing a sibling's. `mesh.shade` with `smooth = false` has
    // no key in the Kreate preset, and the honest answer is silence.
    use kreate_reflect::{PropertyMap, PropertyValue};
    let mut flat = PropertyMap::new();
    flat.set("smooth", PropertyValue::Bool(false));
    assert!(keymap()
        .shortcut_for_action("mesh.shade", &flat, Context::EditMesh)
        .is_none());
}
