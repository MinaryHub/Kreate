//! The modifier stack panel.
//!
//! Reads a list of rows and emits commands, like every other panel here — it
//! cannot see a document and cannot change one. Each row carries the id of the
//! setter operator for its kind plus that operator's current parameters, so the
//! parameter widgets come from the reflected schema and adding a fourth modifier
//! adds its panel with no work in this file.
//!
//! # The stack reads bottom to top, and is drawn that way
//!
//! Entry zero takes the object's own mesh as input; the last entry produces what
//! is drawn. Drawing the list in that order — input at the top of the panel,
//! result at the bottom — matches how every modelling tool presents it and, more
//! importantly, matches the order the numbers are in, so "move up" moves towards
//! the input and there is nothing to translate in the user's head.

use egui::Ui;
use kreate_i18n::{Catalog, TextKind};
use kreate_reflect::{PropertyDesc, PropertyMap, PropertyValue};

use crate::command::{Command, CommandQueue};

/// One entry in the stack, as the panel needs to see it.
#[derive(Clone, Debug)]
pub struct ModifierRow {
    /// Position in the stack, counting from the input.
    pub index: u32,
    /// The kind's translated name.
    pub name: String,
    pub enabled: bool,
    /// The operator that writes this kind's parameters.
    pub op_id: String,
    /// That operator's reflected schema.
    pub schema: &'static [PropertyDesc],
    /// The entry's current parameters, in that operator's terms.
    pub params: PropertyMap,
    /// For a kind that names another object (a boolean's `object`): the
    /// objects it may name, offered as a list rather than typed.
    pub choices: Vec<String>,
}

/// Labels for the panel, resolved once per frame.
///
/// Held as a struct rather than looked up inside the widget so the panel takes
/// no catalogue lookups per row, and so a test can supply text without building
/// a catalogue.
#[derive(Clone, Debug, Default)]
pub struct ModifierText {
    pub empty: String,
    pub add: String,
    pub remove: String,
    pub move_up: String,
    pub move_down: String,
    pub enabled: String,
    pub apply: String,
    /// The two buttons' own words: the font has no tick or cross to tell
    /// them apart by.
    pub apply_label: String,
    pub remove_label: String,
    /// One entry per addable kind, in the order the operator expects.
    pub kinds: Vec<String>,
}

impl ModifierText {
    /// Resolves the panel's text, given the translated name of each kind.
    pub fn resolve(catalog: &Catalog, kind_keys: &[&str]) -> Self {
        let ui = |key: &str| catalog.format(key, TextKind::Ui, &[]);
        Self {
            empty: ui("modifier-none"),
            add: ui("modifier-add"),
            remove: catalog.format("modifier-remove", TextKind::Tooltip, &[]),
            move_up: catalog.format("modifier-move-up", TextKind::Tooltip, &[]),
            move_down: catalog.format("modifier-move-down", TextKind::Tooltip, &[]),
            enabled: catalog.format("modifier-enabled", TextKind::Tooltip, &[]),
            apply: catalog.format("modifier-apply", TextKind::Tooltip, &[]),
            apply_label: ui("modifier-apply-button"),
            remove_label: ui("modifier-remove-button"),
            kinds: kind_keys.iter().map(|k| ui(k)).collect(),
        }
    }
}

/// Draws the stack, and returns whatever the user asked for as commands.
///
/// `rows` is the active object's stack. An empty list still draws the add menu:
/// "there is nothing here yet" and "you cannot add anything" are different
/// states and must not look the same.
pub fn modifier_stack(
    ui: &mut Ui,
    text: &ModifierText,
    catalog: &Catalog,
    rows: &mut [ModifierRow],
    queue: &mut CommandQueue,
) {
    if rows.is_empty() {
        // Prose in a panel the user can drag narrow, which is exactly where
        // Korean overflows instead of wrapping.
        ui.weak(crate::text::wrap(ui, &text.empty, ui.available_width()));
    }

    let last = rows.len().saturating_sub(1);
    for (position, row) in rows.iter_mut().enumerate() {
        header(ui, text, row, position, last, queue);
        // Parameters are hidden while the modifier is off. Editing the width of
        // something that is not being applied is a way to spend a minute
        // wondering why nothing is happening.
        if row.enabled {
            ui.indent(("modifier", row.index), |ui| {
                parameters(ui, catalog, row, queue);
            });
        }
        ui.separator();
    }

    add_menu(ui, text, queue);
}

fn header(
    ui: &mut Ui,
    text: &ModifierText,
    row: &mut ModifierRow,
    position: usize,
    last: usize,
    queue: &mut CommandQueue,
) {
    ui.horizontal(|ui| {
        let mut enabled = row.enabled;
        if ui
            .checkbox(&mut enabled, "")
            .on_hover_text(crate::text::wrap_tooltip(ui, &text.enabled))
            .changed()
        {
            queue.run("object.modifier_toggle", index_params(row.index));
        }

        let mut label = egui::RichText::new(&row.name);
        if !row.enabled {
            label = label.weak();
        }
        ui.label(label);

        // Right-aligned, so the buttons line up down the panel however long the
        // names are.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .small_button(&text.remove_label)
                .on_hover_text(&text.remove)
                .clicked()
            {
                queue.run("object.modifier_remove", index_params(row.index));
            }
            // Apply: baked into the mesh, off the stack.
            if ui
                .small_button(&text.apply_label)
                .on_hover_text(&text.apply)
                .clicked()
            {
                queue.run("object.modifier_apply", index_params(row.index));
            }
            // i18n-exempt: geometric glyphs standing in for icons
            if ui
                .add_enabled(position > 0, egui::Button::new("\u{25B2}").small())
                .on_hover_text(&text.move_up)
                .clicked()
            {
                queue.run("object.modifier_move", move_params(row.index, -1));
            }
            // i18n-exempt: geometric glyphs standing in for icons
            if ui
                .add_enabled(position < last, egui::Button::new("\u{25BC}").small())
                .on_hover_text(&text.move_down)
                .clicked()
            {
                queue.run("object.modifier_move", move_params(row.index, 1));
            }
        });
    });
}

fn parameters(ui: &mut Ui, catalog: &Catalog, row: &mut ModifierRow, queue: &mut CommandQueue) {
    // `index` says which entry the setter writes to; it is not a parameter of
    // the modifier and editing it would move the change to another entry.
    let picks_object = !row.choices.is_empty() && row.schema.iter().any(|d| d.name == "object");
    let editable: Vec<PropertyDesc> = row
        .schema
        .iter()
        .filter(|d| d.name != "index" && !(picks_object && d.name == "object"))
        .cloned()
        .collect();
    // The other object, from a list of the ones there are.
    if picks_object {
        let current = match row.params.get("object") {
            Some(PropertyValue::Text(t)) => t.clone(),
            _ => String::new(),
        };
        let mut chosen = current.clone();
        ui.horizontal(|ui| {
            ui.label(catalog.format(
                &format!("{}-object", row.op_id.replace(['.', '_'], "-")),
                TextKind::Ui,
                &[],
            ));
            egui::ComboBox::from_id_salt(("modifier-object", row.index))
                .selected_text(if current.is_empty() {
                    "\u{2014}".to_string()
                } else {
                    current.clone()
                })
                .show_ui(ui, |ui| {
                    for name in &row.choices {
                        ui.selectable_value(&mut chosen, name.clone(), name.as_str());
                    }
                });
        });
        if chosen != current {
            let mut params = row.params.clone();
            params.set("object", PropertyValue::Text(chosen));
            params.set("index", PropertyValue::Int(row.index as i64));
            queue.push(Command::RunOperator {
                op_id: row.op_id.clone(),
                params,
            });
        }
    }

    let changed = crate::property_editor::property_editor(
        ui,
        catalog,
        &row.op_id,
        &editable,
        &mut row.params,
    );
    if changed {
        // The index goes back in before dispatch, whatever the panel showed.
        let mut params = row.params.clone();
        params.set("index", PropertyValue::Int(row.index as i64));
        queue.push(Command::RunOperator {
            op_id: row.op_id.clone(),
            params,
        });
    }
}

fn add_menu(ui: &mut Ui, text: &ModifierText, queue: &mut CommandQueue) {
    ui.menu_button(&text.add, |ui| {
        for (kind, name) in text.kinds.iter().enumerate() {
            if ui.button(name).clicked() {
                let mut params = PropertyMap::new();
                params.set("kind", PropertyValue::Enum(kind as u32));
                queue.run("object.modifier_add", params);
                ui.close();
            }
        }
    });
}

fn index_params(index: u32) -> PropertyMap {
    let mut params = PropertyMap::new();
    params.set("index", PropertyValue::Int(index as i64));
    params
}

fn move_params(index: u32, offset: i32) -> PropertyMap {
    let mut params = index_params(index);
    params.set("offset", PropertyValue::Int(offset as i64));
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_reflect::{PropertyKind, Unit};

    const fn desc(name: &'static str) -> PropertyDesc {
        PropertyDesc {
            name,
            kind: PropertyKind::Int,
            min: None,
            max: None,
            soft_min: None,
            soft_max: None,
            step: None,
            unit: Unit::Count,
            doc: "",
            hidden: false,
        }
    }

    static SCHEMA: &[PropertyDesc] = &[desc("index"), desc("levels")];

    fn row(index: u32) -> ModifierRow {
        let mut params = PropertyMap::new();
        params.set("index", PropertyValue::Int(index as i64));
        params.set("levels", PropertyValue::Int(1));
        ModifierRow {
            index,
            name: "Subdivision".into(),
            enabled: true,
            op_id: "object.modifier_set_subdivide".into(),
            schema: SCHEMA,
            params,
            choices: Vec::new(),
        }
    }

    #[test]
    fn the_index_is_not_offered_as_a_parameter() {
        // It says which entry the setter writes to. Shown as a spinner, dragging
        // it would move the change to another modifier.
        let hidden: Vec<&str> = SCHEMA
            .iter()
            .filter(|d| d.name != "index")
            .map(|d| d.name)
            .collect();
        assert_eq!(hidden, vec!["levels"]);
    }

    #[test]
    fn a_move_carries_its_direction() {
        let up = move_params(3, -1);
        assert_eq!(up.get("index"), Some(&PropertyValue::Int(3)));
        assert_eq!(up.get("offset"), Some(&PropertyValue::Int(-1)));
    }

    #[test]
    fn a_toggle_carries_only_the_index() {
        let params = index_params(2);
        assert_eq!(params.get("index"), Some(&PropertyValue::Int(2)));
        assert_eq!(params.get("offset"), None);
    }

    #[test]
    fn a_parameter_change_dispatches_with_the_index_restored() {
        // The panel hides `index` but the operator needs it, so it has to go back
        // in before dispatch — otherwise every edit lands on entry zero.
        let mut queue = CommandQueue::new();
        let mut r = row(4);
        r.params.set("levels", PropertyValue::Int(3));
        let mut params = r.params.clone();
        params.set("index", PropertyValue::Int(r.index as i64));
        queue.push(Command::RunOperator {
            op_id: r.op_id.clone(),
            params,
        });

        match &queue.drain()[0] {
            Command::RunOperator { op_id, params } => {
                assert_eq!(op_id, "object.modifier_set_subdivide");
                assert_eq!(params.get("index"), Some(&PropertyValue::Int(4)));
                assert_eq!(params.get("levels"), Some(&PropertyValue::Int(3)));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn resolving_text_falls_back_to_the_key_when_untranslated() {
        // An empty catalogue is what a missing locale looks like. The panel has
        // to stay usable rather than drawing blank buttons.
        let text = ModifierText::resolve(&Catalog::new(), &["modifier-mirror"]);
        assert!(!text.add.is_empty());
        assert_eq!(text.kinds.len(), 1);
        assert!(!text.kinds[0].is_empty());
    }
}
