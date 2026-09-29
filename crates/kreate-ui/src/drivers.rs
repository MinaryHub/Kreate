//! The Drivers editor: each driver of the selected objects, its expression and
//! its variables.
//!
//! Blender's Drivers editor, its right-hand half — the part where a driver is
//! actually written. A list rather than a curve view, because a driver here is
//! an expression and not a curve: there is nothing to draw but the words.
//!
//! Like every editor it changes nothing itself: it draws a [`DriversView`] and
//! returns [`DriverEdit`]s, which the application turns into the
//! `anim.driver_*` operators.
//!
//! # Typing
//!
//! A text field holds what is being typed in [`DriversState`] and sends it when
//! it loses focus — Enter, Tab or a click elsewhere — so an expression is one
//! undo step and not one per letter, and a half-typed `sin(` is not evaluated
//! on every keystroke.

use std::collections::HashMap;

use egui::{Color32, RichText, Ui};

/// One variable, as the editor shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct VariableView {
    pub name: String,
    /// The object it reads, by identity; `None` for nothing yet.
    pub object: Option<u64>,
    /// `anim.driver_variable_set`'s order: location, rotation, scale.
    pub property: u32,
    pub index: u32,
}

/// One driver.
#[derive(Clone, Debug, PartialEq)]
pub struct DriverView {
    /// "Door: Rotation Z".
    pub label: String,
    /// `driver_target`'s text.
    pub target: String,
    pub expression: String,
    /// What it gives now, or why it cannot.
    pub result: Result<f64, String>,
    pub variables: Vec<VariableView>,
}

/// What one frame of the panel shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DriversView {
    pub drivers: Vec<DriverView>,
    /// Every object a variable could read: identity and name.
    pub objects: Vec<(u64, String)>,
    /// The active object's components with no driver yet: label and target.
    pub addable: Vec<(String, String)>,
}

/// What was asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum DriverEdit {
    Add(String),
    Remove(String),
    Expression { target: String, expression: String },
    AddVariable(String),
    SetVariable { target: String, variable: u32, value: VariableView },
    RemoveVariable { target: String, variable: u32 },
}

/// The words, translated by the caller.
#[derive(Clone, Debug, Default)]
pub struct DriversText {
    pub empty: String,
    pub add: String,
    pub remove: String,
    pub expression: String,
    pub add_variable: String,
    pub remove_variable: String,
    pub none: String,
    /// Location, rotation, scale.
    pub properties: [String; 3],
    pub axes: [String; 3],
}

/// What is being typed, by field.
#[derive(Clone, Debug, Default)]
pub struct DriversState {
    typing: HashMap<(String, usize), String>,
}

/// Which text field: the expression, or variable `n`'s name.
const EXPRESSION: usize = usize::MAX;

/// A text field that keeps what is typed until it loses focus, then sends it.
fn field(ui: &mut Ui, state: &mut DriversState, key: (String, usize), shown: &str, width: f32) -> Option<String> {
    let mut text = state.typing.get(&key).cloned().unwrap_or_else(|| shown.to_string());
    let response = ui.add(egui::TextEdit::singleline(&mut text).desired_width(width).font(egui::TextStyle::Monospace));
    if response.has_focus() {
        state.typing.insert(key, text);
        None
    } else if response.lost_focus() {
        state.typing.remove(&key);
        (text != shown).then_some(text)
    } else {
        state.typing.remove(&key);
        None
    }
}

pub fn drivers_editor(ui: &mut Ui, view: &DriversView, state: &mut DriversState, text: &DriversText) -> Vec<DriverEdit> {
    let mut edits = Vec::new();
    ui.horizontal(|ui| {
        ui.add_enabled_ui(!view.addable.is_empty(), |ui| {
            ui.menu_button(&text.add, |ui| {
                for (label, target) in &view.addable {
                    if ui.button(label).clicked() {
                        edits.push(DriverEdit::Add(target.clone()));
                        ui.close();
                    }
                }
            });
        });
    });
    ui.separator();
    if view.drivers.is_empty() {
        ui.label(RichText::new(&text.empty).weak());
        return edits;
    }
    let object_name = |uid: Option<u64>| -> String {
        uid.and_then(|u| view.objects.iter().find(|(o, _)| *o == u).map(|(_, n)| n.clone()))
            .unwrap_or_else(|| text.none.clone())
    };
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        for driver in &view.drivers {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&driver.label).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button(&text.remove).clicked() {
                            edits.push(DriverEdit::Remove(driver.target.clone()));
                        }
                    });
                });
                ui.horizontal(|ui| {
                    ui.label(&text.expression);
                    let width = ui.available_width();
                    if let Some(expression) =
                        field(ui, state, (driver.target.clone(), EXPRESSION), &driver.expression, width)
                    {
                        edits.push(DriverEdit::Expression { target: driver.target.clone(), expression });
                    }
                });
                match &driver.result {
                    // i18n-exempt: a number, formatted
                    Ok(v) => ui.label(RichText::new(format!("= {v:.4}")).monospace()),
                    Err(e) => ui.label(RichText::new(e).color(Color32::from_rgb(0xe0, 0x60, 0x50))),
                };
                for (i, v) in driver.variables.iter().enumerate() {
                    // Two rows: a Drivers editor is often a side panel, and a
                    // row that runs off its edge widens every row above it.
                    let mut value = v.clone();
                    let mut changed = false;
                    let mut remove = false;
                    ui.horizontal(|ui| {
                        if let Some(name) = field(ui, state, (driver.target.clone(), i), &v.name, 70.0) {
                            value.name = name;
                            changed = true;
                        }
                        egui::ComboBox::from_id_salt((&driver.target, i, "object"))
                            .selected_text(object_name(v.object))
                            .width((ui.available_width() - 8.0).clamp(60.0, 160.0))
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(v.object.is_none(), &text.none).clicked() {
                                    value.object = None;
                                    changed = true;
                                }
                                for (uid, name) in &view.objects {
                                    if ui.selectable_label(v.object == Some(*uid), name).clicked() {
                                        value.object = Some(*uid);
                                        changed = true;
                                    }
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.add_space(12.0);
                        egui::ComboBox::from_id_salt((&driver.target, i, "property"))
                            .selected_text(text.properties[(v.property as usize).min(2)].clone())
                            .width(70.0)
                            .show_ui(ui, |ui| {
                                for (p, label) in text.properties.iter().enumerate() {
                                    if ui.selectable_label(v.property == p as u32, label).clicked() {
                                        value.property = p as u32;
                                        changed = true;
                                    }
                                }
                            });
                        egui::ComboBox::from_id_salt((&driver.target, i, "axis"))
                            .selected_text(text.axes[(v.index as usize).min(2)].clone())
                            .width(36.0)
                            .show_ui(ui, |ui| {
                                for (a, label) in text.axes.iter().enumerate() {
                                    if ui.selectable_label(v.index == a as u32, label).clicked() {
                                        value.index = a as u32;
                                        changed = true;
                                    }
                                }
                            });
                        remove = ui.small_button(&text.remove_variable).clicked();
                    });
                    if remove {
                        edits.push(DriverEdit::RemoveVariable { target: driver.target.clone(), variable: i as u32 });
                    } else if changed && value != *v {
                        edits.push(DriverEdit::SetVariable { target: driver.target.clone(), variable: i as u32, value });
                    }
                }
                if ui.small_button(&text.add_variable).clicked() {
                    edits.push(DriverEdit::AddVariable(driver.target.clone()));
                }
            });
            ui.add_space(4.0);
        }
    });
    edits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> DriversView {
        DriversView {
            drivers: vec![DriverView {
                label: "Door: Rotation Z".into(),
                target: "2/rotation_euler/2".into(),
                expression: "h * 2".into(),
                result: Ok(0.8),
                variables: vec![VariableView { name: "h".into(), object: Some(1), property: 0, index: 0 }],
            }],
            objects: vec![(1, "Handle".into()), (2, "Door".into())],
            addable: vec![("Location X".into(), "2/location/0".into())],
        }
    }

    fn text() -> DriversText {
        DriversText {
            empty: "none".into(),
            add: "Add Driver".into(),
            remove: "Remove".into(),
            expression: "Expr".into(),
            add_variable: "Add Variable".into(),
            remove_variable: "-".into(),
            none: "(none)".into(),
            properties: ["Location".into(), "Rotation".into(), "Scale".into()],
            axes: ["X".into(), "Y".into(), "Z".into()],
        }
    }

    /// Runs frames of the panel with the given events, returning the edits.
    fn frames(state: &mut DriversState, events: Vec<Vec<egui::Event>>) -> Vec<DriverEdit> {
        let ctx = egui::Context::default();
        let mut out = Vec::new();
        for batch in events {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 400.0))),
                events: batch,
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    out.extend(drivers_editor(ui, &view(), state, &text()));
                });
            });
        }
        out
    }

    #[test]
    fn a_quiet_frame_sends_nothing() {
        let mut state = DriversState::default();
        assert!(frames(&mut state, vec![vec![], vec![]]).is_empty());
    }

    #[test]
    fn typing_is_held_until_the_field_is_left() {
        // Focus the expression field with a click, type, and press Enter.
        let mut state = DriversState::default();
        let ctx = egui::Context::default();
        let mut sent = Vec::new();
        let mut run = |events: Vec<egui::Event>, sent: &mut Vec<DriverEdit>| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 400.0))),
                events,
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    sent.extend(drivers_editor(ui, &view(), &mut state, &text()));
                });
            });
        };
        run(vec![], &mut sent);
        // Clicking in the expression row's field: under the "Expr" label, to
        // its right, a little down from the top of the group.
        let at = egui::pos2(300.0, 64.0);
        let click = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        run(vec![egui::Event::PointerMoved(at), click(true)], &mut sent);
        run(vec![click(false)], &mut sent);
        run(vec![egui::Event::Text("+1".into())], &mut sent);
        assert!(sent.is_empty(), "sent while typing: {sent:?}");
        run(
            vec![egui::Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Default::default(),
            }],
            &mut sent,
        );
        assert_eq!(
            sent,
            vec![DriverEdit::Expression { target: "2/rotation_euler/2".into(), expression: "h * 2+1".into() }]
        );
    }
}
