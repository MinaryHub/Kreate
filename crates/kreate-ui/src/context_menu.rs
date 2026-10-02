//! The right-click menu, and the tooltips that go on every command.
//!
//! # Why a menu at all, when everything has a key
//!
//! Because nobody knows the keys yet. A modelling application's shortcuts are
//! its whole interface once they are learned, and completely opaque before
//! that — the plan's own §3.2 counts discoverability as one of the costs of
//! building the interface from scratch. A menu under the cursor is where a user
//! looks when they do not know what is possible, and it is the one place the
//! answer can be given *and* the key taught at the same time.
//!
//! That is why every row shows its shortcut. A menu that only runs the command
//! keeps the user coming back to it; a menu that shows the key each time makes
//! itself unnecessary, which is the point.
//!
//! # Why the entries are filtered by `poll`
//!
//! An operator that cannot run is not offered. Greying it out was the
//! alternative and is worse here: a right-click menu is opened *at* something,
//! and a list where most rows are dead makes the user hunt for the live ones.
//! The menu bar can afford to show what is unavailable; a pop-up under the
//! cursor cannot.

use crate::command_info::CommandInfo;
use crate::CommandQueue;
use kreate_reflect::PropertyMap;

/// One row of the menu.
pub struct Entry {
    pub op_id: String,
    /// Parameters the row runs with, so two rows can reach one operator.
    pub params: PropertyMap,
    pub info: CommandInfo,
}

impl Entry {
    pub fn new(op_id: impl Into<String>, info: CommandInfo) -> Self {
        Self {
            op_id: op_id.into(),
            params: PropertyMap::new(),
            info,
        }
    }

    pub fn with_params(mut self, params: PropertyMap) -> Self {
        self.params = params;
        self
    }
}

/// A named group of entries, drawn with a separator between groups.
pub struct Group {
    pub title: String,
    pub entries: Vec<Entry>,
}

/// Draws a command as a button with its shortcut and tooltip.
///
/// The shortcut goes on the right of the row in a dimmer colour, which is where
/// every application in this category puts it and therefore where the eye
/// already looks.
pub fn command_button(ui: &mut egui::Ui, info: &CommandInfo, wide: bool) -> egui::Response {
    let response = ui
        .scope(|ui| {
            if wide {
                // A menu row fills the popup so the shortcut can sit at its far
                // edge. The popup's own width is what bounds it — see
                // `MENU_WIDTH` in the shell — because an unconstrained row
                // stretches to the window and takes the shortcut with it.
                ui.set_min_width(ui.available_width());
            }
            ui.horizontal(|ui| {
                let clicked = ui.button(&info.name).clicked();
                if !info.shortcut.is_empty() {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(&info.shortcut);
                    });
                }
                clicked
            })
            .inner
        })
        .response;

    // The whole row is the hover target, not just the button: a user pointing at
    // the shortcut is pointing at the command.
    let response = response.interact(egui::Sense::click());
    if info.has_tooltip() {
        response.clone().on_hover_text(info.tooltip());
    }
    response
}

/// Queues what choosing `entry` does.
///
/// An application command (opening another menu) is not an operator, and
/// goes the way its key does. Run as an operator it named nothing the
/// registry knows, and the row did nothing.
pub fn choose(entry: &Entry, queue: &mut CommandQueue) {
    if entry.op_id.starts_with("app.") {
        queue.dispatch(entry.op_id.clone(), entry.params.clone());
    } else {
        queue.run(entry.op_id.clone(), entry.params.clone());
    }
}

/// Draws the grouped entries into an open menu, queuing whatever is chosen.
///
/// Returns whether something was chosen, so the caller can close the menu. A
/// menu that stayed open after running a command would put its own popup between
/// the user and the result they asked to see.
pub fn menu_body(ui: &mut egui::Ui, groups: &[Group], queue: &mut CommandQueue) -> bool {
    let mut chosen = false;
    for (index, group) in groups.iter().enumerate() {
        if group.entries.is_empty() {
            continue;
        }
        if index > 0 {
            ui.separator();
        }
        if !group.title.is_empty() {
            ui.weak(&group.title);
        }
        for entry in &group.entries {
            let response = command_button(ui, &entry.info, true);
            if response.clicked() {
                choose(entry, queue);
                chosen = true;
            }
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, shortcut: &str) -> CommandInfo {
        CommandInfo {
            name: name.to_string(),
            description: String::new(),
            shortcut: shortcut.to_string(),
        }
    }

    #[test]
    fn an_entry_carries_its_parameters() {
        // Two rows reaching one operator with different arguments is the same
        // mechanism the keymap uses, and it is why "Add Cube" and "Add Sphere"
        // are not two operators.
        let mut params = PropertyMap::new();
        params.set("shape", kreate_reflect::PropertyValue::Enum(1));
        let entry = Entry::new("object.add_primitive", info("Cube", "A")).with_params(params);
        assert_eq!(entry.op_id, "object.add_primitive");
        assert!(entry.params.get("shape").is_some());
    }

    #[test]
    fn an_application_row_is_dispatched_and_an_operator_row_is_run() {
        // The right-click menu's Add opens the Add menu, which is the
        // application's and not an operator: queued as one, it did nothing.
        let mut queue = CommandQueue::default();
        choose(&Entry::new("app.menu", info("Add...", "Shift A")), &mut queue);
        choose(&Entry::new("object.delete", info("Delete", "X")), &mut queue);
        let commands = queue.drain();
        assert!(matches!(&commands[0], crate::Command::Dispatch { op_id, .. } if op_id == "app.menu"));
        assert!(matches!(&commands[1], crate::Command::RunOperator { op_id, .. } if op_id == "object.delete"));
    }

    #[test]
    fn an_entry_with_no_parameters_starts_empty() {
        let entry = Entry::new("mesh.extrude_region", info("Extrude", "E"));
        assert!(entry.params.is_empty());
    }
}
