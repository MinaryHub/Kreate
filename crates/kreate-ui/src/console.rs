//! The console: the history as a script you can read, and a line you can type.
//!
//! Blender's Python Console and Info editors in one. Every edit the user makes
//! is already a script line — the history writes itself that way — so the top
//! of this editor is simply that log, newest at the bottom. The line at the
//! bottom runs whatever is typed through the same history, one undo step per
//! call, exactly as though the user had pressed the keys.
//!
//! The crate cannot see a `Document` or a `History`, so the application hands
//! over the lines and receives a [`Command::RunScript`]. What the console owns
//! is the text being typed and the scrollback.

use egui::Ui;

use crate::{Command, CommandQueue};

/// One line in the scrollback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsoleLine {
    pub text: String,
    pub kind: LineKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    /// A call the history holds — an edit the user made, by key or by typing.
    History,
    /// What the user typed, echoed.
    Input,
    /// A complaint about a typed line.
    Error,
    /// A note that is neither: "cancelled", "finished".
    Note,
}

/// The console's translated strings.
pub struct ConsoleText {
    /// Shown in the empty input line.
    pub placeholder: String,
    /// Shown when there is nothing in the scrollback yet.
    pub empty: String,
    /// Shown in place of an error that says what the one before it said.
    pub same_error: String,
}

/// Draws the console. `input` is what is being typed; Enter sends it.
pub fn console(
    ui: &mut Ui,
    text: &ConsoleText,
    lines: &[ConsoleLine],
    input: &mut String,
    queue: &mut CommandQueue,
) {
    let theme = crate::theme::from_ui(ui);
    let input_height = ui.spacing().interact_size.y + 8.0;
    let scrollback_height = (ui.available_height() - input_height).max(0.0);

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .max_height(scrollback_height)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            if lines.is_empty() {
                ui.weak(&text.empty);
            }
            // An error the same as the last one is said once: a run of them is
            // one cause, and the column of repeats is what hides it.
            let mut last_error: Option<&str> = None;
            for line in lines {
                if line.kind == LineKind::Error {
                    if last_error == Some(line.text.as_str()) {
                        ui.label(
                            egui::RichText::new(format!("    {}", text.same_error))
                                .monospace()
                                .color(theme.text_faint),
                        );
                        continue;
                    }
                    last_error = Some(line.text.as_str());
                }
                let colour = match line.kind {
                    LineKind::History => theme.text_dim,
                    LineKind::Input => theme.text,
                    LineKind::Error => theme.accent,
                    LineKind::Note => theme.text_faint,
                };
                let prefix = match line.kind {
                    LineKind::Input => ">>> ",
                    LineKind::Error => "    ",
                    _ => "",
                };
                // i18n-exempt: script text and the user's own typing
                ui.label(
                    egui::RichText::new(format!("{prefix}{}", line.text))
                        .monospace()
                        .color(colour),
                );
            }
        });

    ui.separator();
    ui.horizontal(|ui| {
        // i18n-exempt: the prompt is a symbol every console shares
        ui.monospace(">>>");
        let response = ui.add(
            egui::TextEdit::singleline(input)
                .font(egui::TextStyle::Monospace)
                .hint_text(&text.placeholder)
                .desired_width(f32::INFINITY),
        );
        let entered = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if entered && !input.trim().is_empty() {
            queue.push(Command::RunScript {
                source: std::mem::take(input),
            });
            // Keep typing: the next line goes where the last one went.
            response.request_focus();
        }
    });
}
