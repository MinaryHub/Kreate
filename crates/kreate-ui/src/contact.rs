//! Help ▸ Contact and Help ▸ About: writing to the developers, and saying
//! which Kreate this is.
//!
//! The crate draws the forms; the application sends. What is typed lives in a
//! [`ContactForm`] the application keeps, so a message half written survives
//! closing the window, and the send comes back as a [`Command::ContactSend`].

use egui::Ui;

use crate::{Command, CommandQueue};

/// What a message is about, in the order the form offers them.
pub const KINDS: [&str; 4] = ["bug", "feature", "question", "other"];

/// The contact form and where its message has got to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ContactForm {
    pub open: bool,
    /// An index into [`KINDS`].
    pub kind: usize,
    pub name: String,
    pub email: String,
    pub message: String,
    /// Send the version and the platform with it.
    pub include_details: bool,
    pub status: ContactStatus,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ContactStatus {
    #[default]
    Writing,
    Sending,
    Sent,
    /// Why it did not go.
    Failed(String),
}

impl ContactForm {
    pub fn new() -> Self {
        Self {
            include_details: true,
            ..Self::default()
        }
    }

    /// Opens the form, ready for a new message if the last one went.
    pub fn open(&mut self) {
        if self.status == ContactStatus::Sent {
            self.message.clear();
            self.status = ContactStatus::Writing;
        }
        self.open = true;
    }
}

/// The contact form's translated strings.
pub struct ContactText {
    pub title: String,
    pub intro: String,
    pub kinds: [String; 4],
    pub name: String,
    pub email: String,
    pub email_hint: String,
    pub message_hint: String,
    pub details: String,
    pub send: String,
    pub close: String,
    pub sending: String,
    pub sent: String,
}

/// Draws the contact window when it is open. `details` is what "include the
/// version" sends, shown so the user sees exactly what goes.
pub fn contact_window(
    ctx: &egui::Context,
    text: &ContactText,
    form: &mut ContactForm,
    details: &str,
    queue: &mut CommandQueue,
) {
    if !form.open {
        return;
    }
    let mut open = true;
    egui::Window::new(&text.title)
        .id(egui::Id::new("contact-window"))
        .open(&mut open)
        .collapsible(false)
        .resizable(true)
        .default_width(440.0)
        // In the middle of the window, clear of the menus it was opened from.
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(ctx.content_rect().center())
        .show(ctx, |ui| draw_form(ui, text, form, details, queue));
    if !open {
        form.open = false;
    }
}

fn draw_form(
    ui: &mut Ui,
    text: &ContactText,
    form: &mut ContactForm,
    details: &str,
    queue: &mut CommandQueue,
) {
    let theme = crate::theme::from_ui(ui);
    let sending = form.status == ContactStatus::Sending;
    ui.add(egui::Label::new(egui::RichText::new(&text.intro).color(theme.text_dim)).wrap());
    ui.add_space(6.0);
    ui.add_enabled_ui(!sending, |ui| {
        ui.horizontal(|ui| {
            for (i, label) in text.kinds.iter().enumerate() {
                ui.selectable_value(&mut form.kind, i, label);
            }
        });
        ui.add_space(4.0);
        egui::Grid::new("contact-fields")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label(&text.name);
                ui.add(egui::TextEdit::singleline(&mut form.name).desired_width(f32::INFINITY));
                ui.end_row();
                ui.label(&text.email);
                ui.add(
                    egui::TextEdit::singleline(&mut form.email)
                        .hint_text(&text.email_hint)
                        .desired_width(f32::INFINITY),
                );
                ui.end_row();
            });
        ui.add_space(4.0);
        ui.add(
            egui::TextEdit::multiline(&mut form.message)
                .hint_text(&text.message_hint)
                .desired_rows(8)
                .desired_width(f32::INFINITY),
        );
        ui.checkbox(&mut form.include_details, &text.details);
        if form.include_details {
            // i18n-exempt: the version and platform, as sent
            ui.label(egui::RichText::new(details).small().color(theme.text_faint));
        }
    });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let ready = !sending && !form.message.trim().is_empty();
        if ui
            .add_enabled(ready, egui::Button::new(&text.send))
            .clicked()
        {
            form.status = ContactStatus::Sending;
            queue.push(Command::ContactSend {
                kind: KINDS[form.kind.min(KINDS.len() - 1)].to_string(),
                name: form.name.trim().to_string(),
                email: form.email.trim().to_string(),
                message: form.message.trim().to_string(),
                include_details: form.include_details,
            });
        }
        if ui.button(&text.close).clicked() {
            form.open = false;
        }
        match &form.status {
            ContactStatus::Writing => {}
            ContactStatus::Sending => {
                ui.spinner();
                ui.label(egui::RichText::new(&text.sending).color(theme.text_dim));
            }
            ContactStatus::Sent => {
                ui.label(egui::RichText::new(&text.sent).color(theme.text));
            }
            ContactStatus::Failed(why) => {
                // i18n-exempt: the service's own explanation
                ui.add(egui::Label::new(egui::RichText::new(why).color(theme.accent)).wrap());
            }
        }
    });
}

/// The About window's translated strings.
pub struct AboutText {
    pub title: String,
    pub version: String,
    pub close: String,
    /// The button that opens the sponsors' page.
    pub donate: String,
}

/// Help ▸ About: the name, the version and the commit.
/// Returns whether the donate button was pressed.
pub fn about_window(
    ctx: &egui::Context,
    text: &AboutText,
    open: &mut bool,
    version: &str,
    platform: &str,
) -> bool {
    let mut donate = false;
    if !*open {
        return false;
    }
    let mut still = true;
    egui::Window::new(&text.title)
        .id(egui::Id::new("about-window"))
        .open(&mut still)
        .collapsible(false)
        .resizable(false)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(ctx.content_rect().center())
        .show(ctx, |ui| {
            // i18n-exempt: the product's name
            ui.heading("Kreate");
            egui::Grid::new("about-grid").num_columns(2).show(ui, |ui| {
                ui.label(&text.version);
                // i18n-exempt: a version and a platform
                ui.label(egui::RichText::new(version).monospace());
                ui.end_row();
                ui.label("");
                // i18n-exempt: a platform
                ui.label(egui::RichText::new(platform).monospace().small());
                ui.end_row();
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button(&text.donate).clicked() {
                    donate = true;
                }
                if ui.button(&text.close).clicked() {
                    *open = false;
                }
            });
        });
    if !still {
        *open = false;
    }
    donate
}
