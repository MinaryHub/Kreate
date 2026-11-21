//! Help ▸ Report an Issue, Help ▸ Ask a Question, and Help ▸ About.
//!
//! The crate draws the forms; the application opens GitHub with what was
//! written (see `kreate::feedback`). What is typed lives in a
//! [`ContactForm`] the application keeps, so a message half written survives
//! closing the window, and "Open on GitHub" comes back as a
//! [`Command::ContactSend`].

use egui::Ui;

use crate::{Command, CommandQueue};

/// What is being written, in the order the form offers them: a problem (a
/// GitHub issue), a question or an idea (a GitHub discussion).
pub const KINDS: [&str; 3] = ["issue", "question", "idea"];

/// The form and where its message has got to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ContactForm {
    pub open: bool,
    /// An index into [`KINDS`].
    pub kind: usize,
    pub title: String,
    pub message: String,
    /// Put the version, the platform and the graphics card with it.
    pub include_details: bool,
    pub status: ContactStatus,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ContactStatus {
    #[default]
    Writing,
    /// GitHub was opened in the browser at this address.
    Opened(String),
}

impl ContactForm {
    pub fn new() -> Self {
        Self {
            include_details: true,
            ..Self::default()
        }
    }

    /// Opens the form at `kind`, ready for a new message if the last one
    /// went to GitHub.
    pub fn open(&mut self, kind: usize) {
        if matches!(self.status, ContactStatus::Opened(_)) {
            self.title.clear();
            self.message.clear();
            self.status = ContactStatus::Writing;
        }
        self.kind = kind.min(KINDS.len() - 1);
        self.open = true;
    }
}

/// The form's translated strings.
pub struct ContactText {
    pub title: String,
    /// What each kind is for, said under the choice.
    pub intros: [String; 3],
    pub kinds: [String; 3],
    pub subject: String,
    pub subject_hint: String,
    pub message_hint: String,
    pub details: String,
    pub send: String,
    pub close: String,
    pub opened: String,
    pub copy: String,
}

/// Draws the form when it is open. `details` is what "include the
/// version" puts with it, shown so the person sees exactly what goes.
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
        .default_width(460.0)
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
    ui.horizontal(|ui| {
        for (i, label) in text.kinds.iter().enumerate() {
            ui.selectable_value(&mut form.kind, i, label);
        }
    });
    let kind = form.kind.min(KINDS.len() - 1);
    ui.add(egui::Label::new(egui::RichText::new(&text.intros[kind]).color(theme.text_dim)).wrap());
    ui.add_space(6.0);
    ui.add(
        egui::TextEdit::singleline(&mut form.title)
            .hint_text(&text.subject_hint)
            .desired_width(f32::INFINITY),
    )
    .on_hover_text(&text.subject);
    ui.add_space(4.0);
    ui.add(
        egui::TextEdit::multiline(&mut form.message)
            .hint_text(&text.message_hint)
            .desired_rows(8)
            .desired_width(f32::INFINITY),
    );
    ui.checkbox(&mut form.include_details, &text.details);
    if form.include_details {
        // i18n-exempt: the version, platform and graphics card, as sent
        ui.label(egui::RichText::new(details).small().color(theme.text_faint));
    }
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let ready = !form.title.trim().is_empty() && !form.message.trim().is_empty();
        if ui
            .add_enabled(
                ready,
                egui::Button::new(egui::RichText::new(&text.send).strong()),
            )
            .clicked()
        {
            queue.push(Command::ContactSend {
                kind,
                title: form.title.trim().to_string(),
                message: form.message.trim().to_string(),
                include_details: form.include_details,
            });
        }
        if ui.button(&text.close).clicked() {
            form.open = false;
        }
    });
    if let ContactStatus::Opened(url) = &form.status {
        ui.add_space(4.0);
        ui.add(egui::Label::new(egui::RichText::new(&text.opened).color(theme.text)).wrap());
        // The address too, for a computer whose browser did not come up.
        if ui.small_button(&text.copy).clicked() {
            ui.ctx().copy_text(url.clone());
        }
    }
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
