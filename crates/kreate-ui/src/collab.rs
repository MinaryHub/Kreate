//! File ▸ Shared Session: hosting or joining a scene edited together.
//!
//! The crate draws the window; the application connects. What is typed lives
//! in a [`CollabForm`] the application keeps, and how the session stands comes
//! in each frame as a [`CollabView`].

use egui::Ui;

use crate::{Command, CommandQueue};

/// What the person typed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollabForm {
    pub open: bool,
    /// What the others see them as.
    pub name: String,
    /// Where to join: `host` or `host:port`.
    pub address: String,
    /// The session's key, shared out of band.
    pub key: String,
    /// The port to host on.
    pub port: String,
}

impl Default for CollabForm {
    fn default() -> Self {
        Self {
            open: false,
            // The computer's user, so the window opens ready to host: an
            // empty name held the buttons off with nothing saying why.
            name: std::env::var("USERNAME")
                .or_else(|_| std::env::var("USER"))
                .unwrap_or_default(),
            address: String::new(),
            key: String::new(),
            port: "7878".into(),
        }
    }
}

/// How the session stands, for drawing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CollabView {
    pub live: bool,
    pub hosting: bool,
    /// Where others join.
    pub address: String,
    pub key: String,
    /// This participant's colour.
    pub color: [u8; 3],
    /// Everyone else: name, colour, and what they are editing the inside of.
    pub peers: Vec<(String, [u8; 3], Option<String>)>,
    /// Edits not yet taken by the relay.
    pub waiting: usize,
}

/// The window's translated strings.
pub struct CollabText {
    pub title: String,
    pub intro: String,
    pub name: String,
    pub address: String,
    pub address_hint: String,
    pub key: String,
    pub key_hint: String,
    pub port: String,
    pub host: String,
    pub join: String,
    pub join_warning: String,
    pub leave: String,
    pub connecting: String,
    pub hosting_at: String,
    pub joined_at: String,
    pub people: String,
    pub alone: String,
    pub you: String,
    pub editing: String,
    pub sending: String,
    /// Why Host and Join are off: no name yet.
    pub need_name: String,
    /// Why Join is off: no address yet.
    pub need_address: String,
}

/// Draws the session window when it is open.
pub fn collab_window(
    ctx: &egui::Context,
    text: &CollabText,
    form: &mut CollabForm,
    view: Option<&CollabView>,
    queue: &mut CommandQueue,
) {
    if !form.open {
        return;
    }
    let mut open = true;
    egui::Window::new(&text.title)
        .id(egui::Id::new("collab-window"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(380.0)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(ctx.content_rect().center())
        .show(ctx, |ui| match view {
            Some(view) => draw_live(ui, text, form, view, queue),
            None => draw_form(ui, text, form, queue),
        });
    if !open {
        form.open = false;
    }
}

fn rgb(c: [u8; 3]) -> egui::Color32 {
    egui::Color32::from_rgb(c[0], c[1], c[2])
}

fn draw_form(ui: &mut Ui, text: &CollabText, form: &mut CollabForm, queue: &mut CommandQueue) {
    let theme = crate::theme::from_ui(ui);
    ui.add(egui::Label::new(egui::RichText::new(&text.intro).color(theme.text_dim)).wrap());
    ui.add_space(6.0);
    egui::Grid::new("collab-fields")
        .num_columns(2)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            ui.label(&text.name);
            ui.add(egui::TextEdit::singleline(&mut form.name).desired_width(f32::INFINITY));
            ui.end_row();
            ui.label(&text.key);
            ui.add(
                egui::TextEdit::singleline(&mut form.key)
                    .hint_text(&text.key_hint)
                    .password(true)
                    .desired_width(f32::INFINITY),
            );
            ui.end_row();
        });
    let named = !form.name.trim().is_empty();

    ui.separator();
    ui.horizontal(|ui| {
        ui.label(&text.port);
        ui.add(egui::TextEdit::singleline(&mut form.port).desired_width(60.0));
        let port = form.port.trim().parse::<u16>().ok();
        if ui
            .add_enabled(named && port.is_some(), egui::Button::new(&text.host))
            .on_disabled_hover_text(&text.need_name)
            .clicked()
        {
            queue.push(Command::CollabHost {
                name: form.name.trim().to_string(),
                port: port.unwrap_or(7878),
                key: form.key.clone(),
            });
        }
    });

    ui.separator();
    ui.horizontal(|ui| {
        ui.label(&text.address);
        ui.add(
            egui::TextEdit::singleline(&mut form.address)
                .hint_text(&text.address_hint)
                .desired_width(180.0),
        );
        let ready = named && !form.address.trim().is_empty();
        if ui
            .add_enabled(ready, egui::Button::new(&text.join))
            .on_disabled_hover_text(if named {
                &text.need_address
            } else {
                &text.need_name
            })
            .clicked()
        {
            queue.push(Command::CollabJoin {
                name: form.name.trim().to_string(),
                address: form.address.trim().to_string(),
                key: form.key.clone(),
            });
        }
    });
    ui.add(egui::Label::new(egui::RichText::new(&text.join_warning).color(theme.text_dim)).wrap());
}

fn draw_live(
    ui: &mut Ui,
    text: &CollabText,
    form: &mut CollabForm,
    view: &CollabView,
    queue: &mut CommandQueue,
) {
    let theme = crate::theme::from_ui(ui);
    if !view.live {
        ui.label(&text.connecting);
    } else {
        let at = if view.hosting {
            &text.hosting_at
        } else {
            &text.joined_at
        };
        ui.horizontal(|ui| {
            ui.label(at);
            ui.add(
                egui::Label::new(egui::RichText::new(&view.address).monospace()).selectable(true),
            );
        });
    }
    ui.add_space(4.0);
    ui.label(egui::RichText::new(&text.people).strong());
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("●").color(rgb(view.color))); // i18n-exempt: a colour swatch, not a word
        ui.label(format!("{} ({})", form.name, text.you));
    });
    if view.peers.is_empty() {
        ui.label(egui::RichText::new(&text.alone).color(theme.text_dim));
    }
    for (name, color, editing) in &view.peers {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("●").color(rgb(*color))); // i18n-exempt: a colour swatch, not a word
            ui.label(name);
            if let Some(object) = editing {
                ui.label(
                    egui::RichText::new(format!("{} {object}", text.editing)).color(theme.text_dim),
                );
            }
        });
    }
    if view.waiting > 2 {
        ui.label(egui::RichText::new(&text.sending).color(theme.text_dim));
    }
    ui.add_space(6.0);
    if ui.button(&text.leave).clicked() {
        queue.push(Command::CollabLeave);
    }
}
