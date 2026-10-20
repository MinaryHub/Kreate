//! The Claude panel: a conversation with Claude about the open document.
//!
//! Claude works on the document through the same connector an outside client
//! uses — every edit is an operator, every operator an undo step — so what this
//! editor draws is only the conversation: what was asked, what Claude said, and
//! a line for each tool it reached for.
//!
//! The crate cannot run a process, so the application owns the conversation
//! and the child behind it. What the panel owns is the text being typed; what
//! it hands back is a [`Command::ChatSend`], [`Command::ChatStop`] or
//! [`Command::ChatNew`] — and [`Command::ChatAttach`] and
//! [`Command::ChatDetach`] for what goes with the next message, which the
//! application holds too: it reads the files, and the clipboard.

use egui::Ui;

use crate::{Command, CommandQueue};

/// One entry in the conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatRole {
    /// What the user typed.
    User,
    /// What Claude said.
    Claude,
    /// A tool Claude used, and what it was given.
    Tool,
    /// Something went wrong: a tool failed, or Claude could not be reached.
    Error,
    /// A note that is neither: "stopped".
    Note,
    /// What one answer changed in the document, with a button that takes
    /// the whole of it back. The number is the answer's, for
    /// [`Command::ChatRevertTurn`].
    Turn(usize),
}

/// Who the panel talks to Claude as, for the line at its top.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ChatAccount {
    #[default]
    Checking,
    /// Signed in: the account's address, and a line about it (organisation,
    /// plan).
    SignedIn {
        email: String,
        detail: String,
    },
    SignedOut,
    /// Signing in: the page's address once the CLI has said it.
    SigningIn {
        url: Option<String>,
    },
    /// Claude Code cannot be used: why.
    Unavailable(String),
}

/// Something attached to the next message, as the panel shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatAttachment {
    pub name: String,
    /// After the name: "1200 × 800", "PDF, 1.2 MB".
    pub detail: String,
}

/// The panel's translated strings.
pub struct ChatText {
    /// The button that picks files to attach, and what hovering it says.
    pub attach: String,
    pub attach_hint: String,
    /// What hovering an attachment's remove button says.
    pub remove: String,
    /// Shown in the empty input box.
    pub placeholder: String,
    /// Shown when nothing has been said yet.
    pub empty: String,
    pub send: String,
    pub stop: String,
    pub new_chat: String,
    /// The buttons that open the kept conversations and save this one.
    pub history: String,
    pub export: String,
    /// Shown while Claude is answering.
    pub working: String,
    pub checking: String,
    pub signed_out: String,
    pub sign_in: String,
    pub sign_out: String,
    /// Shown while the browser is open for signing in.
    pub signing_in: String,
    pub open_sign_in: String,
    pub code_hint: String,
    pub code_submit: String,
    pub cancel: String,
    /// The button under an answer's changes, and its hover.
    pub revert: String,
    pub revert_hint: String,
    /// The menu item that copies a message, or a block of code in one.
    pub copy: String,
    /// The row a run of tool calls folds under; `{count}` is how many.
    pub tools: String,
}

/// Draws the panel. `input` is what is being typed; Enter sends it and
/// Shift+Enter starts a new line. `busy` is whether Claude is answering.
/// Which assistants the panel can talk to, and which it is talking to.
pub struct ChatProviders<'a> {
    pub names: &'a [&'a str],
    pub current: usize,
    /// The models the assistant can answer with, when it is one that has a
    /// choice (Ollama); empty otherwise.
    pub models: &'a [String],
    pub model: usize,
}

/// The panel itself. `attachments` go with the next message. Returns whether the input box has
/// the keyboard, which is when the application takes Ctrl+V for a picture or
/// files on the clipboard — egui pastes only text.
#[allow(clippy::too_many_arguments)]
pub fn chat(
    ui: &mut Ui,
    text: &ChatText,
    providers: &ChatProviders<'_>,
    account: &ChatAccount,
    messages: &[ChatMessage],
    attachments: &[ChatAttachment],
    input: &mut String,
    code: &mut String,
    busy: bool,
    queue: &mut CommandQueue,
) -> bool {
    let theme = crate::theme::from_ui(ui);

    // Which assistant, first: changing it starts a new conversation, so not
    // while one is answering or signing in.
    let settled = !busy && !matches!(account, ChatAccount::SigningIn { .. });
    ui.horizontal_wrapped(|ui| {
        ui.add_enabled_ui(settled, |ui| {
            let current = providers.names.get(providers.current).copied().unwrap_or("");
            egui::ComboBox::from_id_salt("chat-provider")
                .width(110.0)
                // i18n-exempt: an assistant's name
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for (index, name) in providers.names.iter().enumerate() {
                        // i18n-exempt: an assistant's name
                        if ui.selectable_label(index == providers.current, *name).clicked() && index != providers.current {
                            queue.push(Command::ChatProvider { index });
                        }
                    }
                });
            if !providers.models.is_empty() {
                let current = providers
                    .models
                    .get(providers.model)
                    .map(String::as_str)
                    .unwrap_or("");
                egui::ComboBox::from_id_salt("chat-model")
                    .width(140.0)
                    // i18n-exempt: a model's name
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        for (index, name) in providers.models.iter().enumerate() {
                            // i18n-exempt: a model's name
                            if ui.selectable_label(index == providers.model, name).clicked()
                                && index != providers.model
                            {
                                queue.push(Command::ChatModel { index });
                            }
                        }
                    });
            }
        });
        // To the right when they fit beside the assistant, else on a row
        // of their own.
        let mut buttons = |ui: &mut Ui| {
            if ui
                .add_enabled(
                    !messages.is_empty() && !busy,
                    egui::Button::new(&text.new_chat),
                )
                .clicked()
            {
                queue.push(Command::ChatNew);
            }
            if ui
                .add_enabled(!messages.is_empty(), egui::Button::new(&text.export).frame(false))
                .clicked()
            {
                queue.push(Command::ChatExport);
            }
            if ui.add(egui::Button::new(&text.history).frame(false)).clicked() {
                queue.push(Command::ChatHistory);
            }
        };
        if ui.available_width() >= 190.0 {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), buttons);
        } else {
            ui.end_row();
            buttons(ui);
        }
    });
    draw_account(ui, &theme, text, account, code, queue);
    ui.separator();

    // The input along the bottom, laid out first so the conversation
    // takes exactly what is left above it.
    let focused = egui::Panel::bottom(ui.id().with("claude-chat-composer"))
        .frame(egui::Frame::NONE)
        .resizable(false)
        .show_separator_line(false)
        .show(ui, |ui| {
        // What goes with the next message.
        if !attachments.is_empty() {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                for (index, a) in attachments.iter().enumerate() {
                    egui::Frame::new()
                        .fill(theme.accent_wash)
                        .corner_radius(10.0)
                        .inner_margin(egui::Margin::symmetric(8, 2))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                // i18n-exempt: a file's own name, and its size
                                ui.label(egui::RichText::new(&a.name).color(theme.text));
                                ui.label(
                                    egui::RichText::new(&a.detail)
                                        .small()
                                        .color(theme.text_faint),
                                );
                                // i18n-exempt: a close mark
                                if ui.small_button("×").on_hover_text(&text.remove).clicked() {
                                    queue.push(Command::ChatDetach { index });
                                }
                            });
                        });
                }
            });
        }
        ui.add_space(4.0);
        let id = ui.make_persistent_id("claude-chat-input");
        let mut focused = ui.memory(|m| m.has_focus(id));
        egui::Frame::new()
            .fill(theme.raised)
            .stroke(egui::Stroke::new(
                1.0,
                if focused { theme.accent } else { theme.border_strong },
            ))
            .corner_radius(8.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                let response = ui.add(
                    egui::TextEdit::multiline(input)
                        .id(id)
                        .frame(egui::Frame::NONE)
                        .margin(egui::Margin::ZERO)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY)
                        .hint_text(&text.placeholder)
                        // Enter sends; Shift+Enter is the new line.
                        .return_key(egui::KeyboardShortcut::new(egui::Modifiers::SHIFT, egui::Key::Enter)),
                );
                focused = response.has_focus();
                let entered = response.has_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                // Something attached is worth sending without a word.
                let something = !input.trim().is_empty() || !attachments.is_empty();
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::new(&text.attach).frame(false))
                        .on_hover_text(&text.attach_hint)
                        .clicked()
                    {
                        queue.push(Command::ChatAttach);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let send = if busy {
                            if ui
                                .add(egui::Button::new(&text.stop).min_size(egui::vec2(64.0, 0.0)))
                                .clicked()
                            {
                                queue.push(Command::ChatStop);
                            }
                            false
                        } else {
                            let ink = if something { theme.base } else { theme.text_faint };
                            let clicked = ui
                                .add_enabled(
                                    something,
                                    egui::Button::new(egui::RichText::new(&text.send).color(ink))
                                        .fill(if something { theme.accent } else { theme.sunken })
                                        .min_size(egui::vec2(64.0, 0.0)),
                                )
                                .clicked();
                            clicked || entered
                        };
                        if send && something {
                            queue.push(Command::ChatSend {
                                text: std::mem::take(input).trim().to_string(),
                            });
                            response.request_focus();
                        }
                    });
                });
            });
            ui.add_space(2.0);
            focused
        })
        .inner;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.add_space(4.0);
            if messages.is_empty() {
                draw_empty(ui, &theme, &text.empty);
            }
            let mut index = 0;
            while index < messages.len() {
                // A run of tool lines folds into one row: what Claude did is
                // worth a glance, not a page.
                if messages[index].role == ChatRole::Tool {
                    let end = messages[index..]
                        .iter()
                        .position(|m| m.role != ChatRole::Tool)
                        .map_or(messages.len(), |n| index + n);
                    draw_tools(ui, &theme, text, index, &messages[index..end]);
                    index = end;
                    continue;
                }
                draw_message(ui, &theme, text, &messages[index], queue);
                index += 1;
            }
            if busy {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(egui::RichText::new(&text.working).color(theme.text_dim));
                });
            }
            ui.add_space(6.0);
        });

    focused
}

/// What the panel says before anything has been said.
fn draw_empty(ui: &mut Ui, theme: &crate::Theme, empty: &str) {
    ui.add_space(24.0);
    ui.vertical_centered(|ui| {
        ui.set_max_width(320.0_f32.min(ui.available_width()));
        egui::Frame::new()
            .fill(theme.sunken)
            .corner_radius(8.0)
            .inner_margin(egui::Margin::same(14))
            .show(ui, |ui| {
                ui.add(egui::Label::new(egui::RichText::new(empty).color(theme.text_dim)).wrap());
            });
    });
}

/// A run of tool lines, folded under one row that counts them.
fn draw_tools(
    ui: &mut Ui,
    theme: &crate::Theme,
    text: &ChatText,
    first: usize,
    tools: &[ChatMessage],
) {
    let title = text.tools.replace("{count}", &tools.len().to_string());
    ui.add_space(2.0);
    egui::CollapsingHeader::new(egui::RichText::new(title).small().color(theme.text_faint))
        .id_salt(("chat-tools", first))
        .default_open(false)
        .show(ui, |ui| {
            for tool in tools {
                // i18n-exempt: a tool's name and its input are API text
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(&tool.text)
                            .monospace()
                            .small()
                            .color(theme.text_faint),
                    )
                    .wrap(),
                );
            }
        });
}

/// The account line, and signing in when there is no account.
fn draw_account(
    ui: &mut Ui,
    theme: &crate::Theme,
    text: &ChatText,
    account: &ChatAccount,
    code: &mut String,
    queue: &mut CommandQueue,
) {
    match account {
        ChatAccount::Checking => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(egui::RichText::new(&text.checking).color(theme.text_dim));
            });
        }
        ChatAccount::SignedIn { email, detail } => {
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // A model on this computer has no account to leave.
                    if detail != "(local)"
                        && ui
                            .add(
                                egui::Button::new(egui::RichText::new(&text.sign_out).small())
                                    .frame(false),
                            )
                            .clicked()
                    {
                        queue.push(Command::ChatSignOut);
                    }
                    // What is left of the row, cut short rather than
                    // running under the button.
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        // i18n-exempt: an account's address and its organisation
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(email).small().color(theme.text_dim),
                            )
                            .truncate(),
                        )
                        .on_hover_text(detail);
                        if !detail.is_empty() {
                            // i18n-exempt: the organisation and plan, as Claude names them
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(detail).small().color(theme.text_faint),
                                )
                                .truncate(),
                            );
                        }
                    });
                });
            });
        }
        ChatAccount::SignedOut => {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&text.signed_out).color(theme.text_dim));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(&text.sign_in).clicked() {
                        queue.push(Command::ChatSignIn);
                    }
                });
            });
        }
        ChatAccount::SigningIn { url } => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.add(
                    egui::Label::new(egui::RichText::new(&text.signing_in).color(theme.text_dim))
                        .wrap(),
                );
            });
            if let Some(url) = url {
                ui.hyperlink_to(&text.open_sign_in, url);
            }
            ui.horizontal(|ui| {
                let response = ui.add(
                    egui::TextEdit::singleline(code)
                        .hint_text(&text.code_hint)
                        .desired_width((ui.available_width() - 150.0).max(80.0)),
                );
                let entered =
                    response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if (ui
                    .add_enabled(
                        !code.trim().is_empty(),
                        egui::Button::new(&text.code_submit),
                    )
                    .clicked()
                    || entered)
                    && !code.trim().is_empty()
                {
                    queue.push(Command::ChatSignInCode {
                        code: std::mem::take(code).trim().to_string(),
                    });
                }
                if ui.button(&text.cancel).clicked() {
                    queue.push(Command::ChatSignInCancel);
                }
            });
        }
        ChatAccount::Unavailable(why) => {
            // i18n-exempt: the CLI's own explanation
            ui.add(egui::Label::new(egui::RichText::new(why).color(theme.accent)).wrap());
        }
    }
}

fn draw_message(
    ui: &mut Ui,
    theme: &crate::Theme,
    text: &ChatText,
    message: &ChatMessage,
    queue: &mut CommandQueue,
) {
    match message.role {
        ChatRole::Turn(turn) => {
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(&message.text)
                        .small()
                        .color(theme.text_dim),
                );
                if ui
                    .small_button(&text.revert)
                    .on_hover_text(&text.revert_hint)
                    .clicked()
                {
                    queue.push(Command::ChatRevertTurn { turn });
                }
            });
        }
        ChatRole::User => {
            ui.add_space(10.0);
            // The user's words to the right, in a bubble no wider than most
            // of the panel, as chats are read.
            let most = (ui.available_width() * 0.85).max(80.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                let response = egui::Frame::new()
                    .fill(theme.accent_wash)
                    .corner_radius(egui::CornerRadius {
                        nw: 10,
                        ne: 10,
                        sw: 10,
                        se: 3,
                    })
                    .inner_margin(egui::Margin::symmetric(10, 7))
                    .show(ui, |ui| {
                        ui.set_max_width(most - 20.0);
                        // i18n-exempt: the user's own words
                        ui.add(
                            egui::Label::new(egui::RichText::new(&message.text).color(theme.text))
                                .wrap(),
                        );
                    })
                    .response;
                copy_menu(&response, text, &message.text);
            });
            ui.add_space(4.0);
        }
        ChatRole::Claude => {
            ui.add_space(4.0);
            let response = ui
                .scope(|ui| crate::markdown::show(ui, theme, &message.text, &text.copy))
                .response;
            copy_menu(&response, text, &message.text);
        }
        ChatRole::Tool => {
            draw_tools(ui, theme, text, usize::MAX, std::slice::from_ref(message));
        }
        ChatRole::Error => {
            ui.add_space(4.0);
            egui::Frame::new()
                .stroke(egui::Stroke::new(1.0, theme.accent))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(8, 5))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(
                        egui::Label::new(egui::RichText::new(&message.text).color(theme.accent))
                            .wrap(),
                    );
                });
        }
        ChatRole::Note => {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(&message.text)
                        .italics()
                        .color(theme.text_faint),
                )
                .wrap(),
            );
        }
    }
}

/// Right-click on a message: copy it.
fn copy_menu(response: &egui::Response, text: &ChatText, message: &str) {
    response.context_menu(|ui| {
        if ui.button(&text.copy).clicked() {
            ui.ctx().copy_text(message.to_string());
            ui.close();
        }
    });
}
