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
    /// Shown while Claude is answering.
    pub working: String,
    pub checking: String,
    pub signed_out: String,
    pub sign_in: String,
    /// Shown while the browser is open for signing in.
    pub signing_in: String,
    pub open_sign_in: String,
    pub code_hint: String,
    pub code_submit: String,
    pub cancel: String,
    /// The button under an answer's changes, and its hover.
    pub revert: String,
    pub revert_hint: String,
}

/// Draws the panel. `input` is what is being typed; Enter sends it and
/// Shift+Enter starts a new line. `busy` is whether Claude is answering.
/// Which assistants the panel can talk to, and which it is talking to.
pub struct ChatProviders<'a> {
    pub names: &'a [&'a str],
    pub current: usize,
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
    let row = ui.spacing().interact_size.y;
    let attach_rows = if attachments.is_empty() { 1.0 } else { 2.0 };
    let input_height = row * (3.0 + attach_rows) + 28.0;

    // Which assistant, first: changing it starts a new conversation, so not
    // while one is answering or signing in.
    let settled = !busy && !matches!(account, ChatAccount::SigningIn { .. });
    ui.horizontal(|ui| {
        ui.add_enabled_ui(settled, |ui| {
            let current = providers.names.get(providers.current).copied().unwrap_or("");
            egui::ComboBox::from_id_salt("chat-provider")
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
        });
    });
    draw_account(ui, &theme, text, account, code, queue);
    ui.separator();

    ui.horizontal(|ui| {
        if busy {
            ui.spinner();
            ui.label(egui::RichText::new(&text.working).color(theme.text_dim));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add_enabled(
                    !messages.is_empty() && !busy,
                    egui::Button::new(&text.new_chat),
                )
                .clicked()
            {
                queue.push(Command::ChatNew);
            }
        });
    });
    ui.separator();

    let scrollback_height = (ui.available_height() - input_height).max(0.0);
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .max_height(scrollback_height)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            if messages.is_empty() {
                ui.weak(&text.empty);
            }
            for message in messages {
                draw_message(ui, &theme, text, message, queue);
            }
        });

    ui.separator();
    // What goes with the next message, and the button that adds to it.
    ui.horizontal_wrapped(|ui| {
        if ui
            .button(&text.attach)
            .on_hover_text(&text.attach_hint)
            .clicked()
        {
            queue.push(Command::ChatAttach);
        }
        for (index, a) in attachments.iter().enumerate() {
            egui::Frame::new()
                .fill(theme.accent_wash)
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(6, 2))
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
                        if ui.small_button("x").on_hover_text(&text.remove).clicked() {
                            queue.push(Command::ChatDetach { index });
                        }
                    });
                });
        }
    });
    let id = ui.make_persistent_id("claude-chat-input");
    let mut focused = false;
    ui.horizontal(|ui| {
        let button_width = 64.0;
        let response = ui.add_sized(
            [
                (ui.available_width() - button_width - 8.0).max(40.0),
                row * 3.0,
            ],
            egui::TextEdit::multiline(input)
                .id(id)
                .hint_text(&text.placeholder)
                // Enter sends; Shift+Enter is the new line.
                .return_key(egui::KeyboardShortcut::new(egui::Modifiers::SHIFT, egui::Key::Enter)),
        );
        focused = response.has_focus();
        let entered = response.has_focus()
            && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
        // Something attached is worth sending without a word.
        let something = !input.trim().is_empty() || !attachments.is_empty();
        let send = if busy {
            if ui
                .add_sized([button_width, row], egui::Button::new(&text.stop))
                .clicked()
            {
                queue.push(Command::ChatStop);
            }
            false
        } else {
            let clicked = ui
                .add_enabled_ui(something, |ui| {
                    ui.add_sized([button_width, row], egui::Button::new(&text.send))
                })
                .inner
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
    focused
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
                // i18n-exempt: an account's address and its organisation
                ui.label(egui::RichText::new(email).color(theme.text))
                    .on_hover_text(detail);
                if !detail.is_empty() {
                    // i18n-exempt: the organisation and plan, as Claude names them
                    ui.label(egui::RichText::new(detail).small().color(theme.text_faint));
                }
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
            ui.add_space(6.0);
            egui::Frame::new()
                .fill(theme.accent_wash)
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(8, 6))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    // i18n-exempt: the user's own words
                    ui.label(egui::RichText::new(&message.text).color(theme.text));
                });
            ui.add_space(2.0);
        }
        ChatRole::Claude => {
            ui.add_space(2.0);
            // i18n-exempt: Claude's words
            ui.add(egui::Label::new(egui::RichText::new(&message.text).color(theme.text)).wrap());
        }
        ChatRole::Tool => {
            // i18n-exempt: a tool's name and its input are API text
            ui.add(
                egui::Label::new(
                    egui::RichText::new(&message.text)
                        .monospace()
                        .small()
                        .color(theme.text_faint),
                )
                .wrap(),
            );
        }
        ChatRole::Error => {
            ui.add(egui::Label::new(egui::RichText::new(&message.text).color(theme.accent)).wrap());
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
