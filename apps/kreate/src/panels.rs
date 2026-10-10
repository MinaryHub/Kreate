//! The side panels that talk to the outside: the asset browser's library,
//! downloads and imports, Help ▸ Contact's outbox, and the chat panel's
//! conversation and the turns it can undo.

use super::*;

impl App {
    /// The asset browser: lists asked for, pictures and downloads arrived.
    pub(super) fn serve_assets(&mut self) {
        use kreate::assets::{Event, Kind, Source};
        let Some(library) = &mut self.assets else {
            return;
        };
        let source = Source::ALL
            .get(self.ui.asset_source)
            .copied()
            .unwrap_or(Source::Local);
        let kind = Kind::ALL
            .get(self.ui.asset_kind)
            .copied()
            .unwrap_or(Kind::Model);
        if std::mem::take(&mut self.ui.asset_list_wanted) {
            self.ui.asset_busy = Some(self.ui.catalog.format(
                "asset-loading",
                kreate_i18n::TextKind::Ui,
                &[],
            ));
            self.ui.asset_key_held = kreate::assets::polypizza_key().is_some();
            library.list(
                source,
                kind,
                self.ui.asset_folders.clone(),
                self.ui.asset_query.clone(),
            );
        }
        for id in std::mem::take(&mut self.ui.asset_wanted) {
            if let Some(entry) = self.ui.asset_entries.iter().find(|e| e.id == id) {
                library.thumbnail(entry);
                self.ui.asset_asked.insert(id);
            }
        }
        let mut ready = Vec::new();
        let mut changed = false;
        for event in library.poll() {
            changed = true;
            match event {
                Event::Listed {
                    source: s,
                    kind: k,
                    result,
                } => {
                    // A list for a choice since changed is not the one shown.
                    if (s, k) != (source, kind) {
                        continue;
                    }
                    self.ui.asset_busy = None;
                    match result {
                        Ok(entries) => {
                            self.ui.asset_error = None;
                            self.ui.asset_entries = entries;
                        }
                        Err(why) => self.ui.asset_error = Some(why),
                    }
                }
                Event::Thumbnail { id, image } => {
                    if let Some(image) = image {
                        self.ui.asset_incoming.push((id, image));
                    }
                }
                Event::Progress(said) => {
                    let (key, count) = said.split_once('|').unwrap_or((said.as_str(), ""));
                    self.ui.asset_busy = Some(self.ui.catalog.format(
                        key,
                        kreate_i18n::TextKind::Ui,
                        &[("count", count)],
                    ));
                }
                Event::Fetched { entry, result } => {
                    self.ui.asset_fetching.remove(&entry.id);
                    self.ui.asset_busy = None;
                    // A made model joins the list it will be found in again.
                    if entry.source == Source::Generated {
                        self.ui.asset_generating = false;
                        if source == Source::Generated {
                            self.ui.asset_list_wanted = true;
                        }
                    }
                    match *result {
                        Ok(r) => ready.push((*entry, r)),
                        Err(why) => {
                            if let Some(mut wish) = self.asset_wishes.remove(&entry.id) {
                                wish.answer(Err(why.clone()));
                            }
                            self.ui.asset_error = Some(why);
                        }
                    }
                }
            }
        }
        for (entry, r) in ready {
            self.add_asset(&entry, r);
        }
        if changed {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }

    /// An asset's files are on the disk: add it as its kind says.
    pub(super) fn add_asset(
        &mut self,
        entry: &kreate::assets::Entry,
        ready: kreate::assets::Ready,
    ) {
        use kreate::assets::Ready;
        let mut wish = self.asset_wishes.remove(&entry.id);
        match ready {
            Ready::Model(path) => {
                if let Some(wish) = wish.take() {
                    self.import_wishes.insert(path.clone(), wish);
                }
                self.import_into(&path);
                // Said where the user will see it, and in the console, which
                // keeps it: a CC-BY model is theirs to use if they credit it.
                if let Some(credit) = entry.credit() {
                    let line = self.ui.catalog.format(
                        "asset-credit-owed",
                        kreate_i18n::TextKind::Ui,
                        &[("credit", credit.as_str())],
                    );
                    self.ui.console_log.push(kreate_ui::console::ConsoleLine {
                        text: line.clone(),
                        kind: kreate_ui::console::LineKind::Note,
                    });
                    self.status(&line);
                }
            }
            Ready::Hdri(path) => self.set_world_image_from(&path),
            Ready::Material(maps) => self.add_material_asset(&entry.name, &maps),
        }
        // A material or a light is done now; a model when it is read.
        if let Some(mut wish) = wish {
            let said = self.ui.status.clone().unwrap_or_default();
            wish.answer(Ok(serde_json::json!({ "message": said })));
        }
    }

    /// A material made of texture maps — colour, and roughness, metal and a
    /// normal map when there are — as a node graph, put on the active object.
    ///
    /// The images and the material are added outside the history, as an image
    /// picked for a material is; putting it on the object is the undo step.
    pub(super) fn add_material_asset(&mut self, name: &str, maps: &kreate::assets::Maps) {
        let Some(material) = kreate::assets::material_from_maps(&mut self.doc, name, maps) else {
            return self.status(&format!("cannot read the colour map of {name}"));
        };
        let Some(uid) = self.doc.material_uid(material) else {
            return;
        };
        let has_mesh = self.doc.selection.active.is_some_and(|id| {
            matches!(
                self.doc.object(id).map(|o| &o.data),
                Some(kreate_scene::document::ObjectData::Mesh(_))
            )
        });
        if has_mesh {
            let mut params = PropertyMap::new();
            params.set("slot", PropertyValue::Int(0));
            params.set("uid", PropertyValue::Text(uid.raw().to_string()));
            self.run_op("material.link", params);
            self.status(&format!("{name} is on the active object"));
        } else {
            self.status(&format!(
                "made the material {name}; select a mesh and choose it in the Material tab"
            ));
        }
    }

    /// What became of the messages Help ▸ Contact sent.
    pub(super) fn serve_outbox(&mut self) {
        use kreate_ui::contact::ContactStatus;
        let Some(outbox) = &self.outbox else { return };
        for answer in outbox.poll() {
            self.ui.contact.status = match answer {
                Ok(()) => ContactStatus::Sent,
                Err(why) => ContactStatus::Failed(why),
            };
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Says something to Claude from the Claude panel.
    pub(super) fn chat_send(&mut self, text: String) {
        use kreate_ui::chat::{ChatMessage, ChatRole};
        let Some(chat) = &mut self.chat else { return };
        // The message as the log keeps it: the words, then what went with
        // them, by name.
        let mut shown = text.clone();
        for a in &self.chat_attachments {
            if !shown.is_empty() {
                shown.push('\n');
            }
            // i18n-exempt: a file's own name
            shown.push_str(&format!("[{}]", a.name));
        }
        self.ui.chat_log.push(ChatMessage {
            role: ChatRole::User,
            text: shown,
        });
        match chat.send(&text, &self.chat_attachments) {
            Ok(()) => {
                self.chat_attachments.clear();
                self.chat_turns.push(ChatTurn {
                    start: self.history.cursor(),
                    end: None,
                    objects: self.doc.objects().map(|(_, o)| o.uid.raw()).collect(),
                });
            }
            // Kept, so a message that could not go can be sent another way.
            Err(e) => self.ui.chat_log.push(ChatMessage {
                role: ChatRole::Error,
                text: e,
            }),
        }
        self.ui.chat_busy = chat.busy();
        self.publish_chat_attachments();
    }

    /// The panel's view of what goes with the next message.
    pub(super) fn publish_chat_attachments(&mut self) {
        self.ui.chat_attachments = self
            .chat_attachments
            .iter()
            .map(|a| kreate_ui::chat::ChatAttachment {
                name: a.name.clone(),
                detail: a.describe(),
            })
            .collect();
        self.redraw.set(RedrawReason::UI);
    }

    /// Adds files to the next chat message. One that cannot go is said in
    /// the conversation, and the rest still go.
    pub(super) fn attach_to_chat(
        &mut self,
        attachments: Vec<Result<kreate::attach::Attachment, String>>,
    ) {
        use kreate_ui::chat::{ChatMessage, ChatRole};
        for a in attachments {
            match a {
                Ok(a) => self.chat_attachments.push(a),
                Err(e) => self.ui.chat_log.push(ChatMessage {
                    role: ChatRole::Error,
                    text: e,
                }),
            }
        }
        self.publish_chat_attachments();
    }

    /// The chat panel's Attach button.
    pub(super) fn pick_chat_attachments(&mut self) {
        let Some(paths) = rfd::FileDialog::new()
            .add_filter(
                "Pictures, PDFs and text",
                &[
                    "png", "jpg", "jpeg", "gif", "webp", "pdf", "txt", "md", "json", "csv", "obj",
                    "gltf", "usda",
                ],
            )
            .add_filter("All files", &["*"])
            .set_title("Attach")
            .pick_files()
        else {
            return;
        };
        let read = paths
            .iter()
            .map(|p| kreate::attach::Attachment::from_path(p))
            .collect();
        self.attach_to_chat(read);
    }

    /// Ctrl+V in the chat's box: files copied in the file manager, or a
    /// picture — a screenshot, an image copied from a browser.
    pub(super) fn paste_into_chat(&mut self) {
        let Ok(mut clipboard) = arboard::Clipboard::new() else {
            return;
        };
        if let Ok(paths) = clipboard.get().file_list() {
            if !paths.is_empty() {
                let read = paths
                    .iter()
                    .map(|p| kreate::attach::Attachment::from_path(p))
                    .collect();
                self.attach_to_chat(read);
                return;
            }
        }
        if let Ok(image) = clipboard.get_image() {
            let n = self
                .chat_attachments
                .iter()
                .filter(|a| a.is_image())
                .count()
                + 1;
            let pasted = kreate::attach::Attachment::from_rgba(
                format!("pasted-{n}"),
                image.width as u32,
                image.height as u32,
                image.bytes.into_owned(),
            );
            self.attach_to_chat(vec![pasted]);
        }
    }

    /// Moves what Claude has said since the last wake into the panel.
    pub(super) fn serve_chat(&mut self) {
        use kreate::claude::Event;
        use kreate_ui::chat::{ChatMessage, ChatRole};
        let Some(chat) = &mut self.chat else { return };
        let mut answered = false;
        for event in chat.poll() {
            answered |= matches!(event, Event::Done(_));
            let (role, text) = match event {
                Event::Session(_) | Event::Done(None) => {
                    self.chat_streaming = false;
                    continue;
                }
                // A streamed reply: onto the end of what it is saying, while
                // it is still saying it.
                Event::TextPart(part) => {
                    match self.ui.chat_log.last_mut() {
                        Some(last) if last.role == ChatRole::Claude && self.chat_streaming => {
                            last.text.push_str(&part)
                        }
                        _ => self.ui.chat_log.push(ChatMessage {
                            role: ChatRole::Claude,
                            text: part,
                        }),
                    }
                    self.chat_streaming = true;
                    continue;
                }
                Event::Account(account) => {
                    use kreate::claude::Account;
                    use kreate_ui::chat::ChatAccount;
                    // A sign-in waiting on the browser keeps its box until it
                    // ends, whatever an earlier question answers.
                    if chat.signing_in() {
                        continue;
                    }
                    self.ui.chat_account = match account {
                        Account::Checking => ChatAccount::Checking,
                        Account::SignedOut => ChatAccount::SignedOut,
                        Account::Unavailable(why) => ChatAccount::Unavailable(why),
                        Account::SignedIn {
                            email,
                            organisation,
                            plan,
                        } => ChatAccount::SignedIn {
                            email,
                            detail: [organisation, plan.map(|p| format!("({p})"))]
                                .into_iter()
                                .flatten()
                                .collect::<Vec<_>>()
                                .join(" "),
                        },
                    };
                    continue;
                }
                Event::SignInUrl(url) => {
                    self.ui.chat_account =
                        kreate_ui::chat::ChatAccount::SigningIn { url: Some(url) };
                    continue;
                }
                Event::SignedIn(None) => {
                    self.ui.chat_account = kreate_ui::chat::ChatAccount::Checking;
                    continue;
                }
                Event::SignedIn(Some(why)) if why == "stopped" => {
                    self.ui.chat_account = kreate_ui::chat::ChatAccount::Checking;
                    continue;
                }
                Event::SignedIn(Some(why)) => {
                    self.ui.chat_account = kreate_ui::chat::ChatAccount::Checking;
                    (ChatRole::Error, why)
                }
                Event::Text(text) => (ChatRole::Claude, text),
                Event::Tool { name, detail } if detail.is_empty() => (ChatRole::Tool, name),
                Event::Tool { name, detail } => (ChatRole::Tool, format!("{name}: {detail}")),
                Event::ToolError(text) => (ChatRole::Error, text),
                Event::Done(Some(why)) if why == "stopped" => {
                    (ChatRole::Note, self.ui.catalog.get("chat-stopped"))
                }
                Event::Done(Some(why)) => (ChatRole::Error, why),
            };
            self.ui.chat_log.push(ChatMessage { role, text });
            self.chat_streaming = false;
        }
        self.ui.chat_busy = chat.busy();
        if answered {
            self.finish_chat_turn();
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// An answer has ended: what it did to the document, said under it with
    /// a button that takes it all back. Nothing is said for an answer that
    /// changed nothing.
    pub(super) fn finish_chat_turn(&mut self) {
        use kreate_ui::chat::{ChatMessage, ChatRole};
        let end = self.history.cursor();
        let Some((index, turn)) = self
            .chat_turns
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(_, t)| t.end.is_none())
        else {
            return;
        };
        turn.end = Some(end);
        if end <= turn.start {
            return;
        }
        let now: std::collections::HashSet<u64> =
            self.doc.objects().map(|(_, o)| o.uid.raw()).collect();
        let added = now.difference(&turn.objects).count().to_string();
        let removed = turn.objects.difference(&now).count().to_string();
        let steps = (end - turn.start).to_string();
        let text = self.ui.catalog.format(
            "chat-turn",
            kreate_i18n::TextKind::Ui,
            &[
                ("steps", steps.as_str()),
                ("added", added.as_str()),
                ("removed", removed.as_str()),
            ],
        );
        self.ui.chat_log.push(ChatMessage {
            role: ChatRole::Turn(index),
            text,
        });
    }

    /// Takes back everything one answer did, if nothing has been done since:
    /// undoing past later edits of the user's would take those too.
    pub(super) fn revert_chat_turn(&mut self, index: usize) {
        use kreate_ui::chat::{ChatMessage, ChatRole};
        let Some(turn) = self.chat_turns.get(index) else {
            return;
        };
        let (start, end) = (turn.start, turn.end.unwrap_or(turn.start));
        let message = if self.history.cursor() != end || self.history.entries().len() < end {
            self.ui.catalog.get("chat-turn-later")
        } else {
            while self.history.cursor() > start {
                if !self.history.undo(&mut self.doc) {
                    break;
                }
            }
            self.edit.clear();
            self.redraw.set(RedrawReason::SCENE_CHANGED);
            // Done once: the button goes, and the line says what happened.
            for m in &mut self.ui.chat_log {
                if m.role == ChatRole::Turn(index) {
                    m.role = ChatRole::Note;
                }
            }
            if let Some(turn) = self.chat_turns.get_mut(index) {
                turn.end = Some(start);
            }
            let steps = (end - start).to_string();
            self.ui.catalog.format(
                "chat-turn-reverted",
                kreate_i18n::TextKind::Ui,
                &[("steps", steps.as_str())],
            )
        };
        self.ui.chat_log.push(ChatMessage {
            role: ChatRole::Note,
            text: message,
        });
    }
}
