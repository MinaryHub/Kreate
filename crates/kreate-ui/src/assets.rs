//! The asset browser: models, materials and environments to add, from local
//! folders or an online library, as a grid of pictures.
//!
//! The crate cannot fetch anything or read a folder, so the application owns
//! the library and hands over what to show; what the panel owns is nothing
//! but the controls. What it hands back is a [`Command`]: which source and
//! kind to list, which tile to add, folders to add or forget.

use egui::Ui;

use crate::{Command, CommandQueue};

/// One tile.
#[derive(Clone, Debug)]
pub struct AssetTile {
    /// What [`Command::AssetAdd`] names.
    pub index: usize,
    pub name: String,
    /// Shown on hover: who made it, or where it is.
    pub detail: String,
    /// The picture and its size in pixels, when it has arrived.
    pub texture: Option<(egui::TextureId, [f32; 2])>,
    /// Short marks over the picture's corner: that it moves, that its
    /// licence asks for a credit.
    pub badges: Vec<String>,
}

/// The panel's translated strings.
pub struct AssetText {
    pub sources: Vec<String>,
    pub kinds: Vec<String>,
    pub search: String,
    pub resolution: String,
    pub refresh: String,
    pub empty: String,
    pub loading: String,
    pub add_folder: String,
    pub remove_folder: String,
    pub no_folders: String,
    /// The line an online source asks for, and where it links.
    pub credit: Option<(String, String)>,
    /// What double-clicking a tile does, for its hover.
    pub add_hint: String,
    /// The button that adds the selected asset.
    pub add: String,
    /// The button that links it instead, for a `.kreate`, and its hover.
    pub link: String,
    pub link_hint: String,
    /// The toggle that lists only what moves.
    pub animated_only: String,
    /// What a source that searches itself says under its box.
    pub search_hint: String,
    /// For a source that wants a key: what to do, where to get one, the
    /// box's hint and the button.
    pub key_prompt: String,
    pub key_where: (String, String),
    pub key_hint: String,
    pub key_save: String,
    pub key_change: String,
    /// The AI source's box, its two buttons, the line under them, and its
    /// optional token: what it is for, where to get one, save and forget.
    pub gen_hint: String,
    pub gen_text: String,
    pub gen_image: String,
    pub gen_note: String,
    pub gen_token: String,
    pub gen_token_prompt: String,
    pub gen_token_where: (String, String),
    pub gen_token_saved: String,
    pub gen_token_forget: String,
}

/// The AI source's state, for its controls.
pub struct GeneratorView {
    /// A model is being made: one at a time.
    pub busy: bool,
    pub token_held: bool,
}

/// What the panel shows.
pub struct AssetView<'a> {
    pub source: usize,
    pub kind: usize,
    /// Offered for an online source only: `None` hides the choice.
    pub resolutions: Option<(&'a [&'a str], usize)>,
    /// For the local source: the folders read.
    pub folders: Option<&'a [String]>,
    pub tiles: &'a [AssetTile],
    /// The selected tile, by [`AssetTile::index`].
    pub selected: Option<usize>,
    /// Whether the selected asset can be linked rather than copied.
    pub linkable: bool,
    /// A list or an asset on its way.
    pub busy: Option<String>,
    pub error: Option<String>,
    /// Whether the search box asks the source (on Enter) rather than
    /// filtering what is listed.
    pub searched: bool,
    /// The only-what-moves toggle and its state, for a source that offers it.
    pub animated_only: Option<bool>,
    /// The source wants a key it does not have, or refused the one it has.
    pub key_needed: bool,
    /// The source has a key, which may be changed.
    pub key_held: bool,
    /// For the AI source: what makes a model.
    pub generator: Option<GeneratorView>,
}

/// Draws the panel. Returns the tiles that were on screen, by
/// [`AssetTile::index`], for the application to fetch pictures for.
pub fn asset_browser(
    ui: &mut Ui,
    text: &AssetText,
    view: &AssetView<'_>,
    search: &mut String,
    queue: &mut CommandQueue,
) -> Vec<usize> {
    let theme = crate::theme::from_ui(ui);
    ui.horizontal_wrapped(|ui| {
        egui::ComboBox::from_id_salt("asset-source")
            .selected_text(text.sources.get(view.source).cloned().unwrap_or_default())
            .show_ui(ui, |ui| {
                for (index, name) in text.sources.iter().enumerate() {
                    if ui.selectable_label(index == view.source, name).clicked()
                        && index != view.source
                    {
                        queue.push(Command::AssetSource { index });
                    }
                }
            });
        for (index, name) in text.kinds.iter().enumerate() {
            if ui.selectable_label(index == view.kind, name).clicked() && index != view.kind {
                queue.push(Command::AssetKind { index });
            }
        }
        if let Some((sizes, current)) = view.resolutions {
            egui::ComboBox::from_id_salt("asset-resolution")
                .selected_text(sizes.get(current).copied().unwrap_or(""))
                .show_ui(ui, |ui| {
                    for (index, size) in sizes.iter().enumerate() {
                        // i18n-exempt: an image size, "1k"
                        if ui.selectable_label(index == current, *size).clicked() {
                            queue.push(Command::AssetResolution { index });
                        }
                    }
                })
                .response
                .on_hover_text(&text.resolution);
        }
        if ui.small_button(&text.refresh).clicked() {
            queue.push(Command::AssetRefresh);
        }
    });
    let searched = ui.add(
        egui::TextEdit::singleline(search)
            .hint_text(&text.search)
            .desired_width(f32::INFINITY),
    );
    if view.searched {
        if searched.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            queue.push(Command::AssetSearch {
                text: search.clone(),
            });
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(&text.search_hint)
                    .small()
                    .color(theme.text_dim),
            );
            if let Some(on) = view.animated_only {
                let mut checked = on;
                if ui.checkbox(&mut checked, &text.animated_only).changed() {
                    queue.push(Command::AssetAnimatedOnly { on: checked });
                }
            }
            if view.key_held && !view.key_needed && ui.small_button(&text.key_change).clicked() {
                queue.push(Command::AssetKey { key: String::new() });
            }
        });
    }
    if view.key_needed {
        key_prompt(ui, text, queue);
    }
    if let Some(generator) = &view.generator {
        generator_controls(ui, text, generator, queue);
    }
    if let Some(folders) = view.folders {
        ui.horizontal_wrapped(|ui| {
            if folders.is_empty() {
                ui.weak(&text.no_folders);
            }
            for (index, folder) in folders.iter().enumerate() {
                egui::Frame::new()
                    .fill(theme.accent_wash)
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::symmetric(6, 2))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // i18n-exempt: a folder's own name
                            ui.label(egui::RichText::new(short(folder)).small())
                                .on_hover_text(folder);
                            // i18n-exempt: a close mark
                            if ui
                                .small_button("x")
                                .on_hover_text(&text.remove_folder)
                                .clicked()
                            {
                                queue.push(Command::AssetRemoveFolder { index });
                            }
                        });
                    });
            }
            if ui.small_button(&text.add_folder).clicked() {
                queue.push(Command::AssetAddFolder);
            }
        });
    }
    if let Some((line, url)) = &text.credit {
        ui.hyperlink_to(egui::RichText::new(line).small(), url);
    }
    if let Some(busy) = &view.busy {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(egui::RichText::new(busy).color(theme.text_dim));
        });
    }
    if let Some(error) = &view.error {
        // i18n-exempt: the source's or curl's own explanation
        ui.add(egui::Label::new(egui::RichText::new(error).color(theme.accent)).wrap());
    }
    ui.separator();

    // The selected asset, and the button that adds it: for whoever would
    // rather not double-click, and to say which one a double-click adds.
    if let Some(tile) = view
        .selected
        .and_then(|s| view.tiles.iter().find(|t| t.index == s))
    {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(view.busy.is_none(), egui::Button::new(&text.add))
                .on_hover_text(&text.add_hint)
                .clicked()
            {
                queue.push(Command::AssetAdd { index: tile.index });
            }
            if view.linkable
                && ui
                    .button(&text.link)
                    .on_hover_text(&text.link_hint)
                    .clicked()
            {
                queue.push(Command::AssetLink { index: tile.index });
            }
            // i18n-exempt: an asset's own name and its author or path
            ui.label(egui::RichText::new(&tile.name).strong());
            ui.add(
                egui::Label::new(
                    egui::RichText::new(&tile.detail)
                        .small()
                        .color(theme.text_dim),
                )
                .truncate(),
            );
        });
    }
    let mut seen = Vec::new();
    if view.tiles.is_empty() {
        if view.busy.is_none() {
            ui.weak(&text.empty);
        }
        return seen;
    }
    const TILE: f32 = 112.0;
    let columns = ((ui.available_width() / (TILE + 8.0)).floor() as usize).max(1);
    let rows = view.tiles.len().div_ceil(columns);
    let row_height = TILE + 24.0;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show_rows(ui, row_height, rows, |ui, range| {
            for row in range {
                ui.horizontal(|ui| {
                    for tile in view.tiles.iter().skip(row * columns).take(columns) {
                        seen.push(tile.index);
                        ui.vertical(|ui| {
                            ui.set_width(TILE);
                            let size = egui::vec2(TILE, TILE);
                            let (rect, response) =
                                ui.allocate_exact_size(size, egui::Sense::click());
                            match tile.texture {
                                // Fitted, not stretched: a tall bottle stays
                                // tall, centred in its square.
                                Some((texture, [w, h])) if w > 0.0 && h > 0.0 => {
                                    let scale = (TILE / w).min(TILE / h);
                                    let fitted = egui::Rect::from_center_size(
                                        rect.center(),
                                        egui::vec2(w * scale, h * scale),
                                    );
                                    ui.painter().image(
                                        texture,
                                        fitted,
                                        egui::Rect::from_min_max(
                                            egui::pos2(0.0, 0.0),
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::WHITE,
                                    );
                                }
                                _ => {
                                    ui.painter().rect_filled(rect, 4.0, theme.accent_wash);
                                }
                            }
                            let mut corner = rect.left_top() + egui::vec2(4.0, 4.0);
                            for badge in &tile.badges {
                                let galley = ui.painter().layout_no_wrap(
                                    badge.clone(),
                                    egui::FontId::proportional(10.0),
                                    egui::Color32::WHITE,
                                );
                                let back = egui::Rect::from_min_size(
                                    corner,
                                    galley.size() + egui::vec2(8.0, 2.0),
                                );
                                ui.painter().rect_filled(
                                    back,
                                    3.0,
                                    egui::Color32::from_black_alpha(170),
                                );
                                ui.painter().galley(
                                    corner + egui::vec2(4.0, 1.0),
                                    galley,
                                    egui::Color32::WHITE,
                                );
                                corner.y += back.height() + 2.0;
                            }
                            let response = response.on_hover_text(format!(
                                "{}\n{}\n{}",
                                tile.name, tile.detail, text.add_hint
                            ));
                            let selected = view.selected == Some(tile.index);
                            if selected || response.hovered() {
                                ui.painter().rect_stroke(
                                    response.rect,
                                    4.0,
                                    egui::Stroke::new(
                                        if selected { 2.5 } else { 1.0 },
                                        theme.accent,
                                    ),
                                    egui::StrokeKind::Inside,
                                );
                            }
                            // One click looks, two add: a stray click on a
                            // grid of pictures should not put a chair in the
                            // scene.
                            if response.double_clicked() {
                                queue.push(Command::AssetAdd { index: tile.index });
                            } else if response.clicked() && !selected {
                                queue.push(Command::AssetSelect { index: tile.index });
                            }
                            // i18n-exempt: an asset's own name
                            ui.add(
                                egui::Label::new(egui::RichText::new(&tile.name).small())
                                    .truncate(),
                            );
                        });
                    }
                });
            }
        });
    seen
}

/// A folder's last part, which is what tells two apart at a glance.
fn short(path: &str) -> &str {
    path.trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(path)
}

/// A box for the key a source wants, its link and its button. The key typed
/// is held in egui's memory until it is saved, never in the panel's state.
fn key_prompt(ui: &mut Ui, text: &AssetText, queue: &mut CommandQueue) {
    let theme = crate::theme::from_ui(ui);
    let id = ui.id().with("asset-key");
    let mut key: String = ui.data_mut(|d| d.get_temp(id).unwrap_or_default());
    egui::Frame::new()
        .fill(theme.accent_wash)
        .corner_radius(4.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.add(egui::Label::new(&text.key_prompt).wrap());
            ui.hyperlink_to(&text.key_where.0, &text.key_where.1);
            ui.horizontal(|ui| {
                let field = ui.add(
                    egui::TextEdit::singleline(&mut key)
                        .password(true)
                        .hint_text(&text.key_hint)
                        .desired_width(220.0),
                );
                let entered = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if (ui.button(&text.key_save).clicked() || entered) && !key.trim().is_empty() {
                    queue.push(Command::AssetKey {
                        key: std::mem::take(&mut key),
                    });
                }
            });
        });
    ui.data_mut(|d| d.insert_temp(id, key));
}

/// Words or a picture, and the buttons that make a model of them. The words
/// are held in egui's memory, so they survive the panel being redrawn.
fn generator_controls(
    ui: &mut Ui,
    text: &AssetText,
    generator: &GeneratorView,
    queue: &mut CommandQueue,
) {
    let theme = crate::theme::from_ui(ui);
    let id = ui.id().with("asset-generate");
    let mut words: String = ui.data_mut(|d| d.get_temp(id).unwrap_or_default());
    egui::Frame::new()
        .fill(theme.accent_wash)
        .corner_radius(4.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut words)
                    .hint_text(&text.gen_hint)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal_wrapped(|ui| {
                let idle = !generator.busy;
                if ui
                    .add_enabled(
                        idle && !words.trim().is_empty(),
                        egui::Button::new(&text.gen_text),
                    )
                    .clicked()
                {
                    queue.push(Command::AssetGenerate {
                        text: Some(words.trim().to_string()),
                    });
                }
                if ui
                    .add_enabled(idle, egui::Button::new(&text.gen_image))
                    .clicked()
                {
                    queue.push(Command::AssetGenerate { text: None });
                }
            });
            ui.add(
                egui::Label::new(
                    egui::RichText::new(&text.gen_note)
                        .small()
                        .color(theme.text_dim),
                )
                .wrap(),
            );
            egui::CollapsingHeader::new(&text.gen_token)
                .id_salt("asset-hf-token")
                .show(ui, |ui| {
                    if generator.token_held {
                        ui.horizontal(|ui| {
                            ui.label(&text.gen_token_saved);
                            if ui.small_button(&text.gen_token_forget).clicked() {
                                queue.push(Command::AssetHfToken {
                                    token: String::new(),
                                });
                            }
                        });
                        return;
                    }
                    ui.add(egui::Label::new(&text.gen_token_prompt).wrap());
                    ui.hyperlink_to(&text.gen_token_where.0, &text.gen_token_where.1);
                    let token_id = ui.id().with("asset-hf-token-text");
                    let mut token: String =
                        ui.data_mut(|d| d.get_temp(token_id).unwrap_or_default());
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut token)
                                .password(true)
                                .hint_text("hf_…")
                                .desired_width(220.0),
                        );
                        if ui.button(&text.key_save).clicked() && !token.trim().is_empty() {
                            queue.push(Command::AssetHfToken {
                                token: std::mem::take(&mut token),
                            });
                        }
                    });
                    ui.data_mut(|d| d.insert_temp(token_id, token));
                });
        });
    ui.data_mut(|d| d.insert_temp(id, words));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text() -> AssetText {
        let t = |s: &str| s.to_string();
        AssetText {
            sources: vec![
                t("Local"),
                t("Poly Haven"),
                t("ambientCG"),
                t("Poly Pizza"),
                t("AI"),
            ],
            kinds: vec![t("Models"), t("Materials"), t("HDRIs")],
            search: t("Search"),
            resolution: t("Size"),
            refresh: t("Refresh"),
            empty: t("NOTHING-HERE"),
            loading: t("Loading"),
            add_folder: t("Add folder"),
            remove_folder: t("Remove"),
            no_folders: t("No folders"),
            credit: None,
            add_hint: t("hint"),
            add: t("Add"),
            link: t("Link"),
            link_hint: t("link hint"),
            animated_only: t("ANIMATED-ONLY"),
            search_hint: t("SEARCH-HINT"),
            key_prompt: t("KEY-PROMPT"),
            key_where: (t("KEY-WHERE"), t("https://poly.pizza/settings/api")),
            key_hint: t("API key"),
            key_save: t("Save key"),
            key_change: t("Change key"),
            gen_hint: t("gen"),
            gen_text: t("gen text"),
            gen_image: t("gen image"),
            gen_note: t("gen note"),
            gen_token: t("token"),
            gen_token_prompt: t("token prompt"),
            gen_token_where: (t("w"), t("https://huggingface.co")),
            gen_token_saved: t("saved"),
            gen_token_forget: t("forget"),
        }
    }

    /// Every piece of text one frame of the panel draws.
    fn drawn(view: &AssetView<'_>) -> String {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(420.0, 700.0),
            )),
            ..Default::default()
        };
        let mut queue = CommandQueue::default();
        let mut search = String::new();
        let mut out = String::new();
        for _ in 0..2 {
            let output = ctx.run_ui(input.clone(), |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    asset_browser(ui, &text(), view, &mut search, &mut queue);
                });
            });
            out.clear();
            for shape in &output.shapes {
                if let egui::epaint::Shape::Text(t) = &shape.shape {
                    out.push_str(t.galley.text());
                    out.push('\n');
                }
            }
        }
        out
    }

    #[test]
    fn a_source_that_wants_a_key_asks_for_one() {
        let view = AssetView {
            source: 3,
            kind: 0,
            resolutions: None,
            folders: None,
            tiles: &[],
            selected: None,
            linkable: false,
            busy: None,
            error: None,
            searched: true,
            animated_only: Some(false),
            key_needed: true,
            key_held: false,
            generator: None,
        };
        let shown = drawn(&view);
        assert!(shown.contains("KEY-PROMPT"), "{shown}");
        assert!(shown.contains("KEY-WHERE"), "{shown}");
    }
}
