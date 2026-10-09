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
    /// What clicking a tile does, for its hover.
    pub add_hint: String,
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
    /// A list or an asset on its way.
    pub busy: Option<String>,
    pub error: Option<String>,
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
    ui.add(
        egui::TextEdit::singleline(search)
            .hint_text(&text.search)
            .desired_width(f32::INFINITY),
    );
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
                            let response = response.on_hover_text(format!(
                                "{}\n{}\n{}",
                                tile.name, tile.detail, text.add_hint
                            ));
                            if response.hovered() {
                                ui.painter().rect_stroke(
                                    response.rect,
                                    4.0,
                                    egui::Stroke::new(2.0, theme.accent),
                                    egui::StrokeKind::Inside,
                                );
                            }
                            if response.clicked() {
                                queue.push(Command::AssetAdd { index: tile.index });
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
