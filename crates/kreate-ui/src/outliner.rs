//! The object outliner.
//!
//! Rows are **virtualised**: only what is on screen is laid out. A modelling
//! scene reaches tens or hundreds of thousands of objects, and immediate-mode UI
//! punishes iterating them — building every row every frame is fine at twenty
//! rows and ruinous at fifty thousand, which is exactly the shape of bug that
//! hides until someone opens a real scene.
//!
//! The crate cannot see a `Document`, so the application hands over a
//! [`OutlinerRow`] view model and receives [`crate::Command`]s. Selection
//! arithmetic — plain click, additive click, range click — lives in
//! [`resolve_click`] as a pure function, because that is where the bugs are and
//! it is the part worth testing without a window.

use egui::Ui;
use kreate_i18n::{Catalog, TextKind};

use crate::{Command, CommandQueue};

/// What an object is, for the row's icon and summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    Empty,
    Mesh { verts: u32, faces: u32 },
    Camera,
    Light,
}

impl RowKind {
    /// A short glyph, chosen from characters the subset font actually contains.
    pub fn glyph(self) -> &'static str {
        match self {
            RowKind::Empty => "\u{25CB}",
            RowKind::Mesh { .. } => "\u{25B2}",
            RowKind::Camera => "\u{25B7}",
            RowKind::Light => "\u{25C7}",
        }
    }

    pub fn i18n_key(self) -> &'static str {
        match self {
            RowKind::Empty => "outliner-kind-empty",
            RowKind::Mesh { .. } => "outliner-kind-mesh",
            RowKind::Camera => "outliner-kind-camera",
            RowKind::Light => "outliner-kind-light",
        }
    }
}

/// One row of the outliner.
#[derive(Clone, Debug, PartialEq)]
pub struct OutlinerRow {
    /// Opaque object handle. The UI never interprets it.
    pub id: u32,
    pub name: String,
    /// Indentation level in the hierarchy.
    pub depth: usize,
    pub selected: bool,
    pub active: bool,
    pub visible: bool,
    pub kind: RowKind,
    /// How many objects share this row's mesh. Greater than one means editing it
    /// changes every user, which the outliner must say before the user is
    /// surprised by it.
    pub mesh_users: usize,
}

impl OutlinerRow {
    pub fn new(id: u32, name: impl Into<String>, kind: RowKind) -> Self {
        Self {
            id,
            name: name.into(),
            depth: 0,
            selected: false,
            active: false,
            visible: true,
            kind,
            mesh_users: 1,
        }
    }
}

/// How a click was modified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClickModifiers {
    /// Add to or remove from the selection.
    pub additive: bool,
    /// Select everything between the active row and this one.
    pub range: bool,
}

/// Works out the new selection from a click.
///
/// Returns `(objects, active)`. Pure, because this is where selection bugs live:
/// a range click that forgets to include one end, an additive click that fails to
/// toggle, an active object left pointing at something no longer selected.
pub fn resolve_click(
    rows: &[OutlinerRow],
    clicked_index: usize,
    modifiers: ClickModifiers,
) -> (Vec<u32>, Option<u32>) {
    let Some(clicked) = rows.get(clicked_index) else {
        return (Vec::new(), None);
    };

    if modifiers.range {
        // From the active row to the clicked one, inclusive at both ends. With no
        // active row there is nothing to range *from*, so it degrades to a plain
        // click rather than selecting everything.
        let anchor = rows.iter().position(|r| r.active).unwrap_or(clicked_index);
        let (lo, hi) = if anchor <= clicked_index {
            (anchor, clicked_index)
        } else {
            (clicked_index, anchor)
        };
        let objects: Vec<u32> = rows[lo..=hi].iter().map(|r| r.id).collect();
        return (objects, Some(clicked.id));
    }

    if modifiers.additive {
        let mut objects: Vec<u32> = rows.iter().filter(|r| r.selected).map(|r| r.id).collect();
        if let Some(pos) = objects.iter().position(|id| *id == clicked.id) {
            objects.remove(pos);
            // Deselecting the active row must move the active marker, or later
            // operators act on something the user cannot see is chosen.
            let active = objects.last().copied();
            return (objects, active);
        }
        objects.push(clicked.id);
        return (objects, Some(clicked.id));
    }

    (vec![clicked.id], Some(clicked.id))
}

/// Draws the outliner.
pub fn outliner(ui: &mut Ui, catalog: &Catalog, rows: &[OutlinerRow], queue: &mut CommandQueue) {
    let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        // Only the visible rows are laid out. The whole point: a hundred
        // thousand objects must cost the same as twenty.
        .show_rows(ui, row_height, rows.len(), |ui, visible| {
            for index in visible {
                let Some(row) = rows.get(index) else { continue };
                draw_row(ui, catalog, rows, index, row, queue);
            }
        });
}

fn draw_row(
    ui: &mut Ui,
    catalog: &Catalog,
    rows: &[OutlinerRow],
    index: usize,
    row: &OutlinerRow,
    queue: &mut CommandQueue,
) {
    ui.horizontal(|ui| {
        ui.add_space(row.depth as f32 * 12.0);

        // i18n-exempt: geometric glyphs standing in for icons
        ui.label(row.kind.glyph()).on_hover_text(catalog.format(
            row.kind.i18n_key(),
            TextKind::Tooltip,
            &[],
        ));

        let mut label = egui::RichText::new(&row.name);
        if row.active {
            label = label.strong();
        }
        if !row.visible {
            label = label.weak();
        }

        let response = ui.selectable_label(row.selected, label);
        if response.clicked() {
            let modifiers = ui.input(|i| ClickModifiers {
                additive: i.modifiers.ctrl || i.modifiers.command,
                range: i.modifiers.shift,
            });
            let (objects, active) = resolve_click(rows, index, modifiers);
            queue.push(Command::SetSelection { objects, active });
        }
        if response.double_clicked() {
            queue.push(Command::FrameSelection);
        }

        // A shared mesh is worth flagging in place: editing it changes every
        // user, and a user who does not know that reads the result as a bug.
        if row.mesh_users > 1 {
            ui.weak(format!("\u{00D7}{}", row.mesh_users))
                .on_hover_text(catalog.format(
                    "outliner-shared-mesh",
                    TextKind::Tooltip,
                    &[("count", &row.mesh_users.to_string())],
                ));
        }
    });
}

/// A one-line summary of a row, for the status bar.
pub fn summarise(catalog: &Catalog, row: &OutlinerRow) -> String {
    match row.kind {
        RowKind::Mesh { verts, faces } => catalog.format(
            "outliner-mesh-summary",
            TextKind::Ui,
            &[
                ("name", &row.name),
                ("verts", &verts.to_string()),
                ("faces", &faces.to_string()),
            ],
        ),
        other => catalog.format(other.i18n_key(), TextKind::Ui, &[("name", &row.name)]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize) -> Vec<OutlinerRow> {
        (0..n)
            .map(|i| OutlinerRow::new(i as u32, format!("Object{i}"), RowKind::Empty))
            .collect()
    }

    fn plain() -> ClickModifiers {
        ClickModifiers {
            additive: false,
            range: false,
        }
    }
    fn additive() -> ClickModifiers {
        ClickModifiers {
            additive: true,
            range: false,
        }
    }
    fn range() -> ClickModifiers {
        ClickModifiers {
            additive: false,
            range: true,
        }
    }

    #[test]
    fn a_plain_click_replaces_the_selection() {
        let mut r = rows(5);
        r[0].selected = true;
        r[1].selected = true;
        let (objects, active) = resolve_click(&r, 3, plain());
        assert_eq!(objects, vec![3]);
        assert_eq!(active, Some(3));
    }

    #[test]
    fn an_additive_click_adds_and_toggles_off() {
        let mut r = rows(5);
        r[1].selected = true;

        let (objects, active) = resolve_click(&r, 3, additive());
        assert_eq!(objects, vec![1, 3]);
        assert_eq!(active, Some(3));

        // Clicking an already-selected row removes it.
        r[3].selected = true;
        let (objects, _) = resolve_click(&r, 3, additive());
        assert_eq!(objects, vec![1]);
    }

    #[test]
    fn deselecting_the_active_row_moves_the_active_marker() {
        // Otherwise later operators act on something the user cannot see is
        // chosen, which reads as the tool ignoring them.
        let mut r = rows(5);
        r[1].selected = true;
        r[3].selected = true;
        r[3].active = true;

        let (objects, active) = resolve_click(&r, 3, additive());
        assert_eq!(objects, vec![1]);
        assert_eq!(active, Some(1), "active must follow what is left");
    }

    #[test]
    fn a_range_click_includes_both_ends() {
        // The off-by-one that every list widget gets wrong at least once.
        let mut r = rows(10);
        r[2].active = true;
        r[2].selected = true;

        let (objects, active) = resolve_click(&r, 5, range());
        assert_eq!(objects, vec![2, 3, 4, 5]);
        assert_eq!(active, Some(5));
    }

    #[test]
    fn a_range_click_works_upward_too() {
        let mut r = rows(10);
        r[7].active = true;
        let (objects, _) = resolve_click(&r, 4, range());
        assert_eq!(objects, vec![4, 5, 6, 7]);
    }

    #[test]
    fn a_range_click_with_no_anchor_selects_only_the_clicked_row() {
        // Selecting everything because there was nothing to range from would be
        // a nasty surprise on a large scene.
        let r = rows(100);
        let (objects, active) = resolve_click(&r, 42, range());
        assert_eq!(objects, vec![42]);
        assert_eq!(active, Some(42));
    }

    #[test]
    fn a_range_click_on_the_anchor_itself_selects_just_it() {
        let mut r = rows(10);
        r[3].active = true;
        let (objects, _) = resolve_click(&r, 3, range());
        assert_eq!(objects, vec![3]);
    }

    #[test]
    fn clicking_past_the_end_is_a_no_op() {
        let r = rows(3);
        assert_eq!(resolve_click(&r, 99, plain()), (Vec::new(), None));
        assert_eq!(resolve_click(&[], 0, plain()), (Vec::new(), None));
    }

    #[test]
    fn selection_arithmetic_is_linear_at_a_hundred_thousand_rows() {
        // The plan's scale check. An implementation that is quadratic in the row
        // count looks fine in a demo and is unusable on a real scene.
        let mut r = rows(100_000);
        r[10].active = true;

        let start = std::time::Instant::now();
        let (objects, _) = resolve_click(&r, 99_999, range());
        let elapsed = start.elapsed();

        assert_eq!(objects.len(), 99_990);
        assert!(
            elapsed.as_millis() < 200,
            "resolving a click over 100k rows took {elapsed:?}"
        );
    }

    #[test]
    fn every_row_kind_has_a_glyph_and_a_translation_key() {
        for kind in [
            RowKind::Empty,
            RowKind::Mesh { verts: 8, faces: 6 },
            RowKind::Camera,
            RowKind::Light,
        ] {
            assert!(!kind.glyph().is_empty());
            let key = kind.i18n_key();
            assert!(key.starts_with("outliner-kind-"), "{key}");
            assert!(
                kreate_reflect::is_valid_i18n_key(key),
                "{key} is not a valid Fluent identifier"
            );
        }
    }

    #[test]
    fn a_summary_uses_a_translation_and_falls_back_readably() {
        let mut catalog = Catalog::new();
        catalog
            .add(
                "ko-KR",
                "outliner-mesh-summary = { $name }: 정점 { $verts }, 면 { $faces }\n",
            )
            .unwrap();
        let row = OutlinerRow::new(0, "Cube", RowKind::Mesh { verts: 8, faces: 6 });
        assert_eq!(summarise(&catalog, &row), "Cube: 정점 8, 면 6");

        // With no catalog at all the key comes back, which is visible and
        // obviously wrong rather than silently blank.
        let empty = Catalog::new();
        assert_eq!(summarise(&empty, &row), "outliner-mesh-summary");
    }

    #[test]
    fn a_shared_mesh_is_recorded_on_the_row() {
        let mut row = OutlinerRow::new(0, "Cube", RowKind::Mesh { verts: 8, faces: 6 });
        assert_eq!(row.mesh_users, 1, "a lone object is not 'shared'");
        row.mesh_users = 4;
        assert!(row.mesh_users > 1);
    }
}
