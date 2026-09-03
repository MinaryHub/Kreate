//! The object outliner.
//!
//! Rows are **virtualised twice over**, and the second one is the part that is
//! easy to miss.
//!
//! Only what is on screen is laid out — that much is `ScrollArea::show_rows`.
//! But an immediate-mode UI also punishes *building* what it does not draw, and
//! handing this widget a `Vec` of a hundred thousand rows means a hundred
//! thousand `String` clones every frame for the twenty the user can see. That
//! cost is invisible at twenty objects and ruinous at fifty thousand, which is
//! exactly the shape of bug that hides until someone opens a real scene.
//!
//! So the widget takes the row **order** — one `u32` per object, cheap enough to
//! cache and hand over whole — and a function that materialises a single row.
//! Only the visible ones are ever built.
//!
//! The crate cannot see a `Document`, so the application supplies both and
//! receives [`crate::Command`]s. Selection arithmetic — plain click, additive
//! click, range click — lives in [`resolve_click`] as a pure function over the
//! order, because that is where the bugs are and it is the part worth testing
//! without a window.

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
///
/// Built on demand for the rows on screen, never for the whole scene.
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
/// Returns `(objects, active)`. Pure, and over the row *order* rather than over
/// materialised rows: this is where selection bugs live — a range click that
/// forgets one end, an additive click that fails to toggle, an active object
/// left pointing at something no longer selected — and none of them need a row's
/// name or icon to reproduce.
pub fn resolve_click(
    order: &[u32],
    selected: &[u32],
    active: Option<u32>,
    clicked_index: usize,
    modifiers: ClickModifiers,
) -> (Vec<u32>, Option<u32>) {
    let Some(clicked) = order.get(clicked_index).copied() else {
        return (Vec::new(), None);
    };

    if modifiers.range {
        // From the active row to the clicked one, inclusive at both ends. With no
        // active row there is nothing to range *from*, so it degrades to a plain
        // click rather than selecting everything.
        let anchor = active
            .and_then(|a| order.iter().position(|id| *id == a))
            .unwrap_or(clicked_index);
        let (lo, hi) = if anchor <= clicked_index {
            (anchor, clicked_index)
        } else {
            (clicked_index, anchor)
        };
        return (order[lo..=hi].to_vec(), Some(clicked));
    }

    if modifiers.additive {
        let mut objects = selected.to_vec();
        if let Some(pos) = objects.iter().position(|id| *id == clicked) {
            objects.remove(pos);
            // Deselecting the active row must move the active marker, or later
            // operators act on something the user cannot see is chosen.
            let active = objects.last().copied();
            return (objects, active);
        }
        objects.push(clicked);
        return (objects, Some(clicked));
    }

    (vec![clicked], Some(clicked))
}

/// Draws the outliner.
///
/// `order` is every object in hierarchy order; `row` materialises one by
/// position and is called only for the rows on screen.
pub fn outliner(
    ui: &mut Ui,
    catalog: &Catalog,
    order: &[u32],
    selected: &[u32],
    active: Option<u32>,
    row: impl Fn(usize) -> Option<OutlinerRow>,
    queue: &mut CommandQueue,
) {
    let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        // Only the visible rows are laid out — and, because `row` is a function
        // rather than a slice, only the visible rows are *built*. The whole
        // point: a hundred thousand objects must cost the same as twenty.
        .show_rows(ui, row_height, order.len(), |ui, visible| {
            for index in visible {
                let Some(data) = row(index) else { continue };
                draw_row(ui, catalog, order, selected, active, index, &data, queue);
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn draw_row(
    ui: &mut Ui,
    catalog: &Catalog,
    order: &[u32],
    selected: &[u32],
    active: Option<u32>,
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
            let (objects, active) = resolve_click(order, selected, active, index, modifiers);
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

    fn order(n: u32) -> Vec<u32> {
        (0..n).collect()
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
    fn a_plain_click_selects_only_that_row() {
        let (objects, active) = resolve_click(&order(5), &[0, 1, 2], Some(1), 3, plain());
        assert_eq!(objects, vec![3]);
        assert_eq!(active, Some(3));
    }

    #[test]
    fn an_additive_click_adds_to_the_selection() {
        let (objects, active) = resolve_click(&order(5), &[1], Some(1), 3, additive());
        assert_eq!(objects, vec![1, 3]);
        assert_eq!(active, Some(3));
    }

    #[test]
    fn an_additive_click_on_a_selected_row_removes_it() {
        let (objects, _) = resolve_click(&order(5), &[1, 3], Some(3), 3, additive());
        assert_eq!(objects, vec![1]);
    }

    #[test]
    fn deselecting_the_active_row_moves_the_active_marker() {
        // Otherwise later operators act on something the user cannot see is
        // chosen.
        let (objects, active) = resolve_click(&order(5), &[1, 3], Some(3), 3, additive());
        assert_eq!(active, Some(1));
        assert!(objects.contains(&1));
    }

    #[test]
    fn a_range_click_includes_both_ends() {
        let (objects, active) = resolve_click(&order(6), &[1], Some(1), 4, range());
        assert_eq!(objects, vec![1, 2, 3, 4]);
        assert_eq!(active, Some(4));
    }

    #[test]
    fn a_range_click_works_upward_too() {
        let (objects, _) = resolve_click(&order(6), &[4], Some(4), 1, range());
        assert_eq!(objects, vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_range_click_with_no_anchor_degrades_to_a_plain_click() {
        // Selecting everything would be a destructive surprise from a Shift the
        // user was holding for something else.
        let (objects, _) = resolve_click(&order(6), &[], None, 3, range());
        assert_eq!(objects, vec![3]);
    }

    #[test]
    fn a_range_click_whose_anchor_has_left_the_scene_degrades_too() {
        // The active object can be deleted between frames; ranging from an id
        // that is no longer in the order has no meaning.
        let (objects, _) = resolve_click(&order(6), &[], Some(99), 3, range());
        assert_eq!(objects, vec![3]);
    }

    #[test]
    fn a_click_outside_the_order_selects_nothing() {
        // A stale row index from a frame where the scene was larger.
        let (objects, active) = resolve_click(&order(3), &[1], Some(1), 7, plain());
        assert!(objects.is_empty());
        assert_eq!(active, None);
    }

    #[test]
    fn resolving_a_click_does_not_depend_on_row_contents() {
        // The reason this takes the order rather than materialised rows: at a
        // hundred thousand objects, building rows to answer a click is the same
        // mistake as building them to draw twenty.
        let big = order(100_000);
        let (objects, active) = resolve_click(&big, &[0], Some(0), 99_999, range());
        assert_eq!(objects.len(), 100_000);
        assert_eq!(active, Some(99_999));
    }
}
