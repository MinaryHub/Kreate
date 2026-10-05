//! The three element-mode buttons, drawn rather than typed.
//!
//! **Why these are painted and not glyphs.** The bundled font is a subset with
//! no system fallback, so a symbol outside it is a blank box — see
//! [`crate::fonts::missing_glyphs`], which exists because four symbols in the
//! application's own chrome were exactly that. The geometric shapes the subset
//! *does* carry are squares and circles, and none of them says "vertex" or
//! "edge". Words were the honest fallback until the header grew Mesh, Edge and
//! Face menus beside them: with those in the row, buttons reading "Edge" and
//! "Face" sat two inches from menus reading "Edge" and "Face" and meant
//! something else entirely.
//!
//! Painting solves that and is better than an icon font would have been: the
//! shapes are crisp at any DPI with no atlas, they take the theme's colours
//! rather than carrying their own, and there is no third font to ship.
//!
//! **The picture is Blender's**, because it is the one a modeller can read
//! without being told: a square with its corners marked, and whichever part
//! the mode selects lit up. That the three share a frame is the point — they
//! are three views of the same square, so the difference between them is the
//! only thing that moves.

use egui::{Color32, Response, Sense, Ui, Vec2};

/// Which part of a mesh a click in the viewport picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    Vertex,
    Edge,
    Face,
}

impl Element {
    /// The index the `SetSelectMode` command carries.
    ///
    /// The operators take a plain integer, so this is the one place the
    /// ordering is written down rather than being repeated at each call.
    pub fn index(self) -> u32 {
        match self {
            Self::Vertex => 0,
            Self::Edge => 1,
            Self::Face => 2,
        }
    }

    /// The catalog key for the tooltip.
    ///
    /// A tooltip and not a label: the button is the picture. But the picture
    /// has to be nameable — for a new user, for a screen reader, and for
    /// anybody searching the interface for the word they know.
    pub fn label_key(self) -> &'static str {
        match self {
            Self::Vertex => "select-vert",
            Self::Edge => "select-edge",
            Self::Face => "select-face",
        }
    }

    /// All three, in the order they appear and the order the indices run.
    pub const ALL: [Element; 3] = [Element::Vertex, Element::Edge, Element::Face];
}

/// How much of the button the square takes. The rest is padding.
const GLYPH: f32 = 13.0;

/// A button showing which part of a mesh is being selected.
///
/// Returns the response so the caller can add the tooltip it wants; the widget
/// deliberately does not reach for the catalog, which belongs to the shell.
pub fn element_button(ui: &mut Ui, element: Element, selected: bool) -> Response {
    let padding = ui.spacing().button_padding;
    let size = Vec2::splat(GLYPH) + padding * 2.0;
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    if !ui.is_rect_visible(rect) {
        return response;
    }

    let theme = crate::theme::from_ui(ui);
    let visuals = ui.style().interact_selectable(&response, selected);
    // The button's own background, so that a selected mode reads the same way
    // a selected anything else does.
    if selected || response.hovered() {
        ui.painter()
            .rect_filled(rect, theme.radius as f32, visuals.weak_bg_fill);
    }

    // The square the three share. Inset by half a line width so the stroke
    // lands inside the rectangle rather than straddling its edge, which at
    // this size is the difference between crisp and smudged.
    let square = egui::Rect::from_center_size(rect.center(), Vec2::splat(GLYPH)).shrink(1.0);
    // A strong/weak pair from one base, rather than two theme colours. The
    // first attempt used `text_dim` against `text_faint` for the unselected
    // state, and at thirteen points those two are the same grey: the edge icon
    // had a lit edge nobody could see, which made it and the vertex icon read
    // as the same picture.
    let lit = if selected { theme.accent } else { theme.text };
    let dim = lit.gamma_multiply(0.35);

    let painter = ui.painter();
    let corners = [
        square.left_top(),
        square.right_top(),
        square.right_bottom(),
        square.left_bottom(),
    ];

    // The face first, so the frame and the corners draw over it.
    painter.rect_filled(
        square,
        0.0,
        if element == Element::Face {
            lit
        } else {
            Color32::TRANSPARENT
        },
    );

    // The frame. Every edge dim, then the left edge lit in edge mode — one
    // edge rather than all four, because the mode selects *an* edge and a
    // fully lit frame would read as "the whole outline".
    for (i, from) in corners.iter().enumerate() {
        let to = corners[(i + 1) % 4];
        let highlighted = element == Element::Edge && i == 3;
        painter.line_segment(
            [*from, to],
            egui::Stroke::new(
                if highlighted { 2.4 } else { 1.0 },
                if highlighted { lit } else { dim },
            ),
        );
    }

    // The corners last and always present: they are what makes the shape read
    // as a *mesh* square rather than a checkbox.
    for corner in corners {
        painter.circle_filled(
            corner,
            if element == Element::Vertex { 2.4 } else { 1.2 },
            if element == Element::Vertex { lit } else { dim },
        );
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_indices_are_the_ones_the_operators_take() {
        // The shell sends these as a plain integer and `mesh.dissolve`,
        // `mesh.delete` and `SetSelectMode` all read it. A reordering here
        // would silently start deleting faces when the user asked for vertices.
        assert_eq!(Element::Vertex.index(), 0);
        assert_eq!(Element::Edge.index(), 1);
        assert_eq!(Element::Face.index(), 2);
    }

    #[test]
    fn all_is_in_index_order_and_complete() {
        // The row is drawn by iterating `ALL`, so this is what puts vertex
        // first on screen as well as first in the enum.
        let indices: Vec<u32> = Element::ALL.iter().map(|e| e.index()).collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn every_element_has_its_own_name() {
        let mut keys: Vec<&str> = Element::ALL.iter().map(|e| e.label_key()).collect();
        keys.sort_unstable();
        let count = keys.len();
        keys.dedup();
        assert_eq!(count, keys.len(), "two elements share a tooltip");
    }
}
