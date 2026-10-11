//! A collapsible panel inside an editor, drawn the way Blender draws one.
//!
//! The properties editor is not one form; it is a stack of named panels —
//! Transform, Surface, Settings — each of which folds away. That is what lets a
//! tab hold everything about an object without becoming a page: the user opens
//! the panel they came for and the rest is a row of titles. A flat grid of
//! every field, which is what the tabs were before this, reads fine at four
//! rows and stops reading at twenty.
//!
//! **Its own widget rather than egui's `CollapsingHeader`** because the header
//! has to look like a panel header — a full-width raised bar with a chevron and
//! a bold title — rather than a tree node, and because the body needs the
//! panel's inset. Both are a dozen lines here and a fight against defaults
//! there.

use egui::{Pos2, Response, Sense, Stroke, Ui, Vec2};

/// Draws a panel with `title`, folded or open, and its body when open.
///
/// The open state is remembered per title within the containing `Ui`, which is
/// what makes two properties editors side by side fold independently.
pub fn section<R>(
    ui: &mut Ui,
    title: &str,
    default_open: bool,
    body: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    section_inner(ui, title, default_open, None, body).0
}

/// A section with a close button at the end of its header — for something
/// the panel shows because it is switched on, which the button switches
/// off. Returns the body's value when open, and whether the button was
/// pressed. `close_hint` is what hovering the button says.
pub fn section_closable<R>(
    ui: &mut Ui,
    title: &str,
    default_open: bool,
    close_hint: &str,
    body: impl FnOnce(&mut Ui) -> R,
) -> (Option<R>, bool) {
    section_inner(ui, title, default_open, Some(close_hint), body)
}

fn section_inner<R>(
    ui: &mut Ui,
    title: &str,
    default_open: bool,
    close_hint: Option<&str>,
    body: impl FnOnce(&mut Ui) -> R,
) -> (Option<R>, bool) {
    let theme = crate::theme::from_ui(ui);
    let id = ui.make_persistent_id(("panel", title));
    let mut open = ui.data_mut(|d| *d.get_persisted_mut_or(id, default_open));

    // The header bar: full width, raised, a chevron and the title.
    let height = theme.row_height;
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::click());
    let response: Response = response;
    // The close button, at the header's end: its own click, not the
    // header's.
    let mut closed = false;
    let mut over_close = false;
    let mut close_button = None;
    if let Some(hint) = close_hint {
        let size = height - 6.0;
        let close = egui::Rect::from_center_size(
            Pos2::new(rect.max.x - size * 0.5 - 4.0, rect.center().y),
            Vec2::splat(size),
        );
        let button = ui
            .interact(close, id.with("close"), Sense::click())
            .on_hover_text(hint);
        over_close = button.hovered();
        closed = button.clicked();
        close_button = Some((close, button.hovered()));
    }
    if response.clicked() && !over_close {
        open = !open;
        ui.data_mut(|d| d.insert_persisted(id, open));
    }
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        let fill = if response.hovered() {
            theme.border
        } else {
            theme.raised
        };
        painter.rect_filled(rect, theme.radius as f32, fill);
        // The chevron: right when folded, down when open. Painted so it takes
        // the theme and needs no glyph.
        let c = Pos2::new(rect.min.x + 12.0, rect.center().y);
        let s = 3.5;
        let points = if open {
            vec![
                Pos2::new(c.x - s, c.y - s * 0.5),
                Pos2::new(c.x + s, c.y - s * 0.5),
                Pos2::new(c.x, c.y + s * 0.6),
            ]
        } else {
            vec![
                Pos2::new(c.x - s * 0.5, c.y - s),
                Pos2::new(c.x - s * 0.5, c.y + s),
                Pos2::new(c.x + s * 0.6, c.y),
            ]
        };
        painter.add(egui::Shape::convex_polygon(
            points,
            theme.text_dim,
            Stroke::NONE,
        ));
        let font = egui::TextStyle::Body.resolve(ui.style());
        let galley = painter.layout_no_wrap(title.to_string(), font, theme.text);
        let text_pos = Pos2::new(rect.min.x + 24.0, rect.center().y - galley.size().y * 0.5);
        painter.galley(text_pos, galley, theme.text);
        if let Some((close, hovered)) = close_button {
            if hovered {
                painter.rect_filled(close, theme.radius as f32, theme.border);
            }
            // An ×, painted, so it takes the theme and needs no glyph.
            let c = close.center();
            let r = close.width() * 0.22;
            let stroke = Stroke::new(1.5, if hovered { theme.text } else { theme.text_dim });
            painter.line_segment([c + Vec2::new(-r, -r), c + Vec2::new(r, r)], stroke);
            painter.line_segment([c + Vec2::new(-r, r), c + Vec2::new(r, -r)], stroke);
        }
    }

    if !open {
        ui.add_space(2.0);
        return (None, closed);
    }
    // The body, inset from the header's edge the way a panel's contents are.
    let out = ui
        .scope(|ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                ui.vertical(body).inner
            })
            .inner
        })
        .inner;
    ui.add_space(6.0);
    (Some(out), closed)
}
