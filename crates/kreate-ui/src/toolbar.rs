//! The tool column down the left of the viewport.
//!
//! Blender's `T` panel, and for the reason Blender has one: the transforms and
//! the common mesh tools are reached by key a hundred times a day, and the
//! keys are learnt from somewhere. A column of pictures with the key in the
//! tooltip is where. It also gives a pen or a trackpad — neither of which has a
//! comfortable G — a way to start a drag at all.
//!
//! **Painted, not glyphs.** The bundled font is a subset with no fallback (see
//! [`crate::fonts::missing_glyphs`]), and there is no glyph for "rotate" in
//! any font worth shipping. Shapes drawn with the painter are crisp at any DPI,
//! take the theme's colours, and cost no atlas.
//!
//! The widget knows nothing about operators. It draws a [`Tool`] and reports a
//! click; the shell decides what a click means, which is the same split the
//! rest of this crate keeps.

use egui::{Color32, Pos2, Rect, Response, Sense, Stroke, Ui, Vec2};

/// A tool in the column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    /// Box select: the resting tool.
    Select,
    Move,
    Rotate,
    Scale,
    // --- edit mode ---------------------------------------------------------
    Extrude,
    Inset,
    Bevel,
    LoopCut,
    // --- sculpt mode -------------------------------------------------------
    /// Builds the surface up, or digs it in.
    SculptDraw,
    /// Evens the surface out.
    SculptSmooth,
    /// Drags the surface under the brush along.
    SculptGrab,
    /// Swells the surface along its normals.
    SculptInflate,
}

impl Tool {
    /// The catalog key for the tooltip's name.
    pub fn label_key(self) -> &'static str {
        match self {
            Tool::Select => "tool-select",
            Tool::Move => "tool-move",
            Tool::Rotate => "tool-rotate",
            Tool::Scale => "tool-scale",
            Tool::Extrude => "tool-extrude",
            Tool::Inset => "tool-inset",
            Tool::Bevel => "tool-bevel",
            Tool::LoopCut => "tool-loop-cut",
            Tool::SculptDraw => "sculpt-draw",
            Tool::SculptSmooth => "sculpt-smooth",
            Tool::SculptGrab => "sculpt-grab",
            Tool::SculptInflate => "sculpt-inflate",
        }
    }

    /// The tools every mode has.
    pub const COMMON: [Tool; 4] = [Tool::Select, Tool::Move, Tool::Rotate, Tool::Scale];
    /// The tools edit mode adds, in the order Blender lists them.
    pub const EDIT: [Tool; 4] = [Tool::Extrude, Tool::Inset, Tool::Bevel, Tool::LoopCut];
    /// Sculpt mode's brushes, in the order of the brush menu — so a brush's
    /// index here is its number in the brush setting.
    pub const SCULPT: [Tool; 4] = [
        Tool::SculptDraw,
        Tool::SculptSmooth,
        Tool::SculptGrab,
        Tool::SculptInflate,
    ];
}

/// Side of the icon's drawing area, in points.
const GLYPH: f32 = 18.0;
/// Width of the column, including its padding.
pub const WIDTH: f32 = 34.0;

/// Draws one tool button and returns its response.
///
/// `active` lights the button the way a selected mode is lit everywhere else,
/// which is what tells the user a click in the viewport will drag rather than
/// select.
pub fn tool_button(ui: &mut Ui, tool: Tool, active: bool) -> Response {
    let size = Vec2::splat(WIDTH - 6.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let theme = crate::theme::from_ui(ui);
    let visuals = ui.style().interact_selectable(&response, active);
    if active || response.hovered() {
        ui.painter()
            .rect_filled(rect, theme.radius as f32, visuals.weak_bg_fill);
    }
    let ink = if active { theme.accent } else { theme.text };
    let area = Rect::from_center_size(rect.center(), Vec2::splat(GLYPH));
    paint(ui, tool, area, ink);
    response
}

/// The Select tool's button, showing which shape a drag selects with: 0 a
/// box, 1 a lasso, 2 a circle — Blender's three select tools, in one place.
pub fn select_button(ui: &mut Ui, shape: u32, active: bool) -> Response {
    let size = Vec2::splat(WIDTH - 6.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let theme = crate::theme::from_ui(ui);
    let visuals = ui.style().interact_selectable(&response, active);
    if active || response.hovered() {
        ui.painter()
            .rect_filled(rect, theme.radius as f32, visuals.weak_bg_fill);
    }
    let ink = if active { theme.accent } else { theme.text };
    let area = Rect::from_center_size(rect.center(), Vec2::splat(GLYPH));
    let at = |x: f32, y: f32| {
        Pos2::new(
            area.min.x + x * area.width(),
            area.min.y + y * area.height(),
        )
    };
    let thin = Stroke::new(1.0, ink.gamma_multiply(0.55));
    match shape {
        // A loose loop.
        1 => {
            let points: Vec<Pos2> = (0..=28)
                .map(|i| {
                    let t = i as f32 / 28.0 * std::f32::consts::TAU;
                    let r = 0.32 + 0.06 * (3.0 * t).sin();
                    at(0.42 + r * t.cos(), 0.4 + r * 0.8 * t.sin())
                })
                .collect();
            ui.painter().add(egui::Shape::line(points, thin));
        }
        // A ring.
        2 => {
            ui.painter()
                .circle_stroke(at(0.42, 0.42), area.width() * 0.32, thin);
        }
        _ => dashed_rect(ui, Rect::from_min_max(at(0.08, 0.08), at(0.72, 0.72)), thin),
    }
    arrow_cursor(ui, at(0.55, 0.55), area.width() * 0.42, ink);
    response
}

/// The picture for each tool, in a unit square scaled to `area`.
fn paint(ui: &Ui, tool: Tool, area: Rect, ink: Color32) {
    let p = ui.painter();
    let at = |x: f32, y: f32| {
        Pos2::new(
            area.min.x + x * area.width(),
            area.min.y + y * area.height(),
        )
    };
    let line = Stroke::new(1.6, ink);
    let thin = Stroke::new(1.0, ink.gamma_multiply(0.55));
    match tool {
        // A mound rising out of a line: clay put on.
        Tool::SculptDraw => {
            let ground: Vec<Pos2> = (0..=20)
                .map(|i| {
                    let x = 0.08 + i as f32 / 20.0 * 0.84;
                    let bump = (-((x - 0.5) / 0.17).powi(2)).exp() * 0.42;
                    at(x, 0.78 - bump)
                })
                .collect();
            p.add(egui::Shape::line(ground, line));
            p.line_segment([at(0.5, 0.06), at(0.5, 0.26)], thin);
            p.line_segment([at(0.4, 0.16), at(0.6, 0.16)], thin);
        }
        // A rough line above, a smooth one below.
        Tool::SculptSmooth => {
            let rough: Vec<Pos2> = (0..=12)
                .map(|i| {
                    at(
                        0.1 + i as f32 / 12.0 * 0.8,
                        0.3 + if i % 2 == 0 { -0.08 } else { 0.08 },
                    )
                })
                .collect();
            p.add(egui::Shape::line(rough, thin));
            let smooth: Vec<Pos2> = (0..=20)
                .map(|i| {
                    let t = i as f32 / 20.0;
                    at(
                        0.1 + t * 0.8,
                        0.7 + (t * std::f32::consts::TAU).sin() * 0.06,
                    )
                })
                .collect();
            p.add(egui::Shape::line(smooth, line));
        }
        // A surface pulled sideways by an arrow.
        Tool::SculptGrab => {
            let ground: Vec<Pos2> = (0..=20)
                .map(|i| {
                    let x = 0.08 + i as f32 / 20.0 * 0.84;
                    let bump = (-((x - 0.62) / 0.14).powi(2)).exp() * 0.3;
                    at(x, 0.8 - bump)
                })
                .collect();
            p.add(egui::Shape::line(ground, line));
            p.line_segment([at(0.2, 0.3), at(0.7, 0.3)], line);
            arrow_head(
                ui,
                at(0.74, 0.3),
                Vec2::new(1.0, 0.0),
                area.width() * 0.18,
                ink,
            );
        }
        // A circle with arrows out of it.
        Tool::SculptInflate => {
            let c = at(0.5, 0.5);
            p.circle_stroke(c, area.width() * 0.2, line);
            for (dx, dy) in [(0.0f32, -1.0f32), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)] {
                let from = Pos2::new(
                    c.x + dx * area.width() * 0.27,
                    c.y + dy * area.height() * 0.27,
                );
                let tip = Pos2::new(
                    c.x + dx * area.width() * 0.46,
                    c.y + dy * area.height() * 0.46,
                );
                p.line_segment([from, tip], thin);
                arrow_head(ui, tip, Vec2::new(dx, dy), area.width() * 0.12, ink);
            }
        }
        // A dashed rectangle with a cursor arrow: the marquee.
        Tool::Select => {
            let r = Rect::from_min_max(at(0.08, 0.08), at(0.72, 0.72));
            dashed_rect(ui, r, thin);
            arrow_cursor(ui, at(0.55, 0.55), area.width() * 0.42, ink);
        }
        // Four arrows out from a centre.
        Tool::Move => {
            let c = at(0.5, 0.5);
            for (dx, dy) in [(0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)] {
                let tip = Pos2::new(
                    c.x + dx * area.width() * 0.45,
                    c.y + dy * area.height() * 0.45,
                );
                p.line_segment([c, tip], line);
                arrow_head(ui, tip, Vec2::new(dx, dy), area.width() * 0.18, ink);
            }
        }
        // An open arc with an arrowhead at its end.
        Tool::Rotate => {
            let c = at(0.5, 0.5);
            let r = area.width() * 0.38;
            let points: Vec<Pos2> = (0..=24)
                .map(|i| {
                    let t = -2.3 + i as f32 / 24.0 * 4.6;
                    Pos2::new(c.x + r * t.cos(), c.y + r * t.sin())
                })
                .collect();
            p.add(egui::Shape::line(points.clone(), line));
            if let [.., a, b] = points.as_slice() {
                let dir = (*b - *a).normalized();
                arrow_head(ui, *b, dir, area.width() * 0.2, ink);
            }
        }
        // A small square growing into a large one along the diagonal.
        Tool::Scale => {
            let small = Rect::from_min_max(at(0.1, 0.5), at(0.5, 0.9));
            let big = Rect::from_min_max(at(0.1, 0.1), at(0.9, 0.9));
            p.rect_stroke(small, 0.0, line, egui::StrokeKind::Middle);
            p.rect_stroke(big, 0.0, thin, egui::StrokeKind::Middle);
            p.line_segment([at(0.5, 0.5), at(0.86, 0.14)], line);
            arrow_head(
                ui,
                at(0.86, 0.14),
                Vec2::new(1.0, -1.0).normalized(),
                area.width() * 0.18,
                ink,
            );
        }
        // A face lifted off its base.
        Tool::Extrude => {
            let base = Rect::from_min_max(at(0.15, 0.6), at(0.85, 0.9));
            let top = Rect::from_min_max(at(0.15, 0.1), at(0.85, 0.4));
            p.rect_stroke(base, 0.0, thin, egui::StrokeKind::Middle);
            p.rect_filled(top, 0.0, ink.gamma_multiply(0.85));
            for x in [0.15, 0.85] {
                p.line_segment([at(x, 0.4), at(x, 0.6)], thin);
            }
        }
        // A face inside a face.
        Tool::Inset => {
            let outer = Rect::from_min_max(at(0.1, 0.1), at(0.9, 0.9));
            let inner = Rect::from_min_max(at(0.33, 0.33), at(0.67, 0.67));
            p.rect_stroke(outer, 0.0, thin, egui::StrokeKind::Middle);
            p.rect_filled(inner, 0.0, ink.gamma_multiply(0.85));
        }
        // A corner cut off.
        Tool::Bevel => {
            let pts = vec![
                at(0.15, 0.85),
                at(0.15, 0.45),
                at(0.45, 0.15),
                at(0.85, 0.15),
            ];
            p.add(egui::Shape::line(pts, line));
            p.line_segment([at(0.15, 0.15), at(0.15, 0.45)], thin);
            p.line_segment([at(0.15, 0.15), at(0.45, 0.15)], thin);
        }
        // A quad with a cut across it.
        Tool::LoopCut => {
            let r = Rect::from_min_max(at(0.12, 0.12), at(0.88, 0.88));
            p.rect_stroke(r, 0.0, thin, egui::StrokeKind::Middle);
            p.line_segment([at(0.5, 0.05), at(0.5, 0.95)], Stroke::new(2.0, ink));
        }
    }
}

fn arrow_head(ui: &Ui, tip: Pos2, dir: Vec2, size: f32, ink: Color32) {
    let back = tip - dir * size;
    let side = Vec2::new(-dir.y, dir.x) * size * 0.55;
    ui.painter().add(egui::Shape::convex_polygon(
        vec![tip, back + side, back - side],
        ink,
        Stroke::NONE,
    ));
}

fn arrow_cursor(ui: &Ui, origin: Pos2, size: f32, ink: Color32) {
    let pts = vec![
        origin,
        Pos2::new(origin.x, origin.y + size),
        Pos2::new(origin.x + size * 0.28, origin.y + size * 0.72),
        Pos2::new(origin.x + size * 0.7, origin.y + size * 0.7),
    ];
    ui.painter()
        .add(egui::Shape::convex_polygon(pts, ink, Stroke::NONE));
}

fn dashed_rect(ui: &Ui, r: Rect, stroke: Stroke) {
    let p = ui.painter();
    let corners = [
        r.left_top(),
        r.right_top(),
        r.right_bottom(),
        r.left_bottom(),
    ];
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let len = (b - a).length();
        let dash = 2.5;
        let n = (len / (dash * 2.0)).floor().max(1.0) as usize;
        for k in 0..n {
            let t0 = k as f32 * 2.0 * dash / len;
            let t1 = ((k as f32 * 2.0 + 1.0) * dash / len).min(1.0);
            p.line_segment([a + (b - a) * t0, a + (b - a) * t1], stroke);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_its_own_name() {
        let mut keys: Vec<&str> = Tool::COMMON
            .iter()
            .chain(Tool::EDIT.iter())
            .map(|t| t.label_key())
            .collect();
        let n = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(n, keys.len(), "two tools share a tooltip");
    }

    #[test]
    fn the_common_tools_start_with_select() {
        // The resting tool is first, so that the column reads top to bottom
        // as "select, then do something to the selection".
        assert_eq!(Tool::COMMON[0], Tool::Select);
    }
}
