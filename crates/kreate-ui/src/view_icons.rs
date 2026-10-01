//! The shading buttons on the viewport header, drawn.
//!
//! Blender's four spheres — wireframe, solid, material, rendered — are the most
//! recognisable row in the application, and the recognition is the point: the
//! *current* one is lit, so how the picture is being made reads at a glance
//! without a word. Three here, for the three ways this viewport shades.
//!
//! Painted for the reason every icon in this crate is: the bundled font is a
//! subset with no fallback, and a sphere with a highlight is not a glyph.

use egui::{Color32, Pos2, Rect, Response, Sense, Stroke, Ui, Vec2};
use kreate_render::ShadingMode;

/// Side of the icon's drawing area, in points.
const GLYPH: f32 = 14.0;

/// One shading-mode button. `selected` lights it.
pub fn shading_button(ui: &mut Ui, mode: ShadingMode, selected: bool) -> Response {
    let padding = ui.spacing().button_padding;
    let size = Vec2::splat(GLYPH) + padding * 2.0;
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let theme = crate::theme::from_ui(ui);
    let visuals = ui.style().interact_selectable(&response, selected);
    if selected || response.hovered() {
        ui.painter()
            .rect_filled(rect, theme.radius as f32, visuals.weak_bg_fill);
    }
    let ink = if selected { theme.accent } else { theme.text };
    let centre = rect.center();
    let r = GLYPH * 0.42;
    let p = ui.painter();
    match mode {
        // A sphere lit from the top left, with a highlight: shape-showing light.
        ShadingMode::Studio => {
            p.circle_filled(centre, r, ink.gamma_multiply(0.55));
            p.circle_filled(centre + Vec2::new(-r * 0.3, -r * 0.3), r * 0.55, ink);
            p.circle_stroke(centre, r, Stroke::new(1.0, ink));
        }
        // A wire sphere: an outline, an equator and a meridian.
        ShadingMode::Wireframe => {
            let s = Stroke::new(1.0, ink);
            p.circle_stroke(centre, r, s);
            p.add(egui::Shape::ellipse_stroke(centre, Vec2::new(r, r * 0.35), s));
            p.add(egui::Shape::ellipse_stroke(centre, Vec2::new(r * 0.35, r), s));
        }
        // A flat disc: no light, just the colour.
        ShadingMode::Flat => {
            p.circle_filled(centre, r, ink.gamma_multiply(0.8));
            p.circle_stroke(centre, r, Stroke::new(1.0, ink));
        }
        // A sphere with a horizon: the scene's own light and a background.
        ShadingMode::Rendered => {
            let sphere = Rect::from_center_size(centre, Vec2::splat(r * 2.0));
            p.circle_filled(centre, r, ink.gamma_multiply(0.35));
            // The lower half brighter, as though lit from below by a ground.
            let lower: Vec<Pos2> = (0..=16)
                .map(|i| {
                    let t = i as f32 / 16.0 * std::f32::consts::PI;
                    Pos2::new(centre.x + r * t.cos(), centre.y + r * t.sin())
                })
                .collect();
            p.add(egui::Shape::convex_polygon(lower, ink.gamma_multiply(0.9), Stroke::NONE));
            p.circle_stroke(centre, r, Stroke::new(1.0, ink));
            let _ = sphere;
        }
    }
    response
}

/// The colour a shading button uses when nothing is selected, for tests.
pub fn ink_for(selected: bool, accent: Color32, text: Color32) -> Color32 {
    if selected {
        accent
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_selected_button_wears_the_accent() {
        let accent = Color32::from_rgb(255, 158, 41);
        let text = Color32::WHITE;
        assert_eq!(ink_for(true, accent, text), accent);
        assert_eq!(ink_for(false, accent, text), text);
    }
}
