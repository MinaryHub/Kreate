//! Painted icons for the properties editor's tab column.
//!
//! Painted rather than set in type: the bundled font is a subset with no
//! symbols to spare, and a tab column of □ ■ ◇ said nothing about which tab
//! was which until each was hovered. These are Blender's shapes in outline —
//! a wrench for modifiers, a globe for the world, a sphere for a material —
//! drawn in our own line, as the plan's §2.6 asks: Blender's conventions,
//! not Blender's artwork.

use egui::{Color32, Pos2, Rect, Stroke, Ui, Vec2};

use crate::outliner::RowKind;

/// What a tab's icon shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    /// Sliders: the last operation's settings.
    Tool,
    /// A filled square: the object.
    Object,
    /// The object's datablock, drawn as the outliner draws that kind.
    Data(RowKind),
    /// A bone: a shaft with a knuckle at each end.
    Bone,
    /// A shaded sphere.
    Material,
    /// A globe.
    World,
    /// A wrench.
    Modifiers,
    /// A camera's body and lens: how the picture is made.
    Render,
}

/// Paints `icon` into `rect` in `ink`.
pub fn paint_icon(ui: &Ui, rect: Rect, icon: Icon, ink: Color32) {
    let p = ui.painter();
    let c = rect.center();
    let s = rect.width().min(rect.height()) * 0.38;
    let line = Stroke::new(1.3, ink);
    match icon {
        Icon::Tool => {
            // Three sliders, their knobs at different places.
            for (row, at) in [(-0.6f32, -0.3f32), (0.0, 0.5), (0.6, -0.1)] {
                let y = c.y + row * s;
                p.line_segment([Pos2::new(c.x - s, y), Pos2::new(c.x + s, y)], line);
                p.circle_filled(Pos2::new(c.x + at * s, y), s * 0.22, ink);
            }
        }
        Icon::Object => {
            let r = Rect::from_center_size(c, Vec2::splat(s * 1.5));
            p.rect_filled(r, 1.5, ink);
        }
        Icon::Render => {
            let body = Rect::from_center_size(c + Vec2::new(-s * 0.25, 0.0), Vec2::new(s * 1.3, s * 1.0));
            p.rect_stroke(body, 1.5, line, egui::StrokeKind::Middle);
            let x = body.max.x;
            p.add(egui::Shape::convex_polygon(
                vec![
                    Pos2::new(x, c.y - s * 0.15),
                    Pos2::new(x + s * 0.6, c.y - s * 0.45),
                    Pos2::new(x + s * 0.6, c.y + s * 0.45),
                    Pos2::new(x, c.y + s * 0.15),
                ],
                ink,
                Stroke::NONE,
            ));
        }
        Icon::Data(kind) => crate::outliner::object_icon(ui, kind, rect, true),
        Icon::Bone => {
            let a = c + Vec2::new(-s * 0.7, s * 0.7);
            let b = c + Vec2::new(s * 0.7, -s * 0.7);
            p.line_segment([a, b], Stroke::new(2.0, ink));
            for end in [a, b] {
                p.circle_filled(end + Vec2::new(-s * 0.18, -s * 0.18), s * 0.24, ink);
                p.circle_filled(end + Vec2::new(s * 0.18, s * 0.18), s * 0.24, ink);
            }
        }
        Icon::Material => {
            p.circle_stroke(c, s, line);
            // The highlight, up and to the left, which is what makes a disc a
            // sphere.
            p.circle_filled(c + Vec2::new(-s * 0.35, -s * 0.35), s * 0.3, ink);
        }
        Icon::World => {
            p.circle_stroke(c, s, line);
            p.line_segment([Pos2::new(c.x - s, c.y), Pos2::new(c.x + s, c.y)], line);
            // A meridian: an ellipse, as a polyline.
            let meridian: Vec<Pos2> = (0..=16)
                .map(|i| {
                    let t = i as f32 / 16.0 * std::f32::consts::TAU;
                    Pos2::new(c.x + t.cos() * s * 0.45, c.y + t.sin() * s)
                })
                .collect();
            p.add(egui::Shape::line(meridian, line));
        }
        Icon::Modifiers => {
            // A wrench: a handle and an open jaw.
            let a = c + Vec2::new(-s * 0.8, s * 0.8);
            let b = c + Vec2::new(s * 0.25, -s * 0.25);
            p.line_segment([a, b], Stroke::new(2.2, ink));
            p.circle_stroke(
                c + Vec2::new(s * 0.5, -s * 0.5),
                s * 0.45,
                Stroke::new(1.8, ink),
            );
        }
    }
}
