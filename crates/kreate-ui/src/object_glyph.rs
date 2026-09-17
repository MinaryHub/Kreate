//! The pictures drawn for objects that have no surface: lights and cameras.
//!
//! A lamp has a position and a direction and no mesh, so until this existed
//! adding one changed the lighting and nothing else — in the studio shading,
//! which does not use the scene's lights, it changed nothing at all. A user
//! who adds a light and sees the viewport unchanged concludes the button is
//! broken, and they are right to.
//!
//! **Blender's glyphs**, because a modeller reads them without being told: a
//! small circle for a point light with a dashed line dropped to the ground
//! plane; the same circle with parallel rays for a sun; a circle with a cone
//! for a spot; a square for an area light; and a pyramid for a camera, with a
//! triangle on top saying which way is up.
//!
//! **Drawn by the interface, not the renderer.** These are display-referred
//! marks in the interface palette, like the navigation gizmo, and the
//! interface already owns the layer over the picture. It also keeps them out
//! of the display transform, which would dim a lamp's glyph when the scene got
//! bright. The cost is that they do not hide behind geometry, which Blender's
//! do; that wants the renderer's depth and is left for when picking a lamp by
//! its glyph needs the renderer anyway.

use egui::{Color32, Painter, Pos2, Rect, Stroke};
use kreate_math::{DAffine3, DMat4, DQuat, DVec3};

/// What the interface knows about the camera, enough to put a world point on
/// the screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projector {
    pub eye: DVec3,
    pub rotation: DQuat,
    /// `Camera::projection_matrix(aspect)`, f64 so a far-off object projects
    /// where the renderer put it rather than a pixel away.
    pub projection: DMat4,
    /// The rectangle the picture fills, in interface points.
    pub rect: Rect,
}

impl Projector {
    /// Where a world point lands on screen, or `None` behind the camera.
    pub fn project(&self, world: DVec3) -> Option<Pos2> {
        // Camera-relative first, the way the renderer does it: the subtraction
        // in f64 is what keeps a distant object's glyph on its object.
        let view = self.rotation.inverse() * (world - self.eye);
        let clip = self.projection * view.extend(1.0);
        if clip.w <= 1e-9 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Pos2::new(
            self.rect.min.x + (ndc.x as f32 * 0.5 + 0.5) * self.rect.width(),
            self.rect.min.y + (0.5 - ndc.y as f32 * 0.5) * self.rect.height(),
        ))
    }
}

/// What kind of light a glyph stands for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightGlyph {
    Point,
    Sun,
    /// Full cone angle, in radians.
    Spot { cone_angle: f64 },
    Area,
}

/// The size of a glyph's body on screen, in points: the same whatever the
/// distance, the way a widget is, so a far lamp is still findable.
const BODY: f32 = 7.0;

/// Draws a light at `transform`, lit in the accent when selected.
pub fn light(
    painter: &Painter,
    projector: &Projector,
    transform: &DAffine3,
    kind: LightGlyph,
    ink: Color32,
) {
    let origin = transform.translation;
    let Some(at) = projector.project(origin) else {
        return;
    };
    if !projector.rect.contains(at) {
        return;
    }
    let line = Stroke::new(1.2, ink);
    let faint = Stroke::new(1.0, ink.gamma_multiply(0.55));

    // The body: a small circle with a dot, for every kind but the area light.
    match kind {
        LightGlyph::Area => {
            painter.rect_stroke(
                Rect::from_center_size(at, egui::Vec2::splat(BODY * 2.0)),
                0.0,
                line,
                egui::StrokeKind::Middle,
            );
        }
        _ => {
            painter.circle_stroke(at, BODY, line);
            painter.circle_filled(at, 1.6, ink);
        }
    }

    // The line to the ground: from the light straight down to z = 0, dashed,
    // which is how the eye reads a height. Only when the light is above or
    // below the ground rather than on it.
    if origin.z.abs() > 1e-6 {
        if let Some(foot) = projector.project(DVec3::new(origin.x, origin.y, 0.0)) {
            dashed(painter, at, foot, faint);
            painter.circle_filled(foot, 1.5, ink.gamma_multiply(0.55));
        }
    }

    // The direction, for the kinds that have one. A light points down its own
    // -Z, which is glTF's convention and Blender's.
    let forward = transform.matrix3 * DVec3::NEG_Z;
    match kind {
        LightGlyph::Sun => {
            // Parallel rays: one from the body along the direction, and a
            // few short ones beside it.
            let side = perpendicular(forward);
            for k in -1..=1 {
                let offset = side * (k as f64 * 0.25);
                let (Some(a), Some(b)) = (
                    projector.project(origin + offset * 0.6),
                    projector.project(origin + offset * 0.6 + forward * 1.2),
                ) else {
                    continue;
                };
                painter.line_segment([a, b], if k == 0 { line } else { faint });
            }
        }
        LightGlyph::Spot { cone_angle } => {
            // The cone: four edges from the light to a circle one unit along
            // the direction, whose radius is the cone's.
            let length = 1.5;
            let radius = (cone_angle * 0.5).tan() * length;
            let (u, v) = basis(forward);
            let ring: Vec<Pos2> = (0..24)
                .filter_map(|i| {
                    let t = i as f64 / 24.0 * std::f64::consts::TAU;
                    projector.project(origin + forward * length + (u * t.cos() + v * t.sin()) * radius)
                })
                .collect();
            if ring.len() == 24 {
                painter.add(egui::Shape::closed_line(ring.clone(), faint));
                for i in [0usize, 6, 12, 18] {
                    painter.line_segment([at, ring[i]], faint);
                }
            }
        }
        LightGlyph::Point | LightGlyph::Area => {}
    }
    if kind == LightGlyph::Area {
        if let Some(tip) = projector.project(origin + forward * 0.8) {
            painter.line_segment([at, tip], faint);
        }
    }
}

/// Draws a camera at `transform` as a pyramid with its apex at the camera,
/// opening along -Z, and a triangle on the top edge for "up".
pub fn camera(
    painter: &Painter,
    projector: &Projector,
    transform: &DAffine3,
    // Horizontal and vertical half-extents of the frame one unit ahead.
    half_extents: (f64, f64),
    ink: Color32,
) {
    let origin = transform.translation;
    let m = transform.matrix3;
    let forward = m * DVec3::NEG_Z;
    let right = m * DVec3::X;
    let up = m * DVec3::Y;
    let depth = 1.0;
    let (hw, hh) = half_extents;
    let centre = origin + forward * depth;
    let corners = [
        centre + right * hw + up * hh,
        centre - right * hw + up * hh,
        centre - right * hw - up * hh,
        centre + right * hw - up * hh,
    ];
    let Some(apex) = projector.project(origin) else {
        return;
    };
    let projected: Vec<Pos2> = corners.iter().filter_map(|c| projector.project(*c)).collect();
    if projected.len() != 4 {
        return;
    }
    let line = Stroke::new(1.2, ink);
    painter.add(egui::Shape::closed_line(projected.clone(), line));
    for p in &projected {
        painter.line_segment([apex, *p], Stroke::new(1.0, ink.gamma_multiply(0.7)));
    }
    // The up triangle, sitting on the top edge of the frame.
    let tri = [
        centre + up * hh + right * hw * 0.5,
        centre + up * hh - right * hw * 0.5,
        centre + up * (hh + hh * 0.7),
    ];
    let tri: Vec<Pos2> = tri.iter().filter_map(|c| projector.project(*c)).collect();
    if tri.len() == 3 {
        painter.add(egui::Shape::convex_polygon(tri, ink, Stroke::NONE));
    }
}

fn dashed(painter: &Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    let len = (b - a).length();
    if len < 1.0 {
        return;
    }
    let dash = 4.0;
    let n = (len / (dash * 2.0)).ceil().max(1.0) as usize;
    for k in 0..n {
        let t0 = (k as f32 * 2.0 * dash / len).min(1.0);
        let t1 = ((k as f32 * 2.0 + 1.0) * dash / len).min(1.0);
        painter.line_segment([a + (b - a) * t0, a + (b - a) * t1], stroke);
    }
}

/// A unit vector perpendicular to `v`.
fn perpendicular(v: DVec3) -> DVec3 {
    let helper = if v.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    v.cross(helper).normalize_or_zero()
}

/// Two unit vectors perpendicular to `v` and to each other.
fn basis(v: DVec3) -> (DVec3, DVec3) {
    let u = perpendicular(v);
    (u, v.cross(u).normalize_or_zero())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projector() -> Projector {
        // A camera at +Z looking down -Z at the origin, 90° vertical fov, over
        // a 200×100 rectangle.
        let aspect = 2.0;
        let f = 1.0 / (std::f64::consts::FRAC_PI_4).tan();
        // A plain perspective matrix, +Y up, w = -z.
        let projection = DMat4::from_cols_array(&[
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, -1.0, -1.0,
            0.0, 0.0, -0.1, 0.0,
        ]);
        Projector {
            eye: DVec3::new(0.0, 0.0, 5.0),
            rotation: DQuat::IDENTITY,
            projection,
            rect: Rect::from_min_size(Pos2::new(10.0, 20.0), egui::Vec2::new(200.0, 100.0)),
        }
    }

    #[test]
    fn the_point_the_camera_looks_at_lands_in_the_middle() {
        let p = projector();
        let at = p.project(DVec3::ZERO).expect("in front");
        assert!((at.x - 110.0).abs() < 0.01 && (at.y - 70.0).abs() < 0.01, "{at:?}");
    }

    #[test]
    fn up_in_the_world_is_up_on_the_screen() {
        // Screen y grows downward; a point above the pivot must land above
        // the centre, not below it — the sign the naive mapping gets wrong.
        let p = projector();
        let centre = p.project(DVec3::ZERO).unwrap();
        let above = p.project(DVec3::new(0.0, 1.0, 0.0)).unwrap();
        assert!(above.y < centre.y, "above {above:?} vs centre {centre:?}");
        let right = p.project(DVec3::new(1.0, 0.0, 0.0)).unwrap();
        assert!(right.x > centre.x);
    }

    #[test]
    fn a_point_behind_the_camera_is_not_drawn() {
        let p = projector();
        assert_eq!(p.project(DVec3::new(0.0, 0.0, 6.0)), None);
    }

    #[test]
    fn the_basis_is_orthonormal() {
        let v = DVec3::new(0.3, -0.5, 0.8).normalize();
        let (u, w) = basis(v);
        assert!(u.dot(v).abs() < 1e-9 && w.dot(v).abs() < 1e-9 && u.dot(w).abs() < 1e-9);
        assert!((u.length() - 1.0).abs() < 1e-9 && (w.length() - 1.0).abs() < 1e-9);
    }
}
