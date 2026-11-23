//! Interactive 3D transform gizmo (Move, Rotate, Scale) drawn over the viewport.
//!
//! Projects X, Y, and Z handles from the selection pivot at a constant screen
//! size through [`Projector`], draws them with depth-ordered strokes, and
//! hit-tests pointer positions against shafts, arrowheads, scale boxes,
//! rotation arcs, and the center free-transform ring.

use egui::{Color32, Painter, Pos2, Stroke};
use kreate_math::{DVec3, Vec3};

use crate::object_glyph::Projector;

/// Active 3D transform gizmo style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoKind {
    Move,
    Rotate,
    Scale,
}

impl GizmoKind {
    /// Converts the numeric `ToolSettings::transform_gizmo` index (`1..=3`).
    pub fn from_setting(index: u32) -> Option<Self> {
        match index {
            1 => Some(Self::Move),
            2 => Some(Self::Rotate),
            3 => Some(Self::Scale),
            _ => None,
        }
    }
}

/// Which handle of the 3D gizmo the pointer is over or dragging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoHandle {
    Center,
    AxisX,
    AxisY,
    AxisZ,
}

impl GizmoHandle {
    /// Axis index (`0 = X`, `1 = Y`, `2 = Z`), or `None` for the center handle.
    pub fn axis_index(self) -> Option<usize> {
        match self {
            Self::Center => None,
            Self::AxisX => Some(0),
            Self::AxisY => Some(1),
            Self::AxisZ => Some(2),
        }
    }
}

const ARM_SCREEN_PT: f32 = 68.0;
const INNER_GAP_PT: f32 = 11.0;
const CENTER_RADIUS_PT: f32 = 8.5;
const HIT_RADIUS_PT: f32 = 9.5;
const RING_SEGMENTS: usize = 48;

fn axis_color(axis: usize, active: bool) -> Color32 {
    match (axis, active) {
        (0, false) => Color32::from_rgb(230, 74, 89),
        (0, true) => Color32::from_rgb(255, 165, 174),
        (1, false) => Color32::from_rgb(107, 207, 99),
        (1, true) => Color32::from_rgb(184, 245, 178),
        (_, false) => Color32::from_rgb(77, 148, 240),
        (_, true) => Color32::from_rgb(170, 210, 255),
    }
}

fn handle_for_axis(axis: usize) -> GizmoHandle {
    match axis {
        0 => GizmoHandle::AxisX,
        1 => GizmoHandle::AxisY,
        _ => GizmoHandle::AxisZ,
    }
}

/// World-space length corresponding to `screen_pt` interface points at `pivot`.
pub fn world_arm_length(projector: &Projector, pivot: DVec3, screen_pt: f32) -> Option<f64> {
    let center = projector.project(pivot)?;
    let right_world = projector.rotation * DVec3::X;
    let probe = projector.project(pivot + right_world)?;
    let px_per_unit = center.distance(probe);
    if px_per_unit <= 1e-4 {
        return None;
    }
    Some((screen_pt / px_per_unit) as f64)
}

fn dist_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq <= 1e-5 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

fn normalized_basis(basis: [Vec3; 3]) -> [DVec3; 3] {
    [
        basis[0].normalize_or(Vec3::X).as_dvec3(),
        basis[1].normalize_or(Vec3::Y).as_dvec3(),
        basis[2].normalize_or(Vec3::Z).as_dvec3(),
    ]
}

/// Hit-tests `cursor` against the gizmo handles at `pivot`.
pub fn hit_test(
    projector: &Projector,
    pivot: DVec3,
    basis: [Vec3; 3],
    kind: GizmoKind,
    cursor: Pos2,
) -> Option<GizmoHandle> {
    let center = projector.project(pivot)?;
    if !projector.rect.contains(center) {
        return None;
    }
    let arm = world_arm_length(projector, pivot, ARM_SCREEN_PT)?;
    let axes = normalized_basis(basis);

    let mut best: Option<(f32, GizmoHandle)> = None;
    let mut consider = |d: f32, h: GizmoHandle| {
        if d <= HIT_RADIUS_PT && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, h));
        }
    };

    match kind {
        GizmoKind::Move | GizmoKind::Scale => {
            for (i, dir) in axes.iter().enumerate() {
                let Some(tip) = projector.project(pivot + *dir * arm) else {
                    continue;
                };
                let delta = tip - center;
                let len = delta.length();
                if len <= INNER_GAP_PT + 2.0 {
                    continue;
                }
                let start = center + delta * (INNER_GAP_PT / len);
                let d_shaft = dist_to_segment(cursor, start, tip);
                let d_tip = cursor.distance(tip) * 0.85;
                consider(d_shaft.min(d_tip), handle_for_axis(i));
            }
            let d_center = cursor.distance(center);
            if d_center <= CENTER_RADIUS_PT + 2.5 {
                consider(d_center, GizmoHandle::Center);
            }
        }
        GizmoKind::Rotate => {
            let to_eye = (projector.eye - pivot).normalize_or(DVec3::Z);
            for i in 0..3 {
                let u = axes[(i + 1) % 3];
                let v = axes[(i + 2) % 3];
                let mut prev: Option<(Pos2, bool)> = None;
                for s in 0..=RING_SEGMENTS {
                    let angle = (s as f64 / RING_SEGMENTS as f64) * std::f64::consts::TAU;
                    let offset = u * angle.cos() + v * angle.sin();
                    let front = offset.dot(to_eye) >= -0.15;
                    let pt = projector.project(pivot + offset * arm);
                    if let (Some((pa, fa)), Some(pb)) = (prev, pt) {
                        if fa || front {
                            consider(dist_to_segment(cursor, pa, pb), handle_for_axis(i));
                        }
                    }
                    prev = pt.map(|p| (p, front));
                }
            }
            let d_outer = (cursor.distance(center) - (ARM_SCREEN_PT + 10.0)).abs();
            if d_outer <= 6.5 {
                consider(d_outer + 1.0, GizmoHandle::Center);
            }
        }
    }

    best.map(|(_, h)| h)
}

/// Draws the 3D transform gizmo at `pivot`.
pub fn draw(
    painter: &Painter,
    projector: &Projector,
    pivot: DVec3,
    basis: [Vec3; 3],
    kind: GizmoKind,
    active: Option<GizmoHandle>,
) {
    let Some(center) = projector.project(pivot) else {
        return;
    };
    if !projector.rect.contains(center) {
        return;
    }
    let Some(arm) = world_arm_length(projector, pivot, ARM_SCREEN_PT) else {
        return;
    };
    let axes = normalized_basis(basis);
    let cam_forward = projector.rotation * DVec3::NEG_Z;

    match kind {
        GizmoKind::Move | GizmoKind::Scale => {
            // Draw back-most axes first so front-most handles sit cleanly on top.
            let mut order = [0usize, 1, 2];
            order.sort_by(|&a, &b| {
                let da = axes[a].dot(cam_forward);
                let db = axes[b].dot(cam_forward);
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            });

            for i in order {
                let dir = axes[i];
                let Some(tip) = projector.project(pivot + dir * arm) else {
                    continue;
                };
                let delta = tip - center;
                let len = delta.length();
                if len <= INNER_GAP_PT + 2.0 {
                    continue;
                }
                let unit = delta / len;
                let perp = egui::vec2(-unit.y, unit.x);
                let start = center + unit * INNER_GAP_PT;
                let is_hot = active == Some(handle_for_axis(i));
                let col = axis_color(i, is_hot);
                let width = if is_hot { 3.2 } else { 2.2 };

                if kind == GizmoKind::Move {
                    let head_len = 11.0f32.min(len * 0.35);
                    let head_half = 4.6;
                    let base = tip - unit * head_len;
                    painter.line_segment([start, base], Stroke::new(width, col));
                    painter.add(egui::Shape::convex_polygon(
                        vec![tip, base + perp * head_half, base - perp * head_half],
                        col,
                        Stroke::NONE,
                    ));
                } else {
                    let box_half = if is_hot { 5.2 } else { 4.2 };
                    let stop = tip - unit * box_half;
                    painter.line_segment([start, stop], Stroke::new(width, col));
                    let corners = [
                        tip + unit * box_half + perp * box_half,
                        tip + unit * box_half - perp * box_half,
                        tip - unit * box_half - perp * box_half,
                        tip - unit * box_half + perp * box_half,
                    ];
                    painter.add(egui::Shape::convex_polygon(
                        corners.to_vec(),
                        col,
                        Stroke::NONE,
                    ));
                }
            }

            let center_hot = active == Some(GizmoHandle::Center);
            let ring_col = if center_hot {
                Color32::WHITE
            } else {
                Color32::from_rgba_unmultiplied(235, 238, 245, 210)
            };
            painter.circle_filled(
                center,
                CENTER_RADIUS_PT,
                Color32::from_rgba_unmultiplied(24, 26, 32, 140),
            );
            painter.circle_stroke(
                center,
                CENTER_RADIUS_PT,
                Stroke::new(if center_hot { 2.6 } else { 1.8 }, ring_col),
            );
        }
        GizmoKind::Rotate => {
            let center_hot = active == Some(GizmoHandle::Center);
            let outer_col = if center_hot {
                Color32::WHITE
            } else {
                Color32::from_rgba_unmultiplied(220, 225, 235, 130)
            };
            painter.circle_stroke(
                center,
                ARM_SCREEN_PT + 10.0,
                Stroke::new(if center_hot { 2.6 } else { 1.4 }, outer_col),
            );

            let to_eye = (projector.eye - pivot).normalize_or(DVec3::Z);
            for i in 0..3 {
                let is_hot = active == Some(handle_for_axis(i));
                let front_col = axis_color(i, is_hot);
                let back_col = front_col.gamma_multiply(0.32);
                let width = if is_hot { 3.2 } else { 2.1 };
                let u = axes[(i + 1) % 3];
                let v = axes[(i + 2) % 3];
                let mut prev: Option<(Pos2, bool)> = None;
                for s in 0..=RING_SEGMENTS {
                    let angle = (s as f64 / RING_SEGMENTS as f64) * std::f64::consts::TAU;
                    let offset = u * angle.cos() + v * angle.sin();
                    let front = offset.dot(to_eye) >= 0.0;
                    let pt = projector.project(pivot + offset * arm);
                    if let (Some((pa, fa)), Some(pb)) = (prev, pt) {
                        let col = if fa && front { front_col } else { back_col };
                        painter.line_segment([pa, pb], Stroke::new(width, col));
                    }
                    prev = pt.map(|p| (p, front));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Rect;
    use kreate_math::{DMat4, DQuat};

    fn test_projector() -> Projector {
        let aspect = 2.0;
        let f = 1.0 / (std::f64::consts::FRAC_PI_4).tan();
        let projection = DMat4::from_cols_array(&[
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            -0.1,
            0.0,
        ]);
        Projector {
            eye: DVec3::new(3.0, 3.0, 5.0),
            rotation: DQuat::from_mat4(
                &DMat4::look_at_rh(DVec3::new(3.0, 3.0, 5.0), DVec3::ZERO, DVec3::Y).inverse(),
            ),
            projection,
            rect: Rect::from_min_size(Pos2::ZERO, egui::vec2(400.0, 200.0)),
        }
    }

    #[test]
    fn move_and_scale_handles_hit_each_axis_and_center() {
        let p = test_projector();
        let basis = [Vec3::X, Vec3::Y, Vec3::Z];
        let center = p.project(DVec3::ZERO).unwrap();
        let arm = world_arm_length(&p, DVec3::ZERO, ARM_SCREEN_PT).unwrap();

        for kind in [GizmoKind::Move, GizmoKind::Scale] {
            assert_eq!(
                hit_test(&p, DVec3::ZERO, basis, kind, center),
                Some(GizmoHandle::Center)
            );
            for (dir, want) in [
                (DVec3::X, GizmoHandle::AxisX),
                (DVec3::Y, GizmoHandle::AxisY),
                (DVec3::Z, GizmoHandle::AxisZ),
            ] {
                let tip = p.project(dir * arm).unwrap();
                assert_eq!(hit_test(&p, DVec3::ZERO, basis, kind, tip), Some(want));
            }
            assert_eq!(
                hit_test(&p, DVec3::ZERO, basis, kind, Pos2::new(5.0, 5.0)),
                None
            );
        }
    }

    #[test]
    fn rotate_gizmo_hits_outer_ring_and_axis_arcs() {
        let p = test_projector();
        let basis = [Vec3::X, Vec3::Y, Vec3::Z];
        let center = p.project(DVec3::ZERO).unwrap();
        let arm = world_arm_length(&p, DVec3::ZERO, ARM_SCREEN_PT).unwrap();

        let outer_pt = center + egui::vec2(ARM_SCREEN_PT + 10.0, 0.0);
        assert_eq!(
            hit_test(&p, DVec3::ZERO, basis, GizmoKind::Rotate, outer_pt),
            Some(GizmoHandle::Center)
        );
        // Z-rotation ring passes through +X * arm when facing +Z/+X/+Y.
        let z_arc_pt = p.project(DVec3::X * arm).unwrap();
        assert!(hit_test(&p, DVec3::ZERO, basis, GizmoKind::Rotate, z_arc_pt).is_some());
    }
}
