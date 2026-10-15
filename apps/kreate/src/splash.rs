//! The splash screen's moving backdrop: a ground grid running toward the
//! viewer, a few wireframe solids turning slowly, and motes rising — a
//! modeller's viewport at rest, faint enough that the logo, the message and
//! the bar on top of it read first.
//!
//! Shapes rather than painting, so a test can look at what a moment draws.

use egui::{Color32, Pos2, Rect, Shape, Stroke};

/// A wireframe solid: its corners and the edges between them.
type Solid = (Vec<[f32; 3]>, Vec<[usize; 2]>);

/// The edges of a cube, an octahedron and an icosahedron, each about the
/// unit sphere.
fn solids() -> [Solid; 3] {
    let cube_v: Vec<[f32; 3]> = (0..8)
        .map(|i| {
            let s = |b: usize| if i & b != 0 { 0.58 } else { -0.58 };
            [s(1), s(2), s(4)]
        })
        .collect();
    let mut cube_e = Vec::new();
    for a in 0..8usize {
        for b in [1usize, 2, 4] {
            if a & b == 0 {
                cube_e.push([a, a | b]);
            }
        }
    }
    let octa_v = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut octa_e = Vec::new();
    for a in 0..6usize {
        for b in a + 1..6 {
            if a / 2 != b / 2 {
                octa_e.push([a, b]);
            }
        }
    }
    let g = (1.0 + 5.0f32.sqrt()) * 0.5;
    let n = (1.0 + g * g).sqrt();
    let mut ico_v = Vec::new();
    for (x, y) in [(1.0, g), (-1.0, g), (1.0, -g), (-1.0, -g)] {
        ico_v.push([0.0, x / n, y / n]);
        ico_v.push([x / n, y / n, 0.0]);
        ico_v.push([y / n, 0.0, x / n]);
    }
    let mut ico_e = Vec::new();
    for a in 0..12usize {
        for b in a + 1..12 {
            let d: f32 = (0..3).map(|k| (ico_v[a][k] - ico_v[b][k]).powi(2)).sum();
            // Neighbours on the unit icosahedron are 1.05 apart; the next
            // nearest, 1.7.
            if d.sqrt() < 1.2 {
                ico_e.push([a, b]);
            }
        }
    }
    [(cube_v, cube_e), (octa_v, octa_e), (ico_v, ico_e)]
}

/// A point turned about Y by `yaw` and about X by `pitch`.
fn turn(p: [f32; 3], yaw: f32, pitch: f32) -> [f32; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let x = p[0] * cy + p[2] * sy;
    let z = -p[0] * sy + p[2] * cy;
    let y = p[1] * cp - z * sp;
    let z = p[1] * sp + z * cp;
    [x, y, z]
}

/// What the backdrop draws at `t` seconds, filling `rect`. `ink` is the
/// colour of its lines — the theme's dim text — and `accent` of the
/// solids' and the motes' glow.
pub fn backdrop(rect: Rect, t: f32, ink: Color32, accent: Color32) -> Vec<Shape> {
    let mut out = Vec::new();
    let (w, h) = (rect.width(), rect.height());
    if w <= 1.0 || h <= 1.0 {
        return out;
    }

    // The ground: lines across it receding to a horizon two thirds down,
    // running toward the viewer; lines along it fanning from a vanishing
    // point in the middle.
    let horizon = rect.top() + h * 0.62;
    let centre_x = rect.center().x;
    let depth = |z: f32| horizon + (rect.bottom() - horizon) * (1.0 / z);
    let fade = |a: f32| ink.gamma_multiply(a.clamp(0.0, 1.0));
    let scroll = (t * 0.35).fract();
    for k in 0..14 {
        let z = 1.0 + (k as f32 + 1.0 - scroll) * 0.9;
        let y = depth(z);
        if y <= horizon + 0.5 {
            continue;
        }
        let near = (y - horizon) / (rect.bottom() - horizon);
        out.push(Shape::line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            Stroke::new(1.0, fade(0.06 + 0.22 * near)),
        ));
    }
    for k in -6..=6 {
        let x = centre_x + k as f32 * w * 0.08;
        out.push(Shape::line_segment(
            [
                Pos2::new(centre_x + (x - centre_x) * 0.02, horizon),
                Pos2::new(x, rect.bottom()),
            ],
            Stroke::new(1.0, fade(0.16)),
        ));
    }

    // The solids, around the middle and out of the way of the text: each
    // turning at its own pace, bobbing a little.
    let places = [(0.16, 0.30, 0.10), (0.84, 0.26, 0.085), (0.78, 0.74, 0.065)];
    for (i, ((verts, edges), (fx, fy, size))) in solids().into_iter().zip(places).enumerate() {
        let speed = 0.25 + i as f32 * 0.09;
        let yaw = t * speed + i as f32 * 1.7;
        let pitch = 0.45 + 0.25 * (t * 0.3 + i as f32).sin();
        let r = size * w.min(h * 1.6);
        let c = Pos2::new(
            rect.left() + w * fx,
            rect.top() + h * fy + (t * 0.8 + i as f32 * 2.1).sin() * r * 0.12,
        );
        let projected: Vec<(Pos2, f32)> = verts
            .iter()
            .map(|p| {
                let q = turn(*p, yaw, pitch);
                let s = 3.0 / (3.0 + q[2]);
                (Pos2::new(c.x + q[0] * r * s, c.y - q[1] * r * s), q[2])
            })
            .collect();
        for [a, b] in edges {
            // Nearer edges brighter: the depth reads without a fill.
            let z = (projected[a].1 + projected[b].1) * 0.5;
            let alpha = 0.18 + 0.22 * (1.0 - (z + 1.0) * 0.5);
            out.push(Shape::line_segment(
                [projected[a].0, projected[b].0],
                Stroke::new(1.2, accent.gamma_multiply(alpha)),
            ));
        }
    }

    // Motes rising from the ground and fading out: each on its own column
    // and timing, so they never line up.
    for i in 0..36 {
        let seed = i as f32 * 12.989_8;
        let column = (seed.sin() * 43_758.547).fract().abs();
        let rate = 0.04 + 0.05 * ((seed * 1.7).cos() * 1_000.0).fract().abs();
        let phase = ((seed * 3.1).sin() * 9_000.0).fract().abs();
        let life = (t * rate + phase).fract();
        let x = rect.left() + column * w + (t * 0.5 + seed).sin() * 6.0;
        let y = rect.bottom() - life * h * 0.9;
        let alpha = (life * 4.0).min(1.0) * (1.0 - life) * 0.55;
        if alpha < 0.01 {
            continue;
        }
        out.push(Shape::circle_filled(
            Pos2::new(x, y),
            1.2 + 1.3 * column,
            accent.gamma_multiply(alpha),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> Rect {
        Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 500.0))
    }

    #[test]
    fn the_solids_have_their_edges() {
        let [cube, octa, ico] = solids();
        assert_eq!(cube.1.len(), 12);
        assert_eq!(octa.1.len(), 12);
        assert_eq!(ico.1.len(), 30);
    }

    #[test]
    fn the_backdrop_moves_and_stays_in_its_place() {
        let a = backdrop(rect(), 0.0, Color32::GRAY, Color32::GOLD);
        let b = backdrop(rect(), 0.7, Color32::GRAY, Color32::GOLD);
        assert!(a.len() > 50, "{} shapes", a.len());
        assert_ne!(format!("{a:?}"), format!("{b:?}"), "nothing moved");
        let room = rect().expand(40.0);
        for shape in a.iter().chain(&b) {
            let r = shape.visual_bounding_rect();
            assert!(room.contains_rect(r), "{r:?} is off the screen");
        }
    }

    #[test]
    fn nothing_is_drawn_into_no_room() {
        let empty = Rect::from_min_size(Pos2::ZERO, egui::vec2(0.0, 0.0));
        assert!(backdrop(empty, 1.0, Color32::GRAY, Color32::GOLD).is_empty());
    }
}
