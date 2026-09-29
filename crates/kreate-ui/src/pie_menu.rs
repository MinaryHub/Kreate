//! Pie menus: a handful of choices around the pointer, chosen by direction.
//!
//! Blender's, and for Blender's reason: the command comes to where the hand
//! is. Press the key and flick — the choice is the *direction*, so it is made
//! without looking and gets faster until it is a gesture rather than a lookup.
//!
//! # Two ways to choose
//!
//! **Hold and release.** Press the key, move toward an entry, let go: the
//! entry under the pointer runs. This is the fast one.
//!
//! **Tap, then click.** Press and release without moving and the pie stays
//! open, for somebody still learning where things are; a click chooses.
//!
//! Either way the pointer only has to be in the right *direction* — outside a
//! small dead zone at the centre, nearest slot wins, however far out it is.
//! Distance does not matter, which is what makes the flick work.
//!
//! # Slot order
//!
//! Blender's: left, right, bottom, top, then the diagonals top-left,
//! top-right, bottom-left, bottom-right. A pie's entries are listed in that
//! order, so the first two of every pie are the left and right flick — the
//! easiest two to learn.

use egui::{Color32, Key, Pos2, Rect, Stroke, Vec2};

/// Where each slot points, in screen space (y down).
pub fn slot_direction(slot: usize) -> Vec2 {
    let d = std::f32::consts::FRAC_1_SQRT_2;
    match slot {
        0 => Vec2::new(-1.0, 0.0),
        1 => Vec2::new(1.0, 0.0),
        2 => Vec2::new(0.0, 1.0),
        3 => Vec2::new(0.0, -1.0),
        4 => Vec2::new(-d, -d),
        5 => Vec2::new(d, -d),
        6 => Vec2::new(-d, d),
        _ => Vec2::new(d, d),
    }
}

/// The most slots a pie has. Eight directions are what a flick can tell
/// apart reliably; a ninth would be a list, and a list is a menu.
pub const MAX_SLOTS: usize = 8;

/// Inside this radius the pointer chooses nothing, so a tap can leave the
/// pie open without choosing whatever the hand drifted toward.
pub const DEAD_ZONE: f32 = 18.0;

/// The slot the pointer points at, or `None` inside the dead zone.
///
/// Nearest by angle among the `count` slots there are — so a three-entry pie
/// takes the whole lower half for its third entry rather than leaving most
/// directions empty.
pub fn slot_at(center: Pos2, pointer: Pos2, count: usize) -> Option<usize> {
    let offset = pointer - center;
    if count == 0 || offset.length() < DEAD_ZONE {
        return None;
    }
    let dir = offset.normalized();
    (0..count.min(MAX_SLOTS)).max_by(|a, b| {
        dir.dot(slot_direction(*a))
            .partial_cmp(&dir.dot(slot_direction(*b)))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// One entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PieItem {
    pub label: String,
    pub enabled: bool,
    /// Whether this entry is the current state — the shading in use, the
    /// mode you are in — drawn lit.
    pub current: bool,
}

/// An open pie.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenPie {
    /// Which pie: the application's name for it.
    pub name: String,
    pub center: Pos2,
    /// The key that opened it, whose release chooses. `None` for a pie opened
    /// some other way, which then waits for a click.
    pub key: Option<Key>,
    /// When it opened, so a quick tap can be told from a hold.
    pub opened_at: f64,
}

/// What the application keeps.
#[derive(Clone, Debug, Default)]
pub struct PieState {
    pub open: Option<OpenPie>,
}

impl PieState {
    pub fn open(&mut self, name: &str, center: Pos2, key: Option<Key>, now: f64) {
        self.open = Some(OpenPie {
            name: name.to_string(),
            center,
            key,
            opened_at: now,
        });
    }
}

/// How long a press has to last to count as a hold. Shorter is a tap, and the
/// pie stays open for a click.
const HOLD: f64 = 0.25;

/// Draws the open pie, if any. Returns the chosen entry's index.
pub fn pie_menu(ctx: &egui::Context, state: &mut PieState, title: &str, items: &[PieItem]) -> Option<usize> {
    let open = state.open.clone()?;
    let now = ctx.input(|i| i.time);
    let pointer = ctx.input(|i| i.pointer.hover_pos()).unwrap_or(open.center);
    let hovered = slot_at(open.center, pointer, items.len());

    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("pie")));
    let theme = crate::theme::of(ctx);
    let radius = 110.0;
    // The centre: a ring, and a line toward where the pointer points, so the
    // direction being read is visible before anything is chosen.
    painter.circle_stroke(open.center, DEAD_ZONE, Stroke::new(2.0, theme.border_strong));
    if hovered.is_some() {
        let toward = (pointer - open.center).normalized();
        painter.line_segment(
            [open.center + toward * (DEAD_ZONE - 4.0), open.center + toward * (DEAD_ZONE + 6.0)],
            Stroke::new(3.0, theme.accent),
        );
    }
    let font = egui::FontId::proportional(13.0);
    // The title on a backing of its own: a pie opens wherever the pointer
    // is, often over a white model, and faint text on white is no text.
    let heading = painter.layout_no_wrap(title.to_string(), egui::FontId::proportional(11.0), theme.text_dim);
    let backing = Rect::from_center_size(
        open.center + Vec2::new(0.0, -DEAD_ZONE - 14.0),
        heading.size() + Vec2::new(12.0, 4.0),
    );
    painter.rect_filled(backing, theme.radius, theme.base);
    painter.galley(backing.center() - heading.size() / 2.0, heading, theme.text_dim);
    for (slot, item) in items.iter().enumerate().take(MAX_SLOTS) {
        let at = open.center + slot_direction(slot) * radius;
        let galley = painter.layout_no_wrap(item.label.clone(), font.clone(), theme.text);
        let size = galley.size() + Vec2::new(20.0, 10.0);
        // Anchored on the side facing the centre, so a long label grows
        // outward instead of across its neighbour.
        let d = slot_direction(slot);
        let outward = if d.x.abs() > 0.3 { d.x.signum() * size.x / 2.0 } else { 0.0 };
        let rect = Rect::from_center_size(at + Vec2::new(outward, 0.0), size);
        let lit = hovered == Some(slot) && item.enabled;
        let fill = if lit {
            theme.accent
        } else if item.current {
            theme.accent_wash
        } else {
            theme.raised
        };
        painter.rect_filled(rect, theme.radius, fill);
        painter.rect_stroke(rect, theme.radius, Stroke::new(1.0, theme.border), egui::StrokeKind::Inside);
        let colour = if !item.enabled {
            theme.text_faint
        } else if lit {
            Color32::BLACK
        } else {
            theme.text
        };
        painter.galley(rect.center() - galley.size() / 2.0, galley, colour);
    }
    ctx.request_repaint();

    let (clicked, escape, right) = ctx.input(|i| {
        (
            i.pointer.primary_pressed(),
            i.key_pressed(Key::Escape),
            i.pointer.secondary_pressed(),
        )
    });
    let released = open.key.is_some_and(|k| ctx.input(|i| i.key_released(k)));
    let held = now - open.opened_at >= HOLD;

    let mut chosen = None;
    if clicked || (released && (held || hovered.is_some())) {
        chosen = hovered.filter(|s| items.get(*s).is_some_and(|i| i.enabled));
        // A release in the dead zone after a quick tap leaves the pie up for
        // a click; anything else closes it.
        if chosen.is_some() || clicked || held {
            state.open = None;
        }
    }
    if escape || right {
        state.open = None;
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_flick_finds_its_own_slot() {
        let c = Pos2::new(100.0, 100.0);
        for slot in 0..MAX_SLOTS {
            let p = c + slot_direction(slot) * 60.0;
            assert_eq!(slot_at(c, p, MAX_SLOTS), Some(slot), "slot {slot}");
        }
    }

    #[test]
    fn the_first_four_are_left_right_bottom_top() {
        // Blender's order, which is what makes a pie learnable across
        // applications: left and right are the first two of every pie.
        let c = Pos2::ZERO;
        assert_eq!(slot_at(c, Pos2::new(-50.0, 3.0), 8), Some(0));
        assert_eq!(slot_at(c, Pos2::new(50.0, -3.0), 8), Some(1));
        assert_eq!(slot_at(c, Pos2::new(2.0, 50.0), 8), Some(2));
        assert_eq!(slot_at(c, Pos2::new(-2.0, -50.0), 8), Some(3));
    }

    #[test]
    fn distance_does_not_matter_but_the_centre_chooses_nothing() {
        let c = Pos2::ZERO;
        assert_eq!(slot_at(c, Pos2::new(-30.0, 0.0), 4), Some(0));
        assert_eq!(slot_at(c, Pos2::new(-900.0, 0.0), 4), Some(0));
        assert_eq!(slot_at(c, Pos2::new(-5.0, 2.0), 4), None);
    }

    #[test]
    fn a_small_pie_gives_every_direction_to_something() {
        // Three entries: left, right, bottom. Pointing up-left is nearest to
        // left, not nothing — an empty direction is a flick that does nothing.
        let c = Pos2::ZERO;
        assert_eq!(slot_at(c, Pos2::new(-40.0, -30.0), 3), Some(0));
        assert_eq!(slot_at(c, Pos2::new(40.0, -30.0), 3), Some(1));
        assert!(slot_at(c, Pos2::new(0.0, -40.0), 3).is_some());
        assert_eq!(slot_at(c, Pos2::new(0.0, -40.0), 0), None);
    }
}
