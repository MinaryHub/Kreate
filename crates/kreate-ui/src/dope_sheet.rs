//! The dope sheet: every key of the selected objects, on a grid of time.
//!
//! Blender's Dope Sheet, its core: a row per animated object and a row under
//! it per property, a diamond per key, and the editing an animator does all
//! day — select, box-select, drag to retime, delete, change how a key eases.
//! It changes nothing itself: it draws a [`SheetView`] and returns
//! [`SheetEdit`]s, which the application turns into the `anim.keys_*`
//! operators, so every edit here undoes and replays.
//!
//! # Retiming moves by whole frames
//!
//! A drag moves the selected keys by however many whole frames the pointer
//! has travelled, drawn moved while the pointer is down and sent as one edit
//! when it lets go. Whole frames because keys are on frames; a key dragged to
//! 12.4 is a key that no longer lines up with anything.

use egui::{Color32, Key, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use crate::timeline::{frame_to_x, tick_step, visible, x_to_frame};

/// One key in a row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheetKey {
    pub frame: f64,
    pub selected: bool,
}

/// One row: an object's summary, or one of its properties.
#[derive(Clone, Debug, PartialEq)]
pub struct SheetRow {
    pub label: String,
    /// 0 for an object, 1 for a property under it.
    pub depth: u8,
    /// What the application calls the row back by — `anim::row_target`'s text.
    pub target: String,
    pub keys: Vec<SheetKey>,
}

/// What one frame of the panel shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SheetView {
    pub rows: Vec<SheetRow>,
    pub frame: f64,
    pub start: i32,
    pub end: i32,
}

/// What the pointer and the keys asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum SheetEdit {
    /// Select the keys of these rows between two frames.
    Select { targets: Vec<String>, frame_min: f64, frame_max: f64, extend: bool },
    /// 0 select all, 1 deselect all, 2 invert — `anim.keys_select_all`'s order.
    SelectAll(u32),
    Move { offset: f64 },
    Delete,
    /// `anim.keys_interpolation`'s order: constant, linear, Bézier.
    Interpolation(u32),
    SetFrame(f64),
}

/// The panel's strings.
pub struct SheetText {
    pub empty: String,
    pub interpolation: String,
    pub constant: String,
    pub linear: String,
    pub bezier: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
enum Drag {
    #[default]
    None,
    /// Moving the selection; `from` is where the pointer started, in frames.
    Move { from: f64 },
    /// Moving the selection with `G`: follows the pointer until a click.
    Grab { from: f64 },
    /// A box from this corner.
    Box { from: Pos2 },
    Scrub,
}

/// What the panel remembers, per area.
#[derive(Clone, Debug, Default)]
pub struct SheetState {
    pub rect: Option<Rect>,
    /// The frames on screen, once somebody has zoomed or panned.
    range: Option<(f64, f64)>,
    drag: Drag,
    /// Where the interpolation menu is open, if it is.
    menu: Option<Pos2>,
    /// Object rows folded shut, by target: their property rows are hidden and
    /// their summary row stands for them, as in Blender.
    collapsed: std::collections::BTreeSet<String>,
}

/// How wide the column of row names is.
pub const NAMES: f32 = 150.0;
/// How tall the ruler across the top is.
pub const RULER: f32 = 20.0;
/// How tall a row is.
pub const ROW: f32 = 20.0;
const KEY_RADIUS: f32 = 5.0;

/// Where things are, for one frame of the panel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub keys: Rect,
    pub range: (f64, f64),
}

impl Layout {
    pub fn new(rect: Rect, range: (f64, f64)) -> Self {
        Self {
            keys: Rect::from_min_max(Pos2::new(rect.min.x + NAMES, rect.min.y + RULER), rect.max),
            range,
        }
    }

    pub fn x(&self, frame: f64) -> f32 {
        frame_to_x(frame, self.range, self.keys)
    }

    pub fn frame(&self, x: f32) -> f64 {
        x_to_frame(x, self.range, self.keys)
    }

    pub fn row_y(&self, row: usize) -> f32 {
        self.keys.min.y + (row as f32 + 0.5) * ROW
    }

    pub fn row_at(&self, y: f32) -> Option<usize> {
        (y >= self.keys.min.y).then(|| ((y - self.keys.min.y) / ROW) as usize)
    }

    /// Points a frame is wide on screen.
    pub fn frame_width(&self) -> f32 {
        self.keys.width() / (self.range.1 - self.range.0) as f32
    }
}

/// The key under a point, if one is close enough to be meant.
pub fn key_at(view: &SheetView, layout: &Layout, p: Pos2) -> Option<(usize, f64)> {
    let row = layout.row_at(p.y)?;
    let keys = &view.rows.get(row)?.keys;
    keys.iter()
        .map(|k| (k.frame, (layout.x(k.frame) - p.x).abs()))
        .filter(|(_, d)| *d <= KEY_RADIUS + 3.0)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(f, _)| (row, f))
}

/// The rows a vertical span covers, clamped to the rows there are.
fn rows_between(layout: &Layout, a: f32, b: f32, count: usize) -> std::ops::Range<usize> {
    let (top, bottom) = (a.min(b), a.max(b));
    let first = layout.row_at(top.max(layout.keys.min.y)).unwrap_or(0);
    let last = layout.row_at(bottom).map(|r| r + 1).unwrap_or(0);
    first.min(count)..last.min(count)
}

/// Draws the dope sheet. Returns what was asked for.
pub fn dope_sheet(ui: &mut Ui, view: &SheetView, state: &mut SheetState, text: &SheetText) -> Vec<SheetEdit> {
    // Which object rows have rows under them, and the rows left once the
    // folded ones' children are taken out.
    let parents: std::collections::BTreeSet<String> = view
        .rows
        .windows(2)
        .filter(|w| w[0].depth == 0 && w[1].depth > 0)
        .map(|w| w[0].target.clone())
        .collect();
    let mut folded = false;
    let folded_view = SheetView {
        rows: view
            .rows
            .iter()
            .filter(|r| {
                if r.depth == 0 {
                    folded = state.collapsed.contains(&r.target);
                    true
                } else {
                    !folded
                }
            })
            .cloned()
            .collect(),
        ..view.clone()
    };
    let view = &folded_view;
    let mut out = Vec::new();
    let theme = crate::theme::from_ui(ui);
    let rect = ui.available_rect_before_wrap();
    if rect.width() < NAMES + 40.0 || rect.height() < RULER + ROW {
        return out;
    }
    state.rect = Some(rect);
    let response = ui.allocate_rect(rect, Sense::click_and_drag());
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, theme.sunken);

    let range = state.range.unwrap_or_else(|| visible(view.start, view.end));
    let layout = Layout::new(rect, range);
    let hovered = ui.rect_contains_pointer(rect);
    let pointer = ui.input(|i| i.pointer.interact_pos());

    // --- zoom and pan ------------------------------------------------------------
    if hovered {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            if let Some(p) = pointer {
                let anchor = layout.frame(p.x);
                let k = (1.0 - scroll as f64 * 0.0015).clamp(0.5, 2.0);
                let (a, b) = range;
                state.range = Some((anchor + (a - anchor) * k, anchor + (b - anchor) * k));
            }
        }
    }
    if response.dragged_by(egui::PointerButton::Middle) {
        let df = -(response.drag_delta().x / layout.frame_width()) as f64;
        state.range = Some((range.0 + df, range.1 + df));
    }
    let range = state.range.unwrap_or(range);
    let layout = Layout::new(rect, range);

    // --- the grid ----------------------------------------------------------------
    let ruler = Rect::from_min_max(Pos2::new(layout.keys.min.x, rect.min.y), Pos2::new(rect.max.x, layout.keys.min.y));
    painter.rect_filled(ruler, 0.0, theme.base);
    let names = Rect::from_min_max(rect.min, Pos2::new(layout.keys.min.x, rect.max.y));
    let dim = Color32::from_black_alpha(50);
    painter.rect_filled(Rect::from_min_max(layout.keys.min, Pos2::new(layout.x(view.start as f64), rect.max.y)), 0.0, dim);
    painter.rect_filled(Rect::from_min_max(Pos2::new(layout.x(view.end as f64), layout.keys.min.y), rect.max), 0.0, dim);
    let step = tick_step(layout.frame_width(), 44.0);
    let mut f = ((range.0 / step as f64).ceil() as i64) * step as i64;
    while (f as f64) <= range.1 {
        let x = layout.x(f as f64);
        painter.line_segment([Pos2::new(x, layout.keys.min.y), Pos2::new(x, rect.max.y)], Stroke::new(1.0, theme.border));
        painter.text(Pos2::new(x, rect.min.y + 3.0), egui::Align2::CENTER_TOP, f.to_string(), egui::FontId::proportional(10.0), theme.text_faint);
        f += step as i64;
    }

    if view.rows.is_empty() {
        painter.text(layout.keys.center(), egui::Align2::CENTER_CENTER, &text.empty, egui::FontId::proportional(12.0), theme.text_faint);
    }

    // What a drag in progress does to the selected keys, in whole frames.
    let offset = match (&state.drag, pointer) {
        (Drag::Move { from } | Drag::Grab { from }, Some(p)) => (layout.frame(p.x) - from).round(),
        _ => 0.0,
    };

    // --- rows ----------------------------------------------------------------------
    painter.rect_filled(names, 0.0, theme.base);
    for (i, row) in view.rows.iter().enumerate() {
        let y = layout.row_y(i);
        if row.depth == 0 {
            painter.rect_filled(
                Rect::from_center_size(Pos2::new(rect.center().x, y), Vec2::new(rect.width(), ROW - 1.0)),
                0.0,
                Color32::from_white_alpha(6),
            );
        }
        // The fold: a triangle pointing right when shut, down when open.
        if row.depth == 0 && parents.contains(&row.target) {
            let c = Pos2::new(names.min.x + 9.0, y);
            let shut = state.collapsed.contains(&row.target);
            let points = if shut {
                vec![c + Vec2::new(-2.5, -4.0), c + Vec2::new(3.5, 0.0), c + Vec2::new(-2.5, 4.0)]
            } else {
                vec![c + Vec2::new(-4.0, -2.5), c + Vec2::new(4.0, -2.5), c + Vec2::new(0.0, 3.5)]
            };
            painter.add(egui::Shape::convex_polygon(points, theme.text_dim, Stroke::NONE));
        }
        painter.text(
            Pos2::new(names.min.x + 18.0 + row.depth as f32 * 12.0, y),
            egui::Align2::LEFT_CENTER,
            &row.label,
            egui::FontId::proportional(12.0),
            if row.depth == 0 { theme.text } else { theme.text_dim },
        );
        for key in &row.keys {
            let frame = if key.selected { key.frame + offset } else { key.frame };
            let x = layout.x(frame);
            if x < layout.keys.min.x - KEY_RADIUS || x > rect.max.x + KEY_RADIUS {
                continue;
            }
            let r = if row.depth == 0 { KEY_RADIUS } else { KEY_RADIUS - 1.0 };
            painter.add(egui::Shape::convex_polygon(
                vec![Pos2::new(x, y - r), Pos2::new(x + r, y), Pos2::new(x, y + r), Pos2::new(x - r, y)],
                if key.selected { theme.accent } else { Color32::from_gray(200) },
                Stroke::new(1.0, Color32::BLACK),
            ));
        }
    }

    // The clock.
    let x = layout.x(view.frame);
    painter.line_segment([Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)], Stroke::new(2.0, theme.accent));

    // A box being drawn.
    if let (Drag::Box { from }, Some(p)) = (&state.drag, pointer) {
        let b = Rect::from_two_pos(*from, p);
        painter.rect_filled(b, 0.0, theme.accent_wash);
        painter.rect_stroke(b, 0.0, Stroke::new(1.0, theme.accent), egui::StrokeKind::Inside);
    }

    // --- the pointer ---------------------------------------------------------------
    let shift = ui.input(|i| i.modifiers.shift);
    let press = ui.input(|i| i.pointer.primary_pressed()) && response.hovered();
    let release = ui.input(|i| i.pointer.primary_released());

    if let Drag::Grab { .. } = state.drag {
        // `G`: the keys follow the pointer; a click puts them down, Escape or
        // a right-click puts them back.
        let cancel = ui.input(|i| i.key_pressed(Key::Escape) || i.pointer.secondary_pressed());
        let confirm = ui.input(|i| i.pointer.primary_pressed() || i.key_pressed(Key::Enter));
        if cancel {
            state.drag = Drag::None;
        } else if confirm {
            if offset != 0.0 {
                out.push(SheetEdit::Move { offset });
            }
            state.drag = Drag::None;
        }
    } else if press && state.menu.is_none() {
        if let Some(p) = pointer {
            if p.y < layout.keys.min.y && p.x >= layout.keys.min.x {
                state.drag = Drag::Scrub;
            } else if p.x < layout.keys.min.x {
                // A click on an object's name folds it open or shut.
                if let Some(row) = layout.row_at(p.y).and_then(|i| view.rows.get(i)) {
                    if row.depth == 0 && parents.contains(&row.target) && !state.collapsed.remove(&row.target) {
                        state.collapsed.insert(row.target.clone());
                    }
                }
            } else if let Some((row, frame)) = key_at(view, &layout, p) {
                let already = view.rows[row].keys.iter().any(|k| k.frame == frame && k.selected);
                if !already || shift {
                    out.push(SheetEdit::Select {
                        targets: vec![view.rows[row].target.clone()],
                        frame_min: frame - 1e-3,
                        frame_max: frame + 1e-3,
                        extend: shift,
                    });
                }
                state.drag = Drag::Move { from: layout.frame(p.x) };
            } else if p.x >= layout.keys.min.x {
                state.drag = Drag::Box { from: p };
            }
        }
    }

    if let (Drag::Scrub, Some(p)) = (&state.drag, pointer) {
        let frame = layout.frame(p.x).round();
        if (frame - view.frame).abs() > 1e-9 {
            out.push(SheetEdit::SetFrame(frame));
        }
    }

    if release {
        match std::mem::take(&mut state.drag) {
            Drag::Move { .. } if offset != 0.0 => out.push(SheetEdit::Move { offset }),
            Drag::Box { from } => {
                let to = pointer.unwrap_or(from);
                if from.distance(to) < 4.0 {
                    if !shift {
                        out.push(SheetEdit::SelectAll(1));
                    }
                } else {
                    let rows = rows_between(&layout, from.y, to.y, view.rows.len());
                    let (a, b) = (layout.frame(from.x.min(to.x)), layout.frame(from.x.max(to.x)));
                    let targets: Vec<String> = view.rows[rows].iter().map(|r| r.target.clone()).collect();
                    if !targets.is_empty() {
                        out.push(SheetEdit::Select { targets, frame_min: a, frame_max: b, extend: shift });
                    } else if !shift {
                        out.push(SheetEdit::SelectAll(1));
                    }
                }
            }
            Drag::Grab { from } => state.drag = Drag::Grab { from },
            _ => {}
        }
    }

    // --- keys ------------------------------------------------------------------------
    if hovered && !ui.ctx().egui_wants_keyboard_input() && state.drag == Drag::None {
        let (alt, ctrl) = ui.input(|i| (i.modifiers.alt, i.modifiers.command));
        let pressed = |k: Key| ui.input(|i| i.key_pressed(k));
        if pressed(Key::A) {
            out.push(SheetEdit::SelectAll(if alt { 1 } else { 0 }));
        }
        if ctrl && pressed(Key::I) {
            out.push(SheetEdit::SelectAll(2));
        }
        if pressed(Key::X) || pressed(Key::Delete) {
            out.push(SheetEdit::Delete);
        }
        if pressed(Key::G) {
            if let Some(p) = pointer.or_else(|| ui.input(|i| i.pointer.hover_pos())) {
                state.drag = Drag::Grab { from: layout.frame(p.x) };
            }
        }
        if pressed(Key::T) {
            state.menu = pointer.or_else(|| ui.input(|i| i.pointer.hover_pos()));
        }
    }

    // --- the interpolation menu ------------------------------------------------------
    if let Some(at) = state.menu {
        let mut close = false;
        let area = egui::Area::new(ui.id().with("sheet-interpolation"))
            .order(egui::Order::Foreground)
            .fixed_pos(at)
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(egui::RichText::new(&text.interpolation).strong());
                    ui.separator();
                    for (i, label) in [&text.constant, &text.linear, &text.bezier].into_iter().enumerate() {
                        if ui.button(label).clicked() {
                            out.push(SheetEdit::Interpolation(i as u32));
                            close = true;
                        }
                    }
                });
            });
        let away = ui.input(|i| i.pointer.any_pressed())
            && pointer.is_some_and(|p| !area.response.rect.contains(p));
        if close || away || ui.input(|i| i.key_pressed(Key::Escape)) {
            state.menu = None;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> SheetView {
        let keys = |frames: &[f64], selected: &[f64]| -> Vec<SheetKey> {
            frames.iter().map(|f| SheetKey { frame: *f, selected: selected.contains(f) }).collect()
        };
        SheetView {
            rows: vec![
                SheetRow { label: "Cube".into(), depth: 0, target: "7/".into(), keys: keys(&[1.0, 20.0, 40.0], &[]) },
                SheetRow { label: "Location".into(), depth: 1, target: "7/location".into(), keys: keys(&[1.0, 20.0], &[]) },
                SheetRow { label: "Scale".into(), depth: 1, target: "7/scale".into(), keys: keys(&[40.0], &[]) },
            ],
            frame: 1.0,
            start: 1,
            end: 60,
        }
    }

    struct Rig {
        ctx: egui::Context,
        state: SheetState,
        view: SheetView,
        time: f64,
    }

    impl Rig {
        fn new(view: SheetView) -> Self {
            let mut rig = Self { ctx: egui::Context::default(), state: SheetState::default(), view, time: 0.0 };
            rig.frame(vec![]);
            rig.frame(vec![]);
            rig
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> Vec<SheetEdit> {
            self.time += 1.0 / 60.0;
            let raw = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 300.0))),
                time: Some(self.time),
                events,
                ..Default::default()
            };
            let text = SheetText {
                empty: String::new(),
                interpolation: String::new(),
                constant: "C".into(),
                linear: "L".into(),
                bezier: "B".into(),
            };
            let mut out = Vec::new();
            let (state, view) = (&mut self.state, &self.view);
            let _ = self.ctx.run_ui(raw, |ui| out = dope_sheet(ui, view, state, &text));
            out
        }

        fn layout(&self) -> Layout {
            Layout::new(self.state.rect.unwrap(), visible(self.view.start, self.view.end))
        }

        fn at(&self, row: usize, frame: f64) -> Pos2 {
            let l = self.layout();
            Pos2::new(l.x(frame), l.row_y(row))
        }

        fn drag(&mut self, a: Pos2, b: Pos2, shift: bool) -> Vec<SheetEdit> {
            let modifiers = egui::Modifiers { shift, ..Default::default() };
            let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers };
            let mut out = Vec::new();
            out.extend(self.frame(vec![egui::Event::PointerMoved(a)]));
            out.extend(self.frame(vec![button(a, true)]));
            for i in 1..=4 {
                out.extend(self.frame(vec![egui::Event::PointerMoved(a + (b - a) * (i as f32 / 4.0))]));
            }
            out.extend(self.frame(vec![button(b, false)]));
            out
        }
    }

    #[test]
    fn a_frame_and_a_row_go_to_a_place_and_back() {
        let l = Layout::new(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 200.0)), visible(1, 100));
        for f in [1.0, 33.0, 100.0] {
            assert!((l.frame(l.x(f)) - f).abs() < 1e-3);
        }
        for r in 0..4 {
            assert_eq!(l.row_at(l.row_y(r)), Some(r));
        }
        assert_eq!(l.row_at(l.keys.min.y - 5.0), None, "the ruler is not a row");
    }

    #[test]
    fn clicking_a_key_selects_it_in_its_own_row() {
        let mut rig = Rig::new(view());
        let p = rig.at(1, 20.0);
        let edits = rig.drag(p, p, false);
        assert_eq!(
            edits,
            vec![SheetEdit::Select { targets: vec!["7/location".into()], frame_min: 20.0 - 1e-3, frame_max: 20.0 + 1e-3, extend: false }]
        );
    }

    #[test]
    fn dragging_a_key_moves_the_selection_by_whole_frames() {
        let mut rig = Rig::new(view());
        let a = rig.at(0, 20.0);
        let b = a + Vec2::new(rig.layout().frame_width() * 5.3, 0.0);
        let edits = rig.drag(a, b, false);
        assert!(matches!(edits.first(), Some(SheetEdit::Select { .. })), "{edits:?}");
        assert_eq!(edits.last(), Some(&SheetEdit::Move { offset: 5.0 }), "{edits:?}");
    }

    #[test]
    fn a_box_selects_the_rows_it_crosses_between_its_frames() {
        let mut rig = Rig::new(view());
        let a = rig.at(1, 10.0) - Vec2::new(0.0, 6.0);
        let b = rig.at(2, 45.0) + Vec2::new(0.0, 6.0);
        let edits = rig.drag(a, b, false);
        match edits.as_slice() {
            [SheetEdit::Select { targets, frame_min, frame_max, extend: false }] => {
                assert_eq!(targets, &vec!["7/location".to_string(), "7/scale".to_string()]);
                assert!((frame_min - 10.0).abs() < 0.5 && (frame_max - 45.0).abs() < 0.5, "{frame_min}..{frame_max}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_click_on_nothing_deselects_and_x_deletes() {
        let mut rig = Rig::new(view());
        let p = rig.at(2, 10.0);
        assert_eq!(rig.drag(p, p, false), vec![SheetEdit::SelectAll(1)]);
        let key = |k, pressed| egui::Event::Key { key: k, physical_key: None, pressed, repeat: false, modifiers: Default::default() };
        let edits = rig.frame(vec![egui::Event::PointerMoved(p), key(Key::X, true)]);
        assert_eq!(edits, vec![SheetEdit::Delete]);
    }

    #[test]
    fn the_ruler_scrubs_to_whole_frames() {
        let mut rig = Rig::new(view());
        let l = rig.layout();
        let a = Pos2::new(l.x(10.0), l.keys.min.y - RULER / 2.0);
        let b = Pos2::new(l.x(30.2), a.y);
        let edits = rig.drag(a, b, false);
        let frames: Vec<f64> = edits.iter().filter_map(|e| match e { SheetEdit::SetFrame(f) => Some(*f), _ => None }).collect();
        assert!(!frames.is_empty() && frames.iter().all(|f| f.fract() == 0.0), "{frames:?}");
        assert_eq!(*frames.last().unwrap(), 30.0);
    }

    #[test]
    fn clicking_an_objects_name_folds_its_rows_and_a_click_on_the_fold_opens_it() {
        let mut rig = Rig::new(view());
        let name = Pos2::new(rig.state.rect.unwrap().min.x + 20.0, rig.layout().row_y(0));
        rig.drag(name, name, false);
        assert!(rig.state.collapsed.contains("7/"), "the click did not fold the object");
        // Folded, the second row is gone: a click where Location's key was
        // lands on nothing and selects nothing.
        let edits = rig.drag(rig.at(1, 1.0), rig.at(1, 1.0), false);
        assert!(
            !edits.iter().any(|e| matches!(e, SheetEdit::Select { targets, .. } if targets == &vec!["7/location".to_string()])),
            "a folded row can still be clicked: {edits:?}"
        );
        rig.drag(name, name, false);
        assert!(rig.state.collapsed.is_empty());
    }
}

