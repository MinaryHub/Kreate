//! The NLA editor: each object's strips as bars on a grid of time.
//!
//! Blender's Nonlinear Animation editor, its core: a row per object with its
//! action on top, and under it a row per strip, top strip first — the order
//! they are laid over each other. A strip is dragged along the time to move
//! it, by whole frames; `X` takes it off, `M` mutes it, and the button on an
//! object's row pushes its action down into a new strip. It changes nothing
//! itself: it draws an [`NlaView`] and returns [`NlaEdit`]s, which the
//! application turns into the `anim.nla_*` operators, so every edit undoes
//! and replays.
//!
//! Which strip is picked is the panel's to remember, not the document's: the
//! strips' own fields have no selection, and the operators name a strip by
//! its place.

use egui::{Color32, Key, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use crate::timeline::{frame_to_x, tick_step, visible, x_to_frame};

/// One strip, where it sits in time.
#[derive(Clone, Debug, PartialEq)]
pub struct NlaStripView {
    /// Its place among the object's strips, bottom first.
    pub index: usize,
    pub label: String,
    pub start: f64,
    pub end: f64,
    pub mute: bool,
}

/// One object: its action, if it has one, and its strips.
#[derive(Clone, Debug, PartialEq)]
pub struct NlaTrack {
    /// The object's name, which the application selects it by.
    pub object: String,
    /// The action on top of the strips, and the frames its keys span.
    pub action: Option<(String, f64, f64)>,
    /// Bottom first, as the object holds them.
    pub strips: Vec<NlaStripView>,
}

/// What one frame of the panel shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NlaView {
    pub tracks: Vec<NlaTrack>,
    pub frame: f64,
    pub start: i32,
    pub end: i32,
}

/// What the pointer and the keys asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum NlaEdit {
    /// The object a strip belongs to is to be the active one.
    Select {
        object: String,
        index: usize,
    },
    /// A strip moved so that it starts at `start`.
    Move {
        object: String,
        index: usize,
        start: f64,
    },
    /// A strip muted, or heard again.
    Mute {
        object: String,
        index: usize,
        mute: bool,
    },
    Remove {
        object: String,
        index: usize,
    },
    /// The object's action pushed down into a strip.
    PushDown {
        object: String,
    },
    SetFrame(f64),
}

/// The panel's strings.
pub struct NlaText {
    pub empty: String,
    pub push_down: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
enum Drag {
    #[default]
    None,
    /// Moving a strip; `from` is where the pointer started, in frames.
    Move {
        from: f64,
    },
    Scrub,
}

/// What the panel remembers, per area.
#[derive(Clone, Debug, Default)]
pub struct NlaState {
    pub rect: Option<Rect>,
    range: Option<(f64, f64)>,
    drag: Drag,
    /// The strip picked, by its object and place.
    picked: Option<(String, usize)>,
}

/// How wide the column of row names is.
pub const NAMES: f32 = 150.0;
pub const RULER: f32 = 20.0;
pub const ROW: f32 = 22.0;

/// What a row is.
#[derive(Clone, Debug, PartialEq)]
enum Row<'a> {
    Object(&'a NlaTrack),
    Strip(&'a NlaTrack, &'a NlaStripView),
}

/// The rows, top to bottom: each object, then its strips top first.
fn rows(view: &NlaView) -> Vec<Row<'_>> {
    let mut out = Vec::new();
    for track in &view.tracks {
        out.push(Row::Object(track));
        for strip in track.strips.iter().rev() {
            out.push(Row::Strip(track, strip));
        }
    }
    out
}

/// Where things are, for one frame of the panel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub body: Rect,
    pub range: (f64, f64),
}

impl Layout {
    pub fn new(rect: Rect, range: (f64, f64)) -> Self {
        Self {
            body: Rect::from_min_max(Pos2::new(rect.min.x + NAMES, rect.min.y + RULER), rect.max),
            range,
        }
    }

    pub fn x(&self, frame: f64) -> f32 {
        frame_to_x(frame, self.range, self.body)
    }

    pub fn frame(&self, x: f32) -> f64 {
        x_to_frame(x, self.range, self.body)
    }

    pub fn row_y(&self, row: usize) -> f32 {
        self.body.min.y + (row as f32 + 0.5) * ROW
    }

    pub fn row_at(&self, y: f32) -> Option<usize> {
        (y >= self.body.min.y).then(|| ((y - self.body.min.y) / ROW) as usize)
    }

    pub fn frame_width(&self) -> f32 {
        self.body.width() / (self.range.1 - self.range.0) as f32
    }

    /// A bar from one frame to another in a row.
    fn bar(&self, row: usize, start: f64, end: f64) -> Rect {
        let y = self.row_y(row);
        Rect::from_min_max(
            Pos2::new(self.x(start), y - ROW * 0.5 + 3.0),
            Pos2::new(self.x(end).max(self.x(start) + 3.0), y + ROW * 0.5 - 3.0),
        )
    }

    /// Where the Push Down button sits on an object's row.
    fn push_button(&self, row: usize) -> Rect {
        let y = self.row_y(row);
        Rect::from_min_max(
            Pos2::new(self.body.min.x - 26.0, y - 8.0),
            Pos2::new(self.body.min.x - 6.0, y + 8.0),
        )
    }
}

/// Draws the NLA editor. Returns what was asked for.
pub fn nla_editor(
    ui: &mut Ui,
    view: &NlaView,
    state: &mut NlaState,
    text: &NlaText,
) -> Vec<NlaEdit> {
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
    let rows = rows(view);
    // A strip picked that is no longer there is not picked.
    if let Some((object, index)) = &state.picked {
        let there = view
            .tracks
            .iter()
            .any(|t| &t.object == object && t.strips.iter().any(|s| s.index == *index));
        if !there {
            state.picked = None;
        }
    }

    let range = state.range.unwrap_or_else(|| visible(view.start, view.end));
    let layout = Layout::new(rect, range);
    let hovered = ui.rect_contains_pointer(rect);
    let pointer = ui.input(|i| i.pointer.interact_pos());

    // --- zoom and pan ------------------------------------------------------------
    if hovered {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if let (true, Some(p)) = (scroll != 0.0, pointer) {
            let anchor = layout.frame(p.x);
            let k = (1.0 - scroll as f64 * 0.0015).clamp(0.5, 2.0);
            let (a, b) = range;
            state.range = Some((anchor + (a - anchor) * k, anchor + (b - anchor) * k));
        }
    }
    if response.dragged_by(egui::PointerButton::Middle) {
        let df = -(response.drag_delta().x / layout.frame_width()) as f64;
        state.range = Some((range.0 + df, range.1 + df));
    }
    let range = state.range.unwrap_or(range);
    let layout = Layout::new(rect, range);

    // --- the grid ----------------------------------------------------------------
    let ruler = Rect::from_min_max(
        Pos2::new(layout.body.min.x, rect.min.y),
        Pos2::new(rect.max.x, layout.body.min.y),
    );
    painter.rect_filled(ruler, 0.0, theme.base);
    let names = Rect::from_min_max(rect.min, Pos2::new(layout.body.min.x, rect.max.y));
    let dim = Color32::from_black_alpha(50);
    painter.rect_filled(
        Rect::from_min_max(
            layout.body.min,
            Pos2::new(layout.x(view.start as f64), rect.max.y),
        ),
        0.0,
        dim,
    );
    painter.rect_filled(
        Rect::from_min_max(
            Pos2::new(layout.x(view.end as f64), layout.body.min.y),
            rect.max,
        ),
        0.0,
        dim,
    );
    let step = tick_step(layout.frame_width(), 44.0);
    let mut f = ((range.0 / step as f64).ceil() as i64) * step as i64;
    while (f as f64) <= range.1 {
        let x = layout.x(f as f64);
        painter.line_segment(
            [Pos2::new(x, layout.body.min.y), Pos2::new(x, rect.max.y)],
            Stroke::new(1.0, theme.border),
        );
        painter.text(
            Pos2::new(x, rect.min.y + 3.0),
            egui::Align2::CENTER_TOP,
            f.to_string(),
            egui::FontId::proportional(10.0),
            theme.text_faint,
        );
        f += step as i64;
    }
    if rows.is_empty() {
        painter.text(
            layout.body.center(),
            egui::Align2::CENTER_CENTER,
            &text.empty,
            egui::FontId::proportional(12.0),
            theme.text_faint,
        );
    }

    // How far a drag in progress has moved the picked strip, in whole frames.
    let offset = match (&state.drag, pointer) {
        (Drag::Move { from }, Some(p)) => (layout.frame(p.x) - from).round(),
        _ => 0.0,
    };

    // --- rows ----------------------------------------------------------------------
    painter.rect_filled(names, 0.0, theme.base);
    let font = egui::FontId::proportional(12.0);
    for (i, row) in rows.iter().enumerate() {
        let y = layout.row_y(i);
        match row {
            Row::Object(track) => {
                painter.rect_filled(
                    Rect::from_center_size(
                        Pos2::new(rect.center().x, y),
                        Vec2::new(rect.width(), ROW - 1.0),
                    ),
                    0.0,
                    Color32::from_white_alpha(6),
                );
                painter.text(
                    Pos2::new(names.min.x + 8.0, y),
                    egui::Align2::LEFT_CENTER,
                    &track.object,
                    font.clone(),
                    theme.text,
                );
                if let Some((name, a, b)) = &track.action {
                    let bar = layout.bar(i, *a, *b);
                    painter.rect_filled(bar, 3.0, Color32::from_rgb(150, 120, 60));
                    painter.text(
                        bar.left_center() + Vec2::new(5.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        name,
                        egui::FontId::proportional(11.0),
                        Color32::WHITE,
                    );
                    // Push Down: a downward arrow at the end of the name.
                    let b = layout.push_button(i);
                    let over = pointer.is_some_and(|p| b.contains(p));
                    painter.rect_filled(
                        b,
                        3.0,
                        if over {
                            theme.accent_wash
                        } else {
                            Color32::from_white_alpha(12)
                        },
                    );
                    let c = b.center();
                    painter.add(egui::Shape::convex_polygon(
                        vec![
                            c + Vec2::new(-4.0, -3.0),
                            c + Vec2::new(4.0, -3.0),
                            c + Vec2::new(0.0, 4.0),
                        ],
                        theme.text_dim,
                        Stroke::NONE,
                    ));
                }
            }
            Row::Strip(track, strip) => {
                painter.text(
                    Pos2::new(names.min.x + 20.0, y),
                    egui::Align2::LEFT_CENTER,
                    &strip.label,
                    font.clone(),
                    if strip.mute {
                        theme.text_faint
                    } else {
                        theme.text_dim
                    },
                );
                let picked = state.picked.as_ref() == Some(&(track.object.clone(), strip.index));
                let shift = if picked { offset } else { 0.0 };
                let bar = layout.bar(i, strip.start + shift, strip.end + shift);
                let fill = if strip.mute {
                    Color32::from_gray(70)
                } else {
                    Color32::from_rgb(70, 110, 160)
                };
                painter.rect_filled(bar, 3.0, fill);
                if picked {
                    painter.rect_stroke(
                        bar,
                        3.0,
                        Stroke::new(2.0, theme.accent),
                        egui::StrokeKind::Inside,
                    );
                }
                painter.text(
                    bar.left_center() + Vec2::new(5.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    &strip.label,
                    egui::FontId::proportional(11.0),
                    if strip.mute {
                        Color32::from_gray(150)
                    } else {
                        Color32::WHITE
                    },
                );
            }
        }
    }

    // The clock.
    let x = layout.x(view.frame);
    painter.line_segment(
        [Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)],
        Stroke::new(2.0, theme.accent),
    );

    // --- the pointer ---------------------------------------------------------------
    let press = ui.input(|i| i.pointer.primary_pressed()) && response.hovered();
    let release = ui.input(|i| i.pointer.primary_released());
    if press {
        if let Some(p) = pointer {
            let row = layout
                .row_at(p.y)
                .and_then(|r| rows.get(r).map(|row| (r, row)));
            if p.y < layout.body.min.y && p.x >= layout.body.min.x {
                state.drag = Drag::Scrub;
            } else if let Some((r, Row::Object(track))) = row {
                if track.action.is_some() && layout.push_button(r).contains(p) {
                    out.push(NlaEdit::PushDown {
                        object: track.object.clone(),
                    });
                }
            } else if let Some((r, Row::Strip(track, strip))) = row {
                let hit = layout
                    .bar(r, strip.start, strip.end)
                    .expand(2.0)
                    .contains(p)
                    || p.x < layout.body.min.x;
                if hit {
                    state.picked = Some((track.object.clone(), strip.index));
                    out.push(NlaEdit::Select {
                        object: track.object.clone(),
                        index: strip.index,
                    });
                    if p.x >= layout.body.min.x {
                        state.drag = Drag::Move {
                            from: layout.frame(p.x),
                        };
                    }
                } else {
                    state.picked = None;
                }
            } else {
                state.picked = None;
            }
        }
    }
    if let (Drag::Scrub, Some(p)) = (&state.drag, pointer) {
        let frame = layout.frame(p.x).round();
        if (frame - view.frame).abs() > 1e-9 {
            out.push(NlaEdit::SetFrame(frame));
        }
    }
    if release {
        if let Drag::Move { .. } = std::mem::take(&mut state.drag) {
            if offset != 0.0 {
                if let Some((object, index)) = &state.picked {
                    if let Some(strip) = picked_strip(view, object, *index) {
                        out.push(NlaEdit::Move {
                            object: object.clone(),
                            index: *index,
                            start: strip.start + offset,
                        });
                    }
                }
            }
        }
    }

    // --- keys ------------------------------------------------------------------------
    if hovered && !ui.ctx().egui_wants_keyboard_input() && state.drag == Drag::None {
        let pressed = |k: Key| ui.input(|i| i.key_pressed(k));
        if let Some((object, index)) = state.picked.clone() {
            if pressed(Key::X) || pressed(Key::Delete) {
                out.push(NlaEdit::Remove { object, index });
                state.picked = None;
            } else if pressed(Key::M) {
                if let Some(strip) = picked_strip(view, &object, index) {
                    out.push(NlaEdit::Mute {
                        object,
                        index,
                        mute: !strip.mute,
                    });
                }
            }
        }
    }
    out
}

fn picked_strip<'a>(view: &'a NlaView, object: &str, index: usize) -> Option<&'a NlaStripView> {
    view.tracks
        .iter()
        .find(|t| t.object == object)?
        .strips
        .iter()
        .find(|s| s.index == index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> NlaView {
        let strip = |index, label: &str, start, end| NlaStripView {
            index,
            label: label.into(),
            start,
            end,
            mute: false,
        };
        NlaView {
            tracks: vec![NlaTrack {
                object: "Cube".into(),
                action: Some(("Wave".into(), 1.0, 20.0)),
                strips: vec![strip(0, "Walk", 1.0, 40.0), strip(1, "Sway", 10.0, 30.0)],
            }],
            frame: 1.0,
            start: 1,
            end: 60,
        }
    }

    struct Rig {
        ctx: egui::Context,
        state: NlaState,
        view: NlaView,
        time: f64,
    }

    impl Rig {
        fn new() -> Self {
            let mut rig = Self {
                ctx: egui::Context::default(),
                state: NlaState::default(),
                view: view(),
                time: 0.0,
            };
            rig.frame(vec![]);
            rig.frame(vec![]);
            rig
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> Vec<NlaEdit> {
            self.time += 1.0 / 60.0;
            let raw = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 300.0))),
                time: Some(self.time),
                events,
                ..Default::default()
            };
            let text = NlaText {
                empty: String::new(),
                push_down: String::new(),
            };
            let mut out = Vec::new();
            let (state, view) = (&mut self.state, &self.view);
            let _ = self
                .ctx
                .run_ui(raw, |ui| out = nla_editor(ui, view, state, &text));
            out
        }

        fn layout(&self) -> Layout {
            Layout::new(
                self.state.rect.unwrap(),
                visible(self.view.start, self.view.end),
            )
        }

        fn drag(&mut self, a: Pos2, b: Pos2) -> Vec<NlaEdit> {
            let button = |pos, pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            };
            let mut out = Vec::new();
            out.extend(self.frame(vec![egui::Event::PointerMoved(a)]));
            out.extend(self.frame(vec![button(a, true)]));
            for i in 1..=4 {
                out.extend(self.frame(vec![egui::Event::PointerMoved(
                    a + (b - a) * (i as f32 / 4.0),
                )]));
            }
            out.extend(self.frame(vec![button(b, false)]));
            out
        }

        fn key(&mut self, key: Key) -> Vec<NlaEdit> {
            self.frame(vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Default::default(),
            }])
        }
    }

    #[test]
    fn rows_put_the_top_strip_first() {
        let v = view();
        let r = rows(&v);
        assert!(matches!(r[0], Row::Object(_)));
        assert!(matches!(r[1], Row::Strip(_, s) if s.label == "Sway"));
        assert!(matches!(r[2], Row::Strip(_, s) if s.label == "Walk"));
    }

    #[test]
    fn dragging_a_strip_moves_it_by_whole_frames() {
        let mut rig = Rig::new();
        let l = rig.layout();
        // Walk is the bottom strip: row 2.
        let a = Pos2::new(l.x(20.0), l.row_y(2));
        let b = a + Vec2::new(l.frame_width() * 7.4, 0.0);
        let edits = rig.drag(a, b);
        assert_eq!(
            edits.first(),
            Some(&NlaEdit::Select {
                object: "Cube".into(),
                index: 0
            })
        );
        assert_eq!(
            edits.last(),
            Some(&NlaEdit::Move {
                object: "Cube".into(),
                index: 0,
                start: 8.0
            }),
            "{edits:?}"
        );
    }

    #[test]
    fn a_click_on_a_strip_picks_it_and_moves_nothing() {
        let mut rig = Rig::new();
        let l = rig.layout();
        let p = Pos2::new(l.x(20.0), l.row_y(1));
        let edits = rig.drag(p, p);
        assert_eq!(
            edits,
            vec![NlaEdit::Select {
                object: "Cube".into(),
                index: 1
            }]
        );
    }

    #[test]
    fn x_removes_the_picked_strip_and_m_mutes_it() {
        let mut rig = Rig::new();
        let l = rig.layout();
        let p = Pos2::new(l.x(20.0), l.row_y(1));
        rig.drag(p, p);
        assert_eq!(
            rig.key(Key::M),
            vec![NlaEdit::Mute {
                object: "Cube".into(),
                index: 1,
                mute: true
            }]
        );
        assert_eq!(
            rig.key(Key::X),
            vec![NlaEdit::Remove {
                object: "Cube".into(),
                index: 1
            }]
        );
        assert!(rig.key(Key::X).is_empty(), "nothing is picked after");
    }

    #[test]
    fn a_click_in_empty_time_picks_nothing() {
        let mut rig = Rig::new();
        let l = rig.layout();
        // Row 1 is Sway, 10 to 30: frame 50 is past it.
        let p = Pos2::new(l.x(50.0), l.row_y(1));
        assert!(rig.drag(p, p).is_empty());
        assert!(rig.key(Key::X).is_empty());
    }

    #[test]
    fn the_button_on_an_object_row_pushes_its_action_down() {
        let mut rig = Rig::new();
        let l = rig.layout();
        let p = l.push_button(0).center();
        assert_eq!(
            rig.drag(p, p),
            vec![NlaEdit::PushDown {
                object: "Cube".into()
            }]
        );
    }
}
