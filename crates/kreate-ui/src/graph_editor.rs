//! The graph editor: each curve as a line of value over time.
//!
//! Blender's Graph Editor, its core. Where the dope sheet says *when* keys
//! are, this says *what* they are and how the curve travels between them:
//! every curve of the selected objects drawn on one plane, X red, Y green and
//! Z blue as everywhere else, each key a point to select and drag in time and
//! in value, and each selected key's handle a line to drag that bends the
//! curve through it.
//!
//! Like every panel it changes nothing: it draws a [`GraphView`] and returns
//! [`GraphEdit`]s for the application to turn into the `anim.*` operators.
//! The curves arrive already sampled, because evaluating one is the scene's
//! business and this crate does not see the scene.

use egui::{Color32, Key, Pos2, Rect, Sense, Stroke, Ui};

/// One key on a curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphKey {
    pub frame: f64,
    pub value: f64,
    pub selected: bool,
    /// The slope the curve leaves it with, hand-set or auto — what its handle
    /// shows.
    pub slope: f64,
    /// Frames to the key before and after, for how long its handles are drawn:
    /// a third of the way to each neighbour, which is where the curve's own
    /// Bézier handles sit.
    pub before: f64,
    pub after: f64,
}

/// One curve.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphCurve {
    pub label: String,
    /// `anim::curve_target`'s text.
    pub target: String,
    pub color: Color32,
    pub keys: Vec<GraphKey>,
    /// `[frame, value]` along the curve, close enough together to draw.
    pub samples: Vec<[f64; 2]>,
}

/// What one frame of the panel shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphView {
    pub curves: Vec<GraphCurve>,
    pub frame: f64,
    pub start: i32,
    pub end: i32,
}

/// What was asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphEdit {
    Select {
        targets: Vec<String>,
        frames: (f64, f64),
        values: (f64, f64),
        extend: bool,
    },
    SelectAll(u32),
    Move {
        frames: f64,
        value: f64,
    },
    Delete,
    Interpolation(u32),
    HandleType(u32),
    /// `anim.keys_extrapolation`'s order: constant, linear, cyclic.
    Extrapolation(u32),
    Smooth,
    Bake,
    SetSlope {
        target: String,
        frame: f64,
        slope: f64,
    },
    SetFrame(f64),
}

/// The panel's strings.
pub struct GraphText {
    pub empty: String,
    pub interpolation: String,
    pub constant: String,
    pub linear: String,
    pub bezier: String,
    pub handles: String,
    pub auto: String,
    pub flat: String,
    pub extrapolation: String,
    pub cyclic: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
enum Drag {
    #[default]
    None,
    /// Moving the selection from where the pointer started, in graph units.
    Move {
        from: [f64; 2],
    },
    Grab {
        from: [f64; 2],
    },
    /// A handle of one key, on one curve.
    Handle {
        curve: usize,
        key: usize,
    },
    Box {
        from: Pos2,
    },
    Scrub,
    Pan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Menu {
    Interpolation(Pos2),
    Handles(Pos2),
    Extrapolation(Pos2),
}

/// What the panel remembers, per area.
#[derive(Clone, Debug, Default)]
pub struct GraphState {
    pub rect: Option<Rect>,
    /// Curves hidden from the plot, by target: the channel list's eyes.
    /// Hidden is a view setting, as in Blender — the curve still plays.
    hidden: std::collections::BTreeSet<String>,
    /// The frames and values on screen: `(frame0, frame1, value0, value1)`.
    view: Option<(f64, f64, f64, f64)>,
    drag: Drag,
    menu: Option<Menu>,
}

const RULER: f32 = 20.0;
const AXIS: f32 = 44.0;
const KEY_HIT: f32 = 7.0;

/// Where things are, for one frame of the panel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub plot: Rect,
    pub view: (f64, f64, f64, f64),
}

impl Layout {
    pub fn new(rect: Rect, view: (f64, f64, f64, f64)) -> Self {
        Self {
            plot: Rect::from_min_max(Pos2::new(rect.min.x + AXIS, rect.min.y + RULER), rect.max),
            view,
        }
    }

    pub fn to_screen(&self, frame: f64, value: f64) -> Pos2 {
        let (f0, f1, v0, v1) = self.view;
        let x = self.plot.min.x + ((frame - f0) / (f1 - f0)) as f32 * self.plot.width();
        let y = self.plot.max.y - ((value - v0) / (v1 - v0)) as f32 * self.plot.height();
        Pos2::new(x, y)
    }

    pub fn to_graph(&self, p: Pos2) -> [f64; 2] {
        let (f0, f1, v0, v1) = self.view;
        let tx = ((p.x - self.plot.min.x) / self.plot.width().max(1.0)) as f64;
        let ty = ((self.plot.max.y - p.y) / self.plot.height().max(1.0)) as f64;
        [f0 + tx * (f1 - f0), v0 + ty * (v1 - v0)]
    }
}

/// The view that shows every key and every curve, with a margin — Home.
pub fn frame_all(view: &GraphView) -> (f64, f64, f64, f64) {
    let points = view.curves.iter().flat_map(|c| {
        c.keys
            .iter()
            .map(|k| [k.frame, k.value])
            .chain(c.samples.iter().copied())
    });
    let mut b: Option<(f64, f64, f64, f64)> = None;
    for [f, v] in points {
        b = Some(match b {
            None => (f, f, v, v),
            Some((f0, f1, v0, v1)) => (f0.min(f), f1.max(f), v0.min(v), v1.max(v)),
        });
    }
    let (f0, f1, v0, v1) = b.unwrap_or((view.start as f64, view.end as f64, -1.0, 1.0));
    let (f0, f1) = if f1 - f0 < 2.0 {
        (f0 - 5.0, f1 + 5.0)
    } else {
        (f0, f1)
    };
    let (v0, v1) = if v1 - v0 < 1e-3 {
        (v0 - 1.0, v1 + 1.0)
    } else {
        (v0, v1)
    };
    let (df, dv) = ((f1 - f0) * 0.08, (v1 - v0) * 0.12);
    (f0 - df, f1 + df, v0 - dv, v1 + dv)
}

/// A step for grid lines: 1, 2 or 5 times a power of ten, the smallest that
/// leaves `min_gap` points between lines.
pub fn nice_step(units_per_point: f64, min_gap: f64) -> f64 {
    let raw = units_per_point * min_gap;
    if !(raw.is_finite() && raw > 0.0) {
        return 1.0;
    }
    let decade = 10f64.powf(raw.log10().floor());
    for m in [1.0, 2.0, 5.0, 10.0] {
        if m * decade >= raw {
            return m * decade;
        }
    }
    10.0 * decade
}

/// The ends of a key's two handles, in graph units.
pub fn handles(key: &GraphKey) -> ([f64; 2], [f64; 2]) {
    let back = key.before / 3.0;
    let ahead = key.after / 3.0;
    (
        [key.frame - back, key.value - key.slope * back],
        [key.frame + ahead, key.value + key.slope * ahead],
    )
}

/// The key under a point: `(curve, key)`, nearest within reach.
pub fn key_at(view: &GraphView, layout: &Layout, p: Pos2) -> Option<(usize, usize)> {
    let mut best: Option<(f32, (usize, usize))> = None;
    for (c, curve) in view.curves.iter().enumerate() {
        for (k, key) in curve.keys.iter().enumerate() {
            let d = layout.to_screen(key.frame, key.value).distance(p);
            if d <= KEY_HIT && best.is_none_or(|(b, _)| d < b) {
                best = Some((d, (c, k)));
            }
        }
    }
    best.map(|(_, hit)| hit)
}

/// The handle end under a point, of a selected key.
pub fn handle_at(view: &GraphView, layout: &Layout, p: Pos2) -> Option<(usize, usize)> {
    for (c, curve) in view.curves.iter().enumerate() {
        for (k, key) in curve.keys.iter().enumerate().filter(|(_, k)| k.selected) {
            let (a, b) = handles(key);
            for end in [a, b] {
                if layout.to_screen(end[0], end[1]).distance(p) <= KEY_HIT {
                    return Some((c, k));
                }
            }
        }
    }
    None
}

/// Draws the graph editor. Returns what was asked for.
/// How wide the channel list is.
const CHANNELS: f32 = 170.0;

/// The graph editor: the channel list on the left, the curves on the right.
///
/// Blender's channel region, its core: a row per curve with its colour, its
/// name and an eye. A hidden curve is left out of the plot, so its keys can be
/// neither seen nor picked — the way to work on one curve among twelve.
/// Clicking a row selects that curve's keys, Shift adding to what is selected.
pub fn graph_editor(
    ui: &mut Ui,
    view: &GraphView,
    state: &mut GraphState,
    text: &GraphText,
) -> Vec<GraphEdit> {
    let full = ui.available_rect_before_wrap();
    // A narrow editor keeps the whole width for the curves.
    let list_width = if full.width() > CHANNELS * 2.5 {
        CHANNELS
    } else {
        0.0
    };
    let mut out = Vec::new();
    if list_width > 0.0 && !view.curves.is_empty() {
        let list = Rect::from_min_max(full.min, Pos2::new(full.min.x + list_width, full.max.y));
        let theme = crate::theme::from_ui(ui);
        ui.painter().rect_filled(list, 0.0, theme.base);
        ui.scope_builder(
            egui::UiBuilder::new().max_rect(list.shrink2(egui::vec2(4.0, 2.0))),
            |ui| {
                ui.add_space(RULER - 2.0);
                egui::ScrollArea::vertical()
                    .id_salt("graph-channels")
                    .show(ui, |ui| {
                        for curve in &view.curves {
                            let hidden = state.hidden.contains(&curve.target);
                            ui.horizontal(|ui| {
                                // The eye: a filled dot shown, a ring hidden.
                                let (eye, eye_response) =
                                    ui.allocate_exact_size(egui::vec2(14.0, 14.0), Sense::click());
                                let ink = if hidden {
                                    theme.text_faint
                                } else {
                                    theme.text_dim
                                };
                                if hidden {
                                    ui.painter().circle_stroke(
                                        eye.center(),
                                        4.0,
                                        Stroke::new(1.2, ink),
                                    );
                                } else {
                                    ui.painter().circle_filled(eye.center(), 4.0, ink);
                                }
                                if eye_response.clicked() {
                                    if hidden {
                                        state.hidden.remove(&curve.target);
                                    } else {
                                        state.hidden.insert(curve.target.clone());
                                    }
                                }
                                let (swatch, _) =
                                    ui.allocate_exact_size(egui::vec2(4.0, 14.0), Sense::hover());
                                ui.painter().rect_filled(swatch, 1.0, curve.color);
                                let label = egui::RichText::new(&curve.label).size(11.0);
                                let label = if hidden {
                                    label.color(theme.text_faint)
                                } else {
                                    label
                                };
                                let row = ui
                                    .add(egui::Label::new(label).truncate().sense(Sense::click()));
                                if row.clicked() {
                                    let extend = ui.input(|i| i.modifiers.shift);
                                    out.push(GraphEdit::Select {
                                        targets: vec![curve.target.clone()],
                                        frames: (f64::MIN, f64::MAX),
                                        values: (f64::MIN, f64::MAX),
                                        extend,
                                    });
                                }
                            });
                        }
                    });
            },
        );
    }
    let plot = Rect::from_min_max(Pos2::new(full.min.x + list_width, full.min.y), full.max);
    let shown = GraphView {
        curves: view
            .curves
            .iter()
            .filter(|c| !state.hidden.contains(&c.target))
            .cloned()
            .collect(),
        ..view.clone()
    };
    let edits = ui
        .scope_builder(egui::UiBuilder::new().max_rect(plot), |ui| {
            graph_plot(ui, &shown, state, text)
        })
        .inner;
    out.extend(edits);
    out
}

fn graph_plot(
    ui: &mut Ui,
    view: &GraphView,
    state: &mut GraphState,
    text: &GraphText,
) -> Vec<GraphEdit> {
    let mut out = Vec::new();
    let theme = crate::theme::from_ui(ui);
    let rect = ui.available_rect_before_wrap();
    if rect.width() < AXIS + 60.0 || rect.height() < RULER + 40.0 {
        return out;
    }
    state.rect = Some(rect);
    let response = ui.allocate_rect(rect, Sense::click_and_drag());
    let hovered = ui.rect_contains_pointer(rect);
    let pointer = ui.input(|i| i.pointer.interact_pos().or(i.pointer.hover_pos()));

    let mut viewed = *state.view.get_or_insert_with(|| frame_all(view));

    // --- navigation ---------------------------------------------------------------
    if hovered {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            if let Some(p) = pointer {
                let at = Layout::new(rect, viewed).to_graph(p);
                let k = (1.0 - scroll as f64 * 0.0015).clamp(0.5, 2.0);
                let (f0, f1, v0, v1) = viewed;
                viewed = (
                    at[0] + (f0 - at[0]) * k,
                    at[0] + (f1 - at[0]) * k,
                    at[1] + (v0 - at[1]) * k,
                    at[1] + (v1 - at[1]) * k,
                );
            }
        }
        if ui.input(|i| i.key_pressed(Key::Home)) {
            viewed = frame_all(view);
        }
    }
    if response.dragged_by(egui::PointerButton::Middle) {
        let l = Layout::new(rect, viewed);
        let d = response.drag_delta();
        let (f0, f1, v0, v1) = viewed;
        let df = -(d.x / l.plot.width()) as f64 * (f1 - f0);
        let dv = (d.y / l.plot.height()) as f64 * (v1 - v0);
        viewed = (f0 + df, f1 + df, v0 + dv, v1 + dv);
        state.drag = Drag::Pan;
    }
    state.view = Some(viewed);
    let layout = Layout::new(rect, viewed);
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, theme.sunken);

    // --- the grid -------------------------------------------------------------------
    let (f0, f1, v0, v1) = viewed;
    let fstep = nice_step((f1 - f0) / layout.plot.width() as f64, 50.0).max(1.0);
    let vstep = nice_step((v1 - v0) / layout.plot.height() as f64, 30.0);
    painter.rect_filled(
        Rect::from_min_max(rect.min, Pos2::new(rect.max.x, layout.plot.min.y)),
        0.0,
        theme.base,
    );
    painter.rect_filled(
        Rect::from_min_max(
            Pos2::new(rect.min.x, layout.plot.min.y),
            Pos2::new(layout.plot.min.x, rect.max.y),
        ),
        0.0,
        theme.base,
    );
    let mut f = (f0 / fstep).ceil() * fstep;
    while f <= f1 {
        let x = layout.to_screen(f, 0.0).x;
        painter.line_segment(
            [Pos2::new(x, layout.plot.min.y), Pos2::new(x, rect.max.y)],
            Stroke::new(1.0, theme.border),
        );
        painter.text(
            Pos2::new(x, rect.min.y + 3.0),
            egui::Align2::CENTER_TOP,
            format!("{}", f as i64),
            egui::FontId::proportional(10.0),
            theme.text_faint,
        );
        f += fstep;
    }
    let mut v = (v0 / vstep).ceil() * vstep;
    while v <= v1 {
        let y = layout.to_screen(0.0, v).y;
        let strong = v.abs() < vstep * 1e-6;
        painter.line_segment(
            [Pos2::new(layout.plot.min.x, y), Pos2::new(rect.max.x, y)],
            Stroke::new(
                if strong { 1.5 } else { 1.0 },
                if strong {
                    theme.border_strong
                } else {
                    theme.border
                },
            ),
        );
        // The line at zero is reached by adding steps and lands a hair off it,
        // which prints as "-0".
        let shown = if strong { 0.0 } else { v };
        let label = if vstep >= 1.0 {
            format!("{shown:.0}")
        } else if vstep >= 0.1 {
            format!("{shown:.1}")
        } else {
            format!("{shown:.2}")
        };
        painter.text(
            Pos2::new(layout.plot.min.x - 4.0, y),
            egui::Align2::RIGHT_CENTER,
            label,
            egui::FontId::proportional(10.0),
            theme.text_faint,
        );
        v += vstep;
    }
    let dim = Color32::from_black_alpha(45);
    let sx = layout.to_screen(view.start as f64, 0.0).x;
    let ex = layout.to_screen(view.end as f64, 0.0).x;
    painter.rect_filled(
        Rect::from_min_max(
            layout.plot.min,
            Pos2::new(sx.max(layout.plot.min.x), rect.max.y),
        ),
        0.0,
        dim,
    );
    painter.rect_filled(
        Rect::from_min_max(Pos2::new(ex.min(rect.max.x), layout.plot.min.y), rect.max),
        0.0,
        dim,
    );

    if view.curves.is_empty() {
        painter.text(
            layout.plot.center(),
            egui::Align2::CENTER_CENTER,
            &text.empty,
            egui::FontId::proportional(12.0),
            theme.text_faint,
        );
    }

    // How far a drag in progress has taken the selection: whole frames, any value.
    let offset = match (&state.drag, pointer) {
        (Drag::Move { from } | Drag::Grab { from }, Some(p)) => {
            let at = layout.to_graph(p);
            [(at[0] - from[0]).round(), at[1] - from[1]]
        }
        _ => [0.0, 0.0],
    };

    // --- curves, keys and handles -----------------------------------------------------
    let plot_painter = ui.painter().with_clip_rect(layout.plot);
    for (c, curve) in view.curves.iter().enumerate() {
        let line: Vec<Pos2> = curve
            .samples
            .iter()
            .map(|[f, v]| layout.to_screen(*f, *v))
            .collect();
        if line.len() > 1 {
            plot_painter.add(egui::Shape::line(line, Stroke::new(1.5, curve.color)));
        }
        for (k, key) in curve.keys.iter().enumerate() {
            let (frame, value) = if key.selected {
                (key.frame + offset[0], key.value + offset[1])
            } else {
                (key.frame, key.value)
            };
            let at = layout.to_screen(frame, value);
            if key.selected {
                let mut shown = *key;
                shown.frame = frame;
                shown.value = value;
                if let (Drag::Handle { curve: hc, key: hk }, Some(p)) = (&state.drag, pointer) {
                    if (*hc, *hk) == (c, k) {
                        shown.slope = dragged_slope(key, layout.to_graph(p));
                    }
                }
                let (a, b) = handles(&shown);
                let (a, b) = (layout.to_screen(a[0], a[1]), layout.to_screen(b[0], b[1]));
                plot_painter.line_segment([a, b], Stroke::new(1.0, theme.text_dim));
                for end in [a, b] {
                    plot_painter.circle_filled(end, 3.0, theme.text_dim);
                }
            }
            plot_painter.circle(
                at,
                if key.selected { 4.5 } else { 3.5 },
                if key.selected {
                    theme.accent
                } else {
                    Color32::from_gray(220)
                },
                Stroke::new(1.0, Color32::BLACK),
            );
        }
    }

    // The clock.
    let x = layout.to_screen(view.frame, 0.0).x;
    painter.line_segment(
        [Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)],
        Stroke::new(2.0, theme.accent),
    );

    if let (Drag::Box { from }, Some(p)) = (&state.drag, pointer) {
        let b = Rect::from_two_pos(*from, p);
        painter.rect_filled(b, 0.0, theme.accent_wash);
        painter.rect_stroke(
            b,
            0.0,
            Stroke::new(1.0, theme.accent),
            egui::StrokeKind::Inside,
        );
    }

    // --- the pointer -------------------------------------------------------------------
    let shift = ui.input(|i| i.modifiers.shift);
    let press = ui.input(|i| i.pointer.primary_pressed()) && response.hovered();
    let release = ui.input(|i| i.pointer.primary_released());

    if let Drag::Grab { .. } = state.drag {
        let cancel = ui.input(|i| i.key_pressed(Key::Escape) || i.pointer.secondary_pressed());
        let confirm = ui.input(|i| i.pointer.primary_pressed() || i.key_pressed(Key::Enter));
        if cancel {
            state.drag = Drag::None;
        } else if confirm {
            if offset != [0.0, 0.0] {
                out.push(GraphEdit::Move {
                    frames: offset[0],
                    value: offset[1],
                });
            }
            state.drag = Drag::None;
        }
    } else if press && state.menu.is_none() {
        if let Some(p) = pointer {
            if p.y < layout.plot.min.y && p.x >= layout.plot.min.x {
                state.drag = Drag::Scrub;
            } else if let Some((c, k)) =
                handle_at(view, &layout, p).filter(|_| key_at(view, &layout, p).is_none())
            {
                state.drag = Drag::Handle { curve: c, key: k };
            } else if let Some((c, k)) = key_at(view, &layout, p) {
                let key = &view.curves[c].keys[k];
                if !key.selected || shift {
                    out.push(GraphEdit::Select {
                        targets: vec![view.curves[c].target.clone()],
                        frames: (key.frame - 1e-3, key.frame + 1e-3),
                        values: (key.value - 1e-9, key.value + 1e-9),
                        extend: shift,
                    });
                }
                state.drag = Drag::Move {
                    from: layout.to_graph(p),
                };
            } else if layout.plot.contains(p) {
                state.drag = Drag::Box { from: p };
            }
        }
    }

    if let (Drag::Scrub, Some(p)) = (&state.drag, pointer) {
        let frame = layout.to_graph(p)[0].round();
        if (frame - view.frame).abs() > 1e-9 {
            out.push(GraphEdit::SetFrame(frame));
        }
    }

    if release {
        match std::mem::take(&mut state.drag) {
            Drag::Move { .. } if offset != [0.0, 0.0] => out.push(GraphEdit::Move {
                frames: offset[0],
                value: offset[1],
            }),
            Drag::Handle { curve, key } => {
                if let (Some(p), Some(k)) = (
                    pointer,
                    view.curves.get(curve).and_then(|c| c.keys.get(key)),
                ) {
                    let slope = dragged_slope(k, layout.to_graph(p));
                    if (slope - k.slope).abs() > 1e-12 {
                        out.push(GraphEdit::SetSlope {
                            target: view.curves[curve].target.clone(),
                            frame: k.frame,
                            slope,
                        });
                    }
                }
            }
            Drag::Box { from } => {
                let to = pointer.unwrap_or(from);
                if from.distance(to) < 4.0 {
                    if !shift {
                        out.push(GraphEdit::SelectAll(1));
                    }
                } else {
                    let (a, b) = (layout.to_graph(from), layout.to_graph(to));
                    out.push(GraphEdit::Select {
                        targets: view.curves.iter().map(|c| c.target.clone()).collect(),
                        frames: (a[0].min(b[0]), a[0].max(b[0])),
                        values: (a[1].min(b[1]), a[1].max(b[1])),
                        extend: shift,
                    });
                }
            }
            Drag::Grab { from } => state.drag = Drag::Grab { from },
            _ => {}
        }
    }

    // --- keys ------------------------------------------------------------------------
    if hovered && !ui.ctx().egui_wants_keyboard_input() && state.drag == Drag::None {
        let (alt, ctrl, shift) =
            ui.input(|i| (i.modifiers.alt, i.modifiers.command, i.modifiers.shift));
        let pressed = |k: Key| ui.input(|i| i.key_pressed(k));
        if pressed(Key::A) {
            out.push(GraphEdit::SelectAll(if alt { 1 } else { 0 }));
        }
        if ctrl && pressed(Key::I) {
            out.push(GraphEdit::SelectAll(2));
        }
        if pressed(Key::X) || pressed(Key::Delete) {
            out.push(GraphEdit::Delete);
        }
        if pressed(Key::G) {
            if let Some(p) = pointer {
                state.drag = Drag::Grab {
                    from: layout.to_graph(p),
                };
            }
        }
        if pressed(Key::T) {
            state.menu = pointer.map(Menu::Interpolation);
        }
        if pressed(Key::V) {
            state.menu = pointer.map(Menu::Handles);
        }
        if shift && pressed(Key::E) {
            state.menu = pointer.map(Menu::Extrapolation);
        }
        if alt && pressed(Key::O) {
            out.push(if shift { GraphEdit::Bake } else { GraphEdit::Smooth });
        }
    }

    // --- menus -------------------------------------------------------------------------
    if let Some(menu) = state.menu {
        let (at, title, entries): (Pos2, &str, Vec<&str>) = match menu {
            Menu::Interpolation(at) => (
                at,
                &text.interpolation,
                vec![&text.constant, &text.linear, &text.bezier],
            ),
            Menu::Handles(at) => (at, &text.handles, vec![&text.auto, &text.flat]),
            Menu::Extrapolation(at) => (
                at,
                &text.extrapolation,
                vec![&text.constant, &text.linear, &text.cyclic],
            ),
        };
        let mut close = false;
        let area = egui::Area::new(ui.id().with("graph-menu"))
            .order(egui::Order::Foreground)
            .fixed_pos(at)
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(egui::RichText::new(title).strong());
                    ui.separator();
                    for (i, label) in entries.iter().enumerate() {
                        if ui.button(*label).clicked() {
                            out.push(match menu {
                                Menu::Interpolation(_) => GraphEdit::Interpolation(i as u32),
                                Menu::Handles(_) => GraphEdit::HandleType(i as u32),
                                Menu::Extrapolation(_) => GraphEdit::Extrapolation(i as u32),
                            });
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

/// The slope a handle dragged to `to` gives its key: the line through the key
/// and the pointer. A pointer straight above or below the key would be a
/// vertical line, which is no slope, so the run is kept a sliver away from zero
/// on whichever side the pointer is.
fn dragged_slope(key: &GraphKey, to: [f64; 2]) -> f64 {
    let run = to[0] - key.frame;
    let run = if run.abs() < 1e-3 {
        1e-3f64.copysign(run)
    } else {
        run
    };
    (to[1] - key.value) / run
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(frame: f64, value: f64) -> GraphKey {
        GraphKey {
            frame,
            value,
            selected: false,
            slope: 0.0,
            before: 9.0,
            after: 9.0,
        }
    }

    fn view() -> GraphView {
        GraphView {
            curves: vec![GraphCurve {
                label: "Location X".into(),
                target: "7/location/0".into(),
                color: Color32::RED,
                keys: vec![key(1.0, 0.0), key(10.0, 4.0), key(20.0, 1.0)],
                samples: vec![[1.0, 0.0], [10.0, 4.0], [20.0, 1.0]],
            }],
            frame: 1.0,
            start: 1,
            end: 20,
        }
    }

    #[test]
    fn a_point_goes_to_the_screen_and_back() {
        let l = Layout::new(
            Rect::from_min_size(Pos2::ZERO, egui::vec2(600.0, 300.0)),
            (0.0, 30.0, -2.0, 6.0),
        );
        for p in [[0.0, -2.0], [12.5, 3.3], [30.0, 6.0]] {
            let back = l.to_graph(l.to_screen(p[0], p[1]));
            assert!(
                (back[0] - p[0]).abs() < 1e-3 && (back[1] - p[1]).abs() < 1e-3,
                "{p:?} came back as {back:?}"
            );
        }
        // Up is up: a larger value is higher on screen.
        assert!(l.to_screen(0.0, 5.0).y < l.to_screen(0.0, 1.0).y);
    }

    #[test]
    fn home_shows_every_key() {
        let v = view();
        let (f0, f1, v0, v1) = frame_all(&v);
        for k in &v.curves[0].keys {
            assert!(
                k.frame > f0 && k.frame < f1 && k.value > v0 && k.value < v1,
                "{k:?} is off screen"
            );
        }
    }

    #[test]
    fn grid_steps_are_round_and_never_crowd() {
        for upp in [0.001, 0.03, 0.4, 7.0, 120.0] {
            let step = nice_step(upp, 30.0);
            assert!(step / upp >= 30.0 - 1e-9, "{upp}: {step}");
            let mantissa = step / 10f64.powf(step.log10().floor());
            assert!(
                [1.0, 2.0, 5.0, 10.0]
                    .iter()
                    .any(|m| (mantissa - m).abs() < 1e-9),
                "{step}"
            );
        }
    }

    #[test]
    fn a_handle_sits_a_third_of_the_way_to_each_neighbour_along_the_slope() {
        let k = GraphKey {
            slope: 0.5,
            before: 3.0,
            after: 9.0,
            ..key(10.0, 4.0)
        };
        let (a, b) = handles(&k);
        assert_eq!(a, [9.0, 3.5]);
        assert_eq!(b, [13.0, 5.5]);
        // And dragging one to a point gives the slope through that point.
        assert!((dragged_slope(&k, [12.0, 8.0]) - 2.0).abs() < 1e-12);
        assert!(
            dragged_slope(&k, [10.0, 8.0]).is_finite(),
            "straight above is a finite slope"
        );
    }

    struct Rig {
        ctx: egui::Context,
        state: GraphState,
        view: GraphView,
        time: f64,
    }

    impl Rig {
        fn new(view: GraphView) -> Self {
            let mut rig = Self {
                ctx: egui::Context::default(),
                state: GraphState::default(),
                view,
                time: 0.0,
            };
            rig.frame(vec![]);
            rig.frame(vec![]);
            rig
        }
        fn frame(&mut self, events: Vec<egui::Event>) -> Vec<GraphEdit> {
            self.time += 1.0 / 60.0;
            let raw = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 400.0))),
                time: Some(self.time),
                events,
                ..Default::default()
            };
            let text = GraphText {
                empty: String::new(),
                interpolation: String::new(),
                constant: "C".into(),
                linear: "L".into(),
                bezier: "B".into(),
                handles: String::new(),
                auto: "A".into(),
                flat: "F".into(),
                extrapolation: String::new(),
                cyclic: "Y".into(),
            };
            let mut out = Vec::new();
            let (state, view) = (&mut self.state, &self.view);
            let _ = self
                .ctx
                .run_ui(raw, |ui| out = graph_editor(ui, view, state, &text));
            out
        }
        fn layout(&self) -> Layout {
            Layout::new(self.state.rect.unwrap(), self.state.view.unwrap())
        }
        fn drag(&mut self, a: Pos2, b: Pos2) -> Vec<GraphEdit> {
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
    }

    #[test]
    fn dragging_a_key_moves_it_in_whole_frames_and_any_value() {
        let mut rig = Rig::new(view());
        let l = rig.layout();
        let a = l.to_screen(10.0, 4.0);
        let b = l.to_screen(13.4, 5.0);
        let edits = rig.drag(a, b);
        assert!(
            matches!(edits.first(), Some(GraphEdit::Select { .. })),
            "{edits:?}"
        );
        match edits.last() {
            Some(GraphEdit::Move { frames, value }) => {
                assert_eq!(*frames, 3.0);
                assert!((value - 1.0).abs() < 0.05, "{value}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn dragging_a_selected_keys_handle_sets_its_slope() {
        let mut v = view();
        v.curves[0].keys[1].selected = true;
        let mut rig = Rig::new(v);
        let l = rig.layout();
        let (_, out_handle) = handles(&rig.view.curves[0].keys[1]);
        let a = l.to_screen(out_handle[0], out_handle[1]);
        let b = l.to_screen(13.0, 7.0);
        let edits = rig.drag(a, b);
        match edits.as_slice() {
            [GraphEdit::SetSlope {
                target,
                frame,
                slope,
            }] => {
                assert_eq!((target.as_str(), *frame), ("7/location/0", 10.0));
                assert!((slope - 1.0).abs() < 0.05, "{slope}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_box_selects_by_frame_and_value() {
        let mut rig = Rig::new(view());
        let l = rig.layout();
        // Inside the plot: Home frames the keys, 0 to 4, with a margin.
        let edits = rig.drag(l.to_screen(5.0, 4.3), l.to_screen(18.0, 0.5));
        match edits.as_slice() {
            [GraphEdit::Select { frames, values, .. }] => {
                assert!(
                    (frames.0 - 5.0).abs() < 0.2 && (frames.1 - 18.0).abs() < 0.2,
                    "{frames:?}"
                );
                assert!(
                    (values.0 - 0.5).abs() < 0.05 && (values.1 - 4.3).abs() < 0.05,
                    "{values:?}"
                );
            }
            other => panic!("{other:?}"),
        }
    }
}
