//! The timeline: where the scene's clock is, and a ruler to move it along.
//!
//! Blender's Timeline, the part of it every animator uses: play and stop, jump
//! to either end, the current frame and the range, and a ruler to drag the
//! clock along — scrubbing — with the selected objects' keys marked on it so
//! the next key is somewhere to aim for.
//!
//! Like every panel it changes nothing: it draws a [`TimelineView`] and
//! returns [`TimelineEdit`]s, and the application moves the clock.
//!
//! # Scrubbing lands on whole frames
//!
//! Playback runs between frames, and the clock can hold a fraction. A scrub
//! snaps to the nearest whole frame, which is what a key is put on and what an
//! animator means by "go to frame 12" — a key inserted at 12.37 because the
//! pointer was not quite still is a key nobody can hit again.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};

/// What one frame of the panel shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimelineView {
    pub frame: f64,
    pub start: i32,
    pub end: i32,
    pub playing: bool,
    /// Whether transforms key themselves: the record button.
    pub autokey: bool,
    /// Frames the selected objects have keys on.
    pub keys: Vec<f64>,
    /// The scene's markers, in frame order.
    pub markers: Vec<TimelineMarker>,
    /// Whether the active object is a camera, which a marker can be bound to.
    pub active_camera: bool,
}

/// A marker as the ruler shows it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimelineMarker {
    pub frame: i32,
    pub name: String,
    /// Whether a camera is bound to it: the scene cuts there.
    pub camera: bool,
}

/// What the pointer asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum TimelineEdit {
    SetFrame(f64),
    TogglePlay,
    JumpStart,
    JumpEnd,
    SetRange {
        start: i32,
        end: i32,
    },
    ToggleAutokey,
    /// A marker on the frame the clock is on.
    AddMarker,
    /// The active camera bound to the marker on the clock's frame.
    BindCamera,
    /// The marker on the clock's frame taken away.
    RemoveMarker,
}

/// The panel's strings.
pub struct TimelineText {
    pub play: String,
    pub pause: String,
    pub jump_start: String,
    pub jump_end: String,
    pub start: String,
    pub end: String,
    pub autokey: String,
    /// The ruler's right-click menu.
    pub add_marker: String,
    pub bind_camera: String,
    pub remove_marker: String,
}

/// What the panel remembers.
#[derive(Clone, Debug, Default)]
pub struct TimelineState {
    /// Where it was drawn, for the application to tell whether the pointer is
    /// over it.
    pub rect: Option<Rect>,
    scrubbing: bool,
}

/// The frames the ruler spans: the range, with a margin either side so the
/// first and last frames are not on the panel's edge.
pub fn visible(start: i32, end: i32) -> (f64, f64) {
    let (a, b) = (start as f64, end.max(start + 1) as f64);
    let margin = ((b - a) * 0.04).max(2.0);
    (a - margin, b + margin)
}

/// A frame's x on a ruler drawn in `rect` over `range`.
pub fn frame_to_x(frame: f64, range: (f64, f64), rect: Rect) -> f32 {
    let t = (frame - range.0) / (range.1 - range.0);
    rect.min.x + (t as f32) * rect.width()
}

/// The frame under an x, the inverse of [`frame_to_x`].
pub fn x_to_frame(x: f32, range: (f64, f64), rect: Rect) -> f64 {
    let t = ((x - rect.min.x) / rect.width().max(1.0)) as f64;
    range.0 + t * (range.1 - range.0)
}

/// How many frames apart the ruler's numbers are: the smallest of 1, 2, 5,
/// 10, 20, 50… that leaves `min_gap` points between two numbers.
pub fn tick_step(points_per_frame: f32, min_gap: f32) -> i32 {
    let mut ladder = [1, 2, 5].into_iter().cycle();
    let mut decade = 1i32;
    loop {
        let m = ladder.next().unwrap_or(1);
        let step = m * decade;
        if step as f32 * points_per_frame >= min_gap {
            return step;
        }
        if m == 5 {
            decade = decade.saturating_mul(10);
            if decade > 1_000_000 {
                return step;
            }
        }
    }
}

/// Draws the timeline. Returns what the pointer and the controls asked for.
pub fn timeline(
    ui: &mut Ui,
    view: &TimelineView,
    state: &mut TimelineState,
    text: &TimelineText,
) -> Vec<TimelineEdit> {
    let mut out = Vec::new();
    let theme = crate::theme::from_ui(ui);
    let full = ui.available_rect_before_wrap();
    if full.width() < 40.0 || full.height() < 30.0 {
        return out;
    }
    state.rect = Some(full);

    // --- the controls ----------------------------------------------------------
    let bar_height = ui.spacing().interact_size.y + 6.0;
    let bar = Rect::from_min_size(full.min, Vec2::new(full.width(), bar_height));
    ui.scope_builder(egui::UiBuilder::new().max_rect(bar), |ui| {
        ui.horizontal_centered(|ui| {
            if icon_button(ui, Icon::JumpStart)
                .on_hover_text(&text.jump_start)
                .clicked()
            {
                out.push(TimelineEdit::JumpStart);
            }
            let (icon, tip) = if view.playing {
                (Icon::Pause, &text.pause)
            } else {
                (Icon::Play, &text.play)
            };
            if icon_button(ui, icon).on_hover_text(tip).clicked() {
                out.push(TimelineEdit::TogglePlay);
            }
            if icon_button(ui, Icon::JumpEnd)
                .on_hover_text(&text.jump_end)
                .clicked()
            {
                out.push(TimelineEdit::JumpEnd);
            }
            ui.separator();
            // The record button: a red dot, lit while on — Blender's, and a
            // state rather than an action, so it shows which it is in.
            if record_button(ui, view.autokey)
                .on_hover_text(&text.autokey)
                .clicked()
            {
                out.push(TimelineEdit::ToggleAutokey);
            }
            ui.separator();
            let mut frame = view.frame.round() as i32;
            if ui
                .add(egui::DragValue::new(&mut frame).speed(0.25))
                .changed()
            {
                out.push(TimelineEdit::SetFrame(frame as f64));
            }
            ui.separator();
            let (mut start, mut end) = (view.start, view.end);
            ui.label(&text.start);
            let a = ui.add(egui::DragValue::new(&mut start).speed(0.25));
            ui.label(&text.end);
            let b = ui.add(
                egui::DragValue::new(&mut end)
                    .speed(0.25)
                    .range(start + 1..=i32::MAX),
            );
            if a.changed() || b.changed() {
                out.push(TimelineEdit::SetRange {
                    start,
                    end: end.max(start + 1),
                });
            }
        });
    });

    // --- the ruler -------------------------------------------------------------
    let track = Rect::from_min_max(Pos2::new(full.min.x, bar.max.y), full.max);
    let response = ui.allocate_rect(track, Sense::click_and_drag());
    let painter = ui.painter().with_clip_rect(track);
    painter.rect_filled(track, 0.0, theme.sunken);
    let range = visible(view.start, view.end);
    let points_per_frame = track.width() / (range.1 - range.0) as f32;

    // Outside the range, dimmed: frames the scene does not play.
    let dim = Color32::from_black_alpha(60);
    let a = frame_to_x(view.start as f64, range, track);
    let b = frame_to_x(view.end as f64, range, track);
    painter.rect_filled(
        Rect::from_min_max(track.min, Pos2::new(a, track.max.y)),
        0.0,
        dim,
    );
    painter.rect_filled(
        Rect::from_min_max(Pos2::new(b, track.min.y), track.max),
        0.0,
        dim,
    );

    let label_band = 16.0;
    let step = tick_step(points_per_frame, 44.0);
    let first = ((range.0 / step as f64).ceil() as i64) * step as i64;
    let mut f = first;
    while (f as f64) <= range.1 {
        let x = frame_to_x(f as f64, range, track);
        painter.line_segment(
            [
                Pos2::new(x, track.min.y + label_band),
                Pos2::new(x, track.max.y),
            ],
            Stroke::new(1.0, theme.border),
        );
        painter.text(
            Pos2::new(x, track.min.y + 2.0),
            egui::Align2::CENTER_TOP,
            f.to_string(),
            egui::FontId::proportional(10.0),
            theme.text_faint,
        );
        f += step as i64;
    }

    // Keys, as diamonds on the middle of the track.
    let mid = (track.min.y + label_band + track.max.y) / 2.0;
    for key in &view.keys {
        let x = frame_to_x(*key, range, track);
        let r = 4.5;
        let on = (key - view.frame).abs() < 1e-3;
        painter.add(egui::Shape::convex_polygon(
            vec![
                Pos2::new(x, mid - r),
                Pos2::new(x + r, mid),
                Pos2::new(x, mid + r),
                Pos2::new(x - r, mid),
            ],
            if on { theme.accent } else { theme.text_dim },
            Stroke::new(1.0, Color32::BLACK),
        ));
    }

    // Markers along the foot of the track: a flag and its name, the ones
    // that cut to a camera in the accent colour, with a faint line up.
    for marker in &view.markers {
        let x = frame_to_x(marker.frame as f64, range, track);
        if x < track.min.x - 1.0 || x > track.max.x + 1.0 {
            continue;
        }
        let ink = if marker.camera {
            theme.accent
        } else {
            theme.text_dim
        };
        let foot = track.max.y - 3.0;
        painter.line_segment(
            [Pos2::new(x, track.min.y + label_band), Pos2::new(x, foot)],
            Stroke::new(1.0, ink.gamma_multiply(0.35)),
        );
        painter.add(egui::Shape::convex_polygon(
            vec![
                Pos2::new(x, foot - 9.0),
                Pos2::new(x + 5.0, foot),
                Pos2::new(x - 5.0, foot),
            ],
            ink,
            Stroke::NONE,
        ));
        painter.text(
            Pos2::new(x + 7.0, foot),
            egui::Align2::LEFT_BOTTOM,
            &marker.name,
            egui::FontId::proportional(10.0),
            ink,
        );
    }

    // The clock: a line through the track and its frame in a box on the ruler.
    let x = frame_to_x(view.frame, range, track);
    painter.line_segment(
        [Pos2::new(x, track.min.y), Pos2::new(x, track.max.y)],
        Stroke::new(2.0, theme.accent),
    );
    let label = format!("{}", view.frame.round() as i64);
    let galley = painter.layout_no_wrap(label, egui::FontId::proportional(11.0), Color32::BLACK);
    let tag = Rect::from_center_size(
        Pos2::new(x, track.min.y + 8.0),
        galley.size() + Vec2::new(8.0, 2.0),
    );
    painter.rect_filled(tag, 3.0, theme.accent);
    painter.galley(tag.center() - galley.size() / 2.0, galley, Color32::BLACK);

    // Right-click: markers, on the frame the clock is on.
    let here = view.frame.round() as i32;
    let marked = view.markers.iter().any(|m| m.frame == here);
    response.context_menu(|ui| {
        if ui.button(&text.add_marker).clicked() {
            out.push(TimelineEdit::AddMarker);
            ui.close();
        }
        if ui
            .add_enabled(view.active_camera, egui::Button::new(&text.bind_camera))
            .clicked()
        {
            out.push(TimelineEdit::BindCamera);
            ui.close();
        }
        if ui
            .add_enabled(marked, egui::Button::new(&text.remove_marker))
            .clicked()
        {
            out.push(TimelineEdit::RemoveMarker);
            ui.close();
        }
    });

    // Scrubbing: press anywhere on the track and drag.
    if response.drag_started() || response.clicked() {
        state.scrubbing = true;
    }
    if state.scrubbing {
        if let Some(p) = response
            .interact_pointer_pos()
            .or_else(|| ui.input(|i| i.pointer.interact_pos()))
        {
            let frame = x_to_frame(p.x, range, track).round();
            if (frame - view.frame).abs() > 1e-9 {
                out.push(TimelineEdit::SetFrame(frame));
            }
        }
    }
    if response.drag_stopped() || (state.scrubbing && !ui.input(|i| i.pointer.any_down())) {
        state.scrubbing = false;
    }
    out
}

/// The auto-key toggle: a dot, red and on a lit button while recording.
fn record_button(ui: &mut Ui, on: bool) -> egui::Response {
    let size = Vec2::splat(ui.spacing().interact_size.y);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let visuals = ui.style().interact(&response);
    let theme = crate::theme::from_ui(ui);
    ui.painter().rect_filled(
        rect,
        3.0,
        if on {
            theme.accent_wash
        } else {
            visuals.weak_bg_fill
        },
    );
    let dot = if on {
        Color32::from_rgb(0xe0, 0x3c, 0x3c)
    } else {
        visuals.fg_stroke.color
    };
    ui.painter()
        .circle_filled(rect.center(), rect.height() * 0.2, dot);
    response
}

#[derive(Clone, Copy)]
enum Icon {
    Play,
    Pause,
    JumpStart,
    JumpEnd,
}

/// A small square button with a painted transport icon — painted because the
/// bundled font has no glyphs for them.
fn icon_button(ui: &mut Ui, icon: Icon) -> egui::Response {
    let size = Vec2::splat(ui.spacing().interact_size.y);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let visuals = ui.style().interact(&response);
    ui.painter().rect_filled(rect, 3.0, visuals.weak_bg_fill);
    let c = rect.center();
    let s = rect.height() * 0.22;
    let colour = visuals.fg_stroke.color;
    let triangle = |tip: f32| {
        vec![
            Pos2::new(c.x - s * tip, c.y - s),
            Pos2::new(c.x + s * tip, c.y),
            Pos2::new(c.x - s * tip, c.y + s),
        ]
    };
    let painter = ui.painter();
    match icon {
        Icon::Play => {
            painter.add(egui::Shape::convex_polygon(
                triangle(1.0),
                colour,
                Stroke::NONE,
            ));
        }
        Icon::Pause => {
            for dx in [-s * 0.55, s * 0.55] {
                painter.rect_filled(
                    Rect::from_center_size(Pos2::new(c.x + dx, c.y), Vec2::new(s * 0.5, s * 2.0)),
                    0.0,
                    colour,
                );
            }
        }
        Icon::JumpStart | Icon::JumpEnd => {
            let flip = if matches!(icon, Icon::JumpEnd) {
                1.0
            } else {
                -1.0
            };
            let pts: Vec<Pos2> = triangle(0.8)
                .into_iter()
                .map(|p| Pos2::new(c.x + (p.x - c.x) * flip, p.y))
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, colour, Stroke::NONE));
            let bar_x = c.x + flip * s * 1.0;
            painter.line_segment(
                [Pos2::new(bar_x, c.y - s), Pos2::new(bar_x, c.y + s)],
                Stroke::new(2.0, colour),
            );
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_goes_to_a_place_and_back() {
        let rect = Rect::from_min_size(Pos2::new(40.0, 0.0), Vec2::new(600.0, 50.0));
        let range = visible(1, 250);
        for frame in [1.0, 17.0, 125.5, 250.0] {
            let back = x_to_frame(frame_to_x(frame, range, rect), range, rect);
            assert!((back - frame).abs() < 1e-3, "{frame} came back as {back}");
        }
        // And the whole range is on the ruler, with room either side.
        assert!(frame_to_x(1.0, range, rect) > rect.min.x);
        assert!(frame_to_x(250.0, range, rect) < rect.max.x);
    }

    #[test]
    fn the_numbers_never_crowd() {
        for ppf in [0.1f32, 0.5, 1.0, 3.0, 12.0, 80.0] {
            let step = tick_step(ppf, 44.0);
            assert!(
                step as f32 * ppf >= 44.0,
                "{ppf} points a frame gave a step of {step}"
            );
            // And not needlessly sparse: the next smaller step would crowd.
            let smaller = [1, 2, 5]
                .iter()
                .flat_map(|m| (0..7).map(move |d| m * 10i32.pow(d)))
                .filter(|s| *s < step)
                .max();
            if let Some(smaller) = smaller {
                assert!(
                    (smaller as f32) * ppf < 44.0,
                    "{ppf}: {smaller} would have fitted"
                );
            }
        }
    }

    /// Drives the panel with real pointer events, a frame at a time.
    fn scrub(from_frame: f64, to_frame: f64) -> Vec<TimelineEdit> {
        let ctx = egui::Context::default();
        let mut state = TimelineState::default();
        let view = TimelineView {
            frame: 1.0,
            start: 1,
            end: 100,
            playing: false,
            autokey: false,
            keys: vec![10.0],
            // A marker under the drag: it must not catch the pointer.
            markers: vec![TimelineMarker {
                frame: 40,
                name: "Cam".into(),
                camera: true,
            }],
            active_camera: false,
        };
        let text = TimelineText {
            play: String::new(),
            pause: String::new(),
            jump_start: String::new(),
            jump_end: String::new(),
            start: String::new(),
            end: String::new(),
            autokey: String::new(),
            add_marker: String::new(),
            bind_camera: String::new(),
            remove_marker: String::new(),
        };
        let mut out = Vec::new();
        let mut time = 0.0;
        let mut run =
            |events: Vec<egui::Event>, out: &mut Vec<TimelineEdit>, state: &mut TimelineState| {
                time += 1.0 / 60.0;
                let raw = egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 120.0))),
                    time: Some(time),
                    events,
                    ..Default::default()
                };
                let _ = ctx.run_ui(raw, |ui| out.extend(timeline(ui, &view, state, &text)));
            };
        run(vec![], &mut Vec::new(), &mut state);
        run(vec![], &mut Vec::new(), &mut state);
        let track = {
            let r = state.rect.unwrap();
            Rect::from_min_max(Pos2::new(r.min.x, r.min.y + 40.0), r.max)
        };
        let at = |f: f64| {
            let range = visible(1, 100);
            // The track starts below the control bar; any y inside it will do.
            Pos2::new(
                frame_to_x(
                    f,
                    range,
                    Rect::from_min_max(Pos2::new(track.min.x, 0.0), Pos2::new(track.max.x, 1.0)),
                ),
                track.center().y,
            )
        };
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        let (a, b) = (at(from_frame), at(to_frame));
        run(vec![egui::Event::PointerMoved(a)], &mut out, &mut state);
        run(vec![button(a, true)], &mut out, &mut state);
        for i in 1..=4 {
            run(
                vec![egui::Event::PointerMoved(a + (b - a) * (i as f32 / 4.0))],
                &mut out,
                &mut state,
            );
        }
        run(vec![button(b, false)], &mut out, &mut state);
        out
    }

    #[test]
    fn scrubbing_moves_the_clock_to_whole_frames_under_the_pointer() {
        let edits = scrub(20.0, 60.0);
        let frames: Vec<f64> = edits
            .iter()
            .filter_map(|e| match e {
                TimelineEdit::SetFrame(f) => Some(*f),
                _ => None,
            })
            .collect();
        assert!(!frames.is_empty(), "a scrub moved nothing: {edits:?}");
        assert!(
            frames.iter().all(|f| f.fract() == 0.0),
            "a scrub landed between frames: {frames:?}"
        );
        let last = *frames.last().unwrap();
        assert!(
            (last - 60.0).abs() <= 1.0,
            "the scrub ended at {last}, not under the pointer"
        );
        assert!(
            frames.windows(2).all(|w| w[0] <= w[1]),
            "the clock went backwards on a forward drag"
        );
    }
}
