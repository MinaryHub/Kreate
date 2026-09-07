//! The UV editor: a second viewport, over the texture map instead of the world.
//!
//! Draws the active UV set of the mesh being edited and lets the user select
//! and navigate in it. What it does *not* do is change anything: it emits
//! [`Command`]s like every other panel, and the operators behind them are the
//! same ones a script or the right-click menu reaches.
//!
//! # Layout is separated from drawing
//!
//! [`to_screen`], [`to_uv`], [`pick`] and [`inside`] are pure and take no
//! painter, so "does clicking here select that corner" is a unit test rather
//! than something only a person with a mouse can check. The same split the
//! navigation gizmo uses, for the same reason.
//!
//! # Why egui rather than a wgpu pass
//!
//! The UV view is a few thousand lines and points with no shading, no depth
//! and no occlusion — egui draws that comfortably, and the 3D viewport's
//! machinery would buy nothing. It is not free at scale, though: a
//! million-corner map would need the instanced screen-space quads the 3D
//! wireframe uses. Until it does, [`Detail`] fades the corner dots out at the
//! density where they stop being readable anyway, which is the same answer the
//! 3D overlay gives and for the same reason.

use egui::{Pos2, Rect, Sense, Stroke, Ui, Vec2};
use kreate_reflect::{PropertyMap, PropertyValue};

use crate::command::Command;
use crate::theme;

/// The UV data one frame needs.
///
/// Borrowed slices indexed by *corner slot*, which is what the operators take
/// and what the mesh keeps its UVs in. Nothing here is owned, so publishing
/// this costs no allocation on a frame where nothing changed.
pub struct UvView<'a> {
    /// UVs, by corner slot.
    pub uvs: &'a [egui::Vec2],
    /// Which corners are selected, by corner slot. May be shorter than `uvs`,
    /// or empty when the layer does not exist yet.
    pub selected: &'a [bool],
    /// CSR offsets over [`Self::corners`], one more than the face count.
    pub offsets: &'a [u32],
    /// Corner slots, grouped into faces by [`Self::offsets`].
    pub corners: &'a [u32],
}

impl UvView<'_> {
    pub fn face_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn face(&self, index: usize) -> &[u32] {
        let (Some(from), Some(to)) = (self.offsets.get(index), self.offsets.get(index + 1)) else {
            return &[];
        };
        let (from, to) = (*from as usize, *to as usize);
        self.corners.get(from..to).unwrap_or(&[])
    }

    fn uv(&self, slot: u32) -> Option<egui::Vec2> {
        self.uvs.get(slot as usize).copied()
    }

    fn is_selected(&self, slot: u32) -> bool {
        self.selected.get(slot as usize).copied().unwrap_or(false)
    }

    /// Whether there is anything to draw at all.
    pub fn is_empty(&self) -> bool {
        self.uvs.is_empty() || self.face_count() == 0
    }
}

/// The same data, owned.
///
/// The application publishes this once a frame and the panel borrows a
/// [`UvView`] out of it. Owned rather than borrowed at the application level
/// because the UVs live behind the edit session, which the interface may not
/// hold a reference into — `kreate-ui` reads a *view* of the world, and this
/// is that view made concrete.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UvData {
    pub uvs: Vec<egui::Vec2>,
    pub selected: Vec<bool>,
    pub offsets: Vec<u32>,
    pub corners: Vec<u32>,
}

impl UvData {
    pub fn view(&self) -> UvView<'_> {
        UvView {
            uvs: &self.uvs,
            selected: &self.selected,
            offsets: &self.offsets,
            corners: &self.corners,
        }
    }
}

/// Pan, zoom, and any drag in progress.
///
/// Lives in the application's interface state rather than in egui's memory, so
/// the view survives the panel being closed and reopened — an artist who has
/// zoomed into a sleeve does not want it reset because they glanced at the
/// modifier stack.
#[derive(Clone, Debug)]
pub struct UvState {
    /// The UV coordinate at the centre of the view.
    pub centre: Vec2,
    /// Pixels per UV unit.
    pub scale: f32,
    /// A selection band being dragged, in screen space.
    pub band: Option<(Pos2, Pos2)>,
    /// Whether the band removes rather than adds.
    pub band_deselect: bool,
}

impl Default for UvState {
    fn default() -> Self {
        Self {
            // The unit square centred, at a scale the panel corrects on the
            // first frame it knows its own size.
            centre: Vec2::splat(0.5),
            scale: 256.0,
            band: None,
            band_deselect: false,
        }
    }
}

/// The lowest and highest zoom the view allows.
///
/// Bounded because an unbounded scale is how a wheel event turns into `inf`
/// and every subsequent frame draws nothing, with no error to explain it.
const MIN_SCALE: f32 = 16.0;
const MAX_SCALE: f32 = 65_536.0;

/// How near the cursor a corner must be to be picked, in points.
///
/// Screen-space, not UV-space, so the tolerance is the same whether the view
/// is zoomed in or out — the same rule the 3D picking uses, and the reason
/// picking feels uniform.
pub const PICK_RADIUS: f32 = 10.0;

/// Where a UV lands on screen.
pub fn to_screen(state: &UvState, rect: Rect, uv: Vec2) -> Pos2 {
    let centre = rect.center();
    // V grows upward in UV space and downward on screen. Flipping here rather
    // than in the data is deliberate: the mesh holds one convention and every
    // format's flip is handled at the IO boundary, so a second flip anywhere
    // else would be a second place to get it wrong.
    Pos2::new(
        centre.x + (uv.x - state.centre.x) * state.scale,
        centre.y - (uv.y - state.centre.y) * state.scale,
    )
}

/// Where a screen point falls in UV space.
pub fn to_uv(state: &UvState, rect: Rect, pos: Pos2) -> Vec2 {
    let centre = rect.center();
    Vec2::new(
        state.centre.x + (pos.x - centre.x) / state.scale,
        state.centre.y - (pos.y - centre.y) / state.scale,
    )
}

/// The corner nearest `pos`, if one is within `radius` points.
///
/// Ties break toward the lower slot so a click is reproducible: two corners at
/// the same UV — which is every corner on an unseparated seam — must not
/// select a different one each time.
pub fn pick(view: &UvView<'_>, state: &UvState, rect: Rect, pos: Pos2, radius: f32) -> Option<u32> {
    let mut best: Option<(f32, u32)> = None;
    for face in 0..view.face_count() {
        for slot in view.face(face) {
            let Some(uv) = view.uv(*slot) else {
                continue;
            };
            let d = (to_screen(state, rect, uv) - pos).length();
            if d > radius {
                continue;
            }
            let better = match best {
                None => true,
                Some((bd, bs)) => d < bd || (d == bd && *slot < bs),
            };
            if better {
                best = Some((d, *slot));
            }
        }
    }
    best.map(|(_, slot)| slot)
}

/// Every corner inside a screen-space rectangle, in ascending slot order.
pub fn inside(view: &UvView<'_>, state: &UvState, rect: Rect, band: Rect) -> Vec<u32> {
    let mut out = Vec::new();
    for face in 0..view.face_count() {
        for slot in view.face(face) {
            let Some(uv) = view.uv(*slot) else {
                continue;
            };
            if band.contains(to_screen(state, rect, uv)) {
                out.push(*slot);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Zooms about a fixed screen point.
///
/// About the cursor rather than the centre, so the thing under the pointer
/// stays under it. Zooming about the centre means every zoom is followed by a
/// pan to undo the drift, which is the difference between a view that feels
/// direct and one that feels like it is arguing.
pub fn zoom_about(state: &mut UvState, rect: Rect, anchor: Pos2, factor: f32) {
    let before = to_uv(state, rect, anchor);
    state.scale = (state.scale * factor).clamp(MIN_SCALE, MAX_SCALE);
    let after = to_uv(state, rect, anchor);
    state.centre += before - after;
}

/// Frames the unit square in the panel.
pub fn frame_unit_square(state: &mut UvState, rect: Rect) {
    let shorter = rect.width().min(rect.height());
    state.centre = Vec2::splat(0.5);
    state.scale = (shorter * 0.86).clamp(MIN_SCALE, MAX_SCALE);
}

/// How much detail the view draws at the current density.
///
/// The same idea as the 3D overlay's density LOD: below about one point per
/// pixel the dots are a grey smear that cannot be read or clicked, so drawing
/// them costs time and *reduces* legibility. Reported rather than applied
/// silently so a test can assert on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Detail {
    pub draw_corners: bool,
    pub draw_edges: bool,
}

/// Decides what to draw, from how many corners share how many pixels.
pub fn detail_for(corners: usize, rect: Rect) -> Detail {
    let pixels = (rect.width() * rect.height()).max(1.0);
    let per_pixel = corners as f32 / pixels;
    Detail {
        draw_corners: per_pixel < 0.08,
        // Edges stay much longer: a wireframe fading into a wash still reads
        // as shape, where a field of dots does not.
        draw_edges: per_pixel < 2.0,
    }
}

/// The panel's translated text.
pub struct UvText {
    /// Shown when the mesh has no UVs yet.
    pub no_uvs: String,
    /// Shown when the panel is open outside edit mode.
    pub not_editing: String,
}

/// Draws the editor and returns what the user asked for.
pub fn uv_editor(
    ui: &mut Ui,
    view: Option<&UvView<'_>>,
    state: &mut UvState,
    text: &UvText,
) -> Vec<Command> {
    let mut out = Vec::new();
    let theme = theme::from_ui(ui);
    let rect = ui.available_rect_before_wrap();
    if rect.width() < 32.0 || rect.height() < 32.0 {
        return out;
    }

    let response = ui.allocate_rect(rect, Sense::click_and_drag());
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, theme.sunken);

    let Some(view) = view else {
        centred_note(ui, rect, &text.not_editing, theme.text_faint);
        return out;
    };
    if view.is_empty() {
        draw_unit_square(&painter, state, rect, &theme);
        centred_note(ui, rect, &text.no_uvs, theme.text_faint);
        return out;
    }

    // --- navigation ---------------------------------------------------------
    // Handled before selection so a zoom mid-drag does not also commit a band
    // resolved against the old scale.
    if let Some(pos) = response.hover_pos() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            zoom_about(state, rect, pos, (scroll * 0.004).exp());
        }
    }
    let pan = ui.input(|i| i.pointer.middle_down());
    if pan {
        let delta = response.drag_delta();
        state.centre -= Vec2::new(delta.x, -delta.y) / state.scale;
    }

    // --- drawing ------------------------------------------------------------
    draw_unit_square(&painter, state, rect, &theme);
    let detail = detail_for(view.uvs.len(), rect);

    if detail.draw_edges {
        let wire = theme.border_strong;
        for face in 0..view.face_count() {
            let slots = view.face(face);
            for (i, slot) in slots.iter().enumerate() {
                let next = slots[(i + 1) % slots.len()];
                let (Some(a), Some(b)) = (view.uv(*slot), view.uv(next)) else {
                    continue;
                };
                painter.line_segment(
                    [to_screen(state, rect, a), to_screen(state, rect, b)],
                    Stroke::new(1.0, wire),
                );
            }
        }
    }

    if detail.draw_corners {
        // Selected last, so a selected corner is never hidden under an
        // unselected one drawn over it.
        for pass in [false, true] {
            for face in 0..view.face_count() {
                for slot in view.face(face) {
                    if view.is_selected(*slot) != pass {
                        continue;
                    }
                    let Some(uv) = view.uv(*slot) else {
                        continue;
                    };
                    let colour = if pass { theme.accent } else { theme.text_dim };
                    let size = if pass { 3.5 } else { 2.5 };
                    painter.circle_filled(to_screen(state, rect, uv), size, colour);
                }
            }
        }
    }

    // --- selection ----------------------------------------------------------
    let (extend, deselect) = ui.input(|i| (i.modifiers.shift, i.modifiers.ctrl));

    if response.drag_started() && !pan {
        if let Some(pos) = response.interact_pointer_pos() {
            state.band = Some((pos, pos));
            state.band_deselect = deselect;
        }
    }
    if let Some((from, _)) = state.band {
        if response.dragged() && !pan {
            if let Some(pos) = response.interact_pointer_pos() {
                state.band = Some((from, pos));
            }
        }
    }
    if let Some((from, to)) = state.band {
        let band = Rect::from_two_pos(from, to).intersect(rect);
        if band.width() > 1.0 || band.height() > 1.0 {
            let colour = if state.band_deselect {
                theme.info
            } else {
                theme.accent
            };
            painter.rect_filled(band, 0.0, colour.gamma_multiply(0.12));
            painter.rect_stroke(
                band,
                0.0,
                Stroke::new(1.0, colour),
                egui::StrokeKind::Inside,
            );
        }
    }

    if response.drag_stopped() {
        if let Some((from, to)) = state.band.take() {
            let band = Rect::from_two_pos(from, to);
            // A drag that never moved is a click. Deciding on release rather
            // than on press is what lets one gesture be either without a
            // selection that flickers to something else halfway through.
            if band.width() < 2.0 && band.height() < 2.0 {
                out.extend(click(view, state, rect, from, extend, deselect));
            } else {
                let corners = inside(view, state, rect, band);
                out.push(select_command(corners, extend, state.band_deselect));
            }
        }
    } else if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            out.extend(click(view, state, rect, pos, extend, deselect));
        }
    }

    out
}

/// What a click resolves to.
///
/// A miss with no modifier clears the selection, which is what clicking empty
/// space means everywhere else. A miss *with* a modifier does nothing: the
/// user was adding to a selection and missed, and throwing their work away for
/// a slip is the wrong answer.
fn click(
    view: &UvView<'_>,
    state: &UvState,
    rect: Rect,
    pos: Pos2,
    extend: bool,
    deselect: bool,
) -> Option<Command> {
    match pick(view, state, rect, pos, PICK_RADIUS) {
        Some(slot) => Some(select_command(vec![slot], extend, deselect)),
        None if extend || deselect => None,
        None => Some(select_command(Vec::new(), false, false)),
    }
}

fn select_command(corners: Vec<u32>, extend: bool, deselect: bool) -> Command {
    let mut params = PropertyMap::new();
    params.set("corners", PropertyValue::Ints(corners));
    params.set("extend", PropertyValue::Bool(extend || deselect));
    params.set("deselect", PropertyValue::Bool(deselect));
    Command::RunOperator {
        op_id: "uv.select_set".to_string(),
        params,
    }
}

/// The unit square and the texel grid inside it.
fn draw_unit_square(
    painter: &egui::Painter,
    state: &UvState,
    rect: Rect,
    theme: &crate::theme::Theme,
) {
    let min = to_screen(state, rect, Vec2::new(0.0, 1.0));
    let max = to_screen(state, rect, Vec2::new(1.0, 0.0));
    let square = Rect::from_two_pos(min, max);
    painter.rect_filled(square, 0.0, theme.base);

    // Tenths, and only while they are far enough apart to read. Log-scaled
    // subdivision the way the 3D grid does it would be over-engineering here:
    // UV space has a natural extent, so there are exactly two useful levels.
    let step = state.scale * 0.1;
    if step > 8.0 {
        for i in 1..10 {
            let t = i as f32 * 0.1;
            let x = to_screen(state, rect, Vec2::new(t, 0.0)).x;
            let y = to_screen(state, rect, Vec2::new(0.0, t)).y;
            painter.line_segment(
                [Pos2::new(x, square.min.y), Pos2::new(x, square.max.y)],
                Stroke::new(1.0, theme.border),
            );
            painter.line_segment(
                [Pos2::new(square.min.x, y), Pos2::new(square.max.x, y)],
                Stroke::new(1.0, theme.border),
            );
        }
    }
    painter.rect_stroke(
        square,
        0.0,
        Stroke::new(1.0, theme.border_strong),
        egui::StrokeKind::Outside,
    );
}

fn centred_note(ui: &mut Ui, rect: Rect, message: &str, colour: egui::Color32) {
    let painter = ui.painter().with_clip_rect(rect);
    let font = egui::TextStyle::Body.resolve(ui.style());
    let galley = painter.layout(
        message.to_string(),
        font,
        colour,
        (rect.width() * 0.7).max(64.0),
    );
    let at = rect.center() - egui::vec2(galley.size().x, galley.size().y) * 0.5;
    painter.galley(at, galley, colour);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One quad, UVs at the corners of the unit square.
    fn quad() -> (Vec<egui::Vec2>, Vec<bool>, Vec<u32>, Vec<u32>) {
        let uvs = vec![
            egui::vec2(0.0, 0.0),
            egui::vec2(1.0, 0.0),
            egui::vec2(1.0, 1.0),
            egui::vec2(0.0, 1.0),
        ];
        (uvs, vec![false; 4], vec![0, 4], vec![0, 1, 2, 3])
    }

    fn view<'a>(
        uvs: &'a [egui::Vec2],
        selected: &'a [bool],
        offsets: &'a [u32],
        corners: &'a [u32],
    ) -> UvView<'a> {
        UvView {
            uvs,
            selected,
            offsets,
            corners,
        }
    }

    fn panel() -> Rect {
        Rect::from_min_size(Pos2::new(100.0, 50.0), egui::vec2(400.0, 400.0))
    }

    #[test]
    fn the_two_mappings_are_inverses() {
        let state = UvState {
            centre: Vec2::new(0.3, -0.7),
            scale: 137.0,
            ..Default::default()
        };
        let rect = panel();
        for uv in [
            Vec2::ZERO,
            Vec2::splat(0.5),
            Vec2::new(-2.0, 3.5),
            Vec2::new(1.0, 0.0),
        ] {
            let back = to_uv(&state, rect, to_screen(&state, rect, uv));
            assert!((back - uv).length() < 1e-4, "{uv:?} -> {back:?}");
        }
    }

    #[test]
    fn v_grows_upward_on_screen() {
        // The one convention worth pinning: UV space has V up, screens have Y
        // down. Getting it wrong flips every map vertically, which looks like
        // an import bug and is not one.
        let state = UvState::default();
        let rect = panel();
        let low = to_screen(&state, rect, Vec2::new(0.5, 0.0));
        let high = to_screen(&state, rect, Vec2::new(0.5, 1.0));
        assert!(high.y < low.y, "V=1 must be above V=0: {high:?} {low:?}");
    }

    #[test]
    fn a_click_on_a_corner_picks_it() {
        let (uvs, sel, offsets, corners) = quad();
        let v = view(&uvs, &sel, &offsets, &corners);
        let mut state = UvState::default();
        let rect = panel();
        frame_unit_square(&mut state, rect);

        for (slot, uv) in uvs.iter().enumerate() {
            let at = to_screen(&state, rect, *uv);
            assert_eq!(
                pick(&v, &state, rect, at, PICK_RADIUS),
                Some(slot as u32),
                "clicking {uv:?} did not pick its own corner"
            );
        }
    }

    #[test]
    fn a_click_in_empty_space_picks_nothing() {
        let (uvs, sel, offsets, corners) = quad();
        let v = view(&uvs, &sel, &offsets, &corners);
        let mut state = UvState::default();
        let rect = panel();
        frame_unit_square(&mut state, rect);
        let middle = to_screen(&state, rect, Vec2::splat(0.5));
        assert_eq!(pick(&v, &state, rect, middle, PICK_RADIUS), None);
    }

    #[test]
    fn the_pick_tolerance_is_in_screen_space() {
        // Zooming out must not make corners harder to hit. A tolerance in UV
        // units would shrink with the view and picking would feel worse the
        // further out you go, which is the complaint that made the 3D picking
        // screen-space too.
        let (uvs, sel, offsets, corners) = quad();
        let v = view(&uvs, &sel, &offsets, &corners);
        let rect = panel();

        for scale in [32.0, 256.0, 2048.0] {
            let state = UvState {
                centre: Vec2::splat(0.5),
                scale,
                ..Default::default()
            };
            let at = to_screen(&state, rect, uvs[0]);
            let near = at + egui::vec2(PICK_RADIUS * 0.8, 0.0);
            let far = at + egui::vec2(PICK_RADIUS * 1.5, 0.0);
            assert_eq!(
                pick(&v, &state, rect, near, PICK_RADIUS),
                Some(0),
                "a near miss failed at scale {scale}"
            );
            assert_eq!(
                pick(&v, &state, rect, far, PICK_RADIUS),
                None,
                "a far miss hit at scale {scale}"
            );
        }
    }

    #[test]
    fn a_tie_breaks_toward_the_lower_slot() {
        // Two corners at the same UV is every corner on an unseparated seam. A
        // click there must pick the same one every time, or the selection
        // changes under a user who did not move the mouse.
        //
        // Both iteration orders, because that is the whole claim. With only
        // one order, "the lowest slot wins" and "the last one iterated wins"
        // can agree — which they did, and the first version of this test
        // passed against a rule that was really iteration order.
        for corners in [vec![0u32, 1], vec![1, 0]] {
            let uvs = vec![egui::vec2(0.5, 0.5), egui::vec2(0.5, 0.5)];
            let sel = vec![false; 2];
            let offsets = vec![0, 2];
            let v = view(&uvs, &sel, &offsets, &corners);
            let mut state = UvState::default();
            let rect = panel();
            frame_unit_square(&mut state, rect);
            let at = to_screen(&state, rect, egui::vec2(0.5, 0.5));
            assert_eq!(
                pick(&v, &state, rect, at, PICK_RADIUS),
                Some(0),
                "iteration order {corners:?} changed which corner was picked"
            );
        }
    }

    #[test]
    fn a_band_takes_what_it_encloses_and_no_more() {
        let (uvs, sel, offsets, corners) = quad();
        let v = view(&uvs, &sel, &offsets, &corners);
        let mut state = UvState::default();
        let rect = panel();
        frame_unit_square(&mut state, rect);

        // A band around the bottom edge of the square: UV V=0, which is the
        // *lower* half on screen.
        let a = to_screen(&state, rect, Vec2::new(-0.1, 0.4));
        let b = to_screen(&state, rect, Vec2::new(1.1, -0.1));
        let got = inside(&v, &state, rect, Rect::from_two_pos(a, b));
        assert_eq!(got, vec![0, 1], "got {got:?}");

        // And everything.
        let a = to_screen(&state, rect, Vec2::new(-1.0, 2.0));
        let b = to_screen(&state, rect, Vec2::new(2.0, -1.0));
        assert_eq!(
            inside(&v, &state, rect, Rect::from_two_pos(a, b)),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn zooming_keeps_the_point_under_the_cursor() {
        // The difference between a view that feels direct and one that drifts.
        let mut state = UvState::default();
        let rect = panel();
        frame_unit_square(&mut state, rect);
        let anchor = rect.center() + egui::vec2(80.0, -40.0);
        let before = to_uv(&state, rect, anchor);
        zoom_about(&mut state, rect, anchor, 2.5);
        let after = to_uv(&state, rect, anchor);
        assert!(
            (after - before).length() < 1e-4,
            "the anchor drifted: {before:?} -> {after:?}"
        );
    }

    #[test]
    fn the_zoom_is_bounded() {
        // An unbounded scale is how a wheel event becomes `inf` and every
        // later frame draws nothing with no error to explain it.
        let mut state = UvState::default();
        let rect = panel();
        for _ in 0..200 {
            zoom_about(&mut state, rect, rect.center(), 4.0);
        }
        assert!(state.scale.is_finite() && state.scale <= MAX_SCALE);
        for _ in 0..400 {
            zoom_about(&mut state, rect, rect.center(), 0.25);
        }
        assert!(state.scale >= MIN_SCALE, "{}", state.scale);
    }

    #[test]
    fn framing_puts_the_unit_square_inside_the_panel() {
        let mut state = UvState::default();
        let rect = panel();
        frame_unit_square(&mut state, rect);
        let min = to_screen(&state, rect, Vec2::new(0.0, 1.0));
        let max = to_screen(&state, rect, Vec2::new(1.0, 0.0));
        let square = Rect::from_two_pos(min, max);
        assert!(rect.contains_rect(square), "{square:?} not inside {rect:?}");
        // And filling most of it, rather than a speck in the middle.
        assert!(square.width() > rect.width() * 0.7);
    }

    #[test]
    fn corner_dots_fade_out_before_they_become_a_smear() {
        // The density guard. Below roughly one dot per pixel they are
        // unreadable and unclickable, so drawing them costs time and reduces
        // legibility — the same conclusion the 3D overlay reached.
        let rect = panel();
        assert!(detail_for(500, rect).draw_corners);
        assert!(!detail_for(2_000_000, rect).draw_corners);
        // Edges survive much longer: a wireframe washing out still reads as
        // shape where a field of dots does not.
        assert!(detail_for(100_000, rect).draw_edges);
        assert!(!detail_for(10_000_000, rect).draw_edges);
    }

    #[test]
    fn a_face_with_a_bad_offset_range_is_skipped_rather_than_panicking() {
        // A view published one frame behind a topology edit really does arrive
        // with offsets past the end of the corner list, and a paint function
        // that panicked would take the window down.
        let uvs = vec![egui::vec2(0.0, 0.0)];
        let sel = vec![false];
        let offsets = vec![0, 99];
        let corners = vec![0];
        let v = view(&uvs, &sel, &offsets, &corners);
        let mut state = UvState::default();
        let rect = panel();
        frame_unit_square(&mut state, rect);
        assert_eq!(v.face_count(), 1);
        assert!(pick(&v, &state, rect, rect.center(), PICK_RADIUS).is_none());
        assert!(inside(&v, &state, rect, rect).is_empty());
    }

    #[test]
    fn a_selection_command_names_the_operator_and_the_set() {
        // The insulation rule: the panel emits `(op_id, params)` and nothing
        // else. A panel that reached for the document would be the one place
        // undo, macro recording and scripting stopped agreeing.
        let Command::RunOperator { op_id, params } = select_command(vec![3, 7], true, false) else {
            panic!("not an operator command");
        };
        assert_eq!(op_id, "uv.select_set");
        assert_eq!(
            params.get("corners"),
            Some(&PropertyValue::Ints(vec![3, 7]))
        );
        assert_eq!(params.get("extend"), Some(&PropertyValue::Bool(true)));
        assert_eq!(params.get("deselect"), Some(&PropertyValue::Bool(false)));
    }

    #[test]
    fn deselecting_implies_extending() {
        // Removing from a selection is only meaningful against one that stays,
        // so `deselect` must not arrive with `extend` false — which would ask
        // the operator to replace the selection with nothing and then subtract
        // from it.
        let Command::RunOperator { params, .. } = select_command(vec![1], false, true) else {
            panic!("not an operator command");
        };
        assert_eq!(params.get("extend"), Some(&PropertyValue::Bool(true)));
        assert_eq!(params.get("deselect"), Some(&PropertyValue::Bool(true)));
    }

    #[test]
    fn a_miss_clears_the_selection_but_not_while_a_modifier_is_held() {
        // Clicking empty space means "nothing", the way it does everywhere
        // else. But a *modified* click that missed is a slip while adding to a
        // selection, and throwing the user's work away for a slip is the wrong
        // answer.
        let (uvs, sel, offsets, corners) = quad();
        let v = view(&uvs, &sel, &offsets, &corners);
        let mut state = UvState::default();
        let rect = panel();
        frame_unit_square(&mut state, rect);
        let empty = to_screen(&state, rect, Vec2::splat(0.5));

        let Some(Command::RunOperator { params, .. }) =
            click(&v, &state, rect, empty, false, false)
        else {
            panic!("a bare miss must clear the selection");
        };
        assert_eq!(
            params.get("corners"),
            Some(&PropertyValue::Ints(Vec::new()))
        );

        assert!(click(&v, &state, rect, empty, true, false).is_none());
        assert!(click(&v, &state, rect, empty, false, true).is_none());
    }
}
