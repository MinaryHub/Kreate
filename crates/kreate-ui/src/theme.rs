//! The interface theme: one table of tokens, and the egui style built from it.
//!
//! Before this existed the chrome was egui's defaults with a handful of
//! `Color32::from_rgb` calls scattered across four files. That is not a small
//! tidiness problem: it means a contrast complaint, a density preference or a
//! light mode is a hunt rather than an edit, and it means two places that are
//! meant to show the same state drift apart without anyone noticing.
//!
//! The direction is *warm chrome, neutral viewport*. The surfaces here are
//! warm neutrals, low in saturation, so that a full working day in them is
//! comfortable. The viewport is deliberately **not** one of them — see
//! [`kreate_render::palette::VIEWPORT`] for why the warmth stops at its edge.
//!
//! Anything shared with the viewport — the accent, the axis colours — is read
//! from [`kreate_render::palette`] rather than repeated here. A selected row
//! in the outliner and a selected edge in the viewport are the same state, so
//! they must be the same colour, and the only way to keep that true is to have
//! one constant.

use egui::{Color32, CornerRadius, Stroke};
use kreate_render::palette;

/// How much room a control gets.
///
/// One switch rather than a page of sliders: the honest choice a user makes is
/// "I want to see more at once" or "I want to hit things reliably", and every
/// other metric follows from it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Density {
    /// 24 px controls. More rows on screen.
    Compact,
    /// 28 px controls. The default.
    #[default]
    Comfortable,
}

impl Density {
    /// The height of a control, in points.
    pub fn control_height(self) -> f32 {
        match self {
            Density::Compact => 24.0,
            Density::Comfortable => 28.0,
        }
    }

    /// Vertical padding inside a button, derived from the control height so
    /// the two cannot disagree.
    fn button_padding_y(self) -> f32 {
        match self {
            Density::Compact => 3.0,
            Density::Comfortable => 5.0,
        }
    }
}

/// Every colour and metric the chrome uses.
///
/// A value rather than a set of constants so a preference, a high-contrast
/// variant or a future light mode is a different `Theme`, not a different code
/// path through the panels.
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    /// Input fields and pressed states. The bottom of the ramp.
    pub sunken: Color32,
    /// Panel backgrounds.
    pub base: Color32,
    /// Headers, the status bar, menus and cards.
    pub raised: Color32,
    /// Dividers and control outlines.
    pub border: Color32,
    /// Focus rings and anything that has to read as an edge.
    pub border_strong: Color32,

    /// Values and names.
    pub text: Color32,
    /// Labels.
    pub text_dim: Color32,
    /// Section titles and shortcut hints.
    pub text_faint: Color32,

    /// Selection, an engaged tool, an active mode. The one accent.
    pub accent: Color32,
    /// The accent at low alpha, for a selected row's background.
    pub accent_wash: Color32,
    /// The active object, and informational emphasis.
    pub info: Color32,

    pub axis_x: Color32,
    pub axis_y: Color32,
    pub axis_z: Color32,

    /// Corner radius for controls.
    pub radius: u8,
    /// Height of a list row.
    pub row_height: f32,
    /// Padding inside a panel.
    pub panel_padding: f32,

    pub density: Density,
}

/// A hex constant as an opaque `Color32`.
fn hex(hex: u32) -> Color32 {
    let [r, g, b] = palette::rgb8(hex);
    Color32::from_rgb(r, g, b)
}

impl Default for Theme {
    fn default() -> Self {
        Self::studio(Density::default())
    }
}

impl Theme {
    /// Warm chrome, neutral viewport.
    pub fn studio(density: Density) -> Self {
        Self {
            sunken: hex(0x17_16_14),
            base: hex(0x20_1E_1B),
            raised: hex(0x2A_27_24),
            border: hex(0x36_32_30),
            border_strong: hex(0x47_42_3E),

            text: hex(0xED_E8_E2),
            text_dim: hex(0xA3_9D_95),
            text_faint: hex(0x6E_68_62),

            accent: hex(palette::ACCENT),
            // Alpha rather than a blended constant: the wash sits over more
            // than one surface, and a constant would be wrong on all but one.
            accent_wash: hex(palette::ACCENT).gamma_multiply(0.14),
            info: hex(palette::INFO),

            axis_x: hex(palette::AXIS_X),
            axis_y: hex(palette::AXIS_Y),
            axis_z: hex(palette::AXIS_Z),

            radius: 3,
            row_height: 24.0,
            panel_padding: 12.0,
            density,
        }
    }

    /// The colour of one axis, indexed `0..3`.
    ///
    /// Indexed because the callers that need it — a transform field, the
    /// navigation gizmo, a modal header — all work in loops over three axes,
    /// and three `if`s at each site is how one of them ends up disagreeing.
    pub fn axis(&self, index: usize) -> Color32 {
        match index {
            0 => self.axis_x,
            1 => self.axis_y,
            _ => self.axis_z,
        }
    }

    /// Installs this theme on a context.
    ///
    /// Sets the egui style *and* stores the theme itself, so a widget can
    /// reach a token egui has no slot for — an axis colour, the informational
    /// blue — through [`of`] without every panel signature growing a
    /// parameter. Threading it by hand is how one panel ends up on a stale
    /// copy.
    pub fn install(&self, ctx: &egui::Context) {
        ctx.set_global_style(self.style(&ctx.global_style()));
        ctx.data_mut(|d| d.insert_temp(slot(), *self));
    }

    /// This theme as an egui style, built from whatever the context already
    /// has so the font definitions and text styles are left alone.
    pub fn style(&self, base: &egui::Style) -> egui::Style {
        let mut style = base.clone();
        style.visuals = self.visuals();

        let s = &mut style.spacing;
        s.item_spacing = egui::vec2(8.0, 4.0);
        s.button_padding = egui::vec2(8.0, self.density.button_padding_y());
        s.interact_size = egui::vec2(40.0, self.density.control_height());
        s.window_margin = egui::Margin::same(self.panel_padding as i8);
        s.menu_margin = egui::Margin::same(6);
        s.indent = 16.0;
        s.icon_width = 14.0;
        s.icon_width_inner = 8.0;
        style
    }

    /// This theme as egui visuals.
    pub fn visuals(&self) -> egui::Visuals {
        let radius = CornerRadius::same(self.radius);
        let widget =
            |bg: Color32, weak: Color32, edge: Color32, text: Color32| egui::style::WidgetVisuals {
                bg_fill: bg,
                weak_bg_fill: weak,
                bg_stroke: Stroke::new(1.0, edge),
                corner_radius: radius,
                fg_stroke: Stroke::new(1.0, text),
                // No growth on hover. A control that changes size under the
                // pointer makes a dense panel feel unstable, and in a tool the
                // pointer is over a control most of the time.
                expansion: 0.0,
            };

        let mut visuals = egui::Visuals::dark();
        visuals.dark_mode = true;
        visuals.override_text_color = Some(self.text);
        visuals.weak_text_color = Some(self.text_dim);

        visuals.panel_fill = self.base;
        visuals.window_fill = self.raised;
        visuals.faint_bg_color = self.raised;
        visuals.extreme_bg_color = self.sunken;
        visuals.code_bg_color = self.sunken;
        visuals.window_stroke = Stroke::new(1.0, self.border);
        visuals.window_corner_radius = CornerRadius::same(4);
        visuals.menu_corner_radius = CornerRadius::same(4);

        visuals.widgets.noninteractive = widget(self.base, self.base, self.border, self.text_dim);
        visuals.widgets.inactive = widget(self.raised, self.raised, self.border, self.text);
        visuals.widgets.hovered = widget(self.border, self.border, self.border_strong, self.text);
        // An engaged control wears the accent, because that is the same thing
        // the accent means in the viewport: this is what you are acting on.
        visuals.widgets.active = widget(
            self.border_strong,
            self.border_strong,
            self.accent,
            self.text,
        );
        visuals.widgets.open = widget(self.sunken, self.sunken, self.border_strong, self.text);

        visuals.selection = egui::style::Selection {
            bg_fill: self.accent_wash,
            stroke: Stroke::new(1.0, self.accent),
        };
        visuals.hyperlink_color = self.accent;
        visuals.warn_fg_color = self.accent;
        visuals.error_fg_color = hex(0xE6_73_7F);
        visuals.text_cursor.stroke = Stroke::new(2.0, self.accent);

        visuals
    }
}

/// Installs the default theme on a context.
pub fn install(ctx: &egui::Context) {
    Theme::default().install(ctx);
}

/// Where the installed theme is kept on the context.
fn slot() -> egui::Id {
    egui::Id::new("kreate::theme")
}

/// The theme installed on a context.
///
/// Falls back to the default rather than panicking: a panel drawn before
/// installation should be legible, not black on black, and a paint function is
/// the worst possible place to take the window down.
pub fn of(ctx: &egui::Context) -> Theme {
    ctx.data(|d| d.get_temp(slot())).unwrap_or_default()
}

/// The theme installed on the context this `Ui` belongs to.
pub fn from_ui(ui: &egui::Ui) -> Theme {
    of(ui.ctx())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shared_colours_come_from_the_palette() {
        // The whole reason the palette lives in the lower crate. If these ever
        // stop matching, a selected row and a selected edge are two different
        // oranges and the interface is quietly lying about state.
        let t = Theme::default();
        for (chrome, shared) in [
            (t.accent, palette::ACCENT),
            (t.info, palette::INFO),
            (t.axis_x, palette::AXIS_X),
            (t.axis_y, palette::AXIS_Y),
            (t.axis_z, palette::AXIS_Z),
        ] {
            let [r, g, b] = palette::rgb8(shared);
            assert_eq!(
                (chrome.r(), chrome.g(), chrome.b()),
                (r, g, b),
                "{shared:#08X} drifted"
            );
        }
    }

    #[test]
    fn the_chrome_is_warm_and_the_ramp_ascends() {
        let t = Theme::studio(Density::Comfortable);
        // Warm: red above blue on every surface. This is the direction's whole
        // premise, and a paste of a cool hex would silently undo it.
        for s in [t.sunken, t.base, t.raised, t.border, t.border_strong] {
            assert!(
                s.r() > s.b(),
                "surface {:?} is not warm",
                (s.r(), s.g(), s.b())
            );
        }
        // And the ramp has to actually ascend, or panels stop reading as
        // layers. Compared on the red channel, which is monotone here.
        let ramp = [t.sunken, t.base, t.raised, t.border, t.border_strong];
        for pair in ramp.windows(2) {
            assert!(
                pair[1].r() > pair[0].r(),
                "the surface ramp does not ascend: {:?}",
                ramp.map(|c| c.r())
            );
        }
    }

    #[test]
    fn the_text_ramp_descends_and_stays_legible() {
        let t = Theme::default();
        assert!(t.text.r() > t.text_dim.r());
        assert!(t.text_dim.r() > t.text_faint.r());

        // The faintest text still has to be readable on the darkest surface it
        // is used on. Contrast in linear luminance, which is what the eye
        // actually responds to; 3:1 is the WCAG floor for large text and the
        // honest minimum for a section title.
        let lum = |c: Color32| {
            let f = |v: u8| palette::srgb_to_linear(f32::from(v) / 255.0);
            0.2126 * f(c.r()) + 0.7152 * f(c.g()) + 0.0722 * f(c.b())
        };
        let ratio = (lum(t.text_faint) + 0.05) / (lum(t.sunken) + 0.05);
        assert!(ratio > 3.0, "faint text on sunken is only {ratio:.2}:1");

        let ratio = (lum(t.text) + 0.05) / (lum(t.base) + 0.05);
        assert!(ratio > 7.0, "primary text on base is only {ratio:.2}:1");
    }

    #[test]
    fn the_accent_reads_against_the_chrome_and_against_the_viewport() {
        // It has to work in both places, because it is one colour used in
        // both. A selection that is legible in the panels and invisible in the
        // viewport would be worse than two colours.
        let t = Theme::default();
        let lum = |r: u8, g: u8, b: u8| {
            let f = |v: u8| palette::srgb_to_linear(f32::from(v) / 255.0);
            0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
        };
        let accent = lum(t.accent.r(), t.accent.g(), t.accent.b());

        for (name, hex) in [("base", 0x20_1E_1B), ("viewport", palette::VIEWPORT)] {
            let [r, g, b] = palette::rgb8(hex);
            let ratio = (accent + 0.05) / (lum(r, g, b) + 0.05);
            assert!(ratio > 4.5, "accent on {name} is only {ratio:.2}:1");
        }
    }

    #[test]
    fn density_is_the_only_thing_that_moves_a_control_height() {
        assert_eq!(Density::Compact.control_height(), 24.0);
        assert_eq!(Density::Comfortable.control_height(), 28.0);

        let compact = Theme::studio(Density::Compact);
        let comfortable = Theme::studio(Density::Comfortable);
        let base = egui::Style::default();
        assert_eq!(
            compact.style(&base).spacing.interact_size.y,
            Density::Compact.control_height()
        );
        assert!(
            comfortable.style(&base).spacing.interact_size.y
                > compact.style(&base).spacing.interact_size.y
        );

        // And nothing else about the two differs: density is a metric, not a
        // second palette.
        assert_eq!(compact.accent, comfortable.accent);
        assert_eq!(compact.base, comfortable.base);
    }

    #[test]
    fn installing_a_theme_keeps_the_fonts_alone() {
        // Fonts are installed separately and earlier — see `fonts` — and a
        // theme that replaced the whole style would drop the CJK fallback
        // chain along with the Korean text it renders.
        let ctx = egui::Context::default();
        let mut style = (*ctx.global_style()).clone();
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(17.0));
        ctx.set_global_style(style);

        Theme::default().install(&ctx);
        assert_eq!(
            ctx.global_style().text_styles[&egui::TextStyle::Body],
            egui::FontId::proportional(17.0)
        );
        assert_eq!(ctx.global_style().visuals.panel_fill, Theme::default().base);
    }

    #[test]
    fn an_installed_theme_is_readable_back_and_an_absent_one_is_not_fatal() {
        let ctx = egui::Context::default();
        // Before installation. This is a real state — the first frame of the
        // first panel — and it has to be legible rather than a panic or a
        // black-on-black paint.
        assert_eq!(of(&ctx).base, Theme::default().base);

        let compact = Theme::studio(Density::Compact);
        compact.install(&ctx);
        assert_eq!(of(&ctx).density, Density::Compact);
        assert_eq!(of(&ctx).accent, compact.accent);
    }

    #[test]
    fn axis_colours_are_indexed_consistently() {
        let t = Theme::default();
        assert_eq!(t.axis(0), t.axis_x);
        assert_eq!(t.axis(1), t.axis_y);
        assert_eq!(t.axis(2), t.axis_z);
        // Out of range clamps to Z rather than panicking: the callers are
        // loops over three, and a panic in a paint function takes the window.
        assert_eq!(t.axis(9), t.axis_z);
    }
}
