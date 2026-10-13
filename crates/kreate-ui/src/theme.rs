//! The interface theme: one table of tokens, and the egui style built from it.
//!
//! Before this existed the chrome was egui's defaults with a handful of
//! `Color32::from_rgb` calls scattered across four files. That is not a small
//! tidiness problem: it means a contrast complaint, a density preference or a
//! light mode is a hunt rather than an edit, and it means two places that are
//! meant to show the same state drift apart without anyone noticing.
//!
//! The direction is *Graphite Workshop*: the model is the brightest thing on
//! screen. The chrome is three graphite steps — the window, the panels, the
//! fields and headers on them — a shade cooler and darker than the viewport,
//! so nothing beside the geometry competes with it, and the steps are far
//! enough apart that a panel, a field and a header read as layers at a
//! glance. One warm accent means selected or active, and nothing else.
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
    /// 24 px controls. More rows on screen. The default: a modeller's panels
    /// are lists of values, and the design keeps a row and a field one height.
    #[default]
    Compact,
    /// 28 px controls.
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

/// Which way round the chrome is: graphite, or light grey as Fusion 360 and
/// most CAD tools draw it.
///
/// The viewport is not part of it. The overlays drawn on the model — the
/// wireframe, the gizmos, the labels — are tuned against the dark viewport,
/// and a light chrome around a dark picture is how CAD tools that offer both
/// look anyway.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Appearance {
    #[default]
    Dark,
    Light,
}

impl Appearance {
    pub const ALL: [Appearance; 2] = [Appearance::Dark, Appearance::Light];

    /// The word a preferences file stores it as.
    pub fn tag(self) -> &'static str {
        match self {
            Appearance::Dark => "dark",
            Appearance::Light => "light",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.tag() == tag)
    }
}

/// Every colour and metric the chrome uses.
///
/// A value rather than a set of constants so a preference, a high-contrast
/// variant or a future light mode is a different `Theme`, not a different code
/// path through the panels.
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    /// The window behind the areas, and what a pressed control sinks to.
    /// The bottom of the ramp.
    pub sunken: Color32,
    /// Panel backgrounds.
    pub base: Color32,
    /// Fields, headers, menus and cards: what sits on a panel.
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
    /// Which way round the surfaces go. The panels that paint by hand ask
    /// this, through `dark`, rather than guessing from a colour.
    pub appearance: Appearance,
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
    /// The chrome for an appearance and a density.
    pub fn new(appearance: Appearance, density: Density) -> Self {
        match appearance {
            Appearance::Dark => Self::studio(density),
            Appearance::Light => Self::light(density),
        }
    }

    /// The tokens for what is drawn straight on the viewport — a label over
    /// the picture, a readout. The viewport is dark in either appearance, so
    /// that text is graphite's whatever the panels are.
    pub fn over_viewport(&self) -> Self {
        Self::studio(self.density)
    }

    /// Whether the surfaces are dark.
    pub fn dark(&self) -> bool {
        self.appearance == Appearance::Dark
    }

    /// Light grey chrome, as Fusion 360 draws its panels: white fields on a
    /// pale panel on a slightly darker window, dark text.
    ///
    /// The ramp runs the other way from graphite's — down from white rather
    /// than up from black — so a field still reads as a step from the panel
    /// it sits on. The accent is the same hue, darkened: the graphite
    /// orange on white is under 2:1, which is a selection nobody can find.
    pub fn light(density: Density) -> Self {
        let accent = hex(0x00C4_6A00);
        Self {
            sunken: hex(0x00D9_DCE1),
            base: hex(0x00EE_EFF2),
            raised: hex(0x00FF_FFFF),
            border: hex(0x00D2_D6DC),
            border_strong: hex(0x00B4_BAC3),

            text: hex(0x001C_1F24),
            text_dim: hex(0x0047_4D57),
            text_faint: hex(0x0062_6873),

            accent,
            accent_wash: hex(palette::ACCENT).gamma_multiply(0.22),
            info: hex(0x0026_6FC9),

            axis_x: hex(palette::AXIS_X),
            axis_y: hex(palette::AXIS_Y),
            axis_z: hex(palette::AXIS_Z),

            radius: 4,
            row_height: 24.0,
            panel_padding: 12.0,
            density,
            appearance: Appearance::Light,
        }
    }

    /// Graphite chrome around the viewport.
    pub fn studio(density: Density) -> Self {
        Self {
            sunken: hex(0x0012_1316),
            base: hex(0x001A_1B1F),
            raised: hex(0x0023_252A),
            border: hex(0x002E_3137),
            border_strong: hex(0x003A_3D44),

            text: hex(0x00E6_E7EA),
            text_dim: hex(0x00A3_A7AF),
            text_faint: hex(0x008A_8F98),

            accent: hex(palette::ACCENT),
            // Alpha rather than a blended constant: the wash sits over more
            // than one surface, and a constant would be wrong on all but one.
            accent_wash: hex(palette::ACCENT).gamma_multiply(0.14),
            info: hex(palette::INFO),

            axis_x: hex(palette::AXIS_X),
            axis_y: hex(palette::AXIS_Y),
            axis_z: hex(palette::AXIS_Z),

            radius: 4,
            row_height: 24.0,
            panel_padding: 12.0,
            density,
            appearance: Appearance::Dark,
        }
    }

    /// The quiet tint naming a mode, by its tag: the chip's fill and its
    /// text. `None` for object mode, the resting state, which needs no name
    /// tag. Only the mode's own name wears it — in the viewport header and
    /// the status bar — never a control: orange stays the one colour for
    /// what is selected.
    pub fn mode_tint(&self, tag: &str) -> Option<(Color32, Color32)> {
        // The text a light step on graphite, a dark step on the light chrome:
        // the same hue either way, legible on its own fill.
        let (fill, on_dark, on_light) = match tag {
            "edit_mesh" | "edit_armature" => (0x4F_B3_A9, 0x9F_DC_D5, 0x1D_6B_63),
            "sculpt" => (0xB4_8C_F0, 0xD3_BD_F7, 0x5B_3A_A6),
            "pose" => (0x78_AA_E6, 0xB9_D3_F2, 0x23_5E_A8),
            "weight_paint" => (0xE6_78_AA, 0xF2_BC_D4, 0xA3_2A_62),
            "texture_paint" => (0xD6_C8_78, 0xE8_DC_A2, 0x6E_60_10),
            _ => return None,
        };
        let text = if self.dark() { on_dark } else { on_light };
        Some((hex(fill).gamma_multiply(0.16), hex(text)))
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
        // Numbers in the monospaced face: a column of values lines up, and a
        // value scrubbed under the pointer does not jitter as its digits change.
        style.drag_value_text_style = egui::TextStyle::Monospace;

        let s = &mut style.spacing;
        s.item_spacing = egui::vec2(8.0, 4.0);
        s.button_padding = egui::vec2(8.0, self.density.button_padding_y());
        s.interact_size = egui::vec2(40.0, self.density.control_height());
        s.window_margin = egui::Margin::same(self.panel_padding as i8);
        s.menu_margin = egui::Margin::same(6);
        s.indent = 14.0;
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

        let mut visuals = if self.dark() {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        };
        visuals.dark_mode = self.dark();
        visuals.override_text_color = Some(self.text);
        visuals.weak_text_color = Some(self.text_dim);

        visuals.panel_fill = self.base;
        visuals.window_fill = self.raised;
        visuals.faint_bg_color = self.raised;
        // A text field sits on the panel as the other fields do, a step up,
        // rather than as a hole in it.
        visuals.extreme_bg_color = self.raised;
        visuals.code_bg_color = self.sunken;
        visuals.window_stroke = Stroke::new(1.0, self.border);
        visuals.window_corner_radius = CornerRadius::same(10);
        visuals.menu_corner_radius = CornerRadius::same(8);

        visuals.widgets.noninteractive = widget(self.base, self.base, self.border, self.text_dim);
        visuals.widgets.inactive = widget(self.raised, self.raised, self.border, self.text);
        visuals.widgets.hovered = widget(self.border, self.border, self.border_strong, self.text);
        // Fields and buttons sit on the panel as raised shapes with no
        // outline: the step in shade is the edge, as the design draws it.
        visuals.widgets.inactive.bg_stroke = Stroke::NONE;
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
    fn the_chrome_is_graphite_and_the_ramp_ascends() {
        let t = Theme::studio(Density::Comfortable);
        // Graphite: never warmer than neutral, so the one warm colour on
        // screen is the accent and what it marks.
        for s in [t.sunken, t.base, t.raised, t.border, t.border_strong] {
            assert!(
                s.b() >= s.r(),
                "surface {:?} is warm",
                (s.r(), s.g(), s.b())
            );
        }
        // And the ramp has to actually ascend, or panels stop reading as
        // layers.
        let ramp = [t.sunken, t.base, t.raised, t.border, t.border_strong];
        for pair in ramp.windows(2) {
            assert!(
                pair[1].b() > pair[0].b() && pair[1].r() > pair[0].r(),
                "the surface ramp does not ascend: {:?}",
                ramp.map(|c| (c.r(), c.b()))
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
            let f = |v: u8| kreate_math::srgb_to_linear(f32::from(v) / 255.0);
            0.2126 * f(c.r()) + 0.7152 * f(c.g()) + 0.0722 * f(c.b())
        };
        let ratio = (lum(t.text_faint) + 0.05) / (lum(t.sunken) + 0.05);
        assert!(ratio > 3.0, "faint text on sunken is only {ratio:.2}:1");
        // Faint text is drawn on the panels and the fields too, at the body
        // size: the 4.5:1 floor for small text, on the lighter of the two.
        let ratio = (lum(t.text_faint) + 0.05) / (lum(t.raised) + 0.05);
        assert!(ratio > 4.5, "faint text on a field is only {ratio:.2}:1");

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
            let f = |v: u8| kreate_math::srgb_to_linear(f32::from(v) / 255.0);
            0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
        };
        let accent = lum(t.accent.r(), t.accent.g(), t.accent.b());

        for (name, hex) in [("base", 0x001A_1B1F), ("viewport", palette::VIEWPORT)] {
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
    fn a_mode_is_named_by_its_own_tint_and_object_mode_by_none() {
        let t = Theme::default();
        assert!(t.mode_tint("object").is_none());
        let tints: Vec<_> = [
            "edit_mesh",
            "sculpt",
            "pose",
            "weight_paint",
            "texture_paint",
        ]
        .iter()
        .map(|m| t.mode_tint(m).expect("a tint"))
        .collect();
        // Each mode its own, and none of them the accent.
        for (i, a) in tints.iter().enumerate() {
            assert_ne!(a.1, t.accent);
            for b in &tints[i + 1..] {
                assert_ne!(a.1, b.1);
            }
        }
    }

    #[test]
    fn the_light_chrome_is_legible_and_its_ramp_descends() {
        let t = Theme::new(Appearance::Light, Density::Compact);
        assert!(!t.dark());
        let lum = |c: Color32| {
            let f = |v: u8| kreate_math::srgb_to_linear(f32::from(v) / 255.0);
            0.2126 * f(c.r()) + 0.7152 * f(c.g()) + 0.0722 * f(c.b())
        };
        let ratio = |a: Color32, b: Color32| {
            let (hi, lo) = (lum(a).max(lum(b)), lum(a).min(lum(b)));
            (hi + 0.05) / (lo + 0.05)
        };
        // Down from white: the field, the panel, the window.
        assert!(lum(t.raised) > lum(t.base) && lum(t.base) > lum(t.sunken));
        assert!(ratio(t.text, t.base) > 7.0);
        for surface in [t.base, t.raised] {
            assert!(ratio(t.text_faint, surface) > 4.5, "faint text");
        }
        assert!(ratio(t.text_faint, t.sunken) > 3.0);
        assert!(ratio(t.accent, t.raised) > 3.0, "the accent on white");
        for tag in [
            "edit_mesh",
            "sculpt",
            "pose",
            "weight_paint",
            "texture_paint",
        ] {
            let (_, text) = t.mode_tint(tag).unwrap();
            assert!(
                ratio(text, t.base) > 4.5,
                "{tag}'s name on the light chrome"
            );
        }
        let v = t.visuals();
        assert!(!v.dark_mode);
        assert_eq!(v.panel_fill, t.base);
        // And the default is still graphite.
        assert!(Theme::default().dark());
        assert_eq!(
            Theme::new(Appearance::Dark, Density::Compact).base,
            Theme::default().base
        );
    }

    #[test]
    fn an_appearance_is_stored_by_a_word_that_reads_back() {
        for a in Appearance::ALL {
            assert_eq!(Appearance::from_tag(a.tag()), Some(a));
        }
        assert_eq!(Appearance::from_tag("sepia"), None);
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
