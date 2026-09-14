//! What space a colour is in, said in the type.
//!
//! Three spaces are in play in a renderer, and two of them look identical: both
//! hold three floats that a shader would happily multiply. Telling them apart is
//! the whole content of this module.
//!
//! | | encoding | referred to | range |
//! |---|---|---|---|
//! | [`Srgb`] | sRGB transfer function | the display | 0..1 |
//! | [`DisplayRgb`] | linear | the display | 0..1 |
//! | [`SceneRgb`] | linear | the scene | 0..∞ |
//!
//! # Why this is a type and not a convention
//!
//! Because the convention was already tried and it failed twice. A selection
//! colour went into a shader still sRGB-encoded and came out pale yellow
//! instead of orange; a viewport was composited seven times too dark. Both were
//! a value crossing a boundary without anyone able to say which space it was
//! in — and both were found by looking at the screen, which is the expensive
//! way.
//!
//! §2.1 puts colour management before lighting, materials and rendering for the
//! same reason: all three deal in "how much light", and three systems that each
//! decide privately what a `[f32; 3]` means will disagree silently. This is the
//! agreement, and it is cheap now and unaddable later.
//!
//! # The two linear spaces are the interesting distinction
//!
//! [`DisplayRgb`] and [`SceneRgb`] have the same encoding and different
//! meanings. A `DisplayRgb` of 0.5 means "half the light this monitor can
//! emit"; a `SceneRgb` of 0.5 means "half the light of the scene's reference
//! white", and 20.0 means a bright sky. There is no general way from the second
//! to the first except a **display transform**, which is where exposure and
//! tone mapping live — and getting that wrong by treating one as the other is
//! exactly the 7× bug.
//!
//! The one narrow exception is named rather than convenient:
//! [`Srgb::as_reflectance`]. An authored base colour is a *ratio* of light
//! reflected, so decoding it lands in scene-linear directly with no exposure
//! involved. That is a real conversion and a common one, and it has a name so
//! that nobody reaches for it by accident.
//!
//! # Working space
//!
//! **Scene-linear with Rec.709 primaries.** One space shared by the viewport
//! and the renderer, in which a value is a physical amount of light with no
//! upper bound. Rec.709 rather than something wider because it is what glTF,
//! sRGB images and every consumer display already speak, and a wider gamut is a
//! decision that can be added behind [`SceneRgb`] without touching the callers
//! — which is what having the type at all buys.
//!
//! # Exposure belongs to the display transform, not to the lights
//!
//! Stated here because it is the trap §2.1 names. Turning a light up to make an
//! image brighter changes the *scene*; turning exposure up changes only how it
//! is shown. Once someone has compensated for a dark viewport by brightening
//! lights, the renderer and the viewport have permanently different scenes and
//! nobody can tell which is right.

/// A colour as it is authored and as it is displayed: sRGB-encoded, 0 to 1.
///
/// What a hex constant holds, what a PNG holds, what a colour picker returns,
/// and what egui wants. Not something to do arithmetic on — the encoding is
/// non-linear, so averaging two `Srgb` values is not the average of the two
/// colours. Decode first.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Srgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

/// Linear light, referred to the display: 0 to 1.
///
/// What a shader returns when it writes to an sRGB-format target — the hardware
/// applies the transfer function on the way out, so the value the shader hands
/// over is linear. Writing an `Srgb` value here is the mistake that made the
/// selection colour pale yellow.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DisplayRgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

/// Linear light, referred to the scene: 0 to infinity.
///
/// The working space. A light's colour times its intensity, a surface's
/// radiance, a pixel of an HDRI. One means the scene's reference white and
/// there is no maximum — a sun through a window is legitimately thousands.
///
/// Reaching the display from here needs a display transform, which is the only
/// thing that knows what exposure is set and how to fit an unbounded range into
/// a monitor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SceneRgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

/// One channel, sRGB-encoded to linear.
///
/// The piecewise form from the sRGB specification, not `powf(2.2)`. The two
/// differ by up to about 4% at the dark end, which is where banding lives and
/// where a UI's borders and shadows are.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// One channel, linear to sRGB-encoded. The inverse of [`srgb_to_linear`].
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

impl Srgb {
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
    };

    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// From `0xRR_GG_BB`, the form a palette constant is authored in.
    pub fn from_hex(hex: u32) -> Self {
        let byte = |shift: u32| ((hex >> shift) & 0xFF) as f32 / 255.0;
        Self {
            r: byte(16),
            g: byte(8),
            b: byte(0),
        }
    }

    /// From eight-bit channels, the form an image or a picker gives.
    pub fn from_bytes([r, g, b]: [u8; 3]) -> Self {
        Self {
            r: f32::from(r) / 255.0,
            g: f32::from(g) / 255.0,
            b: f32::from(b) / 255.0,
        }
    }

    /// Back to eight-bit channels, rounded rather than truncated.
    ///
    /// Truncation loses a step at every boundary, so a round trip through bytes
    /// would drift downwards — and a pixel test comparing a rendered colour
    /// against its authored value would be off by one for no reason anybody
    /// could find.
    pub fn to_bytes(self) -> [u8; 3] {
        let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        [byte(self.r), byte(self.g), byte(self.b)]
    }

    /// Removes the transfer function: what a shader should be given.
    pub fn decode(self) -> DisplayRgb {
        DisplayRgb {
            r: srgb_to_linear(self.r),
            g: srgb_to_linear(self.g),
            b: srgb_to_linear(self.b),
        }
    }

    /// Reads this colour as a *reflectance* and lands in the working space.
    ///
    /// The one path from an authored colour to scene-linear, and it is narrow on
    /// purpose. A base colour, an albedo, a vertex colour tinting a surface: all
    /// of those are ratios of light reflected, bounded by one, and decoding the
    /// transfer function is the whole conversion — no exposure, no reference
    /// white, nothing to decide.
    ///
    /// What this is *not* is a way to turn a swatch into an amount of light. A
    /// light's colour is a swatch and its brightness is a separate number in
    /// physical units; multiplying them is the caller's business and says so.
    pub fn as_reflectance(self) -> SceneRgb {
        let DisplayRgb { r, g, b } = self.decode();
        SceneRgb { r, g, b }
    }

    pub fn to_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }
}

impl DisplayRgb {
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };

    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Applies the transfer function: what to store in an eight-bit image or
    /// hand to a widget toolkit.
    pub fn encode(self) -> Srgb {
        Srgb {
            r: linear_to_srgb(self.r),
            g: linear_to_srgb(self.g),
            b: linear_to_srgb(self.b),
        }
    }

    pub fn to_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// With an alpha, for a uniform block.
    pub fn to_rgba(self, alpha: f32) -> [f32; 4] {
        [self.r, self.g, self.b, alpha]
    }
}

impl SceneRgb {
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };

    /// Back to the swatch a colour picker holds.
    ///
    /// The inverse of [`Srgb::as_reflectance`], and it exists for the same one
    /// reason: a panel that shows a stored reflectance has to hand the picker
    /// the numbers the picker works in. Doing it at the call site is where a
    /// widget and the value behind it start disagreeing about which space they
    /// are in — which is the bug §2.1 records twice.
    ///
    /// A reflectance is bounded by one, so nothing is lost on the way back.
    /// Feeding this an amount of *light* instead is the misuse `as_reflectance`
    /// warns about from the other side, and the clamp is what stops a lamp's
    /// thousand candela coming back as white.
    pub fn as_swatch(self) -> Srgb {
        DisplayRgb {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
        .encode()
    }
    /// The scene's reference white: the amount of light a display's white means.
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
    };

    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// A colour scaled by an amount of light.
    ///
    /// How a light's swatch and its intensity combine, and how a material's
    /// emission does. Spelled out rather than an operator, because the whole
    /// point of this type is that multiplying a scene value by a number is a
    /// physical statement and not a convenience.
    pub fn scaled(self, by: f32) -> Self {
        Self {
            r: self.r * by,
            g: self.g * by,
            b: self.b * by,
        }
    }

    pub fn to_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// Whether this is a value a display could show without a transform.
    ///
    /// A scene value inside 0..1 happens to be display-representable, which is
    /// what makes the 7× bug possible: for a dim scene the two spaces agree
    /// closely enough that a mix-up looks like a slightly wrong colour rather
    /// than a broken one, and it survives review.
    pub fn is_display_representable(self) -> bool {
        [self.r, self.g, self.b]
            .iter()
            .all(|c| (0.0..=1.0).contains(c))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_swatch_survives_the_trip_to_light_and_back() {
        // The two halves of one conversion, so a panel can show what was stored
        // and store what was shown without the value creeping each time it is
        // opened.
        for swatch in [
            Srgb::new(0.0, 0.0, 0.0),
            Srgb::new(1.0, 1.0, 1.0),
            Srgb::new(0.8, 0.35, 0.1),
            Srgb::new(0.002, 0.5, 0.999),
        ] {
            let back = swatch.as_reflectance().as_swatch();
            for (a, b) in swatch.to_array().iter().zip(back.to_array().iter()) {
                assert!((a - b).abs() < 1e-5, "{swatch:?} came back as {back:?}");
            }
        }
    }

    #[test]
    fn an_amount_of_light_is_not_a_swatch() {
        // Stated because the type does not stop it: a lamp's radiance is
        // unbounded and a swatch is not, so the way back has to clamp. What it
        // must not do is wrap, or come back black.
        let bright = SceneRgb::new(1200.0, 900.0, 40.0);
        let swatch = bright.as_swatch();
        for v in swatch.to_array() {
            assert!((0.0..=1.0).contains(&v), "{swatch:?}");
        }
        assert!(swatch.r > 0.99 && swatch.g > 0.99 && swatch.b > 0.99);
    }

    use super::*;

    #[test]
    fn the_transfer_function_round_trips() {
        for i in 0..=255u32 {
            let c = i as f32 / 255.0;
            let there_and_back = linear_to_srgb(srgb_to_linear(c));
            assert!(
                (there_and_back - c).abs() < 1e-5,
                "{c} came back as {there_and_back}"
            );
        }
    }

    #[test]
    fn the_piecewise_toe_is_used_at_the_dark_end() {
        // The specification's curve is linear below 0.04045, and a pure power
        // function is not. The difference is a few per cent — which is nothing
        // in a highlight and everything in the near-blacks a dark interface is
        // almost entirely made of.
        let mut worst = 0.0f32;
        for i in 0..=40u32 {
            let c = i as f32 / 1000.0;
            let spec = srgb_to_linear(c);
            let naive = c.powf(2.2);
            worst = worst.max((spec - naive).abs() / spec.max(1e-6));
        }
        assert!(
            worst > 0.2,
            "the piecewise toe differs from powf(2.2) by only {worst}; one of \
             them is not what it claims to be"
        );
    }

    #[test]
    fn hex_and_bytes_agree() {
        let from_hex = Srgb::from_hex(0xFF_9E_29);
        let from_bytes = Srgb::from_bytes([0xFF, 0x9E, 0x29]);
        assert_eq!(from_hex, from_bytes);
        assert_eq!(from_hex.to_bytes(), [0xFF, 0x9E, 0x29]);
    }

    #[test]
    fn bytes_round_trip_without_drifting_down() {
        // Truncation instead of rounding loses a step at every boundary, and a
        // pixel test comparing a rendered colour against its authored value
        // would be off by one with nothing to blame.
        for i in 0..=255u8 {
            let there_and_back = Srgb::from_bytes([i, i, i]).to_bytes();
            assert_eq!(there_and_back, [i, i, i], "{i} drifted");
        }
    }

    #[test]
    fn decoding_an_authored_colour_is_visibly_not_the_identity() {
        // The mistake this module exists to prevent, with its size pinned. The
        // accent's red channel is 255 and its green is 158: sRGB 158 is linear
        // 0.348, and handing 0.620 to a shader instead produces 208 on screen —
        // a pale yellow where an orange was declared.
        let accent = Srgb::from_hex(0xFF_9E_29);
        let linear = accent.decode();
        assert!(
            (accent.g - 0.620).abs() < 0.01,
            "the fixture's green channel moved"
        );
        assert!(
            (linear.g - 0.348).abs() < 0.01,
            "sRGB 158 should decode to about 0.348, got {}",
            linear.g
        );
        // And what skipping the decode would have shown.
        let mistake = DisplayRgb::new(accent.r, accent.g, accent.b).encode();
        assert!(
            mistake.to_bytes()[1] > 200,
            "the naive path should show about 208, showed {}",
            mistake.to_bytes()[1]
        );
    }

    #[test]
    fn a_reflectance_lands_in_the_working_space_unchanged_by_exposure() {
        // The narrow path from authored to scene-linear: a base colour is a
        // ratio, so the decode is the whole conversion. Equal to the display
        // decode channel for channel, which is the point — and why it needs a
        // name rather than being reachable by `as`.
        let swatch = Srgb::from_hex(0x80_40_20);
        let reflectance = swatch.as_reflectance();
        let display = swatch.decode();
        assert_eq!(reflectance.to_array(), display.to_array());
        assert!(
            reflectance.is_display_representable(),
            "a reflectance is bounded by one"
        );
    }

    #[test]
    fn a_scene_value_can_exceed_what_a_display_can_show() {
        // What the type is for. A sky is legitimately twenty times reference
        // white, and a space that could not hold that would force the value to
        // be squashed before anyone decided how.
        let sky = SceneRgb::new(0.6, 0.8, 1.0).scaled(20.0);
        assert!(!sky.is_display_representable());
        assert!((sky.b - 20.0).abs() < 1e-6);

        // And a dim scene value *is* representable, which is exactly what makes
        // confusing the two spaces survive review: it looks like a slightly
        // wrong colour rather than a broken one.
        assert!(SceneRgb::new(0.2, 0.2, 0.2).is_display_representable());
    }

    #[test]
    fn scaling_light_is_linear_where_scaling_an_encoded_value_is_not() {
        // Why arithmetic belongs in a linear space. Two lights of one unit make
        // two units; two sRGB halves do not make an sRGB one.
        let one = SceneRgb::WHITE;
        assert_eq!(one.scaled(2.0).to_array(), [2.0, 2.0, 2.0]);

        let half = Srgb::new(0.5, 0.5, 0.5);
        let doubled_encoded = Srgb::new(1.0, 1.0, 1.0);
        let doubled_properly = SceneRgb {
            r: half.as_reflectance().r * 2.0,
            g: half.as_reflectance().g * 2.0,
            b: half.as_reflectance().b * 2.0,
        };
        assert!(
            (doubled_properly.r - doubled_encoded.as_reflectance().r).abs() > 0.2,
            "doubling in the two spaces should disagree, or the encoding is \
             not doing anything"
        );
    }
}
