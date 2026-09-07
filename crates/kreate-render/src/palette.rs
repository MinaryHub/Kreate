//! The colour palette, in one place.
//!
//! Every colour the product shows — viewport and chrome alike — starts as an
//! sRGB hex constant here. It lives in the renderer rather than in the
//! interface crate because the renderer is the lower of the two in the
//! dependency graph and needs the same values: the accent that paints a
//! selected edge in the viewport is the accent that paints a selected row in
//! the outliner, and two constants that are *meant* to be equal will not stay
//! equal. The theme in `kreate-ui` builds its `Color32`s from these.
//!
//! The interesting part is [`to_linear`]. A shader writes to an sRGB-format
//! target, which means the value it returns is *linear* and the hardware
//! encodes it on the way out. Writing `0.62` for a channel whose sRGB value is
//! 158 therefore does not produce 158 on screen — it produces 208. That is not
//! a rounding difference: the selection colour was a pale yellow rather than
//! the orange it was declared to be, and the grid's X axis was pink. Every
//! colour crossing into a shader goes through [`to_linear`] for that reason,
//! and `the_naive_conversion_is_visibly_wrong` pins the size of the mistake so
//! it cannot be reintroduced by simplifying the maths away.

/// The selection colour, and the single accent of the whole product.
///
/// One hue for selection, for an engaged tool, and for an active mode. A
/// second accent would have to mean something, and nothing else needs saying
/// that badly.
pub const ACCENT: u32 = 0xFF_9E_29;

/// Informational highlight: the active object, and a removing rubber band.
pub const INFO: u32 = 0x78_AA_E6;

pub const AXIS_X: u32 = 0xD6_4A_54;
pub const AXIS_Y: u32 = 0x6A_BA_54;
pub const AXIS_Z: u32 = 0x4A_80_D6;

/// The viewport background — deliberately neutral, and deliberately not a
/// member of the warm surface ramp the chrome uses.
///
/// The chrome is warm so that hours in it are comfortable. The viewport must
/// not be: a warm surround makes a neutral grey read cool, and this is the one
/// screen where the user judges material colour. A tool that shifts the colour
/// it is asked to show is lying, so the warmth stops at the viewport edge.
pub const VIEWPORT: u32 = 0x1C_1C_1E;

/// The ordinary wireframe, on an unselected object.
///
/// Darker than the shaded surface on purpose: the wireframe is structure the
/// user reads *through*, and a light one competes with the shading it is drawn
/// over.
pub const WIRE: u32 = 0x3F_3F_44;

/// Unselected vertex dots, in edit mode.
///
/// Pale rather than dim: these are the handles, and a vertex you cannot see is
/// a vertex you cannot aim at.
pub const HANDLE: u32 = 0xCF_D2_D7;

/// The ground grid's lattice lines.
///
/// Read against [`VIEWPORT`] rather than in isolation: it has to survive a
/// dense lattice fading toward the horizon without becoming the brightest
/// thing on screen.
pub const GRID_LINE: u32 = 0x45_45_48;

/// Edge diagnostics. Loud on purpose — each one marks something the user
/// almost certainly wants to know about and cannot otherwise see.
pub const EDGE_NON_MANIFOLD: u32 = 0xF2_40_D8;
pub const EDGE_SEAM: u32 = 0xE6_73_26;
pub const EDGE_SHARP: u32 = 0x33_BF_F2;

/// The sRGB channels of a hex colour, as `0..=1`.
pub fn channels(hex: u32) -> [f32; 3] {
    [
        ((hex >> 16) & 0xFF) as f32 / 255.0,
        ((hex >> 8) & 0xFF) as f32 / 255.0,
        (hex & 0xFF) as f32 / 255.0,
    ]
}

/// The 8-bit sRGB components of a hex colour.
pub fn rgb8(hex: u32) -> [u8; 3] {
    [
        ((hex >> 16) & 0xFF) as u8,
        ((hex >> 8) & 0xFF) as u8,
        (hex & 0xFF) as u8,
    ]
}

/// One channel, sRGB to linear.
///
/// The piecewise IEC 61966-2-1 transfer function, not the 2.2 power
/// approximation: the approximation is wrong by enough at the dark end to
/// matter, and the dark end is where this palette lives.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// One channel, linear to sRGB. The inverse of [`srgb_to_linear`].
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// A hex colour as an opaque linear RGBA, ready for a shader or a uniform.
pub fn to_linear(hex: u32) -> [f32; 4] {
    let [r, g, b] = channels(hex);
    [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0]
}

/// A hex colour as a clear value.
///
/// `wgpu::Color` is linear for the same reason a shader's output is: the
/// attachment format does the encoding.
pub fn clear(hex: u32) -> wgpu::Color {
    let [r, g, b] = channels(hex);
    wgpu::Color {
        r: srgb_to_linear(r) as f64,
        g: srgb_to_linear(g) as f64,
        b: srgb_to_linear(b) as f64,
        a: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_splits_into_channels() {
        assert_eq!(rgb8(ACCENT), [0xFF, 0x9E, 0x29]);
        assert_eq!(rgb8(VIEWPORT), [0x1C, 0x1C, 0x1E]);
        let [r, g, b] = channels(0xFF_80_00);
        assert!((r - 1.0).abs() < 1e-6);
        assert!((g - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn the_transfer_function_round_trips() {
        // Every channel of every constant in this module, both ways. A
        // transfer function that is subtly wrong in one direction is the kind
        // of bug that shows up as "the colours are nearly right".
        for hex in [
            ACCENT,
            INFO,
            AXIS_X,
            AXIS_Y,
            AXIS_Z,
            VIEWPORT,
            WIRE,
            HANDLE,
            GRID_LINE,
            EDGE_NON_MANIFOLD,
            EDGE_SEAM,
            EDGE_SHARP,
        ] {
            for c in channels(hex) {
                let back = linear_to_srgb(srgb_to_linear(c));
                assert!(
                    (back - c).abs() < 1e-5,
                    "{hex:#08X}: {c} round-tripped to {back}"
                );
            }
        }
    }

    #[test]
    fn the_piecewise_toe_is_used_at_the_dark_end() {
        // Below the knee the function is linear, and a pure 2.2 power would
        // give a visibly different answer. This is the part the approximation
        // gets wrong, and this palette's darkest values sit right here.
        let c = 0.02;
        assert!((srgb_to_linear(c) - c / 12.92).abs() < 1e-9);
        assert!((srgb_to_linear(c) - c.powf(2.2)).abs() > 1e-4);
    }

    #[test]
    fn the_naive_conversion_is_visibly_wrong() {
        // The bug this module exists to prevent: passing sRGB channels to a
        // shader that writes an sRGB target. The selection accent was declared
        // #FF9E29 and drawn as (1.0, 0.62, 0.16), which the hardware then
        // encoded again — arriving on screen as a pale yellow.
        let [_, g, b, _] = to_linear(ACCENT);
        let naive = channels(ACCENT);

        // What the naive value would actually have displayed as, in 8-bit sRGB.
        let displayed = |c: f32| (linear_to_srgb(c) * 255.0).round() as i32;
        assert_eq!(displayed(naive[1]), 206, "declared 158");
        assert_eq!(displayed(naive[2]), 112, "declared 41");

        // And what the corrected value displays as: the colour as written.
        assert_eq!(displayed(g), 158);
        assert_eq!(displayed(b), 41);
    }

    #[test]
    fn a_clear_value_is_linear() {
        let c = clear(VIEWPORT);
        let [r, _, b] = channels(VIEWPORT);
        assert!((c.r - srgb_to_linear(r) as f64).abs() < 1e-9);
        assert!((c.b - srgb_to_linear(b) as f64).abs() < 1e-9);
        assert_eq!(c.a, 1.0);
        // Sanity: the viewport is dark, and this is the value the hardware
        // encodes back up to 0x1C.
        assert!(c.r < 0.02, "{}", c.r);
    }

    #[test]
    fn the_viewport_is_neutral_and_the_accent_is_not() {
        // The one rule of this palette that is not a matter of taste. The
        // viewport must not carry a colour cast, because the user judges
        // material colour against it.
        let [r, g, b] = rgb8(VIEWPORT).map(i32::from);
        assert!(
            (r - g).abs() <= 1 && (g - b).abs() <= 2 && (r - b).abs() <= 2,
            "viewport has a colour cast: {r},{g},{b}"
        );

        // Whereas the accent is meant to be unmistakable against it.
        let [ar, _, ab] = rgb8(ACCENT).map(i32::from);
        assert!(ar - ab > 128, "the accent is not clearly warm");
    }
}
