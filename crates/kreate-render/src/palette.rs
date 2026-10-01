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
//! The interesting part is which *space* a value is in, and that is carried in
//! the type: see [`kreate_math::color`]. A shader writes to an sRGB-format
//! target, which means the value it returns is [`DisplayRgb`] — linear — and
//! the hardware encodes it on the way out. Handing over the [`Srgb`] number
//! instead does not produce 158 on screen for a channel whose sRGB value is
//! 158; it produces 208. That is not a rounding difference: the selection
//! colour was a pale yellow rather than the orange it was declared to be, and
//! the grid's X axis was pink.
//!
//! Which is why the transfer function is not implemented here any more. It
//! lives in `kreate-math` beside the axis convention and the unit of length,
//! because it is the same kind of agreement, and two implementations of it is
//! one too many.

use kreate_math::{DisplayRgb, Srgb};

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

/// A palette entry as the colour it was authored as.
pub fn srgb(hex: u32) -> Srgb {
    Srgb::from_hex(hex)
}

/// The 8-bit sRGB components of a hex colour.
///
/// What a pixel test compares against, because a framebuffer holds encoded
/// bytes.
pub fn rgb8(hex: u32) -> [u8; 3] {
    Srgb::from_hex(hex).to_bytes()
}

/// A palette entry ready for a shader or a uniform.
///
/// The decode is the whole content of this function, and the return type is
/// what says so.
pub fn display(hex: u32) -> DisplayRgb {
    Srgb::from_hex(hex).decode()
}

/// The selection outline around selected objects — Blender's orange.
pub const OUTLINE_SELECTED: u32 = 0xED_57_00;
/// And around the active one, lighter.
pub const OUTLINE_ACTIVE: u32 = 0xFF_AA_40;

/// A palette entry as an opaque RGBA, for a uniform block.
pub fn display_rgba(hex: u32) -> [f32; 4] {
    display(hex).to_rgba(1.0)
}

/// A hex colour as a clear value.
///
/// `wgpu::Color` is linear for the same reason a shader's output is: the
/// attachment format does the encoding.
pub fn clear(hex: u32) -> wgpu::Color {
    let DisplayRgb { r, g, b } = display(hex);
    wgpu::Color {
        r: r as f64,
        g: g as f64,
        b: b as f64,
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
        let Srgb { r, g, b } = srgb(0xFF_80_00);
        assert!((r - 1.0).abs() < 1e-6);
        assert!((g - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(b, 0.0);
    }

    /// Every colour this module ships.
    const EVERY: &[u32] = &[
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
    ];

    #[test]
    fn every_constant_reaches_the_screen_as_the_colour_it_was_written_as() {
        // The bug this module exists to prevent, end to end and for every
        // entry. The accent was declared #FF9E29 and handed to a shader as
        // (1.0, 0.62, 0.16) — which the hardware then encoded a second time,
        // arriving as a pale yellow.
        //
        // `display` decodes and `encode` is what the attachment does, so this
        // walks the whole path a palette colour actually takes.
        for hex in EVERY {
            let authored = srgb(*hex);
            let on_screen = display(*hex).encode();
            assert_eq!(
                on_screen.to_bytes(),
                authored.to_bytes(),
                "{hex:#08X} arrives as {:?}, was written as {:?}",
                on_screen.to_bytes(),
                authored.to_bytes()
            );
        }
    }

    #[test]
    fn skipping_the_decode_would_have_been_visibly_wrong() {
        // The size of the mistake, pinned so nobody simplifies the decode away
        // on the grounds that the numbers look close. It is no longer
        // *representable* — `display` returns a `DisplayRgb` and a uniform
        // cannot be given an `Srgb` — but the arithmetic is worth keeping on
        // the record.
        let authored = srgb(ACCENT);
        let mistake = DisplayRgb::new(authored.r, authored.g, authored.b).encode();
        assert_eq!(mistake.to_bytes()[1], 206, "green was declared 158");
        assert_eq!(mistake.to_bytes()[2], 112, "blue was declared 41");
    }

    #[test]
    fn a_clear_value_is_linear() {
        let c = clear(VIEWPORT);
        let want = display(VIEWPORT);
        assert!((c.r - f64::from(want.r)).abs() < 1e-9);
        assert!((c.b - f64::from(want.b)).abs() < 1e-9);
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
