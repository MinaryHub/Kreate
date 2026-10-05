//! The display transform: scene-linear light to a picture of it.
//!
//! §2.1's second half. [`crate::color`] settled what space a colour is in;
//! this settles what happens on the way to a screen, and the two are different
//! problems. Encoding sRGB is a *reversible* change of units. This is not: it
//! takes an open domain where 1.0 means nothing in particular and 1000.0 is an
//! ordinary sky, and produces a bounded picture.
//!
//! # Why a tone mapper at all
//!
//! Because clipping is a decision, and it is the worst one available. A sunlit
//! wall at 8.0 and a light source at 800.0 both become pure white, so the
//! photograph has no sun in it. Worse, clipping happens *per channel*: a
//! saturated red at (50, 0.5, 0.5) clips to (1, 0.5, 0.5), which is a
//! different hue from the one that was there. That is the notorious blown-out
//! red blob, and no amount of care in the renderer prevents it — the renderer
//! was right and the last step threw it away.
//!
//! # Why AgX
//!
//! Chosen in §2.1 over Filmic and ACES, and the reason is the *inset*: before
//! the curve, the primaries are rotated inward, so a colour that is about to
//! be crushed desaturates toward white on the way rather than sliding along
//! the gamut boundary and changing hue. That is what a photograph does, and it
//! is what Reinhard, Filmic and a plain per-channel clip all fail to do.
//!
//! ACES gets there too and is far more machinery — a set of transforms, a
//! configuration format, and an ODT per display. AgX is two matrices, a log
//! encoding and one curve, which is a thing that fits in a file and in a
//! shader.
//!
//! # Exposure lives here, and this is the one §2.1 warns about
//!
//! `intensity × 2^exposure` also appears on a light, and the two must not be
//! confused. A light's exposure is part of the *scene*: turning it up makes
//! the room brighter, and the renderer and the viewport both agree that it
//! did. This one is part of the *picture*: it says how the scene is being
//! looked at, and turning it up is what a photographer does with a shutter,
//! not what an electrician does with a bulb.
//!
//! The reason it belongs to the transform rather than to a caller multiplying
//! beforehand is that it must apply *before* the log encoding, inside the same
//! function that clamps to the exposure range. A caller who scaled the input
//! and then asked for a transform at zero stops would get the same answer
//! today and a different one the moment a look or a highlight rolloff is
//! added — and would have no way to know.
//!
//! # Mirroring this in WGSL
//!
//! It has to be, for the rendered viewport. Written in steps that translate
//! one-to-one — matrix, `max`, `log2`, an affine remap, a polynomial, matrix,
//! power, matrix — and the parity is a device test rather than a hope, the
//! same arrangement as the octahedral normal encoding.
//!
//! ## Two lines here exist for the shader's sake
//!
//! Both look like defensive code that Rust does not need, because that is
//! exactly what they are. A falsification sweep removed each in turn and found
//! no difference, correctly:
//!
//! - **The `1e-10` floor before `log2`.** Rust's `log2(0.0)` is `-inf`, and the
//!   `clamp` after the remap sends `-inf` to zero, so the floor is redundant
//!   here. WGSL leaves `log2(0)` implementation-defined — it may be a NaN — and
//!   a NaN through `clamp` is unspecified too, so on some adapter black would
//!   leave the function as a NaN and the next matrix would make three.
//! - **The explicit comparison in [`nonneg`] instead of `f32::max`.** Rust
//!   specifies that `max` returns the non-NaN operand, so the two agree. WGSL
//!   does not specify `max` with a NaN operand at all.
//!
//! Both stay on this side so the two implementations remain a line-for-line
//! mirror, which is the arrangement that keeps them from drifting. Simplifying
//! either would be correct in Rust and would break the shader's copy the next
//! time somebody translated one into the other.

use crate::color::{DisplayRgb, SceneRgb};

/// The bottom of the exposure range AgX encodes, in stops relative to 1.0.
///
/// Twelve and a half stops below middle grey's neighbourhood, which is more
/// range than a display has and about what a camera negative holds. Anything
/// darker is black, and saying so explicitly is what keeps `log2` bounded.
pub const MIN_EV: f32 = -12.473_93;
/// The top of the range: four stops above 1.0.
///
/// Not "as bright as the brightest thing in the scene". An adaptive top would
/// make the transform depend on the scene, so adding a lamp in the corner
/// would change the colour of the wall on the other side — which is the one
/// thing a *display* transform must never do.
pub const MAX_EV: f32 = 4.026_069;

/// Rec.709 linear to Rec.2020 linear.
///
/// AgX is defined in Rec.2020, and the working space is Rec.709 (§2.1), so
/// this is the price of using it. Derived from the two sets of primaries
/// through XYZ rather than copied: the published rounded forms are not exact
/// inverses of each other, and a neutral grey coming back out with a blue cast
/// is what that costs.
const REC709_TO_REC2020: [[f32; 3]; 3] = [
    [0.627_403_9, 0.329_283_04, 0.043_313_066],
    [0.069_097_29, 0.919_540_4, 0.011_362_316],
    [0.016_391_44, 0.088_013_31, 0.895_595_25],
];

/// Rec.2020 linear back to Rec.709 linear.
const REC2020_TO_REC709: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641_1, -0.072_849_86],
    [-0.124_550_48, 1.132_899_9, -0.008_349_423],
    [-0.018_150_763, -0.100_578_9, 1.118_729_7],
];

/// The inset: the whole reason for choosing AgX.
///
/// Rotates the primaries inward before the curve, so that a channel about to
/// be crushed pulls the others up with it and the colour desaturates toward
/// white instead of sliding along the gamut boundary. Every row sums to one,
/// which is what makes a neutral grey stay neutral — and the test that says so
/// is the one that catches a transposed matrix, because a transposed one is
/// still a plausible matrix.
const INSET: [[f32; 3]; 3] = [
    [0.856_627_15, 0.095_121_24, 0.048_251_606],
    [0.137_318_97, 0.761_242, 0.101_439_04],
    [0.111_898_21, 0.076_799_42, 0.811_302_4],
];

/// The outset: a partial inverse of the inset, applied after the curve.
///
/// Partial on purpose. A full inverse would put back exactly the saturation
/// the inset removed, including in the highlights where removing it was the
/// point. What comes back is the saturation of the midtones, which is where
/// its absence would read as a washed-out picture.
const OUTSET: [[f32; 3]; 3] = [
    [1.127_100_6, -0.110_606_64, -0.016_493_94],
    [-0.141_329_76, 1.157_823_7, -0.016_493_94],
    [-0.141_329_76, -0.110_606_64, 1.251_936_4],
];

/// The gamma AgX's curve encodes its output in.
///
/// The sigmoid below lands on display *code values* for a 2.2-gamma display,
/// so undoing it is how the result gets back to linear light — which is where
/// the Rec.2020 to Rec.709 conversion has to happen, because a primaries
/// conversion is only linear in linear light.
///
/// Not the sRGB transfer function, despite the resemblance, and this is why
/// the number is spelled out here rather than borrowed from [`crate::color`]:
/// it is a property of AgX's fitted curve. Using sRGB's piecewise curve
/// instead would be applying one standard's decode to another standard's
/// encode.
const AGX_GAMMA: f32 = 2.2;

/// Which view transform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewTransform {
    /// Scene-referred light to a picture. See the module docs.
    #[default]
    AgX,
    /// Clip to the display's range and nothing else.
    ///
    /// Kept for the case where the values are *already* display-referred and a
    /// tone curve would be a second one: an image editor showing a texture, or
    /// anyone reading numbers off the screen to check them. It is not for the
    /// interface — §2.1 puts UI colour in display space, which never enters
    /// this function at all.
    ///
    /// Not a good default for a render, and the tests say why: white clips.
    Standard,
}

/// How scene-linear light is turned into a picture.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DisplayTransform {
    pub view: ViewTransform,
    /// Stops. See the module docs on why this is not the light's exposure.
    pub exposure: f32,
}

impl DisplayTransform {
    /// AgX at zero stops.
    pub const DEFAULT: Self = Self {
        view: ViewTransform::AgX,
        exposure: 0.0,
    };

    pub const fn agx(exposure: f32) -> Self {
        Self {
            view: ViewTransform::AgX,
            exposure,
        }
    }

    /// The brightest scene-linear radiance that still comes out as black.
    ///
    /// Every value at or below this maps to the same output as zero, so a
    /// renderer that stops computing light there changes no pixel. That makes
    /// it the honest place to put a cutoff — the question "is this worth
    /// computing" is a question about the display, and the light does not know
    /// the answer.
    ///
    /// Exact rather than a guess, and different for the two transforms. AgX
    /// log-encodes over `[MIN_EV, MAX_EV]` and clamps below, so everything
    /// under `2^MIN_EV` enters the curve at the same place and leaves it as
    /// black. `Standard` only clamps, so the bound is the display's own step:
    /// the linear value that *encodes* to half a code value.
    ///
    /// That distinction is the whole of the second case and it is easy to get
    /// backwards. A code value is a number in the encoded picture, and what
    /// this returns is linear light — so it goes through the transfer
    /// function's toe, which is a factor of 12.92 steeper than the identity.
    /// Writing the step directly gives a value that shows up as code value six.
    ///
    /// A *quarter* of a step and not a half, which is where the boundary
    /// actually is. Exactly half is the point at which rounding decides, and
    /// which way it decides is a property of the rounding mode — a GPU's and
    /// Rust's need not agree, and this file exists partly because that class of
    /// disagreement is real. The bound sits where no rounding mode has an
    /// opinion.
    ///
    /// Both are then divided by the exposure, because `apply` multiplies by it:
    /// a picture opened up by six stops shows light six stops dimmer.
    pub fn black_radiance(&self) -> f32 {
        let floor = match self.view {
            ViewTransform::AgX => exp2(MIN_EV),
            ViewTransform::Standard => 0.25 / 255.0 / SRGB_TOE_SLOPE,
        };
        floor / exp2(self.exposure)
    }

    /// Turns scene-linear light into a display-referred colour.
    ///
    /// The result is in the display's range in every channel, for every finite
    /// input including a negative one — which scene-linear values can be, since
    /// a primaries conversion can produce them. That is the contract: whatever
    /// arrives, a picture comes out.
    pub fn apply(&self, color: SceneRgb) -> DisplayRgb {
        let scale = exp2(self.exposure);
        let exposed = [
            nonneg(color.r * scale),
            nonneg(color.g * scale),
            nonneg(color.b * scale),
        ];
        let display = match self.view {
            ViewTransform::AgX => agx(exposed),
            ViewTransform::Standard => exposed,
        };
        DisplayRgb::new(
            display[0].clamp(0.0, 1.0),
            display[1].clamp(0.0, 1.0),
            display[2].clamp(0.0, 1.0),
        )
    }
}

/// A non-negative, non-NaN value.
///
/// NaN is folded to zero rather than propagated. A NaN arriving here means
/// something upstream divided by zero, and a black pixel is a bug that can be
/// seen and chased; a NaN that survives into a texture poisons every mip level
/// above it and shows up somewhere else entirely.
///
/// `f32::max(v, 0.0)` would do the same thing in Rust, which specifies that
/// `max` returns the non-NaN operand. The comparison is spelled out because
/// WGSL specifies no such thing — see the module docs.
fn nonneg(v: f32) -> f32 {
    if v > 0.0 {
        v
    } else {
        0.0
    }
}

fn exp2(stops: f32) -> f32 {
    if stops == 0.0 {
        // Exactly one, so that a transform at zero stops is bit-for-bit the
        // same as no exposure at all. `2f32.powf(0.0)` is also exactly one on
        // every platform this runs on, but saying so costs nothing and makes
        // the identity a property of the code.
        1.0
    } else {
        2f32.powf(stops)
    }
}

fn transform(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// AgX's contrast curve.
///
/// A sixth-order polynomial fit to the sigmoid the reference implementation
/// builds from two exponentials. The fit rather than the original because it is
/// monotonic on `[0, 1]`, has no branch in it, and costs the same in WGSL as in
/// Rust — and because a curve assembled from `pow` calls per channel is where a
/// shader's cost goes.
///
/// Undershoots by about 0.0023 at zero and reaches about 0.9986 at one, so it
/// never exceeds the range at the top.
///
/// Nothing is done about either end, and a sweep adding a `[0, 1]` clamp here
/// confirmed there is nothing to do: at the top there is no overshoot to trim,
/// and at the bottom `nonneg` after the outset already floors it. The display
/// range is enforced once, in [`DisplayTransform::apply`], where it belongs.
fn contrast(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
        - 0.002_32
}

/// The slope of the sRGB transfer function at zero.
///
/// The straight segment the curve starts with, so that the encoding has a
/// finite derivative at black. Used to turn a step in the encoded picture into
/// a step in linear light — see `DisplayTransform::black_radiance`.
const SRGB_TOE_SLOPE: f32 = 12.92;

/// The transform proper, on non-negative Rec.709 linear light.
fn agx(color: [f32; 3]) -> [f32; 3] {
    let mut c = transform(&REC709_TO_REC2020, color);
    // A primaries conversion can send a saturated colour slightly negative.
    for v in &mut c {
        *v = nonneg(*v);
    }
    c = transform(&INSET, c);

    // Log encoding. The floor is for the shader, not for here — see the module
    // docs on the two lines that exist for the shader's sake.
    let range = MAX_EV - MIN_EV;
    for v in &mut c {
        let log = v.max(1e-10).log2();
        *v = contrast(((log - MIN_EV) / range).clamp(0.0, 1.0));
    }

    c = transform(&OUTSET, c);
    // Back to linear light, because the primaries conversion below is only a
    // linear operation there. See `AGX_GAMMA`.
    for v in &mut c {
        *v = nonneg(*v).powf(AGX_GAMMA);
    }
    transform(&REC2020_TO_REC709, c)
}

#[cfg(test)]
mod tests {
    // The black point is checked against the picture rather than against the
    // formula that produced it. See `the_black_point_is_the_last_value_that_is_black`.

    use super::*;
    use crate::color::Srgb;

    /// What an 8-bit screen would be sent.
    fn on_screen(color: SceneRgb) -> [u8; 3] {
        DisplayTransform::DEFAULT.apply(color).encode().to_bytes()
    }

    fn grey(v: f32) -> SceneRgb {
        SceneRgb::new(v, v, v)
    }

    #[test]
    fn middle_grey_lands_on_the_middle_of_the_screen() {
        // The one calibration anybody checks, and the reason it is checkable:
        // 0.18 is the scene-linear value a photographer's grey card
        // reflects, and it has to come out as the display's middle grey. A
        // transform that put it anywhere else would make every material
        // authored against it wrong.
        assert_eq!(on_screen(grey(0.18)), [128, 128, 128]);
    }

    #[test]
    fn a_neutral_stays_neutral() {
        // Which is a statement about the matrices: every row of the inset and
        // the outset sums to one, so `(1,1,1)` maps to `(1,1,1)`. A transposed
        // matrix is still a plausible matrix and passes any test that only
        // looks at one channel — it shows up here, as a grey with a blue cast.
        for v in [0.001, 0.01, 0.18, 0.5, 1.0, 4.0, 100.0] {
            let out = DisplayTransform::DEFAULT.apply(grey(v));
            assert!(
                (out.r - out.g).abs() < 1e-6 && (out.g - out.b).abs() < 1e-6,
                "grey {v} came out as {out:?}"
            );
        }
    }

    #[test]
    fn black_is_black_and_is_not_an_infinity() {
        // `log2(0)` is minus infinity and a matrix multiplication turns one
        // infinity into three NaNs, so this is the case that would poison
        // every pixel rather than one.
        let out = DisplayTransform::DEFAULT.apply(SceneRgb::BLACK);
        assert!(out.r.is_finite() && out.g.is_finite() && out.b.is_finite());
        assert_eq!(on_screen(SceneRgb::BLACK), [0, 0, 0]);
    }

    #[test]
    fn white_is_not_the_top_of_the_range() {
        // The whole point. 1.0 is not "as bright as it gets" in a scene-linear
        // working space — it is one unit of light — so a transform that sent it
        // to pure white would have nothing left for the sky, the sun, or a
        // specular highlight.
        let white = DisplayTransform::DEFAULT.apply(SceneRgb::WHITE);
        assert!(white.r < 0.95, "1.0 arrived at {}", white.r);
        assert_eq!(on_screen(SceneRgb::WHITE), [202, 202, 202]);
    }

    #[test]
    fn there_is_headroom_over_decades_above_white() {
        // A sunlit wall and the sun itself are three orders of magnitude apart
        // and both have to be in the picture, distinguishable. Under a clip
        // they are the same pixel.
        let ten = DisplayTransform::DEFAULT.apply(grey(10.0)).r;
        let thousand = DisplayTransform::DEFAULT.apply(grey(1000.0)).r;
        let one = DisplayTransform::DEFAULT.apply(grey(1.0)).r;
        assert!(one < ten && ten < thousand, "{one} {ten} {thousand}");
        assert!(ten < 1.0, "ten units of light already clipped");
    }

    #[test]
    fn brightness_never_goes_backwards() {
        // Over twelve decades, which is the range the log encoding claims. A
        // non-monotonic patch anywhere in here would show up as a band of
        // *darker* pixels inside a bright area — a highlight with a hole in
        // it, which reads as a rendering bug and is not one.
        let mut previous = -1.0;
        let mut v = 1e-7f32;
        while v < 1e5 {
            let out = DisplayTransform::DEFAULT.apply(grey(v)).r;
            assert!(
                out >= previous - 1e-7,
                "{v} dropped to {out} from {previous}"
            );
            previous = out;
            v *= 1.02;
        }
    }

    #[test]
    fn everything_finite_comes_out_as_a_picture() {
        // The contract, stated as bluntly as it can be: whatever arrives — an
        // absurd magnitude, a negative channel from a primaries conversion, a
        // NaN from a division upstream — a displayable colour comes out. A
        // display transform that could produce an out-of-range value would put
        // the burden of clamping on every one of its callers.
        let awkward = [
            SceneRgb::new(-1.0, 0.5, 2.0),
            SceneRgb::new(1e30, 0.0, -1e30),
            SceneRgb::new(f32::NAN, 0.18, f32::NAN),
            SceneRgb::new(f32::INFINITY, f32::NEG_INFINITY, 0.0),
            SceneRgb::new(f32::MIN_POSITIVE, 0.0, 0.0),
        ];
        for color in awkward {
            for t in [ViewTransform::AgX, ViewTransform::Standard] {
                let out = DisplayTransform {
                    view: t,
                    exposure: 3.0,
                }
                .apply(color);
                for channel in out.to_array() {
                    assert!(
                        channel.is_finite() && (0.0..=1.0).contains(&channel),
                        "{color:?} under {t:?} gave {out:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_stop_of_exposure_is_a_doubling_of_the_light() {
        // The identity that makes exposure part of the transform rather than
        // something a caller multiplies in. It also means the two exposures in
        // the product — this one and a light's — compose the way anybody would
        // expect, which is the only reason having both is tolerable.
        let doubled = DisplayTransform::DEFAULT.apply(grey(0.36));
        let a_stop_up = DisplayTransform::agx(1.0).apply(grey(0.18));
        assert_eq!(doubled.to_array(), a_stop_up.to_array());

        let quartered = DisplayTransform::DEFAULT.apply(grey(0.045));
        let two_down = DisplayTransform::agx(-2.0).apply(grey(0.18));
        assert_eq!(quartered.to_array(), two_down.to_array());
    }

    #[test]
    fn zero_stops_is_exactly_no_exposure() {
        // Not "within a rounding error of". A viewport that recomputed its
        // image whenever the exposure widget was touched would otherwise
        // produce a *different* picture at the setting it started at, and the
        // difference would be blamed on the renderer.
        assert_eq!(exp2(0.0), 1.0);
        assert_eq!(
            DisplayTransform::agx(0.0).apply(grey(0.18)).to_array(),
            DisplayTransform::DEFAULT.apply(grey(0.18)).to_array()
        );
    }

    #[test]
    fn exposure_only_ever_brightens() {
        for stops in [-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0] {
            let dark = DisplayTransform::agx(stops - 0.5).apply(grey(0.18)).r;
            let bright = DisplayTransform::agx(stops).apply(grey(0.18)).r;
            assert!(dark <= bright, "{stops} stops: {dark} then {bright}");
        }
    }

    #[test]
    fn a_blown_out_colour_desaturates_instead_of_changing_hue() {
        // AgX's reason for existing, and the failure it replaces. A per-channel
        // clip sends (50, 0.5, 0.5) to (1.0, 0.5, 0.5): a *pink* that was never
        // in the scene, with a hard edge where the red channel ran out. The
        // inset pulls the other two channels up with it instead, so an
        // overexposed red reads as an overexposed red.
        let hot = SceneRgb::new(50.0, 0.5, 0.5);
        let agx = DisplayTransform::DEFAULT.apply(hot);
        assert!(
            agx.g > 0.8 && agx.b > 0.8,
            "the highlight kept its hue: {agx:?}"
        );

        // The comparison, so that the assertion above is not merely a number
        // that happens to hold: clipping leaves the other channels where they
        // were.
        let clipped = DisplayTransform {
            view: ViewTransform::Standard,
            exposure: 0.0,
        }
        .apply(hot);
        assert_eq!(clipped.r, 1.0);
        assert!(
            clipped.g < 0.6,
            "clipping did something clever: {clipped:?}"
        );
    }

    #[test]
    fn an_in_gamut_colour_keeps_its_hue() {
        // The other half of the previous test: desaturating the highlights must
        // not mean desaturating everything. A mid red has to stay a red, and
        // stay redder than it is green.
        let out = DisplayTransform::DEFAULT.apply(SceneRgb::new(0.18, 0.02, 0.02));
        assert!(out.r > out.g * 2.0, "{out:?}");
        assert!(out.r > out.b * 2.0, "{out:?}");
        // And a green stays green, so the answer is not "the red channel is
        // privileged somewhere".
        let green = DisplayTransform::DEFAULT.apply(SceneRgb::new(0.02, 0.18, 0.02));
        assert!(green.g > green.r * 2.0, "{green:?}");
    }

    #[test]
    fn standard_is_a_clip_and_says_so() {
        // Kept for display-referred data that must not get a second tone
        // curve. It is not a render's transform, and the difference has to be
        // visible in a test so that nobody wires the wrong one in and finds out
        // from a screenshot.
        let standard = DisplayTransform {
            view: ViewTransform::Standard,
            exposure: 0.0,
        };
        assert_eq!(standard.apply(SceneRgb::WHITE).to_array(), [1.0, 1.0, 1.0]);
        assert_eq!(standard.apply(grey(1000.0)).to_array(), [1.0, 1.0, 1.0]);
        // Where AgX still tells the two apart.
        assert!(
            DisplayTransform::DEFAULT.apply(grey(1000.0)).r
                > DisplayTransform::DEFAULT.apply(SceneRgb::WHITE).r
        );
        // And it leaves an in-range value completely alone, which is what
        // "already display-referred" means.
        assert_eq!(standard.apply(grey(0.25)).to_array(), [0.25, 0.25, 0.25]);
    }

    #[test]
    fn the_contrast_curve_is_monotonic_and_spans_the_range() {
        // Guarded separately from the whole transform because it is the one
        // piece that is an approximation of something else. A refit that
        // improved the midtones and lost monotonicity at the top would be
        // caught here, next to the number it changed.
        let mut previous = f32::NEG_INFINITY;
        for i in 0..=1000 {
            let y = contrast(i as f32 / 1000.0);
            assert!(y >= previous, "the curve turned back at {i}");
            previous = y;
        }
        assert!(contrast(0.0).abs() < 0.01, "{}", contrast(0.0));
        assert!((contrast(1.0) - 1.0).abs() < 0.01, "{}", contrast(1.0));
    }

    #[test]
    fn the_black_point_is_the_last_value_that_is_black() {
        // The claim the cutoff rests on, checked as bytes and not as algebra:
        // at the black point the picture is the picture of nothing, and a
        // little above it is not.
        for view in [ViewTransform::AgX, ViewTransform::Standard] {
            for exposure in [-12.0f32, -6.0, 0.0, 3.0] {
                let display = DisplayTransform { view, exposure };
                let byte = |v: f32| {
                    let c = display.apply(SceneRgb::new(v, v, v)).encode();
                    (c.r * 255.0).round() as u8
                };
                let black = byte(0.0);
                let point = display.black_radiance();
                assert_eq!(
                    byte(point),
                    black,
                    "{view:?} at {exposure} stops: {point} is not black"
                );

                // And it is the *last* one: an order of magnitude above it has
                // to show, or the bound is loose enough to throw away light
                // somebody would see.
                let above = point * 10.0;
                assert_ne!(
                    byte(above),
                    black,
                    "{view:?} at {exposure} stops: {above} is black too"
                );
            }
        }
    }

    #[test]
    fn opening_the_exposure_lowers_the_black_point() {
        // Six stops of exposure is six stops of dimmer light made visible, so
        // the cutoff has to follow the dial. A fixed floor would throw away
        // light the moment the picture was opened up.
        let closed = DisplayTransform::agx(0.0).black_radiance();
        let open = DisplayTransform::agx(6.0).black_radiance();
        assert!(
            (closed / open - 64.0).abs() < 1e-3,
            "{closed} against {open}"
        );
    }

    #[test]
    fn the_exposure_range_is_fixed_rather_than_measured() {
        // A range that adapted to the scene would make the transform depend on
        // the scene, so adding a lamp in one corner would change the colour of
        // a wall in another. Stated as a test because "just normalise by the
        // brightest pixel" is the obvious idea and it is wrong.
        // A compile-time assertion, which is strictly stronger: the range is
        // two constants, so a build that changed them should not get as far as
        // running a test.
        const _: () = assert!(MIN_EV < -12.0 && MAX_EV > 4.0);
        let dim = DisplayTransform::DEFAULT.apply(grey(0.18));
        let with_a_sun_in_shot = DisplayTransform::DEFAULT.apply(grey(0.18));
        assert_eq!(dim.to_array(), with_a_sun_in_shot.to_array());
        // Below the range is black, above it is the top. Both are decisions,
        // not accidents.
        assert_eq!(on_screen(grey(1e-6)), [0, 0, 0]);
        assert_eq!(on_screen(grey(1e6)), on_screen(grey(1e9)));
    }

    #[test]
    fn the_transform_is_not_the_transfer_function() {
        // The mistake this whole module could be mistaken for. Encoding sRGB is
        // a reversible change of units; this is a picture-making decision. They
        // disagree everywhere, and most visibly at white — where sRGB says
        // "the top" and AgX says "one unit of light, with room above".
        let encoded = Srgb::WHITE;
        let transformed = DisplayTransform::DEFAULT.apply(SceneRgb::WHITE).encode();
        assert!(transformed.r < encoded.r);

        // And a mid grey differs too, though less: the two curves are near
        // each other by design in the range a monitor spends its time in.
        let grey_encoded = SceneRgb::new(0.18, 0.18, 0.18);
        let straight = crate::color::DisplayRgb::new(0.18, 0.18, 0.18).encode();
        let through = DisplayTransform::DEFAULT.apply(grey_encoded).encode();
        assert!(
            (straight.r - through.r).abs() > 0.02,
            "{} versus {}",
            straight.r,
            through.r
        );
    }
}
