// The display transform, on the GPU.
//
// A line-for-line mirror of `kreate_math::agx`, and the two are held together
// by a device test rather than by discipline — the same arrangement as the
// octahedral normal encoding, and for the same reason: two implementations of
// one formula drift, and the drift shows up as "the viewport and the render
// disagree", which §5's second risk says is a bug report that recurs forever.
//
// Everything about *why* is in the Rust module. What is here is the translation
// and the two things that are specific to WGSL.
//
// # Why the matrices are written out as dot products
//
// WGSL's `mat3x3` is column-major and the published AgX constants are quoted
// column-major too, so `mat3x3<f32>(vec3(...), vec3(...), vec3(...))` happens
// to be right — and is right for a reason nobody reading it can check. A
// transposed colour matrix is still a plausible colour matrix: it produces a
// picture, with a cast. Written as explicit rows, the numbers here can be
// compared against the Rust ones by eye, which is what a mirror needs.
//
// # Why a resolve pass rather than a line in the fragment shader
//
// The display transform belongs at the *end*, once, on a linear image. Doing
// it per-fragment would apply it before blending, and blending two
// display-referred colours is not blending light — which is exactly the bug
// §2.1 records as "the viewport composited seven times too dark". A separate
// pass also gives the progressive accumulation of §2.5 the shape it needs:
// accumulate in linear, resolve for the screen.
//
// When the forward path does need it inline, this file's functions are
// concatenated into that shader's source. There is no `#import` in this build
// (no `naga_oil` in the render crate), and inventing one for a single function
// would be a build step to maintain.

const MIN_EV: f32 = -12.47393;
const MAX_EV: f32 = 4.026069;

// Rec.709 linear to Rec.2020 linear, by rows. See the Rust module: derived
// from the primaries rather than copied, because the published rounded forms
// are not exact inverses and a grey with a blue cast is what that costs.
const R709_TO_R2020_0 = vec3<f32>(0.6274039, 0.32928304, 0.043313066);
const R709_TO_R2020_1 = vec3<f32>(0.06909729, 0.9195404, 0.011362316);
const R709_TO_R2020_2 = vec3<f32>(0.016391439, 0.08801331, 0.89559525);

const R2020_TO_R709_0 = vec3<f32>(1.660491, -0.58764114, -0.072849864);
const R2020_TO_R709_1 = vec3<f32>(-0.124550475, 1.1328999, -0.008349423);
const R2020_TO_R709_2 = vec3<f32>(-0.018150763, -0.1005789, 1.1187297);

// The inset: the reason for choosing AgX. Rotates the primaries inward before
// the curve so a channel about to be crushed pulls the others up with it, and
// the colour desaturates toward white instead of changing hue. Every row sums
// to one, which is what keeps a neutral grey neutral.
const INSET_0 = vec3<f32>(0.85662715, 0.09512124, 0.048251606);
const INSET_1 = vec3<f32>(0.13731897, 0.761242, 0.10143904);
const INSET_2 = vec3<f32>(0.11189821, 0.07679942, 0.8113024);

// The outset: a *partial* inverse, so the midtones get their saturation back
// and the highlights do not.
const OUTSET_0 = vec3<f32>(1.1271006, -0.11060664, -0.01649394);
const OUTSET_1 = vec3<f32>(-0.14132976, 1.1578237, -0.01649394);
const OUTSET_2 = vec3<f32>(-0.14132976, -0.11060664, 1.2519364);

// Which view transform. Matches `kreate_math::ViewTransform`'s order, and
// that order is the wire format between the two implementations — a `u32`
// because a uniform block has no enums.
const VIEW_AGX: u32 = 0u;
const VIEW_STANDARD: u32 = 1u;

struct Display {
    /// Stops. The picture's exposure, not a light's — see the Rust module.
    exposure: f32,
    view: u32,
    /// How many colours to resolve. Checked per invocation, because a
    /// dispatch is rounded up to whole workgroups and the tail must not write
    /// past the buffer.
    count: u32,
    _pad: u32,
}

fn row3(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(r0, v), dot(r1, v), dot(r2, v));
}

/// A non-negative, non-NaN value.
///
/// `max(v, 0.0)` would *not* do this: with a NaN operand `max` is allowed to
/// return either one, so a NaN can survive it. The comparison is explicit for
/// that reason — a NaN reaching a texture poisons every mip above it and
/// surfaces somewhere else entirely, while a black pixel is a bug that can be
/// chased.
fn nonneg(v: f32) -> f32 {
    if v > 0.0 {
        return v;
    }
    return 0.0;
}

fn nonneg3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(nonneg(v.x), nonneg(v.y), nonneg(v.z));
}

/// AgX's contrast curve: a sixth-order fit, monotonic on [0, 1].
///
/// A polynomial rather than the reference implementation's pair of
/// exponentials because it is branchless, monotonic, and the same cost here as
/// in Rust. It undershoots by about 0.0023 at zero and reaches about 0.9986 at
/// one, so it never exceeds the range at the top; `nonneg` after the outset
/// floors the bottom, and the display range is enforced once at the end.
fn contrast(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x
        + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(color: vec3<f32>) -> vec3<f32> {
    var c = row3(R709_TO_R2020_0, R709_TO_R2020_1, R709_TO_R2020_2, color);
    // A primaries conversion can send a saturated colour slightly negative.
    c = nonneg3(c);
    c = row3(INSET_0, INSET_1, INSET_2, c);

    // Log encoding.
    //
    // The floor is load-bearing *here*, unlike on the CPU. WGSL leaves log2(0)
    // implementation-defined — it may be minus infinity or a NaN — and a NaN
    // through `clamp` is unspecified too, so on some adapter black would leave
    // this function as a NaN and the next matrix would make three of them.
    //
    // The Rust side keeps the same line even though its `clamp` already
    // handles the infinity, so that the two stay a line-for-line mirror.
    let range = MAX_EV - MIN_EV;
    c = vec3<f32>(
        log2(max(c.x, 1e-10)),
        log2(max(c.y, 1e-10)),
        log2(max(c.z, 1e-10)),
    );
    c = clamp((c - vec3<f32>(MIN_EV)) / vec3<f32>(range), vec3<f32>(0.0), vec3<f32>(1.0));
    c = vec3<f32>(contrast(c.x), contrast(c.y), contrast(c.z));

    c = row3(OUTSET_0, OUTSET_1, OUTSET_2, c);
    // Undoes AgX's own 2.2 display encoding, so the primaries conversion below
    // happens in linear light — where a primaries conversion is linear. This is
    // not the sRGB transfer function and must not be replaced by it: it is a
    // property of the curve above.
    c = pow(nonneg3(c), vec3<f32>(2.2)); // srgb-exempt: AgX's own encoding, not sRGB
    return row3(R2020_TO_R709_0, R2020_TO_R709_1, R2020_TO_R709_2, c);
}

/// Scene-linear light to a display-referred colour, in the display's range.
///
/// The contract the Rust side states and the device test checks: whatever
/// arrives — an absurd magnitude, a negative channel, a NaN from a division
/// upstream — a displayable colour comes out.
fn display_transform(color: vec3<f32>, params: Display) -> vec3<f32> {
    let exposed = nonneg3(color * exp2(params.exposure));
    var display = exposed;
    if params.view == VIEW_AGX {
        display = agx(exposed);
    }
    return clamp(display, vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- the resolve pass ------------------------------------------------------

@group(0) @binding(0) var<storage, read> scene: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> shown: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Display;

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }
    let source = scene[i];
    // Alpha is carried through untouched. It is a coverage fraction, not
    // light: putting it through a tone curve would make a half-covered edge
    // pixel a different shape of edge.
    shown[i] = vec4<f32>(display_transform(source.rgb, params), source.a);
}
