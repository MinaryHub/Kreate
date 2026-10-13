// Denoising the path tracer's mean: an edge-avoiding à-trous wavelet filter
// (Dammertz et al.), guided by the noise as SVGF guides it. Five passes of a
// 5×5 B-spline kernel whose taps spread 1, 2, 4, 8, 16 pixels apart, each tap
// weighted down where the surface turns (normals), steps (depth) or the light
// differs by more than this pixel's own noise explains — so noise is averaged
// within a surface and edges, shadows among them, stay where they are.
//
// **The surface's colour is a guide, as the normals are.** A tap whose first
// surface has another colour counts for nothing, so a texture's squares are
// never averaged into each other: the noise is averaged within each. Dividing
// the colour out and filtering the light alone — the other way to keep a
// texture — was tried and lost: a dielectric's few per cent of untinted
// sheen divided by a saturated colour's dark channels is no smooth light, and
// a red and blue checkerboard came out a tenth dimmer. The colour and the
// noise come from the tracer's own sums (`pt_sum`, two vec4s a pixel).
//
// The last pass blends the filtered picture with the raw one by how few
// samples there are: a fresh mean is mostly filter, a mean of hundreds is
// itself.

struct Params {
    region: vec2<f32>,
    // How far apart this pass's taps are, in pixels.
    step: f32,
    // How much of the filtered picture the blend keeps.
    keep: f32,
    // How far apart two colours may be, relatively, and still be the same
    // surface's noise: the noise a mean of this many samples has. A real
    // edge in the light — a shadow across a face, which the normals and the
    // depth cannot see — is far past it.
    sigma: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var normals: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> params: Params;
// The blend's second picture: the filtered one, beside the raw `source`.
@group(1) @binding(0) var filtered: texture_2d<f32>;
// The tracer's sums.
@group(2) @binding(0) var<storage, read> pt_sum: array<vec4<f32>>;

fn slot(q: vec2<i32>) -> u32 {
    return (u32(q.y) * u32(params.region.x) + u32(q.x)) * 2u;
}

// The first surface's colour at `q`, averaged over its samples.
fn albedo(q: vec2<i32>) -> vec3<f32> {
    let i = slot(q);
    return pt_sum[i + 1u].rgb / max(pt_sum[i].w, 1.0);
}

// How many samples the mean at `q` has.
fn samples(q: vec2<i32>) -> f32 {
    return pt_sum[slot(q)].w;
}

// The variance of the mean of the picture's luminance at `q`: how far that
// mean is likely off from the truth.
fn variance(q: vec2<i32>) -> f32 {
    let i = slot(q);
    let n = max(pt_sum[i].w, 1.0);
    let mean = luminance(pt_sum[i].rgb) / n;
    return max(pt_sum[i + 1u].w / n - mean * mean, 0.0) / n;
}


@vertex
fn denoise_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

const KERNEL: array<f32, 3> = array<f32, 3>(0.375, 0.25, 0.0625);

@fragment
fn denoise_step(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(position.xy);
    let last = vec2<i32>(params.region) - 1;
    let here = textureLoad(source, px, 0).rgb;
    let d = textureLoad(depth, px, 0);
    // The background: nothing to keep apart, and nothing noisy.
    if (d <= 0.0) {
        return vec4<f32>(here, 1.0);
    }
    let n = textureLoad(normals, px, 0).xyz;
    let a = albedo(px);
    let l = luminance(here);
    // How much the light may differ between two taps and still be this
    // pixel's noise: four standard deviations of its mean, averaged over its
    // neighbours (one pixel's estimate is itself noisy) and shrinking as the
    // passes widen, since each pass leaves less noise for the next. Young
    // means — too few samples to measure noise from — fall back to the
    // relative guess.
    var spread = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            spread += variance(clamp(px + vec2<i32>(x, y), vec2<i32>(0), last));
        }
    }
    let measured = samples(px) >= 8.0;
    let sigma_l = 4.0 * sqrt(spread / 9.0) / sqrt(params.step);
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    let step = i32(params.step);
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let q = clamp(px + vec2<i32>(x, y) * step, vec2<i32>(0), last);
            let dq = textureLoad(depth, q, 0);
            if (dq <= 0.0) {
                continue;
            }
            let c = textureLoad(source, q, 0).rgb;
            let nq = textureLoad(normals, q, 0).xyz;
            // Reversed-Z: the depth is a reciprocal of distance, so their
            // ratio is how much nearer or farther the neighbour is.
            let w_depth = exp(-abs(d / dq - 1.0) / (0.02 * params.step));
            let w_normal = pow(max(dot(n, nq), 0.0), 32.0);
            let da = albedo(q) - a;
            let w_albedo = exp(-dot(da, da) / 0.005);
            var w_colour = 1.0;
            if (measured) {
                let apart = abs(l - luminance(c));
                w_colour = exp(-apart / (sigma_l + 1e-4 * (l + luminance(c)) + 1e-12));
            } else {
                let relative = abs(l - luminance(c)) / (0.5 * (l + luminance(c)) + 1e-4);
                w_colour = exp(-relative * relative / (2.0 * params.sigma * params.sigma));
            }
            let w = KERNEL[abs(x)] * KERNEL[abs(y)] * w_depth * w_normal * w_albedo * w_colour;
            sum += c * w;
            total += w;
        }
    }
    return vec4<f32>(sum / max(total, 1e-6), 1.0);
}

@fragment
fn denoise_blend(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(position.xy);
    let raw = textureLoad(source, px, 0).rgb;
    let calm = textureLoad(filtered, px, 0).rgb;
    return vec4<f32>(mix(raw, calm, params.keep), 1.0);
}
