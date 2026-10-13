// Temporal antialiasing: while the view moves, each frame is drawn nudged by a
// different fraction of a pixel and folded into the last picture, carried
// across to where this frame's camera sees it. The still view's mean (see
// `accum.wgsl`) does the same once nothing moves; this is that, in motion.
//
// Where a pixel was last frame comes from its depth: unprojected through this
// frame's camera, projected through the last one's. Only the camera's motion
// is followed — an object moving on its own is not — so the history is held
// to the colours this frame has around the pixel: a ghost of something that
// has moved away is a colour no neighbour has, and is cut off.

struct TemporalParams {
    // This frame's camera, unnudged and eye-relative, inverted.
    inv_current: mat4x4<f32>,
    // The last frame's camera, unnudged and relative to its own eye.
    previous: mat4x4<f32>,
    // How far the eye moved since: this eye minus the last.
    moved: vec4<f32>,
    // The region drawn, in pixels; this frame's nudge, in pixels (carried
    // for the record; see below for why it is not followed).
    region: vec2<f32>,
    jitter: vec2<f32>,
    // How much of this frame goes in; the rest is the history.
    weight: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var history_sampler: sampler;
@group(0) @binding(3) var depth: texture_depth_2d;
@group(0) @binding(4) var<uniform> tp: TemporalParams;

@vertex
fn temporal_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn temporal_resolve(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(position.xy);
    let last = vec2<i32>(tp.region) - 1;
    let now = textureLoad(frame, px, 0).rgb;
    // The colours this frame has around the pixel: what the history may be.
    var lo = now;
    var hi = now;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = textureLoad(frame, clamp(px + vec2<i32>(x, y), vec2<i32>(0), last), 0).rgb;
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }
    // Reversed-Z, infinitely far at zero: the sky is a point a long way off.
    let d = max(textureLoad(depth, px, 0), 1e-7);
    // The pixel's centre, not where this frame's nudge put its sample: the
    // history holds each pixel's samples from many nudges about its centre,
    // and following the nudge would slide the whole history by it every
    // frame — blurred, and drifting.
    let uv = position.xy / tp.region;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let here = tp.inv_current * vec4<f32>(ndc, d, 1.0);
    let point = here.xyz / here.w;
    let before = tp.previous * vec4<f32>(point + tp.moved.xyz, 1.0);
    if (before.w <= 0.0) {
        return vec4<f32>(now, 1.0);
    }
    let was = before.xy / before.w;
    let was_uv = vec2<f32>(was.x * 0.5 + 0.5, 0.5 - was.y * 0.5);
    if (any(was_uv < vec2<f32>(0.0)) || any(was_uv > vec2<f32>(1.0))) {
        return vec4<f32>(now, 1.0);
    }
    let old = textureSampleLevel(history, history_sampler, was_uv * tp.region / vec2<f32>(textureDimensions(history)), 0.0).rgb;
    let kept = clamp(old, lo, hi);
    return vec4<f32>(mix(kept, now, tp.weight), 1.0);
}
