// Bloom: light brighter than the display can show spills into its
// surroundings, as it does in a lens and an eye.
//
// On the linear picture, before the display transform. A prefilter keeps what
// is brighter than white once exposed (a soft knee, so there is no edge where
// it starts), a chain of half-size images blurs it wider at each step, the
// chain is added back up from the smallest, and the result is added to the
// picture. Each step is a small fixed filter; the width comes from the chain.

struct Params {
    // The region drawn into, and the allocation it sits in, in pixels.
    region: vec2<f32>,
    alloc: vec2<f32>,
    // x: the exposure in stops, y: how strong, z: the threshold, once
    // exposed (1 is white), w: the knee.
    settings: vec4<f32>,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) level: u32,
};

@vertex
fn bloom_vertex(@builtin(vertex_index) index: u32, @builtin(instance_index) level: u32) -> Out {
    var out: Out;
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    out.level = level;
    return out;
}

// The size of chain image `i`: half the region, halved again per level.
fn level_size(i: u32) -> vec2<f32> {
    return max(floor(params.region / exp2(f32(i + 1u))), vec2<f32>(1.0));
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(source, source_sampler, uv, 0.0).rgb;
}

// The picture into chain image 0, keeping only what is over the threshold.
@fragment
fn bloom_prefilter(in: Out) -> @location(0) vec4<f32> {
    let f = in.clip.xy / level_size(0u);
    // Into the picture's own coordinates: the region within the allocation.
    let scale = params.region / params.alloc;
    let texel = 1.0 / params.alloc;
    let uv = f * scale;
    let limit = (params.region - 0.5) / params.alloc;
    var c = vec3<f32>(0.0);
    for (var k = 0; k < 4; k++) {
        let o = vec2<f32>(f32(k & 1) - 0.5, f32(k >> 1u) - 0.5) * texel;
        c += tap(min(uv + o, limit));
    }
    c *= 0.25;
    // Exposed, thresholded with a soft knee, and back to scene units.
    let gain = exp2(params.settings.x);
    let bright = max(max(c.r, c.g), c.b) * gain;
    let t = params.settings.z;
    let knee = params.settings.w;
    var soft = clamp(bright - t + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);
    let keep = max(soft, bright - t) / max(bright, 1e-5);
    return vec4<f32>(c * keep, 1.0);
}

// Chain image `level` into `level + 1`: five taps, the centre weighted.
@fragment
fn bloom_down(in: Out) -> @location(0) vec4<f32> {
    let uv = in.clip.xy / level_size(in.level + 1u);
    let texel = 1.0 / level_size(in.level);
    var c = tap(uv) * 4.0;
    c += tap(uv + vec2<f32>(-1.0, -1.0) * texel);
    c += tap(uv + vec2<f32>(1.0, -1.0) * texel);
    c += tap(uv + vec2<f32>(-1.0, 1.0) * texel);
    c += tap(uv + vec2<f32>(1.0, 1.0) * texel);
    return vec4<f32>(c / 8.0, 1.0);
}

// Chain image `level + 1` added into `level`: a tent of eight taps.
@fragment
fn bloom_up(in: Out) -> @location(0) vec4<f32> {
    let uv = in.clip.xy / level_size(in.level);
    let texel = 0.5 / level_size(in.level + 1u);
    var c = vec3<f32>(0.0);
    c += tap(uv + vec2<f32>(-2.0, 0.0) * texel);
    c += tap(uv + vec2<f32>(2.0, 0.0) * texel);
    c += tap(uv + vec2<f32>(0.0, -2.0) * texel);
    c += tap(uv + vec2<f32>(0.0, 2.0) * texel);
    c += tap(uv + vec2<f32>(-1.0, -1.0) * texel) * 2.0;
    c += tap(uv + vec2<f32>(1.0, -1.0) * texel) * 2.0;
    c += tap(uv + vec2<f32>(-1.0, 1.0) * texel) * 2.0;
    c += tap(uv + vec2<f32>(1.0, 1.0) * texel) * 2.0;
    return vec4<f32>(c / 12.0, 1.0);
}

// Chain image 0 added onto the picture.
@fragment
fn bloom_composite(in: Out) -> @location(0) vec4<f32> {
    let uv = in.clip.xy / params.region;
    return vec4<f32>(tap(uv) * params.settings.y, 0.0);
}
