// Edge smoothing after the surfaces are drawn: FXAA, Lottes's.
//
// The surfaces are drawn into an image of their own; this pass reads it and
// writes the target, blending along the edges it finds and copying everything
// else. The overlays are drawn afterwards, onto the target — a wireframe and a
// vertex point smooth their own edges in their own shaders, and running them
// through this as well would soften the lines that have to stay crisp.
//
// Luma is taken on the square root of the linear value, near enough the
// display's encoding: an edge is judged by how different it *looks*, and in
// linear terms the dark half of the range is all crowded near zero.

struct Params {
    // One over the allocated size, in texels.
    inv_size: vec2<f32>,
    // The last texel centre of the region drawn into, as a UV: the allocation
    // is rounded up, and what lies beyond the region is last frame's.
    uv_max: vec2<f32>,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn fxaa_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(source, source_sampler, min(uv, params.uv_max), 0.0).rgb;
}

fn luma(c: vec3<f32>) -> f32 {
    return sqrt(max(dot(c, vec3<f32>(0.299, 0.587, 0.114)), 0.0));
}

// How far along an edge the blend may reach, in texels.
const SPAN_MAX: f32 = 8.0;
const REDUCE_MUL: f32 = 0.125;
const REDUCE_MIN: f32 = 0.0078125;

@fragment
fn fxaa_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let t = params.inv_size;
    let uv = position.xy * t;
    let middle = tap(uv);
    let m = luma(middle);
    let nw = luma(tap(uv + vec2<f32>(-1.0, -1.0) * t));
    let ne = luma(tap(uv + vec2<f32>(1.0, -1.0) * t));
    let sw = luma(tap(uv + vec2<f32>(-1.0, 1.0) * t));
    let se = luma(tap(uv + vec2<f32>(1.0, 1.0) * t));
    let lo = min(m, min(min(nw, ne), min(sw, se)));
    let hi = max(m, max(max(nw, ne), max(sw, se)));
    // No edge worth the name: the pixel as it was. Most of the picture, and
    // what keeps flat colour and fine texture from going soft.
    if (hi - lo < max(0.0312, hi * 0.125)) {
        return vec4<f32>(middle, 1.0);
    }

    // Across the gradient is along the edge.
    var direction = vec2<f32>(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));
    let reduce = max((nw + ne + sw + se) * 0.25 * REDUCE_MUL, REDUCE_MIN);
    let scale = 1.0 / (min(abs(direction.x), abs(direction.y)) + reduce);
    direction = clamp(direction * scale, vec2<f32>(-SPAN_MAX), vec2<f32>(SPAN_MAX)) * t;

    let near = 0.5 * (tap(uv + direction * (1.0 / 3.0 - 0.5)) + tap(uv + direction * (2.0 / 3.0 - 0.5)));
    let far = near * 0.5 + 0.25 * (tap(uv - direction * 0.5) + tap(uv + direction * 0.5));
    // The wider blend overshoots where the edge turns a corner; the narrow one
    // is kept there.
    let f = luma(far);
    if (f < lo || f > hi) {
        return vec4<f32>(near, 1.0);
    }
    return vec4<f32>(far, 1.0);
}
