// The path tracer's running sum, divided out onto the linear picture: what the
// display transform then turns into an image, exactly as it does the
// rasteriser's.

struct ResolveParams {
    size: vec2<u32>,
    _pad: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> sum: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> resolve: ResolveParams;
// Per pixel, three vec4s; the third is the mean carried over from earlier
// frames and how many it stands for.
@group(0) @binding(2) var<storage, read> aux: array<vec4<f32>>;

@vertex
fn resolve_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn resolve_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<u32>(position.xy);
    if (p.x >= resolve.size.x || p.y >= resolve.size.y) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let pixel = p.y * resolve.size.x + p.x;
    let s = sum[pixel * 2u];
    // What was carried over counts for as many frames as it stands for, less
    // the pixel's own samples: it fades out as they come in, so a still view
    // still settles on its own mean.
    let c = aux[pixel * 3u + 2u];
    let k = max(c.w - s.w, 0.0);
    return vec4<f32>((s.rgb + c.rgb * k) / max(s.w + k, 1.0), 1.0);
}
