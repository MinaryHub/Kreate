// The path tracer's running sum, divided out onto the linear picture: what the
// display transform then turns into an image, exactly as it does the
// rasteriser's.

struct ResolveParams {
    size: vec2<u32>,
    _pad: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> sum: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> resolve: ResolveParams;

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
    let s = sum[p.y * resolve.size.x + p.x];
    return vec4<f32>(s.rgb / max(s.w, 1.0), 1.0);
}
