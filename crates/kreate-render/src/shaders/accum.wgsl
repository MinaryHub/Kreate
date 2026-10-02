// Progressive accumulation: while nothing moves, each frame is drawn with the
// camera nudged by a fraction of a pixel and averaged into what came before.
// Sixteen frames in, the edges, the shadows and the occlusion have been
// sampled sixteen ways and the picture is as smooth as it gets.
//
// One shader, two uses. `add` reads the frame just drawn and is blended into
// the running average by the pipeline (the source times 1/n, the destination
// times 1 - 1/n — the blend constant). `present` copies the average onto the
// target.

@group(0) @binding(0) var source: texture_2d<f32>;

@vertex
fn accum_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn accum_copy(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(textureLoad(source, vec2<i32>(position.xy), 0).rgb, 1.0);
}
