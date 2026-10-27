// The environment, drawn where there is no geometry.
//
// A full-screen triangle whose fragment turns its own position back into a
// world direction and samples the cube. Group two is the shading data, which is
// already bound for the pass that follows — the inverse projection rides in the
// same uniform rather than in a buffer of its own, so there is one thing to
// keep in step with the camera instead of two.
//
// Depth is neither written nor tested. Reversed-Z clears the buffer to zero
// and the far plane *is* zero, so a background written there and compared
// `Greater` fails against the clear it was drawn onto — the whole frame comes
// back black, which is what the end-to-end test found. Nothing is lost:
// the background is drawn first, so there is nothing behind it.

struct Vertex {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

/// Three vertices covering the screen, with no vertex buffer.
///
/// A triangle rather than two triangles: a quad's diagonal makes the fragments
/// along it belong to two primitives, which is a measurable cost for a pass
/// that touches every pixel and buys nothing.
@vertex
fn background_vertex(@builtin(vertex_index) index: u32) -> Vertex {
    var out: Vertex;
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    out.ndc = vec2<f32>(x, y);
    // Reversed-Z: zero is the far plane, which is where a background belongs.
    //
    // Nothing reads it today — the pipeline neither writes nor tests depth (see
    // `gpu::depth_ignore_state`), and a sweep that put the triangle on the near
    // plane instead changed nothing. It says the right thing anyway, because
    // the day this pipeline takes a depth state is the day the difference stops
    // being invisible, and a value that was arbitrary then is one nobody would
    // think to look at.
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn background_fragment(in: Vertex) -> @location(0) vec4<f32> {
    // A studio's backdrop, where there is one: what is seen, while the
    // environment still lights.
    if shading.backdrop.w > 0.0 {
        return vec4<f32>(backdrop_at(in.ndc.y), 1.0);
    }
    if shading.world.w <= 0.0 {
        // A flat world. Written rather than left to the clear colour, because
        // the clear happens once and this pass runs over the same pixels — and
        // a background that is sometimes the clear and sometimes a draw is two
        // things to keep the same colour.
        return vec4<f32>(shading.world.rgb, 1.0);
    }
    // The near and far points of the ray through this pixel. Two unprojections
    // rather than one because a single point only gives a direction for a
    // perspective camera — an orthographic one has every ray parallel and its
    // own origin per pixel, and a background that only worked in perspective
    // would be wrong in exactly the view somebody checks a silhouette in.
    let near = shading.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let far = shading.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    var direction: vec3<f32>;
    if abs(far.w) < 1e-6 {
        // **The perspective case, and it is not an edge case.** This camera's
        // projection is infinite reversed-Z: `depth = near / -z`, which reaches
        // zero only at infinity. So the far plane in clip space is a point at
        // infinity, its unprojection has `w = 0`, and dividing by it gives a
        // NaN that `normalize` turns into the same colour for every pixel — a
        // sky painted flat across the frame, which is what the end-to-end test
        // found. A homogeneous point with `w = 0` *is* a direction.
        direction = normalize(far.xyz);
    } else {
        direction = normalize(far.xyz / far.w - near.xyz / near.w);
    }
    // Mip zero: a background is what the environment *is*, not what a rough
    // surface reflects of it.
    let colour = textureSampleLevel(
        environment_cube,
        environment_sampler,
        unrotate(direction),
        0.0,
    ).rgb;
    return vec4<f32>(colour * shading.environment.w, 1.0);
}
