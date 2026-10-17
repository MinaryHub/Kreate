// Ambient occlusion from the depth buffer: corners and creases darken, which
// is half of how a shape is read under even light.
//
// Two passes. The first estimates, per pixel, how much of the hemisphere over
// the surface is closed off by nearby geometry — sixteen samples, turned by a
// four-by-four pattern so neighbouring pixels sample different directions. The
// second averages exactly that four-by-four block, which cancels the pattern,
// and multiplies the result into the surfaces' image.
//
// Everything is in view space, reconstructed from depth through the inverse
// projection, so it holds for the infinite reversed-Z perspective and for an
// orthographic view alike. A depth of zero is the clear value — nothing drawn —
// and is left unoccluded.

struct Params {
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    // The region drawn into, in pixels.
    region: vec2<f32>,
    // How far the samples reach, in view units.
    radius: f32,
    // How dark full occlusion is, 0 to 1.
    strength: f32,
    // Which frame of a still view's average this is: the directions turn
    // with it, so the frames average the estimate's grain away.
    seed: f32,
};

@group(0) @binding(0) var depth: texture_depth_2d;
@group(0) @binding(1) var<uniform> params: Params;

// The apply pass binds only this, in a group of its own.
@group(1) @binding(0) var occlusion: texture_2d<f32>;

@vertex
fn ao_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn clamp_px(px: vec2<i32>) -> vec2<i32> {
    return clamp(px, vec2<i32>(0), vec2<i32>(params.region) - 1);
}

/// The view-space position under a pixel; `w` zero where nothing was drawn.
fn view_at(px: vec2<i32>) -> vec4<f32> {
    let p = clamp_px(px);
    let d = textureLoad(depth, p, 0);
    if (d <= 0.0) {
        return vec4<f32>(0.0);
    }
    let ndc = vec2<f32>(
        (f32(p.x) + 0.5) / params.region.x * 2.0 - 1.0,
        1.0 - (f32(p.y) + 0.5) / params.region.y * 2.0,
    );
    let h = params.inverse_projection * vec4<f32>(ndc, d, 1.0);
    return vec4<f32>(h.xyz / h.w, 1.0);
}

// Sixteen directions in the unit hemisphere about +Z, nearer the centre first
// so close geometry counts most.
const PI: f32 = 3.14159265;
const HALF_PI: f32 = 1.57079633;
const DIRECTIONS: i32 = 4;
const STEPS: i32 = 6;

/// The occlusion at a pixel, by its horizons: Jimenez et al.'s
/// ground-truth ambient occlusion. In each of a few directions across the
/// screen, how high the geometry within reach rises on either side; the
/// light the surface can still see between those two horizons, weighted by
/// the cosine to its normal, is its visibility. It measures what a
/// hemisphere of random samples guesses at, so a crease darkens as a
/// crease does and a flat floor not at all.
@fragment
fn ao_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(position.xy);
    let here = view_at(px);
    if (here.w == 0.0) {
        return vec4<f32>(1.0);
    }
    let p = here.xyz;

    // The surface normal from the neighbours, on whichever side is nearer in
    // depth — across a silhouette the far side belongs to something else.
    let r = view_at(px + vec2<i32>(1, 0));
    let l = view_at(px - vec2<i32>(1, 0));
    let u = view_at(px - vec2<i32>(0, 1));
    let d = view_at(px + vec2<i32>(0, 1));
    var dx = r.xyz - p;
    if (r.w == 0.0 || (l.w != 0.0 && abs(p.z - l.z) < abs(r.z - p.z))) {
        dx = p - l.xyz;
    }
    var dy = u.xyz - p;
    if (u.w == 0.0 || (d.w != 0.0 && abs(p.z - d.z) < abs(u.z - p.z))) {
        dy = p - d.xyz;
    }
    var n = normalize(cross(dx, dy));
    // Toward the eye: +Z for an orthographic view (no perspective divide —
    // the projection's w row is constant), whose eye point can be inside the
    // model when zoomed in; from the point otherwise.
    let ortho = params.projection[2][3] == 0.0;
    let v = select(normalize(-p), vec3<f32>(0.0, 0.0, 1.0), ortho);
    if (dot(n, v) < 0.0) {
        n = -n;
    }

    // The reach on screen, in pixels.
    let scale = params.projection[1][1] * 0.5 * params.region.y;
    let reach_px = select(scale * params.radius / max(-p.z, 1e-4), scale * params.radius, ortho);
    if (reach_px < 1.0) {
        return vec4<f32>(1.0);
    }
    let step_px = max(min(reach_px, 0.2 * params.region.y) / f32(STEPS), 1.0);
    // Turned by the pixel's place in a 4x4 block and by the frame; stepped
    // from an offset that moves the same way.
    let cell = f32((px.x & 3) + (px.y & 3) * 4);
    let turn = fract(cell * 0.0625 + params.seed * 0.618034);
    let offset = fract(52.9829189 * fract(0.06711056 * position.x + 0.00583715 * position.y)
        + params.seed * 0.381966);
    let reach2 = params.radius * params.radius;

    var visible = 0.0;
    for (var i = 0; i < DIRECTIONS; i++) {
        let phi = (f32(i) + turn) * PI / f32(DIRECTIONS);
        let dir = vec2<f32>(cos(phi), sin(phi));
        // On screen y runs down, in the view it runs up.
        let dir3 = vec3<f32>(dir.x, -dir.y, 0.0);
        let along = dir3 - dot(dir3, v) * v;
        let axis = normalize(cross(dir3, v));
        let projected = n - axis * dot(n, axis);
        let length_n = length(projected);
        if (length_n < 1e-4) {
            visible += 1.0;
            continue;
        }
        let cos_n = clamp(dot(projected / length_n, v), -1.0, 1.0);
        let gamma = sign(dot(along, projected)) * acos(cos_n);
        // The highest each side rises, as the cosine to the view.
        var back = -1.0;
        var ahead = -1.0;
        for (var k = 0; k < STEPS; k++) {
            let t = (f32(k) + offset) * step_px + 1.0;
            for (var side = 0; side < 2; side++) {
                let sgn = select(-1.0, 1.0, side == 1);
                let q = view_at(vec2<i32>(position.xy + dir * sgn * t));
                if (q.w == 0.0) {
                    continue;
                }
                let delta = q.xyz - p;
                let dist2 = dot(delta, delta);
                if (dist2 < 1e-10) {
                    continue;
                }
                let c = dot(delta * inverseSqrt(dist2), v);
                // Geometry past the reach shades nothing: a wall far behind
                // an object's edge is not its occluder.
                let near = clamp(1.0 - dist2 / reach2, 0.0, 1.0);
                let h = mix(-1.0, c, near);
                if (side == 1) {
                    ahead = max(ahead, h);
                } else {
                    back = max(back, h);
                }
            }
        }
        let h0 = gamma + max(-acos(back) - gamma, -HALF_PI);
        let h1 = gamma + min(acos(ahead) - gamma, HALF_PI);
        let sg = sin(gamma);
        let cg = cos(gamma);
        let a0 = 0.25 * (-cos(2.0 * h0 - gamma) + cg + 2.0 * h0 * sg);
        let a1 = 0.25 * (-cos(2.0 * h1 - gamma) + cg + 2.0 * h1 * sg);
        visible += length_n * (a0 + a1);
    }
    let seen = clamp(visible / f32(DIRECTIONS), 0.0, 1.0);
    return vec4<f32>(mix(1.0, seen, params.strength));
}

@fragment
fn ao_apply(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(position.xy);
    let last = vec2<i32>(textureDimensions(occlusion)) - 1;
    // Averaged over the 4x4 block the directions turn across — but only with
    // neighbours at about this depth, so the darkness of a crease does not
    // bleed across a silhouette onto what is far behind it.
    let centre = view_at(px);
    if (centre.w == 0.0) {
        return vec4<f32>(1.0);
    }
    let tolerance = 0.03 * abs(centre.z) + 1e-4;
    var sum = 0.0;
    var weights = 0.0;
    for (var y = -2; y < 2; y++) {
        for (var x = -2; x < 2; x++) {
            let at = clamp(px + vec2<i32>(x, y), vec2<i32>(0), last);
            let other = view_at(at);
            let w = select(0.0, exp(-abs(other.z - centre.z) / tolerance), other.w != 0.0);
            sum += textureLoad(occlusion, at, 0).r * w;
            weights += w;
        }
    }
    let ao = select(1.0, sum / weights, weights > 1e-4);
    return vec4<f32>(ao, ao, ao, 1.0);
}
