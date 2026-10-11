// Smoke and fire, marched through a grid.
//
// A fullscreen triangle; each pixel's ray, eye-relative, crosses the
// domain's box and stops at whatever the scene's depth says is in front.
// Along it the smoke absorbs and scatters light — lit from one direction,
// shadowed by a single look towards the light — and the flame glows. The
// result is premultiplied: what the volume adds, and how much of what is
// behind still shows.

struct Params {
    // Eye-relative clip to eye-relative world.
    inv_view_proj: mat4x4<f32>,
    // The box, eye-relative.
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    // The smoke's colour, and how brightly the flame glows.
    smoke_color: vec4<f32>,
    // Towards the light, unit.
    light: vec4<f32>,
    // x: step length, y: the most steps, z: how thick the smoke is per
    // metre at full density, w: 1 for a linear HDR picture, 0 for display.
    march: vec4<f32>,
    // The drawn region, in pixels.
    region: vec4<f32>,
    // x: how much finer-than-cell wisps carve the smoke, 0 to 1; y: their
    // frequency, per metre; z: seconds, which they drift up with.
    detail: vec4<f32>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var grid: texture_3d<f32>;
@group(0) @binding(2) var grid_sampler: sampler;
@group(0) @binding(3) var depth: texture_depth_2d;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

fn unproject(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let h = p.inv_view_proj * vec4<f32>(ndc, z, 1.0);
    return h.xyz / h.w;
}

fn sample_grid(at: vec3<f32>) -> vec2<f32> {
    let uvw = (at - p.box_min.xyz) / (p.box_max.xyz - p.box_min.xyz);
    return textureSampleLevel(grid, grid_sampler, uvw, 0.0).rg;
}

// Dark red to orange to a yellowish white, as a flame cools outward.
fn flame_color(t: f32) -> vec3<f32> {
    let ember = vec3<f32>(0.5, 0.04, 0.0);
    let orange = vec3<f32>(1.0, 0.42, 0.05);
    let white = vec3<f32>(1.0, 0.86, 0.55);
    return mix(mix(ember, orange, smoothstep(0.0, 0.45, t)), white, smoothstep(0.55, 1.0, t));
}

// A lattice value from 0 to 1.
fn lattice(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.3183099 + vec3<f32>(0.71, 0.113, 0.419));
    q = q * 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(lattice(i), lattice(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(lattice(i + vec3<f32>(0.0, 1.0, 0.0)), lattice(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(lattice(i + vec3<f32>(0.0, 0.0, 1.0)), lattice(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(lattice(i + vec3<f32>(0.0, 1.0, 1.0)), lattice(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y),
        u.z);
}

// Three octaves, 0 to 1 about a half.
fn wisps(p: vec3<f32>) -> f32 {
    return (value_noise(p) * 0.5 + value_noise(p * 2.03) * 0.25 + value_noise(p * 4.07) * 0.125) / 0.875;
}

// A small hash, so each pixel starts its march a little apart and the
// steps do not show as bands.
fn hash(v: vec2<f32>) -> f32 {
    return fract(sin(dot(v, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy / p.region.xy;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // Reversed Z: 1 is the near plane.
    let a = unproject(ndc, 1.0);
    let b = unproject(ndc, 0.5);
    let dir = normalize(b - a);

    let inv = 1.0 / dir;
    let t0 = (p.box_min.xyz - a) * inv;
    let t1 = (p.box_max.xyz - a) * inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    var t_in = max(max(max(lo.x, lo.y), lo.z), 0.0);
    var t_out = min(min(hi.x, hi.y), hi.z);

    // Whatever the scene has in front ends the ray. Zero is nothing drawn.
    let d = textureLoad(depth, vec2<i32>(frag.xy), 0);
    if (d > 0.0) {
        let surface = unproject(ndc, d);
        t_out = min(t_out, dot(surface - a, dir));
    }
    if (t_out <= t_in) {
        discard;
    }

    let step = p.march.x;
    let most = i32(p.march.y);
    let thick = p.march.z;
    let hdr = p.march.w > 0.5;
    let glow = select(1.0, p.smoke_color.w, hdr);
    let to_light = p.light.xyz;

    var t = t_in + step * hash(frag.xy);
    var transmit = 1.0;
    var color = vec3<f32>(0.0);
    for (var i = 0; i < most; i = i + 1) {
        if (t >= t_out || transmit < 0.01) {
            break;
        }
        let at = a + dir * t;
        let g = sample_grid(at);
        var smoke = g.r;
        // Finer than the cells: the smoke's thin edges eaten away by wisps,
        // its thick heart left — erosion, as clouds are drawn. A wisp only
        // scaling the smoke averages away along the ray; one cutting it to
        // nothing does not. In the domain's own metres, so they hold still
        // as the view moves, and rising with the smoke.
        if (p.detail.x > 0.0 && smoke > 0.002) {
            let q = (at - p.box_min.xyz) * p.detail.y - vec3<f32>(0.0, 0.0, p.detail.z * 0.6 * p.detail.y);
            let eaten = p.detail.x * 0.85 * (1.0 - wisps(q));
            smoke = clamp((smoke - eaten) / max(1.0 - eaten, 0.05), 0.0, 1.0) * (1.0 + p.detail.x);
        }
        let flame = g.g;
        if (smoke > 0.002 || flame > 0.002) {
            let sigma = smoke * thick;
            let absorbed = 1.0 - exp(-sigma * step);
            // One look towards the light for the smoke's own shadow.
            let toward = sample_grid(at + to_light * step * 4.0).r * thick * step * 4.0;
            let lit = 0.3 + 0.7 * exp(-toward);
            color += transmit * absorbed * p.smoke_color.rgb * lit;
            color += transmit * flame_color(flame) * flame * glow * step * 8.0;
            transmit *= 1.0 - absorbed;
        }
        t += step;
    }
    let alpha = 1.0 - transmit;
    if (!hdr) {
        color = min(color, vec3<f32>(1.0));
    }
    return vec4<f32>(color, alpha);
}
