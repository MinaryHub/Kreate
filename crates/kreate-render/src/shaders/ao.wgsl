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
const KERNEL: array<vec3<f32>, 16> = array<vec3<f32>, 16>(
    vec3<f32>(0.0536, -0.0254, 0.0627),
    vec3<f32>(-0.0756, 0.0505, 0.0410),
    vec3<f32>(0.0251, 0.1063, 0.0592),
    vec3<f32>(-0.1201, -0.0811, 0.0715),
    vec3<f32>(0.1654, 0.0482, 0.0913),
    vec3<f32>(0.0103, -0.2101, 0.1050),
    vec3<f32>(-0.2010, 0.1302, 0.1420),
    vec3<f32>(0.2611, -0.1504, 0.1210),
    vec3<f32>(-0.0730, 0.3102, 0.1903),
    vec3<f32>(-0.3324, -0.1810, 0.1502),
    vec3<f32>(0.3912, 0.1813, 0.2310),
    vec3<f32>(0.0805, -0.4610, 0.2402),
    vec3<f32>(-0.4502, 0.3020, 0.2811),
    vec3<f32>(0.5310, -0.2205, 0.3420),
    vec3<f32>(-0.2402, 0.6110, 0.3603),
    vec3<f32>(0.4012, 0.5503, 0.5802),
);

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
    let to_eye = select(-p, vec3<f32>(0.0, 0.0, 1.0), ortho);
    if (dot(n, to_eye) < 0.0) {
        n = -n;
    }

    // A frame about the normal, turned by the pixel's place in a 4x4 block.
    let cell = (px.x & 3) + (px.y & 3) * 4;
    let angle = f32(cell) * 0.3926991 + 0.2;
    let spin = vec3<f32>(cos(angle), sin(angle), 0.0);
    let t = normalize(spin - n * dot(spin, n) + vec3<f32>(1e-4, 0.0, 0.0));
    let b = cross(n, t);

    let radius = params.radius;
    let bias = radius * 0.03;
    var occluded = 0.0;
    for (var i = 0; i < 16; i++) {
        let k = KERNEL[i];
        let s = p + (t * k.x + b * k.y + n * k.z) * radius;
        let clip = params.projection * vec4<f32>(s, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }
        let ndc = clip.xy / clip.w;
        let spx = vec2<i32>(
            i32((ndc.x * 0.5 + 0.5) * params.region.x),
            i32((0.5 - ndc.y * 0.5) * params.region.y),
        );
        let scene = view_at(spx);
        if (scene.w == 0.0) {
            continue;
        }
        // In front of the sample (nearer the eye, so a larger z) closes it
        // off — but only geometry within reach: a wall far behind an object's
        // edge is not shading it.
        let reach = smoothstep(0.0, 1.0, radius / max(abs(p.z - scene.z), 1e-6));
        occluded += select(0.0, 1.0, scene.z >= s.z + bias) * reach;
    }
    let ao = 1.0 - occluded / 16.0 * params.strength;
    return vec4<f32>(clamp(ao, 0.0, 1.0));
}

// The second pass: the 4x4 block average, multiplied into the image by the
// blend state (destination times source).
@fragment
fn ao_apply(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(position.xy);
    let last = vec2<i32>(textureDimensions(occlusion)) - 1;
    var sum = 0.0;
    for (var y = -2; y < 2; y++) {
        for (var x = -2; x < 2; x++) {
            sum += textureLoad(occlusion, clamp(px + vec2<i32>(x, y), vec2<i32>(0), last), 0).r;
        }
    }
    let ao = sum / 16.0;
    return vec4<f32>(ao, ao, ao, 1.0);
}
