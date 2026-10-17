// The ground shadow: the model's shadow on the floor under it, as Fusion
// draws one in its default look — a soft dark patch that tells the eye where
// the part stands, under any lighting.
//
// A fullscreen triangle and a ray to the floor, as the grid is drawn. Where
// the ray lands is read in a depth map rendered straight down over the
// scene: whatever stands above that point darkens it, the more the nearer
// it is to the floor, and the farther above it is the softer the edge.
//
// The map looks *up* from under the floor, not down onto it: what it keeps
// is each column's lowest surface, the underside. Looking down it kept the
// top, and a cube standing on the floor read as hanging its own height
// above it — its shadow faded to nothing.

struct GroundGlobals {
    view_proj: mat4x4<f32>,
    inv_view_row0: vec4<f32>,
    inv_view_row1: vec4<f32>,
    inv_view_row2: vec4<f32>,
    // xyz unused; w: tan(fov_y / 2), or half the view's height orthographic.
    eye_and_tan: vec4<f32>,
    // x: aspect, y: 1 perspective / 0 orthographic, z: the floor's height,
    // eye-relative, w: how dark the darkest shadow is, 0 to 1.
    params: vec4<f32>,
    // Eye-relative to the down-looking map's clip space, reversed-Z.
    light_view_proj: mat4x4<f32>,
    // x: metres of height per unit of stored depth, y: metres per texel,
    // z: the map's layer, w: the height over which a shadow fades away.
    light: vec4<f32>,
    // x: how far the shadow spreads out from the footprint, in metres.
    extra: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: GroundGlobals;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    out.ndc = vec2<f32>(x, y);
    out.clip = vec4<f32>(x, y, 1.0, 1.0);
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

// How high above the floor the underside the map holds at `texel` is, in
// metres; negative where nothing is.
fn height_at(texel: vec2<i32>, size: vec2<i32>, floor_depth: f32) -> f32 {
    let at = clamp(texel, vec2<i32>(0), size - vec2<i32>(1));
    let d = textureLoad(shadow_map, at, i32(g.light.z), 0);
    // Reversed-Z, looking up from under the floor: the floor is nearest the
    // map, the largest depth, and higher is smaller. Nothing drawn is the
    // clear, zero. A `select` rather than an early return: Metal's
    // translation of a return beside the bounds-checked load named two
    // temporaries alike, and the pipeline failed to build.
    return select(max(floor_depth - d, 0.0) * g.light.x, -1.0, d <= 0.0);
}

// A point of a spiral filling the unit disc, turned by `spin`.
fn spiral(i: u32, count: u32, spin: f32) -> vec2<f32> {
    let r = sqrt((f32(i) + 0.5) / f32(count));
    let a = f32(i) * 2.3999632 + spin;
    return vec2<f32>(cos(a), sin(a)) * r;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var out: FsOut;
    let inv_view = mat3x3<f32>(
        vec3<f32>(g.inv_view_row0.x, g.inv_view_row1.x, g.inv_view_row2.x),
        vec3<f32>(g.inv_view_row0.y, g.inv_view_row1.y, g.inv_view_row2.y),
        vec3<f32>(g.inv_view_row0.z, g.inv_view_row1.z, g.inv_view_row2.z),
    );
    let tan_half = g.eye_and_tan.w;
    let aspect = g.params.x;
    let perspective = g.params.y > 0.5;
    var origin = vec3<f32>(0.0);
    var dir: vec3<f32>;
    if (perspective) {
        dir = normalize(inv_view * vec3<f32>(in.ndc.x * tan_half * aspect, in.ndc.y * tan_half, -1.0));
    } else {
        dir = normalize(inv_view * vec3<f32>(0.0, 0.0, -1.0));
        origin = inv_view * vec3<f32>(in.ndc.x * tan_half * aspect, in.ndc.y * tan_half, 0.0);
    }
    // Seen from under the floor there is no shadow on it to see.
    if (abs(dir.z) < 1e-7 || origin.z <= g.params.z) {
        discard;
    }
    let t = (g.params.z - origin.z) / dir.z;
    if (perspective && t <= 0.0) {
        discard;
    }
    let hit = origin + dir * t;

    // Where that is in the map.
    let lp = g.light_view_proj * vec4<f32>(hit, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        discard;
    }
    let size = vec2<i32>(textureDimensions(shadow_map));
    let centre = uv * vec2<f32>(size);
    let floor_depth = ndc.z;
    let per_texel = max(g.light.y, 1e-6);

    // How much of the floor round here lies under the model, the nearer
    // the middle the more it counts: a soft patch spreading out from the
    // footprint, as a broad light overhead casts. Each underside counts the
    // less the higher it hangs, and nothing at the fade height.
    //
    // The same spiral at every pixel rather than one turned per pixel: a
    // fixed pattern slides smoothly under a moving pixel, where a turned one
    // is noise, and the overlays are not averaged over frames to hide it.
    let radius = clamp(g.extra.x / per_texel, 2.0, 256.0);
    let fade = max(g.light.w, 1e-3);
    var shade = 0.0;
    var total = 0.0;
    for (var i = 0u; i < 48u; i = i + 1u) {
        let o = spiral(i, 48u, 0.0);
        let k = exp(-1.2 * dot(o, o));
        total += k;
        let h = height_at(vec2<i32>(centre + o * radius), size, floor_depth);
        if (h >= 0.0) {
            shade += k * (1.0 - smoothstep(0.0, fade, h));
        }
    }
    // Two thirds covered is as dark as it gets, so the patch under a part
    // is solid and only its rim is soft.
    shade = clamp(shade / total * 1.5, 0.0, 1.0);
    let alpha = shade * g.params.w;
    if (alpha <= 0.002) {
        discard;
    }

    let clip = g.view_proj * vec4<f32>(hit, 1.0);
    // A hair in front of the floor, where the grid sits a hair behind: a
    // floor the user modelled lying on it shows the shadow, rather than
    // the two trading pixels as the view's jitter moves. Reversed-Z, so
    // larger is nearer.
    out.depth = min(clip.z / clip.w * 1.0002, 1.0);
    if (out.depth < 0.0 || out.depth > 1.0) {
        discard;
    }
    out.color = vec4<f32>(0.0, 0.0, 0.0, alpha);
    return out;
}
