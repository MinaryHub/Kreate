// An infinite ground grid, drawn analytically.
//
// A fullscreen triangle plus a ray-plane intersection, rather than a grid mesh.
// That means no extent to run out of, no LOD problem, no geometry at all — and
// crucially, correct depth, so the grid is occluded by the model instead of
// floating in front of or behind it.
//
// Line coverage is computed from `fwidth` of the world coordinate, which gives
// resolution-independent anti-aliasing for free and keeps distant lines fading
// out rather than aliasing into noise.

struct GridGlobals {
    view_proj: mat4x4<f32>,
    // Rows of the inverse view rotation, used to turn an NDC direction into a
    // world-space ray. Packed as vec4 for uniform alignment.
    inv_view_row0: vec4<f32>,
    inv_view_row1: vec4<f32>,
    inv_view_row2: vec4<f32>,
    // xyz: camera position in world space. w: tan(fov_y / 2) in perspective;
    // half the view's height, in metres, in orthographic.
    eye_and_tan: vec4<f32>,
    // x: aspect ratio, y: 1 for perspective / 0 for orthographic, zw: unused.
    params: vec4<f32>,
    // The lattice and the two visible axis lines, linear RGBA. Supplied by the
    // CPU from the shared palette so the grid's X red is the same red the
    // transform panel puts on its X field.
    line_color: vec4<f32>,
    axis_x_color: vec4<f32>,
    axis_y_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: GridGlobals;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

// One oversized triangle rather than two triangles: no shared edge to crack
// along, and one fewer vertex to interpolate.
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

// Coverage of a line lattice of the given spacing, anti-aliased by the screen
// space derivative of the coordinate.
//
// The derivative is the coordinate's, taken once by the caller and divided by
// the spacing here — not `fwidth(coord / spacing)`. Where the spacing steps a
// decade, the pixels of one 2x2 quad use different spacings, and the
// derivative of the quotient took in that step: enormous in one pixel row,
// which then read as lying on a line everywhere along it — a row of bright
// dashes across the floor at every distance the grid changed scale.
fn lattice(coord: vec2<f32>, spacing: f32, coord_width: vec2<f32>) -> f32 {
    let scaled = coord / spacing;
    let deriv = coord_width / spacing;
    // Distance to the nearest lattice line, in pixels.
    let dist = abs(fract(scaled - 0.5) - 0.5) / max(deriv, vec2<f32>(1e-8));
    let line = min(dist.x, dist.y);
    return 1.0 - min(line, 1.0);
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
    // Perspective: every ray from the eye, spreading. Orthographic: every ray
    // the same direction, each from its own place across the view — rays from
    // one point here drew a perspective floor under an orthographic model, its
    // lines converging where the model's do not, and the floor's depth wrong
    // enough to show through the model.
    var origin = vec3<f32>(0.0);
    var dir: vec3<f32>;
    if (perspective) {
        dir = normalize(inv_view * vec3<f32>(
            in.ndc.x * tan_half * aspect,
            in.ndc.y * tan_half,
            -1.0,
        ));
    } else {
        dir = normalize(inv_view * vec3<f32>(0.0, 0.0, -1.0));
        origin = inv_view * vec3<f32>(in.ndc.x * tan_half * aspect, in.ndc.y * tan_half, 0.0);
    }

    // The camera is at the origin of camera-relative space, so the world Z = 0
    // plane sits at -eye.z.
    let eye = g.eye_and_tan.xyz;
    let plane_z = -eye.z;
    if (abs(dir.z) < 1e-7) {
        discard;
    }
    let t = (plane_z - origin.z) / dir.z;
    if (perspective && t <= 0.0) {
        // The plane is behind the camera along this ray: looking up at the sky.
        discard;
    }

    let hit_rel = origin + dir * t;
    let world_xy = hit_rel.xy + eye.xy;

    // Two decades of lines, cross-faded by how dense they would be on screen, so
    // the grid keeps roughly constant visual density at any zoom instead of
    // dissolving into moire or collapsing to a single square.
    let pixel_span = length(fwidth(world_xy));
    let decade = log(max(pixel_span, 1e-8) * 12.0) / log(10.0);
    let level = floor(decade);
    let blend = decade - level;
    let minor_spacing = pow(10.0, level);
    let major_spacing = minor_spacing * 10.0;

    let width = fwidth(world_xy);
    let fine = lattice(world_xy, minor_spacing, width) * (1.0 - blend);
    let coarse = lattice(world_xy, major_spacing, width);
    var coverage = max(fine * 0.55, coarse * 0.85);

    // Fade out toward the horizon, where the lattice degenerates into noise.
    let horizon = 1.0 - smoothstep(0.0, 1.0, abs(t) * 0.5e-3 / max(abs(dir.z), 1e-3));
    coverage = coverage * clamp(horizon, 0.0, 1.0);

    var color = g.line_color.rgb;
    // Axis lines, which is what tells the user which way is which. Drawn from
    // the same lattice maths so they anti-alias identically.
    let axis_w = fwidth(world_xy) * 1.2;
    if (abs(world_xy.y) < axis_w.y) {
        color = g.axis_x_color.rgb; // +X, red
        coverage = max(coverage, 1.0 - abs(world_xy.y) / max(axis_w.y, 1e-8));
        coverage = coverage * clamp(horizon, 0.0, 1.0);
    } else if (abs(world_xy.x) < axis_w.x) {
        color = g.axis_y_color.rgb; // +Y, green
        coverage = max(coverage, 1.0 - abs(world_xy.x) / max(axis_w.x, 1e-8));
        coverage = coverage * clamp(horizon, 0.0, 1.0);
    }

    if (coverage <= 0.001) {
        discard;
    }

    // Real depth from the hit point, so geometry occludes the grid correctly.
    // Reversed-Z, so this is near 1 close to the camera.
    let clip = g.view_proj * vec4<f32>(hit_rel, 1.0);
    // A hair farther than it is — three ten-thousandths of the distance, a
    // few millimetres at a modelling distance — so a face lying in the
    // grid's own plane, the underside of a floor resting on it, wins the tie
    // every frame instead of flickering through it. Reversed-Z: smaller is
    // farther.
    out.depth = clip.z / clip.w * 0.9997;
    // Orthographic depth has ends: past them the floor is not in the view.
    if (out.depth < 0.0 || out.depth > 1.0) {
        discard;
    }
    out.color = vec4<f32>(color, coverage);
    return out;
}
