// Markers: world-space lines and constant-size rings drawn over the picture,
// depth-tested against it.
//
// What lamps and cameras are drawn with. They have no surface, so the solid
// pass has nothing for them; the interface used to paint them over the frame,
// which put a lamp's circle in front of the wall it was behind. These are
// drawn inside the viewport's own pass, against its depth, so a lamp behind a
// wall is behind the wall — Blender's behaviour and the only one a modeller
// can navigate by.
//
// Two primitives, one uniform block. The line is the edge overlay's
// screen-space quad — same construction, same anti-aliasing — fed from a small
// per-frame buffer of segments rather than from a mesh. The ring is the vertex
// point sprite with a hole in it.

struct MarkerGlobals {
    view_proj: mat4x4<f32>,
    // xy: viewport size in pixels. z: relative depth offset toward the viewer.
    // w: unused.
    params: vec4<f32>,
};

struct Line {
    // xyz: camera-relative position of each end. a.w: 1 when dashed.
    a: vec4<f32>,
    b: vec4<f32>,
    color: vec4<f32>,
    // x: width in pixels. yzw: unused.
    style: vec4<f32>,
};

struct Point {
    // xyz: camera-relative position. w: unused.
    position: vec4<f32>,
    color: vec4<f32>,
    // x: outer radius in pixels. y: 0 for a ring, 1 for a square outline.
    // z: 1 to draw a dot in the centre. w: unused.
    style: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: MarkerGlobals;
@group(1) @binding(0) var<storage, read> lines: array<Line>;
@group(2) @binding(0) var<storage, read> points: array<Point>;

// Pull toward the viewer by a fraction of the fragment's own depth, so the
// marker wins a tie with the surface it sits on. Reversed-Z: larger is nearer.
fn toward_viewer(clip: vec4<f32>) -> vec4<f32> {
    var out = clip;
    out.z *= 1.0 + g.params.z;
    return out;
}

// --- lines -----------------------------------------------------------------

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(linear) across_px: f32,
    // Distance along the line in pixels, for the dash pattern.
    @location(1) @interpolate(linear) along_px: f32,
    @location(2) @interpolate(flat) instance: u32,
};

@vertex
fn vs_line(@builtin(vertex_index) vi: u32, @builtin(instance_index) li: u32) -> LineOut {
    var out: LineOut;
    out.instance = li;
    let l = lines[li];
    var a_clip = toward_viewer(g.view_proj * vec4<f32>(l.a.xyz, 1.0));
    var b_clip = toward_viewer(g.view_proj * vec4<f32>(l.b.xyz, 1.0));

    let viewport = g.params.xy;
    let half_width = l.style.x * 0.5;
    let expand = half_width + 1.0;

    let a_w = max(a_clip.w, 1e-6);
    let b_w = max(b_clip.w, 1e-6);
    let a_screen = a_clip.xy / a_w * 0.5 * viewport;
    let b_screen = b_clip.xy / b_w * 0.5 * viewport;

    var along = b_screen - a_screen;
    let length_px = length(along);
    along = select(vec2<f32>(1.0, 0.0), along / length_px, length_px > 1e-5);
    let across = vec2<f32>(-along.y, along.x);

    let at_end = vi >= 2u;
    let side = select(-1.0, 1.0, (vi & 1u) == 1u);
    var clip = select(a_clip, b_clip, at_end);
    let w = select(a_w, b_w, at_end);

    let offset = across * side * expand;
    clip.x += offset.x / (0.5 * viewport.x) * w;
    clip.y += offset.y / (0.5 * viewport.y) * w;

    out.clip = clip;
    out.across_px = side * expand;
    out.along_px = select(0.0, length_px, at_end);
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    let l = lines[in.instance];
    let half_width = l.style.x * 0.5;
    var coverage = clamp(half_width + 0.5 - abs(in.across_px), 0.0, 1.0);
    // A dash pattern in screen pixels, so a line to the ground reads as a
    // measurement whatever the distance. Six on, four off.
    if (l.a.w > 0.5) {
        let phase = fract(in.along_px / 10.0);
        if (phase > 0.6) {
            discard;
        }
    }
    if (coverage <= 0.002) {
        discard;
    }
    return vec4<f32>(l.color.rgb, l.color.a * coverage);
}

// --- rings -----------------------------------------------------------------

struct PointOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(linear) offset_px: vec2<f32>,
    @location(1) @interpolate(flat) instance: u32,
};

@vertex
fn vs_point(@builtin(vertex_index) vi: u32, @builtin(instance_index) pi: u32) -> PointOut {
    var out: PointOut;
    out.instance = pi;
    let p = points[pi];
    var clip = toward_viewer(g.view_proj * vec4<f32>(p.position.xyz, 1.0));
    let w = max(clip.w, 1e-6);
    let viewport = g.params.xy;
    let expand = p.style.x + 1.5;
    let corner = vec2<f32>(
        select(-1.0, 1.0, (vi & 1u) == 1u),
        select(-1.0, 1.0, vi >= 2u),
    );
    clip.x += corner.x * expand / (0.5 * viewport.x) * w;
    clip.y += corner.y * expand / (0.5 * viewport.y) * w;
    out.clip = clip;
    out.offset_px = corner * expand;
    return out;
}

@fragment
fn fs_point(in: PointOut) -> @location(0) vec4<f32> {
    let p = points[in.instance];
    let radius = p.style.x;
    let stroke = 1.2;
    // Distance from the outline: a circle's, or a square's.
    var d: f32;
    if (p.style.y > 0.5) {
        let q = abs(in.offset_px) - vec2<f32>(radius, radius);
        d = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0);
    } else {
        d = length(in.offset_px) - radius;
    }
    var coverage = clamp(stroke * 0.5 + 0.5 - abs(d), 0.0, 1.0);
    // The centre dot.
    if (p.style.z > 0.5) {
        let dot = clamp(1.6 + 0.5 - length(in.offset_px), 0.0, 1.0);
        coverage = max(coverage, dot);
    }
    if (coverage <= 0.002) {
        discard;
    }
    return vec4<f32>(p.color.rgb, p.color.a * coverage);
}
