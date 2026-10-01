// The selection outline: Blender's orange line around what is selected.
//
// Two passes. The mask draws each selected object's visible surface — tested
// against the depth the picture already has, so a part behind something else
// is not outlined — writing how it is selected: a half for selected, one for
// the active object. The edge pass then paints, outside each shape, the
// pixels within reach of it, in the colour of the strongest neighbour.

struct Globals {
    view_proj: mat4x4<f32>,
};

struct ObjectData {
    model: mat4x4<f32>,
    normal_col0: vec4<f32>,
    normal_col1: vec4<f32>,
    normal_col2: vec4<f32>,
    base_color: vec4<f32>,
    // y: how this object is selected — 0.5 selected, 1 active, 0 not.
    paint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: ObjectData;

@vertex
fn mask_vertex(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return globals.view_proj * (object.model * vec4<f32>(position, 1.0));
}

@fragment
fn mask_fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(object.paint.y, 0.0, 0.0, 0.0);
}

struct Params {
    // Linear, as the target encodes on write.
    selected: vec4<f32>,
    active_colour: vec4<f32>,
    // x: how far out the line reaches, in pixels.
    width: vec4<f32>,
};

// The edge pass binds only these, in a group of their own.
@group(2) @binding(0) var mask: texture_2d<f32>;
@group(2) @binding(1) var<uniform> params: Params;

@vertex
fn edge_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn edge_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(position.xy);
    let last = vec2<i32>(textureDimensions(mask)) - 1;
    let here = textureLoad(mask, clamp(px, vec2<i32>(0), last), 0).r;
    // Inside a shape: nothing — the line is drawn around it, not over it.
    if (here > 0.25) {
        return vec4<f32>(0.0);
    }
    let reach = i32(params.width.x);
    var strongest = 0.0;
    var coverage = 0.0;
    for (var y = -reach; y <= reach; y++) {
        for (var x = -reach; x <= reach; x++) {
            let d = length(vec2<f32>(f32(x), f32(y)));
            if (d > params.width.x + 0.5) {
                continue;
            }
            let m = textureLoad(mask, clamp(px + vec2<i32>(x, y), vec2<i32>(0), last), 0).r;
            if (m > 0.25) {
                strongest = max(strongest, m);
                // Softer at the far edge of the line, so it is not a staircase.
                coverage = max(coverage, clamp(params.width.x + 0.5 - d, 0.0, 1.0));
            }
        }
    }
    if (strongest == 0.0) {
        return vec4<f32>(0.0);
    }
    let colour = select(params.selected, params.active_colour, strongest > 0.75);
    return vec4<f32>(colour.rgb, coverage);
}
