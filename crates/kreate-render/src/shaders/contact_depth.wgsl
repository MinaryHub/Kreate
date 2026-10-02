// The rendered view's depth, drawn before it is shaded: what contact shadows
// march through.
//
// The shading pass cannot read the depth it is writing, so the picture's depth
// is drawn once more into a texture of its own first. Positions only and no
// fragment stage — a depth buffer is all this makes, the same judgement the
// shadow maps' pass makes.

struct Globals {
    view_proj: mat4x4<f32>,
    params: vec4<f32>,
}

struct RenderedObject {
    model: mat4x4<f32>,
    normal_col0: vec4<f32>,
    normal_col1: vec4<f32>,
    normal_col2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: RenderedObject;

@vertex
fn contact_vertex(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return globals.view_proj * (object.model * vec4<f32>(position, 1.0));
}
