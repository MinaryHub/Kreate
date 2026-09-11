// The shadow pass: depth, and nothing else.
//
// Positions through the light's view-projection, and no fragment stage at all.
// A shadow map is a depth buffer, so a fragment shader would be arithmetic
// whose output is discarded — and on a tiler, a pipeline with no fragment stage
// is a pass the hardware can run at several times the rate.
//
// # The vertex buffer is positions only
//
// The main pass binds positions and normals; this one binds positions. The
// normal has no bearing on where a triangle is, and the second buffer would be
// bandwidth spent on nothing at the one point in the frame where geometry is
// drawn several times over.
//
// # No cull mode
//
// The same judgement `solid.wgsl` and the rendered pass make, for the same
// reason: a modeller has open and inconsistently wound geometry mid-edit.
// Culling front faces is the usual trick for hiding shadow acne, and it breaks
// on exactly the geometry a modeller has — an open surface has no back face to
// draw, so it stops casting a shadow. The bias handles the acne instead.

struct ShadowGlobals {
    /// Eye-relative world to the light's clip space, reversed-Z.
    view_proj: mat4x4<f32>,
}

struct ShadowObject {
    /// Eye-relative model matrix, as the main pass uses.
    model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> globals: ShadowGlobals;
@group(1) @binding(0) var<uniform> object: ShadowObject;

@vertex
fn vs_shadow(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return globals.view_proj * object.model * vec4<f32>(position, 1.0);
}
