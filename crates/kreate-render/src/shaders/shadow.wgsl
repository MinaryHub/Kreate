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
    /// x: how much light passes through the object (its transmission), so
    /// how many of its texels drop out. y: this frame's seed.
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: ShadowGlobals;
@group(1) @binding(0) var<uniform> object: ShadowObject;

@vertex
fn vs_shadow(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return globals.view_proj * object.model * vec4<f32>(position, 1.0);
}

// A see-through object's shadow: hashed, the way EEVEE shadows a transparent
// surface. Each texel of the map drops the object out with the probability
// that light passes through it, so the map holds it at a fraction of its
// texels; the shadow filter averages neighbouring texels into a partial
// shadow, and the still view's mean, whose frames each take a new seed,
// smooths what the filter leaves. Without it an object either cast a whole
// shadow or none, and a material at a tenth transmission lost all of it.
fn shadow_hash(p: vec2<f32>, seed: f32) -> f32 {
    let q = fract(p * vec2<f32>(0.1031, 0.1030) + vec2<f32>(seed * 0.6180339, seed * 0.4142135));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

@fragment
fn fs_hashed(@builtin(position) position: vec4<f32>) {
    if (shadow_hash(floor(position.xy), object.params.y) < object.params.x) {
        discard;
    }
}
