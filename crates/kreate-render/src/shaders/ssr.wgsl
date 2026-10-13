// Screen-space reflections for the rendered view.
//
// The rendered pass writes, beside its light, three things per pixel: the
// normal and roughness; how much of the light arriving along the mirror
// direction the surface sends back (its specular share); and how much of that
// it already sent back from the environment. Here a ray leaves each pixel
// along the mirror direction and walks the depth buffer; where it meets
// something on screen, the environment's reflection is swapped for that
// thing's light. Swapped, not added: the picture already holds the sky in a
// metal floor, and adding the cube on top would make the floor twice as
// bright where the cube is.
//
// What the screen does not hold — behind the camera, off the edge, behind
// something — keeps the environment's reflection, faded in rather than cut.
// Rough surfaces keep it too: a blurred reflection wants a blur this pass
// does not do.

struct Params {
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    // World (eye-relative) to view: the camera's rotation.
    view: mat4x4<f32>,
    region: vec2<f32>,
    // How far a ray may go, in view units.
    reach: f32,
    // How thick a surface is taken to be behind its depth, in view units.
    thickness: f32,
};

@group(0) @binding(0) var depth: texture_depth_2d;
@group(0) @binding(1) var picture: texture_2d<f32>;
@group(0) @binding(2) var surface: texture_2d<f32>;
@group(0) @binding(3) var share: texture_2d<f32>;
@group(0) @binding(4) var environment: texture_2d<f32>;
@group(0) @binding(5) var<uniform> params: Params;

// The composite binds only this, in a group of its own.
@group(1) @binding(0) var reflections: texture_2d<f32>;

@vertex
fn ssr_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn view_at(px: vec2<i32>) -> vec4<f32> {
    let p = clamp(px, vec2<i32>(0), vec2<i32>(params.region) - 1);
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

fn to_pixel(p: vec3<f32>) -> vec3<f32> {
    let clip = params.projection * vec4<f32>(p, 1.0);
    if (clip.w <= 1e-6) {
        return vec3<f32>(-1.0, -1.0, 0.0);
    }
    let ndc = clip.xy / clip.w;
    return vec3<f32>((ndc.x * 0.5 + 0.5) * params.region.x, (0.5 - ndc.y * 0.5) * params.region.y, 1.0);
}

@fragment
fn ssr_trace(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(position.xy);
    let info = textureLoad(surface, px, 0);
    let roughness = info.a;
    // Nothing drawn here, or too rough for a sharp reflection to mean much.
    if (dot(info.xyz, info.xyz) < 0.5 || roughness > 0.6) {
        return vec4<f32>(0.0);
    }
    let here = view_at(px);
    if (here.w == 0.0) {
        return vec4<f32>(0.0);
    }
    let p = here.xyz;
    let n = normalize((params.view * vec4<f32>(info.xyz, 0.0)).xyz);
    // Away from the eye: -Z everywhere in an orthographic view.
    let v = select(normalize(p), vec3<f32>(0.0, 0.0, -1.0), params.projection[2][3] == 0.0);
    let r = reflect(v, n);

    // March with a per-pixel offset into the first step: the accumulation
    // averages the offsets into a smooth result instead of bands.
    const STEPS: i32 = 48;
    let step = params.reach / f32(STEPS);
    let noise = fract(52.9829189 * fract(0.06711056 * position.x + 0.00583715 * position.y));
    var t = step * (0.5 + noise);
    var hit = vec2<f32>(-1.0);
    var before = 0.0;
    for (var i = 0; i < STEPS; i++) {
        let q = p + r * t;
        let s = to_pixel(q);
        if (s.z == 0.0 || s.x < 0.0 || s.y < 0.0 || s.x >= params.region.x || s.y >= params.region.y) {
            break;
        }
        let scene = view_at(vec2<i32>(s.xy));
        // Behind what the screen shows there, and not far behind it.
        if (scene.w != 0.0 && q.z < scene.z && scene.z - q.z < params.thickness) {
            // A few halvings between the last step in front and this one.
            var lo = before;
            var hi = t;
            for (var k = 0; k < 5; k++) {
                let mid = 0.5 * (lo + hi);
                let m = p + r * mid;
                let ms = to_pixel(m);
                let mscene = view_at(vec2<i32>(ms.xy));
                if (mscene.w != 0.0 && m.z < mscene.z) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hit = to_pixel(p + r * hi).xy;
            t = hi;
            break;
        }
        before = t;
        t += step;
    }
    if (hit.x < 0.0) {
        return vec4<f32>(0.0);
    }
    // Faded towards the screen's edge, the ray's reach, and roughness; and a
    // ray coming back at the camera is the screen's least reliable guess.
    let edge = min(min(hit.x, params.region.x - hit.x), min(hit.y, params.region.y - hit.y));
    var confidence = clamp(edge / (0.08 * min(params.region.x, params.region.y)), 0.0, 1.0);
    confidence *= 1.0 - smoothstep(0.6, 1.0, t / params.reach);
    confidence *= 1.0 - smoothstep(0.3, 0.6, roughness);
    confidence *= 1.0 - smoothstep(0.2, 0.8, r.z);
    let seen = textureLoad(picture, vec2<i32>(hit), 0).rgb;
    let weight = textureLoad(share, px, 0).rgb;
    let sky = textureLoad(environment, px, 0).rgb;
    // The swap: what the surface reflects of the thing it sees, less what it
    // was reflecting of the sky — times how sure the ray is.
    return vec4<f32>((weight * seen - sky) * confidence, 0.0);
}

@fragment
fn ssr_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(textureLoad(reflections, vec2<i32>(position.xy), 0).rgb, 0.0);
}
