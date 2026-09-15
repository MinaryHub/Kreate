// Which lights reach which square of the screen.
//
// One thread a tile, walking every light. That is the shape §2.2 asks for and
// it is the cheap half of the arrangement: the expensive half is the shading
// loop it replaces, which does the same walk *per fragment* — two hundred and
// fifty-six times for every one of a million pixels rather than once for every
// one of eight thousand tiles.
//
// No atomics and no shared memory. A thread owns its tile's whole list, writes
// it, and nothing else touches it; a version that split the lights across a
// workgroup would need an atomic append and would order the list differently
// every frame, which is a picture that shimmers when nothing moved.
//
// # The frustum is built in view space
//
// The light positions arrive eye-relative, so rotating them by the camera's
// inverse puts them in a space where a tile's planes are two multiplications
// and a sign. Building the planes in world space instead would mean four
// rotations a tile rather than one a light.

struct Light {
    position: vec4<f32>,
    direction: vec4<f32>,
    amount: vec4<f32>,
    cone: vec2<f32>,
    falloff: u32,
    flags: u32,
}

struct TileParams {
    /// Rotation from eye-relative world into view space, as a matrix.
    view: mat4x4<f32>,
    /// x, y: how many tiles across and down. z: how many lights. w: how many
    /// indices a tile can hold.
    counts: vec4<u32>,
    /// x: `1 / tan(fov_y / 2)`. y: the aspect, width over height. z: the near
    /// plane in metres. w: nonzero for an orthographic camera, whose tiles are
    /// a box rather than a wedge.
    lens: vec4<f32>,
    /// For an orthographic camera: the half-width and half-height of the whole
    /// view in metres. Unused otherwise.
    ortho: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: TileParams;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var<storage, read_write> tiles: array<u32>;

/// How far a light reaches, in metres, or a negative number for "everywhere".
///
/// The packed `amount.w` is one over the square of the range, which is what the
/// falloff's window multiplies by. Zero means the light was never given an end,
/// and a light with no end cannot be culled — it is in every tile, which is
/// correct and is what makes the range worth setting.
fn reach(light: Light) -> f32 {
    if light.amount.w <= 0.0 {
        return -1.0;
    }
    return inverseSqrt(light.amount.w);
}

// Parallel light is not special-cased here, and the temptation to is worth
// resisting: a sun does not diverge and so does not run out, but that decision
// is already made where the light is packed — `GpuLight::pack` refuses a range
// on a parallel light and writes zero. Testing for it again would be a second
// place holding the same rule, and a sweep that deleted the check found nothing
// to fail because the packing had already handled it.

/// Whether a sphere is on the inside of a plane through the origin, or close
/// enough to it to matter.
///
/// `normal` need not be unit length; it is normalised here, because the planes
/// are built from the lens and come out any length at all.
fn near_plane(normal: vec3<f32>, centre: vec3<f32>, radius: f32) -> bool {
    let n = normalize(normal);
    return dot(n, centre) >= -radius;
}

@compute @workgroup_size(8, 8, 1)
fn cull(@builtin(global_invocation_id) id: vec3<u32>) {
    let across = params.counts.x;
    let down = params.counts.y;
    if id.x >= across || id.y >= down {
        return;
    }
    let count = params.counts.z;
    let capacity = params.counts.w;
    let tile = id.y * across + id.x;
    let base = tile * (capacity + 1u);

    // The tile's bounds in normalised device coordinates. Y is flipped because
    // a tile row counts down the screen and clip space counts up.
    let x0 = f32(id.x) / f32(across) * 2.0 - 1.0;
    let x1 = f32(id.x + 1u) / f32(across) * 2.0 - 1.0;
    let y1 = 1.0 - f32(id.y) / f32(down) * 2.0;
    let y0 = 1.0 - f32(id.y + 1u) / f32(down) * 2.0;

    let f = params.lens.x;
    let aspect = params.lens.y;
    let near = params.lens.z;
    let orthographic = params.lens.w > 0.5;

    var written = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        let light = lights[i];
        let r = reach(light);
        var keep = true;
        if r >= 0.0 {
            // Into view space, where the camera looks down -Z.
            let p = (params.view * vec4<f32>(light.position.xyz, 1.0)).xyz;
            if orthographic {
                // A box: the tile is a rectangle extruded along the view, so
                // the test is the distance from the sphere's centre to it.
                let bx = vec2<f32>(x0, y0) * vec2<f32>(params.ortho.x, params.ortho.y);
                let by = vec2<f32>(x1, y1) * vec2<f32>(params.ortho.x, params.ortho.y);
                let lo = min(bx, by);
                let hi = max(bx, by);
                let d = max(max(lo - p.xy, vec2<f32>(0.0)), p.xy - hi);
                keep = dot(d, d) <= r * r && -p.z >= -r;
            } else {
                // Four side planes and the near plane. A wedge from the eye,
                // and a sphere is in it if it is no further outside any face
                // than its own radius.
                //
                // The planes come straight out of the projection: a point is
                // inside `ndc.x >= x0` exactly when `f/aspect * x + x0 * z >= 0`,
                // and the other three follow by symmetry.
                keep = near_plane(vec3<f32>(f / aspect, 0.0, x0), p, r)
                    && near_plane(vec3<f32>(-f / aspect, 0.0, -x1), p, r)
                    && near_plane(vec3<f32>(0.0, f, y0), p, r)
                    && near_plane(vec3<f32>(0.0, -f, -y1), p, r)
                    && (-p.z + r) >= near;
            }
        }
        if keep && written < capacity {
            tiles[base + 1u + written] = i;
            written = written + 1u;
        }
    }
    // The count last, so a reader that saw the old one reads old indices rather
    // than a new count over stale indices. Nothing reads while this runs, but
    // the ordering costs nothing and the alternative is a rule to remember.
    tiles[base] = written;
}
