// Turning an equirectangular image into a prefiltered cube map.
//
// Two passes, both compute. The first projects the image onto the six faces;
// the second filters each mip for one roughness, so that a lookup at shading
// time is one sample rather than an integral. See `kreate_render::environment`
// for why the split-sum form is what the shading pass wants.

struct Params {
    /// x: the face size of the mip being written.
    /// y: the face size of the mip being read (prefilter only).
    /// z: the roughness this mip stands for.
    /// w: how many samples the prefilter takes.
    size: vec4<f32>,
    /// x: the base face size, for turning a sample's solid angle into a mip.
    /// y: how many mips the source view holds.
    ///
    /// Picking a mip is what keeps a rough level from sparkling: a hundred
    /// samples of a small bright sun is a scatter of bright texels, and no
    /// amount of blurring afterwards removes it. Reading a level whose texel
    /// already covers the sample's solid angle is the standard answer.
    source: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var equirect: texture_2d<f32>;
@group(0) @binding(2) var source_cube: texture_cube<f32>;
@group(0) @binding(3) var smooth_sampler: sampler;
@group(0) @binding(4) var written: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.14159265359;

/// The direction a cube texel stands for.
///
/// The mapping a cube sampler inverts, and it has to be *that* one rather than
/// any self-consistent one — a face written with the axes it prefers comes back
/// mirrored, and a mirrored sky looks like a sky. The order is
/// `+X -X +Y -Y +Z -Z`, and `s`/`t` run across and *down* the face.
fn cube_direction(face: u32, s: f32, t: f32) -> vec3<f32> {
    let a = 2.0 * s - 1.0;
    let b = 2.0 * t - 1.0;
    switch face {
        case 0u: { return normalize(vec3<f32>(1.0, -b, -a)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -b, a)); }
        case 2u: { return normalize(vec3<f32>(a, 1.0, b)); }
        case 3u: { return normalize(vec3<f32>(a, -1.0, -b)); }
        case 4u: { return normalize(vec3<f32>(a, -b, 1.0)); }
        default: { return normalize(vec3<f32>(-a, -b, -1.0)); }
    }
}

/// Where a direction lands in an equirectangular image.
///
/// Kreate is Z-up, so the top row of the image is straight up and the bottom
/// row straight down. The centre column faces `-Y`, which is the direction a
/// default camera looks — so an environment loaded and left alone is behind the
/// subject rather than behind the viewer.
fn equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let u = 0.5 + atan2(dir.x, -dir.y) / (2.0 * PI);
    let v = acos(clamp(dir.z, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

@compute @workgroup_size(8, 8, 1)
fn to_cube(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = u32(params.size.x);
    if id.x >= size || id.y >= size {
        return;
    }
    // The texel's centre, not its corner. Half a texel is the difference
    // between a cube whose faces meet and one with a seam down every edge.
    let s = (f32(id.x) + 0.5) / f32(size);
    let t = (f32(id.y) + 0.5) / f32(size);
    let dir = cube_direction(id.z, s, t);
    let colour = textureSampleLevel(equirect, smooth_sampler, equirect_uv(dir), 0.0);
    textureStore(written, vec2<i32>(i32(id.x), i32(id.y)), i32(id.z), colour);
}

/// Van der Corput radical inverse, for the Hammersley sequence.
fn radical_inverse(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

/// A half-vector drawn from the GGX distribution about `normal`.
///
/// The same importance sampling `brdf::albedo_table` uses, and deliberately the
/// same: the split sum is one integral factored into two, and the two halves
/// have to be sampling the same distribution or the product is not the integral
/// either of them came from.
fn importance_sample(xi: vec2<f32>, normal: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let h = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    var up = vec3<f32>(0.0, 0.0, 1.0);
    if abs(normal.z) > 0.999 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * h.x + bitangent * h.y + normal * h.z);
}

fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-8);
}

@compute @workgroup_size(8, 8, 1)
fn prefilter(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = u32(params.size.x);
    if id.x >= size || id.y >= size {
        return;
    }
    let s = (f32(id.x) + 0.5) / f32(size);
    let t = (f32(id.y) + 0.5) / f32(size);
    // The one approximation split-sum prefiltering makes, and it is worth
    // naming: the normal, the view and the reflection are taken to be the same
    // direction. That is exact head on and wrong at a grazing angle, where a
    // real reflection stretches along the surface — the streak an anisotropic
    // highlight has and this does not. Every real-time renderer makes it, and
    // the alternative is a table with a viewing angle in it.
    let normal = cube_direction(id.z, s, t);

    let roughness = params.size.z;
    let samples = u32(params.size.w);
    var total = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < samples; i = i + 1u) {
        let xi = vec2<f32>(f32(i) / f32(samples), radical_inverse(i));
        let h = importance_sample(xi, normal, roughness);
        let light = normalize(2.0 * dot(normal, h) * h - normal);
        let n_dot_l = dot(normal, light);
        if n_dot_l <= 0.0 {
            continue;
        }
        // Which mip to read. A few hundred samples of a bright, small sun is
        // noise that no amount of blurring afterwards removes, so the sample
        // reads a level whose texel covers roughly the solid angle the sample
        // stands for. `source` carries the base face size for that arithmetic.
        let n_dot_h = max(dot(normal, h), 0.0);
        let d = ggx_distribution(n_dot_h, roughness);
        let pdf = d * n_dot_h / (4.0 * max(dot(h, light), 1e-4)) + 1e-4;
        let texels = 4.0 * PI / (6.0 * params.source.x * params.source.x);
        let sample_solid_angle = 1.0 / (f32(samples) * pdf + 1e-4);
        var level = 0.0;
        if roughness > 0.0 {
            level = max(0.5 * log2(sample_solid_angle / texels), 0.0);
        }
        // Clamped to what has actually been written. A level past the end is
        // clamped by the hardware anyway on every back end anyone has tried,
        // and "anyone has tried" is not a specification — the same reasoning
        // that puts explicit bounds in `agx.wgsl`.
        level = min(level, params.source.y - 1.0);
        total += textureSampleLevel(source_cube, smooth_sampler, light, level).rgb * n_dot_l;
        weight += n_dot_l;
    }
    // Normalised by the accumulated weight rather than by the sample count.
    // That is what makes a constant environment come back as the constant at
    // every roughness — the identity the whole of §7.5's furnace argument rests
    // on, and the one a prefilter gets wrong by dividing by `samples`.
    let colour = total / max(weight, 1e-6);
    textureStore(written, vec2<i32>(i32(id.x), i32(id.y)), i32(id.z), vec4<f32>(colour, 1.0));
}
