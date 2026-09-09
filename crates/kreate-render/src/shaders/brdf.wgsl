// The standard surface, on the GPU.
//
// A line-for-line mirror of `kreate_render::brdf`, held to it by a device test.
// §2.3's rule is one BRDF for the viewport and the renderer both, and §5's
// second risk is why it is a rule: two shading implementations drift, the user
// reports that the viewport and the render disagree, and that report recurs for
// as long as both exist.
//
// Every *reason* is in the Rust module. What is here is the translation, plus
// the three things that are specific to running on a device.
//
// # The DFG table arrives as a buffer, not a texture
//
// A storage buffer behaves the same on every capability tier (§v1 4.7).
// Storage textures need a format check and sampled textures need a sampler and
// a filtering mode that has to match what the CPU's bilinear does — and the
// point of the parity test is that the two agree exactly, so doing the
// interpolation in code rather than in fixed-function hardware is the whole
// idea.
//
// # One buffer holds both tables
//
// The 32x32 pairs first, row-major in roughness, then the 32 hemisphere
// averages. One binding rather than two, because the forward pass will already
// be carrying lights, materials and a camera, and a binding saved there is a
// binding not spent.
//
// # Why the energy terms are computed and not folded away
//
// It is tempting to hoist the compensation and the diffuse coupling out of the
// per-light loop, since they depend only on the surface and the view. That is
// correct and this file does not do it: the mirror has to stay readable against
// the Rust, and the forward pass can hoist them itself once there is one to
// measure.

const ALBEDO_LUT: u32 = 32u;
const DIELECTRIC_F0: f32 = 0.04;
const MIN_ROUGHNESS: f32 = 0.045;
const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;

/// The DFG table, then its per-roughness averages. See the header.
@group(0) @binding(0) var<storage, read> dfg: array<vec2<f32>>;

struct Surface {
    /// A reflectance, bounded by one. The diffuse albedo of a dielectric or the
    /// specular colour of a metal, and `metallic` says which.
    base_color: vec3<f32>,
    metallic: f32,
    /// Perceptual roughness. The distribution uses its square.
    roughness: f32,
}

struct Response {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

fn surface_f0(s: Surface) -> vec3<f32> {
    let m = clamp(s.metallic, 0.0, 1.0);
    return mix(vec3<f32>(DIELECTRIC_F0), s.base_color, m);
}

fn surface_diffuse_albedo(s: Surface) -> vec3<f32> {
    // Zero for a metal, and not as an approximation: a conductor absorbs what
    // enters it, so nothing scatters back out.
    return s.base_color * (1.0 - clamp(s.metallic, 0.0, 1.0));
}

fn surface_alpha(s: Surface) -> f32 {
    let r = clamp(s.roughness, MIN_ROUGHNESS, 1.0);
    return r * r;
}

/// The index a normalised coordinate lands on, matching the CPU's.
fn lut_index(t: f32) -> f32 {
    let last = f32(ALBEDO_LUT - 1u);
    return clamp(clamp(t, 0.0, 1.0) * f32(ALBEDO_LUT) - 0.5, 0.0, last);
}

/// `(a, b)` at `(NoV, roughness)`, interpolated the same way the CPU does.
fn dfg_lookup(nov: f32, roughness: f32) -> vec2<f32> {
    let x = lut_index(nov);
    let y = lut_index(roughness);
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, ALBEDO_LUT - 1u);
    let y1 = min(y0 + 1u, ALBEDO_LUT - 1u);
    let fx = x - floor(x);
    let fy = y - floor(y);

    let near = mix(dfg[y0 * ALBEDO_LUT + x0], dfg[y0 * ALBEDO_LUT + x1], fx);
    let far = mix(dfg[y1 * ALBEDO_LUT + x0], dfg[y1 * ALBEDO_LUT + x1], fx);
    return mix(near, far, fy);
}

/// The cosine-weighted hemisphere average of `(a, b)` at `roughness`.
fn dfg_average(roughness: f32) -> vec2<f32> {
    let base = ALBEDO_LUT * ALBEDO_LUT;
    let y = lut_index(roughness);
    let y0 = u32(floor(y));
    let y1 = min(y0 + 1u, ALBEDO_LUT - 1u);
    return mix(dfg[base + y0], dfg[base + y1], y - floor(y));
}

/// Trowbridge-Reitz.
fn distribution(noh: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = noh * noh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1.17549435e-38);
}

/// Smith's height-correlated visibility, with `1/(4·NoL·NoV)` folded in.
///
/// Height-correlated rather than separable: the separable form is not
/// reciprocal and is brighter at grazing angles by enough to show as a rim
/// that is not there.
fn visibility(nov: f32, nol: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let v = nol * sqrt(nov * nov * (1.0 - a2) + a2);
    let l = nov * sqrt(nol * nol * (1.0 - a2) + a2);
    return 0.5 / max(v + l, 1.17549435e-38);
}

/// Schlick, by five multiplies rather than a `pow`.
fn fresnel(f0: vec3<f32>, voh: f32) -> vec3<f32> {
    let m = clamp(1.0 - voh, 0.0, 1.0);
    let m2 = m * m;
    let f = m2 * m2 * m;
    return f0 + (vec3<f32>(1.0) - f0) * f;
}

/// Turquin's compensation: `1 + F0·(1/E − 1)`.
fn energy_compensation(f0: vec3<f32>, nov: f32, roughness: f32) -> vec3<f32> {
    let ab = dfg_lookup(nov, roughness);
    let e = clamp(ab.x + ab.y, 1e-2, 1.0);
    return vec3<f32>(1.0) + f0 * (1.0 / e - 1.0);
}

/// How much of the arriving light the specular lobe takes, per channel.
fn specular_share(f0: vec3<f32>, nov: f32, roughness: f32) -> vec3<f32> {
    let ab = dfg_lookup(nov, roughness);
    let e = clamp(ab.x + ab.y, 1e-2, 1.0);
    let single = f0 * ab.x + vec3<f32>(ab.y);
    let compensation = vec3<f32>(1.0) + f0 * (1.0 / e - 1.0);
    return clamp(single * compensation, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// The BRDF, per steradian, for one light direction.
///
/// No cosine term and no illuminance: those belong to the caller, which is what
/// makes the energy claims integrable and therefore checkable.
fn brdf_evaluate(s: Surface, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> Response {
    var out: Response;
    out.diffuse = vec3<f32>(0.0);
    out.specular = vec3<f32>(0.0);

    let nol_raw = dot(n, l);
    let nov_raw = dot(n, v);
    if nol_raw <= 0.0 || nov_raw <= 0.0 {
        return out;
    }
    // Clamped rather than trusted: an interpolated normal on a coarse mesh can
    // face slightly away from a viewer who can see the triangle, and the
    // grazing terms divide by these.
    let nol = min(nol_raw, 1.0);
    let nov = clamp(nov_raw, 1e-4, 1.0);

    // Plain `normalize`, and the early return above is what makes it safe:
    // `v + l` vanishes only when `l = -v`, which forces `n·l = -(n·v)`, so the
    // two cannot both be positive. A falsification sweep proved the guard that
    // used to be here was dead. This line therefore depends on that return —
    // relaxing it to shade a two-sided surface brings the NaN back.
    let h = normalize(v + l);
    let noh = clamp(dot(n, h), 0.0, 1.0);
    let voh = clamp(dot(v, h), 0.0, 1.0);

    let alpha = surface_alpha(s);
    let roughness = clamp(s.roughness, 0.0, 1.0);
    let f0 = surface_f0(s);
    let f = fresnel(f0, voh);
    let d = distribution(noh, alpha);
    let vis = visibility(nov, nol, alpha);

    out.specular = d * vis * f * energy_compensation(f0, nov, roughness);

    // Kelemen–Szirmay-Kalos: symmetric in the two directions, so the BRDF stays
    // reciprocal, and its hemispherical integral is exactly the energy the
    // specular lobe left.
    let share_v = specular_share(f0, nov, roughness);
    let share_l = specular_share(f0, nol, roughness);
    let avg_ab = dfg_average(roughness);
    let avg = clamp(f0 * avg_ab.x + vec3<f32>(avg_ab.y), vec3<f32>(0.0), vec3<f32>(0.99));
    out.diffuse = surface_diffuse_albedo(s) * INV_PI
        * (vec3<f32>(1.0) - share_v) * (vec3<f32>(1.0) - share_l)
        / (vec3<f32>(1.0) - avg);

    return out;
}

/// What a fragment adds for one light.
///
/// `illuminance` is what `lights.wgsl`'s `illuminance_at` answers — lux
/// arriving perpendicular to the light — so the `NoL` here is the projection
/// onto the surface and nothing else applies it.
fn brdf_shade(
    s: Surface,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    illuminance: vec3<f32>,
    diffuse_on: bool,
    specular_on: bool,
) -> vec3<f32> {
    let nol = max(dot(n, l), 0.0);
    if nol <= 0.0 {
        return vec3<f32>(0.0);
    }
    let r = brdf_evaluate(s, n, v, l);
    var f = vec3<f32>(0.0);
    if diffuse_on {
        f += r.diffuse;
    }
    if specular_on {
        f += r.specular;
    }
    return f * illuminance * nol;
}

// --- the probe pass --------------------------------------------------------
//
// For the device test: it evaluates the BRDF for a list of (surface, geometry)
// samples and writes both lobes, so the two implementations can be compared
// without a light, a camera or a triangle in the way. A shading pass that
// disagreed with the CPU would otherwise have to be debugged through a picture.

struct Sample {
    /// rgb: base colour. w: metallic.
    material: vec4<f32>,
    /// x: roughness. yzw: unused.
    params: vec4<f32>,
    normal: vec4<f32>,
    view: vec4<f32>,
    light: vec4<f32>,
}

struct Probe {
    count: u32,
    // Spelled out rather than a `vec3<u32>`. A `vec3` aligns to sixteen bytes,
    // so it cannot sit at offset four and the struct would be thirty-two bytes
    // against the sixteen the Rust side writes. wgpu's validation caught that,
    // which is the kind of layout trap that is silent when the sizes happen to
    // agree.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(1) var<storage, read> samples: array<Sample>;
@group(0) @binding(2) var<storage, read_write> answers: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> probe: Probe;

@compute @workgroup_size(64)
fn probe_brdf(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= probe.count {
        return;
    }
    let sample = samples[i];
    var s: Surface;
    s.base_color = sample.material.rgb;
    s.metallic = sample.material.w;
    s.roughness = sample.params.x;

    let r = brdf_evaluate(
        s,
        sample.normal.xyz,
        sample.view.xyz,
        sample.light.xyz,
    );
    // Two entries per sample: the diffuse lobe, then the specular. Separate
    // because a mistake in the coupling shows in one and not the other, and a
    // sum would hide which.
    answers[i * 2u] = vec4<f32>(r.diffuse, 0.0);
    answers[i * 2u + 1u] = vec4<f32>(r.specular, 0.0);
}
