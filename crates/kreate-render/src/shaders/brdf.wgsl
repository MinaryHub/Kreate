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
const DIELECTRIC_IOR: f32 = 1.5;
const MIN_ROUGHNESS: f32 = 0.045;
const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;

/// The DFG table, then its per-roughness averages. See the header.
///
/// Declared by the library rather than passed in, because WGSL will not take a
/// storage pointer as a function parameter. So the binding is part of the
/// contract: a pass composing this file must bind the table at group 2,
/// binding 0.
///
/// Group 2 is shading data by convention: binding 0 is this and binding 1 is
/// the light array. Groups 0 and 1 are left to the pass.
@group(2) @binding(0) var<storage, read> dfg: array<vec2<f32>>;

struct Surface {
    /// A reflectance, bounded by one. The diffuse albedo of a dielectric or the
    /// specular colour of a metal, and `metallic` says which.
    base_color: vec3<f32>,
    metallic: f32,
    /// Perceptual roughness. The distribution uses its square.
    roughness: f32,
    /// A weight on the dielectric's reflection; one is physical.
    specular: f32,
    /// The index of refraction: what a dielectric reflects face on.
    ior: f32,
    /// A clear varnish, zero to one, and its perceptual roughness.
    coat: f32,
    coat_roughness: f32,
    /// A fabric's sheen: its colour (black is none) and spread.
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    /// How far the highlight stretches along the surface, and which way in
    /// turns from the tangent.
    anisotropy: f32,
    anisotropy_rotation: f32,
    /// The surface's tangent, filled in by the pass; zero where there is
    /// none, and the highlight does not stretch.
    tangent: vec3<f32>,
    /// Light that scatters inside: how much, its colour, and how far it goes
    /// in metres. See `brdf::Surface`.
    subsurface: f32,
    subsurface_color: vec3<f32>,
    subsurface_radius: f32,
    /// What light passing through keeps after the distance: tinted glass.
    attenuation_color: vec3<f32>,
    attenuation_distance: f32,
    /// A weight on the diffuse lobe; one but in the path tracer's walk.
    diffuse_weight: f32,
}

/// A surface with every field at the value that leaves it what the three
/// first ones say: what a constructor starts from, because a `var` of a
/// struct is zeros, and an index of refraction of zero is no surface.
fn surface_default() -> Surface {
    var s: Surface;
    s.base_color = vec3<f32>(0.8);
    s.metallic = 0.0;
    s.roughness = 0.5;
    s.specular = 1.0;
    s.ior = DIELECTRIC_IOR;
    s.coat = 0.0;
    s.coat_roughness = 0.03;
    s.sheen_color = vec3<f32>(0.0);
    s.sheen_roughness = 0.3;
    s.anisotropy = 0.0;
    s.anisotropy_rotation = 0.0;
    s.tangent = vec3<f32>(0.0);
    s.subsurface = 0.0;
    s.subsurface_color = vec3<f32>(0.8);
    s.subsurface_radius = 0.01;
    s.attenuation_color = vec3<f32>(1.0);
    s.attenuation_distance = 0.0;
    s.diffuse_weight = 1.0;
    return s;
}

/// `((n − 1)/(n + 1))² · specular`, at most one: see `brdf::dielectric_f0`.
fn dielectric_f0(ior: f32, specular: f32) -> f32 {
    let n = max(ior, 1.0);
    let r = (n - 1.0) / (n + 1.0);
    return min(r * r * max(specular, 0.0), 1.0);
}

struct Response {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

fn surface_f0(s: Surface) -> vec3<f32> {
    let m = clamp(s.metallic, 0.0, 1.0);
    return mix(vec3<f32>(dielectric_f0(s.ior, s.specular)), s.base_color, m);
}

fn surface_diffuse_albedo(s: Surface) -> vec3<f32> {
    // Zero for a metal, and not as an approximation: a conductor absorbs what
    // enters it, so nothing scatters back out.
    // The light that scatters inside comes back its own colour: see
    // `Surface::diffuse_albedo`.
    let albedo = mix(s.base_color, s.subsurface_color, clamp(s.subsurface, 0.0, 1.0));
    return albedo * (1.0 - clamp(s.metallic, 0.0, 1.0)) * clamp(s.diffuse_weight, 0.0, 1.0);
}

/// The roughness the energy terms are read at: see `Surface::lobe_roughness`.
fn surface_lobe_roughness(s: Surface) -> f32 {
    let strength = clamp(s.anisotropy, 0.0, 1.0);
    if strength <= 0.0 || all(s.tangent == vec3<f32>(0.0)) {
        return clamp(s.roughness, 0.0, 1.0);
    }
    let alpha = surface_alpha(s);
    let along = alpha + (1.0 - alpha) * strength * strength;
    return sqrt((along + alpha) * 0.5);
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

// --- sheen: see `brdf.rs`'s section of the same name ---------------------

const MIN_SHEEN_ROUGHNESS: f32 = 0.07;

/// Where the sheen's table starts in the buffer: after the DFG table and its
/// averages. Its own averages follow it.
fn sheen_base() -> u32 {
    return ALBEDO_LUT * ALBEDO_LUT + ALBEDO_LUT;
}

fn sheen_lookup(nov: f32, roughness: f32) -> f32 {
    let x = lut_index(nov);
    let y = lut_index(roughness);
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, ALBEDO_LUT - 1u);
    let y1 = min(y0 + 1u, ALBEDO_LUT - 1u);
    let fx = x - floor(x);
    let fy = y - floor(y);
    let b = sheen_base();
    let near = mix(dfg[b + y0 * ALBEDO_LUT + x0].x, dfg[b + y0 * ALBEDO_LUT + x1].x, fx);
    let far = mix(dfg[b + y1 * ALBEDO_LUT + x0].x, dfg[b + y1 * ALBEDO_LUT + x1].x, fx);
    return mix(near, far, fy);
}

fn sheen_average(roughness: f32) -> f32 {
    let b = sheen_base() + ALBEDO_LUT * ALBEDO_LUT;
    let y = lut_index(roughness);
    let y0 = u32(floor(y));
    let y1 = min(y0 + 1u, ALBEDO_LUT - 1u);
    return mix(dfg[b + y0].x, dfg[b + y1].x, y - floor(y));
}

fn sheen_distribution(noh: f32, alpha: f32) -> f32 {
    let inv = 1.0 / alpha;
    let sin2 = max(1.0 - noh * noh, 0.0078125);
    return (2.0 + inv) * pow(sin2, inv * 0.5) / (2.0 * PI);
}

fn sheen_visibility(nov: f32, nol: f32) -> f32 {
    return 1.0 / max(4.0 * (nol + nov - nol * nov), 1e-4);
}

fn sheen_alpha(roughness: f32) -> f32 {
    let r = clamp(roughness, MIN_SHEEN_ROUGHNESS, 1.0);
    return r * r;
}

/// The sheen colour's brightest channel: how much of the lobe's albedo the
/// sheen takes from the surface below.
fn sheen_weight(s: Surface) -> f32 {
    return clamp(max(max(s.sheen_color.r, s.sheen_color.g), s.sheen_color.b), 0.0, 1.0);
}

/// The stretched lobe's frame and widths: see `brdf::stretch`. `w` is zero
/// for a lobe that does not stretch.
struct Stretch {
    along: vec3<f32>,
    across: vec3<f32>,
    at: f32,
    ab: f32,
    on: bool,
}

fn stretch(s: Surface, n: vec3<f32>, alpha: f32) -> Stretch {
    var out: Stretch;
    out.on = false;
    let strength = clamp(s.anisotropy, 0.0, 1.0);
    if strength <= 0.0 {
        return out;
    }
    var t = s.tangent - n * dot(n, s.tangent);
    if dot(t, t) < 1e-12 {
        return out;
    }
    t = normalize(t);
    let b = cross(n, t);
    let a = s.anisotropy_rotation * 2.0 * PI;
    out.along = t * cos(a) + b * sin(a);
    out.across = cross(n, out.along);
    out.at = alpha + (1.0 - alpha) * strength * strength;
    out.ab = alpha;
    out.on = true;
    return out;
}

fn distribution_stretched(toh: f32, boh: f32, noh: f32, at: f32, ab: f32) -> f32 {
    let x = (toh / at) * (toh / at) + (boh / ab) * (boh / ab) + noh * noh;
    return 1.0 / max(PI * at * ab * x * x, 1.17549435e-38);
}

fn visibility_stretched(tov: f32, bov: f32, nov: f32, tol: f32, bol: f32, nol: f32, at: f32, ab: f32) -> f32 {
    let v = nol * sqrt((at * tov) * (at * tov) + (ab * bov) * (ab * bov) + nov * nov);
    let l = nov * sqrt((at * tol) * (at * tol) + (ab * bol) * (ab * bol) + nol * nol);
    return 0.5 / max(v + l, 1.17549435e-38);
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
    let roughness = surface_lobe_roughness(s);
    let f0 = surface_f0(s);
    let f = fresnel(f0, voh);
    var d = distribution(noh, alpha);
    var vis = visibility(nov, nol, alpha);
    let st = stretch(s, n, alpha);
    if st.on {
        d = distribution_stretched(dot(st.along, h), dot(st.across, h), noh, st.at, st.ab);
        vis = visibility_stretched(
            dot(st.along, v), dot(st.across, v), nov,
            dot(st.along, l), dot(st.across, l), nol,
            st.at, st.ab,
        );
    }

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

    // The sheen over the surface, under any varnish: see `brdf::evaluate`.
    let sw = sheen_weight(s);
    if sw > 0.0 {
        let r = s.sheen_roughness;
        let below = (1.0 - sw * sheen_lookup(nov, r)) * (1.0 - sw * sheen_lookup(nol, r))
            / max(1.0 - sw * sheen_average(r), 1e-3);
        let lobe = sheen_distribution(noh, sheen_alpha(r)) * sheen_visibility(nov, nol);
        out.diffuse *= below;
        out.specular = out.specular * below + clamp(s.sheen_color, vec3<f32>(0.0), vec3<f32>(1.0)) * lobe;
    }

    // The varnish, and what it leaves the surface below: see `brdf::evaluate`.
    let coat = clamp(s.coat, 0.0, 1.0);
    if coat > 0.0 {
        let coat_roughness = clamp(s.coat_roughness, MIN_ROUGHNESS, 1.0);
        let coat_alpha = coat_roughness * coat_roughness;
        let cf0 = vec3<f32>(DIELECTRIC_F0);
        let cf = fresnel(cf0, voh);
        let cd = distribution(noh, coat_alpha);
        let cv = visibility(nov, nol, coat_alpha);
        let cc = energy_compensation(cf0, nov, coat_roughness);
        let e_v = specular_share(cf0, nov, coat_roughness).x;
        let e_l = specular_share(cf0, nol, coat_roughness).x;
        let cab = dfg_average(coat_roughness);
        let e_avg = clamp(DIELECTRIC_F0 * cab.x + cab.y, 0.0, 0.99);
        let below = (1.0 - coat * e_v) * (1.0 - coat * e_l) / (1.0 - coat * e_avg);
        out.diffuse *= below;
        out.specular = out.specular * below + coat * cd * cv * cf * cc;
    }

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
//
// Everything below this marker is stripped when the file is composed into
// another shader. See `brdf::library_source`.

struct Sample {
    /// rgb: base colour. w: metallic.
    material: vec4<f32>,
    /// x: roughness. y: specular. z: index of refraction. w: the coat.
    params: vec4<f32>,
    /// x: the coat's roughness. y: the sheen's. z: anisotropy. w: its turn.
    more: vec4<f32>,
    /// rgb: the sheen's colour.
    sheen: vec4<f32>,
    /// xyz: the tangent.
    tangent: vec4<f32>,
    /// rgb: the subsurface colour. w: how much.
    subsurface: vec4<f32>,
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

@group(0) @binding(0) var<storage, read> samples: array<Sample>;
@group(0) @binding(1) var<storage, read_write> answers: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> probe: Probe;

@compute @workgroup_size(64)
fn probe_brdf(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= probe.count {
        return;
    }
    let sample = samples[i];
    var s = surface_default();
    s.base_color = sample.material.rgb;
    s.metallic = sample.material.w;
    s.roughness = sample.params.x;
    s.specular = sample.params.y;
    s.ior = sample.params.z;
    s.coat = sample.params.w;
    s.coat_roughness = sample.more.x;
    s.sheen_roughness = sample.more.y;
    s.anisotropy = sample.more.z;
    s.anisotropy_rotation = sample.more.w;
    s.sheen_color = sample.sheen.rgb;
    s.tangent = sample.tangent.xyz;
    s.subsurface_color = sample.subsurface.rgb;
    s.subsurface = sample.subsurface.w;

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
