// Scene lights, and the illuminance they deliver.
//
// The shading half of `kreate_render::lights`. What is here is the arithmetic a
// forward pass runs per light per fragment, and a compute entry point that
// evaluates it at given points so a device test can hold it against the CPU's
// answer value for value.
//
// # Why the struct is asserted rather than trusted
//
// A layout that disagrees with the Rust one is silent and total: every light
// after the first reads its fields out of its neighbour, and the result is a
// picture that is merely wrong. Sixteen-byte alignment and sixty-four bytes are
// what make this a valid storage-array element under every layout rule, and the
// device test is what proves the two agree.
//
// # What is not here
//
// A BRDF. `illuminance_at` answers "how much light arrives at this point", in
// lux, with no surface normal and no material — so it can be checked against
// what a light meter would read, which is the whole reason the units are
// photometric. The cosine term and the material belong to the shading pass that
// calls this.

/// Candela: divide by the square of the distance.
const FALLOFF_INVERSE_SQUARE: u32 = 0u;
/// Lux: an illuminance already, and parallel light does not diverge.
const FALLOFF_PARALLEL: u32 = 1u;

const FLAG_SHADOW: u32 = 1u;
const FLAG_DIFFUSE: u32 = 2u;
const FLAG_SPECULAR: u32 = 4u;
const FLAG_GEOMETRY: u32 = 8u;
const FLAG_PROFILE: u32 = 16u;

struct Light {
    /// xyz: eye-relative position in metres. w: source radius in metres.
    position: vec4<f32>,
    /// xyz: unit direction the light points. w: angular radius in radians.
    direction: vec4<f32>,
    /// rgb: candela or lux, per `falloff`. w: one over the square of the
    /// range in metres, or zero for a light that never runs out.
    amount: vec4<f32>,
    /// cos of the outer half-angle, then of the inner. -1 for no cone.
    cone: vec2<f32>,
    falloff: u32,
    flags: u32,
    /// A measured luminaire's shape: 32 samples, four to a vector, from 0°
    /// to 180° off the axis, relative to `amount`. Read with FLAG_PROFILE.
    profile: array<vec4<f32>, 8>,
}

/// A profile's value at `cosine` from its axis: see `lights::profile_at`.
fn profile_at(light: Light, cosine: f32) -> f32 {
    let x = acos(clamp(cosine, -1.0, 1.0)) / 3.14159265 * 31.0;
    let i = min(u32(floor(x)), 30u);
    let t = x - f32(i);
    let a = light.profile[i / 4u][i % 4u];
    let b = light.profile[(i + 1u) / 4u][(i + 1u) % 4u];
    return a * (1.0 - t) + b * t;
}

/// Whether this light sees in every direction.
///
/// A point light with no cone, which is the one kind a single shadow frustum
/// cannot cover. The same two fields `kreate_render::shadow::is_cube` reads,
/// and here because the sampler needs the answer too — a device test holds the
/// two copies together.
fn is_cube(light: Light) -> bool {
    return light.falloff != FALLOFF_PARALLEL && light.cone.x <= -1.0;
}

/// How much light arrives at `at`, in lux per channel.
///
/// `at` is eye-relative, like the packed position.
fn illuminance_at(light: Light, at: vec3<f32>) -> vec3<f32> {
    let axis = light.direction.xyz;
    var attenuation = 1.0;
    // From the surface toward the light.
    var to_light = -axis;

    if light.falloff != FALLOFF_PARALLEL {
        let offset = light.position.xyz - at;
        let distance_squared = dot(offset, offset);
        if distance_squared <= 0.0 {
            // Standing exactly on the light. Not an infinity: a point there has
            // no direction to be lit from either.
            return vec3<f32>(0.0);
        }
        // The window brings the falloff to exactly zero at the light's
        // range, so that culling the light past it leaves no step in the
        // picture. Squared twice, which leaves both ends flat: a linear taper
        // reaches zero too, with a corner at each end, and a corner in a
        // falloff is a ring on a floor.
        let t = min(distance_squared * light.amount.w, 1.0);
        let w = 1.0 - t * t;
        attenuation = (w * w) / distance_squared;
        to_light = offset * inverseSqrt(distance_squared);
    }

    // The cone. Strictly *outside* the outer edge is dark, which matters at
    // exactly cos = -1: a light with no cone packs its outer edge there, and
    // `<=` would exclude the one direction directly behind the light. A point
    // on the edge itself falls through to the interpolation and comes out at
    // zero, so there is no discontinuity either way.
    let cosine = dot(axis, -to_light);
    var cone = 1.0;
    if cosine < light.cone.x {
        cone = 0.0;
    } else if cosine < light.cone.y {
        // Linear in the cosine between the two edges. Smoother curves are
        // available and none is more correct; this one has no discontinuity at
        // either edge, which is what shows.
        cone = (cosine - light.cone.x) / (light.cone.y - light.cone.x);
    }

    // A measured luminaire's shape, by the angle from where it points.
    var shape = 1.0;
    if (light.flags & FLAG_PROFILE) != 0u {
        shape = profile_at(light, cosine);
    }
    return light.amount.rgb * (attenuation * cone * shape);
}

/// Whether this light lights the diffuse response.
fn lights_diffuse(light: Light) -> bool {
    return (light.flags & FLAG_DIFFUSE) != 0u;
}

/// Whether this light lights the specular response.
fn lights_specular(light: Light) -> bool {
    return (light.flags & FLAG_SPECULAR) != 0u;
}

/// Whether this light casts a shadow.
fn casts_shadow(light: Light) -> bool {
    return (light.flags & FLAG_SHADOW) != 0u;
}

/// The scene's lights.
///
/// Declared by the library rather than passed in, because WGSL will not take a
/// storage pointer as a function parameter — the deform shader failed to
/// compile on exactly that once. So the binding is part of the contract: a pass
/// composing this file must bind the light array at group 2, binding 1.
///
/// Group 2 is shading data by convention: binding 0 is the BRDF's DFG table and
/// binding 1 is this. Groups 0 and 1 are left to the pass for its camera and
/// its per-object data.
@group(2) @binding(1) var<storage, read> lights: array<Light>;

// --- the probe pass --------------------------------------------------------
//
// Exists for the device test: it evaluates `illuminance_at` at a list of points
// and writes the answers, so the two implementations can be compared without a
// material, a camera or a triangle in the way. A shading pass that disagreed
// with the CPU would otherwise have to be debugged through a picture.
//
// Everything below this marker is stripped when the file is composed into
// another shader. See `lights::library_source`.

struct Probe {
    /// How many points to evaluate.
    count: u32,
    /// Which light in the array to evaluate. One at a time, so a disagreement
    /// names a light rather than a sum.
    light: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> answers: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> probe: Probe;

@compute @workgroup_size(64)
fn probe_illuminance(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= probe.count {
        return;
    }
    let lux = illuminance_at(lights[probe.light], points[i].xyz);
    answers[i] = vec4<f32>(lux, 0.0);
}
