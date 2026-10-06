// The path tracer: §2.5's renderer, first version.
//
// Appended to the shading library, so the lights, their units, the BRDF and
// the environment are the rendered view's own — one model, two consumers, as
// §3's fifth decision has it. A pixel's ray walks the scene's triangles
// through a BVH; at each surface one light is sampled directly (with a shadow
// ray) and the path continues by sampling the BRDF, diffuse or specular; a
// path that leaves the scene takes the environment's light. One sample per
// pixel per frame, added to a running sum the viewport divides out.
//
// The scene is in world space, not eye-relative like the rasteriser's: a
// path-traced scene is built once and the camera moves through it. The
// lights arrive eye-relative, as every pass gets them, and are moved back with
// `pt.eye` where they are used.

struct PtParams {
    // Clip to world, unjittered: the ray through each pixel.
    inv_view_proj: mat4x4<f32>,
    // xyz: the eye in world space.
    eye: vec4<f32>,
    // The camera's right and up in the world, for the lens.
    right: vec4<f32>,
    up: vec4<f32>,
    // x: the lens's radius in metres, zero for a pinhole. y: the distance in
    // focus, in metres.
    lens: vec4<f32>,
    // The region traced, in pixels.
    size: vec2<u32>,
    // Which sample this frame is, for the random numbers.
    sample: u32,
    light_count: u32,
    max_bounces: u32,
    // The glowing triangles, and their summed area times luminance.
    emitter_count: u32,
    emitter_total: f32,
    _pad0: u32,
    // The environment's importance map: columns, rows, nonzero when there is
    // one.
    env: vec4<u32>,
};

struct PtNode {
    min: vec3<f32>,
    // Leaf: the first index into `order`. Interior: the left child; the
    // right is the next.
    first: u32,
    max: vec3<f32>,
    // Primitives in a leaf; zero for an interior node.
    count: u32,
};

@group(0) @binding(0) var<uniform> pt: PtParams;
@group(0) @binding(1) var<storage, read> pt_positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pt_normals: array<vec4<f32>>;
// a, b, c, and the material.
@group(0) @binding(3) var<storage, read> pt_triangles: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> pt_nodes: array<PtNode>;
@group(0) @binding(5) var<storage, read> pt_order: array<u32>;
// Two per material: base colour and metallic; then roughness, the base
// colour image's layer and the metallic-roughness image's, -1 for none.
@group(0) @binding(6) var<storage, read> pt_surfaces: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> pt_sum: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> pt_uvs: array<vec2<f32>>;
// The images, one per layer at one size: colour (sRGB, decoded by the
// sampler) and data (metallic and roughness, as stored).
@group(0) @binding(9) var pt_colour_images: texture_2d_array<f32>;
@group(0) @binding(10) var pt_data_images: texture_2d_array<f32>;
@group(0) @binding(11) var pt_image_sampler: sampler;
// The glowing triangles: x the triangle's index (its bits), y where it ends
// in the cumulative distribution of area times luminance.
@group(0) @binding(12) var<storage, read> pt_emitters: array<vec2<f32>>;
// The environment's cumulative distributions: the rows, then each row's
// columns. See `environment::ImportanceMap`.
@group(0) @binding(13) var<storage, read> pt_env_cdf: array<f32>;

// --- random numbers -----------------------------------------------------------

fn pt_hash(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

var<private> pt_state: u32;

fn pt_random() -> f32 {
    pt_state = pt_hash(pt_state);
    return f32(pt_state >> 8u) / 16777216.0;
}

// --- intersection -------------------------------------------------------------

struct PtHit {
    t: f32,
    triangle: u32,
    u: f32,
    v: f32,
};

fn pt_box(origin: vec3<f32>, inv_dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, limit: f32) -> bool {
    let a = (lo - origin) * inv_dir;
    let b = (hi - origin) * inv_dir;
    let near = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
    let far = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    return far >= max(near, 0.0) && near < limit;
}

// The nearest hit, or `t` < 0. `any` stops at the first: a shadow ray.
// `skip` is a triangle the ray leaves from and must not meet again, or
// `PT_NONE`. With one, the ray may start exactly on its surface — and so meet
// a surface lying flush against it, the floor under a glass on a table, which
// any offset would step past.
const PT_NONE: u32 = 0xffffffffu;

fn pt_trace(origin: vec3<f32>, dir: vec3<f32>, limit: f32, any: bool, skip: u32) -> PtHit {
    var best = PtHit(-1.0, 0u, 0.0, 0.0);
    var nearest = limit;
    let inv_dir = 1.0 / select(dir, vec3<f32>(1e-12), abs(dir) < vec3<f32>(1e-12));
    var stack: array<u32, 48>;
    var top = 1;
    stack[0] = 0u;
    loop {
        if (top == 0) {
            break;
        }
        top -= 1;
        let node = pt_nodes[stack[top]];
        if (!pt_box(origin, inv_dir, node.min, node.max, nearest)) {
            continue;
        }
        if (node.count > 0u) {
            for (var k = 0u; k < node.count; k++) {
                let index = pt_order[node.first + k];
                if (index == skip) {
                    continue;
                }
                let tri = pt_triangles[index];
                let a = pt_positions[tri.x].xyz;
                let e1 = pt_positions[tri.y].xyz - a;
                let e2 = pt_positions[tri.z].xyz - a;
                let h = cross(dir, e2);
                let det = dot(e1, h);
                if (abs(det) < 1e-12) {
                    continue;
                }
                let f = 1.0 / det;
                let s = origin - a;
                let u = f * dot(s, h);
                if (u < 0.0 || u > 1.0) {
                    continue;
                }
                let q = cross(s, e1);
                let v = f * dot(dir, q);
                if (v < 0.0 || u + v > 1.0) {
                    continue;
                }
                let t = f * dot(e2, q);
                // A shadow ray passes through a see-through surface as
                // often as light does: always through clear glass, a fifth of
                // the time through a fifth-transmissive one, so its shadow is
                // four fifths dark. Untinted — the caustic a real block would
                // focus is a path no shadow ray can find, and a black shadow
                // under a window is the worse of the two errors.
                if (any && pt_random() < pt_surfaces[tri.w * 3u + 1u].w) {
                    continue;
                }
                let near = select(1e-4, -1e-5, skip != PT_NONE);
                if (t > near && t < nearest) {
                    nearest = t;
                    best = PtHit(max(t, 0.0), index, u, v);
                    if (any) {
                        return best;
                    }
                }
            }
        } else if (top < 46) {
            stack[top] = node.first;
            stack[top + 1] = node.first + 1u;
            top += 2;
        }
    }
    return best;
}

// --- shading ------------------------------------------------------------------

fn pt_environment(dir: vec3<f32>) -> vec3<f32> {
    if (shading.world.w <= 0.0) {
        return shading.world.rgb;
    }
    return textureSampleLevel(environment_cube, environment_sampler, unrotate(dir), 0.0).rgb * shading.environment.w;
}

fn pt_orthonormal(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.x) > 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

fn pt_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// --- glowing surfaces as lights ------------------------------------------------
//
// A small bright panel found only by paths that happen to bounce into it is a
// picture of speckles for thousands of samples. So at each surface the tracer
// also aims at one: picked in proportion to its light, a point on it chosen
// uniformly, a shadow ray to it. Both ways can find the same light, so each is
// weighted by multiple importance sampling — the power heuristic, Veach's —
// and the two shares add up to one: nothing is counted twice and each way
// carries the cases it is good at (aiming for small lights and rough
// surfaces, bouncing for big lights and mirrors).

// How many of the three ways to aim there are: at a lamp, at a glowing
// triangle, at the environment's bright part. Each that exists is chosen
// equally often.
fn pt_aim_ways() -> f32 {
    return f32(u32(pt.light_count > 0u) + u32(pt.emitter_count > 0u) + u32(pt.env.z != 0u));
}

// How often a surface aims at a glowing triangle.
fn pt_emitter_share() -> f32 {
    if (pt.emitter_count == 0u) {
        return 0.0;
    }
    return 1.0 / pt_aim_ways();
}

fn pt_env_share() -> f32 {
    if (pt.env.z == 0u) {
        return 0.0;
    }
    return 1.0 / pt_aim_ways();
}

fn pt_lamp_share() -> f32 {
    if (pt.light_count == 0u) {
        return 0.0;
    }
    return 1.0 / pt_aim_ways();
}

// --- the environment as a light -------------------------------------------------
//
// An image sky lit only through bounces is grain around every bright patch,
// and a sun in it is grain for good. So the tracer aims at it too, in
// proportion to where its light is, and weighs that against bouncing into it
// as it does for glowing surfaces.

// The environment's own frame from the world's: its turn about up undone.
fn pt_env_local(dir: vec3<f32>) -> vec3<f32> {
    return unrotate(dir);
}

// The world's frame from the environment's: the turn done.
fn pt_env_world(d: vec3<f32>) -> vec3<f32> {
    let c = shading.environment.x;
    let s = shading.environment.y;
    return vec3<f32>(d.x * c - d.y * s, d.x * s + d.y * c, d.z);
}

// The probability per solid angle of aiming at direction `dir` (world).
fn pt_env_pdf(dir: vec3<f32>) -> f32 {
    let columns = pt.env.x;
    let rows = pt.env.y;
    let d = pt_env_local(dir);
    let u = 0.5 + atan2(d.x, -d.y) / 6.2831853;
    let v = acos(clamp(d.z, -1.0, 1.0)) / 3.14159265;
    let row = min(u32(v * f32(rows)), rows - 1u);
    let column = min(u32(fract(u) * f32(columns)), columns - 1u);
    let row_p = pt_env_cdf[row] - select(0.0, pt_env_cdf[max(row, 1u) - 1u], row > 0u);
    let at = rows + row * columns + column;
    let col_p = pt_env_cdf[at] - select(0.0, pt_env_cdf[max(at, 1u) - 1u], column > 0u);
    let sin_t = sqrt(max(1.0 - d.z * d.z, 0.0));
    if (sin_t < 1e-4) {
        return 0.0;
    }
    // A cell's probability spread over its area of the unit square, then over
    // the sphere: the equirectangular map stretches by 2π² sinθ.
    return row_p * col_p * f32(rows * columns) / (2.0 * 3.14159265 * 3.14159265 * sin_t);
}

// A direction (world) picked in proportion to the environment's light.
fn pt_env_sample() -> vec3<f32> {
    let columns = pt.env.x;
    let rows = pt.env.y;
    let ur = pt_random();
    var lo = 0u;
    var hi = rows - 1u;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        if (pt_env_cdf[mid] < ur) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let row = lo;
    let uc = pt_random();
    lo = 0u;
    hi = columns - 1u;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        if (pt_env_cdf[rows + row * columns + mid] < uc) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let column = lo;
    let u = (f32(column) + pt_random()) / f32(columns);
    let v = (f32(row) + pt_random()) / f32(rows);
    let theta = v * 3.14159265;
    let phi = (u - 0.5) * 6.2831853;
    let local = vec3<f32>(sin(theta) * sin(phi), -sin(theta) * cos(phi), cos(theta));
    return pt_env_world(local);
}

// The triangle whose stretch of the distribution `u` falls in.
fn pt_pick_emitter(u: f32) -> u32 {
    var lo = 0u;
    var hi = pt.emitter_count - 1u;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        if (pt_emitters[mid].y < u) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// The probability, per unit solid angle seen from the shading point, that
// aiming at the glowing triangles lands on a point of glow `le` at distance²
// `dist2` whose surface is at `cos_l` to the ray. A point's probability is
// luminance over the total — area cancels — times the share of aiming.
fn pt_emitter_pdf(le: vec3<f32>, dist2: f32, cos_l: f32) -> f32 {
    if (pt.emitter_total <= 0.0 || cos_l <= 1e-6) {
        return 0.0;
    }
    return pt_emitter_share() * pt_luminance(le) / pt.emitter_total * dist2 / cos_l;
}

// The probability per solid angle that the bounce sampling below picks `l`:
// the specular lobe's GGX half-vector density and the cosine lobe's, mixed as
// the sampler mixes them.
fn pt_bsdf_pdf(s: Surface, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> f32 {
    let nol = dot(n, l);
    if (nol <= 0.0) {
        return 0.0;
    }
    let p_spec = clamp(mix(0.15, 1.0, s.metallic) * (1.0 - 0.5 * s.roughness), 0.1, 0.95);
    let h = normalize(v + l);
    let noh = max(dot(n, h), 1e-4);
    let voh = max(dot(v, h), 1e-4);
    let alpha = s.roughness * s.roughness;
    let spec = distribution(noh, alpha) * noh / (4.0 * voh);
    return p_spec * spec + (1.0 - p_spec) * nol / PI;
}

fn pt_power(a: f32, b: f32) -> f32 {
    let a2 = a * a;
    let b2 = b * b;
    return select(1.0, a2 / (a2 + b2), a2 + b2 > 0.0);
}

@compute @workgroup_size(8, 8)
fn pt_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= pt.size.x || id.y >= pt.size.y) {
        return;
    }
    let pixel = id.y * pt.size.x + id.x;
    pt_state = pt_hash(pixel * 9781u + pt.sample * 6271u + 1u);

    // The ray through a random point in the pixel: the antialiasing.
    let jitter = vec2<f32>(pt_random(), pt_random());
    let ndc = vec2<f32>(
        (f32(id.x) + jitter.x) / f32(pt.size.x) * 2.0 - 1.0,
        1.0 - (f32(id.y) + jitter.y) / f32(pt.size.y) * 2.0,
    );
    let near = pt.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let mid = pt.inv_view_proj * vec4<f32>(ndc, 0.5, 1.0);
    var origin = near.xyz / near.w + pt.eye.xyz;
    var dir = normalize(mid.xyz / mid.w - near.xyz / near.w);
    // A lens, when the camera has an aperture: the ray leaves a random point
    // on it and passes through the point the pinhole ray meets at the focus
    // distance — what is there is sharp, what is not is blurred by as much
    // as a real lens would blur it.
    if (pt.lens.x > 0.0) {
        let forward = normalize(cross(pt.up.xyz, pt.right.xyz));
        let focus = origin + dir * (pt.lens.y / max(dot(dir, forward), 1e-4));
        let r = sqrt(pt_random()) * pt.lens.x;
        let a = 6.2831853 * pt_random();
        origin += pt.right.xyz * (r * cos(a)) + pt.up.xyz * (r * sin(a));
        dir = normalize(focus - origin);
    }

    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    // Glass is not counted against the bounces: a path through a block of it
    // crosses two faces and, at the steep angles total internal reflection
    // traps, several more — counted, it would run out inside and come back
    // black. Its own, larger budget instead.
    var bounce = 0u;
    var crossings = 0u;
    var skip = PT_NONE;
    // The probability per solid angle the last bounce had of going where it
    // went, for weighting a glow it finds; negative when the light-aiming
    // could not have found it (the camera's ray, glass), and it counts whole.
    var last_pdf = -1.0;
    loop {
        if (bounce > pt.max_bounces || crossings > 16u) {
            break;
        }
        let hit = pt_trace(origin, dir, 1e30, false, skip);
        skip = PT_NONE;
        if (hit.t < 0.0) {
            var weight = 1.0;
            if (last_pdf >= 0.0 && pt.env.z != 0u) {
                weight = pt_power(last_pdf, pt_env_share() * pt_env_pdf(dir));
            }
            radiance += throughput * pt_environment(dir) * weight;
            break;
        }
        let tri = pt_triangles[hit.triangle];
        let w = 1.0 - hit.u - hit.v;
        let p = origin + dir * hit.t;
        let a = pt_positions[tri.x].xyz;
        var geometric = normalize(cross(pt_positions[tri.y].xyz - a, pt_positions[tri.z].xyz - a));
        var n = normalize(pt_normals[tri.x].xyz * w + pt_normals[tri.y].xyz * hit.u + pt_normals[tri.z].xyz * hit.v);
        // Which side the path arrived from, before the normals are turned to
        // face it: glass bends the other way on the way out. Asked of the
        // mesh's normal, which points out of the solid; the winding of one
        // triangle says nothing about which side is inside.
        let entering = dot(n, dir) < 0.0;
        // Both sides lit, as the rasteriser's are.
        if (dot(geometric, dir) > 0.0) {
            geometric = -geometric;
        }
        if (dot(n, geometric) < 0.0) {
            n = -n;
        }
        let m = tri.w * 3u;
        // A glowing surface adds its light wherever a path meets it — whole
        // when nothing could have aimed at it, weighted against the aiming
        // otherwise. See `pt_emitter_pdf`.
        let glow = pt_surfaces[m + 2u].rgb;
        if (pt_luminance(glow) > 0.0) {
            var weight = 1.0;
            if (last_pdf >= 0.0) {
                let cos_l = abs(dot(geometric, dir));
                weight = pt_power(last_pdf, pt_emitter_pdf(glow, hit.t * hit.t, cos_l));
            }
            radiance += throughput * glow * weight;
        }
        let extra = pt_surfaces[m + 1u];
        var s: Surface;
        s.base_color = pt_surfaces[m].rgb;
        s.metallic = pt_surfaces[m].a;
        s.roughness = extra.x;
        // The images multiply the dials, as the rasterised material does: the
        // same corner of the same picture, wrapped the same way.
        if (extra.y >= 0.0 || extra.z >= 0.0) {
            let uv = fract(pt_uvs[tri.x] * w + pt_uvs[tri.y] * hit.u + pt_uvs[tri.z] * hit.v);
            if (extra.y >= 0.0) {
                s.base_color *= textureSampleLevel(pt_colour_images, pt_image_sampler, uv, i32(extra.y), 0.0).rgb;
            }
            if (extra.z >= 0.0) {
                let mr = textureSampleLevel(pt_data_images, pt_image_sampler, uv, i32(extra.z), 0.0);
                s.roughness *= mr.g;
                s.metallic *= mr.b;
            }
        }
        s.roughness = max(s.roughness, 0.02);
        let v = -dir;
        let start = p + geometric * 1e-3 * max(1.0, length(p - pt.eye.xyz) * 0.01);
        let transmission = extra.w;

        // Glass, chosen as often as the surface is glass: the path reflects
        // or goes through by Fresnel, bent by the index of refraction, and
        // tinted by the colour on the way in. Its weight is one either way —
        // the choice is the distribution — so the throughput only takes the
        // tint. Lights are not sampled here: a smooth dielectric's lobes are
        // too narrow for a lamp to land in.
        if (transmission > 0.0 && pt_random() < transmission) {
            let ior = pt_surfaces[m + 2u].w;
            let eta = select(ior, 1.0 / ior, entering);
            // Rough glass bends about a microfacet, as rough metal reflects
            // about one.
            let alpha = s.roughness * s.roughness;
            let r1 = pt_random();
            let r2 = pt_random();
            let cos_h = sqrt((1.0 - r1) / (1.0 + (alpha * alpha - 1.0) * r1));
            let sin_h = sqrt(max(1.0 - cos_h * cos_h, 0.0));
            let phi = 6.2831853 * r2;
            var h = pt_orthonormal(n) * vec3<f32>(sin_h * cos(phi), sin_h * sin(phi), cos_h);
            let cos_i = clamp(dot(v, h), 0.0, 1.0);
            let f0 = (1.0 - eta) / (1.0 + eta);
            let fresnel = f0 * f0 + (1.0 - f0 * f0) * pow(1.0 - cos_i, 5.0);
            let bent = refract(dir, h, eta);
            var next = reflect(dir, h);
            var through = false;
            if (dot(bent, bent) > 0.0 && pt_random() >= fresnel) {
                next = bent;
                through = true;
            }
            if (through) {
                if (entering) {
                    throughput *= s.base_color;
                }
                // From the surface itself, leaving this triangle behind: an
                // offset through would step past whatever lies flush against
                // the far side.
                origin = p;
                skip = hit.triangle;
            } else {
                if (dot(next, geometric) <= 0.0) {
                    break;
                }
                origin = start;
            }
            dir = normalize(next);
            crossings += 1u;
            last_pdf = -1.0;
            continue;
        }
        bounce += 1u;

        // Aim at a glowing triangle, or at a lamp: one of the two, its share
        // made up by dividing by how often it is chosen.
        let share = pt_emitter_share();
        let env_share = pt_env_share();
        let pick = pt_random();
        let aim_at_glow = pick < share;
        let aim_at_sky = !aim_at_glow && pick < share + env_share;
        if (aim_at_sky) {
            let l = pt_env_sample();
            if (dot(l, geometric) > 0.0) {
                let pdf = env_share * pt_env_pdf(l);
                if (pdf > 0.0 && pt_trace(start, l, 1e30, true, PT_NONE).t < 0.0) {
                    let e = pt_environment(l) / pdf;
                    let weight = pt_power(pdf, pt_bsdf_pdf(s, n, v, l));
                    radiance += throughput * brdf_shade(s, n, v, l, e, true, true) * weight;
                }
            }
        }
        if (aim_at_glow) {
            let picked = bitcast<u32>(pt_emitters[pt_pick_emitter(pt_random())].x);
            let et = pt_triangles[picked];
            let ea = pt_positions[et.x].xyz;
            let eb = pt_positions[et.y].xyz;
            let ec = pt_positions[et.z].xyz;
            let r1 = sqrt(pt_random());
            let r2 = pt_random();
            let q = ea * (1.0 - r1) + eb * (r1 * (1.0 - r2)) + ec * (r1 * r2);
            let face = cross(eb - ea, ec - ea);
            let to = q - p;
            let dist2 = dot(to, to);
            let l = to * inverseSqrt(max(dist2, 1e-12));
            let cos_l = abs(dot(normalize(face), l));
            let le = pt_surfaces[et.w * 3u + 2u].rgb;
            if (picked != hit.triangle && dot(l, geometric) > 0.0 && cos_l > 1e-6) {
                let visible = pt_trace(start, l, sqrt(dist2) * 0.999, true, picked).t < 0.0;
                if (visible) {
                    let pdf = pt_emitter_pdf(le, dist2, cos_l);
                    // The light arriving, as the lux a lamp would deliver
                    // perpendicular to the ray: what `brdf_shade` takes.
                    let e = le / pdf;
                    let weight = pt_power(pdf, pt_bsdf_pdf(s, n, v, l));
                    radiance += throughput * brdf_shade(s, n, v, l, e, true, true) * weight;
                }
            }
        }

        // One lamp, chosen at random, its share scaled by how many there are.
        if (!aim_at_glow && !aim_at_sky && pt.light_count > 0u) {
            let k = min(u32(pt_random() * f32(pt.light_count)), pt.light_count - 1u);
            let light = lights[k];
            let at = p - pt.eye.xyz;
            let to_light = direction_to(light, at);
            let e = illuminance_at(light, at) * f32(pt.light_count) / pt_lamp_share();
            if (dot(to_light, geometric) > 0.0 && pt_luminance(e) > 0.0) {
                // The shadow ray goes to a random point of the source — a
                // sphere's disc facing the surface, or the sun's disc in the
                // sky — so a big source casts a soft shadow, as it does in the
                // rasterised view.
                var aim = to_light;
                var reach = 1e30;
                let around = pt_orthonormal(to_light);
                if (light.falloff == FALLOFF_PARALLEL) {
                    let spread = light.direction.w;
                    if (spread > 0.0) {
                        let cos_t = 1.0 - pt_random() * (1.0 - cos(spread));
                        let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
                        let a = 6.2831853 * pt_random();
                        aim = around * vec3<f32>(sin_t * cos(a), sin_t * sin(a), cos_t);
                    }
                } else {
                    let r = sqrt(pt_random()) * light.position.w;
                    let a = 6.2831853 * pt_random();
                    let point = light.position.xyz + around * vec3<f32>(r * cos(a), r * sin(a), 0.0);
                    let offset = point - at;
                    reach = length(offset) * 0.999;
                    aim = offset / max(length(offset), 1e-6);
                }
                let lit = !casts_shadow(light) || pt_trace(start, aim, reach, true, PT_NONE).t < 0.0;
                if (lit) {
                    radiance += throughput
                        * brdf_shade(s, n, v, to_light, e, lights_diffuse(light), lights_specular(light));
                }
            }
        }

        // The next direction: the specular lobe or the diffuse one, by how
        // much of each this surface has.
        let f0 = surface_f0(s);
        let p_spec = clamp(mix(0.15, 1.0, s.metallic) * (1.0 - 0.5 * s.roughness), 0.1, 0.95);
        let frame = pt_orthonormal(n);
        var next = vec3<f32>(0.0);
        if (pt_random() < p_spec) {
            // GGX: a half vector from the distribution, the mirror of the
            // view about it.
            let alpha = s.roughness * s.roughness;
            let r1 = pt_random();
            let r2 = pt_random();
            let cos_h = sqrt((1.0 - r1) / (1.0 + (alpha * alpha - 1.0) * r1));
            let sin_h = sqrt(max(1.0 - cos_h * cos_h, 0.0));
            let phi = 6.2831853 * r2;
            let h = frame * vec3<f32>(sin_h * cos(phi), sin_h * sin(phi), cos_h);
            next = reflect(-v, h);
            let nol = dot(n, next);
            let nov = max(dot(n, v), 1e-4);
            let voh = max(dot(v, h), 1e-4);
            let noh = max(dot(n, h), 1e-4);
            if (nol <= 0.0 || dot(next, geometric) <= 0.0) {
                break;
            }
            // Schlick's Fresnel and Smith's masking over the sampling pdf.
            let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - voh, 5.0);
            let k = alpha * 0.5;
            let g = (nol / (nol * (1.0 - k) + k)) * (nov / (nov * (1.0 - k) + k));
            throughput *= fresnel * g * voh / (nov * noh) / p_spec;
        } else {
            // Cosine-weighted, which leaves the albedo as the whole weight.
            let r1 = pt_random();
            let r2 = pt_random();
            let r = sqrt(r1);
            let phi = 6.2831853 * r2;
            next = frame * vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(1.0 - r1, 0.0)));
            if (dot(next, geometric) <= 0.0) {
                break;
            }
            throughput *= surface_diffuse_albedo(s) / (1.0 - p_spec);
        }
        last_pdf = pt_bsdf_pdf(s, n, v, normalize(next));
        // Paths that carry little are ended, the survivors made heavier so the
        // sum stays right.
        if (bounce >= 3u) {
            let keep = clamp(max(max(throughput.r, throughput.g), throughput.b), 0.05, 0.95);
            if (pt_random() > keep) {
                break;
            }
            throughput /= keep;
        }
        origin = start;
        dir = normalize(next);
    }

    // A rare path that found a small bright light through a mirror would
    // stand out as a single white pixel for hundreds of samples. Capped, at
    // a level only such paths reach.
    let l = pt_luminance(radiance);
    let cap = 1e5;
    if (l > cap) {
        radiance *= cap / l;
    }
    if (pt.sample <= 1u) {
        pt_sum[pixel] = vec4<f32>(radiance, 1.0);
    } else {
        pt_sum[pixel] += vec4<f32>(radiance, 1.0);
    }
}
