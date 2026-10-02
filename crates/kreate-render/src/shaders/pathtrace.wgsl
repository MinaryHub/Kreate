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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
fn pt_trace(origin: vec3<f32>, dir: vec3<f32>, limit: f32, any: bool) -> PtHit {
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
                let tri = pt_triangles[pt_order[node.first + k]];
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
                if (t > 1e-4 && t < nearest) {
                    nearest = t;
                    best = PtHit(t, pt_order[node.first + k], u, v);
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
    for (var bounce = 0u; bounce <= pt.max_bounces; bounce++) {
        let hit = pt_trace(origin, dir, 1e30, false);
        if (hit.t < 0.0) {
            radiance += throughput * pt_environment(dir);
            break;
        }
        let tri = pt_triangles[hit.triangle];
        let w = 1.0 - hit.u - hit.v;
        let p = origin + dir * hit.t;
        let a = pt_positions[tri.x].xyz;
        var geometric = normalize(cross(pt_positions[tri.y].xyz - a, pt_positions[tri.z].xyz - a));
        var n = normalize(pt_normals[tri.x].xyz * w + pt_normals[tri.y].xyz * hit.u + pt_normals[tri.z].xyz * hit.v);
        // Both sides lit, as the rasteriser's are.
        if (dot(geometric, dir) > 0.0) {
            geometric = -geometric;
        }
        if (dot(n, geometric) < 0.0) {
            n = -n;
        }
        let m = tri.w * 3u;
        // A glowing surface adds its light wherever a path meets it. Only
        // here: the light sampling picks among the lamps, never among the
        // glowing faces, so nothing is counted twice.
        radiance += throughput * pt_surfaces[m + 2u].rgb;
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

        // One light, chosen at random, its share scaled by how many there are.
        if (pt.light_count > 0u) {
            let k = min(u32(pt_random() * f32(pt.light_count)), pt.light_count - 1u);
            let light = lights[k];
            let at = p - pt.eye.xyz;
            let to_light = direction_to(light, at);
            let e = illuminance_at(light, at) * f32(pt.light_count);
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
                let lit = !casts_shadow(light) || pt_trace(start, aim, reach, true).t < 0.0;
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
        // Paths that carry little are ended, the survivors made heavier so the
        // sum stays right.
        if (bounce >= 2u) {
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
