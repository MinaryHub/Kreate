//! The path tracer: §2.5's renderer on the rendered view's own lights, BRDF
//! and environment. See `shaders/pathtrace.wgsl`.
//!
//! The scene is handed over as triangles in world space and kept until it is
//! handed over again — a camera moving through it changes nothing here. Each
//! frame traces one sample per pixel into a running sum; the viewport divides
//! it out onto the linear picture and the display transform does the rest.
//!
//! # What it does
//!
//! - Materials are the compiled node graphs, the viewport's own code, with
//!   their images, normal maps and bumps.
//! - At each surface one source is aimed at — a lamp (chosen by how much it
//!   lights the point), a glowing triangle, or the environment's bright part
//!   — and the path goes on by sampling the BRDF; a glow or a sky that both
//!   could find is weighted between them by multiple importance sampling.
//! - Glass (transmission, index of refraction, total internal reflection),
//!   emission, cut-outs, a lens with an aperture, baked smoke and fire, a
//!   product shot's floor.
//! - Each pixel's samples are a scrambled Sobol sequence, not independent
//!   draws; a pixel whose mean is quiet stops taking more; a view moved a
//!   little carries the last picture's means over.
//!
//! # What it does not, yet
//!
//! Area lights are discs and spheres. The scene is rebuilt whole when it is
//! handed over again.
//!
//! # Placings
//!
//! A mesh handed over more than once — the same slices of positions and
//! indices, as linked copies of an object share — is stored once, in its own
//! space, and placed: the rays are carried into each placing rather than the
//! triangles out of it. Every other mesh is moved into the world and they go
//! together under one placing that moves nothing, so a scene of distinct
//! objects is walked as one tree as before. Where the device has ray tracing
//! (`wgpu`'s ray queries) the same placings are built into its acceleration
//! structure and the rays asked of it; elsewhere they walk the trees in the
//! shader.

use glam::{DAffine3, Mat4, Vec3};

fn source(materials: &str, count: u32, hardware: bool) -> String {
    format!(
        "{}{}\n{}\n{}\n{materials}\n{}",
        if hardware {
            "enable wgpu_ray_query;\n"
        } else {
            ""
        },
        crate::shading::shader_source(),
        include_str!("shaders/pathtrace.wgsl"),
        if hardware {
            include_str!("shaders/pathtrace_rq.wgsl")
        } else {
            include_str!("shaders/pathtrace_bvh.wgsl")
        },
        dispatch(count)
    )
}

/// Whether a device can trace with ray queries, and is let to:
/// `KREATE_RAY_QUERY=0` keeps the shader's own walk.
pub fn ray_queries_on(device: &wgpu::Device) -> bool {
    device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        && std::env::var("KREATE_RAY_QUERY").map_or(true, |v| v != "0")
}

/// `pt_material`: which compiled material a surface names, by number.
///
/// Generated because WGSL has no function pointers: one arm per material,
/// each calling the function the application compiled for it. With none,
/// the arms are not there and nothing names the material structs — which is
/// a tracer handed only flat surfaces.
fn dispatch(count: u32) -> String {
    let mut out = String::from(
        "fn pt_material(id: u32, uv: vec2<f32>, uv_dx: vec2<f32>, uv_dy: vec2<f32>, p: vec3<f32>, n: vec3<f32>, pdx: vec3<f32>, pdy: vec3<f32>, facing: f32) -> PtMaterialOut {
    var out: PtMaterialOut;
    out.surface = surface_default();
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.gradient = vec3<f32>(0.0);
    out.alpha = 1.0;
    out.clip = 0.0;
",
    );
    if count > 0 {
        out.push_str(
            "    var g: MaterialGeometry;
    g.position = p;
    g.normal = n;
    g.position_dx = pdx;
    g.position_dy = pdy;
    g.facing = facing;
    switch id {
",
        );
        for i in 0..count {
            out.push_str(&format!(
                "        case {i}u: {{
            let m = material_sample_{i}(uv, uv_dx, uv_dy, g);
            out.surface = m.surface;
            out.normal = m.normal;
            out.gradient = m.gradient;
            out.alpha = m.alpha;
            out.clip = m.clip;
        }}
"
            ));
        }
        out.push_str("        default: {}\n    }\n");
    }
    out.push_str("    return out;\n}\n");
    out
}

/// The tracer's shader as a device receives it, for the portability tests.
pub fn shader_source() -> String {
    source("", 0, false)
}

/// The tracer's shader for a device with ray queries.
pub fn shader_source_hardware() -> String {
    source("", 0, true)
}

/// The tracer's shader with materials, as [`PathTracer::set_scene`] builds it
/// for a device without ray queries.
pub fn shader_source_with(materials: &PtMaterials<'_>) -> String {
    source(materials.code, materials.count, false)
}

/// The scene's compiled materials, as the tracer runs them.
///
/// Made by the application, which can see the compiler: `kreate-render`
/// cannot, for the reason it cannot see a document. `code` is the compiler's
/// declarations once and then each material as `material_sample_{i}`, its
/// constants read from `values` and its images from the layers `images`
/// gives them — see `kreate_shader::for_tracer`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PtMaterials<'a> {
    pub code: &'a str,
    pub count: u32,
    /// Every material's constants, one after another.
    pub values: &'a [[f32; 4]],
}

/// The resolve's shader, for the portability tests.
pub fn resolve_source() -> &'static str {
    include_str!("shaders/pathtrace_resolve.wgsl")
}

/// How many samples a still view takes before it stops.
pub const SAMPLES: u32 = 512;

/// One mesh of the scene: its triangles in its own space, where it is, and
/// which of the flat surfaces it is made of.
pub struct PtMesh<'a> {
    pub positions: &'a [[f32; 3]],
    /// Texture coordinates, one per position.
    pub uvs: &'a [[f32; 2]],
    /// Unit normals, one per position.
    pub normals: &'a [Vec3],
    /// Vertex colours, one per position — a byte a channel, red lowest —
    /// which multiply the surface's base colour. Empty for white.
    pub colors: &'a [u32],
    pub indices: &'a [u32],
    pub transform: DAffine3,
    pub surface: u32,
    /// Where the mesh changes surface, for one whose faces wear different
    /// materials: from each run's first index into `indices` on, that
    /// surface. Empty for one surface throughout; before the first run is
    /// `surface`.
    pub runs: &'a [(u32, u32)],
}

/// A thin lens: depth of field, from a camera with an aperture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    /// The aperture's radius, in metres: half of focal length over f-number.
    pub radius: f32,
    /// How far away is in focus, in metres.
    pub focus: f32,
}

/// A standard surface: linear base colour, metallic, roughness, and the
/// images that multiply them — indices into the images handed over with it.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PtSurface {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub base_color_image: Option<usize>,
    /// glTF's packing: roughness in green, metallic in blue.
    pub metallic_roughness_image: Option<usize>,
    /// Light the surface emits by itself, in nits.
    pub emission: [f32; 3],
    /// Glows from the side its normals face only, as an area light's panel
    /// does; both sides otherwise.
    pub one_sided: bool,
    /// How much passes through, zero to one, and how it bends.
    pub transmission: f32,
    pub ior: f32,
    /// Which of the compiled materials gives the surface its colour, metal,
    /// roughness and normal, if any: the flat values and images above are
    /// then not read. A light's panel has none.
    pub material: Option<u32>,
    /// A cut-out's threshold, read against `base_color_image`'s alpha where a
    /// ray meets the surface — so the cut is a hole for shadows too, which
    /// the material's code, run only at the hit kept, cannot make.
    pub alpha_clip: f32,
}

/// An image a surface reads: four bytes a pixel, row by row; sRGB-encoded for
/// a colour, as stored for data.
#[derive(Clone, Copy, Debug)]
pub struct PtImage<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
    pub color: bool,
}

/// The size every image is resampled to for the tracer: one layer each of a
/// texture array, which is what a compute pass can index by number.
const IMAGE_SIZE: u32 = 512;

/// `image` resampled to `IMAGE_SIZE` square, nearest texel.
fn resample(image: &PtImage<'_>) -> Vec<u8> {
    let (w, h) = (image.width.max(1), image.height.max(1));
    let mut out = vec![255u8; (IMAGE_SIZE * IMAGE_SIZE * 4) as usize];
    for y in 0..IMAGE_SIZE {
        let sy = (y * h / IMAGE_SIZE).min(h - 1);
        for x in 0..IMAGE_SIZE {
            let sx = (x * w / IMAGE_SIZE).min(w - 1);
            let from = ((sy * w + sx) * 4) as usize;
            let to = ((y * IMAGE_SIZE + x) * 4) as usize;
            if let Some(src) = image.pixels.get(from..from + 4) {
                out[to..to + 4].copy_from_slice(src);
            }
        }
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    inv_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    lens: [f32; 4],
    size: [u32; 2],
    sample: u32,
    light_count: u32,
    max_bounces: u32,
    /// Glowing triangles the path tracer can aim at, and their summed
    /// weight (area times luminance). See `emitters_of`.
    emitter_count: u32,
    emitter_total: f32,
    /// See [`PathTracer::noise_threshold`].
    noise_threshold: f32,
    /// The environment's importance map: columns, rows, whether there is
    /// one, and nothing.
    env: [u32; 4],
    /// The last traced frame's clip from eye-relative world, and its eye:
    /// where this frame's first hits were on that frame's picture.
    prev_view_proj: [[f32; 4]; 4],
    prev_eye: [f32; 4],
    /// x: nonzero when this sample may carry the last frame's mean over.
    /// y: the most frames of it a pixel keeps.
    reproject: [u32; 4],
    /// x: the floor's height in the world. y: nonzero when there is one.
    /// z: how much of a mirror it is.
    ground: [f32; 4],
}

/// How much of what stands on the floor it shows, looking straight down; a
/// grazing look sees more, as a polished floor does.
const FLOOR_REFLECTION: f32 = 0.08;

/// The most of the last frames a pixel carries over when the view moves:
/// its own samples outweigh them from this many on.
pub const HISTORY: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNode {
    min: [f32; 3],
    first: u32,
    max: [f32; 3],
    count: u32,
}

/// A placing of a mesh, as `PtInstance` in the shader reads it.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstance {
    world: [[f32; 4]; 3],
    local: [[f32; 4]; 3],
    root: u32,
    first: u32,
    pad: [u32; 2],
}

/// An affine map's rows, three of four: what the shader and the
/// acceleration structure both take.
fn rows(m: DAffine3) -> [[f32; 4]; 3] {
    let (c, t) = (m.matrix3, m.translation);
    [
        [c.x_axis.x, c.y_axis.x, c.z_axis.x, t.x],
        [c.x_axis.y, c.y_axis.y, c.z_axis.y, t.y],
        [c.x_axis.z, c.y_axis.z, c.z_axis.z, t.z],
    ]
    .map(|r| r.map(|v| v as f32))
}

/// What makes two meshes one shape, placed twice: the same slices, the same
/// surfaces on them.
#[derive(PartialEq, Eq, Hash)]
struct Shape {
    positions: (usize, usize),
    normals: usize,
    uvs: usize,
    colors: usize,
    indices: (usize, usize),
    surface: u32,
    runs: Vec<(u32, u32)>,
}

impl Shape {
    fn of(mesh: &PtMesh<'_>) -> Self {
        Self {
            positions: (mesh.positions.as_ptr() as usize, mesh.positions.len()),
            normals: mesh.normals.as_ptr() as usize,
            uvs: mesh.uvs.as_ptr() as usize,
            colors: mesh.colors.as_ptr() as usize,
            indices: (mesh.indices.as_ptr() as usize, mesh.indices.len()),
            surface: mesh.surface,
            runs: mesh.runs.to_vec(),
        }
    }
}

/// A run of the scene's vertices and triangles stored together: the
/// meshes placed once, in the world, or one mesh placed many times, in its
/// own space.
#[derive(Clone, Copy, Default)]
struct Stored {
    first_vertex: u32,
    vertices: u32,
    first_triangle: u32,
    triangles: u32,
}

/// The scene laid out for the shader: what [`lay_out`] makes of the meshes.
#[derive(Default)]
struct Layout {
    local: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<u32>,
    positions: Vec<Vec3>,
    normals: Vec<[f32; 4]>,
    triangles: Vec<[u32; 4]>,
    stored: Vec<Stored>,
    /// Each placing: which stored run, and where it goes.
    placings: Vec<(usize, DAffine3)>,
}

/// Sorts the meshes into stored runs and placings. A mesh whose shape comes
/// more than once — and whose placing can be undone — is stored once and
/// placed each time; the rest go into the world, together.
fn lay_out(meshes: &[PtMesh<'_>]) -> Layout {
    let invertible = |m: &PtMesh<'_>| {
        let d = m.transform.matrix3.determinant();
        d.is_finite() && d.abs() > 1e-18 && m.transform.translation.is_finite()
    };
    let mut uses: std::collections::HashMap<Shape, usize> = std::collections::HashMap::new();
    for mesh in meshes.iter().filter(|m| invertible(m)) {
        *uses.entry(Shape::of(mesh)).or_default() += 1;
    }
    let placed = |m: &PtMesh<'_>| invertible(m) && uses.get(&Shape::of(m)).is_some_and(|&n| n > 1);
    let mut out = Layout::default();
    let append = |out: &mut Layout, mesh: &PtMesh<'_>, to: DAffine3| {
        let base = out.positions.len() as u32;
        let normal_matrix = to.matrix3.inverse().transpose();
        out.uvs.extend(
            (0..mesh.positions.len()).map(|i| mesh.uvs.get(i).copied().unwrap_or([0.0; 2])),
        );
        out.colors.extend(
            (0..mesh.positions.len()).map(|i| mesh.colors.get(i).copied().unwrap_or(u32::MAX)),
        );
        for (p, n) in mesh.positions.iter().zip(mesh.normals) {
            out.local.push([p[0], p[1], p[2], 1.0]);
            out.local.push([n.x, n.y, n.z, 0.0]);
            let w = to.transform_point3(glam::DVec3::from_array(p.map(|c| c as f64)));
            out.positions.push(w.as_vec3());
            let wn = (normal_matrix * n.as_dvec3()).normalize_or_zero().as_vec3();
            out.normals.push([wn.x, wn.y, wn.z, 0.0]);
        }
        // Positions without normals: the rest of the run, at the origin,
        // so the indices still land.
        for _ in mesh.normals.len()..mesh.positions.len() {
            out.local.push([0.0, 0.0, 0.0, 1.0]);
            out.local.push([0.0; 4]);
            out.positions.push(Vec3::ZERO);
            out.normals.push([0.0; 4]);
        }
        let mut run = 0;
        let mut surface = mesh.surface;
        for (k, t) in mesh.indices.as_chunks::<3>().0.iter().enumerate() {
            while let Some(&(start, next)) = mesh.runs.get(run) {
                if start as usize > k * 3 {
                    break;
                }
                surface = next;
                run += 1;
            }
            out.triangles
                .push([base + t[0], base + t[1], base + t[2], surface]);
        }
    };
    let begin = |out: &Layout| Stored {
        first_vertex: out.positions.len() as u32,
        first_triangle: out.triangles.len() as u32,
        ..Default::default()
    };
    let end = |out: &Layout, mut s: Stored| {
        s.vertices = out.positions.len() as u32 - s.first_vertex;
        s.triangles = out.triangles.len() as u32 - s.first_triangle;
        s
    };
    let world = begin(&out);
    for mesh in meshes.iter().filter(|m| !placed(m)) {
        append(&mut out, mesh, mesh.transform);
    }
    let world = end(&out, world);
    if world.triangles > 0 {
        out.stored.push(world);
        out.placings.push((0, DAffine3::IDENTITY));
    }
    let mut stored_as: std::collections::HashMap<Shape, usize> = std::collections::HashMap::new();
    for mesh in meshes.iter().filter(|m| placed(m)) {
        let shape = Shape::of(mesh);
        let index = match stored_as.get(&shape) {
            Some(&i) => i,
            None => {
                let s = begin(&out);
                append(&mut out, mesh, DAffine3::IDENTITY);
                let s = end(&out, s);
                out.stored.push(s);
                stored_as.insert(shape, out.stored.len() - 1);
                out.stored.len() - 1
            }
        };
        if out.stored[index].triangles > 0 {
            out.placings.push((index, mesh.transform));
        }
    }
    out
}

/// The world box of a box seen through a placing: its eight corners moved.
fn placed_box(b: kreate_math::Aabb, m: DAffine3) -> kreate_math::Aabb {
    let mut out = kreate_math::Aabb::EMPTY;
    for k in 0..8 {
        let corner = Vec3::new(
            if k & 1 == 0 { b.min.x } else { b.max.x },
            if k & 2 == 0 { b.min.y } else { b.max.y },
            if k & 4 == 0 { b.min.z } else { b.max.z },
        );
        out = out.extend(m.transform_point3(corner.as_dvec3()).as_vec3());
    }
    out
}

/// What the device's ray tracing holds of the scene: one bottom level per
/// stored run, and the top level placing them. Kept alive with the scene.
struct HardwareScene {
    _bottom: Vec<Option<wgpu::Blas>>,
    top: wgpu::Tlas,
}

struct Scene {
    uvs: wgpu::Buffer,
    colors: wgpu::Buffer,
    colour_images: wgpu::TextureView,
    data_images: wgpu::TextureView,
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    triangles: wgpu::Buffer,
    nodes: wgpu::Buffer,
    order: wgpu::Buffer,
    surfaces: wgpu::Buffer,
    /// The glowing triangles: each one's index and its place in the
    /// cumulative distribution.
    emitters: wgpu::Buffer,
    /// Every material's constants.
    values: wgpu::Buffer,
    /// Each vertex in its object's space: position, then normal.
    local: wgpu::Buffer,
    /// The placings: see `GpuInstance`.
    instances: wgpu::Buffer,
    /// The device's acceleration structure, where it traces with one.
    hardware: Option<HardwareScene>,
    emitter_count: u32,
    emitter_total: f32,
    /// The lowest point of the scene: where its floor is, when it is below
    /// the ground.
    lowest: f32,
}

/// The glowing triangles, for the path tracer to aim at: each one's index
/// (its bits, in the first float), the cumulative distribution of area times
/// luminance (the second) and the placing it is seen through (its bits, the
/// third); and that weight's total. A mesh placed twice glows twice.
///
/// Picked in proportion to how much light each gives — a big bright panel
/// more often than a small dim one — and a point on the one picked uniformly,
/// so the probability of a point is luminance over the total, whatever the
/// triangle's size. The shader relies on that.
fn emitters_of(layout: &Layout, surfaces: &[PtSurface]) -> (Vec<[f32; 4]>, f32) {
    let luminance = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let glowing: Vec<Vec<(u32, f32)>> = layout
        .stored
        .iter()
        .map(|run| {
            (run.first_triangle..run.first_triangle + run.triangles)
                .filter_map(|i| {
                    let t = layout.triangles[i as usize];
                    let lum = luminance(surfaces.get(t[3] as usize)?.emission);
                    (lum > 0.0).then_some((i, lum))
                })
                .collect()
        })
        .collect();
    let mut out = Vec::new();
    let mut total = 0.0f32;
    for (inst, &(run, to)) in layout.placings.iter().enumerate() {
        for &(i, lum) in &glowing[run] {
            let t = layout.triangles[i as usize];
            let [a, b, c] = [t[0], t[1], t[2]].map(|v| {
                to.transform_point3(layout.positions[v as usize].as_dvec3())
                    .as_vec3()
            });
            let area = 0.5 * (b - a).cross(c - a).length();
            if area.is_nan() || area <= 0.0 {
                continue;
            }
            total += area * lum;
            out.push([f32::from_bits(i), total, f32::from_bits(inst as u32), 0.0]);
        }
    }
    if total > 0.0 {
        for e in &mut out {
            e[1] /= total;
        }
    }
    (out, total)
}

/// Builds the device's acceleration structure over the placings: a bottom
/// level per stored run, its own positions and indices copied to buffers
/// the build reads, and a top level placing them, each numbered as the
/// shader's placings are.
fn build_hardware(device: &wgpu::Device, queue: &wgpu::Queue, layout: &Layout) -> HardwareScene {
    use wgpu::util::DeviceExt;
    let vertices: Vec<[f32; 3]> = layout.positions.iter().map(|p| p.to_array()).collect();
    let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("path trace build positions"),
        contents: bytemuck::cast_slice(if vertices.is_empty() {
            &[[0.0f32; 3]][..]
        } else {
            &vertices
        }),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    // Each run's indices counted from its own first vertex.
    let mut indices: Vec<u32> = Vec::with_capacity(layout.triangles.len() * 3);
    for run in &layout.stored {
        let first = run.first_triangle as usize;
        for t in &layout.triangles[first..first + run.triangles as usize] {
            indices.extend([t[0], t[1], t[2]].map(|v| v - run.first_vertex));
        }
    }
    if indices.is_empty() {
        indices.extend([0, 0, 0]);
    }
    let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("path trace build indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let sizes: Vec<wgpu::BlasTriangleGeometrySizeDescriptor> = layout
        .stored
        .iter()
        .map(|run| wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: run.vertices,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(run.triangles * 3),
            // Not opaque: every candidate comes back to the shader, which
            // decides about cut-outs, glass and the triangle a ray leaves.
            flags: wgpu::AccelerationStructureGeometryFlags::empty(),
        })
        .collect();
    let bottom: Vec<Option<wgpu::Blas>> = sizes
        .iter()
        .map(|size| {
            // A run with no triangles has nothing to build, and no placing.
            (size.index_count != Some(0)).then(|| {
                device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("path trace mesh"),
                        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![size.clone()],
                    },
                )
            })
        })
        .collect();
    let mut top = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("path trace scene"),
        max_instances: layout.placings.len().max(1) as u32,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    for (i, &(run, to)) in layout.placings.iter().enumerate() {
        if let (Some(slot), Some(blas)) = (top.get_mut_single(i), &bottom[run]) {
            *slot = Some(wgpu::TlasInstance::new(
                blas,
                rows(to).concat().try_into().unwrap_or([0.0; 12]),
                i as u32,
                0xff,
            ));
        }
    }
    let builds: Vec<wgpu::BlasBuildEntry<'_>> = layout
        .stored
        .iter()
        .zip(&bottom)
        .zip(&sizes)
        .filter_map(|((run, blas), size)| Some((run, blas.as_ref()?, size)))
        .map(|(run, blas, size)| wgpu::BlasBuildEntry {
            blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size,
                vertex_buffer: &vertices,
                first_vertex: run.first_vertex,
                vertex_stride: 12,
                index_buffer: Some(&indices),
                first_index: Some(run.first_triangle * 3),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        })
        .collect();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("path trace scene build"),
    });
    encoder.build_acceleration_structures(builds.iter(), [&top]);
    queue.submit([encoder.finish()]);
    HardwareScene {
        _bottom: bottom,
        top,
    }
}

pub struct PathTracer {
    /// The environment's importance map (see `environment::ImportanceMap`),
    /// or one zero when there is none; and its columns and rows.
    env_map: wgpu::Buffer,
    env_size: Option<(u32, u32)>,
    trace: wgpu::ComputePipeline,
    trace_layout: wgpu::PipelineLayout,
    /// The materials' code the trace pipeline was built with: a scene with
    /// other code rebuilds it, one with the same keeps it.
    trace_code: (String, u32),
    resolve: wgpu::RenderPipeline,
    scene_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    image_sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    resolve_params: wgpu::Buffer,
    scene: Option<Scene>,
    sum: wgpu::Buffer,
    sum_pixels: u32,
    /// Per pixel: where its first hit is and whether it has one, that hit's
    /// normal, and the mean carried over from the frames before with how
    /// many frames it stands for. See [`HISTORY`].
    aux: wgpu::Buffer,
    /// The last frame's sum and then its `aux`, copied when the view moves.
    prev: wgpu::Buffer,
    /// The view the sum was traced through: clip from eye-relative world,
    /// and the eye.
    previous: Option<(Mat4, glam::DVec3)>,
    /// Whether the sum is of the scene as it stands — false once the scene,
    /// its lights or its size change, when nothing may be carried over.
    history_ok: bool,
    /// This sample carries the last frame's mean over.
    reproject_now: bool,
    scene_bind: Option<wgpu::BindGroup>,
    resolve_bind: Option<wgpu::BindGroup>,
    size: (u32, u32),
    sample: u32,
    /// Adaptive sampling: a pixel stops taking samples once the standard
    /// error of its mean is under this fraction of the mean. Zero traces
    /// every pixel every sample.
    pub noise_threshold: f32,
    /// How many times a path bounces before it stops.
    pub max_bounces: u32,
    /// A floor under the scene that catches its shadows and, faintly, its
    /// reflections, and is otherwise the backdrop: a product shot's. At the
    /// ground, or under the scene when it reaches below.
    pub ground: bool,
    /// A baked gas the paths travel through: group 1.
    volume: PtVolumeGpu,
    /// The rays are asked of the device's ray tracing, not walked in the
    /// shader.
    hardware: bool,
}

/// One gas's parameters, as `PtVolume` in the shader reads them.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct PtVolumeItem {
    lo: [f32; 4],
    hi: [f32; 4],
    color: [f32; 4],
    detail: [f32; 4],
    cells: [u32; 4],
}

/// The most gases the path tracer travels through at once.
pub const MAX_VOLUMES: usize = 4;

/// `PtVolumes` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct PtVolumeParams {
    info: [u32; 4],
    atlas: [f32; 4],
    items: [PtVolumeItem; MAX_VOLUMES],
}

/// The cells a flame can glow in and each one's running share of the glow,
/// for aiming at it: `[index as bits, share]`. A cell counts when any cell
/// beside it burns — the grid is read smoothly, so the glow reaches a cell
/// past the last that burns, and a cell never chosen where there is glow
/// would be light never found.
#[cfg(test)]
fn flame_cells(flame: &[u8], cells: [u32; 3]) -> Vec<[f32; 2]> {
    running_shares(flame_weights(flame, cells, 0))
}

/// Weighted cells as `[index as bits, running share]`, the last share one.
fn running_shares(weights: Vec<(u32, f64)>) -> Vec<[f32; 2]> {
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let mut running = 0.0;
    weights
        .into_iter()
        .map(|(i, w)| {
            running += w;
            [f32::from_bits(i), (running / total) as f32]
        })
        .collect()
}

/// Each cell a flame can glow in, tagged with which gas in its top three
/// bits, and how brightly: see `flame_cells`.
fn flame_weights(flame: &[u8], cells: [u32; 3], gas: u32) -> Vec<(u32, f64)> {
    let [nx, ny, nz] = cells.map(|c| c as usize);
    let at = |i: usize, j: usize, k: usize| flame[(k * ny + j) * nx + i];
    let mut out = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let mut most = 0u8;
                for c in 0..27 {
                    let (a, b, d) = (c % 3, (c / 3) % 3, c / 9);
                    let (x, y, z) = (i + a, j + b, k + d);
                    if x >= 1 && y >= 1 && z >= 1 && x <= nx && y <= ny && z <= nz {
                        most = most.max(at(x - 1, y - 1, z - 1));
                    }
                }
                if most > 0 {
                    let f = most as f64 / 255.0;
                    // As bright as it glows: the ramp brightens with the flame.
                    out.push((((k * ny + j) * nx + i) as u32 | (gas << 29), f * f));
                }
            }
        }
    }
    out
}

struct PtVolumeGpu {
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Every gas's grid, stacked along Z.
    grid: wgpu::Texture,
    /// The stack's size, and each gas's cells in it.
    atlas: [u32; 3],
    cells: Vec<[u32; 3]>,
    /// What each gas's bytes were: one that is the same is not uploaded
    /// again.
    keys: Vec<usize>,
    bind: wgpu::BindGroup,
    last: PtVolumeParams,
    /// `flame_cells`, and how many.
    flame: wgpu::Buffer,
    flame_count: u32,
}

impl PtVolumeGpu {
    fn new(device: &wgpu::Device) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("path trace volume"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: c,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: c,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: c,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: c,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("path trace volume params"),
            size: std::mem::size_of::<PtVolumeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("path trace volume"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // An empty grid of one cell until there is a gas: the group is
        // always bound, and `lo.w` of zero says there is nothing in it.
        let atlas = [1, 1, 1];
        let grid = Self::grid(device, atlas);
        let flame = Self::flame_buffer(device, &[]);
        let bind = Self::bind(device, &layout, &params, &grid, &sampler, &flame);
        Self {
            layout,
            params,
            sampler,
            grid,
            atlas,
            cells: Vec::new(),
            keys: Vec::new(),
            bind,
            last: PtVolumeParams::default(),
            flame,
            flame_count: 0,
        }
    }

    fn flame_buffer(device: &wgpu::Device, cells: &[[f32; 2]]) -> wgpu::Buffer {
        let one = [[0.0f32; 2]];
        let cells = if cells.is_empty() { &one[..] } else { cells };
        wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("path trace flame cells"),
                contents: bytemuck::cast_slice(cells),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )
    }

    fn grid(device: &wgpu::Device, cells: [u32; 3]) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path trace volume grid"),
            size: wgpu::Extent3d {
                width: cells[0],
                height: cells[1],
                depth_or_array_layers: cells[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        grid: &wgpu::Texture,
        sampler: &wgpu::Sampler,
        flame: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let view = grid.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path trace volume"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: flame.as_entire_binding(),
                },
            ],
        })
    }
}

impl PathTracer {
    /// `shading` is the layout of the shading library's group: the tracer
    /// reads the lights and the environment through it.
    pub fn new(device: &wgpu::Device, shading: &wgpu::BindGroupLayout) -> Self {
        Self::new_tracing(device, shading, ray_queries_on(device))
    }

    /// As [`PathTracer::new`], told whether to trace with the device's ray
    /// queries — which it must have — or the shader's own walk.
    pub fn new_tracing(
        device: &wgpu::Device,
        shading: &wgpu::BindGroupLayout,
        hardware: bool,
    ) -> Self {
        let hardware = hardware
            && device
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        let storage = |binding: u32, read_only: bool, visibility: wgpu::ShaderStages| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let uniform = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let c = wgpu::ShaderStages::COMPUTE;
        let volume = PtVolumeGpu::new(device);
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("path trace scene"),
            entries: &[
                uniform(0, c),
                storage(1, true, c),
                storage(2, true, c),
                storage(3, true, c),
                storage(4, true, c),
                storage(5, true, c),
                storage(6, true, c),
                storage(7, false, c),
                storage(8, true, c),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: c,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: c,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: c,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage(12, true, c),
                storage(13, true, c),
                storage(14, true, c),
                storage(15, true, c),
                storage(16, true, c),
                storage(17, false, c),
                storage(18, true, c),
                storage(19, true, c),
            ]
            .into_iter()
            .chain(hardware.then_some(wgpu::BindGroupLayoutEntry {
                binding: 20,
                visibility: c,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            }))
            .collect::<Vec<_>>(),
        });
        let trace_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("path trace"),
            bind_group_layouts: &[Some(&scene_layout), Some(&volume.layout), Some(shading)],
            immediate_size: 0,
        });
        let trace = Self::trace_pipeline(device, &trace_layout, &source("", 0, hardware));

        let f = wgpu::ShaderStages::FRAGMENT;
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("path trace resolve"),
            entries: &[storage(0, true, f), uniform(1, f), storage(2, true, f)],
        });
        let resolve_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("path trace resolve"),
            source: wgpu::ShaderSource::Wgsl(resolve_source().into()),
        });
        let resolve_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("path trace resolve"),
                bind_group_layouts: &[Some(&resolve_layout)],
                immediate_size: 0,
            });
        let resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("path trace resolve"),
            layout: Some(&resolve_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &resolve_module,
                entry_point: Some("resolve_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &resolve_module,
                entry_point: Some("resolve_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::display::HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let buffer = |label, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let params = buffer(
            "path trace params",
            std::mem::size_of::<Params>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let resolve_params = buffer(
            "path trace resolve params",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let per_pixel = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let sum = buffer("path trace sum", 32, per_pixel);
        let aux = buffer("path trace aux", 48, per_pixel);
        let prev = buffer("path trace previous", 80, per_pixel);
        let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("path trace images"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let env_map = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("path trace environment map"),
                contents: &[0u8; 16],
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        Self {
            env_map,
            env_size: None,
            image_sampler,
            trace,
            trace_layout,
            trace_code: (String::new(), 0),
            resolve,
            scene_layout,
            resolve_layout,
            params,
            resolve_params,
            scene: None,
            sum,
            sum_pixels: 1,
            aux,
            prev,
            previous: None,
            history_ok: false,
            reproject_now: false,
            scene_bind: None,
            resolve_bind: None,
            size: (0, 0),
            sample: 0,
            ground: false,
            noise_threshold: 0.01,
            max_bounces: 4,
            volume,
            hardware,
        }
    }

    /// Whether the rays are asked of the device's ray tracing.
    pub fn traces_in_hardware(&self) -> bool {
        self.hardware
    }

    /// The gases the paths travel through — the first `MAX_VOLUMES` — or
    /// none. A grid is uploaded when it is a new one; whether anything
    /// changed is the answer, for the caller to start the sum again.
    pub fn set_volumes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volumes: &[crate::volume::VolumeData<'_>],
    ) -> bool {
        let v = &mut self.volume;
        let volumes: Vec<&crate::volume::VolumeData<'_>> = volumes
            .iter()
            .filter(|d| {
                let count = d.cells.iter().map(|c| *c as usize).product::<usize>();
                !d.cells.contains(&0) && d.smoke.len() == count && d.flame.len() == count
            })
            .take(MAX_VOLUMES)
            .collect();
        let cells: Vec<[u32; 3]> = volumes.iter().map(|d| d.cells).collect();
        let mut changed = false;
        // A new stack when the gases' sizes change: as wide and deep as the
        // widest, as tall as all of them.
        if cells != v.cells {
            v.atlas = [
                cells.iter().map(|c| c[0]).max().unwrap_or(1).max(1),
                cells.iter().map(|c| c[1]).max().unwrap_or(1).max(1),
                cells.iter().map(|c| c[2]).sum::<u32>().max(1),
            ];
            v.grid = PtVolumeGpu::grid(device, v.atlas);
            v.cells = cells;
            v.keys = vec![usize::MAX; volumes.len()];
            changed = true;
        }
        let mut layer = 0;
        let mut regrid = changed;
        let mut first_layer = Vec::with_capacity(volumes.len());
        for (i, data) in volumes.iter().enumerate() {
            first_layer.push(layer);
            if v.keys[i] != data.key {
                let mut bytes = Vec::with_capacity(data.smoke.len() * 2);
                for (s, f) in data.smoke.iter().zip(data.flame) {
                    bytes.push(*s);
                    bytes.push(*f);
                }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &v.grid,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(data.cells[0] * 2),
                        rows_per_image: Some(data.cells[1]),
                    },
                    wgpu::Extent3d {
                        width: data.cells[0],
                        height: data.cells[1],
                        depth_or_array_layers: data.cells[2],
                    },
                );
                v.keys[i] = data.key;
                regrid = true;
            }
            layer += data.cells[2];
        }
        if regrid {
            let weights = volumes
                .iter()
                .enumerate()
                .flat_map(|(i, d)| flame_weights(d.flame, d.cells, i as u32))
                .collect();
            let shares = running_shares(weights);
            v.flame_count = shares.len() as u32;
            v.flame = PtVolumeGpu::flame_buffer(device, &shares);
            v.bind = PtVolumeGpu::bind(device, &v.layout, &v.params, &v.grid, &v.sampler, &v.flame);
            changed = true;
        }
        // As thick as the raster view draws it, so the two agree.
        let thick = crate::volume::THICKNESS;
        let mut params = PtVolumeParams {
            info: [volumes.len() as u32, v.flame_count, 0, 0],
            atlas: [v.atlas[0] as f32, v.atlas[1] as f32, v.atlas[2] as f32, 0.0],
            ..Default::default()
        };
        for (i, data) in volumes.iter().enumerate() {
            let size = data.max - data.min;
            let cell = (size.x / data.cells[0] as f64)
                .min(size.y / data.cells[1] as f64)
                .min(size.z / data.cells[2] as f64);
            let detail = data.detail.clamp(0.0, 1.0);
            params.items[i] = PtVolumeItem {
                lo: [data.min.x as f32, data.min.y as f32, data.min.z as f32, 1.0],
                hi: [
                    data.max.x as f32,
                    data.max.y as f32,
                    data.max.z as f32,
                    thick,
                ],
                color: [
                    data.smoke_color[0],
                    data.smoke_color[1],
                    data.smoke_color[2],
                    crate::volume::GLOW,
                ],
                detail: [
                    detail,
                    (1.0 / (cell * 2.0)) as f32,
                    data.time,
                    thick * (1.0 + detail),
                ],
                cells: [data.cells[0], data.cells[1], data.cells[2], first_layer[i]],
            };
        }
        if params != v.last {
            queue.write_buffer(&v.params, 0, bytemuck::bytes_of(&params));
            v.last = params;
            changed = true;
        }
        changed
    }

    fn trace_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        source: &str,
    ) -> wgpu::ComputePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("path trace"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("path trace"),
            layout: Some(layout),
            module: &module,
            entry_point: Some("pt_main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Whether a scene has been handed over.
    pub fn has_scene(&self) -> bool {
        self.scene.is_some()
    }

    /// Hands over the scene: its meshes, placed, and the surfaces they name.
    /// Builds the BVH, which is the part that costs; done when the scene
    /// changes, not when the camera does.
    ///
    /// Each image goes to a layer of the array its kind goes in, in the order
    /// given, each array counting its own: what `materials` was told.
    #[allow(clippy::too_many_arguments)]
    pub fn set_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[PtMesh<'_>],
        surfaces: &[PtSurface],
        images: &[PtImage<'_>],
        materials: &PtMaterials<'_>,
    ) {
        if (materials.code, materials.count) != (self.trace_code.0.as_str(), self.trace_code.1) {
            self.trace = Self::trace_pipeline(
                device,
                &self.trace_layout,
                &source(materials.code, materials.count, self.hardware),
            );
            self.trace_code = (materials.code.to_string(), materials.count);
        }
        let layout = lay_out(meshes);
        let indices: Vec<[u32; 3]> = layout
            .triangles
            .iter()
            .map(|t| [t[0], t[1], t[2]])
            .collect();
        // A tree per stored run, over its own triangles.
        let trees: Vec<kreate_math::Bvh> = layout
            .stored
            .iter()
            .map(|run| {
                let first = run.first_triangle as usize;
                kreate_math::Bvh::build(
                    &layout.positions,
                    &indices[first..first + run.triangles as usize],
                )
            })
            .collect();
        // The placings a ray can meet: the ones whose run has a tree.
        let placings: Vec<(usize, DAffine3)> = layout
            .placings
            .iter()
            .copied()
            .filter(|&(run, _)| !trees[run].is_empty())
            .collect();
        let layout = Layout { placings, ..layout };
        let top = kreate_math::Bvh::build_boxes(
            &layout
                .placings
                .iter()
                .map(|&(run, to)| placed_box(trees[run].bounds(), to))
                .collect::<Vec<_>>(),
        );
        let node = |(b, first, count): (kreate_math::Aabb, u32, u32)| GpuNode {
            min: b.min.to_array(),
            first,
            max: b.max.to_array(),
            count,
        };
        // The placings' tree first, its root the walk's start; then each
        // run's, its links moved to where it lands.
        let mut nodes: Vec<GpuNode> = top.flat_nodes().into_iter().map(node).collect();
        let mut order = top.order().to_vec();
        // An empty scene still binds: one node that nothing enters.
        if nodes.is_empty() {
            nodes.push(GpuNode {
                min: [1.0; 3],
                first: 0,
                max: [-1.0; 3],
                count: 0,
            });
        }
        let mut roots = Vec::with_capacity(trees.len());
        for (run, tree) in layout.stored.iter().zip(&trees) {
            let (node_base, order_base) = (nodes.len() as u32, order.len() as u32);
            roots.push(node_base);
            nodes.extend(tree.flat_nodes().into_iter().map(|n| {
                let mut n = node(n);
                n.first += if n.count > 0 { order_base } else { node_base };
                n
            }));
            order.extend(tree.order().iter().map(|i| i + run.first_triangle));
        }
        if order.is_empty() {
            order.push(0);
        }
        let mut instances: Vec<GpuInstance> = layout
            .placings
            .iter()
            .map(|&(run, to)| GpuInstance {
                world: rows(to),
                local: rows(to.inverse()),
                root: roots[run],
                first: layout.stored[run].first_triangle,
                pad: [0; 2],
            })
            .collect();
        if instances.is_empty() {
            instances.push(GpuInstance {
                world: rows(DAffine3::IDENTITY),
                local: rows(DAffine3::IDENTITY),
                root: 0,
                first: 0,
                pad: [0; 2],
            });
        }
        // Each image to a layer of the array its kind goes in, and each
        // surface told which.
        let mut colour_layers: Vec<Vec<u8>> = Vec::new();
        let mut data_layers: Vec<Vec<u8>> = Vec::new();
        let mut layer_of: Vec<i32> = Vec::with_capacity(images.len());
        for image in images {
            let pixels = resample(image);
            if image.color {
                layer_of.push(colour_layers.len() as i32);
                colour_layers.push(pixels);
            } else {
                layer_of.push(data_layers.len() as i32);
                data_layers.push(pixels);
            }
        }
        let layer =
            |i: Option<usize>| i.and_then(|i| layer_of.get(i).copied()).unwrap_or(-1) as f32;
        let mut surface_data: Vec<[f32; 4]> = surfaces
            .iter()
            .flat_map(|s| {
                [
                    [
                        s.base_color[0],
                        s.base_color[1],
                        s.base_color[2],
                        s.metallic,
                    ],
                    [
                        s.roughness,
                        layer(s.base_color_image),
                        layer(s.metallic_roughness_image),
                        s.transmission,
                    ],
                    [s.emission[0], s.emission[1], s.emission[2], s.ior.max(1.0)],
                    [
                        if s.one_sided { 1.0 } else { 0.0 },
                        // One past the material, so zero is none.
                        s.material.map_or(0.0, |m| m as f32 + 1.0),
                        s.alpha_clip.clamp(0.0, 1.0),
                        0.0,
                    ],
                ]
            })
            .collect();
        let array = |label, format, layers: &[Vec<u8>]| {
            let count = layers.len().max(1) as u32;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: IMAGE_SIZE,
                    height: IMAGE_SIZE,
                    depth_or_array_layers: count,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            for (i, pixels) in layers.iter().enumerate() {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: i as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    pixels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(IMAGE_SIZE * 4),
                        rows_per_image: Some(IMAGE_SIZE),
                    },
                    wgpu::Extent3d {
                        width: IMAGE_SIZE,
                        height: IMAGE_SIZE,
                        depth_or_array_layers: 1,
                    },
                );
            }
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let colour_images = array(
            "path trace colour images",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &colour_layers,
        );
        let data_images = array(
            "path trace data images",
            wgpu::TextureFormat::Rgba8Unorm,
            &data_layers,
        );
        if surface_data.is_empty() {
            // Three slots a surface, as the shader reads them.
            surface_data.extend([
                [0.8, 0.8, 0.8, 0.0],
                [0.5, -1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.5],
                [0.0; 4],
            ]);
        }
        let (emitter_data, emitter_total) = emitters_of(&layout, surfaces);
        let emitter_count = emitter_data.len() as u32;
        // The lowest point, for where the floor goes when the scene reaches
        // below the ground.
        let lowest = layout
            .placings
            .iter()
            .flat_map(|&(run, to)| {
                let run = layout.stored[run];
                let row = to.matrix3.row(2);
                layout.positions
                    [run.first_vertex as usize..(run.first_vertex + run.vertices) as usize]
                    .iter()
                    .map(move |p| (row.dot(p.as_dvec3()) + to.translation.z) as f32)
            })
            .fold(f32::INFINITY, f32::min);
        let hardware = self
            .hardware
            .then(|| build_hardware(device, queue, &layout));
        let Layout {
            local,
            uvs,
            colors,
            positions,
            normals,
            triangles,
            ..
        } = layout;
        let positions: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 1.0]).collect();
        let upload = |label, bytes: &[u8]| {
            let bytes = if bytes.is_empty() {
                &[0u8; 16][..]
            } else {
                bytes
            };
            wgpu::util::DeviceExt::create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                },
            )
        };
        self.scene = Some(Scene {
            uvs: upload("path trace uvs", bytemuck::cast_slice(&uvs)),
            colors: upload(
                "path trace colours",
                bytemuck::cast_slice(if colors.is_empty() {
                    &[u32::MAX][..]
                } else {
                    &colors
                }),
            ),
            colour_images,
            data_images,
            positions: upload("path trace positions", bytemuck::cast_slice(&positions)),
            normals: upload("path trace normals", bytemuck::cast_slice(&normals)),
            triangles: upload("path trace triangles", bytemuck::cast_slice(&triangles)),
            nodes: upload("path trace nodes", bytemuck::cast_slice(&nodes)),
            order: upload("path trace order", bytemuck::cast_slice(&order)),
            surfaces: upload("path trace surfaces", bytemuck::cast_slice(&surface_data)),
            emitters: upload("path trace emitters", bytemuck::cast_slice(&emitter_data)),
            values: upload(
                "path trace material values",
                bytemuck::cast_slice(materials.values),
            ),
            local: upload("path trace local", bytemuck::cast_slice(&local)),
            instances: upload("path trace placings", bytemuck::cast_slice(&instances)),
            hardware,
            emitter_count,
            emitter_total,
            lowest: if lowest.is_finite() { lowest } else { 0.0 },
        });
        self.scene_bind = None;
        self.sample = 0;
        self.history_ok = false;
    }

    /// Nothing from before carries over to the next view: the scene's
    /// lights or materials changed, and the picture with them.
    pub fn forget_history(&mut self) {
        self.history_ok = false;
    }

    /// The environment's importance map, or none for a flat world.
    pub fn set_environment(
        &mut self,
        device: &wgpu::Device,
        map: Option<&crate::environment::ImportanceMap>,
    ) {
        let data: &[f32] = map.map_or(&[0.0; 4][..], |m| &m.cdf);
        self.env_map = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("path trace environment map"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        self.env_size = map.map(|m| (m.columns, m.rows));
        self.scene_bind = None;
        self.sample = 0;
        self.history_ok = false;
    }

    /// The per-pixel sums, for the denoiser to read the surface colour and
    /// the noise from; see `pt_sum` in the shader.
    pub fn sums(&self) -> &wgpu::Buffer {
        &self.sum
    }

    /// Samples in the sum so far.
    pub fn samples(&self) -> u32 {
        self.sample
    }

    /// This frame's view. `projection` and `view` are the rasteriser's,
    /// unjittered; `eye` is where the camera is in world space.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        projection: Mat4,
        view: Mat4,
        eye: glam::DVec3,
        region: (u32, u32),
        light_count: u32,
        sample: u32,
        lens: Option<Lens>,
    ) {
        let pixels = region.0.max(1) * region.1.max(1);
        if pixels > self.sum_pixels || self.size != region {
            if pixels > self.sum_pixels {
                let per_pixel = wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST;
                let buffer = |label, vec4s: u64| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: pixels as u64 * vec4s * 16,
                        usage: per_pixel,
                        mapped_at_creation: false,
                    })
                };
                // Two vec4s a pixel: see `pt_sum` in the shader; three for
                // `pt_aux`; and both again for the last frame's.
                self.sum = buffer("path trace sum", 2);
                self.aux = buffer("path trace aux", 3);
                self.prev = buffer("path trace previous", 5);
                self.sum_pixels = pixels;
            }
            self.size = region;
            self.scene_bind = None;
            self.resolve_bind = None;
            self.sample = 0;
            self.history_ok = false;
        }
        if let (None, Some(scene)) = (&self.scene_bind, &self.scene) {
            let mut entries: Vec<wgpu::BindGroupEntry> = [
                &self.params,
                &scene.positions,
                &scene.normals,
                &scene.triangles,
                &scene.nodes,
                &scene.order,
                &scene.surfaces,
                &self.sum,
                &scene.uvs,
            ]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(&scene.colour_images),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(&scene.data_images),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::Sampler(&self.image_sampler),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 12,
                resource: scene.emitters.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 13,
                resource: self.env_map.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 14,
                resource: scene.values.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 15,
                resource: scene.local.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 16,
                resource: scene.colors.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 17,
                resource: self.aux.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 18,
                resource: self.prev.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 19,
                resource: scene.instances.as_entire_binding(),
            });
            if let Some(hardware) = &scene.hardware {
                entries.push(wgpu::BindGroupEntry {
                    binding: 20,
                    resource: hardware.top.as_binding(),
                });
            }
            self.scene_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("path trace scene"),
                layout: &self.scene_layout,
                entries: &entries,
            }));
        }
        if self.resolve_bind.is_none() {
            self.resolve_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("path trace resolve"),
                layout: &self.resolve_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.sum.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.resolve_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.aux.as_entire_binding(),
                    },
                ],
            }));
        }
        self.sample = sample;
        // A first sample of a moved view, the scene as it was: each pixel
        // finds its first hit on the last frame's picture and carries that
        // pixel's mean over, while its own samples are few.
        let here = (projection * view, eye);
        let before = self.previous.filter(|_| self.history_ok);
        self.reproject_now = sample <= 1 && before.is_some();
        self.previous = Some(here);
        self.history_ok = true;
        let (prev_view_proj, prev_eye) = before.unwrap_or(here);
        let moved = (prev_eye - eye).as_vec3();
        // The view's rows are the camera's axes in the world.
        let world = view.inverse();
        let (right, up) = (world.x_axis, world.y_axis);
        let lens = lens.filter(|l| l.radius > 0.0 && l.focus > 0.0);
        let p = Params {
            inv_view_proj: (projection * view).inverse().to_cols_array_2d(),
            eye: [eye.x as f32, eye.y as f32, eye.z as f32, 0.0],
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            lens: lens.map_or([0.0; 4], |l| [l.radius, l.focus, 0.0, 0.0]),
            size: [region.0, region.1],
            sample: self.sample,
            light_count,
            max_bounces: self.max_bounces.clamp(1, 64),
            emitter_count: self.scene.as_ref().map_or(0, |s| s.emitter_count),
            emitter_total: self.scene.as_ref().map_or(0.0, |s| s.emitter_total),
            noise_threshold: self.noise_threshold.max(0.0),
            env: match self.env_size {
                Some((c, r)) => [c, r, 1, 0],
                None => [0, 0, 0, 0],
            },
            prev_view_proj: prev_view_proj.to_cols_array_2d(),
            // Where the last eye is from this one: the shader's hits are
            // this eye's plus an offset, kept small for f32.
            prev_eye: [moved.x, moved.y, moved.z, 0.0],
            reproject: [u32::from(self.reproject_now), HISTORY, 0, 0],
            ground: [
                self.scene.as_ref().map_or(0.0, |s| s.lowest.min(0.0)),
                f32::from(u8::from(self.ground && self.scene.is_some())),
                FLOOR_REFLECTION,
                0.0,
            ],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        queue.write_buffer(
            &self.resolve_params,
            0,
            bytemuck::cast_slice(&[region.0, region.1, 0, 0]),
        );
    }

    /// What the samples have gathered, read back for a finished render's
    /// denoiser: each pixel's mean light, its first surface's colour and
    /// normal — and the sums as they are, for [`PathTracer::replace_mean`].
    pub fn read_gathered(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Gathered, String> {
        let (w, h) = self.size;
        let pixels = w as usize * h as usize;
        if pixels == 0 || self.scene_bind.is_none() {
            return Err("nothing has been traced".into());
        }
        let sums: Vec<[f32; 4]> =
            bytemuck::cast_slice(&read_buffer(device, queue, &self.sum, pixels as u64 * 32)?)
                .to_vec();
        let aux: Vec<[f32; 4]> =
            bytemuck::cast_slice(&read_buffer(device, queue, &self.aux, pixels as u64 * 48)?)
                .to_vec();
        let mut color = Vec::with_capacity(pixels);
        let mut albedo = Vec::with_capacity(pixels);
        let mut normal = Vec::with_capacity(pixels);
        for p in 0..pixels {
            let (s, stats) = (sums[p * 2], sums[p * 2 + 1]);
            // The resolve pass's own arithmetic: the carried mean counts for
            // the frames it stands for, less this pixel's own.
            let c = aux[p * 3 + 2];
            let k = (c[3] - s[3]).max(0.0);
            let d = (s[3] + k).max(1.0);
            color.push([0, 1, 2].map(|i| (s[i] + c[i] * k) / d));
            let n = s[3].max(1.0);
            albedo.push(if s[3] > 0.0 {
                [0, 1, 2].map(|i| stats[i] / n)
            } else {
                [1.0; 3]
            });
            let at = aux[p * 3 + 1];
            normal.push([at[0], at[1], at[2]]);
        }
        Ok(Gathered {
            width: w,
            height: h,
            color,
            albedo,
            normal,
            sums,
            aux,
        })
    }

    /// Puts `mean` in place of what was gathered, so that the next resolve
    /// ([`PathTracer::resolve`]) shows it: the sums are rewritten so that
    /// divided out, carried mean and all, they come to `mean`. The counts
    /// are kept, so nothing downstream sees a picture with no samples.
    pub fn replace_mean(&self, queue: &wgpu::Queue, gathered: &Gathered, mean: &[[f32; 3]]) {
        if gathered.width != self.size.0
            || gathered.height != self.size.1
            || mean.len() != gathered.color.len()
        {
            return;
        }
        let mut sums = gathered.sums.clone();
        for (p, m) in mean.iter().enumerate() {
            let s = sums[p * 2];
            let c = gathered.aux[p * 3 + 2];
            let k = (c[3] - s[3]).max(0.0);
            let d = (s[3] + k).max(1.0);
            for i in 0..3 {
                sums[p * 2][i] = m[i] * d - c[i] * k;
            }
        }
        queue.write_buffer(&self.sum, 0, bytemuck::cast_slice(&sums));
    }

    /// The mean onto `picture` again, with no sample traced: after
    /// [`PathTracer::replace_mean`].
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder, picture: &wgpu::TextureView) {
        let Some(resolve_bind) = &self.resolve_bind else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("path trace resolve again"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: picture,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, self.size.0 as f32, self.size.1 as f32, 0.0, 1.0);
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, resolve_bind, &[]);
        pass.draw(0..3, 0..1);
    }

    /// One sample per pixel into the sum, then the mean onto `picture`.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shading: &wgpu::BindGroup,
        picture: &wgpu::TextureView,
    ) {
        let (Some(scene_bind), Some(resolve_bind)) = (&self.scene_bind, &self.resolve_bind) else {
            return;
        };
        if self.reproject_now {
            // The last frame's, before this sample writes over them.
            let pixels = self.size.0 as u64 * self.size.1 as u64;
            encoder.copy_buffer_to_buffer(&self.sum, 0, &self.prev, 0, pixels * 32);
            encoder.copy_buffer_to_buffer(&self.aux, 0, &self.prev, pixels * 32, pixels * 48);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("path trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace);
            pass.set_bind_group(0, scene_bind, &[]);
            pass.set_bind_group(1, &self.volume.bind, &[]);
            pass.set_bind_group(2, shading, &[]);
            pass.dispatch_workgroups(self.size.0.div_ceil(8), self.size.1.div_ceil(8), 1);
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("path trace resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: picture,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, self.size.0 as f32, self.size.1 as f32, 0.0, 1.0);
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, resolve_bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// What the path tracer has gathered, read back: see
/// [`PathTracer::read_gathered`].
pub struct Gathered {
    pub width: u32,
    pub height: u32,
    /// Each pixel's mean light, scene-linear and before the exposure.
    pub color: Vec<[f32; 3]>,
    /// Each pixel's first surface's colour; white where nothing was met.
    pub albedo: Vec<[f32; 3]>,
    /// Each pixel's first surface's normal, in the world; zero where nothing.
    pub normal: Vec<[f32; 3]>,
    sums: Vec<[f32; 4]>,
    aux: Vec<[f32; 4]>,
}

/// The first `bytes` of a storage buffer, on the CPU.
fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<u8>, String> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("path trace readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("path trace readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("the GPU did not finish: {e:?}"))?;
    let out = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    Ok(out)
}

#[cfg(test)]
mod flame_tests {
    use super::flame_cells;

    #[test]
    fn a_flame_is_aimed_at_by_its_cells_and_their_neighbours_by_how_bright() {
        // One cell burning in the middle of a 4×4×4 grid, one dimly in a
        // corner.
        let mut flame = vec![0u8; 64];
        let index = |i: usize, j: usize, k: usize| (k * 4 + j) * 4 + i;
        flame[index(1, 1, 1)] = 255;
        flame[index(3, 3, 3)] = 51;
        let cells = flame_cells(&flame, [4, 4, 4]);
        let chosen: Vec<u32> = cells.iter().map(|c| c[0].to_bits()).collect();
        // The bright cell's 27 and the dim one's 8, with no cell twice.
        assert!(chosen.contains(&(index(1, 1, 1) as u32)));
        assert!(
            chosen.contains(&(index(2, 2, 2) as u32)),
            "a neighbour it glows into"
        );
        assert!(chosen.contains(&(index(3, 3, 3) as u32)));
        assert!(
            !chosen.contains(&(index(0, 0, 3) as u32)),
            "too far from either"
        );
        // Running shares, ending at all of it.
        assert!(cells.windows(2).all(|w| w[1][1] >= w[0][1]));
        assert!((cells.last().unwrap()[1] - 1.0).abs() < 1e-6);
        // The bright neighbourhood takes nearly all of it.
        let share = |i: usize| {
            let at = cells
                .iter()
                .position(|c| c[0].to_bits() == i as u32)
                .unwrap();
            cells[at][1] - if at > 0 { cells[at - 1][1] } else { 0.0 }
        };
        assert!(share(index(1, 1, 1)) > share(index(3, 3, 3)) * 20.0);
        assert!(flame_cells(&[0u8; 8], [2, 2, 2]).is_empty());
    }
}

#[cfg(test)]
mod placing_tests {
    use super::*;

    fn mesh<'a>(
        positions: &'a [[f32; 3]],
        normals: &'a [Vec3],
        indices: &'a [u32],
        transform: DAffine3,
    ) -> PtMesh<'a> {
        PtMesh {
            positions,
            uvs: &[],
            normals,
            colors: &[],
            indices,
            transform,
            surface: 0,
            runs: &[],
        }
    }

    /// The same slices handed over twice are one stored run, placed twice;
    /// a copy of them is a mesh of its own, moved into the world; and a
    /// glowing triangle placed at twice the size glows four times as much.
    #[test]
    fn a_mesh_handed_over_twice_is_stored_once_and_placed_twice() {
        let positions = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let copy = positions;
        let normals = [Vec3::Z; 3];
        let indices = [0u32, 1, 2];
        let away = DAffine3::from_translation(glam::DVec3::new(5.0, 0.0, -2.0));
        let bigger = DAffine3::from_scale(glam::DVec3::splat(2.0));
        let meshes = [
            mesh(&positions, &normals, &indices, away),
            mesh(&positions, &normals, &indices, bigger),
            mesh(&copy, &normals, &indices, away),
        ];
        let layout = lay_out(&meshes);
        assert_eq!(layout.stored.len(), 2, "the world's run and the placed one");
        assert_eq!(layout.triangles.len(), 2, "each triangle stored once");
        assert_eq!(layout.placings.len(), 3);
        // The copy went into the world, moved.
        assert_eq!(layout.positions[0], Vec3::new(5.0, 0.0, -2.0));
        // The placed one is kept in its own space.
        assert_eq!(layout.positions[3], Vec3::ZERO);
        let glow = PtSurface {
            emission: [1.0; 3],
            ..Default::default()
        };
        let (emitters, total) = emitters_of(&layout, &[glow]);
        assert_eq!(emitters.len(), 3);
        assert!((total - 0.5 * (1.0 + 1.0 + 4.0)).abs() < 1e-5, "{total}");
    }

    /// A placing that flattens its mesh cannot be undone, so the rays could
    /// not be carried into it: such a mesh goes into the world instead.
    #[test]
    fn a_flattened_copy_is_moved_into_the_world() {
        let positions = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = [Vec3::Z; 3];
        let indices = [0u32, 1, 2];
        let flat = DAffine3::from_scale(glam::DVec3::new(1.0, 1.0, 0.0));
        let meshes = [
            mesh(&positions, &normals, &indices, flat),
            mesh(&positions, &normals, &indices, flat),
        ];
        let layout = lay_out(&meshes);
        assert_eq!(layout.stored.len(), 1);
        assert_eq!(layout.placings, vec![(0, DAffine3::IDENTITY)]);
        assert_eq!(layout.triangles.len(), 2);
    }
}
