//! An environment, prefiltered so that shading it costs a lookup.
//!
//! §2.2's IBL step. The world has been one colour in every direction until now,
//! and that case is *exact*: under a uniform environment the prefiltered
//! radiance is that colour whatever the roughness, so the split-sum specular is
//! the DFG table straight out and the diffuse integrates to `albedo × L`. A
//! white surface under `L` returns `L`, and §7.5's furnace test says so.
//!
//! An image is the case that is not exact, and this is what makes it cheap
//! enough to be wrong by a knowable amount rather than by an unknown one.
//!
//! # Two halves, because a surface has two lobes
//!
//! - **Specular** reads a prefiltered cube map: mip `m` holds the environment
//!   convolved with the GGX lobe for the roughness that mip stands for. The
//!   other factor of the split sum is [`crate::brdf`]'s DFG table, which
//!   already exists and was built with the same importance sampling — the two
//!   halves have to be sampling the same distribution or their product is not
//!   the integral either came from.
//! - **Diffuse** reads nine numbers. Irradiance over a hemisphere is so smooth
//!   that second-order spherical harmonics reproduce it to within about one
//!   percent, which is the standard result and the reason nobody stores an
//!   irradiance cube map. Nine `vec3`s in a uniform, against a texture and a
//!   sampler.
//!
//! # A cube rather than the equirectangular image
//!
//! Sampling the image directly would save a pass and a texture, and it would
//! prefilter badly: an equirectangular image's texels cover wildly different
//! solid angles, pinching to nothing at the poles, so a filter kernel measured
//! in texels is a different filter at every latitude. A cube's faces are
//! within a factor of about three of each other. The poles are where a sky's
//! sun and a studio's softbox usually are.
//!
//! # What is approximate, said out loud
//!
//! The prefilter assumes the normal, the view and the reflection are the same
//! direction. That is exact head on and wrong at a grazing angle, where a real
//! reflection stretches along the surface — the streak this cannot produce.
//! Every real-time renderer makes the same trade; the alternative is a table
//! with a viewing angle in it.

use glam::{Vec2, Vec3};

/// The edge of one cube face at mip zero.
///
/// Small, and it can be: what is stored here is *blurred* by construction —
/// mip zero is the only sharp level and it is only ever read by the prefilter
/// and by the background. A sharper environment would cost six times its area
/// at every level and show up nowhere except in a mirror, which is the one
/// case §2.5's path tracer will do properly anyway.
pub const FACE: u32 = 128;

/// How many roughness levels the cube holds.
///
/// Mip `m` stands for roughness `m / (MIPS - 1)`, so the ends are pinned: mip
/// zero is the environment itself and the last mip is roughness one. Six levels
/// takes the smallest face to 4×4, which is where a lobe covering the whole
/// hemisphere has nothing left to average.
pub const MIPS: u32 = 6;

/// How many directions each prefiltered texel averages.
///
/// The number where the noise stops being visible against the mip selection
/// that goes with it. Fewer and a bright sun becomes a scatter of bright
/// texels; more costs build time and shows nothing.
pub const SAMPLES: u32 = 128;

/// The roughness mip `level` stands for.
pub fn roughness_of(level: u32) -> f32 {
    if MIPS <= 1 {
        return 0.0;
    }
    level as f32 / (MIPS - 1) as f32
}

/// The direction a cube texel stands for.
///
/// The mapping a cube sampler inverts — *that* one, not any self-consistent
/// one. A face written with the axes it would prefer comes back mirrored, and
/// a mirrored sky still looks like a sky, so nothing but this arithmetic can
/// catch it.
///
/// `s` runs across the face and `t` runs *down* it, both from zero to one.
pub fn cube_direction(face: u32, s: f32, t: f32) -> Vec3 {
    let a = 2.0 * s - 1.0;
    let b = 2.0 * t - 1.0;
    let v = match face {
        0 => Vec3::new(1.0, -b, -a),
        1 => Vec3::new(-1.0, -b, a),
        2 => Vec3::new(a, 1.0, b),
        3 => Vec3::new(a, -1.0, -b),
        4 => Vec3::new(a, -b, 1.0),
        _ => Vec3::new(-a, -b, -1.0),
    };
    v.normalize()
}

/// Where a direction lands in an equirectangular image.
///
/// Kreate is Z-up, so the top row is straight up and the bottom row straight
/// down. The centre column faces `-Y`, which is where a default camera looks —
/// so an environment loaded and left alone is behind the subject rather than
/// behind the viewer, which is the framing somebody who picked a sky wanted.
pub fn equirect_uv(dir: Vec3) -> Vec2 {
    let u = 0.5 + dir.x.atan2(-dir.y) / std::f32::consts::TAU;
    let v = dir.z.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    Vec2::new(u, v)
}

/// Where the environment's light comes from, for the path tracer to aim at:
/// a grid over the equirectangular image, each cell weighted by its mean
/// luminance times the solid angle it covers, as cumulative distributions —
/// the rows first (`rows` values), then each row's columns (`rows * columns`).
///
/// Every pixel is counted into its cell, not a sample of them: a sun smaller
/// than a cell is the whole reason for the map, and a point sample of the cell
/// would miss it.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportanceMap {
    pub columns: u32,
    pub rows: u32,
    pub cdf: Vec<f32>,
}

impl ImportanceMap {
    pub const COLUMNS: u32 = 256;
    pub const ROWS: u32 = 128;

    /// The map of an `Rgba16Float` equirectangular image.
    pub fn of(width: u32, height: u32, pixels: &[u8]) -> Self {
        let (columns, rows) = (Self::COLUMNS, Self::ROWS);
        let mut sum = vec![0.0f64; (columns * rows) as usize];
        let mut count = vec![0u32; (columns * rows) as usize];
        if width > 0 && height > 0 {
            for y in 0..height {
                let row = (y as u64 * rows as u64 / height as u64) as u32;
                for x in 0..width {
                    let column = (x as u64 * columns as u64 / width as u64) as u32;
                    let c = texel(pixels, width, x, y);
                    let lum = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
                    let i = (row * columns + column) as usize;
                    sum[i] += lum.max(0.0) as f64;
                    count[i] += 1;
                }
            }
        }
        let mut cdf = vec![0.0f32; (rows + rows * columns) as usize];
        let mut row_weights = vec![0.0f64; rows as usize];
        for row in 0..rows {
            let theta = (row as f64 + 0.5) / rows as f64 * std::f64::consts::PI;
            let mut running = 0.0f64;
            for column in 0..columns {
                let i = (row * columns + column) as usize;
                let mean = if count[i] > 0 {
                    sum[i] / count[i] as f64
                } else {
                    0.0
                };
                running += mean * theta.sin();
                cdf[(rows + row * columns + column) as usize] = running as f32;
            }
            row_weights[row as usize] = running;
            // Normalised along the row; an empty row picks its columns evenly.
            for column in 0..columns {
                let at = (rows + row * columns + column) as usize;
                cdf[at] = if running > 0.0 {
                    (cdf[at] as f64 / running) as f32
                } else {
                    (column + 1) as f32 / columns as f32
                };
            }
        }
        let total: f64 = row_weights.iter().sum();
        let mut running = 0.0f64;
        for row in 0..rows {
            running += row_weights[row as usize];
            cdf[row as usize] = if total > 0.0 {
                (running / total) as f32
            } else {
                (row + 1) as f32 / rows as f32
            };
        }
        Self { columns, rows, cdf }
    }
}

/// The nine spherical-harmonic basis values for a direction.
///
/// Second order, which is nine terms. Written out rather than looped because
/// the constants are what they are and a loop over them would need the same
/// nine written somewhere anyway.
pub fn sh9_basis(d: Vec3) -> [f32; 9] {
    [
        0.282_094_8,
        0.488_602_5 * d.y,
        0.488_602_5 * d.z,
        0.488_602_5 * d.x,
        1.092_548_4 * d.x * d.y,
        1.092_548_4 * d.y * d.z,
        0.315_391_6 * (3.0 * d.z * d.z - 1.0),
        1.092_548_4 * d.x * d.z,
        0.546_274_2 * (d.x * d.x - d.y * d.y),
    ]
}

/// The cosine-lobe convolution factor for each band.
///
/// What turns a projection of *radiance* into a reconstruction of *irradiance*.
/// Three numbers for three bands, and the reason nine coefficients are enough:
/// the factors past band two are small enough that the error is about a
/// percent, which is the standard result.
const BAND: [f32; 9] = {
    // Written as the expressions they are rather than as the numbers they come
    // to. `2π/3` and `π/4` are exact, and a reader who wants to check them
    // against the standard result should not have to recognise a decimal.
    let a0 = std::f32::consts::PI;
    let a1 = 2.0 * std::f32::consts::PI / 3.0;
    let a2 = std::f32::consts::FRAC_PI_4;
    [a0, a1, a1, a1, a2, a2, a2, a2, a2]
};

/// Projects an equirectangular image onto the nine irradiance coefficients.
///
/// The band factors are folded in here rather than at lookup, so that
/// [`sh9_irradiance`] is a dot product and the shader mirror of it is too.
///
/// `pixels` is `Rgba16Float`, which is what an environment is: §2.1's working
/// space has no upper bound and a sky is mostly above one. On the CPU because
/// the pixels are already here — the caller read them out of a file — and a
/// GPU pass would mean a reduction and a readback to learn nine numbers.
///
/// Sampled on a fixed grid rather than per texel. A 4K environment is eight
/// million pixels to learn nine `vec3`s from, and irradiance is smooth: the
/// grid is dense enough that a feature the projection could represent is not
/// missed, and the weights are exactly the solid angles, so a constant
/// environment projects to a constant.
pub fn irradiance_sh9(width: u32, height: u32, pixels: &[u8]) -> [Vec3; 9] {
    let mut out = [Vec3::ZERO; 9];
    if width == 0 || height == 0 {
        return out;
    }
    // Enough to catch a softbox and few enough to be instant. A sun is smaller
    // than one cell and is *still* caught, because the cell it lands in carries
    // its radiance — over-weighted a little and under-weighted nowhere, which
    // is an error in the size of the highlight rather than in the energy.
    const ROWS: u32 = 64;
    const COLUMNS: u32 = 128;
    let mut total_weight = 0.0f32;
    for row in 0..ROWS {
        let v = (row as f32 + 0.5) / ROWS as f32;
        let theta = v * std::f32::consts::PI;
        // The solid angle of a cell at this latitude. Without the sine the
        // poles count as much as the equator and every environment tilts.
        let weight = theta.sin()
            * (std::f32::consts::PI / ROWS as f32)
            * (std::f32::consts::TAU / COLUMNS as f32);
        for column in 0..COLUMNS {
            let u = (column as f32 + 0.5) / COLUMNS as f32;
            let phi = (u - 0.5) * std::f32::consts::TAU;
            // The inverse of `equirect_uv`, and it has to be: a projection that
            // read its directions from one mapping and its pixels from another
            // would be a rotated environment, which looks like an environment.
            let dir = Vec3::new(
                theta.sin() * phi.sin(),
                -theta.sin() * phi.cos(),
                theta.cos(),
            );
            let x = ((u * width as f32) as u32).min(width - 1);
            let y = ((v * height as f32) as u32).min(height - 1);
            let radiance = texel(pixels, width, x, y);
            let basis = sh9_basis(dir);
            for i in 0..9 {
                out[i] += radiance * (basis[i] * weight);
            }
            total_weight += weight;
        }
    }
    // The grid's weights sum to very nearly 4π but not exactly, and the
    // difference is a constant scale on the answer.
    //
    // **At this grid size it is a guard rather than a fix, and the sweep says
    // so**: removing it changed nothing any test could see. The midpoint rule
    // over 64 rows has an error of about `(π/64)²/24`, which is a part in ten
    // thousand — so the correction moves the answer by 0.01% and nothing short
    // of a much coarser grid could notice. It stays because `ROWS` is a number
    // somebody may lower, and a scene that is quietly a percent dark is the
    // kind of wrong nobody finds.
    let correction = if total_weight > 0.0 {
        (4.0 * std::f32::consts::PI) / total_weight
    } else {
        0.0
    };
    for i in 0..9 {
        out[i] *= BAND[i] * correction;
    }
    out
}

/// The irradiance arriving at a surface facing `normal`.
///
/// A dot product, which is the whole point of storing nine numbers.
pub fn sh9_irradiance(sh: &[Vec3; 9], normal: Vec3) -> Vec3 {
    let basis = sh9_basis(normal.normalize_or_zero());
    let mut out = Vec3::ZERO;
    for i in 0..9 {
        out += sh[i] * basis[i];
    }
    // Irradiance cannot be negative, and a truncated harmonic series can be:
    // ringing around a bright, small source undershoots. Clamped rather than
    // left, because a negative irradiance darkens a surface the environment is
    // *lighting*, which reads as a shadow nothing casts.
    out.max(Vec3::ZERO)
}

/// One texel of an `Rgba16Float` image.
fn texel(pixels: &[u8], width: u32, x: u32, y: u32) -> Vec3 {
    let at = ((y as usize * width as usize) + x as usize) * 8;
    if at + 6 > pixels.len() {
        return Vec3::ZERO;
    }
    let channel =
        |i: usize| half::f16::from_le_bytes([pixels[at + i * 2], pixels[at + i * 2 + 1]]).to_f32();
    Vec3::new(channel(0), channel(1), channel(2))
}

/// The nine coefficients in the shape a uniform buffer wants.
///
/// Padded to `vec4` because a `vec3` in a uniform array is aligned to sixteen
/// bytes anyway — writing them tight would put every coefficient after the
/// first in the wrong place, which is a lighting bug that looks like an
/// environment somebody else authored.
pub fn sh9_bytes(sh: &[Vec3; 9]) -> [[f32; 4]; 9] {
    let mut out = [[0.0f32; 4]; 9];
    for (i, c) in sh.iter().enumerate() {
        out[i] = [c.x, c.y, c.z, 0.0];
    }
    out
}

/// What the two compute passes need to know.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    /// x: the face size being written. y: unused. z: this mip's roughness.
    /// w: how many samples the prefilter takes.
    size: [f32; 4],
    /// x: the base face size, for turning a sample's solid angle into a mip.
    /// y: how many mips the source view actually holds — a sample may not read
    /// past what has been written, and a `textureSampleLevel` past the end is
    /// clamped silently rather than reported.
    source: [f32; 4],
}

/// A prefiltered environment, ready for the shading pass.
///
/// Built once when an image arrives. §2.2 budgets two hundred milliseconds for
/// it, which is generous for six 128-pixel faces — the expensive part is the
/// prefilter, and it runs over 6 × (128² + 64² + …) texels, about a hundred
/// thousand, at [`SAMPLES`] samples each.
pub struct Environment {
    cube: wgpu::Texture,
    /// The whole chain, for the shading pass to sample by roughness.
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    sh9: [Vec3; 9],
}

impl Environment {
    /// Projects an equirectangular `Rgba16Float` image and prefilters it.
    ///
    /// The pixels are the caller's, not a document's: `kreate-render` never
    /// sees a `Document`, so an environment arrives as bytes and a size the way
    /// a mesh arrives as buffers.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Self {
        // The nine coefficients come off the bytes that are already here. A GPU
        // pass would be a reduction and a readback to learn nine numbers.
        let sh9 = irradiance_sh9(width, height, pixels);

        let equirect = upload_equirect(device, queue, width, height, pixels);
        let cube = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment cube"),
            size: wgpu::Extent3d {
                width: FACE,
                height: FACE,
                depth_or_array_layers: 6,
            },
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            // `STORAGE_BINDING` because both passes write through a storage
            // view: a render pass per face per mip would be thirty-six passes
            // to do what two dispatches do.
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("environment"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let passes = Passes::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("environment"),
        });
        passes.to_cube(device, &mut encoder, &equirect, &sampler, &cube);
        // Level zero is the environment itself, so the filtering starts at one.
        for level in 1..MIPS {
            passes.prefilter(device, &mut encoder, &cube, &sampler, level);
        }
        queue.submit([encoder.finish()]);

        let view = cube.create_view(&wgpu::TextureViewDescriptor {
            label: Some("environment cube"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        Self {
            cube,
            view,
            sampler,
            sh9,
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// The irradiance coefficients, for the shading uniform.
    pub fn sh9(&self) -> &[Vec3; 9] {
        &self.sh9
    }

    /// The cube, for a test that wants to read a face back.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.cube
    }

    /// The roughest mip, which is what a roughness of one reads.
    pub fn roughest(&self) -> f32 {
        (self.cube.mip_level_count() - 1) as f32
    }

    /// A cube that exists only so a bind group has something in it.
    ///
    /// A bind group cannot have a hole, so "no environment" has to be *some*
    /// texture. One black texel per face and no dispatches at all: the flag in
    /// the shading uniform is what says not to read it, and a scene with no
    /// environment should not pay a prefilter to find that out.
    pub fn placeholder(device: &wgpu::Device) -> Self {
        let cube = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("no environment"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = cube.create_view(&wgpu::TextureViewDescriptor {
            label: Some("no environment"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("no environment"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            cube,
            view,
            sampler,
            sh9: [Vec3::ZERO; 9],
        }
    }
}

fn upload_equirect(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> wgpu::TextureView {
    let width = width.max(1);
    let height = height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("environment source"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Zero-filled when the caller's slice is short, rather than refused. An
    // image whose tiles are half resident is a real state (§3.6's sparse
    // residency), and half a sky is better than no sky and a dialog.
    let needed = width as usize * height as usize * 8;
    let mut data = pixels.to_vec();
    data.resize(needed, 0);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The two pipelines and the layout they share.
struct Passes {
    layout: wgpu::BindGroupLayout,
    to_cube: wgpu::ComputePipeline,
    prefilter: wgpu::ComputePipeline,
}

impl Passes {
    fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("environment"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("environment"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/environment.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("environment"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            to_cube: make("to_cube"),
            prefilter: make("prefilter"),
            layout,
        }
    }

    /// Projects the image onto the six faces of mip zero.
    fn to_cube(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        equirect: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        cube: &wgpu::Texture,
    ) {
        let params = uniform(
            device,
            Params {
                size: [FACE as f32, 0.0, 0.0, 0.0],
                source: [FACE as f32, 1.0, 0.0, 0.0],
            },
        );
        // The cube is bound as a *source* here too, because one layout serves
        // both passes and a binding cannot be left out. Mip zero is being
        // written through binding four, so what is read is the rest of the
        // chain — which this pass never samples. Reading and writing different
        // subresources of one texture is what the mip chain is built on.
        let source = read_view(cube, 1, MIPS - 1);
        let target = write_view(cube, 0);
        let bind = self.bind(device, &params, equirect, &source, sampler, &target);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("equirect to cube"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.to_cube);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(FACE.div_ceil(8), FACE.div_ceil(8), 6);
    }

    /// Convolves mip `level` with the GGX lobe for its roughness.
    fn prefilter(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        cube: &wgpu::Texture,
        sampler: &wgpu::Sampler,
        level: u32,
    ) {
        let size = (FACE >> level).max(1);
        let params = uniform(
            device,
            Params {
                size: [size as f32, 0.0, roughness_of(level), SAMPLES as f32],
                source: [FACE as f32, level as f32, 0.0, 0.0],
            },
        );
        // Only the levels already written. A view that reached the level being
        // written would be the same subresource read and stored to in one pass,
        // which is the one thing the usage rules forbid — and it would be a
        // level filtering itself, which is not a filter.
        let source = read_view(cube, 0, level);
        let target = write_view(cube, level);
        // The equirect binding is unused here and still has to be something.
        // The cube's own mip zero as a 2D view is the honest filler: a texture
        // that is already bound, rather than a second one made to be ignored.
        let filler = cube.create_view(&wgpu::TextureViewDescriptor {
            label: Some("unused"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: 0,
            array_layer_count: Some(1),
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let bind = self.bind(device, &params, &filler, &source, sampler, &target);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("environment prefilter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.prefilter);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 6);
    }

    fn bind(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        equirect: &wgpu::TextureView,
        source: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        target: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("environment"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(equirect),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(target),
                },
            ],
        })
    }
}

/// A cube view over part of the chain.
fn read_view(cube: &wgpu::Texture, base: u32, count: u32) -> wgpu::TextureView {
    cube.create_view(&wgpu::TextureViewDescriptor {
        label: Some("environment source"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        base_mip_level: base,
        mip_level_count: Some(count.max(1)),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    })
}

/// A writable view over one mip's six faces.
fn write_view(cube: &wgpu::Texture, level: u32) -> wgpu::TextureView {
    cube.create_view(&wgpu::TextureViewDescriptor {
        label: Some("environment target"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_mip_level: level,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    })
}

fn uniform(device: &wgpu::Device, params: Params) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("environment params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An equirectangular image of one colour.
    fn flat(width: u32, height: u32, value: f32) -> Vec<u8> {
        let half = half::f16::from_f32(value).to_le_bytes();
        let one = half::f16::from_f32(1.0).to_le_bytes();
        let mut out = Vec::with_capacity((width * height) as usize * 8);
        for _ in 0..width * height {
            out.extend_from_slice(&half);
            out.extend_from_slice(&half);
            out.extend_from_slice(&half);
            out.extend_from_slice(&one);
        }
        out
    }

    #[test]
    fn a_uniform_environment_irradiates_every_direction_the_same() {
        // The identity everything else rests on. Under an environment of `L`
        // the irradiance is `πL` in every direction, so the diffuse answer is
        // `albedo × L` — which is exactly what the flat path already computes,
        // and what §7.5's furnace test asserts of the picture.
        //
        // If this drifts, an image environment and a flat one of the same
        // colour light a scene differently, and the difference appears the
        // moment somebody picks a sky and wonders why the model changed.
        let image = flat(64, 32, 3.0);
        let sh = irradiance_sh9(64, 32, &image);
        for dir in [
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::X,
            Vec3::NEG_Y,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
        ] {
            let e = sh9_irradiance(&sh, dir);
            let wanted = std::f32::consts::PI * 3.0;
            assert!(
                (e.x - wanted).abs() < wanted * 0.005,
                "{dir:?} receives {} against {wanted}",
                e.x
            );
        }
    }

    #[test]
    fn the_bands_above_the_first_vanish_for_a_uniform_environment() {
        // The same claim from the other side, and the one that catches a
        // missing sine weight: without it the poles are over-counted, the
        // first-order terms stop cancelling, and a flat grey sky lights the
        // top of a model more than its sides. The picture still looks like a
        // picture, which is why this is asserted on the numbers.
        let sh = irradiance_sh9(64, 32, &flat(64, 32, 1.0));
        for (i, c) in sh.iter().enumerate().skip(1) {
            assert!(
                c.length() < 0.01,
                "band term {i} is {c:?} under a uniform sky"
            );
        }
    }

    #[test]
    fn a_bright_top_half_lights_upward_faces_more() {
        // Direction, which is the whole reason an image environment exists. A
        // projection that lost it would still pass the uniform tests above —
        // they are about what the constant term does.
        let (width, height) = (64u32, 32u32);
        let mut image = flat(width, height, 0.0);
        let bright = half::f16::from_f32(4.0).to_le_bytes();
        for y in 0..height / 2 {
            for x in 0..width {
                let at = ((y * width + x) as usize) * 8;
                for c in 0..3 {
                    image[at + c * 2..at + c * 2 + 2].copy_from_slice(&bright);
                }
            }
        }
        let sh = irradiance_sh9(width, height, &image);
        let up = sh9_irradiance(&sh, Vec3::Z).x;
        let down = sh9_irradiance(&sh, Vec3::NEG_Z).x;
        assert!(
            up > down * 3.0,
            "a sky bright above gave {up} up and {down} down"
        );
        // And the average over the two is still the whole environment's, which
        // is what says the energy was redistributed rather than invented: half
        // the sphere at 4.0 is a mean radiance of 2.0, so `πL` is about 6.28.
        let mean = (up + down) * 0.5;
        assert!(
            (mean - std::f32::consts::PI * 2.0).abs() < 1.0,
            "the two faces average {mean}"
        );
    }

    #[test]
    fn ringing_never_makes_a_surface_darker_than_nothing() {
        // A truncated harmonic series undershoots around a small bright source,
        // and a negative irradiance darkens a surface the environment is
        // lighting — which reads as a shadow nothing casts. A sun in an
        // otherwise black sky is the case that produces it.
        let (width, height) = (64u32, 32u32);
        let mut image = flat(width, height, 0.0);
        let sun = half::f16::from_f32(2000.0).to_le_bytes();
        for c in 0..3 {
            let at = ((4 * width + 20) as usize) * 8;
            image[at + c * 2..at + c * 2 + 2].copy_from_slice(&sun);
        }
        let sh = irradiance_sh9(width, height, &image);
        for dir in [
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
        ] {
            let e = sh9_irradiance(&sh, dir);
            assert!(e.min_element() >= 0.0, "{dir:?} receives {e:?}");
        }
    }

    #[test]
    fn a_cube_texel_and_the_direction_it_stands_for_agree_with_the_sampler() {
        // The centres of the six faces are the six axes, in the order a cube
        // sampler uses. Written as six assertions rather than a loop because
        // the *order* is the claim: a mapping that is self-consistent but
        // permuted produces a sky that is a sky, rotated.
        let centre = |face| cube_direction(face, 0.5, 0.5);
        assert!((centre(0) - Vec3::X).length() < 1e-6);
        assert!((centre(1) - Vec3::NEG_X).length() < 1e-6);
        assert!((centre(2) - Vec3::Y).length() < 1e-6);
        assert!((centre(3) - Vec3::NEG_Y).length() < 1e-6);
        assert!((centre(4) - Vec3::Z).length() < 1e-6);
        assert!((centre(5) - Vec3::NEG_Z).length() < 1e-6);
    }

    #[test]
    fn the_faces_meet_along_their_shared_edges() {
        // What half a texel of offset buys. A face written from its corners
        // rather than its texel centres leaves a seam down every edge of the
        // cube, which shows as a visible cross on a mirror.
        //
        // Checked as a limit rather than at the edge itself: the shared
        // direction is what both faces approach, and a mapping that flipped one
        // of them approaches it from the wrong side.
        let e = 1e-3;
        // The edge between +X and +Z. On +X the `z` component is `-a`, so it is
        // largest at `s = 0`; on +Z the `x` component is `a`, so it is largest
        // at `s = 1`. Working that out is the test — an edge guessed at rather
        // than derived is how the first version of this passed against the
        // wrong pair of borders.
        let from_x = cube_direction(0, e, 0.5);
        let from_z = cube_direction(4, 1.0 - e, 0.5);
        assert!(
            (from_x - from_z).length() < 0.01,
            "+X reaches {from_x:?} where +Z reaches {from_z:?}"
        );
        // And the other pair the same face touches, so a mapping that happened
        // to agree on one axis is not enough: +X at `t = 0` meets +Y.
        let from_x = cube_direction(0, 0.5, e);
        let from_y = cube_direction(2, 1.0 - e, 0.5);
        assert!(
            (from_x - from_y).length() < 0.01,
            "+X reaches {from_x:?} where +Y reaches {from_y:?}"
        );
    }

    #[test]
    fn the_equirect_mapping_puts_up_at_the_top_and_forward_in_the_middle() {
        // A convention, and the only thing that makes one right is that it is
        // written down and that both sides use it. The centre facing `-Y` is
        // the framing somebody who picked a sky wanted: behind the subject
        // rather than behind the viewer.
        assert!((equirect_uv(Vec3::Z).y - 0.0).abs() < 1e-6);
        assert!((equirect_uv(Vec3::NEG_Z).y - 1.0).abs() < 1e-6);
        let forward = equirect_uv(Vec3::NEG_Y);
        assert!(
            (forward.x - 0.5).abs() < 1e-6,
            "forward is at {}",
            forward.x
        );
        assert!((forward.y - 0.5).abs() < 1e-6);
        // And `u` increases to the right, so an image is not mirrored.
        assert!(equirect_uv(Vec3::X).x > 0.5);
        assert!(equirect_uv(Vec3::NEG_X).x < 0.5);
    }

    #[test]
    fn the_projection_reads_its_pixels_where_it_thinks_it_is_looking() {
        // The two mappings are written twice — once forwards in `equirect_uv`
        // and once backwards in the projection's loop — and nothing else would
        // notice if they disagreed. A projection reading its directions from
        // one and its pixels from the other is a rotated environment, and a
        // rotated environment looks like an environment.
        for (u, v) in [(0.5, 0.5), (0.25, 0.3), (0.9, 0.8), (0.0, 0.5)] {
            let theta = v * std::f32::consts::PI;
            let phi = (u - 0.5) * std::f32::consts::TAU;
            let dir = Vec3::new(
                theta.sin() * phi.sin(),
                -theta.sin() * phi.cos(),
                theta.cos(),
            );
            let back = equirect_uv(dir);
            // `u` wraps, so the two ends of the image are the same column.
            let du = (back.x - u).abs().min(1.0 - (back.x - u).abs());
            assert!(du < 1e-5, "({u}, {v}) came back at {back:?}");
            assert!(
                (back.y - v).abs() < 1e-5,
                "({u}, {v}) came back at {back:?}"
            );
        }
    }

    /// An equirectangular image whose radiance is a function of direction.
    fn sky_of(width: u32, height: u32, mut radiance: impl FnMut(Vec3) -> f32) -> Vec<u8> {
        let mut out = Vec::with_capacity((width * height) as usize * 8);
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let theta = v * std::f32::consts::PI;
                let phi = (u - 0.5) * std::f32::consts::TAU;
                let dir = Vec3::new(
                    theta.sin() * phi.sin(),
                    -theta.sin() * phi.cos(),
                    theta.cos(),
                );
                let value = half::f16::from_f32(radiance(dir)).to_le_bytes();
                for _ in 0..3 {
                    out.extend_from_slice(&value);
                }
                out.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
            }
        }
        out
    }

    /// Irradiance by direct integration: `∫ L(ω) max(n·ω, 0) dω`.
    ///
    /// The definition, computed the slow way. This is the only thing in the
    /// file that says what the nine coefficients are *for* — everything else
    /// compares them to themselves, which is how a wrong band factor and a
    /// flipped basis sign both survived a sweep.
    fn numeric_irradiance(normal: Vec3, mut radiance: impl FnMut(Vec3) -> f32) -> f32 {
        const ROWS: u32 = 256;
        const COLUMNS: u32 = 512;
        let mut total = 0.0f32;
        for row in 0..ROWS {
            let theta = (row as f32 + 0.5) / ROWS as f32 * std::f32::consts::PI;
            let weight = theta.sin()
                * (std::f32::consts::PI / ROWS as f32)
                * (std::f32::consts::TAU / COLUMNS as f32);
            for column in 0..COLUMNS {
                let phi = ((column as f32 + 0.5) / COLUMNS as f32 - 0.5) * std::f32::consts::TAU;
                let dir = Vec3::new(
                    theta.sin() * phi.sin(),
                    -theta.sin() * phi.cos(),
                    theta.cos(),
                );
                total += radiance(dir) * normal.dot(dir).max(0.0) * weight;
            }
        }
        total
    }

    #[test]
    fn the_coefficients_reproduce_the_integral_they_stand_for() {
        // **What the nine numbers are for**, checked against the definition
        // rather than against themselves.
        //
        // The sky is a quadratic in the direction, which second-order harmonics
        // represent *exactly* — so the reconstruction should match the integral
        // to within the grid's own error, and a wrong band factor or a flipped
        // basis sign is a large, obvious failure rather than a subtle one.
        //
        // Every other test here compares a projection to another projection, or
        // asks for a ratio. That is how the second band's factor being twice
        // what it should be, and the sign of its basis term, both survived a
        // falsification sweep.
        let axis = Vec3::new(0.3, -0.6, 0.74).normalize();
        let sky = |d: Vec3| 1.0 + 2.0 * d.dot(axis) + 3.0 * d.dot(axis) * d.dot(axis);
        let image = sky_of(256, 128, sky);
        let sh = irradiance_sh9(256, 128, &image);

        for normal in [
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::X,
            Vec3::NEG_Y,
            axis,
            -axis,
            Vec3::new(0.5, 0.5, -0.7).normalize(),
        ] {
            let got = sh9_irradiance(&sh, normal).x;
            let want = numeric_irradiance(normal, sky);
            assert!(
                (got - want).abs() < want.abs().max(0.05) * 0.04,
                "{normal:?}: the coefficients say {got} and the integral says {want}"
            );
        }
    }

    #[test]
    fn a_sky_with_no_second_order_part_is_still_reproduced() {
        // The first band on its own, so that a failure above can be attributed.
        // A linear sky is exact in bands zero and one, and if this passes while
        // the quadratic case fails, what is wrong is the second band.
        let axis = Vec3::new(-0.5, 0.2, 0.84).normalize();
        let sky = move |d: Vec3| 2.0 + 1.5 * d.dot(axis);
        let image = sky_of(256, 128, sky);
        let sh = irradiance_sh9(256, 128, &image);
        for normal in [Vec3::Z, Vec3::X, axis, -axis] {
            let got = sh9_irradiance(&sh, normal).x;
            let want = numeric_irradiance(normal, sky);
            assert!(
                (got - want).abs() < want.abs().max(0.05) * 0.03,
                "{normal:?}: the coefficients say {got} and the integral says {want}"
            );
        }
    }

    #[test]
    fn the_roughest_mip_is_roughness_one_and_the_sharpest_is_zero() {
        // The ends are pinned, which is what lets the shading pass turn a
        // roughness into a mip by multiplying. A chain whose last level stood
        // for something less than one would make the roughest material in a
        // scene the one that is wrong.
        assert_eq!(roughness_of(0), 0.0);
        assert_eq!(roughness_of(MIPS - 1), 1.0);
        for level in 1..MIPS {
            assert!(roughness_of(level) > roughness_of(level - 1));
        }
    }
}

#[cfg(test)]
mod importance_tests {
    use super::*;

    fn image(width: u32, height: u32, bright: impl Fn(u32, u32) -> bool) -> Vec<u8> {
        let mut out = Vec::with_capacity((width * height * 8) as usize);
        for y in 0..height {
            for x in 0..width {
                let v = if bright(x, y) { 1000.0 } else { 1e-4 };
                for c in [v, v, v, 1.0] {
                    out.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
                }
            }
        }
        out
    }

    #[test]
    fn a_one_pixel_sun_takes_nearly_all_of_the_distribution() {
        // A 512x256 sky, dark but for one pixel: smaller than a cell, and the
        // cell it lands in must still carry it.
        let (w, h) = (512, 256);
        let pixels = image(w, h, |x, y| x == 300 && y == 60);
        let map = ImportanceMap::of(w, h, &pixels);
        let (cols, rows) = (map.columns as usize, map.rows as usize);
        let row = 60 * rows / h as usize;
        let col = 300 * cols / w as usize;
        let row_share = map.cdf[row] - if row > 0 { map.cdf[row - 1] } else { 0.0 };
        assert!(
            row_share > 0.9,
            "the sun's row has {row_share} of the distribution"
        );
        let at = rows + row * cols + col;
        let cell_share = map.cdf[at] - map.cdf[at - 1];
        assert!(
            cell_share > 0.9,
            "the sun's cell has {cell_share} of its row"
        );
        assert!(
            (map.cdf[rows - 1] - 1.0).abs() < 1e-5,
            "the rows sum to one"
        );
    }
}
