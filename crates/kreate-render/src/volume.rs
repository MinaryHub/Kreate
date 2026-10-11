//! Smoke and fire in the viewport: a baked gas's grid as a 3D texture,
//! marched over the picture. See `shaders/volume.wgsl`.
//!
//! Raster only: drawn over the solid view, and over the rendered and path
//! traced pictures before their display transform, but the path tracer
//! itself knows nothing of it — the smoke casts no shadow there.

use glam::{DVec3, Mat4};

use crate::camera::Camera;

const SOURCE: &str = include_str!("shaders/volume.wgsl");

/// The shader, for the portability tests.
pub fn source() -> &'static str {
    SOURCE
}

const GRID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

/// One gas to draw this frame.
#[derive(Clone, Copy, Debug)]
pub struct VolumeData<'a> {
    /// Cells along X, Y and Z.
    pub cells: [u32; 3],
    /// A byte a cell each, X fastest.
    pub smoke: &'a [u8],
    pub flame: &'a [u8],
    /// The box the cells fill, in the world.
    pub min: DVec3,
    pub max: DVec3,
    /// Linear.
    pub smoke_color: [f32; 3],
    /// What the grid's bytes are: the same key is not uploaded again.
    pub key: usize,
    /// How much finer-than-cell wisps carve the smoke, 0 to 1.
    pub detail: f32,
    /// Seconds into the animation, which the wisps drift with.
    pub time: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    inv_view_proj: [[f32; 4]; 4],
    box_min: [f32; 4],
    box_max: [f32; 4],
    smoke_color: [f32; 4],
    light: [f32; 4],
    march: [f32; 4],
    region: [f32; 4],
    detail: [f32; 4],
}

struct Item {
    key: usize,
    cells: [u32; 3],
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    params: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

pub struct Volumes {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    display: wgpu::RenderPipeline,
    linear: wgpu::RenderPipeline,
    items: Vec<Item>,
    region: (u32, u32),
}

impl Volumes {
    /// Pipelines for the display-referred picture (`display`) and the linear
    /// HDR one (`linear`).
    pub fn new(
        device: &wgpu::Device,
        display: wgpu::TextureFormat,
        linear: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volume"),
            source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volume"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("volume"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        // Premultiplied: what the volume adds, over what is
                        // left of the picture behind it.
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            display: pipeline(display),
            linear: pipeline(linear),
            layout,
            sampler,
            items: Vec::new(),
            region: (1, 1),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// What to draw this frame, seen by `camera`: grids uploaded when new,
    /// the rest a few numbers each. `depth` is the picture's depth, which
    /// ends each ray; `hdr` whether it is drawn into the linear picture.
    #[allow(clippy::too_many_arguments)]
    pub fn set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        aspect: f64,
        depth: &wgpu::TextureView,
        region: (u32, u32),
        hdr: bool,
        volumes: &[VolumeData<'_>],
    ) {
        self.region = region;
        let mut kept = std::mem::take(&mut self.items);
        let eye = camera.eye();
        let inv: Mat4 = camera.view_projection(aspect).inverse();
        // From above and a little behind the view's right shoulder: smoke
        // reads as round when the light is not where the eye is.
        let light = (DVec3::Z * 2.0 + camera.right() - camera.forward() * 0.5)
            .normalize()
            .as_vec3();
        for v in volumes {
            let count = v.cells.iter().map(|c| *c as usize).product::<usize>();
            if v.cells.contains(&0) || v.smoke.len() != count || v.flame.len() != count {
                continue;
            }
            let found = kept.iter().position(|i| i.cells == v.cells);
            let mut item = match found {
                Some(i) => kept.swap_remove(i),
                None => Self::make(device, &self.layout, &self.sampler, depth, v.cells),
            };
            if item.key != v.key {
                let mut bytes = Vec::with_capacity(count * 2);
                for (s, f) in v.smoke.iter().zip(v.flame) {
                    bytes.push(*s);
                    bytes.push(*f);
                }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &item.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(v.cells[0] * 2),
                        rows_per_image: Some(v.cells[1]),
                    },
                    wgpu::Extent3d {
                        width: v.cells[0],
                        height: v.cells[1],
                        depth_or_array_layers: v.cells[2],
                    },
                );
                item.key = v.key;
            }
            // The depth view is the viewport's current one, which a resize
            // replaces.
            item.bind = Self::bind(device, &self.layout, &item, &self.sampler, depth);
            let size = v.max - v.min;
            let cell = (size.x / v.cells[0] as f64)
                .min(size.y / v.cells[1] as f64)
                .min(size.z / v.cells[2] as f64);
            let (lo, hi) = ((v.min - eye).as_vec3(), (v.max - eye).as_vec3());
            let params = Params {
                inv_view_proj: inv.to_cols_array_2d(),
                box_min: [lo.x, lo.y, lo.z, 0.0],
                box_max: [hi.x, hi.y, hi.z, 0.0],
                smoke_color: [v.smoke_color[0], v.smoke_color[1], v.smoke_color[2], 4.0],
                light: [light.x, light.y, light.z, 0.0],
                march: [
                    (cell * 0.5) as f32,
                    400.0,
                    15.0,
                    if hdr { 1.0 } else { 0.0 },
                ],
                region: [region.0 as f32, region.1 as f32, 0.0, 0.0],
                // A wisp about two cells across at its coarsest, and finer
                // in the octaves above it.
                detail: [
                    v.detail.clamp(0.0, 1.0),
                    (1.0 / (cell * 2.0)) as f32,
                    v.time,
                    0.0,
                ],
            };
            queue.write_buffer(&item.params, 0, bytemuck::bytes_of(&params));
            self.items.push(item);
        }
    }

    fn make(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        depth: &wgpu::TextureView,
        cells: [u32; 3],
    ) -> Item {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume grid"),
            size: wgpu::Extent3d {
                width: cells[0],
                height: cells[1],
                depth_or_array_layers: cells[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: GRID_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume"),
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
                    resource: wgpu::BindingResource::TextureView(depth),
                },
            ],
        });
        Item {
            key: usize::MAX,
            cells,
            texture,
            view,
            params,
            bind,
        }
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        item: &Item,
        sampler: &wgpu::Sampler,
        depth: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: item.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&item.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
            ],
        })
    }

    /// Marches every volume over `target`: the display-referred picture, or
    /// with `hdr` the linear one.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, hdr: bool) {
        if self.items.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volume"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
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
        pass.set_viewport(
            0.0,
            0.0,
            self.region.0 as f32,
            self.region.1 as f32,
            0.0,
            1.0,
        );
        pass.set_pipeline(if hdr { &self.linear } else { &self.display });
        for item in &self.items {
            pass.set_bind_group(0, &item.bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
