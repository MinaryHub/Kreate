//! Bloom for the rendered view: light past white spills into its
//! surroundings. See `shaders/bloom.wgsl`.
//!
//! On the linear picture, before the display transform, so what glows is what
//! is *bright* — a lamp, a sunlit highlight — and not what is merely light in
//! colour.

const SOURCE: &str = include_str!("shaders/bloom.wgsl");

/// The shader, for the portability tests.
pub fn source() -> &'static str {
    SOURCE
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// How many half-size images the blur runs through: the last is a
/// sixty-fourth of the region, which is how far a glow reaches.
const LEVELS: usize = 6;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    region: [f32; 2],
    alloc: [f32; 2],
    settings: [f32; 4],
}

pub struct Bloom {
    prefilter: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    levels: Vec<wgpu::TextureView>,
    level_binds: Vec<wgpu::BindGroup>,
    picture_bind: Option<wgpu::BindGroup>,
    /// The region and allocation the chain and the binding were made for.
    made_for: ((u32, u32), (u32, u32)),
}

impl Bloom {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom"),
            source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let add = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let pipeline = |label: &str, entry: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("bloom_vertex"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: FORMAT, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let prefilter = pipeline("bloom prefilter", "bloom_prefilter", None);
        let down = pipeline("bloom down", "bloom_down", None);
        let up = pipeline("bloom up", "bloom_up", Some(add));
        // The picture is the same format as the chain.
        debug_assert_eq!(FORMAT, crate::display::HDR_FORMAT);
        let composite = pipeline("bloom composite", "bloom_composite", Some(add));
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            prefilter,
            down,
            up,
            composite,
            layout,
            sampler,
            params,
            levels: Vec::new(),
            level_binds: Vec::new(),
            picture_bind: None,
            made_for: ((0, 0), (0, 0)),
        }
    }

    fn level_size(region: (u32, u32), i: usize) -> (u32, u32) {
        ((region.0 >> (i + 1)).max(1), (region.1 >> (i + 1)).max(1))
    }

    fn bind(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.params.as_entire_binding() },
            ],
        })
    }

    /// Readies the chain for this frame's region and picture, and its
    /// settings: the exposure in stops, how strong, and the threshold.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        picture: &wgpu::TextureView,
        region: (u32, u32),
        alloc: (u32, u32),
        exposure: f32,
        intensity: f32,
        threshold: f32,
    ) {
        if self.made_for != (region, alloc) || self.picture_bind.is_none() {
            self.made_for = (region, alloc);
            self.levels = (0..LEVELS)
                .map(|i| {
                    let (w, h) = Self::level_size(region, i);
                    device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: Some("bloom level"),
                            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: FORMAT,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        })
                        .create_view(&wgpu::TextureViewDescriptor::default())
                })
                .collect();
            self.level_binds = self.levels.iter().map(|v| self.bind(device, v)).collect();
        }
        // The picture is remade with the allocation; rebound every time it
        // could have been, which is cheap.
        self.picture_bind = Some(self.bind(device, picture));
        let p = Params {
            region: [region.0 as f32, region.1 as f32],
            alloc: [alloc.0 as f32, alloc.1 as f32],
            settings: [exposure, intensity, threshold, 0.5],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// Adds the glow to `picture`, the view prepared for.
    pub fn apply(&self, encoder: &mut wgpu::CommandEncoder, picture: &wgpu::TextureView) {
        let Some(picture_bind) = &self.picture_bind else { return };
        if self.levels.len() != LEVELS {
            return;
        }
        let region = self.made_for.0;
        let run = |encoder: &mut wgpu::CommandEncoder,
                   target: &wgpu::TextureView,
                   size: (u32, u32),
                   pipeline: &wgpu::RenderPipeline,
                   bind: &wgpu::BindGroup,
                   level: u32,
                   clear: bool| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, level..level + 1);
        };
        run(encoder, &self.levels[0], Self::level_size(region, 0), &self.prefilter, picture_bind, 0, true);
        for i in 0..LEVELS - 1 {
            run(encoder, &self.levels[i + 1], Self::level_size(region, i + 1), &self.down, &self.level_binds[i], i as u32, true);
        }
        for i in (0..LEVELS - 1).rev() {
            run(encoder, &self.levels[i], Self::level_size(region, i), &self.up, &self.level_binds[i + 1], i as u32, false);
        }
        run(encoder, picture, region, &self.composite, &self.level_binds[0], 0, false);
    }
}
