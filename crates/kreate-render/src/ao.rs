//! Ambient occlusion for the studio view: corners and creases darken.
//!
//! Screen space, from the depth the surfaces wrote — see `shaders/ao.wgsl`.
//! Applied to the surfaces' own image before it is smoothed onto the target,
//! so the overlays drawn afterwards are never darkened by it.

use glam::Mat4;

const SOURCE: &str = include_str!("shaders/ao.wgsl");

/// The shader, for the portability tests.
pub fn source() -> &'static str {
    SOURCE
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    projection: [[f32; 4]; 4],
    inverse_projection: [[f32; 4]; 4],
    region: [f32; 2],
    radius: f32,
    strength: f32,
    /// Which frame of a still view's average this is.
    seed: f32,
    _pad: [f32; 3],
}

pub struct Ao {
    estimate: wgpu::RenderPipeline,
    apply: wgpu::RenderPipeline,
    /// The same, into the rendered view's linear picture.
    apply_linear: wgpu::RenderPipeline,
    depth_layout: wgpu::BindGroupLayout,
    occlusion_layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    occlusion: wgpu::TextureView,
    depth_bind: wgpu::BindGroup,
    occlusion_bind: wgpu::BindGroup,
    size: (u32, u32),
}

impl Ao {
    /// `surfaces` is the format of the image the occlusion is multiplied into.
    pub fn new(
        device: &wgpu::Device,
        surfaces: wgpu::TextureFormat,
        depth: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ao"),
            source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
        });
        let depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ao depth"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
        let occlusion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ao occlusion"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline = |label: &str,
                        layouts: &[Option<&wgpu::BindGroupLayout>],
                        entry: &str,
                        format: wgpu::TextureFormat,
                        blend: Option<wgpu::BlendState>| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("ao_vertex"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let estimate = pipeline(
            "ao estimate",
            &[Some(&depth_layout)],
            "ao_fragment",
            FORMAT,
            None,
        );
        // Destination times source: the image darkened by the occlusion, its
        // alpha left as it was.
        let multiply = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::Src,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let apply = pipeline(
            "ao apply",
            &[Some(&depth_layout), Some(&occlusion_layout)],
            "ao_apply",
            surfaces,
            Some(multiply),
        );
        let apply_linear = pipeline(
            "ao apply linear",
            &[Some(&depth_layout), Some(&occlusion_layout)],
            "ao_apply",
            crate::display::HDR_FORMAT,
            Some(multiply),
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ao params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let size = (width.max(1), height.max(1));
        let occlusion = Self::image(device, size);
        let depth_bind = Self::depth_bind(device, &depth_layout, depth, &params);
        let occlusion_bind = Self::occlusion_bind(device, &occlusion_layout, &occlusion);
        Self {
            estimate,
            apply,
            apply_linear,
            depth_layout,
            occlusion_layout,
            params,
            occlusion,
            depth_bind,
            occlusion_bind,
            size,
        }
    }

    fn image(device: &wgpu::Device, size: (u32, u32)) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("ao"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn depth_bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        depth: &wgpu::TextureView,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao depth"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn occlusion_bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        occlusion: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao occlusion"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(occlusion),
            }],
        })
    }

    /// Follows the viewport's allocation. The depth is always rebound — the
    /// viewport remakes it on a resize, and the old view would be read.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        depth: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let size = (width.max(1), height.max(1));
        if size != self.size {
            self.size = size;
            self.occlusion = Self::image(device, size);
            self.occlusion_bind =
                Self::occlusion_bind(device, &self.occlusion_layout, &self.occlusion);
        }
        self.depth_bind = Self::depth_bind(device, &self.depth_layout, depth, &self.params);
    }

    /// The view this frame: its projection, the region drawn, and how far the
    /// occlusion reaches in view units.
    pub fn set_view(
        &self,
        queue: &wgpu::Queue,
        projection: Mat4,
        region: (u32, u32),
        radius: f32,
        strength: f32,
        seed: u32,
    ) {
        let params = Params {
            projection: projection.to_cols_array_2d(),
            inverse_projection: projection.inverse().to_cols_array_2d(),
            region: [region.0 as f32, region.1 as f32],
            radius,
            strength,
            seed: (seed % 64) as f32,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    /// Darkens `surfaces` where geometry closes it in. Reads the depth the
    /// surfaces wrote, so it runs after them and outside their pass.
    pub fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surfaces: &wgpu::TextureView,
        region: (u32, u32),
    ) {
        self.apply_with(encoder, surfaces, region, &self.apply);
    }

    /// [`Ao::apply`], into the rendered view's linear picture.
    pub fn apply_linear(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        picture: &wgpu::TextureView,
        region: (u32, u32),
    ) {
        self.apply_with(encoder, picture, region, &self.apply_linear);
    }

    fn apply_with(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surfaces: &wgpu::TextureView,
        region: (u32, u32),
        apply: &wgpu::RenderPipeline,
    ) {
        {
            let mut p = begin(encoder, &self.occlusion, "ao estimate", true, region);
            p.set_pipeline(&self.estimate);
            p.set_bind_group(0, &self.depth_bind, &[]);
            p.draw(0..3, 0..1);
        }
        let mut p = begin(encoder, surfaces, "ao apply", false, region);
        p.set_pipeline(apply);
        p.set_bind_group(0, &self.depth_bind, &[]);
        p.set_bind_group(1, &self.occlusion_bind, &[]);
        p.draw(0..3, 0..1);
    }
}

/// A colour-only pass over `region` of `view`.
fn begin<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &'a wgpu::TextureView,
    label: &str,
    clear: bool,
    region: (u32, u32),
) -> wgpu::RenderPass<'a> {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: if clear {
                    wgpu::LoadOp::Clear(wgpu::Color::WHITE)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_viewport(0.0, 0.0, region.0 as f32, region.1 as f32, 0.0, 1.0);
    pass
}
