//! Screen-space reflections for the rendered view. See `shaders/ssr.wgsl`.
//!
//! Owns the three images the rendered pass writes beside its light — normal
//! and roughness, specular share, the environment's reflection — because they
//! exist for this pass and nothing else reads them.

use glam::Mat4;

const SOURCE: &str = include_str!("shaders/ssr.wgsl");

/// The shader, for the portability tests.
pub fn source() -> &'static str {
    SOURCE
}

/// The format of the three extra images and of the reflections.
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    projection: [[f32; 4]; 4],
    inverse_projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    region: [f32; 2],
    reach: f32,
    thickness: f32,
}

pub struct Ssr {
    trace: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    trace_layout: wgpu::BindGroupLayout,
    result_layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    /// Normal and roughness, specular share, the environment's reflection.
    gbuffer: [wgpu::TextureView; 3],
    reflections: wgpu::TextureView,
    trace_bind: Option<wgpu::BindGroup>,
    result_bind: wgpu::BindGroup,
    size: (u32, u32),
}

impl Ssr {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr"),
            source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
        });
        let texture = |binding: u32, depth: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: if depth {
                    wgpu::TextureSampleType::Depth
                } else {
                    wgpu::TextureSampleType::Float { filterable: false }
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr trace"),
            entries: &[
                texture(0, true),
                texture(1, false),
                texture(2, false),
                texture(3, false),
                texture(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
        let result_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr result"),
            entries: &[texture(0, false)],
        });
        let pipeline = |label: &str,
                        layouts: &[Option<&wgpu::BindGroupLayout>],
                        entry: &str,
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
                    entry_point: Some("ssr_vertex"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: FORMAT,
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
        let trace = pipeline("ssr trace", &[Some(&trace_layout)], "ssr_trace", None);
        // Added onto the picture, which is the same format; signed, because
        // the swap takes the sky's reflection away where it adds the thing's.
        debug_assert_eq!(FORMAT, crate::display::HDR_FORMAT);
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
        let composite = pipeline(
            "ssr composite",
            &[None, Some(&result_layout)],
            "ssr_composite",
            Some(add),
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssr params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let size = (width.max(1), height.max(1));
        let (gbuffer, reflections) = Self::images(device, size);
        let result_bind = Self::bind_result(device, &result_layout, &reflections);
        Self {
            trace,
            composite,
            trace_layout,
            result_layout,
            params,
            gbuffer,
            reflections,
            trace_bind: None,
            result_bind,
            size,
        }
    }

    fn images(
        device: &wgpu::Device,
        size: (u32, u32),
    ) -> ([wgpu::TextureView; 3], wgpu::TextureView) {
        let image = |label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
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
        };
        (
            [
                image("ssr surface"),
                image("ssr share"),
                image("ssr environment"),
            ],
            image("ssr reflections"),
        )
    }

    fn bind_result(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr result"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        })
    }

    /// Follows the viewport's allocation.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size != self.size {
            self.size = size;
            (self.gbuffer, self.reflections) = Self::images(device, size);
            self.result_bind = Self::bind_result(device, &self.result_layout, &self.reflections);
            self.trace_bind = None;
        }
    }

    /// The three images the rendered pass writes after its light, in order.
    pub fn gbuffer(&self) -> &[wgpu::TextureView; 3] {
        &self.gbuffer
    }

    /// This frame's view, and the picture and depth the rays walk.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth: &wgpu::TextureView,
        picture: &wgpu::TextureView,
        projection: Mat4,
        view: Mat4,
        region: (u32, u32),
        reach: f32,
    ) {
        // Rebound every frame: the depth and the picture are remade with the
        // allocation, and a bind group is cheap.
        self.trace_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr trace"),
            layout: &self.trace_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(picture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.gbuffer[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.gbuffer[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.gbuffer[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.params.as_entire_binding(),
                },
            ],
        }));
        let p = Params {
            projection: projection.to_cols_array_2d(),
            inverse_projection: projection.inverse().to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            region: [region.0 as f32, region.1 as f32],
            reach,
            thickness: reach * 0.05,
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// Swaps the sky's reflection for what the screen holds, on `picture`.
    pub fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        picture: &wgpu::TextureView,
        region: (u32, u32),
    ) {
        let Some(trace_bind) = &self.trace_bind else {
            return;
        };
        let pass = |encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, clear: bool| {
            encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ssr"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if clear {
                                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
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
                })
                .forget_lifetime()
        };
        {
            let mut p = pass(encoder, &self.reflections, true);
            p.set_viewport(0.0, 0.0, region.0 as f32, region.1 as f32, 0.0, 1.0);
            p.set_pipeline(&self.trace);
            p.set_bind_group(0, trace_bind, &[]);
            p.draw(0..3, 0..1);
        }
        let mut p = pass(encoder, picture, false);
        p.set_viewport(0.0, 0.0, region.0 as f32, region.1 as f32, 0.0, 1.0);
        p.set_pipeline(&self.composite);
        p.set_bind_group(1, &self.result_bind, &[]);
        p.draw(0..3, 0..1);
    }
}
