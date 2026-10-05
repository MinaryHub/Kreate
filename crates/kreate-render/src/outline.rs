//! The selection outline in object mode: Blender's orange line around what is
//! selected, lighter around the active object. See `shaders/outline.wgsl`.
//!
//! A mask and an edge pass rather than a line per triangle edge: what is
//! outlined is the *silhouette* — the shape on screen — which no single
//! triangle knows, and the mask is tested against the picture's depth so the
//! hidden part of an object is not outlined through what is in front of it.

const SOURCE: &str = include_str!("shaders/outline.wgsl");

/// The shader, for the portability tests.
pub fn source() -> &'static str {
    SOURCE
}

const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// How far out the line reaches, in pixels.
pub const WIDTH_PX: f32 = 2.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    selected: [f32; 4],
    active: [f32; 4],
    width: [f32; 4],
}

pub struct Outline {
    mask_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    edge_layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    mask: wgpu::TextureView,
    edge_bind: wgpu::BindGroup,
    size: (u32, u32),
}

impl Outline {
    /// `globals` and `objects` are the solid pass's own layouts: the mask
    /// draws the same geometry with the same matrices.
    pub fn new(
        device: &wgpu::Device,
        target: wgpu::TextureFormat,
        globals: &wgpu::BindGroupLayout,
        objects: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outline"),
            source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
        });
        let mask_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("outline mask"),
            bind_group_layouts: &[Some(globals), Some(objects)],
            immediate_size: 0,
        });
        let mask_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("outline mask"),
            layout: Some(&mask_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("mask_vertex"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("mask_fragment"),
                compilation_options: Default::default(),
                // The strongest wins where two selected objects overlap on
                // screen, so the active one's colour is not lost behind another.
                targets: &[Some(wgpu::ColorTargetState {
                    format: MASK_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Max,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            // Against the picture's depth, not writing it: only what can be
            // seen is outlined. Nudged towards the eye, because the same
            // triangles drawn twice need not round to the same depth.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::gpu::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 4,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let edge_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("outline edge"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
        let edge_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("outline edge"),
            bind_group_layouts: &[None, None, Some(&edge_layout)],
            immediate_size: 0,
        });
        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("outline edge"),
            layout: Some(&edge_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("edge_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("edge_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let params = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("outline params"),
                contents: bytemuck::bytes_of(&Params {
                    selected: crate::palette::display_rgba(crate::palette::OUTLINE_SELECTED),
                    active: crate::palette::display_rgba(crate::palette::OUTLINE_ACTIVE),
                    width: [WIDTH_PX, 0.0, 0.0, 0.0],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let size = (width.max(1), height.max(1));
        let mask = Self::image(device, size);
        let edge_bind = Self::bind(device, &edge_layout, &mask, &params);
        Self {
            mask_pipeline,
            edge_pipeline,
            edge_layout,
            params,
            mask,
            edge_bind,
            size,
        }
    }

    fn image(device: &wgpu::Device, size: (u32, u32)) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("outline mask"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: MASK_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        mask: &wgpu::TextureView,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outline edge"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(mask),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    /// Follows the viewport's allocation.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size != self.size {
            self.size = size;
            self.mask = Self::image(device, size);
            self.edge_bind = Self::bind(device, &self.edge_layout, &self.mask, &self.params);
        }
    }

    /// Draws the outline onto `target`. `draws` is every draw with its slot in
    /// the per-object uniform ring, those to outline marked by the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn draw<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        region: (u32, u32),
        globals: &wgpu::BindGroup,
        objects: &wgpu::BindGroup,
        outlined: impl Iterator<Item = (u32, &'a wgpu::Buffer, &'a wgpu::Buffer, u32)>,
    ) {
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("outline mask"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mask,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, region.0 as f32, region.1 as f32, 0.0, 1.0);
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(0, globals, &[]);
            for (offset, positions, indices, triangles) in outlined {
                pass.set_bind_group(1, objects, &[offset]);
                pass.set_vertex_buffer(0, positions.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..triangles * 3, 0, 0..1);
            }
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("outline edge"),
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
        pass.set_viewport(0.0, 0.0, region.0 as f32, region.1 as f32, 0.0, 1.0);
        pass.set_pipeline(&self.edge_pipeline);
        pass.set_bind_group(2, &self.edge_bind, &[]);
        pass.draw(0..3, 0..1);
    }
}
