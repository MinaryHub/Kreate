//! Progressive accumulation: a still view refines itself. See
//! `shaders/accum.wgsl`.
//!
//! The frame that would have gone to the target goes to [`Accum::frame`]
//! instead, [`Accum::add`] averages it into a linear running mean, and
//! [`Accum::present`] puts the mean on the target. Once enough frames are in,
//! the viewport stops drawing the scene and only presents — so a view nobody
//! is moving costs less than it did before, not more.

const SOURCE: &str = include_str!("shaders/accum.wgsl");
const TEMPORAL: &str = include_str!("shaders/temporal.wgsl");

/// The shader, for the portability tests.
pub fn source() -> &'static str {
    SOURCE
}

/// The temporal pass's shader, for the portability tests.
pub fn temporal_source() -> &'static str {
    TEMPORAL
}

/// How much of a moving view's frame goes into its picture; the rest is the
/// last picture, carried along. A tenth: about the last ten frames' worth
/// of nudges, which is what smooths an edge, and quick enough that a change
/// is seen within a few frames.
pub const TEMPORAL_WEIGHT: f32 = 0.1;

/// What the temporal pass needs of this frame and the last.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TemporalParams {
    pub inv_current: [[f32; 4]; 4],
    pub previous: [[f32; 4]; 4],
    pub moved: [f32; 4],
    pub region: [f32; 2],
    pub jitter: [f32; 2],
    pub weight: f32,
    pub _pad: [f32; 3],
}

/// The running mean: linear and wide, so sixteen frames average without the
/// banding eight bits would give.
const MEAN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// How many frames a still view takes. Past this it only presents.
pub const SAMPLES: u32 = 16;

/// The sub-pixel offset of frame `n`, in pixels, each axis within half a
/// pixel. Frame zero is not offset, so the first picture after any change is
/// exactly the one a single frame would show. Halton (2, 3): spread evenly
/// at any count, which is what lets a mean of a few already look smooth.
pub fn jitter(n: u32) -> (f32, f32) {
    if n == 0 {
        return (0.0, 0.0);
    }
    fn halton(mut i: u32, base: u32) -> f32 {
        let (mut f, mut r) = (1.0f32, 0.0f32);
        while i > 0 {
            f /= base as f32;
            r += f * (i % base) as f32;
            i /= base;
        }
        r
    }
    (halton(n, 2) - 0.5, halton(n, 3) - 0.5)
}

pub struct Accum {
    add: wgpu::RenderPipeline,
    temporal: wgpu::RenderPipeline,
    temporal_layout: wgpu::BindGroupLayout,
    temporal_params: wgpu::Buffer,
    temporal_sampler: wgpu::Sampler,
    temporal_bind: Option<wgpu::BindGroup>,
    /// Where the temporal pass writes, before it is copied into the mean it
    /// read from.
    blended: wgpu::Texture,
    mean_texture: wgpu::Texture,
    present: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    target_format: wgpu::TextureFormat,
    frame: wgpu::TextureView,
    mean: wgpu::TextureView,
    frame_bind: wgpu::BindGroup,
    mean_bind: wgpu::BindGroup,
    size: (u32, u32),
}

impl Accum {
    pub fn new(
        device: &wgpu::Device,
        target: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("accum"),
            source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accum"),
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("accum"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline =
            |label: &str, format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("accum_vertex"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("accum_copy"),
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
        // mean = frame * w + mean * (1 - w), w = 1/n from the blend constant.
        let mean_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::OneMinusConstant,
            operation: wgpu::BlendOperation::Add,
        };
        let add = pipeline(
            "accum add",
            MEAN_FORMAT,
            Some(wgpu::BlendState {
                color: mean_blend,
                alpha: mean_blend,
            }),
        );
        let present = pipeline("accum present", target, None);

        let temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("temporal"),
            source: wgpu::ShaderSource::Wgsl(TEMPORAL.into()),
        });
        let texture = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let temporal_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("temporal"),
            entries: &[
                texture(0, false),
                texture(1, true),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
        let temporal_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("temporal"),
                bind_group_layouts: &[Some(&temporal_layout)],
                immediate_size: 0,
            });
        let temporal = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("temporal"),
            layout: Some(&temporal_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &temporal_shader,
                entry_point: Some("temporal_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &temporal_shader,
                entry_point: Some("temporal_resolve"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: MEAN_FORMAT,
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
        let temporal_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temporal params"),
            size: std::mem::size_of::<TemporalParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let temporal_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("temporal history"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let size = (width.max(1), height.max(1));
        let (frame, (mean_texture, mean), blended) = Self::images(device, target, size);
        let frame_bind = Self::bind(device, &layout, &frame);
        let mean_bind = Self::bind(device, &layout, &mean);
        Self {
            add,
            temporal,
            temporal_layout,
            temporal_params,
            temporal_sampler,
            temporal_bind: None,
            blended,
            mean_texture,
            present,
            layout,
            target_format: target,
            frame,
            mean,
            frame_bind,
            mean_bind,
            size,
        }
    }

    #[allow(clippy::type_complexity)]
    fn images(
        device: &wgpu::Device,
        target: wgpu::TextureFormat,
        size: (u32, u32),
    ) -> (
        wgpu::TextureView,
        (wgpu::Texture, wgpu::TextureView),
        wgpu::Texture,
    ) {
        let image = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let mean = image("accum mean", MEAN_FORMAT);
        let mean_view = view(&mean);
        (
            view(&image("accum frame", target)),
            (mean, mean_view),
            image("accum blended", MEAN_FORMAT),
        )
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accum"),
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
            let (frame, (mean_texture, mean), blended) =
                Self::images(device, self.target_format, size);
            (self.frame, self.mean, self.mean_texture, self.blended) =
                (frame, mean, mean_texture, blended);
            self.frame_bind = Self::bind(device, &self.layout, &self.frame);
            self.mean_bind = Self::bind(device, &self.layout, &self.mean);
            self.temporal_bind = None;
        }
    }

    /// Where this frame's picture goes, instead of the target.
    pub fn frame(&self) -> &wgpu::TextureView {
        &self.frame
    }

    /// Averages the frame into the mean as its `n`th sample, one-based.
    pub fn add(&self, encoder: &mut wgpu::CommandEncoder, region: (u32, u32), n: u32) {
        let mut pass = Self::begin(encoder, &self.mean, "accum add");
        pass.set_viewport(0.0, 0.0, region.0 as f32, region.1 as f32, 0.0, 1.0);
        let w = 1.0 / n.max(1) as f64;
        pass.set_blend_constant(wgpu::Color {
            r: w,
            g: w,
            b: w,
            a: w,
        });
        pass.set_pipeline(&self.add);
        pass.set_bind_group(0, &self.frame_bind, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Sets up this frame's temporal pass: its depth and the two cameras.
    pub fn prepare_temporal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth: &wgpu::TextureView,
        params: &TemporalParams,
    ) {
        queue.write_buffer(&self.temporal_params, 0, bytemuck::bytes_of(params));
        self.temporal_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("temporal"),
            layout: &self.temporal_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.frame),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.mean),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.temporal_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.temporal_params.as_entire_binding(),
                },
            ],
        }));
    }

    /// Folds the frame into the last picture, carried to this camera, and
    /// leaves the result as the mean: what a moving view shows, and where a
    /// view that then stops starts its own mean from. Takes the frame as it
    /// is when `prepare_temporal` has not been called.
    pub fn temporal(&self, encoder: &mut wgpu::CommandEncoder, region: (u32, u32)) {
        let Some(bind) = &self.temporal_bind else {
            self.add(encoder, region, 1);
            return;
        };
        {
            let view = self
                .blended
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut pass = Self::begin(encoder, &view, "temporal");
            pass.set_viewport(0.0, 0.0, region.0 as f32, region.1 as f32, 0.0, 1.0);
            pass.set_pipeline(&self.temporal);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            self.blended.as_image_copy(),
            self.mean_texture.as_image_copy(),
            wgpu::Extent3d {
                width: region.0.min(self.size.0),
                height: region.1.min(self.size.1),
                depth_or_array_layers: 1,
            },
        );
    }

    /// Puts the mean on `target`.
    pub fn present(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        region: (u32, u32),
    ) {
        let mut pass = Self::begin(encoder, target, "accum present");
        pass.set_viewport(0.0, 0.0, region.0 as f32, region.1 as f32, 0.0, 1.0);
        pass.set_pipeline(&self.present);
        pass.set_bind_group(0, &self.mean_bind, &[]);
        pass.draw(0..3, 0..1);
    }

    fn begin<'a>(
        encoder: &'a mut wgpu::CommandEncoder,
        view: &'a wgpu::TextureView,
        label: &str,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_frame_is_not_offset_and_the_rest_stay_within_half_a_pixel() {
        assert_eq!(jitter(0), (0.0, 0.0));
        let mut seen = std::collections::HashSet::new();
        for n in 1..SAMPLES {
            let (x, y) = jitter(n);
            assert!(x.abs() <= 0.5 && y.abs() <= 0.5, "{n}: {x} {y}");
            seen.insert(((x * 1000.0) as i32, (y * 1000.0) as i32));
        }
        assert_eq!(
            seen.len(),
            SAMPLES as usize - 1,
            "every offset is different"
        );
    }
}
