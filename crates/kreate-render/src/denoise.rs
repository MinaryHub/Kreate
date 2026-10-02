//! Denoising for the path tracer. See `shaders/denoise.wgsl`.
//!
//! Strong while the mean is young and fading as it fills: a few samples in,
//! the picture is mostly the filter's; a few hundred in, it is the mean's own.

const SOURCE: &str = include_str!("shaders/denoise.wgsl");

/// The shader, for the portability tests.
pub fn source() -> &'static str {
    SOURCE
}

const FORMAT: wgpu::TextureFormat = crate::display::HDR_FORMAT;
const PASSES: usize = 5;

/// How much of the filtered picture to keep at `samples`: all of it for the
/// first few, half by thirty-two, a sliver by five hundred.
pub fn keep(samples: u32) -> f32 {
    (32.0 / (32.0 + samples.saturating_sub(4) as f32)).clamp(0.0, 1.0)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    region: [f32; 2],
    step: f32,
    keep: f32,
    sigma: f32,
    _pad: [f32; 3],
}

/// The relative noise a mean of `samples` has, near enough: what two pixels
/// of one surface may differ by.
fn sigma(samples: u32) -> f32 {
    (0.6 / (samples.max(1) as f32).sqrt()).max(0.05)
}

pub struct Denoise {
    step: wgpu::RenderPipeline,
    blend: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    filtered_layout: wgpu::BindGroupLayout,
    params: Vec<wgpu::Buffer>,
    /// The raw mean, copied out of the picture; then two to pass between.
    raw: (wgpu::Texture, wgpu::TextureView),
    ping: [wgpu::TextureView; 2],
    binds: Vec<wgpu::BindGroup>,
    filtered_bind: Option<wgpu::BindGroup>,
    size: (u32, u32),
}

impl Denoise {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("denoise"),
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("denoise"),
            entries: &[
                texture(0, false),
                texture(1, false),
                texture(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let filtered_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("denoise filtered"),
            entries: &[texture(0, false)],
        });
        let pipeline = |label: &str, layouts: &[Option<&wgpu::BindGroupLayout>], entry: &str| {
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
                    entry_point: Some("denoise_vertex"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let step = pipeline("denoise step", &[Some(&layout)], "denoise_step");
        let blend = pipeline("denoise blend", &[Some(&layout), Some(&filtered_layout)], "denoise_blend");
        let params = (0..=PASSES)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("denoise params"),
                    size: std::mem::size_of::<Params>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let size = (width.max(1), height.max(1));
        let (raw, ping) = Self::images(device, size);
        Self { step, blend, layout, filtered_layout, params, raw, ping, binds: Vec::new(), filtered_bind: None, size }
    }

    fn images(device: &wgpu::Device, size: (u32, u32)) -> ((wgpu::Texture, wgpu::TextureView), [wgpu::TextureView; 2]) {
        let image = |label, copy: bool| {
            let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
            if copy {
                usage |= wgpu::TextureUsages::COPY_DST;
            }
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage,
                view_formats: &[],
            })
        };
        let raw = image("denoise raw", true);
        let raw_view = raw.create_view(&wgpu::TextureViewDescriptor::default());
        let ping = [
            image("denoise a", false).create_view(&wgpu::TextureViewDescriptor::default()),
            image("denoise b", false).create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        ((raw, raw_view), ping)
    }

    /// Follows the viewport's allocation.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size != self.size {
            self.size = size;
            (self.raw, self.ping) = Self::images(device, size);
            self.binds.clear();
            self.filtered_bind = None;
        }
    }

    /// This frame's guides and how strong the filter is.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        normals: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        region: (u32, u32),
        samples: u32,
    ) {
        let bind = |source: &wgpu::TextureView, params: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("denoise"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normals) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
                ],
            })
        };
        // Pass i reads the raw picture first, then the image the pass before
        // it wrote; the blend reads the raw again, beside the last.
        let mut binds = Vec::with_capacity(PASSES + 1);
        for i in 0..PASSES {
            let source = if i == 0 { &self.raw.1 } else { &self.ping[(i - 1) % 2] };
            binds.push(bind(source, &self.params[i]));
            let p = Params {
                region: [region.0 as f32, region.1 as f32],
                step: (1u32 << i) as f32,
                keep: 0.0,
                sigma: sigma(samples),
                _pad: [0.0; 3],
            };
            queue.write_buffer(&self.params[i], 0, bytemuck::bytes_of(&p));
        }
        binds.push(bind(&self.raw.1, &self.params[PASSES]));
        let p = Params {
            region: [region.0 as f32, region.1 as f32],
            step: 1.0,
            keep: keep(samples),
            sigma: sigma(samples),
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.params[PASSES], 0, bytemuck::bytes_of(&p));
        self.binds = binds;
        self.filtered_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("denoise filtered"),
            layout: &self.filtered_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.ping[(PASSES - 1) % 2]),
            }],
        }));
    }

    /// Filters `picture` in place: copied out, filtered, blended back.
    pub fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        picture: &wgpu::Texture,
        picture_view: &wgpu::TextureView,
        region: (u32, u32),
    ) {
        let Some(filtered_bind) = &self.filtered_bind else { return };
        if self.binds.len() != PASSES + 1 {
            return;
        }
        encoder.copy_texture_to_texture(
            picture.as_image_copy(),
            self.raw.0.as_image_copy(),
            wgpu::Extent3d { width: region.0, height: region.1, depth_or_array_layers: 1 },
        );
        let run = |encoder: &mut wgpu::CommandEncoder,
                   target: &wgpu::TextureView,
                   pipeline: &wgpu::RenderPipeline,
                   bind: &wgpu::BindGroup,
                   extra: Option<&wgpu::BindGroup>| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("denoise"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, region.0 as f32, region.1 as f32, 0.0, 1.0);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            if let Some(extra) = extra {
                pass.set_bind_group(1, extra, &[]);
            }
            pass.draw(0..3, 0..1);
        };
        for i in 0..PASSES {
            run(encoder, &self.ping[i % 2], &self.step, &self.binds[i], None);
        }
        run(encoder, picture_view, &self.blend, &self.binds[PASSES], Some(filtered_bind));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_filter_fades_as_the_mean_fills() {
        assert_eq!(super::keep(1), 1.0);
        assert!(super::keep(36) > 0.45 && super::keep(36) < 0.55);
        assert!(super::keep(512) < 0.07);
    }
}
