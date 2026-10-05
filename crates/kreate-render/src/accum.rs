//! Progressive accumulation: a still view refines itself. See
//! `shaders/accum.wgsl`.
//!
//! The frame that would have gone to the target goes to [`Accum::frame`]
//! instead, [`Accum::add`] averages it into a linear running mean, and
//! [`Accum::present`] puts the mean on the target. Once enough frames are in,
//! the viewport stops drawing the scene and only presents — so a view nobody
//! is moving costs less than it did before, not more.

const SOURCE: &str = include_str!("shaders/accum.wgsl");

/// The shader, for the portability tests.
pub fn source() -> &'static str {
    SOURCE
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
        let size = (width.max(1), height.max(1));
        let (frame, mean) = Self::images(device, target, size);
        let frame_bind = Self::bind(device, &layout, &frame);
        let mean_bind = Self::bind(device, &layout, &mean);
        Self {
            add,
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

    fn images(
        device: &wgpu::Device,
        target: wgpu::TextureFormat,
        size: (u32, u32),
    ) -> (wgpu::TextureView, wgpu::TextureView) {
        let image = |label, format| {
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
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        (
            image("accum frame", target),
            image("accum mean", MEAN_FORMAT),
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
            (self.frame, self.mean) = Self::images(device, self.target_format, size);
            self.frame_bind = Self::bind(device, &self.layout, &self.frame);
            self.mean_bind = Self::bind(device, &self.layout, &self.mean);
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
