//! The display transform's resolve pass.
//!
//! Scene-linear light in a buffer, a displayable picture out. The formula
//! itself lives in `kreate_math::agx`; this is the GPU's copy of it and the
//! plumbing that runs it.
//!
//! # Why there are two copies at all
//!
//! Because the transform has to happen where the image is. A viewport's image
//! is on the GPU and a saved frame's pixels are on the CPU, and neither is
//! worth a round trip. §5's second risk is precisely what that costs when it is
//! done carelessly — two shading implementations that drift, and a bug report
//! that recurs forever — so the two are held together by a device test that
//! compares them value for value, the same arrangement as the octahedral normal
//! encoding.
//!
//! # Why a pass over a buffer
//!
//! The transform belongs at the end, once, on a linear image. Applying it
//! per-fragment would put it *before* blending, and blending two
//! display-referred colours is not blending light — which is the bug §2.1
//! records as a viewport composited seven times too dark.
//!
//! A buffer rather than a texture because that is what §2.5's progressive
//! accumulation resolves from, and because a storage buffer is the one binding
//! that behaves identically on every capability tier (§v1 4.7): storage
//! textures need a format check, and the first thing this is wanted for is the
//! rendered viewport on whatever adapter the user has.

use bytemuck::{Pod, Zeroable};

/// One invocation per colour.
const WORKGROUP: u32 = 64;

/// How the display transform is set for a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Display {
    pub view: kreate_math::ViewTransform,
    /// Stops. The picture's exposure, not a light's.
    pub exposure: f32,
}

impl Default for Display {
    fn default() -> Self {
        Self {
            view: kreate_math::ViewTransform::default(),
            exposure: 0.0,
        }
    }
}

impl Display {
    /// The CPU transform this pass is a copy of.
    ///
    /// Here so that a caller with one of these can produce the authoritative
    /// answer without assembling a second object, which is what the parity test
    /// needs and what a saved frame will use.
    pub fn on_cpu(&self) -> kreate_math::DisplayTransform {
        kreate_math::DisplayTransform {
            view: self.view,
            exposure: self.exposure,
        }
    }
}

/// The uniform block, matching `Display` in `agx.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct Params {
    exposure: f32,
    view: u32,
    count: u32,
    _pad: u32,
}

/// The transform's number on the wire.
///
/// A `match` rather than a `#[repr(u32)] as u32` cast, so that adding a
/// transform to the enum fails to compile here instead of silently sending a
/// number the shader reads as AgX. The numbers themselves are asserted against
/// the shader's constants in a test.
fn view_id(view: kreate_math::ViewTransform) -> u32 {
    match view {
        kreate_math::ViewTransform::AgX => 0,
        kreate_math::ViewTransform::Standard => 1,
    }
}

/// How many workgroups cover `items`.
///
/// No `.max(1)`: zero colours is zero workgroups, not one whose every thread
/// returns immediately. The deform pass had the `.max(1)` and it was a latent
/// dispatch over an empty buffer.
pub fn workgroups(items: u32) -> u32 {
    items.div_ceil(WORKGROUP)
}

/// The resolve pipeline.
pub struct DisplayResolve {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl DisplayResolve {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agx"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/agx.wgsl").into()),
        });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display resolve"),
            entries: &[
                storage(0, true),  // scene-linear
                storage(1, false), // shown
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
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
            label: Some("display resolve"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("display resolve"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { layout, pipeline }
    }

    /// The shader source, for the forward path to concatenate.
    ///
    /// There is no `#import` in this build, so a shader that needs the
    /// transform inline takes the text. Exposed as a function rather than
    /// copied into that shader, because a copy is the drift this module exists
    /// to prevent.
    pub fn source() -> &'static str {
        include_str!("shaders/agx.wgsl")
    }

    /// Resolves `count` colours from `scene` into `shown`.
    ///
    /// Both buffers are `vec4<f32>`: rgb is light, alpha is coverage and is
    /// carried through untouched — putting a coverage fraction through a tone
    /// curve would change the shape of an antialiased edge.
    ///
    /// `count` is the number of live colours and need not be the buffers'
    /// length: a viewport allocates at capacity and fills as far as the frame
    /// goes. Zero is allowed and dispatches nothing.
    ///
    /// Both buffers must be non-empty even then, because wgpu refuses a
    /// zero-sized binding — so "no colours to show" is a count of zero and
    /// never an empty buffer. Stated here because the alternative is a
    /// validation error at the bind group, three frames away from the caller
    /// who had nothing to draw.
    ///
    /// The uniform buffer is created here rather than held, because it is
    /// sixteen bytes once a frame and holding it would mean this type had a
    /// lifetime and a queue.
    pub fn resolve(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &wgpu::Buffer,
        shown: &wgpu::Buffer,
        display: Display,
        count: u32,
    ) {
        let params = Params {
            exposure: display.exposure,
            view: view_id(display.view),
            count,
            _pad: 0,
        };
        let uniform = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("display params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display resolve"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: shown.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        let groups = workgroups(count);
        if groups == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("display resolve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// The linear image a viewport shades into.
///
/// `Rgba16Float`, which is renderable and blendable in the WebGPU core and
/// holds the range scene-linear light needs: a surface under a sun is in the
/// thousands and an `Rgba8` target would have clipped it before the display
/// transform ever saw it.
///
/// Half precision rather than full: sixteen bits of float have about three
/// decimal digits of mantissa, which is far more than a display can show, and
/// the target is read once per pixel per frame so the bandwidth is the cost
/// that matters.
pub struct HdrTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// What a viewport shades into before the display transform.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

impl HdrTarget {
    /// The size a request actually gets.
    ///
    /// One is the floor. Separated so the rule is checkable without a device,
    /// which is what makes it a test rather than a comment.
    pub fn floor(width: u32, height: u32) -> (u32, u32) {
        (width.max(1), height.max(1))
    }

    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // A viewport is briefly zero-sized while a window is being laid out,
        // and a zero-sized texture cannot be created.
        let size = Self::floor(width, height);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport hdr"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            // Rendered into, read by the resolve, copied out for a saved
            // frame — and writable, because putting pixels into a linear
            // image is part of being one: a test seeds it today and a
            // resumed accumulation (§2.5) would seed it tomorrow. Usage
            // flags cost nothing but the validation they enable.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            size,
        }
    }

    /// Remakes the texture if the size changed, and says whether it did.
    ///
    /// Compared rather than remade unconditionally, because a texture
    /// reallocated every frame is a stall and a viewport is resized rarely.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let wanted = Self::floor(width, height);
        if wanted == self.size {
            return false;
        }
        *self = Self::new(device, wanted.0, wanted.1);
        true
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }
}

/// The fullscreen pass that turns the linear image into a picture.
///
/// Separate from [`DisplayResolve`], which resolves a *buffer* for §2.5's
/// progressive accumulation. The two share the shader and therefore the
/// transform; what differs is where the pixels come from, and a viewport's come
/// from a texture it just rendered into.
pub struct ScreenResolve {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl ScreenResolve {
    /// `format` is the window's, and the pipeline is built for it.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agx screen"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/agx.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display screen"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display screen"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display screen"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("resolve_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("resolve_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // No blending: this pass replaces the target. Blending a
                    // display-referred result into whatever was there is the
                    // mistake §2.1 records as compositing in the wrong space.
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // No depth: a fullscreen pass has nothing to test against.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { layout, pipeline }
    }

    /// Draws `source` through the display transform onto `target`.
    pub fn draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        target: &wgpu::TextureView,
        display: Display,
    ) {
        let params = Params {
            exposure: display.exposure,
            view: view_id(display.view),
            // Unused by the screen pass, which covers whatever it is given.
            count: 0,
            _pad: 0,
        };
        let uniform = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("display screen params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display screen"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(source),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("display screen"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Cleared and then covered: the triangle writes every
                    // pixel, and a load would be a read of memory nothing
                    // reads.
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        // Three vertices, one triangle, no buffers.
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_view_transform_numbers_are_the_shaders_numbers() {
        // The wire format between the two implementations, and the one part of
        // it a device test cannot catch: a mismatch here would run the *other*
        // transform, and both produce a picture. Read out of the shader source
        // so that changing one and not the other fails.
        let source = DisplayResolve::source();
        for (view, name) in [
            (kreate_math::ViewTransform::AgX, "VIEW_AGX"),
            (kreate_math::ViewTransform::Standard, "VIEW_STANDARD"),
        ] {
            let declaration = format!("const {name}: u32 = {}u;", view_id(view));
            assert!(
                source.contains(&declaration),
                "the shader does not declare `{declaration}`"
            );
        }
    }

    #[test]
    fn the_shader_and_the_cpu_agree_on_the_exposure_range() {
        // Two constants that have to be the same number in two languages, and
        // whose disagreement would be a picture that is subtly the wrong
        // brightness rather than an error.
        let source = DisplayResolve::source();
        assert!(
            source.contains(&format!("MIN_EV: f32 = {}", kreate_math::AGX_MIN_EV)),
            "MIN_EV differs from {}",
            kreate_math::AGX_MIN_EV
        );
        assert!(
            source.contains(&format!("MAX_EV: f32 = {}", kreate_math::AGX_MAX_EV)),
            "MAX_EV differs from {}",
            kreate_math::AGX_MAX_EV
        );
    }

    #[test]
    fn an_hdr_target_is_never_zero_sized() {
        // A viewport is briefly zero-sized while a window is laid out, and a
        // zero-sized texture cannot be created.
        assert_eq!(HdrTarget::floor(0, 0), (1, 1));
        assert_eq!(HdrTarget::floor(1920, 0), (1920, 1));
        assert_eq!(HdrTarget::floor(1920, 1080), (1920, 1080));
    }

    #[test]
    fn the_hdr_format_holds_more_than_a_display_can() {
        // The whole reason for an intermediate: a surface under a sun is in
        // the thousands, and an eight-bit target would clip it before the
        // display transform saw it — which is the same mistake as clamping in
        // the shader, one pass later.
        assert_eq!(HDR_FORMAT, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn nothing_to_resolve_dispatches_nothing() {
        assert_eq!(workgroups(0), 0);
        assert_eq!(workgroups(1), 1);
        assert_eq!(workgroups(64), 1);
        assert_eq!(workgroups(65), 2);
    }

    #[test]
    fn the_shader_parses() {
        // Cheap, and it catches the whole class of mistake that only a device
        // finds otherwise. The deform shader shipped once with a construct
        // WGSL forbids, and the only reason it was found was a device test —
        // which a headless runner skips.
        naga::front::wgsl::parse_str(DisplayResolve::source()).expect("agx.wgsl parses");
    }

    #[test]
    fn the_pass_is_configured_by_something_a_frame_can_hold() {
        // `Display` is `Copy` and small: a viewport keeps one, an operator sets
        // it, and nothing has to be rebuilt when the exposure slider moves. A
        // transform that required rebuilding the pipeline would put a stall in
        // the middle of dragging a slider.
        let display = Display::default();
        assert_eq!(display.view, kreate_math::ViewTransform::AgX);
        assert_eq!(display.exposure, 0.0);
        assert_eq!(display.on_cpu(), kreate_math::DisplayTransform::DEFAULT);
    }
}
