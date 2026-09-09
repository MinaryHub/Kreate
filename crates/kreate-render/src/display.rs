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
