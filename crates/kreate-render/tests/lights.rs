//! The GPU's light has to be the CPU's light.
//!
//! Two claims, and the first is the one that would be most expensive to get
//! wrong quietly.
//!
//! # The struct layout
//!
//! A `GpuLight` that disagrees with the shader's `Light` is silent and total:
//! every light after the first reads its fields out of its neighbour. The
//! picture is not broken, it is merely wrong — a spot pointing somewhere else,
//! a lamp at the wrong brightness — and nothing about it says "layout". So the
//! tests below use *several* lights with distinct values and check each one,
//! rather than one light which any offset would still read correctly.
//!
//! # The arithmetic
//!
//! `illuminance_at` is the function a forward pass will call per light per
//! fragment, and it exists on both sides. Two implementations of one formula
//! drift (§5's risk 2), so the shader is held to the CPU's answer value for
//! value — and the probe entry point exists precisely so this can be checked
//! without a material, a camera or a triangle in the way. A shading pass that
//! disagreed would otherwise have to be debugged through a picture.

use kreate_render::lights::{Falloff, GpuLight, Lights, PlacedLight};
use kreate_render::GpuOptions;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// The one device this test binary uses. See `deform.rs` for why it is shared.
fn gpu() -> Option<Gpu> {
    static SHARED: std::sync::OnceLock<Option<(wgpu::Device, wgpu::Queue)>> =
        std::sync::OnceLock::new();
    let (device, queue) = SHARED
        .get_or_init(|| {
            let opts = GpuOptions::default();
            let instance = kreate_render::Gpu::create_instance(&opts);
            let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
            Some((gpu.device.clone(), gpu.queue.clone()))
        })
        .clone()?;
    Some(Gpu { device, queue })
}

macro_rules! gpu_or_skip {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => {
                eprintln!("no GPU adapter; skipping");
                return;
            }
        }
    };
}

/// A quarter of a lux at the magnitudes these tests use.
///
/// The two sides differ in `inverseSqrt` against `sqrt` and in the order of a
/// couple of multiplies, so bit equality is not the claim. A quarter of a lux
/// out of hundreds is far below anything a picture could show.
const TOLERANCE: f32 = 0.25;

/// Runs the probe pass for one light over `points`, and reads back the lux.
fn on_gpu(
    gpu: &Gpu,
    lights: &[PlacedLight],
    eye: glam::DVec3,
    which: u32,
    points: &[glam::Vec3],
) -> Vec<[f32; 3]> {
    use wgpu::util::DeviceExt as _;

    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lights probe"),
            source: wgpu::ShaderSource::Wgsl(kreate_render::lights::shader_source().into()),
        });

    // Uploaded through `Lights`, so the test exercises the packing and the
    // upload path a frame uses rather than a hand-built buffer.
    let mut buffer = Lights::new(&gpu.device, 1);
    let count = buffer.upload(&gpu.device, &gpu.queue, eye, lights);
    assert_eq!(count as usize, lights.len());

    let padded: Vec<[f32; 4]> = points.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
    let points_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe points"),
            contents: bytemuck::cast_slice(&padded),
            usage: wgpu::BufferUsages::STORAGE,
        });
    // Filled with a value the arithmetic cannot produce, so an unwritten entry
    // is a visible sentinel rather than a plausible zero.
    let sentinel = vec![[-7.0f32; 4]; points.len()];
    let answers = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe answers"),
            contents: bytemuck::cast_slice(&sentinel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Probe {
        count: u32,
        light: u32,
        pad: [u32; 2],
    }
    let probe = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe params"),
            contents: bytemuck::bytes_of(&Probe {
                count: points.len() as u32,
                light: which,
                pad: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
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
    let layout = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lights probe"),
            entries: &[
                storage(0, true),
                storage(1, false),
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
    // The library owns group 2; the probe's own buffers are group 0. Groups
    // between are unused, which the layout says with `None`.
    let shading = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lights shading"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: kreate_render::lights::LIGHTS_BINDING,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lights probe"),
            bind_group_layouts: &[Some(&layout), None, Some(&shading)],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lights probe"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("probe_illuminance"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lights probe"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: answers.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: probe.as_entire_binding(),
            },
        ],
    });
    let shading_bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lights shading"),
        layout: &shading,
        entries: &[wgpu::BindGroupEntry {
            binding: kreate_render::lights::LIGHTS_BINDING,
            resource: buffer.buffer().as_entire_binding(),
        }],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lights probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_bind_group(kreate_render::lights::LIGHTS_GROUP, &shading_bind, &[]);
        pass.dispatch_workgroups((points.len() as u32).div_ceil(64), 1, 1);
    }

    let size = std::mem::size_of_val(&sentinel[..]) as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&answers, 0, &staging, 0, size);
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    let read: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
    read.into_iter().map(|v| [v[0], v[1], v[2]]).collect()
}

/// Points worth asking about: near, far, behind, off-axis, and on the light.
fn probe_points() -> Vec<glam::Vec3> {
    let mut out = vec![
        glam::Vec3::ZERO,
        glam::Vec3::new(0.0, 0.0, -1.0),
        glam::Vec3::new(0.0, 0.0, -3.0),
        glam::Vec3::new(0.0, 0.0, 3.0),
        glam::Vec3::new(2.0, 0.0, -2.0),
        glam::Vec3::new(-1.5, 0.7, -0.3),
        glam::Vec3::new(0.0, 0.0, -100.0),
        glam::Vec3::new(0.01, 0.0, 0.0),
    ];
    // An arc through a cone's edge, so the fade is sampled rather than sampled
    // at one point and assumed.
    for degrees in 0..=90 {
        let angle = (degrees as f32).to_radians();
        out.push(glam::Vec3::new(angle.sin(), 0.0, -angle.cos()) * 4.0);
    }
    out
}

/// Every light this suite compares, deliberately all different.
fn a_lighting_rig() -> Vec<PlacedLight> {
    vec![
        // A bulb, off the origin so a packed position of zero would show.
        PlacedLight {
            position: glam::DVec3::new(0.5, -0.25, 1.0),
            amount: [1000.0, 800.0, 600.0],
            radius: 0.05,
            ..PlacedLight::default()
        },
        // A sun, pointing somewhere that is not an axis.
        PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::new(0.3, -0.6, -0.74).normalize(),
            amount: [20_000.0, 19_000.0, 17_000.0],
            angular_radius: 0.00459,
            ..PlacedLight::default()
        },
        // A spot with a soft edge.
        PlacedLight {
            position: glam::DVec3::new(0.0, 0.0, 2.0),
            direction: glam::Vec3::NEG_Z,
            amount: [5000.0, 5000.0, 5000.0],
            radius: 0.02,
            cone: Some((60f32.to_radians(), 0.35)),
            ..PlacedLight::default()
        },
        // A spot with a hard edge, which is where a zero-width fade could
        // divide by zero.
        PlacedLight {
            position: glam::DVec3::new(1.0, 1.0, 1.0),
            direction: glam::Vec3::new(-1.0, -1.0, -1.0).normalize(),
            amount: [300.0, 100.0, 50.0],
            cone: Some((30f32.to_radians(), 0.0)),
            cast_shadow: false,
            affect_diffuse: false,
            ..PlacedLight::default()
        },
        // An area light as the viewport sees one: a point at its axial candela.
        PlacedLight {
            position: glam::DVec3::new(-2.0, 0.5, 0.5),
            direction: glam::Vec3::X,
            amount: [2000.0, 2000.0, 1800.0],
            radius: 0.5,
            ..PlacedLight::default()
        },
    ]
}

fn compare(gpu: &Gpu, eye: glam::DVec3) {
    let rig = a_lighting_rig();
    let points = probe_points();

    for (which, light) in rig.iter().enumerate() {
        let packed = GpuLight::pack(light, eye);
        let theirs = on_gpu(gpu, &rig, eye, which as u32, &points);
        assert_eq!(theirs.len(), points.len());

        for (at, got) in points.iter().zip(&theirs) {
            let want = packed.illuminance_at(*at);
            for c in 0..3 {
                assert!(
                    got[c].is_finite(),
                    "light {which} at {at}: GPU gave {got:?}"
                );
                assert!(
                    (got[c] - want[c]).abs() <= TOLERANCE,
                    "light {which} at {at} channel {c}: GPU {} against CPU {}",
                    got[c],
                    want[c]
                );
            }
        }
    }
}

#[test]
fn the_gpu_agrees_with_the_cpu_on_every_light_in_a_rig() {
    // Five lights, all different, each read out of the same array. A layout
    // that was off by a field would still read light zero correctly and get
    // every later one wrong, which is why the rig is not one lamp.
    let gpu = gpu_or_skip!(gpu());
    compare(&gpu, glam::DVec3::ZERO);
}

#[test]
fn they_still_agree_with_the_camera_far_from_the_origin() {
    // The eye-relative packing, end to end. A shader reading absolute world
    // positions would pass the test above and fail here — which is the failure
    // that shows up as a scene that lights correctly near the origin and goes
    // dark when the user pans away.
    let gpu = gpu_or_skip!(gpu());
    compare(&gpu, glam::DVec3::new(1e6, -4e6, 2.5e6));
}

#[test]
fn a_lights_fields_are_read_from_its_own_slot() {
    // The layout claim, stated as bluntly as it can be. If `GpuLight`'s size
    // disagreed with the shader's `Light`, the second light would read a
    // position made of the first one's trailing fields — so the two lights here
    // are given amounts that cannot be confused.
    let gpu = gpu_or_skip!(gpu());
    let rig = vec![
        PlacedLight {
            position: glam::DVec3::new(0.0, 0.0, 1.0),
            amount: [1.0, 0.0, 0.0],
            ..PlacedLight::default()
        },
        PlacedLight {
            position: glam::DVec3::new(0.0, 0.0, 1.0),
            amount: [0.0, 0.0, 1000.0],
            ..PlacedLight::default()
        },
    ];
    let at = [glam::Vec3::ZERO];

    let first = on_gpu(&gpu, &rig, glam::DVec3::ZERO, 0, &at)[0];
    let second = on_gpu(&gpu, &rig, glam::DVec3::ZERO, 1, &at)[0];
    assert!(first[0] > 0.0 && first[2] == 0.0, "{first:?}");
    assert!(second[2] > 0.0 && second[0] == 0.0, "{second:?}");
}

#[test]
fn the_buffer_grows_and_every_light_survives_the_growth() {
    // A scene being built one lamp at a time. A grow that reallocated without
    // re-uploading would leave the new light reading whatever the fresh buffer
    // held, which is zeroes — a light that is there and does nothing.
    let gpu = gpu_or_skip!(gpu());
    let mut buffer = Lights::new(&gpu.device, 1);
    assert_eq!(buffer.capacity(), 1);
    assert_eq!(buffer.count(), 0);

    let mut rig = Vec::new();
    for i in 0..9 {
        rig.push(PlacedLight {
            position: glam::DVec3::new(i as f64, 0.0, 0.0),
            amount: [(i + 1) as f32 * 100.0; 3],
            ..PlacedLight::default()
        });
        let count = buffer.upload(&gpu.device, &gpu.queue, glam::DVec3::ZERO, &rig);
        assert_eq!(count as usize, rig.len());
        assert!(buffer.capacity() >= count, "the buffer did not grow");
    }

    // And the last light still reads correctly out of the grown buffer.
    let at = [glam::Vec3::new(8.0, 0.0, 1.0)];
    let got = on_gpu(&gpu, &rig, glam::DVec3::ZERO, 8, &at)[0];
    let want = GpuLight::pack(&rig[8], glam::DVec3::ZERO).illuminance_at(at[0]);
    assert!((got[0] - want[0]).abs() <= TOLERANCE, "{got:?} vs {want:?}");
    assert!(got[0] > 0.0, "the grown buffer held zeroes");
}

#[test]
fn the_count_is_the_authority_and_not_the_buffers_length() {
    // The buffer is never shrunk, so after a scene loses lights the slots past
    // the count still hold the previous frame's. Nothing is wrong with that —
    // it is why growth is cheap — but it means `count` is the only thing that
    // says where the live lights end, and a consumer looping to the buffer's
    // length would light the scene from lamps the user deleted.
    let gpu = gpu_or_skip!(gpu());
    let mut buffer = Lights::new(&gpu.device, 1);

    let many: Vec<PlacedLight> = (0..9)
        .map(|i| PlacedLight {
            position: glam::DVec3::new(i as f64, 0.0, 0.0),
            amount: [500.0; 3],
            ..PlacedLight::default()
        })
        .collect();
    assert_eq!(buffer.upload(&gpu.device, &gpu.queue, glam::DVec3::ZERO, &many), 9);
    let grown = buffer.capacity();
    assert!(grown >= 9);

    // Now two. The capacity stays, the count does not.
    let few = &many[..2];
    assert_eq!(buffer.upload(&gpu.device, &gpu.queue, glam::DVec3::ZERO, few), 2);
    assert_eq!(buffer.count(), 2);
    assert_eq!(buffer.capacity(), grown, "the buffer was reallocated to shrink");

    // And the stale slot is still readable, which is exactly why the count has
    // to be obeyed rather than inferred.
    let at = [glam::Vec3::new(8.0, 0.0, 1.0)];
    let stale = on_gpu(&gpu, &many, glam::DVec3::ZERO, 8, &at)[0];
    assert!(stale[0] > 0.0, "the eighth slot should still hold a light");
}

#[test]
fn no_lights_is_a_state_and_not_an_error() {
    // A scene with nothing in it, and a buffer that still exists because wgpu
    // refuses a zero-sized binding.
    let gpu = gpu_or_skip!(gpu());
    let mut buffer = Lights::new(&gpu.device, 0);
    assert_eq!(buffer.capacity(), 1);
    let count = buffer.upload(&gpu.device, &gpu.queue, glam::DVec3::ZERO, &[]);
    assert_eq!(count, 0);
    assert_eq!(buffer.count(), 0);
}
