//! The composed shader has to shade the way the CPU does.
//!
//! The two libraries were each checked on their own. This checks what happens
//! when they are put together — which is a separate claim, because the
//! composition is a text operation and the loop over lights is new code that
//! exists on both sides.
//!
//! What the loop decides is not visible in a picture, which is exactly why it
//! is worth a test:
//!
//! - Lights add. A renderer taking the largest, or averaging, looks plausible
//!   under one lamp and wrong under two — and two is every scene anybody makes.
//! - A light can skip a lobe (§2.2's cheats), per light rather than per
//!   surface, so two lights on one point can differ.
//! - Nothing is clamped. A surface under a sun is in the thousands, and the
//!   display transform is the one place a picture gets a range.

use kreate_render::brdf::{albedo_table, Surface};
use kreate_render::lights::{Falloff, GpuLight, Lights, PlacedLight};
use kreate_render::shading::{self, Point};
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

/// A scene lit only by its lamps. See `shading::shade`.
const NO_WORLD: [f32; 3] = [0.0; 3];

const RELATIVE: f32 = 2e-3;
const ABSOLUTE: f32 = 1e-4;

fn close(got: f32, want: f32) -> bool {
    let scale = got.abs().max(want.abs());
    (got - want).abs() <= (scale * RELATIVE).max(ABSOLUTE)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Sample {
    material: [f32; 4],
    params: [f32; 4],
    position: [f32; 4],
    normal: [f32; 4],
    view: [f32; 4],
}

fn sample_of(point: &Point) -> Sample {
    let base = point.surface.base_color.to_array();
    Sample {
        material: [base[0], base[1], base[2], point.surface.metallic],
        params: [point.surface.roughness, 0.0, 0.0, 0.0],
        position: [point.position.x, point.position.y, point.position.z, 0.0],
        normal: [point.normal.x, point.normal.y, point.normal.z, 0.0],
        view: [point.view.x, point.view.y, point.view.z, 0.0],
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Shading {
    light_count: u32,
    pad: [u32; 3],
    world: [f32; 4],
}

/// Runs the composed probe over `points` against `lights`.
fn on_gpu(gpu: &Gpu, points: &[Point], lights: &[PlacedLight], count: Option<u32>) -> Vec<[f32; 3]> {
    use wgpu::util::DeviceExt as _;

    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shading probe"),
            source: wgpu::ShaderSource::Wgsl(shading::probe_source().into()),
        });

    let dfg = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dfg"),
            contents: bytemuck::cast_slice(&kreate_render::brdf::dfg_buffer(albedo_table())),
            usage: wgpu::BufferUsages::STORAGE,
        });
    // Through `Lights`, so the test exercises the packing and upload a frame
    // uses rather than a hand-built buffer.
    let mut light_buffer = Lights::new(&gpu.device, 1);
    let live = light_buffer.upload(&gpu.device, &gpu.queue, glam::DVec3::ZERO, lights);
    let shading_globals = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shading globals"),
            contents: bytemuck::bytes_of(&Shading {
                light_count: count.unwrap_or(live),
                pad: [0; 3],
                // These tests are about the lamps; the world has its own.
                world: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let samples: Vec<Sample> = points.iter().map(sample_of).collect();
    let inputs = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shading samples"),
            contents: bytemuck::cast_slice(&samples),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let sentinel = vec![[-7.0f32; 4]; points.len()];
    let answers = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shading answers"),
            contents: bytemuck::cast_slice(&sentinel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Probe {
        count: u32,
        pad: [u32; 3],
    }
    let probe = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shading probe params"),
            contents: bytemuck::bytes_of(&Probe {
                count: points.len() as u32,
                pad: [0; 3],
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
    let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    // Group 0 is the probe's own; group 2 is the shading data the libraries
    // declare. The convention is the whole reason the three files compose.
    let own = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shading probe"),
            entries: &[storage(0, true), storage(1, false), uniform(2)],
        });
    let shared = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shading data"),
            entries: &[storage(0, true), storage(1, true), uniform(2)],
        });
    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shading probe"),
            bind_group_layouts: &[Some(&own), None, Some(&shared)],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shading probe"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("probe_shading"),
            compilation_options: Default::default(),
            cache: None,
        });
    let own_bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shading probe"),
        layout: &own,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.as_entire_binding(),
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
    let shared_bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shading data"),
        layout: &shared,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dfg.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: light_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: shading_globals.as_entire_binding(),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shading probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &own_bind, &[]);
        pass.set_bind_group(2, &shared_bind, &[]);
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

/// A lighting rig with every kind of decision in it.
fn a_rig() -> Vec<PlacedLight> {
    vec![
        PlacedLight {
            position: glam::DVec3::new(1.0, 0.5, 3.0),
            amount: [800.0, 700.0, 600.0],
            radius: 0.05,
            ..PlacedLight::default()
        },
        // Pointing *away* from the camera, so it lights the side the camera
        // can see. A sun pointing the other way lights the far side of
        // everything, which is a fixture that measures nothing.
        PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::new(0.2, -0.3, 0.93).normalize(),
            amount: [12_000.0, 11_500.0, 10_000.0],
            angular_radius: 0.0046,
            ..PlacedLight::default()
        },
        PlacedLight {
            position: glam::DVec3::new(-2.0, 0.0, 2.0),
            direction: glam::Vec3::new(0.6, 0.0, -0.8).normalize(),
            amount: [4000.0; 3],
            cone: Some((50f32.to_radians(), 0.3)),
            ..PlacedLight::default()
        },
        // A rim light: specular only, which is §2.2's cheat and a per-light
        // decision the loop has to read.
        PlacedLight {
            position: glam::DVec3::new(0.0, -3.0, 1.0),
            amount: [2000.0; 3],
            affect_diffuse: false,
            ..PlacedLight::default()
        },
        // And a fill that must not show as a second highlight.
        PlacedLight {
            position: glam::DVec3::new(3.0, 2.0, 0.5),
            amount: [900.0; 3],
            affect_specular: false,
            ..PlacedLight::default()
        },
    ]
}

/// Points across a surface, with materials that exercise both lobes.
///
/// The camera is at the origin, because that is what eye-relative space means,
/// so a point out at `+z` faces it with a normal pointing back toward `-z`.
/// Getting that backwards makes every evaluation zero — and both sides agree on
/// zero, so the parity test passes while measuring nothing. That happened, and
/// `the_gpu_shades_the_way_the_cpu_does` now checks that the rig lights
/// something.
fn some_points() -> Vec<Point> {
    let mut out = Vec::new();
    for (metallic, roughness) in [(0.0f32, 0.25f32), (0.0, 0.8), (1.0, 0.15), (1.0, 0.7), (0.4, 0.5)]
    {
        let surface = Surface {
            base_color: kreate_math::SceneRgb::new(0.85, 0.6, 0.35),
            metallic,
            roughness,
        };
        for i in 0..9 {
            let a = (i as f32 / 9.0) * std::f32::consts::TAU;
            let position = glam::Vec3::new(a.cos(), a.sin(), 4.0 + 0.5 * a.sin());
            // Back toward the camera, tilted around so the lights strike at a
            // spread of angles rather than all head-on.
            let normal =
                (-position.normalize() + glam::Vec3::new(a.cos() * 0.5, a.sin() * 0.5, 0.0))
                    .normalize();
            out.push(Point::facing_the_eye(position, normal, surface));
        }
    }
    out
}

#[test]
fn the_gpu_shades_the_way_the_cpu_does() {
    // The composition, end to end: two libraries concatenated, a loop over a
    // light buffer, and both sides agreeing on the sum.
    let gpu = gpu_or_skip!(gpu());
    let rig = a_rig();
    let points = some_points();

    let packed: Vec<GpuLight> = rig
        .iter()
        .map(|l| GpuLight::pack(l, glam::DVec3::ZERO))
        .collect();
    let got = on_gpu(&gpu, &points, &rig, None);
    assert_eq!(got.len(), points.len());

    for (point, lit) in points.iter().zip(&got) {
        let want = shading::shade(point, &packed, NO_WORLD);
        for c in 0..3 {
            assert!(lit[c].is_finite(), "{point:?} gave {lit:?}");
            assert!(
                close(lit[c], want[c]),
                "channel {c} at {:?}: GPU {} against CPU {}",
                point.position,
                lit[c],
                want[c]
            );
        }
    }

    // The fixture has to light something, or the comparison above is two
    // implementations agreeing on zero. That is not hypothetical: the first
    // version of this test had its normals facing away from the camera, every
    // evaluation was zero on both sides, and it passed.
    let lit_points = got.iter().filter(|c| c[0] > 1e-3).count();
    assert!(
        lit_points * 4 >= points.len(),
        "only {lit_points} of {} points were lit",
        points.len()
    );
    let brightest = got.iter().fold(0.0f32, |m, c| m.max(c[0]));
    assert!(brightest > 1.0, "nothing in the rig is bright: {brightest}");
}

#[test]
fn the_lit_result_is_not_clamped_on_the_device_either() {
    // A surface under a sun is in the thousands. A shader clamping to one
    // would look correct in every dim scene and blow out every bright one, and
    // the display transform would have nothing left to work with.
    let gpu = gpu_or_skip!(gpu());
    let noon = vec![PlacedLight {
        falloff: Falloff::Parallel,
        // Pointing away from the camera, so it lights the visible side.
        direction: glam::Vec3::Z,
        amount: [100_000.0; 3],
        ..PlacedLight::default()
    }];
    let point = Point::facing_the_eye(
        glam::Vec3::new(0.0, 0.0, 5.0),
        glam::Vec3::NEG_Z,
        Surface::default(),
    );
    let lit = on_gpu(&gpu, &[point], &noon, None)[0];
    assert!(lit[0] > 100.0, "the sun was clamped away: {lit:?}");

    let packed = [GpuLight::pack(&noon[0], glam::DVec3::ZERO)];
    assert!(close(lit[0], shading::shade(&point, &packed, NO_WORLD)[0]));
}

#[test]
fn the_device_reads_the_count_and_not_the_buffers_length() {
    // The buffer is grown and never shrunk, so the slots past the count hold
    // an earlier frame's lights. A loop to the array's length would light the
    // scene from lamps the user deleted — and the light left in the tail here
    // is bright enough that including it could not be missed.
    let gpu = gpu_or_skip!(gpu());
    let rig = a_rig();
    let point = Point::facing_the_eye(
        glam::Vec3::new(0.0, 0.0, 4.0),
        glam::Vec3::NEG_Z,
        Surface::default(),
    );

    let all = on_gpu(&gpu, &[point], &rig, None)[0][0];
    let first_two = on_gpu(&gpu, &[point], &rig, Some(2))[0][0];
    let none = on_gpu(&gpu, &[point], &rig, Some(0))[0][0];

    assert_eq!(none, 0.0, "a count of zero still lit something");
    assert!(first_two > 0.0);
    assert!(first_two < all, "{first_two} against {all}");

    // And two lights is exactly the first two, not some other pair.
    let packed: Vec<GpuLight> = rig
        .iter()
        .take(2)
        .map(|l| GpuLight::pack(l, glam::DVec3::ZERO))
        .collect();
    assert!(close(first_two, shading::shade(&point, &packed, NO_WORLD)[0]));
}

#[test]
fn the_per_light_lobe_switches_reach_the_device() {
    // §2.2's cheats, at the far end of the path they travel: a datablock's
    // boolean, through the app's conversion, through the packing, into a flag
    // the shader reads per light. A rim light that blew out a highlight would
    // be a production complaint and not a crash.
    let gpu = gpu_or_skip!(gpu());
    // Between the camera and the surface, so it lights the visible side.
    let at = glam::DVec3::new(0.6, 0.0, 2.5);
    let point = Point::facing_the_eye(
        glam::Vec3::new(0.0, 0.0, 4.0),
        glam::Vec3::NEG_Z,
        Surface {
            base_color: kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.3,
        },
    );

    let full = vec![PlacedLight {
        position: at,
        amount: [3000.0; 3],
        ..PlacedLight::default()
    }];
    let no_specular = vec![PlacedLight {
        affect_specular: false,
        ..full[0]
    }];
    let no_diffuse = vec![PlacedLight {
        affect_diffuse: false,
        ..full[0]
    }];

    let both = on_gpu(&gpu, &[point], &full, None)[0][0];
    let d = on_gpu(&gpu, &[point], &no_specular, None)[0][0];
    let s = on_gpu(&gpu, &[point], &no_diffuse, None)[0][0];

    assert!(d > 0.0 && s > 0.0, "{d} {s}");
    assert!(
        (d + s - both).abs() <= both * RELATIVE,
        "{d} + {s} against {both}"
    );
}

#[test]
fn lights_add_on_the_device() {
    // The arithmetic of light. A shader that took the largest would look
    // plausible under one lamp, and every scene has more than one.
    let gpu = gpu_or_skip!(gpu());
    let point = Point::facing_the_eye(
        glam::Vec3::new(0.0, 0.0, 4.0),
        glam::Vec3::NEG_Z,
        Surface::default(),
    );
    let a = PlacedLight {
        position: glam::DVec3::new(-1.0, 0.0, 2.5),
        amount: [1500.0; 3],
        ..PlacedLight::default()
    };
    let b = PlacedLight {
        position: glam::DVec3::new(1.0, 0.0, 2.5),
        amount: [1500.0; 3],
        ..PlacedLight::default()
    };

    let only_a = on_gpu(&gpu, &[point], &[a], None)[0][0];
    let only_b = on_gpu(&gpu, &[point], &[b], None)[0][0];
    let both = on_gpu(&gpu, &[point], &[a, b], None)[0][0];
    assert!(only_a > 0.0 && only_b > 0.0);
    assert!(
        (both - (only_a + only_b)).abs() <= both * RELATIVE,
        "{only_a} + {only_b} against {both}"
    );
}
