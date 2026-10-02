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
    /// One tile, holding every light. See the list built in `on_gpu`.
    tiles_across: u32,
    tile_capacity: u32,
    pad: [u32; 2],
    /// rgb: the flat world. w: nonzero when the cube is what lights the scene.
    world: [f32; 4],
    /// The world's rotation as a cosine and a sine, and the roughest mip.
    environment: [f32; 4],
    sh9: [[f32; 4]; 9],
    /// The background's inverse projection. The probe does not draw one, and
    /// the field still has to be here: a uniform shorter than the struct the
    /// shader declares is a validation error three frames from the cause.
    inv_view_proj: [[f32; 4]; 4],
    /// The camera for contact shadows, which the probe leaves off.
    view_proj: [[f32; 4]; 4],
}

/// Runs the composed probe over `points` against `lights`.
///
/// No world at all, which is a scene lit by its lamps. The tests that are about
/// an environment go through [`under`] instead.
fn on_gpu(gpu: &Gpu, points: &[Point], lights: &[PlacedLight], count: Option<u32>) -> Vec<[f32; 3]> {
    on_gpu_under(gpu, points, lights, count, [0.0; 4], None)
}

/// The same, under a world.
///
/// `world` carries its own flag in `w`: zero means the colour lights the scene
/// and nonzero means the cube does. Passing an environment and leaving the flag
/// down is a thing a caller can do on purpose — it is how "the cube is bound
/// and ignored" gets checked.
fn on_gpu_under(
    gpu: &Gpu,
    points: &[Point],
    lights: &[PlacedLight],
    count: Option<u32>,
    world: [f32; 4],
    environment: Option<&kreate_render::environment::Environment>,
) -> Vec<[f32; 3]> {
    use wgpu::util::DeviceExt as _;

    // Always something, because a bind group cannot have a hole in it — the
    // same reason the viewport keeps a placeholder.
    let placeholder;
    let environment = match environment {
        Some(e) => e,
        None => {
            placeholder = kreate_render::environment::Environment::placeholder(&gpu.device);
            &placeholder
        }
    };

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
                tiles_across: 1,
                tile_capacity: kreate_render::tiles::MAX_LIGHTS_PER_TILE,
                pad: [0; 2],
                world,
                // No rotation: a turn is its own claim and belongs in a test
                // that makes it rather than in every other test's background.
                environment: {
                    let radians = ROTATION.with(|r| r.get());
                    // Strength one unless a test says otherwise: a dimmer in
                    // the background of all of them would scale every expected
                    // value for no reason, and nothing setting it at all is
                    // how it went untested.
                    [
                        radians.cos(),
                        radians.sin(),
                        environment.roughest(),
                        STRENGTH.with(|s| s.get()),
                    ]
                },
                sh9: kreate_render::environment::sh9_bytes(environment.sh9()),
                inv_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
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
    // Group 2 is the shading data the libraries declare, and it now carries
    // the shadows. These tests are about the lamps, so the entries are present
    // and empty: one shadow per light with `enabled` at zero, and a one-texel
    // depth array nothing samples.
    //
    // Present rather than absent because a bind group layout is all or
    // nothing — and because a probe that dropped them would stop compiling the
    // same shader the viewport runs, which is the whole point of the probe.
    let shadows = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadows"),
            contents: bytemuck::cast_slice(&vec![
                kreate_render::shadow::GpuShadow::NONE;
                lights.len().max(1)
            ]),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let shadow_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("no shadows"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    // One tile holding every light, which is what the probe means by "shade
    // this point": it has no screen and no camera, so there is nothing for a
    // tile to cull against. Present rather than absent because the probe has to
    // compile the same shader the viewport runs — which is the whole point of
    // it — and that shader reads this.
    // As many as the caller says are live, not as many as the buffer holds.
    // The buffer is grown and never shrunk, so the slots past the count still
    // carry an earlier frame's lights — and in the running renderer it is the
    // cull pass that respects the count, which is what
    // `a_cull_never_lists_a_light_past_the_count` holds it to.
    let live_count = count.unwrap_or(live);
    let mut list: Vec<u32> = vec![live_count];
    list.extend(0..live_count);
    // The shading loop clamps the count to the capacity, so the buffer has to
    // be as long as it claims.
    list.resize(kreate_render::tiles::MAX_LIGHTS_PER_TILE as usize + 1, 0);
    let tile_lights = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tile lights"),
            contents: bytemuck::cast_slice(&list),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let shared = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shading data"),
            entries: &[
                storage(0, true),
                storage(1, true),
                uniform(2),
                storage(3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(5, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
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
    let contact_depth = gpu
        .device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("contact depth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shadows.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&shadow_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: tile_lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(environment.view()),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(environment.sampler()),
            },
            // Contact shadows are off in the probe (its region is zero), and
            // a binding still has to be there.
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&contact_depth),
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
        // A lamp with an end, at a range the probe points are inside of — so
        // the window is somewhere in the middle of its curve rather than at
        // either flat end, which is where the two copies of it can disagree.
        //
        // Short on purpose. A range worked out from the display is hundreds of
        // metres, and at that distance the window is one everywhere a test
        // could sample it: the shader's copy would go unchecked.
        PlacedLight {
            position: glam::DVec3::new(-1.0, 1.5, 2.5),
            amount: [1500.0; 3],
            range: Some(4.0),
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

/// An equirectangular `Rgba16Float` image of one colour.
fn uniform_sky(width: u32, height: u32, value: f32) -> Vec<u8> {
    let half = half::f16::from_f32(value).to_le_bytes();
    let one = half::f16::from_f32(1.0).to_le_bytes();
    let mut out = Vec::with_capacity((width * height) as usize * 8);
    for _ in 0..width * height {
        for _ in 0..3 {
            out.extend_from_slice(&half);
        }
        out.extend_from_slice(&one);
    }
    out
}

/// Surfaces across the range that tells the two paths apart.
///
/// A metal and a dielectric, rough and smooth: the split between the specular
/// and diffuse halves moves with all four, and a path that had one of them
/// wrong would agree with the other on a mid-grey plastic.
fn spread() -> Vec<Point> {
    let mut out = Vec::new();
    for metallic in [0.0f32, 1.0] {
        for roughness in [0.05f32, 0.4, 1.0] {
            for normal in [glam::Vec3::Z, glam::Vec3::new(0.6, 0.0, 0.8).normalize()] {
                out.push(Point {
                    position: glam::Vec3::new(0.0, 0.0, 0.0),
                    normal,
                    view: glam::Vec3::Z,
                    surface: Surface {
                        base_color: kreate_math::SceneRgb::new(0.8, 0.6, 0.4),
                        metallic,
                        roughness,
                    },
                });
            }
        }
    }
    out
}

#[test]
fn a_uniform_environment_lights_a_scene_exactly_as_the_same_flat_colour_does() {
    // **The identity the image path is built on.** A world that is one colour
    // in every direction is not a special case of an environment — it is one,
    // and the shading function takes an irradiance and a prefiltered radiance
    // so that both arrive the same way.
    //
    // If this drifts, picking a plain grey sky lights a scene differently from
    // typing the same grey as a flat colour, and nobody would attribute that to
    // the prefilter. It is also the only check that the cube's units and the
    // coefficients' units agree: the diffuse half divides by `π` and the
    // coefficients carry the cosine lobe, and an error in either is a constant
    // factor on half the answer.
    let gpu = gpu_or_skip!(gpu());
    let points = spread();
    const L: f32 = 3.0;

    let flat = on_gpu_under(&gpu, &points, &[], None, [L, L, L, 0.0], None);

    let sky = uniform_sky(64, 32, L);
    let environment =
        kreate_render::environment::Environment::new(&gpu.device, &gpu.queue, 64, 32, &sky);
    let imaged = on_gpu_under(
        &gpu,
        &points,
        &[],
        None,
        [0.0, 0.0, 0.0, 1.0],
        Some(&environment),
    );

    for (i, (a, b)) in flat.iter().zip(&imaged).enumerate() {
        for c in 0..3 {
            let scale = a[c].abs().max(0.01);
            assert!(
                (a[c] - b[c]).abs() < scale * 0.05,
                "point {i} channel {c}: flat {} against imaged {}",
                a[c],
                b[c]
            );
        }
    }

    // And it is not passing because both are zero.
    assert!(
        flat.iter().any(|v| v[0] > 0.1),
        "the flat world lit nothing, so the comparison says nothing"
    );
}

#[test]
fn a_white_surface_under_an_imaged_furnace_returns_what_it_received() {
    // The furnace test, moved onto the image path. §7.5 makes this claim of the
    // flat world and most renderers cannot make it at all; it has to survive
    // the move or the environment is quietly absorbing energy.
    let gpu = gpu_or_skip!(gpu());
    const L: f32 = 2.0;
    let sky = uniform_sky(64, 32, L);
    let environment =
        kreate_render::environment::Environment::new(&gpu.device, &gpu.queue, 64, 32, &sky);

    let points: Vec<Point> = [0.05f32, 0.3, 0.7, 1.0]
        .iter()
        .map(|roughness| Point {
            position: glam::Vec3::ZERO,
            normal: glam::Vec3::Z,
            view: glam::Vec3::Z,
            surface: Surface {
                base_color: kreate_math::SceneRgb::new(1.0, 1.0, 1.0),
                metallic: 1.0,
                roughness: *roughness,
            },
        })
        .collect();

    let out = on_gpu_under(
        &gpu,
        &points,
        &[],
        None,
        [0.0, 0.0, 0.0, 1.0],
        Some(&environment),
    );
    for (i, v) in out.iter().enumerate() {
        assert!(
            (v[0] - L).abs() < L * 0.05,
            "a white metal at roughness {} returned {} under {L}",
            points[i].surface.roughness,
            v[0]
        );
    }
}

#[test]
fn a_bound_environment_that_is_switched_off_does_not_light_anything() {
    // The flag, and what it is for: a world holds one background or the other,
    // and a cube that is bound but not selected must contribute nothing. The
    // cube is always bound — a bind group cannot have a hole — so "bound" and
    // "in use" are different things and only the flag says which.
    let gpu = gpu_or_skip!(gpu());
    let sky = uniform_sky(64, 32, 50.0);
    let environment =
        kreate_render::environment::Environment::new(&gpu.device, &gpu.queue, 64, 32, &sky);
    let points = spread();

    let off = on_gpu_under(
        &gpu,
        &points,
        &[],
        None,
        [0.0; 4],
        Some(&environment),
    );
    for (i, v) in off.iter().enumerate() {
        assert!(
            v.iter().all(|c| *c < 1e-3),
            "point {i} received {v:?} from an environment that is switched off"
        );
    }
}

#[test]
fn turning_the_world_turns_what_a_surface_reflects() {
    // A rotation nobody applied is a slider that does nothing, which is the
    // failure §7.7 found four times over. And it has to reach *both* halves:
    // a diffuse term that turns while the specular stays put is a surface lit
    // from one side and reflecting another.
    let gpu = gpu_or_skip!(gpu());
    // Bright on the +X side of the horizon, dark elsewhere. `u ≈ 0.75` is +X —
    // see `environment::equirect_uv`.
    let (width, height) = (128u32, 64u32);
    let mut pixels = Vec::with_capacity((width * height) as usize * 8);
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let lit = (v - 0.5).abs() < 0.25 && (u - 0.75).abs() < 0.12;
            let value = half::f16::from_f32(if lit { 20.0 } else { 0.0 }).to_le_bytes();
            for _ in 0..3 {
                pixels.extend_from_slice(&value);
            }
            pixels.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
    }
    let environment = kreate_render::environment::Environment::new(
        &gpu.device,
        &gpu.queue,
        width,
        height,
        &pixels,
    );

    // A rough white surface facing +X, so what it receives is dominated by the
    // bright quarter — and a quarter turn should take that away.
    let point = Point {
        position: glam::Vec3::ZERO,
        normal: glam::Vec3::X,
        view: glam::Vec3::X,
        surface: Surface {
            base_color: kreate_math::SceneRgb::new(1.0, 1.0, 1.0),
            metallic: 0.0,
            roughness: 0.8,
        },
    };
    let facing = on_gpu_under(
        &gpu,
        std::slice::from_ref(&point),
        &[],
        None,
        [0.0, 0.0, 0.0, 1.0],
        Some(&environment),
    )[0][0];

    // Half a turn puts the bright quarter behind the surface.
    let turned = with_rotation(&gpu, &point, &environment, std::f32::consts::PI);
    assert!(
        facing > turned * 3.0,
        "facing the bright side gives {facing} and turned away {turned}"
    );
}

/// The probe again, with the world turned by `radians`.
///
/// Its own function rather than another argument on `on_gpu_under`, because
/// every other test would then carry a rotation it does not care about.
fn with_rotation(
    gpu: &Gpu,
    point: &Point,
    environment: &kreate_render::environment::Environment,
    radians: f32,
) -> f32 {
    // The rotation lives in the uniform, so this is the same call with a
    // different `environment` vector — spelled by building the globals here.
    let saved = ROTATION.with(|r| r.replace(radians));
    let out = on_gpu_under(
        gpu,
        std::slice::from_ref(point),
        &[],
        None,
        [0.0, 0.0, 0.0, 1.0],
        Some(environment),
    )[0][0];
    ROTATION.with(|r| r.set(saved));
    out
}

thread_local! {
    /// What `on_gpu_under` writes into the globals' rotation.
    ///
    /// A thread local rather than an argument, so that the dozen call sites
    /// that have nothing to do with a rotation do not grow one.
    static ROTATION: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };
}

/// A world bright enough to matter beside the lamps.
///
/// Not black, which is what every test here used before and what made the
/// GPU/CPU comparison blind to the whole ambient path: `ambient_from` could
/// lose its factor of `π` and both sides would still agree on zero.
const A_WORLD: [f32; 4] = [0.35, 0.3, 0.25, 0.0];

#[test]
fn the_gpu_shades_the_way_the_cpu_does_under_a_world_too() {
    // The same claim as `the_gpu_shades_the_way_the_cpu_does`, with the light
    // that is always there switched on. Its own test rather than a change to
    // that one, because "the lamps agree" and "the world agrees" are different
    // claims and a failure should say which.
    //
    // This is the test that was missing. A sweep that removed the `π` from
    // either side of the ambient arithmetic walked through every device test in
    // this file, because the only fixtures that lit anything lit it with lamps.
    let gpu = gpu_or_skip!(gpu());
    let rig = a_rig();
    let points = some_points();
    let packed: Vec<GpuLight> = rig
        .iter()
        .map(|l| GpuLight::pack(l, glam::DVec3::ZERO))
        .collect();

    let got = on_gpu_under(&gpu, &points, &rig, None, A_WORLD, None);
    let world = [A_WORLD[0], A_WORLD[1], A_WORLD[2]];
    for (point, lit) in points.iter().zip(&got) {
        let want = shading::shade(point, &packed, world);
        for c in 0..3 {
            assert!(
                close(lit[c], want[c]),
                "channel {c} at {:?}: GPU {} against CPU {}",
                point.position,
                lit[c],
                want[c]
            );
        }
    }

    // And the world is doing something: the same points with no world at all
    // come out darker. Without this the comparison is again two
    // implementations agreeing on a term neither of them reached.
    let dark = on_gpu(&gpu, &points, &rig, None);
    assert!(
        got.iter().zip(&dark).any(|(a, b)| a[0] > b[0] + 0.01),
        "the world changed nothing"
    );
}

#[test]
fn the_world_alone_shades_the_way_the_cpu_does() {
    // No lamps at all, so what is compared is the ambient term by itself.
    // Sharper than the test above, where a lamp's contribution could hide a
    // factor in the world's.
    let gpu = gpu_or_skip!(gpu());
    let points = some_points();
    let got = on_gpu_under(&gpu, &points, &[], None, A_WORLD, None);
    let world = [A_WORLD[0], A_WORLD[1], A_WORLD[2]];
    for (point, lit) in points.iter().zip(&got) {
        let want = shading::ambient(&point.surface, point.normal, point.view, world);
        for c in 0..3 {
            assert!(
                close(lit[c], want[c]),
                "channel {c}: GPU {} against CPU {}",
                lit[c],
                want[c]
            );
        }
    }
    assert!(got.iter().any(|v| v[0] > 0.01), "the world lit nothing");
}

/// An environment bright on the `+X` side of the horizon and dark elsewhere.
///
/// Directional, because a uniform sky cannot tell a reflection from a normal,
/// a rough mip from a sharp one, or a rotation from no rotation — and those
/// are three of the things the shading path decides.
fn a_directional_sky(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height) as usize * 8);
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            // `u ≈ 0.75` is +X. See `environment::equirect_uv`.
            let lit = (v - 0.5).abs() < 0.3 && (u - 0.75).abs() < 0.15;
            let value = half::f16::from_f32(if lit { 30.0 } else { 0.02 }).to_le_bytes();
            for _ in 0..3 {
                out.extend_from_slice(&value);
            }
            out.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
    }
    out
}

/// One surface under a directional environment.
fn under_sky(
    gpu: &Gpu,
    environment: &kreate_render::environment::Environment,
    normal: glam::Vec3,
    view: glam::Vec3,
    roughness: f32,
    metallic: f32,
) -> f32 {
    let point = Point {
        position: glam::Vec3::ZERO,
        normal,
        view,
        surface: Surface {
            base_color: kreate_math::SceneRgb::new(1.0, 1.0, 1.0),
            metallic,
            roughness,
        },
    };
    on_gpu_under(
        gpu,
        std::slice::from_ref(&point),
        &[],
        None,
        [0.0, 0.0, 0.0, 1.0],
        Some(environment),
    )[0][0]
}

#[test]
fn a_mirror_reflects_what_is_behind_the_viewer_and_not_what_it_faces() {
    // The reflection direction, which a uniform environment cannot see: every
    // direction gives the same colour, so reading the cube along the normal
    // instead of the mirror passes every test built on one.
    //
    // A smooth metal facing away from the bright quarter, viewed from the
    // direction that puts that quarter in its reflection. What it faces is
    // dark; what it reflects is not.
    let gpu = gpu_or_skip!(gpu());
    let sky = a_directional_sky(128, 64);
    let environment =
        kreate_render::environment::Environment::new(&gpu.device, &gpu.queue, 128, 64, &sky);

    // Normal along -X: the surface faces away from the bright side. Viewed from
    // a direction whose mirror about that normal points at +X.
    let normal = glam::Vec3::NEG_X;
    let view = glam::Vec3::new(-1.0, 0.0, 0.0);
    let facing_away = under_sky(&gpu, &environment, normal, view, 0.05, 1.0);

    // And the same surface turned to face the bright side reflects it.
    let toward = under_sky(&gpu, &environment, glam::Vec3::X, glam::Vec3::X, 0.05, 1.0);
    assert!(
        toward > facing_away * 5.0,
        "a mirror facing the bright quarter returned {toward} and one facing away {facing_away}"
    );
}

#[test]
fn a_rough_metal_reflects_a_blurrier_sky_than_a_smooth_one() {
    // The roughness-to-mip step. Under a uniform environment every level holds
    // the same colour, so a shader that always read level zero would pass every
    // test built on one — and a rough metal would mirror the sky.
    let gpu = gpu_or_skip!(gpu());
    let sky = a_directional_sky(128, 64);
    let environment =
        kreate_render::environment::Environment::new(&gpu.device, &gpu.queue, 128, 64, &sky);

    // Pointed at the bright quarter, where a blur can only take light away.
    let smooth = under_sky(&gpu, &environment, glam::Vec3::X, glam::Vec3::X, 0.02, 1.0);
    let rough = under_sky(&gpu, &environment, glam::Vec3::X, glam::Vec3::X, 1.0, 1.0);
    assert!(
        smooth > rough * 2.0,
        "a smooth metal returned {smooth} and a rough one {rough}"
    );
}

#[test]
fn a_quarter_turn_moves_the_sky_the_way_a_half_turn_cannot_show() {
    // A half turn makes `+sin` and `-sin` the same number, so the rotation's
    // *sign* is invisible to it — which is what the first version of this
    // asked, and what the sweep walked through.
    let gpu = gpu_or_skip!(gpu());
    let sky = a_directional_sky(128, 64);
    let environment =
        kreate_render::environment::Environment::new(&gpu.device, &gpu.queue, 128, 64, &sky);

    let facing = |radians: f32| {
        let saved = ROTATION.with(|r| r.replace(radians));
        let out = under_sky(&gpu, &environment, glam::Vec3::X, glam::Vec3::X, 0.05, 1.0);
        ROTATION.with(|r| r.set(saved));
        out
    };
    let quarter = std::f32::consts::FRAC_PI_2;
    let one_way = facing(quarter);
    let other_way = facing(-quarter);
    assert!(
        (one_way - other_way).abs() > (one_way + other_way) * 0.25,
        "turning the sky a quarter each way gave {one_way} and {other_way}, \
         which is a rotation with no direction"
    );
}

#[test]
fn the_worlds_strength_dims_both_halves_of_the_answer() {
    // Nothing set a strength other than one, so a shader that applied it to the
    // specular lobe and forgot the diffuse — or forgot it entirely — passed.
    // It is a dimmer on the *environment*, so both lobes have to move together.
    let gpu = gpu_or_skip!(gpu());
    let sky = a_directional_sky(128, 64);
    let environment =
        kreate_render::environment::Environment::new(&gpu.device, &gpu.queue, 128, 64, &sky);

    let at = |strength: f32, metallic: f32| {
        let saved = STRENGTH.with(|s| s.replace(strength));
        let out = under_sky(&gpu, &environment, glam::Vec3::X, glam::Vec3::X, 0.4, metallic);
        STRENGTH.with(|s| s.set(saved));
        out
    };
    // A metal, which is specular only, and a dielectric, which is mostly
    // diffuse. Halving the strength has to halve both.
    for metallic in [0.0f32, 1.0] {
        let full = at(1.0, metallic);
        let half = at(0.5, metallic);
        assert!(full > 0.01, "metallic {metallic} was not lit at all");
        assert!(
            (half - full * 0.5).abs() < full * 0.05,
            "metallic {metallic}: full {full}, half {half}"
        );
    }
}

thread_local! {
    /// What `on_gpu_under` writes into the globals' strength.
    static STRENGTH: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
}

#[test]
fn a_surface_facing_away_from_a_sun_is_never_darker_than_black() {
    // A truncated harmonic series undershoots around a small bright source, and
    // a negative irradiance *darkens* a surface the environment is lighting —
    // a shadow nothing casts. The CPU has a test for it; the shader's own
    // `max` did not, and a sweep that removed it walked through everything.
    //
    // A sun in an otherwise black sky is the fixture that produces the ringing,
    // and a dielectric is what shows it: a metal has no diffuse lobe for a
    // negative irradiance to reach.
    let gpu = gpu_or_skip!(gpu());
    let (width, height) = (128u32, 64u32);
    let mut pixels = Vec::with_capacity((width * height) as usize * 8);
    for y in 0..height {
        for x in 0..width {
            let sun = y == 12 && x == 40;
            let value = half::f16::from_f32(if sun { 4000.0 } else { 0.0 }).to_le_bytes();
            for _ in 0..3 {
                pixels.extend_from_slice(&value);
            }
            pixels.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
    }
    let environment =
        kreate_render::environment::Environment::new(&gpu.device, &gpu.queue, width, height, &pixels);

    // Every direction, because which one rings is a property of where the sun
    // is and not something worth pinning to a fixture.
    let points: Vec<Point> = [
        glam::Vec3::X,
        glam::Vec3::NEG_X,
        glam::Vec3::Y,
        glam::Vec3::NEG_Y,
        glam::Vec3::Z,
        glam::Vec3::NEG_Z,
    ]
    .iter()
    .map(|normal| Point {
        position: glam::Vec3::ZERO,
        normal: *normal,
        view: *normal,
        surface: Surface {
            base_color: kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.6,
        },
    })
    .collect();

    let out = on_gpu_under(
        &gpu,
        &points,
        &[],
        None,
        [0.0, 0.0, 0.0, 1.0],
        Some(&environment),
    );
    for (point, lit) in points.iter().zip(&out) {
        for c in 0..3 {
            assert!(
                lit[c] >= 0.0,
                "a surface facing {:?} returned {lit:?}",
                point.normal
            );
        }
    }
}
