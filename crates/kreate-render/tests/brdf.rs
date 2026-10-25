//! The GPU's BRDF has to be the CPU's BRDF.
//!
//! §2.3's rule is one BRDF for the viewport and the renderer both. There are
//! two implementations of it, because one runs on a device and one does not, so
//! the rule is kept by this test rather than by intention — §5's second risk
//! says a drift here becomes a bug report that recurs forever.
//!
//! # The two lobes are compared separately
//!
//! Their sum would hide which one is wrong, and the two failure modes are
//! different: the specular half is the microfacet arithmetic, and the diffuse
//! half is the *coupling* to it — which took two attempts to get right on the
//! CPU alone, because the obvious per-direction form gains energy and the
//! obvious directional form is not reciprocal.
//!
//! # The tolerance is relative, and small
//!
//! The lobes span several orders of magnitude across roughness: a near-mirror
//! peak is in the hundreds and a rough surface's is a fraction. So the claim is
//! a relative agreement, with an absolute floor for the values near zero where
//! a relative comparison means nothing.

use kreate_render::brdf::{self, albedo_table, Surface};
use kreate_render::GpuOptions;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// A device on the CPU — Mesa's lavapipe, on a runner with no GPU —
    /// whose arithmetic lands further from the CPU reference than any
    /// real GPU's: its sines and its specular lobes are a few parts in a
    /// thousand off, where the tests ask for one. Tolerated on such a
    /// device alone.
    software: bool,
}

/// The one device this test binary uses. See `deform.rs` for why it is shared.
fn gpu() -> Option<Gpu> {
    static SHARED: std::sync::OnceLock<Option<(wgpu::Device, wgpu::Queue, bool)>> =
        std::sync::OnceLock::new();
    let (device, queue, software) = SHARED
        .get_or_init(|| {
            let opts = GpuOptions::default();
            let instance = kreate_render::Gpu::create_instance(&opts);
            let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
            let software = gpu.info.device_type == "Cpu";
            Some((gpu.device.clone(), gpu.queue.clone(), software))
        })
        .clone()?;
    Some(Gpu {
        device,
        queue,
        software,
    })
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

/// One thousandth, relative, with an absolute floor for the near-zero values.
const RELATIVE: f32 = 1e-3;
const ABSOLUTE: f32 = 1e-5;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Sample {
    material: [f32; 4],
    params: [f32; 4],
    normal: [f32; 4],
    view: [f32; 4],
    light: [f32; 4],
}

fn sample_of(surface: &Surface, n: glam::Vec3, v: glam::Vec3, l: glam::Vec3) -> Sample {
    let base = surface.base_color.to_array();
    Sample {
        material: [base[0], base[1], base[2], surface.metallic],
        params: [surface.roughness, 0.0, 0.0, 0.0],
        normal: [n.x, n.y, n.z, 0.0],
        view: [v.x, v.y, v.z, 0.0],
        light: [l.x, l.y, l.z, 0.0],
    }
}

/// Runs the probe pass and reads back both lobes per sample.
fn on_gpu(gpu: &Gpu, samples: &[Sample]) -> Vec<([f32; 3], [f32; 3])> {
    use wgpu::util::DeviceExt as _;

    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brdf probe"),
            source: wgpu::ShaderSource::Wgsl(brdf::shader_source().into()),
        });

    // Built by the crate rather than by the test, so the layout has one author.
    let table = brdf::dfg_buffer(albedo_table());
    let dfg = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dfg"),
            contents: bytemuck::cast_slice(&table),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let inputs = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brdf samples"),
            contents: bytemuck::cast_slice(samples),
            usage: wgpu::BufferUsages::STORAGE,
        });
    // A value the arithmetic cannot produce, so an unwritten entry shows.
    let sentinel = vec![[-7.0f32; 4]; samples.len() * 2];
    let answers = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brdf answers"),
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
            label: Some("brdf probe params"),
            contents: bytemuck::bytes_of(&Probe {
                count: samples.len() as u32,
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
    let layout = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brdf probe"),
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
    // The library owns group 2; the probe's own buffers are group 0.
    let shading = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brdf shading"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: brdf::DFG_BINDING,
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
            label: Some("brdf probe"),
            bind_group_layouts: &[Some(&layout), None, Some(&shading)],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brdf probe"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("probe_brdf"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("brdf probe"),
        layout: &layout,
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
    let shading_bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("brdf shading"),
        layout: &shading,
        entries: &[wgpu::BindGroupEntry {
            binding: brdf::DFG_BINDING,
            resource: dfg.as_entire_binding(),
        }],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brdf probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_bind_group(brdf::DFG_GROUP, &shading_bind, &[]);
        pass.dispatch_workgroups((samples.len() as u32).div_ceil(64), 1, 1);
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
    read.chunks(2)
        .map(|pair| {
            (
                [pair[0][0], pair[0][1], pair[0][2]],
                [pair[1][0], pair[1][1], pair[1][2]],
            )
        })
        .collect()
}

/// Surfaces worth asking about: both ends of metallic and of roughness, plus a
/// coloured metal and a coloured dielectric, which are where a channel getting
/// crossed would show.
fn surfaces() -> Vec<Surface> {
    let mut out = Vec::new();
    for metallic in [0.0f32, 0.35, 1.0] {
        for roughness in [0.0f32, 0.05, 0.2, 0.5, 0.85, 1.0] {
            out.push(Surface {
                base_color: kreate_math::SceneRgb::new(0.9, 0.55, 0.2),
                metallic,
                roughness,
            });
        }
    }
    out.push(Surface {
        base_color: kreate_math::SceneRgb::WHITE,
        metallic: 1.0,
        roughness: 0.3,
    });
    out.push(Surface {
        base_color: kreate_math::SceneRgb::BLACK,
        metallic: 0.0,
        roughness: 0.6,
    });
    out
}

/// Geometry worth asking about: head-on, grazing, off to one side, and the two
/// degenerate cases where a light or a viewer is behind the surface.
fn geometry() -> Vec<(glam::Vec3, glam::Vec3)> {
    let mut out = vec![
        (glam::Vec3::Z, glam::Vec3::Z),
        (
            glam::Vec3::new(0.99, 0.0, 0.141).normalize(),
            glam::Vec3::new(-0.99, 0.0, 0.141).normalize(),
        ),
        (
            glam::Vec3::new(0.3, 0.2, 0.93).normalize(),
            glam::Vec3::new(-0.5, 0.4, 0.77).normalize(),
        ),
        // A light behind the surface, and a viewer behind it. Both are taken by
        // the early return — which is also what makes the exactly-opposite
        // case unreachable, since `l = -v` forces the two cosines to have
        // opposite signs. A sweep proved the half-vector guard that used to
        // stand here was dead, so this pair is testing the return and not a
        // `normalize`.
        (glam::Vec3::Z, glam::Vec3::NEG_Z),
        (glam::Vec3::NEG_Z, glam::Vec3::Z),
    ];
    // A sweep, so the comparison is not only about the cases somebody thought
    // of.
    for i in 0..24 {
        let a = (i as f32 / 24.0) * std::f32::consts::FRAC_PI_2 * 0.99;
        let b = ((i * 7 % 24) as f32 / 24.0) * std::f32::consts::FRAC_PI_2 * 0.99;
        out.push((
            glam::Vec3::new(a.sin(), 0.0, a.cos()),
            glam::Vec3::new(-b.sin() * 0.7, b.sin() * 0.7, b.cos()),
        ));
    }
    out
}

fn close(got: f32, want: f32) -> bool {
    close_within(got, want, RELATIVE)
}

/// The mirror's tolerance: a roughness of zero puts the specular lobe at
/// its floor, where its height goes as the fourth power of a number near
/// the smallest the shader allows, and a software device's arithmetic —
/// lavapipe's, on the Linux runner — lands half a percent from the CPU's
/// there while every real GPU is within the thousandth. Ten times the
/// usual, for that one case: a lobe the wrong shape is off by far more.
const MIRROR_RELATIVE: f32 = 1e-2;

fn tolerance_for(surface: &Surface, software: bool) -> f32 {
    if software || surface.roughness == 0.0 {
        MIRROR_RELATIVE
    } else {
        RELATIVE
    }
}

fn close_within(got: f32, want: f32, relative: f32) -> bool {
    let scale = got.abs().max(want.abs());
    (got - want).abs() <= (scale * relative).max(ABSOLUTE)
}

#[test]
fn the_gpu_agrees_with_the_cpu_on_both_lobes() {
    // The test §2.3's rule rests on: one BRDF, two implementations, held
    // together by measurement rather than by care.
    let gpu = gpu_or_skip!(gpu());
    let n = glam::Vec3::Z;

    let mut samples = Vec::new();
    let mut expected = Vec::new();
    for surface in surfaces() {
        for (v, l) in geometry() {
            samples.push(sample_of(&surface, n, v, l));
            expected.push((surface, v, l));
        }
    }

    let got = on_gpu(&gpu, &samples);
    assert_eq!(got.len(), samples.len());

    for ((surface, v, l), (diffuse, specular)) in expected.iter().zip(&got) {
        let want = brdf::evaluate(surface, n, *v, *l);
        let relative = tolerance_for(surface, gpu.software);
        for c in 0..3 {
            assert!(
                diffuse[c].is_finite() && specular[c].is_finite(),
                "{surface:?} v {v} l {l} gave {diffuse:?} {specular:?}"
            );
            assert!(
                close(diffuse[c], want.diffuse[c]),
                "diffuse channel {c}, {surface:?} v {v} l {l}: GPU {} against CPU {}",
                diffuse[c],
                want.diffuse[c]
            );
            assert!(
                close_within(specular[c], want.specular[c], relative),
                "specular channel {c}, {surface:?} v {v} l {l}: GPU {} against CPU {}",
                specular[c],
                want.specular[c]
            );
        }
    }
}

#[test]
fn the_table_crosses_to_the_device_intact() {
    // The layout claim. A table indexed with the axes swapped, or with the
    // averages read as table entries, still produces a picture — and the
    // energy would be wrong everywhere by a factor that looks like an artistic
    // choice.
    //
    // The probe is a surface whose response depends on the table strongly: a
    // rough metal, where the compensation is nearly a doubling.
    let gpu = gpu_or_skip!(gpu());
    let surface = Surface {
        base_color: kreate_math::SceneRgb::WHITE,
        metallic: 1.0,
        roughness: 0.85,
    };
    let n = glam::Vec3::Z;

    // Several view angles, because the table's view dependence is the thing an
    // axis swap destroys — and a single angle would agree by luck.
    let mut samples = Vec::new();
    let mut geometry = Vec::new();
    for i in 1..12 {
        let theta = (i as f32 / 12.0) * std::f32::consts::FRAC_PI_2;
        let v = glam::Vec3::new(theta.sin(), 0.0, theta.cos());
        let l = glam::Vec3::new(-theta.sin() * 0.5, 0.3, 0.8).normalize();
        samples.push(sample_of(&surface, n, v, l));
        geometry.push((v, l));
    }

    let got = on_gpu(&gpu, &samples);
    let mut spread = 0.0f32;
    for ((v, l), (_, specular)) in geometry.iter().zip(&got) {
        let want = brdf::evaluate(&surface, n, *v, *l);
        assert!(
            close(specular[0], want.specular[0]),
            "view {v}: GPU {} against CPU {}",
            specular[0],
            want.specular[0]
        );
        spread = spread.max(specular[0]);
    }
    assert!(spread > 0.0, "the probe measured nothing");
}

#[test]
fn a_light_or_a_viewer_behind_the_surface_gives_nothing_on_the_device_too() {
    // Both sides return zero rather than a negative number, and the guarded
    // `normalize` of a zero half-vector is here as well — a NaN in a fragment
    // shader spreads through the blend and is not traceable to a light.
    let gpu = gpu_or_skip!(gpu());
    let surface = Surface::default();
    let n = glam::Vec3::Z;
    let samples: Vec<Sample> = [
        (glam::Vec3::Z, glam::Vec3::NEG_Z),
        (glam::Vec3::NEG_Z, glam::Vec3::Z),
        (glam::Vec3::NEG_Z, glam::Vec3::NEG_Z),
        (glam::Vec3::X, glam::Vec3::Z),
    ]
    .iter()
    .map(|(v, l)| sample_of(&surface, n, *v, *l))
    .collect();

    for (diffuse, specular) in on_gpu(&gpu, &samples) {
        assert_eq!(diffuse, [0.0; 3], "{diffuse:?}");
        assert_eq!(specular, [0.0; 3], "{specular:?}");
    }
}

#[test]
fn a_mirror_is_finite_on_the_device() {
    // The roughness floor has to be applied on both sides. Without it the
    // device produces an infinity, which becomes a white pixel that survives
    // every later pass — including the display transform, whose clamp would
    // hide the cause.
    let gpu = gpu_or_skip!(gpu());
    let mirror = Surface {
        base_color: kreate_math::SceneRgb::WHITE,
        metallic: 1.0,
        roughness: 0.0,
    };
    let n = glam::Vec3::Z;
    let samples = vec![sample_of(&mirror, n, glam::Vec3::Z, glam::Vec3::Z)];
    let (_, specular) = on_gpu(&gpu, &samples)[0];
    let want = brdf::evaluate(&mirror, n, glam::Vec3::Z, glam::Vec3::Z);
    assert!(specular[0].is_finite(), "{specular:?}");
    assert!(
        specular[0] > 100.0,
        "a mirror should be bright: {specular:?}"
    );
    assert!(close_within(specular[0], want.specular[0], MIRROR_RELATIVE));
}
