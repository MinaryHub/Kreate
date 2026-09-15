//! What a light costs, measured before deciding to cull them.
//!
//! §2.2 asks for a tiled forward path with a budget of 0.3 ms to cull 256
//! lights. That budget presumes the alternative is worse, and nobody has
//! checked: the shading pass walks every light for every fragment, and how
//! badly that scales is a fact about this renderer rather than about renderers
//! in general. A pass built on the presumption would be a pass built on a
//! guess.
//!
//! So this renders the same scene with a growing number of lights and reports
//! what each one adds. Deliberately **not** a threshold assertion, for the same
//! reasons `lod_speedup` gives: this is a wall clock around a submit-and-wait
//! on a shared machine, and a build that fails because another compile was
//! running teaches nothing. It asserts only the shape — that more lights cost
//! more, which would otherwise mean the lights are not reaching the shader at
//! all — and prints the numbers for a human to read with `--nocapture`.

use kreate_render::lights::{Falloff, PlacedLight};
use kreate_render::viewport::{Draw, ShadingMode, Viewport};
use kreate_render::{Camera, GpuOptions, RenderMeshCpu};
use std::time::Instant;

/// Big enough that the shading pass, rather than anything per-draw, is what is
/// being timed.
const SIZE: u32 = 1024;
const FRAMES: u32 = 20;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
    Some(Gpu {
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
    })
}

/// `count` lamps scattered over the scene, none of them casting.
///
/// No shadows on purpose. A shadow is a pass over the geometry and would be
/// what the clock measured; what is in question here is the cost of the
/// *shading* loop, which is the thing a tiled path replaces.
fn lamps(count: usize) -> Vec<PlacedLight> {
    (0..count)
        .map(|i| {
            let a = i as f32 * 2.399_963; // the golden angle, so they spread
            let r = 6.0 + (i % 7) as f32;
            PlacedLight {
                falloff: Falloff::InverseSquare,
                position: glam::DVec3::new(
                    (a.cos() * r) as f64,
                    (a.sin() * r) as f64,
                    3.0 + (i % 5) as f64,
                ),
                amount: [20_000.0 / count.max(1) as f32; 3],
                cast_shadow: false,
                ..PlacedLight::default()
            }
        })
        .collect()
}

/// Milliseconds a frame, averaged, for a scene lit by `count` lamps.
fn measure(gpu: &Gpu, count: usize) -> f64 {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    let floor = RenderMeshCpu::extract(&kreate_mesh::primitives::plane(24.0));
    let floor = kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &floor, glam::Vec4::ONE);
    let camera = Camera {
        distance: 26.0,
        ..Camera::default()
    };
    let mut draws = vec![Draw::new(&floor, glam::DAffine3::IDENTITY)];

    viewport.set_world([0.0; 3]);
    viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &lamps(count));

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("light cost"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // One frame first, so that pipeline creation and buffer growth are not in
    // the average.
    for _ in 0..2 {
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        viewport.render(&mut encoder, &view, &draws);
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    let start = Instant::now();
    for _ in 0..FRAMES {
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        viewport.render(&mut encoder, &view, &draws);
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }
    start.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64
}

#[test]
fn what_a_light_costs_in_the_shading_loop() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    let counts = [1usize, 16, 64, 256];
    let mut times = Vec::new();
    for count in counts {
        let ms = measure(&gpu, count);
        times.push(ms);
        eprintln!("{count:>4} lights: {ms:.3} ms a frame at {SIZE}x{SIZE}");
    }
    let per_light = (times[times.len() - 1] - times[0]) / (counts[counts.len() - 1] - 1) as f64;
    eprintln!("about {:.4} ms a light at this resolution", per_light);

    // The only assertion, and it is about the shape rather than the speed: more
    // lights have to cost more. If they did not, the loop would not be running
    // and every measurement above would be of something else.
    assert!(
        times[times.len() - 1] > times[0],
        "256 lights cost {:.3} ms and one cost {:.3}; the shading loop is not \
         running",
        times[times.len() - 1],
        times[0]
    );
}
