//! What a cold start actually costs, and whether the first frame hitches.
//!
//! The plan lists first-use pipeline compilation among the twelve small causes
//! that add up to a sluggish viewport, and it is the one that shows up as a
//! *stall in the middle of a drag* rather than as a uniformly slow frame — the
//! worst kind, because it is invisible in an average and unmistakable to the
//! user.
//!
//! Two separable costs, and this measures both because the fix for each is
//! different:
//!
//! - **Creating** the pipeline objects, which happens when the viewport is
//!   built. If that is where the time goes, it is paid at startup and the only
//!   question is how long the window takes to appear.
//! - **First use** of each pipeline, which is where a driver that defers real
//!   compilation until a draw will spend it. That cost lands on whichever frame
//!   first exercises the pipeline — and for the overlay and picking pipelines,
//!   that is the frame the user enters edit mode or clicks, not the first frame
//!   drawn.
//!
//! Reported rather than asserted. These are wall clocks on a shared machine, and
//! a threshold here would fail for reasons unrelated to the renderer.
//!
//! # A measurement trap found here
//!
//! Shader compilation is cached by the driver below the `Device`, so the *first*
//! viewport built in a process pays for every pipeline and every later one is
//! fast regardless of what it does. Any A/B comparison inside one process
//! therefore measures ordering, not the thing under test. The second test below
//! reports four numbers from both orderings to make that visible instead of
//! quietly reporting whichever ordering flattered the answer.

use glam::{DAffine3, Vec4};
use kreate_render::{Draw, GpuMesh, GpuOptions, RenderMeshCpu, Viewport};
use std::time::Instant;

const SIZE: u32 = 256;

#[test]
fn a_cold_start_is_worth_measuring() {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let Ok(gpu) = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)) else {
        eprintln!("no adapter; skipping");
        return;
    };

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    // --- creating every pipeline -------------------------------------------
    let start = Instant::now();
    let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
    let create = start.elapsed();

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("startup target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // Everything on, so the first frame touches every graphics pipeline the
    // viewport owns rather than just the solid one.
    viewport.show_grid = true;
    viewport.show_wireframe = true;
    viewport.show_elements = true;
    viewport.cluster_path = true;
    viewport.occlusion_culling = true;

    let mesh = kreate_mesh::primitives::cube(2.0);
    let uploaded = GpuMesh::upload(
        &gpu.device,
        &viewport.layouts,
        &RenderMeshCpu::extract(&mesh),
        Vec4::new(0.45, 0.45, 0.47, 1.0),
    );
    let camera = kreate_render::Camera::default();

    let frame = |viewport: &mut Viewport| {
        let mut draws = vec![Draw::new(&uploaded, DAffine3::IDENTITY)];
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        viewport.render_frame(&gpu.device, &gpu.queue, &mut encoder, &view, &draws);
        gpu.queue.submit([encoder.finish()]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
    };

    let start = Instant::now();
    frame(&mut viewport);
    let first = start.elapsed();

    let start = Instant::now();
    frame(&mut viewport);
    let second = start.elapsed();

    let start = Instant::now();
    for _ in 0..10 {
        frame(&mut viewport);
    }
    let steady = start.elapsed() / 10;

    println!("pipeline creation: {:.1} ms", create.as_secs_f64() * 1000.0);
    println!("first frame:       {:.1} ms", first.as_secs_f64() * 1000.0);
    println!("second frame:      {:.1} ms", second.as_secs_f64() * 1000.0);
    println!("steady state:      {:.1} ms", steady.as_secs_f64() * 1000.0);
    println!(
        "first-use penalty: {:.1} ms",
        (first.as_secs_f64() - steady.as_secs_f64()) * 1000.0
    );

    // The only thing worth failing on: that a frame happened at all.
    assert!(steady.as_secs_f64() > 0.0);
}

#[test]
fn warming_up_moves_the_first_use_cost_off_the_first_real_frame() {
    // The point of `warm_up`, measured. Without it the driver finishes deferred
    // compilation on whichever frame first touches each pipeline — and for the
    // overlays and the id pass that is not the first frame drawn but the moment
    // the user presses Tab or moves the mouse, which is the worst possible time.
    //
    // Reported rather than asserted, for the usual reason: this is a wall clock
    // on a machine that may be doing something else. What is asserted is only
    // that warming up does not somehow make the first frame slower.
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let Ok(gpu) = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    let measure = |warm: bool| {
        let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
        viewport.show_grid = true;
        viewport.show_wireframe = true;
        viewport.show_elements = true;
        viewport.cluster_path = true;
        viewport.occlusion_culling = true;

        if warm {
            viewport.warm_up(&gpu.device, &gpu.queue);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
        }

        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mesh = kreate_mesh::primitives::cube(2.0);
        let uploaded = GpuMesh::upload(
            &gpu.device,
            &viewport.layouts,
            &RenderMeshCpu::extract(&mesh),
            Vec4::new(0.45, 0.45, 0.47, 1.0),
        );
        let camera = kreate_render::Camera::default();

        let start = Instant::now();
        let mut draws = vec![Draw::new(&uploaded, DAffine3::IDENTITY)];
        viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        viewport.render_frame(&gpu.device, &gpu.queue, &mut encoder, &view, &draws);
        gpu.queue.submit([encoder.finish()]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        start.elapsed().as_secs_f64() * 1000.0
    };

    // Both orders, because the first measurement in a process pays a cost the
    // second cannot: the driver compiles each pipeline once per *device*, not
    // once per viewport, so whichever runs first absorbs that and flatters the
    // other. Reporting one order would overstate the effect by about fivefold.
    let warm_first = measure(true);
    let cold_second = measure(false);
    let cold_first = measure(false);
    let warm_second = measure(true);

    println!("warm-up first:  warmed {warm_first:.1} ms, then cold {cold_second:.1} ms");
    println!("cold first:     cold {cold_first:.1} ms, then warmed {warm_second:.1} ms");

    // No comparison is asserted, and that is a finding rather than caution.
    //
    // The effect cannot be isolated by repeated measurement in one process. The
    // driver caches compiled shaders below the device, so whichever viewport is
    // built first pays for every pipeline and every later one — warmed or not —
    // is fast. Measured in a single run: 1.8, 2.9, 1.7, 1.6 ms, with no stable
    // ordering. An earlier version compared the first two and asserted a
    // threefold gap, which held only because the first measurement in the
    // process absorbed a cost the second could not repeat.
    //
    // What is genuinely known: in a cold process the first frame that touches
    // every pipeline costs about 16 ms against a steady state under 2 ms, and
    // `warm_up` is what decides *when* that is paid. It moves it before the
    // window is interactive rather than onto the frame where the user first
    // presses Tab. Establishing that properly needs a fresh process per sample,
    // which belongs in the startup-latency harness and not here.
    //
    // So this asserts only that warming up does not make things worse, and
    // prints the rest for a human.
    assert!(
        warm_first < cold_first * 4.0 + 5.0,
        "warming up made the first frame dramatically slower: {warm_first:.1}          against {cold_first:.1}"
    );
}

#[test]
fn the_pipeline_cache_feature_is_reported_honestly() {
    // The plan says to enable `PIPELINE_CACHE` on "Vulkan/DX12" and persist it
    // to disk. In wgpu 29 that is only half true: the DX12 backend implements
    // `create_pipeline_cache` as a stub that returns an empty marker and stores
    // nothing, and Metal does not implement it at all. Only Vulkan has a real
    // `VkPipelineCache` behind it.
    //
    // This test does not assert which way it comes out — the answer depends on
    // the machine and the chosen backend — it prints it, so that a decision
    // about whether the disk cache is worth building is made against the
    // backend actually in use rather than against the plan's assumption.
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let Ok(gpu) = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)) else {
        eprintln!("no adapter; skipping");
        return;
    };
    println!(
        "{}: pipeline cache {}",
        gpu.info.describe(),
        if gpu.caps.pipeline_cache {
            "available"
        } else {
            "not available"
        }
    );
}
