//! What cluster culling is worth, measured.
//!
//! The cull is correct and the draw is pixel-identical to the direct path. That
//! says nothing about whether it is *faster*, and a GPU-driven path that loses
//! to a plain indexed draw is worse than no path at all — it is complexity with
//! a negative return. So this measures both, on the same geometry.
//!
//! The result decides a default, which is why it exists rather than being an
//! afterthought: `Viewport::cluster_path` is off unless the numbers say it
//! should be on, and the numbers depend on the scene.
//!
//! Deliberately not a threshold assertion. This is a wall clock around a
//! submit-and-wait on a machine that is also compiling, so a build that failed
//! on a ratio would fail for reasons unrelated to the renderer. Run with
//! `--nocapture` to read the numbers.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, ViewAxis, Viewport};
use std::time::Instant;

const SIZE: u32 = 1024;
const FRAMES: u32 = 30;

struct Result {
    ms: f64,
    triangles: u32,
    clusters: u32,
}

fn measure(clusters: bool, instances: usize, distance: f64) -> Option<Result> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cluster timing"),
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

    let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
    viewport.cluster_path = clusters;
    // Only the solid pass is under test. The overlays would dominate the frame
    // and swamp the difference.
    viewport.show_grid = false;
    viewport.show_wireframe = false;
    viewport.show_elements = false;

    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 400, 200);
    let cpu = RenderMeshCpu::extract(&mesh);
    let triangles = cpu.counts.triangles;
    let uploaded = GpuMesh::upload(
        &gpu.device,
        &viewport.layouts,
        &cpu,
        Vec4::new(0.45, 0.45, 0.47, 1.0),
    );
    let cluster_count = uploaded.cluster_count;

    // Linked duplicates: one geometry upload, many transforms. Exactly the case
    // the plan says the cluster path is for, and the case where a per-object
    // draw call multiplies.
    let side = (instances as f64).sqrt().ceil() as i32;
    let mut draws: Vec<Draw> = Vec::new();
    for i in 0..instances as i32 {
        let x = (i % side) as f64 * 2.5 - side as f64 * 1.25;
        let y = (i / side) as f64 * 2.5 - side as f64 * 1.25;
        draws.push(Draw::new(
            &uploaded,
            DAffine3::from_translation(DVec3::new(x, y, 0.0)),
        ));
    }

    let mut camera = Camera::default().looking_along(ViewAxis::PosZ);
    camera.pivot = DVec3::ZERO;
    camera.distance = distance;
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let encode = |gpu: &kreate_render::Gpu| {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        viewport.cull(&mut encoder, &draws);
        viewport.render(&mut encoder, &view, &draws);
        gpu.queue.submit([encoder.finish()]);
    };

    // One frame outside the measurement, so pipeline compilation and first-use
    // allocation are not counted as render cost.
    encode(&gpu);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;

    let start = Instant::now();
    for _ in 0..FRAMES {
        encode(&gpu);
        // Waited on every frame: without this the loop measures how fast
        // commands can be queued, which is unrelated to how fast they run.
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    }
    Some(Result {
        ms: start.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64,
        triangles: triangles * instances as u32,
        clusters: cluster_count * instances as u32,
    })
}

#[test]
fn the_cluster_path_is_worth_measuring() {
    // Three scenes, because the answer is not one number. Culling costs a
    // dispatch and a level of indirection per vertex, and buys the rejection of
    // whole clusters — so it should lose on a small scene, and win where there
    // is something to reject.
    let scenes = [
        // One sphere filling the frame: nothing off screen, so only backface
        // rejection can pay, against the cost of the indirection.
        ("1 sphere, filling the frame", 1usize, 3.0),
        // A grid of linked duplicates, most of them outside the frustum.
        ("64 duplicates, mostly off screen", 64, 6.0),
        // The same grid framed whole: everything visible, so frustum rejection
        // pays nothing and only the cone does.
        ("64 duplicates, all on screen", 64, 60.0),
    ];

    for (name, instances, distance) in scenes {
        let Some(direct) = measure(false, instances, distance) else {
            eprintln!("no adapter; skipping");
            return;
        };
        let clustered = measure(true, instances, distance).expect("adapter worked a moment ago");
        println!(
            "{name}: {} triangles in {} clusters",
            direct.triangles, clustered.clusters
        );
        println!(
            "  direct {:.2} ms   clustered {:.2} ms   ratio {:.2}x",
            direct.ms,
            clustered.ms,
            direct.ms / clustered.ms
        );
    }
}
