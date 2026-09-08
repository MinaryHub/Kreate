//! What a drag frame costs the renderer, both ways.
//!
//! The claim the GPU deform path exists to make: a pointer position should cost
//! a uniform write and three dispatches, not a re-extraction of the mesh. The
//! CPU path is here beside it because a ratio is the only honest form of this
//! measurement — an absolute millisecond figure on one machine says nothing,
//! and "the GPU is fast" is not a claim anyone can check.
//!
//! Loose thresholds, in the manner of `extraction_cost` and `kreate-ops`'
//! `large_mesh`: they are here to catch a change of *shape* — someone
//! reintroducing an extraction into the drag frame — rather than to police
//! milliseconds. The numbers each test prints are the interesting part.
//!
//! Release only, and skipped when there is no adapter.

use std::time::{Duration, Instant};

use kreate_render::{Deform, Deformer, Gesture, GpuOptions, RenderMeshCpu};

/// Two hundred thousand vertices: the size the plan's drag budget is written
/// for.
const DIVISIONS: u32 = 450;

/// Serialised, so two benchmarks do not measure each other.
static BENCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serially() -> std::sync::MutexGuard<'static, ()> {
    BENCH.lock().unwrap_or_else(|e| e.into_inner())
}

fn measurable() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("skipping: mesh arithmetic is ~50x slower without optimisation");
        return false;
    }
    true
}

fn median_time(runs: usize, mut f: impl FnMut()) -> Duration {
    let mut samples: Vec<Duration> = (0..runs.max(1))
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

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

fn drag_mesh() -> kreate_mesh::MeshData {
    let mut mesh = kreate_mesh::primitives::grid(10.0, DIVISIONS, DIVISIONS);
    mesh.ensure_selection_layers();
    kreate_mesh::selection::select_all(&mut mesh);
    mesh
}

#[test]
fn a_previewed_drag_frame_does_not_extract_a_render_mesh() {
    if !measurable() {
        return;
    }
    let _guard = serially();
    let Some(g) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    let mesh = drag_mesh();
    let cpu = RenderMeshCpu::extract(&mesh);
    let verts = cpu.positions.len();
    let viewport =
        kreate_render::Viewport::new(&g.device, wgpu::TextureFormat::Rgba8UnormSrgb, 64, 64);
    let gpu_mesh =
        kreate_render::GpuMesh::upload(&g.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let deformer = Deformer::new(&g.device);

    let mut encoder = g
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let deform = Deform::begin(
        &g.device,
        &mut encoder,
        &deformer,
        &gpu_mesh,
        &cpu,
        &vec![1.0f32; verts],
    )
    .expect("a mesh with vertices");
    g.queue.submit([encoder.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // What a previewed frame does, from the CPU's side: write sixty-four bytes,
    // encode three dispatches, submit. Waited on so that the number includes
    // the GPU actually doing the work rather than only the cost of asking.
    let previewed = median_time(9, || {
        deform.set(&g.queue, &Gesture::moved(glam::Vec3::new(0.1, 0.0, 0.0)));
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        deform.dispatch(&mut encoder, &deformer);
        g.queue.submit([encoder.finish()]);
        g.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    });

    // What the path it replaces does: extract the whole render mesh to find out
    // what to write, then write both arrays. The extraction is the cost — the
    // triangulation, the corner normals and the vertex split, none of which a
    // move changes.
    let on_cpu = median_time(5, || {
        let cpu = RenderMeshCpu::extract(&mesh);
        gpu_mesh.update_positions(&g.queue, &cpu);
        g.queue.submit([]);
        g.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    });

    eprintln!(
        "a drag frame at {verts} render vertices: previewed {previewed:?}, \
         on the CPU {on_cpu:?}"
    );

    // The plan's budget for a drag frame, and the reason this path exists. Ten
    // times looser than the one millisecond it asks for, because this is a
    // shape test on a machine of unknown speed.
    assert!(
        previewed < Duration::from_millis(10),
        "a previewed drag frame took {previewed:?}; something in it is doing \
         work proportional to the mesh"
    );
    // And the comparison, which is the part that cannot be fooled by a fast
    // machine. Measured at 83 ms against about 1 ms when this was written; a
    // factor of four is the floor at which the rewrite would have been pointless.
    assert!(
        previewed * 4 < on_cpu,
        "previewing a frame ({previewed:?}) is not meaningfully cheaper than \
         extracting one ({on_cpu:?}); the deform path has stopped earning its \
         complexity"
    );
}

#[test]
fn beginning_a_drag_costs_about_one_extraction() {
    // What the preview is paid for with. The tables the normal passes walk come
    // from an extraction, and that extraction happens once per drag instead of
    // once per frame — so pressing `G` on a large mesh has a cost, and it is
    // the cost every frame used to have.
    //
    // Worth measuring rather than asserting from the code, because the GPU-side
    // work in `begin` — two buffer copies and five uploads — could easily have
    // dominated, and then the honest answer would have been to cache the tables
    // with the resident mesh instead.
    if !measurable() {
        return;
    }
    let _guard = serially();
    let Some(g) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    let mesh = drag_mesh();
    let viewport =
        kreate_render::Viewport::new(&g.device, wgpu::TextureFormat::Rgba8UnormSrgb, 64, 64);
    let deformer = Deformer::new(&g.device);

    let extraction = median_time(5, || {
        let _ = RenderMeshCpu::extract(&mesh);
    });

    let cpu = RenderMeshCpu::extract(&mesh);
    let gpu_mesh =
        kreate_render::GpuMesh::upload(&g.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let weights = vec![1.0f32; cpu.positions.len()];
    let beginning = median_time(5, || {
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let deform = Deform::begin(
            &g.device,
            &mut encoder,
            &deformer,
            &gpu_mesh,
            &cpu,
            &weights,
        )
        .expect("a mesh with vertices");
        g.queue.submit([encoder.finish()]);
        g.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        drop(deform);
    });

    eprintln!(
        "beginning a drag at {} render vertices: extraction {extraction:?}, \
         GPU setup {beginning:?}",
        cpu.positions.len()
    );
    assert!(
        beginning < extraction,
        "the GPU setup for a drag ({beginning:?}) costs more than the \
         extraction it needs ({extraction:?}); the tables want caching with the \
         resident mesh rather than building per drag"
    );
}
