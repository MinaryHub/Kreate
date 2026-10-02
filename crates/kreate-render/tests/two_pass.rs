//! Two-pass occlusion culling draws exactly the same image as no culling at all.
//!
//! Occlusion culling is the one cluster rejection that can be wrong in a way the
//! geometry cannot rule out: it depends on a lossy summary of a depth buffer
//! that, in the first pass, was built before the camera moved. Reject one
//! cluster too many and the model comes out with a hole — and a hole reads to a
//! user as corrupt geometry, not as a renderer bug. So the standard here is not
//! "looks about right" but pixel-for-pixel equality with an unculled render,
//! across a range of camera motions.
//!
//! # What the pixel tests turned out not to prove
//!
//! They are weaker than they look, and it took a falsification sweep to find out
//! how much. **Deleting the entire second pass leaves every pixel comparison in
//! this file passing** — on spheres, on a flat-faced occluder, and across a
//! sweep of pans and orbits. The second pass demonstrably does work: it recovers
//! eight clusters the first pass wrongly rejected in the pan case below. But
//! every one of those eight turned out to be genuinely hidden anyway. The
//! cluster cull is conservative, so what it recovers are clusters whose
//! *bounding spheres* poke past an occluder while their triangles do not.
//!
//! No scene could be found in which the second pass changes a single pixel. So
//! the case for it is analytic — a stale pyramid provably *can* reject a visible
//! cluster — rather than demonstrated, and it is kept as insurance against a
//! failure whose cost is severe and whose likelihood is unmeasured. That is a
//! weaker justification than this file originally claimed, and worth writing
//! down rather than leaving a reader to assume the tests establish more than
//! they do.
//!
//! What holds the design instead is
//! [`both_passes_do_real_work_after_the_camera_moves`], which reads the argument
//! buffers rather than the framebuffer. It catches all five falsifying edits,
//! including one real bug the pixel tests could never see: the two passes
//! indexing their shared flags differently, so the second redrew everything the
//! first had already drawn, at identical depth, for an identical picture and
//! double the work.
//!
//! # Why the motions are what they are
//!
//! A stale pyramid only misleads when the camera has moved, so every test seeds
//! the pyramid with one frame and compares a second. Which motion matters was
//! also a finding: **orbiting never exercises the second pass at all**. It
//! sweeps hidden geometry out into empty background, background sits at the far
//! plane, and the stale pyramid therefore never claims it is hidden. Panning
//! does the opposite — the occluder slides aside and what was behind it emerges
//! into the screen region the occluder used to fill, which is exactly where the
//! stale depth still reads near.
//!
//! Skips rather than fails when no adapter is available.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::cluster_cull::Pass;
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, ViewAxis, Viewport};

const SIZE: u32 = 256;
const BYTES_PER_ROW: u32 = SIZE * 4;

/// How the frame under test is configured.
#[derive(Clone, Copy)]
struct Mode {
    clusters: bool,
    occlusion: bool,
}

const DIRECT: Mode = Mode {
    clusters: false,
    occlusion: false,
};
const TWO_PASS: Mode = Mode {
    clusters: true,
    occlusion: true,
};

/// Renders a sequence of camera positions and returns the last frame.
///
/// A sequence, not a single frame, because that is the whole point: the pyramid
/// the first pass tests against comes from the *previous* frame, so a single
/// frame never exercises the case the second pass exists for.
fn render_sequence(
    mode: Mode,
    items: &[(&kreate_mesh::MeshData, DAffine3)],
    cameras: &[Camera],
) -> Option<Vec<[u8; 4]>> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("two pass target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
    viewport.cluster_path = mode.clusters;
    viewport.occlusion_culling = mode.occlusion;
    // Only the solid geometry differs between the paths. An overlay in the
    // comparison would be a pixel that matches for reasons unrelated to the
    // claim.
    viewport.show_grid = false;
    viewport.show_wireframe = false;
    viewport.show_elements = false;
    viewport.background = wgpu::Color::BLACK;
    // Culling against a reference, frame for frame: a still camera's frames
    // being averaged into one another is a different claim, and the
    // occlusion path does not accumulate.
    viewport.accumulate = false;

    let meshes: Vec<GpuMesh> = items
        .iter()
        .map(|(mesh, _)| {
            GpuMesh::upload(
                &gpu.device,
                &viewport.layouts,
                &RenderMeshCpu::extract(mesh).with_clusters(),
                Vec4::new(0.45, 0.45, 0.47, 1.0),
            )
        })
        .collect();

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (BYTES_PER_ROW * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    for (frame, camera) in cameras.iter().enumerate() {
        let mut draws: Vec<Draw> = meshes
            .iter()
            .zip(items)
            .map(|(mesh, (_, transform))| Draw::new(mesh, *transform))
            .collect();
        viewport.prepare(&gpu.device, &gpu.queue, camera, &mut draws);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        viewport.render_frame(&gpu.device, &gpu.queue, &mut encoder, &view, &draws);

        // Only the last frame is read back; the earlier ones exist to leave a
        // pyramid behind.
        if frame + 1 == cameras.len() {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(BYTES_PER_ROW),
                        rows_per_image: Some(SIZE),
                    },
                },
                wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    }

    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let data = readback.slice(..).get_mapped_range();
    let pixels = data.as_chunks::<4>().0.to_vec();
    drop(data);
    readback.unmap();
    Some(pixels)
}

fn lit(pixels: &[[u8; 4]]) -> usize {
    pixels
        .iter()
        .filter(|p| p[0] > 25 || p[1] > 25 || p[2] > 25)
        .count()
}

fn compare(a: &[[u8; 4]], b: &[[u8; 4]]) -> (usize, u8) {
    let mut differing = 0;
    let mut worst = 0u8;
    for (x, y) in a.iter().zip(b) {
        let delta = (0..3).map(|c| x[c].abs_diff(y[c])).max().unwrap_or(0);
        if delta > 0 {
            differing += 1;
            worst = worst.max(delta);
        }
    }
    (differing, worst)
}

fn camera_at(distance: f64, axis: ViewAxis, pivot: DVec3) -> Camera {
    let mut camera = Camera::default().looking_along(axis);
    camera.pivot = pivot;
    camera.distance = distance;
    camera
}

/// A scene where occlusion actually rejects clusters.
///
/// One large sphere near the camera and a row of small ones directly behind it.
/// The first version of this file used six equal spheres in a line, which looks
/// occluded but is not: a cluster is rejected only when its whole bounding
/// sphere is hidden, and equal-sized spheres leave every cluster poking out past
/// the silhouette in front of it. Occlusion rejected nothing, and the entire
/// falsification sweep passed with the second pass deleted.
fn occluded_scene() -> Vec<(kreate_mesh::MeshData, DAffine3)> {
    let big = kreate_mesh::primitives::uv_sphere(1.0, 48, 24);
    let small = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
    let mut scene = vec![(big, DAffine3::from_scale(DVec3::splat(2.6)))];
    for i in 0..5 {
        scene.push((
            small.clone(),
            DAffine3::from_translation(DVec3::new(
                (i as f64 - 2.0) * 0.5,
                (i as f64 - 2.0) * 0.4,
                -4.0 - i as f64 * 1.2,
            )) * DAffine3::from_scale(DVec3::splat(0.5)),
        ));
    }
    scene
}

/// How many cluster instances each pass issued, from the argument buffers.
///
/// The only way to tell a working two-pass scheme from one whose second pass
/// never fires: both produce the same picture, which is the point.
fn pass_counts(
    gpu: &kreate_render::Gpu,
    viewport: &Viewport,
    encoder: &mut wgpu::CommandEncoder,
) -> wgpu::Buffer {
    let cull = viewport.cluster_cull();
    let first = cull.slots(Pass::First);
    let second = cull.slots(Pass::Second);
    let end = second
        .last()
        .map(|s| s.args_offset as u64 + 16)
        .unwrap_or(16);
    let _ = first;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("args readback"),
        size: end,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(cull.args_buffer(), 0, &buffer, 0, end);
    buffer
}

fn check(cameras: &[Camera]) -> bool {
    let scene = occluded_scene();
    let items: Vec<(&kreate_mesh::MeshData, DAffine3)> =
        scene.iter().map(|(m, t)| (m, *t)).collect();

    let Some(direct) = render_sequence(DIRECT, &items, cameras) else {
        eprintln!("no adapter; skipping");
        return false;
    };
    let culled = render_sequence(TWO_PASS, &items, cameras).expect("adapter worked a moment ago");

    assert!(
        lit(&direct) > 2_000,
        "the reference frame drew almost nothing, so there is nothing to compare"
    );
    let (differing, worst) = compare(&direct, &culled);
    assert_eq!(
        differing,
        0,
        "{differing} of {} pixels differ, worst by {worst} — occlusion culling \
         removed something that was visible",
        direct.len()
    );
    true
}

#[test]
fn a_still_camera_matches_the_uncalled_reference() {
    // The easy case, and the one a stale pyramid handles correctly on its own:
    // nothing has moved, so the previous frame's depth is this frame's depth.
    let camera = camera_at(9.0, ViewAxis::PosZ, DVec3::ZERO);
    check(&[camera, camera]);
}

#[test]
fn every_pan_matches_the_unculled_reference() {
    // A sweep rather than one value. The amount of pan that makes the second
    // pass fire is narrow — found by search — so a single sample is a single
    // sample, and this is the test whose whole point is that no camera motion
    // produces a hole.
    let scene = occluded_scene();
    let items: Vec<(&kreate_mesh::MeshData, DAffine3)> =
        scene.iter().map(|(m, t)| (m, *t)).collect();
    // Five samples across the range where the second pass fires, not fourteen:
    // each one builds a device and renders four frames, and the full sweep cost
    // forty seconds for no extra coverage.
    for step in 0..5 {
        let amount = 0.15 + 0.1 * step as f64;
        let mut moved = camera_at(9.0, ViewAxis::PosZ, DVec3::ZERO);
        moved.pan(amount, amount * 0.3);
        let cameras = [camera_at(9.0, ViewAxis::PosZ, DVec3::ZERO), moved];

        let Some(direct) = render_sequence(DIRECT, &items, &cameras) else {
            eprintln!("no adapter; skipping");
            return;
        };
        let culled =
            render_sequence(TWO_PASS, &items, &cameras).expect("adapter worked a moment ago");
        let (differing, worst) = compare(&direct, &culled);
        assert_eq!(
            differing, 0,
            "pan {amount}: {differing} pixels differ by up to {worst}"
        );
    }
}

#[test]
fn a_panning_camera_matches_the_unculled_reference() {
    // The case that actually needs the second pass. See
    // `both_passes_do_real_work_after_the_camera_moves` for why panning is the
    // motion that exercises it and orbiting is not: with the second pass
    // deleted, this test fails and every other one in the file still passes.
    let mut moved = camera_at(9.0, ViewAxis::PosZ, DVec3::ZERO);
    moved.pan(0.35, 0.0);
    check(&[camera_at(9.0, ViewAxis::PosZ, DVec3::ZERO), moved]);
}

#[test]
fn an_orbiting_camera_matches_the_uncalled_reference() {
    // The case the second pass exists for. Between the two frames the camera
    // swings around the scene, so the pyramid the first pass tests against
    // describes a completely different view — and clusters it rejects as hidden
    // are, in the new view, in front.
    let mut cameras = Vec::new();
    for step in 0..4 {
        let mut camera = camera_at(9.0, ViewAxis::PosZ, DVec3::ZERO);
        camera.orbit(step as f64 * 0.9, step as f64 * 0.35);
        cameras.push(camera);
    }
    check(&cameras);
}

#[test]
fn a_camera_that_jumps_behind_the_scene_matches_the_reference() {
    // The worst case for a stale pyramid: the view reverses, so what was in
    // front is now behind and the previous frame's depth is not merely
    // out of date but actively misleading.
    check(&[
        camera_at(9.0, ViewAxis::PosZ, DVec3::ZERO),
        camera_at(9.0, ViewAxis::NegZ, DVec3::ZERO),
    ]);
}

#[test]
fn a_camera_that_dollies_in_matches_the_reference() {
    // Approaching changes which clusters are large enough to be tested against
    // coarse pyramid levels, and moves the near plane. Both shift the mip the
    // occlusion test picks.
    check(&[
        camera_at(14.0, ViewAxis::PosZ, DVec3::ZERO),
        camera_at(3.0, ViewAxis::PosZ, DVec3::ZERO),
    ]);
}

#[test]
fn both_passes_do_real_work_after_the_camera_moves() {
    // The test that makes every equality above mean something.
    //
    // Those compare an occlusion-culled frame against an unculled one and
    // require them identical — which is exactly what a cull that rejects
    // *nothing* also produces. A falsification sweep confirmed the danger: with
    // the entire second pass deleted, all five equality tests still passed,
    // because the scene at the time had no cluster-level occlusion at all.
    //
    // So this reads the argument buffers directly and insists that the first
    // pass rejects a real share, and that after the camera moves the second pass
    // recovers a non-zero number of clusters the first one wrongly threw away.
    // The second number is the two-pass scheme earning its existence.
    let scene = occluded_scene();
    let items: Vec<(&kreate_mesh::MeshData, DAffine3)> =
        scene.iter().map(|(m, t)| (m, *t)).collect();

    // A *pan*, not an orbit, and the difference is the whole finding. Orbiting
    // sweeps hidden geometry out into what was empty background, and background
    // is at the far plane, so the stale pyramid never claims it is hidden and
    // the first pass lets it through — the second pass has nothing to recover
    // and drew zero at every angle tried. Panning does the opposite: the
    // occluder slides one way on screen and what was behind it emerges into the
    // region the occluder used to fill, which is exactly where the stale pyramid
    // still reports near depth.
    //
    // 0.35 of a viewport height was found by sweeping. Smaller and nothing has
    // emerged yet; larger and the occluder has moved far enough that the stale
    // depth no longer covers the new position either.
    let mut moved = camera_at(9.0, ViewAxis::PosZ, DVec3::ZERO);
    moved.pan(0.35, 0.0);
    let cameras = [camera_at(9.0, ViewAxis::PosZ, DVec3::ZERO), moved];

    let Some((first, second, unculled)) = counted_run(&items, &cameras) else {
        eprintln!("no adapter; skipping");
        return;
    };
    println!("pass one drew {first}, pass two drew {second}, unculled {unculled}");

    assert!(
        first + second <= unculled,
        "culling drew more clusters than there are"
    );
    assert!(
        first + second < unculled,
        "occlusion rejected nothing: {} of {unculled} drawn, so every equality \
         test in this file is vacuous",
        first + second
    );
    assert!(
        second > 0,
        "the second pass drew nothing even after the camera swung around, so \
         the first pass is not rejecting anything the second has to recover"
    );
}

/// Renders the sequence and returns (pass one, pass two, no occlusion) counts.
fn counted_run(
    items: &[(&kreate_mesh::MeshData, DAffine3)],
    cameras: &[Camera],
) -> Option<(u32, u32, u32)> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("counted target"),
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

    let mut counts = Vec::new();
    for occlusion in [true, false] {
        let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
        viewport.cluster_path = true;
        viewport.occlusion_culling = occlusion;
        viewport.show_grid = false;
        viewport.show_wireframe = false;
        viewport.show_elements = false;

        let meshes: Vec<GpuMesh> = items
            .iter()
            .map(|(mesh, _)| {
                GpuMesh::upload(
                    &gpu.device,
                    &viewport.layouts,
                    &RenderMeshCpu::extract(mesh).with_clusters(),
                    Vec4::new(0.45, 0.45, 0.47, 1.0),
                )
            })
            .collect();

        let mut readback = None;
        for (frame, camera) in cameras.iter().enumerate() {
            let mut draws: Vec<Draw> = meshes
                .iter()
                .zip(items)
                .map(|(mesh, (_, transform))| Draw::new(mesh, *transform))
                .collect();
            viewport.prepare(&gpu.device, &gpu.queue, camera, &mut draws);
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            viewport.render_frame(&gpu.device, &gpu.queue, &mut encoder, &view, &draws);
            if frame + 1 == cameras.len() {
                readback = Some(pass_counts(&gpu, &viewport, &mut encoder));
            }
            gpu.queue.submit([encoder.finish()]);
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        }

        let buffer = readback?;
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let data = buffer.slice(..).get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        let stride =
            viewport.cluster_cull().slots(Pass::Second)[0].args_offset as usize / 4 / items.len();
        let mut first = 0u32;
        let mut second = 0u32;
        for i in 0..items.len() {
            first += words[i * stride + 1];
            second += words[(items.len() + i) * stride + 1];
        }
        drop(data);
        buffer.unmap();
        counts.push((first, second));
    }

    Some((counts[0].0, counts[0].1, counts[1].0))
}
