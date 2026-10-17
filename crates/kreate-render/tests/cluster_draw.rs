//! The cluster path draws the same image as the direct path.
//!
//! This is the claim that makes cluster culling usable rather than interesting.
//! The cull is verified against its CPU specification elsewhere; what is left is
//! everything between "these clusters survived" and "these pixels are lit": the
//! visible list, the index offsets, the degenerate-triangle trick for short
//! clusters, the storage fetch of positions and normals, the instance uniform.
//!
//! Any of those can be wrong in a way that still produces a plausible picture —
//! a mesh missing a few triangles, or one cluster drawing another's geometry,
//! looks like a modelling mistake and not like a renderer bug. So the test is
//! not "does it look right" but **is it the same image**, against the path that
//! was already working.
//!
//! Exact equality is reachable because the two paths share `fs_main` literally,
//! not by resemblance: one shader module, one fragment entry point, the same
//! camera-relative transform, the same octahedral normal decode. The only thing
//! that differs is where the vertex stage got its numbers.
//!
//! # Not covered here
//!
//! A mesh with no geometry at all. `GpuMesh::upload` has never accepted one —
//! its edge overlay bind group rejects a zero-sized positions buffer — and that
//! predates clusters, so a test for it here would be testing someone else's
//! precondition. The empty case in the *bake* is unit tested in `cluster.rs`.
//!
//! Skips rather than fails when no adapter is available.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, ViewAxis, Viewport};

const SIZE: u32 = 256;
const BYTES_PER_ROW: u32 = SIZE * 4;

/// Renders one frame, through either path.
fn render(
    clusters: bool,
    items: &[(&kreate_mesh::MeshData, DAffine3)],
    distance: f64,
) -> Option<Vec<[u8; 4]>> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cluster target"),
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
    viewport.cluster_path = clusters;
    // Only the solid pass differs between the two paths, so everything else is
    // off: a grid line or a vertex dot in the comparison would be a pixel that
    // matches for reasons unrelated to the claim.
    viewport.show_grid = false;
    viewport.show_ground_shadow = false;
    viewport.show_wireframe = false;
    viewport.show_elements = false;
    // Black, so "lit" means "something was drawn". The default background is a
    // dark grey that reaches the sRGB target near 66, which is above any
    // sensible threshold — the first version of this file left it in place and
    // every pixel counted as lit, which made the "there was something to
    // compare" guard pass on an empty frame.
    viewport.background = wgpu::Color::BLACK;

    // One upload per entry, so a frame can mix *different* meshes. That matters
    // for more than coverage breadth: with every draw sharing one mesh, a bug
    // that binds the wrong mesh's geometry to a draw is invisible, because the
    // wrong mesh is the right one.
    let uploaded: Vec<GpuMesh> = items
        .iter()
        .map(|(mesh, _)| {
            let cpu = RenderMeshCpu::extract(mesh).with_clusters();
            GpuMesh::upload(
                &gpu.device,
                &viewport.layouts,
                &cpu,
                Vec4::new(0.45, 0.45, 0.47, 1.0),
            )
        })
        .collect();

    let mut draws: Vec<Draw> = uploaded
        .iter()
        .zip(items)
        .map(|(mesh, (_, transform))| Draw::new(mesh, *transform))
        .collect();
    let mut camera = Camera::default().looking_along(ViewAxis::PosZ);
    camera.pivot = DVec3::ZERO;
    camera.distance = distance;

    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (BYTES_PER_ROW * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    // The compute pass has to be encoded before the render pass begins: a render
    // pass cannot be interrupted to dispatch. wgpu puts the barrier in.
    viewport.cull(&mut encoder, &draws);
    viewport.render(&mut encoder, &view, &draws);
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
    gpu.queue.submit([encoder.finish()]);

    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let data = readback.slice(..).get_mapped_range();
    let pixels = data.as_chunks::<4>().0.to_vec();
    drop(data);
    readback.unmap();
    Some(pixels)
}

/// Pixels that are not the background.
fn lit(pixels: &[[u8; 4]]) -> usize {
    pixels
        .iter()
        .filter(|p| p[0] > 25 || p[1] > 25 || p[2] > 25)
        .count()
}

/// How many pixels differ, and by how much at worst.
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

fn check_identical(items: &[(&kreate_mesh::MeshData, DAffine3)], distance: f64) -> bool {
    let Some(direct) = render(false, items, distance) else {
        eprintln!("no adapter; skipping");
        return false;
    };
    let clustered = render(true, items, distance).expect("adapter worked a moment ago");

    assert!(
        lit(&direct) > 500,
        "the direct path drew almost nothing, so there is nothing to compare"
    );
    let (differing, worst) = compare(&direct, &clustered);
    assert_eq!(
        differing,
        0,
        "{differing} of {} pixels differ, worst by {worst}",
        direct.len()
    );
    true
}

#[test]
fn a_sphere_draws_identically_through_both_paths() {
    // A dense sphere: many clusters, most of them full, and a closed surface so
    // a missing triangle shows as a hole rather than as nothing.
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 64, 32);
    check_identical(&[(&mesh, DAffine3::IDENTITY)], 3.0);
}

#[test]
fn a_cube_draws_identically_through_both_paths() {
    // The short-cluster case. A cube is twelve triangles, so it produces one
    // cluster far below the 128-triangle cap and every draw is mostly
    // degenerate vertices. If the degenerate branch put them anywhere but a
    // single point, the cube would be buried in stray triangles.
    let mesh = kreate_mesh::primitives::cube(2.0);
    check_identical(&[(&mesh, DAffine3::IDENTITY)], 5.0);
}

#[test]
fn a_torus_draws_identically_through_both_paths() {
    // Self-occluding and concave, so it exercises the depth interaction between
    // clusters that a convex shape cannot: a wrong index offset here shows up as
    // geometry from the far side punching through the near side.
    let mesh = kreate_mesh::primitives::torus(1.0, 0.4, 64, 32);
    check_identical(&[(&mesh, DAffine3::IDENTITY)], 4.0);
}

#[test]
fn several_transformed_instances_draw_identically() {
    // The per-draw slots, on screen. Three instances of one mesh share a cluster
    // buffer and must each pick up their own transform, their own argument slot
    // and their own slice of the visible list. Getting any of those wrong draws
    // one instance's clusters at another's position, which a single-instance
    // test cannot see.
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
    check_identical(
        &[
            (
                &mesh,
                DAffine3::from_translation(DVec3::new(-1.6, 0.0, 0.0)),
            ),
            (
                &mesh,
                DAffine3::from_translation(DVec3::new(1.6, 0.3, 0.0))
                    * DAffine3::from_scale(DVec3::new(1.0, 1.6, 1.0)),
            ),
            (
                &mesh,
                DAffine3::from_translation(DVec3::new(0.0, -1.4, 0.4))
                    * DAffine3::from_axis_angle(DVec3::X, 0.6)
                    * DAffine3::from_scale(DVec3::splat(0.7)),
            ),
        ],
        7.0,
    );
}

#[test]
fn the_cluster_path_still_honours_screen_size_culling() {
    // The two culls compose: the object-level decision runs first and removes
    // the draw entirely, so the cluster cull is never even asked. Worth pinning
    // because `prepare` filters the draw list before handing it to the cull, and
    // a filter applied in one place and not the other would misalign every
    // argument offset.
    let mesh = kreate_mesh::primitives::cube(2.0);
    let Some(pixels) = render(true, &[(&mesh, DAffine3::IDENTITY)], 4_000.0) else {
        eprintln!("no adapter; skipping");
        return;
    };
    assert_eq!(lit(&pixels), 0, "a speck should not have been drawn");
}

#[test]
fn a_culled_draw_does_not_shift_the_others() {
    // The argument offsets are assigned by position among the *surviving* draws,
    // because that is the order `prepare` handed the cull. Taking them from the
    // full draw list instead would pair every slot with the wrong mesh.
    //
    // Two *different* meshes, and that is the whole point: an earlier version of
    // this test used two instances of one sphere, and the mispairing was
    // invisible because binding the wrong mesh bound the same mesh. A falsifying
    // edit passed it cleanly.
    let cube = kreate_mesh::primitives::cube(2.0);
    let sphere = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
    let items = [
        // A speck: culled by screen size before the cluster cull sees it.
        (
            &cube,
            DAffine3::from_translation(DVec3::new(0.0, 0.0, -2_000.0))
                * DAffine3::from_scale(DVec3::splat(0.01)),
        ),
        (&sphere, DAffine3::IDENTITY),
    ];
    let Some(clustered) = render(true, &items, 4.0) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let direct = render(false, &items, 4.0).expect("adapter worked a moment ago");
    assert!(
        lit(&clustered) > 500,
        "the surviving instance vanished: {} pixels lit",
        lit(&clustered)
    );
    let (differing, worst) = compare(&direct, &clustered);
    assert_eq!(differing, 0, "{differing} pixels differ, worst by {worst}");
}
