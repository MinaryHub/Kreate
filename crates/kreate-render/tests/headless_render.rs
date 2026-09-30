//! Headless renderer smoke test.
//!
//! Renders to an offscreen texture with no window and asserts on the pixels that
//! come back. This exists for three reasons:
//!
//! - "the application starts without crashing" and "the application draws the
//!   model" are different claims, and only the second one matters;
//! - it makes the graphics-crate cluster (winit / wgpu / egui / naga) upgradeable
//!   in minutes rather than days, because a regression shows up as a failed
//!   assertion instead of a blank window someone has to notice;
//! - it is the foundation the visual-regression harness will sit on once there
//!   are overlays and shading modes worth pinning down.
//!
//! The test skips rather than fails when no adapter is available, so CI runners
//! without a GPU do not report a false negative.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, ShadingMode, Viewport};

const SIZE: u32 = 256;
/// 256 px * 4 bytes is already a multiple of wgpu's 256-byte row alignment, so
/// the readback needs no per-row padding logic.
const BYTES_PER_ROW: u32 = SIZE * 4;

struct Rendered {
    pixels: Vec<[u8; 4]>,
}

impl Rendered {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        self.pixels[(y * SIZE + x) as usize]
    }

    /// Fraction of pixels that differ noticeably from the viewport background.
    fn covered_fraction(&self, background: [u8; 4]) -> f32 {
        let differs = |p: &[u8; 4]| (0..3).any(|i| (p[i] as i32 - background[i] as i32).abs() > 8);
        self.pixels.iter().filter(|p| differs(p)).count() as f32 / self.pixels.len() as f32
    }

    fn distinct_luminances(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for p in &self.pixels {
            let l = (p[0] as u32 * 30 + p[1] as u32 * 59 + p[2] as u32 * 11) / 100;
            seen.insert(l / 4);
        }
        seen.len()
    }
}

/// Renders one frame of a scene offscreen. Returns `None` when there is no GPU.
fn render_scene(
    meshes: &[(kreate_mesh::MeshData, DAffine3)],
    shading: ShadingMode,
    configure_camera: impl FnOnce(&mut Camera),
) -> Option<(Rendered, [u8; 4])> {
    render_scene_opts(meshes, shading, true, configure_camera)
}

fn render_scene_opts(
    meshes: &[(kreate_mesh::MeshData, DAffine3)],
    shading: ShadingMode,
    show_grid: bool,
    configure_camera: impl FnOnce(&mut Camera),
) -> Option<(Rendered, [u8; 4])> {
    render_scene_full(meshes, shading, show_grid, false, configure_camera)
}

fn render_scene_full(
    meshes: &[(kreate_mesh::MeshData, DAffine3)],
    shading: ShadingMode,
    show_grid: bool,
    show_wireframe: bool,
    configure_camera: impl FnOnce(&mut Camera),
) -> Option<(Rendered, [u8; 4])> {
    render_scene_flags(meshes, shading, show_grid, show_wireframe, false, configure_camera)
}

fn render_scene_flags(
    meshes: &[(kreate_mesh::MeshData, DAffine3)],
    shading: ShadingMode,
    show_grid: bool,
    show_wireframe: bool,
    face_orientation: bool,
    configure_camera: impl FnOnce(&mut Camera),
) -> Option<(Rendered, [u8; 4])> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless target"),
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
    viewport.shading = shading;
    viewport.show_grid = show_grid;
    viewport.show_wireframe = show_wireframe;
    viewport.face_orientation = face_orientation;

    let uploaded: Vec<(GpuMesh, DAffine3)> = meshes
        .iter()
        .map(|(mesh, xform)| {
            let cpu = RenderMeshCpu::extract(mesh);
            let gpu_mesh = GpuMesh::upload(
                &gpu.device,
                &viewport.layouts,
                &cpu,
                Vec4::new(0.75, 0.72, 0.68, 1.0),
            );
            (gpu_mesh, *xform)
        })
        .collect();
    let mut refs: Vec<Draw> = uploaded.iter().map(|(m, t)| Draw::new(m, *t)).collect();

    let mut camera = Camera::default();
    configure_camera(&mut camera);

    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut refs);

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (BYTES_PER_ROW * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless"),
        });
    viewport.render(&mut encoder, &view, &refs);
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
    gpu.queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map callback").expect("map");

    let data = readback.slice(..).get_mapped_range();
    let pixels: Vec<[u8; 4]> = data.as_chunks::<4>().0.to_vec();
    drop(data);
    readback.unmap();

    let bg = viewport.background;
    let to_u8 = |v: f64| {
        // The target is sRGB, so the hardware encodes on write; convert the
        // linear clear colour the same way to compare against it.
        let s = if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    let background = [to_u8(bg.r), to_u8(bg.g), to_u8(bg.b), 255];

    Some((Rendered { pixels }, background))
}

/// Skips the test body when no GPU is present, so a CI runner without one does
/// not report a failure it cannot do anything about.
macro_rules! gpu_or_skip {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => {
                eprintln!("no GPU adapter available; skipping");
                return;
            }
        }
    };
}

fn cube_scene() -> Vec<(kreate_mesh::MeshData, DAffine3)> {
    vec![(kreate_mesh::primitives::cube(2.0), DAffine3::IDENTITY)]
}

#[test]
fn a_cube_actually_reaches_the_framebuffer() {
    let (img, bg) = gpu_or_skip!(render_scene(&cube_scene(), ShadingMode::Studio, |c| {
        c.frame_sphere(DVec3::ZERO, 1.8, 1.0);
    }));

    let covered = img.covered_fraction(bg);
    assert!(
        covered > 0.05,
        "only {:.1}% of the frame differs from the background; nothing was drawn",
        covered * 100.0
    );
    assert!(
        covered < 0.98,
        "{:.1}% of the frame is non-background; the view is probably inside the mesh",
        covered * 100.0
    );
}

#[test]
fn studio_shading_produces_a_range_of_tones() {
    // A cube lit from one side must show its three visible faces at different
    // brightnesses. A single flat tone means the normals never arrived — the
    // exact symptom of a broken octahedral decode or a garbled normal matrix.
    let (img, _) = gpu_or_skip!(render_scene(&cube_scene(), ShadingMode::Studio, |c| {
        c.frame_sphere(DVec3::ZERO, 1.8, 1.0);
    }));
    assert!(
        img.distinct_luminances() > 6,
        "only {} distinct tones; shading looks flat",
        img.distinct_luminances()
    );
}

/// Distinct brightness levels found on the object, quantised coarsely.
///
/// A flat-shaded cube shows three visible faces and therefore about three
/// levels. Shading that varies across a face — which is what averaging every
/// face normal at a vertex produces — spreads those into a gradient.
fn tones_on_object(img: &Rendered, background: [u8; 4]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for p in &img.pixels {
        if (0..3).any(|i| (p[i] as i32 - background[i] as i32).abs() > 8) {
            let l = (p[0] as u32 * 30 + p[1] as u32 * 59 + p[2] as u32 * 11) / 100;
            seen.insert(l / 6);
        }
    }
    seen.len()
}

#[test]
fn a_cube_shades_as_flat_faces_not_as_a_rounded_lump() {
    // The defect a user reported as the cube "looking cut". Averaging every face
    // that meets at a vertex gives a cube's eight corners diagonal normals, and
    // interpolating those across a flat face shades it as if it were curved — so
    // each face becomes a gradient and the lighting clamps along a curve through
    // the middle of it.
    //
    // Splitting the shading at edges sharper than the auto-smooth threshold
    // leaves each face a single tone.
    let (img, bg) = gpu_or_skip!(render_scene_opts(
        &cube_scene(),
        ShadingMode::Studio,
        false,
        |c| c.frame_sphere(DVec3::ZERO, 1.6, 1.0)
    ));

    let tones = tones_on_object(&img, bg);
    assert!(
        tones <= 8,
        "{tones} distinct tones on a cube; the faces are shading as curved rather          than flat"
    );
    assert!(
        tones >= 2,
        "only {tones} tones; the faces are not being lit differently at all"
    );
}

#[test]
fn the_wireframe_draws_real_edges_and_not_triangulation_diagonals() {
    // The reason the overlay is built from the mesh's own edge list rather than
    // from the triangles. `PolygonMode::Line`, a `LineList` over the index
    // buffer, and a barycentric derivative trick all draw *triangulation* edges,
    // so every quad shows a diagonal — which is disqualifying for a polygon
    // modeller, where the quad is the thing the user is working with.
    //
    // A single quad viewed face-on makes that unambiguous: its diagonal would run
    // exactly through the centre of the frame.
    let plane = vec![(kreate_mesh::primitives::plane(2.0), DAffine3::IDENTITY)];
    let face_on = |c: &mut Camera| {
        // Straight down the plane's normal, so the quad fills the frame square.
        c.pivot = DVec3::ZERO;
        c.rotation = glam::DQuat::IDENTITY;
        c.distance = 3.0;
    };

    let (without, _) = gpu_or_skip!(render_scene_full(
        &plane,
        ShadingMode::Flat,
        false,
        false,
        face_on
    ));
    let (with, _) = gpu_or_skip!(render_scene_full(
        &plane,
        ShadingMode::Flat,
        false,
        true,
        face_on
    ));

    let centre = SIZE / 2;
    assert_eq!(
        with.at(centre, centre),
        without.at(centre, centre),
        "a line runs through the middle of the quad; the overlay is drawing the          triangulation rather than the mesh's edges"
    );

    // And the real edges must actually be drawn, or the test above passes for
    // the wrong reason.
    let differing = with
        .pixels
        .iter()
        .zip(&without.pixels)
        .filter(|(a, b)| (0..3).any(|i| (a[i] as i32 - b[i] as i32).abs() > 20))
        .count();
    assert!(
        differing > 200,
        "only {differing} pixels changed with the wireframe on; it is not drawing"
    );
}

#[test]
fn a_cube_wireframe_is_visible_against_its_own_shading() {
    // The overlay's default colour and width have to read against the surface it
    // sits on. An earlier pair looked reasonable in the source and was invisible
    // on screen.
    let cube = cube_scene();
    let camera = |c: &mut Camera| c.frame_sphere(DVec3::ZERO, 1.6, 1.0);
    let (plain, _) = gpu_or_skip!(render_scene_full(
        &cube,
        ShadingMode::Studio,
        false,
        false,
        camera
    ));
    let (wire, _) = gpu_or_skip!(render_scene_full(
        &cube,
        ShadingMode::Studio,
        false,
        true,
        camera
    ));

    let changed = wire
        .pixels
        .iter()
        .zip(&plain.pixels)
        .filter(|(a, b)| (0..3).any(|i| (a[i] as i32 - b[i] as i32).abs() > 40))
        .count();
    assert!(
        changed > 400,
        "only {changed} pixels differ strongly; the wireframe is too faint to see"
    );
}

#[test]
fn the_grid_is_drawn_when_the_scene_is_empty() {
    let (img, bg) = gpu_or_skip!(render_scene(&[], ShadingMode::Studio, |c| {
        c.frame_sphere(DVec3::ZERO, 4.0, 1.0);
    }));
    let covered = img.covered_fraction(bg);
    assert!(
        covered > 0.01,
        "the grid should be visible on an empty scene, but only {:.2}% differs from the background",
        covered * 100.0
    );
}

#[test]
fn geometry_occludes_the_grid() {
    // The grid writes real depth from its ray hit, so a solid object sitting on
    // the origin must hide the lines behind it. If the grid were drawn without
    // depth, the centre of the frame would show grid lines over the cube.
    let camera_setup = |c: &mut Camera| {
        c.pivot = DVec3::ZERO;
        c.distance = 6.0;
    };
    let (with_cube, _) = gpu_or_skip!(render_scene(&cube_scene(), ShadingMode::Flat, camera_setup));
    let (empty, bg) = gpu_or_skip!(render_scene(&[], ShadingMode::Flat, camera_setup));

    // The centre pixel: grid or background when empty, cube when not.
    let centre = SIZE / 2;
    let cube_px = with_cube.at(centre, centre);
    let empty_px = empty.at(centre, centre);
    assert_ne!(
        cube_px, empty_px,
        "the cube should change the centre of the frame"
    );

    let differs_from_bg = |p: [u8; 4]| (0..3).any(|i| (p[i] as i32 - bg[i] as i32).abs() > 8);
    assert!(
        differs_from_bg(cube_px),
        "the centre should be covered by the cube, got {cube_px:?}"
    );
}

#[test]
fn a_far_away_scene_renders_identically_to_one_at_the_origin() {
    // Camera-relative rendering means the position of a scene in the world must
    // not affect what it looks like. Without it, a scene 10 km out renders as a
    // jittering mess, and this is the cheapest way to keep that honest.
    //
    // The grid is switched off: it is anchored to the world, so its lattice
    // phase and axis lines legitimately differ 10 km from the origin. Leaving it
    // on would make this test measure the grid rather than the geometry.
    let at_origin = gpu_or_skip!(render_scene_opts(
        &cube_scene(),
        ShadingMode::Studio,
        false,
        |c| {
            c.frame_sphere(DVec3::ZERO, 1.8, 1.0);
        }
    ));

    let far = DVec3::splat(10_000.0);
    let far_scene = vec![(
        kreate_mesh::primitives::cube(2.0),
        DAffine3::from_translation(far),
    )];
    let far_away = gpu_or_skip!(render_scene_opts(
        &far_scene,
        ShadingMode::Studio,
        false,
        move |c| {
            c.frame_sphere(far, 1.8, 1.0);
        }
    ));

    let differing = at_origin
        .0
        .pixels
        .iter()
        .zip(&far_away.0.pixels)
        .filter(|(a, b)| (0..3).any(|i| (a[i] as i32 - b[i] as i32).abs() > 6))
        .count();
    let fraction = differing as f32 / at_origin.0.pixels.len() as f32;
    assert!(
        fraction < 0.02,
        "{:.2}% of pixels differ between a scene at the origin and the same scene \
         10 km away; camera-relative rendering is not working",
        fraction * 100.0
    );
}

#[test]
fn an_empty_viewport_clears_to_the_palette_colour() {
    // The clear value is linear and the target is sRGB, so what reaches the
    // texture is the encoded form. Asserting on the *encoded* byte is the
    // point: it is the only way to catch a clear that was written in the wrong
    // colour space, which is exactly the mistake that made the selection
    // accent a pale yellow before the palette existed.
    let (img, _) = gpu_or_skip!(render_scene_opts(&[], ShadingMode::Studio, false, |_| {}));
    let expected = kreate_render::palette::rgb8(kreate_render::palette::VIEWPORT);
    for (x, y) in [(2, 2), (SIZE - 3, 2), (SIZE / 2, SIZE - 3)] {
        let p = img.at(x, y);
        assert_eq!(
            [p[0], p[1], p[2]],
            expected,
            "cleared pixel at ({x},{y}) is not the palette's viewport colour"
        );
    }
}

#[test]
fn the_grid_axes_are_drawn_in_the_shared_axis_colours() {
    // The grid is what tells the user which way is which, and it has to say it
    // in the same three colours as the navigation gizmo and the transform
    // fields — otherwise each widget teaches a different lesson.
    //
    // Not covered by the visual-regression goldens: two one-pixel lines across
    // a 256-pixel frame are well under that harness's 1.5% budget, so the axis
    // colours could be changed there without a single golden noticing. This
    // asserts on them directly.
    let (img, _) = gpu_or_skip!(render_scene_opts(&[], ShadingMode::Studio, true, |_| {}));

    let count_near = |want: [u8; 3]| {
        img.pixels
            .iter()
            .filter(|p| (0..3).all(|i| (i32::from(p[i]) - i32::from(want[i])).abs() <= 12))
            .count()
    };

    for (name, hex) in [
        ("X", kreate_render::palette::AXIS_X),
        ("Y", kreate_render::palette::AXIS_Y),
    ] {
        let n = count_near(kreate_render::palette::rgb8(hex));
        assert!(
            n > 20,
            "only {n} pixels near the {name} axis colour; the grid is not              drawing it from the palette"
        );
    }

    // And the lattice itself. It is drawn at partial coverage over the
    // background, so no pixel is the palette colour exactly; what is pinned is
    // how bright the brightest lattice pixel gets. Left unchecked, the lattice
    // colour is free to drift — the visual-regression goldens render without
    // the grid, so nothing else looks at it at all.
    let neutral_peak = img
        .pixels
        .iter()
        .filter(|p| {
            // Skip the axis lines, which are deliberately not neutral.
            let (r, g, b) = (i32::from(p[0]), i32::from(p[1]), i32::from(p[2]));
            (r - g).abs() <= 6 && (g - b).abs() <= 8
        })
        .map(|p| i32::from(p[0]))
        .max()
        .unwrap_or(0);
    assert!(
        (50..=90).contains(&neutral_peak),
        "the brightest lattice pixel is {neutral_peak}; the grid line colour          has drifted from the palette"
    );
}

#[test]
fn face_orientation_is_blue_from_the_front_and_red_from_the_back() {
    // A plane facing +Z, seen from above and then from below.
    let plane = || vec![(kreate_mesh::primitives::plane(4.0), DAffine3::IDENTITY)];
    let centre = |r: &Rendered| r.at(SIZE / 2, SIZE / 2);
    let (above, _) = gpu_or_skip!(render_scene_flags(&plane(), ShadingMode::Studio, false, false, true, |c| {
        *c = c.looking_along(kreate_render::ViewAxis::PosZ);
    }));
    let p = centre(&above);
    assert!(p[2] > p[0] + 40, "the front is not blue: {p:?}");
    let (below, _) = gpu_or_skip!(render_scene_flags(&plane(), ShadingMode::Studio, false, false, true, |c| {
        *c = c.looking_along(kreate_render::ViewAxis::NegZ);
    }));
    let p = centre(&below);
    assert!(p[0] > p[2] + 40, "the back is not red: {p:?}");
    // Off, neither.
    let (plain, _) = gpu_or_skip!(render_scene_flags(&plane(), ShadingMode::Studio, false, false, false, |c| {
        *c = c.looking_along(kreate_render::ViewAxis::PosZ);
    }));
    let p = centre(&plain);
    assert!((p[0] as i32 - p[2] as i32).abs() < 40, "tinted with the overlay off: {p:?}");
}
