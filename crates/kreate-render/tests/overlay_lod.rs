//! Screen-size culling and overlay density LOD, against a real GPU.
//!
//! The unit tests in `lod.rs` prove the arithmetic. They cannot prove that the
//! arithmetic reaches the screen, and that gap is where this kind of
//! optimisation silently dies: a uniform written to the wrong slot, a shader
//! reading the wrong component, a draw loop consulting a field nobody fills in.
//! Every one of those leaves the whole feature a no-op and every unit test
//! green.
//!
//! So each claim here is made by holding the geometry and the camera fixed and
//! changing *only* the budget. If the dots disappear when the threshold moves
//! and come back when it moves again, the mechanism is connected. Nothing else
//! about the frame differs, so nothing else can explain the difference.
//!
//! # What a falsification sweep found these tests cannot see
//!
//! Each mechanism was deleted in turn to check that something failed. Two do
//! not show up here, for reasons worth recording rather than papering over:
//!
//! - **The face fill's cull check.** That pass draws only *selected* faces, and
//!   an object with a selection is protected from culling, so the branch is
//!   defensive and unreachable in practice.
//! - **The overlay passes' `draw` check**, which is redundant with their alpha
//!   check: a culled object reports zero alpha as well. Pinned by a unit test
//!   in `lod.rs` instead, so that the day the two stop agreeing is a failure
//!   rather than a surprise.
//!
//! Skips rather than fails when no adapter is available.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::lod::Budget;
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, ViewAxis, Viewport};

const SIZE: u32 = 256;
const BYTES_PER_ROW: u32 = SIZE * 4;

/// What the overlays are drawn over.
///
/// Nearly black, and deliberately so: the vertex dots are pale, and against the
/// renderer's usual mid-grey surface a "count the pale pixels" test counts the
/// surface too and passes whatever happens. Making the surface dark is what
/// turns pale pixels into a measurement of the overlay alone.
const DARK_SURFACE: Vec4 = Vec4::new(0.01, 0.01, 0.012, 1.0);

/// Pale pixels — the vertex dots and the wireframe, over the dark surface.
fn overlay_pixels(pixels: &[[u8; 4]]) -> usize {
    pixels
        .iter()
        .filter(|p| p[0] > 120 && p[1] > 120 && p[2] > 120)
        .count()
}

/// Pixels at the full brightness of an unfaded vertex dot.
///
/// The dots are drawn at a linear 0.62, which reaches the sRGB target near 210.
/// Halving the alpha over a near-black surface drops that to about 150, so a
/// threshold between the two separates a faded dot from an unfaded one — which
/// is the only way to observe a *partial* fade in a pixel count.
fn bright_pixels(pixels: &[[u8; 4]]) -> usize {
    pixels.iter().filter(|p| p[0] > 195 && p[1] > 195).count()
}

/// Pixels that are not the black background.
fn lit_pixels(pixels: &[[u8; 4]]) -> usize {
    pixels.iter().filter(|p| p[0] > 12 || p[1] > 12).count()
}

/// How the frame under test is set up.
struct Scene {
    mesh: kreate_mesh::MeshData,
    /// How far the camera sits from the origin.
    distance: f64,
    budget: Budget,
    wireframe: bool,
    elements: bool,
    selected: bool,
    editing: bool,
}

impl Scene {
    fn new(mesh: kreate_mesh::MeshData) -> Self {
        Self {
            mesh,
            distance: 4.0,
            budget: Budget::default(),
            wireframe: false,
            elements: true,
            selected: false,
            editing: false,
        }
    }
}

fn render(scene: &Scene) -> Option<Vec<[u8; 4]>> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lod target"),
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
    viewport.show_elements = scene.elements;
    viewport.show_wireframe = scene.wireframe;
    // The grid would cover the frame in lines and make every pixel count
    // meaningless.
    viewport.show_grid = false;
    viewport.background = wgpu::Color::BLACK;
    viewport.lod_budget = scene.budget;

    let cpu = RenderMeshCpu::extract(&scene.mesh);
    let uploaded = GpuMesh::upload(&gpu.device, &viewport.layouts, &cpu, DARK_SURFACE);

    let mut draws = vec![Draw::new(&uploaded, DAffine3::IDENTITY)
        .selected(scene.selected)
        .editing(scene.editing)];
    let mut camera = Camera::default().looking_along(ViewAxis::PosZ);
    camera.pivot = DVec3::ZERO;
    camera.distance = scene.distance;

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

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let data = slice.get_mapped_range();
    let pixels = data.as_chunks::<4>().0.to_vec();
    drop(data);
    readback.unmap();
    Some(pixels)
}

/// A budget that never simplifies anything, for the control frame.
fn permissive() -> Budget {
    Budget {
        cull_px: 0.0,
        point_full_px2: 0.0,
        point_none_px2: -1.0,
        wire_full_px2: 0.0,
        wire_none_px2: -1.0,
    }
}

#[test]
fn a_dense_mesh_loses_its_vertex_dots_and_a_looser_budget_brings_them_back() {
    // The central claim, and stated so that only the mechanism can satisfy it:
    // one mesh, one camera, two budgets. A dense grid framed to fill the view
    // has far less than a pixel of screen per vertex, so the dots must be gone.
    // Loosening the threshold on the identical frame must bring them back.
    //
    // Without the second half this test would pass on a renderer that simply
    // failed to draw dots on large meshes for any reason at all.
    let dense = || kreate_mesh::primitives::grid(2.0, 200, 200);

    let faded = Scene::new(dense());
    let Some(faded) = render(&faded) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let mut control = Scene::new(dense());
    control.budget = permissive();
    let control = render(&control).expect("adapter worked a moment ago");

    let faded_dots = overlay_pixels(&faded);
    let control_dots = overlay_pixels(&control);

    assert_eq!(
        faded_dots, 0,
        "40401 vertices over a 256px frame is 1.6 px2 each; the dots should be gone"
    );
    assert!(
        control_dots > 1_000,
        "the control frame drew {control_dots} overlay pixels, so the dots were \
         never being drawn and the test above proved nothing"
    );
}

#[test]
fn a_sparse_mesh_keeps_its_vertex_dots() {
    // The regression this feature could most easily cause: an ordinary mesh at
    // an ordinary distance must be untouched. A fade that fires here would be a
    // visible bug in every normal editing session.
    let mut scene = Scene::new(kreate_mesh::primitives::grid(2.0, 6, 6));
    scene.distance = 4.0;
    let Some(pixels) = render(&scene) else {
        eprintln!("no adapter; skipping");
        return;
    };
    assert!(
        overlay_pixels(&pixels) > 200,
        "49 vertices over a 256px frame is thousands of pixels each; \
         the dots must all be drawn"
    );
}

#[test]
fn a_dense_wireframe_fades_and_a_looser_budget_brings_it_back() {
    let dense = || kreate_mesh::primitives::grid(2.0, 200, 200);

    let mut faded = Scene::new(dense());
    faded.wireframe = true;
    faded.elements = false;
    let Some(faded) = render(&faded) else {
        eprintln!("no adapter; skipping");
        return;
    };

    let mut control = Scene::new(dense());
    control.wireframe = true;
    control.elements = false;
    control.budget = permissive();
    let control = render(&control).expect("adapter worked a moment ago");

    // 80,400 edges over 37,500 px2 is 0.47 px2 each, which lands near the
    // bottom of the fade band: alpha about 0.04. Not skipped — drawn, and far
    // too faint to register. Worth stating precisely, because "no bright pixels"
    // and "no draw" are different claims and only one of them is true here.
    assert_eq!(
        overlay_pixels(&faded),
        0,
        "a wireframe faded to four percent should leave no bright pixels"
    );
    let control_wire = lit_pixels(&control);
    assert!(
        control_wire > 10_000,
        "the control frame lit {control_wire} pixels, so the wireframe was \
         never being drawn"
    );
}

#[test]
fn a_partially_faded_mesh_still_draws_its_dots_but_dimmer() {
    // This covers the one path everything else misses. When the fade reaches
    // zero the draw is skipped, so every other test here is satisfied by the
    // draw loop alone and passes with the shader's fade deleted — verified by
    // deleting it, which changed nothing at all. The *partial* fade is what
    // makes the transition smooth instead of a pop as the user orbits, and it
    // lives entirely in the shader.
    //
    // The awkward part, and the reason this test looks the way it does: at the
    // density that *causes* a partial fade, a 6-pixel dot every 3.7 square
    // pixels means every pixel is covered by about ten dots, and ten layers at
    // 55% alpha composite to 0.9997. The frame saturates and the fade becomes
    // unobservable in a brightness measurement — the first version of this test
    // read 24,959 bright pixels against a control of 25,596 and concluded,
    // wrongly, that the shader was ignoring the fade.
    //
    // Density and overlap are coupled by construction, so they have to be
    // decoupled by moving the *band* instead: a sparse mesh whose dots do not
    // touch, with thresholds placed so that sparse mesh sits mid-fade. That the
    // budget is a parameter is what makes this possible.
    let mesh = || kreate_mesh::primitives::grid(2.0, 6, 6);
    let band = Budget {
        point_none_px2: 400.0,
        point_full_px2: 1200.0,
        ..Budget::default()
    };

    let cpu = RenderMeshCpu::extract(&mesh());
    let mut camera = Camera::default().looking_along(ViewAxis::PosZ);
    camera.pivot = DVec3::ZERO;
    camera.distance = 4.0;
    let size = kreate_render::ViewSize::measure(cpu.bounds, &DAffine3::IDENTITY, &camera, SIZE);
    let expected = kreate_render::Detail::of(size, cpu.counts, &band, false);
    assert!(
        (0.2..0.8).contains(&expected.point_alpha),
        "this mesh and band are meant to sit mid-fade, but the alpha is {}; \
         outside that band the test proves nothing",
        expected.point_alpha
    );

    let mut faded = Scene::new(mesh());
    faded.budget = band;
    let Some(faded) = render(&faded) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let mut control = Scene::new(mesh());
    control.budget = permissive();
    let control = render(&control).expect("adapter worked a moment ago");

    // The dots are still drawn.
    assert!(
        overlay_pixels(&faded) > 100,
        "a half-faded mesh should still show its dots"
    );
    // But none of them reaches the brightness of an unfaded one.
    let faded_bright = bright_pixels(&faded);
    let control_bright = bright_pixels(&control);
    assert!(
        control_bright > 100,
        "the control frame had {control_bright} bright dots, so there was \
         nothing for the faded frame to be dimmer than"
    );
    assert!(
        faded_bright * 4 < control_bright,
        "faded frame had {faded_bright} bright pixels against {control_bright}; \
         the fade is not reaching the shader"
    );
}

#[test]
fn a_wireframe_dense_enough_is_skipped_rather_than_drawn_faint() {
    // The other end of the same mechanism. Past the bottom of the band the draw
    // is dropped entirely, and that is where the performance actually comes
    // from: at this density the wireframe is 180,600 line instances that are
    // never issued.
    let mesh = || kreate_mesh::primitives::grid(2.0, 300, 300);
    let cpu = RenderMeshCpu::extract(&mesh());
    let mut camera = Camera::default().looking_along(ViewAxis::PosZ);
    camera.pivot = DVec3::ZERO;
    camera.distance = 4.0;
    let size = kreate_render::ViewSize::measure(cpu.bounds, &DAffine3::IDENTITY, &camera, SIZE);
    let detail = kreate_render::Detail::of(size, cpu.counts, &Budget::default(), false);
    assert_eq!(
        detail.wire_alpha, 0.0,
        "this density is meant to be past the band entirely"
    );

    let mut scene = Scene::new(mesh());
    scene.wireframe = true;
    scene.elements = false;
    let Some(pixels) = render(&scene) else {
        eprintln!("no adapter; skipping");
        return;
    };
    assert_eq!(overlay_pixels(&pixels), 0);
}

#[test]
fn a_culled_object_draws_nothing_and_a_looser_budget_draws_it() {
    // Culling proved the way the fades are: one mesh, one camera, two budgets.
    // The cube fills the frame, so if it is not drawn that is the cull and
    // nothing else.
    //
    // The distance-based version of this test below cannot make this claim. At a
    // distance great enough to fall under two pixels the cube's *surface* is
    // sub-pixel and rasterises to nothing whether it is culled or not — so
    // deleting the solid pass's cull entirely left every test green. Only the
    // vertex dots, which are a constant size on screen, were carrying it.
    let mesh = || kreate_mesh::primitives::cube(2.0);
    let mut culled = Scene::new(mesh());
    culled.distance = 6.0;
    culled.budget = Budget {
        // Far above anything the frame can produce, so the cull is certain.
        cull_px: 10_000.0,
        ..Budget::default()
    };
    let Some(culled) = render(&culled) else {
        eprintln!("no adapter; skipping");
        return;
    };

    let mut drawn = Scene::new(mesh());
    drawn.distance = 6.0;
    drawn.budget = permissive();
    let drawn = render(&drawn).expect("adapter worked a moment ago");

    assert_eq!(
        lit_pixels(&culled),
        0,
        "a culled object left pixels behind: the surface or an overlay is not \
         consulting the decision"
    );
    assert!(
        lit_pixels(&drawn) > 1_000,
        "the control frame drew nothing, so the assertion above proved nothing"
    );
}

#[test]
fn a_distant_object_falls_under_the_real_threshold() {
    // The threshold itself, at a real distance rather than a contrived budget.
    // What this shows is that the default `cull_px` fires where it should; the
    // test above is the one that shows every pass honours it.
    let mut scene = Scene::new(kreate_mesh::primitives::cube(2.0));
    scene.distance = 4_000.0;
    let Some(pixels) = render(&scene) else {
        eprintln!("no adapter; skipping");
        return;
    };
    // The vertex dots are a constant six pixels on screen regardless of
    // distance, so an unculled speck would still paint them — which is what
    // makes this measurable at all.
    assert_eq!(lit_pixels(&pixels), 0, "a speck should not have been drawn");

    let mut near = Scene::new(kreate_mesh::primitives::cube(2.0));
    near.distance = 6.0;
    let near = render(&near).expect("adapter worked a moment ago");
    assert!(lit_pixels(&near) > 1_000, "the control frame drew nothing");
}

#[test]
fn a_selected_speck_is_still_drawn() {
    // The protection rule, on the GPU. A user who selects something and then
    // watches it vanish has been told a lie about what exists, and that is
    // categorically worse than a slow frame — so selection has to survive the
    // cull even when the object is a single pixel.
    let mut scene = Scene::new(kreate_mesh::primitives::cube(2.0));
    scene.distance = 4_000.0;
    scene.selected = true;
    let Some(pixels) = render(&scene) else {
        eprintln!("no adapter; skipping");
        return;
    };
    assert!(
        lit_pixels(&pixels) > 0,
        "a selected object was culled; it must never be"
    );
}

#[test]
fn the_mesh_being_edited_keeps_its_dots_however_dense_it_is() {
    // The case the flag exists for. In edit mode the object-level `selected`
    // highlight is deliberately switched off — it would drown out the element
    // selection — so the edited mesh is not protected by selection, and without
    // an explicit flag it would be the one object in the scene eligible to have
    // its overlays faded. Exactly backwards: it is the only object whose
    // vertices the user is trying to hit.
    let mut scene = Scene::new(kreate_mesh::primitives::grid(2.0, 200, 200));
    scene.editing = true;
    let Some(pixels) = render(&scene) else {
        eprintln!("no adapter; skipping");
        return;
    };
    assert!(
        overlay_pixels(&pixels) > 1_000,
        "the edited mesh lost its vertex dots"
    );
}
