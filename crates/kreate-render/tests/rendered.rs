//! The rendered mode, end to end.
//!
//! Every ingredient has been checked against the CPU on a device already. What
//! this checks is the *assembly*: that a scene with a light in it comes out
//! lit, that the display transform is applied once, that the overlays are not
//! put through it, and that switching modes changes the picture.
//!
//! Rendered offscreen and read back. A pixel test would confirm that something
//! appeared; these look at what the pixels *are*, because the failures worth
//! catching here — a mode that shades but never resolves, a resolve applied
//! twice, an unlit scene that comes out grey anyway — all produce a picture.

use kreate_render::display::Display;
use kreate_render::lights::{Falloff, PlacedLight};
use kreate_render::viewport::{Draw, ShadingMode, Viewport, VIEWPORT_DEPTH_FORMAT};
use kreate_render::{brdf::Surface, Camera, GpuOptions, RenderMeshCpu};

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

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

const SIZE: u32 = 64;
/// What a window would give, near enough.
///
/// Not a float target, which was the first attempt: the overlay pipelines
/// blend, and `Rgba32Float` is not blendable in the WebGPU core. Eight bits is
/// what a screen has anyway, and it turns out to sharpen the test rather than
/// blunt it — see `the_display_transform_is_applied_and_applied_once`, which
/// counts *saturated* pixels, a measurement only a bounded format can make.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A cube filling most of the view, a camera looking at it, and whatever
/// lights are given.
fn shoot(
    gpu: &Gpu,
    shading: ShadingMode,
    surface: Surface,
    lights: &[PlacedLight],
    display: Display,
) -> Vec<[f32; 4]> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = shading;
    viewport.set_display(&gpu.queue, display);
    // The grid and the wireframe are interface, and this test is about the
    // shading. Left on, they would be the brightest thing in the frame.
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    let mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
    let uploaded =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &mesh, glam::Vec4::ONE);

    // Looking at the origin from six metres, which frames a two-metre cube.
    let camera = Camera {
        distance: 6.0,
        ..Camera::default()
    };

    let mut draws = vec![Draw::new(&uploaded, glam::DAffine3::IDENTITY).with_surface(surface)];
    viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), lights);
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    assert_eq!(VIEWPORT_DEPTH_FORMAT, wgpu::TextureFormat::Depth32Float);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    viewport.render(&mut encoder, &view, &draws);

    let row = (SIZE * 4).div_ceil(256) * 256;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
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
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();

    let mut out = Vec::with_capacity((SIZE * SIZE) as usize);
    for y in 0..SIZE {
        let base = (y * row) as usize;
        let line = &data[base..base + (SIZE * 4) as usize];
        for x in 0..SIZE as usize {
            out.push([
                line[x * 4] as f32 / 255.0,
                line[x * 4 + 1] as f32 / 255.0,
                line[x * 4 + 2] as f32 / 255.0,
                line[x * 4 + 3] as f32 / 255.0,
            ]);
        }
    }
    out
}

/// An exposure that frames a physically lit scene.
///
/// **A gap, recorded rather than hidden.** The lights are photometric — a sun
/// is twenty thousand lux — and the display transform expects scene-linear
/// values where 0.18 is mid grey. Nothing bridges the two yet, so a physically
/// correct scene at the default exposure saturates every pixel, which is
/// exactly what the first version of these tests measured.
///
/// The bridge belongs on the camera: §2.5 already stores the f-stop, the
/// shutter and the focus distance, and the standard photographic exposure
/// falls out of those plus a sensitivity. Until that is wired up, a test has
/// to supply the number, and so would a user.
///
/// Twelve stops down puts a sunlit grey card near one, which is where AgX
/// wants it.
const FRAMED: Display = Display {
    view: kreate_math::ViewTransform::AgX,
    exposure: -12.0,
};

fn grey() -> Surface {
    Surface {
        base_color: kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
        metallic: 0.0,
        roughness: 0.5,
    }
}

/// A sun pointing away from the camera, so it lights the side that faces it.
fn key_light(lux: f32) -> PlacedLight {
    PlacedLight {
        falloff: Falloff::Parallel,
        direction: glam::Vec3::new(-0.3, 0.4, -0.87).normalize(),
        amount: [lux; 3],
        ..PlacedLight::default()
    }
}

/// The brightest pixel, how many are lit, and how many are at the top.
///
/// The third is what says whether the display transform ran: an unresolved
/// linear image of a sunlit surface is thousands of times over the display's
/// range, so every pixel of it saturates. AgX leaves a gradient.
struct Shot {
    brightest: f32,
    lit: usize,
    saturated: usize,
}

fn summary(pixels: &[[f32; 4]]) -> Shot {
    Shot {
        brightest: pixels.iter().fold(0.0f32, |m, p| m.max(p[0])),
        lit: pixels.iter().filter(|p| p[0] > 0.01).count(),
        // 254 rather than 255, so a value one code short of the top still
        // counts: the question is "clipped", not "exactly at the maximum".
        saturated: pixels.iter().filter(|p| p[0] >= 254.0 / 255.0).count(),
    }
}

/// A cube hanging over a wide flat plane, lit from above.
///
/// The one arrangement where a shadow is unambiguous: the plane under the cube
/// is the only part of the picture that should change when shadows are turned
/// on, and it should get darker.
fn cube_over_a_floor(gpu: &Gpu, cast_shadow: bool) -> Vec<[f32; 4]> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.set_display(&gpu.queue, FRAMED);
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    let floor_mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::plane(16.0));
    let cube_mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
    let floor =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &floor_mesh, glam::Vec4::ONE);
    let cube =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cube_mesh, glam::Vec4::ONE);

    // Looking down at the floor from off to one side, so the shadow is in
    // frame rather than hidden behind the cube.
    let camera = Camera {
        distance: 14.0,
        ..Camera::default()
    };
    let eye = camera.eye();
    let lifted = glam::DAffine3::from_translation(glam::DVec3::new(0.0, 0.0, 3.0));

    let mut draws = vec![
        Draw::new(&floor, glam::DAffine3::IDENTITY).with_surface(grey()),
        Draw::new(&cube, lifted).with_surface(grey()),
    ];

    // Straight down, so the shadow lands under the cube.
    let sun = PlacedLight {
        falloff: Falloff::Parallel,
        direction: glam::Vec3::NEG_Z,
        amount: [20_000.0; 3],
        cast_shadow,
        ..PlacedLight::default()
    };
    // Eye-relative, like everything the renderer is given.
    let bounds = kreate_math::Aabb::new(
        glam::Vec3::new(-9.0, -9.0, -1.0) - eye.as_vec3(),
        glam::Vec3::new(9.0, 9.0, 5.0) - eye.as_vec3(),
    );
    viewport.set_scene_bounds(bounds);
    viewport.set_world([0.0; 3]);
    viewport.set_lights(&gpu.device, &gpu.queue, eye, &[sun]);
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow shot"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    viewport.render(&mut encoder, &view, &draws);

    let row = (SIZE * 4).div_ceil(256) * 256;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
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
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();

    let mut out = Vec::with_capacity((SIZE * SIZE) as usize);
    for y in 0..SIZE {
        let base = (y * row) as usize;
        let line = &data[base..base + (SIZE * 4) as usize];
        for x in 0..SIZE as usize {
            out.push([
                line[x * 4] as f32 / 255.0,
                line[x * 4 + 1] as f32 / 255.0,
                line[x * 4 + 2] as f32 / 255.0,
                line[x * 4 + 3] as f32 / 255.0,
            ]);
        }
    }
    out
}

#[test]
fn a_cube_casts_a_shadow_on_the_floor() {
    // The whole point of the two commits, in one measurement: the same scene
    // with the light's `cast_shadow` on and off, and the difference has to be
    // darkness somewhere.
    //
    // A comparison rather than an absolute, because "is this pixel dark" has
    // a dozen innocent reasons to be true and "did turning shadows on darken
    // it" has one.
    let gpu = gpu_or_skip!(gpu());
    let lit = cube_over_a_floor(&gpu, false);
    let shadowed = cube_over_a_floor(&gpu, true);
    assert_eq!(lit.len(), shadowed.len());

    let darkened = lit
        .iter()
        .zip(&shadowed)
        .filter(|(a, b)| b[0] + 0.02 < a[0])
        .count();
    assert!(
        darkened > 30,
        "only {darkened} pixels got darker when shadows were turned on"
    );

    // And nothing got *brighter*, which a bias pushed the wrong way would do.
    let brightened = lit
        .iter()
        .zip(&shadowed)
        .filter(|(a, b)| b[0] > a[0] + 0.02)
        .count();
    assert_eq!(brightened, 0, "{brightened} pixels got brighter");
}

#[test]
fn the_lit_parts_of_the_scene_are_not_darkened() {
    // Shadow acne: a surface shadowing itself because its own depth compares
    // as nearer than what the map stored. It shows as a dirty stipple over
    // everything the light touches, and the bias is what prevents it.
    //
    // Measured as the fraction of the picture that changed at all. A cube over
    // a floor puts a small shadow in a large frame, so acne — which covers
    // every lit surface — is an order of magnitude more pixels than the shadow
    // itself.
    let gpu = gpu_or_skip!(gpu());
    let lit = cube_over_a_floor(&gpu, false);
    let shadowed = cube_over_a_floor(&gpu, true);

    let changed = lit
        .iter()
        .zip(&shadowed)
        .filter(|(a, b)| (a[0] - b[0]).abs() > 0.02)
        .count();
    let visible = lit.iter().filter(|p| p[0] > 0.01).count();
    assert!(
        changed * 3 < visible,
        "{changed} of {visible} lit pixels changed; that is acne, not a shadow"
    );
}

#[test]
fn the_same_scene_renders_the_same_twice() {
    // Not about shadows as such — it is about the buffers the shadow path
    // added. Each frame writes the light entries, the shadow matrices and the
    // per-draw matrices and then reads them back in the same frame, and a
    // stale one shows as the *previous* frame's shadow: a dark patch that
    // lags the light by a frame and vanishes if you look for it.
    //
    // Two identical frames are what catches that, and the shadow path is
    // where it became worth checking.
    //
    // (That `cast_shadow` reaches the renderer at all is what
    // `a_cube_casts_a_shadow_on_the_floor` says, by turning it off.)
    let gpu = gpu_or_skip!(gpu());
    let a = cube_over_a_floor(&gpu, true);
    let b = cube_over_a_floor(&gpu, true);
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x, y, "pixel {i} differed between identical frames");
    }
}

#[test]
fn a_lit_scene_comes_out_lit() {
    // The assembly, at its bluntest. Every ingredient was already checked
    // against the CPU; this says they were wired together.
    let gpu = gpu_or_skip!(gpu());
    let shot = shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(20_000.0)],
        FRAMED,
    );
    let shot = summary(&shot);
    assert!(shot.brightest > 0.1, "nothing was lit: {}", shot.brightest);
    // The cube covers a good part of a 64-pixel frame.
    assert!(shot.lit > 400, "only {} pixels were lit", shot.lit);
}

#[test]
fn an_unlit_scene_is_black_and_not_grey() {
    // A renderer that supplied a default light, or that left the studio
    // shading in the pass, would light this — and the user would have no way
    // to turn it off. Black is the honest answer for a scene with no lights.
    let gpu = gpu_or_skip!(gpu());
    let shot = shoot(&gpu, ShadingMode::Rendered, grey(), &[], FRAMED);
    let shot = summary(&shot);
    assert!(
        shot.brightest < 0.01,
        "an unlit scene was lit: {}",
        shot.brightest
    );
}

#[test]
fn the_display_transform_is_applied_and_applied_once() {
    // The clue that it ran at all: a sun at twenty thousand lux on a grey
    // surface is far outside the display's range, so an unresolved image
    // would read as saturated white everywhere the cube is. AgX leaves
    // headroom.
    let gpu = gpu_or_skip!(gpu());
    let shot = shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(20_000.0)],
        FRAMED,
    );
    let shot = summary(&shot);
    assert!(
        shot.brightest > 0.5,
        "a sunlit surface should be bright: {}",
        shot.brightest
    );
    // The measurement the format makes possible: a linear image of a surface
    // under twenty thousand lux is thousands of times over the display's
    // range, so reaching the target unresolved would clip nearly every pixel
    // of the cube. A transform that ran leaves a gradient.
    assert!(
        shot.saturated * 4 < shot.lit,
        "{} of {} lit pixels are clipped; the image looks unresolved",
        shot.saturated,
        shot.lit
    );

    // And a dimmer scene is dimmer, which a transform applied twice or not at
    // all would both still manage — this is the weaker half, and it is here
    // because a transform that ignored its input would pass everything above.
    let dim = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(200.0)],
        FRAMED,
    ));
    assert!(dim.brightest < shot.brightest);
    assert!(dim.brightest > 0.0);
}

#[test]
fn exposure_moves_the_picture() {
    // The uniform reaches the fragment stage of the resolve, through a
    // viewport that owns the buffer rather than allocating one per frame.
    let gpu = gpu_or_skip!(gpu());
    let flat = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(2_000.0)],
        FRAMED,
    ))
    .brightest;
    let raised = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &[key_light(2_000.0)],
        Display {
            view: kreate_math::ViewTransform::AgX,
            exposure: FRAMED.exposure + 2.0,
        },
    ))
    .brightest;
    assert!(raised > flat * 1.05, "{flat} then {raised}");
}

#[test]
fn the_material_reaches_the_shader() {
    // Per draw, through the ring the pass reads. A pipeline that ignored it
    // would shade everything as the default grey, and a scene of one object
    // would look correct.
    let gpu = gpu_or_skip!(gpu());
    let light = [key_light(5_000.0)];
    let dark = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        Surface {
            base_color: kreate_math::SceneRgb::new(0.05, 0.05, 0.05),
            metallic: 0.0,
            roughness: 0.5,
        },
        &light,
        FRAMED,
    ))
    .brightest;
    let bright = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        Surface {
            base_color: kreate_math::SceneRgb::new(0.9, 0.9, 0.9),
            metallic: 0.0,
            roughness: 0.5,
        },
        &light,
        FRAMED,
    ))
    .brightest;
    assert!(bright > dark * 1.3, "{dark} against {bright}");
}

#[test]
fn the_three_modes_are_three_different_pictures() {
    // Guards every test above: if the mode were ignored and the studio pass
    // ran throughout, the scene would be lit, the transform would appear to
    // clamp, and only this would notice.
    let gpu = gpu_or_skip!(gpu());
    let light = [key_light(20_000.0)];
    let studio = summary(&shoot(
        &gpu,
        ShadingMode::Studio,
        grey(),
        &light,
        FRAMED,
    ));
    let flat = summary(&shoot(
        &gpu,
        ShadingMode::Flat,
        grey(),
        &light,
        FRAMED,
    ));
    let rendered = summary(&shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &light,
        FRAMED,
    ));

    // All three draw the cube.
    for (name, shot) in [
        ("studio", &studio),
        ("flat", &flat),
        ("rendered", &rendered),
    ] {
        assert!(shot.lit > 400, "{name} drew {} pixels", shot.lit);
        assert!(shot.brightest > 0.0, "{name} drew nothing");
    }
    // And the rendered one is not either of the others: the studio pass
    // ignores the scene's lights and this one is lit by a sun through a
    // display transform.
    assert!(
        (rendered.brightest - studio.brightest).abs() > 0.02,
        "rendered {} against studio {}",
        rendered.brightest,
        studio.brightest
    );
}

#[test]
fn a_resize_keeps_the_intermediate_in_step() {
    // The bind group holds the old view, so a resize that remade the texture
    // and not the bind group would resolve last frame's image at the new
    // size — a picture, and the wrong one.
    let gpu = gpu_or_skip!(gpu());
    let mut viewport = Viewport::new(&gpu.device, FORMAT, 32, 32);
    viewport.shading = ShadingMode::Rendered;
    viewport.resize(&gpu.device, 96, 48);

    // Drawing after the resize must not be a validation error, which is what
    // a stale bind group would be once the texture behind it was dropped.
    let mesh = RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0));
    let uploaded =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &mesh, glam::Vec4::ONE);
    // Looking at the origin from six metres, which frames a two-metre cube.
    let camera = Camera {
        distance: 6.0,
        ..Camera::default()
    };
    let mut draws = vec![Draw::new(&uploaded, glam::DAffine3::IDENTITY)];
    viewport.set_lights(
        &gpu.device,
        &gpu.queue,
        camera.eye(),
        &[key_light(10_000.0)],
    );
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resized"),
        size: wgpu::Extent3d {
            width: 96,
            height: 48,
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
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    viewport.render(&mut encoder, &view, &draws);
    gpu.queue.submit([encoder.finish()]);
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
}

#[test]
fn more_lights_than_the_buffer_started_with_still_light_the_scene() {
    // The buffer grows by being replaced, and the bind group holds the old
    // one. A grow without a rebuild is a validation error at best and last
    // frame's lights at worst.
    let gpu = gpu_or_skip!(gpu());
    let many: Vec<PlacedLight> = (0..24)
        .map(|i| PlacedLight {
            position: glam::DVec3::new(i as f64 * 0.1 - 1.2, -4.0, 2.0),
            amount: [400.0; 3],
            ..PlacedLight::default()
        })
        .collect();
    let shot = shoot(
        &gpu,
        ShadingMode::Rendered,
        grey(),
        &many,
        FRAMED,
    );
    let shot = summary(&shot);
    assert!(
        shot.brightest > 0.05,
        "the grown buffer lit nothing: {}",
        shot.brightest
    );
    assert!(shot.lit > 400, "only {} pixels were lit", shot.lit);
}
