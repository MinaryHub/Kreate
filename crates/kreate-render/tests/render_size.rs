//! Every pass draws into the same region of the target.
//!
//! The target is allocated in steps and the window is not, so the region
//! actually drawn — `render_size` — is usually smaller than the texture. Each
//! pass has to set its viewport to that region, and the picture shows it the
//! moment one does not: the rendered pass projected geometry across the whole
//! allocation while the wireframe over it was projected across the drawn
//! region, and a cube sat a few pixels off its own outline. In a test at
//! 48×48 with a 48×48 allocation nothing could see it, which is why this one
//! allocates more than it draws.

use kreate_render::GpuOptions;
use kreate_render::{Camera, Draw, RenderMeshCpu, ShadingMode, Viewport};

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

/// The allocation, and the smaller region that is drawn.
const ALLOC: u32 = 96;
const DRAWN: (u32, u32) = (77, 61);
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One frame of a cube, in the given shading, read back as bytes.
fn shoot(gpu: &Gpu, shading: ShadingMode, wireframe: bool) -> Vec<[u8; 4]> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, ALLOC, ALLOC);
    viewport.set_render_size(DRAWN.0, DRAWN.1);
    viewport.shading = shading;
    viewport.show_grid = false;
    viewport.show_wireframe = wireframe;
    viewport.set_world([1.0; 3]);
    viewport.set_display(
        &gpu.queue,
        kreate_render::display::Display {
            view: kreate_math::ViewTransform::Standard,
            exposure: 0.0,
        },
    );
    let graph = kreate_scene::shader::ShaderGraph::standard(
        "Grey",
        kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
        0.0,
        0.5,
    );
    let compiled = kreate_shader::compile(&graph).expect("compiles");
    viewport.set_materials(
        &gpu.device,
        &gpu.queue,
        &[kreate_render::viewport::MaterialSource {
            wgsl: &compiled.wgsl,
            constants: &compiled.constants,
            image_slots: &compiled.image_slots,
            images: &[],
        }],
    );

    let mesh = kreate_mesh::primitives::cube(1.0);
    let cpu = RenderMeshCpu::extract(&mesh);
    let uploaded =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let camera = Camera {
        distance: 4.0,
        ..Camera::default()
    };
    let mut draws = vec![Draw::new(&uploaded, glam::DAffine3::IDENTITY).selected(true)];
    viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &[]);
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot"),
        size: wgpu::Extent3d {
            width: ALLOC,
            height: ALLOC,
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

    let row = (ALLOC * 4).div_ceil(256) * 256;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * ALLOC) as u64,
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
                rows_per_image: Some(ALLOC),
            },
        },
        wgpu::Extent3d {
            width: ALLOC,
            height: ALLOC,
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

    let mut out = Vec::with_capacity((ALLOC * ALLOC) as usize);
    for y in 0..ALLOC {
        let base = (y * row) as usize;
        for x in 0..ALLOC as usize {
            let at = base + x * 4;
            out.push([data[at], data[at + 1], data[at + 2], data[at + 3]]);
        }
    }
    out
}

/// Bounding box of the pixels `keep` accepts, as (min x, min y, max x, max y).
fn bbox(frame: &[[u8; 4]], keep: impl Fn([u8; 4]) -> bool) -> Option<(u32, u32, u32, u32)> {
    let mut b: Option<(u32, u32, u32, u32)> = None;
    for (i, p) in frame.iter().enumerate() {
        if !keep(*p) {
            continue;
        }
        let (x, y) = ((i as u32) % ALLOC, (i as u32) / ALLOC);
        b = Some(match b {
            None => (x, y, x, y),
            Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
        });
    }
    b
}

/// The selection accent: red well above blue, green in between.
fn is_accent(p: [u8; 4]) -> bool {
    p[0] > 180 && p[2] < 110 && p[1] > 90 && p[1] < 200
}

/// A grey shaded pixel: the three channels near each other and lit, but not
/// the white the rendered mode's background clips to, and not the dark the
/// studio mode's background is.
fn is_grey_surface(p: [u8; 4]) -> bool {
    let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
    (r - g).abs() < 12 && (g - b).abs() < 12 && r > 60 && r < 245 && !is_accent(p)
}

#[test]
fn the_shaded_object_sits_where_its_own_outline_is() {
    // The failure as the user saw it: a rendered cube several pixels off the
    // orange outline drawn over it. Both come from the same camera and the
    // same mesh, so their extents on screen must agree.
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let frame = shoot(&gpu, ShadingMode::Rendered, true);
    let surface = bbox(&frame, is_grey_surface).expect("a shaded cube");
    let outline = bbox(&frame, is_accent).expect("an orange outline");
    let off = |a: u32, b: u32| (a as i32 - b as i32).abs();
    // The outline is a stroke a couple of pixels wide, centred on the edge, so
    // it may stick out by that much; it may not sit somewhere else.
    for (name, a, b) in [
        ("left", surface.0, outline.0),
        ("top", surface.1, outline.1),
        ("right", surface.2, outline.2),
        ("bottom", surface.3, outline.3),
    ] {
        assert!(
            off(a, b) <= 3,
            "{name} edge: the shaded cube is at {a} and its outline at {b}; \
             shaded {surface:?}, outline {outline:?}"
        );
    }
}

/// Anything that is not the background, per mode. The studio clears to the
/// dark viewport colour; the rendered mode's background is a unit world, which
/// clips to white. Told apart per mode rather than excluding both everywhere,
/// because the studio's top face is lit to white too — a test that excluded
/// white from both would trim that face and call the trim a shift.
fn is_object(shading: ShadingMode) -> impl Fn([u8; 4]) -> bool {
    move |p| {
        let dark = p[0] < 40 && p[1] < 40 && p[2] < 40;
        let white = p[0] > 250 && p[1] > 250 && p[2] > 250;
        match shading {
            ShadingMode::Rendered => !dark && !white,
            _ => !dark,
        }
    }
}

#[test]
fn the_rendered_and_studio_pictures_put_the_object_in_the_same_place() {
    // Same camera, same mesh, two shading paths. Studio draws straight into
    // the target; Rendered goes through a linear intermediate and a resolve.
    // The route must not move the object.
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let studio = shoot(&gpu, ShadingMode::Studio, false);
    let rendered = shoot(&gpu, ShadingMode::Rendered, false);
    let a = bbox(&studio, is_object(ShadingMode::Studio)).expect("a studio cube");
    let b = bbox(&rendered, is_object(ShadingMode::Rendered)).expect("a rendered cube");
    let off = |x: u32, y: u32| (x as i32 - y as i32).abs();
    assert!(
        off(a.0, b.0) <= 1 && off(a.1, b.1) <= 1 && off(a.2, b.2) <= 1 && off(a.3, b.3) <= 1,
        "studio puts the cube at {a:?} and rendered at {b:?}"
    );
}

#[test]
fn nothing_is_drawn_outside_the_region_the_caller_samples() {
    // The other half of the same rule. Pixels past `render_size` are never
    // shown, so anything drawn there is wasted — and anything drawn *only*
    // there is a picture the user never sees.
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let frame = shoot(&gpu, ShadingMode::Rendered, true);
    let surface = bbox(&frame, is_grey_surface).expect("a shaded cube");
    assert!(
        surface.2 < DRAWN.0 && surface.3 < DRAWN.1,
        "the cube reaches {surface:?}, past the drawn region {DRAWN:?}"
    );
}
