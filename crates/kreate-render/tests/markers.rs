//! A marker behind geometry is behind it.
//!
//! The whole reason lamp and camera glyphs moved from the interface's painter
//! into the renderer: painted over the frame, a lamp's circle sat in front of
//! the wall it was behind. Drawn in the viewport's own pass against its depth,
//! it does not. Nothing but a picture can check that, so this takes one.

use kreate_render::GpuOptions;
use kreate_render::{Camera, Draw, MarkerShape, OverlayLine, OverlayPoint, RenderMeshCpu, ShadingMode, Viewport};

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

const SIZE: u32 = 96;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// A colour nothing else in the frame has: pure magenta.
const INK: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// A cube at the origin, seen from the default camera, with the given markers.
fn shoot(gpu: &Gpu, lines: &[OverlayLine], points: &[OverlayPoint]) -> (Vec<[u8; 4]>, Camera) {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Studio;
    viewport.show_grid = false;
    viewport.show_wireframe = false;

    let mesh = kreate_mesh::primitives::cube(1.0);
    let cpu = RenderMeshCpu::extract(&mesh);
    let uploaded =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let camera = Camera {
        distance: 4.0,
        ..Camera::default()
    };
    let mut draws = vec![Draw::new(&uploaded, glam::DAffine3::IDENTITY)];
    viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &[]);
    viewport.set_markers(&gpu.device, &gpu.queue, camera.eye(), lines, points);
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
        for x in 0..SIZE as usize {
            let at = base + x * 4;
            out.push([data[at], data[at + 1], data[at + 2], data[at + 3]]);
        }
    }
    (out, camera)
}

fn magenta(frame: &[[u8; 4]]) -> usize {
    frame
        .iter()
        .filter(|p| p[0] > 120 && p[2] > 120 && p[1] < 80)
        .count()
}

fn ring_at(position: glam::DVec3) -> OverlayPoint {
    OverlayPoint {
        position,
        color: INK,
        radius_px: 7.0,
        shape: MarkerShape::Ring,
        dot: true,
    }
}

#[test]
fn a_ring_in_front_of_the_cube_is_drawn_and_one_behind_it_is_not() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // Along the view ray through the cube's centre: 1.5 toward the camera is
    // clear of the unit cube, 1.5 away is hidden by it.
    let camera = Camera {
        distance: 4.0,
        ..Camera::default()
    };
    let toward = (camera.eye() - glam::DVec3::ZERO).normalize();
    let (front, _) = shoot(&gpu, &[], &[ring_at(toward * 1.5)]);
    let (behind, _) = shoot(&gpu, &[], &[ring_at(-toward * 1.5)]);
    let (none, _) = shoot(&gpu, &[], &[]);

    assert!(
        magenta(&front) > 15,
        "a ring in front of the cube drew {} pixels",
        magenta(&front)
    );
    assert_eq!(
        magenta(&behind),
        magenta(&none),
        "a ring behind the cube drew through it: {} pixels",
        magenta(&behind)
    );
}

#[test]
fn a_line_is_hidden_only_where_geometry_is_in_front_of_it() {
    // A line passing behind the cube: its ends are visible either side and
    // its middle is not. The count says the ends drew; a second frame with
    // the cube out of the way says how much the middle would have added.
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let camera = Camera {
        distance: 4.0,
        ..Camera::default()
    };
    let toward = (camera.eye() - glam::DVec3::ZERO).normalize();
    // A direction across the view, so the line spans the picture.
    let across = toward.cross(glam::DVec3::Z).normalize_or_zero();
    let across = if across.length() < 0.5 {
        toward.cross(glam::DVec3::X).normalize()
    } else {
        across
    };
    let behind = -toward * 1.5;
    let line = |at: glam::DVec3| OverlayLine {
        a: at - across * 2.0,
        b: at + across * 2.0,
        color: INK,
        width_px: 2.0,
        dashed: false,
    };
    let (hidden_middle, _) = shoot(&gpu, &[line(behind)], &[]);
    let (clear, _) = shoot(&gpu, &[line(toward * 1.5)], &[]);
    let ends = magenta(&hidden_middle);
    let whole = magenta(&clear);
    assert!(ends > 10, "the visible ends drew only {ends} pixels");
    assert!(
        ends * 10 < whole * 8,
        "the line behind the cube ({ends} px) is not shorter than the one in \
         front ({whole} px); the cube did not hide its middle"
    );
}

#[test]
fn markers_survive_a_frame_with_more_of_them_than_the_last() {
    // The storage buffers grow on demand. Sixteen fit at first; a frame with
    // forty must not drop the ones past the end, and the frame after must
    // still draw.
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let camera = Camera {
        distance: 4.0,
        ..Camera::default()
    };
    let toward = (camera.eye() - glam::DVec3::ZERO).normalize();
    let across = toward.cross(glam::DVec3::Z).normalize_or_zero();
    let many: Vec<OverlayPoint> = (0..40)
        .map(|i| ring_at(toward * 1.5 + across * (i as f64 * 0.08 - 1.6)))
        .collect();
    let (frame, _) = shoot(&gpu, &[], &many);
    let (one, _) = shoot(&gpu, &[], &many[..1]);
    assert!(
        magenta(&frame) > magenta(&one) * 4,
        "forty rings drew {} pixels against one ring's {}",
        magenta(&frame),
        magenta(&one)
    );
}
