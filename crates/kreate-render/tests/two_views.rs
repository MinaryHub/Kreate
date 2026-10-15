//! Two viewports drawing one scene.
//!
//! The application's extra 3D viewports each have a renderer of their own
//! and draw the meshes the first one uploaded, through cameras of their own.
//! That leans on two things a single viewport never tests: a mesh uploaded
//! against one renderer's layouts binds in another's pipelines, and one
//! renderer preparing the draws for its camera leaves the other's picture
//! alone.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, ShadingMode, Viewport};

const SIZE: u32 = 128;
const BYTES_PER_ROW: u32 = SIZE * 4;

fn target(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("two views target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn read(gpu: &kreate_render::Gpu, texture: &wgpu::Texture) -> Vec<[u8; 4]> {
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (BYTES_PER_ROW * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
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
    buffer.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("callback").expect("map");
    let data = buffer.slice(..).get_mapped_range();
    let pixels = data.as_chunks::<4>().0.to_vec();
    drop(data);
    buffer.unmap();
    pixels
}

/// How many pixels are not the background, judged by the corner.
fn covered(pixels: &[[u8; 4]]) -> usize {
    let bg = pixels[0];
    pixels
        .iter()
        .filter(|p| (0..3).any(|i| (p[i] as i32 - bg[i] as i32).abs() > 8))
        .count()
}

#[test]
fn a_second_viewport_draws_the_first_ones_meshes_through_its_own_camera() {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let Ok(gpu) = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)) else {
        eprintln!("no adapter; skipped");
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut first = Viewport::new(&gpu.device, format, SIZE, SIZE);
    let mut second = Viewport::new(&gpu.device, format, SIZE, SIZE);
    for v in [&mut first, &mut second] {
        v.shading = ShadingMode::Studio;
        v.show_grid = false;
    }
    // A long bar along X: from the front it is wide, from the side a square.
    let mut bar = kreate_mesh::primitives::cube(1.0);
    for p in bar.positions_mut() {
        p.x *= 4.0;
    }
    let cpu = RenderMeshCpu::extract(&bar);
    // Uploaded against the first one's layouts, as the application does.
    let mesh = GpuMesh::upload(
        &gpu.device,
        &first.layouts,
        &cpu,
        Vec4::new(0.8, 0.8, 0.8, 1.0),
    );
    let mut draws = vec![Draw::new(&mesh, DAffine3::IDENTITY)];
    let base = Camera {
        pivot: DVec3::ZERO,
        distance: 12.0,
        ..Camera::default()
    };
    let front = base.looking_along(kreate_render::camera::ViewAxis::NegY);
    let side = base.looking_along(kreate_render::camera::ViewAxis::PosX);

    let (a_tex, a_view) = target(&gpu.device);
    let (b_tex, b_view) = target(&gpu.device);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    first.prepare(&gpu.device, &gpu.queue, &front, &mut draws);
    first.render(&mut encoder, &a_view, &draws);
    second.prepare(&gpu.device, &gpu.queue, &side, &mut draws);
    second.render(&mut encoder, &b_view, &draws);
    gpu.queue.submit(Some(encoder.finish()));
    let (a, b) = (covered(&read(&gpu, &a_tex)), covered(&read(&gpu, &b_tex)));
    assert!(b > 50, "the second view drew the bar: {b} pixels");
    // Seen from the front the bar is four times as wide as from its end:
    // each picture is its own camera's, not the other's drawn twice.
    assert!(a as f64 > b as f64 * 2.5, "front {a} against side {b}");
}
