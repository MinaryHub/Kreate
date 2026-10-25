//! The material tab's ball, drawn offscreen and saved for a look.

use kreate::material_preview::{MaterialPreview, PreviewOf, SIZE};

/// Draws a document material and a shelf of physical ones and tiles them
/// into `target/material-preview.png`.
#[test]
#[ignore = "needs a GPU; writes a picture to look at"]
fn shot_material_preview() {
    let opts = kreate_render::GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let Ok(gpu) = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)) else {
        eprintln!("no GPU");
        return;
    };
    let mut renderer = egui_wgpu::Renderer::new(
        &gpu.device,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        egui_wgpu::RendererOptions::default(),
    );
    let mut doc = kreate_scene::Document::new();
    let red = doc.add_material(kreate_scene::document::Material {
        name: "Red".into(),
        base_color: kreate_math::SceneRgb::new(0.6, 0.05, 0.04),
        roughness: 0.3,
        ..Default::default()
    });
    let shown = [
        PreviewOf::Material(red),
        PreviewOf::Physical("oak"),
        PreviewOf::Physical("walnut"),
        PreviewOf::Physical("marble_white"),
        PreviewOf::Physical("granite"),
        PreviewOf::Physical("brushed_steel"),
        PreviewOf::Physical("gold"),
        PreviewOf::Physical("glass_frosted"),
    ];
    let mut preview = MaterialPreview::new(&gpu.device, &mut renderer);
    let across = 4;
    let mut sheet = image::RgbaImage::new(SIZE * across, SIZE * 2);
    for (i, of) in shown.into_iter().enumerate() {
        // Until the running mean is full, as the application draws it.
        for _ in 0..32 {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let drew = preview.draw(&gpu.device, &gpu.queue, &mut encoder, &doc, of, false);
            gpu.queue.submit([encoder.finish()]);
            if !drew {
                break;
            }
        }
        let row = SIZE * 4;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (row * SIZE) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            preview.texture().as_image_copy(),
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
        let (ox, oy) = ((i as u32 % across) * SIZE, (i as u32 / across) * SIZE);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let at = (y * row + x * 4) as usize;
                sheet.put_pixel(
                    ox + x,
                    oy + y,
                    image::Rgba([data[at], data[at + 1], data[at + 2], 255]),
                );
            }
        }
    }
    let out =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/material-preview.png");
    sheet.save(&out).expect("saved");
    eprintln!("{}", out.display());
}
