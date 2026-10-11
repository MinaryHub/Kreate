//! The ground grid, drawn across the distances where its lines step a
//! decade: no stray row of bright dashes where the step falls. The lines'
//! width came from the derivative of the coordinate over the spacing, and a
//! 2x2 quad straddling the step took the step into it — one pixel row lit
//! along the whole boundary.

use kreate_render::{Camera, Draw, GpuOptions, ShadingMode, Viewport};

const W: u32 = 1600;
const H: u32 = 1280;

/// Renders the grid from a low, three-quarter view, as a new scene opens.
/// `None` when there is no GPU.
fn grid_picture() -> Option<Vec<[u8; 4]>> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut viewport = Viewport::new(&gpu.device, format, W, H);
    viewport.shading = ShadingMode::Studio;
    let mut draws: Vec<Draw> = Vec::new();
    let mut camera = Camera {
        distance: 9.0,
        ..Camera::default()
    };
    camera.orbit(0.0, 0.25);
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (W * H * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    viewport.render(&mut enc, &view, &draws);
    enc.copy_texture_to_buffer(
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
                bytes_per_row: Some(W * 4),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(enc.finish()));
    readback.map_async(wgpu::MapMode::Read, .., |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let data = readback.slice(..).get_mapped_range().to_vec();
    Some(data.as_chunks::<4>().0.to_vec())
}

#[test]
fn the_grid_has_no_row_of_dashes_where_it_changes_scale() {
    let Some(pixels) = grid_picture() else { return };
    // A dash: a strip two pixels tall — one 2x2 quad's rows — lit as brightly as
    // the nearest lines, far out where the lines are dim, with bare floor in the rows either side of it, three
    // pixels or more along. The grid's own lines fade off at their edges
    // and slant, and where two cross the rows round the crossing are lines
    // too, not floor.
    let lum = |x: u32, y: u32| pixels[(y * W + x) as usize][0] as i32;
    let floor = {
        let mut count = [0usize; 256];
        for y in H / 2..H {
            for x in 0..W {
                count[lum(x, y) as usize] += 1;
            }
        }
        (0..256).max_by_key(|v| count[*v]).unwrap() as i32
    };
    let mut worst = (0, 0, 0);
    for y in (H * 30 / 100..H - 2).filter(|y| y % 2 == 0) {
        let mut run = 0;
        for x in 0..W {
            let (top, bottom) = (lum(x, y), lum(x, y + 1));
            let dash = top > floor + 30
                && bottom > floor + 30
                && lum(x, y - 1) <= floor + 4
                && lum(x, y + 2) <= floor + 4;
            run = if dash { run + 1 } else { 0 };
            if run > worst.0 {
                worst = (run, x, y);
            }
        }
    }
    assert!(
        worst.0 < 3,
        "a dash {} pixels long ending at {}, {}",
        worst.0,
        worst.1,
        worst.2
    );
}
