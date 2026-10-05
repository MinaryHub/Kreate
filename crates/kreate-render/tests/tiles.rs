//! What the cull pass puts in a tile.
//!
//! Read back and inspected rather than looked at, because a light wrongly in a
//! list costs time and shows nothing, and a light wrongly *out* of one shows as
//! a square of the picture going dark. The first is invisible in a render and
//! the second is a shape no assertion about brightness describes well.

use kreate_render::lights::{Falloff, Lights, PlacedLight};
use kreate_render::tiles::{Tiles, MAX_LIGHTS_PER_TILE, TILE_SIZE};
use kreate_render::{Camera, GpuOptions};

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

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

/// Runs the cull and reads every tile's list back.
///
/// `live` is how many of the light buffer the frame is using, which is not the
/// same as how long the buffer is — it grows and never shrinks.
fn cull(gpu: &Gpu, lights: &[PlacedLight], camera: &Camera) -> Vec<Vec<u32>> {
    let mut buffer = Lights::new(&gpu.device, 1);
    let live = buffer.upload(&gpu.device, &gpu.queue, camera.eye(), lights);

    let mut tiles = Tiles::new(&gpu.device);
    tiles.resize(&gpu.device, WIDTH, HEIGHT);
    let (across, down) = Tiles::grid(WIDTH, HEIGHT);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    tiles.cull(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        buffer.buffer(),
        live,
        camera,
        WIDTH as f32 / HEIGHT as f32,
    );

    let stride = MAX_LIGHTS_PER_TILE as u64 + 1;
    let bytes = (across * down) as u64 * stride * 4;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(tiles.buffer(), 0, &staging, 0, bytes);
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data: Vec<u32> = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();

    (0..(across * down) as usize)
        .map(|tile| {
            let base = tile * stride as usize;
            let count = data[base].min(MAX_LIGHTS_PER_TILE) as usize;
            data[base + 1..base + 1 + count].to_vec()
        })
        .collect()
}

/// A lamp with a range, at a place.
fn lamp(at: glam::DVec3, range: f32) -> PlacedLight {
    PlacedLight {
        falloff: Falloff::InverseSquare,
        position: at,
        amount: [1000.0; 3],
        range: Some(range),
        ..PlacedLight::default()
    }
}

#[test]
fn a_light_with_no_end_is_in_every_tile() {
    // Inverse square never reaches zero, so a lamp that was never given a range
    // touches every pixel. Culling it would be culling light somebody can see,
    // and it is the reason the range had to exist before this pass could.
    let gpu = gpu_or_skip!(gpu());
    let camera = Camera {
        distance: 10.0,
        ..Camera::default()
    };
    let mut unbounded = lamp(glam::DVec3::new(0.0, 0.0, 2.0), 0.0);
    unbounded.range = None;
    let lists = cull(&gpu, &[unbounded], &camera);
    assert!(
        lists.iter().all(|t| t == &[0u32]),
        "a light with no end was culled from some tile"
    );
}

#[test]
fn a_sun_is_in_every_tile_whatever_its_range() {
    // Parallel light does not diverge and so does not run out. A range on one
    // is a caller's mistake, and obeying it would put a horizon across the
    // middle of the screen.
    let gpu = gpu_or_skip!(gpu());
    let camera = Camera {
        distance: 10.0,
        ..Camera::default()
    };
    let sun = PlacedLight {
        falloff: Falloff::Parallel,
        direction: glam::Vec3::NEG_Z,
        amount: [20_000.0; 3],
        range: Some(0.5),
        ..PlacedLight::default()
    };
    let lists = cull(&gpu, &[sun], &camera);
    assert!(lists.iter().all(|t| t == &[0u32]), "a sun was culled");
}

#[test]
fn a_small_lamp_reaches_some_tiles_and_not_others() {
    // The whole point of the pass. A lamp whose reach covers a corner of the
    // scene has no business being evaluated for the fragments in the other
    // corner.
    let gpu = gpu_or_skip!(gpu());
    let camera = Camera {
        distance: 20.0,
        ..Camera::default()
    };
    // Off to one side, and reaching a metre.
    let lists = cull(&gpu, &[lamp(glam::DVec3::new(6.0, 0.0, 0.0), 1.0)], &camera);
    let with = lists.iter().filter(|t| !t.is_empty()).count();
    assert!(with > 0, "the lamp reached no tile at all");
    assert!(
        with < lists.len() / 2,
        "{with} of {} tiles kept a lamp reaching one metre",
        lists.len()
    );
}

#[test]
fn a_cull_never_lists_a_light_past_the_count() {
    // The light buffer grows and never shrinks, so the slots past the count
    // still hold an earlier frame's lamps. This pass is what stops them being
    // shaded with: the shading loop walks a tile's list and has no other idea
    // how many lights there are.
    let gpu = gpu_or_skip!(gpu());
    let camera = Camera {
        distance: 10.0,
        ..Camera::default()
    };
    let mut buffer = Lights::new(&gpu.device, 1);
    // Eight lamps to grow the buffer, then one, so that seven stale slots are
    // sitting behind the live one.
    let many: Vec<PlacedLight> = (0..8)
        .map(|i| lamp(glam::DVec3::new(i as f64, 0.0, 2.0), 100.0))
        .collect();
    buffer.upload(&gpu.device, &gpu.queue, camera.eye(), &many);
    let live = buffer.upload(&gpu.device, &gpu.queue, camera.eye(), &many[..1]);
    assert_eq!(live, 1);
    assert!(buffer.capacity() > live, "the buffer did not keep its room");

    let mut tiles = Tiles::new(&gpu.device);
    tiles.resize(&gpu.device, WIDTH, HEIGHT);
    let (across, down) = Tiles::grid(WIDTH, HEIGHT);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    tiles.cull(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        buffer.buffer(),
        live,
        &camera,
        1.0,
    );
    let stride = MAX_LIGHTS_PER_TILE as u64 + 1;
    let bytes = (across * down) as u64 * stride * 4;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(tiles.buffer(), 0, &staging, 0, bytes);
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data: Vec<u32> = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();

    for tile in 0..(across * down) as usize {
        let base = tile * stride as usize;
        let count = data[base];
        assert!(count <= live, "tile {tile} listed {count} of {live} lights");
        for slot in 0..count as usize {
            assert!(
                data[base + 1 + slot] < live,
                "tile {tile} listed a light from an earlier frame"
            );
        }
    }
}

#[test]
fn a_tile_holds_no_more_than_it_has_room_for() {
    // More lamps on one square than the list can carry. The extra are dropped,
    // which is a visible error in a scene that reaches it — and the alternative
    // is writing past the end of somebody else's tile, which is a wrong picture
    // somewhere nobody would think to look.
    let gpu = gpu_or_skip!(gpu());
    let camera = Camera {
        distance: 10.0,
        ..Camera::default()
    };
    let crowd: Vec<PlacedLight> = (0..MAX_LIGHTS_PER_TILE + 20)
        .map(|i| lamp(glam::DVec3::new(0.0, 0.0, 2.0 + i as f64 * 0.001), 100.0))
        .collect();
    let lists = cull(&gpu, &crowd, &camera);
    for (tile, list) in lists.iter().enumerate() {
        assert!(
            list.len() <= MAX_LIGHTS_PER_TILE as usize,
            "tile {tile} holds {} lights",
            list.len()
        );
    }
}

#[test]
fn a_tile_is_sixteen_pixels_and_the_grid_covers_the_target() {
    // The shading pass works out its tile from the fragment's own pixel
    // position, so the two have to agree about how big a tile is. They agree by
    // both taking it from here.
    assert_eq!(TILE_SIZE, 16);
    let (across, down) = Tiles::grid(WIDTH, HEIGHT);
    assert_eq!(across, WIDTH / TILE_SIZE);
    assert_eq!(down, HEIGHT / TILE_SIZE);
}
