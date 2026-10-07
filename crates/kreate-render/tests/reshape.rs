//! A reshaped mesh on the GPU has to be the one a fresh extract would draw.
//!
//! The positions exactly, read back from the buffer the rasteriser reads; the
//! normals to the shading's precision. A sculpted sphere keeps its smoothing
//! groups, so the fresh extract splits it the same way and can be compared
//! render vertex by render vertex.

use kreate_mesh::MeshData;
use kreate_render::{Deformer, GpuOptions, RenderMeshCpu, Reshape};

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// One device for the binary; see `tests/deform.rs` for why.
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

fn read_back(gpu: &Gpu, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reshape readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    data
}

/// A bump pushed out of one side of the sphere, rounded on top as a brush's
/// falloff makes it. A pointed one would crease, and a crease is the known
/// limit: the fresh extract would split it and the reshape would not.
fn sculpted(m: &MeshData) -> MeshData {
    let mut out = m.clone();
    for p in out.positions_mut() {
        let t = (1.0 - p.distance_squared(glam::Vec3::X)).max(0.0);
        *p *= 1.0 + 0.3 * t * t;
    }
    out
}

#[test]
fn a_reshaped_mesh_is_what_a_fresh_extract_would_draw() {
    let gpu = gpu_or_skip!(gpu());
    let before = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
    let after = sculpted(&before);
    let cpu = RenderMeshCpu::extract(&before);
    let viewport =
        kreate_render::Viewport::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb, 64, 64);
    let mesh = kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let deformer = Deformer::new(&gpu.device);
    let mut reshape = Reshape::new(&gpu.device, &deformer, &mesh, &cpu).expect("a mesh");

    assert!(
        !reshape.write(&gpu.queue, &mesh, &after.positions()[..10]),
        "too few positions is refused"
    );
    assert!(reshape.write(&gpu.queue, &mesh, after.positions()));
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    reshape.dispatch(&mut encoder, &deformer);
    gpu.queue.submit([encoder.finish()]);

    // Compared per mesh vertex, not per render vertex: the fresh extract
    // de-duplicates by normal, so its split can differ by a vertex or two.
    let want = RenderMeshCpu::extract(&after);
    let mut want_normals: Vec<Vec<glam::Vec3>> = vec![Vec::new(); after.num_verts() as usize];
    for (r, v) in want.render_to_vert.iter().enumerate() {
        want_normals[*v as usize].push(kreate_render::mesh_gpu::oct_decode(want.normals[r]));
    }
    let n = cpu.positions.len();
    let positions = read_back(&gpu, mesh.positions(), (n * 12) as u64);
    let normals = read_back(&gpu, mesh.normals(), (n * 4) as u64);
    let mut moved = 0;
    let mut worst = 0.0f32;
    for r in 0..n {
        let v = cpu.render_to_vert[r] as usize;
        let got: [f32; 3] = bytemuck::pod_read_unaligned(&positions[r * 12..r * 12 + 12]);
        assert_eq!(got, after.positions()[v].to_array(), "render vertex {r}");
        if got != cpu.positions[r].0 {
            moved += 1;
        }
        let g = kreate_render::mesh_gpu::oct_decode(u32::from_le_bytes(
            normals[r * 4..r * 4 + 4].try_into().unwrap(),
        ));
        let best = want_normals[v]
            .iter()
            .map(|w| g.angle_between(*w).to_degrees())
            .fold(f32::MAX, f32::min);
        worst = worst.max(best);
    }
    assert!(moved > n / 10, "the bump has to move a good part: {moved} of {n}");
    assert!(worst < 3.0, "normals off by {worst}°");
}
