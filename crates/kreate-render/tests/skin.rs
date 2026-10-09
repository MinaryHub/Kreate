//! The GPU's skinned mesh has to be the CPU's.
//!
//! The CPU's skin is what picking, snapping, framing and the path tracer see;
//! the GPU's is what is drawn. If they disagree the user clicks on something
//! that is not where it is drawn. So the claim is equality, to float
//! precision, read back from the buffers the rasteriser reads.

use kreate_math::{DAffine3, DQuat, DVec3};
use kreate_mesh::attribute::{AttrData, Domain};
use kreate_mesh::MeshData;
use kreate_render::{bone_rows, Deformer, GpuOptions, Influences, RenderMeshCpu, Skin};
use kreate_scene::armature::{Armature, Bone, Pose};
use kreate_scene::skin::{automatic_weights, influences, skin, skin_matrices};

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
        label: Some("skin readback"),
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

/// Three bones in a chain, so a vertex can be weighed on by more than two and
/// a parent's turn reaches its grandchild.
fn arm() -> Armature {
    let mut a = Armature::new("Arm");
    a.bones[0].name = "Upper".into();
    for (name, parent, z) in [("Lower", 0, 2.0), ("Hand", 1, 3.0)] {
        a.add_bone(Bone {
            name: name.into(),
            parent: Some(parent),
            head: DVec3::new(0.0, 0.0, z - 1.0),
            tail: DVec3::new(0.0, 0.0, z),
            connected: true,
            ..Default::default()
        });
    }
    a
}

/// A sphere stretched into a column three units tall, with enough rings that
/// every bone has vertices of its own and the joints have vertices shared.
fn column(a: &Armature, mesh_to_armature: DAffine3) -> MeshData {
    let mut m = kreate_mesh::primitives::uv_sphere(0.5, 16, 24);
    for p in m.positions_mut() {
        p.x *= 0.4;
        p.y *= 0.4;
        p.z = (p.z + 0.5) * 3.0;
    }
    for (name, mut w) in automatic_weights(&m, a, mesh_to_armature) {
        // Weights that do not sum to one, as painted ones often do not: the
        // pass divides by their sum, and with normalised ones it would not
        // have to.
        if name.ends_with("Lower") {
            w.iter_mut().for_each(|x| *x *= 1.7);
        }
        m.attributes_mut()
            .insert(&name, Domain::Point, AttrData::F32(std::sync::Arc::new(w)));
    }
    m
}

fn bent() -> Pose {
    let mut pose = Pose::default();
    pose.entry("Upper").transform.rotation = DQuat::from_rotation_y(0.3);
    pose.entry("Lower").transform.rotation = DQuat::from_rotation_x(1.2);
    pose.entry("Hand").transform.rotation = DQuat::from_rotation_z(-0.8);
    pose.entry("Hand").transform.scale = DVec3::new(1.3, 0.7, 1.1);
    pose
}

/// Skins on the GPU, once per pose in turn, and reads back the positions and
/// normals after the last one.
fn on_gpu(
    gpu: &Gpu,
    mesh: &MeshData,
    a: &Armature,
    poses: &[Vec<DAffine3>],
) -> (Vec<glam::Vec3>, Vec<u32>) {
    on_gpu_with(gpu, mesh, a, poses, false)
}

fn on_gpu_with(
    gpu: &Gpu,
    mesh: &MeshData,
    a: &Armature,
    poses: &[Vec<DAffine3>],
    preserve_volume: bool,
) -> (Vec<glam::Vec3>, Vec<u32>) {
    let cpu = RenderMeshCpu::extract(mesh);
    let viewport =
        kreate_render::Viewport::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb, 64, 64);
    let gpu_mesh =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let deformer = Deformer::new(&gpu.device);
    let (offsets, bones, weights) = influences(mesh, a).expect("weighted");
    let table = Influences {
        offsets,
        bones,
        weights,
    };
    let skinner = Skin::new(
        &gpu.device,
        &deformer,
        &gpu_mesh,
        &cpu,
        &table,
        a.bones.len() as u32,
    )
    .expect("a weighted mesh");
    skinner.set_preserve_volume(&gpu.queue, preserve_volume);
    for matrices in poses {
        let rows: Vec<_> = matrices
            .iter()
            .map(|m| bone_rows(glam::Affine3A::from_mat4(glam::DMat4::from(*m).as_mat4())))
            .collect();
        assert!(skinner.set_bones(&gpu.queue, &rows));
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        skinner.dispatch(&mut encoder, &deformer);
        gpu.queue.submit([encoder.finish()]);
    }
    let n = cpu.positions.len();
    let positions = read_back(gpu, gpu_mesh.positions(), (n * 12) as u64)
        .as_chunks::<12>()
        .0
        .iter()
        .map(|b| {
            let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            glam::Vec3::new(f(0), f(4), f(8))
        })
        .collect();
    let normals = read_back(gpu, gpu_mesh.normals(), (n * 4) as u64)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    (positions, normals)
}

#[test]
fn the_gpu_skins_every_vertex_where_the_cpu_does() {
    let gpu = gpu_or_skip!(gpu());
    // A mesh placed awkwardly against its armature, so the matrices carry a
    // real change of space and not the identity.
    let mesh_to_armature = DAffine3::from_scale_rotation_translation(
        DVec3::new(1.0, 1.0, 1.0),
        DQuat::from_rotation_z(0.6),
        DVec3::new(0.2, -0.1, 0.0),
    );
    let a = arm();
    let m = column(&a, mesh_to_armature);
    let matrices = skin_matrices(&a, &bent(), mesh_to_armature);
    let want = skin(&m, &a, &matrices).expect("weighted");
    // An earlier pose first: each pose is computed from the rest copy, never
    // from whatever the last one left in the buffer.
    let earlier = skin_matrices(&a, &Pose::default(), mesh_to_armature);
    let (got, _) = on_gpu(&gpu, &m, &a, &[earlier, matrices]);
    let cpu = RenderMeshCpu::extract(&m);
    let mut worst = 0.0f32;
    let mut moved = 0;
    for (r, p) in got.iter().enumerate() {
        let v = cpu.render_to_vert[r] as usize;
        worst = worst.max((*p - want.positions()[v]).length());
        if (want.positions()[v] - m.positions()[v]).length() > 0.1 {
            moved += 1;
        }
    }
    assert!(
        moved > got.len() / 3,
        "the pose has to move most of the mesh: {moved}"
    );
    assert!(worst < 1e-4, "GPU and CPU disagree by {worst}");
}

#[test]
fn the_normals_are_the_skinned_surface_s() {
    let gpu = gpu_or_skip!(gpu());
    let a = arm();
    let m = column(&a, DAffine3::IDENTITY);
    let matrices = skin_matrices(&a, &bent(), DAffine3::IDENTITY);
    let skinned = skin(&m, &a, &matrices).expect("weighted");
    let (_, got) = on_gpu(&gpu, &m, &a, &[matrices]);
    let rest = RenderMeshCpu::extract(&m);
    // The reference: the skinned positions, shaded through the rest shape's
    // smooth groups — Newell's normal per face, summed by area over each
    // render vertex's group. That is the claim: the surface's normals under
    // the split it was uploaded with. A pose that would change the split is
    // the known limit the module documents, as a drag's is.
    let at = |r: u32| skinned.positions()[rest.render_to_vert[r as usize] as usize];
    let faces: Vec<glam::Vec3> = rest
        .face_offsets
        .windows(2)
        .map(|w| {
            let ring: Vec<glam::Vec3> = rest.corner_render[w[0] as usize..w[1] as usize]
                .iter()
                .map(|r| at(*r))
                .collect();
            let mut n = glam::Vec3::ZERO;
            for k in 0..ring.len() {
                let (p, c) = (ring[(k + ring.len() - 1) % ring.len()], ring[k]);
                n += glam::Vec3::new(
                    (p.y - c.y) * (p.z + c.z),
                    (p.z - c.z) * (p.x + c.x),
                    (p.x - c.x) * (p.y + c.y),
                );
            }
            n
        })
        .collect();
    let mut worst = 0.0f32;
    let mut stale = 1.0f32;
    for (r, n) in got.iter().enumerate() {
        let sum: glam::Vec3 = rest.group_faces
            [rest.group_offsets[r] as usize..rest.group_offsets[r + 1] as usize]
            .iter()
            .map(|f| faces[*f as usize])
            .sum();
        let want = sum.normalize();
        let g = kreate_render::mesh_gpu::oct_decode(*n);
        worst = worst.max(g.dot(want).clamp(-1.0, 1.0).acos().to_degrees());
        stale = stale.min(kreate_render::mesh_gpu::oct_decode(rest.normals[r]).dot(want));
    }
    // And they really did turn: the rest normals would not pass this.
    assert!(stale < 0.5, "the pose has to turn some normals: {stale}");
    assert!(worst < 1.0, "a skinned normal is {worst} degrees off");
}

#[test]
fn dual_quaternions_on_the_gpu_are_the_cpu_s() {
    let gpu = gpu_or_skip!(gpu());
    let mesh_to_armature = DAffine3::from_scale_rotation_translation(
        DVec3::ONE,
        DQuat::from_rotation_z(0.6),
        DVec3::new(0.2, -0.1, 0.0),
    );
    let a = arm();
    let m = column(&a, mesh_to_armature);
    // Bent past half a turn from the root to the hand, so some vertex's
    // bones sit in opposite hemispheres and the sign has to be chosen.
    let mut pose = bent();
    pose.entry("Lower").transform.rotation = DQuat::from_rotation_x(2.0);
    pose.entry("Hand").transform.rotation = DQuat::from_rotation_x(1.8);
    let matrices = skin_matrices(&a, &pose, mesh_to_armature);
    let want = kreate_scene::skin::skin_with(&m, &a, &matrices, true).expect("weighted");
    let linear = skin(&m, &a, &matrices).expect("weighted");
    let (got, _) = on_gpu_with(&gpu, &m, &a, &[matrices], true);
    let cpu = RenderMeshCpu::extract(&m);
    let mut worst = 0.0f32;
    let mut differs = 0.0f32;
    for (r, p) in got.iter().enumerate() {
        let v = cpu.render_to_vert[r] as usize;
        worst = worst.max((*p - want.positions()[v]).length());
        differs = differs.max((want.positions()[v] - linear.positions()[v]).length());
    }
    // The two blends really differ here, so agreeing with one is not
    // agreeing with the other.
    assert!(differs > 0.01, "{differs}");
    assert!(worst < 1e-4, "GPU and CPU disagree by {worst}");
}

#[test]
fn a_pose_of_the_wrong_size_is_refused() {
    let gpu = gpu_or_skip!(gpu());
    let a = arm();
    let m = column(&a, DAffine3::IDENTITY);
    let cpu = RenderMeshCpu::extract(&m);
    let viewport =
        kreate_render::Viewport::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb, 64, 64);
    let gpu_mesh =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cpu, glam::Vec4::ONE);
    let deformer = Deformer::new(&gpu.device);
    let (offsets, bones, weights) = influences(&m, &a).unwrap();
    let skinner = Skin::new(
        &gpu.device,
        &deformer,
        &gpu_mesh,
        &cpu,
        &Influences {
            offsets,
            bones,
            weights,
        },
        3,
    )
    .unwrap();
    assert!(!skinner.set_bones(&gpu.queue, &[bone_rows(glam::Affine3A::IDENTITY); 2]));
    assert!(skinner.set_bones(&gpu.queue, &[bone_rows(glam::Affine3A::IDENTITY); 3]));
}
