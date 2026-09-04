//! The cull compute shader against its CPU specification, on a real GPU.
//!
//! This is the test the whole cluster design rests on. The predicates in
//! `cluster.rs` are unit tested and provably safe; the shader is a second
//! implementation of the same arithmetic in a different language, with its own
//! matrix conventions, its own buffer layouts and no way to print anything. The
//! ways it can silently disagree are numerous and all look the same from the
//! outside: geometry that vanishes at certain angles.
//!
//! So the shader is not checked against a screenshot or against a plausible
//! count. It is checked **cluster for cluster** against
//! [`kreate_render::survives`], from several camera positions and under several
//! transforms, and any disagreement names the exact cluster and the exact
//! viewpoint.
//!
//! Skips rather than fails when no adapter is available.

use glam::{DAffine3, DVec3};
use kreate_render::cluster_cull::{ClusterCull, CullDraws, CullView, Pass};
use kreate_render::{Camera, ClusterMesh, Frustum, GpuOptions, RenderMeshCpu, ViewAxis};

const HEIGHT: u32 = 512;
const CULL_PX: f32 = 2.0;

struct Harness {
    gpu: kreate_render::Gpu,
    cull: ClusterCull,
    /// Present but unused: occlusion culling is exercised in its own file, and
    /// a bind group cannot have a hole, so a pyramid has to exist here anyway.
    hiz: kreate_render::HiZ,
    clusters: ClusterMesh,
    cluster_bind: wgpu::BindGroup,
}

impl Harness {
    fn new(mesh: &kreate_mesh::MeshData) -> Option<Self> {
        let opts = GpuOptions::default();
        let instance = kreate_render::Gpu::create_instance(&opts);
        let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

        let cpu = RenderMeshCpu::extract(mesh);
        let clusters = ClusterMesh::bake(&cpu);
        let cull = ClusterCull::new(&gpu.device);
        let hiz = kreate_render::HiZ::new(&gpu.device, HEIGHT, HEIGHT);

        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            &gpu.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("clusters"),
                contents: bytemuck::cast_slice(&clusters.clusters),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let cluster_bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clusters"),
            layout: cull.cluster_layout(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Some(Self {
            gpu,
            cull,
            hiz,
            clusters,
            cluster_bind,
        })
    }

    /// Runs the cull and returns the cluster indices the GPU kept.
    fn visible(&mut self, camera: &Camera, transform: &DAffine3) -> Vec<u32> {
        let counts = [self.clusters.clusters.len() as u32];
        let budget = kreate_render::Budget {
            cull_px: CULL_PX,
            ..kreate_render::Budget::default()
        };
        self.cull.prepare(
            &self.gpu.device,
            &self.gpu.queue,
            &CullView {
                camera,
                aspect: 1.0,
                hiz: &self.hiz,
                occlusion: false,
                render_height: HEIGHT,
                budget: &budget,
            },
            &CullDraws {
                transforms: &[*transform],
                cluster_counts: &counts,
                colors: &[[0.5, 0.5, 0.5, 1.0]],
            },
        );

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.cull
            .dispatch(&mut encoder, Pass::Single, &[&self.cluster_bind]);

        // The arguments and the list are read back together: the count lives in
        // the arguments and the identities in the list, and a test that trusted
        // one without the other could pass while they disagreed.
        let args_readback = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("args readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let list_size = (counts[0].max(1) as u64) * 4;
        let list_readback = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visible readback"),
            size: list_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(self.cull.args_buffer(), 0, &args_readback, 0, 16);
        encoder.copy_buffer_to_buffer(self.cull.visible_buffer(), 0, &list_readback, 0, list_size);
        self.gpu.queue.submit([encoder.finish()]);

        args_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        list_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        let args: [u32; 4] = {
            let data = args_readback.slice(..).get_mapped_range();
            bytemuck::pod_read_unaligned(&data)
        };
        args_readback.unmap();
        let count = args[1] as usize;
        assert_eq!(
            args[0],
            kreate_render::VERTICES_PER_CLUSTER,
            "the vertex count was overwritten; the cull must only touch the instance field"
        );

        let list: Vec<u32> = {
            let data = list_readback.slice(..).get_mapped_range();
            bytemuck::cast_slice::<_, u32>(&data)[..count].to_vec()
        };
        list_readback.unmap();
        list
    }

    /// What the CPU specification keeps, for the same camera and transform.
    fn expected(&self, camera: &Camera, transform: &DAffine3) -> Vec<u32> {
        let frustum = Frustum::from_view_projection(&camera.view_projection(1.0));
        self.clusters
            .clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                kreate_render::survives(c, transform, camera, &frustum, HEIGHT, CULL_PX)
            })
            .map(|(i, _)| i as u32)
            .collect()
    }
}

fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}

fn camera_at(distance: f64, axis: ViewAxis) -> Camera {
    let mut camera = Camera::default().looking_along(axis);
    camera.pivot = DVec3::ZERO;
    camera.distance = distance;
    camera
}

#[test]
fn the_shader_keeps_exactly_the_clusters_the_cpu_keeps() {
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 64, 32);
    let Some(mut harness) = Harness::new(&mesh) else {
        eprintln!("no adapter; skipping");
        return;
    };
    assert!(
        harness.clusters.clusters.len() > 20,
        "a sphere this dense should produce plenty of clusters to disagree about"
    );

    // Several viewpoints, because a single one cannot distinguish a correct cone
    // test from one with the axis sign flipped: from straight on, roughly half
    // the clusters are rejected either way.
    for axis in [
        ViewAxis::PosZ,
        ViewAxis::NegZ,
        ViewAxis::PosX,
        ViewAxis::NegY,
    ] {
        for distance in [3.0, 12.0, 400.0] {
            let camera = camera_at(distance, axis);
            let got = sorted(harness.visible(&camera, &DAffine3::IDENTITY));
            let want = sorted(harness.expected(&camera, &DAffine3::IDENTITY));
            assert_eq!(
                got,
                want,
                "disagreement at {axis:?} distance {distance}: \
                 GPU kept {} clusters, CPU kept {}",
                got.len(),
                want.len()
            );
        }
    }
}

#[test]
fn the_shader_agrees_under_rotation_and_non_uniform_scale() {
    // Where the matrix conventions actually bite. The cone axis needs the
    // inverse transpose and the radius needs the largest scale factor, and both
    // are easy to get subtly wrong in a way that only shows on a transformed
    // object — which is every object in a real scene.
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 48, 24);
    let Some(mut harness) = Harness::new(&mesh) else {
        eprintln!("no adapter; skipping");
        return;
    };

    let transforms = [
        DAffine3::from_axis_angle(DVec3::Y, 1.1),
        DAffine3::from_axis_angle(DVec3::new(1.0, 2.0, 3.0).normalize(), 0.7),
        DAffine3::from_scale(DVec3::new(1.0, 1.0, 4.0)),
        DAffine3::from_scale(DVec3::new(3.0, 0.4, 1.0)) * DAffine3::from_axis_angle(DVec3::X, 0.9),
        DAffine3::from_translation(DVec3::new(2.0, -1.0, 0.5))
            * DAffine3::from_scale(DVec3::splat(0.3)),
    ];

    for (i, transform) in transforms.iter().enumerate() {
        for axis in [ViewAxis::PosZ, ViewAxis::NegX] {
            let camera = camera_at(8.0, axis);
            let got = sorted(harness.visible(&camera, transform));
            let want = sorted(harness.expected(&camera, transform));
            assert_eq!(
                got,
                want,
                "transform {i} at {axis:?}: GPU kept {} clusters, CPU kept {}",
                got.len(),
                want.len()
            );
        }
    }
}

#[test]
fn a_closed_mesh_loses_a_useful_share_of_its_clusters() {
    // Agreement is worthless if both sides keep everything. This is what says
    // the cull is doing work: on a sphere facing the camera, frustum plus
    // backface rejection should discard a large minority.
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 64, 32);
    let Some(mut harness) = Harness::new(&mesh) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let total = harness.clusters.clusters.len();
    let camera = camera_at(4.0, ViewAxis::PosZ);
    let kept = harness.visible(&camera, &DAffine3::IDENTITY).len();
    let share = kept as f64 / total as f64;
    println!("{kept} of {total} clusters kept ({:.0}%)", share * 100.0);
    assert!(
        share < 0.7,
        "the cull kept {:.0}% of clusters, so it is barely rejecting anything",
        share * 100.0
    );
    assert!(kept > 0, "the cull rejected everything");
}

#[test]
fn the_visible_list_holds_no_duplicates() {
    // One thread per cluster and one atomic slot each, so a duplicate would
    // mean two threads took the same slot — the classic sign of a counter that
    // is not actually atomic, or of a dispatch covering a cluster twice.
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 64, 32);
    let Some(mut harness) = Harness::new(&mesh) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let camera = camera_at(5.0, ViewAxis::PosZ);
    let list = harness.visible(&camera, &DAffine3::IDENTITY);
    let unique: std::collections::BTreeSet<u32> = list.iter().copied().collect();
    assert_eq!(
        unique.len(),
        list.len(),
        "the visible list repeated a cluster"
    );
}

#[test]
fn a_camera_looking_away_keeps_nothing() {
    // The frustum test, end to end. Everything behind the eye must go, and this
    // is also the case that would expose the reversed-Z depth planes being the
    // wrong way round: with them swapped, the near plane rejects what is in
    // front instead of what is behind.
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
    let Some(mut harness) = Harness::new(&mesh) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let camera = camera_at(5.0, ViewAxis::PosZ);
    // Put the object far behind the camera rather than turning the camera, so
    // only the transform differs from the control below.
    let behind = DAffine3::from_translation(camera.eye() + camera.forward() * -60.0);
    assert!(harness.visible(&camera, &behind).is_empty());

    let front = DAffine3::IDENTITY;
    assert!(
        !harness.visible(&camera, &front).is_empty(),
        "the control kept nothing, so the assertion above proved nothing"
    );
}

#[test]
fn a_speck_is_culled_by_screen_size() {
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
    let Some(mut harness) = Harness::new(&mesh) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let camera = camera_at(5.0, ViewAxis::PosZ);
    let tiny = DAffine3::from_scale(DVec3::splat(1e-4));
    assert!(harness.visible(&camera, &tiny).is_empty());
}

#[test]
fn a_second_draw_gets_its_own_count_and_its_own_slice() {
    // The per-draw layout, on the GPU. Two instances of one mesh share a
    // cluster buffer and must not share a counter: if they did, the second
    // draw's count would include the first's survivors and it would draw
    // clusters from the wrong slice of the visible list.
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 32, 16);
    let Some(mut harness) = Harness::new(&mesh) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let camera = camera_at(6.0, ViewAxis::PosZ);
    let count = harness.clusters.clusters.len() as u32;

    // One in front, one far behind: the counts must differ, and the second must
    // be zero rather than a copy of the first.
    let visible_transform = DAffine3::IDENTITY;
    let hidden_transform = DAffine3::from_translation(camera.eye() + camera.forward() * -60.0);

    let budget = kreate_render::Budget {
        cull_px: CULL_PX,
        ..kreate_render::Budget::default()
    };
    harness.cull.prepare(
        &harness.gpu.device,
        &harness.gpu.queue,
        &CullView {
            camera: &camera,
            aspect: 1.0,
            hiz: &harness.hiz,
            occlusion: false,
            render_height: HEIGHT,
            budget: &budget,
        },
        &CullDraws {
            transforms: &[visible_transform, hidden_transform],
            cluster_counts: &[count, count],
            colors: &[[0.5, 0.5, 0.5, 1.0]; 2],
        },
    );
    let slots: Vec<kreate_render::CullSlot> = harness.cull.slots(Pass::Single).to_vec();
    assert_eq!(slots[1].visible_base, count, "the slices overlap");

    let mut encoder = harness
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    harness.cull.dispatch(
        &mut encoder,
        Pass::Single,
        &[&harness.cluster_bind, &harness.cluster_bind],
    );

    let readback = harness.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("args readback"),
        size: slots[1].args_offset as u64 + 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        harness.cull.args_buffer(),
        0,
        &readback,
        0,
        slots[1].args_offset as u64 + 16,
    );
    harness.gpu.queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    harness
        .gpu
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = readback.slice(..).get_mapped_range();
    let first: [u32; 4] = bytemuck::pod_read_unaligned(&data[..16]);
    let second: [u32; 4] =
        bytemuck::pod_read_unaligned(&data[slots[1].args_offset as usize..][..16]);
    drop(data);
    readback.unmap();

    assert!(first[1] > 0, "the visible instance kept nothing");
    assert_eq!(second[1], 0, "the hidden instance kept clusters");
}
