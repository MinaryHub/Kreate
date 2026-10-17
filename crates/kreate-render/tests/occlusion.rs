//! Occlusion culling: the shader against its CPU specification, and the safety
//! property against the truth.
//!
//! This is the deepest of the three cluster rejections and the only one whose
//! failure is silent and destructive. Frustum and cone rejection are provable
//! from the geometry alone; occlusion depends on a lossy summary of a depth
//! buffer, a projection, a mip level choice and a texel address — five places to
//! be subtly wrong, and being wrong in one direction punches holes in the model.
//!
//! So two different kinds of check:
//!
//! - **Agreement**, cluster for cluster, with [`kreate_render::is_occluded`].
//!   Catches a transposed matrix, a flipped V coordinate, an off-by-one mip.
//! - **Soundness**, against the depth buffer itself rather than against the
//!   reference: nothing the shader rejects may have been visible. This holds
//!   even if the reference is wrong, which is what makes it worth writing
//!   separately.
//!
//! Skips rather than fails when no adapter is available.

use glam::{DAffine3, DVec3, Vec4};
use kreate_render::cluster_cull::{ClusterCull, CullDraws, CullView, Pass};
use kreate_render::hiz::{self, HiZ};
use kreate_render::{
    Budget, Camera, ClusterMesh, Draw, GpuMesh, GpuOptions, RenderMeshCpu, ViewAxis, Viewport,
};

const SIZE: u32 = 256;
const CULL_PX: f32 = 2.0;

/// A scene where one object hides another.
///
/// A large near sphere and a small far one directly behind it, so a good part of
/// the far one's clusters are genuinely occluded and the rest are not. A scene
/// where everything is occluded, or nothing is, cannot distinguish a working
/// test from a stuck one.
struct Scene {
    gpu: kreate_render::Gpu,
    viewport: Viewport,
    /// The occluder and the occluded, in that draw order.
    meshes: Vec<GpuMesh>,
    transforms: Vec<DAffine3>,
    camera: Camera,
    /// Clusters of the *hidden* mesh, which is the one under test.
    hidden: ClusterMesh,
    hidden_index: usize,
}

impl Scene {
    fn new() -> Option<Self> {
        let opts = GpuOptions::default();
        let instance = kreate_render::Gpu::create_instance(&opts);
        let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut viewport = Viewport::new(&gpu.device, format, SIZE, SIZE);
        viewport.show_grid = false;
        viewport.show_ground_shadow = false;
        viewport.show_wireframe = false;
        viewport.show_elements = false;

        let occluder = kreate_mesh::primitives::uv_sphere(1.0, 48, 24);
        let hidden = kreate_mesh::primitives::uv_sphere(1.0, 48, 24);
        let hidden_cpu = RenderMeshCpu::extract(&hidden).with_clusters();
        let meshes = vec![
            GpuMesh::upload(
                &gpu.device,
                &viewport.layouts,
                &RenderMeshCpu::extract(&occluder).with_clusters(),
                Vec4::new(0.45, 0.45, 0.47, 1.0),
            ),
            GpuMesh::upload(
                &gpu.device,
                &viewport.layouts,
                &hidden_cpu,
                Vec4::new(0.45, 0.45, 0.47, 1.0),
            ),
        ];

        let mut camera = Camera::default().looking_along(ViewAxis::PosZ);
        camera.pivot = DVec3::ZERO;
        camera.distance = 6.0;

        Some(Self {
            gpu,
            viewport,
            meshes,
            transforms: vec![
                // Deliberately off centre, and above the middle of the frame
                // rather than on it. A scene symmetric about the screen centre
                // cannot detect a flipped V coordinate — the mirrored region is
                // the same region — and a falsifying edit that dropped the flip
                // passed a centred version of this test cleanly.
                DAffine3::from_translation(DVec3::new(0.3, 0.8, 0.0))
                    * DAffine3::from_scale(DVec3::splat(1.5)),
                // The hidden one, behind the occluder and sized so it is
                // *partly* covered. Both bounds of the share test matter: a
                // scene where everything is hidden, or nothing is, cannot tell
                // a working occlusion test from a stuck one.
                DAffine3::from_translation(DVec3::new(0.3, 0.8, -4.0))
                    * DAffine3::from_scale(DVec3::splat(0.9)),
            ],
            camera,
            hidden: ClusterMesh::bake(&hidden_cpu),
            hidden_index: 1,
        })
    }

    /// Renders the scene, builds the pyramid, and reads it back.
    ///
    /// The pyramid comes from a frame that drew *both* objects, which is the
    /// state a two-pass scheme's second cull sees. Testing against it is
    /// therefore testing the real question: given this depth buffer, which
    /// clusters are hidden.
    fn depth_and_pyramid(&mut self) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
        let target = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colour"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // Built from the fields rather than through a `&self` method: the
        // draws borrow `meshes`, `prepare` needs `&mut viewport`, and Rust can
        // split field borrows but not a whole-`self` one.
        let mut draws: Vec<Draw> = self
            .meshes
            .iter()
            .zip(&self.transforms)
            .map(|(mesh, t)| Draw::new(mesh, *t))
            .collect();
        self.viewport
            .prepare(&self.gpu.device, &self.gpu.queue, &self.camera, &mut draws);

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.viewport.render(&mut encoder, &view, &draws);
        self.viewport
            .build_depth_pyramid(&self.gpu.device, &self.gpu.queue, &mut encoder);

        // Split out of the borrow of `self.viewport`: the readback holds an
        // immutable borrow of the pyramid while the encoder needs a mutable one
        // of the viewport, so the sizes are copied first.
        let (pw, ph, mips) = {
            let hiz = self.viewport.depth_pyramid();
            (hiz.size().0, hiz.size().1, hiz.mip_count())
        };
        let levels = readback_pyramid(
            &self.gpu,
            self.viewport.depth_pyramid(),
            (pw, ph, mips),
            &mut encoder,
        );
        self.gpu.queue.submit([encoder.finish()]);
        let levels = levels.resolve(&self.gpu)?;
        Some((levels[0].clone(), levels))
    }

    /// Runs the cull with occlusion on, and returns the surviving clusters of
    /// the hidden mesh.
    fn visible_hidden_clusters(&mut self) -> Option<Vec<u32>> {
        let mut cull = ClusterCull::new(&self.gpu.device);
        let count = self.hidden.clusters.len() as u32;
        let budget = Budget {
            cull_px: CULL_PX,
            ..Budget::default()
        };
        cull.prepare(
            &self.gpu.device,
            &self.gpu.queue,
            &CullView {
                camera: &self.camera,
                aspect: 1.0,
                hiz: self.viewport.depth_pyramid(),
                occlusion: true,
                render_height: SIZE,
                budget: &budget,
            },
            &CullDraws {
                transforms: &[self.transforms[self.hidden_index]],
                cluster_counts: &[count],
                colors: &[[0.5, 0.5, 0.5, 1.0]],
            },
        );

        let cluster_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &self.gpu.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("clusters"),
                contents: bytemuck::cast_slice(&self.hidden.clusters),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let cluster_bind = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clusters"),
                layout: cull.cluster_layout(),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cluster_buffer.as_entire_binding(),
                }],
            });

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        cull.dispatch(&mut encoder, Pass::Single, &[&cluster_bind]);

        let args = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("args"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let list = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visible"),
            size: (count.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(cull.args_buffer(), 0, &args, 0, 16);
        encoder.copy_buffer_to_buffer(
            cull.visible_buffer(),
            0,
            &list,
            0,
            (count.max(1) * 4) as u64,
        );
        self.gpu.queue.submit([encoder.finish()]);

        args.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        list.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok()?;

        let kept = {
            let data = args.slice(..).get_mapped_range();
            let a: [u32; 4] = bytemuck::pod_read_unaligned(&data);
            a[1] as usize
        };
        args.unmap();
        let out = {
            let data = list.slice(..).get_mapped_range();
            bytemuck::cast_slice::<_, u32>(&data)[..kept].to_vec()
        };
        list.unmap();
        Some(out)
    }
}

/// Pending readbacks of every pyramid level.
struct PyramidReadback {
    buffers: Vec<(wgpu::Buffer, u32, u32, u32)>,
}

fn readback_pyramid(
    gpu: &kreate_render::Gpu,
    hiz: &HiZ,
    dims: (u32, u32, u32),
    encoder: &mut wgpu::CommandEncoder,
) -> PyramidReadback {
    let (w, h, mips) = dims;
    let mut buffers = Vec::new();
    for level in 0..mips {
        let (lw, lh) = hiz::mip_size(w, h, level);
        let padded = (lw * 4).div_ceil(256) * 256;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pyramid readback"),
            size: (padded * lh) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: hiz.texture(),
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(lh),
                },
            },
            wgpu::Extent3d {
                width: lw,
                height: lh,
                depth_or_array_layers: 1,
            },
        );
        buffers.push((buffer, lw, lh, padded));
    }
    PyramidReadback { buffers }
}

impl PyramidReadback {
    fn resolve(self, gpu: &kreate_render::Gpu) -> Option<Vec<Vec<f32>>> {
        for (buffer, ..) in &self.buffers {
            buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let mut levels = Vec::new();
        for (buffer, lw, lh, padded) in &self.buffers {
            let data = buffer.slice(..).get_mapped_range();
            let mut level = Vec::with_capacity((*lw * *lh) as usize);
            for y in 0..*lh {
                let row = &data[(y * padded) as usize..][..(*lw * 4) as usize];
                level.extend_from_slice(bytemuck::cast_slice::<_, f32>(row));
            }
            drop(data);
            buffer.unmap();
            levels.push(level);
        }
        Some(levels)
    }
}

#[test]
fn the_shader_rejects_exactly_the_clusters_the_cpu_rejects() {
    let Some(mut scene) = Scene::new() else {
        eprintln!("no adapter; skipping");
        return;
    };
    let Some((_, levels)) = scene.depth_and_pyramid() else {
        eprintln!("no adapter; skipping");
        return;
    };

    let got: std::collections::BTreeSet<u32> = scene
        .visible_hidden_clusters()
        .expect("cull ran")
        .into_iter()
        .collect();

    let view_proj = scene.camera.view_projection(1.0);
    let frustum = kreate_render::Frustum::from_view_projection(&view_proj);
    let pyramid = kreate_render::PyramidData {
        levels: &levels,
        size: scene.viewport.depth_pyramid().size(),
    };
    let transform = scene.transforms[scene.hidden_index];
    let want: std::collections::BTreeSet<u32> = scene
        .hidden
        .clusters
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            if !kreate_render::survives(c, &transform, &scene.camera, &frustum, SIZE, CULL_PX) {
                return false;
            }
            let sphere = kreate_render::WorldSphere::of(c.sphere, &transform, scene.camera.eye());
            !kreate_render::is_occluded(sphere, &view_proj, &pyramid)
        })
        .map(|(i, _)| i as u32)
        .collect();

    assert_eq!(
        got,
        want,
        "GPU kept {} clusters, CPU kept {}",
        got.len(),
        want.len()
    );
}

#[test]
fn occlusion_rejects_a_useful_share_and_not_everything() {
    // Agreement is worthless if the answer is trivial. A sphere hidden behind a
    // bigger one should lose a real fraction of its clusters and keep the rest —
    // both bounds matter, because "reject nothing" and "reject everything" are
    // the two ways this can be broken while still agreeing with a reference
    // that is broken the same way.
    let Some(mut scene) = Scene::new() else {
        eprintln!("no adapter; skipping");
        return;
    };
    if scene.depth_and_pyramid().is_none() {
        eprintln!("no adapter; skipping");
        return;
    }
    let total = scene.hidden.clusters.len();
    let kept = scene.visible_hidden_clusters().expect("cull ran").len();
    println!("{kept} of {total} clusters kept behind the occluder");
    assert!(kept > 0, "occlusion rejected every cluster");
    assert!(
        kept < total,
        "occlusion rejected nothing, so the test proves nothing"
    );
}

#[test]
fn nothing_rejected_as_occluded_was_actually_visible() {
    // The safety property, checked against the depth buffer itself rather than
    // against the CPU reference — so it holds even if the reference is wrong the
    // same way the shader is.
    //
    // For every cluster the cull rejected, every one of its triangles' vertices
    // is projected and compared against the depth actually recorded at that
    // pixel. If any vertex is nearer than what was drawn there, it was visible
    // and rejecting the cluster made a hole.
    let Some(mut scene) = Scene::new() else {
        eprintln!("no adapter; skipping");
        return;
    };
    let Some((depth, _)) = scene.depth_and_pyramid() else {
        eprintln!("no adapter; skipping");
        return;
    };
    let kept: std::collections::BTreeSet<u32> = scene
        .visible_hidden_clusters()
        .expect("cull ran")
        .into_iter()
        .collect();

    let view_proj = scene.camera.view_projection(1.0);
    let transform = scene.transforms[scene.hidden_index];
    let cpu = RenderMeshCpu::extract(&kreate_mesh::primitives::uv_sphere(1.0, 48, 24));
    let frustum = kreate_render::Frustum::from_view_projection(&view_proj);

    let mut checked = 0usize;
    for (i, cluster) in scene.hidden.clusters.iter().enumerate() {
        if kept.contains(&(i as u32)) {
            continue;
        }
        // Only clusters the *occlusion* test removed are relevant; the others
        // were off screen or backfacing, which this property says nothing about.
        if !kreate_render::survives(cluster, &transform, &scene.camera, &frustum, SIZE, CULL_PX) {
            continue;
        }
        checked += 1;

        let start = cluster.index_offset as usize;
        let end = start + cluster.triangle_count as usize * 3;
        for index in &scene.hidden.indices[start..end] {
            let local = glam::Vec3::from_array(cpu.positions[*index as usize].0);
            let world = transform.transform_point3(local.as_dvec3()) - scene.camera.eye();
            let clip = view_proj * world.as_vec3().extend(1.0);
            if clip.w <= 1e-6 {
                continue;
            }
            let ndc = clip.truncate() / clip.w;
            if ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 {
                continue;
            }
            let px = ((ndc.x * 0.5 + 0.5) * SIZE as f32) as i32;
            let py = ((-ndc.y * 0.5 + 0.5) * SIZE as f32) as i32;
            let px = px.clamp(0, SIZE as i32 - 1) as usize;
            let py = py.clamp(0, SIZE as i32 - 1) as usize;
            let drawn = depth[py * SIZE as usize + px];
            // Reversed-Z: this vertex is visible if it is nearer than what was
            // recorded, i.e. its depth is larger. A small tolerance because the
            // rasteriser's interpolated depth and this recomputed one differ in
            // the last bits.
            assert!(
                ndc.z <= drawn + 1e-4,
                "cluster {i} was rejected as occluded, but a vertex at depth \
                 {} is in front of the recorded {drawn}",
                ndc.z
            );
        }
    }
    assert!(
        checked > 0,
        "no cluster was rejected by occlusion, so this proved nothing"
    );
    println!("{checked} occlusion-rejected clusters verified hidden");
}

#[test]
fn occlusion_off_keeps_everything_occlusion_on_rejects() {
    // The switch does what it says, and the pyramid is actually being consulted.
    // Without this a cull that ignored `occlusion` entirely would still pass the
    // agreement test, because the CPU reference would be asked the same question
    // and both would keep everything.
    let Some(mut scene) = Scene::new() else {
        eprintln!("no adapter; skipping");
        return;
    };
    if scene.depth_and_pyramid().is_none() {
        eprintln!("no adapter; skipping");
        return;
    }
    let with = scene.visible_hidden_clusters().expect("cull ran").len();

    // The same frame with occlusion off.
    let mut cull = ClusterCull::new(&scene.gpu.device);
    let count = scene.hidden.clusters.len() as u32;
    let budget = Budget {
        cull_px: CULL_PX,
        ..Budget::default()
    };
    cull.prepare(
        &scene.gpu.device,
        &scene.gpu.queue,
        &CullView {
            camera: &scene.camera,
            aspect: 1.0,
            hiz: scene.viewport.depth_pyramid(),
            occlusion: false,
            render_height: SIZE,
            budget: &budget,
        },
        &CullDraws {
            transforms: &[scene.transforms[scene.hidden_index]],
            cluster_counts: &[count],
            colors: &[[0.5, 0.5, 0.5, 1.0]],
        },
    );
    let cluster_buffer = wgpu::util::DeviceExt::create_buffer_init(
        &scene.gpu.device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("clusters"),
            contents: bytemuck::cast_slice(&scene.hidden.clusters),
            usage: wgpu::BufferUsages::STORAGE,
        },
    );
    let cluster_bind = scene
        .gpu
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clusters"),
            layout: cull.cluster_layout(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cluster_buffer.as_entire_binding(),
            }],
        });
    let mut encoder = scene
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    cull.dispatch(&mut encoder, Pass::Single, &[&cluster_bind]);
    let args = scene.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("args"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(cull.args_buffer(), 0, &args, 0, 16);
    scene.gpu.queue.submit([encoder.finish()]);
    args.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    scene
        .gpu
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let without = {
        let data = args.slice(..).get_mapped_range();
        let a: [u32; 4] = bytemuck::pod_read_unaligned(&data);
        a[1] as usize
    };
    args.unmap();

    println!("{without} kept without occlusion, {with} with it");
    assert!(
        with < without,
        "turning occlusion on changed nothing: {with} against {without}"
    );
}
