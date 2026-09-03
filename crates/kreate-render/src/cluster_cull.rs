//! The GPU side of cluster culling: one compute dispatch, one indirect draw.
//!
//! [`crate::cluster`] bakes the clusters and defines what "visible" means, with
//! the predicates written and tested on the CPU. This module runs the same
//! predicates on the GPU and produces the argument buffer a `draw_indirect`
//! consumes, so the decision never travels back through the CPU. That is the
//! whole point: a readback would cost two or three frames of latency and put
//! the culling decision a frame behind the camera.
//!
//! # Scope, stated plainly
//!
//! The plan's end state is the *whole scene* in a single indirect draw, which
//! needs every mesh's vertices and indices sub-allocated from one arena. That
//! arena is a separate piece of work and is not built yet, so what runs here is
//! one dispatch and one indirect draw **per mesh**.
//!
//! The ordering is deliberate. All of the risk in this design is in the cull
//! predicates agreeing between CPU and GPU and in the degenerate-triangle
//! trick; none of it is in the arena, which is a mechanical change that turns N
//! draws into one without touching a line of shader code. Getting the arena
//! first would mean debugging both at once.
//!
//! # Measured, and currently slower than a direct draw
//!
//! **This path is off by default because it loses.** On 10.2 million triangles
//! across 64 linked duplicates, a plain indexed draw takes 6.5 ms and the
//! cluster path takes 20.6 ms — 2.4x worse, having been 8x worse before the
//! vertex cap was raised (see [`crate::cluster::MAX_VERTICES`]).
//!
//! The reason is the post-transform vertex cache. A `draw_indexed` shades each
//! vertex once and reuses it across the six triangles that share it; this draw
//! is necessarily *non-indexed*, because the instance is a cluster and each
//! cluster needs its own index range — which a single indexed indirect draw
//! cannot express, since `first_index` is per draw and not per instance. So
//! every triangle corner is a separate vertex shader invocation, and on a sphere
//! that is about six times the vertex work.
//!
//! What is *not* the reason, checked rather than assumed: the cull dispatch
//! itself, which is a few thousand workgroups doing three dot products, and the
//! draw count, which is identical between the two paths here because there is no
//! arena yet.
//!
//! So what would make it pay:
//!
//! - **Occlusion culling.** Frustum and cone rejection are cheap and shallow;
//!   a hi-z pyramid rejects far more, and every cluster it rejects is 384
//!   vertices not shaded. This is the next piece and the reason the work here is
//!   worth keeping.
//! - **The arena**, once a scene has thousands of objects rather than 64. At 64
//!   draws the direct path's draw-call cost is nothing; the argument changes
//!   entirely at 10,000.
//! - **A vertex range per cluster**, reordering the vertex buffer so a cluster's
//!   vertices are contiguous and the draw can be `draw_indexed_indirect` with a
//!   per-cluster `base_vertex`. That recovers the cache at the cost of
//!   duplicating vertices on cluster boundaries.
//!
//! The measurement lives in `tests/cluster_speedup.rs` and prints per scene, so
//! the next change to this file can be judged against a number rather than an
//! expectation.
//!
//! # Why the count is the indirect argument
//!
//! The cull increments the `instance_count` field of the draw arguments in
//! place, rather than filling a counter that a second pass copies across. One
//! dispatch instead of two, and — more to the point — no way for the counter
//! and the arguments to disagree about how many clusters survived.

use crate::camera::{Camera, Projection};
use crate::cluster::{Frustum, MAX_TRIANGLES};
use crate::hiz::HiZ;
use crate::lod::Budget;
use bytemuck::{Pod, Zeroable};
use glam::{DAffine3, Mat3, Mat4};

/// Threads per workgroup in the cull shader. Must match the shader.
const WORKGROUP: u32 = 64;

/// Vertices the indirect draw issues per surviving cluster.
///
/// Fixed, because a single draw cannot vary its vertex count per instance. A
/// cluster with fewer triangles collapses the surplus to a degenerate triangle
/// in the vertex shader — zero area, discarded by the rasteriser for free, and
/// far cheaper than the alternative of one draw per cluster.
pub const VERTICES_PER_CLUSTER: u32 = MAX_TRIANGLES as u32 * 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Globals {
    /// Camera-relative: projection times the view rotation, no translation.
    view_proj: [[f32; 4]; 4],
    planes: [[f32; 4]; Frustum::MAX_PLANES],
    /// x: viewport height in device pixels. y: cull diameter in pixels.
    /// z: tan(fov_y / 2). w: 1 for perspective, 0 for orthographic.
    params: [f32; 4],
    /// x: orthographic view height.
    ortho: [f32; 4],
    /// x, y: pyramid level-0 size. z: level count. w: 1 to test occlusion.
    hiz: [f32; 4],
}

/// Per-draw data shared by the cull and the cluster draw.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct InstanceData {
    model: [[f32; 4]; 4],
    /// The inverse transpose of the linear part, one column per row here.
    ///
    /// Three `vec4`s rather than a `mat3x3` in Rust because that is what WGSL's
    /// layout rules make of a `mat3x3<f32>`: three columns each padded to
    /// sixteen bytes. Writing three tightly packed `vec3`s is the classic way
    /// to get a silently misaligned matrix.
    normal_col0: [f32; 4],
    normal_col1: [f32; 4],
    normal_col2: [f32; 4],
    /// x: cluster count. y: base index into the visible list.
    counts: [u32; 4],
    /// x: largest scale factor in `model`.
    scale: [f32; 4],
    /// The mesh's base colour.
    ///
    /// Here rather than in a second uniform because this struct is bound by
    /// *both* the cull and the cluster draw, and the draw needs a colour. One
    /// buffer written once per frame per draw serves both.
    base_color: [f32; 4],
}

/// The draw arguments the cull writes and `draw_indirect` reads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

/// Where one draw's slice of the shared buffers lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CullSlot {
    pub cluster_count: u32,
    /// Where this draw's survivors are written in the shared visible list.
    pub visible_base: u32,
    /// Byte offset of this draw's `DrawArgs`, for both the dynamic binding and
    /// `draw_indirect`.
    pub args_offset: u32,
    /// Byte offset of this draw's instance uniform.
    pub instance_offset: u32,
}

impl CullSlot {
    /// Workgroups needed to cover this draw's clusters.
    pub fn workgroups(&self) -> u32 {
        self.cluster_count.div_ceil(WORKGROUP)
    }
}

/// Lays out the shared buffers for a frame's draws.
///
/// Separated from anything that touches wgpu so the arithmetic can be tested
/// without a device. It is arithmetic worth testing: an overlapping visible
/// range means two meshes writing survivors into each other's slice, and the
/// symptom is one mesh drawing fragments of another.
pub fn plan(cluster_counts: &[u32], instance_stride: u32, args_stride: u32) -> Vec<CullSlot> {
    let mut slots = Vec::with_capacity(cluster_counts.len());
    let mut base = 0u32;
    for (i, count) in cluster_counts.iter().enumerate() {
        let i = i as u32;
        slots.push(CullSlot {
            cluster_count: *count,
            visible_base: base,
            args_offset: i * args_stride,
            instance_offset: i * instance_stride,
        });
        base += *count;
    }
    slots
}

/// Total entries the visible list needs for a frame's draws.
pub fn visible_capacity(cluster_counts: &[u32]) -> u32 {
    cluster_counts.iter().sum()
}

/// Everything about the camera the cull needs.
///
/// Grouped rather than passed one by one, because a call taking ten positional
/// arguments is a call site nobody can read and two `f64`s in a row is an
/// invitation to swap them.
pub struct CullView<'a> {
    pub camera: &'a Camera,
    pub aspect: f64,
    /// The depth pyramid the occlusion test reads.
    ///
    /// Always supplied, even with `occlusion` off, because a bind group cannot
    /// have a hole. An unused pyramid costs one texture binding.
    pub hiz: &'a HiZ,
    /// Whether to reject occluded clusters.
    ///
    /// A switch and not a constant because the pyramid has to come from
    /// somewhere: testing against a pyramid built from a *stale* frame can
    /// reject clusters that became visible since, which is the one error this
    /// module must not make. The two-pass scheme is what makes it sound, and
    /// until a caller implements that, this stays off.
    pub occlusion: bool,
    /// The render target's height in device pixels, not the widget's — at 70%
    /// dynamic resolution there genuinely are fewer pixels per cluster.
    pub render_height: u32,
    pub budget: &'a Budget,
}

/// The frame's draws, as parallel slices.
///
/// Parallel and in the order the draw loop will use: the argument offset comes
/// from the position in these slices, so a caller that reorders between here and
/// the draw gets another mesh's cluster list.
pub struct CullDraws<'a> {
    pub transforms: &'a [DAffine3],
    pub cluster_counts: &'a [u32],
    pub colors: &'a [[f32; 4]],
}

/// The cull pass and the buffers it shares across draws.
pub struct ClusterCull {
    pipeline: wgpu::ComputePipeline,
    shared_layout: wgpu::BindGroupLayout,
    cluster_layout: wgpu::BindGroupLayout,
    globals: wgpu::Buffer,
    instances: wgpu::Buffer,
    visible: wgpu::Buffer,
    args: wgpu::Buffer,
    bind: wgpu::BindGroup,
    instance_stride: u32,
    args_stride: u32,
    draw_capacity: u32,
    visible_capacity: u32,
    /// The pyramid size group 0 was last built against.
    hiz_size: (u32, u32),
    slots: Vec<CullSlot>,
    /// Bumped whenever the shared buffers are reallocated.
    ///
    /// The cluster draw binds the same instance and visible buffers, so its bind
    /// groups go stale the moment these move. The same pattern the region-select
    /// bitset uses, and for the same reason: a bind group that names a freed
    /// buffer is a validation error at the next draw, not a subtle one.
    generation: u64,
}

impl ClusterCull {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster_cull"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cluster_cull.wgsl").into()),
        });

        let shared_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster cull shared"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // The per-draw instance, reached by dynamic offset: one buffer
                // for every draw in the frame rather than a bind group each.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // The draw arguments, also per draw and also dynamic. Storage
                // *and* indirect on the same buffer, which is what lets the
                // compute pass write the instance count that the draw then
                // reads without either one going through the CPU.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let cluster_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster records"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster cull"),
            bind_group_layouts: &[Some(&shared_layout), Some(&cluster_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cluster cull"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cull"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let storage_align = device.limits().min_storage_buffer_offset_alignment as u64;
        let instance_stride =
            (std::mem::size_of::<InstanceData>() as u64).div_ceil(uniform_align) * uniform_align;
        let args_stride =
            (std::mem::size_of::<DrawArgs>() as u64).div_ceil(storage_align) * storage_align;

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // A bind group cannot name buffers that do not exist yet, and every
        // field has to hold something, so group 0 starts out pointing at a
        // scratch buffer and is rebuilt by the `reserve` below. The alternative
        // is an `Option` that every later path has to unwrap.
        let placeholder = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull placeholder"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        // The pyramid slot needs something even before a real pyramid arrives.
        let placeholder_pyramid = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("cull placeholder pyramid"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: crate::hiz::FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let placeholder_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull placeholder"),
            layout: &shared_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: placeholder.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &placeholder,
                        offset: 0,
                        size: std::num::NonZeroU64::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: placeholder.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &placeholder,
                        offset: 0,
                        size: std::num::NonZeroU64::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&placeholder_pyramid),
                },
            ],
        });

        Self {
            pipeline,
            shared_layout,
            cluster_layout,
            globals,
            // Placeholders, replaced by the first `reserve`. Allocated rather
            // than left as `Option` so every field is a buffer and no code path
            // has to ask whether the pass has been set up yet.
            instances: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cull instances"),
                size: instance_stride,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            visible: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cull visible"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            args: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cull args"),
                size: args_stride,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            bind: placeholder_bind,
            instance_stride: instance_stride as u32,
            args_stride: args_stride as u32,
            draw_capacity: 0,
            visible_capacity: 0,
            hiz_size: (0, 0),
            slots: Vec::new(),
            generation: 0,
        }
    }

    /// The layout a mesh's cluster bind group must use.
    pub fn cluster_layout(&self) -> &wgpu::BindGroupLayout {
        &self.cluster_layout
    }

    /// Where each draw's slice lives, after the last [`Self::prepare`].
    pub fn slots(&self) -> &[CullSlot] {
        &self.slots
    }

    /// Changes whenever the shared buffers are reallocated.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The per-draw instance uniform, bound by the cluster draw as well.
    pub fn instances_buffer(&self) -> &wgpu::Buffer {
        &self.instances
    }

    /// One instance slot's size, for a dynamic binding.
    pub fn instance_size(&self) -> u64 {
        std::mem::size_of::<InstanceData>() as u64
    }

    /// The buffer `draw_indirect` reads.
    pub fn args_buffer(&self) -> &wgpu::Buffer {
        &self.args
    }

    /// The compacted list of surviving cluster indices.
    pub fn visible_buffer(&self) -> &wgpu::Buffer {
        &self.visible
    }

    /// Grows the shared buffers to hold `draws` draws and `clusters` survivors.
    fn reserve(&mut self, device: &wgpu::Device, hiz: &HiZ, draws: u32, clusters: u32) {
        let grow = draws > self.draw_capacity || clusters > self.visible_capacity;
        // The pyramid is owned by the caller and reallocated on resize, so the
        // bind group naming it goes stale independently of the buffers here.
        // Compared by size because that is what a resize changes and a texture
        // has no cheaper identity.
        let moved = hiz.size() != self.hiz_size;
        if !grow && !moved {
            return;
        }
        if !grow {
            self.hiz_size = hiz.size();
            self.rebuild_bind(device, hiz);
            return;
        }
        // Doubling, so a scene that grows steadily does not reallocate every
        // frame. The buffers are small — sixteen bytes a draw, four a cluster —
        // and reallocating mid-drag is a hitch.
        let draws = draws.max(self.draw_capacity * 2).max(1);
        let clusters = clusters.max(self.visible_capacity * 2).max(1);

        self.instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull instances"),
            size: self.instance_stride as u64 * draws as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull args"),
            size: self.args_stride as u64 * draws as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.visible = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull visible"),
            size: 4 * clusters as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.draw_capacity = draws;
        self.visible_capacity = clusters;
        self.hiz_size = hiz.size();
        self.rebuild_bind(device, hiz);
        self.generation += 1;
    }

    /// Rebuilds group 0, which names three buffers and the pyramid.
    fn rebuild_bind(&mut self, device: &wgpu::Device, hiz: &HiZ) {
        self.bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster cull shared"),
            layout: &self.shared_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.instances,
                        offset: 0,
                        size: std::num::NonZeroU64::new(std::mem::size_of::<InstanceData>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.args,
                        offset: 0,
                        size: std::num::NonZeroU64::new(std::mem::size_of::<DrawArgs>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(hiz.view()),
                },
            ],
        });
    }

    /// Writes the frame's cull inputs and resets the draw arguments.
    ///
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &CullView,
        draws: &CullDraws,
    ) {
        let CullView {
            camera,
            aspect,
            hiz,
            occlusion,
            render_height,
            budget,
        } = view;
        let CullDraws {
            transforms,
            cluster_counts,
            colors,
        } = draws;
        debug_assert_eq!(transforms.len(), cluster_counts.len());
        debug_assert_eq!(transforms.len(), colors.len());
        self.reserve(
            device,
            hiz,
            (cluster_counts.len() as u32).max(1),
            visible_capacity(cluster_counts).max(1),
        );
        self.slots = plan(cluster_counts, self.instance_stride, self.args_stride);

        let frustum = Frustum::from_view_projection(&camera.view_projection(*aspect));
        let planes = frustum.to_padded_planes();
        queue.write_buffer(
            &self.globals,
            0,
            bytemuck::bytes_of(&Globals {
                view_proj: camera.view_projection(*aspect).to_cols_array_2d(),
                planes: planes.map(|p| p.to_array()),
                params: [
                    *render_height as f32,
                    budget.cull_px,
                    (camera.fov_y * 0.5).tan() as f32,
                    match camera.projection {
                        Projection::Perspective => 1.0,
                        Projection::Orthographic => 0.0,
                    },
                ],
                ortho: [camera.ortho_height() as f32, 0.0, 0.0, 0.0],
                hiz: [
                    hiz.size().0 as f32,
                    hiz.size().1 as f32,
                    hiz.mip_count() as f32,
                    f32::from(u8::from(*occlusion)),
                ],
            }),
        );

        for ((slot, transform), color) in self.slots.iter().zip(*transforms).zip(*colors) {
            let model = camera.model_matrix_relative(transform);
            let normal = normal_matrix(transform);
            let m = transform.matrix3;
            let scale = m
                .x_axis
                .length()
                .max(m.y_axis.length())
                .max(m.z_axis.length()) as f32;
            queue.write_buffer(
                &self.instances,
                slot.instance_offset as u64,
                bytemuck::bytes_of(&InstanceData {
                    model: model.to_cols_array_2d(),
                    normal_col0: normal.x_axis.extend(0.0).to_array(),
                    normal_col1: normal.y_axis.extend(0.0).to_array(),
                    normal_col2: normal.z_axis.extend(0.0).to_array(),
                    counts: [slot.cluster_count, slot.visible_base, 0, 0],
                    scale: [scale, 0.0, 0.0, 0.0],
                    base_color: *color,
                }),
            );
            // Reset every frame. The cull increments this field, so a stale
            // count would draw last frame's survivors on top of this frame's.
            queue.write_buffer(
                &self.args,
                slot.args_offset as u64,
                bytemuck::bytes_of(&DrawArgs {
                    vertex_count: VERTICES_PER_CLUSTER,
                    instance_count: 0,
                    first_vertex: 0,
                    first_instance: 0,
                }),
            );
        }
    }

    /// Runs the cull for every draw in one compute pass.
    ///
    /// `cluster_binds` is parallel to the slices given to [`Self::prepare`].
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, cluster_binds: &[&wgpu::BindGroup]) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cluster cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (slot, clusters) in self.slots.iter().zip(cluster_binds) {
            if slot.cluster_count == 0 {
                continue;
            }
            pass.set_bind_group(0, &self.bind, &[slot.instance_offset, slot.args_offset]);
            pass.set_bind_group(1, *clusters, &[]);
            pass.dispatch_workgroups(slot.workgroups(), 1, 1);
        }
    }
}

/// The inverse transpose of a transform's linear part.
///
/// Directions do not transform like positions under non-uniform scale: scaling
/// a squashed sphere's normals by the same matrix tilts them the wrong way. The
/// cone axis is a direction, so it needs this.
fn normal_matrix(transform: &DAffine3) -> Mat3 {
    Mat3::from_cols(
        transform.matrix3.x_axis.as_vec3(),
        transform.matrix3.y_axis.as_vec3(),
        transform.matrix3.z_axis.as_vec3(),
    )
    .inverse()
    .transpose()
}

/// Reads the shader source, for the parse test in `viewport`.
pub fn shader_source() -> &'static str {
    include_str!("shaders/cluster_cull.wgsl")
}

/// The view-projection a cull was set up with, for tests that need to agree.
pub fn view_projection(camera: &Camera, aspect: f64) -> Mat4 {
    camera.view_projection(aspect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_of_the_visible_list_never_overlap() {
        // Two meshes writing survivors into each other's slice draws fragments
        // of one mesh inside another, which reads as corrupt topology rather
        // than as a buffer layout bug.
        let counts = [10u32, 0, 250, 1, 64];
        let slots = plan(&counts, 256, 256);
        let mut end = 0u32;
        for slot in &slots {
            assert_eq!(slot.visible_base, end);
            end += slot.cluster_count;
        }
        assert_eq!(end, visible_capacity(&counts));
    }

    #[test]
    fn each_draw_gets_its_own_argument_slot() {
        // The count is incremented in place, so two draws sharing a slot would
        // each draw the other's survivor count.
        let slots = plan(&[4, 4, 4], 256, 256);
        let offsets: Vec<u32> = slots.iter().map(|s| s.args_offset).collect();
        assert_eq!(offsets, vec![0, 256, 512]);
        let instances: Vec<u32> = slots.iter().map(|s| s.instance_offset).collect();
        assert_eq!(instances, vec![0, 256, 512]);
    }

    #[test]
    fn the_workgroup_count_covers_every_cluster() {
        // A cluster past the last workgroup is silently never tested, and since
        // untested means unlisted, it silently never draws.
        for count in [0u32, 1, 63, 64, 65, 128, 129, 4095] {
            let slots = plan(&[count], 256, 256);
            let covered = slots[0].workgroups() * WORKGROUP;
            assert!(
                covered >= count,
                "{count} clusters covered by only {covered} threads"
            );
        }
    }

    #[test]
    fn an_empty_frame_plans_nothing() {
        assert!(plan(&[], 256, 256).is_empty());
        assert_eq!(visible_capacity(&[]), 0);
    }

    #[test]
    fn the_uniform_structs_match_the_shader_layout() {
        // A mat3x3 in WGSL is three columns each padded to sixteen bytes, so
        // three tightly packed vec3s would misalign everything after them and
        // the cone axis would read as garbage. Pinned because the symptom is
        // clusters culled at random rather than a compile error.
        assert_eq!(std::mem::size_of::<Globals>(), 208);
        assert_eq!(std::mem::size_of::<InstanceData>(), 160);
        assert_eq!(std::mem::size_of::<DrawArgs>(), 16);
    }

    #[test]
    fn the_indirect_draw_issues_one_triangle_per_cluster_triangle() {
        assert_eq!(VERTICES_PER_CLUSTER, 384);
        assert_eq!(VERTICES_PER_CLUSTER as usize, MAX_TRIANGLES * 3);
    }

    #[test]
    fn the_shader_parses() {
        let module = naga::front::wgsl::parse_str(shader_source()).expect("cull shader parses");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).expect("cull shader validates");
    }
}
