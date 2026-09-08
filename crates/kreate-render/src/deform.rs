//! The interactive transform, on the GPU.
//!
//! # What this is for
//!
//! Getting a mesh ready for the GPU on the CPU costs **83 ms a frame at two
//! hundred thousand vertices**, against the plan's budget of one. Almost all of
//! it is work a move does not change — the triangulation, the vertex split, the
//! edge list — and the rest is arithmetic a GPU finishes in microseconds.
//!
//! So while a drag is running the CPU stops producing geometry. It uploads a
//! [`Gesture`] — sixty-four bytes — and the compute passes in `deform.wgsl`
//! write the positions and normals the rasteriser reads, from an immutable copy
//! of the mesh as it was when the drag began.
//!
//! # The normals are recomputed, not transformed
//!
//! Transforming them by the same matrix would have been much less work and
//! would have been wrong in two places at once: on the boundary of the moved
//! region, where a vertex's smooth group spans faces that moved and faces that
//! did not, and everywhere under proportional editing, where the surface
//! genuinely bends. Both would show as a shading pop when the pointer came up
//! and the CPU recomputed properly.
//!
//! Instead the second and third passes do what `compute_corner_normals` does:
//! face normals by Newell from the deformed positions, then a sum over each
//! render vertex's own smooth group. A render vertex *is* a (vertex, smooth
//! group) pair — that is what the split is for — so the group is exactly the
//! set the CPU sums, and the answers agree. A device test asserts they do.
//!
//! # What the CPU still owns
//!
//! Everything that has to survive. The document is not touched during the drag;
//! the operator runs once when it is confirmed, and that is what the undo stack,
//! the adjust panel and any script replay see. This is a *preview*, and it is
//! discarded rather than read back.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use crate::mesh_gpu::RenderMeshCpu;

/// One frame of a drag, as the shader sees it.
///
/// Decomposed rather than a matrix, because a weighted transform is not a
/// weighted matrix: proportional editing blends the *transform* towards the
/// identity, and for a rotation that is a slerp rather than a lerp of matrices.
/// `deform.wgsl` says more about why the difference is the whole point.
///
/// A gesture is one of move, rotate or scale, so exactly one of the first three
/// is away from the identity. The shader composes all three, which costs a few
/// instructions and saves it having a mode.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Gesture {
    /// xyz used; w is padding a uniform block requires.
    pub translation: [f32; 4],
    /// A quaternion, xyzw.
    pub rotation: [f32; 4],
    pub scale: [f32; 4],
    /// What the rotation and the scale happen about.
    pub pivot: [f32; 4],
}

impl Default for Gesture {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Gesture {
    /// Moves nothing. What a drag starts from, so the first frame of it is the
    /// mesh exactly as it was.
    pub const IDENTITY: Self = Self {
        translation: [0.0; 4],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0, 0.0],
        pivot: [0.0; 4],
    };

    pub fn moved(offset: glam::Vec3) -> Self {
        Self {
            translation: [offset.x, offset.y, offset.z, 0.0],
            ..Self::IDENTITY
        }
    }

    pub fn turned(rotation: glam::Quat, pivot: glam::Vec3) -> Self {
        Self {
            rotation: rotation.to_array(),
            pivot: [pivot.x, pivot.y, pivot.z, 0.0],
            ..Self::IDENTITY
        }
    }

    pub fn scaled(factor: glam::Vec3, pivot: glam::Vec3) -> Self {
        Self {
            scale: [factor.x, factor.y, factor.z, 0.0],
            pivot: [pivot.x, pivot.y, pivot.z, 0.0],
            ..Self::IDENTITY
        }
    }
}

/// How many threads a workgroup runs. Must match `deform.wgsl`.
const WORKGROUP: u32 = 64;

/// How many workgroups cover `items` threads.
///
/// No `max(1)`: `div_ceil` already gives at least one for any non-zero count,
/// so the clamp only ever applied to *zero* items — where it dispatched a
/// workgroup whose every thread failed its bounds check and returned. A mesh of
/// loose vertices has no faces and reaches exactly that case. WebGPU allows a
/// count of zero, and zero is what nothing to do means.
fn workgroups(items: u32) -> u32 {
    items.div_ceil(WORKGROUP)
}

/// The pipelines, built once for the device.
pub struct Deformer {
    live: wgpu::BindGroupLayout,
    topology: wgpu::BindGroupLayout,
    positions: wgpu::ComputePipeline,
    face_normals: wgpu::ComputePipeline,
    vertex_normals: wgpu::ComputePipeline,
}

impl Deformer {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deform"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/deform.wgsl").into()),
        });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let live = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deform live"),
            entries: &[
                storage(0, true),  // rest positions
                storage(1, true),  // rest normals
                storage(2, false), // live positions
                storage(3, false), // live normals
                storage(4, true),  // weights
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let topology = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deform topology"),
            entries: &[
                storage(0, true),  // face offsets
                storage(1, true),  // corner to render vertex
                storage(2, false), // face normals
                storage(3, true),  // group offsets
                storage(4, true),  // group faces
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("deform"),
            bind_group_layouts: &[Some(&live), Some(&topology)],
            immediate_size: 0,
        });
        let stage = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            positions: stage("deform_positions"),
            face_normals: stage("compute_face_normals"),
            vertex_normals: stage("compute_vertex_normals"),
            live,
            topology,
        }
    }
}

/// One mesh's drag, held for as long as the pointer is down.
pub struct Deform {
    live: wgpu::BindGroup,
    topology: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    /// Kept because the proportional radius is adjustable *during* a drag — see
    /// [`Deform::set_weights`].
    weights: wgpu::Buffer,
    render_verts: u32,
    faces: u32,
}

impl Deform {
    /// Prepares a drag: copies the mesh as it is and uploads what deforming it
    /// needs.
    ///
    /// `weights` is one per *render* vertex — see
    /// [`RenderMeshCpu::render_to_vert`], which is what turns the CPU's answer
    /// per mesh vertex into that. A weight of zero leaves a vertex where it is.
    ///
    /// The rest copies are made on the GPU with a buffer-to-buffer copy rather
    /// than uploaded from the CPU: the bytes are already there, and sending them
    /// again would be the whole cost this exists to avoid.
    pub fn begin(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        deformer: &Deformer,
        mesh: &crate::viewport::GpuMesh,
        cpu: &RenderMeshCpu,
        weights: &[f32],
    ) -> Option<Self> {
        let render_verts = cpu.positions.len() as u32;
        if render_verts == 0 || weights.len() as u32 != render_verts {
            return None;
        }
        let positions_bytes = (render_verts as u64) * 12;
        let normals_bytes = (render_verts as u64) * 4;

        let rest_positions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform rest positions"),
            size: positions_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rest_normals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform rest normals"),
            size: normals_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(mesh.positions(), 0, &rest_positions, 0, positions_bytes);
        encoder.copy_buffer_to_buffer(mesh.normals(), 0, &rest_normals, 0, normals_bytes);

        let init = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        // COPY_DST, unlike the tables below it: the proportional radius is a
        // live control during a drag, so this is the one buffer here that gets
        // rewritten. See `set_weights`.
        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform weights"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let face_offsets = init(
            "deform face offsets",
            bytemuck::cast_slice(&cpu.face_offsets),
        );
        let corner_render = init(
            "deform corner render",
            bytemuck::cast_slice(&cpu.corner_render),
        );
        let group_offsets = init(
            "deform group offsets",
            bytemuck::cast_slice(&cpu.group_offsets),
        );
        // A mesh of one face has no group entries at all, and a zero-sized
        // storage buffer is not a thing wgpu will bind.
        let group_faces = init(
            "deform group faces",
            bytemuck::cast_slice(pad_if_empty(&cpu.group_faces)),
        );
        let faces = cpu.counts.faces.max(1);
        let face_normals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform face normals"),
            size: (faces as u64) * 12,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform gesture"),
            contents: bytemuck::bytes_of(&Gesture::IDENTITY),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let live = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform live"),
            layout: &deformer.live,
            entries: &[
                entry(0, &rest_positions),
                entry(1, &rest_normals),
                entry(2, mesh.positions()),
                entry(3, mesh.normals()),
                entry(4, &weights_buffer),
                entry(5, &uniform),
            ],
        });
        let topology = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform topology"),
            layout: &deformer.topology,
            entries: &[
                entry(0, &face_offsets),
                entry(1, &corner_render),
                entry(2, &face_normals),
                entry(3, &group_offsets),
                entry(4, &group_faces),
            ],
        });

        Some(Self {
            live,
            topology,
            uniform,
            weights: weights_buffer,
            render_verts,
            faces: cpu.counts.faces,
        })
    }

    /// Sets this frame's gesture. Sixty-four bytes, and all the CPU normally
    /// uploads while a drag is running.
    pub fn set(&self, queue: &wgpu::Queue, gesture: &Gesture) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(gesture));
    }

    /// Replaces the falloff, for when the proportional radius changes mid-drag.
    ///
    /// The scroll wheel adjusts that radius while the pointer is still down, so
    /// a `Deform` that could not follow it could not host proportional editing
    /// at all — which is most of what this path was built for. This is the
    /// expensive upload of the drag (four bytes a render vertex against the
    /// gesture's sixty-four in total), so it happens on the frames where the
    /// wheel actually turned and not on the others.
    ///
    /// It is also what makes `deform_positions`' rest-position write on an
    /// unweighted vertex load-bearing rather than tidy: a vertex the falloff
    /// used to reach and no longer does has last frame's answer sitting in the
    /// live buffer, and something has to put it back.
    ///
    /// Returns whether it was applied; a wrong length is refused rather than
    /// written short, since a short write would leave the tail of the falloff
    /// as whatever the last one said.
    #[must_use]
    pub fn set_weights(&self, queue: &wgpu::Queue, weights: &[f32]) -> bool {
        if weights.len() as u32 != self.render_verts {
            return false;
        }
        queue.write_buffer(&self.weights, 0, bytemuck::cast_slice(weights));
        true
    }

    /// Writes the deformed positions and normals into the mesh's own buffers.
    ///
    /// Three passes rather than the two the plan sketched: the positions have to
    /// be finished before the face normals read them, and the face normals
    /// before the vertex pass sums them. WebGPU orders dispatches within a
    /// single pass too, so a version that used one pass for all three also
    /// passes the device tests — the separate passes are the explicit form of
    /// the dependency, and they give each stage its own name in a capture.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, deformer: &Deformer) {
        for (label, pipeline, items) in [
            ("deform positions", &deformer.positions, self.render_verts),
            ("deform face normals", &deformer.face_normals, self.faces),
            (
                "deform vertex normals",
                &deformer.vertex_normals,
                self.render_verts,
            ),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.live, &[]);
            pass.set_bind_group(1, &self.topology, &[]);
            pass.dispatch_workgroups(workgroups(items), 1, 1);
        }
    }
}

/// A one-element slice when the input is empty.
///
/// wgpu will not bind a zero-sized storage buffer, and a mesh can legitimately
/// have nothing in one of these tables.
fn pad_if_empty(values: &[u32]) -> &[u32] {
    if values.is_empty() {
        &[0]
    } else {
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gesture_that_moves_nothing_is_the_identity() {
        // What the first frame of a drag uploads, so pressing the key alone must
        // leave the mesh exactly where it is.
        let g = Gesture::IDENTITY;
        assert_eq!(g.translation, [0.0; 4]);
        assert_eq!(g.rotation, [0.0, 0.0, 0.0, 1.0], "not a unit quaternion");
        assert_eq!(g.scale, [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(Gesture::default(), g);
    }

    #[test]
    fn each_gesture_leaves_the_other_two_at_the_identity() {
        // The shader composes all three without a branch, which is only safe
        // because a gesture is one of them. A constructor that left a zero scale
        // behind would collapse the mesh to its pivot.
        let m = Gesture::moved(glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.scale, Gesture::IDENTITY.scale);
        assert_eq!(m.rotation, Gesture::IDENTITY.rotation);

        let pivot = glam::Vec3::new(4.0, 5.0, 6.0);
        let t = Gesture::turned(glam::Quat::from_rotation_y(1.0), pivot);
        assert_eq!(t.scale, Gesture::IDENTITY.scale);
        assert_eq!(t.translation, Gesture::IDENTITY.translation);
        assert_eq!(t.pivot, [4.0, 5.0, 6.0, 0.0]);

        let s = Gesture::scaled(glam::Vec3::splat(2.0), pivot);
        assert_eq!(s.rotation, Gesture::IDENTITY.rotation);
        assert_eq!(s.translation, Gesture::IDENTITY.translation);
        assert_eq!(s.pivot, [4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn a_gesture_is_the_size_the_uniform_block_expects() {
        // Four `vec4`s. A uniform block is padded to sixteen bytes a member, so
        // a `vec3` here would be silently padded and the shader would read the
        // next field's bytes as the previous one's w.
        assert_eq!(std::mem::size_of::<Gesture>(), 64);
        assert_eq!(std::mem::align_of::<Gesture>(), 4);
    }

    #[test]
    fn the_workgroup_count_covers_every_item() {
        // An off-by-one here leaves the last few vertices of a mesh at their
        // rest position, which reads as the mesh tearing.
        //
        // Zero is in the list, and it is what caught the `max(1)` this function
        // used to end with: one workgroup for no items is not a cover, it is a
        // dispatch whose every thread bails. A mesh of loose vertices has no
        // faces and asks for exactly that.
        for items in [0, 1, 63, 64, 65, 1000, 203_401] {
            let groups = workgroups(items);
            assert!(
                groups * WORKGROUP >= items,
                "{items} items need more than {groups} workgroups"
            );
            // And not wastefully more than one workgroup's worth.
            assert!(
                groups * WORKGROUP < items + WORKGROUP,
                "{items} items dispatched {groups} workgroups"
            );
        }
        assert_eq!(workgroups(0), 0, "nothing to do is nothing dispatched");
        assert_eq!(workgroups(1), 1, "one item still needs a whole workgroup");
    }

    #[test]
    fn an_empty_table_is_padded_rather_than_bound_empty() {
        // wgpu will not bind a zero-sized storage buffer, and a mesh can
        // legitimately have nothing in one of these tables.
        assert_eq!(pad_if_empty(&[]), &[0]);
        assert_eq!(pad_if_empty(&[7, 8]), &[7, 8]);
    }
}
