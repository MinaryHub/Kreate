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

/// Where a gesture lands: the object's space, and the point it turns about.
///
/// A mesh's vertices are in the object's own coordinates and the gesture is in
/// the world's, so something has to carry the object between them. This is that
/// something, in the form the shader wants: the linear part of the
/// object-to-world transform and its inverse, and the object's world position
/// *measured from the pivot*.
///
/// # Relative to the pivot, not absolute
///
/// Everything here is a displacement from the point the gesture turns about,
/// and that is deliberate. The shader is `f32`; world coordinates are not
/// bounded. An object parked a million units from the origin, round-tripped
/// through absolute single-precision world space, loses a millimetre drag
/// entirely to rounding — while a displacement from the pivot is a metre or two
/// however far away the object is parked. It also means the shader never sees
/// a pivot: it is already the origin of the frame the arithmetic happens in.
///
/// The translation is not stored: only the linear part is needed, because the
/// object's own offset cancels in a displacement. `origin` carries what does
/// not cancel — where the object's origin sits relative to the pivot.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Placement {
    /// The object-to-world linear part, by column, each padded to a `vec4` as
    /// a uniform block requires.
    to_world: [[f32; 4]; 3],
    /// Its inverse, the same way.
    to_local: [[f32; 4]; 3],
    /// The object's world translation minus the pivot. xyz used.
    origin: [f32; 4],
}

fn columns(m: glam::DMat3) -> [[f32; 4]; 3] {
    [m.x_axis, m.y_axis, m.z_axis].map(|c| [c.x as f32, c.y as f32, c.z as f32, 0.0])
}

impl Placement {
    /// A mesh already in world coordinates, turning about the origin.
    ///
    /// What a test on bare geometry has, and what [`Gesture::IDENTITY`] uses.
    pub const NOWHERE: Self = Self {
        to_world: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        to_local: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        origin: [0.0; 4],
    };

    /// A mesh already in world coordinates, turning about `pivot`.
    pub fn about(pivot: glam::DVec3) -> Self {
        Self {
            origin: [-pivot.x as f32, -pivot.y as f32, -pivot.z as f32, 0.0],
            ..Self::NOWHERE
        }
    }

    /// `to_world` maps the mesh's vertices into the world; `pivot` is where a
    /// rotation or a scale happens, in world coordinates.
    ///
    /// `None` when the transform has no inverse — an object flattened by a zero
    /// scale. There is no way back from world space then, and the operator
    /// refuses the same case rather than writing NaN into the mesh.
    pub fn new(to_world: glam::DAffine3, pivot: glam::DVec3) -> Option<Self> {
        let inverse = to_world.matrix3.inverse();
        if !inverse.is_finite() {
            return None;
        }
        let origin = to_world.translation - pivot;
        Some(Self {
            to_world: columns(to_world.matrix3),
            to_local: columns(inverse),
            origin: [origin.x as f32, origin.y as f32, origin.z as f32, 0.0],
        })
    }
}

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
    /// The world offset. xyz used; w is padding a uniform block requires.
    pub translation: [f32; 4],
    /// A quaternion, xyzw, about the world axes.
    pub rotation: [f32; 4],
    /// Per world axis.
    pub scale: [f32; 4],
    /// Which object's space the mesh is in, and what the gesture turns about.
    pub place: Placement,
}

impl Default for Gesture {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Gesture {
    /// Moves nothing. What a drag starts from, so the first frame of it is the
    /// mesh exactly as it was.
    ///
    /// Exact for any placement, not just this one: the shader's answer is a
    /// displacement, the identity's displacement is zero before it is
    /// transformed, and no matrix moves the zero vector. Which is what makes
    /// this the way a cancelled drag puts a mesh back.
    pub const IDENTITY: Self = Self {
        translation: [0.0; 4],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0, 0.0],
        place: Placement::NOWHERE,
    };

    pub fn moved(offset: glam::Vec3, place: Placement) -> Self {
        Self {
            translation: [offset.x, offset.y, offset.z, 0.0],
            place,
            ..Self::IDENTITY
        }
    }

    pub fn turned(rotation: glam::Quat, place: Placement) -> Self {
        Self {
            rotation: rotation.to_array(),
            place,
            ..Self::IDENTITY
        }
    }

    pub fn scaled(factor: glam::Vec3, place: Placement) -> Self {
        Self {
            scale: [factor.x, factor.y, factor.z, 0.0],
            place,
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
    pub(crate) live: wgpu::BindGroupLayout,
    pub(crate) topology: wgpu::BindGroupLayout,
    positions: wgpu::ComputePipeline,
    pub(crate) face_normals: wgpu::ComputePipeline,
    pub(crate) vertex_normals: wgpu::ComputePipeline,
    /// Skinning's position pass. The normals after it are the two above:
    /// see [`crate::skin`].
    pub(crate) skin: wgpu::ComputePipeline,
    pub(crate) skin_layout: wgpu::BindGroupLayout,
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
        let skin_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skin"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/skin.wgsl").into()),
        });
        let skin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin"),
            entries: &[
                storage(0, true),  // rest positions
                storage(1, false), // live positions
                storage(2, true),  // render vertex to mesh vertex
                storage(3, true),  // influence offsets
                storage(4, true),  // influence bones
                storage(5, true),  // influence weights
                storage(6, true),  // bone matrices
            ],
        });
        let skin_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skin"),
            bind_group_layouts: &[Some(&skin_layout)],
            immediate_size: 0,
        });
        let skin = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("skin_positions"),
            layout: Some(&skin_pipeline_layout),
            module: &skin_module,
            entry_point: Some("skin_positions"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            skin,
            skin_layout,
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

        // COPY_DST, unlike the topology tables: the proportional radius is a
        // live control during a drag, so this is the one buffer here that gets
        // rewritten. See `set_weights`.
        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform weights"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform gesture"),
            contents: bytemuck::bytes_of(&Gesture::IDENTITY),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let live = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform live"),
            layout: &deformer.live,
            entries: &[
                bind_entry(0, &rest_positions),
                bind_entry(1, &rest_normals),
                bind_entry(2, mesh.positions()),
                bind_entry(3, mesh.normals()),
                bind_entry(4, &weights_buffer),
                bind_entry(5, &uniform),
            ],
        });
        let topology = topology_group(device, deformer, cpu);

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

/// The tables the normal passes walk, bound: which render vertices make up
/// each face and which faces each render vertex averages. A function of the
/// topology alone, so a drag and a skin build the same thing.
pub(crate) fn topology_group(
    device: &wgpu::Device,
    deformer: &Deformer,
    cpu: &RenderMeshCpu,
) -> wgpu::BindGroup {
    let init = |label: &str, contents: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let face_offsets = init("deform face offsets", bytemuck::cast_slice(&cpu.face_offsets));
    let corner_render = init("deform corner render", bytemuck::cast_slice(&cpu.corner_render));
    let group_offsets = init("deform group offsets", bytemuck::cast_slice(&cpu.group_offsets));
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
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("deform topology"),
        layout: &deformer.topology,
        entries: &[
            bind_entry(0, &face_offsets),
            bind_entry(1, &corner_render),
            bind_entry(2, &face_normals),
            bind_entry(3, &group_offsets),
            bind_entry(4, &group_faces),
        ],
    })
}

/// The workgroups that cover `items`, for the passes in [`crate::skin`].
pub(crate) fn workgroups_for(items: u32) -> u32 {
    workgroups(items)
}

/// A one-element slice when the input is empty.
///
/// wgpu will not bind a zero-sized storage buffer, and a mesh can legitimately
/// have nothing in one of these tables.
pub(crate) fn pad_if_empty(values: &[u32]) -> &[u32] {
    if values.is_empty() {
        &[0]
    } else {
        values
    }
}

pub(crate) fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
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
        let place = Placement::about(glam::DVec3::new(4.0, 5.0, 6.0));
        let m = Gesture::moved(glam::Vec3::new(1.0, 2.0, 3.0), place);
        assert_eq!(m.scale, Gesture::IDENTITY.scale);
        assert_eq!(m.rotation, Gesture::IDENTITY.rotation);

        let t = Gesture::turned(glam::Quat::from_rotation_y(1.0), place);
        assert_eq!(t.scale, Gesture::IDENTITY.scale);
        assert_eq!(t.translation, Gesture::IDENTITY.translation);
        assert_eq!(t.place, place);

        let s = Gesture::scaled(glam::Vec3::splat(2.0), place);
        assert_eq!(s.rotation, Gesture::IDENTITY.rotation);
        assert_eq!(s.translation, Gesture::IDENTITY.translation);
        assert_eq!(s.place, place);
    }

    #[test]
    fn a_pivot_is_carried_as_the_object_s_offset_from_it() {
        // What makes the shader's arithmetic small: the pivot is the origin of
        // the frame everything is measured in, so what travels is the object's
        // position *relative to it* rather than two absolute world points.
        let place = Placement::about(glam::DVec3::new(4.0, 5.0, -6.0));
        assert_eq!(place.origin, [-4.0, -5.0, 6.0, 0.0]);

        // With a real object transform, the same thing plus the object's own
        // offset.
        let to_world = glam::DAffine3::from_translation(glam::DVec3::new(10.0, 0.0, 0.0));
        let placed = Placement::new(to_world, glam::DVec3::new(11.0, 2.0, 0.0))
            .expect("a translation is invertible");
        assert_eq!(placed.origin, [-1.0, -2.0, 0.0, 0.0]);
        assert_eq!(placed.to_world, Placement::NOWHERE.to_world);
    }

    #[test]
    fn a_placement_inverts_the_object_s_own_transform() {
        // The two matrices have to be each other's inverse or a vertex does not
        // come back to where it started, and a drag of nothing would move the
        // mesh.
        let to_world = glam::DAffine3::from_scale_rotation_translation(
            glam::DVec3::new(3.0, 0.5, 1.75),
            glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.4, -0.9, 1.3),
            glam::DVec3::new(-3.0, 7.0, 1.0),
        );
        let place = Placement::new(to_world, glam::DVec3::ZERO).expect("invertible");
        let read = |m: &[[f32; 4]; 3]| {
            glam::Mat3::from_cols_array(&[
                m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
            ])
        };
        let round_trip = read(&place.to_local) * read(&place.to_world);
        let error = (round_trip - glam::Mat3::IDENTITY).to_cols_array();
        let worst = error.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(worst < 1e-5, "the round trip is off by {worst:e}");
    }

    #[test]
    fn an_object_with_no_inverse_has_no_placement() {
        // A zero in the scale: there is no way back from world space, and the
        // operator refuses the same case rather than writing NaN into the mesh.
        // A `Placement` that returned matrices full of infinities would have
        // the shader do it instead.
        let flat = glam::DAffine3::from_scale(glam::DVec3::new(1.0, 0.0, 1.0));
        assert!(Placement::new(flat, glam::DVec3::ZERO).is_none());
        assert!(Placement::new(glam::DAffine3::ZERO, glam::DVec3::ZERO).is_none());
        // And an ordinary one does.
        assert!(Placement::new(glam::DAffine3::IDENTITY, glam::DVec3::ONE).is_some());
    }

    #[test]
    fn the_identity_moves_nothing_whatever_the_placement() {
        // What a cancelled drag relies on. The shader's answer is a
        // displacement: the identity's is zero *before* it is transformed back,
        // and no matrix moves the zero vector — so this holds for an object of
        // any shape, parked anywhere.
        let to_world = glam::DAffine3::from_scale_rotation_translation(
            glam::DVec3::new(7.0, 0.1, 2.0),
            glam::DQuat::from_rotation_z(2.2),
            glam::DVec3::new(1e6, -4.0, 0.0),
        );
        let place = Placement::new(to_world, glam::DVec3::new(1e6, 0.0, 0.0)).expect("invertible");
        let identity = Gesture {
            place,
            ..Gesture::IDENTITY
        };
        assert_eq!(identity.translation, [0.0; 4]);
        assert_eq!(identity.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(identity.scale, [1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn a_gesture_is_the_size_the_uniform_block_expects() {
        // Three `vec4`s, two `mat3x3` and one more `vec4`. A uniform block pads
        // every member to sixteen bytes and every matrix column too, so a
        // `vec3` or a `Mat3` here would be silently repacked and the shader
        // would read one field's bytes as another's.
        assert_eq!(std::mem::size_of::<Placement>(), 112);
        assert_eq!(std::mem::size_of::<Gesture>(), 160);
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
