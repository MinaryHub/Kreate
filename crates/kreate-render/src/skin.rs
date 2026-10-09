//! Skinning on the GPU: a posed mesh without re-uploading it.
//!
//! §3-8's decision. The CPU skins in a few milliseconds even at two hundred
//! thousand vertices; what a pose change cost was turning the result into GPU
//! buffers again — the triangulation, the vertex split, the upload — tens of
//! milliseconds for work the pose does not change. So a skinned object keeps
//! one GPU mesh made once from its rest shape, and a pose is the bones'
//! matrices: forty-eight bytes a bone, then three compute passes.
//!
//! The first pass is `skin.wgsl`; the two after it are the drag preview's
//! normal passes, bound to this mesh's buffers. A skinned surface is shaded
//! with normals from its skinned positions by the same arithmetic a dragged
//! one is — and with the same known limit: the vertex split is the rest
//! shape's, so a crease a pose makes is shaded smooth.
//!
//! The renderer never sees an armature. The influences arrive as plain tables,
//! and the matrices as rows; what bone a row is, is the caller's business.

use wgpu::util::DeviceExt;

use crate::deform::{bind_entry, pad_if_empty, topology_group, workgroups_for, Deformer, Gesture};
use crate::mesh_gpu::RenderMeshCpu;
use crate::viewport::GpuMesh;

/// Which bones weigh on each mesh vertex, and how much: CSR over the mesh's
/// vertices. Vertex `v`'s influences are `bones[offsets[v]..offsets[v + 1]]`
/// with the matching `weights`. Weights need not sum to one; the pass divides
/// by their sum, as the CPU does.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Influences {
    pub offsets: Vec<u32>,
    pub bones: Vec<u32>,
    pub weights: Vec<f32>,
}

/// A bone as the shader reads it: the matrix's three rows (the translation
/// last in each), then its rotation and translation as a unit dual
/// quaternion (real, dual), then the rows of what is left — scale and shear —
/// for dual-quaternion skinning. `kreate_scene::skin::DualBone`, in `f32`.
pub type BoneRows = [f32; 32];

/// `m` as the shader reads it.
pub fn bone_rows(m: glam::Affine3A) -> BoneRows {
    let r = glam::Mat4::from(m).transpose();
    let (a, b, c) = (r.x_axis, r.y_axis, r.z_axis);
    let (_, q, t) = m.to_scale_rotation_translation();
    let q = q.normalize();
    let dual = glam::Quat::from_xyzw(t.x, t.y, t.z, 0.0) * q * 0.5;
    let left = (glam::Mat3::from_quat(q.inverse()) * glam::Mat3::from(m.matrix3)).transpose();
    let (s0, s1, s2) = (left.x_axis, left.y_axis, left.z_axis);
    [
        a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, //
        q.x, q.y, q.z, q.w, dual.x, dual.y, dual.z, dual.w, //
        s0.x, s0.y, s0.z, 0.0, s1.x, s1.y, s1.z, 0.0, s2.x, s2.y, s2.z, 0.0,
    ]
}

/// One skinned mesh's GPU state: the rest copy, the tables, the bones.
pub struct Skin {
    skin: wgpu::BindGroup,
    live: wgpu::BindGroup,
    topology: wgpu::BindGroup,
    bones: wgpu::Buffer,
    mode: wgpu::Buffer,
    bone_count: u32,
    render_verts: u32,
    faces: u32,
}

impl Skin {
    /// Prepares `mesh`, uploaded from `cpu`, to be skinned. `cpu` is the rest
    /// shape's render mesh; its positions are copied once and never written.
    ///
    /// `None` for a mesh with no vertices, or with influences that do not
    /// cover its vertices, or no bones at all.
    pub fn new(
        device: &wgpu::Device,
        deformer: &Deformer,
        mesh: &GpuMesh,
        cpu: &RenderMeshCpu,
        influences: &Influences,
        bone_count: u32,
    ) -> Option<Self> {
        let render_verts = cpu.positions.len() as u32;
        let mesh_verts = cpu
            .render_to_vert
            .iter()
            .copied()
            .max()
            .map_or(0, |v| v + 1);
        if render_verts == 0
            || bone_count == 0
            || (influences.offsets.len() as u32) < mesh_verts + 1
            || influences.bones.len() != influences.weights.len()
        {
            return None;
        }
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let rest_positions = init(
            "skin rest positions",
            bytemuck::cast_slice(&cpu.positions),
            storage,
        );
        let rest_normals = init(
            "skin rest normals",
            bytemuck::cast_slice(&cpu.normals),
            storage,
        );
        let render_to_vert = init(
            "skin render to vert",
            bytemuck::cast_slice(&cpu.render_to_vert),
            storage,
        );
        let offsets = init(
            "skin offsets",
            bytemuck::cast_slice(&influences.offsets),
            storage,
        );
        let bones_of = init(
            "skin influence bones",
            bytemuck::cast_slice(pad_if_empty(&influences.bones)),
            storage,
        );
        let weights: &[f32] = if influences.weights.is_empty() {
            &[0.0]
        } else {
            &influences.weights
        };
        let weights_of = init(
            "skin influence weights",
            bytemuck::cast_slice(weights),
            storage,
        );
        let identity: Vec<BoneRows> =
            vec![bone_rows(glam::Affine3A::IDENTITY); bone_count as usize];
        let bones = init(
            "skin bones",
            bytemuck::cast_slice(&identity),
            storage | wgpu::BufferUsages::COPY_DST,
        );
        let mode = init(
            "skin mode",
            bytemuck::cast_slice(&[0u32; 4]),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        // The normal passes' own inputs: every vertex fully "weighted", which
        // is what their bounds check reads the vertex count from, and a
        // gesture they never look at.
        let ones = init(
            "skin normal weights",
            bytemuck::cast_slice(&vec![1.0f32; render_verts as usize]),
            storage,
        );
        let gesture = init(
            "skin gesture",
            bytemuck::bytes_of(&Gesture::IDENTITY),
            wgpu::BufferUsages::UNIFORM,
        );
        let skin = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin"),
            layout: &deformer.skin_layout,
            entries: &[
                bind_entry(0, &rest_positions),
                bind_entry(1, mesh.positions()),
                bind_entry(2, &render_to_vert),
                bind_entry(3, &offsets),
                bind_entry(4, &bones_of),
                bind_entry(5, &weights_of),
                bind_entry(6, &bones),
                bind_entry(7, &mode),
            ],
        });
        let live = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin normals"),
            layout: &deformer.live,
            entries: &[
                bind_entry(0, &rest_positions),
                bind_entry(1, &rest_normals),
                bind_entry(2, mesh.positions()),
                bind_entry(3, mesh.normals()),
                bind_entry(4, &ones),
                bind_entry(5, &gesture),
            ],
        });
        Some(Self {
            skin,
            live,
            topology: topology_group(device, deformer, cpu),
            bones,
            mode,
            bone_count,
            render_verts,
            faces: cpu.counts.faces,
        })
    }

    /// How many bones this was made for.
    pub fn bone_count(&self) -> u32 {
        self.bone_count
    }

    /// Sets the pose. Refused, and nothing written, when the count is not the
    /// one this was made for — a short write would leave the last bones
    /// wherever the previous pose put them.
    #[must_use]
    pub fn set_bones(&self, queue: &wgpu::Queue, bones: &[BoneRows]) -> bool {
        if bones.len() as u32 != self.bone_count {
            return false;
        }
        queue.write_buffer(&self.bones, 0, bytemuck::cast_slice(bones));
        true
    }

    /// Dual quaternions (Blender's Preserve Volume) or the matrices' linear
    /// blend, from the next dispatch on.
    pub fn set_preserve_volume(&self, queue: &wgpu::Queue, on: bool) {
        queue.write_buffer(&self.mode, 0, bytemuck::cast_slice(&[on as u32, 0, 0, 0]));
    }

    /// Writes the skinned positions and their normals into the mesh's buffers.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, deformer: &Deformer) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("skin positions"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&deformer.skin);
            pass.set_bind_group(0, &self.skin, &[]);
            pass.dispatch_workgroups(workgroups_for(self.render_verts), 1, 1);
        }
        for (label, pipeline, items) in [
            ("skin face normals", &deformer.face_normals, self.faces),
            (
                "skin vertex normals",
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
            pass.dispatch_workgroups(workgroups_for(items), 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_the_matrix_read_across() {
        let m = glam::Affine3A::from_scale_rotation_translation(
            glam::Vec3::new(2.0, 3.0, 4.0),
            glam::Quat::from_rotation_z(0.7),
            glam::Vec3::new(5.0, -6.0, 7.0),
        );
        let r = bone_rows(m);
        let p = glam::Vec3::new(0.3, -1.1, 2.5);
        let by_rows = glam::Vec3::new(
            r[0] * p.x + r[1] * p.y + r[2] * p.z + r[3],
            r[4] * p.x + r[5] * p.y + r[6] * p.z + r[7],
            r[8] * p.x + r[9] * p.y + r[10] * p.z + r[11],
        );
        assert!((by_rows - m.transform_point3(p)).length() < 1e-5);
    }
}
