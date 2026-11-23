//! A mesh whose vertices moved and nothing else — a sculpt stroke — shown
//! without extracting it again.
//!
//! Extracting a render mesh is the triangulation, the smoothing groups and
//! the vertex split: some eighty milliseconds at a hundred and thirty
//! thousand vertices, every frame of a stroke, for work the stroke does not
//! change. So the GPU mesh keeps everything it was uploaded with, the CPU
//! writes the new positions into it — each mesh vertex to the render
//! vertices made from it — and the drag preview's two normal passes shade
//! the result.
//!
//! The same known limit as the drag preview and the skin: the vertex split is
//! the one the mesh was uploaded with, so a crease the stroke makes is shaded
//! smooth until the mesh is extracted again. The caller does that when the
//! stroke ends.

use wgpu::util::DeviceExt;

use crate::deform::{bind_entry, topology_group, workgroups_for, Deformer, Gesture};
use crate::mesh_gpu::{GpuPosition, RenderMeshCpu};
use crate::viewport::GpuMesh;

/// Render vertices per dirty-check chunk (`1024 * 12 = 12 KiB`).
///
/// Small enough that a localised or X-mirrored brush dab uploads a few tens of
/// kilobytes rather than twelve megabytes on a million-vertex sculpt, and large
/// enough that contiguous strokes merge into one transfer.
const CHUNK_VERTS: usize = 1024;

/// Returns contiguous `[start..end)` render-vertex ranges whose mapped mesh
/// positions differ from `resident`, updating `resident` in place.
fn update_dirty_chunks(
    resident: &mut [GpuPosition],
    render_to_vert: &[u32],
    positions: &[glam::Vec3],
) -> Vec<std::ops::Range<usize>> {
    let total = resident.len();
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let mut chunk_start = 0;
    while chunk_start < total {
        let chunk_end = (chunk_start + CHUNK_VERTS).min(total);
        let mut dirty = false;
        for (slot, &v) in resident[chunk_start..chunk_end]
            .iter_mut()
            .zip(&render_to_vert[chunk_start..chunk_end])
        {
            let next = positions[v as usize].to_array();
            if slot.0[0].to_bits() != next[0].to_bits()
                || slot.0[1].to_bits() != next[1].to_bits()
                || slot.0[2].to_bits() != next[2].to_bits()
            {
                slot.0 = next;
                dirty = true;
            }
        }
        if dirty {
            if let Some(last) = ranges.last_mut().filter(|r| r.end == chunk_start) {
                last.end = chunk_end;
            } else {
                ranges.push(chunk_start..chunk_end);
            }
        }
        chunk_start = chunk_end;
    }
    ranges
}

/// One mesh's reshaping state: which mesh vertex each render vertex is, and
/// the normal passes bound to the mesh's buffers.
pub struct Reshape {
    live: wgpu::BindGroup,
    topology: wgpu::BindGroup,
    render_to_vert: Vec<u32>,
    mesh_verts: usize,
    faces: u32,
    resident: Vec<GpuPosition>,
    last_bytes: usize,
}

impl Reshape {
    /// Prepares `mesh`, uploaded from `cpu`. `None` for a mesh with no
    /// vertices.
    pub fn new(
        device: &wgpu::Device,
        deformer: &Deformer,
        mesh: &GpuMesh,
        cpu: &RenderMeshCpu,
    ) -> Option<Self> {
        let render_verts = cpu.positions.len();
        if render_verts == 0 || cpu.render_to_vert.len() != render_verts {
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
        // The passes never read the rest positions; the layout binds them
        // because the drag's position pass does.
        let unused = init("reshape unused", &[0u8; 16], storage);
        // Read where a vertex's neighbourhood has collapsed to nothing: the
        // normal it was uploaded with.
        let rest_normals = init(
            "reshape rest normals",
            bytemuck::cast_slice(&cpu.normals),
            storage,
        );
        // Every vertex fully weighted: the bounds check reads the count from it.
        let ones = init(
            "reshape weights",
            bytemuck::cast_slice(&vec![1.0f32; render_verts]),
            storage,
        );
        let gesture = init(
            "reshape gesture",
            bytemuck::bytes_of(&Gesture::IDENTITY),
            wgpu::BufferUsages::UNIFORM,
        );
        let live = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reshape normals"),
            layout: &deformer.live,
            entries: &[
                bind_entry(0, &unused),
                bind_entry(1, &rest_normals),
                bind_entry(2, mesh.positions()),
                bind_entry(3, mesh.normals()),
                bind_entry(4, &ones),
                bind_entry(5, &gesture),
            ],
        });
        Some(Self {
            live,
            topology: topology_group(device, deformer, cpu),
            mesh_verts: cpu
                .render_to_vert
                .iter()
                .copied()
                .max()
                .map_or(0, |v| v as usize + 1),
            render_to_vert: cpu.render_to_vert.clone(),
            faces: cpu.counts.faces,
            resident: cpu.positions.clone(),
            last_bytes: 0,
        })
    }

    /// Bytes uploaded to the GPU in the most recent [`Reshape::write`] call.
    pub fn last_uploaded_bytes(&self) -> usize {
        self.last_bytes
    }

    /// Writes the mesh's changed vertex positions — one per *mesh* vertex, as
    /// `MeshData::positions` has them — into the render vertices made from
    /// them, uploading only dirty 1,024-vertex chunks. Refused, and nothing
    /// written, when there are too few.
    #[must_use]
    pub fn write(&mut self, queue: &wgpu::Queue, mesh: &GpuMesh, positions: &[glam::Vec3]) -> bool {
        if positions.len() < self.mesh_verts {
            return false;
        }
        let ranges = update_dirty_chunks(&mut self.resident, &self.render_to_vert, positions);
        let stride = std::mem::size_of::<GpuPosition>();
        let mut uploaded = 0usize;
        for range in ranges {
            let offset = (range.start * stride) as u64;
            let bytes: &[u8] = bytemuck::cast_slice(&self.resident[range]);
            queue.write_buffer(mesh.positions(), offset, bytes);
            uploaded += bytes.len();
        }
        self.last_bytes = uploaded;
        true
    }

    /// Recomputes the normals from the positions last written.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, deformer: &Deformer) {
        for (label, pipeline, items) in [
            ("reshape face normals", &deformer.face_normals, self.faces),
            (
                "reshape vertex normals",
                &deformer.vertex_normals,
                self.render_to_vert.len() as u32,
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
    fn mirrored_sculpt_dab_uploads_only_touched_chunks() {
        let n = 8192usize;
        let mut resident: Vec<GpuPosition> =
            (0..n).map(|i| GpuPosition([i as f32, 0.0, 0.0])).collect();
        let render_to_vert: Vec<u32> = (0..n as u32).collect();
        let mut positions: Vec<glam::Vec3> = (0..n)
            .map(|i| glam::Vec3::new(i as f32, 0.0, 0.0))
            .collect();

        assert!(update_dirty_chunks(&mut resident, &render_to_vert, &positions).is_empty());

        // Touch two distant vertices (simulating X-mirror sculpt dabs at opposite ends).
        positions[10].y = 1.5;
        positions[7900].y = 1.5;
        let ranges = update_dirty_chunks(&mut resident, &render_to_vert, &positions);
        assert_eq!(
            ranges,
            vec![0..CHUNK_VERTS, 7 * CHUNK_VERTS..8 * CHUNK_VERTS]
        );

        // Adjacent dirty chunks coalesce into one transfer.
        positions[CHUNK_VERTS + 5].z = -2.0;
        positions[12].z = -2.0;
        let coalesced = update_dirty_chunks(&mut resident, &render_to_vert, &positions);
        assert_eq!(coalesced, vec![0..2 * CHUNK_VERTS]);
    }
}
