//! Turning a [`MeshData`] into GPU buffers.
//!
//! # Structure of arrays, split by update frequency
//!
//! The single most consequential decision here is that buffers are split by *how
//! often they change*, not by what they logically belong to. An interleaved
//! vertex buffer forces a full re-upload when only positions move, and a naive
//! design that rebuilds the render mesh whenever "anything" changes turns every
//! click on a dense mesh into a stall.
//!
//! | buffer | changes |
//! |---|---|
//! | positions, normals | every frame during a drag |
//! | indices, `tri_to_face`, `edge_verts` | only on topology change |
//! | vert/edge/face flags | only on selection change |
//!
//! Selection therefore lives in its own flag buffers and never touches geometry.
//! Getting that wrong is what makes rubber-band selection over a dense mesh feel
//! terrible.
//!
//! # Extraction is separated from upload
//!
//! [`RenderMeshCpu`] holds the CPU-side result and knows nothing about wgpu, so
//! all the interesting logic — triangulation, normal encoding, the edge list — is
//! testable without a device. [`RenderMesh`] is the thin upload on top.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use kreate_mesh::attribute::names;
use kreate_mesh::MeshData;

/// Element counts, kept alongside the buffers for draw calls and budget checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshCounts {
    pub verts: u32,
    pub edges: u32,
    pub faces: u32,
    pub triangles: u32,
}

/// Per-element flag bits, shared by the vertex, edge and face flag buffers.
///
/// One `u32` per element. Changing a selection is a write to *this* buffer only,
/// which is why selection can update in the same frame with no geometry upload.
pub mod flags {
    pub const SELECTED: u32 = 1 << 0;
    pub const ACTIVE: u32 = 1 << 1;
    pub const HIDDEN: u32 = 1 << 2;
    /// Edge only: marked sharp, so shading splits across it.
    pub const SHARP: u32 = 1 << 3;
    /// Edge only: a UV seam.
    pub const SEAM: u32 = 1 << 4;
    /// Edge only: used by exactly one face.
    pub const BOUNDARY: u32 = 1 << 5;
    /// Edge only: used by three or more faces.
    pub const NON_MANIFOLD: u32 = 1 << 6;
}

/// Encodes a unit vector into two snorm16 values, octahedron mapping.
///
/// 4 bytes instead of 12, with well under a tenth of a degree of error — far
/// below what shading can show. Normals are re-uploaded every frame during a
/// drag, so this is a third of the bandwidth on one of the two hottest buffers.
pub fn oct_encode(n: Vec3) -> u32 {
    let n = n.normalize_or_zero();
    let denom = n.x.abs() + n.y.abs() + n.z.abs();
    if denom == 0.0 {
        return 0;
    }
    let p = n / denom;
    let (mut x, mut y) = (p.x, p.y);
    if p.z <= 0.0 {
        // Fold the lower hemisphere out onto the square.
        let (px, py) = (x, y);
        x = (1.0 - py.abs()) * if px >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - px.abs()) * if py >= 0.0 { 1.0 } else { -1.0 };
    }
    let q = |v: f32| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i32 as i16) as u16 as u32;
    q(x) | (q(y) << 16)
}

/// Inverse of [`oct_encode`]. Present so the encoding can be tested, and because
/// the same maths has to be mirrored in WGSL.
pub fn oct_decode(packed: u32) -> Vec3 {
    let d = |v: u32| (v as u16 as i16) as f32 / 32767.0;
    let (ex, ey) = (d(packed & 0xFFFF), d(packed >> 16));
    let mut n = Vec3::new(ex, ey, 1.0 - ex.abs() - ey.abs());
    if n.z < 0.0 {
        let (nx, ny) = (n.x, n.y);
        n.x = (1.0 - ny.abs()) * if nx >= 0.0 { 1.0 } else { -1.0 };
        n.y = (1.0 - nx.abs()) * if ny >= 0.0 { 1.0 } else { -1.0 };
    }
    n.normalize_or_zero()
}

/// One vertex position, tightly packed.
///
/// A WGSL `array<vec3<f32>>` has a 16-byte stride, so when this buffer is bound
/// as storage for the edge overlay it is declared `array<f32>` and indexed
/// `3 * i`. That is worth the manual indexing: it is a third of the memory on the
/// largest buffer in the system.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPosition(pub [f32; 3]);

/// The CPU side of a render mesh: everything derived from a [`MeshData`],
/// laid out exactly as it will be uploaded.
#[derive(Clone, Debug, Default)]
pub struct RenderMeshCpu {
    pub positions: Vec<GpuPosition>,
    /// Octahedral-encoded vertex normals, one per vertex.
    pub normals: Vec<u32>,
    /// Triangle indices, three per triangle.
    pub indices: Vec<u32>,
    /// Source face per triangle. What lets id-buffer picking report a face
    /// rather than a triangle, and what an exporter needs to rebuild n-gons.
    pub tri_to_face: Vec<u32>,
    /// Real mesh edges, n-gon correct. The wireframe is drawn from these as
    /// instanced screen-space quads, never from the triangulation — a quad must
    /// not show its diagonal.
    pub edge_verts: Vec<[u32; 2]>,
    pub vert_flags: Vec<u32>,
    pub edge_flags: Vec<u32>,
    pub face_flags: Vec<u32>,
    pub counts: MeshCounts,
}

impl RenderMeshCpu {
    pub fn extract(mesh: &MeshData) -> Self {
        let tri = mesh.triangulate();
        let normals = mesh.compute_vertex_normals();

        let mut out = Self {
            positions: mesh
                .positions()
                .iter()
                .map(|p| GpuPosition(p.to_array()))
                .collect(),
            normals: normals.iter().map(|n| oct_encode(*n)).collect(),
            indices: tri.indices,
            tri_to_face: tri.tri_to_face,
            edge_verts: mesh.edges().to_vec(),
            vert_flags: vec![0; mesh.num_verts() as usize],
            edge_flags: vec![0; mesh.num_edges() as usize],
            face_flags: vec![0; mesh.num_faces() as usize],
            counts: MeshCounts {
                verts: mesh.num_verts(),
                edges: mesh.num_edges(),
                faces: mesh.num_faces(),
                triangles: 0,
            },
        };
        out.counts.triangles = (out.indices.len() / 3) as u32;
        out.update_flags(mesh);
        out
    }

    /// Recomputes the flag buffers from the mesh's attribute layers.
    ///
    /// Separate from [`RenderMeshCpu::extract`] because this is the cheap path: a
    /// selection change re-runs only this, and only the flag buffers are
    /// re-uploaded.
    pub fn update_flags(&mut self, mesh: &MeshData) {
        let attrs = mesh.attributes();

        if let Some(sel) = attrs.get_bool(names::SELECT_VERT) {
            for (f, s) in self.vert_flags.iter_mut().zip(sel) {
                set_bit(f, flags::SELECTED, *s);
            }
        }
        if let Some(sel) = attrs.get_bool(names::SELECT_FACE) {
            for (f, s) in self.face_flags.iter_mut().zip(sel) {
                set_bit(f, flags::SELECTED, *s);
            }
        }
        if let Some(sel) = attrs.get_bool(names::SELECT_EDGE) {
            for (f, s) in self.edge_flags.iter_mut().zip(sel) {
                set_bit(f, flags::SELECTED, *s);
            }
        }
        if let Some(sharp) = attrs.get_bool(names::SHARP_EDGE) {
            for (f, s) in self.edge_flags.iter_mut().zip(sharp) {
                set_bit(f, flags::SHARP, *s);
            }
        }
        if let Some(seam) = attrs.get_bool(names::SEAM) {
            for (f, s) in self.edge_flags.iter_mut().zip(seam) {
                set_bit(f, flags::SEAM, *s);
            }
        }

        self.mark_edge_topology(mesh);
    }

    /// Flags boundary and non-manifold edges.
    ///
    /// Worth surfacing prominently in the viewport: a non-manifold edge is
    /// usually a mistake, and one the user cannot see any other way until an
    /// operation fails on it much later.
    fn mark_edge_topology(&mut self, mesh: &MeshData) {
        let mut face_count = vec![0u32; mesh.num_edges() as usize];
        let mut index: ahash::AHashMap<(u32, u32), u32> = ahash::AHashMap::new();
        for (e, &[a, b]) in mesh.edges().iter().enumerate() {
            index.insert(if a < b { (a, b) } else { (b, a) }, e as u32);
        }
        for f in 0..mesh.num_faces() {
            let verts = mesh.face_verts(f);
            for i in 0..verts.len() {
                let (a, b) = (verts[i], verts[(i + 1) % verts.len()]);
                if a == b {
                    continue;
                }
                if let Some(&e) = index.get(&if a < b { (a, b) } else { (b, a) }) {
                    face_count[e as usize] += 1;
                }
            }
        }
        for (f, count) in self.edge_flags.iter_mut().zip(&face_count) {
            set_bit(f, flags::BOUNDARY, *count == 1);
            set_bit(f, flags::NON_MANIFOLD, *count > 2);
        }
    }

    /// Total GPU bytes this mesh will occupy.
    ///
    /// Tracked as a budget rather than left to chance: memory regressions in this
    /// domain are silent right up until they are fatal.
    pub fn gpu_bytes(&self) -> usize {
        self.positions.len() * 12
            + self.normals.len() * 4
            + self.indices.len() * 4
            + self.tri_to_face.len() * 4
            + self.edge_verts.len() * 8
            + self.vert_flags.len() * 4
            + self.edge_flags.len() * 4
            + self.face_flags.len() * 4
    }

    /// GPU bytes per vertex. The plan's budget is 100.
    pub fn bytes_per_vertex(&self) -> f32 {
        if self.counts.verts == 0 {
            0.0
        } else {
            self.gpu_bytes() as f32 / self.counts.verts as f32
        }
    }
}

#[inline]
fn set_bit(field: &mut u32, bit: u32, on: bool) {
    if on {
        *field |= bit;
    } else {
        *field &= !bit;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_mesh::attribute::{AttrData, Domain};
    use kreate_mesh::primitives;
    use std::sync::Arc;

    #[test]
    fn oct_encoding_round_trips_accurately() {
        // Sample the sphere fairly rather than trusting a few axis directions;
        // the octahedral fold has seams that axis vectors happen to miss.
        let mut worst: f32 = 0.0;
        let n = 40;
        for i in 0..n {
            for j in 0..n {
                let theta = std::f32::consts::PI * (i as f32 + 0.5) / n as f32;
                let phi = std::f32::consts::TAU * (j as f32 + 0.5) / n as f32;
                let v = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                let back = oct_decode(oct_encode(v));
                let angle = v.dot(back).clamp(-1.0, 1.0).acos().to_degrees();
                worst = worst.max(angle);
            }
        }
        assert!(
            worst < 0.1,
            "worst normal error {worst} degrees is too large"
        );
    }

    #[test]
    fn oct_encoding_handles_axis_and_degenerate_vectors() {
        for v in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            let back = oct_decode(oct_encode(v));
            assert!((back - v).length() < 1e-3, "{v} round-tripped to {back}");
        }
        // A zero normal can reach here from a degenerate face; it must not NaN.
        assert!(oct_decode(oct_encode(Vec3::ZERO)).is_finite());
    }

    #[test]
    fn oct_encoding_is_four_bytes() {
        assert_eq!(std::mem::size_of_val(&oct_encode(Vec3::Z)), 4);
    }

    #[test]
    fn extraction_counts_match_the_mesh() {
        let m = primitives::cylinder(1.0, 2.0, 16);
        let r = RenderMeshCpu::extract(&m);
        assert_eq!(r.counts.verts, m.num_verts());
        assert_eq!(r.counts.edges, m.num_edges());
        assert_eq!(r.counts.faces, m.num_faces());
        assert_eq!(r.indices.len(), r.counts.triangles as usize * 3);
        assert_eq!(r.tri_to_face.len(), r.counts.triangles as usize);
        assert!(r.tri_to_face.iter().all(|&f| f < m.num_faces()));
        assert!(r.indices.iter().all(|&v| v < m.num_verts()));
    }

    #[test]
    fn the_wireframe_uses_real_edges_not_triangulation_edges() {
        // A cube must show 12 edges. Drawing the triangulation instead would show
        // 18, with a diagonal across every face — the thing that instantly makes
        // an application look like a debug view rather than a modeller.
        let m = primitives::cube(2.0);
        let r = RenderMeshCpu::extract(&m);
        assert_eq!(r.edge_verts.len(), 12);
        assert_eq!(r.counts.triangles, 12, "6 quads triangulate to 12 tris");
    }

    #[test]
    fn positions_are_tightly_packed() {
        assert_eq!(std::mem::size_of::<GpuPosition>(), 12);
    }

    #[test]
    fn selection_flags_come_from_attribute_layers() {
        let mut m = primitives::cube(2.0);
        m.ensure_selection_layers();
        m.attributes_mut()
            .make_mut_bool(names::SELECT_VERT)
            .unwrap()[3] = true;
        m.attributes_mut()
            .make_mut_bool(names::SELECT_FACE)
            .unwrap()[1] = true;

        let r = RenderMeshCpu::extract(&m);
        assert_eq!(r.vert_flags[3] & flags::SELECTED, flags::SELECTED);
        assert_eq!(r.vert_flags[0] & flags::SELECTED, 0);
        assert_eq!(r.face_flags[1] & flags::SELECTED, flags::SELECTED);
    }

    #[test]
    fn updating_flags_does_not_disturb_geometry() {
        // The property the whole buffer split exists for: changing a selection
        // must not require re-uploading positions or indices.
        let mut m = primitives::cube(2.0);
        m.ensure_selection_layers();
        let mut r = RenderMeshCpu::extract(&m);
        let positions_before = r.positions.iter().map(|p| p.0).collect::<Vec<_>>();
        let indices_before = r.indices.clone();

        m.attributes_mut()
            .make_mut_bool(names::SELECT_VERT)
            .unwrap()[5] = true;
        r.update_flags(&m);

        assert_eq!(r.vert_flags[5] & flags::SELECTED, flags::SELECTED);
        assert_eq!(
            r.positions.iter().map(|p| p.0).collect::<Vec<_>>(),
            positions_before
        );
        assert_eq!(r.indices, indices_before);
    }

    #[test]
    fn deselecting_clears_the_bit() {
        let mut m = primitives::plane(1.0);
        m.ensure_selection_layers();
        m.attributes_mut()
            .make_mut_bool(names::SELECT_VERT)
            .unwrap()[0] = true;
        let mut r = RenderMeshCpu::extract(&m);
        assert_ne!(r.vert_flags[0] & flags::SELECTED, 0);

        m.attributes_mut()
            .make_mut_bool(names::SELECT_VERT)
            .unwrap()[0] = false;
        r.update_flags(&m);
        assert_eq!(
            r.vert_flags[0] & flags::SELECTED,
            0,
            "the bit must clear, not stick"
        );
    }

    #[test]
    fn open_meshes_have_boundary_edges_and_closed_ones_do_not() {
        let plane = RenderMeshCpu::extract(&primitives::plane(1.0));
        assert_eq!(
            plane
                .edge_flags
                .iter()
                .filter(|f| *f & flags::BOUNDARY != 0)
                .count(),
            4,
            "every edge of a single quad is a boundary"
        );

        let cube = RenderMeshCpu::extract(&primitives::cube(1.0));
        assert!(
            cube.edge_flags.iter().all(|f| f & flags::BOUNDARY == 0),
            "a closed cube has no boundary edges"
        );
        assert!(cube.edge_flags.iter().all(|f| f & flags::NON_MANIFOLD == 0));
    }

    #[test]
    fn non_manifold_edges_are_flagged() {
        // Three quads sharing one edge: legal to store, worth showing loudly.
        let m = MeshData::from_faces(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
            ],
            &[vec![0, 1, 3, 2], vec![0, 1, 5, 4], vec![0, 1, 7, 6]],
        );
        let r = RenderMeshCpu::extract(&m);
        let shared = m
            .edges()
            .iter()
            .position(|e| *e == [0, 1])
            .expect("shared edge");
        assert_ne!(r.edge_flags[shared] & flags::NON_MANIFOLD, 0);
        assert_eq!(r.edge_flags[shared] & flags::BOUNDARY, 0);
    }

    #[test]
    fn sharp_edges_reach_the_gpu_flags() {
        let mut m = primitives::cube(1.0);
        let n = m.num_edges() as usize;
        let mut sharp = vec![false; n];
        sharp[2] = true;
        m.attributes_mut().insert(
            names::SHARP_EDGE,
            Domain::Edge,
            AttrData::Bool(Arc::new(sharp)),
        );
        let r = RenderMeshCpu::extract(&m);
        assert_ne!(r.edge_flags[2] & flags::SHARP, 0);
        assert_eq!(r.edge_flags[0] & flags::SHARP, 0);
    }

    #[test]
    fn gpu_memory_stays_within_the_per_vertex_budget() {
        // The plan's budget is 100 bytes per render vertex. Asserted here so a
        // regression fails the build rather than being noticed as an out-of-memory
        // report on an 8 GB card.
        for (name, mesh) in [
            ("grid", primitives::grid(10.0, 40, 40)),
            ("sphere", primitives::uv_sphere(1.0, 48, 24)),
            ("torus", primitives::torus(1.0, 0.3, 48, 24)),
            ("icosphere", primitives::ico_sphere(1.0, 4)),
        ] {
            let r = RenderMeshCpu::extract(&mesh);
            assert!(
                r.bytes_per_vertex() <= 100.0,
                "{name}: {:.1} bytes/vertex exceeds the 100 byte budget",
                r.bytes_per_vertex()
            );
        }
    }

    #[test]
    fn an_empty_mesh_extracts_without_panicking() {
        let r = RenderMeshCpu::extract(&MeshData::empty());
        assert_eq!(r.counts, MeshCounts::default());
        assert_eq!(r.bytes_per_vertex(), 0.0);
        assert!(r.indices.is_empty());
    }
}
