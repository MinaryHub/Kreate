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
use kreate_math::Aabb;
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

impl MeshCounts {
    /// Whether these describe the same number of vertices, edges and faces.
    ///
    /// Deliberately not the derived equality: the triangle count is a property
    /// of the *triangulation*, and the flag buffers are sized by element counts
    /// alone. Something that read only the flags off a mesh never triangulated
    /// it and has nothing to report there, and demanding equality would refuse
    /// every such write.
    pub fn same_elements(&self, other: &MeshCounts) -> bool {
        self.verts == other.verts && self.edges == other.edges && self.faces == other.faces
    }
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
/// The per-element flag buffers, and nothing else.
///
/// Separate from [`RenderMeshCpu`] because a selection click needs *only these*,
/// and going through the render mesh to get them was the most expensive thing a
/// click did: **922 ms on a million faces, measured** — triangulating,
/// computing corner normals and splitting vertices, all of it discarded, and
/// none of it changed by a click.
///
/// Which is this module's own warning arrived at from the other side. The
/// *buffers* were split by update frequency so that a click would write three
/// small ones; what was never split was the extraction that fills them.
pub struct ElementFlags {
    pub vert_flags: Vec<u32>,
    pub edge_flags: Vec<u32>,
    pub face_flags: Vec<u32>,
    /// The element counts these were sized from, so a caller can refuse to
    /// write them into buffers built for a different mesh.
    pub counts: MeshCounts,
}

impl ElementFlags {
    /// Reads the selection and the edge marks off a mesh.
    pub fn extract(mesh: &MeshData) -> Self {
        let mut flags = Self {
            vert_flags: vec![0; mesh.num_verts() as usize],
            edge_flags: vec![0; mesh.num_edges() as usize],
            face_flags: vec![0; mesh.num_faces() as usize],
            counts: MeshCounts {
                verts: mesh.num_verts(),
                edges: mesh.num_edges(),
                faces: mesh.num_faces(),
                // Not knowable without triangulating, and nothing that reads
                // flags needs it. Zero rather than guessed, which is why the
                // guard on a flag write is `same_elements` and not equality.
                triangles: 0,
            },
        };
        flags.read(mesh);
        flags
    }

    /// Fills the flags from the mesh's attribute layers.
    fn read(&mut self, mesh: &MeshData) {
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
        // What edit mode has hidden: the overlays skip it. The faces
        // themselves are left out of the triangles, see `group_by_material`.
        for (layer, flags) in [
            (names::HIDE_VERT, &mut self.vert_flags),
            (names::HIDE_EDGE, &mut self.edge_flags),
            (names::HIDE_FACE, &mut self.face_flags),
        ] {
            if let Some(hidden) = attrs.get_bool(layer) {
                for (f, h) in flags.iter_mut().zip(hidden) {
                    set_bit(f, flags::HIDDEN, *h);
                }
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
    ///
    /// Counted through `corner_edges`, which already answers "which edge does
    /// this corner walk along" as a counting sort. The version before built an
    /// `AHashMap` of every edge keyed on its endpoints and then probed it once
    /// per corner — two million inserts and four million lookups on a million
    /// faces, to recover a mapping the mesh can produce without hashing
    /// anything.
    fn mark_edge_topology(&mut self, mesh: &MeshData) {
        let mut face_count = vec![0u32; mesh.num_edges() as usize];
        for edge in mesh.corner_edges() {
            // `u32::MAX` is a corner with no edge — a face that repeats a
            // vertex, which the validator rejects but an operator may produce
            // in an intermediate state.
            if let Some(slot) = face_count.get_mut(edge as usize) {
                *slot += 1;
            }
        }
        for (f, count) in self.edge_flags.iter_mut().zip(&face_count) {
            set_bit(f, flags::BOUNDARY, *count == 1);
            set_bit(f, flags::NON_MANIFOLD, *count > 2);
        }
    }
}

/// One run of triangles that share a material slot.
///
/// **Why the render mesh carries these and the document does not.** A face's
/// `material_index` is per face, and a GPU draw call is a *range* — so
/// somewhere the faces have to be grouped into runs. Doing it here, once per
/// topology change, is the only place it is cheap: the alternative is a
/// per-frame sort, or a draw call per face.
///
/// A mesh with no `material_index` layer, which is most of them while a model
/// is being built, gets exactly one of these covering everything at slot zero.
/// That keeps the render path identical for both cases — the loop over
/// submeshes is the only loop, rather than a special case beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Submesh {
    /// The mesh's material slot, which the object resolves to a material.
    pub slot: u32,
    /// Where this run starts in `indices`, in indices and not in triangles.
    pub start: u32,
    /// How many indices long it is. Always a multiple of three.
    pub count: u32,
}

pub struct RenderMeshCpu {
    pub positions: Vec<GpuPosition>,
    /// Octahedral-encoded vertex normals, one per vertex.
    pub normals: Vec<u32>,
    /// Surface coordinates, one per vertex.
    ///
    /// Zero for a mesh with no UV layer, which is most of them while a model is
    /// being built. Present anyway, because the vertex layout belongs to the
    /// pipeline and a pipeline per "has UVs or not" would double the shaders a
    /// scene compiles — the same reason a material's textures are one array
    /// binding rather than one binding each.
    ///
    /// Full precision rather than half. Sixteen-bit floats hold a UV to about
    /// one part in two thousand, which is two texels on a 4K map — visible as a
    /// slight crawl, and worse once §2.3's texture painting is writing through
    /// these coordinates rather than only reading.
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices, three per triangle.
    pub indices: Vec<u32>,
    /// Source face per triangle. What lets id-buffer picking report a face
    /// rather than a triangle, and what an exporter needs to rebuild n-gons.
    ///
    /// In the same order as `indices`, which matters now that the triangles
    /// are grouped by material: the id and overlay passes are non-indexed
    /// draws whose `vertex_index` counts corners, so they read this at
    /// `vi / 3` and would report the wrong face if the two ever disagreed.
    pub tri_to_face: Vec<u32>,
    /// The runs of triangles that share a material slot, in index order.
    ///
    /// Covers `indices` exactly: every triangle is in one run and the runs are
    /// contiguous. A mesh with no material layer has a single run at slot zero.
    pub submeshes: Vec<Submesh>,
    /// Real mesh edges, n-gon correct. The wireframe is drawn from these as
    /// instanced screen-space quads, never from the triangulation — a quad must
    /// not show its diagonal.
    pub edge_verts: Vec<[u32; 2]>,
    /// One render vertex per *mesh* vertex, for anything drawn per mesh vertex.
    ///
    /// Splitting for normals means a cube corner is three render vertices, so
    /// "draw a dot per vertex" would draw three dots on top of each other and,
    /// worse, report three different ids for one thing the user thinks of as one
    /// vertex. Indexing through this keeps the vertex point overlay and the id
    /// pass talking about mesh vertices, which is what selection is expressed
    /// in. Which of the split copies is named is irrelevant: they differ only in
    /// their normal.
    pub vert_to_render: Vec<u32>,
    // --- what a drag needs, and only a drag ------------------------------
    //
    // The four fields below describe the mesh in the terms a deformation is
    // expressed in. They are **not** part of what a resident mesh costs on the
    // GPU: `upload` does not send them, `gpu_bytes` does not count them, and
    // the budget test would fail if it did — it caught exactly that, at 116
    // bytes a vertex against a hundred. A deform session uploads them for the
    // one mesh being dragged and drops them when the drag ends; see
    // `deform_bytes` for that cost and why it is affordable.
    /// Which *mesh* vertex each render vertex came from.
    ///
    /// The other direction from `vert_to_render`, and not derivable from it: a
    /// mesh vertex can have several render vertices and that field names only
    /// one. A drag needs this one, because the weight it applies is a property
    /// of the mesh vertex and the buffer it writes is indexed by render vertex.
    pub render_to_vert: Vec<u32>,
    /// Where each face's corners begin, as CSR over `corner_render`.
    ///
    /// A copy of the mesh's own `face_offsets`, because a deform session walks
    /// faces on the GPU and cannot reach the `MeshData` it came from.
    pub face_offsets: Vec<u32>,
    /// Which faces each render vertex averages, as CSR: render vertex `v` owns
    /// `group_faces[group_offsets[v] .. group_offsets[v + 1]]`.
    ///
    /// Its *smooth group*, not every face at its vertex, and not only the
    /// corners that resolved to it.
    ///
    /// A render vertex is a (vertex, smooth group, surface coordinate) triple:
    /// it splits where shading breaks *and* where a texture seams. Only the
    /// first of those is about normals, so the group is the faces sharing the
    /// normal — which is exactly the set `compute_corner_normals` sums, and a
    /// GPU pass over it produces the same normals the CPU would.
    ///
    /// Two render vertices either side of a seam hold the same list. That
    /// costs a copy of it, and buys a sphere that does not crease along the
    /// line where its texture wraps when somebody drags it.
    pub group_offsets: Vec<u32>,
    pub group_faces: Vec<u32>,
    /// Which render vertex each mesh corner resolves to.
    ///
    /// What lets a face be walked in render-vertex terms, which is what
    /// recomputing a face normal from deformed positions needs.
    pub corner_render: Vec<u32>,
    pub vert_flags: Vec<u32>,
    pub edge_flags: Vec<u32>,
    pub face_flags: Vec<u32>,
    pub counts: MeshCounts,
    /// The cluster decomposition, when one was asked for.
    ///
    /// Optional, and that is a memory decision rather than a stylistic one. The
    /// cluster index stream is a second copy of every index — the direct path
    /// needs `indices`, the cluster path needs its own order — so baking
    /// unconditionally roughly doubles the largest buffer in the system for a
    /// path that is off by default and currently measures slower. See
    /// [`Self::with_clusters`].
    pub clusters: Option<crate::cluster::ClusterMesh>,
    /// The mesh's own bounding box, in object space.
    ///
    /// Computed here rather than on demand because this is the one place that
    /// already walks every position, and because the screen-size decision runs
    /// per draw per frame: a bound recomputed there would make the optimisation
    /// cost more than it saves.
    pub bounds: Aabb,
}

/// Angle beyond which shading splits across an edge.
///
/// 30 degrees is the long-standing default across modelling tools: it keeps a
/// moderately tessellated curve smooth while leaving anything that reads as a
/// crease sharp.
pub const DEFAULT_AUTO_SMOOTH: f32 = std::f32::consts::PI / 6.0;

/// Sorts triangles into one contiguous run per material slot.
///
/// Returns the reordered indices, the matching `tri_to_face`, and the runs.
/// A mesh with no `material_index` layer comes back untouched with a single
/// run, which is both the common case and the cheap one: the check is one
/// attribute lookup and the sort is skipped entirely.
fn group_by_material(
    mesh: &MeshData,
    indices: Vec<u32>,
    tri_to_face: Vec<u32>,
) -> (Vec<u32>, Vec<u32>, Vec<Submesh>) {
    // Hidden faces have no triangles: not drawn, not shaded, not clickable.
    // Taken out here, before the regrouping, so every pass that reads the
    // index buffer and `tri_to_face` agrees about it — and the element
    // numbering the flag buffers use is untouched.
    let (indices, tri_to_face) = match mesh.attributes().get_bool(names::HIDE_FACE) {
        Some(hidden) if hidden.iter().any(|h| *h) => {
            let mut kept_indices = Vec::with_capacity(indices.len());
            let mut kept_faces = Vec::with_capacity(tri_to_face.len());
            for (t, face) in tri_to_face.iter().enumerate() {
                if !hidden.get(*face as usize).copied().unwrap_or(false) {
                    kept_indices.extend_from_slice(&indices[t * 3..t * 3 + 3]);
                    kept_faces.push(*face);
                }
            }
            (kept_indices, kept_faces)
        }
        _ => (indices, tri_to_face),
    };
    let triangles = tri_to_face.len();
    let whole = |slot: u32, indices: &[u32]| {
        vec![Submesh {
            slot,
            start: 0,
            count: indices.len() as u32,
        }]
    };
    let Some(face_material) = mesh.attributes().get_i32(names::MATERIAL_INDEX) else {
        let one = whole(0, &indices);
        return (indices, tri_to_face, one);
    };

    // A negative index means "no material", which glTF writes and the exporter
    // reads back. It resolves to slot zero, the same place a mesh with no
    // layer at all lands, so that "unassigned" is one state and not two.
    let slot_of = |t: usize| -> u32 {
        let face = tri_to_face[t] as usize;
        face_material
            .get(face)
            .map(|m| (*m).max(0) as u32)
            .unwrap_or(0)
    };

    // The overwhelmingly common case: every face on one slot. Worth its own
    // arm because it is every mesh in a scene that has not been given a second
    // material, and it turns the whole of this into one pass over the faces.
    let first = if triangles == 0 { 0 } else { slot_of(0) };
    if (0..triangles).all(|t| slot_of(t) == first) {
        let one = whole(first, &indices);
        return (indices, tri_to_face, one);
    }

    let mut order: Vec<usize> = (0..triangles).collect();
    order.sort_by_key(|t| slot_of(*t));

    let mut out_indices = Vec::with_capacity(indices.len());
    let mut out_faces = Vec::with_capacity(triangles);
    let mut submeshes: Vec<Submesh> = Vec::new();
    for t in order {
        let slot = slot_of(t);
        match submeshes.last_mut() {
            Some(run) if run.slot == slot => run.count += 3,
            _ => submeshes.push(Submesh {
                slot,
                start: out_indices.len() as u32,
                count: 3,
            }),
        }
        out_indices.extend_from_slice(&indices[t * 3..t * 3 + 3]);
        out_faces.push(tri_to_face[t]);
    }
    (out_indices, out_faces, submeshes)
}

impl RenderMeshCpu {
    pub fn extract(mesh: &MeshData) -> Self {
        Self::extract_with_auto_smooth(mesh, DEFAULT_AUTO_SMOOTH)
    }

    /// Builds the render mesh, splitting vertices where shading must break.
    ///
    /// Normals are per *corner*, so a cube's corner has three different normals —
    /// one per face. A GPU vertex carries exactly one, which means the render
    /// mesh has to split those into separate vertices. That is allowed and
    /// expected: `MeshData` keeps the topology the user authored, and this is a
    /// derived form built for the rasteriser. Splitting here rather than there is
    /// what lets a cube stay eight vertices in the document while shading
    /// correctly on screen.
    pub fn extract_with_auto_smooth(mesh: &MeshData, angle_threshold: f32) -> Self {
        let tri = mesh.triangulate_corners();
        let corner_normals = mesh.compute_corner_normals(angle_threshold);
        // Per corner, like the normals: a seam is two coordinates at one
        // position, which is what a seam *is*.
        let corner_uvs = mesh.attributes().get_f32x2(names::UV);
        let corner_verts = mesh.corner_verts();
        let mesh_positions = mesh.positions();
        let nverts = mesh.num_verts() as usize;

        // Every corner's normal, packed once. It was packed per *triangle
        // corner* before, so a quad's shared diagonal packed the same normal
        // twice and an n-gon fanned into k triangles packed it k times.
        let packed: Vec<u32> = corner_normals.iter().map(|n| oct_encode(*n)).collect();

        // Corners grouped by their vertex, by counting sort. This replaces a
        // hash map keyed on `(vertex, packed normal)` that was probed once per
        // triangle corner — six million probes on a million faces, and the
        // dominant cost of the whole extraction. Within a vertex the distinct
        // normals are found by scanning its own corners, of which there are as
        // many as the vertex has faces: four, usually.
        let mut offsets = vec![0u32; nverts + 1];
        for v in corner_verts {
            offsets[*v as usize + 1] += 1;
        }
        for v in 0..nverts {
            offsets[v + 1] += offsets[v];
        }
        let mut cursor = offsets.clone();
        let mut incident = vec![0u32; corner_verts.len()];
        for (corner, v) in corner_verts.iter().enumerate() {
            let slot = &mut cursor[*v as usize];
            incident[*slot as usize] = corner as u32;
            *slot += 1;
        }

        // Deduplicate by (mesh vertex, encoded normal, surface coordinate). A
        // smooth mesh collapses back to one vertex each; a hard-surface one
        // splits where its shading breaks, and a textured one splits again
        // along its seams.
        //
        // The coordinate is compared as bits, not as floats. Exact is what is
        // wanted: two corners either carry the same coordinate and share a
        // vertex, or they do not, and an epsilon would weld a seam somebody
        // placed one texel apart. The same rule the glTF exporter follows for
        // the same reason.
        let mut positions = Vec::with_capacity(nverts);
        let mut normals = Vec::with_capacity(nverts);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(nverts);
        let mut render_to_vert: Vec<u32> = Vec::with_capacity(nverts);
        let mut corner_to_render = vec![u32::MAX; corner_verts.len()];
        // One representative render vertex per mesh vertex, so the overlay
        // buffers — which index mesh vertices — still resolve to a position.
        let mut representative = vec![u32::MAX; nverts];
        // The distinct normals of the vertex being worked on. Reused rather than
        // reallocated per vertex, which at a million vertices is a million
        // allocations for a list that never exceeds a vertex's face count.
        let mut distinct: Vec<(u32, [u32; 2], u32)> = Vec::new();

        for v in 0..nverts {
            distinct.clear();
            for &corner in &incident[offsets[v] as usize..offsets[v + 1] as usize] {
                let normal = packed[corner as usize];
                let uv = corner_uvs
                    .map(|u| u[corner as usize])
                    .unwrap_or(glam::Vec2::ZERO);
                let key = [uv.x.to_bits(), uv.y.to_bits()];
                let index = match distinct.iter().find(|(n, k, _)| *n == normal && *k == key) {
                    Some((_, _, index)) => *index,
                    None => {
                        positions.push(GpuPosition(mesh_positions[v].to_array()));
                        normals.push(normal);
                        uvs.push([uv.x, uv.y]);
                        render_to_vert.push(v as u32);
                        let index = (positions.len() - 1) as u32;
                        distinct.push((normal, key, index));
                        index
                    }
                };
                corner_to_render[corner as usize] = index;
            }
            // Which of the split copies is named is irrelevant: they differ only
            // in their normal and their surface coordinate.
            if let Some((_, _, first)) = distinct.first() {
                representative[v] = *first;
            }
        }

        // The index buffer, now a lookup rather than a search. Corners of a
        // degenerate face were never triangulated, so they keep `u32::MAX` and
        // are never named here.
        let indices: Vec<u32> = tri
            .corners
            .iter()
            .map(|c| corner_to_render[*c as usize])
            .collect();

        // The triangles regrouped so that each material slot is one run, and
        // therefore one draw call. A stable sort, so faces keep their authored
        // order inside a slot: the id buffer resolves ties by depth and not by
        // draw order, but an exporter walking `tri_to_face` produces a
        // different file otherwise, and "the same mesh writes the same bytes"
        // is a property the io tests hold.
        //
        // `tri_to_face` moves with the indices. The id and element-overlay
        // passes are *non-indexed* draws whose `vertex_index` counts corners,
        // so they read `tri_to_face[vi / 3]` against this ordering — the two
        // disagreeing would make a click report a different face than the one
        // under the pointer.
        let (indices, tri_to_face, submeshes) =
            group_by_material(mesh, indices, tri.tri_to_face);

        // Which face each corner belongs to, for the groups below. One pass,
        // because the CSR walk needs it per corner and `tri_to_face` answers a
        // different question — it is per *triangle*.
        let mut corner_face = vec![0u32; corner_verts.len()];
        for f in 0..mesh.num_faces() {
            for c in mesh.face_corner_range(f) {
                corner_face[c] = f;
            }
        }

        // Each render vertex's smooth group: the faces its *normal* averages.
        //
        // **Not the corners that resolved to it**, which is what this counted
        // before surface coordinates joined the split key. A seam splits a
        // render vertex without splitting its shading, so counting the corners
        // that landed on it would give each side of the seam half the faces —
        // and a drag, which recomputes normals from this on the GPU, would
        // crease a sphere along the line where its texture wraps. The split
        // exists for two reasons now and only one of them is about normals.
        //
        // So a render vertex's group is every corner at its mesh vertex whose
        // normal matches, whether or not that corner resolved here. Two render
        // vertices either side of a seam therefore hold the same list, which
        // costs a copy at a seam and is what makes them shade identically.
        let render_verts = positions.len();
        let mut group_offsets = vec![0u32; render_verts + 1];
        let corners_of = |r: usize| {
            let v = render_to_vert[r] as usize;
            &incident[offsets[v] as usize..offsets[v + 1] as usize]
        };
        for r in 0..render_verts {
            let normal = normals[r];
            group_offsets[r + 1] = corners_of(r)
                .iter()
                .filter(|c| packed[**c as usize] == normal)
                .count() as u32;
        }
        for r in 0..render_verts {
            group_offsets[r + 1] += group_offsets[r];
        }
        let mut group_faces = vec![0u32; group_offsets[render_verts] as usize];
        for r in 0..render_verts {
            let normal = normals[r];
            let start = group_offsets[r] as usize;
            for (i, corner) in corners_of(r)
                .iter()
                .filter(|c| packed[**c as usize] == normal)
                .enumerate()
            {
                group_faces[start + i] = corner_face[*corner as usize];
            }
        }

        // Edges are authored against mesh vertices; point them at a render vertex
        // that shares the position. Which of the split copies is irrelevant — they
        // differ only in their normal.
        let last = positions.len().saturating_sub(1) as u32;
        let edge_verts: Vec<[u32; 2]> = mesh
            .edges()
            .iter()
            .map(|[a, b]| {
                [
                    representative[*a as usize].min(last),
                    representative[*b as usize].min(last),
                ]
            })
            .collect();
        // A vertex belonging to no face has no render vertex at all. Clamped
        // rather than dropped: the buffer is indexed by mesh vertex, so a hole
        // would shift every later vertex onto the wrong id.
        let vert_to_render: Vec<u32> = representative.iter().map(|r| (*r).min(last)).collect();

        let mut out = Self {
            positions,
            normals,
            uvs,
            indices,
            tri_to_face,
            submeshes,
            edge_verts,
            vert_to_render,
            render_to_vert,
            face_offsets: mesh.face_offsets().to_vec(),
            group_offsets,
            group_faces,
            corner_render: corner_to_render,
            vert_flags: vec![0; mesh.num_verts() as usize],
            edge_flags: vec![0; mesh.num_edges() as usize],
            face_flags: vec![0; mesh.num_faces() as usize],
            counts: MeshCounts {
                verts: mesh.num_verts(),
                edges: mesh.num_edges(),
                faces: mesh.num_faces(),
                triangles: 0,
            },
            // From the *mesh* positions, not the split render vertices. The two
            // enclose the same points, and the mesh array is the shorter walk.
            bounds: Aabb::from_points(mesh.positions().iter().copied()),
            clusters: None,
        };
        out.counts.triangles = (out.indices.len() / 3) as u32;
        out.update_flags(mesh);
        out
    }

    /// Render vertices after splitting, which is what the GPU buffers hold.
    pub fn render_verts(&self) -> u32 {
        self.positions.len() as u32
    }

    /// Recomputes the flag buffers from the mesh's attribute layers.
    ///
    /// Separate from [`RenderMeshCpu::extract`] because this is the cheap path: a
    /// selection change re-runs only this, and only the flag buffers are
    /// re-uploaded.
    pub fn update_flags(&mut self, mesh: &MeshData) {
        // The arrays are moved out and back rather than copied: they are one
        // per element, and a million-vertex mesh should not allocate them twice
        // to set some bits in them.
        let mut flags = ElementFlags {
            vert_flags: std::mem::take(&mut self.vert_flags),
            edge_flags: std::mem::take(&mut self.edge_flags),
            face_flags: std::mem::take(&mut self.face_flags),
            counts: self.counts,
        };
        flags.read(mesh);
        self.vert_flags = flags.vert_flags;
        self.edge_flags = flags.edge_flags;
        self.face_flags = flags.face_flags;
    }

    /// Total GPU bytes this mesh will occupy.
    ///
    /// Tracked as a budget rather than left to chance: memory regressions in this
    /// domain are silent right up until they are fatal.
    pub fn gpu_bytes(&self) -> usize {
        self.positions.len() * 12
            + self.normals.len() * 4
            + self.uvs.len() * 8
            + self.indices.len() * 4
            + self.tri_to_face.len() * 4
            + self.edge_verts.len() * 8
            + self.vert_flags.len() * 4
            + self.edge_flags.len() * 4
            + self.face_flags.len() * 4
            + self.clusters.as_ref().map_or(0, |c| c.memory_bytes())
    }

    /// Bakes the cluster decomposition, for the GPU-driven path.
    ///
    /// Opt-in because it costs a second copy of the index stream. A mesh
    /// uploaded without it draws nothing through the cluster path — the draw is
    /// skipped on a zero cluster count — so the viewport warns rather than
    /// leaving a blank viewport unexplained.
    pub fn with_clusters(mut self) -> Self {
        self.clusters = Some(crate::cluster::ClusterMesh::bake(&self));
        self
    }

    /// GPU bytes per vertex. The budget is 108 — see `tests/vram_budget.rs`.
    /// What a drag on this mesh would cost the GPU on top of its residency.
    ///
    /// Transient and for one mesh at a time: a deform session holds a copy of
    /// the rest positions and normals, a weight per render vertex, and the two
    /// index tables the normal pass walks. At a million render vertices and four
    /// million corners that is about forty megabytes, against the plan's two
    /// gigabytes for a ten-million-vertex scene — and it is released the moment
    /// the pointer comes up.
    pub fn deform_bytes(&self) -> usize {
        let verts = self.positions.len();
        verts * 12 // rest positions
            + verts * 4 // rest normals
            + verts * 4 // weights
            + (verts + 1) * 4 // group offsets
            + self.group_faces.len() * 4
            + self.corner_render.len() * 4
            + (self.counts.faces as usize + 1) * 4 // face offsets
            + self.counts.faces as usize * 12 // face normals
    }

    /// GPU bytes per *render* vertex, which is what is actually uploaded.
    ///
    /// Measured against the split count rather than the document's, because
    /// splitting is what the buffers pay for.
    pub fn bytes_per_vertex(&self) -> f32 {
        if self.positions.is_empty() {
            0.0
        } else {
            self.gpu_bytes() as f32 / self.positions.len() as f32
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
        // Indices address *render* vertices, which are split where shading
        // breaks, not the mesh vertices the document keeps.
        assert!(r.indices.iter().all(|&v| v < r.render_verts()));
        assert!(r.edge_verts.iter().flatten().all(|&v| v < r.render_verts()));
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
    fn a_cube_splits_into_one_vertex_per_face_corner() {
        // Eight vertices in the document, twenty-four on the GPU: each corner
        // needs a different normal for each of the three faces meeting there.
        // Without the split a cube shades as a rounded lump.
        let m = primitives::cube(2.0);
        let r = RenderMeshCpu::extract(&m);
        assert_eq!(m.num_verts(), 8, "the document keeps eight");
        assert_eq!(r.render_verts(), 24, "the GPU needs twenty-four");
        assert_eq!(r.counts.verts, 8, "the reported count is the document's");
    }

    #[test]
    fn a_smooth_mesh_splits_only_where_its_texture_wraps() {
        // The other direction from the cube: a sphere has one normal per vertex,
        // so nothing splits for shading. What does split is the seam — the same
        // position needs `u = 0` on one side of the wrap and `u = 1` on the
        // other, and a vertex buffer can only say one of those.
        //
        // The count is exact rather than a bound, because the two mechanisms
        // that could inflate it — a normal that fails to compare equal, a UV
        // epsilon that lets neighbours merge — both show up here as a number and
        // nowhere else as a picture.
        let (seg, ring) = (32, 16);
        let m = primitives::uv_sphere(1.0, seg, ring);
        let r = RenderMeshCpu::extract(&m);
        assert_eq!(
            r.render_verts() as u32,
            m.num_verts() as u32
                // the seam column, one extra copy each
                + (ring - 1)
                // and a pole, which has a different `u` for every face at it
                + 2 * (seg - 1),
            "a smooth sphere splits at its seam and nowhere else"
        );
    }

    #[test]
    fn the_two_sides_of_a_seam_still_shade_as_one_surface() {
        // **The group table is the smooth group, not the split.** It is what
        // `deform.rs` recomputes normals from while somebody drags, so if each
        // side of the seam only held the faces that resolved to it, a dragged
        // sphere would crease along the line where its texture wraps — a defect
        // that appears during the gesture and disappears when the pointer is
        // released, which is the hardest kind to report.
        let m = primitives::uv_sphere(1.0, 16, 8);
        let r = RenderMeshCpu::extract(&m);
        let mut seams = 0;
        let n = r.render_verts() as usize;
        for a in 0..n {
            for b in (a + 1)..n {
                if r.positions[a].0 != r.positions[b].0 || r.normals[a] != r.normals[b] {
                    continue;
                }
                // Same place, same normal, two render vertices: a seam pair.
                seams += 1;
                let faces = |i: usize| {
                    let (lo, hi) = (r.group_offsets[i] as usize, r.group_offsets[i + 1] as usize);
                    let mut v = r.group_faces[lo..hi].to_vec();
                    v.sort_unstable();
                    v
                };
                assert_eq!(faces(a), faces(b), "the seam pair at {a}/{b} shades apart");
            }
        }
        assert!(seams > 0, "the fixture has no seam to check");
    }

    #[test]
    fn split_vertices_share_a_position_and_differ_only_in_normal() {
        let m = primitives::cube(2.0);
        let r = RenderMeshCpu::extract(&m);
        // Group render vertices by position; each group must hold three, with
        // three distinct normals.
        let mut groups: std::collections::HashMap<[u32; 3], Vec<u32>> = Default::default();
        for (i, p) in r.positions.iter().enumerate() {
            let key = [p.0[0].to_bits(), p.0[1].to_bits(), p.0[2].to_bits()];
            groups.entry(key).or_default().push(r.normals[i]);
        }
        assert_eq!(groups.len(), 8, "eight distinct positions");
        for (key, normals) in &groups {
            assert_eq!(normals.len(), 3, "corner {key:?} should split three ways");
            let unique: std::collections::HashSet<_> = normals.iter().collect();
            assert_eq!(unique.len(), 3, "the three copies must differ");
        }
    }

    #[test]
    fn edges_still_resolve_to_a_position_after_splitting() {
        // The overlay buffers index mesh vertices; after splitting they have to
        // be pointed at a render vertex that shares the position.
        let m = primitives::cube(2.0);
        let r = RenderMeshCpu::extract(&m);
        for (e, [a, b]) in r.edge_verts.iter().enumerate() {
            let (pa, pb) = (r.positions[*a as usize].0, r.positions[*b as usize].0);
            assert_ne!(pa, pb, "edge {e} collapsed to a point");
            let [ma, mb] = m.edges()[e];
            assert_eq!(pa, m.positions()[ma as usize].to_array());
            assert_eq!(pb, m.positions()[mb as usize].to_array());
        }
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
    fn a_hidden_face_has_no_triangles_and_its_elements_say_so() {
        let mut m = primitives::cube(2.0);
        let full = RenderMeshCpu::extract(&m).tri_to_face.len();
        let mut hidden = vec![false; m.num_faces() as usize];
        hidden[1] = true;
        m.attributes_mut().insert(names::HIDE_FACE, kreate_mesh::Domain::Face, kreate_mesh::AttrData::Bool(hidden.into()));
        let mut verts = vec![false; m.num_verts() as usize];
        verts[0] = true;
        m.attributes_mut().insert(names::HIDE_VERT, kreate_mesh::Domain::Point, kreate_mesh::AttrData::Bool(verts.into()));
        let r = RenderMeshCpu::extract(&m);
        assert_eq!(r.tri_to_face.len(), full - 2, "a quad is two triangles");
        assert!(!r.tri_to_face.contains(&1));
        assert_eq!(r.indices.len(), r.tri_to_face.len() * 3);
        assert_ne!(r.face_flags[1] & flags::HIDDEN, 0);
        assert_ne!(r.vert_flags[0] & flags::HIDDEN, 0);
        assert_eq!(r.vert_flags[1] & flags::HIDDEN, 0);
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
    fn a_drag_s_buffers_are_not_counted_as_residency() {
        // The distinction the budget test forced. A resident mesh pays for its
        // positions, normals, indices and flags; a *drag* pays for a rest copy
        // and the tables its normal pass walks, for one mesh, until the pointer
        // comes up. Counting the second as the first put a grid at 116 bytes a
        // vertex against a hundred, and it was not wrong about the memory — only
        // about when it is spent.
        let mesh = primitives::grid(10.0, 40, 40);
        let r = RenderMeshCpu::extract(&mesh);

        let resident = r.gpu_bytes();
        let dragging = r.deform_bytes();
        assert!(dragging > 0, "a drag has to cost something");
        // Both real, and of the same order: the drag is not a rounding error on
        // the residency, which is why it needs saying out loud.
        assert!(
            dragging > resident / 8 && dragging < resident * 2,
            "residency {resident}, drag {dragging}"
        );
        assert!(
            r.bytes_per_vertex() <= 100.0,
            "{:.1} bytes/vertex: a drag's buffers are being counted as residency",
            r.bytes_per_vertex()
        );
    }

    #[test]
    fn a_render_vertex_knows_which_mesh_vertex_it_came_from() {
        // The direction `vert_to_render` cannot give: a cube's corner is three
        // render vertices and that field names one of them. A drag needs all
        // three, because the weight is a property of the mesh vertex and the
        // buffer it writes is indexed by render vertex.
        let cube = primitives::cube(2.0);
        let r = RenderMeshCpu::extract(&cube);
        assert_eq!(r.render_to_vert.len(), r.positions.len());

        // Every render vertex sits exactly where its mesh vertex does.
        let mesh_positions = cube.positions();
        for (i, v) in r.render_to_vert.iter().enumerate() {
            assert_eq!(
                r.positions[i].0,
                mesh_positions[*v as usize].to_array(),
                "render vertex {i} is not where mesh vertex {v} is"
            );
        }
        // And the two directions agree where they overlap.
        for (v, render) in r.vert_to_render.iter().enumerate() {
            assert_eq!(r.render_to_vert[*render as usize], v as u32);
        }
        // A cube splits three ways, so this is not a one-to-one mapping being
        // checked against itself.
        assert_eq!(r.positions.len(), 24);
    }

    #[test]
    fn a_render_vertex_s_group_is_the_faces_its_normal_averages() {
        // What makes a GPU normal pass produce the same answer as the CPU: the
        // group is the set `compute_corner_normals` sums, so summing it again
        // has to give the normal already in the buffer.
        //
        // Two meshes, because they exercise the two ends of the split: a cube
        // where every group is one face, and a sphere where a group is the
        // whole fan.
        for mesh in [primitives::cube(2.0), primitives::uv_sphere(1.0, 12, 6)] {
            let r = RenderMeshCpu::extract(&mesh);
            let faces = mesh.compute_face_normals();
            assert_eq!(r.group_offsets.len(), r.positions.len() + 1);

            for v in 0..r.positions.len() {
                let group =
                    &r.group_faces[r.group_offsets[v] as usize..r.group_offsets[v + 1] as usize];
                assert!(!group.is_empty(), "render vertex {v} averages nothing");

                // No face twice: a face has one corner per vertex, and a render
                // vertex sits at one vertex.
                let mut unique = group.to_vec();
                unique.sort_unstable();
                unique.dedup();
                assert_eq!(unique.len(), group.len(), "a face is in group {v} twice");

                let sum: Vec3 = group.iter().map(|f| faces[*f as usize]).sum();
                let expected = oct_decode(r.normals[v]);
                assert!(
                    sum.normalize_or_zero().dot(expected) > 0.999,
                    "group {v} sums to {:?}, buffer says {expected:?}",
                    sum.normalize_or_zero()
                );
            }
        }
    }

    #[test]
    fn every_corner_resolves_to_a_render_vertex_at_its_own_vertex() {
        // What lets a face be walked in render-vertex terms, which is what
        // recomputing a face normal from deformed positions needs. A corner
        // pointing at a render vertex of the *wrong* mesh vertex would give a
        // face normal computed from someone else's geometry.
        let mesh = primitives::uv_sphere(1.0, 10, 5);
        let r = RenderMeshCpu::extract(&mesh);
        assert_eq!(r.corner_render.len(), mesh.num_corners() as usize);

        let corner_verts = mesh.corner_verts();
        for (corner, render) in r.corner_render.iter().enumerate() {
            assert_ne!(*render, u32::MAX, "corner {corner} resolves to nothing");
            assert_eq!(
                r.render_to_vert[*render as usize], corner_verts[corner],
                "corner {corner} points at a render vertex of another vertex"
            );
        }
    }

    #[test]
    fn the_flags_alone_agree_with_the_whole_render_mesh() {
        // The two must not drift. A click takes the cheap path and everything
        // else takes the full one, so a disagreement would show as a selection
        // highlight that is right until something unrelated forces a re-upload
        // and then changes.
        //
        // The fixture has a boundary, a non-manifold edge and a partial
        // selection, because a closed cube with nothing selected has every flag
        // at zero and two empty answers agree about everything.
        let mut mesh = MeshData::from_faces(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(1.0, 0.0, -1.0),
                Vec3::new(1.0, 1.0, -1.0),
            ],
            // Three faces meeting at the edge 1-2: non-manifold there, and
            // every other edge is a boundary.
            &[vec![0, 1, 2, 3], vec![1, 4, 5, 2], vec![1, 2, 7, 6]],
        );
        mesh.ensure_selection_layers();
        // One of three faces, so both a set and an unset bit are present.
        kreate_mesh::selection::set_selected(&mut mesh, Domain::Face, 1, true);

        let whole = RenderMeshCpu::extract(&mesh);
        let alone = ElementFlags::extract(&mesh);

        assert_eq!(alone.vert_flags, whole.vert_flags);
        assert_eq!(alone.edge_flags, whole.edge_flags);
        assert_eq!(alone.face_flags, whole.face_flags);
        assert!(alone.counts.same_elements(&whole.counts));

        // And the fixture is doing its job.
        assert!(
            alone
                .edge_flags
                .iter()
                .any(|f| f & flags::NON_MANIFOLD != 0),
            "no non-manifold edge, so the topology half is untested"
        );
        assert!(
            alone.edge_flags.iter().any(|f| f & flags::BOUNDARY != 0),
            "no boundary edge"
        );
        assert!(
            alone.face_flags.iter().any(|f| f & flags::SELECTED != 0),
            "nothing selected, so the selection half is untested"
        );
        assert!(
            alone.face_flags.iter().any(|f| f & flags::SELECTED == 0),
            "everything selected, so a flag stuck on would pass"
        );
    }

    #[test]
    fn the_flags_alone_are_refused_for_a_different_mesh() {
        // What stops a write past the end of somebody's buffer. The counts
        // comparison has to ignore the triangle count, which the cheap path
        // never learns, while still catching a mesh of another shape.
        let cube = ElementFlags::extract(&primitives::cube(1.0));
        let sphere = RenderMeshCpu::extract(&primitives::uv_sphere(1.0, 8, 4));
        assert!(!cube.counts.same_elements(&sphere.counts));

        // And the triangle count alone is not grounds for refusal, which is the
        // whole reason `same_elements` exists.
        let whole_cube = RenderMeshCpu::extract(&primitives::cube(1.0));
        assert_ne!(cube.counts, whole_cube.counts, "the triangle count differs");
        assert!(cube.counts.same_elements(&whole_cube.counts));
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
        // A hundred and eight bytes per render vertex, which is the plan's
        // hundred plus the surface coordinate a material needs to sample
        // anything. See `tests/vram_budget.rs`, which carries the reasoning and
        // the scene-level figure that actually holds the line.
        //
        // Asserted here as well so a regression fails the build rather than
        // being noticed as an out-of-memory report on an 8 GB card.
        for (name, mesh) in [
            ("grid", primitives::grid(10.0, 40, 40)),
            ("sphere", primitives::uv_sphere(1.0, 48, 24)),
            ("torus", primitives::torus(1.0, 0.3, 48, 24)),
            ("icosphere", primitives::ico_sphere(1.0, 4)),
        ] {
            let r = RenderMeshCpu::extract(&mesh);
            assert!(
                r.bytes_per_vertex() <= 108.0,
                "{name}: {:.1} bytes/vertex exceeds the 108 byte budget",
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

    /// Where the decoder starts, in both shaders that carry a copy.
    ///
    /// The studio pass and the rendered one are not in a composition with each
    /// other — `solid.wgsl` predates that machinery and stands alone — so the
    /// duplication cannot be removed. The drift can be.
    const OCT_DECODE_WGSL: &str = "fn unpack_snorm16(v: u32) -> f32 {";

    #[test]
    fn the_two_shader_copies_of_the_decode_agree() {
        // `solid.wgsl` and `rendered.wgsl` each carry the decoder, because
        // they are separate shaders and WGSL has no include in this build. A
        // copy that drifted would make one pass shade differently from the
        // other — the viewport disagreeing with itself, which is §5's second
        // risk at closer range than usual.
        //
        // Character for character, from the marker to the end of the
        // function, so a change to either has to be made to both.
        let extract = |source: &str, name: &str| -> String {
            // Line endings normalised first: the files are checked out with
            // CRLF here and the claim is about the code, not about which
            // bytes separate its lines.
            let source = &source.replace("\r\n", "\n");
            let start = source
                .find(OCT_DECODE_WGSL)
                .unwrap_or_else(|| panic!("{name} has no copy of the decoder"));
            let rest = &source[start..];
            let end = rest
                .find("\nfn oct_decode")
                .and_then(|at| rest[at + 1..].find("\n}\n").map(|e| at + 1 + e + 3))
                .unwrap_or_else(|| panic!("{name}'s decoder has no end"));
            rest[..end].to_string()
        };

        let solid = extract(include_str!("shaders/solid.wgsl"), "solid.wgsl");
        let rendered = extract(include_str!("shaders/rendered.wgsl"), "rendered.wgsl");
        assert_eq!(
            solid, rendered,
            "the two shader copies of the octahedral decode have drifted"
        );
        // And the extraction found something, or this compares two empties.
        assert!(solid.contains("fn oct_decode"), "{solid}");
        assert!(solid.len() > 300, "the extraction is too short: {solid}");
    }
}

#[cfg(test)]
mod submesh_tests {
    use super::*;
    use kreate_mesh::attribute::{names, AttrData, Domain};
    use kreate_mesh::primitives;

    fn with_materials(per_face: Vec<i32>) -> RenderMeshCpu {
        let mut mesh = primitives::cube(1.0);
        assert_eq!(mesh.num_faces() as usize, per_face.len());
        mesh.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(per_face.into()),
        );
        RenderMeshCpu::extract(&mesh)
    }

    /// Every triangle appears exactly once, and each run holds the triangles
    /// whose face names its slot.
    fn check_covers(cpu: &RenderMeshCpu, mesh: &kreate_mesh::MeshData) {
        let expected = cpu.indices.len() as u32;
        let covered: u32 = cpu.submeshes.iter().map(|s| s.count).sum();
        assert_eq!(covered, expected, "the runs do not cover the index buffer");

        let mut at = 0u32;
        for run in &cpu.submeshes {
            assert_eq!(run.start, at, "the runs are not contiguous");
            assert_eq!(run.count % 3, 0, "a run is not a whole number of triangles");
            at += run.count;
        }

        let face_material = mesh.attributes().get_i32(names::MATERIAL_INDEX);
        for run in &cpu.submeshes {
            for t in (run.start / 3)..((run.start + run.count) / 3) {
                let face = cpu.tri_to_face[t as usize] as usize;
                let want = face_material
                    .map(|m| m[face].max(0) as u32)
                    .unwrap_or(0);
                assert_eq!(
                    want, run.slot,
                    "a triangle of face {face} is in the run for slot {}",
                    run.slot
                );
            }
        }
    }

    #[test]
    fn a_mesh_with_no_material_layer_is_one_run() {
        // The common case, and the one the render loop must not pay for: a
        // model being built has no material layer at all and has to draw in a
        // single call.
        let mesh = primitives::cube(1.0);
        let cpu = RenderMeshCpu::extract(&mesh);
        assert_eq!(
            cpu.submeshes,
            vec![Submesh {
                slot: 0,
                start: 0,
                count: cpu.indices.len() as u32
            }]
        );
        check_covers(&cpu, &mesh);
    }

    #[test]
    fn faces_on_one_slot_stay_one_run_and_keep_their_order() {
        // A layer that says the same thing everywhere must not cost a sort.
        // Compared against the no-layer extraction, which is the definition of
        // "unchanged": the exporter's "the same mesh writes the same bytes"
        // rests on `tri_to_face` order.
        let plain = RenderMeshCpu::extract(&primitives::cube(1.0));
        let cpu = with_materials(vec![0; 6]);
        assert_eq!(cpu.submeshes.len(), 1);
        assert_eq!(cpu.indices, plain.indices);
        assert_eq!(cpu.tri_to_face, plain.tri_to_face);
    }

    #[test]
    fn faces_on_two_slots_become_two_runs() {
        let mut mesh = primitives::cube(1.0);
        let per_face = vec![1, 0, 1, 0, 1, 0];
        mesh.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(per_face.clone().into()),
        );
        let cpu = RenderMeshCpu::extract(&mesh);
        assert_eq!(cpu.submeshes.len(), 2, "{:?}", cpu.submeshes);
        // Sorted, so slot 0 comes first however the faces were authored.
        assert_eq!(cpu.submeshes[0].slot, 0);
        assert_eq!(cpu.submeshes[1].slot, 1);
        check_covers(&cpu, &mesh);
    }

    #[test]
    fn faces_of_one_slot_are_gathered_even_when_they_are_scattered() {
        // The case a draw call range cannot express unless the triangles move.
        // Three slots interleaved face by face: without the regrouping this
        // would need six ranges and, drawn as three, would paint two faces in
        // the wrong material each.
        let mut mesh = primitives::cube(1.0);
        mesh.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(vec![0, 1, 2, 0, 1, 2].into()),
        );
        let cpu = RenderMeshCpu::extract(&mesh);
        assert_eq!(
            cpu.submeshes.iter().map(|s| s.slot).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "one run per slot, not one per face"
        );
        check_covers(&cpu, &mesh);
    }

    #[test]
    fn a_negative_index_means_no_material_rather_than_a_slot() {
        // What glTF writes for an unassigned face, and what the importer reads
        // back. It has to land where a mesh with no layer lands, or
        // "unassigned" would be two different states that draw differently.
        let mut mesh = primitives::cube(1.0);
        mesh.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(vec![-1; 6].into()),
        );
        let cpu = RenderMeshCpu::extract(&mesh);
        assert_eq!(cpu.submeshes.len(), 1);
        assert_eq!(cpu.submeshes[0].slot, 0);
    }

    #[test]
    fn the_triangles_still_name_the_faces_they_came_from() {
        // The reordering moves `indices` and `tri_to_face` together. If it
        // ever moved only one, the id pass — a non-indexed draw reading
        // `tri_to_face[vertex_index / 3]` — would report a different face than
        // the one under the pointer, and nothing else here would notice.
        let mut mesh = primitives::cube(1.0);
        mesh.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(vec![2, 0, 1, 2, 0, 1].into()),
        );
        let cpu = RenderMeshCpu::extract(&mesh);
        let positions = mesh.positions();
        for (t, face) in cpu.tri_to_face.iter().enumerate() {
            // Every corner of the triangle sits on a vertex of that face.
            let of_face: Vec<u32> = mesh
                .face_corner_range(*face)
                .map(|c| mesh.corner_verts()[c])
                .collect();
            for corner in 0..3 {
                let render = cpu.indices[t * 3 + corner] as usize;
                let vert = cpu.render_to_vert[render];
                assert!(
                    of_face.contains(&vert),
                    "triangle {t} claims face {face} but uses vertex {vert} at {:?}",
                    positions[vert as usize]
                );
            }
        }
    }

    #[test]
    fn every_face_is_still_drawn_after_the_regrouping() {
        // A sort that dropped or duplicated a triangle would leave a hole in
        // the model, which the run arithmetic above would not catch.
        let mut mesh = primitives::cube(1.0);
        mesh.attributes_mut().insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::I32(vec![1, 0, 1, 0, 1, 0].into()),
        );
        let cpu = RenderMeshCpu::extract(&mesh);
        let plain = RenderMeshCpu::extract(&primitives::cube(1.0));
        assert_eq!(cpu.indices.len(), plain.indices.len());
        let mut seen = cpu.tri_to_face.clone();
        seen.sort_unstable();
        let mut want = plain.tri_to_face.clone();
        want.sort_unstable();
        assert_eq!(seen, want, "a triangle was dropped or duplicated");
    }
}
