//! Pipelines and the viewport pass list.
//!
//! Deliberately a hand-written list of passes rather than a general render graph.
//! An editor's pass structure is small, stable, and its ordering carries meaning
//! — the overlay layering *is* the interaction design. A declarative graph would
//! be a tax paid up front for flexibility that is not needed, and a real DAG
//! scheduler is never needed at all on a single queue.

use bytemuck::{Pod, Zeroable};
use glam::{DAffine3, Mat3, Mat4, Vec4};
use wgpu::util::DeviceExt as _;

use crate::camera::{Camera, Projection};
use crate::cluster::ClusterMesh;
use crate::cluster_cull::{ClusterCull, CullDraws, CullView, Pass};
use crate::gpu::{self, DEPTH_CLEAR, DEPTH_FORMAT};
use crate::hiz::HiZ;
use crate::lod::{Budget, Detail, ViewSize};
use crate::mesh_gpu::{ElementFlags, MeshCounts, RenderMeshCpu};
use crate::palette;
use crate::picking::PickDomain;
use kreate_math::Aabb;

/// How opaque a surface is in X-ray: Blender's default half, a little less so
/// an edge behind two surfaces still reads.
const XRAY_ALPHA: f32 = 0.4;

/// How the solid pass shades.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadingMode {
    /// Three camera-relative lights, baked into `solid.wgsl`.
    ///
    /// Kept, and not a legacy path. While modelling, light that shows the
    /// *shape* is the right light — a scene lit for a render is often a scene
    /// you cannot model in, and §4.13 judged this correctly the first time.
    Studio,
    /// No light at all: every surface its plain colour, Blender's Flat.
    Flat,
    /// Edges only, every one of them — the surfaces are not drawn, so the
    /// back of the model shows through, as Blender's Wireframe does.
    Wireframe,
    /// The scene's own lights, one BRDF, through the display transform.
    ///
    /// Shades into a linear `Rgba16Float` target and resolves it in a
    /// fullscreen pass, because the transform belongs at the end on a linear
    /// image — per fragment it would apply before blending, which is the bug
    /// §2.1 records as compositing seven times too dark.
    Rendered,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    params: [f32; 4],
    /// The camera's right, up and back in the world, for lights that turn
    /// with the view.
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    camera_back: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ObjectData {
    model: [[f32; 4]; 4],
    normal_col0: [f32; 4],
    normal_col1: [f32; 4],
    normal_col2: [f32; 4],
    base_color: [f32; 4],
    /// x: one to colour the surface by the weight in its first UV coordinate,
    /// Blender's weight-paint ramp, instead of by `base_color`.
    paint: [f32; 4],
}

/// Per-draw data for the rendered pass. Matches `RenderedObject` in
/// `rendered.wgsl`.
///
/// Separate from [`ObjectData`] rather than widening it, so the studio path is
/// untouched: its uniform is a hot, per-draw write in every frame of a drag,
/// and growing it by sixteen bytes for a mode that is off would be a cost paid
/// by the mode that is on.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RenderedObject {
    model: [[f32; 4]; 4],
    normal_col0: [f32; 4],
    normal_col1: [f32; 4],
    normal_col2: [f32; 4],
}

/// What the rendered pass needs to know about the frame.
///
/// Matches `Shading` in `shading.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ShadingGlobals {
    /// How many tiles across the target is, and how long one tile's list can
    /// be. See `crate::tiles`.
    ///
    /// The light *count* used to live here and does not any more: the shading
    /// loop reads a tile's list, and how long that list is the cull pass's
    /// answer. A count here would be a second opinion about the same question,
    /// written every frame and read by nobody.
    tiles_across: u32,
    tile_capacity: u32,
    /// The drawn region in the rendered view, zero outside it.
    region: [u32; 2],
    /// rgb: the world's radiance, linear. w: nonzero when an environment cube
    /// is bound. See `shading.wgsl`.
    world: [f32; 4],
    /// x, y: the cosine and sine of the world's rotation. z: the roughest mip.
    environment: [f32; 4],
    /// Irradiance coefficients. See `crate::environment::irradiance_sh9`.
    sh9: [[f32; 4]; 9],
    /// Turns a clip position back into a world direction. See `shading.wgsl`.
    inv_view_proj: [[f32; 4]; 4],
    /// This frame's camera, for contact shadows. See `shading.wgsl`.
    view_proj: [[f32; 4]; 4],
    /// x: contact shadows on.
    flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MarkerGlobals {
    view_proj: [[f32; 4]; 4],
    /// xy: viewport size in pixels. z: relative depth offset toward the
    /// viewer. w: unused.
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct EdgeGlobals {
    view_proj: [[f32; 4]; 4],
    params: [f32; 4],
    /// The three edge diagnostics, linear RGBA.
    ///
    /// Per frame rather than per object: they never vary by object, and they
    /// belong in the palette rather than in the shader so a contrast change is
    /// one edit instead of a hunt through WGSL.
    non_manifold_color: [f32; 4],
    seam_color: [f32; 4],
    sharp_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct EdgeObject {
    model: [[f32; 4]; 4],
    base_color: [f32; 4],
    selected_color: [f32; 4],
    flags_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ElementGlobals {
    view_proj: [[f32; 4]; 4],
    params: [f32; 4],
    layers: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ElementObject {
    model: [[f32; 4]; 4],
    selected_color: [f32; 4],
    unselected_color: [f32; 4],
    /// x: whether the rubber-band preview applies to this draw.
    ///
    /// Per draw rather than compared against an object id in the shader: the
    /// preview belongs to the one object being edited, and a flag the caller
    /// already knows is cheaper than an id round trip through two uniforms.
    flags: [f32; 4],
}

/// What a rubber-band drag would select, for the overlays to draw.
///
/// The bitset the compute reduction just produced, read directly. That is the
/// point of it being a bitset rather than a list: the highlight follows the drag
/// with no readback and no upload, so what the user sees during the drag is
/// literally what the commit will apply.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PreviewParams {
    /// x: preview active. y: domain — 0 vert, 1 edge, 2 face. z: whether the
    /// drag removes rather than adds. w: how many elements the bitset covers.
    mode: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct IdGlobals {
    view_proj: [[f32; 4]; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct IdObject {
    model: [[f32; 4]; 4],
    /// x: object id plus one. Zero is reserved for "nothing here", which is what
    /// lets an untouched id pixel mean a miss rather than object zero.
    ids: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GridGlobals {
    view_proj: [[f32; 4]; 4],
    inv_view_row0: [f32; 4],
    inv_view_row1: [f32; 4],
    inv_view_row2: [f32; 4],
    eye_and_tan: [f32; 4],
    params: [f32; 4],
    /// The lattice and the two visible axis lines, linear RGBA.
    ///
    /// Passed in rather than written into the shader so the axis colours are
    /// the same three values the interface puts on a numeric field. A grid
    /// whose X is a different red from the X in the transform panel teaches
    /// the user nothing.
    line_color: [f32; 4],
    axis_x_color: [f32; 4],
    axis_y_color: [f32; 4],
}

/// A mesh resident on the GPU.
///
/// The buffers are separate rather than interleaved so that the hot ones can be
/// rewritten without touching the cold ones — see `mesh_gpu`.
pub struct GpuMesh {
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    /// Surface coordinates, read only by the rendered pass.
    ///
    /// Always present, zeroed for a mesh with no UV layer. A pipeline's vertex
    /// layout is fixed, so "has UVs or not" would otherwise be two pipelines
    /// per material — a scene would compile twice the shaders to save eight
    /// bytes a vertex on the meshes that are not textured.
    uvs: wgpu::Buffer,
    indices: wgpu::Buffer,
    /// Positions as storage, plus the edge list and its flags.
    ///
    /// Per-*mesh*, unlike the transform: geometry is shared between linked
    /// duplicates, and only the transform differs.
    edge_bind: wgpu::BindGroup,
    /// Everything the id pass, the face fill and the vertex points read:
    /// positions, the index stream, the triangle-to-face map, the edge list, the
    /// vertex map and the per-element flags.
    id_bind: wgpu::BindGroup,
    /// The three flag buffers, kept so that a selection change is a write to
    /// *these* and nothing else.
    ///
    /// This is the load-bearing part of the buffer split: a renderer that
    /// rebuilds geometry when a selection changes turns one click on a dense
    /// mesh into a visible stall, and rubber-band selection into a slideshow.
    vert_flags: wgpu::Buffer,
    edge_flags: wgpu::Buffer,
    face_flags: wgpu::Buffer,
    pub counts: MeshCounts,
    /// The runs of triangles that share a material slot.
    ///
    /// Kept on the CPU rather than uploaded: they are draw-call arguments, not
    /// data a shader reads. One run for almost every mesh, so the rendered
    /// pass's loop over them costs nothing until a model actually has a second
    /// material on it.
    pub submeshes: Vec<crate::mesh_gpu::Submesh>,
    /// Render vertices after the normal split, which is what the position buffer
    /// holds. Distinct from `counts.verts`, which counts *mesh* vertices.
    pub render_verts: u32,
    pub color: Vec4,
    /// Object-space bounds, for the screen-size decision. See [`crate::lod`].
    pub bounds: Aabb,
    /// The cluster records, for the cull.
    cluster_bind: wgpu::BindGroup,
    /// Positions, normals, records and the cluster index stream, for the draw.
    cluster_draw_bind: wgpu::BindGroup,
    /// How many clusters this mesh has.
    pub cluster_count: u32,
}

/// The per-mesh bind group layouts a `GpuMesh` needs.
///
/// Grouped rather than passed one by one: every pass that reads geometry from
/// storage adds one, and a function taking four layouts in a row is a call site
/// nobody can read.
pub struct MeshLayouts {
    pub edge: wgpu::BindGroupLayout,
    pub id: wgpu::BindGroupLayout,
    /// The cull's view of a mesh: its cluster records alone.
    pub cluster: wgpu::BindGroupLayout,
    /// The cluster draw's view: positions, normals, records and indices.
    pub cluster_draw: wgpu::BindGroupLayout,
}

impl GpuMesh {
    pub fn upload(
        device: &wgpu::Device,
        layouts: &MeshLayouts,
        cpu: &RenderMeshCpu,
        color: Vec4,
    ) -> Self {
        // COPY_DST on the volatile buffers: they are rewritten during a drag.
        // STORAGE on positions: the edge overlay reads both endpoints of an edge
        // from it to expand a screen-space line quad.
        let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh positions"),
            contents: bytemuck::cast_slice(&cpu.positions),
            // COPY_SRC as well: a drag copies these to a rest buffer on the GPU
            // rather than having the CPU send them again, which is the whole
            // cost it exists to avoid.
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let normals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh normals"),
            contents: bytemuck::cast_slice(&cpu.normals),
            // STORAGE as well as VERTEX: the direct solid pass fetches these as
            // a vertex attribute, and the cluster pass reads the same bytes as
            // an array because its vertex index comes from the cluster index
            // stream rather than from an index buffer.
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let uvs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh uvs"),
            contents: bytemuck::cast_slice(&cpu.uvs),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        // STORAGE as well as INDEX: the id pass draws faces *non-indexed* and
        // reads this as an array, because `vertex_index` in an indexed draw is
        // the vertex fetched from the index buffer and says nothing about which
        // triangle is being drawn — and WebGPU has no portable primitive index.
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&pad_if_empty_u32(&cpu.indices)),
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });
        let tri_to_face = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh tri to face"),
            contents: bytemuck::cast_slice(&pad_if_empty_u32(&cpu.tri_to_face)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let vert_to_render = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vert to render"),
            contents: bytemuck::cast_slice(&pad_if_empty_u32(&cpu.vert_to_render)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let vert_flags = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vert flags"),
            contents: bytemuck::cast_slice(&pad_if_empty_u32(&cpu.vert_flags)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let face_flags = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh face flags"),
            contents: bytemuck::cast_slice(&pad_if_empty_u32(&cpu.face_flags)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // The edge list is authored against render vertices and changes only when
        // topology does; the flags change on selection alone. Keeping them apart
        // is what lets a selection update without touching geometry.
        let edge_verts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh edge verts"),
            contents: bytemuck::cast_slice(&pad_if_empty_u32x2(&cpu.edge_verts)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let edge_flags = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh edge flags"),
            contents: bytemuck::cast_slice(&pad_if_empty_u32(&cpu.edge_flags)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let edge_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("edge bind group"),
            layout: &layouts.edge,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edge_verts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: edge_flags.as_entire_binding(),
                },
            ],
        });

        let id_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("id bind group"),
            layout: &layouts.id,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tri_to_face.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: edge_verts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: vert_to_render.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: vert_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: face_flags.as_entire_binding(),
                },
            ],
        });

        // Whatever the caller decided with `RenderMeshCpu::with_clusters`. Not
        // baked here, because the bake costs a second copy of the whole index
        // stream and the path that reads it is off by default — every mesh in
        // every scene was paying for it.
        //
        // Empty buffers still get allocated when there are no clusters: a
        // zero-sized binding is a validation error, and one 48-byte allocation
        // per mesh is cheaper than making the bind group optional everywhere.
        let empty = ClusterMesh::default();
        let baked = cpu.clusters.as_ref().unwrap_or(&empty);
        let cluster_count = baked.clusters.len() as u32;
        let cluster_records = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh clusters"),
            // An empty mesh still needs a buffer: a zero-sized binding is a
            // validation error, and the draw is skipped on the count anyway.
            contents: if baked.clusters.is_empty() {
                &[0u8; 48]
            } else {
                bytemuck::cast_slice(&baked.clusters)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cluster_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh cluster indices"),
            contents: if baked.indices.is_empty() {
                &[0u8; 4]
            } else {
                bytemuck::cast_slice(&baked.indices)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cluster_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh clusters"),
            layout: &layouts.cluster,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cluster_records.as_entire_binding(),
            }],
        });
        let cluster_draw_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh cluster draw"),
            layout: &layouts.cluster_draw,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: normals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cluster_records.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cluster_indices.as_entire_binding(),
                },
            ],
        });

        Self {
            positions,
            normals,
            uvs,
            indices,
            edge_bind,
            id_bind,
            cluster_bind,
            cluster_draw_bind,
            cluster_count,
            vert_flags,
            edge_flags,
            face_flags,
            counts: cpu.counts,
            submeshes: cpu.submeshes.clone(),
            render_verts: cpu.render_verts(),
            color,
            bounds: cpu.bounds,
        }
    }

    /// The buffers a drag writes into, which are the ones the draw reads.
    ///
    /// Lent rather than copied, and deliberately the *live* ones: a deform
    /// session reads its own rest copy and writes here, so the rasteriser needs
    /// no knowledge that a drag is happening at all.
    pub fn positions(&self) -> &wgpu::Buffer {
        &self.positions
    }

    pub fn normals(&self) -> &wgpu::Buffer {
        &self.normals
    }

    /// Rewrites only the position and normal buffers.
    ///
    /// The path a transform drag would take on the CPU. It exists to prove the
    /// buffer split works; the eventual interactive path does this on the GPU
    /// from an immutable rest-position buffer, so the CPU uploads one small
    /// uniform per frame instead of the whole array.
    pub fn update_positions(&self, queue: &wgpu::Queue, cpu: &RenderMeshCpu) {
        queue.write_buffer(&self.positions, 0, bytemuck::cast_slice(&cpu.positions));
        queue.write_buffer(&self.normals, 0, bytemuck::cast_slice(&cpu.normals));
    }

    /// Rewrites only the three flag buffers.
    ///
    /// What a selection change costs: three small writes, no geometry touched,
    /// and the highlight is correct in the same frame. The buffers are sized from
    /// the mesh's element counts and a selection cannot change those, so a
    /// caller reaching this with a differently shaped mesh has a topology change
    /// on its hands and needs a full upload instead.
    pub fn update_flags(&self, queue: &wgpu::Queue, flags: &ElementFlags) -> bool {
        // The element counts, not the whole `MeshCounts`: what arrives here may
        // have been read off the mesh without triangulating it, so it has no
        // triangle count to agree about. See `MeshCounts::same_elements`.
        if !flags.counts.same_elements(&self.counts) {
            return false;
        }
        queue.write_buffer(
            &self.vert_flags,
            0,
            bytemuck::cast_slice(&pad_if_empty_u32(&flags.vert_flags)),
        );
        queue.write_buffer(
            &self.edge_flags,
            0,
            bytemuck::cast_slice(&pad_if_empty_u32(&flags.edge_flags)),
        );
        queue.write_buffer(
            &self.face_flags,
            0,
            bytemuck::cast_slice(&pad_if_empty_u32(&flags.face_flags)),
        );
        true
    }
}

/// One entry in the frame's draw list: a mesh, where to put it, and the state
/// that belongs to the *object* rather than to the geometry.
///
/// A tuple would do for the first two, but everything object-level lands here —
/// selection now, and later the outline group, the x-ray tier and the per-object
/// display mode — and each of those is precisely what cannot live on the mesh,
/// because two linked duplicates share one datablock and disagree about all of
/// them.
#[derive(Clone, Copy)]
pub struct Draw<'a> {
    pub mesh: &'a GpuMesh,
    pub transform: DAffine3,
    /// Highlights the object's wireframe. Object-mode selection; element
    /// selection travels in the mesh's own flags buffer.
    pub selected: bool,
    /// The active object: outlined lighter than the rest of the selection.
    pub active: bool,
    /// What the id pass writes for this draw, so a pick names the object rather
    /// than the geometry.
    ///
    /// Per draw and not per mesh, which is the whole reason it is here: two
    /// linked duplicates share one datablock and are different objects.
    pub object_id: u32,
    /// Whether the rubber-band preview applies to this draw.
    ///
    /// A drag resolves elements of *one* object — the one being edited — and the
    /// bitset is indexed by that mesh's element numbering. Applied to a
    /// neighbour it would highlight whatever elements happened to share those
    /// indices.
    pub preview: bool,
    /// Whether this object is off limits to screen-size simplification.
    ///
    /// True for the mesh being edited. Selection is protected too, but that is
    /// read from `selected` rather than duplicated here — see
    /// [`Draw::protected`].
    pub editing: bool,
    /// Which of the viewport's materials this object is made of.
    ///
    /// An index, because `kreate-render` cannot see a document: the app
    /// compiles each material's shading graph and hands the results over with
    /// [`Viewport::set_materials`], the same way it hands over the lights.
    ///
    /// Zero is whatever the app put first, and the app puts a plain grey there
    /// — an object with no material assigned is not an error and not invisible.
    /// A fresh cube has no materials and should still be shaded.
    ///
    /// This is the material for slot zero, and for every slot `materials` does
    /// not name. A mesh whose faces are all on one slot — which is nearly all
    /// of them — needs nothing else.
    pub material: u32,
    /// Coloured by weight: see [`Draw::weights`].
    pub weights: bool,
    /// One material per mesh material *slot*, for a mesh with more than one.
    ///
    /// **Borrowed rather than owned** so that `Draw` stays `Copy` and a frame
    /// allocates nothing per object. The caller already holds the mapping: it
    /// is the object's slot list resolved through the compiled materials, and
    /// it changes when the document does rather than when the camera moves.
    ///
    /// Empty is the ordinary case and means "every slot is `material`". A slot
    /// past the end of this falls back the same way, so an object wearing
    /// fewer materials than its mesh has slots draws in the first one instead
    /// of vanishing.
    pub materials: &'a [u32],
    /// How much of this object to draw, filled in by [`Viewport::prepare`].
    ///
    /// Carried on the draw rather than recomputed in each pass because four
    /// passes consult it — solid, faces, wireframe, points — and the id pass
    /// has to agree with all of them or the user can click something that was
    /// never drawn.
    pub detail: Detail,
}

impl<'a> Draw<'a> {
    pub fn new(mesh: &'a GpuMesh, transform: DAffine3) -> Self {
        Self {
            mesh,
            transform,
            selected: false,
            active: false,
            object_id: 0,
            preview: false,
            editing: false,
            material: 0,
            materials: &[],
            weights: false,
            // Full until measured. A draw that is never passed through
            // `prepare` renders completely, which is the safe default: the
            // failure mode of forgetting to measure is a slow frame, not a
            // missing object.
            detail: Detail::FULL,
        }
    }

    /// Colours the object by weight in the studio and flat modes: the first UV
    /// coordinate of each vertex, through Blender's weight-paint ramp. The
    /// application puts the weights there, on a copy made for the screen.
    pub fn weights(mut self, on: bool) -> Self {
        self.weights = on;
        self
    }

    /// Sets which material this object is made of, for the rendered mode.
    pub fn with_material(mut self, material: u32) -> Self {
        self.material = material;
        self
    }

    /// Sets one material per mesh material slot.
    ///
    /// For an object whose faces are not all on one slot. The slice outlives
    /// the draw, which is what keeps a frame free of allocation — see
    /// [`Draw::materials`].
    pub fn with_materials(mut self, materials: &'a [u32]) -> Self {
        self.materials = materials;
        self
    }

    /// Which compiled material a mesh slot draws in.
    ///
    /// Falls back to [`Draw::material`] for a slot the object does not name,
    /// so an object wearing fewer materials than its mesh has slots draws
    /// those faces in its first material rather than not at all. A hole in the
    /// model is the worse answer: it reads as broken geometry, and the thing
    /// that is actually missing is an assignment.
    pub fn material_for(&self, slot: u32) -> u32 {
        self.materials
            .get(slot as usize)
            .copied()
            .unwrap_or(self.material)
    }

    /// Marks this draw as the one being edited, exempting it from every
    /// screen-size simplification.
    pub fn editing(mut self, yes: bool) -> Self {
        self.editing = yes;
        self
    }

    /// Whether simplification is forbidden for this draw.
    ///
    /// The plan's rule, in one place: never simplify the object being edited or
    /// anything selected. Both are cases where the user is looking *for* the
    /// thing, and an editor that hides what was selected has told a lie about
    /// what exists.
    pub fn protected(&self) -> bool {
        self.editing || self.selected
    }

    pub fn selected(mut self, yes: bool) -> Self {
        self.selected = yes;
        self
    }

    /// Marks the active object, whose outline is lighter.
    pub fn active(mut self, yes: bool) -> Self {
        self.active = yes;
        self
    }

    /// What the id pass reports when this draw is picked.
    pub fn object_id(mut self, id: u32) -> Self {
        self.object_id = id;
        self
    }

    /// Marks this draw as the one a rubber-band drag is resolving against.
    pub fn preview(mut self, yes: bool) -> Self {
        self.preview = yes;
        self
    }
}

/// The cluster draw's group 0: the shared globals and the visible list.
///
/// A free function rather than a method because it is called from inside
/// `Viewport::new`, before a `Viewport` exists, as well as on every
/// reallocation. One definition, so the two cannot disagree about binding order.
fn cluster_globals_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    globals: &wgpu::Buffer,
    visible: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cluster globals"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: visible.as_entire_binding(),
            },
        ],
    })
}

/// The cluster draw's group 1: one instance slot, reached by dynamic offset.
fn cluster_instance_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    instances: &wgpu::Buffer,
    size: u64,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cluster instance"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: instances,
                offset: 0,
                size: std::num::NonZeroU64::new(size),
            }),
        }],
    })
}

/// A bind group layout of `count` read-only storage buffers.
///
/// The cluster draw reads four arrays that are all the same kind of thing, so
/// spelling out four near-identical entries would be four chances to
/// mistranscribe a binding index.
fn storage_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    count: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..count)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

/// A growable per-frame uniform array addressed by dynamic offset.
///
/// Object data is per *draw*, not per mesh: two objects can share one mesh
/// datablock — that is the whole point of referencing geometry by id — and each
/// needs its own transform. Writing into a buffer owned by the mesh means the
/// second draw overwrites the first, and both linked duplicates land on top of
/// each other.
struct UniformRing {
    buffer: wgpu::Buffer,
    bind: wgpu::BindGroup,
    /// Distance between slots, rounded up to the device's alignment.
    stride: u32,
    capacity: u32,
}

impl UniformRing {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        element_size: u64,
        capacity: u32,
        label: &str,
    ) -> Self {
        // Dynamic offsets must be a multiple of the device's alignment, which is
        // typically 256 bytes and never smaller than the struct.
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = element_size.div_ceil(align) * align;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: stride * capacity.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(element_size),
                }),
            }],
        });
        Self {
            buffer,
            bind,
            stride: stride as u32,
            capacity: capacity.max(1),
        }
    }

    fn offset(&self, index: u32) -> u32 {
        self.stride * index
    }

    /// Grows to hold `needed` slots, reallocating only when it must.
    fn ensure(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        element_size: u64,
        needed: u32,
        label: &str,
    ) {
        if needed <= self.capacity {
            return;
        }
        // Geometric growth, so a scene that grows steadily does not reallocate
        // on every object added.
        let capacity = needed.next_power_of_two();
        *self = Self::new(device, layout, element_size, capacity, label);
    }
}

/// wgpu rejects a zero-sized buffer, and an empty mesh is a perfectly ordinary
/// document state. One dummy element costs nothing and keeps the bind group
/// valid; nothing is ever drawn from it because the instance count is zero.
fn pad_if_empty_u32(values: &[u32]) -> Vec<u32> {
    if values.is_empty() {
        vec![0]
    } else {
        values.to_vec()
    }
}

fn pad_if_empty_u32x2(values: &[[u32; 2]]) -> Vec<[u32; 2]> {
    if values.is_empty() {
        vec![[0, 0]]
    } else {
        values.to_vec()
    }
}

/// A bind group layout holding one uniform buffer, visible to both stages.
fn uniform_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    uniform_layout(device, label, false, None)
}

/// The same, addressed by a dynamic offset so one buffer can hold a slot per draw.
fn dynamic_uniform_layout(
    device: &wgpu::Device,
    label: &str,
    element_size: u64,
) -> wgpu::BindGroupLayout {
    uniform_layout(device, label, true, std::num::NonZeroU64::new(element_size))
}

fn uniform_layout(
    device: &wgpu::Device,
    label: &str,
    dynamic: bool,
    min_binding_size: Option<std::num::NonZeroU64>,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size,
            },
            count: None,
        }],
    })
}

/// Layout for the wireframe pass: the mesh's three read-only storage buffers.
///
/// Per-mesh, so it holds nothing that varies between two objects sharing one
/// datablock — the transform lives in a separate, per-draw group.
fn edge_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("edge"),
        entries: &[storage(0), storage(1), storage(2)],
    })
}

/// The size of one per-draw id uniform slot.
const ID_OBJECT_SIZE: u64 = std::mem::size_of::<IdObject>() as u64;

/// The size of one per-draw element-overlay uniform slot.
const ELEMENT_OBJECT_SIZE: u64 = std::mem::size_of::<ElementObject>() as u64;

/// What a rubber-band drag in progress would do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreviewSelection {
    /// Which element domain the bitset indexes.
    pub domain: PickDomain,
    /// Whether the drag removes from the selection rather than adding.
    pub deselect: bool,
    /// How many elements the bitset covers, so a stale bit past the end of a
    /// shrunken mesh cannot light up an element that no longer exists.
    pub elements: u32,
}

/// The colour every selected element is drawn in.
///
/// One value for vertices, edges and faces: three shades of orange would read
/// as three different states rather than as one selection. It comes from
/// [`crate::palette`] because it is the same accent the interface uses for a
/// selected row, and it goes through the transfer function on the way because
/// this value is written to an sRGB target — see the palette's module docs for
/// what it looked like when it did not.
fn selected_color() -> [f32; 4] {
    palette::display_rgba(palette::ACCENT)
}

/// Opacity of the selected-face fill.
///
/// Translucent so the shading underneath still reads: an opaque fill hides the
/// form of the thing being worked on.
const FACE_FILL_ALPHA: f32 = 0.28;

/// One id pipeline per element domain, in an occluded and an x-ray variant.
///
/// Six pipelines from one shader module. Built at startup rather than on first
/// use: pipeline creation is expensive and unpredictable, and the one place it
/// must never happen is partway through a drag — which is exactly when a user
/// first switches to vertex mode and starts clicking.
struct IdPipelines {
    /// Indexed by [`PickDomain`] through [`IdPipelines::slot`], then by whether
    /// occluded elements are selectable.
    occluded: [wgpu::RenderPipeline; 3],
    xray: [wgpu::RenderPipeline; 3],
}

impl IdPipelines {
    fn new(
        device: &wgpu::Device,
        globals: &wgpu::BindGroupLayout,
        mesh: &wgpu::BindGroupLayout,
        object: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("id"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/id.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("id layout"),
            bind_group_layouts: &[Some(globals), Some(mesh), Some(object)],
            immediate_size: 0,
        });

        let build = |entry: &str, topology, xray: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("id"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    // No vertex buffers anywhere in this pass: every path reads
                    // geometry from storage and builds its primitives from the
                    // vertex and instance indices.
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_id"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: crate::picking::ID_FORMAT,
                        // Ids are not colours. Blending two ids together would
                        // produce a third that names nothing.
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    // No culling, for the same reason the solid pass does none: a
                    // modeller routinely has inconsistently wound geometry
                    // mid-edit, and an element that is drawn has to be clickable.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(if xray {
                    // See-through: occluded elements are selectable, which is the
                    // whole point of the mode. Two pipeline variants of one
                    // shader, and the semantics fall out of the depth state.
                    gpu::depth_ignore_state()
                } else {
                    gpu::depth_read_state()
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let list = wgpu::PrimitiveTopology::TriangleList;
        let strip = wgpu::PrimitiveTopology::TriangleStrip;
        Self {
            occluded: [
                build("vs_face", list, false),
                build("vs_edge", strip, false),
                build("vs_vert", strip, false),
            ],
            xray: [
                build("vs_face", list, true),
                build("vs_edge", strip, true),
                build("vs_vert", strip, true),
            ],
        }
    }

    /// Which of the three geometry paths a domain draws with.
    ///
    /// Object mode draws faces: an object is picked by the surface the user
    /// clicked, and only the object id is read back.
    fn slot(domain: PickDomain) -> usize {
        match domain {
            PickDomain::Object | PickDomain::Face => 0,
            PickDomain::Edge => 1,
            PickDomain::Vert => 2,
        }
    }

    fn get(&self, domain: PickDomain, xray: bool) -> &wgpu::RenderPipeline {
        let slot = Self::slot(domain);
        if xray {
            &self.xray[slot]
        } else {
            &self.occluded[slot]
        }
    }
}

/// Layout for the rubber-band preview: what a drag in progress would select.
///
/// Its own group rather than folded into the per-mesh or per-draw ones, because
/// it is neither: the bitset belongs to the *drag*, and is the same for every
/// draw in the frame.
fn preview_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("preview"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Layout for the passes that read geometry out of storage: the id pass, the
/// selected-face fill and the vertex points.
///
/// One layout for all three on purpose. They have to agree about what geometry
/// *is* — what is highlighted must be what is pickable — and sharing the binding
/// makes that structural rather than a convention someone maintains.
///
/// Per-mesh, so nothing here varies between two objects sharing a datablock. The
/// object id and the transform do vary, so they live in the per-draw group.
fn mesh_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh"),
        entries: &[
            storage(0),
            storage(1),
            storage(2),
            storage(3),
            storage(4),
            storage(5),
            storage(6),
        ],
    })
}

/// Everything needed to draw one viewport.
pub struct Viewport {
    solid: wgpu::RenderPipeline,
    /// The scene-lit pass. See [`ShadingMode::Rendered`].
    /// The compiled materials, in the order the application set them.
    ///
    /// Empty until [`Viewport::set_materials`] is called, and a rendered pass
    /// with none draws no geometry — a material is not optional the way a
    /// light is. The end-to-end test is what says the application supplies
    /// them; there is no built-in fallback, because a fallback would be a
    /// second `material_sample` and §3's fifth decision is that there is one.
    materials: Vec<crate::material::Material>,
    /// One pipeline per distinct compiled shader, keyed on the text.
    ///
    /// The whole reason a material's constants live in a uniform: a hundred
    /// tinted plastics are one shape and share this.
    material_pipelines: crate::material::PipelineCache,
    /// Which pipeline each material uses, parallel to `materials`.
    material_pipeline_of: Vec<String>,
    material_layout: wgpu::BindGroupLayout,
    /// Everything a rendered pipeline needs besides the material's own code.
    rendered_layout: wgpu::PipelineLayout,
    rendered_object_layout: wgpu::BindGroupLayout,
    rendered_objects: UniformRing,
    /// Group 2: the DFG table, the lights, and the frame's shading globals.
    ///
    /// The convention the shading libraries declare, because WGSL will not
    /// take a storage pointer as a function parameter and a library that needs
    /// a buffer has to declare it.
    shading_layout: wgpu::BindGroupLayout,
    shading_bind: wgpu::BindGroup,
    dfg_buf: wgpu::Buffer,
    shading_globals_buf: wgpu::Buffer,
    lights: crate::lights::Lights,
    /// The depth maps, and the per-light entries that name them.
    shadows: crate::shadow::ShadowMaps,
    shadow_buf: wgpu::Buffer,
    /// What each light's shadow is, parallel to the light buffer.
    shadow_entries: Vec<crate::shadow::GpuShadow>,
    /// The frames this frame's shadow passes were built with, in layer order.
    shadow_views: Vec<crate::shadow::ShadowView>,
    /// Which lights reach which square of the screen.
    tiles: crate::tiles::Tiles,
    /// This frame's lights, packed once.
    ///
    /// Kept because the shadow frames are fitted in `prepare` — which is where
    /// the scene's extent is known — and packing them a second time there
    /// would be a second place for a light to mean something.
    packed_lights: Vec<crate::lights::GpuLight>,
    /// The linear image the rendered pass shades into, and the pass that turns
    /// it into a picture.
    hdr: crate::display::HdrTarget,
    screen: crate::display::ScreenResolve,
    /// The resolve's uniform and its bind group.
    ///
    /// Owned here so a frame costs a sixteen byte write: the bind group is
    /// rebuilt when the intermediate is, which is on a resize.
    screen_params: wgpu::Buffer,
    screen_bind: wgpu::BindGroup,
    /// The world's radiance, linear. See [`Viewport::set_world`].
    ///
    /// What lights the scene when there is no environment, and it is not a
    /// special case of one: a uniform environment supplies exactly `πL` and
    /// `L` to the same function. See `shading::ambient_from`.
    world: [f32; 3],
    /// The prefiltered environment, or a one-texel placeholder.
    ///
    /// Always something, because a bind group cannot have a hole in it.
    /// `has_environment` is what says whether it means anything.
    environment: crate::environment::Environment,
    has_environment: bool,
    /// The environment's turn about up, in radians.
    world_rotation: f32,
    /// Draws the environment where there is no geometry.
    background_pipeline: wgpu::RenderPipeline,
    /// The camera's inverse, for the background's rays.
    ///
    /// Kept rather than recomputed where it is used, because `prepare` is the
    /// call that has the camera and the frame has exactly one.
    inv_view_proj: [[f32; 4]; 4],
    /// How much light the environment gives.
    ///
    /// Folded in at lookup rather than into the cube, for the same reason the
    /// rotation is: dragging the slider costs a uniform write instead of six
    /// faces and their mips. A flat world has it folded into the colour
    /// already, by `World::flat_radiance`.
    world_strength: f32,
    /// Casters the draw list does not contain, in eye-relative metres.
    ///
    /// Normally empty. What the shadows are fitted to is what is *drawn*, which
    /// [`Viewport::prepare`] works out for itself — see `set_extra_bounds` for
    /// why this exists at all.
    extra_bounds: kreate_math::Aabb,
    /// How the picture is looked at. See `kreate_math::DisplayTransform`.
    ///
    /// Private because the uniform behind it has to be written when it
    /// changes; [`Viewport::set_display`] does both.
    display: crate::display::Display,
    grid: wgpu::RenderPipeline,
    globals_buf: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,
    grid_buf: wgpu::Buffer,
    grid_bind: wgpu::BindGroup,
    edges: wgpu::RenderPipeline,
    edge_globals_buf: wgpu::Buffer,
    /// The markers, lamp and camera glyphs, drawn over the picture against
    /// its depth. See [`crate::markers`].
    marker_globals_buf: wgpu::Buffer,
    marker_globals_bind: wgpu::BindGroup,
    marker_storage_layout: wgpu::BindGroupLayout,
    marker_lines: crate::markers::GrowingStorage,
    marker_points: crate::markers::GrowingStorage,
    marker_line_pipeline: wgpu::RenderPipeline,
    marker_point_pipeline: wgpu::RenderPipeline,
    edge_globals_bind: wgpu::BindGroup,
    object_layout: wgpu::BindGroupLayout,
    edge_object_layout: wgpu::BindGroupLayout,
    /// Per-draw object uniforms for the solid and wireframe passes.
    objects: UniformRing,
    edge_objects: UniformRing,
    /// One pipeline per element domain, in two occlusion variants. Built up
    /// front rather than on first use: a pipeline compiled during a drag is a
    /// visible hitch, and there are six of them.
    id_pipelines: IdPipelines,
    id_globals_buf: wgpu::Buffer,
    id_globals_bind: wgpu::BindGroup,
    /// The selected-face fill and the vertex points: the other two thirds of
    /// element selection.
    faces: wgpu::RenderPipeline,
    points: wgpu::RenderPipeline,
    element_globals_buf: wgpu::Buffer,
    element_globals_bind: wgpu::BindGroup,
    element_object_layout: wgpu::BindGroupLayout,
    element_objects: UniformRing,
    preview_layout: wgpu::BindGroupLayout,
    preview_buf: wgpu::Buffer,
    /// Which generation of the bitset the current bind group points at, so it is
    /// rebuilt when the reduction grows one and not on every frame of a drag.
    ///
    /// `None` means it is still pointing at the idle buffer.
    preview_generation: Option<u64>,
    /// Rebuilt whenever the bitset buffer changes, which is whenever the edited
    /// mesh grows past the last drag's capacity.
    preview_bind: wgpu::BindGroup,
    /// A zeroed bitset, bound when no drag has ever run.
    ///
    /// Cheaper than a second set of pipelines without the group, and it means
    /// the pass list has exactly one shape.
    #[allow(dead_code)]
    preview_idle_bits: wgpu::Buffer,
    id_object_layout: wgpu::BindGroupLayout,
    id_objects: UniformRing,
    pub layouts: MeshLayouts,
    /// The colour format this viewport's pipelines were built for.
    ///
    /// Kept so the warm-up can allocate a scratch target the pipelines will
    /// accept; a render pass whose attachment format differs from the pipeline
    /// is a validation error, not a silent mismatch.
    color_format: wgpu::TextureFormat,
    cull: ClusterCull,
    /// The depth pyramid, for occlusion culling.
    ///
    /// Owned here because it is built from *this* viewport's depth buffer and
    /// sized to its render target.
    hiz: HiZ,
    /// Whether the cull rejects occluded clusters.
    ///
    /// The two-pass structure that makes this sound is implemented — see
    /// [`Self::render_frame`] — so this is now safe to turn on. It stays off by
    /// default for the same reason [`Self::cluster_path`] does: the cluster path
    /// it depends on measures slower than the direct draw.
    ///
    /// Honest about the evidence: the second pass provably does work, but no
    /// scene has been found in which removing it changes a pixel. See the module
    /// docs of `tests/two_pass.rs`.
    pub occlusion_culling: bool,
    cluster_solid: wgpu::RenderPipeline,
    cluster_globals_layout: wgpu::BindGroupLayout,
    cluster_instance_layout: wgpu::BindGroupLayout,
    /// Whether the missing-clusters warning has already been said.
    ///
    /// Once per viewport, not once per frame: a warning repeated sixty times a
    /// second is a warning nobody reads.
    cluster_warning: std::sync::atomic::AtomicBool,
    /// Rebuilt when the cull reallocates; see [`ClusterCull::generation`].
    cluster_globals_bind: wgpu::BindGroup,
    cluster_instance_bind: wgpu::BindGroup,
    cluster_generation: u64,
    /// Where the surfaces go to have their edges smoothed, when
    /// [`Viewport::antialias`] is on.
    aa: crate::fxaa::Fxaa,
    /// The orange line around selected objects.
    outline: crate::outline::Outline,
    /// The path tracer, for the rendered view when it is asked for.
    pathtracer: crate::pathtrace::PathTracer,
    /// Whether the rendered view is path traced rather than rasterised.
    pub path_trace: bool,
    /// Filters the path tracer's young mean.
    denoise: crate::denoise::Denoise,
    /// Whether the path tracer's picture is denoised.
    pub path_trace_denoise: bool,
    /// The path tracer's lens, when the view is through a camera with an
    /// aperture: depth of field.
    pub path_trace_lens: Option<crate::pathtrace::Lens>,
    /// The rendered view's reflections of what is on screen.
    ssr: crate::ssr::Ssr,
    /// Whether the rendered view reflects what is on screen.
    pub show_reflections: bool,
    /// The rendered view's glow around what is brighter than white.
    bloom: crate::bloom::Bloom,
    /// Whether the rendered view glows around its brightest light.
    pub show_bloom: bool,
    /// Whether the rendered view darkens where things touch, past what the
    /// shadow maps resolve. See `shading.wgsl`'s `contact_shadow`.
    pub contact_shadows: bool,
    /// The rendered view's depth, drawn before it is shaded, for contact
    /// shadows. Its own texture: the shading pass cannot read the depth it is
    /// writing.
    contact_depth: wgpu::TextureView,
    contact_pipeline: wgpu::RenderPipeline,
    /// The camera the rendered pass draws with this frame, sample nudge and
    /// all.
    shading_view_proj: [[f32; 4]; 4],
    /// Which materials let light through. See `set_transmissive`.
    material_transmissive: Vec<bool>,
    /// The solids' picture, copied before glass is drawn over it: what glass
    /// shows through. Linear, the picture's own format.
    behind: (wgpu::Texture, wgpu::TextureView),
    /// How strong the glow is.
    pub bloom_intensity: f32,
    /// The running mean a still view refines itself into.
    accum: crate::accum::Accum,
    /// Whether a still view refines itself over the next frames.
    pub accumulate: bool,
    /// Frames already in the mean.
    accum_samples: u32,
    /// What the mean is a picture of; a different one starts it again.
    accum_signature: u64,
    /// Whether this frame goes through the mean at all.
    accum_applies: bool,
    /// This frame's place in the mean, one-based; `None` when the mean is
    /// full and the frame only presents it.
    frame_sample: Option<u32>,
    /// What the lights, the world and the display transform were last set
    /// to, for the mean's signature: a lamp moved is a different picture.
    lighting_signature: u64,
    /// Whether selected objects are outlined.
    pub show_outline: bool,
    /// The surfaces see-through, Blender's X-ray: every edge behind shows.
    pub xray: bool,
    solid_xray: wgpu::RenderPipeline,
    /// Darkens the studio view's corners and creases.
    ao: crate::ao::Ao,
    /// Whether the studio view darkens corners and creases.
    pub ambient_occlusion: bool,
    /// Smooths the surfaces' edges in the studio, flat and wireframe views.
    pub antialias: bool,
    depth: wgpu::TextureView,
    /// Allocated size of the render targets.
    size: (u32, u32),
    /// The sub-rectangle actually drawn into.
    ///
    /// The offscreen target is allocated in coarse steps so a panel drag does not
    /// reallocate it every frame, which means it is usually *larger* than the
    /// area on screen. Rendering into the whole thing and then sampling all of it
    /// into a smaller rectangle stretches the image — up to 7% at a 64-pixel
    /// step, which on a cube reads as "it looks wrong" rather than as a bug with
    /// an obvious cause. So the pass is restricted to the exact region and the
    /// caller samples exactly that much.
    render_size: (u32, u32),
    pub shading: ShadingMode,
    pub background: wgpu::Color,
    /// Whether the solid pass goes through cluster culling.
    ///
    /// Off by default, and that is a measurement rather than timidity: it is
    /// currently 2.4x *slower* than the direct path on ten million triangles,
    /// because a non-indexed cluster draw gives up the post-transform vertex
    /// cache. See the module docs of [`crate::cluster_cull`] for the numbers and
    /// for what would change them.
    ///
    /// Kept, and kept working, because occlusion culling needs exactly this
    /// machinery and is where the win actually is.
    pub cluster_path: bool,
    /// Screen-size thresholds for culling and overlay density.
    ///
    /// Public so preferences can reach it: a user on a small laptop screen may
    /// want the wireframe to thin out sooner, and one presenting to a room may
    /// want it never to.
    pub lod_budget: Budget,
    /// Whether to draw the wireframe overlay.
    pub show_wireframe: bool,
    /// Whether to draw the element overlays: selected-face fill and vertex
    /// points. On in edit mode and off outside it — a scene full of vertex dots
    /// is unreadable when the user is arranging objects.
    pub show_elements: bool,
    /// Vertex dot diameter, in pixels.
    pub point_size: f32,
    /// Wireframe line width, in pixels.
    pub wireframe_width: f32,
    /// Whether to draw the ground grid. A user-facing toggle, and what lets a
    /// test compare geometry without the world-anchored lattice in the way.
    pub show_grid: bool,
    /// Colour faces blue from the front and red from the back — Blender's Face
    /// Orientation overlay. Solid shading only.
    pub face_orientation: bool,
    /// Display scaling, applied to the pick tolerance.
    ///
    /// A tolerance chosen in pixels is a tolerance chosen for a particular
    /// screen density. Without this, picking on a high-density display is a
    /// third as forgiving as it was designed to be — which reads as the
    /// application being fussy about where you click.
    pub pick_scale: f32,
}

impl Viewport {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let globals_layout = uniform_bind_group_layout(device, "globals");
        let object_layout = dynamic_uniform_layout(device, "object", OBJECT_SIZE);
        let edge_object_layout = dynamic_uniform_layout(device, "edge object", EDGE_OBJECT_SIZE);
        let grid_layout = uniform_bind_group_layout(device, "grid globals");

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid globals"),
            size: std::mem::size_of::<GridGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = |layout: &wgpu::BindGroupLayout, buf: &wgpu::Buffer, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            })
        };
        let edge_globals_layout = uniform_bind_group_layout(device, "edge globals");
        let cluster_cull = ClusterCull::new(device);
        let layouts = MeshLayouts {
            edge: edge_bind_group_layout(device),
            id: mesh_bind_group_layout(device),
            // Owned by the cull, because the cull's shader declares it and the
            // two must agree. Copied out here so a mesh can build its bind group
            // without reaching into the cull.
            cluster: cluster_cull.cluster_layout().clone(),
            cluster_draw: storage_bind_group_layout(
                device,
                "cluster draw",
                4,
                wgpu::ShaderStages::VERTEX,
            ),
        };
        let id_globals_layout = uniform_bind_group_layout(device, "id globals");
        let id_object_layout = dynamic_uniform_layout(device, "id object", ID_OBJECT_SIZE);
        let id_globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("id globals"),
            size: std::mem::size_of::<IdGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let element_globals_layout = uniform_bind_group_layout(device, "element globals");
        let element_object_layout =
            dynamic_uniform_layout(device, "element object", ELEMENT_OBJECT_SIZE);
        let element_globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("element globals"),
            size: std::mem::size_of::<ElementGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let preview_layout = preview_bind_group_layout(device);
        let preview_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preview params"),
            size: std::mem::size_of::<PreviewParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // A zeroed bitset to bind when no drag is in progress, so the pass list
        // has exactly one shape and the preview group is never absent.
        let preview_idle_bits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preview idle bits"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let preview_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview bind"),
            layout: &preview_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preview_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preview_idle_bits.as_entire_binding(),
                },
            ],
        });
        let edge_globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge globals"),
            size: std::mem::size_of::<EdgeGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let id_globals_bind = bind(&id_globals_layout, &id_globals_buf, "id globals bind");
        let element_globals_bind = bind(
            &element_globals_layout,
            &element_globals_buf,
            "element globals bind",
        );
        let globals_bind = bind(&globals_layout, &globals_buf, "globals bind");
        let grid_bind = bind(&grid_layout, &grid_buf, "grid bind");
        let edge_globals_bind = bind(&edge_globals_layout, &edge_globals_buf, "edge globals bind");

        // The markers' three groups: one uniform block, and a storage buffer
        // each for the lines and the rings, grown on demand.
        let marker_globals_layout = uniform_layout(device, "marker globals layout", false, None);
        let marker_globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("marker globals"),
            size: std::mem::size_of::<MarkerGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let marker_globals_bind = bind(
            &marker_globals_layout,
            &marker_globals_buf,
            "marker globals bind",
        );
        let marker_storage_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("marker storage layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let marker_lines = crate::markers::GrowingStorage::new(
            device,
            &marker_storage_layout,
            "marker lines",
            std::mem::size_of::<crate::markers::GpuLine>() as u64,
        );
        let marker_points = crate::markers::GrowingStorage::new(
            device,
            &marker_storage_layout,
            "marker points",
            std::mem::size_of::<crate::markers::GpuPoint>() as u64,
        );

        let solid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("solid"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/solid.wgsl").into()),
        });
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grid.wgsl").into()),
        });
        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay_edge"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay_edge.wgsl").into()),
        });

        let solid_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solid layout"),
            bind_group_layouts: &[Some(&globals_layout), Some(&object_layout)],
            immediate_size: 0,
        });
        let grid_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grid layout"),
            bind_group_layouts: &[Some(&grid_layout)],
            immediate_size: 0,
        });

        let edge_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("edge layout"),
            bind_group_layouts: &[
                Some(&edge_globals_layout),
                Some(&layouts.edge),
                Some(&edge_object_layout),
                Some(&preview_layout),
            ],
            immediate_size: 0,
        });
        let marker_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("markers"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/markers.wgsl").into()),
        });
        let marker_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("marker layout"),
            bind_group_layouts: &[
                Some(&marker_globals_layout),
                Some(&marker_storage_layout),
                Some(&marker_storage_layout),
            ],
            immediate_size: 0,
        });
        // The edge overlay's recipe: a quad from the vertex and instance
        // indices, alpha-blended, tested against the solid pass but not
        // writing depth.
        let marker_pipeline = |label: &str, vs: &str, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&marker_pl),
                vertex: wgpu::VertexState {
                    module: &marker_shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &marker_shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(gpu::depth_read_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let marker_line_pipeline = marker_pipeline("marker lines", "vs_line", "fs_line");
        let marker_point_pipeline = marker_pipeline("marker points", "vs_point", "fs_point");

        let id_pipelines =
            IdPipelines::new(device, &id_globals_layout, &layouts.id, &id_object_layout);

        let element_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay_element"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay_element.wgsl").into()),
        });
        let element_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("element layout"),
            bind_group_layouts: &[
                Some(&element_globals_layout),
                Some(&layouts.id),
                Some(&element_object_layout),
                Some(&preview_layout),
            ],
            immediate_size: 0,
        });
        let element_pipeline = |label: &str, vs: &str, fs: &str, topology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&element_pl),
                vertex: wgpu::VertexState {
                    module: &element_shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &element_shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    cull_mode: None,
                    ..Default::default()
                },
                // Tests against the solid pass but does not write. Overlays layer
                // by draw order and depth offset rather than fighting each other,
                // which is what stops the shimmer while orbiting.
                depth_stencil: Some(gpu::depth_read_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let faces = element_pipeline(
            "selected faces",
            "vs_face",
            "fs_face",
            wgpu::PrimitiveTopology::TriangleList,
        );
        let points = element_pipeline(
            "vertex points",
            "vs_point",
            "fs_point",
            wgpu::PrimitiveTopology::TriangleStrip,
        );
        let edges = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edges"),
            layout: Some(&edge_pl),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // No vertex buffers: the shader reads positions from storage and
                // builds a quad from the vertex and instance indices, so a mesh's
                // whole wireframe is one draw call.
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            // Tests against the solid pass but does not write: overlays layer by
            // draw order and depth offset rather than fighting each other, which
            // is what stops the shimmer while orbiting.
            depth_stencil: Some(gpu::depth_read_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- the rendered pass ------------------------------------------
        //
        // Built up front rather than on first use: a pipeline compiled the
        // frame somebody switches mode is a visible hitch, and this one has
        // three libraries in it.
        let rendered_object_layout =
            dynamic_uniform_layout(device, "rendered object", RENDERED_OBJECT_SIZE);
        let shading_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shading data"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The rendered view's depth, for contact shadows.
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // The solids' picture, for what glass shows.
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        // The path tracer reads the lights and the environment through the
        // same group, which is why that layout is visible to compute.
        let pathtracer_0 = crate::pathtrace::PathTracer::new(device, &shading_layout);
        let dfg_buf = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("dfg"),
                contents: bytemuck::cast_slice(&crate::brdf::dfg_buffer(
                    crate::brdf::albedo_table(),
                )),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let shading_globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shading globals"),
            size: std::mem::size_of::<ShadingGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights = crate::lights::Lights::new(device, 8);
        let tiles_0 = crate::tiles::Tiles::new(device);
        let shadows_0 = crate::shadow::ShadowMaps::new(device);
        let shadow_buf_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow entries"),
            size: (std::mem::size_of::<crate::shadow::GpuShadow>() * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let environment_0 = crate::environment::Environment::placeholder(device);
        let contact_depth_0 = gpu::create_depth_texture(device, width, height);
        let behind_0 = behind_texture(device, width, height);
        let shading_bind = shading_bind_group(
            device,
            &shading_layout,
            &dfg_buf,
            lights.buffer(),
            &shading_globals_buf,
            &shadow_buf_0,
            shadows_0.array_view(),
            tiles_0.buffer(),
            &environment_0,
            &contact_depth_0,
            &behind_0.1,
        );
        // The rendered pass has no pipeline of its own any more: a material's
        // code *is* the fragment stage's second half, so the pipeline is built
        // when the material is. What is kept here is everything that does not
        // depend on which material it is.
        let material_layout = crate::material::layout(device);
        let rendered_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rendered layout"),
            bind_group_layouts: &[
                Some(&globals_layout),
                Some(&rendered_object_layout),
                Some(&shading_layout),
                Some(&material_layout),
            ],
            immediate_size: 0,
        });

        // The depth contact shadows march through: positions only, no
        // fragment stage, the rendered pass's camera and objects.
        let contact_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact depth"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/contact_depth.wgsl").into()),
        });
        let contact_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("contact depth layout"),
            bind_group_layouts: &[Some(&globals_layout), Some(&rendered_object_layout)],
            immediate_size: 0,
        });
        let contact_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("contact depth"),
            layout: Some(&contact_pl),
            vertex: wgpu::VertexState {
                module: &contact_shader,
                entry_point: Some("contact_vertex"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(gpu::depth_write_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // The background. The shading library, because it reads the same
        // uniform and the same cube — a second copy of either declaration is
        // the drift the parity tests exist to prevent.
        let background_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("background"),
            source: wgpu::ShaderSource::Wgsl(crate::shading::background_source().into()),
        });
        let background_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("background layout"),
            // Groups zero and one are not bound: a full-screen triangle has no
            // camera of its own and no object.
            bind_group_layouts: &[None, None, Some(&shading_layout)],
            immediate_size: 0,
        });
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("background"),
            layout: Some(&background_pl),
            vertex: wgpu::VertexState {
                module: &background_shader,
                entry_point: Some("background_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &background_shader,
                entry_point: Some("background_fragment"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: crate::display::HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // The pass's reflection images, which a background does
                    // not write: they stay clear, which is "nothing here".
                    Some(wgpu::ColorTargetState { format: crate::ssr::FORMAT, blend: None, write_mask: wgpu::ColorWrites::empty() }),
                    Some(wgpu::ColorTargetState { format: crate::ssr::FORMAT, blend: None, write_mask: wgpu::ColorWrites::empty() }),
                    Some(wgpu::ColorTargetState { format: crate::ssr::FORMAT, blend: None, write_mask: wgpu::ColorWrites::empty() }),
                ],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(gpu::depth_ignore_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let depth_0 = gpu::create_depth_texture(device, width, height);
        let outline_0 = crate::outline::Outline::new(
            device,
            color_format,
            &globals_layout,
            &object_layout,
            width.max(1),
            height.max(1),
        );
        let hdr_0 = crate::display::HdrTarget::new(device, width.max(1), height.max(1));
        let screen_0 = crate::display::ScreenResolve::new(device, color_format);
        let screen_params_buf = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("display screen params"),
                contents: &crate::display::ScreenResolve::params_of(
                    crate::display::Display::default(),
                ),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let screen_bind_0 = screen_0.bind_group(device, hdr_0.view(), &screen_params_buf);

        // Two of them: the opaque one, and X-ray's — blended over what is
        // behind and writing no depth, so every surface and every edge behind
        // it shows through, as Blender's Alt+Z does.
        let solid_pipeline = |label: &str, blend: Option<wgpu::BlendState>, depth: wgpu::DepthStencilState| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&solid_pl),
            vertex: wgpu::VertexState {
                module: &solid_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // Positions: tightly packed, 12 bytes.
                    wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
                    // Normals: one octahedron-encoded u32.
                    wgpu::VertexBufferLayout {
                        array_stride: 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 0,
                            shader_location: 1,
                        }],
                    },
                    // Surface coordinates, read only for the weight ramp.
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 2,
                        }],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &solid_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Backface culling stays off in the solid pass. A modeller
                // routinely has open and inconsistently wound geometry mid-edit,
                // and hiding it is worse than showing it: the fragment shader
                // flips the normal instead, so a backface is lit rather than
                // black or missing.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let solid = solid_pipeline("solid", None, gpu::depth_write_state());
        let solid_xray = solid_pipeline(
            "solid x-ray",
            Some(wgpu::BlendState::ALPHA_BLENDING),
            gpu::depth_read_state(),
        );

        // The cluster path: same shader module, same fragment entry point, no
        // vertex buffers at all. Everything comes from storage, indexed through
        // the visible list the cull produced.
        let cluster_globals_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cluster globals"),
                entries: &[
                    // Fragment too: `fs_main` reads the shading mode out of the
                    // same globals block, and it is deliberately the *same*
                    // fragment entry point the direct path uses.
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        // Binding 1, not 0: the direct path's `ObjectData` owns (1, 0) in this
        // module, and one module cannot declare two resources at one binding.
        let cluster_instance_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cluster instance"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let cluster_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster solid layout"),
            bind_group_layouts: &[
                Some(&cluster_globals_layout),
                Some(&cluster_instance_layout),
                Some(&layouts.cluster_draw),
            ],
            immediate_size: 0,
        });
        let cluster_solid = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cluster solid"),
            layout: Some(&cluster_pl),
            vertex: wgpu::VertexState {
                module: &solid_shader,
                entry_point: Some("vs_cluster"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &solid_shader,
                // The *same* fragment entry point as the direct path, which is
                // what makes the two provably shade alike rather than similarly.
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Off, as in the direct pass. The cone test has already removed
                // whole clusters that face away; what is left may still contain
                // inconsistently wound geometry, and a modeller needs to see it.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(gpu::depth_write_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let grid = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid"),
            layout: Some(&grid_pl),
            vertex: wgpu::VertexState {
                module: &grid_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            // The grid writes its own depth from the ray hit, so it is occluded
            // by geometry rather than floating over or under it.
            depth_stencil: Some(gpu::depth_write_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let cluster_globals_bind_0 = cluster_globals_bind(
            device,
            &cluster_globals_layout,
            &globals_buf,
            cluster_cull.visible_buffer(),
        );
        let cluster_instance_bind_0 = cluster_instance_bind(
            device,
            &cluster_instance_layout,
            cluster_cull.instances_buffer(),
            cluster_cull.instance_size(),
        );

        Self {
            solid,
            solid_xray,
            xray: false,
            grid,
            globals_buf,
            globals_bind,
            grid_buf,
            grid_bind,
            edges,
            edge_globals_buf,
            edge_globals_bind,
            marker_globals_buf,
            marker_globals_bind,
            marker_storage_layout,
            marker_lines,
            marker_points,
            marker_line_pipeline,
            marker_point_pipeline,
            id_pipelines,
            id_globals_buf,
            id_globals_bind,
            faces,
            points,
            element_globals_buf,
            element_globals_bind,
            element_objects: UniformRing::new(
                device,
                &element_object_layout,
                ELEMENT_OBJECT_SIZE,
                8,
                "element objects",
            ),
            element_object_layout,
            preview_layout,
            preview_buf,
            preview_generation: None,
            preview_bind,
            preview_idle_bits,
            id_objects: UniformRing::new(
                device,
                &id_object_layout,
                ID_OBJECT_SIZE,
                8,
                "id objects",
            ),
            id_object_layout,
            objects: UniformRing::new(device, &object_layout, OBJECT_SIZE, 8, "objects"),
            materials: Vec::new(),
            material_pipelines: crate::material::PipelineCache::new(),
            material_pipeline_of: Vec::new(),
            material_layout,
            rendered_layout,
            rendered_objects: UniformRing::new(
                device,
                &rendered_object_layout,
                RENDERED_OBJECT_SIZE,
                8,
                "rendered objects",
            ),
            rendered_object_layout,
            shading_layout,
            shading_bind,
            dfg_buf,
            shading_globals_buf,
            lights,
            shadows: shadows_0,
            shadow_buf: shadow_buf_0,
            shadow_entries: Vec::new(),
            shadow_views: Vec::new(),
            packed_lights: Vec::new(),
            tiles: tiles_0,
            screen_params: screen_params_buf,
            screen_bind: screen_bind_0,
            hdr: hdr_0,
            screen: screen_0,
            // Black: a document with no world is lit only by its lamps.
            world: [0.0; 3],
            environment: environment_0,
            background_pipeline,
            has_environment: false,
            world_rotation: 0.0,
            world_strength: 1.0,
            inv_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            extra_bounds: kreate_math::Aabb::EMPTY,
            display: crate::display::Display::default(),
            edge_objects: UniformRing::new(
                device,
                &edge_object_layout,
                EDGE_OBJECT_SIZE,
                8,
                "edge objects",
            ),
            object_layout,
            edge_object_layout,
            layouts,
            cluster_globals_bind: cluster_globals_bind_0,
            cluster_instance_bind: cluster_instance_bind_0,
            cluster_warning: std::sync::atomic::AtomicBool::new(false),
            cluster_generation: cluster_cull.generation(),
            color_format,
            cull: cluster_cull,
            hiz: HiZ::new(device, width.max(1), height.max(1)),
            occlusion_culling: false,
            cluster_solid,
            cluster_globals_layout,
            cluster_instance_layout,
            ao: crate::ao::Ao::new(device, color_format, &depth_0, width.max(1), height.max(1)),
            ambient_occlusion: true,
            depth: depth_0,
            aa: crate::fxaa::Fxaa::new(device, color_format, width.max(1), height.max(1)),
            outline: outline_0,
            accum: crate::accum::Accum::new(device, color_format, width.max(1), height.max(1)),
            accumulate: true,
            pathtracer: pathtracer_0,
            path_trace: false,
            path_trace_lens: None,
            denoise: crate::denoise::Denoise::new(device, width.max(1), height.max(1)),
            path_trace_denoise: true,
            ssr: crate::ssr::Ssr::new(device, width.max(1), height.max(1)),
            show_reflections: true,
            bloom: crate::bloom::Bloom::new(device),
            show_bloom: true,
            contact_shadows: true,
            contact_depth: contact_depth_0,
            contact_pipeline,
            shading_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            material_transmissive: Vec::new(),
            behind: behind_0,
            bloom_intensity: 0.12,
            accum_samples: 0,
            accum_signature: 0,
            accum_applies: false,
            frame_sample: None,
            lighting_signature: 0,
            show_outline: true,
            antialias: true,
            size: (width.max(1), height.max(1)),
            render_size: (width.max(1), height.max(1)),
            shading: ShadingMode::Studio,
            background: palette::clear(palette::VIEWPORT),
            show_grid: true,
            face_orientation: false,
            cluster_path: false,
            lod_budget: Budget::default(),
            show_wireframe: true,
            show_elements: false,
            wireframe_width: 1.5,
            point_size: 6.0,
            pick_scale: 1.0,
        }
    }

    /// Sets the allocated size of the depth target.
    ///
    /// Clamped to the device limits in the same way the depth texture is, so the
    /// recorded size always describes what was actually allocated.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let max = device.limits().max_texture_dimension_2d;
        let size = (width.clamp(1, max), height.clamp(1, max));
        if size != self.size {
            self.size = size;
            self.depth = gpu::create_depth_texture(device, size.0, size.1);
            self.contact_depth = gpu::create_depth_texture(device, size.0, size.1);
            self.behind = behind_texture(device, size.0, size.1);
            self.rebuild_shading_bind(device);
            self.aa.resize(device, size.0, size.1);
            self.outline.resize(device, size.0, size.1);
            self.accum.resize(device, size.0, size.1);
            self.ssr.resize(device, size.0, size.1);
            self.denoise.resize(device, size.0, size.1);
            self.ao.resize(device, &self.depth, size.0, size.1);
            // The linear intermediate is the same size as the depth it shares
            // a pass with. Remade only on a real change, because a texture
            // reallocated every frame is a stall — and the bind group holds
            // the old view, so it is remade with it.
            self.hdr.resize(device, size.0, size.1);
            self.screen_bind =
                self.screen
                    .bind_group(device, self.hdr.view(), &self.screen_params);
        }
        self.render_size = self.render_size.clamp_to(size);
    }

    /// Sets the exact region of the target that is drawn into.
    ///
    /// Must be no larger than the allocated size. Everything else — the aspect
    /// ratio, the pass viewport, and the UVs the caller should sample — follows
    /// from this rather than from the allocation.
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.render_size = (
            width.max(1).min(self.size.0),
            height.max(1).min(self.size.1),
        );
    }

    #[inline]
    pub fn render_size(&self) -> (u32, u32) {
        self.render_size
    }

    /// The fraction of the target that holds the image, for the caller's UVs.
    ///
    /// Sampling the whole texture instead is the stretch this exists to avoid.
    pub fn uv_extent(&self) -> (f32, f32) {
        (
            self.render_size.0 as f32 / self.size.0 as f32,
            self.render_size.1 as f32 / self.size.1 as f32,
        )
    }

    /// Aspect ratio of the drawn region, which is what the projection needs.
    pub fn aspect(&self) -> f64 {
        self.render_size.0 as f64 / self.render_size.1 as f64
    }

    /// Uploads per-frame uniforms.
    ///
    /// Takes `&mut self` because the per-draw uniform rings are sized by the
    /// number of *draws*, which the number of meshes does not bound.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        draws: &mut [Draw],
    ) {
        // The camera's inverse, for the background's rays. Written every frame
        // rather than when the lights change, because an orbit moves the camera
        // and nothing else — a background that only followed a light edit would
        // be a sky painted onto the screen.
        let aspect = self.render_size.0 as f64 / self.render_size.1.max(1) as f64;
        // Inverted in `f32`, which is what the projection already is: the
        // camera's matrix is single precision because it is *eye-relative*
        // (§v1's f64 world, narrowed after the subtraction), and inverting a
        // widened copy would be a longer way to the same numbers.
        //
        // What comes out of it is a direction, so the precision that matters is
        // angular and a metre at the far plane is nothing.
        self.inv_view_proj = camera
            .view_projection(aspect)
            .inverse()
            .to_cols_array_2d();
        queue.write_buffer(
            &self.shading_globals_buf,
            0,
            bytemuck::bytes_of(&self.shading_globals()),
        );

        // Measured first, because the uniforms written below carry the overlay
        // alphas the measurement produces. One place decides, and every pass —
        // including the id pass — reads that decision, so what is drawn and
        // what is clickable cannot disagree.
        for draw in draws.iter_mut() {
            let size = ViewSize::measure(
                draw.mesh.bounds,
                &draw.transform,
                camera,
                // The render target's height, not the widget's: at 70% dynamic
                // resolution there genuinely are fewer pixels per vertex, and
                // the decision should follow.
                self.render_size.1,
            );
            draw.detail = Detail::of(size, draw.mesh.counts, &self.lod_budget, draw.protected());
        }
        let draws: &[Draw] = draws;

        let aspect = self.aspect();

        if self.cluster_path {
            // Only the draws that survived the object-level cull are handed to
            // the cluster cull, and in the same order the draw loop will use:
            // the argument offset comes from the position in these slices, so a
            // caller that reordered in between would draw another mesh's
            // cluster list.
            let transforms: Vec<DAffine3> = draws
                .iter()
                .filter(|d| d.detail.draw)
                .map(|d| d.transform)
                .collect();
            let counts: Vec<u32> = draws
                .iter()
                .filter(|d| d.detail.draw)
                .map(|d| d.mesh.cluster_count)
                .collect();
            let colors: Vec<[f32; 4]> = draws
                .iter()
                .filter(|d| d.detail.draw)
                .map(|d| d.mesh.color.to_array())
                .collect();
            self.cull.prepare(
                device,
                queue,
                &CullView {
                    camera,
                    aspect,
                    hiz: &self.hiz,
                    occlusion: self.occlusion_culling && self.cluster_path,
                    render_height: self.render_size.1,
                    budget: &self.lod_budget,
                },
                &CullDraws {
                    transforms: &transforms,
                    cluster_counts: &counts,
                    colors: &colors,
                },
            );
            if self.cull.generation() != self.cluster_generation {
                // The shared buffers moved, so every bind group naming them is
                // stale. Rebuilt here rather than checked at draw time, because
                // a bind group that names a freed buffer fails validation at the
                // draw and the message points at the draw, not at the cause.
                self.cluster_globals_bind = cluster_globals_bind(
                    device,
                    &self.cluster_globals_layout,
                    &self.globals_buf,
                    self.cull.visible_buffer(),
                );
                self.cluster_instance_bind = cluster_instance_bind(
                    device,
                    &self.cluster_instance_layout,
                    self.cull.instances_buffer(),
                    self.cull.instance_size(),
                );
                self.cluster_generation = self.cull.generation();
            }
        }
        let view_proj = camera.view_projection(aspect);
        let view_proj = self.next_sample(view_proj, draws);
        if self.shading == ShadingMode::Rendered {
            self.shading_view_proj = view_proj.to_cols_array_2d();
            queue.write_buffer(
                &self.shading_globals_buf,
                0,
                bytemuck::bytes_of(&self.shading_globals()),
            );
        }
        if self.path_tracing() {
            if let Some(n) = self.frame_sample {
                self.pathtracer.prepare(
                    device,
                    queue,
                    camera.projection_matrix(aspect),
                    camera.view_rotation(),
                    camera.eye(),
                    self.render_size,
                    self.lights.count(),
                    n,
                    self.path_trace_lens,
                );
                if self.path_trace_denoise {
                    self.denoise.prepare(
                        device,
                        queue,
                        &self.ssr.gbuffer()[0],
                        &self.depth,
                        self.render_size,
                        n,
                    );
                }
            }
        }
        if self.shading == ShadingMode::Rendered && self.show_reflections {
            // As far as the view is from what it turns about, twice: a floor
            // reflects what stands on it, not the horizon.
            self.ssr.prepare(
                device,
                queue,
                &self.depth,
                self.hdr.view(),
                camera.projection_matrix(aspect),
                camera.view_rotation(),
                self.render_size,
                (camera.distance * 2.0).max(1e-3) as f32,
            );
        }
        if self.shading == ShadingMode::Rendered && self.show_bloom {
            // Over white once exposed: the picture's own exposure, so what
            // glows is what the display cannot show.
            self.bloom.prepare(
                device,
                queue,
                self.hdr.view(),
                self.render_size,
                self.size,
                self.display.exposure,
                self.bloom_intensity,
                1.0,
            );
        }
        self.aa.set_region(queue, self.render_size);
        // A reach that follows the view: a tenth of the way to what the view
        // is turning about, so the shading looks the same at any zoom.
        self.ao.set_view(
            queue,
            camera.projection_matrix(aspect),
            self.render_size,
            (camera.distance * 0.1).max(1e-4) as f32,
            // Gentler in the rendered view, where the lights already shade
            // the form and the occlusion is grounding rather than drawing it.
            if self.shading == ShadingMode::Rendered { 0.6 } else { 1.0 },
        );

        let right = camera.right().as_vec3();
        let up = camera.up().as_vec3();
        let back = right.cross(up).normalize_or_zero();
        queue.write_buffer(
            &self.globals_buf,
            0,
            bytemuck::bytes_of(&Globals {
                view_proj: view_proj.to_cols_array_2d(),
                params: [
                    if self.shading == ShadingMode::Studio {
                        0.0
                    } else {
                        1.0
                    },
                    if self.face_orientation { 1.0 } else { 0.0 },
                    // How opaque the surfaces are in X-ray; zero when off.
                    if self.xray { XRAY_ALPHA } else { 0.0 },
                    0.0,
                ],
                camera_right: right.extend(0.0).to_array(),
                camera_up: up.extend(0.0).to_array(),
                camera_back: back.extend(0.0).to_array(),
            }),
        );

        let inv_view = camera.view_rotation().inverse();
        let eye = camera.eye().as_vec3();
        let tan_half = match camera.projection {
            Projection::Perspective => (camera.fov_y * 0.5).tan() as f32,
            // The grid's ray construction assumes a perspective frustum. An
            // orthographic camera needs a different ray origin per pixel, so it
            // is handled by widening the rays to approximate the framing until
            // the ortho path is written properly.
            Projection::Orthographic => (camera.ortho_height() * 0.5) as f32,
        };
        queue.write_buffer(
            &self.grid_buf,
            0,
            bytemuck::bytes_of(&GridGlobals {
                view_proj: view_proj.to_cols_array_2d(),
                inv_view_row0: inv_view.row(0).to_array(),
                inv_view_row1: inv_view.row(1).to_array(),
                inv_view_row2: inv_view.row(2).to_array(),
                eye_and_tan: [eye.x, eye.y, eye.z, tan_half],
                params: [
                    aspect as f32,
                    if camera.projection == Projection::Perspective {
                        1.0
                    } else {
                        0.0
                    },
                    0.0,
                    0.0,
                ],
                line_color: palette::display_rgba(palette::GRID_LINE),
                axis_x_color: palette::display_rgba(palette::AXIS_X),
                axis_y_color: palette::display_rgba(palette::AXIS_Y),
            }),
        );

        queue.write_buffer(
            &self.edge_globals_buf,
            0,
            bytemuck::bytes_of(&EdgeGlobals {
                view_proj: view_proj.to_cols_array_2d(),
                params: [
                    self.render_size.0 as f32,
                    self.render_size.1 as f32,
                    self.wireframe_width,
                    // Overlay layer 2. A fraction of the fragment's own depth, so
                    // it is scale-invariant: the same nudge works at a centimetre
                    // and at a kilometre, which a constant bias cannot manage.
                    EDGE_DEPTH_OFFSET,
                ],
                non_manifold_color: palette::display_rgba(palette::EDGE_NON_MANIFOLD),
                seam_color: palette::display_rgba(palette::EDGE_SEAM),
                sharp_color: palette::display_rgba(palette::EDGE_SHARP),
            }),
        );
        queue.write_buffer(
            &self.marker_globals_buf,
            0,
            bytemuck::bytes_of(&MarkerGlobals {
                view_proj: view_proj.to_cols_array_2d(),
                params: [
                    self.render_size.0 as f32,
                    self.render_size.1 as f32,
                    // The wireframe's layer: a ring sitting on a wall wins the
                    // tie with the wall, and loses to a wall in front of it.
                    EDGE_DEPTH_OFFSET,
                    0.0,
                ],
            }),
        );

        queue.write_buffer(
            &self.element_globals_buf,
            0,
            bytemuck::bytes_of(&ElementGlobals {
                view_proj: view_proj.to_cols_array_2d(),
                params: [
                    self.render_size.0 as f32,
                    self.render_size.1 as f32,
                    self.point_size,
                    0.0,
                ],
                // The same layer offsets the id pass uses, taken from the same
                // place: the z order of the overlays on screen has to be the z
                // order in the id buffer, or the thing that looks clickable is
                // not the thing that gets clicked.
                layers: [
                    PickDomain::Face.depth_offset(),
                    PickDomain::Vert.depth_offset(),
                    FACE_FILL_ALPHA,
                    0.0,
                ],
            }),
        );

        let count = draws.len() as u32;
        self.objects
            .ensure(device, &self.object_layout, OBJECT_SIZE, count, "objects");
        self.shadows.reserve(device, count);
        self.rendered_objects.ensure(
            device,
            &self.rendered_object_layout,
            RENDERED_OBJECT_SIZE,
            count,
            "rendered objects",
        );
        self.element_objects.ensure(
            device,
            &self.element_object_layout,
            ELEMENT_OBJECT_SIZE,
            count,
            "element objects",
        );
        self.edge_objects.ensure(
            device,
            &self.edge_object_layout,
            EDGE_OBJECT_SIZE,
            count,
            "edge objects",
        );

        // What the shadows will be fitted to, accumulated as the draws are
        // written rather than in a pass of its own.
        let mut drawn = self.extra_bounds;

        // One slot per draw, not per mesh. Two linked duplicates share a single
        // datablock and differ only here, so a buffer owned by the mesh would let
        // the second draw overwrite the first and both would land in one place.
        for (i, draw) in draws.iter().enumerate() {
            let i = i as u32;
            let m = camera.model_matrix_relative(&draw.transform);
            let normal = normal_matrix(&m);
            queue.write_buffer(
                &self.objects.buffer,
                self.objects.offset(i) as u64,
                bytemuck::bytes_of(&ObjectData {
                    model: m.to_cols_array_2d(),
                    normal_col0: normal.x_axis.extend(0.0).to_array(),
                    normal_col1: normal.y_axis.extend(0.0).to_array(),
                    normal_col2: normal.z_axis.extend(0.0).to_array(),
                    base_color: draw.mesh.color.to_array(),
                    paint: [
                        if draw.weights { 1.0 } else { 0.0 },
                        // How the outline marks it: see `outline.wgsl`.
                        if !draw.selected {
                            0.0
                        } else if draw.active {
                            1.0
                        } else {
                            0.5
                        },
                        0.0,
                        0.0,
                    ],
                }),
            );
            // The shadow pass draws the same geometry from a different
            // camera, so it needs the same model matrix and nothing else.
            self.shadows.set_object(queue, i, &m);
            drawn = drawn.union(transformed_bounds(&m, draw.mesh.bounds));
            queue.write_buffer(
                &self.rendered_objects.buffer,
                self.rendered_objects.offset(i) as u64,
                bytemuck::bytes_of(&RenderedObject {
                    model: m.to_cols_array_2d(),
                    normal_col0: normal.x_axis.extend(0.0).to_array(),
                    normal_col1: normal.y_axis.extend(0.0).to_array(),
                    normal_col2: normal.z_axis.extend(0.0).to_array(),
                }),
            );
            queue.write_buffer(
                &self.element_objects.buffer,
                self.element_objects.offset(i) as u64,
                bytemuck::bytes_of(&ElementObject {
                    model: m.to_cols_array_2d(),
                    selected_color: selected_color(),
                    unselected_color: palette::display_rgba(palette::HANDLE),
                    flags: [
                        f32::from(u8::from(draw.preview)),
                        // Fades the vertex dots as they grow too dense to read.
                        // The face fill shares this uniform and ignores it: a
                        // fill does not get less legible as the mesh gets
                        // denser, it stays the same colour.
                        draw.detail.point_alpha,
                        0.0,
                        0.0,
                    ],
                }),
            );
            queue.write_buffer(
                &self.edge_objects.buffer,
                self.edge_objects.offset(i) as u64,
                bytemuck::bytes_of(&EdgeObject {
                    model: m.to_cols_array_2d(),
                    base_color: palette::display_rgba(palette::WIRE),
                    selected_color: selected_color(),
                    flags_params: [
                        // The wireframe's base alpha, scaled by the density
                        // fade. Folded in here rather than in the shader so the
                        // shader keeps one alpha and not two to multiply.
                        0.55 * draw.detail.wire_alpha,
                        1.0,
                        f32::from(u8::from(draw.selected)),
                        f32::from(u8::from(draw.preview)),
                    ],
                }),
            );
        }

        // The camera as well as the geometry: a parallel light splits its map
        // along the view, so where the resolution goes depends on where the
        // user is looking.
        let aspect = self.render_size.0 as f32 / self.render_size.1.max(1) as f32;
        self.fit_tiles(device, queue, camera);
        self.fit_shadows(
            queue,
            drawn,
            crate::shadow::CameraFrustum {
                rotation: camera.rotation.as_quat(),
                fov_y: camera.fov_y as f32,
                aspect,
                near: camera.near() as f32,
            },
        );
    }

    /// Works out which lights reach which square of the screen.
    ///
    /// Before the shadow frames and long before the geometry, because the
    /// shading pass reads its result and a compute pass has to be recorded
    /// ahead of the pass that consumes it.
    ///
    /// Only in the mode that shades with the scene's lights. The others light
    /// themselves, and culling for them would be a compute pass whose output
    /// nothing reads.
    fn fit_tiles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, camera: &Camera) {
        if self.shading != ShadingMode::Rendered {
            return;
        }
        let (w, h) = self.render_size;
        if self.tiles.resize(device, w, h) {
            // The lists moved, so the bind group over them is stale.
            self.shading_bind = shading_bind_group(
                device,
                &self.shading_layout,
                &self.dfg_buf,
                self.lights.buffer(),
                &self.shading_globals_buf,
                &self.shadow_buf,
                self.shadows.array_view(),
                self.tiles.buffer(),
                &self.environment,
                &self.contact_depth,
                &self.behind.1,
            );
            // The globals carry how many tiles across the grid is, and it has
            // just changed.
            queue.write_buffer(
                &self.shading_globals_buf,
                0,
                bytemuck::bytes_of(&self.shading_globals()),
            );
        }
        let aspect = w as f32 / h.max(1) as f32;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tile cull"),
        });
        self.tiles.cull(
            device,
            queue,
            &mut encoder,
            self.lights.buffer(),
            self.lights.count(),
            camera,
            aspect,
        );
        queue.submit([encoder.finish()]);
    }

    /// Fits each casting light's shadow frame to what is being drawn.
    ///
    /// Here rather than beside [`Viewport::set_lights`] because a frustum is
    /// fitted to geometry, and this is the call that has it. That ordering was
    /// the other way round once and the cost was the whole feature: the
    /// application never told the renderer what the scene contained, so every
    /// frustum came back `None` and no shadow was ever drawn — while the
    /// renderer's own tests, which did tell it, passed.
    fn fit_shadows(
        &mut self,
        queue: &wgpu::Queue,
        bounds: kreate_math::Aabb,
        camera: crate::shadow::CameraFrustum,
    ) {
        // Every light starts casting nothing, and a light whose frustum sees
        // nothing keeps that entry — which is what stops a matrix from an
        // earlier frame shadowing this one.
        self.shadow_entries.clear();
        self.shadow_views.clear();
        if self.shading != ShadingMode::Rendered {
            // The other modes light themselves. Fitting frames for lights
            // nothing reads would be a pass per lamp for no picture at all.
            return;
        }
        self.shadow_entries
            .resize(self.packed_lights.len(), crate::shadow::GpuShadow::NONE);
        let mut frames: Vec<(usize, crate::shadow::ShadowView, u32)> = Vec::new();
        for index in 0..self.packed_lights.len() {
            let light = self.packed_lights[index];
            if light.flags & crate::lights::FLAG_SHADOW == 0 {
                continue;
            }
            // A point light wants a frame per cube face it can see anything
            // through, so what comes back is a list. The budget is in layers
            // and not in lights: one lamp in a room is six of them.
            frames.clear();
            for (face, view) in crate::shadow::shadow_views(&light, bounds, Some(camera)) {
                if self.shadow_views.len() >= crate::shadow::MAX_SHADOWS {
                    // Out of layers. The faces already taken still shadow, and
                    // the rest of this light does not — which is the same
                    // partial answer a light past the cap gets, and better than
                    // dropping the faces that were affordable.
                    break;
                }
                let layer = self.shadow_views.len();
                self.shadows.set_view(queue, layer, &view);
                self.shadow_views.push(view);
                frames.push((face, view, layer as u32));
            }
            if !frames.is_empty() {
                self.shadow_entries[index] =
                    crate::shadow::GpuShadow::pack(&frames, crate::shadow::DEPTH_BIAS);
            }
        }
        if !self.shadow_entries.is_empty() {
            queue.write_buffer(
                &self.shadow_buf,
                0,
                bytemuck::cast_slice(&self.shadow_entries),
            );
        }
    }

    /// Points the overlays at what a rubber-band drag would select.
    ///
    /// `bits` is the bitset the compute reduction produced this frame, read
    /// directly rather than copied back: that is what lets the highlight follow
    /// the drag with no readback and no upload, so what the user sees during the
    /// drag is what the commit will apply.
    ///
    /// Must be called every frame the overlays are drawn, active or not. A
    /// pipeline with a bind group in its layout needs one bound, and leaving a
    /// stale one in place would keep highlighting the previous drag.
    /// Hands the scene's lights to the rendered pass.
    ///
    /// Per frame, because every position is rebased onto the camera: a light
    /// two metres from a camera parked a thousand million units out is `2.0`
    /// that way and `0.0` through absolute single precision. The eye moves, so
    /// the packing has to be redone.
    ///
    /// Taken as [`crate::lights::PlacedLight`] rather than read from a
    /// document, because this crate cannot see one. The app does the joining
    /// and the unit conversion, which is where `Light::axial_candela` lives.
    /// The world's radiance: one colour in every direction.
    ///
    /// Set beside the lights because it is one, and the brightest one in most
    /// scenes. Black is a scene lit only by its lamps, which is a real thing
    /// to want and is what a document with no world means.
    pub fn set_world(&mut self, radiance: [f32; 3]) {
        self.world = radiance;
        self.note_lighting(&radiance);
    }

    /// Folds a lighting input into the mean's signature.
    fn note_lighting(&mut self, what: &dyn std::fmt::Debug) {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{what:?}").hash(&mut h);
        self.lighting_signature = self.lighting_signature.rotate_left(7) ^ h.finish();
    }

    /// Widens what the shadows are fitted to, in eye-relative metres.
    ///
    /// Not normally needed and empty by default. The shadow frames are fitted
    /// to the draws [`Viewport::prepare`] is given, because those are the scene
    /// as far as this pass is concerned: something absent from the list is not
    /// drawn, so it has nothing to cast onto and nothing to cast from.
    ///
    /// This is for the case that breaks that rule — a caster deliberately kept
    /// out of the draw list, or a frustum a caller wants larger than the
    /// geometry for its own reasons. It was a required call once, and that was
    /// wrong in the way a required call always is: the application forgot it and
    /// every shadow silently vanished, while the renderer's own tests passed.
    pub fn set_extra_bounds(&mut self, bounds: kreate_math::Aabb) {
        self.extra_bounds = bounds;
    }

    /// How many shadow maps this frame renders.
    ///
    /// Exposed because the cost is invisible in the picture: a light that casts
    /// nothing is skipped by the shader as well, so a renderer that gave every
    /// light a map would draw the scene an extra time per lamp and look
    /// identical. Only a count can see that.
    pub fn shadow_pass_count(&self) -> usize {
        self.shadow_views.len()
    }

    pub fn set_lights(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        eye: glam::DVec3,
        lights: &[crate::lights::PlacedLight],
    ) {
        self.note_lighting(&lights);
        let before = self.lights.capacity();
        // The count is kept by `Lights` and read by the cull pass, which is
        // now the only thing that needs to know how many of the buffer is live.
        self.lights.upload(device, queue, eye, lights);
        // The light buffer grows by being replaced, and this bind group holds
        // the old one. The capacity is what says that happened — it changes
        // only in the branch that replaces the buffer — and rebuilding only
        // then keeps a bind group per frame out of the common case.
        if self.lights.capacity() != before {
            // The shadow buffer has to keep pace: it is parallel to the lights
            // and the shader indexes both with the same number.
            self.shadow_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow entries"),
                size: (std::mem::size_of::<crate::shadow::GpuShadow>()
                    * self.lights.capacity().max(1) as usize) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.shading_bind = shading_bind_group(
                device,
                &self.shading_layout,
                &self.dfg_buf,
                self.lights.buffer(),
                &self.shading_globals_buf,
                &self.shadow_buf,
                self.shadows.array_view(),
                self.tiles.buffer(),
                &self.environment,
                &self.contact_depth,
                &self.behind.1,
            );
            // The cull pass holds the light buffer too, and it has just moved.
            self.tiles.forget_bindings();
        }

        // Kept for `prepare`, which fits the shadow frames: it is the call
        // that sees the geometry, and a frustum is fitted to geometry.
        self.packed_lights.clear();
        self.packed_lights
            .extend(lights.iter().map(|light| crate::lights::GpuLight::pack(light, eye)));
        queue.write_buffer(
            &self.shading_globals_buf,
            0,
            bytemuck::bytes_of(&self.shading_globals()),
        );
    }

    /// What the shading pass needs to know about the frame.
    ///
    /// One place, because it was already written out twice and the environment
    /// would have made three — and a uniform assembled differently in three
    /// places is one that is right in two of them.
    fn shading_globals(&self) -> ShadingGlobals {
        ShadingGlobals {
            tiles_across: self.tiles.across(),
            tile_capacity: crate::tiles::MAX_LIGHTS_PER_TILE,
            region: if self.shading == ShadingMode::Rendered {
                [self.render_size.0, self.render_size.1]
            } else {
                [0, 0]
            },
            world: [
                self.world[0],
                self.world[1],
                self.world[2],
                // The flag, and it is what stops the flat colour and the cube
                // both lighting the scene. A world has one background or the
                // other.
                if self.has_environment { 1.0 } else { 0.0 },
            ],
            environment: [
                self.world_rotation.cos(),
                self.world_rotation.sin(),
                self.environment.roughest(),
                self.world_strength,
            ],
            sh9: crate::environment::sh9_bytes(self.environment.sh9()),
            inv_view_proj: self.inv_view_proj,
            view_proj: self.shading_view_proj,
            flags: [if self.contact_applies() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        }
    }

    /// Says which materials let light through: drawn after the solids, over
    /// a copy of them. Parallel to what `set_materials` was given.
    pub fn set_transmissive(&mut self, transmissive: &[bool]) {
        self.material_transmissive.clear();
        self.material_transmissive.extend_from_slice(transmissive);
    }

    fn transmissive(&self, material: u32) -> bool {
        self.material_transmissive.get(material as usize).copied().unwrap_or(false)
    }

    /// Whether every run of a draw is glass: such a draw casts no shadow and
    /// marks no contact.
    fn all_transmissive(&self, draw: &Draw) -> bool {
        !draw.mesh.submeshes.is_empty()
            && draw.mesh.submeshes.iter().all(|run| self.transmissive(draw.material_for(run.slot)))
    }

    /// Whether this frame's rendered view darkens where things touch: on, the
    /// rendered mode, and rasterised — the path tracer's shadows are exact.
    fn contact_applies(&self) -> bool {
        self.contact_shadows && self.shading == ShadingMode::Rendered && !self.path_tracing()
    }

    /// Replaces the environment with one built from an equirectangular image.
    ///
    /// The pixels are `Rgba16Float`, which is what an environment is: §2.1's
    /// working space has no upper bound and a sky is mostly above one.
    ///
    /// Rebuilding is the *whole* of it — six faces and their mips — so this is
    /// for when the image changes and not for when the rotation does. Turning
    /// a sky is [`Viewport::set_world_rotation`], which costs a uniform write.
    pub fn set_environment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) {
        self.environment =
            crate::environment::Environment::new(device, queue, width, height, pixels);
        self.has_environment = true;
        self.rebuild_shading_bind(device);
        queue.write_buffer(
            &self.shading_globals_buf,
            0,
            bytemuck::bytes_of(&self.shading_globals()),
        );
    }

    /// Goes back to a flat world.
    pub fn clear_environment(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.has_environment {
            return;
        }
        self.environment = crate::environment::Environment::placeholder(device);
        self.has_environment = false;
        self.rebuild_shading_bind(device);
        queue.write_buffer(
            &self.shading_globals_buf,
            0,
            bytemuck::bytes_of(&self.shading_globals()),
        );
    }

    /// Replaces the materials a rendered frame can draw with.
    ///
    /// Each one arrives compiled: the code its graph produced, the constants
    /// that go with it, and the pixels of whatever images it samples. The
    /// application compiles them, because `kreate-render` cannot see a document
    /// and a shading graph is one of the document's things.
    ///
    /// A pipeline is built per distinct *code*, not per material, which is what
    /// keeps a scene of tinted plastics to one shader — see
    /// [`crate::material::PipelineCache`].
    pub fn set_materials(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        materials: &[MaterialSource<'_>],
    ) {
        self.materials.clear();
        self.material_pipeline_of.clear();
        for source in materials {
            let layout = &self.rendered_layout;
            self.material_pipelines
                .get(source.wgsl, |wgsl| rendered_pipeline(device, layout, wgsl));
            self.material_pipeline_of.push(source.wgsl.to_string());
            self.materials.push(crate::material::Material::new(
                device,
                queue,
                &self.material_layout,
                crate::material::MaterialData {
                    constants: source.constants,
                    image_slots: source.image_slots,
                },
                source.images,
            ));
        }
    }

    /// Rewrites one material's numbers, leaving its code and textures alone.
    ///
    /// What a colour slider costs. The whole reason the compiler keeps
    /// constants out of the text: this is a buffer write, and the alternative
    /// is a shader and an atlas per keystroke.
    pub fn set_material_constants(
        &self,
        queue: &wgpu::Queue,
        index: usize,
        constants: &[[f32; 4]],
        image_slots: &[usize],
    ) {
        if let Some(material) = self.materials.get(index) {
            material.set_constants(
                queue,
                crate::material::MaterialData {
                    constants,
                    image_slots,
                },
            );
        }
    }

    /// How many materials the viewport is holding.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// How many distinct shaders those materials needed.
    ///
    /// For the test that says a hundred tinted plastics are one pipeline, which
    /// is the claim the whole arrangement rests on.
    pub fn material_pipeline_count(&self) -> usize {
        self.material_pipelines.len()
    }

    /// Turns the environment about up, and says how much light it gives.
    ///
    /// One call because they arrive together — they are two fields of the same
    /// datablock, and a caller that set one and forgot the other would have a
    /// slider that works and one beside it that does not.
    pub fn set_world_shape(&mut self, queue: &wgpu::Queue, radians: f32, strength: f32) {
        self.note_lighting(&(radians, strength));
        self.world_rotation = radians;
        self.world_strength = strength.max(0.0);
        queue.write_buffer(
            &self.shading_globals_buf,
            0,
            bytemuck::bytes_of(&self.shading_globals()),
        );
    }

    fn rebuild_shading_bind(&mut self, device: &wgpu::Device) {
        self.shading_bind = shading_bind_group(
            device,
            &self.shading_layout,
            &self.dfg_buf,
            self.lights.buffer(),
            &self.shading_globals_buf,
            &self.shadow_buf,
            self.shadows.array_view(),
            self.tiles.buffer(),
            &self.environment,
            &self.contact_depth,
            &self.behind.1,
        );
    }

    /// How the picture is looked at: the view transform and the exposure.
    ///
    /// Writes the uniform too, so the two cannot disagree — a field the
    /// caller could set without the buffer following would show as a slider
    /// that does nothing.
    pub fn set_display(&mut self, queue: &wgpu::Queue, display: crate::display::Display) {
        self.note_lighting(&display);
        self.display = display;
        queue.write_buffer(
            &self.screen_params,
            0,
            &crate::display::ScreenResolve::params_of(display),
        );
    }

    pub fn display(&self) -> crate::display::Display {
        self.display
    }

    pub fn prepare_preview(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bits: &wgpu::Buffer,
        generation: u64,
        preview: Option<PreviewSelection>,
    ) {
        let mode = match &preview {
            Some(p) => [
                1,
                match p.domain {
                    PickDomain::Vert => 0,
                    PickDomain::Edge => 1,
                    _ => 2,
                },
                u32::from(p.deselect),
                p.elements,
            ],
            None => [0; 4],
        };
        queue.write_buffer(
            &self.preview_buf,
            0,
            bytemuck::bytes_of(&PreviewParams { mode }),
        );

        // Rebuilt only when the reduction has reallocated its bitset. Rebuilding
        // every frame would put a bind group creation in the drag path, which is
        // the one path that has to stay quiet.
        if self.preview_generation != Some(generation) {
            self.preview_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("preview bind"),
                layout: &self.preview_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.preview_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bits.as_entire_binding(),
                    },
                ],
            });
            self.preview_generation = Some(generation);
        }
    }

    /// Writes the uniforms the id pass reads.
    ///
    /// Separate from [`Viewport::prepare`] because the id pass is not drawn every
    /// frame — object mode has no element picking — and because the domain being
    /// selected in changes the radius and the depth offset.
    pub fn prepare_ids(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        draws: &[Draw],
        domain: PickDomain,
    ) {
        let view_proj = camera.view_projection(self.aspect());
        queue.write_buffer(
            &self.id_globals_buf,
            0,
            bytemuck::bytes_of(&IdGlobals {
                view_proj: view_proj.to_cols_array_2d(),
                params: [
                    self.render_size.0 as f32,
                    self.render_size.1 as f32,
                    // The pick tolerance, in pixels. Scaled by the display
                    // scaling so that a tolerance chosen as "about a vertex dot
                    // and a half" stays that on a high-density screen instead of
                    // shrinking to a third of it.
                    domain.radius_px() * self.pick_scale,
                    domain.depth_offset(),
                ],
            }),
        );

        let count = draws.len() as u32;
        self.id_objects.ensure(
            device,
            &self.id_object_layout,
            ID_OBJECT_SIZE,
            count,
            "id objects",
        );
        for (i, draw) in draws.iter().enumerate() {
            let m = camera.model_matrix_relative(&draw.transform);
            queue.write_buffer(
                &self.id_objects.buffer,
                self.id_objects.offset(i as u32) as u64,
                bytemuck::bytes_of(&IdObject {
                    model: m.to_cols_array_2d(),
                    // Plus one, so that an untouched pixel reads as a miss rather
                    // than as the first object in the scene.
                    ids: [draw.object_id + 1, 0, 0, 0],
                }),
            );
        }
    }

    /// Records the id pass into `target`.
    ///
    /// Must run after [`Viewport::render`] in the same encoder: it tests against
    /// the depth buffer that pass produced, which is what makes occlusion in the
    /// id buffer exactly the occlusion on screen.
    pub fn render_ids(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        draws: &[Draw],
        domain: PickDomain,
        xray: bool,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ids"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Cleared to zero every frame: zero is "nothing here", and a
                    // stale id left behind would be clickable long after the
                    // element stopped being visible.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    // Loaded, never cleared. The depth this tests against is the
                    // one the visible frame was drawn with.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_viewport(
            0.0,
            0.0,
            self.render_size.0 as f32,
            self.render_size.1 as f32,
            0.0,
            1.0,
        );
        pass.set_pipeline(self.id_pipelines.get(domain, xray));
        pass.set_bind_group(0, &self.id_globals_bind, &[]);

        for (i, draw) in draws.iter().enumerate() {
            let mesh = draw.mesh;
            let count = match domain {
                PickDomain::Object | PickDomain::Face => mesh.counts.triangles * 3,
                PickDomain::Edge => mesh.counts.edges,
                PickDomain::Vert => mesh.counts.verts,
            };
            // Screen-size culling applies here too, and it has to: an object
            // skipped by the visible pass but present in the id buffer would be
            // clickable while invisible, and the user would select something
            // they cannot see.
            //
            // The *density* fade deliberately does not apply. It is a statement
            // about how readable an overlay is, not about what exists, and box
            // selection on a dense mesh is exactly when the id buffer matters
            // most. In practice the question rarely arises: the id pass runs on
            // the object being edited, which is protected from fading anyway.
            if count == 0 || !draw.detail.draw {
                continue;
            }
            pass.set_bind_group(1, &mesh.id_bind, &[]);
            pass.set_bind_group(
                2,
                &self.id_objects.bind,
                &[self.id_objects.offset(i as u32)],
            );
            match domain {
                // Non-indexed: the shader reads the index buffer as storage so
                // that `vertex_index` counts triangle corners and the face id is
                // a lookup. See the shader for why the obvious alternatives do
                // not work.
                PickDomain::Object | PickDomain::Face => pass.draw(0..count, 0..1),
                // Four vertices per element, one instance each: every edge or
                // vertex of a mesh in a single draw.
                PickDomain::Edge | PickDomain::Vert => pass.draw(0..4, 0..count),
            }
        }
    }

    /// Records the frame.
    /// Draws one throwaway frame that exercises every pipeline.
    ///
    /// Creating a pipeline object and *compiling* it are not the same thing.
    /// Measured on this machine: building the viewport takes 586 ms, and the
    /// first frame that uses the pipelines then costs 8.5 ms more than the
    /// steady state. That second number is the driver finishing work it deferred
    /// until a draw actually arrived.
    ///
    /// Left alone, that 8.5 ms does not land at startup. The solid pass and the
    /// grid are used on the first frame, but the overlay pipelines are not
    /// touched until the user presses Tab, and the id pass not until they hover
    /// — so the cost lands in the middle of an interaction, which is exactly the
    /// shape of hitch the plan's latency risk is about. A stall the first time
    /// you click is far worse than the same milliseconds spent before the window
    /// appears.
    ///
    /// So this runs every pipeline once, into a scratch target, against a
    /// scratch cube.
    ///
    /// The scratch target is the viewport's own allocated size rather than
    /// something tiny: a render pass requires its colour and depth attachments
    /// to have the same extent, and the depth buffer belongs to the viewport.
    /// Sizing the colour target down needs the depth resized too, which is more
    /// state to restore than a single extra frame of fill is worth.
    ///
    /// Cheap to call and safe to call twice. Callers should call it once, after
    /// construction, before the first real frame.
    pub fn warm_up(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let warm = self.size;

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport warm-up"),
            size: wgpu::Extent3d {
                width: warm.0,
                height: warm.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let mesh = GpuMesh::upload(
            device,
            &self.layouts,
            // With clusters, or the cluster pipelines are never exercised and
            // the warm-up quietly skips the very draws it exists to compile.
            &RenderMeshCpu::extract(&kreate_mesh::primitives::cube(2.0)).with_clusters(),
            Vec4::ONE,
        );

        // Every overlay on, and both cluster passes, so nothing is left for a
        // later frame to compile. Restored afterwards: warming up must not
        // change what the viewport draws.
        let saved = (
            self.show_grid,
            self.show_wireframe,
            self.show_elements,
            self.cluster_path,
            self.occlusion_culling,
        );
        self.show_grid = true;
        self.show_wireframe = true;
        self.show_elements = true;
        self.cluster_path = true;
        self.occlusion_culling = true;

        let camera = Camera::default();
        let mut draws = vec![Draw::new(&mesh, DAffine3::IDENTITY)];
        self.prepare(device, queue, &camera, &mut draws);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("warm-up"),
        });
        self.render_frame(device, queue, &mut encoder, &view, &draws);
        // The id pass too: it is the one the user provokes by moving the mouse,
        // and it has a pipeline per element domain and per x-ray setting.
        for domain in [
            PickDomain::Object,
            PickDomain::Face,
            PickDomain::Edge,
            PickDomain::Vert,
        ] {
            for xray in [false, true] {
                self.prepare_ids(device, queue, &camera, &draws, domain);
                let ids = self.warm_id_target(device, warm);
                self.render_ids(&mut encoder, &ids, &draws, domain, xray);
            }
        }
        queue.submit([encoder.finish()]);

        (
            self.show_grid,
            self.show_wireframe,
            self.show_elements,
            self.cluster_path,
            self.occlusion_culling,
        ) = saved;
    }

    /// A scratch id target for the warm-up, matching the id pass format.
    fn warm_id_target(&self, device: &wgpu::Device, size: (u32, u32)) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("warm-up ids"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: crate::picking::ID_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// The cluster cull, for tests that need to see what it decided.
    ///
    /// Exposed because the two-pass scheme is otherwise unfalsifiable from
    /// outside: its whole purpose is that the image comes out *unchanged*, so a
    /// pixel comparison cannot distinguish "both passes working" from "occlusion
    /// rejecting nothing at all". The instance counts can.
    pub fn cluster_cull(&self) -> &ClusterCull {
        &self.cull
    }

    /// The depth pyramid, as last built.
    pub fn depth_pyramid(&self) -> &HiZ {
        &self.hiz
    }

    /// Builds the depth pyramid from this viewport's depth buffer.
    ///
    /// Encoded *after* the render pass that wrote the depth and before whatever
    /// will read the pyramid. Separate from [`Self::render`] because it is a
    /// chain of compute passes and a render pass cannot be interrupted, and
    /// separate from [`Self::cull`] because the whole point of a two-pass scheme
    /// is that a cull happens on each side of this.
    pub fn build_depth_pyramid(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Sized to the *render* target rather than the allocation: the depth
        // buffer is allocated at `size` but only `render_size` of it is drawn,
        // and a pyramid covering the undrawn remainder would summarise stale
        // depth from a larger frame.
        self.hiz
            .resize(device, self.render_size.0, self.render_size.1);
        let depth = self.depth.clone();
        self.hiz.build(device, queue, encoder, &depth);
    }

    /// Runs cluster culling, if the cluster path is on.
    ///
    /// A separate entry point because it is a *compute* pass and has to be
    /// encoded before the render pass begins — a render pass cannot be
    /// interrupted to dispatch. wgpu inserts the barrier between them, so the
    /// draw sees the instance counts this wrote.
    ///
    /// Does nothing when the cluster path is off, so a caller can always call it.
    pub fn cull(&self, encoder: &mut wgpu::CommandEncoder, draws: &[Draw]) {
        self.cull_pass(encoder, Pass::First, draws);
    }

    /// Runs one pass of the cluster cull.
    ///
    /// [`Pass::First`] is the whole cull when occlusion testing is off. With it
    /// on, the first pass tests against the pyramid as it stands from the
    /// previous frame and records what it drew, and the second re-tests only
    /// what the first rejected — against a pyramid rebuilt from the depth the
    /// first pass just wrote. See [`Pass`] for why that is sound and a single
    /// stale-pyramid test is not.
    pub fn cull_pass(&self, encoder: &mut wgpu::CommandEncoder, which: Pass, draws: &[Draw]) {
        if !self.cluster_path {
            return;
        }
        let binds: Vec<&wgpu::BindGroup> = draws
            .iter()
            .filter(|d| d.detail.draw)
            .map(|d| &d.mesh.cluster_bind)
            .collect();
        self.cull.dispatch(encoder, which, &binds);
    }

    /// Encodes a whole frame, including occlusion culling if it is on.
    ///
    /// The two-pass sequence, in order, and every step is load bearing:
    ///
    /// 1. cull against the pyramid as it stands, from the previous frame;
    /// 2. draw those survivors, clearing colour and depth;
    /// 3. rebuild the pyramid from the depth just written;
    /// 4. cull again, considering only what step 1 rejected;
    /// 5. draw those survivors and every overlay, loading the same depth;
    /// 6. rebuild the pyramid once more, so the next frame starts from complete
    ///    depth rather than from step 2 alone.
    ///
    /// Step 4 is what makes the scheme sound. A single pass against a pyramid
    /// built before the camera moved can reject a cluster that has since come
    /// into view, and the result is a hole in the model — which a user reads as
    /// corrupt geometry, not as a renderer bug. Step 4 re-tests exactly those
    /// clusters against depth from *this* frame, and because that depth contains
    /// strictly less than the full scene, its verdict can only be more
    /// permissive than the truth.
    ///
    /// Falls back to a single pass when occlusion culling is off, which is the
    /// path every existing caller takes.
    pub fn render_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        draws: &[Draw],
    ) {
        if !self.cluster_path || !self.occlusion_culling {
            self.cull(encoder, draws);
            self.render(encoder, target, draws);
            return;
        }

        let smooth = self.smoothing();
        self.cull_pass(encoder, Pass::First, draws);
        {
            let surfaces = if smooth { self.aa.view() } else { target };
            let mut pass = self.begin_pass(encoder, surfaces, true);
            self.encode_solid(&mut pass, draws, Pass::First);
        }
        self.build_depth_pyramid(device, queue, encoder);

        self.cull_pass(encoder, Pass::Second, draws);
        if smooth {
            {
                let mut pass = self.begin_pass(encoder, self.aa.view(), false);
                self.encode_solid(&mut pass, draws, Pass::Second);
            }
            if self.occluding() {
                self.ao.apply(encoder, self.aa.view(), self.render_size);
            }
            self.aa.resolve(encoder, target, self.render_size);
        } else {
            let mut pass = self.begin_pass(encoder, target, false);
            self.encode_solid(&mut pass, draws, Pass::Second);
        }
        self.draw_outline(encoder, target, draws);
        {
            let mut pass = self.begin_pass(encoder, target, false);
            self.encode_overlays(&mut pass, draws);
        }
        // For the next frame first pass. Skipping it would leave the pyramid
        // missing everything the second pass drew, which is safe — the second
        // pass would simply have more to re-test — but wasteful, and this is
        // eleven small dispatches.
        self.build_depth_pyramid(device, queue, encoder);
    }

    /// Draws the opaque geometry, through whichever path is configured.
    ///
    /// Extracted so the two-pass occlusion scheme can call it twice with a
    /// pyramid build in between. `which` selects the cull results to draw and is
    /// ignored by the direct path, which has none.
    fn encode_solid(&self, pass: &mut wgpu::RenderPass<'_>, draws: &[Draw], which: Pass) {
        // Wireframe draws no surfaces, so nothing hides the edges behind.
        if self.shading == ShadingMode::Wireframe {
            return;
        }
        if self.cluster_path {
            // A mesh uploaded without `RenderMeshCpu::with_clusters` has no
            // clusters to draw, and the loop below skips it on the count — so
            // the symptom is an object that silently does not appear. Said out
            // loud once, because "the cluster path renders nothing" is otherwise
            // a long afternoon.
            if !self
                .cluster_warning
                .swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                let missing = draws
                    .iter()
                    .filter(|d| d.detail.draw && d.mesh.counts.triangles > 0)
                    .filter(|d| d.mesh.cluster_count == 0)
                    .count();
                if missing > 0 {
                    tracing::warn!(
                        missing,
                        "the cluster path is on but these meshes were uploaded \
                         without clusters, so they will not be drawn; use \
                         RenderMeshCpu::with_clusters"
                    );
                }
            }
            pass.set_pipeline(&self.cluster_solid);
            pass.set_bind_group(0, &self.cluster_globals_bind, &[]);
            // Indexed by position among the *surviving* draws, because that is
            // what `prepare` handed the cull. Enumerating all draws here would
            // shift every argument offset by however many were culled.
            for (slot, draw) in self
                .cull
                .slots(which)
                .iter()
                .zip(draws.iter().filter(|d| d.detail.draw))
            {
                if slot.cluster_count == 0 {
                    continue;
                }
                pass.set_bind_group(1, &self.cluster_instance_bind, &[slot.instance_offset]);
                pass.set_bind_group(2, &draw.mesh.cluster_draw_bind, &[]);
                // The whole mesh in one draw whose instance count the GPU chose.
                // No index buffer: the vertex shader reads the cluster index
                // stream from storage, which is what lets one draw cover
                // clusters that are not contiguous in the original mesh.
                pass.draw_indirect(self.cull.args_buffer(), slot.args_offset as u64);
            }
        } else {
            pass.set_pipeline(if self.xray { &self.solid_xray } else { &self.solid });
            pass.set_bind_group(0, &self.globals_bind, &[]);
            for (i, draw) in draws.iter().enumerate() {
                let mesh = draw.mesh;
                if mesh.counts.triangles == 0 || !draw.detail.draw {
                    continue;
                }
                pass.set_bind_group(1, &self.objects.bind, &[self.objects.offset(i as u32)]);
                pass.set_vertex_buffer(0, mesh.positions.slice(..));
                pass.set_vertex_buffer(1, mesh.normals.slice(..));
                pass.set_vertex_buffer(2, mesh.uvs.slice(..));
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.counts.triangles * 3, 0, 0..1);
            }
        }
    }

    /// Begins the viewport render pass.
    ///
    /// `clear` distinguishes the first pass of a frame from a later one: the
    /// two-pass occlusion scheme draws into the same colour and depth twice, and
    /// the second must load what the first wrote rather than clear it.
    fn begin_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        target: &'a wgpu::TextureView,
        clear: bool,
    ) -> wgpu::RenderPass<'a> {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewport"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear {
                        wgpu::LoadOp::Clear(self.background)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    load: if clear {
                        // Reversed-Z clears to 0, not 1: nothing has been drawn.
                        wgpu::LoadOp::Clear(DEPTH_CLEAR as f32)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // Draw only into the region the caller will sample. Without this the
        // image is stretched by however much the allocation was rounded up.
        pass.set_viewport(
            0.0,
            0.0,
            self.render_size.0 as f32,
            self.render_size.1 as f32,
            0.0,
            1.0,
        );
        pass
    }

    /// Draws every overlay: selected faces, the wireframe, vertex points, grid.
    ///
    /// Split out of [`Self::render`] so the two-pass scheme can draw them once,
    /// after both halves of the solid geometry. Drawing them after the first
    /// half would put them under the second half.
    fn encode_overlays(&self, pass: &mut wgpu::RenderPass<'_>, draws: &[Draw]) {
        // Every overlay pipeline has the preview group in its layout, so one is
        // always bound — pointing at a zeroed bitset with the preview switched
        // off when no drag is in progress. Built once in `new` rather than
        // branched on here, because a second pass list for the no-preview case
        // is a copy that drifts.
        let preview = &self.preview_bind;

        // Selected faces first among the overlays, at the layer nearest the
        // surface, so a selected face's own edges and vertices still read on top
        // of its fill.
        if self.show_elements {
            pass.set_pipeline(&self.faces);
            pass.set_bind_group(0, &self.element_globals_bind, &[]);
            for (i, draw) in draws.iter().enumerate() {
                let mesh = draw.mesh;
                if mesh.counts.triangles == 0 || !draw.detail.draw {
                    continue;
                }
                pass.set_bind_group(1, &mesh.id_bind, &[]);
                pass.set_bind_group(
                    2,
                    &self.element_objects.bind,
                    &[self.element_objects.offset(i as u32)],
                );
                pass.set_bind_group(3, preview, &[]);
                // Non-indexed, so the shader can name the face a triangle came
                // from — see the shader.
                pass.draw(0..mesh.counts.triangles * 3, 0..1);
            }
        }

        // Wireframe over the solid pass, before the grid, so geometry occludes
        // the grid but its own edges sit on top of it.
        if self.show_wireframe || self.shading == ShadingMode::Wireframe {
            pass.set_pipeline(&self.edges);
            pass.set_bind_group(0, &self.edge_globals_bind, &[]);
            for (i, draw) in draws.iter().enumerate() {
                let mesh = draw.mesh;
                // A fully faded wireframe is skipped rather than drawn
                // transparent: at that density the draw is millions of
                // instances, and the point of the fade is not to pay for them.
                if mesh.counts.edges == 0 || !draw.detail.draw || draw.detail.wire_alpha <= 0.0 {
                    continue;
                }
                pass.set_bind_group(1, &mesh.edge_bind, &[]);
                pass.set_bind_group(
                    2,
                    &self.edge_objects.bind,
                    &[self.edge_objects.offset(i as u32)],
                );
                pass.set_bind_group(3, preview, &[]);
                // Four vertices per edge, one instance per edge: the whole
                // wireframe of a mesh in a single draw.
                pass.draw(0..4, 0..mesh.counts.edges);
            }
        }

        // Vertex points last of the element overlays and at the nearest layer: a
        // vertex sits on top of both the edges and the faces meeting at it, or
        // the handle disappears exactly where the user needs to grab it.
        if self.show_elements {
            pass.set_pipeline(&self.points);
            pass.set_bind_group(0, &self.element_globals_bind, &[]);
            for (i, draw) in draws.iter().enumerate() {
                let mesh = draw.mesh;
                if mesh.counts.verts == 0 || !draw.detail.draw || draw.detail.point_alpha <= 0.0 {
                    continue;
                }
                pass.set_bind_group(1, &mesh.id_bind, &[]);
                pass.set_bind_group(
                    2,
                    &self.element_objects.bind,
                    &[self.element_objects.offset(i as u32)],
                );
                pass.set_bind_group(3, preview, &[]);
                pass.draw(0..4, 0..mesh.counts.verts);
            }
        }

        if self.show_grid {
            pass.set_pipeline(&self.grid);
            pass.set_bind_group(0, &self.grid_bind, &[]);
            pass.draw(0..3, 0..1);
        }

        // The markers last, over the grid, and against the depth the solid
        // pass wrote: a lamp behind a wall is behind the wall.
        if self.marker_lines.count > 0 || self.marker_points.count > 0 {
            pass.set_bind_group(0, &self.marker_globals_bind, &[]);
            pass.set_bind_group(1, &self.marker_lines.bind, &[]);
            pass.set_bind_group(2, &self.marker_points.bind, &[]);
            if self.marker_lines.count > 0 {
                pass.set_pipeline(&self.marker_line_pipeline);
                pass.draw(0..4, 0..self.marker_lines.count);
            }
            if self.marker_points.count > 0 {
                pass.set_pipeline(&self.marker_point_pipeline);
                pass.draw(0..4, 0..self.marker_points.count);
            }
        }
    }

    /// Hands over this frame's markers: the lines and rings that stand for
    /// objects with no surface. See [`crate::markers`].
    ///
    /// `eye` is the camera's, so positions can be rebased before they are
    /// narrowed to `f32`: the same subtraction every world-space thing in this
    /// crate makes, and the one that keeps a glyph on its object far from the
    /// origin.
    pub fn set_markers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        eye: glam::DVec3,
        lines: &[crate::markers::OverlayLine],
        points: &[crate::markers::OverlayPoint],
    ) {
        let packed_lines: Vec<crate::markers::GpuLine> = lines
            .iter()
            .map(|l| crate::markers::GpuLine::pack(l, eye))
            .collect();
        let packed_points: Vec<crate::markers::GpuPoint> = points
            .iter()
            .map(|p| crate::markers::GpuPoint::pack(p, eye))
            .collect();
        self.marker_lines.upload(
            device,
            queue,
            &self.marker_storage_layout,
            bytemuck::cast_slice(&packed_lines),
            packed_lines.len() as u32,
        );
        self.marker_points.upload(
            device,
            queue,
            &self.marker_storage_layout,
            bytemuck::cast_slice(&packed_points),
            packed_points.len() as u32,
        );
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        draws: &[Draw],
    ) {
        if self.shading == ShadingMode::Rendered {
            self.render_lit(encoder, target, draws);
            return;
        }
        if self.smoothing() && self.accum_applies && self.frame_sample.is_none() {
            // A still view whose mean is full: the mean, and nothing redrawn.
            self.accum.present(encoder, target, self.render_size);
        } else if self.smoothing() {
            // The surfaces into their own image, smoothed onto the target, and
            // the overlays over that — testing the depth the surfaces wrote.
            {
                let mut pass = self.begin_pass(encoder, self.aa.view(), true);
                self.encode_solid(&mut pass, draws, Pass::First);
            }
            if self.occluding() {
                self.ao.apply(encoder, self.aa.view(), self.render_size);
            }
            match self.frame_sample.filter(|_| self.accum_applies) {
                // Into the mean, and the mean onto the target.
                Some(n) => {
                    self.aa.resolve(encoder, self.accum.frame(), self.render_size);
                    self.accum.add(encoder, self.render_size, n);
                    self.accum.present(encoder, target, self.render_size);
                }
                None => self.aa.resolve(encoder, target, self.render_size),
            }
        } else {
            let mut pass = self.begin_pass(encoder, target, true);
            // Opaque geometry first, so the grid can be depth-tested against it.
            self.encode_solid(&mut pass, draws, Pass::First);
        }
        self.draw_outline(encoder, target, draws);
        let mut pass = self.begin_pass(encoder, target, false);
        self.encode_overlays(&mut pass, draws);
    }

    /// The selection outline, over the surfaces and under the overlays.
    fn draw_outline(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, draws: &[Draw]) {
        if !self.show_outline || !draws.iter().any(|d| d.selected) {
            return;
        }
        self.outline.draw(
            encoder,
            target,
            &self.depth,
            self.render_size,
            &self.globals_bind,
            &self.objects.bind,
            draws
                .iter()
                .enumerate()
                .filter(|(_, d)| d.selected && d.detail.draw && d.mesh.counts.triangles > 0)
                .map(|(i, d)| (self.objects.offset(i as u32), &d.mesh.positions, &d.mesh.indices, d.mesh.counts.triangles)),
        );
    }

    /// Starts the mean again: something the signature cannot see changed —
    /// a mesh rewritten in place, a light, a material.
    pub fn restart_accumulation(&mut self) {
        self.accum_samples = 0;
    }

    /// Whether a still view still has frames to add, so the caller should
    /// draw again.
    pub fn accumulating(&self) -> bool {
        self.accum_applies && self.accum_samples < self.sample_limit()
    }

    /// How many frames a still view takes: the raster mean's sixteen, or the
    /// path tracer's hundreds.
    fn sample_limit(&self) -> u32 {
        if self.path_tracing() {
            crate::pathtrace::SAMPLES
        } else {
            crate::accum::SAMPLES
        }
    }

    /// Whether this frame is path traced: asked for, in the rendered view,
    /// with a scene handed over.
    pub fn path_tracing(&self) -> bool {
        self.path_trace && self.shading == ShadingMode::Rendered && self.pathtracer.has_scene()
    }

    /// Hands the path tracer its scene: the meshes, placed, and the flat
    /// surfaces they are made of. Starts its sum again.
    pub fn set_path_trace_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[crate::pathtrace::PtMesh<'_>],
        surfaces: &[crate::pathtrace::PtSurface],
        images: &[crate::pathtrace::PtImage<'_>],
    ) {
        self.pathtracer.set_scene(device, queue, meshes, surfaces, images);
        self.accum_samples = 0;
    }

    /// Samples the path tracer has summed, for the header.
    pub fn path_trace_samples(&self) -> u32 {
        if self.path_tracing() {
            self.accum_samples
        } else {
            0
        }
    }

    /// Decides this frame's place in the mean and returns the projection to
    /// draw it with — nudged by a fraction of a pixel after the first.
    fn next_sample(&mut self, view_proj: glam::Mat4, draws: &[Draw]) -> glam::Mat4 {
        use std::hash::{Hash, Hasher};
        self.accum_applies = self.accumulate
            && (self.smoothing() || self.shading == ShadingMode::Rendered)
            && !(self.cluster_path && self.occlusion_culling);
        if !self.accum_applies {
            self.accum_samples = 0;
            self.frame_sample = None;
            return view_proj;
        }
        // What the picture is of: the view, the size, how it is shaded, and
        // each draw's mesh, place and look.
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for v in view_proj.to_cols_array() {
            v.to_bits().hash(&mut h);
        }
        self.render_size.hash(&mut h);
        (self.shading as u8, self.ambient_occlusion, self.xray, self.face_orientation, self.show_bloom, self.show_reflections, self.path_trace).hash(&mut h);
        format!("{:?}", self.path_trace_lens).hash(&mut h);
        // The lighting inputs since the last frame, folded together, and
        // started afresh for the next: the same calls with the same values
        // give the same number.
        self.lighting_signature.hash(&mut h);
        self.lighting_signature = 0;
        for d in draws {
            (d.mesh as *const GpuMesh as usize).hash(&mut h);
            for v in d.transform.to_cols_array() {
                v.to_bits().hash(&mut h);
            }
            (d.material, d.weights, d.detail.draw).hash(&mut h);
        }
        let signature = h.finish();
        if signature != self.accum_signature {
            self.accum_signature = signature;
            self.accum_samples = 0;
        }
        if self.accum_samples >= self.sample_limit() {
            self.frame_sample = None;
            return view_proj;
        }
        let (jx, jy) = crate::accum::jitter(self.accum_samples);
        self.accum_samples += 1;
        self.frame_sample = Some(self.accum_samples);
        let (w, h) = (self.render_size.0.max(1) as f32, self.render_size.1.max(1) as f32);
        glam::Mat4::from_translation(glam::Vec3::new(2.0 * jx / w, -2.0 * jy / h, 0.0)) * view_proj
    }

    /// Whether the surfaces go through the smoothing pass: on, and there are
    /// surfaces — a wireframe has none, and its lines smooth themselves.
    fn smoothing(&self) -> bool {
        self.antialias && self.shading != ShadingMode::Wireframe
    }

    /// Whether the surfaces are darkened in their corners: the studio view,
    /// which is lit to show shape, and only through the smoothing path — the
    /// occlusion goes into the surfaces' own image, never onto the overlays.
    fn occluding(&self) -> bool {
        self.ambient_occlusion && !self.xray && self.shading == ShadingMode::Studio && self.smoothing()
    }

    /// The rendered mode, in three passes.
    ///
    /// 1. Geometry into the linear intermediate, with depth.
    /// 2. The display transform, onto the target.
    /// 3. Overlays onto the target, testing the depth the first pass wrote.
    ///
    /// The overlays come *after* the resolve, and that is the whole reason
    /// the order is this and not something simpler. They are display-referred
    /// interface — a wireframe, a vertex point, a gizmo — drawn in the palette
    /// the rest of the application uses. Putting them through the display
    /// transform would tone-map the interface, and a selection highlight that
    /// dims when the scene gets bright is not a highlight.
    ///
    /// The depth buffer is shared across the first and third, which is what
    /// lets an overlay hide behind geometry it is behind. The resolve writes
    /// no depth and tests none.
    /// The rendered view's shadow maps and its rasterised scene, into the
    /// linear picture, its depth and the reflection images.
    fn encode_lit_scene(&self, encoder: &mut wgpu::CommandEncoder, draws: &[Draw]) {
        // The shadow maps first: the geometry pass samples them, so they have
        // to be finished before it starts. Separate passes rather than one
        // with several attachments, because each is a different camera.
        for layer in 0..self.shadow_views.len() {
            self.shadows.encode(
                encoder,
                layer,
                draws.iter().enumerate().filter_map(|(i, draw)| {
                    // Detail is deliberately ignored: an object simplified out
                    // of the picture still casts a shadow onto what is left,
                    // and an object that stopped casting when it got small
                    // would have its shadow blink out as the camera pulled
                    // back.
                    if draw.mesh.counts.triangles == 0 || self.all_transmissive(draw) {
                        return None;
                    }
                    Some((
                        i as u32,
                        &draw.mesh.positions,
                        &draw.mesh.indices,
                        draw.mesh.counts.triangles,
                    ))
                }),
            );
        }

        if self.contact_applies() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("contact depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.contact_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(DEPTH_CLEAR as f32),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, self.render_size.0 as f32, self.render_size.1 as f32, 0.0, 1.0);
            pass.set_pipeline(&self.contact_pipeline);
            pass.set_bind_group(0, &self.globals_bind, &[]);
            for (i, draw) in draws.iter().enumerate() {
                let mesh = draw.mesh;
                if mesh.counts.triangles == 0 || !draw.detail.draw || self.all_transmissive(draw) {
                    continue;
                }
                pass.set_bind_group(1, &self.rendered_objects.bind, &[self.rendered_objects.offset(i as u32)]);
                pass.set_vertex_buffer(0, mesh.positions.slice(..));
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.counts.triangles * 3, 0, 0..1);
            }
        }

        let glass;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport rendered"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: self.hdr.view(),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // The world's own radiance, which is what a direction
                            // with no geometry in it delivers. Linear, because
                            // this is a linear image — clearing to the interface's
                            // grey would put display-referred paint into it, and
                            // that is the mistake §2.1 records.
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: self.world[0] as f64,
                                g: self.world[1] as f64,
                                b: self.world[2] as f64,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssr.gbuffer()[0],
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssr.gbuffer()[1],
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssr.gbuffer()[2],
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        // Reversed-Z clears to 0: nothing has been drawn.
                        load: wgpu::LoadOp::Clear(DEPTH_CLEAR as f32),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Into the region the caller will sample, like every other pass.
            // This one did not, and the picture showed it: the geometry was
            // projected across the whole allocation while the wireframe over
            // it was projected across the drawn region, so whenever the two
            // differed — a window whose size is not a multiple of the
            // allocation step, which is most windows — the shaded object sat
            // a few pixels off its own outline.
            pass.set_viewport(
                0.0,
                0.0,
                self.render_size.0 as f32,
                self.render_size.1 as f32,
                0.0,
                1.0,
            );
            // The background first, at the far plane, so the geometry that
            // follows tests against it rather than the other way round.
            pass.set_pipeline(&self.background_pipeline);
            pass.set_bind_group(2, &self.shading_bind, &[]);
            pass.draw(0..3, 0..1);

            pass.set_bind_group(0, &self.globals_bind, &[]);
            pass.set_bind_group(2, &self.shading_bind, &[]);
            // The pipeline is the material's, so it is set inside the loop.
            // Redundant switches are avoided by remembering the last one rather
            // than by sorting: a draw order that follows the material would put
            // the objects in an order the depth pre-pass and the id buffer do
            // not share, and those have to agree about what is in front.
            glass = self.draw_rendered_runs(&mut pass, draws, false);
        }

        // Then what lets light through, over a copy of the solids: glass
        // shows what was there before it, so it is drawn after everything it
        // can show and reads the picture as it stood.
        if glass {
            encoder.copy_texture_to_texture(
                self.hdr.texture().as_image_copy(),
                self.behind.0.as_image_copy(),
                wgpu::Extent3d { width: self.render_size.0, height: self.render_size.1, depth_or_array_layers: 1 },
            );
            let load = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport rendered glass"),
                color_attachments: &[
                    load(self.hdr.view()),
                    load(&self.ssr.gbuffer()[0]),
                    load(&self.ssr.gbuffer()[1]),
                    load(&self.ssr.gbuffer()[2]),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, self.render_size.0 as f32, self.render_size.1 as f32, 0.0, 1.0);
            pass.set_bind_group(0, &self.globals_bind, &[]);
            pass.set_bind_group(2, &self.shading_bind, &[]);
            self.draw_rendered_runs(&mut pass, draws, true);
        }
    }

    /// The rendered pass's draws: the solid runs, or the glass ones. Returns
    /// whether a run of the other kind was passed over.
    fn draw_rendered_runs(&self, pass: &mut wgpu::RenderPass<'_>, draws: &[Draw<'_>], glass: bool) -> bool {
        let mut skipped = false;
            let mut current: Option<&str> = None;
            for (i, draw) in draws.iter().enumerate() {
                let mesh = draw.mesh;
                if mesh.counts.triangles == 0 || !draw.detail.draw {
                    continue;
                }
                pass.set_bind_group(
                    1,
                    &self.rendered_objects.bind,
                    &[self.rendered_objects.offset(i as u32)],
                );
                pass.set_vertex_buffer(0, mesh.positions.slice(..));
                pass.set_vertex_buffer(1, mesh.normals.slice(..));
                pass.set_vertex_buffer(2, mesh.uvs.slice(..));
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);

                // One call per run of triangles sharing a material slot. Nearly
                // every mesh has exactly one run, so this is the same single
                // call it always was — the loop is what lets a mesh whose faces
                // carry different materials draw in more than one of them,
                // which until now it could not: `material_index` reached the
                // glTF exporter and never reached the screen.
                for run in &mesh.submeshes {
                    let material = draw.material_for(run.slot);
                    if self.transmissive(material) != glass {
                        skipped = true;
                        continue;
                    }
                    // A run naming a material that was never set draws nothing.
                    // Skipped rather than substituted: a stand-in would be a
                    // grey face where the user put a red one, and a missing
                    // face is the complaint that gets reported.
                    let Some(source) = self.material_pipeline_of.get(material as usize) else {
                        continue;
                    };
                    if current != Some(source.as_str()) {
                        let Some(pipeline) = self.material_pipelines.peek(source) else {
                            continue;
                        };
                        pass.set_pipeline(pipeline);
                        current = Some(source.as_str());
                    }
                    pass.set_bind_group(3, self.materials[material as usize].bind_group(), &[]);
                    pass.draw_indexed(run.start..run.start + run.count, 0, 0..1);
                }
            }
        skipped
    }

    fn render_lit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        draws: &[Draw],
    ) {
        if self.path_tracing() {
            // The path tracer's mean onto the picture — over the rasterised
            // scene, which is still drawn for the depth the overlays test
            // against. A full mean is left on the picture and shown again.
            if self.frame_sample.is_some() || !self.accum_applies {
                self.encode_lit_scene(encoder, draws);
                self.pathtracer.render(encoder, &self.shading_bind, self.hdr.view());
                if self.path_trace_denoise {
                    self.denoise.apply(encoder, self.hdr.texture(), self.hdr.view(), self.render_size);
                }
                if self.show_bloom {
                    self.bloom.apply(encoder, self.hdr.view());
                }
            }
            self.screen.draw_with(encoder, &self.screen_bind, target);
        } else if self.accum_applies && self.frame_sample.is_none() {
            // A still view whose mean is full: the mean, and nothing redrawn.
            self.accum.present(encoder, target, self.render_size);
        } else {
            self.encode_lit_scene(encoder, draws);
            // Reflections first, from the picture as lit; then corners and
            // contact darkened, then the glow on top: occlusion shades light,
            // it does not shade a lens's flare.
            if self.show_reflections {
                self.ssr.apply(encoder, self.hdr.view(), self.render_size);
            }
            if self.ambient_occlusion && !self.xray {
                self.ao.apply_linear(encoder, self.hdr.view(), self.render_size);
            }
            if self.show_bloom {
                self.bloom.apply(encoder, self.hdr.view());
            }
            match self.frame_sample.filter(|_| self.accum_applies) {
                // Into the mean, and the mean onto the target: the display
                // transform's picture, averaged — which is what smooths the
                // rendered view's edges.
                Some(n) => {
                    self.screen.draw_with(encoder, &self.screen_bind, self.accum.frame());
                    self.accum.add(encoder, self.render_size, n);
                    self.accum.present(encoder, target, self.render_size);
                }
                None => self.screen.draw_with(encoder, &self.screen_bind, target),
            }
        }
        self.draw_outline(encoder, target, draws);

        {
            let mut pass = self.begin_pass(encoder, target, false);
            self.encode_overlays(&mut pass, draws);
        }
    }
}

/// Clamps a size to fit inside another.
trait ClampTo {
    fn clamp_to(self, bounds: Self) -> Self;
}

impl ClampTo for (u32, u32) {
    fn clamp_to(self, bounds: Self) -> Self {
        (self.0.min(bounds.0).max(1), self.1.min(bounds.1).max(1))
    }
}

/// The shading bind group: group 2, in the order the libraries declare.
///
/// One function rather than two identical blocks, because the group is built
/// once at startup and again whenever the light buffer grows — and two copies
/// of a binding list is how a rebuild ends up pointing at a stale buffer.
#[allow(clippy::too_many_arguments)]
fn shading_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    dfg: &wgpu::Buffer,
    lights: &wgpu::Buffer,
    globals: &wgpu::Buffer,
    shadows: &wgpu::Buffer,
    maps: &wgpu::TextureView,
    tiles: &wgpu::Buffer,
    environment: &crate::environment::Environment,
    contact_depth: &wgpu::TextureView,
    behind: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shading data"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dfg.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: globals.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shadows.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(maps),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: tiles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(environment.view()),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(environment.sampler()),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(contact_depth),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(behind),
            },
        ],
    })
}

/// A compiled material, as the application hands it over.
///
/// Text, numbers and pixels — nothing from the document. See
/// [`Viewport::set_materials`].
pub struct MaterialSource<'a> {
    /// The `material_sample` function `kreate-shader` produced.
    pub wgsl: &'a str,
    pub constants: &'a [[f32; 4]],
    pub image_slots: &'a [usize],
    pub images: &'a [crate::material::SourceImage<'a>],
}

/// The rendered pass, with one material's code on the end of it.
fn rendered_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    material: &str,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rendered"),
        source: wgpu::ShaderSource::Wgsl(
            crate::shading::rendered_source_with(material).into(),
        ),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("rendered"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_rendered"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32,
                        offset: 0,
                        shader_location: 1,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 2,
                    }],
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_rendered"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    // The linear intermediate, not the window: this pass writes
                    // light and the resolve writes a picture.
                    format: crate::display::HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // What the reflections need, beside it: see `Ssr`.
                Some(wgpu::ColorTargetState { format: crate::ssr::FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                Some(wgpu::ColorTargetState { format: crate::ssr::FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                Some(wgpu::ColorTargetState { format: crate::ssr::FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
            ],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // Not culled, for the reason the solid pass is not: a modeller has
            // open and inconsistently wound geometry mid-edit, and the fragment
            // shader flips the normal instead.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(gpu::depth_write_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// A box around a transformed box.
///
/// The centre through the matrix and the extent through its absolute value,
/// which is exact for an axis-aligned box and is one transform rather than
/// eight. The result is the tightest axis-aligned box containing the
/// transformed one, which is all a shadow frustum needs.
fn transformed_bounds(m: &glam::Mat4, b: kreate_math::Aabb) -> kreate_math::Aabb {
    if b.is_empty() {
        return kreate_math::Aabb::EMPTY;
    }
    let centre = m.transform_point3(b.center());
    let half = b.size() * 0.5;
    let extent = glam::Vec3::new(
        m.x_axis.x.abs() * half.x + m.y_axis.x.abs() * half.y + m.z_axis.x.abs() * half.z,
        m.x_axis.y.abs() * half.x + m.y_axis.y.abs() * half.y + m.z_axis.y.abs() * half.z,
        m.x_axis.z.abs() * half.x + m.y_axis.z.abs() * half.y + m.z_axis.z.abs() * half.z,
    );
    kreate_math::Aabb::new(centre - extent, centre + extent)
}

#[cfg(test)]
mod bounds_tests {
    use super::*;

    #[test]
    fn a_rotated_box_needs_a_bigger_box() {
        // The whole reason the fitting takes a matrix. A scene of translations
        // never asks this — and a sweep that dropped the rotation entirely
        // walked through every device test, because the floor is larger than
        // anything standing on it and sets the bounds by itself.
        let unit = kreate_math::Aabb::new(glam::Vec3::splat(-0.5), glam::Vec3::splat(0.5));
        let turned = glam::Mat4::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let fitted = transformed_bounds(&turned, unit);
        // A unit square turned forty-five degrees is as wide as its diagonal.
        let root_two = std::f32::consts::SQRT_2;
        assert!(
            (fitted.size().x - root_two).abs() < 1e-5,
            "a turned box came out {} wide",
            fitted.size().x
        );
        // And the axis it was turned about is unchanged.
        assert!((fitted.size().z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_moved_box_moves_and_a_scaled_one_grows() {
        let unit = kreate_math::Aabb::new(glam::Vec3::splat(-0.5), glam::Vec3::splat(0.5));
        let moved = transformed_bounds(&glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.0)), unit);
        assert!((moved.center().x - 3.0).abs() < 1e-6);
        assert!((moved.size().x - 1.0).abs() < 1e-6);

        let scaled = transformed_bounds(&glam::Mat4::from_scale(glam::Vec3::new(4.0, 1.0, 1.0)), unit);
        assert!((scaled.size().x - 4.0).abs() < 1e-6);

        // Nothing is not somewhere: an empty box through any matrix stays
        // empty, or a scene with one invisible object gets a frustum fitted to
        // the origin.
        assert!(transformed_bounds(&glam::Mat4::IDENTITY, kreate_math::Aabb::EMPTY).is_empty());
    }
}

/// Inverse transpose of a matrix's linear part.
///
/// Only equal to the rotation when the scale is uniform. Using the model matrix
/// directly tilts every normal under non-uniform scale, which reads as broken
/// shading rather than as a transform bug.
fn normal_matrix(model: &Mat4) -> Mat3 {
    Mat3::from_mat4(*model).inverse().transpose()
}

/// Sizes of the per-draw uniform blocks, used both to allocate the rings and to
/// declare `min_binding_size` on their layouts.
const OBJECT_SIZE: u64 = std::mem::size_of::<ObjectData>() as u64;
const RENDERED_OBJECT_SIZE: u64 = std::mem::size_of::<RenderedObject>() as u64;
const EDGE_OBJECT_SIZE: u64 = std::mem::size_of::<EdgeObject>() as u64;

/// Relative depth nudge for the wireframe, as a fraction of the fragment's own
/// depth. Layer 2 in the overlay stack: above filled faces, below vertex points.
const EDGE_DEPTH_OFFSET: f32 = 3.0e-4;

/// Depth format, re-exported so callers configuring an offscreen viewport target
/// do not have to reach into `gpu`.
pub const VIEWPORT_DEPTH_FORMAT: wgpu::TextureFormat = DEPTH_FORMAT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_structs_meet_std140_alignment() {
        // A uniform block must be a multiple of 16 bytes, and a mat4x4 must start
        // on a 16-byte boundary. Getting this wrong does not fail loudly; it
        // garbles a matrix, which then looks like a camera bug.
        assert_eq!(std::mem::size_of::<Globals>() % 16, 0);
        assert_eq!(std::mem::size_of::<ObjectData>() % 16, 0);
        assert_eq!(std::mem::size_of::<GridGlobals>() % 16, 0);
        assert_eq!(std::mem::size_of::<Globals>(), 128);
        assert_eq!(std::mem::size_of::<ObjectData>(), 144);
        assert_eq!(std::mem::size_of::<GridGlobals>(), 192);
        assert_eq!(std::mem::size_of::<EdgeGlobals>() % 16, 0);
        assert_eq!(std::mem::size_of::<EdgeGlobals>(), 128);
    }

    #[test]
    fn normal_matrix_matches_the_rotation_when_scale_is_uniform() {
        let m = Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(2.0),
            glam::Quat::from_rotation_z(0.7),
            glam::Vec3::new(3.0, 0.0, 0.0),
        );
        let n = normal_matrix(&m);
        let rotated = (n * glam::Vec3::X).normalize();
        let expected = (Mat3::from_mat4(m) * glam::Vec3::X).normalize();
        assert!((rotated - expected).length() < 1e-5);
    }

    #[test]
    fn normal_matrix_compensates_for_non_uniform_scale() {
        let m = Mat4::from_scale(glam::Vec3::new(1.0, 1.0, 0.1));
        let n = (normal_matrix(&m) * glam::Vec3::Z).normalize();
        assert!((n - glam::Vec3::Z).length() < 1e-5);
        // The naive path, which would tilt the normal on any sloped surface.
        let naive = (Mat3::from_mat4(m) * glam::Vec3::new(1.0, 0.0, 1.0)).normalize();
        let correct = (normal_matrix(&m) * glam::Vec3::new(1.0, 0.0, 1.0)).normalize();
        assert!(
            (naive - correct).length() > 0.5,
            "the two must differ under non-uniform scale, or the test proves nothing"
        );
    }

    #[test]
    fn the_drawn_region_drives_the_aspect_ratio_not_the_allocation() {
        // The defect: an offscreen target allocated in coarse steps is larger
        // than the area on screen, and projecting for the allocation while
        // sampling it into a smaller rectangle stretches the image.
        //
        // Exercised through the size arithmetic rather than a device, because
        // that is where the mistake is.
        let alloc = (960u32, 384u32);
        let exact = (960u32, 356u32);
        let aspect_from_alloc = alloc.0 as f64 / alloc.1 as f64;
        let aspect_from_exact = exact.0 as f64 / exact.1 as f64;
        assert!(
            (aspect_from_alloc - aspect_from_exact).abs() > 0.15,
            "the two aspects must differ, or the test proves nothing"
        );

        let uv = (
            exact.0 as f32 / alloc.0 as f32,
            exact.1 as f32 / alloc.1 as f32,
        );
        assert!((uv.0 - 1.0).abs() < 1e-6, "width was not rounded");
        assert!(uv.1 < 0.94, "height was rounded up, so the UV must be < 1");
    }

    #[test]
    fn shaders_parse() {
        // naga is already in the dependency tree via wgpu; parsing the shaders
        // here turns a syntax error into a unit-test failure instead of a blank
        // window and a validation message at run time.
        for (name, src) in [
            ("solid", include_str!("shaders/solid.wgsl")),
            ("grid", include_str!("shaders/grid.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name}.wgsl failed to parse: {e:?}"));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}.wgsl failed validation: {e:?}"));
        }
    }
}

/// The copy of the solids' picture glass reads.
fn behind_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("behind glass"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::display::HDR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
