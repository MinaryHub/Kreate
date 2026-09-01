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
use crate::gpu::{self, DEPTH_CLEAR, DEPTH_FORMAT};
use crate::mesh_gpu::{MeshCounts, RenderMeshCpu};

/// How the solid pass shades.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadingMode {
    Studio,
    Flat,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ObjectData {
    model: [[f32; 4]; 4],
    normal_col0: [f32; 4],
    normal_col1: [f32; 4],
    normal_col2: [f32; 4],
    base_color: [f32; 4],
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
}

/// A mesh resident on the GPU.
///
/// The buffers are separate rather than interleaved so that the hot ones can be
/// rewritten without touching the cold ones — see `mesh_gpu`.
pub struct GpuMesh {
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    indices: wgpu::Buffer,
    object: wgpu::Buffer,
    object_bind: wgpu::BindGroup,
    pub counts: MeshCounts,
    pub color: Vec4,
}

impl GpuMesh {
    pub fn upload(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        cpu: &RenderMeshCpu,
        color: Vec4,
    ) -> Self {
        // COPY_DST on the volatile buffers: they are rewritten during a drag.
        // STORAGE on positions: the edge overlay reads both endpoints of an edge
        // from it to expand a screen-space line quad.
        let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh positions"),
            contents: bytemuck::cast_slice(&cpu.positions),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });
        let normals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh normals"),
            contents: bytemuck::cast_slice(&cpu.normals),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&cpu.indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        let object = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object uniform"),
            size: std::mem::size_of::<ObjectData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let object_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object bind group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: object.as_entire_binding(),
            }],
        });

        Self {
            positions,
            normals,
            indices,
            object,
            object_bind,
            counts: cpu.counts,
            color,
        }
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
}

/// A bind group layout holding one uniform buffer, visible to both stages.
fn uniform_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Everything needed to draw one viewport.
pub struct Viewport {
    solid: wgpu::RenderPipeline,
    grid: wgpu::RenderPipeline,
    globals_buf: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,
    grid_buf: wgpu::Buffer,
    grid_bind: wgpu::BindGroup,
    pub object_layout: wgpu::BindGroupLayout,
    depth: wgpu::TextureView,
    size: (u32, u32),
    pub shading: ShadingMode,
    pub background: wgpu::Color,
    /// Whether to draw the ground grid. A user-facing toggle, and what lets a
    /// test compare geometry without the world-anchored lattice in the way.
    pub show_grid: bool,
}

impl Viewport {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let globals_layout = uniform_bind_group_layout(device, "globals");
        let object_layout = uniform_bind_group_layout(device, "object");
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
        let globals_bind = bind(&globals_layout, &globals_buf, "globals bind");
        let grid_bind = bind(&grid_layout, &grid_buf, "grid bind");

        let solid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("solid"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/solid.wgsl").into()),
        });
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grid.wgsl").into()),
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

        let solid = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("solid"),
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
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &solid_shader,
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
                // Backface culling stays off in the solid pass. A modeller
                // routinely has open and inconsistently wound geometry mid-edit,
                // and hiding it is worse than showing it: the fragment shader
                // flips the normal instead, so a backface is lit rather than
                // black or missing.
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

        Self {
            solid,
            grid,
            globals_buf,
            globals_bind,
            grid_buf,
            grid_bind,
            object_layout,
            depth: gpu::create_depth_texture(device, width, height),
            size: (width.max(1), height.max(1)),
            shading: ShadingMode::Studio,
            background: wgpu::Color {
                r: 0.055,
                g: 0.058,
                b: 0.065,
                a: 1.0,
            },
            show_grid: true,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size != self.size {
            self.size = size;
            self.depth = gpu::create_depth_texture(device, size.0, size.1);
        }
    }

    pub fn aspect(&self) -> f64 {
        self.size.0 as f64 / self.size.1 as f64
    }

    /// Uploads per-frame uniforms.
    pub fn prepare(&self, queue: &wgpu::Queue, camera: &Camera, meshes: &[(&GpuMesh, DAffine3)]) {
        let aspect = self.aspect();
        let view_proj = camera.view_projection(aspect);

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
                    0.0,
                    0.0,
                    0.0,
                ],
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
            }),
        );

        for (mesh, model) in meshes {
            let m = camera.model_matrix_relative(model);
            let normal = normal_matrix(&m);
            queue.write_buffer(
                &mesh.object,
                0,
                bytemuck::bytes_of(&ObjectData {
                    model: m.to_cols_array_2d(),
                    normal_col0: normal.x_axis.extend(0.0).to_array(),
                    normal_col1: normal.y_axis.extend(0.0).to_array(),
                    normal_col2: normal.z_axis.extend(0.0).to_array(),
                    base_color: mesh.color.to_array(),
                }),
            );
        }
    }

    /// Records the frame.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        meshes: &[(&GpuMesh, DAffine3)],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewport"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.background),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    // Reversed-Z clears to 0, not 1: nothing has been drawn yet.
                    load: wgpu::LoadOp::Clear(DEPTH_CLEAR as f32),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Opaque geometry first, so the grid can be depth-tested against it.
        pass.set_pipeline(&self.solid);
        pass.set_bind_group(0, &self.globals_bind, &[]);
        for (mesh, _) in meshes {
            if mesh.counts.triangles == 0 {
                continue;
            }
            pass.set_bind_group(1, &mesh.object_bind, &[]);
            pass.set_vertex_buffer(0, mesh.positions.slice(..));
            pass.set_vertex_buffer(1, mesh.normals.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.counts.triangles * 3, 0, 0..1);
        }

        if self.show_grid {
            pass.set_pipeline(&self.grid);
            pass.set_bind_group(0, &self.grid_bind, &[]);
            pass.draw(0..3, 0..1);
        }
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
        assert_eq!(std::mem::size_of::<Globals>(), 80);
        assert_eq!(std::mem::size_of::<ObjectData>(), 128);
        assert_eq!(std::mem::size_of::<GridGlobals>(), 144);
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
