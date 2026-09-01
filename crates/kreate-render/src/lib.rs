//! Viewport rendering: camera, GPU mesh buffers, and the pass list.
//!
//! This crate never mutates a document. It consumes an immutable snapshot,
//! which is what lets the render path stay lock-free.

pub mod camera;
pub mod gpu;
pub mod mesh_gpu;
pub mod viewport;

pub use camera::{Camera, OrbitStyle, Projection};
pub use gpu::{
    AdapterSummary, Gpu, GpuError, GpuOptions, DEPTH_CLEAR, DEPTH_COMPARE, DEPTH_FORMAT,
};
pub use mesh_gpu::{MeshCounts, RenderMeshCpu};
pub use viewport::{GpuMesh, ShadingMode, Viewport};
