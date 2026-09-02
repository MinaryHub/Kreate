//! Viewport rendering: camera, GPU mesh buffers, and the pass list.
//!
//! This crate never mutates a document. It consumes an immutable snapshot,
//! which is what lets the render path stay lock-free.

pub mod camera;
pub mod gpu;
pub mod mesh_gpu;
pub mod motion;
pub mod picking;
pub mod region_select;
pub mod viewport;

pub use camera::{Camera, OrbitStyle, Projection, ViewAxis};
pub use gpu::{
    AdapterSummary, Gpu, GpuError, GpuOptions, DEPTH_CLEAR, DEPTH_COMPARE, DEPTH_FORMAT,
};
pub use mesh_gpu::{MeshCounts, RenderMeshCpu};
pub use motion::{CameraMotion, DEFAULT_SMOOTHING, DEFAULT_TRANSITION};
pub use picking::{PickDomain, PickHit, PickRegion, PickTarget};
pub use region_select::{Collect, RegionSelect, SelectShape, MAX_LASSO};
pub use viewport::{Draw, GpuMesh, MeshLayouts, PreviewSelection, ShadingMode, Viewport};
