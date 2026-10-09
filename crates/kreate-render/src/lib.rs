//! Viewport rendering: camera, GPU mesh buffers, and the pass list.
//!
//! This crate never mutates a document. It consumes an immutable snapshot,
//! which is what lets the render path stay lock-free.

pub mod accum;
pub mod ao;
pub mod bloom;
pub mod brdf;
pub mod camera;
pub mod capability;
pub mod cluster;
pub mod cluster_cull;
pub mod deform;
pub mod reshape;
pub mod skin;
pub mod solid_textures;
pub mod denoise;
pub mod display;
pub mod environment;
pub mod fxaa;
pub mod gpu;
pub mod hiz;
pub mod lights;
pub mod markers;
pub use markers::{MarkerShape, OverlayLine, OverlayPoint};
pub mod lod;
pub mod material;
pub mod mesh_gpu;
pub mod motion;
pub mod outline;
pub mod palette;
pub mod pathtrace;
pub mod picking;
pub mod region_select;
pub mod shading;
pub mod shadow;
pub mod ssr;
pub mod tiles;
pub mod viewport;

pub use camera::{Camera, OrbitStyle, Projection, ViewAxis};
pub use capability::{Capabilities, Tier};
pub use cluster::{is_occluded, survives, Cluster, ClusterMesh, Frustum, PyramidData, WorldSphere};
pub use cluster_cull::{ClusterCull, CullSlot, VERTICES_PER_CLUSTER};
pub use deform::{Deform, Deformer, Gesture, Placement};
pub use reshape::Reshape;
pub use skin::{bone_rows, BoneRows, Influences, Skin};
pub use gpu::{
    AdapterSummary, Gpu, GpuError, GpuOptions, DEPTH_CLEAR, DEPTH_COMPARE, DEPTH_FORMAT,
};
pub use hiz::HiZ;
pub use lod::{sphere_diameter_px, Budget, Detail, ViewSize};
pub use mesh_gpu::{ElementFlags, MeshCounts, RenderMeshCpu};
pub use motion::{CameraMotion, DEFAULT_SMOOTHING, DEFAULT_TRANSITION};
pub use picking::{PickDomain, PickHit, PickRegion, PickTarget};
pub use region_select::{Collect, RegionSelect, SelectShape, MAX_LASSO};
pub use viewport::{Draw, GpuMesh, MeshLayouts, PreviewSelection, ShadingMode, Viewport};
