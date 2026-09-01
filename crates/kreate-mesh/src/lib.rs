//! Kreate mesh kernel: attribute layers, `MeshData` storage, and (later) the
//! `BMesh` edit structure plus its validator.
//!
//! This crate depends only on `kreate-math`. No wgpu, no egui, no async, no
//! scene — it must stand alone as a library and compile to wasm.

#![forbid(unsafe_code)]

pub mod arena;
pub mod attribute;
pub mod mesh_data;
pub mod primitives;
pub mod tessellate;

pub use arena::{Arena, ArenaIndex, EdgeId, FaceId, LoopId, VertId};
pub use attribute::{AttrData, AttrType, AttributeStore, Domain, Layer};
pub use mesh_data::{MeshData, Violation};
pub use tessellate::{triangulate_polygon, Triangulation};
