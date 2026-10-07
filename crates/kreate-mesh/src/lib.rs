//! Kreate mesh kernel: attribute layers, `MeshData` storage, and the `BMesh`
//! edit structure with its validator.
//!
//! This crate depends only on `kreate-math`. No wgpu, no egui, no async, no
//! scene — it must stand alone as a library and compile to wasm.

#![forbid(unsafe_code)]

pub mod arena;
pub mod attribute;
pub mod bmesh;
pub mod check;
pub mod edit;
pub mod intersect;
pub mod mesh_data;
pub mod measure;
pub mod modifiers;
pub mod primitives;
pub mod sculpt;
pub mod selection;
pub mod tessellate;

pub use arena::{Arena, ArenaIndex, EdgeId, FaceId, LoopId, VertId};
pub use attribute::{AttrData, AttrType, AttributeStore, Domain, Layer};
pub use bmesh::{BMesh, BmError, BmViolation, Edge, ElementId, Face, Loop, SelectMode, Vert};
pub use edit::{
    delete, dissolve_edges, dissolve_faces, dissolve_verts, extrude_faces, extrude_selected_faces,
    subdivide_faces, subdivide_selected, DeleteMode, Dissolution, Extrusion, Subdivision,
};
pub use mesh_data::{MeshData, Violation};
pub use tessellate::{triangulate_polygon, CornerTriangulation, Triangulation};
