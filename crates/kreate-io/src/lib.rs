//! Import and export.
//!
//! Every format goes through the neutral [`ir::ImportScene`] rather than
//! touching a document directly, so the hard interoperability semantics live in
//! exactly one place.

#![forbid(unsafe_code)]

pub mod build;
pub mod clipboard;
pub mod drawing;
pub mod exr;
pub mod container;
pub mod gltf_export;
pub mod gltf_import;
pub mod image_import;
pub mod ir;
pub mod kmesh;
pub mod ktile;
pub mod library;
pub mod migrate;
pub mod obj;
pub mod step;
pub mod stl;
pub mod threemf;
pub mod usd;

pub use build::{build_into, scene_from_document, BuildNote};
pub use container::{ContainerError, Manifest, UnknownEntry};
pub use gltf_export::{write_gltf, write_gltf_scene, GltfExportOptions};
pub use gltf_import::{parse_gltf, GltfError, GltfOptions};
pub use image_import::{load_image, load_image_file, ImageError};
pub use ir::{
    ExportNode, ExportScene, ImportMaterial, ImportMesh, ImportScene, LossyNote, SourceMetadata,
    UvOrigin,
};
pub use kmesh::{KmeshError, UnknownBuffer};
pub use ktile::KtileError;
pub use obj::{parse_obj, write_obj, ObjError, ObjOptions};
pub use stl::{parse_stl, write_stl, StlError, StlOptions};
