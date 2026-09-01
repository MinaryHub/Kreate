//! Import and export.
//!
//! Every format goes through the neutral [`ir::ImportScene`] rather than
//! touching a document directly, so the hard interoperability semantics live in
//! exactly one place.

#![forbid(unsafe_code)]

pub mod ir;
pub mod obj;

pub use ir::{ImportMaterial, ImportMesh, ImportScene, LossyNote, SourceMetadata, UvOrigin};
pub use obj::{parse_obj, write_obj, ObjError, ObjOptions};
