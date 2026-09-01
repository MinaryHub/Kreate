//! The document model: objects, datablocks, hierarchy, selection, and undo.
//!
//! Typed arenas rather than an ECS, and copy-on-write throughout — see
//! [`document`] for why both of those are load-bearing rather than stylistic.

pub mod document;
pub mod undo;

pub use document::{
    Collection, CollectionId, Document, Linkage, Material, MaterialId, MeshId, Object, ObjectData,
    ObjectId, ReparentError, Selection,
};
pub use undo::{DocMut, EditError, UndoPayload};
