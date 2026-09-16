//! The document model: objects, datablocks, hierarchy, selection, and undo.
//!
//! Typed arenas rather than an ECS, and copy-on-write throughout — see
//! [`document`] for why both of those are load-bearing rather than stylistic.

pub mod camera;
pub mod document;
pub mod edit_session;
pub mod image;
pub mod shader;
pub mod light;
pub mod modifier;
pub mod undo;

pub use camera::{Camera, Projection, SensorFit};
pub use document::{
    Collection, CollectionId, Document, Linkage, LightId, Material, MaterialId, MaterialOverrides,
    MeshId, Object, ObjectData, ObjectId, ReparentError, Selection, WorldId,
};
pub use edit_session::{EditSession, EditSessions};
pub use image::{ColorSpace, Image, PixelFormat, Source, Tile, TILE};
pub use light::{AreaShape, Background, Intensity, Light, LightKind, World};
/// A datablock's persistent identity and what kind of thing it names.
///
/// Re-exported rather than re-declared: the type lives in `kreate-reflect`
/// because a property path has to be holdable by the interface, which cannot
/// see a document. Anything that has a document in hand gets it from here.
pub use kreate_reflect::{DatablockKind, Uid};
pub use modifier::{apply_stack, ModifierCache, ModifierInstance, ModifierKind};
pub use undo::{DocMut, EditError, UndoPayload};
