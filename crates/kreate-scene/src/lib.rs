//! The document model: objects, datablocks, hierarchy, selection, and undo.
//!
//! Typed arenas rather than an ECS, and copy-on-write throughout — see
//! [`document`] for why both of those are load-bearing rather than stylistic.

pub mod animation;
pub mod armature;
pub mod camera;
pub mod clipboard;
pub mod design;
pub mod document;
pub mod driver;
pub mod edit_session;
pub mod expression;
pub mod feature;
pub mod geonodes;
pub mod image;
pub mod light;
pub mod modifier;
pub mod nla;
pub mod physics;
pub mod render;
pub mod robot;
pub mod shader;
pub mod shape;
pub mod sketch;
pub mod skin;
pub mod undo;

pub use camera::{Camera, Projection, SensorFit};
pub use document::{
    ActionId, ArmatureId, BlockKey, Changes, Collection, CollectionId, Document, LightId, Linkage,
    Material, MaterialId, MaterialMap, MaterialOverrides, MeshId, Object, ObjectData, ObjectId,
    ReparentError, Selection, WorldId, SITE_SHIFT,
};
pub use edit_session::{EditSession, EditSessions};
pub use image::{ColorSpace, Image, PixelFormat, Source, Tile, TILE};
/// A datablock's persistent identity and what kind of thing it names.
///
/// Re-exported rather than re-declared: the type lives in `kreate-reflect`
/// because a property path has to be holdable by the interface, which cannot
/// see a document. Anything that has a document in hand gets it from here.
pub use kreate_reflect::{DatablockKind, PropertyPath, Uid};
pub use light::{AreaShape, Background, Intensity, Light, LightKind, World};
pub use modifier::{apply_stack, ModifierCache, ModifierInstance, ModifierKind};
pub use undo::{DocMut, EditError, UndoPayload};
