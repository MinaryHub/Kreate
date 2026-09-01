//! Localisation: the translation catalog and Korean-specific text handling.
//!
//! A separate crate rather than living inside `kreate-ui`, because the CLI, the
//! import reporter and error messages all need translated text, and none of them
//! should have to depend on a UI toolkit to get it.

pub mod catalog;
pub mod korean;

pub use catalog::{Catalog, CatalogError, TextKind, TranslationOptions};
pub use korean::{batchim_of, normalize, particle, wrap, Batchim};
