//! Input dispatch: the keymap and the order events are offered to things.
//!
//! The plan's pipeline is `IME gate → modal stack → gizmo hit-test → keymap →
//! viewport navigation → interface`, and the ordering is the design. This crate
//! owns the keymap half of it; the gate itself lives in the application, where
//! the window and the interface toolkit are.
//!
//! Depends on `kreate-reflect` and nothing else — not on the operator registry,
//! not on a window library. A keymap is data about *names*, and keeping it that
//! way is what lets it be loaded from a file, diffed against a preset, and
//! tested without a window.

#![forbid(unsafe_code)]

pub mod keyfile;
pub mod keymap;
pub mod presets;

pub use keyfile::{KeyFile, Problem as KeyFileProblem};
pub use keymap::{
    Action, Binding, BindingId, Context, EventKind, Input, Keymap, Modifiers, MouseButton, Override,
};
pub use presets::Preset;
