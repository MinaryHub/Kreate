//! Operators: the single path through which a document is ever changed.
//!
//! The UI never mutates a document directly; it dispatches `(op_id, params)`.
//! That one rule is what makes undo, macro recording, scripting, plugin parity,
//! and headless testing fall out of the same machinery instead of each needing
//! its own.

// `#[derive(OpParams)]` emits paths rooted at `::kreate_ops`, so that a
// downstream operator crate needs no imports. This alias makes the same
// generated code resolve inside this crate too, which is what lets the tests
// here exercise the real macro rather than a hand-written stand-in.
extern crate self as kreate_ops;

pub mod history;
pub mod operator;
pub mod ops;
/// The reflection types, re-exported so that `#[derive(OpParams)]` can keep
/// emitting `::kreate_ops::property::…` paths.
///
/// They live in `kreate-reflect`, a leaf crate, so that `kreate-ui` can read an
/// operator's schema and build a parameter map without depending on the operator
/// *implementations* — the UI emits `(op_id, params)` and the application
/// dispatches. That separation is what the architecture test enforces.
pub use kreate_reflect as property;

pub use history::{History, HistoryEntry, ModalSession};
pub use kreate_ops_derive::OpParams;
pub use operator::{Context, Mode, OpError, OpResult, Operator, OperatorInfo, OperatorRegistry};
pub use ops::register_builtin;
pub use property::{
    OpParams as OpParamsTrait, PropertyDesc, PropertyKind, PropertyMap, PropertyValue, Unit,
};

/// Re-exported so a caller can hold an edit session without naming the scene
/// crate: the operator layer is what hands one to `dispatch_in_session`.
pub use kreate_scene::EditSession;
