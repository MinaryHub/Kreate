//! The user interface.
//!
//! Reads a view of the world and emits [`command::Command`]s. It cannot mutate a
//! document, reach an operator implementation, or see the registry — the
//! application drains the queue and dispatches. That insulation is what keeps
//! undo, macro recording and scripting working from one mechanism, and it means
//! replacing the toolkit would replace one crate rather than the application.

pub mod command;
pub mod fonts;
pub mod keymap_editor;
pub mod modifier_stack;
pub mod nav_gizmo;
pub mod outliner;
pub mod property_editor;

pub use command::{Command, CommandQueue};
pub use fonts::{install as install_fonts, prewarm as prewarm_fonts};
pub use keymap_editor::{keymap_editor, KeymapEdit, KeymapText};
pub use modifier_stack::{modifier_stack, ModifierRow, ModifierText};
pub use outliner::{outliner, ClickModifiers, OutlinerRow, RowKind};
pub use property_editor::{property_editor, property_widget, UnitDisplay};
