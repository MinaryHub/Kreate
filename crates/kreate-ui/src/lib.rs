//! The user interface.
//!
//! Reads a view of the world and emits [`command::Command`]s. It cannot mutate a
//! document, reach an operator implementation, or see the registry — the
//! application drains the queue and dispatches. That insulation is what keeps
//! undo, macro recording and scripting working from one mechanism, and it means
//! replacing the toolkit would replace one crate rather than the application.

pub mod area;
pub mod assets;
pub mod chat;
pub mod collab;
pub mod command;
pub mod command_info;
pub mod command_palette;
pub mod console;
pub mod contact;
pub mod context_menu;
pub mod dope_sheet;
pub mod drivers;
pub mod element_icon;
pub mod fonts;
pub mod graph_editor;
pub mod icons;
pub mod keymap_editor;
pub mod modifier_stack;
pub mod nav_gizmo;
pub mod nla_editor;
pub mod node_editor;
pub mod object_glyph;
pub mod outliner;
pub mod panel;
pub mod pie_menu;
pub mod property_editor;
pub mod text;
pub mod theme;
pub mod timeline;
pub mod toolbar;
pub mod uv_editor;
pub mod view_icons;
pub mod workspace;
pub mod wsfile;

pub use area::{
    AreaId, AreaRect, Axis, EditorId, EditorInfo, EditorRegistry, Handle, Layout, NodeSpec, Screen,
    SplitId,
};
pub use command::{Command, CommandQueue};
pub use fonts::{install as install_fonts, prewarm as prewarm_fonts};
pub use keymap_editor::{keymap_editor, KeymapEdit, KeymapText};
pub use modifier_stack::{modifier_stack, ModifierRow, ModifierText};
pub use node_editor::{node_editor, GraphView, NodeEdit, NodeEditorState, NodeText};
pub use outliner::{outliner, ClickModifiers, OutlinerRow, RowKind};
pub use property_editor::{
    property_editor, property_editor_full, property_editor_keyed, property_widget, EditorOutcome,
    FieldAction, FieldMenu, FieldMenuText, KeyMark, UnitDisplay,
};
pub use text::{wrap, wrap_to_width, wrap_tooltip, wrapped_label};
pub use theme::{install as install_theme, Appearance, Density, Theme};
pub use uv_editor::{uv_editor, UvData, UvState, UvText, UvView};
pub use workspace::{Workspace, Workspaces};
