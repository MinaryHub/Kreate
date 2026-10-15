//! The egui shell: panels, the offscreen viewport, and the input gate.
//!
//! # The viewport is an offscreen texture, not an egui callback
//!
//! The 3D scene renders to its own texture, which is then drawn as an
//! `egui::Image`. That costs one blit and buys four things a paint callback
//! cannot give:
//!
//! - the viewport renders at a different *resolution* than the UI, so a
//!   render-scale slider and half-res-while-orbiting are possible at all;
//! - the viewport renders at a different *rate*, so a tooltip repaint does no 3D
//!   work;
//! - progressive accumulation while idle needs a persistent target;
//! - multiple viewports fall out naturally.
//!
//! # The input gate
//!
//! Keyboard shortcuts are dispatched only when egui does not want the keyboard.
//! Without that, typing a Korean object name into a rename field fires tool
//! shortcuts: with an IME active, `G` and `R` are swallowed by the composer or —
//! worse — reach both the text field and the keymap.

use egui::{Panel, Ui};
use kreate_i18n::{Catalog, TextKind};
use kreate_ops::{History, OperatorRegistry, PropertyMap};
use kreate_scene::document::{ObjectData, ObjectId};
use kreate_scene::modifier::ModifierKind;
use kreate_scene::Document;
use kreate_ui::outliner::{OutlinerRow, RowKind};
use kreate_ui::{Command, CommandQueue};

/// What the panels need from the application, without being able to touch it.
/// What the right-click menu offers, rebuilt when the mode changes.
///
/// Assembled by the application — only it has the registry, the keymap and the
/// document needed to decide what can run and what key runs it — and handed to
/// the interface, which draws it. The same rule as everywhere else here: the
/// interface reads a view model and dispatches commands.
#[derive(Default)]
pub struct ContextMenu {
    pub groups: Vec<kreate_ui::context_menu::Group>,
    /// Where it was opened, in interface points. `None` when it is closed.
    pub at: Option<egui::Pos2>,
}

pub struct ShellState {
    /// A physics bake running off the main thread: frames done, of how many.
    /// The Physics tab shows it with a Cancel button.
    pub bake_progress: Option<(usize, usize)>,
    /// Files being read, shown as a card over the viewport.
    pub importing: Vec<ImportStatus>,
    /// The read whose Cancel was pressed, for the application to stop.
    pub cancel_import: Option<u64>,
    pub catalog: Catalog,
    /// Every editor this build can show.
    ///
    /// A registry and not a variant list, which is what lets an add-on add one
    /// (Phase 9) and what lets an area name an editor this build does not have
    /// without the layout being thrown away. See [`kreate_ui::EditorRegistry`].
    pub editors: kreate_ui::EditorRegistry,
    pub queue: CommandQueue,
    /// Parameters shown in the adjust-last-operation panel.
    pub adjust_params: Option<(String, PropertyMap)>,
    /// The named layouts, and which is in use.
    ///
    /// This replaced four `show_*` booleans and the panels that read them, and
    /// then a single screen. The booleans could not express the thing a user of
    /// a tool like this expects to be able to do — put an editor where they
    /// want it, at the size they want, and *save that arrangement under a
    /// name*. See `kreate_ui::area` and `kreate_ui::workspace`.
    pub workspaces: kreate_ui::Workspaces,
    /// Layout changes asked for during the frame, applied after it.
    ///
    /// Editing the tree mid-frame would leave the rest of the frame drawing
    /// against a layout that no longer exists, in the same way that mutating a
    /// list while iterating it does. The command queue takes the same shape for
    /// the same reason.
    area_edits: Vec<AreaEdit>,
    /// Whether the layouts have changed since they were last written.
    ///
    /// Set by any applied layout edit, including a resize — which is why the
    /// application also waits for the drag to end before writing. Dragging a
    /// boundary changes the layout sixty times a second, and a file written on
    /// each of those frames is a file the disk is busy with instead of the
    /// frame.
    layout_dirty: bool,
    /// Which boundary a drag is moving.
    ///
    /// Held for the duration of the drag rather than resolved from the pointer
    /// each frame: once the pointer has moved past the boundary it is no longer
    /// over the handle, and a drag that re-resolved would let go.
    dragging: Option<kreate_ui::SplitId>,
    /// Set while egui holds the keyboard, so the keymap stands down.
    pub egui_wants_keyboard: bool,
    /// Set while an IME is composing.
    pub ime_active: bool,
    /// Whether the application is in mesh edit mode.
    /// Which element type edit-mode selection acts on. 0 vert, 1 edge, 2 face.
    pub select_mode: u32,
    /// What is being typed into the console, and what it has said so far.
    ///
    /// The scrollback holds only what was *typed* and the answers; the
    /// history's own lines are read from the history each frame, so an undo
    /// takes them out of the log the way it takes them out of the document.
    pub console_input: String,
    pub console_log: Vec<kreate_ui::console::ConsoleLine>,
    /// The Claude panel: what is being typed, the conversation, and whether
    /// Claude is answering. The process behind it is the application's.
    pub chat_input: String,
    pub chat_log: Vec<kreate_ui::chat::ChatMessage>,
    /// The asset browser. Which source, kind and size are chosen, by their
    /// place in `kreate::assets`'s lists, and what is typed to search.
    pub asset_source: usize,
    pub asset_kind: usize,
    pub asset_resolution: usize,
    pub asset_search: String,
    /// What the chosen source has of the chosen kind; the application's
    /// library fills it.
    pub asset_entries: Vec<crate::assets::Entry>,
    /// The local folders read.
    pub asset_folders: Vec<std::path::PathBuf>,
    /// Pictures arrived and not yet made textures, and the textures.
    pub asset_incoming: Vec<(String, crate::assets::Picture)>,
    pub asset_textures: std::collections::HashMap<String, egui::TextureHandle>,
    /// Tiles on screen with no picture asked for yet, by entry id: the
    /// application asks its library and moves them to `asset_asked`.
    pub asset_wanted: Vec<String>,
    pub asset_asked: std::collections::HashSet<String>,
    /// Set when the panel is drawn with nothing listed and nothing on its
    /// way: the application lists.
    pub asset_list_wanted: bool,
    pub asset_busy: Option<String>,
    pub asset_error: Option<String>,
    /// The asset looked at, by entry id: kept across a search narrowing the
    /// grid, and across a list read again.
    pub asset_selected: Option<String>,
    /// Assets on their way, by entry id: a second request for one is not a
    /// second copy.
    pub asset_fetching: std::collections::HashSet<String>,
    /// What a source that searches itself was last asked for.
    pub asset_query: crate::assets::Query,
    /// Whether the source has the key it wants: checked when the panel
    /// lists, and when the source refuses one.
    pub asset_key_held: bool,
    /// A model is being made by AI.
    pub asset_generating: bool,
    /// What goes with the next message. The application holds the files;
    /// this is what the panel shows of them.
    pub chat_attachments: Vec<kreate_ui::chat::ChatAttachment>,
    /// Whether the chat's input box has the keyboard: Ctrl+V there is the
    /// application's to read, for a picture or files on the clipboard.
    pub chat_focused: bool,
    pub chat_busy: bool,
    /// Which assistant the chat panel talks to: an index into
    /// `kreate::claude::Provider::ALL`.
    pub chat_provider: usize,
    /// The models the assistant can answer with (Ollama's), and which.
    pub chat_models: Vec<String>,
    pub chat_model: usize,
    /// Who Claude Code is signed in as, and the code box for signing in.
    pub chat_account: kreate_ui::chat::ChatAccount,
    pub chat_code: String,
    /// Help ▸ Contact's form, kept while closed so a half-written message
    /// survives, and whether Help ▸ About is showing.
    pub contact: kreate_ui::contact::ContactForm,
    /// File ▸ Shared Session: what was typed, and how the session stands.
    pub collab: kreate_ui::collab::CollabForm,
    pub collab_view: Option<kreate_ui::collab::CollabView>,
    pub about_open: bool,
    /// The last render, shown in its own window until closed.
    pub render_result: Option<RenderResultView>,
    /// Which material slot the Material tab is editing.
    ///
    /// **Interface state, not document state.** Blender keeps this on the
    /// object, which makes it undoable and saved; the operators here take the
    /// slot as a parameter instead, so the document holds only what a file has
    /// to. Clamped at draw time rather than kept in step, because the thing
    /// that shortens the list is a document edit and an undo can lengthen it
    /// again.
    pub material_slot: u32,
    /// The camera's rotation, for drawing the navigation gizmo. Interface state
    /// reads the camera; it never writes to it.
    pub camera_rotation: kreate_math::DQuat,
    /// The viewport camera's focal length in millimetres, for the sidebar's
    /// View tab. Published by the application; the tab asks for a new one
    /// through `view.set_lens`.
    pub view_lens_mm: f64,
    /// Which of the sidebar's tabs is showing.
    pub sidebar_tab: SidebarTab,
    /// Enough of the camera to put a world point on the screen, for the glyphs
    /// drawn over the picture for objects that have no surface — lights and
    /// cameras. `None` until the first frame has said how big the viewport is.
    pub projector: Option<kreate_ui::object_glyph::Projector>,
    /// Where the navigation gizmo is, in interface points.
    ///
    /// Geometry rather than a hover flag: it depends only on the viewport
    /// rectangle, so it does not go stale between frames the way "the pointer is
    /// over me" does. The application uses it to decide who owns a press.
    pub gizmo_circle: Option<(egui::Pos2, f32)>,
    /// Rectangles drawn over the viewport that the pointer belongs to rather
    /// than to the scene.
    ///
    /// Refilled every frame by whatever draws over the viewport, and consulted
    /// by the application before it decides a press is a navigation gesture.
    /// The viewport is *painted* rather than being a widget, so egui's
    /// `wants_pointer_input` answers a different question and answers it
    /// wrongly here — see `App::press_belongs_to_viewport`, which has the whole
    /// story. Geometry is what works, and this is the list of it.
    ///
    /// The navigation gizmo stays a circle of its own beside this: its bounding
    /// rectangle would take four corners of the viewport with it, and a drag
    /// starting in one of those corners should turn the scene.
    pub viewport_blockers: Vec<egui::Rect>,
    /// Which viewport areas have their sidebar open.
    ///
    /// Per area, like the UV editor's view: the layout permits two viewports,
    /// and one of them having the numbers open is not a statement about the
    /// other.
    sidebars: std::collections::BTreeSet<kreate_ui::AreaId>,
    /// The outliner's cached row order.
    ///
    /// Lives here because it is view state: derived from the document, rebuilt
    /// when the document's version moves, and of no interest to anything else.
    pub outliner: OutlinerOrder,
    /// What the outliner's search box holds. Empty: the whole tree.
    pub outliner_search: String,
    /// The rows the search matched, as indices into the tree's order, and
    /// what they were found for — the search and the document's version —
    /// so typing a letter searches once rather than every frame.
    outliner_found: Option<(String, u64, usize, Vec<usize>)>,
    /// The rubber-band rectangle to draw, in viewport-relative points.
    ///
    /// Set by the application while a selection drag is in progress. The
    /// interface draws it and nothing else: the rectangle it *means* is resolved
    /// on the GPU against the id buffer, and the two are the same numbers.
    pub select_band: Option<(egui::Pos2, egui::Pos2)>,
    /// Whether the drag is removing from the selection rather than adding.
    pub select_band_deselect: bool,
    /// The right-click menu: what it offers and whether it is open.
    pub context_menu: ContextMenu,
    /// F2's field: where it opened and the name typed so far.
    pub rename: Option<(egui::Pos2, String)>,
    /// What the viewport is currently doing, for the header to report.
    ///
    /// Pushed in by the application each frame rather than read out of the
    /// viewport, because the shell cannot see one — and because the exposure is
    /// not a viewport setting at all but something worked out from the document
    /// and the camera together.
    pub view_status: ViewStatus,
    /// Which tab each properties area is on.
    property_tabs: std::collections::HashMap<kreate_ui::AreaId, PropertiesTab>,
    /// The operator a panel is in the middle of dragging a value for.
    ///
    /// A slider changes on every frame it moves, and dispatching each change
    /// puts one undo step on the stack per frame — which makes undo useless in
    /// exactly the place people reach for it. So the first change of a gesture
    /// dispatches and the rest adjust the step it made, which is what
    /// `History::adjust_last` is for.
    ///
    /// Cleared when the pointer comes up, because that is what ends a gesture.
    /// Typing into a field and pressing return is one change and one step,
    /// which is also right.
    live_edit: Option<String>,
    /// Every command in the top row, with its shortcut and description.
    ///
    /// Keyed by operator id. Built by the application because resolving a
    /// shortcut needs the keymap, which the interface does not have and should
    /// not: §2.5's whole point is that a binding is data, and a panel that knew
    /// the keys would be a second place they are written down.
    pub command_info: std::collections::BTreeMap<String, kreate_ui::command_info::CommandInfo>,
    /// The last thing that happened, for the status bar.
    ///
    /// A single slot rather than a log: the status bar answers "did that work",
    /// and the answer to that question is always the most recent one.
    pub status: Option<String>,
    /// Autosaves from sessions that did not exit cleanly, as (path, name).
    ///
    /// Held here so the recovery offer is drawn by the interface and answered
    /// through the command queue, like everything else — rather than by a modal
    /// dialog at startup that the user has to dismiss before seeing the
    /// application at all. A crash is already a bad moment; being unable to look
    /// at the program while deciding makes it worse.
    pub recoverable: Vec<RecoveryOffer>,
    /// File ▸ Open Recent, newest first.
    pub recent_files: Vec<std::path::PathBuf>,
    /// The nodes Ctrl+C remembered: the material they are in, by name, which
    /// ones, and the corner of the box around them.
    pub node_clipboard: Option<(String, Vec<u32>, [f32; 2])>,
    /// The viewport drawn at this fraction of its pixels on each side, and
    /// stretched to fit: Blender's Resolution Scale, for a slow machine.
    pub resolution_scale: f32,
    /// The open document's name, `*` in front while it has unsaved changes:
    /// what the application bar's document chip shows.
    pub document_title: String,
    /// Whether the last operation's dialog shows its fields.
    pub adjust_open: bool,
    /// The chrome's appearance, as the preferences set it.
    pub appearance: kreate_ui::Appearance,
    /// Whether the remapping panel is open.

    /// The binding waiting for a keystroke, if any.
    ///
    /// Lives here rather than in the panel because the application, not the
    /// interface, is what sees the key event — the panel only says which binding
    /// is listening.
    pub capturing: Option<kreate_input::BindingId>,
    /// F3's search: open or not, what is typed, what was run from it lately.
    pub palette: kreate_ui::command_palette::PaletteState,
    /// The pie menu that is open, if one is.
    pub pie: kreate_ui::pie_menu::PieState,
    /// What the panel asked for this frame.
    pub keymap_edits: Vec<kreate_ui::KeymapEdit>,
    /// Whether the UV editor is open.

    /// The UV editor's pan, zoom and drag.
    ///
    /// Kept across the panel being closed and reopened: an artist who has
    /// zoomed into a sleeve does not want it reset for glancing at the
    /// modifier stack.
    /// Per-area view state for the UV editor: where it is looking and how
    /// far in.
    ///
    /// Keyed on the area rather than held once, because two UV editors are a
    /// thing the layout now permits and sharing a zoom between them would be
    /// baffling. The area id is stable for the area's life, so the state stays
    /// with it when a neighbour is split or closed.
    uv_views: std::collections::BTreeMap<kreate_ui::AreaId, kreate_ui::UvState>,
    /// Each shader editor's pan, zoom and selection, per area for the reason
    /// `uv_views` is.
    node_views: std::collections::BTreeMap<kreate_ui::AreaId, kreate_ui::NodeEditorState>,
    /// Each timeline's scrub state, per area.
    timeline_views:
        std::collections::BTreeMap<kreate_ui::AreaId, kreate_ui::timeline::TimelineState>,
    /// Each dope sheet's zoom and drag, per area.
    sheet_views: std::collections::BTreeMap<kreate_ui::AreaId, kreate_ui::dope_sheet::SheetState>,
    nla_views: std::collections::BTreeMap<kreate_ui::AreaId, kreate_ui::nla_editor::NlaState>,
    /// Each graph editor's view and drag, per area.
    graph_views: std::collections::BTreeMap<kreate_ui::AreaId, kreate_ui::graph_editor::GraphState>,
    /// The weight-paint brush: what it paints towards, how wide, how hard.
    pub paint: PaintSettings,
    /// The sculpt brush.
    pub sculpt: SculptSettings,
    pub texture_paint: TexturePaintSettings,
    /// `F` or `Shift+F` resizing or re-weighting the brush: drawn as a ring
    /// that stays put while the pointer sets its size.
    pub brush_adjust: Option<BrushAdjust>,
    /// Where `Ctrl+R` would cut, while it is looking for an edge: one world
    /// polyline per loop. Empty otherwise.
    pub loop_cut_preview: Vec<Vec<kreate_math::DVec3>>,
    /// The knife's path while `K` is drawing it: the points clicked, then
    /// where the next one would go. Empty otherwise.
    pub knife_preview: Vec<kreate_math::DVec3>,
    /// The sketch being drawn, as the sketch tools see it, for the overlay:
    /// see [`SketchView`]. `None` outside sketch mode.
    pub sketch_view: Option<SketchView>,
    /// Metres in one of the scene's length unit, for the dialogs.
    pub length_unit: f64,
    /// Every other sketch's lines, drawn thin wherever they are: a sketch
    /// has no faces for the renderer to draw.
    pub sketch_lines: Vec<[kreate_math::DVec3; 2]>,
    /// A dimension being typed into: its constraint's index and the text.
    pub sketch_dim_edit: Option<(usize, String)>,
    /// The extrude and revolve dialog's settings, kept between uses.
    pub solid_dialog: SolidDialog,
    /// The sketch pattern's settings, while its tool is in hand.
    pub pattern_dialog: PatternDialog,
    /// The ISO 286 fit calculator, when open.
    pub fits: FitsDialog,
    /// The feature list: every sketch and feature, in the order made.
    pub features: Vec<FeatureRow>,
    /// A feature being edited from the list.
    pub feature_edit: Option<FeatureEdit>,
    /// The Measure tool's points so far and where the next would go.
    pub measure_preview: Vec<kreate_math::DVec3>,
    /// The active object's volume, surface and middle, in the world, as it
    /// is evaluated. See `kreate_mesh::measure`.
    pub mass: Option<kreate_mesh::measure::MassProperties>,
    /// The thinnest wall Mesh Check allows, in metres.
    pub check_thickness: f64,
    /// What the last Mesh Check found, and on which object: counts in the
    /// order of `kreate_ops::ops::mesh::CHECK_ISSUES`.
    pub mesh_check: Option<(ObjectId, [usize; 7])>,
    /// Circle select's brush radius in interface points, while `C` is on.
    pub circle_select: Option<f32>,
    /// What a snapping drag has caught, for a marker on it.
    pub snap_target: Option<(kreate_math::DVec3, crate::snap::Element)>,
    /// Blender's Statistics overlay: counts in the viewport's corner.
    pub show_stats: bool,
    /// Whether the dimensions — measurements and sketch dimensions — are
    /// drawn over the picture. They stay in the document either way.
    pub show_dimensions: bool,
    /// The history's step names and how many are applied, for the Undo
    /// History menu.
    pub history_steps: Vec<String>,
    pub history_at: usize,
    /// What a rotation or a scale turns about — Blender's Pivot Point:
    /// 0 the median, 1 each object's own origin, 2 the active object, 3 the
    /// 3D cursor.
    pub pivot: u32,
    /// The header's tool settings: snapping, proportional editing, the
    /// transform orientation and the selection tool.
    pub tools: ToolSettings,
    /// A lasso being drawn, in interface points from the viewport's corner.
    pub select_lasso: Vec<egui::Pos2>,
    /// What is being typed in each Drivers editor, per area.
    driver_views: std::collections::BTreeMap<kreate_ui::AreaId, kreate_ui::drivers::DriversState>,
    /// The clock, published by the application each frame.
    pub timeline: kreate_ui::timeline::TimelineView,
    /// The UV map to draw, published by the application each frame.
    ///
    /// `None` outside edit mode, and empty when the mesh has no UVs yet —
    /// which the panel says out loud rather than drawing an empty square and
    /// leaving the user to guess.
    pub uv: Option<kreate_ui::UvData>,
    /// The running modal transform, if one is running.
    ///
    /// Set by the application every frame a gesture is live and cleared the
    /// moment it confirms or cancels. A drag is the one time the tool has to
    /// talk continuously — what it is doing, along which axis, by how much —
    /// and the status bar is the wrong place for it: the user is looking at
    /// the model, not at the bottom of the window.
    pub modal: Option<ModalHeader>,
}

/// One autosave the user is being offered back.
pub struct RecoveryOffer {
    pub path: std::path::PathBuf,
    /// The name the document had when it was written.
    pub name: String,
    /// How long ago, already translated.
    ///
    /// Formatted by the application rather than here: forty autosaves of a
    /// document called `untitled` are indistinguishable by name, and the time
    /// is the only thing that tells the user which one they want. Prose is the
    /// caller's to supply — the same rule every other panel follows.
    pub age: String,
}

/// What the modal header says while a transform is being dragged.
pub struct ModalHeader {
    /// The line the gesture composes: tool, constraint, precision, value.
    pub text: String,
    /// The constrained axis, if any, so the header can wear its colour.
    pub axis: Option<usize>,
    /// Which tool the gesture is, so the tool column can light it. `None` for
    /// a gesture the column has no button for.
    pub tool: Option<kreate_ui::toolbar::Tool>,
}

/// A layout change, queued during a frame and applied after it.
#[derive(Clone, Debug, PartialEq)]
enum AreaEdit {
    SetEditor(kreate_ui::AreaId, kreate_ui::EditorId),
    Split(kreate_ui::AreaId, kreate_ui::Axis),
    Close(kreate_ui::AreaId),
    Resize(kreate_ui::SplitId, f32, f32),
    /// Switch to another workspace. Deferred like the rest, because the frame
    /// is drawing tabs and areas from the list this changes.
    Switch(usize),
}

impl ShellState {
    /// The layout on screen.
    ///
    /// Through the workspace rather than a field of its own, so that an edit
    /// cannot land in a layout the user is not looking at.
    pub fn screen(&self) -> &kreate_ui::Screen {
        self.workspaces.screen()
    }

    /// The layout on screen, to change.
    pub fn screen_mut(&mut self) -> &mut kreate_ui::Screen {
        self.workspaces.screen_mut()
    }

    /// The default state, with a catalogue already loaded.
    ///
    /// A constructor rather than `ShellState { catalog, ..default() }` at the
    /// call site: the layout fields are private, and functional update syntax
    /// needs every field visible even for the ones it takes from the default.
    pub fn with_catalog(catalog: Catalog) -> Self {
        Self {
            catalog,
            ..Self::default()
        }
    }

    /// Asks for an area to be split when the frame ends.
    ///
    /// Named operations rather than a queue the caller pushes to, because there
    /// are three of them and they are the whole vocabulary: a keymap binding
    /// will want them as much as the header menu does, and a shortcut reaching
    /// into a private enum would not be able to.
    ///
    /// Deferred, all three, because the frame is drawing against a layout
    /// derived from the tree. Editing it here would leave every area after this
    /// one placed by a layout that no longer exists.
    pub fn split_area(&mut self, area: kreate_ui::AreaId, axis: kreate_ui::Axis) {
        self.area_edits.push(AreaEdit::Split(area, axis));
    }

    /// Asks for an area to be closed when the frame ends.
    pub fn close_area(&mut self, area: kreate_ui::AreaId) {
        self.area_edits.push(AreaEdit::Close(area));
    }

    /// Asks for an area to show another editor when the frame ends.
    pub fn set_area_editor(&mut self, area: kreate_ui::AreaId, kind: kreate_ui::EditorId) {
        self.area_edits.push(AreaEdit::SetEditor(area, kind));
    }

    /// Asks to switch workspace when the frame ends.
    pub fn switch_workspace(&mut self, index: usize) {
        self.area_edits.push(AreaEdit::Switch(index));
    }

    /// Opens or closes the sidebar of every viewport area.
    ///
    /// Every one rather than the one under the pointer, because the key that
    /// does this is pressed while looking at the viewport and the answer should
    /// not depend on where the pointer happened to rest. With one viewport —
    /// which is the usual case — the two are the same thing.
    pub fn toggle_sidebar(&mut self) {
        let viewports: Vec<kreate_ui::AreaId> = self
            .screen()
            .areas()
            .into_iter()
            .filter(|(_, kind)| *kind == kreate_ui::EditorId::VIEWPORT)
            .map(|(id, _)| id)
            .collect();
        let opening = viewports.iter().any(|id| !self.sidebars.contains(id));
        for id in viewports {
            if opening {
                self.sidebars.insert(id);
            } else {
                self.sidebars.remove(&id);
            }
        }
    }

    /// Which tab each properties area is showing.
    ///
    /// Per area rather than one setting, because two properties editors side by
    /// side — one on the material, one on the modifiers — is a layout somebody
    /// will build the day after this ships.
    pub fn properties_tab(&self, area: kreate_ui::AreaId) -> PropertiesTab {
        self.property_tabs.get(&area).copied().unwrap_or_default()
    }

    pub fn set_properties_tab(&mut self, area: kreate_ui::AreaId, tab: PropertiesTab) {
        self.property_tabs.insert(area, tab);
    }

    /// Whether a viewport area is showing its sidebar.
    pub fn sidebar_open(&self, area: kreate_ui::AreaId) -> bool {
        self.sidebars.contains(&area)
    }

    /// Whether any area is showing this editor.
    ///
    /// What the application asks instead of reading a `show_*` boolean. It is
    /// deliberately a question about the layout rather than about a setting:
    /// there is no separate notion of an editor being "on".
    pub fn has_editor(&self, kind: kreate_ui::EditorId) -> bool {
        self.screen().areas().iter().any(|(_, k)| *k == kind)
    }

    /// The UV editor under `at`, with the state it last drew with.
    ///
    /// `None` covers three situations that mean the same thing to the caller —
    /// no UV editor is open, none has drawn yet, or the pointer is elsewhere —
    /// so they are deliberately not distinguished.
    pub fn uv_at(&self, at: egui::Pos2) -> Option<(kreate_ui::AreaId, &kreate_ui::UvState)> {
        self.screen()
            .areas()
            .into_iter()
            .filter(|(_, kind)| *kind == kreate_ui::EditorId::UV_EDITOR)
            .find_map(|(id, _)| {
                let state = self.uv_views.get(&id)?;
                state.rect.filter(|r| r.contains(at)).map(|_| (id, state))
            })
    }

    /// Whether the pointer is over an editor that handles its own keys — the
    /// shader editor or the dope sheet.
    ///
    /// The application stops resolving its own keys there, all but the global
    /// ones — `X` is "delete the object" over the viewport, "delete the node"
    /// over the graph and "delete the keys" over the dope sheet, and each
    /// editor handles its own.
    pub fn node_editor_at(&self, at: egui::Pos2) -> bool {
        let over = |rect: Option<egui::Rect>| rect.is_some_and(|r| r.contains(at));
        self.screen().areas().into_iter().any(|(id, kind)| {
            if kind == kreate_ui::EditorId::SHADER || kind == kreate_ui::EditorId::GEOMETRY {
                over(self.node_views.get(&id).and_then(|s| s.rect))
            } else if kind == kreate_ui::EditorId::DOPE_SHEET {
                over(self.sheet_views.get(&id).and_then(|s| s.rect))
            } else if kind == kreate_ui::EditorId::NLA {
                over(self.nla_views.get(&id).and_then(|s| s.rect))
            } else if kind == kreate_ui::EditorId::GRAPH {
                over(self.graph_views.get(&id).and_then(|s| s.rect))
            } else {
                false
            }
        })
    }

    /// Whether the layouts are worth writing to disk right now.
    ///
    /// Not while a boundary is being dragged: the layout changes every frame of
    /// a drag and settles when the pointer is released, and only the settled
    /// answer is worth a file.
    pub fn layout_needs_saving(&self) -> bool {
        self.layout_dirty && self.dragging.is_none()
    }

    /// Records that the layouts have been written.
    pub fn layout_saved(&mut self) {
        self.layout_dirty = false;
    }

    fn apply_area_edits(&mut self) {
        let edits = std::mem::take(&mut self.area_edits);
        self.layout_dirty |= !edits.is_empty();
        for edit in edits {
            match edit {
                AreaEdit::SetEditor(area, kind) => {
                    self.screen_mut().set_editor(area, kind);
                }
                AreaEdit::Split(area, axis) => {
                    // The new area shows what the old one showed. Splitting is
                    // "I want two of these here"; changing one of them is a
                    // separate act, and guessing a different editor would be
                    // wrong more often than right.
                    let kind = self
                        .screen()
                        .editor(area)
                        .unwrap_or(kreate_ui::EditorId::VIEWPORT);
                    self.screen_mut().split(area, axis, 0.5, kind);
                }
                AreaEdit::Close(area) => {
                    if self.screen_mut().close(area) {
                        // The state of an area that no longer exists would
                        // otherwise accumulate for the life of the session.
                        self.uv_views.remove(&area);
                        self.node_views.remove(&area);
                        self.timeline_views.remove(&area);
                        self.sheet_views.remove(&area);
                        self.nla_views.remove(&area);
                        self.graph_views.remove(&area);
                        self.driver_views.remove(&area);
                    }
                }
                AreaEdit::Resize(split, ratio, extent) => {
                    self.screen_mut().resize(split, ratio, extent);
                }
                AreaEdit::Switch(index) => {
                    self.workspaces.switch_to(index);
                }
            }
        }
    }
}

impl Default for ShellState {
    fn default() -> Self {
        Self {
            bake_progress: None,
            importing: Vec::new(),
            cancel_import: None,
            editors: kreate_ui::EditorRegistry::with_builtins(),
            catalog: Catalog::new(),
            material_slot: 0,
            console_input: String::new(),
            console_log: Vec::new(),
            chat_input: String::new(),
            chat_log: Vec::new(),
            asset_source: 1,
            asset_kind: 0,
            asset_resolution: 0,
            asset_search: String::new(),
            asset_entries: Vec::new(),
            asset_folders: Vec::new(),
            asset_incoming: Vec::new(),
            asset_textures: std::collections::HashMap::new(),
            asset_wanted: Vec::new(),
            asset_asked: std::collections::HashSet::new(),
            asset_list_wanted: false,
            asset_busy: None,
            asset_error: None,
            asset_selected: None,
            asset_fetching: std::collections::HashSet::new(),
            asset_query: crate::assets::Query::default(),
            asset_key_held: false,
            asset_generating: false,
            chat_attachments: Vec::new(),
            chat_focused: false,
            chat_busy: false,
            chat_provider: 0,
            chat_models: Vec::new(),
            chat_model: 0,
            chat_account: kreate_ui::chat::ChatAccount::Checking,
            chat_code: String::new(),
            contact: kreate_ui::contact::ContactForm::new(),
            collab: Default::default(),
            collab_view: None,
            about_open: false,
            render_result: None,
            queue: CommandQueue::new(),
            adjust_params: None,
            workspaces: kreate_ui::Workspaces::default(),
            area_edits: Vec::new(),
            layout_dirty: false,
            dragging: None,
            egui_wants_keyboard: false,
            ime_active: false,
            select_mode: 2,
            camera_rotation: kreate_math::DQuat::IDENTITY,
            view_lens_mm: 50.0,
            sidebar_tab: SidebarTab::Item,
            projector: None,
            gizmo_circle: None,
            viewport_blockers: Vec::new(),
            sidebars: std::collections::BTreeSet::new(),
            outliner: OutlinerOrder::default(),
            outliner_search: String::new(),
            outliner_found: None,
            select_band: None,
            select_band_deselect: false,
            context_menu: ContextMenu::default(),
            view_status: ViewStatus::default(),
            property_tabs: std::collections::HashMap::new(),
            live_edit: None,
            command_info: std::collections::BTreeMap::new(),
            status: None,
            recoverable: Vec::new(),
            recent_files: Vec::new(),
            node_clipboard: None,
            resolution_scale: 1.0,
            document_title: "untitled".into(),
            adjust_open: false,
            appearance: kreate_ui::Appearance::Dark,
            capturing: None,
            palette: Default::default(),
            pie: Default::default(),
            keymap_edits: Vec::new(),
            uv_views: std::collections::BTreeMap::new(),
            node_views: std::collections::BTreeMap::new(),
            timeline_views: std::collections::BTreeMap::new(),
            sheet_views: std::collections::BTreeMap::new(),
            nla_views: std::collections::BTreeMap::new(),
            graph_views: std::collections::BTreeMap::new(),
            driver_views: std::collections::BTreeMap::new(),
            paint: PaintSettings::default(),
            sculpt: SculptSettings::default(),
            brush_adjust: None,
            texture_paint: TexturePaintSettings::default(),
            loop_cut_preview: Vec::new(),
            knife_preview: Vec::new(),
            sketch_view: None,
            length_unit: 0.001,
            sketch_lines: Vec::new(),
            sketch_dim_edit: None,
            solid_dialog: SolidDialog::default(),
            pattern_dialog: PatternDialog::default(),
            fits: FitsDialog::default(),
            features: Vec::new(),
            feature_edit: None,
            measure_preview: Vec::new(),
            mass: None,
            check_thickness: 0.001,
            mesh_check: None,
            circle_select: None,
            snap_target: None,
            rename: None,
            show_stats: false,
            show_dimensions: true,
            history_steps: Vec::new(),
            history_at: 0,
            pivot: 0,
            tools: ToolSettings {
                auto_merge: true,
                ..ToolSettings::default()
            },
            select_lasso: Vec::new(),
            timeline: Default::default(),
            uv: None,
            modal: None,
        }
    }
}

impl ShellState {
    /// Whether the panel is listening for a keystroke to bind.
    ///
    /// While it is, the keymap must not act on keys — pressing `E` to bind it
    /// would also extrude.
    pub fn is_capturing(&self) -> bool {
        self.capturing.is_some()
    }

    /// Whether a viewport keyboard shortcut may run this frame.
    ///
    /// The gate that keeps a rename field from firing tool shortcuts. It is a
    /// single check in a single place on purpose: scattered `if !typing` tests
    /// are how the bug comes back.
    pub fn keymap_enabled(&self) -> bool {
        // Not while the search is open: its field has the keyboard from the
        // frame after it opens, but on the frame it opens a typed `E` would
        // otherwise extrude.
        !self.egui_wants_keyboard
            && !self.ime_active
            && !self.is_capturing()
            && !self.overlay_open()
    }

    /// Whether a search box or a pie has the keyboard. While one does, keys
    /// are its — and `Escape` closes it rather than doing anything else.
    pub fn overlay_open(&self) -> bool {
        self.palette.open || self.pie.open.is_some()
    }
}

/// The outliner's row order, and what it was built from.
///
/// Cached rather than rebuilt every frame. The order is one `u32` per object —
/// cheap to hold and cheap to hand to the widget whole — while the rows
/// themselves are materialised only for what is on screen. Building a hundred
/// thousand rows to draw twenty is the immediate-mode trap the outliner's own
/// module docs warn about, and it costs a hundred thousand `String` clones a
/// frame.
#[derive(Default)]
pub struct OutlinerOrder {
    /// Object keys in hierarchy order.
    ids: Vec<u32>,
    /// Indentation per row, parallel to `ids`.
    depths: Vec<u16>,
    /// How many objects use each mesh, by mesh slot. Counted once for the whole
    /// scene rather than asked per row, which would be quadratic.
    mesh_users: Vec<u32>,
    /// The collections, as headings: where each one's objects start.
    groups: Vec<kreate_ui::outliner::OutlinerGroup>,
    /// The document version this was built from.
    version: Option<u64>,
}

impl OutlinerOrder {
    /// Rebuilds the order if the document has changed since it was built.
    pub fn sync(&mut self, doc: &Document) {
        if self.version == Some(doc.version()) {
            return;
        }
        let hierarchy = doc.hierarchy();
        self.ids.clear();
        self.depths.clear();
        self.groups.clear();
        self.ids.reserve(hierarchy.len());
        self.depths.reserve(hierarchy.len());
        // Each collection's objects under its heading, one level in; what is
        // in none after them, in the scene itself. A child goes where the top
        // of its family goes, so a hierarchy is never split across headings.
        let collections: Vec<_> = doc
            .collections()
            .map(|(id, c)| (id, c.name.clone(), c.hide))
            .collect();
        let mut buckets: Vec<Vec<(u32, u16)>> = vec![Vec::new(); collections.len() + 1];
        let mut bucket = collections.len();
        for (id, depth) in hierarchy {
            if depth == 0 {
                bucket = doc
                    .collection_of(id)
                    .and_then(|c| collections.iter().position(|(x, _, _)| *x == c))
                    .unwrap_or(collections.len());
            }
            let indent = depth + usize::from(bucket < collections.len());
            buckets[bucket].push((object_key(id), indent.min(u16::MAX as usize) as u16));
        }
        for (i, rows) in buckets.into_iter().enumerate() {
            if let Some((_, name, hide)) = collections.get(i) {
                self.groups.push(kreate_ui::outliner::OutlinerGroup {
                    start: self.ids.len(),
                    name: name.clone(),
                    hidden: *hide,
                });
            }
            for (key, depth) in rows {
                self.ids.push(key);
                self.depths.push(depth);
            }
        }
        self.mesh_users = doc.mesh_user_counts();
        self.version = Some(doc.version());
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Indentation per row, parallel to `ids`, for the tree's collapsing.
    pub fn depths(&self) -> &[u16] {
        &self.depths
    }

    /// The collection headings.
    pub fn groups(&self) -> &[kreate_ui::outliner::OutlinerGroup] {
        &self.groups
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Builds one row, for a position the widget is about to draw.
    ///
    /// By index rather than by key, so the depth is a lookup rather than a
    /// search: searching the order for each visible row would be a couple of
    /// million comparisons a frame at a hundred thousand objects, which is the
    /// same mistake one level down.
    ///
    /// `None` when the row names no live object, which happens for a frame after
    /// a delete.
    pub fn row(&self, doc: &Document, index: usize) -> Option<OutlinerRow> {
        use kreate_mesh::arena::ArenaIndex as _;

        let key = self.ids.get(index).copied()?;
        let id = doc.object_at_slot(key)?;
        let object = doc.object(id)?;
        let kind = match object.data {
            ObjectData::Empty => RowKind::Empty,
            ObjectData::Camera(_) => RowKind::Camera,
            ObjectData::Light(_) => RowKind::Light,
            ObjectData::Armature(_) => RowKind::Armature,
            ObjectData::Mesh(mesh_id) => {
                let mesh = doc.mesh(mesh_id);
                RowKind::Mesh {
                    verts: mesh.map(|m| m.num_verts()).unwrap_or(0),
                    faces: mesh.map(|m| m.num_faces()).unwrap_or(0),
                }
            }
        };
        let depth = self.depths.get(index).copied().unwrap_or(0);
        // What the object is made of, for the rows under it when expanded.
        let data_name = match object.data {
            ObjectData::Empty => None,
            ObjectData::Camera(c) => doc.camera(c).map(|c| c.name.clone()),
            ObjectData::Light(l) => doc.light(l).map(|l| l.name.clone()),
            ObjectData::Armature(a) => doc.armature(a).map(|a| a.name.clone()),
            ObjectData::Mesh(m) => doc.mesh(m).map(|_| object.name.clone()),
        };
        let materials: Vec<String> = match object.data {
            ObjectData::Mesh(m) => (0..doc.mesh_slots(m).len())
                .map(|slot| {
                    doc.material_at(id, slot)
                        .map(|mat| mat.name.clone())
                        .unwrap_or_default()
                })
                .collect(),
            _ => Vec::new(),
        };
        let has_children = self.depths.get(index + 1).is_some_and(|next| *next > depth);
        Some(OutlinerRow {
            id: key,
            name: object.name.clone(),
            depth: depth as usize,
            selected: doc.selection.contains(id),
            active: doc.selection.active == Some(id),
            visible: object.visible,
            shown: doc.is_shown(id),
            kind,
            data_name,
            materials,
            has_children,
            // Each bone at its depth under its parent, the active one lit.
            bones: doc
                .armature_of(id)
                .map(|(_, a)| {
                    a.bones
                        .iter()
                        .map(|b| {
                            let mut depth = 0;
                            let mut at = b.parent;
                            while let Some(p) = at {
                                depth += 1;
                                at = a.bones.get(p).and_then(|b| b.parent);
                            }
                            (
                                b.name.clone(),
                                depth,
                                object.pose.active.as_deref() == Some(b.name.as_str()),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default(),
            mesh_users: object
                .mesh()
                .and_then(|m| self.mesh_users.get(m.index() as usize))
                .copied()
                .unwrap_or(1) as usize,
        })
    }
}

/// Builds every row, for a caller that genuinely wants them all.
///
/// Only tests and the status bar want this; the panel does not — see
/// [`OutlinerOrder`].
pub fn outliner_rows(doc: &Document) -> Vec<OutlinerRow> {
    let mut order = OutlinerOrder::default();
    order.sync(doc);
    (0..order.len())
        .filter_map(|index| order.row(doc, index))
        .collect()
}

/// Builds the modifier stack's view model for the active object.
///
/// Empty when nothing is active, which the panel draws differently from "this
/// object has no modifiers" — the two are different states and must not look
/// the same.
///
/// Each row carries the setter operator's id, its reflected schema and the
/// entry's current parameters *in that operator's terms*, so the panel draws
/// itself from reflection and knows nothing about what a mirror is.
pub fn modifier_rows(
    doc: &Document,
    registry: &OperatorRegistry,
    catalog: &Catalog,
) -> Vec<kreate_ui::ModifierRow> {
    let Some(id) = doc.selection.active else {
        return Vec::new();
    };
    let Some(object) = doc.object(id) else {
        return Vec::new();
    };
    object
        .modifiers
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let op_id = setter_for(&entry.kind);
            let info = registry.get(op_id)?;
            let name = catalog.format(entry.kind.name_key(), TextKind::Ui, &[]);
            Some(kreate_ui::ModifierRow {
                index: index as u32,
                name,
                enabled: entry.enabled,
                op_id: op_id.to_string(),
                schema: info.schema,
                params: kreate_ops::ops::modifier::setter_params(doc, index as u32, &entry.kind),
                // A boolean names another mesh object: offer them.
                choices: match entry.kind {
                    // A path is any other object with points: a mesh's
                    // edges or a sketch's lines.
                    ModifierKind::PathArray(_) => doc
                        .objects()
                        .filter(|(other, o)| *other != id && o.mesh().is_some())
                        .map(|(_, o)| o.name.clone())
                        .collect(),
                    ModifierKind::Boolean(_) => doc
                        .objects()
                        .filter(|(other, o)| *other != id && o.mesh().is_some())
                        .map(|(_, o)| o.name.clone())
                        .collect(),
                    _ => Vec::new(),
                },
            })
        })
        .collect()
}

/// Which operator writes a given kind's parameters.
fn setter_for(kind: &ModifierKind) -> &'static str {
    match kind {
        ModifierKind::Mirror(_) => "object.modifier_set_mirror",
        ModifierKind::Subdivide(_) => "object.modifier_set_subdivide",
        ModifierKind::Array(_) => "object.modifier_set_array",
        ModifierKind::Solidify(_) => "object.modifier_set_solidify",
        ModifierKind::Bevel(_) => "object.modifier_set_bevel",
        ModifierKind::Boolean(_) => "object.modifier_set_boolean",
        ModifierKind::PathArray(_) => "object.modifier_set_path_array",
    }
}

/// Opaque key for an object, so the UI can round-trip a handle it cannot read.
pub fn object_key(id: ObjectId) -> u32 {
    use kreate_mesh::arena::ArenaIndex as _;
    id.index()
}

/// Keeps `Tab` for the keymap unless a text field is being typed in.
///
/// egui spends `Tab` moving keyboard focus between widgets, and with nothing
/// focused it hands focus to the first button on screen. That button then
/// holds the keyboard, so the keymap is off: the first `Tab` entered edit mode
/// and the second did nothing. In a field, `Tab` still moves to the next one.
pub fn keep_tab_for_the_keymap(ctx: &egui::Context, input: &mut egui::RawInput) {
    if ctx.text_edit_focused() {
        return;
    }
    input.events.retain(|event| {
        !matches!(
            event,
            egui::Event::Key {
                key: egui::Key::Tab,
                ..
            }
        )
    });
}

/// Finds the object an outliner row refers to.
///
/// The key *is* the arena slot, so this is a lookup rather than a search — which
/// matters because resolving a selection resolves one key per object, and a
/// scan per key made a large selection quadratic.
///
/// A row from a frame where the object was deleted resolves to nothing. A slot
/// that has since been *reused* resolves to its new occupant: the key carries no
/// generation, and the window between a delete and a click is one frame.
pub fn object_from_key(doc: &Document, key: u32) -> Option<ObjectId> {
    doc.object_at_slot(key)
}

/// The offscreen viewport, and how much of it holds the image.
#[derive(Clone, Copy, Debug)]
pub struct ViewportImage {
    pub texture: egui::TextureId,
    /// Upper-right UV of the drawn region.
    pub uv: (f32, f32),
}

/// What the header says about the picture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewStatus {
    pub shading: kreate_render::ShadingMode,
    /// The exposure the picture is being made at, in stops.
    pub exposure: f32,
    /// Whether that came from the light meter or from the camera's dials.
    ///
    /// Worth saying on screen, because the two behave differently in a way that
    /// otherwise reads as a fault: with the meter on, turning a lamp down
    /// changes nothing — the exposure opens by exactly as much. A number that
    /// visibly moves when the lamp does is the difference between that being
    /// explicable and being a bug report.
    pub metered: bool,
    /// Whether the camera is orthographic. For the view-information text in
    /// the viewport's corner, which is where Blender says it and where a user
    /// looks when a picture has stopped having depth.
    pub orthographic: bool,
    /// Samples the path tracer has summed, zero when it is not tracing.
    pub samples: u32,
    /// Which overlays are on, for the menu's check marks.
    pub overlays: Overlays,
}

/// The header's tool settings. The application owns snapping and
/// proportional editing (keys change them too) and copies them in before the
/// interface is drawn and back out after; the orientation and the selection
/// tool live here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToolSettings {
    pub snap: bool,
    /// An index into `crate::snap::Mode::ALL`.
    pub snap_mode: u32,
    pub proportional: bool,
    /// An index into `kreate_ops::ops::mesh::FALLOFF_CURVES`.
    pub falloff: u32,
    /// 0 global, 1 local: what the first X, Y or Z press constrains to.
    pub orientation: u32,
    /// 0 box, 1 lasso, 2 circle: what a left drag in the viewport selects with.
    pub select_tool: u32,
    /// Edit mode: a vertex moved onto another merges with it.
    pub auto_merge: bool,
}

/// The viewport's on-off settings, as the Overlays menu shows them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overlays {
    /// The solid view lit flat — Blender's Flat lighting — rather than by
    /// the studio's lights.
    pub flat: bool,
    pub grid: bool,
    pub wireframe: bool,
    pub face_orientation: bool,
    pub statistics: bool,
    pub dimensions: bool,
    pub ambient_occlusion: bool,
    pub xray: bool,
    pub outline: bool,
    pub bloom: bool,
    pub reflections: bool,
    pub contact_shadows: bool,
    pub path_trace: bool,
    /// The studio and flat modes show base-colour images, not just colours.
    pub textures: bool,
}

impl Default for ViewStatus {
    fn default() -> Self {
        Self {
            shading: kreate_render::ShadingMode::Studio,
            exposure: 0.0,
            metered: false,
            orthographic: false,
            samples: 0,
            overlays: Overlays::default(),
        }
    }
}

/// What a properties editor is showing.
///
/// A column of these down the left, as §2.6 settled: the shape has to hold
/// render, output, scene, world, object, modifiers, physics, constraints, data
/// and material before it is finished, and a list of headings stops being
/// readable several entries before that.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PropertiesTab {
    /// The last operation's parameters. Blender's F9, docked; the same panel
    /// also floats in the viewport's corner, which is where a hand looks.
    Tool,
    /// Where the active object is. Where the editor opens, as Blender's does.
    #[default]
    Object,
    /// What the active object *is*: its datablock.
    Data,
    /// The active bone of an armature: its pose, and the bone as built.
    Bone,
    /// What it is made of.
    Material,
    /// The light arriving from no object in particular.
    World,
    /// The modifier stack.
    Modifiers,
    /// Rigid body, collision, cloth, particles; the scene's physics and bake.
    Physics,
    /// How the scene renders: size, samples, where the file goes.
    Render,
}

impl PropertiesTab {
    /// The tab's picture. The Data tab wears the icon of what the active
    /// object is — a bulb, a camera, a bone — as Blender's does, so the column
    /// says what that tab is about before it is opened.
    fn icon(self, data: kreate_ui::outliner::RowKind) -> kreate_ui::icons::Icon {
        use kreate_ui::icons::Icon;
        match self {
            Self::Tool => Icon::Tool,
            Self::Object => Icon::Object,
            Self::Data => Icon::Data(data),
            Self::Bone => Icon::Bone,
            Self::Material => Icon::Material,
            Self::World => Icon::World,
            Self::Modifiers => Icon::Modifiers,
            Self::Render => Icon::Render,
            Self::Physics => Icon::Physics,
        }
    }

    fn label_key(self) -> &'static str {
        match self {
            Self::Tool => "tab-tool",
            Self::Object => "tab-object",
            Self::Data => "tab-data",
            Self::Bone => "tab-bone",
            Self::Material => "tab-material",
            Self::World => "tab-world",
            Self::Modifiers => "tab-modifiers",
            Self::Render => "tab-render",
            Self::Physics => "tab-physics",
        }
    }
}

/// The weight-paint brush's settings. Application state, as Blender keeps its
/// tool settings apart from the file's data; a stroke records the values it
/// used, so a replay does not depend on them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintSettings {
    /// What the brush paints towards, 0 to 1.
    pub weight: f32,
    /// Its radius on screen, in points.
    pub radius: f32,
    /// How far towards the weight one stroke goes.
    pub strength: f32,
    /// Only what faces the view — Blender's Front Faces Only — so a stroke
    /// on the front of an arm does not paint the back of it.
    pub front_only: bool,
    /// Softer towards the rim, as Blender's Smooth falloff; off, the whole
    /// disc paints at full strength.
    pub falloff: bool,
}

impl Default for PaintSettings {
    fn default() -> Self {
        Self {
            weight: 1.0,
            radius: 40.0,
            strength: 0.5,
            front_only: true,
            falloff: true,
        }
    }
}

/// The texture paint brush's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexturePaintSettings {
    /// The colour, in the image's own sRGB encoding — what the swatch shows.
    pub color: [f32; 3],
    /// Its radius on screen, in points.
    pub radius: f32,
    pub strength: f32,
}

impl Default for TexturePaintSettings {
    fn default() -> Self {
        Self {
            color: [0.8, 0.1, 0.1],
            radius: 30.0,
            strength: 1.0,
        }
    }
}

/// Blender's `F` / `Shift+F`: the brush's ring held still, its middle where
/// the pointer was less the value, so the pointer starts on the edge and the
/// distance from the middle is the new value.
#[derive(Clone, Copy, Debug)]
pub struct BrushAdjust {
    /// The strength, not the radius.
    pub strength: bool,
    /// The ring's middle, in points.
    pub center: egui::Pos2,
    /// The radius and strength the brush had, for a cancel.
    pub radius: f32,
    pub was: f32,
}

/// Dynamic topology whenever cutting the surface throws nothing away: a
/// mesh with nothing but its shape, or UVs no texture uses.
pub const DYNTOPO_AUTO: u32 = 0;
/// Always, whatever it throws away.
pub const DYNTOPO_ON: u32 = 1;
pub const DYNTOPO_OFF: u32 = 2;

/// The sculpt brush's settings, the application's as the paint brush's are.
#[derive(Clone, Debug, PartialEq)]
pub struct SculptSettings {
    /// Which brush, as `kreate_mesh::sculpt::Brush::ALL` numbers them.
    pub brush: u32,
    /// Its radius on screen, in points.
    pub radius: f32,
    pub strength: f32,
    pub mirror_x: bool,
    pub mirror_y: bool,
    pub mirror_z: bool,
    /// Repeated round Z this many times; 1 is once.
    pub radial: u32,
    /// How it fades to its rim, as `Falloff::ALL` numbers them.
    pub falloff: u32,
    /// Only what faces the view.
    pub front_only: bool,
    /// How much each dab also smooths.
    pub auto_smooth: f32,
    /// Dabs this far apart, as a fraction of the brush's radius.
    pub spacing: f32,
    /// The stabiliser: how far the brush trails the pointer, 0 to nearly 1.
    pub lazy: f32,
    /// A picture stamped through the brush: a built-in name or an image's.
    pub alpha: String,
    /// The pen's pressure scales the strength.
    pub pressure_strength: bool,
    /// The pen's pressure scales the size.
    pub pressure_size: bool,
    /// Dynamic topology — the surface cut finer where the brush goes — as
    /// [`DYNTOPO_AUTO`], [`DYNTOPO_ON`] or [`DYNTOPO_OFF`].
    pub dyntopo: u32,
    /// Dynamic topology's edge length, as a fraction of the brush's radius.
    pub detail: f32,
    /// The voxel remesh's voxel size, in the mesh's units.
    pub voxel_size: f32,
    /// About how many faces the quad remesh makes.
    pub quad_faces: u32,
    /// The mask and face sets shown on the surface.
    pub show_overlay: bool,
}

impl Default for SculptSettings {
    fn default() -> Self {
        Self {
            brush: 0,
            radius: 50.0,
            strength: 0.5,
            mirror_x: false,
            mirror_y: false,
            mirror_z: false,
            radial: 1,
            falloff: 0,
            front_only: false,
            auto_smooth: 0.0,
            spacing: 0.25,
            lazy: 0.0,
            alpha: String::new(),
            pressure_strength: true,
            pressure_size: false,
            dyntopo: DYNTOPO_AUTO,
            detail: 0.2,
            voxel_size: 0.05,
            quad_faces: 2000,
            show_overlay: true,
        }
    }
}

/// Sends a panel's edit, as a new step or as an adjustment to the last one.
///
/// The whole of the rule in one place, because getting it wrong is invisible
/// until somebody presses ctrl-Z and finds three hundred steps of a slider they
/// dragged once. See `ShellState::live_edit`.
fn send_edit(state: &mut ShellState, op_id: &str, params: PropertyMap, continuing: bool) {
    if continuing && state.live_edit.as_deref() == Some(op_id) {
        state.queue.push(Command::AdjustLast { params });
    } else {
        state.queue.run(op_id, params);
        state.live_edit = Some(op_id.to_string());
    }
}

/// Ends a gesture, so the next edit starts a step of its own.
///
/// Called once a frame with whether the pointer is still down. Without it the
/// *second* drag of the same slider would go on adjusting the first one's step,
/// and the first drag would become impossible to undo — which is the same bug
/// as the one `send_edit` exists to prevent, arriving from the other side.
pub fn end_gesture(state: &mut ShellState, pointer_down: bool) {
    if !pointer_down {
        state.live_edit = None;
    }
}

/// Panel and menu labels, resolved once per frame.
struct Labels {
    file_new: String,
    file_open: String,
    file_open_recent: String,
    menu_select_similar: String,
    /// One per `kreate_mesh::selection::SIMILAR`.
    similar: Vec<String>,
    recent_none: String,
    menu_show_hide: String,
    reveal_hidden: String,
    file_save: String,
    file_save_as: String,
    file_import_menu: String,
    file_export_menu: String,
    file_any_format: String,
    file_run_script: String,
    menu_search: String,
    menu_copy_script: String,
    palette_placeholder: String,
    pie_shading: String,
    time_play: String,
    key_property: String,
    time_autokey: String,
    sheet_empty: String,
    graph_handles: String,
    handle_auto: String,
    handle_flat: String,
    sheet_interpolation: String,
    interp_constant: String,
    interp_linear: String,
    interp_bezier: String,
    time_pause: String,
    time_play_menu: String,
    time_jump_start: String,
    time_jump_end: String,
    time_start: String,
    time_end: String,
    menu_animation: String,
    anim_insert: String,
    anim_delete: String,
    pie_view: String,
    pie_mode: String,
    view_left: String,
    view_right: String,
    view_bottom: String,
    view_top: String,
    view_front: String,
    view_back: String,
    /// Each mode's name, by its tag.
    mode_names: std::collections::HashMap<&'static str, String>,
    palette_nothing: String,
    world_pick_image: String,
    world_clear_image: String,
    material_pick_image: String,
    material_clear_image: String,
    tab_material: String,
    material_new: String,
    material_browse: String,
    material_assign: String,
    material_assign_hint: String,
    material_select: String,
    material_select_hint: String,
    material_none: String,
    material_base_colour: String,
    material_metallic_roughness: String,
    material_normal_map: String,
    material_emission_map: String,
    material_occlusion_map: String,
    material_uses_nodes: String,
    node_use_nodes: String,
    node_use_nodes_hint: String,
    node_no_graph: String,
    node_no_material: String,
    node_add: String,
    node_no_image: String,
    node_open_image: String,
    material_empty_slot: String,
    console_placeholder: String,
    console_empty: String,
    chat_placeholder: String,
    chat_attach: String,
    chat_attach_hint: String,
    chat_remove: String,
    chat_empty: String,
    chat_send: String,
    chat_stop: String,
    chat_new: String,
    chat_history: String,
    chat_export: String,
    help_logs: String,
    chat_working: String,
    chat_checking: String,
    chat_signed_out: String,
    chat_sign_in: String,
    chat_sign_out: String,
    chat_signing_in: String,
    chat_open_sign_in: String,
    chat_code_hint: String,
    chat_code_submit: String,
    chat_cancel: String,
    material_no_slots: String,
    material_add_slot: String,
    material_remove_slot: String,
    recover_open: String,
    recover_menu: String,
    recover_none: String,
    recover_discard: String,
    recover_discard_all: String,
    add_mesh: String,
    add_light: String,
    add_camera: String,
    add_empty: String,
    add_armature: String,
    menu_pose: String,
    menu_sculpt: String,
    menu_texture: String,
    menu_sketch: String,
    menu_constrain: String,
    menu_solid: String,
    finish_sketch: String,
    /// Each sketch tool's name, by the name the keys use.
    sketch_tools: Vec<(&'static str, String)>,
    new_texture: String,
    /// Every sculpt brush's name, in `Tool::SCULPT`'s order.
    sculpt_brushes: Vec<String>,
    sculpt_mask_clear: String,
    sculpt_mask_invert: String,
    sculpt_face_set_from_mask: String,
    sculpt_show_all: String,
    sculpt_remesh: String,
    sculpt_quad_remesh: String,
    pose_ik_add: String,
    menu_weights: String,
    group_add: String,
    weights_normalize: String,
    menu_parent: String,
    menu_boolean: String,
    boolean_ops: [String; 3],
    parent_object: String,
    parent_clear: String,
    parent_bone: String,
    robot_rig: String,
    rename: String,
    bind_automatic: String,
    bind_empty: String,
    unbind: String,
    add_humanoid_rig: String,
    menu_armature: String,
    armature_add_bone: String,
    armature_extrude: String,
    armature_delete: String,
    pose_clear_all: String,
    pose_clear_location: String,
    pose_clear_rotation: String,
    pose_clear_scale: String,
    /// One per entry of `ADD_SHAPES`.
    add_shapes: [String; 8],
    /// Each kind of gear the Add menu makes, in `object.add_gear`'s order.
    add_gears: [String; 5],
    /// The Mechanical menu's names, by their catalogue key.
    mech: std::collections::BTreeMap<&'static str, String>,
    shading_studio: String,
    shading_flat: String,
    shading_wireframe: String,
    shading_rendered: String,
    shading_material: String,
    overlay_flat: String,
    shading_hint: String,
    exposure: String,
    exposure_metered: String,
    exposure_manual: String,
    menu_file: String,
    menu_edit: String,
    menu_help: String,
    help_contact: String,
    file_collab: String,
    help_about: String,
    contact_title: String,
    contact_intro: String,
    contact_kind_bug: String,
    contact_kind_feature: String,
    contact_kind_question: String,
    contact_kind_other: String,
    contact_name: String,
    contact_email: String,
    contact_email_hint: String,
    contact_message_hint: String,
    contact_details: String,
    contact_send: String,
    contact_close: String,
    contact_sending: String,
    contact_sent: String,
    about_title: String,
    about_version: String,
    menu_view: String,
    menu_select: String,
    menu_transform: String,
    select_all: String,
    select_none: String,
    select_invert: String,
    view_frame_all: String,
    view_frame_selected: String,
    view_camera: String,
    view_measure: String,
    view_walk: String,
    design_dimensions: String,
    design_clear: String,
    view_perspective: String,
    view_sidebar: String,
    overlays: String,
    overlay_grid: String,
    overlay_wireframe: String,
    overlay_face_orientation: String,
    overlay_textures: String,
    overlay_statistics: String,
    overlay_dimensions: String,
    overlay_ao: String,
    overlay_xray: String,
    overlay_outline: String,
    overlay_bloom: String,
    overlay_reflections: String,
    overlay_contact_shadows: String,
    overlay_path_trace: String,
    transform_move: String,
    transform_rotate: String,
    transform_scale: String,
    view_user_perspective: String,
    view_user_orthographic: String,
    panel_transform: String,
    panel_surface: String,
    panel_light: String,
    menu_object: String,
    pie_snap: String,
    cursor_to_selected: String,
    cursor_to_origin: String,
    selection_to_cursor: String,
    menu_apply: String,
    apply_location: String,
    apply_rotation: String,
    apply_scale: String,
    apply_all: String,
    apply_rotation_scale: String,
    menu_set_origin: String,
    origin_to_geometry: String,
    origin_to_cursor: String,
    geometry_to_origin: String,
    menu_snap: String,
    menu_physics: String,
    physics_active: String,
    physics_passive: String,
    physics_cloth: String,
    physics_soft: String,
    physics_liquid_domain: String,
    physics_liquid: String,
    physics_liquid_inflow: String,
    physics_smoke_domain: String,
    physics_smoke: String,
    physics_fire: String,
    physics_collision: String,
    physics_particles: String,
    physics_bake_menu: String,
    physics_free_menu: String,
    pivot_label: String,
    undo_history: String,
    undo_original: String,
    stats_objects: String,
    stats_vertices: String,
    stats_edges: String,
    stats_faces: String,
    stats_triangles: String,
    stats_length: String,
    sidebar_median: String,
    sidebar_vertex: String,
    stats_area: String,
    pivot_median: String,
    pivot_individual: String,
    pivot_active: String,
    pivot_cursor: String,
    menu_mesh: String,
    menu_edge: String,
    menu_face: String,
    add_menu: String,
    add_light_point: String,
    add_light_sun: String,
    add_light_spot: String,
    add_light_area: String,
    tab_data_light: String,
    tab_data_camera: String,
    tab_world: String,
    tab_render: String,
    menu_render: String,
    render_image: String,
    render_animation: String,
    render_view: String,
    render_result_title: String,
    render_hint: String,
    design_units: String,
    align_faces: String,
    file_export_drawing: String,
    drawing_top: String,
    drawing_front: String,
    drawing_right: String,
    dissolve_limited: String,
    design_section: String,
    world_image: String,
    duplicate: String,
    object_copy: String,
    object_paste: String,
    delete: String,
    undo: String,
    redo: String,
    modifiers: String,
    uv_none: String,
    uv_not_editing: String,
    area_menu: String,
    editor_unavailable: String,
    sidebar_item: String,
    sidebar_tool: String,
    sidebar_view: String,
    sidebar_dimensions: String,
    sidebar_lens: String,
    sidebar_cursor: String,
    sidebar_nothing: String,
    area_split_v: String,
    area_split_h: String,
    area_close: String,
    extrude: String,
    subdivide: String,
    dissolve: String,
    inset: String,
    bridge: String,
    loop_cut: String,
    shade_smooth: String,
    shade_flat: String,
    mark_sharp: String,
    recalc_normals: String,
    fill: String,
    spin: String,
    bevel: String,
    merge: String,
    weld: String,
    select_loop: String,
    select_more: String,
    knife: String,
    mark_seam: String,
    clear_seam: String,
    select_less: String,
    select_linked: String,
    show_hide: String,
    hide_selected: String,
    hide_unselected: String,
    reveal: String,
}

impl Labels {
    fn new(catalog: &Catalog) -> Self {
        let t = |key: &str| catalog.format(key, TextKind::Ui, &[]);
        Self {
            file_new: t("menu-file-new"),
            file_open: t("menu-file-open"),
            file_open_recent: t("menu-file-open-recent"),
            menu_select_similar: t("menu-select-similar"),
            similar: kreate_mesh::selection::SIMILAR
                .iter()
                .map(|(name, _)| t(&format!("similar-{}", name.replace('_', "-"))))
                .collect(),
            recent_none: t("recent-none"),
            menu_show_hide: t("menu-show-hide"),
            reveal_hidden: t("reveal-hidden"),
            file_save: t("menu-file-save"),
            file_save_as: t("menu-file-save-as"),
            file_import_menu: t("menu-file-import-menu"),
            file_export_menu: t("menu-file-export-menu"),
            file_any_format: t("menu-file-any-format"),
            file_run_script: t("menu-file-run-script"),
            menu_search: t("menu-search"),
            menu_copy_script: t("menu-copy-script"),
            palette_placeholder: t("palette-placeholder"),
            pie_shading: t("pie-shading"),
            time_play: t("time-play"),
            key_property: t("key-property"),
            time_autokey: t("time-autokey"),
            sheet_empty: t("sheet-empty"),
            graph_handles: t("graph-handles"),
            handle_auto: t("handle-auto"),
            handle_flat: t("handle-flat"),
            sheet_interpolation: t("sheet-interpolation"),
            interp_constant: t("interp-constant"),
            interp_linear: t("interp-linear"),
            interp_bezier: t("interp-bezier"),
            time_pause: t("time-pause"),
            time_play_menu: t("op-time-play"),
            time_jump_start: t("time-jump-start"),
            time_jump_end: t("time-jump-end"),
            time_start: t("time-start"),
            time_end: t("time-end"),
            menu_animation: t("menu-animation"),
            anim_insert: t("op-anim-keyframe-insert"),
            anim_delete: t("op-anim-keyframe-delete"),
            pie_view: t("pie-view"),
            pie_mode: t("pie-mode"),
            view_left: t("view-left"),
            view_right: t("view-right"),
            view_bottom: t("view-bottom"),
            view_top: t("view-top"),
            view_front: t("view-front"),
            view_back: t("view-back"),
            mode_names: kreate_ops::Mode::ALL
                .iter()
                .map(|m| (m.info().tag, t(m.info().label_key)))
                .collect(),
            palette_nothing: t("palette-nothing"),
            world_pick_image: t("world-pick-image"),
            world_clear_image: t("world-clear-image"),
            material_pick_image: t("material-pick-image"),
            material_clear_image: t("material-clear-image"),
            tab_material: t("tab-material"),
            material_new: t("material-new"),
            material_browse: t("material-browse"),
            material_assign: t("material-assign"),
            material_assign_hint: t("material-assign-hint"),
            material_select: t("material-select"),
            material_select_hint: t("material-select-hint"),
            material_none: t("material-none"),
            material_base_colour: t("material-base-colour"),
            material_metallic_roughness: t("material-metallic-roughness"),
            material_normal_map: t("material-normal-map"),
            material_emission_map: t("material-emission-map"),
            material_occlusion_map: t("material-occlusion-map"),
            material_uses_nodes: t("material-uses-nodes"),
            node_use_nodes: t("node-use-nodes"),
            node_use_nodes_hint: t("node-use-nodes-hint"),
            node_no_graph: t("node-no-graph"),
            node_no_material: t("node-no-material"),
            node_add: t("node-add"),
            node_no_image: t("node-no-image"),
            node_open_image: t("node-open-image"),
            material_empty_slot: t("material-empty-slot"),
            console_placeholder: t("console-placeholder"),
            console_empty: t("console-empty"),
            chat_placeholder: t("chat-placeholder"),
            chat_attach: t("chat-attach"),
            chat_attach_hint: t("chat-attach-hint"),
            chat_remove: t("chat-remove"),
            chat_empty: t("chat-empty"),
            chat_send: t("chat-send"),
            chat_stop: t("chat-stop"),
            chat_new: t("chat-new"),
            chat_history: t("chat-history"),
            chat_export: t("chat-export"),
            help_logs: t("help-logs"),
            chat_working: t("chat-working"),
            chat_checking: t("chat-checking"),
            chat_signed_out: t("chat-signed-out"),
            chat_sign_in: t("chat-sign-in"),
            chat_sign_out: t("chat-sign-out"),
            chat_signing_in: t("chat-signing-in"),
            chat_open_sign_in: t("chat-open-sign-in"),
            chat_code_hint: t("chat-code-hint"),
            chat_code_submit: t("chat-code-submit"),
            chat_cancel: t("chat-cancel"),
            material_no_slots: t("material-no-slots"),
            material_add_slot: t("material-add-slot"),
            material_remove_slot: t("material-remove-slot"),
            recover_open: t("recover-open"),
            recover_menu: t("recover-menu"),
            recover_none: t("recover-none"),
            recover_discard: t("recover-discard"),
            recover_discard_all: t("recover-discard-all"),
            add_mesh: t("menu-add-mesh"),
            add_light: t("menu-add-light"),
            add_camera: t("menu-add-camera"),
            add_empty: t("menu-add-empty"),
            add_armature: t("menu-add-armature"),
            menu_pose: t("menu-pose"),
            menu_sculpt: t("menu-sculpt"),
            menu_texture: t("menu-texture"),
            menu_sketch: t("menu-sketch"),
            menu_constrain: t("menu-constrain"),
            menu_solid: t("menu-solid"),
            finish_sketch: t("sketch-finish"),
            sketch_tools: SKETCH_TOOLS
                .iter()
                .map(|n| (*n, t(&format!("sketch-tool-{n}"))))
                .collect(),
            new_texture: t("op-paint-new-texture"),
            sculpt_brushes: kreate_ui::toolbar::Tool::SCULPT
                .iter()
                .map(|tool| t(tool.label_key()))
                .collect(),
            sculpt_mask_clear: t("sculpt-mask-clear"),
            sculpt_mask_invert: t("sculpt-mask-invert"),
            sculpt_face_set_from_mask: t("sculpt-face-set-from-mask"),
            sculpt_show_all: t("sculpt-show-all"),
            sculpt_remesh: t("sculpt-remesh"),
            sculpt_quad_remesh: t("sculpt-quad-remesh"),
            pose_ik_add: t("op-pose-ik-add"),
            menu_weights: t("menu-weights"),
            group_add: t("op-object-vertex-group-add"),
            weights_normalize: t("op-paint-weights-normalize"),
            menu_parent: t("menu-parent"),
            menu_boolean: t("menu-boolean"),
            boolean_ops: [0, 1, 2].map(|n| t(&format!("op-object-boolean-operation-{n}"))),
            parent_object: t("parent-object"),
            parent_clear: t("parent-clear"),
            parent_bone: t("parent-bone"),
            robot_rig: t("robot-rig"),
            rename: t("outliner-rename"),
            bind_automatic: t("bind-automatic"),
            bind_empty: t("bind-empty"),
            unbind: t("unbind-armature"),
            add_humanoid_rig: t("op-object-add-humanoid-rig"),
            menu_armature: t("menu-armature"),
            armature_add_bone: t("armature-add-bone"),
            armature_extrude: t("armature-extrude"),
            armature_delete: t("armature-delete"),
            pose_clear_all: t("pose-clear-all"),
            pose_clear_location: t("pose-clear-location"),
            pose_clear_rotation: t("pose-clear-rotation"),
            pose_clear_scale: t("pose-clear-scale"),
            add_shapes: [
                t("menu-add-plane"),
                t("menu-add-cube"),
                t("menu-add-grid"),
                t("menu-add-uv-sphere"),
                t("menu-add-ico-sphere"),
                t("menu-add-cylinder"),
                t("menu-add-cone"),
                t("menu-add-torus"),
            ],
            mech: mech_keys().into_iter().map(|k| (k, t(k))).collect(),
            add_gears: [
                t("menu-add-gear"),
                t("menu-add-gear-internal"),
                t("menu-add-gear-rack"),
                t("menu-add-gear-bevel"),
                t("menu-add-gear-worm"),
            ],
            shading_studio: t("shading-studio"),
            shading_flat: t("shading-flat"),
            shading_wireframe: t("shading-wireframe"),
            shading_rendered: t("shading-rendered"),
            shading_material: t("shading-material"),
            overlay_flat: t("overlay-flat"),
            shading_hint: t("shading-hint"),
            exposure: t("exposure"),
            exposure_metered: t("exposure-metered"),
            exposure_manual: t("exposure-manual"),
            menu_file: t("menu-file"),
            menu_edit: t("menu-edit"),
            menu_help: t("menu-help"),
            help_contact: t("help-contact"),
            file_collab: t("menu-file-collab"),
            help_about: t("help-about"),
            contact_title: t("contact-title"),
            contact_intro: t("contact-intro"),
            contact_kind_bug: t("contact-kind-bug"),
            contact_kind_feature: t("contact-kind-feature"),
            contact_kind_question: t("contact-kind-question"),
            contact_kind_other: t("contact-kind-other"),
            contact_name: t("contact-name"),
            contact_email: t("contact-email"),
            contact_email_hint: t("contact-email-hint"),
            contact_message_hint: t("contact-message-hint"),
            contact_details: t("contact-details"),
            contact_send: t("contact-send"),
            contact_close: t("contact-close"),
            contact_sending: t("contact-sending"),
            contact_sent: t("contact-sent"),
            about_title: t("about-title"),
            about_version: t("about-version"),
            menu_view: t("menu-view"),
            menu_select: t("menu-select"),
            menu_transform: t("menu-transform"),
            select_all: t("select-all"),
            select_none: t("select-none"),
            select_invert: t("select-invert"),
            view_frame_all: t("view-frame-all"),
            view_frame_selected: t("view-frame-selected"),
            view_camera: t("view-camera"),
            view_measure: t("view-measure"),
            view_walk: t("view-walk"),
            design_dimensions: t("design-dimensions"),
            design_clear: t("design-clear"),
            view_perspective: t("view-perspective"),
            view_sidebar: t("view-sidebar"),
            overlays: t("overlays"),
            overlay_grid: t("overlay-grid"),
            overlay_wireframe: t("overlay-wireframe"),
            overlay_face_orientation: t("overlay-face-orientation"),
            overlay_textures: t("overlay-textures"),
            overlay_statistics: t("overlay-statistics"),
            overlay_dimensions: t("overlay-dimensions"),
            overlay_ao: t("overlay-ao"),
            overlay_xray: t("overlay-xray"),
            overlay_outline: t("overlay-outline"),
            overlay_bloom: t("overlay-bloom"),
            overlay_reflections: t("overlay-reflections"),
            overlay_contact_shadows: t("overlay-contact-shadows"),
            overlay_path_trace: t("overlay-path-trace"),
            transform_move: t("transform-move"),
            transform_rotate: t("transform-rotate"),
            transform_scale: t("transform-scale"),
            view_user_perspective: t("view-user-perspective"),
            view_user_orthographic: t("view-user-orthographic"),
            panel_transform: t("panel-transform"),
            panel_surface: t("panel-surface"),
            panel_light: t("panel-light"),
            menu_object: t("menu-object"),
            pie_snap: t("pie-snap"),
            cursor_to_selected: t("op-view-cursor-to-selected"),
            cursor_to_origin: t("op-view-cursor-to-origin"),
            selection_to_cursor: t("op-view-selection-to-cursor"),
            menu_apply: t("menu-apply"),
            apply_location: t("apply-location"),
            apply_rotation: t("apply-rotation"),
            apply_scale: t("apply-scale"),
            apply_all: t("apply-all"),
            apply_rotation_scale: t("apply-rotation-scale"),
            menu_set_origin: t("menu-set-origin"),
            origin_to_geometry: t("origin-to-geometry"),
            origin_to_cursor: t("origin-to-cursor"),
            geometry_to_origin: t("geometry-to-origin"),
            menu_snap: t("menu-snap"),
            menu_physics: t("menu-physics"),
            physics_active: t("menu-physics-active"),
            physics_passive: t("menu-physics-passive"),
            physics_cloth: t("menu-physics-cloth"),
            physics_soft: t("menu-physics-soft"),
            physics_liquid_domain: t("menu-physics-liquid-domain"),
            physics_liquid: t("menu-physics-liquid"),
            physics_liquid_inflow: t("menu-physics-liquid-inflow"),
            physics_smoke_domain: t("menu-physics-smoke-domain"),
            physics_smoke: t("menu-physics-smoke"),
            physics_fire: t("menu-physics-fire"),
            physics_collision: t("menu-physics-collision"),
            physics_particles: t("physics-add-emitter"),
            physics_bake_menu: t("menu-physics-bake"),
            physics_free_menu: t("physics-free-bake"),
            pivot_label: t("pivot-point"),
            undo_history: t("menu-undo-history"),
            undo_original: t("undo-original"),
            stats_objects: t("stats-objects"),
            stats_vertices: t("stats-vertices"),
            stats_edges: t("stats-edges"),
            stats_faces: t("stats-faces"),
            stats_triangles: t("stats-triangles"),
            stats_length: t("stats-length"),
            sidebar_median: t("sidebar-median"),
            sidebar_vertex: t("sidebar-vertex"),
            stats_area: t("stats-area"),
            pivot_median: t("pivot-median"),
            pivot_individual: t("pivot-individual"),
            pivot_active: t("pivot-active"),
            pivot_cursor: t("pivot-cursor"),
            menu_mesh: t("menu-mesh"),
            menu_edge: t("menu-edge"),
            menu_face: t("menu-face"),
            add_menu: t("menu-add"),
            add_light_point: t("menu-add-light-point"),
            add_light_sun: t("menu-add-light-sun"),
            add_light_spot: t("menu-add-light-spot"),
            add_light_area: t("menu-add-light-area"),
            tab_data_light: t("tab-data-light"),
            tab_data_camera: t("tab-data-camera"),
            tab_world: t("tab-world"),
            tab_render: t("tab-render"),
            menu_render: t("menu-render"),
            render_image: t("render-image"),
            render_animation: t("render-animation"),
            render_view: t("render-view"),
            render_result_title: t("render-result-title"),
            render_hint: t("render-hint"),
            design_units: t("design-units"),
            align_faces: t("menu-align-faces"),
            file_export_drawing: t("menu-export-drawing"),
            drawing_top: t("drawing-top"),
            drawing_front: t("drawing-front"),
            drawing_right: t("drawing-right"),
            dissolve_limited: t("menu-dissolve-limited"),
            design_section: t("design-section"),
            world_image: t("world-image"),
            duplicate: t("menu-duplicate"),
            object_copy: t("object-copy"),
            object_paste: t("object-paste"),
            delete: t("menu-delete"),
            undo: t("menu-undo"),
            redo: t("menu-redo"),
            modifiers: t("panel-modifiers"),
            uv_none: t("uv-none"),
            uv_not_editing: t("uv-not-editing"),
            area_menu: t("area-menu"),
            editor_unavailable: t("editor-unavailable"),
            sidebar_item: t("sidebar-item"),
            sidebar_tool: t("sidebar-tool"),
            sidebar_view: t("sidebar-view"),
            sidebar_dimensions: t("sidebar-dimensions"),
            sidebar_lens: t("sidebar-lens"),
            sidebar_cursor: t("sidebar-cursor"),
            sidebar_nothing: t("sidebar-nothing"),
            area_split_v: t("area-split-vertical"),
            area_split_h: t("area-split-horizontal"),
            area_close: t("area-close"),
            extrude: t("menu-extrude"),
            subdivide: t("menu-subdivide"),
            dissolve: t("menu-dissolve"),
            inset: t("menu-inset"),
            bridge: t("menu-bridge"),
            loop_cut: t("menu-loop-cut"),
            shade_smooth: t("menu-shade-smooth"),
            shade_flat: t("menu-shade-flat"),
            mark_sharp: t("menu-mark-sharp"),
            recalc_normals: t("menu-recalc-normals"),
            fill: t("menu-fill"),
            spin: t("menu-spin"),
            bevel: t("menu-bevel"),
            merge: t("menu-merge"),
            weld: t("menu-weld"),
            select_loop: t("menu-select-loop"),
            select_more: t("menu-select-more"),
            knife: t("op-mesh-knife"),
            mark_seam: t("menu-mark-seam"),
            clear_seam: t("menu-clear-seam"),
            select_less: t("menu-select-less"),
            select_linked: t("menu-select-linked"),
            show_hide: t("menu-show-hide"),
            hide_selected: t("menu-hide-selected"),
            hide_unselected: t("menu-hide-unselected"),
            reveal: t("menu-reveal"),
        }
    }
}

/// Draws the whole shell and returns the rectangle left for the viewport.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    history: &History,
    keymap: &kreate_input::Keymap,
    preset: kreate_input::Preset,
    overrides: &std::collections::BTreeMap<kreate_input::BindingId, kreate_input::Override>,
    mode: kreate_ops::operator::Mode,
    viewport_texture: Option<ViewportImage>,
) -> egui::Rect {
    // Resolved once per frame into owned strings. A closure over the catalog
    // would hold a borrow of `state` for the whole function, which conflicts with
    // the mutable access the panels need to the command queue.
    let l = Labels::new(&state.catalog);
    // Refilled by whatever draws over the viewport this frame. Cleared here
    // rather than by each drawer, so an overlay that stops being drawn stops
    // blocking the pointer on the very next frame.
    state.viewport_blockers.clear();
    // A gesture lasts as long as the pointer is down. Ended here, at the top of
    // the frame, so that a panel changing a value knows whether it is
    // continuing one — see `send_edit`.
    end_gesture(state, ui.ctx().input(|i| i.pointer.any_down()));
    let status = state.catalog.format(
        "status-counts",
        TextKind::Ui,
        &[
            ("objects", &doc.object_count().to_string()),
            ("meshes", &doc.mesh_count().to_string()),
            ("selected", &doc.selection.objects.len().to_string()),
        ],
    );
    // Rebuilt only when the document changes, and the rows themselves only for
    // what is on screen — see `OutlinerOrder`.
    state.outliner.sync(doc);
    let selected: Vec<u32> = doc
        .selection
        .objects
        .iter()
        .map(|id| object_key(*id))
        .collect();
    let active = doc.selection.active.map(object_key);
    // Built here rather than inside the panel: the panel edits the parameters in
    // place, and a closure over `doc` would hold a borrow across the mutable
    // access the command queue needs.
    let mut modifier_rows = modifier_rows(doc, registry, &state.catalog);

    // **The top bar is the application's, not the viewport's.** Blender's
    // rule, and the reason this row got short: what you can *do to a scene*
    // — add, select, transform, change mode, change shading — belongs to the
    // editor showing the scene, and moves with it when the layout changes. What
    // is left up here is what is true of the whole window: the file, undo, and
    // which workspace is open. A viewport in the UV workspace has the same
    // Object menu as the one in Layout because the menu is *its own*.
    Panel::top("menu").show(ui, |ui| {
        ui.horizontal(|ui| {
            // File and Edit, from the same tree F3 searches. Undo and redo
            // live under Edit rather than as buttons on the row, which is
            // both Blender's arrangement and the honest one: they are reached
            // by keystroke a hundred times for every once by pointer.
            // The steps, for Edit ▸ Undo History.
            state.history_steps = history.entries().iter().map(|e| e.label.clone()).collect();
            state.history_at = history.cursor();
            for menu in top_bar_menus(&l, history.can_undo(), history.can_redo()) {
                ui.menu_button(&menu.label, |ui| draw_nodes(ui, state, &l, &menu.nodes));
            }

            ui.separator();
            draw_app_bar(ui, state, &l, history.can_undo(), history.can_redo());
        });
    });
    // The toolbar under it: the workspace, the mode tabs, the mode's tools.
    draw_ribbon(ui, state, doc, &l, mode);

    Panel::bottom("status").show(ui, |ui| {
        ui.horizontal(|ui| {
            // Edit mode wears the accent, which is the same thing the accent
            // says everywhere else: this is what you are acting on. Object
            // mode is the resting state and says it quietly.
            let theme = kreate_ui::theme::from_ui(ui);
            let name = state.catalog.format(mode.label_key(), TextKind::Ui, &[]);
            // The mode in its own tint, as the viewport header names it;
            // object mode, the resting state, plainly. Not the accent: that
            // means "selected", and a mode is not a selection.
            match theme.mode_tint(mode.tag()) {
                Some((_, text)) => {
                    ui.colored_label(text, name);
                }
                None => {
                    ui.label(name);
                }
            }
            // The mode's everyday keys, as Blender's status bar keeps them in
            // view: what the hand does next, without opening a menu to learn
            // it. Read from the keymap, so a rebinding shows here too.
            let hints: &[&str] = match mode.acts_on() {
                // The sketch tools' keys are on the toolbar's buttons.
                kreate_ops::mode::ActsOn::Sketch => &[],
                kreate_ops::mode::ActsOn::Objects => &[
                    "object.translate",
                    "object.rotate",
                    "object.scale",
                    "anim.keyframe_insert",
                ],
                kreate_ops::mode::ActsOn::MeshElements => &[
                    "mesh.translate",
                    "mesh.rotate",
                    "mesh.scale",
                    "mesh.extrude_region",
                ],
                kreate_ops::mode::ActsOn::Bones => &[
                    "pose.translate",
                    "pose.rotate",
                    "pose.scale",
                    "anim.keyframe_insert",
                ],
                kreate_ops::mode::ActsOn::BoneJoints => &[
                    "armature.translate",
                    "armature.extrude",
                    "armature.bone_add",
                    "armature.delete",
                ],
                // The brush is the mouse; there are no keys to learn yet.
                kreate_ops::mode::ActsOn::Weights
                | kreate_ops::mode::ActsOn::Sculpt
                | kreate_ops::mode::ActsOn::Texture => &[],
            };
            for id in hints {
                if let Some(info) = state
                    .command_info
                    .get(*id)
                    .filter(|i| !i.shortcut.is_empty())
                {
                    ui.add_space(6.0);
                    // A key cap: the shortcut on a raised chip, in the
                    // monospaced face, and what it does beside it.
                    egui::Frame::new()
                        .fill(theme.raised)
                        .stroke(egui::Stroke::new(1.0, theme.border))
                        .corner_radius(4)
                        .inner_margin(egui::Margin::symmetric(5, 0))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(&info.shortcut)
                                    .monospace()
                                    .small()
                                    .color(theme.text),
                            );
                        });
                    ui.weak(egui::RichText::new(&info.name).small());
                }
            }
            // Recovered work waiting under File ▸ Recover. Said here, quietly
            // and for as long as it waits, because a menu nobody opens is a
            // menu nobody finds — and the person this is for has just lost a
            // session.
            if !state.recoverable.is_empty() {
                ui.separator();
                let hint = state.catalog.format(
                    "recover-hint",
                    TextKind::Ui,
                    &[("count", &state.recoverable.len().to_string())],
                );
                ui.colored_label(theme.info, hint);
            }
            // The last thing that happened.
            // A save that reports nothing is a save the user cannot distinguish
            // from one that silently failed.
            if let Some(message) = &state.status {
                ui.separator();
                // i18n-exempt: composed from a translated template and a path
                ui.label(message);
            }
            if let Some(entry) = history.last() {
                ui.separator();
                // i18n-exempt: a generated script line is API text, not prose
                ui.monospace(entry.to_script_preview(SCRIPT_PREVIEW));
            }
            // The standing counts on the right, where Blender keeps its
            // statistics: what is true of the scene, apart from what just
            // happened to it.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(&status);
            });
        });
    });

    draw_history_strip(ui, state, &l);

    // The screen. Boundaries are interacted with *before* the areas, so a drag
    // that starts on one is not stolen by whatever is under it — a handle is
    // four points wide and always overlaps an editor's edge.
    let mut viewport_rect = egui::Rect::ZERO;
    // The gaps between the areas in the window's own colour, not whatever
    // the frame was cleared to: black seams on the light chrome.
    let gaps = kreate_ui::theme::from_ui(ui).sunken;
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(gaps))
        .show(ui, |ui| {
            let screen = ui.available_rect_before_wrap();
            let layout = state.screen().layout(screen);
            drag_handles(ui, state, &layout);

            // Only the first viewport area gets the rendered image. Several
            // viewports means several cameras and several render targets, which
            // is a real feature and a later one; until then the honest thing is
            // for the extra ones to say nothing rather than to show a stale
            // copy of the first one's picture.
            let first_viewport = layout
                .areas
                .iter()
                .find(|a| a.editor == kreate_ui::EditorId::VIEWPORT)
                .map(|a| a.id);

            let mut texture = viewport_texture;
            for area in &layout.areas {
                let image = (Some(area.id) == first_viewport)
                    .then(|| texture.take())
                    .flatten();
                let rect = draw_area(
                    ui,
                    state,
                    area,
                    &l,
                    registry,
                    history,
                    keymap,
                    preset,
                    overrides,
                    doc,
                    &selected,
                    active,
                    &mut modifier_rows,
                    image,
                    mode,
                );
                if Some(area.id) == first_viewport {
                    viewport_rect = rect;
                }
            }
        });
    state.apply_area_edits();
    draw_help_windows(ui.ctx(), state, &l);
    draw_search(ui, state, &l, history, mode);
    draw_pie(ui, state, &l, doc, mode);
    viewport_rect
}

/// A pie's entries and title, by the name its key binding gives.
///
/// Built from menu entries like every other menu, so what a pie runs is what
/// the same entry runs from a menu or the search. Listed in the pie's slot
/// order: left, right, bottom, top, then the diagonals.
fn pie_entries(
    name: &str,
    l: &Labels,
    shading: kreate_render::ShadingMode,
    mode: kreate_ops::operator::Mode,
    has: kreate_ops::mode::Has,
) -> Option<(String, Vec<(MenuEntry, bool)>)> {
    use kreate_render::{ShadingMode, ViewAxis};
    let shade = |id: &'static str, label: &str, to: ShadingMode| {
        (
            MenuEntry::new(id, label, Run::Command(Command::SetShading(to))),
            shading == to,
        )
    };
    let axis = |id: &'static str, label: &str, to: ViewAxis| {
        (
            MenuEntry::new(id, label, Run::Command(Command::SnapToAxis(to))),
            false,
        )
    };
    Some(match name {
        // Blender's Z: wireframe left, rendered right, solid below, the
        // material preview above.
        "shading" => (
            l.pie_shading.clone(),
            vec![
                shade(
                    "view.shading_wireframe",
                    &l.shading_wireframe,
                    ShadingMode::Wireframe,
                ),
                shade(
                    "view.shading_rendered",
                    &l.shading_rendered,
                    ShadingMode::Rendered,
                ),
                shade(
                    "view.shading_studio",
                    &l.shading_studio,
                    ShadingMode::Studio,
                ),
                shade(
                    "view.shading_material",
                    &l.shading_material,
                    ShadingMode::Material,
                ),
            ],
        ),
        // Blender's `~`. Each view is named for where the eye goes: Front
        // looks from -Y, as the numpad's 1 does.
        "view" => (
            l.pie_view.clone(),
            vec![
                axis("view.axis_left", &l.view_left, ViewAxis::NegX),
                axis("view.axis_right", &l.view_right, ViewAxis::PosX),
                axis("view.axis_bottom", &l.view_bottom, ViewAxis::NegZ),
                axis("view.axis_top", &l.view_top, ViewAxis::PosZ),
                axis("view.axis_front", &l.view_front, ViewAxis::NegY),
                axis("view.axis_back", &l.view_back, ViewAxis::PosY),
                // Blender's pie: View Camera and View Selected on the last
                // two slots. Frame All is Home and the View menu.
                (
                    MenuEntry::new("view.camera", &l.view_camera, dispatch("view.camera")),
                    false,
                ),
                (
                    MenuEntry::new(
                        "view.frame_selected",
                        &l.view_frame_selected,
                        Run::Command(Command::FrameSelection),
                    ),
                    false,
                ),
            ],
        ),
        // Blender's Ctrl+Tab: every mode, the one you are in lit, the ones
        // that cannot be entered greyed for the reason the mode menu greys
        // them. The id is the mode's own label key, so its description sits
        // beside its name in the catalog.
        "mode" => (
            l.pie_mode.clone(),
            // This object's modes, greyed where they cannot be entered yet;
            // see `Mode::offered`.
            kreate_ops::Mode::ALL
                .iter()
                .filter(|m| **m == mode || m.offered(has))
                .map(|m| {
                    let info = m.info();
                    let label = l.mode_names.get(info.tag).cloned().unwrap_or_default();
                    (
                        MenuEntry::new(
                            info.label_key,
                            &label,
                            Run::Command(Command::SetMode(info.tag.to_string())),
                        )
                        .enabled(m.can_enter(has)),
                        *m == mode,
                    )
                })
                .collect(),
        ),
        // Blender's Shift+S, the three a modeller reaches for.
        "snap" => (
            l.pie_snap.clone(),
            vec![
                (
                    MenuEntry::new(
                        "view.cursor_to_selected",
                        &l.cursor_to_selected,
                        dispatch("view.cursor_to_selected"),
                    ),
                    false,
                ),
                (
                    MenuEntry::new(
                        "view.selection_to_cursor",
                        &l.selection_to_cursor,
                        dispatch("view.selection_to_cursor"),
                    ),
                    false,
                ),
                (
                    MenuEntry::new(
                        "view.cursor_to_origin",
                        &l.cursor_to_origin,
                        dispatch("view.cursor_to_origin"),
                    ),
                    false,
                ),
            ],
        ),
        _ => return None,
    })
}

/// The object menu's Apply, Set Origin and Snap submenus — Blender's, in the
/// Object menu where it keeps them.
fn transform_nodes(l: &Labels) -> Vec<MenuNode> {
    use kreate_reflect::PropertyValue as V;
    let apply = |label: &str, location: bool, rotation: bool, scale: bool| {
        MenuNode::Entry(
            MenuEntry::new(
                "object.transform_apply",
                label,
                op(
                    "object.transform_apply",
                    &[
                        ("location", V::Bool(location)),
                        ("rotation", V::Bool(rotation)),
                        ("scale", V::Bool(scale)),
                    ],
                ),
            )
            .variant("location", V::Bool(location))
            .variant("rotation", V::Bool(rotation))
            .variant("scale", V::Bool(scale)),
        )
    };
    let origin = |label: &str, kind: u32| {
        MenuNode::Entry(
            MenuEntry::new(
                "object.origin_set",
                label,
                op("object.origin_set", &[("kind", V::Enum(kind))]),
            )
            .variant("kind", V::Enum(kind)),
        )
    };
    vec![
        MenuNode::Sub(
            l.menu_apply.clone(),
            vec![
                apply(&l.apply_location, true, false, false),
                apply(&l.apply_rotation, false, true, false),
                apply(&l.apply_scale, false, false, true),
                apply(&l.apply_all, true, true, true),
                apply(&l.apply_rotation_scale, false, true, true),
            ],
        ),
        MenuNode::Sub(
            l.menu_set_origin.clone(),
            vec![
                origin(&l.origin_to_geometry, 0),
                origin(&l.origin_to_cursor, 1),
                origin(&l.geometry_to_origin, 2),
            ],
        ),
        MenuNode::Sub(
            l.menu_snap.clone(),
            vec![
                entry(
                    "view.cursor_to_selected",
                    &l.cursor_to_selected,
                    dispatch("view.cursor_to_selected"),
                ),
                entry(
                    "view.selection_to_cursor",
                    &l.selection_to_cursor,
                    dispatch("view.selection_to_cursor"),
                ),
                entry(
                    "view.cursor_to_origin",
                    &l.cursor_to_origin,
                    dispatch("view.cursor_to_origin"),
                ),
            ],
        ),
    ]
}

/// The Object menu's Physics: Blender's quick ways to make something fall,
/// drape or be in the way, and the bake. The rest is in the Physics tab.
fn physics_node(l: &Labels) -> MenuNode {
    use kreate_reflect::PropertyValue as V;
    let rigid = |label: &str, kind: u32| {
        MenuNode::Entry(
            MenuEntry::new(
                "physics.rigid_body_set",
                label,
                op(
                    "physics.rigid_body_set",
                    &[("enabled", V::Bool(true)), ("kind", V::Enum(kind))],
                ),
            )
            .variant("kind", V::Enum(kind)),
        )
    };
    let soft = |label: &str, kind: u32| {
        MenuNode::Entry(
            MenuEntry::new(
                "physics.soft_set",
                label,
                op("physics.soft_set", &[("kind", V::Enum(kind))]),
            )
            .variant("kind", V::Enum(kind)),
        )
    };
    // A liquid's domain, the liquid itself, or a tap: Blender's Quick Liquid
    // made of its parts, since the parts are what the user arranges.
    // role, domain kind, substance, flow kind
    let liquid = |label: &str, [role, domain, substance, flow]: [u32; 4]| {
        let given = [
            ("role", V::Enum(role)),
            ("domain", V::Enum(domain)),
            ("substance", V::Enum(substance)),
            ("flow", V::Enum(flow)),
        ];
        let mut entry = MenuEntry::new("physics.fluid_set", label, op("physics.fluid_set", &given));
        for (name, value) in given {
            entry = entry.variant(name, value);
        }
        MenuNode::Entry(entry)
    };
    MenuNode::Sub(
        l.menu_physics.clone(),
        vec![
            rigid(&l.physics_active, 0),
            rigid(&l.physics_passive, 1),
            soft(&l.physics_cloth, 1),
            soft(&l.physics_soft, 2),
            liquid(&l.physics_liquid_domain, [1, 0, 0, 0]),
            liquid(&l.physics_liquid, [2, 0, 0, 0]),
            liquid(&l.physics_liquid_inflow, [2, 0, 0, 1]),
            liquid(&l.physics_smoke_domain, [1, 1, 0, 0]),
            liquid(&l.physics_smoke, [2, 0, 1, 1]),
            liquid(&l.physics_fire, [2, 0, 3, 1]),
            MenuNode::Entry(MenuEntry::new(
                "physics.collision_set",
                &l.physics_collision,
                op("physics.collision_set", &[("enabled", V::Bool(true))]),
            )),
            MenuNode::Entry(MenuEntry::new(
                "physics.particles_add",
                &l.physics_particles,
                op("physics.particles_add", &[]),
            )),
            MenuNode::Separator,
            MenuNode::Entry(MenuEntry::new(
                "physics.bake_start",
                &l.physics_bake_menu,
                dispatch("physics.bake_start"),
            )),
            MenuNode::Entry(MenuEntry::new(
                "physics.free_bake",
                &l.physics_free_menu,
                op("physics.free_bake", &[]),
            )),
        ],
    )
}

/// Draws the open pie, and runs what it chooses.
fn draw_pie(
    ui: &mut Ui,
    state: &mut ShellState,
    l: &Labels,
    doc: &Document,
    mode: kreate_ops::operator::Mode,
) {
    let Some(open) = state.pie.open.clone() else {
        return;
    };
    let has = kreate_ops::mode::Has::of(doc);
    let Some((title, entries)) = pie_entries(&open.name, l, state.view_status.shading, mode, has)
    else {
        // A name nothing defines — a binding from a newer keymap. Closed
        // rather than drawn empty.
        state.pie.open = None;
        return;
    };
    let items: Vec<kreate_ui::pie_menu::PieItem> = entries
        .iter()
        .map(|(e, current)| kreate_ui::pie_menu::PieItem {
            label: e.label.clone(),
            enabled: e.enabled,
            current: *current,
        })
        .collect();
    let ctx = ui.ctx().clone();
    if let Some(index) = kreate_ui::pie_menu::pie_menu(&ctx, &mut state.pie, &title, &items) {
        let entry = entries[index].0.clone();
        run_entry(state, &entry);
    }
}

/// F3's search, over the menus this mode offers.
///
/// Built from the same tree the menus draw, and run through the same
/// `run_entry`, so choosing Extrude here is choosing it from Face — with the
/// menu's parameters, not a guess at them.
fn draw_search(
    ui: &mut Ui,
    state: &mut ShellState,
    l: &Labels,
    history: &History,
    mode: kreate_ops::operator::Mode,
) {
    if !state.palette.open {
        return;
    }
    let entries = flatten(&searchable_menus(
        &state.catalog,
        mode,
        state.select_mode,
        history.can_undo(),
        history.can_redo(),
    ));
    let items: Vec<kreate_ui::command_palette::PaletteItem> = entries
        .iter()
        .map(|(path, e)| {
            let mut map = PropertyMap::new();
            for (name, value) in &e.variant {
                map.set(name, value.clone());
            }
            let variant = kreate_ui::command_info::variant_id(e.id, &map);
            let detail = [variant.as_str(), e.id]
                .iter()
                .map(|key| kreate_ui::command_info::description_key(key))
                .find(|key| state.catalog.has(key))
                .map(|key| state.catalog.format(&key, TextKind::Tooltip, &[]))
                .unwrap_or_default();
            let shortcut = state
                .command_info
                .get(&variant)
                .or_else(|| state.command_info.get(e.id))
                .map(|info| info.shortcut.clone())
                .unwrap_or_default();
            kreate_ui::command_palette::PaletteItem {
                key: format!("{path}/{variant}"),
                title: e.label.clone(),
                path: path.clone(),
                detail,
                shortcut,
                keywords: variant.replace(['.', '_'], " "),
                enabled: e.enabled,
            }
        })
        .collect();
    let text = kreate_ui::command_palette::PaletteText {
        placeholder: l.palette_placeholder.clone(),
        nothing: l.palette_nothing.clone(),
    };
    let ctx = ui.ctx().clone();
    if let Some(index) =
        kreate_ui::command_palette::command_palette(&ctx, &mut state.palette, &items, &text)
    {
        let entry = entries[index].1.clone();
        run_entry(state, &entry);
    }
}

/// One area: a header saying what it is, and the editor below it.
#[allow(clippy::too_many_arguments)]
fn draw_area(
    ui: &mut Ui,
    state: &mut ShellState,
    area: &kreate_ui::AreaRect,
    l: &Labels,
    registry: &OperatorRegistry,
    history: &History,
    keymap: &kreate_input::Keymap,
    preset: kreate_input::Preset,
    overrides: &std::collections::BTreeMap<kreate_input::BindingId, kreate_input::Override>,
    doc: &Document,
    selected: &[u32],
    active: Option<u32>,
    modifier_rows: &mut [kreate_ui::ModifierRow],
    image: Option<ViewportImage>,
    mode: kreate_ops::operator::Mode,
) -> egui::Rect {
    let theme = kreate_ui::theme::from_ui(ui);

    // An area paints its own chrome. `Panel` used to do this, and dropping it
    // for a plain sub-`Ui` left every area transparent — which composited as
    // pure black, because nothing else paints there either. The pixel test
    // caught it at (1002, 240).
    let header_height = theme.row_height + 6.0;
    let header = egui::Rect::from_min_size(
        area.rect.min,
        egui::vec2(area.rect.width(), header_height.min(area.rect.height())),
    );
    let body = egui::Rect::from_min_max(egui::pos2(area.rect.min.x, header.max.y), area.rect.max);
    ui.painter().rect_filled(header, 0.0, theme.raised);
    ui.painter().rect_filled(body, 0.0, theme.base);
    ui.painter().rect_stroke(
        area.rect,
        0.0,
        egui::Stroke::new(1.0, theme.border),
        egui::StrokeKind::Inside,
    );

    // The header, in its own strip. Inset so the combo box does not sit on the
    // border.
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(header.shrink(2.0))
            // Salted with the area's id, so two editors of the same kind do not
            // share widget state — two outliners scrolling together would be
            // the visible symptom.
            .id_salt(("area-header", area.id.0)),
        |ui| {
            ui.set_clip_rect(header);
            draw_area_header(ui, state, area, l, mode);
        },
    );

    // The viewport takes its body whole: the rendered image is meant to reach
    // the edges, and padding around it would be a grey frame the user did not
    // ask for. Everything else gets the panel inset.
    //
    // Asked of the registry rather than compared against one id, so an editor
    // that wants the same treatment says so where it is registered instead of
    // here.
    let inner = if state.editors.full_bleed(&area.editor) {
        body
    } else {
        body.shrink(theme.panel_padding)
    };
    if inner.width() <= 0.0 || inner.height() <= 0.0 {
        // An area squeezed to nothing by a small window. Drawing into an
        // inverted rect is not an error egui reports, it is one it renders.
        return body;
    }
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(inner)
            .id_salt(("area-body", area.id.0)),
        |ui| {
            ui.set_clip_rect(inner);
            match area.editor.as_str() {
                "viewport" => {
                    let sidebar = state.sidebar_open(area.id);
                    draw_viewport(ui, state, doc, mode, l, inner, image, sidebar);
                    // The brush's reach, as a ring under the pointer.
                    let brush = match mode.acts_on() {
                        kreate_ops::mode::ActsOn::Weights => Some(state.paint.radius),
                        kreate_ops::mode::ActsOn::Sculpt => Some(state.sculpt.radius),
                        kreate_ops::mode::ActsOn::Texture => Some(state.texture_paint.radius),
                        _ => None,
                    };
                    // Being sized: the ring stands still, and for strength a
                    // second ring inside it is the strength.
                    if let (Some(radius), Some(adjust)) = (brush, state.brush_adjust) {
                        let theme = kreate_ui::theme::from_ui(ui);
                        let painter = ui.painter().with_clip_rect(inner);
                        painter.circle_stroke(
                            adjust.center,
                            radius,
                            egui::Stroke::new(1.2, theme.text),
                        );
                        if adjust.strength {
                            let strength = match mode.acts_on() {
                                kreate_ops::mode::ActsOn::Weights => state.paint.strength,
                                kreate_ops::mode::ActsOn::Texture => state.texture_paint.strength,
                                _ => state.sculpt.strength,
                            };
                            painter.circle_filled(
                                adjust.center,
                                radius * strength,
                                theme.accent.gamma_multiply(0.35),
                            );
                            painter.text(
                                adjust.center,
                                egui::Align2::CENTER_CENTER,
                                format!("{strength:.2}"),
                                egui::FontId::proportional(13.0),
                                theme.text,
                            );
                        } else {
                            painter.text(
                                adjust.center,
                                egui::Align2::CENTER_CENTER,
                                // i18n-exempt: a number and its unit
                                format!("{radius:.0} px"),
                                egui::FontId::proportional(13.0),
                                theme.text,
                            );
                        }
                    } else if let Some(radius) = brush {
                        if let Some(p) = ui.ctx().pointer_hover_pos().filter(|p| inner.contains(*p))
                        {
                            let theme = kreate_ui::theme::from_ui(ui);
                            ui.painter().with_clip_rect(inner).circle_stroke(
                                p,
                                radius,
                                egui::Stroke::new(1.2, theme.text),
                            );
                        }
                    }
                    // The 3D cursor: Blender's red and white ring with a cross.
                    if let Some(at) = state
                        .projector
                        .and_then(|p| p.project(doc.cursor))
                        .filter(|p| inner.contains(*p))
                    {
                        let painter = ui.painter().with_clip_rect(inner);
                        let r = 9.0;
                        for i in 0..8 {
                            let a0 = i as f32 * std::f32::consts::TAU / 8.0;
                            let a1 = a0 + std::f32::consts::TAU / 8.0;
                            let colour = if i % 2 == 0 {
                                egui::Color32::from_rgb(230, 60, 60)
                            } else {
                                egui::Color32::WHITE
                            };
                            let arc: Vec<egui::Pos2> = (0..=6)
                                .map(|k| {
                                    let t = a0 + (a1 - a0) * k as f32 / 6.0;
                                    at + egui::vec2(t.cos(), t.sin()) * r
                                })
                                .collect();
                            painter.add(egui::Shape::line(arc, egui::Stroke::new(1.5, colour)));
                        }
                        let cross = egui::Stroke::new(1.0, egui::Color32::from_gray(20));
                        for d in [egui::vec2(1.0, 0.0), egui::vec2(0.0, 1.0)] {
                            painter.line_segment([at + d * (r - 4.0), at + d * (r + 5.0)], cross);
                            painter.line_segment([at - d * (r - 4.0), at - d * (r + 5.0)], cross);
                        }
                    }
                    // Circle select's brush, under the pointer.
                    if let Some(radius) = state.circle_select {
                        if let Some(p) = ui.ctx().pointer_hover_pos().filter(|p| inner.contains(*p))
                        {
                            let theme = kreate_ui::theme::from_ui(ui);
                            ui.painter().with_clip_rect(inner).circle_stroke(
                                p,
                                radius,
                                egui::Stroke::new(1.2, theme.text),
                            );
                        }
                    }
                    // What a snapping drag caught, Blender's marker on it: a
                    // square for a vertex, a ring for an edge, a ring with a
                    // dot for a face.
                    if let Some((at, element)) = state.snap_target {
                        if let Some(p) = state
                            .projector
                            .and_then(|pr| pr.project(at))
                            .filter(|p| inner.contains(*p))
                        {
                            let painter = ui.painter().with_clip_rect(inner);
                            let colour = egui::Color32::WHITE;
                            let outline =
                                egui::Stroke::new(3.5, egui::Color32::from_black_alpha(160));
                            let stroke = egui::Stroke::new(1.5, colour);
                            match element {
                                crate::snap::Element::Vertex => {
                                    let r = egui::Rect::from_center_size(p, egui::vec2(11.0, 11.0));
                                    painter.rect_stroke(r, 0.0, outline, egui::StrokeKind::Middle);
                                    painter.rect_stroke(r, 0.0, stroke, egui::StrokeKind::Middle);
                                }
                                crate::snap::Element::Edge | crate::snap::Element::Face => {
                                    painter.circle_stroke(p, 6.0, outline);
                                    painter.circle_stroke(p, 6.0, stroke);
                                    if element == crate::snap::Element::Face {
                                        painter.circle_filled(p, 2.0, colour);
                                    }
                                }
                                // A triangle on an edge's middle, as Blender
                                // marks it.
                                crate::snap::Element::Midpoint => {
                                    let corners = vec![
                                        p + egui::vec2(0.0, -7.0),
                                        p + egui::vec2(6.5, 4.5),
                                        p + egui::vec2(-6.5, 4.5),
                                    ];
                                    painter.add(egui::Shape::closed_line(corners.clone(), outline));
                                    painter.add(egui::Shape::closed_line(corners, stroke));
                                }
                                // A cross in a ring on a face's centre.
                                crate::snap::Element::Centre => {
                                    painter.circle_stroke(p, 7.0, outline);
                                    painter.circle_stroke(p, 7.0, stroke);
                                    for (a, b) in [
                                        (egui::vec2(-4.0, 0.0), egui::vec2(4.0, 0.0)),
                                        (egui::vec2(0.0, -4.0), egui::vec2(0.0, 4.0)),
                                    ] {
                                        painter.line_segment([p + a, p + b], outline);
                                        painter.line_segment([p + a, p + b], stroke);
                                    }
                                }
                            }
                        }
                    }
                    // The dimensions on the scene, and the one being taken.
                    if let Some(projector) = state.projector {
                        let painter = ui.painter().with_clip_rect(inner);
                        if state.show_dimensions {
                            for m in &doc.design.measurements {
                                draw_measurement(&painter, &projector, &m.points, false);
                            }
                            // A sketch's dimensions, where its points are now.
                            draw_sketch_dimensions(&painter, &projector, doc);
                        }
                        draw_measurement(&painter, &projector, &state.measure_preview, true);
                        // The work plane, while the Cursor orientation is
                        // chosen: a grid on the cursor, square to it.
                        if state.tools.orientation == 2 {
                            draw_work_plane(&painter, &projector, doc.cursor, doc.cursor_rotation);
                        }
                    }
                    // The knife's path: the cut so far and the next stretch.
                    if let Some(projector) =
                        state.projector.filter(|_| !state.knife_preview.is_empty())
                    {
                        let painter = ui.painter().with_clip_rect(inner);
                        let green = egui::Color32::from_rgb(90, 230, 120);
                        let points: Vec<egui::Pos2> = state
                            .knife_preview
                            .iter()
                            .filter_map(|p| projector.project(*p))
                            .collect();
                        if points.len() == state.knife_preview.len() {
                            painter.add(egui::Shape::line(
                                points.clone(),
                                egui::Stroke::new(2.0, green),
                            ));
                            for p in &points {
                                painter.circle_filled(*p, 3.5, green);
                            }
                        }
                    }
                    // The sketches: every other one thin, the one being drawn
                    // with its points, constraints and dimensions.
                    if let Some(projector) = state.projector {
                        let painter = ui.painter().with_clip_rect(inner);
                        draw_sketches(ui, &painter, &projector, state, inner);
                    }
                    // The loops `Ctrl+R` would cut, in Blender's yellow.
                    if let Some(projector) = state
                        .projector
                        .filter(|_| !state.loop_cut_preview.is_empty())
                    {
                        let painter = ui.painter().with_clip_rect(inner);
                        let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 214, 51));
                        for line in &state.loop_cut_preview {
                            let points: Vec<egui::Pos2> =
                                line.iter().filter_map(|p| projector.project(*p)).collect();
                            if points.len() == line.len() {
                                painter.add(egui::Shape::line(points, stroke));
                            }
                        }
                    }
                    draw_adjust_float(ui, state, registry, history, inner);
                    if state.sidebar_open(area.id) {
                        draw_item_sidebar(ui, state, doc, registry, area.id, inner, l, mode);
                    }
                }
                "outliner" => {
                    draw_feature_list(ui, state);
                    let order = std::mem::take(&mut state.outliner);
                    // A box to find an object by name: what a scene of
                    // hundreds of parts needs before anything else.
                    // Right to left, so the box takes what the clear button
                    // leaves and the row never grows past the panel — a
                    // wider row widens the tree under it and cuts off its
                    // eyes.
                    let row = egui::vec2(ui.available_width(), ui.spacing().interact_size.y);
                    ui.allocate_ui_with_layout(
                        row,
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            let hint = state.catalog.get("outliner-search");
                            // i18n-exempt: a close mark
                            if !state.outliner_search.is_empty() && ui.small_button("x").clicked() {
                                state.outliner_search.clear();
                            }
                            ui.add(
                                egui::TextEdit::singleline(&mut state.outliner_search)
                                    .hint_text(hint)
                                    .desired_width(ui.available_width()),
                            );
                        },
                    );
                    let query = state.outliner_search.trim().to_lowercase();
                    if query.is_empty() {
                        kreate_ui::outliner(
                            ui,
                            &state.catalog,
                            order.ids(),
                            order.depths(),
                            order.groups(),
                            selected,
                            active,
                            |index| order.row(doc, index),
                            &mut state.queue,
                        );
                    } else {
                        // The objects whose names hold what was typed, in the
                        // tree's order, flat: a match deep in a hierarchy is
                        // found without opening its parents.
                        let fresh = match &state.outliner_found {
                            Some((q, version, len, _)) => {
                                *q != query
                                    || *version != doc.version()
                                    || *len != order.ids().len()
                            }
                            None => true,
                        };
                        if fresh {
                            let found: Vec<usize> = order
                                .ids()
                                .iter()
                                .enumerate()
                                .filter(|(_, key)| {
                                    doc.object_at_slot(**key)
                                        .and_then(|id| doc.object(id))
                                        .is_some_and(|o| o.name.to_lowercase().contains(&query))
                                })
                                .map(|(i, _)| i)
                                .collect();
                            state.outliner_found =
                                Some((query.clone(), doc.version(), order.ids().len(), found));
                        }
                        let found = state
                            .outliner_found
                            .as_ref()
                            .map(|(_, _, _, f)| f.clone())
                            .unwrap_or_default();
                        if found.is_empty() {
                            ui.weak(state.catalog.get("outliner-search-none"));
                        }
                        let ids: Vec<u32> = found.iter().map(|i| order.ids()[*i]).collect();
                        let depths = vec![0u16; ids.len()];
                        kreate_ui::outliner(
                            ui,
                            &state.catalog,
                            &ids,
                            &depths,
                            &[],
                            selected,
                            active,
                            |index| {
                                let mut row = order.row(doc, *found.get(index)?)?;
                                row.depth = 0;
                                Some(row)
                            },
                            &mut state.queue,
                        );
                    }
                    state.outliner = order;
                }
                "properties" => {
                    draw_properties(
                        ui,
                        state,
                        doc,
                        registry,
                        history,
                        modifier_rows,
                        area.id,
                        l,
                        mode,
                    );
                }
                "uv" => {
                    draw_uv_editor(ui, state, area.id, l);
                }
                "console" => {
                    draw_console(ui, state, history, l);
                }
                "claude" => {
                    draw_chat(ui, state, l);
                }
                "assets" => {
                    draw_assets(ui, state);
                }
                "shader" => {
                    draw_shader_editor(ui, state, doc, area.id, l);
                }
                "geometry_nodes" => {
                    draw_geometry_editor(ui, state, doc, area.id, l);
                }
                "timeline" => {
                    draw_timeline(ui, state, area.id, l);
                }
                "dope_sheet" => {
                    draw_dope_sheet(ui, state, doc, area.id, l);
                }
                "nla" => {
                    draw_nla_editor(ui, state, doc, area.id);
                }
                "graph" => {
                    draw_graph_editor(ui, state, doc, area.id, l);
                }
                "drivers" => {
                    draw_drivers(ui, state, doc, area.id);
                }
                "preferences" => {
                    ui.horizontal(|ui| {
                        ui.label(state.catalog.get("prefs-theme"));
                        for a in kreate_ui::Appearance::ALL {
                            let key = match a {
                                kreate_ui::Appearance::Dark => "prefs-theme-dark",
                                kreate_ui::Appearance::Light => "prefs-theme-light",
                            };
                            if ui
                                .selectable_label(state.appearance == a, state.catalog.get(key))
                                .clicked()
                                && state.appearance != a
                            {
                                state.appearance = a;
                                kreate_ui::Theme::new(a, kreate_ui::Density::default())
                                    .install(ui.ctx());
                                if let Some(path) = crate::prefs::interface_path() {
                                    let _ = crate::prefs::save_appearance(&path, a);
                                }
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(state.catalog.get("prefs-resolution-scale"));
                        let mut scale = state.resolution_scale;
                        if ui
                            .add(egui::Slider::new(
                                &mut scale,
                                crate::prefs::MIN_RESOLUTION_SCALE..=1.0,
                            ))
                            .on_hover_text(state.catalog.get("prefs-resolution-scale-desc"))
                            .changed()
                        {
                            state.resolution_scale = scale;
                            if let Some(path) = crate::prefs::viewport_path() {
                                let _ = crate::prefs::save_resolution_scale(&path, scale);
                            }
                        }
                    });
                    ui.separator();
                    let conflicts = keymap.conflicts();
                    let text = keymap_text(&state.catalog, overrides.len(), conflicts.len());
                    let mut edits = std::mem::take(&mut state.keymap_edits);
                    kreate_ui::keymap_editor(
                        ui,
                        keymap,
                        preset,
                        overrides,
                        &conflicts,
                        &text,
                        &mut state.capturing,
                        &mut edits,
                    );
                    state.keymap_edits = edits;
                }
                // An area naming an editor this build does not have. Said out
                // loud, with the id it wants, because the likeliest cause is a
                // missing add-on and the user can act on that. Silently drawing
                // a viewport instead — which is what the closed enum did — has
                // them rearrange a layout that was never broken.
                unknown => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(theme.panel_padding * 2.0);
                        ui.weak(&l.editor_unavailable);
                        // i18n-exempt: an editor id is data from a file
                        ui.monospace(unknown);
                    });
                }
            }
        },
    );
    inner
}

/// The strip along the top of an area: which editor, and that editor's menus.
///
/// **Each editor owns its header.** This is Blender's structure and the reason
/// the top bar is short: the menus for acting on a scene sit on the editor
/// that shows the scene, so they move with it, and a second viewport gets its
/// own. The alternative — one global row that changes with whichever editor
/// is "current" — is a row whose contents the user cannot predict.
///
/// Everything here queues rather than acting. The tree cannot be edited while
/// the frame is drawing against a layout derived from it.
fn draw_area_header(
    ui: &mut Ui,
    state: &mut ShellState,
    area: &kreate_ui::AreaRect,
    l: &Labels,
    mode: kreate_ops::operator::Mode,
) {
    ui.horizontal(|ui| {
        // From the registry, so an editor an add-on registered appears in the
        // dropdown without the shell being told about it. A variant list here
        // was the other half of what made registration impossible.
        let names: Vec<(kreate_ui::EditorId, String)> = state
            .editors
            .iter()
            .map(|info| {
                (
                    info.id.clone(),
                    state.catalog.format(info.name_key, TextKind::Ui, &[]),
                )
            })
            .collect();
        // An area showing an editor this build does not have says so, with the
        // id it is asking for. Falling back to a blank would leave the user
        // unable to tell a missing add-on from a broken layout.
        let current = names
            .iter()
            .find(|(id, _)| *id == area.editor)
            .map(|(_, name)| name.clone())
            // i18n-exempt: an unknown editor's id is data from a file, not prose
            .unwrap_or_else(|| format!("? {}", area.editor));

        egui::ComboBox::from_id_salt(("editor-kind", area.id.0))
            .selected_text(current)
            .show_ui(ui, |ui| {
                for (id, name) in &names {
                    if ui.selectable_label(*id == area.editor, name).clicked() {
                        state.set_area_editor(area.id, id.clone());
                    }
                }
            });

        let is_viewport = area.editor == kreate_ui::EditorId::VIEWPORT;

        // **The right-hand group is measured first and given its width; the
        // menus get what is left, clipped.** A right-to-left layout laid over
        // whatever the menus had not used drew *on top of them* the moment an
        // area was narrower than the two together — the UV Editing workspace's
        // half-width viewport had "Overlays" printed across "Face" and the
        // shading buttons cut off at the edge. Blender clips a header that does
        // not fit; nothing overlaps.
        let right_width = if is_viewport {
            header_right_width(ui, state, l)
        } else {
            0.0
        };
        let left_width = (ui.available_width() - right_width).max(0.0);
        egui::ScrollArea::horizontal()
            .id_salt(("header-menus", area.id.0))
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
            .max_width(left_width)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if is_viewport {
                        // The mode and its menus are the toolbar's now, as
                        // Fusion has them; the brush stays with the picture
                        // it paints.
                        draw_brush_settings(ui, state, area.id.0, mode);
                    }

                    // Every editor has a View menu, and the area operations
                    // live at the bottom of it — which is where Blender keeps
                    // them, and which is what makes "how do I split this" one
                    // answer for every editor.
                    draw_view_menu(ui, state, area, l, is_viewport);

                    if is_viewport {
                        // The element buttons stay buttons rather than becoming
                        // a menu. They are a *state* — which one is lit tells
                        // you what a click in the viewport will pick — and a
                        // state behind a menu is a state you cannot see.
                        // Pictures rather than words: with Mesh, Edge and Face
                        // menus in the same row, buttons reading "Edge" and
                        // "Face" meant something else two inches away.
                        if mode.info().element_buttons {
                            ui.separator();
                            let spacing = ui.spacing().item_spacing.x;
                            ui.spacing_mut().item_spacing.x = 2.0;
                            for element in kreate_ui::element_icon::Element::ALL {
                                let index = element.index();
                                let name = state.catalog.get(element.label_key());
                                if kreate_ui::element_icon::element_button(
                                    ui,
                                    element,
                                    state.select_mode == index,
                                )
                                .on_hover_text(&name)
                                .clicked()
                                {
                                    state.queue.push(Command::SetSelectMode(index));
                                }
                            }
                            ui.spacing_mut().item_spacing.x = spacing;
                        }
                        // Blender's Pivot Point, where it keeps it: in the
                        // header, for the modes G/R/S work on objects or mesh
                        // elements in.
                        if matches!(
                            mode.acts_on(),
                            kreate_ops::mode::ActsOn::Objects
                                | kreate_ops::mode::ActsOn::MeshElements
                        ) {
                            ui.separator();
                            let names = [
                                &l.pivot_median,
                                &l.pivot_individual,
                                &l.pivot_active,
                                &l.pivot_cursor,
                            ];
                            let current = names[state.pivot.min(3) as usize].clone();
                            egui::ComboBox::from_id_salt(("pivot", area.id.0))
                                .width(80.0)
                                .selected_text(current)
                                .show_ui(ui, |ui| {
                                    for (i, name) in names.iter().enumerate() {
                                        ui.selectable_value(
                                            &mut state.pivot,
                                            i as u32,
                                            name.as_str(),
                                        );
                                    }
                                })
                                .response
                                .on_hover_text(&l.pivot_label);
                            draw_tool_settings(ui, state, area.id.0, mode);
                        }
                    }
                });
            });

        if is_viewport {
            // The right-hand group: how the picture is made. Right-aligned
            // like Blender's shading buttons, so it stays put when the menus
            // to its left change with the mode.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_shading_pills(ui, state, l);
                ui.separator();
                draw_overlays_menu(ui, state, l);
                draw_exposure(ui, state, l);
            });
        }
    });
}

/// A brush mode's brush, in its header — first after the mode, ahead of the
/// menus, because it is what the hand is using: a narrow viewport scrolls the
/// menus off the end of the row before it loses the brush.
fn draw_brush_settings(
    ui: &mut Ui,
    state: &mut ShellState,
    area: u32,
    mode: kreate_ops::operator::Mode,
) {
    // The texture brush, in its header: its colour, size and strength, and
    // something to paint on.
    if mode.acts_on() == kreate_ops::mode::ActsOn::Texture {
        let t = &mut state.texture_paint;
        ui.color_edit_button_rgb(&mut t.color);
        ui.label(state.catalog.get("paint-radius"));
        ui.add(
            egui::DragValue::new(&mut state.texture_paint.radius)
                .range(2.0..=400.0)
                .speed(1.0)
                .suffix(" px"),
        );
        ui.label(state.catalog.get("paint-strength"));
        ui.add(
            egui::DragValue::new(&mut state.texture_paint.strength)
                .range(0.0..=1.0)
                .speed(0.01)
                .max_decimals(2),
        );
        if ui
            .button(state.catalog.get("op-paint-new-texture"))
            .on_hover_text(state.catalog.get("op-paint-new-texture-desc"))
            .clicked()
        {
            state.queue.run("paint.new_texture", PropertyMap::new());
        }
        ui.separator();
    }
    // The sculpt brush, in its header.
    if mode.acts_on() == kreate_ops::mode::ActsOn::Sculpt {
        let names: Vec<String> = kreate_ui::toolbar::Tool::SCULPT
            .iter()
            .map(|t| state.catalog.get(t.label_key()))
            .collect();
        let s = &mut state.sculpt;
        let last = names.len() as u32 - 1;
        egui::ComboBox::from_id_salt(("sculpt-brush", area))
            .width(92.0)
            .selected_text(names[s.brush.min(last) as usize].as_str())
            .show_ui(ui, |ui| {
                for (i, name) in names.iter().enumerate() {
                    ui.selectable_value(&mut s.brush, i as u32, name.as_str());
                }
            });
        // The size and strength without their names — the box under the
        // brush column names them — so the four drop-downs and the mode's
        // own menus still fit the header.
        let (radius, strength) = (
            state.catalog.get("paint-radius"),
            state.catalog.get("paint-strength"),
        );
        ui.add(
            egui::DragValue::new(&mut state.sculpt.radius)
                .range(4.0..=400.0)
                .speed(1.0)
                .suffix(" px"),
        )
        .on_hover_text(radius);
        ui.add(
            egui::DragValue::new(&mut state.sculpt.strength)
                .range(0.0..=1.0)
                .speed(0.01)
                .max_decimals(2),
        )
        .on_hover_text(strength);
        draw_sculpt_menus(ui, state);
        ui.separator();
    }
    // The brush, where Blender keeps it: in the header of
    // the mode that uses it.
    if mode.acts_on() == kreate_ops::mode::ActsOn::Weights {
        let p = &mut state.paint;
        ui.label(state.catalog.get("paint-weight"));
        ui.add(
            egui::DragValue::new(&mut p.weight)
                .range(0.0..=1.0)
                .speed(0.01)
                .max_decimals(2),
        );
        ui.label(state.catalog.get("paint-radius"));
        ui.add(
            egui::DragValue::new(&mut p.radius)
                .range(2.0..=400.0)
                .speed(1.0)
                .suffix(" px"),
        );
        ui.label(state.catalog.get("paint-strength"));
        ui.add(
            egui::DragValue::new(&mut p.strength)
                .range(0.0..=1.0)
                .speed(0.01)
                .max_decimals(2),
        );
        ui.checkbox(&mut p.front_only, state.catalog.get("paint-front-only"));
        ui.checkbox(&mut p.falloff, state.catalog.get("paint-falloff"));
        ui.separator();
    }
}

/// How wide the viewport header's right-hand group will be this frame.
///
/// Measured rather than remembered from the last frame, because the group
/// changes with the mode — the exposure readout appears in the rendered mode
/// only — and a one-frame-stale width is one frame of overlap.
fn header_right_width(ui: &Ui, state: &ShellState, l: &Labels) -> f32 {
    let spacing = ui.spacing();
    let text_width = |text: &str| {
        let font = egui::TextStyle::Button.resolve(ui.style());
        ui.painter()
            .layout_no_wrap(text.to_string(), font, egui::Color32::WHITE)
            .size()
            .x
    };
    let button = |text: &str| text_width(text) + spacing.button_padding.x * 2.0;
    // Four shading buttons at two points apart, a separator, Overlays, and
    // the exposure when it is shown, each followed by the row's item spacing.
    // Four since Wireframe joined them: counted as three, the menus were given
    // one button's width too many and ran under Overlays.
    let icon = 14.0 + spacing.button_padding.x * 2.0;
    let mut width = icon * 4.0 + 2.0 * 3.0 + spacing.item_spacing.x;
    width += spacing.item_spacing.x * 2.0 + 1.0; // the separator
    width += button(&l.overlays) + spacing.item_spacing.x;
    if state.view_status.shading == kreate_render::ShadingMode::Rendered {
        width += text_width(&exposure_text(state, l)) + spacing.item_spacing.x;
    }
    width + 6.0
}

/// The View menu: how this area is looked at, and the area itself.
///
/// The bottom section is the same for every editor — split, close — because a
/// user who has learnt it in one editor should not have to find it again in
/// the next. The top section is the viewport's alone.
fn draw_view_menu(
    ui: &mut Ui,
    state: &mut ShellState,
    area: &kreate_ui::AreaRect,
    l: &Labels,
    is_viewport: bool,
) {
    ui.menu_button(&l.menu_view, |ui| {
        if is_viewport {
            draw_nodes(ui, state, l, &view_nodes(l));
            ui.separator();
        }
        ui.menu_button(&l.area_menu, |ui| {
            if menu_item(ui, state, "area.split_vertical", &[], &l.area_split_v) {
                state.split_area(area.id, kreate_ui::Axis::Vertical);
            }
            if menu_item(ui, state, "area.split_horizontal", &[], &l.area_split_h) {
                state.split_area(area.id, kreate_ui::Axis::Horizontal);
            }
            // Disabled rather than hidden on the last area: a menu whose
            // entries come and go is a menu the user cannot learn.
            let last = state.screen().area_count() <= 1;
            if menu_item_enabled(ui, state, "area.close", &[], &l.area_close, !last) {
                state.close_area(area.id);
            }
        });
    });
}

/// A menu entry: the label, and on hover what it does and which key does it.
///
/// **Every entry in every menu goes through this**, which is what makes the
/// tooltips complete rather than present where somebody remembered. The
/// sentence comes from the catalog under the entry's id — `op-file-save-desc`,
/// `op-mesh-select-action-1-desc` — with the variant tried first and the bare
/// operator second, so Select ▸ None can say something other than Select ▸ All
/// while Bevel needs one sentence. The key comes from the keymap, resolved in
/// the current mode, through `command_info`. An entry with neither shows no
/// tooltip rather than an empty box.
///
/// Closes the menu on a click, which a menu entry has to do: leaving it open
/// after running a command hides the very thing the command changed.
fn menu_item(
    ui: &mut Ui,
    state: &ShellState,
    id: &str,
    params: &[(&'static str, kreate_reflect::PropertyValue)],
    label: &str,
) -> bool {
    menu_item_enabled(ui, state, id, params, label, true)
}

/// [`menu_item`], greyed out when it cannot apply. Undo with nothing to undo.
fn menu_item_enabled(
    ui: &mut Ui,
    state: &ShellState,
    id: &str,
    params: &[(&'static str, kreate_reflect::PropertyValue)],
    label: &str,
    enabled: bool,
) -> bool {
    let mut map = PropertyMap::new();
    for (name, value) in params {
        map.set(name, value.clone());
    }
    let tooltip = menu_tooltip(state, id, &map);
    let response = ui.add_enabled(enabled, egui::Button::new(label));
    let response = if tooltip.is_empty() {
        response
    } else {
        response.on_hover_text(kreate_ui::text::wrap_tooltip(ui, &tooltip))
    };
    if response.clicked() {
        ui.close();
        true
    } else {
        false
    }
}

/// What a menu entry's tooltip says: its description and its key.
fn menu_tooltip(state: &ShellState, id: &str, map: &PropertyMap) -> String {
    let variant = kreate_ui::command_info::variant_id(id, map);
    let description = [variant.as_str(), id]
        .iter()
        .map(|key| kreate_ui::command_info::description_key(key))
        .find(|key| state.catalog.has(key))
        .map(|key| state.catalog.format(&key, TextKind::Tooltip, &[]))
        .unwrap_or_default();
    let shortcut = state
        .command_info
        .get(&variant)
        .or_else(|| state.command_info.get(id))
        .map(|info| info.shortcut.clone())
        .unwrap_or_default();
    let tooltip = match (description.is_empty(), shortcut.is_empty()) {
        (true, true) => String::new(),
        (false, true) => description,
        (true, false) => shortcut,
        // i18n-exempt: a translated sentence over a key name
        (false, false) => format!("{description}\n{shortcut}"),
    };
    tooltip
}

// --- the menus, as data ------------------------------------------------------
//
// Every header menu is a tree built here and drawn by `draw_nodes`, and F3's
// search reads the same tree. One description of what each entry is and does,
// so the search cannot offer an entry the menus do not have, or run it with
// different parameters than the menu would — which is exactly what a second
// table would drift into.

/// What choosing an entry does.
#[derive(Clone, Debug)]
enum Run {
    /// Through the history, with exactly these parameters.
    Op(&'static str, PropertyMap),
    /// Through the application's dispatch — the door the keys use — so an app
    /// action runs as its key does, and a transform starts the same drag `G`
    /// starts rather than moving by a fixed amount.
    Dispatch(&'static str, PropertyMap),
    /// A command of the interface's own.
    Command(Command),
    /// Opens the fit calculator, a window of the shell's own.
    Fits,
}

/// One entry.
#[derive(Clone, Debug)]
pub struct MenuEntry {
    pub label: String,
    /// What the entry is described and shortcut by: an operator's id, or an
    /// application action's.
    pub id: &'static str,
    /// The parameters that make this entry a variant of `id`, for its
    /// description and its shortcut. Not always what runs — see `run`.
    pub variant: Vec<(&'static str, kreate_reflect::PropertyValue)>,
    run: Run,
    pub enabled: bool,
}

impl MenuEntry {
    fn new(id: &'static str, label: &str, run: Run) -> Self {
        Self {
            label: label.to_string(),
            id,
            variant: Vec::new(),
            run,
            enabled: true,
        }
    }

    fn variant(mut self, name: &'static str, value: kreate_reflect::PropertyValue) -> Self {
        self.variant.push((name, value));
        self
    }

    fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// The operator or action it runs, for a check that it exists.
    pub fn runs(&self) -> Option<&'static str> {
        match &self.run {
            Run::Op(id, _) | Run::Dispatch(id, _) => Some(id),
            Run::Command(_) | Run::Fits => None,
        }
    }
}

/// A node of a menu.
#[derive(Clone, Debug)]
pub enum MenuNode {
    Entry(MenuEntry),
    Sub(String, Vec<MenuNode>),
    Separator,
    /// Edit ▸ Undo History: every step, the current one marked — drawn by
    /// `draw_undo_history`, from the history of the moment, and not searched.
    UndoHistory,
    /// File ▸ Recover, whose entries are the recoverable documents of the
    /// moment — drawn by `draw_recover_menu`, and not searched.
    Recover,
    /// File ▸ Open Recent, the documents of late — drawn from the list.
    Recent,
    /// Help ▸ Connect to AI Apps, each assistant with where it stands —
    /// drawn as the menu opens, from their settings files.
    ConnectApps,
}

/// A menu on a bar.
#[derive(Clone, Debug)]
pub struct TopMenu {
    pub label: String,
    pub nodes: Vec<MenuNode>,
}

fn op(id: &'static str, params: &[(&'static str, kreate_reflect::PropertyValue)]) -> Run {
    let mut map = PropertyMap::new();
    for (name, value) in params {
        map.set(name, value.clone());
    }
    Run::Op(id, map)
}

fn dispatch(id: &'static str) -> Run {
    Run::Dispatch(id, PropertyMap::new())
}

fn entry(id: &'static str, label: &str, run: Run) -> MenuNode {
    MenuNode::Entry(MenuEntry::new(id, label, run))
}

/// File and Edit: the top bar's own, true of the whole window.
fn top_bar_menus(l: &Labels, can_undo: bool, can_redo: bool) -> Vec<TopMenu> {
    use MenuNode::Separator;
    vec![
        TopMenu {
            label: l.menu_file.clone(),
            nodes: vec![
                entry("file.new", &l.file_new, Run::Command(Command::NewDocument)),
                entry(
                    "file.open",
                    &l.file_open,
                    Run::Command(Command::OpenDocument),
                ),
                MenuNode::Recent,
                // Recovered work under File, where Blender keeps it and where
                // somebody who has just lost a session looks first.
                MenuNode::Recover,
                Separator,
                entry(
                    "file.save",
                    &l.file_save,
                    Run::Command(Command::SaveDocument),
                ),
                entry(
                    "file.save_as",
                    &l.file_save_as,
                    Run::Command(Command::SaveDocumentAs),
                ),
                Separator,
                // A script is a file of history lines, the console's language.
                entry(
                    "file.run_script",
                    &l.file_run_script,
                    dispatch("file.run_script"),
                ),
                Separator,
                entry(
                    "file.collab",
                    &l.file_collab,
                    Run::Command(Command::OpenCollab),
                ),
                Separator,
                // Beside what is there, as one undo step: where Open replaces.
                // Any format the readers know, or one at a time.
                MenuNode::Sub(l.file_import_menu.clone(), {
                    let mut nodes = vec![entry(
                        "file.import",
                        &l.file_any_format,
                        Run::Command(Command::ImportScene),
                    )];
                    nodes.push(Separator);
                    for (index, format) in crate::formats::IMPORT.iter().enumerate() {
                        nodes.push(MenuNode::Entry(
                            MenuEntry::new(
                                "file.import",
                                // i18n-exempt: a file format's own name
                                &format.label(),
                                Run::Command(Command::ImportFormat { index }),
                            )
                            .variant("format", kreate_reflect::PropertyValue::Enum(index as u32)),
                        ));
                    }
                    nodes
                }),
                // An export is not a save: the document keeps no path from it
                // and treating a `.glb` as a working file loses everything the
                // format cannot hold.
                MenuNode::Sub(l.file_export_menu.clone(), {
                    let mut nodes = vec![entry(
                        "file.export",
                        &l.file_any_format,
                        Run::Command(Command::ExportScene),
                    )];
                    nodes.push(Separator);
                    for (index, format) in crate::formats::EXPORT.iter().enumerate() {
                        nodes.push(MenuNode::Entry(
                            MenuEntry::new(
                                "file.export",
                                // i18n-exempt: a file format's own name
                                &format.label(),
                                Run::Command(Command::ExportFormat { index }),
                            )
                            .variant("format", kreate_reflect::PropertyValue::Enum(index as u32)),
                        ));
                    }
                    nodes
                }),
                MenuNode::Sub(
                    l.file_export_drawing.clone(),
                    [
                        (0u32, &l.drawing_top),
                        (1, &l.drawing_front),
                        (2, &l.drawing_right),
                    ]
                    .into_iter()
                    .map(|(view, label)| {
                        let mut p = PropertyMap::new();
                        p.set("view", kreate_reflect::PropertyValue::Enum(view));
                        MenuNode::Entry(
                            MenuEntry::new(
                                "file.export_drawing",
                                label,
                                Run::Dispatch("file.export_drawing", p),
                            )
                            .variant("view", kreate_reflect::PropertyValue::Enum(view)),
                        )
                    })
                    .collect(),
                ),
            ],
        },
        TopMenu {
            label: l.menu_edit.clone(),
            nodes: vec![
                MenuNode::Entry(
                    MenuEntry::new("edit.undo", &l.undo, Run::Command(Command::Undo))
                        .enabled(can_undo),
                ),
                MenuNode::Entry(
                    MenuEntry::new("edit.redo", &l.redo, Run::Command(Command::Redo))
                        .enabled(can_redo),
                ),
                MenuNode::UndoHistory,
                Separator,
                entry("app.search", &l.menu_search, dispatch("app.search")),
                // The history, as the script it already is — Blender's "Copy
                // as Python", for the language this application speaks.
                entry(
                    "edit.copy_script",
                    &l.menu_copy_script,
                    dispatch("edit.copy_script"),
                ),
            ],
        },
        TopMenu {
            label: l.menu_render.clone(),
            nodes: vec![
                entry("render.image", &l.render_image, dispatch("render.image")),
                entry(
                    "render.animation",
                    &l.render_animation,
                    dispatch("render.animation"),
                ),
                Separator,
                entry("render.view", &l.render_view, dispatch("render.view")),
            ],
        },
        TopMenu {
            label: l.menu_help.clone(),
            nodes: vec![
                MenuNode::ConnectApps,
                Separator,
                entry(
                    "help.contact",
                    &l.help_contact,
                    Run::Command(Command::OpenContact),
                ),
                entry(
                    "help.logs",
                    &l.help_logs,
                    Run::Command(Command::OpenLogFolder),
                ),
                Separator,
                entry(
                    "help.about",
                    &l.help_about,
                    Run::Command(Command::OpenAbout),
                ),
            ],
        },
    ]
}

/// The viewport's View entries: how the scene is looked at.
fn view_nodes(l: &Labels) -> Vec<MenuNode> {
    vec![
        entry(
            "view.frame_all",
            &l.view_frame_all,
            dispatch("view.frame_all"),
        ),
        entry(
            "view.frame_selected",
            &l.view_frame_selected,
            Run::Command(Command::FrameSelection),
        ),
        // Numpad 0 is the key, and a keyboard without a numpad has no other
        // way through the camera.
        entry("view.camera", &l.view_camera, dispatch("view.camera")),
        entry("view.walk", &l.view_walk, dispatch("view.walk")),
        MenuNode::Separator,
        entry("view.measure", &l.view_measure, dispatch("view.measure")),
        entry(
            "object.align_faces",
            &l.align_faces,
            dispatch("object.align_faces"),
        ),
        MenuNode::Separator,
        entry(
            "view.toggle_projection",
            &l.view_perspective,
            dispatch("view.toggle_projection"),
        ),
        entry(
            "view.toggle_sidebar",
            &l.view_sidebar,
            dispatch("view.toggle_sidebar"),
        ),
        MenuNode::Separator,
        entry("time.play", &l.time_play_menu, dispatch("time.play")),
    ]
}

/// Select, Add, and the menus the mode's card names.
fn viewport_menus(l: &Labels, mode: kreate_ops::operator::Mode, select_mode: u32) -> Vec<TopMenu> {
    // Sculpt mode selects nothing — the brush is the whole interaction, and
    // Blender's has no Select menu there — and its header needs the room.
    let mut out = if matches!(
        mode.acts_on(),
        kreate_ops::mode::ActsOn::Sculpt | kreate_ops::mode::ActsOn::Sketch
    ) {
        Vec::new()
    } else {
        vec![TopMenu {
            label: l.menu_select.clone(),
            nodes: select_nodes(l, mode, select_mode),
        }]
    };
    // What Add adds is the mode's: objects where objects are acted on, a bone
    // where a skeleton is being built, and nothing in pose mode — a pose adds
    // nothing, as Blender's has no Add menu there.
    match mode.acts_on() {
        // A pose and a brush add nothing, as in Blender.
        kreate_ops::mode::ActsOn::Bones
        | kreate_ops::mode::ActsOn::Weights
        | kreate_ops::mode::ActsOn::Sculpt
        | kreate_ops::mode::ActsOn::Texture
        | kreate_ops::mode::ActsOn::Sketch => {}
        kreate_ops::mode::ActsOn::BoneJoints => out.push(TopMenu {
            label: l.add_menu.clone(),
            nodes: vec![entry(
                "armature.bone_add",
                &l.armature_add_bone,
                op("armature.bone_add", &[]),
            )],
        }),
        _ => out.push(TopMenu {
            label: l.add_menu.clone(),
            nodes: add_nodes(l),
        }),
    }
    // What the menus after Add are is the mode's to say, because what you can
    // act on is. Object mode acts on whole objects; edit mode on the parts of one.
    for menu in mode.info().menus {
        match menu {
            kreate_ops::mode::Menu::Object => out.extend([
                TopMenu {
                    label: l.menu_object.clone(),
                    nodes: {
                        let mut nodes = object_nodes(l);
                        nodes.push(MenuNode::Separator);
                        nodes.extend(transform_nodes(l));
                        nodes.push(physics_node(l));
                        nodes
                    },
                },
                TopMenu {
                    label: l.m("menu-mechanical"),
                    nodes: {
                        let mut n = mechanical_nodes(l);
                        n.push(MenuNode::Separator);
                        n.push(MenuNode::Entry(MenuEntry::new(
                            "mech.fits",
                            &l.m("mech-fits"),
                            Run::Fits,
                        )));
                        n
                    },
                },
            ]),
            kreate_ops::mode::Menu::Mesh => out.extend(mesh_menus(l, select_mode)),
            kreate_ops::mode::Menu::Pose => out.push(TopMenu {
                label: l.menu_pose.clone(),
                nodes: pose_nodes(l),
            }),
            kreate_ops::mode::Menu::Sculpt => out.push(TopMenu {
                label: l.menu_sculpt.clone(),
                nodes: {
                    use kreate_reflect::PropertyValue as V;
                    let mut nodes: Vec<MenuNode> = l
                        .sculpt_brushes
                        .iter()
                        .enumerate()
                        .map(|(i, label)| {
                            MenuNode::Entry(
                                MenuEntry::new(
                                    "sculpt.brush",
                                    label,
                                    Run::Command(Command::SetSculptBrush(i as u32)),
                                )
                                .variant("brush", V::Enum(i as u32)),
                            )
                        })
                        .collect();
                    nodes.push(MenuNode::Separator);
                    for (label, action) in
                        [(&l.sculpt_mask_clear, 0u32), (&l.sculpt_mask_invert, 1)]
                    {
                        nodes.push(MenuNode::Entry(
                            MenuEntry::new(
                                "sculpt.mask",
                                label,
                                op("sculpt.mask", &[("action", V::Enum(action))]),
                            )
                            .variant("action", V::Enum(action)),
                        ));
                    }
                    nodes.push(MenuNode::Entry(MenuEntry::new(
                        "sculpt.face_set_from_mask",
                        &l.sculpt_face_set_from_mask,
                        op("sculpt.face_set_from_mask", &[]),
                    )));
                    nodes.push(MenuNode::Entry(MenuEntry::new(
                        "sculpt.face_set_visibility",
                        &l.sculpt_show_all,
                        op("sculpt.face_set_visibility", &[("action", V::Enum(2))]),
                    )));
                    nodes.push(MenuNode::Separator);
                    nodes.push(entry(
                        "sculpt.remesh",
                        &l.sculpt_remesh,
                        dispatch("sculpt.remesh"),
                    ));
                    nodes.push(entry(
                        "sculpt.quad",
                        &l.sculpt_quad_remesh,
                        dispatch("sculpt.quad"),
                    ));
                    nodes
                },
            }),
            kreate_ops::mode::Menu::Sketch => out.push(TopMenu {
                label: l.menu_sketch.clone(),
                nodes: sketch_tool_nodes(
                    l,
                    &[
                        "select",
                        "line",
                        "rectangle",
                        "circle",
                        "arc",
                        "pattern",
                        "point",
                        "trim",
                    ],
                ),
            }),
            kreate_ops::mode::Menu::Constrain => out.push(TopMenu {
                label: l.menu_constrain.clone(),
                nodes: sketch_tool_nodes(
                    l,
                    &[
                        "dimension",
                        "coincident",
                        "perpendicular",
                        "horizontal",
                        "vertical",
                        "parallel",
                        "equal",
                        "tangent",
                        "concentric",
                        "fixed",
                    ],
                ),
            }),
            kreate_ops::mode::Menu::Solid => out.push(TopMenu {
                label: l.menu_solid.clone(),
                nodes: {
                    let mut nodes = sketch_tool_nodes(l, &["extrude", "revolve"]);
                    nodes.push(MenuNode::Separator);
                    nodes.push(entry(
                        "app.finish_sketch",
                        &l.finish_sketch,
                        Run::Command(Command::SetMode("object".into())),
                    ));
                    nodes
                },
            }),
            kreate_ops::mode::Menu::Texture => out.push(TopMenu {
                label: l.menu_texture.clone(),
                nodes: vec![entry(
                    "paint.new_texture",
                    &l.new_texture,
                    op("paint.new_texture", &[]),
                )],
            }),
            kreate_ops::mode::Menu::Weights => out.push(TopMenu {
                label: l.menu_weights.clone(),
                nodes: vec![
                    entry(
                        "object.vertex_group_add",
                        &l.group_add,
                        op("object.vertex_group_add", &[]),
                    ),
                    MenuNode::Separator,
                    entry(
                        "paint.weights_normalize",
                        &l.weights_normalize,
                        op("paint.weights_normalize", &[]),
                    ),
                ],
            }),
            kreate_ops::mode::Menu::Armature => out.push(TopMenu {
                label: l.menu_armature.clone(),
                nodes: vec![
                    entry(
                        "armature.bone_add",
                        &l.armature_add_bone,
                        op("armature.bone_add", &[]),
                    ),
                    entry(
                        "armature.extrude",
                        &l.armature_extrude,
                        dispatch("armature.extrude"),
                    ),
                    MenuNode::Separator,
                    entry(
                        "armature.delete",
                        &l.armature_delete,
                        op("armature.delete", &[]),
                    ),
                ],
            }),
        }
    }
    out
}

/// Every menu a search offers in this mode: the top bar's, the viewport's
/// View entries, and the viewport's menus.
pub fn searchable_menus(
    catalog: &Catalog,
    mode: kreate_ops::operator::Mode,
    select_mode: u32,
    can_undo: bool,
    can_redo: bool,
) -> Vec<TopMenu> {
    let l = Labels::new(catalog);
    let mut out = top_bar_menus(&l, can_undo, can_redo);
    out.push(TopMenu {
        label: l.menu_view.clone(),
        nodes: view_nodes(&l),
    });
    out.extend(viewport_menus(&l, mode, select_mode));
    out
}

/// Every entry of some menus, with the path to it.
pub fn flatten(menus: &[TopMenu]) -> Vec<(String, MenuEntry)> {
    fn walk(path: &str, nodes: &[MenuNode], out: &mut Vec<(String, MenuEntry)>) {
        for node in nodes {
            match node {
                MenuNode::Entry(e) => out.push((path.to_string(), e.clone())),
                MenuNode::Sub(label, children) => walk(&format!("{path} / {label}"), children, out),
                MenuNode::Separator
                | MenuNode::Recover
                | MenuNode::Recent
                | MenuNode::ConnectApps
                | MenuNode::UndoHistory => {}
            }
        }
    }
    let mut out = Vec::new();
    for menu in menus {
        walk(&menu.label, &menu.nodes, &mut out);
    }
    out
}

/// Draws a tree into an open menu.
fn draw_nodes(ui: &mut Ui, state: &mut ShellState, l: &Labels, nodes: &[MenuNode]) {
    for node in nodes {
        match node {
            MenuNode::Entry(e) => {
                if menu_item_enabled(ui, state, e.id, &e.variant, &e.label, e.enabled) {
                    run_entry(state, e);
                }
            }
            MenuNode::Sub(label, children) => {
                ui.menu_button(label, |ui| draw_nodes(ui, state, l, children));
            }
            MenuNode::Separator => {
                ui.separator();
            }
            MenuNode::Recover => draw_recover_menu(ui, state, l),
            MenuNode::Recent => draw_recent_menu(ui, state, l),
            MenuNode::ConnectApps => draw_connect_menu(ui, state),
            MenuNode::UndoHistory => draw_undo_history(ui, state, l),
        }
    }
}

/// Edit ▸ Undo History: Blender's list of steps. Clicking one undoes or redoes
/// to just after it; "Original" is before the first.
fn draw_undo_history(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    ui.menu_button(&l.undo_history, |ui| {
        let at = state.history_at;
        // The most recent at the bottom, as in Blender, and only the last
        // several dozen — a history of thousands is not a menu.
        const SHOWN: usize = 40;
        let total = state.history_steps.len();
        let first = total.saturating_sub(SHOWN);
        egui::ScrollArea::vertical()
            .max_height(480.0)
            .show(ui, |ui| {
                if first == 0 && ui.selectable_label(at == 0, &l.undo_original).clicked() {
                    state.queue.push(Command::UndoTo(0));
                    ui.close();
                }
                for (i, label) in state.history_steps.iter().enumerate().skip(first) {
                    let applied = i + 1;
                    let text = if applied > at {
                        egui::RichText::new(label).weak()
                    } else {
                        egui::RichText::new(label)
                    };
                    if ui.selectable_label(applied == at, text).clicked() {
                        state.queue.push(Command::UndoTo(applied));
                        ui.close();
                    }
                }
            });
    });
}

/// Does what choosing `entry` does — from a menu or from the search.
fn run_entry(state: &mut ShellState, entry: &MenuEntry) {
    match &entry.run {
        Run::Op(id, params) => state.queue.run(*id, params.clone()),
        Run::Dispatch(id, params) => state.queue.dispatch(*id, params.clone()),
        Run::Command(command) => state.queue.push(command.clone()),
        Run::Fits => state.fits.open = true,
    }
}

/// The Select menu: all, none, invert, and in edit mode the loop.
///
/// One menu in both modes with the same first three entries, which is the
/// point of it: "select all" is one idea, and the user should not have to know
/// that the operator behind it changes name when the mode does.
fn select_nodes(l: &Labels, mode: kreate_ops::operator::Mode, select_mode: u32) -> Vec<MenuNode> {
    use kreate_reflect::PropertyValue as V;
    let mut out = Vec::new();
    let bones = match mode.acts_on() {
        kreate_ops::mode::ActsOn::Bones => Some("pose.select_all"),
        kreate_ops::mode::ActsOn::BoneJoints => Some("armature.select_all"),
        _ => None,
    };
    if let Some(id) = bones {
        for (action, label) in [
            (0u32, &l.select_all),
            (1, &l.select_none),
            (3, &l.select_invert),
        ] {
            out.push(MenuNode::Entry(
                MenuEntry::new(id, label, op(id, &[("action", V::Enum(action))]))
                    .variant("action", V::Enum(action)),
            ));
        }
    } else if mode.edits_elements() {
        for (action, label) in [
            (0u32, &l.select_all),
            (1, &l.select_none),
            (2, &l.select_invert),
        ] {
            out.push(MenuNode::Entry(
                MenuEntry::new(
                    "mesh.select",
                    label,
                    op(
                        "mesh.select",
                        &[("action", V::Enum(action)), ("mode", V::Enum(select_mode))],
                    ),
                )
                .variant("action", V::Enum(action)),
            ));
        }
        out.push(MenuNode::Separator);
        for (action, label) in [
            (6u32, &l.select_more),
            (7, &l.select_less),
            (8, &l.select_linked),
        ] {
            out.push(MenuNode::Entry(
                MenuEntry::new(
                    "mesh.select",
                    label,
                    op(
                        "mesh.select",
                        &[("action", V::Enum(action)), ("mode", V::Enum(select_mode))],
                    ),
                )
                .variant("action", V::Enum(action)),
            ));
        }
        // Blender's Select Similar, for the element type in use (Shift+G).
        let mode = match select_mode {
            1 => kreate_mesh::bmesh::SelectMode::Edge,
            2 => kreate_mesh::bmesh::SelectMode::Face,
            _ => kreate_mesh::bmesh::SelectMode::Vert,
        };
        let similar: Vec<MenuNode> = kreate_mesh::selection::SIMILAR
            .iter()
            .enumerate()
            .filter(|(_, (_, m))| *m == mode)
            .map(|(i, _)| {
                MenuNode::Entry(
                    MenuEntry::new(
                        "mesh.select_similar",
                        &l.similar[i],
                        op("mesh.select_similar", &[("kind", V::Enum(i as u32))]),
                    )
                    .variant("kind", V::Enum(i as u32)),
                )
            })
            .collect();
        out.push(MenuNode::Sub(l.menu_select_similar.clone(), similar));
        out.push(MenuNode::Separator);
        out.push(MenuNode::Entry(
            MenuEntry::new(
                "mesh.select",
                &l.select_loop,
                op("mesh.select", &[("action", V::Enum(3))]),
            )
            .variant("action", V::Enum(3)),
        ));
    } else {
        // `object.select_all` counts select, deselect, toggle, invert; the
        // menu offers the three a person asks for by name.
        for (action, label) in [
            (0u32, &l.select_all),
            (1, &l.select_none),
            (3, &l.select_invert),
        ] {
            out.push(MenuNode::Entry(
                MenuEntry::new(
                    "object.select_all",
                    label,
                    op("object.select_all", &[("action", V::Enum(action))]),
                )
                .variant("action", V::Enum(action)),
            ));
        }
    }
    out
}

/// The Pose menu: putting what was done to the selected bones back.
fn pose_nodes(l: &Labels) -> Vec<MenuNode> {
    use kreate_reflect::PropertyValue as V;
    let mut out: Vec<MenuNode> = vec![
        entry("pose.ik_add", &l.pose_ik_add, op("pose.ik_add", &[])),
        MenuNode::Separator,
    ];
    out.extend(
        [
            (0u32, &l.pose_clear_all),
            (1, &l.pose_clear_location),
            (2, &l.pose_clear_rotation),
            (3, &l.pose_clear_scale),
        ]
        .into_iter()
        .map(|(what, label)| {
            MenuNode::Entry(
                MenuEntry::new(
                    "pose.clear_transform",
                    label,
                    op("pose.clear_transform", &[("what", V::Enum(what))]),
                )
                .variant("what", V::Enum(what)),
            )
        }),
    );
    out
}

/// The Add menu, in Blender's shape: Mesh, Light, Camera, Empty.
///
/// Submenus because the list is long enough to need them — eight shapes and
/// four lamps — and because the grouping is the one a modeller already knows.
/// The shape indices are `PRIMITIVES`' order, which a test holds the labels to.
fn add_nodes(l: &Labels) -> Vec<MenuNode> {
    use kreate_reflect::PropertyValue as V;
    let shapes: Vec<MenuNode> = (0..ADD_SHAPES.len())
        .map(|shape| {
            let value = V::Enum(shape as u32);
            MenuNode::Entry(
                MenuEntry::new(
                    "object.add_primitive",
                    &l.add_shapes[shape],
                    // Standing on the floor, as a part is put down.
                    op(
                        "object.add_primitive",
                        &[("shape", value.clone()), ("on_floor", V::Bool(true))],
                    ),
                )
                .variant("shape", value),
            )
        })
        .collect();
    let lights = [
        (0u32, &l.add_light_point),
        (1, &l.add_light_sun),
        (2, &l.add_light_spot),
        (3, &l.add_light_area),
    ]
    .into_iter()
    .map(|(kind, label)| {
        MenuNode::Entry(
            MenuEntry::new(
                "object.add_light",
                label,
                op("object.add_light", &[("kind", V::Enum(kind))]),
            )
            .variant("kind", V::Enum(kind)),
        )
    })
    .collect();
    let mut shapes = shapes;
    // Parts, not primitives: ISO gears of each kind, set up in the panel
    // after.
    shapes.push(MenuNode::Separator);
    for (kind, label) in l.add_gears.iter().enumerate() {
        let value = V::Enum(kind as u32);
        shapes.push(MenuNode::Entry(
            MenuEntry::new(
                "object.add_gear",
                label,
                op("object.add_gear", &[("kind", value.clone())]),
            )
            .variant("kind", value),
        ));
    }
    vec![
        MenuNode::Sub(l.add_mesh.clone(), shapes),
        MenuNode::Sub(l.add_light.clone(), lights),
        MenuNode::Separator,
        entry(
            "object.add_camera",
            &l.add_camera,
            op("object.add_camera", &[]),
        ),
        entry(
            "object.add_empty",
            &l.add_empty,
            op("object.add_empty", &[]),
        ),
        entry(
            "object.add_armature",
            &l.add_armature,
            op("object.add_armature", &[]),
        ),
    ]
}

/// The operator ids behind the Mesh submenu, in `PRIMITIVES` order — the
/// tooltip, and the thing a test compares against the operator's own list.
const ADD_SHAPES: [&str; 8] = [
    "plane",
    "cube",
    "grid",
    "uv_sphere",
    "ico_sphere",
    "cylinder",
    "cone",
    "torus",
];

/// The Object menu: what can be done to whole objects.
///
/// The transforms are here as well as on keys, and go through `Dispatch` so
/// that "Move" from the menu starts the same drag the G key starts. A menu
/// entry that moved by a fixed amount would be a different command wearing the
/// same name.
/// An argument a Mechanical menu entry gives its operator.
#[derive(Clone, Copy)]
enum MechArg {
    None,
    Enum(&'static str, u32),
    Bool(&'static str, bool),
}

/// The Mechanical menu, as data: each group (a submenu, or "" for entries
/// at the top), and in it each entry's name key, operator and argument.
/// What a machine's parts are made with — edges broken, holes cut, standard
/// parts put in — and the checks on them.
/// One Mechanical menu entry: its name key, operator and argument.
type MechEntry = (&'static str, &'static str, MechArg);

const MECH_MENU: &[(&str, &[MechEntry])] = &[
    (
        "",
        &[
            (
                "mech-fillet",
                "feature.round",
                MechArg::Bool("chamfer", false),
            ),
            (
                "mech-chamfer",
                "feature.round",
                MechArg::Bool("chamfer", true),
            ),
        ],
    ),
    (
        "mech-holes",
        &[
            ("mech-hole-simple", "object.hole", MechArg::Enum("kind", 0)),
            (
                "mech-hole-counterbore",
                "object.hole",
                MechArg::Enum("kind", 1),
            ),
            (
                "mech-hole-countersink",
                "object.hole",
                MechArg::Enum("kind", 2),
            ),
            ("mech-hole-tapped", "object.hole", MechArg::Enum("kind", 3)),
        ],
    ),
    (
        "mech-fasteners",
        &[
            (
                "mech-cap-screw",
                "object.add_fastener",
                MechArg::Enum("kind", 0),
            ),
            (
                "mech-hex-bolt",
                "object.add_fastener",
                MechArg::Enum("kind", 1),
            ),
            ("mech-nut", "object.add_fastener", MechArg::Enum("kind", 2)),
            (
                "mech-washer",
                "object.add_fastener",
                MechArg::Enum("kind", 3),
            ),
        ],
    ),
    (
        "",
        &[(
            "mech-interference",
            "object.check_interference",
            MechArg::None,
        )],
    ),
    (
        "mech-drive",
        &[
            ("mech-bearing", "object.add_bearing", MechArg::None),
            ("mech-shaft", "object.add_shaft", MechArg::None),
            ("mech-pulley", "object.add_pulley", MechArg::None),
            ("mech-sprocket", "object.add_sprocket", MechArg::None),
        ],
    ),
    (
        "mech-springs",
        &[
            (
                "mech-spring-compression",
                "object.add_spring",
                MechArg::Enum("kind", 0),
            ),
            (
                "mech-spring-extension",
                "object.add_spring",
                MechArg::Enum("kind", 1),
            ),
            (
                "mech-spring-torsion",
                "object.add_spring",
                MechArg::Enum("kind", 2),
            ),
        ],
    ),
    ("", &[("mech-gear-link", "object.gear_link", MechArg::None)]),
    (
        "mech-shapes",
        &[
            ("mech-sweep-along", "object.sweep_along", MechArg::None),
            ("mech-loft", "object.loft", MechArg::None),
        ],
    ),
    (
        "mech-sheet",
        &[
            (
                "mech-sheet-folded",
                "object.add_sheet_metal",
                MechArg::Bool("flat", false),
            ),
            (
                "mech-sheet-flat",
                "object.add_sheet_metal",
                MechArg::Bool("flat", true),
            ),
        ],
    ),
];

/// Every key the Mechanical menu names, to look up once.
fn mech_keys() -> Vec<&'static str> {
    // Named here too: the edit mode's Face menu has the shell and draft.
    let mut keys = vec!["menu-mechanical", "mech-shell", "mech-draft", "mech-fits"];
    keys.extend(FITS_KEYS);
    for (group, entries) in MECH_MENU {
        if !group.is_empty() {
            keys.push(group);
        }
        keys.extend(entries.iter().map(|(k, _, _)| *k));
    }
    keys
}

impl Labels {
    /// A Mechanical menu name by its key.
    fn m(&self, key: &str) -> String {
        self.mech
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string())
    }
}

/// The Mechanical menu's entries, from [`MECH_MENU`].
fn mechanical_nodes(l: &Labels) -> Vec<MenuNode> {
    use kreate_reflect::PropertyValue as V;
    let node = |(key, id, arg): &MechEntry| {
        type Pair = (&'static str, V);
        let (pairs, variant): (Vec<Pair>, Option<Pair>) = match *arg {
            MechArg::None => (Vec::new(), None),
            MechArg::Enum(n, v) => (vec![(n, V::Enum(v))], Some((n, V::Enum(v)))),
            MechArg::Bool(n, v) => (vec![(n, V::Bool(v))], Some((n, V::Bool(v)))),
        };
        let mut e = MenuEntry::new(id, &l.m(key), op(id, &pairs));
        if let Some((n, v)) = variant {
            e = e.variant(n, v);
        }
        MenuNode::Entry(e)
    };
    let mut out = Vec::new();
    for (group, entries) in MECH_MENU {
        if group.is_empty() {
            out.extend(entries.iter().map(node));
        } else {
            out.push(MenuNode::Sub(
                l.m(group),
                entries.iter().map(node).collect(),
            ));
        }
    }
    out
}

fn object_nodes(l: &Labels) -> Vec<MenuNode> {
    use kreate_reflect::PropertyValue as V;
    vec![
        MenuNode::Sub(
            l.menu_transform.clone(),
            vec![
                entry(
                    "object.translate",
                    &l.transform_move,
                    dispatch("object.translate"),
                ),
                entry(
                    "object.rotate",
                    &l.transform_rotate,
                    dispatch("object.rotate"),
                ),
                entry("object.scale", &l.transform_scale, dispatch("object.scale")),
            ],
        ),
        MenuNode::Separator,
        MenuNode::Entry(
            MenuEntry::new(
                "object.duplicate",
                &l.duplicate,
                op("object.duplicate", &[("linked", V::Bool(true))]),
            )
            .variant("linked", V::Bool(true)),
        ),
        // Copy and paste beside Duplicate, which is the same family: a paste
        // is a duplicate that survives the original being deleted and can be
        // put down later. Through `Dispatch`, the door the keys use.
        entry("edit.copy", &l.object_copy, dispatch("edit.copy")),
        entry("edit.paste", &l.object_paste, dispatch("edit.paste")),
        MenuNode::Separator,
        // The one-step boolean, for whoever has not met Ctrl+minus: the
        // cutters selected, the object to cut active.
        MenuNode::Sub(
            l.menu_boolean.clone(),
            (0..3u32)
                .map(|n| {
                    MenuNode::Entry(
                        MenuEntry::new(
                            "object.boolean",
                            &l.boolean_ops[n as usize],
                            op("object.boolean", &[("operation", V::Enum(n))]),
                        )
                        .variant("operation", V::Enum(n)),
                    )
                })
                .collect(),
        ),
        // Through `Dispatch`, so the application fills in the frame the
        // timeline is on — the key the menu entry stands for does the same.
        MenuNode::Sub(
            l.menu_animation.clone(),
            vec![
                entry(
                    "anim.keyframe_insert",
                    &l.anim_insert,
                    dispatch("anim.keyframe_insert"),
                ),
                entry(
                    "anim.keyframe_delete",
                    &l.anim_delete,
                    dispatch("anim.keyframe_delete"),
                ),
            ],
        ),
        // Blender's Parent menu, its armature half: select the meshes, then
        // the rig, and bind them with weights or with empty groups to paint.
        MenuNode::Sub(
            l.menu_parent.clone(),
            vec![
                entry("object.parent", &l.parent_object, op("object.parent", &[])),
                MenuNode::Entry(
                    MenuEntry::new(
                        "object.parent",
                        &l.parent_bone,
                        op("object.parent", &[("bone", V::Bool(true))]),
                    )
                    .variant("bone", V::Bool(true)),
                ),
                MenuNode::Entry(
                    MenuEntry::new(
                        "object.parent",
                        &l.parent_clear,
                        op("object.parent", &[("clear", V::Bool(true))]),
                    )
                    .variant("clear", V::Bool(true)),
                ),
                MenuNode::Separator,
                MenuNode::Entry(
                    MenuEntry::new(
                        "object.bind_armature",
                        &l.bind_automatic,
                        op("object.bind_armature", &[("weights", V::Enum(0))]),
                    )
                    .variant("weights", V::Enum(0)),
                ),
                MenuNode::Entry(
                    MenuEntry::new(
                        "object.bind_armature",
                        &l.bind_empty,
                        op("object.bind_armature", &[("weights", V::Enum(1))]),
                    )
                    .variant("weights", V::Enum(1)),
                ),
                MenuNode::Separator,
                entry(
                    "object.add_humanoid_rig",
                    &l.add_humanoid_rig,
                    op("object.add_humanoid_rig", &[]),
                ),
                entry("robot.rig", &l.robot_rig, op("robot.rig", &[])),
                entry(
                    "object.unbind_armature",
                    &l.unbind,
                    op("object.unbind_armature", &[]),
                ),
            ],
        ),
        MenuNode::Sub(
            l.menu_show_hide.clone(),
            vec![
                entry("object.reveal", &l.reveal_hidden, op("object.reveal", &[])),
                entry("object.hide", &l.hide_selected, op("object.hide", &[])),
                MenuNode::Entry(
                    MenuEntry::new(
                        "object.hide",
                        &l.hide_unselected,
                        op("object.hide", &[("unselected", V::Bool(true))]),
                    )
                    .variant("unselected", V::Bool(true)),
                ),
            ],
        ),
        MenuNode::Separator,
        entry("app.rename", &l.rename, dispatch("app.rename")),
        entry("object.delete", &l.delete, op("object.delete", &[])),
    ]
}

/// The three ways of shading the viewport, as a row of exclusive buttons.
///
/// Buttons rather than a dropdown, which is what Blender does and for a
/// reason a dropdown cannot match: the *current* one is lit, so which way the
/// picture is being made is readable without opening anything. In the
/// rendered mode the exposure readout sits beside them, because it is a fact
/// about that mode alone.
fn draw_shading_pills(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    let current = state.view_status.shading;
    let spacing = ui.spacing().item_spacing.x;
    ui.spacing_mut().item_spacing.x = 2.0;
    // Right-to-left layout, so the list is reversed to read as Blender's
    // does from left to right: wireframe, solid, material preview, rendered.
    // Flat lighting is the solid view's, so the solid button is lit for it.
    for mode in [
        kreate_render::ShadingMode::Rendered,
        kreate_render::ShadingMode::Material,
        kreate_render::ShadingMode::Studio,
        kreate_render::ShadingMode::Wireframe,
    ] {
        let on = mode == current
            || (mode == kreate_render::ShadingMode::Studio
                && current == kreate_render::ShadingMode::Flat);
        // i18n-exempt: a translated name and a translated hint
        let tip = format!("{}\n{}", shading_label(mode, l), l.shading_hint);
        if kreate_ui::view_icons::shading_button(ui, mode, on)
            .on_hover_text(tip)
            .clicked()
        {
            state.queue.push(Command::SetShading(mode));
        }
    }
    ui.spacing_mut().item_spacing.x = spacing;
}

/// The overlays: what is drawn over the picture that is not the scene.
///
/// Toggles under one menu rather than a button each, because the set will
/// grow — Blender's has two dozen — and a header full of toggles is a header
/// nobody can read. Each entry says which key does the same thing.
fn draw_overlays_menu(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    let on = state.view_status.overlays;
    ui.menu_button(&l.overlays, |ui| {
        for (op, label, checked) in [
            ("view.toggle_grid", &l.overlay_grid, on.grid),
            ("view.toggle_wireframe", &l.overlay_wireframe, on.wireframe),
            (
                "view.toggle_face_orientation",
                &l.overlay_face_orientation,
                on.face_orientation,
            ),
            (
                "view.toggle_statistics",
                &l.overlay_statistics,
                on.statistics,
            ),
            (
                "view.toggle_dimensions",
                &l.overlay_dimensions,
                on.dimensions,
            ),
            ("view.toggle_flat", &l.overlay_flat, on.flat),
            ("view.toggle_ao", &l.overlay_ao, on.ambient_occlusion),
            ("view.toggle_xray", &l.overlay_xray, on.xray),
            ("view.toggle_outline", &l.overlay_outline, on.outline),
            ("view.toggle_bloom", &l.overlay_bloom, on.bloom),
            (
                "view.toggle_reflections",
                &l.overlay_reflections,
                on.reflections,
            ),
            (
                "view.toggle_contact_shadows",
                &l.overlay_contact_shadows,
                on.contact_shadows,
            ),
            (
                "view.toggle_path_trace",
                &l.overlay_path_trace,
                on.path_trace,
            ),
            ("view.toggle_textures", &l.overlay_textures, on.textures),
        ] {
            // A check box per setting, as Blender's popover: what is on is
            // readable without toggling it to find out. The menu stays open,
            // so several can be set in one visit.
            let mut ticked = checked;
            let tip = menu_tooltip(state, op, &PropertyMap::new());
            let response = ui.checkbox(&mut ticked, label.as_str());
            let response = if tip.is_empty() {
                response
            } else {
                response.on_hover_text(kreate_ui::text::wrap_tooltip(ui, &tip))
            };
            if response.clicked() {
                state.queue.dispatch(op, PropertyMap::new());
            }
        }
    });
}

/// The 3D viewport: the rendered image, and everything painted over it.
#[allow(clippy::too_many_arguments)]
fn draw_viewport(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    mode: kreate_ops::operator::Mode,
    l: &Labels,
    rect: egui::Rect,
    image: Option<ViewportImage>,
    sidebar: bool,
) {
    if let Some(image) = image {
        // Only the part of the target that was drawn into. Sampling the whole
        // texture would stretch the image by however much the allocation was
        // rounded up.
        ui.painter().image(
            image.texture,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(image.uv.0, image.uv.1)),
            egui::Color32::WHITE,
        );
    }
    let toolbar = draw_toolbar(ui, state, mode, rect);
    draw_sculpt_box(ui, state, mode, rect, toolbar);
    draw_view_info(ui, state, doc, mode, l, rect, toolbar);
    draw_select_band(ui, state, rect);
    draw_modal_header(ui, state, rect);
    // Left of the sidebar when it is open, as Blender moves it: under the
    // sidebar it was hidden exactly while the sidebar was in use.
    let mut gizmo_room = rect;
    if sidebar {
        gizmo_room.max.x -= sidebar_width(rect);
    }
    draw_nav_gizmo(ui, state, gizmo_room);
    draw_context_menu(ui, state, rect);
    draw_rename(ui, state, rect);
    draw_imports(ui, state, rect);
}

/// A file being read, as the progress card shows it.
#[derive(Clone, Debug)]
pub struct ImportStatus {
    pub id: u64,
    pub name: String,
    /// File ▸ Open rather than Import.
    pub opening: bool,
    pub stage: kreate_io::scene_file::ReadStage,
    /// How far through the stage, when it can say.
    pub fraction: Option<f32>,
    pub seconds: u64,
    pub cancelling: bool,
}

/// The files being read, over the top of the viewport: what, how far, how
/// long, and a way to stop. A large CAD file takes minutes, and without this
/// nothing on the screen said whether it was coming at all.
fn draw_imports(ui: &mut Ui, state: &mut ShellState, viewport: egui::Rect) {
    if state.importing.is_empty() {
        return;
    }
    let width = 340.0f32.min(viewport.width() - 16.0).max(120.0);
    let mut cancel = None;
    egui::Area::new(egui::Id::new("imports"))
        .order(egui::Order::Foreground)
        .fixed_pos(egui::pos2(
            viewport.center().x - width / 2.0,
            viewport.min.y + 36.0,
        ))
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(width);
                for job in &state.importing {
                    let title = if job.opening {
                        "import-opening-title"
                    } else {
                        "import-progress-title"
                    };
                    ui.label(
                        egui::RichText::new(state.catalog.get_args(title, &[("name", &job.name)]))
                            .strong(),
                    );
                    use kreate_io::scene_file::ReadStage;
                    let stage = state.catalog.get(match job.stage {
                        ReadStage::Reading => "import-stage-reading",
                        ReadStage::Parsing => "import-stage-parsing",
                        ReadStage::Shaping => "import-stage-shaping",
                        ReadStage::Building => "import-stage-building",
                    });
                    let bar = match job.fraction {
                        Some(f) => {
                            let percent = ((f * 100.0).round() as u32).to_string();
                            egui::ProgressBar::new(f).text(state.catalog.get_args(
                                "import-stage-percent",
                                &[("stage", &stage), ("percent", &percent)],
                            ))
                        }
                        None => egui::ProgressBar::new(0.0).animate(true).text(stage),
                    };
                    let accent = kreate_ui::theme::from_ui(ui).accent;
                    ui.add(bar.desired_width(width).fill(accent.gamma_multiply(0.7)));
                    ui.horizontal(|ui| {
                        let seconds = job.seconds.to_string();
                        ui.weak(
                            state
                                .catalog
                                .get_args("import-elapsed", &[("seconds", &seconds)]),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let label = if job.cancelling {
                                "import-cancelling"
                            } else {
                                "import-cancel"
                            };
                            if ui
                                .add_enabled(
                                    !job.cancelling,
                                    egui::Button::new(state.catalog.get(label)),
                                )
                                .clicked()
                            {
                                cancel = Some(job.id);
                            }
                        });
                    });
                    ui.add_space(4.0);
                }
            });
        });
    if cancel.is_some() {
        state.cancel_import = cancel;
    }
}

/// F2, Blender's: the active object's name in a field under the pointer.
/// Enter or a click away keeps it, Escape leaves it as it was.
fn draw_rename(ui: &mut egui::Ui, state: &mut ShellState, viewport: egui::Rect) {
    let Some((at, text)) = state.rename.as_mut() else {
        return;
    };
    // The viewport it was asked for in, as the context menu is.
    if !viewport.contains(*at) {
        return;
    }
    let mut done: Option<bool> = None;
    let title = state.catalog.format("outliner-rename", TextKind::Ui, &[]);
    egui::Area::new(egui::Id::new("rename-popup"))
        .order(egui::Order::Foreground)
        .fixed_pos(*at)
        .constrain_to(viewport)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(title);
                let response = ui.add(egui::TextEdit::singleline(text).desired_width(180.0));
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    done = Some(false);
                } else if response.lost_focus() {
                    done = Some(true);
                } else if !response.has_focus() {
                    response.request_focus();
                }
            });
        });
    if let Some(keep) = done {
        if keep {
            let mut params = PropertyMap::new();
            params.set("name", kreate_reflect::PropertyValue::Text(text.clone()));
            state.queue.run("object.rename", params);
        }
        state.rename = None;
    }
}

/// What a click on `hit` does to an object selection.
///
/// **Blender's rule, and the outliner's.** A plain click selects the hit and
/// nothing else. A shift-click adds it — or, when it is already selected,
/// takes it out again, because the same gesture on the same thing twice
/// should undo itself rather than do nothing. The active object becomes the
/// hit when it was added, and when the hit was removed and was active, falls
/// to the last thing still selected, so there is always something for the
/// panels to show while anything is selected.
///
/// A pure function so it can be tested without a frame: the viewport's click
/// arrives in the application and the outliner's in a widget, and both have
/// to give the same answer to the same gesture.
pub fn click_selection(
    selected: &[u32],
    active: Option<u32>,
    hit: u32,
    extend: bool,
) -> (Vec<u32>, Option<u32>) {
    if !extend {
        return (vec![hit], Some(hit));
    }
    if selected.contains(&hit) {
        let remaining: Vec<u32> = selected.iter().copied().filter(|k| *k != hit).collect();
        let active = if active == Some(hit) {
            remaining.last().copied()
        } else {
            active.filter(|a| remaining.contains(a))
        };
        return (remaining, active);
    }
    let mut objects = selected.to_vec();
    objects.push(hit);
    (objects, Some(hit))
}

/// The lamp or camera whose glyph is under `at`, if any.
///
/// **The other half of drawing them.** The glyphs are drawn by the renderer
/// against its depth, but the renderer's id buffer has never heard of a lamp,
/// so a click on one landed on whatever was behind — which read as "lights
/// cannot be selected". This asks the same projection the glyphs were drawn
/// with, so what can be clicked is exactly what is on screen.
///
/// The nearest wins when two overlap, and a hidden object is never returned:
/// its glyph is not drawn, so there is nothing to have clicked.
pub fn glyph_under(
    doc: &Document,
    projector: &kreate_ui::object_glyph::Projector,
    at: egui::Pos2,
) -> Option<ObjectId> {
    let mut best: Option<(f32, ObjectId)> = None;
    for (id, object) in doc.objects() {
        if !doc.is_shown(id) {
            continue;
        }
        if matches!(object.data, kreate_scene::document::ObjectData::Armature(_)) {
            // An armature has no body of its own to hit; its bones are it.
            if let Some((distance, _)) = bone_under(doc, id, projector, at) {
                if best.is_none_or(|(d, _)| distance < d) {
                    best = Some((distance, id));
                }
            }
            continue;
        }
        if !matches!(
            object.data,
            kreate_scene::document::ObjectData::Light(_)
                | kreate_scene::document::ObjectData::Camera(_)
        ) {
            continue;
        }
        let world = doc.world_transform(id).translation;
        if !kreate_ui::object_glyph::hits(projector, world, at) {
            continue;
        }
        let distance = projector
            .project(world)
            .map(|p| p.distance(at))
            .unwrap_or(f32::MAX);
        if best.is_none_or(|(d, _)| distance < d) {
            best = Some((distance, id));
        }
    }
    best.map(|(_, id)| id)
}

/// The end of a bone at rest nearest the pointer — `(bone, 1)` for a head,
/// `(bone, 2)` for a tail — or, failing that, the bone whose segment is
/// nearest, as `(bone, 0)`: what a click in armature edit mode selects.
pub fn joint_under(
    doc: &Document,
    object: ObjectId,
    projector: &kreate_ui::object_glyph::Projector,
    at: egui::Pos2,
) -> Option<(String, u32)> {
    let (_, armature) = doc.armature_of(object)?;
    let world = doc.world_transform(object);
    let mut best: Option<(f32, String, u32)> = None;
    let consider = |d: f32, name: &str, part: u32, best: &mut Option<(f32, String, u32)>| {
        if d <= BONE_REACH && best.as_ref().is_none_or(|(b, _, _)| d < *b) {
            *best = Some((d, name.to_string(), part));
        }
    };
    for bone in &armature.bones {
        for (point, part) in [(bone.head, 1u32), (bone.tail, 2)] {
            if let Some(p) = projector.project(world.transform_point3(point)) {
                consider(p.distance(at), &bone.name, part, &mut best);
            }
        }
    }
    if best.is_none() {
        for bone in &armature.bones {
            let (Some(a), Some(b)) = (
                projector.project(world.transform_point3(bone.head)),
                projector.project(world.transform_point3(bone.tail)),
            ) else {
                continue;
            };
            let ab = b - a;
            let t = if ab.length_sq() > 0.0 {
                ((at - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            consider((a + ab * t).distance(at), &bone.name, 0, &mut best);
        }
    }
    best.map(|(_, name, part)| (name, part))
}

/// How near, in points, the pointer has to be to a bone to hit it.
const BONE_REACH: f32 = 8.0;

/// The bone of `object` nearest the pointer on screen, and how near, if any is
/// within reach: the distance from the pointer to the segment from head to
/// tail, which is what the eye aims at on a drawn bone.
pub fn bone_under(
    doc: &Document,
    object: ObjectId,
    projector: &kreate_ui::object_glyph::Projector,
    at: egui::Pos2,
) -> Option<(f32, String)> {
    let (_, armature) = doc.armature_of(object)?;
    let frames = doc.posed_bones(object);
    let mut best: Option<(f32, String)> = None;
    for (bone, frame) in armature.bones.iter().zip(&frames) {
        let tail = frame.transform_point3(kreate_math::DVec3::Y * bone.length());
        let (Some(a), Some(b)) = (
            projector.project(frame.translation),
            projector.project(tail),
        ) else {
            continue;
        };
        let ab = b - a;
        let t = if ab.length_sq() > 0.0 {
            ((at - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let distance = (a + ab * t).distance(at);
        if distance <= BONE_REACH && best.as_ref().is_none_or(|(d, _)| distance < *d) {
            best = Some((distance, bone.name.clone()));
        }
    }
    best
}

/// The tool column down the left of the viewport. Returns its rectangle so the
/// view text can start beside it rather than under it.
///
/// Every click goes out through `Dispatch`, the door a key press uses: Move
/// from the column starts the drag G starts, and Extrude runs the same
/// operator E does with the same defaults. The column teaches the keys — each
/// tooltip carries one — which is how a toolbar stops being needed.
fn draw_toolbar(
    ui: &mut Ui,
    state: &mut ShellState,
    mode: kreate_ops::operator::Mode,
    viewport: egui::Rect,
) -> egui::Rect {
    use kreate_ui::toolbar::{tool_button, Tool, WIDTH};
    let theme = kreate_ui::theme::from_ui(ui);
    // A skeleton being built has one tool of its own after the common ones:
    // Extrude, which is most of what building one is.
    let bones = mode.acts_on() == kreate_ops::mode::ActsOn::BoneJoints;
    // Sculpt mode's column is its brushes, as Blender's is: what a drag does
    // there is decided by the brush, and the transform tools mean nothing.
    let sculpt = mode.acts_on() == kreate_ops::mode::ActsOn::Sculpt;
    let tools: Vec<Tool> = if sculpt {
        Tool::SCULPT.to_vec()
    } else if mode.info().edit_tools {
        Tool::COMMON
            .iter()
            .chain(Tool::EDIT.iter())
            .copied()
            .collect()
    } else if bones {
        Tool::COMMON
            .iter()
            .copied()
            .chain([Tool::Extrude])
            .collect()
    } else {
        Tool::COMMON.to_vec()
    };
    let pad = 4.0;
    let button = WIDTH - 6.0;
    let gap = 2.0;
    // A gap between the common group and the edit group, the way Blender
    // separates its tool groups: the eye finds the boundary before it reads.
    let groups_gap = if !sculpt && tools.len() > Tool::COMMON.len() {
        8.0
    } else {
        0.0
    };
    let height = pad * 2.0 + tools.len() as f32 * (button + gap) - gap + groups_gap;
    let rect = egui::Rect::from_min_size(
        viewport.min + egui::vec2(8.0, 8.0),
        egui::vec2(WIDTH, height.min(viewport.height() - 16.0)),
    );
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return egui::Rect::NOTHING;
    }
    // Over the picture, so it takes the pointer: a click on a tool must not
    // also select what is behind it.
    state.viewport_blockers.push(rect);
    let painter = ui.painter().with_clip_rect(viewport);
    let radius = egui::CornerRadius::same(theme.radius + 1);
    painter.rect_filled(rect, radius, theme.raised.gamma_multiply(0.96));
    painter.rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, theme.border),
        egui::StrokeKind::Inside,
    );

    // The lit tool is the transform in progress, if any; otherwise Select,
    // which is what a click does when nothing else is going on. The header
    // says which gesture is running, so that is what is read.
    let active = if sculpt {
        Tool::SCULPT[(state.sculpt.brush as usize).min(Tool::SCULPT.len() - 1)]
    } else {
        state
            .modal
            .as_ref()
            .and_then(|m| m.tool)
            .unwrap_or(Tool::Select)
    };

    ui.scope_builder(
        egui::UiBuilder::new().max_rect(rect.shrink(pad)).id_salt((
            "toolbar",
            viewport.min.x as i32,
            viewport.min.y as i32,
        )),
        |ui| {
            ui.set_clip_rect(rect);
            ui.spacing_mut().item_spacing = egui::vec2(0.0, gap);
            ui.vertical(|ui| {
                for (i, tool) in tools.iter().enumerate() {
                    if !sculpt && i == Tool::COMMON.len() {
                        ui.add_space(groups_gap - gap);
                    }
                    let name = state.catalog.get(tool.label_key());
                    let op = match tool {
                        Tool::Select => None,
                        // The mode's own G, R and S: one table, the one the
                        // modal transform reads too, so the tooltip names the
                        // operator the key actually runs.
                        Tool::Move => Some(mode.info().transform[0]),
                        Tool::Rotate => Some(mode.info().transform[1]),
                        Tool::Scale => Some(mode.info().transform[2]),
                        Tool::Extrude if bones => Some("armature.extrude"),
                        Tool::Extrude => Some("mesh.extrude_region"),
                        Tool::Inset => Some("mesh.inset"),
                        Tool::Bevel => Some("mesh.bevel"),
                        Tool::LoopCut => Some("mesh.loop_cut"),
                        t if t.is_sculpt() => None,
                        _ => None,
                    };
                    let tip = match op.and_then(|id| state.command_info.get(id)) {
                        Some(info) if !info.shortcut.is_empty() => {
                            // i18n-exempt: a name and a key, both already translated
                            format!(
                                "{name}
{}",
                                info.shortcut
                            )
                        }
                        _ => name.clone(),
                    };
                    let response = if *tool == Tool::Select {
                        // Which shape a drag selects with, and a click for
                        // the next: box, lasso, circle.
                        let shape = state.tools.select_tool.min(2);
                        let names = ["select-tool-box", "select-tool-lasso", "select-tool-circle"];
                        // i18n-exempt: translated names joined
                        let tip = format!(
                            "{}\n{}",
                            state.catalog.get(names[shape as usize]),
                            state.catalog.get("select-tool-hint")
                        );
                        kreate_ui::toolbar::select_button(ui, shape, *tool == active)
                            .on_hover_text(tip)
                    } else {
                        tool_button(ui, *tool, *tool == active).on_hover_text(tip)
                    };
                    if response.clicked() {
                        match tool {
                            Tool::Select => {
                                state.tools.select_tool = (state.tools.select_tool + 1) % 3
                            }
                            Tool::Move | Tool::Rotate | Tool::Scale => {
                                if let Some(op) = op {
                                    state.queue.dispatch(op, PropertyMap::new());
                                }
                            }
                            // Through the key's door, so the drag that places
                            // the new bones starts as E starts it.
                            Tool::Extrude if bones => {
                                state.queue.dispatch("armature.extrude", PropertyMap::new())
                            }
                            // Through the key's door, so the tool drags the new
                            // faces out as E does.
                            Tool::Extrude => state
                                .queue
                                .dispatch("mesh.extrude_region", PropertyMap::new()),
                            // Like the extrude, through the keys' door: the
                            // pointer then sets how much.
                            Tool::Inset => state.queue.dispatch("mesh.inset", PropertyMap::new()),
                            Tool::Bevel => state.queue.dispatch("mesh.bevel", PropertyMap::new()),
                            Tool::LoopCut => {
                                state.queue.dispatch("mesh.loop_cut", PropertyMap::new())
                            }
                            // The brush the next stroke uses: the same
                            // setting as the header's brush menu.
                            t if t.is_sculpt() => {
                                if let Some(i) = Tool::SCULPT.iter().position(|t| t == tool) {
                                    state.sculpt.brush = i as u32;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            });
        },
    );
    rect
}

/// Sculpt mode's other settings, in the header's drop-downs as Blender keeps
/// them: symmetry, the stroke, the mask and face sets, and the topology.
fn draw_sculpt_menus(ui: &mut Ui, state: &mut ShellState) {
    use kreate_reflect::PropertyValue as V;
    let c = |key: &str| state.catalog.get(key);
    let (symmetry, stroke, mask, topology) = (
        c("sculpt-menu-symmetry"),
        c("sculpt-menu-stroke"),
        c("sculpt-menu-mask"),
        c("sculpt-menu-topology"),
    );
    let labels: Vec<String> = [
        "sculpt-mirror-x",
        "sculpt-mirror-y",
        "sculpt-mirror-z",
        "sculpt-radial",
        "sculpt-falloff",
        "sculpt-spacing",
        "sculpt-lazy",
        "sculpt-front-only",
        "sculpt-auto-smooth",
        "sculpt-alpha",
        "sculpt-pressure-strength",
        "sculpt-pressure-size",
        "sculpt-mask-clear",
        "sculpt-mask-invert",
        "sculpt-mask-fill",
        "sculpt-face-set-from-mask",
        "sculpt-show-all",
        "sculpt-show-overlay",
        "sculpt-dyntopo",
        "sculpt-detail",
        "sculpt-voxel-size",
        "sculpt-remesh",
        "sculpt-quad-faces",
        "sculpt-quad-remesh",
    ]
    .iter()
    .map(|k| c(k))
    .collect();
    let dyntopo_modes: Vec<String> = ["auto", "on", "off", "hint"]
        .iter()
        .map(|k| c(&format!("sculpt-dyntopo-{k}")))
        .collect();
    let falloffs: Vec<String> = ["smooth", "sphere", "root", "sharp", "linear", "constant"]
        .iter()
        .map(|k| c(&format!("sculpt-falloff-{k}")))
        .collect();
    let s = &mut state.sculpt;
    let mut run: Vec<(&'static str, PropertyMap)> = Vec::new();
    // Symmetry rides in the stroke's menu rather than one of its own, so
    // the header still has room for the mode's menus.
    ui.menu_button(stroke, |ui| {
        ui.strong(&symmetry);
        ui.horizontal(|ui| {
            ui.checkbox(&mut s.mirror_x, "X");
            ui.checkbox(&mut s.mirror_y, "Y");
            ui.checkbox(&mut s.mirror_z, "Z");
        })
        .response
        .on_hover_text(format!("{} / {} / {}", labels[0], labels[1], labels[2]));
        ui.horizontal(|ui| {
            ui.label(&labels[3]);
            ui.add(egui::DragValue::new(&mut s.radial).range(1..=64));
        });
        ui.separator();
        egui::Grid::new("sculpt-stroke")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label(&labels[4]);
                egui::ComboBox::from_id_salt("sculpt-falloff")
                    .selected_text(falloffs[s.falloff.min(5) as usize].as_str())
                    .show_ui(ui, |ui| {
                        for (i, f) in falloffs.iter().enumerate() {
                            ui.selectable_value(&mut s.falloff, i as u32, f.as_str());
                        }
                    });
                ui.end_row();
                ui.label(&labels[5]);
                ui.add(
                    egui::DragValue::new(&mut s.spacing)
                        .range(0.02..=2.0)
                        .speed(0.01),
                );
                ui.end_row();
                ui.label(&labels[6]);
                ui.add(
                    egui::DragValue::new(&mut s.lazy)
                        .range(0.0..=0.95)
                        .speed(0.01),
                );
                ui.end_row();
                ui.label(&labels[8]);
                ui.add(
                    egui::DragValue::new(&mut s.auto_smooth)
                        .range(0.0..=1.0)
                        .speed(0.01),
                );
                ui.end_row();
                ui.label(&labels[9]);
                egui::ComboBox::from_id_salt("sculpt-alpha")
                    .selected_text(if s.alpha.is_empty() {
                        "—"
                    } else {
                        s.alpha.as_str()
                    })
                    .show_ui(ui, |ui| {
                        // i18n-exempt: the built-in alphas' names, which scripts use too
                        for name in ["", "noise", "stripes", "dots"] {
                            let shown = if name.is_empty() { "—" } else { name };
                            ui.selectable_value(&mut s.alpha, name.to_string(), shown);
                        }
                    });
                ui.end_row();
            });
        ui.checkbox(&mut s.front_only, &labels[7]);
        ui.checkbox(&mut s.pressure_strength, &labels[10]);
        ui.checkbox(&mut s.pressure_size, &labels[11]);
    });
    ui.menu_button(mask, |ui| {
        for (k, label) in [(0u32, &labels[12]), (1, &labels[13]), (2, &labels[14])] {
            if ui.button(label).clicked() {
                let mut p = PropertyMap::new();
                p.set("action", V::Enum(k));
                run.push(("sculpt.mask", p));
                ui.close();
            }
        }
        ui.separator();
        if ui.button(&labels[15]).clicked() {
            run.push(("sculpt.face_set_from_mask", PropertyMap::new()));
            ui.close();
        }
        if ui.button(&labels[16]).clicked() {
            let mut p = PropertyMap::new();
            p.set("action", V::Enum(2));
            run.push(("sculpt.face_set_visibility", p));
            ui.close();
        }
        ui.checkbox(&mut s.show_overlay, &labels[17]);
    });
    ui.menu_button(topology, |ui| {
        ui.horizontal(|ui| {
            ui.label(&labels[18]);
            for (value, label) in [
                (DYNTOPO_AUTO, &dyntopo_modes[0]),
                (DYNTOPO_ON, &dyntopo_modes[1]),
                (DYNTOPO_OFF, &dyntopo_modes[2]),
            ] {
                ui.selectable_value(&mut s.dyntopo, value, label.as_str());
            }
        })
        .response
        .on_hover_text(&dyntopo_modes[3]);
        ui.horizontal(|ui| {
            ui.label(&labels[19]);
            ui.add(
                egui::DragValue::new(&mut s.detail)
                    .range(0.02..=1.0)
                    .speed(0.005),
            );
        });
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(&labels[20]);
            ui.add(
                egui::DragValue::new(&mut s.voxel_size)
                    .range(0.001..=10.0)
                    .speed(0.001),
            );
        });
        if ui.button(&labels[21]).clicked() {
            let mut p = PropertyMap::new();
            p.set("voxel_size", V::Float(s.voxel_size as f64));
            run.push(("sculpt.voxel_remesh", p));
            ui.close();
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(&labels[22]);
            ui.add(
                egui::DragValue::new(&mut s.quad_faces)
                    .range(10..=200_000)
                    .speed(10.0),
            );
        });
        if ui.button(&labels[23]).clicked() {
            let mut p = PropertyMap::new();
            p.set("faces", V::Int(s.quad_faces as i64));
            run.push(("sculpt.quad_remesh", p));
            ui.close();
        }
    });
    for (op, p) in run {
        state.queue.run(op, p);
    }
}

/// Sculpt mode's brush settings, in a box under the brush column: where the
/// hand already is, and on screen however narrow the viewport — the header
/// has them too, but a narrow header scrolls them out of sight.
fn draw_sculpt_box(
    ui: &mut Ui,
    state: &mut ShellState,
    mode: kreate_ops::operator::Mode,
    viewport: egui::Rect,
    toolbar: egui::Rect,
) {
    if mode.acts_on() != kreate_ops::mode::ActsOn::Sculpt || toolbar == egui::Rect::NOTHING {
        return;
    }
    let theme = kreate_ui::theme::from_ui(ui);
    let width = 132.0;
    let height = 112.0;
    let rect = egui::Rect::from_min_size(
        toolbar.left_bottom() + egui::vec2(0.0, 8.0),
        egui::vec2(width, height),
    );
    if rect.bottom() > viewport.bottom() - 8.0 || rect.right() > viewport.right() - 8.0 {
        return;
    }
    state.viewport_blockers.push(rect);
    let radius = egui::CornerRadius::same(theme.radius + 1);
    let painter = ui.painter().with_clip_rect(viewport);
    painter.rect_filled(rect, radius, theme.raised.gamma_multiply(0.96));
    painter.rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, theme.border),
        egui::StrokeKind::Inside,
    );
    ui.scope_builder(
        egui::UiBuilder::new().max_rect(rect.shrink(8.0)).id_salt((
            "sculpt-box",
            viewport.min.x as i32,
            viewport.min.y as i32,
        )),
        |ui| {
            ui.set_clip_rect(rect);
            egui::Grid::new(("sculpt-box-grid", viewport.min.x as i32))
                .num_columns(2)
                .spacing([6.0, 6.0])
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(state.catalog.get("paint-radius")).small());
                    ui.add(
                        egui::DragValue::new(&mut state.sculpt.radius)
                            .range(4.0..=400.0)
                            .speed(1.0)
                            .suffix(" px"),
                    );
                    ui.end_row();
                    ui.label(egui::RichText::new(state.catalog.get("paint-strength")).small());
                    ui.add(
                        egui::DragValue::new(&mut state.sculpt.strength)
                            .range(0.0..=1.0)
                            .speed(0.01)
                            .max_decimals(2),
                    );
                    ui.end_row();
                });
            ui.add_space(2.0);
            ui.checkbox(
                &mut state.sculpt.mirror_x,
                egui::RichText::new(state.catalog.get("sculpt-mirror-x")).small(),
            );
        },
    );
}

/// The view information in the viewport's top-left corner.
///
/// Two lines, as Blender writes them: how the scene is being looked at, and
/// what is being acted on. The first answers "why does this have no depth"
/// before it is asked; the second is the one fact a full-screen viewport has
/// no other place to say.
fn draw_view_info(
    ui: &mut Ui,
    state: &ShellState,
    doc: &Document,
    mode: kreate_ops::operator::Mode,
    l: &Labels,
    viewport: egui::Rect,
    toolbar: egui::Rect,
) {
    // Over the picture, which stays dark whichever way round the chrome is.
    let theme = kreate_ui::theme::from_ui(ui).over_viewport();
    let painter = ui.painter().with_clip_rect(viewport);
    let font = egui::TextStyle::Body.resolve(ui.style());
    let view = if state.view_status.orthographic {
        &l.view_user_orthographic
    } else {
        &l.view_user_perspective
    };
    let mode_name = state.catalog.format(mode.label_key(), TextKind::Ui, &[]);
    let active = doc
        .selection
        .active
        .and_then(|id| doc.object(id))
        .map(|o| o.name.as_str());
    // Blender's "Object : Bone" where bones are what is acted on.
    let bone = matches!(
        mode.acts_on(),
        kreate_ops::mode::ActsOn::Bones | kreate_ops::mode::ActsOn::BoneJoints
    )
    .then(|| doc.selection.active.and_then(|id| active_bone(doc, id)))
    .flatten()
    .map(|(_, name)| name);
    // i18n-exempt: composed from translated parts and the user's own names
    let second = match (active, bone) {
        (Some(name), Some(bone)) => format!("{mode_name} | {name} : {bone}"),
        (Some(name), None) => format!("{mode_name} | {name}"),
        (None, _) => mode_name,
    };
    let left = if toolbar.is_positive() {
        toolbar.max.x + 12.0
    } else {
        viewport.min.x + 12.0
    };
    let mut y = viewport.min.y + 10.0;
    let mut lines = vec![(view.clone(), theme.text), (second.clone(), theme.text_dim)];
    if state.show_stats {
        for line in statistics(doc, mode, l) {
            lines.push((line, theme.text));
        }
    }
    for (text, colour) in lines.iter().map(|(t, c)| (t.as_str(), *c)) {
        let galley = painter.layout_no_wrap(text.to_string(), font.clone(), colour);
        // A soft shadow rather than a box: this text sits over the picture and
        // a box would be a panel where the eye expects a scene. Laid out in
        // its own colour: `galley`'s colour argument only fills text laid out
        // without one, so passing it here drew the text twice, a pixel apart.
        let shadow = painter.layout_no_wrap(
            text.to_string(),
            font.clone(),
            theme.sunken.gamma_multiply(0.8),
        );
        painter.galley(egui::pos2(left + 1.0, y + 1.0), shadow, colour);
        painter.galley(egui::pos2(left, y), galley.clone(), colour);
        y += galley.size().y + 2.0;
    }
}

/// What the active object weighs: its volume, surface and middle as it is
/// evaluated, in the world and the scene's unit, and its mass at the scene's
/// density, which is edited here.
fn draw_physical_panel(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
) {
    let Some(mass) = state.mass else { return };
    let title = state.catalog.get("panel-physical");
    let open = !mass.closed;
    kreate_ui::panel::section(ui, &title, true, |ui| {
        if let Some(info) = registry.get("scene.density_set") {
            let mut params = PropertyMap::new();
            params.set(
                "density",
                kreate_reflect::PropertyValue::Float(doc.design.density),
            );
            if kreate_ui::property_editor(ui, &state.catalog, info.id, info.schema, &mut params) {
                let dragging = ui.ctx().input(|i| i.pointer.any_down());
                send_edit(state, "scene.density_set", params, dragging);
            }
        }
        egui::Grid::new("physical").num_columns(2).show(ui, |ui| {
            let c = mass.centroid;
            let rows = [
                (
                    "physical-volume",
                    kreate_reflect::format_volume(mass.volume.abs()),
                ),
                ("physical-area", kreate_reflect::format_area(mass.area)),
                (
                    "physical-mass",
                    kreate_reflect::format_mass(mass.volume.abs() * doc.design.density),
                ),
                (
                    "physical-centroid",
                    format!(
                        "{}, {}, {}",
                        kreate_reflect::format_length(c.x),
                        kreate_reflect::format_length(c.y),
                        kreate_reflect::format_length(c.z)
                    ),
                ),
            ];
            for (key, value) in rows {
                ui.label(state.catalog.get(key));
                // i18n-exempt: measured values
                ui.label(value);
                ui.end_row();
            }
        });
        if open {
            let theme = kreate_ui::theme::from_ui(ui);
            ui.add(
                egui::Label::new(
                    egui::RichText::new(state.catalog.get("physical-open")).color(theme.accent),
                )
                .wrap(),
            );
        }
    });
}

/// Mesh Check: is the active mesh fit to print. The thinnest wall allowed,
/// a button that checks, and what it found with a button to select each.
fn draw_mesh_check_panel(ui: &mut Ui, state: &mut ShellState, doc: &Document, object: ObjectId) {
    if doc.object(object).and_then(|o| o.mesh()).is_none() {
        return;
    }
    let title = state.catalog.get("panel-mesh-check");
    kreate_ui::panel::section(ui, &title, false, |ui| {
        let display = kreate_ui::property_editor::UnitDisplay::of(kreate_reflect::Unit::Length);
        ui.horizontal(|ui| {
            ui.label(state.catalog.get("check-thickness"));
            let mut shown = state.check_thickness * display.to_display;
            if ui
                .add(
                    egui::DragValue::new(&mut shown)
                        .speed(0.01 * display.to_display * 0.01)
                        .suffix(display.suffix)
                        .max_decimals(display.decimals)
                        .range(0.0..=f64::MAX),
                )
                .changed()
            {
                state.check_thickness = display.from_display(shown);
            }
        });
        let thickness = state.check_thickness;
        if ui.button(state.catalog.get("check-run")).clicked() {
            let mut p = PropertyMap::new();
            p.set("thickness", kreate_reflect::PropertyValue::Float(thickness));
            state.queue.dispatch("app.mesh_check", p);
        }
        let Some((_, counts)) = state.mesh_check.filter(|(o, _)| *o == object) else {
            return;
        };
        if counts.iter().all(|c| *c == 0) {
            ui.label(state.catalog.get("check-clean"));
            return;
        }
        egui::Grid::new("mesh-check").num_columns(3).show(ui, |ui| {
            for (i, key) in [
                "check-open",
                "check-non-manifold",
                "check-flipped",
                "check-degenerate",
                "check-loose",
                "check-intersecting",
                "check-thin",
            ]
            .iter()
            .enumerate()
            {
                ui.label(state.catalog.get(key));
                // i18n-exempt: a count
                ui.label(counts[i].to_string());
                if counts[i] > 0 && ui.small_button(state.catalog.get("check-select")).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("issue", kreate_reflect::PropertyValue::Enum(i as u32));
                    p.set("thickness", kreate_reflect::PropertyValue::Float(thickness));
                    state.queue.dispatch("app.mesh_check_select", p);
                }
                ui.end_row();
            }
        });
    });
}

/// The work plane: a grid around the 3D cursor in the plane it is turned to,
/// its X and Y drawn in the axis colours. The spacing is a power of ten that
/// suits how far away it is, so it reads at any zoom.
fn draw_work_plane(
    painter: &egui::Painter,
    projector: &kreate_ui::object_glyph::Projector,
    at: kreate_math::DVec3,
    turn: kreate_math::DQuat,
) {
    const HALF: i32 = 10;
    let distance = (at - projector.eye).length().max(1e-6);
    let step = 10f64.powf((distance / 4.0).log10().floor());
    let (x, y) = (turn * kreate_math::DVec3::X, turn * kreate_math::DVec3::Y);
    let reach = step * HALF as f64;
    let theme_line = egui::Color32::from_rgba_unmultiplied(200, 200, 210, 60);
    let line = |a: kreate_math::DVec3, b: kreate_math::DVec3, colour: egui::Color32, width: f32| {
        if let (Some(a), Some(b)) = (projector.project(a), projector.project(b)) {
            painter.line_segment([a, b], egui::Stroke::new(width, colour));
        }
    };
    for i in -HALF..=HALF {
        let offset = step * i as f64;
        if i == 0 {
            continue;
        }
        line(
            at + x * offset - y * reach,
            at + x * offset + y * reach,
            theme_line,
            1.0,
        );
        line(
            at + y * offset - x * reach,
            at + y * offset + x * reach,
            theme_line,
            1.0,
        );
    }
    line(
        at - x * reach,
        at + x * reach,
        egui::Color32::from_rgb(220, 80, 80),
        1.5,
    );
    line(
        at - y * reach,
        at + y * reach,
        egui::Color32::from_rgb(110, 200, 90),
        1.5,
    );
}

/// Each sketch's distance and angle constraints, drawn as dimensions on its
/// points in the world. Not for a sketch drawn with the sketch tools: those
/// show their dimensions while they are being edited, in the sketch's own
/// view, and keep out of the way of the solids made from them otherwise.
fn draw_sketch_dimensions(
    painter: &egui::Painter,
    projector: &kreate_ui::object_glyph::Projector,
    doc: &Document,
) {
    for (id, object) in doc.objects() {
        if object.sketch.is_empty() || !object.sketch_lines.is_empty() || !doc.is_shown(id) {
            continue;
        }
        let Some(mesh) = object.mesh().and_then(|m| doc.mesh(m)) else {
            continue;
        };
        let world = doc.world_transform(id);
        let at = |v: u32| {
            mesh.positions()
                .get(v as usize)
                .map(|p| world.transform_point3(p.as_dvec3()))
        };
        for c in &object.sketch {
            let points: Option<Vec<kreate_math::DVec3>> = match c {
                kreate_scene::sketch::Constraint::Distance { a, b, .. } => {
                    [*a, *b].iter().map(|v| at(*v)).collect()
                }
                kreate_scene::sketch::Constraint::Angle { a, b, c, d, .. } => {
                    // Drawn at the second line's start: its arms the two lines.
                    let (pa, pb, pc, pd) = (at(*a), at(*b), at(*c), at(*d));
                    match (pa, pb, pc, pd) {
                        (Some(pa), Some(pb), Some(pc), Some(pd)) => {
                            Some(vec![pc + (pb - pa), pc, pd])
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(points) = points {
                draw_measurement(painter, projector, &points, false);
            }
        }
    }
}

/// A dimension over the picture: the line between two points and how long it
/// is, or the two arms of an angle and how wide. `live` is the one the
/// Measure tool is taking, drawn brighter.
fn draw_measurement(
    painter: &egui::Painter,
    projector: &kreate_ui::object_glyph::Projector,
    points: &[kreate_math::DVec3],
    live: bool,
) {
    let colour = if live {
        egui::Color32::from_rgb(255, 214, 90)
    } else {
        egui::Color32::from_rgb(235, 170, 50)
    };
    let screen: Vec<egui::Pos2> = points
        .iter()
        .filter_map(|p| projector.project(*p))
        .collect();
    if screen.len() != points.len() || screen.is_empty() {
        return;
    }
    let stroke = egui::Stroke::new(1.5, colour);
    for pair in screen.windows(2) {
        painter.line_segment([pair[0], pair[1]], stroke);
    }
    for p in &screen {
        painter.circle_filled(*p, 3.0, colour);
    }
    // i18n-exempt: a measured length or angle
    let (text, at) = match points {
        [a, b] => (
            kreate_reflect::format_length(a.distance(*b)),
            screen[0] + (screen[1] - screen[0]) * 0.5,
        ),
        [a, b, c] => {
            let (u, v) = (*a - *b, *c - *b);
            if u.length() == 0.0 || v.length() == 0.0 {
                return;
            }
            (
                format!("{:.2}°", u.angle_between(v).to_degrees()),
                screen[1],
            )
        }
        _ => return,
    };
    let font = egui::FontId::proportional(13.0);
    let galley = painter.layout_no_wrap(text, font, egui::Color32::BLACK);
    let rect = egui::Rect::from_center_size(
        at + egui::vec2(0.0, -12.0),
        galley.size() + egui::vec2(8.0, 4.0),
    );
    painter.rect_filled(rect, 3.0, colour);
    painter.galley(
        rect.min + egui::vec2(4.0, 2.0),
        galley,
        egui::Color32::BLACK,
    );
}

/// Blender's Statistics overlay: how many objects, and how many vertices,
/// edges, faces and triangles — selected of the total in edit mode.
fn statistics(doc: &Document, mode: kreate_ops::operator::Mode, l: &Labels) -> Vec<String> {
    use kreate_mesh::attribute::names;
    let count = |mesh: &kreate_mesh::MeshData, layer: &str| {
        mesh.attributes()
            .get_bool(layer)
            .map(|s| s.iter().filter(|x| **x).count())
            .unwrap_or(0)
    };
    let triangles = |mesh: &kreate_mesh::MeshData| -> usize {
        (0..mesh.num_faces())
            .map(|f| mesh.face_verts(f).len().saturating_sub(2))
            .sum()
    };
    // i18n-exempt: a translated noun and numbers
    let row = |name: &str, value: String| format!("{name}  {value}");
    if mode.acts_on() == kreate_ops::mode::ActsOn::MeshElements {
        let (mut v, mut e, mut f, mut t) = ([0usize; 2], [0usize; 2], [0usize; 2], 0usize);
        // What a designer reads off a selection: how long its edges are, and
        // how much surface its faces cover — in the world, in the scene's unit.
        let (mut length, mut area) = (0.0f64, 0.0f64);
        for target in kreate_ops::ops::mesh::edit_targets(doc) {
            let Some(mesh) = doc.mesh(target.mesh) else {
                continue;
            };
            let at = |v: u32| {
                target
                    .to_world
                    .transform_point3(mesh.positions()[v as usize].as_dvec3())
            };
            if let Some(sel) = mesh.attributes().get_bool(names::SELECT_EDGE) {
                for (edge, on) in mesh.edges().iter().zip(sel) {
                    if *on {
                        length += at(edge[0]).distance(at(edge[1]));
                    }
                }
            }
            if let Some(sel) = mesh.attributes().get_bool(names::SELECT_FACE) {
                for (face, on) in sel.iter().enumerate() {
                    if !*on {
                        continue;
                    }
                    let ring: Vec<_> = mesh
                        .face_verts(face as u32)
                        .iter()
                        .map(|v| at(*v))
                        .collect();
                    let mut n = kreate_math::DVec3::ZERO;
                    for i in 0..ring.len() {
                        n += ring[i].cross(ring[(i + 1) % ring.len()]);
                    }
                    area += n.length() * 0.5;
                }
            }
            v[0] += count(mesh, names::SELECT_VERT);
            v[1] += mesh.num_verts() as usize;
            e[0] += count(mesh, names::SELECT_EDGE);
            e[1] += mesh.num_edges() as usize;
            f[0] += count(mesh, names::SELECT_FACE);
            f[1] += mesh.num_faces() as usize;
            t += triangles(mesh);
        }
        let mut rows = vec![
            row(&l.stats_vertices, format!("{}/{}", v[0], v[1])),
            row(&l.stats_edges, format!("{}/{}", e[0], e[1])),
            row(&l.stats_faces, format!("{}/{}", f[0], f[1])),
            row(&l.stats_triangles, t.to_string()),
        ];
        if e[0] > 0 {
            rows.push(row(&l.stats_length, kreate_reflect::format_length(length)));
        }
        if f[0] > 0 {
            rows.push(row(&l.stats_area, kreate_reflect::format_area(area)));
        }
        return rows;
    }
    let (mut v, mut e, mut f, mut t, mut objects) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for (id, object) in doc.objects() {
        if !doc.is_shown(id) {
            continue;
        }
        objects += 1;
        if let Some(mesh) = object.mesh().and_then(|m| doc.mesh(m)) {
            v += mesh.num_verts() as usize;
            e += mesh.num_edges() as usize;
            f += mesh.num_faces() as usize;
            t += triangles(mesh);
        }
    }
    vec![
        row(
            &l.stats_objects,
            format!("{}/{}", doc.selection.objects.len(), objects),
        ),
        row(&l.stats_vertices, v.to_string()),
        row(&l.stats_edges, e.to_string()),
        row(&l.stats_faces, f.to_string()),
        row(&l.stats_triangles, t.to_string()),
    ]
}

/// How much of a long text value a script line shows: a paste carries
/// megabytes, and the line is for reading.
pub const SCRIPT_PREVIEW: usize = 120;

/// The console: the history as script lines, what was typed, and the answers.
///
/// The history's lines are read fresh each frame rather than copied into the
/// scrollback, so that undo takes a line out of the log the way it takes the
/// edit out of the document — the two say the same thing at every moment.
/// What was typed, and what the application said back, is the shell's own.
fn draw_console(ui: &mut Ui, state: &mut ShellState, history: &History, l: &Labels) {
    use kreate_ui::console::{ConsoleLine, LineKind};
    let mut lines: Vec<ConsoleLine> = history
        .entries()
        .iter()
        .map(|entry| ConsoleLine {
            text: entry.to_script_preview(SCRIPT_PREVIEW),
            kind: LineKind::History,
        })
        .collect();
    lines.extend(state.console_log.iter().cloned());
    let text = kreate_ui::console::ConsoleText {
        placeholder: l.console_placeholder.clone(),
        empty: l.console_empty.clone(),
        same_error: state.catalog.get("console-same-error"),
    };
    let mut input = std::mem::take(&mut state.console_input);
    kreate_ui::console::console(ui, &text, &lines, &mut input, &mut state.queue);
    state.console_input = input;
}

/// Help's two windows: the contact form and About.
fn draw_help_windows(ctx: &egui::Context, state: &mut ShellState, l: &Labels) {
    let text = kreate_ui::contact::ContactText {
        title: l.contact_title.clone(),
        intro: l.contact_intro.clone(),
        kinds: [
            l.contact_kind_bug.clone(),
            l.contact_kind_feature.clone(),
            l.contact_kind_question.clone(),
            l.contact_kind_other.clone(),
        ],
        name: l.contact_name.clone(),
        email: l.contact_email.clone(),
        email_hint: l.contact_email_hint.clone(),
        message_hint: l.contact_message_hint.clone(),
        details: l.contact_details.clone(),
        send: l.contact_send.clone(),
        close: l.contact_close.clone(),
        sending: l.contact_sending.clone(),
        sent: l.contact_sent.clone(),
    };
    // i18n-exempt: a version and a platform
    let details = format!(
        "Kreate {} · {}",
        crate::version::describe(),
        crate::version::platform()
    );
    kreate_ui::contact::contact_window(ctx, &text, &mut state.contact, &details, &mut state.queue);
    let t = |key: &str| state.catalog.get(key);
    let collab = kreate_ui::collab::CollabText {
        title: t("collab-title"),
        intro: t("collab-intro"),
        name: t("collab-name"),
        address: t("collab-address"),
        address_hint: t("collab-address-hint"),
        key: t("collab-key"),
        key_hint: t("collab-key-hint"),
        port: t("collab-port"),
        host: t("collab-host"),
        join: t("collab-join"),
        join_warning: t("collab-join-warning"),
        need_name: t("collab-need-name"),
        need_address: t("collab-need-address"),
        leave: t("collab-leave"),
        connecting: t("collab-connecting"),
        hosting_at: t("collab-hosting-at"),
        joined_at: t("collab-joined-at"),
        people: t("collab-people"),
        alone: t("collab-alone"),
        you: t("collab-you"),
        editing: t("collab-editing"),
        sending: t("collab-sending"),
    };
    kreate_ui::collab::collab_window(
        ctx,
        &collab,
        &mut state.collab,
        state.collab_view.as_ref(),
        &mut state.queue,
    );
    let about = kreate_ui::contact::AboutText {
        title: l.about_title.clone(),
        version: l.about_version.clone(),
        close: l.contact_close.clone(),
    };
    kreate_ui::contact::about_window(
        ctx,
        &about,
        &mut state.about_open,
        &crate::version::describe(),
        &crate::version::platform(),
    );
    draw_render_result(ctx, state, l);
    draw_fits(ctx, state, l);
}

/// The words of the fit calculator, from the Mechanical menu's catalog.
const FITS_KEYS: [&str; 10] = [
    "fits-size",
    "fits-hole",
    "fits-shaft",
    "fits-limits",
    "fits-clearance",
    "fits-transition",
    "fits-interference",
    "fits-max",
    "fits-min",
    "fits-none",
];

/// The fit calculator's choices, kept between openings.
#[derive(Clone, Debug, PartialEq)]
pub struct FitsDialog {
    pub open: bool,
    /// The nominal size, mm.
    pub size: f64,
    /// Indices into `kreate_mesh::fits::LETTERS`.
    pub hole_letter: usize,
    pub hole_grade: u32,
    pub shaft_letter: usize,
    pub shaft_grade: u32,
}

impl Default for FitsDialog {
    fn default() -> Self {
        // H7/g6 at 20 mm: the sliding fit every handbook starts from.
        Self {
            open: false,
            size: 20.0,
            hole_letter: 3,
            hole_grade: 7,
            shaft_letter: 2,
            shaft_grade: 6,
        }
    }
}

/// ISO 286: a hole and a shaft by letter and grade at a size, their limits
/// and what fit they make.
fn draw_fits(ctx: &egui::Context, state: &mut ShellState, l: &Labels) {
    use kreate_mesh::fits;
    if !state.fits.open {
        return;
    }
    let mut d = state.fits.clone();
    let mut open = true;
    egui::Window::new(l.m("mech-fits"))
        .id(egui::Id::new("fits-calculator"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_pos(ctx.content_rect().center())
        .show(ctx, |ui| {
            egui::Grid::new("fits-grid").num_columns(2).show(ui, |ui| {
                ui.label(l.m("fits-size"));
                ui.add(
                    egui::DragValue::new(&mut d.size)
                        .range(0.1..=500.0)
                        .speed(0.1)
                        .suffix(" mm"),
                );
                ui.end_row();
                for (name, letter, grade, upper) in [
                    ("fits-hole", &mut d.hole_letter, &mut d.hole_grade, true),
                    ("fits-shaft", &mut d.shaft_letter, &mut d.shaft_grade, false),
                ] {
                    ui.label(l.m(name));
                    ui.horizontal(|ui| {
                        let show = |i: usize| {
                            if upper {
                                fits::LETTERS[i].to_uppercase()
                            } else {
                                fits::LETTERS[i].to_string()
                            }
                        };
                        egui::ComboBox::from_id_salt((name, "letter"))
                            .width(48.0)
                            .selected_text(show(*letter))
                            .show_ui(ui, |ui| {
                                for i in 0..fits::LETTERS.len() {
                                    ui.selectable_value(letter, i, show(i));
                                }
                            });
                        egui::ComboBox::from_id_salt((name, "grade"))
                            .width(40.0)
                            .selected_text(grade.to_string())
                            .show_ui(ui, |ui| {
                                for g in fits::GRADES {
                                    ui.selectable_value(grade, g, g.to_string());
                                }
                            });
                    });
                    ui.end_row();
                }
            });
            ui.separator();
            let hl = fits::LETTERS[d.hole_letter].to_uppercase();
            let sl = fits::LETTERS[d.shaft_letter];
            let Some(f) = fits::fit(d.size, &hl, d.hole_grade, sl, d.shaft_grade) else {
                ui.label(l.m("fits-none"));
                return;
            };
            // i18n-exempt: numbers and the standard's own symbols
            let um = |v: f64| format!("{v:+.1} µm");
            // i18n-exempt: the standard's own designation
            ui.heading(format!(
                "ø{} {}{}/{}{}",
                d.size, hl, d.hole_grade, sl, d.shaft_grade
            ));
            egui::Grid::new("fits-result")
                .num_columns(3)
                .show(ui, |ui| {
                    ui.label("");
                    ui.label(l.m("fits-hole"));
                    ui.label(l.m("fits-shaft"));
                    ui.end_row();
                    ui.label(l.m("fits-limits"));
                    // i18n-exempt: numbers
                    ui.label(format!("{} / {}", um(f.hole.upper), um(f.hole.lower)));
                    // i18n-exempt: numbers
                    ui.label(format!("{} / {}", um(f.shaft.upper), um(f.shaft.lower)));
                    ui.end_row();
                    // i18n-exempt: a unit
                    ui.label("mm");
                    // i18n-exempt: numbers
                    ui.label(format!(
                        "{:.3} - {:.3}",
                        d.size + f.hole.lower / 1000.0,
                        d.size + f.hole.upper / 1000.0
                    ));
                    // i18n-exempt: numbers
                    ui.label(format!(
                        "{:.3} - {:.3}",
                        d.size + f.shaft.lower / 1000.0,
                        d.size + f.shaft.upper / 1000.0
                    ));
                    ui.end_row();
                });
            ui.separator();
            // i18n-exempt: a translated label and a number
            ui.label(format!("{} {}", l.m("fits-max"), um(f.max_clearance)));
            // i18n-exempt: a translated label and a number
            ui.label(format!("{} {}", l.m("fits-min"), um(f.min_clearance)));
            let kind = match f.kind {
                fits::FitKind::Clearance => "fits-clearance",
                fits::FitKind::Transition => "fits-transition",
                fits::FitKind::Interference => "fits-interference",
            };
            ui.strong(l.m(kind));
        });
    d.open = open;
    state.fits = d;
}

/// A finished render, shown as Blender's Render window shows it: the
/// picture, and where it was written.
pub struct RenderResultView {
    pub texture: egui::TextureHandle,
    pub size: (u32, u32),
    /// What happened: where the file went, or why it did not.
    pub message: String,
}

fn draw_render_result(ctx: &egui::Context, state: &mut ShellState, l: &Labels) {
    let Some(result) = &state.render_result else {
        return;
    };
    let mut open = true;
    let screen = ctx.content_rect();
    // As big as fits in most of the window, never bigger than the picture.
    let fit = ((screen.width() * 0.8) / result.size.0 as f32)
        .min((screen.height() * 0.75) / result.size.1 as f32)
        .min(1.0);
    let shown = egui::vec2(result.size.0 as f32 * fit, result.size.1 as f32 * fit);
    egui::Window::new(&l.render_result_title)
        .id(egui::Id::new("render-result"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(screen.center())
        .show(ctx, |ui| {
            ui.add(egui::Image::new(&result.texture).fit_to_exact_size(shown));
            // i18n-exempt: composed by the application from translated parts and a path
            ui.label(&result.message);
        });
    if !open {
        state.render_result = None;
    }
}

/// One physics panel's fields: an operator's parameters as they stand,
/// edited in place, each change sent through the operator — for the active
/// object only.
fn physics_fields(
    ui: &mut Ui,
    state: &mut ShellState,
    registry: &OperatorRegistry,
    op: &str,
    mut params: PropertyMap,
    keep: impl Fn(&str) -> bool,
) {
    let Some(info) = registry.get(op) else { return };
    // Its own id scope: several panels have a "Kind", and egui names a
    // drop-down by its label.
    ui.push_id(op, |ui| {
        if kreate_ui::property_editor::property_editor_where(
            ui,
            &state.catalog,
            info.id,
            info.schema,
            &mut params,
            keep,
        ) {
            // The tab shows the active object, so it sets the active
            // object: the same values sent to the whole selection would
            // overwrite every other object's own — and switch off the
            // physics of those the active one had none of.
            if info.schema.iter().any(|d| d.name == "active_only") {
                params.set("active_only", kreate_reflect::PropertyValue::Bool(true));
            }
            let dragging = ui.ctx().input(|i| i.pointer.any_down());
            send_edit(state, op, params, dragging);
        }
    });
}

/// A simulation a mesh can be part of, as the Physics tab's buttons offer
/// them — Blender's row of physics types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sim {
    Collision,
    Cloth,
    SoftBody,
    RigidBody,
    FluidDomain,
    FluidFlow,
    Particles,
    Hair,
}

impl Sim {
    /// In the order the buttons stand and the sections open.
    const ALL: [Sim; 8] = [
        Sim::Collision,
        Sim::Cloth,
        Sim::SoftBody,
        Sim::RigidBody,
        Sim::FluidDomain,
        Sim::FluidFlow,
        Sim::Particles,
        Sim::Hair,
    ];

    fn key(self) -> &'static str {
        match self {
            Sim::Collision => "physics-collision",
            Sim::Cloth => "physics-cloth",
            Sim::SoftBody => "physics-soft-body",
            Sim::RigidBody => "physics-rigid",
            Sim::FluidDomain => "physics-fluid-domain",
            Sim::FluidFlow => "physics-fluid-flow",
            Sim::Particles => "physics-particles",
            Sim::Hair => "physics-hair",
        }
    }

    fn on(self, p: &kreate_scene::physics::ObjectPhysics) -> bool {
        use kreate_scene::physics::{Fluid, ParticleKind, SoftKind};
        match self {
            Sim::Collision => p.collision.is_some(),
            Sim::Cloth => p.soft.as_ref().is_some_and(|s| s.kind == SoftKind::Cloth),
            Sim::SoftBody => p.soft.as_ref().is_some_and(|s| s.kind == SoftKind::Soft),
            Sim::RigidBody => p.rigid.is_some(),
            Sim::FluidDomain => matches!(p.fluid, Some(Fluid::Domain(_))),
            Sim::FluidFlow => matches!(p.fluid, Some(Fluid::Flow(_))),
            Sim::Particles => p.particles.iter().any(|s| s.kind == ParticleKind::Emitter),
            Sim::Hair => p.particles.iter().any(|s| s.kind == ParticleKind::Hair),
        }
    }

    /// What this cannot be used with: both would move the mesh, or one is
    /// a container the other would sit inside of.
    fn clashes_with(self, other: Sim) -> bool {
        use Sim::*;
        let pair = |a: Sim, b: Sim| (self == a && other == b) || (self == b && other == a);
        pair(RigidBody, Cloth)
            || pair(RigidBody, SoftBody)
            || pair(Cloth, SoftBody)
            || pair(FluidDomain, RigidBody)
            || pair(FluidDomain, Cloth)
            || pair(FluidDomain, SoftBody)
            || pair(FluidDomain, Collision)
            || pair(FluidDomain, FluidFlow)
            || pair(FluidDomain, Particles)
            || pair(FluidDomain, Hair)
    }

    /// Whether it is simulated over time, and so has a bake.
    fn bakes(self) -> bool {
        self != Sim::Collision
    }
}

/// Switches a simulation on the active mesh on or off, through the
/// operators the panels have always used.
fn set_sim(state: &mut ShellState, p: &kreate_scene::physics::ObjectPhysics, sim: Sim, on: bool) {
    use kreate_ops::ops::physics as ph;
    use kreate_reflect::OpParams as _;
    use kreate_reflect::PropertyValue as V;
    let only = |mut m: PropertyMap| {
        m.set("active_only", V::Bool(true));
        m
    };
    match sim {
        Sim::Collision => {
            let mut m = ph::collision_params(p.collision.as_ref()).to_map();
            m.set("enabled", V::Bool(on));
            state.queue.run("physics.collision_set", only(m));
        }
        Sim::RigidBody => {
            let mut m = ph::rigid_params(p.rigid.as_ref()).to_map();
            m.set("enabled", V::Bool(on));
            state.queue.run("physics.rigid_body_set", only(m));
        }
        Sim::Cloth | Sim::SoftBody => {
            let mut m = ph::soft_params(p.soft.as_ref()).to_map();
            let kind = match (on, sim) {
                (false, _) => 0,
                (true, Sim::Cloth) => 1,
                _ => 2,
            };
            m.set("kind", V::Enum(kind));
            state.queue.run("physics.soft_set", only(m));
        }
        Sim::FluidDomain | Sim::FluidFlow => {
            let mut m = ph::fluid_params(p.fluid.as_ref()).to_map();
            let role = match (on, sim) {
                (false, _) => 0,
                (true, Sim::FluidDomain) => 1,
                _ => 2,
            };
            m.set("role", V::Enum(role));
            state.queue.run("physics.fluid_set", only(m));
        }
        Sim::Particles | Sim::Hair => {
            let kind = if sim == Sim::Hair {
                kreate_scene::physics::ParticleKind::Hair
            } else {
                kreate_scene::physics::ParticleKind::Emitter
            };
            if on {
                let mut m = PropertyMap::new();
                m.set("kind", V::Enum(u32::from(sim == Sim::Hair)));
                state.queue.run("physics.particles_add", m);
            } else {
                // Every system of this kind, last first so the numbers hold.
                for (i, s) in p.particles.iter().enumerate().rev() {
                    if s.kind == kind {
                        let mut m = PropertyMap::new();
                        m.set("index", V::Int(i as i64));
                        state.queue.run("physics.particles_remove", m);
                    }
                }
            }
        }
    }
}

/// The bake buttons, inside a simulation's own section: the bake runs every
/// simulation in the scene, as it always has, and says so. Bake to
/// Keyframes is the rigid body's alone.
fn bake_row(ui: &mut Ui, state: &mut ShellState, doc: &Document, only: u32, with_keys: bool) {
    let cache = &doc.physics_cache;
    let mut which = PropertyMap::new();
    which.set("only", kreate_reflect::PropertyValue::Enum(only));
    ui.horizontal(|ui| {
        if let Some((done, total)) = state.bake_progress {
            // A bake is running: how far, and a way to stop it.
            ui.add(
                egui::ProgressBar::new(done as f32 / total.max(1) as f32)
                    .desired_width(140.0)
                    .text(format!("{done} / {total}")),
            );
            if ui
                .button(state.catalog.get("physics-bake-cancel"))
                .clicked()
            {
                state
                    .queue
                    .dispatch("physics.bake_cancel", PropertyMap::new());
            }
            return;
        }
        let (bake, hint) = if only == 0 {
            ("physics-bake-all", "physics-bake-all-hint")
        } else {
            ("physics-bake", "physics-bake-one-hint")
        };
        if ui
            .button(state.catalog.get(bake))
            .on_hover_text(state.catalog.get(hint))
            .clicked()
        {
            state.queue.dispatch("physics.bake_start", which.clone());
        }
        if ui
            .add_enabled(
                !cache.is_empty(),
                egui::Button::new(state.catalog.get("physics-free-bake")),
            )
            .clicked()
        {
            state.queue.run("physics.free_bake", which.clone());
        }
        // Said before the click: the rigid bodies stop being rigid bodies,
        // which otherwise looks like the settings going missing.
        if with_keys
            && ui
                .add_enabled(
                    !cache.rigid.is_empty(),
                    egui::Button::new(state.catalog.get("physics-bake-to-keys")),
                )
                .on_hover_text(state.catalog.get("op-physics-bake-to-keys-desc"))
                .clicked()
        {
            state.queue.run("physics.bake_to_keys", PropertyMap::new());
        }
    });
    if cache.is_empty() {
        ui.weak(state.catalog.get("physics-not-baked"));
    } else {
        let frames = cache
            .rigid
            .first()
            .map(|t| t.frames.len())
            .or(cache.soft.first().map(|t| t.frames.len()))
            .or(cache.particles.first().map(|t| t.frames.len()))
            .unwrap_or(1);
        let (start, end) = (
            cache.start.to_string(),
            (cache.start + frames as i32 - 1).to_string(),
        );
        ui.weak(state.catalog.get_args(
            "physics-baked",
            &[("start", start.as_str()), ("end", end.as_str())],
        ));
    }
}

/// Blender's Physics tab: a row of buttons, one a simulation, lit for those
/// the active mesh is part of — pressing one switches it on or off — and
/// below, a section for each that is on, with its settings and its bake.
/// What cannot go together is not offered together. The scene's gravity
/// and range close the tab.
fn draw_physics_tab(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
) {
    use kreate_ops::ops::physics as ph;
    use kreate_reflect::OpParams as _;
    use kreate_reflect::PropertyValue as V;
    ui.heading(state.catalog.get("tab-physics"));
    let active = doc.selection.active.and_then(|id| doc.object(id));
    match active {
        None => {
            ui.weak(state.catalog.get("physics-nothing-selected"));
        }
        // Simulations act on meshes: an empty, a camera, a light or an
        // armature has none to take part in.
        Some(o) if o.mesh().is_none() => {
            ui.weak(state.catalog.get("physics-not-applicable"));
        }
        Some(o) => {
            let p = &o.physics;
            let on: Vec<Sim> = Sim::ALL.into_iter().filter(|s| s.on(p)).collect();
            // The buttons: lit when on; pressed again, off. One that clashes
            // with what is on is not offered, and says why.
            ui.horizontal_wrapped(|ui| {
                for sim in Sim::ALL {
                    let lit = on.contains(&sim);
                    let blocker = on.iter().find(|o| sim.clashes_with(**o)).copied();
                    let button = egui::Button::selectable(lit, state.catalog.get(sim.key()));
                    let response = ui.add_enabled(lit || blocker.is_none(), button);
                    let response = match blocker {
                        Some(b) if !lit => {
                            let other = state.catalog.get(b.key());
                            response.on_disabled_hover_text(state.catalog.format(
                                "physics-clash",
                                kreate_i18n::TextKind::Tooltip,
                                &[("other", other.as_str())],
                            ))
                        }
                        _ => response.on_hover_text(state.catalog.get(if lit {
                            "physics-switch-off"
                        } else {
                            "physics-switch-on"
                        })),
                    };
                    if response.clicked() {
                        set_sim(state, p, sim, !lit);
                    }
                }
            });
            if on.is_empty() {
                ui.weak(state.catalog.get("physics-none-on"));
            }
            ui.add_space(4.0);
            // A section for each that is on.
            let off_hint = state.catalog.get("physics-switch-off");
            for sim in on.iter().copied() {
                let title = state.catalog.get(sim.key());
                let (_, closed) = ui
                    .push_id(("sim", sim.key()), |ui| {
                        kreate_ui::panel::section_closable(ui, &title, true, &off_hint, |ui| {
                            match sim {
                                Sim::Collision => physics_fields(
                                    ui,
                                    state,
                                    registry,
                                    "physics.collision_set",
                                    ph::collision_params(p.collision.as_ref()).to_map(),
                                    |n| n != "enabled",
                                ),
                                Sim::RigidBody => physics_fields(
                                    ui,
                                    state,
                                    registry,
                                    "physics.rigid_body_set",
                                    ph::rigid_params(p.rigid.as_ref()).to_map(),
                                    |n| n != "enabled",
                                ),
                                Sim::Cloth | Sim::SoftBody => {
                                    let soft = ph::soft_params(p.soft.as_ref());
                                    let kind = soft.kind;
                                    physics_fields(
                                        ui,
                                        state,
                                        registry,
                                        "physics.soft_set",
                                        soft.to_map(),
                                        |n| n != "kind" && ph::soft_param_applies(kind, n),
                                    );
                                }
                                Sim::FluidDomain | Sim::FluidFlow => {
                                    let fluid = ph::fluid_params(p.fluid.as_ref());
                                    physics_fields(
                                        ui,
                                        state,
                                        registry,
                                        "physics.fluid_set",
                                        fluid.to_map(),
                                        |n| n != "role" && ph::fluid_param_applies(&fluid, n),
                                    );
                                }
                                Sim::Particles | Sim::Hair => {
                                    let kind = if sim == Sim::Hair {
                                        kreate_scene::physics::ParticleKind::Hair
                                    } else {
                                        kreate_scene::physics::ParticleKind::Emitter
                                    };
                                    for (i, s) in p.particles.iter().enumerate() {
                                        if s.kind != kind {
                                            continue;
                                        }
                                        ui.push_id(("particles", i), |ui| {
                                            ui.horizontal(|ui| {
                                                // i18n-exempt: the system's name is the user's own text
                                                ui.strong(&s.name);
                                                if ui
                                                    .small_button(REMOVE_GLYPH)
                                                    .on_hover_text(
                                                        state.catalog.get("physics-remove-system"),
                                                    )
                                                    .clicked()
                                                {
                                                    let mut m = PropertyMap::new();
                                                    m.set("index", V::Int(i as i64));
                                                    state.queue.run("physics.particles_remove", m);
                                                }
                                            });
                                            let params = ph::particle_params(doc, i as u32, s);
                                            let kind = params.kind;
                                            if let Some(info) = registry.get("physics.particles_set")
                                            {
                                                let mut map = params.to_map();
                                                if kreate_ui::property_editor::property_editor_where(
                                                    ui,
                                                    &state.catalog,
                                                    info.id,
                                                    info.schema,
                                                    &mut map,
                                                    |n| {
                                                        n != "kind"
                                                            && ph::particle_param_applies(kind, n)
                                                    },
                                                ) {
                                                    let dragging =
                                                        ui.ctx().input(|i| i.pointer.any_down());
                                                    send_edit(
                                                        state,
                                                        "physics.particles_set",
                                                        map,
                                                        dragging,
                                                    );
                                                }
                                            }
                                            ui.separator();
                                        });
                                    }
                                    // Another system of the same kind.
                                    if ui.button(state.catalog.get("physics-add-another")).clicked()
                                    {
                                        let mut m = PropertyMap::new();
                                        m.set("kind", V::Enum(u32::from(sim == Sim::Hair)));
                                        state.queue.run("physics.particles_add", m);
                                    }
                                }
                            }
                            if sim.bakes() {
                                ui.separator();
                                let only = match sim {
                                    Sim::RigidBody => 1,
                                    Sim::Cloth | Sim::SoftBody => 2,
                                    Sim::Particles | Sim::Hair => 3,
                                    _ => 4,
                                };
                                bake_row(ui, state, doc, only, sim == Sim::RigidBody);
                            }
                        })
                    })
                    .inner;
                if closed {
                    set_sim(state, p, sim, false);
                }
            }
        }
    }
    ui.separator();
    // The scene's: gravity, how finely time is cut, the frames a bake covers.
    let title = state.catalog.get("physics-world");
    ui.push_id("physics.world_set", |ui| {
        kreate_ui::panel::section(ui, &title, false, |ui| {
            physics_fields(
                ui,
                state,
                registry,
                "physics.world_set",
                ph::world_params(&doc.physics).to_map(),
                |_| true,
            );
            // Every simulation at once.
            ui.separator();
            bake_row(ui, state, doc, 0, false);
        });
    });
}

/// The Render tab: the scene's render settings, and the buttons that use them.
fn draw_render_tab(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    l: &Labels,
) {
    // The scene's units, first: they are the scene's as much as how it
    // renders, and a design is set up in them before anything is drawn.
    ui.heading(&l.design_units);
    if let Some(info) = registry.get("scene.units_set") {
        let mut params =
            kreate_reflect::OpParams::to_map(&kreate_ops::ops::design::units_params(doc));
        if kreate_ui::property_editor(ui, &state.catalog, info.id, info.schema, &mut params) {
            send_edit(state, "scene.units_set", params, false);
        }
    }
    // The section view: a plane through the scene, its far side cut away.
    ui.separator();
    ui.strong(&l.design_section);
    if let Some(info) = registry.get("scene.section_set") {
        let mut params =
            kreate_reflect::OpParams::to_map(&kreate_ops::ops::design::section_params(doc));
        if kreate_ui::property_editor(ui, &state.catalog, info.id, info.schema, &mut params) {
            let dragging = ui.ctx().input(|i| i.pointer.any_down());
            send_edit(state, "scene.section_set", params, dragging);
        }
    }
    // The dimensions on the scene: what each reads, and a way to take it off.
    if !doc.design.measurements.is_empty() {
        ui.separator();
        ui.horizontal(|ui| {
            ui.strong(&l.design_dimensions);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button(&l.design_clear).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("all", kreate_reflect::PropertyValue::Bool(true));
                    state.queue.run("measure.remove", p);
                }
            });
        });
        for (i, m) in doc.design.measurements.iter().enumerate() {
            // i18n-exempt: a measured length or angle
            let value = match (m.length(), m.angle()) {
                (Some(d), _) => kreate_reflect::format_length(d),
                (_, Some(a)) => format!("{:.2}°", a.to_degrees()),
                _ => continue,
            };
            ui.horizontal(|ui| {
                // i18n-exempt: a number and a measured value
                ui.label(format!("{}.  {value}", i + 1));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // i18n-exempt: a close mark
                    if ui.small_button("x").clicked() {
                        let mut p = PropertyMap::new();
                        p.set("index", kreate_reflect::PropertyValue::Int(i as i64));
                        state.queue.run("measure.remove", p);
                    }
                });
            });
        }
    }
    ui.separator();
    ui.heading(&l.tab_render);
    ui.horizontal(|ui| {
        if ui.button(&l.render_image).clicked() {
            state.queue.dispatch("render.image", PropertyMap::new());
        }
        if ui.button(&l.render_animation).clicked() {
            state.queue.dispatch("render.animation", PropertyMap::new());
        }
    });
    ui.weak(&l.render_hint);
    ui.separator();
    let Some(info) = registry.get("render.set_settings") else {
        return;
    };
    let mut params = kreate_reflect::OpParams::to_map(&kreate_ops::ops::render::render_params(doc));
    if kreate_ui::property_editor(ui, &state.catalog, info.id, info.schema, &mut params) {
        let dragging = ui.ctx().input(|i| i.pointer.any_down());
        send_edit(state, "render.set_settings", params, dragging);
    }
}

/// The asset browser: the chosen source's assets of the chosen kind, as
/// pictures, with the search applied.
fn draw_assets(ui: &mut Ui, state: &mut ShellState) {
    let t = |key: &str| state.catalog.format(key, kreate_i18n::TextKind::Ui, &[]);
    let source = crate::assets::Source::ALL
        .get(state.asset_source)
        .copied()
        .unwrap_or(crate::assets::Source::Local);
    let text = kreate_ui::assets::AssetText {
        sources: vec![
            t("asset-source-local"),
            t("asset-source-polyhaven"),
            t("asset-source-ambientcg"),
            t("asset-source-objaverse"),
            t("asset-source-polypizza"),
            t("asset-source-generated"),
            t("asset-source-sketchfab"),
        ],
        kinds: vec![
            t("asset-kind-models"),
            t("asset-kind-materials"),
            t("asset-kind-hdris"),
        ],
        search: t("asset-search"),
        resolution: t("asset-resolution"),
        refresh: t("asset-refresh"),
        empty: t(
            if source == crate::assets::Source::Local && state.asset_folders.is_empty() {
                "asset-no-folders"
            } else {
                "asset-empty"
            },
        ),
        loading: t("asset-loading"),
        add_folder: t("asset-add-folder"),
        remove_folder: t("asset-remove-folder"),
        no_folders: t("asset-no-folders"),
        // Poly Haven asks a program that uses its API to say so.
        credit: match source {
            crate::assets::Source::PolyHaven => Some((
                t("asset-credit-polyhaven"),
                "https://polyhaven.com".to_string(),
            )),
            crate::assets::Source::AmbientCg => Some((
                t("asset-credit-ambientcg"),
                "https://ambientcg.com".to_string(),
            )),
            crate::assets::Source::PolyPizza => Some((
                t("asset-credit-polypizza"),
                "https://poly.pizza".to_string(),
            )),
            crate::assets::Source::Objaverse => Some((
                t("asset-credit-objaverse"),
                "https://huggingface.co/datasets/allenai/objaverse".to_string(),
            )),
            crate::assets::Source::Sketchfab => Some((
                t("asset-credit-sketchfab"),
                "https://sketchfab.com".to_string(),
            )),
            crate::assets::Source::Generated => Some((
                t("asset-credit-generated"),
                "https://huggingface.co/microsoft/TRELLIS-image-large".to_string(),
            )),
            crate::assets::Source::Local => None,
        },
        add: t("asset-add"),
        link: t("asset-link"),
        link_hint: t("asset-link-hint"),
        animated_only: t("asset-animated-only"),
        search_hint: match source {
            crate::assets::Source::Objaverse => t("asset-search-hint-objaverse"),
            crate::assets::Source::Sketchfab => match crate::sketchfab::signed_in_as() {
                Some(who) => state.catalog.format(
                    "asset-search-hint-sketchfab",
                    kreate_i18n::TextKind::Ui,
                    &[("name", who.as_str())],
                ),
                None => t("asset-search-hint-sketchfab-out"),
            },
            _ => t("asset-search-hint"),
        },
        key_prompt: t(if source == crate::assets::Source::Sketchfab {
            "asset-sketchfab-prompt"
        } else {
            "asset-key-prompt"
        }),
        key_where: if source == crate::assets::Source::Sketchfab {
            (
                t("asset-sketchfab-where"),
                "https://sketchfab.com/settings/password".to_string(),
            )
        } else {
            (
                t("asset-key-where"),
                "https://poly.pizza/settings/api".to_string(),
            )
        },
        key_hint: t(if source == crate::assets::Source::Sketchfab {
            "asset-sketchfab-hint"
        } else {
            "asset-key-hint"
        }),
        key_save: t("asset-key-save"),
        key_change: t("asset-key-change"),
        gen_hint: t("asset-gen-hint"),
        gen_text: t("asset-gen-text"),
        gen_image: t("asset-gen-image"),
        gen_note: t("asset-gen-note"),
        gen_token: t("asset-gen-token"),
        gen_token_prompt: t("asset-gen-token-prompt"),
        gen_token_where: (
            t("asset-gen-token-where"),
            "https://huggingface.co/settings/tokens".to_string(),
        ),
        gen_token_saved: t("asset-gen-token-saved"),
        gen_token_forget: t("asset-gen-token-forget"),
        add_hint: t(match state.asset_kind {
            1 => "asset-add-hint-material",
            2 => "asset-add-hint-hdri",
            _ => "asset-add-hint-model",
        }),
    };
    // Pictures that arrived since the last frame become textures, once.
    for (id, (w, h, pixels)) in std::mem::take(&mut state.asset_incoming) {
        let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &pixels);
        let handle =
            ui.ctx()
                .load_texture(format!("asset-{id}"), image, egui::TextureOptions::LINEAR);
        state.asset_textures.insert(id, handle);
    }
    let tiles: Vec<kreate_ui::assets::AssetTile> = state
        .asset_entries
        .iter()
        .enumerate()
        // A source that searches itself has already: what it sent is the
        // answer, whether or not each word is in a name.
        .filter(|(_, e)| source.searched() || e.matches(&state.asset_search))
        .map(|(index, e)| kreate_ui::assets::AssetTile {
            index,
            name: e.name.clone(),
            detail: match (&e.path, &e.licence) {
                (Some(p), _) => p.display().to_string(),
                (None, Some(licence)) => format!("{} · {licence}", e.authors),
                (None, None) => e.authors.clone(),
            },
            badges: {
                let mut badges = Vec::new();
                if e.animated {
                    badges.push(t("asset-badge-animated"));
                }
                if e.credit().is_some() {
                    // i18n-exempt: a licence's own name
                    badges.push("CC-BY".to_string());
                }
                badges
            },
            texture: state.asset_textures.get(&e.id).map(|h| {
                let [w, h2] = h.size();
                (h.id(), [w as f32, h2 as f32])
            }),
        })
        .collect();
    let folders: Vec<String> = state
        .asset_folders
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    let view = kreate_ui::assets::AssetView {
        source: state.asset_source,
        kind: state.asset_kind,
        resolutions: source
            .sized()
            .then_some((&crate::assets::RESOLUTIONS[..], state.asset_resolution)),
        folders: (source == crate::assets::Source::Local).then_some(&folders[..]),
        tiles: &tiles,
        selected: state
            .asset_selected
            .as_ref()
            .and_then(|id| state.asset_entries.iter().position(|e| &e.id == id)),
        linkable: state
            .asset_selected
            .as_ref()
            .and_then(|id| state.asset_entries.iter().find(|e| &e.id == id))
            .and_then(|e| e.path.as_ref())
            .is_some_and(|p| {
                p.extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("kreate"))
            }),
        busy: state.asset_busy.clone(),
        error: state
            .asset_error
            .clone()
            .filter(|e| e != crate::assets::KEY_MISSING && e != crate::sketchfab::TOKEN_MISSING),
        searched: source.searched(),
        animated_only: source.searched().then_some(state.asset_query.animated),
        key_needed: matches!(
            source,
            crate::assets::Source::PolyPizza | crate::assets::Source::Sketchfab
        ) && (!state.asset_key_held
            || matches!(
                state.asset_error.as_deref(),
                Some(crate::assets::KEY_REFUSED | crate::sketchfab::TOKEN_REFUSED)
            )),
        key_held: state.asset_key_held,
        generator: (source == crate::assets::Source::Generated).then(|| {
            kreate_ui::assets::GeneratorView {
                busy: state.asset_generating,
                token_held: crate::generate::token().is_some(),
            }
        }),
    };
    let mut search = std::mem::take(&mut state.asset_search);
    let seen = kreate_ui::assets::asset_browser(ui, &text, &view, &mut search, &mut state.queue);
    state.asset_search = search;
    for index in seen {
        if let Some(entry) = state.asset_entries.get(index) {
            if !state.asset_asked.contains(&entry.id) && !state.asset_wanted.contains(&entry.id) {
                state.asset_wanted.push(entry.id.clone());
            }
        }
    }
    if state.asset_entries.is_empty() && state.asset_busy.is_none() && state.asset_error.is_none() {
        state.asset_list_wanted = true;
    }
}

/// The Claude panel: the conversation, and a box to say the next thing.
fn draw_chat(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    let text = kreate_ui::chat::ChatText {
        attach: l.chat_attach.clone(),
        attach_hint: l.chat_attach_hint.clone(),
        remove: l.chat_remove.clone(),
        placeholder: l.chat_placeholder.clone(),
        empty: l.chat_empty.clone(),
        send: l.chat_send.clone(),
        stop: l.chat_stop.clone(),
        new_chat: l.chat_new.clone(),
        history: l.chat_history.clone(),
        export: l.chat_export.clone(),
        working: l.chat_working.clone(),
        checking: l.chat_checking.clone(),
        signed_out: l.chat_signed_out.clone(),
        sign_in: l.chat_sign_in.clone(),
        sign_out: l.chat_sign_out.clone(),
        signing_in: l.chat_signing_in.clone(),
        open_sign_in: l.chat_open_sign_in.clone(),
        code_hint: l.chat_code_hint.clone(),
        code_submit: l.chat_code_submit.clone(),
        cancel: l.chat_cancel.clone(),
        revert: state
            .catalog
            .format("chat-revert", kreate_i18n::TextKind::Ui, &[]),
        revert_hint: state
            .catalog
            .format("chat-revert-hint", kreate_i18n::TextKind::Tooltip, &[]),
    };
    let mut input = std::mem::take(&mut state.chat_input);
    let mut code = std::mem::take(&mut state.chat_code);
    let names: Vec<&str> = crate::claude::Provider::ALL
        .iter()
        .map(|p| p.name())
        .collect();
    let providers = kreate_ui::chat::ChatProviders {
        names: &names,
        current: state.chat_provider,
        models: &state.chat_models,
        model: state.chat_model,
    };
    state.chat_focused = kreate_ui::chat::chat(
        ui,
        &text,
        &providers,
        &state.chat_account,
        &state.chat_log,
        &state.chat_attachments,
        &mut input,
        &mut code,
        state.chat_busy,
        &mut state.queue,
    );
    state.chat_input = input;
    state.chat_code = code;
}

/// The UV editor, with the view state that belongs to this area.
fn draw_uv_editor(ui: &mut Ui, state: &mut ShellState, area: kreate_ui::AreaId, l: &Labels) {
    let text = kreate_ui::UvText {
        no_uvs: l.uv_none.clone(),
        not_editing: l.uv_not_editing.clone(),
    };
    let data = state.uv.clone();
    // Taken out and put back, because the editor writes to it while `state` is
    // also lending its command queue.
    let mut view = state.uv_views.remove(&area).unwrap_or_default();
    let commands = kreate_ui::uv_editor(
        ui,
        data.as_ref().map(|d| d.view()).as_ref(),
        &mut view,
        &text,
    );
    state.uv_views.insert(area, view);
    for command in commands {
        state.queue.push(command);
    }
}

/// The timeline, turning what the pointer asks for into the application's
/// clock actions — the same ones the keys and the menus reach.
fn draw_timeline(ui: &mut Ui, state: &mut ShellState, area: kreate_ui::AreaId, l: &Labels) {
    use kreate_reflect::PropertyValue as V;
    use kreate_ui::timeline::TimelineEdit as E;
    let text = kreate_ui::timeline::TimelineText {
        play: l.time_play.clone(),
        pause: l.time_pause.clone(),
        jump_start: l.time_jump_start.clone(),
        jump_end: l.time_jump_end.clone(),
        start: l.time_start.clone(),
        end: l.time_end.clone(),
        autokey: l.time_autokey.clone(),
    };
    let view = state.timeline.clone();
    let mut panel = state.timeline_views.remove(&area).unwrap_or_default();
    let edits = kreate_ui::timeline::timeline(ui, &view, &mut panel, &text);
    state.timeline_views.insert(area, panel);
    for edit in edits {
        let mut p = PropertyMap::new();
        let id = match edit {
            E::SetFrame(frame) => {
                p.set("frame", V::Float(frame));
                "time.set_frame"
            }
            E::TogglePlay => "time.play",
            E::JumpStart => "time.jump",
            E::JumpEnd => {
                p.set("to_end", V::Bool(true));
                "time.jump"
            }
            E::SetRange { start, end } => {
                p.set("start", V::Int(start as i64));
                p.set("end", V::Int(end as i64));
                "time.set_range"
            }
            E::ToggleAutokey => "time.toggle_autokey",
        };
        state.queue.dispatch(id, p);
    }
}

/// Why a driver cannot be evaluated, in the reader's language.
fn driver_error_text(catalog: &Catalog, error: &kreate_scene::driver::DriverError) -> String {
    use kreate_scene::driver::DriverError as D;
    use kreate_scene::expression::ExprError as X;
    let t = |key: &str, args: &[(&str, &str)]| catalog.format(key, TextKind::Ui, args);
    match error {
        D::Expression(X::Empty) => t("driver-error-empty", &[]),
        D::Expression(X::Syntax { at }) => {
            t("driver-error-syntax", &[("at", &(at + 1).to_string())])
        }
        D::Expression(X::UnknownName(n)) => t("driver-error-name", &[("name", n)]),
        D::Expression(X::UnknownFunction(n)) => t("driver-error-function", &[("name", n)]),
        D::Expression(X::Arguments { function, expected }) => t(
            "driver-error-arguments",
            &[("name", function), ("count", &expected.to_string())],
        ),
        D::Expression(X::NotFinite) => t("driver-error-infinite", &[]),
        D::NoTarget { variable } => t("driver-error-target", &[("name", variable)]),
        D::BadProperty { variable } => t("driver-error-property", &[("name", variable)]),
        D::BadName { variable } => t("driver-error-bad-name", &[("name", variable)]),
    }
}

/// The Drivers editor, over the selected objects' drivers.
fn draw_drivers(ui: &mut Ui, state: &mut ShellState, doc: &Document, area: kreate_ui::AreaId) {
    use kreate_ops::ops::driver::{driver_target, DRIVABLE};
    use kreate_reflect::PropertyValue as V;
    use kreate_ui::drivers::{DriverEdit as E, DriverView, VariableView};
    const AXES: [&str; 3] = ["X", "Y", "Z"];
    let channel = |catalog: &Catalog, property: &str| {
        catalog.format(
            &format!("channel-{}", property.replace('_', "-")),
            TextKind::Ui,
            &[],
        )
    };
    let cycles = doc.driver_cycles();
    let mut drivers = Vec::new();
    for id in &doc.selection.objects {
        let (Some(object), Some(uid)) = (doc.object(*id), doc.object_uid(*id)) else {
            continue;
        };
        for (i, d) in object.drivers.iter().enumerate() {
            let axis = AXES.get(d.index as usize).copied().unwrap_or("");
            let result = if cycles.contains(id) {
                Err(state
                    .catalog
                    .format("driver-error-cycle", TextKind::Ui, &[]))
            } else {
                doc.evaluate_driver(*id, i)
                    .map_err(|e| driver_error_text(&state.catalog, &e))
            };
            drivers.push(DriverView {
                // i18n-exempt: the object's name is the user's own text
                label: format!(
                    "{}: {} {axis}",
                    object.name,
                    channel(&state.catalog, &d.property)
                ),
                target: driver_target(uid.raw(), &d.property, d.index),
                expression: d.expression.clone(),
                result,
                variables: d
                    .variables
                    .iter()
                    .map(|v| VariableView {
                        name: v.name.clone(),
                        object: (v.target != kreate_scene::Uid::NONE).then(|| v.target.raw()),
                        property: DRIVABLE
                            .iter()
                            .position(|p| *p == v.property.as_ref())
                            .unwrap_or(0) as u32,
                        index: v.index as u32,
                    })
                    .collect(),
            });
        }
    }
    let mut addable = Vec::new();
    if let Some((object, uid)) = doc
        .selection
        .active
        .and_then(|id| Some((doc.object(id)?, doc.object_uid(id)?)))
    {
        for property in DRIVABLE {
            for index in 0..3u8 {
                if !object
                    .drivers
                    .iter()
                    .any(|d| d.property.as_ref() == property && d.index == index)
                {
                    addable.push((
                        // i18n-exempt: a translated channel and an axis letter
                        format!(
                            "{} {}",
                            channel(&state.catalog, property),
                            AXES[index as usize]
                        ),
                        driver_target(uid.raw(), property, index),
                    ));
                }
            }
        }
    }
    let view = kreate_ui::drivers::DriversView {
        drivers,
        objects: doc
            .objects()
            .filter_map(|(id, o)| Some((doc.object_uid(id)?.raw(), o.name.clone())))
            .collect(),
        addable,
    };
    let t = |key: &str| state.catalog.format(key, TextKind::Ui, &[]);
    let text = kreate_ui::drivers::DriversText {
        empty: t("drivers-empty"),
        add: t("drivers-add"),
        remove: t("drivers-remove"),
        expression: t("drivers-expression"),
        add_variable: t("drivers-add-variable"),
        remove_variable: t("drivers-remove-variable"),
        none: t("drivers-none"),
        properties: [
            t("channel-location"),
            t("channel-rotation-euler"),
            t("channel-scale"),
        ],
        axes: AXES.map(String::from),
    };
    let mut panel = state.driver_views.remove(&area).unwrap_or_default();
    let edits = kreate_ui::drivers::drivers_editor(ui, &view, &mut panel, &text);
    state.driver_views.insert(area, panel);
    for edit in edits {
        let mut p = PropertyMap::new();
        match edit {
            E::Add(target) => {
                p.set("target", V::Text(target));
                state.queue.run("anim.driver_add", p);
            }
            E::Remove(target) => {
                p.set("target", V::Text(target));
                state.queue.run("anim.driver_remove", p);
            }
            E::Expression { target, expression } => {
                p.set("target", V::Text(target));
                p.set("expression", V::Text(expression));
                state.queue.run("anim.driver_expression", p);
            }
            E::AddVariable(target) => {
                p.set("target", V::Text(target));
                state.queue.run("anim.driver_variable_add", p);
            }
            E::SetVariable {
                target,
                variable,
                value,
            } => {
                p.set("target", V::Text(target));
                p.set("variable", V::Int(variable as i64));
                p.set("name", V::Text(value.name));
                p.set(
                    "object",
                    V::Text(value.object.map(|u| u.to_string()).unwrap_or_default()),
                );
                p.set("property", V::Enum(value.property));
                p.set("index", V::Int(value.index as i64));
                state.queue.run("anim.driver_variable_set", p);
            }
            E::RemoveVariable { target, variable } => {
                p.set("target", V::Text(target));
                p.set("variable", V::Int(variable as i64));
                state.queue.run("anim.driver_variable_remove", p);
            }
        }
    }
}

/// The graph editor, over the curves of every action the dope sheet shows.
fn draw_graph_editor(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    area: kreate_ui::AreaId,
    l: &Labels,
) {
    use kreate_reflect::PropertyValue as V;
    use kreate_ui::graph_editor::{GraphCurve, GraphEdit as E, GraphKey};
    let p = doc.playback;
    let mut curves = Vec::new();
    for action_id in kreate_ops::ops::anim::shown_actions(doc) {
        let (Some(action), Some(uid)) = (doc.action(action_id), doc.action_uid(action_id)) else {
            continue;
        };
        let owner = owner_name(doc, action_id).unwrap_or_else(|| action.name.clone());
        for curve in &action.curves {
            let (Some(index), Some(first), Some(last)) =
                (curve.path.index, curve.keys.first(), curve.keys.last())
            else {
                continue;
            };
            let property = curve.path.property.as_ref();
            let (components, is_color) =
                kreate_scene::animation::describe(action.animates, property)
                    .map(|d| (d.components, d.kind == kreate_reflect::PropertyKind::Color))
                    .unwrap_or((1, false));
            // A quaternion's W first, then its X, Y, Z in the axes' colours.
            let (components, index) = if components == 4 {
                if index == 0 {
                    (1, 0)
                } else {
                    (3, index - 1)
                }
            } else {
                (components, index)
            };
            let quaternion = kreate_scene::animation::describe(action.animates, property)
                .is_some_and(|d| d.components == 4);
            // X red, Y green, Z blue, as the axes everywhere else are; a
            // colour's R, G and B the same, and a single number white.
            let color = match (components, index) {
                (1, _) => egui::Color32::from_gray(210),
                (_, 0) => egui::Color32::from_rgb(0xe0, 0x5a, 0x5a),
                (_, 1) => egui::Color32::from_rgb(0x7c, 0xc4, 0x4c),
                _ => egui::Color32::from_rgb(0x5a, 0x8c, 0xe0),
            };
            let axis = if quaternion && components == 1 {
                "W".to_string()
            } else if components == 1 {
                String::new()
            } else if is_color {
                ["R", "G", "B"]
                    .get(index as usize)
                    .copied()
                    .unwrap_or("")
                    .to_string()
            } else {
                ["X", "Y", "Z"]
                    .get(index as usize)
                    .copied()
                    .unwrap_or("")
                    .to_string()
            };
            let channel = channel_label(&state.catalog, property);
            // Sampled over the range and the keys, with room either side, at
            // most a thousand points: enough to draw, not a thousand per frame
            // of a long scene.
            let from = first.frame.min(p.start as f64) - 20.0;
            let to = last.frame.max(p.end as f64) + 20.0;
            let step = ((to - from) / 1000.0).max(0.25);
            let mut samples = Vec::new();
            let mut f = from;
            while f <= to {
                if let Some(v) = curve.evaluate(f) {
                    samples.push([f, v]);
                }
                f += step;
            }
            let keys = curve
                .keys
                .iter()
                .enumerate()
                .map(|(i, k)| GraphKey {
                    frame: k.frame,
                    value: k.value,
                    selected: k.selected,
                    slope: curve.slope(i),
                    before: i
                        .checked_sub(1)
                        .map(|j| k.frame - curve.keys[j].frame)
                        .unwrap_or(10.0),
                    after: curve
                        .keys
                        .get(i + 1)
                        .map(|n| n.frame - k.frame)
                        .unwrap_or(10.0),
                })
                .collect();
            curves.push(GraphCurve {
                // i18n-exempt: the owner's name is the user's own text
                label: format!("{owner}: {channel} {axis}"),
                target: kreate_ops::ops::anim::curve_target(uid.raw(), property, index),
                color,
                keys,
                samples,
            });
        }
    }
    let view = kreate_ui::graph_editor::GraphView {
        curves,
        frame: p.frame,
        start: p.start,
        end: p.end,
    };
    let text = kreate_ui::graph_editor::GraphText {
        empty: l.sheet_empty.clone(),
        interpolation: l.sheet_interpolation.clone(),
        constant: l.interp_constant.clone(),
        linear: l.interp_linear.clone(),
        bezier: l.interp_bezier.clone(),
        handles: l.graph_handles.clone(),
        auto: l.handle_auto.clone(),
        flat: l.handle_flat.clone(),
        extrapolation: state.catalog.get("graph-extrapolation"),
        cyclic: state.catalog.get("extrapolation-cyclic"),
    };
    let mut panel = state.graph_views.remove(&area).unwrap_or_default();
    let edits = kreate_ui::graph_editor::graph_editor(ui, &view, &mut panel, &text);
    state.graph_views.insert(area, panel);
    for edit in edits {
        let mut p = PropertyMap::new();
        match edit {
            E::Select {
                targets,
                frames,
                values,
                extend,
            } => {
                p.set("targets", V::Text(targets.join(" ")));
                p.set("frame_min", V::Float(frames.0));
                p.set("frame_max", V::Float(frames.1));
                p.set("value_min", V::Float(values.0));
                p.set("value_max", V::Float(values.1));
                p.set("extend", V::Bool(extend));
                state.queue.run("anim.keys_select_region", p);
            }
            E::SelectAll(action) => {
                p.set("action", V::Enum(action));
                state.queue.run("anim.keys_select_all", p);
            }
            E::Move { frames, value } => {
                p.set("offset", V::Float(frames));
                p.set("value", V::Float(value));
                state.queue.run("anim.keys_move", p);
            }
            E::Delete => state.queue.run("anim.keys_delete", p),
            E::Interpolation(kind) => {
                p.set("interpolation", V::Enum(kind));
                state.queue.run("anim.keys_interpolation", p);
            }
            E::HandleType(kind) => {
                p.set("kind", V::Enum(kind));
                state.queue.run("anim.keys_handle_type", p);
            }
            E::Extrapolation(mode) => {
                p.set("extrapolation", V::Enum(mode));
                state.queue.run("anim.keys_extrapolation", p);
            }
            E::Smooth => state.queue.run("anim.keys_smooth", p),
            E::Bake => state.queue.run("anim.keys_bake", p),
            E::SetSlope {
                target,
                frame,
                slope,
            } => {
                p.set("target", V::Text(target));
                p.set("frame", V::Float(frame));
                p.set("slope", V::Float(slope));
                state.queue.run("anim.key_set_slope", p);
            }
            E::SetFrame(frame) => {
                p.set("frame", V::Float(frame));
                state.queue.dispatch("time.set_frame", p);
            }
        }
    }
}

/// A curve's property as a person reads it: the channel's name, and for a
/// bone's the bone's name before it — "Upper: Rotation".
fn channel_label(catalog: &Catalog, property: &str) -> String {
    let channel = |p: &str| {
        catalog.format(
            &format!("channel-{}", p.replace('_', "-")),
            TextKind::Ui,
            &[],
        )
    };
    match kreate_scene::animation::parse_bone_path(property) {
        // i18n-exempt: a bone's name is the user's own text
        Some((bone, p)) => format!("{bone}: {}", channel(p)),
        None => channel(property),
    }
}

/// What plays an action: the object, material or lamp it belongs to, by name.
fn owner_name(doc: &Document, action: kreate_scene::ActionId) -> Option<String> {
    doc.objects()
        .find(|(_, o)| o.action == Some(action))
        .map(|(_, o)| o.name.clone())
        .or_else(|| {
            doc.materials()
                .find(|(_, m)| m.action == Some(action))
                .map(|(_, m)| m.name.clone())
        })
        .or_else(|| {
            doc.lights()
                .find(|(_, l)| l.action == Some(action))
                .map(|(_, l)| l.name.clone())
        })
}

/// The dope sheet, over the selected objects' actions — what Blender's shows
/// with Only Show Selected, and what the `anim.keys_*` operators act on.
fn draw_dope_sheet(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    area: kreate_ui::AreaId,
    l: &Labels,
) {
    use kreate_reflect::PropertyValue as V;
    use kreate_ui::dope_sheet::{SheetEdit as E, SheetKey, SheetRow};
    let mut rows = Vec::new();
    // Every action on show — the objects', then their materials' and lamps' —
    // in the order the operators act on them, each with its owner's name and
    // a row per property its kind animates.
    for action_id in kreate_ops::ops::anim::shown_actions(doc) {
        let (Some(action), Some(uid)) = (doc.action(action_id), doc.action_uid(action_id)) else {
            continue;
        };
        let owner = owner_name(doc, action_id).unwrap_or_else(|| action.name.clone());
        // A key per frame on a row, selected if any key there is — what the
        // summary row and a property's row both mean by one diamond.
        let keys_of = |property: Option<&str>| -> Vec<SheetKey> {
            let mut keys: Vec<SheetKey> = Vec::new();
            for curve in action
                .curves
                .iter()
                .filter(|c| property.is_none_or(|p| c.path.property.as_ref() == p))
            {
                for k in &curve.keys {
                    match keys
                        .iter_mut()
                        .find(|s| (s.frame - k.frame).abs() < kreate_scene::animation::SAME_FRAME)
                    {
                        Some(s) => s.selected |= k.selected,
                        None => keys.push(SheetKey {
                            frame: k.frame,
                            selected: k.selected,
                        }),
                    }
                }
            }
            keys.sort_by(|a, b| a.frame.total_cmp(&b.frame));
            keys
        };
        rows.push(SheetRow {
            // i18n-exempt: a datablock's name is the user's own text
            label: owner,
            depth: 0,
            target: kreate_ops::ops::anim::row_target(uid.raw(), None),
            keys: keys_of(None),
        });
        for desc in kreate_scene::animation::animatable(action.animates) {
            if action
                .curves
                .iter()
                .any(|c| c.path.property.as_ref() == desc.name)
            {
                rows.push(SheetRow {
                    label: state.catalog.format(
                        &format!("channel-{}", desc.name.replace('_', "-")),
                        TextKind::Ui,
                        &[],
                    ),
                    depth: 1,
                    target: kreate_ops::ops::anim::row_target(uid.raw(), Some(desc.name)),
                    keys: keys_of(Some(desc.name)),
                });
            }
        }
        // Then each bone's channels, in the order they were first keyed.
        let mut bone_paths: Vec<&str> = Vec::new();
        for c in &action.curves {
            let path = c.path.property.as_ref();
            if kreate_scene::animation::parse_bone_path(path).is_some()
                && !bone_paths.contains(&path)
            {
                bone_paths.push(path);
            }
        }
        for path in bone_paths {
            rows.push(SheetRow {
                label: channel_label(&state.catalog, path),
                depth: 1,
                target: kreate_ops::ops::anim::row_target(uid.raw(), Some(path)),
                keys: keys_of(Some(path)),
            });
        }
    }
    let view = kreate_ui::dope_sheet::SheetView {
        rows,
        frame: doc.playback.frame,
        start: doc.playback.start,
        end: doc.playback.end,
    };
    let text = kreate_ui::dope_sheet::SheetText {
        empty: l.sheet_empty.clone(),
        interpolation: l.sheet_interpolation.clone(),
        constant: l.interp_constant.clone(),
        linear: l.interp_linear.clone(),
        bezier: l.interp_bezier.clone(),
    };
    let mut panel = state.sheet_views.remove(&area).unwrap_or_default();
    let edits = kreate_ui::dope_sheet::dope_sheet(ui, &view, &mut panel, &text);
    state.sheet_views.insert(area, panel);
    for edit in edits {
        let mut p = PropertyMap::new();
        match edit {
            E::Select {
                targets,
                frame_min,
                frame_max,
                extend,
            } => {
                p.set("targets", V::Text(targets.join(" ")));
                p.set("frame_min", V::Float(frame_min));
                p.set("frame_max", V::Float(frame_max));
                p.set("extend", V::Bool(extend));
                state.queue.run("anim.keys_select_box", p);
            }
            E::SelectAll(action) => {
                p.set("action", V::Enum(action));
                state.queue.run("anim.keys_select_all", p);
            }
            E::Move { offset } => {
                p.set("offset", V::Float(offset));
                state.queue.run("anim.keys_move", p);
            }
            E::Delete => state.queue.run("anim.keys_delete", p),
            E::Interpolation(kind) => {
                p.set("interpolation", V::Enum(kind));
                state.queue.run("anim.keys_interpolation", p);
            }
            E::SetFrame(frame) => {
                p.set("frame", V::Float(frame));
                state.queue.dispatch("time.set_frame", p);
            }
        }
    }
}

/// The NLA editor, for the selected objects' strips.
fn draw_nla_editor(ui: &mut Ui, state: &mut ShellState, doc: &Document, area: kreate_ui::AreaId) {
    use kreate_reflect::PropertyValue as V;
    use kreate_ui::nla_editor::{NlaEdit as E, NlaStripView, NlaTrack};
    // The frames an action's keys span.
    let span = |a| -> Option<(f64, f64)> {
        let action = doc.action(a)?;
        let frames = action
            .curves
            .iter()
            .flat_map(|c| c.keys.iter().map(|k| k.frame));
        let (lo, hi) = frames.fold((f64::MAX, f64::MIN), |(lo, hi), f| (lo.min(f), hi.max(f)));
        (lo <= hi).then_some((lo, hi.max(lo + 1.0)))
    };
    let tracks: Vec<NlaTrack> = doc
        .selection
        .objects
        .iter()
        .filter_map(|id| doc.object(*id))
        .filter(|o| o.action.is_some() || !o.nla.is_empty())
        .map(|o| NlaTrack {
            // i18n-exempt: an object's name is the user's own text
            object: o.name.clone(),
            action: o.action.and_then(|a| {
                let (lo, hi) = span(a)?;
                Some((doc.action(a)?.name.clone(), lo, hi))
            }),
            strips: o
                .nla
                .iter()
                .enumerate()
                .map(|(index, s)| NlaStripView {
                    index,
                    // i18n-exempt: a strip's name is its action's, the user's own text
                    label: s.name.clone(),
                    start: s.start,
                    end: s.end(),
                    mute: s.mute,
                })
                .collect(),
        })
        .collect();
    let view = kreate_ui::nla_editor::NlaView {
        tracks,
        frame: doc.playback.frame,
        start: doc.playback.start,
        end: doc.playback.end,
    };
    let text = kreate_ui::nla_editor::NlaText {
        empty: state.catalog.get("nla-empty"),
        push_down: state.catalog.get("op-anim-nla-push-down"),
    };
    let mut panel = state.nla_views.remove(&area).unwrap_or_default();
    let edits = kreate_ui::nla_editor::nla_editor(ui, &view, &mut panel, &text);
    state.nla_views.insert(area, panel);
    // The operators act on the active object: the strip's object is made it
    // first when it is not.
    let mut active = doc
        .selection
        .active
        .and_then(|id| doc.object(id))
        .map(|o| o.name.clone());
    let mut activate = |state: &mut ShellState, object: &str| {
        if active.as_deref() != Some(object) {
            let mut p = PropertyMap::new();
            p.set("name", V::Text(object.to_string()));
            p.set("extend", V::Bool(true));
            state.queue.run("object.select", p);
            active = Some(object.to_string());
        }
    };
    let strip_of = |object: &str, index: usize| {
        doc.objects()
            .find(|(_, o)| o.name == object)
            .and_then(|(_, o)| o.nla.get(index))
    };
    for edit in edits {
        match edit {
            E::Select { object, .. } => activate(state, &object),
            E::Move {
                object,
                index,
                start,
            } => {
                if let Some(s) = strip_of(&object, index) {
                    activate(state, &object);
                    let mut params = kreate_ops::ops::anim::nla_strip_params(index, s);
                    params.start = start;
                    state.queue.run(
                        "anim.nla_strip_set",
                        kreate_reflect::OpParams::to_map(&params),
                    );
                }
            }
            E::Mute {
                object,
                index,
                mute,
            } => {
                if let Some(s) = strip_of(&object, index) {
                    activate(state, &object);
                    let mut params = kreate_ops::ops::anim::nla_strip_params(index, s);
                    params.mute = mute;
                    state.queue.run(
                        "anim.nla_strip_set",
                        kreate_reflect::OpParams::to_map(&params),
                    );
                }
            }
            E::Remove { object, index } => {
                activate(state, &object);
                let mut p = PropertyMap::new();
                p.set("index", V::Int(index as i64));
                state.queue.run("anim.nla_strip_remove", p);
            }
            E::PushDown { object } => {
                activate(state, &object);
                state.queue.run("anim.nla_push_down", PropertyMap::new());
            }
            E::SetFrame(frame) => {
                let mut p = PropertyMap::new();
                p.set("frame", V::Float(frame));
                state.queue.dispatch("time.set_frame", p);
            }
        }
    }
}

/// The geometry node editor, for the active object's graph.
fn draw_geometry_editor(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    area: kreate_ui::AreaId,
    l: &Labels,
) {
    use kreate_reflect::PropertyValue as V;
    use kreate_scene::geonodes::GeoNodeKind;
    use kreate_ui::node_editor::{Family, FieldView, LinkView, NodeView, SocketType, SocketView};
    let object = doc
        .selection
        .active
        .and_then(|id| doc.object(id))
        .filter(|o| o.mesh().is_some());
    let t = |key: &str| state.catalog.get(key);
    let view = object.and_then(|o| o.geometry_nodes.as_ref()).map(|graph| {
        let nodes = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(n, node)| {
                use kreate_scene::geonodes::GeoSocket;
                let socket = |ty: GeoSocket, label: &str, connected: bool| SocketView {
                    label: t(&format!("geo-socket-{label}")),
                    ty: match ty {
                        GeoSocket::Geometry => SocketType::Geometry,
                        GeoSocket::Float => SocketType::Scalar,
                        GeoSocket::Vector => SocketType::Vector,
                    },
                    connected,
                    default: None,
                };
                let labels = node.kind.input_labels();
                NodeView {
                    // i18n-exempt: a node's name is the document's own text
                    title: node.name.clone(),
                    position: node.position,
                    outputs: node
                        .kind
                        .output_type()
                        .map(|ty| vec![socket(ty, node.kind.output_label(), false)])
                        .unwrap_or_default(),
                    inputs: node
                        .kind
                        .input_types()
                        .iter()
                        .enumerate()
                        .map(|(i, ty)| {
                            socket(
                                *ty,
                                labels.get(i).copied().unwrap_or("geometry"),
                                graph.incoming(n, i).is_some(),
                            )
                        })
                        .collect(),
                    fields: node
                        .kind
                        .fields()
                        .into_iter()
                        .map(|f| match f.choice {
                            Some(options) => FieldView::Choice {
                                field: f.index,
                                value: f.value.round().max(0.0) as u32,
                                options: options
                                    .iter()
                                    .map(|o| t(&format!("geo-option-{}", o.replace('_', "-"))))
                                    .collect(),
                            },
                            None => FieldView::Value {
                                field: f.index,
                                value: f.value,
                                min: f.min,
                                max: f.max,
                                label: t(&format!("geo-field-{}", f.name)),
                            },
                        })
                        .collect(),
                    family: match node.kind {
                        GeoNodeKind::Input | GeoNodeKind::Output => Family::Input,
                        GeoNodeKind::Primitive { .. }
                        | GeoNodeKind::CurveLine { .. }
                        | GeoNodeKind::CurveCircle { .. }
                        | GeoNodeKind::CurveSpiral { .. } => Family::Texture,
                        _ if node.kind.output_type() != Some(GeoSocket::Geometry)
                            && node.kind.has_output() =>
                        {
                            Family::Converter
                        }
                        _ => Family::Geometry,
                    },
                }
            })
            .collect();
        kreate_ui::GraphView {
            nodes,
            links: graph
                .links
                .iter()
                .map(|l| LinkView {
                    from: (l.from_node, 0),
                    to: (l.to_node, l.to_socket),
                })
                .collect(),
            images: Vec::new(),
            addable: GeoNodeKind::ADDABLE
                .iter()
                .map(|tag| t(&format!("geo-kind-{}", tag.replace('_', "-"))))
                .collect(),
        }
    });
    let text = kreate_ui::NodeText {
        no_graph: if object.is_some() {
            t("geo-no-graph")
        } else {
            t("geo-no-object")
        },
        use_nodes: t("geo-enable"),
        add: l.node_add.clone(),
        no_image: String::new(),
        open_image: String::new(),
    };
    let mut editor = state.node_views.remove(&area).unwrap_or_default();
    let mut asked = false;
    let edits = kreate_ui::node_editor(ui, view.as_ref(), &mut editor, &text, &mut asked);
    state.node_views.insert(area, editor);
    if asked && object.is_some() {
        state.queue.run("geo.enable", PropertyMap::new());
    }
    use kreate_ui::NodeEdit as E;
    for edit in edits {
        let mut p = PropertyMap::new();
        match edit {
            E::Translate { nodes, x, y } => {
                p.set("nodes", V::Ints(nodes));
                p.set("x", V::Float(x as f64));
                p.set("y", V::Float(y as f64));
                state.queue.run("geo.translate", p);
            }
            E::Link {
                from_node,
                to_node,
                to_socket,
                ..
            } => {
                p.set("from_node", V::Int(from_node as i64));
                p.set("to_node", V::Int(to_node as i64));
                p.set("to_socket", V::Int(to_socket as i64));
                state.queue.run("geo.link", p);
            }
            E::Unlink { to_node, to_socket } => {
                p.set("to_node", V::Int(to_node as i64));
                p.set("to_socket", V::Int(to_socket as i64));
                state.queue.run("geo.unlink", p);
            }
            E::Add { kind, x, y } => {
                p.set("kind", V::Enum(kind));
                p.set("x", V::Float(x as f64));
                p.set("y", V::Float(y as f64));
                state.queue.run("geo.add", p);
            }
            E::Remove { nodes } => {
                p.set("nodes", V::Ints(nodes));
                state.queue.run("geo.remove", p);
            }
            E::SetValue {
                node,
                field,
                value,
                continuing,
            } => {
                p.set("node", V::Int(node as i64));
                p.set("field", V::Int(field as i64));
                p.set("value", V::Float(value as f64));
                send_edit(state, "geo.set_value", p, continuing);
            }
            // Colours, images and copying are the material graph's.
            _ => {}
        }
    }
}

/// The shader editor, for the material the Material tab is showing.
///
/// Same slot as the tab: the active object's, the row selected there. One
/// "which material" for both, so the graph on screen and the slot list beside
/// it can never disagree about what is being edited.
fn draw_shader_editor(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    area: kreate_ui::AreaId,
    l: &Labels,
) {
    let slot = state.material_slot;
    let material = doc
        .selection
        .active
        .and_then(|object| doc.material_at(object, slot as usize));
    let view = material
        .and_then(|m| m.graph.as_ref())
        .map(|graph| graph_view(doc, graph, &state.catalog));
    let text = kreate_ui::NodeText {
        no_graph: if material.is_some() {
            l.node_no_graph.clone()
        } else {
            l.node_no_material.clone()
        },
        use_nodes: if material.is_some() {
            l.node_use_nodes.clone()
        } else {
            l.material_new.clone()
        },
        add: l.node_add.clone(),
        no_image: l.node_no_image.clone(),
        open_image: l.node_open_image.clone(),
    };
    let mut editor = state.node_views.remove(&area).unwrap_or_default();
    let mut asked = false;
    let edits = kreate_ui::node_editor(ui, view.as_ref(), &mut editor, &text, &mut asked);
    state.node_views.insert(area, editor);
    if asked {
        if material.is_some() {
            let mut p = PropertyMap::new();
            p.set("slot", kreate_reflect::PropertyValue::Int(slot as i64));
            p.set("enabled", kreate_reflect::PropertyValue::Bool(true));
            state.queue.run("material.use_nodes", p);
        } else {
            state.queue.push(Command::NewMaterial);
        }
    }
    let source = material.and_then(|m| Some((m.name.clone(), m.graph.clone()?)));
    apply_node_edits(state, slot, edits, source);
}

/// Turns what the pointer did in the graph into the operators that do it.
fn apply_node_edits(
    state: &mut ShellState,
    slot: u32,
    edits: Vec<kreate_ui::NodeEdit>,
    material_name: Option<(String, kreate_scene::shader::ShaderGraph)>,
) {
    use kreate_reflect::PropertyValue as V;
    use kreate_ui::NodeEdit as E;
    for edit in edits {
        let mut p = PropertyMap::new();
        p.set("slot", V::Int(slot as i64));
        match edit {
            E::Translate { nodes, x, y } => {
                p.set("nodes", V::Ints(nodes));
                p.set("x", V::Float(x as f64));
                p.set("y", V::Float(y as f64));
                state.queue.run("node.translate", p);
            }
            E::Link {
                from_node,
                from_socket,
                to_node,
                to_socket,
            } => {
                p.set("from_node", V::Int(from_node as i64));
                p.set("from_socket", V::Int(from_socket as i64));
                p.set("to_node", V::Int(to_node as i64));
                p.set("to_socket", V::Int(to_socket as i64));
                state.queue.run("node.link", p);
            }
            E::Unlink { to_node, to_socket } => {
                p.set("to_node", V::Int(to_node as i64));
                p.set("to_socket", V::Int(to_socket as i64));
                state.queue.run("node.unlink", p);
            }
            E::Add { kind, x, y } => {
                p.set("kind", V::Enum(kind));
                p.set("x", V::Float(x as f64));
                p.set("y", V::Float(y as f64));
                state.queue.run("node.add", p);
            }
            E::Remove { nodes } => {
                p.set("nodes", V::Ints(nodes));
                state.queue.run("node.remove", p);
            }
            // A drag in a field is one step, folded the way a slider's is.
            E::SetColor {
                node,
                field,
                swatch,
                continuing,
            } => {
                p.set("node", V::Int(node as i64));
                p.set("field", V::Int(field as i64));
                p.set("color", V::Vec3(swatch.map(|c| c as f64)));
                send_edit(state, "node.set_color", p, continuing);
            }
            E::SetValue {
                node,
                field,
                value,
                continuing,
            } => {
                p.set("node", V::Int(node as i64));
                p.set("field", V::Int(field as i64));
                p.set("value", V::Float(value as f64));
                send_edit(state, "node.set_value", p, continuing);
            }
            E::SetImage { node, uid } => {
                p.set("node", V::Int(node as i64));
                p.set("uid", V::Text(uid));
                state.queue.run("node.set_image", p);
            }
            E::OpenImage { node } => state.queue.push(Command::PickNodeImage { slot, node }),
            E::Duplicate { nodes } => {
                p.set("nodes", V::Ints(nodes));
                p.set("x", V::Float(30.0));
                p.set("y", V::Float(30.0));
                state.queue.run("node.duplicate", p);
            }
            E::Copy { nodes } => {
                if let Some((name, graph)) = material_name.as_ref() {
                    // Where the copies' corner was, so a paste lands them with
                    // that corner at the pointer.
                    let corner = nodes
                        .iter()
                        .filter_map(|n| graph.nodes.get(*n as usize))
                        .fold([f32::MAX, f32::MAX], |c, n| {
                            [c[0].min(n.position[0]), c[1].min(n.position[1])]
                        });
                    state.node_clipboard = Some((name.clone(), nodes, corner));
                }
            }
            E::Group { nodes } => {
                p.set("nodes", V::Ints(nodes));
                state.queue.run("node.group", p);
            }
            E::Ungroup { nodes } => {
                p.set("nodes", V::Ints(nodes));
                state.queue.run("node.ungroup", p);
            }
            E::Paste { x, y } => {
                if let Some((source, nodes, corner)) = state.node_clipboard.clone() {
                    p.set("source", V::Text(source));
                    p.set("nodes", V::Ints(nodes));
                    p.set("x", V::Float((x - corner[0]) as f64));
                    p.set("y", V::Float((y - corner[1]) as f64));
                    state.queue.run("node.duplicate", p);
                }
            }
        }
    }
}

/// A graph as the editor draws it.
fn graph_view(
    doc: &Document,
    graph: &kreate_scene::shader::ShaderGraph,
    catalog: &kreate_i18n::Catalog,
) -> kreate_ui::GraphView {
    use kreate_scene::shader::{NodeKind, Socket};
    use kreate_ui::node_editor::{Family, FieldView, LinkView, NodeView, SocketType, SocketView};
    let t = |key: &str| catalog.format(key, kreate_i18n::TextKind::Ui, &[]);
    let ty = |s: Socket| match s {
        Socket::Color => SocketType::Color,
        Socket::Scalar => SocketType::Scalar,
        // Blender draws both purple, and so does this: the wire is checked
        // when it is made, not by its colour.
        Socket::Vector | Socket::Coordinate => SocketType::Vector,
    };
    let swatch = |c: kreate_math::SceneRgb| {
        let s = c.as_swatch();
        [s.r, s.g, s.b]
    };
    let nodes = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(n, node)| {
            let kind = &node.kind;
            // A group's sockets are named by the user; the rest are ours.
            let socket_label = |name: &str| {
                if matches!(kind, NodeKind::Group(_)) {
                    name.to_string()
                } else {
                    t(&format!("node-socket-{}", slug(name)))
                }
            };
            let outputs = kind
                .outputs()
                .iter()
                .zip(kind.output_names())
                .map(|(s, name)| SocketView {
                    label: socket_label(&name),
                    ty: ty(*s),
                    connected: false,
                    default: None,
                })
                .collect();
            let inputs =
                kind.inputs()
                    .iter()
                    .zip(kind.input_names())
                    .enumerate()
                    .map(|(i, (s, name))| SocketView {
                        label: socket_label(&name),
                        ty: ty(*s),
                        connected: graph.incoming(n, i).is_some(),
                        default: match (kind, i) {
                            (NodeKind::Surface { base_color, .. }, 0) => Some(FieldView::Color {
                                field: 0,
                                swatch: swatch(*base_color),
                            }),
                            (NodeKind::Surface { metallic, .. }, 1) => Some(FieldView::Value {
                                field: 1,
                                value: *metallic,
                                min: 0.0,
                                max: 1.0,
                                label: String::new(),
                            }),
                            (NodeKind::Surface { roughness, .. }, 2) => Some(FieldView::Value {
                                field: 2,
                                value: *roughness,
                                min: 0.0,
                                max: 1.0,
                                label: String::new(),
                            }),
                            (NodeKind::MixColor { factor }, 0) => Some(FieldView::Value {
                                field: 0,
                                value: *factor,
                                min: 0.0,
                                max: 1.0,
                                label: String::new(),
                            }),
                            (NodeKind::Math { a, .. }, 0) => Some(FieldView::Value {
                                field: 1,
                                value: *a,
                                min: -1000.0,
                                max: 1000.0,
                                label: String::new(),
                            }),
                            (NodeKind::Math { b, .. }, 1) => Some(FieldView::Value {
                                field: 2,
                                value: *b,
                                min: -1000.0,
                                max: 1000.0,
                                label: String::new(),
                            }),
                            (NodeKind::Bump { strength, .. }, 0) => Some(FieldView::Value {
                                field: 0,
                                value: *strength,
                                min: 0.0,
                                max: 1.0,
                                label: String::new(),
                            }),
                            (NodeKind::Bump { distance, .. }, 1) => Some(FieldView::Value {
                                field: 1,
                                value: *distance,
                                min: 0.0,
                                max: 1.0,
                                label: String::new(),
                            }),
                            (NodeKind::Checker { color1, .. }, 0) => Some(FieldView::Color {
                                field: 0,
                                swatch: swatch(*color1),
                            }),
                            (NodeKind::Checker { color2, .. }, 1) => Some(FieldView::Color {
                                field: 1,
                                swatch: swatch(*color2),
                            }),
                            // A texture's point has no number to show: unwired,
                            // it is the surface's own position.
                            _ if *s == Socket::Coordinate => None,
                            // A texture's numbers are its sockets' defaults, in
                            // the order `NodeKind::values` keeps them.
                            (
                                NodeKind::Noise { .. }
                                | NodeKind::Voronoi { .. }
                                | NodeKind::Checker { .. }
                                | NodeKind::Wave { .. },
                                _,
                            ) => {
                                let field = match kind {
                                    // The checker's two colours come first in
                                    // its sockets and after the scale in its numbers.
                                    NodeKind::Checker { .. } => 0,
                                    _ => i,
                                };
                                let (min, max) =
                                    match (kind, field) {
                                        (_, 0) => (0.0, 100.0),
                                        (NodeKind::Noise { .. }, 1)
                                        | (NodeKind::Wave { .. }, 2) => (0.0, 15.0),
                                        (NodeKind::Noise { .. }, 2)
                                        | (NodeKind::Voronoi { .. }, 1) => (0.0, 1.0),
                                        _ => (-100.0, 100.0),
                                    };
                                Some(FieldView::Value {
                                    field: field as u32,
                                    value: kind.values().get(field).copied().unwrap_or(0.0),
                                    min,
                                    max,
                                    label: String::new(),
                                })
                            }
                            _ => None,
                        },
                    })
                    .collect();
            let fields = match kind {
                NodeKind::Color(c) => vec![FieldView::Color {
                    field: 0,
                    swatch: swatch(*c),
                }],
                NodeKind::Value(v) => vec![FieldView::Value {
                    field: 0,
                    value: *v,
                    min: 0.0,
                    max: 10.0,
                    label: String::new(),
                }],
                NodeKind::NormalMap { strength } => vec![FieldView::Value {
                    field: 0,
                    value: *strength,
                    min: 0.0,
                    max: 2.0,
                    label: t("node-socket-strength"),
                }],
                NodeKind::Image {
                    image,
                    projection,
                    blend,
                    scale,
                } => {
                    let mut fields = vec![
                        FieldView::Image {
                            name: image.and_then(|id| doc.image(id)).map(|i| i.name.clone()),
                        },
                        FieldView::Choice {
                            field: 0,
                            value: projection.index(),
                            options: vec![t("node-projection-flat"), t("node-projection-box")],
                        },
                    ];
                    // The box's own numbers, only while it is a box.
                    if *projection == kreate_scene::shader::Projection::Box {
                        fields.push(FieldView::Value {
                            field: 1,
                            value: *blend,
                            min: 0.0,
                            max: 1.0,
                            label: t("node-socket-blend"),
                        });
                        fields.push(FieldView::Value {
                            field: 2,
                            value: *scale,
                            min: 0.0,
                            max: 100.0,
                            label: t("node-socket-scale"),
                        });
                    }
                    fields
                }
                NodeKind::Math { op, .. } => vec![FieldView::Choice {
                    field: 0,
                    value: op.index(),
                    options: kreate_scene::shader::MathOp::ALL
                        .iter()
                        .map(|o| t(&format!("node-math-{}", o.tag())))
                        .collect(),
                }],
                NodeKind::ColorRamp {
                    interpolation,
                    count,
                    stops,
                } => {
                    let mut fields = vec![
                        FieldView::Choice {
                            field: 0,
                            value: interpolation.index(),
                            options: kreate_scene::shader::RampMode::ALL
                                .iter()
                                .map(|m| t(&format!("node-ramp-{}", m.tag())))
                                .collect(),
                        },
                        FieldView::Value {
                            field: 1,
                            value: *count as f32,
                            min: 2.0,
                            max: kreate_scene::shader::RAMP_STOPS as f32,
                            label: t("node-ramp-stops"),
                        },
                    ];
                    // Each stop in use: where it is, then its colour.
                    for (i, (at, colour)) in stops.iter().take(*count as usize).enumerate() {
                        fields.push(FieldView::Value {
                            field: 2 + i as u32,
                            value: *at,
                            min: 0.0,
                            max: 1.0,
                            label: t("node-ramp-position"),
                        });
                        fields.push(FieldView::Color {
                            field: i as u32,
                            swatch: swatch(*colour),
                        });
                    }
                    fields
                }
                NodeKind::Mapping {
                    location,
                    rotation,
                    scale,
                } => [
                    ("node-socket-location", location, -100.0, 100.0),
                    ("node-socket-rotation", rotation, -360.0, 360.0),
                    ("node-socket-scale", scale, -100.0, 100.0),
                ]
                .iter()
                .enumerate()
                .flat_map(|(row, (key, values, min, max))| {
                    let name = t(key);
                    values
                        .iter()
                        .enumerate()
                        .map(move |(axis, v)| FieldView::Value {
                            field: (row * 3 + axis) as u32,
                            value: *v,
                            min: *min,
                            max: *max,
                            label: format!("{name} {}", ["X", "Y", "Z"][axis]),
                        })
                })
                .collect(),
                NodeKind::Wave { kind, .. } => vec![FieldView::Choice {
                    field: 4,
                    value: kind.index(),
                    options: vec![t("node-wave-bands"), t("node-wave-rings")],
                }],
                _ => Vec::new(),
            };
            NodeView {
                // i18n-exempt: a node's name is the document's own text
                title: node.name.clone(),
                position: node.position,
                outputs,
                fields,
                inputs,
                family: match kind {
                    NodeKind::Color(_) | NodeKind::Value(_) | NodeKind::TexCoord => Family::Input,
                    NodeKind::Image { .. }
                    | NodeKind::Noise { .. }
                    | NodeKind::Voronoi { .. }
                    | NodeKind::Checker { .. }
                    | NodeKind::Wave { .. } => Family::Texture,
                    NodeKind::Multiply
                    | NodeKind::MultiplyValue
                    | NodeKind::SeparateRgb
                    | NodeKind::MixColor { .. }
                    | NodeKind::Math { .. }
                    | NodeKind::ColorRamp { .. }
                    | NodeKind::Invert => Family::Converter,
                    NodeKind::NormalMap { .. }
                    | NodeKind::Bump { .. }
                    | NodeKind::Mapping { .. } => Family::Vector,
                    NodeKind::Surface { .. } => Family::Shader,
                    NodeKind::Group(_) => Family::Group,
                },
            }
        })
        .collect();
    kreate_ui::GraphView {
        nodes,
        links: graph
            .links
            .iter()
            .map(|l| LinkView {
                from: (l.from_node, l.from_socket),
                to: (l.to_node, l.to_socket),
            })
            .collect(),
        images: doc
            .images()
            .filter_map(|(id, image)| {
                Some((doc.image_uid(id)?.raw().to_string(), image.name.clone()))
            })
            .collect(),
        addable: NodeKind::ADDABLE
            .iter()
            .map(|tag| t(&format!("node-kind-{}", slug(tag))))
            .collect(),
    }
}

/// A name as a catalog key segment: lower case, words joined by hyphens.
fn slug(name: &str) -> String {
    name.to_lowercase().replace([' ', '_'], "-")
}

/// Lets the pointer move the boundaries between areas.
fn drag_handles(ui: &mut Ui, state: &mut ShellState, layout: &kreate_ui::Layout) {
    for handle in &layout.handles {
        let response = ui.interact(
            handle.rect,
            ui.id().with(("split", handle.id.0)),
            egui::Sense::drag(),
        );
        let horizontal = handle.axis == kreate_ui::Axis::Vertical;
        if response.hovered() || state.dragging == Some(handle.id) {
            ui.ctx().set_cursor_icon(if horizontal {
                egui::CursorIcon::ResizeHorizontal
            } else {
                egui::CursorIcon::ResizeVertical
            });
        }
        if response.drag_started() {
            state.dragging = Some(handle.id);
        }
        // The ratio comes from where the pointer *is*, not from how far it has
        // moved. A boundary driven by accumulated deltas drifts away from the
        // pointer over a long drag, and the clamp at either end makes it drift
        // permanently: the delta keeps being added after the boundary has
        // stopped moving.
        if state.dragging == Some(handle.id) {
            if let Some(at) = response.interact_pointer_pos() {
                // The *parent's* extent, which is what the ratio is a share
                // of. Measuring against the window would divide by the wrong
                // number, and a boundary two levels down would move at a
                // fraction of the pointer's speed.
                let (pos, min, extent) = if horizontal {
                    (at.x, handle.parent.min.x, handle.parent.width())
                } else {
                    (at.y, handle.parent.min.y, handle.parent.height())
                };
                if extent > 0.0 {
                    state.area_edits.push(AreaEdit::Resize(
                        handle.id,
                        (pos - min) / extent,
                        extent,
                    ));
                }
            }
        }
        if response.drag_stopped() {
            state.dragging = None;
        }
    }
}

/// How wide the viewport sidebar is, in interface points.
///
/// Fixed rather than fitted to its contents: a panel that changes width with
/// the object selected moves its own fields every time the selection does, and
/// the fields are what the eye is tracking.
const SIDEBAR_WIDTH: f32 = 220.0;

/// The viewport sidebar: where the active object is, as numbers.
///
/// # Generated, not written
///
/// The fields come from `object.set_transform`'s own schema, through the same
/// `property_editor` the adjust panel uses. That is §1.6's promise being cashed
/// in for the third time: one `#[derive(OpParams)]` gives the redo panel, the
/// scripting binding, the i18n keys and now this, and adding a field to the
/// operator adds it here with nothing written.
///
/// # Why it takes the pointer
///
/// The viewport is painted rather than being a widget, so the application
/// decides by geometry whether a press is a navigation gesture. The sidebar
/// registers its rectangle in `viewport_blockers` — without which a drag on a
/// coordinate field would type a number *and* orbit the camera.
#[allow(clippy::too_many_arguments)]
fn draw_item_sidebar(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    area: kreate_ui::AreaId,
    viewport: egui::Rect,
    l: &Labels,
    mode: kreate_ops::operator::Mode,
) {
    let width = sidebar_width(viewport);
    if width <= 0.0 {
        return;
    }
    let rect = egui::Rect::from_min_max(
        egui::pos2(viewport.max.x - width, viewport.min.y),
        viewport.max,
    );
    state.viewport_blockers.push(rect);

    let theme = kreate_ui::theme::from_ui(ui);
    ui.painter().rect_filled(rect, 0.0, theme.raised);
    ui.painter().rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(1.0, theme.border),
        egui::StrokeKind::Inside,
    );

    let inner = rect.shrink(theme.panel_padding);
    if inner.width() <= 0.0 || inner.height() <= 0.0 {
        return;
    }
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(inner)
            .id_salt(("sidebar", area.0)),
        |ui| {
            ui.set_clip_rect(inner);
            // Blender's tabs, across the top rather than down the side: the
            // sidebar is narrow, and three words fit on one line.
            ui.horizontal(|ui| {
                for (tab, label) in [
                    (SidebarTab::Item, &l.sidebar_item),
                    (SidebarTab::Tool, &l.sidebar_tool),
                    (SidebarTab::View, &l.sidebar_view),
                ] {
                    if ui
                        .selectable_label(state.sidebar_tab == tab, label.as_str())
                        .clicked()
                    {
                        state.sidebar_tab = tab;
                    }
                }
            });
            ui.separator();
            match state.sidebar_tab {
                SidebarTab::Item => {}
                SidebarTab::Tool => {
                    // The header's own controls, one under another. Their
                    // own id space: the header draws the same ones.
                    ui.vertical(|ui| draw_tool_settings(ui, state, area.0 | 0x8000_0000, mode));
                    return;
                }
                SidebarTab::View => {
                    draw_view_sidebar(ui, state, doc, registry, l);
                    return;
                }
            }

            let Some(active) = doc.selection.active.and_then(|id| doc.object(id)) else {
                // Said out loud rather than drawn as three empty boxes: a
                // coordinate field with nothing behind it invites a number that
                // goes nowhere.
                ui.weak(&l.sidebar_nothing);
                return;
            };
            // i18n-exempt: an object's name is the user's own text
            ui.label(&active.name);

            // Editing a mesh, the item is the selection: where its middle is,
            // in the world, to be typed — or the vertex, when it is one.
            if mode.acts_on() == kreate_ops::mode::ActsOn::MeshElements {
                match (
                    kreate_ops::ops::mesh::selection_median(doc),
                    registry.get("mesh.median_set"),
                ) {
                    (Some((median, count)), Some(info)) => {
                        ui.strong(if count == 1 {
                            &l.sidebar_vertex
                        } else {
                            &l.sidebar_median
                        });
                        let mut params = kreate_reflect::PropertyMap::new();
                        params.set(
                            "location",
                            kreate_reflect::PropertyValue::Vec3(median.to_array()),
                        );
                        if kreate_ui::property_editor(
                            ui,
                            &state.catalog,
                            info.id,
                            info.schema,
                            &mut params,
                        ) {
                            let dragging = ui.ctx().input(|i| i.pointer.any_down());
                            send_edit(state, "mesh.median_set", params, dragging);
                        }
                    }
                    _ => {
                        ui.weak(&l.sidebar_nothing);
                    }
                }
                return;
            }

            // In pose mode and the skeleton's edit mode the item is the active
            // bone, as in Blender: its pose, or its head, tail and roll.
            let bone_op = match mode.acts_on() {
                kreate_ops::mode::ActsOn::Bones => Some("pose.set_bone"),
                kreate_ops::mode::ActsOn::BoneJoints => Some("armature.bone_set"),
                _ => None,
            };
            if let Some(op_id) = bone_op {
                let object = doc
                    .selection
                    .active
                    .expect("an active object was just read");
                let Some((index, name)) = active_bone(doc, object) else {
                    ui.weak(&l.sidebar_nothing);
                    return;
                };
                // i18n-exempt: a bone's name is the user's own text
                ui.strong(&name);
                if op_id == "pose.set_bone" {
                    draw_pose_bone_fields(ui, state, doc, registry, object, &name);
                } else {
                    draw_rest_bone_fields(ui, state, doc, registry, object, index);
                }
                return;
            }

            let Some(info) = registry.get("object.set_transform") else {
                return;
            };
            let t = active.transform;
            let [rx, ry, rz] = kreate_scene::animation::to_euler(t.rotation).to_array();
            // The values the operator would need in order to leave things
            // exactly as they are. Everything the panel does is edit one of
            // them and send all of them, which is what makes the operator
            // absolute rather than a delta — see `object.set_transform`.
            let mut params = kreate_reflect::PropertyMap::new();
            params.set(
                "location",
                kreate_reflect::PropertyValue::Vec3(t.translation.to_array()),
            );
            params.set(
                "rotation",
                kreate_reflect::PropertyValue::Vec3([rx, ry, rz]),
            );
            params.set(
                "scale",
                kreate_reflect::PropertyValue::Vec3(t.scale.to_array()),
            );

            let object = doc
                .selection
                .active
                .expect("an active object was just read");
            draw_object_transform(ui, state, doc, info, object, params, l);
            draw_dimensions(ui, state, doc, registry, object, l);
        },
    );
}

/// Which tab of the viewport sidebar is showing: Blender's three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidebarTab {
    /// The active object: where it is, how it is turned, how big it is.
    Item,
    /// The transform settings the header also shows.
    Tool,
    /// The view itself: its lens and the 3D cursor.
    View,
}

/// How wide the sidebar is in a viewport of this size.
fn sidebar_width(viewport: egui::Rect) -> f32 {
    SIDEBAR_WIDTH.min(viewport.width() * 0.5)
}

/// The active object's size in metres along its own axes — the mesh's extent
/// times the scale, Blender's Dimensions — or `None` for an object without
/// a mesh.
fn object_dimensions(doc: &Document, object: ObjectId) -> Option<[f64; 3]> {
    let o = doc.object(object)?;
    let kreate_scene::document::ObjectData::Mesh(mesh) = o.data else {
        return None;
    };
    let bounds = doc.mesh(mesh)?.bounds();
    if bounds.min.x > bounds.max.x {
        return None;
    }
    Some(
        ((bounds.max - bounds.min).as_dvec3() * o.transform.scale)
            .abs()
            .to_array(),
    )
}

/// Dimensions, under the transform: typing a size sets the scale that gives it.
fn draw_dimensions(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    object: ObjectId,
    l: &Labels,
) {
    let (Some(size), Some(info)) = (
        object_dimensions(doc, object),
        registry.get("object.set_dimensions"),
    ) else {
        return;
    };
    ui.separator();
    ui.strong(&l.sidebar_dimensions);
    let mut params = PropertyMap::new();
    params.set("dimensions", kreate_reflect::PropertyValue::Vec3(size));
    if kreate_ui::property_editor(
        ui,
        &state.catalog,
        "sidebar.dimensions",
        info.schema,
        &mut params,
    ) {
        let dragging = ui.ctx().input(|i| i.pointer.any_down());
        send_edit(state, "object.set_dimensions", params, dragging);
    }
}

/// The View tab: the viewport's lens and the 3D cursor.
fn draw_view_sidebar(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    l: &Labels,
) {
    ui.strong(&l.sidebar_lens);
    let mut lens = state.view_lens_mm;
    if ui
        .add(
            egui::DragValue::new(&mut lens)
                .range(1.0..=1000.0)
                .speed(0.5)
                .max_decimals(1),
        )
        .changed()
    {
        state.view_lens_mm = lens;
        let mut params = PropertyMap::new();
        params.set("lens", kreate_reflect::PropertyValue::Float(lens));
        state.queue.dispatch("view.set_lens", params);
    }
    ui.separator();
    ui.strong(&l.sidebar_cursor);
    let Some(info) = registry.get("scene.cursor_set") else {
        return;
    };
    let mut params = PropertyMap::new();
    params.set(
        "location",
        kreate_reflect::PropertyValue::Vec3(doc.cursor.to_array()),
    );
    if kreate_ui::property_editor(
        ui,
        &state.catalog,
        "sidebar.cursor",
        info.schema,
        &mut params,
    ) {
        let dragging = ui.ctx().input(|i| i.pointer.any_down());
        send_edit(state, "scene.cursor_set", params, dragging);
    }
}

/// The workspace tabs.
///
/// A row of them rather than a dropdown: switching layout is something a
/// modeller does dozens of times an hour, and a tab is one click where a
/// dropdown is two. It is also the only affordance that shows the *set* — a
/// user who cannot see that other layouts exist does not go looking for them.
fn draw_workspace_switcher(ui: &mut Ui, state: &mut ShellState) {
    let active = state.workspaces.active_index();
    // Collected first: the loop borrows the queue, and iterating the list while
    // pushing to it is the same mistake the layout edits are deferred to avoid.
    let names: Vec<String> = state
        .workspaces
        .iter()
        .map(|space| {
            // The built-in layouts' names are the program's words and are
            // translated; a name the user gave one is theirs.
            let key = format!("workspace-{}", space.name.to_lowercase().replace(' ', "-"));
            if state.catalog.has(&key) {
                state.catalog.get(&key)
            } else {
                space.name.clone()
            }
        })
        .collect();
    // One dropdown at the toolbar's left, as Fusion keeps its workspaces: the
    // layout is chosen far less often than a tool, and a row of tabs took the
    // width the tools want.
    let current = names.get(active).cloned().unwrap_or_default();
    egui::ComboBox::from_id_salt("workspace")
        .width(118.0)
        .selected_text(egui::RichText::new(current).strong())
        .show_ui(ui, |ui| {
            for (index, name) in names.iter().enumerate() {
                if ui.selectable_label(index == active, name).clicked() {
                    state.switch_workspace(index);
                }
            }
        });
}

/// The edit-mode menus: Mesh, Edge and Face.
///
/// **Blender's grouping, on purpose.** What a command acts *on* decides which
/// menu it is in — Bevel under Edge, Inset under Face — which is why two
/// commands that both "make the shape bigger somewhere" are filed apart. A
/// modeller already knows this arrangement, and the alternative on offer was
/// the fifteen flat buttons this replaces.
///
/// There is no Vertex menu, which Blender does have. Nothing here is
/// vertex-specific: Merge and Weld read the element mode rather than assuming
/// one, so they belong under Mesh. An empty menu would be a promise the
/// application does not keep.
fn mesh_menus(l: &Labels, select_mode: u32) -> Vec<TopMenu> {
    use kreate_reflect::PropertyValue as V;
    use MenuNode::Separator;
    vec![
        TopMenu {
            label: l.menu_mesh.clone(),
            nodes: vec![
                // Merge and dissolve both act on whatever the element mode is,
                // so the entry carries it rather than asking again.
                entry(
                    "mesh.merge",
                    &l.merge,
                    op("mesh.merge", &[("target", V::Enum(0))]),
                ),
                entry(
                    "mesh.merge.by_distance",
                    &l.weld,
                    op("mesh.merge", &[("distance", V::Float(1e-4))]),
                ),
                Separator,
                entry(
                    "mesh.dissolve",
                    &l.dissolve,
                    op("mesh.dissolve", &[("mode", V::Enum(select_mode))]),
                ),
                entry(
                    "mesh.dissolve_limited",
                    &l.dissolve_limited,
                    op("mesh.dissolve_limited", &[]),
                ),
                entry(
                    "mesh.delete",
                    &l.delete,
                    op("mesh.delete", &[("mode", V::Enum(select_mode))]),
                ),
                Separator,
                entry("mesh.spin", &l.spin, op("mesh.spin", &[])),
                entry("mesh.knife", &l.knife, dispatch("mesh.knife")),
                Separator,
                // Shading is a property of the mesh rather than of a selection,
                // which is why it is here and not under Face.
                MenuNode::Entry(
                    MenuEntry::new(
                        "mesh.shade",
                        &l.shade_smooth,
                        op("mesh.shade", &[("smooth", V::Bool(true))]),
                    )
                    .variant("smooth", V::Bool(true)),
                ),
                MenuNode::Entry(
                    MenuEntry::new(
                        "mesh.shade",
                        &l.shade_flat,
                        op("mesh.shade", &[("smooth", V::Bool(false))]),
                    )
                    .variant("smooth", V::Bool(false)),
                ),
                entry(
                    "mesh.normals",
                    &l.recalc_normals,
                    op("mesh.normals", &[("action", V::Enum(0))]),
                ),
                Separator,
                // Blender's Mesh ▸ Show/Hide.
                MenuNode::Sub(
                    l.show_hide.clone(),
                    vec![
                        entry("mesh.reveal", &l.reveal, op("mesh.reveal", &[])),
                        MenuNode::Entry(
                            MenuEntry::new(
                                "mesh.hide",
                                &l.hide_selected,
                                op(
                                    "mesh.hide",
                                    &[
                                        ("unselected", V::Bool(false)),
                                        ("mode", V::Enum(select_mode)),
                                    ],
                                ),
                            )
                            .variant("unselected", V::Bool(false)),
                        ),
                        MenuNode::Entry(
                            MenuEntry::new(
                                "mesh.hide",
                                &l.hide_unselected,
                                op(
                                    "mesh.hide",
                                    &[
                                        ("unselected", V::Bool(true)),
                                        ("mode", V::Enum(select_mode)),
                                    ],
                                ),
                            )
                            .variant("unselected", V::Bool(true)),
                        ),
                    ],
                ),
            ],
        },
        TopMenu {
            label: l.menu_edge.clone(),
            nodes: vec![
                entry(
                    "mesh.bevel",
                    &l.bevel,
                    op("mesh.bevel", &[("width", V::Float(0.1))]),
                ),
                MenuNode::Entry(
                    MenuEntry::new(
                        "mesh.fillet",
                        &l.m("mech-fillet"),
                        op("mesh.fillet", &[("chamfer", V::Bool(false))]),
                    )
                    .variant("chamfer", V::Bool(false)),
                ),
                MenuNode::Entry(
                    MenuEntry::new(
                        "mesh.fillet",
                        &l.m("mech-chamfer"),
                        op("mesh.fillet", &[("chamfer", V::Bool(true))]),
                    )
                    .variant("chamfer", V::Bool(true)),
                ),
                entry(
                    "mesh.loop_cut",
                    &l.loop_cut,
                    op(
                        "mesh.loop_cut",
                        &[("cuts", V::Int(1)), ("position", V::Float(0.5))],
                    ),
                ),
                entry(
                    "mesh.subdivide",
                    &l.subdivide,
                    op("mesh.subdivide", &[("iterations", V::Int(1))]),
                ),
                entry("mesh.bridge", &l.bridge, op("mesh.bridge", &[])),
                Separator,
                MenuNode::Entry(
                    MenuEntry::new(
                        "mesh.select",
                        &l.select_loop,
                        op("mesh.select", &[("action", V::Enum(3))]),
                    )
                    .variant("action", V::Enum(3)),
                ),
                Separator,
                // By angle rather than by selection: somebody reaching for this
                // on a freshly imported model has nothing selected, and thirty
                // degrees is the answer they want.
                entry(
                    "mesh.mark_sharp",
                    &l.mark_sharp,
                    op(
                        "mesh.mark_sharp",
                        &[("angle", V::Float(30f64.to_radians()))],
                    ),
                ),
                Separator,
                // Where Unwrap cuts the surface open.
                MenuNode::Entry(
                    MenuEntry::new(
                        "mesh.mark_seam",
                        &l.mark_seam,
                        op("mesh.mark_seam", &[("clear", V::Bool(false))]),
                    )
                    .variant("clear", V::Bool(false)),
                ),
                MenuNode::Entry(
                    MenuEntry::new(
                        "mesh.mark_seam",
                        &l.clear_seam,
                        op("mesh.mark_seam", &[("clear", V::Bool(true))]),
                    )
                    .variant("clear", V::Bool(true)),
                ),
            ],
        },
        TopMenu {
            label: l.menu_face.clone(),
            nodes: vec![
                entry(
                    "mesh.extrude_region",
                    &l.extrude,
                    op("mesh.extrude_region", &[("offset", V::Float(1.0))]),
                ),
                entry(
                    "mesh.inset",
                    &l.inset,
                    op("mesh.inset", &[("thickness", V::Float(0.1))]),
                ),
                Separator,
                entry("mesh.fill", &l.fill, op("mesh.fill", &[])),
                Separator,
                entry("mesh.shell", &l.m("mech-shell"), op("mesh.shell", &[])),
                entry("mesh.draft", &l.m("mech-draft"), op("mesh.draft", &[])),
            ],
        },
    ]
}

/// How wide the right-click menu is, in interface points.
///
/// Fixed rather than fitted to the longest row: a menu that changes width with
/// its contents moves its own shortcut column every time the mode changes, and
/// the column is what the eye tracks.
const MENU_WIDTH: f32 = 220.0;

/// Draws the right-click menu, if one is open.
///
/// A window rather than egui's own `context_menu`, because the viewport is
/// painted rather than being a widget with a response — and because the
/// application decides when the menu opens: a right-click that cancels a
/// transform must not also open a menu, and only the application knows a
/// transform is running.
fn draw_context_menu(ui: &mut egui::Ui, state: &mut ShellState, viewport: egui::Rect) {
    let Some(at) = state.context_menu.at else {
        return;
    };
    // In the viewport it was opened in, and no other: every 3D viewport
    // draws its overlays, and a menu drawn by each was two menus with one
    // set of widget ids between them.
    if !viewport.contains(at) {
        return;
    }
    let mut close = false;
    let shown = egui::Area::new(egui::Id::new("viewport-context-menu"))
        .order(egui::Order::Foreground)
        .fixed_pos(at)
        // Kept inside the viewport, so a right-click near the bottom edge does
        // not put half the menu off screen where it cannot be read or reached.
        .constrain_to(viewport)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                // Bounded, or the popup takes every pixel to the right of where
                // it opened: the rows lay out right-to-left to put each shortcut
                // at the far edge, and "the far edge" of an unconstrained area is
                // the edge of the window. The first picture of this menu had its
                // shortcuts six hundred pixels from their labels.
                ui.set_max_width(MENU_WIDTH);
                ui.set_min_width(MENU_WIDTH);
                if kreate_ui::context_menu::menu_body(
                    ui,
                    &state.context_menu.groups,
                    &mut state.queue,
                ) {
                    close = true;
                }
            });
        });

    // A press anywhere else dismisses it, which is what every menu everywhere
    // does and what a user will try first.
    //
    // A *press*, not a click. The right-click that opened the menu is still
    // coming up when the menu appears, and its release reads as a click; the
    // viewport only draws when something happens, so that release was
    // processed on the next pointer move — and the menu vanished the moment
    // the mouse moved towards it. A press outside the menu is never that.
    let menu_rect = shown.response.rect;
    if ui.ctx().input(|i| {
        let outside = i
            .pointer
            .interact_pos()
            .is_some_and(|p| !menu_rect.contains(p));
        (i.pointer.any_pressed()
            && outside
            && !i.pointer.button_pressed(egui::PointerButton::Secondary))
            || i.key_pressed(egui::Key::Escape)
    }) {
        close = true;
    }
    if close {
        state.context_menu.at = None;
    }
}

/// Everything the remapping panel says, resolved from the catalogue.
///
/// The counted lines are interpolated here rather than in the panel: Korean
/// needs the number and the noun to agree in a way a format string cannot
/// express, which is why the catalogue is Fluent.
fn keymap_text(catalog: &Catalog, changed: usize, shadowed: usize) -> kreate_ui::KeymapText {
    let t = |key: &str| catalog.format(key, TextKind::Ui, &[]);
    kreate_ui::KeymapText {
        preset: t("keymap-preset"),
        changed: catalog.format(
            "keymap-changed",
            TextKind::Ui,
            &[("count", &changed.to_string())],
        ),
        reset_all: t("keymap-reset-all"),
        shadowed: if shadowed == 0 {
            String::new()
        } else {
            catalog.format(
                "keymap-shadowed",
                TextKind::Ui,
                &[("count", &shadowed.to_string())],
            )
        },
        reset_hint: t("keymap-reset-hint"),
        shadowed_hint: t("keymap-shadowed-hint"),
        capture_hint: t("keymap-capture-hint"),
        capturing: t("keymap-capturing"),
        contexts: kreate_ui::keymap_editor::CONTEXT_ORDER
            .into_iter()
            .map(|c| t(kreate_ui::keymap_editor::context_key(c)))
            .collect(),
    }
}

/// The rubber band, over the viewport.
///
/// Drawn by the interface because it is a screen-space annotation with no
/// geometry behind it, and egui already owns the layer above the viewport image.
/// What it *selects* is decided by a compute pass over the id buffer from the
/// same rectangle, so the outline and the result cannot disagree.
/// Blender's header tool settings: transform orientation, snapping and
/// proportional editing. The selection tool is the toolbar's.
fn draw_tool_settings(
    ui: &mut Ui,
    state: &mut ShellState,
    area: u32,
    mode: kreate_ops::operator::Mode,
) {
    let get = |state: &ShellState, key: &str| state.catalog.get(key);
    let mut tools = state.tools;

    // Transform orientation.
    let orientations = [
        get(state, "orientation-global"),
        get(state, "orientation-local"),
        get(state, "orientation-cursor"),
    ];
    egui::ComboBox::from_id_salt(("orientation", area))
        .width(58.0)
        .selected_text(orientations[tools.orientation.min(2) as usize].as_str())
        .show_ui(ui, |ui| {
            for (i, name) in orientations.iter().enumerate() {
                ui.selectable_value(&mut tools.orientation, i as u32, name.as_str());
            }
        })
        .response
        .on_hover_text(get(state, "orientation-hint"));

    ui.separator();
    // Snapping, Blender's magnet: one dropdown for whether and to what, lit
    // when on — two controls side by side did not fit a narrow viewport.
    let theme = kreate_ui::theme::from_ui(ui);
    let lit = |text: String, on: bool| {
        let t = egui::RichText::new(text);
        if on {
            t.color(theme.accent)
        } else {
            t
        }
    };
    let modes: Vec<String> = crate::snap::Mode::ALL
        .iter()
        .map(|m| get(state, m.label_key()))
        .collect();
    egui::ComboBox::from_id_salt(("snap", area))
        .width(56.0)
        .selected_text(lit(get(state, "snap-toggle"), tools.snap))
        .show_ui(ui, |ui| {
            ui.checkbox(&mut tools.snap, get(state, "snap-on"));
            ui.separator();
            for (i, name) in modes.iter().enumerate() {
                ui.selectable_value(&mut tools.snap_mode, i as u32, name.as_str());
            }
        })
        .response
        .on_hover_text(get(state, "snap-hint"));

    // Proportional editing: elements only, as the key.
    if mode.edits_elements() {
        let curves: Vec<String> = kreate_ops::ops::mesh::FALLOFF_CURVES
            .iter()
            .map(|c| get(state, &format!("falloff-{c}")))
            .collect();
        egui::ComboBox::from_id_salt(("proportional", area))
            .width(88.0)
            .selected_text(lit(get(state, "proportional-toggle"), tools.proportional))
            .show_ui(ui, |ui| {
                ui.checkbox(&mut tools.proportional, get(state, "proportional-on"));
                ui.separator();
                for (i, name) in curves.iter().enumerate() {
                    ui.selectable_value(&mut tools.falloff, i as u32, name.as_str());
                }
            })
            .response
            .on_hover_text(get(state, "proportional-hint"));
        ui.toggle_value(&mut tools.auto_merge, get(state, "auto-merge"))
            .on_hover_text(get(state, "auto-merge-hint"));
    }

    state.tools = tools;
}

fn draw_select_band(ui: &mut Ui, state: &ShellState, viewport: egui::Rect) {
    // A lasso: its path, closed back to where it began.
    if state.select_lasso.len() >= 2 {
        let theme = kreate_ui::theme::from_ui(ui);
        let colour = if state.select_band_deselect {
            theme.info
        } else {
            theme.accent
        };
        let mut points: Vec<egui::Pos2> = state
            .select_lasso
            .iter()
            .map(|p| viewport.min + p.to_vec2())
            .collect();
        points.push(points[0]);
        ui.painter()
            .with_clip_rect(viewport)
            .add(egui::Shape::line(points, egui::Stroke::new(1.5, colour)));
    }
    let Some((from, to)) = state.select_band else {
        return;
    };
    let band = egui::Rect::from_two_pos(viewport.min + from.to_vec2(), viewport.min + to.to_vec2());
    // Clipped to the viewport: a drag that leaves it should stop at the edge
    // rather than draw over the panels.
    let band = band.intersect(viewport);
    if band.width() < 1.0 || band.height() < 1.0 {
        return;
    }

    let painter = ui.painter().with_clip_rect(viewport);
    // Warm for adding, cool for removing, so which one is happening is visible
    // without reading the modifier keys off your own hand.
    let theme = kreate_ui::theme::from_ui(ui);
    let colour = if state.select_band_deselect {
        theme.info
    } else {
        theme.accent
    };
    painter.rect_filled(band, 0.0, colour.gamma_multiply(0.12));
    painter.rect_stroke(
        band,
        0.0,
        egui::Stroke::new(1.0, colour),
        egui::StrokeKind::Inside,
    );
}

/// Queues a discard for every offer.
///
/// A function rather than three lines inside the button, so the behaviour can
/// be tested: a pixel test can see that the panel is bounded but not that this
/// affordance exists, and with forty-two offers it is the one a user needs
/// most. A falsification sweep deleted the button and nothing failed.
pub fn discard_all_recoveries(state: &mut ShellState) {
    // Every offer named individually rather than a single "discard all"
    // command, because the application answers each one by forgetting a path —
    // and one command per path means the two cannot disagree about which
    // paths were meant.
    for offer in &state.recoverable {
        state.queue.push(Command::DiscardRecovery {
            path: offer.path.clone(),
        });
    }
}

/// How many rows the recovery offer shows before it starts scrolling.
///
/// Four is enough to see that there are several and to reach the newest
/// without scrolling, and small enough that the panel never dominates the
/// window however many there are.
const RECOVERY_VISIBLE_ROWS: f32 = 4.0;

/// The height of one row of the recovery offer, including its spacing.
fn recovery_row_height(ui: &Ui) -> f32 {
    ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y * 2.0
}

/// The recovery offer, as a submenu of File — Blender's File ▸ Recover.
///
/// It was a panel across the top of the window, which made sense when there
/// was nowhere else for it and stopped making sense the moment the top bar
/// became the application's menus: a strip of buttons above File is a second,
/// unlabelled menu. Under File it is where a user who has just lost a session
/// looks first, and it costs the viewport nothing while it waits there. The
/// status bar says the offers exist, so nobody has to know to look.
///
/// One row each in a list that scrolls after a few, the count above and a way
/// to dismiss the lot below. A stale session directory per crashed process is
/// normal, and a run of interrupted tests left forty-two.
/// Help ▸ Connect to AI Apps: each assistant, connected or not, and a click
/// to add Kreate to it. Where each stands is read as the menu opens.
fn draw_connect_menu(ui: &mut Ui, state: &mut ShellState) {
    use crate::connect::{App, Status};
    let t = |key: &str| state.catalog.format(key, kreate_i18n::TextKind::Ui, &[]);
    let (title, hint, connected, absent, missing) = (
        t("connect-menu"),
        state
            .catalog
            .format("connect-hint", kreate_i18n::TextKind::Tooltip, &[]),
        t("connect-connected"),
        t("connect-absent"),
        t("connect-no-connector"),
    );
    ui.menu_button(&title, |ui| {
        let Some(connector) = crate::claude::find_connector() else {
            ui.add_enabled(false, egui::Button::new(&missing));
            return;
        };
        for (index, app) in App::ALL.iter().enumerate() {
            let status = app.status(&connector);
            // i18n-exempt: an application's own name
            let label = match status {
                Status::Connected => format!("✓ {}  ({connected})", app.name()),
                Status::Ready => app.name().to_string(),
                Status::Absent => format!("{}  ({absent})", app.name()),
            };
            let response = ui
                .add_enabled(status != Status::Absent, egui::Button::new(label))
                .on_hover_text(&hint);
            if response.clicked() {
                state.queue.push(Command::ConnectApp { index });
                ui.close();
            }
        }
    });
}

/// File ▸ Open Recent: the name, the whole path on hover, and a document that
/// is no longer there greyed out.
fn draw_recent_menu(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    ui.menu_button(&l.file_open_recent, |ui| {
        if state.recent_files.is_empty() {
            ui.add_enabled(false, egui::Button::new(&l.recent_none));
            return;
        }
        let files = state.recent_files.clone();
        for path in files {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let response = ui
                .add_enabled(path.exists(), egui::Button::new(name))
                .on_hover_text(path.display().to_string());
            if response.clicked() {
                state.queue.push(Command::OpenPath(path));
                ui.close();
            }
        }
    });
}

fn draw_recover_menu(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    ui.menu_button(&l.recover_menu, |ui| {
        let count = state.recoverable.len();
        if count == 0 {
            ui.add_enabled(false, egui::Button::new(&l.recover_none));
            return;
        }
        let heading = state.catalog.format(
            "recover-count",
            TextKind::Ui,
            &[("count", &count.to_string())],
        );
        ui.label(egui::RichText::new(heading).strong());
        ui.separator();

        let row = recovery_row_height(ui);
        // A clone, because the buttons queue commands that will mutate this
        // list and the loop would otherwise hold a borrow across it.
        let offers: Vec<(std::path::PathBuf, String, String)> = state
            .recoverable
            .iter()
            .map(|o| (o.path.clone(), o.name.clone(), o.age.clone()))
            .collect();
        egui::ScrollArea::vertical()
            .max_height(row * RECOVERY_VISIBLE_ROWS * 1.6)
            .show(ui, |ui| {
                ui.set_min_width(320.0);
                for (path, name, age) in offers {
                    ui.horizontal(|ui| {
                        // The full path on hover. Forty autosaves called
                        // `untitled.kreate` differ only by the session
                        // directory they came from, and the user occasionally
                        // needs to see it.
                        ui.label(&name).on_hover_text(path.display().to_string());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let discard =
                                ui.button(&l.recover_discard)
                                    .on_hover_text(state.catalog.format(
                                        "op-recover-discard-desc",
                                        TextKind::Tooltip,
                                        &[],
                                    ));
                            if discard.clicked() {
                                state
                                    .queue
                                    .push(Command::DiscardRecovery { path: path.clone() });
                            }
                            if menu_item(ui, state, "recover.open", &[], &l.recover_open) {
                                state
                                    .queue
                                    .push(Command::RecoverDocument { path: path.clone() });
                            }
                            ui.weak(&age);
                        });
                    });
                }
            });
        ui.separator();
        // Present from the first offer, not only when the list is long: a
        // user who wants none of them should not have to click forty times
        // to say so.
        if menu_item(
            ui,
            state,
            "recover.discard_all",
            &[],
            &l.recover_discard_all,
        ) {
            discard_all_recoveries(state);
        }
    });
}

/// The running transform's header, over the top of the viewport.
///
/// Painted rather than laid out as a widget: it has to be centred over the
/// viewport and must not take pointer input, since every pointer event during
/// a drag belongs to the gesture.
///
/// The border carries the constrained axis's colour — the same three colours
/// as the grid, the navigation gizmo and the transform fields. That is the
/// whole reason those are one table: after a week the user stops reading the
/// letter.
fn draw_modal_header(ui: &mut Ui, state: &ShellState, viewport: egui::Rect) {
    let Some(header) = &state.modal else {
        return;
    };
    let theme = kreate_ui::theme::from_ui(ui);
    let edge = header.axis.map_or(theme.accent, |a| theme.axis(a));

    let painter = ui.painter().with_clip_rect(viewport);
    let font = egui::TextStyle::Body.resolve(ui.style());
    let galley = painter.layout_no_wrap(header.text.clone(), font, theme.text);

    let pad = egui::vec2(10.0, 5.0);
    let size = galley.size() + pad * 2.0;
    let centre = egui::pos2(viewport.center().x, viewport.top() + 12.0 + size.y * 0.5);
    let rect = egui::Rect::from_center_size(centre, size);
    if !viewport.contains_rect(rect) {
        return;
    }

    let radius = egui::CornerRadius::same(theme.radius);
    // Opaque enough to read over a light model, not so opaque it hides what is
    // being dragged.
    painter.rect_filled(rect, radius, theme.raised.gamma_multiply(0.94));
    painter.rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, edge),
        egui::StrokeKind::Inside,
    );
    painter.galley(rect.min + pad, galley, theme.text);
}

/// The navigation gizmo, over the top-right of the viewport.
///
/// Drawn by the interface rather than by the renderer: it is a widget, it needs
/// hover and click handling, and egui already owns the layer above the viewport
/// image. Putting it in a wgpu pass would mean rebuilding pointer handling that
/// exists three lines away.
///
/// One drag must not both turn the gizmo and turn the scene. That is settled by
/// the interface claiming the press: `ui.interact` here makes egui report the
/// press as consumed, and the viewport decides what a drag does once, when it
/// begins. It used to be a flag the viewport consulted on every motion, which
/// went stale between frames — the interface only runs when something redraws.
/// the gizmo and orbit the scene, at twice the rate.
fn draw_nav_gizmo(ui: &mut Ui, state: &mut ShellState, viewport: egui::Rect) {
    use kreate_ui::nav_gizmo;

    let radius = nav_gizmo::radius_for(viewport);
    let centre = nav_gizmo::centre_for(viewport, radius);
    let balls = nav_gizmo::layout(state.camera_rotation, centre, radius);

    state.gizmo_circle = Some((centre, radius));

    let area = egui::Rect::from_center_size(centre, egui::Vec2::splat(radius * 2.2));
    let response = ui.interact(
        area,
        ui.id().with("nav-gizmo"),
        egui::Sense::click_and_drag(),
    );

    let pointer = response
        .hover_pos()
        .or_else(|| response.interact_pointer_pos());

    let hovered = pointer.and_then(|p| nav_gizmo::hit(&balls, p, nav_gizmo::ball_radius(radius)));

    if response.drag_delta() != egui::Vec2::ZERO {
        let d = response.drag_delta();
        state.queue.push(Command::OrbitBy { dx: d.x, dy: d.y });
    } else if response.clicked() {
        // Only a click, never a drag that happened to end on a ball: snapping
        // after an orbit would undo the orbit the user just performed.
        if let Some(axis) = hovered {
            state.queue.push(Command::SnapToAxis(axis));
        }
    }

    nav_gizmo::draw(ui.painter(), centre, radius, &balls, hovered);
}

/// The adjust-last-operation panel.
///
/// Entirely generated from the operator's reflected schema: this function does
/// not know what any operator is. Editing a value emits `AdjustLast`, and the
/// application restores and re-runs — the same path redo takes, which is why
/// this needs no per-operator support.
/// Blender's Adjust Last Operation panel, in the viewport's bottom-left
/// corner: folded to the operation's name, opened to its parameters.
///
/// Where the hand already is after an operation. The same panel as the Tool
/// tab's, so the two cannot come to adjust different things.
fn draw_adjust_float(
    ui: &mut Ui,
    state: &mut ShellState,
    registry: &OperatorRegistry,
    history: &History,
    viewport: egui::Rect,
) {
    let Some(info) = history.last().and_then(|entry| registry.get(&entry.op_id)) else {
        return;
    };
    // An operation with nothing to adjust — a selection, a paste — has no
    // panel, as in Blender.
    if info.schema.iter().all(|d| d.hidden) {
        return;
    }
    let label = state.catalog.format(&info.name_key(), TextKind::Ui, &[]);
    let label = if label == info.name_key() {
        info.label.to_string()
    } else {
        label
    };
    let shown = egui::Area::new(egui::Id::new((
        "adjust-float",
        viewport.min.x as i32,
        viewport.min.y as i32,
    )))
    .order(egui::Order::Middle)
    // Top right, under the navigation gizmo, where Fusion opens a command's
    // dialog: beside the model rather than over the corner the eye starts in.
    .pivot(egui::Align2::RIGHT_TOP)
    .fixed_pos(viewport.right_top() + egui::vec2(-12.0, 132.0))
    .constrain_to(viewport)
    .show(ui.ctx(), |ui| {
        let theme = kreate_ui::theme::from_ui(ui);
        egui::Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_width(260.0);
            // The title strip: the operation's name in capitals, as a
            // dialog's heading.
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(label.to_uppercase())
                        .strong()
                        .small()
                        .color(theme.text),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let open = state.adjust_open;
                    let toggle = state
                        .catalog
                        .get(if open { "adjust-done" } else { "adjust-edit" });
                    if ui.small_button(toggle).clicked() {
                        state.adjust_open = !open;
                    }
                });
            });
            if state.adjust_open {
                ui.separator();
                draw_adjust_fields(ui, state, registry, history);
            }
        });
    });
    // Over the picture: a click on the panel must not select what is behind.
    state.viewport_blockers.push(shown.response.rect);
}

fn draw_adjust_panel(
    ui: &mut Ui,
    state: &mut ShellState,
    registry: &OperatorRegistry,
    history: &History,
) {
    if let Some(info) = history.last().and_then(|entry| registry.get(&entry.op_id)) {
        let label = state.catalog.format(&info.name_key(), TextKind::Ui, &[]);
        let label = if label == info.name_key() {
            info.label.to_string()
        } else {
            label
        };
        ui.label(label);
        ui.separator();
    }
    draw_adjust_fields(ui, state, registry, history);
}

/// The last operation's parameters, editable: what both adjust panels draw.
fn draw_adjust_fields(
    ui: &mut Ui,
    state: &mut ShellState,
    registry: &OperatorRegistry,
    history: &History,
) {
    let Some(entry) = history.last() else {
        ui.weak(state.catalog.format("adjust-nothing", TextKind::Ui, &[]));
        return;
    };
    let Some(info) = registry.get(&entry.op_id) else {
        return;
    };

    // Reset the editable copy whenever the last operation changes, so the panel
    // always describes what is actually applied.
    let stale = state
        .adjust_params
        .as_ref()
        .map(|(id, _)| id != &entry.op_id)
        .unwrap_or(true);
    if stale {
        state.adjust_params = Some((entry.op_id.clone(), entry.params.clone()));
    }

    let Some((_, params)) = state.adjust_params.as_mut() else {
        return;
    };
    let changed = kreate_ui::property_editor::property_editor(
        ui,
        &state.catalog,
        info.id,
        info.schema,
        params,
    );
    if changed {
        state.queue.push(Command::AdjustLast {
            params: params.clone(),
        });
    }
}

fn shading_label(mode: kreate_render::ShadingMode, l: &Labels) -> &str {
    match mode {
        kreate_render::ShadingMode::Studio => &l.shading_studio,
        kreate_render::ShadingMode::Flat => &l.shading_flat,
        kreate_render::ShadingMode::Wireframe => &l.shading_wireframe,
        kreate_render::ShadingMode::Rendered => &l.shading_rendered,
        kreate_render::ShadingMode::Material => &l.shading_material,
    }
}

/// The exposure the picture is being made at.
///
/// Only in the mode that has one. Studio and flat shading light themselves, so
/// an exposure there would be a number that does nothing — which is worse than
/// no number at all.
///
/// It is a readout and not a dial, deliberately, and the dial is the camera's
/// ±EV. What this is for is explaining the picture: with the meter on, turning
/// a lamp down moves this and leaves the picture alone, and seeing it move is
/// the difference between "exposure is doing its job" and "the slider is
/// broken".
/// The rendered view's readout: the exposure, and while the path tracer is
/// summing, how far it has got.
fn exposure_text(state: &ShellState, l: &Labels) -> String {
    let exposure = format!("{} {:+.1}", l.exposure, state.view_status.exposure);
    match state.view_status.samples {
        0 => exposure,
        // i18n-exempt: numbers
        n => format!("{exposure}   {n}/{}", kreate_render::pathtrace::SAMPLES),
    }
}

fn draw_exposure(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    if state.view_status.shading != kreate_render::ShadingMode::Rendered {
        return;
    }
    let text = exposure_text(state, l);
    let hint = if state.view_status.metered {
        &l.exposure_metered
    } else {
        &l.exposure_manual
    };
    ui.label(text).on_hover_text(hint);
}

/// The properties editor: a column of tabs, and the one that is chosen.
///
/// Which tabs exist depends on what is active, and that is the point of asking
/// rather than drawing them all: an object with no datablock worth editing has
/// no Data tab, the way an operator with nothing to act on does not offer
/// itself. Without it every context grows an empty panel, and §2.6 predicted
/// that would be the thing that made the editor unreadable.
#[allow(clippy::too_many_arguments)]
fn draw_properties(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    history: &History,
    modifier_rows: &mut [kreate_ui::ModifierRow],
    area: kreate_ui::AreaId,
    l: &Labels,
    mode: kreate_ops::operator::Mode,
) {
    let tabs = available_tabs(doc);
    let mut current = chosen_tab(state.properties_tab(area), &tabs);

    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            for tab in &tabs {
                let name = state.catalog.get(tab.label_key());
                // The glyph is the button and the name is the tooltip, which is
                // what keeps the column narrow enough to be a column.
                // Painted, and lit like any selected button.
                let (rect, response) =
                    ui.allocate_exact_size(egui::vec2(26.0, 24.0), egui::Sense::click());
                let theme = kreate_ui::theme::from_ui(ui);
                let visuals = ui.style().interact_selectable(&response, *tab == current);
                if *tab == current || response.hovered() {
                    ui.painter()
                        .rect_filled(rect, theme.radius as f32, visuals.weak_bg_fill);
                }
                let ink = if *tab == current {
                    theme.accent
                } else {
                    theme.text_dim
                };
                let data = doc.selection.active.and_then(|id| doc.object(id)).map_or(
                    RowKind::Empty,
                    |o| match o.data {
                        ObjectData::Light(_) => RowKind::Light,
                        ObjectData::Camera(_) => RowKind::Camera,
                        ObjectData::Armature(_) => RowKind::Armature,
                        ObjectData::Mesh(_) => RowKind::Mesh { verts: 0, faces: 0 },
                        ObjectData::Empty => RowKind::Empty,
                    },
                );
                kreate_ui::icons::paint_icon(ui, rect.shrink(3.0), tab.icon(data), ink);
                if response.on_hover_text(&name).clicked() {
                    current = *tab;
                    state.set_properties_tab(area, *tab);
                }
            }
        });
        ui.separator();
        ui.vertical(|ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match current {
                    PropertiesTab::Tool => draw_adjust_panel(ui, state, registry, history),
                    PropertiesTab::Object => draw_object_tab(ui, state, doc, registry, l),
                    PropertiesTab::Data => draw_data_tab(ui, state, doc, registry, l),
                    PropertiesTab::Bone => draw_bone_tab(ui, state, doc, registry),
                    PropertiesTab::Material => draw_material_tab(ui, state, doc, registry, l, mode),
                    PropertiesTab::World => draw_world_tab(ui, state, doc, registry, l),
                    PropertiesTab::Modifiers => {
                        ui.heading(&l.modifiers);
                        draw_modifier_stack(ui, state, modifier_rows);
                    }
                    PropertiesTab::Render => draw_render_tab(ui, state, doc, registry, l),
                    PropertiesTab::Physics => draw_physics_tab(ui, state, doc, registry),
                });
        });
    });
}

/// Which tabs this document's active object has.
///
/// A function of the document alone, so the rule can be checked without a
/// window. Every entry is a `poll` in all but name, and it is the same argument
/// operators make: a context that cannot use a thing should not be offered it,
/// or every context grows an empty panel.
pub fn available_tabs(doc: &Document) -> Vec<PropertiesTab> {
    let active = doc.selection.active.and_then(|id| doc.object(id));
    let has_data = matches!(
        active.map(|o| &o.data),
        Some(kreate_scene::document::ObjectData::Light(_))
            | Some(kreate_scene::document::ObjectData::Camera(_))
            | Some(kreate_scene::document::ObjectData::Armature(_))
            | Some(kreate_scene::document::ObjectData::Mesh(_))
    );
    // An armature with an active bone that is really there: the tab shows
    // one bone, and with none it would be an empty panel.
    let has_bone = doc
        .selection
        .active
        .is_some_and(|id| active_bone(doc, id).is_some());
    [
        PropertiesTab::Tool,
        PropertiesTab::Object,
        PropertiesTab::Data,
        PropertiesTab::Bone,
        PropertiesTab::Material,
        PropertiesTab::World,
        PropertiesTab::Modifiers,
        PropertiesTab::Physics,
        PropertiesTab::Render,
    ]
    .into_iter()
    .filter(|tab| match tab {
        PropertiesTab::Object => active.is_some(),
        PropertiesTab::Data => has_data,
        PropertiesTab::Bone => has_bone,
        // Whenever the active object is a *mesh*, not whenever it already has
        // a material. Offering the tab only once a material exists made the
        // whole material system unreachable: nothing else creates one, so a
        // new document's cube had no way to get its first. The tab is where
        // that happens, which is why it has to be there before there is
        // anything in it, the same way the Modifiers tab shows an empty stack.
        PropertiesTab::Material => matches!(
            active.map(|o| &o.data),
            Some(kreate_scene::document::ObjectData::Mesh(_))
        ),
        // The world is the document's, not a selection's — so it is offered
        // whenever there is one, including with nothing selected at all. That
        // is the tab somebody reaches for to turn the background down, and
        // needing to click an object first would be a puzzle.
        PropertiesTab::World => doc.active_world().is_some(),
        // The stack is per object, but an empty one is worth showing: it is
        // where a modifier gets added from.
        PropertiesTab::Modifiers => active.is_some(),
        // The object's physics and the scene's: any object, since the scene's
        // world and bake are there too.
        PropertiesTab::Physics => active.is_some(),
        PropertiesTab::Tool => true,
        // The scene's, whatever is selected.
        PropertiesTab::Render => true,
    })
    .collect()
}

/// The tab to actually draw, given the one that was remembered.
///
/// The remembered one may have just gone away — a light was deselected, say.
/// Falling back rather than drawing nothing, because a panel that empties as
/// the selection changes reads as the editor breaking.
pub fn chosen_tab(remembered: PropertiesTab, available: &[PropertiesTab]) -> PropertiesTab {
    if available.contains(&remembered) {
        remembered
    } else if available.contains(&PropertiesTab::Object) {
        // What there is to show about what is selected, before the last
        // operation, which the viewport's corner already shows.
        PropertiesTab::Object
    } else {
        PropertiesTab::Tool
    }
}

/// The Object tab: where the active object is.
///
/// The same operator the viewport's sidebar drives, and deliberately the same
/// values — two panels that computed a rotation differently would disagree
/// about what the object is doing.
fn draw_object_tab(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    l: &Labels,
) {
    let Some(active) = doc.selection.active.and_then(|id| doc.object(id)) else {
        return;
    };
    // i18n-exempt: an object's name is the user's own text
    ui.label(&active.name);
    ui.separator();
    let Some(info) = registry.get("object.set_transform") else {
        return;
    };
    let t = active.transform;
    let [rx, ry, rz] = kreate_scene::animation::to_euler(t.rotation).to_array();
    let mut params = kreate_reflect::PropertyMap::new();
    params.set(
        "location",
        kreate_reflect::PropertyValue::Vec3(t.translation.to_array()),
    );
    params.set(
        "rotation",
        kreate_reflect::PropertyValue::Vec3([rx, ry, rz]),
    );
    params.set(
        "scale",
        kreate_reflect::PropertyValue::Vec3(t.scale.to_array()),
    );
    // In a panel, the way Blender's Object tab is a stack of them. One today;
    // the shape is what lets the next — relations, visibility — arrive as a
    // title rather than as more rows.
    let object = doc
        .selection
        .active
        .expect("an active object was just read");
    kreate_ui::panel::section(ui, &l.panel_transform, true, |ui| {
        draw_object_transform(ui, state, doc, info, object, params, l);
    });
    draw_robot_panel(ui, state, doc, object);
    draw_physical_panel(ui, state, doc, registry);
    draw_mesh_check_panel(ui, state, doc, object);
    draw_nla_panel(ui, state, doc, registry, object);
    // What the object hangs from and what bends it: Blender's Relations
    // panel, and the Armature modifier's one field.
    let parent = active
        .parent
        .and_then(|p| doc.object(p))
        .map(|p| p.name.clone());
    let skin = active
        .skin
        .and_then(|u| doc.object_by_uid(u))
        .and_then(|id| doc.object(id))
        .map(|o| o.name.clone());
    if parent.is_some() || skin.is_some() {
        let title = state.catalog.get("panel-relations");
        kreate_ui::panel::section(ui, &title, true, |ui| {
            egui::Grid::new("relations").num_columns(2).show(ui, |ui| {
                if let Some(parent) = &parent {
                    ui.label(state.catalog.get("relations-parent"));
                    // i18n-exempt: an object's name is the user's own text
                    ui.label(parent);
                    ui.end_row();
                }
                if let Some(skin) = &skin {
                    ui.label(state.catalog.get("relations-armature"));
                    ui.horizontal(|ui| {
                        // i18n-exempt: an object's name is the user's own text
                        ui.label(skin);
                        if ui
                            .small_button(REMOVE_GLYPH)
                            .on_hover_text(state.catalog.get("unbind-armature"))
                            .clicked()
                        {
                            state
                                .queue
                                .run("object.unbind_armature", PropertyMap::new());
                        }
                    });
                    ui.end_row();
                    let was = active.preserve_volume;
                    let mut on = was;
                    ui.label("");
                    ui.checkbox(&mut on, state.catalog.get("preserve-volume"))
                        .on_hover_text(state.catalog.get("op-object-set-preserve-volume-desc"));
                    if on != was {
                        let mut p = PropertyMap::new();
                        p.set("preserve_volume", kreate_reflect::PropertyValue::Bool(on));
                        state.queue.run("object.set_preserve_volume", p);
                    }
                    ui.end_row();
                }
            });
        });
    }
}

/// The robot the active object belongs to, joint by joint: a slider each,
/// held to the joint's limits — what a robot description's joint state
/// publisher is, in a panel. A joint that mimics another follows it and is
/// shown, not set.
fn draw_robot_panel(ui: &mut Ui, state: &mut ShellState, doc: &Document, active: ObjectId) {
    use kreate_scene::robot::RobotPart;
    // The robot: everything under the top of the active object's tree, if
    // anything there is a joint.
    let mut root = active;
    let mut guard = 0;
    while let Some(p) = doc.object(root).and_then(|o| o.parent) {
        root = p;
        guard += 1;
        if guard > 10_000 {
            break;
        }
    }
    let under = |mut id: ObjectId| -> bool {
        for _ in 0..10_000 {
            if id == root {
                return true;
            }
            match doc.object(id).and_then(|o| o.parent) {
                Some(p) => id = p,
                None => return false,
            }
        }
        false
    };
    let joints: Vec<(
        ObjectId,
        String,
        kreate_scene::robot::Joint,
        kreate_math::DTransform,
    )> = doc
        .objects()
        .filter_map(|(id, o)| match &o.robot {
            Some(RobotPart::Joint(j)) if j.kind.moves() && under(id) => {
                Some((id, o.name.clone(), j.clone(), o.transform))
            }
            _ => None,
        })
        .collect();
    if joints.is_empty() {
        return;
    }
    let rig = doc.robot_rig(active);
    let title = state.catalog.get("panel-robot");
    kreate_ui::panel::section(ui, &title, true, |ui| {
        ui.horizontal(|ui| {
            if ui.button(state.catalog.get("robot-reset")).clicked() {
                state.queue.run("robot.reset_pose", PropertyMap::new());
            }
            match rig.and_then(|r| doc.object(r)) {
                // Posed by its bones now: the sliders and the bones agree,
                // whichever moves.
                Some(r) => {
                    ui.weak(state.catalog.get_args("robot-rigged", &[("rig", &r.name)]));
                }
                None => {
                    if ui
                        .button(state.catalog.get("robot-rig"))
                        .on_hover_text(state.catalog.get("op-robot-rig-desc"))
                        .clicked()
                    {
                        state.queue.run("robot.rig", PropertyMap::new());
                    }
                }
            }
        });
        let dragging = ui.ctx().input(|i| i.pointer.any_down());
        egui::Grid::new("robot-joints")
            .num_columns(2)
            .show(ui, |ui| {
                for (id, name, joint, transform) in &joints {
                    let position = joint.position_of(transform);
                    let label = ui.label(name);
                    if *id == active {
                        label.highlight();
                    }
                    let slides = joint.kind.slides();
                    let (lo, hi) = if joint.kind.limited() && joint.lower < joint.upper {
                        (joint.lower, joint.upper)
                    } else if slides {
                        (-1.0, 1.0)
                    } else {
                        (-std::f64::consts::PI, std::f64::consts::PI)
                    };
                    // Degrees to look at, radians underneath.
                    let to_shown = |v: f64| if slides { v } else { v.to_degrees() };
                    let mut shown = to_shown(position);
                    let slider = egui::Slider::new(&mut shown, to_shown(lo)..=to_shown(hi))
                        .suffix(if slides { " m" } else { "°" })
                        .max_decimals(if slides { 3 } else { 1 });
                    let follows = joint.mimic.as_ref().map(|m| m.joint.clone());
                    let response = ui.add_enabled(follows.is_none(), slider);
                    let response = match &follows {
                        Some(f) => response.on_disabled_hover_text(state.catalog.format(
                            "robot-follows",
                            kreate_i18n::TextKind::Tooltip,
                            &[("joint", f.as_str())],
                        )),
                        None => response,
                    };
                    if response.changed() {
                        let value = if slides { shown } else { shown.to_radians() };
                        let mut p = PropertyMap::new();
                        p.set("joint", kreate_reflect::PropertyValue::Text(name.clone()));
                        p.set("position", kreate_reflect::PropertyValue::Float(value));
                        send_edit(state, "robot.set_joint", p, dragging);
                    }
                    ui.end_row();
                }
            });
    });
}

/// The object's NLA: Push Down for its action, and each strip's place,
/// speed, repeats, influence and ends — Blender's strip sidebar, in a panel.
fn draw_nla_panel(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    object: ObjectId,
) {
    use kreate_reflect::PropertyValue as V;
    let Some(o) = doc.object(object) else { return };
    if o.action.is_none() && o.nla.is_empty() {
        return;
    }
    let title = state.catalog.get("panel-nla");
    kreate_ui::panel::section(ui, &title, false, |ui| {
        if o.action.is_some()
            && ui
                .button(state.catalog.get("op-anim-nla-push-down"))
                .on_hover_text(state.catalog.get("op-anim-nla-push-down-desc"))
                .clicked()
        {
            state.queue.run("anim.nla_push_down", PropertyMap::new());
        }
        let Some(info) = registry.get("anim.nla_strip_set") else {
            return;
        };
        for (index, strip) in o.nla.iter().enumerate().rev() {
            ui.separator();
            ui.horizontal(|ui| {
                // i18n-exempt: a strip's name is its action's, the user's own text
                ui.strong(&strip.name);
                if ui
                    .small_button(REMOVE_GLYPH)
                    .on_hover_text(state.catalog.get("op-anim-nla-strip-remove"))
                    .clicked()
                {
                    let mut p = PropertyMap::new();
                    p.set("index", V::Int(index as i64));
                    state.queue.run("anim.nla_strip_remove", p);
                }
            });
            let mut map = kreate_reflect::OpParams::to_map(
                &kreate_ops::ops::anim::nla_strip_params(index, strip),
            );
            if kreate_ui::property_editor(ui, &state.catalog, info.id, info.schema, &mut map) {
                state.queue.run("anim.nla_strip_set", map);
            }
        }
    });
}

/// The transform of the active object, with Blender's diamonds and the
/// right-click menu on each value: keys, drivers, reset. The Object tab and
/// the sidebar both draw it, so the two cannot come to offer different things.
fn draw_object_transform(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    info: &kreate_ops::OperatorInfo,
    object: ObjectId,
    mut params: PropertyMap,
    l: &Labels,
) {
    let Some(o) = doc.object(object) else { return };
    let text = field_menu_text(&state.catalog);
    let out = kreate_ui::property_editor_full(
        ui,
        &state.catalog,
        info.id,
        info.schema,
        &mut params,
        |_| true,
        |name| object_animated(name).map(|p| key_mark(doc, o.action, p)),
        &l.key_property,
        |name| {
            let property = object_animated(name)?;
            Some(kreate_ui::FieldMenu {
                animatable: true,
                keyed_here: key_mark(doc, o.action, property) == kreate_ui::KeyMark::Keyed,
                driven: (0..3u8)
                    .map(|i| {
                        o.drivers
                            .iter()
                            .any(|d| d.property.as_ref() == property && d.index == i)
                    })
                    .collect(),
                resettable: true,
            })
        },
        &text,
    );
    if out.changed {
        let dragging = ui.ctx().input(|i| i.pointer.any_down());
        send_edit(state, "object.set_transform", params.clone(), dragging);
    }
    if let Some(property) = out.keyed.and_then(object_animated) {
        toggle_key(state, doc, o.action, 2, 0, property);
    }
    if let Some((field, action)) = out.action {
        let uid = doc.object_uid(object).map(|u| u.raw()).unwrap_or(0);
        field_action(state, info, params, field, action, FieldOwner::Object(uid));
    }
}

/// The animatable property an object transform field drives.
fn object_animated(field: &str) -> Option<&'static str> {
    Some(match field {
        "location" => "location",
        "rotation" => "rotation_euler",
        "scale" => "scale",
        _ => return None,
    })
}

/// The words of a value's right-click menu.
fn field_menu_text(catalog: &Catalog) -> kreate_ui::FieldMenuText {
    let t = |key: &str| catalog.get(key);
    kreate_ui::FieldMenuText {
        insert_key: t("field-insert-key"),
        delete_key: t("field-delete-key"),
        add_driver: t("field-add-driver"),
        remove_driver: t("field-remove-driver"),
        reset: t("field-reset"),
        components: ["X".into(), "Y".into(), "Z".into(), "W".into()],
    }
}

/// Whose value a right-click menu acted on.
enum FieldOwner {
    /// The active object, by its uid: what a driver's target names.
    Object(u64),
    /// A material slot of the active object.
    Material(u32),
    /// The active object's light.
    Light,
}

/// Does what a value's right-click menu chose.
fn field_action(
    state: &mut ShellState,
    info: &kreate_ops::OperatorInfo,
    mut params: PropertyMap,
    field: &'static str,
    action: kreate_ui::FieldAction,
    owner: FieldOwner,
) {
    use kreate_reflect::PropertyValue as V;
    use kreate_ui::FieldAction as A;
    let (datablock, slot, animated) = match owner {
        FieldOwner::Object(_) => (2u32, 0u32, object_animated(field)),
        FieldOwner::Material(slot) => (0, slot, material_animated(field)),
        FieldOwner::Light => (1, 0, light_animated(field)),
    };
    match action {
        A::InsertKey | A::DeleteKey => {
            let Some(property) = animated else { return };
            let mut p = PropertyMap::new();
            p.set("datablock", V::Enum(datablock));
            p.set("slot", V::Int(slot as i64));
            p.set("property", V::Text(property.to_string()));
            let op = if action == A::InsertKey {
                "anim.keyframe_insert_property"
            } else {
                "anim.keyframe_delete_property"
            };
            state.queue.dispatch(op, p);
        }
        A::AddDriver(i) | A::RemoveDriver(i) => {
            let (FieldOwner::Object(uid), Some(property)) = (owner, animated) else {
                return;
            };
            let mut p = PropertyMap::new();
            p.set(
                "target",
                V::Text(kreate_ops::ops::driver::driver_target(uid, property, i)),
            );
            let op = if matches!(action, A::AddDriver(_)) {
                "anim.driver_add"
            } else {
                "anim.driver_remove"
            };
            state.queue.run(op, p);
        }
        A::Reset => {
            let Some(default) = info.defaults().get(field).cloned() else {
                return;
            };
            params.set(field, default);
            state.queue.run(info.id, params);
        }
    }
}

/// The camera half of the Data tab.
fn draw_camera_data(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    l: &Labels,
    id: kreate_scene::document::CameraId,
) {
    let Some(camera) = doc.camera(id) else {
        return;
    };
    let Some(info) = registry.get("camera.set_properties") else {
        return;
    };
    ui.heading(&l.tab_data_camera);
    ui.separator();
    let mut params =
        kreate_reflect::OpParams::to_map(&kreate_ops::ops::object::camera_params(camera));
    if kreate_ui::property_editor::property_editor(
        ui,
        &state.catalog,
        info.id,
        info.schema,
        &mut params,
    ) {
        let dragging = ui.ctx().input(|i| i.pointer.any_down());
        send_edit(state, "camera.set_properties", params, dragging);
    }
}

/// The Material tab: what the active object is made of.
///
/// The first slot, because that is the one the raster pass draws. A panel over
/// the others would be editing materials nothing shows — see §7.5 on why the
/// per-face slots wait for the draws to be split.
/// The cross on a slot that holds something. A glyph rather than the word
/// "Clear", because it sits beside a name that needs the width.
// i18n-exempt: a symbol, not a word
const CLEAR_GLYPH: &str = "×";

/// The plus and minus on a list somebody adds rows to.
///
/// Both are in the bundled subset, which is not a given: see
/// [`kreate_ui::fonts::missing_glyphs`], and the test below that holds every
/// glyph the chrome draws to it.
// i18n-exempt: symbols, not words
const ADD_GLYPH: &str = "+";
// i18n-exempt: a symbol, not a word. An en dash rather than the minus sign
// U+2212 this wants: the bundled subset has no glyph for that one, and the
// test below caught it before it reached a screen.
const REMOVE_GLYPH: &str = "–";

fn draw_material_tab(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    l: &Labels,
    mode: kreate_ops::operator::Mode,
) {
    let Some(object) = doc.selection.active else {
        return;
    };
    ui.heading(&l.tab_material);

    // **The slot list.** A face names a slot and a slot names a material, so
    // this is where a model says how many parts it has. It could not be here
    // until the rendered pass drew one call per slot: before that a second
    // slot was a material nobody could see, and a list of them would have been
    // a control that did nothing.
    let slots = doc
        .object(object)
        .and_then(|o| match &o.data {
            kreate_scene::document::ObjectData::Mesh(mesh) => Some(doc.mesh_slots(*mesh)),
            _ => None,
        })
        .unwrap_or(&[]);
    // Clamped here rather than kept in step: what shortens the list is a
    // document edit, and an undo can lengthen it again.
    let slot = state
        .material_slot
        .min(slots.len().saturating_sub(1) as u32);
    state.material_slot = slot;

    ui.horizontal_top(|ui| {
        // In a frame and full width, so it reads as a list rather than as a
        // column of chips: the row is the thing being selected, and a
        // highlight that stops at the end of the word does not say so.
        let buttons = 34.0;
        let width = (ui.available_width() - buttons).max(80.0);
        egui::Frame::group(ui.style())
            .inner_margin(egui::Margin::symmetric(2, 2))
            .show(ui, |ui| {
                ui.set_width(width);
                // Explicitly vertical: this frame is inside a horizontal
                // layout, so it inherits one and the rows came out side by
                // side and running off the panel.
                ui.vertical(|ui| {
                    for (at, held) in slots.iter().enumerate() {
                        let name = match held.and_then(|id| doc.material(id)) {
                            // i18n-exempt: a material's name is the user's own text
                            Some(material) => material.name.clone(),
                            None => l.material_empty_slot.clone(),
                        };
                        // `left_text` with an empty body, which is how a
                        // button's label sits against the left edge: a list
                        // row's text starts where the row starts, and centred
                        // names in a column read as a stack of buttons rather
                        // than as a list.
                        let row = egui::Button::selectable(at as u32 == slot, "").left_text(name);
                        let height = ui.spacing().interact_size.y;
                        if ui.add_sized([ui.available_width(), height], row).clicked() {
                            state.material_slot = at as u32;
                        }
                    }
                    // An empty frame collapses to nothing, which reads as a
                    // missing control rather than as an empty list.
                    if slots.is_empty() {
                        ui.weak(&l.material_no_slots);
                    }
                });
            });
        // Blender's plus and minus, and the same division of labour: the plus
        // adds an *empty* slot and the New button below fills it. Two steps,
        // because they are two decisions — how many parts this model has, and
        // what each one is made of.
        ui.vertical(|ui| {
            if ui
                .button(ADD_GLYPH)
                .on_hover_text(&l.material_add_slot)
                .clicked()
            {
                state.queue.run("material.add_slot", PropertyMap::new());
            }
            if !slots.is_empty()
                && ui
                    .button(REMOVE_GLYPH)
                    .on_hover_text(&l.material_remove_slot)
                    .clicked()
            {
                let mut p = PropertyMap::new();
                p.set("slot", kreate_reflect::PropertyValue::Int(slot as i64));
                state.queue.run("material.remove_slot", p);
            }
        });
    });

    let slot_param = |extra: Option<(&'static str, kreate_reflect::PropertyValue)>| {
        let mut p = PropertyMap::new();
        p.set("slot", kreate_reflect::PropertyValue::Int(slot as i64));
        if let Some((k, v)) = extra {
            p.set(k, v);
        }
        p
    };
    // Blender's browse button: put a material that already exists in this
    // slot, so parts and models can share one.
    if !slots.is_empty() && doc.materials().next().is_some() {
        let current = slots
            .get(slot as usize)
            .copied()
            .flatten()
            .and_then(|id| doc.material(id))
            .map(|m| m.name.clone())
            .unwrap_or_else(|| l.material_empty_slot.clone());
        ui.horizontal(|ui| {
            ui.label(&l.material_browse);
            egui::ComboBox::from_id_salt("material-browse")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for (_, material) in doc.materials() {
                        // i18n-exempt: a material's name is the user's own text
                        if ui.selectable_label(false, material.name.as_str()).clicked() {
                            let uid =
                                kreate_reflect::PropertyValue::Text(material.uid.raw().to_string());
                            state
                                .queue
                                .run("material.link", slot_param(Some(("uid", uid))));
                        }
                    }
                });
        });
    }
    // Blender's Assign and Select, in edit mode: which faces are this slot.
    if mode.edits_elements() && !slots.is_empty() {
        ui.horizontal(|ui| {
            if ui
                .button(&l.material_assign)
                .on_hover_text(&l.material_assign_hint)
                .clicked()
            {
                state.queue.run("material.assign", slot_param(None));
            }
            if ui
                .button(&l.material_select)
                .on_hover_text(&l.material_select_hint)
                .clicked()
            {
                state.queue.run("material.select", slot_param(None));
            }
        });
    }
    ui.separator();

    // An empty slot, or no slots at all. Both want the same button, because
    // both are "there is nothing here to edit yet" — and a panel that drew
    // nothing was the reason every material control was once unreachable.
    let Some(material) = doc.material_at(object, slot as usize) else {
        ui.add_space(4.0);
        ui.weak(&l.material_none);
        ui.add_space(4.0);
        if ui.button(&l.material_new).clicked() {
            state.queue.push(Command::NewMaterial);
        }
        return;
    };
    let Some(info) = registry.get("material.set_properties") else {
        return;
    };

    // Blender's "Use Nodes". On, the graph is the material and the Shader
    // Editor is where it is edited; the flat controls below would be editing
    // fields the picture no longer reads, so they are not offered.
    let uses_nodes = material.graph.is_some();
    if ui
        .selectable_label(uses_nodes, &l.node_use_nodes)
        .on_hover_text(&l.node_use_nodes_hint)
        .clicked()
    {
        let mut p = PropertyMap::new();
        p.set("slot", kreate_reflect::PropertyValue::Int(slot as i64));
        p.set("enabled", kreate_reflect::PropertyValue::Bool(!uses_nodes));
        state.queue.run("material.use_nodes", p);
    }
    if uses_nodes {
        ui.add_space(4.0);
        ui.weak(&l.material_uses_nodes);
        return;
    }

    // The base colour texture. The image node, the compiler's layer
    // assignment, the texture array and the surface coordinates all stand, and
    // until there is a control here none of it is a feature anybody has.
    //
    // The slot says what is in it. A button alone could not be read: with a
    // texture assigned and a white colour the panel looked exactly like a
    // panel with no texture, so the one question somebody opens this tab to
    // answer had no answer on screen.
    kreate_ui::panel::section(ui, &l.panel_surface, true, |ui| {
        for (index, map) in kreate_scene::MaterialMap::ALL.into_iter().enumerate() {
            let label = match map {
                kreate_scene::MaterialMap::BaseColor => &l.material_base_colour,
                kreate_scene::MaterialMap::MetallicRoughness => &l.material_metallic_roughness,
                kreate_scene::MaterialMap::Normal => &l.material_normal_map,
                kreate_scene::MaterialMap::Emission => &l.material_emission_map,
                kreate_scene::MaterialMap::Occlusion => &l.material_occlusion_map,
            };
            let map_id = index as u32;
            ui.label(label);
            ui.horizontal(|ui| {
                let assigned = material
                    .map(map)
                    .and_then(|id| doc.image(id))
                    .map(|image| image.name.clone());
                match &assigned {
                    // i18n-exempt: an image's name is the user's own text
                    Some(name) => {
                        if ui
                            .button(name)
                            .on_hover_text(&l.material_pick_image)
                            .clicked()
                        {
                            state.queue.push(Command::PickMaterialImage { map: map_id });
                        }
                        if ui
                            .button(CLEAR_GLYPH)
                            .on_hover_text(&l.material_clear_image)
                            .clicked()
                        {
                            state
                                .queue
                                .push(Command::ClearMaterialImage { map: map_id });
                        }
                    }
                    None => {
                        if ui.button(&l.material_pick_image).clicked() {
                            state.queue.push(Command::PickMaterialImage { map: map_id });
                        }
                    }
                }
            });
        }
        ui.separator();
        let mut params = kreate_reflect::OpParams::to_map(
            &kreate_ops::ops::object::material_params(material, slot),
        );
        let text = field_menu_text(&state.catalog);
        let out = kreate_ui::property_editor_full(
            ui,
            &state.catalog,
            info.id,
            info.schema,
            &mut params,
            // The slot is *which* material is being edited, not a property of it.
            // Drawn as a spinner it would be a second, worse copy of the list
            // above, and one that could point at a slot the list does not show.
            // The normal strength only once there is a normal map for it to
            // scale: a dial that does nothing is the dead control this panel was
            // cleaned of once already.
            |name| name != "slot" && (name != "normal_strength" || material.normal_image.is_some()),
            |name| material_animated(name).map(|p| key_mark(doc, material.action, p)),
            &l.key_property,
            |name| {
                Some(kreate_ui::FieldMenu {
                    animatable: material_animated(name).is_some(),
                    keyed_here: material_animated(name).is_some_and(|p| {
                        key_mark(doc, material.action, p) == kreate_ui::KeyMark::Keyed
                    }),
                    driven: Vec::new(),
                    resettable: true,
                })
            },
            &text,
        );
        if out.changed {
            let dragging = ui.ctx().input(|i| i.pointer.any_down());
            send_edit(state, "material.set_properties", params.clone(), dragging);
        }
        if let Some(property) = out.keyed.and_then(material_animated) {
            toggle_key(state, doc, material.action, 0, slot, property);
        }
        if let Some((field, action)) = out.action {
            field_action(
                state,
                info,
                params,
                field,
                action,
                FieldOwner::Material(slot),
            );
        }
    });
}

/// The World tab: the light arriving from no object in particular.
///
/// The document's, not a selection's, which is why it is a tab of its own
/// rather than an arm of the Data tab: there is no object to click on, and
/// making somebody select one first to turn the background down would be a
/// puzzle rather than an interface.
fn draw_world_tab(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    l: &Labels,
) {
    let Some(id) = doc.active_world() else {
        return;
    };
    let Some(world) = doc.world(id) else {
        return;
    };
    let Some(info) = registry.get("world.set_properties") else {
        return;
    };
    ui.heading(&l.tab_world);
    // The control §7.7 recorded as missing. An environment that can be loaded,
    // saved and shaded and cannot be *chosen* is a feature the user does not
    // have — which is the gate that section exists to state.
    ui.horizontal(|ui| {
        if ui.button(&l.world_pick_image).clicked() {
            state.queue.push(Command::PickWorldImage);
        }
        if matches!(world.background, kreate_scene::Background::Image(_))
            && ui.button(&l.world_clear_image).clicked()
        {
            state.queue.push(Command::ClearWorldImage);
        }
    });
    if matches!(world.background, kreate_scene::Background::Image(_)) {
        // The colour below does nothing for an image background, and a control
        // that does nothing is worse than one that is missing. Said rather than
        // hidden, because the strength and the rotation *do* apply and the tab
        // is still the place for them.
        ui.weak(&l.world_image);
    }
    ui.separator();
    let mut params =
        kreate_reflect::OpParams::to_map(&kreate_ops::ops::object::world_params(world));
    kreate_ui::panel::section(ui, &l.panel_surface, true, |ui| {
        if kreate_ui::property_editor::property_editor(
            ui,
            &state.catalog,
            info.id,
            info.schema,
            &mut params,
        ) {
            let dragging = ui.ctx().input(|i| i.pointer.any_down());
            send_edit(state, "world.set_properties", params, dragging);
        }
    });
}

/// The Data tab: what the active object *is*.
///
/// One tab and one operator per kind of datablock, and today there is one kind
/// with a panel. The shape is the point: a light's fields are not written out
/// here, they are the operator's schema drawn by the same reflection widget
/// that draws every other operator — which is §2.6's rule and the reason the
/// material and world tabs will be a `match` arm each rather than a week each.
fn draw_data_tab(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    l: &Labels,
) {
    let Some(active) = doc.selection.active.and_then(|id| doc.object(id)) else {
        return;
    };
    // One arm per kind of datablock, which is what the panel being generated
    // rather than written buys: each is a heading, a lookup, and the operator's
    // own schema.
    match active.data {
        kreate_scene::document::ObjectData::Light(id) => {
            draw_light_data(ui, state, doc, registry, l, id)
        }
        kreate_scene::document::ObjectData::Camera(id) => {
            draw_camera_data(ui, state, doc, registry, l, id)
        }
        kreate_scene::document::ObjectData::Armature(id) => {
            draw_armature_data(ui, state, doc, registry, id)
        }
        kreate_scene::document::ObjectData::Mesh(id) => {
            draw_sketch_panel(ui, state, active);
            draw_vertex_groups(ui, state, doc, id, active);
            draw_shape_keys(ui, state, active);
        }
        _ => {}
    }
}

/// The sketch: its constraints with their dimensions, buttons that add one
/// on what is selected in edit mode, and the solid it becomes.
fn draw_sketch_panel(ui: &mut Ui, state: &mut ShellState, object: &kreate_scene::document::Object) {
    use kreate_reflect::PropertyValue as V;
    let title = state.catalog.get("panel-sketch");
    let open = !object.sketch.is_empty();
    kreate_ui::panel::section(ui, &title, open, |ui| {
        ui.weak(state.catalog.get("sketch-hint"));
        ui.horizontal_wrapped(|ui| {
            for (kind, tag) in kreate_scene::sketch::KINDS.iter().enumerate() {
                let label = state.catalog.get(&format!("sketch-kind-{tag}"));
                if ui.small_button(label).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("kind", V::Enum(kind as u32));
                    state.queue.run("sketch.constraint_add", p);
                }
            }
        });
        let length = kreate_ui::property_editor::UnitDisplay::of(kreate_reflect::Unit::Length);
        for (i, c) in object.sketch.iter().enumerate() {
            ui.horizontal(|ui| {
                let name = state.catalog.get(&format!("sketch-kind-{}", c.tag()));
                // i18n-exempt: a number and a translated kind
                ui.label(format!("{}. {name}", i + 1));
                match c {
                    kreate_scene::sketch::Constraint::Distance { value, .. } => {
                        let mut shown = value * length.to_display;
                        let response = ui.add(
                            egui::DragValue::new(&mut shown)
                                .speed(0.01 * length.to_display * 0.01)
                                .suffix(length.suffix)
                                .max_decimals(length.decimals),
                        );
                        if response.changed() {
                            let mut p = PropertyMap::new();
                            p.set("index", V::Int(i as i64));
                            p.set("value", V::Float(length.from_display(shown)));
                            let dragging = ui.ctx().input(|i| i.pointer.any_down());
                            send_edit(state, "sketch.constraint_set", p, dragging);
                        }
                    }
                    kreate_scene::sketch::Constraint::Angle { value, .. } => {
                        let mut degrees = value.to_degrees();
                        let response =
                            ui.add(egui::DragValue::new(&mut degrees).speed(0.5).suffix("°"));
                        if response.changed() {
                            let mut p = PropertyMap::new();
                            p.set("index", V::Int(i as i64));
                            p.set("value", V::Float(degrees.to_radians()));
                            let dragging = ui.ctx().input(|i| i.pointer.any_down());
                            send_edit(state, "sketch.constraint_set", p, dragging);
                        }
                    }
                    _ => {}
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button(REMOVE_GLYPH).clicked() {
                        let mut p = PropertyMap::new();
                        p.set("index", V::Int(i as i64));
                        state.queue.run("sketch.constraint_remove", p);
                    }
                });
            });
        }
        ui.horizontal(|ui| {
            if !object.sketch.is_empty() && ui.button(state.catalog.get("sketch-solve")).clicked() {
                state.queue.run("sketch.solve", PropertyMap::new());
            }
            if ui.button(state.catalog.get("sketch-extrude")).clicked() {
                let mut p = PropertyMap::new();
                p.set("kind", V::Enum(0));
                // Ten of the scene's units: a part's thickness to start from.
                p.set("height", V::Float(10.0 / length.to_display));
                state.queue.run("sketch.make_solid", p);
            }
            if ui.button(state.catalog.get("sketch-revolve")).clicked() {
                let mut p = PropertyMap::new();
                p.set("kind", V::Enum(1));
                state.queue.run("sketch.make_solid", p);
            }
            // A sketch made on a part's face: back onto the face after the
            // part changed.
            if object.parent.is_some()
                && ui
                    .button(state.catalog.get("sketch-reattach"))
                    .on_hover_text(state.catalog.get("op-sketch-reattach-desc"))
                    .clicked()
            {
                state.queue.run("sketch.reattach", PropertyMap::new());
            }
        });
    });
}

/// The mesh's vertex groups: Blender's list in the Mesh Data tab. Click one to
/// paint and show it; add and remove below.
fn draw_vertex_groups(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    mesh: kreate_scene::MeshId,
    object: &kreate_scene::document::Object,
) {
    use kreate_reflect::PropertyValue as V;
    let title = state.catalog.get("panel-vertex-groups");
    kreate_ui::panel::section(ui, &title, true, |ui| {
        let groups = kreate_ops::ops::paint::groups(doc, mesh);
        if groups.is_empty() {
            ui.weak(state.catalog.get("groups-none"));
        }
        for g in &groups {
            let active = object.active_group.as_deref() == Some(g.as_str());
            if ui.selectable_label(active, g).clicked() && !active {
                let mut p = PropertyMap::new();
                p.set("name", V::Text(g.clone()));
                state.queue.run("object.vertex_group_set_active", p);
            }
        }
        ui.horizontal(|ui| {
            if ui
                .small_button(ADD_GLYPH)
                .on_hover_text(state.catalog.get("op-object-vertex-group-add"))
                .clicked()
            {
                state
                    .queue
                    .run("object.vertex_group_add", PropertyMap::new());
            }
            let can_remove = object.active_group.is_some();
            if ui
                .add_enabled(can_remove, egui::Button::new(REMOVE_GLYPH).small())
                .on_hover_text(state.catalog.get("op-object-vertex-group-remove"))
                .clicked()
            {
                if let Some(g) = &object.active_group {
                    let mut p = PropertyMap::new();
                    p.set("name", V::Text(g.clone()));
                    state.queue.run("object.vertex_group_remove", p);
                }
            }
        });
    });
}

/// The object's shape keys: Blender's list in the Mesh Data tab — the basis,
/// then each key with its value; click one to make it the one edit mode
/// edits.
fn draw_shape_keys(ui: &mut Ui, state: &mut ShellState, object: &kreate_scene::document::Object) {
    use kreate_reflect::PropertyValue as V;
    let title = state.catalog.get("panel-shape-keys");
    kreate_ui::panel::section(ui, &title, true, |ui| {
        let active = object.active_shape.clone();
        if object.editing_shape.is_some() {
            ui.weak(state.catalog.get("shape-keys-editing"));
        }
        if !object.shape_keys.is_empty()
            && ui
                .selectable_label(active.is_none(), state.catalog.get("shape-key-basis"))
                .clicked()
        {
            let mut p = PropertyMap::new();
            p.set("name", V::Text(String::new()));
            state.queue.run("object.shape_key_set_active", p);
        }
        let pointer_down = ui.input(|i| i.pointer.any_down());
        for key in &object.shape_keys {
            ui.horizontal(|ui| {
                let is_active = active.as_deref() == Some(key.name.as_str());
                if ui.selectable_label(is_active, &key.name).clicked() && !is_active {
                    let mut p = PropertyMap::new();
                    p.set("name", V::Text(key.name.clone()));
                    state.queue.run("object.shape_key_set_active", p);
                }
                // Blender's diamond: a key of this value at the current frame.
                // i18n-exempt: a geometric glyph standing in for an icon
                if ui
                    .small_button("\u{25C6}")
                    .on_hover_text(state.catalog.get("op-anim-keyframe-insert-property"))
                    .clicked()
                {
                    let mut p = PropertyMap::new();
                    p.set("datablock", V::Enum(2));
                    p.set(
                        "property",
                        V::Text(kreate_scene::animation::shape_path(&key.name)),
                    );
                    // Through the keys' door, which fills in the current frame.
                    state.queue.dispatch("anim.keyframe_insert_property", p);
                }
                let mut value = key.value;
                let response = ui.add(egui::Slider::new(&mut value, 0.0..=1.0).max_decimals(3));
                if response.changed() {
                    let mut p = PropertyMap::new();
                    p.set("name", V::Text(key.name.clone()));
                    p.set("value", V::Float(value as f64));
                    send_edit(state, "object.shape_key_set", p, pointer_down);
                }
            });
        }
        ui.horizontal(|ui| {
            if ui
                .small_button(ADD_GLYPH)
                .on_hover_text(state.catalog.get("op-object-shape-key-add"))
                .clicked()
            {
                state.queue.run("object.shape_key_add", PropertyMap::new());
            }
            if ui
                .add_enabled(active.is_some(), egui::Button::new(REMOVE_GLYPH).small())
                .on_hover_text(state.catalog.get("op-object-shape-key-remove"))
                .clicked()
            {
                if let Some(name) = &active {
                    let mut p = PropertyMap::new();
                    p.set("name", V::Text(name.clone()));
                    state.queue.run("object.shape_key_remove", p);
                }
            }
            if ui
                .small_button(state.catalog.get("shape-key-from-mix"))
                .on_hover_text(state.catalog.get("op-object-shape-key-add-desc"))
                .clicked()
            {
                let mut p = PropertyMap::new();
                p.set("from_mix", V::Bool(true));
                state.queue.run("object.shape_key_add", p);
            }
        });
    });
}

/// The active bone of an armature object: its index and name.
fn active_bone(doc: &Document, object: ObjectId) -> Option<(usize, String)> {
    let (_, armature) = doc.armature_of(object)?;
    let name = doc.object(object)?.pose.active.clone()?;
    armature.bone_index(&name).map(|i| (i, name))
}

/// The Armature tab: how the skeleton is drawn.
fn draw_armature_data(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    id: kreate_scene::ArmatureId,
) {
    let (Some(armature), Some(info)) = (doc.armature(id), registry.get("armature.set_display"))
    else {
        return;
    };
    // i18n-exempt: the datablock's name is the user's own text
    ui.heading(&armature.name);
    ui.separator();
    let mut params =
        kreate_reflect::OpParams::to_map(&kreate_ops::ops::armature::display_params(armature));
    let title = state.catalog.get("panel-viewport-display");
    kreate_ui::panel::section(ui, &title, true, |ui| {
        if kreate_ui::property_editor::property_editor(
            ui,
            &state.catalog,
            info.id,
            info.schema,
            &mut params,
        ) {
            send_edit(state, "armature.set_display", params, false);
        }
    });
}

/// The Bone tab: the active bone's pose, then the bone as built.
///
/// Both at once rather than one per mode: the pose is what an animator
/// changes, the rest is what a rigger changes, and seeing the one while
/// changing the other is how a person finds out which they meant.
fn draw_bone_tab(ui: &mut Ui, state: &mut ShellState, doc: &Document, registry: &OperatorRegistry) {
    let Some(object) = doc.selection.active else {
        return;
    };
    let Some((index, name)) = active_bone(doc, object) else {
        return;
    };
    // i18n-exempt: the bone's name is the user's own text
    ui.heading(&name);
    ui.separator();
    let title = state.catalog.get("panel-pose");
    kreate_ui::panel::section(ui, &title, true, |ui| {
        draw_pose_bone_fields(ui, state, doc, registry, object, &name);
    });
    let title = state.catalog.get("panel-bone-rest");
    kreate_ui::panel::section(ui, &title, true, |ui| {
        draw_rest_bone_fields(ui, state, doc, registry, object, index);
    });
    let title = state.catalog.get("panel-bone-ik");
    kreate_ui::panel::section(ui, &title, true, |ui| {
        draw_ik_fields(ui, state, doc, object, &name);
    });
    let title = state.catalog.get("panel-bone-constraints");
    kreate_ui::panel::section(ui, &title, true, |ui| {
        draw_constraint_fields(ui, state, doc, registry, object, &name);
    });
}

/// The bone's constraints, each drawn from the set operator's schema, and a
/// menu to add one.
fn draw_constraint_fields(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    object: ObjectId,
    bone: &str,
) {
    use kreate_reflect::PropertyValue as V;
    let Some(o) = doc.object(object) else { return };
    let Some((_, armature)) = doc.armature_of(object) else {
        return;
    };
    let kinds = [
        "constraint-copy-location",
        "constraint-copy-rotation",
        "constraint-copy-scale",
        "constraint-damped-track",
        "constraint-limit-rotation",
        "constraint-limit-location",
    ];
    ui.menu_button(state.catalog.get("op-pose-constraint-add"), |ui| {
        for (i, key) in kinds.iter().enumerate() {
            if ui.button(state.catalog.get(key)).clicked() {
                let mut p = PropertyMap::new();
                p.set("kind", V::Int(i as i64));
                p.set("bone", V::Text(bone.to_string()));
                state.queue.run("pose.constraint_add", p);
                ui.close();
            }
        }
    });
    let Some(info) = registry.get("pose.constraint_set") else {
        return;
    };
    for (index, c) in o.pose.constraints.iter().enumerate() {
        if c.bone != bone {
            continue;
        }
        ui.separator();
        let params = kreate_ops::ops::pose::constraint_params(index, c);
        let mut enabled = c.enabled;
        let mut target = c.target.clone();
        ui.horizontal(|ui| {
            ui.checkbox(&mut enabled, "");
            ui.strong(state.catalog.get(kinds[c.kind.index() as usize]));
            if ui
                .small_button("\u{2715}")
                .on_hover_text(state.catalog.get("op-pose-constraint-remove"))
                .clicked()
            {
                let mut p = PropertyMap::new();
                p.set("index", V::Int(index as i64));
                state.queue.run("pose.constraint_remove", p);
            }
        });
        if c.kind.has_target() {
            ui.horizontal(|ui| {
                ui.label(state.catalog.get("ik-target"));
                egui::ComboBox::from_id_salt(("constraint-target", index))
                    .selected_text(if target.is_empty() {
                        "\u{2014}".to_string()
                    } else {
                        target.clone()
                    })
                    .show_ui(ui, |ui| {
                        for b in armature.bones.iter().filter(|b| b.name != bone) {
                            ui.selectable_value(&mut target, b.name.clone(), b.name.as_str());
                        }
                    });
            });
        }
        let mut map = kreate_reflect::OpParams::to_map(&params);
        let shown = |field: &str| {
            matches!(field, "influence") || (!c.kind.has_target() && matches!(field, "min" | "max"))
        };
        let changed = kreate_ui::property_editor::property_editor_where(
            ui,
            &state.catalog,
            info.id,
            info.schema,
            &mut map,
            shown,
        );
        if changed || enabled != c.enabled || target != c.target {
            map.set("enabled", V::Bool(enabled));
            map.set("target", V::Text(target));
            state.queue.run("pose.constraint_set", map);
        }
    }
}

/// The bone's IK constraint: which bone it reaches for, how many bones turn,
/// whether it is on — or a button to give it one.
fn draw_ik_fields(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    object: ObjectId,
    bone: &str,
) {
    use kreate_reflect::PropertyValue as V;
    let Some(o) = doc.object(object) else { return };
    let Some((_, armature)) = doc.armature_of(object) else {
        return;
    };
    let send =
        |state: &mut ShellState, op: &str, target: &str, chain: u32, enabled: bool, pole: &str| {
            let mut p = PropertyMap::new();
            p.set("bone", V::Text(bone.to_string()));
            p.set("target", V::Text(target.to_string()));
            p.set("chain", V::Int(chain as i64));
            p.set("enabled", V::Bool(enabled));
            p.set("pole", V::Text(pole.to_string()));
            state.queue.run(op, p);
        };
    let Some(ik) = o.pose.ik.iter().find(|c| c.bone == bone).cloned() else {
        if ui.button(state.catalog.get("op-pose-ik-add")).clicked() {
            send(state, "pose.ik_add", "", 2, true, "");
        }
        return;
    };
    let (mut target, mut chain, mut enabled, mut pole) =
        (ik.target.clone(), ik.chain, ik.enabled, ik.pole.clone());
    ui.horizontal(|ui| {
        ui.checkbox(&mut enabled, state.catalog.get("ik-enabled"));
        if ui
            .small_button("\u{2715}")
            .on_hover_text(state.catalog.get("op-pose-ik-remove"))
            .clicked()
        {
            send(state, "pose.ik_remove", "", 0, true, "");
        }
    });
    ui.horizontal(|ui| {
        ui.label(state.catalog.get("ik-target"));
        egui::ComboBox::from_id_salt(("ik-target", bone))
            .selected_text(if target.is_empty() {
                "\u{2014}".to_string()
            } else {
                target.clone()
            })
            .show_ui(ui, |ui| {
                for b in armature.bones.iter().filter(|b| b.name != bone) {
                    ui.selectable_value(&mut target, b.name.clone(), b.name.as_str());
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label(state.catalog.get("ik-pole"));
        egui::ComboBox::from_id_salt(("ik-pole", bone))
            .selected_text(if pole.is_empty() {
                "\u{2014}".to_string()
            } else {
                pole.clone()
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut pole, String::new(), "\u{2014}");
                for b in armature.bones.iter().filter(|b| b.name != bone) {
                    ui.selectable_value(&mut pole, b.name.clone(), b.name.as_str());
                }
            })
            .response
            .on_hover_text(state.catalog.get("ik-pole-desc"));
    });
    ui.horizontal(|ui| {
        ui.label(state.catalog.get("ik-chain"));
        ui.add(egui::DragValue::new(&mut chain).range(0..=64))
            .on_hover_text(state.catalog.get("ik-chain-desc"));
    });
    if (target.as_str(), chain, enabled, pole.as_str())
        != (ik.target.as_str(), ik.chain, ik.enabled, ik.pole.as_str())
    {
        send(state, "pose.ik_set", &target, chain, enabled, &pole);
    }
}

/// The pose property a Bone tab field drives, as its curve names it.
fn pose_animated(field: &str) -> Option<&'static str> {
    Some(match field {
        "location" => "location",
        "rotation" => "rotation_quaternion",
        "scale" => "scale",
        _ => return None,
    })
}

/// A bone's pose, with diamonds and the right-click menu: the Bone tab and
/// the sidebar in pose mode.
fn draw_pose_bone_fields(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    object: ObjectId,
    bone: &str,
) {
    let (Some(o), Some(info)) = (doc.object(object), registry.get("pose.set_bone")) else {
        return;
    };
    let mut params =
        kreate_reflect::OpParams::to_map(&kreate_ops::ops::pose::pose_bone_params(&o.pose, bone));
    let path =
        |field: &str| pose_animated(field).map(|p| kreate_scene::animation::bone_path(bone, p));
    let text = field_menu_text(&state.catalog);
    let tip = state.catalog.get("key-property");
    let out = kreate_ui::property_editor_full(
        ui,
        &state.catalog,
        info.id,
        info.schema,
        &mut params,
        |_| true,
        |field| path(field).map(|p| key_mark(doc, o.action, &p)),
        &tip,
        |field| {
            let p = path(field)?;
            Some(kreate_ui::FieldMenu {
                animatable: true,
                keyed_here: key_mark(doc, o.action, &p) == kreate_ui::KeyMark::Keyed,
                driven: Vec::new(),
                resettable: true,
            })
        },
        &text,
    );
    if out.changed {
        let dragging = ui.ctx().input(|i| i.pointer.any_down());
        send_edit(state, "pose.set_bone", params.clone(), dragging);
    }
    if let Some(p) = out.keyed.and_then(path) {
        toggle_key(state, doc, o.action, 2, 0, &p);
    }
    if let Some((field, action)) = out.action {
        match (action, path(field)) {
            (kreate_ui::FieldAction::InsertKey | kreate_ui::FieldAction::DeleteKey, Some(p)) => {
                use kreate_reflect::PropertyValue as V;
                let mut k = PropertyMap::new();
                k.set("datablock", V::Enum(2));
                k.set("property", V::Text(p));
                let op = if action == kreate_ui::FieldAction::InsertKey {
                    "anim.keyframe_insert_property"
                } else {
                    "anim.keyframe_delete_property"
                };
                state.queue.dispatch(op, k);
            }
            (kreate_ui::FieldAction::Reset, _) => {
                if let Some(default) = info.defaults().get(field).cloned() {
                    params.set(field, default);
                    state.queue.run(info.id, params);
                }
            }
            _ => {}
        }
    }
}

/// A bone as built, with its parent picked from the armature's other bones:
/// the Bone tab and the sidebar in the skeleton's edit mode.
fn draw_rest_bone_fields(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    object: ObjectId,
    index: usize,
) {
    let (Some((_, armature)), Some(info)) =
        (doc.armature_of(object), registry.get("armature.bone_set"))
    else {
        return;
    };
    let Some(bone) = kreate_ops::ops::armature::bone_params(armature, index) else {
        return;
    };
    let mut params = kreate_reflect::OpParams::to_map(&bone);
    let mut changed = kreate_ui::property_editor::property_editor_where(
        ui,
        &state.catalog,
        info.id,
        info.schema,
        &mut params,
        |name| name != "parent",
    );
    // The parent from a list, as Blender's is: every bone that would not make
    // a loop, and none.
    let none = state.catalog.get("drivers-none");
    let current = if bone.parent.is_empty() {
        none.clone()
    } else {
        bone.parent.clone()
    };
    ui.horizontal(|ui| {
        ui.label(state.catalog.get("bone-parent"));
        egui::ComboBox::from_id_salt(("bone-parent", index))
            .selected_text(current)
            .show_ui(ui, |ui| {
                let mut pick = |value: &str, shown: &str| {
                    if ui.selectable_label(bone.parent == value, shown).clicked()
                        && bone.parent != value
                    {
                        params.set(
                            "parent",
                            kreate_reflect::PropertyValue::Text(value.to_string()),
                        );
                        changed = true;
                    }
                };
                pick("", &none);
                for (i, b) in armature.bones.iter().enumerate() {
                    if !kreate_scene::armature::is_ancestor(armature, index, i) {
                        pick(&b.name, &b.name);
                    }
                }
            });
    });
    if changed {
        let dragging = ui.ctx().input(|i| i.pointer.any_down());
        send_edit(state, "armature.bone_set", params, dragging);
    }
}

fn draw_light_data(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    l: &Labels,
    id: kreate_scene::LightId,
) {
    let Some(light) = doc.light(id) else {
        return;
    };
    let Some(info) = registry.get("light.set_properties") else {
        return;
    };

    ui.heading(&l.tab_data_light);
    // Said out loud, because it is the behaviour people are surprised by and
    // it is deliberate: the datablock is shared, so this edits every copy.
    // How many objects share this bulb. Counted here rather than stored,
    // because it is a question the panel asks once a frame and an answer that
    // would otherwise have to be kept in step with every add and delete.
    let users = doc
        .objects()
        .filter(|(_, o)| matches!(o.data, kreate_scene::document::ObjectData::Light(l) if l == id))
        .count();
    if users > 1 {
        let count = users.to_string();
        ui.weak(
            state
                .catalog
                .get_args("light-shared", &[("count", count.as_str())]),
        );
    }
    ui.separator();

    // The values that would leave the light exactly as it is. One function, in
    // the operator's own crate, so the panel and the operator cannot come to
    // disagree about what a field means — and a test says the round trip is an
    // identity.
    let mut params =
        kreate_reflect::OpParams::to_map(&kreate_ops::ops::object::light_params(light));
    // Only the numbers this kind has. All four kinds' parameters ride in one
    // block on purpose — changing the kind is one of the edits — but a point
    // light drawing a cone angle and an angular diameter was offering two
    // dials that did nothing. What a kind has is written down beside the
    // operator, so the panel and the operator cannot come to disagree.
    let kind = match params.get("kind") {
        Some(kreate_reflect::PropertyValue::Enum(k)) => *k,
        _ => 0,
    };
    kreate_ui::panel::section(ui, &l.panel_light, true, |ui| {
        let text = field_menu_text(&state.catalog);
        let out = kreate_ui::property_editor_full(
            ui,
            &state.catalog,
            info.id,
            info.schema,
            &mut params,
            |name| kreate_ops::ops::object::light_param_applies(kind, name),
            |name| light_animated(name).map(|p| key_mark(doc, light.action, p)),
            &l.key_property,
            |name| {
                Some(kreate_ui::FieldMenu {
                    animatable: light_animated(name).is_some(),
                    keyed_here: light_animated(name).is_some_and(|p| {
                        key_mark(doc, light.action, p) == kreate_ui::KeyMark::Keyed
                    }),
                    driven: Vec::new(),
                    resettable: true,
                })
            },
            &text,
        );
        if out.changed {
            let dragging = ui.ctx().input(|i| i.pointer.any_down());
            send_edit(state, "light.set_properties", params.clone(), dragging);
        }
        if let Some(property) = out.keyed.and_then(light_animated) {
            toggle_key(state, doc, light.action, 1, 0, property);
        }
        if let Some((field, action)) = out.action {
            field_action(state, info, params, field, action, FieldOwner::Light);
        }
    });
}

/// The animatable property a material panel field drives, if any.
fn material_animated(field: &str) -> Option<&'static str> {
    Some(match field {
        "color" => "base_color",
        "metallic" => "metallic",
        "roughness" => "roughness",
        "alpha" => "alpha",
        "normal_strength" => "normal_strength",
        _ => return None,
    })
}

/// The animatable property a light panel field drives, if any.
fn light_animated(field: &str) -> Option<&'static str> {
    Some(match field {
        "color" => "color",
        "intensity" => "intensity",
        "exposure" => "exposure",
        _ => return None,
    })
}

/// How a property's diamond looks: keyed on the frame the clock is on, keyed
/// elsewhere, or not at all.
fn key_mark(
    doc: &Document,
    action: Option<kreate_scene::ActionId>,
    property: &str,
) -> kreate_ui::KeyMark {
    let Some(action) = action.and_then(|a| doc.action(a)) else {
        return kreate_ui::KeyMark::None;
    };
    let mut curves = action
        .curves
        .iter()
        .filter(|c| c.path.property.as_ref() == property)
        .peekable();
    if curves.peek().is_none() {
        return kreate_ui::KeyMark::None;
    }
    let frame = doc.playback.frame.round();
    if curves.any(|c| {
        c.keys
            .iter()
            .any(|k| (k.frame - frame).abs() < kreate_scene::animation::SAME_FRAME)
    }) {
        kreate_ui::KeyMark::Keyed
    } else {
        kreate_ui::KeyMark::Animated
    }
}

/// Keys a property, or takes its key away if it is keyed on this frame — what
/// clicking its diamond means. Through `Dispatch`, so the application fills
/// in the frame the clock is on.
fn toggle_key(
    state: &mut ShellState,
    doc: &Document,
    action: Option<kreate_scene::ActionId>,
    datablock: u32,
    slot: u32,
    property: &str,
) {
    use kreate_reflect::PropertyValue as V;
    let mut p = PropertyMap::new();
    p.set("datablock", V::Enum(datablock));
    p.set("slot", V::Int(slot as i64));
    p.set("property", V::Text(property.to_string()));
    let op = if key_mark(doc, action, property) == kreate_ui::KeyMark::Keyed {
        "anim.keyframe_delete_property"
    } else {
        "anim.keyframe_insert_property"
    };
    state.queue.dispatch(op, p);
}

/// Draws the modifier stack for the active object.
fn draw_modifier_stack(ui: &mut Ui, state: &mut ShellState, rows: &mut [kreate_ui::ModifierRow]) {
    let text = kreate_ui::modifier_stack::ModifierText::resolve(&state.catalog, KIND_KEYS);
    kreate_ui::modifier_stack(ui, &text, &state.catalog, rows, &mut state.queue);
}

/// Translation keys for the addable modifier kinds, in the order the add
/// operator's enum expects.
///
/// Two lists that have to agree; a test holds them to it.
const KIND_KEYS: &[&str] = &[
    "modifier-mirror",
    "modifier-subdivide",
    "modifier-array",
    "modifier-solidify",
    "modifier-bevel",
    "modifier-boolean",
    "modifier-path-array",
];

/// The built-in English catalog.
///
/// Shipped as the fallback so an untranslated key resolves to readable text
/// rather than to itself. The Korean resource sits beside it.
pub const EN_FTL: &str = include_str!("../locales/en.ftl");

pub const KO_FTL: &str = include_str!("../locales/ko.ftl");

/// Loads the catalog for a locale, with English as the fallback.
pub fn load_catalog(locale: &str) -> Catalog {
    let mut catalog = Catalog::new();
    if locale.starts_with("ko") {
        if let Err(e) = catalog.add("ko-KR", KO_FTL) {
            tracing::error!(?e, "the built-in Korean catalog failed to load");
        }
    }
    if let Err(e) = catalog.add("en-US", EN_FTL) {
        tracing::error!(?e, "the built-in English catalog failed to load");
    }
    catalog
}

// --- the Fusion-style frame: application bar, toolbar, history strip --------

/// A picture for a toolbar button or an application-bar button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RibbonIcon {
    Plane,
    Cube,
    Sphere,
    Cylinder,
    Light,
    Camera,
    Bone,
    Move,
    Rotate,
    Scale,
    Duplicate,
    Delete,
    Extrude,
    Inset,
    Bevel,
    LoopCut,
    Knife,
    Subdivide,
    Merge,
    Unwrap,
    Key,
    Save,
    Undo,
    Redo,
    Search,
    SkLine,
    SkRect,
    SkCircle,
    SkArc,
    SkPoint,
    SkDimension,
    SkCoincident,
    SkPerpendicular,
    SkHorizontal,
    SkFinish,
    SkSelect,
    SkPattern,
}

impl RibbonIcon {
    /// Which picture a menu entry gets on the toolbar, if any. An entry with
    /// none stays in its group's menu only.
    fn of(entry: &MenuEntry) -> Option<Self> {
        use kreate_reflect::PropertyValue as V;
        let variant = |name: &str| {
            entry.variant.iter().find_map(|(n, v)| match v {
                V::Enum(i) if *n == name => Some(*i),
                _ => None,
            })
        };
        let id = entry.id;
        if id == "app.sketch_tool" {
            let tool = entry.variant.iter().find_map(|(n, v)| match v {
                V::Text(t) if *n == "tool" => Some(t.as_str()),
                _ => None,
            })?;
            return Some(match tool {
                "line" => Self::SkLine,
                "rectangle" => Self::SkRect,
                "circle" => Self::SkCircle,
                "arc" => Self::SkArc,
                "point" => Self::SkPoint,
                "select" => Self::SkSelect,
                "pattern" => Self::SkPattern,
                "dimension" => Self::SkDimension,
                "coincident" => Self::SkCoincident,
                "perpendicular" => Self::SkPerpendicular,
                "horizontal" => Self::SkHorizontal,
                "extrude" => Self::Extrude,
                "revolve" => Self::Rotate,
                _ => return None,
            });
        }
        if id == "app.finish_sketch" {
            return Some(Self::SkFinish);
        }
        Some(match id {
            "object.add_primitive" => match variant("shape")? {
                0 => Self::Plane,
                1 => Self::Cube,
                3 => Self::Sphere,
                5 => Self::Cylinder,
                _ => return None,
            },
            "object.add_light" if variant("kind") == Some(0) => Self::Light,
            "object.add_camera" => Self::Camera,
            "object.add_armature" | "armature.bone_add" => Self::Bone,
            "mesh.inset" => Self::Inset,
            "mesh.bevel" => Self::Bevel,
            "mesh.loop_cut" => Self::LoopCut,
            "mesh.knife" => Self::Knife,
            "mesh.subdivide" => Self::Subdivide,
            "mesh.merge" => Self::Merge,
            "uv.unwrap" => Self::Unwrap,
            "anim.keyframe_insert" => Self::Key,
            _ if id.starts_with("mesh.extrude") || id == "armature.extrude" => Self::Extrude,
            _ if id.ends_with(".translate") => Self::Move,
            _ if id.ends_with(".rotate") => Self::Rotate,
            _ if id.ends_with(".scale") => Self::Scale,
            _ if id.ends_with(".duplicate") => Self::Duplicate,
            _ if id.ends_with(".delete") => Self::Delete,
            _ => return None,
        })
    }

    /// Making things is the accent's; changing a shape the informational
    /// blue; the rest the text's own colour.
    fn colour(self, theme: &kreate_ui::Theme) -> egui::Color32 {
        match self {
            Self::Plane
            | Self::Cube
            | Self::Sphere
            | Self::Cylinder
            | Self::Light
            | Self::Camera
            | Self::Bone
            | Self::SkLine
            | Self::SkRect
            | Self::SkCircle
            | Self::SkArc
            | Self::SkPoint
            | Self::SkPattern => theme.accent,
            Self::SkSelect => theme.text,
            Self::SkDimension | Self::SkCoincident | Self::SkPerpendicular | Self::SkHorizontal => {
                theme.info
            }
            Self::Extrude
            | Self::Inset
            | Self::Bevel
            | Self::LoopCut
            | Self::Knife
            | Self::Subdivide
            | Self::Merge
            | Self::Unwrap => theme.info,
            _ => theme.text,
        }
    }

    /// Draws the picture in `rect`, in strokes.
    fn paint(self, painter: &egui::Painter, rect: egui::Rect, colour: egui::Color32) {
        use egui::{pos2, Shape, Stroke};
        let s = rect.width().min(rect.height());
        let o = rect.center() - egui::vec2(s, s) * 0.5;
        let p = |x: f32, y: f32| pos2(o.x + x * s, o.y + y * s);
        let stroke = Stroke::new(1.5, colour);
        let line = |pts: Vec<egui::Pos2>| painter.add(Shape::line(pts, stroke));
        let closed = |pts: Vec<egui::Pos2>| painter.add(Shape::closed_line(pts, stroke));
        let ellipse = |cx: f32, cy: f32, rx: f32, ry: f32, from: f32, to: f32| {
            (0..=24)
                .map(|i| {
                    let t = from + (to - from) * i as f32 / 24.0;
                    p(cx + rx * t.cos(), cy + ry * t.sin())
                })
                .collect::<Vec<_>>()
        };
        let tau = std::f32::consts::TAU;
        let pi = std::f32::consts::PI;
        match self {
            Self::Plane => {
                closed(vec![p(0.1, 0.6), p(0.5, 0.4), p(0.9, 0.6), p(0.5, 0.8)]);
            }
            Self::Cube => {
                closed(vec![
                    p(0.5, 0.1),
                    p(0.88, 0.3),
                    p(0.88, 0.72),
                    p(0.5, 0.92),
                    p(0.12, 0.72),
                    p(0.12, 0.3),
                ]);
                line(vec![p(0.12, 0.3), p(0.5, 0.5), p(0.88, 0.3)]);
                line(vec![p(0.5, 0.5), p(0.5, 0.92)]);
            }
            Self::Sphere => {
                painter.circle_stroke(p(0.5, 0.5), 0.38 * s, stroke);
                line(ellipse(0.5, 0.5, 0.38, 0.13, 0.0, pi));
            }
            Self::Cylinder => {
                closed(ellipse(0.5, 0.25, 0.3, 0.1, 0.0, tau));
                line(vec![p(0.2, 0.25), p(0.2, 0.75)]);
                line(vec![p(0.8, 0.25), p(0.8, 0.75)]);
                line(ellipse(0.5, 0.75, 0.3, 0.1, 0.0, pi));
            }
            Self::Light => {
                painter.circle_stroke(p(0.5, 0.45), 0.17 * s, stroke);
                for k in 0..8 {
                    let a = k as f32 / 8.0 * tau;
                    let (c, n) = (a.cos(), a.sin());
                    line(vec![
                        p(0.5 + 0.26 * c, 0.45 + 0.26 * n),
                        p(0.5 + 0.38 * c, 0.45 + 0.38 * n),
                    ]);
                }
            }
            Self::Camera => {
                closed(vec![p(0.1, 0.3), p(0.62, 0.3), p(0.62, 0.74), p(0.1, 0.74)]);
                closed(vec![
                    p(0.62, 0.46),
                    p(0.9, 0.32),
                    p(0.9, 0.72),
                    p(0.62, 0.58),
                ]);
            }
            Self::Bone => {
                line(vec![p(0.28, 0.72), p(0.72, 0.28)]);
                painter.circle_stroke(p(0.22, 0.78), 0.09 * s, stroke);
                painter.circle_stroke(p(0.78, 0.22), 0.09 * s, stroke);
            }
            Self::Move => {
                line(vec![p(0.5, 0.1), p(0.5, 0.9)]);
                line(vec![p(0.1, 0.5), p(0.9, 0.5)]);
                line(vec![p(0.38, 0.22), p(0.5, 0.1), p(0.62, 0.22)]);
                line(vec![p(0.78, 0.38), p(0.9, 0.5), p(0.78, 0.62)]);
                line(vec![p(0.38, 0.78), p(0.5, 0.9), p(0.62, 0.78)]);
                line(vec![p(0.22, 0.38), p(0.1, 0.5), p(0.22, 0.62)]);
            }
            Self::Rotate => {
                line(ellipse(0.5, 0.5, 0.36, 0.36, -0.3 * pi, 1.4 * pi));
                let tip = p(
                    0.5 + 0.36 * (-0.3 * pi).cos(),
                    0.5 + 0.36 * (-0.3 * pi).sin(),
                );
                line(vec![
                    tip + egui::vec2(-0.2 * s, -0.02 * s),
                    tip,
                    tip + egui::vec2(0.0, 0.2 * s),
                ]);
            }
            Self::Scale => {
                closed(vec![
                    p(0.12, 0.48),
                    p(0.52, 0.48),
                    p(0.52, 0.88),
                    p(0.12, 0.88),
                ]);
                line(vec![p(0.52, 0.48), p(0.88, 0.12)]);
                line(vec![p(0.62, 0.12), p(0.88, 0.12), p(0.88, 0.38)]);
            }
            Self::Duplicate => {
                closed(vec![
                    p(0.12, 0.32),
                    p(0.62, 0.32),
                    p(0.62, 0.88),
                    p(0.12, 0.88),
                ]);
                line(vec![
                    p(0.36, 0.32),
                    p(0.36, 0.12),
                    p(0.88, 0.12),
                    p(0.88, 0.66),
                    p(0.62, 0.66),
                ]);
            }
            Self::Delete => {
                line(vec![p(0.2, 0.2), p(0.8, 0.8)]);
                line(vec![p(0.8, 0.2), p(0.2, 0.8)]);
            }
            Self::Extrude => {
                closed(vec![
                    p(0.12, 0.62),
                    p(0.88, 0.62),
                    p(0.88, 0.88),
                    p(0.12, 0.88),
                ]);
                line(vec![p(0.5, 0.54), p(0.5, 0.1)]);
                line(vec![p(0.34, 0.26), p(0.5, 0.1), p(0.66, 0.26)]);
            }
            Self::Inset => {
                closed(vec![p(0.1, 0.1), p(0.9, 0.1), p(0.9, 0.9), p(0.1, 0.9)]);
                closed(vec![
                    p(0.32, 0.32),
                    p(0.68, 0.32),
                    p(0.68, 0.68),
                    p(0.32, 0.68),
                ]);
            }
            Self::Bevel => {
                let mut pts = vec![p(0.14, 0.88)];
                pts.extend(ellipse(0.6, 0.6, 0.46, 0.46, pi, 1.5 * pi));
                pts.push(p(0.88, 0.14));
                line(pts);
            }
            Self::LoopCut => {
                closed(vec![p(0.1, 0.18), p(0.9, 0.18), p(0.9, 0.82), p(0.1, 0.82)]);
                for k in 0..4 {
                    let y = 0.22 + k as f32 * 0.16;
                    line(vec![p(0.5, y), p(0.5, y + 0.09)]);
                }
            }
            Self::Knife => {
                line(vec![p(0.15, 0.85), p(0.7, 0.3)]);
                closed(vec![p(0.62, 0.22), p(0.86, 0.1), p(0.78, 0.38)]);
            }
            Self::Subdivide => {
                closed(vec![p(0.1, 0.1), p(0.9, 0.1), p(0.9, 0.9), p(0.1, 0.9)]);
                line(vec![p(0.5, 0.1), p(0.5, 0.9)]);
                line(vec![p(0.1, 0.5), p(0.9, 0.5)]);
            }
            Self::Merge => {
                line(vec![p(0.15, 0.15), p(0.5, 0.5), p(0.85, 0.15)]);
                line(vec![p(0.5, 0.5), p(0.5, 0.9)]);
                painter.circle_filled(p(0.5, 0.5), 0.07 * s, colour);
            }
            Self::Unwrap => {
                closed(vec![p(0.1, 0.1), p(0.9, 0.1), p(0.9, 0.9), p(0.1, 0.9)]);
                line(vec![p(0.1, 0.5), p(0.9, 0.5)]);
                line(vec![p(0.5, 0.1), p(0.5, 0.5)]);
            }
            Self::Key => {
                closed(vec![p(0.5, 0.12), p(0.86, 0.5), p(0.5, 0.88), p(0.14, 0.5)]);
            }
            Self::Save => {
                closed(vec![
                    p(0.14, 0.12),
                    p(0.72, 0.12),
                    p(0.88, 0.28),
                    p(0.88, 0.88),
                    p(0.14, 0.88),
                ]);
                line(vec![
                    p(0.3, 0.12),
                    p(0.3, 0.36),
                    p(0.66, 0.36),
                    p(0.66, 0.12),
                ]);
                closed(vec![p(0.3, 0.6), p(0.72, 0.6), p(0.72, 0.88), p(0.3, 0.88)]);
            }
            Self::Undo | Self::Redo => {
                let flip = |x: f32| if self == Self::Undo { x } else { 1.0 - x };
                let arc: Vec<egui::Pos2> = (0..=16)
                    .map(|i| {
                        let t = -0.5 * pi + pi * i as f32 / 16.0;
                        p(flip(0.52 + 0.3 * t.cos()), 0.56 + 0.26 * t.sin())
                    })
                    .collect();
                let mut pts = vec![p(flip(0.16), 0.3)];
                pts.extend(arc);
                pts.push(p(flip(0.4), 0.82));
                line(pts);
                line(vec![
                    p(flip(0.32), 0.14),
                    p(flip(0.16), 0.3),
                    p(flip(0.32), 0.46),
                ]);
            }
            Self::Search => {
                painter.circle_stroke(p(0.44, 0.44), 0.28 * s, stroke);
                line(vec![p(0.64, 0.64), p(0.88, 0.88)]);
            }
            Self::SkLine => {
                line(vec![p(0.15, 0.85), p(0.85, 0.15)]);
                painter.circle_filled(p(0.15, 0.85), 0.08 * s, colour);
                painter.circle_filled(p(0.85, 0.15), 0.08 * s, colour);
            }
            Self::SkRect => {
                closed(vec![
                    p(0.12, 0.25),
                    p(0.88, 0.25),
                    p(0.88, 0.75),
                    p(0.12, 0.75),
                ]);
            }
            Self::SkCircle => {
                painter.circle_stroke(p(0.5, 0.5), 0.36 * s, stroke);
                painter.circle_filled(p(0.5, 0.5), 0.05 * s, colour);
            }
            Self::SkArc => {
                line(ellipse(0.5, 0.65, 0.38, 0.38, pi, tau));
            }
            Self::SkSelect => {
                // An arrow pointer.
                closed(vec![
                    p(0.3, 0.12),
                    p(0.3, 0.8),
                    p(0.45, 0.65),
                    p(0.58, 0.9),
                    p(0.68, 0.85),
                    p(0.55, 0.6),
                    p(0.75, 0.6),
                ]);
            }
            Self::SkPattern => {
                // Three squares in a row and a fourth below: copies.
                for (x, y) in [(0.1, 0.15), (0.4, 0.15), (0.7, 0.15), (0.1, 0.55)] {
                    closed(vec![
                        p(x, y),
                        p(x + 0.2, y),
                        p(x + 0.2, y + 0.2),
                        p(x, y + 0.2),
                    ]);
                }
            }
            Self::SkPoint => {
                painter.circle_filled(p(0.5, 0.5), 0.12 * s, colour);
                line(vec![p(0.2, 0.5), p(0.35, 0.5)]);
                line(vec![p(0.65, 0.5), p(0.8, 0.5)]);
            }
            Self::SkDimension => {
                line(vec![p(0.12, 0.3), p(0.12, 0.8)]);
                line(vec![p(0.88, 0.3), p(0.88, 0.8)]);
                line(vec![p(0.12, 0.55), p(0.88, 0.55)]);
                line(vec![p(0.25, 0.45), p(0.12, 0.55), p(0.25, 0.65)]);
                line(vec![p(0.75, 0.45), p(0.88, 0.55), p(0.75, 0.65)]);
            }
            Self::SkCoincident => {
                line(vec![p(0.1, 0.8), p(0.5, 0.5)]);
                line(vec![p(0.5, 0.5), p(0.9, 0.75)]);
                painter.circle_filled(p(0.5, 0.5), 0.1 * s, colour);
            }
            Self::SkPerpendicular => {
                line(vec![p(0.15, 0.85), p(0.85, 0.85)]);
                line(vec![p(0.5, 0.85), p(0.5, 0.15)]);
            }
            Self::SkHorizontal => {
                line(vec![p(0.1, 0.5), p(0.9, 0.5)]);
                line(vec![p(0.2, 0.3), p(0.2, 0.7)]);
                line(vec![p(0.8, 0.3), p(0.8, 0.7)]);
            }
            Self::SkFinish => {
                line(vec![p(0.15, 0.55), p(0.4, 0.8), p(0.88, 0.22)]);
            }
        }
    }
}

/// A small icon-only button on the application bar.
fn bar_button(ui: &mut Ui, icon: RibbonIcon, enabled: bool, tip: &str) -> egui::Response {
    let theme = kreate_ui::theme::from_ui(ui);
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(26.0, 24.0), sense);
    if response.hovered() && enabled {
        ui.painter().rect_filled(rect, 4.0, theme.border);
    }
    let colour = if enabled {
        theme.text_dim
    } else {
        theme.border_strong
    };
    icon.paint(ui.painter(), rect.shrink2(egui::vec2(6.0, 5.0)), colour);
    response.on_hover_text(tip)
}

/// What the application bar carries after the menus: save, undo and redo
/// as buttons, the open document, and the command search.
fn draw_app_bar(ui: &mut Ui, state: &mut ShellState, l: &Labels, can_undo: bool, can_redo: bool) {
    let theme = kreate_ui::theme::from_ui(ui);
    ui.spacing_mut().item_spacing.x = 2.0;
    if bar_button(ui, RibbonIcon::Save, true, &l.file_save).clicked() {
        state.queue.push(Command::SaveDocument);
    }
    if bar_button(ui, RibbonIcon::Undo, can_undo, &l.undo).clicked() {
        state.queue.push(Command::Undo);
    }
    if bar_button(ui, RibbonIcon::Redo, can_redo, &l.redo).clicked() {
        state.queue.push(Command::Redo);
    }
    ui.add_space(10.0);
    // The document, as a tab: its name, and a dot while it has changes no
    // save has kept.
    let (modified, name) = match state.document_title.strip_prefix('*') {
        Some(name) => (true, name.to_string()),
        None => (false, state.document_title.clone()),
    };
    egui::Frame::new()
        .fill(theme.sunken)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(egui::CornerRadius {
            nw: 6,
            ne: 6,
            sw: 0,
            se: 0,
        })
        .inner_margin(egui::Margin::symmetric(10, 2))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                // i18n-exempt: the user's own file name
                ui.label(egui::RichText::new(name).color(theme.text));
                if modified {
                    let (dot, _) =
                        ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().circle_filled(dot.center(), 3.0, theme.accent);
                }
            });
        });
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        let tip = state
            .command_info
            .get("app.search")
            .map(|i| i.name.clone())
            .unwrap_or_else(|| state.catalog.get("search-commands"));
        if bar_button(ui, RibbonIcon::Search, true, &tip).clicked() {
            state.palette.open();
        }
    });
}

/// The toolbar under the application bar, as Fusion lays it out: the
/// workspace on the left, then the mode tabs over the mode's tool groups.
///
/// The groups are the mode's own menus — the same tree the header menus and
/// F3 read — with the entries that have a picture brought out as buttons, so
/// a tool is one click and the rest of its group is one more.
fn draw_ribbon(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    l: &Labels,
    mode: kreate_ops::operator::Mode,
) {
    use kreate_ops::operator::Mode;
    let theme = kreate_ui::theme::from_ui(ui);
    Panel::top("ribbon")
        .frame(
            egui::Frame::new()
                .fill(theme.base)
                .inner_margin(egui::Margin::symmetric(8, 4)),
        )
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                draw_workspace_switcher(ui, state);
                ui.separator();
                ui.vertical(|ui| {
                    // The mode tabs: this object's modes, as `Mode::offered`
                    // decides, underlined in the mode's tint when open.
                    let has = kreate_ops::mode::Has::of(doc);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        for kind in Mode::ALL
                            .iter()
                            .copied()
                            .filter(|k| *k == mode || k.offered(has))
                        {
                            let name = state
                                .catalog
                                .format(kind.label_key(), TextKind::Ui, &[])
                                .to_uppercase();
                            let open = kind == mode;
                            let colour = if open { theme.text } else { theme.text_dim };
                            let response = ui.add_enabled(
                                kind.can_enter(has),
                                egui::Button::new(
                                    egui::RichText::new(name).small().strong().color(colour),
                                )
                                .frame(false)
                                .min_size(egui::vec2(0.0, 20.0)),
                            );
                            let r = response.rect;
                            if open {
                                let under = theme
                                    .mode_tint(kind.tag())
                                    .map_or(theme.accent, |(_, text)| text);
                                ui.painter().line_segment(
                                    [
                                        egui::pos2(r.left() + 4.0, r.bottom()),
                                        egui::pos2(r.right() - 4.0, r.bottom()),
                                    ],
                                    egui::Stroke::new(2.0, under),
                                );
                            }
                            if response.clicked() && !open {
                                state.queue.push(Command::SetMode(kind.tag().to_string()));
                            }
                            ui.add_space(10.0);
                        }
                    });
                    // The groups.
                    ui.horizontal(|ui| {
                        let menus = viewport_menus(l, mode, state.select_mode);
                        for (i, menu) in menus.iter().enumerate() {
                            if i > 0 {
                                ui.separator();
                            }
                            draw_tool_group(ui, state, l, menu, &theme);
                        }
                    });
                });
            });
        });
}

/// One group of the toolbar: up to three of its entries as buttons, and its
/// name, which opens the whole menu.
fn draw_tool_group(
    ui: &mut Ui,
    state: &mut ShellState,
    l: &Labels,
    menu: &TopMenu,
    theme: &kreate_ui::Theme,
) {
    fn featured<'a>(
        nodes: &'a [MenuNode],
        most: usize,
        out: &mut Vec<(&'a MenuEntry, RibbonIcon)>,
    ) {
        for node in nodes {
            match node {
                MenuNode::Entry(e) => {
                    if let Some(icon) = RibbonIcon::of(e) {
                        if out.len() < most && !out.iter().any(|(_, i)| *i == icon) {
                            out.push((e, icon));
                        }
                    }
                }
                MenuNode::Sub(_, children) => featured(children, most, out),
                _ => {}
            }
        }
    }
    // The sketch's drawing tools are what it is used for: all of them on
    // show, selecting first.
    let most = if menu.label == l.menu_sketch { 6 } else { 3 };
    let mut buttons = Vec::new();
    featured(&menu.nodes, most, &mut buttons);
    // The sketch tool in hand, lit, so it is plain whether a click draws.
    let in_hand = state.sketch_view.as_ref().map(|v| v.tool);
    let held = |entry: &MenuEntry| {
        entry.id == "app.sketch_tool"
            && entry.variant.iter().any(|(n, v)| {
                *n == "tool"
                    && matches!(v, kreate_reflect::PropertyValue::Text(t) if Some(t.as_str()) == in_hand)
            })
    };
    // As wide as its buttons, its name centred under them; a group with no
    // buttons is its name alone, at the buttons' height.
    let width = (buttons.len() as f32 * 52.0).max(56.0);
    let layout = egui::Layout::top_down(egui::Align::Center);
    ui.allocate_ui_with_layout(egui::vec2(width, 64.0), layout, |ui| {
        if buttons.is_empty() {
            ui.add_space(20.0);
        }
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            for (entry, icon) in &buttons {
                let size = egui::vec2(52.0, 40.0);
                let sense = if entry.enabled {
                    egui::Sense::click()
                } else {
                    egui::Sense::hover()
                };
                let (rect, response) = ui.allocate_exact_size(size, sense);
                if held(entry) {
                    ui.painter()
                        .rect_filled(rect, 5.0, theme.accent.gamma_multiply(0.25));
                    ui.painter().rect_stroke(
                        rect.shrink(0.5),
                        5.0,
                        egui::Stroke::new(1.0, theme.accent),
                        egui::StrokeKind::Inside,
                    );
                } else if response.hovered() && entry.enabled {
                    ui.painter().rect_filled(rect, 5.0, theme.border);
                }
                let picture = egui::Rect::from_center_size(
                    egui::pos2(rect.center().x, rect.top() + 13.0),
                    egui::vec2(20.0, 20.0),
                );
                let colour = if entry.enabled {
                    icon.colour(theme)
                } else {
                    theme.border_strong
                };
                icon.paint(ui.painter(), picture, colour);
                // The name under it, cut to the button's width.
                let font = egui::FontId::proportional(10.0);
                let mut text = entry.label.clone();
                let fits = |t: &str| {
                    ui.painter()
                        .layout_no_wrap(t.to_string(), font.clone(), theme.text)
                        .size()
                        .x
                        <= size.x - 4.0
                };
                while !fits(&text) && text.chars().count() > 1 {
                    text.pop();
                }
                ui.painter().text(
                    egui::pos2(rect.center().x, rect.bottom() - 2.0),
                    egui::Align2::CENTER_BOTTOM,
                    text,
                    font,
                    if entry.enabled {
                        theme.text
                    } else {
                        theme.text_faint
                    },
                );
                let tip = match state.command_info.get(entry.id) {
                    Some(info) if !info.shortcut.is_empty() => {
                        format!("{}  {}", entry.label, info.shortcut)
                    }
                    _ => entry.label.clone(),
                };
                if response.on_hover_text(tip).clicked() {
                    run_entry(state, entry);
                }
            }
        });
        ui.menu_button(
            egui::RichText::new(menu.label.to_uppercase())
                .small()
                .strong()
                .color(theme.text_dim),
            |ui| draw_nodes(ui, state, l, &menu.nodes),
        );
    });
}

/// The history as a strip of steps along the bottom, as Fusion's timeline
/// shows a design's: the steps that are applied, a marker, the ones undone
/// after it. A click on a step goes back, or forward, to just after it.
fn draw_history_strip(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    let theme = kreate_ui::theme::from_ui(ui);
    Panel::bottom("history")
        .frame(
            egui::Frame::new()
                .fill(theme.base)
                .inner_margin(egui::Margin::symmetric(8, 3)),
        )
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let at = state.history_at;
                let total = state.history_steps.len();
                ui.spacing_mut().item_spacing.x = 2.0;
                if bar_button(ui, RibbonIcon::Undo, at > 0, &l.undo).clicked() {
                    state.queue.push(Command::Undo);
                }
                if bar_button(ui, RibbonIcon::Redo, at < total, &l.redo).clicked() {
                    state.queue.push(Command::Redo);
                }
                ui.add_space(8.0);
                egui::ScrollArea::horizontal()
                    .id_salt("history-strip")
                    .stick_to_right(true)
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 3.0;
                            // The last few dozen: a history of thousands is
                            // not a strip.
                            const SHOWN: usize = 60;
                            let first = total.saturating_sub(SHOWN);
                            if at == first && first == 0 {
                                marker(ui, &theme);
                            }
                            for (i, label) in state.history_steps.iter().enumerate().skip(first) {
                                let applied = i + 1;
                                let undone = applied > at;
                                let short: String = label.chars().take(14).collect();
                                let short = if short.chars().count() < label.chars().count() {
                                    format!("{short}…")
                                } else {
                                    short
                                };
                                let text = egui::RichText::new(short).small().color(if undone {
                                    theme.text_faint
                                } else {
                                    theme.text
                                });
                                let chip = egui::Button::new(text)
                                    .fill(if undone {
                                        egui::Color32::TRANSPARENT
                                    } else {
                                        theme.raised
                                    })
                                    .stroke(egui::Stroke::new(
                                        1.0,
                                        if undone { theme.border } else { theme.raised },
                                    ))
                                    .corner_radius(4);
                                if ui.add(chip).on_hover_text(label.as_str()).clicked() {
                                    state.queue.push(Command::UndoTo(applied));
                                }
                                if applied == at {
                                    marker(ui, &theme);
                                }
                            }
                        });
                    });
            });
        });

    fn marker(ui: &mut Ui, theme: &kreate_ui::Theme) {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(4.0, 20.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, theme.accent);
    }
}

/// The sketch tools, by the names the keys and menus use.
const SKETCH_TOOLS: [&str; 20] = [
    "select",
    "pattern",
    "line",
    "rectangle",
    "circle",
    "arc",
    "point",
    "trim",
    "dimension",
    "coincident",
    "perpendicular",
    "horizontal",
    "vertical",
    "parallel",
    "equal",
    "tangent",
    "concentric",
    "fixed",
    "extrude",
    "revolve",
];

/// Menu entries that take up sketch tools, by name.
fn sketch_tool_nodes(l: &Labels, names: &[&'static str]) -> Vec<MenuNode> {
    use kreate_reflect::PropertyValue as V;
    names
        .iter()
        .map(|name| {
            let label = l
                .sketch_tools
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| t.clone())
                .unwrap_or_else(|| name.to_string());
            let mut params = PropertyMap::new();
            params.set("tool", V::Text(name.to_string()));
            MenuNode::Entry(
                MenuEntry::new(
                    "app.sketch_tool",
                    &label,
                    Run::Dispatch("app.sketch_tool", params),
                )
                .variant("tool", V::Text(name.to_string())),
            )
        })
        .collect()
}

// --- sketching: the overlay, the dialogs, the feature list -------------------

/// The sketch being drawn, as the overlay draws it. Filled by the
/// application each frame from the sketch and the tools' state; every point
/// is in the world.
#[derive(Clone, Debug, Default)]
pub struct SketchView {
    /// Each line, and whether both its ends are pinned — Fusion's black
    /// (here the text colour) against its blue.
    pub lines: Vec<([kreate_math::DVec3; 2], bool, bool)>,
    /// The points a tool can pick: pinned, selected.
    pub points: Vec<(kreate_math::DVec3, bool, bool)>,
    /// A constraint's mark, where it is drawn.
    pub glyphs: Vec<(kreate_math::DVec3, String)>,
    pub dims: Vec<SketchDim>,
    /// What the tool in hand would draw, as polylines.
    pub preview: Vec<Vec<kreate_math::DVec3>>,
    /// What the trim tool would take out.
    pub cut: Vec<[kreate_math::DVec3; 2]>,
    /// The point the pointer has snapped to.
    pub snap: Option<kreate_math::DVec3>,
    /// The regions picked for an extrude or a revolve, and the one under
    /// the pointer.
    pub regions: Vec<RegionShape>,
    pub hover_region: Option<RegionShape>,
    /// The tool in hand, by name, and what it asks for next.
    pub tool: &'static str,
    pub prompt: String,
    /// What has been typed.
    pub typed: String,
    /// The picked regions' seeds and the revolve's axis, for the dialog.
    pub seeds: Vec<[f64; 2]>,
    pub axis: Option<[u32; 2]>,
    /// The scene's length unit: metres in one, and its symbol.
    pub unit: (f64, &'static str),
    /// What is selected, by index: the points and the lines.
    pub picked_points: Vec<u32>,
    pub picked_lines: Vec<u32>,
    /// A drag-box being drawn, corner to corner, on screen.
    pub band: Option<[kreate_math::DVec2; 2]>,
    /// Whether the tool in hand draws, for the badge and the pointer.
    pub drawing: bool,
}

/// A closed area of a sketch, for drawing: its outline and holes, and the
/// triangles that fill it.
#[derive(Clone, Debug, Default)]
pub struct RegionShape {
    pub rings: Vec<Vec<kreate_math::DVec3>>,
    pub tris: Vec<[kreate_math::DVec3; 3]>,
}

/// One dimension on the sketch.
#[derive(Clone, Debug)]
pub struct SketchDim {
    /// Its constraint's place in the sketch's list.
    pub index: usize,
    /// Where its number is written, and the two points it measures.
    pub at: kreate_math::DVec3,
    pub ends: [kreate_math::DVec3; 2],
    pub text: String,
    /// What typing into it sets: the number as shown, and how to turn it
    /// back into the constraint's value (times the unit, halved for a
    /// diameter; radians for an angle).
    pub shown: f64,
    pub kind: DimKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimKind {
    Length,
    Diameter,
    Angle,
}

impl SketchDim {
    /// The constraint's value for a number typed into it.
    pub fn value_of(&self, typed: f64, unit_metres: f64) -> f64 {
        match self.kind {
            DimKind::Length => typed * unit_metres,
            DimKind::Diameter => typed * unit_metres * 0.5,
            DimKind::Angle => typed.to_radians(),
        }
    }
}

/// The sketch pattern dialog's settings, in the scene's units and degrees.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternDialog {
    pub circular: bool,
    pub count: u32,
    pub spacing: f64,
    pub count2: u32,
    pub spacing2: f64,
    pub angle: f64,
    /// The circular pattern's centre, in the sketch's plane, in metres.
    pub centre: [f64; 2],
    /// The next click in the view puts the centre.
    pub picking_centre: bool,
}

impl Default for PatternDialog {
    fn default() -> Self {
        Self {
            circular: false,
            count: 3,
            spacing: 10.0,
            count2: 1,
            spacing2: 10.0,
            angle: 360.0,
            centre: [0.0, 0.0],
            picking_centre: false,
        }
    }
}

impl PatternDialog {
    /// The operator's parameters, with `unit` metres to the scene's unit.
    pub fn params(&self, unit: f64) -> kreate_ops::ops::sketch_draw::PatternParams {
        kreate_ops::ops::sketch_draw::PatternParams {
            kind: u32::from(self.circular),
            count: self.count.max(1),
            spacing: self.spacing * unit,
            count2: if self.circular { 1 } else { self.count2.max(1) },
            spacing2: self.spacing2 * unit,
            centre: [self.centre[0], self.centre[1], 0.0],
            angle: self.angle.to_radians(),
            ..Default::default()
        }
    }
}

/// The extrude and revolve dialog's settings.
#[derive(Clone, Debug)]
pub struct SolidDialog {
    /// The distance as typed, in the scene's unit.
    pub distance: f64,
    pub symmetric: bool,
    /// 0 a new body, 1 join, 2 cut.
    pub operation: u32,
    /// The revolve's angle, in degrees.
    pub angle: f64,
    /// The operation was picked by hand: the automatic choice — cut going
    /// into something, join coming out of it, Fusion's — stands aside.
    pub chosen: bool,
    /// What a join or cut acts on, by name, as the automatic choice found
    /// it; empty for the latest body.
    pub target: String,
}

impl Default for SolidDialog {
    fn default() -> Self {
        Self {
            distance: 10.0,
            symmetric: false,
            operation: 0,
            angle: 360.0,
            chosen: false,
            target: String::new(),
        }
    }
}

/// A row of the feature list.
#[derive(Clone, Debug)]
pub struct FeatureRow {
    pub order: u64,
    /// A sketch's name, or a feature's ("Extrude 2").
    pub label: String,
    /// The body a feature is in, and its place there; empty for a sketch.
    pub body: String,
    pub index: u32,
    pub is_sketch: bool,
    pub revolve: bool,
    /// The extrude's distance in metres, or the revolve's angle in radians.
    pub amount: f64,
    pub symmetric: bool,
    pub cut: bool,
}

/// A feature open for editing from the list.
#[derive(Clone, Debug)]
pub struct FeatureEdit {
    pub row: FeatureRow,
    /// As shown: the distance in the scene's unit, or the angle in degrees.
    pub amount: f64,
}

fn tool_label(state: &ShellState, name: &str) -> String {
    let key = format!("sketch-tool-{name}");
    if state.catalog.has(&key) {
        state.catalog.get(&key)
    } else {
        name.to_string()
    }
}

/// Every sketch's lines, and the one being drawn with all it carries.
fn draw_sketches(
    ui: &Ui,
    painter: &egui::Painter,
    projector: &kreate_ui::object_glyph::Projector,
    state: &mut ShellState,
    inner: egui::Rect,
) {
    let theme = kreate_ui::theme::from_ui(ui).over_viewport();
    let blue = egui::Color32::from_rgb(0x4d, 0x9b, 0xf0);
    let pinned_ink = theme.text;
    for [a, b] in &state.sketch_lines {
        if let (Some(p), Some(q)) = (projector.project(*a), projector.project(*b)) {
            painter.line_segment([p, q], egui::Stroke::new(1.0, theme.text_dim));
        }
    }
    let Some(view) = state.sketch_view.clone() else {
        return;
    };
    let project = |p: &kreate_math::DVec3| projector.project(*p);
    // The regions picked, filled faintly; the one under the pointer brighter.
    let fill = |region: &RegionShape, alpha: u8| {
        let c = egui::Color32::from_rgba_unmultiplied(
            theme.accent.r(),
            theme.accent.g(),
            theme.accent.b(),
            alpha,
        );
        // The triangles the region is cut into, holes left out.
        let mut mesh = egui::Mesh::default();
        for t in &region.tris {
            let (Some(a), Some(b), Some(d)) = (project(&t[0]), project(&t[1]), project(&t[2]))
            else {
                continue;
            };
            let i = mesh.vertices.len() as u32;
            for p in [a, b, d] {
                mesh.colored_vertex(p, c);
            }
            mesh.add_triangle(i, i + 1, i + 2);
        }
        painter.add(egui::Shape::mesh(mesh));
        for ring in &region.rings {
            let pts: Vec<egui::Pos2> = ring.iter().filter_map(project).collect();
            if pts.len() == ring.len() && pts.len() >= 3 {
                painter.add(egui::Shape::closed_line(
                    pts,
                    egui::Stroke::new(1.5, theme.accent),
                ));
            }
        }
    };
    for r in &view.regions {
        fill(r, 70);
    }
    if let Some(r) = &view.hover_region {
        fill(r, 35);
    }
    let halo = egui::Color32::from_black_alpha(150);
    for ([a, b], pinned, selected) in &view.lines {
        if let (Some(p), Some(q)) = (project(a), project(b)) {
            let ink = if *selected {
                theme.accent
            } else if *pinned {
                pinned_ink
            } else {
                blue
            };
            // A dark edge under the line, so it reads on a light solid as on
            // the dark ground: a sketch is drawn over the bodies made from
            // it, and their faces are as pale as the line.
            painter.line_segment([p, q], egui::Stroke::new(3.6, halo));
            painter.line_segment([p, q], egui::Stroke::new(1.6, ink));
        }
    }
    for (at, pinned, selected) in &view.points {
        if let Some(p) = project(at) {
            let ink = if *selected {
                theme.accent
            } else if *pinned {
                pinned_ink
            } else {
                blue
            };
            painter.rect_filled(
                egui::Rect::from_center_size(p, egui::vec2(7.0, 7.0)),
                0.0,
                halo,
            );
            painter.rect_filled(
                egui::Rect::from_center_size(p, egui::vec2(5.0, 5.0)),
                0.0,
                ink,
            );
        }
    }
    let font = egui::FontId::proportional(11.0);
    for (at, text) in &view.glyphs {
        if let Some(p) = project(at) {
            let galley = painter.layout_no_wrap(text.clone(), font.clone(), theme.text);
            let r =
                egui::Align2::CENTER_CENTER.anchor_size(p + egui::vec2(8.0, -8.0), galley.size());
            painter.rect_filled(r.expand(2.0), 2.0, theme.sunken.gamma_multiply(0.85));
            painter.rect_stroke(
                r.expand(2.0),
                2.0,
                egui::Stroke::new(1.0, theme.border_strong),
                egui::StrokeKind::Outside,
            );
            painter.galley(r.min, galley, theme.text);
        }
    }
    // Dimensions: a thin line between what they measure and the number.
    let pointer = ui.ctx().pointer_hover_pos();
    let double = ui.ctx().input(|i| {
        i.pointer
            .button_double_clicked(egui::PointerButton::Primary)
    });
    for d in &view.dims {
        let (Some(a), Some(b), Some(at)) =
            (project(&d.ends[0]), project(&d.ends[1]), project(&d.at))
        else {
            continue;
        };
        let line = egui::Stroke::new(1.0, theme.text_dim);
        if d.kind != DimKind::Angle {
            painter.line_segment([a, b], line);
        }
        let r = painter.text(
            at,
            egui::Align2::CENTER_CENTER,
            &d.text,
            egui::FontId::monospace(11.0),
            theme.text,
        );
        painter.rect_filled(r.expand(2.0), 3.0, theme.sunken.gamma_multiply(0.85));
        painter.text(
            at,
            egui::Align2::CENTER_CENTER,
            &d.text,
            egui::FontId::monospace(11.0),
            theme.text,
        );
        // A double click on the number opens it for typing.
        if double && pointer.is_some_and(|p| r.expand(4.0).contains(p) && inner.contains(p)) {
            state.sketch_dim_edit = Some((d.index, format!("{:.3}", d.shown)));
        }
    }
    // What the tool would draw, and where the pointer has caught.
    for line in &view.preview {
        let pts: Vec<egui::Pos2> = line.iter().filter_map(project).collect();
        if pts.len() >= 2 {
            painter.add(egui::Shape::dashed_line(
                &pts,
                egui::Stroke::new(1.5, theme.accent),
                6.0,
                4.0,
            ));
        }
    }
    // The piece a trim would take out, struck through in red.
    let red = egui::Color32::from_rgb(0xe8, 0x4a, 0x4a);
    for [a, b] in &view.cut {
        if let (Some(p), Some(q)) = (project(a), project(b)) {
            painter.line_segment([p, q], egui::Stroke::new(3.0, red));
        }
    }
    if let Some(p) = view.snap.as_ref().and_then(project) {
        painter.circle_stroke(p, 6.0, egui::Stroke::new(1.5, theme.accent));
    }
    // What the tool asks for, by the pointer's corner of the view.
    let prompt = if view.typed.is_empty() {
        view.prompt.clone()
    } else {
        // i18n-exempt: a number being typed, and its unit
        format!("{} {}", view.typed, view.unit.1)
    };
    if !prompt.is_empty() {
        let at = inner.left_bottom() + egui::vec2(12.0, -14.0);
        let r = painter.text(
            at,
            egui::Align2::LEFT_BOTTOM,
            &prompt,
            egui::FontId::proportional(12.0),
            theme.text,
        );
        let _ = r;
    }

    // The dimension being typed into.
    // The drag-box, as the viewport's own box select draws it.
    if let Some([a, b]) = view.band {
        let rect = egui::Rect::from_two_pos(
            egui::pos2(a.x as f32, a.y as f32),
            egui::pos2(b.x as f32, b.y as f32),
        );
        painter.rect_filled(rect, 0.0, theme.accent.gamma_multiply(0.08));
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(1.0, theme.accent),
            egui::StrokeKind::Inside,
        );
    }
    draw_tool_badge(ui, state, &view, inner);
    // A drawing tool's pointer is a cross, so a click is plainly a point.
    if view.drawing
        && ui
            .ctx()
            .pointer_hover_pos()
            .is_some_and(|p| inner.contains(p))
    {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }

    draw_dimension_edit(ui, state, &view, projector);
    // The extrude or revolve dialog, once a region is picked.
    if matches!(view.tool, "extrude" | "revolve") && !view.seeds.is_empty() {
        draw_solid_dialog(ui, state, &view, inner);
    }
    if view.tool == "pattern" {
        draw_pattern_dialog(ui, state, &view, inner);
    }
}

/// The tool in hand, at the top of the view where the eye is: a drawing
/// tool filled in the accent, selecting plain, and how to get back to
/// selecting.
fn draw_tool_badge(ui: &Ui, state: &ShellState, view: &SketchView, inner: egui::Rect) {
    let theme = kreate_ui::theme::from_ui(ui).over_viewport();
    let selecting = view.tool == "select";
    let name = tool_label(state, view.tool);
    let text = if selecting {
        name
    } else {
        format!("{name}   ·   {}", state.catalog.get("sketch-badge-back"))
    };
    let ink = if selecting {
        theme.text
    } else {
        egui::Color32::WHITE
    };
    let at = egui::pos2(inner.center().x, inner.top() + 14.0);
    let galley = ui
        .painter()
        .layout_no_wrap(text, egui::FontId::proportional(13.0), ink);
    let rect = egui::Align2::CENTER_TOP
        .anchor_size(at, galley.size())
        .expand2(egui::vec2(12.0, 5.0));
    let painter = ui.painter();
    if selecting {
        painter.rect_filled(rect, 6.0, theme.sunken.gamma_multiply(0.85));
        painter.rect_stroke(
            rect,
            6.0,
            egui::Stroke::new(1.0, theme.border_strong),
            egui::StrokeKind::Outside,
        );
    } else {
        painter.rect_filled(rect, 6.0, theme.accent);
    }
    painter.galley(rect.min + egui::vec2(12.0, 5.0), galley, ink);
}

/// The sketch pattern's settings, beside the view: rows and columns or
/// round a centre, how many and how far apart. The copies show in the view
/// as they would be made.
fn draw_pattern_dialog(ui: &Ui, state: &mut ShellState, view: &SketchView, inner: egui::Rect) {
    let t = |key: &str| state.catalog.get(key);
    let title = tool_label(state, "pattern").to_uppercase();
    let (ok, cancel) = (t("dialog-ok"), t("dialog-cancel"));
    let kinds = [t("pattern-rectangular"), t("pattern-circular")];
    let count = t("pattern-count");
    let spacing = t("pattern-spacing");
    let count_y = t("pattern-count-y");
    let spacing_y = t("pattern-spacing-y");
    let angle = t("pattern-angle");
    let centre = t("pattern-centre");
    let centre_pick = t("pattern-centre-pick");
    let centre_picking = t("pattern-centre-picking");
    let nothing = t("pattern-nothing");
    let picked = !view.picked_points.is_empty() || !view.picked_lines.is_empty();
    let mut d = state.pattern_dialog.clone();
    let mut done: Option<bool> = None;
    let shown = egui::Area::new(egui::Id::new("pattern-dialog"))
        .order(egui::Order::Middle)
        .pivot(egui::Align2::RIGHT_TOP)
        .fixed_pos(inner.right_top() + egui::vec2(-12.0, 132.0))
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(250.0);
                ui.label(egui::RichText::new(title).strong().small());
                ui.separator();
                if !picked {
                    ui.label(egui::RichText::new(&nothing).small());
                    ui.separator();
                }
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut d.circular, false, &kinds[0]);
                    ui.selectable_value(&mut d.circular, true, &kinds[1]);
                });
                egui::Grid::new("pattern-dialog-grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        let unit = format!(" {}", view.unit.1);
                        ui.label(&count);
                        ui.add(egui::DragValue::new(&mut d.count).range(1..=500).speed(0.1));
                        ui.end_row();
                        if d.circular {
                            ui.label(&angle);
                            ui.add(
                                egui::DragValue::new(&mut d.angle)
                                    .range(-360.0..=360.0)
                                    .speed(1.0)
                                    .suffix("°"),
                            );
                            ui.end_row();
                            ui.label(&centre);
                            let pick = if d.picking_centre {
                                &centre_picking
                            } else {
                                &centre_pick
                            };
                            if ui.selectable_label(d.picking_centre, pick).clicked() {
                                d.picking_centre = !d.picking_centre;
                            }
                            ui.end_row();
                        } else {
                            ui.label(&spacing);
                            ui.add(
                                egui::DragValue::new(&mut d.spacing)
                                    .speed(0.1)
                                    .suffix(unit.clone()),
                            );
                            ui.end_row();
                            ui.label(&count_y);
                            ui.add(
                                egui::DragValue::new(&mut d.count2)
                                    .range(1..=500)
                                    .speed(0.1),
                            );
                            ui.end_row();
                            ui.label(&spacing_y);
                            ui.add(
                                egui::DragValue::new(&mut d.spacing2)
                                    .speed(0.1)
                                    .suffix(unit),
                            );
                            ui.end_row();
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button(&cancel).clicked() {
                        done = Some(false);
                    }
                    if ui.add_enabled(picked, egui::Button::new(&ok)).clicked() {
                        done = Some(true);
                    }
                });
            });
        });
    state.viewport_blockers.push(shown.response.rect);
    state.pattern_dialog = d.clone();
    let mut back = PropertyMap::new();
    back.set("tool", kreate_reflect::PropertyValue::Text("select".into()));
    match done {
        Some(true) => {
            use kreate_reflect::PropertyValue as V;
            let params = d.params(view.unit.0);
            let mut p = PropertyMap::new();
            p.set("points", V::Ints(view.picked_points.clone()));
            p.set("lines", V::Ints(view.picked_lines.clone()));
            p.set("kind", V::Enum(params.kind));
            p.set("count", V::Int(params.count as i64));
            p.set("spacing", V::Float(params.spacing));
            p.set("count2", V::Int(params.count2 as i64));
            p.set("spacing2", V::Float(params.spacing2));
            p.set("centre", V::Vec3(params.centre));
            p.set("angle", V::Float(params.angle));
            state.queue.run("sketch.pattern", p);
            state.queue.dispatch("app.sketch_tool", back);
        }
        Some(false) => state.queue.dispatch("app.sketch_tool", back),
        None => {}
    }
}

fn draw_dimension_edit(
    ui: &Ui,
    state: &mut ShellState,
    view: &SketchView,
    projector: &kreate_ui::object_glyph::Projector,
) {
    let Some((index, mut text)) = state.sketch_dim_edit.take() else {
        return;
    };
    let Some(dim) = view.dims.iter().find(|d| d.index == index) else {
        return;
    };
    let Some(at) = projector.project(dim.at) else {
        return;
    };
    let mut keep = true;
    egui::Area::new(egui::Id::new("sketch-dim-edit"))
        .order(egui::Order::Foreground)
        .fixed_pos(at + egui::vec2(-40.0, -12.0))
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                let field = ui.add(egui::TextEdit::singleline(&mut text).desired_width(80.0));
                field.request_focus();
                let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
                if enter {
                    if let Ok(v) = text.trim().parse::<f64>() {
                        if v.is_finite() && v != 0.0 {
                            let mut p = PropertyMap::new();
                            p.set("index", kreate_reflect::PropertyValue::Int(index as i64));
                            p.set(
                                "value",
                                kreate_reflect::PropertyValue::Float(dim.value_of(v, view.unit.0)),
                            );
                            state.queue.run("sketch.constraint_set", p);
                        }
                    }
                    keep = false;
                }
                if escape {
                    keep = false;
                }
            });
        });
    if keep {
        state.sketch_dim_edit = Some((index, text));
    }
}

fn draw_solid_dialog(ui: &Ui, state: &mut ShellState, view: &SketchView, inner: egui::Rect) {
    let revolve = view.tool == "revolve";
    let title = tool_label(state, view.tool).to_uppercase();
    let (ok, cancel) = (
        state.catalog.get("dialog-ok"),
        state.catalog.get("dialog-cancel"),
    );
    let ops = [
        state.catalog.get("feature-new-body"),
        state.catalog.get("feature-join"),
        state.catalog.get("feature-cut"),
    ];
    let mut d = state.solid_dialog.clone();
    let mut done: Option<bool> = None;
    let shown = egui::Area::new(egui::Id::new("solid-dialog"))
        .order(egui::Order::Middle)
        .pivot(egui::Align2::RIGHT_TOP)
        .fixed_pos(inner.right_top() + egui::vec2(-12.0, 132.0))
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(240.0);
                ui.label(egui::RichText::new(title).strong().small());
                ui.separator();
                egui::Grid::new("solid-dialog-grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        if revolve {
                            ui.label(state.catalog.get("feature-angle"));
                            ui.add(egui::DragValue::new(&mut d.angle).speed(1.0).suffix("°"));
                            ui.end_row();
                            ui.label(state.catalog.get("feature-axis"));
                            ui.label(if view.axis.is_some() {
                                state.catalog.get("feature-axis-set")
                            } else {
                                state.catalog.get("feature-axis-pick")
                            });
                            ui.end_row();
                        } else {
                            ui.label(state.catalog.get("feature-distance"));
                            ui.add(
                                egui::DragValue::new(&mut d.distance)
                                    .speed(0.1)
                                    .suffix(format!(" {}", view.unit.1)),
                            );
                            ui.end_row();
                            ui.label(state.catalog.get("feature-symmetric"));
                            ui.checkbox(&mut d.symmetric, "");
                            ui.end_row();
                        }
                        ui.label(state.catalog.get("feature-operation"));
                        egui::ComboBox::from_id_salt("solid-op")
                            .selected_text(ops[d.operation.min(2) as usize].clone())
                            .show_ui(ui, |ui| {
                                for (i, name) in ops.iter().enumerate() {
                                    if ui
                                        .selectable_value(&mut d.operation, i as u32, name.as_str())
                                        .clicked()
                                    {
                                        d.chosen = true;
                                    }
                                }
                            });
                        ui.end_row();
                        if d.operation != 0 && !d.target.is_empty() {
                            ui.label(state.catalog.get("feature-target"));
                            ui.label(&d.target);
                            ui.end_row();
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button(&cancel).clicked() {
                        done = Some(false);
                    }
                    let ready = !revolve || view.axis.is_some();
                    if ui.add_enabled(ready, egui::Button::new(&ok)).clicked() {
                        done = Some(true);
                    }
                });
            });
        });
    state.viewport_blockers.push(shown.response.rect);
    state.solid_dialog = d.clone();
    match done {
        Some(true) => {
            use kreate_reflect::PropertyValue as V;
            let mut p = PropertyMap::new();
            p.set(
                "seeds",
                V::Floats(view.seeds.iter().flat_map(|s| [s[0], s[1]]).collect()),
            );
            p.set("operation", V::Enum(d.operation));
            if d.operation != 0 {
                p.set("target", V::Text(d.target.clone()));
            }
            if revolve {
                let axis = view.axis.unwrap_or([0, 0]);
                p.set("axis", V::Ints(axis.to_vec()));
                p.set("angle", V::Float(d.angle.to_radians()));
                state.queue.run("feature.revolve", p);
            } else {
                p.set("distance", V::Float(d.distance * view.unit.0));
                p.set("symmetric", V::Bool(d.symmetric));
                state.queue.run("feature.extrude", p);
            }
            // The sketch is finished with, as Fusion leaves it.
            state.queue.push(Command::SetMode("object".into()));
        }
        Some(false) => {
            let mut p = PropertyMap::new();
            p.set("tool", kreate_reflect::PropertyValue::Text("select".into()));
            state.queue.dispatch("app.sketch_tool", p);
        }
        None => {}
    }
}

/// The sketches and features, in the order they were made: the timeline.
/// Double-click a sketch to draw in it again, a feature to change it.
fn draw_feature_list(ui: &mut Ui, state: &mut ShellState) {
    if state.features.is_empty() {
        return;
    }
    let theme = kreate_ui::theme::from_ui(ui);
    let rows = state.features.clone();
    egui::CollapsingHeader::new(state.catalog.get("feature-list"))
        .id_salt("feature-list")
        .default_open(true)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("feature-list-scroll")
                .max_height(160.0)
                .show(ui, |ui| {
                    for row in &rows {
                        let icon = if row.is_sketch {
                            "✎"
                        } else if row.revolve {
                            "⟳"
                        } else {
                            "⬆"
                        };
                        let colour = if row.cut { theme.info } else { theme.text };
                        // i18n-exempt: a symbol and the user's own names
                        let text = if row.is_sketch {
                            format!("{icon}  {}", row.label)
                        } else {
                            format!("{icon}  {}  · {}", row.label, row.body)
                        };
                        let r = ui.selectable_label(false, egui::RichText::new(text).color(colour));
                        if r.double_clicked() {
                            if row.is_sketch {
                                state.queue.push(Command::EditSketch(row.label.clone()));
                            } else {
                                let unit = state.length_unit;
                                state.feature_edit = Some(FeatureEdit {
                                    amount: if row.revolve {
                                        row.amount.to_degrees()
                                    } else {
                                        row.amount / unit.max(1e-12)
                                    },
                                    row: row.clone(),
                                });
                            }
                        }
                    }
                });
        });
    ui.separator();
    draw_feature_edit(ui, state);
}

fn draw_feature_edit(ui: &Ui, state: &mut ShellState) {
    let Some(mut edit) = state.feature_edit.take() else {
        return;
    };
    let mut keep = true;
    let (ok, cancel) = (
        state.catalog.get("dialog-ok"),
        state.catalog.get("dialog-cancel"),
    );
    let title = edit.row.label.clone();
    let unit = state.length_unit;
    egui::Window::new(title)
        .id(egui::Id::new("feature-edit"))
        .collapsible(false)
        .resizable(false)
        .show(ui.ctx(), |ui| {
            egui::Grid::new("feature-edit-grid")
                .num_columns(2)
                .show(ui, |ui| {
                    if edit.row.revolve {
                        ui.label(state.catalog.get("feature-angle"));
                        ui.add(
                            egui::DragValue::new(&mut edit.amount)
                                .speed(1.0)
                                .suffix("°"),
                        );
                    } else {
                        ui.label(state.catalog.get("feature-distance"));
                        ui.add(egui::DragValue::new(&mut edit.amount).speed(0.1));
                        ui.end_row();
                        ui.label(state.catalog.get("feature-symmetric"));
                        ui.checkbox(&mut edit.row.symmetric, "");
                    }
                    ui.end_row();
                    ui.label(state.catalog.get("feature-cut"));
                    ui.checkbox(&mut edit.row.cut, "");
                    ui.end_row();
                });
            ui.horizontal(|ui| {
                if ui.button(&cancel).clicked() {
                    keep = false;
                }
                if ui.button(&ok).clicked() {
                    use kreate_reflect::PropertyValue as V;
                    let mut p = PropertyMap::new();
                    p.set("body", V::Text(edit.row.body.clone()));
                    p.set("index", V::Int(edit.row.index as i64));
                    let amount = if edit.row.revolve {
                        edit.amount.to_radians()
                    } else {
                        edit.amount * unit
                    };
                    p.set("amount", V::Float(amount));
                    p.set("symmetric", V::Bool(edit.row.symmetric));
                    p.set("operation", V::Enum(u32::from(edit.row.cut)));
                    state.queue.run("feature.set", p);
                    keep = false;
                }
            });
        });
    if keep {
        state.feature_edit = Some(edit);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn tab_never_hands_the_keyboard_to_a_button() {
        let ctx = egui::Context::default();
        let frame = |ctx: &egui::Context, tab: bool| {
            let mut input = egui::RawInput::default();
            if tab {
                input.events.push(egui::Event::Key {
                    key: egui::Key::Tab,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                });
            }
            super::keep_tab_for_the_keymap(ctx, &mut input);
            let _ = ctx.run_ui(input, |ui| {
                let _ = ui.button("one");
                let _ = ui.button("two");
            });
        };
        frame(&ctx, false);
        for _ in 0..3 {
            frame(&ctx, true);
            frame(&ctx, false);
            assert!(
                !ctx.egui_wants_keyboard_input(),
                "a button took the keyboard, so the next Tab is lost"
            );
        }
    }

    #[test]
    fn simulations_that_cannot_go_together_say_so_both_ways() {
        for a in Sim::ALL {
            assert!(!a.clashes_with(a), "{a:?} with itself");
            for b in Sim::ALL {
                assert_eq!(a.clashes_with(b), b.clashes_with(a), "{a:?} {b:?}");
            }
        }
        assert!(Sim::Cloth.clashes_with(Sim::RigidBody));
        assert!(Sim::FluidDomain.clashes_with(Sim::Collision));
        assert!(
            !Sim::Collision.clashes_with(Sim::Cloth),
            "cloth falls on colliders"
        );
        assert!(!Sim::Collision.bakes() && Sim::RigidBody.bakes());
    }

    #[test]
    fn a_button_is_lit_by_what_the_object_holds() {
        let mut p = kreate_scene::physics::ObjectPhysics::default();
        assert!(Sim::ALL.iter().all(|s| !s.on(&p)));
        p.soft = Some(kreate_scene::physics::SoftBody {
            kind: kreate_scene::physics::SoftKind::Soft,
            ..Default::default()
        });
        assert!(Sim::SoftBody.on(&p) && !Sim::Cloth.on(&p));
    }

    use super::*;
    use kreate_scene::document::Object;

    #[test]
    fn discarding_all_recoveries_names_every_one() {
        // The affordance that makes a pile of them manageable: a crashed run
        // leaves one autosave per session, and forty-two of them is a normal
        // outcome, not a pathological one. A user who wants none should not
        // have to click forty-two times to say so.
        let mut state = ShellState {
            recoverable: (0..5)
                .map(|i| RecoveryOffer {
                    path: std::path::PathBuf::from(format!("/tmp/session-{i}/a.kreate")),
                    name: "a.kreate".into(),
                    age: String::new(),
                })
                .collect(),
            ..Default::default()
        };

        discard_all_recoveries(&mut state);
        let queued: Vec<std::path::PathBuf> = state
            .queue
            .drain()
            .into_iter()
            .filter_map(|c| match c {
                Command::DiscardRecovery { path } => Some(path),
                _ => None,
            })
            .collect();
        assert_eq!(queued.len(), 5, "not every offer was named");
        // Each path exactly once: a duplicate would ask the application to
        // forget something it has already forgotten.
        let mut unique = queued.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), queued.len(), "a path was queued twice");
    }

    #[test]
    fn discarding_all_of_nothing_queues_nothing() {
        let mut state = ShellState::default();
        discard_all_recoveries(&mut state);
        assert!(state.queue.drain().is_empty());
    }

    /// The default screen and the id of its viewport area.
    fn screen_state() -> (ShellState, kreate_ui::AreaId) {
        let state = ShellState::default();
        let viewport = state
            .screen()
            .areas()
            .into_iter()
            .find(|(_, kind)| *kind == kreate_ui::EditorId::VIEWPORT)
            .expect("the default layout has a viewport")
            .0;
        (state, viewport)
    }

    #[test]
    fn a_queued_split_is_applied_after_the_frame() {
        // Layout changes are queued rather than applied where the button is,
        // because the rest of the frame is drawing against a layout derived
        // from the tree. Applying immediately would leave every area after
        // this one placed by a layout that no longer exists.
        let (mut state, viewport) = screen_state();
        let before = state.screen().area_count();

        state.split_area(viewport, kreate_ui::Axis::Vertical);
        assert_eq!(
            state.screen().area_count(),
            before,
            "the queue acted on its own"
        );

        state.apply_area_edits();
        assert_eq!(state.screen().area_count(), before + 1);
        assert!(state.area_edits.is_empty(), "the queue was not drained");
    }

    #[test]
    fn a_split_area_shows_what_it_was_showing() {
        // Splitting is "I want two of these here". Guessing a different editor
        // for the new half would be wrong more often than right, and the user
        // has the header dropdown for when it is not.
        let (mut state, viewport) = screen_state();
        let outliner = state
            .screen()
            .areas()
            .into_iter()
            .find(|(_, kind)| *kind == kreate_ui::EditorId::OUTLINER)
            .expect("the default layout has an outliner")
            .0;

        for area in [viewport, outliner] {
            let was = state.screen().editor(area).expect("the area is there");
            let known: Vec<_> = state.screen().areas();
            state.split_area(area, kreate_ui::Axis::Horizontal);
            state.apply_area_edits();

            let new_area = state
                .screen()
                .areas()
                .into_iter()
                .find(|(id, _)| !known.iter().any(|(k, _)| k == id))
                .expect("a new area appeared");
            assert_eq!(new_area.1, was, "the new half shows the wrong editor");
        }
    }

    #[test]
    fn closing_an_area_forgets_the_state_that_belonged_to_it() {
        // Otherwise the view state of areas that no longer exist accumulates
        // for the life of the session, and a new area that happened to reuse
        // an id would inherit somebody else's zoom. Ids are not reused, so the
        // second half of that cannot happen — the leak can.
        let (mut state, viewport) = screen_state();
        state.split_area(viewport, kreate_ui::Axis::Horizontal);
        state.apply_area_edits();
        let extra = state
            .screen()
            .areas()
            .into_iter()
            .map(|(id, _)| id)
            .find(|id| *id != viewport)
            .expect("more than one area");

        state.uv_views.insert(extra, kreate_ui::UvState::default());
        assert!(state.uv_views.contains_key(&extra));

        state.close_area(extra);
        state.apply_area_edits();
        assert!(
            !state.uv_views.contains_key(&extra),
            "the closed area's view state was left behind"
        );
    }

    #[test]
    fn a_close_that_the_layout_refuses_leaves_the_state_alone() {
        // The last area cannot be closed, and the state that belongs to it must
        // not be discarded on the strength of a request that did not happen.
        let mut state = ShellState {
            workspaces: kreate_ui::Workspaces::single(
                "Only",
                kreate_ui::Screen::single(kreate_ui::EditorId::UV_EDITOR),
            ),
            ..Default::default()
        };
        let only = state.screen().areas()[0].0;
        state.uv_views.insert(only, kreate_ui::UvState::default());

        state.close_area(only);
        state.apply_area_edits();
        assert_eq!(state.screen().area_count(), 1);
        assert!(
            state.uv_views.contains_key(&only),
            "the view state was discarded although the area survived"
        );
    }

    #[test]
    fn a_queued_resize_moves_the_boundary() {
        let (mut state, viewport) = screen_state();
        let space = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 600.0));
        // Chosen by axis, not by index. Handles come out innermost-first, so
        // `handles[0]` in the default layout is the boundary between the
        // outliner and the properties below it — which this test then resized
        // and reported no change to the viewport, correctly.
        let split = state
            .screen()
            .layout(space)
            .handles
            .iter()
            .find(|h| h.axis == kreate_ui::Axis::Vertical)
            .expect("the default layout has a vertical boundary")
            .id;
        let before = state
            .screen()
            .layout(space)
            .areas
            .iter()
            .find(|a| a.id == viewport)
            .expect("viewport")
            .rect
            .width();

        state
            .area_edits
            .push(AreaEdit::Resize(split, 0.4, space.width()));
        state.apply_area_edits();

        let after = state
            .screen()
            .layout(space)
            .areas
            .iter()
            .find(|a| a.id == viewport)
            .expect("viewport")
            .rect
            .width();
        assert!(
            after < before - 100.0,
            "the viewport went from {before} to {after}"
        );
    }

    #[test]
    fn the_default_layout_has_a_viewport_and_no_duplicates() {
        // What a new session opens onto. The viewport is the one area that has
        // to be there — the application sizes its render target from it, and a
        // layout without one would start up showing nothing.
        let state = ShellState::default();
        let areas = state.screen().areas();
        assert!(state.has_editor(kreate_ui::EditorId::VIEWPORT));
        assert_eq!(areas.len(), 4, "viewport, timeline, outliner, properties");
        assert!(
            state.has_editor(kreate_ui::EditorId::TIMELINE),
            "Layout keeps its clock in reach"
        );

        let mut ids: Vec<_> = areas.iter().map(|(id, _)| *id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), areas.len(), "an id was handed out twice");

        assert!(!state.has_editor(kreate_ui::EditorId::UV_EDITOR));
        assert!(state.uv_at(egui::Pos2::ZERO).is_none(), "no UV editor open");
    }

    #[test]
    fn a_layout_change_asks_to_be_saved_once_it_has_settled() {
        // The file is written when the layout changes, and a boundary drag
        // changes it sixty times a second. So the flag is set by any applied
        // edit and the *question* also asks whether a drag is in progress:
        // writing on every frame of a drag is a file the disk is busy with
        // instead of the frame.
        let (mut state, viewport) = screen_state();
        assert!(!state.layout_needs_saving(), "nothing has changed yet");

        state.split_area(viewport, kreate_ui::Axis::Vertical);
        assert!(
            !state.layout_needs_saving(),
            "the queue asked to be saved before it was applied"
        );
        state.apply_area_edits();
        assert!(state.layout_needs_saving(), "the change was not noticed");

        state.layout_saved();
        assert!(!state.layout_needs_saving(), "saving did not settle it");
    }

    #[test]
    fn a_frame_that_changed_no_layout_asks_for_no_save() {
        // Otherwise the file is rewritten sixty times a second for ever, which
        // is a busy disk and a backup tool reporting a change every minute.
        let (mut state, _) = screen_state();
        for _ in 0..3 {
            state.apply_area_edits();
            assert!(!state.layout_needs_saving());
        }
    }

    #[test]
    fn a_boundary_still_being_dragged_is_not_worth_a_file() {
        // Only the settled answer is. Checked through the same door the drag
        // uses, so a resize queued mid-drag sets the flag and the question still
        // says no.
        let (mut state, viewport) = screen_state();
        let space = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 600.0));
        let split = state
            .screen()
            .layout(space)
            .handles
            .iter()
            .find(|h| h.axis == kreate_ui::Axis::Vertical)
            .expect("a vertical boundary")
            .id;
        let _ = viewport;

        state.dragging = Some(split);
        state.area_edits.push(AreaEdit::Resize(split, 0.4, 1000.0));
        state.apply_area_edits();
        assert!(
            !state.layout_needs_saving(),
            "a file was asked for in the middle of a drag"
        );

        state.dragging = None;
        assert!(
            state.layout_needs_saving(),
            "the settled layout was never asked for"
        );
    }

    #[test]
    fn the_sidebar_toggles_for_every_viewport_at_once() {
        // Pressed while looking at the viewport, so the answer must not depend
        // on where the pointer happened to rest. With one viewport — the usual
        // case — the two readings are the same thing; with two they are not, and
        // one open and one closed is a state neither of them asked for.
        let mut state = ShellState {
            workspaces: kreate_ui::Workspaces::single(
                "Two",
                kreate_ui::Screen::single(kreate_ui::EditorId::VIEWPORT),
            ),
            ..Default::default()
        };
        let first = state.screen().areas()[0].0;
        let second = state
            .screen_mut()
            .split(
                first,
                kreate_ui::Axis::Vertical,
                0.5,
                kreate_ui::EditorId::VIEWPORT,
            )
            .expect("split");

        assert!(!state.sidebar_open(first));
        assert!(!state.sidebar_open(second));

        state.toggle_sidebar();
        assert!(state.sidebar_open(first));
        assert!(state.sidebar_open(second));

        state.toggle_sidebar();
        assert!(!state.sidebar_open(first));
        assert!(!state.sidebar_open(second));
    }

    #[test]
    fn a_half_open_sidebar_closes_rather_than_flipping() {
        // The state a layout change can leave behind: an area opened its sidebar
        // and was then split, so the new half has none. `N` from there should
        // settle them, and settling on *open* is the reading that matches what
        // the user last asked for.
        let mut state = ShellState {
            workspaces: kreate_ui::Workspaces::single(
                "Two",
                kreate_ui::Screen::single(kreate_ui::EditorId::VIEWPORT),
            ),
            ..Default::default()
        };
        let first = state.screen().areas()[0].0;
        state.toggle_sidebar();
        let second = state
            .screen_mut()
            .split(
                first,
                kreate_ui::Axis::Vertical,
                0.5,
                kreate_ui::EditorId::VIEWPORT,
            )
            .expect("split");
        assert!(state.sidebar_open(first));
        assert!(!state.sidebar_open(second), "the new half starts closed");

        state.toggle_sidebar();
        assert!(state.sidebar_open(first));
        assert!(
            state.sidebar_open(second),
            "the odd one out did not join in"
        );
    }

    #[test]
    fn a_sidebar_is_not_offered_where_there_is_no_viewport() {
        // Otherwise `N` in a UV-editing workspace would set a flag nothing
        // reads, and pressing it again would appear to do nothing twice.
        let mut state = ShellState {
            workspaces: kreate_ui::Workspaces::single(
                "None",
                kreate_ui::Screen::single(kreate_ui::EditorId::UV_EDITOR),
            ),
            ..Default::default()
        };
        let only = state.screen().areas()[0].0;
        state.toggle_sidebar();
        assert!(!state.sidebar_open(only));
    }

    /// A document whose active object is a light.
    fn a_light_document() -> (Document, kreate_scene::document::ObjectId) {
        let mut doc = Document::new();
        let id = doc.add_light(kreate_scene::Light::of_kind(
            "Lamp",
            kreate_scene::LightKind::SUN,
        ));
        let object = doc.add_object(kreate_scene::document::Object::new(
            "Lamp",
            kreate_scene::document::ObjectData::Light(id),
        ));
        doc.selection.add(object);
        (doc, object)
    }

    #[test]
    fn a_light_gets_a_data_tab_and_a_cube_does_not() {
        // What makes the tab column readable when there are ten of them rather
        // than four: a context that has nothing to put in a tab does not get
        // the tab. The same argument an operator's `poll` makes.
        let (doc, _) = a_light_document();
        assert!(available_tabs(&doc).contains(&PropertiesTab::Data));

        let mut plain = Document::new();
        let cube = plain.add_object(kreate_scene::document::Object::new(
            "Cube",
            kreate_scene::document::ObjectData::Empty,
        ));
        plain.selection.add(cube);
        assert!(!available_tabs(&plain).contains(&PropertiesTab::Data));
        // And the object's own tabs are there, because something is active.
        assert!(available_tabs(&plain).contains(&PropertiesTab::Object));
    }

    #[test]
    fn the_world_tab_is_there_with_nothing_selected() {
        // The world is the document's, not a selection's. Needing to click an
        // object before the background could be turned down would be a puzzle
        // rather than an interface, so this tab is the one that does not ask
        // what is active.
        let mut doc = Document::new();
        assert!(!available_tabs(&doc).contains(&PropertiesTab::World));
        let id = doc.add_world(kreate_scene::World::new("World"));
        doc.set_active_world(Some(id));
        assert!(available_tabs(&doc).contains(&PropertiesTab::World));
        assert_eq!(doc.selection.active, None, "nothing had to be selected");
    }

    #[test]
    fn a_mesh_gets_a_material_tab_before_it_has_a_material() {
        // **This test used to assert the opposite**, on the reasoning that a
        // mesh with an empty slot has nothing to edit and a panel over it
        // would offer fields with no datablock behind them. The fields part
        // was right and the conclusion was wrong, because it left nowhere to
        // *make* the first material: nothing else in the application creates
        // one, so a new document's cube could never get one and every material
        // control — the colour, the dials, the base colour texture — was
        // unreachable. The tab now shows a button instead of fields, which is
        // what an empty slot deserves.
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
        let object = doc.add_object(kreate_scene::document::Object::new(
            "Cube",
            kreate_scene::document::ObjectData::Mesh(mesh),
        ));
        doc.selection.add(object);
        assert!(available_tabs(&doc).contains(&PropertiesTab::Material));

        let material = doc.add_material(kreate_scene::Material::default());
        doc.set_mesh_slots(mesh, vec![Some(material)]);
        assert!(available_tabs(&doc).contains(&PropertiesTab::Material));
    }

    #[test]
    fn something_that_is_not_a_mesh_gets_no_material_tab() {
        // The half of the old rule that still holds. A lamp has no slots to
        // put a material in, so the tab would be a button that did nothing —
        // and `material.new` refuses for the same reason.
        let mut doc = Document::new();
        let light = doc.add_light(kreate_scene::Light::of_kind(
            "Bulb",
            kreate_scene::LightKind::BULB,
        ));
        let object = doc.add_object(kreate_scene::document::Object::new(
            "Lamp",
            kreate_scene::document::ObjectData::Light(light),
        ));
        doc.selection.add(object);
        assert!(!available_tabs(&doc).contains(&PropertiesTab::Material));
    }

    #[test]
    fn a_camera_gets_the_same_data_tab_a_light_does() {
        // One tab, one arm per kind of datablock. The tab exists because the
        // object *has* one, not because of what kind it is — which is what
        // keeps adding the next kind to a `match` rather than to the tab list.
        let mut doc = Document::new();
        let id = doc.add_camera(kreate_scene::Camera::new("Camera"));
        let object = doc.add_object(kreate_scene::document::Object::new(
            "Camera",
            kreate_scene::document::ObjectData::Camera(id),
        ));
        doc.selection.add(object);
        assert!(available_tabs(&doc).contains(&PropertiesTab::Data));
    }

    #[test]
    fn with_nothing_active_only_the_document_wide_tabs_remain() {
        // An empty document still has a properties editor, and it still has to
        // show something: the last operation, and the render settings, which
        // are the scene's and need no selection.
        let doc = Document::new();
        assert_eq!(
            available_tabs(&doc),
            vec![PropertiesTab::Tool, PropertiesTab::Render]
        );
    }

    #[test]
    fn a_tab_that_went_away_falls_back_rather_than_blanking() {
        // Deselecting a light while the Data tab is open. Keeping the tab and
        // drawing nothing is what reads as the editor breaking.
        let doc = Document::new();
        let tabs = available_tabs(&doc);
        assert_eq!(chosen_tab(PropertiesTab::Data, &tabs), PropertiesTab::Tool);
        // With something selected, the Object tab rather than the last
        // operation.
        let (lit, _) = a_light_document();
        let tabs = available_tabs(&lit);
        assert_eq!(
            chosen_tab(PropertiesTab::Material, &tabs),
            PropertiesTab::Object
        );
        // And one that is still there is kept, or the panel would jump back to
        // Tool every time the selection changed at all.
        let (lit, _) = a_light_document();
        let tabs = available_tabs(&lit);
        assert_eq!(chosen_tab(PropertiesTab::Data, &tabs), PropertiesTab::Data);
    }

    #[test]
    fn a_drag_sends_one_run_and_then_adjustments() {
        // The rule that keeps a slider from putting one undo step on the stack
        // per frame. The first change of a gesture runs the operator; the rest
        // adjust the step it made.
        let mut state = ShellState::default();
        for (i, value) in [1.0f64, 2.0, 3.0].into_iter().enumerate() {
            let mut p = PropertyMap::new();
            p.set("intensity", kreate_reflect::PropertyValue::Float(value));
            // `continuing` is "the pointer is still down", which is what a drag
            // is.
            send_edit(&mut state, "light.set_properties", p, i > 0);
        }
        let sent = state.queue.drain();
        assert_eq!(sent.len(), 3);
        assert!(
            matches!(&sent[0], Command::RunOperator { op_id, .. } if op_id == "light.set_properties"),
            "the first change did not start a step"
        );
        assert!(matches!(&sent[1], Command::AdjustLast { .. }));
        assert!(matches!(&sent[2], Command::AdjustLast { .. }));
    }

    #[test]
    fn letting_go_starts_a_new_step() {
        // The other half, and the one that is easy to leave out: without it the
        // second drag of the same slider goes on adjusting the first one's
        // step, and the first drag becomes impossible to undo.
        let mut state = ShellState::default();
        send_edit(
            &mut state,
            "light.set_properties",
            PropertyMap::new(),
            false,
        );
        end_gesture(&mut state, true);
        send_edit(&mut state, "light.set_properties", PropertyMap::new(), true);
        // Pointer up: the gesture is over.
        end_gesture(&mut state, false);
        send_edit(&mut state, "light.set_properties", PropertyMap::new(), true);

        let sent = state.queue.drain();
        assert!(matches!(&sent[0], Command::RunOperator { .. }));
        assert!(matches!(&sent[1], Command::AdjustLast { .. }));
        assert!(
            matches!(&sent[2], Command::RunOperator { .. }),
            "a second drag adjusted the first one's step"
        );
    }

    #[test]
    fn a_different_panel_starts_its_own_step() {
        // Dragging a light's slider and then an object's, without letting go in
        // between — which a tablet or a stray click can produce. Adjusting the
        // light's step with a transform's parameters would be an operator run
        // with somebody else's arguments.
        let mut state = ShellState::default();
        send_edit(
            &mut state,
            "light.set_properties",
            PropertyMap::new(),
            false,
        );
        send_edit(&mut state, "object.set_transform", PropertyMap::new(), true);
        let sent = state.queue.drain();
        assert!(
            matches!(&sent[1], Command::RunOperator { op_id, .. } if op_id == "object.set_transform")
        );
    }

    #[test]
    fn the_built_in_catalogs_parse() {
        // A malformed built-in resource would leave every label showing its key,
        // which is the sort of thing that ships if nothing checks it.
        let mut c = Catalog::new();
        c.add("en-US", EN_FTL).expect("English");
        let mut k = Catalog::new();
        k.add("ko-KR", KO_FTL).expect("Korean");
    }

    #[test]
    fn the_korean_catalog_is_complete() {
        // Turns "is the translation finished" into a build-time answer, so it
        // cannot rot as keys are added.
        let catalog = load_catalog("ko-KR");
        assert_eq!(
            catalog.missing_in("ko-KR"),
            Vec::<String>::new(),
            "these keys have no Korean translation"
        );
    }

    #[test]
    fn korean_is_preferred_and_english_is_the_fallback() {
        // Any key with both translations serves as the probe; the entry is
        // "Cube" now that it sits under Add ▸ Mesh, where Blender puts it.
        let ko = load_catalog("ko-KR");
        assert_eq!(ko.get("menu-add-cube"), "큐브");
        let en = load_catalog("en-US");
        assert_eq!(en.get("menu-add-cube"), "Cube");
    }

    #[test]
    fn the_status_line_interpolates_counts_in_both_languages() {
        for (locale, expected) in [
            ("en-US", "3 objects, 2 meshes, 1 selected"),
            ("ko-KR", "오브젝트 3개, 메시 2개, 선택 1개"),
        ] {
            let c = load_catalog(locale);
            assert_eq!(
                c.format(
                    "status-counts",
                    TextKind::Ui,
                    &[("objects", "3"), ("meshes", "2"), ("selected", "1")]
                ),
                expected
            );
        }
    }

    #[test]
    fn the_keymap_stands_down_while_egui_holds_the_keyboard() {
        // Without this, typing a Korean object name into a rename field fires
        // tool shortcuts, because an active IME sends `G` and `R` to both.
        let mut state = ShellState::default();
        assert!(state.keymap_enabled());

        state.egui_wants_keyboard = true;
        assert!(!state.keymap_enabled());

        state.egui_wants_keyboard = false;
        state.ime_active = true;
        assert!(
            !state.keymap_enabled(),
            "an IME composing must also silence the keymap"
        );
    }

    fn doc_with_hierarchy() -> (Document, ObjectId, ObjectId) {
        let mut d = Document::new();
        let parent = d.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        let child = d.add_object(Object::new("Helper", ObjectData::Empty));
        d.set_parent(child, Some(parent)).unwrap();
        d.selection.add(parent);
        (d, parent, child)
    }

    #[test]
    fn the_view_model_carries_hierarchy_depth_and_selection() {
        let (doc, parent, _child) = doc_with_hierarchy();
        let rows = outliner_rows(&doc);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "Cube");
        assert_eq!(rows[0].depth, 0);
        assert!(rows[0].selected && rows[0].active);
        assert_eq!(rows[1].name, "Helper");
        assert_eq!(rows[1].depth, 1, "a child is indented");
        assert!(!rows[1].selected);
        assert_eq!(rows[0].id, object_key(parent));
    }

    #[test]
    fn the_view_model_reports_mesh_statistics_and_sharing() {
        let mut doc = Document::new();
        let id = doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        doc.duplicate_linked(id).unwrap();

        let rows = outliner_rows(&doc);
        assert_eq!(rows.len(), 2);
        for row in &rows {
            assert_eq!(row.kind, RowKind::Mesh { verts: 8, faces: 6 });
            assert_eq!(row.mesh_users, 2, "a linked duplicate must be flagged");
        }
    }

    #[test]
    fn an_object_key_round_trips_through_the_ui_boundary() {
        let (doc, parent, child) = doc_with_hierarchy();
        assert_eq!(object_from_key(&doc, object_key(parent)), Some(parent));
        assert_eq!(object_from_key(&doc, object_key(child)), Some(child));
    }

    #[test]
    fn a_key_for_a_deleted_object_resolves_to_nothing() {
        // A row from the frame before a deletion must not resolve to whatever
        // reused the slot; that would silently act on the wrong object.
        let (mut doc, parent, child) = doc_with_hierarchy();
        let key = object_key(child);
        doc.remove_object(child);
        assert_eq!(object_from_key(&doc, key), None);
        assert_eq!(object_from_key(&doc, object_key(parent)), Some(parent));
    }

    #[test]
    fn an_empty_document_produces_no_rows() {
        assert!(outliner_rows(&Document::new()).is_empty());
    }

    #[test]
    fn every_key_the_shell_uses_exists_in_the_catalog() {
        // The other half of the i18n gate: the scanner proves no bare strings
        // reach a label, and this proves the keys that do are real. A typo in a
        // key would otherwise show up on screen as the key itself.
        let catalog = load_catalog("ko-KR");
        for key in [
            "object-copy",
            "object-paste",
            "menu-view",
            "menu-select",
            "menu-transform",
            "select-all",
            "select-none",
            "select-invert",
            "view-frame-all",
            "view-frame-selected",
            "view-camera",
            "view-perspective",
            "view-sidebar",
            "overlays",
            "overlay-grid",
            "overlay-wireframe",
            "transform-move",
            "transform-rotate",
            "transform-scale",
            "view-user-perspective",
            "view-user-orthographic",
            "panel-transform",
            "panel-surface",
            "panel-light",
            "outliner-hide",
            "outliner-show",
            "tool-select",
            "tool-move",
            "tool-rotate",
            "tool-scale",
            "tool-extrude",
            "tool-inset",
            "tool-bevel",
            "tool-loop-cut",
            "menu-file-new",
            "menu-file-open",
            "menu-file-save",
            "menu-file-save-as",
            "menu-file-export",
            "menu-file-import",
            "material-new",
            "material-none",
            "material-base-colour",
            "material-metallic-roughness",
            "material-normal-map",
            "material-empty-slot",
            "console-placeholder",
            "editor-shader",
            "material-uses-nodes",
            "node-use-nodes",
            "node-use-nodes-hint",
            "node-no-graph",
            "node-no-material",
            "node-add",
            "node-no-image",
            "node-open-image",
            "node-kind-color",
            "node-kind-value",
            "node-kind-image",
            "node-kind-multiply",
            "node-kind-multiply-value",
            "node-kind-separate-rgb",
            "node-kind-normal-map",
            "node-socket-base-color",
            "node-socket-metallic",
            "node-socket-roughness",
            "node-socket-normal",
            "node-socket-color",
            "node-socket-value",
            "node-socket-alpha",
            "node-socket-strength",
            "console-empty",
            "editor-console",
            "editor-claude",
            "chat-placeholder",
            "chat-empty",
            "chat-send",
            "chat-stop",
            "chat-new",
            "chat-working",
            "chat-stopped",
            "chat-checking",
            "chat-signed-out",
            "chat-sign-in",
            "chat-sign-out",
            "chat-signing-in",
            "chat-open-sign-in",
            "chat-code-hint",
            "chat-code-submit",
            "chat-cancel",
            "sculpt-coarse",
            "menu-help",
            "help-contact",
            "help-logs",
            "chat-history",
            "chat-export",
            "chat-restored",
            "chat-nothing-to-export",
            "menu-file-collab",
            "help-about",
            "contact-title",
            "contact-intro",
            "contact-kind-bug",
            "contact-kind-feature",
            "contact-kind-question",
            "contact-kind-other",
            "contact-name",
            "contact-email",
            "contact-email-hint",
            "contact-message-hint",
            "contact-details",
            "contact-send",
            "contact-close",
            "contact-sending",
            "contact-sent",
            "about-title",
            "about-version",
            "material-no-slots",
            "material-add-slot",
            "material-remove-slot",
            "world-pick-image",
            "world-clear-image",
            "material-pick-image",
            "material-clear-image",
            "recover-count",
            "recover-open",
            "recover-menu",
            "recover-none",
            "recover-hint",
            "recover-discard",
            "recover-discard-all",
            "recover-age-now",
            "recover-age-minutes",
            "recover-age-hours",
            "recover-age-days",
            "menu-add-cube",
            "menu-duplicate",
            "menu-delete",
            "menu-undo",
            "menu-redo",
            "uv-none",
            "uv-not-editing",
            "area-menu",
            "sidebar-item",
            "sidebar-nothing",
            "area-split-vertical",
            "area-split-horizontal",
            "area-close",
            "editor-viewport",
            "editor-outliner",
            "editor-properties",
            "editor-uv",
            "editor-preferences",
            "adjust-nothing",
            "status-counts",
            "outliner-kind-empty",
            "outliner-kind-mesh",
            "outliner-scene-collection",
            "menu-add-mesh",
            "menu-add-light",
            "menu-add-camera",
            "menu-add-empty",
            "menu-add-plane",
            "menu-add-grid",
            "menu-add-uv-sphere",
            "menu-add-ico-sphere",
            "menu-add-cylinder",
            "menu-add-cone",
            "menu-add-torus",
            "outliner-kind-camera",
            "outliner-kind-light",
            "outliner-shared-mesh",
            "outliner-mesh-summary",
        ] {
            assert!(catalog.has(key), "`{key}` is missing from the catalog");
        }
    }

    // --- what the chrome can actually draw -------------------------------

    /// Every tab, so that a new one cannot be added without a glyph.
    const ALL_TABS: [PropertiesTab; 8] = [
        PropertiesTab::Tool,
        PropertiesTab::Object,
        PropertiesTab::Data,
        PropertiesTab::Bone,
        PropertiesTab::Material,
        PropertiesTab::World,
        PropertiesTab::Modifiers,
        PropertiesTab::Physics,
    ];

    #[test]
    fn every_glyph_the_chrome_draws_is_in_the_bundled_font() {
        // The font is a subset and there is no system fallback, so a symbol
        // outside it is a blank box rather than a substituted icon. The tab
        // column was exactly that once; it is painted now, and these are the
        // symbols still set in type.
        let mut drawn = String::new();
        drawn.push_str(CLEAR_GLYPH);
        drawn.push_str(ADD_GLYPH);
        drawn.push_str(REMOVE_GLYPH);
        assert_eq!(
            kreate_ui::fonts::missing_glyphs(&drawn),
            Vec::<char>::new(),
            "the properties column draws a character the font has no glyph for"
        );
    }

    #[test]
    fn no_two_tabs_wear_the_same_icon() {
        // The icon *is* the button — the name is only a tooltip — so two tabs
        // sharing one are two buttons a user cannot tell apart without
        // hovering both.
        let icons: Vec<_> = ALL_TABS.iter().map(|t| t.icon(RowKind::Light)).collect();
        for (i, a) in icons.iter().enumerate() {
            assert!(
                !icons[i + 1..].contains(a),
                "two property tabs share an icon: {a:?}"
            );
        }
    }

    // --- picking a lamp by its glyph -------------------------------------

    /// A camera at +Z looking down -Z at the origin, over a 400×300 picture.
    fn projector() -> kreate_ui::object_glyph::Projector {
        let aspect = 4.0 / 3.0;
        let f = 1.0 / (std::f64::consts::FRAC_PI_4).tan();
        let projection = kreate_math::DMat4::from_cols_array(&[
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            -0.1,
            0.0,
        ]);
        kreate_ui::object_glyph::Projector {
            eye: kreate_math::DVec3::new(0.0, 0.0, 6.0),
            rotation: kreate_math::DQuat::IDENTITY,
            projection,
            rect: egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0)),
        }
    }

    fn lamp_at(doc: &mut Document, name: &str, at: kreate_math::DVec3) -> ObjectId {
        let light = doc.add_light(kreate_scene::Light::of_kind(
            name,
            kreate_scene::LightKind::BULB,
        ));
        let mut object = kreate_scene::document::Object::new(
            name,
            kreate_scene::document::ObjectData::Light(light),
        );
        object.transform.translation = at;
        doc.add_object(object)
    }

    #[test]
    fn a_click_on_a_lamps_glyph_finds_the_lamp() {
        // The complaint: lights could not be selected in the viewport. The id
        // buffer has never heard of a lamp, so the interface that draws the
        // glyph is what has to answer the click — with the same projection.
        let mut doc = Document::new();
        doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        let at = kreate_math::DVec3::new(1.5, 0.5, 0.0);
        let lamp = lamp_at(&mut doc, "Lamp", at);
        let p = projector();
        let on_screen = p.project(at).expect("in front of the camera");
        assert_eq!(glyph_under(&doc, &p, on_screen), Some(lamp));
        assert_eq!(
            glyph_under(&doc, &p, on_screen + egui::vec2(3.0, -2.0)),
            Some(lamp),
            "a near miss inside the hit radius should still be the lamp"
        );
        assert_eq!(
            glyph_under(&doc, &p, on_screen + egui::vec2(60.0, 0.0)),
            None,
            "a click well away from the glyph is not a click on it"
        );
    }

    #[test]
    fn a_hidden_lamp_cannot_be_clicked() {
        // Its glyph is not drawn, so there is nothing to have clicked — and
        // selecting an invisible object from the viewport would be a surprise.
        let mut doc = Document::new();
        let at = kreate_math::DVec3::new(0.0, 1.0, 0.0);
        let lamp = lamp_at(&mut doc, "Lamp", at);
        doc.object_mut(lamp).unwrap().visible = false;
        let p = projector();
        let on_screen = p.project(at).unwrap();
        assert_eq!(glyph_under(&doc, &p, on_screen), None);
    }

    #[test]
    fn the_nearer_of_two_overlapping_glyphs_wins() {
        let mut doc = Document::new();
        let a = lamp_at(&mut doc, "A", kreate_math::DVec3::new(0.0, 0.0, 0.0));
        let b = lamp_at(&mut doc, "B", kreate_math::DVec3::new(0.12, 0.0, 0.0));
        let p = projector();
        let at_a = p.project(kreate_math::DVec3::ZERO).unwrap();
        let at_b = p.project(kreate_math::DVec3::new(0.12, 0.0, 0.0)).unwrap();
        assert!(
            at_a.distance(at_b) < kreate_ui::object_glyph::HIT_RADIUS,
            "the fixture must overlap"
        );
        assert_eq!(glyph_under(&doc, &p, at_a), Some(a));
        assert_eq!(glyph_under(&doc, &p, at_b), Some(b));
    }

    #[test]
    fn a_camera_object_is_clickable_too() {
        let mut doc = Document::new();
        let cam = doc.add_camera(kreate_scene::Camera::new("Camera"));
        let mut object = kreate_scene::document::Object::new(
            "Camera",
            kreate_scene::document::ObjectData::Camera(cam),
        );
        let at = kreate_math::DVec3::new(-1.0, -0.5, 0.0);
        object.transform.translation = at;
        let id = doc.add_object(object);
        let p = projector();
        assert_eq!(glyph_under(&doc, &p, p.project(at).unwrap()), Some(id));
    }

    #[test]
    fn a_mesh_is_never_answered_by_the_glyph_test() {
        // Meshes are the id buffer's business; answering for one here would
        // shadow the real pick with a guess at the object's origin.
        let mut doc = Document::new();
        doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        let p = projector();
        let origin = p.project(kreate_math::DVec3::ZERO).unwrap();
        assert_eq!(glyph_under(&doc, &p, origin), None);
    }

    // --- what a click does to the selection --------------------------------

    #[test]
    fn a_plain_click_selects_the_hit_and_nothing_else() {
        assert_eq!(
            click_selection(&[1, 2], Some(2), 3, false),
            (vec![3], Some(3))
        );
        assert_eq!(
            click_selection(&[1, 2], Some(2), 1, false),
            (vec![1], Some(1))
        );
    }

    #[test]
    fn a_shift_click_adds_the_hit_and_makes_it_active() {
        // The complaint: shift-click replaced the selection in the viewport
        // while adding to it in the outliner. One rule now.
        assert_eq!(
            click_selection(&[1, 2], Some(2), 3, true),
            (vec![1, 2, 3], Some(3))
        );
        assert_eq!(click_selection(&[], None, 3, true), (vec![3], Some(3)));
    }

    #[test]
    fn a_shift_click_on_a_selected_object_takes_it_out_again() {
        // The same gesture on the same thing twice undoes itself rather than
        // doing nothing, which is Blender's rule and the one a user expects.
        assert_eq!(
            click_selection(&[1, 2, 3], Some(1), 2, true),
            (vec![1, 3], Some(1))
        );
    }

    #[test]
    fn removing_the_active_object_hands_active_to_the_last_still_selected() {
        // There should always be something for the panels to show while
        // anything is selected.
        assert_eq!(
            click_selection(&[1, 2, 3], Some(3), 3, true),
            (vec![1, 2], Some(2))
        );
        assert_eq!(
            click_selection(&[3], Some(3), 3, true),
            (Vec::<u32>::new(), None)
        );
    }

    #[test]
    fn every_catalog_string_can_be_drawn_by_the_bundled_font() {
        // The font is a subset with no fallback, so a character outside it is
        // a blank box. The tab glyphs are checked above; this checks every
        // *sentence* — the place a stray typographic character lands when a
        // label is written, and where nobody looks until a user reports a box
        // in the middle of a status message. A Fluent placeholder is stripped
        // first: `{ $count }` is not drawn.
        for (locale, ftl) in [("en", EN_FTL), ("ko-KR", KO_FTL)] {
            for line in ftl.lines() {
                let Some((key, value)) = line.split_once(" = ") else {
                    continue;
                };
                let mut text = String::new();
                let mut depth = 0;
                for ch in value.chars() {
                    match ch {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        c if depth == 0 => text.push(c),
                        _ => {}
                    }
                }
                let missing = kreate_ui::fonts::missing_glyphs(&text);
                assert!(
                    missing.is_empty(),
                    "{locale}: `{key}` uses {missing:?}, which the bundled font cannot draw"
                );
            }
        }
    }

    #[test]
    fn the_add_menus_shapes_are_the_operators_in_the_operators_order() {
        // The menu sends an index and the operator reads it against its own
        // list; a shape inserted in one and not the other would add a cone
        // when the user asked for a cylinder.
        assert_eq!(&ADD_SHAPES[..], kreate_ops::ops::object::PRIMITIVES);
    }

    #[test]
    fn every_menu_entry_has_a_sentence_to_show_on_hover() {
        // The promise `menu_item` makes: hover any entry and it says what it
        // does. Kept by walking the menu tree, in both modes, and asking the
        // catalog for each entry's sentence — plus the few entries still drawn
        // by hand (the area's own), found by reading this file for them. An
        // entry added without a description fails here rather than being the
        // one silent item in a menu that otherwise explains itself.
        let source = include_str!("shell.rs");
        let catalog = load_catalog("en");
        let mut ids: Vec<&str> = Vec::new();
        for needle in ["menu_item(ui, state, \"", "menu_item_enabled(ui, state, \""] {
            for (at, _) in source.match_indices(needle) {
                let rest = &source[at + needle.len()..];
                if let Some(end) = rest.find('"') {
                    ids.push(&rest[..end]);
                }
            }
        }
        for mode in kreate_ops::Mode::ALL {
            for (_, entry) in flatten(&searchable_menus(&catalog, mode, 0, true, true)) {
                ids.push(entry.id);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        assert!(
            ids.len() > 30,
            "the walk found only {} entries; it has lost the menus",
            ids.len()
        );
        let missing: Vec<&str> = ids
            .iter()
            .copied()
            .filter(|id| !catalog.has(&kreate_ui::command_info::description_key(id)))
            .collect();
        assert!(
            missing.is_empty(),
            "these menu entries have nothing to say on hover: {missing:?}"
        );
    }

    #[test]
    fn every_pie_a_key_opens_is_a_pie_and_says_what_its_entries_do() {
        // A binding names its pie by string, so a typo would be a key that
        // opens nothing — checked for every preset. And every entry says what
        // it does, like every menu entry.
        let catalog = load_catalog("en");
        let l = Labels::new(&catalog);
        let mut names = Vec::new();
        for preset in kreate_input::presets::ALL {
            for binding in kreate_input::presets::load(*preset).bindings() {
                if binding.action.op_id != "app.pie" {
                    continue;
                }
                match binding.action.params.get("menu") {
                    Some(kreate_reflect::PropertyValue::Text(name)) => names.push(name.clone()),
                    other => panic!("{preset:?}: a pie binding names no pie: {other:?}"),
                }
            }
        }
        names.sort();
        names.dedup();
        assert_eq!(names, vec!["mode", "shading", "snap", "view"]);
        for name in &names {
            let (_, entries) = pie_entries(
                name,
                &l,
                kreate_render::ShadingMode::Studio,
                kreate_ops::Mode::RESTING,
                kreate_ops::mode::Has::EVERYTHING,
            )
            .unwrap_or_else(|| panic!("no pie called {name}"));
            assert!(
                !entries.is_empty() && entries.len() <= kreate_ui::pie_menu::MAX_SLOTS,
                "{name}"
            );
            for (entry, _) in &entries {
                assert!(
                    catalog.has(&kreate_ui::command_info::description_key(entry.id)),
                    "{name}: `{}` has nothing to say on hover",
                    entry.id
                );
                assert!(
                    !entry.label.is_empty() && !entry.label.contains('-'),
                    "{name}: label {:?}",
                    entry.label
                );
            }
        }
        assert!(pie_entries(
            "from_the_future",
            &l,
            kreate_render::ShadingMode::Studio,
            kreate_ops::Mode::RESTING,
            kreate_ops::mode::Has::EVERYTHING
        )
        .is_none());
    }

    #[test]
    fn the_view_pie_looks_from_where_it_says() {
        // Front is where the numpad's 1 puts the eye: on the -Y side. A swapped
        // pair here is a pie that shows the back when asked for the front.
        let catalog = load_catalog("en");
        let l = Labels::new(&catalog);
        let (_, entries) = pie_entries(
            "view",
            &l,
            kreate_render::ShadingMode::Studio,
            kreate_ops::Mode::RESTING,
            kreate_ops::mode::Has::EVERYTHING,
        )
        .unwrap();
        let axis_of = |id: &str| match &entries.iter().find(|(e, _)| e.id == id).unwrap().0.run {
            Run::Command(Command::SnapToAxis(axis)) => *axis,
            other => panic!("{id} runs {other:?}"),
        };
        use kreate_render::ViewAxis::*;
        assert_eq!(axis_of("view.axis_front"), NegY);
        assert_eq!(axis_of("view.axis_back"), PosY);
        assert_eq!(axis_of("view.axis_right"), PosX);
        assert_eq!(axis_of("view.axis_top"), PosZ);
        // And the first two slots are left and right, Blender's order.
        assert_eq!(entries[0].0.id, "view.axis_left");
        assert_eq!(entries[1].0.id, "view.axis_right");
        // The camera is reachable without a numpad: from the pie and the menu.
        assert!(entries.iter().any(|(e, _)| e.id == "view.camera"));
        assert!(view_nodes(&l)
            .iter()
            .any(|n| matches!(n, MenuNode::Entry(e) if e.id == "view.camera")));
    }

    #[test]
    fn the_mode_pie_lights_the_mode_you_are_in_and_greys_what_you_cannot_enter() {
        let catalog = load_catalog("en");
        let l = Labels::new(&catalog);
        use kreate_ops::mode::Has;
        // With nothing, object mode and a greyed Edit Mode to learn from;
        // with an armature, its two modes and not the mesh's Edit Mode.
        let (_, entries) = pie_entries(
            "mode",
            &l,
            kreate_render::ShadingMode::Studio,
            kreate_ops::Mode::RESTING,
            Has::NOTHING,
        )
        .unwrap();
        let ids: Vec<&str> = entries.iter().map(|(e, _)| e.id).collect();
        assert_eq!(
            ids,
            vec![
                "mode-object",
                "mode-edit",
                "mode-weight-paint",
                "mode-sculpt",
                "mode-texture-paint",
                // Needs nothing: a sketch starts from a plane.
                "mode-sketch"
            ]
        );
        assert!(entries[0].1, "the mode you are in is not lit");
        assert!(
            !entries[1].0.enabled,
            "edit mode with nothing to edit is not greyed"
        );
        assert!(
            !entries[2].0.enabled,
            "weight paint with nothing to paint is not greyed"
        );
        let armature = Has {
            mesh: false,
            armature: true,
        };
        let (_, entries) = pie_entries(
            "mode",
            &l,
            kreate_render::ShadingMode::Studio,
            kreate_ops::Mode::RESTING,
            armature,
        )
        .unwrap();
        let ids: Vec<&str> = entries.iter().map(|(e, _)| e.id).collect();
        assert_eq!(
            ids,
            vec![
                "mode-object",
                "mode-edit-armature",
                "mode-pose",
                "mode-sketch"
            ]
        );
        for (entry, current) in &entries {
            assert!(entry.enabled);
            assert_eq!(*current, entry.id == "mode-object");
        }
    }

    #[test]
    fn the_search_offers_what_the_menus_offer_in_each_mode() {
        // One tree for both, so this is about the shape of the tree: object
        // mode searches the Object menu and not the Mesh menus, edit mode the
        // other way round, and both reach the top bar and Add.
        let catalog = load_catalog("en");
        let ids = |mode| -> Vec<&'static str> {
            flatten(&searchable_menus(&catalog, mode, 0, true, true))
                .into_iter()
                .map(|(_, e)| e.id)
                .collect()
        };
        let object = ids(kreate_ops::Mode::Object);
        let edit = ids(kreate_ops::Mode::EditMesh);
        for common in [
            "file.save",
            "edit.undo",
            "app.search",
            "object.add_primitive",
            "view.frame_all",
        ] {
            assert!(
                object.contains(&common) && edit.contains(&common),
                "{common} is missing somewhere"
            );
        }
        assert!(object.contains(&"object.delete") && !object.contains(&"mesh.bevel"));
        assert!(edit.contains(&"mesh.bevel") && !edit.contains(&"object.delete"));
        // Every shape of the Add menu, each its own entry.
        assert_eq!(
            object
                .iter()
                .filter(|id| **id == "object.add_primitive")
                .count(),
            ADD_SHAPES.len()
        );
    }

    #[test]
    fn a_searched_entry_runs_what_its_menu_entry_runs() {
        // Extrude from the search is Extrude from Face ▸ Extrude: the same
        // operator with the same parameters, because it is the same entry.
        let catalog = load_catalog("en");
        let entries = flatten(&searchable_menus(
            &catalog,
            kreate_ops::Mode::EditMesh,
            2,
            true,
            true,
        ));
        let (path, extrude) = entries
            .iter()
            .find(|(_, e)| e.id == "mesh.extrude_region")
            .expect("extrude is searchable in edit mode");
        assert!(path.ends_with(&Labels::new(&catalog).menu_face), "{path}");
        let mut state = ShellState::default();
        run_entry(&mut state, extrude);
        let commands = state.queue.drain();
        match commands.as_slice() {
            [Command::RunOperator { op_id, params }] => {
                assert_eq!(op_id, "mesh.extrude_region");
                assert_eq!(
                    params.get("offset"),
                    Some(&kreate_reflect::PropertyValue::Float(1.0))
                );
            }
            other => panic!("ran {other:?}"),
        }
        // And an entry that reads the element mode reads the one it was
        // built with.
        let (_, delete) = entries.iter().find(|(_, e)| e.id == "mesh.delete").unwrap();
        run_entry(&mut state, delete);
        match state.queue.drain().as_slice() {
            [Command::RunOperator { params, .. }] => {
                assert_eq!(
                    params.get("mode"),
                    Some(&kreate_reflect::PropertyValue::Enum(2))
                );
            }
            other => panic!("ran {other:?}"),
        }
    }
}
