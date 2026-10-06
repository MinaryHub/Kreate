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
    pub chat_busy: bool,
    /// Which assistant the chat panel talks to: an index into
    /// `kreate::claude::Provider::ALL`.
    pub chat_provider: usize,
    /// Who Claude Code is signed in as, and the code box for signing in.
    pub chat_account: kreate_ui::chat::ChatAccount,
    pub chat_code: String,
    /// Help ▸ Contact's form, kept while closed so a half-written message
    /// survives, and whether Help ▸ About is showing.
    pub contact: kreate_ui::contact::ContactForm,
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
    /// Each graph editor's view and drag, per area.
    graph_views: std::collections::BTreeMap<kreate_ui::AreaId, kreate_ui::graph_editor::GraphState>,
    /// The weight-paint brush: what it paints towards, how wide, how hard.
    pub paint: PaintSettings,
    /// The sculpt brush.
    pub sculpt: SculptSettings,
    /// Where `Ctrl+R` would cut, while it is looking for an edge: one world
    /// polyline per loop. Empty otherwise.
    pub loop_cut_preview: Vec<Vec<kreate_math::DVec3>>,
    /// The knife's path while `K` is drawing it: the points clicked, then
    /// where the next one would go. Empty otherwise.
    pub knife_preview: Vec<kreate_math::DVec3>,
    /// Circle select's brush radius in interface points, while `C` is on.
    pub circle_select: Option<f32>,
    /// What a snapping drag has caught, for a marker on it.
    pub snap_target: Option<(kreate_math::DVec3, crate::snap::Element)>,
    /// Blender's Statistics overlay: counts in the viewport's corner.
    pub show_stats: bool,
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
            if kind == kreate_ui::EditorId::SHADER {
                over(self.node_views.get(&id).and_then(|s| s.rect))
            } else if kind == kreate_ui::EditorId::DOPE_SHEET {
                over(self.sheet_views.get(&id).and_then(|s| s.rect))
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
            editors: kreate_ui::EditorRegistry::with_builtins(),
            catalog: Catalog::new(),
            material_slot: 0,
            console_input: String::new(),
            console_log: Vec::new(),
            chat_input: String::new(),
            chat_log: Vec::new(),
            chat_busy: false,
            chat_provider: 0,
            chat_account: kreate_ui::chat::ChatAccount::Checking,
            chat_code: String::new(),
            contact: kreate_ui::contact::ContactForm::new(),
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
            capturing: None,
            palette: Default::default(),
            pie: Default::default(),
            keymap_edits: Vec::new(),
            uv_views: std::collections::BTreeMap::new(),
            node_views: std::collections::BTreeMap::new(),
            timeline_views: std::collections::BTreeMap::new(),
            sheet_views: std::collections::BTreeMap::new(),
            graph_views: std::collections::BTreeMap::new(),
            driver_views: std::collections::BTreeMap::new(),
            paint: PaintSettings::default(),
            sculpt: SculptSettings::default(),
            loop_cut_preview: Vec::new(),
            knife_preview: Vec::new(),
            circle_select: None,
            snap_target: None,
            rename: None,
            show_stats: false,
            history_steps: Vec::new(),
            history_at: 0,
            pivot: 0,
            tools: ToolSettings::default(),
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
    }
}

/// Opaque key for an object, so the UI can round-trip a handle it cannot read.
pub fn object_key(id: ObjectId) -> u32 {
    use kreate_mesh::arena::ArenaIndex as _;
    id.index()
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
}

/// The viewport's on-off settings, as the Overlays menu shows them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overlays {
    pub grid: bool,
    pub wireframe: bool,
    pub face_orientation: bool,
    pub statistics: bool,
    pub ambient_occlusion: bool,
    pub xray: bool,
    pub outline: bool,
    pub bloom: bool,
    pub reflections: bool,
    pub contact_shadows: bool,
    pub path_trace: bool,
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

/// The sculpt brush's settings, the application's as the paint brush's are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SculptSettings {
    /// Draw, smooth, grab, inflate.
    pub brush: u32,
    /// Its radius on screen, in points.
    pub radius: f32,
    pub strength: f32,
    pub mirror_x: bool,
}

impl Default for SculptSettings {
    fn default() -> Self {
        Self {
            brush: 0,
            radius: 50.0,
            strength: 0.5,
            mirror_x: false,
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
    file_export: String,
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
    chat_empty: String,
    chat_send: String,
    chat_stop: String,
    chat_new: String,
    chat_working: String,
    chat_checking: String,
    chat_signed_out: String,
    chat_sign_in: String,
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
    sculpt_draw: String,
    sculpt_smooth: String,
    sculpt_grab: String,
    sculpt_inflate: String,
    pose_ik_add: String,
    menu_weights: String,
    group_add: String,
    weights_normalize: String,
    menu_parent: String,
    menu_boolean: String,
    boolean_ops: [String; 3],
    parent_object: String,
    parent_clear: String,
    rename: String,
    bind_automatic: String,
    bind_empty: String,
    unbind: String,
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
    shading_studio: String,
    shading_flat: String,
    shading_wireframe: String,
    shading_rendered: String,
    shading_hint: String,
    exposure: String,
    exposure_metered: String,
    exposure_manual: String,
    menu_file: String,
    menu_edit: String,
    menu_help: String,
    help_contact: String,
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
    view_perspective: String,
    view_sidebar: String,
    overlays: String,
    overlay_grid: String,
    overlay_wireframe: String,
    overlay_face_orientation: String,
    overlay_statistics: String,
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
    pivot_label: String,
    undo_history: String,
    undo_original: String,
    stats_objects: String,
    stats_vertices: String,
    stats_edges: String,
    stats_faces: String,
    stats_triangles: String,
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
            file_export: t("menu-file-export"),
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
            chat_empty: t("chat-empty"),
            chat_send: t("chat-send"),
            chat_stop: t("chat-stop"),
            chat_new: t("chat-new"),
            chat_working: t("chat-working"),
            chat_checking: t("chat-checking"),
            chat_signed_out: t("chat-signed-out"),
            chat_sign_in: t("chat-sign-in"),
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
            sculpt_draw: t("sculpt-draw"),
            sculpt_smooth: t("sculpt-smooth"),
            sculpt_grab: t("sculpt-grab"),
            sculpt_inflate: t("sculpt-inflate"),
            pose_ik_add: t("op-pose-ik-add"),
            menu_weights: t("menu-weights"),
            group_add: t("op-object-vertex-group-add"),
            weights_normalize: t("op-paint-weights-normalize"),
            menu_parent: t("menu-parent"),
            menu_boolean: t("menu-boolean"),
            boolean_ops: [0, 1, 2].map(|n| t(&format!("op-object-boolean-operation-{n}"))),
            parent_object: t("parent-object"),
            parent_clear: t("parent-clear"),
            rename: t("outliner-rename"),
            bind_automatic: t("bind-automatic"),
            bind_empty: t("bind-empty"),
            unbind: t("unbind-armature"),
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
            shading_studio: t("shading-studio"),
            shading_flat: t("shading-flat"),
            shading_wireframe: t("shading-wireframe"),
            shading_rendered: t("shading-rendered"),
            shading_hint: t("shading-hint"),
            exposure: t("exposure"),
            exposure_metered: t("exposure-metered"),
            exposure_manual: t("exposure-manual"),
            menu_file: t("menu-file"),
            menu_edit: t("menu-edit"),
            menu_help: t("menu-help"),
            help_contact: t("help-contact"),
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
            view_perspective: t("view-perspective"),
            view_sidebar: t("view-sidebar"),
            overlays: t("overlays"),
            overlay_grid: t("overlay-grid"),
            overlay_wireframe: t("overlay-wireframe"),
            overlay_face_orientation: t("overlay-face-orientation"),
            overlay_statistics: t("overlay-statistics"),
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
            pivot_label: t("pivot-point"),
            undo_history: t("menu-undo-history"),
            undo_original: t("undo-original"),
            stats_objects: t("stats-objects"),
            stats_vertices: t("stats-vertices"),
            stats_edges: t("stats-edges"),
            stats_faces: t("stats-faces"),
            stats_triangles: t("stats-triangles"),
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
            // Workspaces right after the menus, where Blender puts them: they
            // are the top bar's own tabs, and the eye finds them at the same
            // place whatever the window width.
            draw_workspace_tabs(ui, state);
        });
    });

    Panel::bottom("status").show(ui, |ui| {
        ui.horizontal(|ui| {
            // Edit mode wears the accent, which is the same thing the accent
            // says everywhere else: this is what you are acting on. Object
            // mode is the resting state and says it quietly.
            let theme = kreate_ui::theme::from_ui(ui);
            let name = state.catalog.format(mode.label_key(), TextKind::Ui, &[]);
            if mode == Default::default() {
                ui.label(name);
            } else {
                // Anything but the resting state wears the accent, which is
                // what the accent says everywhere else: this is what you are
                // acting on.
                ui.colored_label(theme.accent, name);
            }
            // The mode's everyday keys, as Blender's status bar keeps them in
            // view: what the hand does next, without opening a menu to learn
            // it. Read from the keymap, so a rebinding shows here too.
            let hints: &[&str] = match mode.acts_on() {
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
                kreate_ops::mode::ActsOn::Weights | kreate_ops::mode::ActsOn::Sculpt => &[],
            };
            for id in hints {
                if let Some(info) = state
                    .command_info
                    .get(*id)
                    .filter(|i| !i.shortcut.is_empty())
                {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(&info.shortcut).strong().small());
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
                ui.monospace(entry.to_script_call());
            }
            // The standing counts on the right, where Blender keeps its
            // statistics: what is true of the scene, apart from what just
            // happened to it.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(&status);
            });
        });
    });

    // The screen. Boundaries are interacted with *before* the areas, so a drag
    // that starts on one is not stolen by whatever is under it — a handle is
    // four points wide and always overlaps an editor's edge.
    let mut viewport_rect = egui::Rect::ZERO;
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
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
        // flat look above.
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
                shade("view.shading_flat", &l.shading_flat, ShadingMode::Flat),
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
            draw_area_header(ui, state, area, l, doc, mode);
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
                        _ => None,
                    };
                    if let Some(radius) = brush {
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
                            }
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
                    let order = std::mem::take(&mut state.outliner);
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
                "shader" => {
                    draw_shader_editor(ui, state, doc, area.id, l);
                }
                "timeline" => {
                    draw_timeline(ui, state, area.id, l);
                }
                "dope_sheet" => {
                    draw_dope_sheet(ui, state, doc, area.id, l);
                }
                "graph" => {
                    draw_graph_editor(ui, state, doc, area.id, l);
                }
                "drivers" => {
                    draw_drivers(ui, state, doc, area.id);
                }
                "preferences" => {
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
    doc: &Document,
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
                        // Which mode, first, because it governs what every
                        // other key does and what the menus after it contain.
                        draw_mode_menu(ui, state, doc, mode);
                        ui.separator();
                        draw_brush_settings(ui, state, l, area.id.0, mode);
                    }

                    // Every editor has a View menu, and the area operations
                    // live at the bottom of it — which is where Blender keeps
                    // them, and which is what makes "how do I split this" one
                    // answer for every editor.
                    draw_view_menu(ui, state, area, l, is_viewport);

                    if is_viewport {
                        // Select, Add, and whatever the mode's card says acts on
                        // what it acts on — from the tree F3 searches.
                        for menu in viewport_menus(l, mode, state.select_mode) {
                            ui.menu_button(&menu.label, |ui| draw_nodes(ui, state, l, &menu.nodes));
                        }

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
    l: &Labels,
    area: u32,
    mode: kreate_ops::operator::Mode,
) {
    // The sculpt brush, in its header.
    if mode.acts_on() == kreate_ops::mode::ActsOn::Sculpt {
        let names = [
            &l.sculpt_draw,
            &l.sculpt_smooth,
            &l.sculpt_grab,
            &l.sculpt_inflate,
        ];
        let s = &mut state.sculpt;
        egui::ComboBox::from_id_salt(("sculpt-brush", area))
            .selected_text(names[s.brush.min(3) as usize].as_str())
            .show_ui(ui, |ui| {
                for (i, name) in names.iter().enumerate() {
                    ui.selectable_value(&mut s.brush, i as u32, name.as_str());
                }
            });
        ui.label(state.catalog.get("paint-radius"));
        ui.add(
            egui::DragValue::new(&mut state.sculpt.radius)
                .range(4.0..=400.0)
                .speed(1.0)
                .suffix(" px"),
        );
        ui.label(state.catalog.get("paint-strength"));
        ui.add(
            egui::DragValue::new(&mut state.sculpt.strength)
                .range(0.0..=1.0)
                .speed(0.01)
                .max_decimals(2),
        );
        ui.checkbox(
            &mut state.sculpt.mirror_x,
            state.catalog.get("sculpt-mirror-x"),
        );
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
            Run::Command(_) => None,
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
                // After a separator, because an export is not a save: the
                // document keeps no path from it and treating a `.glb` as a
                // working file loses everything the format cannot hold.
                entry(
                    "file.export",
                    &l.file_export,
                    Run::Command(Command::ExportScene),
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
                entry(
                    "help.contact",
                    &l.help_contact,
                    Run::Command(Command::OpenContact),
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
    let mut out = vec![TopMenu {
        label: l.menu_select.clone(),
        nodes: select_nodes(l, mode, select_mode),
    }];
    // What Add adds is the mode's: objects where objects are acted on, a bone
    // where a skeleton is being built, and nothing in pose mode — a pose adds
    // nothing, as Blender's has no Add menu there.
    match mode.acts_on() {
        // A pose and a brush add nothing, as in Blender.
        kreate_ops::mode::ActsOn::Bones
        | kreate_ops::mode::ActsOn::Weights
        | kreate_ops::mode::ActsOn::Sculpt => {}
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
            kreate_ops::mode::Menu::Object => out.push(TopMenu {
                label: l.menu_object.clone(),
                nodes: {
                    let mut nodes = object_nodes(l);
                    nodes.push(MenuNode::Separator);
                    nodes.extend(transform_nodes(l));
                    nodes
                },
            }),
            kreate_ops::mode::Menu::Mesh => out.extend(mesh_menus(l, select_mode)),
            kreate_ops::mode::Menu::Pose => out.push(TopMenu {
                label: l.menu_pose.clone(),
                nodes: pose_nodes(l),
            }),
            kreate_ops::mode::Menu::Sculpt => out.push(TopMenu {
                label: l.menu_sculpt.clone(),
                nodes: [
                    &l.sculpt_draw,
                    &l.sculpt_smooth,
                    &l.sculpt_grab,
                    &l.sculpt_inflate,
                ]
                .into_iter()
                .enumerate()
                .map(|(i, label)| {
                    MenuNode::Entry(
                        MenuEntry::new(
                            "sculpt.brush",
                            label,
                            Run::Command(Command::SetSculptBrush(i as u32)),
                        )
                        .variant("brush", kreate_reflect::PropertyValue::Enum(i as u32)),
                    )
                })
                .collect(),
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
    let shapes = (0..ADD_SHAPES.len())
        .map(|shape| {
            let value = V::Enum(shape as u32);
            MenuNode::Entry(
                MenuEntry::new(
                    "object.add_primitive",
                    &l.add_shapes[shape],
                    op("object.add_primitive", &[("shape", value.clone())]),
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
    // Right-to-left layout, so the list is reversed to read Studio, Flat,
    // Rendered from left to right on screen.
    for mode in [
        kreate_render::ShadingMode::Rendered,
        kreate_render::ShadingMode::Flat,
        kreate_render::ShadingMode::Studio,
        kreate_render::ShadingMode::Wireframe,
    ] {
        // i18n-exempt: a translated name and a translated hint
        let tip = format!("{}\n{}", shading_label(mode, l), l.shading_hint);
        if kreate_ui::view_icons::shading_button(ui, mode, mode == current)
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
}

/// F2, Blender's: the active object's name in a field under the pointer.
/// Enter or a click away keeps it, Escape leaves it as it was.
fn draw_rename(ui: &mut egui::Ui, state: &mut ShellState, viewport: egui::Rect) {
    let Some((at, text)) = state.rename.as_mut() else {
        return;
    };
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
                        Tool::SculptDraw
                        | Tool::SculptSmooth
                        | Tool::SculptGrab
                        | Tool::SculptInflate => None,
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
                            Tool::SculptDraw
                            | Tool::SculptSmooth
                            | Tool::SculptGrab
                            | Tool::SculptInflate => {
                                if let Some(i) = Tool::SCULPT.iter().position(|t| t == tool) {
                                    state.sculpt.brush = i as u32;
                                }
                            }
                        }
                    }
                }
            });
        },
    );
    rect
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
    let theme = kreate_ui::theme::from_ui(ui);
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
        // a box would be a panel where the eye expects a scene.
        painter.galley(
            egui::pos2(left + 1.0, y + 1.0),
            galley.clone(),
            theme.sunken.gamma_multiply(0.8),
        );
        painter.galley(egui::pos2(left, y), galley.clone(), colour);
        y += galley.size().y + 2.0;
    }
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
        for target in kreate_ops::ops::mesh::edit_targets(doc) {
            let Some(mesh) = doc.mesh(target.mesh) else {
                continue;
            };
            v[0] += count(mesh, names::SELECT_VERT);
            v[1] += mesh.num_verts() as usize;
            e[0] += count(mesh, names::SELECT_EDGE);
            e[1] += mesh.num_edges() as usize;
            f[0] += count(mesh, names::SELECT_FACE);
            f[1] += mesh.num_faces() as usize;
            t += triangles(mesh);
        }
        return vec![
            row(&l.stats_vertices, format!("{}/{}", v[0], v[1])),
            row(&l.stats_edges, format!("{}/{}", e[0], e[1])),
            row(&l.stats_faces, format!("{}/{}", f[0], f[1])),
            row(&l.stats_triangles, t.to_string()),
        ];
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
            text: entry.to_script_call(),
            kind: LineKind::History,
        })
        .collect();
    lines.extend(state.console_log.iter().cloned());
    let text = kreate_ui::console::ConsoleText {
        placeholder: l.console_placeholder.clone(),
        empty: l.console_empty.clone(),
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
    let Some(result) = &state.render_result else { return };
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

/// The Render tab: the scene's render settings, and the buttons that use them.
fn draw_render_tab(ui: &mut Ui, state: &mut ShellState, doc: &Document, registry: &OperatorRegistry, l: &Labels) {
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
    let Some(info) = registry.get("render.set_settings") else { return };
    let mut params = kreate_reflect::OpParams::to_map(&kreate_ops::ops::render::render_params(doc));
    if kreate_ui::property_editor(ui, &state.catalog, info.id, info.schema, &mut params) {
        let dragging = ui.ctx().input(|i| i.pointer.any_down());
        send_edit(state, "render.set_settings", params, dragging);
    }
}

/// The Claude panel: the conversation, and a box to say the next thing.
fn draw_chat(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    let text = kreate_ui::chat::ChatText {
        placeholder: l.chat_placeholder.clone(),
        empty: l.chat_empty.clone(),
        send: l.chat_send.clone(),
        stop: l.chat_stop.clone(),
        new_chat: l.chat_new.clone(),
        working: l.chat_working.clone(),
        checking: l.chat_checking.clone(),
        signed_out: l.chat_signed_out.clone(),
        sign_in: l.chat_sign_in.clone(),
        signing_in: l.chat_signing_in.clone(),
        open_sign_in: l.chat_open_sign_in.clone(),
        code_hint: l.chat_code_hint.clone(),
        code_submit: l.chat_code_submit.clone(),
        cancel: l.chat_cancel.clone(),
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
    };
    kreate_ui::chat::chat(
        ui,
        &text,
        &providers,
        &state.chat_account,
        &state.chat_log,
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
    apply_node_edits(state, slot, edits);
}

/// Turns what the pointer did in the graph into the operators that do it.
fn apply_node_edits(state: &mut ShellState, slot: u32, edits: Vec<kreate_ui::NodeEdit>) {
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
        Socket::Vector => SocketType::Vector,
    };
    let socket_label = |name: &str| t(&format!("node-socket-{}", slug(name)));
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
            let outputs = kind
                .outputs()
                .iter()
                .zip(kind.output_names())
                .map(|(s, name)| SocketView {
                    label: socket_label(name),
                    ty: ty(*s),
                    connected: false,
                    default: None,
                })
                .collect();
            let inputs = kind
                .inputs()
                .iter()
                .zip(kind.input_names())
                .enumerate()
                .map(|(i, (s, name))| SocketView {
                    label: socket_label(name),
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
                NodeKind::Image { image } => vec![FieldView::Image {
                    name: image.and_then(|id| doc.image(id)).map(|i| i.name.clone()),
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
                    NodeKind::Color(_) | NodeKind::Value(_) => Family::Input,
                    NodeKind::Image { .. } => Family::Texture,
                    NodeKind::Multiply | NodeKind::MultiplyValue | NodeKind::SeparateRgb => {
                        Family::Converter
                    }
                    NodeKind::NormalMap { .. } => Family::Vector,
                    NodeKind::Surface { .. } => Family::Shader,
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
                    if ui.selectable_label(state.sidebar_tab == tab, label.as_str()).clicked() {
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
    Some(((bounds.max - bounds.min).as_dvec3() * o.transform.scale).abs().to_array())
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
    let (Some(size), Some(info)) = (object_dimensions(doc, object), registry.get("object.set_dimensions")) else {
        return;
    };
    ui.separator();
    ui.strong(&l.sidebar_dimensions);
    let mut params = PropertyMap::new();
    params.set("dimensions", kreate_reflect::PropertyValue::Vec3(size));
    if kreate_ui::property_editor(ui, &state.catalog, "sidebar.dimensions", info.schema, &mut params) {
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
        .add(egui::DragValue::new(&mut lens).range(1.0..=1000.0).speed(0.5).max_decimals(1))
        .changed()
    {
        state.view_lens_mm = lens;
        let mut params = PropertyMap::new();
        params.set("lens", kreate_reflect::PropertyValue::Float(lens));
        state.queue.dispatch("view.set_lens", params);
    }
    ui.separator();
    ui.strong(&l.sidebar_cursor);
    let Some(info) = registry.get("scene.cursor_set") else { return };
    let mut params = PropertyMap::new();
    params.set("location", kreate_reflect::PropertyValue::Vec3(doc.cursor.to_array()));
    if kreate_ui::property_editor(ui, &state.catalog, "sidebar.cursor", info.schema, &mut params) {
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
fn draw_workspace_tabs(ui: &mut Ui, state: &mut ShellState) {
    let active = state.workspaces.active_index();
    // Collected first: the loop borrows the queue, and iterating the list while
    // pushing to it is the same mistake the layout edits are deferred to avoid.
    let names: Vec<String> = state
        .workspaces
        .iter()
        .map(|space| space.name.clone())
        .collect();
    for (index, name) in names.iter().enumerate() {
        // The built-in layouts' names are the program's words and are
        // translated; a name the user gave one is theirs and is not.
        let key = format!("workspace-{}", name.to_lowercase().replace(' ', "-"));
        let shown = if state.catalog.has(&key) {
            state.catalog.get(&key)
        } else {
            name.clone()
        };
        if ui.selectable_label(index == active, shown).clicked() {
            state.switch_workspace(index);
        }
    }
}

/// The mode dropdown in the top bar.
///
/// Every mode, with the ones that cannot be entered disabled rather than
/// hidden: a menu whose entries come and go is a menu the user cannot learn,
/// and "edit mode is greyed out because nothing is selected" is a useful thing
/// to be told. Whether a mode can be entered comes from the mode itself — see
/// `Mode::needs_mesh`.
fn draw_mode_menu(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    mode: kreate_ops::operator::Mode,
) {
    use kreate_ops::operator::Mode;

    let current = state.catalog.format(mode.label_key(), TextKind::Ui, &[]);
    let names: Vec<(Mode, String)> = Mode::ALL
        .iter()
        .map(|kind| {
            (
                *kind,
                state.catalog.format(kind.label_key(), TextKind::Ui, &[]),
            )
        })
        .collect();
    // Asked once, outside the closure: the answer is the same for every entry
    // and reading the document inside would hold a borrow across the queue.
    let has = kreate_ops::mode::Has::of(doc);

    egui::ComboBox::from_id_salt("mode")
        .selected_text(current)
        .show_ui(ui, |ui| {
            // This object's modes; see `Mode::offered`.
            for (kind, name) in names.iter().filter(|(k, _)| *k == mode || k.offered(has)) {
                ui.add_enabled_ui(kind.can_enter(has), |ui| {
                    if ui.selectable_label(*kind == mode, name).clicked() {
                        state.queue.push(Command::SetMode(kind.tag().to_string()));
                    }
                });
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
    ];
    egui::ComboBox::from_id_salt(("orientation", area))
        .width(58.0)
        .selected_text(orientations[tools.orientation.min(1) as usize].as_str())
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
    .pivot(egui::Align2::LEFT_BOTTOM)
    .fixed_pos(viewport.left_bottom() + egui::vec2(10.0, -10.0))
    .constrain_to(viewport)
    .show(ui.ctx(), |ui| {
        egui::Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_max_width(250.0);
            egui::CollapsingHeader::new(label)
                .id_salt("adjust-last")
                .default_open(false)
                .show(ui, |ui| draw_adjust_fields(ui, state, registry, history));
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
                }
            });
        });
    }
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
            draw_vertex_groups(ui, state, doc, id, active);
            draw_shape_keys(ui, state, active);
        }
        _ => {}
    }
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
];

/// The built-in English catalog.
///
/// Shipped as the fallback so an untranslated key resolves to readable text
/// rather than to itself. The Korean resource sits beside it.
pub const EN_FTL: &str = r#"
menu-file-new = New
menu-file-open = Open...
menu-file-save = Save
menu-file-save-as = Save As...
menu-file-export = Export...
menu-file-run-script = Run Script...
menu-search = Search...
menu-copy-script = Copy History as Script
palette-placeholder = Search the menus: extrude, add cube, save...
palette-nothing = Nothing matches.
pie-shading = Shading
editor-timeline = Timeline
editor-dope-sheet = Dope Sheet
editor-graph = Graph Editor
editor-drivers = Drivers
drivers-empty = No drivers on the selected objects
drivers-add = Add Driver
drivers-remove = Remove
drivers-expression = Expression
drivers-add-variable = Add Variable
drivers-remove-variable = Remove
drivers-none = (none)
driver-error-empty = The expression is empty
driver-error-syntax = Syntax error at character { $at }
driver-error-name = Unknown name: { $name }
driver-error-function = Unknown function: { $name }
driver-error-arguments = { $name } takes { $count } arguments
driver-error-infinite = The result is not a finite number
driver-error-target = “{ $name }” reads no object
driver-error-property = “{ $name }” reads no such property
driver-error-bad-name = “{ $name }” is not a valid name
driver-error-cycle = Drivers read each other in a loop
graph-handles = Handles
handle-auto = Auto Clamped
handle-flat = Flat
sheet-empty = Select an animated object to see its keys.
sheet-interpolation = Interpolation
interp-constant = Constant
interp-linear = Linear
interp-bezier = Bezier
channel-location = Location
channel-rotation-euler = Rotation
channel-scale = Scale
channel-rotation-quaternion = Rotation
channel-base-color = Base Color
channel-metallic = Metallic
channel-roughness = Roughness
channel-alpha = Alpha
channel-normal-strength = Normal Strength
channel-color = Color
channel-intensity = Intensity
channel-exposure = Exposure
time-play = Play
key-property = Keyframe: keys this value at the current frame, or takes away its key there
time-autokey = Auto-keying: while on, moving, turning or scaling an object keys it at the current frame
time-pause = Pause
time-jump-start = Jump to the start
time-jump-end = Jump to the end
time-start = Start
time-end = End
menu-animation = Animation
op-time-play = Play Animation
op-time-play-desc = Plays the scene from the frame it is on, looping at the end of the range. Again to stop.
op-anim-keyframe-insert = Insert Keyframe
op-anim-keyframe-insert-desc = Keys the selected objects' location, rotation and scale at the current frame.
op-anim-keyframe-delete = Delete Keyframe
op-anim-keyframe-delete-desc = Takes away the selected objects' keys at the current frame.
pie-view = View
pie-mode = Mode
view-left = Left
view-right = Right
view-bottom = Bottom
view-top = Top
view-front = Front
view-back = Back
op-view-shading-studio-desc = Lights that show the shape, whatever the scene's lamps are doing.
op-view-shading-flat-desc = Every face its plain colour, with no light at all.
op-view-shading-wireframe-desc = Only the edges, the back of the model showing through.
op-view-shading-rendered-desc = The scene's own lamps and sky, through the camera's exposure.
op-view-axis-left-desc = Looks at the pivot from the left, along +X.
op-view-axis-right-desc = Looks at the pivot from the right, along -X.
op-view-axis-bottom-desc = Looks up at the pivot from below.
op-view-axis-top-desc = Looks down at the pivot from above.
op-view-axis-front-desc = Looks at the pivot from the front, along +Y.
op-view-axis-back-desc = Looks at the pivot from behind, along -Y.
op-mode-object-desc = Acts on whole objects: select, move and add them.
op-mode-edit-desc = Acts on the vertices, edges and faces of the meshes being edited.
op-mode-pose-desc = Acts on the bones of the active armature, as it holds them.
op-app-search = Search
op-app-search-desc = Find any menu command by typing part of its name, and run it with Enter.
op-edit-copy-script = Copy History as Script
op-edit-copy-script-desc = Puts every edit so far on the clipboard as script lines the console and Run Script can replay.
op-help-contact = Contact the Developers
op-help-contact-desc = Writes to the developers: a problem, a feature you would like, a question. An email address is only needed for a reply.
op-help-about = About Kreate
op-help-about-desc = Shows which version of Kreate this is and the commit it was built from.
op-file-run-script = Run Script
op-file-run-script-desc = Runs a text file of script lines, one undo step per line, the way the console runs what is typed into it.
world-pick-image = Environment...
world-clear-image = Clear
material-pick-image = Open Image...
material-clear-image = Clear
material-new = New Material
material-browse = Material
material-assign = Assign
material-assign-hint = The selected faces take this slot's material.
material-select = Select
material-select-hint = Selects the faces on this slot.
op-material-assign = Assign Material
op-material-assign-desc = The selected faces take this slot's material.
op-material-select = Select by Material
op-material-select-desc = Selects the faces on this slot.
op-material-link = Link Material
op-material-link-desc = Puts an existing material in this slot.
material-none = This object has no material.
material-base-colour = Base Colour
material-metallic-roughness = Metallic / Roughness
material-normal-map = Normal Map
material-empty-slot = Empty
material-no-slots = No slots
material-add-slot = Add a material slot
material-remove-slot = Remove this material slot
recover-count = { $count } recovered documents
recover-menu = Recover
recover-none = Nothing to recover
recover-hint = { $count } recovered documents waiting in File ‣ Recover
recover-open = Open
recover-discard = Discard
recover-discard-all = Discard all
recover-age-now = just now
recover-age-minutes = { $n } min ago
recover-age-hours = { $n } h ago
recover-age-days = { $n } d ago
status-saved = Saved { $path }
status-opened = Opened { $path }
menu-add-mesh = Mesh
menu-add-light = Light
menu-add-camera = Camera
menu-add-empty = Empty
menu-add-armature = Armature
menu-pose = Pose
menu-armature = Armature
armature-add-bone = Add Bone
armature-extrude = Extrude
armature-delete = Delete Bones
mode-edit-armature = Edit Mode
keymap-context-edit-armature = Armature Edit Mode
op-mode-edit-armature-desc = Acts on the heads and tails of the active armature's bones, as built.
op-armature-bone-add-desc = Adds a free bone standing a unit tall.
op-armature-extrude-desc = Grows a connected bone out of each selected tail, then moves it.
op-armature-delete-desc = Deletes the selected bones; their children go to their parents.
op-armature-select-all-desc = Selects every bone end, none, or the others.
pose-clear-all = Clear Transform
pose-clear-location = Clear Location
pose-clear-rotation = Clear Rotation
pose-clear-scale = Clear Scale
mode-pose = Pose Mode
outliner-kind-armature = Armature
keymap-context-pose = Pose Mode
menu-add-plane = Plane
menu-add-cube = Cube
menu-add-grid = Grid
menu-add-uv-sphere = UV Sphere
menu-add-ico-sphere = Ico Sphere
menu-add-cylinder = Cylinder
menu-add-cone = Cone
menu-add-torus = Torus
outliner-scene-collection = Scene Collection
menu-duplicate = Duplicate
menu-delete = Delete
menu-undo = Undo
menu-redo = Redo
menu-extrude = Extrude
menu-subdivide = Subdivide
menu-dissolve = Dissolve
menu-inset = Inset
menu-bridge = Bridge
menu-loop-cut = Loop Cut
menu-shade-smooth = Smooth
menu-shade-flat = Flat
menu-mark-sharp = Auto Sharp
menu-recalc-normals = Recalc Normals
menu-fill = Fill
menu-spin = Spin
menu-bevel = Bevel
menu-merge = Merge
menu-weld = Weld
menu-select-loop = Loop
pie-snap = Snap
menu-snap = Snap
menu-apply = Apply
apply-location = Location
apply-rotation = Rotation
apply-scale = Scale
apply-all = All Transforms
apply-rotation-scale = Rotation & Scale
menu-set-origin = Set Origin
origin-to-geometry = Origin to Geometry
origin-to-cursor = Origin to 3D Cursor
geometry-to-origin = Geometry to Origin
pivot-point = Pivot Point
pivot-median = Median Point
orientation-global = Global
orientation-local = Local
orientation-hint = Transform orientation: what the first X, Y or Z press constrains to
snap-toggle = Snap
snap-on = Snapping
snap-hint = Snap while moving (Shift+Tab); Ctrl turns it the other way for one drag, B mid-drag too
snap-increment = Increment
snap-element = Vertex, Edge, Face
snap-vertex = Vertex
snap-edge = Edge
snap-midpoint = Edge Midpoint
snap-face = Face
proportional-toggle = Proportional
proportional-on = Proportional Editing
proportional-hint = Proportional editing (O): the selection pulls its neighbours along; the wheel sets how far
falloff-smooth = Smooth
falloff-sphere = Sphere
falloff-root = Root
falloff-sharp = Sharp
falloff-linear = Linear
falloff-constant = Constant
select-tool-box = Select Box
select-tool-lasso = Select Lasso
select-tool-circle = Select Circle
select-tool-hint = Click for the next: box, lasso, circle
pivot-individual = Individual Origins
pivot-active = Active Element
pivot-cursor = 3D Cursor
op-view-cursor-to-selected = Cursor to Selected
op-view-cursor-to-selected-desc = Puts the 3D cursor in the middle of the selection.
op-view-cursor-to-origin = Cursor to World Origin
op-view-cursor-to-origin-desc = Puts the 3D cursor back at the world's origin.
op-view-selection-to-cursor = Selection to Cursor
op-view-selection-to-cursor-desc = Moves the selection so its middle is on the 3D cursor.
op-view-cursor-place = Place 3D Cursor
op-mesh-knife = Knife
panel-shape-keys = Shape Keys
shape-key-basis = Basis
shape-key-from-mix = From Mix
shape-keys-editing = Edit mode is editing the active key.
op-object-shape-key-add = Add Shape Key
op-object-shape-key-add-desc = Adds a shape key — from the basis, or from what the keys make now.
op-object-shape-key-remove = Remove Shape Key
op-object-shape-key-set = Set Shape Key
op-object-shape-key-set-desc = Turns a shape key up or down.
op-object-shape-key-set-active = Set Active Shape Key
op-object-shape-key-edit = Edit Shape Key
panel-bone-ik = Inverse Kinematics
op-pose-ik-add = Add IK
op-pose-ik-add-desc = Makes the active bone reach for the other selected bone, turning the bones above it.
op-pose-ik-set = Set IK
op-pose-ik-set-desc = Changes a bone's IK target, chain or whether it is on.
op-pose-ik-remove = Remove IK
op-pose-ik-remove-desc = Takes a bone's IK off.
ik-enabled = On
ik-target = Target
ik-chain = Chain
ik-pole = Pole
ik-pole-desc = A bone the elbow or knee bends towards.
ik-chain-desc = How many bones turn, counting this one; 0 for all the way up.
menu-move-to-collection = Move to Collection
collection-scene = Scene Collection
collection-new = New Collection
op-object-move-to-collection = Move to Collection
op-object-move-to-collection-desc = Moves the selected objects into a collection.
op-collection-set-visible = Show or Hide Collection
op-collection-set-visible-desc = Hides a collection and everything in it, or shows it again.
op-collection-remove = Remove Collection
op-collection-remove-desc = Removes a collection; what was in it stays in the scene.
op-collection-rename = Rename Collection
op-collection-rename-desc = Renames a collection.
overlay-face-orientation = Face Orientation
menu-undo-history = Undo History
undo-original = Original
overlay-statistics = Statistics
overlay-ao = Ambient Occlusion
overlay-xray = X-Ray
overlay-outline = Outline Selected
overlay-bloom = Bloom (Rendered)
overlay-reflections = Reflections (Rendered)
overlay-contact-shadows = Contact Shadows (Rendered)
overlay-path-trace = Path Tracing (Rendered)
op-view-toggle-path-trace = Path Tracing
op-view-toggle-path-trace-desc = The rendered view traced with light bouncing between surfaces; it refines while the view is still.
op-view-toggle-reflections = Reflections
op-view-toggle-reflections-desc = In the rendered view, shiny surfaces reflect what is on screen.
op-view-toggle-contact-shadows = Contact Shadows
op-view-toggle-contact-shadows-desc = In the rendered view, darkens the last centimetres where things touch, finer than the shadow maps can.
op-view-toggle-bloom = Bloom
op-render-image = Render Image
op-render-image-desc = Renders the picture through the scene camera with the path tracer, at the Render tab's size and samples, and saves it. Esc stops it.
op-render-animation = Render Animation
op-render-animation-desc = Renders every frame of the frame range through the scene camera and saves each one, numbered. Esc stops it.
op-render-view = View Render
op-render-view-desc = Shows the last rendered picture again.
op-view-toggle-bloom-desc = In the rendered view, light brighter than white glows into its surroundings.
op-view-toggle-outline = Outline Selected
op-view-toggle-outline-desc = Draws an orange line around the selected objects, lighter around the active one.
op-view-toggle-xray = X-Ray
op-view-toggle-xray-desc = See through the surfaces, and select what is behind them (Alt+Z).
op-view-toggle-ao = Ambient Occlusion
op-view-toggle-ao-desc = Darkens corners and creases in the studio view, so the shape reads.
op-view-toggle-face-orientation = Face Orientation
op-view-toggle-face-orientation-desc = Colours faces blue from the front and red from the back.
op-view-toggle-statistics = Statistics
op-view-toggle-statistics-desc = Shows how many objects, vertices, edges and faces there are.
stats-objects = Objects
stats-vertices = Vertices
stats-edges = Edges
stats-faces = Faces
stats-triangles = Triangles
op-view-circle-select = Circle Select
op-view-circle-select-desc = A brush that selects what it passes over. Shift to deselect, wheel for size, Esc or right-click to stop.
op-mesh-knife-desc = Cuts new edges along a path clicked over the surface. Enter cuts, Esc gives up.
op-view-cursor-place-desc = Puts the 3D cursor on the surface under the pointer.
op-scene-cursor-set = Set 3D Cursor
op-scene-cursor-set-desc = Moves the 3D cursor, where new objects appear.
op-object-origin-set = Set Origin
op-object-origin-set-desc = Moves the origin of the selected meshes without moving what is drawn.
op-object-transform-apply = Apply Transform
op-object-transform-apply-desc = Bakes the transform into the mesh and resets it, leaving everything in place.
menu-select-more = More
menu-select-less = Less
menu-select-linked = Linked
menu-hide-selected = Hide Selected
menu-hide-unselected = Hide Unselected
menu-reveal = Reveal Hidden
op-mesh-hide = Hide
op-mesh-hide-desc = Puts the selection out of sight in edit mode.
op-mesh-reveal = Reveal Hidden
op-mesh-reveal-desc = Brings back everything hidden, selected.
mode-object = Object Mode
mode-edit = Edit Mode
select-mode-vert = Vertex
select-mode-edge = Edge
select-mode-face = Face
keymap-preset = Preset
keymap-changed = { $count } changed
keymap-reset-all = Reset all
keymap-shadowed = { $count } shadowed
keymap-reset-hint = Back to the preset
keymap-shadowed-hint = Another binding takes this trigger first
keymap-capture-hint = Click, then press the key you want
keymap-capturing = press a key…
keymap-context-global = Global
keymap-context-viewport = Viewport
keymap-context-object = Object Mode
keymap-context-edit = Edit Mode
keymap-context-uv = UV Editor
keymap-context-modal = Modal Tools
uv-none = This mesh has no UVs yet. Project some with U, or from the right-click menu.
area-menu = Area
sidebar-item = Item
tab-render = Render
menu-render = Render
render-image = Render Image
render-animation = Render Animation
render-view = View Render
render-result-title = Render Result
render-hint = Renders through the scene camera with the path tracer. Esc stops a render.
render-saved = Saved { $path }
render-failed = Not saved: { $why }
sidebar-tool = Tool
sidebar-view = View
sidebar-dimensions = Dimensions
sidebar-lens = Focal Length (mm)
sidebar-cursor = 3D Cursor
mode-weight-paint = Weight Paint
keymap-context-weight-paint = Weight Paint
keymap-context-sculpt = Sculpt
mode-sculpt = Sculpt Mode
op-mode-sculpt-desc = Pushes and pulls the surface with brushes.
menu-sculpt = Sculpt
sculpt-brush = Brush
sculpt-draw = Draw
sculpt-smooth = Smooth
sculpt-grab = Grab
sculpt-inflate = Inflate
sculpt-mirror-x = X Symmetry
op-sculpt-stroke = Sculpt
op-sculpt-stroke-desc = One stroke of a sculpt brush.
op-sculpt-brush = Sculpt Brush
op-sculpt-brush-desc = Picks the sculpt brush.
op-mode-weight-paint-desc = Paints how much each vertex follows each bone.
menu-weights = Weights
op-object-vertex-group-add = Add Vertex Group
op-object-vertex-group-add-desc = Adds an empty vertex group and makes it the one painted.
op-object-vertex-group-remove = Remove Vertex Group
op-paint-weights-normalize = Normalize All
op-paint-weights-normalize-desc = Scales every vertex's weights so they add up to one.
panel-vertex-groups = Vertex Groups
groups-none = No vertex groups
paint-weight = Weight
paint-radius = Radius
paint-strength = Strength
paint-front-only = Front Faces Only
paint-falloff = Soft Edge
menu-parent = Parent
menu-select-similar = Select Similar
op-mesh-select-shortest-path = Select Shortest Path
op-mesh-select-shortest-path-desc = Selects the shortest way from the selection to the clicked element (Ctrl+click).
similar-face-normal = Normal
similar-face-area = Area
similar-face-coplanar = Coplanar
similar-face-material = Material
similar-face-sides = Number of Sides
similar-edge-length = Length
similar-edge-direction = Direction
similar-vert-normal = Normal
similar-vert-faces = Number of Faces
op-mesh-select-similar = Select Similar
op-mesh-select-similar-desc = Selects everything like what is selected (Shift+G).
menu-file-open-recent = Open Recent
recent-none = Nothing opened yet
menu-show-hide = Show/Hide
reveal-hidden = Reveal Hidden
op-object-hide = Hide
op-object-hide-desc = Hides the selected objects (H).
op-object-hide-unselected-desc = Hides everything not selected (Shift+H).
op-object-reveal = Reveal Hidden
op-object-reveal-desc = Shows every hidden object, and selects them (Alt+H).
op-object-parent-one = Parent
op-object-parent-one-desc = Parents one object to another, by name — the outliner's drag and drop.
menu-parent-to = Set Parent To
parent-object = Object
parent-clear = Clear Parent
outliner-rename = Rename
outliner-remove-collection = Delete Collection
op-app-rename = Rename
op-app-rename-desc = Renames the active object (F2).
op-object-parent = Parent
op-object-parent-desc = Makes the active object the parent of the other selected objects; they stay where they are.
op-object-parent-clear-desc = Frees the selected objects from their parent; they stay where they are.
op-object-rename = Rename
op-object-rename-desc = Renames the active object.
bind-automatic = With Automatic Weights
bind-empty = With Empty Groups
unbind-armature = Remove Armature Deform
panel-relations = Relations
relations-parent = Parent
relations-armature = Armature
op-object-bind-armature-desc = Parents the selected meshes to the active armature, and makes them follow its bones.
op-object-bind-armature-weights-0-desc = Parents the selected meshes to the active armature, with weights from how near each vertex is to each bone.
op-object-bind-armature-weights-1-desc = Parents the selected meshes to the active armature, with an empty group per bone to paint.
op-object-unbind-armature-desc = Stops the selected meshes following their armature; their weights are kept.
bone-parent = Parent
workspace-layout = Layout
workspace-modeling = Modeling
workspace-uv-editing = UV Editing
workspace-shading = Shading
workspace-animation = Animation
workspace-scripting = Scripting
field-insert-key = Insert Keyframe
field-delete-key = Delete Keyframe
field-add-driver = Add Driver
field-remove-driver = Remove Driver
field-reset = Reset to Default
tab-bone = Bone
panel-pose = Pose
panel-bone-rest = Bone
panel-viewport-display = Viewport Display
sidebar-nothing = Nothing is active. Select an object to see where it is.
area-split-vertical = Split left / right
area-split-horizontal = Split top / bottom
area-close = Close area
editor-viewport = 3D Viewport
editor-outliner = Outliner
editor-properties = Properties
editor-uv = UV Editor
editor-preferences = Preferences
editor-console = Console
editor-claude = AI Chat
editor-shader = Shader Editor
material-uses-nodes = This material uses nodes. Edit it in the Shader Editor.
node-use-nodes = Use Nodes
node-use-nodes-hint = Give this material a node graph, built from what it already is
node-no-graph = This material has no node graph yet.
node-no-material = The active object has no material in this slot.
node-add = Add
node-no-image = No image
node-open-image = Open Image...
node-kind-color = RGB
node-kind-value = Value
node-kind-image = Image Texture
node-kind-multiply = Mix (Multiply)
node-kind-multiply-value = Math (Multiply)
node-kind-separate-rgb = Separate RGB
node-kind-normal-map = Normal Map
node-socket-base-color = Base Color
node-socket-metallic = Metallic
node-socket-roughness = Roughness
node-socket-normal = Normal
node-socket-color = Color
node-socket-value = Value
node-socket-alpha = Alpha
node-socket-a = A
node-socket-b = B
node-socket-r = R
node-socket-g = G
node-socket-strength = Strength
console-placeholder = kreate.ops.mesh.extrude_region(offset=1.0)
console-empty = Every edit you make appears here as a line of script. Type one to run it.
chat-placeholder = Ask for something to be built or changed…
chat-empty = Ask about the scene, or to change it — with Claude, Gemini or ChatGPT, chosen above. Every edit it makes is one undo step.
chat-send = Send
chat-stop = Stop
chat-new = New Chat
chat-working = Working…
chat-stopped = Stopped.
chat-checking = Checking the account…
chat-signed-out = Not signed in.
chat-sign-in = Sign In
chat-signing-in = Finish signing in in the browser that opened.
chat-open-sign-in = Open the sign-in page
chat-code-hint = Paste the code here if the page shows one
chat-code-submit = Submit
chat-cancel = Cancel
sculpt-coarse = This mesh has { $count } vertices: too few for a brush to shape. Add detail first, with a subdivide modifier applied or Subdivide in edit mode.
menu-help = Help
help-contact = Contact the Developers…
help-about = About Kreate
contact-title = Contact the Developers
contact-intro = Report a problem, ask for a feature or ask a question. Leave an email address if you would like an answer.
contact-kind-bug = Bug
contact-kind-feature = Feature Request
contact-kind-question = Question
contact-kind-other = Other
contact-name = Name
contact-email = Email
contact-email-hint = Optional, for a reply
contact-message-hint = What happened, or what you would like…
contact-details = Include the version and platform
contact-send = Send
contact-close = Close
contact-sending = Sending…
contact-sent = Sent. Thank you!
about-title = About Kreate
about-version = Version
splash-title = Kreate
splash-shaders = Preparing the renderer…
splash-warming = Compiling shaders…
splash-opening = Opening the document…
splash-version = Version { $version }
uv-not-editing = The UV editor shows the mesh being edited. Press Tab to enter edit mode.
adjust-nothing = No operation to adjust
status-counts = { $objects } objects, { $meshes } meshes, { $selected } selected
outliner-kind-empty = Empty
outliner-kind-mesh = Mesh
outliner-kind-camera = Camera
outliner-kind-light = Light
outliner-shared-mesh = Shared by { $count } objects
outliner-mesh-summary = { $name }: { $verts } verts, { $faces } faces
shading-studio = Studio
shading-flat = Flat
shading-wireframe = Wireframe
shading-rendered = Rendered
shading-hint = How the viewport shades (Z)
exposure = EV
exposure-metered = Metered off the scene. Turning a lamp down moves this and leaves the picture where it was — which is what exposure means. Turn it off on the camera to set it by hand.
exposure-manual = From the camera's f-stop, shutter and ISO.
tab-tool = Tool
tab-object = Object
tab-data = Data
tab-modifiers = Modifiers
tab-data-light = Light
tab-data-camera = Camera
tab-material = Material
tab-world = World
world-image = The background is an image; the colour below does not apply to it.
light-shared = Shared by { $count } objects — editing this changes all of them
menu-file = File
menu-edit = Edit
menu-view = View
menu-select = Select
menu-transform = Transform
select-all = All
select-none = None
select-invert = Invert
view-frame-all = Frame All
view-frame-selected = Frame Selected
view-camera = Camera
view-perspective = Perspective/Orthographic
view-sidebar = Sidebar
overlays = Overlays
overlay-grid = Grid
overlay-wireframe = Wireframe
transform-move = Move
transform-rotate = Rotate
transform-scale = Scale
view-user-perspective = User Perspective
view-user-orthographic = User Orthographic
panel-transform = Transform
panel-surface = Surface
panel-light = Light
outliner-hide = Hide in viewport
outliner-show = Show in viewport
tool-select = Select Box
tool-move = Move
tool-rotate = Rotate
tool-scale = Scale
tool-extrude = Extrude Region
tool-inset = Inset Faces
tool-bevel = Bevel
tool-loop-cut = Loop Cut
menu-object = Object
menu-mesh = Mesh
menu-edge = Edge
menu-face = Face
menu-add = Add
object-copy = Copy
object-paste = Paste
menu-add-light-point = Point Light
menu-add-light-sun = Sun
menu-add-light-spot = Spot Light
menu-add-light-area = Area Light
panel-modifiers = Modifiers
modifier-none = No modifiers
modifier-add = Add Modifier
modifier-remove = Remove
modifier-move-up = Move towards the input
modifier-move-down = Move towards the result
modifier-enabled = Use this modifier
modifier-apply = Apply: bake into the mesh and remove from the stack
op-object-modifier-set-boolean-object = Object
op-object-modifier-apply = Apply Modifier
op-object-modifier-apply-desc = Bakes the modifier into the mesh and takes it off the stack.
modifier-mirror = Mirror
modifier-subdivide = Subdivision
modifier-array = Array
modifier-solidify = Solidify
modifier-bevel = Bevel
modifier-boolean = Boolean

# --- right-click menu groups ---
menu-group-transform = Transform
menu-group-shape = Shape
menu-group-remove = Remove
menu-group-shading = Shading
menu-group-uv = UV
menu-group-object = Object

# --- operator names and descriptions ---
#
# The key is the operator id with dots and underscores turned to dashes, because
# a Fluent identifier cannot contain either. A missing entry falls back to the
# operator's built-in English label, so a partial translation stays usable.
#
# Descriptions are one sentence and say what the command *does to the model*,
# not what it is called. "Extrude Region" is the name; "pulls the selected faces
# out along their shared normal" is what someone who does not yet know the word
# needs to read.
op-mesh-translate = Move
op-mesh-translate-desc = Moves the selected vertices, edges or faces.
op-file-new-desc = Starts an empty document. Asks first if the open one has unsaved work.
op-file-open-desc = Opens a .kreate file, replacing the open document.
op-file-save-desc = Writes the document to its file, or asks where if it has none yet.
op-file-save-as-desc = Writes the document to a new file and works from there on.
op-file-export-desc = Writes the scene in an interchange format. The document keeps its own file.
op-edit-undo-desc = Takes back the last change.
op-edit-redo-desc = Does the last undone change again.
op-edit-copy-desc = Copies the selected objects, with their meshes, materials and textures.
op-edit-paste-desc = Adds the copied objects as new objects, in the same place, and selects them.
op-recover-open-desc = Opens this autosave as a new, unsaved document.
op-recover-discard-desc = Deletes this autosave.
op-recover-discard-all-desc = Deletes every autosave listed here.
op-view-frame-all-desc = Moves the view so the whole scene fits.
op-view-frame-selected-desc = Moves the view so the selection fills it.
op-view-camera-desc = Looks through the active camera, or the scene's first; again to come back.
op-view-toggle-projection-desc = Switches between perspective and orthographic views.
op-view-toggle-sidebar-desc = Shows or hides the sidebar with the selection's placement.
op-view-toggle-grid-desc = Shows or hides the floor grid and the axis lines.
op-view-toggle-wireframe-desc = Shows or hides every edge over the surfaces.
op-area-split-vertical-desc = Splits this area into two, side by side.
op-area-split-horizontal-desc = Splits this area into two, one above the other.
op-area-close-desc = Closes this area and gives its room to its neighbour.
op-mesh-select-action-0-desc = Selects every element of the mesh.
op-mesh-select-action-1-desc = Deselects everything.
op-mesh-select-action-2-desc = Selects what is not selected and deselects what is.
op-mesh-select-action-3-desc = Extends the selection along the edge loop it lies on.
op-mesh-select-action-6-desc = Grows the selection by one step.
op-mesh-select-action-7-desc = Shrinks the selection by one step.
op-mesh-select-action-8-desc = Selects every piece the selection touches.
op-object-select-all-action-0-desc = Selects every object.
op-object-select-all-action-1-desc = Deselects every object.
op-object-select-all-action-3-desc = Selects the objects that are not selected and deselects the rest.
op-object-add-primitive-shape-0-desc = Adds a flat square of one face.
op-object-add-primitive-shape-1-desc = Adds a cube of six faces.
op-object-add-primitive-shape-2-desc = Adds a flat square divided into many faces, for terrain and cloth.
op-object-add-primitive-shape-3-desc = Adds a sphere of rings and segments, with poles.
op-object-add-primitive-shape-4-desc = Adds a sphere of even triangles, with no poles.
op-object-add-primitive-shape-5-desc = Adds a cylinder with closed ends.
op-object-add-primitive-shape-6-desc = Adds a cone with a closed base.
op-object-add-primitive-shape-7-desc = Adds a ring, round in section.
op-object-add-light-kind-0-desc = Adds a light that shines in every direction from one point.
op-object-add-light-kind-1-desc = Adds a light from infinitely far away, so its rays are parallel. Only its direction matters.
op-object-add-light-kind-2-desc = Adds a light that shines in a cone.
op-object-add-light-kind-3-desc = Adds a light that shines from a rectangle, for soft shadows.
op-object-add-camera-desc = Adds a camera object and the lens behind it.
op-object-add-empty-desc = Adds an object with a place and nothing else, to parent things to or aim things at.
op-object-add-armature-desc = Adds a skeleton of one bone, to build a rig from.
op-pose-clear-transform-desc = Puts the selected bones back where the skeleton was built.
op-pose-select-all-desc = Selects every bone, none, or the others.
op-mesh-merge-by-distance-desc = Merges vertices that sit on top of one another.
op-mesh-select-desc = Changes what is selected.
op-object-select-all-desc = Changes which objects are selected.
op-mesh-shade-desc = Sets how the selected faces are shaded.
op-mesh-normals-desc = Sets which way the selected faces point.
op-object-add-light-desc = Adds a light.
op-mesh-rotate = Rotate
op-mesh-rotate-desc = Turns the selection about its own centre.
op-mesh-scale = Scale
op-mesh-scale-desc = Grows or shrinks the selection about its own centre.
op-mesh-extrude-region = Extrude
op-mesh-extrude-region-desc = Pulls the selected faces out along their shared normal, adding walls behind them.
op-mesh-inset = Inset
op-mesh-inset-desc = Adds a smaller copy of each selected face inside it.
op-mesh-bevel = Bevel
op-mesh-bevel-desc = Replaces each selected edge with a flat strip, rounding the corner.
op-mesh-subdivide = Subdivide
op-mesh-subdivide-desc = Cuts each selected face into smaller ones.
op-mesh-loop-cut = Loop Cut
op-mesh-loop-cut-desc = Adds a ring of edges around the mesh, following the quads.
op-mesh-bridge = Bridge
op-mesh-bridge-desc = Joins two open borders with a band of faces.
op-mesh-fill = Fill
op-mesh-fill-desc = Closes the selected border with a face.
op-mesh-delete = Delete
op-mesh-delete-desc = Removes the selection and everything built on it.
op-mesh-dissolve = Dissolve
op-mesh-dissolve-desc = Removes the selection but keeps the surface, merging what surrounded it.
op-mesh-merge = Merge
op-mesh-merge-desc = Collapses the selected vertices into one.
op-mesh-split = Split
op-mesh-split-desc = Detaches the selected faces from their neighbours.
op-mesh-shade-smooth-true = Shade Smooth
op-mesh-shade-smooth-true-desc = Blends the shading across the selected faces instead of faceting them.
op-mesh-shade-smooth-false = Shade Flat
op-mesh-shade-smooth-false-desc = Facets the selected faces, giving each its own flat shading.
op-mesh-mark-sharp = Mark Sharp
op-mesh-mark-sharp-desc = Keeps a crease across the selected edges however smooth the rest is.
op-mesh-normals-action-0 = Recalculate Normals
op-mesh-normals-action-0-desc = Turns every face to point outwards.
op-mesh-normals-action-2 = Flip Normals
op-mesh-normals-action-2-desc = Turns the selected faces to point the other way.
op-mesh-spin = Spin
op-mesh-spin-desc = Sweeps the selection around an axis, leaving a surface behind it.
op-uv-project-mode-0 = Project from View
op-uv-project-mode-0-desc = Flattens the selection onto the map as the viewport sees it.
op-uv-project-mode-1 = Box Project
op-uv-project-mode-1-desc = Unwraps each face onto whichever of six sides it faces most nearly.
op-uv-project-mode-2 = Sphere Project
op-uv-project-mode-2-desc = Wraps the selection on as longitude and latitude about an axis.
op-uv-project-mode-3 = Cylinder Project
op-uv-project-mode-3-desc = Wraps the selection around an axis, leaving height as it is.
op-uv-project = Project UVs
op-uv-project-desc = Lays the selection out on the texture map by projection.
op-uv-smart-project = Smart UV Project
op-uv-unwrap = Unwrap
op-uv-unwrap-desc = Cuts the selected faces open along their seams and lays them flat with the least stretch, then packs them.
op-mesh-mark-seam = Mark Seam
op-mesh-mark-seam-desc = Marks the selected edges as seams, where Unwrap cuts the surface open.
menu-mark-seam = Mark Seam
menu-clear-seam = Clear Seam
op-uv-smart-project-desc = Groups faces that face the same way, unwraps each group, and lays the pieces out on the map.
op-uv-pack = Pack Islands
op-uv-pack-desc = Rearranges the pieces already there to fit the map, leaving the cuts alone.
op-object-translate = Move
op-object-translate-desc = Moves the selected objects.
op-pose-translate = Move
op-pose-translate-desc = Moves the selected bones.
op-pose-rotate = Rotate
op-pose-rotate-desc = Turns the selected bones about their heads' middle.
op-pose-scale = Scale
op-pose-scale-desc = Scales the selected bones along their own axes.
op-pose-clear-transform-what-0 = Clear Transform
op-pose-clear-transform-what-1 = Clear Location
op-pose-clear-transform-what-2 = Clear Rotation
op-pose-clear-transform-what-3 = Clear Scale
op-armature-translate = Move
op-armature-translate-desc = Moves the selected bone ends.
op-armature-rotate = Rotate
op-armature-rotate-desc = Turns the selected bone ends about their middle.
op-armature-scale = Scale
op-armature-scale-desc = Scales the selected bone ends about their middle.
op-armature-extrude = Extrude
op-armature-bone-add = Add Bone
op-armature-delete = Delete Bones
menu-group-keys = Keyframes
menu-group-clear = Clear
menu-group-bones = Bones
op-object-rotate = Rotate
op-object-rotate-desc = Turns the selected objects.
op-object-scale = Scale
op-object-scale-desc = Grows or shrinks the selected objects.
op-object-add-primitive = Add
op-object-add-primitive-desc = Adds a new shape at the origin.
op-object-duplicate = Duplicate
op-object-duplicate-desc = Copies the selected objects. A linked copy shares its mesh, so editing one edits both.
op-object-delete = Delete
op-object-delete-desc = Removes the selected objects.
op-object-join = Join
op-object-join-desc = Merges the selected objects' meshes into the active one, leaving everything where it was on screen.
op-object-select = Select Object
op-object-select-desc = Selects an object by its name and makes it active. With extend, adds it to the selection.
menu-boolean = Boolean
op-object-boolean = Boolean
op-object-boolean-desc = Cuts the active object with the other selected meshes. Select the cutters, then the object to cut.
op-object-boolean-operation-0 = Difference
op-object-boolean-operation-0-desc = Cuts the other selected meshes out of the active object (Ctrl+minus). They show as wireframes while they cut.
op-object-boolean-operation-1 = Union
op-object-boolean-operation-1-desc = Joins the other selected meshes onto the active object as one solid (Ctrl+plus).
op-object-boolean-operation-2 = Intersect
op-object-boolean-operation-2-desc = Keeps only what the active object shares with the other selected meshes (Ctrl+*).
op-mesh-separate = Separate
op-mesh-separate-desc = Moves the selected faces into a new object. Ownership changes, not shape.
op-object-duplicate-linked-false = Duplicate
op-object-duplicate-linked-false-desc = Copies the selected objects, mesh and all, so the two can be edited apart.
op-object-duplicate-linked-true = Duplicate Linked
op-object-duplicate-linked-true-desc = Copies the selected objects but shares one mesh, so editing either edits both.
"#;

pub const KO_FTL: &str = r#"
menu-file-new = 새로
menu-file-open = 열기...
menu-file-save = 저장
menu-file-save-as = 다른 이름으로 저장...
menu-file-export = 내보내기...
menu-file-run-script = 스크립트 실행...
menu-search = 검색...
menu-copy-script = 히스토리를 스크립트로 복사
palette-placeholder = 메뉴 검색: extrude, 큐브 추가, 저장...
palette-nothing = 일치하는 항목이 없습니다.
pie-shading = 셰이딩
editor-timeline = 타임라인
editor-dope-sheet = 도프 시트
editor-graph = 그래프 에디터
editor-drivers = 드라이버
drivers-empty = 선택한 물체에 드라이버가 없습니다
drivers-add = 드라이버 추가
drivers-remove = 제거
drivers-expression = 수식
drivers-add-variable = 변수 추가
drivers-remove-variable = 제거
drivers-none = (없음)
driver-error-empty = 수식이 비어 있습니다
driver-error-syntax = { $at }번째 글자에서 문법 오류
driver-error-name = 모르는 이름: { $name }
driver-error-function = 모르는 함수: { $name }
driver-error-arguments = { $name }에는 인수가 { $count }개 필요합니다
driver-error-infinite = 결과가 유한한 수가 아닙니다
driver-error-target = “{ $name }”이(가) 읽을 물체가 없습니다
driver-error-property = “{ $name }”이(가) 읽는 속성이 없습니다
driver-error-bad-name = “{ $name }”은(는) 쓸 수 없는 이름입니다
driver-error-cycle = 드라이버들이 서로를 읽는 고리가 있습니다
graph-handles = 핸들
handle-auto = 자동 (클램프)
handle-flat = 평평하게
sheet-empty = 애니메이션이 있는 오브젝트를 선택하면 키가 보입니다.
sheet-interpolation = 보간
interp-constant = 고정
interp-linear = 선형
interp-bezier = 베지어
channel-location = 위치
channel-rotation-euler = 회전
channel-scale = 크기
channel-rotation-quaternion = 회전
channel-base-color = 베이스 컬러
channel-metallic = 메탈릭
channel-roughness = 러프니스
channel-alpha = 알파
channel-normal-strength = 노멀 세기
channel-color = 색상
channel-intensity = 세기
channel-exposure = 노출
time-play = 재생
key-property = 키프레임: 이 값을 현재 프레임에 키로 넣거나, 이미 있으면 뺍니다
time-autokey = 자동 키: 켜져 있으면 오브젝트를 옮기거나 돌리거나 크기를 바꿀 때 현재 프레임에 키가 들어갑니다
time-pause = 일시정지
time-jump-start = 처음으로
time-jump-end = 끝으로
time-start = 시작
time-end = 끝
menu-animation = 애니메이션
op-time-play = 애니메이션 재생
op-time-play-desc = 지금 프레임부터 장면을 재생하고, 범위 끝에서 처음으로 돌아갑니다. 다시 누르면 멈춥니다.
op-anim-keyframe-insert = 키프레임 삽입
op-anim-keyframe-insert-desc = 선택한 오브젝트의 위치, 회전, 크기를 현재 프레임에 키로 기록합니다.
op-anim-keyframe-delete = 키프레임 삭제
op-anim-keyframe-delete-desc = 선택한 오브젝트의 현재 프레임 키를 지웁니다.
pie-view = 보기
pie-mode = 모드
view-left = 왼쪽
view-right = 오른쪽
view-bottom = 아래
view-top = 위
view-front = 앞
view-back = 뒤
op-view-shading-studio-desc = 장면의 조명과 상관없이 형태가 잘 보이는 조명으로 보여줍니다.
op-view-shading-flat-desc = 조명 없이 각 면을 원래 색 그대로 보여줍니다.
op-view-shading-wireframe-desc = 면 없이 모서리만, 뒤쪽까지 비쳐 보여줍니다.
op-view-shading-rendered-desc = 장면의 조명과 하늘을 카메라 노출로 보여줍니다.
op-view-axis-left-desc = 왼쪽에서 +X 방향으로 피벗을 봅니다.
op-view-axis-right-desc = 오른쪽에서 -X 방향으로 피벗을 봅니다.
op-view-axis-bottom-desc = 아래에서 피벗을 올려다봅니다.
op-view-axis-top-desc = 위에서 피벗을 내려다봅니다.
op-view-axis-front-desc = 앞에서 +Y 방향으로 피벗을 봅니다.
op-view-axis-back-desc = 뒤에서 -Y 방향으로 피벗을 봅니다.
op-mode-object-desc = 오브젝트 전체에 작용합니다: 선택하고, 옮기고, 추가합니다.
op-mode-edit-desc = 편집 중인 메시의 점, 선, 면에 작용합니다.
op-mode-pose-desc = 활성 아마튜어의 뼈대에 작용합니다.
op-app-search = 검색
op-app-search-desc = 메뉴 명령의 이름 일부를 입력해 찾고, Enter로 실행합니다.
op-edit-copy-script = 히스토리를 스크립트로 복사
op-edit-copy-script-desc = 지금까지의 모든 편집을 콘솔과 스크립트 실행이 다시 돌릴 수 있는 스크립트 줄로 클립보드에 넣습니다.
op-help-contact = 개발자에게 문의하기
op-help-contact-desc = 개발자에게 문제, 원하는 기능, 질문을 보냅니다. 답장을 받으려면 이메일 주소만 적으면 됩니다.
op-help-about = Kreate 정보
op-help-about-desc = 이 Kreate의 버전과 빌드한 커밋을 보여 줍니다.
op-file-run-script = 스크립트 실행
op-file-run-script-desc = 스크립트 줄이 적힌 텍스트 파일을 실행합니다. 콘솔에 입력한 것처럼 한 줄이 되돌리기 한 단계입니다.
world-pick-image = 환경 이미지...
world-clear-image = 지우기
material-pick-image = 이미지 열기...
material-clear-image = 지우기
material-new = 새 재질
material-browse = 재질
material-assign = 지정
material-assign-hint = 선택한 면에 이 칸의 재질을 입힙니다.
material-select = 선택
material-select-hint = 이 칸의 재질을 쓰는 면을 선택합니다.
op-material-assign = 재질 지정
op-material-assign-desc = 선택한 면에 이 칸의 재질을 입힙니다.
op-material-select = 재질로 선택
op-material-select-desc = 이 칸의 재질을 쓰는 면을 선택합니다.
op-material-link = 재질 연결
op-material-link-desc = 이미 있는 재질을 이 칸에 넣습니다.
material-none = 이 물체에는 재질이 없습니다.
material-base-colour = 베이스 컬러
material-metallic-roughness = 메탈릭 / 러프니스
material-normal-map = 노멀 맵
material-empty-slot = 비어 있음
material-no-slots = 슬롯 없음
material-add-slot = 재질 슬롯 추가
material-remove-slot = 이 재질 슬롯 삭제
recover-count = 복구된 작업 { $count }개
recover-menu = 복구
recover-none = 복구할 작업 없음
recover-hint = 복구된 작업 { $count }개 — 파일 ‣ 복구
recover-open = 열기
recover-discard = 버리기
recover-discard-all = 모두 버리기
recover-age-now = 방금
recover-age-minutes = { $n }분 전
recover-age-hours = { $n }시간 전
recover-age-days = { $n }일 전
status-saved = { $path } 저장함
status-opened = { $path } 열었음
menu-add-mesh = 메시
menu-add-light = 조명
menu-add-camera = 카메라
menu-add-empty = 엠프티
menu-add-armature = 아마튜어
menu-pose = 포즈
menu-armature = 아마튜어
armature-add-bone = 뼈 추가
armature-extrude = 돌출
armature-delete = 뼈 삭제
mode-edit-armature = 편집 모드
keymap-context-edit-armature = 아마튜어 편집 모드
op-mode-edit-armature-desc = 활성 아마튜어 뼈의 머리와 꼬리에 작용합니다(만들어진 자세 그대로).
op-armature-bone-add-desc = 한 단위 높이로 선 독립된 뼈를 추가합니다.
op-armature-extrude-desc = 선택한 꼬리마다 이어진 뼈를 뽑아내고 바로 움직입니다.
op-armature-delete-desc = 선택한 뼈를 삭제합니다. 자식 뼈는 부모에게 넘어갑니다.
op-armature-select-all-desc = 모든 뼈 끝을 선택하거나, 해제하거나, 반전합니다.
pose-clear-all = 트랜스폼 지우기
pose-clear-location = 위치 지우기
pose-clear-rotation = 회전 지우기
pose-clear-scale = 스케일 지우기
mode-pose = 포즈 모드
outliner-kind-armature = 아마튜어
keymap-context-pose = 포즈 모드
menu-add-plane = 평면
menu-add-cube = 큐브
menu-add-grid = 격자
menu-add-uv-sphere = UV 구
menu-add-ico-sphere = 아이코 구
menu-add-cylinder = 원기둥
menu-add-cone = 원뿔
menu-add-torus = 토러스
outliner-scene-collection = 씬 컬렉션
menu-duplicate = 복제
menu-delete = 삭제
menu-undo = 되돌리기
menu-redo = 다시 실행
menu-extrude = 돌출
menu-subdivide = 세분
menu-dissolve = 녹이기
menu-inset = 인셋
menu-bridge = 브리지
menu-loop-cut = 루프 컷
menu-shade-smooth = 부드럽게
menu-shade-flat = 평평하게
menu-mark-sharp = 자동 샤프
menu-recalc-normals = 노멀 재계산
menu-fill = 채우기
menu-spin = 회전
menu-bevel = 베벨
menu-merge = 병합
menu-weld = 용접
menu-select-loop = 루프
pie-snap = 스냅
menu-snap = 스냅
menu-apply = 적용
apply-location = 위치
apply-rotation = 회전
apply-scale = 크기
apply-all = 모든 트랜스폼
apply-rotation-scale = 회전과 크기
menu-set-origin = 원점 설정
origin-to-geometry = 원점을 형상 중심으로
origin-to-cursor = 원점을 3D 커서로
geometry-to-origin = 형상을 원점으로
pivot-point = 기준점
pivot-median = 중앙점
orientation-global = 월드
orientation-local = 로컬
orientation-hint = 방향 기준: X·Y·Z를 처음 누를 때 따르는 축
snap-toggle = 스냅
snap-on = 스냅 켜기
snap-hint = 이동할 때 달라붙기 (Shift+Tab). Ctrl은 한 번만 반대로, 이동 중 B도 됩니다
snap-increment = 간격
snap-element = 점·선·면
snap-vertex = 점
snap-edge = 선
snap-midpoint = 선의 가운데
snap-face = 면
proportional-toggle = 비례
proportional-on = 비례 편집 켜기
proportional-hint = 비례 편집 (O): 선택한 것이 주변까지 끌고 갑니다. 휠로 범위를 정합니다
falloff-smooth = 부드럽게
falloff-sphere = 구
falloff-root = 루트
falloff-sharp = 뾰족하게
falloff-linear = 직선
falloff-constant = 일정
select-tool-box = 네모 선택
select-tool-lasso = 올가미 선택
select-tool-circle = 원 선택
select-tool-hint = 누르면 다음 모양으로: 네모, 올가미, 원
pivot-individual = 개별 원점
pivot-active = 활성 요소
pivot-cursor = 3D 커서
op-view-cursor-to-selected = 커서를 선택으로
op-view-cursor-to-selected-desc = 3D 커서를 선택의 가운데에 둡니다.
op-view-cursor-to-origin = 커서를 월드 원점으로
op-view-cursor-to-origin-desc = 3D 커서를 월드 원점으로 되돌립니다.
op-view-selection-to-cursor = 선택을 커서로
op-view-selection-to-cursor-desc = 선택의 가운데가 3D 커서에 오도록 옮깁니다.
op-view-cursor-place = 3D 커서 놓기
op-mesh-knife = 나이프
panel-shape-keys = 셰이프 키
shape-key-basis = 기본형
shape-key-from-mix = 현재 모양으로
shape-keys-editing = 편집 모드가 활성 키를 편집하고 있습니다.
op-object-shape-key-add = 셰이프 키 추가
op-object-shape-key-add-desc = 셰이프 키를 추가합니다 — 기본형에서, 또는 지금 섞인 모양에서.
op-object-shape-key-remove = 셰이프 키 제거
op-object-shape-key-set = 셰이프 키 설정
op-object-shape-key-set-desc = 셰이프 키 값을 올리거나 내립니다.
op-object-shape-key-set-active = 활성 셰이프 키 설정
op-object-shape-key-edit = 셰이프 키 편집
panel-bone-ik = IK (역운동학)
op-pose-ik-add = IK 추가
op-pose-ik-add-desc = 활성 뼈가 선택한 다른 뼈를 향해 뻗도록, 그 위의 뼈들을 돌립니다.
op-pose-ik-set = IK 설정
op-pose-ik-set-desc = 뼈의 IK 목표·체인·켜짐을 바꿉니다.
op-pose-ik-remove = IK 제거
op-pose-ik-remove-desc = 뼈의 IK를 없앱니다.
ik-enabled = 켜짐
ik-target = 목표
ik-chain = 체인
ik-pole = 폴
ik-pole-desc = 팔꿈치·무릎이 굽는 쪽을 정하는 뼈.
ik-chain-desc = 이 뼈를 포함해 몇 개의 뼈가 돌지. 0이면 맨 위까지.
menu-move-to-collection = 컬렉션으로 이동
collection-scene = 씬 컬렉션
collection-new = 새 컬렉션
op-object-move-to-collection = 컬렉션으로 이동
op-object-move-to-collection-desc = 선택한 오브젝트를 컬렉션으로 옮깁니다.
op-collection-set-visible = 컬렉션 표시/숨기기
op-collection-set-visible-desc = 컬렉션과 그 안의 모든 것을 숨기거나 다시 보이게 합니다.
op-collection-remove = 컬렉션 제거
op-collection-remove-desc = 컬렉션을 없앱니다. 안에 있던 것은 씬에 남습니다.
op-collection-rename = 컬렉션 이름 바꾸기
op-collection-rename-desc = 컬렉션 이름을 바꿉니다.
overlay-face-orientation = 면 방향
menu-undo-history = 되돌리기 기록
undo-original = 처음 상태
overlay-statistics = 통계
overlay-ao = 앰비언트 오클루전
overlay-xray = 투시
overlay-outline = 선택 외곽선
overlay-bloom = 블룸 (렌더)
overlay-reflections = 반사 (렌더)
overlay-contact-shadows = 접촉 그림자 (렌더)
overlay-path-trace = 패스 트레이싱 (렌더)
op-view-toggle-path-trace = 패스 트레이싱
op-view-toggle-path-trace-desc = 렌더 보기를 빛이 면 사이를 튕기는 방식으로 그립니다. 화면이 멈추면 점점 깨끗해집니다.
op-view-toggle-reflections = 반사
op-view-toggle-reflections-desc = 렌더 보기에서 매끈한 면이 화면에 보이는 것을 비춥니다.
op-view-toggle-contact-shadows = 접촉 그림자
op-view-toggle-contact-shadows-desc = 렌더 보기에서 물체가 닿는 곳의 마지막 몇 센티미터를 그림자 맵보다 촘촘하게 어둡게 합니다.
op-view-toggle-bloom = 블룸
op-render-image = 이미지 렌더
op-render-image-desc = 씬 카메라로 렌더 탭의 크기와 샘플 수만큼 패스 트레이싱해 저장합니다. Esc로 멈춥니다.
op-render-animation = 애니메이션 렌더
op-render-animation-desc = 프레임 범위의 모든 프레임을 씬 카메라로 렌더해 번호를 붙여 저장합니다. Esc로 멈춥니다.
op-render-view = 렌더 결과 보기
op-render-view-desc = 마지막 렌더 결과를 다시 보여 줍니다.
op-view-toggle-bloom-desc = 렌더 보기에서 흰색보다 밝은 빛이 주변으로 번집니다.
op-view-toggle-outline = 선택 외곽선
op-view-toggle-outline-desc = 선택한 오브젝트 둘레에 주황 선을 그립니다. 활성 오브젝트는 더 밝게.
op-view-toggle-xray = 투시
op-light-set_properties-param-kind-name = 종류
op-light-set_properties-param-color-name = 색
op-light-set_properties-param-intensity-name = 세기
op-light-set_properties-param-exposure-name = 노출
op-light-set_properties-param-radius-name = 반지름
op-light-set_properties-param-angular_diameter-name = 겉보기 지름
op-light-set_properties-param-cone_angle-name = 원뿔 각도
op-light-set_properties-param-blend-name = 가장자리 흐림
op-light-set_properties-param-shape-name = 모양
op-light-set_properties-param-width-name = 너비
op-light-set_properties-param-height-name = 높이
op-light-set_properties-param-spread-name = 퍼짐
op-light-set_properties-param-cast_shadow-name = 그림자
op-light-set_properties-param-affect_diffuse-name = 확산광에 적용
op-light-set_properties-param-affect_specular-name = 반사광에 적용
op-view-toggle-xray-desc = 표면을 비쳐 보이게 하고, 뒤에 있는 것도 선택되게 합니다 (Alt+Z).
op-view-toggle-ao = 앰비언트 오클루전
op-view-toggle-ao-desc = 스튜디오 보기에서 모서리와 틈을 어둡게 해 형태가 잘 보이게 합니다.
op-view-toggle-face-orientation = 면 방향
op-view-toggle-face-orientation-desc = 면의 앞쪽은 파랑, 뒤쪽은 빨강으로 칠합니다.
op-view-toggle-statistics = 통계
op-view-toggle-statistics-desc = 오브젝트·정점·엣지·면 수를 보여 줍니다.
stats-objects = 오브젝트
stats-vertices = 정점
stats-edges = 엣지
stats-faces = 면
stats-triangles = 삼각형
op-view-circle-select = 원형 선택
op-view-circle-select-desc = 지나가는 것을 선택하는 브러시입니다. Shift는 해제, 휠은 크기, Esc나 우클릭으로 끝냅니다.
op-mesh-knife-desc = 표면 위에 클릭한 경로를 따라 새 엣지를 자릅니다. Enter로 자르고 Esc로 취소합니다.
op-view-cursor-place-desc = 포인터 아래 표면에 3D 커서를 둡니다.
op-scene-cursor-set = 3D 커서 설정
op-scene-cursor-set-desc = 새 물체가 생기는 3D 커서를 옮깁니다.
op-object-origin-set = 원점 설정
op-object-origin-set-desc = 그려진 모양은 그대로 두고 선택한 메시의 원점을 옮깁니다.
op-object-transform-apply = 트랜스폼 적용
op-object-transform-apply-desc = 트랜스폼을 메시에 굳히고 초기화합니다. 화면의 모습은 그대로입니다.
menu-select-more = 넓히기
menu-select-less = 좁히기
menu-select-linked = 연결된 것
menu-show-hide = 표시/숨기기
menu-hide-selected = 선택 숨기기
menu-hide-unselected = 선택 안 된 것 숨기기
menu-reveal = 숨긴 것 보이기
op-mesh-hide = 숨기기
op-mesh-hide-desc = 편집 모드에서 선택한 것을 안 보이게 합니다.
op-mesh-reveal = 숨긴 것 보이기
op-mesh-reveal-desc = 숨긴 것을 모두 다시 보이게 하고 선택합니다.
mode-object = 오브젝트 모드
mode-edit = 편집 모드
select-mode-vert = 정점
select-mode-edge = 모서리
select-mode-face = 면
keymap-preset = 프리셋
keymap-changed = { $count }개 변경됨
keymap-reset-all = 전부 초기화
keymap-shadowed = { $count }개가 가려짐
keymap-reset-hint = 프리셋 값으로 되돌리기
keymap-shadowed-hint = 다른 바인딩이 이 조합을 먼저 가져갑니다
keymap-capture-hint = 클릭한 뒤 원하는 키를 누르세요
keymap-capturing = 키를 누르세요…
keymap-context-global = 전역
keymap-context-viewport = 뷰포트
keymap-context-object = 오브젝트 모드
keymap-context-edit = 편집 모드
keymap-context-uv = UV 편집기
keymap-context-modal = 모달 툴
uv-none = 이 메시에는 아직 UV가 없습니다. U를 누르거나 오른클릭 메뉴에서 펼치세요.
area-menu = 영역
sidebar-item = 항목
tab-render = 렌더
menu-render = 렌더
render-image = 이미지 렌더
render-animation = 애니메이션 렌더
render-view = 렌더 결과 보기
render-result-title = 렌더 결과
render-hint = 씬 카메라로 패스 트레이서 렌더를 합니다. Esc로 멈춥니다.
render-saved = 저장됨: { $path }
render-failed = 저장 안 됨: { $why }
sidebar-tool = 도구
sidebar-view = 보기
sidebar-dimensions = 치수
sidebar-lens = 초점 거리 (mm)
sidebar-cursor = 3D 커서
mode-weight-paint = 웨이트 페인트
keymap-context-weight-paint = 웨이트 페인트
keymap-context-sculpt = 스컬프트
mode-sculpt = 스컬프트 모드
op-mode-sculpt-desc = 붓으로 표면을 밀고 당깁니다.
menu-sculpt = 스컬프트
sculpt-brush = 붓
sculpt-draw = 그리기
sculpt-smooth = 매끄럽게
sculpt-grab = 잡기
sculpt-inflate = 부풀리기
sculpt-mirror-x = X 대칭
op-sculpt-stroke = 스컬프트
op-sculpt-stroke-desc = 스컬프트 붓의 한 획.
op-sculpt-brush = 스컬프트 붓
op-sculpt-brush-desc = 스컬프트 붓을 고릅니다.
op-mode-weight-paint-desc = 각 정점이 각 뼈를 얼마나 따를지 칠합니다.
menu-weights = 웨이트
op-object-vertex-group-add = 버텍스 그룹 추가
op-object-vertex-group-add-desc = 빈 버텍스 그룹을 추가하고 칠할 그룹으로 정합니다.
op-object-vertex-group-remove = 버텍스 그룹 제거
op-paint-weights-normalize = 모두 정규화
op-paint-weights-normalize-desc = 각 정점의 웨이트 합이 1이 되도록 맞춥니다.
panel-vertex-groups = 버텍스 그룹
groups-none = 버텍스 그룹 없음
paint-weight = 웨이트
paint-radius = 반경
paint-strength = 세기
paint-front-only = 앞면만
paint-falloff = 부드러운 가장자리
menu-parent = 부모
menu-select-similar = 비슷한 것 선택
op-mesh-select-shortest-path = 최단 경로 선택
op-mesh-select-shortest-path-desc = 선택한 것에서 클릭한 곳까지 가장 짧은 길을 선택합니다 (Ctrl+클릭).
similar-face-normal = 같은 방향
similar-face-area = 같은 넓이
similar-face-coplanar = 같은 평면
similar-face-material = 같은 재질
similar-face-sides = 같은 변의 수
similar-edge-length = 같은 길이
similar-edge-direction = 같은 방향
similar-vert-normal = 같은 방향
similar-vert-faces = 같은 면 수
op-mesh-select-similar = 비슷한 것 선택
op-mesh-select-similar-desc = 선택한 것과 비슷한 것을 모두 선택합니다 (Shift+G).
menu-file-open-recent = 최근 파일 열기
recent-none = 아직 연 파일이 없습니다
reveal-hidden = 숨긴 것 보이기
op-object-hide = 숨기기
op-object-hide-desc = 선택한 오브젝트를 숨깁니다 (H).
op-object-hide-unselected-desc = 선택하지 않은 것을 모두 숨깁니다 (Shift+H).
op-object-reveal = 숨긴 것 보이기
op-object-reveal-desc = 숨긴 오브젝트를 모두 보이고 선택합니다 (Alt+H).
op-object-parent-one = 부모 지정
op-object-parent-one-desc = 한 오브젝트를 다른 오브젝트의 자식으로 만듭니다 — 아웃라이너의 끌어다 놓기.
menu-parent-to = 부모 지정
parent-object = 오브젝트
parent-clear = 부모 해제
outliner-rename = 이름 바꾸기
outliner-remove-collection = 컬렉션 삭제
op-app-rename = 이름 바꾸기
op-app-rename-desc = 활성 오브젝트의 이름을 바꿉니다 (F2).
op-object-parent = 부모 지정
op-object-parent-desc = 선택한 다른 오브젝트들의 부모를 활성 오브젝트로 정합니다. 위치는 그대로입니다.
op-object-parent-clear-desc = 선택한 오브젝트를 부모에게서 떼어 냅니다. 위치는 그대로입니다.
op-object-rename = 이름 바꾸기
op-object-rename-desc = 활성 오브젝트의 이름을 바꿉니다.
bind-automatic = 자동 웨이트로
bind-empty = 빈 그룹으로
unbind-armature = 아마튜어 변형 제거
panel-relations = 관계
relations-parent = 부모
relations-armature = 아마튜어
op-object-bind-armature-desc = 선택한 메시를 활성 아마튜어의 자식으로 만들고 뼈를 따라 움직이게 합니다.
op-object-bind-armature-weights-0-desc = 선택한 메시를 활성 아마튜어에 붙이고, 각 정점이 뼈에 얼마나 가까운지로 웨이트를 정합니다.
op-object-bind-armature-weights-1-desc = 선택한 메시를 활성 아마튜어에 붙이고, 뼈마다 칠할 빈 그룹을 만듭니다.
op-object-unbind-armature-desc = 선택한 메시가 아마튜어를 따르지 않게 합니다. 웨이트는 남깁니다.
bone-parent = 부모
workspace-layout = 레이아웃
workspace-modeling = 모델링
workspace-uv-editing = UV 편집
workspace-shading = 셰이딩
workspace-animation = 애니메이션
workspace-scripting = 스크립팅
field-insert-key = 키프레임 넣기
field-delete-key = 키프레임 지우기
field-add-driver = 드라이버 추가
field-remove-driver = 드라이버 제거
field-reset = 기본값으로
tab-bone = 뼈
panel-pose = 포즈
panel-bone-rest = 뼈
panel-viewport-display = 뷰포트 표시
sidebar-nothing = 활성 오버젝트가 없습니다. 오버젝트를 선택하면 위치가 보입니다.
area-split-vertical = 좌우로 나누기
area-split-horizontal = 위아래로 나누기
area-close = 영역 닫기
editor-viewport = 3D 뷰포트
editor-outliner = 아웃라이너
editor-properties = 특성
editor-uv = UV 에디터
editor-preferences = 환경 설정
editor-console = 콘솔
editor-claude = AI 채팅
editor-shader = 셰이더 에디터
material-uses-nodes = 이 재질은 노드를 사용합니다. 셰이더 에디터에서 편집하세요.
node-use-nodes = 노드 사용
node-use-nodes-hint = 이 재질의 현재 모습 그대로 노드 그래프를 만듭니다
node-no-graph = 이 재질에는 아직 노드 그래프가 없습니다.
node-no-material = 활성 오브젝트의 이 슬롯에 재질이 없습니다.
node-add = 추가
node-no-image = 이미지 없음
node-open-image = 이미지 열기...
node-kind-color = RGB
node-kind-value = 값
node-kind-image = 이미지 텍스처
node-kind-multiply = 혼합 (곱하기)
node-kind-multiply-value = 수학 (곱하기)
node-kind-separate-rgb = RGB 분리
node-kind-normal-map = 노멀 맵
node-socket-base-color = 베이스 컬러
node-socket-metallic = 메탈릭
node-socket-roughness = 러프니스
node-socket-normal = 노멀
node-socket-color = 색상
node-socket-value = 값
node-socket-alpha = 알파
node-socket-a = A
node-socket-b = B
node-socket-r = R
node-socket-g = G
node-socket-strength = 세기
console-placeholder = kreate.ops.mesh.extrude_region(offset=1.0)
console-empty = 편집하는 모든 것이 여기에 스크립트 한 줄로 나타납니다. 한 줄 입력해 실행해 보세요.
chat-placeholder = 만들거나 바꿀 것을 말해 보세요…
chat-empty = 장면에 대해 묻거나 바꿔 달라고 해 보세요. 위에서 Claude, Gemini, ChatGPT 중에 고를 수 있고, AI가 하는 편집은 하나하나 실행 취소할 수 있습니다.
chat-send = 보내기
chat-stop = 중지
chat-new = 새 대화
chat-working = 작업 중…
chat-stopped = 중지했습니다.
chat-checking = 계정 확인 중…
chat-signed-out = 로그인되어 있지 않습니다.
chat-sign-in = 로그인
chat-signing-in = 열린 브라우저에서 로그인을 마치세요.
chat-open-sign-in = 로그인 페이지 열기
chat-code-hint = 페이지에 코드가 나오면 여기에 붙여 넣으세요
chat-code-submit = 확인
chat-cancel = 취소
sculpt-coarse = 이 메시는 정점이 { $count }개라 브러시로 모양을 만들기에 너무 적습니다. 먼저 서브디비전 모디파이어를 적용하거나 편집 모드에서 세분화(Subdivide)로 정점을 늘리세요.
menu-help = 도움말
help-contact = 개발자에게 문의하기…
help-about = Kreate 정보
contact-title = 개발자에게 문의하기
contact-intro = 문제를 알리거나 기능을 요청하거나 질문을 남겨 주세요. 답장을 받으려면 이메일 주소를 적어 주세요.
contact-kind-bug = 버그
contact-kind-feature = 기능 요청
contact-kind-question = 질문
contact-kind-other = 기타
contact-name = 이름
contact-email = 이메일
contact-email-hint = 선택, 답장 받을 주소
contact-message-hint = 무슨 일이 있었는지, 무엇을 원하시는지 적어 주세요…
contact-details = 버전과 플랫폼 정보 함께 보내기
contact-send = 보내기
contact-close = 닫기
contact-sending = 보내는 중…
contact-sent = 보냈습니다. 감사합니다!
about-title = Kreate 정보
about-version = 버전
splash-title = Kreate
splash-shaders = 렌더러 준비 중…
splash-warming = 셰이더 컴파일 중…
splash-opening = 문서 여는 중…
splash-version = 버전 { $version }
uv-not-editing = UV 에디터는 편집 중인 메시를 보여줍니다. Tab을 눌러 편집 모드로 들어가세요.
adjust-nothing = 조정할 작업이 없습니다
status-counts = 오브젝트 { $objects }개, 메시 { $meshes }개, 선택 { $selected }개
outliner-kind-empty = 빈 오브젝트
outliner-kind-mesh = 메시
outliner-kind-camera = 카메라
outliner-kind-light = 조명
outliner-shared-mesh = 오브젝트 { $count }개가 공유
outliner-mesh-summary = { $name }: 정점 { $verts }, 면 { $faces }
shading-studio = 스튜디오
shading-flat = 플랫
shading-wireframe = 와이어프레임
shading-rendered = 렌더
shading-hint = 뷰포트 셰이딩 방식 (Z)
exposure = EV
exposure-metered = 씨을 측광해 정합니다. 램프를 줄이면 이 숫자가 움직이고 그림은 그대로입니다 — 그것이 노출의 뜻입니다. 카메라에서 끄면 직접 정합니다.
exposure-manual = 카메라의 조리개·셔터·ISO에서 나온 값입니다.
tab-tool = 도구
tab-object = 오브젝트
tab-data = 데이터
tab-modifiers = 모디파이어
tab-data-light = 라이트
tab-data-camera = 카메라
tab-material = 재질
tab-world = 월드
world-image = 배경이 이미지입니다. 아래 색은 적용되지 않습니다.
light-shared = { $count }개의 오브젝트가 공유 — 여기서 고치면 전부 바뀍니다
menu-file = 파일
menu-edit = 편집
menu-view = 보기
menu-select = 선택
menu-transform = 변환
select-all = 모두
select-none = 없음
select-invert = 반전
view-frame-all = 전체 보기
view-frame-selected = 선택 항목 보기
view-camera = 카메라 시점
view-perspective = 원근/직교
view-sidebar = 사이드바
overlays = 오버레이
overlay-grid = 격자
overlay-wireframe = 와이어프레임
transform-move = 이동
transform-rotate = 회전
transform-scale = 크기
view-user-perspective = 사용자 원근
view-user-orthographic = 사용자 직교
panel-transform = 변환
panel-surface = 표면
panel-light = 라이트
outliner-hide = 뷰포트에서 숨기기
outliner-show = 뷰포트에 보이기
tool-select = 박스 선택
tool-move = 이동
tool-rotate = 회전
tool-scale = 크기
tool-extrude = 영역 돌출
tool-inset = 면 인셋
tool-bevel = 베벨
tool-loop-cut = 루프 잘라내기
menu-object = 물체
menu-mesh = 메시
menu-edge = 모서리
menu-face = 면
menu-add = 추가
object-copy = 복사
object-paste = 붙여넣기
menu-add-light-point = 점광원
menu-add-light-sun = 태양
menu-add-light-spot = 스포트라이트
menu-add-light-area = 면광원
panel-modifiers = 모디파이어
modifier-none = 모디파이어 없음
modifier-add = 모디파이어 추가
modifier-remove = 제거
modifier-move-up = 입력 쪽으로 이동
modifier-move-down = 결과 쪽으로 이동
modifier-enabled = 이 모디파이어 사용
modifier-apply = 적용: 메시에 굳히고 스택에서 뺍니다
op-object-modifier-set-boolean-object = 오브젝트
op-object-modifier-apply = 모디파이어 적용
op-object-modifier-apply-desc = 모디파이어를 메시에 굳히고 스택에서 뺍니다.
modifier-mirror = 미러
modifier-subdivide = 서브디비전
modifier-array = 배열
modifier-solidify = 두께
modifier-bevel = 베벨
modifier-boolean = 불리언

# --- 오른클릭 메뉴 그룹 ---
menu-group-transform = 변형
menu-group-shape = 형상
menu-group-remove = 제거
menu-group-shading = 셰이딩
menu-group-uv = UV
menu-group-object = 오브젝트

# --- 오퍼레이터 이름과 설명 ---
op-mesh-translate = 이동
op-mesh-translate-desc = 선택한 점, 선, 면을 옮깁니다.
op-file-new-desc = 빈 문서를 시작합니다. 저장하지 않은 작업이 있으면 먼저 묻습니다.
op-file-open-desc = .kreate 파일을 열어 지금 문서를 대체합니다.
op-file-save-desc = 문서를 파일에 씁니다. 아직 파일이 없으면 어디에 쓸지 묻습니다.
op-file-save-as-desc = 문서를 새 파일에 쓰고, 이후로는 그 파일로 작업합니다.
op-file-export-desc = 장면을 교환 형식으로 내보냅니다. 문서의 파일은 그대로입니다.
op-edit-undo-desc = 마지막 변경을 되돌립니다.
op-edit-redo-desc = 되돌린 변경을 다시 적용합니다.
op-edit-copy-desc = 선택한 물체를 메시·재질·텍스처와 함께 복사합니다.
op-edit-paste-desc = 복사한 물체를 같은 자리에 새 물체로 추가하고 선택합니다.
op-recover-open-desc = 이 자동 저장을 저장되지 않은 새 문서로 엽니다.
op-recover-discard-desc = 이 자동 저장을 삭제합니다.
op-recover-discard-all-desc = 여기 있는 자동 저장을 모두 삭제합니다.
op-view-frame-all-desc = 장면 전체가 들어오게 시점을 옮깁니다.
op-view-frame-selected-desc = 선택한 것이 화면을 채우게 시점을 옮깁니다.
op-view-camera-desc = 활성 카메라(없으면 첫 카메라)로 봅니다. 다시 누르면 돌아옵니다.
op-view-toggle-projection-desc = 원근 시점과 직교 시점을 전환합니다.
op-view-toggle-sidebar-desc = 선택 항목의 위치를 보여주는 사이드바를 켜거나 끕니다.
op-view-toggle-grid-desc = 바닥 격자와 축 선을 켜거나 끕니다.
op-view-toggle-wireframe-desc = 표면 위의 모든 모서리를 켜거나 끕니다.
op-area-split-vertical-desc = 이 영역을 좌우 둘로 나눕니다.
op-area-split-horizontal-desc = 이 영역을 위아래 둘로 나눕니다.
op-area-close-desc = 이 영역을 닫고 그 자리를 이웃 영역에 줍니다.
op-mesh-select-action-0-desc = 메시의 모든 요소를 선택합니다.
op-mesh-select-action-1-desc = 모두 선택 해제합니다.
op-mesh-select-action-2-desc = 선택된 것은 해제하고, 안 된 것은 선택합니다.
op-mesh-select-action-3-desc = 선택이 놓인 모서리 루프를 따라 선택을 넓힙니다.
op-mesh-select-action-6-desc = 선택을 한 칸 넓힙니다.
op-mesh-select-action-7-desc = 선택을 한 칸 좁힙니다.
op-mesh-select-action-8-desc = 선택이 닿은 조각 전체를 선택합니다.
op-object-select-all-action-0-desc = 모든 물체를 선택합니다.
op-object-select-all-action-1-desc = 모든 물체를 선택 해제합니다.
op-object-select-all-action-3-desc = 선택되지 않은 물체를 선택하고 나머지는 해제합니다.
op-object-add-primitive-shape-0-desc = 면 하나짜리 평평한 사각형을 추가합니다.
op-object-add-primitive-shape-1-desc = 여섯 면의 큐브를 추가합니다.
op-object-add-primitive-shape-2-desc = 여러 면으로 나뉜 평평한 사각형을 추가합니다. 지형이나 천에 씁니다.
op-object-add-primitive-shape-3-desc = 링과 세그먼트로 이루어진, 극이 있는 구를 추가합니다.
op-object-add-primitive-shape-4-desc = 균일한 삼각형으로 이루어진, 극이 없는 구를 추가합니다.
op-object-add-primitive-shape-5-desc = 양 끝이 막힌 원기둥을 추가합니다.
op-object-add-primitive-shape-6-desc = 바닥이 막힌 원뿔을 추가합니다.
op-object-add-primitive-shape-7-desc = 단면이 둥근 고리를 추가합니다.
op-object-add-light-kind-0-desc = 한 점에서 모든 방향으로 비추는 조명을 추가합니다.
op-object-add-light-kind-1-desc = 무한히 멀리서 오는, 광선이 평행한 조명을 추가합니다. 방향만 의미가 있습니다.
op-object-add-light-kind-2-desc = 원뿔 모양으로 비추는 조명을 추가합니다.
op-object-add-light-kind-3-desc = 사각형 면에서 비추는 조명을 추가합니다. 그림자가 부드럽습니다.
op-object-add-camera-desc = 카메라 물체와 그 뒤의 렌즈를 추가합니다.
op-object-add-empty-desc = 위치만 있는 물체를 추가합니다. 다른 것의 부모나 목표로 씁니다.
op-object-add-armature-desc = 뼈 하나짜리 골격을 추가합니다. 여기서 리그를 만듭니다.
op-pose-clear-transform-desc = 선택한 뼈를 만들어진 자리로 되돌립니다.
op-pose-select-all-desc = 모든 뼈를 선택하거나, 해제하거나, 반전합니다.
op-mesh-merge-by-distance-desc = 서로 겹쳐 있는 정점들을 하나로 합칩니다.
op-mesh-select-desc = 선택을 바꿉니다.
op-object-select-all-desc = 어느 물체가 선택되는지 바꿉니다.
op-mesh-shade-desc = 선택한 면의 셰이딩 방식을 정합니다.
op-mesh-normals-desc = 선택한 면이 향하는 방향을 정합니다.
op-object-add-light-desc = 조명을 추가합니다.
op-mesh-rotate = 회전
op-mesh-rotate-desc = 선택 영역을 그 중심을 기준으로 돌립니다.
op-mesh-scale = 크기
op-mesh-scale-desc = 선택 영역을 그 중심을 기준으로 키우거나 줄입니다.
op-mesh-extrude-region = 돌출
op-mesh-extrude-region-desc = 선택한 면을 법선 방향으로 밀어내고 그 뒤에 벽을 만듭니다.
op-mesh-inset = 인셋
op-mesh-inset-desc = 선택한 각 면 안쪽에 더 작은 면을 만듭니다.
op-mesh-bevel = 베벨
op-mesh-bevel-desc = 선택한 엣지를 납작한 띠로 바꿔 모서리를 깎습니다.
op-mesh-subdivide = 세분
op-mesh-subdivide-desc = 선택한 면을 더 작은 면들로 나눕니다.
op-mesh-loop-cut = 루프 컷
op-mesh-loop-cut-desc = 쿼드를 따라 메시를 한 바퀴 도는 엣지 링을 넣습니다.
op-mesh-bridge = 브리지
op-mesh-bridge-desc = 열린 경계 둘을 면의 띠로 잇습니다.
op-mesh-fill = 채우기
op-mesh-fill-desc = 선택한 경계를 면으로 막습니다.
op-mesh-delete = 삭제
op-mesh-delete-desc = 선택 영역과 그 위에 세워진 것을 함께 지웁니다.
op-mesh-dissolve = 녹이기
op-mesh-dissolve-desc = 선택 영역을 지우되 표면은 남기고 주변을 합칩니다.
op-mesh-merge = 병합
op-mesh-merge-desc = 선택한 정점들을 하나로 모읍니다.
op-mesh-split = 분리
op-mesh-split-desc = 선택한 면을 이웃에서 떼어냅니다.
op-mesh-shade-smooth-true = 부드럽게
op-mesh-shade-smooth-true-desc = 선택한 면들의 셰이딩을 각지게 두지 않고 이어서 섞습니다.
op-mesh-shade-smooth-false = 평평하게
op-mesh-shade-smooth-false-desc = 선택한 면을 각지게 하여 면마다 자기 셰이딩을 갖게 합니다.
op-mesh-mark-sharp = 샤프 표시
op-mesh-mark-sharp-desc = 나머지가 아무리 부드러워도 선택한 엣지에는 각을 남깁니다.
op-mesh-normals-action-0 = 노멀 재계산
op-mesh-normals-action-0-desc = 모든 면이 바깥을 향하도록 돌립니다.
op-mesh-normals-action-2 = 노멀 뒤집기
op-mesh-normals-action-2-desc = 선택한 면이 반대쪽을 향하도록 돌립니다.
op-mesh-spin = 스핀
op-mesh-spin-desc = 선택 영역을 축 둘레로 쓸어 그 자취를 면으로 남깁니다.
op-uv-project-mode-0 = 뷰에서 펼치기
op-uv-project-mode-0-desc = 뷰포트가 보는 그대로 선택 영역을 맵 위에 펼칩니다.
op-uv-project-mode-1 = 박스 펼치기
op-uv-project-mode-1-desc = 면마다 가장 가깝게 향하는 여섯 면 중 하나로 펼칩니다.
op-uv-project-mode-2 = 구 펼치기
op-uv-project-mode-2-desc = 축을 기준으로 경도와 위도로 감습니다.
op-uv-project-mode-3 = 실린더 펼치기
op-uv-project-mode-3-desc = 높이는 그대로 둔 채 축 둘레로 감습니다.
op-uv-project = UV 펼치기
op-uv-project-desc = 선택 영역을 투사로 텍스처 맵 위에 배열합니다.
op-uv-smart-project = 스마트 UV 펼치기
op-uv-unwrap = 펼치기
op-uv-unwrap-desc = 선택한 면을 심을 따라 잘라 늘어남이 가장 적게 평평하게 편 뒤 배치합니다.
op-mesh-mark-seam = 심 표시
op-mesh-mark-seam-desc = 선택한 엣지를 심으로 표시합니다. 펼치기가 여기서 표면을 자릅니다.
menu-mark-seam = 심 표시
menu-clear-seam = 심 지우기
op-uv-smart-project-desc = 각도가 비슷한 면끼리 묶어 조각별로 펼친 뒤, 조각들을 맵 안에 배치합니다.
op-uv-pack = 아일랜드 배치
op-uv-pack-desc = 자른 자리는 그대로 두고, 이미 있는 조각들만 맵 안에 다시 정렬합니다.
op-object-translate = 이동
op-object-translate-desc = 선택한 오브젝트를 옮깁니다.
op-pose-translate = 이동
op-pose-translate-desc = 선택한 뼈를 옮깁니다.
op-pose-rotate = 회전
op-pose-rotate-desc = 선택한 뼈를 머리들의 가운데를 중심으로 돌립니다.
op-pose-scale = 크기
op-pose-scale-desc = 선택한 뼈를 각자의 축으로 늘이거나 줄입니다.
op-pose-clear-transform-what-0 = 트랜스폼 지우기
op-pose-clear-transform-what-1 = 위치 지우기
op-pose-clear-transform-what-2 = 회전 지우기
op-pose-clear-transform-what-3 = 크기 지우기
op-armature-translate = 이동
op-armature-translate-desc = 선택한 뼈 끝을 옮깁니다.
op-armature-rotate = 회전
op-armature-rotate-desc = 선택한 뼈 끝을 가운데를 중심으로 돌립니다.
op-armature-scale = 크기
op-armature-scale-desc = 선택한 뼈 끝을 가운데를 중심으로 늘이거나 줄입니다.
op-armature-extrude = 돌출
op-armature-bone-add = 뼈 추가
op-armature-delete = 뼈 삭제
menu-group-keys = 키프레임
menu-group-clear = 지우기
menu-group-bones = 뼈
op-object-rotate = 회전
op-object-rotate-desc = 선택한 오브젝트를 돌립니다.
op-object-scale = 크기
op-object-scale-desc = 선택한 오브젝트를 키우거나 줄입니다.
op-object-add-primitive = 추가
op-object-add-primitive-desc = 원점에 새 도형을 만듭니다.
op-object-duplicate = 복제
op-object-duplicate-desc = 선택한 오브젝트를 복사합니다. 링크드 복제는 메시를 공유하므로 한쪽을 고치면 양쪽이 바뀝니다.
op-object-delete = 삭제
op-object-delete-desc = 선택한 오브젝트를 지웁니다.
op-object-join = 합치기
op-object-join-desc = 선택한 오브젝트의 메시를 활성 오브젝트로 합칩니다. 화면 위 위치는 그대로입니다.
op-object-select = 오브젝트 선택
op-object-select-desc = 이름으로 오브젝트를 선택하고 활성으로 만듭니다. extend를 켜면 기존 선택에 더합니다.
menu-boolean = 불리언
op-object-boolean = 불리언
op-object-boolean-desc = 활성 오브젝트를 나머지 선택한 메시로 깎습니다. 깎을 도구를 먼저, 깎일 오브젝트를 마지막에 선택하세요.
op-object-boolean-operation-0 = 빼기
op-object-boolean-operation-0-desc = 나머지 선택한 메시만큼 활성 오브젝트를 파냅니다 (Ctrl+-). 깎는 도구는 와이어프레임으로 보입니다.
op-object-boolean-operation-1 = 합집합
op-object-boolean-operation-1-desc = 나머지 선택한 메시를 활성 오브젝트와 하나의 덩어리로 합칩니다 (Ctrl++).
op-object-boolean-operation-2 = 교집합
op-object-boolean-operation-2-desc = 활성 오브젝트와 나머지 선택한 메시가 겹치는 부분만 남깁니다 (Ctrl+*).
op-mesh-separate = 분리
op-mesh-separate-desc = 선택한 면을 새 오브젝트로 옮깁니다. 모양이 아니라 소유가 바뀝니다.
op-object-duplicate-linked-false = 복제
op-object-duplicate-linked-false-desc = 선택한 오브젝트를 메시까지 복사해 서로 따로 편집할 수 있게 합니다.
op-object-duplicate-linked-true = 링크드 복제
op-object-duplicate-linked-true-desc = 선택한 오브젝트를 복사하되 메시는 공유하므로 한쪽을 고치면 양쪽이 바뀝니다.
"#;

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

#[cfg(test)]
mod tests {
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
        assert_eq!(available_tabs(&doc), vec![PropertiesTab::Tool, PropertiesTab::Render]);
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
            "chat-signing-in",
            "chat-open-sign-in",
            "chat-code-hint",
            "chat-code-submit",
            "chat-cancel",
            "sculpt-coarse",
            "menu-help",
            "help-contact",
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
    const ALL_TABS: [PropertiesTab; 7] = [
        PropertiesTab::Tool,
        PropertiesTab::Object,
        PropertiesTab::Data,
        PropertiesTab::Bone,
        PropertiesTab::Material,
        PropertiesTab::World,
        PropertiesTab::Modifiers,
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
                "mode-sculpt"
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
        assert_eq!(ids, vec!["mode-object", "mode-edit-armature", "mode-pose"]);
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
