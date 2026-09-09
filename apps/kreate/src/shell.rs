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
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
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
    /// The camera's rotation, for drawing the navigation gizmo. Interface state
    /// reads the camera; it never writes to it.
    pub camera_rotation: kreate_math::DQuat,
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
    /// Whether the remapping panel is open.

    /// The binding waiting for a keystroke, if any.
    ///
    /// Lives here rather than in the panel because the application, not the
    /// interface, is what sees the key event — the panel only says which binding
    /// is listening.
    pub capturing: Option<kreate_input::BindingId>,
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
}

/// A layout change, queued during a frame and applied after it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum AreaEdit {
    SetEditor(kreate_ui::AreaId, kreate_ui::EditorKind),
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
    pub fn set_area_editor(&mut self, area: kreate_ui::AreaId, kind: kreate_ui::EditorKind) {
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
            .filter(|(_, kind)| *kind == kreate_ui::EditorKind::Viewport)
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

    /// Whether a viewport area is showing its sidebar.
    pub fn sidebar_open(&self, area: kreate_ui::AreaId) -> bool {
        self.sidebars.contains(&area)
    }

    /// Whether any area is showing this editor.
    ///
    /// What the application asks instead of reading a `show_*` boolean. It is
    /// deliberately a question about the layout rather than about a setting:
    /// there is no separate notion of an editor being "on".
    pub fn has_editor(&self, kind: kreate_ui::EditorKind) -> bool {
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
            .filter(|(_, kind)| *kind == kreate_ui::EditorKind::UvEditor)
            .find_map(|(id, _)| {
                let state = self.uv_views.get(&id)?;
                state.rect.filter(|r| r.contains(at)).map(|_| (id, state))
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
                        .unwrap_or(kreate_ui::EditorKind::Viewport);
                    self.screen_mut().split(area, axis, 0.5, kind);
                }
                AreaEdit::Close(area) => {
                    if self.screen_mut().close(area) {
                        // The state of an area that no longer exists would
                        // otherwise accumulate for the life of the session.
                        self.uv_views.remove(&area);
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
            catalog: Catalog::new(),
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
            gizmo_circle: None,
            viewport_blockers: Vec::new(),
            sidebars: std::collections::BTreeSet::new(),
            outliner: OutlinerOrder::default(),
            select_band: None,
            select_band_deselect: false,
            context_menu: ContextMenu::default(),
            command_info: std::collections::BTreeMap::new(),
            status: None,
            recoverable: Vec::new(),
            capturing: None,
            keymap_edits: Vec::new(),
            uv_views: std::collections::BTreeMap::new(),
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
        !self.egui_wants_keyboard && !self.ime_active && !self.is_capturing()
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
        self.ids.reserve(hierarchy.len());
        self.depths.reserve(hierarchy.len());
        for (id, depth) in hierarchy {
            self.ids.push(object_key(id));
            self.depths.push(depth.min(u16::MAX as usize) as u16);
        }
        self.mesh_users = doc.mesh_user_counts();
        self.version = Some(doc.version());
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
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
            ObjectData::Light => RowKind::Light,
            ObjectData::Mesh(mesh_id) => {
                let mesh = doc.mesh(mesh_id);
                RowKind::Mesh {
                    verts: mesh.map(|m| m.num_verts()).unwrap_or(0),
                    faces: mesh.map(|m| m.num_faces()).unwrap_or(0),
                }
            }
        };
        let depth = self.depths.get(index).copied().unwrap_or(0);
        Some(OutlinerRow {
            id: key,
            name: object.name.clone(),
            depth: depth as usize,
            selected: doc.selection.contains(id),
            active: doc.selection.active == Some(id),
            visible: object.visible,
            kind,
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
                params: setter_params(index as u32, &entry.kind),
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
    }
}

/// A modifier's parameters, in its setter operator's terms.
///
/// The panel edits this and hands it straight back to the operator, so the
/// values shown are the values applied — there is no third representation to
/// drift out of step.
fn setter_params(index: u32, kind: &ModifierKind) -> PropertyMap {
    let mut p = PropertyMap::new();
    p.set("index", PropertyValue::Int(index as i64));
    match kind {
        ModifierKind::Mirror(m) => {
            p.set("x", PropertyValue::Bool(m.x));
            p.set("y", PropertyValue::Bool(m.y));
            p.set("z", PropertyValue::Bool(m.z));
            p.set(
                "merge_threshold",
                PropertyValue::Float(m.merge_threshold as f64),
            );
        }
        ModifierKind::Subdivide(sub) => {
            p.set("levels", PropertyValue::Int(sub.levels as i64));
            p.set("smooth", PropertyValue::Bool(sub.smooth));
        }
        ModifierKind::Array(a) => {
            p.set("count", PropertyValue::Int(a.count as i64));
            p.set(
                "relative",
                PropertyValue::Vec3([
                    a.relative.x as f64,
                    a.relative.y as f64,
                    a.relative.z as f64,
                ]),
            );
            p.set(
                "constant",
                PropertyValue::Vec3([
                    a.constant.x as f64,
                    a.constant.y as f64,
                    a.constant.z as f64,
                ]),
            );
        }
    }
    p
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

/// Panel and menu labels, resolved once per frame.
struct Labels {
    file_new: String,
    file_open: String,
    file_save: String,
    file_save_as: String,
    recover_open: String,
    recover_discard: String,
    recover_discard_all: String,
    add_cube: String,
    duplicate: String,
    delete: String,
    undo: String,
    redo: String,
    modifiers: String,
    uv_none: String,
    uv_not_editing: String,
    area_menu: String,
    sidebar_item: String,
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
    select_vert: String,
    select_edge: String,
    select_face: String,
}

impl Labels {
    fn new(catalog: &Catalog) -> Self {
        let t = |key: &str| catalog.format(key, TextKind::Ui, &[]);
        Self {
            file_new: t("menu-file-new"),
            file_open: t("menu-file-open"),
            file_save: t("menu-file-save"),
            file_save_as: t("menu-file-save-as"),
            recover_open: t("recover-open"),
            recover_discard: t("recover-discard"),
            recover_discard_all: t("recover-discard-all"),
            add_cube: t("menu-add-cube"),
            duplicate: t("menu-duplicate"),
            delete: t("menu-delete"),
            undo: t("menu-undo"),
            redo: t("menu-redo"),
            modifiers: t("panel-modifiers"),
            uv_none: t("uv-none"),
            uv_not_editing: t("uv-not-editing"),
            area_menu: t("area-menu"),
            sidebar_item: t("sidebar-item"),
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
            select_vert: t("select-mode-vert"),
            select_edge: t("select-mode-edge"),
            select_face: t("select-mode-face"),
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

    // Drawn before the menu so it is the first thing in the window, and as a
    // panel rather than a modal dialog: a user who has just lost a session wants
    // to see their application, not a box in front of it. It also means the
    // choice goes through the command queue like every other action instead of
    // blocking the event loop at startup.
    if !state.recoverable.is_empty() {
        // The height goes on the *panel*, and it takes both setters. Measured,
        // because none of this is guessable:
        //
        // - `ScrollArea::max_height` inside a non-resizable top panel is
        //   inert. A hundred points came out whether it asked for two rows or
        //   twenty-five.
        // - `Panel::max_size` alone only ever *lowers* the height: two rows
        //   worked, ten did nothing, because a non-resizable top panel falls
        //   back to a default of about a hundred points.
        // - `default_size` is the one that sets it, and `max_size` beside it
        //   keeps a one-entry offer short.
        //
        // With both, the height tracks the row count: one row costs the
        // viewport 47 points, four cost 117, ten cost 258.
        let row = recovery_row_height(ui);
        let height = row * (RECOVERY_VISIBLE_ROWS + 1.0);
        Panel::top("recover")
            .default_size(height)
            .max_size(height)
            .show(ui, |ui| {
                draw_recovery(ui, state, &l);
            });
    }

    Panel::top("menu").show(ui, |ui| {
        // Wrapped, not clipped. The edit-mode toolbar is already wide enough to
        // run off a narrow window, and a clipped row silently hides buttons —
        // the user concludes the tool does not exist. Wrapping costs a little
        // height and loses nothing.
        //
        // Grouped menus are the real answer and the plan budgets for them; this
        // is what keeps the row honest until then.
        ui.horizontal_wrapped(|ui| {
            // First in the row, because that is where every application in this
            // category puts it and where a user's eye goes for it.
            if ui.button(&l.file_new).clicked() {
                state.queue.push(Command::NewDocument);
            }
            if ui.button(&l.file_open).clicked() {
                state.queue.push(Command::OpenDocument);
            }
            if ui.button(&l.file_save).clicked() {
                state.queue.push(Command::SaveDocument);
            }
            if ui.button(&l.file_save_as).clicked() {
                state.queue.push(Command::SaveDocumentAs);
            }
            ui.separator();
            if command_tool(ui, state, "object.add_primitive", &l.add_cube) {
                let mut p = PropertyMap::new();
                p.set("shape", kreate_reflect::PropertyValue::Enum(1));
                state.queue.run("object.add_primitive", p);
            }
            if command_tool(ui, state, "object.duplicate", &l.duplicate) {
                let mut p = PropertyMap::new();
                p.set("linked", kreate_reflect::PropertyValue::Bool(true));
                state.queue.run("object.duplicate", p);
            }
            if command_tool(ui, state, "object.delete", &l.delete) {
                state.queue.run("object.delete", PropertyMap::new());
            }
            ui.separator();
            ui.add_enabled_ui(history.can_undo(), |ui| {
                if ui.button(&l.undo).clicked() {
                    state.queue.push(Command::Undo);
                }
            });
            ui.add_enabled_ui(history.can_redo(), |ui| {
                if ui.button(&l.redo).clicked() {
                    state.queue.push(Command::Redo);
                }
            });
            ui.separator();

            // A dropdown over every mode rather than a button that toggles
            // between two. Which mode you are in governs what every other key
            // does, so it has to be readable at a glance — and it has to still
            // read as one thing when there are seven of them.
            draw_mode_menu(ui, state, doc, mode);

            if mode.shows_elements() {
                ui.separator();
                for (index, text) in [
                    (0u32, &l.select_vert),
                    (1, &l.select_edge),
                    (2, &l.select_face),
                ] {
                    if ui
                        .selectable_label(state.select_mode == index, text)
                        .clicked()
                    {
                        state.queue.push(Command::SetSelectMode(index));
                    }
                }
                ui.separator();
                if command_tool(ui, state, "mesh.extrude_region", &l.extrude) {
                    let mut p = PropertyMap::new();
                    p.set("offset", kreate_reflect::PropertyValue::Float(1.0));
                    state.queue.run("mesh.extrude_region", p);
                }
                if command_tool(ui, state, "mesh.inset", &l.inset) {
                    let mut p = PropertyMap::new();
                    p.set("thickness", kreate_reflect::PropertyValue::Float(0.1));
                    state.queue.run("mesh.inset", p);
                }
                if command_tool(ui, state, "mesh.bridge", &l.bridge) {
                    state.queue.run("mesh.bridge", PropertyMap::new());
                }
                if command_tool(ui, state, "mesh.loop_cut", &l.loop_cut) {
                    let mut p = PropertyMap::new();
                    p.set("cuts", kreate_reflect::PropertyValue::Int(1));
                    p.set("position", kreate_reflect::PropertyValue::Float(0.5));
                    state.queue.run("mesh.loop_cut", p);
                }
                if command_tool(ui, state, "mesh.bevel", &l.bevel) {
                    let mut p = PropertyMap::new();
                    p.set("width", kreate_reflect::PropertyValue::Float(0.1));
                    state.queue.run("mesh.bevel", p);
                }
                if command_tool(ui, state, "mesh.subdivide", &l.subdivide) {
                    let mut p = PropertyMap::new();
                    p.set("iterations", kreate_reflect::PropertyValue::Int(1));
                    state.queue.run("mesh.subdivide", p);
                }
                // Delete and dissolve both act on whatever the current element
                // mode is, so the button carries it rather than asking again.
                if command_tool(ui, state, "mesh.dissolve", &l.dissolve) {
                    let mut p = PropertyMap::new();
                    p.set(
                        "mode",
                        kreate_reflect::PropertyValue::Enum(state.select_mode),
                    );
                    state.queue.run("mesh.dissolve", p);
                }
                if command_tool(ui, state, "mesh.delete", &l.delete) {
                    let mut p = PropertyMap::new();
                    p.set(
                        "mode",
                        kreate_reflect::PropertyValue::Enum(state.select_mode),
                    );
                    state.queue.run("mesh.delete", p);
                }
                ui.separator();
                if command_tool(ui, state, "mesh.merge", &l.merge) {
                    let mut p = PropertyMap::new();
                    p.set("target", kreate_reflect::PropertyValue::Enum(0));
                    state.queue.run("mesh.merge", p);
                }
                if command_tool(ui, state, "mesh.merge", &l.weld) {
                    let mut p = PropertyMap::new();
                    p.set("distance", kreate_reflect::PropertyValue::Float(1e-4));
                    state.queue.run("mesh.merge", p);
                }
                if command_tool(ui, state, "mesh.select", &l.select_loop) {
                    let mut p = PropertyMap::new();
                    p.set("action", kreate_reflect::PropertyValue::Enum(3));
                    state.queue.run("mesh.select", p);
                }
                ui.separator();
                if command_tool(ui, state, "mesh.shade", &l.shade_smooth) {
                    let mut p = PropertyMap::new();
                    p.set("smooth", kreate_reflect::PropertyValue::Bool(true));
                    state.queue.run("mesh.shade", p);
                }
                if command_tool(ui, state, "mesh.shade", &l.shade_flat) {
                    let mut p = PropertyMap::new();
                    p.set("smooth", kreate_reflect::PropertyValue::Bool(false));
                    state.queue.run("mesh.shade", p);
                }
                if ui.button(&l.mark_sharp).clicked() {
                    // By angle rather than by selection: a user reaching for
                    // this on a freshly imported model has nothing selected, and
                    // thirty degrees is the answer they want.
                    let mut p = PropertyMap::new();
                    p.set(
                        "angle",
                        kreate_reflect::PropertyValue::Float(30f64.to_radians()),
                    );
                    state.queue.run("mesh.mark_sharp", p);
                }
                if command_tool(ui, state, "mesh.fill", &l.fill) {
                    state.queue.run("mesh.fill", PropertyMap::new());
                }
                if command_tool(ui, state, "mesh.spin", &l.spin) {
                    state.queue.run("mesh.spin", PropertyMap::new());
                }
                if command_tool(ui, state, "mesh.normals", &l.recalc_normals) {
                    let mut p = PropertyMap::new();
                    p.set("action", kreate_reflect::PropertyValue::Enum(0));
                    state.queue.run("mesh.normals", p);
                }
            }

            ui.separator();
            draw_workspace_tabs(ui, state);
            // No panel toggles here any more. An editor is opened by
            // pointing an area at it, from that area's own header — which is
            // where the user is looking when they want to change what they are
            // looking at, and which scales to the nine more editors the plan
            // has scheduled.
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
            ui.separator();
            ui.label(&status);
            // The last thing that happened, to the right of the standing counts.
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
                .find(|a| a.editor == kreate_ui::EditorKind::Viewport)
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
                );
                if Some(area.id) == first_viewport {
                    viewport_rect = rect;
                }
            }
        });
    state.apply_area_edits();
    viewport_rect
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
            draw_area_header(ui, state, area, l);
        },
    );

    // The viewport takes its body whole: the rendered image is meant to reach
    // the edges, and padding around it would be a grey frame the user did not
    // ask for. Everything else gets the panel inset.
    let inner = if area.editor == kreate_ui::EditorKind::Viewport {
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
            match area.editor {
                kreate_ui::EditorKind::Viewport => {
                    draw_viewport(ui, state, inner, image);
                    if state.sidebar_open(area.id) {
                        draw_item_sidebar(ui, state, doc, registry, area.id, inner, l);
                    }
                }
                kreate_ui::EditorKind::Outliner => {
                    let order = std::mem::take(&mut state.outliner);
                    kreate_ui::outliner(
                        ui,
                        &state.catalog,
                        order.ids(),
                        selected,
                        active,
                        |index| order.row(doc, index),
                        &mut state.queue,
                    );
                    state.outliner = order;
                }
                kreate_ui::EditorKind::Properties => {
                    draw_adjust_panel(ui, state, registry, history);
                    ui.separator();
                    ui.heading(&l.modifiers);
                    draw_modifier_stack(ui, state, modifier_rows);
                }
                kreate_ui::EditorKind::UvEditor => {
                    draw_uv_editor(ui, state, area.id, l);
                }
                kreate_ui::EditorKind::Preferences => {
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
            }
        },
    );
    inner
}

/// The strip along the top of an area: which editor, and what can be done to
/// the area itself.
///
/// Everything here queues rather than acting. The tree cannot be edited while
/// the frame is drawing against a layout derived from it.
fn draw_area_header(ui: &mut Ui, state: &mut ShellState, area: &kreate_ui::AreaRect, l: &Labels) {
    ui.horizontal(|ui| {
        let names: Vec<(kreate_ui::EditorKind, String)> = kreate_ui::EditorKind::ALL
            .iter()
            .map(|kind| {
                (
                    *kind,
                    state.catalog.format(kind.name_key(), TextKind::Ui, &[]),
                )
            })
            .collect();
        let current = names
            .iter()
            .find(|(kind, _)| *kind == area.editor)
            .map(|(_, name)| name.clone())
            .unwrap_or_default();

        egui::ComboBox::from_id_salt(("editor-kind", area.id.0))
            .selected_text(current)
            .show_ui(ui, |ui| {
                for (kind, name) in &names {
                    if ui.selectable_label(*kind == area.editor, name).clicked() {
                        state.set_area_editor(area.id, *kind);
                    }
                }
            });

        ui.menu_button(&l.area_menu, |ui| {
            if ui.button(&l.area_split_v).clicked() {
                state.split_area(area.id, kreate_ui::Axis::Vertical);
                ui.close();
            }
            if ui.button(&l.area_split_h).clicked() {
                state.split_area(area.id, kreate_ui::Axis::Horizontal);
                ui.close();
            }
            // Disabled rather than hidden on the last area: a menu whose
            // entries come and go is a menu the user cannot learn.
            ui.add_enabled_ui(state.screen().area_count() > 1, |ui| {
                if ui.button(&l.area_close).clicked() {
                    state.close_area(area.id);
                    ui.close();
                }
            });
        });
    });
}

/// The 3D viewport: the rendered image, and everything painted over it.
fn draw_viewport(
    ui: &mut Ui,
    state: &mut ShellState,
    rect: egui::Rect,
    image: Option<ViewportImage>,
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
    draw_select_band(ui, state, rect);
    draw_modal_header(ui, state, rect);
    draw_nav_gizmo(ui, state, rect);
    draw_context_menu(ui, state, rect);
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
) {
    let width = SIDEBAR_WIDTH.min(viewport.width() * 0.5);
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
            ui.heading(&l.sidebar_item);
            ui.separator();

            let Some(active) = doc.selection.active.and_then(|id| doc.object(id)) else {
                // Said out loud rather than drawn as three empty boxes: a
                // coordinate field with nothing behind it invites a number that
                // goes nowhere.
                ui.weak(&l.sidebar_nothing);
                return;
            };
            // i18n-exempt: an object's name is the user's own text
            ui.label(&active.name);

            let Some(info) = registry.get("object.set_transform") else {
                return;
            };
            let t = active.transform;
            let (rx, ry, rz) = t.rotation.to_euler(kreate_math::glam::EulerRot::XYZ);
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

            if kreate_ui::property_editor::property_editor(
                ui,
                &state.catalog,
                info.id,
                info.schema,
                &mut params,
            ) {
                state.queue.run("object.set_transform", params);
            }
        },
    );
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
        if ui.selectable_label(index == active, name).clicked() {
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
    let has_mesh = kreate_ops::ops::mesh::target_mesh(doc).is_some();

    egui::ComboBox::from_id_salt("mode")
        .selected_text(current)
        .show_ui(ui, |ui| {
            for (kind, name) in &names {
                let available = kind.can_enter(has_mesh);
                ui.add_enabled_ui(available, |ui| {
                    if ui.selectable_label(*kind == mode, name).clicked() {
                        state.queue.push(Command::SetMode(kind.tag().to_string()));
                    }
                });
            }
        });
}

/// A toolbar button carrying whatever is known about the command behind it.
///
/// The label stays as the top row already spells it — the buttons are laid out
/// and translated there — and the *tooltip* is what the command information
/// adds: the sentence and, above all, the shortcut. Teaching the key at the
/// moment the user reaches for the button is the whole point; a toolbar that
/// only ever runs commands keeps its users coming back to it forever.
fn command_tool(ui: &mut egui::Ui, state: &ShellState, op_id: &str, label: &str) -> bool {
    let response = ui.button(label);
    match state.command_info.get(op_id) {
        Some(info) if info.has_tooltip() => response.on_hover_text(info.tooltip()).clicked(),
        _ => response.clicked(),
    }
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
    egui::Area::new(egui::Id::new("viewport-context-menu"))
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

    // Anywhere else dismisses it, which is what every menu everywhere does and
    // what a user will try first.
    if ui.ctx().input(|i| {
        i.pointer.any_click() && !i.pointer.any_pressed() || i.key_pressed(egui::Key::Escape)
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
fn draw_select_band(ui: &mut Ui, state: &ShellState, viewport: egui::Rect) {
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

/// The recovery offer: one row per autosave, in a bounded scrolling list.
///
/// It used to be a `horizontal_wrapped` flow of every entry with its two
/// buttons, which is fine for one and unusable for forty: a stale session
/// directory per crashed process is normal, and a run of interrupted tests
/// left forty-two. The offer filled two thirds of the window and pushed the
/// viewport off the bottom.
///
/// So: a fixed maximum height, one row each, the count in the heading, and a
/// way to dismiss the lot. The list is newest first, which `find_recoverable`
/// guarantees — the one a user wants is almost always the last one written.
fn draw_recovery(ui: &mut Ui, state: &mut ShellState, l: &Labels) {
    let count = state.recoverable.len();
    let heading = state.catalog.format(
        "recover-count",
        TextKind::Ui,
        &[("count", &count.to_string())],
    );

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(heading).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Present from the first row, not only when the list is long: a
            // user who wants none of them should not have to click forty times
            // to say so.
            if ui.button(&l.recover_discard_all).clicked() {
                discard_all_recoveries(state);
            }
        });
    });

    // The scroll area is what makes the overflow reachable rather than
    // clipped; the panel above is what bounds it.
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // A clone, because the buttons queue commands that will mutate this
            // list and the loop would otherwise hold a borrow across it.
            let offers: Vec<(std::path::PathBuf, String, String)> = state
                .recoverable
                .iter()
                .map(|o| (o.path.clone(), o.name.clone(), o.age.clone()))
                .collect();
            for (path, name, age) in offers {
                ui.horizontal(|ui| {
                    // The full path on hover. Forty autosaves called
                    // `untitled.kreate` differ only by the session directory
                    // they came from, and the user occasionally needs to see it.
                    ui.label(&name).on_hover_text(path.display().to_string());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(&l.recover_discard).clicked() {
                            state
                                .queue
                                .push(Command::DiscardRecovery { path: path.clone() });
                        }
                        if ui.button(&l.recover_open).clicked() {
                            state
                                .queue
                                .push(Command::RecoverDocument { path: path.clone() });
                        }
                        ui.weak(&age);
                    });
                });
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
fn draw_adjust_panel(
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

    let label = state.catalog.format(&info.name_key(), TextKind::Ui, &[]);
    let label = if label == info.name_key() {
        info.label.to_string()
    } else {
        label
    };
    ui.label(label);
    ui.separator();

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

/// Draws the modifier stack for the active object.
fn draw_modifier_stack(ui: &mut Ui, state: &mut ShellState, rows: &mut [kreate_ui::ModifierRow]) {
    let text = kreate_ui::modifier_stack::ModifierText::resolve(&state.catalog, KIND_KEYS);
    kreate_ui::modifier_stack(ui, &text, &state.catalog, rows, &mut state.queue);
}

/// Translation keys for the addable modifier kinds, in the order the add
/// operator's enum expects.
///
/// Two lists that have to agree; a test holds them to it.
const KIND_KEYS: &[&str] = &["modifier-mirror", "modifier-subdivide", "modifier-array"];

/// The built-in English catalog.
///
/// Shipped as the fallback so an untranslated key resolves to readable text
/// rather than to itself. The Korean resource sits beside it.
pub const EN_FTL: &str = r#"
menu-file-new = New
menu-file-open = Open...
menu-file-save = Save
menu-file-save-as = Save As...
recover-count = { $count } recovered documents
recover-open = Open
recover-discard = Discard
recover-discard-all = Discard all
recover-age-now = just now
recover-age-minutes = { $n } min ago
recover-age-hours = { $n } h ago
recover-age-days = { $n } d ago
status-saved = Saved { $path }
status-opened = Opened { $path }
menu-add-cube = Add Cube
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
sidebar-nothing = Nothing is active. Select an object to see where it is.
area-split-vertical = Split left / right
area-split-horizontal = Split top / bottom
area-close = Close area
editor-viewport = 3D Viewport
editor-outliner = Outliner
editor-properties = Properties
editor-uv = UV Editor
editor-preferences = Preferences
uv-not-editing = The UV editor shows the mesh being edited. Press Tab to enter edit mode.
adjust-nothing = No operation to adjust
status-counts = { $objects } objects, { $meshes } meshes, { $selected } selected
outliner-kind-empty = Empty
outliner-kind-mesh = Mesh
outliner-kind-camera = Camera
outliner-kind-light = Light
outliner-shared-mesh = Shared by { $count } objects
outliner-mesh-summary = { $name }: { $verts } verts, { $faces } faces
panel-modifiers = Modifiers
modifier-none = No modifiers
modifier-add = Add Modifier
modifier-remove = Remove
modifier-move-up = Move towards the input
modifier-move-down = Move towards the result
modifier-enabled = Apply this modifier
modifier-mirror = Mirror
modifier-subdivide = Subdivision
modifier-array = Array

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
op-uv-smart-project-desc = Groups faces that face the same way, unwraps each group, and lays the pieces out on the map.
op-uv-pack = Pack Islands
op-uv-pack-desc = Rearranges the pieces already there to fit the map, leaving the cuts alone.
op-object-translate = Move
op-object-translate-desc = Moves the selected objects.
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
recover-count = 복구된 작업 { $count }개
recover-open = 열기
recover-discard = 버리기
recover-discard-all = 모두 버리기
recover-age-now = 방금
recover-age-minutes = { $n }분 전
recover-age-hours = { $n }시간 전
recover-age-days = { $n }일 전
status-saved = { $path } 저장함
status-opened = { $path } 열었음
menu-add-cube = 큐브 추가
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
area-menu = 여역
sidebar-item = 항목
sidebar-nothing = 활성 오버젝트가 없습니다. 오버젝트를 선택하면 위치가 보입니다.
area-split-vertical = 종으로 나누기
area-split-horizontal = 가로로 나누기
area-close = 여역 닫기
editor-viewport = 3D 뷰포트
editor-outliner = 아웃라이너
editor-properties = 특성
editor-uv = UV 에디터
editor-preferences = 환경 설정
uv-not-editing = UV 에디터는 편집 중인 메시를 보여줍니다. Tab을 눌러 편집 모드로 들어가세요.
adjust-nothing = 조정할 작업이 없습니다
status-counts = 오브젝트 { $objects }개, 메시 { $meshes }개, 선택 { $selected }개
outliner-kind-empty = 빈 오브젝트
outliner-kind-mesh = 메시
outliner-kind-camera = 카메라
outliner-kind-light = 조명
outliner-shared-mesh = 오브젝트 { $count }개가 공유
outliner-mesh-summary = { $name }: 정점 { $verts }, 면 { $faces }
panel-modifiers = 모디파이어
modifier-none = 모디파이어 없음
modifier-add = 모디파이어 추가
modifier-remove = 제거
modifier-move-up = 입력 쪽으로 이동
modifier-move-down = 결과 쪽으로 이동
modifier-enabled = 이 모디파이어 적용
modifier-mirror = 미러
modifier-subdivide = 서브디비전
modifier-array = 배열

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
op-uv-smart-project-desc = 각도가 비슷한 면끼리 묶어 조각별로 펼친 뒤, 조각들을 맵 안에 배치합니다.
op-uv-pack = 아일랜드 배치
op-uv-pack-desc = 자른 자리는 그대로 두고, 이미 있는 조각들만 맵 안에 다시 정렬합니다.
op-object-translate = 이동
op-object-translate-desc = 선택한 오브젝트를 옮깁니다.
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
            .find(|(_, kind)| *kind == kreate_ui::EditorKind::Viewport)
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
            .find(|(_, kind)| *kind == kreate_ui::EditorKind::Outliner)
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
                kreate_ui::Screen::single(kreate_ui::EditorKind::UvEditor),
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
        assert!(state.has_editor(kreate_ui::EditorKind::Viewport));
        assert_eq!(areas.len(), 3, "viewport, outliner, properties");

        let mut ids: Vec<_> = areas.iter().map(|(id, _)| *id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), areas.len(), "an id was handed out twice");

        assert!(!state.has_editor(kreate_ui::EditorKind::UvEditor));
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
                kreate_ui::Screen::single(kreate_ui::EditorKind::Viewport),
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
                kreate_ui::EditorKind::Viewport,
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
                kreate_ui::Screen::single(kreate_ui::EditorKind::Viewport),
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
                kreate_ui::EditorKind::Viewport,
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
                kreate_ui::Screen::single(kreate_ui::EditorKind::UvEditor),
            ),
            ..Default::default()
        };
        let only = state.screen().areas()[0].0;
        state.toggle_sidebar();
        assert!(!state.sidebar_open(only));
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
        let ko = load_catalog("ko-KR");
        assert_eq!(ko.get("menu-add-cube"), "큐브 추가");
        let en = load_catalog("en-US");
        assert_eq!(en.get("menu-add-cube"), "Add Cube");
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
            "menu-file-new",
            "menu-file-open",
            "menu-file-save",
            "menu-file-save-as",
            "recover-count",
            "recover-open",
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
            "outliner-kind-camera",
            "outliner-kind-light",
            "outliner-shared-mesh",
            "outliner-mesh-summary",
        ] {
            assert!(catalog.has(key), "`{key}` is missing from the catalog");
        }
    }
}
