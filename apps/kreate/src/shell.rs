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
pub struct ShellState {
    pub catalog: Catalog,
    pub queue: CommandQueue,
    /// Parameters shown in the adjust-last-operation panel.
    pub adjust_params: Option<(String, PropertyMap)>,
    pub show_outliner: bool,
    pub show_properties: bool,
    /// Set while egui holds the keyboard, so the keymap stands down.
    pub egui_wants_keyboard: bool,
    /// Set while an IME is composing.
    pub ime_active: bool,
    /// Whether the application is in mesh edit mode.
    pub edit_mode: bool,
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
    /// Whether the remapping panel is open.
    pub show_keymap: bool,
    /// The binding waiting for a keystroke, if any.
    ///
    /// Lives here rather than in the panel because the application, not the
    /// interface, is what sees the key event — the panel only says which binding
    /// is listening.
    pub capturing: Option<kreate_input::BindingId>,
    /// What the panel asked for this frame.
    pub keymap_edits: Vec<kreate_ui::KeymapEdit>,
}

impl Default for ShellState {
    fn default() -> Self {
        Self {
            catalog: Catalog::new(),
            queue: CommandQueue::new(),
            adjust_params: None,
            show_outliner: true,
            show_properties: true,
            egui_wants_keyboard: false,
            ime_active: false,
            edit_mode: false,
            select_mode: 2,
            camera_rotation: kreate_math::DQuat::IDENTITY,
            gizmo_circle: None,
            outliner: OutlinerOrder::default(),
            select_band: None,
            select_band_deselect: false,
            show_keymap: false,
            capturing: None,
            keymap_edits: Vec::new(),
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
            ObjectData::Camera => RowKind::Camera,
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
    add_cube: String,
    duplicate: String,
    delete: String,
    undo: String,
    redo: String,
    outliner: String,
    properties: String,
    modifiers: String,
    keymap: String,
    mode_object: String,
    mode_edit: String,
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
            add_cube: t("menu-add-cube"),
            duplicate: t("menu-duplicate"),
            delete: t("menu-delete"),
            undo: t("menu-undo"),
            redo: t("menu-redo"),
            outliner: t("panel-outliner"),
            properties: t("panel-properties"),
            modifiers: t("panel-modifiers"),
            keymap: t("panel-keymap"),
            mode_object: t("mode-object"),
            mode_edit: t("mode-edit"),
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
    viewport_texture: Option<ViewportImage>,
) -> egui::Rect {
    // Resolved once per frame into owned strings. A closure over the catalog
    // would hold a borrow of `state` for the whole function, which conflicts with
    // the mutable access the panels need to the command queue.
    let l = Labels::new(&state.catalog);
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

    Panel::top("menu").show(ui, |ui| {
        // Wrapped, not clipped. The edit-mode toolbar is already wide enough to
        // run off a narrow window, and a clipped row silently hides buttons —
        // the user concludes the tool does not exist. Wrapping costs a little
        // height and loses nothing.
        //
        // Grouped menus are the real answer and the plan budgets for them; this
        // is what keeps the row honest until then.
        ui.horizontal_wrapped(|ui| {
            if ui.button(&l.add_cube).clicked() {
                let mut p = PropertyMap::new();
                p.set("shape", kreate_reflect::PropertyValue::Enum(1));
                state.queue.run("object.add_primitive", p);
            }
            if ui.button(&l.duplicate).clicked() {
                let mut p = PropertyMap::new();
                p.set("linked", kreate_reflect::PropertyValue::Bool(true));
                state.queue.run("object.duplicate", p);
            }
            if ui.button(&l.delete).clicked() {
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

            // The mode switch is a pair of exclusive buttons rather than a
            // checkbox: which mode you are in governs what every other key does,
            // so it has to be readable at a glance rather than inferred from a
            // tick.
            let label = if state.edit_mode {
                &l.mode_edit
            } else {
                &l.mode_object
            };
            if ui
                .add_enabled(
                    state.edit_mode || doc.selection.active.is_some(),
                    egui::Button::new(label),
                )
                .clicked()
            {
                state.queue.push(Command::ToggleEditMode);
            }

            if state.edit_mode {
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
                if ui.button(&l.extrude).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("offset", kreate_reflect::PropertyValue::Float(1.0));
                    state.queue.run("mesh.extrude_region", p);
                }
                if ui.button(&l.inset).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("thickness", kreate_reflect::PropertyValue::Float(0.1));
                    state.queue.run("mesh.inset", p);
                }
                if ui.button(&l.bridge).clicked() {
                    state.queue.run("mesh.bridge", PropertyMap::new());
                }
                if ui.button(&l.loop_cut).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("cuts", kreate_reflect::PropertyValue::Int(1));
                    p.set("position", kreate_reflect::PropertyValue::Float(0.5));
                    state.queue.run("mesh.loop_cut", p);
                }
                if ui.button(&l.bevel).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("width", kreate_reflect::PropertyValue::Float(0.1));
                    state.queue.run("mesh.bevel", p);
                }
                if ui.button(&l.subdivide).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("iterations", kreate_reflect::PropertyValue::Int(1));
                    state.queue.run("mesh.subdivide", p);
                }
                // Delete and dissolve both act on whatever the current element
                // mode is, so the button carries it rather than asking again.
                if ui.button(&l.dissolve).clicked() {
                    let mut p = PropertyMap::new();
                    p.set(
                        "mode",
                        kreate_reflect::PropertyValue::Enum(state.select_mode),
                    );
                    state.queue.run("mesh.dissolve", p);
                }
                if ui.button(&l.delete).clicked() {
                    let mut p = PropertyMap::new();
                    p.set(
                        "mode",
                        kreate_reflect::PropertyValue::Enum(state.select_mode),
                    );
                    state.queue.run("mesh.delete", p);
                }
                ui.separator();
                if ui.button(&l.merge).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("target", kreate_reflect::PropertyValue::Enum(0));
                    state.queue.run("mesh.merge", p);
                }
                if ui.button(&l.weld).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("distance", kreate_reflect::PropertyValue::Float(1e-4));
                    state.queue.run("mesh.merge", p);
                }
                if ui.button(&l.select_loop).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("action", kreate_reflect::PropertyValue::Enum(3));
                    state.queue.run("mesh.select", p);
                }
                ui.separator();
                if ui.button(&l.shade_smooth).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("smooth", kreate_reflect::PropertyValue::Bool(true));
                    state.queue.run("mesh.shade", p);
                }
                if ui.button(&l.shade_flat).clicked() {
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
                if ui.button(&l.fill).clicked() {
                    state.queue.run("mesh.fill", PropertyMap::new());
                }
                if ui.button(&l.spin).clicked() {
                    state.queue.run("mesh.spin", PropertyMap::new());
                }
                if ui.button(&l.recalc_normals).clicked() {
                    let mut p = PropertyMap::new();
                    p.set("action", kreate_reflect::PropertyValue::Enum(0));
                    state.queue.run("mesh.normals", p);
                }
            }

            ui.separator();
            ui.checkbox(&mut state.show_outliner, &l.outliner);
            ui.checkbox(&mut state.show_properties, &l.properties);
            ui.checkbox(&mut state.show_keymap, &l.keymap);
        });
    });

    Panel::bottom("status").show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(if state.edit_mode {
                &l.mode_edit
            } else {
                &l.mode_object
            });
            ui.separator();
            ui.label(&status);
            if let Some(entry) = history.last() {
                ui.separator();
                // i18n-exempt: a generated script line is API text, not prose
                ui.monospace(entry.to_script_call());
            }
        });
    });

    if state.show_outliner {
        Panel::left("outliner").default_size(220.0).show(ui, |ui| {
            ui.heading(&l.outliner);
            let order = std::mem::take(&mut state.outliner);
            kreate_ui::outliner(
                ui,
                &state.catalog,
                order.ids(),
                &selected,
                active,
                |index| order.row(doc, index),
                &mut state.queue,
            );
            state.outliner = order;
        });
    }

    if state.show_properties {
        Panel::right("properties")
            .default_size(260.0)
            .show(ui, |ui| {
                ui.heading(&l.properties);
                draw_adjust_panel(ui, state, registry, history);
                ui.separator();
                ui.heading(&l.modifiers);
                draw_modifier_stack(ui, state, &mut modifier_rows);
            });
    }

    if state.show_keymap {
        Panel::right("keymap").default_size(340.0).show(ui, |ui| {
            ui.heading(&l.keymap);
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
        });
    }

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            if let Some(image) = viewport_texture {
                // Only the part of the target that was drawn into. Sampling the
                // whole texture would stretch the image by however much the
                // allocation was rounded up.
                ui.painter().image(
                    image.texture,
                    rect,
                    egui::Rect::from_min_max(
                        egui::pos2(0.0, 0.0),
                        egui::pos2(image.uv.0, image.uv.1),
                    ),
                    egui::Color32::WHITE,
                );
            }
            draw_select_band(ui, state, rect);
            draw_nav_gizmo(ui, state, rect);
            rect
        })
        .inner
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
    let colour = if state.select_band_deselect {
        egui::Color32::from_rgb(120, 170, 230)
    } else {
        egui::Color32::from_rgb(255, 158, 41)
    };
    painter.rect_filled(band, 0.0, colour.gamma_multiply(0.12));
    painter.rect_stroke(
        band,
        0.0,
        egui::Stroke::new(1.0, colour),
        egui::StrokeKind::Inside,
    );
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
panel-keymap = Keymap
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
keymap-context-modal = Modal Tools
panel-outliner = Outliner
panel-properties = Properties
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
"#;

pub const KO_FTL: &str = r#"
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
panel-keymap = 키맵
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
keymap-context-modal = 모달 툴
panel-outliner = 아웃라이너
panel-properties = 속성
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
            "menu-add-cube",
            "menu-duplicate",
            "menu-delete",
            "menu-undo",
            "menu-redo",
            "panel-outliner",
            "panel-properties",
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
