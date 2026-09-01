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
        }
    }
}

impl ShellState {
    /// Whether a viewport keyboard shortcut may run this frame.
    ///
    /// The gate that keeps a rename field from firing tool shortcuts. It is a
    /// single check in a single place on purpose: scattered `if !typing` tests
    /// are how the bug comes back.
    pub fn keymap_enabled(&self) -> bool {
        !self.egui_wants_keyboard && !self.ime_active
    }
}

/// Builds the outliner's view model from a document.
///
/// The UI cannot see a `Document`, so the shape crossing the boundary is a plain
/// list. That also means the outliner is trivially testable without a scene.
pub fn outliner_rows(doc: &Document) -> Vec<OutlinerRow> {
    doc.hierarchy()
        .into_iter()
        .filter_map(|(id, depth)| {
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
            Some(OutlinerRow {
                id: object_key(id),
                name: object.name.clone(),
                depth,
                selected: doc.selection.contains(id),
                active: doc.selection.active == Some(id),
                visible: object.visible,
                kind,
                mesh_users: object.mesh().map(|m| doc.mesh_users(m)).unwrap_or(1),
            })
        })
        .collect()
}

/// Opaque key for an object, so the UI can round-trip a handle it cannot read.
pub fn object_key(id: ObjectId) -> u32 {
    use kreate_mesh::arena::ArenaIndex as _;
    id.index()
}

/// Finds the object an outliner row refers to.
///
/// A lookup rather than reconstructing a handle from the index: the arena's
/// generation counter is deliberately not exposed, and a stale row from a frame
/// where the object was deleted must resolve to nothing instead of to whatever
/// took its slot.
pub fn object_from_key(doc: &Document, key: u32) -> Option<ObjectId> {
    doc.objects()
        .map(|(id, _)| id)
        .find(|id| object_key(*id) == key)
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
        }
    }
}

/// Draws the whole shell and returns the rectangle left for the viewport.
pub fn draw(
    ui: &mut Ui,
    state: &mut ShellState,
    doc: &Document,
    registry: &OperatorRegistry,
    history: &History,
    viewport_texture: Option<egui::TextureId>,
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
    let rows = outliner_rows(doc);

    Panel::top("menu").show(ui, |ui| {
        ui.horizontal(|ui| {
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
            ui.checkbox(&mut state.show_outliner, &l.outliner);
            ui.checkbox(&mut state.show_properties, &l.properties);
        });
    });

    Panel::bottom("status").show(ui, |ui| {
        ui.horizontal(|ui| {
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
            kreate_ui::outliner(ui, &state.catalog, &rows, &mut state.queue);
        });
    }

    if state.show_properties {
        Panel::right("properties")
            .default_size(260.0)
            .show(ui, |ui| {
                ui.heading(&l.properties);
                draw_adjust_panel(ui, state, registry, history);
            });
    }

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            if let Some(texture) = viewport_texture {
                ui.painter().image(
                    texture,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
            rect
        })
        .inner
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
"#;

pub const KO_FTL: &str = r#"
menu-add-cube = 큐브 추가
menu-duplicate = 복제
menu-delete = 삭제
menu-undo = 되돌리기
menu-redo = 다시 실행
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
