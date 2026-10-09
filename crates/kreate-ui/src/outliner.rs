//! The object outliner.
//!
//! Rows are **virtualised twice over**, and the second one is the part that is
//! easy to miss.
//!
//! Only what is on screen is laid out — that much is `ScrollArea::show_rows`.
//! But an immediate-mode UI also punishes *building* what it does not draw, and
//! handing this widget a `Vec` of a hundred thousand rows means a hundred
//! thousand `String` clones every frame for the twenty the user can see. That
//! cost is invisible at twenty objects and ruinous at fifty thousand, which is
//! exactly the shape of bug that hides until someone opens a real scene.
//!
//! So the widget takes the row **order** — one `u32` per object, cheap enough to
//! cache and hand over whole — and a function that materialises a single row.
//! Only the visible ones are ever built.
//!
//! The crate cannot see a `Document`, so the application supplies both and
//! receives [`crate::Command`]s. Selection arithmetic — plain click, additive
//! click, range click — lives in [`resolve_click`] as a pure function over the
//! order, because that is where the bugs are and it is the part worth testing
//! without a window.

use egui::Ui;
use kreate_i18n::{Catalog, TextKind};

use crate::{Command, CommandQueue};

/// What an object is, for the row's icon and summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    Empty,
    Mesh { verts: u32, faces: u32 },
    Camera,
    Light,
    Armature,
}

impl RowKind {
    pub fn i18n_key(self) -> &'static str {
        match self {
            RowKind::Empty => "outliner-kind-empty",
            RowKind::Mesh { .. } => "outliner-kind-mesh",
            RowKind::Camera => "outliner-kind-camera",
            RowKind::Light => "outliner-kind-light",
            RowKind::Armature => "outliner-kind-armature",
        }
    }
}

/// One row of the outliner.
///
/// Built on demand for the rows on screen, never for the whole scene.
#[derive(Clone, Debug, PartialEq)]
pub struct OutlinerRow {
    /// Opaque object handle. The UI never interprets it.
    pub id: u32,
    pub name: String,
    /// Indentation level in the hierarchy.
    pub depth: usize,
    pub selected: bool,
    pub active: bool,
    /// The object's own eye.
    pub visible: bool,
    /// Whether it is drawn at all: its eye, every parent's and its
    /// collection's. A child of a hidden parent keeps its eye open and is
    /// drawn faint, so the user sees why it is gone and that it will return.
    pub shown: bool,
    pub kind: RowKind,
    /// How many objects share this row's mesh. Greater than one means editing it
    /// changes every user, which the outliner must say before the user is
    /// surprised by it.
    pub mesh_users: usize,
    /// The datablock behind the object — the mesh, the light, the camera — by
    /// name, for the row under the object when it is expanded. `None` for an
    /// empty, which has none.
    pub data_name: Option<String>,
    /// The materials on the mesh's slots, in slot order, for the rows under
    /// the data row. Blender shows them there; a modeller looking for "which
    /// material is on this thing" looks there.
    pub materials: Vec<String>,
    /// Whether the rows after this one, at greater depth, are this object's
    /// children — which is what earns it an expander.
    pub has_children: bool,
    /// An armature's bones, under its data row: name, depth in the bone
    /// hierarchy, and whether selected — Blender lists them there, and a rig
    /// of eighty bones is found by name, not by clicking in the viewport.
    pub bones: Vec<(String, usize, bool)>,
}

impl OutlinerRow {
    pub fn new(id: u32, name: impl Into<String>, kind: RowKind) -> Self {
        Self {
            id,
            name: name.into(),
            depth: 0,
            selected: false,
            active: false,
            visible: true,
            shown: true,
            kind,
            mesh_users: 1,
            data_name: None,
            materials: Vec::new(),
            bones: Vec::new(),
            has_children: false,
        }
    }
}

/// One line of the tree as drawn: which kind of row, and for an object row,
/// its position in the order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeRow {
    /// "Scene Collection", Blender's root. Not an object; not selectable.
    Root,
    /// A collection's heading, by its place in the groups.
    Collection(usize),
    /// The object at this position in the order.
    Object(usize),
    /// The datablock under the object at this position.
    Data(usize),
    /// The material in this slot, under the object at this position.
    Material(usize, usize),
    /// A bone of the armature at this position, by its place in the list.
    Bone(usize, usize),
}

/// The rows on screen, given which objects are expanded.
///
/// Pure, over the order and the depths, so it can be tested without a window:
/// a collapsed parent hides everything after it that is deeper, and an
/// expanded object gains a row for its datablock and one per material. `row`
/// is consulted only for expanded objects — the ones whose children are being
/// listed — which keeps this linear in the number of objects rather than the
/// number of strings.
pub fn visible_rows(
    depths: &[u16],
    expanded: impl Fn(usize) -> bool,
    row: impl Fn(usize) -> Option<OutlinerRow>,
) -> Vec<TreeRow> {
    visible_rows_grouped(depths, &[], expanded, row)
}

/// A collection in the outliner: where its objects start in the order, its
/// name, and whether it is hidden. Its objects follow it, one level in.
#[derive(Clone, Debug, PartialEq)]
pub struct OutlinerGroup {
    pub start: usize,
    pub name: String,
    pub hidden: bool,
}

/// [`visible_rows`], with a heading row before each collection's objects.
/// `starts` are where each group begins in the order, ascending; an empty
/// collection starts where the next thing does.
pub fn visible_rows_grouped(
    depths: &[u16],
    starts: &[usize],
    expanded: impl Fn(usize) -> bool,
    row: impl Fn(usize) -> Option<OutlinerRow>,
) -> Vec<TreeRow> {
    let mut out = vec![TreeRow::Root];
    // The depth below which rows are hidden by a collapsed ancestor, if any.
    let mut hidden_below: Option<u16> = None;
    let mut next_group = 0;
    let mut heading = |index: usize, out: &mut Vec<TreeRow>, hidden_below: &mut Option<u16>| {
        while next_group < starts.len() && starts[next_group] <= index {
            out.push(TreeRow::Collection(next_group));
            *hidden_below = None;
            next_group += 1;
        }
    };
    for (index, depth) in depths.iter().enumerate() {
        heading(index, &mut out, &mut hidden_below);
        if let Some(limit) = hidden_below {
            if *depth > limit {
                continue;
            }
            hidden_below = None;
        }
        out.push(TreeRow::Object(index));
        if expanded(index) {
            if let Some(r) = row(index) {
                if r.data_name.is_some() {
                    out.push(TreeRow::Data(index));
                }
                for slot in 0..r.materials.len() {
                    out.push(TreeRow::Material(index, slot));
                }
                for bone in 0..r.bones.len() {
                    out.push(TreeRow::Bone(index, bone));
                }
            }
        } else {
            hidden_below = Some(*depth);
        }
    }
    heading(usize::MAX, &mut out, &mut hidden_below);
    out
}

/// How a click was modified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClickModifiers {
    /// Add to or remove from the selection.
    pub additive: bool,
    /// Select everything between the active row and this one.
    pub range: bool,
}

/// Works out the new selection from a click.
///
/// Returns `(objects, active)`. Pure, and over the row *order* rather than over
/// materialised rows: this is where selection bugs live — a range click that
/// forgets one end, an additive click that fails to toggle, an active object
/// left pointing at something no longer selected — and none of them need a row's
/// name or icon to reproduce.
pub fn resolve_click(
    order: &[u32],
    selected: &[u32],
    active: Option<u32>,
    clicked_index: usize,
    modifiers: ClickModifiers,
) -> (Vec<u32>, Option<u32>) {
    let Some(clicked) = order.get(clicked_index).copied() else {
        return (Vec::new(), None);
    };

    if modifiers.range {
        // From the active row to the clicked one, inclusive at both ends. With no
        // active row there is nothing to range *from*, so it degrades to a plain
        // click rather than selecting everything.
        let anchor = active
            .and_then(|a| order.iter().position(|id| *id == a))
            .unwrap_or(clicked_index);
        let (lo, hi) = if anchor <= clicked_index {
            (anchor, clicked_index)
        } else {
            (clicked_index, anchor)
        };
        return (order[lo..=hi].to_vec(), Some(clicked));
    }

    if modifiers.additive {
        let mut objects = selected.to_vec();
        if let Some(pos) = objects.iter().position(|id| *id == clicked) {
            objects.remove(pos);
            // Deselecting the active row must move the active marker, or later
            // operators act on something the user cannot see is chosen.
            let active = objects.last().copied();
            return (objects, active);
        }
        objects.push(clicked);
        return (objects, Some(clicked));
    }

    (vec![clicked], Some(clicked))
}

/// Draws the outliner as a tree.
///
/// **Blender's shape.** A "Scene Collection" root, every object under it at
/// its hierarchy depth, and an expander on each that opens to show what the
/// object is made of — its mesh, light or camera, and the materials on the
/// mesh's slots. A collapsed parent hides its children. What a modeller comes
/// here to answer is "what is this thing, and what is on it", and a flat list
/// of names answered neither.
///
/// `order` is every object in hierarchy order; `row` materialises one by
/// position and is called for the rows on screen and for the objects that are
/// expanded. Which objects are expanded is the widget's own memory, kept in
/// egui's data by object key, so it survives the document changing.
#[allow(clippy::too_many_arguments)]
pub fn outliner(
    ui: &mut Ui,
    catalog: &Catalog,
    order: &[u32],
    depths: &[u16],
    groups: &[OutlinerGroup],
    selected: &[u32],
    active: Option<u32>,
    row: impl Fn(usize) -> Option<OutlinerRow>,
    queue: &mut CommandQueue,
) {
    let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
    let expanded_id = ui.id().with("expanded");
    let expanded: std::collections::HashSet<u32> =
        ui.data(|d| d.get_temp(expanded_id)).unwrap_or_default();

    let starts: Vec<usize> = groups.iter().map(|g| g.start).collect();
    let rows = visible_rows_grouped(
        depths,
        &starts,
        |index| order.get(index).is_some_and(|key| expanded.contains(key)),
        &row,
    );
    let mut toggled: Option<u32> = None;
    // A name being typed, Blender's double-click: kept across frames until
    // Enter, a click away, or Escape.
    let rename_id = ui.id().with("renaming");
    let mut renaming: Option<Renaming> = ui.data(|d| d.get_temp(rename_id));

    // A scroll bar with room of its own, not one floating over the rows: the
    // eyes sit against the right edge, and a floating bar lay on top of them.
    ui.spacing_mut().scroll = egui::style::ScrollStyle::solid();
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        // Only the visible rows are laid out — and, because `row` is a function
        // rather than a slice, only the visible rows are *built*. The whole
        // point: a hundred thousand objects must cost the same as twenty.
        .show_rows(ui, row_height, rows.len(), |ui, visible| {
            for at in visible {
                match rows[at] {
                    TreeRow::Root => draw_root(ui, catalog, queue),
                    TreeRow::Collection(g) => {
                        if let Some(group) = groups.get(g) {
                            draw_collection_row(ui, catalog, group, queue, &mut renaming);
                        }
                    }
                    TreeRow::Object(index) => {
                        let Some(data) = row(index) else { continue };
                        let open = expanded.contains(&data.id);
                        if draw_object_row(ui, catalog, order, selected, active, index, &data, open, queue, &mut renaming) {
                            toggled = Some(data.id);
                        }
                    }
                    TreeRow::Data(index) => {
                        if let Some(data) = row(index) {
                            draw_data_row(ui, catalog, &data);
                        }
                    }
                    TreeRow::Material(index, slot) => {
                        if let Some(data) = row(index) {
                            draw_material_row(ui, &data, slot);
                        }
                    }
                    TreeRow::Bone(index, bone) => {
                        if let Some(data) = row(index) {
                            draw_bone_row(ui, &data, bone, queue);
                        }
                    }
                }
            }
        });

    ui.data_mut(|d| match renaming {
        Some(r) => {
            d.insert_temp(rename_id, r);
        }
        None => {
            d.remove::<Renaming>(rename_id);
        }
    });

    if let Some(key) = toggled {
        let mut set = expanded;
        if !set.remove(&key) {
            set.insert(key);
        }
        ui.data_mut(|d| d.insert_temp(expanded_id, set));
    }
}

/// Indentation per level of the tree, in points.
const INDENT: f32 = 14.0;

/// An object row being dragged, by its key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DraggedObject(u32);

/// Outlines a row something can be dropped on while a drag hovers it.
fn drop_highlight(ui: &Ui, response: &egui::Response) {
    if response.dnd_hover_payload::<DraggedObject>().is_some() {
        let theme = crate::theme::from_ui(ui);
        ui.painter().rect_stroke(
            response.rect.expand(1.0),
            2.0,
            egui::Stroke::new(1.5, theme.accent),
            egui::StrokeKind::Outside,
        );
    }
}

/// What is being renamed in place, and the text typed so far.
#[derive(Clone, Debug, PartialEq)]
enum Renaming {
    Object(u32, String),
    /// By its current name, which is how the operators find a collection.
    Collection(String, String),
}

/// The field a name is typed into. `Some(true)` to keep what was typed —
/// Enter, or a click elsewhere, as Blender takes it — `Some(false)` on
/// Escape, `None` while typing.
fn rename_field(ui: &mut Ui, text: &mut String) -> Option<bool> {
    let response = ui.add(egui::TextEdit::singleline(text).desired_width(150.0));
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        return Some(false);
    }
    if response.lost_focus() {
        return Some(true);
    }
    if !response.has_focus() {
        response.request_focus();
    }
    None
}

fn draw_root(ui: &mut Ui, catalog: &Catalog, queue: &mut CommandQueue) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(16.0), egui::Sense::hover());
        collection_icon(ui, rect);
        let response = ui.label(
            egui::RichText::new(catalog.format("outliner-scene-collection", TextKind::Ui, &[]))
                .strong(),
        );
        // Dropped on the scene: out of any collection.
        drop_highlight(ui, &response);
        if let Some(dragged) = response.dnd_release_payload::<DraggedObject>() {
            queue.push(Command::MoveObjectToCollection {
                object: dragged.0,
                collection: String::new(),
            });
        }
    });
}

/// A collection's heading: its icon, its name, and its eye.
fn draw_collection_row(
    ui: &mut Ui,
    catalog: &Catalog,
    group: &OutlinerGroup,
    queue: &mut CommandQueue,
    renaming: &mut Option<Renaming>,
) {
    ui.horizontal(|ui| {
        ui.add_space(INDENT);
        let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(16.0), egui::Sense::hover());
        collection_icon(ui, rect);
        let typing = match renaming {
            Some(Renaming::Collection(name, text)) if *name == group.name => Some(text),
            _ => None,
        };
        if let Some(text) = typing {
            if let Some(keep) = rename_field(ui, text) {
                if keep {
                    let mut params = kreate_reflect::PropertyMap::new();
                    params.set(
                        "name",
                        kreate_reflect::PropertyValue::Text(group.name.clone()),
                    );
                    params.set(
                        "new_name",
                        kreate_reflect::PropertyValue::Text(text.clone()),
                    );
                    queue.push(Command::RunOperator {
                        op_id: "collection.rename".into(),
                        params,
                    });
                }
                *renaming = None;
            }
        } else {
            let mut label = egui::RichText::new(&group.name);
            if group.hidden {
                label = label.weak();
            }
            let response = ui.add(egui::Label::new(label).sense(egui::Sense::click()));
            if response.double_clicked() {
                *renaming = Some(Renaming::Collection(group.name.clone(), group.name.clone()));
            }
            // Dropped on a collection: into it, as Blender's.
            drop_highlight(ui, &response);
            if let Some(dragged) = response.dnd_release_payload::<DraggedObject>() {
                queue.push(Command::MoveObjectToCollection {
                    object: dragged.0,
                    collection: group.name.clone(),
                });
            }
            // Blender's right-click on a collection: rename it, or take it
            // away and leave what was in it in the scene.
            response.context_menu(|ui| {
                if ui
                    .button(catalog.format("outliner-rename", TextKind::Ui, &[]))
                    .clicked()
                {
                    *renaming = Some(Renaming::Collection(group.name.clone(), group.name.clone()));
                    ui.close();
                }
                if ui
                    .button(catalog.format("outliner-remove-collection", TextKind::Ui, &[]))
                    .clicked()
                {
                    let mut params = kreate_reflect::PropertyMap::new();
                    params.set(
                        "name",
                        kreate_reflect::PropertyValue::Text(group.name.clone()),
                    );
                    queue.push(Command::RunOperator {
                        op_id: "collection.remove".into(),
                        params,
                    });
                    ui.close();
                }
            });
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (eye, response) =
                ui.allocate_exact_size(egui::Vec2::splat(18.0), egui::Sense::click());
            eye_icon(ui, eye, !group.hidden, response.hovered());
            let response = response.on_hover_text(catalog.format(
                if group.hidden {
                    "outliner-show"
                } else {
                    "outliner-hide"
                },
                TextKind::Tooltip,
                &[],
            ));
            if response.clicked() {
                let mut params = kreate_reflect::PropertyMap::new();
                params.set(
                    "name",
                    kreate_reflect::PropertyValue::Text(group.name.clone()),
                );
                params.set("visible", kreate_reflect::PropertyValue::Bool(group.hidden));
                queue.push(Command::RunOperator {
                    op_id: "collection.set_visible".into(),
                    params,
                });
            }
        });
    });
}

/// Draws one object's row. Returns whether its expander was clicked.
#[allow(clippy::too_many_arguments)]
fn draw_object_row(
    ui: &mut Ui,
    catalog: &Catalog,
    order: &[u32],
    selected: &[u32],
    active: Option<u32>,
    index: usize,
    row: &OutlinerRow,
    open: bool,
    queue: &mut CommandQueue,
    renaming: &mut Option<Renaming>,
) -> bool {
    let mut toggled = false;
    ui.horizontal(|ui| {
        ui.add_space(INDENT * (row.depth as f32 + 1.0));

        // The expander: a chevron when there is something to open, a gap the
        // same width when there is not, so the icons line up.
        let expandable = row.has_children || row.data_name.is_some();
        let (chevron, response) =
            ui.allocate_exact_size(egui::Vec2::splat(12.0), egui::Sense::click());
        if expandable {
            paint_chevron(ui, chevron, open, response.hovered());
            if response.clicked() {
                toggled = true;
            }
        }

        // A drawn icon for what the object is, the way Blender's tree says
        // mesh, light or camera before the name does.
        let (icon_rect, _) = ui.allocate_exact_size(egui::Vec2::splat(16.0), egui::Sense::hover());
        object_icon(ui, row.kind, icon_rect, row.shown);
        ui.interact(
            icon_rect,
            ui.id().with(("kind", row.id)),
            egui::Sense::hover(),
        )
        .on_hover_text(catalog.format(row.kind.i18n_key(), TextKind::Tooltip, &[]));

        let mut label = egui::RichText::new(&row.name);
        if row.active {
            label = label.strong();
        }
        if !row.shown {
            label = label.weak();
        }

        let typing = match renaming {
            Some(Renaming::Object(id, text)) if *id == row.id => Some(text),
            _ => None,
        };
        if let Some(text) = typing {
            if let Some(keep) = rename_field(ui, text) {
                if keep {
                    // The rename acts on the active object, which the click
                    // that began this made this one.
                    let mut params = kreate_reflect::PropertyMap::new();
                    params.set("name", kreate_reflect::PropertyValue::Text(text.clone()));
                    queue.push(Command::RunOperator {
                        op_id: "object.rename".into(),
                        params,
                    });
                }
                *renaming = None;
            }
        } else {
            let response = ui.add(
                egui::Button::selectable(row.selected, label).sense(egui::Sense::click_and_drag()),
            );
            // Drag the name onto another to parent it there, or onto a
            // collection's heading to move it in — Blender's outliner.
            if response.drag_started() {
                response.dnd_set_drag_payload(DraggedObject(row.id));
            }
            drop_highlight(ui, &response);
            if let Some(dragged) = response.dnd_release_payload::<DraggedObject>() {
                if dragged.0 != row.id {
                    queue.push(Command::ParentObject {
                        child: dragged.0,
                        parent: row.id,
                    });
                }
            }
            if response.clicked() {
                let modifiers = ui.input(|i| ClickModifiers {
                    additive: i.modifiers.ctrl || i.modifiers.command,
                    range: i.modifiers.shift,
                });
                let (objects, active) = resolve_click(order, selected, active, index, modifiers);
                queue.push(Command::SetSelection { objects, active });
            }
            // Blender's double-click: rename in place. Framing it is the
            // numpad's full stop.
            if response.double_clicked() {
                queue.push(Command::SetSelection {
                    objects: vec![row.id],
                    active: Some(row.id),
                });
                *renaming = Some(Renaming::Object(row.id, row.name.clone()));
            }
            response.context_menu(|ui| {
                if ui
                    .button(catalog.format("outliner-rename", TextKind::Ui, &[]))
                    .clicked()
                {
                    queue.push(Command::SetSelection {
                        objects: vec![row.id],
                        active: Some(row.id),
                    });
                    *renaming = Some(Renaming::Object(row.id, row.name.clone()));
                    ui.close();
                }
            });
        }

        // A shared mesh is worth flagging in place: editing it changes every
        // user, and a user who does not know that reads the result as a bug.
        if row.mesh_users > 1 {
            ui.weak(format!("\u{00D7}{}", row.mesh_users))
                .on_hover_text(catalog.format(
                    "outliner-shared-mesh",
                    TextKind::Tooltip,
                    &[("count", &row.mesh_users.to_string())],
                ));
        }

        // The eye, against the right edge where Blender keeps it. On the row
        // and not on the selection: hiding acts on what is in the way.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (eye, response) =
                ui.allocate_exact_size(egui::Vec2::splat(18.0), egui::Sense::click());
            eye_icon(ui, eye, row.visible, response.hovered());
            let response = response.on_hover_text(catalog.format(
                if row.visible {
                    "outliner-hide"
                } else {
                    "outliner-show"
                },
                TextKind::Tooltip,
                &[],
            ));
            if response.clicked() {
                queue.push(Command::SetVisible {
                    object: row.id,
                    visible: !row.visible,
                });
            }
        });
    });
    toggled
}

/// The datablock under an expanded object: what it is made of.
fn draw_data_row(ui: &mut Ui, catalog: &Catalog, row: &OutlinerRow) {
    let Some(name) = &row.data_name else { return };
    ui.horizontal(|ui| {
        ui.add_space(INDENT * (row.depth as f32 + 2.0) + 12.0);
        let (icon_rect, _) = ui.allocate_exact_size(egui::Vec2::splat(16.0), egui::Sense::hover());
        object_icon(ui, row.kind, icon_rect, true);
        // Dim: this is a fact about the object, not a thing to select. The
        // summary carries the counts a modeller wants to glance at.
        let text = match row.kind {
            RowKind::Mesh { verts, faces } => format!("{name}  {verts}v {faces}f"),
            _ => name.clone(),
        };
        ui.weak(text)
            .on_hover_text(catalog.format(row.kind.i18n_key(), TextKind::Tooltip, &[]));
    });
}

/// One material slot under an expanded object.
fn draw_material_row(ui: &mut Ui, row: &OutlinerRow, slot: usize) {
    let Some(name) = row.materials.get(slot) else {
        return;
    };
    ui.horizontal(|ui| {
        ui.add_space(INDENT * (row.depth as f32 + 3.0) + 12.0);
        let theme = crate::theme::from_ui(ui);
        let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(16.0), egui::Sense::hover());
        // The material's sphere, the same mark the properties tab wears.
        ui.painter()
            .circle_filled(rect.center(), 4.0, theme.text_dim);
        ui.weak(name);
    });
}

/// One bone under an expanded armature. A click selects it, as a click on
/// the bone in the viewport would.
fn draw_bone_row(ui: &mut Ui, row: &OutlinerRow, bone: usize, queue: &mut CommandQueue) {
    let Some((name, depth, selected)) = row.bones.get(bone) else {
        return;
    };
    ui.horizontal(|ui| {
        ui.add_space(INDENT * (row.depth as f32 + 3.0 + *depth as f32) + 12.0);
        let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(16.0), egui::Sense::hover());
        object_icon(ui, RowKind::Armature, rect, true);
        if ui.selectable_label(*selected, name).clicked() {
            let extend = ui.input(|i| i.modifiers.shift);
            queue.push(crate::Command::SelectBone {
                object: row.id,
                bone: name.clone(),
                extend,
            });
        }
    });
}

/// The expander: right when folded, down when open.
fn paint_chevron(ui: &Ui, rect: egui::Rect, open: bool, hovered: bool) {
    let theme = crate::theme::from_ui(ui);
    let ink = if hovered { theme.text } else { theme.text_dim };
    let c = rect.center();
    let s = 3.0;
    let points = if open {
        vec![
            egui::Pos2::new(c.x - s, c.y - s * 0.5),
            egui::Pos2::new(c.x + s, c.y - s * 0.5),
            egui::Pos2::new(c.x, c.y + s * 0.6),
        ]
    } else {
        vec![
            egui::Pos2::new(c.x - s * 0.5, c.y - s),
            egui::Pos2::new(c.x - s * 0.5, c.y + s),
            egui::Pos2::new(c.x + s * 0.6, c.y),
        ]
    };
    ui.painter()
        .add(egui::Shape::convex_polygon(points, ink, egui::Stroke::NONE));
}

/// Blender's collection icon: a box with a lid.
fn collection_icon(ui: &Ui, rect: egui::Rect) {
    let theme = crate::theme::from_ui(ui);
    let line = egui::Stroke::new(1.2, theme.text_dim);
    let body = egui::Rect::from_center_size(
        rect.center() + egui::Vec2::new(0.0, 1.5),
        egui::Vec2::new(11.0, 8.0),
    );
    ui.painter()
        .rect_stroke(body, 1.0, line, egui::StrokeKind::Middle);
    ui.painter().line_segment(
        [
            egui::Pos2::new(body.min.x - 1.0, body.min.y),
            egui::Pos2::new(body.max.x + 1.0, body.min.y),
        ],
        line,
    );
}

/// The icon for what an object is.
pub(crate) fn object_icon(ui: &Ui, kind: RowKind, rect: egui::Rect, visible: bool) {
    let theme = crate::theme::from_ui(ui);
    let ink = if visible {
        theme.text_dim
    } else {
        theme.text_faint
    };
    let p = ui.painter();
    let c = rect.center();
    let s = rect.width() * 0.34;
    let line = egui::Stroke::new(1.2, ink);
    match kind {
        // A cube in isometric outline: three visible faces.
        RowKind::Mesh { .. } => {
            let top = egui::Pos2::new(c.x, c.y - s);
            let left = egui::Pos2::new(c.x - s, c.y - s * 0.5);
            let right = egui::Pos2::new(c.x + s, c.y - s * 0.5);
            let bottom = egui::Pos2::new(c.x, c.y + s);
            let bl = egui::Pos2::new(c.x - s, c.y + s * 0.5);
            let br = egui::Pos2::new(c.x + s, c.y + s * 0.5);
            p.add(egui::Shape::closed_line(
                vec![top, right, br, bottom, bl, left],
                line,
            ));
            p.line_segment([left, c], line);
            p.line_segment([right, c], line);
            p.line_segment([c, bottom], line);
        }
        // A bulb: a circle on a short base.
        RowKind::Light => {
            p.circle_stroke(egui::Pos2::new(c.x, c.y - s * 0.25), s * 0.8, line);
            p.line_segment(
                [
                    egui::Pos2::new(c.x - s * 0.35, c.y + s * 0.75),
                    egui::Pos2::new(c.x + s * 0.35, c.y + s * 0.75),
                ],
                line,
            );
            p.line_segment(
                [
                    egui::Pos2::new(c.x - s * 0.25, c.y + s * 1.05),
                    egui::Pos2::new(c.x + s * 0.25, c.y + s * 1.05),
                ],
                line,
            );
        }
        // A camera: a body with a lens pointing right.
        RowKind::Camera => {
            let body = egui::Rect::from_center_size(
                egui::Pos2::new(c.x - s * 0.25, c.y),
                egui::Vec2::new(s * 1.5, s * 1.2),
            );
            p.rect_stroke(body, 1.0, line, egui::StrokeKind::Middle);
            p.add(egui::Shape::convex_polygon(
                vec![
                    egui::Pos2::new(body.max.x, c.y - s * 0.35),
                    egui::Pos2::new(body.max.x + s * 0.7, c.y - s * 0.7),
                    egui::Pos2::new(body.max.x + s * 0.7, c.y + s * 0.7),
                    egui::Pos2::new(body.max.x, c.y + s * 0.35),
                ],
                egui::Color32::TRANSPARENT,
                line,
            ));
        }
        // A bone: Blender's octahedron, seen side on — a long thin diamond
        // leaning to the right.
        RowKind::Armature => {
            let head = egui::Pos2::new(c.x - s * 0.8, c.y + s * 0.8);
            let tail = egui::Pos2::new(c.x + s * 0.8, c.y - s * 0.8);
            // A quarter of the way up, a little either side of the axis.
            let waist = head + (tail - head) * 0.25;
            let side = egui::vec2(s * 0.3, s * 0.3);
            p.add(egui::Shape::closed_line(
                vec![head, waist + side, tail, waist - side],
                line,
            ));
        }
        // An empty: three axes from a point.
        RowKind::Empty => {
            p.line_segment(
                [egui::Pos2::new(c.x - s, c.y), egui::Pos2::new(c.x + s, c.y)],
                line,
            );
            p.line_segment(
                [egui::Pos2::new(c.x, c.y - s), egui::Pos2::new(c.x, c.y + s)],
                line,
            );
        }
    }
}

/// The eye that says whether an object is shown, open or shut.
fn eye_icon(ui: &Ui, rect: egui::Rect, open: bool, hovered: bool) {
    let theme = crate::theme::from_ui(ui);
    let ink = match (open, hovered) {
        (true, _) => theme.text_dim,
        (false, true) => theme.text_dim,
        (false, false) => theme.text_faint,
    };
    let p = ui.painter();
    let c = rect.center();
    let w = rect.width() * 0.36;
    let h = rect.height() * 0.22;
    let line = egui::Stroke::new(1.2, ink);
    // The lid: two arcs meeting at the corners.
    let upper: Vec<egui::Pos2> = (0..=12)
        .map(|i| {
            let t = i as f32 / 12.0;
            let x = c.x - w + 2.0 * w * t;
            let y = c.y - h * (1.0 - (2.0 * t - 1.0).powi(2));
            egui::Pos2::new(x, y)
        })
        .collect();
    p.add(egui::Shape::line(upper, line));
    if open {
        let lower: Vec<egui::Pos2> = (0..=12)
            .map(|i| {
                let t = i as f32 / 12.0;
                let x = c.x - w + 2.0 * w * t;
                let y = c.y + h * (1.0 - (2.0 * t - 1.0).powi(2));
                egui::Pos2::new(x, y)
            })
            .collect();
        p.add(egui::Shape::line(lower, line));
        p.circle_filled(c, h * 0.55, ink);
    } else {
        // Shut: the lid alone, with lashes, which is how a closed eye reads
        // at sixteen pixels.
        for dx in [-0.6f32, 0.0, 0.6] {
            let x = c.x + dx * w;
            p.line_segment(
                [
                    egui::Pos2::new(x, c.y - h * 0.2),
                    egui::Pos2::new(x + dx * 1.5, c.y + h * 0.9),
                ],
                line,
            );
        }
    }
}

/// A one-line summary of a row, for the status bar.
pub fn summarise(catalog: &Catalog, row: &OutlinerRow) -> String {
    match row.kind {
        RowKind::Mesh { verts, faces } => catalog.format(
            "outliner-mesh-summary",
            TextKind::Ui,
            &[
                ("name", &row.name),
                ("verts", &verts.to_string()),
                ("faces", &faces.to_string()),
            ],
        ),
        other => catalog.format(other.i18n_key(), TextKind::Ui, &[("name", &row.name)]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    fn plain() -> ClickModifiers {
        ClickModifiers {
            additive: false,
            range: false,
        }
    }
    fn additive() -> ClickModifiers {
        ClickModifiers {
            additive: true,
            range: false,
        }
    }
    fn range() -> ClickModifiers {
        ClickModifiers {
            additive: false,
            range: true,
        }
    }

    #[test]
    fn a_plain_click_selects_only_that_row() {
        let (objects, active) = resolve_click(&order(5), &[0, 1, 2], Some(1), 3, plain());
        assert_eq!(objects, vec![3]);
        assert_eq!(active, Some(3));
    }

    #[test]
    fn an_additive_click_adds_to_the_selection() {
        let (objects, active) = resolve_click(&order(5), &[1], Some(1), 3, additive());
        assert_eq!(objects, vec![1, 3]);
        assert_eq!(active, Some(3));
    }

    #[test]
    fn an_additive_click_on_a_selected_row_removes_it() {
        let (objects, _) = resolve_click(&order(5), &[1, 3], Some(3), 3, additive());
        assert_eq!(objects, vec![1]);
    }

    #[test]
    fn deselecting_the_active_row_moves_the_active_marker() {
        // Otherwise later operators act on something the user cannot see is
        // chosen.
        let (objects, active) = resolve_click(&order(5), &[1, 3], Some(3), 3, additive());
        assert_eq!(active, Some(1));
        assert!(objects.contains(&1));
    }

    #[test]
    fn a_range_click_includes_both_ends() {
        let (objects, active) = resolve_click(&order(6), &[1], Some(1), 4, range());
        assert_eq!(objects, vec![1, 2, 3, 4]);
        assert_eq!(active, Some(4));
    }

    #[test]
    fn a_range_click_works_upward_too() {
        let (objects, _) = resolve_click(&order(6), &[4], Some(4), 1, range());
        assert_eq!(objects, vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_range_click_with_no_anchor_degrades_to_a_plain_click() {
        // Selecting everything would be a destructive surprise from a Shift the
        // user was holding for something else.
        let (objects, _) = resolve_click(&order(6), &[], None, 3, range());
        assert_eq!(objects, vec![3]);
    }

    #[test]
    fn a_range_click_whose_anchor_has_left_the_scene_degrades_too() {
        // The active object can be deleted between frames; ranging from an id
        // that is no longer in the order has no meaning.
        let (objects, _) = resolve_click(&order(6), &[], Some(99), 3, range());
        assert_eq!(objects, vec![3]);
    }

    #[test]
    fn a_click_outside_the_order_selects_nothing() {
        // A stale row index from a frame where the scene was larger.
        let (objects, active) = resolve_click(&order(3), &[1], Some(1), 7, plain());
        assert!(objects.is_empty());
        assert_eq!(active, None);
    }

    #[test]
    fn resolving_a_click_does_not_depend_on_row_contents() {
        // The reason this takes the order rather than materialised rows: at a
        // hundred thousand objects, building rows to answer a click is the same
        // mistake as building them to draw twenty.
        let big = order(100_000);
        let (objects, active) = resolve_click(&big, &[0], Some(0), 99_999, range());
        assert_eq!(objects.len(), 100_000);
        assert_eq!(active, Some(99_999));
    }

    // --- the tree ---------------------------------------------------------

    fn leaf(id: u32) -> OutlinerRow {
        OutlinerRow::new(id, "x", RowKind::Empty)
    }

    #[test]
    fn the_tree_starts_with_the_scene_collection() {
        let rows = visible_rows(&[], |_| false, |_| None);
        assert_eq!(rows, vec![TreeRow::Root]);
    }

    #[test]
    fn a_collapsed_parent_hides_everything_deeper_after_it() {
        // Parent at depth 0, two children at depth 1, a grandchild at 2, then a
        // sibling of the parent at depth 0 again.
        let depths = [0u16, 1, 2, 1, 0];
        let rows = visible_rows(&depths, |_| false, |i| Some(leaf(i as u32)));
        assert_eq!(
            rows,
            vec![TreeRow::Root, TreeRow::Object(0), TreeRow::Object(4)]
        );
    }

    #[test]
    fn an_expanded_parent_shows_its_children_but_their_collapsed_children_stay_hidden() {
        let depths = [0u16, 1, 2, 1, 0];
        let rows = visible_rows(&depths, |i| i == 0, |i| Some(leaf(i as u32)));
        assert_eq!(
            rows,
            vec![
                TreeRow::Root,
                TreeRow::Object(0),
                TreeRow::Object(1),
                TreeRow::Object(3),
                TreeRow::Object(4),
            ],
            "the grandchild under the collapsed child must not show"
        );
    }

    #[test]
    fn an_expanded_armature_lists_its_bones_under_its_data() {
        let rows = visible_rows(
            &[0u16],
            |_| true,
            |_| {
                let mut r = OutlinerRow::new(9, "Rig", RowKind::Armature);
                r.data_name = Some("Arm".into());
                r.bones = vec![("Upper".into(), 0, false), ("Forearm".into(), 1, true)];
                Some(r)
            },
        );
        assert_eq!(
            rows,
            vec![
                TreeRow::Root,
                TreeRow::Object(0),
                TreeRow::Data(0),
                TreeRow::Bone(0, 0),
                TreeRow::Bone(0, 1)
            ]
        );
        // Folded, no bones.
        let rows = visible_rows(&[0u16], |_| false, |_| None);
        assert_eq!(rows, vec![TreeRow::Root, TreeRow::Object(0)]);
    }

    #[test]
    fn an_expanded_object_shows_what_it_is_made_of() {
        let depths = [0u16];
        let rows = visible_rows(
            &depths,
            |_| true,
            |_| {
                let mut r = OutlinerRow::new(7, "Cube", RowKind::Mesh { verts: 8, faces: 6 });
                r.data_name = Some("Cube".into());
                r.materials = vec!["Body".into(), "Trim".into()];
                Some(r)
            },
        );
        assert_eq!(
            rows,
            vec![
                TreeRow::Root,
                TreeRow::Object(0),
                TreeRow::Data(0),
                TreeRow::Material(0, 0),
                TreeRow::Material(0, 1),
            ]
        );
    }

    #[test]
    fn an_empty_has_no_data_row_even_when_expanded() {
        let rows = visible_rows(&[0u16], |_| true, |_| Some(leaf(1)));
        assert_eq!(rows, vec![TreeRow::Root, TreeRow::Object(0)]);
    }

    #[test]
    fn a_collection_heading_comes_before_its_objects_and_an_empty_one_still_shows() {
        // Objects 0 and 1 in the first collection, the second empty, 2 loose.
        let depths = [1u16, 1, 0];
        let rows = visible_rows_grouped(&depths, &[0, 2], |_| false, |_| None);
        assert_eq!(
            rows,
            vec![
                TreeRow::Root,
                TreeRow::Collection(0),
                TreeRow::Object(0),
                TreeRow::Object(1),
                TreeRow::Collection(1),
                TreeRow::Object(2),
            ]
        );
        // A collection with nothing after it at all.
        let rows = visible_rows_grouped(&[0u16], &[1], |_| false, |_| None);
        assert_eq!(
            rows,
            vec![TreeRow::Root, TreeRow::Object(0), TreeRow::Collection(0)]
        );
    }
}
