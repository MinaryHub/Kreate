//! The shader editor: a material's graph as boxes and wires.
//!
//! Blender's node editor, for the graph `kreate-shader` compiles. Like every
//! other panel it changes nothing: it draws a [`GraphView`] and returns the
//! [`NodeEdit`]s the pointer asked for, and the application turns those into
//! the `node.*` operators — so an edit here undoes, replays and is scriptable
//! exactly as one made any other way.
//!
//! # What it knows
//!
//! Plain data: titles, positions, socket types, the values a node holds. Not a
//! `ShaderGraph` — this crate does not see the scene model, for the reason the
//! UV editor gives — and not node *kinds*. The application says what each box
//! is; the editor draws boxes.
//!
//! # A drag is one edit
//!
//! Moving nodes moves them on screen while the pointer is down and reports one
//! [`NodeEdit::Translate`] when it lets go, so a drag is one step in the
//! history rather than a step per frame. Values dragged in a field say whether
//! the pointer is still down, and the application folds those into one step
//! the way the properties panel folds a slider.
//!
//! # Layout is separate from drawing
//!
//! Where a socket sits and what the pointer is over are pure functions of the
//! view and the camera — [`socket_position`], [`hit_socket`], [`hit_node`] —
//! so "dropping a wire here connects that" is a unit test, not something only
//! a person with a mouse can check.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use crate::theme;

/// What travels along a wire, for drawing and for refusing a drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SocketType {
    Color,
    Scalar,
    Vector,
}

impl SocketType {
    /// Blender's socket colours: yellow for colour, grey for a number, purple
    /// for a vector. Borrowed because a person who knows one node editor reads
    /// a wire's type by its colour before its label.
    pub fn color(self) -> Color32 {
        match self {
            SocketType::Color => Color32::from_rgb(0xc7, 0xc7, 0x29),
            SocketType::Scalar => Color32::from_rgb(0xa1, 0xa1, 0xa1),
            SocketType::Vector => Color32::from_rgb(0x63, 0x63, 0xc7),
        }
    }
}

/// A value a node holds, drawn as a control in its box.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldView {
    /// A colour, as a swatch shows it.
    Color { field: u32, swatch: [f32; 3] },
    /// A number, with the range its control drags over. `label` is drawn in
    /// the control when the field stands alone — a strength on its own row —
    /// and left empty when a socket's label already sits beside it.
    Value { field: u32, value: f32, min: f32, max: f32, label: String },
    /// Which image a texture node reads; `None` is empty.
    Image { name: Option<String> },
}

/// One socket.
#[derive(Clone, Debug, PartialEq)]
pub struct SocketView {
    pub label: String,
    pub ty: SocketType,
    /// For an input: whether a wire feeds it. An unfed input shows its
    /// `default` instead, which is what the compiler uses in its place.
    pub connected: bool,
    pub default: Option<FieldView>,
}

/// One node.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeView {
    pub title: String,
    /// Graph units: where the box's top-left corner sits.
    pub position: [f32; 2],
    pub outputs: Vec<SocketView>,
    /// The node's own values, drawn between its outputs and its inputs.
    pub fields: Vec<FieldView>,
    pub inputs: Vec<SocketView>,
    /// What family the node is in, for the colour of its header.
    pub family: Family,
}

/// Blender's header colours, by what a node does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Input,
    Texture,
    Converter,
    Vector,
    Shader,
}

impl Family {
    fn header(self) -> Color32 {
        match self {
            Family::Input => Color32::from_rgb(0x83, 0x31, 0x4a),
            Family::Texture => Color32::from_rgb(0xa8, 0x5a, 0x24),
            Family::Converter => Color32::from_rgb(0x24, 0x62, 0x83),
            Family::Vector => Color32::from_rgb(0x3c, 0x3c, 0x83),
            Family::Shader => Color32::from_rgb(0x2b, 0x65, 0x2b),
        }
    }
}

/// A wire: `(node, output)` to `(node, input)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkView {
    pub from: (usize, usize),
    pub to: (usize, usize),
}

/// Everything one frame draws.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphView {
    pub nodes: Vec<NodeView>,
    pub links: Vec<LinkView>,
    /// The document's images, `(uid, name)`, for a texture node's picker.
    pub images: Vec<(String, String)>,
    /// What the Add menu offers, in the order `node.add` numbers them.
    pub addable: Vec<String>,
}

/// What the pointer asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeEdit {
    Translate { nodes: Vec<u32>, x: f32, y: f32 },
    Link { from_node: u32, from_socket: u32, to_node: u32, to_socket: u32 },
    Unlink { to_node: u32, to_socket: u32 },
    Add { kind: u32, x: f32, y: f32 },
    Remove { nodes: Vec<u32> },
    /// `continuing` while the pointer that is changing it is still down.
    SetColor { node: u32, field: u32, swatch: [f32; 3], continuing: bool },
    SetValue { node: u32, field: u32, value: f32, continuing: bool },
    SetImage { node: u32, uid: String },
    /// Ask for a file and put it in this texture node.
    OpenImage { node: u32 },
}

/// The editor's strings.
pub struct NodeText {
    /// Shown when there is no graph to edit, with what to do about it.
    pub no_graph: String,
    pub use_nodes: String,
    pub add: String,
    pub no_image: String,
    pub open_image: String,
}

/// Which end of a socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Output(usize, usize),
    Input(usize, usize),
}

/// What the pointer is doing.
#[derive(Clone, Debug, Default, PartialEq)]
enum Drag {
    #[default]
    None,
    /// Moving the selection; the offset is in graph units.
    Nodes { offset: Vec2 },
    /// A wire from `from` following the pointer. `detached` is the input it
    /// was picked up from, if it was lifted off one.
    Wire { from: End, detached: Option<(usize, usize)> },
    Pan,
}

/// What the editor remembers between frames, per area.
#[derive(Clone, Debug)]
pub struct NodeEditorState {
    /// Where graph origin sits relative to the panel's centre, in points.
    pub pan: Vec2,
    pub zoom: f32,
    pub selected: Vec<usize>,
    drag: Drag,
    /// Where the Add menu was opened, in graph units.
    add_at: Option<Vec2>,
    /// Whether the view has been framed on its graph yet.
    framed: bool,
    /// The panel's rect last frame, so the application can tell whether the
    /// pointer is over it.
    pub rect: Option<Rect>,
}

impl Default for NodeEditorState {
    fn default() -> Self {
        Self {
            pan: Vec2::ZERO,
            zoom: 1.0,
            selected: Vec::new(),
            drag: Drag::None,
            add_at: None,
            framed: false,
            rect: None,
        }
    }
}

// --- geometry ----------------------------------------------------------------

/// A node's width, in graph units.
pub const NODE_WIDTH: f32 = 180.0;
const HEADER: f32 = 24.0;
const ROW: f32 = 22.0;
const PAD: f32 = 6.0;
const SOCKET_RADIUS: f32 = 5.0;

/// Which row each part of a node sits on: outputs, then fields, then inputs.
fn rows(node: &NodeView) -> usize {
    node.outputs.len() + node.fields.len() + node.inputs.len()
}

/// A node's box, in graph units.
pub fn node_rect(node: &NodeView) -> Rect {
    let size = Vec2::new(NODE_WIDTH, HEADER + rows(node) as f32 * ROW + PAD);
    Rect::from_min_size(Pos2::new(node.position[0], node.position[1]), size)
}

/// The vertical middle of row `row`, in graph units from the box's top.
fn row_middle(row: usize) -> f32 {
    HEADER + (row as f32 + 0.5) * ROW
}

/// Where a socket sits, in graph units.
pub fn socket_position(view: &GraphView, end: End) -> Option<Pos2> {
    match end {
        End::Output(n, s) => {
            let node = view.nodes.get(n)?;
            (s < node.outputs.len()).then(|| {
                let r = node_rect(node);
                Pos2::new(r.max.x, r.min.y + row_middle(s))
            })
        }
        End::Input(n, s) => {
            let node = view.nodes.get(n)?;
            (s < node.inputs.len()).then(|| {
                let r = node_rect(node);
                let row = node.outputs.len() + node.fields.len() + s;
                Pos2::new(r.min.x, r.min.y + row_middle(row))
            })
        }
    }
}

/// The camera: graph units to screen points and back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub centre: Pos2,
    pub pan: Vec2,
    pub zoom: f32,
}

impl Camera {
    pub fn to_screen(&self, p: Pos2) -> Pos2 {
        self.centre + (p.to_vec2() + self.pan) * self.zoom
    }
    pub fn to_graph(&self, p: Pos2) -> Pos2 {
        ((p - self.centre) / self.zoom - self.pan).to_pos2()
    }
    fn rect(&self, r: Rect) -> Rect {
        Rect::from_min_max(self.to_screen(r.min), self.to_screen(r.max))
    }
}

/// The socket under a screen point, if any. Generous on purpose: a socket is
/// a small target and missing it starts a node drag instead.
pub fn hit_socket(view: &GraphView, camera: &Camera, pointer: Pos2) -> Option<End> {
    let reach = (SOCKET_RADIUS * 2.2 * camera.zoom).max(8.0);
    let mut best: Option<(f32, End)> = None;
    for (n, node) in view.nodes.iter().enumerate() {
        let ends = (0..node.outputs.len())
            .map(move |s| End::Output(n, s))
            .chain((0..node.inputs.len()).map(move |s| End::Input(n, s)));
        for end in ends {
            let Some(at) = socket_position(view, end) else { continue };
            let d = camera.to_screen(at).distance(pointer);
            if d <= reach && best.is_none_or(|(b, _)| d < b) {
                best = Some((d, end));
            }
        }
    }
    best.map(|(_, end)| end)
}

/// The node under a screen point: the topmost, which is the last drawn.
pub fn hit_node(view: &GraphView, camera: &Camera, pointer: Pos2) -> Option<usize> {
    (0..view.nodes.len())
        .rev()
        .find(|n| camera.rect(node_rect(&view.nodes[*n])).contains(pointer))
}

/// Whether a wire may be dropped from `a` onto `b`: an output onto an input
/// or the other way, of one type, between two different nodes.
pub fn compatible(view: &GraphView, a: End, b: End) -> Option<(usize, usize, usize, usize)> {
    let (out, inp) = match (a, b) {
        (End::Output(on, os), End::Input(inn, is)) | (End::Input(inn, is), End::Output(on, os)) => {
            ((on, os), (inn, is))
        }
        _ => return None,
    };
    if out.0 == inp.0 {
        return None;
    }
    let from = view.nodes.get(out.0)?.outputs.get(out.1)?;
    let to = view.nodes.get(inp.0)?.inputs.get(inp.1)?;
    (from.ty == to.ty).then_some((out.0, out.1, inp.0, inp.1))
}

/// The pan and zoom that fit every node in `size`.
pub fn frame(view: &GraphView, size: Vec2) -> (Vec2, f32) {
    let Some(bounds) = view
        .nodes
        .iter()
        .map(node_rect)
        .reduce(|a, b| a.union(b))
    else {
        return (Vec2::ZERO, 1.0);
    };
    let margin = 40.0;
    let zoom = ((size.x - margin) / bounds.width())
        .min((size.y - margin) / bounds.height())
        .clamp(MIN_ZOOM, 1.0);
    (-bounds.center().to_vec2(), zoom)
}

const MIN_ZOOM: f32 = 0.4;
const MAX_ZOOM: f32 = 2.0;

// --- the panel ---------------------------------------------------------------

/// Draws the editor. `view` is `None` when the material has no graph, and the
/// panel offers to make one.
pub fn node_editor(
    ui: &mut Ui,
    view: Option<&GraphView>,
    state: &mut NodeEditorState,
    text: &NodeText,
    use_nodes: &mut bool,
) -> Vec<NodeEdit> {
    let mut out = Vec::new();
    let theme = theme::from_ui(ui);
    let rect = ui.available_rect_before_wrap();
    if rect.width() < 32.0 || rect.height() < 32.0 {
        return out;
    }
    state.rect = Some(rect);
    let background = ui.allocate_rect(rect, Sense::click_and_drag());
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, theme.sunken);

    let Some(view) = view else {
        state.framed = false;
        let centre = rect.center();
        ui.scope_builder(
            egui::UiBuilder::new().max_rect(Rect::from_center_size(centre, Vec2::new(260.0, 60.0))),
            |ui| {
                ui.vertical_centered(|ui| {
                    ui.weak(&text.no_graph);
                    if ui.button(&text.use_nodes).clicked() {
                        *use_nodes = true;
                    }
                });
            },
        );
        return out;
    };
    // Every node index the state holds must still name a node: the graph may
    // have lost some since last frame, by undo or by a script.
    state.selected.retain(|n| *n < view.nodes.len());

    if !state.framed {
        let (pan, zoom) = frame(view, rect.size());
        state.pan = pan;
        state.zoom = zoom;
        state.framed = true;
    }

    // Zoom about the pointer, pan with the middle button.
    let hovered = ui.rect_contains_pointer(rect);
    if hovered {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
                let before = Camera { centre: rect.center(), pan: state.pan, zoom: state.zoom };
                let anchor = before.to_graph(pointer);
                state.zoom = (state.zoom * (1.0 + scroll * 0.0015)).clamp(MIN_ZOOM, MAX_ZOOM);
                let after = Camera { centre: rect.center(), pan: state.pan, zoom: state.zoom };
                // Keep the graph point under the pointer where it was.
                state.pan += after.to_graph(pointer) - anchor;
            }
        }
    }
    if background.dragged_by(egui::PointerButton::Middle) {
        state.pan += background.drag_delta() / state.zoom;
        state.drag = Drag::Pan;
    }

    let camera = Camera { centre: rect.center(), pan: state.pan, zoom: state.zoom };
    draw_grid(&painter, rect, &camera, theme.base);

    // Where each node is drawn this frame: moved by a drag in progress.
    let offset = match state.drag {
        Drag::Nodes { offset } => offset,
        _ => Vec2::ZERO,
    };
    let shown = |n: usize| -> Vec2 {
        if state.selected.contains(&n) { offset } else { Vec2::ZERO }
    };

    // Wires under the boxes.
    for link in &view.links {
        let (Some(a), Some(b)) = (
            socket_position(view, End::Output(link.from.0, link.from.1)),
            socket_position(view, End::Input(link.to.0, link.to.1)),
        ) else {
            continue;
        };
        let ty = view.nodes[link.from.0].outputs[link.from.1].ty;
        let picked_up = matches!(state.drag, Drag::Wire { detached: Some(d), .. } if d == link.to);
        if picked_up {
            continue;
        }
        wire(
            &painter,
            camera.to_screen(a + shown(link.from.0)),
            camera.to_screen(b + shown(link.to.0)),
            ty.color(),
            camera.zoom,
        );
    }

    // The boxes, and the controls inside them.
    let pointer = ui.input(|i| i.pointer.interact_pos());
    for (n, node) in view.nodes.iter().enumerate() {
        draw_node(ui, &painter, view, n, node, &camera, shown(n), state, &theme, text, &mut out);
    }

    // A wire being dragged.
    if let Drag::Wire { from, .. } = state.drag {
        if let (Some(start), Some(p)) = (socket_position(view, from), pointer) {
            let ty = socket_type(view, from).unwrap_or(SocketType::Scalar);
            let start = camera.to_screen(start);
            let (a, b) = match from {
                End::Output(..) => (start, p),
                End::Input(..) => (p, start),
            };
            wire(&painter, a, b, ty.color(), camera.zoom);
        }
    }

    // --- the pointer ---------------------------------------------------------
    let press = ui.input(|i| i.pointer.primary_pressed());
    let release = ui.input(|i| i.pointer.primary_released());
    let shift = ui.input(|i| i.modifiers.shift);
    // Only presses that land on the background or a node's body count here.
    // A control inside a box is a widget above the background, and egui's
    // hit test gives it the press — so the background is not "hovered" then,
    // and a click on a colour does not also start dragging its node.
    let on_panel = background.hovered();

    if press && on_panel {
        if let Some(p) = pointer {
            if let Some(end) = hit_socket(view, &camera, p) {
                state.drag = match end {
                    // Lifting a wire off an input picks the wire up, the way
                    // Blender does: drop it elsewhere to move it, or on
                    // nothing to disconnect it.
                    End::Input(n, s) => match view.links.iter().find(|l| l.to == (n, s)) {
                        Some(link) => Drag::Wire {
                            from: End::Output(link.from.0, link.from.1),
                            detached: Some((n, s)),
                        },
                        None => Drag::Wire { from: end, detached: None },
                    },
                    End::Output(..) => Drag::Wire { from: end, detached: None },
                };
            } else if let Some(n) = hit_node(view, &camera, p) {
                if shift {
                    if let Some(at) = state.selected.iter().position(|s| *s == n) {
                        state.selected.remove(at);
                    } else {
                        state.selected.push(n);
                    }
                } else if !state.selected.contains(&n) {
                    state.selected = vec![n];
                }
                if state.selected.contains(&n) {
                    state.drag = Drag::Nodes { offset: Vec2::ZERO };
                }
            } else if !shift {
                state.selected.clear();
            }
        }
    }

    if let Drag::Nodes { offset } = &mut state.drag {
        let delta = ui.input(|i| i.pointer.delta());
        *offset += delta / camera.zoom;
    }

    if release {
        match std::mem::take(&mut state.drag) {
            Drag::Nodes { offset } if offset.length() > 0.0 => {
                let mut nodes: Vec<u32> = state.selected.iter().map(|n| *n as u32).collect();
                nodes.sort_unstable();
                out.push(NodeEdit::Translate { nodes, x: offset.x, y: offset.y });
            }
            Drag::Wire { from, detached } => {
                let target = pointer.and_then(|p| hit_socket(view, &camera, p));
                let joined = target.and_then(|t| compatible(view, from, t));
                match (joined, detached) {
                    (Some((fnode, fsock, tnode, tsock)), detached) => {
                        if let Some((dn, ds)) = detached {
                            if (dn, ds) != (tnode, tsock) {
                                out.push(NodeEdit::Unlink { to_node: dn as u32, to_socket: ds as u32 });
                            }
                        }
                        out.push(NodeEdit::Link {
                            from_node: fnode as u32,
                            from_socket: fsock as u32,
                            to_node: tnode as u32,
                            to_socket: tsock as u32,
                        });
                    }
                    (None, Some((dn, ds))) => {
                        out.push(NodeEdit::Unlink { to_node: dn as u32, to_socket: ds as u32 });
                    }
                    (None, None) => {}
                }
            }
            _ => {}
        }
    }

    // --- keys and the Add menu -----------------------------------------------
    if hovered && !ui.ctx().egui_wants_keyboard_input() {
        let (delete, add, home) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::X) || i.key_pressed(egui::Key::Delete),
                i.modifiers.shift && i.key_pressed(egui::Key::A),
                i.key_pressed(egui::Key::Home),
            )
        });
        if delete && !state.selected.is_empty() {
            let mut nodes: Vec<u32> = state.selected.iter().map(|n| *n as u32).collect();
            nodes.sort_unstable();
            out.push(NodeEdit::Remove { nodes });
            state.selected.clear();
        }
        if add {
            state.add_at = pointer.map(|p| camera.to_graph(p).to_vec2());
        }
        if home {
            let (pan, zoom) = frame(view, rect.size());
            state.pan = pan;
            state.zoom = zoom;
        }
    }
    if background.secondary_clicked() {
        state.add_at = pointer.map(|p| camera.to_graph(p).to_vec2());
    }
    if let Some(at) = state.add_at {
        let anchor = camera.to_screen(at.to_pos2());
        let mut close = false;
        egui::Area::new(ui.id().with("node-add"))
            .order(egui::Order::Foreground)
            .fixed_pos(anchor)
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(egui::RichText::new(&text.add).strong());
                    ui.separator();
                    for (kind, label) in view.addable.iter().enumerate() {
                        if ui.button(label).clicked() {
                            out.push(NodeEdit::Add { kind: kind as u32, x: at.x, y: at.y });
                            close = true;
                        }
                    }
                });
            });
        let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
        let clicked_away = press && on_panel;
        if close || escape || clicked_away {
            state.add_at = None;
        }
    }
    out
}

fn socket_type(view: &GraphView, end: End) -> Option<SocketType> {
    match end {
        End::Output(n, s) => view.nodes.get(n)?.outputs.get(s).map(|x| x.ty),
        End::Input(n, s) => view.nodes.get(n)?.inputs.get(s).map(|x| x.ty),
    }
}

fn draw_grid(painter: &egui::Painter, rect: Rect, camera: &Camera, colour: Color32) {
    let step = 40.0 * camera.zoom;
    if step < 8.0 {
        return;
    }
    let origin = camera.to_screen(Pos2::ZERO);
    let first_x = rect.min.x + (origin.x - rect.min.x).rem_euclid(step);
    let first_y = rect.min.y + (origin.y - rect.min.y).rem_euclid(step);
    let mut x = first_x;
    while x < rect.max.x {
        let mut y = first_y;
        while y < rect.max.y {
            painter.circle_filled(Pos2::new(x, y), 1.0, colour);
            y += step;
        }
        x += step;
    }
}

/// A wire: a curve leaving the output going right and arriving at the input
/// from the left, the way every node editor draws one.
fn wire(painter: &egui::Painter, a: Pos2, b: Pos2, colour: Color32, zoom: f32) {
    let reach = ((b.x - a.x).abs() * 0.5).max(40.0 * zoom);
    let curve = egui::epaint::CubicBezierShape::from_points_stroke(
        [a, a + Vec2::new(reach, 0.0), b - Vec2::new(reach, 0.0), b],
        false,
        Color32::TRANSPARENT,
        Stroke::new(2.0 * zoom.max(0.6), colour),
    );
    painter.add(curve);
}

#[allow(clippy::too_many_arguments)]
fn draw_node(
    ui: &mut Ui,
    painter: &egui::Painter,
    view: &GraphView,
    n: usize,
    node: &NodeView,
    camera: &Camera,
    offset: Vec2,
    state: &NodeEditorState,
    theme: &theme::Theme,
    text: &NodeText,
    out: &mut Vec<NodeEdit>,
) {
    let graph_rect = node_rect(node).translate(offset);
    let r = camera.rect(graph_rect);
    if !painter.clip_rect().intersects(r) {
        return;
    }
    let z = camera.zoom;
    let radius = (5.0 * z) as u8;
    painter.rect_filled(r, radius, theme.raised);
    let header = Rect::from_min_size(r.min, Vec2::new(r.width(), HEADER * z));
    painter.rect_filled(
        header,
        egui::CornerRadius { nw: radius, ne: radius, sw: 0, se: 0 },
        node.family.header(),
    );
    let font = egui::FontId::proportional(12.5 * z);
    painter.text(
        header.left_center() + Vec2::new(8.0 * z, 0.0),
        egui::Align2::LEFT_CENTER,
        &node.title,
        font.clone(),
        Color32::WHITE,
    );
    let selected = state.selected.contains(&n);
    let outline = if selected {
        Stroke::new(2.0, theme.accent)
    } else {
        Stroke::new(1.0, theme.border)
    };
    painter.rect_stroke(r, radius, outline, egui::StrokeKind::Outside);

    let row_rect = |row: usize| {
        let top = graph_rect.min.y + HEADER + row as f32 * ROW;
        camera.rect(Rect::from_min_size(
            Pos2::new(graph_rect.min.x + 10.0, top + 1.0),
            Vec2::new(NODE_WIDTH - 20.0, ROW - 2.0),
        ))
    };
    let dot = |at: Pos2, ty: SocketType| {
        painter.circle(at, SOCKET_RADIUS * z, ty.color(), Stroke::new(1.0, Color32::BLACK));
    };

    let mut row = 0;
    for (s, socket) in node.outputs.iter().enumerate() {
        let at = camera.to_screen(socket_position(view, End::Output(n, s)).unwrap_or_default() + offset);
        dot(at, socket.ty);
        painter.text(
            row_rect(row).right_center(),
            egui::Align2::RIGHT_CENTER,
            &socket.label,
            font.clone(),
            theme.text,
        );
        row += 1;
    }
    let show_controls = z >= 0.45;
    for field in &node.fields {
        if show_controls {
            control(ui, row_rect(row), (n, row), field, &font, view, text, out);
        }
        row += 1;
    }
    for (s, socket) in node.inputs.iter().enumerate() {
        let at = camera.to_screen(socket_position(view, End::Input(n, s)).unwrap_or_default() + offset);
        dot(at, socket.ty);
        let cell = row_rect(row);
        match (&socket.default, socket.connected, show_controls) {
            (Some(field), false, true) => {
                // Label on the left half, the value it takes on the right.
                let (left, right) = cell.split_left_right_at_fraction(0.45);
                painter.text(left.left_center(), egui::Align2::LEFT_CENTER, &socket.label, font.clone(), theme.text_dim);
                control(ui, right, (n, row), field, &font, view, text, out);
            }
            _ => {
                painter.text(cell.left_center(), egui::Align2::LEFT_CENTER, &socket.label, font.clone(), theme.text_dim);
            }
        }
        row += 1;
    }
}

/// One field's control, placed in `cell`.
#[allow(clippy::too_many_arguments)]
fn control(
    ui: &mut Ui,
    cell: Rect,
    (n, row): (usize, usize),
    field: &FieldView,
    font: &egui::FontId,
    view: &GraphView,
    text: &NodeText,
    out: &mut Vec<NodeEdit>,
) {
    let down = ui.input(|i| i.pointer.any_down());
    ui.scope_builder(egui::UiBuilder::new().max_rect(cell).id_salt(("node-field", n, row)), |ui| {
        ui.style_mut().override_font_id = Some(font.clone());
        ui.spacing_mut().interact_size.y = cell.height();
        match field {
            FieldView::Color { field, swatch } => {
                let mut value = *swatch;
                let response = ui.add_sized(cell.size(), |ui: &mut Ui| {
                    egui::color_picker::color_edit_button_rgb(ui, &mut value)
                });
                if response.changed() || value != *swatch {
                    out.push(NodeEdit::SetColor { node: n as u32, field: *field, swatch: value, continuing: down });
                }
            }
            FieldView::Value { field, value, min, max, label } => {
                let mut v = *value;
                let prefix = if label.is_empty() { String::new() } else { format!("{label}  ") };
                let response = ui.add_sized(
                    cell.size(),
                    egui::DragValue::new(&mut v)
                        .range(*min..=*max)
                        .speed(((*max - *min) / 200.0) as f64)
                        .max_decimals(3)
                        .prefix(prefix),
                );
                if response.changed() && v != *value {
                    out.push(NodeEdit::SetValue { node: n as u32, field: *field, value: v, continuing: down });
                }
            }
            FieldView::Image { name } => {
                let shown = name.clone().unwrap_or_else(|| text.no_image.clone());
                egui::ComboBox::from_id_salt(("node-image", n))
                    .width(cell.width())
                    .selected_text(shown)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(name.is_none(), &text.no_image).clicked() {
                            out.push(NodeEdit::SetImage { node: n as u32, uid: String::new() });
                        }
                        for (uid, label) in &view.images {
                            // i18n-exempt: an image's name is the user's own text
                            if ui.selectable_label(name.as_deref() == Some(label.as_str()), label).clicked() {
                                out.push(NodeEdit::SetImage { node: n as u32, uid: uid.clone() });
                            }
                        }
                        ui.separator();
                        if ui.button(&text.open_image).clicked() {
                            out.push(NodeEdit::OpenImage { node: n as u32 });
                        }
                    });
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn socket(label: &str, ty: SocketType) -> SocketView {
        SocketView { label: label.into(), ty, connected: false, default: None }
    }

    /// A Value node on the left feeding a surface on the right.
    fn two() -> GraphView {
        GraphView {
            nodes: vec![
                NodeView {
                    title: "Value".into(),
                    position: [-300.0, 0.0],
                    outputs: vec![socket("Value", SocketType::Scalar)],
                    fields: vec![FieldView::Value {
                        field: 0,
                        value: 0.5,
                        min: 0.0,
                        max: 1.0,
                        label: String::new(),
                    }],
                    inputs: vec![],
                    family: Family::Input,
                },
                NodeView {
                    title: "Surface".into(),
                    position: [0.0, 0.0],
                    outputs: vec![],
                    fields: vec![],
                    inputs: vec![
                        socket("Base Color", SocketType::Color),
                        socket("Metallic", SocketType::Scalar),
                        socket("Roughness", SocketType::Scalar),
                    ],
                    family: Family::Shader,
                },
            ],
            links: vec![],
            images: vec![],
            addable: vec![],
        }
    }

    fn camera() -> Camera {
        Camera { centre: Pos2::new(400.0, 300.0), pan: Vec2::new(50.0, -20.0), zoom: 1.5 }
    }

    #[test]
    fn the_camera_goes_there_and_back() {
        let c = camera();
        for p in [Pos2::ZERO, Pos2::new(-300.0, 45.0), Pos2::new(12.5, -99.0)] {
            let back = c.to_graph(c.to_screen(p));
            assert!(back.distance(p) < 1e-3, "{p:?} came back as {back:?}");
        }
    }

    #[test]
    fn outputs_are_on_the_right_and_inputs_on_the_left_in_row_order() {
        let view = two();
        let out = socket_position(&view, End::Output(0, 0)).unwrap();
        assert_eq!(out.x, -300.0 + NODE_WIDTH);
        let inputs: Vec<Pos2> = (0..3).map(|s| socket_position(&view, End::Input(1, s)).unwrap()).collect();
        assert!(inputs.iter().all(|p| p.x == 0.0));
        assert!(inputs.windows(2).all(|w| w[0].y < w[1].y), "inputs are not in order down the box");
        // Every socket is inside its box's height.
        let r = node_rect(&view.nodes[1]);
        assert!(inputs.iter().all(|p| p.y > r.min.y + HEADER && p.y < r.max.y));
        assert_eq!(socket_position(&view, End::Input(1, 3)), None);
    }

    #[test]
    fn the_socket_under_the_pointer_is_the_one_it_is_on() {
        let view = two();
        let c = camera();
        for end in [End::Output(0, 0), End::Input(1, 0), End::Input(1, 1), End::Input(1, 2)] {
            let at = c.to_screen(socket_position(&view, end).unwrap());
            assert_eq!(hit_socket(&view, &c, at + Vec2::new(2.0, 1.0)), Some(end));
        }
        // And the middle of a box is a node, not a socket.
        let middle = c.to_screen(node_rect(&view.nodes[1]).center());
        assert_eq!(hit_socket(&view, &c, middle), None);
        assert_eq!(hit_node(&view, &c, middle), Some(1));
        assert_eq!(hit_node(&view, &c, c.to_screen(Pos2::new(-1000.0, -1000.0))), None);
    }

    #[test]
    fn a_wire_joins_an_output_to_an_input_of_its_own_type_only() {
        let view = two();
        // A number into roughness, from either end.
        assert_eq!(compatible(&view, End::Output(0, 0), End::Input(1, 2)), Some((0, 0, 1, 2)));
        assert_eq!(compatible(&view, End::Input(1, 2), End::Output(0, 0)), Some((0, 0, 1, 2)));
        // A number into a colour, an input into an input.
        assert_eq!(compatible(&view, End::Output(0, 0), End::Input(1, 0)), None);
        assert_eq!(compatible(&view, End::Input(1, 1), End::Input(1, 2)), None);
    }

    #[test]
    fn framing_puts_every_node_on_screen() {
        let view = two();
        let size = Vec2::new(500.0, 300.0);
        let (pan, zoom) = frame(&view, size);
        let c = Camera { centre: Pos2::new(250.0, 150.0), pan, zoom };
        let screen = Rect::from_min_size(Pos2::ZERO, size);
        for node in &view.nodes {
            let r = c.rect(node_rect(node));
            assert!(screen.contains_rect(r), "{} is off screen: {r:?}", node.title);
        }
    }

    // --- gestures, through egui ----------------------------------------------
    //
    // The functions above say where things are; these say what a *drag* does,
    // by feeding the editor real pointer events a frame at a time. Without
    // them the only check on "pull a wire and let go" is a person with a mouse.

    struct Rig {
        ctx: egui::Context,
        state: NodeEditorState,
        view: GraphView,
        time: f64,
    }

    impl Rig {
        fn new(view: GraphView) -> Self {
            let mut rig = Self {
                ctx: egui::Context::default(),
                state: NodeEditorState::default(),
                view,
                time: 0.0,
            };
            // Two quiet frames: the first frames the view, the second gives
            // egui last frame's widget rects to hit-test against.
            rig.frame(vec![]);
            rig.frame(vec![]);
            rig
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> Vec<NodeEdit> {
            self.time += 1.0 / 60.0;
            let raw = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 600.0))),
                time: Some(self.time),
                events,
                ..Default::default()
            };
            let text = NodeText {
                no_graph: String::new(),
                use_nodes: String::new(),
                add: String::new(),
                no_image: String::new(),
                open_image: String::new(),
            };
            let mut out = Vec::new();
            let (state, view) = (&mut self.state, &self.view);
            let _ = self.ctx.run_ui(raw, |ui| {
                let mut asked = false;
                out = node_editor(ui, Some(view), state, &text, &mut asked);
            });
            out
        }

        fn camera(&self) -> Camera {
            Camera {
                centre: self.state.rect.expect("drawn").center(),
                pan: self.state.pan,
                zoom: self.state.zoom,
            }
        }

        fn screen(&self, end: End) -> Pos2 {
            self.camera().to_screen(socket_position(&self.view, end).unwrap())
        }

        /// Press at `a`, drag to `b` over a few frames, release; every edit
        /// the gesture produced.
        fn drag(&mut self, a: Pos2, b: Pos2) -> Vec<NodeEdit> {
            let button = |pos, pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            };
            let mut out = Vec::new();
            out.extend(self.frame(vec![egui::Event::PointerMoved(a)]));
            out.extend(self.frame(vec![button(a, true)]));
            for step in 1..=4 {
                let p = a + (b - a) * (step as f32 / 4.0);
                out.extend(self.frame(vec![egui::Event::PointerMoved(p)]));
            }
            out.extend(self.frame(vec![button(b, false)]));
            out.extend(self.frame(vec![]));
            out
        }
    }

    #[test]
    fn dragging_a_node_is_one_translate_by_the_distance_dragged() {
        let mut rig = Rig::new(two());
        let zoom = rig.state.zoom;
        let body = rig.camera().body_of(0);
        let edits = rig.drag(body, body + Vec2::new(60.0, 30.0));
        let moves: Vec<&NodeEdit> =
            edits.iter().filter(|e| matches!(e, NodeEdit::Translate { .. })).collect();
        assert_eq!(moves.len(), 1, "a drag was not one edit: {edits:?}");
        let NodeEdit::Translate { nodes, x, y } = moves[0] else { unreachable!() };
        assert_eq!(nodes, &vec![0]);
        // Screen points back to graph units.
        assert!((x - 60.0 / zoom).abs() < 0.5 && (y - 30.0 / zoom).abs() < 0.5, "{x} {y}");
    }

    #[test]
    fn pulling_a_wire_from_an_output_to_an_input_links_them() {
        let mut rig = Rig::new(two());
        let from = rig.screen(End::Output(0, 0));
        let to = rig.screen(End::Input(1, 2));
        let edits = rig.drag(from, to);
        assert_eq!(
            edits,
            vec![NodeEdit::Link { from_node: 0, from_socket: 0, to_node: 1, to_socket: 2 }]
        );
    }

    #[test]
    fn a_wire_dropped_on_the_wrong_type_does_nothing() {
        let mut rig = Rig::new(two());
        let from = rig.screen(End::Output(0, 0));
        let to = rig.screen(End::Input(1, 0));
        assert_eq!(rig.drag(from, to), vec![]);
    }

    #[test]
    fn a_wire_lifted_off_an_input_and_dropped_on_nothing_is_disconnected() {
        let mut view = two();
        view.links.push(LinkView { from: (0, 0), to: (1, 2) });
        view.nodes[1].inputs[2].connected = true;
        let mut rig = Rig::new(view);
        let from = rig.screen(End::Input(1, 2));
        let edits = rig.drag(from, from + Vec2::new(-40.0, 120.0));
        assert_eq!(edits, vec![NodeEdit::Unlink { to_node: 1, to_socket: 2 }]);
    }

    #[test]
    fn a_wire_lifted_off_one_input_onto_another_moves_there() {
        let mut view = two();
        view.links.push(LinkView { from: (0, 0), to: (1, 2) });
        view.nodes[1].inputs[2].connected = true;
        let mut rig = Rig::new(view);
        let from = rig.screen(End::Input(1, 2));
        let to = rig.screen(End::Input(1, 1));
        assert_eq!(
            rig.drag(from, to),
            vec![
                NodeEdit::Unlink { to_node: 1, to_socket: 2 },
                NodeEdit::Link { from_node: 0, from_socket: 0, to_node: 1, to_socket: 1 },
            ]
        );
    }

    #[test]
    fn x_deletes_what_is_selected_and_only_that() {
        let mut rig = Rig::new(two());
        let body = rig.camera().body_of(1);
        // A click selects without moving.
        let edits = rig.drag(body, body);
        assert!(edits.is_empty(), "a click moved something: {edits:?}");
        assert_eq!(rig.state.selected, vec![1]);
        let key = |pressed| egui::Event::Key {
            key: egui::Key::X,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: Default::default(),
        };
        let edits = rig.frame(vec![egui::Event::PointerMoved(body), key(true)]);
        assert_eq!(edits, vec![NodeEdit::Remove { nodes: vec![1] }]);
        rig.frame(vec![key(false)]);
        assert!(rig.state.selected.is_empty());
    }

    #[test]
    fn dragging_in_a_field_changes_the_value_and_not_the_node() {
        // The control sits on top of the box. A drag there is the control's:
        // the value changes, and the node stays where it is.
        let mut rig = Rig::new(two());
        let field = rig.camera().to_screen(Pos2::new(
            -300.0 + NODE_WIDTH / 2.0,
            HEADER + ROW * 1.5,
        ));
        let edits = rig.drag(field, field + Vec2::new(80.0, 0.0));
        assert!(
            !edits.iter().any(|e| matches!(e, NodeEdit::Translate { .. })),
            "a drag in a field moved the node: {edits:?}"
        );
        assert!(
            edits.iter().any(|e| matches!(e, NodeEdit::SetValue { node: 0, field: 0, .. })),
            "the field did not change: {edits:?}"
        );
    }

    impl Camera {
        /// The centre of node `n`'s box, on screen — its body, away from
        /// every socket.
        fn body_of(&self, n: usize) -> Pos2 {
            let view = two();
            self.to_screen(node_rect(&view.nodes[n]).center() + Vec2::new(0.0, 4.0))
        }
    }
}

