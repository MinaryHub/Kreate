//! Editing a material's shading graph.
//!
//! What the node editor's gestures become. Every one of them is an operator
//! for the reasons every edit is: it undoes, it replays from the history, and
//! a script can say it. The editor itself changes nothing — it reports what the
//! pointer did and these decide what that means.
//!
//! # Which graph
//!
//! The material in a slot of the active object, the same `slot` parameter the
//! Material tab's operators take and for the same reason: the slot is
//! interface state, and an operator that read it from the interface could not
//! be replayed.
//!
//! # Nodes by index
//!
//! A node is named by its position in the graph, which is what a link names it
//! by too. Removing a node renumbers the ones after it; the editor rebuilds its
//! view from the graph every frame, so it never holds a stale index across an
//! edit. A script that removes and then refers must count, the way it would
//! with any list.
//!
//! # The surface stays
//!
//! Every edit here leaves the graph with exactly one surface. Adding refuses a
//! second (the graph already does) and removing skips the one there is. A
//! material with no output compiles to nothing, and the moment to refuse that
//! is the edit — not three frames later in a viewport that has gone grey.

use kreate_ops_derive::OpParams;
use kreate_scene::shader::{Link, Node, NodeKind, ShaderGraph};
use kreate_scene::{DocMut, Document, MaterialId};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::ops::object::{active_slot_count, material_in_slot};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<UseNodes>();
    registry.register::<AddNode>();
    registry.register::<RemoveNodes>();
    registry.register::<LinkNodes>();
    registry.register::<Unlink>();
    registry.register::<TranslateNodes>();
    registry.register::<SetNodeColor>();
    registry.register::<SetNodeValue>();
    registry.register::<SetNodeImage>();
}

/// The graph of the material in `slot`, to edit, or `None` when there is no
/// such material or it has no graph.
fn graph_in<'d>(doc: &'d mut DocMut<'_>, slot: u32) -> Option<&'d mut ShaderGraph> {
    let id = material_in_slot(doc.document(), slot)?;
    doc.document_mut().material_mut(id)?.graph.as_mut()
}

/// Whether the active object's material in `slot` has a graph.
pub fn has_graph(doc: &Document, slot: u32) -> bool {
    material_in_slot(doc, slot)
        .and_then(|id| doc.material(id))
        .is_some_and(|m| m.graph.is_some())
}

fn any_graph(doc: &Document) -> bool {
    (0..active_slot_count(doc) as u32).any(|slot| has_graph(doc, slot))
}

// --- use nodes --------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct UseNodesParams {
    #[param(min = 0.0)]
    pub slot: u32,
    /// On builds the graph from the flat fields; off takes it away and the
    /// flat fields draw again.
    pub enabled: bool,
}

impl Default for UseNodesParams {
    fn default() -> Self {
        Self {
            slot: 0,
            enabled: true,
        }
    }
}

/// Blender's "Use Nodes": gives a material a graph of its own, or takes it away.
///
/// **On builds the graph the material already is.** `ShaderGraph::for_material`
/// is the definition of what the flat fields mean, so turning nodes on changes
/// no pixel — the textures, the tint, the packed map are all there as nodes,
/// wired the way they were being drawn. A fresh surface node instead would
/// throw away every texture the moment somebody opened the editor to look.
///
/// Off discards the graph, and undo brings it back. The flat fields were never
/// touched, so the material returns to what it was before nodes.
pub struct UseNodes;

impl Operator for UseNodes {
    type Params = UseNodesParams;
    fn id() -> &'static str {
        "material.use_nodes"
    }
    fn label() -> &'static str {
        "Use Nodes"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        active_slot_count(doc) > 0
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = material_in_slot(doc.document(), params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(material) = doc.document_mut().material_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        match (params.enabled, material.graph.is_some()) {
            (true, false) => material.graph = Some(ShaderGraph::for_material(material)),
            (false, true) => material.graph = None,
            // Already so. A cancel rather than an edit, so undo brings back
            // the last change and not a no-op.
            _ => return Ok(OpResult::Cancelled),
        }
        Ok(OpResult::Finished)
    }
}

// --- add ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddNodeParams {
    #[param(min = 0.0)]
    pub slot: u32,
    /// Which kind, in the order of `NodeKind::ADDABLE`.
    #[param(variants(
        "color",
        "value",
        "image",
        "multiply",
        "multiply_value",
        "separate_rgb",
        "normal_map"
    ))]
    pub kind: u32,
    /// Where it lands, in graph units.
    pub x: f64,
    pub y: f64,
}

/// Adds a node where the pointer was.
pub struct AddNode;

impl Operator for AddNode {
    type Params = AddNodeParams;
    fn id() -> &'static str {
        "node.add"
    }
    fn label() -> &'static str {
        "Add Node"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(tag) = NodeKind::ADDABLE.get(params.kind as usize) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(kind) = NodeKind::new_of(tag) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(graph) = graph_in(doc, params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        let mut node = Node::new(default_name(&kind), kind);
        node.position = [params.x as f32, params.y as f32];
        match graph.add_node(node) {
            Some(_) => Ok(OpResult::Finished),
            None => Ok(OpResult::Cancelled),
        }
    }
}

/// What a new node is called: what it is, the way Blender titles its nodes.
pub fn default_name(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Color(_) => "RGB",
        NodeKind::Value(_) => "Value",
        NodeKind::Image { .. } => "Image Texture",
        NodeKind::Multiply => "Mix (Multiply)",
        NodeKind::MultiplyValue => "Math (Multiply)",
        NodeKind::SeparateRgb => "Separate RGB",
        NodeKind::NormalMap { .. } => "Normal Map",
        NodeKind::Surface { .. } => "Principled BSDF",
    }
}

// --- remove ------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct RemoveNodesParams {
    #[param(min = 0.0)]
    pub slot: u32,
    /// The nodes, by index.
    pub nodes: Vec<u32>,
}

/// Removes nodes and every wire touching them. The surface is never removed.
pub struct RemoveNodes;

impl Operator for RemoveNodes {
    type Params = RemoveNodesParams;
    fn id() -> &'static str {
        "node.remove"
    }
    fn label() -> &'static str {
        "Delete Nodes"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(graph) = graph_in(doc, params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        let mut doomed: Vec<usize> = params
            .nodes
            .iter()
            .map(|n| *n as usize)
            .filter(|n| {
                graph
                    .nodes
                    .get(*n)
                    .is_some_and(|node| !matches!(node.kind, NodeKind::Surface { .. }))
            })
            .collect();
        if doomed.is_empty() {
            return Ok(OpResult::Cancelled);
        }
        // Highest first, so each removal leaves the indices still to come
        // pointing at the nodes they meant.
        doomed.sort_unstable();
        doomed.dedup();
        for node in doomed.into_iter().rev() {
            graph.remove_node(node);
        }
        Ok(OpResult::Finished)
    }
}

// --- link and unlink ---------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct LinkNodesParams {
    #[param(min = 0.0)]
    pub slot: u32,
    #[param(min = 0.0)]
    pub from_node: u32,
    #[param(min = 0.0)]
    pub from_socket: u32,
    #[param(min = 0.0)]
    pub to_node: u32,
    #[param(min = 0.0)]
    pub to_socket: u32,
}

/// Connects an output to an input, replacing whatever fed that input.
///
/// A link the graph refuses — mismatched types, a cycle, a socket that is not
/// there — is a cancel with nothing to undo, not an error: dropping a wire in
/// the wrong place is an ordinary gesture.
pub struct LinkNodes;

impl Operator for LinkNodes {
    type Params = LinkNodesParams;
    fn id() -> &'static str {
        "node.link"
    }
    fn label() -> &'static str {
        "Link Nodes"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(graph) = graph_in(doc, params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        let link = Link {
            from_node: params.from_node as usize,
            from_socket: params.from_socket as usize,
            to_node: params.to_node as usize,
            to_socket: params.to_socket as usize,
        };
        if graph.links.contains(&link) {
            return Ok(OpResult::Cancelled);
        }
        match graph.link(link) {
            Ok(()) => Ok(OpResult::Finished),
            Err(_) => Ok(OpResult::Cancelled),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct UnlinkParams {
    #[param(min = 0.0)]
    pub slot: u32,
    #[param(min = 0.0)]
    pub to_node: u32,
    #[param(min = 0.0)]
    pub to_socket: u32,
}

/// Disconnects whatever feeds one input. The input falls back to its own value.
pub struct Unlink;

impl Operator for Unlink {
    type Params = UnlinkParams;
    fn id() -> &'static str {
        "node.unlink"
    }
    fn label() -> &'static str {
        "Disconnect"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(graph) = graph_in(doc, params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        let before = graph.links.len();
        graph.links.retain(|l| {
            !(l.to_node == params.to_node as usize && l.to_socket == params.to_socket as usize)
        });
        Ok(if graph.links.len() == before {
            OpResult::Cancelled
        } else {
            OpResult::Finished
        })
    }
}

// --- move --------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TranslateNodesParams {
    #[param(min = 0.0)]
    pub slot: u32,
    pub nodes: Vec<u32>,
    /// How far, in graph units.
    pub x: f64,
    pub y: f64,
}

/// Moves nodes. One step for a whole drag: the editor moves them on screen
/// while the pointer is down and sends this when it lets go.
pub struct TranslateNodes;

impl Operator for TranslateNodes {
    type Params = TranslateNodesParams;
    fn id() -> &'static str {
        "node.translate"
    }
    fn label() -> &'static str {
        "Move Nodes"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if params.x == 0.0 && params.y == 0.0 {
            return Ok(OpResult::Cancelled);
        }
        let Some(graph) = graph_in(doc, params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        let mut moved = false;
        let mut seen = std::collections::HashSet::new();
        for node in &params.nodes {
            if !seen.insert(*node) {
                continue;
            }
            if let Some(node) = graph.nodes.get_mut(*node as usize) {
                node.position[0] += params.x as f32;
                node.position[1] += params.y as f32;
                moved = true;
            }
        }
        Ok(if moved {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

// --- values ------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetNodeColorParams {
    #[param(min = 0.0)]
    pub slot: u32,
    #[param(min = 0.0)]
    pub node: u32,
    /// Which of the node's colours. Zero for an RGB node's own colour and for
    /// a surface's base colour, which are the only colours a node holds.
    #[param(min = 0.0)]
    pub field: u32,
    /// As a swatch shows it, the way `material.set_properties` takes it.
    #[param(color)]
    pub color: [f64; 3],
}

impl Default for SetNodeColorParams {
    fn default() -> Self {
        Self {
            slot: 0,
            node: 0,
            field: 0,
            color: [0.8, 0.8, 0.8],
        }
    }
}

/// Sets a colour a node holds.
pub struct SetNodeColor;

impl Operator for SetNodeColor {
    type Params = SetNodeColorParams;
    fn id() -> &'static str {
        "node.set_color"
    }
    fn label() -> &'static str {
        "Set Node Color"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(graph) = graph_in(doc, params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(node) = graph.nodes.get_mut(params.node as usize) else {
            return Ok(OpResult::Cancelled);
        };
        // Through the same conversion the material panel uses: a reflectance
        // picked from a swatch.
        let colour = kreate_math::Srgb::new(
            params.color[0] as f32,
            params.color[1] as f32,
            params.color[2] as f32,
        )
        .as_reflectance();
        match (&mut node.kind, params.field) {
            (NodeKind::Color(c), 0) => *c = colour,
            (NodeKind::Surface { base_color, .. }, 0) => *base_color = colour,
            _ => return Ok(OpResult::Cancelled),
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetNodeValueParams {
    #[param(min = 0.0)]
    pub slot: u32,
    #[param(min = 0.0)]
    pub node: u32,
    /// Which of the node's numbers: zero for a Value node's value and a Normal
    /// Map's strength; one and two for a surface's metallic and roughness —
    /// the numbering `NodeKind::values` uses, less the surface's colour.
    #[param(min = 0.0)]
    pub field: u32,
    pub value: f64,
}

/// Sets a number a node holds.
pub struct SetNodeValue;

impl Operator for SetNodeValue {
    type Params = SetNodeValueParams;
    fn id() -> &'static str {
        "node.set_value"
    }
    fn label() -> &'static str {
        "Set Node Value"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(graph) = graph_in(doc, params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(node) = graph.nodes.get_mut(params.node as usize) else {
            return Ok(OpResult::Cancelled);
        };
        let v = params.value as f32;
        match (&mut node.kind, params.field) {
            (NodeKind::Value(x), 0) => *x = v,
            (NodeKind::NormalMap { strength }, 0) => *strength = v.max(0.0),
            (NodeKind::Surface { metallic, .. }, 1) => *metallic = v.clamp(0.0, 1.0),
            (NodeKind::Surface { roughness, .. }, 2) => *roughness = v.clamp(0.0, 1.0),
            _ => return Ok(OpResult::Cancelled),
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetNodeImageParams {
    #[param(min = 0.0)]
    pub slot: u32,
    #[param(min = 0.0)]
    pub node: u32,
    /// The image's identity, in decimal; empty for none. See
    /// `SetMaterialImageParams::uid` on why an identity.
    pub uid: String,
}

/// Points an Image Texture node at an image, or at none.
pub struct SetNodeImage;

impl Operator for SetNodeImage {
    type Params = SetNodeImageParams;
    fn id() -> &'static str {
        "node.set_image"
    }
    fn label() -> &'static str {
        "Set Node Image"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let image = match params.uid.trim() {
            "" => None,
            text => {
                let Ok(raw) = text.parse::<u64>() else {
                    return Ok(OpResult::Cancelled);
                };
                match doc.document().image_by_uid(kreate_scene::Uid::from_raw(raw)) {
                    Some(image) => Some(image),
                    None => return Ok(OpResult::Cancelled),
                }
            }
        };
        let Some(graph) = graph_in(doc, params.slot) else {
            return Ok(OpResult::Cancelled);
        };
        match graph.nodes.get_mut(params.node as usize).map(|n| &mut n.kind) {
            Some(NodeKind::Image { image: slot }) => {
                if *slot == image {
                    return Ok(OpResult::Cancelled);
                }
                *slot = image;
                Ok(OpResult::Finished)
            }
            _ => Ok(OpResult::Cancelled),
        }
    }
}

/// The material being edited in `slot`, for callers that want its id.
pub fn material_of(doc: &Document, slot: u32) -> Option<MaterialId> {
    material_in_slot(doc, slot)
}
