//! Editing the active object's geometry node graph.
//!
//! What the Geometry Nodes editor's gestures become, on the operator model:
//! each is undoable, replays from a script, and never leaves the graph
//! without its group input and output.

use kreate_ops_derive::OpParams;
use kreate_scene::geonodes::{GeoGraph, GeoNode, GeoNodeKind};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<Enable>();
    registry.register::<Add>();
    registry.register::<Remove>();
    registry.register::<Link>();
    registry.register::<Unlink>();
    registry.register::<Translate>();
    registry.register::<SetValue>();
}

/// The active mesh object's graph.
fn graph(doc: &Document) -> Option<&GeoGraph> {
    let o = doc.object(doc.selection.active?)?;
    o.mesh()?;
    o.geometry_nodes.as_ref()
}

fn graph_mut<'d>(doc: &'d mut DocMut<'_>) -> Option<&'d mut GeoGraph> {
    let id = doc.document().selection.active?;
    doc.document_mut().object_mut(id)?.geometry_nodes.as_mut()
}

fn has_graph(doc: &Document) -> bool {
    graph(doc).is_some()
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct EnableParams {
    /// On gives the object a graph that passes its mesh straight through;
    /// off takes the graph away.
    pub enabled: bool,
}

impl Default for EnableParams {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Gives the active mesh object geometry nodes, or takes them away.
pub struct Enable;

impl Operator for Enable {
    type Params = EnableParams;
    fn id() -> &'static str {
        "geo.enable"
    }
    fn label() -> &'static str {
        "Geometry Nodes"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .active
            .and_then(|id| doc.object(id))
            .is_some_and(|o| o.mesh().is_some())
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(id) = doc.document().selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let Some(o) = doc.document_mut().object_mut(id) else {
            return Ok(OpResult::Cancelled);
        };
        match (params.enabled, o.geometry_nodes.is_some()) {
            (true, false) => o.geometry_nodes = Some(GeoGraph::passthrough()),
            (false, true) => o.geometry_nodes = None,
            _ => return Ok(OpResult::Cancelled),
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct AddParams {
    /// Which kind, in the order of `GeoNodeKind::ADDABLE`.
    #[param(variants(
        "transform",
        "array",
        "subdivide",
        "mirror",
        "join",
        "primitive",
        "extrude_profile",
        "revolve_profile"
    ))]
    pub kind: u32,
    pub x: f64,
    pub y: f64,
}

/// Adds a node where the pointer was.
pub struct Add;

impl Operator for Add {
    type Params = AddParams;
    fn id() -> &'static str {
        "geo.add"
    }
    fn label() -> &'static str {
        "Add Geometry Node"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        has_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(kind) = GeoNodeKind::ADDABLE
            .get(params.kind as usize)
            .and_then(|tag| GeoNodeKind::from_parts(tag, &[]))
        else {
            return Ok(OpResult::Cancelled);
        };
        let Some(g) = graph_mut(doc) else {
            return Ok(OpResult::Cancelled);
        };
        g.nodes.push(GeoNode {
            name: default_name(&kind).into(),
            kind,
            position: [params.x as f32, params.y as f32],
        });
        Ok(OpResult::Finished)
    }
}

/// What a new node is called, as Blender titles them.
pub fn default_name(kind: &GeoNodeKind) -> &'static str {
    match kind {
        GeoNodeKind::Input => "Group Input",
        GeoNodeKind::Output => "Group Output",
        GeoNodeKind::Transform { .. } => "Transform Geometry",
        GeoNodeKind::Array { .. } => "Array",
        GeoNodeKind::Subdivide { .. } => "Subdivide",
        GeoNodeKind::Mirror { .. } => "Mirror",
        GeoNodeKind::Join => "Join Geometry",
        GeoNodeKind::Primitive { .. } => "Mesh Primitive",
        GeoNodeKind::ExtrudeProfile { .. } => "Extrude Profile",
        GeoNodeKind::RevolveProfile { .. } => "Revolve Profile",
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct NodesParams {
    pub nodes: Vec<u32>,
}

/// Removes nodes and their wires. Never the group's input or output.
pub struct Remove;

impl Operator for Remove {
    type Params = NodesParams;
    fn id() -> &'static str {
        "geo.remove"
    }
    fn label() -> &'static str {
        "Delete Geometry Nodes"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        has_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let doomed: Vec<usize> = params.nodes.iter().map(|n| *n as usize).collect();
        let Some(g) = graph_mut(doc) else {
            return Ok(OpResult::Cancelled);
        };
        Ok(if g.remove(&doomed) {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct LinkParams {
    pub from_node: u32,
    pub to_node: u32,
    pub to_socket: u32,
}

/// Wires one node's geometry into another's input, replacing what was there.
pub struct Link;

impl Operator for Link {
    type Params = LinkParams;
    fn id() -> &'static str {
        "geo.link"
    }
    fn label() -> &'static str {
        "Link Geometry Nodes"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        has_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(g) = graph_mut(doc) else {
            return Ok(OpResult::Cancelled);
        };
        match g.link(
            params.from_node as usize,
            params.to_node as usize,
            params.to_socket as usize,
        ) {
            Ok(()) => Ok(OpResult::Finished),
            Err(e) => Err(OpError::Failed(format!("{e:?}"))),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct UnlinkParams {
    pub to_node: u32,
    pub to_socket: u32,
}

/// Takes the wire off an input.
pub struct Unlink;

impl Operator for Unlink {
    type Params = UnlinkParams;
    fn id() -> &'static str {
        "geo.unlink"
    }
    fn label() -> &'static str {
        "Unlink Geometry Node"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        has_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(g) = graph_mut(doc) else {
            return Ok(OpResult::Cancelled);
        };
        let before = g.links.len();
        g.links.retain(|l| {
            !(l.to_node == params.to_node as usize && l.to_socket == params.to_socket as usize)
        });
        Ok(if g.links.len() < before {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct TranslateParams {
    pub nodes: Vec<u32>,
    pub x: f64,
    pub y: f64,
}

/// Moves nodes in the editor.
pub struct Translate;

impl Operator for Translate {
    type Params = TranslateParams;
    fn id() -> &'static str {
        "geo.translate"
    }
    fn label() -> &'static str {
        "Move Geometry Nodes"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        has_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(g) = graph_mut(doc) else {
            return Ok(OpResult::Cancelled);
        };
        let mut moved = false;
        for n in &params.nodes {
            if let Some(node) = g.nodes.get_mut(*n as usize) {
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

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetValueParams {
    pub node: u32,
    /// Which of the node's numbers, as `GeoNodeKind::fields` numbers them.
    pub field: u32,
    pub value: f64,
}

/// Sets one of a node's numbers, kept in its range.
pub struct SetValue;

impl Operator for SetValue {
    type Params = SetValueParams;
    fn id() -> &'static str {
        "geo.set_value"
    }
    fn label() -> &'static str {
        "Set Geometry Node Value"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        has_graph(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(g) = graph_mut(doc) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(node) = g.nodes.get_mut(params.node as usize) else {
            return Ok(OpResult::Cancelled);
        };
        let Some(kind) = node.kind.with_value(params.field, params.value as f32) else {
            return Ok(OpResult::Cancelled);
        };
        if kind == node.kind {
            return Ok(OpResult::Cancelled);
        }
        node.kind = kind;
        Ok(OpResult::Finished)
    }
}
