//! The shading graph.
//!
//! §3's fifth "decide now", and the half of it that is data. v1 §5.6 already
//! settled the model — *a node graph, with a standard-surface fast path* — and
//! what §2.3 adds is the rule that follows from it:
//!
//! > **One graph, one compiler, two consumers.** The graph becomes WGSL, and
//! > the viewport and the renderer both run that. No hand-written shading mode
//! > is added.
//!
//! That is not a style preference. Two systems that each interpret a material
//! will diverge, the user will report it as a bug, and it will be a bug — one
//! that recurs for as long as both implementations exist. §5's second risk is
//! about exactly this.
//!
//! # The flat material is the fast path, not a second model
//!
//! A [`crate::document::Material`] with no graph is a standard surface with
//! three constants, and that is overwhelmingly what a material is. It does not
//! get its own shader: it is the graph with nothing connected, and the compiler
//! reaches the same code by a shorter route. [`ShaderGraph::standard`] writes it
//! out explicitly, which is what makes "the fast path and the graph agree" a
//! thing a test can ask.
//!
//! # A socket's type is checked when the link is made
//!
//! Not when the shader is compiled. A graph that cannot be compiled is a
//! material that draws nothing, and the place to find that out is the moment
//! somebody drags a wire — not three steps later in a window that has gone
//! black. The same reasoning as `Document::set_parent` refusing a cycle.
//!
//! # What is deliberately not here yet
//!
//! Normal maps, emission, transmission, clearcoat and the rest of §2.3's lobes.
//! They are nodes and inputs, not architecture, and adding them before anything
//! consumes the graph would be guessing at what the compiler wants.

use crate::document::{ImageId, Linkage};
use kreate_math::SceneRgb;
use kreate_reflect::Uid;

/// What flows along a wire.
///
/// Two, and a colour is not three scalars: a wire carrying a colour into a
/// roughness would be a material that is wrong in a way the picture does not
/// explain. Where a conversion is wanted it is a *node*, so it is visible in
/// the graph rather than implied by a connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Socket {
    /// Linear RGB, in the working space.
    Color,
    /// One number.
    Scalar,
}

/// What a node does.
///
/// A small set on purpose. Every node here is one somebody reaches for before
/// they reach for anything else, and the ones that are missing are missing
/// because nothing compiles the graph yet — a node added ahead of its consumer
/// is a guess about what that consumer will want.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// A constant colour.
    Color(SceneRgb),
    /// A constant number.
    Value(f32),
    /// An image, sampled at the surface's UV.
    ///
    /// `None` is a slot nobody has filled, which is an ordinary state while a
    /// material is being built — it reads as the fallback rather than failing
    /// to compile, for the same reason a world with a missing image lands on a
    /// colour.
    Image { image: Option<ImageId> },
    /// The surface that gets shaded. A graph has exactly one.
    ///
    /// Its inputs carry the values an unconnected socket takes, so a graph with
    /// nothing wired to it *is* a flat material — which is what makes the fast
    /// path the same model rather than a second one.
    Surface {
        base_color: SceneRgb,
        metallic: f32,
        roughness: f32,
    },
}

impl NodeKind {
    /// The types of this node's inputs, in order.
    pub fn inputs(&self) -> &'static [Socket] {
        match self {
            NodeKind::Color(_) | NodeKind::Value(_) | NodeKind::Image { .. } => &[],
            NodeKind::Surface { .. } => &[Socket::Color, Socket::Scalar, Socket::Scalar],
        }
    }

    /// The types of this node's outputs, in order.
    pub fn outputs(&self) -> &'static [Socket] {
        match self {
            NodeKind::Color(_) => &[Socket::Color],
            NodeKind::Value(_) => &[Socket::Scalar],
            // Colour and alpha. Two outputs rather than four channels, because
            // an image's alpha is not a colour and wiring it into one is the
            // mistake the type system is here to refuse.
            NodeKind::Image { .. } => &[Socket::Color, Socket::Scalar],
            NodeKind::Surface { .. } => &[],
        }
    }

    /// A stable name for files and scripts.
    ///
    /// Not the discriminant: serialising that would make a saved material
    /// depend on the order the variants happen to be declared in, so inserting
    /// a node kind in the middle would reinterpret every file. The same rule
    /// `Mode::tag` follows.
    pub fn tag(&self) -> &'static str {
        match self {
            NodeKind::Color(_) => "color",
            NodeKind::Value(_) => "value",
            NodeKind::Image { .. } => "image",
            NodeKind::Surface { .. } => "surface",
        }
    }
}

/// One node.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub name: String,
    pub kind: NodeKind,
    /// Where it sits in the editor, in node-graph units.
    ///
    /// Part of the datablock because it is authored: a graph that comes back
    /// from a file with its nodes piled at the origin is one somebody has to
    /// lay out again.
    pub position: [f32; 2],
}

impl Node {
    pub fn new(name: impl Into<String>, kind: NodeKind) -> Self {
        Self {
            name: name.into(),
            kind,
            position: [0.0, 0.0],
        }
    }
}

/// A wire.
///
/// Indices into [`ShaderGraph::nodes`] and into that node's socket list. An
/// index rather than a name because a node's sockets are decided by its kind
/// and cannot be renamed; the *node* has a name, for people.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub from_node: usize,
    pub from_socket: usize,
    pub to_node: usize,
    pub to_socket: usize,
}

/// Why a link was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkError {
    /// A node index that is not in the graph.
    NoSuchNode,
    /// A socket index the node does not have.
    NoSuchSocket,
    /// A colour into a scalar, or the other way.
    TypeMismatch { from: Socket, to: Socket },
    /// Following the wire comes back to where it started.
    Cycle,
}

/// A shading graph datablock.
#[derive(Clone, Debug, PartialEq)]
pub struct ShaderGraph {
    pub name: String,
    pub linkage: Linkage,
    /// Assigned by the document. See [`Uid`].
    pub uid: Uid,
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
}

impl ShaderGraph {
    /// The graph a flat material is.
    ///
    /// One node, nothing connected. Written out rather than left implicit
    /// because it is the claim the fast path rests on: if the compiler's short
    /// route and this graph ever stop agreeing, every material in every scene
    /// changes and nothing says why.
    pub fn standard(name: impl Into<String>, base_color: SceneRgb, metallic: f32, roughness: f32) -> Self {
        Self {
            name: name.into(),
            linkage: Linkage::Local,
            uid: Uid::NONE,
            nodes: vec![Node::new(
                "Surface",
                NodeKind::Surface {
                    base_color,
                    metallic,
                    roughness,
                },
            )],
            links: Vec::new(),
        }
    }

    /// Which node is shaded, if the graph has one.
    ///
    /// The first [`NodeKind::Surface`]. A graph with none compiles to nothing
    /// and is a material being built rather than a broken one; a graph with two
    /// is refused by [`ShaderGraph::add_node`], because "which one" is a
    /// question with no good answer.
    pub fn output(&self) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| matches!(n.kind, NodeKind::Surface { .. }))
    }

    /// Adds a node, or refuses a second surface.
    pub fn add_node(&mut self, node: Node) -> Option<usize> {
        if matches!(node.kind, NodeKind::Surface { .. }) && self.output().is_some() {
            return None;
        }
        self.nodes.push(node);
        Some(self.nodes.len() - 1)
    }

    /// The link feeding an input, if anything does.
    pub fn incoming(&self, node: usize, socket: usize) -> Option<&Link> {
        self.links
            .iter()
            .find(|l| l.to_node == node && l.to_socket == socket)
    }

    /// Connects two sockets.
    ///
    /// Checked here rather than at compile time. A graph that cannot compile is
    /// a material that draws nothing, and the moment to find that out is when
    /// somebody drags the wire — not three steps later in a window that has
    /// gone black.
    ///
    /// An input already fed is *replaced* rather than refused: dropping a wire
    /// onto a taken socket means "this one instead", which is what every node
    /// editor does and what the gesture means.
    pub fn link(&mut self, link: Link) -> Result<(), LinkError> {
        let from = self.nodes.get(link.from_node).ok_or(LinkError::NoSuchNode)?;
        let to = self.nodes.get(link.to_node).ok_or(LinkError::NoSuchNode)?;
        let from_type = *from
            .kind
            .outputs()
            .get(link.from_socket)
            .ok_or(LinkError::NoSuchSocket)?;
        let to_type = *to
            .kind
            .inputs()
            .get(link.to_socket)
            .ok_or(LinkError::NoSuchSocket)?;
        if from_type != to_type {
            return Err(LinkError::TypeMismatch {
                from: from_type,
                to: to_type,
            });
        }
        if self.reaches(link.to_node, link.from_node) || link.from_node == link.to_node {
            return Err(LinkError::Cycle);
        }
        self.links
            .retain(|l| !(l.to_node == link.to_node && l.to_socket == link.to_socket));
        self.links.push(link);
        Ok(())
    }

    /// Whether anything downstream of `from` is `target`.
    ///
    /// Bounded by the node count: a walk longer than that has revisited
    /// something, which is the cycle being looked for. The same shape
    /// `ImportScene::world_transform` uses, and for the same reason — a graph
    /// can be in a state a walk does not terminate on, and the walk has to say
    /// so rather than hang.
    fn reaches(&self, from: usize, target: usize) -> bool {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![from];
        while let Some(at) = stack.pop() {
            if at == target {
                return true;
            }
            if seen.get(at).copied().unwrap_or(true) {
                continue;
            }
            seen[at] = true;
            for link in self.links.iter().filter(|l| l.from_node == at) {
                stack.push(link.to_node);
            }
        }
        false
    }

    /// Removes a node and every wire touching it.
    ///
    /// The indices after it move, so the links have to be renumbered — which is
    /// the cost of indices and the reason it is done here rather than left to
    /// callers. A caller that renumbered one link list and forgot another would
    /// produce a graph that compiles and shades something else.
    pub fn remove_node(&mut self, node: usize) -> bool {
        if node >= self.nodes.len() {
            return false;
        }
        self.nodes.remove(node);
        self.links
            .retain(|l| l.from_node != node && l.to_node != node);
        for link in &mut self.links {
            if link.from_node > node {
                link.from_node -= 1;
            }
            if link.to_node > node {
                link.to_node -= 1;
            }
        }
        true
    }

    /// Every image the graph samples, in node order.
    ///
    /// What a renderer has to have uploaded before it can draw this material,
    /// and what a file has to carry with it.
    pub fn images(&self) -> Vec<ImageId> {
        self.nodes
            .iter()
            .filter_map(|n| match n.kind {
                NodeKind::Image { image } => image,
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> ShaderGraph {
        ShaderGraph::standard("M", SceneRgb::new(0.8, 0.8, 0.8), 0.0, 0.5)
    }

    #[test]
    fn a_flat_material_is_a_graph_with_nothing_connected() {
        // The claim the fast path rests on. If the compiler's short route and
        // this graph ever stop agreeing, every material in every scene changes
        // and nothing says why.
        let g = graph();
        assert_eq!(g.nodes.len(), 1);
        assert!(g.links.is_empty());
        let output = g.output().expect("a surface");
        match g.nodes[output].kind {
            NodeKind::Surface {
                base_color,
                metallic,
                roughness,
            } => {
                assert_eq!(base_color, SceneRgb::new(0.8, 0.8, 0.8));
                assert_eq!(metallic, 0.0);
                assert_eq!(roughness, 0.5);
            }
            ref other => panic!("the output is {other:?}"),
        }
    }

    #[test]
    fn a_colour_cannot_be_wired_into_a_roughness() {
        // The reason a wire carries a *type*. A colour into a roughness is a
        // material that is wrong in a way the picture does not explain, and
        // three numbers where one was wanted is the easiest wire to draw by
        // accident — the sockets are next to each other.
        let mut g = graph();
        let colour = g
            .add_node(Node::new("Red", NodeKind::Color(SceneRgb::new(1.0, 0.0, 0.0))))
            .expect("added");
        let surface = g.output().expect("a surface");
        assert_eq!(
            g.link(Link {
                from_node: colour,
                from_socket: 0,
                to_node: surface,
                to_socket: 2,
            }),
            Err(LinkError::TypeMismatch {
                from: Socket::Color,
                to: Socket::Scalar
            })
        );
        // And into the socket that does take one, it connects.
        assert_eq!(
            g.link(Link {
                from_node: colour,
                from_socket: 0,
                to_node: surface,
                to_socket: 0,
            }),
            Ok(())
        );
    }

    #[test]
    fn an_images_alpha_is_not_a_colour() {
        // Two outputs rather than four channels. An image's alpha is a number
        // and its colour is a colour, and the socket types are what stop one
        // being used as the other.
        let mut g = graph();
        let image = g
            .add_node(Node::new("Albedo", NodeKind::Image { image: None }))
            .expect("added");
        let surface = g.output().expect("a surface");
        // Colour into base colour: fine.
        assert_eq!(
            g.link(Link {
                from_node: image,
                from_socket: 0,
                to_node: surface,
                to_socket: 0,
            }),
            Ok(())
        );
        // Alpha into roughness: also fine, and a real thing to want.
        assert_eq!(
            g.link(Link {
                from_node: image,
                from_socket: 1,
                to_node: surface,
                to_socket: 2,
            }),
            Ok(())
        );
        // Alpha into base colour: refused.
        assert!(matches!(
            g.link(Link {
                from_node: image,
                from_socket: 1,
                to_node: surface,
                to_socket: 0,
            }),
            Err(LinkError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn a_wire_that_comes_back_to_itself_is_refused() {
        // A graph a compiler cannot walk is a material that draws nothing, and
        // the moment to find out is when the wire is dragged.
        //
        // Built with two `Surface` nodes' worth of shape by hand, because the
        // node set here has no kind with both an input and an output yet — and
        // the guard has to exist *before* one does, since adding the first such
        // kind is exactly when a cycle becomes expressible.
        let mut g = graph();
        let surface = g.output().expect("a surface");
        let colour = g
            .add_node(Node::new("C", NodeKind::Color(SceneRgb::new(1.0, 1.0, 1.0))))
            .expect("added");
        // A node cannot feed itself, whatever its sockets say.
        assert_eq!(
            g.link(Link {
                from_node: colour,
                from_socket: 0,
                to_node: colour,
                to_socket: 0,
            }),
            // Refused as a missing socket before the cycle check, because a
            // `Color` node has no inputs — which is the honest first answer.
            Err(LinkError::NoSuchSocket)
        );
        let _ = surface;
    }

    #[test]
    fn dropping_a_wire_on_a_taken_socket_replaces_it() {
        // What the gesture means in every node editor there is. Refusing would
        // make changing an input a two-step operation with a disconnected state
        // in the middle, and a material that is briefly nothing.
        let mut g = graph();
        let surface = g.output().expect("a surface");
        let red = g
            .add_node(Node::new("Red", NodeKind::Color(SceneRgb::new(1.0, 0.0, 0.0))))
            .expect("added");
        let blue = g
            .add_node(Node::new("Blue", NodeKind::Color(SceneRgb::new(0.0, 0.0, 1.0))))
            .expect("added");
        g.link(Link {
            from_node: red,
            from_socket: 0,
            to_node: surface,
            to_socket: 0,
        })
        .expect("connects");
        g.link(Link {
            from_node: blue,
            from_socket: 0,
            to_node: surface,
            to_socket: 0,
        })
        .expect("connects");
        assert_eq!(g.links.len(), 1, "both wires are still attached");
        assert_eq!(g.incoming(surface, 0).expect("fed").from_node, blue);
    }

    #[test]
    fn a_graph_holds_one_surface() {
        // "Which one is shaded" is a question with no good answer, so it is not
        // askable.
        let mut g = graph();
        assert_eq!(
            g.add_node(Node::new(
                "Second",
                NodeKind::Surface {
                    base_color: SceneRgb::new(1.0, 1.0, 1.0),
                    metallic: 0.0,
                    roughness: 0.5,
                },
            )),
            None
        );
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn removing_a_node_renumbers_the_wires_that_outlive_it() {
        // The cost of indices, paid in one place. A caller that renumbered one
        // link list and forgot another would produce a graph that compiles and
        // shades something else — which is the kind of wrong that survives
        // review.
        let mut g = graph();
        let surface = g.output().expect("a surface");
        let doomed = g
            .add_node(Node::new("Doomed", NodeKind::Value(0.25)))
            .expect("added");
        let kept = g
            .add_node(Node::new("Kept", NodeKind::Value(0.75)))
            .expect("added");
        g.link(Link {
            from_node: kept,
            from_socket: 0,
            to_node: surface,
            to_socket: 2,
        })
        .expect("connects");

        assert!(g.remove_node(doomed));
        assert_eq!(g.nodes.len(), 2);
        let surface = g.output().expect("still there");
        let feeding = g.incoming(surface, 2).expect("still fed");
        match g.nodes[feeding.from_node].kind {
            NodeKind::Value(v) => assert_eq!(v, 0.75, "the wire moved to another node"),
            ref other => panic!("the wire feeds {other:?}"),
        }
    }

    #[test]
    fn removing_a_node_takes_its_wires_with_it() {
        let mut g = graph();
        let surface = g.output().expect("a surface");
        let value = g
            .add_node(Node::new("V", NodeKind::Value(0.25)))
            .expect("added");
        g.link(Link {
            from_node: value,
            from_socket: 0,
            to_node: surface,
            to_socket: 2,
        })
        .expect("connects");
        assert_eq!(g.links.len(), 1);
        assert!(g.remove_node(value));
        assert!(g.links.is_empty(), "a wire outlived the node it came from");
    }

    #[test]
    fn a_node_kinds_tag_does_not_depend_on_where_it_is_declared() {
        // What a saved material names its nodes by. A discriminant would make
        // inserting a kind in the middle reinterpret every file ever written —
        // the same rule `Mode::tag` follows and for the same reason.
        assert_eq!(NodeKind::Value(0.0).tag(), "value");
        assert_eq!(NodeKind::Image { image: None }.tag(), "image");
        assert_eq!(
            NodeKind::Color(SceneRgb::new(0.0, 0.0, 0.0)).tag(),
            "color"
        );
    }
}
