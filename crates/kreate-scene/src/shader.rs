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
//! Transmission, clearcoat and the rest of §2.3's lobes. Emission is here as
//! a value on the surface, not yet as a socket.
//! They are nodes and inputs, not architecture, and adding them before anything
//! consumes the graph would be guessing at what the compiler wants.

use crate::document::{ImageId, Linkage};
use kreate_math::SceneRgb;
use kreate_reflect::Uid;

/// What flows along a wire.
///
/// A colour is not three scalars: a wire carrying a colour into a
/// roughness would be a material that is wrong in a way the picture does not
/// explain. Where a conversion is wanted it is a *node*, so it is visible in
/// the graph rather than implied by a connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Socket {
    /// Linear RGB, in the working space.
    Color,
    /// One number.
    Scalar,
    /// A direction in the surface's tangent space: +Z out of the surface, +X
    /// along the texture's U, +Y along its V.
    ///
    /// Its own type because a normal is not a colour even when it arrives in
    /// one. A normal map *is* a colour image on disk, and the node that turns
    /// it into a direction is where the `* 2 - 1` lives — wired straight into a
    /// colour it would be a surface lit by a purple picture of itself.
    Vector,
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
    /// Two colours, multiplied.
    ///
    /// Here because a texture has to *tint* rather than replace: glTF's base
    /// colour is `baseColorTexture * baseColorFactor`, and a material whose
    /// colour stopped meaning anything the moment a texture was assigned would
    /// be a surprise once and a round-trip loss every time after. Giving the
    /// image node a factor of its own would have been the same rule written
    /// where only one node can reach it.
    Multiply,
    /// A colour's three channels as three numbers.
    ///
    /// What reading a packed map needs: glTF puts roughness in green and
    /// metallic in blue of one image, and a socket type that refused the
    /// colour-to-number wire would be useless without a node that says which
    /// channel.
    SeparateRgb,
    /// Two numbers, multiplied. The scalar [`NodeKind::Multiply`]: a roughness
    /// map scales the roughness dial rather than replacing it, for the reason
    /// a base colour map tints.
    MultiplyValue,
    /// A tangent-space normal map's colour, decoded into a direction.
    ///
    /// `strength` scales the tilt before renormalising, which is glTF's
    /// `normalTexture.scale`: zero is flat, one is the map as painted.
    NormalMap { strength: f32 },
    /// Two colours mixed by a factor: Blender's Mix (Color, Mix). `factor` is
    /// what an unwired Factor socket takes.
    MixColor { factor: f32 },
    /// Two numbers through an operation: Blender's Math. `a` and `b` are
    /// what unwired sockets take.
    Math { op: MathOp, a: f32, b: f32 },
    /// One minus a colour: Blender's Invert Color.
    Invert,
    /// The surface that gets shaded. A graph has exactly one.
    ///
    /// Its inputs carry the values an unconnected socket takes, so a graph with
    /// nothing wired to it *is* a flat material — which is what makes the fast
    /// path the same model rather than a second one.
    Surface {
        base_color: SceneRgb,
        metallic: f32,
        roughness: f32,
        /// What the surface emits by itself, in nits: the material's colour
        /// times its strength. A value rather than a socket for now.
        emission: SceneRgb,
        /// How much passes through, zero to one. See `Material::transmission`.
        transmission: f32,
        /// The index of refraction of what passes through.
        ior: f32,
    },
}

/// What a Math node does to its two numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MathOp {
    #[default]
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Minimum,
    Maximum,
}

impl MathOp {
    pub const ALL: [Self; 7] = [
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Divide,
        Self::Power,
        Self::Minimum,
        Self::Maximum,
    ];

    pub fn tag(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Divide => "divide",
            Self::Power => "power",
            Self::Minimum => "minimum",
            Self::Maximum => "maximum",
        }
    }

    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|o| *o == self).unwrap_or(0) as u32
    }

    pub fn from_index(i: u32) -> Self {
        Self::ALL.get(i as usize).copied().unwrap_or_default()
    }
}

impl NodeKind {
    /// The types of this node's inputs, in order.
    pub fn inputs(&self) -> &'static [Socket] {
        match self {
            NodeKind::Color(_) | NodeKind::Value(_) | NodeKind::Image { .. } => &[],
            NodeKind::Multiply => &[Socket::Color, Socket::Color],
            NodeKind::SeparateRgb | NodeKind::NormalMap { .. } => &[Socket::Color],
            NodeKind::MultiplyValue | NodeKind::Math { .. } => &[Socket::Scalar, Socket::Scalar],
            NodeKind::MixColor { .. } => &[Socket::Scalar, Socket::Color, Socket::Color],
            NodeKind::Invert => &[Socket::Color],
            // Base colour, metallic, roughness, normal, and a tint on the glow.
            NodeKind::Surface { .. } => &[
                Socket::Color,
                Socket::Scalar,
                Socket::Scalar,
                Socket::Vector,
                Socket::Color,
            ],
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
            NodeKind::Multiply => &[Socket::Color],
            NodeKind::SeparateRgb => &[Socket::Scalar, Socket::Scalar, Socket::Scalar],
            NodeKind::MultiplyValue | NodeKind::Math { .. } => &[Socket::Scalar],
            NodeKind::MixColor { .. } | NodeKind::Invert => &[Socket::Color],
            NodeKind::NormalMap { .. } => &[Socket::Vector],
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
            NodeKind::Multiply => "multiply",
            NodeKind::SeparateRgb => "separate_rgb",
            NodeKind::MultiplyValue => "multiply_value",
            NodeKind::NormalMap { .. } => "normal_map",
            NodeKind::MixColor { .. } => "mix",
            NodeKind::Math { .. } => "math",
            NodeKind::Invert => "invert",
            NodeKind::Surface { .. } => "surface",
        }
    }

    /// The numbers the node holds, flat, for a file.
    ///
    /// The inverse of [`NodeKind::from_parts`]. An image is not a number and
    /// travels beside these, as an identity.
    pub fn values(&self) -> Vec<f32> {
        match self {
            NodeKind::Color(c) => vec![c.r, c.g, c.b],
            NodeKind::Value(v) => vec![*v],
            NodeKind::NormalMap { strength } => vec![*strength],
            NodeKind::MixColor { factor } => vec![*factor],
            NodeKind::Math { op, a, b } => vec![op.index() as f32, *a, *b],
            NodeKind::Surface {
                base_color,
                metallic,
                roughness,
                emission,
                transmission,
                ior,
            } => {
                vec![
                    base_color.r,
                    base_color.g,
                    base_color.b,
                    *metallic,
                    *roughness,
                    emission.r,
                    emission.g,
                    emission.b,
                    *transmission,
                    *ior,
                ]
            }
            NodeKind::Image { .. }
            | NodeKind::Multiply
            | NodeKind::SeparateRgb
            | NodeKind::MultiplyValue
            | NodeKind::Invert => Vec::new(),
        }
    }

    /// A node kind from its tag, its numbers and its image.
    ///
    /// `None` for a tag this build does not know — a file from a newer one —
    /// which the caller must treat as "this graph cannot be read", not as a
    /// node to skip: a graph with a node silently missing is a different
    /// material. Missing numbers take the kind's defaults, so a file that
    /// wrote fewer than this build expects still reads.
    pub fn from_parts(tag: &str, values: &[f32], image: Option<ImageId>) -> Option<Self> {
        let at = |i: usize, default: f32| values.get(i).copied().unwrap_or(default);
        Some(match tag {
            "color" => NodeKind::Color(SceneRgb::new(at(0, 0.8), at(1, 0.8), at(2, 0.8))),
            "value" => NodeKind::Value(at(0, 0.5)),
            "image" => NodeKind::Image { image },
            "multiply" => NodeKind::Multiply,
            "separate_rgb" => NodeKind::SeparateRgb,
            "multiply_value" => NodeKind::MultiplyValue,
            "normal_map" => NodeKind::NormalMap {
                strength: at(0, 1.0),
            },
            "mix" => NodeKind::MixColor { factor: at(0, 0.5) },
            "math" => NodeKind::Math {
                op: MathOp::from_index(at(0, 0.0).max(0.0) as u32),
                a: at(1, 0.5),
                b: at(2, 0.5),
            },
            "invert" => NodeKind::Invert,
            "surface" => NodeKind::Surface {
                base_color: SceneRgb::new(at(0, 0.8), at(1, 0.8), at(2, 0.8)),
                metallic: at(3, 0.0),
                roughness: at(4, 0.5),
                // Absent in every file written before emission: none.
                emission: SceneRgb::new(at(5, 0.0), at(6, 0.0), at(7, 0.0)),
                transmission: at(8, 0.0),
                ior: at(9, 1.5),
            },
            _ => return None,
        })
    }

    /// A fresh node of the kind a tag names, as the Add menu makes it.
    pub fn new_of(tag: &str) -> Option<Self> {
        Self::from_parts(tag, &[], None)
    }

    /// The kinds somebody can add, in the order the Add menu lists them.
    ///
    /// Not the surface: a graph has exactly one, and it is never added.
    pub const ADDABLE: [&'static str; 10] = [
        "color",
        "value",
        "image",
        "multiply",
        "multiply_value",
        "separate_rgb",
        "normal_map",
        "mix",
        "math",
        "invert",
    ];

    /// What each input is called, for a person.
    pub fn input_names(&self) -> &'static [&'static str] {
        match self {
            NodeKind::Multiply | NodeKind::MultiplyValue | NodeKind::Math { .. } => &["A", "B"],
            NodeKind::SeparateRgb | NodeKind::NormalMap { .. } | NodeKind::Invert => &["Color"],
            NodeKind::MixColor { .. } => &["Factor", "A", "B"],
            NodeKind::Surface { .. } => {
                &["Base Color", "Metallic", "Roughness", "Normal", "Emission"]
            }
            NodeKind::Color(_) | NodeKind::Value(_) | NodeKind::Image { .. } => &[],
        }
    }

    /// What each output is called.
    pub fn output_names(&self) -> &'static [&'static str] {
        match self {
            NodeKind::Color(_)
            | NodeKind::Multiply
            | NodeKind::MixColor { .. }
            | NodeKind::Invert => &["Color"],
            NodeKind::Value(_) | NodeKind::MultiplyValue | NodeKind::Math { .. } => &["Value"],
            NodeKind::Image { .. } => &["Color", "Alpha"],
            NodeKind::SeparateRgb => &["R", "G", "B"],
            NodeKind::NormalMap { .. } => &["Normal"],
            NodeKind::Surface { .. } => &[],
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
    pub fn standard(
        name: impl Into<String>,
        base_color: SceneRgb,
        metallic: f32,
        roughness: f32,
    ) -> Self {
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
                    emission: SceneRgb::new(0.0, 0.0, 0.0),
                    transmission: 0.0,
                    ior: 1.5,
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

    /// A material's flat fields as the graph they are.
    ///
    /// **One definition of what a standard surface means**, and it lives beside
    /// the fields rather than in whatever is compiling today — a second caller
    /// that built the same graph by hand would be a second answer to "what is a
    /// material", and the two would part company at the first field added.
    /// This is the field that was added.
    pub fn for_material(material: &crate::document::Material) -> Self {
        let mut graph = Self::standard(
            material.name.clone(),
            material.base_color,
            material.metallic,
            material.roughness,
        );
        if let NodeKind::Surface {
            emission,
            transmission,
            ior,
            ..
        } = &mut graph.nodes[0].kind
        {
            *emission = material.emitted();
            *transmission = material.transmission;
            *ior = material.ior;
        }
        // `standard` puts the surface at zero and adds nothing else, so every
        // link below reaches nodes that are certainly there, of the types the
        // sockets want — which is why each `expect` is a statement about this
        // function and not about the caller.
        let add = |graph: &mut Self, name: &str, kind| {
            graph
                .add_node(Node::new(name, kind))
                .expect("only the surface can be refused")
        };
        let wire = |graph: &mut Self, from_node, from_socket, to_node, to_socket| {
            graph
                .link(Link {
                    from_node,
                    from_socket,
                    to_node,
                    to_socket,
                })
                .expect("the sockets agree")
        };
        if let Some(image) = material.base_color_image {
            let texture = add(
                &mut graph,
                "Base Colour",
                NodeKind::Image { image: Some(image) },
            );
            let factor = add(&mut graph, "Tint", NodeKind::Color(material.base_color));
            let product = add(&mut graph, "Base Colour x Tint", NodeKind::Multiply);
            wire(&mut graph, texture, 0, product, 0);
            wire(&mut graph, factor, 0, product, 1);
            wire(&mut graph, product, 0, 0, 0);
        }
        if let Some(image) = material.metallic_roughness_image {
            // glTF's packing: green is roughness, blue is metallic, and each
            // multiplies its factor. Red is occlusion and is not read here.
            let texture = add(
                &mut graph,
                "Metallic Roughness",
                NodeKind::Image { image: Some(image) },
            );
            let channels = add(&mut graph, "Channels", NodeKind::SeparateRgb);
            wire(&mut graph, texture, 0, channels, 0);
            for (channel, name, factor, socket) in [
                (1, "Roughness", material.roughness, 2),
                (2, "Metallic", material.metallic, 1),
            ] {
                let value = add(&mut graph, name, NodeKind::Value(factor));
                let product = add(&mut graph, name, NodeKind::MultiplyValue);
                wire(&mut graph, channels, channel, product, 0);
                wire(&mut graph, value, 0, product, 1);
                wire(&mut graph, product, 0, 0, socket);
            }
        }
        if let Some(image) = material.normal_image {
            let texture = add(&mut graph, "Normal", NodeKind::Image { image: Some(image) });
            let decode = add(
                &mut graph,
                "Normal Map",
                NodeKind::NormalMap {
                    strength: material.normal_strength,
                },
            );
            wire(&mut graph, texture, 0, decode, 0);
            wire(&mut graph, decode, 0, 0, 3);
        }
        graph.arrange();
        graph
    }

    /// What an image read by `node` should be declared as, judged by where
    /// its wires go.
    ///
    /// For the moment a file is picked for a texture node: a PNG does not say
    /// whether it is a colour or a roughness, and the slot is what knows
    /// (§2.1). Here the "slot" is the graph — an image feeding a normal map,
    /// a channel split or a number is data, and anything else is a colour.
    /// A guess only when the node is not wired yet, and then the guess is the
    /// common case.
    pub fn image_space(&self, node: usize) -> crate::image::ColorSpace {
        let data = self.links.iter().filter(|l| l.from_node == node).any(|l| {
            match self.nodes.get(l.to_node).map(|n| &n.kind) {
                Some(NodeKind::NormalMap { .. })
                | Some(NodeKind::SeparateRgb)
                | Some(NodeKind::MultiplyValue) => true,
                Some(NodeKind::Surface { .. }) => l.to_socket != 0,
                _ => false,
            }
        });
        if data {
            crate::image::ColorSpace::Raw
        } else {
            crate::image::ColorSpace::Srgb
        }
    }

    /// Lays the nodes out in columns by how far they are from the surface.
    ///
    /// For a graph nobody has arranged — one built from flat fields, or read
    /// from a file that had no positions. The surface on the right, and each
    /// node one column left of the leftmost thing it feeds, which is how a
    /// person reading a node tree expects to follow it: left to right, towards
    /// the output.
    pub fn arrange(&mut self) {
        const COLUMN: f32 = 240.0;
        const ROW: f32 = 140.0;
        let count = self.nodes.len();
        // Longest distance to the output, by relaxation. Bounded by the node
        // count because the graph is acyclic by construction; the bound is
        // what keeps a graph from a damaged file from looping.
        let mut depth = vec![0usize; count];
        for _ in 0..count {
            let mut changed = false;
            for link in &self.links {
                let (Some(&to), true) = (depth.get(link.to_node), link.from_node < count) else {
                    continue;
                };
                if depth[link.from_node] < to + 1 {
                    depth[link.from_node] = to + 1;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut rows = std::collections::HashMap::new();
        for (index, node) in self.nodes.iter_mut().enumerate() {
            let row = rows.entry(depth[index]).or_insert(0usize);
            node.position = [-(depth[index] as f32) * COLUMN, *row as f32 * ROW];
            *row += 1;
        }
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
        let from = self
            .nodes
            .get(link.from_node)
            .ok_or(LinkError::NoSuchNode)?;
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
                ..
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
            .add_node(Node::new(
                "Red",
                NodeKind::Color(SceneRgb::new(1.0, 0.0, 0.0)),
            ))
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
            .add_node(Node::new(
                "C",
                NodeKind::Color(SceneRgb::new(1.0, 1.0, 1.0)),
            ))
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
            .add_node(Node::new(
                "Red",
                NodeKind::Color(SceneRgb::new(1.0, 0.0, 0.0)),
            ))
            .expect("added");
        let blue = g
            .add_node(Node::new(
                "Blue",
                NodeKind::Color(SceneRgb::new(0.0, 0.0, 1.0)),
            ))
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
                    emission: SceneRgb::new(0.0, 0.0, 0.0),
                    transmission: 0.0,
                    ior: 1.5,
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
        assert_eq!(NodeKind::Color(SceneRgb::new(0.0, 0.0, 0.0)).tag(), "color");
    }

    // --- a material as a graph -----------------------------------------

    /// An image identity, which only a document can hand out.
    fn an_image(doc: &mut crate::Document) -> crate::document::ImageId {
        doc.add_image(crate::Image::new("Bricks", 4, 4, crate::PixelFormat::Rgba8))
    }

    #[test]
    fn a_material_without_a_texture_is_the_flat_graph() {
        let material = crate::document::Material {
            name: "Plastic".into(),
            base_color: SceneRgb::new(0.2, 0.4, 0.6),
            metallic: 0.25,
            roughness: 0.75,
            ..Default::default()
        };
        assert_eq!(
            ShaderGraph::for_material(&material),
            ShaderGraph::standard("Plastic", SceneRgb::new(0.2, 0.4, 0.6), 0.25, 0.75),
            "the shorter route stopped reaching the same model"
        );
    }

    #[test]
    fn a_texture_tints_the_colour_rather_than_replacing_it() {
        // **glTF's rule**: base colour is `baseColorTexture * baseColorFactor`.
        // A graph that wired the image straight into the surface would compile,
        // draw, and quietly stop the colour in the panel meaning anything — and
        // fixing that later would change the look of every material anybody had
        // already assigned a texture to.
        let mut doc = crate::Document::new();
        let image = an_image(&mut doc);
        let material = crate::document::Material {
            base_color: SceneRgb::new(0.5, 0.5, 0.5),
            base_color_image: Some(image),
            ..Default::default()
        };
        let graph = ShaderGraph::for_material(&material);
        let surface = graph.output().expect("a surface");
        let into_base = graph
            .incoming(surface, 0)
            .expect("the base colour is wired");
        let product = &graph.nodes[into_base.from_node];
        assert_eq!(
            product.kind,
            NodeKind::Multiply,
            "a texture replaced the colour"
        );

        // And both factors are there: the image on one side and the colour on
        // the other, in some order.
        let feeding: Vec<&NodeKind> = (0..2)
            .filter_map(|socket| graph.incoming(into_base.from_node, socket))
            .map(|l| &graph.nodes[l.from_node].kind)
            .collect();
        assert_eq!(feeding.len(), 2, "a factor is unwired");
        assert!(
            feeding.contains(&&NodeKind::Image { image: Some(image) }),
            "the texture is not a factor"
        );
        assert!(
            feeding.contains(&&NodeKind::Color(SceneRgb::new(0.5, 0.5, 0.5))),
            "the colour is not a factor"
        );
    }

    #[test]
    fn the_texture_a_material_names_is_the_one_the_graph_asks_for() {
        // The join the application makes: the compiler hands back identities
        // and the application resolves them. An identity that did not reach
        // `images()` would be a texture nobody uploaded and a surface sampling
        // whatever was in that layer.
        let mut doc = crate::Document::new();
        let image = an_image(&mut doc);
        let material = crate::document::Material {
            base_color_image: Some(image),
            ..Default::default()
        };
        assert_eq!(ShaderGraph::for_material(&material).images(), vec![image]);
    }

    #[test]
    fn every_map_a_material_names_is_one_the_graph_asks_for() {
        // Three maps, three images, each once. A map the graph did not reach
        // is a texture that never uploads and a slot that silently does
        // nothing.
        let mut doc = crate::Document::new();
        let (a, b, c) = (an_image(&mut doc), an_image(&mut doc), an_image(&mut doc));
        let material = crate::document::Material {
            base_color_image: Some(a),
            metallic_roughness_image: Some(b),
            normal_image: Some(c),
            ..Default::default()
        };
        let graph = ShaderGraph::for_material(&material);
        let images = graph.images();
        assert_eq!(images.len(), 3, "{images:?}");
        for id in [a, b, c] {
            assert!(images.contains(&id), "{id:?} is not asked for");
        }
        let surface = graph.output().expect("a surface");
        for socket in 0..4 {
            assert!(
                graph.incoming(surface, socket).is_some(),
                "socket {socket} is unwired"
            );
        }
    }

    #[test]
    fn a_normal_is_not_a_colour() {
        // The type that keeps a normal map from being wired into a base colour
        // raw — which would compile, and draw a surface lit by a purple
        // picture of itself.
        let mut graph = ShaderGraph::standard("M", SceneRgb::new(1.0, 1.0, 1.0), 0.0, 0.5);
        let decode = graph
            .add_node(Node::new("N", NodeKind::NormalMap { strength: 1.0 }))
            .expect("added");
        assert_eq!(
            graph.link(Link {
                from_node: decode,
                from_socket: 0,
                to_node: 0,
                to_socket: 0
            }),
            Err(LinkError::TypeMismatch {
                from: Socket::Vector,
                to: Socket::Color
            })
        );
        assert!(graph
            .link(Link {
                from_node: decode,
                from_socket: 0,
                to_node: 0,
                to_socket: 3
            })
            .is_ok());
    }

    #[test]
    fn every_kind_comes_back_from_its_parts() {
        // The file's inverse. A kind whose numbers came back in another order
        // is a material that loads as somebody else's.
        let mut doc = crate::Document::new();
        let image = an_image(&mut doc);
        let kinds = [
            NodeKind::Color(SceneRgb::new(0.1, 0.2, 0.3)),
            NodeKind::Value(0.7),
            NodeKind::Image { image: Some(image) },
            NodeKind::Multiply,
            NodeKind::SeparateRgb,
            NodeKind::MultiplyValue,
            NodeKind::NormalMap { strength: 0.4 },
            NodeKind::Surface {
                base_color: SceneRgb::new(0.9, 0.8, 0.7),
                metallic: 0.6,
                roughness: 0.3,
                emission: SceneRgb::new(2.0, 0.5, 0.0),
                transmission: 0.75,
                ior: 1.33,
            },
        ];
        for kind in kinds {
            let image = match kind {
                NodeKind::Image { image } => image,
                _ => None,
            };
            assert_eq!(
                NodeKind::from_parts(kind.tag(), &kind.values(), image),
                Some(kind.clone()),
                "{} did not come back",
                kind.tag()
            );
            assert_eq!(
                kind.input_names().len(),
                kind.inputs().len(),
                "{}",
                kind.tag()
            );
            assert_eq!(
                kind.output_names().len(),
                kind.outputs().len(),
                "{}",
                kind.tag()
            );
        }
        assert_eq!(NodeKind::from_parts("from_the_future", &[], None), None);
        for tag in NodeKind::ADDABLE {
            let made = NodeKind::new_of(tag).expect("an addable kind is a kind");
            assert_ne!(made.tag(), "surface");
        }
    }

    #[test]
    fn an_arranged_graph_reads_left_to_right() {
        // Every wire runs from a node to one strictly to its right, so the
        // tree reads towards the output.
        let mut doc = crate::Document::new();
        let (a, b) = (an_image(&mut doc), an_image(&mut doc));
        let graph = ShaderGraph::for_material(&crate::document::Material {
            base_color_image: Some(a),
            normal_image: Some(b),
            ..Default::default()
        });
        for link in &graph.links {
            assert!(
                graph.nodes[link.from_node].position[0] < graph.nodes[link.to_node].position[0],
                "a wire runs backwards: {link:?}"
            );
        }
        // And no two nodes sit on top of each other.
        for (i, x) in graph.nodes.iter().enumerate() {
            for y in &graph.nodes[i + 1..] {
                assert_ne!(x.position, y.position, "{} and {} overlap", x.name, y.name);
            }
        }
    }
}
