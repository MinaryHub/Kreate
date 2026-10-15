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
    /// A point a texture is read at: the surface's position in its object,
    /// its UV, or either moved by a Mapping node.
    ///
    /// Not a [`Socket::Vector`], though Blender draws both alike: a normal
    /// wired in as a place to read a noise at is a texture that follows the
    /// light around, which nobody meant.
    Coordinate,
}

impl Socket {
    pub const ALL: [Self; 4] = [Self::Color, Self::Scalar, Self::Vector, Self::Coordinate];

    /// A stable name for files.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Color => "color",
            Self::Scalar => "scalar",
            Self::Vector => "vector",
            Self::Coordinate => "coordinate",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.tag() == tag)
    }
}

/// One of a group's sockets: what it is called, what flows through it, and
/// where inside the group it goes.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupPort {
    pub name: String,
    pub socket: Socket,
    /// For an input, every inner socket it feeds, as (node, input); for an
    /// output, the one inner socket it reads, as (node, output).
    pub inner: Vec<(usize, usize)>,
}

/// Nodes folded into one: Blender's node group, carried by the node that
/// stands for it.
///
/// Its nodes keep their own wires and positions — relative to the group's
/// middle, so they come back round the node when it is ungrouped. A wire into
/// one of the group's inputs feeds every inner socket the input names; an
/// inner socket nothing outside feeds keeps its own value. The group is
/// opened up again before compiling, so the shader is the same whether the
/// nodes are folded or not.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeGroup {
    pub name: String,
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub inputs: Vec<GroupPort>,
    pub outputs: Vec<GroupPort>,
}

/// Most stops a Color Ramp holds.
pub const RAMP_STOPS: usize = 8;

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
    /// An image, sampled at the surface's UV — or projected from three sides.
    ///
    /// `None` is a slot nobody has filled, which is an ordinary state while a
    /// material is being built — it reads as the fallback rather than failing
    /// to compile, for the same reason a world with a missing image lands on a
    /// colour.
    Image {
        image: Option<ImageId>,
        projection: Projection,
        /// For a box projection: how softly the three sides cross over, zero
        /// a hard edge and one an even blend.
        blend: f32,
        /// For a box projection: repeats per metre of the object.
        scale: f32,
    },
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
    /// A height, turned into a tilt of the surface: Blender's Bump.
    ///
    /// `distance` is how far one unit of height stands off the surface, in
    /// metres, and `strength` how much of the tilt is used. A normal wired in
    /// is tilted further, so a bump can sit on top of a normal map.
    Bump { strength: f32, distance: f32 },
    /// A tangent-space normal map's colour, decoded into a direction.
    ///
    /// `strength` scales the tilt before renormalising, which is glTF's
    /// `normalTexture.scale`: zero is flat, one is the map as painted.
    NormalMap { strength: f32 },
    /// Two colours mixed by a factor: Blender's Mix (Color, Mix). `factor` is
    /// what an unwired Factor socket takes.
    MixColor { factor: f32 },
    /// A number mapped through a row of colours: Blender's Color Ramp. The
    /// first `count` stops are used, each a position and a colour; what is
    /// before the first is its colour, after the last the last's.
    ColorRamp {
        interpolation: RampMode,
        count: u8,
        stops: [(f32, SceneRgb); RAMP_STOPS],
    },
    /// Where a surface point is, several ways: Blender's Texture Coordinate.
    TexCoord,
    /// Nodes folded into one. See [`NodeGroup`].
    Group(Box<NodeGroup>),
    /// A coordinate scaled, then rotated (in degrees, X then Y then Z), then
    /// moved: Blender's Mapping, as a point.
    Mapping {
        location: [f32; 3],
        rotation: [f32; 3],
        scale: [f32; 3],
    },
    /// Two numbers through an operation: Blender's Math. `a` and `b` are
    /// what unwired sockets take.
    Math { op: MathOp, a: f32, b: f32 },
    /// One minus a colour: Blender's Invert Color.
    Invert,
    /// Fractal gradient noise, read at the surface's position in its object:
    /// Blender's Noise Texture. The numbers are what unwired sockets take.
    ///
    /// Object space rather than UVs, so a mesh nobody unwrapped still gets a
    /// texture without seams — and it moves with the object, not the world.
    Noise {
        scale: f32,
        /// How many octaves, fractional between two.
        detail: f32,
        /// How much each octave keeps of the one before.
        roughness: f32,
        /// How far the position is pushed by noise of its own first.
        distortion: f32,
    },
    /// Cells around scattered points: Blender's Voronoi Texture, F1.
    Voronoi { scale: f32, randomness: f32 },
    /// Alternating cubes of two colours: Blender's Checker Texture.
    Checker {
        scale: f32,
        color1: SceneRgb,
        color2: SceneRgb,
    },
    /// Bands or rings, bent by noise: Blender's Wave Texture, which is where
    /// wood and marble start.
    Wave {
        kind: WaveKind,
        scale: f32,
        distortion: f32,
        detail: f32,
        detail_scale: f32,
    },
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
        /// Opacity where the Alpha socket is not wired, and what it multiplies
        /// where it is.
        alpha: f32,
        /// Cut away below this opacity; zero is never.
        clip: f32,
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

/// How a Color Ramp goes from one stop to the next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RampMode {
    #[default]
    Linear,
    /// Each stop's colour until the next: hard bands.
    Constant,
    /// Smoothly in and out of each stop.
    Ease,
}

impl RampMode {
    pub const ALL: [Self; 3] = [Self::Linear, Self::Constant, Self::Ease];

    pub fn tag(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Constant => "constant",
            Self::Ease => "ease",
        }
    }

    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|o| *o == self).unwrap_or(0) as u32
    }

    pub fn from_index(i: u32) -> Self {
        Self::ALL.get(i as usize).copied().unwrap_or_default()
    }
}

/// How an image is laid on a surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    /// By the mesh's UVs.
    #[default]
    Flat,
    /// From the object's three axes, each side blended by how squarely it
    /// faces that axis: Blender's Box, otherwise called triplanar. A mesh
    /// nobody unwrapped still takes a tiling texture without stretching.
    Box,
}

impl Projection {
    pub const ALL: [Self; 2] = [Self::Flat, Self::Box];

    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|o| *o == self).unwrap_or(0) as u32
    }

    pub fn from_index(i: u32) -> Self {
        Self::ALL.get(i as usize).copied().unwrap_or_default()
    }
}

/// Which way a Wave node's stripes run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WaveKind {
    /// Straight, across the diagonal.
    #[default]
    Bands,
    /// Around the object's origin: wood.
    Rings,
}

impl WaveKind {
    pub const ALL: [Self; 2] = [Self::Bands, Self::Rings];

    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|o| *o == self).unwrap_or(0) as u32
    }

    pub fn from_index(i: u32) -> Self {
        Self::ALL.get(i as usize).copied().unwrap_or_default()
    }
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
    /// An image node laid on by UV, as every image node was before there was
    /// a choice.
    pub fn image(image: Option<ImageId>) -> Self {
        NodeKind::Image {
            image,
            projection: Projection::Flat,
            blend: 0.2,
            scale: 1.0,
        }
    }

    /// The types of this node's inputs, in order.
    pub fn inputs(&self) -> std::borrow::Cow<'static, [Socket]> {
        if let NodeKind::Group(g) = self {
            return g.inputs.iter().map(|p| p.socket).collect();
        }
        std::borrow::Cow::Borrowed(self.fixed_inputs())
    }

    fn fixed_inputs(&self) -> &'static [Socket] {
        match self {
            NodeKind::Group(_) => &[],
            NodeKind::Color(_) | NodeKind::Value(_) | NodeKind::Image { .. } => &[],
            NodeKind::Multiply => &[Socket::Color, Socket::Color],
            NodeKind::SeparateRgb | NodeKind::NormalMap { .. } => &[Socket::Color],
            NodeKind::MultiplyValue | NodeKind::Math { .. } => &[Socket::Scalar, Socket::Scalar],
            NodeKind::MixColor { .. } => &[Socket::Scalar, Socket::Color, Socket::Color],
            NodeKind::Invert => &[Socket::Color],
            // Each texture's point last, so the sockets files already name
            // keep their numbers.
            NodeKind::Noise { .. } | NodeKind::Wave { .. } => &[
                Socket::Scalar,
                Socket::Scalar,
                Socket::Scalar,
                Socket::Scalar,
                Socket::Coordinate,
            ],
            NodeKind::Voronoi { .. } => &[Socket::Scalar, Socket::Scalar, Socket::Coordinate],
            NodeKind::Checker { .. } => &[
                Socket::Color,
                Socket::Color,
                Socket::Scalar,
                Socket::Coordinate,
            ],
            NodeKind::ColorRamp { .. } => &[Socket::Scalar],
            NodeKind::TexCoord => &[],
            NodeKind::Mapping { .. } => &[Socket::Coordinate],
            NodeKind::Bump { .. } => &[
                Socket::Scalar,
                Socket::Scalar,
                Socket::Scalar,
                Socket::Vector,
            ],
            // Base colour, metallic, roughness, normal, a tint on the glow,
            // opacity, and occlusion — each new one last, so the sockets
            // files name keep their numbers.
            NodeKind::Surface { .. } => &[
                Socket::Color,
                Socket::Scalar,
                Socket::Scalar,
                Socket::Vector,
                Socket::Color,
                Socket::Scalar,
                Socket::Scalar,
            ],
        }
    }

    /// The types of this node's outputs, in order.
    pub fn outputs(&self) -> std::borrow::Cow<'static, [Socket]> {
        if let NodeKind::Group(g) = self {
            return g.outputs.iter().map(|p| p.socket).collect();
        }
        std::borrow::Cow::Borrowed(self.fixed_outputs())
    }

    fn fixed_outputs(&self) -> &'static [Socket] {
        match self {
            NodeKind::Group(_) => &[],
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
            NodeKind::NormalMap { .. } | NodeKind::Bump { .. } => &[Socket::Vector],
            // Blender's order: the noises lead with their number, the
            // patterns with their colour.
            NodeKind::Noise { .. } | NodeKind::Voronoi { .. } => &[Socket::Scalar, Socket::Color],
            NodeKind::Checker { .. } | NodeKind::Wave { .. } => &[Socket::Color, Socket::Scalar],
            NodeKind::ColorRamp { .. } => &[Socket::Color],
            // Object, UV, and the normal as a direction to read at.
            NodeKind::TexCoord => &[Socket::Coordinate, Socket::Coordinate, Socket::Coordinate],
            NodeKind::Mapping { .. } => &[Socket::Coordinate],
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
            NodeKind::Bump { .. } => "bump",
            NodeKind::MixColor { .. } => "mix",
            NodeKind::Math { .. } => "math",
            NodeKind::Invert => "invert",
            NodeKind::Noise { .. } => "noise",
            NodeKind::Voronoi { .. } => "voronoi",
            NodeKind::Checker { .. } => "checker",
            NodeKind::Wave { .. } => "wave",
            NodeKind::ColorRamp { .. } => "color_ramp",
            NodeKind::TexCoord => "tex_coord",
            NodeKind::Mapping { .. } => "mapping",
            NodeKind::Surface { .. } => "surface",
            NodeKind::Group(_) => "group",
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
            NodeKind::Bump { strength, distance } => vec![*strength, *distance],
            NodeKind::MixColor { factor } => vec![*factor],
            NodeKind::Math { op, a, b } => vec![op.index() as f32, *a, *b],
            NodeKind::Image {
                projection,
                blend,
                scale,
                ..
            } => vec![projection.index() as f32, *blend, *scale],
            NodeKind::Noise {
                scale,
                detail,
                roughness,
                distortion,
            } => vec![*scale, *detail, *roughness, *distortion],
            NodeKind::Voronoi { scale, randomness } => vec![*scale, *randomness],
            NodeKind::Checker {
                scale,
                color1,
                color2,
            } => vec![
                *scale, color1.r, color1.g, color1.b, color2.r, color2.g, color2.b,
            ],
            NodeKind::Wave {
                kind,
                scale,
                distortion,
                detail,
                detail_scale,
            } => vec![
                *scale,
                *distortion,
                *detail,
                *detail_scale,
                kind.index() as f32,
            ],
            NodeKind::ColorRamp {
                interpolation,
                count,
                stops,
            } => {
                let mut v = vec![interpolation.index() as f32, *count as f32];
                for (at, c) in stops {
                    v.extend([*at, c.r, c.g, c.b]);
                }
                v
            }
            NodeKind::Mapping {
                location,
                rotation,
                scale,
            } => location
                .iter()
                .chain(rotation)
                .chain(scale)
                .copied()
                .collect(),
            NodeKind::Surface {
                base_color,
                metallic,
                roughness,
                emission,
                transmission,
                ior,
                alpha,
                clip,
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
                    *alpha,
                    *clip,
                ]
            }
            NodeKind::Multiply
            | NodeKind::TexCoord
            | NodeKind::SeparateRgb
            | NodeKind::MultiplyValue
            | NodeKind::Invert
            | NodeKind::Group(_) => Vec::new(),
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
            // A file from before projections has no numbers: by UV.
            "image" => NodeKind::Image {
                image,
                projection: Projection::from_index(at(0, 0.0).max(0.0) as u32),
                blend: at(1, 0.2),
                scale: at(2, 1.0),
            },
            "multiply" => NodeKind::Multiply,
            "separate_rgb" => NodeKind::SeparateRgb,
            "multiply_value" => NodeKind::MultiplyValue,
            "normal_map" => NodeKind::NormalMap {
                strength: at(0, 1.0),
            },
            "bump" => NodeKind::Bump {
                strength: at(0, 1.0),
                distance: at(1, 0.1),
            },
            "mix" => NodeKind::MixColor { factor: at(0, 0.5) },
            "math" => NodeKind::Math {
                op: MathOp::from_index(at(0, 0.0).max(0.0) as u32),
                a: at(1, 0.5),
                b: at(2, 0.5),
            },
            "invert" => NodeKind::Invert,
            // Blender's defaults, so a node dropped in looks like the one a
            // tutorial shows.
            "noise" => NodeKind::Noise {
                scale: at(0, 5.0),
                detail: at(1, 2.0),
                roughness: at(2, 0.5),
                distortion: at(3, 0.0),
            },
            "voronoi" => NodeKind::Voronoi {
                scale: at(0, 5.0),
                randomness: at(1, 1.0),
            },
            "checker" => NodeKind::Checker {
                scale: at(0, 5.0),
                color1: SceneRgb::new(at(1, 0.8), at(2, 0.8), at(3, 0.8)),
                color2: SceneRgb::new(at(4, 0.2), at(5, 0.2), at(6, 0.2)),
            },
            "wave" => NodeKind::Wave {
                scale: at(0, 5.0),
                distortion: at(1, 0.0),
                detail: at(2, 2.0),
                detail_scale: at(3, 1.0),
                kind: WaveKind::from_index(at(4, 0.0).max(0.0) as u32),
            },
            "color_ramp" => {
                // Black to white, as Blender's starts.
                let mut stops = [(1.0, SceneRgb::new(1.0, 1.0, 1.0)); RAMP_STOPS];
                for (i, stop) in stops.iter_mut().enumerate() {
                    let base = 2 + i * 4;
                    let (at_d, grey) = if i == 0 { (0.0, 0.0) } else { (1.0, 1.0) };
                    *stop = (
                        at(base, at_d),
                        SceneRgb::new(at(base + 1, grey), at(base + 2, grey), at(base + 3, grey)),
                    );
                }
                NodeKind::ColorRamp {
                    interpolation: RampMode::from_index(at(0, 0.0).max(0.0) as u32),
                    count: (at(1, 2.0).round() as i64).clamp(2, RAMP_STOPS as i64) as u8,
                    stops,
                }
            }
            "tex_coord" => NodeKind::TexCoord,
            "mapping" => NodeKind::Mapping {
                location: [at(0, 0.0), at(1, 0.0), at(2, 0.0)],
                rotation: [at(3, 0.0), at(4, 0.0), at(5, 0.0)],
                scale: [at(6, 1.0), at(7, 1.0), at(8, 1.0)],
            },
            "surface" => NodeKind::Surface {
                base_color: SceneRgb::new(at(0, 0.8), at(1, 0.8), at(2, 0.8)),
                metallic: at(3, 0.0),
                roughness: at(4, 0.5),
                // Absent in every file written before emission: none.
                emission: SceneRgb::new(at(5, 0.0), at(6, 0.0), at(7, 0.0)),
                transmission: at(8, 0.0),
                ior: at(9, 1.5),
                // Absent in every file written before cut-outs: solid.
                alpha: at(10, 1.0),
                clip: at(11, 0.0),
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
    pub const ADDABLE: [&'static str; 18] = [
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
        "noise",
        "voronoi",
        "checker",
        "wave",
        "bump",
        "color_ramp",
        "tex_coord",
        "mapping",
    ];

    /// What each input is called, for a person.
    pub fn input_names(&self) -> Vec<std::borrow::Cow<'static, str>> {
        match self {
            NodeKind::Group(g) => g.inputs.iter().map(|p| p.name.clone().into()).collect(),
            _ => self
                .fixed_input_names()
                .iter()
                .map(|n| (*n).into())
                .collect(),
        }
    }

    fn fixed_input_names(&self) -> &'static [&'static str] {
        match self {
            NodeKind::Group(_) => &[],
            NodeKind::Multiply | NodeKind::MultiplyValue | NodeKind::Math { .. } => &["A", "B"],
            NodeKind::SeparateRgb | NodeKind::NormalMap { .. } | NodeKind::Invert => &["Color"],
            NodeKind::MixColor { .. } => &["Factor", "A", "B"],
            NodeKind::Noise { .. } => &["Scale", "Detail", "Roughness", "Distortion", "Vector"],
            NodeKind::Voronoi { .. } => &["Scale", "Randomness", "Vector"],
            NodeKind::ColorRamp { .. } => &["Fac"],
            NodeKind::TexCoord => &[],
            NodeKind::Mapping { .. } => &["Vector"],
            NodeKind::Bump { .. } => &["Strength", "Distance", "Height", "Normal"],
            NodeKind::Checker { .. } => &["Color1", "Color2", "Scale", "Vector"],
            NodeKind::Wave { .. } => &["Scale", "Distortion", "Detail", "Detail Scale", "Vector"],
            NodeKind::Surface { .. } => &[
                "Base Color",
                "Metallic",
                "Roughness",
                "Normal",
                "Emission",
                "Alpha",
                "Occlusion",
            ],
            NodeKind::Color(_) | NodeKind::Value(_) | NodeKind::Image { .. } => &[],
        }
    }

    /// What each output is called.
    pub fn output_names(&self) -> Vec<std::borrow::Cow<'static, str>> {
        match self {
            NodeKind::Group(g) => g.outputs.iter().map(|p| p.name.clone().into()).collect(),
            _ => self
                .fixed_output_names()
                .iter()
                .map(|n| (*n).into())
                .collect(),
        }
    }

    fn fixed_output_names(&self) -> &'static [&'static str] {
        match self {
            NodeKind::Group(_) => &[],
            NodeKind::Color(_)
            | NodeKind::Multiply
            | NodeKind::MixColor { .. }
            | NodeKind::Invert => &["Color"],
            NodeKind::Value(_) | NodeKind::MultiplyValue | NodeKind::Math { .. } => &["Value"],
            NodeKind::Image { .. } => &["Color", "Alpha"],
            NodeKind::SeparateRgb => &["R", "G", "B"],
            NodeKind::NormalMap { .. } | NodeKind::Bump { .. } => &["Normal"],
            NodeKind::Noise { .. } => &["Fac", "Color"],
            NodeKind::Voronoi { .. } => &["Distance", "Color"],
            NodeKind::Checker { .. } | NodeKind::Wave { .. } => &["Color", "Fac"],
            NodeKind::ColorRamp { .. } => &["Color"],
            NodeKind::TexCoord => &["Object", "UV", "Normal"],
            NodeKind::Mapping { .. } => &["Vector"],
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
                    alpha: 1.0,
                    clip: 0.0,
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
            alpha,
            clip,
            ..
        } = &mut graph.nodes[0].kind
        {
            *emission = material.emitted();
            *transmission = material.transmission;
            *ior = material.ior;
            *alpha = material.alpha;
            *clip = material.alpha_clip.unwrap_or(0.0);
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
            let texture = add(&mut graph, "Base Colour", NodeKind::image(Some(image)));
            let factor = add(&mut graph, "Tint", NodeKind::Color(material.base_color));
            let product = add(&mut graph, "Base Colour x Tint", NodeKind::Multiply);
            wire(&mut graph, texture, 0, product, 0);
            wire(&mut graph, factor, 0, product, 1);
            wire(&mut graph, product, 0, 0, 0);
            // A cut-out reads the picture's own alpha: a leaf is where the
            // image says, and the material's opacity multiplies it.
            if material.alpha_clip.is_some() {
                wire(&mut graph, texture, 1, 0, 5);
            }
        }
        // An occlusion map's red, eased towards none by its strength: the
        // light from the surroundings that reaches into a crease. From the
        // metal-roughness picture's own red when the file packed them
        // together, as glTF's ORM maps do.
        // r · strength + (1 − strength): all of it at full strength, none
        // at zero.
        let strength = material.occlusion_strength.clamp(0.0, 1.0);
        let occlusion_from = |graph: &mut Self, channels: usize| {
            let scaled = add(
                graph,
                "Occlusion Strength",
                NodeKind::Math {
                    op: MathOp::Multiply,
                    a: 1.0,
                    b: strength,
                },
            );
            let eased = add(
                graph,
                "Occlusion",
                NodeKind::Math {
                    op: MathOp::Add,
                    a: 0.0,
                    b: 1.0 - strength,
                },
            );
            wire(graph, channels, 0, scaled, 0);
            wire(graph, scaled, 0, eased, 0);
            wire(graph, eased, 0, 0, 6);
        };
        if let Some(image) = material
            .occlusion_image
            .filter(|o| Some(*o) != material.metallic_roughness_image)
        {
            let texture = add(&mut graph, "Occlusion Map", NodeKind::image(Some(image)));
            let channels = add(&mut graph, "Occlusion Channels", NodeKind::SeparateRgb);
            wire(&mut graph, texture, 0, channels, 0);
            occlusion_from(&mut graph, channels);
        }
        if let Some(image) = material.metallic_roughness_image {
            // glTF's packing: green is roughness, blue is metallic, and each
            // multiplies its factor; red is occlusion, when the file says so.
            let texture = add(
                &mut graph,
                "Metallic Roughness",
                NodeKind::image(Some(image)),
            );
            let channels = add(&mut graph, "Channels", NodeKind::SeparateRgb);
            wire(&mut graph, texture, 0, channels, 0);
            if material.occlusion_image == Some(image) {
                occlusion_from(&mut graph, channels);
            }
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
            let texture = add(&mut graph, "Normal", NodeKind::image(Some(image)));
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
        if let Some(image) = material.emission_image {
            // The glow's tint: the strength and colour stay the material's,
            // the picture says where and in what colour.
            let texture = add(&mut graph, "Emission", NodeKind::image(Some(image)));
            wire(&mut graph, texture, 0, 0, 4);
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
                // Metallic, roughness and the normal are numbers; the base
                // colour, the glow's tint and an alpha are not about which.
                Some(NodeKind::Surface { .. }) => matches!(l.to_socket, 1..=3 | 6),
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
        fn walk(nodes: &[Node], out: &mut Vec<ImageId>) {
            for n in nodes {
                match &n.kind {
                    NodeKind::Image {
                        image: Some(image), ..
                    } => out.push(*image),
                    NodeKind::Group(g) => walk(&g.nodes, out),
                    _ => {}
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.nodes, &mut out);
        out
    }

    /// Whether any node here is a group.
    pub fn has_groups(&self) -> bool {
        self.nodes
            .iter()
            .any(|n| matches!(n.kind, NodeKind::Group(_)))
    }

    /// The graph with every group opened up, however deep: what the compiler
    /// reads. Positions are where ungrouping would put the nodes.
    pub fn flattened(&self) -> ShaderGraph {
        let mut g = self.clone();
        // Each pass opens the first group; a group's own groups come out as
        // nodes of this graph and are opened in a later pass.
        while let Some(n) = g
            .nodes
            .iter()
            .position(|n| matches!(n.kind, NodeKind::Group(_)))
        {
            g.ungroup(n);
        }
        g
    }

    /// Opens the group at `node`: its nodes become this graph's, wired as
    /// the group wired them, round where the group stood. Returns the
    /// indices the inner nodes landed at, or `None` when `node` is not a
    /// group.
    pub fn ungroup(&mut self, node: usize) -> Option<std::ops::Range<usize>> {
        let NodeKind::Group(group) = self.nodes.get(node)?.kind.clone() else {
            return None;
        };
        let at = self.nodes[node].position;
        let offset = self.nodes.len();
        for inner in &group.nodes {
            let mut inner = inner.clone();
            inner.position = [inner.position[0] + at[0], inner.position[1] + at[1]];
            self.nodes.push(inner);
        }
        let mut links: Vec<Link> = group
            .links
            .iter()
            .map(|l| Link {
                from_node: l.from_node + offset,
                from_socket: l.from_socket,
                to_node: l.to_node + offset,
                to_socket: l.to_socket,
            })
            .collect();
        for l in &self.links {
            if l.to_node == node {
                // Into the group: to every inner socket the input feeds.
                if let Some(port) = group.inputs.get(l.to_socket) {
                    for (n, s) in &port.inner {
                        links.push(Link {
                            from_node: l.from_node,
                            from_socket: l.from_socket,
                            to_node: n + offset,
                            to_socket: *s,
                        });
                    }
                }
            } else if l.from_node == node {
                // Out of it: from the inner socket the output reads.
                if let Some((n, s)) = group
                    .outputs
                    .get(l.from_socket)
                    .and_then(|p| p.inner.first())
                {
                    links.push(Link {
                        from_node: n + offset,
                        from_socket: *s,
                        to_node: l.to_node,
                        to_socket: l.to_socket,
                    });
                }
            } else {
                links.push(*l);
            }
        }
        self.links = links;
        let count = group.nodes.len();
        // The group goes; everything after it moves down one.
        self.remove_node(node);
        Some(offset - 1..offset - 1 + count)
    }

    /// Folds `picked` into one group node, called `name`, and returns where
    /// it landed. A wire from outside into the picked nodes becomes one of
    /// the group's inputs — one per outside socket, feeding everything it
    /// fed; a wire from them to outside becomes an output. The surface
    /// cannot be grouped; `None` when nothing else is picked.
    pub fn group(&mut self, picked: &[usize], name: &str) -> Option<usize> {
        let mut picked: Vec<usize> = picked
            .iter()
            .copied()
            .filter(|n| {
                self.nodes
                    .get(*n)
                    .is_some_and(|node| !matches!(node.kind, NodeKind::Surface { .. }))
            })
            .collect();
        picked.sort_unstable();
        picked.dedup();
        if picked.is_empty() {
            return None;
        }
        let inside = |n: usize| picked.binary_search(&n).ok();
        let middle = picked.iter().fold([0.0f32; 2], |m, n| {
            let p = self.nodes[*n].position;
            [m[0] + p[0], m[1] + p[1]]
        });
        let middle = [
            middle[0] / picked.len() as f32,
            middle[1] / picked.len() as f32,
        ];
        let nodes: Vec<Node> = picked
            .iter()
            .map(|n| {
                let mut node = self.nodes[*n].clone();
                node.position = [node.position[0] - middle[0], node.position[1] - middle[1]];
                node
            })
            .collect();
        let mut group = NodeGroup {
            name: name.to_string(),
            nodes,
            links: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        // Outside wires, kept to be re-made once the group stands.
        let mut feeds: Vec<(usize, usize, usize)> = Vec::new(); // (from node, from socket, port)
        let mut reads: Vec<(usize, usize, usize)> = Vec::new(); // (port, to node, to socket)
        for l in &self.links {
            match (inside(l.from_node), inside(l.to_node)) {
                (Some(a), Some(b)) => group.links.push(Link {
                    from_node: a,
                    from_socket: l.from_socket,
                    to_node: b,
                    to_socket: l.to_socket,
                }),
                (None, Some(b)) => {
                    let port = match feeds
                        .iter()
                        .find(|(n, s, _)| *n == l.from_node && *s == l.from_socket)
                    {
                        Some((_, _, port)) => *port,
                        None => {
                            let node = &self.nodes[l.to_node];
                            group.inputs.push(GroupPort {
                                name: node
                                    .kind
                                    .input_names()
                                    .get(l.to_socket)
                                    .map_or_else(String::new, |n| n.to_string()),
                                socket: node.kind.inputs()[l.to_socket],
                                inner: Vec::new(),
                            });
                            feeds.push((l.from_node, l.from_socket, group.inputs.len() - 1));
                            group.inputs.len() - 1
                        }
                    };
                    group.inputs[port].inner.push((b, l.to_socket));
                }
                (Some(a), None) => {
                    let port = match group
                        .outputs
                        .iter()
                        .position(|p| p.inner.first() == Some(&(a, l.from_socket)))
                    {
                        Some(port) => port,
                        None => {
                            let node = &self.nodes[l.from_node];
                            group.outputs.push(GroupPort {
                                name: node
                                    .kind
                                    .output_names()
                                    .get(l.from_socket)
                                    .map_or_else(String::new, |n| n.to_string()),
                                socket: node.kind.outputs()[l.from_socket],
                                inner: vec![(a, l.from_socket)],
                            });
                            group.outputs.len() - 1
                        }
                    };
                    reads.push((port, l.to_node, l.to_socket));
                }
                (None, None) => {}
            }
        }
        // What is left, renumbered, and the group last.
        let keep: Vec<usize> = (0..self.nodes.len())
            .filter(|n| inside(*n).is_none())
            .collect();
        let renumber = |n: usize| keep.binary_search(&n).ok();
        let mut graph_nodes: Vec<Node> = keep.iter().map(|n| self.nodes[*n].clone()).collect();
        let at = graph_nodes.len();
        let mut node = Node::new(name, NodeKind::Group(Box::new(group)));
        node.position = middle;
        graph_nodes.push(node);
        let mut links: Vec<Link> = self
            .links
            .iter()
            .filter_map(|l| {
                Some(Link {
                    from_node: renumber(l.from_node)?,
                    from_socket: l.from_socket,
                    to_node: renumber(l.to_node)?,
                    to_socket: l.to_socket,
                })
            })
            .collect();
        for (from, socket, port) in feeds {
            if let Some(from) = renumber(from) {
                links.push(Link {
                    from_node: from,
                    from_socket: socket,
                    to_node: at,
                    to_socket: port,
                });
            }
        }
        for (port, to, socket) in reads {
            if let Some(to) = renumber(to) {
                links.push(Link {
                    from_node: at,
                    from_socket: port,
                    to_node: to,
                    to_socket: socket,
                });
            }
        }
        self.nodes = graph_nodes;
        self.links = links;
        Some(at)
    }
}

#[cfg(test)]
mod tests {

    /// A small chain: noise into a ramp into the surface's colour, and the
    /// noise's number also into the roughness.
    fn chain() -> ShaderGraph {
        let mut g = graph();
        let noise = g
            .add_node(Node::new("Noise", NodeKind::new_of("noise").unwrap()))
            .unwrap();
        let ramp = g
            .add_node(Node::new("Ramp", NodeKind::new_of("color_ramp").unwrap()))
            .unwrap();
        let coord = g.add_node(Node::new("Coord", NodeKind::TexCoord)).unwrap();
        let out = g.output().unwrap();
        for (a, s, b, t) in [
            (coord, 1, noise, 4),
            (noise, 0, ramp, 0),
            (ramp, 0, out, 0),
            (noise, 0, out, 2),
        ] {
            g.link(Link {
                from_node: a,
                from_socket: s,
                to_node: b,
                to_socket: t,
            })
            .unwrap();
        }
        g
    }

    /// Folding nodes into a group and opening it again gives back the same
    /// nodes and wires; the flattened group is the graph it was.
    #[test]
    fn a_group_opens_back_into_the_nodes_it_folded() {
        let g = chain();
        let mut grouped = g.clone();
        // The noise and the ramp, not the coordinate or the surface.
        let at = grouped.group(&[1, 2], "Pattern").unwrap();
        assert_eq!(grouped.nodes.len(), 3, "surface, coordinate, group");
        let NodeKind::Group(group) = &grouped.nodes[at].kind else {
            panic!("a group")
        };
        assert_eq!(group.inputs.len(), 1, "the coordinate, once");
        assert_eq!(
            group.outputs.len(),
            2,
            "the ramp's colour and the noise's number"
        );
        assert_eq!(&*grouped.nodes[at].kind.inputs(), &[Socket::Coordinate]);
        // Opened, it is the same graph up to the order of nodes and wires.
        let flat = grouped.flattened();
        assert!(!flat.has_groups());
        let shape = |g: &ShaderGraph| {
            let name = |n: usize| g.nodes[n].name.clone();
            let mut links: Vec<_> = g
                .links
                .iter()
                .map(|l| {
                    (
                        name(l.from_node),
                        l.from_socket,
                        name(l.to_node),
                        l.to_socket,
                    )
                })
                .collect();
            links.sort();
            let mut nodes: Vec<_> = g
                .nodes
                .iter()
                .map(|n| (n.name.clone(), n.kind.clone(), n.position))
                .collect();
            nodes.sort_by(|a, b| a.0.cmp(&b.0));
            (nodes, links)
        };
        assert_eq!(shape(&flat), shape(&g));
    }

    #[test]
    fn a_group_inside_a_group_opens_all_the_way() {
        let mut g = chain();
        let inner = g.group(&[1, 2], "Inner").unwrap();
        let coord = g.nodes.iter().position(|n| n.name == "Coord").unwrap();
        g.group(&[inner, coord], "Outer").unwrap();
        assert_eq!(g.nodes.len(), 2);
        let flat = g.flattened();
        assert_eq!(flat.nodes.len(), 4);
        assert_eq!(flat.links.len(), 4);
        assert_eq!(flat.images(), Vec::<ImageId>::new());
    }

    #[test]
    fn the_surface_is_never_grouped() {
        let mut g = chain();
        let out = g.output().unwrap();
        assert_eq!(g.group(&[out], "Nothing"), None);
        let at = g.group(&[out, 1], "Noise only").unwrap();
        assert!(g.output().is_some());
        assert!(matches!(g.nodes[at].kind, NodeKind::Group(_)));
    }

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
            .add_node(Node::new("Albedo", NodeKind::image(None)))
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
                    alpha: 1.0,
                    clip: 0.0,
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
        assert_eq!(NodeKind::image(None).tag(), "image");
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
            feeding.contains(&&NodeKind::image(Some(image))),
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
            NodeKind::image(Some(image)),
            NodeKind::Image {
                image: Some(image),
                projection: Projection::Box,
                blend: 0.6,
                scale: 3.0,
            },
            NodeKind::Multiply,
            NodeKind::SeparateRgb,
            NodeKind::MultiplyValue,
            NodeKind::NormalMap { strength: 0.4 },
            NodeKind::Bump {
                strength: 0.7,
                distance: 0.02,
            },
            NodeKind::ColorRamp {
                interpolation: RampMode::Ease,
                count: 3,
                stops: [(0.25, SceneRgb::new(0.1, 0.2, 0.3)); RAMP_STOPS],
            },
            NodeKind::TexCoord,
            NodeKind::Mapping {
                location: [1.0, 2.0, 3.0],
                rotation: [10.0, 20.0, 30.0],
                scale: [0.5, 0.25, 2.0],
            },
            NodeKind::Noise {
                scale: 3.0,
                detail: 4.5,
                roughness: 0.6,
                distortion: 0.2,
            },
            NodeKind::Voronoi {
                scale: 7.0,
                randomness: 0.25,
            },
            NodeKind::Checker {
                scale: 2.0,
                color1: SceneRgb::new(1.0, 0.0, 0.0),
                color2: SceneRgb::new(0.0, 0.0, 1.0),
            },
            NodeKind::Wave {
                kind: WaveKind::Rings,
                scale: 1.5,
                distortion: 4.0,
                detail: 3.0,
                detail_scale: 2.0,
            },
            NodeKind::Surface {
                base_color: SceneRgb::new(0.9, 0.8, 0.7),
                metallic: 0.6,
                roughness: 0.3,
                emission: SceneRgb::new(2.0, 0.5, 0.0),
                transmission: 0.75,
                ior: 1.33,
                alpha: 0.8,
                clip: 0.5,
            },
        ];
        for kind in kinds {
            let image = match kind {
                NodeKind::Image { image, .. } => image,
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
