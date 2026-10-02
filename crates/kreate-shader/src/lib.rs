//! Turning a shading graph into WGSL.
//!
//! §3's fifth "decide now", and the half of it that is code:
//!
//! > **One graph, one compiler, two consumers.** The viewport and the renderer
//! > both run what this emits. No hand-written shading mode is added.
//!
//! Two systems that each interpret a material will diverge, the user will
//! report it as a bug, and it will be one — recurring for as long as both
//! implementations exist. §5's second risk is about exactly this, and the only
//! defence that works is that there is nothing to diverge *from*.
//!
//! # Its own crate, and the seam it keeps
//!
//! `kreate-render` never sees a [`Document`](kreate_scene::Document), and it
//! does not see a [`ShaderGraph`](kreate_scene::shader::ShaderGraph) either: a
//! material reaches it as text and a block of numbers, the way a mesh reaches
//! it as buffers. So the compiler sits where it can see the datablock and
//! cannot see the GPU, and what binds it to the renderer is a *contract* —
//! the `Surface` struct and the bindings named below — held honest by a test on
//! the renderer's side that composes what this emits against the real library.
//!
//! # Constants are data, not text
//!
//! A colour typed into a panel must not recompile a shader. So a node's
//! constants go into a uniform and the *shape* of the graph decides the text:
//! two materials that differ only in their numbers produce identical WGSL and
//! share a pipeline. That is what makes a pipeline cache possible at all, and
//! it is a property this file's tests assert rather than a hope.
//!
//! # What the emitted code assumes
//!
//! ```text
//! struct Surface        from kreate-render's brdf.wgsl
//! @group(3) @binding(0) var<uniform> material: MaterialConstants
//! @group(3) @binding(1) var material_textures: texture_2d_array<f32>   colours
//! @group(3) @binding(2) var material_sampler: sampler
//! @group(3) @binding(3) var material_data: texture_2d_array<f32>       numbers
//! ```
//!
//! A texture *array* rather than one binding per image, because bindless is not
//! in wgpu's core (v1 §4.7) and a binding per image would make the layout —
//! and therefore the pipeline — depend on how many images a material happens to
//! use.
//!
//! **Two arrays, because an image is either a colour or a number.** The colour
//! array is an sRGB format so the sampler decodes each tap before it filters,
//! which is the only order that is right; a roughness or normal map through
//! that array would come back bent by a curve it was never encoded with. The
//! data array is plain `Unorm`. Which one an image goes in is its declared
//! [`ColorSpace`], asked of the caller — see [`compile_with`] — and an image's
//! layer is its position among the images [`Compiled::images`] lists *of the
//! same space*.
//!
//! # What the emitted function is
//!
//! ```text
//! fn material_sample(uv, uv_dx, uv_dy) -> MaterialSample
//! struct MaterialSample { surface: Surface, normal: vec3<f32> }
//! ```
//!
//! The UV's screen derivatives are *arguments*. A fragment pass has them from
//! `dpdx`, and a compute pass — §2.5's path tracer — has none and must supply
//! its own from a ray cone. Computing them in here would make the material
//! compile only into fragment shaders. They choose the mip level: the sample
//! is `textureSampleGrad`, never `textureSample`, because the UV is wrapped
//! with `fract` first and the derivative of a wrapped coordinate jumps at the
//! wrap — `textureSample` would read that jump as a texture minified to one
//! texel and draw a seam in the smallest mip.
//!
//! `normal` is tangent-space, +Z out of the surface. The pass owns the tangent
//! frame, because it is geometry and the material is not.

use kreate_scene::document::ImageId;
use kreate_scene::shader::{NodeKind, ShaderGraph, Socket};
use kreate_scene::ColorSpace;
use std::fmt::Write as _;

/// The bind group a material's own data arrives in.
pub const MATERIAL_GROUP: u32 = 3;

/// What a compiled material needs alongside its code.
#[derive(Clone, Debug, PartialEq)]
pub struct Compiled {
    /// Which constant holds the transmission (x) and the index of
    /// refraction (y): what a renderer reads to know the material is glass,
    /// which decides when it is drawn.
    pub transmission_slot: usize,
    /// The function `material_sample(uv, uv_dx, uv_dy) -> MaterialSample`
    /// and its struct, and nothing else: no entry point, no library. A pass
    /// appends this to the shading library and calls it.
    pub wgsl: String,
    /// One `vec4` per constant slot, in the order the code indexes them.
    ///
    /// Padded to `vec4` because a `vec3` in a uniform array is aligned to
    /// sixteen bytes anyway — packing them tight would put every slot after the
    /// first in the wrong place, which is a material that is wrong in a way
    /// nobody can see the cause of.
    pub constants: Vec<[f32; 4]>,
    /// The images the graph samples, in the order the graph reaches them.
    pub images: Vec<ImageId>,
    /// Parallel to `images`: which array each one is in.
    ///
    /// An image's layer is its position among the images of its own space, so
    /// a renderer that packs each array in this list's order puts every image
    /// where the code looks for it.
    pub image_spaces: Vec<ColorSpace>,
    /// Which constant slot each image's rectangle goes in, parallel to
    /// `images`.
    ///
    /// The compiler reserves the slot and the *renderer* fills it, because
    /// where an image lands in an atlas is not something a graph knows. Stated
    /// as a list rather than left to be derived, so that a packer changing its
    /// mind about layers cannot silently write into a colour.
    pub image_slots: Vec<usize>,
    /// What the graph asked for and did not get.
    pub notes: Vec<Note>,
}

/// Something the compiler had to decide for the author.
///
/// Returned rather than logged, for the reason [`kreate_io`]'s notes are: a
/// fidelity claim that reports into a log is one nothing can test.
///
/// [`kreate_io`]: https://docs.rs/kreate-io
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Note {
    /// An image node with no image. It reads as white, so a material being
    /// built looks unfinished rather than broken.
    EmptyImage { node: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// Nothing to shade.
    NoSurface,
    /// A wire pointing at a node that is not there.
    ///
    /// Not reachable through [`ShaderGraph::link`], which checks. Reachable
    /// through a file, which can say anything.
    BrokenLink,
}

/// Compiles a graph whose images are all colours.
///
/// For a graph that only has a base colour map — and for tests, which is most
/// callers. Anything with a data map wants [`compile_with`], because the
/// images' spaces are the document's to say and not this crate's to guess.
pub fn compile(graph: &ShaderGraph) -> Result<Compiled, CompileError> {
    compile_with(graph, |_| ColorSpace::Srgb)
}

/// Compiles a graph, asking `space_of` what each image's numbers are.
///
/// A lookup rather than a copy in the graph, because the image datablock
/// already says — a second record of it in every node that samples it is a
/// second authority, and the two part company the first time somebody changes
/// an image's declared space.
pub fn compile_with(
    graph: &ShaderGraph,
    space_of: impl Fn(ImageId) -> ColorSpace,
) -> Result<Compiled, CompileError> {
    let output = graph.output().ok_or(CompileError::NoSurface)?;
    let mut out = Emitter {
        graph,
        space_of: &space_of,
        constants: Vec::new(),
        images: Vec::new(),
        image_spaces: Vec::new(),
        image_slots: Vec::new(),
        notes: Vec::new(),
        body: String::new(),
        emitted: vec![None; graph.nodes.len()],
    };
    out.emit(output)?;

    let NodeKind::Surface { .. } = graph.nodes[output].kind else {
        return Err(CompileError::NoSurface);
    };
    // The surface's own slot: rgb is the base colour a disconnected socket
    // takes, w is the metallic. Emitted even when every socket is wired, so
    // that the slot layout is a function of the graph's *shape* — a layout that
    // shrank when a wire was added would make two materials of the same shape
    // index different slots.
    let surface_slot = out.emitted[output].expect("the surface was emitted");

    let mut wgsl = String::new();
    // **Nothing here names the material**, and the omission is load-bearing.
    // A header comment saying which graph this came from is the obvious
    // courtesy, and it is the one thing that must not be written: the text is
    // the pipeline cache's key, so a name in it makes every material its own
    // pipeline and a scene of tinted plastics compiles a shader apiece. The
    // name belongs on the pipeline's *label*, where a debugger reads it and a
    // cache does not.
    let _ = writeln!(
        wgsl,
        "struct MaterialConstants {{ values: array<vec4<f32>, {}> }}",
        out.constants.len().max(1)
    );
    let _ = writeln!(
        wgsl,
        "@group({MATERIAL_GROUP}) @binding(0) var<uniform> material: MaterialConstants;"
    );
    let _ = writeln!(
        wgsl,
        "@group({MATERIAL_GROUP}) @binding(1) var material_textures: texture_2d_array<f32>;"
    );
    let _ = writeln!(
        wgsl,
        "@group({MATERIAL_GROUP}) @binding(2) var material_sampler: sampler;"
    );
    let _ = writeln!(
        wgsl,
        "@group({MATERIAL_GROUP}) @binding(3) var material_data: texture_2d_array<f32>;"
    );
    let _ = writeln!(wgsl);
    let _ = writeln!(
        wgsl,
        "struct MaterialSample {{ surface: Surface, normal: vec3<f32>, emission: vec3<f32>, transmission: f32, ior: f32 }}"
    );
    let _ = writeln!(wgsl);
    let _ = writeln!(
        wgsl,
        "fn material_sample(uv: vec2<f32>, uv_dx: vec2<f32>, uv_dy: vec2<f32>) -> MaterialSample {{\n{}    var s: Surface;",
        out.body
    );

    // The three inputs, each either a wire or the slot beside it.
    let base = out.input_expr(output, 0, &format!("material.values[{surface_slot}].rgb"))?;
    let metallic = out.input_expr(output, 1, &format!("material.values[{surface_slot}].w"))?;
    let roughness =
        out.input_expr(output, 2, &format!("material.values[{}].x", surface_slot + 1))?;
    let _ = writeln!(wgsl, "    s.base_color = {base};");
    let _ = writeln!(wgsl, "    s.metallic = clamp({metallic}, 0.0, 1.0);");
    // Clamped away from zero at the bottom, because a roughness of exactly zero
    // is a delta lobe and the microfacet arithmetic divides by its square.
    let _ = writeln!(wgsl, "    s.roughness = clamp({roughness}, 0.02, 1.0);");
    // Straight up when nothing is wired: the geometric normal, unchanged.
    let normal = out.input_expr(output, 3, "vec3<f32>(0.0, 0.0, 1.0)")?;
    let _ = writeln!(wgsl, "    var out: MaterialSample;");
    let _ = writeln!(wgsl, "    out.surface = s;");
    let _ = writeln!(wgsl, "    out.normal = {normal};");
    // In the roughness's slot, beside it: emitted light, in nits.
    let _ = writeln!(wgsl, "    out.emission = material.values[{}].yzw;", surface_slot + 1);
    // The third slot: how much passes through, and how it bends.
    let _ = writeln!(
        wgsl,
        "    out.transmission = clamp(material.values[{0}].x, 0.0, 1.0);\n    out.ior = max(material.values[{0}].y, 1.0);",
        surface_slot + 2
    );
    let _ = writeln!(wgsl, "    return out;\n}}");

    Ok(Compiled {
        transmission_slot: surface_slot + 2,
        wgsl,
        constants: out.constants,
        images: out.images,
        image_spaces: out.image_spaces,
        image_slots: out.image_slots,
        notes: out.notes,
    })
}

/// Walks the graph and writes the statements.
struct Emitter<'a> {
    graph: &'a ShaderGraph,
    space_of: &'a dyn Fn(ImageId) -> ColorSpace,
    constants: Vec<[f32; 4]>,
    images: Vec<ImageId>,
    image_spaces: Vec<ColorSpace>,
    image_slots: Vec<usize>,
    notes: Vec<Note>,
    body: String,
    /// For each node, the first constant slot it was given — and whether it has
    /// been emitted at all, which is what keeps a node feeding two sockets from
    /// being evaluated twice.
    emitted: Vec<Option<usize>>,
}

impl Emitter<'_> {
    /// Emits `node` and everything it depends on, once.
    fn emit(&mut self, node: usize) -> Result<(), CompileError> {
        if self.emitted.get(node).copied().flatten().is_some() {
            return Ok(());
        }
        let kind = &self
            .graph
            .nodes
            .get(node)
            .ok_or(CompileError::BrokenLink)?
            .kind;
        // Depth first, so an upstream node's `let` is written before the
        // statement that reads it. WGSL wants a declaration before its use, and
        // a graph is acyclic by construction — `ShaderGraph::link` refuses the
        // alternative.
        let feeding: Vec<usize> = (0..kind.inputs().len())
            .filter_map(|socket| self.graph.incoming(node, socket).map(|l| l.from_node))
            .collect();
        for upstream in feeding {
            self.emit(upstream)?;
        }

        let slot = self.constants.len();
        let name = self.graph.nodes[node].name.clone();
        match self.graph.nodes[node].kind {
            NodeKind::Color(c) => {
                self.constants.push([c.r, c.g, c.b, 0.0]);
                let _ = writeln!(
                    self.body,
                    "    let n{node} = material.values[{slot}].rgb;"
                );
            }
            NodeKind::Value(v) => {
                self.constants.push([v, 0.0, 0.0, 0.0]);
                let _ = writeln!(self.body, "    let n{node} = material.values[{slot}].x;");
            }
            NodeKind::Image { image } => {
                // A slot even though nothing reads it, so that the constant
                // layout depends on the graph's shape and not on which kinds
                // happen to need numbers. A layout that skipped a node would
                // change when an image was swapped for a colour.
                self.constants.push([0.0; 4]);
                match image {
                    Some(id) => {
                        let space = (self.space_of)(id);
                        let layer = self.image_spaces.iter().filter(|s| **s == space).count();
                        let array = match space {
                            ColorSpace::Srgb => "material_textures",
                            ColorSpace::Raw => "material_data",
                        };
                        self.images.push(id);
                        self.image_spaces.push(space);
                        self.image_slots.push(slot);
                        // The rectangle moves the UV into the image's corner of
                        // its layer. `fract` first, so that a surface whose UVs
                        // run past one repeats *its own image* rather than
                        // walking into whatever is packed beside it.
                        //
                        // An image smaller than its layer still picks up a
                        // little of the padding at its edge, where a linear tap
                        // straddles the boundary. A gutter is what a packer
                        // adds and this is where it would go; with one image to
                        // a layer it shows only on the right and bottom of the
                        // smaller images in a material.
                        let _ = writeln!(
                            self.body,
                            "    let n{node}_rect = material.values[{slot}];"
                        );
                        // The gradients are the UV's, scaled into the rectangle
                        // the same way the coordinate is — and taken from the
                        // *unwrapped* UV, which is the point of passing them:
                        // see the module docs on the seam `fract` would draw.
                        let _ = writeln!(
                            self.body,
                            "    let n{node}_rgba = textureSampleGrad({array}, material_sampler, n{node}_rect.xy + fract(uv) * n{node}_rect.zw, {layer}, uv_dx * n{node}_rect.zw, uv_dy * n{node}_rect.zw);"
                        );
                    }
                    None => {
                        // White, so a material being built looks unfinished
                        // rather than broken — the same judgement a missing
                        // environment's grey makes.
                        self.notes.push(Note::EmptyImage { node: name });
                        let _ = writeln!(self.body, "    let n{node}_rgba = vec4<f32>(1.0);");
                    }
                }
            }
            NodeKind::Multiply => {
                // A slot it does not read, for the reason the image node takes
                // one: the constant layout is the graph's *shape*, so swapping
                // a kind for another must not move everything after it.
                self.constants.push([0.0; 4]);
                // White on both sides, so an unwired factor is the identity
                // rather than black — a node that darkens to nothing while it
                // is being built reads as a broken material.
                let a = self.input_expr(node, 0, "vec3<f32>(1.0)")?;
                let b = self.input_expr(node, 1, "vec3<f32>(1.0)")?;
                let _ = writeln!(self.body, "    let n{node} = {a} * {b};");
            }
            NodeKind::SeparateRgb => {
                self.constants.push([0.0; 4]);
                let c = self.input_expr(node, 0, "vec3<f32>(1.0)")?;
                let _ = writeln!(self.body, "    let n{node} = {c};");
            }
            NodeKind::MultiplyValue => {
                self.constants.push([0.0; 4]);
                // One on both sides, for the reason the colour multiply uses
                // white: an unwired factor is the identity.
                let a = self.input_expr(node, 0, "1.0")?;
                let b = self.input_expr(node, 1, "1.0")?;
                let _ = writeln!(self.body, "    let n{node} = {a} * {b};");
            }
            NodeKind::NormalMap { strength } => {
                self.constants.push([strength, 0.0, 0.0, 0.0]);
                // Unwired is the flat normal's encoding, so the node alone is
                // the identity rather than a surface tipped toward +X+Y.
                let c = self.input_expr(node, 0, "vec3<f32>(0.5, 0.5, 1.0)")?;
                let _ = writeln!(self.body, "    let n{node}_t = {c} * 2.0 - vec3<f32>(1.0);");
                // Strength scales the tilt, not the vector: glTF's rule. Z is
                // kept off zero and off the back, because a badly encoded map
                // with a texel pointing into the surface would otherwise light
                // that texel from behind.
                let _ = writeln!(
                    self.body,
                    "    let n{node} = normalize(vec3<f32>(n{node}_t.xy * material.values[{slot}].x, max(n{node}_t.z, 1e-3)));"
                );
            }
            NodeKind::Surface { base_color, metallic, roughness, emission, transmission, ior } => {
                // Two slots: the colour and the metallic share one, the
                // roughness has the next. Always both, so that the layout is
                // the graph's shape — see the comment where they are read.
                self.constants.push([base_color.r, base_color.g, base_color.b, metallic]);
                self.constants.push([roughness, emission.r, emission.g, emission.b]);
                self.constants.push([transmission, ior, 0.0, 0.0]);
            }
        }
        self.emitted[node] = Some(slot);
        Ok(())
    }

    /// What feeds one input: a wire's expression, or the fallback.
    fn input_expr(
        &self,
        node: usize,
        socket: usize,
        fallback: &str,
    ) -> Result<String, CompileError> {
        let Some(link) = self.graph.incoming(node, socket) else {
            return Ok(fallback.to_string());
        };
        let from = self
            .graph
            .nodes
            .get(link.from_node)
            .ok_or(CompileError::BrokenLink)?;
        let kind = *from
            .kind
            .outputs()
            .get(link.from_socket)
            .ok_or(CompileError::BrokenLink)?;
        Ok(match (&from.kind, kind) {
            // An image's two outputs come out of one sample, which is why the
            // statement names the whole `vec4` and the sockets slice it.
            (NodeKind::Image { .. }, Socket::Color) => format!("n{}_rgba.rgb", link.from_node),
            (NodeKind::Image { .. }, Socket::Scalar) => format!("n{}_rgba.a", link.from_node),
            (NodeKind::SeparateRgb, _) => {
                let channel = ["x", "y", "z"]
                    .get(link.from_socket)
                    .ok_or(CompileError::BrokenLink)?;
                format!("n{}.{channel}", link.from_node)
            }
            _ => format!("n{}", link.from_node),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_math::SceneRgb;
    use kreate_scene::shader::{Link, Node};

    fn flat() -> ShaderGraph {
        ShaderGraph::standard("Grey", SceneRgb::new(0.8, 0.7, 0.6), 0.25, 0.4)
    }

    #[test]
    fn emission_rides_beside_the_roughness() {
        // In the slot the roughness already has, so a glow changes a number
        // and never the shape of the constants — the pipeline cache's key.
        let mut g = flat();
        let plain = compile(&g).expect("compiles");
        if let NodeKind::Surface { emission, .. } = &mut g.nodes[0].kind {
            *emission = SceneRgb::new(3.0, 2.0, 1.0);
        }
        let glowing = compile(&g).expect("compiles");
        assert_eq!(plain.wgsl, glowing.wgsl);
        assert_eq!(glowing.constants[1], [0.4, 3.0, 2.0, 1.0]);
        assert!(glowing.wgsl.contains("out.emission = material.values[1].yzw;"));
    }

    /// An image identity, which only a document can hand out.
    ///
    /// A real one rather than a placeholder, because a node with no image takes
    /// the white fallback and never samples anything — which is how the first
    /// version of the sampling test asked its question of a branch that does
    /// not sample.
    fn an_image() -> kreate_scene::document::ImageId {
        let mut doc = kreate_scene::Document::new();
        doc.add_image(kreate_scene::Image::new(
            "Albedo",
            16,
            16,
            kreate_scene::PixelFormat::Rgba8,
        ))
    }

    /// Parses the emitted function against a stand-in for the real `Surface`.
    ///
    /// A stand-in and not the real library, because that lives in
    /// `kreate-render` and this crate does not depend on it. What binds the two
    /// is a test *there* that composes this output against the real thing —
    /// this one only says the generated text is WGSL at all.
    fn parses(compiled: &Compiled) {
        let source = format!(
            "struct Surface {{ base_color: vec3<f32>, metallic: f32, roughness: f32 }}\n{}",
            compiled.wgsl
        );
        naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    }

    #[test]
    fn a_flat_material_compiles_to_its_three_constants() {
        let compiled = compile(&flat()).expect("compiles");
        parses(&compiled);
        assert_eq!(
            compiled.constants,
            vec![[0.8, 0.7, 0.6, 0.25], [0.4, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0]]
        );
        assert!(compiled.images.is_empty());
        assert!(compiled.notes.is_empty());
    }

    #[test]
    fn two_materials_of_the_same_shape_compile_to_the_same_code() {
        // **The property a pipeline cache is built on**, and the reason
        // constants are data rather than literals. A colour typed into a panel
        // must not recompile a shader — and the way that goes wrong is silent:
        // everything works, and the application stutters once per keystroke on
        // a colour slider.
        let a = compile(&flat()).expect("compiles");
        let b = compile(&ShaderGraph::standard(
            "Other",
            SceneRgb::new(0.1, 0.2, 0.3),
            1.0,
            0.9,
        ))
        .expect("compiles");
        // Compared whole, not with the comments stripped. The first version of
        // this stripped them, and that hid what the generator was doing: it was
        // writing the graph's *name* into a header comment, so every material
        // was its own pipeline — and the test meant to say otherwise had
        // removed the evidence. A test that filters what it compares has
        // decided in advance what could be wrong.
        //
        // Found by the renderer's cache test, which counted a hundred
        // pipelines for a hundred tinted plastics.
        assert_eq!(a.wgsl, b.wgsl, "the same graph shape produced different code");
        assert_ne!(a.constants, b.constants, "the numbers did not travel");
        // And the two graphs really were named differently, or the claim above
        // is about nothing.
        assert_ne!(flat().name, "Other");
    }

    #[test]
    fn a_wire_wins_over_the_socket_it_covers() {
        // What a connected input means. The constant beside it stays in the
        // layout — see the note where the slots are written — so that adding a
        // wire does not renumber every slot after it.
        let mut graph = flat();
        let surface = graph.output().expect("a surface");
        let red = graph
            .add_node(Node::new("Red", NodeKind::Color(SceneRgb::new(1.0, 0.0, 0.0))))
            .expect("added");
        graph
            .link(Link {
                from_node: red,
                from_socket: 0,
                to_node: surface,
                to_socket: 0,
            })
            .expect("connects");

        let compiled = compile(&graph).expect("compiles");
        parses(&compiled);
        assert!(
            compiled.wgsl.contains(&format!("s.base_color = n{red};")),
            "the base colour reads {:?}",
            compiled.wgsl
        );
        // And the other two still read their slots.
        assert!(compiled.wgsl.contains("s.metallic = clamp(material.values["));
    }

    #[test]
    fn a_node_feeding_two_sockets_is_evaluated_once() {
        // The reason the walk remembers what it has written. A texture sampled
        // twice because it feeds two inputs is the commonest way a node-based
        // material becomes twice as expensive as it looks.
        let mut graph = flat();
        let surface = graph.output().expect("a surface");
        let value = graph
            .add_node(Node::new("V", NodeKind::Value(0.3)))
            .expect("added");
        for socket in [1, 2] {
            graph
                .link(Link {
                    from_node: value,
                    from_socket: 0,
                    to_node: surface,
                    to_socket: socket,
                })
                .expect("connects");
        }
        let compiled = compile(&graph).expect("compiles");
        parses(&compiled);
        assert_eq!(
            compiled.wgsl.matches(&format!("let n{value} =")).count(),
            1,
            "the node was evaluated more than once:\n{}",
            compiled.wgsl
        );
    }

    #[test]
    fn an_image_is_sampled_once_and_sliced_twice() {
        // Its colour and its alpha come out of one sample. Two `textureSample`
        // calls for one image would be the same cost mistake as above, and this
        // is the case where it is easiest to make — the two sockets look
        // independent.
        let mut graph = flat();
        let surface = graph.output().expect("a surface");
        let image = graph
            .add_node(Node::new(
                "Albedo",
                NodeKind::Image {
                    image: Some(an_image()),
                },
            ))
            .expect("added");
        graph
            .link(Link {
                from_node: image,
                from_socket: 0,
                to_node: surface,
                to_socket: 0,
            })
            .expect("connects");
        graph
            .link(Link {
                from_node: image,
                from_socket: 1,
                to_node: surface,
                to_socket: 2,
            })
            .expect("connects");

        let compiled = compile(&graph).expect("compiles");
        assert_eq!(compiled.images.len(), 1);
        assert_eq!(compiled.image_slots.len(), 1);
        parses(&compiled);
        assert_eq!(compiled.wgsl.matches("textureSampleGrad(").count(), 1);
        assert!(compiled.wgsl.contains(&format!("s.base_color = n{image}_rgba.rgb;")));
        assert!(compiled
            .wgsl
            .contains(&format!("clamp(n{image}_rgba.a, 0.02, 1.0)")));
    }

    #[test]
    fn an_image_with_nothing_in_it_reads_white_and_says_so() {
        // A slot nobody has filled is an ordinary state while a material is
        // being built. White makes it look unfinished; black would make it look
        // broken, and the two are worth telling apart — the same judgement the
        // archive's missing-environment grey makes.
        let mut graph = flat();
        let surface = graph.output().expect("a surface");
        let image = graph
            .add_node(Node::new("Albedo", NodeKind::Image { image: None }))
            .expect("added");
        graph
            .link(Link {
                from_node: image,
                from_socket: 0,
                to_node: surface,
                to_socket: 0,
            })
            .expect("connects");
        let compiled = compile(&graph).expect("compiles");
        parses(&compiled);
        assert!(compiled.wgsl.contains("vec4<f32>(1.0)"));
        assert_eq!(
            compiled.notes,
            vec![Note::EmptyImage {
                node: "Albedo".into()
            }]
        );
        assert!(compiled.images.is_empty(), "an empty slot took a layer");
    }

    #[test]
    fn the_roughness_never_reaches_zero() {
        // A roughness of exactly zero is a delta lobe, and the microfacet
        // arithmetic divides by its square. Clamped in the generated code
        // rather than at every place a value could come from — a wire can carry
        // anything, including a texture's alpha.
        let compiled = compile(&flat()).expect("compiles");
        assert!(compiled.wgsl.contains("clamp(") && compiled.wgsl.contains("0.02"));
    }

    #[test]
    fn a_graph_with_nothing_to_shade_is_refused() {
        // Rather than compiled into something that draws a default. A material
        // whose author deleted the surface should say so, not silently become
        // grey plastic.
        let graph = ShaderGraph {
            name: "Empty".into(),
            linkage: kreate_scene::document::Linkage::Local,
            uid: kreate_scene::Uid::NONE,
            nodes: Vec::new(),
            links: Vec::new(),
        };
        assert_eq!(compile(&graph), Err(CompileError::NoSurface));
    }

    #[test]
    fn a_wire_from_a_node_that_is_not_there_is_refused() {
        // Not reachable through `link`, which checks. Reachable through a file,
        // which can say anything — and the compiler is the last place before a
        // shader that indexes past the end of an array.
        let mut graph = flat();
        let surface = graph.output().expect("a surface");
        graph.links.push(Link {
            from_node: 99,
            from_socket: 0,
            to_node: surface,
            to_socket: 0,
        });
        assert_eq!(compile(&graph), Err(CompileError::BrokenLink));
    }
}

#[cfg(test)]
mod multiply_tests {
    use super::*;
    use kreate_math::SceneRgb;
    use kreate_scene::shader::{Link, Node};

    /// The graph a textured material compiles to: image and colour, multiplied.
    fn tinted() -> (ShaderGraph, kreate_scene::document::ImageId) {
        let mut doc = kreate_scene::Document::new();
        let image = doc.add_image(kreate_scene::Image::new(
            "Bricks",
            4,
            4,
            kreate_scene::PixelFormat::Rgba8,
        ));
        let material = kreate_scene::Material {
            base_color: SceneRgb::new(0.5, 0.25, 0.125),
            base_color_image: Some(image),
            ..Default::default()
        };
        (ShaderGraph::for_material(&material), image)
    }

    #[test]
    fn a_tinted_texture_samples_once_and_multiplies() {
        let (graph, image) = tinted();
        let compiled = compile(&graph).expect("it compiles");
        assert_eq!(compiled.images, vec![image], "the texture is not uploaded");
        assert_eq!(
            compiled.wgsl.matches("textureSampleGrad").count(),
            1,
            "a tint should not cost a second tap"
        );
        assert!(
            compiled.wgsl.contains(" * "),
            "nothing is multiplied:\n{}",
            compiled.wgsl
        );
    }

    #[test]
    fn an_unwired_factor_is_white_rather_than_black() {
        // A node being built has one side connected and the other not. Zero
        // would make the surface black, which reads as a broken material rather
        // than an unfinished one — the same judgement the empty image node's
        // white fallback makes.
        let mut graph = ShaderGraph::standard("M", SceneRgb::new(0.8, 0.8, 0.8), 0.0, 0.5);
        let node = graph
            .add_node(Node::new("Multiply", kreate_scene::shader::NodeKind::Multiply))
            .expect("not a surface");
        graph
            .link(Link { from_node: node, from_socket: 0, to_node: 0, to_socket: 0 })
            .expect("a colour into a base colour");
        let compiled = compile(&graph).expect("it compiles");
        assert!(
            compiled.wgsl.contains("vec3<f32>(1.0) * vec3<f32>(1.0)"),
            "an unwired multiply is not the identity:\n{}",
            compiled.wgsl
        );
    }

    #[test]
    fn two_materials_that_differ_only_in_their_numbers_share_a_pipeline() {
        // The claim the whole design rests on, now that a graph has more than
        // one shape. Two textured materials with different colours and
        // different *images* must compile to the same text: the image is a
        // layer index and the colour is a constant, and neither is code.
        let mut doc = kreate_scene::Document::new();
        let one = doc.add_image(kreate_scene::Image::new("A", 4, 4, kreate_scene::PixelFormat::Rgba8));
        let two = doc.add_image(kreate_scene::Image::new("B", 8, 8, kreate_scene::PixelFormat::Rgba8));
        let graph_of = |image, color| {
            ShaderGraph::for_material(&kreate_scene::Material {
                base_color: color,
                base_color_image: Some(image),
                ..Default::default()
            })
        };
        let a = compile(&graph_of(one, SceneRgb::new(1.0, 0.0, 0.0))).expect("compiles");
        let b = compile(&graph_of(two, SceneRgb::new(0.0, 0.0, 1.0))).expect("compiles");
        assert_eq!(a.wgsl, b.wgsl, "a texture swap became a pipeline");
        assert_ne!(a.constants, b.constants, "the colours did not reach the numbers");
    }
}

#[cfg(test)]
mod map_tests {
    use super::*;
    use kreate_math::SceneRgb;

    /// A material with all three maps, each its own image.
    fn mapped() -> (ShaderGraph, [ImageId; 3]) {
        let mut doc = kreate_scene::Document::new();
        let mut add = |name: &str| {
            doc.add_image(kreate_scene::Image::new(name, 4, 4, kreate_scene::PixelFormat::Rgba8))
        };
        let (colour, packed, normal) = (add("C"), add("MR"), add("N"));
        let material = kreate_scene::Material {
            base_color: SceneRgb::new(0.5, 0.5, 0.5),
            metallic: 0.25,
            roughness: 0.75,
            base_color_image: Some(colour),
            metallic_roughness_image: Some(packed),
            normal_image: Some(normal),
            normal_strength: 0.5,
            ..Default::default()
        };
        (ShaderGraph::for_material(&material), [colour, packed, normal])
    }

    /// The spaces a document would answer with: the colour map sRGB, the
    /// others raw.
    fn spaces(ids: [ImageId; 3]) -> impl Fn(ImageId) -> ColorSpace {
        move |id| if id == ids[0] { ColorSpace::Srgb } else { ColorSpace::Raw }
    }

    fn parses(compiled: &Compiled) {
        let source = format!(
            "struct Surface {{ base_color: vec3<f32>, metallic: f32, roughness: f32 }}\n{}",
            compiled.wgsl
        );
        naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    }

    #[test]
    fn each_image_is_sampled_from_the_array_its_space_says() {
        // Colour from the sRGB array and the two data maps from the plain one,
        // each numbered from zero within its own array — the layer rule the
        // renderer's packer follows.
        let (graph, ids) = mapped();
        let compiled = compile_with(&graph, spaces(ids)).expect("compiles");
        parses(&compiled);
        assert_eq!(compiled.images.len(), 3);
        assert_eq!(
            compiled.image_spaces,
            compiled
                .images
                .iter()
                .map(|id| spaces(ids)(*id))
                .collect::<Vec<_>>()
        );
        assert_eq!(compiled.wgsl.matches("textureSampleGrad(material_textures").count(), 1);
        assert_eq!(compiled.wgsl.matches("textureSampleGrad(material_data").count(), 2);
        assert!(compiled.wgsl.contains("material_textures, material_sampler, n"));
        // Two data images: layers 0 and 1 of the data array, not 1 and 2.
        let data_layers: Vec<&str> = compiled
            .wgsl
            .lines()
            .filter(|l| l.contains("textureSampleGrad(material_data"))
            .map(|l| l.split(".zw, ").nth(1).and_then(|r| r.split(',').next()).unwrap_or(""))
            .collect();
        assert_eq!(data_layers, vec!["0", "1"], "{}", compiled.wgsl);
    }

    #[test]
    fn the_packed_map_feeds_roughness_from_green_and_metallic_from_blue() {
        // glTF's packing. Swapping the two is a material that is shiny where it
        // was painted rough, and nothing but the picture would say so.
        let (graph, ids) = mapped();
        let compiled = compile_with(&graph, spaces(ids)).expect("compiles");
        let line = |field: &str| {
            compiled
                .wgsl
                .lines()
                .find(|l| l.contains(&format!("s.{field} = ")))
                .expect("the field is set")
                .to_string()
        };
        // Each is a product node; find what feeds it.
        let product_of = |field: &str| {
            let l = line(field);
            let name = l.split("clamp(").nth(1).and_then(|r| r.split(',').next()).expect("a node");
            compiled
                .wgsl
                .lines()
                .find(|x| x.contains(&format!("let {name} = ")))
                .expect("the product is emitted")
                .to_string()
        };
        assert!(product_of("roughness").contains(".y *"), "{}", compiled.wgsl);
        assert!(product_of("metallic").contains(".z *"), "{}", compiled.wgsl);
    }

    #[test]
    fn a_normal_map_reaches_the_output_and_nothing_else_does() {
        let (graph, ids) = mapped();
        let compiled = compile_with(&graph, spaces(ids)).expect("compiles");
        assert!(compiled.wgsl.contains("* 2.0 - vec3<f32>(1.0)"), "the map is not decoded");
        assert!(!compiled.wgsl.contains("out.normal = vec3<f32>(0.0, 0.0, 1.0);"));
        // And its strength is a constant, not text: two strengths, one shader.
        let flat = compile(&ShaderGraph::standard("F", SceneRgb::new(1.0, 1.0, 1.0), 0.0, 0.5))
            .expect("compiles");
        assert!(flat.wgsl.contains("out.normal = vec3<f32>(0.0, 0.0, 1.0);"));
        assert!(compiled.constants.iter().any(|c| c[0] == 0.5), "the strength did not travel");
    }

    #[test]
    fn a_graph_of_colour_images_compiles_the_same_either_way() {
        // `compile` is `compile_with` everything sRGB. If the two ever
        // differed for a graph with only a base colour, the tests that use the
        // short form would be about a different shader from the application's.
        let (graph, _) = mapped();
        assert_eq!(
            compile(&graph).expect("compiles"),
            compile_with(&graph, |_| ColorSpace::Srgb).expect("compiles")
        );
    }
}

impl Compiled {
    /// Whether light passes through: drawn after the solids, over them.
    pub fn transmissive(&self) -> bool {
        self.constants.get(self.transmission_slot).is_some_and(|c| c[0] > 0.0)
    }
}
