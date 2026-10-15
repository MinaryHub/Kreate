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
//! fn material_sample(uv, uv_dx, uv_dy, geometry) -> MaterialSample
//! struct MaterialSample { surface: Surface, normal: vec3<f32> }
//! struct MaterialGeometry { position, normal, position_dx, position_dy }
//! ```
//!
//! `geometry` is the point being shaded *in its object's space*, and how that
//! point moves across a pixel: what a procedural texture is read at. Object
//! space so that a texture moves with its object and does not swim when the
//! camera does, which an eye-relative position would make it do.
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
use kreate_scene::shader::{MathOp, NodeKind, Projection, RampMode, ShaderGraph, Socket, WaveKind};
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
    /// The function `material_sample(uv, uv_dx, uv_dy, geometry) -> MaterialSample`
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
    // Groups opened up: the shader is the same folded or not.
    let flat;
    let graph = if graph.has_groups() {
        flat = graph.flattened();
        &flat
    } else {
        graph
    };
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
        procedural: false,
        functions: String::new(),
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
        "struct MaterialGeometry {{ position: vec3<f32>, normal: vec3<f32>, position_dx: vec3<f32>, position_dy: vec3<f32> }}"
    );
    // Where a coordinate lands in an image, which is stored top row first
    // while V runs up — Blender's and OBJ's way, and Kreate's. Without the
    // turn every texture was upside down, which an atlas of parts shows as
    // the wrong part on every face.
    if !out.images.is_empty() {
        wgsl.push_str(TEXEL);
    }
    // Only when something reads it: the text is the pipeline cache's key, and
    // two hundred lines of noise in every flat plastic's shader is a slower
    // compile for nothing.
    if out.procedural {
        wgsl.push_str(PROCEDURAL);
    }
    let _ = writeln!(
        wgsl,
        "struct MaterialSample {{ surface: Surface, normal: vec3<f32>, gradient: vec3<f32>, emission: vec3<f32>, transmission: f32, ior: f32, alpha: f32, clip: f32, occlusion: f32 }}"
    );
    let _ = writeln!(wgsl);
    wgsl.push_str(&out.functions);
    let _ = writeln!(
        wgsl,
        "fn material_sample(uv: vec2<f32>, uv_dx: vec2<f32>, uv_dy: vec2<f32>, geometry: MaterialGeometry) -> MaterialSample {{\n{}    var s: Surface;",
        out.body
    );

    // The three inputs, each either a wire or the slot beside it.
    let base = out.input_expr(output, 0, &format!("material.values[{surface_slot}].rgb"))?;
    let metallic = out.input_expr(output, 1, &format!("material.values[{surface_slot}].w"))?;
    let roughness = out.input_expr(
        output,
        2,
        &format!("material.values[{}].x", surface_slot + 1),
    )?;
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
    let gradient = out.gradient_expr(output, 3)?;
    let _ = writeln!(wgsl, "    out.gradient = {gradient};");
    // In the roughness's slot, beside it: emitted light, in nits — tinted by
    // whatever is wired to the Emission socket, so a texture can say where a
    // surface glows and the material says how brightly.
    // Unwired, the line is what it always was: the shader text is the
    // pipeline cache's key, and a glow nobody tinted should not cost one.
    match out.graph.incoming(output, 4) {
        Some(_) => {
            let tint = out.input_expr(output, 4, "vec3<f32>(1.0)")?;
            let _ = writeln!(
                wgsl,
                "    out.emission = material.values[{}].yzw * {tint};",
                surface_slot + 1
            );
        }
        None => {
            let _ = writeln!(
                wgsl,
                "    out.emission = material.values[{}].yzw;",
                surface_slot + 1
            );
        }
    }
    // The third slot: how much passes through, and how it bends.
    let _ = writeln!(
        wgsl,
        "    out.transmission = clamp(material.values[{0}].x, 0.0, 1.0);\n    out.ior = max(material.values[{0}].y, 1.0);",
        surface_slot + 2
    );
    // Opacity, and below what it is cut away: the constant, times whatever
    // is wired to Alpha — an image's alpha, as glTF multiplies its factor.
    let alpha = match out.graph.incoming(output, 5) {
        Some(_) => format!(
            "material.values[{}].z * {}",
            surface_slot + 2,
            out.input_expr(output, 5, "1.0")?
        ),
        None => format!("material.values[{}].z", surface_slot + 2),
    };
    let _ = writeln!(
        wgsl,
        "    out.alpha = clamp({alpha}, 0.0, 1.0);\n    out.clip = material.values[{}].w;",
        surface_slot + 2
    );
    // How much of the surroundings' light reaches in: one, all of it, when
    // nothing says otherwise.
    let occlusion = out.input_expr(output, 6, "1.0")?;
    let _ = writeln!(wgsl, "    out.occlusion = clamp({occlusion}, 0.0, 1.0);");
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
    /// Whether a procedural texture was reached, and so the noise library is
    /// wanted.
    procedural: bool,
    /// Functions the body calls: a bump's height, which it reads three times.
    functions: String,
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
        // Not a bump's height: that is read in a function of its own, at three
        // points, and evaluating it here as well would be a fourth for nothing.
        let bump = matches!(kind, NodeKind::Bump { .. });
        let feeding: Vec<usize> = (0..kind.inputs().len())
            .filter(|socket| !(bump && *socket == 2))
            .filter_map(|socket| self.graph.incoming(node, socket).map(|l| l.from_node))
            .collect();
        for upstream in feeding {
            self.emit(upstream)?;
        }

        let slot = self.constants.len();
        let name = self.graph.nodes[node].name.clone();
        match self.graph.nodes[node].kind {
            // Opened before compiling; one still here came round some other
            // way and is not something to guess about.
            NodeKind::Group(_) => return Err(CompileError::BrokenLink),
            NodeKind::Color(c) => {
                self.constants.push([c.r, c.g, c.b, 0.0]);
                let _ = writeln!(self.body, "    let n{node} = material.values[{slot}].rgb;");
            }
            NodeKind::Value(v) => {
                self.constants.push([v, 0.0, 0.0, 0.0]);
                let _ = writeln!(self.body, "    let n{node} = material.values[{slot}].x;");
            }
            NodeKind::Image {
                image,
                projection,
                blend,
                scale,
            } => {
                // A slot even though nothing reads it, so that the constant
                // layout depends on the graph's shape and not on which kinds
                // happen to need numbers. A layout that skipped a node would
                // change when an image was swapped for a colour.
                self.constants.push([0.0; 4]);
                // And for a box, a second: the blend and the scale. Only for
                // a box, which is a different shape anyway.
                if projection == Projection::Box {
                    self.constants.push([blend, scale, 0.0, 0.0]);
                }
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
                        let _ =
                            writeln!(self.body, "    let n{node}_rect = material.values[{slot}];");
                        // The gradients are the UV's, scaled into the rectangle
                        // the same way the coordinate is — and taken from the
                        // *unwrapped* UV, which is the point of passing them:
                        // see the module docs on the seam `fract` would draw.
                        match projection {
                            Projection::Flat => {
                                let _ = writeln!(
                                    self.body,
                                    "    let n{node}_rgba = textureSampleGrad({array}, material_sampler, n{node}_rect.xy + kr_texel(uv) * n{node}_rect.zw, {layer}, kr_down(uv_dx) * n{node}_rect.zw, kr_down(uv_dy) * n{node}_rect.zw);"
                                );
                            }
                            Projection::Box => {
                                // Three samples, one down each axis, each with
                                // the position's own gradients in that plane —
                                // and weighted by how squarely the surface
                                // faces it. A blend of zero is a hard edge
                                // (a high power), one an even mix.
                                let b = slot + 1;
                                let _ = writeln!(
                                    self.body,
                                    "    let n{node}_k = mix(48.0, 1.0, clamp(material.values[{b}].x, 0.0, 1.0));\n    let n{node}_a = pow(abs(geometry.normal), vec3<f32>(n{node}_k)) + vec3<f32>(1e-6);\n    let n{node}_w = n{node}_a / (n{node}_a.x + n{node}_a.y + n{node}_a.z);\n    let n{node}_p = geometry.position * material.values[{b}].y;\n    let n{node}_dx = geometry.position_dx * material.values[{b}].y;\n    let n{node}_dy = geometry.position_dy * material.values[{b}].y;"
                                );
                                let mut taps = Vec::new();
                                for (axis, plane) in [("x", "yz"), ("y", "xz"), ("z", "xy")] {
                                    let _ = writeln!(
                                        self.body,
                                        "    let n{node}_{axis} = textureSampleGrad({array}, material_sampler, n{node}_rect.xy + kr_texel(n{node}_p.{plane}) * n{node}_rect.zw, {layer}, kr_down(n{node}_dx.{plane}) * n{node}_rect.zw, kr_down(n{node}_dy.{plane}) * n{node}_rect.zw);"
                                    );
                                    taps.push(format!("n{node}_{axis} * n{node}_w.{axis}"));
                                }
                                let _ = writeln!(
                                    self.body,
                                    "    let n{node}_rgba = {};",
                                    taps.join(" + ")
                                );
                            }
                        }
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
            NodeKind::MixColor { factor } => {
                self.constants.push([factor, 0.0, 0.0, 0.0]);
                let f = self.input_expr(node, 0, &format!("material.values[{slot}].x"))?;
                let a = self.input_expr(node, 1, "vec3<f32>(0.5)")?;
                let b = self.input_expr(node, 2, "vec3<f32>(0.5)")?;
                let _ = writeln!(
                    self.body,
                    "    let n{node} = mix({a}, {b}, clamp({f}, 0.0, 1.0));"
                );
            }
            NodeKind::Math { op, a, b } => {
                self.constants.push([a, b, 0.0, 0.0]);
                let x = self.input_expr(node, 0, &format!("material.values[{slot}].x"))?;
                let y = self.input_expr(node, 1, &format!("material.values[{slot}].y"))?;
                // Blender's answers where the arithmetic has none: zero for a
                // division by zero, and for a power of a negative number.
                let e = match op {
                    MathOp::Add => format!("{x} + {y}"),
                    MathOp::Subtract => format!("{x} - {y}"),
                    MathOp::Multiply => format!("{x} * {y}"),
                    MathOp::Divide => format!("select(0.0, {x} / {y}, {y} != 0.0)"),
                    MathOp::Power => format!("select(0.0, pow({x}, {y}), {x} > 0.0)"),
                    MathOp::Minimum => format!("min({x}, {y})"),
                    MathOp::Maximum => format!("max({x}, {y})"),
                };
                let _ = writeln!(self.body, "    let n{node} = {e};");
            }
            NodeKind::Invert => {
                self.constants.push([0.0; 4]);
                let c = self.input_expr(node, 0, "vec3<f32>(0.0)")?;
                let _ = writeln!(self.body, "    let n{node} = vec3<f32>(1.0) - {c};");
            }
            NodeKind::Bump { strength, distance } => {
                self.constants.push([strength, distance, 0.0, 0.0]);
                let strength = self.input_expr(node, 0, &format!("material.values[{slot}].x"))?;
                let distance = self.input_expr(node, 1, &format!("material.values[{slot}].y"))?;
                match self.graph.incoming(node, 2).map(|l| l.from_node) {
                    Some(from) => {
                        // The height's subgraph, as a function of where it is
                        // read. Emitted with its own record of what has been
                        // written — the function cannot see the body's `let`s —
                        // into the same constants, so its slots are the shape's.
                        let body = std::mem::take(&mut self.body);
                        let emitted = std::mem::replace(
                            &mut self.emitted,
                            vec![None; self.graph.nodes.len()],
                        );
                        self.emit(from)?;
                        let height = self.input_expr(node, 2, "0.0")?;
                        let inner = std::mem::replace(&mut self.body, body);
                        self.emitted = emitted;
                        let _ = writeln!(
                            self.functions,
                            "fn kr_height_{node}(uv: vec2<f32>, uv_dx: vec2<f32>, uv_dy: vec2<f32>, geometry: MaterialGeometry) -> f32 {{\n{inner}    return {height};\n}}\n"
                        );
                        // Read here and one pixel across and down, the way a
                        // fragment's derivatives would — but by evaluating, so
                        // the same code runs where there are no derivatives.
                        // Then Mikkelsen's surface gradient, in object space.
                        let _ = writeln!(
                            self.body,
                            "    var n{node}_gx = geometry;\n    n{node}_gx.position = geometry.position + geometry.position_dx;\n    var n{node}_gy = geometry;\n    n{node}_gy.position = geometry.position + geometry.position_dy;\n    let n{node}_h = kr_height_{node}(uv, uv_dx, uv_dy, geometry);\n    let n{node}_dhx = (kr_height_{node}(uv + uv_dx, uv_dx, uv_dy, n{node}_gx) - n{node}_h) * {distance};\n    let n{node}_dhy = (kr_height_{node}(uv + uv_dy, uv_dx, uv_dy, n{node}_gy) - n{node}_h) * {distance};\n    let n{node}_r1 = cross(geometry.position_dy, geometry.normal);\n    let n{node}_r2 = cross(geometry.normal, geometry.position_dx);\n    let n{node}_det = dot(geometry.position_dx, n{node}_r1);\n    let n{node}_g = select(vec3<f32>(0.0), (n{node}_dhx * n{node}_r1 + n{node}_dhy * n{node}_r2) * (sign(n{node}_det) / max(abs(n{node}_det), 1e-30)), abs(n{node}_det) > 1e-30) * clamp({strength}, 0.0, 1.0);"
                        );
                    }
                    None => {
                        let _ = writeln!(self.body, "    let n{node}_g = vec3<f32>(0.0);");
                    }
                }
                // The tilt passes a normal wired into it through unchanged:
                // the bump's own tilt rides beside it, as a gradient.
                let n = self.input_expr(node, 3, "vec3<f32>(0.0, 0.0, 1.0)")?;
                let _ = writeln!(self.body, "    let n{node} = {n};");
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
            NodeKind::Noise {
                scale,
                detail,
                roughness,
                distortion,
            } => {
                self.procedural = true;
                self.constants.push([scale, detail, roughness, distortion]);
                let [s, d, r, w] = self.numbers(node, slot)?;
                let p = format!("n{node}_p");
                let at = self.input_expr(node, 4, "geometry.position")?;
                let _ = writeln!(self.body, "    let {p} = {at} * {s};");
                let _ = writeln!(
                    self.body,
                    "    let n{node}_q = kr_distort({p}, {w}, {d}, {r});"
                );
                let _ = writeln!(
                    self.body,
                    "    let n{node}_f = kr_fbm(n{node}_q, {d}, {r});"
                );
                // Two more octave stacks for the colour, so only when the
                // colour is wired — reading the shape, as everything here does.
                if self.feeds(node, 1) {
                    let _ = writeln!(
                        self.body,
                        "    let n{node}_c = vec3<f32>(n{node}_f, kr_fbm(n{node}_q + vec3<f32>(17.3, 5.1, 9.7), {d}, {r}), kr_fbm(n{node}_q + vec3<f32>(3.9, 23.2, 13.4), {d}, {r}));"
                    );
                }
            }
            NodeKind::Voronoi { scale, randomness } => {
                self.procedural = true;
                self.constants.push([scale, randomness, 0.0, 0.0]);
                let s = self.input_expr(node, 0, &format!("material.values[{slot}].x"))?;
                let j = self.input_expr(node, 1, &format!("material.values[{slot}].y"))?;
                let at = self.input_expr(node, 2, "geometry.position")?;
                let _ = writeln!(
                    self.body,
                    "    let n{node}_v = kr_voronoi({at} * {s}, clamp({j}, 0.0, 1.0));"
                );
                let _ = writeln!(self.body, "    let n{node}_f = n{node}_v.x;");
                let _ = writeln!(self.body, "    let n{node}_c = n{node}_v.yzw;");
            }
            NodeKind::Checker {
                scale,
                color1,
                color2,
            } => {
                self.procedural = true;
                self.constants.push([scale, 0.0, 0.0, 0.0]);
                self.constants.push([color1.r, color1.g, color1.b, 0.0]);
                self.constants.push([color2.r, color2.g, color2.b, 0.0]);
                let a = self.input_expr(node, 0, &format!("material.values[{}].rgb", slot + 1))?;
                let b = self.input_expr(node, 1, &format!("material.values[{}].rgb", slot + 2))?;
                let s = self.input_expr(node, 2, &format!("material.values[{slot}].x"))?;
                let at = self.input_expr(node, 3, "geometry.position")?;
                let _ = writeln!(self.body, "    let n{node}_f = kr_checker({at} * {s});");
                let _ = writeln!(self.body, "    let n{node}_c = mix({b}, {a}, n{node}_f);");
            }
            NodeKind::Wave {
                kind,
                scale,
                distortion,
                detail,
                detail_scale,
            } => {
                self.procedural = true;
                self.constants
                    .push([scale, distortion, detail, detail_scale]);
                let [s, w, d, ds] = self.numbers(node, slot)?;
                // Which way the stripes run is the shape, like a Math node's
                // operation: it changes what the code does, not a number in it.
                let rings = match kind {
                    WaveKind::Bands => "false",
                    WaveKind::Rings => "true",
                };
                let at = self.input_expr(node, 4, "geometry.position")?;
                let _ = writeln!(
                    self.body,
                    "    let n{node}_f = kr_wave({at} * {s}, {rings}, {w}, {d}, {ds});"
                );
                let _ = writeln!(self.body, "    let n{node}_c = vec3<f32>(n{node}_f);");
            }
            NodeKind::ColorRamp {
                interpolation,
                count,
                stops,
            } => {
                // One slot a stop, in order along the ramp: the position and
                // the colour. How many and how they blend are the shape.
                let count = (count as usize).clamp(2, kreate_scene::shader::RAMP_STOPS);
                let mut used = stops[..count].to_vec();
                used.sort_by(|a, b| a.0.total_cmp(&b.0));
                for (at, c) in &used {
                    self.constants.push([*at, c.r, c.g, c.b]);
                }
                let fac = self.input_expr(node, 0, "0.5")?;
                let _ = writeln!(
                    self.body,
                    "    let n{node}_t = clamp({fac}, 0.0, 1.0);\n    var n{node} = material.values[{slot}].yzw;"
                );
                for i in 1..count {
                    let (here, before) = (slot + i, slot + i - 1);
                    let w = match interpolation {
                        RampMode::Constant => {
                            format!("select(0.0, 1.0, n{node}_t >= material.values[{here}].x)")
                        }
                        RampMode::Linear | RampMode::Ease => {
                            let linear = format!(
                                "clamp((n{node}_t - material.values[{before}].x) / max(material.values[{here}].x - material.values[{before}].x, 1e-6), 0.0, 1.0)"
                            );
                            if interpolation == RampMode::Ease {
                                format!("smoothstep(0.0, 1.0, {linear})")
                            } else {
                                linear
                            }
                        }
                    };
                    let _ = writeln!(
                        self.body,
                        "    n{node} = mix(n{node}, material.values[{here}].yzw, {w});"
                    );
                }
            }
            NodeKind::TexCoord => {
                // Nothing to compute: each output is a name the function
                // already has. A slot all the same, for the shape.
                self.constants.push([0.0; 4]);
            }
            NodeKind::Mapping {
                location,
                rotation,
                scale,
            } => {
                self.procedural = true;
                self.constants
                    .push([location[0], location[1], location[2], 0.0]);
                self.constants
                    .push([rotation[0], rotation[1], rotation[2], 0.0]);
                self.constants.push([scale[0], scale[1], scale[2], 0.0]);
                let v = self.input_expr(node, 0, "geometry.position")?;
                let _ = writeln!(
                    self.body,
                    "    let n{node} = kr_rotate({v} * material.values[{}].xyz, radians(material.values[{}].xyz)) + material.values[{slot}].xyz;",
                    slot + 2,
                    slot + 1
                );
            }
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
                // Two slots: the colour and the metallic share one, the
                // roughness has the next. Always both, so that the layout is
                // the graph's shape — see the comment where they are read.
                self.constants
                    .push([base_color.r, base_color.g, base_color.b, metallic]);
                self.constants
                    .push([roughness, emission.r, emission.g, emission.b]);
                self.constants.push([transmission, ior, alpha, clip]);
            }
        }
        self.emitted[node] = Some(slot);
        Ok(())
    }

    /// A node's four scalar inputs, each a wire or its slot's channel.
    fn numbers(&self, node: usize, slot: usize) -> Result<[String; 4], CompileError> {
        let mut out: [String; 4] = Default::default();
        for (socket, channel) in ["x", "y", "z", "w"].into_iter().enumerate() {
            out[socket] =
                self.input_expr(node, socket, &format!("material.values[{slot}].{channel}"))?;
        }
        Ok(out)
    }

    /// Whether anything reads an output.
    fn feeds(&self, node: usize, socket: usize) -> bool {
        self.graph
            .links
            .iter()
            .any(|l| l.from_node == node && l.from_socket == socket)
    }

    /// The surface gradient a normal input carries: what every bump up its
    /// wire adds, in object space. Zero when nothing bumps it.
    fn gradient_expr(&self, node: usize, socket: usize) -> Result<String, CompileError> {
        let Some(link) = self.graph.incoming(node, socket) else {
            return Ok("vec3<f32>(0.0)".to_string());
        };
        let from = self
            .graph
            .nodes
            .get(link.from_node)
            .ok_or(CompileError::BrokenLink)?;
        Ok(match from.kind {
            NodeKind::Bump { .. } => {
                let below = self.gradient_expr(link.from_node, 3)?;
                if below == "vec3<f32>(0.0)" {
                    format!("n{}_g", link.from_node)
                } else {
                    format!("(n{}_g + {below})", link.from_node)
                }
            }
            _ => "vec3<f32>(0.0)".to_string(),
        })
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
            // A procedural texture names its number and its colour apart,
            // because only some of them compute both.
            (
                NodeKind::Noise { .. }
                | NodeKind::Voronoi { .. }
                | NodeKind::Checker { .. }
                | NodeKind::Wave { .. },
                Socket::Scalar,
            ) => format!("n{}_f", link.from_node),
            (
                NodeKind::Noise { .. }
                | NodeKind::Voronoi { .. }
                | NodeKind::Checker { .. }
                | NodeKind::Wave { .. },
                _,
            ) => format!("n{}_c", link.from_node),
            (NodeKind::TexCoord, _) => match link.from_socket {
                0 => "geometry.position".to_string(),
                1 => "vec3<f32>(uv, 0.0)".to_string(),
                _ => "geometry.normal".to_string(),
            },
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

/// What a path tracer declares once, ahead of every material it runs: the
/// two structs the material functions trade in, and the noise library.
///
/// Once rather than per material, because a module with two `struct
/// MaterialSample`s does not compile — which is also why
/// [`for_tracer`] strips them from each material.
pub fn tracer_declarations() -> String {
    format!(
        "struct MaterialGeometry {{ position: vec3<f32>, normal: vec3<f32>, position_dx: vec3<f32>, position_dy: vec3<f32> }}\nstruct MaterialSample {{ surface: Surface, normal: vec3<f32>, gradient: vec3<f32>, emission: vec3<f32>, transmission: f32, ior: f32, alpha: f32, clip: f32, occlusion: f32 }}\n{TEXEL}{PROCEDURAL}"
    )
}

/// A compiled material rewritten to run inside the path tracer, as
/// `material_sample_{index}`.
///
/// **The same code, not a second interpretation of the graph.** What the
/// viewport runs is what this returns, with its names moved: the tracer has
/// one buffer of every material's constants rather than a uniform each, and
/// one array of images per space rather than an atlas each. So each
/// `material.values[slot]` reads `values + slot` of that buffer, and each
/// image's layer is moved up by where this material's images start in its
/// space's array. The tracer gives each image a whole layer, so the
/// rectangles it writes into the constants are the unit square.
pub fn for_tracer(
    compiled: &Compiled,
    index: usize,
    values: u32,
    colour_layer: u32,
    data_layer: u32,
) -> String {
    let code = compiled.wgsl.replace(PROCEDURAL, "").replace(TEXEL, "");
    let mut out = String::new();
    for line in code.lines() {
        // The declarations the tracer makes once, and the bindings it has
        // its own of.
        if line.starts_with("struct MaterialConstants")
            || line.starts_with("struct MaterialSample")
            || line.starts_with("struct MaterialGeometry")
            || line.starts_with("@group(")
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    let out = out
        .replace(
            "fn material_sample(",
            &format!("fn material_sample_{index}("),
        )
        .replace("kr_height_", &format!("kr_height_{index}_"));
    let out = renumber(&out, "material.values[", "]", |n| {
        format!("pt_material_values[{}u]", values + n)
    });
    let out = out
        .replace(
            "textureSampleGrad(material_textures, material_sampler, ",
            "textureSampleGrad(pt_colour_images, pt_image_sampler, ",
        )
        .replace(
            "textureSampleGrad(material_data, material_sampler, ",
            "textureSampleGrad(pt_data_images, pt_image_sampler, ",
        );
    let out = relayer(&out, "pt_colour_images", colour_layer);
    relayer(&out, "pt_data_images", data_layer)
}

/// Replaces each `{open}N{close}` with `with(N)`; anything else after
/// `open` is left as it was.
fn renumber(text: &str, open: &str, close: &str, with: impl Fn(u32) -> String) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(open) {
        out.push_str(&rest[..at]);
        let after = &rest[at + open.len()..];
        match after
            .find(close)
            .and_then(|end| Some((end, after[..end].trim().parse::<u32>().ok()?)))
        {
            Some((end, n)) => {
                out.push_str(&with(n));
                rest = &after[end + close.len()..];
            }
            None => {
                out.push_str(open);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Moves the layer of every sample from `array` up by `base`: the second
/// argument after the sampler, which is a number the compiler wrote.
fn relayer(text: &str, array: &str, base: u32) -> String {
    let open = format!("textureSampleGrad({array}, pt_image_sampler, ");
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(&open) {
        out.push_str(&rest[..at + open.len()]);
        rest = &rest[at + open.len()..];
        // The coordinate, then the layer.
        let Some(comma) = rest.find(',') else { break };
        out.push_str(&rest[..comma + 1]);
        rest = &rest[comma + 1..];
        let Some(end) = rest.find(',') else { break };
        match rest[..end].trim().parse::<u32>() {
            Ok(layer) => out.push_str(&format!(" {}", base + layer)),
            Err(_) => out.push_str(&rest[..end]),
        }
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// The image coordinate of a texture coordinate, and of a step in one: wrapped,
/// and turned so that V runs up the picture.
const TEXEL: &str = "fn kr_texel(uv: vec2<f32>) -> vec2<f32> { return vec2<f32>(fract(uv.x), 1.0 - fract(uv.y)); }\nfn kr_down(d: vec2<f32>) -> vec2<f32> { return vec2<f32>(d.x, -d.y); }\n";

/// What the procedural textures are made of, written once into a material
/// that uses any.
///
/// Hashed with integers (Jarzynski and Olano's PCG3D) rather than the
/// `fract(sin(...))` every listing uses: that one depends on how a GPU rounds
/// a large sine, so the same material would be a different texture on
/// another vendor's card — and a different one again in the path tracer.
///
/// `kr_` on every name, because this lands in the same module as the shading
/// library and a pass's own functions.
const PROCEDURAL: &str = r#"
fn kr_pcg3d(seed: vec3<u32>) -> vec3<u32> {
    var v = seed * 1664525u + vec3<u32>(1013904223u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// Three numbers in [0, 1) for a lattice point.
fn kr_cell(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(kr_pcg3d(bitcast<vec3<u32>>(vec3<i32>(c)))) * (1.0 / 4294967296.0);
}

fn kr_corner(i: vec3<f32>, f: vec3<f32>, o: vec3<f32>) -> f32 {
    return dot(kr_cell(i + o) * 2.0 - vec3<f32>(1.0), f - o);
}

// Gradient noise, roughly -1 to 1.
fn kr_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - vec3<f32>(15.0)) + vec3<f32>(10.0));
    let x00 = mix(kr_corner(i, f, vec3<f32>(0.0, 0.0, 0.0)), kr_corner(i, f, vec3<f32>(1.0, 0.0, 0.0)), u.x);
    let x10 = mix(kr_corner(i, f, vec3<f32>(0.0, 1.0, 0.0)), kr_corner(i, f, vec3<f32>(1.0, 1.0, 0.0)), u.x);
    let x01 = mix(kr_corner(i, f, vec3<f32>(0.0, 0.0, 1.0)), kr_corner(i, f, vec3<f32>(1.0, 0.0, 1.0)), u.x);
    let x11 = mix(kr_corner(i, f, vec3<f32>(0.0, 1.0, 1.0)), kr_corner(i, f, vec3<f32>(1.0, 1.0, 1.0)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// Octaves of noise, zero to one. A fractional detail fades the last octave in,
// so dragging the number does not jump.
fn kr_fbm(p: vec3<f32>, detail: f32, roughness: f32) -> f32 {
    let octaves = clamp(detail, 0.0, 15.0);
    let whole = i32(floor(octaves));
    let gain = clamp(roughness, 0.0, 1.0);
    var sum = 0.0;
    var norm = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    for (var o = 0; o <= whole; o++) {
        sum += amplitude * kr_noise(p * frequency);
        norm += amplitude;
        amplitude *= gain;
        frequency *= 2.0;
    }
    var value = sum / norm;
    let part = octaves - f32(whole);
    if part > 0.0 {
        let more = (sum + amplitude * kr_noise(p * frequency)) / (norm + amplitude);
        value = mix(value, more, part);
    }
    return clamp(value * 0.5 + 0.5, 0.0, 1.0);
}

// The position pushed by noise of its own: Blender's distortion.
fn kr_distort(p: vec3<f32>, amount: f32, detail: f32, roughness: f32) -> vec3<f32> {
    if amount == 0.0 {
        return p;
    }
    let d = vec3<f32>(
        kr_fbm(p + vec3<f32>(13.5, 0.0, 0.0), detail, roughness),
        kr_fbm(p + vec3<f32>(0.0, 13.5, 0.0), detail, roughness),
        kr_fbm(p + vec3<f32>(0.0, 0.0, 13.5), detail, roughness),
    );
    return p + (d * 2.0 - vec3<f32>(1.0)) * amount;
}

// The distance to the nearest scattered point, and that point's cell's colour.
fn kr_voronoi(p: vec3<f32>, randomness: f32) -> vec4<f32> {
    let i = floor(p);
    var best = 1e9;
    var cell = i;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let c = i + vec3<f32>(f32(x), f32(y), f32(z));
                let point = c + kr_cell(c) * randomness;
                let d = distance(p, point);
                if d < best {
                    best = d;
                    cell = c;
                }
            }
        }
    }
    return vec4<f32>(best, kr_cell(cell + vec3<f32>(71.0, 37.0, 11.0)));
}

// One where the cube the point is in is even, zero where odd. Nudged off the
// integers so a face lying exactly on one does not flicker between the two.
fn kr_checker(p: vec3<f32>) -> f32 {
    let c = floor(p * 0.999999 + vec3<f32>(1e-6));
    let s = c.x + c.y + c.z;
    return select(0.0, 1.0, s - 2.0 * floor(s * 0.5) < 0.5);
}

// Rotated about X, then Y, then Z, by angles in radians: Blender's XYZ Euler.
fn kr_rotate(p: vec3<f32>, r: vec3<f32>) -> vec3<f32> {
    let c = cos(r);
    let s = sin(r);
    let x = vec3<f32>(p.x, c.x * p.y - s.x * p.z, s.x * p.y + c.x * p.z);
    let y = vec3<f32>(c.y * x.x + s.y * x.z, x.y, -s.y * x.x + c.y * x.z);
    return vec3<f32>(c.z * y.x - s.z * y.y, s.z * y.x + c.z * y.y, y.z);
}

// Blender's sine profile, along the diagonal or out from the origin.
fn kr_wave(p: vec3<f32>, rings: bool, distortion: f32, detail: f32, detail_scale: f32) -> f32 {
    var n = (p.x + p.y + p.z) * 10.0;
    if rings {
        n = length(p) * 20.0;
    }
    if distortion != 0.0 {
        n += distortion * (kr_fbm(p * detail_scale, detail, 0.5) * 2.0 - 1.0);
    }
    return 0.5 + 0.5 * sin(n - 1.5707963);
}
"#;

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
        assert!(glowing
            .wgsl
            .contains("out.emission = material.values[1].yzw;"));
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

    /// Parses and validates, against the stand-in `Surface`.
    fn validates(compiled: &Compiled) {
        let source = format!(
            "struct Surface {{ base_color: vec3<f32>, metallic: f32, roughness: f32 }}\n{}",
            compiled.wgsl
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    }

    /// Every procedural texture, each output wired somewhere: noise's colour
    /// mixed with a checker by a voronoi's distance into the base colour, a
    /// wave's number into the roughness and the noise's into the metallic.
    fn procedural(wave: WaveKind, noise_scale: f32) -> ShaderGraph {
        let mut g = flat();
        let add = |g: &mut ShaderGraph, kind| g.add_node(Node::new("n", kind)).expect("added");
        let noise = add(
            &mut g,
            NodeKind::Noise {
                scale: noise_scale,
                detail: 2.5,
                roughness: 0.5,
                distortion: 0.3,
            },
        );
        let voronoi = add(&mut g, NodeKind::new_of("voronoi").expect("a kind"));
        let checker = add(&mut g, NodeKind::new_of("checker").expect("a kind"));
        let wave = add(
            &mut g,
            NodeKind::Wave {
                kind: wave,
                scale: 2.0,
                distortion: 3.0,
                detail: 2.0,
                detail_scale: 1.0,
            },
        );
        let mix = add(&mut g, NodeKind::MixColor { factor: 0.5 });
        for (from, fs, to, ts) in [
            (voronoi, 0, mix, 0),
            (noise, 1, mix, 1),
            (checker, 0, mix, 2),
            (voronoi, 1, checker, 0),
            (mix, 0, 0, 0),
            (wave, 1, 0, 2),
            (noise, 0, 0, 1),
            (checker, 1, wave, 0),
        ] {
            g.link(Link {
                from_node: from,
                from_socket: fs,
                to_node: to,
                to_socket: ts,
            })
            .unwrap_or_else(|e| panic!("{from}->{to}: {e:?}"));
        }
        g
    }

    #[test]
    fn procedural_textures_compile_and_type_check() {
        for kind in WaveKind::ALL {
            let compiled = compile(&procedural(kind, 5.0)).expect("compiles");
            validates(&compiled);
            assert_eq!(compiled.wgsl.matches("fn kr_fbm").count(), 1);
            assert!(compiled.wgsl.contains("geometry.position"));
        }
        // Each unwired, too: the fallbacks are expressions of their own.
        for tag in ["noise", "voronoi", "checker", "wave"] {
            let mut g = flat();
            let node = g
                .add_node(Node::new("n", NodeKind::new_of(tag).expect("a kind")))
                .expect("added");
            let colour = node_output(&g, node, Socket::Color);
            g.link(Link {
                from_node: node,
                from_socket: colour,
                to_node: 0,
                to_socket: 0,
            })
            .expect("a colour into the base colour");
            validates(&compile(&g).expect("compiles"));
        }
    }

    fn node_output(g: &ShaderGraph, node: usize, socket: Socket) -> usize {
        g.nodes[node]
            .kind
            .outputs()
            .iter()
            .position(|s| *s == socket)
            .expect("the socket")
    }

    #[test]
    fn a_textures_numbers_are_constants_and_its_kind_is_the_shape() {
        // A scale typed into a panel must not recompile; bands to rings must,
        // because it is a different function.
        let a = compile(&procedural(WaveKind::Bands, 5.0)).expect("compiles");
        let b = compile(&procedural(WaveKind::Bands, 9.0)).expect("compiles");
        let c = compile(&procedural(WaveKind::Rings, 5.0)).expect("compiles");
        assert_eq!(a.wgsl, b.wgsl);
        assert_ne!(a.constants, b.constants);
        assert_ne!(a.wgsl, c.wgsl);
    }

    #[test]
    fn a_box_projected_image_samples_three_ways_and_type_checks() {
        let mut g = flat();
        let image = g
            .add_node(Node::new(
                "Tiles",
                NodeKind::Image {
                    image: Some(an_image()),
                    projection: Projection::Box,
                    blend: 0.3,
                    scale: 2.0,
                },
            ))
            .expect("added");
        g.link(Link {
            from_node: image,
            from_socket: 0,
            to_node: 0,
            to_socket: 0,
        })
        .expect("a colour into the base colour");
        let compiled = compile(&g).expect("compiles");
        validates(&compiled);
        // One image, sampled down each axis.
        assert_eq!(compiled.images.len(), 1);
        assert_eq!(compiled.wgsl.matches("textureSampleGrad").count(), 3);
        // The rectangle's slot is still the one the renderer fills, and the
        // blend and scale sit after it.
        let rect = compiled.image_slots[0];
        assert_eq!(compiled.constants[rect + 1][..2], [0.3, 2.0]);
    }

    #[test]
    fn a_bump_reads_its_height_three_times_and_chains_with_a_normal_map() {
        let mut g = flat();
        let add = |g: &mut ShaderGraph, kind| g.add_node(Node::new("n", kind)).expect("added");
        let noise = add(&mut g, NodeKind::new_of("noise").expect("a kind"));
        let bump = add(&mut g, NodeKind::new_of("bump").expect("a kind"));
        let image = add(&mut g, NodeKind::image(Some(an_image())));
        let map = add(&mut g, NodeKind::NormalMap { strength: 1.0 });
        let second = add(&mut g, NodeKind::new_of("bump").expect("a kind"));
        let wave = add(&mut g, NodeKind::new_of("wave").expect("a kind"));
        for (from, fs, to, ts) in [
            (noise, 0, bump, 2),
            (image, 0, map, 0),
            (map, 0, bump, 3),
            (wave, 1, second, 2),
            (bump, 0, second, 3),
            (second, 0, 0, 3),
            // The same noise also colours the surface: it is in the body and
            // in the height function both.
            (noise, 1, 0, 0),
        ] {
            g.link(Link {
                from_node: from,
                from_socket: fs,
                to_node: to,
                to_socket: ts,
            })
            .unwrap_or_else(|e| panic!("{from}->{to}: {e:?}"));
        }
        let compiled = compile(&g).expect("compiles");
        validates(&compiled);
        assert_eq!(
            compiled.wgsl.matches(&format!("kr_height_{bump}(")).count(),
            4
        );
        assert!(compiled
            .wgsl
            .contains(&format!("out.gradient = (n{second}_g + n{bump}_g);")));
        // The normal map's tilt goes through both bumps to the surface.
        assert!(compiled.wgsl.contains(&format!("let n{bump} = n{map};")));
    }

    #[test]
    fn without_a_bump_the_gradient_is_zero() {
        let compiled = compile(&flat()).expect("compiles");
        assert!(compiled.wgsl.contains("out.gradient = vec3<f32>(0.0);"));
        // A bump with no height tilts nothing, and needs no function.
        let mut g = flat();
        let bump = g
            .add_node(Node::new("b", NodeKind::new_of("bump").expect("a kind")))
            .expect("added");
        g.link(Link {
            from_node: bump,
            from_socket: 0,
            to_node: 0,
            to_socket: 3,
        })
        .expect("a normal into the normal");
        let compiled = compile(&g).expect("compiles");
        validates(&compiled);
        assert!(!compiled.wgsl.contains("kr_height_"));
    }

    #[test]
    fn materials_rewritten_for_the_tracer_live_in_one_module() {
        // Two materials of every kind of node, side by side in one module
        // against the tracer's bindings: no declaration twice, no name the
        // viewport's bindings own, and each image moved to its layer.
        let mut bumped = procedural(WaveKind::Rings, 3.0);
        let bump = bumped
            .add_node(Node::new("b", NodeKind::new_of("bump").expect("a kind")))
            .expect("added");
        let wave = bumped
            .add_node(Node::new("w", NodeKind::new_of("wave").expect("a kind")))
            .expect("added");
        for (from, fs, to, ts) in [(wave, 1, bump, 2), (bump, 0, 0, 3)] {
            bumped
                .link(Link {
                    from_node: from,
                    from_socket: fs,
                    to_node: to,
                    to_socket: ts,
                })
                .expect("connects");
        }
        let (mapped, ids) = crate::map_tests::mapped();
        let first = compile(&bumped).expect("compiles");
        let second = compile_with(&mapped, crate::map_tests::spaces(ids)).expect("compiles");
        let a = for_tracer(&first, 0, 0, 0, 0);
        let b = for_tracer(&second, 1, first.constants.len() as u32, 5, 7);
        assert!(!a.contains("material.values") && !b.contains("material.values"));
        assert!(!b.contains("material_textures") && !b.contains("material_data"));
        assert!(b.contains("pt_colour_images, pt_image_sampler"));
        assert!(b.contains("pt_data_images, pt_image_sampler"));
        // The data images are the second material's first two of their kind:
        // layers seven and eight.
        assert!(b.contains(", 7, ") && b.contains(", 8, "), "{b}");
        let source = format!(
            "struct Surface {{ base_color: vec3<f32>, metallic: f32, roughness: f32 }}
@group(0) @binding(0) var<storage, read> pt_material_values: array<vec4<f32>>;
@group(0) @binding(1) var pt_colour_images: texture_2d_array<f32>;
@group(0) @binding(2) var pt_data_images: texture_2d_array<f32>;
@group(0) @binding(3) var pt_image_sampler: sampler;
{}
{a}
{b}",
            tracer_declarations()
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    }

    #[test]
    fn a_ramp_and_a_mapped_coordinate_compile_and_type_check() {
        // UV through a mapping into a wave, the wave's number through a
        // ramp into the base colour — each ramp mode — and the object's
        // normal straight into a checker.
        for mode in RampMode::ALL {
            let mut g = flat();
            let add = |g: &mut ShaderGraph, kind| g.add_node(Node::new("n", kind)).expect("added");
            let coords = add(&mut g, NodeKind::TexCoord);
            let mapping = add(&mut g, NodeKind::new_of("mapping").expect("a kind"));
            let wave = add(&mut g, NodeKind::new_of("wave").expect("a kind"));
            let ramp = add(
                &mut g,
                NodeKind::ColorRamp {
                    interpolation: mode,
                    count: 3,
                    stops: [(0.5, SceneRgb::new(1.0, 0.5, 0.0)); kreate_scene::shader::RAMP_STOPS],
                },
            );
            let checker = add(&mut g, NodeKind::new_of("checker").expect("a kind"));
            for (from, fs, to, ts) in [
                (coords, 1, mapping, 0),
                (mapping, 0, wave, 4),
                (wave, 1, ramp, 0),
                (ramp, 0, 0, 0),
                (coords, 2, checker, 3),
                (checker, 1, 0, 2),
            ] {
                g.link(Link {
                    from_node: from,
                    from_socket: fs,
                    to_node: to,
                    to_socket: ts,
                })
                .unwrap_or_else(|e| panic!("{from}->{to}: {e:?}"));
            }
            let compiled = compile(&g).expect("compiles");
            validates(&compiled);
            assert!(compiled.wgsl.contains("kr_wave(n"), "{}", compiled.wgsl);
            assert!(compiled.wgsl.contains("vec3<f32>(uv, 0.0)"));
            assert!(compiled.wgsl.contains("kr_checker(geometry.normal"));
        }
        // A normal map's direction is not a place to read a texture at.
        let mut g = flat();
        let map = g
            .add_node(Node::new("m", NodeKind::NormalMap { strength: 1.0 }))
            .expect("added");
        let noise = g
            .add_node(Node::new("n", NodeKind::new_of("noise").expect("a kind")))
            .expect("added");
        assert!(g
            .link(Link {
                from_node: map,
                from_socket: 0,
                to_node: noise,
                to_socket: 4,
            })
            .is_err());
    }

    #[test]
    fn a_ramps_stops_are_read_in_order_whatever_order_they_were_given() {
        let mut stops = [(0.0, SceneRgb::new(0.0, 0.0, 0.0)); kreate_scene::shader::RAMP_STOPS];
        stops[0] = (0.9, SceneRgb::new(1.0, 0.0, 0.0));
        stops[1] = (0.1, SceneRgb::new(0.0, 0.0, 1.0));
        let mut g = flat();
        let ramp = g
            .add_node(Node::new(
                "r",
                NodeKind::ColorRamp {
                    interpolation: RampMode::Linear,
                    count: 2,
                    stops,
                },
            ))
            .expect("added");
        g.link(Link {
            from_node: ramp,
            from_socket: 0,
            to_node: 0,
            to_socket: 0,
        })
        .expect("connects");
        let compiled = compile(&g).expect("compiles");
        let slot = compiled
            .constants
            .iter()
            .position(|c| c[0] == 0.1)
            .expect("the blue stop");
        assert_eq!(compiled.constants[slot + 1][0], 0.9, "red after blue");
    }

    #[test]
    fn a_material_without_a_procedural_texture_carries_no_noise() {
        assert!(!compile(&flat()).expect("compiles").wgsl.contains("kr_"));
    }

    #[test]
    fn mix_math_invert_and_the_emission_socket_compile_and_type_check() {
        // Every operation, each feeding the next, the last into a mix factor;
        // an inverted colour into the glow. Validated, not only parsed: a
        // vec3 where an f32 belongs parses and is still wrong.
        for op in MathOp::ALL {
            let mut g = flat();
            let math = g
                .add_node(Node::new("M", NodeKind::Math { op, a: 0.3, b: 2.0 }))
                .unwrap();
            let mix = g
                .add_node(Node::new("X", NodeKind::MixColor { factor: 0.5 }))
                .unwrap();
            let colour = g
                .add_node(Node::new(
                    "C",
                    NodeKind::Color(SceneRgb::new(0.2, 0.4, 0.6)),
                ))
                .unwrap();
            let invert = g.add_node(Node::new("I", NodeKind::Invert)).unwrap();
            for (from, fs, to, ts) in [
                (math, 0, mix, 0),
                (colour, 0, mix, 1),
                (mix, 0, 0, 0),
                (colour, 0, invert, 0),
                (invert, 0, 0, 4),
            ] {
                g.link(Link {
                    from_node: from,
                    from_socket: fs,
                    to_node: to,
                    to_socket: ts,
                })
                .unwrap_or_else(|e| panic!("{op:?}: {from}->{to}: {e:?}"));
            }
            let compiled = compile(&g).expect("compiles");
            let source = format!(
                "struct Surface {{ base_color: vec3<f32>, metallic: f32, roughness: f32 }}
{}",
                compiled.wgsl
            );
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{op:?}: {}", e.emit_to_string(&source)));
            assert!(compiled.wgsl.contains("mix("), "{}", compiled.wgsl);
            assert!(compiled.wgsl.contains("].yzw * n"), "{}", compiled.wgsl);
        }
    }

    #[test]
    fn the_math_node_keeps_its_two_numbers_and_the_operation_is_the_shape() {
        // Wired into the roughness: a node nothing reads is not compiled.
        let with = |op, a, b| {
            let mut g = flat();
            let m = g
                .add_node(Node::new("M", NodeKind::Math { op, a, b }))
                .unwrap();
            g.link(Link {
                from_node: m,
                from_socket: 0,
                to_node: 0,
                to_socket: 2,
            })
            .unwrap();
            g
        };
        let a = with(MathOp::Add, 1.0, 2.0);
        let b = with(MathOp::Add, 7.0, 9.0);
        let c = with(MathOp::Power, 1.0, 2.0);
        let (a, b, c) = (
            compile(&a).unwrap(),
            compile(&b).unwrap(),
            compile(&c).unwrap(),
        );
        // The numbers are constants, so they share a pipeline; the operation
        // is code, so it does not.
        assert_eq!(a.wgsl, b.wgsl);
        assert_ne!(a.wgsl, c.wgsl);
        assert!(b.constants.contains(&[7.0, 9.0, 0.0, 0.0]));
    }

    #[test]
    fn a_flat_material_compiles_to_its_three_constants() {
        let compiled = compile(&flat()).expect("compiles");
        parses(&compiled);
        assert_eq!(
            compiled.constants,
            vec![
                [0.8, 0.7, 0.6, 0.25],
                [0.4, 0.0, 0.0, 0.0],
                // Transmission, IOR, opacity, and no cut-out.
                [0.0, 1.5, 1.0, 0.0]
            ]
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
        assert_eq!(
            a.wgsl, b.wgsl,
            "the same graph shape produced different code"
        );
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
            .add_node(Node::new(
                "Red",
                NodeKind::Color(SceneRgb::new(1.0, 0.0, 0.0)),
            ))
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
        assert!(compiled
            .wgsl
            .contains("s.metallic = clamp(material.values["));
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
            .add_node(Node::new("Albedo", NodeKind::image(Some(an_image()))))
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
        assert!(compiled
            .wgsl
            .contains(&format!("s.base_color = n{image}_rgba.rgb;")));
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
            .add_node(Node::new("Albedo", NodeKind::image(None)))
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
            .add_node(Node::new(
                "Multiply",
                kreate_scene::shader::NodeKind::Multiply,
            ))
            .expect("not a surface");
        graph
            .link(Link {
                from_node: node,
                from_socket: 0,
                to_node: 0,
                to_socket: 0,
            })
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
        let one = doc.add_image(kreate_scene::Image::new(
            "A",
            4,
            4,
            kreate_scene::PixelFormat::Rgba8,
        ));
        let two = doc.add_image(kreate_scene::Image::new(
            "B",
            8,
            8,
            kreate_scene::PixelFormat::Rgba8,
        ));
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
        assert_ne!(
            a.constants, b.constants,
            "the colours did not reach the numbers"
        );
    }
}

#[cfg(test)]
mod map_tests {
    use super::*;
    use kreate_math::SceneRgb;

    /// A material with all three maps, each its own image.
    pub(crate) fn mapped() -> (ShaderGraph, [ImageId; 3]) {
        let mut doc = kreate_scene::Document::new();
        let mut add = |name: &str| {
            doc.add_image(kreate_scene::Image::new(
                name,
                4,
                4,
                kreate_scene::PixelFormat::Rgba8,
            ))
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
        (
            ShaderGraph::for_material(&material),
            [colour, packed, normal],
        )
    }

    /// The spaces a document would answer with: the colour map sRGB, the
    /// others raw.
    pub(crate) fn spaces(ids: [ImageId; 3]) -> impl Fn(ImageId) -> ColorSpace {
        move |id| {
            if id == ids[0] {
                ColorSpace::Srgb
            } else {
                ColorSpace::Raw
            }
        }
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
        assert_eq!(
            compiled
                .wgsl
                .matches("textureSampleGrad(material_textures")
                .count(),
            1
        );
        assert_eq!(
            compiled
                .wgsl
                .matches("textureSampleGrad(material_data")
                .count(),
            2
        );
        assert!(compiled
            .wgsl
            .contains("material_textures, material_sampler, n"));
        // Two data images: layers 0 and 1 of the data array, not 1 and 2.
        let data_layers: Vec<&str> = compiled
            .wgsl
            .lines()
            .filter(|l| l.contains("textureSampleGrad(material_data"))
            .map(|l| {
                l.split(".zw, ")
                    .nth(1)
                    .and_then(|r| r.split(',').next())
                    .unwrap_or("")
            })
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
            let name = l
                .split("clamp(")
                .nth(1)
                .and_then(|r| r.split(',').next())
                .expect("a node");
            compiled
                .wgsl
                .lines()
                .find(|x| x.contains(&format!("let {name} = ")))
                .expect("the product is emitted")
                .to_string()
        };
        assert!(
            product_of("roughness").contains(".y *"),
            "{}",
            compiled.wgsl
        );
        assert!(product_of("metallic").contains(".z *"), "{}", compiled.wgsl);
    }

    #[test]
    fn a_normal_map_reaches_the_output_and_nothing_else_does() {
        let (graph, ids) = mapped();
        let compiled = compile_with(&graph, spaces(ids)).expect("compiles");
        assert!(
            compiled.wgsl.contains("* 2.0 - vec3<f32>(1.0)"),
            "the map is not decoded"
        );
        assert!(!compiled
            .wgsl
            .contains("out.normal = vec3<f32>(0.0, 0.0, 1.0);"));
        // And its strength is a constant, not text: two strengths, one shader.
        let flat = compile(&ShaderGraph::standard(
            "F",
            SceneRgb::new(1.0, 1.0, 1.0),
            0.0,
            0.5,
        ))
        .expect("compiles");
        assert!(flat.wgsl.contains("out.normal = vec3<f32>(0.0, 0.0, 1.0);"));
        assert!(
            compiled.constants.iter().any(|c| c[0] == 0.5),
            "the strength did not travel"
        );
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
    /// How much light passes through, zero to one.
    pub fn transmission(&self) -> f32 {
        self.constants
            .get(self.transmission_slot)
            .map_or(0.0, |c| c[0].clamp(0.0, 1.0))
    }

    /// Whether light passes through: drawn after the solids, over them.
    pub fn transmissive(&self) -> bool {
        self.constants
            .get(self.transmission_slot)
            .is_some_and(|c| c[0] > 0.0)
    }
}
