//! Compiling the document's materials for the viewport.
//!
//! The joining §0 leaves to the application. `kreate-render` cannot see a
//! `ShaderGraph` any more than it can see a `Document`, so a material reaches
//! it as text, numbers and pixels — and this is where the document turns into
//! those.
//!
//! # Recompiled every frame, re-uploaded almost never
//!
//! Compiling is string building and costs microseconds; uploading rebuilds a
//! texture atlas and a pipeline. So every frame compiles, and what is compared
//! is the *code*: if it is the same, only the numbers are written, which is
//! sixteen bytes a material. That is what keeps dragging a colour slider from
//! rebuilding a scene's worth of textures, and it works because the compiler
//! keeps constants out of the text on purpose.
//!
//! # Index zero is the material an object has not got
//!
//! A fresh cube has no material and must still be shaded. So the list always
//! starts with a plain grey, and an object with nothing assigned points at it.
//! Not an error and not invisible — the same judgement the renderer used to
//! make with its default `Surface`.

use kreate_scene::document::MaterialId;
use kreate_scene::shader::ShaderGraph;
use kreate_scene::Document;

/// The document's materials, compiled.
pub struct Materials {
    /// Parallel: what each material compiled to.
    pub compiled: Vec<kreate_shader::Compiled>,
    /// Where each of the document's materials landed in `compiled`.
    index: std::collections::HashMap<MaterialId, u32>,
    /// Every image the materials read, with how many times its pixels have
    /// been written: a stroke changes the second, and the atlas goes up
    /// again. See `Image::revision`.
    revisions: Vec<(kreate_scene::document::ImageId, u64)>,
}

impl Materials {
    /// Compiles every material in the document, plus the default at zero.
    pub fn compile(doc: &Document) -> Self {
        // The default first, so its index is zero whatever else happens.
        let mut compiled = vec![compile_one(
            doc,
            &ShaderGraph::standard(
                "Default",
                kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
                0.0,
                0.5,
            ),
        )];
        let mut index = std::collections::HashMap::new();
        for (id, material) in doc.materials() {
            index.insert(id, compiled.len() as u32);
            // The flat fields as the graph they are. A material will carry its
            // own graph when there is an editor to build one in; until then
            // this is the same model reached by a shorter route, and the route
            // lives beside the fields rather than here — one answer to "what is
            // a standard surface", which is what keeps a field added to the
            // material from having to be added twice.
            compiled.push(compile_one(doc, &material.shading_graph()));
        }
        let revisions = compiled
            .iter()
            .flat_map(|c| c.images.iter())
            .map(|id| (*id, doc.image(*id).map_or(0, |i| i.revision())))
            .collect();
        Self {
            compiled,
            index,
            revisions,
        }
    }

    /// Which material an object is drawn with.
    ///
    /// Slot zero, which is every object that has not been given per-face
    /// materials — per-face slots need the draw split by slot, which is a
    /// change to how a mesh is uploaded rather than to what is handed over
    /// here.
    pub fn index_of(&self, doc: &Document, object: kreate_scene::ObjectId) -> u32 {
        doc.material_id_at(object, 0)
            .and_then(|id| self.index.get(&id).copied())
            .unwrap_or(0)
    }

    /// Each compiled material as the path tracer's flat surface, by compiled
    /// index: base colour, metallic and roughness — the material's flat
    /// fields, which is what its graph was built from unless it has one.
    ///
    /// With the images they read, which the surfaces name by position.
    pub fn flat_surfaces(
        &self,
        doc: &Document,
    ) -> (Vec<kreate_render::pathtrace::PtSurface>, Vec<TextureImage>) {
        let default = kreate_render::pathtrace::PtSurface {
            base_color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            ..Default::default()
        };
        let mut out = vec![default; self.compiled.len()];
        let mut images: Vec<TextureImage> = Vec::new();
        let mut take = |id: Option<kreate_scene::document::ImageId>,
                        space: kreate_scene::ColorSpace|
         -> Option<usize> {
            let image = doc.image(id?)?;
            images.push(TextureImage {
                width: image.width(),
                height: image.height(),
                kind: kind_of(space),
                pixels: encoded_rgba(image, space),
            });
            Some(images.len() - 1)
        };
        // In a fixed order, so the same document hands over the same list.
        let mut ids: Vec<_> = self.index.iter().collect();
        ids.sort_by_key(|(_, index)| **index);
        for (id, index) in ids {
            if let (Some(m), Some(slot)) = (doc.material(*id), out.get_mut(*index as usize)) {
                *slot = kreate_render::pathtrace::PtSurface {
                    base_color: [m.base_color.r, m.base_color.g, m.base_color.b],
                    metallic: m.metallic,
                    roughness: m.roughness,
                    base_color_image: take(m.base_color_image, kreate_scene::ColorSpace::Srgb),
                    metallic_roughness_image: take(
                        m.metallic_roughness_image,
                        kreate_scene::ColorSpace::Raw,
                    ),
                    emission: graph_emission(m),
                    transmission: graph_glass(m).0,
                    ior: graph_glass(m).1,
                    // A material glows both ways, as the rasterised view
                    // draws it; only a light's panel is one-sided.
                    one_sided: false,
                };
            }
        }
        (out, images)
    }

    /// One compiled material per mesh slot, in slot order.
    ///
    /// **The mapping the renderer's per-slot draws need.** A face names a
    /// slot, a slot names a material, and the viewport speaks in compiled
    /// indices — this is that chain resolved once per object per frame, which
    /// is where it is cheap: it changes when the document does, not when the
    /// camera moves.
    ///
    /// Through `material_id_at`, so the object's override of a slot wins over
    /// the mesh's own the same way it does everywhere else. A slot with
    /// nothing in it resolves to the default grey at zero rather than to
    /// nothing: an unassigned face is not an error, and a hole in the model
    /// reads as broken geometry when what is missing is an assignment.
    pub fn slots_of(&self, doc: &Document, object: kreate_scene::ObjectId) -> Vec<u32> {
        let Some(&kreate_scene::document::ObjectData::Mesh(mesh)) =
            doc.object(object).map(|o| &o.data)
        else {
            return Vec::new();
        };
        (0..doc.mesh_slots(mesh).len())
            .map(|slot| {
                doc.material_id_at(object, slot)
                    .and_then(|id| self.index.get(&id).copied())
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Whether these compiled to the same code as `other`.
    ///
    /// The whole list, because a material inserted in the middle moves every
    /// index after it — and an index that moved without the viewport being told
    /// is an object wearing somebody else's material.
    pub fn same_code_as(&self, other: &Materials) -> bool {
        self.revisions == other.revisions
            && self.compiled.len() == other.compiled.len()
            && self
                .compiled
                .iter()
                .zip(&other.compiled)
                .all(|(a, b)| a.wgsl == b.wgsl)
    }
}

/// The pixels one compiled material's images resolve to, parallel to
/// [`kreate_shader::Compiled::images`].
///
/// §0's seam: the compiler hands back image *identities*, which belong to the
/// document, and `kreate-render` is given what they resolve to. This is the
/// joining, and it is the whole of what the application does here.
///
/// **Parallel, even when an image is missing.** A material naming an image the
/// document no longer has gets one white texel in its place. Dropping it would
/// shift every later image down a layer, and the material after the gap would
/// be sampling its neighbour's texture.
///
/// **Colours encoded, not decoded.** The colour atlas is an sRGB format and the
/// sampler applies the transfer function, so what goes over is the file's own
/// bytes — see `kreate_render::material::SourceImage::pixels` for why decoding
/// here would cost the darks. Data goes over as the numbers it is.
pub fn pixels_for(doc: &Document, compiled: &kreate_shader::Compiled) -> Vec<TextureImage> {
    compiled
        .images
        .iter()
        .zip(&compiled.image_spaces)
        .map(|(id, space)| {
            let kind = kind_of(*space);
            match doc.image(*id) {
                Some(image) => TextureImage {
                    width: image.width(),
                    height: image.height(),
                    kind,
                    pixels: encoded_rgba(image, *space),
                },
                None => TextureImage {
                    width: 1,
                    height: 1,
                    kind,
                    pixels: vec![255; 4],
                },
            }
        })
        .collect()
}

/// Which atlas an image in this space belongs in.
pub fn kind_of(space: kreate_scene::ColorSpace) -> kreate_render::material::TextureKind {
    match space {
        kreate_scene::ColorSpace::Srgb => kreate_render::material::TextureKind::Color,
        kreate_scene::ColorSpace::Raw => kreate_render::material::TextureKind::Data,
    }
}

/// One image's bytes, untiled into a row-major `Rgba8` block.
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub kind: kreate_render::material::TextureKind,
    pub pixels: Vec<u8>,
}

impl TextureImage {
    /// Borrowed as the renderer takes it.
    pub fn source(&self) -> kreate_render::material::SourceImage<'_> {
        kreate_render::material::SourceImage {
            width: self.width,
            height: self.height,
            kind: self.kind,
            pixels: &self.pixels,
        }
    }
}

/// Walks the tiling back out into the rows a texture upload wants.
///
/// The same walk an export does, so it lives beside the image loader: one
/// answer to "what are this image's bytes". A tile that is not resident reads
/// as transparent black — a partly loaded image is a real state (§3.6) and half
/// a texture beats a failure.
fn encoded_rgba(image: &kreate_scene::Image, space: kreate_scene::ColorSpace) -> Vec<u8> {
    kreate_io::image_import::rgba8_of(image, space)
}

fn compile_one(doc: &Document, graph: &ShaderGraph) -> kreate_shader::Compiled {
    // A graph built from flat fields cannot fail to compile. One somebody
    // edited can — the operators keep its surface, but a file can say
    // anything — and a material that fails draws as the plain grey rather
    // than taking the frame down with an `expect`. Grey and not invisible:
    // a hole in the model reads as broken geometry, and what is broken is a
    // material.
    //
    // Each image's space is the datablock's own declaration, made when it was
    // loaded into the slot that wanted it. An image the document has lost is
    // a colour, which is also what the white stand-in `pixels_for` sends is.
    kreate_shader::compile_with(graph, |id| {
        doc.image(id)
            .map(|image| image.color_space)
            .unwrap_or(kreate_scene::ColorSpace::Srgb)
    })
    .unwrap_or_else(|_| {
        kreate_shader::compile(&ShaderGraph::standard(
            "Default",
            kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
            0.0,
            0.5,
        ))
        .expect("a standard surface always compiles")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::document::{Material, Object, ObjectData};

    fn doc_with_materials(n: usize) -> Document {
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
        let mut slots = Vec::new();
        for i in 0..n {
            slots.push(Some(doc.add_material(Material {
                name: format!("M{i}"),
                base_color: kreate_math::SceneRgb::new(i as f32 / 10.0, 0.5, 0.5),
                ..Default::default()
            })));
        }
        doc.set_mesh_slots(mesh, slots);
        doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        doc
    }

    #[test]
    fn an_object_with_no_material_takes_the_one_at_zero() {
        // A fresh cube has no material and must still be shaded. The index it
        // gets has to be a real one, or the draw is skipped and the object
        // vanishes.
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
        let cube = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        let materials = Materials::compile(&doc);
        assert_eq!(materials.index_of(&doc, cube), 0);
        assert_eq!(materials.compiled.len(), 1, "only the default");
    }

    #[test]
    fn every_material_gets_its_own_numbers_and_they_share_one_shader() {
        // The property the pipeline cache is built on, checked where the
        // document meets it: ten materials, ten sets of constants, one piece of
        // code. If the code differed, a scene of tinted plastics would compile
        // a shader apiece.
        let doc = doc_with_materials(10);
        let materials = Materials::compile(&doc);
        assert_eq!(materials.compiled.len(), 11, "ten and the default");
        let first = &materials.compiled[0].wgsl;
        assert!(materials.compiled.iter().all(|c| &c.wgsl == first));
        // And the numbers really do differ, or the claim above is about
        // nothing.
        assert_ne!(
            materials.compiled[1].constants,
            materials.compiled[2].constants
        );
    }

    #[test]
    fn an_objects_material_is_the_one_in_its_first_slot() {
        let doc = doc_with_materials(3);
        let (object, _) = doc.objects().next().expect("the cube");
        let materials = Materials::compile(&doc);
        let index = materials.index_of(&doc, object);
        assert_ne!(index, 0, "the cube took the default");
        // And it carries that material's colour rather than another's.
        let (_, first) = doc.materials().next().expect("a material");
        let constants = &materials.compiled[index as usize].constants[0];
        assert_eq!(constants[0], first.base_color.r);
    }

    #[test]
    fn changing_a_colour_does_not_change_the_code() {
        // What decides between writing sixteen bytes and rebuilding a scene's
        // textures. If this ever stops holding, dragging a colour slider
        // rebuilds every material's atlas once a frame.
        let mut doc = doc_with_materials(3);
        let before = Materials::compile(&doc);
        let (id, _) = doc.materials().next().expect("a material");
        doc.material_mut(id).expect("it is there").base_color =
            kreate_math::SceneRgb::new(1.0, 0.0, 0.0);
        let after = Materials::compile(&doc);
        assert!(before.same_code_as(&after));
        assert_ne!(before.compiled[1].constants, after.compiled[1].constants);
    }

    #[test]
    fn adding_a_material_does_change_the_list() {
        // The other half: a material added in the middle moves every index
        // after it, and an index that moved without the viewport being told is
        // an object wearing somebody else's material.
        let doc = doc_with_materials(3);
        let before = Materials::compile(&doc);
        let doc = doc_with_materials(4);
        let after = Materials::compile(&doc);
        assert!(!before.same_code_as(&after));
    }
}

#[cfg(test)]
mod texture_tests {
    use super::*;
    use kreate_scene::document::Material;
    use kreate_scene::{Image, PixelFormat};

    /// An image whose every texel says where it is, so a walk that loses the
    /// tiling shows up as a value in the wrong place rather than as a wash.
    fn numbered(width: u32, height: u32) -> Image {
        let mut image = Image::new("N", width, height, PixelFormat::Rgba8);
        let (across, down) = image.tile_counts();
        let tile = kreate_scene::image::TILE;
        for ty in 0..down {
            for tx in 0..across {
                let Some(held) = image.tile_mut(tx, ty) else {
                    continue;
                };
                let bytes = held.bytes_mut();
                for row in 0..tile {
                    for column in 0..tile {
                        let (x, y) = (tx * tile + column, ty * tile + row);
                        let at = (row * tile + column) as usize * 4;
                        bytes[at] = (x % 256) as u8;
                        bytes[at + 1] = (y % 256) as u8;
                        bytes[at + 2] = 7;
                        bytes[at + 3] = 255;
                    }
                }
            }
        }
        image
    }

    fn doc_with_texture(image: Image) -> (Document, kreate_scene::MaterialId) {
        let mut doc = Document::new();
        let id = doc.add_image(image);
        let material = doc.add_material(Material {
            name: "Textured".into(),
            base_color_image: Some(id),
            ..Default::default()
        });
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
        doc.set_mesh_slots(mesh, vec![Some(material)]);
        doc.add_object(kreate_scene::document::Object::new(
            "Cube",
            kreate_scene::document::ObjectData::Mesh(mesh),
        ));
        (doc, material)
    }

    #[test]
    fn the_tiling_comes_back_out_as_rows() {
        // A size that is not a multiple of the tile, so the walk has to stop at
        // the image's edge on both axes rather than at the tile's. §7.9 found
        // exactly this: the right edge had a test and the bottom did not.
        let (doc, _) = doc_with_texture(numbered(200, 150));
        let materials = Materials::compile(&doc);
        let sets = pixels_for(&doc, &materials.compiled[1]);
        assert_eq!(sets.len(), 1, "the texture did not reach the upload");
        let image = &sets[0];
        assert_eq!((image.width, image.height), (200, 150));
        assert_eq!(image.pixels.len(), 200 * 150 * 4);
        for (x, y) in [(0u32, 0u32), (199, 0), (0, 149), (199, 149), (130, 130)] {
            let at = (y as usize * 200 + x as usize) * 4;
            assert_eq!(
                &image.pixels[at..at + 4],
                &[(x % 256) as u8, (y % 256) as u8, 7, 255],
                "the texel at {x},{y} came from somewhere else"
            );
        }
    }

    #[test]
    fn eight_bit_pixels_go_over_as_they_were_written() {
        // **Encoded, not decoded.** The atlas is an sRGB format, so a byte of
        // 128 has to arrive as 128 and be turned into light by the sampler.
        // Decoding here would send 55 and the picture would be dark — and every
        // test that compares two halves of a frame would still pass.
        let mut image = Image::new("Grey", 4, 4, PixelFormat::Rgba8);
        image.tile_mut(0, 0).expect("one tile").bytes_mut()[..4]
            .copy_from_slice(&[128, 64, 32, 255]);
        let (doc, _) = doc_with_texture(image);
        let materials = Materials::compile(&doc);
        let sets = pixels_for(&doc, &materials.compiled[1]);
        assert_eq!(&sets[0].pixels[..4], &[128, 64, 32, 255]);
    }

    #[test]
    fn a_material_with_no_texture_asks_for_no_pixels() {
        let mut doc = Document::new();
        doc.add_material(Material::default());
        let materials = Materials::compile(&doc);
        assert!(pixels_for(&doc, &materials.compiled[1]).is_empty());
        // And the default at zero never had one either.
        assert!(pixels_for(&doc, &materials.compiled[0]).is_empty());
    }

    #[test]
    fn a_single_channel_image_arrives_opaque() {
        // A mask replicated across three channels. A zero alpha would make it
        // vanish, which is the failure that looks like the texture not loading.
        let mut image = Image::new("Mask", 4, 4, PixelFormat::R8);
        image.tile_mut(0, 0).expect("one tile").bytes_mut()[0] = 90;
        let (doc, _) = doc_with_texture(image);
        let materials = Materials::compile(&doc);
        let sets = pixels_for(&doc, &materials.compiled[1]);
        assert_eq!(&sets[0].pixels[..4], &[90, 90, 90, 255]);
    }
}

/// The light a material emits: its graph's surface when it has one, which is
/// what the rasterised view draws, and its own fields otherwise.
fn graph_emission(m: &kreate_scene::Material) -> [f32; 3] {
    let e = m
        .graph
        .as_ref()
        .and_then(|g| g.output().map(|o| &g.nodes[o].kind))
        .and_then(|k| match k {
            kreate_scene::shader::NodeKind::Surface { emission, .. } => Some(*emission),
            _ => None,
        })
        .unwrap_or_else(|| m.emitted());
    [e.r, e.g, e.b]
}

/// How much a material lets through and how it bends, from where the
/// rasterised view takes them: its graph's surface, or its own fields.
fn graph_glass(m: &kreate_scene::Material) -> (f32, f32) {
    m.graph
        .as_ref()
        .and_then(|g| g.output().map(|o| &g.nodes[o].kind))
        .and_then(|k| match k {
            kreate_scene::shader::NodeKind::Surface {
                transmission, ior, ..
            } => Some((*transmission, *ior)),
            _ => None,
        })
        .unwrap_or((m.transmission, m.ior))
}
