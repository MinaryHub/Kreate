//! What a compiled material needs on the device.
//!
//! `kreate-shader` turns a graph into WGSL and a block of numbers; this is the
//! other side of that contract — the bind group the generated code names, and
//! the pipeline cache that makes "one graph, one compiler" affordable.
//!
//! **This crate does not depend on that one**, and the reason is the seam §0
//! draws: `kreate-render` never sees a `Document`, and a compiled material
//! carries image *identities*, which are the document's. So what arrives here
//! is what arrives everywhere else — text, numbers and pixels — and the
//! application does the joining. A dependency the other way would have been one
//! line and would have quietly given the renderer the whole scene model.
//!
//! # The shape of the graph is the cache key
//!
//! The compiler puts a material's constants in a uniform rather than in the
//! text, so two materials that differ only in their numbers produce identical
//! code. That is what lets this keep one pipeline per *shape* instead of one
//! per material, and it is why a colour slider does not stutter: dragging it
//! writes sixteen bytes.
//!
//! # Textures: an array, with a rectangle per image
//!
//! Bindless is not in wgpu's core (v1 §4.7), so a material's images cannot be
//! indexed by a number the shader computes. The choices are a binding per image
//! — which would make the bind group layout, and therefore the pipeline, depend
//! on how many images a material happens to use — or one array binding with a
//! layer per image. The array wins for exactly that reason.
//!
//! # Two atlases: colours in sRGB, numbers as they are
//!
//! The colour atlas is `Rgba8UnormSrgb` and what is uploaded is the file's own
//! bytes. The alternative — decode on the way in and upload linear eight-bit —
//! loses the darks: sRGB spends twenty code values where linear spends two,
//! and that difference is the whole reason the encoding exists.
//!
//! A roughness or a normal map is not a colour, and through that atlas it
//! would come back bent by a curve it was never encoded with — a roughness of
//! 0.5 would shade as 0.21. So it goes in the second atlas, plain `Rgba8Unorm`,
//! and which one is the image's [`TextureKind`], decided by the document and
//! never guessed here.
//!
//! # Every level of detail, made once
//!
//! Each layer carries its whole mip chain, built on the CPU at upload. Without
//! one, a brick wall a few metres away is a field of shimmering noise: every
//! pixel covers dozens of texels and the sampler reads one of them. The
//! colour chain is averaged *in light*, not in code values — the mean of black
//! and white is 0.5 of the light and 188 of the bytes, and averaging the bytes
//! darkens every high-contrast texture as it recedes.
//!
//! An image smaller than its layer is padded by repeating its last row and
//! column, not with black. Coarse levels average across the image's edge, and
//! a black margin would creep inward as a dark rim one level at a time.
//!
//! An array's layers are all one size, and a material's images are not. So each
//! image carries a rectangle in the constants and the generated code scales its
//! UV by it. Today the packer is trivial: **one image per layer, at the size of
//! the largest**, so a 64-pixel mask beside a 4K albedo occupies a 4K layer.
//! That is wasteful and it is not a lie — nothing is resampled and nothing is
//! dropped. A real packer puts several images on one layer and changes nothing
//! about the shader, because the rectangle is already how an image is found.

/// The bind group a material's own data arrives in.
///
/// Mirrors `kreate_shader::MATERIAL_GROUP`. Two constants rather than one
/// because the crates do not see each other; they are held together by the
/// contract test that composes a compiled material against this library.
pub const MATERIAL_GROUP: u32 = 3;

/// What a compiled material brings with it, as plain data.
///
/// The compiler's `Compiled` without the parts that belong to a document — its
/// image identities stay on the application's side of the seam, and what
/// reaches here is the pixels they resolve to.
#[derive(Clone, Copy, Debug)]
pub struct MaterialData<'a> {
    /// One `vec4` per slot, in the order the generated code indexes them.
    pub constants: &'a [[f32; 4]],
    /// Which slot each image's rectangle goes in, parallel to the images.
    pub image_slots: &'a [usize],
}

/// One image, as the atlas holds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub layer: u32,
    /// Where the image sits in its layer, in unit coordinates: `xy` is the
    /// offset and `zw` the size.
    pub rect: [f32; 4],
}

/// Which atlas an image belongs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureKind {
    /// sRGB-encoded colours: decoded by the sampler, averaged in light.
    Color,
    /// Numbers — roughness, metallic, normals — read back exactly as stored.
    Data,
}

/// An image on its way to the device.
///
/// Pixels and a size, because `kreate-render` does not see an
/// [`Image`](kreate_scene::image::Image) any more than it sees a `Document`.
/// The application flattens the tiles, the same way it does for an environment.
pub struct SourceImage<'a> {
    pub width: u32,
    pub height: u32,
    /// Which atlas it goes in. Must agree with the compiled code's choice of
    /// array for this image, which is the same declaration read on the
    /// application's side of the seam.
    pub kind: TextureKind,
    /// The image's own bytes, four to a pixel.
    ///
    /// For a [`TextureKind::Color`], sRGB-encoded and **not decoded**, and that
    /// is the point: the atlas is an sRGB format, so the sampler applies the
    /// transfer function on every tap and does it in the filter's precision.
    /// Handing over linear eight-bit instead would crush the darks — sRGB's
    /// twenty darkest code values land inside the first two of a linear byte,
    /// which is the banding the encoding exists to prevent.
    ///
    /// For a [`TextureKind::Data`], the numbers as stored.
    pub pixels: &'a [u8],
}

/// The bind group layout every material shares.
///
/// One layout, because the pipeline depends on it and a pipeline per material
/// would undo the whole point of keeping the constants out of the text.
pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

/// A material's data on the device.
pub struct Material {
    constants: wgpu::Buffer,
    /// The two atlases. Always present, even for a material with no images — a
    /// bind group cannot have a hole, so "no textures" is one white texel.
    #[allow(dead_code)]
    atlases: [wgpu::Texture; 2],
    bind: wgpu::BindGroup,
    placed: Vec<Placed>,
}

impl Material {
    /// Uploads a compiled material.
    ///
    /// `images` is parallel to [`Compiled::images`]: the compiler decided the
    /// layer order and this fills it — each atlas in the order its own kind
    /// appears in the list. A caller handing them in another order would give
    /// the material somebody else's textures, which is why the contract is
    /// stated rather than implied.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        data: MaterialData<'_>,
        images: &[SourceImage<'_>],
    ) -> Self {
        let (colours, colour_placed) = pack(device, queue, images, TextureKind::Color);
        let (numbers, data_placed) = pack(device, queue, images, TextureKind::Data);
        // Back into the order of `images`, which is the order the compiler
        // reserved rectangle slots in.
        let (mut colour_placed, mut data_placed) =
            (colour_placed.into_iter(), data_placed.into_iter());
        let placed: Vec<Placed> = images
            .iter()
            .filter_map(|image| match image.kind {
                TextureKind::Color => colour_placed.next(),
                TextureKind::Data => data_placed.next(),
            })
            .collect();
        let view_of = |atlas: &wgpu::Texture| {
            atlas.create_view(&wgpu::TextureViewDescriptor {
                label: Some("material atlas"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let colour_view = view_of(&colours);
        let data_view = view_of(&numbers);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material"),
            // Repeat, which is what a texture on a surface means when the UVs
            // run past one. The rectangle makes that a lie at a layer's edge —
            // a repeat would wrap into whatever is packed beside it — and the
            // generated code takes the fraction before applying the rectangle,
            // so the wrap happens in the image's own space.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            // A floor seen at a glancing angle is minified far more along one
            // axis than the other, and a plain trilinear tap picks the level
            // for the worse axis — the floor goes to mush a few metres out.
            // Sixteen is what every desktop part supports; wgpu clamps it
            // where it is not.
            anisotropy_clamp: 16,
            ..Default::default()
        });

        // The rectangles go in beside the compiler's constants, in the slots it
        // left for them. See `Compiled::constants`.
        let mut values = data.constants.to_vec();
        for (slot, rect) in data.image_slots.iter().zip(&placed) {
            if let Some(v) = values.get_mut(*slot) {
                *v = rect.rect;
            }
        }
        // A uniform cannot be empty, and a material with no constants at all is
        // not a thing the compiler emits — but the buffer's size is its own
        // rule and the two should not have to agree by luck.
        if values.is_empty() {
            values.push([0.0; 4]);
        }
        let constants = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("material constants"),
                contents: bytemuck::cast_slice(&values),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: constants.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&colour_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&data_view),
                },
            ],
        });
        Self {
            constants,
            atlases: [colours, numbers],
            bind,
            placed,
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind
    }

    /// Where each image landed.
    pub fn placed(&self) -> &[Placed] {
        &self.placed
    }

    /// Rewrites the numbers without touching the code or the textures.
    ///
    /// What a colour slider costs: one buffer write. The whole reason the
    /// compiler keeps constants out of the text.
    pub fn set_constants(&self, queue: &wgpu::Queue, data: MaterialData<'_>) {
        let mut values = data.constants.to_vec();
        for (slot, rect) in data.image_slots.iter().zip(&self.placed) {
            if let Some(v) = values.get_mut(*slot) {
                *v = rect.rect;
            }
        }
        if values.is_empty() {
            values.push([0.0; 4]);
        }
        queue.write_buffer(&self.constants, 0, bytemuck::cast_slice(&values));
    }
}

/// Builds one atlas from the images of one kind.
///
/// One image per layer at the size of the largest, which is the trivial packer
/// the module docs describe. The rectangles it returns are what a real packer
/// would return too, so replacing it changes nothing above this line.
fn pack(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    images: &[SourceImage<'_>],
    kind: TextureKind,
) -> (wgpu::Texture, Vec<Placed>) {
    let images: Vec<&SourceImage<'_>> = images.iter().filter(|i| i.kind == kind).collect();
    let width = images.iter().map(|i| i.width).max().unwrap_or(1).max(1);
    let height = images.iter().map(|i| i.height).max().unwrap_or(1).max(1);
    let layers = images.len().max(1) as u32;
    let levels = mip_levels(width, height);
    let atlas = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(match kind {
            TextureKind::Color => "material colour atlas",
            TextureKind::Data => "material data atlas",
        }),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: match kind {
            TextureKind::Color => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureKind::Data => wgpu::TextureFormat::Rgba8Unorm,
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // No images is one white layer, not an uninitialised one: a material whose
    // code does not sample this atlas never reads it, but a zeroed texel is
    // transparent black and white is the value every fallback here chooses.
    let white = [255u8; 4];
    let empty = SourceImage {
        width: 1,
        height: 1,
        kind,
        pixels: &white,
    };
    let stand_in = images.is_empty();
    let fill: Vec<&SourceImage<'_>> = if stand_in { vec![&empty] } else { images };

    let mut placed = Vec::with_capacity(fill.len());
    for (layer, image) in fill.iter().enumerate() {
        let w = image.width.min(width).max(1);
        let h = image.height.min(height).max(1);
        let chain = mip_chain(&padded(image, w, h, width, height), width, height, kind);
        for (level, (bytes, lw, lh)) in chain.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(lw * 4),
                    rows_per_image: Some(*lh),
                },
                wgpu::Extent3d {
                    width: *lw,
                    height: *lh,
                    depth_or_array_layers: 1,
                },
            );
        }
        placed.push(Placed {
            layer: layer as u32,
            rect: [0.0, 0.0, w as f32 / width as f32, h as f32 / height as f32],
        });
    }
    if stand_in {
        // The white stand-in is not one of the caller's images and has no
        // rectangle slot to fill.
        placed.clear();
    }
    (atlas, placed)
}

/// How many levels a full chain has, down to one texel.
pub fn mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// One image in a `width` x `height` layer, as bytes.
///
/// The image sits at the origin and the rest of the layer repeats its last row
/// and column. See the module docs on why not black. Short pixels are padded
/// rather than refused: a half-loaded image is a real state (§3.6's sparse
/// residency) and half a texture beats a dialog — the same judgement the
/// environment's upload makes.
fn padded(image: &SourceImage<'_>, w: u32, h: u32, width: u32, height: u32) -> Vec<u8> {
    let texel = |x: u32, y: u32| -> [u8; 4] {
        let at = ((y.min(h - 1) * image.width + x.min(w - 1)) * 4) as usize;
        match image.pixels.get(at..at + 4) {
            Some(p) => [p[0], p[1], p[2], p[3]],
            None => [0; 4],
        }
    };
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            out.extend_from_slice(&texel(x, y));
        }
    }
    out
}

/// The whole chain for one layer, level zero first, each with its size.
///
/// A two-by-two box, with the last row or column reused when a side is odd.
/// Colours are averaged in light and re-encoded; data is averaged as it is.
/// Alpha is never a colour and is always averaged as stored.
pub fn mip_chain(
    level0: &[u8],
    width: u32,
    height: u32,
    kind: TextureKind,
) -> Vec<(Vec<u8>, u32, u32)> {
    let decode: Vec<f32> = (0..256)
        .map(|b| match kind {
            TextureKind::Color => kreate_math::srgb_to_linear(b as f32 / 255.0),
            TextureKind::Data => b as f32 / 255.0,
        })
        .collect();
    let encode = |v: f32| -> u8 {
        let v = v.clamp(0.0, 1.0);
        let e = match kind {
            TextureKind::Color => kreate_math::linear_to_srgb(v),
            TextureKind::Data => v,
        };
        (e * 255.0).round() as u8
    };

    let mut chain = vec![(level0.to_vec(), width, height)];
    while let Some((prev, pw, ph)) = chain.last() {
        if *pw == 1 && *ph == 1 {
            break;
        }
        let (pw, ph) = (*pw, *ph);
        let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
        let mut next = Vec::with_capacity((nw * nh * 4) as usize);
        for y in 0..nh {
            for x in 0..nw {
                let xs = [(2 * x).min(pw - 1), (2 * x + 1).min(pw - 1)];
                let ys = [(2 * y).min(ph - 1), (2 * y + 1).min(ph - 1)];
                let mut sum = [0.0f32; 4];
                for sy in ys {
                    for sx in xs {
                        let at = ((sy * pw + sx) * 4) as usize;
                        for c in 0..3 {
                            sum[c] += decode[prev[at + c] as usize];
                        }
                        sum[3] += prev[at + 3] as f32 / 255.0;
                    }
                }
                for channel in &sum[..3] {
                    next.push(encode(channel / 4.0));
                }
                next.push(((sum[3] / 4.0).clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        chain.push((next, nw, nh));
    }
    chain
}

/// Pipelines, kept by the code that produced them.
///
/// Keyed on the generated text rather than on a material's identity, because
/// the text is what a pipeline *is* — two materials of the same shape share
/// one, and the same material recompiled after a colour change does not make a
/// second.
pub struct PipelineCache {
    by_source: ahash::AHashMap<String, wgpu::RenderPipeline>,
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            by_source: ahash::AHashMap::new(),
        }
    }

    /// The pipeline for this code, built by `make` if it is new.
    pub fn get(
        &mut self,
        wgsl: &str,
        make: impl FnOnce(&str) -> wgpu::RenderPipeline,
    ) -> &wgpu::RenderPipeline {
        self.by_source
            .entry(wgsl.to_string())
            .or_insert_with(|| make(wgsl))
    }

    /// The pipeline for this code, if it has been built.
    ///
    /// For a draw loop, which cannot build one: it holds the pass and the
    /// device's encoder, and making a pipeline in the middle of a frame is the
    /// hitch a cache exists to avoid.
    pub fn peek(&self, wgsl: &str) -> Option<&wgpu::RenderPipeline> {
        self.by_source.get(wgsl)
    }

    /// How many distinct pipelines exist.
    ///
    /// For the test that says a hundred materials of one shape are one
    /// pipeline, which is the claim the whole arrangement rests on.
    pub fn len(&self) -> usize {
        self.by_source.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_source.is_empty()
    }
}

#[cfg(test)]
mod mip_tests {
    use super::*;

    #[test]
    fn a_full_chain_ends_at_one_texel() {
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(256, 256), 9);
        // The longer side decides, and a side that is not a power of two
        // still gets a level for every halving it survives.
        assert_eq!(mip_levels(300, 20), 9);
        let chain = mip_chain(&vec![0u8; 300 * 20 * 4], 300, 20, TextureKind::Data);
        assert_eq!(chain.len(), 9);
        assert_eq!((chain[8].1, chain[8].2), (1, 1));
        for (bytes, w, h) in &chain {
            assert_eq!(
                bytes.len(),
                (w * h * 4) as usize,
                "a level is the wrong size"
            );
        }
    }

    #[test]
    fn a_colour_chain_averages_light_not_bytes() {
        // Black and white checks. Their mean is half the light, which sRGB
        // writes as 188 — averaging the bytes would give 128, which is 22% of
        // the light, and every checked, striped or speckled texture would go
        // dark as it recedes.
        let mut level0 = Vec::new();
        for i in 0..4 {
            let v = if i % 3 == 0 { 255 } else { 0 };
            level0.extend_from_slice(&[v, v, v, 255]);
        }
        let colour = mip_chain(&level0, 2, 2, TextureKind::Color);
        assert_eq!(colour[1].0[0], 188, "the colour chain averaged the bytes");
        // And data is exactly the arithmetic mean: a roughness of 0 and 1 is
        // a roughness of a half.
        let data = mip_chain(&level0, 2, 2, TextureKind::Data);
        assert_eq!(data[1].0[0], 128);
        // Alpha is not a colour in either.
        assert_eq!(colour[1].0[3], 255);
    }

    #[test]
    fn a_small_image_is_padded_with_its_own_edge() {
        // A 2x1 image in a 4x2 layer: every padding texel is a copy of the
        // nearest edge texel, never black.
        let pixels = [10, 20, 30, 255, 40, 50, 60, 255];
        let image = SourceImage {
            width: 2,
            height: 1,
            kind: TextureKind::Color,
            pixels: &pixels,
        };
        let layer = padded(&image, 2, 1, 4, 2);
        let at = |x: usize, y: usize| &layer[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4];
        assert_eq!(at(0, 0), &[10, 20, 30, 255]);
        assert_eq!(
            at(3, 0),
            &[40, 50, 60, 255],
            "the right margin is not the edge"
        );
        assert_eq!(
            at(0, 1),
            &[10, 20, 30, 255],
            "the bottom margin is not the edge"
        );
        assert_eq!(at(3, 1), &[40, 50, 60, 255]);
    }
}
