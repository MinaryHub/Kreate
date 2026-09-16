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

/// An image on its way to the device.
///
/// Pixels and a size, because `kreate-render` does not see an
/// [`Image`](kreate_scene::image::Image) any more than it sees a `Document`.
/// The application flattens the tiles, the same way it does for an environment.
pub struct SourceImage<'a> {
    pub width: u32,
    pub height: u32,
    /// `Rgba8Unorm`, already decoded into the working space by the caller.
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
        ],
    })
}

/// A material's data on the device.
pub struct Material {
    constants: wgpu::Buffer,
    /// The atlas. Always present, even for a material with no images — a bind
    /// group cannot have a hole, so "no textures" is one white texel.
    #[allow(dead_code)]
    atlas: wgpu::Texture,
    bind: wgpu::BindGroup,
    placed: Vec<Placed>,
}

impl Material {
    /// Uploads a compiled material.
    ///
    /// `images` is parallel to [`Compiled::images`]: the compiler decided the
    /// layer order and this fills it. A caller handing them in another order
    /// would give the material somebody else's textures, which is why the
    /// contract is stated rather than implied.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        data: MaterialData<'_>,
        images: &[SourceImage<'_>],
    ) -> Self {
        let (atlas, placed) = pack(device, queue, images);
        let view = atlas.create_view(&wgpu::TextureViewDescriptor {
            label: Some("material atlas"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
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
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        Self {
            constants,
            atlas,
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

/// Builds the atlas.
///
/// One image per layer at the size of the largest, which is the trivial packer
/// the module docs describe. The rectangles it returns are what a real packer
/// would return too, so replacing it changes nothing above this line.
fn pack(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    images: &[SourceImage<'_>],
) -> (wgpu::Texture, Vec<Placed>) {
    let width = images.iter().map(|i| i.width).max().unwrap_or(1).max(1);
    let height = images.iter().map(|i| i.height).max().unwrap_or(1).max(1);
    let layers = images.len().max(1) as u32;
    let atlas = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("material atlas"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut placed = Vec::with_capacity(images.len());
    for (layer, image) in images.iter().enumerate() {
        let w = image.width.min(width).max(1);
        let h = image.height.min(height).max(1);
        // Short pixels are padded rather than refused: a half-loaded image is a
        // real state (§3.6's sparse residency) and half a texture beats a
        // dialog. The same judgement the environment's upload makes.
        let needed = w as usize * h as usize * 4;
        let mut data = image.pixels.to_vec();
        data.resize(needed, 0);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        placed.push(Placed {
            layer: layer as u32,
            rect: [
                0.0,
                0.0,
                w as f32 / width as f32,
                h as f32 / height as f32,
            ],
        });
    }
    (atlas, placed)
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
