//! The images the studio and flat modes show: Blender's solid-mode Texture
//! colour.
//!
//! Not the rendered mode's material atlas, on purpose. Those modes compile no
//! materials — that is what keeps them light — and all they want of an image
//! is its colour across the surface's UVs. So this is one small array of its
//! own, one layer per base-colour image, every image resampled to one square
//! size, with its mip chain so a distant brick wall does not shimmer. A slot
//! names its layer in the object's uniform (`slot_layers`), and the solid
//! shader reads it there.

use crate::material::{mip_chain, TextureKind};

/// The largest a layer is: four layers of this cost four megabytes each with
/// their mips, which is what a modelling view can spend on a preview.
pub const MAX_SIZE: u32 = 512;
/// How many images can be shown at once; slots past these show their colour.
pub const MAX_LAYERS: usize = 16;

/// One image's colour bytes, sRGB-encoded, four to a pixel.
#[derive(Clone, Copy, Debug)]
pub struct SolidImage<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
}

pub struct SolidTextures {
    pub layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pub bind: wgpu::BindGroup,
    _texture: wgpu::Texture,
    layers: usize,
}

/// `pixels` (`width` × `height`) resampled to `size` square, bilinear in the
/// encoded values — a preview's accuracy, and what keeps this free of a
/// decode per texel.
fn resample(pixels: &[u8], width: u32, height: u32, size: u32) -> Vec<u8> {
    let (w, h) = (width.max(1) as usize, height.max(1) as usize);
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    if pixels.len() < w * h * 4 {
        out.resize((size * size * 4) as usize, 255);
        return out;
    }
    for y in 0..size {
        for x in 0..size {
            let fx = (x as f32 + 0.5) / size as f32 * w as f32 - 0.5;
            let fy = (y as f32 + 0.5) / size as f32 * h as f32 - 0.5;
            let (x0, y0) = (fx.floor().max(0.0) as usize, fy.floor().max(0.0) as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            let (tx, ty) = ((fx - x0 as f32).clamp(0.0, 1.0), (fy - y0 as f32).clamp(0.0, 1.0));
            for c in 0..4 {
                let at = |px: usize, py: usize| pixels[(py * w + px) * 4 + c] as f32;
                let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
                let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
                out.push((top * (1.0 - ty) + bottom * ty).round().clamp(0.0, 255.0) as u8);
            }
        }
    }
    out
}

impl SolidTextures {
    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("solid textures"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    /// Nothing to show yet: one empty layer, which no slot names.
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = Self::layout(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("solid textures"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let (texture, bind) = Self::upload(device, None, &layout, &sampler, &[]);
        Self {
            layout,
            sampler,
            bind,
            _texture: texture,
            layers: 0,
        }
    }

    /// How many images it holds now.
    pub fn len(&self) -> usize {
        self.layers
    }

    pub fn is_empty(&self) -> bool {
        self.layers == 0
    }

    /// Replaces what it holds with `images`, one layer each, the first
    /// [`MAX_LAYERS`] of them.
    pub fn set(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, images: &[SolidImage<'_>]) {
        let images = &images[..images.len().min(MAX_LAYERS)];
        let (texture, bind) = Self::upload(device, Some(queue), &self.layout, &self.sampler, images);
        self._texture = texture;
        self.bind = bind;
        self.layers = images.len();
    }

    fn upload(
        device: &wgpu::Device,
        queue: Option<&wgpu::Queue>,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        images: &[SolidImage<'_>],
    ) -> (wgpu::Texture, wgpu::BindGroup) {
        // As large as the largest image needs, a power of two, no more than
        // the cap.
        let largest = images.iter().map(|i| i.width.max(i.height)).max().unwrap_or(1);
        let size = largest.next_power_of_two().clamp(1, MAX_SIZE);
        let levels = 32 - size.leading_zeros();
        let layers = images.len().max(1) as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("solid textures"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let white = vec![255u8; (size * size * 4) as usize];
        for layer in (0..layers).filter(|_| queue.is_some()) {
            let Some(queue) = queue else { break };
            let level0 = images
                .get(layer as usize)
                .map(|i| resample(i.pixels, i.width, i.height, size))
                .unwrap_or_else(|| white.clone());
            for (level, (bytes, w, h)) in mip_chain(&level0, size, size, TextureKind::Color).into_iter().enumerate() {
                if level as u32 >= levels {
                    break;
                }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: level as u32,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &bytes,
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
            }
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("solid textures"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        (texture, bind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_keeps_a_flat_colour_and_scales_a_checker() {
        let red = [200u8, 10, 10, 255].repeat(4);
        let out = resample(&red, 2, 2, 8);
        assert_eq!(out.len(), 8 * 8 * 4);
        assert!(out.chunks(4).all(|p| p == [200, 10, 10, 255]));
        // Left half black, right half white: still so at a larger size.
        let half = [[0u8, 0, 0, 255], [255, 255, 255, 255]].concat();
        let out = resample(&half, 2, 1, 4);
        assert_eq!(&out[0..4], &[0, 0, 0, 255]);
        assert_eq!(&out[12..16], &[255, 255, 255, 255]);
    }
}
