//! The depth pyramid, for occlusion culling.
//!
//! Frustum and normal-cone rejection are cheap and shallow: they remove what is
//! off screen and what faces away, and nothing else. In a scene with any depth
//! complexity the largest share of clusters is neither — it is *behind
//! something*, and rejecting those needs a depth buffer.
//!
//! A pyramid is what makes the test affordable. Testing a cluster against every
//! pixel it covers would cost more than drawing it; testing it against one texel
//! of a mip chosen so the cluster spans about two texels costs four samples,
//! whatever the cluster's size on screen.
//!
//! # Reversed-Z inverts the reduction
//!
//! Every description of a hi-z pyramid written for standard Z says to take the
//! **maximum** depth. This renderer uses reversed-Z — 1.0 at the near plane,
//! 0.0 at infinity — so the farthest depth is the **minimum**, and the pyramid
//! min-reduces. See the shader for the derivation.
//!
//! The failure mode if that is wrong is worth stating, because it is not a
//! visual glitch: the pyramid would report everything as nearer than it is, the
//! cull would reject clusters that are actually visible, and the model would
//! come out with holes. In an editor a hole reads as corrupt geometry, and the
//! user blames their mesh.
//!
//! # Conservative in one direction only
//!
//! Every approximation here widens the region a texel covers, never narrows it.
//! Widening can only make the reported depth farther, and a farther depth culls
//! less. That is the safe direction, and it is why the odd-dimension handling in
//! the shader exists rather than being rounded away.

use crate::gpu;

/// Format of the pyramid.
///
/// `R32Float` because the depth it mirrors is `Depth32Float` and a narrower
/// format would quantise: a pyramid whose texel is *nearer* than the true
/// farthest depth over-culls, which is the direction that makes holes. It is
/// also one of the formats WebGPU core allows as a write-only storage texture,
/// so no optional feature is involved.
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Threads per side in the build shader. Must match the shader.
const WORKGROUP: u32 = 8;

/// How many mip levels a pyramid of this size has.
///
/// Down to a single texel, so the top level answers "is this cluster behind
/// everything in the entire frame" in one sample. The chain is short — 11 levels
/// at 1080p — and the small levels cost nothing.
pub fn mip_count(width: u32, height: u32) -> u32 {
    let largest = width.max(height).max(1);
    32 - largest.leading_zeros()
}

/// The size of one level, floored the way a mip chain's sizes are.
pub fn mip_size(width: u32, height: u32, level: u32) -> (u32, u32) {
    ((width >> level).max(1), (height >> level).max(1))
}

/// The reduction the shader implements, on the CPU.
///
/// This is the *specification*: the shader is a second implementation in another
/// language with its own indexing, and a pyramid that is subtly wrong produces
/// holes rather than an obviously broken image. The GPU build is checked against
/// this level for level.
///
/// Exact rather than approximate comparison is possible because a minimum of
/// floats introduces no arithmetic.
pub fn reduce_reference(src: &[f32], src_size: (u32, u32), dst_size: (u32, u32)) -> Vec<f32> {
    let (sw, sh) = (src_size.0 as usize, src_size.1 as usize);
    let (dw, dh) = (dst_size.0 as usize, dst_size.1 as usize);
    let at = |x: usize, y: usize| src[y.min(sh - 1) * sw + x.min(sw - 1)];

    let mut out = vec![f32::INFINITY; dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            let (bx, by) = (x * 2, y * 2);
            let mut m = at(bx, by);
            m = m.min(at(bx + 1, by));
            m = m.min(at(bx, by + 1));
            m = m.min(at(bx + 1, by + 1));
            // The row or column a floored mip size leaves behind. Swallowed by
            // the last destination texel, which is safe because widening a
            // minimum's coverage can only report a farther depth.
            if sw % 2 == 1 && x + 1 == dw {
                m = m.min(at(sw - 1, by));
                m = m.min(at(sw - 1, by + 1));
            }
            if sh % 2 == 1 && y + 1 == dh {
                m = m.min(at(bx, sh - 1));
                m = m.min(at(bx + 1, sh - 1));
            }
            if sw % 2 == 1 && sh % 2 == 1 && x + 1 == dw && y + 1 == dh {
                m = m.min(at(sw - 1, sh - 1));
            }
            out[y * dw + x] = m;
        }
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    /// x, y: destination size. z, w: source size.
    size: [u32; 4],
}

/// The pyramid and the passes that build it.
pub struct HiZ {
    texture: wgpu::Texture,
    /// One view per level, because a storage-texture binding addresses exactly
    /// one mip.
    levels: Vec<wgpu::TextureView>,
    /// A view of the whole chain, for the cull shader to sample.
    all: wgpu::TextureView,
    size: (u32, u32),
    copy_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    copy_layout: wgpu::BindGroupLayout,
    reduce_layout: wgpu::BindGroupLayout,
    /// One params slot per level, addressed by dynamic offset.
    params: wgpu::Buffer,
    params_stride: u32,
    /// Bind groups per level, rebuilt on resize.
    copy_bind: Option<wgpu::BindGroup>,
    reduce_binds: Vec<wgpu::BindGroup>,
}

impl HiZ {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_build"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hiz_build.wgsl").into()),
        });

        let copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz copy"),
            entries: &[
                params_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_texture_entry(2),
            ],
        });
        let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz reduce"),
            entries: &[
                params_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_texture_entry(2),
            ],
        });

        // Group 1 in the shader, so the layout list needs a hole at group 0.
        let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz reduce layout"),
            bind_group_layouts: &[None, Some(&reduce_layout)],
            immediate_size: 0,
        });
        let copy_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz copy layout"),
            bind_group_layouts: &[Some(&copy_layout)],
            immediate_size: 0,
        });
        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hiz copy"),
            layout: Some(&copy_pl),
            module: &shader,
            entry_point: Some("from_depth"),
            compilation_options: Default::default(),
            cache: None,
        });
        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hiz reduce"),
            layout: Some(&reduce_pl),
            module: &shader,
            entry_point: Some("reduce"),
            compilation_options: Default::default(),
            cache: None,
        });

        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let params_stride = (std::mem::size_of::<Params>() as u64).div_ceil(align) * align;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hiz params"),
            size: params_stride * 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (texture, levels, all) = allocate(device, width, height);
        let mut hiz = Self {
            texture,
            levels,
            all,
            size: (width.max(1), height.max(1)),
            copy_pipeline,
            reduce_pipeline,
            copy_layout,
            reduce_layout,
            params,
            params_stride: params_stride as u32,
            copy_bind: None,
            reduce_binds: Vec::new(),
        };
        hiz.build_reduce_binds(device);
        hiz
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn mip_count(&self) -> u32 {
        self.levels.len() as u32
    }

    /// A view of the whole chain, for sampling in the cull.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.all
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Reallocates for a new render size, if it changed.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size == self.size {
            return;
        }
        let (texture, levels, all) = allocate(device, size.0, size.1);
        self.texture = texture;
        self.levels = levels;
        self.all = all;
        self.size = size;
        self.copy_bind = None;
        self.build_reduce_binds(device);
    }

    /// The reduce bind groups, one per level above zero.
    fn build_reduce_binds(&mut self, device: &wgpu::Device) {
        self.reduce_binds = (1..self.levels.len())
            .map(|level| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hiz reduce"),
                    layout: &self.reduce_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.params,
                                offset: 0,
                                size: std::num::NonZeroU64::new(
                                    std::mem::size_of::<Params>() as u64
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.levels[level - 1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.levels[level]),
                        },
                    ],
                })
            })
            .collect();
    }

    /// Encodes the pyramid build from a depth view.
    ///
    /// `depth` must be the depth texture the last colour pass wrote, at the same
    /// resolution this pyramid was sized for. One dispatch per level, each
    /// reading the level below — which is why they cannot be merged into one
    /// pass without a barrier between every step anyway.
    pub fn build(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
    ) {
        if self.copy_bind.is_none() {
            self.copy_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hiz copy"),
                layout: &self.copy_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.params,
                            offset: 0,
                            size: std::num::NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.levels[0]),
                    },
                ],
            }));
        }

        // Every level's sizes written up front, so the dispatches below need no
        // interleaved buffer writes — a queue write between two passes in one
        // encoder is a hazard, not a convenience.
        for level in 0..self.levels.len() as u32 {
            let dst = mip_size(self.size.0, self.size.1, level);
            let src = if level == 0 {
                dst
            } else {
                mip_size(self.size.0, self.size.1, level - 1)
            };
            queue.write_buffer(
                &self.params,
                (level * self.params_stride) as u64,
                bytemuck::bytes_of(&Params {
                    size: [dst.0, dst.1, src.0, src.1],
                }),
            );
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz level 0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, self.copy_bind.as_ref().unwrap(), &[0]);
            let (w, h) = self.size;
            pass.dispatch_workgroups(w.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
        }

        for level in 1..self.levels.len() as u32 {
            // A pass per level. They cannot share one: each reads what the
            // previous wrote, and wgpu inserts the barrier at the pass boundary.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduce_pipeline);
            pass.set_bind_group(
                1,
                &self.reduce_binds[level as usize - 1],
                &[level * self.params_stride],
            );
            let (w, h) = mip_size(self.size.0, self.size.1, level);
            pass.dispatch_workgroups(w.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
        }
    }
}

fn params_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn allocate(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, Vec<wgpu::TextureView>, wgpu::TextureView) {
    let (width, height) = (width.max(1), height.max(1));
    let levels = mip_count(width, height);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hiz pyramid"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let views = (0..levels)
        .map(|level| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hiz level"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    let all = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("hiz all levels"),
        ..Default::default()
    });
    (texture, views, all)
}

/// Asserts the pyramid mirrors the depth format it reduces.
///
/// Not a runtime check — a compile-time reminder that the two are coupled. A
/// narrower pyramid format would quantise towards *nearer* depths, which
/// over-culls, which makes holes.
const _: () = assert!(matches!(
    gpu::DEPTH_FORMAT,
    wgpu::TextureFormat::Depth32Float
));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chain_reaches_a_single_texel() {
        // The top level has to be 1x1, because the cheapest useful question is
        // "is this behind everything in the frame" and that should be one sample.
        for (w, h) in [(1u32, 1u32), (2, 2), (1024, 1024), (1920, 1080), (1023, 17)] {
            let levels = mip_count(w, h);
            let top = mip_size(w, h, levels - 1);
            assert_eq!(top, (1, 1), "{w}x{h} topped out at {top:?}");
        }
    }

    #[test]
    fn every_level_is_at_least_one_texel() {
        // A zero-sized level would be a zero-sized dispatch and a zero-sized
        // binding, and the latter is a validation error.
        let (w, h) = (1920u32, 1080u32);
        for level in 0..mip_count(w, h) {
            let (lw, lh) = mip_size(w, h, level);
            assert!(lw >= 1 && lh >= 1);
        }
    }

    #[test]
    fn the_reference_reduction_takes_the_farthest_depth() {
        // Reversed-Z: smaller is farther, so the reduction is a minimum. The
        // single most consequential line in this module, and the one every
        // article written for standard Z would have backwards.
        let src = vec![0.9, 0.1, 0.5, 0.7];
        let out = reduce_reference(&src, (2, 2), (1, 1));
        assert_eq!(out, vec![0.1]);
    }

    #[test]
    fn an_odd_source_does_not_lose_its_last_column() {
        // Three columns reducing to one: the middle 2x2 block covers columns 0
        // and 1, and column 2 has to be swallowed by the same destination texel.
        // Dropped instead, the pyramid would report 0.9 — nearer than the truth
        // — and the cull would reject clusters visible at the screen edge.
        let src = vec![0.9, 0.8, 0.2];
        let out = reduce_reference(&src, (3, 1), (1, 1));
        assert_eq!(out, vec![0.2], "the odd column was dropped");
    }

    #[test]
    fn an_odd_source_does_not_lose_its_last_row() {
        let src = vec![0.9, 0.8, 0.2];
        let out = reduce_reference(&src, (1, 3), (1, 1));
        assert_eq!(out, vec![0.2], "the odd row was dropped");
    }

    #[test]
    fn an_odd_source_does_not_lose_its_far_corner() {
        // Both dimensions odd, so the corner texel is reached by neither the
        // row nor the column fix-up and needs its own.
        let mut src = vec![0.9; 9];
        src[8] = 0.05;
        let out = reduce_reference(&src, (3, 3), (1, 1));
        assert_eq!(out, vec![0.05], "the odd corner was dropped");
    }

    #[test]
    fn the_reduction_never_reports_a_nearer_depth_than_the_truth() {
        // The safety property, over the whole chain: at every level, a texel's
        // value must be no nearer than the farthest depth in the region it
        // covers. Reporting nearer is what makes holes.
        //
        // Checked by reducing a pattern all the way down and comparing the top
        // against the global minimum, at a size where every level is odd and the
        // fix-ups all fire.
        let (w, h) = (7usize, 5usize);
        let src: Vec<f32> = (0..w * h).map(|i| (i as f32 * 0.017) % 1.0).collect();
        let global = src.iter().copied().fold(f32::INFINITY, f32::min);

        let mut level = src.clone();
        let mut size = (w as u32, h as u32);
        for l in 1..mip_count(w as u32, h as u32) {
            let next = mip_size(w as u32, h as u32, l);
            level = reduce_reference(&level, size, next);
            size = next;
        }
        assert_eq!(size, (1, 1));
        assert_eq!(
            level[0], global,
            "the chain lost the farthest depth in the frame"
        );
    }

    #[test]
    fn the_shader_parses() {
        let module =
            naga::front::wgsl::parse_str(include_str!("shaders/hiz_build.wgsl")).expect("parses");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).expect("validates");
    }
}
