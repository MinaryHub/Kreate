//! The depth pyramid on a real GPU, against its CPU specification.
//!
//! The reduction is arithmetic-free — a minimum of floats — so the comparison is
//! **exact**, level for level, with no tolerance. That is worth insisting on: an
//! approximate comparison would accept a pyramid that is systematically a little
//! too near, and a pyramid that reports depths nearer than the truth rejects
//! clusters the user can see. Holes in a model read as corrupt geometry.
//!
//! The awkward cases are the point. Odd dimensions leave a row or column that
//! the 2x2 blocks never reach, and a mip chain's sizes are floored, so the last
//! texel of each level has to swallow the remainder. That fix-up is easy to get
//! right in one language and wrong in the other, which is exactly what a
//! two-implementation comparison catches.
//!
//! # Two things this had to work around
//!
//! WebGPU forbids copying *into* a `Depth32Float` texture, so a synthetic depth
//! pattern cannot be uploaded — it has to be rendered, by a fragment shader that
//! writes `frag_depth`. That is closer to real use anyway.
//!
//! And the CPU reference chain starts from the pyramid's own level 0 rather than
//! from the pattern the shader was asked to draw. Reproducing that pattern in
//! Rust bit-for-bit would make the test depend on two languages rounding the
//! same expression identically, which has nothing to do with what is being
//! tested. Level 0 is a straight copy of the depth buffer; the reduction above
//! it is the interesting part, and this compares exactly that.
//!
//! Skips rather than fails when no adapter is available.

use kreate_render::hiz::{self, HiZ};
use kreate_render::GpuOptions;

/// Writes a depth pattern with a distinctive value in every awkward place.
///
/// The far corner, the last column and the last row each get a depth nothing
/// else in the frame has, so a fix-up that fails to reach one shows up as a
/// specific missing number rather than as a small discrepancy. Reversed-Z, so
/// these small values are the *farthest* things in the frame.
const PATTERN: &str = r#"
struct Size { dims: vec4<u32> };
@group(0) @binding(0) var<uniform> s: Size;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // One oversized triangle covering the viewport.
    var xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(xy[vi], 0.5, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let x = u32(pos.x);
    let y = u32(pos.y);
    let w = s.dims.x;
    let h = s.dims.y;
    if (x + 1u == w && y + 1u == h) { return 0.011; }
    if (x + 1u == w) { return 0.022; }
    if (y + 1u == h) { return 0.033; }
    return 0.30 + f32((x * 7u + y * 13u) % 61u) * 0.01;
}
"#;

/// Every level of a built pyramid, read back.
fn build(width: u32, height: u32) -> Option<Vec<Vec<f32>>> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    let depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // --- render the pattern into the depth buffer ---------------------------
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth pattern"),
            source: wgpu::ShaderSource::Wgsl(PATTERN.into()),
        });
    let size_buffer = wgpu::util::DeviceExt::create_buffer_init(
        &gpu.device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("size"),
            contents: bytemuck::cast_slice(&[width, height, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );
    let size_layout = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("size"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let size_bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("size"),
        layout: &size_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: size_buffer.as_entire_binding(),
        }],
    });
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth pattern"),
            bind_group_layouts: &[Some(&size_layout)],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth pattern"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                // Always, so every fragment's written depth stands whatever the
                // clear was — the pattern is the point, not a depth test.
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let mut hiz = HiZ::new(&gpu.device, width, height);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("depth pattern"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &size_bind, &[]);
        pass.draw(0..3, 0..1);
    }

    hiz.build(&gpu.device, &gpu.queue, &mut encoder, &depth_view);

    // --- read every level back ---------------------------------------------
    let mut readbacks = Vec::new();
    for level in 0..hiz.mip_count() {
        let (lw, lh) = hiz::mip_size(width, height, level);
        let padded = (lw * 4).div_ceil(256) * 256;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hiz readback"),
            size: (padded * lh) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: hiz.texture(),
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(lh),
                },
            },
            wgpu::Extent3d {
                width: lw,
                height: lh,
                depth_or_array_layers: 1,
            },
        );
        readbacks.push((buffer, lw, lh, padded));
    }
    gpu.queue.submit([encoder.finish()]);

    for (buffer, ..) in &readbacks {
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    }
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;

    let mut levels = Vec::new();
    for (buffer, lw, lh, padded) in &readbacks {
        let data = buffer.slice(..).get_mapped_range();
        let mut level = Vec::with_capacity((*lw * *lh) as usize);
        for y in 0..*lh {
            let row = &data[(y * padded) as usize..][..(*lw * 4) as usize];
            level.extend_from_slice(bytemuck::cast_slice::<_, f32>(row));
        }
        drop(data);
        buffer.unmap();
        levels.push(level);
    }
    Some(levels)
}

fn check(width: u32, height: u32) -> bool {
    let Some(levels) = build(width, height) else {
        eprintln!("no adapter; skipping");
        return false;
    };
    assert_eq!(levels.len(), hiz::mip_count(width, height) as usize);

    // The comparison would be vacuous on a constant depth buffer: every
    // reduction would agree trivially. So the pattern has to have actually
    // varied.
    let distinct: std::collections::BTreeSet<u32> = levels[0].iter().map(|d| d.to_bits()).collect();
    assert!(
        distinct.len() > 20,
        "level 0 has only {} distinct depths, so the comparison proves little",
        distinct.len()
    );

    // The chain, from what the GPU actually has at level 0.
    let mut want = levels[0].clone();
    let mut size = (width, height);
    for (level, got) in levels.iter().enumerate().skip(1) {
        let next = hiz::mip_size(width, height, level as u32);
        want = hiz::reduce_reference(&want, size, next);
        size = next;

        assert_eq!(got.len(), want.len(), "level {level} size");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g,
                w,
                "{width}x{height} level {level} texel ({}, {}) of {}x{}: \
                 GPU {g} against CPU {w}",
                i % next.0 as usize,
                i / next.0 as usize,
                next.0,
                next.1,
            );
        }
    }
    true
}

#[test]
fn a_power_of_two_pyramid_matches_the_reference() {
    check(64, 64);
}

#[test]
fn a_non_square_pyramid_matches_the_reference() {
    // Different numbers of levels in each axis, so the shorter one clamps at a
    // single texel while the longer keeps halving.
    check(64, 16);
}

#[test]
fn an_odd_width_pyramid_matches_the_reference() {
    // The column fix-up. Without it the last column never reaches the pyramid,
    // its depths are lost, and the cull over-rejects at the screen edge.
    check(35, 16);
}

#[test]
fn an_odd_height_pyramid_matches_the_reference() {
    check(16, 35);
}

#[test]
fn a_pyramid_odd_in_both_axes_matches_the_reference() {
    // Both fix-ups plus the corner, which neither of them reaches alone.
    check(35, 21);
}

#[test]
fn a_realistic_viewport_size_matches_the_reference() {
    // 1080p halves to 540, 270, 135 and is odd from there down, so the fix-ups
    // fire at four consecutive levels on a completely ordinary window size.
    // This is not a contrived case.
    check(1920, 1080);
}

#[test]
fn the_top_level_is_the_farthest_depth_in_the_frame() {
    // The claim the occlusion test rests on, and the one an inverted reduction
    // would break most visibly. In reversed-Z the farthest depth is the
    // *smallest*, so this is a minimum — the opposite of what every hi-z
    // description written for standard Z says.
    let (w, h) = (35u32, 21u32);
    let Some(levels) = build(w, h) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let global = levels[0].iter().copied().fold(f32::INFINITY, f32::min);
    let top = levels.last().unwrap();
    assert_eq!(top.len(), 1);
    assert_eq!(
        top[0], global,
        "the pyramid lost the farthest depth in the frame"
    );
    // And that extremum is the value the pattern hid in the odd corner, so this
    // is not passing because the farthest depth happened to sit somewhere the
    // plain 2x2 blocks already reach.
    assert_eq!(global, 0.011, "the corner value did not survive");
}

#[test]
fn the_pyramid_never_reports_a_nearer_depth_than_its_region() {
    // The safety property, checked directly rather than inferred from the
    // reference agreeing: for every texel of every level, the value must be no
    // nearer than the farthest depth anywhere in the screen region it covers.
    //
    // Reporting nearer is what makes holes, so this is the assertion that
    // actually matters — and it holds independently of whether the CPU
    // reference is right.
    let (w, h) = (35u32, 21u32);
    let Some(levels) = build(w, h) else {
        eprintln!("no adapter; skipping");
        return;
    };
    let base = &levels[0];

    for (level, values) in levels.iter().enumerate().skip(1) {
        let (lw, lh) = hiz::mip_size(w, h, level as u32);
        let scale = 1u32 << level;
        for y in 0..lh {
            for x in 0..lw {
                // The region this texel is responsible for, including the
                // remainder the last texel in each axis swallows.
                let x0 = x * scale;
                let y0 = y * scale;
                let x1 = if x + 1 == lw { w } else { (x + 1) * scale };
                let y1 = if y + 1 == lh { h } else { (y + 1) * scale };
                let mut farthest = f32::INFINITY;
                for sy in y0..y1.min(h) {
                    for sx in x0..x1.min(w) {
                        farthest = farthest.min(base[(sy * w + sx) as usize]);
                    }
                }
                let got = values[(y * lw + x) as usize];
                assert!(
                    got <= farthest,
                    "level {level} texel ({x}, {y}) reports {got}, nearer than \
                     the farthest depth {farthest} in its region"
                );
            }
        }
    }
}
