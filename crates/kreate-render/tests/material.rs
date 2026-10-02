//! The contract between the compiler and the library it writes for.
//!
//! `kreate-shader` emits a function that names a `Surface`, a bind group and a
//! sampler it has never seen — the two crates do not depend on each other, on
//! purpose, so that the renderer does not acquire the scene model along the way.
//! What holds them together is this: the generated text is composed with the
//! real shading library and put through the real device.
//!
//! Every mistake the arrangement invites is a mistake nothing else would catch.
//! A field renamed in `brdf.wgsl`, a binding number changed on one side, a
//! group taken for something else — each of them compiles on both sides and
//! fails where the two meet, which is at pipeline creation on a user's machine.

use kreate_render::material::{Material, MaterialData, SourceImage, MATERIAL_GROUP};
use kreate_render::GpuOptions;
use kreate_scene::shader::{Link, Node, NodeKind, ShaderGraph};

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// The one device this binary uses. See `deform.rs` for why it is shared.
fn gpu() -> Option<Gpu> {
    static SHARED: std::sync::OnceLock<Option<(wgpu::Device, wgpu::Queue)>> =
        std::sync::OnceLock::new();
    let (device, queue) = SHARED
        .get_or_init(|| {
            let opts = GpuOptions::default();
            let instance = kreate_render::Gpu::create_instance(&opts);
            let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
            Some((gpu.device.clone(), gpu.queue.clone()))
        })
        .clone()?;
    Some(Gpu { device, queue })
}

macro_rules! gpu_or_skip {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => {
                eprintln!("no GPU adapter; skipping");
                return;
            }
        }
    };
}

/// A grey material, the graph a flat one is.
fn flat() -> ShaderGraph {
    ShaderGraph::standard("Grey", kreate_math::SceneRgb::new(0.8, 0.7, 0.6), 0.2, 0.4)
}

/// The same with an image feeding its base colour.
fn textured() -> (ShaderGraph, kreate_scene::Document) {
    let mut doc = kreate_scene::Document::new();
    let id = doc.add_image(kreate_scene::Image::new(
        "Albedo",
        8,
        4,
        kreate_scene::PixelFormat::Rgba8,
    ));
    let mut graph = flat();
    let surface = graph.output().expect("a surface");
    let image = graph
        .add_node(Node::new("Albedo", NodeKind::Image { image: Some(id) }))
        .expect("added");
    graph
        .link(Link {
            from_node: image,
            from_socket: 0,
            to_node: surface,
            to_socket: 0,
        })
        .expect("connects");
    (graph, doc)
}

/// The shading library with a compiled material on the end, and a fragment
/// stage that uses it — which is what a drawing pass will be.
fn composed(graph: &ShaderGraph) -> (String, kreate_shader::Compiled) {
    let compiled = kreate_shader::compile(graph).expect("compiles");
    let source = format!(
        "{}\n{}\n{}",
        kreate_render::shading::shader_source(),
        compiled.wgsl,
        // A stage that calls it, so the function cannot be dropped as unused
        // and its uses of the library are actually resolved.
        r#"
struct Vertex {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Vertex {
    var out: Vertex;
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1);
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs(in: Vertex) -> @location(0) vec4<f32> {
    let s = material_sample(in.uv, dpdx(in.uv), dpdy(in.uv)).surface;
    return vec4<f32>(shade_point(s, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), vec2<u32>(0u, 0u)), 1.0);
}
"#
    );
    (source, compiled)
}

#[test]
fn a_compiled_material_builds_a_pipeline_against_the_real_library() {
    // **The whole contract, in one call.** The compiler names a `Surface`, a
    // uniform, a texture array and a sampler that it has never seen; this is
    // where the names either exist or do not.
    //
    // A pipeline rather than a shader module, because a module compiles with a
    // binding nothing provides — it is the *layout* that has to agree, and a
    // layout mismatch is a validation error on a user's machine at the moment
    // they first select a material.
    let gpu = gpu_or_skip!(gpu());
    let (source, _) = composed(&flat());
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("material contract"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let material = kreate_render::material::layout(&gpu.device);
    let shading = shading_layout(&gpu.device);
    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("material contract"),
            // Groups zero and one belong to the camera and the object; the
            // shading data is two and the material is three.
            bind_group_layouts: &[None, None, Some(&shading), Some(&material)],
            immediate_size: 0,
        });
    assert_eq!(MATERIAL_GROUP, 3, "the group the layout above assumes");
    let _pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("material contract"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: kreate_render::display::HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
}

#[test]
fn a_textured_material_builds_one_too() {
    // The other shape the compiler emits: a sample, a rectangle and two slices
    // of one `vec4`. Its own test because it uses bindings the flat one never
    // touches — a texture array and a sampler that exist only for it.
    let gpu = gpu_or_skip!(gpu());
    let (graph, _doc) = textured();
    let (source, compiled) = composed(&graph);
    assert_eq!(compiled.images.len(), 1);
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("textured contract"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let material = kreate_render::material::layout(&gpu.device);
    let shading = shading_layout(&gpu.device);
    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("textured contract"),
            bind_group_layouts: &[None, None, Some(&shading), Some(&material)],
            immediate_size: 0,
        });
    let _pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("textured contract"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: kreate_render::display::HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
}

#[test]
fn the_uploaded_constants_are_as_long_as_the_shader_declares() {
    // A uniform shorter than the array the code indexes is a validation error
    // three frames from its cause — which is exactly the shape of the mistake
    // the shading globals made when they grew a field and a test did not.
    let gpu = gpu_or_skip!(gpu());
    let (graph, _doc) = textured();
    let compiled = kreate_shader::compile(&graph).expect("compiles");
    let declared = compiled
        .wgsl
        .split("array<vec4<f32>, ")
        .nth(1)
        .and_then(|rest| rest.split('>').next())
        .and_then(|n| n.trim().parse::<usize>().ok())
        .expect("the code declares an array");
    assert_eq!(
        declared,
        compiled.constants.len(),
        "the code indexes {declared} slots and {} were produced",
        compiled.constants.len()
    );

    // And the material uploads that many: a bind group is built here rather
    // than asserted about, because the buffer's size is checked by the device.
    let layout = kreate_render::material::layout(&gpu.device);
    let pixels = vec![255u8; 8 * 4 * 4];
    let material = Material::new(
        &gpu.device,
        &gpu.queue,
        &layout,
        MaterialData {
            constants: &compiled.constants,
            image_slots: &compiled.image_slots,
        },
        &[SourceImage {
            kind: kreate_render::material::TextureKind::Color,
            width: 8,
            height: 4,
            pixels: &pixels,
        }],
    );
    assert_eq!(material.placed().len(), 1);
}

#[test]
fn an_image_lands_where_the_shader_will_look_for_it() {
    // The rectangle, which is the compiler's slot and the renderer's answer.
    // Neither side can check it alone: the compiler does not know how an atlas
    // packs and the renderer does not know which slot the code reads.
    let gpu = gpu_or_skip!(gpu());
    let (graph, _doc) = textured();
    let compiled = kreate_shader::compile(&graph).expect("compiles");
    let layout = kreate_render::material::layout(&gpu.device);
    let small = vec![255u8; 8 * 4 * 4];
    let material = Material::new(
        &gpu.device,
        &gpu.queue,
        &layout,
        MaterialData {
            constants: &compiled.constants,
            image_slots: &compiled.image_slots,
        },
        &[SourceImage {
            kind: kreate_render::material::TextureKind::Color,
            width: 8,
            height: 4,
            pixels: &small,
        }],
    );
    // One image, alone on its layer, so it fills it.
    assert_eq!(material.placed()[0].layer, 0);
    assert_eq!(material.placed()[0].rect, [0.0, 0.0, 1.0, 1.0]);
    // And the slot the compiler reserved is one the constants actually have.
    assert!(compiled.image_slots[0] < compiled.constants.len());
}

#[test]
fn a_smaller_image_gets_a_rectangle_rather_than_being_stretched() {
    // Two images of different sizes on one material. The layer is the size of
    // the larger, and the smaller carries the fraction of it that is its own —
    // nothing is resampled, which is the claim the trivial packer makes.
    let gpu = gpu_or_skip!(gpu());
    let layout = kreate_render::material::layout(&gpu.device);
    let big = vec![255u8; 16 * 16 * 4];
    let small = vec![128u8; 4 * 8 * 4];
    let material = Material::new(
        &gpu.device,
        &gpu.queue,
        &layout,
        MaterialData {
            constants: &[[0.0; 4], [0.0; 4]],
            image_slots: &[0, 1],
        },
        &[
            SourceImage {
                kind: kreate_render::material::TextureKind::Color,
                width: 16,
                height: 16,
                pixels: &big,
            },
            SourceImage {
                kind: kreate_render::material::TextureKind::Color,
                width: 4,
                height: 8,
                pixels: &small,
            },
        ],
    );
    assert_eq!(material.placed()[0].rect, [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(material.placed()[1].rect, [0.0, 0.0, 4.0 / 16.0, 8.0 / 16.0]);
    assert_eq!(material.placed()[1].layer, 1);
}

#[test]
fn a_hundred_materials_of_one_shape_are_one_pipeline() {
    // **What the whole arrangement is for.** Constants live in a uniform so
    // that the text depends on the graph's shape alone — and if that stops
    // being true, a scene with a hundred tinted plastics compiles a hundred
    // shaders and the application stops while somebody drags a colour slider.
    let gpu = gpu_or_skip!(gpu());
    let mut cache = kreate_render::material::PipelineCache::new();
    let mut built = 0;
    for i in 0..100 {
        let graph = ShaderGraph::standard(
            format!("Plastic {i}"),
            kreate_math::SceneRgb::new(i as f32 / 100.0, 0.5, 0.5),
            0.0,
            i as f32 / 100.0,
        );
        let compiled = kreate_shader::compile(&graph).expect("compiles");
        cache.get(&compiled.wgsl, |_| {
            built += 1;
            trivial_pipeline(&gpu.device)
        });
    }
    assert_eq!(built, 1, "a hundred tinted plastics built {built} pipelines");
    assert_eq!(cache.len(), 1);

    // And a material of a *different* shape is a second one, or the cache is
    // handing out somebody else's code.
    let (graph, _doc) = textured();
    let compiled = kreate_shader::compile(&graph).expect("compiles");
    cache.get(&compiled.wgsl, |_| {
        built += 1;
        trivial_pipeline(&gpu.device)
    });
    assert_eq!(built, 2);
}

/// A pipeline that does nothing, for counting how many were made.
fn trivial_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
             @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }"
                .into(),
        ),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        // A target, because a pipeline with neither a colour attachment nor a
        // depth one is refused — which is what this counting stand-in first was.
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: kreate_render::display::HDR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// The shading library's bind group layout, as the rendered pass declares it.
///
/// Spelled out here rather than reached for, because `Viewport` owns the real
/// one and building a viewport to compile a shader would make this test about
/// a window. What it has to be is *compatible*, which is what the device
/// checks.
fn shading_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shading data"),
        entries: &[
            storage(0),
            storage(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(3),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            storage(5),
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// And the picture.
//
// Everything above checks that the pieces fit: the code compiles, the layouts
// agree, the rectangles line up. None of it would notice a material that is
// wired correctly and shades nothing — a surface coordinate stuck at zero, an
// atlas nobody sampled, a texture bound to the wrong layer. Those are questions
// about a picture, and the only way to ask them is to render one.

/// Renders a quad with one material and reads the frame back.
///
/// A quad rather than a cube so that the whole frame is one surface with UVs
/// running corner to corner, which is what makes "the left half differs from
/// the right half" a statement about a texture.
fn shoot(gpu: &Gpu, graph: &ShaderGraph, images: &[kreate_render::material::SourceImage<'_>]) -> Vec<[f32; 4]> {
    let compiled = kreate_shader::compile(graph).expect("compiles");
    shoot_with(gpu, &compiled, images, &Stage::default())
}

/// What surrounds the quad: its lights, its sky, how often the texture repeats.
struct Stage {
    lights: Vec<kreate_render::lights::PlacedLight>,
    world: [f32; 3],
    /// How many times the UVs cross the quad. More than one is a texture
    /// minified, which is what a mip chain is for.
    repeat: f32,
}

impl Default for Stage {
    fn default() -> Self {
        Self {
            lights: Vec::new(),
            world: [1.0; 3],
            repeat: 1.0,
        }
    }
}

/// The same with a material already compiled — for one whose images are not
/// all colours, which `kreate_shader::compile` assumes they are — and a stage.
fn shoot_with(
    gpu: &Gpu,
    compiled: &kreate_shader::Compiled,
    images: &[kreate_render::material::SourceImage<'_>],
    stage: &Stage,
) -> Vec<[f32; 4]> {
    use kreate_render::{Camera, Draw, RenderMeshCpu, ShadingMode, Viewport};

    const SIZE: u32 = 48;
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.show_grid = false;
    viewport.show_wireframe = false;
    // A plain white environment, so what reaches the camera is the surface's
    // own colour rather than a lamp's angle. Every texel of the quad faces the
    // same way, so a difference across the frame can only come from the
    // material.
    viewport.set_world(stage.world);
    viewport.set_display(
        &gpu.queue,
        kreate_render::display::Display {
            view: kreate_math::ViewTransform::Standard,
            exposure: 0.0,
        },
    );

    viewport.set_materials(
        &gpu.device,
        &gpu.queue,
        &[kreate_render::viewport::MaterialSource {
            wgsl: &compiled.wgsl,
            constants: &compiled.constants,
            image_slots: &compiled.image_slots,
            images,
        }],
    );

    // A quad with coordinates projected onto it.
    //
    // **The primitives carry none**, which this test found: a cube added from
    // the menu has no UV layer at all, so a texture assigned to it would sample
    // one texel. That is a gap in the primitives rather than in the renderer,
    // and it is not papered over here — what is being asked about is the path
    // from a coordinate to a pixel, so the fixture makes its own coordinates.
    let mesh = planar_uvs(kreate_mesh::primitives::grid(2.0, 1, 1), stage.repeat);
    let cpu = RenderMeshCpu::extract(&mesh);
    assert!(
        cpu.uvs.iter().any(|uv| uv[0] != 0.0 || uv[1] != 0.0),
        "the fixture has no surface coordinates, so it cannot say anything \
         about a texture"
    );
    let uploaded =
        kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cpu, glam::Vec4::ONE);

    // Looking straight down at the grid, which lies in the XY plane.
    let camera = Camera {
        distance: 2.2,
        ..Camera::default()
    };
    let mut draws = vec![Draw::new(&uploaded, glam::DAffine3::IDENTITY)];
    viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &stage.lights);
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    viewport.render(&mut encoder, &view, &draws);

    let row = (SIZE * 4).div_ceil(256) * 256;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();

    let mut out = Vec::with_capacity((SIZE * SIZE) as usize);
    for y in 0..SIZE {
        let base = (y * row) as usize;
        for x in 0..SIZE as usize {
            let at = base + x * 4;
            out.push([
                data[at] as f32 / 255.0,
                data[at + 1] as f32 / 255.0,
                data[at + 2] as f32 / 255.0,
                data[at + 3] as f32 / 255.0,
            ]);
        }
    }
    out
}

/// Projects the mesh's own XY onto a UV layer, over a two-metre square.
///
/// The simplest projection there is, and enough: what is being checked is that
/// a coordinate reaches a pixel, not that any particular unwrap is good.
fn planar_uvs(mut mesh: kreate_mesh::MeshData, repeat: f32) -> kreate_mesh::MeshData {
    use kreate_mesh::attribute::{names, AttrData, Domain};
    let positions = mesh.positions().to_vec();
    let uvs: Vec<glam::Vec2> = mesh
        .corner_verts()
        .iter()
        .map(|v| {
            let p = positions[*v as usize];
            glam::Vec2::new(p.x * 0.5 + 0.5, p.y * 0.5 + 0.5) * repeat
        })
        .collect();
    mesh.attributes_mut()
        .insert(names::UV, Domain::Corner, AttrData::F32x2(uvs.into()));
    mesh
}

/// Half red, half blue, split down the middle.
fn two_tone(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height) as usize * 4);
    for _ in 0..height {
        for x in 0..width {
            let left = x < width / 2;
            out.extend_from_slice(if left {
                &[255u8, 0, 0, 255]
            } else {
                &[0u8, 0, 255, 255]
            });
        }
    }
    out
}

#[test]
fn a_texture_reaches_the_picture_and_lands_the_right_way_round() {
    // **What all of this was for.** A material samples an image, the image
    // reaches the frame, and the half of it that is red is on the side the
    // surface coordinates say it is.
    //
    // Nothing above would catch a coordinate stuck at zero: the shader compiles,
    // the layouts agree, the atlas uploads, and every fragment samples the same
    // texel. The frame comes out one flat colour and every other test passes.
    let gpu = gpu_or_skip!(gpu());
    let (graph, _doc) = textured();
    let pixels = two_tone(16, 16);
    let frame = shoot(
        &gpu,
        &graph,
        &[kreate_render::material::SourceImage {
            kind: kreate_render::material::TextureKind::Color,
            width: 16,
            height: 16,
            pixels: &pixels,
        }],
    );

    // The middle row, left quarter against right quarter. A quarter in from
    // each edge, so the quad's own silhouette is not what is being measured.
    let size = 48usize;
    let row = size / 2;
    let left = frame[row * size + size / 4];
    let right = frame[row * size + 3 * size / 4];
    assert!(
        left[0] > left[2] + 0.1,
        "the left of the frame is {left:?}, which is not the red half"
    );
    assert!(
        right[2] > right[0] + 0.1,
        "the right of the frame is {right:?}, which is not the blue half"
    );
}

#[test]
fn a_material_with_no_texture_is_the_colour_its_constants_say() {
    // The other half of the claim above: a frame that differs across its width
    // only because a texture is there. Without this, a shader that happened to
    // vary with position would pass the test above for the wrong reason.
    let gpu = gpu_or_skip!(gpu());
    let graph = ShaderGraph::standard(
        "Flat",
        kreate_math::SceneRgb::new(0.6, 0.1, 0.1),
        0.0,
        0.5,
    );
    let frame = shoot(&gpu, &graph, &[]);
    let size = 48usize;
    let row = size / 2;
    let left = frame[row * size + size / 4];
    let right = frame[row * size + 3 * size / 4];
    for c in 0..3 {
        assert!(
            (left[c] - right[c]).abs() < 0.02,
            "a flat material is not flat across the frame: {left:?} against {right:?}"
        );
    }
    // And it is the colour that was asked for: more red than blue.
    assert!(left[0] > left[2] + 0.1, "the frame is {left:?}");
}

#[test]
fn the_atlas_decodes_srgb_rather_than_taking_bytes_as_light() {
    // **The one claim that distinguishes a correct texture from a washed-out
    // one**, and no picture test above can see it: every assertion so far is a
    // comparison between two halves of a frame, and a missing transfer function
    // moves both halves the same way.
    //
    // A mid-grey byte is 128. What it *means* is 0.216 of the light a white
    // surface reflects, not 0.502 — the encoding spends its code values where
    // the eye does. Taking the byte as the value makes everything textured too
    // bright, most of all in the shadows, and it looks like a lighting problem.
    // The atlas is an sRGB format so the sampler does this, exactly and in the
    // filter's own precision.
    //
    // **What the frame is, and why three greys.** A pixel here is not the
    // texel: it is `gain * texel + rest`, where the gain is the lighting and
    // the `rest` is the specular lobe, which does not scale with base colour.
    // Two attempts got a wrong answer from a right renderer before this one —
    // a ratio against white clips at the top of `Standard`'s range, and a ratio
    // between two greys is thrown off by `rest`. Evenly spaced code values
    // remove both: differencing cancels `rest`, and a ratio of differences
    // cancels the gain, leaving nothing but the shape of the transfer function.
    //
    // Three bytes a quarter apart. sRGB bends upward, so the upper step is
    // nearly twice the lower one. Bytes taken as light are a straight line and
    // the two steps are equal — and no lighting, exposure or clip can turn one
    // of those numbers into the other.
    let gpu = gpu_or_skip!(gpu());
    let (graph, _doc) = textured();
    let grey = |value: u8| {
        let pixels = [value, value, value, 255].repeat(16 * 16);
        let frame = shoot(
            &gpu,
            &graph,
            &[kreate_render::material::SourceImage {
                kind: kreate_render::material::TextureKind::Color,
                width: 16,
                height: 16,
                pixels: &pixels,
            }],
        );
        let size = 48usize;
        frame[size / 2 * size + size / 2]
    };
    let (low, mid, high) = (grey(64), grey(128), grey(192));

    let linear = |byte: f32| kreate_math::srgb_to_linear(byte / 255.0);
    let want = (linear(192.0) - linear(128.0)) / (linear(128.0) - linear(64.0));
    for c in 0..3 {
        assert!(
            low[c] > 1e-4 && high[c] < 0.99,
            "channel {c} is against a clip ({} and {}), so the steps say nothing",
            low[c],
            high[c]
        );
        let steps = (high[c] - mid[c]) / (mid[c] - low[c]);
        assert!(
            (steps - want).abs() < 0.3,
            "channel {c}: the upper quarter of the range is {steps:.2} times the \
             lower one; sRGB says {want:.2} and bytes taken as light would say 1.00"
        );
    }
}

// ---------------------------------------------------------------------------
// Mips, data maps and normal maps.

/// A one-texel checkerboard: the worst case for a texture seen from far away.
fn checker(size: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let v = if (x + y) % 2 == 0 { 255 } else { 0 };
            out.extend_from_slice(&[v, v, v, 255]);
        }
    }
    out
}

#[test]
fn a_minified_texture_is_its_average_rather_than_noise() {
    // **What a mip chain is for.** A one-texel checkerboard repeated sixteen
    // times across a 48-pixel quad puts more than twenty texels under every
    // pixel. Read at level zero, each pixel lands on black or white by the
    // accident of where its centre falls, and the frame is static. With the
    // chain the sampler reads a level where the checks have already been
    // averaged, and the frame is a flat grey.
    let gpu = gpu_or_skip!(gpu());
    let (graph, _doc) = textured();
    let compiled = kreate_shader::compile(&graph).expect("compiles");
    let pixels = checker(64);
    let frame = shoot_with(
        &gpu,
        &compiled,
        &[kreate_render::material::SourceImage {
            kind: kreate_render::material::TextureKind::Color,
            width: 64,
            height: 64,
            pixels: &pixels,
        }],
        &Stage {
            repeat: 16.0,
            ..Stage::default()
        },
    );
    let size = 48usize;
    let row: Vec<f32> = (size / 4..3 * size / 4)
        .map(|x| frame[size / 2 * size + x][1])
        .collect();
    let lo = row.iter().cloned().fold(f32::MAX, f32::min);
    let hi = row.iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        hi - lo < 0.08,
        "a minified checkerboard ranges from {lo:.2} to {hi:.2} across one row: \
         the sampler is reading texels, not a level that averages them"
    );
    // And the average is half the *light*, not half the bytes — the chain is
    // built in light — which is well away from black.
    assert!(lo > 0.2, "the minified checkerboard went dark: {lo:.2}");
}

#[test]
fn a_data_map_is_read_as_the_numbers_it_holds() {
    // The same three greys the sRGB test uses, through the *data* atlas. There
    // the steps must come out equal: a roughness of 0.75 is half again as far
    // from 0.25 as 0.5 is, and no curve may bend it. Through the colour atlas
    // they would be nearly two to one — a roughness map loaded as a colour,
    // which is what this atlas exists to prevent.
    let gpu = gpu_or_skip!(gpu());
    let (graph, _doc) = textured();
    let compiled =
        kreate_shader::compile_with(&graph, |_| kreate_scene::ColorSpace::Raw).expect("compiles");
    assert!(
        compiled.wgsl.contains("material_data"),
        "a raw image was sampled from the colour atlas"
    );
    let grey = |value: u8| {
        let pixels = [value, value, value, 255].repeat(16 * 16);
        let frame = shoot_with(
            &gpu,
            &compiled,
            &[kreate_render::material::SourceImage {
                kind: kreate_render::material::TextureKind::Data,
                width: 16,
                height: 16,
                pixels: &pixels,
            }],
            &Stage::default(),
        );
        let size = 48usize;
        frame[size / 2 * size + size / 2]
    };
    let (low, mid, high) = (grey(64), grey(128), grey(192));
    for c in 0..3 {
        assert!(low[c] > 1e-4 && high[c] < 0.99, "channel {c} is against a clip");
        let steps = (high[c] - mid[c]) / (mid[c] - low[c]);
        assert!(
            (steps - 1.0).abs() < 0.25,
            "channel {c}: the steps are {steps:.2} to one; numbers read as numbers \
             are 1.00, and read through sRGB they would be nearly 2"
        );
    }
}

/// A material whose only texture is a normal map of one texel value.
fn normal_mapped(texel: [u8; 3]) -> (kreate_shader::Compiled, Vec<u8>) {
    let mut doc = kreate_scene::Document::new();
    let image = doc.add_image(kreate_scene::Image::new(
        "Normal",
        4,
        4,
        kreate_scene::PixelFormat::Rgba8,
    ));
    let material = kreate_scene::Material {
        base_color: kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
        roughness: 1.0,
        normal_image: Some(image),
        ..Default::default()
    };
    let graph = ShaderGraph::for_material(&material);
    let compiled =
        kreate_shader::compile_with(&graph, |_| kreate_scene::ColorSpace::Raw).expect("compiles");
    let pixels = [texel[0], texel[1], texel[2], 255].repeat(4 * 4);
    (compiled, pixels)
}

/// The centre of the frame under a sun leaning in from `toward`, with a
/// normal map of one texel — or none.
fn lit_centre(gpu: &Gpu, toward: glam::Vec3, texel: Option<[u8; 3]>) -> f32 {
    let stage = Stage {
        // A sun from the side the map may tilt toward, and no sky: under a
        // uniform sky every normal receives the same light, and a normal map
        // would be invisible by construction.
        lights: vec![kreate_render::lights::PlacedLight {
            falloff: kreate_render::lights::Falloff::Parallel,
            direction: (-toward - glam::Vec3::Z).normalize(),
            amount: [2.0; 3],
            cast_shadow: false,
            ..Default::default()
        }],
        world: [0.0; 3],
        repeat: 1.0,
    };
    let frame = match texel {
        Some(texel) => {
            let (compiled, pixels) = normal_mapped(texel);
            shoot_with(
                gpu,
                &compiled,
                &[kreate_render::material::SourceImage {
                    kind: kreate_render::material::TextureKind::Data,
                    width: 4,
                    height: 4,
                    pixels: &pixels,
                }],
                &stage,
            )
        }
        None => {
            let graph = ShaderGraph::standard(
                "Plain",
                kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
                0.0,
                1.0,
            );
            let compiled = kreate_shader::compile(&graph).expect("compiles");
            shoot_with(gpu, &compiled, &[], &stage)
        }
    };
    let size = 48usize;
    let p = frame[size / 2 * size + size / 2];
    (p[0] + p[1] + p[2]) / 3.0
}

#[test]
fn a_flat_normal_map_changes_nothing() {
    // (128, 128, 255) is straight up. A material wearing it must draw as the
    // material without it — or every normal map tilts the whole surface a
    // little, which reads as a lighting bug nobody can find.
    let gpu = gpu_or_skip!(gpu());
    let toward = glam::Vec3::X;
    let plain = lit_centre(&gpu, toward, None);
    let flat = lit_centre(&gpu, toward, Some([128, 128, 255]));
    assert!(plain > 0.05, "the fixture is dark: {plain}");
    assert!(
        (plain - flat).abs() < 0.02,
        "a flat normal map changed the surface: {plain:.3} without, {flat:.3} with"
    );
}

#[test]
fn a_normal_map_tilts_the_surface_the_way_its_channels_say() {
    // Red is +U and green is +V, which this quad maps to +X and +Y. A map
    // tilted toward the light faces it and is brighter; tilted away, darker.
    // Checked on both axes and both signs, because the frame is solved from
    // derivatives and a sign error on either axis is a map lit from the wrong
    // side — the classic inside-out normal map, which looks embossed where it
    // was painted engraved.
    let gpu = gpu_or_skip!(gpu());
    for (axis, toward, lean, away) in [
        // Twenty-five degrees rather than forty-five: this camera sees the
        // quad from an angle, and a normal tipped far enough away from it
        // faces away from the viewer — which the BRDF rightly draws black, and
        // which would make "darker" true for the wrong reason.
        ("red", glam::Vec3::X, [182, 128, 238], [74, 128, 238]),
        ("green", glam::Vec3::Y, [128, 182, 238], [128, 74, 238]),
    ] {
        let flat = lit_centre(&gpu, toward, Some([128, 128, 255]));
        let into = lit_centre(&gpu, toward, Some(lean));
        let from = lit_centre(&gpu, toward, Some(away));
        assert!(
            into > flat + 0.03 && from < flat - 0.03,
            "{axis}: tilted toward the light {into:.3}, flat {flat:.3}, tilted away {from:.3}"
        );
    }
}


