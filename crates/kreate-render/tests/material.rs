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
    let s = material_surface(in.uv);
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
                width: 16,
                height: 16,
                pixels: &big,
            },
            SourceImage {
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
        ],
    })
}
