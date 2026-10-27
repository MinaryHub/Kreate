//! What a user gets by pressing the buttons.
//!
//! Everything else about lighting is checked a layer at a time: the datablock,
//! the join, the packing, the shading, the display transform. All of it passed
//! while the shipped program drew a black screen, because nothing asked the one
//! question a user asks — *add a cube, add a lamp, switch to rendered: is there
//! a picture?*
//!
//! So this builds the document through the operators the menu runs, with the
//! parameters the menu sends, and renders it the way the frame loop does. It is
//! the only test here that would notice a default nobody can see by.

use kreate::scenelights;
use kreate_ops::operator::{Context, Mode};
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_render::viewport::{Draw, ShadingMode, Viewport};
use kreate_render::{Camera, GpuOptions, RenderMeshCpu};
use kreate_scene::Document;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    let opts = GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
    Some(Gpu {
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
    })
}

const SIZE: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Runs an operator the way the header menu does.
fn menu(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    op: &str,
    pairs: &[(&'static str, PropertyValue)],
) {
    let mut params = PropertyMap::new();
    for (k, v) in pairs {
        params.set(k, v.clone());
    }
    history
        .dispatch(registry, doc, &Context { mode: Mode::Object }, op, &params)
        .expect("dispatch");
}

/// A cube and a lamp, added exactly as the menu adds them.
fn a_scene_a_user_would_build(kind: u32) -> Document {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();

    // The header's Add menu sends `shape` for a cube and `kind` for a light,
    // and nothing else: every other parameter is the operator's default. That
    // is the point — the defaults are what is on trial.
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_primitive",
        &[
            ("size", PropertyValue::Float(2.0)),
            ("shape", PropertyValue::Enum(1)),
        ],
    );
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_light",
        &[("kind", PropertyValue::Enum(kind))],
    );
    doc
}

/// Renders the document's meshes under its lights, as the frame loop does.
fn shoot(gpu: &Gpu, doc: &Document) -> Vec<[f32; 4]> {
    shoot_from(gpu, doc, Camera::default())
}

/// The same, from a camera the caller chose.
///
/// Its own function because *only* the camera moving is a case worth being able
/// to render: everything else about a frame is set by a call that also writes
/// the shading uniform, so a uniform that stopped following the camera would be
/// invisible to a test that changed anything else at the same time.
fn shoot_from(gpu: &Gpu, doc: &Document, camera: Camera) -> Vec<[f32; 4]> {
    shoot_with(gpu, doc, camera, false)
}

/// The same, path traced: the scene handed to the tracer the way the frame
/// loop hands it over, and a few samples summed.
fn shoot_traced(gpu: &Gpu, doc: &Document, camera: Camera) -> Vec<[f32; 4]> {
    shoot_with(gpu, doc, camera, true)
}

fn shoot_with(gpu: &Gpu, doc: &Document, camera: Camera, traced: bool) -> Vec<[f32; 4]> {
    shoot_mode(gpu, doc, camera, traced, false)
}

fn shoot_mode(
    gpu: &Gpu,
    doc: &Document,
    camera: Camera,
    traced: bool,
    preview: bool,
) -> Vec<[f32; 4]> {
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = if preview {
        ShadingMode::Material
    } else {
        ShadingMode::Rendered
    };
    viewport.path_trace = traced;
    viewport.show_grid = false;
    viewport.show_ground_shadow = false;
    viewport.show_wireframe = false;

    // The same three calls `main` makes, in the same order, from the same
    // functions. A test that assembled the lights itself would be checking its
    // own arithmetic. The material preview's are its own.
    let light = preview.then(|| scenelights::material_preview(camera.rotation));
    viewport.set_world(
        light
            .as_ref()
            .map_or_else(|| scenelights::world_radiance(doc), |p| p.world),
    );
    viewport.set_display(
        &gpu.queue,
        kreate_render::display::Display {
            view: kreate_math::ViewTransform::AgX,
            exposure: light.as_ref().map_or_else(
                || scenelights::shot_exposure(doc, 24.0, camera.pivot) as f32,
                |p| p.exposure,
            ),
        },
    );
    let lights = light
        .as_ref()
        .map_or_else(|| scenelights::placed_lights(doc), |p| p.lights.clone());
    viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &lights);

    // The materials, compiled the way the frame loop compiles them. Without
    // this the rendered pass has nothing to shade with and draws no geometry —
    // which is what these tests found the moment the viewport stopped carrying
    // a built-in surface, and is the reason it has no fallback: a fallback
    // would be a second `material_sample` for §3's fifth decision to have
    // ruled out.
    let materials = kreate::materials::Materials::compile(doc);
    let pixels: Vec<Vec<kreate::materials::TextureImage>> = materials
        .compiled
        .iter()
        .map(|c| kreate::materials::pixels_for(doc, c))
        .collect();
    let images: Vec<Vec<kreate_render::material::SourceImage<'_>>> = pixels
        .iter()
        .map(|set| set.iter().map(|i| i.source()).collect())
        .collect();
    let sources: Vec<kreate_render::viewport::MaterialSource<'_>> = materials
        .compiled
        .iter()
        .zip(&images)
        .map(|(c, images)| kreate_render::viewport::MaterialSource {
            wgsl: &c.wgsl,
            constants: &c.constants,
            image_slots: &c.image_slots,
            images,
        })
        .collect();
    viewport.set_materials(&gpu.device, &gpu.queue, &sources);

    // The environment, through the same two calls the frame loop makes. A
    // no-op for a flat world, which is most of the tests here — and present in
    // all of them so that "the frame loop does this" stays true of the harness.
    if let Some(image) = kreate::environment::background_image(doc) {
        viewport.set_environment(
            &gpu.device,
            &gpu.queue,
            image.width(),
            image.height(),
            &kreate::environment::pixels_of(image),
        );
    }
    let (rotation, strength) = doc
        .active_world()
        .and_then(|id| doc.world(id))
        .map(|w| (w.rotation as f32, w.strength as f32))
        .unwrap_or((0.0, 1.0));
    viewport.set_world_shape(&gpu.queue, rotation, strength);

    // Through `world_transform`, like the frame loop: an object's own
    // transform is its parent's space and a lamp under a rig would be placed
    // wrongly by anything else.
    let meshes: Vec<_> = doc
        .objects()
        .filter_map(|(id, o)| match o.data {
            kreate_scene::document::ObjectData::Mesh(mesh) => Some((
                doc.mesh(mesh)?,
                doc.world_transform(id),
                // Which compiled material this object wears. The harness drew
                // everything with index zero until a texture made the
                // difference visible — and index zero is the *default* grey, so
                // every material the document held was being ignored while
                // every test here passed.
                materials.index_of(doc, id),
                // And one per mesh slot, the way the frame loop does it, so a
                // mesh whose faces carry different materials draws in more
                // than one of them here too.
                materials.slots_of(doc, id),
            )),
            _ => None,
        })
        .collect();
    let uploaded: Vec<_> = meshes
        .iter()
        .map(|(mesh, transform, material, slots)| {
            let cpu = RenderMeshCpu::extract(mesh);
            (
                kreate_render::GpuMesh::upload(
                    &gpu.device,
                    &viewport.layouts,
                    &cpu,
                    glam::Vec4::ONE,
                ),
                *transform,
                *material,
                slots.clone(),
            )
        })
        .collect();
    let mut draws: Vec<Draw> = uploaded
        .iter()
        .map(|(mesh, transform, material, slots)| {
            Draw::new(mesh, *transform)
                .with_material(*material)
                .with_materials(slots)
        })
        .collect();

    // The lamp and camera glyphs, the way the frame loop hands them over.
    let (marker_lines, marker_points) = kreate::glyphs::markers_for(doc);
    viewport.set_markers(
        &gpu.device,
        &gpu.queue,
        camera.eye(),
        &marker_lines,
        &marker_points,
    );
    if traced {
        // As `main` builds it: the materials' code and images from
        // `for_tracer`, each object naming its compiled material.
        let traced_materials = materials.for_tracer(doc);
        let extracted: Vec<_> = meshes
            .iter()
            .map(|(mesh, transform, material, slots)| {
                let cpu = RenderMeshCpu::extract(mesh);
                // As `Materials::runs_of` resolves them.
                let runs: Vec<(u32, u32)> = cpu
                    .submeshes
                    .iter()
                    .map(|r| {
                        (
                            r.start,
                            slots.get(r.slot as usize).copied().unwrap_or(*material),
                        )
                    })
                    .collect();
                let positions: Vec<[f32; 3]> = cpu.positions.iter().map(|p| p.0).collect();
                let normals: Vec<kreate_math::Vec3> = cpu
                    .normals
                    .iter()
                    .map(|n| kreate_render::mesh_gpu::oct_decode(*n))
                    .collect();
                (
                    positions,
                    cpu.uvs.clone(),
                    normals,
                    cpu.indices.clone(),
                    *transform,
                    *material,
                    runs,
                )
            })
            .collect();
        let pt_meshes: Vec<_> = extracted
            .iter()
            .map(
                |(positions, uvs, normals, indices, transform, surface, runs)| {
                    kreate_render::pathtrace::PtMesh {
                        positions,
                        uvs,
                        normals,
                        colors: &[],
                        indices,
                        transform: *transform,
                        surface: *surface,
                        runs,
                    }
                },
            )
            .collect();
        let pt_images: Vec<_> = traced_materials
            .images
            .iter()
            .map(|i| kreate_render::pathtrace::PtImage {
                width: i.width,
                height: i.height,
                pixels: &i.pixels,
                color: i.kind == kreate_render::material::TextureKind::Color,
            })
            .collect();
        viewport.set_path_trace_scene(
            &gpu.device,
            &gpu.queue,
            &pt_meshes,
            &traced_materials.surfaces,
            &pt_images,
            &traced_materials.materials(),
        );
        assert!(viewport.path_tracing(), "the tracer took no scene");
        // Samples before the one read back.
        for _ in 0..15 {
            viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
            let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("sample"),
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            viewport.render(&mut encoder, &view, &draws);
            gpu.queue.submit([encoder.finish()]);
        }
    }
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
        let line = &data[base..base + (SIZE * 4) as usize];
        for x in 0..SIZE as usize {
            out.push([
                line[x * 4] as f32 / 255.0,
                line[x * 4 + 1] as f32 / 255.0,
                line[x * 4 + 2] as f32 / 255.0,
                line[x * 4 + 3] as f32 / 255.0,
            ]);
        }
    }
    out
}

/// The frame loop over several frames, the camera moving between them, with
/// the light list the application hands over — culled at the subject — and
/// the last frame read back.
fn shoot_path(gpu: &Gpu, doc: &Document, cameras: &[Camera]) -> Vec<[f32; 4]> {
    let camera = cameras[0];
    let mut viewport = Viewport::new(&gpu.device, FORMAT, SIZE, SIZE);
    viewport.shading = ShadingMode::Rendered;
    viewport.show_grid = false;
    viewport.show_ground_shadow = false;
    viewport.show_wireframe = false;

    // The same three calls `main` makes, in the same order, from the same
    // functions. A test that assembled the lights itself would be checking its
    // own arithmetic.
    viewport.set_world(scenelights::world_radiance(doc));
    viewport.set_display(
        &gpu.queue,
        kreate_render::display::Display {
            view: kreate_math::ViewTransform::AgX,
            exposure: scenelights::shot_exposure(doc, 24.0, camera.pivot) as f32,
        },
    );
    let lights = scenelights::placed_lights(doc);
    viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &lights);

    // The materials, compiled the way the frame loop compiles them. Without
    // this the rendered pass has nothing to shade with and draws no geometry —
    // which is what these tests found the moment the viewport stopped carrying
    // a built-in surface, and is the reason it has no fallback: a fallback
    // would be a second `material_sample` for §3's fifth decision to have
    // ruled out.
    let materials = kreate::materials::Materials::compile(doc);
    let pixels: Vec<Vec<kreate::materials::TextureImage>> = materials
        .compiled
        .iter()
        .map(|c| kreate::materials::pixels_for(doc, c))
        .collect();
    let images: Vec<Vec<kreate_render::material::SourceImage<'_>>> = pixels
        .iter()
        .map(|set| set.iter().map(|i| i.source()).collect())
        .collect();
    let sources: Vec<kreate_render::viewport::MaterialSource<'_>> = materials
        .compiled
        .iter()
        .zip(&images)
        .map(|(c, images)| kreate_render::viewport::MaterialSource {
            wgsl: &c.wgsl,
            constants: &c.constants,
            image_slots: &c.image_slots,
            images,
        })
        .collect();
    viewport.set_materials(&gpu.device, &gpu.queue, &sources);

    // The environment, through the same two calls the frame loop makes. A
    // no-op for a flat world, which is most of the tests here — and present in
    // all of them so that "the frame loop does this" stays true of the harness.
    if let Some(image) = kreate::environment::background_image(doc) {
        viewport.set_environment(
            &gpu.device,
            &gpu.queue,
            image.width(),
            image.height(),
            &kreate::environment::pixels_of(image),
        );
    }
    let (rotation, strength) = doc
        .active_world()
        .and_then(|id| doc.world(id))
        .map(|w| (w.rotation as f32, w.strength as f32))
        .unwrap_or((0.0, 1.0));
    viewport.set_world_shape(&gpu.queue, rotation, strength);

    // Through `world_transform`, like the frame loop: an object's own
    // transform is its parent's space and a lamp under a rig would be placed
    // wrongly by anything else.
    let meshes: Vec<_> = doc
        .objects()
        .filter_map(|(id, o)| match o.data {
            kreate_scene::document::ObjectData::Mesh(mesh) => Some((
                doc.mesh(mesh)?,
                doc.world_transform(id),
                // Which compiled material this object wears. The harness drew
                // everything with index zero until a texture made the
                // difference visible — and index zero is the *default* grey, so
                // every material the document held was being ignored while
                // every test here passed.
                materials.index_of(doc, id),
                // And one per mesh slot, the way the frame loop does it, so a
                // mesh whose faces carry different materials draws in more
                // than one of them here too.
                materials.slots_of(doc, id),
            )),
            _ => None,
        })
        .collect();
    let uploaded: Vec<_> = meshes
        .iter()
        .map(|(mesh, transform, material, slots)| {
            let cpu = RenderMeshCpu::extract(mesh);
            (
                kreate_render::GpuMesh::upload(
                    &gpu.device,
                    &viewport.layouts,
                    &cpu,
                    glam::Vec4::ONE,
                ),
                *transform,
                *material,
                slots.clone(),
            )
        })
        .collect();
    let mut draws: Vec<Draw> = uploaded
        .iter()
        .map(|(mesh, transform, material, slots)| {
            Draw::new(mesh, *transform)
                .with_material(*material)
                .with_materials(slots)
        })
        .collect();

    // The lamp and camera glyphs, the way the frame loop hands them over.
    let (marker_lines, marker_points) = kreate::glyphs::markers_for(doc);
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
    for (n, camera) in cameras.iter().enumerate() {
        viewport.set_markers(
            &gpu.device,
            &gpu.queue,
            camera.eye(),
            &marker_lines,
            &marker_points,
        );
        let lights = scenelights::culled_lights(doc, camera.pivot);
        viewport.set_lights(&gpu.device, &gpu.queue, camera.eye(), &lights);
        viewport.prepare(&gpu.device, &gpu.queue, camera, &mut draws);
        viewport.render(&mut encoder, &view, &draws);
        if n + 1 < cameras.len() {
            gpu.queue.submit([encoder.finish()]);
            encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        }
    }

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
        let line = &data[base..base + (SIZE * 4) as usize];
        for x in 0..SIZE as usize {
            out.push([
                line[x * 4] as f32 / 255.0,
                line[x * 4 + 1] as f32 / 255.0,
                line[x * 4 + 2] as f32 / 255.0,
                line[x * 4 + 3] as f32 / 255.0,
            ]);
        }
    }
    out
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

#[test]
fn a_new_document_renders_a_picture_with_nothing_added() {
    // The end of a long thread, as one assertion: open the application, press
    // `Z`, see something.
    //
    // A new document used to contain a cube and nothing else, so the rendered
    // mode was correctly and uselessly black — nothing was lighting anything,
    // and until the Add menu shipped there was no way to change that. The fix
    // is not for the renderer to invent light, which §2.1 is emphatic about,
    // but for a new document to be a scene with some in it.
    let gpu = gpu_or_skip!(gpu());
    let mut doc = Document::new();
    kreate::startup::sample_scene(&mut doc);

    let pixels = shoot(&gpu, &doc);
    let brightest = pixels.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
    assert!(
        brightest > 0.1,
        "a new document rendered black: brightest {brightest}"
    );
    let saturated = pixels.iter().filter(|p| p[0] >= 0.99).count();
    let lit = pixels.iter().filter(|p| p[0] > 0.01).count();
    assert!(
        saturated * 2 < lit.max(1),
        "a new document blew out: {saturated} of {lit} lit pixels clipped"
    );

    // And every panel the properties editor offers has something behind it. A
    // world nobody can reach is a tab that never appears; a camera nobody can
    // reach is an exposure nobody can set.
    assert!(doc.active_world().is_some(), "a new document has no world");
    assert!(
        doc.cameras().next().is_some(),
        "a new document has no camera"
    );
    assert!(
        doc.objects()
            .any(|(_, o)| matches!(o.data, kreate_scene::document::ObjectData::Light(_))),
        "a new document has no lamp"
    );
    // The cube is what is selected, because it is what somebody opening the
    // application is going to do something to.
    let active = doc.selection.active.and_then(|id| doc.object(id));
    assert!(
        matches!(
            active.map(|o| &o.data),
            Some(kreate_scene::document::ObjectData::Mesh(_))
        ),
        "a new document does not start on the cube"
    );
}

#[test]
fn adding_a_lamp_from_the_menu_lights_the_scene() {
    // The whole complaint in one test. Every kind, because the four have
    // different units and different defaults, and "1000" is a bright room in
    // candela and a dull afternoon in lux — a single default shared between
    // them would leave three of the four invisible.
    let gpu = gpu_or_skip!(gpu());
    for (kind, name) in [(0u32, "point"), (1, "sun"), (2, "spot"), (3, "area")] {
        let doc = a_scene_a_user_would_build(kind);
        let pixels = shoot(&gpu, &doc);

        let brightest = pixels.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
        assert!(
            brightest > 0.1,
            "a {name} added from the menu left the picture black: brightest {brightest}"
        );

        // And not blown out either, which is the other way a default can be
        // useless: a picture where everything is white shows no shape at all.
        let saturated = pixels.iter().filter(|p| p[0] >= 0.99).count();
        let lit = pixels.iter().filter(|p| p[0] > 0.01).count();
        assert!(
            saturated * 2 < lit.max(1),
            "a {name} blew the picture out: {saturated} of {lit} lit pixels clipped"
        );
    }
}

/// The material preview lights a material with no lamp in the scene, and
/// lights it as a picture should be lit: the cube seen, its faces told apart,
/// nothing blown to white.
#[test]
fn the_material_preview_shows_a_material_without_a_lamp() {
    let gpu = match gpu() {
        Some(gpu) => gpu,
        None => return,
    };
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_primitive",
        &[
            ("size", PropertyValue::Float(2.0)),
            ("shape", PropertyValue::Enum(1)),
        ],
    );
    let shot = shoot_mode(&gpu, &doc, Camera::default(), false, true);
    if let Ok(dir) = std::env::var("KREATE_SHOT_DIR") {
        let bytes: Vec<u8> = shot
            .iter()
            .flat_map(|c| c.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8))
            .collect();
        let _ = image::RgbaImage::from_raw(SIZE, SIZE, bytes)
            .map(|i| i.save(std::path::Path::new(&dir).join("material-preview.png")));
    }
    let corner = shot[0];
    let cube: Vec<f32> = shot
        .iter()
        .filter(|p| (0..3).any(|i| (p[i] - corner[i]).abs() > 0.02))
        .map(|p| (p[0] + p[1] + p[2]) / 3.0)
        .collect();
    assert!(
        cube.len() > 200,
        "the cube is hardly there: {} pixels",
        cube.len()
    );
    let (lo, hi) = cube
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let mean = cube.iter().sum::<f32>() / cube.len() as f32;
    assert!(hi < 0.97, "blown out: {hi}");
    assert!(mean > 0.2 && mean < 0.85, "the cube's mean is {mean}");
    assert!(hi - lo > 0.15, "its faces are not told apart: {lo}..{hi}");
}

#[test]
fn a_scene_with_no_lamp_is_black_and_that_is_why_the_menu_matters() {
    // The control, and the reason the test above is worth its runtime: a cube
    // on its own in rendered mode is black, correctly — there is nothing
    // lighting it. That was the shipped experience, because the menu had no way
    // to add the thing that fixes it.
    let gpu = gpu_or_skip!(gpu());
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_primitive",
        &[
            ("size", PropertyValue::Float(2.0)),
            ("shape", PropertyValue::Enum(1)),
        ],
    );

    let pixels = shoot(&gpu, &doc);
    let brightest = pixels.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
    assert!(
        brightest < 0.02,
        "an unlit scene was not black: {brightest}"
    );
}

#[test]
fn turning_a_lamp_down_moves_the_meter_and_not_the_picture() {
    // The panel's end of it, and a result worth stating plainly because it
    // looks like a bug: with one lamp in the scene, turning its power down
    // changes nothing on screen. The meter follows it and the exposure opens by
    // exactly as much.
    //
    // That is what exposure *means*. A correctly exposed photograph of a room
    // does not say whether the room was lit by a bright lamp or a dim one; what
    // it says is how the lamps compare with each other and how the light falls
    // off. So the thing to check is that the meter moved, not that the picture
    // did — and the picture not moving is the assertion, not the omission.
    let gpu = gpu_or_skip!(gpu());
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = a_scene_a_user_would_build(0);

    let before_lux =
        scenelights::metered_illuminance(&doc, kreate_math::DVec3::ZERO).expect("a lit scene");
    let bright = shoot(&gpu, &doc);
    menu(
        &registry,
        &mut history,
        &mut doc,
        "light.set_properties",
        &[
            ("kind", PropertyValue::Enum(0)),
            ("intensity", PropertyValue::Float(100.0)),
        ],
    );
    let after_lux =
        scenelights::metered_illuminance(&doc, kreate_math::DVec3::ZERO).expect("still lit");
    let dim = shoot(&gpu, &doc);

    // A tenth of the power is a tenth of the reading.
    assert!(
        (before_lux / after_lux - 10.0).abs() < 0.1,
        "{before_lux} then {after_lux}"
    );
    let a = bright.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
    let b = dim.iter().fold(0.0f32, |acc, p| acc.max(p[0]));
    assert!(
        (a - b).abs() < 0.02,
        "the meter did not keep up: {a} then {b}"
    );
}

#[test]
fn with_two_lamps_turning_one_down_changes_the_balance() {
    // What *is* visible, and the reason the panel is worth having. The meter
    // holds the overall level, so what an edit shows is the difference between
    // the lamps — which is what lighting a shot consists of.
    let gpu = gpu_or_skip!(gpu());
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = a_scene_a_user_would_build(0);

    // A second lamp on the other side, so the two light different faces.
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_light",
        &[
            ("kind", PropertyValue::Enum(0)),
            ("location", PropertyValue::Vec3([-4.0, 0.0, 1.0])),
        ],
    );
    let even = shoot(&gpu, &doc);

    // The second lamp is the active one, so the panel edits that one.
    menu(
        &registry,
        &mut history,
        &mut doc,
        "light.set_properties",
        &[
            ("kind", PropertyValue::Enum(0)),
            ("intensity", PropertyValue::Float(20.0)),
        ],
    );
    let lopsided = shoot(&gpu, &doc);

    let moved = even
        .iter()
        .zip(&lopsided)
        .filter(|(a, b)| (a[0] - b[0]).abs() > 0.05)
        .count();
    assert!(
        moved > 50,
        "turning one of two lamps down changed {moved} pixels"
    );
}

// ---------------------------------------------------------------------------
// The environment.
//
// §7.7's gate, applied to the pillar's last piece: an HDRI that can be loaded,
// saved, prefiltered and shaded is a feature the user does not have until
// picking one puts a sky on the screen. Everything below the interface has its
// own tests and all of them passed while there was no way to choose an image.

/// An equirectangular sky: bright on one side of the horizon, dark elsewhere.
///
/// Not uniform, deliberately. A uniform sky would make "the environment is
/// lighting the scene" and "the flat colour is" the same picture, which is the
/// identity the renderer's tests are *for* — here the point is that a direction
/// survives all the way to the screen.
fn a_sky(width: u32, height: u32) -> kreate_scene::Image {
    let mut image =
        kreate_scene::Image::new("Sky", width, height, kreate_scene::PixelFormat::Rgba16F);
    image.color_space = kreate_scene::ColorSpace::Raw;
    let (across, down) = image.tile_counts();
    for ty in 0..down {
        for tx in 0..across {
            let tile = image.tile_mut(tx, ty).expect("in range");
            for row in 0..kreate_scene::TILE {
                for column in 0..kreate_scene::TILE {
                    let x = tx * kreate_scene::TILE + column;
                    let y = ty * kreate_scene::TILE + row;
                    if x >= width || y >= height {
                        continue;
                    }
                    let u = (x as f32 + 0.5) / width as f32;
                    // Bright over one half of the sphere and dark over the
                    // other. Split by longitude rather than by height, so that
                    // *which way the camera looks* is not part of the claim —
                    // the first version of this asked whether the top of the
                    // frame was brighter than the bottom, and the answer
                    // depended on where `Camera::default` happens to aim.
                    //
                    // Neither half is the flat grey the comparison below uses.
                    // The first version made the dark half 0.05, which *is*
                    // that grey — so half the frame matched by construction and
                    // the test failed for a reason that had nothing to do with
                    // the renderer.
                    let value = if u < 0.5 { 8.0 } else { 0.8 };
                    let bits = half::f16::from_f32(value).to_le_bytes();
                    let at = (row * kreate_scene::TILE + column) as usize * 8;
                    for c in 0..3 {
                        tile.bytes_mut()[at + c * 2..at + c * 2 + 2].copy_from_slice(&bits);
                    }
                    tile.bytes_mut()[at + 6..at + 8]
                        .copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                }
            }
        }
    }
    image
}

/// A new document with a sky chosen, through the operator the panel runs.
fn a_scene_with_a_sky() -> Document {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    // The starting scene, because that is what a user has when they open the
    // world tab: a cube, a lamp, a camera and a world to point at an image.
    let mut doc = Document::new();
    kreate::startup::sample_scene(&mut doc);

    // The application adds the datablock and the operator makes the link —
    // which is the split `pick_world_image` makes, for the reason an operator
    // cannot open a file.
    let id = doc.add_image(a_sky(128, 64));
    let uid = doc.image_uid(id).expect("assigned on add").raw();
    menu(
        &registry,
        &mut history,
        &mut doc,
        "world.set_image",
        &[("uid", PropertyValue::Text(uid.to_string()))],
    );
    doc
}

#[test]
fn choosing_a_sky_puts_a_sky_on_the_screen() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let doc = a_scene_with_a_sky();
    assert!(
        matches!(
            doc.world(doc.active_world().expect("a world"))
                .expect("it is there")
                .background,
            kreate_scene::Background::Image(_)
        ),
        "the operator did not make the link"
    );

    let sky = shoot(&gpu, &doc);

    // It is the sky and not the clear colour: the same document with a flat
    // background renders a different picture.
    let mut flat = a_scene_with_a_sky();
    let world = flat.active_world().expect("a world");
    flat.world_mut(world).expect("it is there").background =
        kreate_scene::Background::Flat(kreate_math::SceneRgb::new(0.05, 0.05, 0.06));
    let grey = shoot(&gpu, &flat);
    assert!(
        differing(&sky, &grey) > (SIZE * SIZE / 4) as usize,
        "only {} pixels of {} differ between a sky and a grey world",
        differing(&sky, &grey),
        SIZE * SIZE
    );

    // And it is *oriented*: turning the world half a turn swaps the bright half
    // of the sphere for the dark one, so the picture has to change.
    //
    // Asked this way rather than "the top of the frame is brighter than the
    // bottom", which is what this was first — that answer depends on where
    // `Camera::default` happens to aim, and a test that depends on a default
    // nobody chose for it is a test that breaks for an unrelated reason.
    let mut turned = a_scene_with_a_sky();
    let world = turned.active_world().expect("a world");
    turned.world_mut(world).expect("it is there").rotation = std::f64::consts::PI;
    let other_way = shoot(&gpu, &turned);
    assert!(
        differing(&sky, &other_way) > (SIZE * SIZE / 8) as usize,
        "turning the world changed only {} pixels of {}",
        differing(&sky, &other_way),
        SIZE * SIZE
    );
}

/// How many pixels of two shots differ enough to see.
fn differing(a: &[[f32; 4]], b: &[[f32; 4]]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(p, q)| (p[0] - q[0]).abs() > 0.02)
        .count()
}

#[test]
fn turning_the_sky_off_again_goes_back_to_a_colour() {
    // The other half of the control, and the one that is easy to leave out: a
    // user who picks a sky to look at it has to be able to stop.
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = a_scene_with_a_sky();

    menu(
        &registry,
        &mut history,
        &mut doc,
        "world.set_image",
        &[("uid", PropertyValue::Text(String::new()))],
    );
    assert!(matches!(
        doc.world(doc.active_world().expect("a world"))
            .expect("it is there")
            .background,
        kreate_scene::Background::Flat(_)
    ));
    // The image stays in the document with nothing pointing at it, which is
    // what an unused datablock is — and what makes undoing the link safe.
    assert_eq!(doc.image_count(), 1);
}

#[test]
fn a_world_image_that_is_not_in_the_document_is_refused() {
    // A uid naming an image that is not here comes from a script, or from a
    // redo whose datablock has since gone. The honest answer is to refuse
    // rather than to make a background that points at nothing — and nothing
    // else asked, because the application only ever passes a uid it has just
    // been handed.
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    let world = doc.add_world(kreate_scene::World::new("Sky"));
    doc.set_active_world(Some(world));

    let mut params = PropertyMap::new();
    params.set("uid", PropertyValue::Text("123456".into()));
    let outcome = history.dispatch(
        &registry,
        &mut doc,
        &Context { mode: Mode::Object },
        "world.set_image",
        &params,
    );
    // Cancelled, not an error and not a success: the operator ran and decided
    // there was nothing to do, which is what an unresolvable reference is.
    assert!(outcome.is_ok(), "{outcome:?}");
    assert!(
        matches!(
            doc.world(world).expect("still there").background,
            kreate_scene::Background::Flat(_)
        ),
        "the world points at an image that is not in the document"
    );

    // And a uid that is not a number at all, which is what a hand-edited file
    // or a mistyped script sends.
    let mut params = PropertyMap::new();
    params.set("uid", PropertyValue::Text("not a number".into()));
    assert!(history
        .dispatch(
            &registry,
            &mut doc,
            &Context { mode: Mode::Object },
            "world.set_image",
            &params,
        )
        .is_ok());
    assert!(matches!(
        doc.world(world).expect("still there").background,
        kreate_scene::Background::Flat(_)
    ));
}

#[test]
fn the_sky_follows_the_camera() {
    // The background's rays come from the camera's inverse, which `prepare`
    // writes every frame. Everything else in that uniform is written by a call
    // that changes something else as well — so a uniform that stopped following
    // the camera would be invisible to every other test here, and was.
    //
    // Two views of one document, with nothing else different.
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let doc = a_scene_with_a_sky();

    let mut turned = Camera::default();
    turned.orbit(400.0, 0.0);
    let one_way = shoot_from(&gpu, &doc, Camera::default());
    let other_way = shoot_from(&gpu, &doc, turned);
    assert!(
        differing(&one_way, &other_way) > (SIZE * SIZE / 8) as usize,
        "turning the camera changed only {} pixels of {}",
        differing(&one_way, &other_way),
        SIZE * SIZE
    );
}

#[test]
fn the_worlds_strength_dims_the_sky_as_well_as_the_scene() {
    // A dimmer on the environment, so the background moves with it. Nothing set
    // a strength other than one, which is how the background came to ignore it
    // and nothing noticed.
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let full = a_scene_with_a_sky();
    let bright = shoot(&gpu, &full);

    let mut dim = a_scene_with_a_sky();
    let world = dim.active_world().expect("a world");
    dim.world_mut(world).expect("it is there").strength = 0.1;
    let dark = shoot(&gpu, &dim);

    // The frame as a whole is darker, and by enough that an exposure meter
    // chasing it could not have made up the difference.
    let mean = |p: &[[f32; 4]]| p.iter().map(|v| v[0]).sum::<f32>() / p.len() as f32;
    assert!(
        mean(&bright) > mean(&dark) + 0.05,
        "a tenth of the strength gave {} against {}",
        mean(&dark),
        mean(&bright)
    );
}

#[test]
fn a_texture_assigned_from_the_panel_reaches_the_picture() {
    // **§7.7's gate, for the material's texture.** The image node, the
    // compiler's layer assignment, the atlas, the surface coordinates and the
    // operator all have their own tests and all pass. This is the only one that
    // asks whether a person who picks a file sees anything change.
    //
    // The document is built by the operator the panel's button runs, and the
    // frame is taken by the harness that does what the frame loop does. Between
    // them there is nowhere for a picture to go missing without this noticing.
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    let mut doc = Document::new();
    kreate::startup::sample_scene(&mut doc);

    // The cube the starter document comes with, and a material on it — the
    // panel needs one to point at, which is why the tab does not appear
    // without one.
    let (mesh, object) = doc
        .objects()
        .find_map(|(id, o)| match o.data {
            kreate_scene::document::ObjectData::Mesh(mesh) => Some((mesh, id)),
            _ => None,
        })
        .expect("the starter document has a cube");
    let material = doc.add_material(kreate_scene::Material {
        name: "Painted".into(),
        base_color: kreate_math::SceneRgb::new(1.0, 1.0, 1.0),
        ..Default::default()
    });
    doc.set_mesh_slot(mesh, 0, Some(material));
    doc.selection.add(object);

    let before = shoot(&gpu, &doc);

    // Half red and half blue, which no shading can produce on its own — a
    // brightness change would also follow from the material simply existing.
    let mut image = kreate_scene::Image::new("Bricks", 64, 64, kreate_scene::PixelFormat::Rgba8);
    {
        let tile = image.tile_mut(0, 0).expect("one tile");
        let bytes = tile.bytes_mut();
        let side = kreate_scene::image::TILE as usize;
        for row in 0..side {
            for column in 0..side {
                let at = (row * side + column) * 4;
                let left = column < 32;
                bytes[at..at + 4].copy_from_slice(if left {
                    &[255u8, 0, 0, 255]
                } else {
                    &[0u8, 0, 255, 255]
                });
            }
        }
    }
    let id = doc.add_image(image);
    let uid = doc.image_uid(id).expect("an identity").raw().to_string();
    menu(
        &registry,
        &mut history,
        &mut doc,
        "material.set_image",
        &[("uid", PropertyValue::Text(uid))],
    );
    assert_eq!(
        doc.material(material)
            .expect("still there")
            .base_color_image,
        Some(id),
        "the operator did not make the link"
    );

    let after = shoot(&gpu, &doc);
    let changed = before
        .iter()
        .zip(&after)
        .filter(|(a, b)| (a[0] - b[0]).abs() + (a[2] - b[2]).abs() > 0.02)
        .count();
    assert!(
        changed > 200,
        "assigning a texture changed {changed} pixels, which is not a picture"
    );

    // And it is the texture and not a tint: the frame now holds both a reddish
    // pixel and a bluish one, which one colour cannot do.
    let reddish = after.iter().filter(|p| p[0] > p[2] + 0.05).count();
    let bluish = after.iter().filter(|p| p[2] > p[0] + 0.05).count();
    assert!(
        reddish > 50 && bluish > 50,
        "the frame has {reddish} red pixels and {bluish} blue ones, \
         so the image is not on the surface"
    );
}

#[test]
fn one_object_can_wear_two_materials_at_once() {
    // **What the draw split was for.** A face's `material_index` reached the
    // glTF exporter the day it was written and never reached the screen: the
    // renderer drew a whole object with one material, so a second slot was a
    // material nobody could see. Assigning one changed the file and not the
    // picture.
    //
    // A plane split down the middle, red on one half and blue on the other,
    // seen face on. One object, one mesh, one draw entry — and two colours.
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut doc = Document::new();
    kreate::startup::sample_scene(&mut doc);
    for (id, _) in doc
        .objects()
        .filter(|(_, o)| matches!(o.data, kreate_scene::document::ObjectData::Mesh(_)))
        .map(|(id, o)| (id, o.name.clone()))
        .collect::<Vec<_>>()
    {
        doc.remove_object(id);
    }

    // A two-by-two grid, so there are four faces to hand out.
    let mut plane = kreate_mesh::primitives::grid(2.0, 2, 2);
    let faces = plane.num_faces() as usize;
    assert_eq!(faces, 4, "the fixture needs four faces to split");
    plane.attributes_mut().insert(
        kreate_mesh::attribute::names::MATERIAL_INDEX,
        kreate_mesh::attribute::Domain::Face,
        // The left column on slot 0 and the right on slot 1. `grid` walks rows
        // then columns, so this is two faces each whichever way round it is.
        kreate_mesh::attribute::AttrData::I32(vec![0, 1, 0, 1].into()),
    );
    let mesh = doc.add_mesh(plane);
    let red = doc.add_material(kreate_scene::Material {
        name: "Red".into(),
        base_color: kreate_math::SceneRgb::new(0.8, 0.02, 0.02),
        ..Default::default()
    });
    let blue = doc.add_material(kreate_scene::Material {
        name: "Blue".into(),
        base_color: kreate_math::SceneRgb::new(0.02, 0.02, 0.8),
        ..Default::default()
    });
    doc.set_mesh_slots(mesh, vec![Some(red), Some(blue)]);
    let object = doc.add_object(kreate_scene::document::Object::new(
        "Plane",
        kreate_scene::document::ObjectData::Mesh(mesh),
    ));
    doc.selection.add(object);

    // Looking straight down at the plane, which lies in the XY plane — in
    // the viewport and in the path tracer, which is handed the slots too.
    let camera = Camera {
        distance: 4.0,
        ..Camera::default()
    };
    for (frame, how) in [
        (shoot_from(&gpu, &doc, camera), "drawn"),
        (shoot_traced(&gpu, &doc, camera), "traced"),
    ] {
        let reddish = frame.iter().filter(|p| p[0] > p[2] + 0.05).count();
        let bluish = frame.iter().filter(|p| p[2] > p[0] + 0.05).count();
        assert!(
            reddish > 300 && bluish > 300,
            "one object {how} {reddish} red pixels and {bluish} blue ones; \
             both halves have to be there, or the slots are not reaching it"
        );
    }
}

/// Shadows stay while the view moves. The new document, under its own lamp,
/// drawn frame after frame as an orbit turns it: the frame mid-orbit is the
/// frame a still view from the same place draws.
#[test]
fn the_new_documents_shadow_stays_while_the_view_moves() {
    let gpu = gpu_or_skip!(gpu());
    let mut doc = Document::new();
    kreate::startup::sample_scene(&mut doc);
    // A floor under the cube, as the user's scene has: something to fall on.
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    menu(
        &registry,
        &mut history,
        &mut doc,
        "object.add_primitive",
        &[
            ("shape", PropertyValue::Enum(0)),
            ("size", PropertyValue::Float(12.0)),
            ("location", PropertyValue::Vec3([0.0, 0.0, -1.0])),
        ],
    );
    let base = Camera::default();
    let at = |turn: f64| Camera {
        rotation: glam::DQuat::from_rotation_z(turn) * base.rotation,
        ..base
    };
    let still: Vec<Camera> = (0..20).map(|_| at(0.3)).collect();
    let still = shoot_path(&gpu, &doc, &still);
    let mut path: Vec<Camera> = (0..20).map(|_| at(0.0)).collect();
    path.extend((1..=6).map(|i| at(0.05 * i as f64)));
    let moving = shoot_path(&gpu, &doc, &path);
    let lum = |c: &[f32; 4]| c[0] + c[1] + c[2];
    let dark = |p: &Vec<[f32; 4]>| p.iter().filter(|c| lum(c) < 0.15).count();
    eprintln!("SHADOW still {} moving {}", dark(&still), dark(&moving));
    let differ = differing(&still, &moving);
    assert!(
        differ * 20 < still.len(),
        "{differ} pixels differ mid-orbit; dark {} still, {} moving",
        dark(&still),
        dark(&moving)
    );
}

/// The starter scene's lamp over one square face, seen face on, wearing a
/// material made of `graph`.
fn a_plane_wearing(graph: kreate_scene::shader::ShaderGraph) -> Vec<[f32; 4]> {
    a_plane_wearing_with(|_| graph)
}

/// The same, with the graph made in the document — for one that names an
/// image the document has to hold.
fn a_plane_wearing_with(
    make: impl FnOnce(&mut Document) -> kreate_scene::shader::ShaderGraph,
) -> Vec<[f32; 4]> {
    a_plane_shot(make, |_| {}, false)
}

/// The same, built from a material rather than a graph, rasterised or traced.
fn a_plane_shot(
    make: impl FnOnce(&mut Document) -> kreate_scene::shader::ShaderGraph,
    adjust: impl FnOnce(&mut kreate_scene::Material),
    traced: bool,
) -> Vec<[f32; 4]> {
    let gpu = gpu().expect("checked by the caller");
    let mut doc = Document::new();
    kreate::startup::sample_scene(&mut doc);
    for id in doc
        .objects()
        .filter(|(_, o)| matches!(o.data, kreate_scene::document::ObjectData::Mesh(_)))
        .map(|(id, _)| id)
        .collect::<Vec<_>>()
    {
        doc.remove_object(id);
    }
    // One face, so that anything varying across it is the material's doing.
    let mesh = doc.add_mesh(kreate_mesh::primitives::grid(2.0, 1, 1));
    let graph = make(&mut doc);
    let mut material = kreate_scene::Material {
        name: "Procedural".into(),
        graph: Some(graph),
        ..Default::default()
    };
    adjust(&mut material);
    let material = doc.add_material(material);
    doc.set_mesh_slot(mesh, 0, Some(material));
    let object = doc.add_object(kreate_scene::document::Object::new(
        "Plane",
        kreate_scene::document::ObjectData::Mesh(mesh),
    ));
    doc.selection.add(object);
    shoot_with(
        &gpu,
        &doc,
        Camera {
            distance: 4.0,
            ..Camera::default()
        },
        traced,
    )
}

/// A grey surface with one node wired into one of its sockets.
fn grey_with(
    kind: kreate_scene::shader::NodeKind,
    from_socket: usize,
    to_socket: usize,
) -> kreate_scene::shader::ShaderGraph {
    let mut graph = kreate_scene::shader::ShaderGraph::standard(
        "Grey",
        kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
        0.0,
        0.5,
    );
    let node = graph
        .add_node(kreate_scene::shader::Node::new("Node", kind))
        .expect("added");
    graph
        .link(kreate_scene::shader::Link {
            from_node: node,
            from_socket,
            to_node: 0,
            to_socket,
        })
        .expect("the sockets agree");
    graph
}

/// A procedural texture reaches the picture. A plane wearing a red and blue
/// checker, seen face on: both colours on one face, which neither a flat
/// material nor its lighting can make — the texture is read at each pixel's
/// own position in the object.
#[test]
fn a_checker_texture_paints_one_face_in_two_colours() {
    if gpu().is_none() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let frame = a_plane_wearing(grey_with(
        kreate_scene::shader::NodeKind::Checker {
            scale: 2.0,
            color1: kreate_math::SceneRgb::new(0.8, 0.02, 0.02),
            color2: kreate_math::SceneRgb::new(0.02, 0.02, 0.8),
        },
        0,
        0,
    ));
    let reddish = frame.iter().filter(|p| p[0] > p[2] + 0.05).count();
    let bluish = frame.iter().filter(|p| p[2] > p[0] + 0.05).count();
    assert!(
        reddish > 200 && bluish > 200,
        "one face drew {reddish} red pixels and {bluish} blue ones; a checker \
         has both, or the texture is not being read where the surface is"
    );
}

/// A bump tilts the light on a flat face. Grey, one face, the same lamp: with
/// a wave's bands as its height the face goes light and dark in stripes, where
/// without it every pixel of it is lit nearly alike.
#[test]
fn a_bump_shades_a_flat_face_in_stripes() {
    if gpu().is_none() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let flat = a_plane_wearing(kreate_scene::shader::ShaderGraph::standard(
        "Grey",
        kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
        0.0,
        0.5,
    ));
    let mut graph = grey_with(
        kreate_scene::shader::NodeKind::Bump {
            strength: 1.0,
            distance: 0.05,
        },
        0,
        3,
    );
    let wave = graph
        .add_node(kreate_scene::shader::Node::new(
            "Wave",
            kreate_scene::shader::NodeKind::new_of("wave").expect("a kind"),
        ))
        .expect("added");
    graph
        .link(kreate_scene::shader::Link {
            from_node: wave,
            from_socket: 1,
            to_node: 1,
            to_socket: 2,
        })
        .expect("a number into the height");
    let bumped = a_plane_wearing(graph);
    // Pixel by pixel against the flat face: the background is the same in
    // both, so whatever differs is the face, lit at another angle.
    let lum = |p: &[f32; 4]| p[0] + p[1] + p[2];
    let brighter = flat
        .iter()
        .zip(&bumped)
        .filter(|(a, b)| lum(b) > lum(a) + 0.05)
        .count();
    let darker = flat
        .iter()
        .zip(&bumped)
        .filter(|(a, b)| lum(b) + 0.05 < lum(a))
        .count();
    assert!(
        brighter > 100 && darker > 100,
        "against the flat face, {brighter} pixels of the bumped one are lighter \
         and {darker} darker; bands tilt both ways, so both should be many"
    );
}

/// The path tracer runs every material the way the viewport does — several
/// at once, an image projected from three sides, a procedural colour and a
/// bump — handed over as the frame loop hands them. A material whose code or
/// layers did not fit the tracer's would fail its pipeline here, and the
/// checker would not be in the picture.
#[test]
fn the_path_tracer_draws_the_documents_own_materials() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut doc = Document::new();
    kreate::startup::sample_scene(&mut doc);
    for id in doc
        .objects()
        .filter(|(_, o)| matches!(o.data, kreate_scene::document::ObjectData::Mesh(_)))
        .map(|(id, _)| id)
        .collect::<Vec<_>>()
    {
        doc.remove_object(id);
    }
    use kreate_scene::shader::{Link, Node, NodeKind, Projection};
    // An image material first, so the checker's code comes second and reads
    // constants and layers after another material's.
    let image = doc.add_image(kreate_scene::Image::new(
        "Grey",
        8,
        8,
        kreate_scene::PixelFormat::Rgba8,
    ));
    let mut boxed = grey_with(
        NodeKind::Image {
            image: Some(image),
            projection: Projection::Box,
            blend: 0.2,
            scale: 1.0,
        },
        0,
        0,
    );
    let bump = boxed
        .add_node(Node::new("Bump", NodeKind::new_of("bump").expect("a kind")))
        .expect("added");
    let noise = boxed
        .add_node(Node::new(
            "Noise",
            NodeKind::new_of("noise").expect("a kind"),
        ))
        .expect("added");
    for (from_node, from_socket, to_node, to_socket) in [(noise, 0, bump, 2), (bump, 0, 0, 3)] {
        boxed
            .link(Link {
                from_node,
                from_socket,
                to_node,
                to_socket,
            })
            .expect("connects");
    }
    let first = doc.add_material(kreate_scene::Material {
        name: "Boxed".into(),
        graph: Some(boxed),
        ..Default::default()
    });
    let checker = doc.add_material(kreate_scene::Material {
        name: "Checker".into(),
        graph: Some(grey_with(
            NodeKind::Checker {
                scale: 2.0,
                color1: kreate_math::SceneRgb::new(0.8, 0.02, 0.02),
                color2: kreate_math::SceneRgb::new(0.02, 0.02, 0.8),
            },
            0,
            0,
        )),
        ..Default::default()
    });
    for (material, x) in [(first, 5.0), (checker, 0.0)] {
        let mesh = doc.add_mesh(kreate_mesh::primitives::grid(2.0, 1, 1));
        doc.set_mesh_slot(mesh, 0, Some(material));
        let object = doc.add_object(kreate_scene::document::Object::new(
            "Plane",
            kreate_scene::document::ObjectData::Mesh(mesh),
        ));
        if let Some(o) = doc.object_mut(object) {
            o.transform.translation = glam::DVec3::new(x, 0.0, 0.0);
        }
    }
    let frame = shoot_traced(
        &gpu,
        &doc,
        Camera {
            distance: 4.0,
            ..Camera::default()
        },
    );
    let reddish = frame.iter().filter(|p| p[0] > p[2] + 0.05).count();
    let bluish = frame.iter().filter(|p| p[2] > p[0] + 0.05).count();
    assert!(
        reddish > 100 && bluish > 100,
        "the traced checker drew {reddish} red pixels and {bluish} blue ones"
    );
}

/// A texture read at the UV instead of the object's position, through a
/// ramp: the checker's number picks between the ramp's two stops, so both
/// colours are on the face — by UV, which a 2 m plane spans 0 to 1 of.
#[test]
fn a_checker_read_by_uv_through_a_ramp_paints_two_colours() {
    if gpu().is_none() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    use kreate_scene::shader::{Link, Node, NodeKind, RampMode, RAMP_STOPS};
    let mut stops = [(1.0, kreate_math::SceneRgb::new(0.02, 0.02, 0.8)); RAMP_STOPS];
    stops[0] = (0.0, kreate_math::SceneRgb::new(0.8, 0.02, 0.02));
    let mut graph = grey_with(
        NodeKind::ColorRamp {
            interpolation: RampMode::Constant,
            count: 2,
            stops,
        },
        0,
        0,
    );
    let coords = graph
        .add_node(Node::new("Coords", NodeKind::TexCoord))
        .expect("added");
    let checker = graph
        .add_node(Node::new(
            "Checker",
            NodeKind::Checker {
                scale: 4.0,
                color1: kreate_math::SceneRgb::new(1.0, 1.0, 1.0),
                color2: kreate_math::SceneRgb::new(0.0, 0.0, 0.0),
            },
        ))
        .expect("added");
    for (from_node, from_socket, to_node, to_socket) in
        [(coords, 1, checker, 3), (checker, 1, 1, 0)]
    {
        graph
            .link(Link {
                from_node,
                from_socket,
                to_node,
                to_socket,
            })
            .expect("connects");
    }
    let frame = a_plane_wearing(graph);
    let reddish = frame.iter().filter(|p| p[0] > p[2] + 0.05).count();
    let bluish = frame.iter().filter(|p| p[2] > p[0] + 0.05).count();
    assert!(
        reddish > 200 && bluish > 200,
        "the face drew {reddish} red pixels and {bluish} blue ones"
    );
}

/// A picture is the right way up: V runs up it, as in Blender and OBJ. A
/// plane wearing an image red in its top half and blue in its bottom shows
/// the red at its +Y edge, where its V is high — which the default view puts
/// at the top of the screen. Upside down, the halves trade places.
#[test]
fn an_image_is_the_right_way_up_on_its_uvs() {
    if gpu().is_none() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    use kreate_scene::shader::NodeKind;
    let pictured = a_plane_wearing_with(|doc| {
        let mut image =
            kreate_scene::Image::new("Halves", 64, 64, kreate_scene::PixelFormat::Rgba8);
        {
            let tile = image.tile_mut(0, 0).expect("one tile");
            let bytes = tile.bytes_mut();
            let side = kreate_scene::image::TILE as usize;
            for row in 0..side {
                for column in 0..side {
                    let at = (row * side + column) * 4;
                    // Stored top row first: the top half red.
                    bytes[at..at + 4].copy_from_slice(if row < 32 {
                        &[255u8, 0, 0, 255]
                    } else {
                        &[0u8, 0, 255, 255]
                    });
                }
            }
        }
        let id = doc.add_image(image);
        grey_with(NodeKind::image(Some(id)), 0, 0)
    });
    let reddish = |p: &[f32; 4]| p[0] > p[2] + 0.05;
    let bluish = |p: &[f32; 4]| p[2] > p[0] + 0.05;
    // Where the plane's +Y edge is on the screen: the default camera looks
    // down the plane with +Y away from it, so up the screen.
    let rows = |pick: &dyn Fn(&[f32; 4]) -> bool, top: bool| {
        let range = if top {
            0..(SIZE as usize / 2)
        } else {
            (SIZE as usize / 2)..SIZE as usize
        };
        range
            .flat_map(|y| (0..SIZE as usize).map(move |x| y * SIZE as usize + x))
            .filter(|i| pick(&pictured[*i]))
            .count()
    };
    let (red_top, red_bottom) = (rows(&reddish, true), rows(&reddish, false));
    let (blue_top, blue_bottom) = (rows(&bluish, true), rows(&bluish, false));
    assert!(
        red_top > red_bottom && blue_bottom > blue_top,
        "red {red_top} up and {red_bottom} down, blue {blue_top} up and {blue_bottom} down: \
         the picture's top should be the plane's top"
    );
}

/// A cut-out: a plane whose picture is clear on its left half and opaque on
/// its right, with a threshold, shows the background through the left — and
/// without the threshold, the whole plane.
#[test]
fn a_cut_out_material_is_cut_where_its_picture_is_clear() {
    if gpu().is_none() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let shoot = |clip: Option<f32>, traced: bool| {
        let image_id = std::cell::Cell::new(None);
        a_plane_shot(
            |doc| {
                let mut image =
                    kreate_scene::Image::new("Leaf", 64, 64, kreate_scene::PixelFormat::Rgba8);
                {
                    let tile = image.tile_mut(0, 0).expect("one tile");
                    let bytes = tile.bytes_mut();
                    let side = kreate_scene::image::TILE as usize;
                    for row in 0..side {
                        for column in 0..side {
                            let at = (row * side + column) * 4;
                            let alpha = if column < 32 { 0 } else { 255 };
                            bytes[at..at + 4].copy_from_slice(&[230, 20, 20, alpha]);
                        }
                    }
                }
                let id = doc.add_image(image);
                image_id.set(Some(id));
                kreate_scene::shader::ShaderGraph::for_material(&kreate_scene::Material {
                    base_color: kreate_math::SceneRgb::new(1.0, 1.0, 1.0),
                    base_color_image: Some(id),
                    alpha_clip: clip,
                    ..Default::default()
                })
            },
            |material| {
                material.base_color = kreate_math::SceneRgb::new(1.0, 1.0, 1.0);
                material.base_color_image = image_id.get();
                material.alpha_clip = clip;
            },
            traced,
        )
    };
    let reddish = |f: &[[f32; 4]]| f.iter().filter(|p| p[0] > p[2] + 0.1).count();
    for traced in [false, true] {
        let (whole, cut) = (
            reddish(&shoot(None, traced)),
            reddish(&shoot(Some(0.5), traced)),
        );
        assert!(
            whole > 400 && cut * 3 < whole * 2 && cut > whole / 4,
            "traced {traced}: {whole} red pixels solid, {cut} cut: about half should go"
        );
    }
}
