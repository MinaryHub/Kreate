//! Generates repository screenshots (`screenshots/01..06.png`) and multi-frame
//! animation sequences (`target/gif-frames/{overview,gears,humanoid}/`) using
//! the real application shell, operators, PBR/studio renderer, and overlays.
//!
//! Run manually when updating visual documentation:
//! `cargo test -p kreate --test media -- --ignored --nocapture`

use std::path::{Path, PathBuf};

use kreate::scenelights;
use kreate_math::{DQuat, DVec3, Srgb, Vec3};
use kreate_ops::operator::{Context, Mode};
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue as V};
use kreate_render::viewport::{Draw, ShadingMode, Viewport};
use kreate_render::{Camera, GpuOptions, Projection, RenderMeshCpu};
use kreate_scene::animation::{bone_path, Action, Interpolation};
use kreate_scene::armature::{Armature, Bone};
use kreate_scene::document::{Collection, Material, Object, ObjectData, ObjectId};
use kreate_scene::Document;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const SAMPLED_AS: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct ShotSpec<'a> {
    doc: &'a Document,
    history: &'a History,
    locale: &'a str,
    mode: Mode,
    edit_mode: bool,
    shading: ShadingMode,
    centre: DVec3,
    radius: f64,
    rotation: DQuat,
    ortho: bool,
    title: &'a str,
}

struct MediaRig {
    gpu: kreate_render::Gpu,
    w: u32,
    h: u32,
    bytes_per_row: u32,
    surface: wgpu::Texture,
    surface_view: wgpu::TextureView,
    vp_view: wgpu::TextureView,
    egui_renderer: egui_wgpu::Renderer,
    vp_id: egui::TextureId,
    viewport: Viewport,
    readback: wgpu::Buffer,
    registry: OperatorRegistry,
    keymap: kreate_input::Keymap,
}

impl MediaRig {
    fn new(w: u32, h: u32) -> Option<Self> {
        let opts = GpuOptions::default();
        let instance = kreate_render::Gpu::create_instance(&opts);
        let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
        let bytes_per_row = (w * 4).div_ceil(256) * 256;

        let make_target = |label, usage, view_formats: &[wgpu::TextureFormat]| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage,
                view_formats,
            })
        };

        let surface = make_target(
            "media-surface",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            &[],
        );
        let surface_view = surface.create_view(&wgpu::TextureViewDescriptor::default());
        let vp = make_target(
            "media-viewport",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            &[SAMPLED_AS],
        );
        let vp_view = vp.create_view(&wgpu::TextureViewDescriptor::default());
        let vp_sampled = vp.create_view(&wgpu::TextureViewDescriptor {
            format: Some(SAMPLED_AS),
            ..Default::default()
        });

        let mut egui_renderer =
            egui_wgpu::Renderer::new(&gpu.device, FORMAT, egui_wgpu::RendererOptions::default());
        let vp_id = egui_renderer.register_native_texture(
            &gpu.device,
            &vp_sampled,
            wgpu::FilterMode::Linear,
        );

        let viewport = Viewport::new(&gpu.device, FORMAT, w, h);
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("media-readback"),
            size: (bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut registry = OperatorRegistry::new();
        kreate_ops::register_builtin(&mut registry);
        let keymap = kreate_input::presets::load(kreate_input::Preset::default());

        Some(Self {
            gpu,
            w,
            h,
            bytes_per_row,
            surface,
            surface_view,
            vp_view,
            egui_renderer,
            vp_id,
            viewport,
            readback,
            registry,
            keymap,
        })
    }

    fn render_to_png(
        &mut self,
        spec: &ShotSpec<'_>,
        out_path: &Path,
        arrange: impl FnOnce(&mut kreate::shell::ShellState),
    ) {
        let ctx = egui::Context::default();
        kreate_ui::install_fonts(&ctx);
        kreate_ui::Theme::new(kreate_ui::Appearance::Dark, kreate_ui::Density::default())
            .install(&ctx);
        ctx.set_pixels_per_point(1.0);

        let overrides = std::collections::BTreeMap::new();
        let mut state =
            kreate::shell::ShellState::with_catalog(kreate::shell::load_catalog(spec.locale));
        state.document_title = spec.title.to_string();
        state.length_unit = spec.doc.design.length_unit.metres();
        state.view_status.shading = spec.shading;
        state.view_status.orthographic = spec.ortho;
        state.timeline = kreate_ui::timeline::TimelineView {
            frame: spec.doc.playback.frame,
            start: spec.doc.playback.start,
            end: spec.doc.playback.end,
            fps: spec.doc.playback.fps,
            playing: false,
            autokey: false,
            keys: Vec::new(),
            markers: Vec::new(),
            active_camera: false,
        };
        if spec.edit_mode {
            state.uv = kreate_ops::ops::mesh::target_mesh(spec.doc)
                .and_then(|mesh| spec.doc.mesh(mesh))
                .map(|mesh| kreate::uvview::from_bmesh(&mesh.to_bmesh()).unwrap_or_default());
        }
        arrange(&mut state);

        self.viewport.shading = spec.shading;
        self.viewport.path_trace = false;
        self.viewport.show_grid = spec.shading != ShadingMode::Rendered;
        self.viewport.show_ground_shadow = spec.shading != ShadingMode::Rendered;
        self.viewport.show_elements = spec.edit_mode;

        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(self.w as f32, self.h as f32),
            )),
            ..Default::default()
        };

        let mut viewport_rect = egui::Rect::ZERO;
        let mut uv_extent = (1.0f32, 1.0f32);
        let mut last_output: Option<egui::FullOutput> = None;
        let mut texture_deltas: Vec<(egui::TextureId, egui::epaint::ImageDelta)> = Vec::new();
        let mut camera = Camera::default();

        for pass in 0..4 {
            let mut input = raw_input.clone();
            input.time = Some(pass as f64 * 0.25);
            camera.rotation = spec.rotation;
            camera.projection = if spec.ortho {
                Projection::Orthographic
            } else {
                Projection::Perspective
            };
            camera.frame_sphere(spec.centre, spec.radius, self.viewport.aspect());
            state.camera_rotation = camera.rotation;
            if viewport_rect.width() > 0.0 {
                state.projector = Some(kreate_ui::object_glyph::Projector {
                    eye: camera.eye(),
                    rotation: camera.rotation,
                    projection: camera.projection_matrix(self.viewport.aspect()).as_dmat4(),
                    rect: viewport_rect,
                });
            }
            let output = ctx.run_ui(input, |ui| {
                viewport_rect = kreate::shell::draw(
                    ui,
                    &mut state,
                    spec.doc,
                    &self.registry,
                    spec.history,
                    &self.keymap,
                    kreate_input::Preset::default(),
                    &overrides,
                    spec.mode,
                    Some(kreate::shell::ViewportImage {
                        texture: self.vp_id,
                        uv: uv_extent,
                    }),
                );
            });
            self.viewport.set_render_size(
                (viewport_rect.width() * output.pixels_per_point).max(1.0) as u32,
                (viewport_rect.height() * output.pixels_per_point).max(1.0) as u32,
            );
            uv_extent = self.viewport.uv_extent();
            texture_deltas.extend(output.textures_delta.set.iter().cloned());
            last_output = Some(output);
        }
        let output = last_output.expect("egui passes ran");

        let materials = kreate::materials::Materials::compile(spec.doc);
        if matches!(spec.shading, ShadingMode::Material | ShadingMode::Rendered) {
            let preview = spec.shading == ShadingMode::Material;
            let light = preview.then(|| scenelights::material_preview(camera.rotation));
            self.viewport.set_world(
                light
                    .as_ref()
                    .map_or_else(|| scenelights::world_radiance(spec.doc), |p| p.world),
            );
            self.viewport.set_display(
                &self.gpu.queue,
                kreate_render::display::Display {
                    view: kreate_math::ViewTransform::AgX,
                    exposure: light.as_ref().map_or_else(
                        || scenelights::shot_exposure(spec.doc, 24.0, camera.pivot) as f32,
                        |p| p.exposure,
                    ),
                },
            );
            let lights = light.as_ref().map_or_else(
                || scenelights::placed_lights(spec.doc),
                |p| p.lights.clone(),
            );
            self.viewport
                .set_lights(&self.gpu.device, &self.gpu.queue, camera.eye(), &lights);

            let tex_pixels: Vec<Vec<kreate::materials::TextureImage>> = materials
                .compiled
                .iter()
                .map(|c| kreate::materials::pixels_for(spec.doc, c))
                .collect();
            let tex_images: Vec<Vec<kreate_render::material::SourceImage<'_>>> = tex_pixels
                .iter()
                .map(|set| set.iter().map(|i| i.source()).collect())
                .collect();
            let mat_sources: Vec<kreate_render::viewport::MaterialSource<'_>> = materials
                .compiled
                .iter()
                .zip(&tex_images)
                .map(|(c, imgs)| kreate_render::viewport::MaterialSource {
                    wgsl: &c.wgsl,
                    constants: &c.constants,
                    image_slots: &c.image_slots,
                    images: imgs,
                })
                .collect();
            self.viewport
                .set_materials(&self.gpu.device, &self.gpu.queue, &mat_sources);
            let (w_rot, w_str) = spec
                .doc
                .active_world()
                .and_then(|id| spec.doc.world(id))
                .map(|w| (w.rotation as f32, w.strength as f32))
                .unwrap_or((0.0, 1.0));
            self.viewport.set_world_shape(&self.gpu.queue, w_rot, w_str);
        }

        let mut cache = kreate_scene::modifier::ModifierCache::new();
        let eval_ctx = kreate_scene::modifier::EvalCtx::still();
        let uploaded: Vec<(ObjectId, kreate_render::GpuMesh, u32, Vec<u32>)> = spec
            .doc
            .objects()
            .filter(|(id, _)| spec.doc.is_shown(*id))
            .filter_map(|(id, _)| {
                let mesh = cache.evaluated(spec.doc, id, &eval_ctx)?;
                let sculpting = spec.mode.acts_on() == kreate_ops::mode::ActsOn::Sculpt
                    && spec.doc.selection.active == Some(id);
                let mesh = if sculpting {
                    kreate::sculptview::SculptDisplay::default()
                        .show(&mesh)
                        .unwrap_or(mesh)
                } else {
                    mesh
                };
                let tint = spec
                    .doc
                    .material_at(id, 0)
                    .map(|m| {
                        let s = m.base_color.as_swatch();
                        glam::Vec4::new(s.r, s.g, s.b, 1.0)
                    })
                    .unwrap_or(glam::Vec4::new(0.76, 0.75, 0.73, 1.0));
                let cpu = RenderMeshCpu::extract(&mesh);
                Some((
                    id,
                    kreate_render::GpuMesh::upload(
                        &self.gpu.device,
                        &self.viewport.layouts,
                        &cpu,
                        tint,
                    ),
                    materials.index_of(spec.doc, id),
                    materials.slots_of(spec.doc, id),
                ))
            })
            .collect();

        let mut draws: Vec<Draw> = uploaded
            .iter()
            .map(|(id, m, mat_idx, slots)| {
                Draw::new(m, spec.doc.world_transform(*id))
                    .with_material(*mat_idx)
                    .with_materials(slots)
                    .selected(!spec.edit_mode && spec.doc.selection.contains(*id))
            })
            .collect();

        let shown = match spec.mode.acts_on() {
            kreate_ops::mode::ActsOn::Bones => spec
                .doc
                .selection
                .active
                .map(|id| (id, kreate::glyphs::Bones::Pose)),
            kreate_ops::mode::ActsOn::BoneJoints => spec
                .doc
                .selection
                .active
                .map(|id| (id, kreate::glyphs::Bones::Edit)),
            _ => None,
        };
        let (marker_lines, marker_points) = if spec.shading == ShadingMode::Rendered {
            (Vec::new(), Vec::new())
        } else {
            kreate::glyphs::markers_with(spec.doc, shown)
        };
        self.viewport.set_markers(
            &self.gpu.device,
            &self.gpu.queue,
            camera.eye(),
            &marker_lines,
            &marker_points,
        );

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("media-frame"),
            });
        self.viewport
            .prepare(&self.gpu.device, &self.gpu.queue, &camera, &mut draws);
        self.viewport.render(&mut encoder, &self.vp_view, &draws);

        let paint_jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
        for (id, delta) in &texture_deltas {
            self.egui_renderer
                .update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.w, self.h],
            pixels_per_point: output.pixels_per_point,
        };
        let extra = self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("media-egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.egui_renderer
                .render(&mut pass.forget_lifetime(), &paint_jobs, &screen);
        }
        for id in &output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.surface,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(self.h),
                },
            },
            wgpu::Extent3d {
                width: self.w,
                height: self.h,
                depth_or_array_layers: 1,
            },
        );
        self.gpu
            .queue
            .submit(extra.into_iter().chain(Some(encoder.finish())));

        let (tx, rx) = std::sync::mpsc::channel();
        self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        rx.recv().expect("map callback").expect("map");
        let data = self.readback.slice(..).get_mapped_range();
        let mut rgb = Vec::with_capacity((self.w * self.h * 3) as usize);
        for row in 0..self.h {
            let start = (row * self.bytes_per_row) as usize;
            let end = start + (self.w * 4) as usize;
            for &[r, g, b, _] in data[start..end].as_chunks::<4>().0 {
                rgb.push(r);
                rgb.push(g);
                rgb.push(b);
            }
        }
        drop(data);
        self.readback.unmap();

        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        image::save_buffer(out_path, &rgb, self.w, self.h, image::ColorType::Rgb8)
            .expect("write png");
        eprintln!("wrote {}", out_path.display());
    }
}

fn run_op(
    r: &OperatorRegistry,
    h: &mut History,
    doc: &mut Document,
    mode: Mode,
    op: &str,
    pairs: &[(&'static str, V)],
) {
    let mut p = PropertyMap::new();
    for (k, v) in pairs {
        p.set(k, v.clone());
    }
    h.dispatch(r, doc, &Context { mode }, op, &p)
        .unwrap_or_else(|e| panic!("{op} failed: {e}"));
}

fn assign_mat(doc: &mut Document, id: ObjectId, mat: Material) {
    let mid = doc.add_material(mat);
    if let Some(mesh_id) = doc.object(id).and_then(|o| o.mesh()) {
        doc.set_mesh_slot(mesh_id, 0, Some(mid));
    }
}

fn world_basis() -> [Vec3; 3] {
    [Vec3::X, Vec3::Y, Vec3::Z]
}

// --- Scene 1: Mechanical Gearbox Assembly ( + Meshing Gear Train ) -----------

fn build_mechanical_scene(
    r: &OperatorRegistry,
    angle_rad: f64,
) -> (Document, History, ObjectId, ObjectId) {
    let mut h = History::new();
    let mut doc = Document::new();
    kreate::startup::starting_scene(&mut doc);
    // Remove default startup cube, keep Sun & Camera.
    let cube = doc
        .objects()
        .find(|(_, o)| o.mesh().is_some())
        .map(|(id, _)| id);
    if let Some(cube) = cube {
        doc.remove_object(cube);
    }
    doc.design.length_unit = kreate_reflect::LengthUnit::Millimetres;

    // Baseplate: 196 x 116 x 10 mm anodized dark aluminium (`cube(2.0)` spans -1..+1)
    let plate_mesh = {
        let mut m = kreate_mesh::primitives::cube(2.0);
        for p in m.positions_mut() {
            *p = Vec3::new(p.x * 0.098, p.y * 0.058, p.z * 0.005);
        }
        m
    };
    let plate = doc.add_mesh_object("Baseplate 196x116", plate_mesh);
    doc.object_mut(plate).unwrap().transform.translation = DVec3::new(0.030, 0.0, -0.005);
    assign_mat(
        &mut doc,
        plate,
        Material {
            name: "Anodized Slate".into(),
            base_color: Srgb::new(0.18, 0.20, 0.25).as_reflectance(),
            metallic: 0.75,
            roughness: 0.30,
            ..Default::default()
        },
    );

    // Meshed Helical Gear Pair: module index 6 ("2" mm), z1 = 36, z2 = 24
    // Centre distance = 2 * (36 + 24) / 2 = 60 mm = 0.060 m
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Object,
        "object.add_gear",
        &[
            ("module", V::Enum(6)),
            ("teeth", V::Int(36)),
            ("mate_teeth", V::Int(24)),
            ("width", V::Float(0.014)),
            ("bore", V::Float(0.012)),
            ("helix", V::Float(0.28)),
        ],
    );

    let gear_main = doc
        .objects()
        .find(|(_, o)| o.name.contains("36"))
        .map(|(id, _)| id)
        .expect("main gear");
    let gear_pinion = doc
        .objects()
        .find(|(_, o)| o.name.contains("24"))
        .map(|(id, _)| id)
        .expect("pinion gear");

    doc.object_mut(gear_main).unwrap().transform.translation.z = 0.024;
    doc.object_mut(gear_main).unwrap().transform.rotation = DQuat::from_rotation_z(angle_rad);
    let pinion_pos = doc.object(gear_pinion).unwrap().transform.translation;
    let pinion_base_rot = doc.object(gear_pinion).unwrap().transform.rotation;
    doc.object_mut(gear_pinion).unwrap().transform.translation.z = 0.024;
    doc.object_mut(gear_pinion).unwrap().transform.rotation =
        DQuat::from_rotation_z(-angle_rad * (36.0 / 24.0)) * pinion_base_rot;

    assign_mat(
        &mut doc,
        gear_main,
        Material {
            name: "Naval Brass".into(),
            base_color: Srgb::new(0.90, 0.66, 0.24).as_reflectance(),
            metallic: 0.88,
            roughness: 0.24,
            ..Default::default()
        },
    );
    assign_mat(
        &mut doc,
        gear_pinion,
        Material {
            name: "Hardened Steel".into(),
            base_color: Srgb::new(0.76, 0.80, 0.86).as_reflectance(),
            metallic: 0.82,
            roughness: 0.26,
            ..Default::default()
        },
    );

    // Shafts & Bearings at x=0 and x=pinion_pos.x
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Object,
        "object.add_shaft",
        &[
            ("diameter", V::Float(0.012)),
            ("length", V::Float(0.082)),
            ("keyway", V::Bool(true)),
            ("with_key", V::Bool(true)),
            ("key_length", V::Float(0.024)),
            ("at_cursor", V::Bool(false)),
            ("location", V::Vec3([0.0, 0.0, -0.008])),
        ],
    );
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Object,
        "object.add_bearing",
        &[
            ("size", V::Enum(4)), // 6001 (12x28x8)
            ("shields", V::Enum(2)),
            ("at_cursor", V::Bool(false)),
            ("location", V::Vec3([0.0, 0.0, 0.002])),
        ],
    );
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Object,
        "object.add_shaft",
        &[
            ("diameter", V::Float(0.010)),
            ("length", V::Float(0.068)),
            ("keyway", V::Bool(true)),
            ("with_key", V::Bool(true)),
            ("key_length", V::Float(0.020)),
            ("at_cursor", V::Bool(false)),
            ("location", V::Vec3([pinion_pos.x, pinion_pos.y, -0.008])),
        ],
    );
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Object,
        "object.add_bearing",
        &[
            ("size", V::Enum(3)), // 6000 (10x26x8)
            ("shields", V::Enum(2)),
            ("at_cursor", V::Bool(false)),
            ("location", V::Vec3([pinion_pos.x, pinion_pos.y, 0.002])),
        ],
    );

    let steel = doc.add_material(Material {
        name: "Chrome Steel".into(),
        base_color: Srgb::new(0.72, 0.76, 0.82).as_reflectance(),
        metallic: 0.82,
        roughness: 0.22,
        ..Default::default()
    });
    let shaft_ids: Vec<ObjectId> = doc
        .objects()
        .filter(|(id, o)| {
            *id != plate
                && *id != gear_main
                && *id != gear_pinion
                && o.mesh().is_some()
                && doc.material_at(*id, 0).is_none()
        })
        .map(|(id, _)| id)
        .collect();
    for id in shaft_ids {
        if let Some(mid) = doc.object(id).and_then(|o| o.mesh()) {
            doc.set_mesh_slot(mid, 0, Some(steel));
        }
    }

    // Group mechanical parts into a named Outliner collection
    let col = doc.add_collection(Collection {
        name: "Gearbox Assembly".into(),
        objects: Vec::new(),
        hide: false,
    });
    let mech_ids: Vec<ObjectId> = doc
        .objects()
        .filter(|(_, o)| o.mesh().is_some())
        .map(|(id, _)| id)
        .collect();
    for id in mech_ids {
        doc.move_to_collection(id, Some(col));
    }

    let main_name = doc.object(gear_main).unwrap().name.clone();
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Object,
        "object.select",
        &[("name", V::Text(main_name)), ("extend", V::Bool(false))],
    );
    doc.playback.fps = 24.0;
    doc.playback.start = 1;
    doc.playback.end = 120;

    (doc, h, gear_main, gear_pinion)
}

// --- Scene 2: Rendered Still Life (Ceramic Vase, Mug, Glass Sphere) ----------

fn build_rendered_scene(r: &OperatorRegistry) -> (Document, History, ObjectId) {
    use kreate_scene::feature::AXIS_Y;

    let mut h = History::new();
    let mut doc = Document::new();
    doc.design.length_unit = kreate_reflect::LengthUnit::Millimetres;

    // Wide studio table so no horizon seam cuts across behind the still life
    let mut table_mesh = kreate_mesh::primitives::cube(1.0);
    for p in table_mesh.positions_mut() {
        *p = Vec3::new(p.x * 3.5, p.y * 3.5, (p.z - 1.0) * 0.02);
    }
    let table = doc.add_mesh_object("Studio Table", table_mesh);
    assign_mat(
        &mut doc,
        table,
        Material {
            name: "Warm Clay Table".into(),
            base_color: Srgb::new(0.76, 0.53, 0.36).as_reflectance(),
            metallic: 0.0,
            roughness: 0.58,
            ..Default::default()
        },
    );

    // Smoothly curved Revolved Ceramic Vase via dense profile points
    run_op(r, &mut h, &mut doc, Mode::Object, "sketch.new", &[]);
    let sketch = kreate_ops::ops::sketch_draw::active_sketch(&doc).unwrap();
    let mut profile_pts = vec![0.0, 0.0];
    for step in 0..=24 {
        let t = step as f64 / 24.0;
        let y = t * 0.185;
        let belly = (t * std::f64::consts::PI).sin().powf(0.85) * 0.026;
        let neck = (-((t - 0.82).powi(2)) / 0.018).exp() * -0.018;
        let lip = ((t - 0.88).max(0.0) / 0.12).powi(2) * 0.008;
        let rad = 0.034 + belly + neck + lip;
        profile_pts.push(rad);
        profile_pts.push(y);
    }
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Sketch,
        "sketch.line",
        &[("points", V::Floats(profile_pts))],
    );
    let shape = kreate::sketching::revolve_shape([AXIS_Y; 2]);
    let seeds = kreate_scene::feature::shape_profile_seeds(&doc, sketch, &shape);
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Sketch,
        "feature.revolve",
        &[
            ("seeds", V::Floats(seeds[0].to_vec())),
            ("about", V::Enum(2)),
        ],
    );
    doc.object_mut(sketch).unwrap().visible = false;
    let vase = doc
        .objects()
        .find(|(_, o)| !o.features.is_empty())
        .map(|(id, _)| id)
        .expect("revolved vase");
    doc.object_mut(vase).unwrap().name = "Glazed Ceramic Vase".into();
    doc.object_mut(vase).unwrap().transform.rotation =
        DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    doc.object_mut(vase).unwrap().transform.translation = DVec3::new(-0.055, 0.035, 0.0);

    let mut vase_mat = Material {
        name: "Teal Ceramic Glaze".into(),
        base_color: Srgb::new(0.04, 0.42, 0.52).as_reflectance(),
        metallic: 0.05,
        roughness: 0.16,
        layers: kreate_scene::shader::SurfaceLayers {
            coat: 0.92,
            coat_roughness: 0.06,
            ..Default::default()
        },
        ..Default::default()
    };
    vase_mat.graph = Some(vase_mat.shading_graph());
    let vase_mid = doc.add_material(vase_mat);
    // Attach vase_mid to the vase's mesh slot 0 as well so the Material panel displays it!
    if let Some(mid) = doc.object(vase).and_then(|o| o.mesh()) {
        doc.set_mesh_slot(mid, 0, Some(vase_mid));
    }
    doc.object_mut(vase).unwrap().material_overrides = vec![Some(vase_mid)];

    // Porcelain Mug (smooth-shaded cylinder + torus handle)
    let mut mug_body = kreate_mesh::primitives::cylinder(0.038, 0.082, 64);
    for p in mug_body.positions_mut() {
        p.z += 0.041;
    }
    let mug_nf = mug_body.num_faces() as usize;
    mug_body.attributes_mut().insert(
        kreate_mesh::attribute::names::SHADE_SMOOTH,
        kreate_mesh::attribute::Domain::Face,
        kreate_mesh::attribute::AttrData::Bool(std::sync::Arc::new(vec![true; mug_nf])),
    );
    let mug = doc.add_mesh_object("Porcelain Mug", mug_body);
    doc.object_mut(mug).unwrap().transform.translation = DVec3::new(0.038, -0.015, 0.0);
    let porcelain = Material {
        name: "Ivory Porcelain".into(),
        base_color: Srgb::new(0.92, 0.90, 0.87).as_reflectance(),
        metallic: 0.0,
        roughness: 0.18,
        layers: kreate_scene::shader::SurfaceLayers {
            coat: 0.65,
            coat_roughness: 0.10,
            ..Default::default()
        },
        ..Default::default()
    };
    assign_mat(&mut doc, mug, porcelain.clone());

    let mut handle_mesh = kreate_mesh::primitives::torus(0.024, 0.0065, 36, 20);
    let h_nf = handle_mesh.num_faces() as usize;
    handle_mesh.attributes_mut().insert(
        kreate_mesh::attribute::names::SHADE_SMOOTH,
        kreate_mesh::attribute::Domain::Face,
        kreate_mesh::attribute::AttrData::Bool(std::sync::Arc::new(vec![true; h_nf])),
    );
    let handle = doc.add_mesh_object("Mug Handle", handle_mesh);
    doc.object_mut(handle).unwrap().transform.translation = DVec3::new(0.074, -0.015, 0.041);
    doc.object_mut(handle).unwrap().transform.rotation =
        DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    assign_mat(&mut doc, handle, porcelain);

    // Smooth Amber Glass Sphere
    let mut glass_mesh = kreate_mesh::primitives::uv_sphere(0.028, 64, 48);
    let g_nf = glass_mesh.num_faces() as usize;
    glass_mesh.attributes_mut().insert(
        kreate_mesh::attribute::names::SHADE_SMOOTH,
        kreate_mesh::attribute::Domain::Face,
        kreate_mesh::attribute::AttrData::Bool(std::sync::Arc::new(vec![true; g_nf])),
    );
    let glass = doc.add_mesh_object("Amber Glass Sphere", glass_mesh);
    doc.object_mut(glass).unwrap().transform.translation = DVec3::new(-0.048, -0.032, 0.028);
    assign_mat(
        &mut doc,
        glass,
        Material {
            name: "Amber Glass".into(),
            base_color: Srgb::new(0.92, 0.68, 0.42).as_reflectance(),
            metallic: 0.0,
            roughness: 0.08,
            transmission: 0.88,
            ior: 1.52,
            layers: kreate_scene::shader::SurfaceLayers {
                coat: 0.80,
                coat_roughness: 0.04,
                ..Default::default()
            },
            ..Default::default()
        },
    );

    // Warm Sun Light for crisp studio lighting & ground shadow
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Object,
        "object.add_light",
        &[("kind", V::Enum(1))], // Sun
    );
    let sun = doc.selection.active.unwrap();
    doc.object_mut(sun).unwrap().name = "Key Sun".into();
    doc.object_mut(sun).unwrap().transform.rotation =
        DQuat::from_rotation_z(-0.65) * DQuat::from_rotation_x(0.72);
    if let ObjectData::Light(lid) = doc.object(sun).unwrap().data {
        let l = doc.light_mut(lid).unwrap();
        l.intensity = 3.2;
        l.color = Srgb::new(1.0, 0.96, 0.90).as_reflectance();
    }

    let world = doc.add_world(kreate_scene::World {
        name: "Warm Studio".into(),
        linkage: kreate_scene::document::Linkage::Local,
        uid: kreate_scene::Uid::NONE,
        background: kreate_scene::Background::Flat(Srgb::new(0.32, 0.36, 0.42).as_reflectance()),
        strength: 0.85,
        rotation: 0.0,
    });
    doc.set_active_world(Some(world));

    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Object,
        "object.select",
        &[
            ("name", V::Text("Glazed Ceramic Vase".into())),
            ("extend", V::Bool(false)),
        ],
    );
    (doc, h, vase)
}

// --- Scene 3: Hard-Surface Edit Mode + UV Unwrap & Checker Backdrop ----------

fn build_edit_mode_scene(r: &OperatorRegistry) -> (Document, History, DVec3) {
    let mut h = History::new();
    let mut doc = Document::new();
    kreate::startup::starting_scene(&mut doc);
    doc.design.length_unit = kreate_reflect::LengthUnit::Millimetres;

    let mut base_cube = kreate_mesh::primitives::cube(0.20);
    for p in base_cube.positions_mut() {
        p.z += 0.10;
    }
    let cube = doc.add_mesh_object("Sci-Fi Panel Block", base_cube);
    doc.selection.clear();
    doc.selection.add(cube);
    doc.selection.active = Some(cube);

    // Select all & subdivide twice -> 4x4 grid per face (16 pads per face)
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::EditMesh,
        "mesh.select",
        &[("action", V::Enum(0))],
    );
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::EditMesh,
        "mesh.subdivide",
        &[("cuts", V::Int(1))],
    );
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::EditMesh,
        "mesh.subdivide",
        &[("cuts", V::Int(1))],
    );

    // Select top (+Z) and right (+X) faces, inset individually, then extrude!
    if let Some(mid) = doc.object(cube).and_then(|o| o.mesh()) {
        let mut m = doc.mesh(mid).unwrap().clone();
        let mut sel_face = vec![false; m.num_faces() as usize];
        for f in 0..m.num_faces() {
            let verts = m.face_verts(f);
            let avg_z: f32 = verts
                .iter()
                .map(|&v| m.positions()[v as usize].z)
                .sum::<f32>()
                / verts.len() as f32;
            let avg_x: f32 = verts
                .iter()
                .map(|&v| m.positions()[v as usize].x)
                .sum::<f32>()
                / verts.len() as f32;
            if avg_z > 0.195 || avg_x > 0.095 {
                sel_face[f as usize] = true;
            }
        }
        m.attributes_mut().insert(
            kreate_mesh::attribute::names::SELECT_FACE,
            kreate_mesh::attribute::Domain::Face,
            kreate_mesh::attribute::AttrData::Bool(std::sync::Arc::new(sel_face)),
        );
        *doc.mesh_mut(mid).unwrap() = m;
    }

    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::EditMesh,
        "mesh.inset",
        &[
            ("thickness", V::Float(0.004)),
            ("individual", V::Bool(true)),
        ],
    );
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::EditMesh,
        "mesh.extrude_region",
        &[("offset", V::Float(0.012))],
    );
    run_op(r, &mut h, &mut doc, Mode::EditMesh, "uv.smart_project", &[]);

    let mut uv_grid =
        kreate_scene::Image::new("uv_checker.png", 128, 128, kreate_scene::PixelFormat::Rgba8);
    for y in 0..128u32 {
        for x in 0..128u32 {
            let even = ((x / 16) + (y / 16)) % 2 == 0;
            let border = (x % 16 == 0) || (y % 16 == 0);
            let c = if border {
                [72u8, 86, 104, 255]
            } else if even {
                [54, 60, 68, 255]
            } else {
                [42, 47, 54, 255]
            };
            let tile = kreate_scene::TILE;
            let t = uv_grid.tile_mut(x / tile, y / tile).unwrap();
            let at = (((y % tile) * tile + (x % tile)) * 4) as usize;
            t.bytes_mut()[at..at + 4].copy_from_slice(&c);
        }
    }
    let grid_img = doc.add_image(uv_grid);

    assign_mat(
        &mut doc,
        cube,
        Material {
            name: "Machined Titanium".into(),
            base_color: Srgb::new(0.76, 0.80, 0.86).as_reflectance(),
            base_color_image: Some(grid_img),
            metallic: 0.7,
            roughness: 0.35,
            ..Default::default()
        },
    );

    (doc, h, DVec3::new(0.0, 0.0, 0.212))
}

// --- Scene 4: Digital Clay Sculpting Bust + Symmetry Dots --------------------

fn build_sculpt_scene() -> (Document, History, DVec3) {
    let mut h = History::new();
    let mut doc = Document::new();
    kreate::startup::starting_scene(&mut doc);
    let cube = doc
        .objects()
        .find(|(_, o)| o.mesh().is_some())
        .map(|(id, _)| id);
    if let Some(cube) = cube {
        doc.remove_object(cube);
    }
    doc.design.length_unit = kreate_reflect::LengthUnit::Millimetres;

    // Procedurally sculpted high-resolution classical clay bust (symmetric across X!)
    let mut mesh = kreate_mesh::primitives::uv_sphere(0.11, 96, 64);
    for p in mesh.positions_mut() {
        let mut q = *p / 0.11; // unit sphere coords, facing -Y, Z up
                               // Cranium & jaw taper
        if q.z < -0.15 {
            let t = ((-0.15 - q.z) / 0.85).clamp(0.0, 1.0);
            // Neck cylinder transition
            if q.z < -0.55 {
                let neck = ((q.z + 0.55) / -0.45).clamp(0.0, 1.0);
                q.x *= 0.54 + 0.10 * neck;
                q.y *= 0.54 + 0.12 * neck;
                q.z -= 0.18 * neck;
            } else {
                q.x *= 1.0 - 0.28 * t;
                q.y *= 1.0 - 0.12 * t;
            }
        } else {
            q.z *= 1.08;
            q.x *= 0.92;
        }
        // Brow ridge (facing -Y)
        let brow = (-((q.z - 0.20).powi(2) / 0.012) - ((q.y + 0.90).powi(2) / 0.08)).exp()
            * (-((q.x.abs() - 0.24).powi(2) / 0.08)).exp();
        q.y -= 0.11 * brow;
        // Eye sockets (symmetric concave hollows)
        let eye = (-(((q.x.abs() - 0.30).powi(2)) / 0.022)
            - (((q.z - 0.10).powi(2)) / 0.015)
            - (((q.y + 0.88).powi(2)) / 0.10))
            .exp();
        q.y += 0.13 * eye;
        // Nose bridge & tip
        let nose_bridge = (-(q.x.powi(2) / 0.010)
            - (((q.z - 0.02).powi(2)) / 0.045)
            - (((q.y + 0.92).powi(2)) / 0.12))
            .exp();
        let nose_tip = (-(q.x.powi(2) / 0.014)
            - (((q.z + 0.10).powi(2)) / 0.012)
            - (((q.y + 0.92).powi(2)) / 0.12))
            .exp();
        q.y -= 0.16 * nose_bridge + 0.14 * nose_tip;
        // Cheekbones
        let cheek = (-(((q.x.abs() - 0.48).powi(2)) / 0.045)
            - (((q.z + 0.05).powi(2)) / 0.035)
            - (((q.y + 0.72).powi(2)) / 0.15))
            .exp();
        q.x += q.x.signum() * 0.07 * cheek;
        q.y -= 0.08 * cheek;
        // Lips & chin prominence
        let lips = (-(q.x.powi(2) / 0.030)
            - (((q.z + 0.27).powi(2)) / 0.006)
            - (((q.y + 0.85).powi(2)) / 0.10))
            .exp();
        let chin = (-(q.x.powi(2) / 0.035)
            - (((q.z + 0.44).powi(2)) / 0.015)
            - (((q.y + 0.80).powi(2)) / 0.12))
            .exp();
        q.y -= 0.07 * lips + 0.10 * chin;

        *p = q * 0.11 + Vec3::new(0.0, 0.0, 0.12);
    }
    let num_faces = mesh.num_faces() as usize;
    mesh.attributes_mut().insert(
        kreate_mesh::attribute::names::SHADE_SMOOTH,
        kreate_mesh::attribute::Domain::Face,
        kreate_mesh::attribute::AttrData::Bool(std::sync::Arc::new(vec![true; num_faces])),
    );

    let bust = doc.add_mesh_object("Clay Bust", mesh);
    assign_mat(
        &mut doc,
        bust,
        Material {
            name: "Terracotta Sculpt Clay".into(),
            base_color: Srgb::new(0.76, 0.46, 0.32).as_reflectance(),
            metallic: 0.0,
            roughness: 0.48,
            ..Default::default()
        },
    );

    doc.selection.clear();
    doc.selection.add(bust);
    doc.selection.active = Some(bust);

    // Record clean sculpt history steps for the bottom history strip
    let mut reg = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut reg);
    run_op(
        &reg,
        &mut h,
        &mut doc,
        Mode::Object,
        "object.select",
        &[
            ("name", V::Text("Clay Bust".into())),
            ("extend", V::Bool(false)),
        ],
    );

    // Hover point on right cheekbone (+X) -> symmetry indicator mirrors to (-X)!
    let hover = DVec3::new(0.056, -0.088, 0.115);
    (doc, h, hover)
}

// --- Scene 5: Parametric 2D Sketch + Extruded Solid + Feature Edit -----------

fn build_sketch_cad_scene(
    r: &OperatorRegistry,
    width_m: f64,
    extrude_m: f64,
) -> (Document, History, ObjectId) {
    let mut h = History::new();
    let mut doc = Document::new();
    kreate::startup::starting_scene(&mut doc);
    let cube = doc
        .objects()
        .find(|(_, o)| o.mesh().is_some())
        .map(|(id, _)| id);
    if let Some(cube) = cube {
        doc.remove_object(cube);
    }
    doc.design.length_unit = kreate_reflect::LengthUnit::Millimetres;

    run_op(r, &mut h, &mut doc, Mode::Object, "sketch.new", &[]);
    let sketch = kreate_ops::ops::sketch_draw::active_sketch(&doc).unwrap();
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Sketch,
        "sketch.rectangle",
        &[
            ("a", V::Vec3([0.0, 0.0, 0.0])),
            ("b", V::Vec3([0.060, 0.036, 0.0])),
            ("join", V::Int(1)),
        ],
    );
    for (pts, val) in [(vec![0u32, 1], width_m), (vec![1u32, 2], 0.036)] {
        run_op(
            r,
            &mut h,
            &mut doc,
            Mode::Sketch,
            "sketch.constrain",
            &[
                ("kind", V::Enum(0)),
                ("points", V::Ints(pts)),
                ("value", V::Float(val)),
            ],
        );
    }
    // Two mounting holes that track inside the bracket
    for cx in [0.016, width_m - 0.016] {
        run_op(
            r,
            &mut h,
            &mut doc,
            Mode::Sketch,
            "sketch.circle",
            &[
                ("centre", V::Vec3([cx, 0.018, 0.0])),
                ("radius", V::Float(0.006)),
            ],
        );
    }
    // Constrain first hole radius so Ø / R dimension appears cleanly
    run_op(
        r,
        &mut h,
        &mut doc,
        Mode::Sketch,
        "sketch.constrain",
        &[
            ("kind", V::Enum(0)),
            ("points", V::Ints(vec![4, 5])),
            ("value", V::Float(0.006)),
        ],
    );

    if extrude_m > 0.0005 {
        run_op(
            r,
            &mut h,
            &mut doc,
            Mode::Sketch,
            "feature.extrude",
            &[
                ("seeds", V::Floats(vec![0.003, 0.003])),
                ("distance", V::Float(extrude_m)),
            ],
        );
        let body = doc
            .objects()
            .find(|(_, o)| !o.features.is_empty())
            .map(|(id, _)| id);
        if let Some(body) = body {
            doc.object_mut(body).unwrap().name = "Mounting Bracket".into();
            let mat = doc.add_material(Material {
                name: "Aerospace Aluminium".into(),
                base_color: Srgb::new(0.46, 0.62, 0.82).as_reflectance(),
                metallic: 0.82,
                roughness: 0.26,
                ..Default::default()
            });
            doc.object_mut(body).unwrap().material_overrides = vec![Some(mat)];
        }
    }

    doc.selection.clear();
    doc.selection.add(sketch);
    doc.selection.active = Some(sketch);
    (doc, h, sketch)
}

// --- Scene 6: Articulated Humanoid Robot Rig & Pose Animation ----------------

fn build_humanoid_scene(phase: f64) -> (Document, History, ObjectId, DVec3) {
    let h = History::new();
    let mut doc = Document::new();
    kreate::startup::starting_scene(&mut doc);
    let cube = doc
        .objects()
        .find(|(_, o)| o.mesh().is_some())
        .map(|(id, _)| id);
    if let Some(cube) = cube {
        doc.remove_object(cube);
    }
    doc.design.length_unit = kreate_reflect::LengthUnit::Millimetres;

    // Build an articulated humanoid armature
    let mut arm = Armature::new("HumanoidRig");
    arm.in_front = true;
    arm.bones.clear();
    let add_b = |arm: &mut Armature,
                 name: &str,
                 parent: Option<&str>,
                 head: DVec3,
                 tail: DVec3,
                 connected: bool| {
        let p_idx = parent.and_then(|p| arm.bone_index(p));
        arm.add_bone(Bone {
            name: name.into(),
            parent: p_idx,
            head,
            tail,
            connected,
            ..Default::default()
        });
    };

    add_b(
        &mut arm,
        "Hips",
        None,
        DVec3::new(0.0, 0.0, 0.92),
        DVec3::new(0.0, 0.0, 1.08),
        false,
    );
    add_b(
        &mut arm,
        "Chest",
        Some("Hips"),
        DVec3::new(0.0, 0.0, 1.08),
        DVec3::new(0.0, 0.0, 1.42),
        true,
    );
    add_b(
        &mut arm,
        "Head",
        Some("Chest"),
        DVec3::new(0.0, 0.0, 1.44),
        DVec3::new(0.0, 0.0, 1.68),
        false,
    );

    for (suf, sx) in [(".L", 1.0), (".R", -1.0)] {
        add_b(
            &mut arm,
            &format!("UpperArm{suf}"),
            Some("Chest"),
            DVec3::new(sx * 0.22, 0.0, 1.38),
            DVec3::new(sx * 0.48, 0.0, 1.32),
            false,
        );
        add_b(
            &mut arm,
            &format!("Forearm{suf}"),
            Some(&format!("UpperArm{suf}")),
            DVec3::new(sx * 0.48, 0.0, 1.32),
            DVec3::new(sx * 0.72, 0.0, 1.30),
            true,
        );
        add_b(
            &mut arm,
            &format!("Hand{suf}"),
            Some(&format!("Forearm{suf}")),
            DVec3::new(sx * 0.72, 0.0, 1.30),
            DVec3::new(sx * 0.84, 0.0, 1.30),
            true,
        );
        add_b(
            &mut arm,
            &format!("Thigh{suf}"),
            Some("Hips"),
            DVec3::new(sx * 0.12, 0.0, 0.90),
            DVec3::new(sx * 0.13, 0.0, 0.48),
            false,
        );
        add_b(
            &mut arm,
            &format!("Shin{suf}"),
            Some(&format!("Thigh{suf}")),
            DVec3::new(sx * 0.13, 0.0, 0.48),
            DVec3::new(sx * 0.13, 0.0, 0.06),
            true,
        );
        add_b(
            &mut arm,
            &format!("Foot{suf}"),
            Some(&format!("Shin{suf}")),
            DVec3::new(sx * 0.13, 0.0, 0.06),
            DVec3::new(sx * 0.13, -0.14, 0.02),
            true,
        );
    }

    let arm_id = doc.add_armature(arm);
    let mut rig_obj = Object::new("Atlas Humanoid", ObjectData::Armature(arm_id));

    // Smooth sinusoidal pose cycle driven by `phase` in [0, TAU)
    let wave = phase.sin();
    let cos_p = phase.cos();
    rig_obj.pose.entry("Hips").transform.translation =
        DVec3::new(0.03 * wave, 0.0, 0.015 * cos_p.abs());
    rig_obj.pose.entry("Chest").transform.rotation =
        DQuat::from_rotation_z(0.10 * wave) * DQuat::from_rotation_x(-0.06);
    rig_obj.pose.entry("Head").transform.rotation = DQuat::from_rotation_z(-0.12 * wave);
    // Left arm raised in an expressive greeting/wave!
    rig_obj.pose.entry("UpperArm.L").transform.rotation =
        DQuat::from_rotation_y(-0.85 - 0.12 * wave) * DQuat::from_rotation_z(-0.35);
    rig_obj.pose.entry("Forearm.L").transform.rotation =
        DQuat::from_rotation_z(-0.85 + 0.38 * wave);
    rig_obj.pose.entry("Forearm.L").selected = true;
    rig_obj.pose.active = Some("Forearm.L".into());
    // Right arm relaxed athletic guard
    rig_obj.pose.entry("UpperArm.R").transform.rotation =
        DQuat::from_rotation_y(1.05) * DQuat::from_rotation_x(-0.22 * wave);
    rig_obj.pose.entry("Forearm.R").transform.rotation = DQuat::from_rotation_z(0.55);
    // Subtle athletic stance in legs
    rig_obj.pose.entry("Thigh.L").transform.rotation = DQuat::from_rotation_x(-0.14 + 0.10 * wave);
    rig_obj.pose.entry("Shin.L").transform.rotation = DQuat::from_rotation_x(0.22);
    rig_obj.pose.entry("Thigh.R").transform.rotation = DQuat::from_rotation_x(0.12 - 0.10 * wave);
    rig_obj.pose.entry("Shin.R").transform.rotation = DQuat::from_rotation_x(0.16);

    let rig = doc.add_object(rig_obj);

    // Attach sleek armored robot plates parented to each bone!
    let armor_white = doc.add_material(Material {
        name: "Titanium Armor".into(),
        base_color: Srgb::new(0.80, 0.85, 0.92).as_reflectance(),
        metallic: 0.72,
        roughness: 0.26,
        ..Default::default()
    });
    let joint_dark = doc.add_material(Material {
        name: "Cobalt Actuator".into(),
        base_color: Srgb::new(0.22, 0.34, 0.52).as_reflectance(),
        metallic: 0.80,
        roughness: 0.30,
        ..Default::default()
    });
    let visor_cyan = doc.add_material(Material {
        name: "Cyan Visor LED".into(),
        base_color: Srgb::new(0.12, 0.88, 1.0).as_reflectance(),
        emission: Srgb::new(0.20, 0.90, 1.0).as_reflectance(),
        emission_strength: 5.0,
        roughness: 0.12,
        ..Default::default()
    });

    let (_, arm_ref) = doc.armature_of(rig).unwrap();
    let bone_frames: Vec<(String, kreate_math::DAffine3, f64)> = arm_ref
        .bones
        .iter()
        .map(|b| (b.name.clone(), b.rest(), b.length()))
        .collect();

    // `primitives::cube(2.0)` spans -1..+1 so `sx, sz` are true half-extents!
    let add_limb = |doc: &mut Document,
                    name: &str,
                    bone: &str,
                    sx: f32,
                    sy: f32,
                    sz: f32,
                    mat: kreate_scene::document::MaterialId| {
        let (_, frame, len) = bone_frames.iter().find(|(n, _, _)| n == bone).unwrap();
        let mut m = kreate_mesh::primitives::cube(2.0);
        for p in m.positions_mut() {
            *p = Vec3::new(p.x * sx, (p.y + 1.0) * 0.5 * (*len as f32) * sy, p.z * sz);
        }
        let mid = doc.add_mesh(m);
        doc.set_mesh_slot(mid, 0, Some(mat));
        let mut obj = Object::new(name, ObjectData::Mesh(mid));
        obj.transform = kreate_math::DTransform::from_affine(*frame);
        let oid = doc.add_object(obj);
        doc.set_parent_bone(oid, rig, bone).expect("bone parent");
    };

    add_limb(
        &mut doc,
        "Pelvis Chassis",
        "Hips",
        0.15,
        0.95,
        0.10,
        joint_dark,
    );
    add_limb(
        &mut doc,
        "Torso Armor",
        "Chest",
        0.19,
        0.94,
        0.12,
        armor_white,
    );
    add_limb(&mut doc, "Helmet", "Head", 0.105, 0.90, 0.11, armor_white);
    // Glowing Visor on Head
    {
        let (_, frame, len) = bone_frames.iter().find(|(n, _, _)| n == "Head").unwrap();
        let mut m = kreate_mesh::primitives::cube(2.0);
        for p in m.positions_mut() {
            *p = Vec3::new(
                p.x * 0.092,
                (*len as f32) * 0.52 + p.y * 0.028,
                -0.105 + p.z * 0.022,
            );
        }
        let mid = doc.add_mesh(m);
        doc.set_mesh_slot(mid, 0, Some(visor_cyan));
        let mut obj = Object::new("Optic Visor", ObjectData::Mesh(mid));
        obj.transform = kreate_math::DTransform::from_affine(*frame);
        let oid = doc.add_object(obj);
        doc.set_parent_bone(oid, rig, "Head").unwrap();
    }

    for suf in [".L", ".R"] {
        add_limb(
            &mut doc,
            &format!("UpperArm Plate{suf}"),
            &format!("UpperArm{suf}"),
            0.056,
            0.90,
            0.056,
            armor_white,
        );
        add_limb(
            &mut doc,
            &format!("Forearm Plate{suf}"),
            &format!("Forearm{suf}"),
            0.048,
            0.90,
            0.048,
            joint_dark,
        );
        add_limb(
            &mut doc,
            &format!("Hand Manipulator{suf}"),
            &format!("Hand{suf}"),
            0.040,
            0.88,
            0.028,
            armor_white,
        );
        add_limb(
            &mut doc,
            &format!("Thigh Actuator{suf}"),
            &format!("Thigh{suf}"),
            0.074,
            0.92,
            0.074,
            armor_white,
        );
        add_limb(
            &mut doc,
            &format!("Shin Guard{suf}"),
            &format!("Shin{suf}"),
            0.060,
            0.92,
            0.064,
            joint_dark,
        );
        add_limb(
            &mut doc,
            &format!("Foot Boot{suf}"),
            &format!("Foot{suf}"),
            0.066,
            0.96,
            0.042,
            armor_white,
        );
    }

    // Add keyframe action on the rig so Dope Sheet & Timeline display rich keyframe diamonds!
    let mut action = Action::new("WaveCycle");
    for bone_name in [
        "Hips",
        "Chest",
        "UpperArm.L",
        "Forearm.L",
        "Thigh.L",
        "Thigh.R",
    ] {
        for f in [1.0, 12.0, 24.0, 36.0, 48.0] {
            for c in 0..4u8 {
                action
                    .curve_mut(&bone_path(bone_name, "rotation_quaternion"), c)
                    .insert(f, if c == 0 { 1.0 } else { 0.15 }, Interpolation::Bezier);
            }
        }
    }
    action.select_all_with(|_| false);
    action.select_range(
        Some(&bone_path("Forearm.L", "rotation_quaternion")),
        12.0,
        24.0,
        true,
    );
    let act_id = doc.add_action(action);
    doc.object_mut(rig).unwrap().action = Some(act_id);

    let frame = 1.0 + (phase / std::f64::consts::TAU) * 47.0;
    doc.playback.fps = 24.0;
    doc.playback.start = 1;
    doc.playback.end = 48;
    doc.playback.frame = frame;

    let posed = doc.posed_bones(rig);
    let (_, arm_ref) = doc.armature_of(rig).unwrap();
    let fa_idx = arm_ref.bone_index("Forearm.L").unwrap();
    let gizmo_pos = posed[fa_idx].translation;

    doc.selection.clear();
    doc.selection.add(rig);
    doc.selection.active = Some(rig);

    (doc, h, rig, gizmo_pos)
}

#[test]
#[ignore]
fn generate_screenshots_and_frames() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let shots_dir = repo_root.join("screenshots");
    let frames_root = repo_root.join("target/gif-frames");

    // =========================================================================
    // 1. Render all 6 Full-HD (1920 x 1080) Repository Screenshots
    // =========================================================================
    let mut hd = MediaRig::new(1920, 1080).expect("GPU adapter available");
    let r = {
        let mut reg = OperatorRegistry::new();
        kreate_ops::register_builtin(&mut reg);
        reg
    };

    // 01-mechanical.png
    {
        let (doc, h, gear_main, _) = build_mechanical_scene(&r, 0.12);
        let pivot = doc.world_transform(gear_main).translation;
        let rot = DQuat::from_rotation_z(-0.58) * DQuat::from_rotation_x(1.04);
        hd.render_to_png(
            &ShotSpec {
                doc: &doc,
                history: &h,
                locale: "en",
                mode: Mode::Object,
                edit_mode: false,
                shading: ShadingMode::Studio,
                centre: DVec3::new(0.028, 0.0, 0.020),
                radius: 0.072,
                rotation: rot,
                ortho: false,
                title: "*gearbox.kreate",
            },
            &shots_dir.join("01-mechanical.png"),
            |state| {
                state.tools.transform_gizmo = 1; // Move + Planar XY/XZ/YZ Gizmo!
                state.gizmo_target = Some((pivot, world_basis()));
            },
        );
    }

    // 02-rendered.png
    {
        let (doc, h, _) = build_rendered_scene(&r);
        let rot = DQuat::from_rotation_z(-0.38) * DQuat::from_rotation_x(1.18);
        hd.render_to_png(
            &ShotSpec {
                doc: &doc,
                history: &h,
                locale: "en",
                mode: Mode::Object,
                edit_mode: false,
                shading: ShadingMode::Rendered,
                centre: DVec3::new(-0.010, 0.005, 0.082),
                radius: 0.135,
                rotation: rot,
                ortho: false,
                title: "ceramics.kreate",
            },
            &shots_dir.join("02-rendered.png"),
            |state| {
                let areas: Vec<_> = state.screen().areas().iter().map(|(a, _)| *a).collect();
                for area in areas {
                    state.set_properties_tab(area, kreate::shell::PropertiesTab::Material);
                }
            },
        );
    }

    // 03-edit-mode.png
    {
        let (doc, h, gizmo_pos) = build_edit_mode_scene(&r);
        let rot = DQuat::from_rotation_z(-0.68) * DQuat::from_rotation_x(1.06);
        hd.render_to_png(
            &ShotSpec {
                doc: &doc,
                history: &h,
                locale: "en",
                mode: Mode::EditMesh,
                edit_mode: true,
                shading: ShadingMode::Studio,
                centre: DVec3::new(0.0, 0.0, 0.105),
                radius: 0.165,
                rotation: rot,
                ortho: false,
                title: "*panel_block.kreate",
            },
            &shots_dir.join("03-edit-mode.png"),
            |state| {
                state.switch_workspace(2); // UV Editing workspace!
                let areas: Vec<_> = state.screen().areas().iter().map(|(a, _)| *a).collect();
                for area in areas {
                    state.set_uv_checker(area, true);
                }
                state.tools.transform_gizmo = 1;
                state.gizmo_target = Some((gizmo_pos, world_basis()));
            },
        );
    }

    // 04-sculpt.png
    {
        let (doc, h, hover) = build_sculpt_scene();
        let rot = DQuat::from_rotation_z(-0.34) * DQuat::from_rotation_x(1.38);
        hd.render_to_png(
            &ShotSpec {
                doc: &doc,
                history: &h,
                locale: "en",
                mode: Mode::Sculpt,
                edit_mode: false,
                shading: ShadingMode::Studio,
                centre: DVec3::new(0.0, 0.0, 0.125),
                radius: 0.130,
                rotation: rot,
                ortho: false,
                title: "*clay_bust.kreate",
            },
            &shots_dir.join("04-sculpt.png"),
            |state| {
                state.sculpt.mirror_x = true; // X Symmetry ON
                state.sculpt_hover_world = Some(hover); // 3D Symmetry mirror dots!
            },
        );
    }

    // 05-sketch.png
    {
        let (doc, h, sketch) = build_sketch_cad_scene(&r, 0.080, 0.014);
        let rot = DQuat::from_rotation_z(-0.48) * DQuat::from_rotation_x(0.72);
        let ui = kreate::sketching::SketchUi {
            tool: kreate::sketching::Tool::Dimension,
            seeds: vec![glam::DVec2::new(0.004, 0.004)],
            ..Default::default()
        };
        let points = kreate_ops::ops::sketch_draw::points(&doc, sketch);
        let pinned = kreate::sketching::pinned(&points, &doc.object(sketch).unwrap().sketch);
        hd.render_to_png(
            &ShotSpec {
                doc: &doc,
                history: &h,
                locale: "en",
                mode: Mode::Sketch,
                edit_mode: false,
                shading: ShadingMode::Studio,
                centre: DVec3::new(0.040, 0.018, 0.006),
                radius: 0.052,
                rotation: rot,
                ortho: false,
                title: "*mounting_bracket.kreate",
            },
            &shots_dir.join("05-sketch.png"),
            |state| {
                state.sketch_view = Some(kreate::sketching::view(
                    &doc,
                    sketch,
                    &ui,
                    &state.catalog,
                    &pinned,
                ));
                state.features =
                    kreate::sketching::feature_rows(&doc, &state.catalog, &|_| Vec::new());
                // Open the Extrude feature row (`!is_sketch`) to showcase Through All / Taper / Join-Cut-Intersect!
                if let Some(ext_row) = state
                    .features
                    .iter()
                    .find(|f| !f.is_sketch)
                    .cloned()
                    .or_else(|| state.features.last().cloned())
                {
                    state.feature_edit = Some(kreate::shell::FeatureEdit {
                        amount: 14.0,
                        second: 0.0,
                        offset: 0.0,
                        taper_deg: 0.0,
                        expression: String::new(),
                        row: ext_row,
                    });
                }
            },
        );
    }

    // 06-humanoid.png
    {
        let (doc, h, _, gizmo_pos) = build_humanoid_scene(0.65);
        let rot = DQuat::from_rotation_z(-0.36) * DQuat::from_rotation_x(1.38);
        hd.render_to_png(
            &ShotSpec {
                doc: &doc,
                history: &h,
                locale: "en",
                mode: Mode::Pose,
                edit_mode: false,
                shading: ShadingMode::Studio,
                centre: DVec3::new(0.05, 0.0, 0.86),
                radius: 0.96,
                rotation: rot,
                ortho: false,
                title: "atlas_humanoid.kreate",
            },
            &shots_dir.join("06-humanoid.png"),
            |state| {
                state.switch_workspace(4); // Animation workspace (Dope Sheet + Timeline)
                let areas: Vec<_> = state.screen().areas().iter().map(|(a, _)| *a).collect();
                for area in areas {
                    state.set_properties_tab(area, kreate::shell::PropertiesTab::Bone);
                }
                state.tools.transform_gizmo = 2; // 3D Rotate Gizmo on active bone!
                state.gizmo_target = Some((gizmo_pos, world_basis()));
            },
        );
    }

    drop(hd);

    // =========================================================================
    // 2. Render 1280 x 720 Multi-Frame Sequences for Animated GIFs
    // =========================================================================
    let mut anim = MediaRig::new(1280, 720).expect("GPU adapter available");

    // GIF 1: `gears` — 20-frame 100% seamless meshing helical gear train loop!
    let gears_dir = frames_root.join("gears");
    let gear_frames = 20usize;
    let tooth_pitch_36 = std::f64::consts::TAU / 36.0;
    for i in 0..gear_frames {
        let t = i as f64 / gear_frames as f64;
        // Advance 3 teeth over 20 frames -> seamless tooth-to-tooth wrap!
        let angle = t * 3.0 * tooth_pitch_36;
        let sway = (t * std::f64::consts::TAU).sin();
        let (mut doc, h, gear_main, _) = build_mechanical_scene(&r, angle);
        doc.playback.frame = 1.0 + t * 119.0;
        let pivot = doc.world_transform(gear_main).translation;
        let rot = DQuat::from_rotation_z(-0.58 + 0.14 * sway)
            * DQuat::from_rotation_x(1.04 + 0.05 * (t * std::f64::consts::TAU).cos());
        anim.render_to_png(
            &ShotSpec {
                doc: &doc,
                history: &h,
                locale: "ko-KR",
                mode: Mode::Object,
                edit_mode: false,
                shading: ShadingMode::Studio,
                centre: DVec3::new(0.028, 0.0, 0.020),
                radius: 0.072,
                rotation: rot,
                ortho: false,
                title: "*기어박스_조립품.kreate",
            },
            &gears_dir.join(format!("frame-{i:03}.png")),
            |state| {
                state.tools.transform_gizmo = 1;
                state.gizmo_target = Some((pivot, world_basis()));
            },
        );
    }

    // GIF 2: `humanoid` — 20-frame 100% seamless humanoid pose & timeline loop!
    let hum_dir = frames_root.join("humanoid");
    let hum_frames = 20usize;
    for i in 0..hum_frames {
        let t = i as f64 / hum_frames as f64;
        let phase = t * std::f64::consts::TAU;
        let (doc, h, _, gizmo_pos) = build_humanoid_scene(phase);
        let rot = DQuat::from_rotation_z(-0.36 + 0.16 * phase.sin())
            * DQuat::from_rotation_x(1.38 + 0.04 * phase.cos());
        anim.render_to_png(
            &ShotSpec {
                doc: &doc,
                history: &h,
                locale: "ko-KR",
                mode: Mode::Pose,
                edit_mode: false,
                shading: ShadingMode::Studio,
                centre: DVec3::new(0.05, 0.0, 0.86),
                radius: 0.96,
                rotation: rot,
                ortho: false,
                title: "휴머노이드_리깅.kreate",
            },
            &hum_dir.join(format!("frame-{i:03}.png")),
            |state| {
                state.switch_workspace(4);
                let areas: Vec<_> = state.screen().areas().iter().map(|(a, _)| *a).collect();
                for area in areas {
                    state.set_properties_tab(area, kreate::shell::PropertiesTab::Bone);
                }
                state.tools.transform_gizmo = 2;
                state.gizmo_target = Some((gizmo_pos, world_basis()));
            },
        );
    }

    // GIF 3: `overview` — 24-frame Parametric 2D Sketch -> 3D Extrude -> Live Resize!
    let ov_dir = frames_root.join("overview");
    let ov_frames = 24usize;
    for i in 0..ov_frames {
        let (width_m, extrude_m, yaw, pitch, locale) = if i < 6 {
            // Act 1: 2D Sketch angled tilt into 3D
            let u = i as f64 / 5.0;
            (0.060, 0.0, -0.45 * u, 0.72 * u, "ko-KR")
        } else if i < 14 {
            // Act 2: Smoothly extrude bracket 0 -> 16 mm
            let u = (i - 6) as f64 / 7.0;
            (
                0.060,
                0.001 + 0.015 * u,
                -0.45 - 0.15 * u,
                0.72 + 0.22 * u,
                "ko-KR",
            )
        } else {
            // Act 3: Parametric width live resize 60 mm -> 90 mm -> 60 mm + orbit
            let u = (i - 14) as f64 / 9.0;
            let pulse = (u * std::f64::consts::PI).sin();
            (
                0.060 + 0.030 * pulse,
                0.016,
                -0.60 + 0.35 * pulse,
                0.94,
                "ko-KR",
            )
        };
        let (doc, h, sketch) = build_sketch_cad_scene(&r, width_m, extrude_m);
        let ui = kreate::sketching::SketchUi {
            tool: if extrude_m < 0.0005 {
                kreate::sketching::Tool::Extrude
            } else {
                kreate::sketching::Tool::Dimension
            },
            seeds: vec![glam::DVec2::new(0.004, 0.004)],
            ..Default::default()
        };
        let points = kreate_ops::ops::sketch_draw::points(&doc, sketch);
        let pinned = kreate::sketching::pinned(&points, &doc.object(sketch).unwrap().sketch);
        let rot = DQuat::from_rotation_z(yaw) * DQuat::from_rotation_x(pitch);
        anim.render_to_png(
            &ShotSpec {
                doc: &doc,
                history: &h,
                locale,
                mode: Mode::Sketch,
                edit_mode: false,
                shading: ShadingMode::Studio,
                centre: DVec3::new(width_m * 0.5, 0.018, extrude_m * 0.4),
                radius: 0.056,
                rotation: rot,
                ortho: i == 0,
                title: "*파라메트릭_브래킷.kreate",
            },
            &ov_dir.join(format!("frame-{i:03}.png")),
            |state| {
                state.sketch_view = Some(kreate::sketching::view(
                    &doc,
                    sketch,
                    &ui,
                    &state.catalog,
                    &pinned,
                ));
                state.features =
                    kreate::sketching::feature_rows(&doc, &state.catalog, &|_| Vec::new());
            },
        );
    }
}
