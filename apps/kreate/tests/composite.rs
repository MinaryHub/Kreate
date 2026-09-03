//! Does the interface actually reach the framebuffer?
//!
//! The 3D viewport has its own headless test, and the panels have their own unit
//! tests, but the seam between them had none: the offscreen texture is drawn by
//! `ui.painter().image(...)` into a rectangle the panel layout decides, and if
//! that rectangle comes back empty the scene renders perfectly into a texture
//! nobody ever sees. "It launches without a validation error" does not rule that
//! out, which is exactly why this exists.
//!
//! So this runs the real shell through egui, composites it with egui-wgpu the way
//! the application does, and asserts on the resulting pixels.

use kreate_ops::{History, OperatorRegistry};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

// A realistic window. At 640 wide the side panels would take three quarters of
// it, which produces a portrait viewport no user would ever see and made an
// earlier version of this test misleading rather than wrong.
const W: u32 = 1280;
const H: u32 = 800;
const BYTES_PER_ROW: u32 = W * 4;
/// The surface format egui composites into.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

struct Composited {
    pixels: Vec<[u8; 4]>,
    /// The rectangle the shell reserved for the viewport, in points.
    viewport_rect: egui::Rect,
}

impl Composited {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        self.pixels[(y.min(H - 1) * W + x.min(W - 1)) as usize]
    }

    fn distinct_colours(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for p in &self.pixels {
            seen.insert((p[0] / 8, p[1] / 8, p[2] / 8));
        }
        seen.len()
    }

    /// Fraction of pixels that are not the clear colour.
    fn painted_fraction(&self) -> f32 {
        let n = self
            .pixels
            .iter()
            .filter(|p| p[0] > 4 || p[1] > 4 || p[2] > 4)
            .count();
        n as f32 / self.pixels.len() as f32
    }
}

/// Runs one frame of the whole application shell and reads back the surface.
///
/// Returns `None` when there is no GPU, so a runner without one skips rather
/// than reporting a failure it cannot act on.
fn composite(doc: &Document, locale: &str) -> Option<Composited> {
    composite_in(doc, locale, false)
}

fn composite_in(doc: &Document, locale: &str, edit_mode: bool) -> Option<Composited> {
    // The filename distinguishes every variant, including how much is selected:
    // when several frames share one, only the last survives and the picture on
    // disk is not the one that failed.
    let selected: usize = doc
        .meshes()
        .map(|(_, m)| {
            m.attributes()
                .get_bool(kreate_mesh::attribute::names::SELECT_VERT)
                .map(|s| s.iter().filter(|x| **x).count())
                .unwrap_or(0)
        })
        .sum();
    let label = match (doc.object_count(), edit_mode) {
        (0, _) => format!("empty-{locale}"),
        (_, true) => format!("edit{selected}-{locale}"),
        _ => format!("scene-{locale}"),
    };
    let opts = kreate_render::GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    // --- the surface stand-in, and the offscreen viewport target -------------
    let make_target = |label, usage| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage,
            view_formats: &[],
        })
    };
    let surface = make_target(
        "surface",
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let surface_view = surface.create_view(&wgpu::TextureViewDescriptor::default());
    let vp = make_target(
        "viewport",
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let vp_view = vp.create_view(&wgpu::TextureViewDescriptor::default());

    let mut egui_renderer =
        egui_wgpu::Renderer::new(&gpu.device, FORMAT, egui_wgpu::RendererOptions::default());
    let vp_id =
        egui_renderer.register_native_texture(&gpu.device, &vp_view, wgpu::FilterMode::Linear);

    let mut viewport = kreate_render::Viewport::new(&gpu.device, FORMAT, W, H);
    viewport.show_grid = true;

    // --- run the shell -------------------------------------------------------
    let ctx = egui::Context::default();
    kreate_ui::install_fonts(&ctx);
    // The surface is described to the renderer in *pixels* while egui lays out in
    // *points*. Pinning the ratio to 1 keeps the two consistent here; the
    // application takes the real value from the window instead.
    ctx.set_pixels_per_point(1.0);

    let registry = {
        let mut r = OperatorRegistry::new();
        kreate_ops::register_builtin(&mut r);
        r
    };
    let history = History::new();
    let keymap = kreate_input::presets::load(kreate_input::Preset::default());
    let overrides = std::collections::BTreeMap::new();
    let mut state = kreate::shell::ShellState {
        catalog: kreate::shell::load_catalog(locale),
        edit_mode,
        ..Default::default()
    };

    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(W as f32, H as f32),
        )),
        ..Default::default()
    };

    // Two passes: the first learns how much room the panels leave, the second
    // draws with the UVs that match. The application gets this for free, because
    // it runs a frame every time round its loop.
    let mut viewport_rect = egui::Rect::ZERO;
    let mut root_rect = egui::Rect::ZERO;
    let mut uv_extent = (1.0f32, 1.0f32);
    let mut last_output: Option<egui::FullOutput> = None;
    // egui emits its font atlas in the *first* pass's texture delta, and every
    // solid shape it draws samples a white texel from that atlas. Keeping only
    // the last pass's delta would leave all panel chrome and text sampling a
    // texture that was never uploaded, so they vanish while the separately
    // registered viewport image survives — which is exactly the misleading
    // picture this produced before.
    let mut texture_deltas: Vec<(egui::TextureId, egui::epaint::ImageDelta)> = Vec::new();
    for _ in 0..2 {
        let output = ctx.run_ui(raw_input.clone(), |ui| {
            root_rect = ui.max_rect();
            viewport_rect = kreate::shell::draw(
                ui,
                &mut state,
                doc,
                &registry,
                &history,
                &keymap,
                kreate_input::Preset::default(),
                &overrides,
                Some(kreate::shell::ViewportImage {
                    texture: vp_id,
                    uv: uv_extent,
                }),
            );
        });
        viewport.set_render_size(
            (viewport_rect.width() * output.pixels_per_point) as u32,
            (viewport_rect.height() * output.pixels_per_point) as u32,
        );
        uv_extent = viewport.uv_extent();
        texture_deltas.extend(output.textures_delta.set.iter().cloned());
        last_output = Some(output);
    }
    let output = last_output.expect("two passes ran");
    eprintln!(
        "screen {}x{}  root {:?}  viewport {:?}  render {:?}  uv {:?}  ppp {}",
        W,
        H,
        root_rect,
        viewport_rect,
        viewport.render_size(),
        uv_extent,
        output.pixels_per_point
    );

    // --- render the 3D into its target, then the interface over the surface ---
    let mut uploaded = Vec::new();
    for (_, object) in doc.objects() {
        let Some(mesh_id) = object.mesh() else {
            continue;
        };
        if uploaded.iter().any(|(m, _)| *m == mesh_id) {
            continue;
        }
        let Some(mesh) = doc.mesh(mesh_id) else {
            continue;
        };
        let cpu = kreate_render::RenderMeshCpu::extract(mesh);
        uploaded.push((
            mesh_id,
            kreate_render::GpuMesh::upload(
                &gpu.device,
                &viewport.layouts,
                &cpu,
                glam::Vec4::new(0.75, 0.72, 0.68, 1.0),
            ),
        ));
    }
    let mut draws: Vec<kreate_render::Draw> = doc
        .objects()
        .filter_map(|(id, o)| {
            let mesh_id = o.mesh()?;
            let (_, m) = uploaded.iter().find(|(mid, _)| *mid == mesh_id)?;
            Some(
                kreate_render::Draw::new(m, doc.world_transform(id))
                    .selected(!edit_mode && doc.selection.contains(id)),
            )
        })
        .collect();

    let mut camera = kreate_render::Camera::default();
    camera.frame_sphere(glam::DVec3::ZERO, 2.0, viewport.aspect());

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("composite"),
        });
    viewport.prepare(&gpu.device, &gpu.queue, &camera, &mut draws);
    viewport.render(&mut encoder, &vp_view, &draws);

    let paint_jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
    for (id, delta) in &texture_deltas {
        egui_renderer.update_texture(&gpu.device, &gpu.queue, *id, delta);
    }
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [W, H],
        pixels_per_point: output.pixels_per_point,
    };
    let extra =
        egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut encoder, &paint_jobs, &screen);
    {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &surface_view,
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
        egui_renderer.render(&mut pass.forget_lifetime(), &paint_jobs, &screen);
    }

    // --- read the surface back ----------------------------------------------
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (BYTES_PER_ROW * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &surface,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(BYTES_PER_ROW),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue
        .submit(extra.into_iter().chain(Some(encoder.finish())));

    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map callback").expect("map");
    let data = readback.slice(..).get_mapped_range();
    let pixels = data.as_chunks::<4>().0.to_vec();
    drop(data);
    readback.unmap();

    let composited = Composited {
        pixels,
        viewport_rect,
    };
    composited.dump(&label);
    Some(composited)
}

impl Composited {
    /// Writes the frame to `target/composite-<locale>.png`.
    ///
    /// Always, not behind a flag: when a layout defect is reported, the fastest
    /// route to a diagnosis is looking at the frame, and a file that is only
    /// written when someone remembers to set an environment variable is a file
    /// that does not exist when it is needed.
    fn dump(&self, label: &str) {
        let mut bytes = Vec::with_capacity(self.pixels.len() * 4);
        for p in &self.pixels {
            bytes.extend_from_slice(p);
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target")
            .join(format!("composite-{label}.png"));
        if let Some(img) = image::RgbaImage::from_raw(W, H, bytes) {
            if let Err(e) = img.save(&path) {
                eprintln!("could not write {}: {e}", path.display());
            } else {
                eprintln!("wrote {}", path.display());
            }
        }
    }
}

macro_rules! gpu_or_skip {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => {
                eprintln!("no GPU adapter available; skipping");
                return;
            }
        }
    };
}

fn doc_with_cube() -> Document {
    let mut doc = Document::new();
    let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
    let id = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
    doc.selection.add(id);
    doc
}

#[test]
fn the_shell_reserves_a_real_rectangle_for_the_viewport() {
    // The specific failure this file exists to catch: if the central panel hands
    // back an empty rect, the scene renders perfectly into a texture that is then
    // composited at zero size, and the application looks broken while every other
    // test passes.
    let c = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));
    let r = c.viewport_rect;
    assert!(
        r.width() > 100.0 && r.height() > 100.0,
        "the viewport rect is {r:?}; the panels left nothing to draw into"
    );
    // And it must be inside the window, not off somewhere.
    assert!(r.min.x >= 0.0 && r.min.y >= 0.0);
    assert!(r.max.x <= W as f32 + 1.0 && r.max.y <= H as f32 + 1.0);
}

#[test]
fn the_composited_frame_is_not_blank() {
    let c = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));
    let painted = c.painted_fraction();
    assert!(
        painted > 0.5,
        "only {:.1}% of the frame was painted; the interface did not reach the \
         framebuffer",
        painted * 100.0
    );
}

#[test]
fn the_viewport_texture_is_visible_in_the_middle_of_the_frame() {
    // Distinguishes "the panels drew but the 3D is missing" from "nothing drew".
    // The centre of the viewport rect must differ from the panel background.
    let c = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));
    let r = c.viewport_rect;
    let centre = c.at(r.center().x as u32, r.center().y as u32);
    // A pixel in the left panel, for comparison.
    let panel = c.at(20, H / 2);
    assert_ne!(
        centre, panel,
        "the middle of the viewport looks identical to the side panel; the \
         offscreen texture is probably not being sampled"
    );
    assert!(
        centre.iter().take(3).any(|c| *c > 8),
        "the middle of the viewport is black: {centre:?}"
    );
}

#[test]
fn panels_and_viewport_produce_many_distinct_colours() {
    // Text, panel chrome, the grid and the shaded cube together should span a
    // wide range. A handful of colours means one of those layers is missing.
    let c = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));
    assert!(
        c.distinct_colours() > 20,
        "only {} distinct colours; a layer is missing",
        c.distinct_colours()
    );
}

#[test]
fn korean_text_renders_rather_than_showing_tofu() {
    // Compares against the same frame in English. If the Korean build fell back
    // to notdef boxes, the two frames would be *more* alike, not less, because
    // every glyph would collapse to the same rectangle.
    let ko = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));
    let en = gpu_or_skip!(composite(&doc_with_cube(), "en-US"));

    let differing = ko
        .pixels
        .iter()
        .zip(&en.pixels)
        .filter(|(a, b)| (0..3).any(|i| (a[i] as i32 - b[i] as i32).abs() > 8))
        .count();
    assert!(
        differing > 200,
        "only {differing} pixels differ between the Korean and English interfaces; \
         the Korean text is probably not rendering"
    );
    // And the Korean frame must be as rich as the English one, not poorer.
    assert!(
        ko.distinct_colours() as f32 > en.distinct_colours() as f32 * 0.6,
        "the Korean frame has far fewer distinct colours ({}) than the English one \
         ({}), which is what tofu looks like",
        ko.distinct_colours(),
        en.distinct_colours()
    );
}

#[test]
fn an_empty_document_still_draws_the_interface() {
    let c = gpu_or_skip!(composite(&Document::new(), "ko-KR"));
    assert!(
        c.painted_fraction() > 0.5,
        "the panels should draw even with nothing in the scene"
    );
    assert!(c.viewport_rect.width() > 100.0);
}

#[test]
fn edit_mode_shows_element_selection_instead_of_the_object_outline() {
    // Two frames of the same mesh in edit mode, differing only in what is
    // selected. Rendered rather than asserted on state, because "the selection
    // is set but nothing is orange" is the failure that actually happens — the
    // attribute layer, the flags buffer and the shader are three layers apart,
    // and every one of them can drop it silently.
    let plain = doc_with_cube();

    let mut selected = doc_with_cube();
    let mesh_id = selected.meshes().next().map(|(id, _)| id).unwrap();
    {
        let mesh = selected.mesh_mut(mesh_id).unwrap();
        let mut bm = mesh.to_bmesh();
        bm.select_all();
        *mesh = bm.to_mesh_data();
    }

    // The selection colour is a saturated orange (255, 158, 41). Red alone is
    // not enough of a test: the grid's X axis is pure red and contributes
    // several hundred pixels to every frame, which is exactly the sort of
    // baseline that makes a threshold pass for the wrong reason.
    let orange = |c: &Composited| {
        c.pixels
            .iter()
            .filter(|p| {
                i32::from(p[0]) - i32::from(p[2]) > 60 && i32::from(p[1]) - i32::from(p[2]) > 40
            })
            .count()
    };

    let object_mode = gpu_or_skip!(composite(&plain, "ko-KR"));
    let edit_empty = gpu_or_skip!(composite_in(&plain, "ko-KR", true));
    let edit_all = gpu_or_skip!(composite_in(&selected, "ko-KR", true));

    assert!(
        orange(&object_mode) > 1000,
        "object mode outlines the selected object: {} orange pixels",
        orange(&object_mode)
    );
    assert!(
        orange(&edit_empty) < 200,
        "with no element selected, edit mode must show a plain wireframe and          not the object outline: {} orange pixels",
        orange(&edit_empty)
    );
    assert!(
        orange(&edit_all) > orange(&edit_empty) + 1000,
        "selecting every element must light the wireframe up: {} then {}",
        orange(&edit_empty),
        orange(&edit_all)
    );

    // And the toolbar grew: the mode button, three element-mode buttons and
    // Extrude all appear, which no object-mode frame has.
    let toolbar_width = |c: &Composited| {
        (0..W)
            .filter(|x| {
                (0..24).any(|y| {
                    let p = c.pixels[(y * W + x) as usize];
                    p[0] > 80 || p[1] > 80 || p[2] > 80
                })
            })
            .count()
    };
    assert!(
        toolbar_width(&edit_all) > toolbar_width(&object_mode),
        "the edit-mode toolbar should be wider than the object-mode one"
    );
}
