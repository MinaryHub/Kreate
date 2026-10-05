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
/// Rows inside the UV panel, below its heading and above the status bar.
///
/// Assertions about the panel are counted here rather than over the lower half
/// of the frame: the viewport is also showing a selected object in the accent,
/// and the shell draws the panel's heading whatever the panel contains. Both
/// satisfied earlier versions of those counts on their own, which is how two
/// of these assertions came to be measuring nothing.
const PANEL_BODY: std::ops::Range<u32> = (H - 240)..(H - 60);

/// The surface format egui composites into.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The format egui samples the viewport through. See the comment at the view.
const SAMPLED_AS: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Composited {
    pixels: Vec<[u8; 4]>,
    /// The rectangle the shell reserved for the viewport, in points.
    viewport_rect: egui::Rect,
    /// The shell state as it stood after the frame.
    ///
    /// Kept because some of what a frame does is not visible in its pixels. The
    /// layout edits an area's header queues are applied *after* the frame, so
    /// "did the split happen" is a question about this and not about the
    /// picture — and a sweep that stopped `draw` from applying them at all
    /// broke nothing until this was here.
    state: kreate::shell::ShellState,
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
    composite_with(doc, locale, edit_mode, |_| {})
}

thread_local! {
    /// A mode for `composite_with` to draw in instead of object or mesh edit
    /// mode — set by `composite_in_mode` for the one call it wraps.
    static FORCED_MODE: std::cell::Cell<Option<kreate_ops::operator::Mode>> = const { std::cell::Cell::new(None) };
}

/// `composite_with` in any mode: pose mode, armature edit mode.
fn composite_in_mode(
    doc: &Document,
    locale: &str,
    mode: kreate_ops::operator::Mode,
    arrange: impl FnOnce(&mut kreate::shell::ShellState),
) -> Option<Composited> {
    FORCED_MODE.with(|m| m.set(Some(mode)));
    let out = composite_with(doc, locale, false, arrange);
    FORCED_MODE.with(|m| m.set(None));
    out
}

/// The same, with a chance to arrange the interface state first.
///
/// For anything that is *opened* rather than always drawn — the right-click
/// menu is the first — so a picture of it can be looked at rather than trusted.
fn composite_with(
    doc: &Document,
    locale: &str,
    edit_mode: bool,
    arrange: impl FnOnce(&mut kreate::shell::ShellState),
) -> Option<Composited> {
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
    // Refined below once the arrangement has run, so an open menu does not
    // overwrite the picture of a closed one.
    let mut label = label;
    let opts = kreate_render::GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);
    let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;

    // --- the surface stand-in, and the offscreen viewport target -------------
    let make_target = |label, usage, view_formats: &[wgpu::TextureFormat]| {
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
            view_formats,
        })
    };
    let surface = make_target(
        "surface",
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        &[],
    );
    let surface_view = surface.create_view(&wgpu::TextureViewDescriptor::default());
    let vp = make_target(
        "viewport",
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        &[SAMPLED_AS],
    );
    let vp_view = vp.create_view(&wgpu::TextureViewDescriptor::default());
    // The scene renders into the sRGB view and egui samples the plain one —
    // the application does exactly this, and for the reason documented on
    // `ViewportTarget::SAMPLED_AS`. Sampling through the sRGB view decodes a
    // second time and composites the whole viewport far too dark, so a harness
    // that did it differently would be picturing a frame nobody sees.
    let vp_sampled = vp.create_view(&wgpu::TextureViewDescriptor {
        format: Some(SAMPLED_AS),
        ..Default::default()
    });

    let mut egui_renderer =
        egui_wgpu::Renderer::new(&gpu.device, FORMAT, egui_wgpu::RendererOptions::default());
    let vp_id =
        egui_renderer.register_native_texture(&gpu.device, &vp_sampled, wgpu::FilterMode::Linear);

    let mut viewport = kreate_render::Viewport::new(&gpu.device, FORMAT, W, H);
    viewport.show_grid = true;
    // The element overlays, the way the application switches them on in
    // edit mode; without this the harness draws edit mode with no vertices.
    viewport.show_elements = edit_mode;

    // --- run the shell -------------------------------------------------------
    let ctx = egui::Context::default();
    kreate_ui::install_fonts(&ctx);
    // The same order the application uses. Without this the harness would be
    // picturing egui's defaults rather than the product, which is the one thing
    // a compositing test must not do.
    kreate_ui::install_theme(&ctx);
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
    let mut state = kreate::shell::ShellState::with_catalog(kreate::shell::load_catalog(locale));
    // What the UV editor shows, built the way the application builds it from
    // the edit session: without this the harness's UV editor always said
    // "press Tab to enter edit mode", in edit mode, and a picture of the
    // editor could not be taken.
    if edit_mode {
        state.uv = kreate_ops::ops::mesh::target_mesh(doc)
            .and_then(|mesh| doc.mesh(mesh))
            .map(|mesh| kreate::uvview::from_bmesh(&mesh.to_bmesh()).unwrap_or_default());
    }
    let mode = FORCED_MODE.with(|m| m.get()).unwrap_or(if edit_mode {
        kreate_ops::operator::Mode::EditMesh
    } else {
        kreate_ops::operator::Mode::Object
    });
    arrange(&mut state);
    if state.context_menu.at.is_some() {
        label = format!("menu-{label}");
    }

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
    // Four passes with the clock advancing, not two at a standstill.
    //
    // Two was enough while everything on screen was a panel. It is not enough
    // for anything that *appears*: egui fades a new `Area` in, and a fade driven
    // by a clock that never moves is caught part way — the first picture of the
    // right-click menu was a translucent ghost with two empty buttons in it, and
    // a test counting changed pixels passed against it.
    // The camera, before the passes rather than after them: the shell draws
    // the lamp and camera glyphs from a projector built out of it, so the
    // frame has to know the camera while it is being drawn. Re-framed each
    // pass as the viewport's aspect settles, and the same camera then renders
    // the picture — one camera for both, or the glyphs sit beside their objects.
    let mut camera = kreate_render::Camera::default();
    for pass in 0..4 {
        let mut raw_input = raw_input.clone();
        raw_input.time = Some(pass as f64 * 0.25);
        camera.frame_sphere(glam::DVec3::ZERO, 2.0, viewport.aspect());
        state.camera_rotation = camera.rotation;
        if viewport_rect.width() > 0.0 {
            state.projector = Some(kreate_ui::object_glyph::Projector {
                eye: camera.eye(),
                rotation: camera.rotation,
                projection: camera.projection_matrix(viewport.aspect()).as_dmat4(),
                rect: viewport_rect,
            });
        }
        let output = ctx.run_ui(raw_input, |ui| {
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
                mode,
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
    // The evaluated meshes, as the application draws them: modifiers applied
    // and bound meshes skinned. Uploading the document's own would picture a
    // posed character standing at rest.
    let mut cache = kreate_scene::modifier::ModifierCache::new();
    let uploaded: Vec<(kreate_scene::ObjectId, kreate_render::GpuMesh)> = doc
        .objects()
        .filter_map(|(id, _)| {
            let mesh = cache.evaluated(doc, id, &kreate_scene::modifier::EvalCtx::still())?;
            // In weight paint mode the painted mesh shows its weights, as the
            // application shows it.
            let painting = mode.acts_on() == kreate_ops::mode::ActsOn::Weights && doc.selection.active == Some(id);
            let mesh = if painting {
                let group = doc.object(id).and_then(|o| o.active_group.clone());
                std::sync::Arc::new(kreate_ops::ops::paint::weight_display(&mesh, group.as_deref()))
            } else {
                mesh
            };
            let cpu = kreate_render::RenderMeshCpu::extract(&mesh);
            Some((
                id,
                kreate_render::GpuMesh::upload(&gpu.device, &viewport.layouts, &cpu, glam::Vec4::new(0.75, 0.72, 0.68, 1.0)),
            ))
        })
        .collect();
    let mut draws: Vec<kreate_render::Draw> = uploaded
        .iter()
        .map(|(id, m)| {
            kreate_render::Draw::new(m, doc.world_transform(*id))
                .selected(!edit_mode && doc.selection.contains(*id))
                .weights(mode.acts_on() == kreate_ops::mode::ActsOn::Weights && doc.selection.active == Some(*id))
        })
        .collect();

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("composite"),
        });
    // The lamp and camera glyphs, the way the frame loop hands them over.
    let shown = match mode.acts_on() {
        kreate_ops::mode::ActsOn::Bones => doc.selection.active.map(|id| (id, kreate::glyphs::Bones::Pose)),
        kreate_ops::mode::ActsOn::BoneJoints => doc.selection.active.map(|id| (id, kreate::glyphs::Bones::Edit)),
        _ => None,
    };
    let (marker_lines, marker_points) = kreate::glyphs::markers_with(doc, shown);
    viewport.set_markers(&gpu.device, &gpu.queue, camera.eye(), &marker_lines, &marker_points);
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
        state,
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
fn the_sidebar_covers_the_viewport_and_takes_the_pointer() {
    // Two claims, one frame, because they are the same rectangle.
    //
    // **It is drawn.** Over the scene, on the right, in the panel colour rather
    // than the viewport's — so the pixels there change when it opens.
    //
    // **It takes the pointer.** The viewport is *painted* rather than being a
    // widget, so the application decides by geometry whether a press is a
    // navigation gesture; without the sidebar registering its rectangle, a drag
    // on a coordinate field would type a number *and* orbit the camera. There
    // is no `wants_pointer_input` that answers this — see
    // `App::press_belongs_to_viewport`, which explains why.
    let closed = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));
    let open = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        state.toggle_sidebar();
    }));

    // The blocker that *appears* when the sidebar opens is the sidebar. Not
    // "the only blocker": the tool column down the left registers one too, in
    // both frames, and a click on a tool must not also select what is behind
    // it. What the sidebar adds is what this is asking after.
    let added: Vec<egui::Rect> = open
        .state
        .viewport_blockers
        .iter()
        .copied()
        .filter(|r| !closed.state.viewport_blockers.contains(r))
        .collect();
    assert_eq!(
        added.len(),
        1,
        "opening the sidebar should add exactly one blocker; closed {:?}, open {:?}",
        closed.state.viewport_blockers,
        open.state.viewport_blockers
    );
    let bar = added[0];

    // Inside the viewport, against its right edge, and a strip rather than the
    // whole thing.
    let vr = open.viewport_rect;
    assert!(
        vr.expand(1.0).contains_rect(bar),
        "{bar:?} is not inside {vr:?}"
    );
    assert!(
        (bar.max.x - vr.max.x).abs() < 1.0,
        "the sidebar is not against the right edge: {bar:?} in {vr:?}"
    );
    assert!(
        bar.width() < vr.width() * 0.6 && bar.width() > 40.0,
        "the sidebar is {} of {} wide",
        bar.width(),
        vr.width()
    );

    // And it is painted: the strip differs from the frame without it, while the
    // left of the viewport does not.
    let band = |from: f32, to: f32| -> usize {
        let mut n = 0;
        for y in (vr.min.y as u32 + 40)..(vr.max.y as u32 - 8) {
            for x in (from as u32)..(to as u32).min(W) {
                if open.at(x, y) != closed.at(x, y) {
                    n += 1;
                }
            }
        }
        n
    };
    let under_bar = band(bar.min.x + 2.0, bar.max.x - 2.0);
    let left_of_it = band(vr.min.x + 2.0, bar.min.x - 8.0);
    assert!(
        under_bar > 2000,
        "only {under_bar} pixels changed where the sidebar is; it did not draw"
    );
    assert!(
        left_of_it * 20 < under_bar,
        "{left_of_it} pixels changed away from the sidebar; it redrew the scene"
    );
}

#[test]
fn the_sidebar_shows_the_active_object_s_placement() {
    // The reason it exists: a coordinate can be typed, not only dragged. The
    // fields come from `object.set_transform`'s own schema, so what this checks
    // is that the schema reached the panel at all — an empty one draws the
    // "nothing active" note instead, and the two differ in ink.
    let mut doc = Document::new();
    let object = doc.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));

    let nothing_active = gpu_or_skip!(composite_with(&doc, "ko-KR", false, |state| {
        state.toggle_sidebar();
    }));
    doc.selection.add(object);
    let active = gpu_or_skip!(composite_with(&doc, "ko-KR", false, |state| {
        state.toggle_sidebar();
    }));

    // The sidebar is the blocker against the viewport's right edge; the tool
    // column is the one down the left.
    let vr = active.viewport_rect;
    let bar = *active
        .state
        .viewport_blockers
        .iter()
        .find(|r| (r.max.x - vr.max.x).abs() < 1.0)
        .expect("a blocker against the right edge, which is the sidebar");
    // The fields are widgets with borders and numbers; the note is a line of
    // dim text. Counted as distinct colours inside the strip, which a row of
    // drag fields has many more of.
    let variety = |c: &Composited| -> usize {
        let mut seen = std::collections::HashSet::new();
        for y in (bar.min.y as u32 + 40)..(bar.max.y as u32 - 8) {
            for x in (bar.min.x as u32 + 2)..(bar.max.x as u32 - 2) {
                seen.insert(c.at(x, y));
            }
        }
        seen.len()
    };
    let with = variety(&active);
    let without = variety(&nothing_active);
    assert!(
        with > without + 8,
        "the panel with an active object has {with} colours against {without}; \
         the transform fields did not draw"
    );
}

#[test]
fn switching_workspace_changes_the_layout_on_screen() {
    // The point of named layouts: one click and the window is arranged for a
    // different job. Checked through the same deferred queue the layout edits
    // use, because it has the same reason to be deferred — the frame is drawing
    // tabs and areas from the list this changes.
    //
    // Asserted as the *viewport rect*, not just the index: a switch that moved
    // the index and went on drawing the old arrangement would pass any test of
    // the model alone, and the application sizes its render target from that
    // rectangle.
    let shipped = kreate_ui::Workspaces::default();
    let uv_editing = shipped
        .iter()
        .position(|space| {
            space
                .screen
                .areas()
                .iter()
                .any(|(_, kind)| *kind == kreate_ui::EditorId::UV_EDITOR)
        })
        .expect("a shipped workspace has a UV editor");
    assert_ne!(
        uv_editing, 0,
        "the fixture must switch away from the default"
    );

    let first = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));
    let switched = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        state.switch_workspace(uv_editing)
    }));

    assert_eq!(
        switched.state.workspaces.active_index(),
        uv_editing,
        "the switch was asked for and never happened"
    );
    // The default layout gives the viewport most of the width; the UV one
    // splits it in half with the UV editor. So the rectangle has to shrink, and
    // by a lot.
    assert!(
        switched.viewport_rect.width() < first.viewport_rect.width() * 0.75,
        "the viewport went from {} to {}; the old layout is still being drawn",
        first.viewport_rect.width(),
        switched.viewport_rect.width()
    );
    // And a UV editor is on screen now, which is what the workspace is for.
    assert!(switched.state.has_editor(kreate_ui::EditorId::UV_EDITOR));
    assert!(!first.state.has_editor(kreate_ui::EditorId::UV_EDITOR));
}

#[test]
fn an_edit_lands_in_the_workspace_that_is_showing() {
    // Otherwise arranging one layout quietly rearranges another, and the user
    // finds out the next time they switch. `screen_mut` going through the active
    // workspace is what prevents it; a field of its own could not.
    let target = 1;
    let c = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        state.switch_workspace(target);
    }));
    assert_eq!(c.state.workspaces.active_index(), target);

    let after = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        state.switch_workspace(target);
        // Queued behind the switch, so it lands in the workspace switched *to*.
        let area = state.screen().areas()[0].0;
        state.split_area(area, kreate_ui::Axis::Horizontal);
    }));

    let untouched = kreate_ui::Workspaces::default();
    for (index, space) in after.state.workspaces.iter().enumerate() {
        let expected = untouched
            .iter()
            .nth(index)
            .expect("the same number of workspaces");
        if index == target {
            assert_ne!(
                space.screen.to_nodes(),
                expected.screen.to_nodes(),
                "the split did not land in the active workspace"
            );
        } else {
            assert_eq!(
                space.screen.to_nodes(),
                expected.screen.to_nodes(),
                "workspace {index} was rearranged behind the user's back"
            );
        }
    }
}

#[test]
fn a_layout_edit_asked_for_during_a_frame_is_applied_after_it() {
    // Splitting and closing are deferred: the frame is drawing against a layout
    // derived from the tree, so editing it where the button is would leave every
    // area after that one placed by a layout that no longer exists.
    //
    // Which means the queue has to actually be drained, and by `draw` rather
    // than by a test calling `apply_area_edits` itself. A sweep that replaced
    // the call with `clear()` — splitting and closing dead in the application,
    // silently — passed everything until this existed.
    let before = kreate::shell::ShellState::default().screen().area_count();

    let split = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        let viewport = state
            .screen()
            .areas()
            .into_iter()
            .find(|(_, kind)| *kind == kreate_ui::EditorId::VIEWPORT)
            .expect("the default layout has a viewport")
            .0;
        state.split_area(viewport, kreate_ui::Axis::Horizontal);
    }));
    assert_eq!(
        split.state.screen().area_count(),
        before + 1,
        "the split was asked for and never happened"
    );

    let closed = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        let properties = state
            .screen()
            .areas()
            .into_iter()
            .find(|(_, kind)| *kind == kreate_ui::EditorId::PROPERTIES)
            .expect("the default layout has a properties editor")
            .0;
        state.close_area(properties);
    }));
    assert_eq!(closed.state.screen().area_count(), before - 1);
    assert!(!closed.state.has_editor(kreate_ui::EditorId::PROPERTIES));
}

#[test]
fn a_second_viewport_does_not_show_the_first_one_s_picture() {
    // One render target, so only one area can be showing a rendered image.
    // Several viewports means several cameras and several targets, which is a
    // real feature and a later one; until then the honest thing is for the
    // extra ones to show their own background rather than a copy of the first
    // one's frame, which would be a picture from the wrong camera.
    let mut screen = kreate_ui::Screen::single(kreate_ui::EditorId::VIEWPORT);
    let first = screen.areas()[0].0;
    screen
        .split(
            first,
            kreate_ui::Axis::Vertical,
            0.5,
            kreate_ui::EditorId::VIEWPORT,
        )
        .expect("split");
    let plan = screen.clone();

    let c = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        *state.screen_mut() = plan.clone()
    }));

    // The cube is at the centre of whichever viewport is rendering, so it is the
    // *variety* of colour that separates a rendered half from an empty one: a
    // flat background is one colour, a shaded cube on a grid is many.
    let vr = c.viewport_rect;
    // Below the view text and right of the tool column, which every viewport
    // draws over its picture: the question is whether the *scene* reached the
    // second viewport, and its own chrome is not the scene.
    let chrome_left = kreate_ui::toolbar::WIDTH as u32 + 24;
    let chrome_top = 60;
    let band = |from: u32, to: u32| -> usize {
        let mut seen: std::collections::HashSet<[u8; 4]> = std::collections::HashSet::new();
        for y in (vr.min.y as u32 + chrome_top)..(vr.max.y as u32 - 8) {
            for x in from..to {
                seen.insert(c.at(x, y));
            }
        }
        seen.len()
    };
    let rendered = band(vr.min.x as u32 + chrome_left, vr.max.x as u32 - 4);
    let other = band(vr.max.x as u32 + 8 + chrome_left, W - 4);
    assert!(
        rendered > 20,
        "the first viewport has only {rendered} distinct colours; nothing rendered"
    );
    assert!(
        other * 4 < rendered,
        "the second viewport has {other} distinct colours against {rendered}; it is \
         showing the first one's picture"
    );
}

#[test]
fn two_uv_editors_keep_their_own_view() {
    // The layout permits two of anything, and view state keyed once would mean
    // they scroll and zoom together, which is baffling. Keyed on the area
    // instead — and the area id is stable for the area's life, so the state
    // stays put when a neighbour is split or closed.
    //
    // Asserted through `uv_at`, which is what the application asks in order to
    // route `G` to the editor under the pointer: with the state shared, both
    // points resolve to the same area, and a UV drag over one panel would move
    // the other one's selection.
    let mut screen = kreate_ui::Screen::single(kreate_ui::EditorId::UV_EDITOR);
    let first = screen.areas()[0].0;
    let second = screen
        .split(
            first,
            kreate_ui::Axis::Vertical,
            0.5,
            kreate_ui::EditorId::UV_EDITOR,
        )
        .expect("split");
    let plan = screen.clone();

    let c = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", true, |state| {
        *state.screen_mut() = plan.clone();
        state.uv = Some(kreate_ui::UvData::default());
    }));

    // A point well inside each half.
    let left = egui::pos2(f32::from(W as u16) * 0.25, f32::from(H as u16) * 0.6);
    let right = egui::pos2(f32::from(W as u16) * 0.75, f32::from(H as u16) * 0.6);
    let a = c.state.uv_at(left).map(|(id, _)| id);
    let b = c.state.uv_at(right).map(|(id, _)| id);

    assert_eq!(
        a,
        Some(first),
        "the left half is not the first editor: {a:?}"
    );
    assert_eq!(b, Some(second), "the right half is not the second: {b:?}");
    assert_ne!(
        a, b,
        "both halves report the same area, so they share a view"
    );
}

#[test]
fn every_area_paints_its_own_background() {
    // The defect this was written after. The shell used to draw each panel with
    // `Panel::left` and friends, which paint a themed frame; laying the areas
    // out with a plain sub-`Ui` instead left every one of them transparent, and
    // nothing else paints there, so they composited as **pure black**.
    //
    // Sampled just inside each area's own boundary, where its background shows
    // before any widget it draws. Pure black is the failure: an area that
    // painted nothing, or one laid out at an inverted rect and so not painted
    // at all.
    let mut layout = kreate_ui::Screen::single(kreate_ui::EditorId::VIEWPORT);
    let viewport = layout.areas()[0].0;
    let outliner = layout
        .split(
            viewport,
            kreate_ui::Axis::Vertical,
            0.6,
            kreate_ui::EditorId::OUTLINER,
        )
        .expect("split the only area");
    layout
        .split(
            outliner,
            kreate_ui::Axis::Horizontal,
            0.5,
            kreate_ui::EditorId::PROPERTIES,
        )
        .expect("split the outliner");
    layout
        .split(
            viewport,
            kreate_ui::Axis::Horizontal,
            0.7,
            kreate_ui::EditorId::UV_EDITOR,
        )
        .expect("split the viewport");
    let expected_areas = layout.area_count();
    assert_eq!(expected_areas, 4, "the fixture should have four areas");

    let plan = layout.clone();
    let c = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        *state.screen_mut() = plan.clone()
    }));

    // The screen occupies what is left after the top bar and the status bar.
    // Taken from the viewport rect the shell reported rather than guessed at:
    // the viewport is one of the areas, so its rect pins the screen's bounds in
    // both axes.
    let vr = c.viewport_rect;
    assert!(vr.width() > 0.0 && vr.height() > 0.0, "no viewport: {vr:?}");

    let mut sampled = 0;
    for (id, kind) in layout.areas() {
        let rect = layout
            .layout(egui::Rect::from_min_max(
                egui::pos2(0.0, vr.min.y - 30.0),
                egui::pos2(f32::from(W as u16), f32::from(H as u16) - 24.0),
            ))
            .areas
            .iter()
            .find(|a| a.id == id)
            .expect("every area is laid out")
            .rect;
        // Well inside, and below the header strip, so the sample is the
        // editor's own background rather than the header's or the border's.
        let x = (rect.center().x) as u32;
        let y = (rect.min.y + rect.height() * 0.75) as u32;
        if x >= W || y >= H {
            continue;
        }
        let p = c.at(x, y);
        assert!(
            p[0] > 4 || p[1] > 4 || p[2] > 4,
            "{kind:?} at ({x},{y}) is pure black: {p:?} — the area painted nothing"
        );
        sampled += 1;
    }
    assert_eq!(
        sampled, expected_areas,
        "only {sampled} of {expected_areas} areas were sampled"
    );
}

#[test]
fn the_viewport_follows_the_layout_it_is_given() {
    // The point of the layout being data: the shell has to draw where the tree
    // says, not where a `Panel::left` call happens to put things. Checked as
    // the *reported viewport rect*, because that is what the application sizes
    // its render target from — a shell that ignored the tree would hand the
    // renderer the wrong rectangle and the image would be stretched.
    //
    // Two arrangements that put the viewport on opposite sides, so a shell that
    // returned a constant would fail one of them.
    let left = {
        let mut screen = kreate_ui::Screen::single(kreate_ui::EditorId::VIEWPORT);
        let viewport = screen.areas()[0].0;
        screen
            .split(
                viewport,
                kreate_ui::Axis::Vertical,
                0.3,
                kreate_ui::EditorId::OUTLINER,
            )
            .expect("split");
        screen
    };
    let right = {
        let mut screen = kreate_ui::Screen::single(kreate_ui::EditorId::OUTLINER);
        let outliner = screen.areas()[0].0;
        screen
            .split(
                outliner,
                kreate_ui::Axis::Vertical,
                0.3,
                kreate_ui::EditorId::VIEWPORT,
            )
            .expect("split");
        screen
    };

    let narrow = {
        let plan = left.clone();
        gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
            *state.screen_mut() = plan.clone()
        }))
    };
    let wide = {
        let plan = right.clone();
        gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
            *state.screen_mut() = plan.clone()
        }))
    };

    // Thirty per cent of the width on the left, seventy on the right.
    assert!(
        narrow.viewport_rect.width() < wide.viewport_rect.width(),
        "the viewport was {} wide either way; the tree is being ignored",
        narrow.viewport_rect.width()
    );
    assert!(
        narrow.viewport_rect.min.x < wide.viewport_rect.min.x,
        "the viewport did not move: {:?} then {:?}",
        narrow.viewport_rect,
        wide.viewport_rect
    );
    // And the wide one really is most of the window, so this is not two
    // arbitrary numbers that happen to differ.
    assert!(
        wide.viewport_rect.width() > f32::from(W as u16) * 0.55,
        "the right-hand viewport is only {} wide",
        wide.viewport_rect.width()
    );
}

#[test]
fn the_chrome_is_warm_and_the_viewport_is_not() {
    // The whole premise of the theme, checked where it actually matters: on
    // the composited frame, not on the token table. Warm chrome makes a
    // neutral grey read cool, so the viewport must stay neutral or the tool is
    // misreporting the one colour the user is judging.
    //
    // Two failures this catches that a unit test cannot: the theme never being
    // installed on the context at all, and the viewport clear drifting warm to
    // "match" the panels.
    let c = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));

    // Derived from where the viewport ended up rather than hardcoded. The
    // layout is data now, and the coordinates that used to be written in here
    // went stale the first time the default arrangement changed — they were
    // sampling the left edge, which is viewport and not chrome any more.
    //
    // Just past the boundary on the right, where the editor's own background
    // shows before any row it draws, and in the status bar under everything.
    let vr = c.viewport_rect;
    let inside_column = (vr.max.x + kreate_ui::area::HANDLE + 2.0) as u32;
    let panel_samples = [
        (inside_column, (vr.min.y + vr.height() * 0.25) as u32),
        (inside_column, (vr.min.y + vr.height() * 0.75) as u32),
        (W / 2, H - 4),
    ];
    for (x, y) in panel_samples {
        let p = c.at(x, y);
        let (r, b) = (i32::from(p[0]), i32::from(p[2]));
        assert!(
            r > b,
            "panel pixel at ({x},{y}) is not warm: {p:?} — is the theme installed?"
        );
    }

    // And an empty corner of the viewport, away from the cube and the axis
    // lines, which are deliberately coloured.
    let x = (vr.min.x + vr.width() * 0.08) as u32;
    let y = (vr.min.y + vr.height() * 0.08) as u32;
    let p = c.at(x, y);
    let (r, g, b) = (i32::from(p[0]), i32::from(p[1]), i32::from(p[2]));
    assert!(
        (r - g).abs() <= 3 && (r - b).abs() <= 4,
        "the viewport background has a colour cast at ({x},{y}): {p:?}"
    );
    // Neutral, not black: a clear that failed to happen would also pass the
    // test above.
    assert!(
        (20..=48).contains(&r),
        "the viewport background is not the expected dark neutral: {p:?}"
    );
}

#[test]
fn the_modal_header_draws_over_the_viewport_in_the_constrained_axis_colour() {
    // A transform in progress is the one time the tool has to talk
    // continuously, and the status bar is the wrong place for it: the user is
    // looking at the model. Rendered rather than asserted on state, because
    // "the header is set" and "the header is on screen where the eye is" are
    // different claims.
    let axis = 0; // X
    let with = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        state.modal = Some(kreate::shell::ModalHeader {
            text: "이동 X  1.400 0.000 0.000".into(),
            axis: Some(axis),
            tool: None,
        });
    }));
    let without = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));

    let vr = with.viewport_rect;
    // The band the header occupies: just under the top of the viewport, across
    // the middle third where a centred pill lands.
    let band = |c: &Composited| {
        let mut changed = 0usize;
        let mut axis_coloured = 0usize;
        let want = kreate_render::palette::rgb8(kreate_render::palette::AXIS_X);
        for y in (vr.min.y as u32 + 8)..(vr.min.y as u32 + 40) {
            for x in (vr.center().x as u32 - 160)..(vr.center().x as u32 + 160) {
                let p = c.at(x, y);
                if p != without.at(x, y) {
                    changed += 1;
                }
                let near = (0..3).all(|i| (i32::from(p[i]) - i32::from(want[i])).abs() < 40);
                if near {
                    axis_coloured += 1;
                }
            }
        }
        (changed, axis_coloured)
    };

    let (changed, axis_coloured) = band(&with);
    assert!(
        changed > 2000,
        "only {changed} pixels changed where the header should be; it did not draw"
    );
    // And the border is the X colour, not the generic accent. This is what the
    // one axis table buys: the same red as the grid and the transform field.
    assert!(
        axis_coloured > 40,
        "found {axis_coloured} pixels near the X axis colour; the header is not          wearing the constrained axis"
    );

    // Nothing outside the viewport moved: the header is an overlay, not a panel.
    let left_panel_changed = (0..H)
        .filter(|y| with.at(20, *y) != without.at(20, *y))
        .count();
    assert_eq!(
        left_panel_changed, 0,
        "{left_panel_changed} pixels changed in the side panel"
    );
}

/// Puts a UV editor along the bottom of the screen, under the viewport.
///
/// The layout is data now, so a test that wants a UV editor has to say where.
/// Explicit rather than "point some area at it": `PANEL_BODY` below is a band of
/// scanlines, and it has to fall inside the editor's *body* rather than across
/// its header — the header draws either way, and a scan that included it was
/// satisfied by it once already.
fn uv_below_viewport(state: &mut kreate::shell::ShellState) {
    let mut screen = kreate_ui::Screen::single(kreate_ui::EditorId::VIEWPORT);
    let viewport = screen.areas()[0].0;
    screen
        .split(
            viewport,
            kreate_ui::Axis::Horizontal,
            0.65,
            kreate_ui::EditorId::UV_EDITOR,
        )
        .expect("the screen has one area");
    *state.screen_mut() = screen;
}

#[test]
fn the_uv_editor_draws_its_map_and_says_so_when_there_is_none() {
    // Two states worth a picture, because they are the two the user actually
    // meets: a mesh with UVs, and a mesh without. The second used to be an
    // empty square with no explanation, which reads as a broken panel.
    let with_uvs = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", true, |state| {
        uv_below_viewport(state);
        // Two faces of a box-projected cube, one of them selected, in the
        // arena's own corner numbering — which is what the panel is given
        // and what the operators take.
        state.uv = Some(kreate_ui::UvData {
            uvs: vec![
                egui::vec2(0.1, 0.1),
                egui::vec2(0.4, 0.1),
                egui::vec2(0.4, 0.4),
                egui::vec2(0.1, 0.4),
                egui::vec2(0.6, 0.6),
                egui::vec2(0.9, 0.6),
                egui::vec2(0.9, 0.9),
                egui::vec2(0.6, 0.9),
            ],
            selected: vec![true, true, true, true, false, false, false, false],
            offsets: vec![0, 4, 8],
            corners: vec![0, 1, 2, 3, 4, 5, 6, 7],
        });
    }));

    let without = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", true, |state| {
        uv_below_viewport(state);
        state.uv = Some(kreate_ui::UvData::default());
    }));

    // The panel is at the bottom, so the two frames must differ there and
    // agree above it — a UV panel that redrew the viewport would be a layout
    // bug hiding as a rendering one.
    let band = |c: &Composited, from: u32, to: u32| -> usize {
        let mut n = 0;
        for y in from..to {
            for x in 0..W {
                if c.at(x, y) != without.at(x, y) {
                    n += 1;
                }
            }
        }
        n
    };
    let lower = band(&with_uvs, H * 3 / 4, H - 40);
    assert!(
        lower > 500,
        "only {lower} pixels differ in the UV panel; the map did not draw"
    );

    // The selected corners wear the accent. Checked against the palette, so a
    // panel that invented its own orange would fail here — and counted only
    // inside the panel, because the 3D viewport is *also* showing a selected
    // cube in the accent. The first version of this scan covered the lower
    // half of the whole frame and was satisfied by the cube, so recolouring
    // the panel's selection changed nothing.
    let accent = kreate_render::palette::rgb8(kreate_render::palette::ACCENT);
    let mut accented = 0;
    for y in PANEL_BODY {
        for x in 0..W {
            let p = with_uvs.at(x, y);
            if (0..3).all(|i| (i32::from(p[i]) - i32::from(accent[i])).abs() < 40) {
                accented += 1;
            }
        }
    }
    assert!(
        accented > 4,
        "found {accented} accent-coloured pixels in the UV panel; the selection is not shown"
    );

    // And the empty case says something rather than drawing a blank square.
    // Below the panel's heading, which the shell draws either way: the first
    // version of this scan included the heading and was satisfied by it, so
    // deleting the note itself changed nothing.
    let mut empty_ink = 0;
    for y in PANEL_BODY {
        for x in 0..W {
            let p = without.at(x, y);
            if p[0] > 80 && p[1] > 74 {
                empty_ink += 1;
            }
        }
    }
    assert!(
        empty_ink > 60,
        "the empty UV panel drew {empty_ink} light pixels below its heading; it should be explaining itself rather than showing a blank square"
    );
}

#[test]
fn many_recovered_documents_do_not_eat_the_window() {
    // A stale session directory per crashed process is normal, and a run of
    // interrupted tests left forty-two. The offer once filled two thirds of
    // the window; then it was a bounded panel across the top; now it is a
    // submenu of File, the way Blender keeps it, and costs the viewport
    // nothing at all while it waits there.
    //
    // Measured as the viewport's height, because that is what the user loses.
    let none = gpu_or_skip!(composite(&doc_with_cube(), "ko-KR"));
    let many = gpu_or_skip!(composite_with(&doc_with_cube(), "ko-KR", false, |state| {
        state.recoverable = (0..60)
            .map(|i| kreate::shell::RecoveryOffer {
                path: std::path::PathBuf::from(format!("/tmp/session-{i}/recovery-loop.kreate")),
                name: "recovery-loop.kreate".into(),
                age: format!("{i}분 전"),
            })
            .collect();
    }));

    let lost = none.viewport_rect.height() - many.viewport_rect.height();
    assert!(
        lost.abs() < 1.0,
        "sixty recovery offers cost the viewport {lost} points of height; the \
         offer belongs under File and should cost nothing"
    );

    // And the user is still told: the status bar differs from the frame with
    // nothing to recover, where the hint is.
    let bar = |c: &Composited| -> usize {
        let mut n = 0;
        for y in (H - 30)..H {
            for x in 0..(W / 2) {
                let p = c.at(x, y);
                if p[0] > 60 || p[1] > 60 || p[2] > 60 {
                    n += 1;
                }
            }
        }
        n
    };
    assert!(
        bar(&many) > bar(&none) + 40,
        "the status bar says nothing about {} recovered documents",
        many.state.recoverable.len()
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
    //
    // **Counted inside the viewport only.** The claim here is about what the
    // renderer draws, and the chrome wears the same accent for everything it
    // marks as selected — the workspace tab, the mode label, the element
    // buttons. Counting the whole frame made this test fail the day the
    // element buttons became painted icons instead of words, which changed
    // nothing about the picture it is asking after.
    let orange = |c: &Composited| {
        let r = c.viewport_rect;
        let mut n = 0;
        for y in (r.top().max(0.0) as u32)..(r.bottom().min(H as f32) as u32) {
            for x in (r.left().max(0.0) as u32)..(r.right().min(W as f32) as u32) {
                let p = c.at(x, y);
                if i32::from(p[0]) - i32::from(p[2]) > 60
                    && i32::from(p[1]) - i32::from(p[2]) > 40
                {
                    n += 1;
                }
            }
        }
        n
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

    // And the *viewport's header* grew: Mesh, Edge and Face and the three
    // element buttons appear, which no object-mode frame has.
    //
    // The viewport's header and not the top bar. The top bar used to carry
    // the mode's tools and this measured it there; since the editors took
    // their menus into their own headers — Blender's arrangement — the top bar
    // is the same in both modes by design, and a test that wanted it to
    // change would be asking for the old layout back.
    let header_width = |c: &Composited| {
        let top = c.viewport_rect.top() as u32;
        let band = top.saturating_sub(28)..top;
        let right = c.viewport_rect.right() as u32;
        (0..right.min(W))
            .filter(|x| {
                band.clone().any(|y| {
                    let p = c.pixels[(y * W + x) as usize];
                    p[0] > 80 || p[1] > 80 || p[2] > 80
                })
            })
            .count()
    };
    assert!(
        header_width(&edit_all) > header_width(&object_mode),
        "the edit-mode viewport header should be wider than the object-mode one: {} vs {}",
        header_width(&edit_all),
        header_width(&object_mode)
    );
}

#[test]
fn the_right_click_menu_draws_over_the_viewport() {
    // A menu that is built but never painted is the failure this catches: the
    // data is right, every test about its contents passes, and nothing appears
    // on screen. Checked by the picture, because that is the claim.
    use kreate_ui::command_info::CommandInfo;
    use kreate_ui::context_menu::{Entry, Group};

    let doc = doc_with_cube();
    let closed = composite(&doc, "ko-KR");
    let Some(closed) = closed else {
        eprintln!("no usable GPU adapter; skipping");
        return;
    };

    let open = composite_with(&doc, "ko-KR", false, |state| {
        state.context_menu.groups = vec![Group {
            title: "변형".to_string(),
            entries: vec![
                Entry::new(
                    "object.translate",
                    CommandInfo {
                        name: "이동".into(),
                        description: "선택한 오브젝트를 옮깁니다.".into(),
                        shortcut: "M".into(),
                    },
                ),
                Entry::new(
                    "object.rotate",
                    CommandInfo {
                        name: "회전".into(),
                        description: "선택한 오브젝트를 돌립니다.".into(),
                        shortcut: "R".into(),
                    },
                ),
            ],
        }];
        // Well inside the viewport, so the menu is over the render rather than
        // over a panel.
        state.context_menu.at = Some(egui::pos2(420.0, 300.0));
    })
    .expect("the adapter was there a moment ago");
    open.dump("menu");

    // The menu is a light popup over a dark viewport, so it shows up as a
    // change to a lot of pixels in that region — and only in that region.
    let changed = |x0: u32, y0: u32, x1: u32, y1: u32| {
        let mut n = 0;
        for y in y0..y1 {
            for x in x0..x1 {
                if closed.at(x, y) != open.at(x, y) {
                    n += 1;
                }
            }
        }
        n
    };
    let under_menu = changed(420, 300, 620, 420);
    assert!(
        under_menu > 4_000,
        "only {under_menu} pixels changed where the menu should be"
    );

    // And it has to be *readable*, not merely present. A first version of this
    // test counted changed pixels alone and passed against a menu whose frame
    // drew and whose text did not — a translucent ghost with two empty buttons
    // in it. Text is high-contrast detail, so it shows up as many distinct
    // luminances in a small region; a blank panel has two or three.
    let mut shades = std::collections::HashSet::new();
    for y in 300..420 {
        for x in 420..620 {
            let p = open.at(x, y);
            let l = (p[0] as u32 * 30 + p[1] as u32 * 59 + p[2] as u32 * 11) / 100;
            shades.insert(l / 4);
        }
    }
    assert!(
        shades.len() > 12,
        "the menu region has only {} distinct shades, so nothing legible is in it",
        shades.len()
    );
    // The interface elsewhere is untouched: a popup that repainted the panels
    // would be drawing in the wrong place or at the wrong size.
    let far_away = changed(0, 0, 200, 100);
    assert_eq!(far_away, 0, "{far_away} pixels changed away from the menu");

    // And it is a menu rather than a band across the window. The rows lay out
    // right-to-left so each shortcut sits at the row's far edge, and the far
    // edge of an unconstrained area is the edge of the window — the first
    // picture of this menu ran to the viewport's right side with its shortcuts
    // six hundred pixels from their labels.
    let mut widest = 0;
    for y in 300..420 {
        let mut leftmost = None;
        let mut rightmost = 0;
        for x in 220..1020 {
            if closed.at(x, y) != open.at(x, y) {
                leftmost.get_or_insert(x);
                rightmost = x;
            }
        }
        if let Some(left) = leftmost {
            widest = widest.max(rightmost - left);
        }
    }
    assert!(
        widest < 320,
        "the menu is {widest} pixels wide, so it is stretching to fill the viewport"
    );
}

#[test]
fn a_menu_row_shows_its_shortcut() {
    // The whole reason the menu carries them. Checked as pixels: the shortcut is
    // drawn at the right-hand end of the row, so removing it leaves that column
    // empty and the picture changes there.
    use kreate_ui::command_info::CommandInfo;
    use kreate_ui::context_menu::{Entry, Group};

    let doc = doc_with_cube();
    let entry = |shortcut: &str| Group {
        title: String::new(),
        entries: vec![Entry::new(
            "object.translate",
            CommandInfo {
                name: "이동".into(),
                description: String::new(),
                shortcut: shortcut.to_string(),
            },
        )],
    };

    let Some(with) = composite_with(&doc, "ko-KR", false, |state| {
        state.context_menu.groups = vec![entry("M")];
        state.context_menu.at = Some(egui::pos2(420.0, 300.0));
    }) else {
        eprintln!("no usable GPU adapter; skipping");
        return;
    };
    let without = composite_with(&doc, "ko-KR", false, |state| {
        state.context_menu.groups = vec![entry("")];
        state.context_menu.at = Some(egui::pos2(420.0, 300.0));
    })
    .expect("adapter");

    let mut differing = 0;
    for y in 300..340 {
        for x in 420..640 {
            if with.at(x, y) != without.at(x, y) {
                differing += 1;
            }
        }
    }
    assert!(
        differing > 10,
        "the row looks the same with and without a shortcut ({differing} pixels)"
    );
}

/// A picture of the Material tab, for looking at rather than asserting on.
#[test]
fn shot_material_tab() {
    // Two documents side by side: the cube a new document gives you, and the
    // same cube with a material on it. The tab is the same tab.
    let bare = doc_with_cube();
    let mut dressed = doc_with_cube();
    {
        let mesh = dressed.meshes().next().map(|(id, _)| id).expect("a mesh");
        let material = dressed.add_material(kreate_scene::Material {
            name: "Material".into(),
            base_color: kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
            ..Default::default()
        });
        dressed.set_mesh_slot(mesh, 0, Some(material));
    }
    let mut textured = doc_with_cube();
    {
        let mesh = textured.meshes().next().map(|(id, _)| id).expect("a mesh");
        let image = textured.add_image(kreate_scene::Image::new(
            "bricks.png",
            512,
            512,
            kreate_scene::PixelFormat::Rgba8,
        ));
        let normal = textured.add_image(kreate_scene::Image::new(
            "bricks_normal.png",
            512,
            512,
            kreate_scene::PixelFormat::Rgba8,
        ));
        let material = textured.add_material(kreate_scene::Material {
            name: "Brick".into(),
            base_color_image: Some(image),
            normal_image: Some(normal),
            ..Default::default()
        });
        textured.set_mesh_slot(mesh, 0, Some(material));
    }
    let mut two = doc_with_cube();
    {
        let mesh = two.meshes().next().map(|(id, _)| id).expect("a mesh");
        let body = two.add_material(kreate_scene::Material {
            name: "Body".into(),
            ..Default::default()
        });
        let trim = two.add_material(kreate_scene::Material {
            name: "Trim".into(),
            ..Default::default()
        });
        two.set_mesh_slots(mesh, vec![Some(body), Some(trim), None]);
    }
    for (label, doc) in [
        ("bare", &bare),
        ("dressed", &dressed),
        ("textured", &textured),
        ("slots", &two),
    ] {
        let Some(shot) = composite_with(doc, "en", false, |state| {
            let areas: Vec<_> = state.screen().areas().iter().map(|(a, _)| *a).collect();
            for area in areas {
                state.set_properties_tab(area, kreate::shell::PropertiesTab::Material);
            }
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(&format!("material-{label}"));
    }
}

/// A picture of the top row in both modes, for looking at.
#[test]
fn shot_menu_row() {
    let doc = doc_with_cube();
    for (label, locale, edit) in [
        ("object", "en", false),
        ("edit", "en", true),
        ("edit-ko", "ko-KR", true),
    ] {
        let Some(shot) = composite_in(&doc, locale, edit) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(&format!("menu-{label}"));
    }
}

/// A picture of each properties tab against a realistic document.
#[test]
fn shot_every_tab() {
    let mut doc = Document::new();
    kreate::startup::starting_scene(&mut doc);
    let lamp = doc
        .objects()
        .find(|(_, o)| matches!(o.data, kreate_scene::document::ObjectData::Light(_)))
        .map(|(id, _)| id);
    let cube = doc
        .objects()
        .find(|(_, o)| matches!(o.data, kreate_scene::document::ObjectData::Mesh(_)))
        .map(|(id, _)| id);

    for (label, tab, active) in [
        ("object", kreate::shell::PropertiesTab::Object, cube),
        ("data-light", kreate::shell::PropertiesTab::Data, lamp),
        ("world", kreate::shell::PropertiesTab::World, cube),
        ("modifiers", kreate::shell::PropertiesTab::Modifiers, cube),
        ("tool", kreate::shell::PropertiesTab::Tool, cube),
    ] {
        let mut doc = doc.clone();
        doc.selection.clear();
        if let Some(id) = active {
            doc.selection.add(id);
        }
        let Some(shot) = composite_with(&doc, "en", false, |state| {
            let areas: Vec<_> = state.screen().areas().iter().map(|(a, _)| *a).collect();
            for area in areas {
                state.set_properties_tab(area, tab);
            }
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(&format!("tab-{label}"));
    }
}

/// A cube with a point light in frame beside it, shown or hidden.
fn doc_with_lamp(visible: bool) -> (Document, glam::DVec3) {
    let mut doc = doc_with_cube();
    let at = glam::DVec3::new(1.3, -0.9, 1.4);
    let light = doc.add_light(kreate_scene::Light::of_kind(
        "Lamp",
        kreate_scene::LightKind::BULB,
    ));
    let mut lamp = Object::new("Lamp", ObjectData::Light(light));
    lamp.transform.translation = at;
    lamp.visible = visible;
    doc.add_object(lamp);
    (doc, at)
}

#[test]
fn a_lamp_is_drawn_where_it_is() {
    // The complaint this answers: adding a light changed nothing on screen. A
    // lamp has no surface, so the renderer draws nothing for it; the interface
    // draws a glyph over the picture at the lamp's projected position. Which
    // is checked *at that position* — a glyph somewhere else would be a lamp
    // the user cannot find.
    let (shown, at) = doc_with_lamp(true);
    let (hidden, _) = doc_with_lamp(false);
    let with = gpu_or_skip!(composite(&shown, "en"));
    let without = gpu_or_skip!(composite(&hidden, "en"));
    with.dump("lamp");

    let projector = with.state.projector.expect("the frame knew its camera");
    let centre = projector.project(at).expect("the lamp is in front of the camera");
    assert!(
        with.viewport_rect.contains(centre),
        "the fixture's lamp is off screen at {centre:?}; move it"
    );

    // Pixels that differ between the two frames, within the glyph's reach of
    // the lamp — and none far from it, since hiding a lamp changes nothing
    // else about a studio-shaded picture.
    let differ = |from: f32, to: f32| -> usize {
        let mut n = 0;
        for y in (with.viewport_rect.top() as u32)..(with.viewport_rect.bottom() as u32) {
            for x in (with.viewport_rect.left() as u32)..(with.viewport_rect.right() as u32) {
                let d = egui::pos2(x as f32, y as f32).distance(centre);
                if d < from || d >= to {
                    continue;
                }
                if with.at(x, y) != without.at(x, y) {
                    n += 1;
                }
            }
        }
        n
    };
    let near = differ(0.0, 14.0);
    assert!(near > 30, "only {near} pixels changed around the lamp at {centre:?}");
    // The ground line runs from the lamp down to z = 0, so the band between
    // is allowed to differ; well beyond both, nothing may.
    let foot = projector
        .project(glam::DVec3::new(at.x, at.y, 0.0))
        .expect("the foot is in front of the camera");
    let reach = centre.distance(foot) + 20.0;
    let far = differ(reach, 10_000.0);
    assert!(
        far < 20,
        "{far} pixels changed far from the lamp; hiding it should touch only its glyph"
    );
}

/// A picture of each workspace, in edit mode, for looking at.
#[test]
fn shot_workspaces() {
    let doc = doc_with_cube();
    for (index, label) in [(1usize, "modeling"), (2, "uv-editing")] {
        let Some(shot) = composite_with(&doc, "en", true, |state| {
            state.switch_workspace(index);
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(&format!("workspace-{label}"));
    }
}

/// A picture of the Scripting workspace with a few lines in the console.
#[test]
fn shot_scripting() {
    use kreate_ui::console::{ConsoleLine, LineKind};
    let doc = doc_with_cube();
    let Some(shot) = composite_with(&doc, "en", false, |state| {
        state.switch_workspace(5);
        state.console_log = vec![
            ConsoleLine {
                text: "kreate.ops.object.add_primitive(shape=1)".into(),
                kind: LineKind::History,
            },
            ConsoleLine {
                text: "kreate.ops.object.translate(translation=(1.4, 0.0, 0.0))".into(),
                kind: LineKind::History,
            },
            ConsoleLine {
                text: "mesh.nothing()".into(),
                kind: LineKind::Input,
            },
            ConsoleLine {
                text: "no operator called `mesh.nothing`".into(),
                kind: LineKind::Error,
            },
        ];
    }) else {
        eprintln!("no GPU; skipping");
        return;
    };
    shot.dump("workspace-scripting");
}

/// A picture of the Scripting workspace with a conversation in the Claude panel.
#[test]
fn shot_claude() {
    use kreate_ui::chat::{ChatMessage, ChatRole};
    let doc = doc_with_cube();
    for (label, locale) in [("claude", "en"), ("claude-ko", "ko-KR")] {
        let Some(shot) = composite_with(&doc, locale, false, |state| {
            state.switch_workspace(5);
            state.chat_busy = true;
            state.chat_account = kreate_ui::chat::ChatAccount::SignedIn {
                email: "artist@studio.kr".into(),
                detail: "Studio (team)".into(),
            };
            state.chat_log = vec![
                ChatMessage {
                    role: ChatRole::User,
                    text: "빨간 구를 원점 위 2m에 하나 추가해줘".into(),
                },
                ChatMessage {
                    role: ChatRole::Tool,
                    text: "run_script: kreate.ops.object.add_primitive(shape=\"uv_sphere\", location=(0.0, 0.0, 2.0))".into(),
                },
                ChatMessage {
                    role: ChatRole::Error,
                    text: "no object called Nope".into(),
                },
                ChatMessage {
                    role: ChatRole::Claude,
                    text: "원점 위 2m 지점 (0, 0, 2)에 빨간 구를 추가했습니다. 새 머티리얼을 만들어 빨간색으로 지정했습니다.".into(),
                },
                ChatMessage {
                    role: ChatRole::Note,
                    text: "Stopped.".into(),
                },
            ];
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(&format!("workspace-{label}"));
    }
}

/// The Claude panel before an account is signed in, and while one is.
#[test]
fn shot_claude_sign_in() {
    use kreate_ui::chat::ChatAccount;
    let doc = doc_with_cube();
    for (label, account) in [
        ("claude-signed-out-ko", ChatAccount::SignedOut),
        ("claude-signing-in-ko", ChatAccount::SigningIn { url: Some("https://claude.com/cai/oauth/authorize".into()) }),
    ] {
        let Some(shot) = composite_with(&doc, "ko-KR", false, |state| {
            state.switch_workspace(5);
            state.chat_account = account.clone();
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(&format!("workspace-{label}"));
    }
}

/// Help ▸ Contact's form, with a message being written, and About.
#[test]
fn shot_contact() {
    let doc = doc_with_cube();
    let Some(shot) = composite_with(&doc, "ko-KR", false, |state| {
        state.contact.open();
        state.contact.kind = 1;
        state.contact.email = "artist@studio.kr".into();
        state.contact.message = "회전체 단면을 곡선으로 그릴 수 있으면 좋겠습니다.".into();
        state.contact.status = kreate_ui::contact::ContactStatus::Failed("the contact service is not set up".into());
    }) else {
        eprintln!("no GPU; skipping");
        return;
    };
    shot.dump("contact-ko");
    let Some(shot) = composite_with(&doc, "ko-KR", false, |state| state.about_open = true) else { return };
    shot.dump("about-ko");
}

/// A picture of the Shading workspace, with a textured material's graph.
#[test]
fn shot_shading() {
    for (label, locale) in [("shading", "en"), ("shading-ko", "ko-KR")] {
        let mut doc = doc_with_cube();
        let mesh = doc.meshes().next().map(|(id, _)| id).expect("a mesh");
        let colour = doc.add_image(kreate_scene::Image::new(
            "bricks.png",
            64,
            64,
            kreate_scene::PixelFormat::Rgba8,
        ));
        let normal = doc.add_image(kreate_scene::Image::new(
            "bricks_normal.png",
            64,
            64,
            kreate_scene::PixelFormat::Rgba8,
        ));
        let mut material = kreate_scene::Material {
            name: "Brick".into(),
            base_color_image: Some(colour),
            normal_image: Some(normal),
            ..Default::default()
        };
        material.graph = Some(material.shading_graph());
        let material = doc.add_material(material);
        doc.set_mesh_slot(mesh, 0, Some(material));
        let Some(shot) = composite_with(&doc, locale, false, |state| {
            state.switch_workspace(3);
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(&format!("workspace-{label}"));
    }
}

/// A picture of F3's search, typed into, in both languages.
#[test]
fn shot_search() {
    let doc = doc_with_cube();
    for (label, locale, edit, query) in [
        ("search", "en", false, "add"),
        ("search-edit-ko", "ko-KR", true, "ext"),
    ] {
        let Some(shot) = composite_with(&doc, locale, edit, |state| {
            state.palette.open_with(query);
            // The harness has no keymap, so the shortcut the application
            // would resolve is given here, to see the column.
            state.command_info.insert(
                "mesh.extrude_region".to_string(),
                kreate_ui::command_info::CommandInfo {
                    name: String::new(),
                    description: String::new(),
                    shortcut: "E".to_string(),
                },
            );
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(label);
    }
}


/// A picture of the shading and view pies, open at the middle of the window.
#[test]
fn shot_pies() {
    let doc = doc_with_cube();
    for (label, locale, pie) in [("pie-shading", "en", "shading"), ("pie-view-ko", "ko-KR", "view")] {
        let Some(shot) = composite_with(&doc, locale, false, |state| {
            state.pie.open(pie, egui::pos2(500.0, 400.0), None, 0.0);
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(label);
    }
}

/// A picture of the Layout workspace's timeline, with keys on it.
#[test]
fn shot_timeline() {
    let doc = doc_with_cube();
    for (label, locale) in [("timeline", "en"), ("timeline-ko", "ko-KR")] {
        let Some(shot) = composite_with(&doc, locale, false, |state| {
            // The application publishes this each frame; the harness has no
            // application, so it is given here.
            state.timeline = kreate_ui::timeline::TimelineView {
                frame: 24.0,
                start: 1,
                end: 120,
                playing: false,
                autokey: true,
                keys: vec![1.0, 24.0, 60.0, 96.0],
            };
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(label);
    }
}

/// A picture of the Animation workspace: a keyed cube in the dope sheet.
#[test]
fn shot_dope_sheet() {
    use kreate_scene::animation::{Action, Interpolation};
    for (label, locale) in [("dope-sheet", "en"), ("dope-sheet-ko", "ko-KR")] {
        let mut doc = doc_with_cube();
        let cube = doc.objects().next().map(|(id, _)| id).expect("a cube");
        let mut action = Action::new("CubeAction");
        for (frame, x) in [(1.0, 0.0), (24.0, 2.0), (60.0, -1.0)] {
            for i in 0..3u8 {
                action.curve_mut("location", i).insert(frame, if i == 0 { x } else { 0.0 }, Interpolation::Bezier);
            }
        }
        for i in 0..3u8 {
            action.curve_mut("scale", i).insert(40.0, 1.5, Interpolation::Bezier);
        }
        action.select_all_with(|_| false);
        action.select_range(Some("location"), 24.0, 24.0, true);
        let action = doc.add_action(action);
        doc.object_mut(cube).unwrap().action = Some(action);
        doc.selection.clear();
        doc.selection.add(cube);
        doc.playback.frame = 24.0;
        let Some(shot) = composite_with(&doc, locale, false, |state| {
            state.switch_workspace(4);
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(label);
    }
}

/// The dope sheet's area turned into a graph editor: a bouncing X, a flat
/// Y and Z, one key selected with its handles showing.
#[test]
fn shot_graph_editor() {
    use kreate_scene::animation::{Action, Interpolation};
    for (label, locale) in [("graph-editor", "en"), ("graph-editor-ko", "ko-KR")] {
        let mut doc = doc_with_cube();
        let cube = doc.objects().next().map(|(id, _)| id).expect("a cube");
        let mut action = Action::new("CubeAction");
        for (frame, x, z) in [(1.0, 0.0, 0.0), (24.0, 2.0, 1.0), (48.0, -1.0, 0.5), (72.0, 1.0, 0.0)] {
            action.curve_mut("location", 0).insert(frame, x, Interpolation::Bezier);
            action.curve_mut("location", 2).insert(frame, z, Interpolation::Bezier);
        }
        action.select_all_with(|_| false);
        action.select_region("location", 0, (24.0, 24.0), (1.9, 2.1));
        let action = doc.add_action(action);
        doc.object_mut(cube).unwrap().action = Some(action);
        doc.selection.clear();
        doc.selection.add(cube);
        doc.playback.frame = 30.0;
        let Some(shot) = composite_with(&doc, locale, false, |state| {
            // Layout's timeline, made a graph editor: the layout edits are
            // deferred to the frame's end, so a workspace switch here would
            // leave nothing to change yet.
            let timelines: Vec<_> = state
                .screen()
                .areas()
                .into_iter()
                .filter(|(_, k)| *k == kreate_ui::EditorId::TIMELINE)
                .map(|(a, _)| a)
                .collect();
            assert!(!timelines.is_empty(), "Layout has a timeline");
            for area in timelines {
                state.screen_mut().set_editor(area, kreate_ui::EditorId::GRAPH);
            }
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(label);
    }
}

/// The properties area as a Drivers editor: a working driver with its
/// variable, and a broken one saying why.
#[test]
fn shot_drivers() {
    use kreate_scene::driver::{Driver, DriverVariable};
    for (label, locale) in [("drivers", "en"), ("drivers-ko", "ko-KR")] {
        let mut doc = doc_with_cube();
        let cube = doc.objects().next().map(|(id, _)| id).expect("a cube");
        let handle = doc.add_object(kreate_scene::document::Object::new(
            "Handle",
            kreate_scene::document::ObjectData::Empty,
        ));
        doc.object_mut(handle).unwrap().transform.translation.x = 0.3;
        let mut turn = Driver::new("rotation_euler", 2, "h * 2");
        turn.variables.push(DriverVariable {
            name: "h".into(),
            target: doc.object_uid(handle).unwrap(),
            property: "location".into(),
            index: 0,
        });
        let broken = Driver::new("location", 2, "sin(frame / 10) * height");
        doc.object_mut(cube).unwrap().drivers = vec![turn, broken];
        doc.drive();
        doc.selection.clear();
        doc.selection.add(cube);
        let Some(shot) = composite_with(&doc, locale, false, |state| {
            let timelines: Vec<_> = state
                .screen()
                .areas()
                .into_iter()
                .filter(|(_, k)| *k == kreate_ui::EditorId::PROPERTIES)
                .map(|(a, _)| a)
                .collect();
            for area in timelines {
                state.screen_mut().set_editor(area, kreate_ui::EditorId::DRIVERS);
            }
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(label);
    }
}

/// An arm of three bones beside the cube, bent at the elbow and the wrist.
#[test]
fn shot_armature() {
    use kreate_scene::armature::{Armature, Bone};
    let mut doc = doc_with_cube();
    let mut arm = Armature::new("Arm");
    arm.bones[0].name = "Upper".into();
    arm.bones[0].tail = kreate_math::DVec3::new(0.0, 0.0, 1.2);
    arm.add_bone(Bone {
        name: "Lower".into(),
        parent: Some(0),
        head: kreate_math::DVec3::new(0.0, 0.0, 1.2),
        tail: kreate_math::DVec3::new(0.0, 0.0, 2.2),
        roll: 0.0,
        ..Default::default()
    });
    arm.add_bone(Bone {
        name: "Hand".into(),
        parent: Some(1),
        head: kreate_math::DVec3::new(0.0, 0.0, 2.2),
        tail: kreate_math::DVec3::new(0.0, 0.0, 2.6),
        roll: 0.0,
        ..Default::default()
    });
    let id = doc.add_armature(arm);
    let mut object = kreate_scene::document::Object::new("Rig", kreate_scene::document::ObjectData::Armature(id));
    object.transform.translation = kreate_math::DVec3::new(0.6, -2.0, 0.0);
    object.pose.entry("Lower").transform.rotation = kreate_math::DQuat::from_rotation_x(-0.9);
    object.pose.entry("Hand").transform.rotation = kreate_math::DQuat::from_rotation_x(-0.6);
    let rig = doc.add_object(object);
    doc.selection.clear();
    doc.selection.add(rig);
    let Some(shot) = composite_with(&doc, "en", false, |_| {}) else {
        eprintln!("no GPU; skipping");
        return;
    };
    shot.dump("armature");
}

/// The Animation workspace with an armature's bones keyed: a row per bone
/// channel under the rig.
#[test]
fn shot_pose_keys() {
    use kreate_scene::animation::{bone_path, Action, Interpolation};
    for (label, locale) in [("pose-keys", "en"), ("pose-keys-ko", "ko-KR")] {
        let mut doc = doc_with_cube();
        let id = doc.add_armature(kreate_scene::armature::Armature::new("Arm"));
        let rig = doc.add_object(kreate_scene::document::Object::new("Rig", kreate_scene::document::ObjectData::Armature(id)));
        let mut action = Action::new("RigAction");
        for frame in [1.0, 12.0, 24.0] {
            for i in 0..4u8 {
                action.curve_mut(&bone_path("Bone", "rotation_quaternion"), i).insert(frame, if i == 0 { 1.0 } else { 0.0 }, Interpolation::Bezier);
            }
        }
        for i in 0..3u8 {
            action.curve_mut(&bone_path("Bone", "location"), i).insert(12.0, 0.0, Interpolation::Bezier);
        }
        let action = doc.add_action(action);
        doc.object_mut(rig).unwrap().action = Some(action);
        doc.selection.clear();
        doc.selection.add(rig);
        let Some(shot) = composite_with(&doc, locale, false, |state| {
            state.switch_workspace(4);
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(label);
    }
}

/// Pose mode on a three-bone arm: the Bone tab, the sidebar on the active
/// bone, the Pose menu and the corner naming the bone.
#[test]
fn shot_pose_mode() {
    use kreate_scene::armature::{Armature, Bone};
    for (label, locale, mode) in [
        ("pose-mode", "en", "pose"),
        ("pose-mode-ko", "ko-KR", "pose"),
        ("armature-edit", "en", "edit_armature"),
    ] {
        let mut doc = doc_with_cube();
        let mut arm = Armature::new("Arm");
        arm.bones[0].name = "Upper".into();
        arm.bones[0].tail = kreate_math::DVec3::new(0.0, 0.0, 1.2);
        arm.bones[0].select_tail = true;
        arm.add_bone(Bone {
            name: "Forearm".into(),
            parent: Some(0),
            head: kreate_math::DVec3::new(0.0, 0.0, 1.2),
            tail: kreate_math::DVec3::new(0.0, 0.0, 2.2),
            connected: true,
            select_tail: true,
            ..Default::default()
        });
        arm.in_front = true;
        let id = doc.add_armature(arm);
        let mut object = kreate_scene::document::Object::new("Rig", kreate_scene::document::ObjectData::Armature(id));
        object.transform.translation = kreate_math::DVec3::new(0.6, -2.0, 0.0);
        let lower = object.pose.entry("Forearm");
        lower.transform.rotation = kreate_math::DQuat::from_rotation_x(-0.9);
        lower.selected = true;
        object.pose.active = Some("Forearm".into());
        let rig = doc.add_object(object);
        doc.selection.clear();
        doc.selection.add(rig);
        let Some(shot) = composite_in_mode(&doc, locale, kreate_ops::Mode::from_tag(mode).unwrap(), |state| {
            let areas: Vec<_> = state.screen().areas().iter().map(|(a, _)| *a).collect();
            for area in areas {
                state.set_properties_tab(area, kreate::shell::PropertiesTab::Bone);
            }
            state.toggle_sidebar();
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(label);
    }
}

/// The Animation workspace on a keyed rig: the dope sheet with its folds and
/// the timeline under it, the outliner and properties beside; then the dope
/// sheet turned into the graph editor, with its channel list.
#[test]
fn shot_animation_workspace() {
    use kreate_scene::animation::{bone_path, Action, Interpolation};
    for (label, graph) in [("animation-workspace", false), ("graph-channels", true)] {
        let mut doc = doc_with_cube();
        let cube = doc.objects().next().map(|(id, _)| id).expect("a cube");
        let mut cube_action = Action::new("CubeAction");
        for (frame, x) in [(1.0, 0.0), (24.0, 2.0), (48.0, 0.5)] {
            for i in 0..3u8 {
                cube_action.curve_mut("location", i).insert(frame, if i == 0 { x } else { 0.0 }, Interpolation::Bezier);
            }
        }
        let cube_action = doc.add_action(cube_action);
        doc.object_mut(cube).unwrap().action = Some(cube_action);
        let id = doc.add_armature(kreate_scene::armature::Armature::new("Arm"));
        let rig = doc.add_object(kreate_scene::document::Object::new("Rig", kreate_scene::document::ObjectData::Armature(id)));
        let mut action = Action::new("RigAction");
        for (frame, w) in [(1.0, 1.0), (12.0, 0.8), (36.0, 1.0)] {
            let q = [w, (1.0f64 - w * w).max(0.0).sqrt(), 0.0, 0.0];
            for (i, v) in q.into_iter().enumerate() {
                action.curve_mut(&bone_path("Bone", "rotation_quaternion"), i as u8).insert(frame, v, Interpolation::Bezier);
            }
        }
        let action = doc.add_action(action);
        doc.object_mut(rig).unwrap().action = Some(action);
        doc.selection.clear();
        doc.selection.add(cube);
        doc.selection.add(rig);
        doc.playback.frame = 12.0;
        let Some(shot) = composite_with(&doc, "ko-KR", false, |state| {
            state.switch_workspace(4);
            let _ = graph;
        }) else {
            eprintln!("no GPU; skipping");
            return;
        };
        shot.dump(label);
        if !graph {
            continue;
        }
        // The graph editor in the Layout workspace's timeline area, where the
        // switch is not deferred.
        let Some(shot) = composite_with(&doc, "ko-KR", false, |state| {
            let timelines: Vec<_> = state
                .screen()
                .areas()
                .into_iter()
                .filter(|(_, k)| *k == kreate_ui::EditorId::TIMELINE)
                .map(|(a, _)| a)
                .collect();
            for area in timelines {
                state.screen_mut().set_editor(area, kreate_ui::EditorId::GRAPH);
            }
        }) else {
            return;
        };
        shot.dump(label);
    }
}

/// A capsule bound to a two-bone arm with automatic weights, the lower bone
/// bent: the mesh follows the bones.
#[test]
fn shot_skinned() {
    use kreate_scene::armature::{Armature, Bone};
    let mut doc = Document::new();
    let mut body = kreate_mesh::primitives::uv_sphere(0.3, 16, 24);
    for p in body.positions_mut() {
        p.z = p.z * 3.7 + 1.1;
    }
    let mut arm = Armature::new("Arm");
    arm.bones[0].name = "Upper".into();
    arm.bones[0].tail = kreate_math::DVec3::new(0.0, 0.0, 1.1);
    arm.add_bone(Bone {
        name: "Lower".into(),
        parent: Some(0),
        head: kreate_math::DVec3::new(0.0, 0.0, 1.1),
        tail: kreate_math::DVec3::new(0.0, 0.0, 2.2),
        connected: true,
        ..Default::default()
    });
    arm.in_front = true;
    for (name, w) in kreate_scene::skin::automatic_weights(&body, &arm, kreate_math::DAffine3::IDENTITY) {
        body.attributes_mut().insert(&name, kreate_mesh::attribute::Domain::Point, kreate_mesh::attribute::AttrData::F32(std::sync::Arc::new(w)));
    }
    let arm = doc.add_armature(arm);
    let mut rig = kreate_scene::document::Object::new("Rig", kreate_scene::document::ObjectData::Armature(arm));
    rig.pose.entry("Lower").transform.rotation = kreate_math::DQuat::from_rotation_x(-1.1);
    let rig = doc.add_object(rig);
    let body = doc.add_mesh(body);
    let mut object = kreate_scene::document::Object::new("Body", kreate_scene::document::ObjectData::Mesh(body));
    object.skin = doc.object_uid(rig);
    let body = doc.add_object(object);
    doc.set_parent(body, Some(rig)).unwrap();
    doc.selection.add(rig);
    let Some(shot) = composite_with(&doc, "ko-KR", false, |_| {}) else {
        eprintln!("no GPU; skipping");
        return;
    };
    shot.dump("skinned");

    // The same body in weight paint mode, the lower bone's group shown.
    doc.selection.clear();
    doc.selection.add(body);
    doc.object_mut(body).unwrap().active_group = Some("Lower".into());
    let Some(shot) = composite_in_mode(&doc, "ko-KR", kreate_ops::Mode::from_tag("weight_paint").unwrap(), |state| {
        let areas: Vec<_> = state.screen().areas().iter().map(|(a, _)| *a).collect();
        for area in areas {
            state.set_properties_tab(area, kreate::shell::PropertiesTab::Data);
        }
    }) else {
        return;
    };
    shot.dump("weight-paint");
}

/// A picture of the Material tab with keyed values, the diamonds lit.
#[test]
fn shot_keyed_material() {
    use kreate_scene::animation::{Action, Interpolation};
    let mut doc = doc_with_cube();
    let mesh = doc.meshes().next().map(|(id, _)| id).expect("a mesh");
    let mut action = Action::for_kind("RedAction", kreate_scene::DatablockKind::Material);
    for i in 0..3u8 {
        action.curve_mut("base_color", i).insert(1.0, 0.5, Interpolation::Bezier);
    }
    action.curve_mut("roughness", 0).insert(30.0, 0.2, Interpolation::Bezier);
    let action = doc.add_action(action);
    let material = doc.add_material(kreate_scene::Material {
        name: "Red".into(),
        action: Some(action),
        ..Default::default()
    });
    doc.set_mesh_slot(mesh, 0, Some(material));
    let Some(shot) = composite_with(&doc, "en", false, |state| {
        let areas: Vec<_> = state.screen().areas().iter().map(|(a, _)| *a).collect();
        for area in areas {
            state.set_properties_tab(area, kreate::shell::PropertiesTab::Material);
        }
    }) else {
        eprintln!("no GPU; skipping");
        return;
    };
    shot.dump("material-keyed");
}
