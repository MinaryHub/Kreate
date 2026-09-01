//! The Kreate application shell.
//!
//! Drives winit directly rather than going through a framework, because the
//! things a framework abstracts away — present mode, frame latency, adapter
//! selection, and the redraw policy — are exactly the things that decide whether
//! the viewport feels responsive.
//!
//! Nothing in this file mutates the document. Keys and panels alike dispatch
//! operators through the registry. That is the spine rule, and holding to it in
//! the shell is what keeps undo, scripting and macro recording working without
//! any cooperation from the UI layer.

mod shell;

use std::sync::Arc;
use std::time::Instant;

use glam::{DAffine3, DVec3, Vec4};
use kreate_io::obj::{parse_obj, ObjOptions};
use kreate_ops::{Context as OpContext, History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_render::{Camera, GpuMesh, GpuOptions, RenderMeshCpu, ShadingMode, Viewport};
use kreate_scene::document::{MeshId, Object, ObjectData};
use kreate_scene::Document;
use kreate_ui::Command;
use shell::ShellState;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Vendor hints telling a hybrid-GPU laptop to hand us the discrete adapter.
///
/// These must be exported from the *executable*, not a library, or the driver
/// never sees them. `PowerPreference::HighPerformance` alone is not reliably
/// enough, and quietly landing on the integrated GPU is a leading cause of
/// "your application is slow" reports.
#[no_mangle]
pub static NvOptimusEnablement: u32 = 1;
#[no_mangle]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;

const HELP: &str = "\
Kreate

  Navigate   MMB orbit  Shift+MMB pan  wheel zoom  F frame all
  View       Z shading  O orthographic  G grid
  Add        A cube  Shift+A next primitive shape
  Edit       D duplicate (linked)  X delete  Ctrl+A select all
  Transform  T move +Z  R rotate 15 deg  S scale 1.2
  Adjust     [ / ]  re-run the last operation with a smaller/larger value
  History    Ctrl+Z undo  Ctrl+Y redo  P print history as a script
  Quit       Esc  (F1 for this list)

  Shortcuts stand down while a text field or an IME has the keyboard.";

/// Why the next frame needs drawing.
///
/// Redraw is explicit and auditable rather than "just render every frame". An
/// idle Kreate window must use no CPU and no GPU: users leave a modelling
/// application open all day, and one that drains a battery while doing nothing
/// gets closed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RedrawReason(u32);

impl RedrawReason {
    const CAMERA_MOVED: u32 = 1 << 0;
    const SCENE_CHANGED: u32 = 1 << 1;
    const RESIZE: u32 = 1 << 2;
    const SETTINGS: u32 = 1 << 3;
    const UI: u32 = 1 << 4;

    fn set(&mut self, bit: u32) {
        self.0 |= bit;
    }
    fn any(self) -> bool {
        self.0 != 0
    }
    fn clear(&mut self) {
        self.0 = 0;
    }
}

/// Input accumulated between frames.
///
/// Cursor *deltas* accumulate, because dropping any of them would lose motion;
/// but everything is applied once per frame rather than once per event. Handling
/// a backlog of stale mouse-moves one per frame is the single most common cause
/// of a cursor that visibly runs ahead of the model.
#[derive(Default)]
struct Input {
    cursor: Option<(f64, f64)>,
    orbit_delta: (f64, f64),
    pan_delta: (f64, f64),
    scroll: f64,
    middle_down: bool,
    shift_down: bool,
    ctrl_down: bool,
    /// When the oldest unprocessed input arrived, for the latency measurement.
    first_event_at: Option<Instant>,
}

impl Input {
    fn note_event(&mut self) {
        self.first_event_at.get_or_insert_with(Instant::now);
    }

    fn has_pending(&self) -> bool {
        self.orbit_delta != (0.0, 0.0) || self.pan_delta != (0.0, 0.0) || self.scroll != 0.0
    }
}

/// Rolling motion-to-photon statistics.
///
/// Latency does not show up in a frame-rate counter: an application can sit at a
/// solid 144 fps and still feel awful. Measuring it from day one is what stops a
/// dozen small regressions from adding up to an unusable tool.
#[derive(Default)]
struct LatencyStats {
    samples: Vec<f32>,
    reported_at: Option<Instant>,
}

impl LatencyStats {
    fn record(&mut self, ms: f32) {
        self.samples.push(ms);
        let now = Instant::now();
        let due = self
            .reported_at
            .is_none_or(|t| now.duration_since(t).as_secs_f32() > 5.0);
        if due && self.samples.len() >= 30 {
            self.samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p = |q: f32| self.samples[((self.samples.len() - 1) as f32 * q) as usize];
            tracing::info!(
                p50 = format!("{:.1}ms", p(0.5)),
                p99 = format!("{:.1}ms", p(0.99)),
                n = self.samples.len(),
                "motion-to-photon"
            );
            self.samples.clear();
            self.reported_at = Some(now);
        }
    }
}

/// The offscreen target the 3D scene renders into.
///
/// A texture rather than a paint callback, so the viewport can render at a
/// different resolution and a different rate than the interface. See `shell`.
struct ViewportTarget {
    /// Only the view is kept: a `TextureView` holds the texture alive through
    /// wgpu's own refcount, and the handle is never needed again. Explicit
    /// `destroy` happens by dropping the view when the target is replaced.
    view: wgpu::TextureView,
    id: egui::TextureId,
    size: (u32, u32),
}

impl ViewportTarget {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    fn new(
        device: &wgpu::Device,
        renderer: &mut egui_wgpu::Renderer,
        width: u32,
        height: u32,
    ) -> Self {
        let size = (width.max(1), height.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport target"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = renderer.register_native_texture(device, &view, wgpu::FilterMode::Linear);
        Self { view, id, size }
    }

    /// Reallocates if the requested size differs enough to matter.
    ///
    /// Sizes are rounded to a 64-pixel step: a panel drag would otherwise
    /// reallocate the texture and re-register it every single frame, which is a
    /// stall for no visual benefit.
    fn resize(
        &mut self,
        device: &wgpu::Device,
        renderer: &mut egui_wgpu::Renderer,
        width: u32,
        height: u32,
    ) -> bool {
        let round = |v: u32| v.max(1).div_ceil(64) * 64;
        let wanted = (round(width), round(height));
        if wanted == self.size {
            return false;
        }
        renderer.free_texture(&self.id);
        *self = Self::new(device, renderer, wanted.0, wanted.1);
        true
    }
}

struct Gfx {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: kreate_render::Gpu,
    viewport: Viewport,
    /// GPU residency, keyed by datablock. Linked duplicates share an entry,
    /// which is the whole point of referencing meshes by id.
    meshes: Vec<(MeshId, GpuMesh)>,
    /// Document version the GPU state was built from.
    ///
    /// A version stamp is the placeholder for real dirty tracking: it rebuilds
    /// everything whenever anything changes. Correct but coarse — and coarse
    /// invalidation is the leading cause of hitches in an application of this
    /// kind. The per-buffer dirty ranges and the GPU transform preview belong
    /// here, and this is the line that will stop being acceptable first.
    built_from_version: u64,

    egui_renderer: egui_wgpu::Renderer,
    target: ViewportTarget,
}

struct App {
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    fonts_warmed: bool,

    camera: Camera,
    input: Input,
    redraw: RedrawReason,
    latency: LatencyStats,

    doc: Document,
    registry: OperatorRegistry,
    history: History,
    op_ctx: OpContext,
    ui: ShellState,
    /// Which primitive Shift+A adds next.
    next_shape: u32,
}

impl App {
    fn new(doc: Document, locale: &str) -> Self {
        let mut registry = OperatorRegistry::new();
        kreate_ops::register_builtin(&mut registry);
        Self {
            window: None,
            gfx: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            fonts_warmed: false,
            camera: Camera::default(),
            input: Input::default(),
            redraw: RedrawReason(RedrawReason::SCENE_CHANGED),
            latency: LatencyStats::default(),
            doc,
            registry,
            history: History::new(),
            op_ctx: OpContext::default(),
            ui: ShellState {
                catalog: shell::load_catalog(locale),
                ..ShellState::default()
            },
            next_shape: 1,
        }
    }

    /// Dispatches an operator. The only way this application changes the document.
    fn run_op(&mut self, op_id: &str, params: PropertyMap) {
        match self
            .history
            .dispatch(&self.registry, &mut self.doc, &self.op_ctx, op_id, &params)
        {
            Ok(kreate_ops::OpResult::Finished) => {
                if let Some(entry) = self.history.last() {
                    println!("{}", entry.to_script_call());
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            Ok(other) => tracing::debug!(?other, op_id, "operator did not finish"),
            Err(e) => println!("  {op_id}: {e}"),
        }
    }

    fn run_op_with(&mut self, op_id: &str, pairs: &[(&'static str, PropertyValue)]) {
        let mut params = PropertyMap::new();
        for (k, v) in pairs {
            params.set(k, v.clone());
        }
        self.run_op(op_id, params);
    }

    /// Applies the commands the panels queued this frame.
    fn drain_ui_commands(&mut self) {
        for command in self.ui.queue.drain() {
            match command {
                Command::RunOperator { op_id, params } => self.run_op(&op_id, params),
                Command::AdjustLast { params } => {
                    match self.history.adjust_last(
                        &self.registry,
                        &mut self.doc,
                        &self.op_ctx,
                        &params,
                    ) {
                        Ok(_) => self.redraw.set(RedrawReason::SCENE_CHANGED),
                        Err(e) => println!("  adjust failed: {e}"),
                    }
                }
                Command::Undo => self.undo(),
                Command::Redo => self.redo(),
                Command::SetSelection { objects, active } => {
                    // Selection from the outliner arrives as opaque keys, which
                    // are resolved against the *current* document: a row from a
                    // frame where an object still existed must not act on
                    // whatever reused its slot.
                    let resolved: Vec<_> = objects
                        .iter()
                        .filter_map(|k| shell::object_from_key(&self.doc, *k))
                        .collect();
                    self.doc.selection.clear();
                    for id in resolved {
                        self.doc.selection.add(id);
                    }
                    if let Some(a) = active.and_then(|k| shell::object_from_key(&self.doc, k)) {
                        self.doc.selection.add(a);
                    }
                    self.redraw.set(RedrawReason::SCENE_CHANGED);
                }
                Command::FrameSelection => self.frame_all(),
            }
        }
    }

    /// Re-runs the last operation with its primary numeric parameter scaled.
    ///
    /// The keyboard route to the adjust panel. It needs no per-operator support:
    /// restore, then run again with edited values — the same path redo takes.
    fn adjust_last(&mut self, factor: f64) {
        let Some(entry) = self.history.last() else {
            println!("  nothing to adjust");
            return;
        };
        let op_id = entry.op_id.clone();
        let Some(info) = self.registry.get(&op_id) else {
            return;
        };

        let mut params = entry.params.clone();
        let mut adjusted = None;
        for desc in info.schema {
            match params.get(desc.name).cloned() {
                Some(PropertyValue::Float(v)) if v != 0.0 => {
                    params.set(desc.name, PropertyValue::Float(v * factor));
                    adjusted = Some(desc.name);
                    break;
                }
                Some(PropertyValue::Vec3(v)) if v.iter().any(|c| *c != 0.0) => {
                    params.set(
                        desc.name,
                        PropertyValue::Vec3([v[0] * factor, v[1] * factor, v[2] * factor]),
                    );
                    adjusted = Some(desc.name);
                    break;
                }
                _ => {}
            }
        }
        let Some(name) = adjusted else {
            println!("  {op_id}: no numeric parameter to adjust");
            return;
        };

        match self
            .history
            .adjust_last(&self.registry, &mut self.doc, &self.op_ctx, &params)
        {
            Ok(true) => {
                let call = self
                    .history
                    .last()
                    .map(|e| e.to_script_call())
                    .unwrap_or_default();
                println!("  adjusted {name} -> {call}");
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            Ok(false) => {}
            Err(e) => println!("  adjust failed: {e}"),
        }
    }

    fn undo(&mut self) {
        if self.history.undo(&mut self.doc) {
            println!("  undo");
            self.redraw.set(RedrawReason::SCENE_CHANGED);
        } else {
            println!("  nothing to undo");
        }
    }

    fn redo(&mut self) {
        match self
            .history
            .redo(&self.registry, &mut self.doc, &self.op_ctx)
        {
            Ok(true) => {
                println!("  redo");
                self.redraw.set(RedrawReason::SCENE_CHANGED);
            }
            Ok(false) => println!("  nothing to redo"),
            Err(e) => println!("  redo failed: {e}"),
        }
    }

    fn frame_all(&mut self) {
        let bounds = self.doc.world_bounds();
        if bounds.is_empty() {
            return;
        }
        let c = bounds.center();
        self.camera.frame_sphere(
            DVec3::new(c.x as f64, c.y as f64, c.z as f64),
            bounds.bounding_radius().max(1e-3) as f64,
        );
        self.redraw.set(RedrawReason::CAMERA_MOVED);
    }

    /// Brings GPU residency into step with the document.
    fn sync_gpu(&mut self) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        if gfx.built_from_version == self.doc.version() {
            return;
        }

        const PALETTE: [[f32; 3]; 4] = [
            [0.72, 0.73, 0.76],
            [0.78, 0.62, 0.42],
            [0.52, 0.68, 0.78],
            [0.68, 0.58, 0.74],
        ];
        gfx.meshes.clear();
        let mut seen: Vec<MeshId> = Vec::new();
        for (_, object) in self.doc.objects() {
            let Some(mesh_id) = object.mesh() else {
                continue;
            };
            if seen.contains(&mesh_id) {
                continue;
            }
            seen.push(mesh_id);
            let Some(mesh) = self.doc.mesh(mesh_id) else {
                continue;
            };
            let cpu = RenderMeshCpu::extract(mesh);
            let c = PALETTE[seen.len() % PALETTE.len()];
            gfx.meshes.push((
                mesh_id,
                GpuMesh::upload(
                    &gfx.gpu.device,
                    &gfx.viewport.object_layout,
                    &cpu,
                    Vec4::new(c[0], c[1], c[2], 1.0),
                ),
            ));
        }
        gfx.built_from_version = self.doc.version();
    }

    fn apply_input(&mut self) {
        let height = self
            .window
            .as_ref()
            .map(|w| w.inner_size().height)
            .unwrap_or(1)
            .max(1) as f64;

        if self.input.orbit_delta != (0.0, 0.0) {
            let (dx, dy) = self.input.orbit_delta;
            let scale = std::f64::consts::TAU / (2.0 * height);
            self.camera.orbit(-dx * scale, -dy * scale);
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
        if self.input.pan_delta != (0.0, 0.0) {
            let (dx, dy) = self.input.pan_delta;
            self.camera.pan(dx / height, dy / height);
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
        if self.input.scroll != 0.0 {
            self.camera.dolly(self.input.scroll);
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
        self.input.orbit_delta = (0.0, 0.0);
        self.input.pan_delta = (0.0, 0.0);
        self.input.scroll = 0.0;
    }

    fn render(&mut self) {
        self.sync_gpu();
        let Some(window) = self.window.clone() else {
            return;
        };
        let Some(state) = self.egui_state.as_mut() else {
            return;
        };

        // --- run the interface, which tells us how big the viewport is -------
        let raw_input = state.take_egui_input(&window);
        let texture_id = self.gfx.as_ref().map(|g| g.target.id);
        let mut viewport_rect = egui::Rect::ZERO;
        let output = {
            let ui_state = &mut self.ui;
            let doc = &self.doc;
            let registry = &self.registry;
            let history = &self.history;
            self.egui_ctx.run_ui(raw_input, |ui| {
                viewport_rect = shell::draw(ui, ui_state, doc, registry, history, texture_id);
            })
        };
        state.handle_platform_output(&window, output.platform_output.clone());

        // The single input gate. Scattered `if !typing` checks are how a rename
        // field ends up firing tool shortcuts again.
        self.ui.egui_wants_keyboard = self.egui_ctx.egui_wants_keyboard_input();
        self.ui.ime_active = output.platform_output.ime.is_some();

        self.drain_ui_commands();
        self.sync_gpu();

        let pixels_per_point = self.egui_ctx.pixels_per_point();
        let Some(gfx) = self.gfx.as_mut() else { return };

        // --- resize the offscreen viewport to the space the panels left ------
        let vp_w = (viewport_rect.width() * pixels_per_point) as u32;
        let vp_h = (viewport_rect.height() * pixels_per_point) as u32;
        if gfx
            .target
            .resize(&gfx.gpu.device, &mut gfx.egui_renderer, vp_w, vp_h)
        {
            gfx.viewport
                .resize(&gfx.gpu.device, gfx.target.size.0, gfx.target.size.1);
            self.redraw.set(RedrawReason::RESIZE);
        }

        use wgpu::CurrentSurfaceTexture as Acquired;
        let frame = match gfx.surface.get_current_texture() {
            Acquired::Success(f) => f,
            Acquired::Suboptimal(f) => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.config);
                f
            }
            Acquired::Outdated | Acquired::Lost => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.config);
                self.redraw.set(RedrawReason::RESIZE);
                return;
            }
            Acquired::Occluded => return,
            Acquired::Timeout => {
                tracing::warn!("timed out acquiring a surface texture");
                return;
            }
            Acquired::Validation => {
                tracing::error!("validation error acquiring a surface texture");
                return;
            }
        };
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // One draw entry per *object*, sharing GPU geometry by mesh id — so a
        // thousand linked duplicates cost a thousand transforms.
        let mut draws: Vec<(&GpuMesh, DAffine3)> = Vec::new();
        for (id, object) in self.doc.objects() {
            if !object.visible {
                continue;
            }
            let Some(mesh_id) = object.mesh() else {
                continue;
            };
            if let Some((_, gpu_mesh)) = gfx.meshes.iter().find(|(m, _)| *m == mesh_id) {
                draws.push((gpu_mesh, self.doc.world_transform(id)));
            }
        }

        let mut encoder = gfx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // 3D into the offscreen target...
        gfx.viewport.prepare(&gfx.gpu.queue, &self.camera, &draws);
        gfx.viewport.render(&mut encoder, &gfx.target.view, &draws);

        // ...then the interface over the surface, with the target as an image.
        let paint_jobs = self
            .egui_ctx
            .tessellate(output.shapes, output.pixels_per_point);
        for (id, delta) in &output.textures_delta.set {
            gfx.egui_renderer
                .update_texture(&gfx.gpu.device, &gfx.gpu.queue, *id, delta);
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gfx.config.width, gfx.config.height],
            pixels_per_point,
        };
        let extra = gfx.egui_renderer.update_buffers(
            &gfx.gpu.device,
            &gfx.gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );
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
            gfx.egui_renderer
                .render(&mut pass.forget_lifetime(), &paint_jobs, &screen);
        }

        gfx.gpu
            .queue
            .submit(extra.into_iter().chain(Some(encoder.finish())));
        frame.present();

        for id in &output.textures_delta.free {
            gfx.egui_renderer.free_texture(id);
        }

        // egui builds its font set lazily inside the first pass, so warming the
        // atlas has to wait until now rather than happening beside `install`.
        if !self.fonts_warmed {
            kreate_ui::prewarm_fonts(&self.egui_ctx);
            self.fonts_warmed = true;
        }

        if let Some(t) = self.input.first_event_at.take() {
            self.latency.record(t.elapsed().as_secs_f32() * 1000.0);
        }
    }

    fn handle_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        // Escape and F1 work regardless, so there is always a way out.
        match code {
            KeyCode::Escape => {
                event_loop.exit();
                return;
            }
            KeyCode::F1 => {
                println!("{HELP}");
                return;
            }
            _ => {}
        }
        // Everything else stands down while a text field or an IME has the
        // keyboard: with a Korean IME active, `G` and `R` would otherwise reach
        // both the composer and the keymap.
        if !self.ui.keymap_enabled() {
            return;
        }

        let shift = self.input.shift_down;
        let ctrl = self.input.ctrl_down;

        match code {
            KeyCode::KeyZ if ctrl => self.undo(),
            KeyCode::KeyY if ctrl => self.redo(),
            KeyCode::KeyP => {
                let script = self.history.to_script();
                println!("--- history as a script ---");
                println!(
                    "{}",
                    if script.is_empty() {
                        "(empty)"
                    } else {
                        &script
                    }
                );
                println!("---");
            }

            KeyCode::KeyA if ctrl => {
                self.run_op_with("object.select_all", &[("action", PropertyValue::Enum(2))])
            }
            KeyCode::KeyA if shift => {
                let shape = self.next_shape;
                self.next_shape =
                    (self.next_shape + 1) % kreate_ops::ops::object::PRIMITIVES.len() as u32;
                let n = self.doc.object_count() as f64;
                self.run_op_with(
                    "object.add_primitive",
                    &[
                        ("shape", PropertyValue::Enum(shape)),
                        ("location", PropertyValue::Vec3([n * 3.0, 0.0, 0.0])),
                    ],
                );
            }
            KeyCode::KeyA => {
                self.run_op_with("object.add_primitive", &[("shape", PropertyValue::Enum(1))])
            }

            KeyCode::KeyD => {
                self.run_op_with("object.duplicate", &[("linked", PropertyValue::Bool(true))]);
                self.run_op_with(
                    "object.translate",
                    &[("translation", PropertyValue::Vec3([0.0, 3.0, 0.0]))],
                );
            }
            KeyCode::KeyX => self.run_op_with("object.delete", &[]),

            KeyCode::KeyT => self.run_op_with(
                "object.translate",
                &[("translation", PropertyValue::Vec3([0.0, 0.0, 1.0]))],
            ),
            KeyCode::KeyR => self.run_op_with(
                "object.rotate",
                &[
                    ("angle", PropertyValue::Float(15f64.to_radians())),
                    ("axis", PropertyValue::Enum(2)),
                ],
            ),
            KeyCode::KeyS => self.run_op_with(
                "object.scale",
                &[("scale", PropertyValue::Vec3([1.2, 1.2, 1.2]))],
            ),

            KeyCode::BracketLeft => self.adjust_last(0.5),
            KeyCode::BracketRight => self.adjust_last(2.0),

            KeyCode::KeyF => self.frame_all(),
            KeyCode::KeyG => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.show_grid = !gfx.viewport.show_grid;
                    self.redraw.set(RedrawReason::SETTINGS);
                }
            }
            KeyCode::KeyZ => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.shading = match gfx.viewport.shading {
                        ShadingMode::Studio => ShadingMode::Flat,
                        ShadingMode::Flat => ShadingMode::Studio,
                    };
                    self.redraw.set(RedrawReason::SETTINGS);
                }
            }
            KeyCode::KeyO => {
                self.camera.projection = match self.camera.projection {
                    kreate_render::Projection::Perspective => {
                        kreate_render::Projection::Orthographic
                    }
                    kreate_render::Projection::Orthographic => {
                        kreate_render::Projection::Perspective
                    }
                };
                self.redraw.set(RedrawReason::CAMERA_MOVED);
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Kreate")
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let opts = GpuOptions::default();
        let instance = kreate_render::Gpu::create_instance(&opts);
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, Some(&surface)))
            .expect("initialise GPU");
        println!("GPU: {}\n", gpu.info.describe());

        let size = window.inner_size();
        let config =
            kreate_render::gpu::surface_config(&surface, &gpu.adapter, size.width, size.height);
        surface.configure(&gpu.device, &config);

        kreate_ui::install_fonts(&self.egui_ctx);
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        ));

        let mut egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            config.format,
            egui_wgpu::RendererOptions::default(),
        );
        let target = ViewportTarget::new(&gpu.device, &mut egui_renderer, size.width, size.height);
        let viewport = Viewport::new(
            &gpu.device,
            ViewportTarget::FORMAT,
            target.size.0,
            target.size.1,
        );

        self.gfx = Some(Gfx {
            surface,
            config,
            gpu,
            viewport,
            meshes: Vec::new(),
            built_from_version: u64::MAX,
            egui_renderer,
            target,
        });
        self.window = Some(window);
        self.frame_all();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // egui gets first refusal. Anything it consumes must not also reach the
        // viewport, or a click on a panel would orbit the camera behind it.
        let consumed = match (self.egui_state.as_mut(), self.window.as_ref()) {
            (Some(state), Some(window)) => {
                let response = state.on_window_event(window, &event);
                if response.repaint {
                    self.redraw.set(RedrawReason::UI);
                }
                response.consumed
            }
            _ => false,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.config.width = size.width.max(1);
                    gfx.config.height = size.height.max(1);
                    gfx.surface.configure(&gfx.gpu.device, &gfx.config);
                }
                self.redraw.set(RedrawReason::RESIZE);
            }

            WindowEvent::ScaleFactorChanged { .. } => {
                // A DPI change discards the whole font atlas; re-warm it rather
                // than re-rasterising a Korean interface lazily mid-drag.
                self.fonts_warmed = false;
                self.redraw.set(RedrawReason::RESIZE);
            }

            WindowEvent::MouseInput { state, button, .. } if !consumed => {
                if button == MouseButton::Middle {
                    self.input.middle_down = state == ElementState::Pressed;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let p = (position.x, position.y);
                if let Some(prev) = self.input.cursor {
                    let delta = (p.0 - prev.0, p.1 - prev.1);
                    if self.input.middle_down {
                        self.input.note_event();
                        if self.input.shift_down {
                            self.input.pan_delta.0 += delta.0;
                            self.input.pan_delta.1 += delta.1;
                        } else {
                            self.input.orbit_delta.0 += delta.0;
                            self.input.orbit_delta.1 += delta.1;
                        }
                    }
                }
                self.input.cursor = Some(p);
            }

            WindowEvent::MouseWheel { delta, .. } if !consumed => {
                self.input.note_event();
                self.input.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 60.0,
                };
            }

            WindowEvent::ModifiersChanged(m) => {
                self.input.shift_down = m.state().shift_key();
                self.input.ctrl_down = m.state().control_key();
            }

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input.note_event();
                    self.handle_key(code, event_loop);
                }
            }

            WindowEvent::RedrawRequested => {
                self.apply_input();
                if self.redraw.any() {
                    self.render();
                    self.redraw.clear();
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // While the user is dragging, stay on Poll so motion is continuous;
        // otherwise Wait, so an idle window costs nothing.
        if self.input.has_pending() || self.redraw.any() || !self.ui.queue.is_empty() {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::Poll);
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

/// Builds the starting document, from a file if one was named.
fn load_document(path: Option<&str>) -> Document {
    let mut doc = Document::new();
    match path {
        Some(path) => {
            let text =
                std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
            let scene = parse_obj(&text, ObjOptions::default())
                .unwrap_or_else(|e| panic!("cannot parse {path}: {e}"));
            let (meshes, source) = scene.into_mesh_data();
            for note in &source.notes {
                tracing::info!(?note, "import note");
            }
            for (name, mesh) in meshes {
                doc.add_mesh_object(name, mesh);
            }
        }
        None => {
            let mesh = doc.add_mesh(kreate_mesh::primitives::cube(2.0));
            let id = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
            doc.selection.add(id);
        }
    }
    doc
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kreate=info,kreate_render=info".into()),
        )
        .init();

    // `--lang ko` / `--lang en`, defaulting to Korean because that is who this
    // is being built for. A real preference file replaces this.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut locale = "ko-KR".to_string();
    let mut path = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--lang" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    locale = v.clone();
                }
            }
            other => path = Some(other.to_string()),
        }
        i += 1;
    }

    println!("{HELP}\n");
    let doc = load_document(path.as_deref());
    let event_loop = EventLoop::new().expect("create event loop");
    let mut app = App::new(doc, &locale);
    event_loop.run_app(&mut app).expect("run");
}
