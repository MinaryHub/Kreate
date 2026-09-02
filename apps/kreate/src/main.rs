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

// The shell lives in this package's library target so tests can drive it.
use kreate::{gpu_sync, shell};

use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, Vec4};
use kreate::shell::ShellState;
use kreate_input::{
    keymap::{Action, EventKind, Keymap, Modifiers as KeyMods, MouseButton as KeyMouse},
    presets, Context as KeyContext, Input as KeyInput, Preset,
};
use kreate_io::obj::{parse_obj, ObjOptions};
use kreate_mesh::MeshData;
use kreate_ops::{
    Context as OpContext, History, Mode as OpMode, OperatorRegistry, PropertyMap, PropertyValue,
};
use kreate_render::{Camera, Draw, GpuMesh, GpuOptions, RenderMeshCpu, ShadingMode, Viewport};
use kreate_scene::document::{Object, ObjectData, ObjectId};
use kreate_scene::Document;
use kreate_ui::Command;
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
  View       Z shading  O orthographic  G grid  W wireframe
             Y auto-depth orbit (on by default)
  Add        A cube  Shift+A next primitive shape
  Edit       D duplicate (linked)  X delete  Ctrl+A select all
  Edit mode  Tab enter/leave  E extrude  X delete  Ctrl+X dissolve
             S subdivide  I inset  B bridge borders  Ctrl+R loop cut
             [ / ] after a loop cut slides it
             M merge selection  Shift+M weld by distance
             N shade smooth  Ctrl+N recalculate  Shift+N flip
             K mark sharp  Alt+K clear  Shift+K auto-smooth 30 deg
             F fill hole  Shift+F grid fill
             Y split off faces  V rip along edges  Ctrl+B bevel
             1/2/3 vert/edge/face select  L loop  Shift+L ring
             Ctrl+A all elements  Alt+A none  Ctrl+I invert
  Transform  T move +Z  R rotate 15 deg  S scale 1.2
  Adjust     [ / ]  re-run the last operation with a smaller/larger value
  History    Ctrl+Z undo  Ctrl+Y redo  P print history as a script
  Keymap     F1 prints the resolved keymap, preset and all
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
    /// What is under the cursor changed, so the hover highlight is stale.
    ///
    /// Its own reason rather than folding into `UI`: a hover is the most
    /// frequent thing that asks for a frame, and having it named is what lets
    /// "why is this window redrawing" be answered rather than guessed at.
    const HOVER_CHANGED: u32 = 1 << 5;

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
    shift_down: bool,
    ctrl_down: bool,
    alt_down: bool,
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

/// A rubber-band selection being dragged.
///
/// A press does not commit anything: a click and the start of a box look
/// identical until the pointer moves. Deciding on release is what lets one
/// gesture be both without either a spurious history step or a selection that
/// flickers to something else mid-drag.
struct DragSelect {
    /// Where the drag began, in the viewport's own pixels.
    origin: (f32, f32),
    current: (f32, f32),
    /// Whether the pointer has moved far enough to be a drag rather than a
    /// click. Latching rather than recomputed, so a drag that returns to its
    /// origin stays a drag — releasing there should select nothing, not
    /// everything under the pointer.
    moved: bool,
    /// Add to the selection rather than replacing it.
    extend: bool,
    /// Remove from the selection instead.
    deselect: bool,
}

/// How far the pointer has to move for a press to become a box.
///
/// Small enough that a deliberate drag is recognised immediately, large enough
/// that the hand tremor in a click does not turn it into a one-pixel box that
/// selects nothing.
const DRAG_THRESHOLD_PX: f32 = 4.0;

impl DragSelect {
    fn shape(&self) -> kreate_render::SelectShape {
        kreate_render::SelectShape::Box {
            from: self.origin,
            to: self.current,
        }
    }
}

struct Gfx {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: kreate_render::Gpu,
    viewport: Viewport,
    /// GPU residency, keyed by the identity of the evaluated mesh. Instances
    /// share an entry — linked duplicates, and objects that happen to share a
    /// modifier stack — which is the whole point of referencing meshes by id.
    ///
    /// Keyed on the evaluated handle rather than on a `MeshId` because two
    /// objects can share a datablock and have different stacks, and then an
    /// id-keyed cache hands one of them the other's geometry.
    ///
    /// Each entry keeps the `Arc` it was built from. That is not bookkeeping: it
    /// is what makes `Arc::ptr_eq` a valid change test, because `make_mut` only
    /// copies when the handle is shared. It also makes the renderer's view of the
    /// document genuinely immutable rather than merely intended to be.
    meshes: Vec<(Arc<MeshData>, GpuMesh)>,
    /// The id buffer and the ring of readback buffers behind it.
    pick_target: kreate_render::PickTarget,
    /// The compute reduction behind box, circle and lasso selection.
    region: kreate_render::RegionSelect,

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
    /// Which element type edit-mode selection acts on. 0 vert, 1 edge, 2 face.
    select_mode: u32,
    /// Lazily built BVHs, for auto-depth and — later — snapping and knife.
    pick: kreate::pick::PickCache,
    /// Evaluated geometry, kept between frames.
    ///
    /// Deliberately not inside the `Document`: the document is cloned for every
    /// undo snapshot and restored from one, so a cache living there would travel
    /// backwards in time along with the geometry.
    modifiers: kreate_scene::ModifierCache,
    /// This frame's evaluated scene: what the renderer draws and what the picker
    /// queries. Rebuilt at the top of a frame, and the only thing either of them
    /// is allowed to look at — the document holds the cage the user edits.
    evaluated: Vec<(ObjectId, Arc<MeshData>)>,
    /// Whether orbiting re-centres on whatever is under the cursor.
    auto_depth: bool,
    /// What the last id readback said is near the cursor.
    ///
    /// Held so that a click consumes data already in flight rather than starting
    /// a readback and waiting two frames for it — which is the difference
    /// between a selection that changes on the click and one that lags behind it
    /// by thirty milliseconds, however high the frame rate is.
    hover: Option<kreate_render::PickHit>,
    /// Whether occluded elements are selectable.
    xray: bool,
    /// The rubber-band drag in progress, if any.
    drag_select: Option<DragSelect>,
    /// The single table every input event is resolved against.
    keymap: Keymap,
    preset: Preset,
    /// The user's changes, as they will be written back. Held separately from
    /// the resolved keymap so a save writes only the diff.
    key_overrides: std::collections::BTreeMap<kreate_input::BindingId, kreate_input::Override>,
    /// Which mouse buttons are held, for resolving drag bindings.
    buttons: [bool; 3],
    /// Where the camera is going. `camera` follows it.
    motion: kreate_render::CameraMotion,
    /// When the last frame was, for frame-rate-independent easing.
    last_frame: Option<Instant>,
    /// The viewport rectangle in physical pixels, for turning a cursor position
    /// into a ray. Set by the previous frame's layout.
    viewport_px: Option<(f32, f32, f32, f32)>,
    ui: ShellState,
    /// Size the viewport wanted last frame, in physical pixels.
    ///
    /// The resize is driven from the *previous* frame rather than the current
    /// one, because reallocating the offscreen target frees the egui texture id
    /// that this frame's paint jobs already reference. Acting a frame late is
    /// invisible; freeing a texture the renderer is about to use is a validation
    /// error and a dead device.
    wanted_viewport: (u32, u32),
    /// Which primitive Shift+A adds next.
    next_shape: u32,
}

impl App {
    fn new(doc: Document, locale: &str) -> Self {
        let mut registry = OperatorRegistry::new();
        kreate_ops::register_builtin(&mut registry);
        let mut app = Self::empty(doc, locale, registry);
        app.load_keymap();
        app
    }

    fn empty(doc: Document, locale: &str, registry: OperatorRegistry) -> Self {
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
            select_mode: 2,
            pick: kreate::pick::PickCache::new(),
            modifiers: kreate_scene::ModifierCache::new(),
            evaluated: Vec::new(),
            auto_depth: true,
            hover: None,
            xray: false,
            drag_select: None,
            motion: kreate_render::CameraMotion::new(kreate_render::Camera::default()),
            last_frame: None,
            keymap: Keymap::default(),
            preset: Preset::default(),
            key_overrides: std::collections::BTreeMap::new(),
            buttons: [false; 3],
            viewport_px: None,
            ui: ShellState {
                catalog: shell::load_catalog(locale),
                ..ShellState::default()
            },
            wanted_viewport: (0, 0),
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

    /// Loads the keymap file and applies it over the chosen preset.
    ///
    /// Anything wrong with the file is logged and the rest still loads. A broken
    /// preferences file must cost a setting, not a session.
    fn load_keymap(&mut self) {
        let mut file = kreate_input::KeyFile::default();
        if let Some(path) = kreate::prefs::keymap_path() {
            let (loaded, problems) = kreate::prefs::load_keymap(&path);
            for problem in problems {
                tracing::warn!("{problem}");
                println!("keymap: {problem}");
            }
            file = loaded;
        }
        self.preset = file.preset;
        self.key_overrides = file.overrides;
        self.rebuild_keymap();
    }

    /// Rebuilds the resolved keymap from the preset and the user's changes.
    ///
    /// Layered every time rather than flattened once: that is what lets a later
    /// preset improvement reach a user who has remapped something.
    fn rebuild_keymap(&mut self) {
        self.keymap = presets::load(self.preset);
        for (id, change) in &self.key_overrides {
            self.keymap.set_override(id, change.clone());
        }
        for (shadowed, by) in self.keymap.conflicts() {
            println!(
                "keymap: `{}` can never fire; `{}` takes it first",
                shadowed.0, by.0
            );
        }
    }

    /// Applies whatever the remapping panel asked for.
    fn apply_keymap_edits(&mut self) {
        let edits = std::mem::take(&mut self.ui.keymap_edits);
        if edits.is_empty() {
            return;
        }
        for edit in edits {
            match edit {
                kreate_ui::KeymapEdit::UsePreset(preset) => {
                    // The user's diff is *kept*: it is a diff against ids, and
                    // the ids are the same in every preset. Discarding it here
                    // would make trying a preset destructive.
                    self.preset = preset;
                }
                kreate_ui::KeymapEdit::Set(id, change) => {
                    self.key_overrides.insert(id, change);
                }
                kreate_ui::KeymapEdit::Reset(id) => {
                    self.key_overrides.remove(&id);
                }
                kreate_ui::KeymapEdit::ResetAll => self.key_overrides.clear(),
            }
        }
        self.rebuild_keymap();
        self.save_keymap();
        self.redraw.set(RedrawReason::UI);
    }

    fn save_keymap(&self) {
        let Some(path) = kreate::prefs::keymap_path() else {
            return;
        };
        let file = kreate_input::KeyFile {
            preset: self.preset,
            overrides: self.key_overrides.clone(),
        };
        match kreate::prefs::save_keymap(&path, &file) {
            Ok(()) => println!("keymap: saved to {}", path.display()),
            // Reported, never fatal: a read-only home directory is not a reason
            // to stop working.
            Err(e) => println!("keymap: could not save to {}: {e}", path.display()),
        }
    }

    /// Enters or leaves edit mode on the active object.
    ///
    /// Entering selects every element, because a mesh with nothing selected
    /// makes every tool a no-op and the user's first keystroke appear to do
    /// nothing at all. Leaving clears it, so the object-mode outline is not
    /// competing with a stale element highlight.
    fn toggle_edit_mode(&mut self) {
        let entering = self.op_ctx.mode == OpMode::Object;
        if entering && kreate_ops::ops::mesh::target_mesh(&self.doc).is_none() {
            return;
        }
        self.op_ctx.mode = if entering {
            OpMode::EditMesh
        } else {
            OpMode::Object
        };
        self.run_op_with(
            "mesh.select",
            &[("action", PropertyValue::Enum(if entering { 0 } else { 1 }))],
        );
        self.ui.edit_mode = self.op_ctx.mode == OpMode::EditMesh;
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    fn set_select_mode(&mut self, mode: u32) {
        self.select_mode = mode;
        self.ui.select_mode = mode;
        self.redraw.set(RedrawReason::UI);
    }

    /// Applies the commands the panels queued this frame.
    fn drain_ui_commands(&mut self) {
        for command in self.ui.queue.drain() {
            match command {
                Command::RunOperator { op_id, params } => self.run_op(&op_id, params),
                Command::ToggleEditMode => self.toggle_edit_mode(),
                Command::SetSelectMode(mode) => self.set_select_mode(mode),
                Command::SnapToAxis(axis) => {
                    let destination = self.camera.looking_along(axis);
                    self.motion.animate_to(
                        self.camera,
                        destination,
                        kreate_render::DEFAULT_TRANSITION,
                    );
                    self.redraw.set(RedrawReason::CAMERA_MOVED);
                }
                Command::OrbitBy { dx, dy } => {
                    // The same scale the viewport drag uses, so dragging the
                    // gizmo and dragging the scene turn the view at one rate.
                    let height = self
                        .window
                        .as_ref()
                        .map(|w| w.inner_size().height)
                        .unwrap_or(1)
                        .max(1) as f64;
                    let scale = std::f64::consts::TAU / (2.0 * height);
                    self.motion.interrupt(&self.camera);
                    self.motion
                        .target
                        .orbit(-dx as f64 * scale, -dy as f64 * scale);
                    self.redraw.set(RedrawReason::CAMERA_MOVED);
                }
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
        // Framing needs the aspect ratio, or a viewport narrowed by side panels
        // clips the subject horizontally.
        let aspect = self
            .gfx
            .as_ref()
            .map(|g| g.viewport.aspect())
            .unwrap_or(16.0 / 9.0);
        let mut destination = self.camera;
        destination.frame_sphere(
            DVec3::new(c.x as f64, c.y as f64, c.z as f64),
            bounds.bounding_radius().max(1e-3) as f64,
            aspect,
        );
        // Animated, never teleported. A camera that jumps costs the user their
        // sense of where they are, and they then have to orbit around to
        // rebuild it — which is a worse outcome than a quarter-second wait.
        self.motion
            .animate_to(self.camera, destination, kreate_render::DEFAULT_TRANSITION);
        self.redraw.set(RedrawReason::CAMERA_MOVED);
    }

    /// Runs every object's modifier stack, and hands the result to the renderer
    /// and the picker.
    ///
    /// Cheap on an unchanged scene: an object with no modifiers evaluates to its
    /// own datablock handle, and one with a stack hits the cache on the identity
    /// of that handle. It is called more than once a frame on purpose — the
    /// auto-depth pivot needs the evaluated scene before the frame is drawn —
    /// and the second call is hits all the way down.
    fn evaluate_scene(&mut self) {
        self.evaluated = self.modifiers.evaluate_scene(&self.doc);
        // Both caches are keyed on handle identity, so an edit leaves the old
        // entry behind rather than replacing it — they have to be told when a
        // handle has left the scene or they grow with every edit. The modifier
        // cache is swept at the end of the frame, once the renderer has finished
        // with what it handed out; the pick cache holds nothing anyone else
        // borrows, so it can be swept here.
        //
        // Linear in entries × objects. That is nothing at this scale and will
        // want the same generation stamp the modifier cache uses once scenes are
        // large; it belongs with the rest of the large-scene work rather than as
        // a guess now.
        self.pick.retain_live(&self.evaluated);
    }

    /// Brings GPU residency into step with the evaluated scene.
    ///
    /// Only what actually changed is re-uploaded. The decision lives in
    /// [`gpu_sync::plan_sync`], where it can be tested without a device — "did it
    /// upload more than it had to" is invisible at a glance and only shows up as
    /// a stutter on a scene big enough that it is already too late.
    fn sync_gpu(&mut self) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        let held: Vec<Arc<MeshData>> = gfx.meshes.iter().map(|(arc, _)| arc.clone()).collect();
        let plan = gpu_sync::plan_sync(&held, &self.evaluated);
        if !plan.needs_work(held.len()) {
            return;
        }

        const PALETTE: [[f32; 3]; 4] = [
            [0.72, 0.73, 0.76],
            [0.78, 0.62, 0.42],
            [0.52, 0.68, 0.78],
            [0.68, 0.58, 0.74],
        ];
        // Kept entries are moved across rather than rebuilt, so a GPU mesh — and
        // the palette colour it was uploaded with — survives its neighbours
        // coming and going. Anything the plan does not keep is dropped by
        // falling out of `old` here.
        let mut old: Vec<Option<(Arc<MeshData>, GpuMesh)>> =
            gfx.meshes.drain(..).map(Some).collect();
        let mut next: Vec<(Arc<MeshData>, GpuMesh)> = Vec::with_capacity(plan.total());
        for slot in &plan.keep {
            if let Some(entry) = old.get_mut(*slot).and_then(Option::take) {
                next.push(entry);
            }
        }
        for arc in &plan.uploads {
            let cpu = RenderMeshCpu::extract(arc);
            let c = PALETTE[next.len() % PALETTE.len()];
            let uploaded = GpuMesh::upload(
                &gfx.gpu.device,
                &gfx.viewport.layouts,
                &cpu,
                Vec4::new(c[0], c[1], c[2], 1.0),
            );
            next.push((arc.clone(), uploaded));
        }
        gfx.meshes = next;
    }

    /// The cursor's position in the viewport, in the viewport's own pixels.
    ///
    /// `None` when the cursor is outside it, which is what stops a click on a
    /// panel from being treated as a click into the scene.
    ///
    /// The offscreen target's own coordinates, not the window's: the id buffer is
    /// that texture's size, so a pick is a coordinate in one read out of the
    /// other and an offset between the two would show up as picking that is
    /// subtly wrong near the edges.
    fn cursor_viewport_px(&self) -> Option<(i32, i32)> {
        let (x, y) = self.input.cursor?;
        let (rx, ry, w, h) = self.viewport_px?;
        let (u, v) = ((x as f32 - rx) / w, (y as f32 - ry) / h);
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        let (rw, rh) = self.gfx.as_ref()?.viewport.render_size();
        Some((
            (u * rw as f32) as i32,
            // The texture has +y down, like the window and unlike NDC.
            (v * rh as f32) as i32,
        ))
    }

    /// Which element domain the id pass should draw.
    fn pick_domain(&self) -> kreate_render::PickDomain {
        use kreate_render::PickDomain;
        if self.op_ctx.mode != OpMode::EditMesh {
            return PickDomain::Object;
        }
        match self.select_mode {
            0 => PickDomain::Vert,
            1 => PickDomain::Edge,
            _ => PickDomain::Face,
        }
    }

    /// Collects whatever the readback ring has finished with.
    ///
    /// Never blocks. Waiting here is exactly the stall the ring exists to avoid,
    /// and there is always the previous frame's region to answer from — at most
    /// one or two frames old, which is imperceptible for a hover and free for a
    /// click.
    fn read_picks(&mut self) {
        // The cursor position is read before the mutable borrow: it needs the
        // viewport's render size, which lives inside `gfx`.
        let cursor = self.cursor_viewport_px();
        let Some(gfx) = self.gfx.as_mut() else { return };
        gfx.pick_target.poll(&gfx.gpu.device);
        gfx.region.poll(&gfx.gpu.device);
        let previous = self.hover;
        self.hover = match (gfx.pick_target.latest(), cursor) {
            (Some(region), Some(cursor)) => region.nearest(cursor),
            _ => None,
        };
        // Only a *change* is worth a frame. Redrawing every frame because
        // something is under the cursor is how an idle window ends up burning a
        // laptop battery.
        if self.hover != previous {
            self.redraw.set(RedrawReason::HOVER_CHANGED);
        }
    }

    /// What a rubber band in progress should collect.
    ///
    /// `None` when nothing is being dragged, which is the overwhelmingly common
    /// case and costs nothing.
    fn region_request(&self) -> Option<(kreate_render::SelectShape, kreate_render::Collect)> {
        let drag = self.drag_select.as_ref()?;
        if !drag.moved {
            return None;
        }
        if self.op_ctx.mode != OpMode::EditMesh {
            // One past the largest key in the scene. Keys are arena indices, so
            // they outrun the object *count* as soon as anything has been
            // deleted — sizing the bitset by the count would drop the objects
            // living in the slots above it.
            let count = self
                .doc
                .objects()
                .map(|(id, _)| shell::object_key(id) + 1)
                .max()
                .unwrap_or(0);
            return Some((drag.shape(), kreate_render::Collect::Objects { count }));
        }

        // The object being edited, and how many elements its *cage* has in the
        // domain being selected. The cage rather than the evaluated mesh,
        // because that is what edit mode draws and therefore what the ids index.
        let id = self.doc.selection.active?;
        let mesh = self.doc.mesh(self.doc.object(id)?.mesh()?)?;
        let count = match self.pick_domain() {
            kreate_render::PickDomain::Vert => mesh.num_verts(),
            kreate_render::PickDomain::Edge => mesh.num_edges(),
            _ => mesh.num_faces(),
        };
        Some((
            drag.shape(),
            kreate_render::Collect::Elements {
                object: shell::object_key(id),
                count,
            },
        ))
    }

    /// Begins a selection gesture at the cursor.
    fn begin_drag_select(&mut self) {
        let Some(origin) = self.cursor_viewport_px() else {
            self.drag_select = None;
            return;
        };
        let origin = (origin.0 as f32, origin.1 as f32);
        self.drag_select = Some(DragSelect {
            origin,
            current: origin,
            moved: false,
            extend: self.input.shift_down,
            deselect: self.input.ctrl_down,
        });
    }

    /// Follows the pointer during a selection gesture.
    ///
    /// Called from the cursor event rather than per frame, so the rectangle
    /// tracks the pointer even on a frame the viewport does not redraw.
    fn track_drag_select(&mut self) {
        let Some(at) = self.cursor_viewport_px() else {
            return;
        };
        let at = (at.0 as f32, at.1 as f32);
        let Some(drag) = self.drag_select.as_mut() else {
            return;
        };
        drag.current = at;
        let moved = (at.0 - drag.origin.0).abs() > DRAG_THRESHOLD_PX
            || (at.1 - drag.origin.1).abs() > DRAG_THRESHOLD_PX;
        if moved && !drag.moved {
            drag.moved = true;
        }
        if drag.moved {
            // The rubber band is part of the interface, and the reduction that
            // resolves it runs in the frame — so both want one.
            self.redraw.set(RedrawReason::HOVER_CHANGED);
        }
    }

    /// Ends a selection gesture: a click if the pointer stayed put, a box if it
    /// moved.
    ///
    /// Either way this costs no waiting. A click consumes the id region already
    /// read back, and a box consumes the reduction the drag has been running
    /// every frame — so the answer for the final rectangle is already in hand.
    fn finish_drag_select(&mut self) {
        let Some(drag) = self.drag_select.take() else {
            return;
        };
        if !drag.moved {
            self.click_select(drag.extend);
            return;
        }
        let Some(resolved) = self
            .gfx
            .as_ref()
            .and_then(|g| g.region.resolved())
            .map(|e| e.to_vec())
        else {
            return;
        };

        if self.op_ctx.mode != OpMode::EditMesh {
            // The reduction collected object ids, which are the same keys the
            // outliner round-trips — so this goes through the same command, and
            // the same resolution against the *current* document.
            let mut objects = resolved;
            if drag.extend || drag.deselect {
                let existing: Vec<u32> = self
                    .doc
                    .selection
                    .objects
                    .iter()
                    .map(|id| shell::object_key(*id))
                    .collect();
                if drag.deselect {
                    objects = existing
                        .into_iter()
                        .filter(|k| !objects.contains(k))
                        .collect();
                } else {
                    for key in existing {
                        if !objects.contains(&key) {
                            objects.push(key);
                        }
                    }
                }
            }
            let active = objects.last().copied();
            self.ui
                .queue
                .push(kreate_ui::Command::SetSelection { objects, active });
            return;
        }

        let elements = resolved;
        let mode = match self.pick_domain() {
            kreate_render::PickDomain::Vert => 0u32,
            kreate_render::PickDomain::Edge => 1,
            _ => 2,
        };
        let mut params = PropertyMap::new();
        params.set("elements", PropertyValue::Ints(elements));
        params.set("mode", PropertyValue::Enum(mode));
        params.set("extend", PropertyValue::Bool(drag.extend || drag.deselect));
        params.set("deselect", PropertyValue::Bool(drag.deselect));
        self.run_op("mesh.select_set", params);
    }

    /// Turns a click in the viewport into a selection.
    ///
    /// Consumes the region already read back rather than starting one, so this
    /// costs no latency at all: the selection changes on the same frame as the
    /// click. Returns whether the click was handled, so a miss can fall through
    /// to whatever else wants it.
    fn click_select(&mut self, extend: bool) -> bool {
        if self.cursor_viewport_px().is_none() {
            return false;
        }
        let hit = self.hover;
        match self.op_ctx.mode {
            OpMode::EditMesh => {
                let mode = match self.pick_domain() {
                    kreate_render::PickDomain::Vert => 0u32,
                    kreate_render::PickDomain::Edge => 1,
                    _ => 2,
                };
                match hit {
                    Some(hit) => {
                        let mut params = PropertyMap::new();
                        params.set("element", PropertyValue::Int(hit.element as i64));
                        params.set("mode", PropertyValue::Enum(mode));
                        params.set("extend", PropertyValue::Bool(extend));
                        self.run_op("mesh.select_pick", params);
                    }
                    // Clicking empty space clears the selection — but not when
                    // extending, where the user is adding to it and a miss
                    // should leave what they have alone.
                    None if !extend => {
                        self.run_op("mesh.select_nothing", PropertyMap::new());
                    }
                    None => {}
                }
            }
            _ => {
                let objects = match hit {
                    Some(hit) => vec![hit.object],
                    None if extend => return true,
                    None => Vec::new(),
                };
                let active = objects.last().copied();
                // Queued rather than applied here, so it goes through the same
                // key resolution the outliner does: a key from a frame where an
                // object still existed must not act on whatever reused its slot.
                self.ui
                    .queue
                    .push(kreate_ui::Command::SetSelection { objects, active });
            }
        }
        true
    }

    /// The cursor's position in the viewport, as normalised device coordinates.
    ///
    /// `None` when the cursor is outside the viewport, which is what stops a
    /// click on a panel from being treated as a click into the scene.
    fn cursor_ndc(&self) -> Option<glam::Vec2> {
        let (x, y) = self.input.cursor?;
        let (rx, ry, w, h) = self.viewport_px?;
        let (u, v) = ((x as f32 - rx) / w, (y as f32 - ry) / h);
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        // NDC has +y up; a window's cursor has +y down.
        Some(glam::Vec2::new(u * 2.0 - 1.0, 1.0 - v * 2.0))
    }

    fn viewport_aspect(&self) -> f64 {
        self.viewport_px
            .map(|(_, _, w, h)| (w / h.max(1.0)) as f64)
            .unwrap_or(1.0)
    }

    /// Re-centres the orbit on whatever is under the cursor.
    ///
    /// Called once when a drag begins rather than every frame: moving the pivot
    /// mid-orbit would make the model swim under the pointer. A miss leaves the
    /// pivot alone, so orbiting over empty space keeps turning about whatever it
    /// was turning about before — which is what the user expects, and is why this
    /// does not fall back to the origin.
    fn set_auto_depth_pivot(&mut self) {
        if !self.auto_depth {
            return;
        }
        let Some(ndc) = self.cursor_ndc() else { return };
        let aspect = self.viewport_aspect();
        // Evaluated here rather than relying on the last frame's list: a drag can
        // begin on the same frame an object appears, and orbiting about a pivot
        // that ignores it is the sort of thing that reads as the tool being
        // vague about where things are.
        self.evaluate_scene();
        let Some(hit) =
            self.pick
                .raycast_ndc(&self.doc, &self.evaluated, &self.camera, ndc, aspect)
        else {
            return;
        };
        if let Some(depth) = self.camera.depth_of(hit.point) {
            // Both, so the easing does not immediately drag the pivot back to
            // where it was. Auto-depth is not a motion to be smoothed — it is a
            // change of what the motion is *about*.
            self.camera.set_pivot_depth(depth);
            self.motion.target.set_pivot_depth(depth);
        }
    }

    fn apply_input(&mut self) {
        let height = self
            .window
            .as_ref()
            .map(|w| w.inner_size().height)
            .unwrap_or(1)
            .max(1) as f64;

        // Navigation writes to the *target*; the drawn camera eases toward it.
        // Taking hold of the view also cancels any framing move in flight — a
        // camera that keeps drifting after the user grabs it is one fighting
        // them.
        if self.input.has_pending() {
            self.motion.interrupt(&self.camera);
        }
        if self.input.orbit_delta != (0.0, 0.0) {
            let (dx, dy) = self.input.orbit_delta;
            let scale = std::f64::consts::TAU / (2.0 * height);
            self.motion.target.orbit(-dx * scale, -dy * scale);
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
        if self.input.pan_delta != (0.0, 0.0) {
            let (dx, dy) = self.input.pan_delta;
            self.motion.target.pan(dx / height, dy / height);
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
        if self.input.scroll != 0.0 {
            // Zoom toward the cursor when it is over the scene, so a user closes
            // in on a detail by pointing at it instead of alternating zoom and
            // pan. Falls back to a plain dolly over the panels.
            match self.cursor_ndc() {
                Some(ndc) => {
                    let aspect = self.viewport_aspect();
                    let direction = self.camera.ray_direction(ndc, aspect).as_dvec3();
                    self.motion
                        .target
                        .dolly_towards(self.input.scroll, direction);
                }
                None => self.motion.target.dolly(self.input.scroll),
            }
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
        self.input.orbit_delta = (0.0, 0.0);
        self.input.pan_delta = (0.0, 0.0);
        self.input.scroll = 0.0;
    }

    /// Eases the drawn camera toward its target.
    ///
    /// Returns whether anything is still in flight, which is what keeps the
    /// window redrawing — and, just as importantly, what lets it stop.
    fn advance_camera(&mut self) -> bool {
        let now = Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map(|t| now.duration_since(t).as_secs_f64())
            // A long gap — the window was hidden, or a dialogue was open —
            // must not teleport the camera through its whole easing in one
            // step, so the step is capped at a slow frame rather than trusted.
            .map_or(1.0 / 60.0, |dt| dt.clamp(1e-4, 0.1));
        let moving = self.motion.advance(&mut self.camera, dt);
        if moving {
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
        moving
    }

    fn render(&mut self) {
        // Before anything reads the camera: the eased position is what this
        // frame is drawn from, and it has to be settled before the viewport
        // projection, the ray queries and the interface all ask for it.
        self.advance_camera();
        self.read_picks();
        self.evaluate_scene();
        self.sync_gpu();
        let Some(window) = self.window.clone() else {
            return;
        };
        let Some(state) = self.egui_state.as_mut() else {
            return;
        };

        // --- resize first, using what the previous frame asked for ----------
        // Must happen before the interface runs: reallocating the target frees
        // the egui texture id, and doing that after the paint jobs already
        // reference it hands the renderer a dangling binding.
        if let Some(gfx) = self.gfx.as_mut() {
            let (w, h) = self.wanted_viewport;
            if w > 0
                && h > 0
                && gfx
                    .target
                    .resize(&gfx.gpu.device, &mut gfx.egui_renderer, w, h)
            {
                gfx.viewport
                    .resize(&gfx.gpu.device, gfx.target.size.0, gfx.target.size.1);
                // The id buffer has to match the colour target pixel for pixel:
                // a pick is a coordinate in one read out of the other.
                gfx.pick_target
                    .resize(&gfx.gpu.device, gfx.target.size.0, gfx.target.size.1);
                self.hover = None;
            }
            if w > 0 && h > 0 {
                // The allocation is coarse; the drawn region is exact. Keeping
                // these apart is what stops the image being stretched into the
                // panel gap.
                gfx.viewport.set_render_size(w, h);
            }
        }

        // What the interface has to draw over the viewport. Published here
        // rather than read from the application inside the closure, because the
        // interface may not reach into it — see `kreate-ui`'s insulation rule.
        //
        // Viewport pixels back into points: the drag is tracked against the id
        // buffer, which is the offscreen texture's size, and egui lays out in
        // logical points.
        self.ui.select_band = self.drag_select.as_ref().filter(|d| d.moved).map(|d| {
            let scale = self.egui_ctx.pixels_per_point();
            let to_points = |p: (f32, f32)| egui::pos2(p.0 / scale, p.1 / scale);
            (to_points(d.origin), to_points(d.current))
        });
        self.ui.select_band_deselect = self.drag_select.as_ref().is_some_and(|d| d.deselect);

        // --- run the interface, which tells us how big the viewport is -------
        let raw_input = state.take_egui_input(&window);
        let image = self.gfx.as_ref().map(|g| shell::ViewportImage {
            texture: g.target.id,
            uv: g.viewport.uv_extent(),
        });
        let mut viewport_rect = egui::Rect::ZERO;
        let output = {
            let ui_state = &mut self.ui;
            let doc = &self.doc;
            let registry = &self.registry;
            let history = &self.history;
            let keymap = &self.keymap;
            let preset = self.preset;
            let overrides = &self.key_overrides;
            self.egui_ctx.run_ui(raw_input, |ui| {
                viewport_rect = shell::draw(
                    ui, ui_state, doc, registry, history, keymap, preset, overrides, image,
                );
            })
        };
        state.handle_platform_output(&window, output.platform_output.clone());

        // The single input gate. Scattered `if !typing` checks are how a rename
        // field ends up firing tool shortcuts again.
        self.ui.egui_wants_keyboard = self.egui_ctx.egui_wants_keyboard_input();
        self.ui.ime_active = output.platform_output.ime.is_some();

        self.apply_keymap_edits();
        self.drain_ui_commands();
        // The panels may have changed the scene — a modifier added, a parameter
        // dragged — so it is re-evaluated before the frame is encoded rather
        // than showing the change one frame late.
        self.evaluate_scene();
        self.sync_gpu();

        let pixels_per_point = self.egui_ctx.pixels_per_point();

        // What the panels left this frame, applied at the start of the next one.
        let wanted = (
            (viewport_rect.width().max(0.0) * pixels_per_point) as u32,
            (viewport_rect.height().max(0.0) * pixels_per_point) as u32,
        );
        if wanted != self.wanted_viewport && wanted.0 > 0 && wanted.1 > 0 {
            self.wanted_viewport = wanted;
            self.redraw.set(RedrawReason::RESIZE);
        }
        // Kept in physical pixels because that is what the cursor arrives in.
        // Converting the cursor to logical units instead would go wrong on a
        // fractional scale factor, and the error is a ray that misses by a few
        // pixels — which reads as picking being unreliable rather than as a unit
        // mistake.
        self.viewport_px = (viewport_rect.width() > 0.0).then(|| {
            (
                viewport_rect.min.x * pixels_per_point,
                viewport_rect.min.y * pixels_per_point,
                viewport_rect.width() * pixels_per_point,
                viewport_rect.height() * pixels_per_point,
            )
        });

        // What the id pass will be asked for, settled before `gfx` is borrowed
        // mutably for the rest of the frame: both parts read `self`, and one of
        // them reads the viewport's own size.
        let pick_request = self
            .cursor_viewport_px()
            .map(|cursor| (self.pick_domain(), cursor));
        let region_request = self.region_request();
        // Settled before `gfx` is borrowed for the frame, like the other two.
        // Only an element drag previews through the overlays: the bitset an
        // object drag fills is indexed by object, and the overlays highlight
        // elements. An object drag shows its rubber band and the objects it
        // covers highlight on release.
        let preview_state = region_request.as_ref().and_then(|(_, collect)| {
            let kreate_render::Collect::Elements { count, .. } = collect else {
                return None;
            };
            Some(kreate_render::PreviewSelection {
                domain: self.pick_domain(),
                deselect: self.drag_select.as_ref().is_some_and(|d| d.deselect),
                elements: *count,
            })
        });

        let Some(gfx) = self.gfx.as_mut() else { return };

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

        // One draw entry per *object*, sharing GPU geometry by the identity of the
        // evaluated mesh — so a thousand linked duplicates cost a thousand
        // transforms. Driven from the evaluated scene rather than from the
        // document, which is what stops the renderer from being able to draw
        // something the picker cannot hit.
        let preview_target = region_request.as_ref().and(self.doc.selection.active);
        let mut draws: Vec<Draw> = Vec::new();
        for (id, mesh) in &self.evaluated {
            if let Some((_, gpu_mesh)) = gfx.meshes.iter().find(|(m, _)| Arc::ptr_eq(m, mesh)) {
                // In edit mode the object-level highlight stands down: the
                // whole wireframe going orange would drown out the element
                // selection, which is the only thing the user is looking at.
                draws.push(
                    Draw::new(gpu_mesh, self.doc.world_transform(*id))
                        .selected(
                            self.op_ctx.mode == OpMode::Object
                                && self.doc.selection.contains(*id),
                        )
                        // The same key the outliner round-trips, so a pick and a
                        // row in the tree name the object the same way.
                        .object_id(shell::object_key(*id))
                        // A drag resolves elements of the object being edited,
                        // and the bitset is indexed by that mesh's numbering —
                        // so only that draw may read it.
                        .preview(Some(*id) == preview_target),
                );
            }
        }

        let mut encoder = gfx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // Element overlays and the pick tolerance both belong to edit mode. A
        // scene full of vertex dots is unreadable when the user is arranging
        // objects, and picking an element they cannot see is worse.
        gfx.viewport.show_elements = self.op_ctx.mode == OpMode::EditMesh;
        // A tolerance chosen in pixels is a tolerance chosen for one screen
        // density; without this, picking on a high-density display is a third as
        // forgiving as it was designed to be.
        gfx.viewport.pick_scale = pixels_per_point;

        // What the overlays should show as already selected. The bitset comes
        // from the *previous* frame's reduction: the dispatch needs the id
        // buffer, which needs the depth this frame's pass has not written yet.
        // One frame of lag on a highlight that follows the pointer is not
        // perceptible, and it is the same trade the id readback ring makes.
        let preview = preview_state;
        gfx.viewport.prepare_preview(
            &gfx.gpu.device,
            &gfx.gpu.queue,
            gfx.region.bits(),
            gfx.region.generation(),
            preview,
        );

        // 3D into the offscreen target...
        gfx.viewport
            .prepare(&gfx.gpu.device, &gfx.gpu.queue, &self.camera, &draws);
        gfx.viewport.render(&mut encoder, &gfx.target.view, &draws);

        // The id pass runs every frame, not on the click. The naive flow — click,
        // render ids, map, poll, interpret — costs two to four frames *after* the
        // click, which reads as the application being sluggish however high the
        // frame rate is. Running it continuously means a click consumes data
        // already in flight and the selection changes on the same frame.
        if let Some((domain, cursor)) = pick_request {
            gfx.viewport.prepare_ids(
                &gfx.gpu.device,
                &gfx.gpu.queue,
                &self.camera,
                &draws,
                domain,
            );
            // After the visible pass in the same encoder: it tests against the
            // depth buffer that pass produced, which is what makes occlusion in
            // the id buffer exactly the occlusion on screen.
            gfx.viewport.render_ids(
                &mut encoder,
                gfx.pick_target.view(),
                &draws,
                domain,
                self.xray,
            );
            gfx.pick_target.capture(&mut encoder, cursor, domain);

            // A rubber band in progress resolves against the id buffer that was
            // just drawn, every frame. The whole point of the compute reduction
            // is that this is affordable: it scans only the rectangle, on the
            // GPU, and brings back one bit per element rather than sixteen
            // megabytes of ids.
            if let Some((shape, collect)) = region_request {
                gfx.region.dispatch(
                    &gfx.gpu.device,
                    &gfx.gpu.queue,
                    &mut encoder,
                    gfx.pick_target.view(),
                    gfx.viewport.render_size(),
                    &shape,
                    collect,
                );
            }
        }

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
        // Only after submit: mapping a buffer whose copy has not been submitted
        // waits for a command that will never arrive.
        gfx.pick_target.after_submit();
        gfx.region.after_submit();

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

        // Last, once the frame is submitted and nothing is still reading the
        // meshes this handed out. Without it the cache keeps an evaluated copy
        // for every parameter value the user has dragged through — on a
        // subdivision modifier, one per frame of the drag.
        self.modifiers.sweep();
    }

    /// The context an input event is resolved in.
    ///
    /// One value, computed once. Scattering `if mode == EditMesh` through the
    /// handler is the thing the keymap exists to remove.
    fn key_context(&self) -> KeyContext {
        match self.op_ctx.mode {
            OpMode::EditMesh => KeyContext::EditMesh,
            OpMode::Object => KeyContext::ObjectMode,
        }
    }

    fn modifiers(&self) -> KeyMods {
        KeyMods {
            shift: self.input.shift_down,
            ctrl: self.input.ctrl_down,
            alt: self.input.alt_down,
        }
    }

    fn handle_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        // Escape and F1 are deliberately outside the keymap: there must always
        // be a way out and a way to ask, even from a keymap the user has broken.
        match code {
            KeyCode::Escape => {
                event_loop.exit();
                return;
            }
            KeyCode::F1 => {
                println!("{HELP}");
                self.print_keymap();
                return;
            }
            _ => {}
        }
        // A binding is listening: this keystroke is the new trigger, not a
        // command. Checked before the keymap gate, because the gate is exactly
        // what capturing turns off.
        if let Some(id) = self.ui.capturing.clone() {
            if let Some(name) = key_name(code) {
                let change = kreate_ui::keymap_editor::capture(
                    self.key_overrides.get(&id),
                    name,
                    self.modifiers(),
                );
                self.ui
                    .keymap_edits
                    .push(kreate_ui::KeymapEdit::Set(id, change));
            }
            // Cleared either way, so an unmappable key ends the capture rather
            // than leaving the panel waiting for one that will never come.
            self.ui.capturing = None;
            self.redraw.set(RedrawReason::UI);
            return;
        }
        // Everything else stands down while a text field or an IME has the
        // keyboard: with a Korean IME active, `G` and `R` would otherwise reach
        // both the composer and the keymap.
        if !self.ui.keymap_enabled() {
            return;
        }
        let Some(name) = key_name(code) else { return };
        let Some(action) = self.keymap.resolve(
            self.key_context(),
            &KeyInput::key(name),
            self.modifiers(),
            EventKind::Press,
        ) else {
            return;
        };
        self.dispatch(&action);
    }

    /// Sends a resolved action to whichever half of the application owns it.
    ///
    /// `view.` and `app.` ids act on the viewport and the session; everything
    /// else is an operator and goes through the history. The split is what keeps
    /// a camera move out of the undo stack while still letting it be rebound
    /// from the same table as an extrude.
    fn dispatch(&mut self, action: &Action) {
        match action.op_id.as_str() {
            "edit.undo" => self.undo(),
            "edit.redo" => self.redo(),
            "edit.adjust_last" => {
                let factor = match action.params.get("factor") {
                    Some(PropertyValue::Float(f)) => *f,
                    _ => 1.0,
                };
                self.adjust_last(factor);
            }
            "app.print_script" => {
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
            "view.frame_all" => self.frame_all(),
            "view.toggle_edit_mode" => self.toggle_edit_mode(),
            "view.select_mode" => {
                if let Some(PropertyValue::Enum(mode)) = action.params.get("mode") {
                    self.set_select_mode(*mode);
                }
            }
            "view.toggle_auto_depth" => {
                self.auto_depth = !self.auto_depth;
                println!(
                    "auto-depth orbit {}",
                    if self.auto_depth { "on" } else { "off" }
                );
            }
            "view.toggle_grid" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.show_grid = !gfx.viewport.show_grid;
                    self.redraw.set(RedrawReason::SETTINGS);
                }
            }
            "view.toggle_wireframe" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.show_wireframe = !gfx.viewport.show_wireframe;
                    self.redraw.set(RedrawReason::SETTINGS);
                }
            }
            "view.cycle_shading" => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.shading = match gfx.viewport.shading {
                        ShadingMode::Studio => ShadingMode::Flat,
                        ShadingMode::Flat => ShadingMode::Studio,
                    };
                    self.redraw.set(RedrawReason::SETTINGS);
                }
            }
            "view.toggle_projection" => {
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
            // Navigation drags carry their motion in the input state rather than
            // in parameters, so they are handled where the deltas are.
            "view.orbit" | "view.pan" | "view.dolly" => {}
            "object.add_next_primitive" => {
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
            _ => {
                // An operator. The element mode is supplied where the binding
                // did not name one, so `X` deletes whatever the user is
                // selecting rather than always vertices.
                let mut params = action.params.clone();
                if action.op_id.starts_with("mesh.")
                    && params.get("mode").is_none()
                    && matches!(
                        action.op_id.as_str(),
                        "mesh.delete" | "mesh.dissolve" | "mesh.select"
                    )
                {
                    params.set("mode", PropertyValue::Enum(self.select_mode));
                }
                self.run_op(&action.op_id, params);
            }
        }
    }

    /// Which navigation action, if any, the currently held buttons mean.
    fn drag_action(&self, button: KeyMouse) -> Option<Action> {
        self.keymap.resolve(
            KeyContext::Viewport,
            &KeyInput::Mouse(button),
            self.modifiers(),
            EventKind::Drag,
        )
    }

    fn print_keymap(&self) {
        println!("--- keymap: {} ---", self.preset.name());
        for b in self.keymap.bindings() {
            if b.disabled {
                continue;
            }
            println!(
                "  {:<24} {:?} {:?} {:?} -> {}",
                b.id.0, b.context, b.input, b.modifiers, b.action.op_id
            );
        }
        println!("---");
    }
}

/// A winit key code as the name a keymap stores.
///
/// Named rather than passed through as an enum so a keymap file is readable and
/// stays valid if the window library renames or reorders anything. Unmapped keys
/// simply do not resolve, which is the right answer for a key nothing binds.
fn key_name(code: KeyCode) -> Option<&'static str> {
    Some(match code {
        KeyCode::KeyA => "A",
        KeyCode::KeyB => "B",
        KeyCode::KeyC => "C",
        KeyCode::KeyD => "D",
        KeyCode::KeyE => "E",
        KeyCode::KeyF => "F",
        KeyCode::KeyG => "G",
        KeyCode::KeyH => "H",
        KeyCode::KeyI => "I",
        KeyCode::KeyJ => "J",
        KeyCode::KeyK => "K",
        KeyCode::KeyL => "L",
        KeyCode::KeyM => "M",
        KeyCode::KeyN => "N",
        KeyCode::KeyO => "O",
        KeyCode::KeyP => "P",
        KeyCode::KeyQ => "Q",
        KeyCode::KeyR => "R",
        KeyCode::KeyS => "S",
        KeyCode::KeyT => "T",
        KeyCode::KeyU => "U",
        KeyCode::KeyV => "V",
        KeyCode::KeyW => "W",
        KeyCode::KeyX => "X",
        KeyCode::KeyY => "Y",
        KeyCode::KeyZ => "Z",
        KeyCode::Digit1 => "Digit1",
        KeyCode::Digit2 => "Digit2",
        KeyCode::Digit3 => "Digit3",
        KeyCode::Digit4 => "Digit4",
        KeyCode::Digit5 => "Digit5",
        KeyCode::Tab => "Tab",
        KeyCode::BracketLeft => "BracketLeft",
        KeyCode::BracketRight => "BracketRight",
        _ => return None,
    })
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

        let pick_target = kreate_render::PickTarget::new(&gpu.device, target.size.0, target.size.1);
        let region = kreate_render::RegionSelect::new(&gpu.device);
        self.gfx = Some(Gfx {
            surface,
            config,
            gpu,
            viewport,
            meshes: Vec::new(),
            pick_target,
            region,
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
                let pressed = state == ElementState::Pressed;
                let slot = match button {
                    MouseButton::Left => Some(0),
                    MouseButton::Middle => Some(1),
                    MouseButton::Right => Some(2),
                    _ => None,
                };
                if let Some(slot) = slot {
                    let mapped = [KeyMouse::Left, KeyMouse::Middle, KeyMouse::Right][slot];
                    // On the press, not on the first motion: by the time the
                    // pointer has moved the view has already turned once about
                    // the old pivot, and that first frame is exactly the jump
                    // auto-depth exists to remove.
                    let starts_orbit = self
                        .drag_action(mapped)
                        .is_some_and(|a| a.op_id == "view.orbit");
                    if pressed && !self.buttons[slot] && starts_orbit {
                        self.set_auto_depth_pivot();
                    }
                    // A left press that is not bound to a drag begins a
                    // selection gesture — which is a click or a box depending on
                    // whether the pointer moves, and cannot be told apart until
                    // it does. Nothing is committed here.
                    //
                    // Shift extends and Ctrl removes, read from the live modifier
                    // state rather than from the keymap: this is one gesture
                    // rather than several bindings, and a preset that rebinds
                    // selection has to move its variants with it.
                    let selecting = mapped == KeyMouse::Left
                        && !starts_orbit
                        && self.drag_action(mapped).is_none();
                    if selecting {
                        if pressed && !self.buttons[slot] {
                            self.begin_drag_select();
                        } else if !pressed && self.buttons[slot] {
                            self.finish_drag_select();
                        }
                    }
                    self.buttons[slot] = pressed;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.track_drag_select();
                let p = (position.x, position.y);
                if let Some(prev) = self.input.cursor {
                    let delta = (p.0 - prev.0, p.1 - prev.1);
                    // Which navigation this is comes from the keymap, so the
                    // Maya preset's Alt+LMB orbit needs no code here at all.
                    // One drag must not both turn the gizmo and turn the scene;
                    // the gizmo took the pointer first.
                    let blocked = self.ui.gizmo_active;
                    for (slot, mapped) in [KeyMouse::Left, KeyMouse::Middle, KeyMouse::Right]
                        .into_iter()
                        .enumerate()
                    {
                        if blocked || !self.buttons[slot] {
                            continue;
                        }
                        let Some(action) = self.drag_action(mapped) else {
                            continue;
                        };
                        self.input.note_event();
                        match action.op_id.as_str() {
                            "view.orbit" => {
                                self.input.orbit_delta.0 += delta.0;
                                self.input.orbit_delta.1 += delta.1;
                            }
                            "view.pan" => {
                                self.input.pan_delta.0 += delta.0;
                                self.input.pan_delta.1 += delta.1;
                            }
                            "view.dolly" => {
                                // Vertical drag zooms, at a gentler rate than
                                // the wheel's notches.
                                self.input.scroll -= delta.1 * 0.02;
                            }
                            _ => {}
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
                self.input.alt_down = m.state().alt_key();
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
        if self.input.has_pending()
            || self.redraw.any()
            || self.motion.is_animating()
            || !self.ui.queue.is_empty()
        {
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
